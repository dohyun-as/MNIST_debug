#!/usr/bin/env python
"""
visualize_masked_diff_generation.py
===================================
Visualizations for the CLEVR ``masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0``
run (4 level multi-res: 1×1, 2×2, 4×4, 8×8 continuous tokens + diffusion head).

Produces 4 artifact groups per split (easy/medium/hard) × 5 samples per split:

  1. generation_order/      — per-token unmask-step heatmaps for the 3 samplers
                              (ddpm_cache, confidence_top1, confidence_cosine),
                              split into the 4 level maps.
  2. level_ablation/        — fix generated tokens, decode the pretrained DiT
                              using ONLY tokens up to level L (1×1, 2×2, 4×4,
                              8×8) with 3 different decoder noise seeds.
  3. final_grid_overlay/    — final 8×8 full-resolution images with a visible
                              8×8 grid overlay so we can eyeball what that
                              token scale covers.
  4. meta.json              — condition text + sample index per split.

Usage (4-GPU distributed, round-robin sharded):
    GPUS=0,1,2,3 bash script/visualize_masked_diff_generation.sh

Or directly:
    accelerate launch --multi_gpu --num_processes 4 \
        src/visualize_masked_diff_generation.py \
        --run_dir ./runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0 \
        --step 50000 --variant ema --samples_per_split 5 --num_decode_seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

# Reuse training-script helpers ------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_discrete_diffusion_v2 import (  # noqa: E402
    CachedContinuousTokenDataset,
    CLEVRImageDataset,
    DIT,
    DiscreteDiffusion,
    PretrainedTextConditionEncoder,
    _cont_tokens_to_level_features,
    _extract_raw_text,
    _select_eval_indices,
    cache_all_continuous_tokens,
    load_pretrained_model,
)
from diffloss import DiffLoss  # noqa: E402
from eval_clevr_condition import (  # noqa: E402
    clevr_text_to_condition_json,
    eval_clevr_conditions,
    load_eval_models,
)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--step", type=int, required=True,
                   help="Checkpoint step to load (e.g. 50000)")
    p.add_argument("--variant", choices=["ema", "base"], default="ema")
    p.add_argument("--output_subdir", default="visualize",
                   help="Subdir under run_dir to save visualizations")

    # Splits / sampling
    p.add_argument("--splits", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--samples_per_split", type=int, default=5,
                   help="Total eval samples per split — ALL of these get "
                        "token generation + full-level decode + eval score. "
                        "Set to 30 to match training's eval set.")
    p.add_argument("--viz_samples_per_split", type=int, default=None,
                   help="How many of the per-split eval samples ALSO get "
                        "saved visualizations (generation_order PNG, "
                        "history.pt, decoded PNG, 8x8 overlay, "
                        "level_ablation figure, sampler_compare figure). "
                        "Defaults to samples_per_split (visualize all). "
                        "Set small (e.g. 2) for a big eval set where you "
                        "only want to eyeball a few.")
    p.add_argument("--num_decode_seeds", type=int, default=3,
                   help="Per level ablation: how many decoder noise seeds")
    p.add_argument("--samplers", nargs="+",
                   default=["ddpm_cache", "confidence_top1", "confidence_cosine"])

    # Dataset paths (same defaults as the training .sh)
    p.add_argument("--val_image_root", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/images")
    p.add_argument("--val_cond_dir", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text")

    # Eval / decode params (match train script defaults)
    p.add_argument("--eval_num_steps", type=int, default=128)
    p.add_argument("--decode_num_steps", type=int, default=50)
    # Batching — HUGE speed impact. Viz subset still runs batch=1 so we can
    # save per-sample generation_order history; non-viz samples and all decode
    # calls are batched.
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="Batch size for non-viz sample generation. "
                        "Viz subset always uses batch=1 to capture history.")
    p.add_argument("--decode_batch_size", type=int, default=8,
                   help="Batch size for the flow-matching decoder.")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--mixed_precision", default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    # Let user override decoder seed for level-ablation if desired
    p.add_argument("--decode_ablation_sampler", default="ddpm_cache",
                   choices=["ddpm_cache", "confidence_top1", "confidence_cosine"],
                   help="Which sampler's tokens to use for the level ablation.")

    # CFG sweep (values + schedules). Each (sampler × cfg × schedule) gets
    # its own generation + decoded image + per-cell eval. Level-ablation and
    # 8x8 overlay use only the designated ablation (cfg, schedule) to avoid
    # combinatorial blow-up. Defaults to the single (cfg, schedule) recorded
    # in run_config.json — i.e. identical to the old behavior.
    p.add_argument("--cfg_values", type=float, nargs="+", default=None,
                   help="CFG scales to sweep. Defaults to run_cfg.diff_head_cfg.")
    p.add_argument("--cfg_schedules", nargs="+", default=None,
                   choices=["linear", "constant"],
                   help="CFG schedules to sweep. Defaults to run_cfg.cfg_schedule.")
    p.add_argument("--decode_ablation_cfg", type=float, default=None,
                   help="CFG used for the level-ablation decode. "
                        "Defaults to the first element of --cfg_values.")
    p.add_argument("--decode_ablation_schedule", default=None,
                   choices=["linear", "constant"],
                   help="CFG schedule used for the level-ablation decode. "
                        "Defaults to the first element of --cfg_schedules.")

    # Eval (detector + classifier)
    p.add_argument("--run_eval", action="store_true", default=True,
                   help="Run detector+classifier eval on decoded images.")
    p.add_argument("--no_run_eval", dest="run_eval", action="store_false")
    p.add_argument("--eval_det_threshold", type=float, default=0.3)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _compute_unmask_step(mask_history):
    """Given a list of (B, L) bool tensors (True = still masked), return the
    step index at which each (b, l) position was first unmasked.

    Returns: (B, L) long tensor, values in [0, len(mask_history)-1].
    A position that was known from the start → step 0. Positions never
    unmasked (shouldn't happen) → len(mask_history).
    """
    steps = torch.stack(mask_history, dim=0).long()  # (T, B, L), 1 = masked
    T, B, L = steps.shape
    unmask_step = torch.full((B, L), T, dtype=torch.long)
    for t in range(T):
        not_yet = (unmask_step == T)
        newly_un = (steps[t] == 0) & not_yet
        unmask_step[newly_un] = t
    return unmask_step  # (B, L), on cpu


def _split_by_level(flat, level_sizes):
    """Split a flat (L,) tensor into per-level (S,S) tensors.

    level_sizes is finest-first, e.g. [8, 4, 2, 1]. Sequence order: finest
    at the front, coarsest at the back.
    """
    out = {}
    offset = 0
    for s in sorted(level_sizes, reverse=True):  # matches the seq packing
        n = s * s
        out[s] = flat[offset:offset + n].reshape(s, s)
        offset += n
    return out


def _render_order_figure(unmask_step_b, cond_text, level_sizes, title,
                          max_step):
    """Render one sample's generation order as a 1×(num_levels) panel.

    Each panel shows the step-index heatmap laid out in the level's native
    spatial grid (1×1, 2×2, 4×4, 8×8).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ordered_levels = sorted(level_sizes)   # coarsest first: 1, 2, 4, 8
    n = len(ordered_levels)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 1.0, 2.8),
                              gridspec_kw={'width_ratios': [1] * n})
    if n == 1:
        axes = [axes]

    per_level = _split_by_level(unmask_step_b, level_sizes)
    norm = Normalize(vmin=0, vmax=max(max_step, 1))

    for ax, s in zip(axes, ordered_levels):
        arr = per_level[s].numpy()
        im = ax.imshow(arr, cmap="viridis", norm=norm, interpolation="nearest")
        ax.set_title(f"{s}×{s}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Overlay step index text (only for small maps)
        if s <= 8:
            for r in range(s):
                for c in range(s):
                    ax.text(c, r, int(arr[r, c]),
                            ha="center", va="center",
                            color="white" if arr[r, c] < max_step * 0.6 else "black",
                            fontsize=max(5, 10 - s))

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02,
                         fraction=0.04, label="unmask step")

    if cond_text:
        fig.suptitle(f"{title}\n“{cond_text[:120]}”", fontsize=9)
    else:
        fig.suptitle(title, fontsize=10)

    return fig


def _eval_one_image(img_chw, cond_text, detector, classifier, det_threshold):
    """Run detector+classifier eval on a single (-1, 1) image tensor.

    Returns per-image metrics dict with both raw counts and percentages:
      - count_correct    : bool
      - entity_found / entity_total  (raw)
      - rel_correct / rel_total      (raw)
      - count_acc, entity_acc, rel_acc : in [0, 100] (None if denom=0)
      - mean_acc        : simple average of the 3 acc numbers (skipping None)
    Returns None if eval models unavailable.
    """
    if detector is None or classifier is None:
        return None
    img01 = (img_chw.detach().float() * 0.5 + 0.5).clamp(0, 1)
    img01 = img01.unsqueeze(0).to(next(detector.parameters()).device)
    cond_json = clevr_text_to_condition_json(cond_text)
    res = eval_clevr_conditions(img01, [cond_json], detector, classifier,
                                 det_threshold=det_threshold)
    per = res["per_sample"][0]

    count_correct = bool(per["count_correct"])
    n_det = int(per.get("count_pred", -1))
    n_gt = int(per.get("count_gt", -1))
    ent_f, ent_t = int(per["entity_found"]), int(per["entity_total"])
    rel_c, rel_t = int(per["rel_correct"]), int(per["rel_total"])

    return {
        # raw per-image metrics (this is what goes into the JSON)
        "count_correct": count_correct,
        "n_detected": n_det,        # detector output count
        "n_expected": n_gt,         # GT entity count from caption
        "entity_matched": ent_f,    # GT entities with ANY matching detection
        "entity_total": ent_t,      # total GT entities
        "rel_satisfied": rel_c,     # GT relations satisfied by detections
        "rel_total": rel_t,         # total GT relations
    }


def _fmt_score(s):
    """Two-line display:
        #obj (detected/expected) [✓/✗]
        attrs N/M   rel N/M
    """
    if s is None:
        return ("(no eval)", "")
    mark = "✓" if s["count_correct"] else "✗"
    line1 = f"#obj det={s['n_detected']} / gt={s['n_expected']} {mark}"
    line2 = (f"attrs {s['entity_matched']}/{s['entity_total']}   "
             f"rel {s['rel_satisfied']}/{s['rel_total']}")
    return (line1, line2)


def _tensor_to_pil(img_chw):
    """(-1,1) float tensor (C,H,W) → PIL RGB."""
    arr = (img_chw * 0.5 + 0.5).clamp(0, 1).mul(255).byte().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr).convert("RGB")


def _overlay_8x8_grid(pil_img, grid=8, color=(255, 64, 64), width=2):
    """Draw a (grid × grid) overlay on an RGB PIL image."""
    img = pil_img.copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)
    cell_w = W / grid
    cell_h = H / grid
    for i in range(1, grid):
        x = int(round(i * cell_w))
        draw.line([(x, 0), (x, H)], fill=color, width=width)
        y = int(round(i * cell_h))
        draw.line([(0, y), (W, y)], fill=color, width=width)
    # Outer border
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline=color, width=width)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained DiT decode with selectable keep_levels
# ──────────────────────────────────────────────────────────────────────────────
def _build_keep_level_cond(pretrained_model, level_features, keep_levels_k,
                            device, dtype, B):
    """Build cond_tokens for ``forward_from_level_features`` where only the
    top-``keep_levels_k`` coarsest levels use real features; finer levels use
    the learned null_cond.

    Order of self.encoder.level_sizes is finest→coarsest.
    coarsest-first index:  cf_idx = num_levels - 1 - finest_idx
    A level is "active" iff cf_idx < keep_levels_k (same convention the
    training backbone uses for level-drop CFG).
    """
    num_levels = pretrained_model.num_levels
    cond_tokens_list = []
    for finest_i, s in enumerate(pretrained_model.encoder.level_sizes):
        cf_idx = num_levels - 1 - finest_i
        if cf_idx < keep_levels_k:
            feat = level_features[s].to(dtype)
            tokens = feat.flatten(2).transpose(1, 2)     # (B, s*s, feat_ch)
        else:
            tokens = pretrained_model.null_cond[str(s)].expand(B, -1, -1).to(dtype)
        cond_tokens_list.append(tokens)
    return cond_tokens_list


@torch.no_grad()
def _pretrained_forward_keep(pretrained_model, x_t, t, level_features,
                              keep_levels_k):
    """Monkey-patched copy of MultiResConditionalDiT.forward_from_level_features
    that lets us cap the usable levels via keep_levels_k (1 = coarsest only,
    num_levels = all levels)."""
    from model_multires import _sinusoidal_timestep_embedding

    B = x_t.shape[0]
    device = x_t.device
    dtype = x_t.dtype
    K = pretrained_model.in_context_len

    cond_tokens_list = _build_keep_level_cond(
        pretrained_model, level_features, keep_levels_k, device, dtype, B)
    keep_levels = torch.full((B,), keep_levels_k, device=device,
                              dtype=torch.long)

    cond_tokens = torch.cat(cond_tokens_list, dim=1)
    cond_tokens = pretrained_model.cond_proj(cond_tokens)

    offset = 0
    for s in pretrained_model.encoder.level_sizes:
        n = s * s
        cond_tokens[:, offset:offset + n] = (
            cond_tokens[:, offset:offset + n]
            + pretrained_model.cond_pos_embeds[str(s)])
        offset += n

    img_tokens = pretrained_model.patch_embed(x_t) + pretrained_model.pos_embed
    t_freq = _sinusoidal_timestep_embedding(t, pretrained_model._t_freq_dim)
    t_freq = t_freq.to(dtype=dtype)
    c = pretrained_model.time_embed(t_freq)

    tokens = img_tokens
    for i, block in enumerate(pretrained_model.blocks):
        if K > 0 and i == pretrained_model.in_context_start:
            ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
            ic_tokens = ic_tokens + pretrained_model.in_context_posemb
            tokens = torch.cat([ic_tokens, tokens], dim=1)

        has_prefix = (K > 0 and i >= pretrained_model.in_context_start)
        if has_prefix:
            rope_cos = pretrained_model._rope_cos_ext
            rope_sin = pretrained_model._rope_sin_ext
        else:
            rope_cos = pretrained_model._rope_cos
            rope_sin = pretrained_model._rope_sin

        xa_mask = pretrained_model._get_xa_mask(keep_levels,
                                                 has_prefix=has_prefix)
        tokens = block(tokens, c, cond=cond_tokens,
                       sa_mask=None, xa_mask=xa_mask,
                       rope_cos=rope_cos, rope_sin=rope_sin)

    if K > 0:
        img_out = tokens[:, K:, :]
    else:
        img_out = tokens
    img_out = pretrained_model.final_layer(img_out, c)
    return pretrained_model._unpatchify(img_out)


@torch.no_grad()
def decode_tokens_with_keep_levels(cont_tokens_b, level_sizes,
                                    pretrained_model, device,
                                    keep_levels_k, num_steps,
                                    noise_seed, t_eps=0.05):
    """Flow-matching Euler decode with a fixed noise seed and capped
    keep_levels_k.

    cont_tokens_b: (B, seq_len, feat_dim) float — one or more samples.
    noise_seed:    int (single seed for whole batch) OR sequence of ints
                   of length B (one seed per sample, for reproducibility
                   when batching samples that previously used distinct
                   per-sample seeds).
    Returns: (3, H, W) if B==1, else (B, 3, H, W).
    """
    pretrained_model.eval()
    pretrained_model.to(device)

    B = cont_tokens_b.shape[0]
    image_size = pretrained_model.image_size
    in_channels = getattr(pretrained_model, '_in_channels', 3)
    vae_factor = getattr(pretrained_model, 'vae_downsample_factor', 1)
    latent_size = image_size // vae_factor

    # Build initial noise. If caller gave a list/tuple of per-sample seeds,
    # draw each row with its own generator so the output is bit-identical to
    # the old per-sample decode calls.
    if isinstance(noise_seed, (list, tuple)):
        assert len(noise_seed) == B
        rows = []
        for s in noise_seed:
            g = torch.Generator(device=device).manual_seed(int(s))
            rows.append(torch.randn(1, in_channels, latent_size, latent_size,
                                     device=device, generator=g))
        z = torch.cat(rows, dim=0)
    else:
        gen = torch.Generator(device=device).manual_seed(int(noise_seed))
        z = torch.randn(B, in_channels, latent_size, latent_size,
                        device=device, generator=gen)

    level_features = _cont_tokens_to_level_features(
        cont_tokens_b, level_sizes, device)

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur
        t_batch = t_cur.expand(B)
        t_expand = t_cur.view(1, 1, 1, 1)

        x_pred = _pretrained_forward_keep(
            pretrained_model, z, t_batch, level_features, keep_levels_k)
        v = (x_pred - z) / (1.0 - t_expand).clamp_min(t_eps)
        z = z + dt * v

    z = z.clamp(-1, 1).cpu().float()
    if B == 1:
        return z.squeeze(0)
    return z


# ──────────────────────────────────────────────────────────────────────────────
# Model build + ckpt load
# ──────────────────────────────────────────────────────────────────────────────
def build_model(cfg, level_sizes, seq_len, cont_feat_dim, device):
    """Rebuild the DIT+DiscreteDiffusion+DiffLoss exactly as the training
    script does for this run (use_diffusion_head=True, cfg_mode=backbone,
    pos_emb_type=multires, continuous mode)."""
    dit_kwargs = dict(
        vocab_size=1,  # placeholder, unused in continuous mode
        seq_len=seq_len,
        hidden_size=cfg["hidden_size"],
        n_heads=cfg["n_heads"],
        n_blocks=cfg["n_blocks"],
        cond_dim=cfg["cond_dim"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["model_dropout"],
        causal=False,
        pos_emb_type=cfg["pos_emb_type"],
        level_sizes=level_sizes,
        continuous_mode=True,
        continuous_dim=cont_feat_dim,
        factorized_head=False,
    )
    backbone = DIT(**dit_kwargs)
    diff_head = DiffLoss(
        target_channels=cont_feat_dim,
        z_channels=cfg["hidden_size"],
        depth=cfg["diff_head_depth"],
        width=cfg["diff_head_width"],
        num_sampling_steps=cfg["diff_head_num_sampling_steps"],
        # cfg_mode=backbone → head drop forced to 0 at train time
        cond_drop_prob=0.0,
    )
    diffusion = DiscreteDiffusion(
        backbone=backbone,
        vocab_size=1,
        noise_type=cfg["noise_type"],
        noise_eps=cfg["noise_eps"],
        antithetic_sampling=cfg["antithetic_sampling"],
        importance_sampling=cfg["importance_sampling"],
        change_of_variables=cfg["change_of_variables"],
        sampling_eps=cfg["sampling_eps"],
        diff_head=diff_head,
        diffusion_batch_mul=cfg["diff_head_batch_mul"],
        time_conditioning=cfg["time_conditioning"],
    )
    return diffusion


def load_checkpoint(ckpt_dir, diffusion, text_encoder, device, variant):
    """Load model/model_1 safetensors (+ optional EMA shadow)."""
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    cond_path = os.path.join(ckpt_dir, "model_1.safetensors")

    state = load_file(model_path, device=str(device))
    diffusion.load_state_dict(state, strict=True)

    if os.path.isfile(cond_path):
        cond_state = load_file(cond_path, device=str(device))
        text_encoder.load_state_dict(cond_state, strict=True)

    if variant == "ema":
        ema_path = os.path.join(ckpt_dir, "ema.pt")
        if os.path.isfile(ema_path):
            shadow = torch.load(ema_path, map_location=device,
                                weights_only=False)
            params = list(diffusion.parameters())
            assert len(shadow) == len(params), (
                f"EMA shadow length mismatch: {len(shadow)} vs {len(params)}")
            for s, p in zip(shadow, params):
                p.data.copy_(s.to(device))
        else:
            print(f"[warn] ema.pt not found in {ckpt_dir} — using base weights")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    out_dir = os.path.join(args.run_dir, args.output_subdir)
    order_dir = os.path.join(out_dir, "generation_order")
    ablation_dir = os.path.join(out_dir, "level_ablation")
    overlay_dir = os.path.join(out_dir, "final_grid_overlay")
    if is_main:
        for d in (out_dir, order_dir, ablation_dir, overlay_dir):
            os.makedirs(d, exist_ok=True)

    # ── Load train config ────────────────────────────────────────────────────
    run_cfg_path = os.path.join(args.run_dir, "run_config.json")
    with open(run_cfg_path) as f:
        run_cfg = json.load(f)["args"]

    pretrained_dir = run_cfg["pretrained_output_dir"]
    if not os.path.isabs(pretrained_dir):
        # "./runs/..." relative to MNIST_debug/
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))
        pretrained_dir = os.path.normpath(
            os.path.join(repo_root, pretrained_dir))

    # ── Load pretrained (backbone + discretizer) ─────────────────────────────
    accelerator.print(f"[pretrained] Loading from {pretrained_dir}")
    pretrained_model, encoder, discretizer, level_sizes, data_vocab_size, pre_cfg = \
        load_pretrained_model(pretrained_dir, device=device)
    pretrained_model.eval()
    seq_len = sum(s * s for s in level_sizes)
    accelerator.print(
        f"[pretrained] level_sizes={level_sizes} "
        f"(finest→coarsest), seq_len={seq_len}")

    # ── CLEVR val dataset ────────────────────────────────────────────────────
    val_image_root = args.val_image_root
    val_cond_dir = args.val_cond_dir
    if not os.path.isabs(val_image_root):
        # Keep consistent with training working dir (MNIST_debug/)
        val_image_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), val_image_root))
    if not os.path.isabs(val_cond_dir):
        val_cond_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), val_cond_dir))

    accelerator.print(
        f"[data] val images={val_image_root}\n"
        f"[data] val cond  ={val_cond_dir}")
    val_img_ds = CLEVRImageDataset(
        val_image_root, condition_dir=val_cond_dir,
        image_size=args.image_size, splits=args.splits, cond_type="text")
    val_img_only = CLEVRImageDataset(
        val_image_root, condition_dir=None,
        image_size=args.image_size, splits=args.splits)

    cache_dir = os.path.join(args.run_dir, "token_cache")
    val_cache_path = os.path.join(cache_dir, "clevr_val_cont.pt")
    val_feats_unique = cache_all_continuous_tokens(
        encoder, val_img_only, device,
        batch_size=32, cache_path=val_cache_path, accelerator=accelerator)
    accelerator.wait_for_everyone()

    val_path_to_idx = {p: i for i, p in enumerate(val_img_only.image_paths)}
    val_tok_indices = [val_path_to_idx[p] for p in val_img_ds.image_paths]
    val_feats = val_feats_unique[val_tok_indices]

    val_conditions = [val_img_ds.get_condition(i) for i in range(len(val_img_ds))]
    cont_feat_dim = val_feats.shape[-1]
    accelerator.print(
        f"[data] seq_len={val_feats.shape[1]}, feat_dim={cont_feat_dim}, "
        f"n_val={len(val_feats)}")

    val_dataset = CachedContinuousTokenDataset(
        val_feats, clevr_conditions=val_conditions,
        cond_tokenizer_fn=None, return_raw_text=True,
        source_image_ds=val_img_ds)

    # Move pretrained to CPU while building the generator; loaded back to
    # GPU only during decode.
    pretrained_model.cpu()
    torch.cuda.empty_cache()

    # ── Select samples (deterministic, balanced per split) ───────────────────
    # NOTE: _select_eval_indices is the exact same routine used at training
    # time (seed=42 inside), so `samples_per_split=30` reproduces the full
    # training eval set; smaller values select a prefix of it.
    selected_indices, sample_splits = _select_eval_indices(
        val_dataset, args.samples_per_split)
    n_samples = len(selected_indices)

    cond_texts = []
    cond_jsons = []
    for idx in selected_indices:
        sample = val_dataset[idx]
        cond_texts.append(sample.get(
            "cond_text",
            _extract_raw_text(val_dataset.get_condition(idx))))
        cond_jsons.append(val_dataset.get_condition(idx))

    # ── Decide which samples get heavy visualizations ────────────────────────
    # ALL samples contribute to eval scores. Only `viz_samples_per_split` (per
    # split) additionally get: generation_order heatmap, history.pt, decoded
    # PNG, 8x8 overlay, level_ablation figure, sampler_compare figure.
    viz_per_split = (args.viz_samples_per_split
                     if args.viz_samples_per_split is not None
                     else args.samples_per_split)
    viz_per_split = min(viz_per_split, args.samples_per_split)
    viz_gi_set = set()
    _seen_per_split = {}
    for gi, split in enumerate(sample_splits):
        c = _seen_per_split.get(split, 0)
        if c < viz_per_split:
            viz_gi_set.add(gi)
            _seen_per_split[split] = c + 1
    accelerator.print(
        f"[samples] eval n={n_samples} "
        f"(per_split={args.samples_per_split}); "
        f"viz subset n={len(viz_gi_set)} (per_split={viz_per_split})")

    if is_main:
        meta = {
            "step": args.step,
            "variant": args.variant,
            "n_samples": n_samples,
            "samples_per_split": args.samples_per_split,
            "viz_samples_per_split": viz_per_split,
            "viz_sample_indices": sorted(int(g) for g in viz_gi_set),
            "splits": sample_splits,
            "selected_indices": [int(i) for i in selected_indices],
            "cond_texts": cond_texts,
            "level_sizes": level_sizes,
            "samplers": args.samplers,
            "num_decode_seeds": args.num_decode_seeds,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Build text encoder + diffusion model ─────────────────────────────────
    text_encoder = PretrainedTextConditionEncoder(
        model_name=run_cfg["pretrained_text_model_name"],
        hidden_size=run_cfg["hidden_size"],
        max_length=run_cfg["pretrained_text_max_length"],
        freeze=run_cfg["freeze_text_encoder"],
    ).to(device)

    diffusion = build_model(run_cfg, level_sizes, seq_len, cont_feat_dim, device)
    diffusion = diffusion.to(device)

    ckpt_dir = os.path.join(args.run_dir, "ckpt", f"step{args.step}")
    load_checkpoint(ckpt_dir, diffusion, text_encoder, device, args.variant)
    accelerator.print(
        f"[ckpt] loaded step={args.step} variant={args.variant} from {ckpt_dir}")

    # Semi-autoregressive (coarse-to-fine) sampling: if the run was trained
    # with --semi_autoregressive, replicate the same position→level map so
    # sample_continuous dispatches to _sample_continuous_semi_ar.
    if run_cfg.get("semi_autoregressive", False):
        sorted_sizes = sorted(level_sizes, reverse=True)  # finest → coarsest
        idx = []
        for li, s in enumerate(sorted_sizes):
            idx.extend([li] * (s * s))
        semi_ar_level_idx = torch.tensor(idx, dtype=torch.long, device=device)
        diffusion.set_semi_ar(semi_ar_level_idx, len(sorted_sizes))
        accelerator.print(
            f"[semi_ar] enabled for visualization — "
            f"n_levels={len(sorted_sizes)}, "
            f"level_sizes(finest→coarsest)={sorted_sizes}")

    diffusion.eval()
    text_encoder.eval()

    # ── Encode all selected conditions (same on every rank) ──────────────────
    with torch.no_grad():
        text_tokens = text_encoder.tokenize(cond_texts, device)
        all_cond, _ = text_encoder(text_tokens)                   # (N, L, D)
        all_uncond = text_encoder.get_null_cond(
            all_cond.shape[0], all_cond.shape[1], device).to(all_cond.dtype)

    # ── Round-robin shard across ranks ───────────────────────────────────────
    my_idx = list(range(rank, n_samples, world_size))
    accelerator.print(
        f"[rank {rank}/{world_size}] handling {len(my_idx)} / {n_samples} samples")

    # ── Sampler configs (match train_discrete_diffusion_v2.py eval_clevr) ────
    sampler_lookup = {
        "ddpm_cache":        dict(sampler="ddpm_cache",  tokens_per_step=0),
        "confidence_top1":   dict(sampler="confidence",  tokens_per_step=1),
        "confidence_cosine": dict(sampler="confidence",  tokens_per_step=0),
    }

    inner = diffusion  # no accelerate.prepare — use directly
    max_step = args.eval_num_steps + 2  # final pass / argmax pass may add one

    # Resolve CFG sweep. Defaults reproduce the single (cfg, schedule) stored
    # in run_config.json → backward-compatible behavior.
    cfg_values = list(args.cfg_values) if args.cfg_values is not None \
        else [float(run_cfg["diff_head_cfg"])]
    cfg_schedules = list(args.cfg_schedules) if args.cfg_schedules is not None \
        else [run_cfg["cfg_schedule"]]
    decode_ablation_cfg = (args.decode_ablation_cfg
                           if args.decode_ablation_cfg is not None
                           else cfg_values[0])
    decode_ablation_schedule = (args.decode_ablation_schedule
                                if args.decode_ablation_schedule is not None
                                else cfg_schedules[0])
    accelerator.print(
        f"[cfg-sweep] values={cfg_values}  schedules={cfg_schedules}\n"
        f"[cfg-sweep] level-ablation fixed at cfg={decode_ablation_cfg} "
        f"schedule={decode_ablation_schedule} "
        f"(sampler={args.decode_ablation_sampler})")

    def _cs_tag(cfg_val: float, sched: str) -> str:
        """Filename-safe suffix for a (cfg, schedule) combo."""
        return f"cfg{cfg_val:g}_{sched}"

    # Storage keyed by (tag, cfg_val, sched, gi) → (1, L, D) cpu tensor
    tokens_by_sampler = {}
    tokens_by_sample = {}  # gi → tokens for designated (ablation_sampler, ablation_cfg, ablation_sched)

    # Per-sample eval scores: nested dict filled later
    #   eval_scores[gi] = {
    #     "sampler/{tag}/{cs_tag}": {...},  # full-levels decode, seed=0
    #     "ablation/L{k}_seed{s}": {...},   # level k = 1..num_levels, seed s
    #   }
    eval_scores = {}

    # ═══════════════════════════════════════════════════════════════════════
    # (1) Generation — swept over (sampler × cfg × sched)
    #     Viz subset: batch=1 (return_history, save generation_order heatmap)
    #     Non-viz:    batched by --gen_batch_size (return_history=False)
    # ═══════════════════════════════════════════════════════════════════════
    viz_my_idx = [gi for gi in my_idx if gi in viz_gi_set]
    non_viz_my_idx = [gi for gi in my_idx if gi not in viz_gi_set]

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    for tag in args.samplers:
        if tag not in sampler_lookup:
            accelerator.print(f"[warn] unknown sampler {tag} — skipping")
            continue
        sc = sampler_lookup[tag]

        for cfg_val in cfg_values:
            for sched in cfg_schedules:
                cs = _cs_tag(cfg_val, sched)
                accelerator.print(
                    f"\n[sampler] {tag} | {cs} (rank {rank}) — "
                    f"viz={len(viz_my_idx)} + non-viz={len(non_viz_my_idx)} "
                    f"(gen_bs={args.gen_batch_size})")

                # ── Viz: per-sample B=1 to preserve history ──────────────
                for gi in viz_my_idx:
                    split = sample_splits[gi]
                    cond = all_cond[gi:gi + 1]
                    uncond = all_uncond[gi:gi + 1]
                    torch.manual_seed(args.seed + gi)
                    with torch.no_grad():
                        x_final, mask_history = inner.sample_continuous(
                            batch_size=1, seq_len=seq_len,
                            feat_dim=cont_feat_dim,
                            num_steps=args.eval_num_steps,
                            device=device,
                            sampler=sc["sampler"],
                            tokens_per_step=sc["tokens_per_step"],
                            cond_tokens=cond,
                            uncond_cond_tokens=uncond,
                            temperature=run_cfg["diff_head_temperature"],
                            cfg=cfg_val,
                            cfg_schedule=sched,
                            cfg_mode=run_cfg["cfg_mode"],
                            null_class_index=None,
                            return_history=True,
                        )
                    tokens_by_sampler[(tag, cfg_val, sched, gi)] = \
                        x_final.detach().cpu()
                    if (tag == args.decode_ablation_sampler
                            and cfg_val == decode_ablation_cfg
                            and sched == decode_ablation_schedule):
                        tokens_by_sample[gi] = x_final.detach().cpu()

                    unmask_step = _compute_unmask_step(mask_history)
                    fig = _render_order_figure(
                        unmask_step[0], cond_texts[gi], level_sizes,
                        title=f"[{tag}] {cs} split={split} sample={gi}",
                        max_step=len(mask_history) - 1)
                    out_path = os.path.join(
                        order_dir,
                        f"{tag}_{cs}_split-{split}_sample-{gi:02d}_order.png")
                    fig.savefig(out_path, dpi=140, bbox_inches="tight")
                    import matplotlib.pyplot as plt
                    plt.close(fig)

                    torch.save(
                        {
                            "tag": tag, "cfg": cfg_val,
                            "cfg_schedule": sched, "split": split,
                            "sample_idx": int(gi),
                            "cond_text": cond_texts[gi],
                            "level_sizes": list(level_sizes),
                            "mask_history": torch.stack(mask_history).bool(),
                            "unmask_step": unmask_step.long(),
                            "tokens_final": x_final.detach().cpu().float(),
                        },
                        os.path.join(
                            order_dir,
                            f"{tag}_{cs}_split-{split}_sample-{gi:02d}_history.pt"))

                # ── Non-viz: batched, no history ─────────────────────────
                for chunk in _chunks(non_viz_my_idx, args.gen_batch_size):
                    B = len(chunk)
                    idx_t = torch.tensor(chunk, dtype=torch.long,
                                         device=all_cond.device)
                    cond_b = all_cond.index_select(0, idx_t)
                    uncond_b = all_uncond.index_select(0, idx_t)
                    torch.manual_seed(args.seed + chunk[0])
                    with torch.no_grad():
                        x_final = inner.sample_continuous(
                            batch_size=B, seq_len=seq_len,
                            feat_dim=cont_feat_dim,
                            num_steps=args.eval_num_steps,
                            device=device,
                            sampler=sc["sampler"],
                            tokens_per_step=sc["tokens_per_step"],
                            cond_tokens=cond_b,
                            uncond_cond_tokens=uncond_b,
                            temperature=run_cfg["diff_head_temperature"],
                            cfg=cfg_val,
                            cfg_schedule=sched,
                            cfg_mode=run_cfg["cfg_mode"],
                            null_class_index=None,
                            return_history=False,
                        )
                    x_final_cpu = x_final.detach().cpu()
                    for i, gi in enumerate(chunk):
                        tokens_by_sampler[(tag, cfg_val, sched, gi)] = \
                            x_final_cpu[i:i + 1]
                        if (tag == args.decode_ablation_sampler
                                and cfg_val == decode_ablation_cfg
                                and sched == decode_ablation_schedule):
                            tokens_by_sample[gi] = x_final_cpu[i:i + 1]

    accelerator.wait_for_everyone()
    # Generators we just ran put the big pretrained model back on GPU too, but
    # only after we reach the decode stage. For decode we want a clean GPU.
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # (2) Level-ablation decodes (same tokens, different decoder noise seed)
    # (3) Final 8x8 grid overlay — uses the K=num_levels decode output.
    # (4) Per-sampler full-levels decode for eval comparison.
    # ═══════════════════════════════════════════════════════════════════════
    # Move diffusion off GPU to free decoder memory.
    diffusion.cpu()
    text_encoder.cpu()
    torch.cuda.empty_cache()
    pretrained_model.to(device)

    # ── Load detector + classifier (rule-based eval) ─────────────────────────
    detector = None
    classifier = None
    if args.run_eval:
        try:
            detector, classifier = load_eval_models(device=str(device))
            accelerator.print(
                f"[eval] loaded detector + classifier on {device}")
        except Exception as e:
            accelerator.print(f"[eval] load_eval_models failed: {e} — "
                              f"continuing without eval")

    num_levels = len(level_sizes)
    # coarsest-first levels: 1, 2, 4, 8 → keep_levels_k = 1, 2, 3, 4
    ordered_levels = sorted(level_sizes)  # [1, 2, 4, 8]

    # Font helpers
    def _font(size):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _font_regular(size):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    # Val dataset for GT image lookup
    val_img_ds_ref = val_img_ds

    # ═══════════════════════════════════════════════════════════════════════
    # Phase A: Full-level decode for ALL samples (eval score) — batched
    #          per (cfg, sched, tag) across samples.
    # ═══════════════════════════════════════════════════════════════════════
    decoded_images = {}   # (gi, cfg, sched, tag) -> (3, H, W) float cpu tensor
    sample_evals = {int(gi): {} for gi in my_idx}

    for cfg_val in cfg_values:
        for sched in cfg_schedules:
            cs = _cs_tag(cfg_val, sched)
            for tag in args.samplers:
                avail = [gi for gi in my_idx
                         if (tag, cfg_val, sched, gi) in tokens_by_sampler]
                if not avail:
                    continue
                accelerator.print(
                    f"[decode full] {tag} | {cs} — {len(avail)} samples "
                    f"(bs={args.decode_batch_size})")
                for chunk in _chunks(avail, args.decode_batch_size):
                    B = len(chunk)
                    tok_batch = torch.cat(
                        [tokens_by_sampler[(tag, cfg_val, sched, gi)]
                         for gi in chunk], dim=0).to(device)
                    seeds = [args.seed * 10000 + gi * 100 + 0 for gi in chunk]
                    imgs = decode_tokens_with_keep_levels(
                        tok_batch, level_sizes, pretrained_model, device,
                        keep_levels_k=num_levels,
                        num_steps=args.decode_num_steps,
                        noise_seed=seeds,
                    )  # (B, 3, H, W) since B>=1 may still be 1
                    if imgs.dim() == 3:
                        imgs = imgs.unsqueeze(0)
                    for i, gi in enumerate(chunk):
                        img = imgs[i]
                        s = _eval_one_image(
                            img, cond_texts[gi], detector, classifier,
                            args.eval_det_threshold)
                        sample_evals[int(gi)][f"sampler/{tag}/{cs}"] = s
                        # Only retain decoded image for viz samples — Phase B
                        # uses them for PNG saves and sampler_compare. For
                        # eval-only (viz_samples_per_split=0) this keeps
                        # memory bounded on a 90-sample run.
                        if gi in viz_gi_set:
                            decoded_images[(gi, cfg_val, sched, tag)] = img

    # ═══════════════════════════════════════════════════════════════════════
    # Phase B: Per-sample artifacts. Non-viz → just persist eval scores.
    #          Viz samples → save PNGs, level-ablation, overlay, compare fig.
    # ═══════════════════════════════════════════════════════════════════════
    for li, gi in enumerate(my_idx):
        split = sample_splits[gi]
        cond_txt = cond_texts[gi]
        sample_eval = sample_evals[int(gi)]
        is_viz_sample = (gi in viz_gi_set)

        # Build sampler_images_by_cs from decoded_images lookup.
        sampler_images_by_cs = {}
        for cfg_val in cfg_values:
            for sched in cfg_schedules:
                sampler_images_by_cs[(cfg_val, sched)] = {}
                for tag in args.samplers:
                    img = decoded_images.get((gi, cfg_val, sched, tag))
                    if img is not None:
                        sampler_images_by_cs[(cfg_val, sched)][tag] = img

        # Save per-sampler decoded PNG + 8x8 overlay for viz samples only.
        if is_viz_sample:
            for cfg_val in cfg_values:
                for sched in cfg_schedules:
                    cs = _cs_tag(cfg_val, sched)
                    for tag in args.samplers:
                        img = sampler_images_by_cs[(cfg_val, sched)].get(tag)
                        if img is None:
                            continue
                        s = sample_eval.get(f"sampler/{tag}/{cs}")
                        pil_decoded = _tensor_to_pil(img).resize(
                            (512, 512), Image.BILINEAR)
                        pil_decoded.save(os.path.join(
                            order_dir,
                            f"{tag}_{cs}_split-{split}_sample-{gi:02d}_decoded.png"))
                        pil_overlay = _overlay_8x8_grid(
                            pil_decoded.copy(), grid=8, width=3)
                        if s is not None:
                            l1, l2 = _fmt_score(s)
                            _od = ImageDraw.Draw(pil_overlay)
                            f1 = _font(22); f2 = _font(20)
                            tb1 = _od.textbbox((0, 0), l1, font=f1)
                            tb2 = _od.textbbox((0, 0), l2, font=f2)
                            bw = max(tb1[2] - tb1[0], tb2[2] - tb2[0]) + 14
                            bh = (tb1[3] - tb1[1]) + (tb2[3] - tb2[1]) + 16
                            _od.rectangle([(6, 6), (6 + bw, 6 + bh)],
                                          fill=(0, 0, 0))
                            _od.text((12, 10), l1,
                                     fill=(255, 240, 100), font=f1)
                            _od.text((12, 10 + 26), l2,
                                     fill=(255, 255, 255), font=f2)
                        pil_overlay.save(os.path.join(
                            order_dir,
                            f"{tag}_{cs}_split-{split}_sample-{gi:02d}_decoded_8x8overlay.png"))

        # Non-viz samples: finalize eval entry and move on.
        if not is_viz_sample:
            eval_scores[gi] = {
                "split": split,
                "sample_idx": int(gi),
                "cond_text": cond_txt,
                "scores": sample_eval,
            }
            continue

        # Viz samples: need tokens for level ablation.
        if gi not in tokens_by_sample:
            accelerator.print(
                f"[warn] rank {rank} viz sample {gi} has no tokens_by_sample "
                f"(ablation sampler {args.decode_ablation_sampler} cfg="
                f"{decode_ablation_cfg} sched={decode_ablation_schedule} "
                f"not in sweep) — skipping ablation/overlay")
            eval_scores[gi] = {
                "split": split,
                "sample_idx": int(gi),
                "cond_text": cond_txt,
                "scores": sample_eval,
            }
            continue
        tokens_b = tokens_by_sample[gi].to(device)   # (1, L, D)

        # ── GT image + GT 8×8 overlay — saved under generation_order/ ────────
        try:
            gt_tensor_early = val_img_ds_ref[selected_indices[gi]]["image"]
            gt_pil_early = _tensor_to_pil(gt_tensor_early).resize(
                (512, 512), Image.BILINEAR)
            gt_pil_early.save(os.path.join(
                order_dir,
                f"gt_split-{split}_sample-{gi:02d}.png"))
            gt_overlay = _overlay_8x8_grid(gt_pil_early.copy(), grid=8,
                                            width=3)
            gt_od = ImageDraw.Draw(gt_overlay)
            gt_od.text((12, 10), "GT", fill=(255, 210, 0), font=_font(24))
            gt_overlay.save(os.path.join(
                order_dir,
                f"gt_split-{split}_sample-{gi:02d}_8x8overlay.png"))
        except Exception as _e:
            accelerator.print(
                f"[warn] GT save failed for sample {gi}: {_e}")

        # ── (2) Per-level ablation — batched across decoder noise seeds ─────
        rows = []          # rows: levels; cols: seeds (each entry = CHW tensor)
        row_evals = []     # parallel 2D list of eval dicts
        n_seeds = args.num_decode_seeds
        tok_replicated = tokens_b.expand(n_seeds, -1, -1).contiguous()
        for li_idx, s in enumerate(ordered_levels):
            k = li_idx + 1                           # keep_levels_k
            seeds = [args.seed * 10000 + gi * 100 + seed_i
                     for seed_i in range(n_seeds)]
            imgs = decode_tokens_with_keep_levels(
                tok_replicated, level_sizes, pretrained_model, device,
                keep_levels_k=k,
                num_steps=args.decode_num_steps,
                noise_seed=seeds,
            )  # (n_seeds, 3, H, W) or (3, H, W) if n_seeds==1
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            row = [imgs[i] for i in range(n_seeds)]
            row_ev = []
            for seed_i in range(n_seeds):
                ev = _eval_one_image(row[seed_i], cond_txt, detector,
                                     classifier, args.eval_det_threshold)
                row_ev.append(ev)
                sample_eval[f"ablation/L{k}_seed{seed_i}"] = ev
            rows.append(row)
            row_evals.append(row_ev)

        # ─────────────────────────────────────────────────────────────────────
        # Compose the level-ablation figure manually so that each cell has a
        # clean eval-score caption directly underneath it.
        # Layout:
        #   left margin (level labels)   [cell][cap]  [cell][cap]  [cell][cap]
        #   caption: condition text at the bottom
        # ─────────────────────────────────────────────────────────────────────
        cell = 512          # upscaled from model's native 256 for readability
        cap_h = 70          # per-cell caption area (fits two lines + padding)
        left_w = 110        # left margin for level label
        pad = 10
        ncols = args.num_decode_seeds
        nrows = len(ordered_levels)
        top_label_h = 36
        bottom_cap_h = 60
        grid_w = left_w + ncols * cell + (ncols + 1) * pad
        grid_h = (top_label_h
                  + nrows * (cell + cap_h)
                  + (nrows + 1) * pad
                  + bottom_cap_h)
        canvas = Image.new("RGB", (grid_w, grid_h), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        f_level = _font(36)
        f_cap1 = _font(22)            # main eval line (bold)
        f_cap2 = _font_regular(20)    # secondary eval line
        f_seed = _font(22)
        f_bot = _font_regular(22)

        for ri, s in enumerate(ordered_levels):
            y_cell = top_label_h + pad + ri * (cell + cap_h + pad)
            # Level label on the left
            draw.text((8, y_cell + cell // 2 - 18),
                      f"{s}×{s}", fill=(255, 210, 0), font=f_level)
            for ci in range(ncols):
                x_cell = left_w + pad + ci * (cell + pad)
                pil_cell = _tensor_to_pil(rows[ri][ci])
                if pil_cell.size != (cell, cell):
                    pil_cell = pil_cell.resize((cell, cell), Image.BILINEAR)
                canvas.paste(pil_cell, (x_cell, y_cell))
                # Per-cell eval caption (two lines)
                ev = row_evals[ri][ci]
                line1, line2 = _fmt_score(ev)
                # Seed label on the first row
                if ri == 0:
                    draw.text((x_cell + 6, 6),
                              f"seed {ci}", fill=(220, 220, 220), font=f_seed)
                draw.text((x_cell + 6, y_cell + cell + 6),
                          line1, fill=(255, 240, 130), font=f_cap1)
                draw.text((x_cell + 6, y_cell + cell + 6 + 28),
                          line2, fill=(235, 235, 235), font=f_cap2)

        # Bottom caption: condition text
        cap = cond_txt[:240]
        y_bot = top_label_h + nrows * (cell + cap_h + pad) + pad
        draw.text((8, y_bot + 10),
                  f"[{split} / sample {gi}] {cap}",
                  fill=(230, 230, 230), font=f_bot)

        out_path = os.path.join(
            ablation_dir,
            f"split-{split}_sample-{gi:02d}_level_ablation.png")
        canvas.save(out_path)

        # ── (3) 8×8 overlay on the canonical image (last row, seed 0) ────────
        full_img = rows[-1][0]
        pil_full = _tensor_to_pil(full_img)
        # Upscale for readability
        pil_full_up = pil_full.resize((512, 512), Image.BILINEAR)
        overlay = _overlay_8x8_grid(pil_full_up, grid=8, width=3)
        # Add eval score text on overlay (two lines with dark background)
        full_ev = row_evals[-1][0]
        ov_draw = ImageDraw.Draw(overlay)
        line1, line2 = _fmt_score(full_ev)
        f_ov1 = _font(24)
        f_ov2 = _font(22)
        tb1 = ov_draw.textbbox((0, 0), line1, font=f_ov1)
        tb2 = ov_draw.textbbox((0, 0), line2, font=f_ov2)
        box_w = max(tb1[2] - tb1[0], tb2[2] - tb2[0]) + 14
        box_h = (tb1[3] - tb1[1]) + (tb2[3] - tb2[1]) + 18
        ov_draw.rectangle([(6, 6), (6 + box_w, 6 + box_h)],
                          fill=(0, 0, 0))
        ov_draw.text((12, 10), line1, fill=(255, 240, 100), font=f_ov1)
        ov_draw.text((12, 10 + 28), line2, fill=(255, 255, 255), font=f_ov2)
        overlay.save(os.path.join(
            overlay_dir,
            f"split-{split}_sample-{gi:02d}_8x8_overlay.png"))
        pil_full_up.save(os.path.join(
            overlay_dir,
            f"split-{split}_sample-{gi:02d}_full.png"))

        # ── Per-sampler comparison figure per (cfg, sched) ───────────────────
        # [GT][sampler_A][sampler_B]... side-by-side so we can eyeball how
        # each sampler responds to a given CFG setting.
        try:
            gt_tensor = val_img_ds_ref[selected_indices[gi]]["image"]
            gt_pil = _tensor_to_pil(gt_tensor)
        except Exception:
            gt_pil = None

        for cfg_val in cfg_values:
            for sched in cfg_schedules:
                cs = _cs_tag(cfg_val, sched)
                sampler_images_cs = sampler_images_by_cs.get(
                    (cfg_val, sched), {})

                panels = []
                panel_titles = []
                panel_scores = []
                if gt_pil is not None:
                    panels.append(gt_pil)
                    panel_titles.append("GT")
                    panel_scores.append(None)
                for tag in args.samplers:
                    if tag in sampler_images_cs:
                        panels.append(_tensor_to_pil(sampler_images_cs[tag]))
                        panel_titles.append(tag)
                        panel_scores.append(
                            sample_eval.get(f"sampler/{tag}/{cs}"))

                if not panels:
                    continue

                n_p = len(panels)
                cmp_cell = 512
                cmp_cap = 70
                cmp_title = 36
                header_h = 40   # extra strip for cfg/sched header
                cmp_w = n_p * cmp_cell + (n_p + 1) * pad
                cmp_h = (header_h + cmp_title + cmp_cell + cmp_cap
                         + 3 * pad + 60)
                cmp_canvas = Image.new("RGB", (cmp_w, cmp_h), (18, 18, 18))
                cmp_draw = ImageDraw.Draw(cmp_canvas)
                f_header = _font(28)
                f_title = _font(30)
                f_cap1 = _font(22)
                f_cap2 = _font_regular(20)

                cmp_draw.text(
                    (8, 6),
                    f"CFG={cfg_val:g} schedule={sched}",
                    fill=(120, 220, 255), font=f_header)

                for pi, (p_img, p_title, p_score) in enumerate(
                        zip(panels, panel_titles, panel_scores)):
                    x = pad + pi * (cmp_cell + pad)
                    y_title = header_h + pad
                    y_cell = y_title + cmp_title + pad
                    y_cap = y_cell + cmp_cell + 6
                    if p_img.size != (cmp_cell, cmp_cell):
                        p_img = p_img.resize(
                            (cmp_cell, cmp_cell), Image.BILINEAR)
                    cmp_canvas.paste(p_img, (x, y_cell))
                    cmp_draw.text((x + 6, y_title + 2), p_title,
                                  fill=(255, 210, 0), font=f_title)
                    if p_score is None:
                        cmp_draw.text((x + 6, y_cap),
                                      "(ground truth — no eval)",
                                      fill=(180, 180, 180), font=f_cap2)
                    else:
                        l1, l2 = _fmt_score(p_score)
                        cmp_draw.text((x + 6, y_cap), l1,
                                      fill=(255, 240, 130), font=f_cap1)
                        cmp_draw.text((x + 6, y_cap + 28), l2,
                                      fill=(235, 235, 235), font=f_cap2)

                y_bot = y_cap + cmp_cap + pad
                cmp_draw.text(
                    (8, y_bot),
                    f"[{split} / sample {gi}] {cond_txt[:240]}",
                    fill=(230, 230, 230), font=f_bot)
                cmp_canvas.save(os.path.join(
                    ablation_dir,
                    f"split-{split}_sample-{gi:02d}_{cs}_sampler_compare.png"))

        # Save GT by itself for easy browsing (once per sample).
        if gt_pil is not None:
            gt_pil.save(os.path.join(
                overlay_dir,
                f"split-{split}_sample-{gi:02d}_gt.png"))

        eval_scores[gi] = {
            "split": split,
            "sample_idx": int(gi),
            "cond_text": cond_txt,
            "scores": sample_eval,
        }

        accelerator.print(
            f"[rank {rank}] sample {gi} ({split}) → ablation + overlay + "
            f"sampler-compare written")

    accelerator.wait_for_everyone()

    # ── Gather eval scores across ranks to rank 0 ────────────────────────────
    if accelerator.num_processes > 1:
        import pickle
        import tempfile
        scratch = os.path.join(out_dir, "_eval_shards")
        if is_main:
            os.makedirs(scratch, exist_ok=True)
        accelerator.wait_for_everyone()
        with open(os.path.join(scratch, f"rank{rank}.pkl"), "wb") as f:
            pickle.dump(eval_scores, f)
        accelerator.wait_for_everyone()
        if is_main:
            merged = {}
            for r in range(accelerator.num_processes):
                shard = os.path.join(scratch, f"rank{r}.pkl")
                if os.path.isfile(shard):
                    with open(shard, "rb") as f:
                        merged.update(pickle.load(f))
            eval_scores = merged
    else:
        merged = eval_scores

    # ── Save eval_scores.json + print summary ────────────────────────────────
    if is_main:
        final_scores = merged if accelerator.num_processes > 1 else eval_scores
        out_json = os.path.join(out_dir, "eval_scores.json")
        with open(out_json, "w") as f:
            json.dump(final_scores, f, indent=2,
                      default=lambda o: None)
        accelerator.print(f"\n[eval] wrote per-sample eval to {out_json}")

        # Aggregate per (sampler, cfg, schedule) and per ablation level.
        def _empty_agg():
            return dict(n=0, count_correct=0,
                        entity_matched=0, entity_total=0,
                        rel_satisfied=0, rel_total=0)

        # per_sampler_cs[(tag, cfg, sched)] — new multi-key bucket
        # per_sampler[tag]                  — legacy/collapsed view
        per_sampler_cs = {}
        per_sampler = {t: _empty_agg() for t in args.samplers}
        per_level = {k: _empty_agg() for k in range(1, num_levels + 1)}

        for gi, rec in final_scores.items():
            for key, sc in rec["scores"].items():
                if sc is None:
                    continue
                if key.startswith("sampler/"):
                    # "sampler/{tag}/cfg{X}_{sched}" or (legacy) "sampler/{tag}"
                    parts = key.split("/", 2)
                    tag = parts[1] if len(parts) > 1 else ""
                    cs_key = parts[2] if len(parts) > 2 else "_default"
                    if tag not in per_sampler:
                        continue
                    ck = (tag, cs_key)
                    per_sampler_cs.setdefault(ck, _empty_agg())
                    targets = [per_sampler[tag], per_sampler_cs[ck]]
                elif key.startswith("ablation/L"):
                    k = int(key.split("/")[1].split("_")[0].lstrip("L"))
                    targets = [per_level[k]]
                else:
                    continue
                for a in targets:
                    a["n"] += 1
                    a["count_correct"] += int(sc["count_correct"])
                    a["entity_matched"] += sc["entity_matched"]
                    a["entity_total"]   += sc["entity_total"]
                    a["rel_satisfied"]  += sc["rel_satisfied"]
                    a["rel_total"]      += sc["rel_total"]

        def _fmt_agg(a):
            if a["n"] == 0:
                return "(no samples)"
            ca = a["count_correct"] / a["n"] * 100
            ea = (a["entity_matched"] / a["entity_total"] * 100
                  if a["entity_total"] > 0 else 0.0)
            ra = (a["rel_satisfied"] / a["rel_total"] * 100
                  if a["rel_total"] > 0 else 0.0)
            return (f"n={a['n']:3d}  "
                    f"count={ca:5.1f}%  "
                    f"attrs={ea:5.1f}% ({a['entity_matched']}/{a['entity_total']})  "
                    f"rel={ra:5.1f}% ({a['rel_satisfied']}/{a['rel_total']})")

        accelerator.print(
            "\n[eval summary — per (sampler × cfg × schedule), full levels, "
            "seed 0]")
        for tag in args.samplers:
            for cfg_val in cfg_values:
                for sched in cfg_schedules:
                    cs = _cs_tag(cfg_val, sched)
                    a = per_sampler_cs.get((tag, cs))
                    label = f"{tag:22s} {cs:22s}"
                    if a is None:
                        accelerator.print(f"  {label} (no samples)")
                    else:
                        accelerator.print(f"  {label} {_fmt_agg(a)}")

        accelerator.print(
            "\n[eval summary — collapsed per sampler "
            "(averaged over swept cfg × schedule)]")
        for tag in args.samplers:
            accelerator.print(f"  {tag:22s} {_fmt_agg(per_sampler[tag])}")

        accelerator.print(
            f"\n[eval summary — per level (ablation sampler "
            f"= {args.decode_ablation_sampler}, "
            f"cfg={decode_ablation_cfg:g} sched={decode_ablation_schedule}, "
            f"{args.num_decode_seeds} seeds)]")
        for k in range(1, num_levels + 1):
            s = ordered_levels[k - 1]
            accelerator.print(
                f"  up to {s}×{s} (k={k})   {_fmt_agg(per_level[k])}")

        # Also dump the summary as JSON
        with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
            json.dump({
                "cfg_sweep": {
                    "cfg_values": cfg_values,
                    "cfg_schedules": cfg_schedules,
                    "decode_ablation_cfg": decode_ablation_cfg,
                    "decode_ablation_schedule": decode_ablation_schedule,
                },
                "per_sampler_cfg_schedule": {
                    f"{tag}/{cs}": a
                    for (tag, cs), a in per_sampler_cs.items()
                },
                "per_sampler_collapsed": per_sampler,
                "per_level_ablation": {f"up_to_{s}x{s}": per_level[i + 1]
                                       for i, s in enumerate(ordered_levels)},
            }, f, indent=2)

        accelerator.print(f"\n[done] outputs under {out_dir}")


if __name__ == "__main__":
    main()
