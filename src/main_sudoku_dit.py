"""
Sudoku DiT — Multi-Resolution Conditional DiT for MNIST-Sudoku
===============================================================

DiT backbone with ViT encoder for sudoku image generation.
Supports custom level_sizes (e.g. [9] or [9, 3, 1]) for sudoku grids.

Evaluation: rule_acc (sudoku constraint satisfaction) + cell_acc (vs GT).

Usage:
  # Single resolution (9×9 only)
  accelerate launch src/main_sudoku_dit.py --backbone dit --level_sizes 9 ...

  # Multi-resolution (9×9, 3×3, 1×1)
  accelerate launch src/main_sudoku_dit.py --backbone dit --level_sizes 9 3 1 ...
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf

# SRM dataset & evaluator
from SRM.datasets import get_dataset
from SRM.evaluation.sudoku_eval_only import MnistSudokuEvalOnly


# ──────────────────────────────────────────────────────────────────
#  EMA
# ──────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.data.copy_(mb.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # --- paths ---
    p.add_argument("--output_dir", type=str, default="runs/sudoku_dit")
    p.add_argument("--resume_dir", type=str, default=None)
    p.add_argument("--sudoku_config", type=str, required=True,
                   help="Path to sudoku dataset config JSON (e.g. config/sudoku_config.json)")
    p.add_argument("--classifier_pth", type=str, required=True,
                   help="Path to MNIST classifier .pth for sudoku evaluation")

    # --- image / model ---
    p.add_argument("--image_size", type=int, default=288,
                   help="Padded image size (e.g. 288 for 252→288 sudoku)")
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--cond_in_channels", type=int, default=1)

    # --- encoder ---
    p.add_argument("--level_sizes", type=int, nargs="+", default=[9],
                   help="Encoder level sizes (e.g. '9' for single, '9 3 1' for multi-res)")
    p.add_argument("--min_patch_size", type=int, default=32,
                   help="Fallback min_patch_size (ignored when level_sizes is set)")
    p.add_argument("--num_levels", type=int, default=None)
    p.add_argument("--feat_channels", type=int, default=256)
    p.add_argument("--depth_per_level", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cnn_base_channels", type=int, default=64)
    p.add_argument("--mae_mask_ratio", type=float, default=0.0)

    p.add_argument("--encoder_internal_dim", type=int, default=None,
                   help="Encoder internal dim. If set, encoder runs at this dim "
                        "and projects to feat_channels at output.")

    # --- ViT encoder ---
    p.add_argument("--encoder_type", type=str, default="vit",
                   choices=["cnn", "vit", "swin"])
    p.add_argument("--vit_patch_size", type=int, default=4)
    p.add_argument("--vit_depth", type=int, default=4)
    p.add_argument("--vit_num_heads", type=int, default=4)
    p.add_argument("--vit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--vit_use_cnn_stem", action="store_true", default=True)
    p.add_argument("--vit_no_cnn_stem", action="store_true", default=False)
    p.add_argument("--vit_cnn_stem_reduction", type=int, default=4)

    # --- discretization ---
    p.add_argument("--use_fsq", action="store_true", default=False)
    p.add_argument("--fsq_levels", type=int, nargs="+", default=None)
    p.add_argument("--fsq_drop_quant_p", type=float, default=0.0)
    p.add_argument("--fsq_corrupt_tokens_p", type=float, default=0.0)
    p.add_argument("--use_vq", action="store_true", default=False)
    p.add_argument("--vq_codebook_size", type=int, default=512)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--vq_loss_weight", type=float, default=1.0)

    # --- backbone ---
    p.add_argument("--backbone", type=str, default="dit",
                   choices=["dit"],
                   help="Denoising backbone (only dit supported)")

    # --- DiT ---
    p.add_argument("--dit_patch_size", type=int, default=16)
    p.add_argument("--dit_hidden_size", type=int, default=768)
    p.add_argument("--dit_n_heads", type=int, default=12)
    p.add_argument("--dit_n_blocks", type=int, default=12)
    p.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--dit_dropout", type=float, default=0.0)
    p.add_argument("--dit_bottleneck_dim", type=int, default=128)
    p.add_argument("--dit_in_context_len", type=int, default=0)
    p.add_argument("--dit_in_context_start", type=int, default=4)
    p.add_argument("--cond_noise_std", type=float, default=0.0,
                   help="Gaussian noise std added to encoder features before "
                        "DiT conditioning (train only). 0 disables.")
    p.add_argument("--cond_noise_relative", action="store_true", default=False,
                   help="If set, noise std is scaled by per-sample feature std "
                        "(relative perturbation). Otherwise absolute std.")
    p.add_argument("--cond_token_drop_prob", type=float, default=0.0,
                   help="Max per-token drop ratio on encoder cond tokens "
                        "(train only). Each sample gets p_b ~ U(0, this), then "
                        "Bernoulli(p_b) per position. Dropped positions use "
                        "the learned null embedding. Independent from "
                        "uncond_drop_prob. Inference is auto-disabled "
                        "(training=False) → matches p=0 end of the spectrum.")

    # --- level drop ---
    p.add_argument("--level_drop", action="store_true", default=False)
    p.add_argument("--no_level_drop", dest="level_drop", action="store_false")
    p.add_argument("--min_keep_levels", type=int, default=1)
    p.add_argument("--level_drop_after_steps", type=int, default=-1)
    p.add_argument("--eval_num_active_levels", type=int, default=None)

    # --- flow matching ---
    p.add_argument("--use_flow_matching", action="store_true", default=False)
    p.add_argument("--flow_P_mean", type=float, default=-0.8)
    p.add_argument("--flow_P_std", type=float, default=0.8)
    p.add_argument("--flow_t_eps", type=float, default=0.05)
    p.add_argument("--flow_noise_scale", type=float, default=1.0)
    p.add_argument("--flow_sampling_method", type=str, default="euler",
                   choices=["euler", "heun"])

    # --- diffusion (DDPM fallback) ---
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--beta_start", type=float, default=2e-5)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--prediction_type", type=str, default="sample")

    # --- training ---
    p.add_argument("--max_train_steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--blr", type=float, default=2.5e-5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr_schedule", type=str, default="constant",
                   choices=["cosine", "constant"],
                   help="LR schedule after warmup: cosine decay or constant")
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--uncond_drop_prob", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # --- eval ---
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=1500)
    p.add_argument("--eval_num_steps", type=int, default=50)
    p.add_argument("--eval_num_samples", type=int, default=81,
                   help="Number of samples for sudoku evaluation")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--sudoku_eval_grid_size", type=int, default=9)
    p.add_argument("--num_workers", type=int, default=4)

    # --- eval t-SNE ---
    p.add_argument("--eval_tsne", action="store_true", default=True,
                   help="Run t-SNE on encoder tokens during eval")
    p.add_argument("--no_eval_tsne", dest="eval_tsne", action="store_false")
    p.add_argument("--eval_tsne_max_samples", type=int, default=2000,
                   help="Max tokens used for t-SNE during eval")
    p.add_argument("--eval_tsne_perplexity", type=float, default=30.0)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Model builder
# ──────────────────────────────────────────────────────────────────

def build_model(args):
    from model_multires import MultiResConditionalDiT
    return MultiResConditionalDiT(
        image_size=args.image_size,
        in_channels=args.in_channels,
        cond_in_channels=args.cond_in_channels,
        vae_downsample_factor=1,  # pixel space
        min_patch_size=args.min_patch_size,
        num_levels=args.num_levels,
        feat_channels=args.feat_channels,
        dit_patch_size=args.dit_patch_size,
        dit_hidden_size=args.dit_hidden_size,
        dit_n_heads=args.dit_n_heads,
        dit_n_blocks=args.dit_n_blocks,
        dit_mlp_ratio=args.dit_mlp_ratio,
        dit_dropout=args.dit_dropout,
        dit_bottleneck_dim=args.dit_bottleneck_dim,
        dit_in_context_len=args.dit_in_context_len,
        dit_in_context_start=args.dit_in_context_start,
        uncond_drop_prob=args.uncond_drop_prob,
        level_drop=args.level_drop,
        min_keep_levels=args.min_keep_levels,
        depth_per_level=args.depth_per_level,
        mlp_ratio=args.mlp_ratio,
        cnn_base_channels=args.cnn_base_channels,
        level_drop_after_steps=args.level_drop_after_steps,
        cond_use_latent=False,
        mae_mask_ratio=args.mae_mask_ratio,
        encoder_type=args.encoder_type,
        vit_patch_size=args.vit_patch_size,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_ratio=args.vit_mlp_ratio,
        vit_use_cnn_stem=args.vit_use_cnn_stem and not args.vit_no_cnn_stem,
        vit_cnn_stem_reduction=args.vit_cnn_stem_reduction,
        encoder_internal_dim=args.encoder_internal_dim,
        use_fsq=args.use_fsq,
        fsq_levels=args.fsq_levels,
        fsq_drop_quant_p=args.fsq_drop_quant_p,
        fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
        use_vq=args.use_vq,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        level_sizes=args.level_sizes,
        cond_noise_std=args.cond_noise_std,
        cond_noise_relative=args.cond_noise_relative,
        cond_token_drop_prob=args.cond_token_drop_prob,
    )


# ──────────────────────────────────────────────────────────────────
#  LR scheduler (cosine with warmup)
# ──────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=1e-6, schedule="cosine"):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────
#  Flow matching ODE sampling
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow_ode(model, cond_images, num_steps=50,
                    guidance_scale=1.0, in_channels=1,
                    num_active_levels=None, method="euler",
                    noise_scale=1.0, t_eps=0.05):
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    z = noise_scale * torch.randn(B, in_channels, latent_size, latent_size,
                                  device=device, dtype=dtype)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def _compute_velocity(z_cur, t_scalar):
        t_batch = t_scalar.expand(B)
        t_expand = t_scalar.view(1, 1, 1, 1)

        if guidance_scale != 1.0:
            x_cond = model(z_cur, t_batch, cond_image=cond_images,
                           num_active_levels=num_active_levels)
            x_uncond = model(z_cur, t_batch, cond_image=cond_images,
                             return_uncond=True)
            v_cond = (x_cond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            v_uncond = (x_uncond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            x_pred = model(z_cur, t_batch, cond_image=cond_images,
                           num_active_levels=num_active_levels)
            return (x_pred - z_cur) / (1.0 - t_expand).clamp_min(t_eps)

    for i in range(num_steps):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur

        if method == "heun" and i < num_steps - 1:
            v1 = _compute_velocity(z, t_cur)
            z_euler = z + dt * v1
            v2 = _compute_velocity(z_euler, t_next)
            z = z + dt * 0.5 * (v1 + v2)
        else:
            v = _compute_velocity(z, t_cur)
            z = z + dt * v

    return z.clamp(-1, 1)


# ──────────────────────────────────────────────────────────────────
#  DDIM sampling
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_ddim(model, scheduler, cond_images, num_steps=50,
                guidance_scale=1.0, in_channels=1,
                num_active_levels=None):
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(B, in_channels, latent_size, latent_size,
                          device=device, dtype=dtype)

    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        if guidance_scale != 1.0:
            noise_cond = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)
            noise_uncond = model(latents, t_batch, cond_image=cond_images,
                                return_uncond=True)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents.clamp(-1, 1)


# ──────────────────────────────────────────────────────────────────
#  Grid visualization helpers (PIL)
# ──────────────────────────────────────────────────────────────────

def render_digit_grid(
    grid_9x9,
    wrong_mask_9x9=None,     # 빨간 테두리 (GT vs PRED mismatch)
    bg_mask_9x9=None,        # 연분홍 배경 (GT vs TOK2DIG mismatch)
    cell=34, pad=3, border=3, title=None,
    font_size=18,
):
    """Render a 9x9 digit grid as a PIL image with optional error highlighting."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    grid_np = grid_9x9.detach().cpu().numpy() if torch.is_tensor(grid_9x9) else grid_9x9
    if wrong_mask_9x9 is not None and torch.is_tensor(wrong_mask_9x9):
        wrong_mask_9x9 = wrong_mask_9x9.detach().cpu().numpy()
    if bg_mask_9x9 is not None and torch.is_tensor(bg_mask_9x9):
        bg_mask_9x9 = bg_mask_9x9.detach().cpu().numpy()

    H, W = grid_np.shape
    title_h = 18 if title else 0
    img_w = W * cell + 2 * pad
    img_h = H * cell + 2 * pad + title_h

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if title:
        draw.text((pad, 0), title, fill=(0, 0, 0), font=font)

    y0 = title_h
    pink = (255, 230, 235)

    for r in range(H):
        for c in range(W):
            x1 = pad + c * cell
            y1 = y0 + pad + r * cell
            x2 = x1 + cell
            y2 = y1 + cell

            # 연분홍 배경 (GT vs TOK2DIG mismatch)
            if bg_mask_9x9 is not None and bool(bg_mask_9x9[r, c]):
                draw.rectangle([x1, y1, x2, y2], fill=pink)

            # 기본 셀 테두리
            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)

            val = int(grid_np[r, c])
            s = str(val)
            bbox = draw.textbbox((0, 0), s, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x1 + (cell - tw) / 2
            ty = y1 + (cell - th) / 2
            draw.text((tx, ty), s, fill=(0, 0, 0), font=font)

            # 빨간 테두리 (GT vs PRED mismatch)
            if wrong_mask_9x9 is not None and bool(wrong_mask_9x9[r, c]):
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=border)

    return img


def tile_images(img_list, nrow, pad_px=6, bg=(255, 255, 255)):
    """Tile a list of PIL images into a grid."""
    from PIL import Image
    if not img_list:
        return None
    w, h = img_list[0].size
    ncol = nrow
    nrows = math.ceil(len(img_list) / ncol)
    out_w = ncol * w + (ncol + 1) * pad_px
    out_h = nrows * h + (nrows + 1) * pad_px
    canvas = Image.new("RGB", (out_w, out_h), bg)
    for i, im in enumerate(img_list):
        rr, cc = i // ncol, i % ncol
        canvas.paste(im, (pad_px + cc * (w + pad_px), pad_px + rr * (h + pad_px)))
    return canvas


# ──────────────────────────────────────────────────────────────────
#  Sudoku evaluation
# ──────────────────────────────────────────────────────────────────

_DIGIT_COLORS = {
    1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd',
    5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf',
}


@torch.no_grad()
def run_encoder_tsne_eval(eval_model, ref_images, ref_grids, args,
                          accelerator, global_step, tag=""):
    """Extract 9x9 encoder tokens (continuous pre-discretizer) and save a
    t-SNE PNG colored by GT digit class.  Main-process only."""
    if not accelerator.is_main_process:
        return
    if ref_grids is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except Exception as e:
        accelerator.print(f"[Eval][tSNE] skipped (import error): {e}")
        return

    device = accelerator.device
    level_features = eval_model.encoder.forward_injection(ref_images)
    feat_2d = None
    for s, f2d in level_features.items():
        if s == 9:
            feat_2d = f2d
            break
    if feat_2d is None:
        return

    B, D, H, W = feat_2d.shape  # expect H=W=9
    tokens = feat_2d.flatten(2).transpose(1, 2).reshape(-1, D)  # (B*81, D)
    digits = ref_grids.to(device).long().reshape(-1)            # (B*81,)

    mask = (digits >= 1) & (digits <= 9)
    tokens = tokens[mask].float().cpu().numpy()
    digits = digits[mask].cpu().numpy()

    if len(tokens) > args.eval_tsne_max_samples:
        rng = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(tokens), generator=rng)[:args.eval_tsne_max_samples].numpy()
        tokens = tokens[perm]
        digits = digits[perm]

    if len(tokens) < 10:
        accelerator.print(f"[Eval][tSNE] too few tokens ({len(tokens)}), skipping")
        return

    perp = min(args.eval_tsne_perplexity, max(5.0, (len(tokens) - 1) / 3.0))
    emb = TSNE(n_components=2, perplexity=perp,
               random_state=args.seed, max_iter=1000).fit_transform(tokens)

    out_dir = os.path.join(args.output_dir, "eval_tsne")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for d in sorted(_DIGIT_COLORS.keys()):
        m = digits == d
        if m.sum() == 0:
            continue
        ax.scatter(emb[m, 0], emb[m, 1], c=_DIGIT_COLORS[d],
                   s=8, alpha=0.5, label=f"{d} ({m.sum()})", rasterized=True)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, ncol=2, title="Digit", markerscale=2, loc='upper right')
    tag_label = f"[{tag}] " if tag else ""
    fig.suptitle(f"{tag_label}encoder tokens — step {global_step} "
                 f"(9x9, {D}D, n={len(tokens)})",
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    tag_suffix = f"_{tag}" if tag else ""
    out_path = os.path.join(out_dir, f"step_{global_step:07d}{tag_suffix}.png")
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    accelerator.print(f"[Eval][tSNE] Saved {out_path}")


@torch.no_grad()
def run_sudoku_eval(model, eval_dataset, noise_scheduler, args, accelerator,
                    global_step, sudoku_evaluator, ema_model=None, tag=""):
    """Generate samples and evaluate rule_acc / cell_acc."""
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device

    # Collect reference samples from dataset
    n_samples = min(args.eval_num_samples, len(eval_dataset))
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(eval_dataset), generator=rng)[:n_samples].tolist()

    ref_images = []
    ref_grids = []
    for idx in indices:
        sample = eval_dataset[idx]
        if isinstance(sample, dict):
            ref_images.append(sample["image"].unsqueeze(0))
            if "grid" in sample:
                ref_grids.append(sample["grid"].unsqueeze(0))
        else:
            ref_images.append(sample[0].unsqueeze(0))

    ref_images = torch.cat(ref_images, dim=0).to(device)  # (B, 1, H, W)
    ref_grids = torch.cat(ref_grids, dim=0) if ref_grids else None  # (B, 9, 9) or None

    # Generate samples
    cond_input = ref_images
    if args.use_flow_matching:
        samples = sample_flow_ode(
            eval_model, cond_input,
            num_steps=args.eval_num_steps,
            guidance_scale=args.guidance_scale,
            in_channels=args.in_channels,
            num_active_levels=args.eval_num_active_levels,
            method=args.flow_sampling_method,
            noise_scale=args.flow_noise_scale,
            t_eps=args.flow_t_eps,
        )
    else:
        eval_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        )
        samples = sample_ddim(
            eval_model, eval_scheduler, cond_input,
            num_steps=args.eval_num_steps,
            guidance_scale=args.guidance_scale,
            in_channels=args.in_channels,
            num_active_levels=args.eval_num_active_levels,
        )

    if not accelerator.is_main_process:
        model.train()
        return

    # Save visual grid: [GT | generated] pairs
    real_01 = (ref_images.clamp(-1, 1) + 1) * 0.5
    fake_01 = (samples.clamp(-1, 1) + 1) * 0.5

    n_show = min(16, n_samples)
    pair_imgs = []
    for i in range(n_show):
        pair = torch.cat([real_01[i], fake_01[i]], dim=2)  # (C, H, 2W)
        pair = F.pad(pair, (1, 1, 1, 1), value=1.0)
        pair_imgs.append(pair.unsqueeze(0))
    pair_imgs = torch.cat(pair_imgs, dim=0)
    grid = make_grid(pair_imgs, nrow=4, padding=2)

    tag_suffix = f"_{tag}" if tag else ""
    save_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(save_dir, exist_ok=True)
    save_image(grid, os.path.join(save_dir, f"step_{global_step:07d}{tag_suffix}.png"))

    # Sudoku evaluation
    fake_m11 = fake_01 * 2.0 - 1.0
    s_eval = sudoku_evaluator.eval_images(fake_m11)

    rule_acc = s_eval["accuracy"].item()
    dist_mean = s_eval["distance"].float().mean().item()

    # Cell accuracy vs GT grid
    cell_acc = None
    if ref_grids is not None:
        pred_grid = s_eval["discrete"].to(device).long()
        gt = ref_grids.to(device).long()
        wrong_mask = (pred_grid != gt)
        cell_acc = (~wrong_mask).float().mean().item()

    # Cell accuracy vs GT image (classifier on real images)
    real_m11 = real_01 * 2.0 - 1.0
    s_gtimg = sudoku_evaluator.eval_images(real_m11)
    gt_from_img = s_gtimg["discrete"].to(device).long()
    pred_grid = s_eval["discrete"].to(device).long()
    wrong_mask_img = (pred_grid != gt_from_img)
    cell_acc_img = (~wrong_mask_img).float().mean().item()

    # Log
    tag_label = f"[{tag}]" if tag else ""
    msg = (f"[Eval]{tag_label} step={global_step} rule_acc={rule_acc:.4f} "
           f"dist_mean={dist_mean:.2f} cell_acc_img={cell_acc_img:.4f}")
    if cell_acc is not None:
        msg += f" cell_acc_grid={cell_acc:.4f}"
    accelerator.print(msg)

    prefix = f"eval_{tag}" if tag else "eval"
    log_dict = {
        f"{prefix}/sudoku_rule_acc": rule_acc,
        f"{prefix}/sudoku_dist_mean": dist_mean,
        f"{prefix}/sudoku_cell_acc_img": cell_acc_img,
    }
    if cell_acc is not None:
        log_dict[f"{prefix}/sudoku_cell_acc_grid"] = cell_acc

    # ============================================================
    # Token-to-Digit mapping analysis + digit grid visualization
    # ============================================================
    if ref_grids is not None:
        pred_9 = s_eval["discrete"].to(device).long()   # (B, 9, 9) from classifier
        gt_9 = ref_grids.to(device).long()               # (B, 9, 9) ground truth
        B = gt_9.shape[0]
        wrong_mask_pred = (pred_9 != gt_9)  # (B, 9, 9)

        # --- Token-to-Digit analysis (FSQ / VQ only) ---
        tok_ids_2d = None
        pred_from_tok = None
        if eval_model.discretizer is not None:
            # Extract token IDs from encoder + discretizer
            level_features = eval_model.encoder.forward_injection(ref_images)
            for s, feat_2d in level_features.items():
                D = feat_2d.shape[1]
                tokens_flat = feat_2d.flatten(2).transpose(1, 2)  # (B, S*S, D)
                _, t_ids = eval_model.discretizer(tokens_flat)     # (B, S*S)
                if s == 9:  # 9x9 sudoku level
                    tok_ids_2d = t_ids.view(B, 9, 9).long()
                    break

            if tok_ids_2d is not None:
                # tok_id -> digit 통계 매핑 (batch 전체)
                vocab = int(tok_ids_2d.max().item()) + 1
                tid_flat = tok_ids_2d.reshape(-1)
                gt_flat = gt_9.reshape(-1).clamp(0, 9)

                idx = (tid_flat * 10 + gt_flat).to(torch.long)
                counts = torch.bincount(idx, minlength=vocab * 10).view(vocab, 10)

                tok2digit = counts.argmax(dim=1)
                tok_conf = counts.max(dim=1).values.float() / (counts.sum(dim=1).float() + 1e-9)

                pred_from_tok = tok2digit[tok_ids_2d]  # (B, 9, 9)

                # Accuracy metrics
                wrong_tok_vs_gt = (pred_from_tok != gt_9)
                acc_tok_vs_gt = (~wrong_tok_vs_gt).float().mean().item()
                wrong_tok_vs_pred = (pred_from_tok != pred_9)
                acc_tok_vs_pred = (~wrong_tok_vs_pred).float().mean().item()

                accelerator.print(
                    f"[Eval][Tok2Digit] vocab={vocab} "
                    f"acc(tok->digit vs GT)={acc_tok_vs_gt:.4f} "
                    f"acc(tok->digit vs PRED)={acc_tok_vs_pred:.4f}"
                )

                log_dict[f"{prefix}/tok2digit_acc_vs_gt"] = acc_tok_vs_gt
                log_dict[f"{prefix}/tok2digit_acc_vs_pred"] = acc_tok_vs_pred

        # --- Confusion matrix (always: GT vs PRED; optionally: GT vs TOK2DIG) ---
        def confusion_10x10(gt_grid, pr_grid):
            g = gt_grid.reshape(-1).clamp(0, 9).to(torch.long)
            p = pr_grid.reshape(-1).clamp(0, 9).to(torch.long)
            return torch.bincount(g * 10 + p, minlength=100).view(10, 10)

        def print_full_confusion(cm, name):
            accelerator.print(f"[Eval][ConfusionFull] {name} (rows=GT, cols=Pred)")
            for gt_d in range(10):
                row = cm[gt_d]
                total = int(row.sum().item())
                if total == 0:
                    continue
                parts = [f"{pr_d}:{int(row[pr_d].item())}"
                         for pr_d in range(10) if int(row[pr_d].item()) > 0]
                accelerator.print(f"  GT {gt_d} (n={total}) -> " + ", ".join(parts))

        cm_pred = confusion_10x10(gt_9, pred_9)
        print_full_confusion(cm_pred, "GT->PRED")

        if pred_from_tok is not None:
            cm_tok = confusion_10x10(gt_9, pred_from_tok)
            print_full_confusion(cm_tok, "GT->TOK2DIG")

        # --- Digit grid rendering (PIL) — GT / PRED (+ TOK2DIG if available) ---
        wrong_mask_tok = (pred_from_tok != gt_9) if pred_from_tok is not None else wrong_mask_pred
        gt_imgs, pred_imgs = [], []
        tok_imgs = [] if pred_from_tok is not None else None

        n_viz = min(B, 16)
        for i in range(n_viz):
            wm_red = wrong_mask_pred[i]
            wm_pink = wrong_mask_tok[i]

            gt_imgs.append(render_digit_grid(
                gt_9[i], wrong_mask_9x9=wm_red, bg_mask_9x9=wm_pink))
            pred_imgs.append(render_digit_grid(
                pred_9[i], wrong_mask_9x9=wm_red, bg_mask_9x9=wm_pink))
            if tok_imgs is not None:
                tok_imgs.append(render_digit_grid(
                    pred_from_tok[i], wrong_mask_9x9=wm_red, bg_mask_9x9=wm_pink))

        nrow = min(n_viz, 9)
        gt_canvas = tile_images(gt_imgs, nrow=nrow)
        pr_canvas = tile_images(pred_imgs, nrow=nrow)

        viz_dir = os.path.join(save_dir, "grid_digits")
        os.makedirs(viz_dir, exist_ok=True)

        gt_canvas.save(os.path.join(viz_dir, f"step_{global_step:07d}{tag_suffix}_GT.png"))
        pr_canvas.save(os.path.join(viz_dir, f"step_{global_step:07d}{tag_suffix}_PRED.png"))

        if tok_imgs is not None:
            tok_canvas = tile_images(tok_imgs, nrow=nrow)
            tok_canvas.save(os.path.join(viz_dir, f"step_{global_step:07d}{tag_suffix}_TOK2DIG.png"))

        accelerator.print(
            f"[Eval] Saved digit grids to {viz_dir}/step_{global_step:07d}_*.png")

    accelerator.log(log_dict, step=global_step)

    # ── Encoder token t-SNE ──
    if getattr(args, "eval_tsne", False):
        run_encoder_tsne_eval(
            eval_model, ref_images, ref_grids, args,
            accelerator, global_step, tag=tag,
        )

    model.train()


# ──────────────────────────────────────────────────────────────────
#  Checkpoint save / load
# ──────────────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizer, step, args, ema=None):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "checkpoints", f"step_{step:07d}")
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        ckpt = {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
        }
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))
        accelerator.print(f"Saved checkpoint at step {step}")


def load_checkpoint(accelerator, model, optimizer, args, ema=None):
    resume = args.resume_dir or args.output_dir
    ckpt_dir = os.path.join(resume, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return 0

    steps = []
    for d in os.listdir(ckpt_dir):
        if d.startswith("step_"):
            try:
                steps.append(int(d.split("_")[1]))
            except ValueError:
                pass
    if not steps:
        return 0

    latest = max(steps)
    path = os.path.join(ckpt_dir, f"step_{latest:07d}", "checkpoint.pt")
    if not os.path.isfile(path):
        return 0

    accelerator.print(f"Resuming from step {latest}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
        accelerator.print("Loaded EMA state")
    return ckpt.get("step", latest)


# ──────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────

def train(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator.init_trackers("sudoku_dit")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed + accelerator.process_index)

    # ── LR scaling ──
    effective_bs = args.batch_size * args.grad_accum_steps * accelerator.num_processes
    if args.lr is not None:
        lr = args.lr
    else:
        lr = args.blr * effective_bs / 256
    accelerator.print(f"Effective batch size: {effective_bs}, LR: {lr:.2e}")

    # ── Dataset (SRM sudoku) ──
    cfg = OmegaConf.load(args.sudoku_config)
    srm_ds_cfg = cfg.SRM_dataset_cfg
    srm_cond_cfg = cfg.SRM_conditioning_cfg
    train_ds = get_dataset(srm_ds_cfg, srm_cond_cfg, "train")
    val_ds = get_dataset(srm_ds_cfg, srm_cond_cfg, "val")
    accelerator.print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    sudoku_evaluator = MnistSudokuEvalOnly(
        mnist_classifier_path=args.classifier_pth,
        grid_size=(args.sudoku_eval_grid_size, args.sudoku_eval_grid_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ──
    model = build_model(args)
    if accelerator.is_main_process:
        accelerator.print(model.describe())
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_encoder = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        n_backbone = n_total - n_encoder
        accelerator.print(
            f"Parameters: {n_total:.1f}M "
            f"(encoder: {n_encoder:.1f}M, dit: {n_backbone:.1f}M)"
        )

    # ── Optimizer ──
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.95))

    # ── Noise scheduler ──
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    # ── Prepare ──
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ── EMA ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(model), decay=args.ema_decay)
        accelerator.print(f"EMA: decay={args.ema_decay}")

    # ── Resume ──
    global_step = load_checkpoint(accelerator, model, optimizer, args, ema=ema)

    # ── Train ──
    accelerator.print(f"Starting training from step {global_step}")
    model.train()
    epoch = 0
    t_start = time.time()

    pbar = tqdm(
        initial=global_step, total=args.max_train_steps,
        desc="Training", dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    while global_step < args.max_train_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            # Unpack SRM sudoku batch
            if isinstance(batch, dict):
                images = batch["image"]    # (B, 1, H, W)
            else:
                images, _ = batch

            x0 = images

            # LR schedule
            cur_lr = get_lr(global_step, args.warmup_steps, args.max_train_steps, lr, schedule=args.lr_schedule)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Update step for level drop schedule
            accelerator.unwrap_model(model).set_step(global_step)

            with accelerator.accumulate(model):
                cond_images = images

                if args.use_flow_matching:
                    # Flow matching (V-loss)
                    z_t = (torch.randn(x0.shape[0], device=x0.device)
                           * args.flow_P_std + args.flow_P_mean)
                    t_flow = torch.sigmoid(z_t)
                    t_expand = t_flow.view(-1, 1, 1, 1)

                    e = torch.randn_like(x0) * args.flow_noise_scale
                    noisy = t_expand * x0 + (1 - t_expand) * e

                    with accelerator.autocast():
                        use_aux = args.use_vq
                        if use_aux:
                            x_pred, aux = model(
                                noisy, t_flow, cond_image=cond_images,
                                return_aux_loss=True)
                        else:
                            x_pred = model(noisy, t_flow,
                                           cond_image=cond_images)
                            aux = {}

                        # JiT-style velocity loss
                        v_target = (x0 - noisy) / (1 - t_expand).clamp_min(args.flow_t_eps)
                        v_pred = (x_pred - noisy) / (1 - t_expand).clamp_min(args.flow_t_eps)
                        loss = F.mse_loss(v_pred, v_target)

                        if "vq_loss" in aux:
                            loss = loss + args.vq_loss_weight * aux["vq_loss"]
                else:
                    # Standard DDPM training
                    noise = torch.randn_like(x0)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (x0.shape[0],), device=x0.device, dtype=torch.long,
                    )
                    noisy = noise_scheduler.add_noise(x0, noise, timesteps)

                    with accelerator.autocast():
                        use_aux = args.use_vq
                        if use_aux:
                            pred, aux = model(noisy, timesteps,
                                              cond_image=cond_images,
                                              return_aux_loss=True)
                        else:
                            pred = model(noisy, timesteps,
                                         cond_image=cond_images)
                            aux = {}

                        if args.prediction_type == "epsilon":
                            target = noise
                        elif args.prediction_type == "sample":
                            target = x0
                        else:
                            target = noise_scheduler.get_velocity(
                                x0, noise, timesteps)

                        loss = F.mse_loss(pred, target)

                        if "vq_loss" in aux:
                            loss = loss + args.vq_loss_weight * aux["vq_loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            # EMA update
            if ema is not None:
                ema.update(accelerator.unwrap_model(model))

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.1e}", refresh=False)

            # ── Logging ──
            if global_step % args.log_every == 0:
                log_dict = {"loss": loss.item(), "lr": cur_lr}
                if "vq_loss" in aux:
                    log_dict["vq_loss"] = aux["vq_loss"].item()
                accelerator.log(log_dict, step=global_step)

            # ── Eval (sudoku rule_acc / cell_acc) ──
            if global_step % args.eval_every == 0:
                # Evaluate original model
                run_sudoku_eval(
                    model, val_ds, noise_scheduler, args,
                    accelerator, global_step, sudoku_evaluator,
                    ema_model=None, tag="online",
                )
                # Evaluate EMA model (if enabled)
                if ema is not None:
                    run_sudoku_eval(
                        model, val_ds, noise_scheduler, args,
                        accelerator, global_step, sudoku_evaluator,
                        ema_model=ema.shadow, tag="ema",
                    )

            # ── Save ──
            if global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, optimizer, global_step, args,
                                ema=ema)

    pbar.close()
    save_checkpoint(accelerator, model, optimizer, global_step, args, ema=ema)
    accelerator.print("Training complete.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
