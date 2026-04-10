#!/usr/bin/env python
"""
train_discrete_diffusion.py
===========================
Sudoku grid discrete (absorbing-state) diffusion training script.
Follows MDLM (Masked Diffusion Language Model) exactly:
  - subs parameterization  (continuous-time ELBO)
  - loglinear noise schedule
  - antithetic time sampling
  - DDPM-cache reverse sampler
  - optional EMA

Loads SRM sudoku dataset via --sudoku_config.
Uses batch["grid"] (B, 9, 9) int64  →  flattened to (B, 81) token sequence.

Usage (single GPU):
    python train_discrete_diffusion.py \\
        --sudoku_config ../config/sudoku_config.json \\
        --output_dir ./outputs/discrete_diff

Usage (multi-GPU with accelerate):
    accelerate launch train_discrete_diffusion.py \\
        --sudoku_config ../config/sudoku_config.json \\
        --output_dir ./outputs/discrete_diff
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, tqdm
from omegaconf import OmegaConf

# SRM dataset  (same as main.py)
from SRM.datasets import DatasetCfg, get_dataset, get_dataset_class
from SRM.type_extensions import ConditioningCfg

from dit_model import DIT
from discrete_diffusion import DiscreteDiffusion
from noise_schedule import get_noise

# image_cond_mode imports (lazy; only used when --image_cond_mode)
_ConditionalUNet = None
_DDIMScheduler = None
_SudokuEval = None


def _lazy_image_cond_imports():
    global _ConditionalUNet, _DDIMScheduler, _SudokuEval
    if _ConditionalUNet is None:
        from model import ConditionalUNet as _CU
        from diffusers import DDIMScheduler as _DS
        from SRM.evaluation.sudoku_eval_only import MnistSudokuEvalOnly as _SE
        _ConditionalUNet = _CU
        _DDIMScheduler = _DS
        _SudokuEval = _SE


# ────────────────────────────────────────────────────────────
#  EMA  (same logic as MDLM's models/ema.py)
# ────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in parameters]
        self.backup = []

    @torch.no_grad()
    def update(self, parameters):
        for s, p in zip(self.shadow, parameters):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def store(self, parameters):
        self.backup = [p.clone() for p in parameters]

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s)

    def restore(self, parameters):
        for b, p in zip(self.backup, parameters):
            p.data.copy_(b)
        self.backup = []

    def to(self, device):
        self.shadow = [s.to(device) for s in self.shadow]
        return self


# ────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────

def parse_step_from_dir(path: str) -> int:
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("step"):
        try:
            return int(base.replace("step", ""))
        except Exception:
            pass
    return 0


def count_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_n(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)


# ────────────────────────────────────────────────────────────
#  Grid visualization helpers (PIL)
# ────────────────────────────────────────────────────────────

def render_digit_grid(grid_2d, wrong_mask=None, cell=34, pad=3, border=3, font_size=18):
    """Render a 2D digit grid (H,W) as a PIL image."""
    from PIL import Image, ImageDraw, ImageFont

    grid_np = grid_2d.detach().cpu().numpy() if torch.is_tensor(grid_2d) else grid_2d
    H, W = grid_np.shape
    img_w = W * cell + 2 * pad
    img_h = H * cell + 2 * pad
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    if wrong_mask is not None and torch.is_tensor(wrong_mask):
        wrong_mask = wrong_mask.detach().cpu().numpy()
    for r in range(H):
        for c in range(W):
            x1, y1 = pad + c * cell, pad + r * cell
            x2, y2 = x1 + cell, y1 + cell
            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)
            s = str(int(grid_np[r, c]))
            bbox = draw.textbbox((0, 0), s, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2), s, fill=(0, 0, 0), font=font)
            if wrong_mask is not None and bool(wrong_mask[r, c]):
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=border)
    return img


def tile_images(img_list, nrow, pad_px=6, bg=(255, 255, 255)):
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


def _find_rule_violations(grid_np, grid_hw: int = 9):
    """Find cells that participate in a sudoku rule violation.

    Returns a (grid_hw, grid_hw) bool numpy array: True = violating cell.
    A cell is 'violating' if its value appears more than once in any of
    its row / column / 3×3 box.
    """
    import numpy as np
    violations = np.zeros((grid_hw, grid_hw), dtype=bool)
    box_h = box_w = 3

    for r in range(grid_hw):
        row = grid_np[r]
        for v in range(grid_hw):
            positions = [c for c in range(grid_hw) if row[c] == v]
            if len(positions) > 1:
                for c in positions:
                    violations[r, c] = True

    for c in range(grid_hw):
        col = grid_np[:, c]
        for v in range(grid_hw):
            positions = [r for r in range(grid_hw) if col[r] == v]
            if len(positions) > 1:
                for r in positions:
                    violations[r, c] = True

    for br in range(grid_hw // box_h):
        for bc in range(grid_hw // box_w):
            box = grid_np[br*box_h:(br+1)*box_h, bc*box_w:(bc+1)*box_w]
            for v in range(grid_hw):
                positions = []
                for dr in range(box_h):
                    for dc in range(box_w):
                        if box[dr, dc] == v:
                            positions.append((br*box_h+dr, bc*box_w+dc))
                if len(positions) > 1:
                    for (rr, cc) in positions:
                        violations[rr, cc] = True

    return violations


def render_sampling_gif(
    history: list,
    sample_idx: int,
    grid_hw: int,
    mask_index: int,
    save_path: str,
    confidence_history: list | None = None,
    pred_history: list | None = None,
    known_mask: torch.Tensor | None = None,
    max_frames: int = 60,
    frame_duration_ms: int = 600,
    cell: int = 34,
    pad: int = 3,
    font_size: int = 18,
    save_format: str = "gif",
):
    """Render a GIF/MP4 showing step-by-step unmasking for one sample.

    Each frame shows the 9x9 grid at that denoising step.
    Masked cells show predictions with confidence-based coloring.
    Newly unmasked cells are highlighted with blue border.

    Args:
        history:   list of (B, L) tensors, one per denoising step
        sample_idx: which sample in the batch to visualize
        grid_hw:   grid height/width (9)
        mask_index: the [MASK] token index
        confidence_history: optional list of (B, L) confidence maps
        pred_history: optional list of (B, L) predicted tokens
        known_mask: optional (B, L) bool for given cells
        save_path: output path (extension overridden by save_format)
        max_frames: subsample history if longer than this
        frame_duration_ms: milliseconds per frame
        save_format: "gif" or "mp4"
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    n_total = len(history)
    # subsample frames if too many
    if n_total > max_frames:
        # always include first and last
        indices = [0] + list(
            range(1, n_total - 1,
                  max(1, (n_total - 2) // (max_frames - 2)))
        ) + [n_total - 1]
        # deduplicate & sort
        indices = sorted(set(indices))
    else:
        indices = list(range(n_total))

    img_w = grid_hw * cell + 2 * pad
    img_h = grid_hw * cell + 2 * pad + 24  # extra space for step label
    frames = []

    prev_grid = None
    given_mask = None
    if known_mask is not None:
        given_mask = known_mask[sample_idx].view(grid_hw, grid_hw).cpu().numpy()

    # pre-compute rule violations on the final grid for the last frame
    import numpy as np
    final_seq = history[-1][sample_idx]
    final_grid_np = final_seq.view(grid_hw, grid_hw).numpy()
    is_complete = int((final_seq != mask_index).sum().item()) == grid_hw * grid_hw
    violations = _find_rule_violations(final_grid_np, grid_hw) if is_complete else None

    for frame_i, hist_idx in enumerate(indices):
        seq = history[hist_idx][sample_idx]  # (L,)
        grid = seq.view(grid_hw, grid_hw).numpy()
        is_last_frame = (hist_idx == indices[-1])
        if confidence_history is not None and hist_idx < len(confidence_history):
            conf = confidence_history[hist_idx][sample_idx].view(grid_hw, grid_hw).numpy()
        else:
            conf = None
        if pred_history is not None and hist_idx < len(pred_history):
            pred = pred_history[hist_idx][sample_idx].view(grid_hw, grid_hw).numpy()
        else:
            pred = None

        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # step label
        label = f"step {hist_idx}/{n_total - 1}"
        draw.text((pad, img_h - 20), label, fill=(100, 100, 100), font=font)

        for r in range(grid_hw):
            for c in range(grid_hw):
                x1, y1 = pad + c * cell, pad + r * cell
                x2, y2 = x1 + cell, y1 + cell
                val = int(grid[r, c])

                if val == mask_index:
                    # masked cell: show prediction with confidence color
                    # confidence range: 1/9 (uniform) ~ 1.0 (certain)
                    # normalize so uniform=0, certain=1
                    if conf is not None:
                        cval = float(conf[r, c])
                        cval = max(0.0, min(1.0, cval))
                        # normalize: 1/9=0.111 → 0, 1.0 → 1
                        norm_conf = max(0.0, (cval - 0.111) / 0.889)
                        # magma-like colormap: yellow (low) → orange → red → dark (high)
                        # Low confidence (uncertain): light yellow (255, 255, 200)
                        # High confidence (certain): dark red (139, 0, 0)
                        r_col = int(255 - 116 * norm_conf)  # 255 → 139
                        g_col = int(255 - 255 * norm_conf)  # 255 → 0
                        b_col = int(200 - 200 * norm_conf)  # 200 → 0
                        fill = (r_col, g_col, b_col)
                    else:
                        norm_conf = 0.0
                        fill = (255, 255, 200)  # light yellow (uncertain)
                    draw.rectangle([x1, y1, x2, y2],
                                   fill=fill,
                                   outline=(180, 180, 180), width=1)
                    # show predicted digit (internal 0-8 → display 1-9)
                    if pred is not None:
                        pred_val = int(pred[r, c]) + 1  # +1 for display
                        s = str(pred_val)
                        # text: white on dark, black on light
                        if norm_conf > 0.5:
                            text_color = (255, 255, 255)
                        else:
                            text_color = (0, 0, 0)
                    else:
                        s = "·"
                        text_color = (100, 100, 100)
                    bbox = draw.textbbox((0, 0), s, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2),
                              s, fill=text_color, font=font)
                else:
                    # check if newly unmasked this step
                    newly_unmasked = False
                    if prev_grid is not None:
                        if int(prev_grid[r, c]) == mask_index:
                            newly_unmasked = True

                    is_given = False
                    if given_mask is not None and bool(given_mask[r, c]):
                        is_given = True

                    # check rule violation on final frame
                    is_violation = (is_last_frame and violations is not None
                                    and bool(violations[r, c]))

                    fill = (255, 255, 255)

                    if is_violation:
                        # rule-violating cell → red
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(255, 230, 230),
                                       outline=(220, 0, 0), width=2)
                        text_color = (200, 0, 0)
                    elif newly_unmasked:
                        # blue highlight for newly placed digit
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=fill,
                                       outline=(0, 100, 255), width=2)
                        text_color = (0, 80, 200)
                    elif is_given:
                        # given (hint) cell → green
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(230, 255, 230),
                                       outline=(100, 180, 100), width=1)
                        text_color = (0, 100, 0)
                    else:
                        # already placed (previously unmasked)
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=fill,
                                       outline=(200, 200, 200), width=1)
                        text_color = (0, 0, 0)

                    s = str(val + 1)  # internal 0-8 → display 1-9
                    bbox = draw.textbbox((0, 0), s, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2),
                              s, fill=text_color, font=font)

        frames.append(img)
        prev_grid = grid.copy()

    if frames:
        # ensure correct extension
        base, _ = os.path.splitext(save_path)
        if save_format == "mp4":
            save_path = base + ".mp4"
            import numpy as np
            import imageio
            fps = max(1, 1000 // frame_duration_ms)
            # convert PIL images to numpy arrays, pad to 16-multiple for codec
            np_frames = [np.array(f) for f in frames]
            h, w = np_frames[0].shape[:2]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h or pad_w:
                np_frames = [
                    np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                           mode='constant', constant_values=255)
                    for f in np_frames
                ]
            # repeat final frame to hold it longer
            n_hold = max(1, fps * 3)  # hold ~3 seconds
            np_frames.extend([np_frames[-1]] * n_hold)
            try:
                imageio.mimwrite(save_path, np_frames, format="FFMPEG",
                                 fps=fps, codec="libx264",
                                 pixelformat="yuv420p")
            except (ImportError, OSError):
                # FFMPEG plugin not available — fall back to GIF
                save_path = base + ".gif"
                save_format = "gif"   # drop through to gif branch below
        if save_format != "mp4":
            save_path = base + ".gif"
            # hold final frame longer
            durations = [frame_duration_ms] * len(frames)
            durations[-1] = frame_duration_ms * 5
            frames[0].save(
                save_path, save_all=True, append_images=frames[1:],
                duration=durations, loop=0,
            )


# ────────────────────────────────────────────────────────────
#  GridOnlyDataset: lightweight wrapper that skips image loading
# ────────────────────────────────────────────────────────────

class GridOnlyDataset(torch.utils.data.Dataset):
    """Wraps an SRM dataset and returns only the sudoku grid,
    skipping the expensive MNIST image assembly + transforms."""

    def __init__(self, inner_dataset):
        self.grids = inner_dataset.sudoku_grids  # (N, 9, 9) tensor

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return {"grid": self.grids[idx].long()}


# ────────────────────────────────────────────────────────────
#  Args
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sudoku Grid Discrete Diffusion (MDLM-style)")

    # basic
    p.add_argument("--output_dir", type=str, default="./outputs/discrete_diffusion")
    p.add_argument("--seed", type=int, default=42)

    # SRM sudoku dataset config  (same format as main.py --sudoku_config)
    p.add_argument("--sudoku_config", type=str, required=True,
                   help="Path to sudoku dataset config (JSON/YAML with "
                        "SRM_dataset_cfg / SRM_conditioning_cfg).")
    p.add_argument("--grid_only", action="store_true", default=False,
                   help="Skip loading images from SRM dataset; only load "
                        "sudoku grids.  Much faster when image is not needed.")
    p.add_argument("--conditional_training", action="store_true", default=False,
                   help="Train with random hint conditioning (inpainting-style). "
                        "Each sample gets a random number of hint cells [0, seq_len-1].")
    p.add_argument("--cond_hint_min", type=int, default=0,
                   help="Minimum number of hint cells during conditional training.")
    p.add_argument("--cond_hint_max", type=int, default=None,
                   help="Maximum number of hint cells (default: seq_len - 1).")

    # grid params
    p.add_argument("--grid_hw", type=int, default=9,
                   help="Grid height & width (9 for standard sudoku).")
    p.add_argument("--grid_vocab_size", type=int, default=9,
                   help="Number of distinct digit values in grid (0-8 → 9, representing 1-9).")
    p.add_argument("--pos_emb_type", type=str, default="2d",
                   choices=["1d", "2d", "sudoku"],
                   help="Positional embedding type: '1d' (learned only), "
                        "'2d' (+ row/col), 'sudoku' (+ row/col/box).")

    # DiT backbone
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--cond_dim", type=int, default=128)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--model_dropout", type=float, default=0.1)

    # TokenBridge-style factorized AR head
    p.add_argument("--factorized_head", action="store_true", default=False,
                   help="Use factorized per-dim AR head instead of flat softmax. "
                        "FSQ levels are auto-detected from pretrained encoder.")
    p.add_argument("--ar_head_dim", type=int, default=256,
                   help="Internal dimension of the factorized AR head.")
    p.add_argument("--ar_head_layers", type=int, default=2,
                   help="Number of layers in the factorized AR head.")

    # noise schedule
    p.add_argument("--noise_type", type=str, default="loglinear",
                   choices=["loglinear", "cosine"])
    p.add_argument("--noise_eps", type=float, default=1e-3)

    # diffusion training  (MDLM defaults)
    p.add_argument("--antithetic_sampling", action="store_true", default=True)
    p.add_argument("--importance_sampling", action="store_true", default=False)
    p.add_argument("--change_of_variables", action="store_true", default=False)
    p.add_argument("--sampling_eps", type=float, default=1e-3)

    # training
    p.add_argument("--max_train_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA decay rate (0 to disable).")

    # logging / eval / save
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--eval_num_samples", type=int, default=64)
    p.add_argument("--eval_num_steps", type=int, default=128)
    p.add_argument("--eval_gif_samples", type=int, default=4,
                   help="Number of samples to render as step-by-step GIFs "
                        "during eval (0 to disable). GIFs show the "
                        "unmasking process frame by frame.")
    p.add_argument("--eval_save_format", type=str, default="gif",
                   choices=["gif", "mp4"],
                   help="Save step-by-step sampling visualizations as "
                        "GIF or MP4 video.")
    p.add_argument("--sampler", type=str, default="ddpm_cache",
                   choices=["ddpm", "ddpm_cache", "confidence"],
                   help="Sampling method: ddpm, ddpm_cache (MDLM default), "
                        "or confidence (MaskGIT-style top-k unmasking).")
    p.add_argument("--tokens_per_step", type=int, default=0,
                   help="For confidence sampler: unmask exactly this many "
                        "tokens per step (linear schedule). 0 = cosine schedule. "
                        "Use 1 for step-by-step unmasking with num_steps=81.")

    # resume
    p.add_argument("--resume_dir", type=str, default=None)

    # ── image_cond_mode: learn discrete image tokens conditioned on grid ──
    p.add_argument("--image_cond_mode", action="store_true", default=False,
                   help="When set, train discrete diffusion on image-derived "
                        "discrete tokens (from encoder+discretizer) conditioned "
                        "on the sudoku grid.  Requires --cond_unet_ckpt.")
    p.add_argument("--cond_unet_ckpt", type=str, default=None,
                   help="Path to accelerate checkpoint dir that contains a "
                        "pretrained ConditionalUNet (with encoder+discretizer). "
                        "Only used when --image_cond_mode is set.")
    p.add_argument("--cond_unet_config", type=str, default=None,
                   help="Path to UNet config JSON (needed to rebuild "
                        "ConditionalUNet architecture for loading weights).")
    p.add_argument("--cond_image_size", type=int, default=32,
                   help="Image size that the pretrained encoder expects.")
    p.add_argument("--cond_feat_channels", type=int, default=128,
                   help="feat_channels of the pretrained encoder.")
    p.add_argument("--cond_discretizer_type", type=str, default="fsq",
                   choices=["fsq", "vq"],
                   help="Which discretizer the pretrained model uses.")
    p.add_argument("--cond_fsq_levels", type=int, nargs="+", default=[8,8,8,5],
                   help="FSQ levels (only used with --cond_discretizer_type fsq).")
    p.add_argument("--cond_vq_codebook_size", type=int, default=9,
                   help="VQ codebook size (only with --cond_discretizer_type vq).")
    p.add_argument("--cond_concat_downsample_factor", type=int, default=16,
                   help="Downsample factor of the pretrained encoder.")
    p.add_argument("--cond_patch_conditioning", action="store_true", default=False,
                   help="If the pretrained encoder is patch-based.")
    p.add_argument("--cond_patch_grid_size", type=int, default=9)
    p.add_argument("--cond_eval_ddim_steps", type=int, default=50,
                   help="Number of DDIM steps for rendering images during "
                        "image_cond_mode eval.")
    p.add_argument("--eval_render_batch_size", type=int, default=16,
                   help="Mini-batch size for DDIM image rendering during "
                        "eval.  Reduce if VRAM is tight.")
    p.add_argument("--init_embed_from_fsq", action="store_true", default=False,
                   help="Initialize DiT token_emb weights from the FSQ "
                        "codebook vectors (projected to hidden_size). "
                        "Gives the model structural knowledge about token "
                        "similarity from the start.  Only used with "
                        "--image_cond_mode.")
    p.add_argument("--token_cache_dir", type=str, default=None,
                   help="Directory to save/load cached image-encoder tok_ids. "
                        "If set, tok_ids are saved to disk on first run and "
                        "loaded from disk on subsequent runs, skipping the "
                        "encoder entirely.")

    # accelerate
    p.add_argument("--mixed_precision", type=str, default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--log_with", type=str, default=None,
                   help='Accelerate tracker: "tensorboard" or "wandb".')

    return p.parse_args()


# ────────────────────────────────────────────────────────────
#  LR scheduler (linear warmup → constant, same as MDLM)
# ────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
#  Sudoku rule checker
# ────────────────────────────────────────────────────────────

def _check_sudoku_single(g):
    """Check if a single (9,9) sudoku grid is valid. Returns bool."""
    g = g.cpu() if torch.is_tensor(g) else g
    for r in range(9):
        if len(set(g[r].tolist())) != 9:
            return False
    for c in range(9):
        if len(set(g[:, c].tolist())) != 9:
            return False
    for br in range(3):
        for bc in range(3):
            box = g[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten().tolist()
            if len(set(box)) != 9:
                return False
    return True


def check_sudoku_rules(grids: torch.Tensor):
    """Check sudoku validity for a batch of (B, 9, 9) grids.

    Each row, column, and 3×3 box must contain exactly 9 unique values
    (no duplicates).  Works regardless of whether the digit set is
    {1..9} or {0..8}.

    Returns:
        (num_valid, num_row_valid, num_col_valid, num_box_valid, total)
    """
    B = grids.shape[0]
    valid = 0
    row_valid_cnt = 0
    col_valid_cnt = 0
    box_valid_cnt = 0

    for b in range(B):
        g = grids[b].cpu()
        ok_row = True
        ok_col = True
        ok_box = True

        # ── row check ──
        for r in range(9):
            if len(set(g[r].tolist())) != 9:
                ok_row = False; break
        if ok_row:
            row_valid_cnt += 1

        # ── col check ──
        for c in range(9):
            if len(set(g[:, c].tolist())) != 9:
                ok_col = False; break
        if ok_col:
            col_valid_cnt += 1

        # ── box (3×3) check ──
        for br in range(3):
            for bc in range(3):
                box = g[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten().tolist()
                if len(set(box)) != 9:
                    ok_box = False; break
            if not ok_box:
                break
        if ok_box:
            box_valid_cnt += 1

        if ok_row and ok_col and ok_box:
            valid += 1

    return valid, row_valid_cnt, col_valid_cnt, box_valid_cnt, B


# ────────────────────────────────────────────────────────────
#  Evaluation  (sample grids + compare with val GT)
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_save(
    diffusion: DiscreteDiffusion,
    step: int,
    args,
    accelerator: Accelerator,
    ema: EMA | None,
    val_dataset=None,
    cond_unet=None,
    val_cached_tok_ids=None,
    val_grids=None,
    val_dataset_raw=None,
):
    """Generate samples and evaluate.

    - grid mode:        sample grid digits, visualize, check sudoku rules
    - image_cond_mode:  sample tok_ids → render images → sudoku eval
    """
    if not accelerator.is_main_process:
        return

    # ── image_cond_mode: delegate to evaluate_image_cond ──
    if args.image_cond_mode:
        evaluate_image_cond(
            diffusion, step, args, accelerator, ema,
            cond_unet=cond_unet,
            val_cached_tok_ids=val_cached_tok_ids,
            val_grids=val_grids,
            val_dataset_raw=val_dataset_raw,
        )
        return

    device = accelerator.device
    model = accelerator.unwrap_model(diffusion)
    model.eval()

    params = list(model.parameters())
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)

    save_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(save_dir, exist_ok=True)

    # ── grid mode only from here ──
    grid_hw = args.grid_hw
    seq_len = grid_hw * grid_hw
    vocab_size = args.grid_vocab_size

    # ── sample ──
    cond_tokens = None
    if args.conditional_training:
        # All-MASK prefix → unconditional generation from conditional model
        mask_token = model.mask_index
        cond_prefix = torch.full(
            (args.eval_num_samples, seq_len), mask_token,
            dtype=torch.long, device=device)
        backbone = model.backbone
        cond_tokens = backbone.token_emb(cond_prefix)
        cond_tokens = backbone.pos_emb(cond_tokens)
    tokens = model.sample(
        batch_size=args.eval_num_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
        tokens_per_step=args.tokens_per_step,
        cond_tokens=cond_tokens,
    )  # (B, seq_len)

    # ════════════════════════════════════════════════════════
    #  Unconditional evaluation  (rule checks + images + videos)
    # ════════════════════════════════════════════════════════
    uncond_dir = os.path.join(save_dir, "unconditional")
    os.makedirs(uncond_dir, exist_ok=True)

    B = tokens.shape[0]
    grids = tokens.view(B, grid_hw, grid_hw)  # (B, 9, 9) values in [0,8]

    # ── sudoku rule check ──
    n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(grids)
    rule_acc = n_valid / max(n_total, 1)
    row_acc  = n_row  / max(n_total, 1)
    col_acc  = n_col  / max(n_total, 1)
    box_acc  = n_box  / max(n_total, 1)

    # ── digit distribution ──
    vals = grids.flatten()
    dist_strs = []
    for v in range(vocab_size):
        cnt = (vals == v).sum().item()
        dist_strs.append(f"digit {v+1}: {cnt} ({cnt/vals.numel()*100:.1f}%)")

    accelerator.print(
        f"[eval/uncond] step={step}  n={n_total}  "
        f"rule_acc={rule_acc:.4f}  "
        f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})")

    # ── save images (with rule violations marked) ──
    nrow = min(8, int(math.sqrt(B)))
    n_vis = min(64, B)

    sample_imgs = []
    for i in range(n_vis):
        g_np = grids[i].cpu().numpy()
        viol = _find_rule_violations(g_np, grid_hw)
        viol_mask = torch.from_numpy(viol)
        sample_imgs.append(
            render_digit_grid(grids[i] + 1, wrong_mask=viol_mask))
    canvas = tile_images(sample_imgs, nrow=nrow)
    if canvas is not None:
        canvas.save(os.path.join(uncond_dir, f"step_{step:07d}_sampled.png"))

    # ── save text details ──
    txt_path = os.path.join(uncond_dir, f"step_{step:07d}_details.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step}  samples={n_total}\n")
        f.write(f"rule_acc={rule_acc:.6f}  "
                f"row={row_acc:.6f}  col={col_acc:.6f}  box={box_acc:.6f}\n")
        f.write("digit distribution:\n")
        for dd in dist_strs:
            f.write(f"  {dd}\n")
        f.write("\nfirst 8 grids (internal 0-8 values, flat):\n")
        for i in range(min(8, B)):
            f.write(f"  sample {i}: {tokens[i].tolist()}\n")

    # ── logging ──
    uncond_log = {
        "eval/uncond/rule_acc": rule_acc,
        "eval/uncond/row_acc": row_acc,
        "eval/uncond/col_acc": col_acc,
        "eval/uncond/box_acc": box_acc,
    }
    if args.log_with:
        accelerator.log(uncond_log, step=step)

    # ── save unconditional sampling videos ──
    n_gif = getattr(args, "eval_gif_samples", 0)
    vid_fmt = getattr(args, "eval_save_format", "gif")
    if n_gif > 0:
        vid_dir = os.path.join(uncond_dir, "gifs")
        os.makedirs(vid_dir, exist_ok=True)
        vid_tokens, vid_history = model.sample(
            batch_size=n_gif,
            seq_len=seq_len,
            num_steps=args.eval_num_steps,
            device=device,
            sampler=args.sampler,
            noise_removal=True,
            return_history=True,
            tokens_per_step=args.tokens_per_step,
        )
        ext = getattr(args, "eval_save_format", "gif")
        for vi in range(n_gif):
            vi_grid = vid_tokens[vi].view(grid_hw, grid_hw)
            vi_valid = _check_sudoku_single(vi_grid)
            tag = "valid" if vi_valid else "invalid"
            vid_path = os.path.join(
                vid_dir,
                f"step_{step:07d}_sample{vi}_{tag}.{ext}")
            render_sampling_gif(
                history=vid_history,
                sample_idx=vi,
                grid_hw=grid_hw,
                mask_index=model.mask_index,
                save_path=vid_path,
                save_format=ext,
            )
        accelerator.print(
            f"[eval/uncond] Saved {n_gif} GIFs → {vid_dir}/")

    accelerator.print(f"[eval/uncond] saved → {uncond_dir}/")

    # ════════════════════════════════════════════════════════
    #  grid mode → delegate to evaluate_difficulty (per-task)
    # ════════════════════════════════════════════════════════
    if ema is not None:
        ema.restore(params)
    model.train()
    evaluate_difficulty(
        diffusion, step, args, accelerator, ema, val_dataset,
    )


# ────────────────────────────────────────────────────────────
#  Difficulty-based evaluation (inpainting)
# ────────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = {
    "hard":   (0,  26),   # 0-26 given cells  → model fills 55-81
    "medium": (27, 53),   # 27-53 given cells → model fills 28-54
    "easy":   (54, 80),   # 54-80 given cells → model fills 1-27
}


@torch.no_grad()
def evaluate_difficulty(
    diffusion: DiscreteDiffusion,
    step: int,
    args,
    accelerator: Accelerator,
    ema: EMA | None,
    val_dataset,
):
    """Evaluate the model per difficulty level (hard / medium / easy).

    Each level reveals a random number of GT hint cells; the model fills
    the rest via ``sample_inpaint``.  Results are saved into per-task
    subdirectories:  ``eval_samples/hard/``, ``medium/``, ``easy/``.
    """
    if not accelerator.is_main_process:
        return
    if val_dataset is None:
        accelerator.print("[eval] No val_dataset, skipping difficulty eval.")
        return

    device = accelerator.device
    model = accelerator.unwrap_model(diffusion)
    model.eval()

    params = list(model.parameters())
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)

    n_samples = min(args.eval_num_samples, len(val_dataset))
    grid_hw = args.grid_hw
    seq_len = grid_hw * grid_hw  # 81

    # gather GT grids  (B, 9, 9) → (B, 81)
    gt_list = []
    for idx in range(n_samples):
        sample = val_dataset[idx]
        if isinstance(sample, dict) and "grid" in sample:
            gt_list.append(sample["grid"].unsqueeze(0))
    if not gt_list:
        accelerator.print("[eval] No GT grids found, skipping.")
        if ema is not None:
            ema.restore(params)
        model.train()
        return
    gt_grids = torch.cat(gt_list, dim=0).to(device).long()           # (B, 9, 9)
    x_gt = gt_grids.view(gt_grids.shape[0], -1) - 1                   # (B, 81) in [0,8]
    B = x_gt.shape[0]
    nrow = min(8, int(math.sqrt(B)))
    n_vis = min(64, B)
    n_gif = getattr(args, "eval_gif_samples", 0)

    log_dict = {}
    summary_lines = [f"[eval] step={step}  samples_per_level={B}"]

    for level_name, (hint_lo, hint_hi) in DIFFICULTY_LEVELS.items():
        # ── per-task save directory ──
        task_dir = os.path.join(
            args.output_dir, "eval_samples", level_name)
        os.makedirs(task_dir, exist_ok=True)

        # Fixed seed per level → same hint pattern every eval step
        rng = torch.Generator(device=device)
        level_seed = hash(level_name) & 0xFFFFFFFF
        rng.manual_seed(level_seed)

        n_hints = torch.randint(
            hint_lo, hint_hi + 1, (B,),
            device=device, generator=rng)                             # (B,)

        known_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        for b in range(B):
            nh = n_hints[b].item()
            if nh > 0:
                perm = torch.randperm(seq_len, device=device,
                                      generator=rng)[:nh]
                known_mask[b, perm] = True

        # ── generate unknown cells ──
        if args.conditional_training:
            # Prefix conditioning: build masked condition prefix
            mask_token = model.mask_index
            cond_prefix = x_gt.clone()
            cond_prefix[~known_mask] = mask_token  # unknown → MASK
            # Embed condition prefix
            backbone = model.backbone
            cond_emb = backbone.token_emb(cond_prefix)
            cond_emb = backbone.pos_emb(cond_emb)
            completed_flat = model.sample(
                batch_size=B,
                seq_len=seq_len,
                num_steps=args.eval_num_steps,
                device=device,
                sampler=args.sampler,
                noise_removal=True,
                cond_tokens=cond_emb,
                tokens_per_step=args.tokens_per_step,
            )
            step_logs = None
        else:
            # Inpainting: force known positions during denoising
            completed_flat = model.sample_inpaint(
                x_gt=x_gt,
                known_mask=known_mask,
                num_steps=args.eval_num_steps,
                sampler=args.sampler,
                noise_removal=True,
                tokens_per_step=args.tokens_per_step,
                return_step_logs=True,
            )
            if isinstance(completed_flat, tuple):
                completed_flat, step_logs = completed_flat
            else:
                step_logs = None
            
        completed_grids = completed_flat.view(B, grid_hw, grid_hw)

        # ── cell accuracy (unknown positions only) ──
        unknown_mask = ~known_mask
        n_unknown = unknown_mask.float().sum()
        if n_unknown > 0:
            cell_acc = (
                (completed_flat == x_gt) & unknown_mask
            ).float().sum() / n_unknown
            cell_acc = cell_acc.item()
        else:
            cell_acc = 1.0

        # ── full-grid cell accuracy ──
        full_cell_acc = (completed_flat == x_gt).float().mean().item()
        wrong_per_grid = (
            completed_grids != gt_grids
        ).float().sum(dim=(1, 2)).mean().item()

        # ── sudoku rule check ──
        n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(
            completed_grids)
        rule_acc = n_valid / max(n_total, 1)
        row_acc  = n_row  / max(n_total, 1)
        col_acc  = n_col  / max(n_total, 1)
        box_acc  = n_box  / max(n_total, 1)

        avg_hints = n_hints.float().mean().item()

        # ── digit distribution ──
        digit_dist = []
        vals = completed_grids.flatten()
        for v in range(args.grid_vocab_size):
            cnt = (vals == v).sum().item()
            digit_dist.append(f"  digit {v+1}: {cnt} ({cnt/vals.numel()*100:.1f}%)")  # v+1 for display

        # ── logging ──
        prefix = f"eval/{level_name}"
        log_dict[f"{prefix}/cell_acc_unknown"] = cell_acc
        log_dict[f"{prefix}/cell_acc"] = full_cell_acc
        log_dict[f"{prefix}/wrong_per_grid"] = wrong_per_grid
        log_dict[f"{prefix}/rule_acc"] = rule_acc
        log_dict[f"{prefix}/row_acc"] = row_acc
        log_dict[f"{prefix}/col_acc"] = col_acc
        log_dict[f"{prefix}/box_acc"] = box_acc
        log_dict[f"{prefix}/avg_hints"] = avg_hints

        summary_lines.append(
            f"  {level_name:6s}  hints={avg_hints:5.1f}  "
            f"cell_acc(unk)={cell_acc:.4f}  cell_acc(all)={full_cell_acc:.4f}  "
            f"rule_acc={rule_acc:.4f}  "
            f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})")

        # ── save: sampled (completed) grids ──
        sample_imgs = [
            render_digit_grid(completed_grids[i])
            for i in range(n_vis)]
        canvas = tile_images(sample_imgs, nrow=nrow)
        if canvas is not None:
            canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_sampled.png"))

        # ── save: GT grids ──
        gt_imgs = [
            render_digit_grid(gt_grids[i])
            for i in range(min(gt_grids.shape[0], n_vis))]
        gt_canvas = tile_images(gt_imgs, nrow=nrow)
        if gt_canvas is not None:
            gt_canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_gt.png"))

        # ── save: comparison (wrong filled cells in red) ──
        cmp_imgs = []
        for i in range(n_vis):
            hint_2d = known_mask[i].view(grid_hw, grid_hw)
            wrong = (completed_grids[i] != gt_grids[i])
            cmp_imgs.append(
                render_digit_grid(
                    completed_grids[i],
                    wrong_mask=wrong & ~hint_2d,
                ))
        cmp_canvas = tile_images(cmp_imgs, nrow=nrow)
        if cmp_canvas is not None:
            cmp_canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_cmp.png"))

        # ── save: text details ──
        txt_path = os.path.join(
            task_dir, f"step_{step:07d}_details.txt")
        with open(txt_path, "w") as f:
            f.write(f"step={step}  level={level_name}  "
                    f"samples={B}  avg_hints={avg_hints:.1f}\n")
            f.write(f"cell_acc(unknown)={cell_acc:.6f}\n")
            f.write(f"cell_acc(all)={full_cell_acc:.6f}\n")
            f.write(f"wrong_per_grid={wrong_per_grid:.2f}\n")
            f.write(f"rule_acc={rule_acc:.6f}  "
                    f"row={row_acc:.6f}  col={col_acc:.6f}  "
                    f"box={box_acc:.6f}\n")
            f.write("digit distribution:\n")
            for dd in digit_dist:
                f.write(dd + "\n")
            f.write("\nfirst 8 completed grids (flat):\n")
            for i in range(min(8, B)):
                f.write(f"  sample {i}: {completed_flat[i].tolist()}\n")
        
        # ── save: step logs (sampling trajectory) ──
        if step_logs is not None:
            step_logs_path = os.path.join(
                task_dir, f"step_{step:07d}_sampling_logs.txt")
            with open(step_logs_path, "w") as f:
                f.write(f"Sampling logs for {level_name} level\n")
                f.write("=" * 80 + "\n\n")
                for log_entry in step_logs:
                    step_idx = log_entry['step']
                    t_val = log_entry['t']
                    # Convert tensor to scalar if needed
                    if isinstance(t_val, torch.Tensor):
                        t_val = t_val.item()
                    n_masked_list = log_entry['n_masked']
                    # n_masked_list is per-batch, average it
                    n_masked_avg = sum(n_masked_list) / len(n_masked_list)
                    f.write(f"Step {step_idx:3d}: t={t_val:.6f}  n_masked={n_masked_avg:.1f}/81\n")

        # ── save: GIFs (for a few samples) ──
        if n_gif > 0:
            gif_dir = os.path.join(task_dir, "gifs")
            os.makedirs(gif_dir, exist_ok=True)
            # Re-run inpaint with return_history for gif samples
            gif_result = model.sample_inpaint(
                x_gt=x_gt[:n_gif],
                known_mask=known_mask[:n_gif],
                num_steps=args.eval_num_steps,
                sampler=args.sampler,
                noise_removal=True,
                return_history=True,
                return_confidence_history=True,
                tokens_per_step=args.tokens_per_step,
                return_step_logs=True,
            )
            # Unpack result
            if isinstance(gif_result, tuple):
                if len(gif_result) == 5:
                    gif_completed, gif_history, gif_conf, gif_pred, gif_step_logs = gif_result
                else:
                    gif_completed, gif_history, gif_conf, gif_pred = gif_result
                    gif_step_logs = None
            else:
                gif_completed = gif_result
                gif_history = None
                gif_conf = None
                gif_pred = None
                gif_step_logs = None
                
            if gif_history is not None:
                # Debug: print confidence stats and step logs
                if gif_conf is not None and len(gif_conf) > 0:
                    for gi_step in [0, min(5, len(gif_conf)-1), len(gif_conf)-1]:
                        if gi_step < len(gif_conf):
                            masked = (gif_history[gi_step][0] == model.mask_index)
                            if masked.any():
                                conf_vals = gif_conf[gi_step][0][masked]
                                accelerator.print(
                                    f"  [GIF debug] step {gi_step}: n_masked={masked.sum().item()}, "
                                    f"conf=[{conf_vals.min():.3f}, {conf_vals.max():.3f}]")
                for gi in range(min(n_gif, len(gif_history[0]))):
                    ext = args.eval_save_format
                    # determine per-sample success for filename tag
                    gi_grid = gif_completed[gi].view(grid_hw, grid_hw)
                    gi_gt = x_gt[gi].view(grid_hw, grid_hw)
                    gi_n_wrong = int((gi_grid != gi_gt).sum().item())
                    gi_valid = _check_sudoku_single(gi_grid)
                    if gi_valid and gi_n_wrong == 0:
                        tag = "perfect"
                    elif gi_valid:
                        tag = f"valid_w{gi_n_wrong}"
                    else:
                        tag = f"fail_w{gi_n_wrong}"
                    gif_path = os.path.join(
                        gif_dir,
                        f"step_{step:07d}_sample{gi}_{tag}.{ext}")
                    render_sampling_gif(
                        history=gif_history,
                        sample_idx=gi,
                        grid_hw=grid_hw,
                        mask_index=model.mask_index,
                        confidence_history=gif_conf,
                        pred_history=gif_pred,
                        known_mask=known_mask[:n_gif],
                        save_path=gif_path,
                        save_format=ext,
                    )
                accelerator.print(
                    f"[eval/{level_name}] Saved {min(n_gif, len(gif_history[0]))} "
                    f"GIFs → {gif_dir}/")

        accelerator.print(
            f"[eval/{level_name}] step={step}  "
            f"cell_acc(unk)={cell_acc:.4f}  rule_acc={rule_acc:.4f}  "
            f"saved → {task_dir}/")

    # ── overall summary ──
    for line in summary_lines:
        accelerator.print(line)
    if args.log_with:
        accelerator.log(log_dict, step=step)

    # save combined summary
    summary_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(summary_dir, exist_ok=True)
    txt_path = os.path.join(summary_dir, f"step_{step:07d}_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    if ema is not None:
        ema.restore(params)
    model.train()


# ────────────────────────────────────────────────────────────
#  Load full ConditionalUNet for image_cond_mode
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def load_cond_unet(args, accelerator):
    """Load a pretrained ConditionalUNet from an accelerate checkpoint.

    Uses run_config.json in the parent output_dir of --cond_unet_ckpt
    to reconstruct the exact architecture.

    Returns
    -------
    cond_unet : ConditionalUNet    frozen, eval mode
    image_token_vocab_size : int   number of discrete token classes
    image_token_seq_len : int      number of tokens per image (h*w)
    """
    import glob
    _lazy_image_cond_imports()

    assert args.cond_unet_ckpt is not None, \
        "--image_cond_mode requires --cond_unet_ckpt"
    assert args.cond_unet_config is not None, \
        "--image_cond_mode requires --cond_unet_config"

    # 1) load unet config
    with open(args.cond_unet_config, "r") as f:
        unet_config = json.load(f)

    # 2) try to load run_config.json near the checkpoint for exact args
    ckpt_dir = args.cond_unet_ckpt
    run_cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_dir)),
                                "run_config.json")
    orig_args = {}
    if os.path.isfile(run_cfg_path):
        with open(run_cfg_path, "r") as f:
            orig_args = json.load(f).get("args", {})
        accelerator.print(f"[image_cond] Loaded run_config from {run_cfg_path}")

    use_fsq = orig_args.get("use_fsq", args.cond_discretizer_type == "fsq")
    use_vq = orig_args.get("use_vq_discretizer", args.cond_discretizer_type == "vq")
    fsq_levels = orig_args.get("fsq_levels", args.cond_fsq_levels)
    cond_dim = orig_args.get("cond_dim", args.cond_dim)
    feat_channels = orig_args.get("feat_channels", args.cond_feat_channels)
    image_size = orig_args.get("pad_image_size", args.cond_image_size)
    patch_cond = orig_args.get("patch_conditioning", args.cond_patch_conditioning)
    patch_grid = orig_args.get("patch_grid_size", args.cond_patch_grid_size)
    concat_cond = orig_args.get("concat_conditioning", False)
    concat_down = orig_args.get("concat_downsample_factor",
                                args.cond_concat_downsample_factor)
    concat_channels = orig_args.get("concat_channels", 4)
    uncond_drop = orig_args.get("uncond_drop_prob", 0.0)
    vq_codebook = orig_args.get("vq_codebook_size", args.cond_vq_codebook_size)
    vq_beta = orig_args.get("vq_beta", 0.25)
    pred_type = orig_args.get("prediction_type", "epsilon")

    # 3) rebuild ConditionalUNet with exact same args
    cond_unet = _ConditionalUNet(
        num_classes=10,
        class_embed_dim=unet_config.get("cross_attention_dim", 128),
        image_size=image_size,
        image_conditioning=True,
        cond_in_channels=1,
        feat_channels=feat_channels,
        cond_dim=cond_dim,
        unet_config=unet_config,
        grid_conditioning=False,
        use_fsq=use_fsq,
        fsq_levels=fsq_levels,
        fsq_drop_quant_p=0.0,
        fsq_corrupt_tokens_p=0.0,
        use_vq_discretizer=use_vq,
        vq_codebook_size=vq_codebook,
        vq_beta=vq_beta,
        concat_conditioning=concat_cond,
        concat_downsample_factor=concat_down,
        concat_channels=concat_channels,
        patch_conditioning=patch_cond,
        patch_grid_size=patch_grid,
        uncond_drop_prob=uncond_drop,
    )

    # 4) load weights
    model_files = (glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")) +
                   glob.glob(os.path.join(ckpt_dir, "model*.safetensors")))
    if model_files:
        for mf in model_files:
            if mf.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(mf)
            else:
                state = torch.load(mf, map_location="cpu", weights_only=True)
            cond_unet.load_state_dict(state, strict=False)
            accelerator.print(f"[image_cond] Loaded weights from {mf}")
    else:
        accelerator.print(
            f"[image_cond] WARNING: no model weights found in {ckpt_dir}")

    # 5) freeze everything & keep on CPU
    cond_unet.eval()
    for p in cond_unet.parameters():
        p.requires_grad_(False)

    # Stay on CPU — moved to GPU only when needed (caching / rendering)
    cond_unet = cond_unet.to("cpu")

    # 6) compute vocab & seq_len
    discretizer = getattr(cond_unet, "discretizer", None)
    if use_fsq and discretizer is not None:
        image_token_vocab_size = 1
        for lv in fsq_levels:
            image_token_vocab_size *= lv
    elif use_vq and discretizer is not None:
        image_token_vocab_size = vq_codebook
    else:
        raise RuntimeError("image_cond_mode requires a discretizer (fsq or vq)")

    if patch_cond:
        image_token_seq_len = patch_grid * patch_grid
    else:
        image_token_seq_len = (image_size // concat_down) ** 2

    accelerator.print(
        f"[image_cond] ConditionalUNet loaded (frozen)  │ "
        f"discretizer={'fsq' if use_fsq else 'vq'} │ "
        f"tok_vocab={image_token_vocab_size} │ "
        f"tok_seq_len={image_token_seq_len} │ "
        f"prediction_type={pred_type}")

    # store prediction_type on args for eval
    args._cond_prediction_type = pred_type

    return cond_unet, image_token_vocab_size, image_token_seq_len


# ────────────────────────────────────────────────────────────
#  Cache all tok_ids at startup  (run encoder once)
# ────────────────────────────────────────────────────────────

class CachedTokenDataset(torch.utils.data.Dataset):
    """Dataset that holds pre-computed tok_ids + sudoku grids."""

    def __init__(self, tok_ids: torch.Tensor, grids: torch.Tensor):
        """
        tok_ids: (N, L)  int64
        grids:   (N, 9, 9)  int64
        """
        self.tok_ids = tok_ids
        self.grids = grids

    def __len__(self):
        return len(self.tok_ids)

    def __getitem__(self, idx):
        return {
            "tok_ids": self.tok_ids[idx],
            "grid": self.grids[idx],
        }


@torch.no_grad()
def cache_all_tokens(
    cond_unet,
    raw_dataset,
    device,
    batch_size: int = 64,
    accelerator=None,
    cache_path: str = None,
):
    """Run all images through encoder+discretizer once and cache tok_ids.

    If *cache_path* is given and already exists on disk, tok_ids are loaded
    directly (encoder is never invoked).  Otherwise **all ranks split the
    dataset and encode in parallel**, then all-gather the results.  Rank 0
    saves the combined tensor to *cache_path* for future runs.

    Args:
        cond_unet: frozen ConditionalUNet
        raw_dataset: SRM dataset with __getitem__ returning {"image": ...}
        device: torch device
        batch_size: encoding batch size (doesn't need to match training)
        cache_path: optional .pt file path for disk caching

    Returns:
        tok_ids: (N, L) int64 on CPU
    """
    is_main = accelerator is None or accelerator.is_main_process
    num_procs = 1 if accelerator is None else accelerator.num_processes
    rank = 0 if accelerator is None else accelerator.process_index

    # ── Try loading from disk (all ranks) ──
    if cache_path is not None and os.path.isfile(cache_path):
        all_tok_ids = torch.load(cache_path, map_location="cpu")
        if accelerator is not None:
            accelerator.print(
                f"[cache] Loaded tok_ids from {cache_path}  "
                f"shape={all_tok_ids.shape}, "
                f"range=[{all_tok_ids.min().item()}, {all_tok_ids.max().item()}]")
        return all_tok_ids

    # ── Encode from scratch: split across ranks ──
    cond_unet.eval()
    cond_unet.to(device)  # move to GPU for encoding
    N = len(raw_dataset)

    # Determine this rank's shard of indices
    indices = list(range(N))
    shard_size = (N + num_procs - 1) // num_procs
    start = rank * shard_size
    end = min(start + shard_size, N)
    my_indices = indices[start:end]

    if accelerator is not None:
        accelerator.print(
            f"[cache] Rank {rank}: encoding samples {start}..{end-1} "
            f"({len(my_indices)}/{N})")

    shard_dataset = torch.utils.data.Subset(raw_dataset, my_indices)
    loader = DataLoader(
        shard_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    my_tok_ids = []
    for i, batch in enumerate(loader):
        if isinstance(batch, dict):
            images = batch["image"].to(device)
        else:
            images = batch[0].to(device)

        _, tok_ids = cond_unet.cond_encoding(
            cond_image=images,
            return_token_ids=True,
        )
        my_tok_ids.append(tok_ids.cpu())

        if accelerator is not None and (i + 1) % 50 == 0:
            done = min((i + 1) * batch_size, len(my_indices))
            print(
                f"[cache] Rank {rank}: encoded {done}/{len(my_indices)}...")

    cond_unet.to("cpu")  # move back to CPU after encoding
    torch.cuda.empty_cache()
    my_tok_ids = torch.cat(my_tok_ids, dim=0)  # (shard_N, L)

    # ── All-gather across ranks ──
    if num_procs > 1:
        import torch.distributed as dist

        # Pad shards to equal length for all_gather
        L = my_tok_ids.shape[1]
        padded = torch.zeros(shard_size, L, dtype=my_tok_ids.dtype)
        padded[:len(my_tok_ids)] = my_tok_ids
        padded = padded.to(device)

        gathered = [torch.zeros_like(padded) for _ in range(num_procs)]
        dist.all_gather(gathered, padded)

        # Trim padding and concatenate in original order
        parts = []
        for r in range(num_procs):
            r_start = r * shard_size
            r_end = min(r_start + shard_size, N)
            r_count = r_end - r_start
            parts.append(gathered[r][:r_count].cpu())
        all_tok_ids = torch.cat(parts, dim=0)  # (N, L)
    else:
        all_tok_ids = my_tok_ids

    if accelerator is not None:
        accelerator.print(
            f"[cache] Done! tok_ids shape={all_tok_ids.shape}, "
            f"range=[{all_tok_ids.min().item()}, {all_tok_ids.max().item()}]")

    # ── Save to disk (rank 0 only) ──
    if cache_path is not None and is_main:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(all_tok_ids, cache_path)
        if accelerator is not None:
            accelerator.print(f"[cache] Saved tok_ids to {cache_path}")

    if accelerator is not None:
        accelerator.wait_for_everyone()

    return all_tok_ids


# ────────────────────────────────────────────────────────────
#  Image rendering from tok_ids  (DDIM denoising)
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def _render_batch(
    tok_ids: torch.Tensor,
    cond_unet,
    num_inference_steps: int = 50,
    prediction_type: str = "sample",
):
    """Render a single mini-batch of tok_ids → images (internal helper)."""
    _lazy_image_cond_imports()

    device = tok_ids.device
    B = tok_ids.shape[0]

    # Move cond_unet to GPU for rendering
    cond_unet.to(device)

    # 1) tok_ids → continuous embeddings via discretizer.decode
    discretizer = cond_unet.discretizer
    cond_tokens = discretizer.decode(tok_ids)  # (B, L, cond_dim)

    # 2) set up DDIM scheduler matching the training scheduler
    ddim_scheduler = _DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=2e-5,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type=prediction_type,
    )
    ddim_scheduler.set_timesteps(num_inference_steps, device=device)

    # 3) get UNet config for sample_size
    unet = cond_unet.unet
    sample_size = unet.config.sample_size

    # 4) start from random noise
    x = torch.randn(B, 1, sample_size, sample_size, device=device)

    # 5) dummy cond_image to satisfy forward() guard check.
    dummy_cond_image = torch.zeros(
        B, 1, sample_size, sample_size, device=device)

    # 6) DDIM denoising loop
    for t in ddim_scheduler.timesteps:
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred = cond_unet(
            x, t_batch,
            encoder_hidden_states=cond_tokens,
            cond_image=dummy_cond_image,
        )
        x = ddim_scheduler.step(pred, t, x).prev_sample

    # Move cond_unet back to CPU to free GPU memory
    cond_unet.to("cpu")
    torch.cuda.empty_cache()

    return x  # (B, 1, H, W) in ~[-1, 1]


def render_images_from_tok_ids(
    tok_ids: torch.Tensor,
    cond_unet,
    num_inference_steps: int = 50,
    prediction_type: str = "sample",
    render_batch_size: int = 16,
):
    """Decode tok_ids to continuous embeddings and denoise to images.

    Processes *render_batch_size* samples at a time to avoid VRAM OOM
    when the total number of samples is large.

    Args:
        tok_ids: (B, L) int64 token indices
        cond_unet: frozen ConditionalUNet (has .discretizer, .unet, etc.)
        num_inference_steps: DDIM steps
        prediction_type: "epsilon" | "sample" | "v_prediction"
        render_batch_size: mini-batch size for DDIM rendering

    Returns:
        images: (B, C, H, W) in [-1, 1]
    """
    B = tok_ids.shape[0]
    if B <= render_batch_size:
        return _render_batch(tok_ids, cond_unet, num_inference_steps,
                             prediction_type)

    chunks = []
    for start in range(0, B, render_batch_size):
        end = min(start + render_batch_size, B)
        chunk = _render_batch(tok_ids[start:end], cond_unet,
                              num_inference_steps, prediction_type)
        chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0).to(tok_ids.device)


# ────────────────────────────────────────────────────────────
#  image_cond_mode evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_image_cond(
    diffusion: DiscreteDiffusion,
    step: int,
    args,
    accelerator: Accelerator,
    ema: EMA | None,
    cond_unet,
    val_cached_tok_ids: torch.Tensor,
    val_grids: torch.Tensor,
    val_dataset_raw=None,
):
    """Evaluate image_cond_mode (mirrors grid-mode evaluate_and_save).

    Unconditional:
      1) Sample tok_ids → render images → eval_images → digit grids
      2) Rule checks, digit distribution, violation visualisation
      3) GT rendering sanity check + tok2digit analysis

    Then delegates to evaluate_image_cond_difficulty for per-task eval.
    """
    if not accelerator.is_main_process:
        return

    _lazy_image_cond_imports()
    from torchvision.utils import make_grid, save_image

    device = accelerator.device
    model = accelerator.unwrap_model(diffusion)
    model.eval()

    params = list(model.parameters())
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)

    # ════════════════════════════════════════════════════════
    #  Unconditional evaluation
    # ════════════════════════════════════════════════════════
    uncond_dir = os.path.join(args.output_dir, "eval_samples", "unconditional")
    os.makedirs(uncond_dir, exist_ok=True)

    seq_len = model.backbone.seq_len
    vocab_size = model.data_vocab_size
    grid_hw = 9  # sudoku
    n_samples = min(args.eval_num_samples, len(val_cached_tok_ids))
    B = n_samples

    # ── 1) Sample tok_ids from discrete diffusion ──
    sampled_tok = model.sample(
        batch_size=n_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
        tokens_per_step=args.tokens_per_step,
    )  # (B, L)

    # ── 2) tok_acc vs GT (image_cond specific) ──
    gt_tok = val_cached_tok_ids[:n_samples].to(device)
    tok_acc = (sampled_tok == gt_tok).float().mean().item()
    wrong_per_seq = (sampled_tok != gt_tok).float().sum(dim=1).mean().item()

    # token distribution (top-k)
    tok_vals = sampled_tok.flatten()
    tok_counts = torch.bincount(tok_vals, minlength=vocab_size)
    top_k = min(10, vocab_size)
    top_vals, top_ids = tok_counts.topk(top_k)
    tok_dist_strs = [f"tok={tid}:{cnt}" for tid, cnt in
                     zip(top_ids.tolist(), top_vals.tolist())]

    accelerator.print(
        f"[eval/uncond] step={step}  tok_acc={tok_acc:.4f}  "
        f"wrong_per_seq={wrong_per_seq:.2f}/{seq_len}")
    accelerator.print(
        f"[eval/uncond] token dist (top-{top_k}): {', '.join(tok_dist_strs)}")

    # ── 3) Render images from sampled tok_ids (ALL samples) ──
    pred_type = getattr(args, "_cond_prediction_type", "sample")
    ddim_steps = getattr(args, "cond_eval_ddim_steps", 50)
    rbs = getattr(args, "eval_render_batch_size", 16)

    rendered_imgs = render_images_from_tok_ids(
        sampled_tok, cond_unet,
        num_inference_steps=ddim_steps,
        prediction_type=pred_type,
        render_batch_size=rbs,
    )  # (B, 1, H, W) in [-1,1]
    rendered_01 = (rendered_imgs.clamp(-1, 1) + 1) * 0.5

    # Also render GT tok_ids for sanity check
    gt_rendered = render_images_from_tok_ids(
        gt_tok, cond_unet,
        num_inference_steps=ddim_steps,
        prediction_type=pred_type,
        render_batch_size=rbs,
    )
    gt_rendered_01 = (gt_rendered.clamp(-1, 1) + 1) * 0.5

    # ── 4) Save rendered image grids (for visual inspection) ──
    n_vis = min(64, B)
    nrow = min(8, int(math.sqrt(n_vis)))
    nrow_img = min(9, int(math.sqrt(n_vis)) + 1)

    grid_sampled = make_grid(rendered_01[:n_vis], nrow=nrow_img, padding=2)
    save_image(grid_sampled,
               os.path.join(uncond_dir, f"step_{step:07d}_sampled_imgs.png"))
    grid_gt = make_grid(gt_rendered_01[:n_vis], nrow=nrow_img, padding=2)
    save_image(grid_gt,
               os.path.join(uncond_dir, f"step_{step:07d}_gt_imgs.png"))

    # ── 5) Sudoku eval via eval_images ──
    # Initialize metrics that may be set inside try block
    rule_acc = None
    row_acc = None
    col_acc = None
    box_acc = None
    cell_acc = None
    cell_acc_gt = None
    rule_acc_gt = None
    tok2digit_acc = None
    digit_dist_strs = []

    try:
        classifier_pth = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets", "mnist_sudoku", "mnist_classifier.pth")
        if not os.path.isfile(classifier_pth):
            classifier_pth = os.path.join(
                "datasets", "mnist_sudoku", "mnist_classifier.pth")

        evaluator = _SudokuEval(
            mnist_classifier_path=classifier_pth,
            grid_size=(9, 9),
        )

        gt_9x9 = val_grids[:n_samples].to(device).long()  # (B, 9, 9) in [1,9]

        # ── eval on sampled-tok rendered images ──
        s_eval = evaluator.eval_images(rendered_imgs.clamp(-1, 1))
        pred_grid = s_eval["discrete"].long()  # (B, 9, 9) in [1,9]

        # cell accuracy (pred vs GT)
        wrong_mask = (pred_grid != gt_9x9)
        n_wrong = wrong_mask.flatten(1).sum(dim=1)
        cell_acc = (~wrong_mask).float().mean().item()

        # sudoku rule check
        n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(pred_grid)
        rule_acc = n_valid / max(n_total, 1)
        row_acc  = n_row  / max(n_total, 1)
        col_acc  = n_col  / max(n_total, 1)
        box_acc  = n_box  / max(n_total, 1)

        # digit distribution (from predicted grids, [1,9])
        pred_vals = pred_grid.flatten()
        for v in range(1, 10):
            cnt = (pred_vals == v).sum().item()
            digit_dist_strs.append(
                f"digit {v}: {cnt} ({cnt / pred_vals.numel() * 100:.1f}%)")

        accelerator.print(
            f"[eval/uncond] step={step}  n={n_total}  "
            f"rule_acc={rule_acc:.4f}  "
            f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})  "
            f"cell_acc={cell_acc:.4f}  "
            f"wrong_mean={n_wrong.float().mean():.2f}")

        # ── save digit grid images (with rule violations marked) ──
        sample_digit_imgs = []
        for i in range(n_vis):
            g_np = (pred_grid[i] - 1).cpu().numpy()  # [0,8] for violation check
            viol = _find_rule_violations(g_np, grid_hw)
            viol_mask = torch.from_numpy(viol)
            sample_digit_imgs.append(
                render_digit_grid(pred_grid[i], wrong_mask=viol_mask))
        canvas = tile_images(sample_digit_imgs, nrow=nrow)
        if canvas is not None:
            canvas.save(os.path.join(
                uncond_dir, f"step_{step:07d}_sampled.png"))

        # detailed wrong cell analysis (only when few errors)
        total_wrong = int(wrong_mask.sum().item())
        if total_wrong < 10:
            accelerator.print(
                f"[eval/uncond] TOTAL wrong cells={total_wrong} (<10)")
            wm = wrong_mask.detach().cpu()
            gt_cpu = gt_9x9.detach().cpu()
            pr_cpu = pred_grid.detach().cpu()
            for bi in range(wm.shape[0]):
                coords = torch.nonzero(wm[bi], as_tuple=False)
                if coords.numel() == 0:
                    continue
                accelerator.print(
                    f"[eval/uncond] sample#{bi} wrong={coords.shape[0]}")
                for r, c in coords.tolist():
                    accelerator.print(
                        f"  - (row={r}, col={c}) "
                        f"GT={int(gt_cpu[bi,r,c])} "
                        f"PRED={int(pr_cpu[bi,r,c])}")

        # ── eval on GT-tok rendered images (sanity check) ──
        s_gt_eval = evaluator.eval_images(gt_rendered.clamp(-1, 1))
        gt_pred_grid = s_gt_eval["discrete"].long()
        wrong_mask_gt = (gt_pred_grid != gt_9x9)
        cell_acc_gt = (~wrong_mask_gt).float().mean().item()
        n_valid_gt, _, _, _, _ = check_sudoku_rules(gt_pred_grid)
        rule_acc_gt = n_valid_gt / max(n_total, 1)

        accelerator.print(
            f"[eval/uncond][gt_tok] step={step}  "
            f"cell_acc={cell_acc_gt:.4f}  "
            f"rule_acc={rule_acc_gt:.4f} ({n_valid_gt}/{n_total})  "
            f"wrong_mean={wrong_mask_gt.flatten(1).sum(1).float().mean():.2f}")

        # ── tok2digit analysis ──
        tok_ids_2d = sampled_tok.view(-1, 9, 9).long()
        gt_flat = gt_9x9.reshape(-1).clamp(0, 9).long()
        tid_flat = tok_ids_2d.reshape(-1).long()
        vocab_n = int(tid_flat.max().item()) + 1
        idx = (tid_flat * 10 + gt_flat).long()
        cnt = torch.bincount(idx, minlength=vocab_n * 10).view(vocab_n, 10)
        tok2digit = cnt.argmax(dim=1)
        pred_from_tok = tok2digit[tok_ids_2d]
        tok2digit_acc = (pred_from_tok == gt_9x9).float().mean().item()

        accelerator.print(
            f"[eval/uncond] tok2digit_acc={tok2digit_acc:.4f}")

        # ── logging ──
        uncond_log = {
            "eval/uncond/rule_acc": rule_acc,
            "eval/uncond/row_acc": row_acc,
            "eval/uncond/col_acc": col_acc,
            "eval/uncond/box_acc": box_acc,
            "eval/uncond/cell_acc": cell_acc,
            "eval/uncond/tok_acc": tok_acc,
            "eval/uncond/wrong_per_seq": wrong_per_seq,
            "eval/uncond/cell_acc_gt_tok": cell_acc_gt,
            "eval/uncond/rule_acc_gt_tok": rule_acc_gt,
            "eval/uncond/tok2digit_acc": tok2digit_acc,
        }
        if args.log_with:
            accelerator.log(uncond_log, step=step)

    except Exception as e:
        import traceback
        accelerator.print(f"[eval/uncond] Sudoku evaluator failed: {e}")
        accelerator.print(traceback.format_exc())
        if args.log_with:
            accelerator.log({
                "eval/uncond/tok_acc": tok_acc,
                "eval/uncond/wrong_per_seq": wrong_per_seq,
            }, step=step)

    # ── 6) Save text details ──
    txt_path = os.path.join(uncond_dir, f"step_{step:07d}_details.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step}  samples={n_samples}\n")
        f.write(f"tok_acc={tok_acc:.6f}  wrong_per_seq={wrong_per_seq:.2f}\n")
        if rule_acc is not None:
            f.write(f"rule_acc={rule_acc:.6f}  "
                    f"row={row_acc:.6f}  col={col_acc:.6f}  "
                    f"box={box_acc:.6f}\n")
        if cell_acc is not None:
            f.write(f"cell_acc={cell_acc:.6f}\n")
        if digit_dist_strs:
            f.write("digit distribution:\n")
            for dd in digit_dist_strs:
                f.write(f"  {dd}\n")
        f.write(f"\nFirst 8 sampled tok_ids:\n")
        for i in range(min(8, n_samples)):
            f.write(f"  sample {i}: {sampled_tok[i].tolist()}\n")
        f.write(f"\nFirst 8 GT tok_ids:\n")
        for i in range(min(8, n_samples)):
            f.write(f"  gt     {i}: {gt_tok[i].tolist()}\n")

    accelerator.print(f"[eval/uncond] saved → {uncond_dir}/")

    # ════════════════════════════════════════════════════════
    #  Per-task inpainting (hard / medium / easy)
    # ════════════════════════════════════════════════════════
    if ema is not None:
        ema.restore(params)
    model.train()
    evaluate_image_cond_difficulty(
        diffusion, step, args, accelerator, ema,
        cond_unet=cond_unet,
        val_cached_tok_ids=val_cached_tok_ids,
        val_grids=val_grids,
    )


# ────────────────────────────────────────────────────────────
#  image_cond_mode: difficulty-based inpainting evaluation
#  (mirrors evaluate_difficulty but with token→render→eval_images pipeline)
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_image_cond_difficulty(
    diffusion: DiscreteDiffusion,
    step: int,
    args,
    accelerator: Accelerator,
    ema: EMA | None,
    cond_unet,
    val_cached_tok_ids: torch.Tensor,
    val_grids: torch.Tensor,
):
    """Per-difficulty inpainting eval for image_cond_mode.

    For each level (hard / medium / easy):
      1) Reveal some GT tok_ids as hints
      2) Inpaint the rest via sample_inpaint()
      3) Render completed tok_ids → images via DDIM
      4) Run evaluator.eval_images() → digit grids
      5) Evaluate: cell_acc (unknown / all), rule_acc (row/col/box),
         digit distribution, comparison images with wrong-cell marking
    """
    if not accelerator.is_main_process:
        return

    _lazy_image_cond_imports()
    from torchvision.utils import make_grid, save_image

    device = accelerator.device
    model = accelerator.unwrap_model(diffusion)
    model.eval()

    params = list(model.parameters())
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)

    seq_len = model.backbone.seq_len
    grid_hw = 9
    n_samples = min(args.eval_num_samples, len(val_cached_tok_ids))
    gt_tok = val_cached_tok_ids[:n_samples].to(device)   # (B, L)
    gt_9x9 = val_grids[:n_samples].to(device).long()     # (B, 9, 9) in [1,9]
    B = gt_tok.shape[0]

    pred_type = getattr(args, "_cond_prediction_type", "sample")
    ddim_steps = getattr(args, "cond_eval_ddim_steps", 50)
    rbs = getattr(args, "eval_render_batch_size", 16)

    nrow = min(8, int(math.sqrt(B)))
    n_vis = min(64, B)

    log_dict = {}
    summary_lines = [f"[eval] step={step}  samples_per_level={B}"]

    # ── set up evaluator once ──
    evaluator = None
    try:
        classifier_pth = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets", "mnist_sudoku", "mnist_classifier.pth")
        if not os.path.isfile(classifier_pth):
            classifier_pth = os.path.join(
                "datasets", "mnist_sudoku", "mnist_classifier.pth")
        evaluator = _SudokuEval(
            mnist_classifier_path=classifier_pth,
            grid_size=(9, 9),
        )
    except Exception as e:
        accelerator.print(f"[eval/diff] Failed to load evaluator: {e}")

    for level_name, (hint_lo, hint_hi) in DIFFICULTY_LEVELS.items():
        # ── per-task save directory ──
        task_dir = os.path.join(
            args.output_dir, "eval_samples", level_name)
        os.makedirs(task_dir, exist_ok=True)

        # ── build hint mask (same RNG as grid mode) ──
        rng = torch.Generator(device=device)
        level_seed = hash(level_name) & 0xFFFFFFFF
        rng.manual_seed(level_seed)

        n_hints = torch.randint(
            hint_lo, hint_hi + 1, (B,),
            device=device, generator=rng)

        known_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        for b in range(B):
            nh = n_hints[b].item()
            if nh > 0:
                perm = torch.randperm(seq_len, device=device,
                                      generator=rng)[:nh]
                known_mask[b, perm] = True

        avg_hints = n_hints.float().mean().item()

        # ── inpaint tok_ids ──
        completed_tok = model.sample_inpaint(
            x_gt=gt_tok,
            known_mask=known_mask,
            num_steps=args.eval_num_steps,
            sampler=args.sampler,
            noise_removal=True,
            tokens_per_step=args.tokens_per_step,
            return_step_logs=True,
        )
        if isinstance(completed_tok, tuple):
            completed_tok, step_logs = completed_tok
        else:
            step_logs = None

        # ── tok_ids accuracy ──
        unknown_mask = ~known_mask
        n_unknown = unknown_mask.float().sum()
        if n_unknown > 0:
            tok_acc_unk = (
                (completed_tok == gt_tok) & unknown_mask
            ).float().sum() / n_unknown
            tok_acc_unk = tok_acc_unk.item()
        else:
            tok_acc_unk = 1.0
        tok_acc_all = (completed_tok == gt_tok).float().mean().item()

        # ── render ALL images ──
        rendered = render_images_from_tok_ids(
            completed_tok, cond_unet,
            num_inference_steps=ddim_steps,
            prediction_type=pred_type,
            render_batch_size=rbs,
        )  # (B, 1, H, W)
        rendered_01 = (rendered.clamp(-1, 1) + 1) * 0.5

        # ── save rendered image grid ──
        nrow_img = min(9, int(math.sqrt(n_vis)) + 1)
        grid_img = make_grid(rendered_01[:n_vis], nrow=nrow_img, padding=2)
        save_image(
            grid_img,
            os.path.join(task_dir, f"step_{step:07d}_rendered.png"))

        # ── sudoku eval via eval_images ──
        cell_acc = None
        full_cell_acc = None
        rule_acc = None
        row_acc = None
        col_acc = None
        box_acc = None
        wrong_per_grid = None
        digit_dist = []

        if evaluator is not None:
            try:
                s_eval = evaluator.eval_images(rendered.clamp(-1, 1))
                pred_grid = s_eval["discrete"].long()  # (B, 9, 9) in [1,9]

                # ── cell accuracy (unknown positions only) ──
                known_mask_2d = known_mask.view(B, grid_hw, grid_hw)
                unknown_mask_2d = ~known_mask_2d
                n_unknown_cells = unknown_mask_2d.float().sum()
                if n_unknown_cells > 0:
                    cell_acc = (
                        (pred_grid == gt_9x9) & unknown_mask_2d
                    ).float().sum() / n_unknown_cells
                    cell_acc = cell_acc.item()
                else:
                    cell_acc = 1.0

                # ── full-grid cell accuracy ──
                full_cell_acc = (pred_grid == gt_9x9).float().mean().item()
                wrong_per_grid = (
                    pred_grid != gt_9x9
                ).float().sum(dim=(1, 2)).mean().item()

                # ── sudoku rule check ──
                n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(
                    pred_grid)
                rule_acc = n_valid / max(n_total, 1)
                row_acc  = n_row  / max(n_total, 1)
                col_acc  = n_col  / max(n_total, 1)
                box_acc  = n_box  / max(n_total, 1)

                # ── digit distribution ──
                vals = pred_grid.flatten()
                for v in range(1, 10):
                    cnt = (vals == v).sum().item()
                    digit_dist.append(
                        f"  digit {v}: {cnt} "
                        f"({cnt / vals.numel() * 100:.1f}%)")

                # ── logging ──
                prefix = f"eval/{level_name}"
                log_dict[f"{prefix}/cell_acc_unknown"] = cell_acc
                log_dict[f"{prefix}/cell_acc"] = full_cell_acc
                log_dict[f"{prefix}/wrong_per_grid"] = wrong_per_grid
                log_dict[f"{prefix}/rule_acc"] = rule_acc
                log_dict[f"{prefix}/row_acc"] = row_acc
                log_dict[f"{prefix}/col_acc"] = col_acc
                log_dict[f"{prefix}/box_acc"] = box_acc
                log_dict[f"{prefix}/avg_hints"] = avg_hints
                log_dict[f"{prefix}/tok_acc_unk"] = tok_acc_unk
                log_dict[f"{prefix}/tok_acc_all"] = tok_acc_all

                summary_lines.append(
                    f"  {level_name:6s}  hints={avg_hints:5.1f}  "
                    f"cell_acc(unk)={cell_acc:.4f}  "
                    f"cell_acc(all)={full_cell_acc:.4f}  "
                    f"rule_acc={rule_acc:.4f}  "
                    f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})  "
                    f"tok_acc(unk)={tok_acc_unk:.4f}")

                # ── save: digit-grid images (sampled) ──
                sample_imgs = [
                    render_digit_grid(pred_grid[i])
                    for i in range(n_vis)]
                canvas = tile_images(sample_imgs, nrow=nrow)
                if canvas is not None:
                    canvas.save(os.path.join(
                        task_dir, f"step_{step:07d}_sampled.png"))

                # ── save: GT digit-grid images ──
                gt_imgs = [
                    render_digit_grid(gt_9x9[i])
                    for i in range(min(gt_9x9.shape[0], n_vis))]
                gt_canvas = tile_images(gt_imgs, nrow=nrow)
                if gt_canvas is not None:
                    gt_canvas.save(os.path.join(
                        task_dir, f"step_{step:07d}_gt.png"))

                # ── save: comparison (wrong filled cells in red) ──
                cmp_imgs = []
                for i in range(n_vis):
                    hint_2d = known_mask_2d[i]
                    wrong = (pred_grid[i] != gt_9x9[i])
                    cmp_imgs.append(
                        render_digit_grid(
                            pred_grid[i],
                            wrong_mask=wrong & ~hint_2d,
                        ))
                cmp_canvas = tile_images(cmp_imgs, nrow=nrow)
                if cmp_canvas is not None:
                    cmp_canvas.save(os.path.join(
                        task_dir, f"step_{step:07d}_cmp.png"))

                # detailed wrong-cell for easy level
                if level_name == "easy":
                    wrong_mask_all = (pred_grid != gt_9x9)
                    total_wrong = int(wrong_mask_all.sum().item())
                    if total_wrong < 20:
                        accelerator.print(
                            f"[eval/{level_name}] wrong cells={total_wrong}")
                        wm = wrong_mask_all.detach().cpu()
                        gt_cpu = gt_9x9.detach().cpu()
                        pr_cpu = pred_grid.detach().cpu()
                        for bi in range(min(8, wm.shape[0])):
                            coords = torch.nonzero(wm[bi], as_tuple=False)
                            if coords.numel() == 0:
                                continue
                            accelerator.print(
                                f"  sample#{bi} wrong={coords.shape[0]}")
                            for r, c in coords.tolist():
                                accelerator.print(
                                    f"    (r={r},c={c}) "
                                    f"GT={int(gt_cpu[bi,r,c])} "
                                    f"PRED={int(pr_cpu[bi,r,c])}")

            except Exception as e:
                accelerator.print(
                    f"[eval/{level_name}] Sudoku eval failed: {e}")
                summary_lines.append(
                    f"  {level_name:6s}  hints={avg_hints:5.1f}  "
                    f"tok_acc(unk)={tok_acc_unk:.4f}  "
                    f"tok_acc(all)={tok_acc_all:.4f}  "
                    f"(sudoku eval failed)")
        else:
            summary_lines.append(
                f"  {level_name:6s}  hints={avg_hints:5.1f}  "
                f"tok_acc(unk)={tok_acc_unk:.4f}  "
                f"tok_acc(all)={tok_acc_all:.4f}  "
                f"(no evaluator)")

        # ── save: step logs (sampling trajectory) ──
        if step_logs is not None:
            step_logs_path = os.path.join(
                task_dir, f"step_{step:07d}_sampling_logs.txt")
            with open(step_logs_path, "w") as f:
                f.write(f"Sampling logs for {level_name} level\n")
                f.write("=" * 80 + "\n\n")
                for log_entry in step_logs:
                    step_idx = log_entry['step']
                    t_val = log_entry['t']
                    if isinstance(t_val, torch.Tensor):
                        t_val = t_val.item()
                    n_masked_list = log_entry['n_masked']
                    n_masked_avg = sum(n_masked_list) / len(n_masked_list)
                    f.write(f"Step {step_idx:3d}: t={t_val:.6f}  "
                            f"n_masked={n_masked_avg:.1f}/{seq_len}\n")

        # ── save: text details ──
        txt_path = os.path.join(
            task_dir, f"step_{step:07d}_details.txt")
        with open(txt_path, "w") as f:
            f.write(f"step={step}  level={level_name}  "
                    f"samples={B}  avg_hints={avg_hints:.1f}\n")
            f.write(f"tok_acc(unknown)={tok_acc_unk:.6f}\n")
            f.write(f"tok_acc(all)={tok_acc_all:.6f}\n")
            if cell_acc is not None:
                f.write(f"cell_acc(unknown)={cell_acc:.6f}\n")
            if full_cell_acc is not None:
                f.write(f"cell_acc(all)={full_cell_acc:.6f}\n")
            if wrong_per_grid is not None:
                f.write(f"wrong_per_grid={wrong_per_grid:.2f}\n")
            if rule_acc is not None:
                f.write(f"rule_acc={rule_acc:.6f}  "
                        f"row={row_acc:.6f}  col={col_acc:.6f}  "
                        f"box={box_acc:.6f}\n")
            if digit_dist:
                f.write("digit distribution:\n")
                for dd in digit_dist:
                    f.write(dd + "\n")
            f.write(f"\nfirst 8 completed tok_ids (flat):\n")
            for i in range(min(8, B)):
                f.write(f"  sample {i}: {completed_tok[i].tolist()}\n")

        accelerator.print(
            f"[eval/{level_name}] step={step}  "
            f"cell_acc(unk)={cell_acc if cell_acc is not None else 'N/A'}  "
            f"rule_acc={rule_acc if rule_acc is not None else 'N/A'}  "
            f"tok_acc(unk)={tok_acc_unk:.4f}  "
            f"saved → {task_dir}/")

    # ── overall summary ──
    for line in summary_lines:
        accelerator.print(line)
    if args.log_with and log_dict:
        accelerator.log(log_dict, step=step)

    # save combined summary
    summary_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(summary_dir, exist_ok=True)
    txt_path = os.path.join(summary_dir, f"step_{step:07d}_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    if ema is not None:
        ema.restore(params)
    model.train()


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── save run config ──
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({"cmd": " ".join(sys.argv), "args": vars(args)},
                  f, indent=2, sort_keys=True)

    # ── accelerator ──
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=project_config,
    )
    accelerator.print("=" * 60)
    accelerator.print("Sudoku Grid Discrete Diffusion Training (MDLM-style)")
    accelerator.print("=" * 60)

    if accelerator.is_main_process and args.log_with is not None:
        # tensorboard add_hparams only accepts int/float/str/bool
        sanitized_cfg = {}
        for k, v in vars(args).items():
            if v is None:
                sanitized_cfg[k] = "None"
            elif isinstance(v, (list, tuple)):
                sanitized_cfg[k] = str(v)
            else:
                sanitized_cfg[k] = v
        accelerator.init_trackers(
            project_name="sudoku_discrete_diffusion",
            config=sanitized_cfg,
        )

    # ─────────────────────────────────────────────────────────
    #  Dataset  (SRM sudoku — same as main.py)
    # ─────────────────────────────────────────────────────────
    accelerator.print(f"[data] Loading SRM dataset config: {args.sudoku_config}")
    cfg = OmegaConf.load(args.sudoku_config)

    srm_ds_cfg = cfg.SRM_dataset_cfg
    srm_cond_cfg = cfg.SRM_conditioning_cfg

    train_dataset_raw = get_dataset(srm_ds_cfg, srm_cond_cfg, "train")
    validation_dataset_raw = get_dataset(srm_ds_cfg, srm_cond_cfg, "val")

    # ─────────────────────────────────────────────────────────
    #  image_cond_mode: load ConditionalUNet + cache tokens
    # ─────────────────────────────────────────────────────────
    cond_unet = None
    image_token_vocab_size = 0
    image_token_seq_len = 0
    train_cached_tok_ids = None
    val_cached_tok_ids = None
    val_grids = None

    if args.image_cond_mode:
        # 1) Load full ConditionalUNet (frozen)
        cond_unet, image_token_vocab_size, image_token_seq_len = (
            load_cond_unet(args, accelerator)
        )

        # 2) Cache all tok_ids (run encoder once; disk-cached if --token_cache_dir)
        train_cache_path = None
        val_cache_path = None
        if args.token_cache_dir is not None:
            os.makedirs(args.token_cache_dir, exist_ok=True)
            train_cache_path = os.path.join(args.token_cache_dir, "train_tok_ids.pt")
            val_cache_path = os.path.join(args.token_cache_dir, "val_tok_ids.pt")

        accelerator.print("[cache] Caching train tok_ids...")
        train_cached_tok_ids = cache_all_tokens(
            cond_unet, train_dataset_raw, accelerator.device,
            batch_size=64, accelerator=accelerator,
            cache_path=train_cache_path)

        accelerator.print("[cache] Caching val tok_ids...")
        val_cached_tok_ids = cache_all_tokens(
            cond_unet, validation_dataset_raw, accelerator.device,
            batch_size=64, accelerator=accelerator,
            cache_path=val_cache_path)

        # 3) Get grids from raw dataset
        train_grids = train_dataset_raw.sudoku_grids.long()  # (N, 9, 9)
        val_grids = validation_dataset_raw.sudoku_grids.long()

        # 4) Create CachedTokenDataset
        train_dataset = CachedTokenDataset(train_cached_tok_ids, train_grids)
        validation_dataset = CachedTokenDataset(val_cached_tok_ids, val_grids)

        accelerator.print(
            f"[data] Cached dataset: Train={len(train_dataset)}, "
            f"Val={len(validation_dataset)}")

    elif args.grid_only:
        accelerator.print("[data] --grid_only: skipping image loading")
        train_dataset = GridOnlyDataset(train_dataset_raw)
        validation_dataset = GridOnlyDataset(validation_dataset_raw)
    else:
        train_dataset = train_dataset_raw
        validation_dataset = validation_dataset_raw

    accelerator.print(f"[data] Train size: {len(train_dataset)}, "
                      f"Val size: {len(validation_dataset)}")

    # sanity check
    sample0 = train_dataset[0]
    if args.image_cond_mode:
        assert "tok_ids" in sample0, \
            f"Expected dict with 'tok_ids', got keys={list(sample0.keys())}"
        accelerator.print(
            f"[data] tok_ids shape={sample0['tok_ids'].shape}, "
            f"range=[{sample0['tok_ids'].min().item()}, "
            f"{sample0['tok_ids'].max().item()}]")
    else:
        assert isinstance(sample0, dict) and "grid" in sample0, \
            f"Expected dict with 'grid', got {type(sample0)}"
        accelerator.print(
            f"[data] grid shape={sample0['grid'].shape}, "
            f"dtype={sample0['grid'].dtype}, "
            f"range=[{sample0['grid'].min().item()}, "
            f"{sample0['grid'].max().item()}]")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ─────────────────────────────────────────────────────────
    #  Model
    # ─────────────────────────────────────────────────────────
    grid_hw = args.grid_hw
    seq_len_grid = grid_hw * grid_hw                  # 81

    if args.image_cond_mode:
        # target = image cond-embedding tok_ids (unconditional generation)
        data_vocab_size = image_token_vocab_size
        seq_len = image_token_seq_len
        accelerator.print(
            f"[image_cond] Training on cond-embedding tokens: "
            f"vocab={data_vocab_size}, seq_len={seq_len}")
    else:
        # target = grid digits
        data_vocab_size = args.grid_vocab_size        # 9 (0-8 representing 1-9)
        seq_len = seq_len_grid                        # 81

    backbone_vocab_size = data_vocab_size + 1     # +1 for [MASK]

    backbone = DIT(
        vocab_size=backbone_vocab_size,
        seq_len=seq_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        cond_dim=args.cond_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.model_dropout,
        pos_emb_type=args.pos_emb_type,
        factorized_head=args.factorized_head,
        fsq_levels=fsq_levels if args.factorized_head else None,
        ar_head_dim=args.ar_head_dim,
        ar_head_layers=args.ar_head_layers,
    )

    # ── optionally initialize token_emb from FSQ codebook ──
    if args.image_cond_mode and args.init_embed_from_fsq and cond_unet is not None:
        discretizer = cond_unet.discretizer
        all_ids = torch.arange(data_vocab_size, device="cpu")
        with torch.no_grad():
            # (V, slot_dim)  e.g. (4096, 128)  — runs on CPU
            codebook_vecs = discretizer.decode(all_ids.unsqueeze(0)).squeeze(0)
        slot_dim = codebook_vecs.shape[-1]  # 128
        hidden = args.hidden_size           # 256

        # project codebook → hidden_size  (simple linear, init once)
        proj = nn.Linear(slot_dim, hidden, bias=False).to(codebook_vecs.device)
        nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            projected = proj(codebook_vecs)  # (V, hidden)
        backbone.token_emb.weight.data[:data_vocab_size] = projected.cpu()
        # mask token (index = data_vocab_size) stays random
        accelerator.print(
            f"[init] Initialized token_emb[:{data_vocab_size}] from FSQ "
            f"codebook ({slot_dim}→{hidden}), "
            f"mask token [{data_vocab_size}] left random")

    diffusion = DiscreteDiffusion(
        backbone=backbone,
        vocab_size=data_vocab_size,
        noise_type=args.noise_type,
        noise_eps=args.noise_eps,
        antithetic_sampling=args.antithetic_sampling,
        importance_sampling=args.importance_sampling,
        change_of_variables=args.change_of_variables,
        sampling_eps=args.sampling_eps,
    )

    total_p, train_p = count_params(diffusion)
    accelerator.print(
        f"[model] DiT: hidden={args.hidden_size}, heads={args.n_heads}, "
        f"blocks={args.n_blocks}, seq_len={seq_len}, "
        f"data_vocab={data_vocab_size}, mask_idx={diffusion.mask_index}")
    accelerator.print(
        f"[model] Total params: {format_n(total_p)} "
        f"(trainable {format_n(train_p)})")
    accelerator.print(
        f"[diffusion] noise={args.noise_type}, "
        f"antithetic={args.antithetic_sampling}, "
        f"importance={args.importance_sampling}, "
        f"cov={args.change_of_variables}")

    # ── optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_train_steps)

    # ── prepare ──
    diffusion, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        diffusion, optimizer, train_loader, lr_scheduler
    )

    # ── EMA ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(diffusion).parameters(),
                  decay=args.ema_decay)
        ema.to(accelerator.device)
        accelerator.print(f"[ema] decay = {args.ema_decay}")

    # ── resume ──
    global_step = 0
    if args.resume_dir and os.path.isdir(args.resume_dir):
        accelerator.load_state(args.resume_dir)
        global_step = parse_step_from_dir(args.resume_dir)
        accelerator.print(f"[resume] Resumed from {args.resume_dir}, step={global_step}")

    # ── save config ──
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ════════════════════════════════════════════════════════
    #  Training loop
    # ════════════════════════════════════════════════════════
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.grad_accum_steps)
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    accelerator.print(
        f"\n[train] Starting for {args.max_train_steps} steps "
        f"(~{num_epochs} epochs) ...\n")
    diffusion.train()
    epoch = 0
    running_loss = 0.0

    while global_step < args.max_train_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            # ── batch → x0 token sequence ──
            if args.image_cond_mode:
                # x0 = cached tok_ids (already computed at startup)
                tokens = batch["tok_ids"].to(accelerator.device).long()
            else:
                # x0 = grid digits (data is 1-9, convert to 0-8 for model)
                grid = batch["grid"].to(accelerator.device).long()
                tokens = grid.view(grid.shape[0], -1) - 1   # (B, 81) in range [0, 8]

            with accelerator.accumulate(diffusion):
                # ── Conditional training: prefix conditioning ──
                cond_tokens = None
                if args.conditional_training:
                    B, L = tokens.shape
                    model_ = accelerator.unwrap_model(diffusion)
                    mask_token = model_.mask_index
                    mask_ratio = torch.empty(B, 1, device=tokens.device).uniform_(0.0, 1.0)
                    rand = torch.rand(B, L, device=tokens.device)
                    mask = rand < mask_ratio  # True → masked in condition
                    cond_prefix = tokens.clone()
                    cond_prefix[mask] = mask_token
                    # Embed via backbone's own token_emb + pos_emb → (B, L, D)
                    backbone = model_.backbone
                    cond_tokens = backbone.token_emb(cond_prefix)
                    cond_tokens = backbone.pos_emb(cond_tokens)

                loss_out = accelerator.unwrap_model(diffusion).compute_loss(
                    tokens, cond_tokens=cond_tokens)
                loss = loss_out.loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # EMA update
            if ema is not None:
                ema.update(accelerator.unwrap_model(diffusion).parameters())

            global_step += 1
            running_loss += loss.item()
            progress_bar.update(1)
            loss_val = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

            # ── logging ──
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"[step {global_step:>7d}]  epoch={epoch}  "
                    f"loss={avg_loss:.4f}  lr={lr:.2e}")
                if accelerator.is_main_process and args.log_with:
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/lr": lr,
                    }, step=global_step)
                if accelerator.is_main_process:
                    model = accelerator.unwrap_model(diffusion)
                    with torch.no_grad():
                        debug_x0 = tokens[:1]
                        t_dbg = model._sample_t(1, debug_x0.device)
                        if model.change_of_variables:
                            f_T = torch.log1p(-torch.exp(-model.noise.sigma_max))
                            f_0 = torch.log1p(-torch.exp(-model.noise.sigma_min))
                            move_chance = torch.exp(f_0 + t_dbg * (f_T - f_0))
                            sigma_dbg = t_dbg[:, None]
                        else:
                            sigma_dbg, _ = model.noise(t_dbg)
                            move_chance = 1 - torch.exp(-sigma_dbg)
                            sigma_dbg = sigma_dbg[:, None]
                        xt_dbg = model.q_xt(debug_x0, move_chance[:, None])
                        log_p = model.forward(xt_dbg, sigma_dbg)
                        pred = log_p.argmax(dim=-1)
                        
                        # Check accuracy on MASKED positions only
                        is_masked = (xt_dbg == model.mask_index)
                        n_masked = is_masked.sum().item()
                        seq_len_dbg = debug_x0.shape[1]
                        if n_masked > 0:
                            masked_correct = ((pred == debug_x0) & is_masked).sum().item()
                            masked_acc = masked_correct / n_masked
                        else:
                            masked_acc = 1.0
                        
                        # Calculate confidence
                        probs = torch.softmax(log_p, dim=-1)
                        confidence = probs.max(dim=-1).values
                        if n_masked > 0:
                            masked_conf = confidence[is_masked].mean().item()
                            masked_conf_min = confidence[is_masked].min().item()
                            masked_conf_max = confidence[is_masked].max().item()
                        else:
                            masked_conf = masked_conf_min = masked_conf_max = 0.0
                        
                        accelerator.print(
                            f"[debug] t={t_dbg.item():.4f} "
                            f"move_chance={move_chance.item():.4f} "
                            f"n_masked={n_masked}/{seq_len_dbg} "
                            f"masked_acc={masked_acc:.3f} "
                            f"masked_conf=[{masked_conf_min:.3f}, {masked_conf:.3f}, {masked_conf_max:.3f}]")

                        # grid debug only for grid mode (not image_cond)
                        if not args.image_cond_mode:
                            accelerator.print("[debug] x0 grid:")
                            for row in range(9):
                                start = row * 9
                                end = start + 9
                                accelerator.print(
                                    " ".join(str(v + 1) for v in debug_x0[0, start:end].tolist()))
                            accelerator.print("[debug] xt grid:")
                            for row in range(9):
                                start = row * 9
                                end = start + 9
                                accelerator.print(
                                    " ".join(str(v + 1) for v in xt_dbg[0, start:end].tolist()))
                            accelerator.print("[debug] pred grid:")
                            for row in range(9):
                                start = row * 9
                                end = start + 9
                                accelerator.print(
                                    " ".join(str(v + 1) for v in pred[0, start:end].tolist()))
                running_loss = 0.0

            # ── eval ──
            if global_step % args.eval_every == 0:
                evaluate_and_save(
                    diffusion, global_step, args, accelerator, ema,
                    val_dataset=validation_dataset,
                    cond_unet=cond_unet,
                    val_cached_tok_ids=val_cached_tok_ids,
                    val_grids=val_grids,
                    val_dataset_raw=validation_dataset_raw,
                )
                diffusion.train()

            # ── save ──
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, "ckpt",
                                        f"step{global_step}")
                accelerator.save_state(ckpt_dir)
                accelerator.print(f"[save] Checkpoint → {ckpt_dir}")
                if accelerator.is_main_process:
                    meta = {"epoch": epoch, "global_step": global_step,
                            "args": vars(args)}
                    os.makedirs(ckpt_dir, exist_ok=True)
                    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2, sort_keys=True)

        if global_step >= args.max_train_steps:
            break

    # ── final save ──
    ckpt_dir = os.path.join(args.output_dir, "ckpt", f"step{global_step}_final")
    accelerator.save_state(ckpt_dir)
    accelerator.print(f"[save] Final checkpoint → {ckpt_dir}")

    # ── final eval ──
    evaluate_and_save(
        diffusion, global_step, args, accelerator, ema,
        val_dataset=validation_dataset,
        cond_unet=cond_unet,
        val_cached_tok_ids=val_cached_tok_ids,
        val_grids=val_grids,
        val_dataset_raw=validation_dataset_raw,
    )
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
