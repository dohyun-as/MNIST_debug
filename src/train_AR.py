#!/usr/bin/env python
"""
train_AR.py
===========
Autoregressive (AR) sudoku generation training & evaluation.

Two modes controlled by ``--conditional``:

1. **Unconditional** (default):
   Standard causal language model on flattened 9×9 sudoku grids (81 tokens,
   digits 0-8 representing 1-9).  Generates complete sudoku grids from
   scratch by sampling left-to-right.

2. **Conditional** (``--conditional``):
   Prefix-conditioned causal model.  During training a random fraction of
   cells is masked (token 9 = MASK).  The masked condition grid (81 tokens)
   is prepended to the full target grid (81 tokens), giving a 162-token
   sequence.  Causal attention lets each target token attend to the entire
   condition plus all preceding target tokens.

   At inference the conditional model can also do unconditional generation
   by using an all-MASK condition prefix.

Evaluation:
  - Unconditional generation → sudoku rule check
  - Per-difficulty inpainting (easy / medium / hard) → cell accuracy + rules
    (only for conditional models)

Uses the same SRM sudoku dataset as ``train_discrete_diffusion.py``.

Usage:
    # unconditional
    python train_AR.py --sudoku_config ../config/sudoku_config.json \\
        --grid_only --output_dir ./outputs/ar_uncond

    # conditional
    python train_AR.py --sudoku_config ../config/sudoku_config.json \\
        --grid_only --conditional --output_dir ./outputs/ar_cond
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

# SRM dataset
from SRM.datasets import DatasetCfg, get_dataset, get_dataset_class
from SRM.type_extensions import ConditioningCfg


# ────────────────────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────────────────────

NUM_DIGITS = 9               # tokens 0-8 representing digits 1-9
MASK_TOKEN = NUM_DIGITS       # = 9, used in condition prefix
VOCAB_SIZE = NUM_DIGITS + 1   # = 10 (digits + MASK)
GRID_LEN   = 81              # 9 × 9

DIFFICULTY_LEVELS = {
    "hard":   (0,  26),       # 0–26 given cells → model fills 55–81
    "medium": (27, 53),       # 27–53 given cells → model fills 28–54
    "easy":   (54, 80),       # 54–80 given cells → model fills 1–27
}


# ────────────────────────────────────────────────────────────
#  Model components
# ────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class CausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, dim: int, n_heads: int,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # ── self-attention with causal mask ──
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.unbind(dim=2)      # each (B, L, H, Dh)
        q = q.transpose(1, 2)             # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))

        # ── MLP ──
        h = self.norm2(x)
        x = x + self.dropout(self.mlp(h))
        return x


class ARTransformer(nn.Module):
    """Causal Transformer for autoregressive sudoku generation.

    When ``conditional=True`` the model expects 162-token inputs
    (81 condition + 81 target).  The two halves share the same 81-position
    grid positional embeddings and are distinguished by a learned segment
    embedding.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_blocks: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        conditional: bool = False,
        pos_emb_type: str = "2d",
        sudoku_hw: int = 9,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.conditional = conditional
        self.sudoku_hw = sudoku_hw
        self.grid_len = sudoku_hw * sudoku_hw    # 81
        self.pos_emb_type = pos_emb_type

        # ── embeddings ──
        self.token_emb = nn.Embedding(vocab_size, hidden_size)

        # learned 1-D positional embedding (shared for cond & target halves)
        self.pos_emb = nn.Embedding(self.grid_len, hidden_size)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)

        # segment embedding (conditional mode only)
        if conditional:
            self.segment_emb = nn.Embedding(2, hidden_size)

        # 2-D / sudoku positional embeddings
        if pos_emb_type in ("2d", "sudoku"):
            self.row_emb = nn.Embedding(sudoku_hw, hidden_size)
            self.col_emb = nn.Embedding(sudoku_hw, hidden_size)
        if pos_emb_type == "sudoku":
            self.box_emb = nn.Embedding(sudoku_hw, hidden_size)

        # ── transformer ──
        self.blocks = nn.ModuleList([
            CausalBlock(hidden_size, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.ln_f = LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    # ────────────────────────────────────────────
    def _grid_pos_emb(self, grid_pos: torch.Tensor) -> torch.Tensor:
        """Return summed positional embeddings for *grid_pos* (0-80)."""
        h = self.pos_emb(grid_pos)
        if self.pos_emb_type in ("2d", "sudoku"):
            rows = grid_pos // self.sudoku_hw
            cols = grid_pos % self.sudoku_hw
            h = h + self.row_emb(rows) + self.col_emb(cols)
        if self.pos_emb_type == "sudoku":
            rows = grid_pos // self.sudoku_hw
            cols = grid_pos % self.sudoku_hw
            boxes = (rows // 3) * 3 + (cols // 3)
            h = h + self.box_emb(boxes)
        return h

    # ────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args
        ----
        x           : (B, L) int64 token indices
        segment_ids : (B, L) or (L,) — 0 for condition, 1 for target

        Returns
        -------
        logits : (B, L, vocab_size)
        """
        B, L = x.shape

        h = self.token_emb(x.long())                          # (B, L, D)

        # grid-relative position: pos % grid_len
        grid_pos = torch.arange(L, device=x.device) % self.grid_len
        h = h + self._grid_pos_emb(grid_pos).unsqueeze(0)     # broadcast

        if self.conditional and segment_ids is not None:
            seg = segment_ids
            if seg.dim() == 1:
                seg = seg.unsqueeze(0).expand(B, -1)
            h = h + self.segment_emb(seg.long())

        for blk in self.blocks:
            h = blk(h)

        h = self.ln_f(h)
        return self.head(h)                                    # (B, L, V)

    # ────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        condition: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        device: str | torch.device = "cuda",
        return_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Autoregressive sampling.

        Unconditional model
            Generates 81 tokens from scratch.  The first token is sampled
            uniformly; the rest are generated autoregressively.

        Conditional model
            *condition* is (B, 81) with MASK / digit values.  The model
            generates 81 target tokens conditioned on this prefix.  When
            *condition* is ``None`` an all-MASK prefix is used (≈ uncond).

        Returns
        -------
        grids : (B, 81) int64 — predicted digit tokens (0-8)
        history : list[(B, 81)] — if return_history, snapshots after each token
        """
        self.eval()
        B = batch_size
        history: list[torch.Tensor] = []
        EMPTY = MASK_TOKEN  # use MASK as "empty" placeholder in history

        if self.conditional:
            # ── conditional generation ──
            if condition is None:
                cond = torch.full((B, self.grid_len), MASK_TOKEN,
                                  dtype=torch.long, device=device)
            else:
                cond = condition.to(device).long()
                B = cond.shape[0]
            seq = cond.clone()                     # start with condition prefix

            if return_history:
                # first frame: show given cells already filled
                snap = cond.clone()  # given cells visible, MASK for rest
                history.append(snap.cpu())

            for t in range(self.grid_len):
                L = seq.shape[1]
                seg = torch.zeros(B, L, dtype=torch.long, device=device)
                seg[:, self.grid_len:] = 1
                logits = self.forward(seq, segment_ids=seg)     # (B, L, V)
                next_logits = logits[:, -1, :NUM_DIGITS]        # digits only
                next_token = _sample_token(next_logits, temperature, top_k, top_p)
                seq = torch.cat([seq, next_token], dim=1)

                if return_history:
                    snap = cond.clone()  # always keep given cells
                    generated = seq[:, self.grid_len:]           # tokens so far
                    snap[:, :generated.shape[1]] = generated
                    history.append(snap.cpu())

            target = seq[:, self.grid_len:]        # (B, 81)
            if return_history:
                return target, history
            return target

        else:
            # ── unconditional generation ──
            first = torch.randint(0, NUM_DIGITS, (B, 1), device=device)
            seq = first

            if return_history:
                snap = torch.full((B, self.grid_len), EMPTY,
                                  dtype=torch.long, device=device)
                snap[:, 0] = first.squeeze(1)
                history.append(snap.cpu())

            for t in range(self.grid_len - 1):
                logits = self.forward(seq)           # (B, L, V)
                next_logits = logits[:, -1, :NUM_DIGITS]
                next_token = _sample_token(next_logits, temperature, top_k, top_p)
                seq = torch.cat([seq, next_token], dim=1)

                if return_history:
                    snap = torch.full((B, self.grid_len), EMPTY,
                                      dtype=torch.long, device=device)
                    snap[:, :seq.shape[1]] = seq
                    history.append(snap.cpu())

            if return_history:
                return seq, history              # (B, 81), list[(B,81)]
            return seq                              # (B, 81)


# ────────────────────────────────────────────────────────────
#  Sampling helpers
# ────────────────────────────────────────────────────────────

def _sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Sample a single token per batch element.

    Args:
        logits: (B, V) raw logits
        temperature: softmax temperature (0 → greedy)
        top_k: keep only top-k logits (0 = no filtering)
        top_p: nucleus sampling threshold (0 = disabled)

    Returns:
        (B, 1) sampled token indices
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = logits.topk(top_k, dim=-1)
        logits[logits < vals[:, -1:]] = float("-inf")

    # nucleus (top-p) filtering
    if top_p > 0.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # remove tokens with cumulative probability above threshold
        mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        # scatter back
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


# ────────────────────────────────────────────────────────────
#  EMA
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
#  GridOnlyDataset  (same as train_discrete_diffusion.py)
# ────────────────────────────────────────────────────────────

class GridOnlyDataset(torch.utils.data.Dataset):
    """Wraps an SRM dataset and returns only the sudoku grid."""

    def __init__(self, inner_dataset):
        self.grids = inner_dataset.sudoku_grids   # (N, 9, 9) tensor

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return {"grid": self.grids[idx].long()}


# ────────────────────────────────────────────────────────────
#  Utility functions
# ────────────────────────────────────────────────────────────

def count_params(module: nn.Module):
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


def parse_step_from_dir(path: str) -> int:
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("step"):
        try:
            return int(base.replace("step", "").replace("_final", ""))
        except Exception:
            pass
    return 0


# ────────────────────────────────────────────────────────────
#  Sudoku rule checker  (same as train_discrete_diffusion.py)
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

    Returns (num_valid, num_row_valid, num_col_valid, num_box_valid, total).
    """
    B = grids.shape[0]
    valid = row_valid_cnt = col_valid_cnt = box_valid_cnt = 0

    for b in range(B):
        g = grids[b].cpu()
        ok_row = ok_col = ok_box = True

        for r in range(9):
            if len(set(g[r].tolist())) != 9:
                ok_row = False
                break
        if ok_row:
            row_valid_cnt += 1

        for c in range(9):
            if len(set(g[:, c].tolist())) != 9:
                ok_col = False
                break
        if ok_col:
            col_valid_cnt += 1

        for br in range(3):
            for bc in range(3):
                box = g[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten().tolist()
                if len(set(box)) != 9:
                    ok_box = False
                    break
            if not ok_box:
                break
        if ok_box:
            box_valid_cnt += 1

        if ok_row and ok_col and ok_box:
            valid += 1

    return valid, row_valid_cnt, col_valid_cnt, box_valid_cnt, B


# ────────────────────────────────────────────────────────────
#  Visualization helpers
# ────────────────────────────────────────────────────────────

def render_digit_grid(grid_2d, wrong_mask=None,
                      cell=34, pad=3, border=3, font_size=18):
    """Render a 2-D digit grid as a PIL image."""
    from PIL import Image, ImageDraw, ImageFont

    grid_np = grid_2d.detach().cpu().numpy() if torch.is_tensor(grid_2d) else grid_2d
    H, W = grid_np.shape
    img_w = W * cell + 2 * pad
    img_h = H * cell + 2 * pad
    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
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
            draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2),
                      s, fill=(0, 0, 0), font=font)
            if wrong_mask is not None and bool(wrong_mask[r, c]):
                draw.rectangle([x1, y1, x2, y2],
                               outline=(255, 0, 0), width=border)
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
        canvas.paste(im, (pad_px + cc * (w + pad_px),
                          pad_px + rr * (h + pad_px)))
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
        for v in range(grid_hw):  # values 0..8
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


def render_ar_sampling_video(
    history: list[torch.Tensor],
    sample_idx: int,
    grid_hw: int,
    empty_token: int,
    save_path: str,
    condition: torch.Tensor | None = None,
    max_frames: int = 82,
    frame_duration_ms: int = 200,
    cell: int = 34,
    pad: int = 3,
    font_size: int = 18,
    save_format: str = "mp4",
):
    """Render AR sampling as GIF/MP4.

    Each frame shows the 9×9 grid with tokens generated so far.
    Empty cells are shown as gray dots; newly placed tokens are blue.
    Condition-given cells (if any) are shown in green from the start.
    On the final frame, rule-violating cells are highlighted in red.
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

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
    if n_total > max_frames:
        indices = [0] + list(
            range(1, n_total - 1,
                  max(1, (n_total - 2) // (max_frames - 2)))
        ) + [n_total - 1]
        indices = sorted(set(indices))
    else:
        indices = list(range(n_total))

    img_w = grid_hw * cell + 2 * pad
    img_h = grid_hw * cell + 2 * pad + 24
    frames = []

    # condition mask: which cells are given (for conditional mode)
    cond_given = None
    if condition is not None:
        cond_flat = condition[sample_idx].cpu()
        cond_given = (cond_flat != empty_token)  # (81,)

    # pre-compute rule violations on the final grid for the last frame
    final_seq = history[-1][sample_idx]
    final_grid_np = final_seq.view(grid_hw, grid_hw).numpy()
    is_complete = int((final_seq != empty_token).sum().item()) == grid_hw * grid_hw
    violations = _find_rule_violations(final_grid_np, grid_hw) if is_complete else None

    prev_grid = None
    for frame_i, hist_idx in enumerate(indices):
        seq = history[hist_idx][sample_idx]  # (81,)
        grid = seq.view(grid_hw, grid_hw).numpy()
        is_last_frame = (hist_idx == indices[-1])

        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # step label
        n_filled = int((seq != empty_token).sum().item())
        if cond_given is not None:
            n_given = int(cond_given.sum().item())
            n_generated = n_filled - n_given
            label = f"given: {n_given}  generated: {max(0, n_generated)}/{grid_hw*grid_hw - n_given}"
        else:
            label = f"tokens: {n_filled}/{grid_hw*grid_hw}"
        draw.text((pad, img_h - 20), label, fill=(100, 100, 100), font=font)

        for r in range(grid_hw):
            for c in range(grid_hw):
                x1, y1 = pad + c * cell, pad + r * cell
                x2, y2 = x1 + cell, y1 + cell
                pos = r * grid_hw + c
                val = int(grid[r, c])

                # check rule violation on final frame
                is_violation = (is_last_frame and violations is not None
                                and bool(violations[r, c]))

                if val == empty_token:
                    # empty cell
                    draw.rectangle([x1, y1, x2, y2],
                                   fill=(245, 245, 245),
                                   outline=(200, 200, 200), width=1)
                    draw.text((x1 + cell // 2 - 3, y1 + cell // 2 - 5),
                              "·", fill=(180, 180, 180), font=font)
                else:
                    # check if newly placed
                    newly_placed = False
                    if prev_grid is not None and int(prev_grid[r, c]) == empty_token:
                        newly_placed = True

                    is_given = (cond_given is not None and
                                bool(cond_given[pos]))

                    if is_violation:
                        # rule-violating cell → red
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(255, 230, 230),
                                       outline=(220, 0, 0), width=2)
                        text_color = (200, 0, 0)
                    elif newly_placed:
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(255, 255, 255),
                                       outline=(0, 100, 255), width=2)
                        text_color = (0, 80, 200)
                    elif is_given:
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(230, 255, 230),
                                       outline=(100, 180, 100), width=1)
                        text_color = (0, 100, 0)
                    else:
                        draw.rectangle([x1, y1, x2, y2],
                                       fill=(255, 255, 255),
                                       outline=(200, 200, 200), width=1)
                        text_color = (0, 0, 0)

                    s = str(val + 1)  # 0-8 → 1-9
                    bbox = draw.textbbox((0, 0), s, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((x1 + (cell - tw) / 2, y1 + (cell - th) / 2),
                              s, fill=text_color, font=font)

        frames.append(img)
        prev_grid = grid.copy()

    if not frames:
        return

    base, _ = os.path.splitext(save_path)
    if save_format == "mp4":
        save_path = base + ".mp4"
        import numpy as np
        try:
            import imageio
        except ImportError:
            save_format = "gif"  # fallback

    if save_format == "mp4":
        import numpy as np
        import imageio
        fps = max(1, 1000 // frame_duration_ms)
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
        n_hold = max(1, fps * 3)
        np_frames.extend([np_frames[-1]] * n_hold)
        try:
            imageio.mimwrite(save_path, np_frames, format="FFMPEG",
                             fps=fps, codec="libx264",
                             pixelformat="yuv420p")
        except (ImportError, OSError):
            save_path = base + ".gif"
            save_format = "gif"

    if save_format != "mp4":
        save_path = base + ".gif"
        durations = [frame_duration_ms] * len(frames)
        durations[-1] = frame_duration_ms * 5
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=durations, loop=0,
        )


# ────────────────────────────────────────────────────────────
#  Args
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Autoregressive Sudoku Generation")

    # basic
    p.add_argument("--output_dir", type=str, default="./outputs/ar")
    p.add_argument("--seed", type=int, default=42)

    # dataset
    p.add_argument("--sudoku_config", type=str, required=True,
                   help="Path to sudoku dataset config JSON.")
    p.add_argument("--grid_only", action="store_true", default=False,
                   help="Skip image loading; only load grids.")

    # grid params
    p.add_argument("--grid_hw", type=int, default=9)
    p.add_argument("--pos_emb_type", type=str, default="2d",
                   choices=["1d", "2d", "sudoku"])

    # conditional mode
    p.add_argument("--conditional", action="store_true", default=False,
                   help="Enable conditional AR (prefix-conditioned on "
                        "partially-masked grid).")
    p.add_argument("--mask_ratio_min", type=float, default=0.0,
                   help="Min mask ratio for condition during training.")
    p.add_argument("--mask_ratio_max", type=float, default=1.0,
                   help="Max mask ratio for condition during training.")

    # model
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--model_dropout", type=float, default=0.1)

    # training
    p.add_argument("--max_train_steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA decay (0 to disable).")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for CE loss.")

    # sampling
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k sampling (0 = disabled).")
    p.add_argument("--top_p", type=float, default=0.0,
                   help="Nucleus sampling threshold (0 = disabled).")

    # logging / eval / save
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--eval_num_samples", type=int, default=64)
    p.add_argument("--eval_video_samples", type=int, default=4,
                   help="Number of samples to render as step-by-step "
                        "videos during eval (0 to disable).")
    p.add_argument("--eval_save_format", type=str, default="mp4",
                   choices=["gif", "mp4"],
                   help="Save sampling visualizations as GIF or MP4.")

    # resume
    p.add_argument("--resume_dir", type=str, default=None)

    # accelerate
    p.add_argument("--mixed_precision", type=str, default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--log_with", type=str, default=None,
                   help='"tensorboard" or "wandb"')

    return p.parse_args()


# ────────────────────────────────────────────────────────────
#  LR scheduler  (linear warmup → constant)
# ────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
#  Loss computation
# ────────────────────────────────────────────────────────────

def compute_loss(
    model: ARTransformer,
    x: torch.Tensor,
    conditional: bool,
    mask_ratio_min: float = 0.0,
    mask_ratio_max: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute the AR cross-entropy loss.

    Args:
        model: ARTransformer instance
        x: (B, 81) int64 tokens in [0, 8]
        conditional: whether to use prefix-conditioning
        mask_ratio_min/max: range for random masking
        label_smoothing: CE label smoothing

    Returns:
        loss: scalar
    """
    B = x.shape[0]
    device = x.device

    if conditional:
        # ── build masked condition prefix ──
        mask_ratio = torch.empty(B, 1, device=device).uniform_(
            mask_ratio_min, mask_ratio_max)                      # (B, 1)
        rand = torch.rand(B, GRID_LEN, device=device)           # (B, 81)
        mask = rand < mask_ratio                                 # True → masked

        cond = x.clone()
        cond[mask] = MASK_TOKEN

        full_seq = torch.cat([cond, x], dim=1)                  # (B, 162)

        # segment ids
        seg = torch.zeros(B, 2 * GRID_LEN, dtype=torch.long, device=device)
        seg[:, GRID_LEN:] = 1

        logits = model(full_seq, segment_ids=seg)                # (B, 162, V)

        # loss on target tokens only (positions 81-161)
        # logits[:, 80:161] → predicts tokens at positions 81-161 = x[:, 0:81]
        pred = logits[:, GRID_LEN - 1: 2 * GRID_LEN - 1]       # (B, 81, V)
        target = full_seq[:, GRID_LEN:]                          # (B, 81)

        loss = F.cross_entropy(
            pred.reshape(-1, model.vocab_size),
            target.reshape(-1),
            label_smoothing=label_smoothing,
        )
    else:
        # ── unconditional: standard next-token prediction ──
        logits = model(x)                                        # (B, 81, V)
        # logits[:, i] predicts x[:, i+1]
        pred = logits[:, :-1]                                    # (B, 80, V)
        target = x[:, 1:]                                        # (B, 80)

        loss = F.cross_entropy(
            pred.reshape(-1, model.vocab_size),
            target.reshape(-1),
            label_smoothing=label_smoothing,
        )

    return loss


# ────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_save(
    model: ARTransformer,
    step: int,
    args,
    accelerator: Accelerator,
    ema: EMA | None,
    val_dataset=None,
):
    """Run evaluation: unconditional generation + difficulty levels."""
    if not accelerator.is_main_process:
        return

    device = accelerator.device
    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()

    params = list(raw_model.parameters())
    if ema is not None:
        ema.store(params)
        ema.copy_to(params)

    # ── 1) unconditional generation ──
    evaluate_unconditional(raw_model, step, args, accelerator)

    # ── 2) difficulty-based evaluation (conditional model only) ──
    if args.conditional and val_dataset is not None:
        evaluate_difficulty(raw_model, step, args, accelerator, val_dataset)

    if ema is not None:
        ema.restore(params)
    raw_model.train()


# ────────────────────────────────────────────────────────────
#  Unconditional generation evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_unconditional(
    model: ARTransformer,
    step: int,
    args,
    accelerator: Accelerator,
):
    """Generate sudoku grids from scratch and evaluate validity."""
    device = accelerator.device
    n = args.eval_num_samples
    grid_hw = args.grid_hw

    save_dir = os.path.join(args.output_dir, "eval_samples", "unconditional")
    os.makedirs(save_dir, exist_ok=True)

    # ── sample ──
    if args.conditional:
        # conditional model → all-MASK condition for uncond generation
        cond = torch.full((n, GRID_LEN), MASK_TOKEN,
                          dtype=torch.long, device=device)
        grids_flat = model.generate(
            batch_size=n,
            condition=cond,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
    else:
        grids_flat = model.generate(
            batch_size=n,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )

    grids = grids_flat.view(n, grid_hw, grid_hw)                 # [0-8]

    # ── sudoku rule check ──
    n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(grids)
    rule_acc = n_valid / max(n_total, 1)
    row_acc  = n_row  / max(n_total, 1)
    col_acc  = n_col  / max(n_total, 1)
    box_acc  = n_box  / max(n_total, 1)

    # ── digit distribution ──
    vals = grids.flatten()
    dist_strs = []
    for v in range(NUM_DIGITS):
        cnt = (vals == v).sum().item()
        dist_strs.append(f"digit {v+1}: {cnt} ({cnt/vals.numel()*100:.1f}%)")

    line = (
        f"[eval/uncond] step={step}  n={n_total}  "
        f"rule_acc={rule_acc:.4f}  "
        f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})")
    accelerator.print(line)

    # ── save images ──
    nrow = min(8, int(math.sqrt(n)))
    n_vis = min(64, n)
    imgs = [render_digit_grid(grids[i] + 1) for i in range(n_vis)]
    canvas = tile_images(imgs, nrow=nrow)
    if canvas is not None:
        canvas.save(os.path.join(save_dir, f"step_{step:07d}_sampled.png"))

    # ── save text details ──
    txt_path = os.path.join(save_dir, f"step_{step:07d}_details.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step}  samples={n_total}\n")
        f.write(f"rule_acc={rule_acc:.6f}  "
                f"row={row_acc:.6f}  col={col_acc:.6f}  box={box_acc:.6f}\n")
        f.write("digit distribution:\n")
        for dd in dist_strs:
            f.write(f"  {dd}\n")
        f.write("\nfirst 8 grids (internal 0-8 values, flat):\n")
        for i in range(min(8, n)):
            f.write(f"  sample {i}: {grids_flat[i].tolist()}\n")

    # ── logging ──
    log_dict = {
        "eval/uncond/rule_acc": rule_acc,
        "eval/uncond/row_acc": row_acc,
        "eval/uncond/col_acc": col_acc,
        "eval/uncond/box_acc": box_acc,
    }
    if args.log_with:
        accelerator.log(log_dict, step=step)

    # ── save sampling videos ──
    n_vid = getattr(args, "eval_video_samples", 0)
    vid_fmt = getattr(args, "eval_save_format", "mp4")
    if n_vid > 0:
        vid_dir = os.path.join(save_dir, "videos")
        os.makedirs(vid_dir, exist_ok=True)

        # generate a small batch with history
        if args.conditional:
            vid_cond = torch.full((n_vid, GRID_LEN), MASK_TOKEN,
                                  dtype=torch.long, device=device)
            vid_grids, vid_history = model.generate(
                batch_size=n_vid, condition=vid_cond,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, device=device,
                return_history=True,
            )
        else:
            vid_grids, vid_history = model.generate(
                batch_size=n_vid,
                temperature=args.temperature, top_k=args.top_k,
                top_p=args.top_p, device=device,
                return_history=True,
            )
            vid_cond = None

        for vi in range(n_vid):
            vi_grid = vid_grids[vi].view(grid_hw, grid_hw)
            vi_valid = _check_sudoku_single(vi_grid)
            tag = "valid" if vi_valid else "invalid"
            vid_path = os.path.join(
                vid_dir, f"step_{step:07d}_sample{vi}_{tag}.{vid_fmt}")
            render_ar_sampling_video(
                history=vid_history,
                sample_idx=vi,
                grid_hw=grid_hw,
                empty_token=MASK_TOKEN,
                save_path=vid_path,
                condition=vid_cond,
                save_format=vid_fmt,
            )
        accelerator.print(
            f"[eval/uncond] Saved {n_vid} videos → {vid_dir}/")

    accelerator.print(f"[eval/uncond] saved → {save_dir}/")


# ────────────────────────────────────────────────────────────
#  Difficulty-based evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_difficulty(
    model: ARTransformer,
    step: int,
    args,
    accelerator: Accelerator,
    val_dataset,
):
    """For the conditional model: evaluate per-difficulty (easy/medium/hard).

    Reveal some GT cells as condition hints, generate target, compare.
    """
    device = accelerator.device
    grid_hw = args.grid_hw
    n_samples = min(args.eval_num_samples, len(val_dataset))

    # gather GT grids
    gt_list = []
    for idx in range(n_samples):
        sample = val_dataset[idx]
        if isinstance(sample, dict) and "grid" in sample:
            gt_list.append(sample["grid"].unsqueeze(0))
    if not gt_list:
        accelerator.print("[eval] No GT grids found, skipping difficulty eval.")
        return
    gt_grids = torch.cat(gt_list, dim=0).to(device).long()      # (B, 9, 9)
    x_gt = gt_grids.view(gt_grids.shape[0], -1) - 1             # (B, 81) [0-8]
    B = x_gt.shape[0]
    nrow = min(8, int(math.sqrt(B)))
    n_vis = min(64, B)

    log_dict = {}
    summary_lines = [f"[eval/diff] step={step}  samples={B}"]

    for level_name, (hint_lo, hint_hi) in DIFFICULTY_LEVELS.items():
        task_dir = os.path.join(args.output_dir, "eval_samples", level_name)
        os.makedirs(task_dir, exist_ok=True)

        # ── build hint mask (deterministic per level) ──
        rng = torch.Generator(device=device)
        rng.manual_seed(hash(level_name) & 0xFFFFFFFF)

        n_hints = torch.randint(
            hint_lo, hint_hi + 1, (B,), device=device, generator=rng)

        known_mask = torch.zeros(B, GRID_LEN, dtype=torch.bool, device=device)
        for b in range(B):
            nh = n_hints[b].item()
            if nh > 0:
                perm = torch.randperm(GRID_LEN, device=device, generator=rng)[:nh]
                known_mask[b, perm] = True

        # ── build condition ──
        cond = torch.full((B, GRID_LEN), MASK_TOKEN,
                          dtype=torch.long, device=device)
        cond[known_mask] = x_gt[known_mask]

        # ── generate target ──
        completed_flat = model.generate(
            batch_size=B,
            condition=cond,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )                                                        # (B, 81)

        completed_grids = completed_flat.view(B, grid_hw, grid_hw)

        # ── cell accuracy on unknown positions ──
        unknown_mask = ~known_mask
        n_unknown = unknown_mask.float().sum()
        if n_unknown > 0:
            cell_acc_unk = (
                (completed_flat == x_gt) & unknown_mask
            ).float().sum() / n_unknown
            cell_acc_unk = cell_acc_unk.item()
        else:
            cell_acc_unk = 1.0

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

        # ── logging ──
        prefix = f"eval/{level_name}"
        log_dict[f"{prefix}/cell_acc_unknown"] = cell_acc_unk
        log_dict[f"{prefix}/cell_acc"] = full_cell_acc
        log_dict[f"{prefix}/wrong_per_grid"] = wrong_per_grid
        log_dict[f"{prefix}/rule_acc"] = rule_acc
        log_dict[f"{prefix}/row_acc"] = row_acc
        log_dict[f"{prefix}/col_acc"] = col_acc
        log_dict[f"{prefix}/box_acc"] = box_acc

        summary_lines.append(
            f"  {level_name:6s}  hints={avg_hints:5.1f}  "
            f"cell_acc(unk)={cell_acc_unk:.4f}  "
            f"cell_acc(all)={full_cell_acc:.4f}  "
            f"rule_acc={rule_acc:.4f}  "
            f"(row={row_acc:.4f} col={col_acc:.4f} box={box_acc:.4f})")

        # ── save: sampled grids (display as 1-9) ──
        sample_imgs = [render_digit_grid(completed_grids[i] + 1)
                       for i in range(n_vis)]
        canvas = tile_images(sample_imgs, nrow=nrow)
        if canvas is not None:
            canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_sampled.png"))

        # ── save: GT grids ──
        gt_imgs = [render_digit_grid(gt_grids[i])
                   for i in range(min(gt_grids.shape[0], n_vis))]
        gt_canvas = tile_images(gt_imgs, nrow=nrow)
        if gt_canvas is not None:
            gt_canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_gt.png"))

        # ── save: comparison (wrong = red) ──
        cmp_imgs = []
        for i in range(n_vis):
            hint_2d = known_mask[i].view(grid_hw, grid_hw)
            wrong = (completed_grids[i] != gt_grids[i])
            cmp_imgs.append(render_digit_grid(
                completed_grids[i] + 1,
                wrong_mask=wrong & ~hint_2d))
        cmp_canvas = tile_images(cmp_imgs, nrow=nrow)
        if cmp_canvas is not None:
            cmp_canvas.save(os.path.join(
                task_dir, f"step_{step:07d}_cmp.png"))

        # ── save: text details ──
        txt_path = os.path.join(task_dir, f"step_{step:07d}_details.txt")
        with open(txt_path, "w") as f:
            f.write(f"step={step}  level={level_name}  "
                    f"samples={B}  avg_hints={avg_hints:.1f}\n")
            f.write(f"cell_acc(unknown)={cell_acc_unk:.6f}\n")
            f.write(f"cell_acc(all)={full_cell_acc:.6f}\n")
            f.write(f"wrong_per_grid={wrong_per_grid:.2f}\n")
            f.write(f"rule_acc={rule_acc:.6f}  "
                    f"row={row_acc:.6f}  col={col_acc:.6f}  "
                    f"box={box_acc:.6f}\n")
            f.write("\nfirst 8 completed grids (flat, 0-indexed):\n")
            for i in range(min(8, B)):
                f.write(f"  sample {i}: {completed_flat[i].tolist()}\n")

        # ── save: sampling videos ──
        n_vid = getattr(args, "eval_video_samples", 0)
        vid_fmt = getattr(args, "eval_save_format", "mp4")
        if n_vid > 0:
            vid_dir = os.path.join(task_dir, "videos")
            os.makedirs(vid_dir, exist_ok=True)

            vid_n = min(n_vid, B)
            vid_cond = cond[:vid_n]
            vid_completed, vid_history = model.generate(
                batch_size=vid_n,
                condition=vid_cond,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                return_history=True,
            )

            for vi in range(vid_n):
                vi_grid = vid_completed[vi].view(grid_hw, grid_hw)
                vi_gt = x_gt[vi].view(grid_hw, grid_hw)
                vi_n_wrong = int((vi_grid != vi_gt).sum().item())
                vi_valid = _check_sudoku_single(vi_grid)
                if vi_valid and vi_n_wrong == 0:
                    tag = "perfect"
                elif vi_valid:
                    tag = f"valid_w{vi_n_wrong}"
                else:
                    tag = f"fail_w{vi_n_wrong}"
                vid_path = os.path.join(
                    vid_dir,
                    f"step_{step:07d}_sample{vi}_{tag}.{vid_fmt}")
                render_ar_sampling_video(
                    history=vid_history,
                    sample_idx=vi,
                    grid_hw=grid_hw,
                    empty_token=MASK_TOKEN,
                    save_path=vid_path,
                    condition=vid_cond,
                    save_format=vid_fmt,
                )
            accelerator.print(
                f"[eval/{level_name}] Saved {vid_n} videos → {vid_dir}/")

        accelerator.print(
            f"[eval/{level_name}] step={step}  "
            f"cell_acc(unk)={cell_acc_unk:.4f}  rule_acc={rule_acc:.4f}  "
            f"saved → {task_dir}/")

    # ── summary ──
    for line in summary_lines:
        accelerator.print(line)
    if args.log_with and log_dict:
        accelerator.log(log_dict, step=step)

    # save combined summary
    summary_dir = os.path.join(args.output_dir, "eval_samples")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir,
                           f"step_{step:07d}_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")


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
    mode_str = "Conditional" if args.conditional else "Unconditional"
    accelerator.print("=" * 60)
    accelerator.print(f"Autoregressive Sudoku Generation ({mode_str})")
    accelerator.print("=" * 60)

    if accelerator.is_main_process and args.log_with is not None:
        sanitized_cfg = {}
        for k, v in vars(args).items():
            if v is None:
                sanitized_cfg[k] = "None"
            elif isinstance(v, (list, tuple)):
                sanitized_cfg[k] = str(v)
            else:
                sanitized_cfg[k] = v
        accelerator.init_trackers(
            project_name="sudoku_ar",
            config=sanitized_cfg,
        )

    # ─────────────────────────────────────────────────────────
    #  Dataset
    # ─────────────────────────────────────────────────────────
    accelerator.print(f"[data] Loading config: {args.sudoku_config}")
    cfg = OmegaConf.load(args.sudoku_config)
    srm_ds_cfg = cfg.SRM_dataset_cfg
    srm_cond_cfg = cfg.SRM_conditioning_cfg

    train_dataset_raw = get_dataset(srm_ds_cfg, srm_cond_cfg, "train")
    val_dataset_raw = get_dataset(srm_ds_cfg, srm_cond_cfg, "val")

    if args.grid_only:
        accelerator.print("[data] --grid_only: skipping image loading")
        train_dataset = GridOnlyDataset(train_dataset_raw)
        val_dataset = GridOnlyDataset(val_dataset_raw)
    else:
        train_dataset = train_dataset_raw
        val_dataset = val_dataset_raw

    accelerator.print(
        f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # sanity check
    s0 = train_dataset[0]
    assert isinstance(s0, dict) and "grid" in s0, \
        f"Expected dict with 'grid', got {type(s0)}"
    accelerator.print(
        f"[data] grid shape={s0['grid'].shape}, "
        f"range=[{s0['grid'].min().item()}, {s0['grid'].max().item()}]")

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
    model = ARTransformer(
        vocab_size=VOCAB_SIZE,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        mlp_ratio=args.mlp_ratio,
        dropout=args.model_dropout,
        conditional=args.conditional,
        pos_emb_type=args.pos_emb_type,
        sudoku_hw=args.grid_hw,
    )

    total_p, train_p = count_params(model)
    accelerator.print(
        f"[model] ARTransformer: hidden={args.hidden_size}, "
        f"heads={args.n_heads}, blocks={args.n_blocks}, "
        f"conditional={args.conditional}, pos_emb={args.pos_emb_type}")
    accelerator.print(
        f"[model] Total: {format_n(total_p)}  "
        f"(trainable {format_n(train_p)})")
    if args.conditional:
        accelerator.print(
            f"[cond] mask_ratio=[{args.mask_ratio_min}, "
            f"{args.mask_ratio_max}]")

    # ── optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_lr_scheduler(
        optimizer, args.warmup_steps, args.max_train_steps)

    # ── prepare ──
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler)

    # ── EMA ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(model).parameters(),
                  decay=args.ema_decay)
        ema.to(accelerator.device)
        accelerator.print(f"[ema] decay = {args.ema_decay}")

    # ── resume ──
    global_step = 0
    if args.resume_dir and os.path.isdir(args.resume_dir):
        accelerator.load_state(args.resume_dir)
        global_step = parse_step_from_dir(args.resume_dir)
        accelerator.print(
            f"[resume] Resumed from {args.resume_dir}, step={global_step}")

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

    model.train()
    epoch = 0
    running_loss = 0.0

    while global_step < args.max_train_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            grid = batch["grid"].to(accelerator.device).long()
            tokens = grid.view(grid.shape[0], -1) - 1           # (B, 81) [0-8]

            with accelerator.accumulate(model):
                loss = compute_loss(
                    accelerator.unwrap_model(model),
                    tokens,
                    conditional=args.conditional,
                    mask_ratio_min=args.mask_ratio_min,
                    mask_ratio_max=args.mask_ratio_max,
                    label_smoothing=args.label_smoothing,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if ema is not None:
                ema.update(accelerator.unwrap_model(model).parameters())

            global_step += 1
            running_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}")

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

                # ── debug: print one-step predictions ──
                if accelerator.is_main_process:
                    raw_model = accelerator.unwrap_model(model)
                    raw_model.eval()
                    with torch.no_grad():
                        dbg_x = tokens[:1]                       # (1, 81)
                        if args.conditional:
                            # show prediction with ~50% masking
                            mask = torch.rand(1, GRID_LEN,
                                              device=dbg_x.device) < 0.5
                            cond = dbg_x.clone()
                            cond[mask] = MASK_TOKEN
                            full = torch.cat([cond, dbg_x], dim=1)
                            seg = torch.zeros(1, 2 * GRID_LEN,
                                              dtype=torch.long,
                                              device=dbg_x.device)
                            seg[:, GRID_LEN:] = 1
                            logits = raw_model(full, segment_ids=seg)
                            pred_logits = logits[:, GRID_LEN-1:2*GRID_LEN-1]
                            pred = pred_logits.argmax(dim=-1)    # (1, 81)
                            acc = (pred == dbg_x).float().mean().item()
                            n_masked = mask.sum().item()
                            masked_acc = 0.0
                            if n_masked > 0:
                                masked_acc = (
                                    (pred == dbg_x) & mask
                                ).float().sum().item() / n_masked
                            accelerator.print(
                                f"[debug] cond_mask={n_masked}/81  "
                                f"pred_acc(all)={acc:.3f}  "
                                f"pred_acc(mask)={masked_acc:.3f}")
                        else:
                            logits = raw_model(dbg_x)            # (1, 81, V)
                            pred = logits[:, :-1].argmax(dim=-1) # (1, 80)
                            target = dbg_x[:, 1:]               # (1, 80)
                            acc = (pred == target).float().mean().item()
                            accelerator.print(
                                f"[debug] next_tok_acc={acc:.3f}")

                        # print grid
                        accelerator.print("[debug] GT grid:")
                        for row in range(9):
                            s, e = row * 9, (row + 1) * 9
                            accelerator.print(
                                " ".join(str(v+1) for v in
                                         dbg_x[0, s:e].tolist()))
                        if args.conditional:
                            accelerator.print("[debug] Pred grid:")
                            for row in range(9):
                                s, e = row * 9, (row + 1) * 9
                                accelerator.print(
                                    " ".join(str(v+1) for v in
                                             pred[0, s:e].tolist()))

                    raw_model.train()
                running_loss = 0.0

            # ── eval ──
            if global_step % args.eval_every == 0:
                evaluate_and_save(
                    model, global_step, args, accelerator, ema,
                    val_dataset=val_dataset,
                )
                model.train()

            # ── save ──
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(
                    args.output_dir, "ckpt", f"step{global_step}")
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

    # ── final save & eval ──
    ckpt_dir = os.path.join(
        args.output_dir, "ckpt", f"step{global_step}_final")
    accelerator.save_state(ckpt_dir)
    accelerator.print(f"[save] Final checkpoint → {ckpt_dir}")

    evaluate_and_save(
        model, global_step, args, accelerator, ema,
        val_dataset=val_dataset,
    )
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
