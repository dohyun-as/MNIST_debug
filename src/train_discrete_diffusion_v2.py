#!/usr/bin/env python
"""
train_discrete_diffusion_v2.py
==============================
Multi-dataset discrete (absorbing-state) diffusion training.
Supports: ImageNet (class-label cond), CLEVR (JSON cond), Sudoku (grid).

Generates multi-resolution FSQ/VQ tokens from a pretrained continuous
diffusion model's encoder, then trains a discrete diffusion model
(MDLM-style) to generate those tokens conditioned on dataset-specific
information.

For Sudoku, also supports --grid_only mode (no pretrained model needed).

Usage:
    # ImageNet (class-conditioned)
    accelerate launch train_discrete_diffusion_v2.py \
        --dataset_type imagenet \
        --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
        --pretrained_output_dir runs/imagenet_256_pixel_dit_flow_fsq_mask075_CA \
        --output_dir runs/imagenet_discrete_diff

    # CLEVR (JSON-conditioned)
    accelerate launch train_discrete_diffusion_v2.py \
        --dataset_type clevr \
        --dataset_root ../clevr-dataset-gen/output/clevr_256_varied/images \
        --clevr_condition_dir ../clevr-dataset-gen/output/clevr_256_varied/conditions_margin30 \
        --pretrained_output_dir runs/clevr_256_dit_flow_fsq_CA \
        --output_dir runs/clevr_discrete_diff

    # Sudoku (grid-only, no pretrained model)
    accelerate launch train_discrete_diffusion_v2.py \
        --dataset_type sudoku \
        --sudoku_config ../config/sudoku_config.json \
        --grid_only \
        --output_dir runs/sudoku_discrete_diff
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, ProjectConfiguration, tqdm
from PIL import Image
from torchvision import transforms

from dit_model import DIT
from discrete_diffusion import DiscreteDiffusion
from ar_model import AutoregressiveModel
from noise_schedule import get_noise


# ────────────────────────────────────────────────────────────
#  EMA  (same logic as MDLM's models/ema.py)
# ────────────────────────────────────────────────────────────

class EMA:
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

def count_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def format_n(n: int) -> str:
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000: return f"{n/1_000_000:.3f}M"
    if n >= 1_000: return f"{n/1_000:.3f}K"
    return str(n)

def parse_step_from_dir(path: str) -> int:
    base = os.path.basename(os.path.normpath(path))
    if base.startswith("step"):
        try: return int(base.replace("step", ""))
        except Exception: pass
    return 0


# ────────────────────────────────────────────────────────────
#  CLEVR tokenizer + condition encoder
# ────────────────────────────────────────────────────────────
#
#  JSON → word-level token sequence → nn.Embedding → prefix tokens.
#  Entity names are included so relations can reference them.
#  Partial attributes are handled (missing attrs are simply omitted).
#
#  Example (partial attrs, like the augmented conditions):
#    {"entities": [{"name":"A", "attrs":{"color":"purple"}},
#                  {"name":"B", "attrs":{"color":"purple","shape":"cylinder","size":"small"}},
#                  {"name":"C", "attrs":{"color":"purple","material":"rubber"}}],
#     "relations": [{"subj":"C","rel":"in_front_of","obj":"A"},
#                   {"subj":"A","rel":"left_of","obj":"B"}]}
#  →  "A purple SEP B purple cylinder small SEP C purple rubber SEP
#      C in_front_of A SEP A left_of B SEP"
#
#  Both AR and discrete diffusion use these as prefix context
#  (no cross-attention needed).

CLEVR_VOCAB = (
    # 0-7: colors
    ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    # 8-10: shapes
    + ["cube", "sphere", "cylinder"]
    # 11-12: sizes
    + ["small", "large"]
    # 13-14: materials
    + ["rubber", "metal"]
    # 15-18: relations
    + ["left_of", "right_of", "in_front_of", "behind"]
    # 19: separator
    + ["SEP"]
    # 20-31: entity names A..L
    + [chr(ord("A") + i) for i in range(12)]
    # 32: padding
    + ["PAD"]
)
CLEVR_WORD2ID = {w: i for i, w in enumerate(CLEVR_VOCAB)}
CLEVR_PAD_ID = CLEVR_WORD2ID["PAD"]
CLEVR_SEP_ID = CLEVR_WORD2ID["SEP"]
CLEVR_VOCAB_SIZE = len(CLEVR_VOCAB)  # 33
MAX_CLEVR_COND_LEN = 128


def clevr_json_to_token_ids(cond_json: dict) -> torch.Tensor:
    """Convert CLEVR condition JSON → flat token ID sequence.

    Entity format: "NAME attr1 attr2 ... SEP"  (name first, then present attrs)
    Relation format: "SUBJ rel OBJ SEP"

    Returns:
        token_ids: (MAX_CLEVR_COND_LEN,) int64, padded with PAD_ID.
    """
    tokens = []
    entities = cond_json.get("entities", [])
    relations = cond_json.get("relations", [])

    # Entities: "NAME [size] [color] [material] [shape] SEP"
    for ent in entities:
        name = ent.get("name", "A")
        if name in CLEVR_WORD2ID:
            tokens.append(CLEVR_WORD2ID[name])
        attrs = ent.get("attrs", {})
        for key in ("size", "color", "material", "shape"):
            word = attrs.get(key, "")
            if word in CLEVR_WORD2ID:
                tokens.append(CLEVR_WORD2ID[word])
        tokens.append(CLEVR_SEP_ID)

    # Relations: "SUBJ rel OBJ SEP"
    for rel in relations:
        for word in (rel.get("subj", ""), rel.get("rel", ""),
                     rel.get("obj", "")):
            if word in CLEVR_WORD2ID:
                tokens.append(CLEVR_WORD2ID[word])
        tokens.append(CLEVR_SEP_ID)

    # Truncate and pad
    tokens = tokens[:MAX_CLEVR_COND_LEN]
    n_pad = MAX_CLEVR_COND_LEN - len(tokens)
    tokens = tokens + [CLEVR_PAD_ID] * n_pad

    return torch.tensor(tokens, dtype=torch.long)


class CLEVRConditionEncoder(nn.Module):
    """Embeds CLEVR token IDs → prefix context vectors.

    nn.Embedding + positional embedding + 2-layer transformer encoder
    for token interaction.  Output is used as prefix for both AR and
    discrete diffusion (no cross-attention).
    """

    def __init__(self, hidden_size: int, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(CLEVR_VOCAB_SIZE, hidden_size)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, MAX_CLEVR_COND_LEN, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.pad_id = CLEVR_PAD_ID

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) int64, padded with PAD_ID
        Returns:
            cond_tokens: (B, L_valid, hidden_size) — padding stripped,
                         only non-pad tokens returned.
        """
        B, L = token_ids.shape
        pad_mask = (token_ids == self.pad_id)  # (B, L) True=pad

        h = self.token_emb(token_ids) + self.pos_emb[:, :L, :]
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # Strip padding: keep only up to the max non-pad length in batch
        valid_lens = (~pad_mask).sum(dim=1)  # (B,)
        max_valid = valid_lens.max().item()
        if max_valid == 0:
            max_valid = 1  # at least 1 token
        h = h[:, :max_valid, :]

        # Zero out per-sample padding within the trimmed range
        trimmed_mask = pad_mask[:, :max_valid]
        h = h * (~trimmed_mask).unsqueeze(-1).float()
        return h


# ────────────────────────────────────────────────────────────
#  Sudoku digit-grid condition support
# ────────────────────────────────────────────────────────────

SUDOKU_GRID_LEN = 81          # 9×9
SUDOKU_DIGIT_VOCAB = 9        # digits 0-8  (display: 1-9)
SUDOKU_MASK_ID = 9            # [MASK] for unknown cells
SUDOKU_COND_VOCAB_SIZE = 11   # 0-8 + [MASK]=9 + [PAD]=10
SUDOKU_PAD_ID = 10


class SudokuDigitCellEncoder(nn.Module):
    """Per-cell digit condition for AdaLN-style injection.

    (B, 81) digit ids in {0..8 = digits 1..9, 9=UNKNOWN, 10=PAD}
    → (B, 81, hidden_size) residual embedding added to token features.
    """

    def __init__(self, hidden_size: int, grid_len: int = SUDOKU_GRID_LEN):
        super().__init__()
        # 11-way embedding: digits 0..8, UNKNOWN=9, PAD=10
        self.digit_emb = nn.Embedding(SUDOKU_COND_VOCAB_SIZE, hidden_size)
        nn.init.trunc_normal_(self.digit_emb.weight, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, grid_len, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.grid_len = grid_len

    def forward(self, digit_ids: torch.Tensor) -> torch.Tensor:
        B, L = digit_ids.shape
        h = self.digit_emb(digit_ids) + self.pos_emb[:, :L, :]
        return h


class SudokuConditionEncoder(nn.Module):
    """Embeds a (possibly masked) 9×9 digit grid → prefix context vectors.

    Same architecture as CLEVRConditionEncoder:
    nn.Embedding + positional embedding + 2-layer transformer encoder.
    Output is prepended as prefix tokens for discrete diffusion / AR.

    During training, the full 81-digit grid is given as condition.
    During eval, some digits can be replaced with SUDOKU_MASK_ID for
    difficulty control (easy/medium/hard).
    """

    def __init__(self, hidden_size: int, n_layers: int = 2,
                 n_heads: int = 4, grid_len: int = SUDOKU_GRID_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(SUDOKU_COND_VOCAB_SIZE, hidden_size)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, grid_len, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.grid_len = grid_len

    def forward(self, digit_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            digit_ids: (B, 81) int64 — values in [0..8] for known digits,
                       SUDOKU_MASK_ID (9) for unknown, SUDOKU_PAD_ID (10) for padding.
        Returns:
            cond_tokens: (B, 81, hidden_size)
        """
        B, L = digit_ids.shape
        h = self.token_emb(digit_ids) + self.pos_emb[:, :L, :]
        h = self.encoder(h)
        return h


# ────────────────────────────────────────────────────────────
#  CLEVR text condition support
# ────────────────────────────────────────────────────────────

# Word-level vocab for CLEVR text captions
CLEVR_TEXT_VOCAB = [
    # numbers
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # colors
    "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
    # shapes
    "cube", "sphere", "cylinder",
    # sizes
    "small", "large",
    # materials
    "rubber", "metal",
    # relations / spatial
    "left", "right", "behind", "in", "front", "of", "to",
    # structural words
    "there", "is", "are", "objects", "the", "a", "and",
    # punctuation
    ",", ".", ":",
    # special
    "SEP", "PAD",
]
CLEVR_TEXT_WORD2ID = {w: i for i, w in enumerate(CLEVR_TEXT_VOCAB)}
CLEVR_TEXT_PAD_ID = CLEVR_TEXT_WORD2ID["PAD"]
CLEVR_TEXT_VOCAB_SIZE = len(CLEVR_TEXT_VOCAB)
MAX_CLEVR_TEXT_LEN = 64


def clevr_text_to_token_ids(caption) -> torch.Tensor:
    """Convert a CLEVR text caption → flat token ID sequence.

    Args:
        caption: str or dict with "text" key.

    Returns:
        token_ids: (MAX_CLEVR_TEXT_LEN,) int64, padded with PAD_ID.
    """
    import re
    if isinstance(caption, dict):
        caption = caption.get("text", "")
    words = re.findall(r'[a-zA-Z_]+|[0-9]+|[.,:]', caption.lower())
    tokens = []
    for w in words:
        if w in CLEVR_TEXT_WORD2ID:
            tokens.append(CLEVR_TEXT_WORD2ID[w])
        # skip unknown words (shouldn't happen for CLEVR)

    tokens = tokens[:MAX_CLEVR_TEXT_LEN]
    n_pad = MAX_CLEVR_TEXT_LEN - len(tokens)
    tokens = tokens + [CLEVR_TEXT_PAD_ID] * n_pad
    return torch.tensor(tokens, dtype=torch.long)


class CLEVRTextConditionEncoder(nn.Module):
    """Embeds CLEVR text caption token IDs → prefix context vectors.

    Same architecture as CLEVRConditionEncoder but with text vocab.
    """

    def __init__(self, hidden_size: int, n_layers: int = 2,
                 n_heads: int = 4):
        super().__init__()
        self.token_emb = nn.Embedding(CLEVR_TEXT_VOCAB_SIZE, hidden_size)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, MAX_CLEVR_TEXT_LEN, hidden_size))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        self.pad_id = CLEVR_TEXT_PAD_ID

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) int64, padded with PAD_ID
        Returns:
            cond_tokens: (B, L_valid, hidden_size)
        """
        B, L = token_ids.shape
        pad_mask = (token_ids == self.pad_id)

        h = self.token_emb(token_ids) + self.pos_emb[:, :L, :]
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        valid_lens = (~pad_mask).sum(dim=1)
        max_valid = valid_lens.max().item()
        if max_valid == 0:
            max_valid = 1
        h = h[:, :max_valid, :]

        trimmed_mask = pad_mask[:, :max_valid]
        h = h * (~trimmed_mask).unsqueeze(-1).float()
        return h


class PretrainedTextConditionEncoder(nn.Module):
    """Wraps a pretrained HF text encoder (CLIP / T5) — mirrors the naive
    ``model_text_conditioned.PretrainedLMEncoder`` recipe:
      * per-batch dynamic padding (``padding=True``)
      * tokenizer's real ``attention_mask`` (correct for CLIP eos==pad)
      * learnable ``null_embed`` for CFG uncond (not the backbone's)

    Call pattern:
        tokens = enc.tokenize(list_of_texts, device)      # dict with ids+mask
        cond, attn = enc(tokens)                          # (B, L, D), (B, L) bool
        null = enc.get_null_cond(B, L, device)            # (B, L, D)
    """

    def __init__(self, model_name: str, hidden_size: int,
                 max_length: int = 77, freeze: bool = False):
        super().__init__()
        name_lower = model_name.lower()
        if "clip" in name_lower:
            from transformers import CLIPTextModel, AutoTokenizer
            self._kind = "clip"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
            enc_dim = self.encoder.config.hidden_size
        elif "t5" in name_lower:
            from transformers import T5EncoderModel, AutoTokenizer
            self._kind = "t5"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            enc_dim = self.encoder.config.d_model
        else:
            raise ValueError(
                f"Unsupported pretrained text encoder: {model_name} "
                "(expected CLIP or T5 variant)")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_name = model_name
        self.max_length = max_length
        self.freeze = freeze
        self.hidden_size = hidden_size
        self.proj = nn.Linear(enc_dim, hidden_size)

        # Learnable null-cond embedding (naive-style CFG uncond).
        self.null_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.null_embed, std=0.02)

        if freeze:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def tokenize(self, texts, device):
        """Tokenize a list of captions with per-batch dynamic padding."""
        if isinstance(texts, str):
            texts = [texts]
        norm = [(t.get("text", "") if isinstance(t, dict) else t)
                for t in texts]
        return self.tokenizer(
            norm, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        ).to(device)

    def forward(self, text_tokens):
        """
        Args:
            text_tokens: dict with ``input_ids`` and ``attention_mask``
                ((B, L) each, produced by :meth:`tokenize`).
        Returns:
            cond_tokens: (B, L, hidden_size)
            cond_mask:   (B, L) bool (True = valid token)
        """
        input_ids = text_tokens["input_ids"]
        attn_mask = text_tokens["attention_mask"]

        if self.freeze:
            with torch.no_grad():
                out = self.encoder(input_ids=input_ids,
                                   attention_mask=attn_mask)
                hidden = out.last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids,
                               attention_mask=attn_mask)
            hidden = out.last_hidden_state

        cond = self.proj(hidden.float())
        # Zero padding positions (downstream backbone doesn't take a mask).
        cond = cond * attn_mask.unsqueeze(-1).float()
        return cond, attn_mask.bool()

    def get_null_cond(self, batch_size: int, seq_len: int,
                      device: torch.device):
        """Return broadcasted null-cond tokens (B, L, D) for CFG."""
        return self.null_embed.to(device).expand(batch_size, seq_len, -1)


# ────────────────────────────────────────────────────────────
#  Dataset classes
# ────────────────────────────────────────────────────────────

def _extract_raw_text(cond):
    """Extract raw caption string from a CLEVR text condition entry."""
    if isinstance(cond, dict):
        return cond.get("text", "")
    return cond


class CachedTokenDataset(Dataset):
    """Returns cached tok_ids + optional labels/conditions.

    If ``return_raw_text=True``, yields raw caption strings under
    ``"cond_text"`` instead of pre-tokenized ``"cond_token_ids"``; used for
    the pretrained (CLIP/T5) text-encoder path which tokenizes per-batch.
    """
    def __init__(self, tok_ids, labels=None, clevr_conditions=None,
                 cond_tokenizer_fn=None, sudoku_digit_grids=None,
                 return_raw_text=False, source_image_ds=None):
        self.tok_ids = tok_ids  # (N, seq_len) long
        self.labels = labels    # (N,) long or None
        self.clevr_conditions = clevr_conditions  # list of dicts or None
        self.cond_tokenizer_fn = cond_tokenizer_fn or clevr_json_to_token_ids
        self.sudoku_digit_grids = sudoku_digit_grids  # (N, 9, 9) or None
        self.return_raw_text = return_raw_text
        # Reference to the source image dataset (same indexing order) so
        # eval can fetch the GT image for a given index without reloading.
        self.source_image_ds = source_image_ds

    def __len__(self):
        return len(self.tok_ids)

    def __getitem__(self, idx):
        item = {"tok_ids": self.tok_ids[idx].long()}
        if self.labels is not None:
            item["class_label"] = self.labels[idx]
        if self.clevr_conditions is not None:
            cond = self.clevr_conditions[idx]
            if self.return_raw_text:
                item["cond_text"] = _extract_raw_text(cond)
            else:
                item["cond_token_ids"] = self.cond_tokenizer_fn(cond)
        if self.sudoku_digit_grids is not None:
            # (9, 9) → (81,) int64, values 1-9 → 0-8
            grid = self.sudoku_digit_grids[idx]  # (9, 9)
            digits = grid.reshape(-1).long() - 1  # (81,) in [0, 8]
            item["cond_token_ids"] = digits
        return item

    def get_condition(self, idx):
        """Return raw condition dict (for eval logging)."""
        if self.clevr_conditions is not None:
            return self.clevr_conditions[idx]
        return {}


class CachedContinuousTokenDataset(Dataset):
    """Returns cached continuous feature vectors + optional conditions."""
    def __init__(self, features, labels=None, clevr_conditions=None,
                 cond_tokenizer_fn=None, sudoku_digit_grids=None,
                 return_raw_text=False, source_image_ds=None):
        self.features = features  # (N, seq_len, feat_dim) float
        self.labels = labels
        self.clevr_conditions = clevr_conditions
        self.cond_tokenizer_fn = cond_tokenizer_fn or clevr_json_to_token_ids
        self.sudoku_digit_grids = sudoku_digit_grids
        self.return_raw_text = return_raw_text
        self.source_image_ds = source_image_ds

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"cont_tokens": self.features[idx].float()}
        if self.labels is not None:
            item["class_label"] = self.labels[idx]
        if self.clevr_conditions is not None:
            cond = self.clevr_conditions[idx]
            if self.return_raw_text:
                item["cond_text"] = _extract_raw_text(cond)
            else:
                item["cond_token_ids"] = self.cond_tokenizer_fn(cond)
        if self.sudoku_digit_grids is not None:
            grid = self.sudoku_digit_grids[idx]
            digits = grid.reshape(-1).long() - 1  # (81,) in [0, 8]
            item["cond_token_ids"] = digits
        return item

    def get_condition(self, idx):
        """Return raw condition dict (for eval logging)."""
        if self.clevr_conditions is not None:
            return self.clevr_conditions[idx]
        return {}


class GridOnlyDataset(Dataset):
    """Sudoku grids only (no images)."""
    def __init__(self, inner_dataset):
        self.grids = inner_dataset.sudoku_grids

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return {"grid": self.grids[idx].long()}


# ────────────────────────────────────────────────────────────
#  Token extraction from pretrained multi-res model
# ────────────────────────────────────────────────────────────

def load_pretrained_model(pretrained_output_dir: str, device: str = "cpu"):
    """Load the full pretrained multi-res model (encoder + decoder).

    Returns: (model, encoder, discretizer, level_sizes, vocab_size, config_args)
    """
    # Load config — try both args.json and run_config.json
    config_path = os.path.join(pretrained_output_dir, "args.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(pretrained_output_dir, "run_config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(pretrained_output_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    # Extract args from config (run_config.json wraps in {"args": ...})
    if "args" in cfg:
        cfg = cfg["args"]

    # Determine backbone and rebuild model
    backbone_type = cfg.get("backbone", "dit")

    # Common ViT encoder kwargs.  main_multires.py resolves the CNN-stem flag
    # as `vit_use_cnn_stem and not vit_no_cnn_stem`, but both raw flags are
    # saved to args.json.  Respect both here so vit_global backbones trained
    # with --vit_no_cnn_stem rebuild with matching architecture.
    resolved_use_stem = (cfg.get("vit_use_cnn_stem", True)
                         and not cfg.get("vit_no_cnn_stem", False))
    vit_kwargs = dict(
        encoder_type=cfg.get("encoder_type", "cnn"),
        vit_patch_size=cfg.get("vit_patch_size", 4),
        vit_depth=cfg.get("vit_depth", 4),
        vit_num_heads=cfg.get("vit_num_heads", 4),
        vit_mlp_ratio=cfg.get("vit_mlp_ratio", 4.0),
        vit_use_cnn_stem=resolved_use_stem,
        vit_cnn_stem_reduction=cfg.get("vit_cnn_stem_reduction", 4),
    )

    if backbone_type == "baseline_1d":
        from model_baseline_1d import Baseline1DConditionalDiT
        model = Baseline1DConditionalDiT(
            image_size=cfg["image_size"],
            in_channels=cfg.get("in_channels", 3),
            cond_in_channels=cfg.get("cond_in_channels", 3),
            vae_downsample_factor=cfg.get("vae_downsample_factor", 1),
            num_slots=cfg.get("num_slots", 256),
            slot_dim=cfg.get("slot_dim", 16),
            enc_embed_dim=cfg.get("enc_embed_dim", 768),
            enc_depth=cfg.get("enc_depth", 12),
            enc_num_heads=cfg.get("enc_num_heads", 12),
            enc_drop_path_rate=cfg.get("enc_drop_path_rate", 0.1),
            is_causal=cfg.get("is_causal", True),
            enable_nest=cfg.get("enable_nest", True),
            enable_nest_after_steps=cfg.get("enable_nest_after_steps", -1),
            dit_patch_size=cfg.get("dit_patch_size", 16),
            dit_hidden_size=cfg.get("dit_hidden_size", 768),
            dit_n_heads=cfg.get("dit_n_heads", 12),
            dit_n_blocks=cfg.get("dit_n_blocks", 12),
            dit_mlp_ratio=cfg.get("dit_mlp_ratio", 4.0),
            dit_dropout=cfg.get("dit_dropout", 0.0),
            dit_bottleneck_dim=cfg.get("dit_bottleneck_dim", 128),
            dit_in_context_len=cfg.get("dit_in_context_len", 0),
            dit_in_context_start=cfg.get("dit_in_context_start", 4),
            dit_attn_mode=cfg.get("dit_attn_mode", "self_concat"),
            uncond_drop_prob=0.0,
            use_fsq=cfg.get("use_fsq", False),
            fsq_levels=cfg.get("fsq_levels", None),
            fsq_drop_quant_p=0.0,
            fsq_corrupt_tokens_p=0.0,
            use_vq=cfg.get("use_vq", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            vq_beta=cfg.get("vq_beta", 0.25),
        )
    elif backbone_type == "dit":
        from model_multires import MultiResConditionalDiT
        model = MultiResConditionalDiT(
            image_size=cfg["image_size"],
            in_channels=cfg.get("in_channels", 3),
            cond_in_channels=cfg.get("cond_in_channels", 3),
            vae_downsample_factor=cfg.get("vae_downsample_factor", 1),
            min_patch_size=cfg.get("min_patch_size", 32),
            num_levels=cfg.get("num_levels", None),
            feat_channels=cfg.get("feat_channels", 256),
            encoder_internal_dim=cfg.get("encoder_internal_dim", None),
            dit_patch_size=cfg.get("dit_patch_size", 2),
            dit_hidden_size=cfg.get("dit_hidden_size", 768),
            dit_n_heads=cfg.get("dit_n_heads", 12),
            dit_n_blocks=cfg.get("dit_n_blocks", 12),
            dit_mlp_ratio=cfg.get("dit_mlp_ratio", 4.0),
            dit_dropout=cfg.get("dit_dropout", 0.0),
            dit_bottleneck_dim=cfg.get("dit_bottleneck_dim", 128),
            dit_in_context_len=cfg.get("dit_in_context_len", 0),
            dit_in_context_start=cfg.get("dit_in_context_start", 4),
            uncond_drop_prob=0.0,
            level_drop=False,
            min_keep_levels=1,
            depth_per_level=cfg.get("depth_per_level", 2),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            cnn_base_channels=cfg.get("cnn_base_channels", 64),
            cond_use_latent=cfg.get("cond_use_latent", False),
            mae_mask_ratio=0.0,
            use_fsq=cfg.get("use_fsq", False),
            fsq_levels=cfg.get("fsq_levels", None),
            use_vq=cfg.get("use_vq", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            vq_beta=cfg.get("vq_beta", 0.25),
            level_sizes=cfg.get("level_sizes", None),
            **vit_kwargs,
        )
    else:
        from model_multires import MultiResConditionalUNet
        model = MultiResConditionalUNet(
            image_size=cfg["image_size"],
            in_channels=cfg.get("in_channels", 3),
            cond_in_channels=cfg.get("cond_in_channels", 3),
            vae_downsample_factor=cfg.get("vae_downsample_factor", 1),
            min_patch_size=cfg.get("min_patch_size", 32),
            num_levels=cfg.get("num_levels", None),
            feat_channels=cfg.get("feat_channels", 256),
            uncond_drop_prob=0.0,
            level_drop=False,
            depth_per_level=cfg.get("depth_per_level", 2),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            cnn_base_channels=cfg.get("cnn_base_channels", 64),
            use_fsq=cfg.get("use_fsq", False),
            fsq_levels=cfg.get("fsq_levels", None),
            use_vq=cfg.get("use_vq", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            **vit_kwargs,
        )

    # Load checkpoint weights
    ckpt_dir = os.path.join(pretrained_output_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        steps = []
        for d in os.listdir(ckpt_dir):
            if d.startswith("step_"):
                try: steps.append(int(d.split("_")[1]))
                except ValueError: pass
        if steps:
            latest = max(steps)
            ckpt_path = os.path.join(ckpt_dir, f"step_{latest:07d}", "checkpoint.pt")
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = ckpt.get("model", ckpt.get("ema", ckpt))
                model.load_state_dict(state, strict=False)
                print(f"[pretrained] Loaded checkpoint step {latest}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    encoder = model.encoder
    discretizer = model.discretizer

    # level_sizes: multi-res has spatial hierarchy, baseline_1d has flat 1D slots
    if hasattr(encoder, 'level_sizes'):
        level_sizes = list(encoder.level_sizes)
    else:
        # baseline_1d: num_slots flat tokens, represent as single "level"
        num_slots = getattr(model, 'num_slots', encoder.num_slots)
        level_sizes = [num_slots]  # single level with num_slots tokens

    # Compute vocab size
    if discretizer is not None:
        if hasattr(discretizer, 'codebook_size'):
            vocab_size = discretizer.codebook_size
        elif hasattr(discretizer, 'fsq'):
            vocab_size = discretizer.fsq.codebook_size
        else:
            vocab_size = 512
    else:
        # No discretizer — continuous mode (diffusion head) only
        vocab_size = 0

    return model, encoder, discretizer, level_sizes, vocab_size, cfg


@torch.no_grad()
def extract_tokens(encoder, discretizer, images, device):
    """Extract flat token IDs from a batch of images.

    Args:
        encoder: HierarchicalMultiResEncoder or SemanticistViTEncoder
        discretizer: FSQDiscretizer or VQDiscretizer
        images: (B, C, H, W) tensor
        device: target device
    Returns:
        tok_ids: (B, total_tokens) long
    """
    images = images.to(device)

    if hasattr(encoder, 'forward_injection'):
        # Multi-res encoder: spatial hierarchy
        level_features = encoder.forward_injection(images)
        all_tok_ids = []
        for s in sorted(level_features.keys(), reverse=True):
            feat_2d = level_features[s]  # (B, D, S, S)
            B, D = feat_2d.shape[:2]
            tokens_2d = feat_2d.flatten(2).transpose(1, 2)  # (B, S*S, D)
            _, tok_indices = discretizer(tokens_2d)  # (B, S*S)
            all_tok_ids.append(tok_indices)
        return torch.cat(all_tok_ids, dim=1)
    else:
        # Semanticist-style 1D encoder: flat slot tokens
        slots = encoder(images)  # (B, num_slots, slot_dim)
        _, tok_indices = discretizer(slots)  # (B, num_slots)
        return tok_indices


@torch.no_grad()
def extract_continuous_tokens(encoder, images, device):
    """Extract flat continuous feature vectors from a batch of images.

    Args:
        encoder: HierarchicalMultiResEncoder
        images: (B, C, H, W) tensor
        device: target device
    Returns:
        features: (B, total_tokens, feat_dim) float
    """
    images = images.to(device)

    if hasattr(encoder, 'forward_injection'):
        level_features = encoder.forward_injection(images)
        all_feats = []
        for s in sorted(level_features.keys(), reverse=True):
            feat_2d = level_features[s]  # (B, D, S, S)
            tokens_2d = feat_2d.flatten(2).transpose(1, 2)  # (B, S*S, D)
            all_feats.append(tokens_2d)
        return torch.cat(all_feats, dim=1)  # (B, total_tokens, D)
    else:
        slots = encoder(images)  # (B, num_slots, slot_dim)
        return slots


@torch.no_grad()
def cache_all_continuous_tokens(encoder, dataset, device,
                                batch_size=64, cache_path=None,
                                accelerator=None):
    """Extract and cache continuous feature vectors for an entire dataset."""
    # Synchronize the "cache exists" decision across all ranks.
    # (Filesystem visibility can differ between ranks on NFS, causing
    # some to load and others to enter the distributed caching path → NCCL hang.)
    has_cache_local = cache_path is not None and os.path.isfile(cache_path)
    if accelerator is not None and accelerator.num_processes > 1:
        import torch.distributed as dist
        has_t = torch.tensor(int(has_cache_local), device=device)
        dist.broadcast(has_t, src=0)
        has_cache = bool(has_t.item())
    else:
        has_cache = has_cache_local
    if has_cache:
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        if accelerator:
            accelerator.print(f"[cache] Loaded from {cache_path}, shape={feats.shape}")
        return feats

    from torch.utils.data import DataLoader, DistributedSampler

    use_distributed = accelerator is not None and accelerator.num_processes > 1

    if use_distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=accelerator.num_processes,
            rank=accelerator.process_index, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    local_feats = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            images = batch["image"]
        else:
            images = batch
        images = images.to(device)
        feat = extract_continuous_tokens(encoder, images, device)
        local_feats.append(feat.cpu())

    local_feat = torch.cat(local_feats, dim=0)  # (local_N, seq_len, feat_dim)

    if use_distributed:
        import torch.distributed as dist
        world_size = accelerator.num_processes
        local_size = torch.tensor([local_feat.shape[0]], dtype=torch.long, device=device)
        all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)

        if local_feat.shape[0] < max_size:
            pad = torch.zeros(max_size - local_feat.shape[0], *local_feat.shape[1:],
                              dtype=local_feat.dtype)
            local_feat_padded = torch.cat([local_feat, pad], dim=0)
        else:
            local_feat_padded = local_feat

        local_feat_gpu = local_feat_padded.to(device)
        gathered = [torch.zeros_like(local_feat_gpu) for _ in range(world_size)]
        dist.all_gather(gathered, local_feat_gpu)

        total_len = len(dataset)
        feats = torch.zeros(total_len, *local_feat.shape[1:], dtype=local_feat.dtype)
        for rank_idx in range(world_size):
            rank_size = int(all_sizes[rank_idx].item())
            rank_data = gathered[rank_idx][:rank_size].cpu()
            indices = list(range(rank_idx, total_len, world_size))
            feats[indices[:rank_size]] = rank_data

        accelerator.print(f"[cache] Distributed caching done: {world_size} GPUs, shape={feats.shape}")
    else:
        feats = local_feat

    if cache_path is not None and (not use_distributed or accelerator.is_main_process):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(feats, cache_path)
        if accelerator:
            accelerator.print(f"[cache] Saved to {cache_path}, shape={feats.shape}")

    return feats


@torch.no_grad()
def cache_all_tokens(encoder, discretizer, dataset, device,
                     batch_size=64, cache_path=None, accelerator=None):
    """Extract and cache token IDs for an entire dataset using all GPUs.

    Each rank processes a shard of the dataset in parallel, then rank 0
    gathers and saves the full cache.  All ranks return the complete tensor.
    """
    has_cache_local = cache_path is not None and os.path.isfile(cache_path)
    if accelerator is not None and accelerator.num_processes > 1:
        import torch.distributed as dist
        has_t = torch.tensor(int(has_cache_local), device=device)
        dist.broadcast(has_t, src=0)
        has_cache = bool(has_t.item())
    else:
        has_cache = has_cache_local
    if has_cache:
        tok_ids = torch.load(cache_path, map_location="cpu", weights_only=True)
        if accelerator:
            accelerator.print(f"[cache] Loaded from {cache_path}, shape={tok_ids.shape}")
        return tok_ids

    from torch.utils.data import DataLoader, DistributedSampler

    use_distributed = accelerator is not None and accelerator.num_processes > 1

    if use_distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=accelerator.num_processes,
            rank=accelerator.process_index, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)

    local_ids = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            images = batch["image"]
        else:
            images = batch
        images = images.to(device)
        tok = extract_tokens(encoder, discretizer, images, device)
        local_ids.append(tok.cpu())

    local_tok = torch.cat(local_ids, dim=0)  # (local_N, seq_len)

    if use_distributed:
        # Gather all shards on every rank
        import torch.distributed as dist
        world_size = accelerator.num_processes
        # Pad local tensors to same size for all_gather
        local_size = torch.tensor([local_tok.shape[0]], dtype=torch.long, device=device)
        all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)

        # Pad to max_size
        if local_tok.shape[0] < max_size:
            pad = torch.zeros(max_size - local_tok.shape[0], *local_tok.shape[1:],
                              dtype=local_tok.dtype)
            local_tok_padded = torch.cat([local_tok, pad], dim=0)
        else:
            local_tok_padded = local_tok

        # all_gather on GPU then move to CPU
        local_tok_gpu = local_tok_padded.to(device)
        gathered = [torch.zeros_like(local_tok_gpu) for _ in range(world_size)]
        dist.all_gather(gathered, local_tok_gpu)

        # Trim padding and interleave to restore original order
        # DistributedSampler assigns indices round-robin, so we interleave
        total_len = len(dataset)
        tok_ids = torch.zeros(total_len, *local_tok.shape[1:], dtype=local_tok.dtype)
        for rank_idx in range(world_size):
            rank_size = int(all_sizes[rank_idx].item())
            rank_data = gathered[rank_idx][:rank_size].cpu()
            # DistributedSampler assigns: indices[i] goes to rank (i % world_size)
            # So rank r gets indices [r, r+W, r+2W, ...]
            indices = list(range(rank_idx, total_len, world_size))
            tok_ids[indices[:rank_size]] = rank_data

        accelerator.print(f"[cache] Distributed caching done: {world_size} GPUs, shape={tok_ids.shape}")
    else:
        tok_ids = local_tok

    if cache_path is not None and (not use_distributed or accelerator.is_main_process):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(tok_ids, cache_path)
        if accelerator:
            accelerator.print(f"[cache] Saved to {cache_path}, shape={tok_ids.shape}")

    return tok_ids


# ────────────────────────────────────────────────────────────
#  Image-only dataset wrappers (for token extraction)
# ────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """ImageFolder-style dataset returning (image_tensor, class_label)."""
    def __init__(self, root, split="train", image_size=256):
        from torchvision.datasets import ImageFolder
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = root
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.ds = ImageFolder(split_dir, transform=self.transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        return {"image": img, "class_label": label}


class CLEVRImageDataset(Dataset):
    """CLEVR images with conditions (JSON structured or text captions).

    Supports augmented conditions: if multiple condition JSONs / captions
    map to the same image, each (image, condition) pair becomes a separate
    dataset entry.

    Args:
        cond_type: "json" (structured entities+relations) or "text" (natural
                   language captions).
    """
    def __init__(self, image_root, condition_dir=None, image_size=256,
                 splits=("easy", "medium", "hard"), cond_type="json"):
        self.image_paths = []
        self.labels = []
        self.conditions = []
        self.cond_type = cond_type
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # Load conditions: image_filename → list of conditions
        from collections import defaultdict
        cond_map = defaultdict(list)
        if condition_dir is not None:
            if cond_type == "text":
                self._load_text_conditions(condition_dir, splits, cond_map)
            else:
                self._load_json_conditions(condition_dir, splits, cond_map)

        # Collect (image, condition) pairs — one entry per condition
        for split in splits:
            split_dir = os.path.join(image_root, split)
            if not os.path.isdir(split_dir):
                continue
            for fn in sorted(os.listdir(split_dir)):
                if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(split_dir, fn)
                conds = cond_map.get(fn, [{}])
                for cond in conds:
                    self.image_paths.append(img_path)
                    self.labels.append(0)
                    self.conditions.append(cond)

    def _load_json_conditions(self, condition_dir, splits, cond_map):
        """Load structured JSON conditions (entities + relations)."""
        for split in splits:
            cond_file = os.path.join(condition_dir, f"conditions_{split}.json")
            if os.path.isfile(cond_file):
                with open(cond_file) as f:
                    cond_list = json.load(f)
                for c in cond_list:
                    cond_map[c["image_filename"]].append(c)
            else:
                per_file_dir = os.path.join(condition_dir, split)
                if os.path.isdir(per_file_dir):
                    for fn in sorted(os.listdir(per_file_dir)):
                        if fn.endswith(".json"):
                            fpath = os.path.join(per_file_dir, fn)
                            with open(fpath) as f:
                                c = json.load(f)
                            cond_map[c["image_filename"]].append(c)

    def _load_text_conditions(self, condition_dir, splits, cond_map):
        """Load text caption conditions.

        Supports two formats:
          1. Combined: captions_{split}.json — list of {image_filename, split, captions}
          2. Per-file: {split}/CLEVR_*.json — {image_filename, split, captions}

        If only per-file JSONs exist, automatically builds and saves a combined
        captions_{split}.json so subsequent runs load instantly.

        Each caption becomes a separate dataset entry, stored as a dict with
        "text", "image_filename", "split" keys (so eval can recover split info).
        """
        for split in splits:
            combined = os.path.join(condition_dir, f"captions_{split}.json")
            if os.path.isfile(combined):
                # Fast path: load combined JSON
                print(f"[data] Loading combined captions: {combined}")
                with open(combined) as f:
                    items = json.load(f)
            else:
                # Slow path: read per-file JSONs, then save combined for next time
                per_file_dir = os.path.join(condition_dir, split)
                if not os.path.isdir(per_file_dir):
                    continue
                files = sorted(fn for fn in os.listdir(per_file_dir) if fn.endswith(".json"))
                print(f"[data] Building combined captions from {len(files)} per-file JSONs ({split})...")
                from collections import defaultdict as _dd
                per_image = _dd(lambda: {"image_filename": "", "split": split, "captions": []})
                for fn in files:
                    fpath = os.path.join(per_file_dir, fn)
                    with open(fpath) as f:
                        data = json.load(f)
                    img_fn = data.get("image_filename", "")
                    sp = data.get("split", split)
                    entry = per_image[img_fn]
                    entry["image_filename"] = img_fn
                    entry["split"] = sp
                    entry["captions"].extend(data.get("captions", []))
                items = list(per_image.values())
                # Save combined JSON for future runs
                try:
                    with open(combined, "w") as f:
                        json.dump(items, f)
                    print(f"[data] Saved combined captions: {combined} ({len(items)} images)")
                except OSError as e:
                    print(f"[data] Warning: could not save combined captions: {e}")

            for item in items:
                img_fn = item.get("image_filename", "")
                sp = item.get("split", split)
                for cap in item.get("captions", []):
                    cond_map[img_fn].append({
                        "text": cap, "image_filename": img_fn, "split": sp})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        item = {"image": img, "class_label": self.labels[idx]}
        return item

    def get_condition(self, idx):
        return self.conditions[idx]


# ────────────────────────────────────────────────────────────
#  LR scheduler
# ────────────────────────────────────────────────────────────

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int,
                     schedule: str = "constant", min_lr_ratio: float = 0.1):
    """Warmup + {constant | cosine decay to min_lr_ratio * base_lr}."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        if schedule == "cosine":
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
#  Sudoku evaluation helpers (imported from v1)
# ────────────────────────────────────────────────────────────

def check_sudoku_rules(grids):
    B = grids.shape[0]
    valid = row_v = col_v = box_v = 0
    for b in range(B):
        g = grids[b].cpu()
        ok_r = all(len(set(g[r].tolist())) == 9 for r in range(9))
        ok_c = all(len(set(g[:, c].tolist())) == 9 for c in range(9))
        ok_b = True
        for br in range(3):
            for bc in range(3):
                box = g[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten().tolist()
                if len(set(box)) != 9: ok_b = False; break
            if not ok_b: break
        if ok_r: row_v += 1
        if ok_c: col_v += 1
        if ok_b: box_v += 1
        if ok_r and ok_c and ok_b: valid += 1
    return valid, row_v, col_v, box_v, B


# ────────────────────────────────────────────────────────────
#  Sample visualization
# ────────────────────────────────────────────────────────────

def _cont_tokens_to_level_features(cont_tokens, level_sizes, device):
    """Decode flat continuous feature vectors → per-level 2D feature maps.

    Args:
        cont_tokens: (B, total_tokens, feat_dim) float
        level_sizes: list of spatial sizes, finest-first (e.g., [9])
    Returns:
        level_features: {spatial_size: (B, feat_dim, S, S)}
    """
    B = cont_tokens.shape[0]
    offset = 0
    level_features = {}
    for s in sorted(level_sizes, reverse=True):
        n_tok = s * s
        level_feat = cont_tokens[:, offset:offset + n_tok, :].to(device)  # (B, s*s, D)
        offset += n_tok
        feat_dim = level_feat.shape[-1]
        level_features[s] = level_feat.transpose(1, 2).view(B, feat_dim, s, s)
    return level_features


def _tok_ids_to_level_features(tok_ids, level_sizes, discretizer, device):
    """Decode flat token IDs → per-level 2D feature maps.

    Args:
        tok_ids: (B, total_tokens) long
        level_sizes: list of spatial sizes, finest-first (e.g., [8, 4, 2, 1])
        discretizer: FSQDiscretizer / VQDiscretizer
    Returns:
        level_features: {spatial_size: (B, feat_channels, S, S)}
    """
    B = tok_ids.shape[0]
    offset = 0
    level_features = {}
    for s in sorted(level_sizes, reverse=True):  # finest first
        n_tok = s * s
        level_tok = tok_ids[:, offset:offset + n_tok].to(device)  # (B, s*s)
        offset += n_tok
        quant_slots = discretizer.decode(level_tok)  # (B, s*s, feat_dim)
        feat_dim = quant_slots.shape[-1]
        level_features[s] = quant_slots.transpose(1, 2).view(B, feat_dim, s, s)
    return level_features


def _forward_from_slots(pretrained_model, z, t_batch, slot_features, return_uncond=False):
    """Forward pass for baseline_1d model using pre-decoded slot features.

    Bypasses the encoder: directly injects decoded slot tokens as conditioning.
    Mirrors ``Baseline1DConditionalDiT.forward`` and supports BOTH attention
    modes: ``self_concat`` (legacy Semanticist-style) and ``cross``
    (SlotDiffusion-style image→slot cross-attn).

    Args:
        pretrained_model: Baseline1DConditionalDiT
        z: (B, C, H, W) noisy image
        t_batch: (B,) timestep
        slot_features: (B, num_slots, slot_dim) decoded FSQ features
        return_uncond: if True, use null conditioning
    Returns:
        pred: (B, C, H, W) predicted clean image
    """
    from model_multires import _sinusoidal_timestep_embedding

    B = z.shape[0]
    dtype = z.dtype
    model = pretrained_model
    K = model.in_context_len
    attn_mode = getattr(model, "dit_attn_mode", "self_concat")

    if return_uncond:
        slots = model.null_cond.expand(B, -1, -1).to(dtype)
    else:
        slots = slot_features.to(dtype)

    # Project cond tokens (+ pos embed only in self_concat)
    cond_tokens = model.cond_proj(slots)
    if getattr(model, "cond_pos_embed", None) is not None:
        cond_tokens = cond_tokens + model.cond_pos_embed

    # Patchify image tokens
    img_tokens = model.patch_embed(z)
    img_tokens = img_tokens + model.pos_embed

    # Timestep embedding
    t_freq = _sinusoidal_timestep_embedding(t_batch, model._t_freq_dim)
    t_freq = t_freq.to(dtype=dtype)
    c = model.time_embed(t_freq)

    if attn_mode == "cross":
        # ── Cross-attn: image tokens cross-attend to slots each block;
        #    in-context tokens (if any) join self-attn pool but stay out
        #    of the cross-attn step.
        tokens = img_tokens
        num_in_ctx = 0
        for i, block in enumerate(model.blocks):
            if K > 0 and i == model.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
                ic_tokens = ic_tokens + model.in_context_posemb
                tokens = torch.cat([ic_tokens, tokens], dim=1)
                num_in_ctx = K
            rope_cos, rope_sin = model._build_rope_for_seq(num_in_ctx)
            tokens = block(tokens, cond_tokens, c,
                           num_in_ctx=num_in_ctx,
                           rope_cos=rope_cos, rope_sin=rope_sin)
        img_out = tokens[:, num_in_ctx:]
    else:
        # ── self_concat (Semanticist-style) ──
        tokens = torch.cat([cond_tokens, img_tokens], dim=1)
        num_prefix = model.num_slots

        for i, block in enumerate(model.blocks):
            if K > 0 and i == model.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
                ic_tokens = ic_tokens + model.in_context_posemb
                cond_part = tokens[:, :model.num_slots]
                img_part = tokens[:, model.num_slots:]
                tokens = torch.cat([cond_part, ic_tokens, img_part], dim=1)
                num_prefix = model.num_slots + K

            cur_prefix = model.num_slots + (
                K if (K > 0 and i >= model.in_context_start) else 0)
            rope_cos, rope_sin = model._build_rope_for_seq(cur_prefix)
            tokens = block(tokens, c, rope_cos=rope_cos, rope_sin=rope_sin)
        img_out = tokens[:, num_prefix:]

    img_out = model.final_layer(img_out, c)
    pred = model._unpatchify(img_out)
    return pred


@torch.no_grad()
def decode_tokens_to_images(
    tok_ids, level_sizes, pretrained_model, discretizer, device,
    num_steps=50, guidance_scale=1.0, batch_size=16,
    noise_scale=1.0, t_eps=0.05,
):
    """Decode generated token IDs back to images using pretrained model.

    Uses flow matching ODE (Euler) for DiT backbone, or DDIM for UNet.
    Supports both multi-res and baseline_1d pretrained models.

    Args:
        tok_ids: (N, total_tokens) long
        level_sizes: list of spatial sizes (e.g., [8, 4, 2, 1])
                     or [num_slots] for baseline_1d
        pretrained_model: full multi-res or baseline_1d model
        discretizer: FSQDiscretizer/VQDiscretizer
        device: torch device
    Returns:
        images: (N, C, H, W) float32 in [-1, 1]
    """
    pretrained_model.eval()
    pretrained_model.to(device)
    N = tok_ids.shape[0]
    all_images = []

    image_size = pretrained_model.image_size
    in_channels = pretrained_model._in_channels if hasattr(pretrained_model, '_in_channels') else 3
    vae_factor = getattr(pretrained_model, 'vae_downsample_factor', 1)
    latent_size = image_size // vae_factor
    is_multires_dit = hasattr(pretrained_model, 'forward_from_level_features')
    is_baseline_1d = hasattr(pretrained_model, 'num_slots') and not is_multires_dit

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_tok = tok_ids[start:end].to(device)
        B_cur = batch_tok.shape[0]

        z = noise_scale * torch.randn(
            B_cur, in_channels, latent_size, latent_size, device=device)

        if is_baseline_1d:
            # Decode token IDs → slot features (B, num_slots, slot_dim)
            slot_features = discretizer.decode(batch_tok)  # (B, num_slots, slot_dim)

            # Flow matching ODE (Euler)
            timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_cur
                t_batch = t_cur.expand(B_cur)
                t_expand = t_cur.view(1, 1, 1, 1)

                if guidance_scale != 1.0:
                    x_cond = _forward_from_slots(
                        pretrained_model, z, t_batch, slot_features, return_uncond=False)
                    x_uncond = _forward_from_slots(
                        pretrained_model, z, t_batch, slot_features, return_uncond=True)
                    v_cond = (x_cond - z) / (1.0 - t_expand).clamp_min(t_eps)
                    v_uncond = (x_uncond - z) / (1.0 - t_expand).clamp_min(t_eps)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    x_pred = _forward_from_slots(
                        pretrained_model, z, t_batch, slot_features, return_uncond=False)
                    v = (x_pred - z) / (1.0 - t_expand).clamp_min(t_eps)
                z = z + dt * v

        elif is_multires_dit:
            # Decode token IDs → per-level feature maps
            level_features = _tok_ids_to_level_features(
                batch_tok, level_sizes, discretizer, device)
            # Flow matching ODE (Euler) with forward_from_level_features
            timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_cur
                t_batch = t_cur.expand(B_cur)
                t_expand = t_cur.view(1, 1, 1, 1)

                if guidance_scale != 1.0:
                    x_cond = pretrained_model.forward_from_level_features(
                        z, t_batch, level_features, return_uncond=False)
                    x_uncond = pretrained_model.forward_from_level_features(
                        z, t_batch, level_features, return_uncond=True)
                    v_cond = (x_cond - z) / (1.0 - t_expand).clamp_min(t_eps)
                    v_uncond = (x_uncond - z) / (1.0 - t_expand).clamp_min(t_eps)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    x_pred = pretrained_model.forward_from_level_features(
                        z, t_batch, level_features, return_uncond=False)
                    v = (x_pred - z) / (1.0 - t_expand).clamp_min(t_eps)

                z = z + dt * v
        else:
            # UNet backbone — DDIM sampling
            from diffusers import DDIMScheduler
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="v_prediction",
            )
            scheduler.set_timesteps(num_steps, device=device)

            # Decode token IDs → per-level feature maps
            level_features = _tok_ids_to_level_features(
                batch_tok, level_sizes, discretizer, device)

            # Build upsampled features for UNet injection
            upsampled = {}
            for s, feat_2d in level_features.items():
                if str(s) in pretrained_model.level_upsamplers:
                    upsampled[s] = pretrained_model.level_upsamplers[str(s)](feat_2d)
                else:
                    upsampled[s] = feat_2d

            for t_idx in scheduler.timesteps:
                t = t_idx.expand(B_cur).to(device)
                down_inj = {}
                for s_key in upsampled:
                    if s_key in pretrained_model._encoder_to_block:
                        block_idx = pretrained_model._encoder_to_block[s_key]
                        down_inj[block_idx] = pretrained_model.down_injection_convs[
                            str(s_key)](upsampled[s_key])
                mid_res = None
                coarsest = min(upsampled.keys())
                if coarsest in upsampled and hasattr(pretrained_model, 'mid_injection_conv'):
                    mid_res = pretrained_model.mid_injection_conv(upsampled[coarsest])
                pred = pretrained_model._forward_unet(z, t, down_inj, mid_res)
                z = scheduler.step(pred, t_idx, z).prev_sample

        all_images.append(z.cpu())

    return torch.cat(all_images, dim=0).clamp(-1, 1)


@torch.no_grad()
def decode_continuous_tokens_to_images(
    cont_tokens, level_sizes, pretrained_model, device,
    num_steps=50, guidance_scale=1.0, batch_size=16,
    noise_scale=1.0, t_eps=0.05,
):
    """Decode generated continuous feature vectors back to images.

    Same as decode_tokens_to_images but skips the discretizer decode step.

    Args:
        cont_tokens: (N, total_tokens, feat_dim) float
        level_sizes: list of spatial sizes
        pretrained_model: full multi-res model
        device: torch device
    Returns:
        images: (N, C, H, W) float32 in [-1, 1]
    """
    pretrained_model.eval()
    pretrained_model.to(device)
    N = cont_tokens.shape[0]
    all_images = []

    image_size = pretrained_model.image_size
    in_channels = pretrained_model._in_channels if hasattr(pretrained_model, '_in_channels') else 3
    vae_factor = getattr(pretrained_model, 'vae_downsample_factor', 1)
    latent_size = image_size // vae_factor

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_cont = cont_tokens[start:end].to(device)
        B_cur = batch_cont.shape[0]

        z = noise_scale * torch.randn(
            B_cur, in_channels, latent_size, latent_size, device=device)

        # Convert continuous tokens → per-level feature maps
        level_features = _cont_tokens_to_level_features(
            batch_cont, level_sizes, device)

        # Flow matching ODE (Euler)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur
            t_batch = t_cur.expand(B_cur)
            t_expand = t_cur.view(1, 1, 1, 1)

            if guidance_scale != 1.0:
                x_cond = pretrained_model.forward_from_level_features(
                    z, t_batch, level_features, return_uncond=False)
                x_uncond = pretrained_model.forward_from_level_features(
                    z, t_batch, level_features, return_uncond=True)
                v_cond = (x_cond - z) / (1.0 - t_expand).clamp_min(t_eps)
                v_uncond = (x_uncond - z) / (1.0 - t_expand).clamp_min(t_eps)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                x_pred = pretrained_model.forward_from_level_features(
                    z, t_batch, level_features, return_uncond=False)
                v = (x_pred - z) / (1.0 - t_expand).clamp_min(t_eps)

            z = z + dt * v

        # VAE decode if needed
        if vae_factor > 1 and hasattr(pretrained_model, 'vae'):
            z = pretrained_model.vae.decode(z / pretrained_model.vae_scaling).sample
        all_images.append(z.cpu().float())

    return torch.cat(all_images, dim=0).clamp(-1, 1)


def save_sample_grid(images, path, nrow=8):
    """Save a grid of images as PNG."""
    from torchvision.utils import make_grid
    grid = make_grid(images * 0.5 + 0.5, nrow=nrow, padding=2)
    grid = grid.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    Image.fromarray(grid).save(path)


def save_sample_grid_with_hints(
    images, path, cond_digits, known_mask, grid_hw=9, nrow=8,
    border_color=(0, 200, 0), text_color=(0, 180, 0), text_bg=(255, 255, 255),
):
    """Save image grid with hint cells outlined and the given digit overlaid.

    Args:
        images: (B, C, H, W) in [-1, 1] — each image is a grid_hw × grid_hw board
        cond_digits: (B, grid_hw*grid_hw) long — condition digits (0-8 = digit 1-9, SUDOKU_MASK_ID = unknown)
        known_mask:  (B, grid_hw*grid_hw) bool — True for hint cells
    """
    from torchvision.utils import make_grid
    from PIL import ImageDraw, ImageFont

    grid = make_grid(images * 0.5 + 0.5, nrow=nrow, padding=2)
    grid = grid.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    img = Image.fromarray(grid).convert("RGB")
    draw = ImageDraw.Draw(img)

    B, _, H, W = images.shape
    cell_h = H // grid_hw
    cell_w = W // grid_hw
    padding = 2
    ncol = nrow

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(8, cell_h // 3))
    except Exception:
        font = ImageFont.load_default()

    km = known_mask.detach().cpu().numpy() if hasattr(known_mask, "detach") else known_mask
    cd = cond_digits.detach().cpu().numpy() if hasattr(cond_digits, "detach") else cond_digits

    for b in range(B):
        row = b // ncol
        col = b % ncol
        ox = padding + col * (W + padding)
        oy = padding + row * (H + padding)
        for idx in range(grid_hw * grid_hw):
            if not bool(km[b, idx]):
                continue
            r = idx // grid_hw
            c = idx % grid_hw
            x0 = ox + c * cell_w
            y0 = oy + r * cell_h
            x1 = x0 + cell_w - 1
            y1 = y0 + cell_h - 1
            draw.rectangle([x0, y0, x1, y1], outline=border_color, width=2)
            digit_val = int(cd[b, idx]) + 1  # 0-8 → 1-9
            txt = str(digit_val)
            bbox = font.getbbox(txt)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + 1
            ty = y0 + 1
            draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1],
                           fill=text_bg)
            draw.text((tx, ty - bbox[1]), txt, fill=text_color, font=font)

    img.save(path)


@torch.no_grad()
def compute_fid(
    model, args, accelerator,
    pretrained_model, discretizer, level_sizes,
    step: int,
):
    """Generative FID: sample tokens → decode to images → compare to real.

    Multi-GPU: each rank generates its share of samples.  Main process
    merges per-rank dirs and runs torch_fidelity.  Follows the same
    pattern as main_multires.py::evaluate_fid.
    """
    import torch_fidelity

    model.eval()
    device = accelerator.device

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    total = args.fid_num_samples
    is_baseline_1d = hasattr(pretrained_model, 'num_slots')
    seq_len = sum(level_sizes) if is_baseline_1d else sum(s * s for s in level_sizes)

    per_gpu = math.ceil(total / world_size)
    my_start = rank * per_gpu
    my_end = min(my_start + per_gpu, total)
    my_count = my_end - my_start

    gen_dir = os.path.join(args.output_dir, f"fid_gen_step{step}_rank{rank}")
    real_dir = os.path.join(args.output_dir, f"fid_real_step{step}_rank{rank}")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    # Prepare real images (from val set)
    val_root = os.path.join(args.dataset_root, "val")
    if not os.path.isdir(val_root):
        val_root = args.dataset_root
    from torchvision.datasets import ImageFolder
    real_transform = transforms.Compose([
        transforms.Resize(args.image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    real_ds = ImageFolder(val_root, transform=real_transform)

    for i in range(my_start, my_end):
        idx = i % len(real_ds)
        img_t, _ = real_ds[idx]
        real_img = transforms.ToPILImage()(img_t)
        real_img.save(os.path.join(real_dir, f"{i:06d}.png"))

    # Generate images
    gen_bs = min(args.batch_size, 32)
    generated = 0

    while generated < my_count:
        bs = min(gen_bs, my_count - generated)
        class_labels = torch.randint(0, args.num_classes, (bs,), device=device)

        sample_kwargs = dict(
            batch_size=bs, seq_len=seq_len,
            num_steps=args.eval_num_steps, device=device,
            sampler=args.sampler, noise_removal=True,
            class_labels=class_labels,
        )
        if args.model_type == "ar":
            sample_kwargs.update(temperature=args.ar_temperature,
                                 top_k=args.ar_top_k, top_p=args.ar_top_p)

        tok = model.sample(**sample_kwargs)
        images = decode_tokens_to_images(
            tok, level_sizes, pretrained_model, discretizer, device,
            num_steps=args.decode_num_steps, batch_size=bs,
        )

        images_01 = (images * 0.5 + 0.5).clamp(0, 1)
        for j in range(images_01.shape[0]):
            img_idx = my_start + generated + j
            gen_img = transforms.ToPILImage()(images_01[j].cpu())
            gen_img.save(os.path.join(gen_dir, f"{img_idx:06d}.png"))
        generated += bs

    accelerator.wait_for_everyone()

    fid_value = None
    if accelerator.is_main_process:
        # Merge per-rank dirs
        merged_gen = os.path.join(args.output_dir, f"fid_gen_step{step}")
        merged_real = os.path.join(args.output_dir, f"fid_real_step{step}")
        os.makedirs(merged_gen, exist_ok=True)
        os.makedirs(merged_real, exist_ok=True)

        for r in range(world_size):
            for prefix, merged in [("fid_gen", merged_gen),
                                    ("fid_real", merged_real)]:
                rank_dir = os.path.join(
                    args.output_dir, f"{prefix}_step{step}_rank{r}")
                if os.path.isdir(rank_dir):
                    for fname in os.listdir(rank_dir):
                        shutil.move(os.path.join(rank_dir, fname),
                                    os.path.join(merged, fname))
                    shutil.rmtree(rank_dir, ignore_errors=True)

        num_gen = len(os.listdir(merged_gen))
        accelerator.print(
            f"[fid] step={step} | {num_gen} generated, computing metrics...")

        try:
            kwargs = dict(input1=merged_gen, cuda=True, fid=True, isc=True)
            if args.fid_real_dir:
                kwargs["input2"] = args.fid_real_dir
            else:
                kwargs["input2"] = merged_real

            metrics = torch_fidelity.calculate_metrics(**kwargs)
            fid_value = metrics.get("frechet_inception_distance")
            isc_value = metrics.get("inception_score_mean")
            accelerator.print(
                f"[fid] step={step} | FID: {fid_value:.2f} | IS: {isc_value:.2f}")

            if args.log_with:
                accelerator.log({"eval/fid": fid_value, "eval/is": isc_value},
                                step=step)

            # Append to log
            log_path = os.path.join(args.output_dir, "fid_log.txt")
            with open(log_path, "a") as f:
                f.write(f"step={step} fid={fid_value:.4f} is={isc_value:.4f}\n")
        except Exception as e:
            accelerator.print(f"[fid] computation failed: {e}")

        shutil.rmtree(merged_gen, ignore_errors=True)
        shutil.rmtree(merged_real, ignore_errors=True)
    else:
        shutil.rmtree(gen_dir, ignore_errors=True)
        shutil.rmtree(real_dir, ignore_errors=True)

    accelerator.wait_for_everyone()
    model.train()
    return fid_value


# ────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_save(
    diffusion, step, args, accelerator, ema,
    pretrained_model=None, discretizer=None, level_sizes=None,
    clevr_cond_encoder=None, val_dataset=None,
    clevr_detector=None, clevr_classifier=None,
    train_dataset=None,
    sudoku_cell_cond_encoder=None,
):
    """All ranks participate in eval (distributed sampling + decode).
    Only rank 0 does condition eval and saves files."""
    import time as _time
    _rank = accelerator.process_index
    device = accelerator.device

    print(f"[eval/debug] rank={_rank} entering evaluate_and_save step={step} "
          f"(GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / "
          f"reserved: {torch.cuda.memory_reserved(device)/1e9:.2f}GB)", flush=True)

    # Free fragmented CUDA cache BEFORE heavy eval allocations
    torch.cuda.empty_cache()

    model = accelerator.unwrap_model(diffusion)
    model.eval()

    params = list(model.parameters())
    if ema is not None:
        print(f"[eval/debug] rank={_rank} ema.store start "
              f"(GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / "
              f"reserved: {torch.cuda.memory_reserved(device)/1e9:.2f}GB)", flush=True)
        ema.store(params)
        ema.copy_to(params)
        print(f"[eval/debug] rank={_rank} ema.copy_to done "
              f"(GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB)", flush=True)

    save_dir = os.path.join(args.output_dir, "eval_samples")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # Barrier: ensure all ranks enter eval together before any collective ops
    print(f"[eval/debug] rank={_rank} entering wait_for_everyone (pre-eval barrier)", flush=True)
    _tb0 = _time.time()
    accelerator.wait_for_everyone()
    print(f"[eval/debug] rank={_rank} wait_for_everyone done ({_time.time()-_tb0:.1f}s)", flush=True)

    if args.dataset_type == "sudoku" and args.grid_only:
        if accelerator.is_main_process:
            _eval_sudoku(model, step, args, accelerator, save_dir)
    elif args.dataset_type == "sudoku" and not args.grid_only:
        # All ranks participate in eval (sampling can be slow with diffusion head)
        cond_enc = clevr_cond_encoder
        if cond_enc is not None and hasattr(cond_enc, 'module'):
            cond_enc = cond_enc.module
        cc_enc = sudoku_cell_cond_encoder
        if cc_enc is not None and hasattr(cc_enc, 'module'):
            cc_enc = cc_enc.module
        _eval_sudoku_image(model, step, args, accelerator, save_dir,
                           pretrained_model, discretizer, level_sizes,
                           val_dataset=val_dataset,
                           sudoku_cond_encoder=cond_enc,
                           train_dataset=train_dataset,
                           sudoku_cell_cond_encoder=cc_enc)
    elif args.dataset_type == "imagenet":
        if accelerator.is_main_process:
            _eval_imagenet(model, step, args, accelerator, save_dir,
                           pretrained_model, discretizer, level_sizes)
    elif args.dataset_type == "clevr":
        print(f"[eval/debug] rank={_rank} starting _eval_clevr (val)", flush=True)
        _teval0 = _time.time()
        _eval_clevr(model, step, args, accelerator, save_dir,
                    pretrained_model, discretizer, level_sizes,
                    clevr_cond_encoder, val_dataset,
                    clevr_detector=clevr_detector,
                    clevr_classifier=clevr_classifier)
        print(f"[eval/debug] rank={_rank} _eval_clevr (val) done "
              f"({_time.time()-_teval0:.1f}s, "
              f"GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB)", flush=True)
        # Also eval on train set if provided
        if train_dataset is not None:
            train_save_dir = os.path.join(args.output_dir, "eval_train_samples")
            if accelerator.is_main_process:
                os.makedirs(train_save_dir, exist_ok=True)
            print(f"[eval/debug] rank={_rank} starting _eval_clevr (train)", flush=True)
            _teval1 = _time.time()
            _eval_clevr(model, step, args, accelerator, train_save_dir,
                        pretrained_model, discretizer, level_sizes,
                        clevr_cond_encoder, train_dataset,
                        clevr_detector=clevr_detector,
                        clevr_classifier=clevr_classifier,
                        log_prefix="eval_train")
            print(f"[eval/debug] rank={_rank} _eval_clevr (train) done "
                  f"({_time.time()-_teval1:.1f}s)", flush=True)

    # Move pretrained model back to CPU to free GPU for training
    if pretrained_model is not None:
        pretrained_model.cpu()
        torch.cuda.empty_cache()

    # Sync all ranks before restoring EMA / resuming training.
    print(f"[eval/debug] rank={_rank} entering final wait_for_everyone", flush=True)
    _tb1 = _time.time()
    accelerator.wait_for_everyone()
    print(f"[eval/debug] rank={_rank} final wait_for_everyone done "
          f"({_time.time()-_tb1:.1f}s)", flush=True)

    if ema is not None:
        ema.restore(params)
    model.train()


def _eval_sudoku(model, step, args, accelerator, save_dir):
    """Sudoku eval: generate grids, check rules."""
    device = accelerator.device
    grid_hw = args.grid_hw
    seq_len = grid_hw * grid_hw

    sample_kwargs = dict(
        batch_size=args.eval_num_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
    )
    if args.model_type == "ar":
        sample_kwargs.update(temperature=args.ar_temperature,
                             top_k=args.ar_top_k, top_p=args.ar_top_p)
    tokens = model.sample(**sample_kwargs)
    B = tokens.shape[0]
    grids = tokens.view(B, grid_hw, grid_hw)
    n_valid, n_row, n_col, n_box, n_total = check_sudoku_rules(grids)
    rule_acc = n_valid / max(n_total, 1)
    accelerator.print(
        f"[eval/sudoku] step={step} rule_acc={rule_acc:.4f} "
        f"({n_valid}/{n_total})")

    txt_path = os.path.join(save_dir, f"step_{step:07d}_sudoku.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step} rule_acc={rule_acc:.6f}\n")
        for i in range(min(8, B)):
            f.write(f"sample {i}: {tokens[i].tolist()}\n")

    if args.log_with:
        accelerator.log({"eval/rule_acc": rule_acc}, step=step)


def _find_error_cells(grid, grid_hw=9):
    """Find cells that violate sudoku rules (row/col/box duplicates).

    Returns a set of (r, c) tuples for cells involved in violations.
    """
    errors = set()
    box_h = box_w = int(grid_hw ** 0.5)  # 3 for 9x9

    for i in range(grid_hw):
        # Row check
        row = grid[i]
        for v in range(1, grid_hw + 1):
            cols = [c for c in range(grid_hw) if row[c] == v]
            if len(cols) > 1:
                for c in cols:
                    errors.add((i, c))

        # Col check
        col = grid[:, i]
        for v in range(1, grid_hw + 1):
            rows = [r for r in range(grid_hw) if col[r] == v]
            if len(rows) > 1:
                for r in rows:
                    errors.add((r, i))

    # Box check
    for br in range(0, grid_hw, box_h):
        for bc in range(0, grid_hw, box_w):
            box = grid[br:br+box_h, bc:bc+box_w].flatten()
            for v in range(1, grid_hw + 1):
                positions = []
                for idx, val in enumerate(box):
                    if val == v:
                        r = br + idx // box_w
                        c = bc + idx % box_w
                        positions.append((r, c))
                if len(positions) > 1:
                    for pos in positions:
                        errors.add(pos)

    return errors


def _render_sudoku_grid_frame(draw, grid, grid_hw, cell_size, font,
                               mask_positions=None, error_cells=None,
                               hint_cells=None, hint_mismatch_cells=None,
                               hint_gt_digits=None):
    """Draw a single sudoku grid frame.

    Args:
        mask_positions: (L,) bool tensor — True = still masked
        error_cells:    set of (r,c) — cells that violate rules (final frame only)
        hint_cells:     set of (r,c) — cells given as hints (green background)
        hint_mismatch_cells: set of (r,c) — hints whose model/classifier output
                             doesn't match the given digit (red-tinted hint)
        hint_gt_digits: dict {(r,c) -> int given_digit}, for rendering expected
                        value at mismatch cells.
    """
    for r in range(grid_hw):
        for c in range(grid_hw):
            x0 = c * cell_size
            y0 = r * cell_size
            cell_idx = r * grid_hw + c

            is_masked = (mask_positions is not None
                         and cell_idx < mask_positions.numel()
                         and mask_positions[cell_idx].item())
            is_error = (error_cells is not None and (r, c) in error_cells)
            is_hint = (hint_cells is not None and (r, c) in hint_cells)
            is_hint_mismatch = (hint_mismatch_cells is not None
                                and (r, c) in hint_mismatch_cells)

            if is_hint_mismatch:
                # Hint given but model/classifier output disagrees — red-tint hint.
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size],
                               fill=(255, 210, 210))
                txt = str(int(grid[r, c]))
                txt_color = (200, 0, 0)
                # Small superscript showing the expected hint digit
                if hint_gt_digits is not None and (r, c) in hint_gt_digits:
                    exp_d = int(hint_gt_digits[(r, c)])
                    try:
                        sm_font = font.font_variant(size=max(12, cell_size // 4))
                    except Exception:
                        sm_font = font
                    draw.text((x0 + 2, y0 + 2), f"({exp_d})",
                              fill=(120, 0, 0), font=sm_font)
            elif is_hint:
                # Hint cell — green background throughout (always shown,
                # regardless of mask state, since hints are clean from t=0).
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size],
                               fill=(200, 240, 200))
                txt = str(int(grid[r, c]))
                txt_color = (0, 110, 0)
            elif is_masked:
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size],
                               fill=(200, 200, 200))
                txt = "\u00b7"
                txt_color = (0, 0, 0)
            elif is_error:
                draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size],
                               fill=(255, 220, 220))
                txt = str(int(grid[r, c]))
                txt_color = (200, 0, 0)
            else:
                txt = str(int(grid[r, c]))
                txt_color = (0, 0, 0)

            bbox = font.getbbox(txt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x0 + (cell_size - tw) // 2
            ty = y0 + (cell_size - th) // 2
            draw.text((tx, ty), txt, fill=txt_color, font=font)

    for i in range(grid_hw + 1):
        lw = 3 if i % 3 == 0 else 1
        draw.line([(i * cell_size, 0), (i * cell_size, grid_hw * cell_size)],
                  fill=(0, 0, 0), width=lw)
        draw.line([(0, i * cell_size), (grid_hw * cell_size, i * cell_size)],
                  fill=(0, 0, 0), width=lw)


def _render_sudoku_grid_video(
    history, final_grid, sample_idx=0, save_path="video_grid.mp4",
    max_frames=32, fps=8, grid_hw=9, hint_mask=None, title=None,
    hint_gt_digits=None,
):
    """Render video of 9x9 digit grid being filled during denoising.

    Uses the final decoded digit grid and only checks mask positions
    from history at each step — no intermediate DiT decoding.
    Final frame highlights rule-violating cells in red.
    Hint cells (given as condition) shown in blue throughout.

    Args:
        history:    list of (B, L) token tensors from discrete diffusion
        final_grid: (B, grid_hw, grid_hw) int tensor, classified digits (1-9)
        sample_idx: which sample in the batch to render
        hint_mask:  (B, 81) bool tensor — True = hint cell. None for uncond.
        title:      optional string shown at top (e.g. "easy", "hard")
    """
    import numpy as np
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    n_total = len(history)
    if n_total > max_frames:
        indices = [0] + list(
            range(1, n_total - 1,
                  max(1, (n_total - 2) // (max_frames - 2)))
        ) + [n_total - 1]
        indices = sorted(set(indices))
    else:
        indices = list(range(n_total))

    # Support both discrete history (int tokens) and continuous mask_history (bool)
    _is_bool_history = (history[0].dtype == torch.bool)
    mask_index = None if _is_bool_history else history[0].max().item()
    grid = final_grid[sample_idx].cpu().numpy()  # (9, 9)
    error_cells = _find_error_cells(grid, grid_hw)

    # Build hint_cells set for this sample
    hint_cells = None
    hint_mismatch_cells = None
    hint_gt_map = None
    if hint_mask is not None:
        hm = hint_mask[sample_idx].cpu()  # (81,)
        hint_cells = set()
        for ci in range(min(hm.numel(), grid_hw * grid_hw)):
            if hm[ci].item():
                hint_cells.add((ci // grid_hw, ci % grid_hw))

        # Compare given hint digits with classified final_grid → mismatches
        if hint_gt_digits is not None:
            gt = hint_gt_digits[sample_idx].cpu()  # (81,) long in [1..9]
            hint_mismatch_cells = set()
            hint_gt_map = {}
            for (r, c) in hint_cells:
                ci = r * grid_hw + c
                exp_d = int(gt[ci].item())
                hint_gt_map[(r, c)] = exp_d
                if int(grid[r, c]) != exp_d:
                    hint_mismatch_cells.add((r, c))

    cell_size = 48
    img_w = grid_hw * cell_size
    title_h = 28 if title else 0
    grid_h = grid_hw * cell_size
    footer_h = 32
    img_h = title_h + grid_h + footer_h
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 28)
    except (IOError, OSError):
        font = ImageFont.load_default()
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (IOError, OSError):
        small_font = ImageFont.load_default()

    frames_np = []
    for fi, idx in enumerate(indices):
        if _is_bool_history:
            mask_positions = history[idx][sample_idx]  # (L,) bool
        else:
            mask_positions = (history[idx][sample_idx] == mask_index)  # (L,)
        n_masked = mask_positions.sum().item()
        n_total_tok = mask_positions.numel()
        pct = 100.0 * (1 - n_masked / n_total_tok)

        is_final = (fi == len(indices) - 1)

        frame = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(frame)

        # Title
        if title:
            n_hints = len(hint_cells) if hint_cells else 0
            title_txt = f"[{title}]" + (f"  {n_hints} hints" if n_hints > 0 else "")
            draw.text((4, 4), title_txt, fill=(0, 0, 0), font=small_font)

        # Offset grid drawing by title height
        grid_frame = Image.new("RGB", (img_w, grid_h), color=(255, 255, 255))
        grid_draw = ImageDraw.Draw(grid_frame)

        _render_sudoku_grid_frame(
            grid_draw, grid, grid_hw, cell_size, font,
            mask_positions=mask_positions,
            error_cells=error_cells if is_final else None,
            hint_cells=hint_cells,
            hint_mismatch_cells=hint_mismatch_cells,
            hint_gt_digits=hint_gt_map)

        frame.paste(grid_frame, (0, title_h))

        # Footer
        footer_y = title_h + grid_h + 4
        n_errors = len(error_cells)
        if is_final and n_errors > 0:
            draw.text((4, footer_y),
                      f"DONE  {n_errors} cells violate rules",
                      fill=(200, 0, 0), font=small_font)
        elif is_final:
            draw.text((4, footer_y),
                      f"DONE  valid sudoku!",
                      fill=(0, 150, 0), font=small_font)
        else:
            draw.text((4, footer_y),
                      f"step {idx}/{n_total-1}  ({pct:.0f}% denoised)",
                      fill=(80, 80, 80), font=small_font)

        frames_np.append(np.array(frame))

    if not frames_np:
        return

    h, w = frames_np[0].shape[:2]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        frames_np = [
            np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                   mode='constant', constant_values=255)
            for f in frames_np
        ]

    n_hold = max(1, fps * 3)
    frames_np.extend([frames_np[-1]] * n_hold)

    base, _ = os.path.splitext(save_path)
    try:
        save_path = base + ".mp4"
        imageio.mimwrite(save_path, frames_np, format="FFMPEG",
                         fps=fps, codec="libx264", pixelformat="yuv420p")
    except (ImportError, OSError):
        save_path = base + ".gif"
        imageio.mimwrite(save_path, frames_np, duration=1000 // fps)




def _eval_sudoku_image(model, step, args, accelerator, save_dir,
                       pretrained_model, discretizer, level_sizes,
                       val_dataset=None, sudoku_cond_encoder=None,
                       train_dataset=None,
                       sudoku_cell_cond_encoder=None):
    """Sudoku image eval with digit-grid conditioning (like CLEVR prefix).

    1) Unconditional generation (all-MASK condition) with 3 samplers
    2) Difficulty-based conditioned generation (easy/medium/hard)
       - easy:   54-80 given digits → fill 1-27 cells
       - medium: 27-53 given digits → fill 28-54 cells
       - hard:   0-26  given digits → fill 55-81 cells
    3) MP4 video rendering of denoising process
    """
    device = accelerator.device
    is_baseline_1d = hasattr(pretrained_model, 'num_slots')
    seq_len = sum(level_sizes) if is_baseline_1d else sum(s * s for s in level_sizes)
    n_samples = min(args.eval_num_samples, 64)

    from SRM.evaluation.sudoku_eval_only import MnistSudokuEvalOnly
    evaluator = MnistSudokuEvalOnly(
        mnist_classifier_path=args.classifier_pth,
        grid_size=(args.grid_hw, args.grid_hw))

    # ── Define sampler configs (like ImageNet/CLEVR) ──
    if getattr(args, 'use_diffusion_head', False):
        # Continuous mode: support ddpm + confidence (top1/cosine)
        sampler_configs = [
            {"name": "ddpm_cache", "sampler": "ddpm_cache",
             "tokens_per_step": 0},
            {"name": "confidence_top1", "sampler": "confidence",
             "tokens_per_step": 1},
            {"name": "confidence_cosine", "sampler": "confidence",
             "tokens_per_step": 0},
        ]
    elif args.model_type == "ar":
        sampler_configs = [{"name": "ar", "sampler": "ar"}]
    else:
        sampler_configs = [
            {"name": "ddpm_cache", "sampler": "ddpm_cache",
             "tokens_per_step": 0},
            {"name": "confidence_top1", "sampler": "confidence",
             "tokens_per_step": 1},
            {"name": "confidence_cosine", "sampler": "confidence",
             "tokens_per_step": 0},
        ]

    eval_video_samples = getattr(args, "eval_video_samples", 4)
    eval_save_format = getattr(args, "eval_save_format", "mp4")

    # ─────────────────────────────────────────────────────
    #  1) Unconditional generation (all-MASK condition)
    # ─────────────────────────────────────────────────────
    uncond_cond_tokens = None
    if sudoku_cond_encoder is not None:
        # All digits masked → unconditional
        all_mask = torch.full((n_samples, SUDOKU_GRID_LEN),
                              SUDOKU_MASK_ID, dtype=torch.long, device=device)
        uncond_cond_tokens = sudoku_cond_encoder(all_mask)

    use_diffusion_head = getattr(args, 'use_diffusion_head', False)
    # For diffusion head: get feat_dim from pretrained encoder
    _diff_head_feat_dim = getattr(args, '_diff_head_feat_dim', 16)
    if use_diffusion_head and hasattr(model, 'diff_head') and model.diff_head is not None:
        _diff_head_feat_dim = model.diff_head.in_channels

    for sc in sampler_configs:
        tag = sc["name"]

        if use_diffusion_head:
            # ── Continuous mode: sample_continuous ──
            sample_kwargs = dict(
                batch_size=n_samples, seq_len=seq_len,
                feat_dim=_diff_head_feat_dim,
                num_steps=args.eval_num_steps, device=device,
                sampler=sc["sampler"],
                tokens_per_step=sc.get("tokens_per_step", 0),
                cond_tokens=uncond_cond_tokens,
                temperature=getattr(args, 'diff_head_temperature', 1.0),
                cfg=getattr(args, 'diff_head_cfg', 1.0),
                cfg_schedule=getattr(args, 'cfg_schedule', 'constant'),
                cfg_mode=getattr(args, 'cfg_mode', 'head'),
                null_class_index=getattr(args, 'num_classes', None)
                    if args.dataset_type == "imagenet" else None,
            )
            need_video = (eval_video_samples > 0 and eval_save_format in ("mp4", "gif"))
            if need_video:
                sample_kwargs["return_history"] = True
                cont_tokens_sampled, mask_history = model.sample_continuous(**sample_kwargs)
            else:
                cont_tokens_sampled = model.sample_continuous(**sample_kwargs)
                mask_history = None

            # Decode continuous tokens → images
            images = decode_continuous_tokens_to_images(
                cont_tokens_sampled, level_sizes, pretrained_model, device,
                num_steps=args.decode_num_steps, batch_size=16)
            history = mask_history  # list of (B, L) bool tensors
        else:
            # ── Discrete mode: standard sampling ──
            sample_kwargs = dict(
                batch_size=n_samples, seq_len=seq_len,
                num_steps=args.eval_num_steps, device=device,
                sampler=sc["sampler"], noise_removal=True,
                tokens_per_step=sc.get("tokens_per_step", 0),
                cond_tokens=uncond_cond_tokens,
            )
            if args.model_type == "ar":
                sample_kwargs.update(temperature=args.ar_temperature,
                                     top_k=args.ar_top_k, top_p=args.ar_top_p)

            # Sample with history for video rendering
            need_video = (eval_video_samples > 0 and eval_save_format in ("mp4", "gif"))
            if need_video:
                sample_kwargs["return_history"] = True
                result = model.sample(**sample_kwargs)
                tokens, history = result
            else:
                tokens = model.sample(**sample_kwargs)
                history = None

            # Decode tokens → images
            images = decode_tokens_to_images(
                tokens, level_sizes, pretrained_model, discretizer, device,
                num_steps=args.decode_num_steps, batch_size=16)

        # Decode + save + eval — only rank 0
        if accelerator.is_main_process:
            # Save image grid
            img_path = os.path.join(save_dir, f"step_{step:07d}_sudoku_{tag}.png")
            save_sample_grid(images, img_path, nrow=8)
            accelerator.print(
                f"[eval/sudoku_image] step={step} [{tag}] saved → {img_path}")

            # Save denoising grid videos
            if need_video and history is not None:
                if images.shape[1] == 3:
                    eval_imgs = images.mean(dim=1, keepdim=True)
                else:
                    eval_imgs = images
                final_grids = evaluator.eval_images(eval_imgs.to(device))["discrete"]

                vid_dir = os.path.join(save_dir, "videos")
                os.makedirs(vid_dir, exist_ok=True)
                n_vid = min(eval_video_samples, n_samples)
                for vi in range(n_vid):
                    grid_vid_path = os.path.join(
                        vid_dir,
                        f"step_{step:07d}_sudoku_{tag}_sample{vi}_grid.{eval_save_format}")
                    _render_sudoku_grid_video(
                        history, final_grids,
                        sample_idx=vi, save_path=grid_vid_path,
                        max_frames=args.eval_num_steps, fps=8,
                        grid_hw=args.grid_hw, title=f"uncond / {tag}")
                accelerator.print(
                    f"[eval/sudoku_image] step={step} [{tag}] "
                    f"saved {n_vid} grid videos → {vid_dir}/")

            # MNIST classify → grid → rule check
            _log_sudoku_image_metrics(
                images, evaluator, tag, step, n_samples,
                save_dir, accelerator, args, prefix="eval")

    # ─────────────────────────────────────────────────────
    #  2) Difficulty-based conditioned generation
    #     (like train_AR_cond.sh easy/medium/hard)
    # ─────────────────────────────────────────────────────
    has_prefix = (sudoku_cond_encoder is not None)
    has_val_tokens = (val_dataset is not None
                      and hasattr(val_dataset, 'sudoku_grids')
                      and (has_prefix
                           or hasattr(val_dataset, 'features')
                           or hasattr(val_dataset, 'tok_ids')))
    if has_val_tokens and args.model_type != "ar":
        accelerator.print(
            f"\n========== [eval_cond] split=VAL  step={step}  "
            f"save_dir={save_dir} ==========")
        _eval_sudoku_image_difficulty(
            model, step, args, accelerator, save_dir,
            pretrained_model, discretizer, level_sizes,
            val_dataset, evaluator, sudoku_cond_encoder,
            log_prefix="eval_cond", split_tag="val",
            sudoku_cell_cond_encoder=sudoku_cell_cond_encoder)

    has_train_tokens = (train_dataset is not None
                        and hasattr(train_dataset, 'sudoku_grids')
                        and (has_prefix
                             or hasattr(train_dataset, 'features')
                             or hasattr(train_dataset, 'tok_ids')))
    if has_train_tokens and args.model_type != "ar":
        train_save_dir = os.path.join(
            os.path.dirname(save_dir), "eval_train_samples")
        if accelerator.is_main_process:
            os.makedirs(train_save_dir, exist_ok=True)
        accelerator.print(
            f"\n========== [eval_cond_train] split=TRAIN  step={step}  "
            f"save_dir={train_save_dir} ==========")
        _eval_sudoku_image_difficulty(
            model, step, args, accelerator, train_save_dir,
            pretrained_model, discretizer, level_sizes,
            train_dataset, evaluator, sudoku_cond_encoder,
            log_prefix="eval_cond_train", split_tag="train",
            sudoku_cell_cond_encoder=sudoku_cell_cond_encoder)


# Difficulty levels matching train_AR_cond.sh
SUDOKU_DIFFICULTY_LEVELS = {
    "easy":   (54, 80),   # 54-80 given cells → fill 1-27
    "medium": (27, 53),   # 27-53 given cells → fill 28-54
    "hard":   (0,  26),   # 0-26  given cells → fill 55-81
}


def _eval_sudoku_image_difficulty(
    model, step, args, accelerator, save_dir,
    pretrained_model, discretizer, level_sizes,
    val_dataset, evaluator, sudoku_cond_encoder,
    log_prefix="eval_cond", split_tag="val",
    sudoku_cell_cond_encoder=None,
):
    """Conditioned generation eval with easy/medium/hard difficulty.

    Two conditioning modes:
      * prefix (legacy, --use_sudoku_prefix): masked digit grid → encoder →
        prefix tokens prepended during sampling.
      * inpainting (default): each val sample's own image tokens at known
        cells are fixed as clean tokens; unknown cells start [MASK] and are
        denoised. MDLM carry-over preserves known cells throughout.

    For each difficulty level:
    1. Take GT digit grids from val set
    2. Mask digits according to difficulty (fewer hints = harder)
    3a. (prefix mode) Encode masked digit grid → condition prefix
    3b. (inpainting mode) Build known_mask/known_tokens from val features
    4. Generate FSQ tokens
    5. Decode to images → MNIST classify → rule check
    """
    device = accelerator.device
    is_baseline_1d = hasattr(pretrained_model, 'num_slots')
    seq_len = sum(level_sizes) if is_baseline_1d else sum(s * s for s in level_sizes)
    n_eval = min(len(val_dataset), args.eval_num_samples, 64)

    use_prefix_cond = (sudoku_cond_encoder is not None)
    use_cell_cond = (getattr(args, 'use_sudoku_cell_cond', False)
                     and sudoku_cell_cond_encoder is not None)

    # Gather GT digit grids from val set
    grids = val_dataset.sudoku_grids[:n_eval]  # (n_eval, 9, 9)
    if not isinstance(grids, torch.Tensor):
        grids = torch.tensor(grids)
    x_gt = grids.reshape(n_eval, -1).long().to(device) - 1  # (n_eval, 81) in [0, 8]

    # Gather val image tokens (continuous feats or discrete ids) for inpainting.
    # When using cell_cond, we do NOT inpaint — all cells start masked.
    val_cont_tokens = None
    val_tok_ids = None
    if (not use_prefix_cond) and (not use_cell_cond):
        if hasattr(val_dataset, 'features') and val_dataset.features is not None:
            val_cont_tokens = val_dataset.features[:n_eval].to(device).float()
        elif hasattr(val_dataset, 'tok_ids') and val_dataset.tok_ids is not None:
            val_tok_ids = val_dataset.tok_ids[:n_eval].to(device).long()

    # Use ddpm_cache for conditioned generation
    sampler = "ddpm_cache"

    for level_name, (hint_lo, hint_hi) in SUDOKU_DIFFICULTY_LEVELS.items():
        task_dir = os.path.join(save_dir, level_name)
        os.makedirs(task_dir, exist_ok=True)

        # Build hint mask (deterministic per level, like train_AR.py)
        rng = torch.Generator(device=device)
        rng.manual_seed(hash(level_name) & 0xFFFFFFFF)

        n_hints = torch.randint(
            hint_lo, hint_hi + 1, (n_eval,), device=device, generator=rng)

        known_mask = torch.zeros(n_eval, SUDOKU_GRID_LEN,
                                 dtype=torch.bool, device=device)
        for b in range(n_eval):
            nh = n_hints[b].item()
            if nh > 0:
                perm = torch.randperm(SUDOKU_GRID_LEN, device=device,
                                      generator=rng)[:nh]
                known_mask[b, perm] = True

        # Build masked condition digits (also used for hint-grid viz)
        cond_digits = torch.full((n_eval, SUDOKU_GRID_LEN),
                                 SUDOKU_MASK_ID, dtype=torch.long, device=device)
        cond_digits[known_mask] = x_gt[known_mask]

        # Build conditioning tensors for the modes
        cond_tokens = None
        cell_cond = None
        known_cont_tokens = None
        known_discrete_tokens = None
        if use_cell_cond:
            # Per-cell AdaLN-like: hint digits as cell cond, fully-masked start
            cc_enc = sudoku_cell_cond_encoder
            if hasattr(cc_enc, 'module'):
                cc_enc = cc_enc.module
            cell_cond = cc_enc(cond_digits)  # (n_eval, 81, H)
        elif use_prefix_cond:
            # Legacy: encode masked grid → prefix
            cond_tokens = sudoku_cond_encoder(cond_digits)
        else:
            # Inpainting: fix known positions with this val sample's own
            # image tokens; leave unknowns masked.
            if val_cont_tokens is not None:
                known_cont_tokens = val_cont_tokens
            elif val_tok_ids is not None:
                known_discrete_tokens = val_tok_ids

        # Generate tokens conditioned on digit prefix
        eval_video_samples = getattr(args, "eval_video_samples", 4)
        eval_save_format = getattr(args, "eval_save_format", "mp4")
        use_diffusion_head = getattr(args, 'use_diffusion_head', False)

        if use_diffusion_head:
            _dh_feat_dim = model.diff_head.in_channels if model.diff_head is not None else 16
            dh_sampler_configs = [
                {"name": "ddpm_cache", "sampler": "ddpm_cache", "tokens_per_step": 0},
                {"name": "confidence_top1", "sampler": "confidence", "tokens_per_step": 1},
                {"name": "confidence_cosine", "sampler": "confidence", "tokens_per_step": 0},
            ]
            need_video = (eval_video_samples > 0 and eval_save_format in ("mp4", "gif"))
            for sc in dh_sampler_configs:
                sc_tag = sc["name"]
                sample_kwargs = dict(
                    batch_size=n_eval, seq_len=seq_len,
                    feat_dim=_dh_feat_dim,
                    num_steps=args.eval_num_steps, device=device,
                    sampler=sc["sampler"],
                    tokens_per_step=sc.get("tokens_per_step", 0),
                    cond_tokens=cond_tokens,
                    temperature=getattr(args, 'diff_head_temperature', 1.0),
                    cfg=getattr(args, 'diff_head_cfg', 1.0),
                    cfg_schedule=getattr(args, 'cfg_schedule', 'constant'),
                    known_mask=known_mask if known_cont_tokens is not None else None,
                    known_tokens=known_cont_tokens,
                    cell_cond=cell_cond,
                    cfg_mode=getattr(args, 'cfg_mode', 'head'),
                    null_class_index=getattr(args, 'num_classes', None)
                        if args.dataset_type == "imagenet" else None,
                )
                if need_video:
                    sample_kwargs["return_history"] = True
                    cont_tokens_sampled, mask_history = model.sample_continuous(**sample_kwargs)
                else:
                    cont_tokens_sampled = model.sample_continuous(**sample_kwargs)
                    mask_history = None
                images = decode_continuous_tokens_to_images(
                    cont_tokens_sampled, level_sizes, pretrained_model, device,
                    num_steps=args.decode_num_steps, batch_size=16)
                history = mask_history

                if accelerator.is_main_process:
                    img_path = os.path.join(
                        task_dir,
                        f"step_{step:07d}_sudoku_{level_name}_{sc_tag}.png")
                    save_sample_grid(images, img_path, nrow=8)
                    hint_img_path = os.path.join(
                        task_dir,
                        f"step_{step:07d}_sudoku_{level_name}_{sc_tag}_hints.png")
                    save_sample_grid_with_hints(
                        images, hint_img_path,
                        cond_digits=cond_digits, known_mask=known_mask,
                        grid_hw=args.grid_hw, nrow=8)

                    if need_video and history is not None:
                        if images.shape[1] == 3:
                            eval_imgs = images.mean(dim=1, keepdim=True)
                        else:
                            eval_imgs = images
                        final_grids = evaluator.eval_images(eval_imgs.to(device))["discrete"]
                        vid_dir = os.path.join(task_dir, "videos")
                        os.makedirs(vid_dir, exist_ok=True)
                        n_vid = min(eval_video_samples, n_eval)
                        for vi in range(n_vid):
                            grid_vid_path = os.path.join(
                                vid_dir,
                                f"step_{step:07d}_sudoku_{level_name}_{sc_tag}_sample{vi}_grid.{eval_save_format}")
                            _render_sudoku_grid_video(
                                history, final_grids,
                                sample_idx=vi, save_path=grid_vid_path,
                                max_frames=args.eval_num_steps, fps=8,
                                grid_hw=args.grid_hw, hint_mask=known_mask,
                                title=f"[{split_tag}] {level_name} / {sc_tag}",
                                hint_gt_digits=(x_gt + 1))
                        accelerator.print(
                            f"[{log_prefix}/sudoku_image] step={step} split={split_tag} "
                            f"[{level_name}/{sc_tag}] saved {n_vid} grid videos → {vid_dir}/")

                    _log_sudoku_image_metrics(
                        images, evaluator, f"{level_name}_{sc_tag}",
                        step, n_eval, task_dir, accelerator, args,
                        prefix=log_prefix,
                        gt_digits=x_gt, hint_mask=known_mask)
            continue  # skip the shared save/eval block below
        else:
            need_video = (eval_video_samples > 0 and eval_save_format in ("mp4", "gif"))
            sample_kwargs = dict(
                batch_size=n_eval, seq_len=seq_len,
                num_steps=args.eval_num_steps, device=device,
                sampler=sampler, noise_removal=True,
                cond_tokens=cond_tokens, tokens_per_step=0,
                known_mask=known_mask if known_discrete_tokens is not None else None,
                known_tokens=known_discrete_tokens,
            )
            if need_video:
                sample_kwargs["return_history"] = True
                tokens, history = model.sample(**sample_kwargs)
            else:
                tokens = model.sample(**sample_kwargs)
                history = None

            # Decode tokens → images
            images = decode_tokens_to_images(
                tokens, level_sizes, pretrained_model, discretizer, device,
                num_steps=args.decode_num_steps, batch_size=16)

        # Decode + save + eval — only rank 0
        if accelerator.is_main_process:
            # Save image grid
            img_path = os.path.join(
                task_dir, f"step_{step:07d}_sudoku_{level_name}.png")
            save_sample_grid(images, img_path, nrow=8)
            hint_img_path = os.path.join(
                task_dir, f"step_{step:07d}_sudoku_{level_name}_hints.png")
            save_sample_grid_with_hints(
                images, hint_img_path,
                cond_digits=cond_digits, known_mask=known_mask,
                grid_hw=args.grid_hw, nrow=8)

            # Save denoising grid videos with hint highlighting
            if need_video and history is not None:
                if images.shape[1] == 3:
                    eval_imgs = images.mean(dim=1, keepdim=True)
                else:
                    eval_imgs = images
                final_grids = evaluator.eval_images(eval_imgs.to(device))["discrete"]

                vid_dir = os.path.join(task_dir, "videos")
                os.makedirs(vid_dir, exist_ok=True)
                n_vid = min(eval_video_samples, n_eval)
                for vi in range(n_vid):
                    grid_vid_path = os.path.join(
                        vid_dir,
                        f"step_{step:07d}_sudoku_{level_name}_sample{vi}_grid.{eval_save_format}")
                    _render_sudoku_grid_video(
                        history, final_grids,
                        sample_idx=vi, save_path=grid_vid_path,
                        max_frames=args.eval_num_steps, fps=8,
                        grid_hw=args.grid_hw, hint_mask=known_mask,
                        title=f"[{split_tag}] {level_name}",
                        hint_gt_digits=(x_gt + 1))
                accelerator.print(
                    f"[{log_prefix}/sudoku_image] step={step} split={split_tag} "
                    f"[{level_name}] saved {n_vid} grid videos → {vid_dir}/")

            # MNIST classify → grid → rule check
            _log_sudoku_image_metrics(
                images, evaluator, level_name, step, n_eval,
                task_dir, accelerator, args, prefix=log_prefix,
                gt_digits=x_gt, hint_mask=known_mask)


def _log_sudoku_image_metrics(
    images, evaluator, tag, step, n_samples,
    save_dir, accelerator, args, prefix="eval",
    gt_digits=None, hint_mask=None,
):
    """Shared helper: MNIST classify → rule check → log.

    Optional diagnostics (when gt_digits + hint_mask provided):
      * hint_acc:    fraction of hint cells correctly read back by classifier
                     (isolates encoder→decoder→MNIST roundtrip error)
      * gen_acc:     fraction of NON-hint (model-generated) cells matching GT
      * hint_err_frac: 1 - hint_acc, the "wrong-rendering rate" requested
    """
    device = images.device if images.is_cuda else accelerator.device
    if images.shape[1] == 3:
        eval_images = images.mean(dim=1, keepdim=True)
    else:
        eval_images = images

    eval_result = evaluator.eval_images(eval_images.to(device))
    rule_acc = eval_result["accuracy"].item()
    n_valid = eval_result["labels"].sum().item()

    extra_txt = ""
    extra_log = {}
    if gt_digits is not None and hint_mask is not None:
        # eval_result["discrete"]: (B, 9, 9) classified digits in [1..9]
        pred = eval_result["discrete"].to(device).long().view(n_samples, -1)
        gt = gt_digits.to(device).long().view(n_samples, -1)
        # gt_digits stored as [0..8]; classifier returns [1..9]. Normalize.
        if gt.min().item() == 0:
            gt = gt + 1
        hm = hint_mask.to(device).bool().view(n_samples, -1)

        match = (pred == gt)  # (B, 81)
        n_hint = hm.sum().clamp(min=1).item()
        n_gen = (~hm).sum().clamp(min=1).item()
        hint_acc = (match & hm).sum().item() / n_hint
        gen_acc = (match & (~hm)).sum().item() / n_gen
        hint_err = 1.0 - hint_acc

        extra_txt = (f" hint_acc={hint_acc:.4f} "
                     f"hint_err={hint_err:.4f} "
                     f"gen_acc={gen_acc:.4f} "
                     f"(hints={int(n_hint)}, gen={int(n_gen)})")
        extra_log = {
            f"{prefix}/{tag}/hint_acc": hint_acc,
            f"{prefix}/{tag}/hint_err_frac": hint_err,
            f"{prefix}/{tag}/gen_acc": gen_acc,
        }

    accelerator.print(
        f"[{prefix}/sudoku_image] step={step} [{tag}] "
        f"rule_acc={rule_acc:.4f} ({int(n_valid)}/{n_samples}){extra_txt}")

    txt_path = os.path.join(save_dir, f"step_{step:07d}_sudoku_{tag}.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step} tag={tag} rule_acc={rule_acc:.6f}{extra_txt}\n")
        for i in range(min(8, n_samples)):
            f.write(f"sample {i}: {eval_result['discrete'][i].tolist()}\n")

    if args.log_with:
        accelerator.log({
            f"{prefix}/{tag}/rule_acc": rule_acc,
            **extra_log,
        }, step=step)


def _eval_imagenet(model, step, args, accelerator, save_dir,
                   pretrained_model, discretizer, level_sizes):
    """ImageNet eval: generate tokens for random classes, decode to images.
    Runs multiple samplers and saves each."""
    device = accelerator.device
    if level_sizes is None:
        seq_len = args.seq_len
    elif hasattr(pretrained_model, 'num_slots'):
        seq_len = sum(level_sizes)
    else:
        seq_len = sum(s * s for s in level_sizes)
    n_samples = min(args.eval_num_samples, 64)

    class_labels = torch.randint(0, args.num_classes, (n_samples,), device=device)

    if args.model_type == "ar":
        sampler_configs = [{"name": "ar", "sampler": "ar"}]
    else:
        sampler_configs = [
            {"name": "ddpm_cache", "sampler": "ddpm_cache",
             "tokens_per_step": 0},
            {"name": "confidence_top1", "sampler": "confidence",
             "tokens_per_step": 1},
            {"name": "confidence_cosine", "sampler": "confidence",
             "tokens_per_step": 0},
        ]

    for sc in sampler_configs:
        tag = sc["name"]
        sample_kwargs = dict(
            batch_size=n_samples, seq_len=seq_len,
            num_steps=args.eval_num_steps, device=device,
            sampler=sc["sampler"], noise_removal=True,
            class_labels=class_labels,
            tokens_per_step=sc.get("tokens_per_step", 0),
        )
        if args.model_type == "ar":
            sample_kwargs.update(temperature=args.ar_temperature,
                                 top_k=args.ar_top_k, top_p=args.ar_top_p)

        tokens = model.sample(**sample_kwargs)

        txt_path = os.path.join(save_dir, f"step_{step:07d}_imagenet_{tag}.txt")
        with open(txt_path, "w") as f:
            f.write(f"step={step} sampler={tag} n_samples={n_samples}\n")
            f.write(f"class_labels={class_labels.tolist()}\n")
            f.write(f"token_range=[{tokens.min().item()}, {tokens.max().item()}]\n")

        if pretrained_model is not None and discretizer is not None \
                and level_sizes is not None:
            try:
                images = decode_tokens_to_images(
                    tokens, level_sizes, pretrained_model, discretizer, device,
                    num_steps=args.decode_num_steps,
                    batch_size=min(16, n_samples),
                )
                img_path = os.path.join(
                    save_dir, f"step_{step:07d}_imagenet_{tag}.png")
                save_sample_grid(images, img_path, nrow=8)
                accelerator.print(
                    f"[eval/imagenet] step={step} [{tag}] saved {n_samples} samples")
            except Exception as e:
                accelerator.print(f"[eval/imagenet] [{tag}] decode failed: {e}")
        else:
            accelerator.print(
                f"[eval/imagenet] [{tag}] tokens generated (no decode)")


def _select_eval_indices(val_dataset, n_per_split):
    """Select split-balanced, unique-image eval indices (deterministic).

    Args:
        val_dataset: dataset with get_condition(i) method
        n_per_split: number of samples **per split** (e.g. 30)

    Returns:
        selected: list of dataset indices
        splits_for_selected: list of split names (same length as selected)
    """
    import random as _rng
    from collections import OrderedDict
    eval_rng = _rng.Random(42)

    image_to_indices = OrderedDict()
    for i in range(len(val_dataset)):
        cond = val_dataset.get_condition(i)
        if isinstance(cond, dict):
            fn = cond.get("image_filename", f"img_{i}")
            split = cond.get("split", "unknown")
        else:
            fn = f"img_{i}"
            split = "unknown"
        key = f"{split}/{fn}"
        if key not in image_to_indices:
            image_to_indices[key] = []
        image_to_indices[key].append(i)

    split_images = {}
    for key, indices in image_to_indices.items():
        split = key.split("/")[0]
        if split not in split_images:
            split_images[split] = []
        split_images[split].append((key, indices))

    splits = sorted(split_images.keys())
    selected = []
    splits_for_selected = []
    for split in splits:
        pool = split_images[split]
        eval_rng.shuffle(pool)
        for key, indices in pool[:n_per_split]:
            selected.append(eval_rng.choice(indices))
            splits_for_selected.append(split)

    return selected, splits_for_selected


def _eval_clevr(model, step, args, accelerator, save_dir,
                pretrained_model, discretizer, level_sizes,
                clevr_cond_encoder, val_dataset,
                clevr_detector=None, clevr_classifier=None,
                log_prefix="eval"):
    """CLEVR eval — fully distributed: all ranks sample, decode, AND eval.

    All ranks: sample tokens → decode to images → run condition eval on own shard.
    Results are reduced (all_reduce) across ranks.
    Rank 0: gathers images for grid saving, logs/saves per-split results.
    """
    device = accelerator.device
    if level_sizes is None:
        seq_len = args.seq_len
    elif hasattr(pretrained_model, 'num_slots'):
        seq_len = sum(level_sizes)
    else:
        seq_len = sum(s * s for s in level_sizes)
    n_per_split = args.eval_num_samples  # per-split sample count
    n_samples = n_per_split  # fallback; overwritten when val_dataset exists

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    # ── Select eval samples (same on all ranks — deterministic) ──
    cond_tokens = None
    cond_jsons = []
    sample_splits = []  # split label per sample
    if clevr_cond_encoder is not None and val_dataset is not None:
        selected_indices, sample_splits = _select_eval_indices(
            val_dataset, n_per_split)
        n_samples = len(selected_indices)

        cond_encoder = clevr_cond_encoder
        if hasattr(cond_encoder, 'module'):
            cond_encoder = cond_encoder.module

        is_pretrained_te = isinstance(
            cond_encoder, PretrainedTextConditionEncoder)

        if is_pretrained_te:
            texts = []
            for idx in selected_indices:
                sample = val_dataset[idx]
                # Dataset yields "cond_text" when return_raw_text=True.
                texts.append(sample.get(
                    "cond_text",
                    _extract_raw_text(val_dataset.get_condition(idx))))
                cond_jsons.append(val_dataset.get_condition(idx))
            text_tokens = cond_encoder.tokenize(texts, device)
            cond_tokens, _cond_mask = cond_encoder(text_tokens)
        else:
            cond_id_list = []
            for idx in selected_indices:
                sample = val_dataset[idx]
                cond_id_list.append(sample["cond_token_ids"])
                cond_jsons.append(val_dataset.get_condition(idx))
            cond_ids = torch.stack(cond_id_list).to(device)
            cond_tokens = cond_encoder(cond_ids)

    # Save condition meta (rank 0 only)
    if is_main:
        meta_path = os.path.join(save_dir, f"step_{step:07d}_clevr_meta.json")
        with open(meta_path, "w") as f:
            json.dump({"step": step, "n_samples": n_samples,
                       "splits": sample_splits,
                       "conditions": cond_jsons}, f, indent=2)

    # Build split info for this rank's shard
    split_names = sorted(set(sample_splits)) if sample_splits else []

    # ── Shard samples across ranks (round-robin) ──
    my_indices = list(range(rank, n_samples, world_size))
    my_n = len(my_indices)
    max_n = (n_samples + world_size - 1) // world_size  # for padding in gather

    my_cond_tokens = None
    my_uncond_cond_tokens = None  # pretrained-TE path: encoder.null_embed-based
    if cond_tokens is not None and my_n > 0:
        my_cond_tokens = cond_tokens[my_indices]
        if (clevr_cond_encoder is not None
                and isinstance(cond_encoder, PretrainedTextConditionEncoder)):
            my_uncond_cond_tokens = cond_encoder.get_null_cond(
                my_cond_tokens.shape[0], my_cond_tokens.shape[1],
                my_cond_tokens.device).to(my_cond_tokens.dtype)

    # Per-shard condition jsons and split labels
    my_cond_jsons = [cond_jsons[i] for i in my_indices] if cond_jsons else []
    my_splits = [sample_splits[i] for i in my_indices] if sample_splits else []

    # ── Define sampler configs ──
    use_diffusion_head = getattr(args, 'use_diffusion_head', False)
    if args.model_type == "ar":
        sampler_configs = [{"name": "ar", "sampler": "ar"}]
    elif getattr(args, "factorized_head", False):
        # Factorized head does not produce per-token confidence scores,
        # so only use ddpm_cache sampler for eval.
        sampler_configs = [
            {"name": "ddpm_cache", "sampler": "ddpm_cache",
             "tokens_per_step": 0},
        ]
    else:
        sampler_configs = [
            {"name": "ddpm_cache", "sampler": "ddpm_cache",
             "tokens_per_step": 0},
            {"name": "confidence_top1", "sampler": "confidence",
             "tokens_per_step": 1},
            {"name": "confidence_cosine", "sampler": "confidence",
             "tokens_per_step": 0},
        ]

    # Continuous mode: discretizer not needed (tokens are continuous vectors)
    if use_diffusion_head:
        can_decode = (pretrained_model is not None and level_sizes is not None)
    else:
        can_decode = (pretrained_model is not None and discretizer is not None
                      and level_sizes is not None)
    # Resolve feature dim for continuous sampling
    _dh_feat_dim = 16
    if use_diffusion_head:
        _inner = accelerator.unwrap_model(model)
        if getattr(_inner, 'diff_head', None) is not None:
            _dh_feat_dim = _inner.diff_head.in_channels
    img_size = args.image_size
    has_eval_models = (clevr_detector is not None
                       and clevr_classifier is not None
                       and len(cond_jsons) > 0)

    # Synchronize has_eval_models across all ranks to avoid collective mismatch.
    # If any rank failed to load eval models, skip cond eval on ALL ranks.
    if world_size > 1:
        _flag = torch.tensor([1.0 if has_eval_models else 0.0], device=device)
        torch.distributed.all_reduce(_flag, op=torch.distributed.ReduceOp.MIN)
        has_eval_models = (_flag.item() > 0.5)
        del _flag

    import time as _time

    # Barrier: ensure all ranks enter eval together (training may desync slightly)
    if world_size > 1:
        torch.distributed.barrier()

    # Free training-related GPU cache before eval sampling
    torch.cuda.empty_cache()

    eval_sample_bs = getattr(args, "eval_sample_batch_size", 8)

    for sc in sampler_configs:
        tag = sc["name"]
        _t0 = _time.time()

        # ── 1. Each rank samples its shard of tokens (chunked) ──
        if my_n > 0:
            token_chunks = []
            for chunk_start in range(0, my_n, eval_sample_bs):
                chunk_end = min(chunk_start + eval_sample_bs, my_n)
                chunk_bs = chunk_end - chunk_start
                chunk_cond = (my_cond_tokens[chunk_start:chunk_end]
                              if my_cond_tokens is not None else None)
                chunk_uncond = (my_uncond_cond_tokens[chunk_start:chunk_end]
                                if my_uncond_cond_tokens is not None else None)
                if use_diffusion_head:
                    sample_kwargs = dict(
                        batch_size=chunk_bs, seq_len=seq_len,
                        feat_dim=_dh_feat_dim,
                        num_steps=args.eval_num_steps, device=device,
                        sampler=sc["sampler"],
                        tokens_per_step=sc.get("tokens_per_step", 0),
                        cond_tokens=chunk_cond,
                        uncond_cond_tokens=chunk_uncond,
                        temperature=getattr(args, 'diff_head_temperature', 1.0),
                        cfg=getattr(args, 'diff_head_cfg', 1.0),
                        cfg_schedule=getattr(args, 'cfg_schedule', 'constant'),
                        cfg_mode=getattr(args, 'cfg_mode', 'head'),
                        null_class_index=None,  # CLEVR has no class labels
                    )
                    token_chunks.append(model.sample_continuous(**sample_kwargs))
                else:
                    sample_kwargs = dict(
                        batch_size=chunk_bs, seq_len=seq_len,
                        num_steps=args.eval_num_steps, device=device,
                        sampler=sc["sampler"], noise_removal=True,
                        cond_tokens=chunk_cond,
                        tokens_per_step=sc.get("tokens_per_step", 0),
                    )
                    if args.model_type == "ar":
                        sample_kwargs.update(temperature=args.ar_temperature,
                                             top_k=args.ar_top_k, top_p=args.ar_top_p)
                    token_chunks.append(model.sample(**sample_kwargs))
            my_tokens = torch.cat(token_chunks, dim=0)
        else:
            if use_diffusion_head:
                my_tokens = torch.zeros(0, seq_len, _dh_feat_dim, device=device)
            else:
                my_tokens = torch.zeros(0, seq_len, dtype=torch.long, device=device)

        torch.cuda.synchronize()
        _t1 = _time.time()
        print(f"[eval/debug] rank={rank} [{tag}] sample: {_t1-_t0:.1f}s "
              f"(my_n={my_n})", flush=True)

        # ── 2. Each rank decodes its shard to images ──
        my_images = None
        if can_decode and my_n > 0:
            print(f"[eval/debug] rank={rank} [{tag}] decode start "
                  f"(GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / "
                  f"{torch.cuda.max_memory_allocated(device)/1e9:.2f}GB peak)", flush=True)
            try:
                _decode_bs = min(
                    getattr(args, "eval_decode_batch_size", 4), my_n)
                if use_diffusion_head:
                    my_images = decode_continuous_tokens_to_images(
                        my_tokens, level_sizes, pretrained_model, device,
                        num_steps=args.decode_num_steps,
                        batch_size=_decode_bs,
                    )
                else:
                    my_images = decode_tokens_to_images(
                        my_tokens, level_sizes, pretrained_model,
                        discretizer, device,
                        num_steps=args.decode_num_steps,
                        batch_size=_decode_bs,
                    )  # (my_n, 3, H, W) in [0, 1]
            except Exception as e:
                import traceback
                print(f"[eval/debug] rank={rank} [{tag}] decode FAILED: {e}\n"
                      f"{traceback.format_exc()}", flush=True)
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        _t2 = _time.time()
        print(f"[eval/debug] rank={rank} [{tag}] decode: {_t2-_t1:.1f}s", flush=True)

        # ── 3. Each rank runs condition eval on its own shard ──
        my_eval_per_split = {}  # split -> {count_correct, entity_found, ...}
        can_eval_cond = has_eval_models
        if can_eval_cond and my_images is not None and my_n > 0:
            try:
                from eval_clevr_condition import (
                    eval_clevr_conditions, clevr_text_to_condition_json)
                # For text conditions, parse captions to structured JSON for eval
                eval_cond_jsons = my_cond_jsons
                if args.clevr_cond_type == "text":
                    def _to_eval_json(c):
                        if isinstance(c, str):
                            return clevr_text_to_condition_json(c)
                        if isinstance(c, dict) and "text" in c:
                            return clevr_text_to_condition_json(c["text"])
                        return c
                    eval_cond_jsons = [_to_eval_json(c) for c in my_cond_jsons]
                # `decode_*_tokens_to_images` returns images in [-1, 1], but
                # `eval_clevr_conditions` (and the underlying detector +
                # classifier) expect them in [0, 1]. Rescale before eval —
                # otherwise the detector input is normalized to roughly
                # [-3, 1] and presence/relation scores collapse, even though
                # the saved grid PNG looks fine (the grid-saver does its own
                # 0.5+0.5 rescale).
                my_images_eval = (my_images * 0.5 + 0.5).clamp(0, 1)
                # Run eval on this rank's shard
                my_eval = eval_clevr_conditions(
                    my_images_eval, eval_cond_jsons,
                    clevr_detector, clevr_classifier,
                )
                # Split results by difficulty
                per_sample = my_eval["per_sample"]
                for local_i, sp in enumerate(my_splits):
                    if sp not in my_eval_per_split:
                        my_eval_per_split[sp] = {
                            "n": 0, "count_correct": 0,
                            "entity_found": 0, "entity_total": 0,
                            "rel_correct": 0, "rel_total": 0,
                        }
                    d = my_eval_per_split[sp]
                    r = per_sample[local_i]
                    d["n"] += 1
                    if r["count_correct"]:
                        d["count_correct"] += 1
                    d["entity_found"] += r["entity_found"]
                    d["entity_total"] += r["entity_total"]
                    d["rel_correct"] += r["rel_correct"]
                    d["rel_total"] += r["rel_total"]
            except Exception as e:
                accelerator.print(
                    f"[{log_prefix}/clevr] [{tag}] rank {rank} cond eval failed: {e}")

        torch.cuda.synchronize()
        _t3 = _time.time()
        print(f"[eval/debug] rank={rank} [{tag}] cond_eval: {_t3-_t2:.1f}s", flush=True)
        print(f"[eval/debug] rank={rank} [{tag}] total_before_allreduce: "
              f"{_t3-_t0:.1f}s", flush=True)

        # ── 4. All-reduce eval counts across ranks ──
        if can_eval_cond:
            # Build a flat tensor of counts: for each split, pack the scalars
            # Order: n, count_correct, entity_found, entity_total,
            #        rel_correct, rel_total  = 6 values per split
            n_vals = 6
            counts = torch.zeros(len(split_names) * n_vals, device=device)
            for si, sp in enumerate(split_names):
                if sp in my_eval_per_split:
                    d = my_eval_per_split[sp]
                    off = si * n_vals
                    counts[off + 0] = d["n"]
                    counts[off + 1] = d["count_correct"]
                    counts[off + 2] = d["entity_found"]
                    counts[off + 3] = d["entity_total"]
                    counts[off + 4] = d["rel_correct"]
                    counts[off + 5] = d["rel_total"]

            # all_reduce to sum across ranks
            print(f"[eval/debug] rank={rank} [{tag}] entering all_reduce", flush=True)
            torch.distributed.all_reduce(counts,
                                         op=torch.distributed.ReduceOp.SUM)
            print(f"[eval/debug] rank={rank} [{tag}] all_reduce done", flush=True)

        # ── 5. Gather decoded images to rank 0 for grid saving ──
        if can_decode and my_images is not None:
            padded_img = torch.zeros(max_n, 3, img_size, img_size,
                                     dtype=my_images.dtype, device=device)
            padded_img[:my_n] = my_images
            del my_images
        else:
            padded_img = torch.zeros(max_n, 3, img_size, img_size,
                                     device=device)

        print(f"[eval/debug] rank={rank} [{tag}] entering gather "
              f"(GPU mem: {torch.cuda.memory_allocated(device)/1e9:.2f}GB)", flush=True)
        all_padded_img = accelerator.gather(padded_img)
        print(f"[eval/debug] rank={rank} [{tag}] gather done", flush=True)
        del padded_img
        torch.cuda.empty_cache()
        # shape: (world_size * max_n, 3, H, W) — move to CPU to save GPU mem
        all_padded_img = all_padded_img.cpu()

        if is_main:
            # Reconstruct original order from round-robin sharding (on CPU)
            all_images = torch.zeros(n_samples, 3, img_size, img_size)
            for r in range(world_size):
                r_indices = list(range(r, n_samples, world_size))
                r_n = len(r_indices)
                r_imgs = all_padded_img[r * max_n : r * max_n + r_n]
                for local_i, global_i in enumerate(r_indices):
                    all_images[global_i] = r_imgs[local_i]

            # Save image grid — interleave (GT, gen) pairs when the source
            # image dataset is available; otherwise fall back to gen-only.
            img_path = os.path.join(
                save_dir, f"step_{step:07d}_clevr_{tag}.png")
            src_ds = getattr(val_dataset, "source_image_ds", None)
            if src_ds is not None:
                gt_imgs = torch.stack([
                    src_ds[selected_indices[i]]["image"]
                    for i in range(n_samples)
                ])  # (N, 3, H, W) in [-1, 1]
                # (gt, gen) pairs → (2N, 3, H, W); rows alternate gt/gen.
                paired = torch.stack([gt_imgs, all_images], dim=1).view(
                    -1, 3, img_size, img_size)
                save_sample_grid(paired, img_path, nrow=8)
            else:
                save_sample_grid(all_images, img_path, nrow=8)
            accelerator.print(
                f"[{log_prefix}/clevr] step={step} [{tag}] saved → {img_path}")

            # ── 6. Log/save per-split + overall results (rank 0 only) ──
            if can_eval_cond:
                overall = {"n": 0, "count_correct": 0,
                           "entity_found": 0, "entity_total": 0,
                           "rel_correct": 0, "rel_total": 0}
                all_split_results = {}

                for si, sp in enumerate(split_names):
                    off = si * n_vals
                    n_sp = int(counts[off + 0].item())
                    if n_sp == 0:
                        continue
                    cc = int(counts[off + 1].item())
                    ef = int(counts[off + 2].item())
                    et = int(counts[off + 3].item())
                    rc = int(counts[off + 4].item())
                    rt = int(counts[off + 5].item())

                    sp_result = _build_eval_result(n_sp, cc, ef, et, rc, rt)
                    all_split_results[sp] = sp_result

                    accelerator.print(
                        f"[{log_prefix}/clevr] step={step} [{tag}] "
                        f"split={sp} ({n_sp} samples):")
                    accelerator.print(_format_split_result(sp_result))

                    # Accumulate overall
                    overall["n"] += n_sp
                    overall["count_correct"] += cc
                    overall["entity_found"] += ef
                    overall["entity_total"] += et
                    overall["rel_correct"] += rc
                    overall["rel_total"] += rt

                # Overall
                if overall["n"] > 0:
                    overall_result = _build_eval_result(
                        overall["n"], overall["count_correct"],
                        overall["entity_found"], overall["entity_total"],
                        overall["rel_correct"], overall["rel_total"])
                    accelerator.print(
                        f"[{log_prefix}/clevr] step={step} [{tag}] "
                        f"overall ({overall['n']} samples):")
                    accelerator.print(_format_split_result(overall_result))

                    # Save JSON with per-split breakdown
                    eval_save = {
                        "step": step, "sampler": tag,
                        "overall": overall_result,
                        "per_split": all_split_results,
                    }
                    eval_path = os.path.join(
                        save_dir,
                        f"step_{step:07d}_clevr_{tag}_cond_eval.json")
                    with open(eval_path, "w") as f:
                        json.dump(eval_save, f, indent=2)

                    # Tensorboard logging
                    if args.log_with:
                        log_dict = {}
                        for sp, sr in all_split_results.items():
                            pfx = f"{log_prefix}/{tag}/{sp}"
                            log_dict[f"{pfx}/count_acc"] = sr["count_accuracy"]
                            log_dict[f"{pfx}/entity_presence_acc"] = sr["entity_presence_accuracy"]
                            log_dict[f"{pfx}/rel_acc"] = sr["rel_accuracy"]
                        # Overall
                        pfx = f"{log_prefix}/{tag}/overall"
                        log_dict[f"{pfx}/count_acc"] = overall_result["count_accuracy"]
                        log_dict[f"{pfx}/entity_presence_acc"] = overall_result["entity_presence_accuracy"]
                        log_dict[f"{pfx}/rel_acc"] = overall_result["rel_accuracy"]
                        accelerator.log(log_dict, step=step)

        # Sync before next sampler
        accelerator.wait_for_everyone()


def _build_eval_result(n, count_correct, entity_found, entity_total,
                       rel_correct, rel_total):
    """Build a result dict from raw counts (same schema as eval_clevr_conditions)."""
    return {
        "n_samples": n,
        "count_accuracy": count_correct / n * 100 if n > 0 else 0.0,
        "entity_presence_accuracy": (entity_found / entity_total * 100
                                     if entity_total > 0 else 0.0),
        "rel_accuracy": (rel_correct / rel_total * 100
                         if rel_total > 0 else 0.0),
        "count_correct": count_correct,
        "entity_found": entity_found,
        "entity_total": entity_total,
        "rel_correct": rel_correct,
        "rel_total": rel_total,
    }


def _format_split_result(r):
    """Format a single split/overall result dict as a readable string."""
    lines = []
    lines.append(f"    Count acc:           {r['count_accuracy']:.1f}% "
                 f"({r['count_correct']}/{r['n_samples']})")
    lines.append(f"    Entity presence acc: {r['entity_presence_accuracy']:.1f}% "
                 f"({r['entity_found']}/{r['entity_total']})")
    lines.append(f"    Relation acc:        {r['rel_accuracy']:.1f}% "
                 f"({r['rel_correct']}/{r['rel_total']})")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────
#  Args
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-dataset Discrete Diffusion Training (MDLM-style)")

    # ── dataset type ──
    p.add_argument("--dataset_type", type=str, required=True,
                   choices=["imagenet", "clevr", "sudoku"])
    p.add_argument("--output_dir", type=str, default="./outputs/discrete_diffusion_v2")
    p.add_argument("--seed", type=int, default=42)

    # ── ImageNet ──
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Root dir for ImageNet or CLEVR images")
    p.add_argument("--num_classes", type=int, default=1000,
                   help="Number of classes for ImageNet class-cond.")
    p.add_argument("--image_size", type=int, default=256)

    # ── CLEVR ──
    p.add_argument("--clevr_condition_dir", type=str, default=None,
                   help="Directory with CLEVR condition JSONs (train).")
    p.add_argument("--clevr_val_image_root", type=str, default=None,
                   help="Val image root (default: inferred from dataset_root).")
    p.add_argument("--clevr_val_condition_dir", type=str, default=None,
                   help="Val condition dir (default: same as clevr_condition_dir).")
    p.add_argument("--clevr_train_splits", type=str, nargs="+",
                   default=["easy", "medium", "hard"])
    p.add_argument("--clevr_val_splits", type=str, nargs="+",
                   default=["easy"])
    p.add_argument("--clevr_cond_type", type=str, default="json",
                   choices=["json", "text"],
                   help="Condition format: 'json' (structured) or 'text' (captions).")

    # ── Pretrained text encoder (CLIP / T5), only for cond_type=text ──
    p.add_argument("--use_pretrained_text_encoder", action="store_true",
                   default=False,
                   help="Replace the word-level CLEVR text encoder with a "
                        "pretrained HF encoder (CLIP / T5). Only used when "
                        "--clevr_cond_type=text.")
    p.add_argument("--pretrained_text_model_name", type=str,
                   default="openai/clip-vit-base-patch32",
                   help="HF model name for the pretrained text encoder "
                        "(e.g. 'openai/clip-vit-base-patch32', "
                        "'google-t5/t5-base').")
    p.add_argument("--pretrained_text_max_length", type=int, default=77,
                   help="Tokenizer max_length for the pretrained text encoder "
                        "(CLIP hard-caps at 77).")
    p.add_argument("--freeze_text_encoder", action="store_true",
                   dest="freeze_text_encoder", default=True,
                   help="Freeze the pretrained text encoder weights "
                        "(only the projection is trained). Default: frozen.")
    p.add_argument("--unfreeze_text_encoder", action="store_false",
                   dest="freeze_text_encoder",
                   help="Fine-tune the pretrained text encoder (separate LR "
                        "via --text_encoder_lr).")
    p.add_argument("--text_encoder_lr", type=float, default=None,
                   help="LR for the unfrozen pretrained text encoder "
                        "(default: 0.1 * --lr).")

    # ── Sudoku ──
    p.add_argument("--sudoku_config", type=str, default=None)
    p.add_argument("--grid_only", action="store_true", default=False)
    p.add_argument("--grid_hw", type=int, default=9)
    p.add_argument("--grid_vocab_size", type=int, default=9)
    p.add_argument("--classifier_pth", type=str, default=None,
                   help="Path to MNIST classifier for sudoku image eval")
    p.add_argument("--mask_ratio_min", type=float, default=0.0,
                   help="Min condition mask ratio for sudoku image training "
                        "(0.0 = all digits visible)")
    p.add_argument("--mask_ratio_max", type=float, default=1.0,
                   help="Max condition mask ratio for sudoku image training "
                        "(1.0 = all digits masked)")

    # ── Pretrained model (for token extraction) ──
    p.add_argument("--pretrained_output_dir", type=str, default=None,
                   help="Output dir of pretrained continuous diffusion model.")
    p.add_argument("--token_cache_dir", type=str, default=None,
                   help="Dir to cache extracted token IDs.")

    # ── DiT backbone ──
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_blocks", type=int, default=12)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--model_dropout", type=float, default=0.1)
    p.add_argument("--pos_emb_type", type=str, default="multires",
                   choices=["1d", "2d", "sudoku", "multires"])

    # ── TokenBridge-style factorized AR head ──
    p.add_argument("--factorized_head", action="store_true", default=False,
                   help="Use factorized per-dim AR head instead of flat softmax. "
                        "FSQ levels are auto-detected from pretrained encoder.")
    p.add_argument("--ar_head_dim", type=int, default=256)
    p.add_argument("--ar_head_layers", type=int, default=2)

    # ── Diffusion head (MAR-style continuous tokens) ──
    p.add_argument("--use_diffusion_head", action="store_true", default=False,
                   help="Use diffusion head on continuous tokens (MAR-style). "
                        "Encoder features are NOT discretized; instead, the "
                        "backbone outputs hidden states that condition a small "
                        "diffusion MLP to predict continuous token vectors.")
    p.add_argument("--diff_head_depth", type=int, default=6,
                   help="Number of ResBlocks in diffusion head MLP.")
    p.add_argument("--diff_head_width", type=int, default=1024,
                   help="Hidden dim of diffusion head MLP.")
    p.add_argument("--diff_head_num_sampling_steps", type=int, default=100,
                   help="Euler ODE steps for diffusion head sampling.")
    p.add_argument("--diff_head_batch_mul", type=int, default=4,
                   help="Batch multiplier for diffusion head loss (variance reduction).")
    p.add_argument("--diff_head_cond_drop_prob", type=float, default=0.1,
                   help="Prob of replacing backbone-z with learned null embedding "
                        "during diff-head training (MAR/semanticist-style CFG).")
    p.add_argument("--diff_head_cfg", type=float, default=1.0,
                   help="Classifier-free guidance scale used when sampling from "
                        "the diff head at eval time. 1.0 = no guidance.")
    p.add_argument("--diff_head_temperature", type=float, default=1.0,
                   help="Temperature for diffusion head sampling.")

    # ── Sudoku conditioning mode ──
    p.add_argument("--use_sudoku_prefix", action="store_true", default=False,
                   help="Use digit-grid prefix (SudokuConditionEncoder). "
                        "Default: no prefix. Condition is injected at "
                        "inference by fixing known cells' image tokens.")
    p.add_argument("--use_sudoku_cell_cond", action="store_true", default=False,
                   help="Per-cell digit conditioning via additive residual "
                        "(AdaLN-like). Start from fully-masked at inference; "
                        "hint digits only influence generation via "
                        "SudokuDigitCellEncoder. Independent from inpainting.")
    p.add_argument("--semi_autoregressive", action="store_true", default=False,
        help="Enable coarse-to-fine semi-autoregressive masked diffusion. "
             "Per batch item, sample a target resolution level k: coarser "
             "levels (1x1,...) are fully revealed as conditioning, target "
             "level is diffused with the MDLM schedule, finer levels are "
             "fully masked. Loss is computed only on target-level positions.")
    p.add_argument("--time_conditioning", action="store_true", default=False,
                   help="Feed sigma/t into AdaLN (MDLM flag). Default False "
                        "(MDLM paper: absorbing diffusion doesn't need it).")

    # ── model type ──
    p.add_argument("--model_type", type=str, default="diffusion",
                   choices=["diffusion", "ar"],
                   help="'diffusion' = MDLM mask diffusion, 'ar' = autoregressive")
    p.add_argument("--ar_temperature", type=float, default=1.0)
    p.add_argument("--ar_top_k", type=int, default=0)
    p.add_argument("--ar_top_p", type=float, default=1.0)

    # ── noise schedule (diffusion only) ──
    p.add_argument("--noise_type", type=str, default="loglinear",
                   choices=["loglinear", "cosine"])
    p.add_argument("--noise_eps", type=float, default=1e-3)
    p.add_argument("--init_embed_from_fsq", action="store_true", default=False,
                   help="Initialize DiT token_emb from FSQ codebook vectors "
                        "(projected to hidden_size). Only for image token modes.")

    # ── diffusion ──
    p.add_argument("--antithetic_sampling", action="store_true", default=True)
    p.add_argument("--importance_sampling", action="store_true", default=False)
    p.add_argument("--change_of_variables", action="store_true", default=False)
    p.add_argument("--sampling_eps", type=float, default=1e-3)

    # ── conditioning ──
    p.add_argument("--uncond_drop_prob", type=float, default=0.1,
                   help="Probability of dropping condition (CFG training).")
    p.add_argument("--cfg_mode", type=str, default="head",
                   choices=["head", "backbone"],
                   help="'head' (default, sudoku): drop z at diffusion head input "
                        "(diff_head_cond_drop_prob). 'backbone' (MAR/semanticist "
                        "style, clevr/imagenet): swap backbone-input condition with "
                        "learned null_cond, backbone forwards twice for CFG. "
                        "When 'backbone', diff_head_cond_drop_prob is forced to 0.")
    p.add_argument("--cfg_schedule", type=str, default="constant",
                   choices=["constant", "linear"],
                   help="Per-step CFG schedule during sampling (semanticist): "
                        "'constant' applies --diff_head_cfg at every step; "
                        "'linear' ramps from 1.0 at step 0 to --diff_head_cfg "
                        "at the final step.")

    # ── training ──
    p.add_argument("--max_train_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--lr_schedule", type=str, default="constant",
                   choices=["constant", "cosine"],
                   help="LR schedule after warmup. 'cosine' decays to min_lr_ratio * lr.")
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="Min LR as a fraction of base LR for cosine schedule.")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # ── logging / eval ──
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--eval_num_samples", type=int, default=64)
    p.add_argument("--eval_num_steps", type=int, default=128)
    p.add_argument("--decode_num_steps", type=int, default=50,
                   help="DDIM steps for decoding tokens to images during eval.")
    p.add_argument("--eval_sample_batch_size", type=int, default=8,
                   help="Per-rank batch size for token sampling at eval. "
                        "Bump (e.g. 32-64) when GPU util is low during sweeps.")
    p.add_argument("--eval_decode_batch_size", type=int, default=4,
                   help="Per-rank batch size for decoding sampled tokens to "
                        "images at eval. Currently hardcoded to min(4,my_n) — "
                        "this CLI lets the sweep bump it (e.g. 16-32).")
    p.add_argument("--sampler", type=str, default="ddpm_cache",
                   choices=["ddpm", "ddpm_cache", "confidence"])
    p.add_argument("--tokens_per_step", type=int, default=0)
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="Guidance scale for pretrained decoder only "
                        "(flow matching ODE). Not used for discrete diffusion.")
    p.add_argument("--eval_video_samples", type=int, default=4,
                   help="Number of denoising process videos to save (0=disable).")
    p.add_argument("--eval_save_format", type=str, default="mp4",
                   choices=["gif", "mp4"],
                   help="Format for denoising process videos (GIF or MP4).")

    # ── FID (ImageNet) ──
    p.add_argument("--fid_every", type=int, default=0,
                   help="Compute FID every N steps (0 = disabled). Slow!")
    p.add_argument("--fid_num_samples", type=int, default=5000,
                   help="Number of samples for FID computation.")
    p.add_argument("--fid_real_dir", type=str, default=None,
                   help="Dir with real images for FID. If None, extracts "
                        "from val set on first run.")

    # ── resume ──
    p.add_argument("--resume_dir", type=str, default=None)
    p.add_argument("--eval_only", action="store_true",
                   help="Load checkpoint, run one evaluate_and_save, and exit.")

    # ── accelerate ──
    p.add_argument("--mixed_precision", type=str, default="no",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--log_with", type=str, default=None)

    return p.parse_args()


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # In eval-only mode we must NOT clobber the original training
    # run_config.json — sweepers re-run this script with overrides and we
    # want the canonical config to stay readable.
    if not getattr(args, "eval_only", False):
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump({"cmd": " ".join(sys.argv), "args": vars(args)},
                      f, indent=2, sort_keys=True)

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=project_config,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        rng_types=[],
    )
    accelerator.print("=" * 60)
    accelerator.print(f"Discrete Diffusion V2 — dataset={args.dataset_type}")
    accelerator.print("=" * 60)

    if accelerator.is_main_process and args.log_with is not None:
        sanitized = {}
        for k, v in vars(args).items():
            if v is None: sanitized[k] = "None"
            elif isinstance(v, (list, tuple)): sanitized[k] = str(v)
            else: sanitized[k] = v
        accelerator.init_trackers(
            project_name=f"discrete_diff_{args.dataset_type}",
            config=sanitized,
        )

    # ─────────────────────────────────────────────────────────
    #  Load pretrained encoder (if needed)
    # ─────────────────────────────────────────────────────────
    encoder = None
    discretizer = None
    level_sizes = None
    pretrained_model = None
    data_vocab_size = 0
    seq_len = 0
    cont_feat_dim = 0  # set when use_diffusion_head=True

    needs_pretrained = (
        args.dataset_type in ("imagenet", "clevr")
        or (args.dataset_type == "sudoku" and not args.grid_only)
    )
    if needs_pretrained:
        assert args.pretrained_output_dir is not None, \
            f"--pretrained_output_dir required for dataset_type={args.dataset_type}"

        accelerator.print(f"[pretrained] Loading from {args.pretrained_output_dir}")
        pretrained_model, encoder, discretizer, level_sizes, data_vocab_size, pretrained_cfg = \
            load_pretrained_model(args.pretrained_output_dir,
                                  device=accelerator.device)
        is_baseline_1d = (pretrained_cfg.get("backbone") == "baseline_1d")
        if is_baseline_1d:
            seq_len = sum(level_sizes)          # flat 1D slots
        else:
            seq_len = sum(s * s for s in level_sizes)  # spatial grid per level
        accelerator.print(
            f"[pretrained] level_sizes={level_sizes}, "
            f"seq_len={seq_len}, vocab_size={data_vocab_size}")

        # After caching is done below, we move pretrained model to CPU
        # to free GPU memory; it's loaded back to GPU only during eval

    # ─────────────────────────────────────────────────────────
    #  Semi-autoregressive position→level map
    # ─────────────────────────────────────────────────────────
    semi_ar_level_idx = None
    semi_ar_n_levels = None
    if args.semi_autoregressive:
        assert level_sizes is not None and not is_baseline_1d, (
            "--semi_autoregressive requires multi-resolution spatial tokens "
            "(level_sizes from a hierarchical encoder, not baseline_1d).")
        # Token order in x0 follows extract_tokens: finest spatial size first.
        # level_sizes is already sorted finest→coarsest by the encoder
        # (see multi_res_encoder.HierarchicalMultiResEncoder). So level 0 =
        # finest, n_levels-1 = coarsest (e.g. 1x1).
        sorted_sizes = sorted(level_sizes, reverse=True)
        idx = []
        for li, s in enumerate(sorted_sizes):
            idx.extend([li] * (s * s))
        semi_ar_level_idx = torch.tensor(
            idx, dtype=torch.long, device=accelerator.device)
        semi_ar_n_levels = len(sorted_sizes)
        accelerator.print(
            f"[semi_ar] enabled — n_levels={semi_ar_n_levels}, "
            f"level_sizes(finest→coarsest)={sorted_sizes}")

    # ─────────────────────────────────────────────────────────
    #  Dataset
    # ─────────────────────────────────────────────────────────
    clevr_cond_encoder = None
    sudoku_cell_cond_encoder = None
    clevr_detector = None
    clevr_classifier = None
    train_dataset = None
    val_dataset = None

    if args.dataset_type == "imagenet":
        accelerator.print(f"[data] Loading ImageNet from {args.dataset_root}")
        train_img_ds = ImageFolderDataset(args.dataset_root, split="train",
                                          image_size=args.image_size)
        val_img_ds = ImageFolderDataset(args.dataset_root, split="val",
                                        image_size=args.image_size)

        # Cache tokens (all GPUs participate in parallel)
        cache_dir = args.token_cache_dir or os.path.join(args.output_dir, "token_cache")
        train_cache_path = os.path.join(cache_dir, "imagenet_train_tok.pt")
        val_cache_path = os.path.join(cache_dir, "imagenet_val_tok.pt")

        train_tok = cache_all_tokens(
            encoder, discretizer, train_img_ds, accelerator.device,
            batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
        val_tok = cache_all_tokens(
            encoder, discretizer, val_img_ds, accelerator.device,
            batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
        accelerator.wait_for_everyone()

        # Extract class labels
        train_labels = torch.tensor([train_img_ds.ds.targets[i]
                                     for i in range(len(train_img_ds))], dtype=torch.long)
        val_labels = torch.tensor([val_img_ds.ds.targets[i]
                                   for i in range(len(val_img_ds))], dtype=torch.long)

        train_dataset = CachedTokenDataset(train_tok, labels=train_labels)
        val_dataset = CachedTokenDataset(val_tok, labels=val_labels)

    elif args.dataset_type == "clevr":
        # Resolve val paths
        val_image_root = args.clevr_val_image_root or args.dataset_root
        val_cond_dir = args.clevr_val_condition_dir or args.clevr_condition_dir
        cond_type = args.clevr_cond_type

        accelerator.print(f"[data] Loading CLEVR train from {args.dataset_root}")
        accelerator.print(f"[data]   train cond: {args.clevr_condition_dir} (type={cond_type})")
        accelerator.print(f"[data]   val images: {val_image_root}")
        accelerator.print(f"[data]   val cond:   {val_cond_dir}")

        # Load datasets WITH augmented conditions
        # (one entry per (image, condition) pair — may have multiple per image)
        train_img_ds = CLEVRImageDataset(
            args.dataset_root, condition_dir=args.clevr_condition_dir,
            image_size=args.image_size, splits=args.clevr_train_splits,
            cond_type=cond_type)
        val_img_ds = CLEVRImageDataset(
            val_image_root, condition_dir=val_cond_dir,
            image_size=args.image_size, splits=args.clevr_val_splits,
            cond_type=cond_type)

        # Token caching: use image-only datasets (no condition augmentation)
        # to avoid encoding the same image multiple times
        train_img_only = CLEVRImageDataset(
            args.dataset_root, condition_dir=None,
            image_size=args.image_size, splits=args.clevr_train_splits)
        val_img_only = CLEVRImageDataset(
            val_image_root, condition_dir=None,
            image_size=args.image_size, splits=args.clevr_val_splits)

        accelerator.print(
            f"[data] Train: {len(train_img_ds)} (image,cond) pairs "
            f"from {len(train_img_only)} unique images")
        accelerator.print(
            f"[data] Val:   {len(val_img_ds)} (image,cond) pairs "
            f"from {len(val_img_only)} unique images")

        cache_dir = args.token_cache_dir or os.path.join(args.output_dir, "token_cache")

        if args.use_diffusion_head:
            # ── Continuous mode: extract feature vectors (no discretizer) ──
            train_cache_path = os.path.join(cache_dir, "clevr_train_cont.pt")
            val_cache_path = os.path.join(cache_dir, "clevr_val_cont.pt")
            train_feats_unique = cache_all_continuous_tokens(
                encoder, train_img_only, accelerator.device,
                batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
            val_feats_unique = cache_all_continuous_tokens(
                encoder, val_img_only, accelerator.device,
                batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
        else:
            train_cache_path = os.path.join(cache_dir, "clevr_train_tok.pt")
            val_cache_path = os.path.join(cache_dir, "clevr_val_tok.pt")
            train_tok_unique = cache_all_tokens(
                encoder, discretizer, train_img_only, accelerator.device,
                batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
            val_tok_unique = cache_all_tokens(
                encoder, discretizer, val_img_only, accelerator.device,
                batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
        accelerator.wait_for_everyone()

        # Build path → token index mapping for unique images
        train_path_to_idx = {p: i for i, p in enumerate(train_img_only.image_paths)}
        val_path_to_idx = {p: i for i, p in enumerate(val_img_only.image_paths)}

        # Expand tokens to match augmented dataset (index into unique tokens)
        train_tok_indices = [train_path_to_idx[p] for p in train_img_ds.image_paths]
        val_tok_indices = [val_path_to_idx[p] for p in val_img_ds.image_paths]

        if args.use_diffusion_head:
            train_feats = train_feats_unique[train_tok_indices]
            val_feats = val_feats_unique[val_tok_indices]
        else:
            train_tok = train_tok_unique[train_tok_indices]
            val_tok = val_tok_unique[val_tok_indices]

        # Collect CLEVR conditions
        train_conditions = [train_img_ds.get_condition(i) for i in range(len(train_img_ds))]
        val_conditions = [val_img_ds.get_condition(i) for i in range(len(val_img_ds))]

        # Select tokenizer and encoder based on condition type
        if cond_type == "text":
            if args.use_pretrained_text_encoder:
                clevr_cond_encoder = PretrainedTextConditionEncoder(
                    model_name=args.pretrained_text_model_name,
                    hidden_size=args.hidden_size,
                    max_length=args.pretrained_text_max_length,
                    freeze=args.freeze_text_encoder,
                )
                cond_tokenizer_fn = None  # pretrained path tokenizes per-batch
                accelerator.print(
                    f"[clevr] TEXT condition encoder: PRETRAINED "
                    f"({clevr_cond_encoder._kind}, "
                    f"{args.pretrained_text_model_name}, "
                    f"max_len={args.pretrained_text_max_length}, "
                    f"freeze={args.freeze_text_encoder}) → "
                    f"hidden_size={args.hidden_size}")
            else:
                cond_tokenizer_fn = clevr_text_to_token_ids
                clevr_cond_encoder = CLEVRTextConditionEncoder(args.hidden_size)
                accelerator.print(
                    f"[clevr] TEXT condition encoder: word-vocab, "
                    f"vocab_size={CLEVR_TEXT_VOCAB_SIZE}, "
                    f"hidden_size={args.hidden_size}")
        else:
            cond_tokenizer_fn = clevr_json_to_token_ids
            clevr_cond_encoder = CLEVRConditionEncoder(args.hidden_size)
            accelerator.print(f"[clevr] JSON condition encoder built, hidden_size={args.hidden_size}")

        if args.use_diffusion_head:
            cont_feat_dim = train_feats.shape[-1]
            seq_len = train_feats.shape[1]
            accelerator.print(
                f"[clevr-diffhead] Continuous tokens: seq_len={seq_len}, "
                f"feat_dim={cont_feat_dim}, shape={train_feats.shape}")
            train_dataset = CachedContinuousTokenDataset(
                train_feats, clevr_conditions=train_conditions,
                cond_tokenizer_fn=cond_tokenizer_fn,
                return_raw_text=args.use_pretrained_text_encoder
                                and cond_type == "text",
                source_image_ds=train_img_ds)
            val_dataset = CachedContinuousTokenDataset(
                val_feats, clevr_conditions=val_conditions,
                cond_tokenizer_fn=cond_tokenizer_fn,
                return_raw_text=args.use_pretrained_text_encoder
                                and cond_type == "text",
                source_image_ds=val_img_ds)
        else:
            train_dataset = CachedTokenDataset(
                train_tok, clevr_conditions=train_conditions,
                cond_tokenizer_fn=cond_tokenizer_fn,
                return_raw_text=args.use_pretrained_text_encoder
                                and cond_type == "text",
                source_image_ds=train_img_ds)
            val_dataset = CachedTokenDataset(
                val_tok, clevr_conditions=val_conditions,
                cond_tokenizer_fn=cond_tokenizer_fn,
                return_raw_text=args.use_pretrained_text_encoder
                                and cond_type == "text",
                source_image_ds=val_img_ds)

        # Load CLEVR eval models (detector + classifier) on ALL ranks
        # so that condition eval can run distributed (each rank evals its shard).
        try:
            from eval_clevr_condition import load_eval_models
            clevr_detector, clevr_classifier = load_eval_models(
                device=accelerator.device)
            accelerator.print("[clevr] loaded detector + classifier for condition eval")
        except Exception as e:
            accelerator.print(f"[clevr] WARNING: could not load eval models: {e}")
            accelerator.print("[clevr] condition alignment eval will be skipped")

    elif args.dataset_type == "sudoku":
        from omegaconf import OmegaConf
        from SRM.datasets import get_dataset
        from SRM.type_extensions import ConditioningCfg

        cfg = OmegaConf.load(args.sudoku_config)
        train_raw = get_dataset(cfg.SRM_dataset_cfg, cfg.SRM_conditioning_cfg, "train")
        val_raw = get_dataset(cfg.SRM_dataset_cfg, cfg.SRM_conditioning_cfg, "val")

        if args.grid_only:
            train_dataset = GridOnlyDataset(train_raw)
            val_dataset = GridOnlyDataset(val_raw)
            data_vocab_size = args.grid_vocab_size
            seq_len = args.grid_hw * args.grid_hw
        else:
            # Image-based: use pretrained encoder to tokenize sudoku images
            assert encoder is not None, "Pretrained encoder required for sudoku image mode"

            # Extract grids for eval
            train_grids = train_raw.sudoku_grids  # (N, 9, 9)
            val_grids = val_raw.sudoku_grids

            cache_dir = args.token_cache_dir or os.path.join(args.output_dir, "token_cache")

            if args.use_diffusion_head:
                # ── Continuous mode: extract feature vectors (no discretizer) ──
                train_cache_path = os.path.join(cache_dir, "sudoku_train_cont.pt")
                val_cache_path = os.path.join(cache_dir, "sudoku_val_cont.pt")

                train_feats = cache_all_continuous_tokens(
                    encoder, train_raw, accelerator.device,
                    batch_size=64, cache_path=train_cache_path,
                    accelerator=accelerator)
                val_feats = cache_all_continuous_tokens(
                    encoder, val_raw, accelerator.device,
                    batch_size=64, cache_path=val_cache_path,
                    accelerator=accelerator)
                accelerator.wait_for_everyone()

                if not isinstance(train_grids, torch.Tensor):
                    train_grids = torch.tensor(train_grids)
                if not isinstance(val_grids, torch.Tensor):
                    val_grids = torch.tensor(val_grids)

                cont_feat_dim = train_feats.shape[-1]
                seq_len = train_feats.shape[1]
                accelerator.print(
                    f"[sudoku-diffhead] Continuous tokens: seq_len={seq_len}, "
                    f"feat_dim={cont_feat_dim}, shape={train_feats.shape}")

                train_dataset = CachedContinuousTokenDataset(
                    train_feats, sudoku_digit_grids=train_grids)
                val_dataset = CachedContinuousTokenDataset(
                    val_feats, sudoku_digit_grids=val_grids)
                train_dataset.sudoku_grids = train_grids
                val_dataset.sudoku_grids = val_grids

            else:
                # ── Discrete mode: extract token IDs via discretizer ──
                train_cache_path = os.path.join(cache_dir, "sudoku_train_tok.pt")
                val_cache_path = os.path.join(cache_dir, "sudoku_val_tok.pt")

                train_tok = cache_all_tokens(
                    encoder, discretizer, train_raw, accelerator.device,
                    batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
                val_tok = cache_all_tokens(
                    encoder, discretizer, val_raw, accelerator.device,
                    batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
                accelerator.wait_for_everyone()

                if not isinstance(train_grids, torch.Tensor):
                    train_grids = torch.tensor(train_grids)
                if not isinstance(val_grids, torch.Tensor):
                    val_grids = torch.tensor(val_grids)

                train_dataset = CachedTokenDataset(
                    train_tok, sudoku_digit_grids=train_grids)
                val_dataset = CachedTokenDataset(
                    val_tok, sudoku_digit_grids=val_grids)
                train_dataset.sudoku_grids = train_grids
                val_dataset.sudoku_grids = val_grids

            # Sudoku condition encoder (digit grid → prefix embeddings)
            # Reuse clevr_cond_encoder variable (only one dataset type active)
            if args.use_sudoku_prefix:
                clevr_cond_encoder = SudokuConditionEncoder(args.hidden_size)
                accelerator.print(
                    f"[sudoku] Condition encoder: digit grid (81) → "
                    f"prefix embeddings, mask_ratio=[{args.mask_ratio_min}, "
                    f"{args.mask_ratio_max}]")
            else:
                clevr_cond_encoder = None
                accelerator.print(
                    "[sudoku] No prefix condition encoder. "
                    "Training = standard MDLM on 81 image tokens. "
                    "Inference conditioning via fixed known image tokens.")

            # Per-cell digit condition encoder (AdaLN-like residual)
            sudoku_cell_cond_encoder = None
            if args.use_sudoku_cell_cond:
                sudoku_cell_cond_encoder = SudokuDigitCellEncoder(
                    args.hidden_size, grid_len=SUDOKU_GRID_LEN)
                accelerator.print(
                    f"[sudoku] Per-cell digit condition: enabled. "
                    f"mask_ratio=[{args.mask_ratio_min}, {args.mask_ratio_max}]. "
                    f"Inference: fully-masked start + hint digits as cell cond.")

    accelerator.print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Move pretrained model to CPU to free GPU for training
    if pretrained_model is not None:
        pretrained_model.cpu()
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # ─────────────────────────────────────────────────────────
    #  Build model (diffusion or AR)
    # ─────────────────────────────────────────────────────────
    is_ar = (args.model_type == "ar")

    # AR doesn't need [MASK]; diffusion adds +1 for mask token
    backbone_vocab_size = data_vocab_size if is_ar else (data_vocab_size + 1)

    # Auto-detect FSQ levels from pretrained encoder
    fsq_levels_for_head = None
    if args.factorized_head:
        if discretizer is not None and hasattr(discretizer, 'fsq'):
            fsq_levels_for_head = discretizer.fsq._levels.tolist()
        elif discretizer is not None and hasattr(discretizer, 'fsq_dim'):
            # FSQDiscretizer wraps FSQ
            fsq_levels_for_head = discretizer.fsq._levels.tolist()
        if fsq_levels_for_head is not None:
            accelerator.print(f"[factorized] FSQ levels from encoder: {fsq_levels_for_head}")
        else:
            raise ValueError("--factorized_head requires a pretrained model with FSQ discretizer")

    # Determine pos_emb_type and extra kwargs
    dit_kwargs = dict(
        vocab_size=backbone_vocab_size,
        seq_len=seq_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        cond_dim=args.cond_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.model_dropout,
        causal=is_ar,
        factorized_head=args.factorized_head,
        fsq_levels=fsq_levels_for_head,
        ar_head_dim=args.ar_head_dim,
        ar_head_layers=args.ar_head_layers,
    )

    # ── Continuous mode with diffusion head ──
    if args.use_diffusion_head:
        dit_kwargs["continuous_mode"] = True
        dit_kwargs["continuous_dim"] = cont_feat_dim
        # Override: no discrete vocab needed
        dit_kwargs["vocab_size"] = 1  # placeholder, unused in continuous mode
        dit_kwargs["factorized_head"] = False
        dit_kwargs["fsq_levels"] = None

    if args.dataset_type == "sudoku" and args.grid_only:
        dit_kwargs["pos_emb_type"] = args.pos_emb_type if args.pos_emb_type != "multires" else "2d"
        dit_kwargs["sudoku_hw"] = args.grid_hw
    elif args.dataset_type == "sudoku" and not args.grid_only:
        dit_kwargs["pos_emb_type"] = args.pos_emb_type
        dit_kwargs["level_sizes"] = level_sizes
    elif args.dataset_type == "imagenet":
        dit_kwargs["pos_emb_type"] = "multires"
        dit_kwargs["level_sizes"] = level_sizes
        dit_kwargs["num_classes"] = args.num_classes
    elif args.dataset_type == "clevr":
        dit_kwargs["pos_emb_type"] = args.pos_emb_type
        dit_kwargs["level_sizes"] = level_sizes
        # Both AR and diffusion use prefix concat for CLEVR conditioning
        # (no cross-attention needed)

    backbone = DIT(**dit_kwargs)

    # ── optionally initialize token_emb from FSQ codebook ──
    if args.init_embed_from_fsq and discretizer is not None and not args.use_diffusion_head:
        all_ids = torch.arange(data_vocab_size, device="cpu")
        with torch.no_grad():
            codebook_vecs = discretizer.decode(all_ids.unsqueeze(0)).squeeze(0)  # (V, slot_dim)
        slot_dim = codebook_vecs.shape[-1]
        hidden = args.hidden_size
        proj = nn.Linear(slot_dim, hidden, bias=False).to(codebook_vecs.device)
        nn.init.xavier_uniform_(proj.weight)
        with torch.no_grad():
            projected = proj(codebook_vecs)  # (V, hidden)
        backbone.token_emb.weight.data[:data_vocab_size] = projected
        accelerator.print(
            f"[init] token_emb[:{data_vocab_size}] from FSQ codebook "
            f"({slot_dim}→{hidden}), mask token [{data_vocab_size}] left random")

    # ── Build diffusion head (MAR-style) ──
    diff_head = None
    if args.use_diffusion_head:
        from diffloss import DiffLoss
        # In backbone-CFG mode, force head-level drop off (only one level drops).
        _head_drop = 0.0 if args.cfg_mode == "backbone" else args.diff_head_cond_drop_prob
        if args.cfg_mode == "backbone" and args.diff_head_cond_drop_prob > 0:
            accelerator.print(
                f"[cfg_mode=backbone] overriding diff_head_cond_drop_prob "
                f"{args.diff_head_cond_drop_prob} → 0.0 (CFG handled at backbone)")
        diff_head = DiffLoss(
            target_channels=cont_feat_dim,
            z_channels=args.hidden_size,
            depth=args.diff_head_depth,
            width=args.diff_head_width,
            num_sampling_steps=args.diff_head_num_sampling_steps,
            cond_drop_prob=_head_drop,
        )
        dh_total, dh_train = count_params(diff_head)
        accelerator.print(
            f"[diffusion-head] DiffLoss: target_dim={cont_feat_dim}, "
            f"z_dim={args.hidden_size}, depth={args.diff_head_depth}, "
            f"width={args.diff_head_width}, "
            f"sampling_steps={args.diff_head_num_sampling_steps}, "
            f"batch_mul={args.diff_head_batch_mul}, "
            f"params={format_n(dh_total)}")

    if is_ar:
        diffusion = AutoregressiveModel(
            backbone=backbone,
            vocab_size=data_vocab_size,
        )
        accelerator.print(
            f"[model] AR: hidden={args.hidden_size}, heads={args.n_heads}, "
            f"blocks={args.n_blocks}, seq_len={seq_len}, "
            f"data_vocab={data_vocab_size}")
    else:
        diffusion = DiscreteDiffusion(
            backbone=backbone,
            vocab_size=data_vocab_size,
            noise_type=args.noise_type,
            noise_eps=args.noise_eps,
            antithetic_sampling=args.antithetic_sampling,
            importance_sampling=args.importance_sampling,
            change_of_variables=args.change_of_variables,
            sampling_eps=args.sampling_eps,
            diff_head=diff_head,
            diffusion_batch_mul=args.diff_head_batch_mul,
            time_conditioning=args.time_conditioning,
        )
        if args.semi_autoregressive:
            # Activates coarse-to-fine training loss + coarse-to-fine
            # sampling inside sample_continuous.
            diffusion.set_semi_ar(semi_ar_level_idx, semi_ar_n_levels)
            accelerator.print(
                f"[model] semi-AR enabled on diffusion "
                f"(n_levels={semi_ar_n_levels})")
        if args.use_diffusion_head:
            accelerator.print(
                f"[model] Diffusion + DiffHead (continuous): "
                f"hidden={args.hidden_size}, heads={args.n_heads}, "
                f"blocks={args.n_blocks}, seq_len={seq_len}, "
                f"feat_dim={cont_feat_dim}")
        else:
            accelerator.print(
                f"[model] Diffusion: hidden={args.hidden_size}, heads={args.n_heads}, "
                f"blocks={args.n_blocks}, seq_len={seq_len}, "
                f"data_vocab={data_vocab_size}, mask_idx={diffusion.mask_index}")

    total_p, train_p = count_params(diffusion)
    accelerator.print(
        f"[model] Total: {format_n(total_p)} (trainable {format_n(train_p)})")

    # Optimizer
    all_params = list(diffusion.parameters())
    te_params = []  # unfrozen pretrained text-encoder params (separate LR)
    if clevr_cond_encoder is not None:
        ce_total, ce_train = count_params(clevr_cond_encoder)
        accelerator.print(f"[model] CLEVR cond encoder: {format_n(ce_total)} params")

        is_pretrained_te = isinstance(
            clevr_cond_encoder, PretrainedTextConditionEncoder)
        if is_pretrained_te and not clevr_cond_encoder.freeze:
            # Separate LR for the pretrained backbone; projection stays at main LR.
            for name, param in clevr_cond_encoder.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("encoder."):
                    te_params.append(param)
                else:
                    all_params.append(param)
        else:
            all_params += [p for p in clevr_cond_encoder.parameters()
                           if p.requires_grad]

    if sudoku_cell_cond_encoder is not None:
        all_params += list(sudoku_cell_cond_encoder.parameters())
        cc_total, _ = count_params(sudoku_cell_cond_encoder)
        accelerator.print(f"[model] Sudoku cell cond encoder: {format_n(cc_total)} params")

    param_groups = [{"params": all_params, "lr": args.lr,
                     "weight_decay": args.weight_decay}]
    if te_params:
        te_lr = (args.text_encoder_lr if args.text_encoder_lr is not None
                 else args.lr * 0.1)
        param_groups.append({"params": te_params, "lr": te_lr,
                             "weight_decay": args.weight_decay})
        n_te = sum(p.numel() for p in te_params)
        accelerator.print(
            f"[model] Pretrained text encoder unfrozen: "
            f"{format_n(n_te)} params, LR={te_lr:.2e} "
            f"(main LR={args.lr:.2e})")

    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=args.weight_decay)
    # Preserve behavior of `all_params` for grad clipping below.
    all_params = all_params + te_params
    lr_scheduler = get_lr_scheduler(
        optimizer, args.warmup_steps, args.max_train_steps,
        schedule=args.lr_schedule, min_lr_ratio=args.min_lr_ratio,
    )

    # Prepare
    prepare_list = [diffusion, optimizer, train_loader, lr_scheduler]
    insert_idx = 1
    if clevr_cond_encoder is not None:
        prepare_list.insert(insert_idx, clevr_cond_encoder)
        insert_idx += 1
    if sudoku_cell_cond_encoder is not None:
        prepare_list.insert(insert_idx, sudoku_cell_cond_encoder)
    prepared = accelerator.prepare(*prepare_list)

    # Unpack in the same order we inserted
    prepared = list(prepared)
    diffusion = prepared.pop(0)
    if clevr_cond_encoder is not None:
        clevr_cond_encoder = prepared.pop(0)
    if sudoku_cell_cond_encoder is not None:
        sudoku_cell_cond_encoder = prepared.pop(0)
    optimizer, train_loader, lr_scheduler = prepared

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(diffusion).parameters(),
                  decay=args.ema_decay)
        ema.to(accelerator.device)

    # Resume — auto-detect latest checkpoint in output_dir/ckpt/
    global_step = 0
    resume_dir = args.resume_dir
    if not resume_dir:
        ckpt_root = os.path.join(args.output_dir, "ckpt")
        if os.path.isdir(ckpt_root):
            ckpt_dirs = sorted(
                [d for d in os.listdir(ckpt_root)
                 if d.startswith("step") and os.path.isdir(
                     os.path.join(ckpt_root, d))],
                key=lambda x: parse_step_from_dir(x))
            if ckpt_dirs:
                resume_dir = os.path.join(ckpt_root, ckpt_dirs[-1])
    if resume_dir and os.path.isdir(resume_dir):
        try:
            accelerator.load_state(resume_dir)
        except (ValueError, RuntimeError) as _e:
            # Eval-only sweepers may build the model with a slightly
            # different head/optimizer config than what the ckpt was saved
            # under (e.g. freeze_text_encoder mismatch). For eval we only
            # care about the model + EMA weights — fall back to loading
            # those manually and skip optimizer / scheduler state.
            if not getattr(args, "eval_only", False):
                raise
            accelerator.print(
                f"[resume] full load_state failed ({_e}); "
                f"falling back to model+EMA-only load for eval_only.")
            from safetensors.torch import load_file as _load_safetensors
            unwrapped = accelerator.unwrap_model(diffusion)
            for fn in ("model.safetensors", "model_1.safetensors"):
                p = os.path.join(resume_dir, fn)
                if os.path.isfile(p):
                    sd = _load_safetensors(p)
                    # Keys that match
                    msd = unwrapped.state_dict()
                    matched = {k: v for k, v in sd.items() if k in msd}
                    unwrapped.load_state_dict(matched, strict=False)
                    accelerator.print(
                        f"[resume] loaded {len(matched)}/{len(sd)} keys "
                        f"from {fn}")
        global_step = parse_step_from_dir(resume_dir)
        # Restore EMA shadow weights
        ema_path = os.path.join(resume_dir, "ema.pt")
        if ema is not None and os.path.isfile(ema_path):
            shadow = torch.load(ema_path, map_location=accelerator.device)
            ema.shadow = shadow
            accelerator.print(f"[resume] EMA restored from {ema_path}")
        accelerator.print(f"[resume] {resume_dir}  →  step={global_step}")

    if args.eval_only:
        accelerator.print(f"[eval_only] running one evaluate_and_save at step={global_step}")
        # Skip the train-set eval branch in eval-only mode — sweepers only
        # care about the val-set scores.
        evaluate_and_save(
            diffusion, global_step, args, accelerator, ema,
            pretrained_model=pretrained_model,
            discretizer=discretizer,
            level_sizes=level_sizes,
            clevr_cond_encoder=clevr_cond_encoder,
            val_dataset=val_dataset,
            clevr_detector=clevr_detector,
            clevr_classifier=clevr_classifier,
            train_dataset=None,
            sudoku_cell_cond_encoder=sudoku_cell_cond_encoder,
        )
        accelerator.print("[eval_only] done")
        return

    # ════════════════════════════════════════════════════════
    #  Training loop
    # ════════════════════════════════════════════════════════
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress = tqdm(range(global_step, args.max_train_steps),
                    disable=not accelerator.is_local_main_process)
    progress.set_description("Steps")

    accelerator.print(f"\n[train] Starting for {args.max_train_steps} steps ...\n")
    diffusion.train()
    if clevr_cond_encoder is not None:
        clevr_cond_encoder.train()
    if sudoku_cell_cond_encoder is not None:
        sudoku_cell_cond_encoder.train()
    running_loss = 0.0

    _epoch = 0
    _batches_in_epoch = len(train_loader)
    _usable = (_batches_in_epoch // args.grad_accum_steps) * args.grad_accum_steps
    _skipped = _batches_in_epoch - _usable
    if _skipped > 0:
        accelerator.print(
            f"[train] batches/epoch={_batches_in_epoch}, grad_accum={args.grad_accum_steps} "
            f"→ usable={_usable}, SKIPPING last {_skipped} batches per epoch")

    while global_step < args.max_train_steps:
        # Ensure DistributedSampler uses a different shuffle per epoch
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(_epoch)
        _epoch += 1

        _batch_idx = 0
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            # Skip incomplete accumulation batches at end-of-epoch to keep
            # all ranks synchronised on the same global_step.
            _batch_idx += 1
            if _batch_idx > _usable:
                continue

            # ── Prepare batch ──
            class_labels = None
            cond_tokens = None
            cell_cond = None  # per-cell digit residual (AdaLN-style)
            cont_tokens = None  # for diffusion head mode

            if args.use_diffusion_head:
                cont_tokens = batch["cont_tokens"].to(accelerator.device)
                tokens = None  # not used in continuous mode
            elif args.dataset_type == "sudoku" and args.grid_only:
                grid = batch["grid"].to(accelerator.device).long()
                tokens = grid.view(grid.shape[0], -1) - 1  # (B, 81) in [0, 8]
            else:
                tokens = batch["tok_ids"].to(accelerator.device).long()

            if args.dataset_type == "imagenet":
                class_labels = batch["class_label"].to(accelerator.device).long()
                # CFG dropout: replace class with null_class
                if args.uncond_drop_prob > 0 and diffusion.training:
                    drop_mask = torch.rand(class_labels.shape[0],
                                           device=accelerator.device) < args.uncond_drop_prob
                    class_labels = torch.where(
                        drop_mask, args.num_classes, class_labels)  # num_classes = null index

            elif args.dataset_type == "sudoku" and not args.grid_only \
                    and args.use_sudoku_prefix:
                # Sudoku image mode (legacy): digit grid → condition prefix
                cond_ids = batch["cond_token_ids"].to(accelerator.device)  # (B, 81) in [0, 8]
                B_cond = cond_ids.shape[0]

                # Random masking of condition digits (like AR cond training)
                mask_ratio = torch.empty(B_cond, 1, device=accelerator.device).uniform_(
                    args.mask_ratio_min, args.mask_ratio_max)
                rand = torch.rand(B_cond, SUDOKU_GRID_LEN, device=accelerator.device)
                mask = rand < mask_ratio
                cond_ids = cond_ids.clone()
                cond_ids[mask] = SUDOKU_MASK_ID  # replace masked digits with [MASK]

                cond_enc = clevr_cond_encoder  # reused variable name
                if hasattr(cond_enc, 'module'):
                    cond_enc = cond_enc.module
                cond_tokens = cond_enc(cond_ids)

                # CFG dropout: zero out all cond tokens
                B_for_drop = cont_tokens.shape[0] if cont_tokens is not None else tokens.shape[0]
                if args.uncond_drop_prob > 0 and diffusion.training:
                    drop_mask = (torch.rand(B_for_drop,
                                            device=accelerator.device) < args.uncond_drop_prob)
                    cond_tokens = cond_tokens * (~drop_mask).float().unsqueeze(-1).unsqueeze(-1)

            # Independent path: per-cell sudoku digit conditioning.
            # Can combine with any of the above (prefix / grid_only / etc).
            if (args.dataset_type == "sudoku" and not args.grid_only
                    and args.use_sudoku_cell_cond
                    and sudoku_cell_cond_encoder is not None
                    and "cond_token_ids" in batch):
                digit_ids = batch["cond_token_ids"].to(accelerator.device).long()  # (B, 81) in [0..8]
                B_cc = digit_ids.shape[0]
                mr = torch.empty(B_cc, 1, device=accelerator.device).uniform_(
                    args.mask_ratio_min, args.mask_ratio_max)
                rand = torch.rand(B_cc, SUDOKU_GRID_LEN, device=accelerator.device)
                mask_cc = rand < mr
                digit_ids = digit_ids.clone()
                digit_ids[mask_cc] = SUDOKU_MASK_ID  # UNKNOWN

                cc_enc = sudoku_cell_cond_encoder
                if hasattr(cc_enc, 'module'):
                    cc_enc = cc_enc.module
                cell_cond = cc_enc(digit_ids)  # (B, 81, H)

                # CFG dropout: zero out cell cond for a fraction of batch
                if args.uncond_drop_prob > 0 and diffusion.training:
                    B_for_drop = cont_tokens.shape[0] if cont_tokens is not None else tokens.shape[0]
                    drop_mask = (torch.rand(B_for_drop,
                                            device=accelerator.device) < args.uncond_drop_prob)
                    cell_cond = cell_cond * (~drop_mask).float().unsqueeze(-1).unsqueeze(-1)

            if args.dataset_type == "clevr":
                cond_enc = clevr_cond_encoder
                if hasattr(cond_enc, 'module'):
                    cond_enc = cond_enc.module

                is_pretrained_te = isinstance(
                    cond_enc, PretrainedTextConditionEncoder)

                if is_pretrained_te:
                    # Naive-style: tokenize per-batch with dynamic padding,
                    # use tokenizer's real attention_mask, and draw uncond
                    # tokens from encoder.null_embed.
                    texts = batch["cond_text"]
                    text_tokens = cond_enc.tokenize(texts, accelerator.device)
                    cond_tokens, _cond_mask = cond_enc(text_tokens)
                else:
                    cond_ids = batch["cond_token_ids"].to(accelerator.device)
                    cond_tokens = cond_enc(cond_ids)

                # CFG dropout at backbone input
                B_for_drop = cont_tokens.shape[0] if cont_tokens is not None else tokens.shape[0]
                if args.uncond_drop_prob > 0 and diffusion.training:
                    drop_mask = (torch.rand(B_for_drop,
                                            device=accelerator.device) < args.uncond_drop_prob)
                    if is_pretrained_te:
                        # Naive recipe: swap with encoder's learnable null_embed.
                        null_expanded = cond_enc.get_null_cond(
                            B_for_drop, cond_tokens.shape[1],
                            accelerator.device).to(cond_tokens.dtype)
                        cond_tokens = torch.where(
                            drop_mask[:, None, None], null_expanded, cond_tokens)
                    elif args.cfg_mode == "backbone":
                        # MAR/semanticist style: swap with backbone null_cond.
                        bb = accelerator.unwrap_model(diffusion).backbone
                        null_tok = bb.null_cond_token.to(cond_tokens.dtype)
                        null_expanded = null_tok.expand(B_for_drop,
                                                        cond_tokens.shape[1],
                                                        -1)
                        cond_tokens = torch.where(
                            drop_mask[:, None, None], null_expanded, cond_tokens)
                    else:
                        # Legacy head-CFG mode: zero out
                        cond_tokens = cond_tokens * (~drop_mask).float().unsqueeze(-1).unsqueeze(-1)

            # ── Forward + backward ──
            with accelerator.accumulate(diffusion):
                if args.use_diffusion_head:
                    loss_out = accelerator.unwrap_model(diffusion).compute_loss_continuous(
                        cont_tokens, cond_tokens=cond_tokens,
                        class_labels=class_labels, cell_cond=cell_cond,
                        semi_ar_level_idx=semi_ar_level_idx,
                        semi_ar_n_levels=semi_ar_n_levels)
                else:
                    loss_out = accelerator.unwrap_model(diffusion).compute_loss(
                        tokens, cond_tokens=cond_tokens, class_labels=class_labels,
                        semi_ar_level_idx=semi_ar_level_idx,
                        semi_ar_n_levels=semi_ar_n_levels)
                loss = loss_out.loss

                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            if not accelerator.sync_gradients:
                continue

            if ema is not None:
                ema.update(accelerator.unwrap_model(diffusion).parameters())

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            # ── Logging ──
            if global_step % args.log_every == 0:
                avg = running_loss / (args.log_every * args.grad_accum_steps)
                lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"[step {global_step:>7d}] loss={avg:.4f} lr={lr:.2e}")
                if accelerator.is_main_process and args.log_with:
                    accelerator.log({"train/loss": avg, "train/lr": lr},
                                    step=global_step)
                running_loss = 0.0

            # ── Eval ──
            if global_step % args.eval_every == 0:
                # Verify all ranks agree on global_step before collective eval
                if accelerator.num_processes > 1:
                    _gs = torch.tensor([global_step], dtype=torch.long,
                                       device=accelerator.device)
                    _gs_all = accelerator.gather(_gs)
                    if not (_gs_all == global_step).all():
                        print(f"[FATAL] rank={accelerator.process_index} "
                              f"step desync detected: {_gs_all.tolist()}",
                              flush=True)
                evaluate_and_save(
                    diffusion, global_step, args, accelerator, ema,
                    pretrained_model=pretrained_model,
                    discretizer=discretizer,
                    level_sizes=level_sizes,
                    clevr_cond_encoder=clevr_cond_encoder,
                    val_dataset=val_dataset,
                    clevr_detector=clevr_detector,
                    clevr_classifier=clevr_classifier,
                    train_dataset=train_dataset,
                    sudoku_cell_cond_encoder=sudoku_cell_cond_encoder,
                )
                diffusion.train()
                if clevr_cond_encoder is not None:
                    clevr_cond_encoder.train()
                if sudoku_cell_cond_encoder is not None:
                    sudoku_cell_cond_encoder.train()

            # ── Save ──
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, "ckpt",
                                        f"step{global_step}")
                accelerator.save_state(ckpt_dir)
                if ema is not None and accelerator.is_main_process:
                    ema_path = os.path.join(ckpt_dir, "ema.pt")
                    torch.save([s.cpu() for s in ema.shadow], ema_path)
                accelerator.print(f"[save] → {ckpt_dir}")
                if accelerator.is_main_process:
                    meta = {"step": global_step, "args": vars(args)}
                    os.makedirs(ckpt_dir, exist_ok=True)
                    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2, sort_keys=True)

            # ── FID (ImageNet only) ──
            if (args.fid_every > 0
                    and global_step % args.fid_every == 0
                    and args.dataset_type == "imagenet"
                    and pretrained_model is not None):
                compute_fid(
                    accelerator.unwrap_model(diffusion),
                    args, accelerator,
                    pretrained_model, discretizer, level_sizes,
                    step=global_step,
                )
                diffusion.train()

        if global_step >= args.max_train_steps:
            break

    # Final save + eval
    ckpt_dir = os.path.join(args.output_dir, "ckpt", f"step{global_step}_final")
    accelerator.save_state(ckpt_dir)
    evaluate_and_save(
        diffusion, global_step, args, accelerator, ema,
        pretrained_model=pretrained_model,
        discretizer=discretizer,
        level_sizes=level_sizes,
        clevr_cond_encoder=clevr_cond_encoder,
        val_dataset=val_dataset,
        clevr_detector=clevr_detector,
        clevr_classifier=clevr_classifier,
        train_dataset=train_dataset,
        sudoku_cell_cond_encoder=sudoku_cell_cond_encoder,
    )
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
