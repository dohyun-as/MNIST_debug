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
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration, tqdm
from PIL import Image
from torchvision import transforms

from dit_model import DIT
from discrete_diffusion import DiscreteDiffusion
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
#  CLEVR condition encoder
# ────────────────────────────────────────────────────────────

CLEVR_COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
CLEVR_SHAPES = ["cube", "sphere", "cylinder"]
CLEVR_SIZES = ["small", "large"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_RELATIONS = ["left_of", "right_of", "in_front_of", "behind"]
MAX_CLEVR_ENTITIES = 12
MAX_CLEVR_RELATIONS = 30


def clevr_json_to_tensors(cond_json: dict):
    """Convert CLEVR condition JSON → fixed-size integer tensors.

    Returns:
        entity_attrs:  (max_entities, 4) int64  [color, shape, size, material]
        entity_mask:   (max_entities,) bool
        relation_data: (max_relations, 3) int64  [subj_idx, rel_idx, obj_idx]
        relation_mask: (max_relations,) bool
    """
    entities = cond_json.get("entities", [])
    relations = cond_json.get("relations", [])

    # Build name→index map
    name_to_idx = {}
    entity_attrs = torch.zeros(MAX_CLEVR_ENTITIES, 4, dtype=torch.long)
    entity_mask = torch.zeros(MAX_CLEVR_ENTITIES, dtype=torch.bool)

    for i, ent in enumerate(entities[:MAX_CLEVR_ENTITIES]):
        name_to_idx[ent["name"]] = i
        attrs = ent["attrs"]
        entity_attrs[i, 0] = CLEVR_COLORS.index(attrs["color"]) if attrs["color"] in CLEVR_COLORS else 0
        entity_attrs[i, 1] = CLEVR_SHAPES.index(attrs["shape"]) if attrs["shape"] in CLEVR_SHAPES else 0
        entity_attrs[i, 2] = CLEVR_SIZES.index(attrs["size"]) if attrs["size"] in CLEVR_SIZES else 0
        entity_attrs[i, 3] = CLEVR_MATERIALS.index(attrs["material"]) if attrs["material"] in CLEVR_MATERIALS else 0
        entity_mask[i] = True

    relation_data = torch.zeros(MAX_CLEVR_RELATIONS, 3, dtype=torch.long)
    relation_mask = torch.zeros(MAX_CLEVR_RELATIONS, dtype=torch.bool)

    for i, rel in enumerate(relations[:MAX_CLEVR_RELATIONS]):
        subj_idx = name_to_idx.get(rel["subj"], 0)
        obj_idx = name_to_idx.get(rel["obj"], 0)
        rel_idx = CLEVR_RELATIONS.index(rel["rel"]) if rel["rel"] in CLEVR_RELATIONS else 0
        relation_data[i] = torch.tensor([subj_idx, rel_idx, obj_idx])
        relation_mask[i] = True

    return entity_attrs, entity_mask, relation_data, relation_mask


class CLEVRConditionEncoder(nn.Module):
    """Encodes CLEVR entity/relation conditions → cross-attention tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.color_emb = nn.Embedding(len(CLEVR_COLORS), hidden_size)
        self.shape_emb = nn.Embedding(len(CLEVR_SHAPES), hidden_size)
        self.size_emb = nn.Embedding(len(CLEVR_SIZES), hidden_size)
        self.material_emb = nn.Embedding(len(CLEVR_MATERIALS), hidden_size)
        self.entity_proj = nn.Linear(hidden_size, hidden_size)

        self.entity_idx_emb = nn.Embedding(MAX_CLEVR_ENTITIES, hidden_size)
        self.rel_type_emb = nn.Embedding(len(CLEVR_RELATIONS), hidden_size)
        self.relation_proj = nn.Linear(hidden_size, hidden_size)

        # null token for padding
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, entity_attrs, entity_mask, relation_data, relation_mask):
        """
        Args:
            entity_attrs:  (B, max_E, 4)  int64
            entity_mask:   (B, max_E)     bool
            relation_data: (B, max_R, 3)  int64  [subj_idx, rel_idx, obj_idx]
            relation_mask: (B, max_R)     bool
        Returns:
            cond_tokens: (B, max_E + max_R, hidden_size)
        """
        B = entity_attrs.shape[0]

        # Entity tokens: sum of attribute embeddings
        e = (self.color_emb(entity_attrs[:, :, 0]) +
             self.shape_emb(entity_attrs[:, :, 1]) +
             self.size_emb(entity_attrs[:, :, 2]) +
             self.material_emb(entity_attrs[:, :, 3]))
        e = self.entity_proj(F.silu(e))  # (B, max_E, D)

        # Mask invalid entities
        e = e * entity_mask.unsqueeze(-1).float()

        # Relation tokens: sum of subject + relation_type + object embeddings
        r = (self.entity_idx_emb(relation_data[:, :, 0]) +
             self.rel_type_emb(relation_data[:, :, 1]) +
             self.entity_idx_emb(relation_data[:, :, 2]))
        r = self.relation_proj(F.silu(r))  # (B, max_R, D)
        r = r * relation_mask.unsqueeze(-1).float()

        # Concatenate entity + relation tokens
        cond_tokens = torch.cat([e, r], dim=1)  # (B, max_E + max_R, D)

        # Replace padded positions with null token
        full_mask = torch.cat([entity_mask, relation_mask], dim=1)  # (B, max_E+max_R)
        null = self.null_token.expand(B, cond_tokens.shape[1], -1)
        cond_tokens = torch.where(full_mask.unsqueeze(-1), cond_tokens, null)

        return cond_tokens


# ────────────────────────────────────────────────────────────
#  Dataset classes
# ────────────────────────────────────────────────────────────

class CachedTokenDataset(Dataset):
    """Returns cached tok_ids + optional labels/conditions."""
    def __init__(self, tok_ids, labels=None, clevr_conditions=None):
        self.tok_ids = tok_ids  # (N, seq_len) long
        self.labels = labels    # (N,) long or None
        self.clevr_conditions = clevr_conditions  # list of dicts or None

    def __len__(self):
        return len(self.tok_ids)

    def __getitem__(self, idx):
        item = {"tok_ids": self.tok_ids[idx].long()}
        if self.labels is not None:
            item["class_label"] = self.labels[idx]
        if self.clevr_conditions is not None:
            cond = self.clevr_conditions[idx]
            ea, em, rd, rm = clevr_json_to_tensors(cond)
            item["entity_attrs"] = ea
            item["entity_mask"] = em
            item["relation_data"] = rd
            item["relation_mask"] = rm
        return item


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

    # Common ViT encoder kwargs
    vit_kwargs = dict(
        encoder_type=cfg.get("encoder_type", "cnn"),
        vit_patch_size=cfg.get("vit_patch_size", 4),
        vit_depth=cfg.get("vit_depth", 4),
        vit_num_heads=cfg.get("vit_num_heads", 4),
        vit_mlp_ratio=cfg.get("vit_mlp_ratio", 4.0),
        vit_use_cnn_stem=cfg.get("vit_use_cnn_stem", True),
        vit_cnn_stem_reduction=cfg.get("vit_cnn_stem_reduction", 4),
    )

    if backbone_type == "dit":
        from model_multires import MultiResConditionalDiT
        model = MultiResConditionalDiT(
            image_size=cfg["image_size"],
            in_channels=cfg.get("in_channels", 3),
            cond_in_channels=cfg.get("cond_in_channels", 3),
            vae_downsample_factor=cfg.get("vae_downsample_factor", 1),
            min_patch_size=cfg.get("min_patch_size", 32),
            num_levels=cfg.get("num_levels", None),
            feat_channels=cfg.get("feat_channels", 256),
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
    level_sizes = list(encoder.level_sizes)

    # Compute vocab size
    if discretizer is not None:
        if hasattr(discretizer, 'fsq'):
            vocab_size = 1
            for l in discretizer.fsq.levels:
                vocab_size *= l
        elif hasattr(discretizer, 'codebook_size'):
            vocab_size = discretizer.codebook_size
        else:
            vocab_size = 512
    else:
        raise ValueError("Pretrained model has no discretizer")

    return model, encoder, discretizer, level_sizes, vocab_size, cfg


@torch.no_grad()
def extract_tokens(encoder, discretizer, images, device):
    """Extract flat token IDs from a batch of images.

    Args:
        encoder: HierarchicalMultiResEncoder
        discretizer: FSQDiscretizer or VQDiscretizer
        images: (B, C, H, W) tensor
        device: target device
    Returns:
        tok_ids: (B, total_tokens) long
    """
    images = images.to(device)
    level_features = encoder.forward_injection(images)

    all_tok_ids = []
    # Process levels in descending order (finest first, matching level_sizes order)
    for s in sorted(level_features.keys(), reverse=True):
        feat_2d = level_features[s]  # (B, D, S, S)
        B, D = feat_2d.shape[:2]
        tokens_2d = feat_2d.flatten(2).transpose(1, 2)  # (B, S*S, D)
        _, tok_indices = discretizer(tokens_2d)  # (B, S*S)
        all_tok_ids.append(tok_indices)

    return torch.cat(all_tok_ids, dim=1)  # (B, total_tokens)


@torch.no_grad()
def cache_all_tokens(encoder, discretizer, dataset, device,
                     batch_size=64, cache_path=None, accelerator=None):
    """Extract and cache token IDs for an entire dataset."""
    if cache_path is not None and os.path.isfile(cache_path):
        tok_ids = torch.load(cache_path, map_location="cpu", weights_only=True)
        if accelerator:
            accelerator.print(f"[cache] Loaded from {cache_path}, shape={tok_ids.shape}")
        return tok_ids

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    all_ids = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        elif isinstance(batch, dict) and "image" in batch:
            images = batch["image"]
        else:
            images = batch
        images = images.to(device)
        tok = extract_tokens(encoder, discretizer, images, device)
        all_ids.append(tok.cpu())

    tok_ids = torch.cat(all_ids, dim=0)
    if cache_path is not None:
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
    """CLEVR images with JSON conditions."""
    def __init__(self, image_root, condition_dir=None, image_size=256,
                 splits=("easy", "medium", "hard")):
        self.image_paths = []
        self.labels = []
        self.conditions = []
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # Load conditions (combined files)
        cond_map = {}
        if condition_dir is not None:
            for split in splits:
                cond_file = os.path.join(condition_dir, f"conditions_{split}.json")
                if os.path.isfile(cond_file):
                    with open(cond_file) as f:
                        cond_list = json.load(f)
                    for c in cond_list:
                        cond_map[c["image_filename"]] = c

                # Also try per-file conditions
                per_file_dir = os.path.join(condition_dir, split)
                if os.path.isdir(per_file_dir):
                    for fn in sorted(os.listdir(per_file_dir)):
                        if fn.endswith(".json"):
                            fpath = os.path.join(per_file_dir, fn)
                            with open(fpath) as f:
                                c = json.load(f)
                            cond_map[c["image_filename"]] = c

        # Collect images
        for split in splits:
            split_dir = os.path.join(image_root, split)
            if not os.path.isdir(split_dir):
                continue
            for fn in sorted(os.listdir(split_dir)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(os.path.join(split_dir, fn))
                    self.labels.append(0)  # dummy label
                    self.conditions.append(cond_map.get(fn, {}))

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

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
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


@torch.no_grad()
def decode_tokens_to_images(
    tok_ids, level_sizes, pretrained_model, discretizer, device,
    num_steps=50, guidance_scale=1.0, batch_size=16,
    noise_scale=1.0, t_eps=0.05,
):
    """Decode generated token IDs back to images using pretrained model.

    Uses flow matching ODE (Euler) for DiT backbone, or DDIM for UNet.

    Args:
        tok_ids: (N, total_tokens) long
        level_sizes: list of spatial sizes (e.g., [8, 4, 2, 1])
        pretrained_model: full multi-res model (encoder + decoder)
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
    is_dit = hasattr(pretrained_model, 'forward_from_level_features')

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_tok = tok_ids[start:end].to(device)
        B_cur = batch_tok.shape[0]

        # Decode token IDs → per-level feature maps
        level_features = _tok_ids_to_level_features(
            batch_tok, level_sizes, discretizer, device)

        z = noise_scale * torch.randn(
            B_cur, in_channels, latent_size, latent_size, device=device)

        if is_dit:
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


def save_sample_grid(images, path, nrow=8):
    """Save a grid of images as PNG."""
    from torchvision.utils import make_grid
    grid = make_grid(images * 0.5 + 0.5, nrow=nrow, padding=2)
    grid = grid.clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()
    Image.fromarray(grid).save(path)


# ────────────────────────────────────────────────────────────
#  Evaluation
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_save(
    diffusion, step, args, accelerator, ema,
    pretrained_model=None, discretizer=None, level_sizes=None,
    clevr_cond_encoder=None, val_dataset=None,
):
    if not accelerator.is_main_process:
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

    if args.dataset_type == "sudoku":
        _eval_sudoku(model, step, args, accelerator, save_dir)
    elif args.dataset_type == "imagenet":
        _eval_imagenet(model, step, args, accelerator, save_dir,
                       pretrained_model, discretizer, level_sizes)
    elif args.dataset_type == "clevr":
        _eval_clevr(model, step, args, accelerator, save_dir,
                    pretrained_model, discretizer, level_sizes,
                    clevr_cond_encoder, val_dataset)

    if ema is not None:
        ema.restore(params)
    model.train()


def _eval_sudoku(model, step, args, accelerator, save_dir):
    """Sudoku eval: generate grids, check rules."""
    device = accelerator.device
    grid_hw = args.grid_hw
    seq_len = grid_hw * grid_hw

    tokens = model.sample(
        batch_size=args.eval_num_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
    )
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


def _eval_imagenet(model, step, args, accelerator, save_dir,
                   pretrained_model, discretizer, level_sizes):
    """ImageNet eval: generate tokens for random classes, decode to images."""
    device = accelerator.device
    seq_len = sum(s * s for s in level_sizes) if level_sizes else args.seq_len
    n_samples = min(args.eval_num_samples, 64)

    # Sample with random class labels
    class_labels = torch.randint(0, args.num_classes, (n_samples,), device=device)

    tokens = model.sample(
        batch_size=n_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
        class_labels=class_labels,
    )

    # Save token stats
    txt_path = os.path.join(save_dir, f"step_{step:07d}_imagenet.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step} n_samples={n_samples}\n")
        f.write(f"class_labels={class_labels.tolist()}\n")
        f.write(f"token_range=[{tokens.min().item()}, {tokens.max().item()}]\n")
        for i in range(min(4, n_samples)):
            f.write(f"sample {i}: {tokens[i, :20].tolist()}...\n")

    # Decode to images if pretrained model available
    if pretrained_model is not None and discretizer is not None and level_sizes is not None:
        try:
            images = decode_tokens_to_images(
                tokens, level_sizes, pretrained_model, discretizer, device,
                num_steps=args.decode_num_steps,
                batch_size=min(16, n_samples),
            )
            img_path = os.path.join(save_dir, f"step_{step:07d}_imagenet_samples.png")
            save_sample_grid(images, img_path, nrow=8)
            accelerator.print(f"[eval/imagenet] step={step} saved {n_samples} samples")
        except Exception as e:
            accelerator.print(f"[eval/imagenet] decode failed: {e}")
    else:
        accelerator.print(f"[eval/imagenet] step={step} tokens generated (no decode)")


def _eval_clevr(model, step, args, accelerator, save_dir,
                pretrained_model, discretizer, level_sizes,
                clevr_cond_encoder, val_dataset):
    """CLEVR eval: generate tokens with val conditions, decode to images."""
    device = accelerator.device
    seq_len = sum(s * s for s in level_sizes) if level_sizes else args.seq_len
    n_samples = min(args.eval_num_samples, 32)

    # Get conditions from validation set
    cond_tokens = None
    if clevr_cond_encoder is not None and val_dataset is not None:
        # Collect conditions
        ea_list, em_list, rd_list, rm_list = [], [], [], []
        for i in range(n_samples):
            idx = i % len(val_dataset)
            sample = val_dataset[idx]
            ea_list.append(sample["entity_attrs"])
            em_list.append(sample["entity_mask"])
            rd_list.append(sample["relation_data"])
            rm_list.append(sample["relation_mask"])

        ea = torch.stack(ea_list).to(device)
        em = torch.stack(em_list).to(device)
        rd = torch.stack(rd_list).to(device)
        rm = torch.stack(rm_list).to(device)

        cond_encoder = clevr_cond_encoder
        if hasattr(cond_encoder, 'module'):
            cond_encoder = cond_encoder.module
        cond_tokens = cond_encoder(ea, em, rd, rm)

    tokens = model.sample(
        batch_size=n_samples,
        seq_len=seq_len,
        num_steps=args.eval_num_steps,
        device=device,
        sampler=args.sampler,
        noise_removal=True,
        cond_tokens=cond_tokens,
    )

    txt_path = os.path.join(save_dir, f"step_{step:07d}_clevr.txt")
    with open(txt_path, "w") as f:
        f.write(f"step={step} n_samples={n_samples}\n")
        f.write(f"token_range=[{tokens.min().item()}, {tokens.max().item()}]\n")

    # Decode to images
    if pretrained_model is not None and discretizer is not None and level_sizes is not None:
        try:
            images = decode_tokens_to_images(
                tokens, level_sizes, pretrained_model, discretizer, device,
                num_steps=args.decode_num_steps,
                batch_size=min(16, n_samples),
            )
            img_path = os.path.join(save_dir, f"step_{step:07d}_clevr_samples.png")
            save_sample_grid(images, img_path, nrow=8)
            accelerator.print(f"[eval/clevr] step={step} saved {n_samples} samples")
        except Exception as e:
            accelerator.print(f"[eval/clevr] decode failed: {e}")
    else:
        accelerator.print(f"[eval/clevr] step={step} tokens generated (no decode)")


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
                   help="Directory with CLEVR condition JSONs.")
    p.add_argument("--clevr_train_splits", type=str, nargs="+",
                   default=["easy", "medium", "hard"])
    p.add_argument("--clevr_val_splits", type=str, nargs="+",
                   default=["easy"])

    # ── Sudoku ──
    p.add_argument("--sudoku_config", type=str, default=None)
    p.add_argument("--grid_only", action="store_true", default=False)
    p.add_argument("--grid_hw", type=int, default=9)
    p.add_argument("--grid_vocab_size", type=int, default=9)

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

    # ── noise schedule ──
    p.add_argument("--noise_type", type=str, default="loglinear",
                   choices=["loglinear", "cosine"])
    p.add_argument("--noise_eps", type=float, default=1e-3)

    # ── diffusion ──
    p.add_argument("--antithetic_sampling", action="store_true", default=True)
    p.add_argument("--importance_sampling", action="store_true", default=False)
    p.add_argument("--change_of_variables", action="store_true", default=False)
    p.add_argument("--sampling_eps", type=float, default=1e-3)

    # ── conditioning ──
    p.add_argument("--uncond_drop_prob", type=float, default=0.1,
                   help="Probability of dropping condition (CFG training).")

    # ── training ──
    p.add_argument("--max_train_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=2000)
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
    p.add_argument("--sampler", type=str, default="ddpm_cache",
                   choices=["ddpm", "ddpm_cache", "confidence"])
    p.add_argument("--tokens_per_step", type=int, default=0)

    # ── resume ──
    p.add_argument("--resume_dir", type=str, default=None)

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

    if args.dataset_type in ("imagenet", "clevr"):
        assert args.pretrained_output_dir is not None, \
            f"--pretrained_output_dir required for dataset_type={args.dataset_type}"

        accelerator.print(f"[pretrained] Loading from {args.pretrained_output_dir}")
        pretrained_model, encoder, discretizer, level_sizes, data_vocab_size, pretrained_cfg = \
            load_pretrained_model(args.pretrained_output_dir,
                                  device=accelerator.device)
        seq_len = sum(s * s for s in level_sizes)
        accelerator.print(
            f"[pretrained] level_sizes={level_sizes}, "
            f"seq_len={seq_len}, vocab_size={data_vocab_size}")

        # After caching is done below, we move pretrained model to CPU
        # to free GPU memory; it's loaded back to GPU only during eval

    # ─────────────────────────────────────────────────────────
    #  Dataset
    # ─────────────────────────────────────────────────────────
    clevr_cond_encoder = None
    train_dataset = None
    val_dataset = None

    if args.dataset_type == "imagenet":
        accelerator.print(f"[data] Loading ImageNet from {args.dataset_root}")
        train_img_ds = ImageFolderDataset(args.dataset_root, split="train",
                                          image_size=args.image_size)
        val_img_ds = ImageFolderDataset(args.dataset_root, split="val",
                                        image_size=args.image_size)

        # Cache tokens (main process caches, others wait then load)
        cache_dir = args.token_cache_dir or os.path.join(args.output_dir, "token_cache")
        train_cache_path = os.path.join(cache_dir, "imagenet_train_tok.pt")
        val_cache_path = os.path.join(cache_dir, "imagenet_val_tok.pt")

        if accelerator.is_main_process:
            train_tok = cache_all_tokens(
                encoder, discretizer, train_img_ds, accelerator.device,
                batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
            val_tok = cache_all_tokens(
                encoder, discretizer, val_img_ds, accelerator.device,
                batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            train_tok = torch.load(train_cache_path, map_location="cpu", weights_only=True)
            val_tok = torch.load(val_cache_path, map_location="cpu", weights_only=True)

        # Extract class labels
        train_labels = torch.tensor([train_img_ds.ds.targets[i]
                                     for i in range(len(train_img_ds))], dtype=torch.long)
        val_labels = torch.tensor([val_img_ds.ds.targets[i]
                                   for i in range(len(val_img_ds))], dtype=torch.long)

        train_dataset = CachedTokenDataset(train_tok, labels=train_labels)
        val_dataset = CachedTokenDataset(val_tok, labels=val_labels)

    elif args.dataset_type == "clevr":
        accelerator.print(f"[data] Loading CLEVR from {args.dataset_root}")
        train_img_ds = CLEVRImageDataset(
            args.dataset_root, condition_dir=args.clevr_condition_dir,
            image_size=args.image_size, splits=args.clevr_train_splits)
        val_img_ds = CLEVRImageDataset(
            args.dataset_root, condition_dir=args.clevr_condition_dir,
            image_size=args.image_size, splits=args.clevr_val_splits)

        cache_dir = args.token_cache_dir or os.path.join(args.output_dir, "token_cache")
        train_cache_path = os.path.join(cache_dir, "clevr_train_tok.pt")
        val_cache_path = os.path.join(cache_dir, "clevr_val_tok.pt")

        if accelerator.is_main_process:
            train_tok = cache_all_tokens(
                encoder, discretizer, train_img_ds, accelerator.device,
                batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
            val_tok = cache_all_tokens(
                encoder, discretizer, val_img_ds, accelerator.device,
                batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            train_tok = torch.load(train_cache_path, map_location="cpu", weights_only=True)
            val_tok = torch.load(val_cache_path, map_location="cpu", weights_only=True)

        # Collect CLEVR conditions
        train_conditions = [train_img_ds.get_condition(i) for i in range(len(train_img_ds))]
        val_conditions = [val_img_ds.get_condition(i) for i in range(len(val_img_ds))]

        train_dataset = CachedTokenDataset(train_tok, clevr_conditions=train_conditions)
        val_dataset = CachedTokenDataset(val_tok, clevr_conditions=val_conditions)

        # Build CLEVR condition encoder
        clevr_cond_encoder = CLEVRConditionEncoder(args.hidden_size)
        accelerator.print(f"[clevr] condition encoder built, hidden_size={args.hidden_size}")

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
        else:
            train_dataset = train_raw
            val_dataset = val_raw

        data_vocab_size = args.grid_vocab_size
        seq_len = args.grid_hw * args.grid_hw

    accelerator.print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Move pretrained model to CPU to free GPU for training
    if pretrained_model is not None:
        pretrained_model.cpu()
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # ─────────────────────────────────────────────────────────
    #  Build discrete diffusion model
    # ─────────────────────────────────────────────────────────
    backbone_vocab_size = data_vocab_size + 1  # +1 for [MASK]

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
    )

    if args.dataset_type == "sudoku":
        dit_kwargs["pos_emb_type"] = args.pos_emb_type if args.pos_emb_type != "multires" else "2d"
        dit_kwargs["sudoku_hw"] = args.grid_hw
    elif args.dataset_type == "imagenet":
        dit_kwargs["pos_emb_type"] = "multires"
        dit_kwargs["level_sizes"] = level_sizes
        dit_kwargs["num_classes"] = args.num_classes
    elif args.dataset_type == "clevr":
        dit_kwargs["pos_emb_type"] = "multires"
        dit_kwargs["level_sizes"] = level_sizes
        dit_kwargs["use_cross_attn"] = True

    backbone = DIT(**dit_kwargs)

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
        f"[model] Total: {format_n(total_p)} (trainable {format_n(train_p)})")

    # Optimizer
    all_params = list(diffusion.parameters())
    if clevr_cond_encoder is not None:
        all_params += list(clevr_cond_encoder.parameters())
        ce_total, ce_train = count_params(clevr_cond_encoder)
        accelerator.print(f"[model] CLEVR cond encoder: {format_n(ce_total)} params")

    optimizer = torch.optim.AdamW(
        all_params, lr=args.lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=args.weight_decay)
    lr_scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_train_steps)

    # Prepare
    prepare_list = [diffusion, optimizer, train_loader, lr_scheduler]
    if clevr_cond_encoder is not None:
        prepare_list.insert(1, clevr_cond_encoder)
    prepared = accelerator.prepare(*prepare_list)

    if clevr_cond_encoder is not None:
        diffusion, clevr_cond_encoder, optimizer, train_loader, lr_scheduler = prepared
    else:
        diffusion, optimizer, train_loader, lr_scheduler = prepared

    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(diffusion).parameters(),
                  decay=args.ema_decay)
        ema.to(accelerator.device)

    # Resume
    global_step = 0
    if args.resume_dir and os.path.isdir(args.resume_dir):
        accelerator.load_state(args.resume_dir)
        global_step = parse_step_from_dir(args.resume_dir)
        accelerator.print(f"[resume] step={global_step}")

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
    running_loss = 0.0

    while global_step < args.max_train_steps:
        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            # ── Prepare batch ──
            class_labels = None
            cond_tokens = None

            if args.dataset_type == "sudoku":
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

            elif args.dataset_type == "clevr":
                # Encode CLEVR conditions
                ea = batch["entity_attrs"].to(accelerator.device)
                em = batch["entity_mask"].to(accelerator.device)
                rd = batch["relation_data"].to(accelerator.device)
                rm = batch["relation_mask"].to(accelerator.device)

                cond_enc = clevr_cond_encoder
                if hasattr(cond_enc, 'module'):
                    cond_enc = cond_enc.module
                cond_tokens = cond_enc(ea, em, rd, rm)

                # CFG dropout: zero out all cond tokens
                if args.uncond_drop_prob > 0 and diffusion.training:
                    drop_mask = (torch.rand(tokens.shape[0],
                                            device=accelerator.device) < args.uncond_drop_prob)
                    cond_tokens = cond_tokens * (~drop_mask).float().unsqueeze(-1).unsqueeze(-1)

            # ── Forward + backward ──
            with accelerator.accumulate(diffusion):
                loss_out = accelerator.unwrap_model(diffusion).compute_loss(
                    tokens, cond_tokens=cond_tokens, class_labels=class_labels)
                loss = loss_out.loss

                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if ema is not None:
                ema.update(accelerator.unwrap_model(diffusion).parameters())

            global_step += 1
            running_loss += loss.item()
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            # ── Logging ──
            if global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"[step {global_step:>7d}] loss={avg:.4f} lr={lr:.2e}")
                if accelerator.is_main_process and args.log_with:
                    accelerator.log({"train/loss": avg, "train/lr": lr},
                                    step=global_step)
                running_loss = 0.0

            # ── Eval ──
            if global_step % args.eval_every == 0:
                evaluate_and_save(
                    diffusion, global_step, args, accelerator, ema,
                    pretrained_model=pretrained_model,
                    discretizer=discretizer,
                    level_sizes=level_sizes,
                    clevr_cond_encoder=clevr_cond_encoder,
                    val_dataset=val_dataset,
                )
                diffusion.train()
                if clevr_cond_encoder is not None:
                    clevr_cond_encoder.train()

            # ── Save ──
            if global_step % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, "ckpt",
                                        f"step{global_step}")
                accelerator.save_state(ckpt_dir)
                accelerator.print(f"[save] → {ckpt_dir}")
                if accelerator.is_main_process:
                    meta = {"step": global_step, "args": vars(args)}
                    os.makedirs(ckpt_dir, exist_ok=True)
                    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2, sort_keys=True)

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
    )
    accelerator.print("Done!")


if __name__ == "__main__":
    main()
