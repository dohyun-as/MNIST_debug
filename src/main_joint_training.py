"""
Joint Training: Stage 1 (encoder + DiT, flow matching) + Stage 2 (masked
discrete diffusion on continuous encoder tokens, MAR-style diff head).

The two stages are trained simultaneously. Stage 2's masked-prediction
loss back-propagates through the encoder via the *visible* context tokens.
The masked-position *target* tokens come from an EMA teacher copy of the
encoder (stop-grad), so the encoder cannot trivially shrink its output
space to make prediction easy.

Existing files are imported, never modified:
    - model_multires.MultiResConditionalDiT       (stage 1: encoder + DiT)
    - dit_model.DIT                                (stage 2 backbone)
    - diffloss.DiffLoss                            (stage 2 head)
    - discrete_diffusion.DiscreteDiffusion         (loss math we re-use)
    - train_discrete_diffusion_v2.{CLEVRImageDataset,
            PretrainedTextConditionEncoder,
            CLEVRTextConditionEncoder, CLEVR_TEXT_VOCAB_SIZE,
            clevr_text_to_token_ids}              (stage 2 data + text enc)
    - main_multires.{build_vae, vae_encode,
            generate_samples, get_lr,
            save_checkpoint, load_checkpoint, EMA}  (stage 1 helpers)

Usage (see script/train_clevr_joint_continuous.sh):
    accelerate launch --multi_gpu src/main_joint_training.py \
        --output_dir runs/clevr/joint/256_dit_vit_flow_cont_out16 \
        --train_dir ../clevr_output/clevr_256_varied/images \
        --clevr_condition_dir ../clevr_output/clevr_256_varied/conditions_text \
        ... [encoder/DiT/diff-head/text-enc args]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# ── existing modules (read-only imports) ──
from model_multires import MultiResConditionalDiT
from dit_model import DIT
from diffloss import DiffLoss
from discrete_diffusion import DiscreteDiffusion
from train_discrete_diffusion_v2 import (
    CLEVRImageDataset,
    PretrainedTextConditionEncoder,
    CLEVRTextConditionEncoder,
    CLEVR_TEXT_VOCAB_SIZE,
    clevr_text_to_token_ids,
    # ── stage 2 eval drivers (text → token → image → CLEVR cond eval) ──
    _eval_clevr,
)
from main_multires import (
    EMA,
    build_vae,
    vae_encode,
    get_lr,
    generate_samples,
    evaluate_clevr,                # stage 1 recon → detector + classifier
    save_checkpoint as _save_stage1_ckpt,
)
# torchvision ImageFolder is what main_multires.evaluate_clevr expects as
# val_dataset (uses .samples and __getitem__→(img,label)).
from torchvision import datasets, transforms


# ──────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # ── output / resume ──
    p.add_argument("--output_dir", type=str, default="runs/clevr/joint")
    p.add_argument("--resume_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # ── data ──
    p.add_argument("--train_dir", type=str, required=True,
                   help="CLEVR train image root (contains easy/medium/hard).")
    p.add_argument("--val_dir", type=str, default=None,
                   help="CLEVR val image root (for sample grids).")
    p.add_argument("--clevr_condition_dir", type=str, required=True)
    p.add_argument("--clevr_val_condition_dir", type=str, default=None)
    p.add_argument("--clevr_train_splits", type=str, nargs="+",
                   default=["easy", "medium", "hard"])
    p.add_argument("--clevr_val_splits", type=str, nargs="+",
                   default=["easy", "medium", "hard"])
    p.add_argument("--clevr_cond_type", type=str, default="text",
                   choices=["text"],
                   help="Joint training only supports text-conditioned CLEVR.")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--cond_in_channels", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=8)

    # ── VAE (optional latent space) ──
    p.add_argument("--vae_pretrained", type=str, default=None)
    p.add_argument("--vae_downsample_factor", type=int, default=1)
    p.add_argument("--cond_use_latent", action="store_true", default=False)

    # ── encoder (passed through to MultiResConditionalDiT) ──
    p.add_argument("--encoder_type", type=str, default="vit",
                   choices=["cnn", "vit", "swin", "vit_global"])
    p.add_argument("--min_patch_size", type=int, default=32)
    p.add_argument("--num_levels", type=int, default=None)
    p.add_argument("--feat_channels", type=int, default=16)
    p.add_argument("--encoder_internal_dim", type=int, default=256)
    p.add_argument("--depth_per_level", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cnn_base_channels", type=int, default=64)
    p.add_argument("--vit_patch_size", type=int, default=4)
    p.add_argument("--vit_depth", type=int, default=4)
    p.add_argument("--vit_num_heads", type=int, default=4)
    p.add_argument("--vit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--vit_use_cnn_stem", action="store_true", default=True)
    p.add_argument("--vit_no_cnn_stem", action="store_true", default=False)
    p.add_argument("--vit_cnn_stem_reduction", type=int, default=4)
    p.add_argument("--vit_init_clip", action="store_true", default=False)
    p.add_argument("--clip_model_name", type=str,
                   default="openai/clip-vit-base-patch16")
    p.add_argument("--mae_mask_ratio", type=float, default=0.0)
    p.add_argument("--level_sizes", type=int, nargs="+", default=None)

    # ── stage 1 DiT ──
    p.add_argument("--dit_patch_size", type=int, default=16)
    p.add_argument("--dit_hidden_size", type=int, default=768)
    p.add_argument("--dit_n_heads", type=int, default=12)
    p.add_argument("--dit_n_blocks", type=int, default=12)
    p.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--dit_dropout", type=float, default=0.0)
    p.add_argument("--dit_bottleneck_dim", type=int, default=128)
    p.add_argument("--dit_in_context_len", type=int, default=32)
    p.add_argument("--dit_in_context_start", type=int, default=4)

    # ── stage 1 flow matching ──
    p.add_argument("--use_flow_matching", action="store_true", default=True)
    p.add_argument("--flow_P_mean", type=float, default=-0.8)
    p.add_argument("--flow_P_std", type=float, default=0.8)
    p.add_argument("--flow_t_eps", type=float, default=0.05)
    p.add_argument("--flow_noise_scale", type=float, default=1.0)
    p.add_argument("--flow_sampling_method", type=str, default="euler")

    # ── stage 1 conditioning drop ──
    p.add_argument("--uncond_drop_prob", type=float, default=0.1)
    p.add_argument("--level_drop", action="store_true", default=False)
    p.add_argument("--min_keep_levels", type=int, default=1)
    p.add_argument("--level_drop_after_steps", type=int, default=0)
    p.add_argument("--cond_token_drop_prob", type=float, default=0.0)
    p.add_argument("--guidance_scale", type=float, default=3.0)

    # ── stage 2 backbone (DIT, continuous mode) ──
    p.add_argument("--s2_hidden_size", type=int, default=768)
    p.add_argument("--s2_n_heads", type=int, default=12)
    p.add_argument("--s2_n_blocks", type=int, default=12)
    p.add_argument("--s2_cond_dim", type=int, default=256)
    p.add_argument("--s2_mlp_ratio", type=int, default=4,
                   help="DIT (stage-2 backbone) MLP expansion (must be int).")
    p.add_argument("--s2_dropout", type=float, default=0.1)
    p.add_argument("--s2_pos_emb_type", type=str, default="multires",
                   choices=["1d", "2d", "multires"])

    # ── stage 2 noise schedule (MDLM) ──
    p.add_argument("--s2_noise_type", type=str, default="loglinear",
                   choices=["loglinear", "cosine"])
    p.add_argument("--s2_noise_eps", type=float, default=1e-3)
    p.add_argument("--s2_sampling_eps", type=float, default=1e-3)
    p.add_argument("--s2_antithetic_sampling", action="store_true",
                   default=True)
    p.add_argument("--s2_uncond_drop_prob", type=float, default=0.1,
                   help="CFG dropout for text condition tokens (stage 2).")

    # ── stage 2 diffusion head ──
    p.add_argument("--diff_head_depth", type=int, default=6)
    p.add_argument("--diff_head_width", type=int, default=1024)
    p.add_argument("--diff_head_num_sampling_steps", type=int, default=100)
    p.add_argument("--diff_head_batch_mul", type=int, default=4)
    p.add_argument("--diff_head_cond_drop_prob", type=float, default=0.0)
    # CFG mode = backbone (uncond replaces cond_tokens with text-encoder
    # null_embed at backbone input). diffusion-head-side drop is forced 0.

    # ── stage 2 text encoder (CLIP / T5) ──
    p.add_argument("--use_pretrained_text_encoder", action="store_true",
                   default=True)
    p.add_argument("--pretrained_text_model_name", type=str,
                   default="openai/clip-vit-base-patch32")
    p.add_argument("--pretrained_text_max_length", type=int, default=77)
    p.add_argument("--freeze_text_encoder", action="store_true", default=False)
    p.add_argument("--unfreeze_text_encoder", action="store_true",
                   default=True)
    p.add_argument("--text_encoder_lr", type=float, default=3e-5)

    # ── joint training knobs ──
    p.add_argument("--lambda_stage1", type=float, default=1.0)
    p.add_argument("--lambda_stage2", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9995,
                   help="EMA decay for the *teacher* stage-1 model. "
                        "Teacher provides target tokens for the masked diff "
                        "loss with stop-grad. Set 0 to use the live encoder "
                        "(detached) as teacher (less stable).")
    p.add_argument("--stage2_warmup_steps", type=int, default=0,
                   help="Hold stage-2 loss off for the first N steps so the "
                        "encoder can stabilise on stage-1 alone.")

    # ── training ──
    p.add_argument("--max_train_steps", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--blr", type=float, default=2.5e-5,
                   help="Base LR; effective LR = blr * eff_bs / 256.")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr_schedule", type=str, default="constant",
                   choices=["constant", "cosine"])
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])

    # ── logging / saving / sampling ──
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=5000,
                   help="Stage-1 [GT|recon] grid (cheap, every step).")

    # ── stage 1 recon eval (mirrors --clevr_eval_every from stage 1 script) ──
    p.add_argument("--clevr_eval_every", type=int, default=5000,
                   help="Run main_multires.evaluate_clevr (recon → "
                        "detector + classifier) every N steps. 0 = off.")
    p.add_argument("--clevr_eval_samples", type=int, default=50,
                   help="# val images for stage-1 recon eval.")

    # ── stage 2 text → token → image eval (mirrors --eval_every from stage 2 script) ──
    p.add_argument("--eval_every", type=int, default=10000,
                   help="Run train_discrete_diffusion_v2._eval_clevr "
                        "(text → tokens → image → CLEVR cond eval) "
                        "every N steps. 0 = off.")
    p.add_argument("--eval_num_samples", type=int, default=30,
                   help="Per-split CLEVR samples for stage-2 eval.")
    p.add_argument("--eval_num_steps", type=int, default=128,
                   help="MDLM sampling steps for stage-2 token gen "
                        "(also used for stage-1 sample ODE).")
    p.add_argument("--decode_num_steps", type=int, default=50,
                   help="Stage-1 ODE steps for token → image decode.")
    p.add_argument("--eval_sample_batch_size", type=int, default=8,
                   help="Per-rank chunk size during stage-2 sampling.")
    p.add_argument("--diff_head_temperature", type=float, default=1.0)
    p.add_argument("--diff_head_cfg", type=float, default=3.0)
    p.add_argument("--cfg_schedule", type=str, default="linear",
                   choices=["constant", "linear"])
    p.add_argument("--cfg_mode", type=str, default="backbone",
                   choices=["head", "backbone"])
    p.add_argument("--sampler", type=str, default="ddpm_cache",
                   choices=["ddpm_cache", "confidence"])

    # ── compatibility shims for imported eval functions ──
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Used by main_multires.evaluate_clevr to derive "
                        "scenes dir; defaults to --val_dir parent.")
    p.add_argument("--dataset_type", type=str, default="clevr",
                   choices=["clevr"])

    # arg compatibility shims expected by imported helpers
    p.add_argument("--backbone", type=str, default="dit")
    p.add_argument("--use_fsq", action="store_true", default=False)
    p.add_argument("--fsq_levels", type=int, nargs="+", default=None)
    p.add_argument("--fsq_drop_quant_p", type=float, default=0.0)
    p.add_argument("--fsq_corrupt_tokens_p", type=float, default=0.0)
    p.add_argument("--use_vq", action="store_true", default=False)
    p.add_argument("--vq_codebook_size", type=int, default=512)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--vq_loss_weight", type=float, default=0.1)
    p.add_argument("--eval_slot_configs", type=int, nargs="+", default=[256])

    args = p.parse_args()

    # vit_use_cnn_stem resolution (matches main_multires)
    if args.vit_no_cnn_stem:
        args.vit_use_cnn_stem = False

    # Hard-coded for joint trainer (read by _eval_clevr).
    args.use_diffusion_head = True
    args.model_type = "diffusion"
    args.factorized_head = False
    args.ar_temperature = 1.0
    args.ar_top_k = 0
    args.ar_top_p = 1.0
    args.log_with = "tensorboard"
    args.tokens_per_step = 0

    if args.dataset_root is None:
        args.dataset_root = args.val_dir or args.train_dir

    return args


# ──────────────────────────────────────────────────────────────────
#  Dataset wrapper: image + raw text caption per item
# ──────────────────────────────────────────────────────────────────

class JointCLEVRDataset(torch.utils.data.Dataset):
    """Yields {"image": (3,H,W), "cond_text": str} per sample.

    Also exposes the API that train_discrete_diffusion_v2._eval_clevr
    expects:
        get_condition(idx)  -> raw condition dict
        source_image_ds     -> dataset whose __getitem__ returns
                                {"image": ...} (used for GT grids).
    """

    def __init__(self, image_root, condition_dir, image_size, splits,
                 cond_type="text"):
        self.inner = CLEVRImageDataset(
            image_root, condition_dir=condition_dir,
            image_size=image_size, splits=splits, cond_type=cond_type)
        # Used by _eval_clevr to build [GT | gen] sample grids.
        self.source_image_ds = self.inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        cond = self.inner.get_condition(idx)
        if isinstance(cond, dict):
            text = cond.get("text", "")
        elif isinstance(cond, str):
            text = cond
        else:
            text = ""
        return {"image": item["image"], "cond_text": text}

    def get_condition(self, idx):
        return self.inner.get_condition(idx)


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def encoder_features_to_tokens(level_features: dict) -> torch.Tensor:
    """Concat multi-res features into (B, L, D), matching the order used by
    train_discrete_diffusion_v2.extract_continuous_tokens (descending size)."""
    feats = []
    for s in sorted(level_features.keys(), reverse=True):
        feat_2d = level_features[s]                       # (B, D, S, S)
        feats.append(feat_2d.flatten(2).transpose(1, 2))  # (B, S*S, D)
    return torch.cat(feats, dim=1)


def build_models(args, accelerator):
    """Construct stage 1 model, stage 2 backbone, diff head, text encoder."""

    # ── Stage 1: encoder + flow-matching DiT ──
    stage1_model = MultiResConditionalDiT(
        image_size=args.image_size,
        in_channels=args.in_channels,
        cond_in_channels=args.cond_in_channels,
        vae_downsample_factor=args.vae_downsample_factor,
        min_patch_size=args.min_patch_size,
        num_levels=args.num_levels,
        feat_channels=args.feat_channels,
        encoder_internal_dim=args.encoder_internal_dim,
        depth_per_level=args.depth_per_level,
        mlp_ratio=args.mlp_ratio,
        cnn_base_channels=args.cnn_base_channels,
        encoder_type=args.encoder_type,
        vit_patch_size=args.vit_patch_size,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_ratio=args.vit_mlp_ratio,
        vit_use_cnn_stem=(args.vit_use_cnn_stem and not args.vit_no_cnn_stem),
        vit_cnn_stem_reduction=args.vit_cnn_stem_reduction,
        vit_init_clip=args.vit_init_clip,
        clip_model_name=args.clip_model_name,
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
        level_drop_after_steps=args.level_drop_after_steps,
        cond_token_drop_prob=args.cond_token_drop_prob,
        cond_use_latent=args.cond_use_latent,
        mae_mask_ratio=args.mae_mask_ratio,
        use_fsq=args.use_fsq,
        fsq_levels=args.fsq_levels,
        use_vq=args.use_vq,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        level_sizes=args.level_sizes,
    )

    encoder = stage1_model.encoder
    level_sizes = list(encoder.level_sizes)         # descending
    feat_dim = encoder.feat_channels
    seq_len = sum(s * s for s in level_sizes)

    accelerator.print(
        f"[stage1] encoder.level_sizes={level_sizes}  feat_dim={feat_dim}")
    accelerator.print(f"[stage2] continuous seq_len={seq_len}")

    # ── Stage 2: DIT backbone (continuous mode) ──
    stage2_backbone = DIT(
        vocab_size=1,                       # unused in continuous mode
        seq_len=seq_len,
        hidden_size=args.s2_hidden_size,
        n_heads=args.s2_n_heads,
        n_blocks=args.s2_n_blocks,
        cond_dim=args.s2_cond_dim,
        mlp_ratio=args.s2_mlp_ratio,
        dropout=args.s2_dropout,
        causal=False,
        factorized_head=False,
        fsq_levels=None,
        pos_emb_type=args.s2_pos_emb_type,
        level_sizes=level_sizes,
        continuous_mode=True,
        continuous_dim=feat_dim,
    )

    # ── Stage 2: diffusion head ──
    diff_head = DiffLoss(
        target_channels=feat_dim,
        z_channels=args.s2_hidden_size,
        depth=args.diff_head_depth,
        width=args.diff_head_width,
        num_sampling_steps=args.diff_head_num_sampling_steps,
        cond_drop_prob=0.0,                 # backbone-side CFG only
    )

    diffusion = DiscreteDiffusion(
        backbone=stage2_backbone,
        vocab_size=1,
        noise_type=args.s2_noise_type,
        noise_eps=args.s2_noise_eps,
        antithetic_sampling=args.s2_antithetic_sampling,
        importance_sampling=False,
        change_of_variables=False,
        sampling_eps=args.s2_sampling_eps,
        diff_head=diff_head,
        diffusion_batch_mul=args.diff_head_batch_mul,
        time_conditioning=False,
    )

    # ── Stage 2: text encoder ──
    if args.use_pretrained_text_encoder:
        freeze = args.freeze_text_encoder and not args.unfreeze_text_encoder
        text_encoder = PretrainedTextConditionEncoder(
            model_name=args.pretrained_text_model_name,
            hidden_size=args.s2_hidden_size,
            max_length=args.pretrained_text_max_length,
            freeze=freeze,
        )
        accelerator.print(
            f"[text-enc] PRETRAINED ({text_encoder._kind}, "
            f"{args.pretrained_text_model_name}, "
            f"max_len={args.pretrained_text_max_length}, freeze={freeze})")
    else:
        text_encoder = CLEVRTextConditionEncoder(args.s2_hidden_size)
        accelerator.print(
            f"[text-enc] word-vocab CLEVR encoder, "
            f"vocab_size={CLEVR_TEXT_VOCAB_SIZE}, "
            f"hidden_size={args.s2_hidden_size}")

    return stage1_model, diffusion, text_encoder, level_sizes, feat_dim, seq_len


def encode_text_batch(text_encoder, texts, device, drop_prob, training):
    """Encode raw caption strings → (B, L, D) cond tokens. CFG dropout
    swaps a fraction of the batch with the encoder's null_embed."""

    if isinstance(text_encoder, PretrainedTextConditionEncoder):
        text_tokens = text_encoder.tokenize(list(texts), device)
        cond, _attn = text_encoder(text_tokens)         # (B, L, D)
    else:
        # fallback: word-vocab encoder. tokenize per-sample → pad to max len
        ids_list = [clevr_text_to_token_ids(t) for t in texts]
        max_l = max(int(x.shape[0]) for x in ids_list)
        padded = torch.zeros(len(ids_list), max_l, dtype=torch.long)
        for i, x in enumerate(ids_list):
            padded[i, :x.shape[0]] = x
        cond = text_encoder(padded.to(device))          # (B, L, D)

    if drop_prob > 0 and training:
        B, L, _ = cond.shape
        drop_mask = (torch.rand(B, device=device) < drop_prob)
        if isinstance(text_encoder, PretrainedTextConditionEncoder):
            null_expanded = text_encoder.get_null_cond(B, L, device).to(cond.dtype)
            cond = torch.where(drop_mask[:, None, None], null_expanded, cond)
        else:
            cond = cond * (~drop_mask)[:, None, None].float()

    return cond


@torch.no_grad()
def encode_teacher_tokens(teacher_stage1, images):
    """EMA-teacher continuous tokens. Returns (B, L, D), detached."""
    teacher_stage1.eval()
    level_features = teacher_stage1.encoder.forward_injection(images)
    z = encoder_features_to_tokens(level_features)
    return z.detach()


def compute_masked_diff_loss(diffusion, z_student, z_teacher,
                             cond_tokens, accelerator):
    """Re-implements DiscreteDiffusion.compute_loss_continuous with
    *separate* student (visible context, has grad) and teacher (target,
    detached) latents. This is the only way to get stop-grad targets
    without modifying discrete_diffusion.py.

    Args:
        diffusion:   DiscreteDiffusion (continuous mode, has diff_head)
        z_student:   (B, L, D) student encoder tokens — gradients flow back
        z_teacher:   (B, L, D) teacher encoder tokens — detached
        cond_tokens: (B, L_cond, D_h) text-prefix conditioning
    Returns:
        scalar loss
    """
    B, L, D = z_student.shape
    device = z_student.device

    # Sample t and absorbing-state mask probability.
    t = diffusion._sample_t(B, device)
    sigma, _dsigma = diffusion.noise(t)
    move_chance = 1 - torch.exp(-sigma[:, None])         # (B, 1)
    mask = torch.rand(B, L, device=device) < move_chance # (B, L) bool

    # Backbone input: visible positions use student (grad), masked positions
    # use teacher (detached, but the value is overwritten by cont_mask_emb
    # inside DIT.forward anyway — see dit_model.py:700-703). Putting teacher
    # there is harmless and a touch safer should DIT semantics ever change.
    x0_input = torch.where(mask.unsqueeze(-1), z_teacher, z_student)

    sigma_in = diffusion._t(sigma if sigma.ndim == 1 else sigma.squeeze(-1))

    # Forward backbone. cond_tokens (B, L_cond, hidden) are text tokens
    # already projected to s2_hidden_size by the text encoder.
    hidden = diffusion.backbone(
        indices=None,
        sigma=sigma_in,
        cond_tokens=cond_tokens,
        prefix_mode=(cond_tokens is not None),
        cont_tokens=x0_input,
        mask=mask,
    )                                                    # (B, L, hidden)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    masked_hidden = hidden[mask]                         # (N, hidden)
    masked_target = z_teacher[mask]                      # (N, D) — detached

    if diffusion.diffusion_batch_mul > 1:
        masked_hidden = masked_hidden.repeat(diffusion.diffusion_batch_mul, 1)
        masked_target = masked_target.repeat(diffusion.diffusion_batch_mul, 1)

    return diffusion.diff_head(target=masked_target, z=masked_hidden)


# ──────────────────────────────────────────────────────────────────
#  Stage 2 eval driver
# ──────────────────────────────────────────────────────────────────

def run_stage2_eval(diffusion, text_encoder, stage1_decoder, level_sizes,
                    val_dataset, clevr_detector, clevr_classifier,
                    step, args, accelerator):
    """Wraps train_discrete_diffusion_v2._eval_clevr.

    The signature of _eval_clevr is:
        _eval_clevr(model, step, args, accelerator, save_dir,
                    pretrained_model, discretizer, level_sizes,
                    clevr_cond_encoder, val_dataset, ...)

    For joint training:
        model            = unwrapped DiscreteDiffusion (continuous mode)
        pretrained_model = stage 1 model (EMA shadow if available) — used
                            by decode_continuous_tokens_to_images to ODE
                            from generated tokens back to pixels.
        discretizer      = None (we don't quantize — diffusion head only).
        clevr_cond_encoder = text encoder.
    """
    save_dir = os.path.join(args.output_dir, "eval_samples")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    model = accelerator.unwrap_model(diffusion)
    cond_enc = accelerator.unwrap_model(text_encoder)
    decoder = stage1_decoder      # already unwrapped or EMA shadow

    model.eval()
    cond_enc.eval()
    decoder.eval()

    # _eval_clevr expects the decoder on GPU (it doesn't move it itself
    # for us — only the original stage 2 trainer .cpu()'s it after eval).
    decoder.to(accelerator.device)

    accelerator.wait_for_everyone()

    _eval_clevr(
        model, step, args, accelerator, save_dir,
        pretrained_model=decoder,
        discretizer=None,
        level_sizes=level_sizes,
        clevr_cond_encoder=cond_enc,
        val_dataset=val_dataset,
        clevr_detector=clevr_detector,
        clevr_classifier=clevr_classifier,
        log_prefix="eval",
    )

    accelerator.wait_for_everyone()


# ──────────────────────────────────────────────────────────────────
#  Checkpoint
# ──────────────────────────────────────────────────────────────────

def save_joint_checkpoint(accelerator, stage1_model, diffusion, text_encoder,
                          optimizer, step, args, ema=None):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    save_dir = os.path.join(args.output_dir, "checkpoints", f"step_{step:07d}")
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "stage1_model": accelerator.unwrap_model(stage1_model).state_dict(),
        "diffusion": accelerator.unwrap_model(diffusion).state_dict(),
        "text_encoder": accelerator.unwrap_model(text_encoder).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    if ema is not None:
        ckpt["ema_stage1"] = ema.state_dict()
    torch.save(ckpt, os.path.join(save_dir, "checkpoint.pt"))
    accelerator.print(f"[ckpt] saved → {save_dir}")


def load_joint_checkpoint(accelerator, stage1_model, diffusion, text_encoder,
                          optimizer, args, ema=None):
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
    accelerator.print(f"[resume] {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    accelerator.unwrap_model(stage1_model).load_state_dict(ckpt["stage1_model"])
    accelerator.unwrap_model(diffusion).load_state_dict(ckpt["diffusion"])
    accelerator.unwrap_model(text_encoder).load_state_dict(ckpt["text_encoder"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema_stage1" in ckpt:
        ema.load_state_dict(ckpt["ema_stage1"])
    return ckpt.get("step", latest)


# ──────────────────────────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────────────────────────

def train(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_config=project_config,
    )
    accelerator.init_trackers("joint_clevr")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump(
                {"cmd": " ".join(sys.argv), "args": vars(args)},
                f, indent=2, sort_keys=True,
            )

    set_seed(args.seed + accelerator.process_index)

    # ── effective LR ──
    eff_bs = args.batch_size * args.grad_accum_steps * accelerator.num_processes
    lr = args.lr if args.lr is not None else args.blr * eff_bs / 256
    accelerator.print(f"[lr] eff_bs={eff_bs}  lr={lr:.2e}")

    # ── data ──
    val_image_root = args.val_dir or args.train_dir
    val_cond_dir = args.clevr_val_condition_dir or args.clevr_condition_dir

    train_ds = JointCLEVRDataset(
        image_root=args.train_dir,
        condition_dir=args.clevr_condition_dir,
        image_size=args.image_size,
        splits=args.clevr_train_splits,
        cond_type=args.clevr_cond_type,
    )
    val_ds = JointCLEVRDataset(
        image_root=val_image_root,
        condition_dir=val_cond_dir,
        image_size=args.image_size,
        splits=args.clevr_val_splits,
        cond_type=args.clevr_cond_type,
    )
    accelerator.print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # Stage-1 recon eval expects an ImageFolder-shape dataset (uses .samples
    # and __getitem__→(img, label)) — same setup as main_multires.train.
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    try:
        val_imagefolder_ds = datasets.ImageFolder(
            val_image_root, transform=val_transform)
        accelerator.print(
            f"[data] val ImageFolder ({val_image_root}): "
            f"{len(val_imagefolder_ds)} images for stage-1 recon eval")
    except Exception as exc:
        accelerator.print(
            f"[data] could not build ImageFolder for stage-1 recon eval: {exc}")
        val_imagefolder_ds = None

    # CLEVR detector + classifier for both eval paths. Same loader as the
    # stage-2 script (see train_discrete_diffusion_v2.main:3826-3834).
    clevr_detector = clevr_classifier = None
    try:
        from eval_clevr_condition import load_eval_models
        clevr_detector, clevr_classifier = load_eval_models(
            device=accelerator.device)
        accelerator.print("[clevr] detector + classifier loaded")
    except Exception as exc:
        accelerator.print(
            f"[clevr] WARNING: could not load detector/classifier: {exc}\n"
            f"[clevr] stage-1 recon eval and stage-2 cond eval will skip.")

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── models ──
    stage1_model, diffusion, text_encoder, level_sizes, feat_dim, seq_len = \
        build_models(args, accelerator)

    # ── optimizer (split text-encoder LR like stage 2 script) ──
    decay, no_decay = [], []
    for n, p in stage1_model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "bias" in n) else decay).append(p)
    for n, p in diffusion.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "bias" in n) else decay).append(p)

    main_params = decay + no_decay
    te_params = []
    is_pretrained_te = isinstance(text_encoder, PretrainedTextConditionEncoder)
    if is_pretrained_te and not text_encoder.freeze:
        for n, p in text_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("encoder."):
                te_params.append(p)
            else:
                main_params.append(p)
    else:
        for p in text_encoder.parameters():
            if p.requires_grad:
                main_params.append(p)

    param_groups = [{"params": main_params, "lr": lr,
                     "weight_decay": args.weight_decay}]
    if te_params:
        te_lr = args.text_encoder_lr if args.text_encoder_lr is not None else lr * 0.1
        param_groups.append({"params": te_params, "lr": te_lr,
                             "weight_decay": args.weight_decay})
        accelerator.print(
            f"[opt] text-encoder unfrozen, lr={te_lr:.2e} "
            f"(main={lr:.2e})")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # ── Accelerate prepare ──
    stage1_model, diffusion, text_encoder, optimizer, train_loader = \
        accelerator.prepare(
            stage1_model, diffusion, text_encoder, optimizer, train_loader)

    # ── EMA teacher (deepcopy of unwrapped stage1) ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(stage1_model),
                  decay=args.ema_decay)
        accelerator.print(f"[ema] teacher stage-1 model, decay={args.ema_decay}")

    # ── VAE (optional, frozen) ──
    vae = build_vae(args, accelerator.device)
    if vae is not None:
        accelerator.print(
            f"[vae] {args.vae_pretrained} (×{args.vae_downsample_factor})")

    # ── resume ──
    global_step = load_joint_checkpoint(
        accelerator, stage1_model, diffusion, text_encoder, optimizer, args,
        ema=ema)
    accelerator.print(f"[start] step={global_step}")

    # ── train loop ──
    pbar = tqdm(initial=global_step, total=args.max_train_steps,
                disable=not accelerator.is_main_process,
                desc="joint", dynamic_ncols=True)

    stage1_model.train()
    diffusion.train()
    text_encoder.train()

    epoch = 0
    t_start = time.time()
    all_params = [g["params"] for g in param_groups]
    all_params_flat = [p for g in all_params for p in g]

    # collate that keeps cond_text as a list of strings
    def collate(batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        texts = [b["cond_text"] for b in batch]
        return {"image": images, "cond_text": texts}

    train_loader.collate_fn = collate

    while global_step < args.max_train_steps:
        train_sampler.set_epoch(epoch)
        epoch += 1

        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            images = batch["image"].to(accelerator.device,
                                       non_blocking=True)
            texts = batch["cond_text"]

            # LR schedule
            cur_lr = get_lr(global_step, args.warmup_steps,
                            args.max_train_steps, lr,
                            schedule=args.lr_schedule)
            for pg_i, pg in enumerate(optimizer.param_groups):
                if pg_i == 0:
                    pg["lr"] = cur_lr
                else:
                    # text encoder group: scale by ratio of the original LRs
                    pg["lr"] = cur_lr * (args.text_encoder_lr / lr) \
                        if args.text_encoder_lr is not None else cur_lr * 0.1

            # propagate step to the (DDP-wrapped) stage 1 model so its
            # level-drop schedule advances.
            accelerator.unwrap_model(stage1_model).set_step(global_step)

            with accelerator.accumulate(stage1_model, diffusion,
                                        text_encoder):
                # -------- Stage 1 input prep --------
                if vae is not None:
                    x0 = vae_encode(vae, images)
                else:
                    x0 = images
                cond_images = x0 if args.cond_use_latent else images

                with accelerator.autocast():
                    # ============================================
                    # Stage 1 loss: flow matching
                    # ============================================
                    z_t = (torch.randn(x0.shape[0], device=x0.device)
                           * args.flow_P_std + args.flow_P_mean)
                    t_flow = torch.sigmoid(z_t)              # (B,)
                    t_exp = t_flow.view(-1, 1, 1, 1)

                    e = torch.randn_like(x0) * args.flow_noise_scale
                    noisy = t_exp * x0 + (1 - t_exp) * e
                    v_target = ((x0 - noisy)
                                / (1 - t_exp).clamp_min(args.flow_t_eps))

                    use_aux = args.use_vq
                    if use_aux:
                        x_pred, aux = stage1_model(
                            noisy, t_flow, cond_image=cond_images,
                            return_aux_loss=True)
                    else:
                        x_pred = stage1_model(
                            noisy, t_flow, cond_image=cond_images)
                        aux = {}

                    v_pred = ((x_pred - noisy)
                              / (1 - t_exp).clamp_min(args.flow_t_eps))
                    loss_stage1 = F.mse_loss(v_pred, v_target)
                    if "vq_loss" in aux:
                        loss_stage1 = (loss_stage1
                                       + args.vq_loss_weight * aux["vq_loss"])

                    # ============================================
                    # Stage 2 loss: masked continuous diffusion
                    # ============================================
                    do_stage2 = (args.lambda_stage2 > 0
                                 and global_step >= args.stage2_warmup_steps)
                    if do_stage2:
                        # Student tokens: same encoder, fresh forward pass
                        # so gradients flow only via stage 2 path here.
                        student_lvl = accelerator.unwrap_model(
                            stage1_model).encoder.forward_injection(images)
                        z_student = encoder_features_to_tokens(student_lvl)

                        # Teacher tokens: EMA snapshot, no grad.
                        if ema is not None:
                            z_teacher = encode_teacher_tokens(
                                ema.shadow, images)
                        else:
                            # fallback: detached copy of student
                            z_teacher = z_student.detach()

                        cond_tokens = encode_text_batch(
                            accelerator.unwrap_model(text_encoder),
                            texts, accelerator.device,
                            drop_prob=args.s2_uncond_drop_prob,
                            training=True,
                        )

                        loss_stage2 = compute_masked_diff_loss(
                            accelerator.unwrap_model(diffusion),
                            z_student, z_teacher,
                            cond_tokens, accelerator)
                    else:
                        loss_stage2 = torch.zeros((), device=accelerator.device)

                    loss = (args.lambda_stage1 * loss_stage1
                            + args.lambda_stage2 * loss_stage2)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_params_flat,
                                                args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            if ema is not None:
                ema.update(accelerator.unwrap_model(stage1_model))

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(
                s1=f"{loss_stage1.item():.4f}",
                s2=f"{loss_stage2.item():.4f}",
                lr=f"{cur_lr:.1e}",
                refresh=False,
            )

            if global_step % args.log_every == 0:
                log = {
                    "loss/total": loss.item(),
                    "loss/stage1": loss_stage1.item(),
                    "loss/stage2": loss_stage2.item(),
                    "lr": cur_lr,
                }
                if "vq_loss" in aux:
                    log["vq_loss"] = aux["vq_loss"].item()
                accelerator.log(log, step=global_step)

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                # Cheap visual sanity: stage-1 [GT|recon] grid.
                ema_eval = ema.shadow if ema is not None else None
                try:
                    generate_samples(
                        stage1_model, val_imagefolder_ds, None, args,
                        accelerator, global_step, vae=vae,
                        ema_model=ema_eval)
                except Exception as exc:
                    accelerator.print(f"[sample] skipped: {exc}")

            # ── Stage 1 recon eval (matches train_clevr_dit_our_continuous.sh) ──
            if (args.clevr_eval_every > 0
                    and global_step % args.clevr_eval_every == 0
                    and global_step > 0
                    and val_imagefolder_ds is not None):
                ema_eval = ema.shadow if ema is not None else None
                try:
                    evaluate_clevr(
                        stage1_model, val_imagefolder_ds, args, accelerator,
                        global_step, vae=vae, ema_model=ema_eval,
                        num_samples=args.clevr_eval_samples)
                except Exception as exc:
                    accelerator.print(f"[clevr-recon-eval] skipped: {exc}")

            # ── Stage 2 text → token → image eval ──
            # (matches train_discrete_diffusion_clevr_ours_text_diffhead_clip.sh)
            if (args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and global_step > 0):
                try:
                    run_stage2_eval(
                        diffusion=diffusion,
                        text_encoder=text_encoder,
                        stage1_decoder=(ema.shadow if ema is not None
                                        else accelerator.unwrap_model(
                                            stage1_model)),
                        level_sizes=level_sizes,
                        val_dataset=val_ds,
                        clevr_detector=clevr_detector,
                        clevr_classifier=clevr_classifier,
                        step=global_step,
                        args=args,
                        accelerator=accelerator,
                    )
                except Exception as exc:
                    import traceback
                    accelerator.print(
                        f"[stage2-eval] failed: {exc}\n"
                        f"{traceback.format_exc()}")
                # _eval_clevr puts modules in eval(); restore train()
                stage1_model.train(); diffusion.train(); text_encoder.train()

            if global_step % args.save_every == 0:
                save_joint_checkpoint(
                    accelerator, stage1_model, diffusion, text_encoder,
                    optimizer, global_step, args, ema=ema)

    pbar.close()
    save_joint_checkpoint(
        accelerator, stage1_model, diffusion, text_encoder,
        optimizer, global_step, args, ema=ema)
    accelerator.print(f"[done] elapsed={time.time() - t_start:.0f}s")
    accelerator.end_training()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
