"""
Text-Conditioned Diffusion — Training Script (Baseline)
========================================================

Baseline for CLEVR conditional generation using standard text-to-image
diffusion approach:

  Mode 1 (pretrained, default):
    CLEVR JSON → natural language text → frozen T5/CLIP → projection →
    cross-attention → DiT → image

  Mode 2 (scratch, ablation):
    CLEVR JSON → structured tensors → from-scratch encoder → DiT → image

No multi-resolution encoding, no discrete diffusion stage.

Usage:
  bash script/train_text_conditioned_clevr.sh
"""

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image


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
    p.add_argument("--output_dir", type=str, default="runs/text_cond_clevr")
    p.add_argument("--resume_dir", type=str, default=None)
    p.add_argument("--eval_only", action="store_true",
                   help="Resume from ckpt (--resume_dir or output_dir's latest), "
                        "run evaluate_clevr once with --clevr_eval_samples, exit.")
    p.add_argument("--clevr_image_root", type=str, required=True)
    p.add_argument("--clevr_condition_dir", type=str, required=True)
    p.add_argument("--clevr_train_splits", type=str, nargs="+",
                   default=["easy", "medium", "hard"])
    p.add_argument("--clevr_val_splits", type=str, nargs="+",
                   default=["easy"])

    # --- image ---
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)

    # --- encoder mode ---
    p.add_argument("--encoder_mode", type=str, default="pretrained",
                   choices=["pretrained", "scratch"],
                   help="'pretrained': frozen T5/CLIP, 'scratch': from-scratch")
    p.add_argument("--pretrained_model_name", type=str,
                   default="google-t5/t5-base",
                   help="HF model name (T5 or CLIP)")
    p.add_argument("--pretrained_max_length", type=int, default=256)
    p.add_argument("--freeze_text_encoder", action="store_true", default=True)
    p.add_argument("--unfreeze_text_encoder", dest="freeze_text_encoder",
                   action="store_false")
    p.add_argument("--text_encoder_lr", type=float, default=None,
                   help="Separate LR for text encoder when unfrozen "
                        "(default: 1/10 of main LR)")

    # --- scratch encoder params ---
    p.add_argument("--cond_hidden_size", type=int, default=512)
    p.add_argument("--cond_n_transformer_layers", type=int, default=4)
    p.add_argument("--cond_n_heads", type=int, default=8)
    p.add_argument("--cond_dropout", type=float, default=0.0)

    # --- DiT backbone ---
    p.add_argument("--dit_patch_size", type=int, default=16)
    p.add_argument("--dit_hidden_size", type=int, default=768)
    p.add_argument("--dit_n_heads", type=int, default=12)
    p.add_argument("--dit_n_blocks", type=int, default=12)
    p.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--dit_dropout", type=float, default=0.0)
    p.add_argument("--dit_bottleneck_dim", type=int, default=128)
    p.add_argument("--dit_in_context_len", type=int, default=0)
    p.add_argument("--dit_in_context_start", type=int, default=4)

    # --- flow matching ---
    p.add_argument("--use_flow_matching", action="store_true", default=False)
    p.add_argument("--flow_P_mean", type=float, default=-0.8)
    p.add_argument("--flow_P_std", type=float, default=0.8)
    p.add_argument("--flow_t_eps", type=float, default=0.05)
    p.add_argument("--flow_noise_scale", type=float, default=1.0)
    p.add_argument("--flow_sampling_method", type=str, default="euler",
                   choices=["euler", "heun"])

    # --- diffusion (DDPM) ---
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="scaled_linear")
    p.add_argument("--prediction_type", type=str, default="epsilon")

    # --- training ---
    p.add_argument("--max_train_steps", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--blr", type=float, default=2.5e-5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--lr_schedule", type=str, default="constant",
                   choices=["cosine", "constant"],
                   help="LR schedule after warmup: cosine decay or constant")
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--uncond_drop_prob", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # --- eval ---
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=50000)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--eval_num_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--clevr_eval_every", type=int, default=0)
    p.add_argument("--clevr_eval_samples", type=int, default=30)
    p.add_argument("--clevr_cond_type", type=str, default="json",
                   choices=["json", "text"],
                   help="Condition format: 'json' (structured, converted to text) "
                        "or 'text' (pre-made captions from conditions_text/).")
    p.add_argument("--clevr_val_image_root", type=str, default=None)
    p.add_argument("--clevr_val_condition_dir", type=str, default=None)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────

class CLEVRTextCondDataset(Dataset):
    """Loads CLEVR images with conditions (JSON or text captions).

    Returns text strings (for pretrained encoder) and/or structured tensors
    (for scratch encoder), depending on mode.

    Args:
        cond_type: "json" (structured JSON → converted to text) or
                   "text" (pre-made captions from conditions_text/).
    """

    def __init__(self, image_root, condition_dir, splits, image_size=256,
                 augment=True, mode="pretrained", cond_type="json"):
        self.image_size = image_size
        self.mode = mode
        self.cond_type = cond_type

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(image_size,
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size,
                                  interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

        self.samples = []  # list of (image_path, condition)
        # condition is dict (json mode) or str (text mode)

        if cond_type == "text":
            self._load_text_conditions(image_root, condition_dir, splits)
        else:
            self._load_json_conditions(image_root, condition_dir, splits)

        print(f"CLEVRTextCondDataset: {len(self.samples)} samples "
              f"from splits {splits} (mode={mode}, cond_type={cond_type})")

    def _load_json_conditions(self, image_root, condition_dir, splits):
        for split in splits:
            img_dir = os.path.join(image_root, split)
            cond_split_dir = os.path.join(condition_dir, split)

            if os.path.isdir(cond_split_dir):
                cond_files = sorted([
                    f for f in os.listdir(cond_split_dir) if f.endswith(".json")])
                cond_map = {}
                for cf in cond_files:
                    with open(os.path.join(cond_split_dir, cf)) as fh:
                        c = json.load(fh)
                    img_fn = c.get("image_filename", cf.replace(".json", ".png"))
                    cond_map[img_fn] = c
                if os.path.isdir(img_dir):
                    for img_fn in sorted(os.listdir(img_dir)):
                        if not img_fn.endswith((".png", ".jpg", ".jpeg")):
                            continue
                        if img_fn in cond_map:
                            self.samples.append((
                                os.path.join(img_dir, img_fn), cond_map[img_fn]))
            else:
                combined_path = os.path.join(condition_dir, f"conditions_{split}.json")
                if os.path.isfile(combined_path):
                    with open(combined_path) as fh:
                        all_conds = json.load(fh)
                    if isinstance(all_conds, list):
                        for c in all_conds:
                            img_fn = c.get("image_filename", "")
                            img_path = os.path.join(img_dir, img_fn)
                            if os.path.isfile(img_path):
                                self.samples.append((img_path, c))

    def _load_text_conditions(self, image_root, condition_dir, splits):
        """Load pre-made text captions.

        Supports three formats (auto-detected per caption):
          1. Combined plain:  captions_{split}.json — captions: [str, ...]
          2. Per-file plain:  {split}/CLEVR_*.json — captions: [str, ...]
          3. Styled per-file: {split}/CLEVR_*.json — captions:
             [{family, variant, text, exposed}, ...] + top-level `gt`

        Plain captions become a (img_path, str) entry.
        Styled captions become (img_path, {text, family, variant, exposed,
        gt, image_filename, split}) so eval can route to complex_text.

        On first run with per-file styled inputs, builds and saves a combined
        captions_{split}.json so subsequent runs load instantly.
        """
        for split in splits:
            img_dir = os.path.join(image_root, split)
            combined = os.path.join(condition_dir, f"captions_{split}.json")
            if os.path.isfile(combined):
                with open(combined) as fh:
                    items = json.load(fh)
            else:
                cond_split_dir = os.path.join(condition_dir, split)
                if not os.path.isdir(cond_split_dir):
                    continue
                files = sorted(fn for fn in os.listdir(cond_split_dir)
                               if fn.endswith(".json"))
                print(f"[data] Building combined captions from "
                      f"{len(files)} per-file JSONs ({split})...")
                from collections import defaultdict as _dd
                per_image = _dd(lambda: {"image_filename": "", "split": split,
                                          "captions": [], "gt": None})
                for fn in files:
                    with open(os.path.join(cond_split_dir, fn)) as fh:
                        data = json.load(fh)
                    img_fn = data.get("image_filename", "")
                    entry = per_image[img_fn]
                    entry["image_filename"] = img_fn
                    entry["split"] = data.get("split", split)
                    if data.get("gt") is not None:
                        entry["gt"] = data["gt"]
                    entry["captions"].extend(data.get("captions", []))
                items = list(per_image.values())
                try:
                    with open(combined, "w") as fh:
                        json.dump(items, fh)
                    print(f"[data] Saved combined captions: {combined} "
                          f"({len(items)} images)")
                except OSError as e:
                    print(f"[data] Warning: could not save combined captions: {e}")

            for item in items:
                img_fn = item.get("image_filename", "")
                img_path = os.path.join(img_dir, img_fn)
                if not os.path.isfile(img_path):
                    continue
                sp = item.get("split", split)
                gt = item.get("gt")
                for cap in item.get("captions", []):
                    if isinstance(cap, str):
                        self.samples.append((img_path, cap))
                    elif isinstance(cap, dict):
                        # Styled caption — keep the full record so eval can
                        # score against `exposed`.
                        self.samples.append((img_path, {
                            "text": cap.get("text", ""),
                            "family": cap.get("family"),
                            "variant": cap.get("variant"),
                            "exposed": cap.get("exposed"),
                            "gt": gt,
                            "image_filename": img_fn,
                            "split": sp,
                        }))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cond = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        item = {"image": img}

        if self.cond_type == "text":
            # `cond` is either a caption string (plain) or a styled caption
            # record dict; the model only consumes the text.
            item["text"] = cond["text"] if isinstance(cond, dict) else cond
        elif self.mode == "pretrained":
            from model_text_conditioned import clevr_json_to_text
            item["text"] = clevr_json_to_text(cond)
        else:
            from model_text_conditioned import clevr_json_to_tensors
            ea, em, rd, rm = clevr_json_to_tensors(cond)
            item["entity_attrs"] = ea
            item["entity_mask"] = em
            item["relation_data"] = rd
            item["relation_mask"] = rm

        return item

    def get_condition(self, idx):
        """Return raw condition (dict for json, str for text)."""
        return self.samples[idx][1]


def collate_fn_pretrained(batch):
    """Custom collate: stack tensors, collect text strings as list."""
    images = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"image": images, "text": texts}


def collate_fn_scratch(batch):
    """Standard collate for scratch mode."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "entity_attrs": torch.stack([b["entity_attrs"] for b in batch]),
        "entity_mask": torch.stack([b["entity_mask"] for b in batch]),
        "relation_data": torch.stack([b["relation_data"] for b in batch]),
        "relation_mask": torch.stack([b["relation_mask"] for b in batch]),
    }


# ──────────────────────────────────────────────────────────────────
#  Model builder
# ──────────────────────────────────────────────────────────────────

def build_model(args):
    from model_text_conditioned import TextConditionedDiT
    return TextConditionedDiT(
        image_size=args.image_size,
        in_channels=args.in_channels,
        vae_downsample_factor=1,
        encoder_mode=args.encoder_mode,
        pretrained_model_name=args.pretrained_model_name,
        pretrained_max_length=args.pretrained_max_length,
        freeze_text_encoder=args.freeze_text_encoder,
        cond_hidden_size=args.cond_hidden_size,
        cond_n_transformer_layers=args.cond_n_transformer_layers,
        cond_n_heads=args.cond_n_heads,
        cond_dropout=args.cond_dropout,
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
    )


# ──────────────────────────────────────────────────────────────────
#  LR scheduler
# ──────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=1e-6, schedule="constant"):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────
#  Checkpointing
# ──────────────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizer, step, args, ema=None):
    if not accelerator.is_main_process:
        return
    ckpt_dir = os.path.join(args.output_dir, f"step{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    # Save only trainable params for pretrained mode (skip frozen LM)
    sd = {k: v for k, v in unwrapped.state_dict().items()}
    torch.save(sd, os.path.join(ckpt_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save({"step": step}, os.path.join(ckpt_dir, "meta.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(ckpt_dir, "ema.pt"))
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    accelerator.print(f"Saved checkpoint at step {step}")


def load_checkpoint(accelerator, model, optimizer, args, ema=None):
    resume_dir = args.resume_dir or args.output_dir
    if not os.path.isdir(resume_dir):
        return 0
    # If resume_dir is itself a step* subdir (eval-only on a specific ckpt),
    # use it directly; otherwise pick the latest step subdir under it.
    bn = os.path.basename(resume_dir.rstrip("/"))
    if bn.startswith("step") and os.path.isfile(
            os.path.join(resume_dir, "model.pt")):
        latest = resume_dir
    else:
        ckpt_dirs = sorted([
            d for d in os.listdir(resume_dir)
            if d.startswith("step") and os.path.isdir(os.path.join(resume_dir, d))
        ])
        if not ckpt_dirs:
            return 0
        latest = os.path.join(resume_dir, ckpt_dirs[-1])
    model_path = os.path.join(latest, "model.pt")
    if not os.path.isfile(model_path):
        return 0

    accelerator.print(f"Resuming from {latest}")
    unwrapped = accelerator.unwrap_model(model)
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    unwrapped.load_state_dict(sd, strict=True)

    # In eval-only mode the optimizer is irrelevant and its saved state may
    # have a different number of parameter groups than the freshly-built
    # one (e.g. trained with --unfreeze_text_encoder, evaluated frozen).
    # Skipping the optimizer load avoids that crash.
    skip_optim = bool(getattr(args, "eval_only", False))
    opt_path = os.path.join(latest, "optimizer.pt")
    if not skip_optim and os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=True))
        except (ValueError, RuntimeError) as e:
            accelerator.print(f"[resume] optimizer load skipped: {e}")

    meta_path = os.path.join(latest, "meta.pt")
    step = 0
    if os.path.isfile(meta_path):
        step = torch.load(meta_path, map_location="cpu", weights_only=True)["step"]

    if ema is not None:
        ema_path = os.path.join(latest, "ema.pt")
        if os.path.isfile(ema_path):
            ema.load_state_dict(
                torch.load(ema_path, map_location="cpu", weights_only=True))
    return step


# ──────────────────────────────────────────────────────────────────
#  Tokenize text batch (pretrained mode helper)
# ──────────────────────────────────────────────────────────────────

def tokenize_batch(model, texts, device):
    """Tokenize text strings using the model's pretrained tokenizer."""
    unwrapped = model
    if hasattr(model, 'module'):
        unwrapped = model.module
    return unwrapped.text_encoder.tokenize(texts, device)


# ──────────────────────────────────────────────────────────────────
#  Sampling
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow_ode(model, cond_kwargs, num_steps=50, guidance_scale=1.5,
                    in_channels=3, method="euler", noise_scale=1.0, t_eps=0.05):
    """ODE sampling for flow matching."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B = cond_kwargs.get("_batch_size", 1)
    latent_size = model.latent_size if not hasattr(model, 'module') else model.module.latent_size

    z = noise_scale * torch.randn(B, in_channels, latent_size, latent_size,
                                  device=device, dtype=dtype)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # Remove helper key
    fwd_kwargs = {k: v for k, v in cond_kwargs.items() if k != "_batch_size"}

    def _compute_velocity(z_cur, t_scalar):
        t_batch = t_scalar.expand(B)
        t_expand = t_scalar.view(1, 1, 1, 1)
        if guidance_scale != 1.0:
            x_cond = model(z_cur, t_batch, **fwd_kwargs)
            x_uncond = model(z_cur, t_batch, return_uncond=True)
            v_cond = (x_cond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            v_uncond = (x_uncond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            x_pred = model(z_cur, t_batch, **fwd_kwargs)
            return (x_pred - z_cur) / (1.0 - t_expand).clamp_min(t_eps)

    for i in range(num_steps):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t_cur
        if method == "heun" and i < num_steps - 1:
            v1 = _compute_velocity(z, t_cur)
            v2 = _compute_velocity(z + dt * v1, t_next)
            z = z + dt * 0.5 * (v1 + v2)
        else:
            z = z + dt * _compute_velocity(z, t_cur)
    return z.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model, scheduler, cond_kwargs, num_steps=50,
                guidance_scale=1.5, in_channels=3):
    """DDIM sampling with CFG."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    B = cond_kwargs.get("_batch_size", 1)
    latent_size = model.latent_size if not hasattr(model, 'module') else model.module.latent_size

    fwd_kwargs = {k: v for k, v in cond_kwargs.items() if k != "_batch_size"}

    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(B, in_channels, latent_size, latent_size,
                          device=device, dtype=dtype)
    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        if guidance_scale != 1.0:
            n_c = model(latents, t_batch, **fwd_kwargs)
            n_u = model(latents, t_batch, return_uncond=True)
            noise_pred = n_u + guidance_scale * (n_c - n_u)
        else:
            noise_pred = model(latents, t_batch, **fwd_kwargs)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents.clamp(-1, 1)


# ──────────────────────────────────────────────────────────────────
#  Build condition kwargs for sampling
# ──────────────────────────────────────────────────────────────────

def build_cond_kwargs(batch_items, model, device, mode):
    """Build forward kwargs from dataset items for eval/sampling."""
    B = len(batch_items)
    if mode == "pretrained":
        texts = [b["text"] for b in batch_items]
        text_tokens = tokenize_batch(model, texts, device)
        return {"text_tokens": text_tokens, "_batch_size": B}
    else:
        return {
            "entity_attrs": torch.stack([b["entity_attrs"] for b in batch_items]).to(device),
            "entity_mask": torch.stack([b["entity_mask"] for b in batch_items]).to(device),
            "relation_data": torch.stack([b["relation_data"] for b in batch_items]).to(device),
            "relation_mask": torch.stack([b["relation_mask"] for b in batch_items]).to(device),
            "_batch_size": B,
        }


# ──────────────────────────────────────────────────────────────────
#  Sample generation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, val_dataset, scheduler, args, accelerator, step,
                     ema_model=None):
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device

    n_samples = 8
    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(val_dataset), generator=rng)[:n_samples]

    batch_items = [val_dataset[i] for i in indices]
    images = torch.stack([b["image"] for b in batch_items]).to(device)
    cond_kwargs = build_cond_kwargs(batch_items, eval_model, device, args.encoder_mode)

    guidance_scales = [1.0]
    if args.uncond_drop_prob > 0:
        guidance_scales.insert(0, 0.0)
        if args.guidance_scale != 1.0:
            guidance_scales.append(args.guidance_scale)

    save_dir = os.path.join(args.output_dir, "samples")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    for gs in guidance_scales:
        if args.use_flow_matching:
            samples = sample_flow_ode(
                eval_model, cond_kwargs, num_steps=args.eval_num_steps,
                guidance_scale=gs, in_channels=args.in_channels,
                method=args.flow_sampling_method,
                noise_scale=args.flow_noise_scale, t_eps=args.flow_t_eps)
        else:
            samples = sample_ddim(
                eval_model, scheduler, cond_kwargs,
                num_steps=args.eval_num_steps, guidance_scale=gs,
                in_channels=args.in_channels)

        if accelerator.is_main_process:
            gt_01 = (images * 0.5 + 0.5).clamp(0, 1)
            gen_01 = (samples * 0.5 + 0.5).clamp(0, 1)
            combined = torch.stack([gt_01, gen_01], dim=1).view(
                -1, 3, args.image_size, args.image_size)
            grid = make_grid(combined, nrow=4, padding=2)
            save_image(grid, os.path.join(save_dir,
                       f"step_{step:07d}_cfg{gs:.1f}.png"))
    model.train()


# ──────────────────────────────────────────────────────────────────
#  CLEVR eval
# ──────────────────────────────────────────────────────────────────

def _get_split_from_path(img_path):
    """Extract split name (easy/medium/hard) from image path like .../images/easy/xxx.png."""
    parts = img_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "images" and i + 1 < len(parts):
            return parts[i + 1]
    # fallback: parent directory name
    return os.path.basename(os.path.dirname(img_path))


def _select_eval_indices_balanced(val_dataset, n_per_split):
    """Select split-balanced eval indices (deterministic), like discrete diffusion."""
    import random as _rng
    eval_rng = _rng.Random(42)

    split_to_indices = {}
    for i in range(len(val_dataset)):
        img_path = val_dataset.samples[i][0]
        split = _get_split_from_path(img_path)
        if split not in split_to_indices:
            split_to_indices[split] = []
        split_to_indices[split].append(i)

    selected = []
    splits_for_selected = []
    for split in sorted(split_to_indices.keys()):
        pool = list(split_to_indices[split])
        eval_rng.shuffle(pool)
        for idx in pool[:n_per_split]:
            selected.append(idx)
            splits_for_selected.append(split)

    return selected, splits_for_selected


def _build_eval_result(n, count_correct, entity_found, entity_total,
                       rel_correct, rel_total):
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
    lines = []
    lines.append(f"    Count acc:           {r['count_accuracy']:.1f}% "
                 f"({r['count_correct']}/{r['n_samples']})")
    lines.append(f"    Entity presence acc: {r['entity_presence_accuracy']:.1f}% "
                 f"({r['entity_found']}/{r['entity_total']})")
    lines.append(f"    Relation acc:        {r['rel_accuracy']:.1f}% "
                 f"({r['rel_correct']}/{r['rel_total']})")
    return "\n".join(lines)


def _build_styled_eval_result_local(d):
    """Build a (split, family) cell result from raw accumulators.

    Mirrors the structure used by train_discrete_diffusion_v2.py so the
    saved JSONs and TB scalars are interchangeable across trainers.
    """
    out = {"n_samples": d["n"]}
    if d["count_n"] > 0:
        out["count_accuracy"] = d["count_correct"] / d["count_n"] * 100
        out["count_correct"] = d["count_correct"]
        out["count_n"] = d["count_n"]
    if d["ent_t"] > 0:
        out["entity_inv_accuracy"] = d["ent_f"] / d["ent_t"] * 100
        out["entity_groups_found"] = d["ent_f"]
        out["entity_groups_total"] = d["ent_t"]
    if d["rel_t"] > 0:
        out["rel_accuracy"] = d["rel_c"] / d["rel_t"] * 100
        out["rel_correct"] = d["rel_c"]
        out["rel_total"] = d["rel_t"]
    return out


def _fmt_styled_local(r):
    bits = [f"n={r['n_samples']}"]
    if "count_accuracy" in r:
        bits.append(f"count={r['count_accuracy']:.1f}%"
                    f" ({r['count_correct']}/{r['count_n']})")
    if "entity_inv_accuracy" in r:
        bits.append(f"entity_inv={r['entity_inv_accuracy']:.1f}%"
                    f" ({r['entity_groups_found']}/{r['entity_groups_total']})")
    if "rel_accuracy" in r:
        bits.append(f"rel={r['rel_accuracy']:.1f}%"
                    f" ({r['rel_correct']}/{r['rel_total']})")
    return "  ".join(bits)


@torch.no_grad()
def evaluate_clevr(model, val_dataset, args, accelerator, step,
                   ema_model=None, num_samples=30,
                   clevr_detector=None, clevr_classifier=None):
    """CLEVR eval with per-split (easy/medium/hard) breakdown."""
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device

    # Select balanced indices across splits
    selected_indices, sample_splits = _select_eval_indices_balanced(
        val_dataset, num_samples)
    n_total = len(selected_indices)

    batch_items = [val_dataset[i] for i in selected_indices]
    images = torch.stack([b["image"] for b in batch_items]).to(device)
    cond_kwargs = build_cond_kwargs(batch_items, eval_model, device, args.encoder_mode)

    if args.use_flow_matching:
        samples = sample_flow_ode(
            eval_model, cond_kwargs, num_steps=args.eval_num_steps,
            guidance_scale=args.guidance_scale, in_channels=args.in_channels,
            method=args.flow_sampling_method,
            noise_scale=args.flow_noise_scale, t_eps=args.flow_t_eps)
    else:
        samples = sample_ddim(
            eval_model, None, cond_kwargs,
            num_steps=args.eval_num_steps, guidance_scale=args.guidance_scale,
            in_channels=args.in_channels)

    if accelerator.is_main_process:
        save_dir = os.path.join(args.output_dir, "clevr_eval")
        os.makedirs(save_dir, exist_ok=True)
        gt_01 = (images * 0.5 + 0.5).clamp(0, 1)
        gen_01 = (samples * 0.5 + 0.5).clamp(0, 1)
        combined = torch.stack([gt_01, gen_01], dim=1).view(
            -1, 3, args.image_size, args.image_size)
        grid = make_grid(combined, nrow=min(8, n_total), padding=2)
        save_image(grid, os.path.join(save_dir, f"clevr_eval_step{step:07d}.png"))

        # Save meta JSON so post-hoc annotation knows which condition each
        # cell in the grid belongs to (mirrors what train_discrete_diffusion_v2
        # writes alongside its grid).
        meta_conditions = []
        for i in selected_indices:
            raw_cond = val_dataset.get_condition(i)
            img_fn = os.path.basename(val_dataset.samples[i][0])
            if isinstance(raw_cond, str):
                meta_conditions.append({
                    "text": raw_cond, "image_filename": img_fn,
                    "split": _get_split_from_path(val_dataset.samples[i][0])})
            elif isinstance(raw_cond, dict):
                m = dict(raw_cond)
                m.setdefault("image_filename", img_fn)
                m.setdefault("split", _get_split_from_path(val_dataset.samples[i][0]))
                meta_conditions.append(m)
            else:
                meta_conditions.append({"text": str(raw_cond),
                                         "image_filename": img_fn})
        meta_path = os.path.join(save_dir, f"clevr_eval_step{step:07d}_meta.json")
        with open(meta_path, "w") as fmeta:
            json.dump({"step": step, "n_samples": n_total,
                       "splits": sample_splits,
                       "conditions": meta_conditions}, fmeta, indent=2)

        # Condition eval with per-split breakdown.
        # Auto-detects styled captions (caption dicts carrying `exposed`)
        # and routes them through the family-aware evaluator. Plain text
        # captions stay on the legacy 3-metric path.
        if clevr_detector is not None and clevr_classifier is not None:
            try:
                from eval_clevr_condition import (
                    eval_clevr_conditions, eval_clevr_complex_text,
                    clevr_text_to_condition_json)

                # Inspect first condition to decide route.
                raw_conds = [val_dataset.get_condition(i)
                             for i in selected_indices]
                is_styled = (raw_conds and isinstance(raw_conds[0], dict)
                             and raw_conds[0].get("exposed") is not None)

                if is_styled:
                    eval_result = eval_clevr_complex_text(
                        gen_01, raw_conds, clevr_detector, clevr_classifier)
                    per_sample = eval_result["per_sample"]

                    FAM_LIST = ["C", "E", "R", "C+E", "C+R", "E+R"]
                    # split × family bucket
                    split_fam = {}
                    for local_i, sp in enumerate(sample_splits):
                        fam = per_sample[local_i].get("family", "?")
                        bucket = split_fam.setdefault(sp, {})
                        d = bucket.setdefault(fam, {
                            "n": 0,
                            "count_n": 0, "count_correct": 0,
                            "ent_f": 0, "ent_t": 0,
                            "rel_c": 0, "rel_t": 0,
                        })
                        r = per_sample[local_i]
                        d["n"] += 1
                        if "count_correct" in r:
                            d["count_n"] += 1
                            d["count_correct"] += r["count_correct"]
                        if "entity_groups_total" in r:
                            d["ent_f"] += r["entity_groups_found"]
                            d["ent_t"] += r["entity_groups_total"]
                        if "rel_total" in r:
                            d["rel_c"] += r["rel_correct"]
                            d["rel_t"] += r["rel_total"]

                    log_dict = {}
                    eval_save = {"step": step,
                                 "per_split_family": {},
                                 "overall_family": {}}
                    overall_fam = {}
                    for sp in sorted(split_fam.keys()):
                        sp_block = {}
                        for fam in FAM_LIST:
                            if fam not in split_fam[sp]:
                                continue
                            d = split_fam[sp][fam]
                            fam_result = _build_styled_eval_result_local(d)
                            sp_block[fam] = fam_result
                            accelerator.print(
                                f"[clevr_eval] step={step} split={sp} "
                                f"fam={fam}: {_fmt_styled_local(fam_result)}")
                            pfx = f"clevr_eval/{sp}/{fam}"
                            for key in ("count_accuracy", "entity_inv_accuracy",
                                        "rel_accuracy"):
                                if key in fam_result:
                                    log_dict[f"{pfx}/{key.replace('_accuracy','_acc')}"] = fam_result[key]
                            agg = overall_fam.setdefault(fam, {
                                "n": 0, "count_n": 0, "count_correct": 0,
                                "ent_f": 0, "ent_t": 0, "rel_c": 0, "rel_t": 0,
                            })
                            for k in agg: agg[k] += d[k]
                        if sp_block:
                            eval_save["per_split_family"][sp] = sp_block

                    for fam, agg in overall_fam.items():
                        fam_result = _build_styled_eval_result_local(agg)
                        eval_save["overall_family"][fam] = fam_result
                        accelerator.print(
                            f"[clevr_eval] step={step} overall "
                            f"fam={fam}: {_fmt_styled_local(fam_result)}")
                        pfx = f"clevr_eval/overall/{fam}"
                        for key in ("count_accuracy", "entity_inv_accuracy",
                                    "rel_accuracy"):
                            if key in fam_result:
                                log_dict[f"{pfx}/{key.replace('_accuracy','_acc')}"] = fam_result[key]

                    accelerator.log(log_dict, step=step)
                    eval_path = os.path.join(
                        save_dir, f"clevr_eval_step{step:07d}.json")
                    with open(eval_path, "w") as f:
                        json.dump(eval_save, f, indent=2)

                else:
                    # Legacy plain-text path.
                    cond_jsons = []
                    for raw_cond in raw_conds:
                        if isinstance(raw_cond, str):
                            cond_jsons.append(clevr_text_to_condition_json(raw_cond))
                        elif isinstance(raw_cond, dict) and "text" in raw_cond and "exposed" not in raw_cond:
                            cond_jsons.append(clevr_text_to_condition_json(raw_cond["text"]))
                        else:
                            cond_jsons.append(raw_cond)

                    eval_result = eval_clevr_conditions(
                        gen_01, cond_jsons, clevr_detector, clevr_classifier)

                    per_sample = eval_result["per_sample"]
                    split_counts = {}
                    for local_i, sp in enumerate(sample_splits):
                        if sp not in split_counts:
                            split_counts[sp] = {
                                "n": 0, "count_correct": 0,
                                "entity_found": 0, "entity_total": 0,
                                "rel_correct": 0, "rel_total": 0,
                            }
                        d = split_counts[sp]
                        r = per_sample[local_i]
                        d["n"] += 1
                        if r["count_correct"]:
                            d["count_correct"] += 1
                        d["entity_found"] += r["entity_found"]
                        d["entity_total"] += r["entity_total"]
                        d["rel_correct"] += r["rel_correct"]
                        d["rel_total"] += r["rel_total"]

                    overall = {"n": 0, "count_correct": 0,
                               "entity_found": 0, "entity_total": 0,
                               "rel_correct": 0, "rel_total": 0}
                    all_split_results = {}
                    log_dict = {}

                    for sp in sorted(split_counts.keys()):
                        d = split_counts[sp]
                        sp_result = _build_eval_result(
                            d["n"], d["count_correct"],
                            d["entity_found"], d["entity_total"],
                            d["rel_correct"], d["rel_total"])
                        all_split_results[sp] = sp_result
                        accelerator.print(
                            f"[clevr_eval] step={step} split={sp} "
                            f"({d['n']} samples):")
                        accelerator.print(_format_split_result(sp_result))
                        for k in overall:
                            overall[k] += d[k]
                        log_dict[f"clevr_eval/{sp}/count_acc"] = sp_result["count_accuracy"]
                        log_dict[f"clevr_eval/{sp}/entity_presence_acc"] = sp_result["entity_presence_accuracy"]
                        log_dict[f"clevr_eval/{sp}/rel_acc"] = sp_result["rel_accuracy"]

                    if overall["n"] > 0:
                        overall_result = _build_eval_result(
                            overall["n"], overall["count_correct"],
                            overall["entity_found"], overall["entity_total"],
                            overall["rel_correct"], overall["rel_total"])
                        all_split_results["overall"] = overall_result
                        accelerator.print(
                            f"[clevr_eval] step={step} overall "
                            f"({overall['n']} samples):")
                        accelerator.print(_format_split_result(overall_result))
                        log_dict["clevr_eval/overall/count_acc"] = overall_result["count_accuracy"]
                        log_dict["clevr_eval/overall/entity_presence_acc"] = overall_result["entity_presence_accuracy"]
                        log_dict["clevr_eval/overall/rel_acc"] = overall_result["rel_accuracy"]

                    accelerator.log(log_dict, step=step)
                    eval_path = os.path.join(
                        save_dir, f"clevr_eval_step{step:07d}.json")
                    with open(eval_path, "w") as f:
                        json.dump({"step": step,
                                    "overall": all_split_results.get("overall"),
                                    "per_split": {k: v for k, v in all_split_results.items()
                                                  if k != "overall"}}, f, indent=2)

            except Exception as e:
                accelerator.print(f"[clevr_eval] condition eval failed: {e}")

    model.train()


# ──────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────

def train(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # In eval-only mode we must NOT clobber the original training
        # args.json — sweepers re-run this script with overrides and we want
        # the canonical config to stay readable.
        if not getattr(args, "eval_only", False):
            with open(os.path.join(args.output_dir, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
    accelerator.init_trackers("text_cond_clevr")

    # ── Dataset ──
    val_image_root = args.clevr_val_image_root or args.clevr_image_root
    val_cond_dir = args.clevr_val_condition_dir or args.clevr_condition_dir

    train_ds = CLEVRTextCondDataset(
        args.clevr_image_root, args.clevr_condition_dir,
        args.clevr_train_splits, args.image_size,
        augment=True, mode=args.encoder_mode, cond_type=args.clevr_cond_type)
    val_ds = CLEVRTextCondDataset(
        val_image_root, val_cond_dir,
        args.clevr_val_splits, args.image_size,
        augment=False, mode=args.encoder_mode, cond_type=args.clevr_cond_type)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=accelerator.num_processes,
        rank=accelerator.process_index, shuffle=True,
    ) if accelerator.num_processes > 1 else None

    # text cond_type always uses pretrained collate (text strings)
    use_pretrained_collate = (args.encoder_mode == "pretrained"
                              or args.clevr_cond_type == "text")
    collate = collate_fn_pretrained if use_pretrained_collate else collate_fn_scratch
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate)

    accelerator.print(f"Train: {len(train_ds)}, Val: {len(val_ds)} "
                      f"(mode={args.encoder_mode})")

    # ── CLEVR eval models (detector + classifier) ──
    clevr_detector, clevr_classifier = None, None
    if args.clevr_eval_every > 0:
        try:
            from eval_clevr_condition import load_eval_models
            clevr_detector, clevr_classifier = load_eval_models(
                device=accelerator.device)
            accelerator.print("[clevr] loaded detector + classifier for eval")
        except Exception as e:
            accelerator.print(f"[clevr] WARNING: could not load eval models: {e}")

    # ── Model ──
    model = build_model(args)
    accelerator.print(model.describe())

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Parameters: {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")

    # ── Optimizer ──
    eff_bs = args.batch_size * accelerator.num_processes * args.grad_accum_steps
    lr = args.lr if args.lr is not None else args.blr * eff_bs / 256
    accelerator.print(f"Effective batch size: {eff_bs}, LR: {lr:.2e}")

    # Separate text encoder params (when unfrozen) from DiT params
    te_decay, te_no_decay = [], []
    dit_decay, dit_no_decay = [], []

    is_unfrozen_lm = (args.encoder_mode == "pretrained"
                      and not args.freeze_text_encoder)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_te = name.startswith("text_encoder.encoder.")
        is_nodecay = (param.ndim <= 1 or "bias" in name
                      or "norm" in name or "emb" in name)
        if is_te and is_unfrozen_lm:
            (te_no_decay if is_nodecay else te_decay).append(param)
        else:
            (dit_no_decay if is_nodecay else dit_decay).append(param)

    param_groups = [
        {"params": dit_decay, "weight_decay": args.weight_decay},
        {"params": dit_no_decay, "weight_decay": 0.0},
    ]

    if is_unfrozen_lm and (te_decay or te_no_decay):
        te_lr = args.text_encoder_lr if args.text_encoder_lr else lr * 0.1
        param_groups.extend([
            {"params": te_decay, "weight_decay": args.weight_decay,
             "lr": te_lr},
            {"params": te_no_decay, "weight_decay": 0.0,
             "lr": te_lr},
        ])
        n_te = sum(p.numel() for p in te_decay + te_no_decay)
        accelerator.print(f"Text encoder unfrozen: {n_te/1e6:.1f}M params, "
                          f"LR: {te_lr:.2e}")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # ── Schedulers ──
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type)
    eval_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type)

    # ── Prepare ──
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader)

    # ── EMA ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(model), decay=args.ema_decay)

    # ── Resume ──
    global_step = load_checkpoint(accelerator, model, optimizer, args, ema=ema)

    # ── Eval-only short-circuit ──
    # `--eval_only` runs evaluate_clevr once on the loaded ckpt (use with
    # `--resume_dir <step_dir>`) and exits. Useful for sweeping every saved
    # ckpt with a fixed `--clevr_eval_samples`.
    if getattr(args, "eval_only", False):
        accelerator.print(f"[eval_only] step={global_step} "
                          f"clevr_eval_samples={args.clevr_eval_samples}")
        ema_for_eval = (ema if (ema is not None and args.ema_decay > 0)
                        else None)
        # Bring EMA shadow weights into a model copy if available
        ema_eval_model = None
        if ema_for_eval is not None:
            from copy import deepcopy
            ema_eval_model = deepcopy(accelerator.unwrap_model(model))
            ema.copy_to(ema_eval_model)
        evaluate_clevr(model, val_ds, args, accelerator, global_step,
                       ema_model=ema_eval_model,
                       num_samples=args.clevr_eval_samples,
                       clevr_detector=clevr_detector,
                       clevr_classifier=clevr_classifier)
        accelerator.print("[eval_only] done")
        return

    # ── Train loop ──
    accelerator.print(f"Starting training from step {global_step}")
    model.train()
    epoch = 0

    pbar = tqdm(initial=global_step, total=args.max_train_steps,
                desc="Training", dynamic_ncols=True,
                disable=not accelerator.is_main_process)

    while global_step < args.max_train_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            x0 = batch["image"]

            # Build condition kwargs based on mode
            if args.encoder_mode == "pretrained":
                texts = batch["text"]
                text_tokens = tokenize_batch(
                    accelerator.unwrap_model(model), texts, x0.device)
                cond_kwargs = {"text_tokens": text_tokens}
            else:
                cond_kwargs = {
                    "entity_attrs": batch["entity_attrs"],
                    "entity_mask": batch["entity_mask"],
                    "relation_data": batch["relation_data"],
                    "relation_mask": batch["relation_mask"],
                }

            cur_lr = get_lr(global_step, args.warmup_steps,
                            args.max_train_steps, lr, schedule=args.lr_schedule)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            with accelerator.accumulate(model):
                if args.use_flow_matching:
                    z_t = (torch.randn(x0.shape[0], device=x0.device)
                           * args.flow_P_std + args.flow_P_mean)
                    t_flow = torch.sigmoid(z_t)
                    t_expand = t_flow.view(-1, 1, 1, 1)
                    e = torch.randn_like(x0) * args.flow_noise_scale
                    noisy = t_expand * x0 + (1 - t_expand) * e
                    v_target = (x0 - noisy) / (1 - t_expand).clamp_min(args.flow_t_eps)

                    with accelerator.autocast():
                        x_pred = model(noisy, t_flow, **cond_kwargs)
                        v_pred = (x_pred - noisy) / (1 - t_expand).clamp_min(args.flow_t_eps)
                        loss = F.mse_loss(v_pred, v_target)
                else:
                    noise = torch.randn_like(x0)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (x0.shape[0],), device=x0.device, dtype=torch.long)
                    noisy = noise_scheduler.add_noise(x0, noise, timesteps)
                    with accelerator.autocast():
                        pred = model(noisy, timesteps, **cond_kwargs)
                        if args.prediction_type == "epsilon":
                            target = noise
                        elif args.prediction_type == "sample":
                            target = x0
                        else:
                            target = noise_scheduler.get_velocity(x0, noise, timesteps)
                        loss = F.mse_loss(pred, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            if ema is not None:
                ema.update(accelerator.unwrap_model(model))

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.1e}",
                             refresh=False)

            if global_step % args.log_every == 0:
                accelerator.log({"loss": loss.item(), "lr": cur_lr},
                                step=global_step)

            if global_step % args.sample_every == 0:
                ema_eval = ema.shadow if ema is not None else None
                generate_samples(model, val_ds, eval_scheduler, args,
                                 accelerator, global_step, ema_model=ema_eval)

            if global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, optimizer, global_step,
                                args, ema=ema)

            if (args.clevr_eval_every > 0
                    and global_step % args.clevr_eval_every == 0
                    and global_step > 0):
                ema_eval = ema.shadow if ema is not None else None
                evaluate_clevr(model, val_ds, args, accelerator, global_step,
                               ema_model=ema_eval,
                               num_samples=args.clevr_eval_samples,
                               clevr_detector=clevr_detector,
                               clevr_classifier=clevr_classifier)

        epoch += 1

    pbar.close()
    save_checkpoint(accelerator, model, optimizer, global_step, args, ema=ema)
    accelerator.print("Training complete.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
