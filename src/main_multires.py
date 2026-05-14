"""
Multi-Resolution Conditional Diffusion — Training Script
=========================================================

Patchify → per-patch CNN → merge hierarchy → learned upsample → UNet injection.

Usage:
  # Single GPU
  python src/main_multires.py --dataset_root /path/to/imagenet/ILSVRC/Data/CLS-LOC

  # Multi-GPU with accelerate
  accelerate launch --multi_gpu src/main_multires.py --dataset_root ...

  # FID evaluation only
  accelerate launch --multi_gpu src/main_multires.py --eval_only --resume_dir runs/exp1
"""

import argparse
import copy
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import DDIMScheduler, DDPMScheduler, AutoencoderKL
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# ──────────────────────────────────────────────────────────────────
#  EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────

class EMA:
    """Maintains an exponential moving average of model parameters."""

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
    p.add_argument("--output_dir", type=str, default="runs/multires")
    p.add_argument("--resume_dir", type=str, default=None)
    p.add_argument("--dataset_root", type=str,
                   default="/workspace/NAS/project/imagenet/ILSVRC/Data/CLS-LOC")
    p.add_argument("--fid_stats", type=str, default=None,
                   help="Pre-computed .npz for FID (optional)")
    p.add_argument("--train_dir", type=str, default=None,
                   help="Direct path to training images dir (overrides dataset_root/train)")
    p.add_argument("--val_dir", type=str, default=None,
                   help="Separate validation images dir for sampling grid "
                        "(auto-detected as dataset_root/val if not set)")
    p.add_argument("--fid_real_dir", type=str, default=None,
                   help="Real image dir for FID comparison (default: dataset val dir)")

    # --- VAE caching ---
    p.add_argument("--cache_latents", action="store_true", default=False,
                   help="Pre-encode all images with VAE and cache to disk")
    p.add_argument("--latent_cache_dir", type=str, default=None,
                   help="Directory for cached latents (default: output_dir/latent_cache)")

    # --- Local-SSD image cache (avoid NAS I/O every step) ---
    p.add_argument("--cache_to_local_disk", action="store_true", default=False,
                   help="Preprocess all images once (resize+center_crop) into a uint8 "
                        "memmap file on local SSD. All ranks mmap-share via OS page cache.")
    p.add_argument("--local_cache_dir", type=str, default=None,
                   help="Base dir for local memmap cache (default: /workspace/cache). "
                        "Per-dataset subdir is auto-named from the source path + image_size.")

    # --- image / model ---
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3,
                   help="UNet input ch (3=pixel, 4=VAE latent)")
    p.add_argument("--cond_in_channels", type=int, default=3)
    p.add_argument("--vae_downsample_factor", type=int, default=1,
                   help="1=no VAE (pixel space), 8=8x VAE, 16=16x VAE")
    p.add_argument("--vae_pretrained", type=str, default=None,
                   help="HF model ID for VAE (e.g. 'stabilityai/sd-vae-ft-ema')")

    # --- encoder ---
    p.add_argument("--min_patch_size", type=int, default=32)
    p.add_argument("--num_levels", type=int, default=None)
    p.add_argument("--feat_channels", type=int, default=256)
    p.add_argument("--depth_per_level", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--cnn_base_channels", type=int, default=64)
    p.add_argument("--mae_mask_ratio", type=float, default=0.0,
                   help="MAE-style patch masking ratio (training only, 0=disabled)")

    # --- ViT encoder ---
    p.add_argument("--encoder_type", type=str, default="cnn",
                   choices=["cnn", "vit", "swin", "vit_global"],
                   help="Encoder backend: 'cnn' (PatchCNN+merge), "
                        "'vit' (shared CellViT per cell), "
                        "'swin' (Swin Transformer), or "
                        "'vit_global' (single ViT forward → avg pool per level)")
    p.add_argument("--vit_patch_size", type=int, default=4,
                   help="ViT sub-patch size for finest level")
    p.add_argument("--vit_depth", type=int, default=4,
                   help="ViT transformer depth")
    p.add_argument("--vit_num_heads", type=int, default=4,
                   help="ViT attention heads")
    p.add_argument("--vit_mlp_ratio", type=float, default=4.0,
                   help="ViT MLP expansion ratio")
    p.add_argument("--vit_use_cnn_stem", action="store_true", default=True,
                   help="Use CNN stem before ViT patch projection (recommended)")
    p.add_argument("--vit_no_cnn_stem", action="store_true", default=False,
                   help="Disable CNN stem")
    p.add_argument("--vit_cnn_stem_reduction", type=int, default=4,
                   help="CNN stem spatial reduction factor")

    # --- CLIP init for vit_global ---
    p.add_argument("--vit_init_clip", action="store_true", default=False,
                   help="Initialize vit_global encoder from pretrained CLIP "
                        "(requires --vit_no_cnn_stem, --feat_channels 768, "
                        "--vit_patch_size 16, --vit_depth 12, --vit_num_heads 12)")
    p.add_argument("--clip_model_name", type=str,
                   default="openai/clip-vit-base-patch16",
                   help="HuggingFace CLIP model to load weights from")

    p.add_argument("--encoder_internal_dim", type=int, default=None,
                   help="Encoder internal dim (ViT hidden dim). If set, encoder runs at this dim "
                        "and projects to feat_channels at output. Default: same as feat_channels.")

    # --- Swin encoder ---
    p.add_argument("--swin_patch_size", type=int, default=16,
                   help="Swin initial patch embedding size (image_size / swin_patch_size = initial tokens per side)")
    p.add_argument("--swin_embed_dim", type=int, default=96,
                   help="Swin base embedding dimension (doubles per stage)")
    p.add_argument("--swin_depths", type=int, nargs="+", default=None,
                   help="Swin blocks per stage (e.g. 2 2 6 2). Must match num_levels.")
    p.add_argument("--swin_num_heads", type=int, nargs="+", default=None,
                   help="Swin attention heads per stage (e.g. 3 6 12 24)")
    p.add_argument("--swin_window_size", type=int, default=4,
                   help="Swin window size for local attention")
    p.add_argument("--swin_mlp_ratio", type=float, default=4.0,
                   help="Swin MLP expansion ratio")

    # --- discretization ---
    p.add_argument("--use_fsq", action="store_true", default=False)
    p.add_argument("--fsq_levels", type=int, nargs="+", default=None,
                   help="FSQ quantization levels (e.g. 8 6 5)")
    p.add_argument("--fsq_drop_quant_p", type=float, default=0.0)
    p.add_argument("--fsq_corrupt_tokens_p", type=float, default=0.0)
    p.add_argument("--use_vq", action="store_true", default=False)
    p.add_argument("--vq_codebook_size", type=int, default=512)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--vq_loss_weight", type=float, default=1.0,
                   help="Weight for VQ auxiliary loss")

    # --- UNet ---
    p.add_argument("--unet_config", type=str, default=None,
                   help="Path to JSON with UNet config, or inline JSON string")
    p.add_argument("--block_out_channels", type=int, nargs="+",
                   default=[128, 128, 256, 256, 512, 512],
                   help="UNet block output channels (used if --unet_config not set)")
    p.add_argument("--layers_per_block", type=int, default=2)
    p.add_argument("--attn_resolutions", type=int, nargs="*", default=None,
                   help="Spatial resolutions where down/up blocks use self-attention "
                        "(e.g. --attn_resolutions 32 16). Default: no attention in down/up blocks.")
    p.add_argument("--no_mid_attn", action="store_true", default=False,
                   help="Disable self-attention in mid block (use pure conv mid block)")

    # --- backbone selection ---
    p.add_argument("--backbone", type=str, default="unet",
                   choices=["unet", "dit", "baseline_1d"],
                   help="Denoising backbone: 'unet', 'dit', or 'baseline_1d' (Semanticist-style)")

    # --- DiT (only used when --backbone dit) ---
    p.add_argument("--dit_patch_size", type=int, default=2,
                   help="DiT patch size for latent patchification")
    p.add_argument("--dit_hidden_size", type=int, default=768)
    p.add_argument("--dit_n_heads", type=int, default=12)
    p.add_argument("--dit_n_blocks", type=int, default=12)
    p.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    p.add_argument("--dit_dropout", type=float, default=0.0)
    p.add_argument("--dit_bottleneck_dim", type=int, default=128,
                   help="BottleneckPatchEmbed bottleneck dim (JiT-B/L=128, H=256)")
    p.add_argument("--dit_in_context_len", type=int, default=0,
                   help="Number of in-context learnable tokens (JiT=32, 0=disabled). "
                        "Forced to 0 when --dit_attn_mode=cross (cross-attn DiT "
                        "blocks have no in-context path).")
    p.add_argument("--dit_in_context_start", type=int, default=4,
                   help="Layer index to prepend in-context tokens (JiT-B=4, L=8, H=10)")
    p.add_argument("--dit_attn_mode", type=str, default="self_concat",
                   choices=["self_concat", "cross"],
                   help="Slot conditioning attention mode for baseline_1d. "
                        "'self_concat' = Semanticist-style (cond concatenated, "
                        "per-slot pos embed, in-context tokens). "
                        "'cross' = SlotDiffusion/DINOSAUR-style (image cross-"
                        "attends to slots, no per-slot pos embed, no in-context "
                        "tokens). Use 'cross' for permutation-equivariant Slot "
                        "Attention to avoid position-binding collapse.")

    # --- baseline_1d (Semanticist-style, only used when --backbone baseline_1d) ---
    p.add_argument("--num_slots", type=int, default=256,
                   help="Number of 1D condition tokens (Semanticist-style)")
    p.add_argument("--slot_dim", type=int, default=16,
                   help="Per-slot dimension before projection")
    p.add_argument("--enc_embed_dim", type=int, default=768,
                   help="Semanticist ViT encoder hidden dim")
    p.add_argument("--enc_depth", type=int, default=12,
                   help="Semanticist ViT encoder depth")
    p.add_argument("--enc_num_heads", type=int, default=12,
                   help="Semanticist ViT encoder attention heads")
    p.add_argument("--enc_drop_path_rate", type=float, default=0.1,
                   help="Semanticist ViT encoder drop path rate")
    p.add_argument("--is_causal", action="store_true", default=True,
                   help="Use causal attention mask on slots")
    p.add_argument("--no_causal", dest="is_causal", action="store_false")
    p.add_argument("--enable_nest", action="store_true", default=True,
                   help="Enable nested (progressive) token dropping")
    p.add_argument("--no_nest", dest="enable_nest", action="store_false")
    p.add_argument("--enable_nest_after_steps", type=int, default=-1,
                   help="Enable nested drop after N steps (-1 = from start)")
    p.add_argument("--eval_slot_configs", type=int, nargs="+",
                   default=[1, 4, 16, 64, 256],
                   help="Slot counts to evaluate during sampling (baseline_1d only)")

    # --- injection (UNet only) ---
    p.add_argument("--upsample_factor", type=int, default=None,
                   help="Auto-computed if not set")

    # --- level drop ---
    p.add_argument("--level_drop", action="store_true", default=True,
                   help="Random level drop during training")
    p.add_argument("--no_level_drop", dest="level_drop", action="store_false")
    p.add_argument("--min_keep_levels", type=int, default=1,
                   help="Min levels to keep (1=global only)")
    p.add_argument("--level_drop_after_steps", type=int, default=-1,
                   help="Enable level drop after this many steps (-1=from start)")
    p.add_argument("--eval_num_active_levels", type=int, default=None,
                   help="Override active levels at eval (None=all)")
    p.add_argument("--level_sizes", type=int, nargs="+", default=None,
                   help="Custom encoder level sizes (e.g. 9 3 1 for sudoku). "
                        "Overrides min_patch_size / num_levels auto-computation.")

    # --- diffusion ---
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="scaled_linear")
    p.add_argument("--prediction_type", type=str, default="epsilon")

    # --- flow matching (JiT-style V-loss) ---
    p.add_argument("--use_flow_matching", action="store_true", default=False,
                   help="Use flow matching with V-loss (JiT-style) instead of DDPM")
    p.add_argument("--flow_P_mean", type=float, default=-0.8,
                   help="Logit-normal timestep sampling: mean")
    p.add_argument("--flow_P_std", type=float, default=0.8,
                   help="Logit-normal timestep sampling: std")
    p.add_argument("--flow_t_eps", type=float, default=0.05,
                   help="Minimum (1-t) clamp to avoid div-by-zero")
    p.add_argument("--flow_noise_scale", type=float, default=1.0,
                   help="Scale for initial noise in flow matching")
    p.add_argument("--flow_sampling_method", type=str, default="euler",
                   choices=["euler", "heun"],
                   help="ODE solver for flow matching sampling")

    # --- training ---
    p.add_argument("--max_train_steps", type=int, default=500000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--blr", type=float, default=2.5e-5,
                   help="Base LR (scaled by effective_bs/256). Ignored if --lr is set.")
    p.add_argument("--lr", type=float, default=None,
                   help="Absolute LR (overrides --blr if set)")
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
    p.add_argument("--cond_token_drop_prob", type=float, default=0.0,
                   help="Per-token random drop max ratio (train-only). "
                        "Applied only to the finest kept level per sample "
                        "after level_drop.")
    p.add_argument("--cond_token_drop_all_levels", action="store_true",
                   default=False,
                   help="If set, cond_token_drop_prob is applied to every "
                        "kept level (not just the finest). Useful when "
                        "level_drop is disabled and you want random drop "
                        "across all multi-res levels.")
    p.add_argument("--cond_token_drop_linear", action="store_true",
                   default=False,
                   help="Sample drop ratio from linear distribution f(p)=2p "
                        "(biased toward higher masking) instead of uniform.")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay rate (0=disabled)")
    p.add_argument("--cond_use_latent", action="store_true", default=False,
                   help="Feed VAE latent to encoder instead of raw image")
    p.add_argument("--seed", type=int, default=42)

    # --- eval ---
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--fid_every", type=int, default=50000)
    p.add_argument("--eval_num_steps", type=int, default=50,
                   help="DDIM sampling steps")
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--fid_num_samples", type=int, default=50000)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_clevr_only", action="store_true",
                   help="Run only the CLEVR detection+attribute eval (no FID, no training)")
    p.add_argument("--clevr_eval_every", type=int, default=0,
                   help="Run CLEVR detection+attribute eval every N steps (0=disabled)")
    p.add_argument("--clevr_eval_samples", type=int, default=30,
                   help="Number of val samples for CLEVR eval")
    p.add_argument("--clevr_eval_n_annotated_random", type=int, default=8,
                   help="Number of random samples to render with bboxes per CLEVR eval (0=off)")
    p.add_argument("--clevr_eval_n_annotated_worst", type=int, default=4,
                   help="Number of worst-scoring samples to render with bboxes per CLEVR eval (0=off)")
    p.add_argument("--clevr_eval_annot_thresh", type=int, default=10,
                   help="Distance threshold (px) used for matched/missed bbox coloring")
    p.add_argument("--num_workers", type=int, default=8)

    args = p.parse_args()

    # vit_global compatibility checks
    if args.encoder_type == 'vit_global' and args.mae_mask_ratio > 0:
        p.error("--encoder_type vit_global does not support --mae_mask_ratio > 0 "
                "(all spatial tokens are needed for avg pooling)")

    # CLIP init compatibility checks
    if args.vit_init_clip:
        if args.encoder_type != 'vit_global':
            p.error("--vit_init_clip requires --encoder_type vit_global")
        if args.vit_use_cnn_stem and not args.vit_no_cnn_stem:
            p.error("--vit_init_clip requires --vit_no_cnn_stem "
                    "(CLIP has no CNN stem)")
        # CLIP init loads into the transformer's INTERNAL dim. When
        # --encoder_internal_dim is set, that's the ViT hidden; else
        # --feat_channels. Check against the right one.
        internal_dim = (args.encoder_internal_dim
                        if args.encoder_internal_dim is not None
                        else args.feat_channels)
        expected_b16 = (
            internal_dim == 768 and args.vit_depth == 12
            and args.vit_num_heads == 12 and args.vit_patch_size == 16
        )
        if 'base-patch16' in args.clip_model_name and not expected_b16:
            p.error(
                f"--clip_model_name {args.clip_model_name} (ViT-B/16) requires "
                "internal dim 768 (set --encoder_internal_dim 768 or "
                "--feat_channels 768), --vit_depth 12 --vit_num_heads 12 "
                "--vit_patch_size 16"
            )

    return args


# ──────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────

def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


def build_datasets(args):
    train_dir = args.train_dir or os.path.join(args.dataset_root, "train")
    train_ds = datasets.ImageFolder(train_dir, transform=build_train_transform(args.image_size))
    val_ds = datasets.ImageFolder(train_dir, transform=build_val_transform(args.image_size))
    return train_ds, val_ds


def _resolve_local_cache_subdir(args, root_dir):
    base = args.local_cache_dir or "/workspace/cache"
    # Name subdir after the leaf of root_dir + image_size (so train/val don't collide).
    tag = os.path.basename(os.path.normpath(root_dir)) or "root"
    return os.path.join(base, f"{tag}_{args.image_size}")


def build_memmap_image_cache(root_dir, cache_dir, image_size, accelerator, name):
    """Build (on main proc) or reuse a uint8 memmap cache of an ImageFolder.

    Layout:
      {cache_dir}/{name}.bin        — (N, 3, H, H) uint8 raw bytes
      {cache_dir}/{name}.meta.json  — num_images, image_size, labels, classes, samples
    Other ranks wait on accelerator.wait_for_everyone() until main finishes.
    Returns a dict with the meta fields + bin_path.
    """
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    bin_path = os.path.join(cache_dir, f"{name}.bin")
    meta_path = os.path.join(cache_dir, f"{name}.meta.json")

    if accelerator.is_main_process:
        need_build = True
        if os.path.isfile(bin_path) and os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                expected_bytes = meta["num_images"] * 3 * image_size * image_size
                if (meta.get("image_size") == image_size
                        and os.path.getsize(bin_path) == expected_bytes
                        and meta.get("root_dir") == os.path.abspath(root_dir)):
                    need_build = False
                    accelerator.print(f"[memmap-cache] Reuse {name}: {bin_path} "
                                      f"({meta['num_images']} imgs)")
            except Exception:
                need_build = True

        if need_build:
            accelerator.print(f"[memmap-cache] Building {name}: {bin_path}")
            base = datasets.ImageFolder(root_dir)
            num_images = len(base.samples)
            arr = np.memmap(bin_path, dtype=np.uint8, mode='w+',
                            shape=(num_images, 3, image_size, image_size))
            resize = transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC)
            crop = transforms.CenterCrop(image_size)
            labels, samples = [], []
            for i, (path, label) in enumerate(
                    tqdm(base.samples, desc=f"Preload {name}")):
                img = Image.open(path).convert("RGB")
                img = crop(resize(img))
                hwc = np.asarray(img, dtype=np.uint8)
                arr[i] = np.transpose(hwc, (2, 0, 1))
                labels.append(int(label))
                samples.append([path, int(label)])
                if (i + 1) % 5000 == 0:
                    arr.flush()
            arr.flush()
            del arr

            meta = {
                "num_images": num_images,
                "image_size": image_size,
                "channels": 3,
                "labels": labels,
                "classes": list(base.classes),
                "samples": samples,
                "root_dir": os.path.abspath(root_dir),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            accelerator.print(
                f"[memmap-cache] Built {name}: {num_images} imgs, "
                f"{os.path.getsize(bin_path) / (1 << 30):.2f} GiB")

    accelerator.wait_for_everyone()

    with open(meta_path) as f:
        meta = json.load(f)
    meta["bin_path"] = bin_path
    return meta


class MemmapImageDataset(torch.utils.data.Dataset):
    """Serves images from a uint8 memmap file built by build_memmap_image_cache.

    __getitem__ returns (image, label) with image normalized to [-1, 1] float CHW
    — same contract as ImageFolder + build_{train,val}_transform. When train=True,
    a 50% horizontal flip is applied (matching RandomHorizontalFlip).
    Exposes .samples / .classes / .targets for ImageFolder-compatible downstream code.
    """
    def __init__(self, meta, train=True):
        self.bin_path = meta["bin_path"]
        self.num_images = int(meta["num_images"])
        self.image_size = int(meta["image_size"])
        self.channels = int(meta["channels"])
        self.labels = [int(x) for x in meta["labels"]]
        self.classes = list(meta["classes"])
        self.samples = [(p, int(l)) for p, l in meta["samples"]]
        self.targets = list(self.labels)
        self.train = train
        self._arr = None

    def _get_arr(self):
        if self._arr is None:
            self._arr = np.memmap(
                self.bin_path, dtype=np.uint8, mode='r',
                shape=(self.num_images, self.channels,
                       self.image_size, self.image_size),
            )
        return self._arr

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        arr = self._get_arr()
        img = torch.from_numpy(np.array(arr[idx], dtype=np.uint8))
        if self.train and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
        img = img.to(torch.float32).div_(255.0).sub_(0.5).mul_(2.0)
        return img, self.labels[idx]


def consolidate_cache(cache_dir, latent_only=False):
    """Merge individual .pt files into a single stacked tensor file.

    Creates latent_cache/consolidated.pt (or consolidated_latent_only.pt)
    for fast single-file loading. Only runs once; subsequent calls skip.
    """
    suffix = "_latent_only" if latent_only else ""
    consolidated_path = os.path.join(cache_dir, f"consolidated{suffix}.pt")
    if os.path.isfile(consolidated_path):
        return consolidated_path

    files = sorted([f for f in os.listdir(cache_dir)
                    if f.endswith('.pt') and not f.startswith('consolidated')])
    print(f"Consolidating {len(files)} cache files → {consolidated_path} ...")

    latents, latents_flip = [], []
    images, images_flip = ([], []) if not latent_only else (None, None)
    for f in tqdm(files, desc="Consolidate"):
        data = torch.load(os.path.join(cache_dir, f),
                          map_location='cpu', weights_only=True)
        latents.append(data['latent'])
        latents_flip.append(data['latent_flip'])
        if not latent_only:
            images.append(data['image'])
            images_flip.append(data['image_flip'])

    out = {
        'latent': torch.stack(latents),
        'latent_flip': torch.stack(latents_flip),
    }
    if not latent_only:
        out['image'] = torch.stack(images)
        out['image_flip'] = torch.stack(images_flip)

    torch.save(out, consolidated_path)
    print(f"Consolidated: {consolidated_path}")
    return consolidated_path


class CachedLatentDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-cached VAE latents (+ optionally raw images).

    Supports two loading modes:
    - consolidated (default): loads a single stacked tensor file (fast)
    - individual .pt files: fallback per-sample loading from disk
    """

    def __init__(self, cache_dir, flip_aug=True, latent_only=False, preload=False,
                 rank=0, world_size=1):
        self.flip_aug = flip_aug
        self.latent_only = latent_only
        self.data = None

        if preload:
            # Load consolidated file, keep only this rank's shard
            consolidated_path = consolidate_cache(cache_dir, latent_only=latent_only)
            print(f"[Rank {rank}] Loading consolidated cache (shard {rank}/{world_size}) ...")
            full = torch.load(consolidated_path, map_location='cpu',
                              weights_only=True)
            self.data = {
                'latent': full['latent'][rank::world_size].clone(),
                'latent_flip': full['latent_flip'][rank::world_size].clone(),
            }
            if not latent_only:
                self.data['image'] = full['image'][rank::world_size].clone()
                self.data['image_flip'] = full['image_flip'][rank::world_size].clone()
            del full
            self._len = self.data['latent'].shape[0]
            print(f"[Rank {rank}] Loaded {self._len} samples into RAM.")
        else:
            self.cache_dir = cache_dir
            self.files = sorted([f for f in os.listdir(cache_dir)
                                 if f.endswith('.pt') and not f.startswith('consolidated')])
            self._len = len(self.files)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.data is not None:
            flip = self.flip_aug and torch.rand(1).item() < 0.5
            latent = self.data['latent_flip' if flip else 'latent'][idx]
            if self.latent_only:
                return latent
            image = self.data['image_flip' if flip else 'image'][idx]
            return image, latent

        data = torch.load(os.path.join(self.cache_dir, self.files[idx]),
                          map_location='cpu', weights_only=True)
        if self.flip_aug and torch.rand(1).item() < 0.5:
            latent = data['latent_flip']
            image = None if self.latent_only else data.get('image_flip')
        else:
            latent = data['latent']
            image = None if self.latent_only else data.get('image')
        if self.latent_only:
            return latent
        return image, latent


@torch.no_grad()
def cache_vae_latents(args, accelerator, latent_only=False):
    """Pre-encode all training images with VAE and save to disk."""
    cache_dir = args.latent_cache_dir or os.path.join(args.output_dir, "latent_cache")

    # Check if already cached
    if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
        n_cached = len([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
        accelerator.print(f"Found {n_cached} cached latents in {cache_dir}, skipping.")
        return cache_dir

    accelerator.print(f"Caching VAE latents to {cache_dir} ...")
    os.makedirs(cache_dir, exist_ok=True)

    vae = build_vae(args, accelerator.device)
    assert vae is not None, "VAE required for latent caching"

    # Deterministic transform (no random crop/flip — we apply flip at load time)
    ds = datasets.ImageFolder(
        os.path.join(args.dataset_root, "train"),
        transform=build_val_transform(args.image_size),
    )

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    per_gpu = math.ceil(len(ds) / world_size)
    my_start = rank * per_gpu
    my_end = min(my_start + per_gpu, len(ds))

    loader = DataLoader(
        torch.utils.data.Subset(ds, range(my_start, my_end)),
        batch_size=64, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    idx = my_start
    for images, _ in tqdm(loader, desc=f"Caching (rank {rank})",
                          disable=not accelerator.is_main_process):
        images = images.to(accelerator.device)
        latents = vae_encode(vae, images)

        images_flip = images.flip(-1)
        latents_flip = vae_encode(vae, images_flip)

        for i in range(images.shape[0]):
            entry = {
                'latent': latents[i].cpu().half(),
                'latent_flip': latents_flip[i].cpu().half(),
            }
            if not latent_only:
                entry['image'] = images[i].cpu().half()
                entry['image_flip'] = images_flip[i].cpu().half()
            torch.save(entry, os.path.join(cache_dir, f"{idx:07d}.pt"))
            idx += 1

    accelerator.wait_for_everyone()
    n_total = len([f for f in os.listdir(cache_dir) if f.endswith('.pt')])
    accelerator.print(f"Cached {n_total} latents to {cache_dir}")

    del vae
    torch.cuda.empty_cache()
    return cache_dir


# ──────────────────────────────────────────────────────────────────
#  Model builder
# ──────────────────────────────────────────────────────────────────

def build_model(args):
    # ── Baseline 1D (Semanticist-style) ──
    if args.backbone == "baseline_1d":
        from model_baseline_1d import Baseline1DConditionalDiT
        return Baseline1DConditionalDiT(
            image_size=args.image_size,
            in_channels=args.in_channels,
            cond_in_channels=args.cond_in_channels,
            vae_downsample_factor=args.vae_downsample_factor,
            num_slots=args.num_slots,
            slot_dim=args.slot_dim,
            enc_embed_dim=args.enc_embed_dim,
            enc_depth=args.enc_depth,
            enc_num_heads=args.enc_num_heads,
            enc_drop_path_rate=args.enc_drop_path_rate,
            is_causal=args.is_causal,
            enable_nest=args.enable_nest,
            enable_nest_after_steps=args.enable_nest_after_steps,
            dit_patch_size=args.dit_patch_size,
            dit_hidden_size=args.dit_hidden_size,
            dit_n_heads=args.dit_n_heads,
            dit_n_blocks=args.dit_n_blocks,
            dit_mlp_ratio=args.dit_mlp_ratio,
            dit_dropout=args.dit_dropout,
            dit_bottleneck_dim=args.dit_bottleneck_dim,
            dit_in_context_len=args.dit_in_context_len,
            dit_in_context_start=args.dit_in_context_start,
            dit_attn_mode=args.dit_attn_mode,
            uncond_drop_prob=args.uncond_drop_prob,
            use_fsq=args.use_fsq,
            fsq_levels=args.fsq_levels,
            fsq_drop_quant_p=args.fsq_drop_quant_p,
            fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
            use_vq=args.use_vq,
            vq_codebook_size=args.vq_codebook_size,
            vq_beta=args.vq_beta,
        )

    # ── DiT backbone ──
    if args.backbone == "dit":
        from model_multires import MultiResConditionalDiT
        return MultiResConditionalDiT(
            image_size=args.image_size,
            in_channels=args.in_channels,
            cond_in_channels=args.cond_in_channels,
            vae_downsample_factor=args.vae_downsample_factor,
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
            cond_use_latent=args.cond_use_latent,
            mae_mask_ratio=args.mae_mask_ratio,
            encoder_type=args.encoder_type,
            vit_patch_size=args.vit_patch_size,
            vit_depth=args.vit_depth,
            vit_num_heads=args.vit_num_heads,
            vit_mlp_ratio=args.vit_mlp_ratio,
            vit_use_cnn_stem=args.vit_use_cnn_stem and not args.vit_no_cnn_stem,
            vit_cnn_stem_reduction=args.vit_cnn_stem_reduction,
            encoder_internal_dim=args.encoder_internal_dim,
            swin_patch_size=args.swin_patch_size,
            swin_embed_dim=args.swin_embed_dim,
            swin_depths=args.swin_depths,
            swin_num_heads=args.swin_num_heads,
            swin_window_size=args.swin_window_size,
            swin_mlp_ratio=args.swin_mlp_ratio,
            vit_init_clip=args.vit_init_clip,
            clip_model_name=args.clip_model_name,
            use_fsq=args.use_fsq,
            fsq_levels=args.fsq_levels,
            fsq_drop_quant_p=args.fsq_drop_quant_p,
            fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
            use_vq=args.use_vq,
            vq_codebook_size=args.vq_codebook_size,
            vq_beta=args.vq_beta,
            level_sizes=args.level_sizes,
            cond_token_drop_prob=args.cond_token_drop_prob,
            cond_token_drop_all_levels=args.cond_token_drop_all_levels,
            cond_token_drop_linear=args.cond_token_drop_linear,
        )

    # ── UNet backbone (original) ──
    from model_multires import MultiResConditionalUNet

    if args.unet_config is not None:
        if os.path.isfile(args.unet_config):
            with open(args.unet_config) as f:
                unet_config = json.load(f)
        else:
            unet_config = json.loads(args.unet_config)
    else:
        # Determine block types based on --attn_resolutions
        n_blocks = len(args.block_out_channels)
        attn_set = set(args.attn_resolutions) if args.attn_resolutions else set()
        latent_size = args.image_size // args.vae_downsample_factor

        down_types, up_types = [], []
        res = latent_size
        for i in range(n_blocks):
            # Attention operates at the block's input resolution
            if res in attn_set:
                down_types.append("AttnDownBlock2D")
            else:
                down_types.append("DownBlock2D")
            if i < n_blocks - 1:
                res = res // 2

        # Up blocks mirror down blocks in reverse
        up_types = [
            "AttnUpBlock2D" if d == "AttnDownBlock2D" else "UpBlock2D"
            for d in reversed(down_types)
        ]

        unet_config = {
            "block_out_channels": args.block_out_channels,
            "layers_per_block": args.layers_per_block,
            "down_block_types": down_types,
            "up_block_types": up_types,
        }
        if args.no_mid_attn:
            unet_config["mid_block_type"] = "UNetMidBlock2D"

    model = MultiResConditionalUNet(
        image_size=args.image_size,
        in_channels=args.in_channels,
        cond_in_channels=args.cond_in_channels,
        vae_downsample_factor=args.vae_downsample_factor,
        min_patch_size=args.min_patch_size,
        num_levels=args.num_levels,
        feat_channels=args.feat_channels,
        unet_config=unet_config,
        upsample_factor=args.upsample_factor,
        uncond_drop_prob=args.uncond_drop_prob,
        level_drop=args.level_drop,
        min_keep_levels=args.min_keep_levels,
        depth_per_level=args.depth_per_level,
        mlp_ratio=args.mlp_ratio,
        cnn_base_channels=args.cnn_base_channels,
        level_drop_after_steps=args.level_drop_after_steps,
        cond_use_latent=args.cond_use_latent,
        mae_mask_ratio=args.mae_mask_ratio,
        encoder_type=args.encoder_type,
        vit_patch_size=args.vit_patch_size,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_ratio=args.vit_mlp_ratio,
        vit_use_cnn_stem=args.vit_use_cnn_stem and not args.vit_no_cnn_stem,
        vit_cnn_stem_reduction=args.vit_cnn_stem_reduction,
        vit_init_clip=args.vit_init_clip,
        clip_model_name=args.clip_model_name,
        use_fsq=args.use_fsq,
        fsq_levels=args.fsq_levels,
        fsq_drop_quant_p=args.fsq_drop_quant_p,
        fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
        use_vq=args.use_vq,
        vq_codebook_size=args.vq_codebook_size,
        vq_beta=args.vq_beta,
        level_sizes=args.level_sizes,
    )
    return model


def build_vae(args, device):
    """Build frozen VAE for latent-space diffusion."""
    if args.vae_pretrained is None or args.vae_downsample_factor <= 1:
        return None
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def vae_encode(vae, images):
    """Encode images → latents. Returns (B, C, H/f, W/f)."""
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents


@torch.no_grad()
def vae_decode(vae, latents):
    """Decode latents → images. Returns (B, 3, H, W)."""
    images = vae.decode(latents / vae.config.scaling_factor).sample
    return images


# ──────────────────────────────────────────────────────────────────
#  LR scheduler (cosine with warmup)
# ──────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=1e-6, schedule="constant"):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return base_lr


# ──────────────────────────────────────────────────────────────────
#  DDIM sampling with CFG
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_ddim(model, scheduler, cond_images, num_steps=50,
                guidance_scale=1.5, in_channels=3, vae=None,
                num_active_levels=None, num_active_slots=None):
    """DDIM sampling with CFG, optional VAE decode.

    Args:
        model: MultiResConditionalUNet or Baseline1DConditionalDiT (unwrapped)
        scheduler: DDIMScheduler
        cond_images: (B, C, H, W) conditioning images (original resolution)
        num_steps: DDIM denoising steps
        guidance_scale: CFG scale (1.0 = no guidance)
        in_channels: UNet input channels
        vae: optional VAE for latent→pixel decode
        num_active_levels: override level count for inference (multi-res)
        num_active_slots: override slot count for inference (baseline_1d)
    """
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    extra_kwargs = {}
    if num_active_levels is not None:
        extra_kwargs["num_active_levels"] = num_active_levels
    if num_active_slots is not None:
        extra_kwargs["num_active_slots"] = num_active_slots

    scheduler.set_timesteps(num_steps, device=device)

    latents = torch.randn(B, in_channels, latent_size, latent_size,
                          device=device, dtype=dtype)

    for t in scheduler.timesteps:
        t_batch = t.expand(B)

        if guidance_scale == 0.0:
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               return_uncond=True)
        elif guidance_scale != 1.0:
            noise_cond = model(latents, t_batch, cond_image=cond_images,
                               **extra_kwargs)
            noise_uncond = model(latents, t_batch, cond_image=cond_images,
                                return_uncond=True)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               **extra_kwargs)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    if vae is not None:
        pixels = vae_decode(vae, latents)
        return pixels.clamp(-1, 1)

    return latents.clamp(-1, 1)


# ──────────────────────────────────────────────────────────────────
#  Flow matching ODE sampling (JiT-style)
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow_ode(model, cond_images, num_steps=50,
                    guidance_scale=1.5, in_channels=3, vae=None,
                    num_active_levels=None, num_active_slots=None,
                    method="euler", noise_scale=1.0, t_eps=0.05):
    """ODE-based sampling for flow matching models.

    Integrates from t=0 (noise) to t=1 (data).
    Model predicts x0; velocity is computed as v = (x_pred - z) / (1-t).
    """
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    extra_kwargs = {}
    if num_active_levels is not None:
        extra_kwargs["num_active_levels"] = num_active_levels
    if num_active_slots is not None:
        extra_kwargs["num_active_slots"] = num_active_slots

    z = noise_scale * torch.randn(B, in_channels, latent_size, latent_size,
                                  device=device, dtype=dtype)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def _compute_velocity(z_cur, t_scalar):
        """Compute velocity with optional CFG."""
        t_batch = t_scalar.expand(B)
        t_expand = t_scalar.view(1, 1, 1, 1)

        if guidance_scale != 1.0:
            x_cond = model(z_cur, t_batch, cond_image=cond_images,
                           **extra_kwargs)
            x_uncond = model(z_cur, t_batch, cond_image=cond_images,
                             return_uncond=True)
            v_cond = (x_cond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            v_uncond = (x_uncond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            x_pred = model(z_cur, t_batch, cond_image=cond_images,
                           **extra_kwargs)
            return (x_pred - z_cur) / (1.0 - t_expand).clamp_min(t_eps)

    for i in range(num_steps):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_cur

        if method == "heun" and i < num_steps - 1:
            # Heun's method (2nd order)
            v1 = _compute_velocity(z, t_cur)
            z_euler = z + dt * v1
            v2 = _compute_velocity(z_euler, t_next)
            z = z + dt * 0.5 * (v1 + v2)
        else:
            # Euler step (always use Euler for last step)
            v = _compute_velocity(z, t_cur)
            z = z + dt * v

    if vae is not None:
        pixels = vae_decode(vae, z)
        return pixels.clamp(-1, 1)

    return z.clamp(-1, 1)


# ──────────────────────────────────────────────────────────────────
#  Visual sample generation
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, val_dataset, scheduler, args, accelerator, step,
                     vae=None, ema_model=None, train_dataset=None):
    """Generate grids of [GT | generated] pairs for multiple (level, cfg) configs.

    If train_dataset is provided, upper half of grid uses train samples and
    lower half uses val samples.  Otherwise all samples come from val_dataset.
    """
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device

    # Fixed random GT samples (deterministic, diverse across dataset)
    n_samples = 8
    rng = torch.Generator().manual_seed(args.seed)

    if train_dataset is not None and len(train_dataset) > 0:
        n_train = n_samples // 2
        n_val = n_samples - n_train
        train_idx = torch.randperm(len(train_dataset), generator=rng)[:n_train]
        val_idx = torch.randperm(len(val_dataset), generator=rng)[:n_val]
        images = torch.cat([
            torch.stack([train_dataset[i][0] for i in train_idx]),
            torch.stack([val_dataset[i][0] for i in val_idx]),
        ], dim=0).to(device)
    else:
        indices = torch.randperm(len(val_dataset), generator=rng)[:n_samples]
        images = torch.stack([val_dataset[i][0] for i in indices]).to(device)

    # Conditioning input
    if args.cond_use_latent and vae is not None:
        cond_input = vae_encode(vae, images)
    else:
        cond_input = images

    # Sampling configs to evaluate
    unwrapped = accelerator.unwrap_model(model)
    is_baseline_1d = (args.backbone == "baseline_1d")

    if is_baseline_1d:
        # Slot count configs: progressive subset of tokens
        total_slots = unwrapped.num_slots
        slot_configs = [s for s in args.eval_slot_configs if s <= total_slots]
        if total_slots not in slot_configs:
            slot_configs.append(total_slots)
    else:
        # Level configs: [all, N-1, ..., 1]
        # Render per-level variants whenever the model was trained to be robust
        # to missing levels — either via nested level drop OR via per-token
        # random drop applied to all levels.
        num_levels = unwrapped.num_levels
        eval_per_level = args.level_drop or (
            args.cond_token_drop_prob > 0 and args.cond_token_drop_all_levels
        )
        if eval_per_level:
            level_configs = list(range(num_levels, 0, -1))
        else:
            level_configs = [num_levels]

    # Guidance scales to evaluate
    guidance_scales = [1.0]
    if args.uncond_drop_prob > 0:
        guidance_scales.insert(0, 0.0)  # unconditional
        if args.guidance_scale != 1.0:
            guidance_scales.append(args.guidance_scale)

    save_dir = os.path.join(args.output_dir, "samples")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # Build list of (sample_kwargs, description_str) configs
    if is_baseline_1d:
        sample_configs = []
        for n_slots in slot_configs:
            kwargs = {"num_active_slots": n_slots if n_slots < total_slots else None}
            desc = "all" if n_slots >= total_slots else f"{n_slots}tok"
            sample_configs.append((kwargs, desc))
    else:
        sample_configs = []
        for n_lv in level_configs:
            kwargs = {"num_active_levels": n_lv if n_lv < num_levels else None}
            desc = "all" if n_lv >= num_levels else f"{n_lv}lv"
            sample_configs.append((kwargs, desc))

    for extra_kwargs, cond_desc in sample_configs:
        for gs in guidance_scales:
            if args.use_flow_matching:
                samples = sample_flow_ode(
                    eval_model, cond_input,
                    num_steps=args.eval_num_steps,
                    guidance_scale=gs,
                    in_channels=args.in_channels,
                    vae=vae,
                    method=args.flow_sampling_method,
                    noise_scale=args.flow_noise_scale,
                    t_eps=args.flow_t_eps,
                    **extra_kwargs,
                )
            else:
                samples = sample_ddim(
                    eval_model, scheduler, cond_input,
                    num_steps=args.eval_num_steps,
                    guidance_scale=gs,
                    in_channels=args.in_channels,
                    vae=vae,
                    **extra_kwargs,
                )

            if accelerator.is_main_process:
                cond_01 = (images * 0.5 + 0.5).clamp(0, 1)
                gen_01 = (samples * 0.5 + 0.5).clamp(0, 1)
                combined = torch.stack([cond_01, gen_01], dim=1).view(
                    -1, 3, args.image_size, args.image_size)
                grid = make_grid(combined, nrow=4, padding=2)

                cfg_desc = f"cfg{gs:.1f}"
                fname = f"step_{step:07d}_{cond_desc}_{cfg_desc}.png"
                save_image(grid, os.path.join(save_dir, fname))

    model.train()


# ──────────────────────────────────────────────────────────────────
#  FID evaluation (multi-GPU)
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_fid(model, val_dataset, scheduler, args, accelerator, step,
                 vae=None, ema_model=None):
    """Reconstruction FID: val images → encode → generate → compare to val originals.

    Each GPU reconstructs its share of val images.
    Both real (original val) and reconstructed images are saved.
    FID is computed between the two sets on main process.
    """
    import torch_fidelity

    # Use EMA model if available
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()
    device = accelerator.device

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    total = min(args.fid_num_samples, len(val_dataset))

    per_gpu = math.ceil(total / world_size)
    my_start = rank * per_gpu
    my_end = min(my_start + per_gpu, total)
    my_count = my_end - my_start

    rec_dir = os.path.join(args.output_dir, f"fid_rec_step{step}_rank{rank}")
    real_dir = os.path.join(args.output_dir, f"fid_real_step{step}_rank{rank}")
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    gen_bs = min(args.batch_size, 32)
    generated = 0

    while generated < my_count:
        bs = min(gen_bs, my_count - generated)
        batch_indices = [my_start + generated + i for i in range(bs)]
        cond_imgs = torch.stack([val_dataset[i][0] for i in batch_indices]).to(device)

        # Encoder conditioning: latent or raw image
        if args.cond_use_latent and vae is not None:
            cond_input = vae_encode(vae, cond_imgs)
        else:
            cond_input = cond_imgs

        # Reconstruct: encode conditioning → denoise from noise
        if args.use_flow_matching:
            samples = sample_flow_ode(
                eval_model, cond_input,
                num_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                in_channels=args.in_channels,
                vae=vae,
                num_active_levels=args.eval_num_active_levels,
                method=args.flow_sampling_method,
                noise_scale=args.flow_noise_scale,
                t_eps=args.flow_t_eps,
            )
        else:
            samples = sample_ddim(
                eval_model, scheduler, cond_input,
                num_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                in_channels=args.in_channels,
                vae=vae,
                num_active_levels=args.eval_num_active_levels,
            )

        # Save reconstructed
        samples_01 = (samples * 0.5 + 0.5).clamp(0, 1)
        # Save real (original conditioning images)
        real_01 = (cond_imgs * 0.5 + 0.5).clamp(0, 1)

        for i in range(bs):
            img_idx = my_start + generated + i
            rec_img = transforms.ToPILImage()(samples_01[i].cpu())
            rec_img.save(os.path.join(rec_dir, f"{img_idx:06d}.png"))
            real_img = transforms.ToPILImage()(real_01[i].cpu())
            real_img.save(os.path.join(real_dir, f"{img_idx:06d}.png"))
        generated += bs

    accelerator.wait_for_everyone()

    fid_value = None
    if accelerator.is_main_process:
        # Merge per-rank dirs
        merged_rec = os.path.join(args.output_dir, f"fid_rec_step{step}")
        merged_real = os.path.join(args.output_dir, f"fid_real_step{step}")
        os.makedirs(merged_rec, exist_ok=True)
        os.makedirs(merged_real, exist_ok=True)

        for r in range(world_size):
            for prefix, merged in [("fid_rec", merged_rec), ("fid_real", merged_real)]:
                rank_dir = os.path.join(args.output_dir, f"{prefix}_step{step}_rank{r}")
                if os.path.isdir(rank_dir):
                    for fname in os.listdir(rank_dir):
                        shutil.move(os.path.join(rank_dir, fname),
                                    os.path.join(merged, fname))
                    shutil.rmtree(rank_dir, ignore_errors=True)

        num_gen = len(os.listdir(merged_rec))
        accelerator.print(f"Reconstruction FID: {num_gen} pairs, computing metrics...")

        try:
            kwargs = dict(input1=merged_rec, cuda=True, fid=True, isc=True)
            if args.fid_stats:
                kwargs["fid_statistics_file"] = args.fid_stats
            elif args.fid_real_dir:
                kwargs["input2"] = args.fid_real_dir
            else:
                # Compare reconstructions to the saved real images
                kwargs["input2"] = merged_real

            metrics = torch_fidelity.calculate_metrics(**kwargs)
            fid_value = metrics.get("frechet_inception_distance")
            isc_value = metrics.get("inception_score_mean")
            accelerator.print(f"Step {step} | rFID: {fid_value:.2f} | IS: {isc_value:.2f}")
        except Exception as e:
            accelerator.print(f"FID computation failed: {e}")

        shutil.rmtree(merged_rec, ignore_errors=True)
        shutil.rmtree(merged_real, ignore_errors=True)
    else:
        shutil.rmtree(rec_dir, ignore_errors=True)
        shutil.rmtree(real_dir, ignore_errors=True)

    accelerator.wait_for_everyone()
    model.train()
    return fid_value


# ──────────────────────────────────────────────────────────────────
#  CLEVR detection + attribute evaluation
# ──────────────────────────────────────────────────────────────────

def _clevr_load_font(size, bold=False):
    from PIL import ImageFont
    cands = []
    if bold:
        cands.append("/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
    cands.append("/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")
    for p in cands:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_clevr_bbox(draw, canvas_size, cx, cy, color, label, font, box_size=48):
    """Draw a single bbox + colored label header."""
    W, H = canvas_size
    half = box_size // 2
    x1 = max(int(cx) - half, 0)
    y1 = max(int(cy) - half, 0)
    x2 = min(int(cx) + half, W - 1)
    y2 = min(int(cy) + half, H - 1)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    if label:
        tw = int(draw.textlength(label, font=font))
        th = 12
        ty = max(y1 - th - 1, 0)
        tx = max(min(x1, W - tw - 4), 0)
        draw.rectangle([tx, ty, tx + tw + 4, ty + th], fill=color)
        draw.text((tx + 2, ty - 1), label, fill="white", font=font)


def _annotate_clevr_pair(gt_pil, recon_pil, gt_centers, gt_attrs,
                         peaks, peak_attr_idx, matched_pred, matched_gt,
                         attr_names, clevr_cfg):
    """Draw bboxes on (GT, GEN) PIL images.

    GT side  : green = matched by detector, orange = missed
    GEN side : green = matched + all attrs correct,
               yellow = matched but >=1 attr wrong,
               red    = no GT match (extra/spurious)
    """
    from PIL import ImageDraw

    font = _clevr_load_font(10, bold=True)
    matched_pred_set = set(int(p) for p in matched_pred)
    matched_gt_set = set(int(g) for g in matched_gt)
    pred_to_gt = {int(p): int(g) for p, g in zip(matched_pred, matched_gt)}

    def _idx_label(attrs):
        if isinstance(attrs, dict):
            c = clevr_cfg.COLORS[attrs["color"]]
            sh = clevr_cfg.SHAPES[attrs["shape"]]
            sz = clevr_cfg.SIZES[attrs["size"]]
            ma = clevr_cfg.MATERIALS[attrs["material"]]
        else:
            c = clevr_cfg.COLORS[attrs[0]]
            sh = clevr_cfg.SHAPES[attrs[1]]
            sz = clevr_cfg.SIZES[attrs[2]]
            ma = clevr_cfg.MATERIALS[attrs[3]]
        return f"{sz[:2]} {c[:2]} {ma[:2]} {sh[:2]}"

    # GT side
    gt_anno = gt_pil.copy()
    d_gt = ImageDraw.Draw(gt_anno)
    for gi, (gx, gy) in enumerate(gt_centers):
        color = (16, 200, 64) if gi in matched_gt_set else (255, 140, 0)
        _draw_clevr_bbox(d_gt, gt_anno.size, gx, gy, color,
                         _idx_label(gt_attrs[gi]), font)

    # GEN side
    gen_anno = recon_pil.copy()
    d_gen = ImageDraw.Draw(gen_anno)
    for pi, (px, py) in enumerate(peaks):
        if pi in matched_pred_set:
            gi = pred_to_gt[pi]
            ok = all(peak_attr_idx[pi][a] == gt_attrs[gi][ai]
                     for ai, a in enumerate(attr_names))
            color = (16, 200, 64) if ok else (240, 200, 0)
        else:
            color = (235, 50, 50)
        _draw_clevr_bbox(d_gen, gen_anno.size, px, py, color,
                         _idx_label(peak_attr_idx[pi]), font)

    return gt_anno, gen_anno


@torch.no_grad()
def evaluate_clevr(model, val_dataset, args, accelerator, step,
                   vae=None, ema_model=None, num_samples=30):
    """Reconstruct val images and evaluate with CLEVR detector + classifier.

    Compares reconstructed images against GT scene annotations:
      - Detection: are all objects found? (precision / recall / F1)
      - Classification: are attributes (color, shape, size, material) correct?

    Requires pre-trained detector & classifier checkpoints in clevr_eval/output/checkpoints/.
    """
    import sys
    import numpy as np
    from PIL import Image

    clevr_eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "..", "clevr_eval")
    clevr_eval_dir = os.path.normpath(clevr_eval_dir)
    if not os.path.isdir(clevr_eval_dir):
        accelerator.print(f"[CLEVR eval] clevr_eval dir not found: {clevr_eval_dir}, skipping")
        return None

    if clevr_eval_dir not in sys.path:
        sys.path.insert(0, clevr_eval_dir)

    try:
        import config as clevr_cfg
        from models.detector import CenterDetector
        from models.classifier import AttributeClassifier
        from evaluate import extract_peaks, match_detections
    except ImportError as e:
        accelerator.print(f"[CLEVR eval] import error: {e}, skipping")
        return None

    det_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "detector_best.pt")
    cls_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "classifier_best.pt")
    if not os.path.exists(det_ckpt) or not os.path.exists(cls_ckpt):
        accelerator.print(f"[CLEVR eval] detector/classifier checkpoints not found, skipping")
        return None

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return None

    device = accelerator.device

    # Load detector & classifier
    detector = CenterDetector(backbone_name=clevr_cfg.DETECTOR_BACKBONE).to(device)
    detector.load_state_dict(
        torch.load(det_ckpt, map_location=device, weights_only=True)["model"])
    detector.eval()

    classifier = AttributeClassifier().to(device)
    classifier.load_state_dict(
        torch.load(cls_ckpt, map_location=device, weights_only=True)["model"])
    classifier.eval()

    # Pick samples deterministically
    eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
    eval_model.eval()

    rng = torch.Generator().manual_seed(args.seed + 7777)
    n = min(num_samples, len(val_dataset))
    indices = torch.randperm(len(val_dataset), generator=rng)[:n].tolist()

    # Scene dir: derive from val_dir path
    # val_dir = .../clevr_256_varied_val/images  → scenes_dir = .../clevr_256_varied_val/scenes
    val_images_dir = args.val_dir or os.path.join(args.dataset_root, "val")
    val_root = os.path.dirname(val_images_dir.rstrip("/"))
    scenes_dir = os.path.join(val_root, "scenes")
    if not os.path.isdir(scenes_dir):
        accelerator.print(f"[CLEVR eval] scenes dir not found: {scenes_dir}, skipping")
        return None

    # Transforms for detector/classifier input (match clevr_eval conventions)
    det_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    crop_transform = transforms.Compose([
        transforms.Resize((clevr_cfg.CROP_SIZE, clevr_cfg.CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    attr_names = ["color", "shape", "size", "material"]
    dist_thresholds = clevr_cfg.DETECTOR_DIST_THRESH
    stats = {}
    for t in dist_thresholds:
        stats[t] = {
            "correct": {a: 0 for a in attr_names},
            "correct_all": 0,
            "total_matched": 0,
            "total_pred": 0,
            "total_gt": 0,
        }

    # Per-sample annotation bookkeeping (only when at least one of N_random/N_worst > 0)
    n_annot_rand = max(int(getattr(args, "clevr_eval_n_annotated_random", 0)), 0)
    n_annot_worst = max(int(getattr(args, "clevr_eval_n_annotated_worst", 0)), 0)
    save_annotated = (n_annot_rand + n_annot_worst) > 0
    annot_records = []

    for idx in indices:
        img_path, class_idx = val_dataset.samples[idx]

        # Derive scene JSON path from image path
        # e.g. .../images/easy/CLEVR_easy_000042.png → .../scenes/easy/CLEVR_easy_000042.json
        rel = os.path.relpath(img_path, val_images_dir)
        scene_path = os.path.join(scenes_dir, os.path.splitext(rel)[0] + ".json")
        if not os.path.isfile(scene_path):
            continue

        with open(scene_path) as f:
            scene = json.load(f)

        # GT info
        gt_centers = []
        gt_attrs = []
        for obj in scene["objects"]:
            cx = np.clip(obj["pixel_coords"][0], 0, clevr_cfg.IMG_SIZE - 1)
            cy = np.clip(obj["pixel_coords"][1], 0, clevr_cfg.IMG_SIZE - 1)
            gt_centers.append([cx, cy])
            gt_attrs.append([
                clevr_cfg.COLORS.index(obj["color"]),
                clevr_cfg.SHAPES.index(obj["shape"]),
                clevr_cfg.SIZES.index(obj["size"]),
                clevr_cfg.MATERIALS.index(obj["material"]),
            ])
        gt_centers = np.array(gt_centers) if gt_centers else np.zeros((0, 2))
        n_gt = len(gt_centers)

        # Reconstruct: condition image → model → reconstructed image
        cond_img = val_dataset[idx][0].unsqueeze(0).to(device)  # (1, 3, H, W)
        if args.cond_use_latent and vae is not None:
            cond_input = vae_encode(vae, cond_img)
        else:
            cond_input = cond_img

        if args.use_flow_matching:
            recon = sample_flow_ode(
                eval_model, cond_input,
                num_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                in_channels=args.in_channels, vae=vae,
                method=args.flow_sampling_method,
                noise_scale=args.flow_noise_scale,
                t_eps=args.flow_t_eps,
            )
        else:
            recon = sample_ddim(
                eval_model, None, cond_input,
                num_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                in_channels=args.in_channels, vae=vae,
            )

        # Convert reconstruction to PIL for cropping
        recon_01 = (recon[0] * 0.5 + 0.5).clamp(0, 1)
        recon_pil = transforms.ToPILImage()(recon_01.cpu())
        w, h = recon_pil.size

        # Run detector on reconstruction
        det_input = det_transform(recon_pil).unsqueeze(0).to(device)
        pred_heatmap = detector(det_input).cpu().numpy()[0, 0]
        peaks = extract_peaks(pred_heatmap, threshold=0.3)

        # Pre-classify ALL peaks once (for annotation labels & per-sample score).
        # Used only when save_annotated=True; cheap (one extra classifier call).
        # NOTE: extract_peaks returns (x, y, score) tuples — unpack first 2 only.
        peak_attr_idx = []
        if save_annotated and len(peaks) > 0:
            crops_all = []
            half = clevr_cfg.CROP_SIZE // 2
            for peak in peaks:
                px, py = int(peak[0]), int(peak[1])
                x1, y1 = max(px - half, 0), max(py - half, 0)
                x2, y2 = min(px + half, w), min(py + half, h)
                crops_all.append(crop_transform(recon_pil.crop((x1, y1, x2, y2))))
            preds_all = classifier(torch.stack(crops_all).to(device))
            for k in range(len(peaks)):
                peak_attr_idx.append({
                    a: int(preds_all[a][k].argmax().item()) for a in attr_names
                })

        for t, s in stats.items():
            s["total_gt"] += n_gt
            s["total_pred"] += len(peaks)
            mp, mg, _ = match_detections(peaks, gt_centers, t)
            s["total_matched"] += len(mp)

            if len(mp) == 0:
                continue

            # Crop and classify each matched detection
            crops = []
            half = clevr_cfg.CROP_SIZE // 2
            for pi in mp:
                px, py = int(peaks[pi][0]), int(peaks[pi][1])
                x1, y1 = max(px - half, 0), max(py - half, 0)
                x2, y2 = min(px + half, w), min(py + half, h)
                crops.append(crop_transform(recon_pil.crop((x1, y1, x2, y2))))

            crop_batch = torch.stack(crops).to(device)
            preds = classifier(crop_batch)

            for k, (pi, gi) in enumerate(zip(mp, mg)):
                gt = gt_attrs[gi]
                all_ok = True
                for ai, a in enumerate(attr_names):
                    if preds[a][k].argmax().item() == gt[ai]:
                        s["correct"][a] += 1
                    else:
                        all_ok = False
                if all_ok:
                    s["correct_all"] += 1

        # Record per-sample annotation data using the canonical threshold.
        # Score = F1 over (correct = matched + all 4 attrs right):
        #   2 * n_correct / (n_pred + n_gt)
        if save_annotated:
            ct = int(args.clevr_eval_annot_thresh)
            mp_a, mg_a, _ = match_detections(peaks, gt_centers, ct) \
                if len(peaks) > 0 and n_gt > 0 else ([], [], [])
            n_correct_a = 0
            for k_, (pi_, gi_) in enumerate(zip(mp_a, mg_a)):
                if all(peak_attr_idx[pi_][a] == gt_attrs[gi_][ai]
                       for ai, a in enumerate(attr_names)):
                    n_correct_a += 1
            score = (2.0 * n_correct_a) / max(len(peaks) + n_gt, 1)

            gt_01 = (cond_img[0] * 0.5 + 0.5).clamp(0, 1)
            gt_pil_anno = transforms.ToPILImage()(gt_01.cpu())
            annot_records.append({
                "idx": int(idx),
                "score": float(score),
                "n_gt": int(n_gt),
                "n_pred": int(len(peaks)),
                "n_correct": int(n_correct_a),
                "gt_pil": gt_pil_anno,
                "recon_pil": recon_pil,
                "gt_centers": [list(map(int, c)) for c in gt_centers],
                "gt_attrs": [list(map(int, a)) for a in gt_attrs],
                "peaks": [[int(p[0]), int(p[1])] for p in peaks],
                "peak_attr_idx": peak_attr_idx,
                "matched_pred": list(mp_a),
                "matched_gt": list(mg_a),
                "img_path": img_path,
            })

    # Print & log results
    results = {}
    accelerator.print(f"\n=== CLEVR Eval (step {step}, {n} samples) ===")
    for t, s in stats.items():
        nm = max(s["total_matched"], 1)
        det_prec = s["total_matched"] / max(s["total_pred"], 1)
        det_rec = s["total_matched"] / max(s["total_gt"], 1)
        det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-8)
        attr_acc = {a: s["correct"][a] / nm * 100 for a in attr_names}
        all_acc = s["correct_all"] / nm * 100

        accelerator.print(
            f"  @{t}px  Det: P={det_prec:.3f} R={det_rec:.3f} F1={det_f1:.3f}  "
            f"Attr: " + " ".join(f"{a}={attr_acc[a]:.1f}%" for a in attr_names) +
            f"  all={all_acc:.1f}%"
        )
        results[t] = {
            "det_P": det_prec, "det_R": det_rec, "det_F1": det_f1,
            "attr_acc": attr_acc, "all_attrs_acc": all_acc,
        }

    # Save to JSON
    save_path = os.path.join(args.output_dir, "clevr_eval",
                             f"step_{step:07d}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    # Log scalar to tensorboard (use @10px as representative)
    if 10 in results:
        r = results[10]
        accelerator.log({
            "clevr/det_F1@10px": r["det_F1"],
            "clevr/all_attrs_acc@10px": r["all_attrs_acc"],
        }, step=step)

    # ── Save annotated GT|GEN bbox grids (random + worst) ──
    if save_annotated and annot_records:
        out_dir = os.path.join(args.output_dir, "clevr_eval")
        os.makedirs(out_dir, exist_ok=True)

        # worst = lowest F1 score; random = deterministic-but-varying per step
        sorted_recs = sorted(annot_records, key=lambda r: r["score"])
        worst_recs = sorted_recs[: min(n_annot_worst, len(annot_records))]
        if n_annot_rand > 0:
            rng2 = np.random.RandomState(args.seed + step)
            n_rand = min(n_annot_rand, len(annot_records))
            rand_idx = rng2.choice(len(annot_records), size=n_rand,
                                   replace=False).tolist()
            rand_recs = [annot_records[i] for i in rand_idx]
        else:
            rand_recs = []

        for tag, recs in [("random", rand_recs), ("worst", worst_recs)]:
            if not recs:
                continue
            tiles = []
            for r in recs:
                gt_a, gen_a = _annotate_clevr_pair(
                    r["gt_pil"], r["recon_pil"],
                    r["gt_centers"], r["gt_attrs"],
                    r["peaks"], r["peak_attr_idx"],
                    r["matched_pred"], r["matched_gt"],
                    attr_names, clevr_cfg)
                tiles.append(transforms.ToTensor()(gt_a))
                tiles.append(transforms.ToTensor()(gen_a))
            grid = make_grid(torch.stack(tiles), nrow=2,
                             padding=4, pad_value=1.0)
            grid_path = os.path.join(
                out_dir, f"step_{step:07d}_annotated_{tag}.png")
            save_image(grid, grid_path)
            scores_str = ", ".join(f"{r['score']:.2f}" for r in recs)
            accelerator.print(
                f"[CLEVR eval] saved {tag} bbox grid: {grid_path} "
                f"(scores=[{scores_str}])")

    # Cleanup: remove clevr_eval from sys.path to avoid conflicts
    if clevr_eval_dir in sys.path:
        sys.path.remove(clevr_eval_dir)

    model.train()
    accelerator.wait_for_everyone()
    return results


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
    accelerator.init_trackers("multires_diffusion")

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

    # ── VAE latent caching ──
    use_cached = args.cache_latents and args.vae_pretrained is not None and args.vae_downsample_factor > 1
    cache_dir = None
    latent_only = use_cached and args.cond_use_latent
    if use_cached:
        cache_dir = cache_vae_latents(args, accelerator, latent_only=latent_only)

    # ── Ensure consolidated cache exists for latent-only preload ──
    if use_cached and latent_only:
        suffix = "_latent_only"
        consolidated_path = os.path.join(cache_dir, f"consolidated{suffix}.pt")
        if not os.path.isfile(consolidated_path):
            if accelerator.is_main_process:
                accelerator.print(
                    f"Consolidated cache missing, creating automatically: {consolidated_path}"
                )
                consolidate_cache(cache_dir, latent_only=True)
            accelerator.wait_for_everyone()

    # ── Dataset ──
    preload_sharded = use_cached and latent_only
    if use_cached:
        train_ds = CachedLatentDataset(
            cache_dir, flip_aug=True, latent_only=latent_only,
            preload=latent_only,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        accelerator.print(f"Train (cached): {len(train_ds)} per rank")
    else:
        train_img_dir = args.train_dir or os.path.join(args.dataset_root, "train")
        if args.cache_to_local_disk:
            cache_sub = _resolve_local_cache_subdir(args, train_img_dir)
            meta = build_memmap_image_cache(
                train_img_dir, cache_sub, args.image_size, accelerator, "train")
            train_ds = MemmapImageDataset(meta, train=True)
        else:
            train_ds, _ = build_datasets(args)
        accelerator.print(f"Train: {len(train_ds)}")

    # Val dataset (always raw images for FID/sampling)
    # Priority: --val_dir > dataset_root/val (auto-detect) > train dir
    train_img_dir = args.train_dir or os.path.join(args.dataset_root, "train")
    if args.val_dir and os.path.isdir(args.val_dir):
        val_dir = args.val_dir
    else:
        val_candidate = os.path.join(args.dataset_root, "val")
        if os.path.isdir(val_candidate):
            val_dir = val_candidate
        else:
            val_dir = train_img_dir
    if val_dir != train_img_dir:
        accelerator.print(f"Val (separate): {val_dir}")
    if args.cache_to_local_disk:
        val_cache_sub = _resolve_local_cache_subdir(args, val_dir)
        val_meta = build_memmap_image_cache(
            val_dir, val_cache_sub, args.image_size, accelerator, "val")
        val_ds = MemmapImageDataset(val_meta, train=False)
    else:
        val_ds = datasets.ImageFolder(
            val_dir,
            transform=build_val_transform(args.image_size),
        )
    # Train dataset with val transform for sampling grid (no augmentation)
    sample_train_ds = None
    if val_dir != train_img_dir:
        if args.cache_to_local_disk:
            # Reuse train cache built above (or build here if train path took cache branch)
            sample_cache_sub = _resolve_local_cache_subdir(args, train_img_dir)
            sample_meta = build_memmap_image_cache(
                train_img_dir, sample_cache_sub, args.image_size, accelerator, "train")
            sample_train_ds = MemmapImageDataset(sample_meta, train=False)
        else:
            sample_train_ds = datasets.ImageFolder(
                train_img_dir,
                transform=build_val_transform(args.image_size),
            )

    if preload_sharded:
        # Data already split per rank → no DistributedSampler, just shuffle
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
    else:
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
    # ── Model ──
    model = build_model(args)
    if accelerator.is_main_process:
        accelerator.print(model.describe())
        n_total = sum(p.numel() for p in model.parameters()) / 1e6
        n_encoder = sum(p.numel() for p in model.encoder.parameters()) / 1e6
        if hasattr(model, 'unet'):
            n_backbone = sum(p.numel() for p in model.unet.parameters()) / 1e6
            bname = "unet"
        else:
            n_backbone = n_total - n_encoder
            bname = "dit"
        n_other = n_total - n_encoder - n_backbone
        accelerator.print(
            f"Parameters: {n_total:.1f}M "
            f"(encoder: {n_encoder:.1f}M, {bname}: {n_backbone:.1f}M, other: {n_other:.1f}M)"
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
    )
    eval_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    # ── VAE (frozen) ──
    # If using cached latents, VAE is only needed for eval/sampling (loaded lazily)
    if use_cached:
        vae = None
        accelerator.print(f"VAE: cached latents (no VAE loaded for training)")
    else:
        vae = build_vae(args, accelerator.device)
        if vae is not None:
            accelerator.print(f"VAE: {args.vae_pretrained} (×{args.vae_downsample_factor})")

    # ── Prepare ──
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # ── EMA (after prepare so it's on the right device) ──
    ema = None
    if args.ema_decay > 0:
        ema = EMA(accelerator.unwrap_model(model), decay=args.ema_decay)
        accelerator.print(f"EMA: decay={args.ema_decay}")

    # ── Resume ──
    global_step = load_checkpoint(accelerator, model, optimizer, args, ema=ema)

    # Helper: lazily load VAE for eval/sampling when using cached training
    def get_eval_vae():
        nonlocal vae
        if vae is None and args.vae_pretrained and args.vae_downsample_factor > 1:
            accelerator.print("Loading VAE for evaluation...")
            vae = build_vae(args, accelerator.device)
        return vae

    if args.eval_only:
        accelerator.print("Running FID evaluation only...")
        ema_eval = ema.shadow if ema is not None else None
        evaluate_fid(model, val_ds, eval_scheduler, args, accelerator, global_step,
                     vae=get_eval_vae(), ema_model=ema_eval)
        return

    if args.eval_clevr_only:
        accelerator.print(f"Running CLEVR eval only at step {global_step}...")
        ema_eval = ema.shadow if ema is not None else None
        evaluate_clevr(model, val_ds, args, accelerator, global_step,
                       vae=get_eval_vae(), ema_model=ema_eval,
                       num_samples=args.clevr_eval_samples)
        return

    # ── Train ──
    accelerator.print(f"Starting training from step {global_step}")
    model.train()
    epoch = 0
    start_step = global_step
    t_start = time.time()

    # Progress bar (main process only)
    pbar = tqdm(
        initial=global_step, total=args.max_train_steps,
        desc="Training", dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    while global_step < args.max_train_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= args.max_train_steps:
                break

            # Unpack batch: cached latent-only → latent,
            #               cached w/ image → (image, latent), normal → (image, label)
            if use_cached and latent_only:
                x0 = batch.float()
                images = None
            elif use_cached:
                images, x0 = batch[0], batch[1]
                x0 = x0.float()       # cached as half
                images = images.float()
            else:
                images, _ = batch
                x0 = None

            # LR schedule
            cur_lr = get_lr(global_step, args.warmup_steps, args.max_train_steps, lr, schedule=args.lr_schedule)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Update step for level drop schedule
            accelerator.unwrap_model(model).set_step(global_step)

            with accelerator.accumulate(model):
                # VAE: encode to latent space (skip if cached)
                if x0 is None:
                    if vae is not None:
                        x0 = vae_encode(vae, images)
                    else:
                        x0 = images

                # Encoder input: raw image or VAE latent
                cond_images = x0 if args.cond_use_latent else images

                if args.use_flow_matching:
                    # ── Flow matching (JiT-style V-loss) ──
                    # Sample t from logit-normal distribution
                    z_t = (torch.randn(x0.shape[0], device=x0.device)
                           * args.flow_P_std + args.flow_P_mean)
                    t_flow = torch.sigmoid(z_t)  # (B,) in (0, 1)
                    t_expand = t_flow.view(-1, 1, 1, 1)

                    # Create noisy sample: z = t*x + (1-t)*e
                    e = torch.randn_like(x0) * args.flow_noise_scale
                    noisy = t_expand * x0 + (1 - t_expand) * e

                    # Velocity target: v = (x - z) / (1 - t)
                    v_target = ((x0 - noisy)
                                / (1 - t_expand).clamp_min(args.flow_t_eps))

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

                        # Convert x-prediction to velocity
                        v_pred = ((x_pred - noisy)
                                  / (1 - t_expand).clamp_min(args.flow_t_eps))

                        loss = F.mse_loss(v_pred, v_target)

                        if "vq_loss" in aux:
                            loss = loss + args.vq_loss_weight * aux["vq_loss"]
                else:
                    # ── Standard DDPM training ──
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

            # Only count a step when gradients are actually synced (accum finished)
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
                    log_dict["vq_perplexity"] = aux.get("vq_perplexity", torch.tensor(0.0)).item()
                    log_dict["vq_usage"] = aux.get("vq_usage", torch.tensor(0.0)).item()
                accelerator.log(log_dict, step=global_step)

            # ── Sample ──
            if global_step % args.sample_every == 0:
                ema_eval = ema.shadow if ema is not None else None
                generate_samples(model, val_ds, eval_scheduler, args,
                                 accelerator, global_step, vae=get_eval_vae(),
                                 ema_model=ema_eval,
                                 train_dataset=sample_train_ds)

            # ── Save ──
            if global_step % args.save_every == 0:
                save_checkpoint(accelerator, model, optimizer, global_step, args,
                                ema=ema)

            # ── FID ──
            if global_step % args.fid_every == 0 and global_step > 0:
                ema_eval = ema.shadow if ema is not None else None
                fid = evaluate_fid(model, val_ds, eval_scheduler, args,
                                   accelerator, global_step, vae=get_eval_vae(),
                                   ema_model=ema_eval)
                if fid is not None:
                    accelerator.log({"fid": fid}, step=global_step)

            # ── CLEVR eval ──
            if (args.clevr_eval_every > 0
                    and global_step % args.clevr_eval_every == 0
                    and global_step > 0):
                ema_eval = ema.shadow if ema is not None else None
                evaluate_clevr(model, val_ds, args, accelerator, global_step,
                               vae=get_eval_vae(), ema_model=ema_eval,
                               num_samples=args.clevr_eval_samples)

        epoch += 1

    pbar.close()
    save_checkpoint(accelerator, model, optimizer, global_step, args, ema=ema)
    accelerator.print("Training complete.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
