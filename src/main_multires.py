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
                   choices=["cnn", "vit"],
                   help="Encoder backend: 'cnn' (PatchCNN+merge) or 'vit' (shared CellViT)")
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
                   choices=["unet", "dit"],
                   help="Denoising backbone: 'unet' (default) or 'dit'")

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
                   help="Number of in-context learnable tokens (JiT=32, 0=disabled)")
    p.add_argument("--dit_in_context_start", type=int, default=4,
                   help="Layer index to prepend in-context tokens (JiT-B=4, L=8, H=10)")

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
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--uncond_drop_prob", type=float, default=0.1)
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
    p.add_argument("--num_workers", type=int, default=8)

    return p.parse_args()


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
            use_fsq=args.use_fsq,
            fsq_levels=args.fsq_levels,
            fsq_drop_quant_p=args.fsq_drop_quant_p,
            fsq_corrupt_tokens_p=args.fsq_corrupt_tokens_p,
            use_vq=args.use_vq,
            vq_codebook_size=args.vq_codebook_size,
            vq_beta=args.vq_beta,
            level_sizes=args.level_sizes,
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

def get_lr(step, warmup_steps, max_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────
#  DDIM sampling with CFG
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_ddim(model, scheduler, cond_images, num_steps=50,
                guidance_scale=1.5, in_channels=3, vae=None,
                num_active_levels=None):
    """DDIM sampling with CFG, optional VAE decode.

    Args:
        model: MultiResConditionalUNet (unwrapped)
        scheduler: DDIMScheduler
        cond_images: (B, C, H, W) conditioning images (original resolution)
        num_steps: DDIM denoising steps
        guidance_scale: CFG scale (1.0 = no guidance)
        in_channels: UNet input channels
        vae: optional VAE for latent→pixel decode
        num_active_levels: override level count for inference
    """
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    scheduler.set_timesteps(num_steps, device=device)

    latents = torch.randn(B, in_channels, latent_size, latent_size,
                          device=device, dtype=dtype)

    for t in scheduler.timesteps:
        t_batch = t.expand(B)

        if guidance_scale == 0.0:
            # Pure unconditional generation
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               return_uncond=True)
        elif guidance_scale != 1.0:
            noise_cond = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)
            noise_uncond = model(latents, t_batch, cond_image=cond_images,
                                return_uncond=True)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode if VAE
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
                    num_active_levels=None, method="euler",
                    noise_scale=1.0, t_eps=0.05):
    """ODE-based sampling for flow matching models.

    Integrates from t=0 (noise) to t=1 (data).
    Model predicts x0; velocity is computed as v = (x_pred - z) / (1-t).
    """
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    z = noise_scale * torch.randn(B, in_channels, latent_size, latent_size,
                                  device=device, dtype=dtype)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def _compute_velocity(z_cur, t_scalar):
        """Compute velocity with optional CFG."""
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

    # Level configs to evaluate: [all, N-1, ..., 1]
    unwrapped = accelerator.unwrap_model(model)
    num_levels = unwrapped.num_levels
    if args.level_drop:
        level_configs = list(range(num_levels, 0, -1))  # e.g. [4, 3, 2, 1]
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

    for n_lv in level_configs:
        for gs in guidance_scales:
            if args.use_flow_matching:
                samples = sample_flow_ode(
                    eval_model, cond_input,
                    num_steps=args.eval_num_steps,
                    guidance_scale=gs,
                    in_channels=args.in_channels,
                    vae=vae,
                    num_active_levels=n_lv if n_lv < num_levels else None,
                    method=args.flow_sampling_method,
                    noise_scale=args.flow_noise_scale,
                    t_eps=args.flow_t_eps,
                )
            else:
                samples = sample_ddim(
                    eval_model, scheduler, cond_input,
                    num_steps=args.eval_num_steps,
                    guidance_scale=gs,
                    in_channels=args.in_channels,
                    vae=vae,
                    num_active_levels=n_lv if n_lv < num_levels else None,
                )

            if accelerator.is_main_process:
                cond_01 = (images * 0.5 + 0.5).clamp(0, 1)
                gen_01 = (samples * 0.5 + 0.5).clamp(0, 1)
                combined = torch.stack([cond_01, gen_01], dim=1).view(
                    -1, 3, args.image_size, args.image_size)
                grid = make_grid(combined, nrow=4, padding=2)

                lv_desc = "all" if n_lv >= num_levels else f"{n_lv}lv"
                cfg_desc = f"cfg{gs:.1f}"
                fname = f"step_{step:07d}_{lv_desc}_{cfg_desc}.png"
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
    val_ds = datasets.ImageFolder(
        val_dir,
        transform=build_val_transform(args.image_size),
    )
    # Train dataset with val transform for sampling grid (no augmentation)
    sample_train_ds = None
    if val_dir != train_img_dir:
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
            cur_lr = get_lr(global_step, args.warmup_steps, args.max_train_steps, lr)
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

        epoch += 1

    pbar.close()
    save_checkpoint(accelerator, model, optimizer, global_step, args, ema=ema)
    accelerator.print("Training complete.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
