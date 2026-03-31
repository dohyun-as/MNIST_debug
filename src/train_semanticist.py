"""
SemantIST Stage-1 Tokenizer training script.

Matches the architecture & training config of:
  "Principal Components Enable A New Language of Images" (SemantIST)
  - ViT-B encoder with causal slot attention
  - DiT-L/2 decoder (adaLN-Zero, learn_sigma)
  - REPA alignment with DINOv2
  - Nested slot sampling (variable # of active slots)
  - CFG dropout 10%

Uses pre-cached VAE latents (from main_multires.py cache) to skip
VAE encoding during training.
"""

import argparse
import json
import math
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_fidelity
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from semanticist.stage1.diffuse_slot import DiffuseSlot
from semanticist.stage1 import vision_transformer
from semanticist.stage1.vision_transformer import VisionTransformer
from semanticist.utils.lr_scheduler import build_scheduler


class DiffuseSlotCached(DiffuseSlot):
    """DiffuseSlot variant that supports cached VAE latents.

    Two modes controlled by enc_use_latent:
      False (default): encoder takes [0,1] images, cache provides (image, latent)
      True:            encoder takes VAE latents directly (patch_size=1, in_chans=16),
                       cache provides latent-only. REPA is disabled.

    forward() accepts (x, x_vae) for DDP-compatible gradient sync.
    """

    def __init__(self, enc_use_latent=False, **kwargs):
        if enc_use_latent:
            kwargs["use_repa"] = False
        super().__init__(**kwargs)
        self.enc_use_latent = enc_use_latent

        if enc_use_latent:
            # Replace ViT encoder: image 256×256×3 patch16
            #                    → latent 16×16×16   patch1
            # num_patches stays 256, ViT body unchanged
            latent_size = self.enc_img_size // 16  # 256 → 16
            drop_path = kwargs.get("drop_path_rate", 0.1)
            slot_dim = kwargs.get("slot_dim", 16)

            # Use VisionTransformer directly to override patch_size and in_chans
            # (vit_base_patch16 hardcodes patch_size=16)
            self.encoder = VisionTransformer(
                img_size=[latent_size],
                patch_size=1,
                in_chans=16,
                embed_dim=self.encoder.embed_dim,
                depth=len(self.encoder.blocks),
                num_heads=self.encoder.blocks[0].attn.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_path_rate=drop_path,
                num_slots=self.num_slots,
            )
            self.num_channels = self.encoder.num_features
            self.encoder2slot = nn.Linear(self.num_channels, slot_dim)

    def forward(self, x, x_vae=None, sample=False,
                epoch=None, inference_with_n_slots=-1, cfg=1.0):
        if self.enc_use_latent:
            # x is the VAE latent (encoder input & diffusion target)
            if sample:
                # For sampling: encode slots, then generate
                slots = self.encode_slots(x)
                return self.forward_with_latents(
                    x, slots, None,
                    sample=True, epoch=epoch,
                    inference_with_n_slots=inference_with_n_slots, cfg=cfg,
                )
            slots = self.encode_slots(x)
            return self.forward_with_latents(
                x, slots, None,
                sample=False, epoch=epoch,
                inference_with_n_slots=inference_with_n_slots, cfg=cfg,
            )

        if x_vae is None:
            # Eval path: raw images → full original forward (VAE encode inside)
            return super().forward(x, sample=sample, epoch=epoch,
                                   inference_with_n_slots=inference_with_n_slots,
                                   cfg=cfg)
        # Training path: images for encoder, cached latent for diffusion
        slots = self.encode_slots(x)
        z = self.repa_encode(x) if self.use_repa else None
        return self.forward_with_latents(
            x_vae, slots, z,
            sample=sample, epoch=epoch,
            inference_with_n_slots=inference_with_n_slots, cfg=cfg,
        )


# ──────────────────────────────────────────────────────────────────
#  EMA
# ──────────────────────────────────────────────────────────────────

class EMAModel:
    def __init__(self, model, device, decay=0.999):
        self.device = device
        self.decay = decay
        self.ema_params = OrderedDict(
            (name, param.clone().detach().to(device))
            for name, param in model.named_parameters()
            if param.requires_grad
        )

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.ema_params[name].lerp_(param.data, 1 - self.decay)

    def apply(self, model):
        """Copy EMA params into model (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.ema_params:
                param.data.copy_(self.ema_params[name])

    def state_dict(self):
        return self.ema_params

    def load_state_dict(self, params):
        self.ema_params = OrderedDict(
            (name, param.clone().detach().to(self.device))
            for name, param in params.items()
        )


# ──────────────────────────────────────────────────────────────────
#  Dataset: cached image + latent pairs
# ──────────────────────────────────────────────────────────────────

class CachedImageLatentDataset(torch.utils.data.Dataset):
    """Loads cached (image, latent) pairs from individual .pt files.

    The cache was created with Normalize([0.5]*3, [0.5]*3), so images
    are in [-1, 1].  We convert to [0, 1] for SemantIST's ViT encoder.
    Flip augmentation: each file stores original + flipped versions.
    """

    def __init__(self, cache_dir, flip_aug=True):
        self.cache_dir = cache_dir
        self.flip_aug = flip_aug
        self.files = sorted(
            f for f in os.listdir(cache_dir)
            if f.endswith('.pt') and not f.startswith('consolidated')
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.cache_dir, self.files[idx]),
            map_location='cpu', weights_only=True,
        )
        flip = self.flip_aug and torch.rand(1).item() < 0.5
        suffix = '_flip' if flip else ''
        image = data[f'image{suffix}']    # [-1, 1] fp16
        latent = data[f'latent{suffix}']  # scaled, fp16
        # Convert image to [0, 1] for SemantIST
        image = (image.float() + 1) / 2
        return image, latent.float()


class CachedLatentOnlyDataset(torch.utils.data.Dataset):
    """Latent-only dataset loaded from consolidated .pt file into RAM.

    Used when enc_use_latent=True: encoder receives latents directly,
    so no image loading is needed. Each rank holds only its shard.
    Expects consolidated_latent_only.pt to already exist (created before
    dataset init via consolidate_cache).
    """

    def __init__(self, cache_dir, flip_aug=True, rank=0, world_size=1):
        self.flip_aug = flip_aug

        consolidated_path = os.path.join(cache_dir, "consolidated_latent_only.pt")
        assert os.path.isfile(consolidated_path), \
            f"Consolidated cache not found: {consolidated_path}"
        print(f"[Rank {rank}] Loading consolidated latent cache "
              f"(shard {rank}/{world_size}) ...")
        full = torch.load(consolidated_path, map_location='cpu',
                          weights_only=True)
        self.latent = full['latent'][rank::world_size].clone()
        self.latent_flip = full['latent_flip'][rank::world_size].clone()
        del full
        print(f"[Rank {rank}] Loaded {len(self.latent)} latent samples.")

    def __len__(self):
        return self.latent.shape[0]

    def __getitem__(self, idx):
        flip = self.flip_aug and torch.rand(1).item() < 0.5
        if flip:
            return self.latent_flip[idx]
        return self.latent[idx]


# ──────────────────────────────────────────────────────────────────
#  Optimizer (SemantIST style: decay 2D+ params only)
# ──────────────────────────────────────────────────────────────────

def create_optimizer(model, weight_decay, lr, betas=(0.9, 0.95)):
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay = [p for p in param_dict.values() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ──────────────────────────────────────────────────────────────────
#  Sampling & evaluation helpers
# ──────────────────────────────────────────────────────────────────

def build_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def generate_samples(model_unwrapped, val_loader, device, save_path,
                     num_slots, cfg=1.0, max_batches=1,
                     enc_use_latent=False):
    """Generate reconstruction samples for visual inspection."""
    model_unwrapped.eval()

    if enc_use_latent:
        # val_loader yields (image, label). Encode through VAE to get latents
        # for the encoder, then decode reconstructions with VAE.
        for i, (imgs, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            imgs = imgs.to(device)
            x_vae = model_unwrapped.vae_encode(imgs)
            # DiffuseSlotCached.forward handles enc_use_latent:
            # encode_slots(x_vae) + forward_with_latents + sample() which already vae_decodes
            recs = model_unwrapped(x_vae, sample=True,
                                   inference_with_n_slots=num_slots,
                                   cfg=cfg)
            pairs = torch.stack((imgs, recs.to(imgs.device)), dim=1)
            pairs = pairs.view(-1, *imgs.shape[1:])
            grid = make_grid(pairs, nrow=6, normalize=True, value_range=(0, 1))
            suffix = f"_cfg{cfg}" if cfg != 1.0 else ""
            save_image(grid, save_path.replace(".jpg", f"{suffix}.jpg"))
    else:
        for i, (imgs, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            imgs = imgs.to(device)
            recs = model_unwrapped(imgs, sample=True,
                                   inference_with_n_slots=num_slots, cfg=cfg)
            pairs = torch.stack((imgs, recs.to(imgs.device)), dim=1)
            pairs = pairs.view(-1, *imgs.shape[1:])
            grid = make_grid(pairs, nrow=6, normalize=True, value_range=(0, 1))
            suffix = f"_cfg{cfg}" if cfg != 1.0 else ""
            save_image(grid, save_path.replace(".jpg", f"{suffix}.jpg"))


# ──────────────────────────────────────────────────────────────────
#  FID evaluation (multi-GPU reconstruction)
# ──────────────────────────────────────────────────────────────────

def _save_img_batch(imgs, save_paths):
    """Save a batch of [0,1] tensors as PNG files using a thread pool."""
    imgs_np = np.clip(imgs.float().numpy().transpose(0, 2, 3, 1) * 255,
                      0, 255).astype(np.uint8)
    imgs_np = imgs_np[:, :, :, ::-1]  # RGB → BGR for cv2
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(cv2.imwrite, p, img)
                   for p, img in zip(save_paths, imgs_np)]
        for f in futures:
            f.result()


@torch.no_grad()
def evaluate_fid(model_unwrapped, val_dataset, accelerator, args,
                 global_step, num_slots, cfg=1.0, fid_stats=None,
                 enc_use_latent=False, fid_num_samples=50000):
    """Reconstruction FID: val images → encode → reconstruct → compare.

    Each GPU reconstructs its share (manual index split, same as main_multires.py).
    Both real and reconstructed images are saved per rank, then merged on main.
    """
    model_unwrapped.eval()
    device = accelerator.device

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    total = min(fid_num_samples, len(val_dataset))

    per_gpu = math.ceil(total / world_size)
    my_start = rank * per_gpu
    my_end = min(my_start + per_gpu, total)
    my_count = my_end - my_start

    rec_dir = os.path.join(args.output_dir, f"fid_rec_step{global_step}_rank{rank}")
    real_dir = os.path.join(args.output_dir, f"fid_real_step{global_step}_rank{rank}")
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    gen_bs = 32
    generated = 0

    while generated < my_count:
        bs = min(gen_bs, my_count - generated)
        batch_indices = [my_start + generated + i for i in range(bs)]
        imgs = torch.stack([val_dataset[i][0] for i in batch_indices]).to(device)

        with accelerator.autocast():
            if enc_use_latent:
                x_vae = model_unwrapped.vae_encode(imgs)
                recs = model_unwrapped(
                    x_vae, sample=True,
                    inference_with_n_slots=num_slots, cfg=cfg)
            else:
                recs = model_unwrapped(
                    imgs, sample=True,
                    inference_with_n_slots=num_slots, cfg=cfg)

        # Save reconstructed (clamp to [0,1])
        recs_01 = recs.clamp(0, 1).float().cpu()
        # Save real (images are already [0,1] from build_val_transform)
        real_01 = imgs.clamp(0, 1).float().cpu()

        rec_paths = [os.path.join(rec_dir, f"{my_start + generated + j:06d}.png")
                     for j in range(bs)]
        real_paths = [os.path.join(real_dir, f"{my_start + generated + j:06d}.png")
                      for j in range(bs)]
        _save_img_batch(recs_01, rec_paths)
        _save_img_batch(real_01, real_paths)

        generated += bs

    accelerator.wait_for_everyone()

    fid_value = None
    if accelerator.is_main_process:
        # Merge per-rank directories
        merged_rec = os.path.join(args.output_dir, f"fid_rec_step{global_step}")
        merged_real = os.path.join(args.output_dir, f"fid_real_step{global_step}")
        os.makedirs(merged_rec, exist_ok=True)
        os.makedirs(merged_real, exist_ok=True)

        for r in range(world_size):
            for prefix, merged in [("fid_rec", merged_rec), ("fid_real", merged_real)]:
                rank_dir = os.path.join(args.output_dir,
                                        f"{prefix}_step{global_step}_rank{r}")
                if os.path.isdir(rank_dir):
                    for fname in os.listdir(rank_dir):
                        shutil.move(os.path.join(rank_dir, fname),
                                    os.path.join(merged, fname))
                    shutil.rmtree(rank_dir, ignore_errors=True)

        num_gen = len(os.listdir(merged_rec))
        accelerator.print(
            f"FID eval: {num_gen} reconstructions, computing metrics...")
        try:
            fid_kwargs = dict(
                input1=merged_rec,
                cuda=True,
                fid=True,
                isc=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            if fid_stats:
                fid_kwargs["fid_statistics_file"] = fid_stats
            else:
                fid_kwargs["input2"] = merged_real

            metrics = torch_fidelity.calculate_metrics(**fid_kwargs)
            fid_value = metrics.get("frechet_inception_distance")
            isc_value = metrics.get("inception_score_mean")
            accelerator.print(
                f"Step {global_step} | cfg={cfg} | "
                f"rFID: {fid_value:.2f} | IS: {isc_value:.2f}")
            accelerator.log({
                f"fid_cfg{cfg}": fid_value,
                f"isc_cfg{cfg}": isc_value,
            }, step=global_step)
        except Exception as e:
            accelerator.print(f"FID computation failed: {e}")

        shutil.rmtree(merged_rec, ignore_errors=True)
        shutil.rmtree(merged_real, ignore_errors=True)
    else:
        shutil.rmtree(rec_dir, ignore_errors=True)
        shutil.rmtree(real_dir, ignore_errors=True)

    accelerator.wait_for_everyone()
    return fid_value


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("SemantIST Stage-1 Tokenizer Training")

    # Paths
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_root", type=str, required=True,
                   help="Path to ImageNet root (with train/ and val/ subdirs)")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Latent cache directory. Default: <output_dir>/latent_cache")
    p.add_argument("--fid_stats", type=str, default=None,
                   help="Path to FID reference stats (.npz)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint directory to resume from")

    # Model architecture (SemantIST defaults)
    p.add_argument("--encoder", type=str, default="vit_base_patch16")
    p.add_argument("--enc_img_size", type=int, default=256)
    p.add_argument("--enc_causal", type=bool, default=True)
    p.add_argument("--num_slots", type=int, default=256)
    p.add_argument("--slot_dim", type=int, default=16)
    p.add_argument("--norm_slots", type=bool, default=True)
    p.add_argument("--dit_model", type=str, default="DiT-L-2")
    p.add_argument("--vae", type=str, default="xwen99/mar-vae-kl16")
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    # Nested slot sampling
    p.add_argument("--enable_nest_after", type=int, default=50,
                   help="Enable nested sampling after this epoch (-1=never)")

    # Encoder input mode
    p.add_argument("--enc_use_latent", action="store_true", default=False,
                   help="Feed VAE latents (16×16×16) to ViT encoder instead of images. "
                        "Enables latent-only preload. Disables REPA.")

    # REPA
    p.add_argument("--use_repa", action="store_true", default=True)
    p.add_argument("--repa_loss_weight", type=float, default=1.0)
    p.add_argument("--repa_encoder_depth", type=int, default=8)

    # Diffusion / sampling
    p.add_argument("--num_sampling_steps", type=str, default="250")
    p.add_argument("--cfg", type=float, default=3.0,
                   help="CFG scale for evaluation sampling")

    # Training
    p.add_argument("--num_epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Per-GPU batch size")
    p.add_argument("--blr", type=float, default=2.5e-5,
                   help="Base learning rate (scaled by effective_bs/256)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=3.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--compile", action="store_true", default=False)

    # EMA
    p.add_argument("--enable_ema", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--fid_every", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Eval
    p.add_argument("--test_num_slots", type=int, default=None,
                   help="Number of slots for eval sampling (default: num_slots)")
    p.add_argument("--eval_only", action="store_true")

    return p.parse_args()


def train(args):
    # ── Accelerator ──
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    if args.seed is not None:
        torch.manual_seed(args.seed + accelerator.process_index)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    # Save args
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Model ──
    model = DiffuseSlotCached(
        enc_use_latent=args.enc_use_latent,
        encoder=args.encoder,
        drop_path_rate=args.drop_path_rate,
        enc_img_size=args.enc_img_size,
        enc_causal=args.enc_causal,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        norm_slots=args.norm_slots,
        enable_nest=False,
        enable_nest_after=args.enable_nest_after,
        vae=args.vae,
        dit_model=args.dit_model,
        num_sampling_steps=args.num_sampling_steps,
        use_repa=args.use_repa,
        repa_encoder_depth=args.repa_encoder_depth,
        repa_loss_weight=args.repa_loss_weight,
    )

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) / 1e6
    n_e2s = sum(p.numel() for p in model.encoder2slot.parameters() if p.requires_grad) / 1e6
    n_dit = sum(p.numel() for p in model.dit.parameters() if p.requires_grad) / 1e6
    n_other = n_total - n_encoder - n_e2s - n_dit
    accelerator.print(
        f"Trainable parameters: {n_total:.1f}M "
        f"(encoder: {n_encoder:.1f}M, encoder2slot: {n_e2s:.1f}M, "
        f"dit: {n_dit:.1f}M, other: {n_other:.1f}M)"
    )

    # ── LR ──
    effective_bs = (args.batch_size * args.grad_accum_steps
                    * accelerator.num_processes)
    lr = args.blr * effective_bs / 256
    accelerator.print(f"Effective batch size: {effective_bs}, LR: {lr:.2e}")

    # ── Dataset ──
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "latent_cache")
    assert os.path.isdir(cache_dir), f"Cache dir not found: {cache_dir}"

    if args.enc_use_latent:
        # Consolidate once from main process, then all load
        consolidated_path = os.path.join(cache_dir, "consolidated_latent_only.pt")
        if accelerator.is_main_process and not os.path.isfile(consolidated_path):
            # Import and run consolidation
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from main_multires import consolidate_cache
            consolidate_cache(cache_dir, latent_only=True)
        accelerator.wait_for_everyone()

        # Load one rank at a time to avoid peak RAM spike
        for loading_rank in range(accelerator.num_processes):
            if accelerator.process_index == loading_rank:
                train_ds = CachedLatentOnlyDataset(
                    cache_dir, flip_aug=True,
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                )
            accelerator.wait_for_everyone()
        train_sampler = None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
    else:
        train_ds = CachedImageLatentDataset(cache_dir, flip_aug=True)
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
    accelerator.print(f"Training samples: {len(train_ds)} (per rank)")

    # Val dataset (raw images for visual sampling)
    val_ds = datasets.ImageFolder(
        os.path.join(args.dataset_root, "train"),
        transform=build_val_transform(args.enc_img_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Optimizer & Scheduler ──
    optimizer = create_optimizer(model, args.weight_decay, lr)

    warmup_steps = args.warmup_epochs * len(train_loader)
    decay_steps = args.num_epochs * len(train_loader)
    scheduler = build_scheduler(
        optimizer,
        args.num_epochs,
        len(train_loader),
        lr_min=0,
        warmup_steps=warmup_steps,
        warmup_lr_init=0,
        decay_steps=decay_steps,
        cosine_lr=True,
    )
    
    # ── Prepare ──
    # NOTE: pre-sharded data (enc_use_latent) must NOT go through
    #       accelerator.prepare — it would add a DistributedSampler
    #       on top of already-sharded data (double sharding bug).
    if args.enc_use_latent:
        model, optimizer, scheduler, val_loader = accelerator.prepare(
            model, optimizer, scheduler, val_loader,
        )
    else:
        model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader,
        )

    if args.compile:
        m = accelerator.unwrap_model(model)
        m.vae = torch.compile(m.vae, mode="reduce-overhead")
        m.dit = torch.compile(m.dit, mode="reduce-overhead")
        m.encoder2slot = torch.compile(m.encoder2slot, mode="reduce-overhead")

    # ── EMA ──
    ema = None
    if args.enable_ema:
        ema = EMAModel(accelerator.unwrap_model(model), accelerator.device,
                       decay=args.ema_decay)

    # ── Resume ──
    global_step = 0
    start_epoch = 0
    if args.resume and os.path.isdir(args.resume):
        accelerator.load_state(args.resume)
        # Extract step from path like "models/step10000"
        try:
            global_step = int(args.resume.rstrip("/").split("step")[-1])
            start_epoch = global_step // len(train_loader)
        except ValueError:
            pass
        accelerator.print(f"Resumed from {args.resume}, step={global_step}")

    test_num_slots = args.test_num_slots or args.num_slots
    accelerator.init_trackers("semanticist")

    # ── Eval only ──
    if args.eval_only:
        if accelerator.is_main_process:
            m = accelerator.unwrap_model(model)
            generate_samples(
                m, val_loader, accelerator.device,
                os.path.join(args.output_dir, "images", f"eval_step{global_step}.jpg"),
                num_slots=test_num_slots, cfg=args.cfg,
                enc_use_latent=args.enc_use_latent,
            )
        accelerator.wait_for_everyone()
        return

    # ── Training loop ──
    # steps_per_epoch / total_steps count optimizer steps (after grad accum)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = args.num_epochs * steps_per_epoch
    pbar = tqdm(
        initial=global_step, total=total_steps,
        desc="Training", dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()

        loss_accum = {}
        for batch_idx, batch in enumerate(train_loader):
            if args.enc_use_latent:
                # batch is just latents (enc input & diffusion target)
                latents = batch
                images, x_vae = latents, None
            else:
                # batch is (images, latents)
                images, latents = batch
                x_vae = latents

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses = model(images, x_vae, epoch=epoch)
                    loss = sum(losses.values())

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)
                optimizer.step()
                scheduler.step_update(global_step)
                optimizer.zero_grad()

            # Only count a step when gradients are actually synced (accum finished)
            if not accelerator.sync_gradients:
                continue

            # EMA update
            if ema is not None:
                ema.update(accelerator.unwrap_model(model))

            global_step += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                epoch=epoch,
                refresh=False,
            )

            # Logging
            for k, v in losses.items():
                loss_accum.setdefault(k, 0.0)
                loss_accum[k] += v.item()

            if global_step % args.log_every == 0:
                log_dict = {
                    k: v / args.log_every for k, v in loss_accum.items()
                }
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
                log_dict["epoch"] = epoch
                accelerator.log(log_dict, step=global_step)

                if accelerator.is_main_process:
                    loss_str = " ".join(
                        f"{k}={v / args.log_every:.4f}"
                        for k, v in loss_accum.items()
                    )
                    accelerator.print(
                        f"[step {global_step}] epoch={epoch} {loss_str} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                loss_accum = {}

            # Save checkpoint
            if global_step % args.save_every == 0:
                save_dir = os.path.join(args.output_dir, "models",
                                        f"step{global_step}")
                accelerator.save_state(save_dir)
                if ema is not None:
                    ema_path = os.path.join(save_dir, "ema.pt")
                    if accelerator.is_main_process:
                        torch.save(ema.state_dict(), ema_path)

            # Sample
            if global_step % args.sample_every == 0 and accelerator.is_main_process:
                m = accelerator.unwrap_model(model)
                save_path = os.path.join(
                    args.output_dir, "images",
                    f"step_{global_step}_slots{test_num_slots}.jpg",
                )
                generate_samples(m, val_loader, accelerator.device,
                                 save_path, test_num_slots, cfg=1.0,
                                 enc_use_latent=args.enc_use_latent)
                if args.cfg != 1.0:
                    generate_samples(m, val_loader, accelerator.device,
                                     save_path, test_num_slots, cfg=args.cfg,
                                     enc_use_latent=args.enc_use_latent)
                model.train()

            # FID evaluation
            if (global_step % args.fid_every == 0 and global_step > 0):
                m = accelerator.unwrap_model(model)
                evaluate_fid(m, val_ds, accelerator, args,
                             global_step, test_num_slots, cfg=1.0,
                             fid_stats=args.fid_stats,
                             enc_use_latent=args.enc_use_latent)
                if args.cfg != 1.0:
                    evaluate_fid(m, val_ds, accelerator, args,
                                 global_step, test_num_slots, cfg=args.cfg,
                                 fid_stats=args.fid_stats,
                                 enc_use_latent=args.enc_use_latent)
                model.train()

    pbar.close()
    accelerator.end_training()
    accelerator.print("Training finished!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
