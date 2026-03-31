#!/usr/bin/env python
"""
Sample from a trained multi-res diffusion checkpoint with various CFG scales.

Usage:
  python script/sample_cfg_sweep.py \
    --ckpt runs/clevr_256_varied_multires_fsq/checkpoints/step_0190000/checkpoint.pt \
    --data_dir /workspace/NAS/sr_diffusion/project/clevr-dataset-gen/output/clevr_256_varied/images \
    --cfg_scales 0 1.0 1.5 2.0 2.5 3.0 \
    --num_steps 50 \
    --n_samples 8 \
    --device cuda:0
"""

import argparse
import os
import sys
import types

import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from diffusers import DDIMScheduler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_model_from_ckpt(ckpt_path, device="cuda"):
    """Rebuild model from checkpoint args and load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]

    # Convert dict → namespace (fill missing keys with defaults)
    defaults = dict(
        encoder_type="cnn",
        vit_patch_size=4, vit_depth=4, vit_num_heads=4,
        vit_mlp_ratio=4.0, vit_use_cnn_stem=True, vit_no_cnn_stem=False,
        vit_cnn_stem_reduction=4,
        use_flow_matching=False,
    )
    for k, v in defaults.items():
        saved_args.setdefault(k, v)

    args = types.SimpleNamespace(**saved_args)

    # ── Build model (same logic as main_multires.py build_model) ──
    from model_multires import MultiResConditionalUNet
    import json

    if args.unet_config is not None:
        if os.path.isfile(args.unet_config):
            with open(args.unet_config) as f:
                unet_config = json.load(f)
        else:
            unet_config = json.loads(args.unet_config)
    else:
        n_blocks = len(args.block_out_channels)
        attn_set = set(args.attn_resolutions) if args.attn_resolutions else set()
        latent_size = args.image_size // args.vae_downsample_factor

        down_types, up_types = [], []
        res = latent_size
        for i in range(n_blocks):
            if res in attn_set:
                down_types.append("AttnDownBlock2D")
            else:
                down_types.append("DownBlock2D")
            if i < n_blocks - 1:
                res = res // 2

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
        if getattr(args, "no_mid_attn", False):
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
        cond_use_latent=getattr(args, "cond_use_latent", False),
        mae_mask_ratio=getattr(args, "mae_mask_ratio", 0.0),
        encoder_type=args.encoder_type,
        vit_patch_size=args.vit_patch_size,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_ratio=args.vit_mlp_ratio,
        vit_use_cnn_stem=args.vit_use_cnn_stem and not args.vit_no_cnn_stem,
        vit_cnn_stem_reduction=args.vit_cnn_stem_reduction,
        use_fsq=getattr(args, "use_fsq", False),
        fsq_levels=getattr(args, "fsq_levels", None),
        fsq_drop_quant_p=getattr(args, "fsq_drop_quant_p", 0.0),
        fsq_corrupt_tokens_p=getattr(args, "fsq_corrupt_tokens_p", 0.0),
        use_vq=getattr(args, "use_vq", False),
        vq_codebook_size=getattr(args, "vq_codebook_size", 512),
        vq_beta=getattr(args, "vq_beta", 0.25),
    )

    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, args


@torch.no_grad()
def sample_ddim(model, scheduler, cond_images, num_steps=50,
                guidance_scale=1.5, in_channels=3,
                num_active_levels=None):
    """DDIM sampling with CFG.

    guidance_scale=0  → pure unconditional (return_uncond only)
    guidance_scale=1  → conditional, no guidance
    guidance_scale>1  → classifier-free guidance
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
            # Pure unconditional
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               return_uncond=True)
        elif guidance_scale != 1.0:
            # CFG
            noise_cond = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)
            noise_uncond = model(latents, t_batch, cond_image=cond_images,
                                 return_uncond=True)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            # Conditional, no guidance
            noise_pred = model(latents, t_batch, cond_image=cond_images,
                               num_active_levels=num_active_levels)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Image directory (ImageFolder compatible)")
    parser.add_argument("--cfg_scales", type=float, nargs="+",
                        default=[0, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir (default: <ckpt_dir>/../cfg_sweep)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 for inference")
    cli = parser.parse_args()

    # ── Output dir ──
    if cli.output_dir is None:
        ckpt_parent = os.path.dirname(os.path.dirname(cli.ckpt))
        step_name = os.path.basename(os.path.dirname(cli.ckpt))
        cli.output_dir = os.path.join(ckpt_parent, "cfg_sweep", step_name)
    os.makedirs(cli.output_dir, exist_ok=True)
    print(f"Output dir: {cli.output_dir}")

    # ── Load model ──
    print("Loading model...")
    model, train_args = load_model_from_ckpt(cli.ckpt, device=cli.device)
    dtype = torch.bfloat16 if cli.bf16 else torch.float32
    print(f"  image_size={train_args.image_size}, "
          f"latent_size={model.latent_size}, "
          f"num_levels={model.num_levels}, "
          f"prediction_type={train_args.prediction_type}")

    # ── Scheduler ──
    scheduler = DDIMScheduler(
        num_train_timesteps=train_args.num_train_timesteps,
        beta_schedule=train_args.beta_schedule,
        prediction_type=train_args.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
    )

    # ── Dataset & sample selection ──
    tfm = transforms.Compose([
        transforms.Resize(train_args.image_size),
        transforms.CenterCrop(train_args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(cli.data_dir, transform=tfm)
    rng = torch.Generator().manual_seed(cli.seed)
    indices = torch.randperm(len(ds), generator=rng)[:cli.n_samples]
    images = torch.stack([ds[i][0] for i in indices]).to(cli.device)
    # Keep images in float32; autocast handles the rest

    # Save GT
    gt_01 = (images.float() * 0.5 + 0.5).clamp(0, 1)
    save_image(make_grid(gt_01, nrow=cli.n_samples, padding=2),
               os.path.join(cli.output_dir, "gt.png"))
    print(f"Saved GT ({cli.n_samples} images)")

    # ── Sample for each CFG scale ──
    # Use same noise seed for all scales for fair comparison
    for cfg in cli.cfg_scales:
        print(f"\nSampling with cfg_scale={cfg:.1f}, steps={cli.num_steps} ...")
        torch.manual_seed(cli.seed)  # Same starting noise

        with torch.autocast("cuda", dtype=dtype):
            samples = sample_ddim(
                model, scheduler, images,
                num_steps=cli.num_steps,
                guidance_scale=cfg,
                in_channels=train_args.in_channels,
            )

        # Save per-cfg grid: interleaved [GT, Gen] pairs
        gen_01 = (samples.float() * 0.5 + 0.5).clamp(0, 1)
        combined = torch.stack([gt_01, gen_01], dim=1).reshape(
            -1, 3, train_args.image_size, train_args.image_size)
        grid = make_grid(combined, nrow=min(8, cli.n_samples * 2), padding=2)
        fname = f"cfg_{cfg:.1f}.png"
        save_image(grid, os.path.join(cli.output_dir, fname))
        print(f"  → saved {fname}")

    # ── Also make a summary grid: rows=cfg scales, cols=samples ──
    print("\nGenerating summary grid (rows=cfg, cols=samples)...")
    all_rows = [gt_01]  # First row = GT
    for cfg in cli.cfg_scales:
        torch.manual_seed(cli.seed)
        with torch.autocast("cuda", dtype=dtype):
            samples = sample_ddim(
                model, scheduler, images,
                num_steps=cli.num_steps,
                guidance_scale=cfg,
                in_channels=train_args.in_channels,
            )
        all_rows.append((samples.float() * 0.5 + 0.5).clamp(0, 1))

    # Stack: (n_rows, n_samples, C, H, W) → (n_rows * n_samples, C, H, W)
    all_images = torch.cat(all_rows, dim=0)  # (n_rows * n_samples, C, H, W)
    summary_grid = make_grid(all_images, nrow=cli.n_samples, padding=2)
    save_image(summary_grid, os.path.join(cli.output_dir, "summary.png"))
    labels = ["GT"] + [f"cfg={c:.1f}" for c in cli.cfg_scales]
    print(f"  → saved summary.png  (rows: {', '.join(labels)})")

    print(f"\nDone! All outputs in: {cli.output_dir}")


if __name__ == "__main__":
    main()
