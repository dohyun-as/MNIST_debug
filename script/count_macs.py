#!/usr/bin/env python3
"""
MAC (Multiply-Accumulate) counter for three model configurations.

1) ImageNet Injection  — MultiResConditionalUNet, UNet input 16×16  (latent)
2) SemantIST           — ViT-B encoder + DiT-L/2 decoder, DiT input 16×16  (latent)
3) CLEVR               — MultiResConditionalUNet, UNet input 256×256 (pixel)

Usage:
    python script/count_macs.py

Requirements:
    pip install thop   (if not installed)
"""

import sys, os

# ── path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
# semanticist package
SEMANTICIST_ROOT = os.path.join(PROJECT_DIR, "..", "semanticist")
sys.path.insert(0, SEMANTICIST_ROOT)

import torch
import torch.nn as nn

try:
    from thop import profile, clever_format
except ImportError:
    print("thop not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile, clever_format


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ══════════════════════════════════════════════════════════════════
#  1) ImageNet Injection  (input 16×16 latent)
# ══════════════════════════════════════════════════════════════════

def build_imagenet_injection():
    """Matches train_imagenet_injection.sh config."""
    from model_multires import MultiResConditionalUNet

    # Build UNet config (same logic as main_multires.py)
    block_out_channels = [256, 512, 768, 768]
    n_blocks = len(block_out_channels)
    attn_set = {8, 4, 2}  # --attn_resolutions 8 4 2
    latent_size = 256 // 16  # = 16

    down_types, up_types = [], []
    res = latent_size  # 16
    for i in range(n_blocks):
        if res in attn_set:
            down_types.append("AttnDownBlock2D")
        else:
            down_types.append("DownBlock2D")
        if i < n_blocks - 1:
            res //= 2

    up_types = [
        "AttnUpBlock2D" if d == "AttnDownBlock2D" else "UpBlock2D"
        for d in reversed(down_types)
    ]

    unet_config = {
        "block_out_channels": block_out_channels,
        "layers_per_block": 2,
        "down_block_types": down_types,
        "up_block_types": up_types,
    }

    model = MultiResConditionalUNet(
        image_size=256,
        in_channels=16,
        cond_in_channels=3,
        vae_downsample_factor=16,
        min_patch_size=32,
        feat_channels=256,
        unet_config=unet_config,
        depth_per_level=2,
        cnn_base_channels=64,
        cond_use_latent=True,  # encoder receives VAE latent
    )
    return model


def profile_imagenet_injection():
    print("=" * 70)
    print("  1) ImageNet Injection — MultiResConditionalUNet (input 16×16)")
    print("=" * 70)

    model = build_imagenet_injection()
    model.eval()
    print(model.describe())
    print()

    total_p, train_p = count_params(model)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")

    # Inputs: UNet receives 16×16 latent, encoder receives 16×16 latent (cond_use_latent)
    B = 1
    latent_size = 16
    x_t = torch.randn(B, 16, latent_size, latent_size)
    t = torch.tensor([500])
    cond_image = torch.randn(B, 16, latent_size, latent_size)  # cond_use_latent → latent input

    # Wrap for thop (thop only takes *args)
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x_t, t, cond):
            return self.m(x_t, t, cond)

    wrapper = Wrapper(model)
    with torch.no_grad():
        macs, params = profile(wrapper, inputs=(x_t, t, cond_image), verbose=False)

    macs_str, params_str = clever_format([macs, params], "%.3f")
    print(f"MACs:       {macs_str}")
    print(f"Params:     {params_str}")
    print(f"MACs (raw): {macs:,.0f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  2) SemantIST — ViT-B encoder + DiT-L/2 decoder (input 16×16)
# ══════════════════════════════════════════════════════════════════

def build_semanticist_components():
    """Build encoder and decoder separately (skip frozen VAE & REPA)."""
    from semanticist.stage1 import vision_transformer
    from semanticist.stage1.diffuse_slot import DiT_with_autoenc_cond_models

    # ViT-B encoder: img_size=256, patch16 → 256 patches, embed_dim=768
    encoder = vision_transformer.vit_base_patch16(
        img_size=[256],
        num_slots=256,
        drop_path_rate=0.1,
    )
    num_channels = encoder.num_features  # 768
    encoder2slot = nn.Linear(num_channels, 16)  # slot_dim=16

    # DiT-L/2 decoder: input_size=16 (256//16), in_channels=16 (mar-vae-kl16)
    dit = DiT_with_autoenc_cond_models["DiT-L-2"](
        input_size=16,       # 256 // 16
        in_channels=16,      # mar-vae-kl16
        num_autoenc=256,     # num_slots
        autoenc_dim=16,      # slot_dim
        use_repa=True,
        encoder_depth=8,
        z_dim=768,
    )

    return encoder, encoder2slot, dit


def profile_semanticist():
    print("=" * 70)
    print("  2) SemantIST — ViT-B + DiT-L/2 (DiT input 16×16)")
    print("=" * 70)

    encoder, encoder2slot, dit = build_semanticist_components()
    encoder.eval()
    encoder2slot.eval()
    dit.eval()

    B = 1

    # ── Encoder: ViT-B ──
    # Input: 256×256×3 image
    enc_input = torch.randn(B, 3, 256, 256)

    class EncoderWrapper(nn.Module):
        def __init__(self, enc, e2s):
            super().__init__()
            self.enc = enc
            self.e2s = e2s
        def forward(self, x):
            slots = self.enc(x, is_causal=True)
            slots = self.e2s(slots)
            return slots

    enc_wrapper = EncoderWrapper(encoder, encoder2slot)
    with torch.no_grad():
        enc_macs, enc_params = profile(enc_wrapper, inputs=(enc_input,), verbose=False)

    enc_total_p, _ = count_params(enc_wrapper)
    print(f"\n[Encoder] ViT-B (patch16, img=256, 256 slots)")
    print(f"  Parameters: {enc_total_p:,}")
    enc_m, enc_p = clever_format([enc_macs, enc_params], "%.3f")
    print(f"  MACs:       {enc_m}")
    print(f"  MACs (raw): {enc_macs:,.0f}")

    # ── Decoder: DiT-L/2 ──
    # Input: 16×16×16 latent + 256 slots of dim 16
    dit_input = torch.randn(B, 16, 16, 16)  # (B, C, H, W) latent
    t = torch.tensor([500])
    slots = torch.randn(B, 256, 16)  # (B, num_slots, slot_dim)

    class DiTWrapper(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
        def forward(self, x, t, slots):
            return self.d(x, t, slots)

    dit_wrapper = DiTWrapper(dit)
    with torch.no_grad():
        dit_macs, dit_params = profile(dit_wrapper, inputs=(dit_input, t, slots), verbose=False)

    dit_total_p, _ = count_params(dit)
    print(f"\n[Decoder] DiT-L/2 (depth=24, hidden=1024, patch=2)")
    print(f"  Parameters: {dit_total_p:,}")
    dit_m, dit_p = clever_format([dit_macs, dit_params], "%.3f")
    print(f"  MACs:       {dit_m}")
    print(f"  MACs (raw): {dit_macs:,.0f}")

    # ── Total ──
    total_macs = enc_macs + dit_macs
    total_params = enc_total_p + dit_total_p
    t_m, t_p = clever_format([total_macs, total_params], "%.3f")
    print(f"\n[Total] Encoder + Decoder")
    print(f"  Parameters: {total_params:,}")
    print(f"  MACs:       {t_m}")
    print(f"  MACs (raw): {total_macs:,.0f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  3) CLEVR — MultiResConditionalUNet (input 256×256 pixel)
# ══════════════════════════════════════════════════════════════════

def build_clevr():
    """Matches train_clevr.sh config."""
    from model_multires import MultiResConditionalUNet

    block_out_channels = [128, 256, 256, 512]
    n_blocks = len(block_out_channels)

    # no attn_resolutions specified (commented out), all DownBlock2D
    down_types = ["DownBlock2D"] * n_blocks
    up_types = ["UpBlock2D"] * n_blocks

    unet_config = {
        "block_out_channels": block_out_channels,
        "layers_per_block": 2,
        "down_block_types": down_types,
        "up_block_types": up_types,
        "mid_block_type": "UNetMidBlock2D",  # --no_mid_attn
    }

    model = MultiResConditionalUNet(
        image_size=256,
        in_channels=3,
        cond_in_channels=3,
        vae_downsample_factor=1,  # pixel space
        min_patch_size=32,
        feat_channels=256,
        unet_config=unet_config,
        depth_per_level=2,
        cnn_base_channels=64,
        use_fsq=True,
        fsq_levels=[8, 8, 8, 8, 8, 5],
    )
    return model


def profile_clevr():
    print("=" * 70)
    print("  3) CLEVR — MultiResConditionalUNet (input 256×256 pixel)")
    print("=" * 70)

    model = build_clevr()
    model.eval()
    print(model.describe())
    print()

    total_p, train_p = count_params(model)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")

    B = 1
    x_t = torch.randn(B, 3, 256, 256)
    t = torch.tensor([500])
    cond_image = torch.randn(B, 3, 256, 256)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x_t, t, cond):
            return self.m(x_t, t, cond)

    wrapper = Wrapper(model)
    with torch.no_grad():
        macs, params = profile(wrapper, inputs=(x_t, t, cond_image), verbose=False)

    macs_str, params_str = clever_format([macs, params], "%.3f")
    print(f"MACs:       {macs_str}")
    print(f"Params:     {params_str}")
    print(f"MACs (raw): {macs:,.0f}")
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MAC / Parameter Counter")
    print("=" * 70 + "\n")

    profile_imagenet_injection()
    profile_semanticist()
    profile_clevr()

    print("Done!")
