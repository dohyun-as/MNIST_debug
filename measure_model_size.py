"""
Model size comparison script
Measures parameters and MACs for:
  A) SemantIST (train_semanticist.sh): ViT-B/16 encoder + DiT-L/2 decoder
  B) MultiRes Injection (train_imagenet_injection.sh): HierarchicalEncoder + UNet

Image size: 16x16 latent (from VAE 16x downsample of 256x256)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../semanticist'))

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)

def measure_macs(model, inputs, use_thop=True):
    """Measure MACs using thop (preferred) or fvcore."""
    model = model.eval().cpu()

    if use_thop:
        try:
            from thop import profile
            if isinstance(inputs, (list, tuple)):
                macs, params = profile(model, inputs=inputs, verbose=False)
            else:
                macs, params = profile(model, inputs=(inputs,), verbose=False)
            return int(macs)
        except Exception as e:
            print(f"  [thop failed: {e}]")

    try:
        from fvcore.nn import FlopCountAnalysis
        if isinstance(inputs, (list, tuple)):
            flops = FlopCountAnalysis(model, inputs)
        else:
            flops = FlopCountAnalysis(model, (inputs,))
        flops.unsupported_ops_settings(raise_if_max_unset=False)
        return flops.total() // 2  # FLOPs → MACs
    except Exception as e:
        print(f"  [fvcore failed: {e}]")
        return None


# ══════════════════════════════════════════════════════════════════
#  Model A: SemantIST  (train_semanticist.sh)
#  - Encoder: ViT-B/16   (processes 256×256×3 images)
#  - Decoder: DiT-L/2 with autoenc_cond
#             (processes 16×16×16 latent + 256 slots of dim=16)
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("MODEL A: SemantIST  (train_semanticist.sh)")
print("  Encoder : ViT-B/16  (256×256×3 → 256 slots × dim16)")
print("  Decoder : DiT-L/2   (16×16×16 latent, patch=2, depth=24, dim=1024)")
print("=" * 70)

# ── A1: DiT-L/2 Decoder (without encoder) ──
from semanticist.stage1.diffuse_slot import DiT_with_autoenc_cond_models

dit_l2 = DiT_with_autoenc_cond_models["DiT-L-2"](
    input_size=16,          # 16×16 latent
    in_channels=16,         # 16-ch latent (kl16 VAE)
    num_autoenc=256,        # 256 slots
    autoenc_dim=16,         # slot_dim=16
    use_repa=True,          # REPA projector included
    encoder_depth=8,
    z_dim=768,
)

p_dit_total, p_dit_train = count_params(dit_l2)
print(f"\n[A-Decoder] DiT-L/2")
print(f"  Params (total/trainable): {fmt(p_dit_total)} / {fmt(p_dit_train)}")

# MACs for decoder
# Input: 16×16×16 noisy latent, t, 256 slots of dim 16
x_latent = torch.zeros(1, 16, 16, 16)
t_step   = torch.zeros(1, dtype=torch.long)
slots    = torch.zeros(1, 256, 16)

class DiTWrapper(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit
    def forward(self, x, t, slots):
        return self.dit(x, t, slots)

macs_dit = measure_macs(DiTWrapper(dit_l2), [x_latent, t_step, slots])
if macs_dit:
    print(f"  MACs (16×16 latent, 256 slots): {fmt(macs_dit)}")

# ── A2: Encoder: ViT-B/16 ──
from semanticist.stage1 import vision_transformer

vit_b = vision_transformer.vit_base_patch16(
    img_size=[256],
    num_slots=256,
    drop_path_rate=0.1,
)
encoder2slot = nn.Linear(768, 16)  # encoder2slot: 768 → slot_dim=16

p_enc_total, p_enc_train = count_params(vit_b)
p_e2s_total, _ = count_params(encoder2slot)
print(f"\n[A-Encoder] ViT-B/16 + encoder2slot")
print(f"  ViT-B/16 params: {fmt(p_enc_total)}")
print(f"  encoder2slot params: {fmt(p_e2s_total)}")
print(f"  Encoder total: {fmt(p_enc_total + p_e2s_total)}")

# MACs for encoder (256×256×3 image input)
class EncWrapper(nn.Module):
    def __init__(self, enc, e2s):
        super().__init__()
        self.enc = enc
        self.e2s = e2s
    def forward(self, x):
        slots = self.enc(x, is_causal=True)
        return self.e2s(slots)

x_img = torch.zeros(1, 3, 256, 256)
macs_enc = measure_macs(EncWrapper(vit_b, encoder2slot), x_img)
if macs_enc:
    print(f"  MACs (256×256×3 image): {fmt(macs_enc)}")

# ── A-Total ──
p_a_no_enc = p_dit_total
p_a_with_enc = p_dit_total + p_enc_total + p_e2s_total
print(f"\n[A-TOTAL]")
print(f"  Decoder only  : {fmt(p_a_no_enc)} params")
if macs_dit:
    print(f"                  {fmt(macs_dit)} MACs  (16×16 latent)")
print(f"  Encoder+Decoder: {fmt(p_a_with_enc)} params")
if macs_dit and macs_enc:
    print(f"                   {fmt(macs_dit + macs_enc)} MACs  (16×16 latent + 256×256 image)")


# ══════════════════════════════════════════════════════════════════
#  Model B: MultiRes Injection  (train_imagenet_injection.sh)
#  - Encoder: HierarchicalMultiResEncoder
#             (cond_use_latent=True → processes 16×16×16 latent)
#             levels: [8,4,2,1], feat_channels=256
#  - UNet:    UNet2DConditionModel
#             block_out_channels=[256,512,768,768], layers_per_block=2
#             processes 16×16×16 latent
# ══════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("MODEL B: MultiRes Injection  (train_imagenet_injection.sh)")
print("  Encoder : HierarchicalMultiResEncoder (16×16×16 → 4 levels)")
print("  UNet    : UNet2DConditionModel  [256,512,768,768] × 2")
print("=" * 70)

from model_multires import MultiResConditionalUNet

# Build model B with exact args from the shell script
image_size = 256
in_channels = 16
vae_downsample_factor = 16  # → latent 16×16
min_patch_size = 32
feat_channels = 256
depth_per_level = 2
cnn_base_channels = 64
block_out_channels = [256, 512, 768, 768]
layers_per_block = 2
attn_resolutions = {8, 4, 2}
cond_use_latent = True

latent_size = image_size // vae_downsample_factor  # 16

# Build UNet block types (mirrors build_model in main_multires.py)
n_blocks = len(block_out_channels)
down_types, up_types = [], []
res = latent_size
for i in range(n_blocks):
    if res in attn_resolutions:
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
    "layers_per_block": layers_per_block,
    "down_block_types": down_types,
    "up_block_types": up_types,
}

model_b = MultiResConditionalUNet(
    image_size=image_size,
    in_channels=in_channels,
    cond_in_channels=in_channels,
    vae_downsample_factor=vae_downsample_factor,
    min_patch_size=min_patch_size,
    feat_channels=feat_channels,
    unet_config=unet_config,
    depth_per_level=depth_per_level,
    cnn_base_channels=cnn_base_channels,
    level_drop=True,
    min_keep_levels=1,
    cond_use_latent=cond_use_latent,
)

print(f"\nArchitecture summary:")
print(model_b.describe())

# ── B: UNet only ──
p_unet_total, p_unet_train = count_params(model_b.unet)
# injection convs + upsamplers + null_cond
injection_params = (
    sum(p.numel() for p in model_b.level_upsamplers.parameters()) +
    sum(p.numel() for p in model_b.down_injection_convs.parameters()) +
    (sum(p.numel() for p in model_b.mid_injection_conv.parameters()) if model_b.mid_injection_conv else 0) +
    sum(p.numel() for p in model_b.null_cond.parameters())
)
p_unet_and_inject = p_unet_total + injection_params

print(f"\n[B-UNet only (no encoder)]")
print(f"  UNet params:              {fmt(p_unet_total)}")
print(f"  Injection modules params: {fmt(injection_params)}")
print(f"  UNet + injection total:   {fmt(p_unet_and_inject)}")

# ── B: Encoder ──
p_enc_b_total, _ = count_params(model_b.encoder)
print(f"\n[B-Encoder] HierarchicalMultiResEncoder")
print(f"  Encoder params: {fmt(p_enc_b_total)}")

# ── B-Total ──
p_b_total, p_b_trainable = count_params(model_b)
print(f"\n[B-TOTAL] MultiResConditionalUNet")
print(f"  UNet+injection only : {fmt(p_unet_and_inject)} params")
print(f"  Full model (w/ enc) : {fmt(p_b_total)} params")

# ── B MACs ──
# Without encoder: UNet forward with zero conditioning
class UNetOnlyWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_t, t):
        # Zero conditioning (skip encoder)
        upsampled = {}
        for s in self.m.encoder.level_sizes:
            target_res = s * self.m._upsample_factor
            upsampled[s] = torch.zeros(
                x_t.shape[0], self.m.feat_channels, target_res, target_res,
                device=x_t.device, dtype=x_t.dtype,
            )
        down_injections = {}
        for s, block_idx in self.m._encoder_to_block.items():
            if s in upsampled and str(s) in self.m.down_injection_convs:
                down_injections[block_idx] = self.m.down_injection_convs[str(s)](upsampled[s])
        mid_residual = None
        if self.m.mid_injection_conv is not None:
            coarsest = min(self.m.encoder.level_sizes)
            if coarsest in upsampled:
                mid_residual = self.m.mid_injection_conv(upsampled[coarsest])
        return self.m._forward_unet(x_t, t, down_injections, mid_residual)

x_lat = torch.zeros(1, 16, 16, 16)
t_b   = torch.zeros(1, dtype=torch.long)

print(f"\n[B MACs] measuring...")
macs_unet_only = measure_macs(UNetOnlyWrapper(model_b), [x_lat, t_b])
if macs_unet_only:
    print(f"  UNet+injection only MACs (16×16 latent): {fmt(macs_unet_only)}")

# Full model MACs (encoder + unet)
class FullModelBWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x_t, t, cond):
        return self.m(x_t, t, cond_image=cond)

cond_lat = torch.zeros(1, 16, 16, 16)  # VAE latent as conditioning (cond_use_latent=True)
macs_full_b = measure_macs(FullModelBWrapper(model_b), [x_lat, t_b, cond_lat])
if macs_full_b:
    print(f"  Full model (enc+unet) MACs (16×16 latent): {fmt(macs_full_b)}")


# ══════════════════════════════════════════════════════════════════
#  Summary Table
# ══════════════════════════════════════════════════════════════════

print()
print("=" * 70)
print("SUMMARY  (latent 16×16, 16ch)")
print("=" * 70)
print(f"{'Component':<40} {'Params':>10}  {'MACs':>12}")
print("-" * 70)

# Model A
print(f"[A] SemantIST")
print(f"  {'Decoder (DiT-L/2)':<38} {fmt(p_dit_total):>10}  {fmt(macs_dit) if macs_dit else 'N/A':>12}")
print(f"  {'Encoder (ViT-B/16 + e2slot)':<38} {fmt(p_enc_total + p_e2s_total):>10}  {fmt(macs_enc) if macs_enc else 'N/A':>12}")
print(f"  {'TOTAL (enc+dec)':<38} {fmt(p_a_with_enc):>10}  {fmt(macs_dit + macs_enc) if (macs_dit and macs_enc) else 'N/A':>12}")
print()
# Model B
print(f"[B] MultiRes Injection")
print(f"  {'UNet+injection only':<38} {fmt(p_unet_and_inject):>10}  {fmt(macs_unet_only) if macs_unet_only else 'N/A':>12}")
print(f"  {'Encoder (HierMultiRes)':<38} {fmt(p_enc_b_total):>10}  {'(included below)':>12}")
print(f"  {'TOTAL (enc+unet)':<38} {fmt(p_b_total):>10}  {fmt(macs_full_b) if macs_full_b else 'N/A':>12}")
print()
print("Notes:")
print("  - SemantIST encoder input: 256×256×3 image (MACs for encoder)")
print("  - Injection encoder input: 16×16×16 VAE latent (cond_use_latent=True)")
print("  - MACs = multiply-accumulate ops (1 MAC = 2 FLOPs)")
print("  - VAE and DINOv2 (frozen, non-trainable) are excluded")
print("  - SemantIST REPA projector MLP(1024→2048→768) IS included in decoder")
