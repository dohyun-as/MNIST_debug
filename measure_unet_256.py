"""
UNet MACs comparison:
  B-ImageNet : train_imagenet_injection.sh  (16×16 latent, [256,512,768,768])
  B-CLEVR    : train_clevr.sh               (256×256 pixel, [128,256,256,512], no attn, FSQ)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from thop import profile

def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    return str(n)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def measure(model, inputs):
    model.eval()
    try:
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return int(macs)
    except Exception as e:
        print(f"  [thop error: {e}]")
        return None


configs = [
    dict(
        label          = "B-ImageNet  (train_imagenet_injection.sh)",
        image_size     = 256,
        in_channels    = 16,
        vae_factor     = 16,          # latent 16×16
        block_out_ch   = [256, 512, 768, 768],
        layers         = 2,
        attn_res       = {8, 4, 2},   # latent-space resolutions
        no_mid_attn    = False,
        cond_use_latent= True,
        min_patch_size = 32,
        feat_channels  = 256,
        depth_per_level= 2,
        cnn_base_ch    = 64,
        use_fsq        = False,
    ),
    dict(
        label          = "B-CLEVR     (train_clevr.sh)",
        image_size     = 256,
        in_channels    = 3,
        vae_factor     = 1,           # pixel space 256×256
        block_out_ch   = [128, 256, 256, 512],
        layers         = 2,
        attn_res       = set(),       # attn_resolutions commented out → no attn
        no_mid_attn    = True,        # --no_mid_attn
        cond_use_latent= False,       # encoder receives raw image
        min_patch_size = 32,
        feat_channels  = 256,
        depth_per_level= 2,
        cnn_base_ch    = 64,
        use_fsq        = True,        # --use_fsq --fsq_levels 8 8 8 8 8 5
    ),
]

from model_multires import MultiResConditionalUNet

print(f"{'='*70}")
print(f"{'Component':<44} {'Params':>10}  {'MACs':>10}")
print(f"{'-'*70}")

for cfg in configs:
    latent_size = cfg['image_size'] // cfg['vae_factor']
    n_blocks    = len(cfg['block_out_ch'])

    # Build block types
    res = latent_size
    down_types = []
    for i in range(n_blocks):
        down_types.append("AttnDownBlock2D" if res in cfg['attn_res'] else "DownBlock2D")
        if i < n_blocks - 1:
            res //= 2
    up_types = ["AttnUpBlock2D" if d == "AttnDownBlock2D" else "UpBlock2D"
                for d in reversed(down_types)]

    unet_config = {
        "block_out_channels": cfg['block_out_ch'],
        "layers_per_block":   cfg['layers'],
        "down_block_types":   down_types,
        "up_block_types":     up_types,
    }
    if cfg['no_mid_attn']:
        unet_config["mid_block_type"] = "UNetMidBlock2D"

    model = MultiResConditionalUNet(
        image_size          = cfg['image_size'],
        in_channels         = cfg['in_channels'],
        cond_in_channels    = cfg['in_channels'],
        vae_downsample_factor= cfg['vae_factor'],
        min_patch_size      = cfg['min_patch_size'],
        feat_channels       = cfg['feat_channels'],
        unet_config         = unet_config,
        depth_per_level     = cfg['depth_per_level'],
        cnn_base_channels   = cfg['cnn_base_ch'],
        level_drop          = True,
        min_keep_levels     = 1,
        cond_use_latent     = cfg['cond_use_latent'],
        use_fsq             = cfg['use_fsq'],
        fsq_levels          = [8,8,8,8,8,5] if cfg['use_fsq'] else None,
    )

    p_enc   = count_params(model.encoder)
    p_unet  = count_params(model.unet)
    p_inj   = (sum(p.numel() for p in model.level_upsamplers.parameters()) +
               sum(p.numel() for p in model.down_injection_convs.parameters()) +
               (sum(p.numel() for p in model.mid_injection_conv.parameters()) if model.mid_injection_conv else 0) +
               sum(p.numel() for p in model.null_cond.parameters()))
    p_total = count_params(model)

    # Input tensors
    in_ch = cfg['in_channels']
    x_t   = torch.zeros(1, in_ch, latent_size, latent_size)
    t     = torch.zeros(1, dtype=torch.long)
    cond  = torch.zeros(1, in_ch, cfg['image_size'] if not cfg['cond_use_latent'] else latent_size,
                                  cfg['image_size'] if not cfg['cond_use_latent'] else latent_size)

    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, t, c): return self.m(x, t, cond_image=c)

    macs_full = measure(Wrapper(model), (x_t, t, cond))

    # UNet-only MACs (zero conditioning)
    class UNetOnly(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, t):
            upsampled = {s: torch.zeros(x.shape[0], self.m.feat_channels,
                                        s * self.m._upsample_factor, s * self.m._upsample_factor)
                         for s in self.m.encoder.level_sizes}
            down_inj = {self.m._encoder_to_block[s]:
                        self.m.down_injection_convs[str(s)](upsampled[s])
                        for s in self.m._encoder_to_block if str(s) in self.m.down_injection_convs}
            mid = None
            if self.m.mid_injection_conv is not None:
                c = min(self.m.encoder.level_sizes)
                mid = self.m.mid_injection_conv(upsampled[c])
            return self.m._forward_unet(x, t, down_inj, mid)

    macs_unet = measure(UNetOnly(model), (x_t, t))

    print(f"\n{cfg['label']}")
    print(f"  Encoder (HierMultiRes)       {fmt(p_enc):>10}")
    print(f"  UNet                         {fmt(p_unet):>10}")
    print(f"  Injection modules            {fmt(p_inj):>10}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  UNet + injection  (no enc)   {fmt(p_unet+p_inj):>10}  {fmt(macs_unet) if macs_unet else 'N/A':>10}")
    print(f"  TOTAL (enc + unet)           {fmt(p_total):>10}  {fmt(macs_full) if macs_full else 'N/A':>10}")
    print(f"  (latent/pixel size: {latent_size}×{latent_size}, block_out: {cfg['block_out_ch']})")
