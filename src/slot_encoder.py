"""
Slot Attention encoder — ports the official SlotDiffusion implementation
(https://github.com/Wuziyi616/SlotDiffusion) so that the encoder is a
drop-in replacement for `Baseline1DConditionalDiT.encoder`
(SemanticistViTEncoder).

Pipeline:
    image (B, 3, H, W)
      → ResNet18 (small_inputs=True, GN, optionally use_layer4)
      → feature map (B, C, H', W')          # H'=H/4 or H/8
      → SoftPositionEmbed                    # absolute position info
      → flatten + MLP                        # (B, H'*W', enc_out_channels)
      → SlotAttention (3 iterations)         # (B, K, slot_dim)

Outputs `(B, num_slots, slot_dim)` — same shape contract as
`SemanticistViTEncoder` so `Baseline1DConditionalDiT` works without
modification when we swap `model.encoder = SlotAttentionEncoder(...)`.

References:
- SlotDiffusion / slotdiffusion / img_based / models / slot_attention.py
- SlotDiffusion / video_based / models / resnet.py    (ResNet18 GN-norm)
- SlotDiffusion / video_based / models / utils.py     (SoftPositionEmbed)
- Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020
"""

from __future__ import annotations

import functools
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from PIL import Image
from torchvision.utils import make_grid, save_image


# ──────────────────────────────────────────────────────────────────
#  ResNet18 (small_inputs, GroupNorm) — ported from SlotDiffusion
# ──────────────────────────────────────────────────────────────────

def _gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups, num_channels)


def _conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None, dilation=1):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = _conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNet18Slot(nn.Module):
    """Modified ResNet18 used by SlotDiffusion / SAVi.

    - `small_inputs=True`: stem is 3x3 stride-1 conv + Identity maxpool
      (no early downsampling, suited for ≤256-px scenes).
    - `use_layer4=False` keeps output at /4 (e.g. 256→64). With layer4,
      output is /8 (256→32). For 256×256 CLEVR we want layer4=True
      (32×32=1024 spatial tokens for slot competition).
    - GroupNorm by default (more stable with small batches than BN).
    """

    def __init__(self, small_inputs: bool = True, use_layer4: bool = True,
                 stem_stride: int = 1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(_gn, num_groups=32)
        self._norm = norm_layer
        self.use_layer4 = use_layer4

        self.inplanes = 64
        if small_inputs:
            # SlotDiffusion-default stem. ``stem_stride>1`` halves the early
            # activation footprint without changing the rest of the network
            # — stem & layer1 then run at H/stem_stride resolution.
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride,
                                   padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        if use_layer4:
            self.layer4 = self._make_layer(512, 2, stride=2)
            self.out_channels = 512
        else:
            self.out_channels = 256

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                self._norm(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride,
                              downsample, norm_layer=self._norm)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes,
                                      norm_layer=self._norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.use_layer4:
            x = self.layer4(x)
        return x  # (B, C, H', W')


# ──────────────────────────────────────────────────────────────────
#  SoftPositionEmbed
# ──────────────────────────────────────────────────────────────────

def _build_grid(resolution: Tuple[int, int]) -> Tensor:
    """Returns grid of shape (1, H, W, 4) holding (y, x, 1-y, 1-x)
    normalised to [0, 1]. The "4 channels" encode top/left/bottom/right
    distances and let a single Linear(4, C) learn arbitrary 2-D pos
    embeddings."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)              # (H, W, 2)
    grid = grid.unsqueeze(0)                       # (1, H, W, 2)
    return torch.cat([grid, 1.0 - grid], dim=-1)   # (1, H, W, 4)


class SoftPositionEmbed(nn.Module):
    """Adds learned absolute-position info to a feature map.

    Why: ResNet/CNN encoders are translation-equivariant — the same patch
    content yields the same response regardless of where it lives in the
    image. Slot Attention needs each spatial token to know *where* it is
    so that slots can carry position into their final vectors. ViT-style
    encoders already have positional embeddings, so they don't need this.
    """

    def __init__(self, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.register_buffer('grid', _build_grid(resolution))  # (1, H, W, 4)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (B, C, H, W)
        emb = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb


# ──────────────────────────────────────────────────────────────────
#  Slot Attention module
# ──────────────────────────────────────────────────────────────────

class SlotAttention(nn.Module):
    """Iteratively performs cross-attention from slots to inputs.

    Key trick: ``softmax(dim=-1)`` over the *slot* axis of attention
    logits. Spatial tokens (rows of ``attn``) compete to "claim" slots,
    so each token softly votes for one slot — this competition is what
    drives the object-centric partition.
    """

    def __init__(self, in_features: int, num_iterations: int,
                 num_slots: int, slot_size: int, mlp_hidden_size: int,
                 eps: float = 1e-6):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.eps = eps
        self.attn_scale = slot_size ** -0.5

        self.norm_inputs = nn.LayerNorm(in_features)

        self.project_q = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, slot_size, bias=False)
        self.project_v = nn.Linear(in_features, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size),
        )

    def forward(self, inputs: Tensor, slots: Tensor,
                return_attn: bool = False
                ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: (B, N, in_features) — flattened spatial features
            slots:  (B, K, slot_size)   — initial slots
            return_attn: also return last-iter attn (B, N, K)
        """
        B, N, _ = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # (B, N, slot_size)
        v = self.project_v(inputs)  # (B, N, slot_size)

        attn = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.project_q(slots)                       # (B, K, slot_size)

            # Logits: (B, N, K)
            attn_logits = self.attn_scale * torch.einsum(
                'bnc,bmc->bnm', k, q)
            # ★ softmax OVER SLOTS (the K dim) — competition
            attn = F.softmax(attn_logits, dim=-1)           # (B, N, K)

            # Per-slot weighted mean over inputs
            attn_norm = attn + self.eps
            attn_norm = attn_norm / attn_norm.sum(dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn_norm, v)  # (B, K, slot_size)

            # GRU + residual MLP
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_size),
                slots_prev.reshape(B * self.num_slots, self.slot_size),
            ).reshape(B, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn  # last-iter attn, (B, N, K)
        return slots


# ──────────────────────────────────────────────────────────────────
#  Full encoder
# ──────────────────────────────────────────────────────────────────

class SlotAttentionEncoder(nn.Module):
    """Image → slots. Drop-in replacement for SemanticistViTEncoder.

    Output: ``(B, num_slots, slot_dim)``.

    Args:
        image_size: input resolution (e.g. 256).
        num_slots: K.
        slot_dim: D — slot bottleneck. SlotDiffusion CLEVRTex uses 192.
                  768 is too big; slot is meant to be a compact
                  per-object summary.
        num_iterations: SA iterations (3 default).
        mlp_hidden_size: SA inner MLP. Usually 2 * slot_dim.
        enc_backbone: ``"resnet18"`` or ``"resnet18_no_layer4"``.
        slot_init: ``"learned"`` (default, follows SlotDiffusion) or
                   ``"random"`` (Locatello original).
    """

    def __init__(
        self,
        image_size: int = 256,
        num_slots: int = 16,
        slot_dim: int = 192,
        num_iterations: int = 3,
        mlp_hidden_size: Optional[int] = None,
        enc_backbone: str = "resnet18",
        slot_init: str = "learned",
    ):
        super().__init__()
        if mlp_hidden_size is None:
            mlp_hidden_size = slot_dim * 2

        # ── Backbone ──
        # CNN family — needs explicit SoftPositionEmbed (translation-equiv.).
        #   resnet18            : SlotDiffusion default (128-px tuned).
        #   resnet18_no_layer4  : same but stops at layer3.
        #   resnet18_strided    : stem stride-2 (recommended for ≥256-px;
        #                         same 32×32 grid, ~4× less stem memory).
        # ViT family — patch position embedding is built-in; no SoftPos needed.
        #   vit_b16             : trainable ViT-B/16 from scratch (256
        #                         patches × 768 dim for 256-px input).
        #   dinov1_b16          : frozen pretrained DINOv1 ViT-B/16. Almost
        #                         free in memory (no backward), strong
        #                         object-emergence prior.
        self.enc_backbone = enc_backbone
        self._is_vit = enc_backbone in ("vit_b16", "dinov1_b16",
                                        "dinov2_base")
        self._is_frozen = enc_backbone in ("dinov1_b16", "dinov2_base")

        if enc_backbone == "resnet18":
            self.cnn = _ResNet18Slot(small_inputs=True, use_layer4=True,
                                     stem_stride=1)
            self.feat_resolution = (image_size // 8, image_size // 8)
        elif enc_backbone == "resnet18_no_layer4":
            self.cnn = _ResNet18Slot(small_inputs=True, use_layer4=False,
                                     stem_stride=1)
            self.feat_resolution = (image_size // 4, image_size // 4)
        elif enc_backbone == "resnet18_strided":
            self.cnn = _ResNet18Slot(small_inputs=True, use_layer4=False,
                                     stem_stride=2)
            self.feat_resolution = (image_size // 8, image_size // 8)
        elif enc_backbone == "vit_b16":
            self.cnn = _VitB16Slot(image_size=image_size, patch_size=16,
                                   embed_dim=768, depth=12, num_heads=12,
                                   mlp_ratio=4.0, dropout=0.0)
            self.feat_resolution = (self.cnn.grid_size, self.cnn.grid_size)
        elif enc_backbone == "dinov1_b16":
            self.cnn = _DinoFrozen("facebook/dino-vitb16")
            # DINO ViT-B/16 expects 16-stride patches; for 256 input → 16×16.
            assert image_size % 16 == 0, \
                "dinov1_b16 requires image_size divisible by 16"
            self.feat_resolution = (image_size // 16, image_size // 16)
        elif enc_backbone == "dinov2_base":
            # DINOv2 patch_size=14. For 256-px input the wrapper internally
            # resizes to the nearest multiple of 14 (252) and uses
            # interpolate_pos_encoding=True. Output grid: 18×18 = 324 tokens.
            self.cnn = _Dinov2Frozen("facebook/dinov2-base",
                                     image_size=image_size, patch_size=14)
            self.feat_resolution = (self.cnn.feat_grid, self.cnn.feat_grid)
        else:
            raise ValueError(f"Unknown enc_backbone: {enc_backbone}")
        feat_channels = self.cnn.out_channels

        # ── Position embedding + MLP head before SlotAttention ──
        # SoftPositionEmbed is only needed for translation-equivariant CNN
        # backbones. ViT already injects 2-D position info via its own
        # learnable patch position embeddings, so we skip it.
        self.pos_emb = (None if self._is_vit
                        else SoftPositionEmbed(feat_channels,
                                               self.feat_resolution))
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(feat_channels),
            nn.Linear(feat_channels, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim),
        )

        # ── Slot init: learnable parameter (SlotDiffusion-style) ──
        # Locatello's original samples slots from N(mu, sigma) per image;
        # SlotDiffusion uses a single learnable init for all images, which
        # is more stable empirically. We follow SlotDiffusion.
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.slot_init = slot_init
        # Slot Attention is permutation-invariant — no causal ordering.
        # `is_causal` is consumed by Baseline1DConditionalDiT.describe()
        # so we expose it for compatibility.
        self.is_causal = False
        if slot_init == "learned":
            self.init_latents = nn.Parameter(
                torch.empty(1, num_slots, slot_dim))
            nn.init.normal_(self.init_latents, std=0.02)
        elif slot_init == "random":
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
            nn.init.xavier_uniform_(self.slots_logsigma)
        else:
            raise ValueError(f"Unknown slot_init: {slot_init}")

        # ── Slot attention ──
        self.slot_attention = SlotAttention(
            in_features=slot_dim,
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=slot_dim,
            mlp_hidden_size=mlp_hidden_size,
        )

    def _init_slots(self, B: int, device, dtype) -> Tensor:
        if self.slot_init == "learned":
            return self.init_latents.expand(B, -1, -1).to(device=device,
                                                          dtype=dtype)
        # random per image
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
        return (mu + sigma * torch.randn_like(mu)).to(device=device,
                                                     dtype=dtype)

    def _features(self, images: Tensor) -> Tensor:
        # images: (B, 3, H, W)
        if self._is_vit:
            # ViT path: output is already (B, N, C) with patch pos baked in.
            # Frozen DINO runs no_grad (its own forward decorator handles it).
            feat = self.cnn(images)                       # (B, N, C)
        else:
            # CNN path: feature map → SoftPositionEmbed → flatten.
            feat = self.cnn(images)                       # (B, C, H', W')
            feat = self.pos_emb(feat)                     # absolute pos
            feat = feat.flatten(2).transpose(1, 2)        # (B, H'*W', C)
        feat = self.encoder_out_layer(feat)               # (B, N, slot_dim)
        return feat

    def forward(self, images: Tensor,
                return_attn: bool = False
                ) -> Tensor | Tuple[Tensor, Tensor]:
        feats = self._features(images)
        slots_init = self._init_slots(images.shape[0], images.device,
                                      images.dtype)
        if return_attn:
            slots, attn = self.slot_attention(feats, slots_init,
                                              return_attn=True)
            return slots, attn  # (B, K, D), (B, H'*W', K)
        return self.slot_attention(feats, slots_init)


# ──────────────────────────────────────────────────────────────────
#  Visualization: attention → segmentation map
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
#  ViT backbones
# ──────────────────────────────────────────────────────────────────

class _PatchEmbed(nn.Module):
    """Conv-based patchify (image → flat patch sequence)."""

    def __init__(self, image_size: int, patch_size: int, in_chans: int,
                 embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size {image_size} not divisible by patch_size {patch_size}"
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                                  # (B, C, h, w)
        return x.flatten(2).transpose(1, 2)               # (B, N, C)


class _ViTBlock(nn.Module):
    """Standard ViT pre-norm transformer block (GELU MLP, no drop_path)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True,
                                          dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _VitB16Slot(nn.Module):
    """Plain ViT-B/16 used as the spatial-feature backbone for slot
    attention. No CLS token, no slot tokens — just patch tokens. Position
    embedding is the standard learned 1-D table over patch indices, which
    already encodes 2-D position because patch order is fixed.

    For 256 input with patch_size=16: 16×16=256 patch tokens, 768 dim.
    """

    def __init__(self, image_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = _PatchEmbed(image_size, patch_size, 3, embed_dim)
        N = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, N, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_channels = embed_dim
        self.num_patches = N
        self.grid_size = self.patch_embed.grid_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                                          # (B, N, embed_dim)


class _DinoFrozen(nn.Module):
    """Frozen DINOv1 ViT-B/16 from HuggingFace.

    Returns last hidden states EXCLUDING the [CLS] token, so the output
    shape is (B, num_patches, embed_dim). For 256 input + patch_size=16:
    256 patches, 768 dim. No backward through DINO → near-zero activation
    cost during training of the slot stack on top.
    """

    def __init__(self, model_name: str = "facebook/dino-vitb16"):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.out_channels = self.encoder.config.hidden_size

    def train(self, mode: bool = True):
        # Always keep DINO in eval (BN/dropout). Only return self for chaining.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder(pixel_values=x)
        # last_hidden_state: (B, 1+N, C) — drop the CLS at index 0
        return out.last_hidden_state[:, 1:, :]            # (B, N, C)


class _Dinov2Frozen(nn.Module):
    """Frozen DINOv2-base from HuggingFace.

    DINOv2 uses patch_size=14 (native 224 / 518 input). For arbitrary
    input we resize to the nearest multiple of 14 ≤ input_size and pass
    ``interpolate_pos_encoding=True`` so the model bilinearly interpolates
    its learned position table.

    For 256-px input → resized to 252×252 → 18×18 = 324 patch tokens,
    768 dim.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base",
                 image_size: int = 256, patch_size: int = 14):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.out_channels = self.encoder.config.hidden_size

        # Round image_size DOWN to nearest multiple of patch_size.
        target = (image_size // patch_size) * patch_size
        self.target_size = target
        self.feat_grid = target // patch_size
        self.input_size = image_size

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        out = self.encoder(pixel_values=x,
                           interpolate_pos_encoding=True)
        # (B, 1+N, C) → drop CLS
        return out.last_hidden_state[:, 1:, :]            # (B, N, C)


def _color_palette(K: int) -> torch.Tensor:
    """Distinguishable colors for K slots, deterministic."""
    import colorsys
    colors = []
    for i in range(K):
        h = (i / max(K, 1)) % 1.0
        s = 0.7 + 0.3 * ((i % 3) / 2)        # 0.7, 0.85, 1.0
        v = 0.7 + 0.3 * ((i % 5) / 4)        # 0.7..1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([r, g, b])
    return torch.tensor(colors, dtype=torch.float32)  # (K, 3) in [0, 1]


@torch.no_grad()
def visualize_slot_segmentation(
    encoder: SlotAttentionEncoder,
    images: Tensor,
    save_path: str,
    image_size: Optional[int] = None,
    nrow: int = 1,
    save_per_slot_masks: bool = True,
):
    """Save a [original | seg map | overlay] grid plus optionally
    per-slot soft-mask heatmaps.

    Args:
        encoder: SlotAttentionEncoder.
        images: (B, 3, H, W) in [-1, 1].
        save_path: PNG path for the main grid.
        save_per_slot_masks: if True, also saves
            ``<save_path>.slot_masks.png`` showing per-slot attention.
    """
    encoder.eval()
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    images = images.to(device=device, dtype=dtype)

    slots, attn = encoder(images, return_attn=True)
    # attn: (B, N, K) where N = H'*W'
    B, N, K = attn.shape
    H_feat, W_feat = encoder.feat_resolution
    assert N == H_feat * W_feat, f"{N} != {H_feat * W_feat}"

    H_img, W_img = images.shape[-2], images.shape[-1]
    if image_size is None:
        image_size = H_img

    # Per-pixel argmax over slots
    seg = attn.argmax(dim=-1).view(B, H_feat, W_feat)              # (B, H', W')
    seg_up = F.interpolate(
        seg.float().unsqueeze(1),
        size=(H_img, W_img),
        mode='nearest',
    ).squeeze(1).long().cpu()                                       # (B, H, W)

    palette = _color_palette(K)                                     # (K, 3)
    seg_rgb = palette[seg_up]                                       # (B, H, W, 3)
    seg_rgb = seg_rgb.permute(0, 3, 1, 2).contiguous()              # (B, 3, H, W)

    img_01 = (images.cpu().float() * 0.5 + 0.5).clamp(0, 1)
    overlay = (0.55 * img_01 + 0.45 * seg_rgb).clamp(0, 1)

    # Concat width-wise: [original | seg | overlay]
    triplet = torch.cat([img_01, seg_rgb, overlay], dim=-1)         # (B, 3, H, 3W)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save_image(make_grid(triplet, nrow=nrow, padding=2), save_path)

    if save_per_slot_masks:
        # Per-slot attention heatmap (soft masks).
        attn_2d = attn.view(B, H_feat, W_feat, K).permute(0, 3, 1, 2)
        attn_up = F.interpolate(
            attn_2d.reshape(B * K, 1, H_feat, W_feat),
            size=(H_img, W_img), mode='bilinear', align_corners=False,
        ).reshape(B, K, H_img, W_img).cpu().float()
        # Normalise per (b, k) for display
        attn_min = attn_up.amin(dim=(-1, -2), keepdim=True)
        attn_max = attn_up.amax(dim=(-1, -2), keepdim=True)
        attn_n = (attn_up - attn_min) / (attn_max - attn_min + 1e-6)
        attn_rgb = attn_n.unsqueeze(2).expand(-1, -1, 3, -1, -1)     # (B, K, 3, H, W)
        # Colorize each slot's mask with its palette colour
        col = palette.view(1, K, 3, 1, 1)
        attn_rgb = attn_rgb * col + (1 - attn_rgb) * 1.0  # white where low

        # Lay out as (rows = batch, cols = slots)
        flat = attn_rgb.reshape(B * K, 3, H_img, W_img)
        grid = make_grid(flat, nrow=K, padding=2)
        masks_path = save_path.replace(".png", ".slot_masks.png")
        if masks_path == save_path:
            masks_path = save_path + ".slot_masks.png"
        save_image(grid, masks_path)
