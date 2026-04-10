"""
Semanticist-style ViT Encoder for 1D (non-spatial) condition tokens.
====================================================================

Follows the Semanticist paper architecture:
  - ViT-Base backbone (patch_size=16, embed_dim=768, depth=12, heads=12)
  - Learnable slot embeddings appended to patch tokens
  - Causal attention mask on slots (earlier slots can't see later slots)
  - Patches/CLS cannot see slots
  - Output: (B, num_slots, slot_dim) — 1D ordered tokens

Combined with NestedSampler for progressive token dropping:
  - During training: randomly keep first N tokens, replace rest with null
  - Earlier tokens become more informative through this mechanism
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ──────────────────────────────────────────────────────────────────
#  Building blocks (from Semanticist's vision_transformer.py)
# ──────────────────────────────────────────────────────────────────

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None):
        y = self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────
#  Semanticist-style ViT Encoder
# ──────────────────────────────────────────────────────────────────

class SemanticistViTEncoder(nn.Module):
    """ViT encoder with learnable causal slot tokens.

    Architecture follows Semanticist (vit_base_patch16):
      - Patch embedding: img → (num_patches) patch tokens
      - CLS token + slot tokens appended
      - Causal mask: slots attend causally to each other,
        patches/CLS cannot see slots
      - Output: slot tokens projected to slot_dim

    Args:
        img_size: Input image size (default 256)
        patch_size: Patch size (default 16)
        in_chans: Input channels (default 3)
        embed_dim: ViT hidden dimension (default 768)
        depth: Number of transformer blocks (default 12)
        num_heads: Attention heads (default 12)
        mlp_ratio: MLP expansion ratio (default 4.0)
        num_slots: Number of output slot tokens (default 256)
        slot_dim: Output dimension per slot (default 16)
        drop_path_rate: Stochastic depth rate (default 0.1)
        is_causal: Use causal attention mask on slots (default True)
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_slots: int = 256,
        slot_dim: int = 16,
        drop_path_rate: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.is_causal = is_causal
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.slot_embed = nn.Parameter(torch.zeros(1, num_slots, embed_dim))
        # pos_embed covers: CLS (1) + patches + slots
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches + num_slots, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Project from ViT hidden dim to slot_dim
        self.encoder2slot = nn.Linear(embed_dim, slot_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.slot_embed, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)            # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)   # (B, num_patches, embed_dim)
        x = torch.cat([
            self.cls_token.expand(B, -1, -1),
            x,
            self.slot_embed.expand(B, -1, -1),
        ], dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x):
        """
        Args:
            x: (B, 3, img_size, img_size) input image

        Returns:
            slots: (B, num_slots, slot_dim) — 1D ordered condition tokens
        """
        x = self.prepare_tokens(x)
        seq_len = x.shape[1]

        # Build attention mask
        if self.is_causal:
            attn_mask = torch.ones(seq_len, seq_len,
                                   device=x.device, dtype=torch.bool)
            # Slots are causal to each other
            causal_mask = torch.ones(
                self.num_slots, self.num_slots,
                device=x.device, dtype=torch.bool).tril(diagonal=0)
            attn_mask[-self.num_slots:, -self.num_slots:] = causal_mask
            # CLS and patches cannot see slots
            attn_mask[:-self.num_slots, -self.num_slots:] = False
        else:
            attn_mask = None

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        slots = x[:, -self.num_slots:]  # extract slot tokens
        slots = self.encoder2slot(slots)  # project to slot_dim
        return slots


# ──────────────────────────────────────────────────────────────────
#  Nested Sampler (progressive token dropping)
# ──────────────────────────────────────────────────────────────────

class NestedSampler(nn.Module):
    """Semanticist-style progressive dropping.

    During training: randomly sample how many slots to keep per sample.
    Always keeps the first N slots (most informative due to causal ordering).
    Dropped slots are replaced with null_cond externally.
    """

    def __init__(self, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.register_buffer("arange", torch.arange(num_slots))

    def forward(self, batch_size, device, inference_with_n_slots=-1):
        if self.training:
            # Uniform [1, num_slots]
            b = torch.randint(1, self.num_slots + 1,
                              (batch_size,), device=device)
        else:
            if inference_with_n_slots != -1:
                b = torch.full((batch_size,), inference_with_n_slots,
                               device=device)
            else:
                b = torch.full((batch_size,), self.num_slots,
                               device=device)
        b = torch.clamp(b, max=self.num_slots)
        # True = keep, False = drop
        slot_mask = self.arange[None, :] < b[:, None]  # (B, num_slots)
        return slot_mask
