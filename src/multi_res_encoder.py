"""
Hierarchical Multi-Resolution Image Condition Encoder
=====================================================

Three encoder backends:

1. **CNN** (default, ``encoder_type='cnn'``):
   Patchify → per-patch CNN (독립, cross-patch 교류 없음) → merge hierarchy.

2. **ViT** (``encoder_type='vit'``):
   각 level을 독립적으로 처리.  Image를 grid cell로 나누고, 각 cell을
   shared ViT로 encoding 후 average-pool → 1 feature per cell.
   모든 level에서 cell 내부 token 수가 동일하도록 ViT patch size를
   자동 조절:
     Level 8×8: cell 32×32, vit_patch 4  → 8×8 = 64 tokens
     Level 4×4: cell 64×64, vit_patch 8  → 8×8 = 64 tokens
     Level 2×2: cell 128×128, vit_patch 16 → 8×8 = 64 tokens
     Level 1×1: cell 256×256, vit_patch 32 → 8×8 = 64 tokens

   Options:
     - ``vit_use_cnn_stem``: 큰 patch에 대해 CNN stem 적용 (권장)
     - ``mae_mask_ratio``: training 시 cell 내부 token MAE masking
       (inference 시 masking 없이 전체 token 사용)

3. **ViT-Global (single-forward)** (``encoder_type='vit_global'``):
   전체 이미지를 **한 번만** ViT로 처리하여 finest grid 크기의 spatial
   tokens를 생성한 뒤, coarser level은 avg pooling으로 파생.
     Patchify → ViT forward (한 번) → (B, dim, tps, tps)
     Level 8×8: tps×tps → avg_pool(tps/8) → (B, dim, 8, 8)
     Level 4×4: tps×tps → avg_pool(tps/4) → (B, dim, 4, 4)
     Level 2×2: tps×tps → avg_pool(tps/2) → (B, dim, 2, 2)
     Level 1×1: tps×tps → avg_pool(tps)   → (B, dim, 1, 1)

   ViT를 한 번만 돌리므로 cell-based 대비 훨씬 저렴 (~23 GFLOPs vs 469).
   Options:
     - ``vit_init_clip``: CLIP ViT weights로 patch_embed / pos_emb /
       transformer 12 layers 초기화 (requires 256×256 image, patch=16,
       dim=768, depth=12, heads=12, no CNN stem).
   MAE masking 사용 불가 (avg pool에 모든 token 필요).

Configurable parameters:
  - image_size: 원본 이미지 크기 (e.g. 256)
  - min_patch_size: 가장 작은 patch 크기 (e.g. 32)
    → finest grid = image_size / min_patch_size (e.g. 8×8)
  - num_levels: merge 횟수 + 1 (e.g. 4 → 8×8, 4×4, 2×2, 1×1)
  - vae_downsample_factor: VAE compression ratio (e.g. 8)
    → UNet latent size = image_size / vae_downsample_factor
  - unet_down_blocks: UNet down block 수
    → bottleneck resolution 계산에 사용
  - upsample_factor: 자동 계산 또는 수동 지정
    → encoder level × upsample_factor = UNet resolution에 매칭
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
#  Building blocks
# ──────────────────────────────────────────────────────────────────

class ResBlock2D(nn.Module):
    """Residual block with optional downsampling."""

    def __init__(self, c_in, c_out, down=False, groups=8):
        super().__init__()
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.act = nn.SiLU(inplace=True)
        self.skip = None
        if down or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)

    def forward(self, x):
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        s = self.skip(x) if self.skip is not None else x
        return self.act(h + s)


class PatchCNN(nn.Module):
    """Per-patch CNN: min_patch_size×min_patch_size → 1×1 feature vector.

    독립 처리 — 다른 patch의 정보에 접근 불가.
    stride-2 conv를 반복해서 spatial을 1×1로 줄임.
    """

    def __init__(self, in_channels: int, dim: int, min_patch_size: int,
                 base_channels: int = 64):
        super().__init__()
        n_down = int(math.log2(min_patch_size))
        assert 2 ** n_down == min_patch_size, \
            f"min_patch_size must be power of 2, got {min_patch_size}"

        layers = []
        ch = in_channels
        for i in range(n_down):
            ch_out = min(dim, base_channels * (2 ** i))
            layers.append(ResBlock2D(ch, ch_out, down=True))
            ch = ch_out
        # Final projection to dim
        layers.append(nn.Conv2d(ch, dim, 1))
        layers.append(nn.GroupNorm(min(8, dim), dim))
        layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N_patches, C, patch_h, patch_w)
        Returns: (B*N_patches, dim, 1, 1)
        """
        return self.net(x)


class MergeLayer(nn.Module):
    """2×2 spatial merge: concat 4 neighbors → linear projection.

    (B, N, D) with spatial (h, w) → (B, N/4, D).
    Swin Transformer의 Patch Merging과 동일한 원리.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, D = x.shape
        assert N == h * w
        assert h % 2 == 0 and w % 2 == 0

        x = x.view(B, h, w, D)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        merged = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, h/2, w/2, 4D)
        merged = merged.view(B, -1, 4 * D)
        merged = self.norm(merged)
        return self.proj(merged)


class MLPBlock(nn.Module):
    """Per-token MLP (no cross-token interaction)."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


# ──────────────────────────────────────────────────────────────────
#  CellViT: shared ViT that encodes one grid cell → 1 feature vector
# ──────────────────────────────────────────────────────────────────

class ConvStem(nn.Module):
    """Lightweight CNN stem for large ViT patches (e.g. 32×32).

    Reduces spatial resolution by ``reduction`` (power of 2) while
    increasing channels, so the subsequent linear patch projection
    operates on smaller, richer feature maps.
    """

    def __init__(self, in_channels: int, embed_dim: int, reduction: int = 4):
        super().__init__()
        assert reduction >= 2 and (reduction & (reduction - 1)) == 0
        n_stages = int(math.log2(reduction))
        layers = []
        ch = in_channels
        for i in range(n_stages):
            ch_out = embed_dim // (2 ** max(0, n_stages - 1 - i))
            ch_out = max(ch_out, 32)
            layers.extend([
                nn.Conv2d(ch, ch_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, ch_out), ch_out),
                nn.SiLU(inplace=True),
            ])
            ch = ch_out
        self.net = nn.Sequential(*layers)
        self.out_channels = ch
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellViT(nn.Module):
    """Shared ViT encoder for per-cell processing across all hierarchy levels.

    Each grid cell (variable spatial size) is split into sub-patches,
    projected to tokens, run through a transformer, then average-pooled
    to produce a single feature vector.

    The ViT patch size at each level is computed so that the number of
    tokens per cell is constant:
        tokens_per_side = cell_size_finest / vit_patch_size
    For coarser levels, the effective patch size scales proportionally.

    Parameters
    ----------
    in_channels : int
        Input image channels.
    dim : int
        Output feature dimension (also transformer hidden dim).
    vit_patch_size : int
        ViT patch size for the **finest** level (e.g. 4).
    depth : int
        Number of transformer encoder layers.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Transformer MLP expansion ratio.
    max_tokens : int
        Maximum number of tokens per cell (for positional embedding).
    use_cnn_stem : bool
        If True, apply CNN stem before patch projection (recommended
        for large effective patch sizes ≥ 16).
    cnn_stem_reduction : int
        Spatial reduction factor of the CNN stem (power of 2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 256,
        vit_patch_size: int = 4,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_tokens: int = 256,
        use_cnn_stem: bool = True,
        cnn_stem_reduction: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.vit_patch_size = vit_patch_size
        self.use_cnn_stem = use_cnn_stem

        # CNN stem (optional): reduces effective patch size for large patches
        if use_cnn_stem:
            self.cnn_stem = ConvStem(in_channels, dim, reduction=cnn_stem_reduction)
            self._stem_reduction = cnn_stem_reduction
            stem_ch = self.cnn_stem.out_channels
        else:
            self.cnn_stem = None
            self._stem_reduction = 1
            stem_ch = in_channels

        # Patch projection (linear): patch_dim → dim
        # actual patch_dim depends on effective patch size (after stem),
        # so we use a Conv2d that adapts at forward time
        # We store stem_ch and compute projection dynamically? No —
        # since tokens_per_side is constant, the effective patch size
        # after stem is also constant = vit_patch_size / stem_reduction.
        # But vit_patch_size is for finest level; coarser levels scale up.
        # → We need the projection to handle variable patch_dim.
        # Solution: use a Conv2d with stride = effective_patch_size applied
        # at forward time.  Simpler: just use a Linear.
        self._stem_ch = stem_ch
        # We'll compute patch_dim at forward time and cache the projection
        # Actually, since the *number of tokens* is constant across levels,
        # and we apply the stem, the effective patch size after stem varies.
        # To keep a single projection: we always adaptive-pool or resize
        # the stem output so that patch_size_after_stem = vit_patch_size.
        # Then patch_dim is always stem_ch * vit_patch_size^2.
        #
        # Better approach: use a strided Conv2d as patch projection.
        # This naturally handles any cell size: we apply stem (optional),
        # then Conv2d with kernel=vit_patch_size, stride=vit_patch_size.
        # Number of output tokens = (cell_after_stem) / vit_patch_size.
        # For finest: cell=32, stem→8, conv stride vit_patch_size=4 → 2 tokens? No.
        #
        # Let's keep it simple: Linear projection, compute patch_dim per call.
        # Since the actual patch pixel size changes per level, we use
        # nn.Linear(patch_dim_max, dim) with zero-padded input? No, ugly.
        #
        # Cleanest: one nn.Linear per unique patch_dim? Too many.
        #
        # Actually the simplest: just use a Conv2d projection like standard ViT.
        # The Conv2d kernel_size = stride = effective_vit_patch_size.
        # Since effective_vit_patch_size varies per level, we can't use Conv2d.
        #
        # → Use nn.Linear(patch_dim, dim).  At each level the patch_dim
        #   = stem_ch * eff_p * eff_p.  This varies per level.  So we need
        #   per-level projections or a single adaptive one.
        #
        # User's design: token count per cell is CONSTANT across levels.
        # Finest: cell=32, vit_p=4 → tokens_per_side=8, tokens=64
        # Next:   cell=64, vit_p=8 → tokens_per_side=8, tokens=64
        # ...
        # With CNN stem (reduction=4):
        #   Finest: cell=32 →stem→ 8, need 64 tokens → eff_p = 8/8 = 1
        #   Next:   cell=64 →stem→ 16, need 64 tokens → eff_p = 16/8 = 2
        #   Next:   cell=128→stem→ 32, need 64 tokens → eff_p = 32/8 = 4
        #   Coarsest: cell=256→stem→64, need 64 tokens → eff_p = 64/8 = 8
        # patch_dim varies: 1*1*stem_ch, 2*2*stem_ch, 4*4*stem_ch, 8*8*stem_ch
        #
        # Without stem:
        #   Finest: vit_p=4,   patch_dim = 4*4*3 = 48
        #   Next:   vit_p=8,   patch_dim = 8*8*3 = 192
        #   ...
        #   Coarsest: vit_p=32, patch_dim = 32*32*3 = 3072
        #
        # So patch_dim IS different per level.  We need per-level projections.
        # This is fine — it's just a linear layer per level.

        # We'll create projections lazily or compute at init if level info given.
        # For now, store None and build in the encoder's __init__ after levels known.
        self.patch_projs = nn.ModuleDict()  # str(level_size) → nn.Linear

        # Per-level scale embedding: tells the shared transformer which
        # resolution level it is processing (e.g. 8×8 local vs 1×1 global).
        # Registered per level via build_scale_emb(), broadcast-added to all tokens.
        self.scale_embs = nn.ParameterDict()

        # Positional embedding (max_tokens)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def register_level(self, level_size: int, cell_size: int):
        """Register a hierarchy level so we can create its patch projection.

        Called by the parent encoder during __init__.
        """
        if self.use_cnn_stem:
            eff_cell = cell_size // self._stem_reduction
        else:
            eff_cell = cell_size

        # tokens_per_side for this level
        tokens_per_side_finest = None  # computed from finest
        # Actually we just need: eff_vit_patch = eff_cell / tokens_per_side
        # tokens_per_side is determined by finest level.
        # But we don't know tokens_per_side here.  The parent will pass it.
        # Let's compute patch_dim = stem_ch * eff_vit_patch^2
        # We store the cell_size → eff_vit_patch mapping externally.
        # For simplicity, let the parent call build_projection directly.
        pass

    def build_projection(self, key: str, patch_dim: int):
        """Build a linear projection for a specific patch dimension."""
        self.patch_projs[key] = nn.Linear(patch_dim, self.dim)

    def build_scale_emb(self, key: str):
        """Build a learnable scale embedding for a hierarchy level."""
        self.scale_embs[key] = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.scale_embs[key], std=0.02)

    def forward(
        self,
        cells: torch.Tensor,
        level_key: str,
        vit_patch_size: int,
        mae_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Encode a batch of grid cells → feature vectors.

        Args:
            cells: (N, C, cell_h, cell_w) — all cells from one level,
                   across all batch elements.
            level_key: str key for the patch projection.
            vit_patch_size: effective ViT patch size for this level
                (after CNN stem if used).
            mae_mask_ratio: fraction of tokens to mask (training only).

        Returns:
            (N, dim) — one feature vector per cell.
        """
        N, C, H, W = cells.shape

        # Optional CNN stem
        if self.cnn_stem is not None:
            cells = self.cnn_stem(cells)  # (N, stem_ch, H', W')
            _, C, H, W = cells.shape

        p = vit_patch_size
        assert H % p == 0 and W % p == 0, \
            f"After stem: ({H},{W}) not divisible by vit_patch {p}"
        n_h, n_w = H // p, W // p
        n_tokens = n_h * n_w

        # Patchify: (N, C, H, W) → (N, n_tokens, C*p*p)
        # unfold approach
        tokens = cells.unfold(2, p, p).unfold(3, p, p)  # (N, C, n_h, n_w, p, p)
        tokens = tokens.contiguous().view(N, C, n_tokens, p, p)
        tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()  # (N, n_tokens, C, p, p)
        tokens = tokens.view(N, n_tokens, C * p * p)  # (N, n_tokens, patch_dim)

        # Linear projection
        tokens = self.patch_projs[level_key](tokens)  # (N, n_tokens, dim)

        # Positional embedding
        tokens = tokens + self.pos_emb[:, :n_tokens, :]

        # Scale embedding: broadcast to all tokens in this level
        if level_key in self.scale_embs:
            tokens = tokens + self.scale_embs[level_key]

        # MAE masking: drop random tokens (training only)
        if self.training and mae_mask_ratio > 0:
            n_keep = n_tokens - int(n_tokens * mae_mask_ratio)
            n_keep = max(n_keep, 1)
            noise = torch.rand(N, n_tokens, device=tokens.device)
            ids_keep = noise.argsort(dim=1)[:, :n_keep]
            ids_keep, _ = ids_keep.sort(dim=1)
            tokens = torch.gather(
                tokens, 1,
                ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        # Transformer
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # Average pool → (N, dim)
        feat = tokens.mean(dim=1)
        return feat


# ──────────────────────────────────────────────────────────────────
#  Swin Transformer encoder components
# ──────────────────────────────────────────────────────────────────

def _swin_window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    return (x.view(B, H // ws, ws, W // ws, ws, C)
             .permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C))


def _swin_window_unpartition(w: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    nH, nW_ = H // ws, W // ws
    B = w.shape[0] // (nH * nW_)
    return (w.view(B, nH, nW_, ws, ws, -1)
             .permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1))


class SwinWindowAttention(nn.Module):
    """Window multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.ws = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.rpb_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        rel = coords.flatten(1)[:, :, None] - coords.flatten(1)[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpb_index", rel.sum(-1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B_, N, C = x.shape
        hd = C // self.num_heads
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rpb_table[self.rpb_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, N, N)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    """Swin block: LN → W-MSA (optionally shifted) → LN → MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.ws = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SwinWindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        ws = self.ws
        # No shift when single window covers entire feature map
        shift = self.shift_size if min(H, W) > ws else 0

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        if shift > 0:
            x = torch.roll(x, (-shift, -shift), (1, 2))
            mask = self._attn_mask(Hp, Wp, shift, x.device)
        else:
            mask = None

        x = _swin_window_partition(x, ws).view(-1, ws * ws, C)
        x = self.attn(x, mask)
        x = x.view(-1, ws, ws, C)
        x = _swin_window_unpartition(x, ws, Hp, Wp)

        if shift > 0:
            x = torch.roll(x, (shift, shift), (1, 2))
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :]

        x = shortcut + x.reshape(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x

    def _attn_mask(self, Hp: int, Wp: int, shift: int, device):
        ws = self.ws
        m = torch.zeros(1, Hp, Wp, 1, device=device)
        for cnt, (h, w) in enumerate(
            (h, w)
            for h in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            for w in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        ):
            m[:, h, w, :] = cnt
        mw = _swin_window_partition(m, ws).view(-1, ws * ws)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)


class SwinPatchMerge(nn.Module):
    """2×2 spatial merge: resolution halved, channels doubled."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


# ──────────────────────────────────────────────────────────────────
#  Hierarchical Multi-Resolution Encoder
# ──────────────────────────────────────────────────────────────────

class HierarchicalMultiResEncoder(nn.Module):
    """
    Patchify → per-patch CNN → merge hierarchy encoder.

    Two backends:

    ``encoder_type='cnn'`` (default):
        PatchCNN → merge hierarchy (original behaviour).

    ``encoder_type='vit'``:
        각 level을 독립적으로 처리.  Image → grid cells → shared CellViT
        → avg pool → 1 feature per cell.  Merge layers는 사용하지 않음.

    Parameters
    ----------
    in_channels : int
        입력 이미지 채널 수
    dim : int
        token feature dimension
    image_size : int
        원본 이미지 크기 (정사각형)
    min_patch_size : int
        가장 작은 patch 크기 (e.g. 32). Must be power of 2.
        finest grid = image_size / min_patch_size
    num_levels : int or None
        피라미드 레벨 수. None이면 1×1까지 자동 계산.
    depth_per_level : int or list[int]
        각 레벨의 MLP block 수 (CNN mode only)
    mlp_ratio : float
        MLP expansion ratio
    cnn_base_channels : int
        PatchCNN의 base channel 수 (CNN mode only)
    encoder_type : str
        'cnn' or 'vit'
    vit_patch_size : int
        ViT sub-patch size for finest level (ViT mode)
    vit_depth : int
        Transformer depth (ViT mode)
    vit_num_heads : int
        Attention heads (ViT mode)
    vit_mlp_ratio : float
        Transformer MLP ratio (ViT mode)
    vit_use_cnn_stem : bool
        CNN stem before patch projection (ViT mode, 큰 patch 권장)
    vit_cnn_stem_reduction : int
        CNN stem spatial reduction (ViT mode)
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 256,
        image_size: int = 256,
        min_patch_size: int = 32,
        num_levels: int | None = None,
        depth_per_level: int | list[int] = 2,
        mlp_ratio: float = 4.0,
        cnn_base_channels: int = 64,
        mae_mask_ratio: float = 0.0,
        # ── ViT-specific ──
        encoder_type: str = 'cnn',
        vit_patch_size: int = 4,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_use_cnn_stem: bool = True,
        vit_cnn_stem_reduction: int = 4,
        # ── Internal dim (encoder runs at this, projects to dim at output) ──
        encoder_internal_dim: int | None = None,
        # ── Swin-specific ──
        swin_patch_size: int = 16,
        swin_embed_dim: int = 96,
        swin_depths: list[int] | None = None,
        swin_num_heads: list[int] | None = None,
        swin_window_size: int = 4,
        swin_mlp_ratio: float = 4.0,
        # ── Custom level sizes (optional) ──
        level_sizes: list[int] | None = None,
        # ── CLIP initialization for vit_global (optional) ──
        vit_init_clip: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch16",
    ):
        super().__init__()

        if level_sizes is not None:
            # Custom level sizes (e.g. [9, 3, 1] for sudoku)
            # Validate: each level must evenly divide image_size
            for s in level_sizes:
                assert image_size % s == 0, \
                    f"image_size({image_size}) must be divisible by level_size({s})"
            self.level_sizes = sorted(level_sizes, reverse=True)
            finest_size = self.level_sizes[0]
            num_levels = len(self.level_sizes)
            # Derive min_patch_size from finest level
            min_patch_size = image_size // finest_size
        else:
            finest_size = image_size // min_patch_size
            assert finest_size >= 1, \
                f"image_size({image_size}) must be >= min_patch_size({min_patch_size})"
            assert finest_size & (finest_size - 1) == 0, \
                f"finest_size({finest_size}) must be power of 2"

            if num_levels is None:
                num_levels = int(math.log2(finest_size)) + 1
            assert num_levels >= 1

            self.level_sizes = [finest_size // (2 ** i) for i in range(num_levels)]
            self.level_sizes = [s for s in self.level_sizes if s >= 1]
            num_levels = len(self.level_sizes)

        self.num_levels = num_levels
        self.dim = dim
        self._enc_dim = encoder_internal_dim or dim  # internal encoder dim
        self.image_size = image_size
        self.min_patch_size = min_patch_size
        self.finest_size = finest_size
        self.feat_channels = dim
        self.mae_mask_ratio = mae_mask_ratio
        self.encoder_type = encoder_type

        if encoder_type == 'cnn':
            assert level_sizes is None, \
                "CNN encoder does not support custom level_sizes (requires power-of-2)"
            self._init_cnn(in_channels, dim, min_patch_size, num_levels,
                           depth_per_level, mlp_ratio, cnn_base_channels)
        elif encoder_type == 'vit':
            self._init_vit(in_channels, dim, image_size, min_patch_size,
                           vit_patch_size, vit_depth, vit_num_heads,
                           vit_mlp_ratio, vit_use_cnn_stem,
                           vit_cnn_stem_reduction)
        elif encoder_type == 'swin':
            self._init_swin(in_channels, dim, image_size,
                            swin_patch_size, swin_embed_dim,
                            swin_depths or [2, 2, 6, 2],
                            swin_num_heads or [3, 6, 12, 24],
                            swin_window_size, swin_mlp_ratio)
        elif encoder_type == 'vit_global':
            assert mae_mask_ratio == 0.0, \
                "vit_global does not support MAE masking (all spatial tokens " \
                "are needed for avg pooling to derive coarser levels)"
            self._init_vit_global(in_channels, dim, image_size,
                                  vit_patch_size, vit_depth, vit_num_heads,
                                  vit_mlp_ratio, vit_use_cnn_stem,
                                  vit_cnn_stem_reduction)
            if vit_init_clip:
                self.init_from_clip(clip_model_name)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def _init_cnn(self, in_channels, dim, min_patch_size, num_levels,
                  depth_per_level, mlp_ratio, cnn_base_channels):
        if isinstance(depth_per_level, int):
            depths = [depth_per_level] * num_levels
        else:
            depths = list(depth_per_level)
            assert len(depths) == num_levels

        # ── 1) Per-patch CNN ──
        self.patch_cnn = PatchCNN(
            in_channels=in_channels,
            dim=dim,
            min_patch_size=min_patch_size,
            base_channels=cnn_base_channels,
        )

        # ── 2) Per-level MLP blocks (no self-attention!) ──
        self.level_mlps = nn.ModuleList()
        self.level_norms = nn.ModuleList()
        for lvl in range(num_levels):
            blocks = nn.Sequential(*[
                MLPBlock(dim, mlp_ratio) for _ in range(depths[lvl])
            ])
            self.level_mlps.append(blocks)
            self.level_norms.append(nn.LayerNorm(dim))

        # ── 3) Merge layers ──
        self.merge_layers = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.merge_layers.append(MergeLayer(dim))

    def _init_vit(self, in_channels, dim, image_size, min_patch_size,
                  vit_patch_size, vit_depth, vit_num_heads, vit_mlp_ratio,
                  vit_use_cnn_stem, vit_cnn_stem_reduction):
        """Initialize ViT encoder backend.

        All levels share a single CellViT.  Each level has its own
        patch projection (because effective patch dims differ).

        When encoder_internal_dim is set (self._enc_dim != dim), the ViT
        runs at _enc_dim internally and a per-level 1×1 conv projects to dim.
        """
        enc_d = self._enc_dim  # internal dim (may differ from output dim)

        finest_cell = min_patch_size  # cell size at finest level
        tokens_per_side = finest_cell // vit_patch_size
        assert finest_cell % vit_patch_size == 0, \
            f"min_patch_size({min_patch_size}) must be divisible by " \
            f"vit_patch_size({vit_patch_size})"
        self._tokens_per_side = tokens_per_side
        self._vit_patch_size = vit_patch_size

        max_tokens = tokens_per_side * tokens_per_side

        self.cell_vit = CellViT(
            in_channels=in_channels,
            dim=enc_d,
            vit_patch_size=vit_patch_size,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            max_tokens=max_tokens,
            use_cnn_stem=vit_use_cnn_stem,
            cnn_stem_reduction=vit_cnn_stem_reduction,
        )

        # Compute per-level effective ViT patch size (after stem) and
        # build linear projections
        stem_r = vit_cnn_stem_reduction if vit_use_cnn_stem else 1
        stem_ch = self.cell_vit._stem_ch if vit_use_cnn_stem else in_channels

        self._level_vit_info = {}  # level_size → (eff_vit_patch, level_key)
        for s in self.level_sizes:
            cell_size = image_size // s  # pixel size of one cell
            eff_cell = cell_size // stem_r  # after CNN stem
            eff_p = eff_cell // tokens_per_side  # effective patch size after stem
            assert eff_p >= 1, \
                f"Level {s}: eff_cell({eff_cell}) / tokens_per_side({tokens_per_side}) < 1. " \
                f"Reduce vit_cnn_stem_reduction or vit_patch_size."
            patch_dim = stem_ch * eff_p * eff_p
            key = str(s)
            self.cell_vit.build_projection(key, patch_dim)
            self.cell_vit.build_scale_emb(key)
            self._level_vit_info[s] = (eff_p, key)

        # Per-level positional embedding for the grid (2D) — at internal dim
        self.grid_pos_embs = nn.ParameterDict()
        for s in self.level_sizes:
            self.grid_pos_embs[str(s)] = nn.Parameter(
                torch.zeros(1, enc_d, s, s))
            nn.init.trunc_normal_(self.grid_pos_embs[str(s)], std=0.02)

        # Output projection: enc_dim → dim (when they differ)
        if enc_d != dim:
            self._vit_out_projs = nn.ModuleDict()
            for s in self.level_sizes:
                self._vit_out_projs[str(s)] = nn.Conv2d(enc_d, dim, 1)
        else:
            self._vit_out_projs = None

    # ──────────────────────────────────────────────────────────────
    #  Swin backend
    # ──────────────────────────────────────────────────────────────

    def _init_swin(self, in_channels, dim, image_size,
                   swin_patch_size, swin_embed_dim, swin_depths,
                   swin_num_heads, swin_window_size, swin_mlp_ratio):
        """Initialize Swin Transformer encoder backend.

        Architecture (image_size=256, swin_patch_size=16):
            PatchEmbed → 16×16 tokens, C=embed_dim
            Stage 0 (16×16) → avg_pool 2×2 → 8×8 → proj → level 8
              PatchMerge → 8×8, C=2D
            Stage 1 (8×8)  → avg_pool 2×2 → 4×4 → proj → level 4
              PatchMerge → 4×4, C=4D
            Stage 2 (4×4)  → avg_pool 2×2 → 2×2 → proj → level 2
              PatchMerge → 2×2, C=8D
            Stage 3 (2×2)  → avg_pool 2×2 → 1×1 → proj → level 1
        """
        num_stages = len(swin_depths)
        assert num_stages == self.num_levels, \
            f"len(swin_depths)={num_stages} must equal num_levels={self.num_levels}"
        assert len(swin_num_heads) == num_stages

        initial_res = image_size // swin_patch_size
        assert image_size % swin_patch_size == 0

        # Validate: stage resolution = 2 × target level_size
        res = initial_res
        for i, ls in enumerate(self.level_sizes):
            assert res == 2 * ls, \
                f"Stage {i}: resolution {res} != 2 × level_size {ls}. " \
                f"Adjust swin_patch_size (current={swin_patch_size})."
            if i < num_stages - 1:
                res //= 2

        self._swin_patch_size = swin_patch_size
        self._swin_initial_res = initial_res

        # Patch embedding
        self.swin_patch_embed = nn.Conv2d(
            in_channels, swin_embed_dim,
            kernel_size=swin_patch_size, stride=swin_patch_size)
        self.swin_patch_norm = nn.LayerNorm(swin_embed_dim)

        # Per-stage: blocks, merge, output norm + projection
        self.swin_blocks = nn.ModuleList()
        self.swin_merges = nn.ModuleList()
        self.swin_out_norms = nn.ModuleList()
        self.swin_out_projs = nn.ModuleList()

        stage_res = initial_res
        ch = swin_embed_dim
        for i in range(num_stages):
            ws = min(swin_window_size, stage_res)

            # Transformer blocks (alternating W-MSA / SW-MSA)
            blocks = nn.ModuleList()
            for j in range(swin_depths[i]):
                shift = 0 if (j % 2 == 0) else ws // 2
                blocks.append(SwinTransformerBlock(
                    ch, swin_num_heads[i], ws, shift, swin_mlp_ratio))
            self.swin_blocks.append(blocks)

            # Output: norm → linear → feat_channels
            self.swin_out_norms.append(nn.LayerNorm(ch))
            self.swin_out_projs.append(nn.Linear(ch, dim))

            # PatchMerge (except last stage)
            if i < num_stages - 1:
                self.swin_merges.append(SwinPatchMerge(ch))
                ch *= 2
                stage_res //= 2

    def _forward_swin(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Single-forward Swin encoder → multi-resolution features."""
        B = x.shape[0]

        # Patch embed
        x = self.swin_patch_embed(x)           # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, embed_dim)
        x = self.swin_patch_norm(x)

        H = W = self._swin_initial_res
        level_features = {}

        for i, (blocks, level_size) in enumerate(
            zip(self.swin_blocks, self.level_sizes)
        ):
            for blk in blocks:
                x = blk(x, H, W)

            # Extract feature at this stage: norm → proj → reshape → avg_pool
            feat = self.swin_out_projs[i](self.swin_out_norms[i](x))
            feat = feat.transpose(1, 2).view(B, self.dim, H, W)
            feat = F.avg_pool2d(feat, 2)        # (B, dim, H/2, W/2)
            level_features[level_size] = feat

            # Merge for next stage (except last)
            if i < len(self.swin_merges):
                x = self.swin_merges[i](x, H, W)
                H //= 2
                W //= 2

        return level_features

    # ──────────────────────────────────────────────────────────────
    #  ViT-Global backend: single-forward shared ViT + avg pool
    # ──────────────────────────────────────────────────────────────

    def _init_vit_global(self, in_channels, dim, image_size,
                         vit_patch_size, vit_depth, vit_num_heads,
                         vit_mlp_ratio, vit_use_cnn_stem,
                         vit_cnn_stem_reduction):
        """Initialize global ViT encoder (single forward).

        전체 이미지를 ViT로 **한 번만** 처리한 뒤, level마다 avg pool만
        다르게 적용.  Compute = 1× ViT forward (cell-based 대비 ~5× 저렴).

        When ``encoder_internal_dim`` is set (self._enc_dim != dim), the ViT
        runs at _enc_dim internally (e.g. 768 for CLIP) and a per-level 1×1
        conv projects to `dim` (output feat_channels) as an information
        bottleneck — matches the ``vit`` mode design.

        예) image=256, no stem, vit_patch=16, enc_d=768, dim=16:
          ViT forward (한 번) → (B, 768, 16, 16)
          Level 8: avg_pool(2)  → (B, 768, 8, 8) → Conv2d(768→16) → (B, 16, 8, 8)
          Level 4: avg_pool(4)  → (B, 768, 4, 4) → Conv2d(768→16) → (B, 16, 4, 4)
          Level 2: avg_pool(8)  → (B, 768, 2, 2) → Conv2d(768→16) → (B, 16, 2, 2)
          Level 1: avg_pool(16) → (B, 768, 1, 1) → Conv2d(768→16) → (B, 16, 1, 1)
        """
        enc_d = self._enc_dim  # transformer internal dim (e.g. 768 for CLIP)

        # CNN stem (optional) — CLIP init 시에는 반드시 비활성 (CLIP은 plain
        # patch_embed Conv만 가지고 있음)
        if vit_use_cnn_stem:
            self._global_stem = ConvStem(
                in_channels, enc_d, reduction=vit_cnn_stem_reduction)
            stem_ch = self._global_stem.out_channels
            stem_r = vit_cnn_stem_reduction
        else:
            self._global_stem = None
            stem_ch = in_channels
            stem_r = 1

        # Token grid: image_size / stem / vit_patch
        tokens_per_side = image_size // (stem_r * vit_patch_size)
        assert tokens_per_side >= self.finest_size, \
            f"tokens_per_side({tokens_per_side}) must be >= " \
            f"finest_size({self.finest_size}). Reduce vit_patch_size " \
            f"or vit_cnn_stem_reduction."
        for s in self.level_sizes:
            assert tokens_per_side % s == 0, \
                f"tokens_per_side({tokens_per_side}) must be divisible " \
                f"by level_size({s})"

        self._global_tokens_per_side = tokens_per_side
        self._global_vit_patch_size = vit_patch_size
        n_tokens = tokens_per_side * tokens_per_side

        # Patch projection: flatten (C, p, p) → enc_d
        patch_dim = stem_ch * vit_patch_size * vit_patch_size
        self._global_patch_proj = nn.Linear(patch_dim, enc_d)

        # Positional embedding (1D, over all patches) — at internal dim
        self._global_pos_emb = nn.Parameter(torch.zeros(1, n_tokens, enc_d))
        nn.init.trunc_normal_(self._global_pos_emb, std=0.02)

        # Shared transformer (pre-norm, GELU) — single forward for all levels
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_d,
            nhead=vit_num_heads,
            dim_feedforward=int(enc_d * vit_mlp_ratio),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self._global_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=vit_depth)
        self._global_norm = nn.LayerNorm(enc_d)

        # Per-level 2D grid positional embedding — at INTERNAL dim (enc_d),
        # added BEFORE output projection. Matches the `vit` mode pattern.
        self.grid_pos_embs = nn.ParameterDict()
        for s in self.level_sizes:
            self.grid_pos_embs[str(s)] = nn.Parameter(
                torch.zeros(1, enc_d, s, s))
            nn.init.trunc_normal_(self.grid_pos_embs[str(s)], std=0.02)

        # Output projection: enc_d → dim (bottleneck, per-level Conv2d 1×1).
        # Applied AFTER grid_pos_embs — same order as `vit` mode.
        if enc_d != dim:
            self._vit_out_projs = nn.ModuleDict()
            for s in self.level_sizes:
                self._vit_out_projs[str(s)] = nn.Conv2d(enc_d, dim, 1)
        else:
            self._vit_out_projs = None

    def _patchify_global(self, x: torch.Tensor) -> torch.Tensor:
        """Stem (optional) + patchify + linear proj + pos_emb → token sequence.

        Returns:
            (B, N, dim) tokens ready for transformer.
        """
        if self._global_stem is not None:
            x = self._global_stem(x)

        B, C, _, _ = x.shape
        p = self._global_vit_patch_size
        tps = self._global_tokens_per_side

        # Flatten patches in (C, p_row, p_col) order to match CLIP Conv2d
        # weight layout when reshaped to (out, in*p*p).
        tokens = x.unfold(2, p, p).unfold(3, p, p)          # (B, C, tps, tps, p, p)
        tokens = tokens.contiguous().view(B, C, tps * tps, p, p)
        tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()  # (B, N, C, p, p)
        tokens = tokens.view(B, tps * tps, C * p * p)        # (B, N, patch_dim)

        tokens = self._global_patch_proj(tokens) + self._global_pos_emb
        return tokens

    def _forward_vit_global(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Global ViT forward (single pass) + per-level avg pool (+ project).

        Compute: 1× ViT-B/16 forward on (B, 256, enc_d) ≈ 23 GFLOPs.

        Args:
            x: (B, C, H, W) conditioning image

        Returns:
            {spatial_size: (B, dim, S, S)} for each level, where `dim` is the
            output feat_channels (post-bottleneck when enc_d != dim).
        """
        B = x.shape[0]
        tps = self._global_tokens_per_side
        enc_d = self._enc_dim

        # Patchify + single transformer forward (at internal dim)
        tokens = self._patchify_global(x)              # (B, tps², enc_d)
        tokens = self._global_transformer(tokens)
        tokens = self._global_norm(tokens)             # (B, tps², enc_d)

        # Reshape to 2D grid
        feat = tokens.transpose(1, 2).view(B, enc_d, tps, tps)

        # Per-level: avg pool → + 2D pos emb (at enc_d) → (optional) Conv2d bottleneck
        # Order matches the `vit` mode: pos_emb at internal dim, then project.
        level_features = {}
        for s in self.level_sizes:
            pool_k = tps // s
            pooled = F.avg_pool2d(feat, kernel_size=pool_k) if pool_k > 1 else feat
            pooled = pooled + self.grid_pos_embs[str(s)]        # at enc_d
            if self._vit_out_projs is not None:
                pooled = self._vit_out_projs[str(s)](pooled)    # enc_d → dim
            level_features[s] = pooled

        return level_features

    # ──────────────────────────────────────────────────────────────
    #  CLIP weight loading (for vit_global)
    # ──────────────────────────────────────────────────────────────

    def init_from_clip(self, model_name: str = "openai/clip-vit-base-patch16"):
        """Load CLIP ViT weights into vit_global encoder.

        Maps:
          CLIP patch_embedding (Conv2d)  → _global_patch_proj (Linear)
          CLIP position_embedding (197,D) → _global_pos_emb (interpolated
                                           to tps×tps, CLS dropped)
          CLIP encoder.layers[i]          → _global_transformer.layers[i]
            q/k/v_proj, out_proj, layer_norm1/2, mlp.fc1/fc2
          CLIP post_layernorm             → _global_norm

        Requirements (raises AssertionError otherwise):
          - encoder_type == 'vit_global'
          - No CNN stem (CLIP uses plain Conv2d patch embed)
          - internal dim (`encoder_internal_dim` or `feat_channels`) ==
            CLIP hidden size (768 for ViT-B)
          - vit_patch_size == CLIP patch size (16 for ViT-B/16)
          - depth / heads match CLIP (12 / 12 for ViT-B)

        The per-level output projection `_vit_out_projs` (if present, i.e.
        when enc_d != dim) is left at random init — it is the bottleneck
        head that maps CLIP features → feat_channels.

        Note on minor architectural differences:
          - CLIP has `pre_layrnorm` before the encoder; we don't.
          - CLIP applies `post_layernorm` only to CLS; we apply `_global_norm`
            to all patch tokens (init'd from CLIP's post_layernorm).
          - CLIP uses QuickGELU inside MLP; we use GELU.
          These shift the initial distribution slightly but the 12 transformer
          layers carry the vast majority of pretrained knowledge and training
          adapts the rest.
        """
        assert self.encoder_type == 'vit_global', \
            f"init_from_clip requires encoder_type='vit_global', got {self.encoder_type}"
        assert self._global_stem is None, \
            "init_from_clip requires --vit_no_cnn_stem (CLIP has no CNN stem)"

        try:
            from transformers import CLIPVisionModel
        except ImportError as e:
            raise ImportError(
                "init_from_clip requires the `transformers` package. "
                "Install with: pip install transformers"
            ) from e

        clip = CLIPVisionModel.from_pretrained(model_name)
        vm = clip.vision_model  # CLIPVisionTransformer

        clip_hidden = vm.config.hidden_size
        clip_depth = vm.config.num_hidden_layers
        clip_heads = vm.config.num_attention_heads
        clip_patch = vm.config.patch_size

        assert self._enc_dim == clip_hidden, \
            f"internal dim mismatch: model._enc_dim={self._enc_dim}, " \
            f"CLIP hidden={clip_hidden}. Set --encoder_internal_dim " \
            f"{clip_hidden} (or --feat_channels {clip_hidden} if no bottleneck)."
        assert self._global_vit_patch_size == clip_patch, \
            f"patch size mismatch: model={self._global_vit_patch_size}, " \
            f"CLIP={clip_patch}. Set --vit_patch_size {clip_patch}."
        assert len(self._global_transformer.layers) == clip_depth, \
            f"depth mismatch: model={len(self._global_transformer.layers)}, " \
            f"CLIP={clip_depth}. Set --vit_depth {clip_depth}."

        with torch.no_grad():
            # ── 1) Patch embedding: Conv2d (D, C, p, p) → Linear (D, C*p*p) ──
            conv_w = vm.embeddings.patch_embedding.weight  # (D, C, p, p)
            D = conv_w.shape[0]
            self._global_patch_proj.weight.copy_(conv_w.reshape(D, -1))
            if self._global_patch_proj.bias is not None:
                if vm.embeddings.patch_embedding.bias is not None:
                    self._global_patch_proj.bias.copy_(
                        vm.embeddings.patch_embedding.bias)
                else:
                    self._global_patch_proj.bias.zero_()

            # ── 2) Positional embedding (skip CLS, bicubic interpolate) ──
            pe = vm.embeddings.position_embedding.weight   # (197, D) for 224px
            patch_pe = pe[1:]                               # drop CLS → (196, D)
            src_side = int(math.isqrt(patch_pe.shape[0]))
            assert src_side * src_side == patch_pe.shape[0], \
                f"CLIP pos_emb not square-gridded: {patch_pe.shape[0]} tokens"
            tgt_side = self._global_tokens_per_side
            patch_pe = patch_pe.view(1, src_side, src_side, D).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(
                patch_pe, size=(tgt_side, tgt_side),
                mode='bicubic', align_corners=False,
            )
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, tgt_side * tgt_side, D)
            self._global_pos_emb.data.copy_(patch_pe)

            # ── 3) Transformer layers ──
            for i, our_layer in enumerate(self._global_transformer.layers):
                cl = vm.encoder.layers[i]

                # Attention: concat Q/K/V into in_proj; copy out_proj
                qkv_w = torch.cat([
                    cl.self_attn.q_proj.weight,
                    cl.self_attn.k_proj.weight,
                    cl.self_attn.v_proj.weight,
                ], dim=0)
                qkv_b = torch.cat([
                    cl.self_attn.q_proj.bias,
                    cl.self_attn.k_proj.bias,
                    cl.self_attn.v_proj.bias,
                ], dim=0)
                our_layer.self_attn.in_proj_weight.copy_(qkv_w)
                our_layer.self_attn.in_proj_bias.copy_(qkv_b)
                our_layer.self_attn.out_proj.weight.copy_(cl.self_attn.out_proj.weight)
                our_layer.self_attn.out_proj.bias.copy_(cl.self_attn.out_proj.bias)

                # MLP (fc1/fc2)
                our_layer.linear1.weight.copy_(cl.mlp.fc1.weight)
                our_layer.linear1.bias.copy_(cl.mlp.fc1.bias)
                our_layer.linear2.weight.copy_(cl.mlp.fc2.weight)
                our_layer.linear2.bias.copy_(cl.mlp.fc2.bias)

                # LayerNorms (pre-norm: norm1 before attn, norm2 before mlp)
                our_layer.norm1.weight.copy_(cl.layer_norm1.weight)
                our_layer.norm1.bias.copy_(cl.layer_norm1.bias)
                our_layer.norm2.weight.copy_(cl.layer_norm2.weight)
                our_layer.norm2.bias.copy_(cl.layer_norm2.bias)

            # ── 4) Post-LN (init from CLIP's CLS post_layernorm) ──
            self._global_norm.weight.copy_(vm.post_layernorm.weight)
            self._global_norm.bias.copy_(vm.post_layernorm.bias)

        del clip
        print(f"[vit_global] Initialized from CLIP: {model_name} "
              f"(patch={clip_patch}, dim={clip_hidden}, depth={clip_depth}, "
              f"heads={clip_heads}, pos_emb {src_side}×{src_side}→{tgt_side}×{tgt_side})")

    # ──────────────────────────────────────────────────────────────
    #  CNN backend: patchify + hierarchy
    # ──────────────────────────────────────────────────────────────

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Image → patches → per-patch CNN → tokens.

        (B, C, H, W) → (B, finest_size², dim)

        MAE masking (training only):
          - Drop mae_mask_ratio fraction of patches (replaced with zeros)
          - Forces encoder to produce robust features from partial info
          - Inference: all patches visible
        """
        B, C, H, W = x.shape
        p = self.min_patch_size
        gh, gw = H // p, W // p  # grid size
        N = gh * gw

        # Unfold into patches: (B, C, gh, gw, p, p)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # → (B*gh*gw, C, p, p)
        patches = patches.contiguous().view(B * N, C, p, p)

        # Per-patch CNN (독립 처리)
        features = self.patch_cnn(patches)  # (B*N, dim, 1, 1)
        features = features.view(B, N, self.dim)  # (B, N, dim)

        # MAE masking: training 시 일부 patch를 0으로 drop
        if self.training and self.mae_mask_ratio > 0:
            n_mask = int(N * self.mae_mask_ratio)
            if n_mask > 0:
                noise = torch.rand(B, N, device=x.device)
                ids_sort = noise.argsort(dim=1)
                mask_ids = ids_sort[:, :n_mask]  # (B, n_mask)
                mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
                mask.scatter_(1, mask_ids, True)  # True = masked
                features = features.masked_fill(mask.unsqueeze(-1), 0.0)

        return features

    def _run_hierarchy(self, tokens: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
        """Run MLP + merge hierarchy.

        Returns: [(tokens, spatial_size), ...] from finest to coarsest
        """
        level_outputs = []
        current = tokens

        for lvl in range(self.num_levels):
            if lvl > 0:
                prev_s = self.level_sizes[lvl - 1]
                current = self.merge_layers[lvl - 1](
                    level_outputs[-1][0], prev_s, prev_s
                )

            current = self.level_mlps[lvl](current)
            current = self.level_norms[lvl](current)
            level_outputs.append((current, self.level_sizes[lvl]))

        return level_outputs

    # ──────────────────────────────────────────────────────────────
    #  ViT backend: per-level independent encoding
    # ──────────────────────────────────────────────────────────────

    def _encode_level_vit(
        self, x: torch.Tensor, level_size: int,
    ) -> torch.Tensor:
        """Encode one hierarchy level with CellViT.

        Args:
            x: (B, C, H, W) full image
            level_size: grid size for this level (e.g. 8, 4, 2, 1)

        Returns:
            (B, dim, level_size, level_size)
        """
        B, C, H, W = x.shape
        s = level_size
        cell_h, cell_w = H // s, W // s

        if s == 1:
            # Whole image as a single cell
            cells = x  # (B, C, H, W)
        else:
            # Split into grid cells: (B*s*s, C, cell_h, cell_w)
            cells = x.unfold(2, cell_h, cell_h).unfold(3, cell_w, cell_w)
            # cells: (B, C, s, s, cell_h, cell_w)
            cells = cells.permute(0, 2, 3, 1, 4, 5).contiguous()
            cells = cells.view(B * s * s, C, cell_h, cell_w)

        eff_p, key = self._level_vit_info[s]
        mae_ratio = self.mae_mask_ratio if self.training else 0.0

        feat = self.cell_vit(
            cells, level_key=key, vit_patch_size=eff_p,
            mae_mask_ratio=mae_ratio,
        )  # (B*s*s, enc_dim) or (B, enc_dim) if s==1

        # Reshape to (B, enc_dim, s, s)
        enc_d = self._enc_dim
        feat = feat.view(B, s * s, enc_d)
        feat = feat.transpose(1, 2).view(B, enc_d, s, s)

        # Grid positional embedding (at enc_dim)
        feat = feat + self.grid_pos_embs[str(s)]

        # Project enc_dim → dim if they differ
        if self._vit_out_projs is not None:
            feat = self._vit_out_projs[str(s)](feat)

        return feat

    # ──────────────────────────────────────────────────────────────
    #  Forward: injection mode (primary)
    # ──────────────────────────────────────────────────────────────

    def forward_injection(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Return per-level features as 2D maps.

        Args:
            x: (B, C, H, W) conditioning image

        Returns:
            {spatial_size: (B, dim, S, S)} for each level
            e.g. {8: (B,256,8,8), 4: (B,256,4,4), 2: (B,256,2,2), 1: (B,256,1,1)}
        """
        if self.encoder_type == 'swin':
            return self._forward_swin(x)

        if self.encoder_type == 'vit_global':
            return self._forward_vit_global(x)

        if self.encoder_type == 'vit':
            level_features = {}
            for s in self.level_sizes:
                level_features[s] = self._encode_level_vit(x, s)
            return level_features

        # CNN path
        B = x.shape[0]
        tokens = self._patchify(x)
        level_outputs = self._run_hierarchy(tokens)

        level_features = {}
        for lvl, (tok, s) in enumerate(level_outputs):
            feat_2d = tok.transpose(1, 2).view(B, self.dim, s, s)
            level_features[s] = feat_2d

        return level_features

    # ──────────────────────────────────────────────────────────────
    #  Forward: legacy modes (concat / cross-attn)
    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Default forward = injection mode."""
        return self.forward_injection(x)

    # ──────────────────────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────────────────────

    def get_total_tokens(self) -> int:
        return sum(s * s for s in self.level_sizes)
