"""
Hierarchical Multi-Resolution Image Condition Encoder
=====================================================

Two encoder backends:

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
        # ── Custom level sizes (optional) ──
        level_sizes: list[int] | None = None,
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
        """
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
            dim=dim,
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

        # Per-level positional embedding for the grid (2D)
        self.grid_pos_embs = nn.ParameterDict()
        for s in self.level_sizes:
            self.grid_pos_embs[str(s)] = nn.Parameter(
                torch.zeros(1, dim, s, s))
            nn.init.trunc_normal_(self.grid_pos_embs[str(s)], std=0.02)

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
        )  # (B*s*s, dim) or (B, dim) if s==1

        # Reshape to (B, dim, s, s)
        feat = feat.view(B, s * s, self.dim)
        feat = feat.transpose(1, 2).view(B, self.dim, s, s)

        # Grid positional embedding
        feat = feat + self.grid_pos_embs[str(s)]

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
