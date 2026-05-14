"""
Multi-Resolution Conditional UNet (Injection Mode)
===================================================

Patchify → per-patch CNN → merge hierarchy → learned upsample → UNet injection.

Features:
  - VAE support: vae_downsample_factor > 1 → UNet operates on latents
  - Level drop: training 시 finest level부터 random drop (coarsest는 항상 유지)
  - CFG drop: uncond_drop_prob로 전체 conditioning을 drop

모든 파라미터가 configurable:
  - image_size: 원본 이미지 크기 (e.g. 256)
  - min_patch_size: encoder 최소 patch (e.g. 32)
  - vae_downsample_factor: VAE compression (e.g. 8 → latent = 256/8 = 32)
  - unet_config: UNet block 구성
  → upsample_factor 자동 계산: encoder level × factor = UNet stage resolution

Level drop example (levels=[8,4,2,1], tokens=[64,16,4,1]):
  Training 시 uniform random으로 keep_levels를 결정:
    keep_levels=4 → [8,4,2,1] 전부 사용 (85 tokens)
    keep_levels=3 → [4,2,1] (8×8 drop, 21 tokens)
    keep_levels=2 → [2,1] (5 tokens)
    keep_levels=1 → [1] (1 token, global만)
  Inference 시 num_active_levels로 제어.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from multi_res_encoder import HierarchicalMultiResEncoder
from Discretizer import FSQDiscretizer, VQDiscretizer


# ──────────────────────────────────────────────────────────────────
#  Learned Upsample
# ──────────────────────────────────────────────────────────────────

class LearnedUpsample(nn.Module):
    """Multi-stage 2× upsampling.

    factor=4 → 2 stages of (bilinear 2× + conv3×3 + GN + SiLU).
    """

    def __init__(self, dim: int, factor: int):
        super().__init__()
        assert factor >= 1 and (factor & (factor - 1)) == 0, \
            f"factor must be power of 2, got {factor}"
        n_stages = int(math.log2(factor)) if factor > 1 else 0
        layers = []
        for _ in range(n_stages):
            layers.extend([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GroupNorm(min(8, dim), dim),
                nn.SiLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────
#  Multi-Resolution Conditional UNet
# ──────────────────────────────────────────────────────────────────

class MultiResConditionalUNet(nn.Module):
    """
    Parameters
    ----------
    image_size : int
        원본 이미지 크기 (e.g. 256)
    in_channels : int
        UNet 입력 채널 (latent channels, e.g. 4 for VAE, 3 for pixel)
    cond_in_channels : int
        conditioning 이미지 채널 (원본 이미지, e.g. 3)
    vae_downsample_factor : int
        VAE spatial compression ratio (1 = no VAE, 8 = 8× downsample)
    min_patch_size : int
        encoder 최소 patch 크기 in pixels (e.g. 32)
    num_levels : int or None
        encoder hierarchy 레벨 수 (None = 1×1까지 자동)
    feat_channels : int
        encoder feature dimension
    unet_config : dict
        UNet2DConditionModel 설정
    upsample_factor : int or None
        encoder level → UNet resolution 변환 factor (None = 자동 계산)
    uncond_drop_prob : float
        CFG를 위한 전체 conditioning dropout
    level_drop : bool
        finest level부터 random drop 활성화
    min_keep_levels : int
        level drop 시 최소 유지 레벨 수 (1 = global만)
    depth_per_level : int or list[int]
        encoder 각 레벨의 MLP block 수
    mlp_ratio : float
        MLP expansion ratio
    cnn_base_channels : int
        PatchCNN base channels
    mae_mask_ratio : float
        finest level에서 MAE-style patch masking 비율 (training only)
    use_fsq : bool
        FSQ discretization 사용 여부
    fsq_levels : list[int]
        FSQ quantization levels
    use_vq : bool
        VQ discretization 사용 여부
    vq_codebook_size : int
        VQ codebook size
    level_drop_after_steps : int
        level drop을 활성화할 training step 기준 (-1 = 처음부터 활성화)
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 4,
        cond_in_channels: int = 3,
        vae_downsample_factor: int = 8,
        min_patch_size: int = 32,
        num_levels: int | None = None,
        feat_channels: int = 256,
        unet_config: dict | None = None,
        upsample_factor: int | None = None,
        uncond_drop_prob: float = 0.1,
        level_drop: bool = True,
        min_keep_levels: int = 1,
        depth_per_level: int | list[int] = 2,
        mlp_ratio: float = 4.0,
        cnn_base_channels: int = 64,
        # --- MAE masking ---
        mae_mask_ratio: float = 0.0,
        # --- ViT encoder ---
        encoder_type: str = 'cnn',
        vit_patch_size: int = 4,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_use_cnn_stem: bool = True,
        vit_cnn_stem_reduction: int = 4,
        vit_small_cell_tps: int = 2,
        vit_default_tps: int | None = None,
        encoder_internal_dim: int | None = None,
        # --- Swin encoder ---
        swin_patch_size: int = 16,
        swin_embed_dim: int = 96,
        swin_depths: list[int] | None = None,
        swin_num_heads: list[int] | None = None,
        swin_window_size: int = 4,
        swin_mlp_ratio: float = 4.0,
        # --- CLIP init (vit_global only) ---
        vit_init_clip: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        # --- vit_global pool backend ---
        vit_global_pool_type: str = 'avg',
        vit_global_pool_mlp_ratio: float = 4.0,
        vit_global_pool_use_ffn: bool = True,
        # --- Discretization ---
        use_fsq: bool = False,
        fsq_levels: list[int] | None = None,
        fsq_drop_quant_p: float = 0.0,
        fsq_corrupt_tokens_p: float = 0.0,
        use_vq: bool = False,
        vq_codebook_size: int = 512,
        vq_beta: float = 0.25,
        level_drop_after_steps: int = -1,
        cond_use_latent: bool = False,
        # --- Custom level sizes (e.g. [9,3,1] for sudoku) ---
        level_sizes: list[int] | None = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.vae_downsample_factor = vae_downsample_factor
        self.latent_size = image_size // vae_downsample_factor
        self.feat_channels = feat_channels
        self.uncond_drop_prob = uncond_drop_prob
        self.level_drop = level_drop
        self.min_keep_levels = min_keep_levels
        self.level_drop_after_steps = level_drop_after_steps
        self._level_drop_enabled = (level_drop_after_steps == -1)  # 즉시 or step 이후
        self.cond_use_latent = cond_use_latent

        # ── Encoder ──
        # cond_use_latent=True: encoder receives VAE latent (latent_size, in_channels)
        # cond_use_latent=False: encoder receives raw image (image_size, cond_in_channels)
        if cond_use_latent:
            enc_image_size = self.latent_size
            enc_in_channels = in_channels
        else:
            enc_image_size = image_size
            enc_in_channels = cond_in_channels

        self.encoder = HierarchicalMultiResEncoder(
            in_channels=enc_in_channels,
            dim=feat_channels,
            image_size=enc_image_size,
            # cond_use_latent: match the same finest grid as raw-image mode
            # raw finest = image_size // min_patch_size (e.g. 256//32=8)
            # latent patch = enc_image_size // finest  (e.g.  16// 8=2)
            min_patch_size=min_patch_size if not cond_use_latent else enc_image_size // (image_size // min_patch_size),
            num_levels=num_levels,
            depth_per_level=depth_per_level,
            mlp_ratio=mlp_ratio,
            cnn_base_channels=cnn_base_channels,
            mae_mask_ratio=mae_mask_ratio,
            encoder_type=encoder_type,
            vit_patch_size=vit_patch_size,
            vit_depth=vit_depth,
            vit_num_heads=vit_num_heads,
            vit_mlp_ratio=vit_mlp_ratio,
            vit_use_cnn_stem=vit_use_cnn_stem,
            vit_cnn_stem_reduction=vit_cnn_stem_reduction,
            vit_small_cell_tps=vit_small_cell_tps,
            vit_default_tps=vit_default_tps,
            encoder_internal_dim=encoder_internal_dim,
            swin_patch_size=swin_patch_size,
            swin_embed_dim=swin_embed_dim,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_mlp_ratio=swin_mlp_ratio,
            level_sizes=level_sizes,
            vit_init_clip=vit_init_clip,
            clip_model_name=clip_model_name,
            vit_global_pool_type=vit_global_pool_type,
            vit_global_pool_mlp_ratio=vit_global_pool_mlp_ratio,
            vit_global_pool_use_ffn=vit_global_pool_use_ffn,
        )

        # ── Discretizer (optional) ──
        self.use_fsq = use_fsq
        self.use_vq = use_vq
        self.discretizer = None
        if use_fsq:
            assert fsq_levels is not None, "fsq_levels required when use_fsq=True"
            self.discretizer = FSQDiscretizer(
                slot_dim=feat_channels,
                levels=fsq_levels,
                drop_quant_p=fsq_drop_quant_p,
                corrupt_tokens_p=fsq_corrupt_tokens_p,
            )
        elif use_vq:
            self.discretizer = VQDiscretizer(
                slot_dim=feat_channels,
                codebook_size=vq_codebook_size,
                beta=vq_beta,
            )

        # level_sizes: [8, 4, 2, 1] (finest → coarsest)
        # For level drop, we think in terms of coarsest-first ordering:
        # keep_levels=1 → [1]        (global only)
        # keep_levels=2 → [2, 1]     (+ 2×2)
        # keep_levels=3 → [4, 2, 1]  (+ 4×4)
        # keep_levels=4 → [8,4,2,1]  (all)
        self.num_levels = self.encoder.num_levels

        # ── UNet ──
        unet_config = dict(unet_config) if unet_config else {}
        unet_config.setdefault("sample_size", self.latent_size)
        unet_config.setdefault("in_channels", in_channels)
        unet_config.setdefault("out_channels", in_channels)

        block_out_channels = list(
            unet_config.get("block_out_channels", [128, 256, 256, 512])
        )
        layers_per_block = unet_config.get("layers_per_block", 2)
        num_blocks = len(block_out_channels)

        # cross_attention_dim for mid/down/up block attention
        # Only needed when attention blocks exist
        mid_type = unet_config.get("mid_block_type", "UNetMidBlock2DCrossAttn")
        has_any_attn = (mid_type != "UNetMidBlock2D" or
                        any("Attn" in t for t in unet_config.get("down_block_types", [])))
        if has_any_attn:
            unet_config["cross_attention_dim"] = block_out_channels[-1]

        self.unet = UNet2DConditionModel(**unet_config)

        import inspect
        self._mid_block_has_cross_attn = (
            self.unet.mid_block is not None
            and 'encoder_hidden_states' in inspect.signature(self.unet.mid_block.forward).parameters
        )

        # ── Compute UNet bottleneck resolution ──
        spatial = self.latent_size
        for i in range(num_blocks):
            if i < num_blocks - 1:
                spatial //= 2
        self._bottleneck_size = spatial

        # ── Upsample factor ──
        coarsest_encoder = min(self.encoder.level_sizes)
        if upsample_factor is not None:
            self._upsample_factor = upsample_factor
        else:
            self._upsample_factor = self._bottleneck_size // coarsest_encoder
        assert self._upsample_factor >= 1, \
            f"bottleneck({self._bottleneck_size}) < coarsest({coarsest_encoder})"
        assert self._upsample_factor & (self._upsample_factor - 1) == 0, \
            f"upsample_factor({self._upsample_factor}) must be power of 2"

        # ── Compute UNet bottleneck info ──
        self._mid_channels = block_out_channels[-1]
        spatial = self.latent_size
        for i in range(num_blocks):
            if i < num_blocks - 1:
                spatial //= 2
        self._mid_spatial = spatial

        # ── Down-path injection ──
        # Conditioning is added to the sample BEFORE each down block.
        # The UNet encoder sees conditioning directly → skip connections
        # naturally carry conditioned features to the decoder.
        INIT_STD = 1e-5

        self.level_upsamplers = nn.ModuleDict()
        self.down_injection_convs = nn.ModuleDict()
        self._encoder_to_block = {}

        # Input channels for each down block
        down_input_ch = [block_out_channels[0]]  # block 0: conv_in output
        for i in range(num_blocks - 1):
            down_input_ch.append(block_out_channels[i])

        # Resolution before each down block
        block_res = self.latent_size
        block_resolutions = []
        for i in range(num_blocks):
            block_resolutions.append(block_res)
            if i < num_blocks - 1:
                block_res //= 2

        for s in self.encoder.level_sizes:
            self.level_upsamplers[str(s)] = LearnedUpsample(
                feat_channels, self._upsample_factor
            )
            target_res = s * self._upsample_factor
            for block_idx, br in enumerate(block_resolutions):
                if target_res == br:
                    conv = nn.Conv2d(feat_channels, down_input_ch[block_idx], 1, bias=True)
                    nn.init.normal_(conv.weight, std=INIT_STD)
                    nn.init.zeros_(conv.bias)
                    self.down_injection_convs[str(s)] = conv
                    self._encoder_to_block[s] = block_idx
                    break

        # Coarsest → mid block
        coarsest_target = coarsest_encoder * self._upsample_factor
        if coarsest_target == self._mid_spatial:
            self.mid_injection_conv = nn.Conv2d(
                feat_channels, self._mid_channels, 1, bias=True
            )
            nn.init.normal_(self.mid_injection_conv.weight, std=INIT_STD)
            nn.init.zeros_(self.mid_injection_conv.bias)
        else:
            self.mid_injection_conv = None

        # ── Learned null conditioning (for CFG) ──
        # Per-level learned null embedding, semanticist 스타일
        self.null_cond = nn.ParameterDict()
        for s in self.encoder.level_sizes:
            self.null_cond[str(s)] = nn.Parameter(
                torch.zeros(1, feat_channels, s, s)
            )

    # ──────────────────────────────────────────────────────────────
    #  Step tracking for level drop schedule
    # ──────────────────────────────────────────────────────────────

    def set_step(self, step: int):
        """Training loop에서 호출: level drop 활성화 시점 체크."""
        if (self.level_drop_after_steps != -1
                and step >= self.level_drop_after_steps):
            self._level_drop_enabled = True

    # ──────────────────────────────────────────────────────────────
    #  Info
    # ──────────────────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable summary of the architecture."""
        lines = [
            f"Image: {self.image_size}×{self.image_size}",
            f"VAE: ×{self.vae_downsample_factor} → latent {self.latent_size}×{self.latent_size}",
            f"Encoder input: {'VAE latent' if self.cond_use_latent else 'raw image'} ({self.encoder.image_size}×{self.encoder.image_size}, type={self.encoder.encoder_type})",
            f"Encoder min_patch: {self.encoder.min_patch_size}",
            f"Encoder levels: {self.encoder.level_sizes}",
            f"Upsample factor: ×{self._upsample_factor}",
            f"UNet bottleneck: {self._bottleneck_size}×{self._bottleneck_size}",
            f"Level drop: {self.level_drop} (min_keep={self.min_keep_levels}, after_steps={self.level_drop_after_steps})",
            f"CFG uncond drop: {self.uncond_drop_prob} (learned null_cond)",
            "",
            "Down-path injection:",
        ]
        for s in sorted(self.encoder.level_sizes, reverse=True):
            target = s * self._upsample_factor
            if s in self._encoder_to_block:
                block_idx = self._encoder_to_block[s]
                lines.append(
                    f"  {s}×{s} ({s*s} tokens) → ×{self._upsample_factor} → "
                    f"{target}×{target} → + sample before down_block[{block_idx}]"
                )
            else:
                lines.append(
                    f"  {s}×{s} ({s*s} tokens) → ×{self._upsample_factor} → "
                    f"{target}×{target} → SKIPPED"
                )
        if self.mid_injection_conv is not None:
            c = min(self.encoder.level_sizes)
            ct = c * self._upsample_factor
            lines.append(
                f"  {c}×{c} → ×{self._upsample_factor} → "
                f"{ct}×{ct} → mid block"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    #  Level drop logic
    # ──────────────────────────────────────────────────────────────

    def _sample_keep_levels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample how many levels to keep per batch element.

        Returns:
            (B,) int tensor, values in [min_keep_levels, num_levels].
            1 = global only, num_levels = all levels.
        """
        # Uniform over [min_keep_levels, num_levels]
        return torch.randint(
            self.min_keep_levels, self.num_levels + 1,
            (batch_size,), device=device,
        )

    def _get_level_mask(self, keep_levels: int | torch.Tensor) -> dict[int, bool]:
        """Convert keep_levels to a per-level mask.

        Levels are ordered coarsest-first: level_sizes reversed = [1, 2, 4, 8]
        keep_levels=2 → keep [1, 2], drop [4, 8]
        """
        # For per-sample variable masking, we return a mask tensor per level
        # For simplicity: keep_levels is scalar (all samples same)
        if isinstance(keep_levels, torch.Tensor):
            keep_levels = keep_levels.item() if keep_levels.numel() == 1 else None

        # level_sizes: [8, 4, 2, 1] (finest → coarsest)
        # coarsest_first: [1, 2, 4, 8]
        coarsest_first = list(reversed(self.encoder.level_sizes))
        mask = {}
        if keep_levels is not None:
            for i, s in enumerate(coarsest_first):
                mask[s] = (i < keep_levels)
        return mask

    # ──────────────────────────────────────────────────────────────
    #  Encoding
    # ──────────────────────────────────────────────────────────────

    def encode_condition(
        self,
        cond_image: torch.Tensor | None = None,
        return_uncond: bool = False,
        num_active_levels: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Encode conditioning image → per-level upsampled features.

        Args:
            cond_image: (B, C, image_size, image_size) raw image
            return_uncond: if True, return zero features (CFG uncond)
            num_active_levels: override for inference (None = all levels)

        Returns:
            {encoder_spatial: (B, feat_ch, target_res, target_res)}
        """
        if return_uncond or cond_image is None:
            # CFG unconditional: zero injection (gradient 차단)
            B = cond_image.shape[0] if cond_image is not None else 1
            device = cond_image.device if cond_image is not None else "cpu"
            dtype = cond_image.dtype if cond_image is not None else torch.float32
            result = {}
            for s in self.encoder.level_sizes:
                target_res = s * self._upsample_factor
                result[s] = torch.zeros(
                    B, self.feat_channels, target_res, target_res,
                    device=device, dtype=dtype,
                )
            return result

        B = cond_image.shape[0]

        # Encoder → per-level 2D features
        level_features = self.encoder.forward_injection(cond_image)

        # ── Discretization (optional, applied per level) ──
        if self.discretizer is not None:
            for s, feat_2d in level_features.items():
                # feat_2d: (B, D, S, S) → (B, S*S, D) for discretizer
                D = feat_2d.shape[1]
                tokens_2d = feat_2d.flatten(2).transpose(1, 2)  # (B, S*S, D)
                quant_tokens, _ = self.discretizer(tokens_2d)   # (B, S*S, D)
                level_features[s] = quant_tokens.transpose(1, 2).view(B, D, s, s)

        # Learned upsample
        upsampled = {}
        for s, feat_2d in level_features.items():
            if str(s) in self.level_upsamplers:
                upsampled[s] = self.level_upsamplers[str(s)](feat_2d)

        # ── Level drop (training, step threshold 이후만) ──
        # Finest levels부터 drop, coarsest는 항상 유지
        # drop된 level은 learned null_cond로 대체
        if self.training and self.level_drop and self._level_drop_enabled:
            keep_levels = self._sample_keep_levels(B, cond_image.device)
            coarsest_first = list(reversed(self.encoder.level_sizes))
            for level_idx, s in enumerate(coarsest_first):
                if s in upsampled:
                    # keep if level_idx < keep_levels[b] → mask=1, else 0
                    mask = (level_idx < keep_levels).float().view(B, 1, 1, 1)
                    upsampled[s] = upsampled[s] * mask  # drop된 level은 0
        elif num_active_levels is not None:
            # Inference: explicit level control
            coarsest_first = list(reversed(self.encoder.level_sizes))
            for level_idx, s in enumerate(coarsest_first):
                if s in upsampled and level_idx >= num_active_levels:
                    upsampled[s] = torch.zeros_like(upsampled[s])

        # ── CFG dropout (training) ──
        # Independent of level drop: drops ALL conditioning for some samples
        # → zero injection (UNet에 아무것도 안 더함 = unconditional)
        if self.training and self.uncond_drop_prob > 0.0:
            drop_mask = (
                torch.rand(B, device=cond_image.device) < self.uncond_drop_prob
            )
            if drop_mask.any():
                keep_mask = (~drop_mask).view(B, 1, 1, 1).float()
                for s in upsampled:
                    upsampled[s] = upsampled[s] * keep_mask

        return upsampled

    # ──────────────────────────────────────────────────────────────
    #  UNet forward with down-path injection
    # ──────────────────────────────────────────────────────────────

    def _forward_unet(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        down_injections: dict[int, torch.Tensor],
        mid_residual: torch.Tensor | None,
    ) -> torch.Tensor:
        """UNet forward with conditioning injected into the down path.

        Instead of adding residuals to skip connections after the fact,
        we add conditioning to the sample before each down block.
        The UNet encoder sees conditioning directly.
        """
        unet = self.unet

        # Timestep embedding
        timesteps = t
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x_t.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x_t.device)
        timesteps = timesteps.expand(x_t.shape[0])

        t_emb = unet.time_proj(timesteps).to(dtype=x_t.dtype)
        emb = unet.time_embedding(t_emb)

        # Pre-process
        sample = unet.conv_in(x_t)

        # Down path — inject conditioning before each block
        down_block_res_samples = (sample,)
        for i, down_block in enumerate(unet.down_blocks):
            if i in down_injections:
                sample = sample + down_injections[i]
            sample, res_samples = down_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # Mid block
        if unet.mid_block is not None:
            if self._mid_block_has_cross_attn:
                sample = unet.mid_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=None,
                )
            else:
                sample = unet.mid_block(hidden_states=sample, temb=emb)
        if mid_residual is not None:
            sample = sample + mid_residual

        # Up path
        for i, up_block in enumerate(unet.up_blocks):
            n_res = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_res:]
            down_block_res_samples = down_block_res_samples[:-n_res]
            sample = up_block(
                hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples,
            )

        # Post-process
        if unet.conv_norm_out:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)

        return sample

    # ──────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image: torch.Tensor | None = None,
        return_uncond: bool = False,
        num_active_levels: int | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x_t: (B, C, latent_size, latent_size) noisy latent (or pixel)
            t: (B,) diffusion timesteps
            cond_image: (B, C_img, image_size, image_size) raw conditioning image
            return_uncond: use zero conditioning (for CFG eval)
            num_active_levels: override level count for inference
            return_aux_loss: if True, return (pred, aux_dict) with VQ loss etc.
        """
        upsampled = self.encode_condition(
            cond_image, return_uncond=return_uncond,
            num_active_levels=num_active_levels,
        )

        # Build down-path injections: block_index → conditioning tensor
        down_injections = {}
        for s, block_idx in self._encoder_to_block.items():
            if s in upsampled and str(s) in self.down_injection_convs:
                down_injections[block_idx] = self.down_injection_convs[str(s)](upsampled[s])

        # Mid block injection (coarsest level)
        mid_residual = None
        if self.mid_injection_conv is not None:
            coarsest = min(self.encoder.level_sizes)
            if coarsest in upsampled:
                mid_residual = self.mid_injection_conv(upsampled[coarsest])

        pred = self._forward_unet(x_t, t, down_injections, mid_residual)

        if return_aux_loss:
            aux = {}
            if self.use_vq and self.discretizer is not None and self.discretizer.last_vq_loss is not None:
                aux["vq_loss"] = self.discretizer.last_vq_loss
            if self.use_vq and self.discretizer is not None and self.discretizer.last_perplexity is not None:
                aux["vq_perplexity"] = self.discretizer.last_perplexity
                aux["vq_usage"] = self.discretizer.last_usage
            return pred, aux

        return pred


# ──────────────────────────────────────────────────────────────────
#  DiT building blocks (for MultiResConditionalDiT)
# ──────────────────────────────────────────────────────────────────

def _sinusoidal_timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (same as DiT / DDPM)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ── Positional embeddings (sin-cos, like JiT / MAE) ──

def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """2D sin-cos positional embedding → (grid_size*grid_size, embed_dim)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    pos_h = grid[0].reshape(-1)
    pos_w = grid[1].reshape(-1)

    half = embed_dim // 2
    omega = np.arange(half // 2, dtype=np.float64)
    omega /= half / 2.0
    omega = 1.0 / (10000.0 ** omega)

    out_h = np.einsum('m,d->md', pos_h, omega)
    out_w = np.einsum('m,d->md', pos_w, omega)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)
    return np.concatenate([emb_h, emb_w], axis=1)


# ── RoPE (Vision Rotary Position Embedding, adapted from JiT) ──

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of elements: (a, b) → (-b, a)."""
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_2d_rope(half_head_dim: int, hw_seq_len: int,
                   num_prefix_tokens: int = 0):
    """Build 2D vision RoPE cos/sin buffers.

    Args:
        half_head_dim: hidden_size // num_heads // 2
        hw_seq_len: grid size (H/patch_size)
        num_prefix_tokens: condition / cls tokens prepended before image tokens.
            These get identity rotation (cos=1, sin=0).

    Returns:
        (freqs_cos, freqs_sin) each of shape (num_prefix + hw^2, head_dim).
    """
    dim = half_head_dim
    freqs = 1.0 / (10000.0 ** (
        torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(hw_seq_len).float()
    freqs = torch.einsum('i, j -> ij', t, freqs)          # (hw, dim//2)
    freqs = freqs.repeat(1, 2)                              # (hw, dim)
    freqs_2d = torch.cat([
        freqs[:, None, :].expand(-1, hw_seq_len, -1),      # h-dim
        freqs[None, :, :].expand(hw_seq_len, -1, -1),      # w-dim
    ], dim=-1)                                               # (hw, hw, 2*dim=head_dim)
    freqs_flat = freqs_2d.reshape(-1, freqs_2d.shape[-1])   # (hw*hw, head_dim)

    cos_img = freqs_flat.cos()
    sin_img = freqs_flat.sin()

    if num_prefix_tokens > 0:
        D = cos_img.shape[-1]
        cos_pad = torch.ones(num_prefix_tokens, D)
        sin_pad = torch.zeros(num_prefix_tokens, D)
        cos_img = torch.cat([cos_pad, cos_img], dim=0)
        sin_img = torch.cat([sin_pad, sin_img], dim=0)

    return cos_img, sin_img


# ── Bottleneck Patch Embedding (JiT-style, 2-stage) ──

class _BottleneckPatchEmbed(nn.Module):
    """Image → patch embedding with a bottleneck (Conv → Conv1x1)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 bottleneck_dim: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj1 = nn.Conv2d(in_channels, bottleneck_dim,
                               kernel_size=patch_size, stride=patch_size,
                               bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim,
                               kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_patches, embed_dim)"""
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


# ── Building blocks ──

class _RMSNorm(nn.Module):
    """RMSNorm (LLaMA-style), used in JiT DiT blocks."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (as in JiT / LLaMA)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


class _DiTBlock(nn.Module):
    """DiT block: self-attn → cross-attn (to cond) → MLP, with adaLN-Zero.

    Uses RMSNorm, QK-norm, SwiGLU, RoPE (JiT-style).
    Cross-attention has spatial alignment masking for multi-res conditioning.

    adaLN produces 9 modulation params:
      self-attn:  shift1, scale1, gate1
      cross-attn: gate_xa  (no shift/scale on query — cond already projected)
      MLP:        shift2, scale2, gate2
    Total: 9 * D  (self 3 + cross 1 + mlp 3 + cross_q_norm 1 + cross_k_norm 1 = 9)

    Actually simplified: 8 params.
      self-attn:  shift1, scale1, gate1  (3)
      cross-attn: gate_xa               (1)
      MLP:        shift2, scale2, gate2  (3)
      reserved:   (none — norms on cross q/k are parameter-free RMSNorm)
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        assert dim % n_heads == 0
        head_dim = dim // n_heads

        # ── Self-attention ──
        self.norm1 = _RMSNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.q_norm = _RMSNorm(head_dim)
        self.k_norm = _RMSNorm(head_dim)
        self.sa_out_proj = nn.Linear(dim, dim)

        # ── Cross-attention (image queries → cond keys/values) ──
        self.xa_q_norm = _RMSNorm(dim, eps=1e-6)
        self.xa_q_proj = nn.Linear(dim, dim, bias=True)
        self.xa_kv_norm = _RMSNorm(dim, eps=1e-6)
        self.xa_kv_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.xa_q_head_norm = _RMSNorm(head_dim)
        self.xa_k_head_norm = _RMSNorm(head_dim)
        self.xa_out_proj = nn.Linear(dim, dim)

        # ── MLP ──
        self.norm2 = _RMSNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = _SwiGLUFFN(dim, mlp_hidden, drop=proj_drop)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # adaLN-Zero: 7 params (shift1, scale1, gate_sa, gate_xa, shift2, scale2, gate_mlp)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 7 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cond: torch.Tensor,
                sa_mask: torch.Tensor | None = None,
                xa_mask: torch.Tensor | None = None,
                rope_cos: torch.Tensor | None = None,
                rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) image tokens (+ in-context tokens if applicable)
            c: (B, D) timestep conditioning
            cond: (B, M, D) condition tokens (multi-res, projected)
            sa_mask: (B, 1, N, N) self-attention mask (True=attend), or None
            xa_mask: (B, 1, N, M) cross-attention mask (True=attend)
            rope_cos, rope_sin: (N, head_dim) RoPE buffers for self-attn
        """
        B, N, D = x.shape
        H = self.n_heads
        hd = D // H

        shift1, scale1, gate_sa, gate_xa, shift2, scale2, gate_mlp = \
            self.adaLN(c).unsqueeze(1).chunk(7, dim=-1)

        # ── 1) Self-attention with QK-norm + RoPE ──
        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, N, 3, H, hd)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cos is not None:
            q = q * rope_cos + _rotate_half(q) * rope_sin
            k = k * rope_cos + _rotate_half(k) * rope_sin
        sa_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        sa_out = sa_out.transpose(1, 2).reshape(B, N, D)
        x = x + gate_sa * self.proj_drop(self.sa_out_proj(sa_out))

        # ── 2) Cross-attention: x queries → cond keys/values ──
        M = cond.shape[1]
        xq = self.xa_q_proj(self.xa_q_norm(x)).reshape(B, N, H, hd).transpose(1, 2)
        kv = self.xa_kv_proj(self.xa_kv_norm(cond)).reshape(B, M, 2, H, hd)
        xk, xv = kv.unbind(dim=2)
        xk, xv = xk.transpose(1, 2), xv.transpose(1, 2)
        xq = self.xa_q_head_norm(xq)
        xk = self.xa_k_head_norm(xk)
        # No RoPE on cross-attention (cond tokens have no spatial position)
        xa_out = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=xa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        xa_out = xa_out.transpose(1, 2).reshape(B, N, D)
        x = x + gate_xa * self.proj_drop(self.xa_out_proj(xa_out))

        # ── 3) MLP (SwiGLU) ──
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate_mlp * self.mlp(h)
        return x


class _DiTFinalLayer(nn.Module):
    """Final layer with adaLN-Zero and RMSNorm (JiT-style)."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = _RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


# ──────────────────────────────────────────────────────────────────
#  Multi-Resolution Conditional DiT (JiT-style architecture)
# ──────────────────────────────────────────────────────────────────

class MultiResConditionalDiT(nn.Module):
    """
    Multi-Resolution Conditional DiT (JiT-style Diffusion Transformer).

    Architecture: self-attn → cross-attn → MLP per block.
      - Self-attention: image tokens (+ in-context tokens from in_context_start)
      - Cross-attention: image tokens query against multi-res condition tokens
        with spatial alignment masking.
      - In-context tokens: self-attention only (no cross-attention to cond).

    JiT features: BottleneckPatchEmbed, fixed sin-cos pos embed, 2D RoPE,
    in-context tokens, middle-50% dropout, adaLN-Zero, RMSNorm, QK-norm, SwiGLU.

    Self-attn sequence (before in_context_start):  [image_tokens]
    Self-attn sequence (from in_context_start):    [in_context, image_tokens]
    Cross-attn: image_tokens → [cond_finest, ..., cond_coarsest]  (spatial mask)
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        cond_in_channels: int = 3,
        vae_downsample_factor: int = 1,
        min_patch_size: int = 32,
        num_levels: int | None = None,
        feat_channels: int = 256,
        # --- DiT backbone (JiT-style) ---
        dit_patch_size: int = 16,
        dit_hidden_size: int = 768,
        dit_n_heads: int = 12,
        dit_n_blocks: int = 12,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.0,
        dit_bottleneck_dim: int = 128,
        dit_in_context_len: int = 0,
        dit_in_context_start: int = 4,
        # --- common ---
        uncond_drop_prob: float = 0.1,
        level_drop: bool = True,
        min_keep_levels: int = 1,
        depth_per_level: int | list[int] = 2,
        mlp_ratio: float = 4.0,
        cnn_base_channels: int = 64,
        mae_mask_ratio: float = 0.0,
        # --- ViT encoder ---
        encoder_type: str = 'cnn',
        vit_patch_size: int = 4,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_use_cnn_stem: bool = True,
        vit_cnn_stem_reduction: int = 4,
        vit_small_cell_tps: int = 2,
        vit_default_tps: int | None = None,
        encoder_internal_dim: int | None = None,
        # --- Swin encoder ---
        swin_patch_size: int = 16,
        swin_embed_dim: int = 96,
        swin_depths: list[int] | None = None,
        swin_num_heads: list[int] | None = None,
        swin_window_size: int = 4,
        swin_mlp_ratio: float = 4.0,
        # --- CLIP init (vit_global only) ---
        vit_init_clip: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        # --- vit_global pool backend ---
        vit_global_pool_type: str = 'avg',
        vit_global_pool_mlp_ratio: float = 4.0,
        vit_global_pool_use_ffn: bool = True,
        # --- Discretization ---
        use_fsq: bool = False,
        fsq_levels: list[int] | None = None,
        fsq_drop_quant_p: float = 0.0,
        fsq_corrupt_tokens_p: float = 0.0,
        use_vq: bool = False,
        vq_codebook_size: int = 512,
        vq_beta: float = 0.25,
        level_drop_after_steps: int = -1,
        cond_use_latent: bool = False,
        # --- Custom level sizes (e.g. [9,3,1] for sudoku) ---
        level_sizes: list[int] | None = None,
        # --- Condition noise injection (train-only regularizer) ---
        cond_noise_std: float = 0.0,
        cond_noise_relative: bool = False,
        # --- Per-token random drop (train-only regularizer) ---
        cond_token_drop_prob: float = 0.0,
        cond_token_drop_all_levels: bool = False,
        cond_token_drop_linear: bool = False,
    ):
        super().__init__()

        self.image_size = image_size
        self.vae_downsample_factor = vae_downsample_factor
        self.latent_size = image_size // vae_downsample_factor
        self.feat_channels = feat_channels
        self.uncond_drop_prob = uncond_drop_prob
        self.cond_noise_std = cond_noise_std
        self.cond_noise_relative = cond_noise_relative
        self.cond_token_drop_prob = cond_token_drop_prob
        self.cond_token_drop_all_levels = cond_token_drop_all_levels
        self.cond_token_drop_linear = cond_token_drop_linear
        self.level_drop = level_drop
        self.min_keep_levels = min_keep_levels
        self.level_drop_after_steps = level_drop_after_steps
        self._level_drop_enabled = (level_drop_after_steps == -1)
        self.cond_use_latent = cond_use_latent

        self.dit_patch_size = dit_patch_size
        self.dit_hidden_size = dit_hidden_size
        self.dit_n_heads = dit_n_heads
        self.dit_n_blocks = dit_n_blocks
        self._in_channels = in_channels
        self.in_context_len = dit_in_context_len
        self.in_context_start = dit_in_context_start

        assert self.latent_size % dit_patch_size == 0, \
            f"latent_size({self.latent_size}) must be divisible by dit_patch_size({dit_patch_size})"
        self.grid_size = self.latent_size // dit_patch_size

        # ── Encoder (identical to UNet version) ──
        if cond_use_latent:
            enc_image_size = self.latent_size
            enc_in_channels = in_channels
        else:
            enc_image_size = image_size
            enc_in_channels = cond_in_channels

        self.encoder = HierarchicalMultiResEncoder(
            in_channels=enc_in_channels,
            dim=feat_channels,
            image_size=enc_image_size,
            min_patch_size=(min_patch_size if not cond_use_latent
                            else enc_image_size // (image_size // min_patch_size)),
            num_levels=num_levels,
            depth_per_level=depth_per_level,
            mlp_ratio=mlp_ratio,
            cnn_base_channels=cnn_base_channels,
            mae_mask_ratio=mae_mask_ratio,
            encoder_type=encoder_type,
            vit_patch_size=vit_patch_size,
            vit_depth=vit_depth,
            vit_num_heads=vit_num_heads,
            vit_mlp_ratio=vit_mlp_ratio,
            vit_use_cnn_stem=vit_use_cnn_stem,
            vit_cnn_stem_reduction=vit_cnn_stem_reduction,
            vit_small_cell_tps=vit_small_cell_tps,
            vit_default_tps=vit_default_tps,
            encoder_internal_dim=encoder_internal_dim,
            swin_patch_size=swin_patch_size,
            swin_embed_dim=swin_embed_dim,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            swin_window_size=swin_window_size,
            swin_mlp_ratio=swin_mlp_ratio,
            level_sizes=level_sizes,
            vit_init_clip=vit_init_clip,
            clip_model_name=clip_model_name,
            vit_global_pool_type=vit_global_pool_type,
            vit_global_pool_mlp_ratio=vit_global_pool_mlp_ratio,
            vit_global_pool_use_ffn=vit_global_pool_use_ffn,
        )

        # ── Discretizer (identical to UNet version) ──
        self.use_fsq = use_fsq
        self.use_vq = use_vq
        self.discretizer = None
        if use_fsq:
            assert fsq_levels is not None
            self.discretizer = FSQDiscretizer(
                slot_dim=feat_channels, levels=fsq_levels,
                drop_quant_p=fsq_drop_quant_p,
                corrupt_tokens_p=fsq_corrupt_tokens_p,
            )
        elif use_vq:
            self.discretizer = VQDiscretizer(
                slot_dim=feat_channels, codebook_size=vq_codebook_size,
                beta=vq_beta,
            )

        self.num_levels = self.encoder.num_levels

        # ── Patch embedding (JiT BottleneckPatchEmbed: Conv→Conv1x1) ──
        self.patch_embed = _BottleneckPatchEmbed(
            self.latent_size, dit_patch_size, in_channels,
            dit_bottleneck_dim, dit_hidden_size,
        )

        # ── Condition projection (feat_channels → dit_hidden_size) ──
        self.cond_proj = nn.Linear(feat_channels, dit_hidden_size)

        # ── Positional embeddings: fixed sin-cos (frozen, JiT-style) ──
        num_img_tokens = self.grid_size ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_img_tokens, dit_hidden_size),
            requires_grad=False,
        )

        # Per-level positional embeddings for condition tokens (learnable)
        self.cond_pos_embeds = nn.ParameterDict()
        for s in self.encoder.level_sizes:
            self.cond_pos_embeds[str(s)] = nn.Parameter(
                torch.zeros(1, s * s, dit_hidden_size)
            )
            nn.init.trunc_normal_(self.cond_pos_embeds[str(s)], std=0.02)

        # ── Learned null condition embeddings (for CFG & level drop) ──
        self.null_cond = nn.ParameterDict()
        for s in self.encoder.level_sizes:
            self.null_cond[str(s)] = nn.Parameter(
                torch.zeros(1, s * s, feat_channels)
            )

        # ── In-context tokens (JiT-style, optional) ──
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, dit_hidden_size))
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # ── Timestep embedding (JiT-style: sinusoidal freq → MLP) ──
        self._t_freq_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self._t_freq_dim, dit_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(dit_hidden_size, dit_hidden_size, bias=True),
        )

        # ── 2D RoPE for self-attention (image tokens only, prefix=in_context) ──
        half_head_dim = dit_hidden_size // dit_n_heads // 2
        hw = self.grid_size

        # Base RoPE: image tokens only (no prefix)
        rope_cos, rope_sin = _build_2d_rope(half_head_dim, hw, 0)
        self.register_buffer('_rope_cos', rope_cos)
        self.register_buffer('_rope_sin', rope_sin)

        # Extended RoPE: in_context prefix + image tokens
        if self.in_context_len > 0:
            rope_cos_ext, rope_sin_ext = _build_2d_rope(
                half_head_dim, hw, self.in_context_len)
            self.register_buffer('_rope_cos_ext', rope_cos_ext)
            self.register_buffer('_rope_sin_ext', rope_sin_ext)

        # ── Transformer blocks (JiT-style: middle 50% get dropout) ──
        self.blocks = nn.ModuleList()
        for i in range(dit_n_blocks):
            in_middle = (dit_n_blocks // 4 * 3 > i >= dit_n_blocks // 4)
            a_drop = dit_dropout if in_middle else 0.0
            p_drop = dit_dropout if in_middle else 0.0
            self.blocks.append(
                _DiTBlock(dit_hidden_size, dit_n_heads, dit_mlp_ratio,
                          attn_drop=a_drop, proj_drop=p_drop))

        # ── Final layer ──
        self.final_layer = _DiTFinalLayer(
            dit_hidden_size, in_channels * dit_patch_size ** 2,
        )

        # ── Precompute cross-attention masks ──
        self._precompute_masks()

        # ── JiT-style weight initialization ──
        self._initialize_weights()

    # ──────────────────────────────────────────────────────────────
    #  Mask precomputation (cross-attention: image → cond)
    # ──────────────────────────────────────────────────────────────

    def _precompute_masks(self):
        """Precompute cross-attention masks for spatial alignment.

        Cross-attn mask shape: (N_img, M_cond) — each image token attends
        to its spatially aligned condition token(s) at each level.

        Also stores per-cond-token level indices for level-drop masking.
        """
        G = self.grid_size
        num_img = G * G
        level_sizes = self.encoder.level_sizes  # finest → coarsest

        # Cond offsets
        cond_offsets = {}
        offset = 0
        for s in level_sizes:
            cond_offsets[s] = offset
            offset += s * s
        num_cond = offset
        self._num_cond = num_cond

        # ── Cross-attention mask: (num_img, num_cond) ──
        xa_mask = torch.zeros(num_img, num_cond, dtype=torch.bool)
        for s in level_sizes:
            start = cond_offsets[s]
            for r in range(G):
                for col in range(G):
                    img_idx = r * G + col
                    if G >= s:
                        cond_r = r * s // G
                        cond_c = col * s // G
                        xa_mask[img_idx, start + cond_r * s + cond_c] = True
                    else:
                        ratio = s // G
                        for dr in range(ratio):
                            for dc in range(ratio):
                                cr = r * ratio + dr
                                cc = col * ratio + dc
                                xa_mask[img_idx, start + cr * s + cc] = True
        # Store as (1, 1, N_img, M_cond) for broadcasting
        self.register_buffer('_xa_mask', xa_mask.unsqueeze(0).unsqueeze(0))

        # Per-cond-token level index (coarsest-first ordering for level drop)
        coarsest_first = list(reversed(level_sizes))
        cond_token_level = torch.full((num_cond,), -1, dtype=torch.long)
        for cf_idx, s in enumerate(coarsest_first):
            start = cond_offsets[s]
            cond_token_level[start:start + s * s] = cf_idx
        self.register_buffer('_cond_token_level', cond_token_level)

        self._size_to_cf_idx = {s: i for i, s in enumerate(coarsest_first)}

    # ──────────────────────────────────────────────────────────────
    #  Weight initialization (JiT-style)
    # ──────────────────────────────────────────────────────────────

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Fixed sin-cos positional embedding for image tokens
        pos = _get_2d_sincos_pos_embed(self.dit_hidden_size, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0))

        # Timestep MLP: normal init (std=0.02)
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.time_embed[2].weight, std=0.02)

        # Patch embed: xavier for both stages
        w1 = self.patch_embed.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.patch_embed.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj2.bias, 0)

        # Zero-out adaLN modulation layers in all blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)

        # NOTE: xa_out_proj keeps xavier init from _basic_init.
        # Only gate_xa (from adaLN) is zero-init → cross-attn starts as
        # identity but gradients flow through the non-zero projection.

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ──────────────────────────────────────────────────────────────
    #  Dynamic cross-attention mask with level drop
    # ──────────────────────────────────────────────────────────────

    def _get_xa_mask(self, keep_levels: torch.Tensor,
                     has_prefix: bool = False) -> torch.Tensor:
        """Build per-sample cross-attention mask with level drop.

        Args:
            keep_levels: (B,) int tensor, values in [0, num_levels].
            has_prefix: if True, prepend rows of zeros for in-context tokens
                        (in-context tokens don't cross-attend to cond).

        Returns:
            (B, 1, N_query, M_cond) boolean mask.
            N_query = (K + num_img) if has_prefix else num_img.
        """
        B = keep_levels.shape[0]
        # Per-cond-token visibility: (B, M_cond)
        cond_visible = self._cond_token_level.unsqueeze(0) < keep_levels.unsqueeze(1)
        # Base spatial mask: (1, 1, N_img, M_cond) & cond visibility: (B, 1, 1, M_cond)
        xa_mask = self._xa_mask & cond_visible.unsqueeze(1).unsqueeze(1)

        if has_prefix and self.in_context_len > 0:
            K = self.in_context_len
            # In-context tokens: no cross-attention to cond
            prefix_mask = torch.zeros(
                B, 1, K, self._num_cond,
                dtype=torch.bool, device=xa_mask.device)
            xa_mask = torch.cat([prefix_mask, xa_mask], dim=2)

        return xa_mask

    # ──────────────────────────────────────────────────────────────
    #  Step tracking / Level drop
    # ──────────────────────────────────────────────────────────────

    def set_step(self, step: int):
        if (self.level_drop_after_steps != -1
                and step >= self.level_drop_after_steps):
            self._level_drop_enabled = True

    def _sample_keep_levels(self, batch_size: int,
                            device: torch.device) -> torch.Tensor:
        return torch.randint(
            self.min_keep_levels, self.num_levels + 1,
            (batch_size,), device=device,
        )

    # ──────────────────────────────────────────────────────────────
    #  Unpatchify
    # ──────────────────────────────────────────────────────────────

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, G*G, C*p*p) → (B, C, H, W)"""
        B = x.shape[0]
        G = self.grid_size
        p = self.dit_patch_size
        C = self._in_channels
        x = x.view(B, G, G, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, G * p, G * p)

    # ──────────────────────────────────────────────────────────────
    #  Info
    # ──────────────────────────────────────────────────────────────

    def describe(self) -> str:
        total_cond = sum(s * s for s in self.encoder.level_sizes)
        num_img = self.grid_size ** 2
        G = self.grid_size
        K = self.in_context_len
        lines = [
            f"Image: {self.image_size}×{self.image_size}",
            f"VAE: ×{self.vae_downsample_factor} → latent {self.latent_size}×{self.latent_size}",
            f"Backbone: DiT (patch={self.dit_patch_size}, hidden={self.dit_hidden_size}, "
            f"heads={self.dit_n_heads}, blocks={self.dit_n_blocks})",
            f"Attention: self-attn → cross-attn (spatial mask) → MLP",
            f"Patch embed: BottleneckPatchEmbed → {num_img} image tokens",
            f"RoPE: 2D vision RoPE (self-attn only)",
            f"In-context: {K} tokens (from layer {self.in_context_start}, self-attn only)" if K > 0
            else "In-context: disabled",
            f"Pos embed: fixed sin-cos (frozen)",
            f"Encoder levels: {self.encoder.level_sizes}",
            f"Image tokens: {G}×{G} = {num_img}",
            f"Condition tokens: {total_cond} "
            f"({' + '.join(f'{s}×{s}' for s in self.encoder.level_sizes)})",
            f"Self-attn seq: {num_img}" + (f" → {K + num_img} (from layer {self.in_context_start})" if K > 0 else ""),
            f"Cross-attn: ({num_img}, {total_cond}) spatial mask",
            f"Level drop: {self.level_drop} (min_keep={self.min_keep_levels}, "
            f"after_steps={self.level_drop_after_steps})",
            f"CFG uncond drop: {self.uncond_drop_prob}",
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_image: torch.Tensor | None = None,
        return_uncond: bool = False,
        num_active_levels: int | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        K = self.in_context_len

        # ── Encode condition ──
        if return_uncond or cond_image is None:
            cond_tokens_list = []
            for s in self.encoder.level_sizes:
                cond_tokens_list.append(
                    self.null_cond[str(s)].expand(B, -1, -1).to(dtype))
            keep_levels = torch.zeros(B, device=device, dtype=torch.long)
        else:
            level_features = self.encoder.forward_injection(cond_image)

            if self.discretizer is not None:
                for s, feat_2d in level_features.items():
                    D = feat_2d.shape[1]
                    tokens_2d = feat_2d.flatten(2).transpose(1, 2)
                    quant_tokens, _ = self.discretizer(tokens_2d)
                    level_features[s] = quant_tokens.transpose(
                        1, 2).view(B, D, s, s)

            if self.training and self.cond_noise_std > 0:
                for s, feat_2d in level_features.items():
                    if self.cond_noise_relative:
                        scale = feat_2d.detach().flatten(1).std(
                            dim=1, keepdim=True).clamp_min(1e-6)
                        scale = scale.view(B, 1, 1, 1)
                    else:
                        scale = 1.0
                    level_features[s] = feat_2d + (
                        self.cond_noise_std
                        * scale
                        * torch.randn_like(feat_2d))

            cond_tokens_list = []
            for s in self.encoder.level_sizes:
                feat = level_features[s]
                tokens = feat.flatten(2).transpose(1, 2)
                cond_tokens_list.append(tokens)

            if self.training and self.level_drop and self._level_drop_enabled:
                keep_levels = self._sample_keep_levels(B, device)
            elif num_active_levels is not None:
                keep_levels = torch.full(
                    (B,), num_active_levels, device=device, dtype=torch.long)
            else:
                keep_levels = torch.full(
                    (B,), self.num_levels, device=device, dtype=torch.long)

            if self.training and self.uncond_drop_prob > 0:
                drop_mask = (
                    torch.rand(B, device=device) < self.uncond_drop_prob)
                keep_levels = torch.where(
                    drop_mask, torch.zeros_like(keep_levels), keep_levels)

            # Replace dropped levels with learned null embeddings
            for i, s in enumerate(self.encoder.level_sizes):
                cf_idx = self._size_to_cf_idx[s]
                active = (cf_idx < keep_levels).float()
                null_tokens = self.null_cond[str(s)].expand(B, -1, -1).to(dtype)
                mask = active.unsqueeze(-1).unsqueeze(-1)
                cond_tokens_list[i] = (
                    cond_tokens_list[i] * mask
                    + null_tokens * (1 - mask))

            # ── Per-token random drop (train-only) ──
            # 기본: 각 sample 의 "가장 finest 한 kept level" 에만
            # per-position Bernoulli(p_b) drop 적용 (hierarchical 가정 유지).
            # cond_token_drop_all_levels=True: kept 인 모든 level 에 동일 적용
            # (coarse level 도 random drop). dropped(null로 치환된) level 은
            # 어차피 null 이라 영향 없음.
            # keep_levels[b]=0 (fully uncond) 인 샘플은 kept level 자체가
            # 없으므로 drop 대상 없음 → 자동 skip.
            if self.training and self.cond_token_drop_prob > 0.0:
                u = torch.rand(B, 1, device=device)
                if self.cond_token_drop_linear:
                    u = u.sqrt()                               # PDF f(p)=2p, biased toward 1
                p_sample = u * self.cond_token_drop_prob       # (B, 1)
                finest_kept_cf_idx = keep_levels - 1           # (B,)
                for i, s in enumerate(self.encoder.level_sizes):
                    cf_idx = self._size_to_cf_idx[s]
                    if self.cond_token_drop_all_levels:
                        # apply to every kept level for this sample
                        active_lvl = (cf_idx < keep_levels).view(B, 1)
                    else:
                        active_lvl = (finest_kept_cf_idx == cf_idx).view(B, 1)
                    tokens = cond_tokens_list[i]               # (B, s*s, D)
                    null_tokens = self.null_cond[str(s)].expand(
                        B, -1, -1).to(dtype)
                    drop = ((torch.rand(B, s * s, device=device) < p_sample)
                            & active_lvl)                      # (B, s*s)
                    keep = (~drop).to(tokens.dtype).unsqueeze(-1)
                    cond_tokens_list[i] = (
                        tokens * keep + null_tokens * (1.0 - keep))

        # ── Project cond tokens + positional embeddings ──
        cond_tokens = torch.cat(cond_tokens_list, dim=1)  # (B, M, feat_ch)
        cond_tokens = self.cond_proj(cond_tokens)          # (B, M, H)

        offset = 0
        for s in self.encoder.level_sizes:
            n = s * s
            cond_tokens[:, offset:offset + n] = (
                cond_tokens[:, offset:offset + n]
                + self.cond_pos_embeds[str(s)])
            offset += n

        # ── Patchify image tokens ──
        img_tokens = self.patch_embed(x_t)         # (B, G*G, H)
        img_tokens = img_tokens + self.pos_embed    # fixed sin-cos

        # ── Timestep embedding ──
        t_freq = _sinusoidal_timestep_embedding(t, self._t_freq_dim)
        t_freq = t_freq.to(dtype=dtype)
        c = self.time_embed(t_freq)                 # (B, hidden)

        # ── Transformer: self-attn → cross-attn → MLP per block ──
        tokens = img_tokens                         # (B, N_img, H)

        for i, block in enumerate(self.blocks):
            # In-context token insertion at in_context_start
            if K > 0 and i == self.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
                ic_tokens = ic_tokens + self.in_context_posemb
                tokens = torch.cat([ic_tokens, tokens], dim=1)

            has_prefix = (K > 0 and i >= self.in_context_start)

            # Self-attention: no mask needed (all tokens attend to all)
            # RoPE: image gets 2D spatial, in-context prefix gets identity
            if has_prefix:
                rope_cos = self._rope_cos_ext
                rope_sin = self._rope_sin_ext
            else:
                rope_cos = self._rope_cos
                rope_sin = self._rope_sin

            # Cross-attention mask: spatial alignment + level drop
            xa_mask = self._get_xa_mask(keep_levels, has_prefix=has_prefix)

            tokens = block(tokens, c, cond=cond_tokens,
                           sa_mask=None, xa_mask=xa_mask,
                           rope_cos=rope_cos, rope_sin=rope_sin)

        # ── Extract image tokens → final layer → unpatchify ──
        if K > 0:
            img_out = tokens[:, K:, :]   # skip in-context prefix
        else:
            img_out = tokens
        img_out = self.final_layer(img_out, c)
        pred = self._unpatchify(img_out)

        if return_aux_loss:
            aux = {}
            if self.use_vq and self.discretizer is not None:
                if self.discretizer.last_vq_loss is not None:
                    aux["vq_loss"] = self.discretizer.last_vq_loss
                if self.discretizer.last_perplexity is not None:
                    aux["vq_perplexity"] = self.discretizer.last_perplexity
                    aux["vq_usage"] = self.discretizer.last_usage
            return pred, aux

        return pred

    def forward_from_level_features(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        level_features: dict[int, torch.Tensor],
        return_uncond: bool = False,
    ) -> torch.Tensor:
        """Forward pass using pre-decoded level features (bypass encoder).

        Used by discrete diffusion decoding: token IDs are decoded to
        continuous features externally, then injected here as conditioning.

        Args:
            x_t: (B, C, H, W) noisy image
            t:   (B,) timestep
            level_features: {spatial_size: (B, feat_channels, S, S)}
            return_uncond: if True, use null conditioning
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        K = self.in_context_len

        if return_uncond:
            cond_tokens_list = []
            for s in self.encoder.level_sizes:
                cond_tokens_list.append(
                    self.null_cond[str(s)].expand(B, -1, -1).to(dtype))
            keep_levels = torch.zeros(B, device=device, dtype=torch.long)
        else:
            cond_tokens_list = []
            for s in self.encoder.level_sizes:
                feat = level_features[s]
                tokens = feat.flatten(2).transpose(1, 2)  # (B, S*S, feat_ch)
                cond_tokens_list.append(tokens)
            keep_levels = torch.full(
                (B,), self.num_levels, device=device, dtype=torch.long)

        # Project cond tokens + positional embeddings
        cond_tokens = torch.cat(cond_tokens_list, dim=1)
        cond_tokens = self.cond_proj(cond_tokens)

        offset = 0
        for s in self.encoder.level_sizes:
            n = s * s
            cond_tokens[:, offset:offset + n] = (
                cond_tokens[:, offset:offset + n]
                + self.cond_pos_embeds[str(s)])
            offset += n

        # Patchify + pos embed
        img_tokens = self.patch_embed(x_t)
        img_tokens = img_tokens + self.pos_embed

        # Timestep embedding
        t_freq = _sinusoidal_timestep_embedding(t, self._t_freq_dim)
        t_freq = t_freq.to(dtype=dtype)
        c = self.time_embed(t_freq)

        # Transformer blocks
        tokens = img_tokens
        for i, block in enumerate(self.blocks):
            if K > 0 and i == self.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
                ic_tokens = ic_tokens + self.in_context_posemb
                tokens = torch.cat([ic_tokens, tokens], dim=1)

            has_prefix = (K > 0 and i >= self.in_context_start)
            if has_prefix:
                rope_cos = self._rope_cos_ext
                rope_sin = self._rope_sin_ext
            else:
                rope_cos = self._rope_cos
                rope_sin = self._rope_sin

            xa_mask = self._get_xa_mask(keep_levels, has_prefix=has_prefix)
            tokens = block(tokens, c, cond=cond_tokens,
                           sa_mask=None, xa_mask=xa_mask,
                           rope_cos=rope_cos, rope_sin=rope_sin)

        if K > 0:
            img_out = tokens[:, K:, :]
        else:
            img_out = tokens
        img_out = self.final_layer(img_out, c)
        return self._unpatchify(img_out)
