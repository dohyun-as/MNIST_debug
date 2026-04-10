"""
Baseline 1D-Conditioned DiT (Semanticist-style conditioning)
=============================================================

Uses the same DiT backbone as MultiResConditionalDiT (JiT-style:
BottleneckPatchEmbed, 2D RoPE, in-context tokens, adaLN-Zero, RMSNorm,
QK-norm, SwiGLU), but replaces multi-resolution spatially-aligned
conditioning with Semanticist-style 1D token conditioning:

  - Encoder: SemanticistViTEncoder → (B, num_slots, slot_dim)
    Causal ViT slots, non-spatial, ordered by importance.
  - FSQ discretization on the flat 1D tokens (same FSQ as ours)
  - Conditioning: tokens are projected to dit_hidden_size and
    concatenated with image tokens in self-attention (no cross-attention,
    no spatial masking — like Semanticist's DiT_with_autoenc_cond).
  - NestedSampler: progressive dropping (keep first N tokens)
  - CFG: drop all cond with uncond_drop_prob → replace with null_cond

This is a baseline for comparing spatially-aligned multi-res conditioning
(ours) vs. 1D unstructured conditioning (Semanticist-style).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Discretizer import FSQDiscretizer, VQDiscretizer
from semanticist_encoder import SemanticistViTEncoder, NestedSampler


# ──────────────────────────────────────────────────────────────────
#  Reuse building blocks from model_multires.py
# ──────────────────────────────────────────────────────────────────

from model_multires import (
    _sinusoidal_timestep_embedding,
    _get_2d_sincos_pos_embed,
    _build_2d_rope,
    _rotate_half,
    _BottleneckPatchEmbed,
    _RMSNorm,
    _SwiGLUFFN,
    _DiTFinalLayer,
)


# ──────────────────────────────────────────────────────────────────
#  DiT Block for 1D conditioning (self-attention only, no cross-attn)
# ──────────────────────────────────────────────────────────────────

class _DiTBlock1D(nn.Module):
    """DiT block with self-attention only (no cross-attention).

    Condition tokens are concatenated with image tokens in self-attention,
    following the Semanticist approach.

    adaLN-Zero produces 6 params:
      self-attn:  shift1, scale1, gate_sa  (3)
      MLP:        shift2, scale2, gate_mlp  (3)
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

        # ── MLP ──
        self.norm2 = _RMSNorm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = _SwiGLUFFN(dim, mlp_hidden, drop=proj_drop)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # adaLN-Zero: 6 params (shift1, scale1, gate_sa, shift2, scale2, gate_mlp)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                rope_cos: torch.Tensor | None = None,
                rope_sin: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) all tokens (image + cond + in-context)
            c: (B, D) timestep conditioning
            rope_cos, rope_sin: (N, head_dim) RoPE buffers
        """
        B, N, D = x.shape
        H = self.n_heads
        hd = D // H

        shift1, scale1, gate_sa, shift2, scale2, gate_mlp = \
            self.adaLN(c).unsqueeze(1).chunk(6, dim=-1)

        # ── Self-attention with QK-norm + RoPE ──
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
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        sa_out = sa_out.transpose(1, 2).reshape(B, N, D)
        x = x + gate_sa * self.proj_drop(self.sa_out_proj(sa_out))

        # ── MLP (SwiGLU) ──
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate_mlp * self.mlp(h)
        return x


# ──────────────────────────────────────────────────────────────────
#  Baseline 1D-Conditioned DiT
# ──────────────────────────────────────────────────────────────────

class Baseline1DConditionalDiT(nn.Module):
    """
    DiT with Semanticist-style 1D conditioning.

    Conditioning flow:
      Image → SemanticistViTEncoder → (B, num_slots, slot_dim)
      → FSQ quantization → project to dit_hidden_size
      → concatenate with image tokens in self-attention

    The model output only takes the image token positions.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        cond_in_channels: int = 3,
        vae_downsample_factor: int = 1,
        # --- Semanticist encoder ---
        num_slots: int = 256,
        slot_dim: int = 16,
        enc_embed_dim: int = 768,
        enc_depth: int = 12,
        enc_num_heads: int = 12,
        enc_drop_path_rate: float = 0.1,
        is_causal: bool = True,
        # --- Nested dropping ---
        enable_nest: bool = True,
        enable_nest_after_steps: int = -1,
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
        # --- Discretization ---
        use_fsq: bool = False,
        fsq_levels: list[int] | None = None,
        fsq_drop_quant_p: float = 0.0,
        fsq_corrupt_tokens_p: float = 0.0,
        use_vq: bool = False,
        vq_codebook_size: int = 512,
        vq_beta: float = 0.25,
    ):
        super().__init__()

        self.image_size = image_size
        self.vae_downsample_factor = vae_downsample_factor
        self.latent_size = image_size // vae_downsample_factor
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.uncond_drop_prob = uncond_drop_prob
        self.enable_nest = enable_nest
        self.enable_nest_after_steps = enable_nest_after_steps
        self._nest_enabled = (enable_nest_after_steps == -1) and enable_nest

        self.dit_patch_size = dit_patch_size
        self.dit_hidden_size = dit_hidden_size
        self.dit_n_heads = dit_n_heads
        self.dit_n_blocks = dit_n_blocks
        self._in_channels = in_channels
        self.in_context_len = dit_in_context_len
        self.in_context_start = dit_in_context_start

        assert self.latent_size % dit_patch_size == 0
        self.grid_size = self.latent_size // dit_patch_size

        # ── Semanticist-style ViT Encoder ──
        self.encoder = SemanticistViTEncoder(
            img_size=image_size,
            patch_size=16,
            in_chans=cond_in_channels,
            embed_dim=enc_embed_dim,
            depth=enc_depth,
            num_heads=enc_num_heads,
            mlp_ratio=4.0,
            num_slots=num_slots,
            slot_dim=slot_dim,
            drop_path_rate=enc_drop_path_rate,
            is_causal=is_causal,
        )

        # ── Nested Sampler ──
        self.nested_sampler = NestedSampler(num_slots)

        # ── Discretizer (optional) ──
        self.use_fsq = use_fsq
        self.use_vq = use_vq
        self.discretizer = None
        if use_fsq:
            assert fsq_levels is not None
            self.discretizer = FSQDiscretizer(
                slot_dim=slot_dim, levels=fsq_levels,
                drop_quant_p=fsq_drop_quant_p,
                corrupt_tokens_p=fsq_corrupt_tokens_p,
            )
        elif use_vq:
            self.discretizer = VQDiscretizer(
                slot_dim=slot_dim, codebook_size=vq_codebook_size,
                beta=vq_beta,
            )

        # ── Condition projection (slot_dim → dit_hidden_size) ──
        self.cond_proj = nn.Linear(slot_dim, dit_hidden_size)

        # ── Null condition (for CFG & nested drop) ──
        self.null_cond = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.normal_(self.null_cond, std=0.02)

        # ── Learnable positional embedding for condition tokens ──
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(1, num_slots, dit_hidden_size))
        nn.init.trunc_normal_(self.cond_pos_embed, std=0.02)

        # ── Patch embedding (JiT BottleneckPatchEmbed) ──
        self.patch_embed = _BottleneckPatchEmbed(
            self.latent_size, dit_patch_size, in_channels,
            dit_bottleneck_dim, dit_hidden_size,
        )

        # ── Positional embeddings: fixed sin-cos (frozen, JiT-style) ──
        num_img_tokens = self.grid_size ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_img_tokens, dit_hidden_size),
            requires_grad=False,
        )

        # ── In-context tokens (JiT-style, optional) ──
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, self.in_context_len, dit_hidden_size))
            nn.init.normal_(self.in_context_posemb, std=0.02)

        # ── Timestep embedding ──
        self._t_freq_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self._t_freq_dim, dit_hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(dit_hidden_size, dit_hidden_size, bias=True),
        )

        # ── 2D RoPE for self-attention ──
        # RoPE is applied to image token positions.
        # Condition tokens and in-context tokens get identity rotation.
        half_head_dim = dit_hidden_size // dit_n_heads // 2
        hw = self.grid_size

        # We'll build RoPE dynamically in forward based on sequence composition
        rope_cos, rope_sin = _build_2d_rope(half_head_dim, hw, 0)
        self.register_buffer('_rope_cos_img', rope_cos)
        self.register_buffer('_rope_sin_img', rope_sin)

        # ── Transformer blocks (JiT-style: self-attn only, middle 50% dropout) ──
        self.blocks = nn.ModuleList()
        for i in range(dit_n_blocks):
            in_middle = (dit_n_blocks // 4 * 3 > i >= dit_n_blocks // 4)
            a_drop = dit_dropout if in_middle else 0.0
            p_drop = dit_dropout if in_middle else 0.0
            self.blocks.append(
                _DiTBlock1D(dit_hidden_size, dit_n_heads, dit_mlp_ratio,
                            attn_drop=a_drop, proj_drop=p_drop))

        # ── Final layer ──
        self.final_layer = _DiTFinalLayer(
            dit_hidden_size, in_channels * dit_patch_size ** 2,
        )

        self._initialize_weights()

    # ──────────────────────────────────────────────────────────────
    #  Weight initialization
    # ──────────────────────────────────────────────────────────────

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Only init DiT blocks and projections, not encoder (already inited)
        for block in self.blocks:
            block.apply(_basic_init)
        self.final_layer.apply(_basic_init)
        self.cond_proj.apply(_basic_init)

        # Fixed sin-cos positional embedding for image tokens
        pos = _get_2d_sincos_pos_embed(self.dit_hidden_size, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos).float().unsqueeze(0))

        # Timestep MLP
        nn.init.normal_(self.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.time_embed[2].weight, std=0.02)

        # Patch embed
        w1 = self.patch_embed.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.patch_embed.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj2.bias, 0)

        # Zero-out adaLN modulation
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ──────────────────────────────────────────────────────────────
    #  Step tracking
    # ──────────────────────────────────────────────────────────────

    @property
    def num_levels(self):
        """Compatibility with multi-res API: baseline has no hierarchy."""
        return 1

    @property
    def level_drop(self):
        """Compatibility: no level drop in baseline."""
        return False

    def set_step(self, step: int):
        if (self.enable_nest_after_steps != -1
                and step >= self.enable_nest_after_steps
                and self.enable_nest):
            self._nest_enabled = True

    # ──────────────────────────────────────────────────────────────
    #  RoPE helper
    # ──────────────────────────────────────────────────────────────

    def _build_rope_for_seq(self, num_prefix: int) -> tuple:
        """Build RoPE cos/sin for [prefix_tokens, image_tokens] sequence.

        Prefix tokens (cond + in-context) get identity rotation (cos=1, sin=0).
        Image tokens get 2D spatial RoPE.
        """
        D = self._rope_cos_img.shape[-1]
        if num_prefix > 0:
            cos_pad = torch.ones(num_prefix, D,
                                 device=self._rope_cos_img.device,
                                 dtype=self._rope_cos_img.dtype)
            sin_pad = torch.zeros(num_prefix, D,
                                  device=self._rope_cos_img.device,
                                  dtype=self._rope_cos_img.dtype)
            rope_cos = torch.cat([cos_pad, self._rope_cos_img], dim=0)
            rope_sin = torch.cat([sin_pad, self._rope_sin_img], dim=0)
        else:
            rope_cos = self._rope_cos_img
            rope_sin = self._rope_sin_img
        return rope_cos, rope_sin

    # ──────────────────────────────────────────────────────────────
    #  Unpatchify
    # ──────────────────────────────────────────────────────────────

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
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
        num_img = self.grid_size ** 2
        K = self.in_context_len
        lines = [
            f"=== Baseline 1D-Conditioned DiT (Semanticist-style) ===",
            f"Image: {self.image_size}x{self.image_size}",
            f"VAE: x{self.vae_downsample_factor} -> latent {self.latent_size}x{self.latent_size}",
            f"Backbone: DiT (patch={self.dit_patch_size}, hidden={self.dit_hidden_size}, "
            f"heads={self.dit_n_heads}, blocks={self.dit_n_blocks})",
            f"Attention: self-attn only (cond tokens concatenated, no spatial mask)",
            f"Encoder: SemanticistViT (slots={self.num_slots}, slot_dim={self.slot_dim}, "
            f"causal={self.encoder.is_causal})",
            f"Conditioning: 1D token concatenation (Semanticist-style)",
            f"Image tokens: {self.grid_size}x{self.grid_size} = {num_img}",
            f"Cond tokens: {self.num_slots} (1D, non-spatial)",
            f"Self-attn seq: {self.num_slots} + {num_img}" +
            (f" + {K} in-context" if K > 0 else ""),
            f"Nested drop: {self.enable_nest} (after_steps={self.enable_nest_after_steps})",
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
        num_active_slots: int | None = None,
        num_active_levels: int | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Args:
            x_t: (B, C, H, W) noisy image/latent
            t:   (B,) timestep (flow matching t in [0,1] or DDPM int)
            cond_image: (B, 3, img_size, img_size) conditioning image
            return_uncond: use null conditioning (for CFG)
            num_active_slots: override slot count at inference
            num_active_levels: ignored (compatibility with multi-res API)
            return_aux_loss: return auxiliary losses (VQ etc.)
        """
        B = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype
        K = self.in_context_len

        # ── Encode condition ──
        if return_uncond or cond_image is None:
            # Full null conditioning
            slots = self.null_cond.expand(B, -1, -1).to(dtype)
        else:
            # Encode image → 1D slots
            slots = self.encoder(cond_image)  # (B, num_slots, slot_dim)

            # Discretize
            if self.discretizer is not None:
                slots, _ = self.discretizer(slots)

            # ── Nested dropping (training) ──
            if self.training and self._nest_enabled:
                drop_mask = self.nested_sampler(B, device)  # (B, num_slots) True=keep
                # Replace dropped slots with null_cond
                null_expanded = self.null_cond.expand(B, -1, -1).to(dtype)
                slots = torch.where(
                    drop_mask.unsqueeze(-1), slots, null_expanded)
            elif num_active_slots is not None:
                # Inference: keep first N slots
                drop_mask = self.nested_sampler(
                    B, device, inference_with_n_slots=num_active_slots)
                null_expanded = self.null_cond.expand(B, -1, -1).to(dtype)
                slots = torch.where(
                    drop_mask.unsqueeze(-1), slots, null_expanded)

            # ── CFG dropout (training) ──
            if self.training and self.uncond_drop_prob > 0:
                cfg_drop = (torch.rand(B, device=device) < self.uncond_drop_prob)
                if cfg_drop.any():
                    null_expanded = self.null_cond.expand(B, -1, -1).to(dtype)
                    cfg_mask = cfg_drop.view(B, 1, 1).float()
                    slots = slots * (1 - cfg_mask) + null_expanded * cfg_mask

        # ── Project cond tokens + positional embedding ──
        cond_tokens = self.cond_proj(slots)  # (B, num_slots, H)
        cond_tokens = cond_tokens + self.cond_pos_embed

        # ── Patchify image tokens ──
        img_tokens = self.patch_embed(x_t)     # (B, G*G, H)
        img_tokens = img_tokens + self.pos_embed

        # ── Timestep embedding ──
        t_freq = _sinusoidal_timestep_embedding(t, self._t_freq_dim)
        t_freq = t_freq.to(dtype=dtype)
        c = self.time_embed(t_freq)  # (B, hidden)

        # ── Build sequence: [cond_tokens, image_tokens] ──
        # Semanticist concatenates cond at the end, but for RoPE it's
        # cleaner to put cond tokens first (they get identity rotation)
        # and image tokens after (they get 2D spatial RoPE).
        tokens = torch.cat([cond_tokens, img_tokens], dim=1)
        num_prefix = self.num_slots  # cond tokens before image

        for i, block in enumerate(self.blocks):
            # In-context token insertion
            if K > 0 and i == self.in_context_start:
                ic_tokens = c.unsqueeze(1).expand(-1, K, -1)
                ic_tokens = ic_tokens + self.in_context_posemb
                # Insert between cond and image: [cond, in_context, image]
                cond_part = tokens[:, :self.num_slots]
                img_part = tokens[:, self.num_slots:]
                tokens = torch.cat([cond_part, ic_tokens, img_part], dim=1)
                num_prefix = self.num_slots + K

            cur_prefix = self.num_slots + (K if (K > 0 and i >= self.in_context_start) else 0)
            rope_cos, rope_sin = self._build_rope_for_seq(cur_prefix)

            tokens = block(tokens, c,
                           rope_cos=rope_cos, rope_sin=rope_sin)

        # ── Extract image tokens → final layer → unpatchify ──
        img_out = tokens[:, num_prefix:]
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
