"""
Lightweight DiT (Diffusion Transformer) backbone for *discrete* tokens.
Based on mdlm/models/dit.py but stripped of flash-attn / HuggingFace deps
so it runs on any GPU.

Input : token indices (B, L)  with values in [0, vocab_size)
Output: logits          (B, L, vocab_size)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ────────────────────────────────────────────────────────────
#  Sinusoidal timestep embedding (same as MDLM / original DiT)
# ────────────────────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


# ────────────────────────────────────────────────────────────
#  Learnable 1-D positional embedding
# ────────────────────────────────────────────────────────────

class LearnedPosEmb(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.emb[:, : x.size(1), :]


# ────────────────────────────────────────────────────────────
#  Transformer block (vanilla multi-head self-attention + MLP)
# ────────────────────────────────────────────────────────────

class DDiTBlock(nn.Module):
    """Adaptive-LN DiT block (without flash-attn).

    Optionally supports cross-attention conditioning when
    ``use_cross_attn=True``.  If ``cond_tokens`` is ``None`` during
    forward, the cross-attention is simply skipped.

    When ``causal=True``, self-attention uses a causal mask so each
    position can only attend to itself and earlier positions.
    """

    def __init__(self, dim: int, n_heads: int, cond_dim: int,
                 mlp_ratio: int = 4, dropout: float = 0.1,
                 use_cross_attn: bool = False, causal: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.use_cross_attn = use_cross_attn
        self.causal = causal
        head_dim = dim // n_heads

        self.norm1 = LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # ---- optional cross-attention ----
        if use_cross_attn:
            self.norm_cross = LayerNorm(dim)
            self.q_cross = nn.Linear(dim, dim, bias=False)
            self.kv_cross = nn.Linear(dim, 2 * dim, bias=False)
            self.out_cross = nn.Linear(dim, dim, bias=False)
            self.gate_cross_dense = nn.Linear(cond_dim, dim)
            nn.init.zeros_(self.gate_cross_dense.weight)
            nn.init.zeros_(self.gate_cross_dense.bias)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )

        self.dropout = nn.Dropout(dropout)

        # adaLN modulation: 6 * dim  (shift_msa, scale_msa, gate_msa,
        #                              shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x:           (B, L, D) hidden states
            c:           (B, cond_dim) timestep conditioning
            cond_tokens: (B, L_cond, D) optional cross-attn context
        """
        B, L, D = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)
        )

        # ---- self-attention ----
        h = modulate(self.norm1(x), shift_msa.squeeze(1), scale_msa.squeeze(1))
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, Dh)
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal)  # (B, H, L, Dh)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)

        x = x + gate_msa * self.dropout(self.out_proj(attn_out))

        # ---- cross-attention (optional) ----
        if self.use_cross_attn and cond_tokens is not None:
            L_c = cond_tokens.shape[1]
            h_c = self.norm_cross(x)
            q_c = self.q_cross(h_c).reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
            kv_c = self.kv_cross(cond_tokens).reshape(B, L_c, 2, self.n_heads, D // self.n_heads)
            k_c, v_c = kv_c.unbind(dim=2)
            k_c = k_c.transpose(1, 2)
            v_c = v_c.transpose(1, 2)
            cross_out = F.scaled_dot_product_attention(q_c, k_c, v_c)
            cross_out = cross_out.transpose(1, 2).reshape(B, L, D)
            gate_c = torch.sigmoid(self.gate_cross_dense(c)).unsqueeze(1)  # (B,1,D)
            x = x + gate_c * self.dropout(self.out_cross(cross_out))

        # ---- MLP ----
        h = modulate(self.norm2(x), shift_mlp.squeeze(1), scale_mlp.squeeze(1))
        x = x + gate_mlp * self.dropout(self.mlp(h))
        return x


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift.squeeze(1), scale.squeeze(1))
        return self.linear(x)


# ────────────────────────────────────────────────────────────
#  TokenBridge-style Factorized AR Head for FSQ tokens
# ────────────────────────────────────────────────────────────

class FactorizedARHead(nn.Module):
    """Predicts FSQ token by autoregressively predicting each dimension.

    Instead of a single (hidden_size → vocab_size) linear layer,
    this decomposes prediction into D sequential steps:
        p(token) = Π_{d=0}^{D-1} p(dim_d | dim_{<d}, hidden_state)

    Each step is a small classifier over L_d levels (e.g., 5 or 8),
    conditioned on the backbone hidden state + previously predicted dims.

    Args:
        hidden_size:  backbone output dimension
        cond_dim:     adaLN conditioning dimension
        fsq_levels:   list of FSQ levels, e.g. [8, 8, 8, 5, 5, 5]
        ar_dim:       internal dimension of the AR head
        n_ar_layers:  number of transformer layers in the AR head
    """

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        fsq_levels: list[int],
        ar_dim: int = 256,
        n_ar_layers: int = 2,
    ):
        super().__init__()
        self.fsq_levels = list(fsq_levels)
        self.n_dims = len(fsq_levels)
        self.ar_dim = ar_dim
        self.hidden_size = hidden_size

        # Compute flat vocab properties for compatibility
        import functools, operator
        self.vocab_size = functools.reduce(operator.mul, fsq_levels, 1)

        # Basis for flat index ↔ per-dim code conversion
        basis = [1]
        for l in fsq_levels[:-1]:
            basis.append(basis[-1] * l)
        self.register_buffer("_basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("_levels", torch.tensor(fsq_levels, dtype=torch.long))

        # adaLN (same interface as DDiTFinalLayer)
        self.norm = LayerNorm(hidden_size)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

        # Project backbone hidden → AR dim
        self.input_proj = nn.Linear(hidden_size, ar_dim)

        # Per-dim embeddings (for already-predicted dims)
        # Each dim value is embedded and added to the running state
        max_level = max(fsq_levels)
        self.dim_value_embs = nn.ModuleList([
            nn.Embedding(fsq_levels[d], ar_dim)
            for d in range(self.n_dims)
        ])
        # Dim position embedding (which dim are we predicting)
        self.dim_pos_emb = nn.Embedding(self.n_dims, ar_dim)

        # Small AR transformer (shared across all dims)
        if n_ar_layers > 0:
            ar_layer = nn.TransformerEncoderLayer(
                d_model=ar_dim,
                nhead=max(1, ar_dim // 64),
                dim_feedforward=ar_dim * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.ar_transformer = nn.TransformerEncoder(
                ar_layer, num_layers=n_ar_layers)
        else:
            # Simple MLP fallback
            self.ar_transformer = None

        # Per-dim output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(ar_dim, fsq_levels[d])
            for d in range(self.n_dims)
        ])
        # Zero-init output heads
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def flat_index_to_dims(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Convert flat token IDs to per-dim codes.
        Args:
            flat_indices: (...) int64 in [0, vocab_size)
        Returns:
            dims: (..., n_dims) int64, each in [0, level_d)
        """
        indices = flat_indices.unsqueeze(-1)  # (..., 1)
        return ((indices // self._basis) % self._levels).long()

    def dims_to_flat_index(self, dim_codes: torch.Tensor) -> torch.Tensor:
        """Convert per-dim codes back to flat token IDs.
        Args:
            dim_codes: (..., n_dims) int64
        Returns:
            flat_indices: (...) int64
        """
        return (dim_codes * self._basis).sum(dim=-1)

    def forward_factorized(
        self, x: torch.Tensor, c: torch.Tensor,
        target_dims: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Compute per-dim logits.

        Args:
            x:  (B, L, hidden_size) backbone output (pre-output-layer)
            c:  (B, cond_dim) timestep conditioning
            target_dims: (B, L, n_dims) int64 ground-truth dim codes
                         (teacher forcing during training)
        Returns:
            list of D tensors, each (B, L, level_d) — logits per dim
        """
        # adaLN
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift.squeeze(1), scale.squeeze(1))

        B, L, _ = x.shape
        h = self.input_proj(x)  # (B, L, ar_dim)

        dim_logits = []
        for d in range(self.n_dims):
            # Add dim position embedding
            h_d = h + self.dim_pos_emb.weight[d]  # (B, L, ar_dim)

            # Predict dim d
            logits_d = self.output_heads[d](h_d)  # (B, L, level_d)
            dim_logits.append(logits_d)

            # Update h with the (teacher-forced) value of dim d
            if d < self.n_dims - 1:
                if target_dims is not None:
                    # Training: teacher forcing
                    dim_val = target_dims[:, :, d]  # (B, L)
                else:
                    # Inference: use argmax of predicted logits
                    dim_val = logits_d.argmax(dim=-1)  # (B, L)
                dim_emb = self.dim_value_embs[d](dim_val)  # (B, L, ar_dim)
                h = h + dim_emb

        return dim_logits

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compatible interface: returns flat (B, L, vocab_size+1) logits.

        Runs AR head greedily per-dim, returns sparse logits with the
        predicted token having high probability.
        """
        tokens, log_conf = self.sample_with_confidence(x, c)
        # Build sparse logits: predicted token = 0, everything else = -100
        B, L = tokens.shape
        logits = torch.full((B, L, self.vocab_size + 1), -100.0,
                           device=x.device, dtype=x.dtype)
        logits.scatter_(-1, tokens.unsqueeze(-1), 0.0)
        return logits

    def sample_with_confidence(
        self, x: torch.Tensor, c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """AR-greedy sample + proper confidence for each position.

        Runs the AR head autoregressively:
          dim0 ~ p(d0|h), dim1 ~ p(d1|d0,h), ..., dimD ~ p(dD|d<D,h)

        Returns:
            tokens: (B, L) flat token IDs
            log_confidence: (B, L) log p(token) = Σ_d log p(d_d | d_{<d}, h)
        """
        # adaLN
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x_norm = modulate(self.norm(x), shift.squeeze(1), scale.squeeze(1))

        B, L, _ = x_norm.shape
        h = self.input_proj(x_norm)  # (B, L, ar_dim)

        dim_codes = []
        log_conf = torch.zeros(B, L, device=x.device, dtype=x.dtype)

        for d in range(self.n_dims):
            h_d = h + self.dim_pos_emb.weight[d]
            logits_d = self.output_heads[d](h_d)       # (B, L, level_d)
            log_probs_d = F.log_softmax(logits_d, dim=-1)

            # Sample (or argmax)
            pred_d = logits_d.argmax(dim=-1)            # (B, L)
            dim_codes.append(pred_d)

            # Accumulate confidence
            lp = torch.gather(log_probs_d, dim=-1,
                             index=pred_d.unsqueeze(-1)).squeeze(-1)
            log_conf = log_conf + lp

            # Feed back for next dim
            if d < self.n_dims - 1:
                dim_emb = self.dim_value_embs[d](pred_d)
                h = h + dim_emb

        dim_codes = torch.stack(dim_codes, dim=-1)  # (B, L, D)
        tokens = self.dims_to_flat_index(dim_codes)  # (B, L)

        return tokens, log_conf


# ────────────────────────────────────────────────────────────
#  Full DiT model
# ────────────────────────────────────────────────────────────

class DIT(nn.Module):
    """
    DiT backbone for discrete diffusion on flattened token sequences.

    Supports:
    - Sudoku grids (pos_emb_type="2d" / "sudoku")
    - Multi-resolution image tokens (pos_emb_type="multires")
    - Generic 1D sequences (pos_emb_type="1d")
    - Optional class-label conditioning via adaLN (num_classes > 0)
    - Optional cross-attention conditioning (use_cross_attn=True)

    Args:
        vocab_size:   number of discrete token values (including mask token)
        seq_len:      sequence length
        hidden_size:  transformer hidden dim
        n_heads:      number of attention heads
        n_blocks:     number of transformer blocks
        cond_dim:     conditioning vector size (timestep embedding dim)
        mlp_ratio:    MLP expansion ratio
        dropout:      dropout probability
        num_classes:  number of class labels for adaLN conditioning (0=disabled)
        level_sizes:  list of spatial sizes per level for multires pos emb,
                      e.g. [8, 4, 2, 1].  Required when pos_emb_type="multires".
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 784,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_blocks: int = 6,
        cond_dim: int = 128,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_cross_attn: bool = False,
        cond_seq_codebook_size: int = 0,
        cond_seq_len: int = 0,
        sudoku_hw: int = 9,
        use_sudoku_pos_emb: bool = False,
        pos_emb_type: str = "2d",
        num_classes: int = 0,
        level_sizes: list | None = None,
        causal: bool = False,
        # ── TokenBridge-style factorized AR head ──
        factorized_head: bool = False,
        fsq_levels: list[int] | None = None,
        ar_head_dim: int = 256,
        ar_head_layers: int = 2,
        # ── Continuous input mode (for diffusion head) ──
        continuous_mode: bool = False,
        continuous_dim: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.use_cross_attn = use_cross_attn
        self.num_classes = num_classes
        self.causal = causal
        self.continuous_mode = continuous_mode

        # backward compat: old use_sudoku_pos_emb=True → pos_emb_type="sudoku"
        if use_sudoku_pos_emb:
            pos_emb_type = "sudoku"

        assert pos_emb_type in ("1d", "2d", "sudoku", "multires"), (
            f"pos_emb_type must be '1d', '2d', 'sudoku', or 'multires', "
            f"got '{pos_emb_type}'")
        self.pos_emb_type = pos_emb_type

        self.factorized_head = factorized_head

        # ── Continuous mode: linear projection + mask embedding ──
        if continuous_mode:
            assert continuous_dim > 0, "continuous_dim required for continuous_mode"
            self.cont_input_proj = nn.Linear(continuous_dim, hidden_size)
            self.cont_mask_emb = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.dim_embs = None
            self.mask_emb = None
            self.token_emb = None
        elif factorized_head and fsq_levels is not None:
            # Factorized embedding: per-dim embeddings summed together
            # Token ID → FSQ dim codes → sum of per-dim embeddings
            # Much fewer params (39 vs 64K) and structurally meaningful
            import functools, operator
            _levels = fsq_levels
            _basis = [1]
            for l in _levels[:-1]:
                _basis.append(_basis[-1] * l)
            self.register_buffer("_emb_basis",
                                 torch.tensor(_basis, dtype=torch.long))
            self.register_buffer("_emb_levels",
                                 torch.tensor(_levels, dtype=torch.long))
            self.dim_embs = nn.ModuleList([
                nn.Embedding(l, hidden_size) for l in _levels
            ])
            # Mask token gets its own embedding (last index = vocab_size - 1)
            self.mask_emb = nn.Parameter(torch.randn(hidden_size) * 0.02)
            self.token_emb = None  # not used
        else:
            self.dim_embs = None
            self.mask_emb = None
            self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = LearnedPosEmb(seq_len, hidden_size)

        if pos_emb_type in ("2d", "sudoku"):
            if seq_len != sudoku_hw * sudoku_hw:
                raise ValueError(
                    f"pos_emb_type='{pos_emb_type}' requires seq_len="
                    f"{sudoku_hw*sudoku_hw}, got seq_len={seq_len}"
                )
            self.row_emb = nn.Embedding(sudoku_hw, hidden_size)
            self.col_emb = nn.Embedding(sudoku_hw, hidden_size)
            rows = torch.arange(seq_len) // sudoku_hw
            cols = torch.arange(seq_len) % sudoku_hw
            self.register_buffer("row_idx", rows, persistent=False)
            self.register_buffer("col_idx", cols, persistent=False)

        if pos_emb_type == "sudoku":
            self.box_emb = nn.Embedding(sudoku_hw, hidden_size)
            rows = torch.arange(seq_len) // sudoku_hw
            cols = torch.arange(seq_len) % sudoku_hw
            boxes = (rows // 3) * 3 + (cols // 3)
            self.register_buffer("box_idx", boxes, persistent=False)

        # ── multi-resolution positional embedding ──
        if pos_emb_type == "multires":
            assert level_sizes is not None, (
                "level_sizes required for pos_emb_type='multires'")
            self.level_sizes = list(level_sizes)
            expected_len = sum(s * s for s in level_sizes)
            if seq_len != expected_len:
                raise ValueError(
                    f"pos_emb_type='multires' with level_sizes={level_sizes} "
                    f"expects seq_len={expected_len}, got {seq_len}")
            n_levels = len(level_sizes)
            max_s = max(level_sizes)
            self.mr_level_emb = nn.Embedding(n_levels, hidden_size)
            self.mr_row_emb = nn.Embedding(max_s, hidden_size)
            self.mr_col_emb = nn.Embedding(max_s, hidden_size)
            level_ids, row_ids, col_ids = [], [], []
            for li, s in enumerate(level_sizes):
                for r in range(s):
                    for c in range(s):
                        level_ids.append(li)
                        row_ids.append(r)
                        col_ids.append(c)
            self.register_buffer("mr_level_idx",
                                 torch.tensor(level_ids), persistent=False)
            self.register_buffer("mr_row_idx",
                                 torch.tensor(row_ids), persistent=False)
            self.register_buffer("mr_col_idx",
                                 torch.tensor(col_ids), persistent=False)

        self.sigma_map = TimestepEmbedder(cond_dim)

        # ── class-label conditioning (adaLN) ──
        if num_classes > 0:
            # +1 index reserved for "null class" (CFG unconditional)
            self.label_emb = nn.Embedding(num_classes + 1, cond_dim)
        else:
            self.label_emb = None

        # ── optional conditioning embeddings for cross-attention ──
        if use_cross_attn and cond_seq_codebook_size > 0:
            self.cond_emb = nn.Embedding(cond_seq_codebook_size, hidden_size)
            self.cond_pos_emb = LearnedPosEmb(max(cond_seq_len, 1), hidden_size)
        else:
            self.cond_emb = None
            self.cond_pos_emb = None

        # MAR/semanticist-style learnable unconditional token (single vector,
        # broadcast to L_cond when used). Activated only when --cfg_mode=backbone.
        self.null_cond_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, mlp_ratio, dropout,
                      use_cross_attn=use_cross_attn, causal=causal)
            for _ in range(n_blocks)
        ])

        if continuous_mode:
            # No output_layer — returns hidden states for diffusion head
            self.output_layer = None
        elif factorized_head:
            assert fsq_levels is not None, \
                "fsq_levels required when factorized_head=True"
            self.output_layer = FactorizedARHead(
                hidden_size=hidden_size,
                cond_dim=cond_dim,
                fsq_levels=fsq_levels,
                ar_dim=ar_head_dim,
                n_ar_layers=ar_head_layers,
            )
        else:
            self.output_layer = DDiTFinalLayer(hidden_size, vocab_size, cond_dim)

    def _factorized_embed(self, indices: torch.Tensor) -> torch.Tensor:
        """Factorized token embedding: decompose flat ID into per-dim codes,
        embed each dim separately, and sum.

        Mask token (last index) gets a special learned embedding.

        Args:
            indices: (B, L) int64 token IDs, may include mask_index
        Returns:
            (B, L, hidden_size)
        """
        is_mask = (indices == self.vocab_size - 1)

        # Decompose flat index → per-dim codes
        # Clamp to valid range first (mask tokens would be out of range)
        safe_indices = indices.clamp(0, self.vocab_size - 2)
        idx = safe_indices.unsqueeze(-1)  # (B, L, 1)
        dim_codes = ((idx // self._emb_basis) % self._emb_levels).long()  # (B, L, D)

        # Sum per-dim embeddings
        x = torch.zeros(*indices.shape, self.dim_embs[0].embedding_dim,
                        device=indices.device, dtype=self.dim_embs[0].weight.dtype)
        for d, emb in enumerate(self.dim_embs):
            x = x + emb(dim_codes[:, :, d])

        # Replace mask positions with mask embedding
        if is_mask.any():
            x[is_mask] = self.mask_emb

        return x

    def embed_cond(self, cond_indices: torch.Tensor) -> torch.Tensor:
        """Embed conditioning token indices → hidden states for cross-attn.

        Args:
            cond_indices: (B, L_cond) int64
        Returns:
            cond_tokens: (B, L_cond, hidden_size)
        """
        if self.cond_emb is None:
            raise RuntimeError("DIT was created without cond_emb. "
                               "Set use_cross_attn=True and cond_seq_codebook_size>0.")
        h = self.cond_emb(cond_indices)
        h = self.cond_pos_emb(h)
        return h

    def forward(self, indices: torch.Tensor, sigma: torch.Tensor,
                cond_tokens: Optional[torch.Tensor] = None,
                class_labels: Optional[torch.Tensor] = None,
                prefix_mode: bool = False,
                # ── continuous mode inputs ──
                cont_tokens: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Args:
            indices:      (B, L) int64 token indices  [discrete mode]
            sigma:        (B,) float  noise level
            cond_tokens:  (B, L_cond, D) float — external condition embeddings.
                          Can also be (B, L_cond) int64 if cond_emb exists.
            class_labels: (B,) int64 optional class labels for adaLN cond.
            prefix_mode:  if True, prepend cond_tokens as prefix to the
                          data sequence (no cross-attention).  The output
                          is sliced to return only data positions.
            cont_tokens:  (B, L, D_feat) float continuous tokens  [continuous mode]
            mask:         (B, L) bool — True = masked positions  [continuous mode]
        Returns:
            logits: (B, L, vocab_size)  [discrete mode]
            hidden: (B, L, hidden_size) [continuous mode]
        """
        # auto-embed integer conditioning
        if cond_tokens is not None and cond_tokens.dtype in (torch.long, torch.int):
            cond_tokens = self.embed_cond(cond_tokens)

        if self.continuous_mode:
            # ── Continuous input: project tokens, apply mask embedding ──
            assert cont_tokens is not None, \
                "cont_tokens required in continuous_mode"
            x = self.cont_input_proj(cont_tokens)  # (B, L, hidden)
            if mask is not None:
                # Replace masked positions with learned mask embedding
                mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
                x = torch.where(mask_expanded, self.cont_mask_emb.unsqueeze(0).unsqueeze(0), x)
        else:
            # ── Discrete input: embed token indices ──
            indices = indices.long()
            if self.dim_embs is not None:
                x = self._factorized_embed(indices)  # (B, L, D)
            else:
                x = self.token_emb(indices)  # (B, L, D)

        x = self.pos_emb(x)
        if self.pos_emb_type in ("2d", "sudoku"):
            pos = (self.row_emb(self.row_idx) +
                   self.col_emb(self.col_idx))
            if self.pos_emb_type == "sudoku":
                pos = pos + self.box_emb(self.box_idx)
            x = x + pos[None, :, :]
        elif self.pos_emb_type == "multires":
            pos = (self.mr_level_emb(self.mr_level_idx) +
                   self.mr_row_emb(self.mr_row_idx) +
                   self.mr_col_emb(self.mr_col_idx))
            x = x + pos[None, :, :]

        c = F.silu(self.sigma_map(sigma))  # (B, cond_dim)

        # add class-label conditioning
        if class_labels is not None and self.label_emb is not None:
            c = c + F.silu(self.label_emb(class_labels))

        # Prefix mode: prepend cond_tokens, skip cross-attention
        n_prefix = 0
        if prefix_mode and cond_tokens is not None:
            n_prefix = cond_tokens.shape[1]
            x = torch.cat([cond_tokens, x], dim=1)  # (B, C+L, D)
            cond_tokens = None  # don't also use cross-attention

        for blk in self.blocks:
            x = blk(x, c, cond_tokens=cond_tokens)

        # Strip prefix BEFORE output layer (so output layer sees only data)
        if n_prefix > 0:
            x = x[:, n_prefix:, :]

        if self.continuous_mode:
            # Return hidden states — diffusion head will process them
            return x  # (B, L, hidden_size)

        x = self.output_layer(x, c)  # (B, L, vocab_size)

        return x
