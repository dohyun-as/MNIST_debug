"""
Discrete (absorbing-state) diffusion  –  MDLM-style.

Implements the full forward-process noising, loss computation, and
reverse-process sampling, exactly matching the MDLM (Masked Diffusion
Language Model) paper & codebase:

  * subs parameterization  (continuous-time, ELBO-based)
  * loglinear noise schedule
  * absorbing (mask) state forward process  q(x_t | x_0)
  * DDPM-cache reverse sampler

References
----------
- Sahoo et al., "Simple and Effective Masked Diffusion Language Models", 2024.
- https://github.com/kuleshov-group/mdlm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from noise_schedule import Noise, get_noise
from dit_model import DIT


NEG_INF = -1_000_000.0


# ────────────────────────────────────────────────────────────
#  Helper
# ────────────────────────────────────────────────────────────

def _sample_categorical(categorical_probs: Tensor) -> Tensor:
    """Gumbel-max trick to sample from categorical probs."""
    gumbel_norm = (
        1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclass
class LossOutput:
    loss: Tensor        # scalar (mean over tokens)
    nlls: Tensor        # (B, L) per-token NLL
    token_mask: Tensor  # (B, L) which tokens are valid


# ────────────────────────────────────────────────────────────
#  Main class
# ────────────────────────────────────────────────────────────

class DiscreteDiffusion(nn.Module):
    """
    Absorbing-state discrete diffusion following MDLM.

    Parameters
    ----------
    backbone : nn.Module
        A model that takes ``(indices, sigma)`` and returns logits
        of shape ``(B, L, vocab_size)``.  ``DIT`` is the default.
    vocab_size : int
        Number of *data* categories (mask token is appended as ``vocab_size``).
    factorized_head : bool
        If True, uses TokenBridge-style factorized per-dim loss during
        training. The backbone must have a FactorizedARHead output layer.
    noise_type : str
        ``"loglinear"`` (default) or ``"cosine"``.
    noise_eps : float
        Small epsilon for noise schedule.
    antithetic_sampling : bool
        Use antithetic time sampling (reduces variance).
    importance_sampling : bool
        Use importance-weighted time sampling.
    change_of_variables : bool
        Use the change-of-variables ELBO estimator.
    sampling_eps : float
        Minimum time during training time sampling.
    """

    def __init__(
        self,
        backbone: nn.Module,
        vocab_size: int,
        noise_type: str = "loglinear",
        noise_eps: float = 1e-3,
        antithetic_sampling: bool = True,
        importance_sampling: bool = False,
        change_of_variables: bool = False,
        sampling_eps: float = 1e-3,
        # ── Continuous mode with diffusion head (MAR-style) ──
        diff_head: Optional[nn.Module] = None,
        diffusion_batch_mul: int = 1,
        # ── MDLM-style time-conditioning toggle ──
        time_conditioning: bool = False,
    ):
        super().__init__()
        assert not (change_of_variables and importance_sampling), \
            "Cannot use both change_of_variables and importance_sampling"
        self.time_conditioning = time_conditioning

        self.backbone = backbone
        self.data_vocab_size = vocab_size
        # mask token = last index (appended category)
        self.mask_index = vocab_size
        self.vocab_size = vocab_size + 1   # includes mask

        # ── Continuous mode with diffusion head ──
        self.continuous_mode = getattr(backbone, 'continuous_mode', False)
        self.diff_head = diff_head
        self.diffusion_batch_mul = diffusion_batch_mul

        # Detect factorized head from backbone
        from dit_model import FactorizedARHead
        self.factorized_head = getattr(backbone, 'factorized_head', False)
        if self.factorized_head:
            self._ar_head: FactorizedARHead = backbone.output_layer

        self.noise = get_noise(noise_type, eps=noise_eps)

        self.antithetic_sampling = antithetic_sampling
        self.importance_sampling = importance_sampling
        self.change_of_variables = change_of_variables
        self.sampling_eps = sampling_eps

    # ─────────────── time-conditioning helper ──────────────

    def _t(self, sigma: Tensor) -> Tensor:
        """Return sigma or zeros based on time_conditioning flag.

        MDLM paper: for absorbing-state diffusion, network can infer
        noise level from input (presence of [MASK]) so explicit time
        conditioning is not needed. When ``time_conditioning=False``
        we pass zero to sigma_map everywhere.
        """
        if self.time_conditioning:
            return sigma
        return torch.zeros_like(sigma)

    # ─────────────── forward-process noising ───────────────

    def q_xt(self, x: Tensor, move_chance: Tensor) -> Tensor:
        """Apply absorbing noise: independently replace each token with
        ``mask_index`` with probability ``move_chance``.

        Args:
            x:           (B, L) int64 clean tokens
            move_chance: (B, 1) float in [0, 1]
        Returns:
            xt: (B, L) int64 noised tokens
        """
        move = torch.rand_like(x.float()) < move_chance
        return torch.where(move, self.mask_index, x)

    # ─────────────── parameterization ──────────────────────

    def _subs_parameterization(self, logits: Tensor, xt: Tensor) -> Tensor:
        """MDLM 'subs' (substitution) parameterization.

        * For masked positions: network output is a distribution over
          *unmasked* tokens (mask logit = -inf).
        * For unmasked positions: output is a delta at the current token.

        Returns log-probabilities (B, L, vocab_size).
        """
        # zero out mask logit
        logits[:, :, self.mask_index] += NEG_INF

        # normalize
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # for unmasked positions → delta at current token
        unmasked = xt != self.mask_index
        logits[unmasked] = NEG_INF
        logits[unmasked, xt[unmasked]] = 0.0

        return logits

    def _run_backbone_hidden(self, x: Tensor, sigma: Tensor,
                             cond_tokens: Optional[Tensor] = None,
                             class_labels: Optional[Tensor] = None):
        """Run backbone up to (but not including) output_layer.
        Returns hidden states (B, L, hidden_size) and cond vector c.
        Only used when factorized_head=True.
        """
        backbone = self.backbone
        x_long = x.long()
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        sigma = self._t(sigma)

        if cond_tokens is not None and cond_tokens.dtype in (torch.long, torch.int):
            cond_tokens = backbone.embed_cond(cond_tokens)

        if backbone.dim_embs is not None:
            h = backbone._factorized_embed(x_long)
        else:
            h = backbone.token_emb(x_long)
        h = backbone.pos_emb(h)
        if backbone.pos_emb_type in ("2d", "sudoku"):
            pos = (backbone.row_emb(backbone.row_idx) +
                   backbone.col_emb(backbone.col_idx))
            if backbone.pos_emb_type == "sudoku":
                pos = pos + backbone.box_emb(backbone.box_idx)
            h = h + pos[None, :, :]
        elif backbone.pos_emb_type == "multires":
            pos = (backbone.mr_level_emb(backbone.mr_level_idx) +
                   backbone.mr_row_emb(backbone.mr_row_idx) +
                   backbone.mr_col_emb(backbone.mr_col_idx))
            h = h + pos[None, :, :]

        c = F.silu(backbone.sigma_map(sigma))
        if hasattr(backbone, 'label_emb') and backbone.label_emb is not None:
            if class_labels is not None:
                c = c + F.silu(backbone.label_emb(class_labels))

        n_prefix = 0
        use_prefix = (cond_tokens is not None)
        if use_prefix:
            n_prefix = cond_tokens.shape[1]
            h = torch.cat([cond_tokens, h], dim=1)
            cond_tokens = None

        for blk in backbone.blocks:
            h = blk(h, c, cond_tokens=cond_tokens)

        if n_prefix > 0:
            h = h[:, n_prefix:, :]

        return h, c

    def forward(self, x: Tensor, sigma: Tensor,
                cond_tokens: Optional[Tensor] = None,
                class_labels: Optional[Tensor] = None) -> Tensor:
        """Run backbone + apply subs parameterization. Returns log p_θ.

        Args:
            x:            (B, L)  int64  noised tokens
            sigma:        (B,)    float  noise level σ(t)
            cond_tokens:  (B, C, D) optional prefix conditioning.
                          Prepended to x inside backbone, output is
                          sliced back to (B, L, vocab_size).
            class_labels: optional (B,) int64 class labels for adaLN cond
        Returns:
            log_probs: (B, L, vocab_size)
        """
        # accelerate fp16 autocast can cast x to float — force long
        x = x.long()
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        sigma = self._t(sigma)
        # Use prefix_mode when cond_tokens are provided
        use_prefix = (cond_tokens is not None)
        logits = self.backbone(x, sigma, cond_tokens=cond_tokens,
                               class_labels=class_labels,
                               prefix_mode=use_prefix)
        return self._subs_parameterization(logits, x)

    # ─────────────── training loss ─────────────────────────

    def _sample_t(self, n: int, device: torch.device) -> Tensor:
        """Sample diffusion time t ∈ [sampling_eps, 1]."""
        eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device).float() / n
            eps_t = (eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * eps_t + self.sampling_eps
        if self.importance_sampling:
            t = self.noise.importance_sampling_transformation(t)
        return t

    def _forward_pass_diffusion_factorized(
        self, x0: Tensor,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """Factorized per-dim diffusion loss (TokenBridge-style).

        Instead of log p_θ(x0 | xt, t) as a single 64K classification,
        computes Σ_d log p_θ(dim_d | dim_{<d}, xt, t) with teacher forcing.

        Returns: (B, L) per-token loss (non-negative), same as standard.
        """
        t = self._sample_t(x0.shape[0], x0.device)

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)

        # Run backbone UP TO the output layer (get hidden states)
        xt_long = xt.long()
        if unet_conditioning.ndim > 1:
            sigma_in = unet_conditioning.squeeze(-1)
        else:
            sigma_in = unet_conditioning

        use_prefix = (cond_tokens is not None)
        sigma_in = self._t(sigma_in)
        # We need the hidden states before the output layer
        # Run backbone manually: embed → blocks → (skip output_layer)
        backbone = self.backbone
        if cond_tokens is not None and cond_tokens.dtype in (torch.long, torch.int):
            cond_tokens = backbone.embed_cond(cond_tokens)
        if backbone.dim_embs is not None:
            x = backbone._factorized_embed(xt_long)
        else:
            x = backbone.token_emb(xt_long)
        x = backbone.pos_emb(x)
        if backbone.pos_emb_type in ("2d", "sudoku"):
            pos = (backbone.row_emb(backbone.row_idx) +
                   backbone.col_emb(backbone.col_idx))
            if backbone.pos_emb_type == "sudoku":
                pos = pos + backbone.box_emb(backbone.box_idx)
            x = x + pos[None, :, :]
        elif backbone.pos_emb_type == "multires":
            pos = (backbone.mr_level_emb(backbone.mr_level_idx) +
                   backbone.mr_row_emb(backbone.mr_row_idx) +
                   backbone.mr_col_emb(backbone.mr_col_idx))
            x = x + pos[None, :, :]
        c = F.silu(backbone.sigma_map(sigma_in))
        if hasattr(backbone, 'label_emb') and backbone.label_emb is not None:
            if class_labels is not None:
                c = c + F.silu(backbone.label_emb(class_labels))

        n_prefix = 0
        if use_prefix and cond_tokens is not None:
            n_prefix = cond_tokens.shape[1]
            x = torch.cat([cond_tokens, x], dim=1)
            cond_tokens = None

        for blk in backbone.blocks:
            x = blk(x, c, cond_tokens=cond_tokens)

        if n_prefix > 0:
            x = x[:, n_prefix:, :]

        # x is now (B, L, hidden_size) — hidden states before output layer

        # Get ground-truth per-dim codes for teacher forcing
        target_dims = self._ar_head.flat_index_to_dims(x0)  # (B, L, D)

        # Run factorized AR head with teacher forcing
        dim_logits = self._ar_head.forward_factorized(x, c, target_dims=target_dims)

        # Compute per-dim cross-entropy, sum across dims
        # Only count loss for MASKED positions (same as standard MDLM)
        masked = (xt == self.mask_index)  # (B, L)

        total_log_p = torch.zeros_like(x0, dtype=torch.float32)  # (B, L)
        for d in range(self._ar_head.n_dims):
            log_probs_d = F.log_softmax(dim_logits[d], dim=-1)  # (B, L, level_d)
            target_d = target_dims[:, :, d]  # (B, L)
            log_p_d = torch.gather(log_probs_d, dim=-1,
                                   index=target_d.unsqueeze(-1)).squeeze(-1)
            total_log_p = total_log_p + log_p_d  # (B, L)

        # For unmasked positions, log_p should be 0 (delta)
        total_log_p = total_log_p * masked.float()

        # Apply ELBO weighting (same as standard)
        if self.change_of_variables or self.importance_sampling:
            return total_log_p * torch.log1p(-torch.exp(-self.noise.sigma_min))
        return -total_log_p * (dsigma / torch.expm1(sigma))[:, None]

    def _forward_pass_diffusion(self, x0: Tensor,
                                 cond_tokens: Optional[Tensor] = None,
                                 class_labels: Optional[Tensor] = None) -> Tensor:
        """Compute per-token diffusion loss for a batch of clean tokens.

        Exactly matches MDLM's continuous-time subs loss:
            L(x0) = - (dsigma / (exp(sigma) - 1)) * log p_θ(x0 | xt, t)

        Args:
            x0:           (B, L) int64 clean tokens
            cond_tokens:  optional conditioning for cross-attn
            class_labels: optional (B,) int64 class labels for adaLN cond
        Returns:
            loss: (B, L) per-token loss (non-negative)
        """
        t = self._sample_t(x0.shape[0], x0.device)

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        model_output = self.forward(xt, unet_conditioning,
                                    cond_tokens=cond_tokens,
                                    class_labels=class_labels)  # log p_θ

        # Gather log p_θ(x0_i | xt, t) for each position
        log_p_theta = torch.gather(
            model_output, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)  # (B, L)

        if self.change_of_variables or self.importance_sampling:
            # change-of-variables ELBO:  L = log_p * log(1 - exp(-sigma_min))
            return log_p_theta * torch.log1p(
                -torch.exp(-self.noise.sigma_min)
            )

        # Standard continuous-time ELBO:
        #   L = -log_p * dsigma / (exp(sigma) - 1)
        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def compute_loss(self, x0: Tensor, attention_mask: Optional[Tensor] = None,
                     cond_tokens: Optional[Tensor] = None,
                     class_labels: Optional[Tensor] = None) -> LossOutput:
        """Full loss computation (used in training step).

        Args:
            x0:             (B, L) int64 clean tokens
            attention_mask: (B, L) float mask (1 = valid, 0 = padding)
            cond_tokens:    optional conditioning for cross-attn
        Returns:
            LossOutput with scalar loss, per-token nlls, and mask.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(x0, dtype=torch.float32)

        if self.factorized_head:
            loss_per_token = self._forward_pass_diffusion_factorized(
                x0, cond_tokens=cond_tokens,
                class_labels=class_labels)
        else:
            loss_per_token = self._forward_pass_diffusion(
                x0, cond_tokens=cond_tokens,
                class_labels=class_labels)  # (B, L)
        nlls = loss_per_token * attention_mask
        count = attention_mask.sum()
        token_nll = nlls.sum() / count

        return LossOutput(loss=token_nll, nlls=nlls, token_mask=attention_mask)

    # ─────────────── continuous mode (diffusion head) ──────

    def q_xt_continuous(self, x: Tensor, move_chance: Tensor) -> tuple[Tensor, Tensor]:
        """Apply absorbing noise to continuous tokens by masking positions.

        Args:
            x:           (B, L, D) float continuous tokens
            move_chance: (B, 1) float in [0, 1]
        Returns:
            mask: (B, L) bool — True = masked (absorbed)
        """
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < move_chance
        return mask

    def compute_loss_continuous(
        self, x0: Tensor,
        attention_mask: Optional[Tensor] = None,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
    ) -> LossOutput:
        """Loss for continuous mode with diffusion head.

        Args:
            x0:             (B, L, D) float clean continuous tokens
            attention_mask: (B, L) float mask (1 = valid, 0 = padding)
            cond_tokens:    optional prefix conditioning
            class_labels:   optional class labels
        Returns:
            LossOutput with scalar loss.
        """
        assert self.diff_head is not None, \
            "diff_head required for continuous mode"
        B, L, D = x0.shape
        device = x0.device

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device, dtype=torch.float32)

        # Sample time t and compute mask probability
        t = self._sample_t(B, device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])  # (B, 1)

        # Mask some positions
        mask = self.q_xt_continuous(x0, move_chance)  # (B, L) bool

        # Run backbone → hidden states (B, L, hidden_size)
        if sigma.ndim > 1:
            sigma_in = sigma.squeeze(-1)
        else:
            sigma_in = sigma
        sigma_in = self._t(sigma_in)

        use_prefix = (cond_tokens is not None)
        hidden = self.backbone(
            indices=None, sigma=sigma_in,
            cond_tokens=cond_tokens, class_labels=class_labels,
            prefix_mode=use_prefix,
            cont_tokens=x0, mask=mask,
        )  # (B, L, hidden_size)

        # Extract masked positions for diffusion head
        # mask: (B, L) bool, hidden: (B, L, H), x0: (B, L, D)
        masked_hidden = hidden[mask]  # (N_masked, H)
        masked_target = x0[mask]      # (N_masked, D)

        if masked_hidden.shape[0] == 0:
            # Edge case: no masked tokens — return zero loss
            return LossOutput(
                loss=torch.tensor(0.0, device=device, requires_grad=True),
                nlls=torch.zeros(B, L, device=device),
                token_mask=attention_mask,
            )

        # Optionally multiply the diffusion head batch (for variance reduction)
        if self.diffusion_batch_mul > 1:
            masked_hidden = masked_hidden.repeat(self.diffusion_batch_mul, 1)
            masked_target = masked_target.repeat(self.diffusion_batch_mul, 1)

        # Diffusion head loss on masked positions
        loss = self.diff_head(target=masked_target, z=masked_hidden)

        return LossOutput(
            loss=loss,
            nlls=torch.zeros(B, L, device=device),
            token_mask=attention_mask,
        )

    def _cheap_confidence(self, hidden: Tensor) -> Tensor:
        """Estimate confidence with a single forward pass at t=1 (pure noise).

        Evaluates the diffusion head once on Gaussian noise at t=1 and uses
        the predicted log_variance as a confidence signal.  Lower variance
        means the model is more confident about this position.

        Args:
            hidden: (N, H) backbone hidden states for masked positions
        Returns:
            confidence: (N,) higher = more confident
        """
        D = self.diff_head.in_channels
        device = hidden.device
        N = hidden.shape[0]

        x = torch.randn(N, D, device=device)
        t = torch.ones(N, device=device)  # t=1, pure noise

        out = self.diff_head.net(x, t, hidden)  # (N, 2*D)
        log_var = out[:, D:]

        # Lower sigma → higher confidence
        return -torch.exp(0.5 * log_var).mean(dim=-1)

    @torch.no_grad()
    def sample_continuous(
        self,
        batch_size: int,
        seq_len: int,
        feat_dim: int,
        num_steps: int = 128,
        device: torch.device = torch.device("cpu"),
        sampler: str = "confidence",
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        temperature: float = 1.0,
        cfg: float = 1.0,
        return_history: bool = False,
        tokens_per_step: int = 0,
        known_mask: Optional[Tensor] = None,
        known_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate continuous tokens via MaskGIT-style iterative unmasking.

        Inpainting-style conditioning (when known_mask/known_tokens given):
          - known positions are initialized to clean token values and marked
            unmasked from step 0.
          - MDLM carry-over parameterization keeps them fixed throughout.
          - Sampler only touches masked (unknown) positions.

        Each step:
        1. Backbone forward (1×) → hidden states for all positions
        2. Cheap confidence (1 MLP forward, no ODE) → rank masked positions
        3. Select top-k positions to unmask (cosine schedule)
        4. Full ODE only for selected positions → get actual token values
        5. Place tokens, repeat

        Cost per step: 1 backbone fwd + N_masked cheap MLP + k ODE (k << N_masked)
        Total ODE calls across all steps = seq_len (each token denoised exactly once)

        Args:
            batch_size, seq_len, feat_dim: output shape
            num_steps:  number of unmasking steps
            temperature: diffusion head sampling temperature
            cfg: classifier-free guidance scale
        Returns:
            x: (batch_size, seq_len, feat_dim) sampled continuous tokens
        """
        import math as _math
        assert self.diff_head is not None, \
            "diff_head required for continuous sampling"

        x = torch.zeros(batch_size, seq_len, feat_dim, device=device)
        is_masked = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Inpainting init: fix known positions with clean tokens
        if known_mask is not None:
            assert known_tokens is not None, \
                "known_tokens required when known_mask is given"
            # (B, L) bool, (B, L, D) float
            x = torch.where(known_mask.unsqueeze(-1), known_tokens, x)
            is_masked = is_masked & (~known_mask)

        history = None
        mask_history = None
        if return_history:
            history = [x.clone().cpu()]
            mask_history = [is_masked.clone().cpu()]

        # DDPM-style: random unmask based on noise schedule probability
        if sampler in ("ddpm", "ddpm_cache"):
            dt = 1.0 / num_steps
            for step in range(num_steps):
                if not is_masked.any():
                    break
                t_cur = torch.full((batch_size,), 1.0 - step * dt, device=device)
                t_nxt = (t_cur - dt).clamp(min=1e-5)
                sigma_t = self.noise(t_cur)[0]
                sigma_s = self.noise(t_nxt)[0]
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                if sigma_s.ndim > 1:
                    sigma_s = sigma_s.squeeze(-1)
                move_t = (1 - torch.exp(-sigma_t)).clamp(min=1e-8)
                move_s = (1 - torch.exp(-sigma_s))
                unmask_prob = (1 - move_s / move_t).clamp(0, 1)  # (B,)

                use_prefix = (cond_tokens is not None)
                hidden = self.backbone(
                    indices=None, sigma=self._t(sigma_t),
                    cond_tokens=cond_tokens, class_labels=class_labels,
                    prefix_mode=use_prefix,
                    cont_tokens=x, mask=is_masked,
                )  # (B, L, H)

                rand = torch.rand(batch_size, seq_len, device=device)
                do_unmask = (rand < unmask_prob[:, None]) & is_masked
                if do_unmask.any():
                    sel_hidden = hidden[do_unmask]
                    sel_tokens = self.diff_head.sample(
                        sel_hidden, temperature=temperature, cfg=cfg)
                    x[do_unmask] = sel_tokens
                    is_masked[do_unmask] = False

                if return_history:
                    history.append(x.clone().cpu())
                    mask_history.append(is_masked.clone().cpu())

            # Final pass: unmask leftovers
            if is_masked.any():
                n_masked_b = is_masked.float().sum(dim=1)
                masked_ratio = (n_masked_b / seq_len).clamp(1e-5, 1)
                sigma_t = self.noise(masked_ratio.view(-1, 1))[0]
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                use_prefix = (cond_tokens is not None)
                hidden = self.backbone(
                    indices=None, sigma=self._t(sigma_t),
                    cond_tokens=cond_tokens, class_labels=class_labels,
                    prefix_mode=use_prefix,
                    cont_tokens=x, mask=is_masked,
                )
                masked_hidden = hidden[is_masked]
                sampled_tokens = self.diff_head.sample(
                    masked_hidden, temperature=temperature, cfg=cfg)
                x[is_masked] = sampled_tokens
                is_masked.fill_(False)
                if return_history:
                    history.append(x.clone().cpu())
                    mask_history.append(is_masked.clone().cpu())

            if return_history:
                return x, mask_history
            return x

        for step in range(num_steps):
            n_masked = is_masked.float().sum(dim=1)  # (B,)
            if n_masked.max().item() == 0:
                break

            # Compute t from current masked ratio
            masked_ratio = (n_masked / seq_len).clamp(1e-5, 1)
            sigma_t = self.noise(masked_ratio.view(-1, 1))[0]
            if sigma_t.ndim > 1:
                sigma_t = sigma_t.squeeze(-1)

            # 1) Backbone forward — 1 pass
            use_prefix = (cond_tokens is not None)
            hidden = self.backbone(
                indices=None, sigma=self._t(sigma_t),
                cond_tokens=cond_tokens, class_labels=class_labels,
                prefix_mode=use_prefix,
                cont_tokens=x, mask=is_masked,
            )  # (B, L, H)

            masked_indices = is_masked.nonzero(as_tuple=False)  # (N_masked, 2)
            if masked_indices.shape[0] == 0:
                break

            masked_hidden = hidden[is_masked]  # (N_masked, H)

            # 2) Cheap confidence — 1 MLP forward, no ODE loop
            confidence = self._cheap_confidence(masked_hidden)  # (N_masked,)

            # 3) Schedule: how many to unmask this step (cumulative target)
            if tokens_per_step > 0:
                # Linear: unmask exactly tokens_per_step per iteration
                target_unmasked = min(seq_len, (step + 1) * tokens_per_step)
            else:
                # Cosine
                ratio = (step + 1) / num_steps
                unmask_frac = _math.cos(_math.pi / 2 * (1 - ratio))
                target_unmasked = int(unmask_frac * seq_len)

            # 4) Select top-k per batch, gather, single ODE call
            all_selected_hidden = []
            all_batch_ids = []
            all_positions = []

            for b in range(batch_size):
                b_mask = is_masked[b]
                n_cur = int(b_mask.sum().item())
                if n_cur == 0:
                    continue
                n_already_unmasked = seq_len - n_cur
                n_to_unmask = max(1, target_unmasked - n_already_unmasked)
                n_to_unmask = min(n_to_unmask, n_cur)

                b_masked_pos = b_mask.nonzero(as_tuple=False).squeeze(-1)
                b_selector = (masked_indices[:, 0] == b)
                b_conf = confidence[b_selector]
                b_hidden = masked_hidden[b_selector]  # (n_cur, H)

                _, topk = b_conf.topk(n_to_unmask)
                all_selected_hidden.append(b_hidden[topk])
                all_batch_ids.extend([b] * n_to_unmask)
                all_positions.append(b_masked_pos[topk])

            if all_selected_hidden:
                cat_hidden = torch.cat(all_selected_hidden, dim=0)  # (K_total, H)
                cat_tokens = self.diff_head.sample(
                    cat_hidden, temperature=temperature, cfg=cfg
                )  # (K_total, D)

                # Scatter back (vectorized)
                bid = torch.tensor(all_batch_ids, device=device, dtype=torch.long)
                pos = torch.cat(all_positions, dim=0)  # (K_total,)
                x[bid, pos] = cat_tokens
                is_masked[bid, pos] = False

            if return_history:
                history.append(x.clone().cpu())
                mask_history.append(is_masked.clone().cpu())

        # Final pass: unmask any remaining
        if is_masked.any():
            n_masked = is_masked.float().sum(dim=1)
            masked_ratio = (n_masked / seq_len).clamp(1e-5, 1)
            sigma_t = self.noise(masked_ratio.view(-1, 1))[0]
            if sigma_t.ndim > 1:
                sigma_t = sigma_t.squeeze(-1)

            use_prefix = (cond_tokens is not None)
            hidden = self.backbone(
                indices=None, sigma=self._t(sigma_t),
                cond_tokens=cond_tokens, class_labels=class_labels,
                prefix_mode=use_prefix,
                cont_tokens=x, mask=is_masked,
            )
            masked_hidden = hidden[is_masked]
            sampled_tokens = self.diff_head.sample(
                masked_hidden, temperature=temperature, cfg=cfg)

            x[is_masked] = sampled_tokens
            is_masked.fill_(False)

            if return_history:
                history.append(x.clone().cpu())
                mask_history.append(is_masked.clone().cpu())

        if return_history:
            return x, mask_history
        return x

    # ─────────────── sampling (reverse process) ────────────

    @torch.no_grad()
    def _ddpm_caching_update(
        self, x: Tensor, t: Tensor, dt: float,
        p_x0: Optional[Tensor] = None,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
    ):
        """One DDPM step with caching (MDLM's ddpm_cache sampler)."""
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)

        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        if self.factorized_head:
            # Factorized head: AR head gives a one-hot (delta) distribution,
            # so we sample x0 directly and apply the ddpm transition
            # analytically without allocating a (B, L, vocab_size) tensor.
            #
            # For one-hot p_x0 at token k, the ddpm transition simplifies to:
            #   P(unmask to k) = (move_t - move_s) / move_t = 1 - move_s/move_t
            #   P(stay masked) = move_s / move_t
            h, c_vec = self._run_backbone_hidden(
                x, sigma_t, cond_tokens=cond_tokens,
                class_labels=class_labels)
            sampled_x0, _ = self._ar_head.sample_with_confidence(h, c_vec)

            # For masked positions: unmask with prob 1 - move_s/move_t
            ratio = move_chance_s / move_chance_t.clamp(min=1e-8)  # (B, 1, 1)
            unmask_prob = (1 - ratio).squeeze(-1)  # (B, 1)
            do_unmask = torch.rand_like(x.float()) < unmask_prob
            is_masked = (x == self.mask_index)
            x_new = torch.where(is_masked & do_unmask, sampled_x0, x)
            # No caching for factorized (each step re-runs AR head)
            return None, x_new

        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t, cond_tokens=cond_tokens,
                                class_labels=class_labels).exp()

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        x_new = torch.where((x != self.mask_index), x, _x)
        return p_x0, x_new

    @torch.no_grad()
    def _ddpm_update(self, x: Tensor, t: Tensor, dt: float,
                     cond_tokens: Optional[Tensor] = None,
                     class_labels: Optional[Tensor] = None) -> Tensor:
        """One DDPM step (no caching)."""
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)

        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        if self.factorized_head:
            # Same analytic transition as _ddpm_caching_update
            h, c_vec = self._run_backbone_hidden(
                x, sigma_t, cond_tokens=cond_tokens,
                class_labels=class_labels)
            sampled_x0, _ = self._ar_head.sample_with_confidence(h, c_vec)
            ratio = move_chance_s / move_chance_t.clamp(min=1e-8)
            unmask_prob = (1 - ratio).squeeze(-1)
            do_unmask = torch.rand_like(x.float()) < unmask_prob
            is_masked = (x == self.mask_index)
            return torch.where(is_masked & do_unmask, sampled_x0, x)

        log_p_x0 = self.forward(x, sigma_t, cond_tokens=cond_tokens,
                                class_labels=class_labels)
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        x_new = torch.where((x != self.mask_index), x, _x)
        return x_new

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 128,
        device: torch.device = torch.device("cpu"),
        sampler: str = "ddpm_cache",
        noise_removal: bool = True,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        return_history: bool = False,
        tokens_per_step: int = 0,
        known_mask: Optional[Tensor] = None,
        known_tokens: Optional[Tensor] = None,
        **kwargs,  # absorb unused args (e.g. guidance_scale from old configs)
    ) -> Tensor:
        """Generate samples via iterative denoising.

        Inpainting conditioning: pass known_mask (B,L bool) and known_tokens
        (B,L int64). Known positions are initialized unmasked; MDLM carry-over
        keeps them fixed. Only masked positions are denoised.

        Args:
            batch_size:  number of samples
            seq_len:     sequence length
            num_steps:   number of denoising steps
            device:      target device
            sampler:     ``"ddpm"``, ``"ddpm_cache"``, or ``"confidence"``
            noise_removal: if True, apply a final argmax denoising step
            cond_tokens: optional prefix conditioning
            class_labels: optional (B,) int64 class labels for adaLN cond
            tokens_per_step: for confidence sampler – unmask exactly this
                many tokens per step (0 = cosine schedule, 1 = one at a time)
        Returns:
            x: (batch_size, seq_len) int64 sampled tokens
        """
        if sampler == "confidence":
            return self._sample_confidence(
                batch_size, seq_len, num_steps, device, cond_tokens,
                class_labels=class_labels,
                return_history=return_history,
                tokens_per_step=tokens_per_step,
                known_mask=known_mask, known_tokens=known_tokens)

        eps = 1e-5
        x = torch.full((batch_size, seq_len), self.mask_index,
                        dtype=torch.long, device=device)
        if known_mask is not None:
            assert known_tokens is not None
            x = torch.where(known_mask, known_tokens.long(), x)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        history = [x.clone().cpu()] if return_history else None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(batch_size, 1, device=device)
            if sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache, cond_tokens=cond_tokens,
                    class_labels=class_labels)
                if not torch.equal(x_next, x):
                    p_x0_cache = None
                x = x_next
            else:
                x = self._ddpm_update(x, t, dt, cond_tokens=cond_tokens,
                                      class_labels=class_labels)

            if return_history:
                history.append(x.clone().cpu())

        if noise_removal:
            t = timesteps[-1] * torch.ones(batch_size, 1, device=device)
            sigma_t = self.noise(t)[0]
            if self.factorized_head:
                # Use AR head directly — avoids allocating (B, L, 64K) logits
                h, c_vec = self._run_backbone_hidden(
                    x, sigma_t, cond_tokens=cond_tokens,
                    class_labels=class_labels)
                final_pred, _ = self._ar_head.sample_with_confidence(h, c_vec)
                still_masked = (x == self.mask_index)
                x = torch.where(still_masked, final_pred, x)
            else:
                x = self.forward(x, sigma_t, cond_tokens=cond_tokens,
                                 class_labels=class_labels).argmax(dim=-1)
            if return_history:
                history.append(x.clone().cpu())

        if return_history:
            return x, history
        return x

    @torch.no_grad()
    def _sample_confidence(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        device: torch.device,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        return_history: bool = False,
        tokens_per_step: int = 0,
        known_mask: Optional[Tensor] = None,
        known_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """MaskGIT-style confidence-based sampling.

        At each step:
        1. Predict p(x0|xt) for all masked positions.
        2. Take argmax as the predicted token, max-prob as confidence.
        3. Unmask the top-k most confident positions (k decreases per step).
        4. Repeat until all positions are unmasked.

        Args:
            tokens_per_step: if > 0, unmask exactly this many tokens per
                step (linear schedule). If 0, use cosine schedule.
                Use tokens_per_step=1 with num_steps=seq_len for
                one-token-at-a-time unmasking.
        """
        use_linear = (tokens_per_step > 0)

        x = torch.full((batch_size, seq_len), self.mask_index,
                        dtype=torch.long, device=device)
        if known_mask is not None:
            assert known_tokens is not None
            x = torch.where(known_mask, known_tokens.long(), x)

        history = [x.clone().cpu()] if return_history else None

        for step in range(num_steps):
            is_masked = (x == self.mask_index)  # (B, L)
            n_masked = is_masked.float().sum(dim=1)  # (B,)

            if n_masked.max().item() == 0:
                break  # all unmasked

            if not use_linear:
                ratio = (step + 1) / num_steps
                unmask_frac = math.cos(math.pi / 2 * (1 - ratio))
                target_masked = ((1 - unmask_frac) * seq_len)

            masked_ratio = (n_masked / seq_len).clamp(1e-5, 1)
            t = masked_ratio.view(-1, 1)
            sigma_t = self.noise(t)[0]

            if self.factorized_head:
                # Factorized: get tokens + proper joint confidence
                h, c_vec = self._run_backbone_hidden(
                    x, sigma_t, cond_tokens=cond_tokens,
                    class_labels=class_labels)
                sampled_tokens, log_confidence = \
                    self._ar_head.sample_with_confidence(h, c_vec)
                confidence = log_confidence.exp()  # (B, L)
            else:
                log_p_x0 = self.forward(x, sigma_t, cond_tokens=cond_tokens,
                                        class_labels=class_labels)
                log_p_x0[:, :, self.mask_index] = NEG_INF

                probs = torch.softmax(log_p_x0, dim=-1)  # (B, L, V)

                # Sample tokens stochastically from p(x0|xt)
                B_cur, L_cur, V = probs.shape
                sampled_tokens = torch.multinomial(
                    probs.view(B_cur * L_cur, V), 1
                ).view(B_cur, L_cur)  # (B, L)

                # Confidence = probability of the sampled token
                confidence = torch.gather(
                    probs, dim=-1, index=sampled_tokens[:, :, None]
                ).squeeze(-1)  # (B, L)

            confidence[~is_masked] = float('inf')

            for b in range(batch_size):
                n_cur_masked = int(n_masked[b].item())
                if n_cur_masked == 0:
                    continue

                if use_linear:
                    n_to_unmask = min(tokens_per_step, n_cur_masked)
                else:
                    n_to_unmask = max(1, n_cur_masked - int(target_masked))
                    n_to_unmask = min(n_to_unmask, n_cur_masked)

                # Top-k most confident among masked positions
                masked_conf = confidence[b].clone()
                masked_conf[~is_masked[b]] = -1.0
                _, topk_idx = masked_conf.topk(n_to_unmask)
                x[b, topk_idx] = sampled_tokens[b, topk_idx]

            if return_history:
                history.append(x.clone().cpu())

        # final pass: unmask any remaining
        still_masked = (x == self.mask_index)
        if still_masked.any():
            t = torch.full((batch_size, 1), 1e-5, device=device)
            sigma_t = self.noise(t)[0]
            if self.factorized_head:
                h, c_vec = self._run_backbone_hidden(
                    x, sigma_t, cond_tokens=cond_tokens,
                    class_labels=class_labels)
                final_pred, _ = self._ar_head.sample_with_confidence(h, c_vec)
            else:
                log_p = self.forward(x, sigma_t, cond_tokens=cond_tokens,
                                     class_labels=class_labels)
                log_p[:, :, self.mask_index] = NEG_INF
                final_pred = log_p.argmax(dim=-1)
            x[still_masked] = final_pred[still_masked]
            if return_history:
                history.append(x.clone().cpu())

        if return_history:
            return x, history
        return x

    # ─────────────── inpainting (partial conditioning) ─────

    @torch.no_grad()
    def sample_inpaint(
        self,
        x_gt: Tensor,
        known_mask: Tensor,
        num_steps: int = 128,
        sampler: str = "ddpm_cache",
        noise_removal: bool = True,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        return_history: bool = False,
        return_confidence_history: bool = False,
        tokens_per_step: int = 0,
        return_step_logs: bool = False,
    ):
        """Sample with some positions fixed to ground-truth values.

        At every denoising step, the known positions are forced back to
        their GT values.  Only the unknown (masked) positions are filled
        by the model.  No training change needed.

        Args:
            x_gt:       (B, L) int64  full ground-truth token sequence
            known_mask: (B, L) bool   True = this position is given as hint
            num_steps:  number of denoising steps
            sampler:    "ddpm", "ddpm_cache", or "confidence"
            noise_removal: final argmax step
            cond_tokens: optional conditioning for cross-attn
            return_history: if True, also return list of (B,L) snapshots
            tokens_per_step: for confidence sampler only - if > 0, use linear
                             schedule unmasking exactly this many per step
        Returns:
            x: (B, L) int64 completed sequence
            history (optional): list of (B, L) tensors per step
        """
        device = x_gt.device
        batch_size, seq_len = x_gt.shape

        if sampler == "confidence":
            return self._inpaint_confidence(
                x_gt, known_mask, num_steps, cond_tokens,
                class_labels=class_labels,
                return_history=return_history,
                return_confidence_history=return_confidence_history,
                tokens_per_step=tokens_per_step,
                return_step_logs=return_step_logs)

        eps = 1e-5
        # start: known positions = GT, unknown = [MASK]
        x = torch.where(known_mask, x_gt, self.mask_index)
        n_unknown = (~known_mask).float().sum(dim=1).clamp(min=1)  # (B,) total unknown positions
        
        # For dt calculation: use fixed step size based on initial masked ratio
        # Initial masked ratio = 1.0 (all unknown are masked at start)
        # We want to go from ratio=1.0 to ratio=0.0 in num_steps
        dt = 1.0 / num_steps
        
        p_x0_cache = None
        history = [x.clone().cpu()] if return_history else None
        confidence_history = [] if return_confidence_history else None
        pred_history = [] if return_confidence_history else None
        step_logs = [] if return_step_logs else None

        def compute_confidence_and_pred(x_t: Tensor, t_scalar: Tensor):
            sigma_t = self.noise(t_scalar)[0]
            log_p = self.forward(x_t, sigma_t, cond_tokens=cond_tokens,
                                 class_labels=class_labels)
            log_p[:, :, self.mask_index] = NEG_INF
            probs = torch.softmax(log_p, dim=-1)  # SOFTMAX to get true probabilities
            return probs.max(dim=-1).values, probs.argmax(dim=-1)

        if return_confidence_history:
            # Initial: compute based on current masked ratio (should be 1.0 for all unknowns)
            is_masked_init = (x == self.mask_index)
            n_masked_init = is_masked_init.float().sum(dim=1)
            masked_ratio_init = (n_masked_init / n_unknown).clamp(eps, 1)
            t0 = masked_ratio_init.view(-1, 1)
            conf0, pred0 = compute_confidence_and_pred(x, t0)
            confidence_history.append(conf0.cpu())
            pred_history.append(pred0.cpu())

        for i in range(num_steps):
            # ⭐ KEY FIX: t = current masked ratio (matches training distribution) ⭐
            is_masked = (x == self.mask_index)
            n_masked = is_masked.float().sum(dim=1)
            masked_ratio = (n_masked / n_unknown).clamp(eps, 1)  # (B,)
            
            if n_masked.max().item() == 0:
                break  # all unmasked
            
            t = masked_ratio.view(-1, 1)  # (B, 1) - t = masked ratio!
            
            if sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache, cond_tokens=cond_tokens,
                    class_labels=class_labels)
                if not torch.equal(x_next, x):
                    p_x0_cache = None
                x = x_next.long()  # ENSURE int64
            else:
                x = self._ddpm_update(x, t, dt, cond_tokens=cond_tokens,
                                      class_labels=class_labels).long()  # ENSURE int64
            
            # force known positions back to GT (must be after casting to int!)
            x = torch.where(known_mask, x_gt, x)
            
            # record step logs if requested
            if return_step_logs:
                is_masked = (x == self.mask_index)
                n_masked = is_masked.sum(dim=1)
                sigma_t, _ = self.noise(t)
                log_p = self.forward(x, sigma_t, cond_tokens=cond_tokens, class_labels=class_labels)
                log_p[:, :, self.mask_index] = NEG_INF
                probs = torch.softmax(log_p, dim=-1)
                pred_tokens = probs.argmax(dim=-1)
                step_logs.append({
                    'step': i,
                    't': t[0, 0].item(),
                    'n_masked': n_masked.cpu().tolist(),  # list per batch
                    'x_t': x.cpu().clone(),
                    'pred': pred_tokens.cpu().clone(),
                })
            
            if return_history:
                history.append(x.clone().cpu())
            if return_confidence_history:
                # t for next step = current masked ratio after this step's update
                is_masked_now = (x == self.mask_index)
                n_masked_now = is_masked_now.float().sum(dim=1)
                masked_ratio_now = (n_masked_now / n_unknown).clamp(eps, 1)
                t_next = masked_ratio_now.view(-1, 1)
                conf_next, pred_next = compute_confidence_and_pred(x, t_next)
                confidence_history.append(conf_next.cpu())
                pred_history.append(pred_next.cpu())

        if noise_removal:
            # Final step: use very small t (near clean)
            is_masked_final = (x == self.mask_index)
            n_masked_final = is_masked_final.float().sum(dim=1)
            masked_ratio_final = (n_masked_final / n_unknown).clamp(eps, 1)
            t = masked_ratio_final.view(-1, 1)
            sigma_t = self.noise(t)[0]
            log_p = self.forward(x, sigma_t, cond_tokens=cond_tokens, class_labels=class_labels)
            log_p[:, :, self.mask_index] = NEG_INF
            final_probs = torch.softmax(log_p, dim=-1)
            pred = final_probs.argmax(dim=-1).long()  # ENSURE int64
            # only fill unknown positions
            x = torch.where(known_mask, x_gt, pred).long()  # ENSURE int64
            if return_history:
                history.append(x.clone().cpu())
            if return_confidence_history:
                confidence_history.append(final_probs.max(dim=-1).values.cpu())
                pred_history.append(pred.cpu())
            if return_step_logs:
                is_masked = (x == self.mask_index)
                n_masked = is_masked.sum(dim=1)
                step_logs.append({
                    'step': num_steps,
                    't': t[0, 0].item(),
                    'n_masked': n_masked.cpu().tolist(),
                    'x_t': x.cpu().clone(),
                    'pred': pred.cpu().clone(),
                })

        # Build return value
        ret = x
        if return_history:
            ret = (ret, history)
        if return_confidence_history:
            if not return_history:
                ret = (ret, None, confidence_history, pred_history)
            else:
                ret = (x, history, confidence_history, pred_history)
        if return_step_logs:
            ret = (ret, step_logs) if isinstance(ret, torch.Tensor) else (*ret, step_logs)
        
        return ret

    @torch.no_grad()
    def _inpaint_confidence(
        self,
        x_gt: Tensor,
        known_mask: Tensor,
        num_steps: int,
        cond_tokens: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        return_history: bool = False,
        return_confidence_history: bool = False,
        tokens_per_step: int = 0,
        return_step_logs: bool = False,
    ):
        """Confidence-based inpainting: only unmask unknown positions.
        
        Args:
            tokens_per_step: if > 0, unmask exactly this many tokens per step
                             (linear schedule). If 0, use cosine schedule.
        """
        device = x_gt.device
        batch_size, seq_len = x_gt.shape
        x = torch.where(known_mask, x_gt, self.mask_index)
        history = [x.clone().cpu()] if return_history else None
        confidence_history = [] if return_confidence_history else None
        pred_history = [] if return_confidence_history else None
        step_logs = [] if return_step_logs else None
        
        use_linear = (tokens_per_step > 0)

        # initial confidence for the first history entry
        if return_confidence_history:
            t0 = torch.full((batch_size, 1), 1.0 - 1e-5, device=device)
            sigma_t0 = self.noise(t0)[0]
            log_p0 = self.forward(x, sigma_t0, cond_tokens=cond_tokens, class_labels=class_labels)
            log_p0[:, :, self.mask_index] = NEG_INF
            probs0 = torch.softmax(log_p0, dim=-1)  # SOFTMAX to get true probabilities
            confidence_history.append(probs0.max(dim=-1).values.cpu())
            pred_history.append(probs0.argmax(dim=-1).cpu())

        for step in range(num_steps):
            is_masked = (x == self.mask_index)
            n_masked = is_masked.float().sum(dim=1)
            if n_masked.max().item() == 0:
                break

            if use_linear:
                # Linear schedule: unmask exactly tokens_per_step per iteration
                n_to_unmask_target = tokens_per_step
            else:
                # Cosine schedule
                ratio = (step + 1) / num_steps
                unmask_frac = math.cos(math.pi / 2 * (1 - ratio))
                n_unknown = (~known_mask).float().sum(dim=1)  # total unknowns
                target_masked = ((1 - unmask_frac) * n_unknown).clamp(min=0)
                n_to_unmask_target = None  # computed per-batch below

            # ⭐ KEY FIX: t should reflect CURRENT masked ratio, not schedule progress ⭐
            # This ensures the model sees the same distribution as training
            n_unknown = (~known_mask).float().sum(dim=1).clamp(min=1)  # (B,)
            n_masked_now = is_masked.float().sum(dim=1)  # (B,)
            masked_ratio = (n_masked_now / n_unknown).clamp(0, 1)  # (B,) in [0,1]
            
            # Map: masked_ratio=1.0→t=1.0 (fully masked), 0.0→t=1e-5 (unmasked)
            t_val = masked_ratio + 1e-5  # (B,)
            t = t_val.view(-1, 1)  # (B, 1)
            sigma_t = self.noise(t)[0]

            # b0 = 0
            # print(
            #     f"[inpaint] step={step:03d} "
            #     f"t[b0]={t[b0, 0].item():.6f} "
            #     f"masked_ratio[b0]={masked_ratio[b0].item():.6f} "
            #     f"n_masked_now[b0]={int(n_masked_now[b0].item())}"
            # )


            log_p_x0 = self.forward(x, sigma_t, cond_tokens=cond_tokens, class_labels=class_labels)

            # # ===== INPAINT DEBUG (ALWAYS ON) =====
            # b0 = 0
            # pred = log_p_x0.argmax(dim=-1)

            # # move_chance (training debug와 동일한 계산)
            # if self.change_of_variables:
            #     f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            #     f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            #     move_chance_dbg = torch.exp(f_0 + t[b0, 0] * (f_T - f_0)).item()
            # else:
            #     sigma_dbg = self.noise(t[b0:b0+1])[0]
            #     move_chance_dbg = (1 - torch.exp(-sigma_dbg)).item()

            # # masked stats
            # is_masked_dbg = (x[b0] == self.mask_index)
            # n_masked_dbg = int(is_masked_dbg.sum().item())
            # if n_masked_dbg > 0:
            #     masked_correct = ((pred[b0] == x_gt[b0]) & is_masked_dbg).sum().item()
            #     masked_acc = masked_correct / n_masked_dbg
            # else:
            #     masked_acc = 1.0

            # # confidence
            # probs_dbg = torch.softmax(log_p_x0[b0], dim=-1)
            # conf_dbg = probs_dbg.max(dim=-1).values
            # if n_masked_dbg > 0:
            #     conf_min = conf_dbg[is_masked_dbg].min().item()
            #     conf_mean = conf_dbg[is_masked_dbg].mean().item()
            #     conf_max = conf_dbg[is_masked_dbg].max().item()
            # else:
            #     conf_min = conf_mean = conf_max = 0.0

            # print(
            #     f"[inpaint-debug] step={step:03d} "
            #     f"t={t[b0,0].item():.4f} "
            #     f"move_chance={move_chance_dbg:.4f} "
            #     f"n_masked={n_masked_dbg}/81 "
            #     f"masked_acc={masked_acc:.3f} "
            #     f"masked_conf=[{conf_min:.3f}, {conf_mean:.3f}, {conf_max:.3f}]"
            # )

            # # grid print (Sudoku)
            # def _print_grid(tag, vec):
            #     print(f"[inpaint-debug] {tag} grid:")
            #     for r in range(9):
            #         s = r * 9
            #         e = s + 9
            #         row = vec[s:e].tolist()
            #         row = [(v + 1) if v != self.mask_index else (self.mask_index + 1) for v in row]
            #         print(" ".join(str(v) for v in row))

            # _print_grid("x0(gt)", x_gt[b0])
            # _print_grid("x(t)",  x[b0])
            # _print_grid("pred",  pred[b0])
            # # ===== END INPAINT DEBUG =====



            log_p_x0[:, :, self.mask_index] = NEG_INF
            probs = torch.softmax(log_p_x0, dim=-1)  # SOFTMAX to get true probabilities
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            confidence[~is_masked] = float('inf')

            for b in range(batch_size):
                n_cur = int(n_masked[b].item())
                if n_cur == 0:
                    continue
                if use_linear:
                    n_to_unmask = min(n_to_unmask_target, n_cur)
                else:
                    n_to_unmask = max(1, n_cur - int(target_masked[b].item()))
                    n_to_unmask = min(n_to_unmask, n_cur)
                masked_conf = confidence[b].clone()
                masked_conf[~is_masked[b]] = -1.0
                _, topk_idx = masked_conf.topk(n_to_unmask)
                x[b, topk_idx] = pred_tokens[b, topk_idx].long()

            # force known
            x = torch.where(known_mask, x_gt, x).long()  # ENSURE int64
            
            # record step logs if requested
            if return_step_logs:
                is_masked_after = (x == self.mask_index)
                n_masked_after = is_masked_after.sum(dim=1)
                step_logs.append({
                    'step': step,
                    't': t_val[0].item() if t_val.dim() > 0 else t_val.item(),  # convert to scalar
                    'n_masked': n_masked_after.cpu().tolist(),
                    'x_t': x.cpu().clone(),
                    'pred': pred_tokens.cpu().clone(),
                })
            
            if return_history:
                history.append(x.clone().cpu())
            if return_confidence_history:
                # Re-forward on updated x to get current predictions for remaining masked cells
                t_next = torch.full((batch_size, 1), max(1e-5, 1.0 - (step + 2) / num_steps), device=device)
                sigma_next = self.noise(t_next)[0]
                log_p_new = self.forward(x, sigma_next, cond_tokens=cond_tokens, class_labels=class_labels)
                log_p_new[:, :, self.mask_index] = NEG_INF
                probs_new = torch.softmax(log_p_new, dim=-1)  # SOFTMAX to get true probabilities
                confidence_history.append(probs_new.max(dim=-1).values.cpu())
                pred_history.append(probs_new.argmax(dim=-1).cpu())

        # final
        still_masked = (x == self.mask_index)
        if still_masked.any():
            t = torch.full((batch_size, 1), 1e-5, device=device)
            sigma_t = self.noise(t)[0]
            log_p = self.forward(x, sigma_t, cond_tokens=cond_tokens, class_labels=class_labels)
            log_p[:, :, self.mask_index] = NEG_INF
            final_probs = torch.softmax(log_p, dim=-1)  # SOFTMAX to get true probabilities
            pred = final_probs.argmax(dim=-1)
            x = torch.where(known_mask, x_gt, pred).long()  # ENSURE int64
            if return_history:
                history.append(x.clone().cpu())
            if return_confidence_history:
                confidence_history.append(final_probs.max(dim=-1).values.cpu())
                pred_history.append(pred.cpu())
            if return_step_logs:
                is_masked_final = (x == self.mask_index)
                n_masked_final = is_masked_final.sum(dim=1)
                step_logs.append({
                    'step': num_steps,
                    't': 1e-5,
                    'n_masked': n_masked_final.cpu().tolist(),
                    'x_t': x.cpu().clone(),
                    'pred': pred.cpu().clone(),
                })

        # Build return value - ENSURE int64
        ret = x.long()
        extras = []
        if return_history:
            extras.append(history)
        if return_confidence_history:
            if not return_history:
                extras.append(None)  # placeholder for history
            extras.append(confidence_history)
            extras.append(pred_history)
        if return_step_logs:
            extras.append(step_logs)

        if extras:
            return (ret, *extras)
        return ret
