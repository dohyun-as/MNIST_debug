"""
diffloss.py — Diffusion Head for continuous token prediction (MAR-style)
=========================================================================

A small MLP-based diffusion model that generates continuous token vectors
conditioned on backbone hidden states.  Used as the "output head" of a
discrete (mask-based) diffusion backbone operating on continuous tokens.

Reference: MAR (Li et al., 2024), Semanticist (Liu et al., 2025)

This implementation uses **flow matching** (simpler than DDPM):
  - Training: sample t ~ U[0,1], x_t = (1-t)*x_0 + t*noise,
              loss = MSE(predicted_velocity, noise - x_0) + variance loss
  - Sampling: Euler ODE from noise (t=1) to data (t=0)

The network predicts [velocity, log_variance] jointly.  During sampling,
the accumulated log_variance serves as a confidence metric: lower
sigma_theta = model is more certain about its prediction.
(Inspired by SRM's sequential_adaptive_sampler.)
"""

import math
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → vector embedding."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10_000):
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

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim))


class ResBlock(nn.Module):
    """Residual block with AdaLN modulation."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels),
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x, t_cond):
        shift, scale, gate = self.adaLN(t_cond).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class FinalLayer(nn.Module):
    """Final output layer with AdaLN."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels),
        )
        self.linear = nn.Linear(model_channels, out_channels)

    def forward(self, x, t_cond):
        shift, scale = self.adaLN(t_cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """Small MLP network for the diffusion head."""

    def __init__(self, in_channels, model_channels, out_channels,
                 z_channels, num_res_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulations → identity at init
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN[-1].bias, 0)

        # Zero-out output layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Args:
            x: (N, in_channels) noised token vectors
            t: (N,) float timesteps in [0, 1]
            c: (N, z_channels) conditioning from backbone
        Returns:
            (N, out_channels) — [velocity, log_variance] concatenated
        """
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        t_combined = t_emb + c_emb

        for block in self.res_blocks:
            x = block(x, t_combined)
        return self.final_layer(x, t_combined)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        """Classifier-free guidance forward."""
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        guided = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([guided, guided], dim=0)


# ────────────────────────────────────────────────────────────
#  DiffLoss — Flow-Matching Diffusion Head with Variance
# ────────────────────────────────────────────────────────────

class DiffLoss(nn.Module):
    """Diffusion head using flow matching with learned variance.

    The network outputs [velocity, log_variance] jointly:
      - velocity:     used for the ODE trajectory
      - log_variance: learned per-sample uncertainty (sigma_theta)

    Training:
        sample t ~ U[eps, 1-eps]
        x_t = (1-t)*x_0 + t*noise
        velocity_target = noise - x_0
        [velocity_pred, log_var] = net(x_t, t, c)
        loss = MSE(velocity_pred, velocity_target) / exp(log_var) + log_var

    Sampling:
        Euler ODE as before, but also accumulate sigma_theta across
        timesteps.  Lower accumulated sigma = higher confidence.

    Args:
        target_channels: dim of continuous tokens (e.g. 16)
        z_channels: dim of backbone hidden states (conditioning)
        depth: number of ResBlocks in the MLP
        width: hidden dim of the MLP
        num_sampling_steps: Euler steps during inference
    """

    def __init__(self, target_channels: int, z_channels: int,
                 depth: int = 6, width: int = 1024,
                 num_sampling_steps: int = 100,
                 cond_drop_prob: float = 0.1):
        super().__init__()
        self.in_channels = target_channels
        self.z_channels = z_channels
        self.num_sampling_steps = num_sampling_steps
        self.cond_drop_prob = cond_drop_prob

        # Learned unconditional embedding for classifier-free guidance
        # (MAR / semanticist style: z is replaced with null_cond at prob
        # cond_drop_prob during training, enabling CFG at sampling time).
        self.null_cond = nn.Parameter(torch.randn(z_channels) * 0.02)

        # Output: velocity (target_channels) + log_variance (target_channels)
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,
            z_channels=z_channels,
            num_res_blocks=depth,
        )

    def forward(self, target, z, mask=None):
        """Compute flow-matching loss with learned variance.

        Loss = MSE(v_pred, v_target) / exp(log_var) + log_var
        This is the heteroscedastic Gaussian NLL: the network learns
        where it's uncertain and adjusts loss weighting accordingly.

        Args:
            target: (N, target_channels) clean continuous tokens
            z:      (N, z_channels) conditioning from backbone
            mask:   (N,) optional bool mask (True = compute loss)
        Returns:
            scalar loss
        """
        device = target.device
        B = target.shape[0]
        D = self.in_channels

        # Sample timestep t ~ U[eps, 1-eps]
        eps = 0.001
        t = torch.rand(B, device=device) * (1 - 2 * eps) + eps

        # Classifier-free guidance: replace z with null_cond for a random
        # fraction of the batch so the net learns p(x) as well as p(x|z).
        if self.training and self.cond_drop_prob > 0.0:
            drop = (torch.rand(B, device=device) < self.cond_drop_prob)
            if drop.any():
                z = torch.where(drop[:, None], self.null_cond[None, :].to(z.dtype), z)

        # Noise
        noise = torch.randn_like(target)

        # Interpolate: x_t = (1-t)*x_0 + t*noise
        t_expand = t[:, None]  # (B, 1)
        x_t = (1 - t_expand) * target + t_expand * noise

        # Predict velocity + log_variance
        out = self.net(x_t, t, z)  # (B, 2*D)
        pred_velocity = out[:, :D]
        # Clamp log_var to SRM's range to prevent exp(-log_var) blow-up
        # that otherwise causes late-stage NaN (see SRM DiagonalGaussian).
        log_var = out[:, D:].clamp(-30.0, 20.0)  # (B, D)

        # Target velocity: noise - x_0
        target_velocity = noise - target

        # Split velocity and variance gradients (SRM-style):
        #   - velocity is supervised by plain MSE (no variance weighting)
        #   - log_var is supervised by heteroscedastic NLL on detached MSE,
        #     so the variance head cannot warp velocity learning.
        sq_err = (pred_velocity - target_velocity).pow(2)               # (B, D)
        velocity_term = sq_err
        variance_term = sq_err.detach() * torch.exp(-log_var) + log_var  # (B, D)
        loss_per_dim = velocity_term + variance_term                     # (B, D)
        loss = loss_per_dim.mean(dim=-1)  # (B,)

        if mask is not None:
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(self, z, temperature=1.0, cfg=1.0, z_uncond=None):
        """Generate continuous tokens via Euler ODE (no confidence).

        Args:
            z: (N, z_channels) conditioning from backbone
            temperature: noise scaling
            cfg: classifier-free guidance scale (1.0 = no guidance)
            z_uncond: (N, z_channels) optional precomputed unconditional
                hidden states (MAR/semanticist-style backbone CFG). When
                provided, overrides the internal null_cond embedding.
        Returns:
            (N, target_channels) sampled continuous tokens
        """
        tokens, _ = self.sample_with_confidence(
            z, temperature=temperature, cfg=cfg, z_uncond=z_uncond)
        return tokens

    @torch.no_grad()
    def sample_with_confidence(self, z, temperature=1.0, cfg=1.0, z_uncond=None):
        """Generate continuous tokens + confidence via Euler ODE.

        Confidence = negative mean sigma_theta accumulated over the last
        few ODE steps (when t is small, i.e., near data).
        Lower sigma_theta → model is more confident about this token.

        Args:
            z: (N, z_channels) conditioning from backbone
            temperature: noise scaling
            cfg: classifier-free guidance scale (1.0 = no guidance)
        Returns:
            tokens:     (N, target_channels) sampled continuous tokens
            confidence: (N,) float — higher = more confident
        """
        device = z.device
        D = self.in_channels
        N = z.shape[0]
        use_cfg = (cfg != 1.0)

        # Start from noise (t=1)
        x = torch.randn(N, D, device=device) * temperature

        # For CFG, build a stacked conditioning tensor [cond; uncond]
        # so we can run the net once on 2N inputs per ODE step.
        if use_cfg:
            if z_uncond is not None:
                # Backbone-CFG: caller supplies uncond hidden states from a
                # separate backbone forward (MAR/semanticist style).
                assert z_uncond.shape == z.shape, \
                    f"z_uncond shape {z_uncond.shape} != z {z.shape}"
                null_z = z_uncond.to(z.dtype)
            else:
                # Head-CFG: use this module's learned null_cond embedding.
                null_z = self.null_cond[None, :].to(z.dtype).expand(N, -1)
            z_cat = torch.cat([z, null_z], dim=0)  # (2N, H)

        dt = 1.0 / self.num_sampling_steps
        sigma_accum = torch.zeros(N, D, device=device)
        accum_start = int(self.num_sampling_steps * 0.8)

        for i in range(self.num_sampling_steps):
            t_val = 1.0 - i * dt

            if use_cfg:
                x_cat = torch.cat([x, x], dim=0)                     # (2N, D)
                t_cat = torch.full((2 * N,), t_val, device=device)
                out = self.net(x_cat, t_cat, z_cat)                  # (2N, 2D)
                vel_cond, vel_uncond = out[:N, :D], out[N:, :D]
                velocity = vel_uncond + cfg * (vel_cond - vel_uncond)
                log_var = out[:N, D:].clamp(-30.0, 20.0)             # cond branch
            else:
                t = torch.full((N,), t_val, device=device)
                out = self.net(x, t, z)                              # (N, 2D)
                velocity = out[:, :D]
                log_var = out[:, D:].clamp(-30.0, 20.0)

            # Euler step on the single trajectory
            x = x - dt * velocity

            if i >= accum_start:
                sigma_accum = sigma_accum + torch.exp(0.5 * log_var)

        mean_sigma = sigma_accum.mean(dim=-1)  # (N,)
        confidence = -mean_sigma

        return x, confidence
