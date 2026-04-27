"""
Stage 2 trainer: masked discrete diffusion on Slot Attention tokens.

Wraps ``train_discrete_diffusion_v2.main()`` and overrides exactly two
functions on the imported module:

    load_pretrained_model
        After the original loader builds the baseline_1d skeleton, swap
        ``model.encoder`` to ``SlotAttentionEncoder`` and load the slot
        encoder weights from the stage-1 checkpoint.

    decode_continuous_tokens_to_images
        The original only handles multi-res ``forward_from_level_features``.
        Add a baseline_1d branch that calls the existing
        ``_forward_from_slots`` helper (already used by the discrete
        ``decode_tokens_to_images`` path).

Everything else — data, training loop, eval, sampling, CLEVR cond eval —
is reused unchanged from train_discrete_diffusion_v2.

Usage (same flags as the original stage-2 script):
    bash script/train_clevr_slot_stage2.sh
"""

from __future__ import annotations

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_discrete_diffusion_v2 as t2v2
from slot_encoder import SlotAttentionEncoder


# ──────────────────────────────────────────────────────────────────
#  Patch: load_pretrained_model
# ──────────────────────────────────────────────────────────────────

_orig_load = t2v2.load_pretrained_model


def _load_with_slot(pretrained_output_dir: str, device: str = "cpu"):
    """Run the original loader, then swap encoder if cfg marks it slot."""
    model, encoder, discretizer, level_sizes, vocab_size, cfg = _orig_load(
        pretrained_output_dir, device=device)

    if not cfg.get("use_slot_encoder", False):
        return model, encoder, discretizer, level_sizes, vocab_size, cfg

    # Build SlotAttentionEncoder with the same hyperparameters used at
    # stage 1 (saved into args.json by main_multires).
    new_enc = SlotAttentionEncoder(
        image_size=cfg["image_size"],
        num_slots=cfg.get("num_slots", 16),
        slot_dim=cfg.get("slot_dim", 192),
        num_iterations=cfg.get("slot_iters", 3),
        mlp_hidden_size=cfg.get("slot_mlp_size", 384),
        enc_backbone=cfg.get("slot_enc_backbone", "resnet18"),
        slot_init=cfg.get("slot_init", "learned"),
    )

    # Locate latest checkpoint and pull encoder.* weights.
    ckpt_dir = os.path.join(pretrained_output_dir, "checkpoints")
    steps = []
    if os.path.isdir(ckpt_dir):
        for d in os.listdir(ckpt_dir):
            if d.startswith("step_"):
                try:
                    steps.append(int(d.split("_")[1]))
                except ValueError:
                    pass
    if not steps:
        raise RuntimeError(
            f"[slot-load] no step_* checkpoints in {ckpt_dir}")
    latest = max(steps)
    ckpt_path = os.path.join(ckpt_dir, f"step_{latest:07d}", "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("ema", ckpt))
    enc_state = {k[len("encoder."):]: v for k, v in state.items()
                 if k.startswith("encoder.")}
    if not enc_state:
        raise RuntimeError(
            f"[slot-load] no 'encoder.*' keys in {ckpt_path}")

    missing, unexpected = new_enc.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"[slot-load] WARN missing keys ({len(missing)}): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
              flush=True)
    if unexpected:
        print(f"[slot-load] WARN unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
              flush=True)

    new_enc.eval()
    for p in new_enc.parameters():
        p.requires_grad_(False)
    new_enc.to(device)

    model.encoder = new_enc
    encoder = new_enc

    print(f"[slot-load] swapped to SlotAttentionEncoder "
          f"(K={new_enc.num_slots}, D={new_enc.slot_dim}, "
          f"backbone={cfg.get('slot_enc_backbone', 'resnet18')}) "
          f"from step {latest}", flush=True)

    return model, encoder, discretizer, level_sizes, vocab_size, cfg


# ──────────────────────────────────────────────────────────────────
#  Patch: decode_continuous_tokens_to_images (baseline_1d branch)
# ──────────────────────────────────────────────────────────────────

_orig_decode_cont = t2v2.decode_continuous_tokens_to_images


@torch.no_grad()
def _decode_cont_with_slot_branch(
    cont_tokens, level_sizes, pretrained_model, device,
    num_steps=50, guidance_scale=1.0, batch_size=16,
    noise_scale=1.0, t_eps=0.05,
):
    """Decode continuous slot tokens to images via flow-matching ODE.

    For multi-res models (the existing CLEVR/ImageNet setup) this defers
    to the unmodified original function. For baseline_1d models (slot
    encoder) it runs an Euler ODE through ``_forward_from_slots`` — the
    same helper the discrete-token decoder already uses.
    """
    is_multires = hasattr(pretrained_model, "forward_from_level_features")
    is_baseline_1d = (hasattr(pretrained_model, "num_slots")
                      and not is_multires)

    if not is_baseline_1d:
        return _orig_decode_cont(
            cont_tokens, level_sizes, pretrained_model, device,
            num_steps=num_steps, guidance_scale=guidance_scale,
            batch_size=batch_size, noise_scale=noise_scale, t_eps=t_eps)

    pretrained_model.eval()
    pretrained_model.to(device)
    N = cont_tokens.shape[0]
    all_images = []

    image_size = pretrained_model.image_size
    in_channels = (pretrained_model._in_channels
                   if hasattr(pretrained_model, "_in_channels") else 3)
    vae_factor = getattr(pretrained_model, "vae_downsample_factor", 1)
    latent_size = image_size // vae_factor

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        slot_feats = cont_tokens[start:end].to(device)   # (B, K, slot_dim)
        B_cur = slot_feats.shape[0]

        z = noise_scale * torch.randn(
            B_cur, in_channels, latent_size, latent_size, device=device)

        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur
            t_batch = t_cur.expand(B_cur)
            t_expand = t_cur.view(1, 1, 1, 1)

            if guidance_scale != 1.0:
                x_cond = t2v2._forward_from_slots(
                    pretrained_model, z, t_batch, slot_feats,
                    return_uncond=False)
                x_uncond = t2v2._forward_from_slots(
                    pretrained_model, z, t_batch, slot_feats,
                    return_uncond=True)
                v_cond = (x_cond - z) / (1.0 - t_expand).clamp_min(t_eps)
                v_uncond = (x_uncond - z) / (1.0 - t_expand).clamp_min(t_eps)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                x_pred = t2v2._forward_from_slots(
                    pretrained_model, z, t_batch, slot_feats,
                    return_uncond=False)
                v = (x_pred - z) / (1.0 - t_expand).clamp_min(t_eps)

            z = z + dt * v

        # VAE decode if needed
        if vae_factor > 1 and hasattr(pretrained_model, "vae"):
            z = pretrained_model.vae.decode(
                z / pretrained_model.vae_scaling).sample
        all_images.append(z.cpu().float())

    return torch.cat(all_images, dim=0).clamp(-1, 1)


# Apply patches
t2v2.load_pretrained_model = _load_with_slot
t2v2.decode_continuous_tokens_to_images = _decode_cont_with_slot_branch


# ──────────────────────────────────────────────────────────────────
#  Entrypoint — defer entirely to the original main()
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t2v2.main()
