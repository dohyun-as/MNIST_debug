"""
Stage 1 trainer: Slot Attention encoder + Baseline1DConditionalDiT.

This file is a thin wrapper around `main_multires.train()`. It does NOT
modify ``main_multires.py``. Instead it monkey-patches two functions on
the imported ``main_multires`` module before delegating control:

    main_multires.build_model       — swaps ``model.encoder`` to
                                      ``SlotAttentionEncoder`` when slot
                                      training is requested
    main_multires.generate_samples  — appends a slot-attention
                                      segmentation visualisation hook

The patches live on the in-memory ``main_multires`` object only. Other
scripts (e.g. the original ``train_clevr_dit_our_continuous.sh``) never
import this file, so they remain bit-for-bit identical in behaviour.

Usage:
    bash script/train_clevr_slot_stage1.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import torch

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import main_multires as mm
from slot_encoder import SlotAttentionEncoder, visualize_slot_segmentation


# ──────────────────────────────────────────────────────────────────
#  Pre-parser for slot-only args (so main_multires.parse_args sees a
#  clean argv without our slot flags)
# ──────────────────────────────────────────────────────────────────

def _split_slot_args(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--slot_iters", type=int, default=3,
                     help="Slot Attention iteration count (default 3).")
    pre.add_argument("--slot_mlp_size", type=int, default=384,
                     help="MLP hidden size inside SlotAttention (default 2*slot_dim).")
    pre.add_argument("--slot_enc_backbone", type=str, default="vit_b16",
                     choices=["resnet18", "resnet18_no_layer4",
                              "resnet18_strided",
                              "vit_b16", "dinov1_b16", "dinov2_base"],
                     help="Backbone for slot encoder. "
                          "'resnet18'/'resnet18_strided' — CNN (need SoftPos). "
                          "'vit_b16' — trainable ViT-B/16 from scratch. "
                          "'dinov1_b16'/'dinov2_base' — frozen pretrained "
                          "ViT (very low memory, strong object emergence).")
    pre.add_argument("--slot_init", type=str, default="learned",
                     choices=["learned", "random"],
                     help="Slot init: 'learned' (SlotDiffusion) or 'random' (Locatello).")
    pre.add_argument("--slot_viz_every", type=int, default=5000,
                     help="Save slot-attention segmentation viz every N steps.")
    pre.add_argument("--slot_viz_n_samples", type=int, default=8,
                     help="How many val images to visualise per slot-viz step.")
    return pre.parse_known_args(argv)


# ──────────────────────────────────────────────────────────────────
#  Monkey patches
# ──────────────────────────────────────────────────────────────────

_orig_build_model = mm.build_model


def _build_model_with_slot(args):
    """Wrap mm.build_model: build the standard Baseline1DConditionalDiT
    first, then swap its encoder to SlotAttentionEncoder."""

    # Pass-through if not a slot run (defensive — main_slot_stage1.py
    # always sets use_slot_encoder=True so this branch is unused).
    if not getattr(args, "use_slot_encoder", False):
        return _orig_build_model(args)

    if args.backbone != "baseline_1d":
        raise ValueError(
            f"--backbone must be 'baseline_1d' for slot stage-1 training "
            f"(got '{args.backbone}'). Slot encoder requires the 1D-slot "
            f"DiT structure provided by Baseline1DConditionalDiT.")

    # Build baseline_1d skeleton with default SemanticistViTEncoder.
    model = _orig_build_model(args)

    # Replace encoder. The default SemanticistViTEncoder is dropped
    # immediately — its parameters never see an optimizer step.
    model.encoder = SlotAttentionEncoder(
        image_size=args.image_size,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_iterations=args.slot_iters,
        mlp_hidden_size=args.slot_mlp_size,
        enc_backbone=args.slot_enc_backbone,
        slot_init=args.slot_init,
    )

    # Disable nested-sampler progressive slot dropping (semanticist-only
    # trick; not meaningful for permutation-invariant slot attention).
    if hasattr(model, "enable_nest_after_steps"):
        model.enable_nest_after_steps = -1
    if hasattr(model, "enable_nest"):
        model.enable_nest = False

    return model


_orig_generate_samples = mm.generate_samples


def _generate_samples_with_slot_viz(model, val_dataset, scheduler, args,
                                    accelerator, step, vae=None,
                                    ema_model=None, train_dataset=None):
    """Run the original sample-grid logic, then save slot-attention
    segmentation overlays on a fresh batch of val images."""

    out = _orig_generate_samples(
        model, val_dataset, scheduler, args, accelerator, step,
        vae=vae, ema_model=ema_model, train_dataset=train_dataset)

    if not getattr(args, "use_slot_encoder", False):
        return out
    if not accelerator.is_main_process:
        return out
    # Only fire on slot_viz_every cadence (cheap; encoder forward only).
    every = getattr(args, "slot_viz_every", 5000)
    if every <= 0 or step % every != 0:
        return out

    try:
        eval_model = (ema_model if ema_model is not None
                      else accelerator.unwrap_model(model))
        encoder = eval_model.encoder
        device = accelerator.device

        # Pick a deterministic small subset of val images
        n = min(getattr(args, "slot_viz_n_samples", 8), len(val_dataset))
        rng = torch.Generator().manual_seed(args.seed + 12345)
        idx = torch.randperm(len(val_dataset), generator=rng)[:n].tolist()
        images = torch.stack([val_dataset[i][0] for i in idx]).to(device)

        save_dir = os.path.join(args.output_dir, "slot_viz")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"step_{step:07d}.png")

        # Encoder may be wrapped in DDP; pull underlying module
        enc = encoder.module if hasattr(encoder, "module") else encoder
        visualize_slot_segmentation(enc, images, save_path)
        accelerator.print(f"[slot-viz] saved → {save_path}")
    except Exception as exc:
        accelerator.print(f"[slot-viz] skipped: {exc}")

    return out


# Apply patches (only affects the in-memory mm object for this process).
mm.build_model = _build_model_with_slot
mm.generate_samples = _generate_samples_with_slot_viz


# ──────────────────────────────────────────────────────────────────
#  Entrypoint
# ──────────────────────────────────────────────────────────────────

def main():
    # Pull our slot-specific args out of argv first.
    slot_ns, remaining = _split_slot_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    # Now main_multires.parse_args sees only its own flags.
    args = mm.parse_args()

    # Inject slot args into the namespace so build_model & checkpoint
    # serialization both pick them up.
    for k, v in vars(slot_ns).items():
        setattr(args, k, v)
    args.use_slot_encoder = True

    if args.backbone != "baseline_1d":
        raise ValueError(
            f"slot stage 1 requires --backbone baseline_1d (got '{args.backbone}')")

    mm.train(args)


if __name__ == "__main__":
    main()
