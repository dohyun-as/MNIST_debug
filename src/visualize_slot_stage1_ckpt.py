"""Standalone visualization for saved slot-stage1 checkpoints.

Iterates over EVERY ``<run_dir>/checkpoints/step_NNNNNNN/`` and writes
unified per-sample diagnostic images
``[original | encoder slot seg | DiT cross-attn seg | reconstruction]``
to ``<run_dir>/slot_unified_eval/step_NNNNNNN.t{NNN}.png`` —
filename includes the cross-attn time ``t`` so results from different t
don't overwrite each other.

Multiple t values can be visualised in one run via ``--dit_attn_t 0.25 0.5 0.75``.

Single-GPU only (no accelerate). Auto-detects EMA weights and prefers them
when present. Re-uses the slot-encoder swap from ``main_slot_stage1`` so
the architecture matches the training run exactly. The model architecture
is built once; weights are reloaded per step (much faster than rebuilding).

Usage:
    # All steps, default t=0.5, 16 samples
    python src/visualize_slot_stage1_ckpt.py \
        --run_dir runs/clevr/slot_stage1/256_slot16_d64_resnet18s_crossattn

    # Specific step
    python src/visualize_slot_stage1_ckpt.py --run_dir <run_dir> --step 50000

    # Multiple t values for comparison (one file per t per step)
    python src/visualize_slot_stage1_ckpt.py \
        --run_dir <run_dir> --dit_attn_t 0.25 0.5 0.75 0.9

    # More samples
    python src/visualize_slot_stage1_ckpt.py \
        --run_dir <run_dir> --n_samples 32

    # Last N steps only
    python src/visualize_slot_stage1_ckpt.py --run_dir <run_dir> --last_n 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from torchvision import datasets, transforms

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import main_multires as mm
import main_slot_stage1 as mss  # applies monkey-patches on import
from slot_encoder import (visualize_slot_unified,
                          visualize_slot_segmentation,
                          visualize_dit_cross_attention)


def _list_steps(ckpt_root: str) -> list[int]:
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"No checkpoints/ under {ckpt_root}")
    steps = []
    for d in os.listdir(ckpt_root):
        if d.startswith("step_"):
            try:
                steps.append(int(d.split("_")[1]))
            except ValueError:
                pass
    if not steps:
        raise FileNotFoundError(f"No step_* dirs in {ckpt_root}")
    return sorted(steps)


def _build_val_dataset(val_dir: str, image_size: int):
    """Plain ImageFolder-or-flat val loader. Matches the [-1, 1]
    normalisation that training uses, but skips augmentation."""
    tfm = transforms.Compose([
        transforms.Resize(image_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    direct = any(
        f.lower().endswith(('.png', '.jpg', '.jpeg'))
        for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))
    )
    if direct:
        files = sorted([
            os.path.join(val_dir, f) for f in os.listdir(val_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        class _FlatImageDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(files)

            def __getitem__(self, idx):
                from PIL import Image
                img = Image.open(files[idx]).convert("RGB")
                return tfm(img), 0

        return _FlatImageDataset()
    return datasets.ImageFolder(val_dir, transform=tfm)


def _t_tag(t_val: float) -> str:
    return f"t{int(round(float(t_val) * 100)):03d}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to a slot-stage1 run dir (contains "
                        "args.json + checkpoints/).")
    p.add_argument("--step", type=int, default=None,
                   help="Single step to process. If absent, ALL saved "
                        "steps are processed (use --last_n to cap).")
    p.add_argument("--last_n", type=int, default=None,
                   help="Only process the latest N saved steps.")
    p.add_argument("--val_dir", type=str, default=None,
                   help="Override val image dir (default: args.val_dir).")
    p.add_argument("--n_samples", type=int, default=16,
                   help="Number of val images to visualise (default 16).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for deterministic val sample subset.")
    p.add_argument("--num_sampling_steps", type=int, default=50,
                   help="Euler steps for reconstruction.")
    p.add_argument("--dit_attn_t", type=float, nargs="+", default=[0.5],
                   help="Flow time(s) at which to capture DiT cross-attn. "
                        "0=noise, 1=clean. Pass multiple for one file per t. "
                        "Default: 0.5")
    p.add_argument("--average_last_n_blocks", type=int, default=4,
                   help="Avg cross-attn over this many trailing DiT blocks.")
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Use EMA weights if available (default: yes).")
    p.add_argument("--no_ema", dest="use_ema", action="store_false",
                   help="Force non-EMA weights.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda / cpu / cuda:N.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip steps whose output PNGs already exist.")
    args_cli = p.parse_args()

    run_dir = os.path.abspath(args_cli.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    # ── Load training args ──
    args_path = os.path.join(run_dir, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"args.json missing: {args_path}")
    with open(args_path) as f:
        train_args_dict = json.load(f)
    train_args = SimpleNamespace(**train_args_dict)
    # Defensive defaults for new flags missing in older runs
    for k, v in [("dit_attn_mode", "self_concat"),
                 ("use_slot_encoder", True),
                 ("slot_iters", 3),
                 ("slot_mlp_size", None),
                 ("slot_enc_backbone", "resnet18"),
                 ("slot_init", "learned")]:
        if not hasattr(train_args, k):
            setattr(train_args, k, v)
    if train_args.slot_mlp_size is None:
        train_args.slot_mlp_size = train_args.slot_dim * 2
    train_args.use_slot_encoder = True

    # ── Resolve steps ──
    ckpt_root = os.path.join(run_dir, "checkpoints")
    all_steps = _list_steps(ckpt_root)
    if args_cli.step is not None:
        if args_cli.step not in all_steps:
            raise ValueError(
                f"step {args_cli.step} not in checkpoints; available {all_steps}")
        steps_to_run = [args_cli.step]
    else:
        steps_to_run = all_steps
        if args_cli.last_n is not None and args_cli.last_n > 0:
            steps_to_run = steps_to_run[-args_cli.last_n:]

    print(f"[viz] {len(steps_to_run)} step(s) × {len(args_cli.dit_attn_t)} "
          f"t value(s)  =  {len(steps_to_run) * len(args_cli.dit_attn_t)} "
          f"output file(s)")
    print(f"[viz] steps: {steps_to_run}")
    print(f"[viz] t:     {args_cli.dit_attn_t}")

    # ── Build model ONCE ──
    device = torch.device(args_cli.device
                          if torch.cuda.is_available() else "cpu")
    model = mm.build_model(train_args)
    model._nest_enabled = False
    model = model.to(device).eval()

    # ── Build val dataset ──
    val_dir = args_cli.val_dir or getattr(train_args, "val_dir", None)
    if val_dir is None:
        raise ValueError(
            "val_dir not in args.json and not passed via --val_dir")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"val_dir does not exist: {val_dir}")
    val_ds = _build_val_dataset(val_dir, train_args.image_size)
    print(f"[viz] val dataset: {len(val_ds)} images at {val_dir}")

    # ── Deterministic sample subset (re-used across all steps & t) ──
    n = min(args_cli.n_samples, len(val_ds))
    rng = torch.Generator().manual_seed(args_cli.seed + 12345)
    idx = torch.randperm(len(val_ds), generator=rng)[:n].tolist()
    images = torch.stack([val_ds[i][0] for i in idx]).to(device)
    print(f"[viz] using {n} val samples")

    unified_dir = os.path.join(run_dir, "slot_unified_eval")
    slotviz_dir = os.path.join(run_dir, "slot_viz_eval")
    ditattn_dir = os.path.join(run_dir, "dit_attn_eval")
    os.makedirs(unified_dir, exist_ok=True)
    os.makedirs(slotviz_dir, exist_ok=True)
    os.makedirs(ditattn_dir, exist_ok=True)

    # ── Loop ──
    flow_t_eps = getattr(train_args, "flow_t_eps", 0.05)
    for si, step in enumerate(steps_to_run):
        ckpt_path = os.path.join(ckpt_root, f"step_{step:07d}", "checkpoint.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[viz] step {step}: checkpoint.pt missing, skip")
            continue

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if args_cli.use_ema and "ema" in ckpt:
            tag = "ema"
            sd = ckpt["ema"]
        else:
            tag = "model"
            sd = ckpt["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[viz] step {step} ({tag}): "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        del ckpt, sd
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── (A) Encoder slot seg + per-slot encoder masks ──
        sv_path = os.path.join(slotviz_dir, f"step_{step:07d}.png")
        if args_cli.skip_existing and os.path.isfile(sv_path):
            print(f"  [{si+1}/{len(steps_to_run)}] step={step} slot_viz "
                  f"→ exists, skipped")
        else:
            enc = (model.encoder.module
                   if hasattr(model.encoder, "module") else model.encoder)
            visualize_slot_segmentation(enc, images, sv_path)
            print(f"  [{si+1}/{len(steps_to_run)}] step={step} slot_viz "
                  f"→ {sv_path}")

        # ── (B) DiT cross-attn argmax + per-slot heatmaps (multi-t in one go) ──
        da_path = os.path.join(ditattn_dir, f"step_{step:07d}.png")
        try:
            visualize_dit_cross_attention(
                model, images, da_path,
                t_value=list(args_cli.dit_attn_t),
                average_last_n_blocks=args_cli.average_last_n_blocks,
            )
            print(f"  [{si+1}/{len(steps_to_run)}] step={step} dit_attn "
                  f"→ {da_path}.t*.png")
        except Exception as exc:
            print(f"  [{si+1}/{len(steps_to_run)}] step={step} dit_attn "
                  f"FAILED: {exc}")

        # ── (C) Unified per-sample grid (one file per t) ──
        for t_val in args_cli.dit_attn_t:
            out_path = os.path.join(
                unified_dir, f"step_{step:07d}.{_t_tag(t_val)}.png")
            if args_cli.skip_existing and os.path.isfile(out_path):
                print(f"  [{si+1}/{len(steps_to_run)}] step={step} "
                      f"unified t={t_val} → exists, skipped")
                continue
            print(f"  [{si+1}/{len(steps_to_run)}] step={step} "
                  f"unified t={t_val} → {out_path}")
            visualize_slot_unified(
                model, images, out_path,
                num_sampling_steps=args_cli.num_sampling_steps,
                dit_attn_t=float(t_val),
                average_last_n_blocks=args_cli.average_last_n_blocks,
                flow_t_eps=flow_t_eps,
            )

    print(f"[viz] done")
    print(f"  encoder slot viz  → {slotviz_dir}")
    print(f"  DiT cross-attn    → {ditattn_dir}")
    print(f"  unified           → {unified_dir}")


if __name__ == "__main__":
    main()
