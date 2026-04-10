#!/usr/bin/env python
"""
Evaluate trained multi-res diffusion checkpoints on CLEVR using detector + classifier.

Multi-GPU via accelerate. Eval set is deterministic: for each split (easy/medium/hard),
N samples are chosen with a fixed seed — same set regardless of GPU count or when run.

Usage:
  # Single GPU
  python script/eval_clevr_ckpts.py \
    --run_dir runs/clevr_256_dit_vit_flow_fsq_mask075_CA_bugfix

  # Multi-GPU
  accelerate launch --num_processes 4 script/eval_clevr_ckpts.py \
    --run_dir runs/clevr_256_dit_vit_flow_fsq_mask075_CA_bugfix

  # Specific steps, more samples
  accelerate launch --num_processes 4 script/eval_clevr_ckpts.py \
    --run_dir runs/clevr_256_dit_vit_flow_fsq_mask075_CA_bugfix \
    --steps 50000 100000 --num_samples_per_split 50
"""

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.distributed as dist
from torchvision import datasets, transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path, device="cuda", use_ema=True):
    """Rebuild model from checkpoint args and load weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]

    defaults = dict(
        encoder_type="cnn",
        vit_patch_size=4, vit_depth=4, vit_num_heads=4,
        vit_mlp_ratio=4.0, vit_use_cnn_stem=True, vit_no_cnn_stem=False,
        vit_cnn_stem_reduction=4,
        use_flow_matching=False,
        flow_sampling_method="euler", flow_noise_scale=1.0, flow_t_eps=0.05,
        use_fsq=False, fsq_levels=None, fsq_drop_quant_p=0.0,
        fsq_corrupt_tokens_p=0.0,
        use_vq=False, vq_codebook_size=512, vq_beta=0.25,
        cond_use_latent=False, mae_mask_ratio=0.0,
        level_sizes=None, backbone="unet",
    )
    for k, v in defaults.items():
        saved_args.setdefault(k, v)

    args = types.SimpleNamespace(**saved_args)

    from main_multires import build_model
    model = build_model(args)

    if use_ema and "ema" in ckpt and ckpt["ema"]:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])

    model.eval().to(device)
    return model, args


# ── Deterministic eval set ───────────────────────────────────────────────────

def build_eval_indices(val_dataset, val_images_dir, num_per_split=30, seed=42):
    """Select a fixed set of indices: num_per_split samples per split (easy/medium/hard).

    Returns a list of (global_dataset_index, split_name) tuples.
    The selection is deterministic given (seed, num_per_split) — independent of GPU count.
    """
    # Group dataset indices by split (subdirectory name)
    split_to_indices = {}
    for idx, (path, _) in enumerate(val_dataset.samples):
        rel = os.path.relpath(path, val_images_dir)
        split_name = rel.split(os.sep)[0]  # e.g. "easy", "medium", "hard"
        split_to_indices.setdefault(split_name, []).append(idx)

    selected = []
    for split_name in sorted(split_to_indices.keys()):
        pool = sorted(split_to_indices[split_name])  # sort for determinism
        rng = torch.Generator().manual_seed(seed + hash(split_name) % (2**31))
        n = min(num_per_split, len(pool))
        perm = torch.randperm(len(pool), generator=rng)[:n]
        for p in perm.tolist():
            selected.append((pool[p], split_name))

    return selected


# ── Sampling ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_flow_ode(model, cond_images, num_steps=50, guidance_scale=1.5,
                    in_channels=3, method="euler", noise_scale=1.0, t_eps=0.05):
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    z = noise_scale * torch.randn(B, in_channels, latent_size, latent_size,
                                  device=device, dtype=dtype)
    timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    def _velocity(z_cur, t_scalar):
        t_batch = t_scalar.expand(B)
        t_expand = t_scalar.view(1, 1, 1, 1)
        if guidance_scale != 1.0:
            x_cond = model(z_cur, t_batch, cond_image=cond_images)
            x_uncond = model(z_cur, t_batch, cond_image=cond_images, return_uncond=True)
            v_cond = (x_cond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            v_uncond = (x_uncond - z_cur) / (1.0 - t_expand).clamp_min(t_eps)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            x_pred = model(z_cur, t_batch, cond_image=cond_images)
            return (x_pred - z_cur) / (1.0 - t_expand).clamp_min(t_eps)

    for i in range(num_steps):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        dt = t_next - t_cur
        if method == "heun" and i < num_steps - 1:
            v1 = _velocity(z, t_cur)
            v2 = _velocity(z + dt * v1, t_next)
            z = z + dt * 0.5 * (v1 + v2)
        else:
            z = z + dt * _velocity(z, t_cur)

    return z.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model, cond_images, num_steps=50, guidance_scale=1.5,
                in_channels=3, args=None):
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        clip_sample=True, clip_sample_range=1.0,
    )
    device = cond_images.device
    dtype = cond_images.dtype
    B = cond_images.shape[0]
    latent_size = model.latent_size

    scheduler.set_timesteps(num_steps, device=device)
    latents = torch.randn(B, in_channels, latent_size, latent_size,
                          device=device, dtype=dtype)
    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        if guidance_scale != 1.0:
            n_cond = model(latents, t_batch, cond_image=cond_images)
            n_uncond = model(latents, t_batch, cond_image=cond_images, return_uncond=True)
            noise_pred = n_uncond + guidance_scale * (n_cond - n_uncond)
        else:
            noise_pred = model(latents, t_batch, cond_image=cond_images)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    return latents.clamp(-1, 1)


# ── Batched eval ─────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_batch(model, train_args, val_dataset, batch_indices, val_images_dir,
               scenes_dir, detector, classifier, clevr_cfg,
               extract_peaks, match_detections,
               device, guidance_scale, num_steps,
               det_transform, crop_transform):
    """Reconstruct a batch of images and evaluate detection + classification.

    batch_indices: list of (dataset_idx, split_name) tuples.
    Returns list of per-sample stats dicts (one per index).
    """
    attr_names = ["color", "shape", "size", "material"]

    # ── 1. Load GT info for all samples in batch ──
    valid = []  # (position_in_batch, scene_dict, gt_centers, gt_attrs)
    cond_tensors = []
    for pos, (idx, _) in enumerate(batch_indices):
        img_path, _ = val_dataset.samples[idx]
        rel = os.path.relpath(img_path, val_images_dir)
        scene_path = os.path.join(scenes_dir, os.path.splitext(rel)[0] + ".json")
        if not os.path.isfile(scene_path):
            continue

        with open(scene_path) as f:
            scene = json.load(f)

        gt_centers, gt_attrs = [], []
        for obj in scene["objects"]:
            cx = np.clip(obj["pixel_coords"][0], 0, clevr_cfg.IMG_SIZE - 1)
            cy = np.clip(obj["pixel_coords"][1], 0, clevr_cfg.IMG_SIZE - 1)
            gt_centers.append([cx, cy])
            gt_attrs.append([
                clevr_cfg.COLORS.index(obj["color"]),
                clevr_cfg.SHAPES.index(obj["shape"]),
                clevr_cfg.SIZES.index(obj["size"]),
                clevr_cfg.MATERIALS.index(obj["material"]),
            ])
        gt_centers_np = np.array(gt_centers) if gt_centers else np.zeros((0, 2))
        valid.append((pos, gt_centers_np, gt_attrs))
        cond_tensors.append(val_dataset[idx][0])

    if not valid:
        return [None] * len(batch_indices)

    # ── 2. Batch reconstruction ──
    cond_batch = torch.stack(cond_tensors).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if getattr(train_args, "use_flow_matching", False):
            recon_batch = sample_flow_ode(
                model, cond_batch, num_steps=num_steps,
                guidance_scale=guidance_scale,
                in_channels=train_args.in_channels,
                method=getattr(train_args, "flow_sampling_method", "euler"),
                noise_scale=getattr(train_args, "flow_noise_scale", 1.0),
                t_eps=getattr(train_args, "flow_t_eps", 0.05),
            )
        else:
            recon_batch = sample_ddim(
                model, cond_batch, num_steps=num_steps,
                guidance_scale=guidance_scale,
                in_channels=train_args.in_channels, args=train_args,
            )

    recon_batch = (recon_batch * 0.5 + 0.5).clamp(0, 1).cpu()

    # ── 3. Batch detection ──
    det_inputs = torch.stack([
        transforms.Normalize([0.5]*3, [0.5]*3)(recon_batch[i])
        for i in range(len(valid))
    ]).to(device)
    pred_heatmaps = detector(det_inputs).cpu().numpy()[:, 0]  # (B, H, W)

    # ── 4. Per-sample: peak extraction, matching, classification ──
    results_list = [None] * len(batch_indices)
    to_pil = transforms.ToPILImage()

    for batch_pos, (_, gt_centers_np, gt_attrs) in enumerate(valid):
        n_gt = len(gt_centers_np)
        peaks = extract_peaks(pred_heatmaps[batch_pos], threshold=0.3)
        recon_pil = to_pil(recon_batch[batch_pos])
        w, h = recon_pil.size

        sample_stats = {}
        for t in clevr_cfg.DETECTOR_DIST_THRESH:
            s = {"correct": {a: 0 for a in attr_names}, "correct_all": 0,
                 "total_matched": 0, "total_pred": len(peaks), "total_gt": n_gt}

            mp, mg, _ = match_detections(peaks, gt_centers_np, t)
            s["total_matched"] = len(mp)

            if len(mp) > 0:
                crops = []
                half = clevr_cfg.CROP_SIZE // 2
                for pi in mp:
                    px, py = int(peaks[pi][0]), int(peaks[pi][1])
                    x1, y1 = max(px - half, 0), max(py - half, 0)
                    x2, y2 = min(px + half, w), min(py + half, h)
                    crops.append(crop_transform(recon_pil.crop((x1, y1, x2, y2))))

                crop_batch = torch.stack(crops).to(device)
                preds = classifier(crop_batch)

                for k, (pi, gi) in enumerate(zip(mp, mg)):
                    gt = gt_attrs[gi]
                    all_ok = True
                    for ai, a in enumerate(attr_names):
                        if preds[a][k].argmax().item() == gt[ai]:
                            s["correct"][a] += 1
                        else:
                            all_ok = False
                    if all_ok:
                        s["correct_all"] += 1

            sample_stats[t] = s

        # Map back to original position
        orig_pos = valid[batch_pos][0]
        results_list[orig_pos] = sample_stats

    return results_list


# ── Merge stats (per-split aware) ────────────────────────────────────────────

ATTR_NAMES = ["color", "shape", "size", "material"]


def _empty_stats(clevr_cfg):
    """Create an empty stats dict for one split."""
    return {t: {"correct": {a: 0 for a in ATTR_NAMES}, "correct_all": 0,
                "total_matched": 0, "total_pred": 0, "total_gt": 0}
            for t in clevr_cfg.DETECTOR_DIST_THRESH}


def merge_stats(local_stats_list, clevr_cfg):
    """Merge a list of (sample_stats, split_name) tuples into per-split + overall stats.

    Returns dict: {"overall": {...}, "easy": {...}, "medium": {...}, "hard": {...}}
    Each value is {threshold: {correct, ...}}.
    """
    splits = set()
    for _, split_name in local_stats_list:
        if split_name is not None:
            splits.add(split_name)

    merged = {"overall": _empty_stats(clevr_cfg)}
    for sp in splits:
        merged[sp] = _empty_stats(clevr_cfg)

    for sample_stats, split_name in local_stats_list:
        if sample_stats is None:
            continue
        targets = ["overall"]
        if split_name is not None:
            targets.append(split_name)
        for key in targets:
            for t in clevr_cfg.DETECTOR_DIST_THRESH:
                s = sample_stats[t]
                m = merged[key][t]
                for a in ATTR_NAMES:
                    m["correct"][a] += s["correct"][a]
                m["correct_all"] += s["correct_all"]
                m["total_matched"] += s["total_matched"]
                m["total_pred"] += s["total_pred"]
                m["total_gt"] += s["total_gt"]

    return merged


def stats_to_results(merged_per_split, clevr_cfg):
    """Convert per-split merged stats to final results dict.

    Returns {"overall": {thresh: metrics}, "easy": {...}, ...}
    """
    results = {}
    for split_key, merged in merged_per_split.items():
        results[split_key] = {}
        for t in clevr_cfg.DETECTOR_DIST_THRESH:
            s = merged[t]
            nm = max(s["total_matched"], 1)
            det_prec = s["total_matched"] / max(s["total_pred"], 1)
            det_rec = s["total_matched"] / max(s["total_gt"], 1)
            det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-8)
            attr_acc = {a: s["correct"][a] / nm * 100 for a in ATTR_NAMES}
            all_acc = s["correct_all"] / nm * 100
            results[split_key][t] = {
                "det_P": det_prec, "det_R": det_rec, "det_F1": det_f1,
                "attr_acc": attr_acc, "all_attrs_acc": all_acc,
                "total_matched": s["total_matched"],
                "total_pred": s["total_pred"], "total_gt": s["total_gt"],
            }
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Training run directory (contains checkpoints/)")
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Specific steps to eval (default: all checkpoints)")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Val images dir (default: from checkpoint args)")
    parser.add_argument("--num_samples_per_split", type=int, default=30,
                        help="Number of val samples per split (easy/medium/hard)")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Diffusion sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for reconstruction per GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    cli = parser.parse_args()

    # ── Accelerate setup ──
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process

    # ── Find checkpoints ──
    ckpt_dir = os.path.join(cli.run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        if is_main:
            print(f"No checkpoints dir found: {ckpt_dir}")
        return

    all_steps = []
    for d in sorted(os.listdir(ckpt_dir)):
        if d.startswith("step_"):
            try:
                s = int(d.split("_")[1])
                if os.path.isfile(os.path.join(ckpt_dir, d, "checkpoint.pt")):
                    all_steps.append(s)
            except ValueError:
                pass

    if cli.steps:
        steps = [s for s in cli.steps if s in all_steps]
        if is_main:
            missing = [s for s in cli.steps if s not in all_steps]
            if missing:
                print(f"Warning: steps not found: {missing}")
                print(f"Available: {all_steps}")
    else:
        steps = all_steps

    if not steps:
        if is_main:
            print("No checkpoints to evaluate.")
        return

    if is_main:
        print(f"Evaluating {len(steps)} checkpoint(s): {steps}")
        print(f"GPUs: {accelerator.num_processes}, "
              f"samples/split: {cli.num_samples_per_split}, "
              f"cfg: {cli.guidance_scale}, steps: {cli.num_steps}")

    # ── Load CLEVR eval tools (all ranks) ──
    clevr_eval_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "clevr_eval"))
    if clevr_eval_dir not in sys.path:
        sys.path.insert(0, clevr_eval_dir)

    import config as clevr_cfg
    from models.detector import CenterDetector
    from models.classifier import AttributeClassifier
    from evaluate import extract_peaks, match_detections

    det_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "detector_best.pt")
    cls_ckpt = os.path.join(clevr_cfg.CHECKPOINT_DIR, "classifier_best.pt")
    if not os.path.exists(det_ckpt) or not os.path.exists(cls_ckpt):
        if is_main:
            print("ERROR: detector/classifier checkpoints not found!")
            print(f"  Expected: {det_ckpt}")
            print(f"  Expected: {cls_ckpt}")
        return

    # Load detector & classifier on each GPU
    detector = CenterDetector(backbone_name=clevr_cfg.DETECTOR_BACKBONE).to(device)
    detector.load_state_dict(
        torch.load(det_ckpt, map_location=device, weights_only=True)["model"])
    detector.eval()

    classifier = AttributeClassifier().to(device)
    classifier.load_state_dict(
        torch.load(cls_ckpt, map_location=device, weights_only=True)["model"])
    classifier.eval()

    det_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    crop_transform = transforms.Compose([
        transforms.Resize((clevr_cfg.CROP_SIZE, clevr_cfg.CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    output_dir = os.path.join(cli.run_dir, "clevr_eval")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for step in sorted(steps):
        if is_main:
            print(f"\n{'#'*60}")
            print(f"  Step {step}")
            print(f"{'#'*60}")

        ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}", "checkpoint.pt")
        model, train_args = load_model_from_ckpt(ckpt_path, device=device,
                                                  use_ema=cli.use_ema)
        # Wrap with accelerate for DDP
        model = accelerator.prepare(model)
        raw_model = accelerator.unwrap_model(model)

        # Val dir
        val_images_dir = cli.val_dir or getattr(train_args, "val_dir", None)
        if val_images_dir is None:
            train_dir = getattr(train_args, "train_dir", None)
            if train_dir and "varied" in train_dir:
                val_images_dir = train_dir.replace("varied", "varied_val")
        if val_images_dir is None or not os.path.isdir(val_images_dir):
            if is_main:
                print(f"ERROR: cannot determine val_dir. Use --val_dir")
            continue

        scenes_dir = os.path.join(os.path.dirname(val_images_dir.rstrip("/")), "scenes")
        if not os.path.isdir(scenes_dir):
            if is_main:
                print(f"ERROR: scenes dir not found: {scenes_dir}")
            continue

        # Build val dataset
        tfm = transforms.Compose([
            transforms.Resize(train_args.image_size),
            transforms.CenterCrop(train_args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        val_ds = datasets.ImageFolder(val_images_dir, transform=tfm)

        # Build deterministic eval set (same on all ranks)
        eval_set = build_eval_indices(val_ds, val_images_dir,
                                      num_per_split=cli.num_samples_per_split,
                                      seed=cli.seed)
        total_samples = len(eval_set)
        if is_main:
            splits_count = {}
            for _, sn in eval_set:
                splits_count[sn] = splits_count.get(sn, 0) + 1
            print(f"  Eval set: {total_samples} samples "
                  f"({', '.join(f'{k}={v}' for k, v in sorted(splits_count.items()))})")

        # Shard across GPUs: each rank takes its slice
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        my_items = eval_set[rank::world_size]

        # Run eval in batches on this rank's shard
        # local_stats: list of (sample_stats_dict, split_name)
        local_stats = []
        n_batches = (len(my_items) + cli.batch_size - 1) // cli.batch_size
        pbar = tqdm(range(n_batches), desc=f"[GPU {rank}] CLEVR eval",
                    disable=not is_main)
        for bi in pbar:
            batch_items = my_items[bi * cli.batch_size : (bi + 1) * cli.batch_size]
            batch_results = eval_batch(
                raw_model, train_args, val_ds, batch_items, val_images_dir,
                scenes_dir, detector, classifier, clevr_cfg,
                extract_peaks, match_detections,
                device, cli.guidance_scale, cli.num_steps,
                det_transform, crop_transform,
            )
            # Pair each result with its split name
            for res, (_, split_name) in zip(batch_results, batch_items):
                local_stats.append((res, split_name))

        # Merge local stats (per-split + overall)
        local_merged = merge_stats(local_stats, clevr_cfg)

        # Gather across GPUs
        accelerator.wait_for_everyone()
        if accelerator.num_processes > 1:
            local_json = json.dumps(local_merged, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x)
            local_bytes = local_json.encode("utf-8")

            local_len = torch.tensor([len(local_bytes)], dtype=torch.long, device=device)
            all_lens = accelerator.gather(local_len)
            max_len = all_lens.max().item()

            padded = torch.zeros(max_len, dtype=torch.uint8, device=device)
            padded[:len(local_bytes)] = torch.tensor(list(local_bytes), dtype=torch.uint8, device=device)
            all_padded = accelerator.gather(padded)
            all_padded = all_padded.view(accelerator.num_processes, max_len)

            if is_main:
                # Merge all ranks' per-split stats
                global_merged = {}
                for r in range(accelerator.num_processes):
                    rlen = all_lens[r].item()
                    rbytes = bytes(all_padded[r, :rlen].cpu().tolist())
                    rank_merged = json.loads(rbytes.decode("utf-8"))
                    # rank_merged: {"overall": {thresh: stats}, "easy": {...}, ...}
                    for split_key, thresh_stats in rank_merged.items():
                        if split_key not in global_merged:
                            global_merged[split_key] = _empty_stats(clevr_cfg)
                        for t_str, s in thresh_stats.items():
                            t = int(t_str)
                            m = global_merged[split_key][t]
                            for a in ATTR_NAMES:
                                m["correct"][a] += s["correct"][a]
                            m["correct_all"] += s["correct_all"]
                            m["total_matched"] += s["total_matched"]
                            m["total_pred"] += s["total_pred"]
                            m["total_gt"] += s["total_gt"]

                results = stats_to_results(global_merged, clevr_cfg)
        else:
            if is_main:
                results = stats_to_results(local_merged, clevr_cfg)

        # Print & save (main only)
        if is_main:
            print(f"\n{'='*60}")
            print(f"  CLEVR Eval — Step {step} ({total_samples} samples, cfg={cli.guidance_scale})")
            print(f"{'='*60}")
            # Print per-split then overall
            split_order = sorted([k for k in results if k != "overall"]) + ["overall"]
            for split_key in split_order:
                split_results = results[split_key]
                label = split_key.upper() if split_key != "overall" else "OVERALL"
                print(f"\n  [{label}]")
                for t in clevr_cfg.DETECTOR_DIST_THRESH:
                    r = split_results[t]
                    aa = r["attr_acc"]
                    print(f"    @{t}px  Det: P={r['det_P']:.3f} R={r['det_R']:.3f} F1={r['det_F1']:.3f}  "
                          + " ".join(f"{a}={aa[a]:.1f}%" for a in ATTR_NAMES)
                          + f"  all={r['all_attrs_acc']:.1f}%")

            all_results[step] = results
            with open(os.path.join(output_dir, f"step_{step:07d}.json"), "w") as f:
                json.dump(results, f, indent=2)

        # Cleanup
        del model, raw_model
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    # Summary
    if is_main and all_results:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        # Get split keys from first result
        first_result = next(iter(all_results.values()))
        split_keys = sorted([k for k in first_result if k != "overall"]) + ["overall"]

        for split_key in split_keys:
            label = split_key.upper() if split_key != "overall" else "OVERALL"
            print(f"\n{'='*70}")
            print(f"  Summary (@10px) [{label}] — {cli.num_samples_per_split} samples/split, seed={cli.seed}")
            print(f"{'='*70}")
            print(f"  {'Step':>8s}  {'Det_F1':>6s}  {'Color':>6s}  {'Shape':>6s}  "
                  f"{'Size':>6s}  {'Mater':>6s}  {'All':>6s}")
            print(f"  {'-'*62}")
            for step in sorted(all_results.keys()):
                split_data = all_results[step].get(split_key, {})
                r = split_data.get(10, split_data.get("10", {}))
                if not r:
                    continue
                aa = r.get("attr_acc", {})
                print(f"  {step:>8d}  {r['det_F1']:>6.3f}  {aa.get('color',0):>5.1f}%  "
                      f"{aa.get('shape',0):>5.1f}%  {aa.get('size',0):>5.1f}%  "
                      f"{aa.get('material',0):>5.1f}%  {r['all_attrs_acc']:>5.1f}%")

        # Plot
        plot_results(all_results, output_dir, clevr_cfg)


def plot_results(all_results, output_dir, clevr_cfg):
    """Plot CLEVR eval metrics across checkpoints, per split + overall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(all_results.keys())
    steps_k = [s / 1000 for s in steps]
    thresh = 10  # main threshold for plots

    # Determine splits from data
    first_result = next(iter(all_results.values()))
    split_keys = sorted([k for k in first_result if k != "overall"]) + ["overall"]

    def _get(step, split_key, key):
        r = all_results[step].get(split_key, {}).get(thresh, all_results[step].get(split_key, {}).get(str(thresh), {}))
        return r.get(key, 0)

    def _get_attr(step, split_key, attr):
        r = all_results[step].get(split_key, {}).get(thresh, all_results[step].get(split_key, {}).get(str(thresh), {}))
        return r.get("attr_acc", {}).get(attr, 0)

    # ── Plot 1: Per-split overview (one row per split) ──
    n_splits = len(split_keys)
    fig, axes = plt.subplots(n_splits, 3, figsize=(18, 4.5 * n_splits), squeeze=False)

    attr_colors = {"color": "tab:red", "shape": "tab:blue", "size": "tab:green", "material": "tab:purple"}

    for row, split_key in enumerate(split_keys):
        label = split_key.upper() if split_key != "overall" else "OVERALL"

        # Col 0: Detection P/R/F1
        ax = axes[row, 0]
        ax.plot(steps_k, [_get(s, split_key, "det_P") for s in steps], "o-", label="P", color="tab:blue", markersize=4)
        ax.plot(steps_k, [_get(s, split_key, "det_R") for s in steps], "s-", label="R", color="tab:orange", markersize=4)
        ax.plot(steps_k, [_get(s, split_key, "det_F1") for s in steps], "D-", label="F1", color="tab:green", linewidth=2, markersize=4)
        ax.set_ylabel("Score")
        ax.set_title(f"[{label}] Detection @{thresh}px")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 1: Per-attribute accuracy
        ax = axes[row, 1]
        for a in ATTR_NAMES:
            ax.plot(steps_k, [_get_attr(s, split_key, a) for s in steps], "o-", label=a, color=attr_colors[a], markersize=4)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"[{label}] Attribute Accuracy @{thresh}px")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 2: All-correct + Det F1
        ax = axes[row, 2]
        ax.plot(steps_k, [_get(s, split_key, "all_attrs_acc") for s in steps], "D-",
                label="All Correct", color="tab:red", linewidth=2, markersize=4)
        ax.set_ylabel("Accuracy (%)", color="tab:red")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(steps_k, [_get(s, split_key, "det_F1") for s in steps], "s--",
                 label="Det F1", color="tab:blue", linewidth=2, alpha=0.7, markersize=4)
        ax2.set_ylabel("Det F1", color="tab:blue")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
        ax.set_title(f"[{label}] Quality @{thresh}px")

        # x-label only on bottom row
        if row == n_splits - 1:
            for c in range(3):
                axes[row, c].set_xlabel("Step (k)")

    plt.suptitle("CLEVR Eval over Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "clevr_eval_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # ── Plot 2: Cross-split comparison (Det F1 and All Correct) ──
    split_colors = {"easy": "tab:green", "medium": "tab:orange", "hard": "tab:red", "overall": "tab:blue"}
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    for split_key in split_keys:
        c = split_colors.get(split_key, "gray")
        lw = 2.5 if split_key == "overall" else 1.5
        ax.plot(steps_k, [_get(s, split_key, "det_F1") for s in steps],
                "o-", label=split_key, color=c, linewidth=lw, markersize=4)
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("F1")
    ax.set_title(f"Detection F1 @{thresh}px by Split")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for split_key in split_keys:
        c = split_colors.get(split_key, "gray")
        lw = 2.5 if split_key == "overall" else 1.5
        ax.plot(steps_k, [_get(s, split_key, "all_attrs_acc") for s in steps],
                "o-", label=split_key, color=c, linewidth=lw, markersize=4)
    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"All Attrs Correct @{thresh}px by Split")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("CLEVR Eval — Split Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "clevr_eval_splits.png")
    fig2.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Split comparison plot saved to {plot_path2}")


if __name__ == "__main__":
    main()
