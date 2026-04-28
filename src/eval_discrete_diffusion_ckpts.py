#!/usr/bin/env python
"""
eval_discrete_diffusion_ckpts.py
================================
Evaluate discrete diffusion checkpoints on train AND val sets.
Tests both EMA and non-EMA weights where available.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
        src/eval_discrete_diffusion_ckpts.py \
        --ckpt_root ./runs/clevr/discrete_diff/ckpt \
        --steps 50000 100000 150000 200000 \
        --pretrained_dir ./runs/clevr/dit_vit_flow_fsq_mask075_CA \
        --eval_num_samples 100 \
        --eval_num_steps 128 \
        --decode_num_steps 50 \
        --output_dir ./runs/clevr/discrete_diff/ckpt_eval_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed

# Reuse everything from training script
from train_discrete_diffusion_v2 import (
    load_pretrained_model,
    cache_all_tokens,
    decode_tokens_to_images,
    save_sample_grid,
    CLEVRImageDataset,
    CachedTokenDataset,
    CLEVRConditionEncoder,
    CLEVRTextConditionEncoder,
    clevr_text_to_token_ids,
    clevr_json_to_token_ids,
    DIT,
    DiscreteDiffusion,
    EMA,
    _eval_clevr,
    _select_eval_indices,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", type=str, required=True,
                   help="Root dir containing step* checkpoint dirs")
    p.add_argument("--steps", type=int, nargs="+", required=True,
                   help="Which steps to evaluate, e.g. 50000 100000 150000 200000")
    p.add_argument("--pretrained_dir", type=str, required=True,
                   help="Pretrained continuous diffusion model dir")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to save eval results")

    # Dataset paths (defaults match training script)
    p.add_argument("--train_image_root", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied/images")
    p.add_argument("--train_cond_dir", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied/conditions_margin50_augmented")
    p.add_argument("--val_image_root", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/images")
    p.add_argument("--val_cond_dir", type=str,
                   default="../clevr-dataset-gen/output/clevr_256_varied_val/conditions_margin50_augmented")
    p.add_argument("--splits", nargs="+", default=["easy", "medium", "hard"])

    # Token cache (reuse from training)
    p.add_argument("--token_cache_dir", type=str, default=None,
                   help="If set, reuse cached tokens from training run")

    # Eval config
    p.add_argument("--eval_num_samples", type=int, default=100,
                   help="Number of samples per split")
    p.add_argument("--eval_num_steps", type=int, default=128)
    p.add_argument("--decode_num_steps", type=int, default=50)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--sampler", type=str, default="ddpm_cache",
                   choices=["ddpm_cache"],
                   help="Sampler to use (ddpm_cache only for speed)")

    # Model config (must match training)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_blocks", type=int, default=12)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--mlp_ratio", type=int, default=4)
    p.add_argument("--model_dropout", type=float, default=0.1)
    p.add_argument("--pos_emb_type", type=str, default="multires",
                   choices=["multires", "1d"])
    p.add_argument("--cond_type", type=str, default="json",
                   choices=["json", "text"])
    p.add_argument("--noise_type", type=str, default="loglinear")
    p.add_argument("--noise_eps", type=float, default=1e-3)
    p.add_argument("--sampling_eps", type=float, default=1e-3)

    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(args, data_vocab_size, seq_len, level_sizes, device):
    """Build the discrete diffusion model (same arch as training)."""
    backbone_vocab_size = data_vocab_size + 1  # +1 for mask token

    dit_kwargs = dict(
        vocab_size=backbone_vocab_size,
        seq_len=seq_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        cond_dim=args.cond_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.model_dropout,
        causal=False,
        pos_emb_type=args.pos_emb_type,
        level_sizes=level_sizes,
    )
    backbone = DIT(**dit_kwargs)
    diffusion = DiscreteDiffusion(
        backbone=backbone,
        vocab_size=data_vocab_size,
        noise_type=args.noise_type,
        noise_eps=args.noise_eps,
        antithetic_sampling=True,
        importance_sampling=False,
        change_of_variables=False,
        sampling_eps=args.sampling_eps,
    )
    return diffusion


def load_checkpoint_weights(accelerator, diffusion, clevr_cond_encoder, ckpt_dir):
    """Load model + cond_encoder weights from accelerate checkpoint."""
    from safetensors.torch import load_file

    # accelerate saves model.safetensors (diffusion) and model_1.safetensors (cond_encoder)
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    cond_path = os.path.join(ckpt_dir, "model_1.safetensors")

    if os.path.isfile(model_path):
        state = load_file(model_path, device=str(accelerator.device))
        diffusion.load_state_dict(state, strict=True)
    else:
        raise FileNotFoundError(f"No model.safetensors in {ckpt_dir}")

    if os.path.isfile(cond_path) and clevr_cond_encoder is not None:
        cond_state = load_file(cond_path, device=str(accelerator.device))
        clevr_cond_encoder.load_state_dict(cond_state, strict=True)

    return diffusion, clevr_cond_encoder


def load_ema_weights(diffusion, ema_path, device):
    """Load EMA shadow weights and apply to model."""
    shadow = torch.load(ema_path, map_location=device, weights_only=False)
    params = list(diffusion.parameters())
    assert len(shadow) == len(params), \
        f"EMA shadow length {len(shadow)} != model params {len(params)}"
    for s, p in zip(shadow, params):
        p.data.copy_(s.to(device))
    return diffusion


def run_eval_on_dataset(model, dataset, step, args, accelerator,
                        pretrained_model, discretizer, level_sizes,
                        clevr_cond_encoder, clevr_detector, clevr_classifier,
                        save_dir, label):
    """Run _eval_clevr on a dataset and collect results."""
    # Create a mock args object with needed fields
    class EvalArgs:
        pass
    eval_args = EvalArgs()
    eval_args.eval_num_samples = args.eval_num_samples
    eval_args.eval_num_steps = args.eval_num_steps
    eval_args.decode_num_steps = args.decode_num_steps
    eval_args.image_size = args.image_size
    eval_args.model_type = "diffusion"
    eval_args.sampler = args.sampler
    eval_args.log_with = None  # no tensorboard logging
    eval_args.dataset_type = "clevr"
    eval_args.seq_len = sum(s * s for s in level_sizes)
    eval_args.output_dir = save_dir

    model.eval()
    _eval_clevr(
        model, step, eval_args, accelerator, save_dir,
        pretrained_model, discretizer, level_sizes,
        clevr_cond_encoder, dataset,
        clevr_detector=clevr_detector,
        clevr_classifier=clevr_classifier,
        log_prefix=label,
    )
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    accelerator.print("=" * 60)
    accelerator.print("Discrete Diffusion — Checkpoint Evaluation")
    accelerator.print(f"Steps to evaluate: {args.steps}")
    accelerator.print(f"Samples per split: {args.eval_num_samples}")
    accelerator.print("=" * 60)

    # ── Load pretrained model ──
    accelerator.print(f"[pretrained] Loading from {args.pretrained_dir}")
    pretrained_model, encoder, discretizer, level_sizes, data_vocab_size, _ = \
        load_pretrained_model(args.pretrained_dir, device=accelerator.device)
    seq_len = sum(s * s for s in level_sizes)
    accelerator.print(f"[pretrained] level_sizes={level_sizes}, seq_len={seq_len}, vocab={data_vocab_size}")

    # ── Load datasets ──
    accelerator.print("[data] Loading CLEVR datasets...")
    train_img_ds = CLEVRImageDataset(
        args.train_image_root, condition_dir=args.train_cond_dir,
        image_size=args.image_size, splits=args.splits)
    val_img_ds = CLEVRImageDataset(
        args.val_image_root, condition_dir=args.val_cond_dir,
        image_size=args.image_size, splits=args.splits)

    # Token caching
    cache_dir = args.token_cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(args.output_dir, "token_cache")

    train_img_only = CLEVRImageDataset(
        args.train_image_root, condition_dir=None,
        image_size=args.image_size, splits=args.splits)
    val_img_only = CLEVRImageDataset(
        args.val_image_root, condition_dir=None,
        image_size=args.image_size, splits=args.splits)

    train_cache_path = os.path.join(cache_dir, "clevr_train_tok.pt")
    val_cache_path = os.path.join(cache_dir, "clevr_val_tok.pt")

    train_tok_unique = cache_all_tokens(
        encoder, discretizer, train_img_only, accelerator.device,
        batch_size=64, cache_path=train_cache_path, accelerator=accelerator)
    val_tok_unique = cache_all_tokens(
        encoder, discretizer, val_img_only, accelerator.device,
        batch_size=64, cache_path=val_cache_path, accelerator=accelerator)
    accelerator.wait_for_everyone()

    train_path_to_idx = {p: i for i, p in enumerate(train_img_only.image_paths)}
    val_path_to_idx = {p: i for i, p in enumerate(val_img_only.image_paths)}

    train_tok_indices = [train_path_to_idx[p] for p in train_img_ds.image_paths]
    val_tok_indices = [val_path_to_idx[p] for p in val_img_ds.image_paths]

    train_tok = train_tok_unique[train_tok_indices]
    val_tok = val_tok_unique[val_tok_indices]

    train_conditions = [train_img_ds.get_condition(i) for i in range(len(train_img_ds))]
    val_conditions = [val_img_ds.get_condition(i) for i in range(len(val_img_ds))]

    if args.cond_type == "text":
        cond_tokenizer_fn = clevr_text_to_token_ids
    else:
        cond_tokenizer_fn = clevr_json_to_token_ids

    train_dataset = CachedTokenDataset(
        train_tok, clevr_conditions=train_conditions,
        cond_tokenizer_fn=cond_tokenizer_fn)
    val_dataset = CachedTokenDataset(
        val_tok, clevr_conditions=val_conditions,
        cond_tokenizer_fn=cond_tokenizer_fn)

    accelerator.print(f"[data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Move pretrained encoder to CPU (only need decoder for eval)
    encoder.cpu()
    del encoder
    torch.cuda.empty_cache()

    # ── Load eval models (detector + classifier) ──
    clevr_detector, clevr_classifier = None, None
    try:
        from eval_clevr_condition import load_eval_models
        clevr_detector, clevr_classifier = load_eval_models(device=accelerator.device)
        accelerator.print("[clevr] loaded detector + classifier")
    except Exception as e:
        accelerator.print(f"[clevr] WARNING: could not load eval models: {e}")

    # ── All results collector ──
    all_results = {}

    # ── Evaluate each checkpoint ──
    for step in args.steps:
        ckpt_dir = os.path.join(args.ckpt_root, f"step{step}")
        if not os.path.isdir(ckpt_dir):
            accelerator.print(f"\n[SKIP] {ckpt_dir} does not exist")
            continue

        ema_path = os.path.join(ckpt_dir, "ema.pt")
        has_ema = os.path.isfile(ema_path)

        # Determine which variants to eval
        variants = ["base"]
        if has_ema:
            variants.append("ema")

        for variant in variants:
            label = f"step{step}_{variant}"
            accelerator.print(f"\n{'='*60}")
            accelerator.print(f"Evaluating: {label}")
            accelerator.print(f"{'='*60}")

            # Build fresh model
            diffusion = build_model(args, data_vocab_size, seq_len,
                                    level_sizes, accelerator.device)
            if args.cond_type == "text":
                clevr_cond_encoder = CLEVRTextConditionEncoder(args.hidden_size)
            else:
                clevr_cond_encoder = CLEVRConditionEncoder(args.hidden_size)

            # Load weights
            diffusion, clevr_cond_encoder = load_checkpoint_weights(
                accelerator, diffusion, clevr_cond_encoder, ckpt_dir)

            if variant == "ema":
                diffusion = load_ema_weights(diffusion, ema_path, accelerator.device)
                accelerator.print(f"  [ema] Applied EMA weights from {ema_path}")

            diffusion = diffusion.to(accelerator.device)
            clevr_cond_encoder = clevr_cond_encoder.to(accelerator.device)
            diffusion.eval()
            clevr_cond_encoder.eval()

            step_results = {"step": step, "variant": variant}

            # ── Eval on val set ──
            val_save_dir = os.path.join(args.output_dir, label, "val")
            os.makedirs(val_save_dir, exist_ok=True)

            accelerator.print(f"\n  --- Val set eval ---")
            t0 = time.time()
            run_eval_on_dataset(
                diffusion, val_dataset, step, args, accelerator,
                pretrained_model, discretizer, level_sizes,
                clevr_cond_encoder, clevr_detector, clevr_classifier,
                val_save_dir, f"{label}/val")
            dt_val = time.time() - t0
            accelerator.print(f"  Val eval took {dt_val:.1f}s")

            # Read val results
            val_json = os.path.join(
                val_save_dir, f"step_{step:07d}_clevr_ddpm_cache_cond_eval.json")
            if os.path.isfile(val_json):
                with open(val_json) as f:
                    step_results["val"] = json.load(f)

            # ── Eval on train set ──
            train_save_dir = os.path.join(args.output_dir, label, "train")
            os.makedirs(train_save_dir, exist_ok=True)

            accelerator.print(f"\n  --- Train set eval ---")
            t0 = time.time()
            run_eval_on_dataset(
                diffusion, train_dataset, step, args, accelerator,
                pretrained_model, discretizer, level_sizes,
                clevr_cond_encoder, clevr_detector, clevr_classifier,
                train_save_dir, f"{label}/train")
            dt_train = time.time() - t0
            accelerator.print(f"  Train eval took {dt_train:.1f}s")

            # Read train results
            train_json = os.path.join(
                train_save_dir, f"step_{step:07d}_clevr_ddpm_cache_cond_eval.json")
            if os.path.isfile(train_json):
                with open(train_json) as f:
                    step_results["train"] = json.load(f)

            all_results[label] = step_results

            # Clean up
            del diffusion, clevr_cond_encoder
            torch.cuda.empty_cache()

    # ── Save summary ──
    if accelerator.is_main_process:
        summary_path = os.path.join(args.output_dir, "eval_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        accelerator.print(f"\n[done] Summary saved to {summary_path}")

        # Print compact summary table
        accelerator.print(f"\n{'='*70}")
        accelerator.print(f"{'Label':<25s} {'Split':<6s} {'Count':>8s} "
                          f"{'EntPres':>8s} {'RelAcc':>8s}")
        accelerator.print(f"{'-'*70}")
        for label, res in sorted(all_results.items()):
            for split_name in ["val", "train"]:
                if split_name not in res:
                    continue
                d = res[split_name].get("overall", {})
                if not d:
                    continue
                cnt = d.get("count_accuracy", 0)
                ent = d.get("entity_presence_accuracy", 0)
                rel = d.get("rel_accuracy", 0)
                accelerator.print(
                    f"{label:<25s} {split_name:<6s} {cnt:>7.1f}% "
                    f"{ent:>7.1f}% {rel:>7.1f}%")
        accelerator.print(f"{'='*70}")


if __name__ == "__main__":
    main()
