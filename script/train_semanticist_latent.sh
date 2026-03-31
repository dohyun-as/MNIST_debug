#!/bin/bash
# SemantIST Stage-1 Training — Latent-only encoder mode
#
# Same as train_semanticist.sh but with --enc_use_latent:
#   - ViT encoder receives VAE latents (16×16×16) instead of images
#   - Latent-only consolidated cache → RAM preload (~5GB/rank)
#   - No image loading, no REPA (no DINOv2)
#
# Usage:
#   bash script/train_semanticist_latent.sh
#   GPUS=0,1,2,3 bash script/train_semanticist_latent.sh

set -e

# ── PYTHONPATH for semanticist imports ──
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../semanticist:${SCRIPT_DIR}/src:${PYTHONPATH}"

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"

# Performance note:
# - Do NOT force debug/slow-path NCCL settings in normal training.
# - Enable CUDA_LAUNCH_BLOCKING only when debugging correctness issues.
if [ "${DEBUG_SYNC:-0}" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    echo "DEBUG_SYNC=1 -> CUDA_LAUNCH_BLOCKING enabled (this will reduce throughput)."
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=64
GRAD_ACCUM=1
if [ $((BATCH_PER_GPU * NUM_GPUS)) -lt 2048 ]; then
    GRAD_ACCUM=$((2048 / (BATCH_PER_GPU * NUM_GPUS)))
fi

EFFECTIVE=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$EFFECTIVE"

# ── Cache directory (reuse existing) ──
CACHE_DIR="runs/imagenet_256_injection/latent_cache"

# CUDA_VISIBLE_DEVICES=0 python3 src/train_semanticist.py \
#   --output_dir runs/test_debug \
#   --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
#   --cache_dir runs/imagenet_256_injection/latent_cache \
#   --enc_use_latent \
#   --encoder vit_base_patch16 --enc_img_size 256 \
#   --num_slots 256 --slot_dim 16 --norm_slots True \
#   --dit_model DiT-L-2 --vae xwen99/mar-vae-kl16 \
#   --drop_path_rate 0.1 --enable_nest_after 50 \
#   --num_sampling_steps 250 --cfg 3.0 \
#   --num_epochs 1 --batch_size 4 --blr 2.5e-5 \
#   --weight_decay 0.05 --warmup_epochs 100 \
#   --max_grad_norm 3.0 --mixed_precision bf16 \
#   --num_workers 0 --seed 42


accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/train_semanticist.py \
    --output_dir runs/semanticist_l_latent \
    --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
    --cache_dir $CACHE_DIR \
    --enc_use_latent \
    --encoder vit_base_patch16 \
    --enc_img_size 256 \
    --num_slots 256 \
    --slot_dim 16 \
    --norm_slots True \
    --dit_model DiT-L-2 \
    --vae xwen99/mar-vae-kl16 \
    --drop_path_rate 0.1 \
    --enable_nest_after 50 \
    --num_sampling_steps 250 \
    --cfg 3.0 \
    --num_epochs 400 \
    --batch_size $BATCH_PER_GPU \
    --blr 2.5e-5 \
    --weight_decay 0.05 \
    --warmup_epochs 100 \
    --max_grad_norm 3.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --enable_ema \
    --ema_decay 0.999 \
    --log_every 100 \
    --save_every 10000 \
    --sample_every 5000 \
    --fid_every 50000 \
    --num_workers 0 \
    --seed 42 \
    "$@"
