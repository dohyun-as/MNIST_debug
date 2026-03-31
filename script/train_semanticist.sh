#!/bin/bash
# SemantIST Stage-1 Tokenizer Training on ImageNet 256×256
#
# Model: ViT-B encoder + DiT-L/2 decoder + REPA (DINOv2)
# VAE:   xwen99/mar-vae-kl16 (16× downsample, 16ch latent)
#
# Matches tokenizer_l.yaml config from the SemantIST paper:
#   - 256 slots, slot_dim=16, norm_slots, causal attention
#   - Nested slot sampling after epoch 50
#   - REPA alignment weight=1.0
#   - Cosine LR, 100 epoch warmup, 400 total epochs
#
# Usage:
#   bash script/train_semanticist.sh
#   GPUS=0,1,2,3 bash script/train_semanticist.sh

set -e

# ── PYTHONPATH for semanticist imports ──
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../semanticist:${PYTHONPATH}"

# ── GPU / batch config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=256
GRAD_ACCUM=1
# SemantIST default: effective batch = 256 * 8 = 2048
# With 4 GPUs: 256 * 4 = 1024 → grad_accum=2 for 2048
if [ $((BATCH_PER_GPU * NUM_GPUS)) -lt 2048 ]; then
    GRAD_ACCUM=$((2048 / (BATCH_PER_GPU * NUM_GPUS)))
fi

EFFECTIVE=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS), batch/gpu=$BATCH_PER_GPU, accum=$GRAD_ACCUM, effective=$EFFECTIVE"

# ── Cache directory (reuse existing) ──
CACHE_DIR="runs/imagenet_256_injection/latent_cache"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/train_semanticist.py \
    --output_dir runs/semanticist_l \
    --dataset_root ../imagenet/ILSVRC/Data/CLS-LOC \
    --cache_dir $CACHE_DIR \
    --encoder vit_base_patch16 \
    --enc_img_size 256 \
    --num_slots 256 \
    --slot_dim 16 \
    --norm_slots True \
    --dit_model DiT-L-2 \
    --vae xwen99/mar-vae-kl16 \
    --drop_path_rate 0.1 \
    --enable_nest_after 50 \
    --use_repa \
    --repa_loss_weight 1.0 \
    --repa_encoder_depth 8 \
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
    --num_workers 8 \
    --seed 42 \
    "$@"
