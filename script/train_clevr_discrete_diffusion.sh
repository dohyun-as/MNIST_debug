#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — Discrete Diffusion (MDLM-style)
#  JSON condition-based token generation
# ──────────────────────────────────────────────────────────────────
#
#  Extracts multi-resolution FSQ tokens from a pretrained continuous
#  diffusion model (DiT backbone), then trains an MDLM discrete
#  diffusion model to generate those tokens conditioned on CLEVR
#  scene descriptions (entities + spatial relations from JSON).
#
#  Usage:
#    bash script/train_clevr_discrete_diffusion.sh
#    GPUS=0,1 bash script/train_clevr_discrete_diffusion.sh
#    PRETRAINED=runs/clevr_256_dit_vit_flow_fsq_mask075_CA_bugfix \
#      bash script/train_clevr_discrete_diffusion.sh

set -e

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── GPU config ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Dynamic port
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ── Paths ──
CLEVR_IMAGE_ROOT="/workspace/NAS/sr_diffusion/project/clevr-dataset-gen/output/clevr_256_varied/images"
CLEVR_COND_DIR="/workspace/NAS/sr_diffusion/project/clevr-dataset-gen/output/clevr_256_varied/conditions_margin30"
PRETRAINED_DIR="${PRETRAINED:-runs/clevr_256_dit_vit_flow_fsq_mask075_CA_bugfix}"
OUTPUT_DIR="runs/clevr_discrete_diffusion"

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
echo "Pretrained: $PRETRAINED_DIR"
echo "Output: $OUTPUT_DIR"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    --main_process_port $MASTER_PORT \
    src/train_discrete_diffusion_v2.py \
    --dataset_type clevr \
    --dataset_root "$CLEVR_IMAGE_ROOT" \
    --clevr_condition_dir "$CLEVR_COND_DIR" \
    --clevr_train_splits easy medium hard \
    --clevr_val_splits easy \
    --pretrained_output_dir "$PRETRAINED_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --image_size 256 \
    --hidden_size 512 \
    --n_heads 8 \
    --n_blocks 12 \
    --cond_dim 256 \
    --mlp_ratio 4 \
    --model_dropout 0.1 \
    --pos_emb_type multires \
    --noise_type loglinear \
    --antithetic_sampling \
    --sampler ddpm_cache \
    --max_train_steps 200000 \
    --batch_size 256 \
    --lr 3e-4 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --ema_decay 0.9999 \
    --uncond_drop_prob 0.1 \
    --mixed_precision fp16 \
    --log_every 100 \
    --eval_every 5000 \
    --save_every 10000 \
    --eval_num_samples 32 \
    --eval_num_steps 128 \
    --decode_num_steps 50 \
    --seed 42 \
    --log_with tensorboard \
