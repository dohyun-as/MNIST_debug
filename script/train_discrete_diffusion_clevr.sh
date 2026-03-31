#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR — Discrete Diffusion (MDLM-style) on multi-res FSQ tokens
#  JSON-conditioned (entity attrs + spatial relations) via cross-attn
# ──────────────────────────────────────────────────────────────────
#
#  Requires: pretrained continuous diffusion model (e.g., from
#  train_clevr_dit.sh) with FSQ encoder + discretizer.
#
#  Token extraction: encoder(image) → FSQ → flat token sequence
#  Conditioning: CLEVR JSON (entities, relations) → cross-attention
#  Positional embedding: multires (level + 2D row/col per level)
# ──────────────────────────────────────────────────────────────────

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── GPU config ──
GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ── Paths ──
PRETRAINED_DIR=${PRETRAINED_DIR:-"./runs/clevr_256_dit_flow_fsq_CA"}
CLEVR_IMAGE_ROOT=${CLEVR_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied/images"}
CLEVR_COND_DIR=${CLEVR_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied/conditions_margin30"}
OUTPUT_DIR="./runs/clevr_discrete_diff_v2"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type clevr \
  --dataset_root ${CLEVR_IMAGE_ROOT} \
  --clevr_condition_dir ${CLEVR_COND_DIR} \
  --clevr_train_splits easy medium hard \
  --clevr_val_splits easy \
  --pretrained_output_dir ${PRETRAINED_DIR} \
  --image_size 256 \
  --max_train_steps 200000 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  --hidden_size 512 \
  --n_heads 8 \
  --n_blocks 12 \
  --cond_dim 256 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type multires \
  --noise_type loglinear \
  --uncond_drop_prob 0.1 \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 5000 \
  --log_every 100 \
  --eval_num_samples 32 \
  --eval_num_steps 128 \
  --decode_num_steps 50 \
  --seed 42 \
  --mixed_precision bf16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --sampler ddpm_cache \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    src/train_discrete_diffusion_v2.py $COMMON_ARGS
fi
