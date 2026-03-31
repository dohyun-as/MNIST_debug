#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_LAUNCH_BLOCKING=1


CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRM_CONFIG="./config/sudoku_config.json"


# ──────────────────────────────────────────────────────────
#  1) Unconditional AR Training
# ──────────────────────────────────────────────────────────

# OUTPUT_DIR="./outputs_ar/unconditional"
# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
#   --sudoku_config ${SRM_CONFIG} \
#   --grid_only \
#   --max_train_steps 50000 \
#   --batch_size 1024 \
#   --lr 3e-4 \
#   --weight_decay 0.01 \
#   --warmup_steps 1000 \
#   --hidden_size 256 \
#   --n_heads 8 \
#   --n_blocks 6 \
#   --mlp_ratio 4 \
#   --model_dropout 0.1 \
#   --pos_emb_type 2d \
#   --ema_decay 0 \
#   --save_every 10000 \
#   --eval_every 3000 \
#   --log_every 100 \
#   --eval_num_samples 2048 \
#   --temperature 1.0 \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 1 \
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_AR.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_AR.py $COMMON_ARGS
# fi


# ──────────────────────────────────────────────────────────
#  2) Conditional AR Training
#     mask_ratio 0.0~1.0 → 전부 보이는 상태(0%)부터
#     전부 masked(100%)까지 uniform 하게 학습
# ──────────────────────────────────────────────────────────

OUTPUT_DIR="./outputs_ar/conditional_debug"
mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --sudoku_config ${SRM_CONFIG} \
  --grid_only \
  --conditional \
  --mask_ratio_min 0.0 \
  --mask_ratio_max 1.0 \
  --max_train_steps 50000 \
  --batch_size 1024 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --hidden_size 256 \
  --n_heads 8 \
  --n_blocks 6 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type 2d \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 1 \
  --log_every 100 \
  --eval_num_samples 2048 \
  --temperature 1.0 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --resume_dir "./outputs_ar/conditional/ckpt/step40000" \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_AR.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_AR.py $COMMON_ARGS
fi
