#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Sudoku — Discrete Diffusion V2 (grid-only, compatible with v1)
# ──────────────────────────────────────────────────────────────────

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

GPUS=${GPUS:-"0,1"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

SRM_CONFIG="./config/sudoku_config.json"
OUTPUT_DIR="./runs/sudoku_discrete_diff_v2"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --dataset_type sudoku \
  --sudoku_config ${SRM_CONFIG} \
  --grid_only \
  --grid_hw 9 \
  --grid_vocab_size 9 \
  --max_train_steps 50000 \
  --batch_size 1024 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --hidden_size 256 \
  --n_heads 8 \
  --n_blocks 6 \
  --cond_dim 128 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --pos_emb_type 2d \
  --noise_type loglinear \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 3000 \
  --log_every 100 \
  --eval_num_samples 2048 \
  --eval_num_steps 81 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --sampler confidence \
  --tokens_per_step 1 \
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
