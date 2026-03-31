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
OUTPUT_DIR="./outputs/outputs_discrete_diffusion/image_cond"
TOKEN_CACHE_DIR="./outputs/outputs_discrete_diffusion/image_cond/token_cache"
COND_UNET_CKPT="./outputs/concat_pixel_sample_fsq888/ckpt/step40000"
COND_UNET_CONFIG="./config/unet_mnist_concat.json"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
  --sudoku_config ${SRM_CONFIG} \
  --image_cond_mode \
  --token_cache_dir ${TOKEN_CACHE_DIR} \
  --cond_unet_ckpt ${COND_UNET_CKPT} \
  --cond_unet_config ${COND_UNET_CONFIG} \
  --cond_image_size 288 \
  --cond_feat_channels 128 \
  --cond_discretizer_type fsq \
  --cond_fsq_levels 8 8 8 \
  --cond_concat_downsample_factor 32 \
  --cond_patch_conditioning \
  --cond_patch_grid_size 9 \
  --cond_eval_ddim_steps 50 \
  --eval_render_batch_size 16 \
  --init_embed_from_fsq \
  --max_train_steps 100000 \
  --batch_size 512 \
  --lr 3e-4 \
  --weight_decay 0.01 \
  --warmup_steps 1000 \
  --hidden_size 256 \
  --n_heads 8 \
  --n_blocks 6 \
  --cond_dim 128 \
  --mlp_ratio 4 \
  --model_dropout 0.1 \
  --noise_type loglinear \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 1 \
  --log_every 100 \
  --eval_num_samples 64 \
  --eval_num_steps 81 \
  --sampler confidence \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 2 \
  --eval_save_format mp4 \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_discrete_diffusion.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_discrete_diffusion.py $COMMON_ARGS
fi
