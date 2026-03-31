#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_LAUNCH_BLOCKING=1


CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SRM_CONFIG="./config/sudoku_config.json"
# OUTPUT_DIR="./outputs_discrete_diffusion/confidence"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--output_dir "${OUTPUT_DIR}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --grid_only \
#   --max_train_steps 100000 \
#   --batch_size 2048 \
#   --lr 3e-4 \
#   --weight_decay 0.01 \
#   --warmup_steps 1000 \
#   --hidden_size 256 \
#   --n_heads 8 \
#   --n_blocks 6 \
#   --cond_dim 128 \
#   --mlp_ratio 4 \
#   --model_dropout 0.1 \
#   --noise_type loglinear \
#   --ema_decay 0 \
#   --save_every 10000 \
#   --eval_every 1000 \
#   --log_every 100 \
#   --eval_num_samples 2048 \
#   --eval_num_steps 81 \
#   --eval_gif_samples 4 \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 1 \
#   --sampler confidence \
# "
#   # --tokens_per_step 1 \
#   # --sampler ddpm_cache \
#   # --resume_dir "/workspace/NAS/project/MNIST_debug/outputs_discrete_diffusion/confidence/ckpt/step10000" \

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_discrete_diffusion.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_discrete_diffusion.py $COMMON_ARGS
# fi



PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRM_CONFIG="./config/sudoku_config.json"
OUTPUT_DIR="./outputs_discrete_diffusion/base_model"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--output_dir "${OUTPUT_DIR}" \
  --sudoku_config "${SRM_CONFIG}" \
  --grid_only \
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
  --noise_type loglinear \
  --ema_decay 0 \
  --save_every 10000 \
  --eval_every 3000 \
  --log_every 100 \
  --eval_num_samples 2048 \
  --eval_num_steps 81 \
  --eval_gif_samples 16 \
  --eval_save_format mp4 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --sampler confidence \
  --tokens_per_step 1 \
"
  # --sampler ddpm_cache \
  # --resume_dir "/workspace/NAS/project/MNIST_debug/outputs_discrete_diffusion/confidence_1token_2dpos/ckpt/step40000" \

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_discrete_diffusion.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_discrete_diffusion.py $COMMON_ARGS
fi




# ──────────────────────────────────────────────────────────
#  image_cond_mode: image encoder tok_ids를 x0로 학습
#  tok_ids를 startup 시 caching해서 빠르게 학습
# ──────────────────────────────────────────────────────────

# CUDA_VISIBLE_DEVICES=0
# NUM_GPUS=1

# SRM_CONFIG="./config/sudoku_config.json"
# OUTPUT_DIR="./outputs_discrete_diffusion/image_cond"
# COND_UNET_CKPT="./outputs/concat_pixel_sample_fsq8888/ckpt/step40000"
# COND_UNET_CONFIG="./config/unet_mnist_concat.json"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
#   --sudoku_config ${SRM_CONFIG} \
#   --image_cond_mode \
#   --cond_unet_ckpt ${COND_UNET_CKPT} \
#   --cond_unet_config ${COND_UNET_CONFIG} \
#   --cond_image_size 288 \
#   --cond_feat_channels 128 \
#   --cond_discretizer_type fsq \
#   --cond_fsq_levels 8 8 8 8 \
#   --cond_concat_downsample_factor 32 \
#   --cond_patch_conditioning \
#   --cond_patch_grid_size 9 \
#   --cond_eval_ddim_steps 50 \
#   --max_train_steps 100000 \
#   --batch_size 2048 \
#   --lr 3e-4 \
#   --weight_decay 0.01 \
#   --warmup_steps 1000 \
#   --hidden_size 256 \
#   --n_heads 8 \
#   --n_blocks 6 \
#   --cond_dim 128 \
#   --mlp_ratio 4 \
#   --model_dropout 0.1 \
#   --noise_type loglinear \
#   --ema_decay 0 \
#   --save_every 10000 \
#   --eval_every 5000 \
#   --log_every 100 \
#   --eval_num_samples 64 \
#   --eval_num_steps 81 \
#   --sampler confidence \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 1 \
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_discrete_diffusion.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_discrete_diffusion.py $COMMON_ARGS
# fi
