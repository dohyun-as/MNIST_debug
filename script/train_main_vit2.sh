#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=4,5,6,7  
NUM_GPUS=4

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/outputs_encoder/vit_mae075_patch_fsq888_token_avg"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# SRM_CONFIG="./config/sudoku_config.json"
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 50000 \
#   --batch_size 8 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5000 \
#   --eval_every 1500 \
#   --log_every 100 \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 4 \
#   --guidance_scale 1.0 \
#   --pad_image_size 288 \
#   --uncond_drop_prob 0 \
#   --image_conditioning \
#   --concat_conditioning \
#   --concat_downsample_factor 32 \
#   --cond_dim 4 \
#   --eval_num_steps 50 \
#   --prediction_type sample \
#   --patch_conditioning \
#   --patch_grid_size 9 \
#   --encoder_type vit \
#   --vit_patch_size 4 \
#   --vit_depth 4 \
#   --vit_num_heads 4 \
#   --vit_mlp_ratio 4.0 \
#   --mae_patch_mask_ratio 0.75 \
#   --mae_cell_mask_ratio 0.0 \
#   --use_fsq \
#   --fsq_levels 8 8 8 \
#   --use_averaged_features \
#   --resume_dir "./outputs/outputs_encoder/vit_mae075_patch_fsq888_token_avg/ckpt/step20000" \
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi




#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=4,5,6,7  
NUM_GPUS=4

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="./data"
OUTPUT_DIR="./outputs/outputs_encoder/vit_mae075_patch_fsq888_token_avg_t_buckets4"
UNET_CONFIG="./config/unet_mnist_concat.json"
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --sudoku_config "${SRM_CONFIG}" \
  --classifier_pth "${CLASSIFIER}" \
  --max_train_steps 50000 \
  --batch_size 8 \
  --lr 2e-5 \
  --num_train_timesteps 1000 \
  --beta_start 2e-5 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 5000 \
  --eval_every 1500 \
  --log_every 100 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 4 \
  --guidance_scale 1.0 \
  --pad_image_size 288 \
  --uncond_drop_prob 0 \
  --image_conditioning \
  --concat_conditioning \
  --concat_downsample_factor 32 \
  --cond_dim 4 \
  --eval_num_steps 50 \
  --prediction_type sample \
  --patch_conditioning \
  --patch_grid_size 9 \
  --encoder_type vit \
  --vit_patch_size 4 \
  --vit_depth 4 \
  --vit_num_heads 4 \
  --vit_mlp_ratio 4.0 \
  --mae_patch_mask_ratio 0.75 \
  --mae_cell_mask_ratio 0.0 \
  --use_fsq \
  --fsq_levels 8 8 8 \
  --use_averaged_features \
  --num_timestep_buckets 4 \
"

  # --resume_dir "./outputs/outputs_encoder/vit_mae075_patch_fsq888_token_avg/ckpt/step20000" \

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
fi




