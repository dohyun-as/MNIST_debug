#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_patch_wo_fsq"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# SRM_CONFIG="./config/sudoku_config.json"
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 30000 \
#   --batch_size 8 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5000 \
#   --eval_every 1000 \
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
# "
#   # --use_fsq \
#   # --fsq_levels 3 3 \

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi



# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SRM_CONFIG="./config/sudoku_config.json"
# OUTPUT_DIR="./outputs/outputs_discrete_diffusion/image_cond"
# TOKEN_CACHE_DIR="./outputs/outputs_discrete_diffusion/image_cond/token_cache"
# COND_UNET_CKPT="./outputs/outputs_encoder/concat_pixel_sample_fsq888/ckpt/step40000"
# COND_UNET_CONFIG="./config/unet_mnist_concat.json"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--output_dir ${OUTPUT_DIR} \
#   --sudoku_config ${SRM_CONFIG} \
#   --image_cond_mode \
#   --token_cache_dir ${TOKEN_CACHE_DIR} \
#   --cond_unet_ckpt ${COND_UNET_CKPT} \
#   --cond_unet_config ${COND_UNET_CONFIG} \
#   --cond_image_size 288 \
#   --cond_feat_channels 128 \
#   --cond_discretizer_type fsq \
#   --cond_fsq_levels 8 8 8 \
#   --cond_concat_downsample_factor 32 \
#   --cond_patch_conditioning \
#   --cond_patch_grid_size 9 \
#   --cond_eval_ddim_steps 50 \
#   --eval_render_batch_size 16 \
#   --init_embed_from_fsq \
#   --max_train_steps 50000 \
#   --batch_size 512 \
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
#   --eval_every 3000 \
#   --log_every 100 \
#   --eval_num_samples 64 \
#   --eval_num_steps 81 \
#   --sampler confidence \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 1 \
#   --eval_save_format mp4 \
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/train_discrete_diffusion.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/train_discrete_diffusion.py $COMMON_ARGS
# fi



#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3 
NUM_GPUS=4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="./data"
OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_patch05_fsq888_norm"
UNET_CONFIG="./config/unet_mnist_concat.json"
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --sudoku_config "${SRM_CONFIG}" \
  --classifier_pth "${CLASSIFIER}" \
  --max_train_steps 20000 \
  --batch_size 8 \
  --lr 2e-5 \
  --num_train_timesteps 1000 \
  --beta_start 2e-5 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 5000 \
  --eval_every 1000 \
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
  --mae_patch_mask_ratio 0.5 \
  --mae_cell_mask_ratio 0.0 \
  --use_fsq \
  --fsq_levels 8 8 8 \
  --normalize_concat \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
fi

DATA_DIR="./data"
OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_patch025_fsq888_norm"
UNET_CONFIG="./config/unet_mnist_concat.json"
SRM_CONFIG="./config/sudoku_config.json"
CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --sudoku_config "${SRM_CONFIG}" \
  --classifier_pth "${CLASSIFIER}" \
  --max_train_steps 20000 \
  --batch_size 8 \
  --lr 2e-5 \
  --num_train_timesteps 1000 \
  --beta_start 2e-5 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 5000 \
  --eval_every 1000 \
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
  --mae_patch_mask_ratio 0.25 \
  --mae_cell_mask_ratio 0.0 \
  --use_fsq \
  --fsq_levels 8 8 8 \
  --normalize_concat \
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
fi

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_patch_fsq888"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# SRM_CONFIG="./config/sudoku_config.json"
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 20000 \
#   --batch_size 8 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5000 \
#   --eval_every 1000 \
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
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi

# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_mask_patch_vq9"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# SRM_CONFIG="./config/sudoku_config.json"
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 20000 \
#   --batch_size 8 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5000 \
#   --eval_every 1000 \
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
#   --use_vq_discretizer \
# "
#   # --use_fsq \
#   # --fsq_levels 8 8 8 \

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi


# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/outputs_encoder/vit_mae_cellmask_patch_fsq888"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# SRM_CONFIG="./config/sudoku_config.json"
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 20000 \
#   --batch_size 8 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5000 \
#   --eval_every 1000 \
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
#   --mae_cell_mask_ratio 0.2 \
#   --use_fsq \
#   --fsq_levels 8 8 8 \
# "

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi


