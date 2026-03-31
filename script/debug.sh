#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1   # 이미 쓰는 중이면 유지
export NCCL_ASYNC_ERROR_HANDLING=1


export CUDA_LAUNCH_BLOCKING=1


CUDA_VISIBLE_DEVICES=0 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=1 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="./data"
OUTPUT_DIR="./outputs/debug"
UNET_CONFIG="./config/unet_mnist_concat.json"
VAE_CONFIG="./config/vae_sudoku_full.json" 
SRM_CONFIG="./config/sudoku_config.json" 
CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --sudoku_config "${SRM_CONFIG}" \
  --classifier_pth "${CLASSIFIER}" \
  --max_train_steps 50000 \
  --batch_size 1 \
  --lr 2e-5 \
  --num_train_timesteps 1000 \
  --beta_start 2e-5 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 10000 \
  --eval_every 1 \
  --log_every 100 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --grad_accum_steps 1 \
  --guidance_scale 1.0 \
  --pad_image_size 288 \
  --uncond_drop_prob 0 \
  --image_conditioning \
  --concat_conditioning \
  --concat_downsample_factor 32 \
  --cond_dim 128 \
  --eval_num_steps 50 \
  --use_fsq \
  --fsq_levels 8 8 8 \
  --prediction_type sample \
  --resume_dir "/workspace/NAS/project/MNIST_debug/outputs/concat_pixel_sample_fsq888/ckpt/step40000"
  --patch_conditioning \
  --patch_grid_size 9 \
"
  # --vae_test \
  # --vae_config ${VAE_CONFIG} \

  
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
fi





# #!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1   # 이미 쓰는 중이면 유지
# export NCCL_ASYNC_ERROR_HANDLING=1


# export CUDA_LAUNCH_BLOCKING=1


# CUDA_VISIBLE_DEVICES=0,1,2,3 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
# NUM_GPUS=4 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# DATA_DIR="./data"
# OUTPUT_DIR="./outputs/concat_pixel_sample_mnist_fsq33"
# UNET_CONFIG="./config/unet_mnist_concat.json"
# VAE_CONFIG="./config/vae_sudoku_full.json" 
# SRM_CONFIG="./config/sudoku_config_center_crop.json" 
# CLASSIFIER="/workspace/NAS/project/MNIST_debug/datasets/mnist_sudoku/mnist_classifier.pth"

# mkdir -p "${OUTPUT_DIR}"

# COMMON_ARGS="--data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --sudoku_config "${SRM_CONFIG}" \
#   --classifier_pth "${CLASSIFIER}" \
#   --max_train_steps 120000 \
#   --batch_size 32 \
#   --lr 2e-5 \
#   --num_train_timesteps 1000 \
#   --beta_start 2e-5 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 10000 \
#   --eval_every 1500 \
#   --log_every 100 \
#   --seed 42 \
#   --mixed_precision fp16 \
#   --log_with tensorboard \
#   --grad_accum_steps 1 \
#   --guidance_scale 1.0 \
#   --pad_image_size 32 \
#   --uncond_drop_prob 0 \
#   --image_conditioning \
#   --concat_conditioning \
#   --concat_downsample_factor 32 \
#   --cond_dim 4 \
#   --eval_num_steps 50 \
#   --use_fsq \
#   --fsq_levels 3 3 \
#   --prediction_type sample \
# "
#   # --resume_dir "/workspace/NAS/project/MNIST_debug/outputs/concat_pixel_sample_fsq33/ckpt/step60000"
#   # --vae_test \
#   # --vae_config ${VAE_CONFIG} \
#   # --patch_conditioning \
#   # --patch_grid_size 9 \

  
# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# if [ ${NUM_GPUS} -gt 1 ]; then
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
# else
#   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
# fi
