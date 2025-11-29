#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=5 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="./data"
OUTPUT_DIR="./outputs/mnist_diffusion_larger_unet_pad288_vae_2digit"
UNET_CONFIG="./config/unet_mnist_32_ch2.json"
VAE_CONFIG="./config/vae.json" 
SRM_CONFIG="./config/sudoku_config_2digit.yaml" 

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --vae_config ${VAE_CONFIG} \
  --sudoku_config ${SRM_CONFIG} \
  --max_train_steps 50000 \
  --batch_size 100 \
  --lr 2e-5 \
  --num_train_timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 50000 \
  --eval_every 5000 \
  --log_every 100 \
  --seed 42 \
  --mixed_precision fp16 \
  --log_with tensorboard \
  --pad_image_size 288 \
  --grad_accum_steps 1 \
  --vae_test
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src/main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src/main.py $COMMON_ARGS
fi

