#!/bin/bash

# GPU 설정 원하면
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/mnist_diffusion_unet32"
UNET_CONFIG="${PROJECT_ROOT}/configs/unet_mnist_32.json"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=4,5,6,7 #$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=4 #$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

COMMON_ARGS="--data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --unet_config "${UNET_CONFIG}" \
  --max_train_steps 10000 \
  --batch_size 128 \
  --lr 2e-4 \
  --num_train_timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 0.02 \
  --beta_schedule linear \
  --save_every 5 \
  --log_every 100 \
  --seed 42
"

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} src.main.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch src.main.py $COMMON_ARGS
fi




# python -m mnist_diffusion.train_mnist_diffusion \
#   --data_dir "${DATA_DIR}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --unet_config "${UNET_CONFIG}" \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 2e-4 \
#   --num_train_timesteps 1000 \
#   --beta_start 1e-4 \
#   --beta_end 0.02 \
#   --beta_schedule linear \
#   --save_every 5 \
#   --log_every 100 \
#   --seed 42