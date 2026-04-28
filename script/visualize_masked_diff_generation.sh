#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Visualize the CLEVR masked-diffusion run:
#    runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0
#
#  Produces under ${RUN_DIR}/visualize/:
#    (1) generation_order/    — per-token unmask-step heatmaps for the
#                                3 samplers (ddpm_cache, confidence_top1,
#                                confidence_cosine), split into the 4
#                                level grids (1×1, 2×2, 4×4, 8×8)
#    (2) level_ablation/      — same generated tokens, decoded by the
#                                pretrained DiT while varying (a) how
#                                many coarse levels are used (1→4) and
#                                (b) the decoder noise seed (3 seeds)
#    (3) final_grid_overlay/  — the completed 256×256 image with a red
#                                8×8 grid overlay so we can eyeball what
#                                the finest token scale covers
#    (4) meta.json            — split / index / condition text per sample
#
#  Defaults: 4 GPUs, 5 samples/split × 3 splits = 15 samples, sharded
#  round-robin across ranks.
#
#  Usage:
#    bash script/visualize_masked_diff_generation.sh
#    GPUS=0,1 bash script/visualize_masked_diff_generation.sh
#    STEP=100000 VARIANT=base bash script/visualize_masked_diff_generation.sh
#    RUN_DIR=... bash script/visualize_masked_diff_generation.sh
# ──────────────────────────────────────────────────────────────────

set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

RUN_DIR=${RUN_DIR:-"./runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0"}
STEP=${STEP:-50000}
VARIANT=${VARIANT:-"ema"}

SAMPLES_PER_SPLIT=${SAMPLES_PER_SPLIT:-5}
# How many of those per-split samples ALSO get saved visualizations. Unset
# (default) = same as SAMPLES_PER_SPLIT (visualize all). Set small to match
# training's 30-per-split eval while only eyeballing a few.
VIZ_SAMPLES_PER_SPLIT=${VIZ_SAMPLES_PER_SPLIT:-""}
NUM_DECODE_SEEDS=${NUM_DECODE_SEEDS:-3}
SPLITS=${SPLITS:-"easy medium hard"}
SAMPLERS=${SAMPLERS:-"ddpm_cache confidence_top1 confidence_cosine"}
DECODE_ABLATION_SAMPLER=${DECODE_ABLATION_SAMPLER:-"ddpm_cache"}

# CFG sweep. Leave unset to fall back to the single (cfg, schedule) stored in
# run_config.json. Set both to explore the grid.
#   Example: CFG_VALUES="1.0 3.0 5.0" CFG_SCHEDULES="linear constant"
CFG_VALUES=${CFG_VALUES:-""}
CFG_SCHEDULES=${CFG_SCHEDULES:-""}
DECODE_ABLATION_CFG=${DECODE_ABLATION_CFG:-""}
DECODE_ABLATION_SCHEDULE=${DECODE_ABLATION_SCHEDULE:-""}

EVAL_NUM_STEPS=${EVAL_NUM_STEPS:-128}
DECODE_NUM_STEPS=${DECODE_NUM_STEPS:-50}
# Batching — big speed win. Lower these if you hit OOM.
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-8}
DECODE_BATCH_SIZE=${DECODE_BATCH_SIZE:-8}
IMAGE_SIZE=${IMAGE_SIZE:-256}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}
SEED=${SEED:-42}

VAL_IMAGE_ROOT=${VAL_IMAGE_ROOT:-"../clevr-dataset-gen/output/clevr_256_varied_val/images"}
VAL_COND_DIR=${VAL_COND_DIR:-"../clevr-dataset-gen/output/clevr_256_varied_val/conditions_text"}

COMMON_ARGS="--run_dir ${RUN_DIR} \
  --step ${STEP} \
  --variant ${VARIANT} \
  --splits ${SPLITS} \
  --samples_per_split ${SAMPLES_PER_SPLIT} \
  --num_decode_seeds ${NUM_DECODE_SEEDS} \
  --samplers ${SAMPLERS} \
  --decode_ablation_sampler ${DECODE_ABLATION_SAMPLER} \
  --val_image_root ${VAL_IMAGE_ROOT} \
  --val_cond_dir ${VAL_COND_DIR} \
  --eval_num_steps ${EVAL_NUM_STEPS} \
  --decode_num_steps ${DECODE_NUM_STEPS} \
  --gen_batch_size ${GEN_BATCH_SIZE} \
  --decode_batch_size ${DECODE_BATCH_SIZE} \
  --image_size ${IMAGE_SIZE} \
  --mixed_precision ${MIXED_PRECISION} \
  --seed ${SEED}"

if [ -n "${VIZ_SAMPLES_PER_SPLIT}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --viz_samples_per_split ${VIZ_SAMPLES_PER_SPLIT}"
fi
if [ -n "${CFG_VALUES}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --cfg_values ${CFG_VALUES}"
fi
if [ -n "${CFG_SCHEDULES}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --cfg_schedules ${CFG_SCHEDULES}"
fi
if [ -n "${DECODE_ABLATION_CFG}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --decode_ablation_cfg ${DECODE_ABLATION_CFG}"
fi
if [ -n "${DECODE_ABLATION_SCHEDULE}" ]; then
  COMMON_ARGS="${COMMON_ARGS} --decode_ablation_schedule ${DECODE_ABLATION_SCHEDULE}"
fi

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
if [ ${NUM_GPUS} -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT --multi_gpu --num_processes ${NUM_GPUS} \
    src/visualize_masked_diff_generation.py $COMMON_ARGS
else
  CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
    --main_process_port $PORT \
    src/visualize_masked_diff_generation.py $COMMON_ARGS
fi
