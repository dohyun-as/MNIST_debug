#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Visualize-only pass for the CLEVR masked-diffusion run:
#    runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0
#
#  NO detector/classifier eval — purely generates tokens, decodes to
#  images, and saves:
#    - generation_order/ (per-sample unmask-step heatmaps)
#    - level_ablation/   (keep_levels K=1..4 × num_decode_seeds grids)
#    - final_grid_overlay/ (8x8 overlays, final image)
#  Eval-score captions simply say "(no eval)".
#
#  For score-only (training's 90-sample set, no PNGs), use the paired
#  script: eval_only_masked_diff_generation.sh.
#
#  Usage:
#    bash script/visualize_only_masked_diff_generation.sh
#    GPUS=0,1 SAMPLES_PER_SPLIT=5 \
#      CFG_VALUES="1.0 3.0 5.0" CFG_SCHEDULES="linear constant" \
#      bash script/visualize_only_masked_diff_generation.sh
# ──────────────────────────────────────────────────────────────────

set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

RUN_DIR=${RUN_DIR:-"./runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0"}
STEP=${STEP:-50000}
VARIANT=${VARIANT:-"ema"}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-"visualize_only"}

# In viz-only mode, default to visualizing ALL selected samples.
SAMPLES_PER_SPLIT=${SAMPLES_PER_SPLIT:-5}
VIZ_SAMPLES_PER_SPLIT=${VIZ_SAMPLES_PER_SPLIT:-""}  # unset = same as SAMPLES_PER_SPLIT
NUM_DECODE_SEEDS=${NUM_DECODE_SEEDS:-3}
SPLITS=${SPLITS:-"easy medium hard"}
SAMPLERS=${SAMPLERS:-"ddpm_cache confidence_top1 confidence_cosine"}
DECODE_ABLATION_SAMPLER=${DECODE_ABLATION_SAMPLER:-"ddpm_cache"}

# CFG sweep (unset = single value from run_config.json).
CFG_VALUES=${CFG_VALUES:-""}
CFG_SCHEDULES=${CFG_SCHEDULES:-""}
DECODE_ABLATION_CFG=${DECODE_ABLATION_CFG:-""}
DECODE_ABLATION_SCHEDULE=${DECODE_ABLATION_SCHEDULE:-""}

EVAL_NUM_STEPS=${EVAL_NUM_STEPS:-128}
DECODE_NUM_STEPS=${DECODE_NUM_STEPS:-50}
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
  --output_subdir ${OUTPUT_SUBDIR} \
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
  --seed ${SEED} \
  --no_run_eval"

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
