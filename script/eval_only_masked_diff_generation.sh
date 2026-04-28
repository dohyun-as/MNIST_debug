#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  Eval-only pass for the CLEVR masked-diffusion run:
#    runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0
#
#  Matches training's 90-sample eval set (30 per split × 3 splits) and
#  writes ONLY eval scores — NO generation-order heatmaps, decoded PNGs,
#  level-ablation grids, sampler-compare figures, or overlays.
#
#  Produces:
#    ${OUTPUT_SUBDIR}/eval_scores.json   (per-sample raw)
#    ${OUTPUT_SUBDIR}/eval_summary.json  (per-sampler × cfg × schedule)
#    ${OUTPUT_SUBDIR}/meta.json
#
#  Usage:
#    bash script/eval_only_masked_diff_generation.sh
#    CFG_VALUES="1.0 3.0 5.0" CFG_SCHEDULES="linear constant" \
#      bash script/eval_only_masked_diff_generation.sh
# ──────────────────────────────────────────────────────────────────

set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

RUN_DIR=${RUN_DIR:-"./runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0"}
STEP=${STEP:-50000}
VARIANT=${VARIANT:-"ema"}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-"eval_only"}

# Training default is --eval_num_samples 30 (per split). viz_samples_per_split=0
# disables ALL visualization saves — only eval scores are produced.
SAMPLES_PER_SPLIT=${SAMPLES_PER_SPLIT:-30}
VIZ_SAMPLES_PER_SPLIT=${VIZ_SAMPLES_PER_SPLIT:-0}
NUM_DECODE_SEEDS=${NUM_DECODE_SEEDS:-3}
SPLITS=${SPLITS:-"easy medium hard"}
SAMPLERS=${SAMPLERS:-"ddpm_cache confidence_top1 confidence_cosine"}
DECODE_ABLATION_SAMPLER=${DECODE_ABLATION_SAMPLER:-"ddpm_cache"}

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
  --viz_samples_per_split ${VIZ_SAMPLES_PER_SPLIT} \
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
