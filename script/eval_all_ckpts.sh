#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  eval_all_ckpts.sh
#  Re-run the (corrected) condition eval on every saved ckpt of the
#  five comparison runs the user asked about, using ALL 4 GPUs in
#  parallel via the existing train scripts' --eval_only mode.
#
#  Hardcoded defaults (override via env vars):
#    GPUS                   = 0,1,2,3
#    NUM_PER_SPLIT          = 100      (→ 300 total per ckpt-sampler)
#    OUTPUT_CSV / OUTPUT_JSON
#    ANNOTATE_LAST          = 1        (also annotate the latest ckpt's grid)
#    REUSE_EXISTING         = 0        (set to 1 to skip ckpts that already
#                                       have a cond_eval JSON)
#  Usage:
#    bash script/eval_all_ckpts.sh
#    NUM_PER_SPLIT=200 bash script/eval_all_ckpts.sh
#    GPUS=4,5,6,7 bash script/eval_all_ckpts.sh
#    REUSE_EXISTING=1 bash script/eval_all_ckpts.sh
# ──────────────────────────────────────────────────────────────────
set -e

GPUS=${GPUS:-"0,1,2,3"}
NUM_PER_SPLIT=${NUM_PER_SPLIT:-100}
OUTPUT_ROOT=${OUTPUT_ROOT:-"runs/eval_outputs"}
OUTPUT_CSV=${OUTPUT_CSV:-"${OUTPUT_ROOT}/eval_sweep.csv"}
OUTPUT_JSON=${OUTPUT_JSON:-"${OUTPUT_ROOT}/eval_sweep.json"}
ANNOTATE_LAST=${ANNOTATE_LAST:-1}
REUSE_EXISTING=${REUSE_EXISTING:-0}
# Per-rank batch sizes for v2 sampling/decoding. Bump these if GPU util is
# low (the training defaults — sample=8, decode=4 — are tiny). On big
# GPUs (e.g. 80GB Blackwell) try 64/32 or 128/64.
EVAL_SAMPLE_BS=${EVAL_SAMPLE_BS:-32}
EVAL_DECODE_BS=${EVAL_DECODE_BS:-16}

RUNS=(
  runs/clevr/masked_diff/ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0
  runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0
  runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0_semiar
  runs/clevr/slot_stage2/256_slot16_d64_resnet18s_crossattn_clip
  runs/clevr/naive_dit_256_text_cond_clip
)

extra_args=()
if [ "${ANNOTATE_LAST}" -eq 1 ]; then
  extra_args+=(--annotate_last)
fi
if [ "${REUSE_EXISTING}" -eq 1 ]; then
  extra_args+=(--reuse_existing)
fi

# Inference-only semi-AR: take the listed multi-level run dirs, do an EXTRA
# eval pass forcing --semi_autoregressive on (model trained without it).
# Outputs go to '<run>_infer_semiar/' so original results aren't touched.
# Default: empty (skip).  Set INFER_SEMIAR_RUNS to a space-separated list
# of run dirs to enable.
INFER_SEMIAR_RUNS=${INFER_SEMIAR_RUNS:-""}
if [ -n "${INFER_SEMIAR_RUNS}" ]; then
  extra_args+=(--inference_semi_ar_for ${INFER_SEMIAR_RUNS})
fi

echo "[sweep] GPUs=${GPUS}  N/split=${NUM_PER_SPLIT}  runs=${#RUNS[@]}"
echo "[sweep] sample_bs=${EVAL_SAMPLE_BS}  decode_bs=${EVAL_DECODE_BS}"
echo "[sweep] CSV=${OUTPUT_CSV}  JSON=${OUTPUT_JSON}  annotate_last=${ANNOTATE_LAST}  reuse=${REUSE_EXISTING}"

python src/eval_all_ckpts.py \
  --run_dirs "${RUNS[@]}" \
  --num_samples_per_split "${NUM_PER_SPLIT}" \
  --gpus "${GPUS}" \
  --eval_sample_bs "${EVAL_SAMPLE_BS}" \
  --eval_decode_bs "${EVAL_DECODE_BS}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_CSV}" \
  --output_json "${OUTPUT_JSON}" \
  "${extra_args[@]}"

echo "[sweep] done"
echo "[sweep] CSV at: ${OUTPUT_CSV}"
echo "[sweep] JSON at: ${OUTPUT_JSON}"
