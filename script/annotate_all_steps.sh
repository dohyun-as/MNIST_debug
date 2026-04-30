#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  annotate_all_steps.sh
#  Run the annotator on EVERY saved eval grid PNG (i.e. 10k / 20k /
#  ... / 200k — every cadence at which the training script wrote a
#  grid). Each ckpt has 3 saved grids (one per sampler) for v2 runs
#  and 1 grid for the naive T2I run.
#
#  This re-evaluates each grid with the (corrected) detector +
#  classifier path and produces:
#    - per-(run, step, sampler) annotations.json
#    - per-(run, step, sampler) all_annotated.png + sample_*.png
#
#  Combine with the saved cond_eval.json sweep (eval_all_ckpts.sh)
#  via `python src/eval_all_ckpts.py --aggregate_only ...` to get a
#  full CSV trajectory.
#
#  Usage:
#    bash script/annotate_all_steps.sh
#    GPUS=0,1 bash script/annotate_all_steps.sh
# ──────────────────────────────────────────────────────────────────
set -e

GPUS=${GPUS:-"0,1,2,3"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"runs/eval_outputs"}

# (full_run_dir, short_name) pairs — short name is what shows up under
# OUTPUT_ROOT/annotated/<short>/
RUNS_FULL=(
  runs/clevr/masked_diff/ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0
  runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0
  runs/clevr/masked_diff/ours_text_diffhead_clip_out16_tokdrop1.0_semiar
  runs/clevr/slot_stage2/256_slot16_d64_resnet18s_crossattn_clip
  runs/clevr/naive_dit_256_text_cond_clip
)
RUNS_SHORT=(
  ours-only8x8
  ours-multilevel
  ours-multilevel-semiar
  slot-stage2
  naive-T2I
)

IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

run_one_grid() {
  local grid="$1" cond_arg="$2" out="$3"
  mkdir -p "$out"
  if [ "${N_GPUS}" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python src/annotate_clevr_eval_grid.py \
      --grid_png "$grid" $cond_arg --output_dir "$out"
    return
  fi
  pids=()
  for ((s=0; s<N_GPUS; s++)); do
    g="${GPU_LIST[$s]}"
    log="${out}/shard${s}.log"
    CUDA_VISIBLE_DEVICES=$g python src/annotate_clevr_eval_grid.py \
      --grid_png "$grid" $cond_arg --output_dir "$out" \
      --shard_idx $s --num_shards $N_GPUS > "$log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  python src/annotate_clevr_eval_grid.py --merge \
    --grid_png "$grid" --output_dir "$out"
}

for i in "${!RUNS_FULL[@]}"; do
  rd="${RUNS_FULL[$i]}"
  short="${RUNS_SHORT[$i]}"
  out_base="${OUTPUT_ROOT}/annotated/${short}"
  echo "================ ${rd}  →  ${out_base} ================"
  if [ -f "${rd}/run_config.json" ]; then
    eval_dir="${rd}/eval_samples"
    if [ ! -d "${eval_dir}" ]; then continue; fi
    for meta in "${eval_dir}"/step_*_clevr_meta.json; do
      [ -f "$meta" ] || continue
      step=$(basename "$meta" | sed -E 's/^step_0*([0-9]+)_clevr_meta\.json$/\1/')
      step_padded=$(printf "%07d" "$step")
      for samp in confidence_top1 confidence_cosine ddpm_cache; do
        grid="${eval_dir}/step_${step_padded}_clevr_${samp}.png"
        [ -f "$grid" ] || continue
        out="${out_base}/step_${step_padded}_${samp}"
        if [ -f "${out}/annotations.json" ]; then
          echo "[skip] ${out} already done"
          continue
        fi
        echo "[run]  ${out}"
        run_one_grid "$grid" "--meta_json ${meta}" "$out"
      done
    done
  elif [ -f "${rd}/args.json" ]; then
    eval_dir="${rd}/clevr_eval"
    if [ ! -d "${eval_dir}" ]; then continue; fi
    for grid in "${eval_dir}"/clevr_eval_step*.png; do
      [ -f "$grid" ] || continue
      step=$(basename "$grid" | sed -E 's/^clevr_eval_step0*([0-9]+)\.png$/\1/')
      step_padded=$(printf "%07d" "$step")
      meta="${eval_dir}/clevr_eval_step${step_padded}_meta.json"
      out="${out_base}/step_${step_padded}"
      if [ -f "${out}/annotations.json" ]; then
        echo "[skip] ${out} already done"
        continue
      fi
      echo "[run]  ${out}"
      if [ -f "$meta" ]; then
        run_one_grid "$grid" "--meta_json ${meta}" "$out"
      else
        run_one_grid "$grid" "--derive_naive_t2i" "$out"
      fi
    done
  fi
done

echo "[done] all eval grids annotated → ${OUTPUT_ROOT}/annotated/"
echo "[hint] aggregate trajectories with:"
echo "       python src/eval_all_ckpts.py --aggregate_only \\"
echo "         --run_dirs ${RUNS_FULL[@]} \\"
echo "         --output_root ${OUTPUT_ROOT}"
