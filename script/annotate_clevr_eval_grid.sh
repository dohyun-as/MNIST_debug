#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  annotate_clevr_eval_grid.sh
#  Re-runs the CLEVR detector+classifier on the saved eval grid PNG
#  and produces:
#    - per-sample annotated PNGs (GT|GEN side-by-side, boxes + caption)
#    - all_annotated.png (one big stack of all samples)
#    - annotations.json (per-sample machine-readable record)
#
#  By default annotates both runs at step 200000:
#    - ours (discrete diffusion + diff head, 3 samplers)
#    - naive T2I baseline
#
#  Multi-GPU: by default uses GPUs ${GPUS:-0,1,2,3} in parallel — each
#  process handles a shard (round-robin over samples), and a final
#  merge step rebuilds annotations.json + all_annotated.png.
#
#  Usage:
#    bash script/annotate_clevr_eval_grid.sh
#    GPUS=0,1 STEP=150000 bash script/annotate_clevr_eval_grid.sh
#    OURS_ONLY=1 bash script/annotate_clevr_eval_grid.sh
#    NAIVE_ONLY=1 bash script/annotate_clevr_eval_grid.sh
# ──────────────────────────────────────────────────────────────────
set -e

STEP=${STEP:-200000}
OURS_DIR=${OURS_DIR:-"runs/clevr/masked_diff/ours_text_diffhead_clip_dit_vit_flow_cont_out16_only8x8_tokdrop1.0"}
NAIVE_DIR=${NAIVE_DIR:-"runs/clevr/naive_dit_256_text_cond_clip"}
SAMPLERS=${SAMPLERS:-"confidence_top1 ddpm_cache confidence_cosine"}

GPUS=${GPUS:-"0,1,2,3"}
IFS=',' read -ra GPU_LIST <<< "$GPUS"
N_GPUS=${#GPU_LIST[@]}

step_padded=$(printf "%07d" "$STEP")

run_one() {
  # $1: grid_png, $2: meta or "--derive_naive_t2i", $3: output_dir
  local grid="$1"
  local cond_arg="$2"
  local out="$3"

  mkdir -p "$out"
  echo "=== ${out}  (${N_GPUS} shards)"

  if [ "${N_GPUS}" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python src/annotate_clevr_eval_grid.py \
      --grid_png "$grid" $cond_arg --output_dir "$out"
    return
  fi

  # Launch one process per GPU, each owning a shard (i % N_GPUS == shard_idx)
  pids=()
  for ((s=0; s<N_GPUS; s++)); do
    gpu="${GPU_LIST[$s]}"
    log="${out}/shard${s}.log"
    CUDA_VISIBLE_DEVICES=$gpu python src/annotate_clevr_eval_grid.py \
      --grid_png "$grid" $cond_arg --output_dir "$out" \
      --shard_idx $s --num_shards $N_GPUS > "$log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  # Merge shard JSONs + rebuild all_annotated.png
  python src/annotate_clevr_eval_grid.py --merge \
    --grid_png "$grid" --output_dir "$out"
}

if [ -z "${NAIVE_ONLY}" ]; then
  meta="${OURS_DIR}/eval_samples/step_${step_padded}_clevr_meta.json"
  if [ ! -f "${meta}" ]; then
    echo "[skip ours] no meta at ${meta}" >&2
  else
    for samp in ${SAMPLERS}; do
      grid="${OURS_DIR}/eval_samples/step_${step_padded}_clevr_${samp}.png"
      if [ ! -f "${grid}" ]; then
        echo "[skip ours/${samp}] no grid at ${grid}" >&2
        continue
      fi
      out="${OURS_DIR}/eval_samples/annotated/step_${step_padded}_${samp}"
      run_one "$grid" "--meta_json ${meta}" "$out"
    done
  fi
fi

if [ -z "${OURS_ONLY}" ]; then
  ngrid="${NAIVE_DIR}/clevr_eval/clevr_eval_step${step_padded}.png"
  nmeta="${NAIVE_DIR}/clevr_eval/clevr_eval_step${step_padded}_meta.json"
  if [ ! -f "${ngrid}" ]; then
    echo "[skip naive] no grid at ${ngrid}" >&2
  else
    nout="${NAIVE_DIR}/clevr_eval/annotated/step_${step_padded}"
    if [ -f "${nmeta}" ]; then
      run_one "$ngrid" "--meta_json ${nmeta}" "$nout"
    else
      run_one "$ngrid" "--derive_naive_t2i" "$nout"
    fi
  fi
fi
