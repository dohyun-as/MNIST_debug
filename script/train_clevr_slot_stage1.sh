#!/bin/bash
# ──────────────────────────────────────────────────────────────────
#  CLEVR 256×256 — STAGE 1: Slot Attention encoder + Baseline-1D DiT
#
#  Encoder: SlotAttentionEncoder (SlotDiffusion-port)
#    - ResNet18 (small_inputs=True, GroupNorm)
#    - SoftPositionEmbed → MLP → SlotAttention (3 iterations)
#    - K=16 slots × 192 dim   (matches SlotDiffusion CLEVRTex config)
#  DiT decoder: same JiT-B/16 backbone as the existing baseline script
#    (cross-attends image patches → slots, flow matching loss)
#
#  Reference (existing): script/train_clevr_dit_baseline_continuous.sh
#    The semanticist 85-causal-slot encoder is replaced with real Slot
#    Attention via main_slot_stage1.py monkey-patch — no edits to
#    main_multires.py.
#
#  Eval: identical to existing baseline (sample grid, recon eval, CLEVR
#        detector/classifier eval). PLUS slot-attention segmentation
#        viz saved to <output_dir>/slot_viz/step_*.png at every
#        --slot_viz_every steps.
#
#  Usage:
#    bash script/train_clevr_slot_stage1.sh
#    GPUS=0,1 BATCH_PER_GPU=16 bash script/train_clevr_slot_stage1.sh
# ──────────────────────────────────────────────────────────────────

set -e

# ── GPU / batch ──
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

BATCH_PER_GPU=${BATCH_PER_GPU:-256}
GRAD_ACCUM=${GRAD_ACCUM:-1}

echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS)  batch/gpu=$BATCH_PER_GPU  accum=$GRAD_ACCUM  effective=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))"

# ── Data ──
CLEVR_DIR=${CLEVR_DIR:-"../clevr_output/clevr_256_varied/images"}
CLEVR_VAL=${CLEVR_VAL:-"../clevr_output/clevr_256_varied_val/images"}

OUTPUT_DIR=${OUTPUT_DIR:-"./runs/clevr/slot_stage1/256_slot16_d192"}
mkdir -p "${OUTPUT_DIR}"

accelerate launch \
    --num_processes $NUM_GPUS \
    --multi_gpu \
    src/main_slot_stage1.py \
    --backbone baseline_1d \
    --output_dir ${OUTPUT_DIR} \
    --train_dir "$CLEVR_DIR" \
    --val_dir "$CLEVR_VAL" \
    --dataset_root "$CLEVR_DIR" \
    --image_size 256 \
    --in_channels 3 \
    --vae_downsample_factor 1 \
    \
    `# ── slot encoder hyperparams (match SlotDiffusion CLEVRTex) ── ` \
    --num_slots 16 \
    --slot_dim 192 \
    --slot_iters 3 \
    --slot_mlp_size 384 \
    --slot_enc_backbone vit_b16 \
    --slot_init learned \
    --slot_viz_every 5000 \
    --slot_viz_n_samples 8 \
    \
    `# ── unused (Baseline1DConditionalDiT defaults; encoder is replaced) ── ` \
    --enc_embed_dim 768 \
    --enc_depth 12 \
    --enc_num_heads 12 \
    --enc_drop_path_rate 0.1 \
    \
    `# ── DiT decoder (same as baseline) ── ` \
    --dit_patch_size 16 \
    --dit_hidden_size 768 \
    --dit_n_heads 12 \
    --dit_n_blocks 12 \
    --dit_mlp_ratio 4.0 \
    --dit_dropout 0.0 \
    --dit_bottleneck_dim 128 \
    --dit_in_context_len 32 \
    --dit_in_context_start 4 \
    \
    `# ── flow matching ── ` \
    --use_flow_matching \
    --flow_P_mean -0.8 \
    --flow_P_std 0.8 \
    --flow_t_eps 0.05 \
    --flow_noise_scale 1.0 \
    --flow_sampling_method euler \
    \
    `# ── training ── ` \
    --max_train_steps 200000 \
    --batch_size $BATCH_PER_GPU \
    --blr 1e-4 \
    --weight_decay 0.0 \
    --warmup_steps 5000 \
    --max_grad_norm 1.0 \
    --grad_accum_steps $GRAD_ACCUM \
    --mixed_precision bf16 \
    --ema_decay 0 \
    --uncond_drop_prob 0.1 \
    --guidance_scale 3.0 \
    \
    `# ── logging / eval ── ` \
    --log_every 100 \
    --save_every 10000 \
    --sample_every 5000 \
    --fid_every 9999999 \
    --eval_num_steps 50 \
    --num_workers 8 \
    --seed 42 \
    --mae_mask_ratio 0.0 \
    --clevr_eval_every 5000 \
    --clevr_eval_samples 50
