#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# place_cube_right sample-efficiency sweep: one α fine-tune per N, as in
# lingbot-va's yam_place_cube_right_joint_n{N}_train configs.
#
# Only the episode subset differs between runs (configs/dataloader/
# yam_place_cube_right_n<N>.yaml: the first N/14 demos of every cube position).
# Every N gets the same 512 optimizer steps at global batch 256 (16 per GPU x
# 2 GPUs x 8 accumulation; the n56 10-epoch budget) and a constant LR after a
# 5% warmup, saving at 1/4, 1/2, 3/4 and the end (micro-steps 1024..4096).
# With a constant LR the step-S checkpoint is what an S-step run would give,
# so the runs form an (N, step) grid; the diagonal n14@1024, n28@2048,
# n42@3072, n56@4096 gives every demo equal exposure.
#
# DELTA=1 trains the arm joints as delta actions (openpi's DeltaActions:
# action - chunk-start state, grippers absolute; see delta_action_dims in
# configs/dataloader/yam.yaml) under run names place_cube_right_n<N>_delta.
# Deploy adds the state back, so evaluation is unchanged.
#
# LORA=1 fine-tunes with LoRA instead of in full (training.lora in configs/train.yaml):
# rank-128 adapters on the 5B video DiT at LORA_LR (default 5e-4), while the action
# expert, the proprio encoder and the DiT's head / patch embedding / biases / norms
# train in full at the same 1e-4. Prompt embeddings are cached and the frozen text
# encoder stays on CPU. It fits 46 GB GPUs (L40S) and defaults to 4 of them; the
# per-GPU batch is 32 / NPROC_PER_NODE, so the global batch stays 256 and every step
# count above is unchanged. Checkpoints are saved merged: deploy and open-loop eval
# treat them like any other. Run names get a _lora suffix.
#
# Checkpoints: outputs/openwam_checkpoints/place_cube_right_n<N>[_delta][_lora]/<launch time>/
# wandb run:   place_cube_right_n<N>[_delta][_lora]_<launch time>
#
# Usage:
#   bash scripts/train_place_cube_right_sweep.sh               # n14 n28 n42 n56, in turn
#   bash scripts/train_place_cube_right_sweep.sh 14 28 42      # some of them
#   bash scripts/train_place_cube_right_sweep.sh 14 training.debug=true   # args with = go to Hydra
#   DELTA=1 bash scripts/train_place_cube_right_sweep.sh       # the delta-action sweep
#   CUDA_VISIBLE_DEVICES=0,1,2,3 LORA=1 DELTA=1 bash scripts/train_place_cube_right_sweep.sh 56
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# The global batch is per-GPU batch x GPUs x accumulation: keep it at 256
# (16 x 2 x 8 in full; 32/GPUs per GPU x 8 under LORA=1), or the runs are not comparable.
batch_size=16
lora_args=()
if [[ "${LORA:-0}" == "1" ]]; then
    export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    if (( 32 % NPROC_PER_NODE != 0 )); then
        echo "LORA=1: NPROC_PER_NODE=${NPROC_PER_NODE} must divide 32 (global batch 256 = per-GPU x GPUs x 8)" >&2
        exit 1
    fi
    batch_size=$(( 32 / NPROC_PER_NODE ))
    lora_args=(
        training.lora.enabled=true
        "training.lora.lr=${LORA_LR:-5e-4}"
        training.text_embedding_cache=true
        training.text_encoder_offload=true
    )
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
FOUNDATION="${FOUNDATION:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"

sizes=()
overrides=()
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then overrides+=("$arg"); else sizes+=("$arg"); fi
done
[[ ${#sizes[@]} -gt 0 ]] || sizes=(14 28 42 56)

suffix=""
if [[ "${DELTA:-0}" == "1" ]]; then
    suffix="_delta"
    overrides=('dataloader.delta_action_dims=["0-5","7-12"]' ${overrides[@]+"${overrides[@]}"})
fi
if [[ "${LORA:-0}" == "1" ]]; then suffix+="_lora"; fi

for n in "${sizes[@]}"; do
    bash scripts/train.sh \
        dataloader="yam_place_cube_right_n${n}" \
        training.finetune_ckpt_path="${FOUNDATION}" \
        training.run_name="place_cube_right_n${n}${suffix}" \
        training.num_epochs=null \
        training.max_steps=4096 \
        training.batch_size="${batch_size}" \
        training.gradient_accumulation_steps=8 \
        training.learning_rate=1e-4 \
        training.lr_scheduler=constant \
        training.warmup_ratio=0.05 \
        training.save_steps=1024 \
        training.keep_last_k_ckpts=4 \
        ${lora_args[@]+"${lora_args[@]}"} \
        ${overrides[@]+"${overrides[@]}"}
done
