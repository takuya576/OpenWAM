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
# Checkpoints: outputs/openwam_checkpoints/place_cube_right_n<N>[_delta]/<launch time>/
# wandb run:   place_cube_right_n<N>[_delta]_<launch time>
#
# Usage:
#   bash scripts/train_place_cube_right_sweep.sh               # n14 n28 n42 n56, in turn
#   bash scripts/train_place_cube_right_sweep.sh 14 28 42      # some of them
#   bash scripts/train_place_cube_right_sweep.sh 14 training.debug=true   # args with = go to Hydra
#   DELTA=1 bash scripts/train_place_cube_right_sweep.sh       # the delta-action sweep
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# The global batch is per-GPU batch x GPUs x accumulation: keep 2 GPUs for every
# N, or the runs are not comparable.
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

for n in "${sizes[@]}"; do
    bash scripts/train.sh \
        dataloader="yam_place_cube_right_n${n}" \
        training.finetune_ckpt_path="${FOUNDATION}" \
        training.run_name="place_cube_right_n${n}${suffix}" \
        training.num_epochs=null \
        training.max_steps=4096 \
        training.batch_size=16 \
        training.gradient_accumulation_steps=8 \
        training.learning_rate=1e-4 \
        training.lr_scheduler=constant \
        training.warmup_ratio=0.05 \
        training.save_steps=1024 \
        training.keep_last_k_ckpts=4 \
        ${overrides[@]+"${overrides[@]}"}
done
