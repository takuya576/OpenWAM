#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# stand_up_bottle α fine-tune: episodes 0-49 train (and fit the normalization
# stats), 50-54 are held out for open-loop eval (configs/dataloader/
# yam_stand_up_bottle.yaml).
#
# The recipe is the place_cube_right *_delta one: global batch 256 (16 per GPU x
# 2 GPUs x 8 accumulation), LR 1e-4, 5% warmup then constant. Episodes here are
# ~2x longer (~500 frames vs ~233), so the budget is 8192 micro-steps = 1024
# optimizer steps ≈ 10.4 epochs (the paper's single-arm 10 epochs). Checkpoints
# land every 2048 micro-steps; step 4096 matches the place_cube_right sweep's
# 512 optimizer steps.
#
# Arm joints train as delta actions (action - chunk-start state, grippers
# absolute); deploy adds the state back. DELTA=0 trains absolute joints instead.
#
# Batch 16 needs both GPUs to itself: each optimizer step allocates a fresh
# 11.2 GiB fp32 grad partition on top of ~68 GiB resident, and fails (OOM or a
# DeepSpeed segfault) beside another process's ~5 GB. Do not work around it
# with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: that NaNs the model at
# the first optimizer step. WAIT_FOR_GPUS=1 blocks until no other compute
# process is on the GPUs (two checks a minute apart), then launches.
#
# LORA=1 fine-tunes with LoRA instead of in full (training.lora in configs/train.yaml;
# see scripts/train_place_cube_right_sweep.sh): it fits 46 GB GPUs (L40S), defaults to 4
# of them with 32 / NPROC_PER_NODE per GPU (global batch still 256, same step counts),
# adapter LR LORA_LR (default 5e-4). Run names get a _lora suffix.
#
# Checkpoints: outputs/openwam_checkpoints/stand_up_bottle_n50[_delta][_lora]/<launch time>/
# wandb run:   stand_up_bottle_n50[_delta][_lora]_<launch time>
#
# Usage:
#   bash scripts/train_stand_up_bottle.sh
#   WAIT_FOR_GPUS=1 bash scripts/train_stand_up_bottle.sh
#   bash scripts/train_stand_up_bottle.sh training.debug=true   # args go to Hydra
#   CUDA_VISIBLE_DEVICES=0,1,2,3 LORA=1 bash scripts/train_stand_up_bottle.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

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
        training.text_encoder_offload=true
    )
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
FOUNDATION="${FOUNDATION:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"

run_name="stand_up_bottle_n50"
delta_args=()
if [[ "${DELTA:-1}" == "1" ]]; then
    run_name+="_delta"
    delta_args=('dataloader.delta_action_dims=["0-5","7-12"]')
fi
if [[ "${LORA:-0}" == "1" ]]; then run_name+="_lora"; fi

if [[ "${WAIT_FOR_GPUS:-0}" == "1" ]]; then
    echo "[$(date '+%F %T')] waiting until no other compute process is on the GPUs..."
    free_checks=0
    while (( free_checks < 2 )); do
        if apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) && [[ -z "$apps" ]]; then
            free_checks=$((free_checks + 1))
        else
            free_checks=0
        fi
        (( free_checks >= 2 )) || sleep 60
    done
    echo "[$(date '+%F %T')] GPUs free; launching ${run_name}."
fi

bash scripts/train.sh \
    dataloader=yam_stand_up_bottle \
    ${delta_args[@]+"${delta_args[@]}"} \
    training.finetune_ckpt_path="${FOUNDATION}" \
    training.run_name="${run_name}" \
    training.num_epochs=null \
    training.max_steps=8192 \
    training.batch_size="${batch_size}" \
    training.gradient_accumulation_steps=8 \
    training.learning_rate=1e-4 \
    training.lr_scheduler=constant \
    training.warmup_ratio=0.05 \
    training.save_steps=2048 \
    training.keep_last_k_ckpts=4 \
    ${lora_args[@]+"${lora_args[@]}"} \
    "$@"
