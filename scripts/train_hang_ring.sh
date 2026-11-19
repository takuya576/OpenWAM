#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# hang_ring α fine-tune: episodes 0-49 train (and fit the normalization
# stats), 50-53 are held out for open-loop eval (configs/dataloader/
# yam_hang_ring.yaml).
#
# The recipe is the place_cube_right *_delta one: global batch 256 (16 per GPU x
# 2 GPUs x 8 accumulation), LR 1e-4, 5% warmup then constant. The 50 train
# episodes hold ~15.5k windows (~311 frames each vs ~233), so the budget is 5120
# micro-steps = 640 optimizer steps ≈ 10.5 epochs (the paper's single-arm 10
# epochs; place_cube_right n56's 4096 is 10.1). Checkpoints land every 1280
# micro-steps (≈ 2.6 epochs apart).
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
# GAE_ALIGN=1 adds the GAE alignment loss (docs/gae_alignment.md): block GAE_LAYER (default 15),
# loss GAE_LOSS (cosine | mse | cosine_mse, default cosine), weight GAE_WEIGHT (default 0.5),
# targets from GAE_CACHE (default assets/gae_cache/hang_ring, written by scripts/geometry/gae_cache.py).
# Run names get a _gae<layer><loss> suffix.
#
# RGF=1 adds the Relational Geometry Forcing baseline (docs/rgf_baseline.md): layer RGF_LAYER (default 15),
# weight RGF_WEIGHT (default 0.5), teacher RGF_TEACHER (da3 | vggt | gae, default da3) whose cache is RGF_CACHE
# (default assets/rgf_cache/hang_ring_<teacher>, written by scripts/geometry/rgf_cache.py). Run names get an
# _rgf<layer> suffix, plus the teacher name for teachers other than da3 (e.g. _rgf15vggt). Cannot be combined
# with GAE_ALIGN=1.
#
# RUN_SUFFIX is appended to the run name as given (e.g. RUN_SUFFIX=_ctrl for a control run).
#
# Checkpoints: outputs/openwam_checkpoints/hang_ring_n50[_delta][_lora][_gae...]/<launch time>/
# wandb run:   hang_ring_n50[_delta][_lora][_gae...]_<launch time>
#
# Usage:
#   bash scripts/train_hang_ring.sh
#   WAIT_FOR_GPUS=1 bash scripts/train_hang_ring.sh
#   bash scripts/train_hang_ring.sh training.debug=true   # args go to Hydra
#   CUDA_VISIBLE_DEVICES=0,1,2,3 LORA=1 bash scripts/train_hang_ring.sh
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

run_name="hang_ring_n50"
delta_args=()
if [[ "${DELTA:-1}" == "1" ]]; then
    run_name+="_delta"
    delta_args=('dataloader.delta_action_dims=["0-5","7-12"]')
fi
if [[ "${LORA:-0}" == "1" ]]; then run_name+="_lora"; fi
gae_args=()
if [[ "${GAE_ALIGN:-0}" == "1" ]]; then
    GAE_CACHE="${GAE_CACHE:-$PWD/assets/gae_cache/hang_ring}"
    if [[ ! -f "${GAE_CACHE}/meta.json" ]]; then
        echo "GAE_ALIGN=1: no GAE cache at ${GAE_CACHE} (run scripts/geometry/gae_cache.py first)" >&2
        exit 1
    fi
    gae_args=(
        training.gae_align.enabled=true
        "training.gae_align.layer=${GAE_LAYER:-15}"
        "training.gae_align.loss=${GAE_LOSS:-cosine}"
        "training.gae_align.weight=${GAE_WEIGHT:-0.5}"
        "dataloader.gae_cache_dir=${GAE_CACHE}"
    )
    run_name+="_gae${GAE_LAYER:-15}${GAE_LOSS:-cosine}"
fi
rgf_args=()
if [[ "${RGF:-0}" == "1" ]]; then
    if [[ "${GAE_ALIGN:-0}" == "1" ]]; then
        echo "RGF=1 and GAE_ALIGN=1 cannot be combined" >&2
        exit 1
    fi
    RGF_TEACHER="${RGF_TEACHER:-da3}"
    RGF_CACHE="${RGF_CACHE:-$PWD/assets/rgf_cache/hang_ring_${RGF_TEACHER}}"
    if [[ ! -f "${RGF_CACHE}/meta.json" ]]; then
        echo "RGF=1: no teacher cache at ${RGF_CACHE} (run scripts/geometry/rgf_cache.py first)" >&2
        exit 1
    fi
    cache_teacher=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['teacher'])" "${RGF_CACHE}/meta.json")
    if [[ "${cache_teacher}" != "${RGF_TEACHER}" ]]; then
        echo "RGF=1: RGF_TEACHER=${RGF_TEACHER} but ${RGF_CACHE} holds ${cache_teacher} features" >&2
        exit 1
    fi
    rgf_args=(
        training.rgf.enabled=true
        "training.rgf.layer=${RGF_LAYER:-15}"
        "training.rgf.weight=${RGF_WEIGHT:-0.5}"
        "dataloader.rgf_cache_dir=${RGF_CACHE}"
    )
    run_name+="_rgf${RGF_LAYER:-15}"
    if [[ "${RGF_TEACHER}" != "da3" ]]; then run_name+="${RGF_TEACHER}"; fi
fi
run_name+="${RUN_SUFFIX:-}"

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
    dataloader=yam_hang_ring \
    ${delta_args[@]+"${delta_args[@]}"} \
    training.finetune_ckpt_path="${FOUNDATION}" \
    training.run_name="${run_name}" \
    training.num_epochs=null \
    training.max_steps=5120 \
    training.batch_size="${batch_size}" \
    training.gradient_accumulation_steps=8 \
    training.learning_rate=1e-4 \
    training.lr_scheduler=constant \
    training.warmup_ratio=0.05 \
    training.save_steps=1280 \
    training.keep_last_k_ckpts=4 \
    ${lora_args[@]+"${lora_args[@]}"} \
    ${gae_args[@]+"${gae_args[@]}"} \
    ${rgf_args[@]+"${rgf_args[@]}"} \
    "$@"
