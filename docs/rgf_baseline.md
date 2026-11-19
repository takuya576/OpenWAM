# Relational Geometry Forcing (RGF) baseline

A re-implementation of *Relational Geometry Forcing for World Action Models* (arXiv 2610.29282,
Section 3.2 and Appendix A.2) for OpenWAM. It is a training-only auxiliary loss used as a published
baseline next to our pointwise GAE alignment (`docs/gae_alignment.md`). Nothing changes at deployment.

## What the loss does

After block 15 of **both** experts (video DiT and action DiT), the hidden states pass through two small
projection heads. Their pairwise cosine similarities are made to match those of a frozen multi-view 3D
teacher:

| Term | Student relation | Teacher relation | Entries |
|---|---|---|---|
| intra-frame | tokens of one camera view in one latent frame, after `g_vid` | teacher patches of that view, resampled to the same tokens | all pairs except the diagonal |
| cross-view | tokens of view v vs view u (v < u), same latent frame | same for the teacher | all pairs |
| action-temporal | the 32 action tokens, after `g_act` | scene vectors `s_t` = per-view mean teacher features of the observation at each of the 32 steps, concatenated | all pairs except the diagonal and padded steps |

Each relation pair is compared with a Huber (smooth-L1, β = 1) loss averaged over the valid entries. The
intra and cross terms are averaged over views, view pairs, latent frames and samples; the temporal term
over samples.

    loss = video + action + 0.5 · (L_intra + L_cross + L_temporal)

On the YAM canvas the token sets are: head 8×10 tokens, each wrist 4×5 tokens, 3 latent frames.

## Faithful to the paper

- Cosine relation matrices with row-wise L2 normalisation and no temperature; Huber β = 1; the diagonal
  and padded steps are masked; the three terms are equally weighted; λ_RGF = 0.5 (paper App. A.2).
- Layer 15 in both experts (paper Table A5).
- Heads: separate two-layer MLPs with SiLU, video 3072 → 2048 and action 1024 → 2048 (paper App. B.1).
  They are trained, dropped at deployment, and saved separately for analysis as
  `rgf_projector_step_N.safetensors`.
- Teacher features are cached before training. The views of one time step are encoded together as one
  set. The final-layer patch grid is bilinearly resampled to each view's token grid. Scene vectors are
  spatial means of the full-resolution patches.
- The 32 action steps are paired with the observations at the same 32 steps, independent of the 9-frame
  video sampling.
- RGF is on for the whole fine-tune, with no warm-up.

## Differences, and why

| Point | Paper | Here | Reason |
|---|---|---|---|
| Teacher | VGGT-1B, last aggregator layer (2048-d) | three runs: **VGGT-1B** as in the paper (`RGF_TEACHER=vggt`); **DA3-Giant**, final block 39 of 40 (3072-d) (`RGF_TEACHER=da3`, the first run); and **GAE** (DA3-Giant + GAE-64, 64-d) (`RGF_TEACHER=gae`) | DA3 was used first because the VGGT weights were not yet available here. It is also a multi-view 3D foundation model and the backbone of our GAE teacher. For VGGT and DA3 the views of one time step are encoded as one set. The GAE run uses exactly the targets of the pointwise GAE alignment run (standardized GAE-64 posterior means, each view encoded on its own), so it differs from that run only in the loss. |
| Layer index | "layer 15" | hidden states after 15 of 30 blocks | same convention as our GAE alignment, so the two are comparable |
| RGB frame per latent frame | "frames corresponding to the VAE latent timesteps" | the last frame of each causal VAE group: clip frames 0, 4, 8 = episode offsets +0, +16, +32 | the paper does not say which frame of a 4-frame group |
| Resampling | bilinear | bilinear with antialiasing | 27×36 → 4×5 is a 7× downsampling; plain bilinear would sample only a few patches |
| Head hidden width | not stated | 2048 | |
| Fine-tuning | not stated (LoRA vs full) | LoRA rank 128 like our other runs; the heads are trained in full at the base learning rate | matched to the GAE run and its control |
| Which hidden states | not stated; implied by Algorithm A1: the normal training forward | the training forward pass, at the sampled noise levels; all 3 latent frames, including the clean first frame | |

## Commands

```bash
# 1. teacher cache (conda env gae); one process per GPU, about 25 min on 2 x A6000. TEACHER = vggt or da3
CUDA_VISIBLE_DEVICES=0 python scripts/geometry/rgf_cache.py --teacher $TEACHER \
    --dataset-dir assets/datasets/hang_ring --out assets/rgf_cache/hang_ring_$TEACHER --shard 0/2
CUDA_VISIBLE_DEVICES=1 python scripts/geometry/rgf_cache.py --teacher $TEACHER \
    --dataset-dir assets/datasets/hang_ring --out assets/rgf_cache/hang_ring_$TEACHER --shard 1/2
# TEACHER = gae instead: read from the GAE alignment cache (assets/gae_cache/hang_ring), CPU only, about 30 s
python scripts/geometry/rgf_cache.py --teacher gae --device cpu \
    --dataset-dir assets/datasets/hang_ring --out assets/rgf_cache/hang_ring_gae
python scripts/geometry/rgf_trivial_floors.py assets/rgf_cache/hang_ring_$TEACHER   # loss levels of trivial students

# 2. fine-tune (conda env openwam); same settings as the GAE run and its control
RGF=1 RGF_TEACHER=$TEACHER LORA=1 NPROC_PER_NODE=2 bash scripts/train_hang_ring.sh training.batch_size=8 \
    training.gradient_accumulation_steps=16 training.max_steps=10240 training.save_steps=2560
```

Runs are named `hang_ring_n50_delta_lora_rgf15` (DA3), `hang_ring_n50_delta_lora_rgf15vggt` (VGGT) and
`hang_ring_n50_delta_lora_rgf15gae` (GAE). The script refuses a cache whose `meta.json` names a different teacher.
Logged: `train/loss_rgf` (the unweighted sum of the three terms) and `train/loss_rgf_intra`, `_cross`, `_temporal`.

Cache size for `hang_ring`: `(T, 120, D)` + `(T, 3, D)` float16 per episode, D = 3072 (DA3, 12 GB), 2048
(VGGT, about 8 GB) or 64 (GAE, 254 MB).

### VGGT-1B checkpoint

- `assets/vggt/VGGT-1B/model.safetensors` and `config.json`, from Hugging Face `facebook/VGGT-1B` at revision
  `860abec7937da0a4c03c41d3c269c366e82abdf9`. The checkpoint's sha256 is
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`, matching the hash the Hub publishes.
- **License: CC-BY-NC-4.0, non-commercial.** The separately gated `facebook/VGGT-1B-Commercial` is the
  commercial checkpoint.
- Source code: `/data2/takuya/vggt` (also at `/home/takuya/vggt`), a clone of `https://github.com/takuya576/vggt`
  (fork of facebookresearch/vggt) at commit `a288dd0`. `rgf_cache.py --vggt-src` points at it.
- The cache script builds VGGT without its four heads and loads the checkpoint into the aggregator. Any
  missing weight is an error; the only keys allowed to be left over are the heads'. Input preprocessing is
  VGGT's own "crop" mode: 640×480 → 518×392, bicubic, then [0, 1]. On GPU it runs in bf16, as the VGGT README
  advises for compute capability 8 and up.

## Code

| File | Change |
|---|---|
| `openwam/model/rgf_align.py` | New. Config, cache reader, heads, the three terms |
| `scripts/geometry/rgf_cache.py` | New. Teacher cache (VGGT-1B or DA3-Giant) |
| `scripts/geometry/rgf_trivial_floors.py` | New. Loss of trivial students on a cache |
| `openwam/dataloader/bases/lerobot_v3_reader.py` | `rgf_cache_dir`; samples carry `rgf_grid`, `rgf_pool` and their validity masks |
| `openwam/model/architectures/dual_system/mot_driver.py` | the block loop also keeps the action hidden states |
| `openwam/model/architectures/base.py` | `enable_rgf`, `_compute_rgf_loss`, heads kept out of the deploy checkpoint |
| `openwam/train/...` | config wiring, LoRA treatment of the heads, logging, `rgf_projector_step_N` files |
| `configs/train.yaml`, `configs/dataloader/yam.yaml`, `scripts/train_hang_ring.sh` | `training.rgf`, `rgf_cache_dir`, `RGF=1` |
| `tests/test_rgf_align.py` | 9 tests |

## What has been checked

On the workstation, 18 Nov 2026 (CPU only; the GPUs were training the control run):

- **Tests.** `tests/test_rgf_align.py`, 9 tests:
  - the slot layout covers the 12×10 grid exactly;
  - the loss is zero when the relations match, and invariant to feature scale;
  - invalid frames and padded steps do not change the loss;
  - values match a hand-computed Huber error in both branches;
  - cache indexing;
  - the heads are kept out of checkpoints and saved separately;
  - the block loop keeps the action state after the same block as a manual step-by-step run;
  - end to end on a tiny architecture: `loss = video + action + 0.5 · rgf`, and both heads receive gradients.

  The CI suite (`make test`) gives 2,234 passed and 21 skipped.
- **Dataset.** On the real `hang_ring` training set with a cache whose values are the frame index:
  samples carry frames offset + {0, 16, 32} and offset + 0…31, with correct validity at episode ends.
- **Teacher, on frames of episode 0.**
  - The features are finite, and the slot tokens equal a direct resample.
  - Encoding the 3 views together changes the head-view features (cosine 0.87 vs encoding the head view
    alone), so the views attend to each other.
  - The raw final-layer patches are strongly similar to one another: mean pairwise cosine 0.91–0.94 inside
    a view. The relation targets therefore vary in a narrow band. The paper does not centre features, and
    neither does this implementation.
- **Full cache** (`assets/rgf_cache/hang_ring_da3`, 12 GB): 54 of 54 episodes, all shapes correct, sampled
  rows finite and non-zero.
- **GPU smoke run**, 1 GPU, 20 micro-steps, 10 optimizer steps, LoRA:
  - `loss = video + action + 0.5 · rgf` holds in the logged values;
  - heads: 16.8M parameters, all trainable;
  - the checkpoint has exactly the control checkpoint's 2,089 keys, none from the heads;
  - `rgf_projector_step_20.safetensors` holds the 8 head tensors and the config;
  - about 2.3 s per micro-step, as in the control.

### How much of the loss is geometry

Loss of trivial students on 216 random windows of the cache (`outputs/gae_align_eval/summary_rgf_trivial.txt`),
using the training code's relation and Huber functions:

| Term | Teacher relations: mean, spread within a matrix | All vectors equal (every relation = 1) | One constant relation per matrix, set to the teacher's mean | Smoke run: start → after 10 steps |
|---|---|---|---|---|
| intra-frame | 0.90, std 0.065 | 0.0083 | 0.0024 | 0.065 → 0.0047 |
| cross-view | 0.68, std 0.071 | 0.0645 | 0.0029 | 0.029 → 0.0098 |
| action-temporal | 0.93, std 0.043 | 0.0037 | 0.0011 | 0.0047 → 0.0014 |

Most of the initial loss is the heads' overall similarity level, and the heads fix it within a few steps.
Only the part below the last trivial column, "one constant relation per matrix", is geometric structure. A
run's final RGF terms should be read against that column.

### VGGT-1B teacher, checked on CPU on 18 Nov 2026

The GPUs were busy with the DA3 run, so these checks ran on CPU.

- **Load.** The full model loads strictly: 1,797 tensors, 1.257B parameters, 0.909B of them in the
  aggregator. The cache script's aggregator-only load gives identical aggregator weights and leaves exactly
  the 587 head tensors over.
- **Depth.** The full model's depth maps on the three views of episode 0, step 100, match the scene: the ring
  stand stands out from the table, the gripper fingers are nearest, and the ring is just above the table in
  the right-wrist view.
- **Teacher features.** Output shape (T, 3, 28, 37, 2048), all finite. Encoding the three views together
  changes the head-view features (cosine 0.956 against the head view encoded alone).
- **Relations against DA3, episode 0.** Measured on steps 0, 100 and 200, and for the temporal term on
  steps 90–121:

  | Term | DA3-Giant: mean, std | VGGT-1B: mean, std |
  |---|---|---|
  | intra-frame | 0.895, 0.075 | 0.907, 0.076 |
  | cross-view | 0.691, 0.086 | 0.821, 0.095 |
  | action-temporal (scene vectors) | 0.852, 0.092 | **0.987, 0.009** |

  VGGT's per-view mean-pooled scene vectors barely change within 32 steps, so with VGGT the action-temporal
  target is close to flat. These numbers come from one window; `rgf_trivial_floors.py` on the full VGGT
  cache gives the dataset-wide values (`outputs/gae_align_eval/summary_rgf_trivial_vggt.txt`).

### The three teachers compared, 19 Nov 2026

The GAE cache (`assets/rgf_cache/hang_ring_gae`, 54 of 54 episodes) matches a direct resample and mean of the raw
GAE cache to within float16 rounding (episodes 0, 7, 53; first, middle and last step). Dataset-wide relations and
trivial floors, from `rgf_trivial_floors.py` (216 windows each; `summary_rgf_trivial{,_vggt,_gae}.txt`):

| Term | DA3: mean, spread, mean floor | VGGT: mean, spread, mean floor | GAE: mean, spread, mean floor |
|---|---|---|---|
| intra-frame | 0.90, 0.065, 0.0024 | 0.89, 0.086, 0.0042 | 0.34, 0.27, 0.037 |
| cross-view | 0.68, 0.071, 0.0029 | 0.80, 0.11, 0.0063 | 0.21, 0.21, 0.022 |
| action-temporal | 0.93, 0.043, 0.0011 | 0.98, 0.014, 0.0001 | 0.97, 0.021, 0.0003 |

The standardized 64-d GAE latents give spatial relation targets with 3–4 times the spread of the raw DA3 and VGGT
features, so the intra and cross terms carry much more structure. All three teachers give a nearly flat
action-temporal target. λ stays at 0.5 for all three, so the RGF loss is larger in absolute terms with GAE.
