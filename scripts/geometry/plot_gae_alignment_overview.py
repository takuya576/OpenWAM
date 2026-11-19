#!/usr/bin/env python3
"""Draws docs/gae_alignment_overview.png (the procedure of the GAE alignment loss).  Any env with matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
FROZEN, TRAIN, NEW, DATA = "#e9e8e4", "#d6e6f8", "#fbe0d4", "#ffffff"
E_FROZEN, E_TRAIN, E_NEW, E_DATA = "#8d8c86", "#2a78d6", "#eb6834", "#52514e"

fig, ax = plt.subplots(figsize=(15.5, 9.4), facecolor=SURF)
ax.set_xlim(0, 155)
ax.set_ylim(0, 94)
ax.axis("off")


def box(x, y, w, h, title, body="", fc=DATA, ec=E_DATA, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(
        x + w / 2,
        y + h - 2.1 if body else y + h / 2,
        title,
        ha="center",
        va="top" if body else "center",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    if body:
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.35)


def arrow(p, q, color=E_DATA, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle=style,
            mutation_scale=13,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
            shrinkA=1,
            shrinkB=1,
        )
    )


def lane(y, h, label):
    ax.add_patch(
        FancyBboxPatch(
            (1, y), 153, h, boxstyle="round,pad=0.2,rounding_size=1.5", fc="none", ec="#cfcec9", lw=1.0, ls=(0, (5, 4))
        )
    )
    ax.text(2.6, y + h - 1.6, label, ha="left", va="top", fontsize=11.5, color=INK, fontweight="bold")


ax.text(
    1,
    92.6,
    "GAE alignment loss for OpenWAM fine-tuning: procedure",
    fontsize=15,
    color=INK,
    fontweight="bold",
    va="top",
)

# ---- lane 1: offline cache
lane(69.5, 17.5, "1  Offline, once per dataset: cache the geometry targets")
box(6, 71.5, 28, 11, "Demo camera frames", "every frame, 3 cameras\noriginal 640×480")
box(46, 71.5, 31, 11, "DA3-Giant + GAE encoder", "frozen, one frame at a time\ninput 378×504", FROZEN, E_FROZEN)
box(89, 71.5, 30, 11, "GAE latent", "64 channels × 27×36 tokens\nper camera frame, standardised")
box(129, 71.5, 21, 11, "Cache on disk", "one file per episode\n(T, 3, 64, 27, 36)")
arrow((34, 77), (46, 77))
arrow((77, 77), (89, 77))
arrow((119, 77), (129, 77))

# ---- lane 2: training step
lane(12.5, 55, "2  Every training step")
box(
    4,
    44,
    25,
    17,
    "Batch",
    "9 canvas frames 384×320\n(frame 0 + 8 future)\n32 actions, proprio,\nepisode and frame indices",
)
box(35, 52.5, 19, 8.5, "Video VAE", "frozen", FROZEN, E_FROZEN)
box(35, 39.5, 19, 10, "Add noise", "independent levels\nfor video and actions")
box(
    60,
    37,
    34,
    24,
    "Video DiT + Action DiT",
    "30 blocks, joint attention,\nfine-tuned (LoRA or full)\nvideo tokens: 3 latent frames\n× 12×10 tokens × 3072",
    TRAIN,
    E_TRAIN,
)
ax.plot([63, 91], [41.6, 41.6], color=E_NEW, lw=2.4)
ax.text(77, 39.1, "block 15", ha="center", va="center", fontsize=9.5, color=E_NEW, fontweight="bold")
box(100, 52.5, 23, 8.5, "Video loss", "flow matching, unchanged")
box(100, 41, 23, 8.5, "Action loss", "flow matching, unchanged")
box(129, 41, 21, 20, "Total loss", "video\n+ action\n+ λ · alignment", "#ffffff", INK, 1.8)
arrow((29, 56.7), (35, 56.7))
arrow((29, 47), (35, 45.5))
arrow((44.5, 52.5), (44.5, 49.5))
arrow((54, 44.5), (60, 47))
arrow((94, 56.7), (100, 56.7))
arrow((94, 45.2), (100, 45.2))
arrow((123, 56.7), (129, 54))
arrow((123, 45.2), (129, 47))

box(
    56,
    16,
    40,
    14,
    "Projector (training only)",
    "hidden state of each future video token\n→ MLP → a patch of GAE tokens for each of\nthe 4 video frames that token stands for",
    NEW,
    E_NEW,
)
box(102, 16, 21, 14, "Alignment loss", "1 − cosine similarity,\nmean per camera, then\nover the 3 cameras", NEW, E_NEW)
box(129, 16, 21, 14, "Targets", "8 future frames × 3\ncameras from the\ncache, on a 32×40 grid")
arrow((77, 37), (77, 30), E_NEW, 2.2)
ax.text(78.3, 33.6, "future-frame video tokens", ha="left", va="center", fontsize=9, color=E_NEW)
arrow((96, 23), (102, 23), E_NEW, 2.2)
arrow((129, 23), (123, 23))
arrow((112.5, 30), (131, 41), E_NEW, 2.2, rad=-0.18)
ax.plot([139.5, 139.5, 152.4, 152.4], [71.5, 66.2, 66.2, 23], color=E_DATA, lw=1.2, ls=(0, (4, 3)))
arrow((152.4, 23), (150, 23), E_DATA, 1.2)
ax.text(
    4,
    32.5,
    "Targets are looked up by the episode and\nframe indices of the batch. Each camera's\n27×36 GAE grid is resized to the fine grid of\nits canvas slot: one video token predicts\n4×4 GAE tokens in the head view and 8×8 in\na wrist view, so every camera is 32×40.",
    ha="left",
    va="top",
    fontsize=9,
    color=MUTED,
    linespacing=1.4,
)

# ---- lane 3: deployment
lane(1.5, 9, "3  Deployment")
ax.text(
    30,
    6.0,
    "The projector and the cache are not used. The deploy checkpoint has the same layout, inputs and cost as before;\nthe projector is saved beside it in its own file (gae_projector_step_N.safetensors) for analysis only.",
    fontsize=10,
    color=INK,
    va="center",
)

# legend
for k, (fc, ec, t) in enumerate(
    [(FROZEN, E_FROZEN, "frozen"), (TRAIN, E_TRAIN, "fine-tuned"), (NEW, E_NEW, "new, training only")]
):
    ax.add_patch(
        FancyBboxPatch((104 + k * 17, 89.3), 3, 2.2, boxstyle="round,pad=0.1,rounding_size=0.5", fc=fc, ec=ec, lw=1.2)
    )
    ax.text(108 + k * 17, 90.4, t, fontsize=9.5, color=INK, va="center")

out = Path(__file__).resolve().parents[2] / "docs" / "gae_alignment_overview.png"
fig.savefig(out, dpi=130, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
print(out)
