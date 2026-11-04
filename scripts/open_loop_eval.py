#!/usr/bin/env python3
"""Open-loop evaluation of a trained OpenWAM checkpoint on recorded episodes.

Runs the policy server's inference path in-process on each episode's recorded
observations and writes into the output directory:

  episode_XXXXXX_actions.png  predicted vs recorded actions, one panel per dimension
  episode_XXXXXX_video.mp4    generated | recorded | amplified difference
  episode_XXXXXX.npz          recorded, executed and hold-baseline actions per step
  summary.json, summary.png   errors per dimension group, by step within the chunk
                              against holding the start state, and video PSNR

Scoring rules are documented in openwam/eval/open_loop.py.

Defaults come from the checkpoint itself: its dataset, its ``val`` split as the
episodes, its latest checkpoint_step_*.safetensors, and configs/deploy.yaml for
inference settings. Results are filed under the model that produced them, at
its place under outputs/openwam_checkpoints/ mirrored into outputs/eval_out/:

  outputs/openwam_checkpoints/<run_name>/<time>/checkpoint_step_N.safetensors
  outputs/eval_out/<run_name>/<time>/checkpoint_step_N/open_loop/

scripts/run_yam_policy.py files the same checkpoint's real-robot rollouts
beside them, in closed_loop/.

Usage:
    python scripts/open_loop_eval.py --ckpt-dir outputs/openwam_checkpoints/<run>
    python scripts/open_loop_eval.py --ckpt-dir outputs/openwam_checkpoints/<run> \\
        --ckpt-name checkpoint_step_3000.safetensors --episodes 56 57
    # deploy-config overrides as dotted keys, e.g. without the DiT cache:
    python scripts/open_loop_eval.py --ckpt-dir ... optimization.dit_cache.enabled=false
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from openwam.eval.open_loop import evaluate  # noqa: E402
from openwam.eval.policy import EnginePolicy  # noqa: E402
from openwam.train.utils.checkpointing import find_latest_weights  # noqa: E402

logger = logging.getLogger("open_loop_eval")


def eval_dir(ckpt_dir: Path, ckpt_name: str) -> Path:
    """outputs/eval_out/<ckpt_dir under outputs/openwam_checkpoints>/<checkpoint>/open_loop.

    A checkpoint dir outside outputs/openwam_checkpoints is filed by its own name.
    scripts/run_yam_policy.py's rollout_dir applies the same rule for closed_loop/.
    """
    root = (PROJECT_ROOT / "outputs" / "openwam_checkpoints").resolve()
    run = ckpt_dir.resolve()
    run = run.relative_to(root) if run.is_relative_to(root) else Path(run.name)
    return PROJECT_ROOT / "outputs" / "eval_out" / run / Path(ckpt_name).stem / "open_loop"


def _episode_range(spec) -> list[int]:
    """A split's episodes: a half-open "start:end" range, or an explicit list (an episode subset)."""
    if not isinstance(spec, str):
        return [int(ep) for ep in spec]
    start, end = (int(part) for part in spec.split(":"))
    return list(range(start, end))


def _splits(train_cfg, dataset_dir: Path) -> dict:
    """The split ranges the checkpoint trained with: its yaml override, else the dataset's info.json."""
    if OmegaConf.select(train_cfg, "dataloader.splits", default=None) is not None:
        return dict(OmegaConf.to_container(train_cfg.dataloader.splits))
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    return dict(info.get("splits", {}) or {})


def _training_data_unchanged(train_cfg, ckpt_dir: Path, dataset_dir: Path, splits: dict):
    """Whether the train-split files still hash to what the checkpoint's stats were fitted on.

    Only YAM stats carry the fingerprint; returns None when there is nothing to compare.
    """
    if OmegaConf.select(train_cfg, "dataloader.type", default=None) != "yam":
        return None
    stats = np.load(ckpt_dir / "normalization_stats.npy", allow_pickle=True).item()
    recorded = stats.get("joint", {}).get("source_sha256")
    if recorded is None:
        return None
    from openwam.dataloader.yam import train_data_fingerprint

    return train_data_fingerprint(dataset_dir, splits) == recorded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-dir", required=True, help="Training output directory (config.yaml + checkpoints)")
    parser.add_argument("--ckpt-name", default=None, help="Checkpoint file (default: highest checkpoint_step_*)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Default: the checkpoint's val split")
    parser.add_argument("--dataset-dir", default=None, help="Default: the checkpoint's dataloader.dataset_dir")
    parser.add_argument(
        "--horizon", type=int, default=None, help="Actions executed per chunk (default: deploy inference_horizon)"
    )
    parser.add_argument("--no-video", action="store_true", help="Skip VAE decoding and the comparison videos")
    parser.add_argument("--seed", type=int, default=42, help="Denoising seed (the server always uses 42)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="Default: outputs/eval_out/<run>/<checkpoint>/open_loop")
    parser.add_argument("overrides", nargs="*", help="Deploy-config overrides as dotted keys, key=value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_name = args.ckpt_name or os.path.basename(find_latest_weights(str(ckpt_dir)))
    if not (ckpt_dir / ckpt_name).is_file():
        parser.error(f"{ckpt_dir / ckpt_name} does not exist")
    train_cfg = OmegaConf.load(ckpt_dir / "config.yaml")
    dataset_dir = Path(args.dataset_dir or train_cfg.dataloader.dataset_dir)
    splits = _splits(train_cfg, dataset_dir)

    if args.episodes:
        episodes = list(args.episodes)
    elif "val" in splits:
        episodes = _episode_range(splits["val"])
    else:
        parser.error("the checkpoint has no val split; pass --episodes")
    with open(dataset_dir / "meta" / "episodes.jsonl", encoding="utf-8") as handle:
        available = {json.loads(line)["episode_index"] for line in handle if line.strip()}
    missing = sorted(set(episodes) - available)
    if missing:
        parser.error(f"episodes {missing} are not in {dataset_dir}")
    in_train = sorted(set(episodes) & set(_episode_range(splits["train"]))) if "train" in splits else []
    if in_train:
        logger.warning("episodes %s are in the training split: their scores are not held-out results", in_train)
    unchanged = _training_data_unchanged(train_cfg, ckpt_dir, dataset_dir, splits)
    if unchanged is False:
        logger.warning("the training episodes on disk changed since this checkpoint's stats were fitted")

    out = Path(args.out) if args.out else eval_dir(ckpt_dir, ckpt_name)
    policy = EnginePolicy(ckpt_dir, ckpt_name, device=args.device, overrides=args.overrides, seed=args.seed)
    metadata = {
        "checkpoint": str(ckpt_dir / ckpt_name),
        "inference": policy.settings(),
        "episodes_in_train_split": in_train,
        "training_data_unchanged": unchanged,
    }
    summary = evaluate(
        policy,
        dataset_dir,
        episodes,
        out,
        horizon=args.horizon or policy.horizon,
        video=not args.no_video,
        metadata=metadata,
    )

    overall = summary["overall"]
    print(f"\n{ckpt_name} on episodes {episodes} (horizon {summary['horizon']}):")
    for group, mae in overall["mae"].items():
        print(f"  {group:12s} MAE {mae:.4f}   holding the start state: {overall['hold_mae'][group]:.4f}")
    if overall["video_psnr_by_offset"]:
        psnrs = ", ".join(f"+{k}: {v:.1f}" for k, v in overall["video_psnr_by_offset"].items())
        print(f"  video PSNR (dB) by recorded steps after the input frame: {psnrs}")
    print(f"wrote plots, videos and summary.json to {out}")


if __name__ == "__main__":
    main()
