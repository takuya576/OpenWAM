#!/usr/bin/env python3
"""Phase 0B (step 2): how much geometry do OpenWAM's frame-0 video tokens carry?

For every window start in a targets file from ``phase0_targets.py``, this runs
the deploy inference path (``architecture.generate``) for its first denoising
step only — future video and actions at full noise, the current observation
clean, exactly what the action expert attends to first at deploy — and records
the hidden states of the 120 frame-0 video tokens (12x10, 32 px each) after
DiT blocks ``--layers`` (0 = the patch embedding, i.e. a linear map of the
Wan VAE latent).

Linear ridge probes then map each token to targets inside its 32x32 cell:

* depth edges   : 4x4 cells of 8 px, label = max |grad log depth| > tau
                  (metrics: average precision, best F1 exact / within 1 cell)
* relative depth: 4x4 cells of 8 px, log depth minus the camera slot's median (R^2)
* GAE latent    : 2x2 cells of 16 px x 64 channels, standardized (R^2) — the
                  quantity a Phase 2 alignment loss would push up

Probes are fitted on the ``fit`` episodes, the ridge penalty is chosen per
target on the ``select`` episodes, and everything is reported on the ``test``
(held-out) episodes. A reference row probes the GAE latent itself (2x2x64 per
token) for edges and depth: what geometry-aligned tokens could reach.

Features are never written to disk: fit/select enter as sufficient statistics,
test features are kept in memory.

Run in the ``openwam`` conda env:
    CUDA_VISIBLE_DEVICES=5 python scripts/geometry/phase0_probe.py \\
        --ckpt-dir outputs/openwam_checkpoints/place_cube_right_n56_delta_lora/2026-11-04_15-26-58 \\
        --ckpt-name checkpoint_step_3072.safetensors \\
        --targets outputs/geometry_phase0/place_cube_right/targets.npz \\
        --out outputs/geometry_phase0/place_cube_right/probe_step3072.json

``--ckpt-name`` may be an absolute path to other weights with the same
architecture (e.g. the foundation model); config, dataset and normalizer still
come from ``--ckpt-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

from omegaconf import OmegaConf  # noqa: E402

logger = logging.getLogger("phase0_probe")

# Ridge penalties added to the eigenvalues of the feature correlation matrix.
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
GROUPS = ("edge", "depth", "gae")
TOK_H, TOK_W = 12, 10  # frame-0 token grid of the 384x320 canvas


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="training run dir (config.yaml, normalization stats)")
    p.add_argument("--ckpt-name", required=True, help="weights file in --ckpt-dir, or an absolute path")
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0,5,10,15,20,25,30")
    p.add_argument("--tau", type=float, default=0.03, help="depth-edge threshold (as in phase0_gae_recon)")
    p.add_argument("--denoise-steps", type=int, default=None, help="default: configs/deploy.yaml")
    p.add_argument("--limit", type=int, default=None, help="use only the first N frames per split (smoke test)")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viz-frames", type=int, default=6)
    return p.parse_args()


# ── targets → per-token matrices ────────────────────────────────────────────


def tokens_from_cells(a: np.ndarray, sub: int) -> np.ndarray:
    """(C, 12*sub, 10*sub) cell map -> (120, sub*sub*C) rows in token order (h, w)."""
    c = a.shape[0]
    a = a.reshape(c, TOK_H, sub, TOK_W, sub).transpose(1, 3, 2, 4, 0)
    return a.reshape(TOK_H * TOK_W, sub * sub * c)


def cells_from_tokens(rows: np.ndarray, sub: int, c: int = 1) -> np.ndarray:
    """Inverse of tokens_from_cells: (120, sub*sub*c) -> (c, 12*sub, 10*sub)."""
    a = rows.reshape(TOK_H, TOK_W, sub, sub, c).transpose(4, 0, 2, 1, 3)
    return a.reshape(c, TOK_H * sub, TOK_W * sub)


def target_rows(t: dict, i: int, tau: float) -> dict:
    return {
        "edge": tokens_from_cells((t["depth_edge"][i][None].astype(np.float32) > tau).astype(np.float32), 4),
        "depth": tokens_from_cells(t["logdepth"][i][None].astype(np.float32), 4),
        "gae": tokens_from_cells(t["gae"][i].astype(np.float32), 2),
    }


def slot_of_token() -> np.ndarray:
    """0 = head (top 8 token rows), 1 = wrists."""
    s = np.zeros((TOK_H, TOK_W), dtype=np.int64)
    s[8:] = 1
    return s.reshape(-1)


# ── ridge via sufficient statistics ─────────────────────────────────────────


class Moments:
    """Accumulates X'X, X'Y, sums and Y'Y-diagonals in float64 on the GPU."""

    def __init__(self, d: int, k: int, device):
        f64 = dict(dtype=torch.float64, device=device)
        self.xx = torch.zeros(d, d, **f64)
        self.xy = torch.zeros(d, k, **f64)
        self.sx = torch.zeros(d, **f64)
        self.sy = torch.zeros(k, **f64)
        self.yy = torch.zeros(k, **f64)
        self.n = 0

    def to(self, device) -> "Moments":
        for name in ("xx", "xy", "sx", "sy", "yy"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def add(self, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.xx)
        y = y.to(self.xx)
        self.xx += x.T @ x
        self.xy += x.T @ y
        self.sx += x.sum(0)
        self.sy += y.sum(0)
        self.yy += (y * y).sum(0)
        self.n += x.shape[0]


def fit_ridge(m: Moments, lambdas):
    """Standardized-feature ridge for every lambda: returns [(W, b)] in raw feature space."""
    n = m.n
    mu = m.sx / n
    cov = m.xx / n - torch.outer(mu, mu)
    sd = cov.diagonal().clamp_min(0).sqrt()
    sd = sd.clamp_min(1e-3 * float(sd.median()) + 1e-12)  # near-constant features stay tame
    zz = cov / torch.outer(sd, sd)  # correlation of standardized features
    ymu = m.sy / n
    zy = (m.xy / n - torch.outer(mu, ymu)) / sd[:, None]
    evals, evecs = torch.linalg.eigh(zz)
    proj = evecs.T @ zy
    out = []
    for lam in lambdas:
        wz = evecs @ (proj / (evals + lam)[:, None])
        w = wz / sd[:, None]
        b = ymu - mu @ w
        out.append((w, b))
    return out


def moments_sse(m: Moments, w: torch.Tensor, b: torch.Tensor, sl: slice) -> torch.Tensor:
    """Per-output sum of squared errors of x @ w + b for outputs ``sl`` of the data in ``m``."""
    quad = ((m.xx @ w) * w).sum(0) + 2 * b * (m.sx @ w) + m.n * b * b
    cross = (m.xy[:, sl] * w).sum(0) + b * m.sy[sl]
    return m.yy[sl] - 2 * cross + quad


# ── metrics ─────────────────────────────────────────────────────────────────


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    lab = labels[order].astype(np.float64)
    tp = np.cumsum(lab)
    if tp[-1] == 0:
        return float("nan")
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / tp[-1])


def best_f1(scores: np.ndarray, labels: np.ndarray, tol: int, n_thr: int = 60) -> dict:
    """Dataset-wide best F1 over thresholds. ``scores``/``labels`` are (N, H, W) maps;
    with tol > 0, a detection counts if a label lies within ``tol`` cells and vice versa."""
    lab = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
    lab_d = F.max_pool2d(lab, 2 * tol + 1, 1, tol) > 0.5 if tol else lab > 0.5
    sc = torch.from_numpy(scores.astype(np.float32)).unsqueeze(1)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float("nan")}
    for thr in np.quantile(scores, np.linspace(0.5, 0.995, n_thr)):
        pred = sc > float(thr)
        pred_d = F.max_pool2d(pred.float(), 2 * tol + 1, 1, tol) > 0.5 if tol else pred
        n_pred = float(pred.sum())
        n_lab = float((lab > 0.5).sum())
        if n_pred == 0 or n_lab == 0:
            continue
        p = float((pred & lab_d).sum()) / n_pred
        r = float(((lab > 0.5) & pred_d).sum()) / n_lab
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best["f1"]:
            best = {"f1": f, "precision": p, "recall": r, "threshold": float(thr)}
    return best


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    sse = float(((pred - y) ** 2).sum())
    sst = float(((y - y.mean(0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def static_prior(t: dict, rows: np.ndarray, tau: float) -> dict:
    """Per-position predictions from the fit frames alone (no model): edge frequency,
    mean relative log depth and mean GAE latent of every cell, in token-row layout."""
    return {
        "edge": tokens_from_cells((t["depth_edge"][rows].astype(np.float32) > tau).mean(0)[None], 4),
        "depth": tokens_from_cells(t["logdepth"][rows].astype(np.float32).mean(0)[None], 4),
        "gae": tokens_from_cells(t["gae"][rows].astype(np.float32).mean(0), 2),
    }


def sse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(((pred - y) ** 2).sum())


def evaluate(pred: dict, ys: dict, slot: np.ndarray, prior: dict, dynamic: np.ndarray) -> dict:
    """pred/ys: group -> (N_frames, 120, k). slot: (120,) 0 head / 1 wrist.

    ``prior`` (group -> (120, k)) is the per-position fit-set average. Scores
    against it isolate what a token knows beyond "where in the fixed scene am I":
    * ``*_r2_vs_prior``: 1 - SSE(probe) / SSE(prior)
    * ``edge_ap_dynamic``: AP over cells whose fit-set edge frequency is within
      [0.1, 0.9] (``dynamic`` (120, 16)), i.e. cells that are edges in some frames
      and not in others (arm, gripper, objects); static table / background edges excluded.
    """
    res = {}
    for part, sel in (("all", np.ones_like(slot, dtype=bool)), ("head", slot == 0), ("wrist", slot == 1)):
        e_p = pred["edge"][:, sel].reshape(-1)
        e_y = ys["edge"][:, sel].reshape(-1)
        res[f"edge_ap/{part}"] = average_precision(e_p, e_y)
        res[f"edge_base_rate/{part}"] = float(e_y.mean())
        dyn = np.broadcast_to(dynamic[sel], pred["edge"][:, sel].shape).reshape(-1)
        res[f"edge_ap_dynamic/{part}"] = average_precision(e_p[dyn], e_y[dyn]) if dyn.any() else float("nan")
        res[f"edge_dynamic_base_rate/{part}"] = float(e_y[dyn].mean()) if dyn.any() else float("nan")
        res[f"edge_dynamic_cell_frac/{part}"] = float(dynamic[sel].mean())
        for g, k in (("depth", 16), ("gae", None)):
            if g not in pred:
                continue
            k = k or ys[g].shape[-1]
            p, y = pred[g][:, sel].reshape(-1, k), ys[g][:, sel].reshape(-1, k)
            pr = np.broadcast_to(prior[g][sel], pred[g][:, sel].shape).reshape(-1, k)
            res[f"{g}_r2/{part}"] = r2(p, y)
            res[f"{g}_r2_vs_prior/{part}"] = 1.0 - sse(p, y) / max(sse(pr, y), 1e-12)
    n = pred["edge"].shape[0]
    maps_p = np.stack([cells_from_tokens(pred["edge"][i], 4)[0] for i in range(n)])
    maps_y = np.stack([cells_from_tokens(ys["edge"][i], 4)[0] for i in range(n)])
    res["edge_best_f1/exact"] = best_f1(maps_p, maps_y, tol=0)
    res["edge_best_f1/tol1"] = best_f1(maps_p, maps_y, tol=1)
    return res


# ── model / data ────────────────────────────────────────────────────────────


def window_index(ds, episode: int, start: int):
    """Dataset index whose window starts at (episode, start), or None."""
    eps = ds._eps_df["episode_index"].to_numpy()
    hit = np.nonzero(eps == episode)[0]
    if hit.size == 0:
        return None
    e = int(hit[0])
    rel = start - int(ds._ep_valid_start[e])
    if rel < 0 or rel % ds._window_stride:
        return None
    idx = int(ds._cum_n_starts[e]) + rel // ds._window_stride
    return idx if idx < int(ds._cum_n_starts[e + 1]) else None


class _Subset(torch.utils.data.Dataset):
    def __init__(self, ds, indices):
        self.ds, self.indices = ds, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[self.indices[i]]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    t = dict(np.load(args.targets))
    frames = t["frames"]
    split_names = [str(s) for s in t["split_names"]]

    from openwam.dataloader.registry import build_dataset
    from openwam.deploy import model_loader
    from openwam.deploy.denoise_schedule import make_schedule

    # The foundation checkpoint ships without YAM normalization stats; the probe
    # never un-normalizes actions, so a missing normalizer is harmless here.
    _orig_build_normalizer = model_loader._build_normalizer

    def _tolerant_build_normalizer(cfg, ckpt_dir):
        try:
            return _orig_build_normalizer(cfg, ckpt_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("normalizer unavailable (%s); continuing without it", exc)
            return None

    model_loader._build_normalizer = _tolerant_build_normalizer
    # Load on CPU, then move everything except the ~11 GB umT5 text encoder to the
    # GPU: the task prompt is encoded once on CPU and served from the deploy-style
    # prompt cache, so the probe needs ~15 GB of GPU memory instead of ~24 GB.
    cfg, arch = model_loader.load_from_checkpoint_dir(args.ckpt_dir, device="cpu", ckpt_name=args.ckpt_name)
    arch.eval()
    text_encoder = arch.video_backbone.text_encoder
    text_encoder._apply = lambda fn, recurse=True: text_encoder  # pin to CPU during the move
    arch.set_dtype_device(arch.dtype, device)
    del text_encoder._apply

    deploy_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "deploy.yaml")
    denoise_steps = int(args.denoise_steps or deploy_cfg.inference.denoise_steps)
    shift_action = float(arch.action_backbone.shift_action)
    shift_video = float(arch.video_backbone.shift_video)
    schedule = make_schedule(
        "sync",
        video_scheduler=arch.video_scheduler,
        action_scheduler=arch.action_scheduler,
        num_steps=denoise_steps,
        shift=shift_action,
        shift_video=shift_video,
    )
    first_step = schedule[:2]  # generate() runs exactly one joint forward
    action_num_frames = int(cfg.dataloader.num_frames)
    video_num_frames = len(range(0, action_num_frames, int(cfg.dataloader.video_stride)))
    logger.info("first step %s of %d; video frames %d, action frames %d", first_step, denoise_steps,
                video_num_frames, action_num_frames)

    # ── hooks: frame-0 token hidden states after patchify and after each block ──
    n0 = TOK_H * TOK_W
    captured: dict = {}
    vb = arch.video_backbone
    orig_prepare = vb.prepare

    def prepare_hook(**kw):
        state = orig_prepare(**kw)
        if state.grid_height * state.grid_width != n0:
            raise RuntimeError(f"unexpected token grid {state.grid_height}x{state.grid_width}")
        if 0 in layers:
            captured[0] = state.hidden_states[0, :n0].float()
        return state

    vb.prepare = prepare_hook
    driver = arch.mot_driver or arch.build_mot_driver()
    orig_step = driver.step

    def step_hook(layer_id, vstate, astate, **kw):
        vstate, astate = orig_step(layer_id, vstate, astate, **kw)
        if layer_id + 1 in layers:
            captured[layer_id + 1] = vstate.hidden_states[0, :n0].float()
        return vstate, astate

    driver.step = step_hook

    # ── datasets: the training config, colour jitter off ──
    dl_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
    dl_cfg["color_jitter"] = {"enabled": False}
    datasets = {"train": build_dataset(OmegaConf.create(dl_cfg), split="train"),
                "val": build_dataset(OmegaConf.create(dl_cfg), split="val")}
    which = {"fit": "train", "select": "train", "test": "val"}

    dim = int(vb.dim)
    k_out = {"edge": 16, "depth": 16, "gae": 4 * t["gae"].shape[1]}
    k_total = sum(k_out.values())
    # Accumulators sit on the GPU only while their split is being processed.
    mom: dict = {}
    gae_mom: dict = {}
    test_feats = {l: [] for l in layers}
    test_y = {g: [] for g in GROUPS}
    test_gae_x = []
    test_meta, test_canvases = [], []
    prompt_cache: dict = {}

    def ensure_prompt(prompt: str) -> None:
        """Encode a prompt on CPU once; generate() then takes it from the cache."""
        if prompt in prompt_cache:
            return
        from openwam.model.video_backbone.wan import encode as wan_encode

        threads = torch.get_num_threads()
        torch.set_num_threads(32)
        with torch.inference_mode():
            ctx, lens = wan_encode.encode_text(
                [prompt], tokenizer=vb._tokenizer, text_encoder=text_encoder, device=torch.device("cpu")
            )
        torch.set_num_threads(threads)
        prompt_cache[prompt] = (ctx.to(device), lens.to(device))
        logger.info("encoded prompt on CPU: %r", prompt)

    t0 = time.time()

    for split_code, split in enumerate(split_names):
        rows = np.nonzero(frames[:, 0] == split_code)[0]
        if args.limit:
            rows = rows[: args.limit]
        ds = datasets[which[split]]
        pairs = [(r, window_index(ds, int(frames[r, 1]), int(frames[r, 2]))) for r in rows]
        missing = [frames[r].tolist() for r, i in pairs if i is None]
        if missing:
            raise RuntimeError(f"{len(missing)} {split} frames have no dataset window, e.g. {missing[:3]}")
        loader = torch.utils.data.DataLoader(
            _Subset(ds, [i for _, i in pairs]), batch_size=None, shuffle=False,
            num_workers=args.num_workers, persistent_workers=False,
        )
        if split != "test":
            for l in layers:
                mom[(split, l)] = Moments(dim, k_total, device)
            gae_mom[split] = Moments(4 * t["gae"].shape[1], 32, device)
        for (r, _), sample in zip(pairs, loader):
            captured.clear()
            ensure_prompt(sample["prompt"])
            with torch.inference_mode():
                arch.generate(
                    first_step,
                    sample["prompt"],
                    first_frame_image=sample["first_frame_image"],
                    num_frames=video_num_frames,
                    action_num_frames=action_num_frames,
                    height=384,
                    width=320,
                    seed=42,
                    num_inference_steps=denoise_steps,
                    shift=shift_action,
                    dit_cache=None,
                    decode_video=False,
                    prompt_embed_cache=prompt_cache,
                    proprio=sample["proprio"],
                )
            ys = target_rows(t, r, args.tau)
            y_all = torch.from_numpy(np.concatenate([ys[g] for g in GROUPS], axis=1))
            gae_x = torch.from_numpy(ys["gae"])
            if split == "test":
                for l in layers:
                    test_feats[l].append(captured[l].half().cpu())
                for g in GROUPS:
                    test_y[g].append(ys[g])
                test_gae_x.append(ys["gae"])
                test_meta.append(frames[r].tolist())
                if len(test_canvases) < args.viz_frames:
                    test_canvases.append(np.asarray(sample["first_frame_image"][0]))
            else:
                for l in layers:
                    mom[(split, l)].add(captured[l], y_all.to(device))
                gae_mom[split].add(gae_x.to(device), torch.cat([torch.from_numpy(ys["edge"]),
                                                               torch.from_numpy(ys["depth"])], 1).to(device))
        if split != "test":
            for l in layers:
                mom[(split, l)].to("cpu")
            gae_mom[split].to("cpu")
            torch.cuda.empty_cache()
        logger.info("%s: %d frames done (%.0fs)", split, len(rows), time.time() - t0)

    # ── fit, select lambda per target group, evaluate on test ──
    slot = slot_of_token()
    offsets = {}
    o = 0
    for g in GROUPS:
        offsets[g] = slice(o, o + k_out[g])
        o += k_out[g]
    ys_test = {g: np.stack(test_y[g]) for g in GROUPS}
    fit_rows = np.nonzero(frames[:, 0] == split_names.index("fit"))[0]
    if args.limit:
        fit_rows = fit_rows[: args.limit]
    prior = static_prior(t, fit_rows, args.tau)
    dynamic = (prior["edge"] >= 0.1) & (prior["edge"] <= 0.9)
    saved = {"frames": np.array(test_meta), "label_edge": ys_test["edge"].astype(np.float16),
             "label_depth": ys_test["depth"].astype(np.float16), "dynamic": dynamic,
             "prior_edge": prior["edge"], "prior_depth": prior["depth"]}
    results = {"layers": {}, "settings": {
        "ckpt_dir": str(Path(args.ckpt_dir).resolve()), "ckpt_name": args.ckpt_name, "targets": args.targets,
        "tau": args.tau, "denoise_steps": denoise_steps, "first_step": [list(map(float, s)) for s in first_step],
        "n_frames": {s: int((frames[:, 0] == i).sum()) if not args.limit else args.limit
                     for i, s in enumerate(split_names)},
        "lambdas": LAMBDAS,
    }}
    viz_preds = {}
    for l in layers:
        m_fit, m_sel = mom[("fit", l)].to(device), mom[("select", l)].to(device)
        sols = fit_ridge(m_fit, LAMBDAS)
        chosen = {}
        for g in GROUPS:
            sl = offsets[g]
            sel_err = [float(moments_sse(m_sel, w[:, sl], b[sl], sl).sum()) for w, b in sols]
            chosen[g] = int(np.argmin(sel_err))
        m_fit.to("cpu"), m_sel.to("cpu")
        x_test = torch.stack(test_feats[l]).to(device=device, dtype=torch.float64)  # (N, 120, D)
        pred = {}
        for g in GROUPS:
            w, b = sols[chosen[g]]
            pred[g] = (x_test @ w[:, offsets[g]] + b[offsets[g]]).cpu().numpy()
        res = evaluate(pred, ys_test, slot, prior, dynamic)
        res["lambda"] = {g: LAMBDAS[i] for g, i in chosen.items()}
        results["layers"][str(l)] = res
        viz_preds[l] = pred["edge"][: args.viz_frames]
        saved[f"pred_edge_L{l}"] = pred["edge"].astype(np.float16)
        saved[f"pred_depth_L{l}"] = pred["depth"].astype(np.float16)
        logger.info(
            "layer %d: edge AP %.3f, dynamic-cell AP %.3f | depth R2 vs prior %.3f | gae R2 %.3f (vs prior %.3f)",
            l, res["edge_ap/all"], res["edge_ap_dynamic/all"], res["depth_r2_vs_prior/all"],
            res["gae_r2/all"], res["gae_r2_vs_prior/all"],
        )
        del x_test

    # Reference: the GAE latent itself as probe input (edge + depth only).
    gae_mom["fit"].to(device), gae_mom["select"].to(device)
    sols = fit_ridge(gae_mom["fit"], LAMBDAS)
    ref_offsets = {"edge": slice(0, 16), "depth": slice(16, 32)}
    x_test = torch.from_numpy(np.stack(test_gae_x)).to(device=device, dtype=torch.float64)
    pred = {}
    for g, sl in ref_offsets.items():
        sel_err = [float(moments_sse(gae_mom["select"], w[:, sl], b[sl], sl).sum()) for w, b in sols]
        w, b = sols[int(np.argmin(sel_err))]
        pred[g] = (x_test @ w[:, sl] + b[sl]).cpu().numpy()
    results["reference_gae_latent_input"] = evaluate(pred, ys_test, slot, prior, dynamic)
    viz_preds["gae_ref"] = pred["edge"][: args.viz_frames]
    saved["pred_edge_gae_ref"] = pred["edge"].astype(np.float16)
    saved["pred_depth_gae_ref"] = pred["depth"].astype(np.float16)
    # Reference: the static per-position prior, no model at all.
    prior_pred = {g: np.broadcast_to(prior[g], ys_test[g].shape) for g in GROUPS}
    results["reference_static_prior"] = evaluate(prior_pred, ys_test, slot, prior, dynamic)
    np.savez_compressed(out_path.with_suffix(".preds.npz"), **saved)

    best_layer = max(layers, key=lambda l: results["layers"][str(l)]["edge_ap/all"])
    results["best_edge_layer"] = best_layer
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s (%.0fs total)", out_path, time.time() - t0)

    # Visualization: canvas | label cells | layer-0 | best layer | GAE-latent reference.
    try:
        from PIL import Image

        def heat(rows):
            m = cells_from_tokens(rows, 4)[0]
            m = np.clip(m, 0, 1)
            img = (np.stack([m, m * 0.6, 1 - m], -1) * 255).astype(np.uint8)
            return np.asarray(Image.fromarray(img).resize((320, 384), Image.Resampling.NEAREST))

        rows_img = []
        cols = [l for l in (layers[0], best_layer) if l in viz_preds] + ["gae_ref"]
        for i, canvas in enumerate(test_canvases):
            lab = heat(ys_test["edge"][i])
            rows_img.append(np.concatenate([canvas, lab] + [heat(viz_preds[c][i]) for c in cols], axis=1))
        Image.fromarray(np.concatenate(rows_img, axis=0)).save(out_path.with_suffix(".png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("visualization failed: %s", exc)


if __name__ == "__main__":
    main()
