"""The policy server's inference path, in-process, for offline evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Client image keys the server maps onto camera_layout positions, in order.
_CLIENT_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")


class EnginePolicy:
    """A checkpoint served exactly as the policy server serves it, plus the video it never sends.

    Built through :func:`openwam.deploy.server.build_server_from_config` with
    ``configs/deploy.yaml`` (plus dotted ``overrides``), so checkpoint loading,
    denoising, compile, DiT cache, observation preprocessing and normalization
    all match deployment. The one change is ``optimization.decode_video=true``:
    decoding runs after denoising, so it does not alter the actions.
    """

    def __init__(
        self,
        ckpt_dir: str | Path,
        ckpt_name: str,
        *,
        device: str = "cuda:0",
        overrides: Sequence[str] = (),
        seed: int = 42,
    ):
        from omegaconf import OmegaConf

        from openwam.deploy.obs_preprocess import ObsPreprocessor
        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

        cfg = _load_deploy_yaml()
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        OmegaConf.update(cfg, "optimization.decode_video", True, merge=False)
        server = build_server_from_config(cfg=cfg, ckpt_dir=str(ckpt_dir), device=device, ckpt_name=ckpt_name)
        self.cfg = server.cfg
        self._engine = server.engine
        self._preprocessor = ObsPreprocessor.from_cfg(server.cfg, server.engine)
        if not self._preprocessor.multiview:
            raise NotImplementedError("open-loop evaluation supports multiview checkpoints only")
        self.cameras: Tuple[Optional[str], ...] = tuple(self._preprocessor.camera_layout[:3])
        self.chunk_len = int(OmegaConf.select(self.cfg, "inference.num_frames")) - 1
        self.video_stride = int(OmegaConf.select(self.cfg, "dataloader.video_stride", default=1))
        self.horizon = int(OmegaConf.select(self.cfg, "inference.inference_horizon", default=None) or self.chunk_len)
        self.seed = int(seed)

    def _observation(self, images: Mapping[str, Image.Image], state: np.ndarray, prompt: str) -> dict:
        client = {key: (images.get(camera) if camera else None) for key, camera in zip(_CLIENT_KEYS, self.cameras)}
        return self._preprocessor.preprocess({"images": client, "prompt": prompt, "state": state})

    def predict(
        self, images: Mapping[str, Image.Image], state: np.ndarray, prompt: str
    ) -> Tuple[np.ndarray, Optional[List[Image.Image]]]:
        """One chunk, from the conditions the server's ``WAMPolicy`` builds for the same observation."""
        obs = self._observation(images, state, prompt)
        result = self._engine.generate(
            {
                "observation": obs,
                "first_frame_image": [obs["image"]],
                "prompt": obs["prompt"],
                "proprio": obs["state"],
                "seed": self.seed,  # the engine's own default is 42, so the default matches the server
            }
        )
        return np.asarray(result["actions"], dtype=np.float64), result["video"]

    def canvas(self, images: Mapping[str, Image.Image], state: np.ndarray) -> Image.Image:
        return self._observation(images, state, "")["image"]

    def settings(self) -> dict:
        from omegaconf import OmegaConf

        keys = {
            "denoise_steps": "inference.denoise_steps",
            "denoise_mode": "inference.denoise_mode",
            "dit_cache": "optimization.dit_cache.enabled",
            "compile": "optimization.compile.enabled",
        }
        settings = {name: OmegaConf.select(self.cfg, key, default=None) for name, key in keys.items()}
        return {**settings, "seed": self.seed, "chunk_len": self.chunk_len, "video_stride": self.video_stride}


__all__ = ["EnginePolicy"]
