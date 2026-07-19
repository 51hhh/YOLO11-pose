"""Optional neural feature backends for offline CPU/GPU probes.

The realtime NX path should use TensorRT engines with the same extractor/matcher
contract. These Python backends are for local correctness and visualization.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .common import FeatureSet, RawMatch, TimedResult


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class BackendConfig:
    name: str
    top_k: int = 32
    device: str = "cpu"
    xfeat_repo: Optional[str] = None
    allow_torch_hub: bool = False
    use_lightglue: bool = False


def _load_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise BackendUnavailable("PyTorch is required for neural feature probes") from exc
    return torch


def _to_torch_image(torch: Any, image_bgr: np.ndarray, device: str):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _feature_from_lightglue_dict(data: Dict[str, Any], image_size: tuple[int, int]) -> FeatureSet:
    keypoints = _tensor_to_numpy(data.get("keypoints"))
    descriptors = _tensor_to_numpy(data.get("descriptors"))
    scores = _tensor_to_numpy(data.get("keypoint_scores", data.get("scores")))

    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if descriptors.ndim == 3:
        descriptors = descriptors[0]
    # Some LightGlue versions return descriptors as D x N.
    if descriptors.ndim == 2 and keypoints.ndim == 2 and descriptors.shape[0] != keypoints.shape[0]:
        descriptors = descriptors.T
    if scores.ndim > 1:
        scores = scores.reshape(-1)
    return FeatureSet(keypoints, descriptors, scores if scores.size else None, image_size)


def _matches_from_lightglue_dict(data: Dict[str, Any]) -> List[RawMatch]:
    matches = _tensor_to_numpy(data.get("matches"))
    scores = _tensor_to_numpy(data.get("scores", data.get("matching_scores")))
    if matches.ndim == 3:
        matches = matches[0]
    if matches.size:
        out: List[RawMatch] = []
        for i, pair in enumerate(matches.reshape(-1, 2)):
            score = float(scores.reshape(-1)[i]) if scores.size > i else 1.0
            out.append(RawMatch(int(pair[0]), int(pair[1]), score))
        return out

    matches0 = _tensor_to_numpy(data.get("matches0"))
    scores0 = _tensor_to_numpy(data.get("matching_scores0", data.get("scores0")))
    if matches0.ndim > 1:
        matches0 = matches0.reshape(-1)
    if scores0.ndim > 1:
        scores0 = scores0.reshape(-1)
    out = []
    for qi, ti in enumerate(matches0.tolist()):
        if int(ti) < 0:
            continue
        score = float(scores0[qi]) if scores0.size > qi else 1.0
        out.append(RawMatch(qi, int(ti), score))
    return out


class NeuralBackend:
    def run(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> TimedResult:
        raise NotImplementedError


class XFeatBackend(NeuralBackend):
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.torch = _load_torch()
        if cfg.xfeat_repo:
            repo = str(Path(cfg.xfeat_repo).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)
        try:
            from modules.xfeat import XFeat  # type: ignore
            self.model = XFeat().eval().to(cfg.device)
        except Exception as local_exc:
            if not cfg.allow_torch_hub:
                raise BackendUnavailable(
                    "XFeat not importable. Pass --xfeat-repo /path/to/accelerated_features "
                    "or enable --allow-torch-hub when network is allowed."
                ) from local_exc
            self.model = self.torch.hub.load(
                "verlab/accelerated_features", "XFeat", pretrained=True
            ).eval().to(cfg.device)

    def _extract(self, image_bgr: np.ndarray) -> FeatureSet:
        tensor = _to_torch_image(self.torch, image_bgr, self.cfg.device)
        with self.torch.no_grad():
            output = self.model.detectAndCompute(tensor, top_k=self.cfg.top_k)[0]
        return FeatureSet(
            _tensor_to_numpy(output.get("keypoints")),
            _tensor_to_numpy(output.get("descriptors")),
            _tensor_to_numpy(output.get("scores")),
            image_bgr.shape[:2],
        )

    def run(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> TimedResult:
        t0 = time.perf_counter()
        left = self._extract(left_bgr)
        right = self._extract(right_bgr)
        return TimedResult(
            left,
            right,
            timings_ms={"extract": (time.perf_counter() - t0) * 1000.0},
            notes="xfeat_descriptor_nn",
        )


class LightGlueExtractorBackend(NeuralBackend):
    def __init__(self, cfg: BackendConfig, extractor_name: str):
        self.cfg = cfg
        self.extractor_name = extractor_name
        self.torch = _load_torch()
        try:
            from lightglue import ALIKED, LightGlue, SuperPoint  # type: ignore
        except Exception as exc:
            raise BackendUnavailable("lightglue package is required") from exc

        if extractor_name == "aliked":
            self.extractor = ALIKED(max_num_keypoints=cfg.top_k).eval().to(cfg.device)
            self.matcher = LightGlue(features="aliked").eval().to(cfg.device) if cfg.use_lightglue else None
        elif extractor_name == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=cfg.top_k).eval().to(cfg.device)
            self.matcher = LightGlue(features="superpoint").eval().to(cfg.device)
        else:
            raise ValueError(extractor_name)

    def run(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> TimedResult:
        left_tensor = _to_torch_image(self.torch, left_bgr, self.cfg.device)
        right_tensor = _to_torch_image(self.torch, right_bgr, self.cfg.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            left_raw = self.extractor.extract(left_tensor)
            right_raw = self.extractor.extract(right_tensor)
            matches: List[RawMatch] = []
            if self.matcher is not None:
                matched = self.matcher({"image0": left_raw, "image1": right_raw})
                matches = _matches_from_lightglue_dict(matched)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return TimedResult(
            _feature_from_lightglue_dict(left_raw, left_bgr.shape[:2]),
            _feature_from_lightglue_dict(right_raw, right_bgr.shape[:2]),
            matches=matches,
            timings_ms={"extract_match": elapsed},
            notes=f"{self.extractor_name}_{'lightglue' if self.matcher is not None else 'descriptor_nn'}",
        )


def create_backend(cfg: BackendConfig) -> NeuralBackend:
    name = cfg.name.lower().replace("-", "_")
    if name == "xfeat":
        return XFeatBackend(cfg)
    if name == "aliked":
        return LightGlueExtractorBackend(cfg, "aliked")
    if name in {"superpoint_lightglue", "superpoint+lightglue", "superpoint"}:
        forced = BackendConfig(**{**cfg.__dict__, "use_lightglue": True})
        return LightGlueExtractorBackend(forced, "superpoint")
    raise ValueError(f"unknown neural backend: {cfg.name}")
