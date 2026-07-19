#!/usr/bin/env python3
"""Fit per-method measurement noise calibration from known-distance clips.

This is the deployable calibration layer between raw depth candidates and an
EKF/UKF measurement model. It uses tape-measured static clips to estimate, for
each depth method and image/depth region:

  bias_m:          median residual after d0 reprojection
  sigma_m:         robust residual scale for Rz = sigma_m^2
  outlier_rate:    empirical residual outlier probability
  valid_rate:      candidate availability in that region

The output intentionally calibrates measurement noise, not disparity d0. d0 is
loaded from calibrate_disparity_offset.py output and kept global for the current
code/calibration pair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

try:
    from .dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from .fit_method_calibration import (
        _known_z_calibration_allowed,
        _metadata_float,
        resolve_clips,
    )
    from .manifest import DatasetClip
    from .reproject import (
        load_reprojection_model,
        method_pixel_columns,
        reproject_row,
    )
except ImportError:  # pragma: no cover - direct script execution
    from dataset import METHOD_COLUMNS, load_legacy_sequences, resolve_method_allowlist
    from fit_method_calibration import (
        _known_z_calibration_allowed,
        _metadata_float,
        resolve_clips,
    )
    from manifest import DatasetClip
    from reproject import (
        load_reprojection_model,
        method_pixel_columns,
        reproject_row,
    )


INF_LABEL = "inf"


@dataclass(frozen=True)
class NoiseCalibrationConfig:
    train_split: str = "train"
    min_count: int = 30
    min_sigma: float = 0.015
    mad_sigma_scale: float = 1.4826
    outlier_threshold_m: float = 0.15
    distance_bin_m: float = 1.0
    radial_edges_px: Tuple[float, ...] = (0.0, 200.0, 400.0, math.inf)
    image_width: float = 1440.0
    image_height: float = 1080.0
    grid_x_bins: int = 3
    grid_y_bins: int = 3
    shrink_count: int = 100
    min_disparity: float = 0.1


@dataclass
class BinAccumulator:
    total_count: int = 0
    residuals: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.residuals is None:
            self.residuals = []

    @property
    def valid_count(self) -> int:
        return len(self.residuals)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on", "static"}:
        return True
    if token in {"0", "false", "no", "off", "dynamic"}:
        return False
    return default


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values: Sequence[float], center: float | None = None) -> float | None:
    if not values:
        return None
    med = _median(values) if center is None else center
    if med is None:
        return None
    return _median([abs(value - med) for value in values])


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _parse_edges(raw: str) -> Tuple[float, ...]:
    edges: List[float] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"inf", "+inf", "infinity"}:
            edges.append(math.inf)
        else:
            edges.append(float(token))
    if len(edges) < 2:
        raise argparse.ArgumentTypeError("at least two radial edges are required")
    if edges[0] != 0.0:
        raise argparse.ArgumentTypeError("radial edges must start at 0")
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            raise argparse.ArgumentTypeError("radial edges must be strictly increasing")
    return tuple(edges)


def _edge_label(value: float) -> str:
    if math.isinf(value):
        return INF_LABEL
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _distance_bin(known_z: float, width: float) -> str:
    if width <= 0:
        return f"z_{known_z:g}"
    lo = math.floor(known_z / width) * width
    hi = lo + width
    return f"z_{lo:g}_{hi:g}"


def _radial_bin(r_px: float, edges: Sequence[float]) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if r_px >= lo and r_px < hi:
            return f"r_{_edge_label(lo)}_{_edge_label(hi)}"
    return f"r_{_edge_label(edges[-2])}_{_edge_label(edges[-1])}"


def _grid_bin(u: float, v: float, cfg: NoiseCalibrationConfig) -> str:
    x_bins = max(1, cfg.grid_x_bins)
    y_bins = max(1, cfg.grid_y_bins)
    x = int(math.floor((u / max(cfg.image_width, 1.0)) * x_bins))
    y = int(math.floor((v / max(cfg.image_height, 1.0)) * y_bins))
    x = min(max(0, x), x_bins - 1)
    y = min(max(0, y), y_bins - 1)
    return f"grid_x{x}_y{y}"


def _load_mapping_file(path: str | Path) -> Dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(f"YAML labels require PyYAML: {source}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"known-z label file must contain a mapping: {source}")
    return data


def _run_id_tokens(*values: object) -> List[str]:
    tokens: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            tokens.append(text)
        tokens.extend(re.findall(r"\d{6,}", text))
    return tokens


def _register_label(labels: Dict[str, Dict[str, Any]], item: Dict[str, Any]) -> None:
    known_z = _metadata_float(item, ("known_z_m", "known_z", "known_distance_m"))
    if known_z <= 0.0:
        return

    metadata: Dict[str, Any] = {"known_z": known_z}
    if "static" in item or "is_static" in item:
        metadata["static"] = _safe_bool(item.get("static", item.get("is_static")), False)
    if "known_z_calibration" in item:
        metadata["known_z_calibration"] = _safe_bool(item.get("known_z_calibration"), False)
    elif "use_for_calibration" in item:
        metadata["known_z_calibration"] = _safe_bool(item.get("use_for_calibration"), False)
    elif "fit_ok" in item:
        metadata["known_z_calibration"] = _safe_bool(item.get("fit_ok"), False)

    for token in _run_id_tokens(
        item.get("name"),
        item.get("run_id"),
        item.get("id"),
        item.get("csv"),
    ):
        labels[token] = dict(metadata)


def load_known_z_labels(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    """Load optional clip-level known-z labels.

    The file may be the same JSON/YAML used as a manifest. Entries under
    ``clips`` are matched by ``name``, ``run_id`` or the numeric run token in the
    CSV path. A ``labels`` mapping is also accepted for compact ad-hoc use.
    """

    if path is None:
        return {}
    data = _load_mapping_file(path)
    labels: Dict[str, Dict[str, Any]] = {}

    raw_labels = data.get("labels")
    if isinstance(raw_labels, dict):
        for key, value in raw_labels.items():
            if isinstance(value, dict):
                item = {"name": key, **value}
            else:
                item = {"name": key, "known_z": value, "static": True}
            _register_label(labels, item)

    raw_clips = data.get("clips")
    if isinstance(raw_clips, list):
        for item in raw_clips:
            if isinstance(item, dict):
                _register_label(labels, item)

    return labels


def _label_metadata_for_clip(
    clip: DatasetClip,
    labels: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    for token in _run_id_tokens(clip.name, clip.csv.stem, clip.csv.parent.name, clip.csv):
        label = labels.get(token)
        if label:
            return dict(label)
    return {}


def _stats(
    acc: BinAccumulator,
    cfg: NoiseCalibrationConfig,
    *,
    parent: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    residuals = acc.residuals
    bias = _median(residuals)
    mad = _mad(residuals, bias)
    std = pstdev(residuals) if len(residuals) > 1 else 0.0
    robust_sigma = cfg.mad_sigma_scale * (mad or 0.0)
    sigma = max(cfg.min_sigma, robust_sigma, std * 0.5)
    centered_abs = [abs(value - (bias or 0.0)) for value in residuals]
    outlier_count = sum(1 for value in centered_abs if value > cfg.outlier_threshold_m)
    valid_rate = acc.valid_count / acc.total_count if acc.total_count > 0 else 0.0
    outlier_rate = outlier_count / acc.valid_count if acc.valid_count > 0 else None
    calibrated_confidence = valid_rate * (1.0 - (outlier_rate or 0.0))

    result: Dict[str, Any] = {
        "total_count": acc.total_count,
        "valid_count": acc.valid_count,
        "valid_rate": valid_rate,
        "bias_median": bias,
        "bias_mean": mean(residuals) if residuals else None,
        "mad": mad,
        "std": std if residuals else None,
        "sigma": sigma if residuals else None,
        "abs_p68": _percentile([abs(value) for value in residuals], 68.0),
        "abs_p95": _percentile([abs(value) for value in residuals], 95.0),
        "outlier_threshold_m": cfg.outlier_threshold_m,
        "outlier_count": outlier_count,
        "outlier_rate": outlier_rate,
        "calibrated_confidence": calibrated_confidence,
        "weight_hint": calibrated_confidence / max(sigma * sigma, 1e-9) if residuals else 0.0,
        "sufficient": acc.valid_count >= cfg.min_count,
    }

    if parent and residuals:
        parent_bias = parent.get("bias_median")
        parent_sigma = parent.get("sigma")
        if parent_bias is not None and parent_sigma is not None:
            w = acc.valid_count / (acc.valid_count + max(0, cfg.shrink_count))
            result["shrink_weight"] = w
            result["bias_shrunk"] = w * bias + (1.0 - w) * parent_bias
            result["sigma_shrunk"] = math.sqrt(w * sigma * sigma + (1.0 - w) * parent_sigma * parent_sigma)
    return result


def _make_accumulators(methods: Iterable[str]) -> Dict[str, Dict[str, DefaultDict[str, BinAccumulator]]]:
    return {
        method: {
            "global": defaultdict(BinAccumulator),
            "by_distance": defaultdict(BinAccumulator),
            "by_distance_radial": defaultdict(BinAccumulator),
            "by_distance_grid": defaultdict(BinAccumulator),
        }
        for method in methods
    }


def _add_total(
    groups: Dict[str, DefaultDict[str, BinAccumulator]],
    keys: Dict[str, str],
) -> None:
    for group_name, key in keys.items():
        groups[group_name][key].total_count += 1


def _add_residual(
    groups: Dict[str, DefaultDict[str, BinAccumulator]],
    keys: Dict[str, str],
    residual: float,
) -> None:
    for group_name, key in keys.items():
        groups[group_name][key].residuals.append(residual)


def fit_measurement_noise_calibration(
    clips: Sequence[DatasetClip],
    *,
    calib: str | Path,
    offset_fit: str | Path | None,
    d0_override: float | None,
    cfg: NoiseCalibrationConfig,
    method_names: Sequence[str] | str | None = None,
    known_z_labels: Dict[str, Dict[str, Any]] | None = None,
    known_z_label_source: str | Path | None = None,
) -> Dict[str, Any]:
    method_allowlist = resolve_method_allowlist(method_names)
    enabled = set(method_allowlist) if method_allowlist is not None else None
    methods = [(name, column) for name, column in METHOD_COLUMNS if name != "mono" and (enabled is None or name in enabled)]
    method_names_order = [name for name, _column in methods]
    reprojection_model = load_reprojection_model(calib, offset_fit, d0_override)

    accs = _make_accumulators(method_names_order)
    used_clips: List[Dict[str, Any]] = []
    skipped_clips: List[Dict[str, Any]] = []
    known_frame_count = 0
    labels = known_z_labels or {}

    for clip in clips:
        if clip.split != cfg.train_split:
            skipped_clips.append({"csv": str(clip.csv), "split": clip.split, "reason": "split"})
            continue
        sequences = load_legacy_sequences(clip.csv, metadata_path=clip.metadata)
        label_metadata = _label_metadata_for_clip(clip, labels)
        clip_frames = 0
        clip_known_values: List[float] = []
        blocked = False
        had_known = False
        for sequence in sequences:
            sequence_metadata = dict(label_metadata)
            for key, value in sequence.metadata.items():
                if value is not None and value != "":
                    sequence_metadata[key] = value
            known_z = _metadata_float(sequence_metadata, ("known_z_m", "known_z", "known_distance_m"))
            if known_z <= 0.0:
                continue
            had_known = True
            if not _known_z_calibration_allowed(sequence_metadata):
                blocked = True
                continue
            clip_known_values.append(known_z)
            for row in sequence.rows:
                known_frame_count += 1
                clip_frames += 1
                points = reproject_row(row, reprojection_model, methods, min_disparity=cfg.min_disparity)
                dist_key = _distance_bin(known_z, cfg.distance_bin_m)

                for method, _column in methods:
                    u_col, v_col = method_pixel_columns(method)
                    u = _safe_float(row.get(u_col))
                    v = _safe_float(row.get(v_col))
                    if u is None or v is None:
                        # Keep global/distance denominator even when position
                        # columns are missing, but skip position-specific bins.
                        keys = {
                            "global": "all",
                            "by_distance": dist_key,
                        }
                    else:
                        intr = reprojection_model.intrinsics
                        r_px = math.hypot(u - intr.cx, v - intr.cy)
                        radial_key = f"{dist_key}|{_radial_bin(r_px, cfg.radial_edges_px)}"
                        grid_key = f"{dist_key}|{_grid_bin(u, v, cfg)}"
                        keys = {
                            "global": "all",
                            "by_distance": dist_key,
                            "by_distance_radial": radial_key,
                            "by_distance_grid": grid_key,
                        }
                    _add_total(accs[method], keys)
                    pt = points[method]
                    if pt.valid:
                        _add_residual(accs[method], keys, pt.z - known_z)

        if clip_frames > 0:
            used_clips.append(
                {
                    "csv": str(clip.csv),
                    "metadata": str(clip.metadata) if clip.metadata else None,
                    "known_z_label_source": str(known_z_label_source) if label_metadata and known_z_label_source else None,
                    "split": clip.split,
                    "name": clip.name,
                    "frames": clip_frames,
                    "known_z_values": sorted(set(clip_known_values)),
                }
            )
        else:
            reason = "known_z_not_static_for_calibration" if had_known and blocked else "missing_known_z"
            skipped_clips.append({"csv": str(clip.csv), "split": clip.split, "reason": reason})

    methods_out: Dict[str, Any] = {}
    for method in method_names_order:
        global_acc = accs[method]["global"]["all"]
        global_stats = _stats(global_acc, cfg)
        method_out: Dict[str, Any] = {"global": global_stats}
        for group_name in ("by_distance", "by_distance_radial", "by_distance_grid"):
            group_out: Dict[str, Any] = {}
            for key, acc in sorted(accs[method][group_name].items()):
                parent = global_stats
                if group_name != "by_distance":
                    dist_key = key.split("|", 1)[0]
                    dist_acc = accs[method]["by_distance"].get(dist_key)
                    if dist_acc is not None and dist_acc.valid_count > 0:
                        parent = _stats(dist_acc, cfg, parent=global_stats)
                group_out[key] = _stats(acc, cfg, parent=parent)
            method_out[group_name] = group_out
        methods_out[method] = method_out

    return {
        "version": 1,
        "type": "measurement_noise_calibration",
        "model": "z_corrected = z_d0_reprojected - bias; Rz = sigma^2; Student-t handles residual outliers",
        "config": {
            "train_split": cfg.train_split,
            "min_count": cfg.min_count,
            "min_sigma": cfg.min_sigma,
            "mad_sigma_scale": cfg.mad_sigma_scale,
            "outlier_threshold_m": cfg.outlier_threshold_m,
            "distance_bin_m": cfg.distance_bin_m,
            "radial_edges_px": [_edge_label(edge) for edge in cfg.radial_edges_px],
            "image_width": cfg.image_width,
            "image_height": cfg.image_height,
            "grid_x_bins": cfg.grid_x_bins,
            "grid_y_bins": cfg.grid_y_bins,
            "shrink_count": cfg.shrink_count,
            "method_allowlist": list(method_allowlist) if method_allowlist is not None else None,
            "known_z_labels": str(known_z_label_source) if known_z_label_source else None,
        },
        "reprojection": {
            "calib": str(calib),
            "offset_fit": str(offset_fit) if offset_fit else None,
            "d0": reprojection_model.d0,
            "fB": reprojection_model.fB,
            "fx": reprojection_model.intrinsics.fx,
            "fy": reprojection_model.intrinsics.fy,
            "cx": reprojection_model.intrinsics.cx,
            "cy": reprojection_model.intrinsics.cy,
            "baseline_m": reprojection_model.intrinsics.baseline_m,
        },
        "known_frame_count": known_frame_count,
        "used_clips": used_clips,
        "skipped_clips": skipped_clips,
        "methods": methods_out,
    }


def write_calibration(path: str | Path, calibration: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_report(calibration: Dict[str, Any], *, top: int = 12) -> None:
    print(
        "known_frames={frames} used_clips={clips} methods={methods} d0={d0:.3f} fB={fB:.3f}".format(
            frames=calibration["known_frame_count"],
            clips=len(calibration["used_clips"]),
            methods=len(calibration["methods"]),
            d0=float(calibration["reprojection"]["d0"]),
            fB=float(calibration["reprojection"]["fB"]),
        )
    )
    print("method,total,valid,valid_rate,bias,sigma,outlier,confidence,weight_hint")
    rows = []
    for method, block in calibration["methods"].items():
        stats = block["global"]
        rows.append((method, stats))
    rows.sort(key=lambda item: (-float(item[1].get("weight_hint", 0.0)), item[0]))
    for method, stats in rows[:top]:
        print(
            "{method},{total},{valid},{valid_rate},{bias},{sigma},{outlier},{conf},{weight}".format(
                method=method,
                total=stats["total_count"],
                valid=stats["valid_count"],
                valid_rate=_fmt(stats["valid_rate"]),
                bias=_fmt(stats["bias_median"]),
                sigma=_fmt(stats["sigma"]),
                outlier=_fmt(stats["outlier_rate"]),
                conf=_fmt(stats["calibrated_confidence"]),
                weight=_fmt(stats["weight_hint"]),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV file(s), or one dataset manifest YAML/JSON")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--metadata", help="Optional metadata YAML for one CSV input")
    parser.add_argument("--calib", required=True, help="stereo_calib.yaml used for d0 reprojection")
    parser.add_argument("--offset-fit", help="disparity_offset_fit.json from calibrate_disparity_offset.py")
    parser.add_argument("--d0", type=float, help="Override disparity offset in pixels")
    parser.add_argument("--known-z-labels", help="Optional JSON/YAML mapping or manifest with known_z/static labels")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--methods", default="p0p1_ncc_xfeat", help="Method allowlist/preset")
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--min-sigma", type=float, default=0.015)
    parser.add_argument("--outlier-threshold-m", type=float, default=0.15)
    parser.add_argument("--distance-bin-m", type=float, default=1.0)
    parser.add_argument("--radial-edges-px", type=_parse_edges, default=_parse_edges("0,200,400,inf"))
    parser.add_argument("--image-width", type=float, default=1440.0)
    parser.add_argument("--image-height", type=float, default=1080.0)
    parser.add_argument("--grid-x-bins", type=int, default=3)
    parser.add_argument("--grid-y-bins", type=int, default=3)
    parser.add_argument("--shrink-count", type=int, default=100)
    parser.add_argument("--min-disparity", type=float, default=0.1)
    parser.add_argument("--top", type=int, default=12, help="Rows to print in the console report")
    args = parser.parse_args()

    clips = resolve_clips(args.inputs, metadata=args.metadata)
    cfg = NoiseCalibrationConfig(
        train_split=args.train_split,
        min_count=args.min_count,
        min_sigma=args.min_sigma,
        outlier_threshold_m=args.outlier_threshold_m,
        distance_bin_m=args.distance_bin_m,
        radial_edges_px=args.radial_edges_px,
        image_width=args.image_width,
        image_height=args.image_height,
        grid_x_bins=args.grid_x_bins,
        grid_y_bins=args.grid_y_bins,
        shrink_count=args.shrink_count,
        min_disparity=args.min_disparity,
    )
    calibration = fit_measurement_noise_calibration(
        clips,
        calib=args.calib,
        offset_fit=args.offset_fit,
        d0_override=args.d0,
        cfg=cfg,
        method_names=args.methods,
        known_z_labels=load_known_z_labels(args.known_z_labels),
        known_z_label_source=args.known_z_labels,
    )
    write_calibration(args.output, calibration)
    print_report(calibration, top=args.top)
    print(f"wrote {args.output}")
    if calibration["known_frame_count"] <= 0:
        print("no known-distance frames found; check metadata/static labels")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
