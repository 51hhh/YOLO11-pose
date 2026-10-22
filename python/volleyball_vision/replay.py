"""对 v2.1 真实 trial 做确定性回放检查和影子轨迹评估。"""

from __future__ import annotations

from bisect import bisect_left
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from .config import ResolvedConfig, canonical_json, find_project_root
from .dataset import DatasetError, validate_trial


DERIVED_SCHEMA_VERSION = 1
SHADOW_REQUIRED_COLUMNS = ("timestamp_s", "x_m", "y_m", "z_m")
COVARIANCE_COLUMNS = (
    "cov_xx_m2",
    "cov_xy_m2",
    "cov_xz_m2",
    "cov_yy_m2",
    "cov_yz_m2",
    "cov_zz_m2",
)
MATCH_COLUMNS = (
    "reference_sample_index",
    "reference_timestamp_s",
    "shadow_sample_index",
    "shadow_timestamp_s",
    "delta_t_s",
    "error_x_m",
    "error_y_m",
    "error_z_m",
    "error_3d_m",
    "mahalanobis_sq",
)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    output_directory: Path
    fingerprint_sha256: str
    metrics: Mapping[str, Any]
    reused: bool = False


@dataclass(frozen=True, slots=True)
class _Sample:
    index: int
    timestamp_s: float
    position_m: tuple[float, float, float]
    covariance_m2: tuple[float, float, float, float, float, float] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"缺少文件: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"JSON 无效 {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DatasetError(f"CSV 缺少表头: {path}")
            return list(reader.fieldnames), list(reader)
    except FileNotFoundError as error:
        raise DatasetError(f"缺少文件: {path}") from error


def _finite(value: str, *, field: str, allow_blank: bool = False) -> float | None:
    if value == "" and allow_blank:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{field} 必须是数字") from error
    if not math.isfinite(result):
        raise DatasetError(f"{field} 必须为有限数")
    return result


def _load_samples(path: Path, *, source_name: str) -> list[_Sample]:
    columns, rows = _read_csv(path)
    missing = [name for name in SHADOW_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise DatasetError(f"{source_name} 缺少列: {', '.join(missing)}")
    covariance_present = [name in columns for name in COVARIANCE_COLUMNS]
    if any(covariance_present) and not all(covariance_present):
        raise DatasetError(f"{source_name} 协方差列必须全部提供或全部省略")

    samples: list[_Sample] = []
    previous_time = -math.inf
    for row_number, row in enumerate(rows, start=2):
        timestamp = _finite(
            row["timestamp_s"],
            field=f"{source_name}:{row_number}.timestamp_s",
        )
        assert timestamp is not None
        if timestamp <= previous_time:
            raise DatasetError(f"{source_name} timestamp_s 必须严格递增")
        previous_time = timestamp
        position_values = tuple(
            _finite(row[name], field=f"{source_name}:{row_number}.{name}")
            for name in ("x_m", "y_m", "z_m")
        )
        assert all(value is not None for value in position_values)
        covariance = None
        if all(covariance_present):
            parsed = tuple(
                _finite(
                    row[name],
                    field=f"{source_name}:{row_number}.{name}",
                    allow_blank=True,
                )
                for name in COVARIANCE_COLUMNS
            )
            if any(value is not None for value in parsed):
                if not all(value is not None for value in parsed):
                    raise DatasetError(
                        f"{source_name}:{row_number} 协方差不得只填写部分元素"
                    )
                covariance = tuple(float(value) for value in parsed)  # type: ignore[arg-type]
        sample_index_text = row.get("sample_index", "")
        try:
            sample_index = int(sample_index_text) if sample_index_text else len(samples)
        except ValueError as error:
            raise DatasetError(
                f"{source_name}:{row_number}.sample_index 必须是整数"
            ) from error
        samples.append(
            _Sample(
                index=sample_index,
                timestamp_s=timestamp,
                position_m=tuple(float(value) for value in position_values),  # type: ignore[arg-type]
                covariance_m2=covariance,
            )
        )
    return samples


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    """为延迟与误差提供稳定、无第三方依赖的摘要。"""

    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _optional_timestamp_ns(row: Mapping[str, str], name: str) -> int | None:
    value = row.get(name, "")
    if value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"camera_frames.{name} 必须是整数纳秒") from error
    if parsed < 0:
        raise DatasetError(f"camera_frames.{name} 不得为负")
    return parsed


def _mahalanobis_sq(
    error: tuple[float, float, float],
    covariance: tuple[float, float, float, float, float, float] | None,
) -> float | None:
    if covariance is None:
        return None
    xx, xy, xz, yy, yz, zz = covariance
    if not all(math.isfinite(value) for value in covariance) or xx <= 0.0:
        return None
    # Cholesky 分解同时验证 3x3 协方差正定，避免引入重型数值依赖。
    l00 = math.sqrt(xx)
    l10 = xy / l00
    l20 = xz / l00
    diagonal_1 = yy - l10 * l10
    if diagonal_1 <= 0.0:
        return None
    l11 = math.sqrt(diagonal_1)
    l21 = (yz - l20 * l10) / l11
    diagonal_2 = zz - l20 * l20 - l21 * l21
    if diagonal_2 <= 0.0:
        return None
    l22 = math.sqrt(diagonal_2)
    y0 = error[0] / l00
    y1 = (error[1] - l10 * y0) / l11
    y2 = (error[2] - l20 * y0 - l21 * y1) / l22
    result = y0 * y0 + y1 * y1 + y2 * y2
    return result if math.isfinite(result) else None


def _nearest_matches(
    references: list[_Sample],
    shadow: list[_Sample],
    max_time_delta_s: float,
) -> tuple[list[dict[str, Any]], set[int]]:
    times = [sample.timestamp_s for sample in shadow]
    matches: list[dict[str, Any]] = []
    used_shadow: set[int] = set()
    for reference in references:
        insertion = bisect_left(times, reference.timestamp_s)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(shadow) and index not in used_shadow
        ]
        if not candidates:
            continue
        shadow_index = min(
            candidates,
            key=lambda index: (abs(times[index] - reference.timestamp_s), index),
        )
        candidate = shadow[shadow_index]
        delta_t = candidate.timestamp_s - reference.timestamp_s
        if abs(delta_t) > max_time_delta_s:
            continue
        error = tuple(
            candidate.position_m[axis] - reference.position_m[axis]
            for axis in range(3)
        )
        error_3d = math.sqrt(sum(value * value for value in error))
        mahalanobis_sq = _mahalanobis_sq(error, candidate.covariance_m2)
        matches.append(
            {
                "reference_sample_index": reference.index,
                "reference_timestamp_s": reference.timestamp_s,
                "shadow_sample_index": candidate.index,
                "shadow_timestamp_s": candidate.timestamp_s,
                "delta_t_s": delta_t,
                "error_x_m": error[0],
                "error_y_m": error[1],
                "error_z_m": error[2],
                "error_3d_m": error_3d,
                "mahalanobis_sq": mahalanobis_sq,
            }
        )
        used_shadow.add(shadow_index)
    return matches, used_shadow


def _comparison_metrics(
    references: list[_Sample],
    shadow: list[_Sample],
    matches: list[dict[str, Any]],
    used_shadow: set[int],
) -> dict[str, Any]:
    errors = [float(row["error_3d_m"]) for row in matches]
    time_errors_ms = [abs(float(row["delta_t_s"])) * 1000.0 for row in matches]
    axes = {
        axis: [float(row[f"error_{axis}_m"]) for row in matches]
        for axis in ("x", "y", "z")
    }
    mahalanobis = [
        float(row["mahalanobis_sq"])
        for row in matches
        if row["mahalanobis_sq"] is not None
    ]
    return {
        "matching_method": "nearest_timestamp_one_to_one",
        "reference_samples": len(references),
        "shadow_samples": len(shadow),
        "matched_samples": len(matches),
        "unmatched_reference_samples": len(references) - len(matches),
        "unused_shadow_samples": len(shadow) - len(used_shadow),
        "match_rate": len(matches) / len(references) if references else None,
        "time_delta_abs_ms": {
            "p50": _percentile(time_errors_ms, 0.50),
            "p95": _percentile(time_errors_ms, 0.95),
            "max": max(time_errors_ms) if time_errors_ms else None,
        },
        "position_error_m": {
            "mean": sum(errors) / len(errors) if errors else None,
            "rmse": math.sqrt(sum(value * value for value in errors) / len(errors))
            if errors
            else None,
            "p50": _percentile(errors, 0.50),
            "p95": _percentile(errors, 0.95),
            "max": max(errors) if errors else None,
        },
        "axis_bias_m": {
            axis: sum(values) / len(values) if values else None
            for axis, values in axes.items()
        },
        "axis_rmse_m": {
            axis: math.sqrt(sum(value * value for value in values) / len(values))
            if values
            else None
            for axis, values in axes.items()
        },
        "uncertainty": {
            "usable_samples": len(mahalanobis),
            "mean_mahalanobis_sq": sum(mahalanobis) / len(mahalanobis)
            if mahalanobis
            else None,
            "coverage_68": sum(value <= 3.506 for value in mahalanobis) / len(mahalanobis)
            if mahalanobis
            else None,
            "coverage_95": sum(value <= 7.815 for value in mahalanobis) / len(mahalanobis)
            if mahalanobis
            else None,
        },
    }


def _trial_quality(trial: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    _, frames = _read_csv(trial / "camera_frames.csv")
    _, detections = _read_csv(trial / "detections_2d.csv")
    _, pairs = _read_csv(trial / "stereo_pairs.csv")
    _, trajectory = _read_csv(trial / "trajectory.csv")
    roles = _read_json(
        trial.parent.parent / "calibrations" / f"{metadata['calibration_id']}.json"
    )["camera_roles"]
    frame_counts = {
        role: sum(row["camera_id"] == camera_id for row in frames)
        for role, camera_id in roles.items()
    }
    frames_by_role: dict[str, dict[int, dict[str, str]]] = {
        role: {
            int(row["frame_index"]): row
            for row in frames
            if row["camera_id"] == camera_id
        }
        for role, camera_id in roles.items()
    }
    common_indices = sorted(
        set(frames_by_role.get("left", {})) & set(frames_by_role.get("right", {}))
    )
    common_frames = len(common_indices)

    receive_latency_ms: dict[str, list[float]] = {"left": [], "right": []}
    inference_latency_ms: dict[str, list[float]] = {"left": [], "right": []}
    arrival_delta_abs_ms: list[float] = []
    arrival_order = {"left_first": 0, "right_first": 0, "tie": 0, "unknown": 0}
    for frame_index in common_indices:
        paired_rows = {
            role: frames_by_role[role][frame_index]
            for role in ("left", "right")
        }
        receives: dict[str, int | None] = {}
        for role, row in paired_rows.items():
            exposure = _optional_timestamp_ns(row, "exposure_start_ns")
            receive = _optional_timestamp_ns(row, "receive_timestamp_ns")
            inference = _optional_timestamp_ns(row, "inference_done_timestamp_ns")
            receives[role] = receive
            if exposure is not None and receive is not None:
                receive_latency_ms[role].append((receive - exposure) / 1.0e6)
            if exposure is not None and inference is not None:
                inference_latency_ms[role].append((inference - exposure) / 1.0e6)

        left_receive = receives["left"]
        right_receive = receives["right"]
        if left_receive is None or right_receive is None:
            arrival_order["unknown"] += 1
        elif left_receive < right_receive:
            arrival_order["left_first"] += 1
            arrival_delta_abs_ms.append((right_receive - left_receive) / 1.0e6)
        elif right_receive < left_receive:
            arrival_order["right_first"] += 1
            arrival_delta_abs_ms.append((left_receive - right_receive) / 1.0e6)
        else:
            arrival_order["tie"] += 1
            arrival_delta_abs_ms.append(0.0)
    candidate_counts = {
        role: sum(row["camera_id"] == camera_id for row in detections)
        for role, camera_id in roles.items()
    }
    selected_counts = {
        role: sum(
            row["camera_id"] == camera_id and row["selected"] in {"1", "true", "True"}
            for row in detections
        )
        for role, camera_id in roles.items()
    }
    reject_reasons: dict[str, int] = {}
    accepted_pairs = 0
    for row in pairs:
        if row["accepted"] in {"1", "true", "True"}:
            accepted_pairs += 1
        else:
            reason = row["reject_reason"] or "unspecified"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
    times = [float(row["timestamp_s"]) for row in trajectory]
    gaps_ms = [
        (right - left) * 1000.0
        for left, right in zip(times, times[1:])
    ]
    return {
        "camera_frames": frame_counts,
        "stereo_frame_count": common_frames,
        "detection_candidates": candidate_counts,
        "selected_detections": selected_counts,
        "pair_attempts": len(pairs),
        "accepted_pairs": accepted_pairs,
        "rejected_pairs": len(pairs) - accepted_pairs,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "direct_observations": len(trajectory),
        "direct_observation_coverage": len(trajectory) / common_frames
        if common_frames
        else None,
        "observation_gap_ms": {
            "p50": _percentile(gaps_ms, 0.50),
            "p95": _percentile(gaps_ms, 0.95),
            "max": max(gaps_ms) if gaps_ms else None,
        },
        "transport_timing": {
            "exposure_to_receive_ms": {
                role: _distribution(values)
                for role, values in receive_latency_ms.items()
            },
            "exposure_to_inference_done_ms": {
                role: _distribution(values)
                for role, values in inference_latency_ms.items()
            },
            "left_right_arrival_delta_abs_ms": _distribution(arrival_delta_abs_ms),
            "arrival_order": arrival_order,
            "pairing_key": "frame_counter",
            "arrival_time_used_for_pairing": False,
        },
        "raw_files": len(metadata.get("raw_manifest", [])),
    }


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_replay(
    trial_directory: str | Path,
    config: ResolvedConfig,
    *,
    shadow_trajectory: str | Path | None = None,
    output_root: str | Path | None = None,
    max_time_delta_ms: float = 20.0,
    verify_hashes: bool = True,
    project_root: str | Path | None = None,
) -> ReplayResult:
    """校验原 trial，并以内容指纹原子发布派生回放报告。"""

    if config.data["runtime"]["mode"] != "replay":
        raise DatasetError("replay 命令必须使用 runtime.mode=replay 的 profile")
    if not math.isfinite(max_time_delta_ms) or max_time_delta_ms < 0.0:
        raise DatasetError("max_time_delta_ms 必须是非负有限数")
    root = find_project_root(project_root)
    trial = Path(trial_directory).resolve()
    validate_trial(
        trial,
        project_root=root,
        verify_hashes=verify_hashes,
    )
    metadata_path = trial / "metadata.json"
    trajectory_path = trial / "trajectory.csv"
    metadata = _read_json(metadata_path)
    references = _load_samples(trajectory_path, source_name="trajectory.csv")

    shadow_path = Path(shadow_trajectory).resolve() if shadow_trajectory else None
    shadow = _load_samples(shadow_path, source_name="shadow trajectory") if shadow_path else []
    inputs = {
        "derived_schema_version": DERIVED_SCHEMA_VERSION,
        "tool": "volleyball_vision.replay",
        "code_revision": _git_revision(root),
        "config_sha256": config.fingerprint_sha256,
        "source_trial_id": metadata["trial_id"],
        "source_metadata_sha256": _sha256(metadata_path),
        "source_trajectory_sha256": _sha256(trajectory_path),
        "source_pipeline_fingerprints": [
            pipeline["fingerprint_sha256"]
            for pipeline in metadata["perception_pipelines"]
        ],
        "shadow_trajectory_sha256": _sha256(shadow_path) if shadow_path else None,
        "max_time_delta_ms": max_time_delta_ms,
        "matching_method": "nearest_timestamp_one_to_one",
    }
    fingerprint = hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()
    parent = Path(output_root).resolve() if output_root else trial / "derived"
    destination = parent / fingerprint
    if destination.exists():
        manifest = _read_json(destination / "manifest.json")
        if manifest.get("fingerprint_sha256") != fingerprint:
            raise DatasetError(f"派生输出目录指纹冲突: {destination}")
        metrics = _read_json(destination / "metrics.json")
        return ReplayResult(destination, fingerprint, metrics, reused=True)

    quality = _trial_quality(trial, metadata)
    matches: list[dict[str, Any]] = []
    comparison: dict[str, Any] | None = None
    if shadow_path is not None:
        matches, used_shadow = _nearest_matches(
            references,
            shadow,
            max_time_delta_ms / 1000.0,
        )
        comparison = _comparison_metrics(references, shadow, matches, used_shadow)
    metrics: dict[str, Any] = {
        "trial_id": metadata["trial_id"],
        "quality": quality,
        "shadow_comparison": comparison,
    }

    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=parent))
    try:
        observation_rows = [
            {
                "sample_index": sample.index,
                "timestamp_s": sample.timestamp_s,
                "x_m": sample.position_m[0],
                "y_m": sample.position_m[1],
                "z_m": sample.position_m[2],
            }
            for sample in references
        ]
        _write_csv(
            temporary / "direct_observations.csv",
            ("sample_index", "timestamp_s", "x_m", "y_m", "z_m"),
            observation_rows,
        )
        if shadow_path is not None:
            _write_csv(temporary / "matches.csv", MATCH_COLUMNS, matches)
        _write_json(temporary / "metrics.json", metrics)
        output_files = {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            **inputs,
            "fingerprint_sha256": fingerprint,
            "source_trial_relative": metadata["trial_id"],
            "outputs": output_files,
        }
        _write_json(temporary / "manifest.json", manifest)
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.is_dir():
                raise
            existing = _read_json(destination / "manifest.json")
            if existing.get("fingerprint_sha256") != fingerprint:
                raise DatasetError(f"派生输出发布冲突: {destination}")
            shutil.rmtree(temporary)
            metrics = _read_json(destination / "metrics.json")
            return ReplayResult(destination, fingerprint, metrics, reused=True)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return ReplayResult(destination, fingerprint, metrics)
