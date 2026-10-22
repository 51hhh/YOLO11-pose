"""v2.1 实机采集编排与旧 NX 管线 staging 适配。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

from .calibration import ensure_dataset_calibration
from .config import ConfigError, ResolvedConfig, find_project_root
from .dataset import DatasetError, TrialPackageWriter, with_pipeline_fingerprint
from .legacy_config import write_legacy_pipeline_config
from .model_manifest import resolve_model, sha256_file


STAGE_SCHEMA_VERSION = 1
STAGE_FRAME_COLUMNS = [
    "frame_id",
    "host_capture_timestamp_ns",
    "callback_timestamp_ns",
    "left_device_timestamp_ns",
    "right_device_timestamp_ns",
    "stereo_timestamp_residual_ns",
    "left_frame_number",
    "right_frame_number",
    "left_frame_counter",
    "right_frame_counter",
    "left_trigger_index",
    "right_trigger_index",
    "raw_left",
    "raw_right",
]
STAGE_DETECTION_COLUMNS = [
    "frame_id",
    "camera_role",
    "candidate_id",
    "cx_px",
    "cy_px",
    "width_px",
    "height_px",
    "confidence",
    "class_id",
    "selected",
]
STAGE_PAIR_COLUMNS = [
    "pair_id",
    "frame_id",
    "left_candidate_id",
    "right_candidate_id",
    "disparity_px",
    "epipolar_error_px",
    "pair_score",
    "semantic_confidence",
    "x_rig_m",
    "y_rig_m",
    "z_rig_m",
    "accepted",
    "reject_reason",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _read_csv_exact(path: Path, columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != columns:
                raise DatasetError(
                    f"staging 表头不匹配 {path.name}: "
                    f"expected={columns}, actual={reader.fieldnames}"
                )
            return list(reader)
    except FileNotFoundError as error:
        raise DatasetError(f"staging 缺少文件: {path}") from error


def _int(row: Mapping[str, str], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as error:
        raise DatasetError(f"staging 字段 {key} 必须是整数") from error
    return value


def _float(row: Mapping[str, str], key: str, *, optional: bool = False) -> float | None:
    value = row.get(key, "")
    if value == "" and optional:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"staging 字段 {key} 必须是数字") from error
    if not math.isfinite(parsed):
        raise DatasetError(f"staging 字段 {key} 必须为有限数")
    return parsed


def _bool(row: Mapping[str, str], key: str) -> bool:
    value = row.get(key)
    if value in {"1", "true", "True"}:
        return True
    if value in {"0", "false", "False"}:
        return False
    raise DatasetError(f"staging 字段 {key} 必须是布尔值")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"staging 缺少文件: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"staging JSON 无效 {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetError(f"staging JSON 顶层必须是对象: {path}")
    return value


def build_capture_context(
    config: ResolvedConfig,
    *,
    trial_id: str,
    session_id: str,
    ball_id: str,
    trajectory_kind: str = "free_flight",
    conditions: Mapping[str, Any] | None = None,
    notes: str = "",
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root)
    entry, artifact = resolve_model(config, project_root=root)
    if not artifact.is_file():
        raise DatasetError(f"模型文件不存在: {artifact}")
    weights_sha256 = sha256_file(artifact)
    expected = entry.get("sha256")
    if not expected:
        raise DatasetError("模型清单必须先固定 weights SHA-256 才能发布真实数据")
    if expected != weights_sha256:
        raise DatasetError("模型文件 SHA-256 与 models/manifest.yaml 不一致")
    detector = config.data["detector"]
    pipeline = with_pipeline_fingerprint(
        {
            "pipeline_id": detector["model_id"],
            "detector_family": entry["family"],
            "detector_version": str(entry.get("version", detector["model_id"])),
            "code_revision": _git_revision(root),
            "config_sha256": config.fingerprint_sha256,
            "weights_sha256": weights_sha256,
            "input_size_px": list(detector["input_resolution"]),
            "resize_mode": detector["resize_mode"],
            "score_threshold": detector["confidence_threshold"],
            "nms_iou_threshold": detector["nms_iou_threshold"],
            "center_method": detector["center_method"],
        }
    )
    return {
        "stage_schema_version": STAGE_SCHEMA_VERSION,
        "trial_id": trial_id,
        "session_id": session_id,
        "ball_id": ball_id,
        "calibration_id": config.data["calibration"]["id"],
        "captured_at_utc": _utc_now(),
        "trajectory_kind": trajectory_kind,
        "conditions": dict(conditions or {}),
        "notes": notes,
        "profile": config.profile,
        "config_sha256": config.fingerprint_sha256,
        "perception_pipeline": pipeline,
    }


def prepare_capture_stage(stage: str | Path, context: Mapping[str, Any]) -> Path:
    directory = Path(stage).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "context.json").open("x", encoding="utf-8") as handle:
        json.dump(dict(context), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return directory


def _shared_capture_timestamps(row: Mapping[str, str]) -> tuple[int, int]:
    left = _int(row, "left_device_timestamp_ns")
    residual = _int(row, "stereo_timestamp_residual_ns")
    # residual=(left_raw-right_raw)-fixed_offset，故 right 映射到左时钟后为 left-residual。
    return left, left - residual


def _fit_host_clock_mapping(
    frames: list[dict[str, str]],
    shared_left_ns: list[int],
    reference_clock: str,
) -> dict[str, Any] | None:
    points = [
        (_int(row, "host_capture_timestamp_ns"), reference)
        for row, reference in zip(frames, shared_left_ns)
        if _int(row, "host_capture_timestamp_ns") > 0
    ]
    if not points:
        return None
    source_anchor, reference_anchor = points[0]
    if len(points) >= 2:
        xs = [source - source_anchor for source, _ in points]
        ys = [reference - reference_anchor for _, reference in points]
        denominator = sum(value * value for value in xs)
        scale = sum(x * y for x, y in zip(xs, ys)) / denominator if denominator else 1.0
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    drift_ppm = (scale - 1.0) * 1.0e6
    residuals = [
        abs(reference_anchor + round((source - source_anchor) * scale) - reference)
        for source, reference in points
    ]
    callback_times = [_int(row, "callback_timestamp_ns") for row in frames]
    all_source = [source for source, _ in points] + callback_times
    return {
        "mapping_id": "nx-realtime-to-camera-0001",
        "source_clock_domain": "nx-realtime",
        "reference_clock_domain": reference_clock,
        "source_anchor_ns": source_anchor,
        "reference_anchor_ns": reference_anchor,
        "drift_ppm": drift_ppm,
        "uncertainty_ns": max(100_000.0, float(max(residuals, default=0))),
        "valid_from_source_ns": min(all_source),
        "valid_to_source_ns": max(all_source),
        "source": "capture-stage-affine-fit",
    }


def _environment_record(
    conditions: Mapping[str, Any],
    timestamp_ns: int,
    clock_domain: str,
) -> dict[str, Any] | None:
    value = conditions.get("environment")
    if not isinstance(value, Mapping):
        return None
    return {
        "timestamp_ns": timestamp_ns,
        "clock_domain": clock_domain,
        "air_temperature_c": value.get("air_temperature_c"),
        "relative_humidity": value.get("relative_humidity"),
        "static_pressure_pa": value.get("static_pressure_pa"),
        "wind_x_mps": value.get("wind_x_mps"),
        "wind_y_mps": value.get("wind_y_mps"),
        "wind_z_mps": value.get("wind_z_mps"),
        "hvac_state": value.get("hvac_state"),
        "source": value.get("source", "manual"),
    }


def finalize_capture_stage(
    stage: str | Path,
    dataset_root: str | Path,
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> Path:
    directory = Path(stage).resolve()
    context = _load_json_object(directory / "context.json")
    status = _load_json_object(directory / "status.json")
    if context.get("stage_schema_version") != STAGE_SCHEMA_VERSION:
        raise DatasetError("capture staging schema 版本不兼容")
    if status.get("stage_schema_version") != STAGE_SCHEMA_VERSION:
        raise DatasetError("capture status schema 版本不兼容")
    if not status.get("complete"):
        raise DatasetError(f"采集未完整结束: {status.get('fatal_error') or 'unknown'}")
    for counter in ("dropped_frames", "copy_failures", "write_failures"):
        if int(status.get(counter, 0)) != 0:
            raise DatasetError(f"采集存在 {counter}={status[counter]}，fail_trial 禁止发布")

    frames = _read_csv_exact(directory / "frames.csv", STAGE_FRAME_COLUMNS)
    detections = _read_csv_exact(directory / "detections.csv", STAGE_DETECTION_COLUMNS)
    pairs = _read_csv_exact(directory / "pairs.csv", STAGE_PAIR_COLUMNS)
    if not frames:
        raise DatasetError("采集没有任何完整双目帧")
    frame_ids = [_int(row, "frame_id") for row in frames]
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise DatasetError("staging frame_id 必须严格递增")
    if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
        raise DatasetError("staging 出现管线帧缺口，fail_trial 禁止发布")
    left_counters = [_int(row, "left_frame_counter") for row in frames]
    right_counters = [_int(row, "right_frame_counter") for row in frames]
    if any(left != right for left, right in zip(left_counters, right_counters)):
        raise DatasetError("左右 FrameCounter 不一致")
    if any(right != left + 1 for left, right in zip(left_counters, left_counters[1:])):
        raise DatasetError("硬件 FrameCounter 出现缺口，fail_trial 禁止发布")

    root = Path(dataset_root).resolve()
    ensure_dataset_calibration(root, config, project_root=project_root)
    left_id = config.data["cameras"]["left"]["id"]
    right_id = config.data["cameras"]["right"]["id"]
    capture_times = [_shared_capture_timestamps(row) for row in frames]
    origin_ns = min(round((left + right) / 2) for left, right in capture_times)
    camera_clock = config.data["cameras"]["sync"]["clock_domain"]
    metadata = {
        "trial_id": context["trial_id"],
        "session_id": context["session_id"],
        "ball_id": context["ball_id"],
        "calibration_id": context["calibration_id"],
        "captured_at_utc": context["captured_at_utc"],
        "trajectory_kind": context["trajectory_kind"],
        "coordinate_frame": {
            "name": config.data["calibration"]["rig_frame"],
            "origin": "left rectified camera optical center",
            "x_axis": "image right",
            "y_axis": "image down",
            "z_axis": "optical forward",
            "handedness": "right",
        },
        "capture": {
            "nominal_fps": config.data["runtime"]["nominal_fps"],
            "sync_method": config.data["cameras"]["sync"]["method"],
            "estimated_sync_error_s": (
                None
                if config.data["cameras"]["sync"]["estimated_error_ns"] is None
                else config.data["cameras"]["sync"]["estimated_error_ns"] * 1.0e-9
            ),
            "trajectory_timebase": {
                "clock_domain": camera_clock,
                "origin_ns": origin_ns,
                "timestamp_semantics": "stereo_midpoint_capture",
            },
        },
        "conditions": context.get("conditions", {}),
        "notes": context.get("notes", ""),
        "perception_pipelines": [context["perception_pipeline"]],
    }
    frame_lookup = dict(zip(frame_ids, capture_times))
    raw_mode = config.data["recording"]["raw_frames"]
    with TrialPackageWriter(root, metadata, project_root=project_root) as writer:
        for row, (left_capture, right_capture) in zip(frames, capture_times):
            frame_id = _int(row, "frame_id")
            common = {
                "frame_index": frame_id,
                "clock_domain": camera_clock,
                "exposure_start_ns": None,
                "receive_timestamp_ns": None,
                "inference_done_timestamp_ns": None,
                "exposure_us": config.data["cameras"]["left"]["exposure_us"],
                "gain_db": config.data["cameras"]["left"]["gain_db"],
                "roi_x": 0,
                "roi_y": 0,
                "roi_width": config.data["rectification"]["output_resolution"][0],
                "roi_height": config.data["rectification"]["output_resolution"][1],
                "pixel_format": "rectified_gray8",
                "dropped": False,
                "drop_reason": None,
            }
            writer.append(
                "camera_frames",
                {
                    **common,
                    "camera_id": left_id,
                    "hardware_timestamp_ns": left_capture,
                    "capture_timestamp_ns": left_capture,
                },
            )
            writer.append(
                "camera_frames",
                {
                    **common,
                    "camera_id": right_id,
                    "hardware_timestamp_ns": right_capture,
                    "capture_timestamp_ns": right_capture,
                },
            )

        pipeline_id = context["perception_pipeline"]["pipeline_id"]
        for row in detections:
            role = row["camera_role"]
            if role not in {"left", "right"}:
                raise DatasetError(f"未知 camera_role: {role!r}")
            width = _float(row, "width_px")
            height = _float(row, "height_px")
            cx = _float(row, "cx_px")
            cy = _float(row, "cy_px")
            assert width is not None and height is not None and cx is not None and cy is not None
            writer.append(
                "detections_2d",
                {
                    "camera_id": left_id if role == "left" else right_id,
                    "frame_index": _int(row, "frame_id"),
                    "candidate_id": _int(row, "candidate_id"),
                    "u_px": cx,
                    "v_px": cy,
                    "radius_px": (width + height) * 0.25,
                    "bbox_x": cx - width * 0.5,
                    "bbox_y": cy - height * 0.5,
                    "bbox_w": width,
                    "bbox_h": height,
                    "du_dt_pxps": None,
                    "dv_dt_pxps": None,
                    "score": _float(row, "confidence"),
                    "cov_uu_px2": None,
                    "cov_uv_px2": None,
                    "cov_vv_px2": None,
                    "selected": _bool(row, "selected"),
                    "source": pipeline_id,
                },
            )

        accepted_by_frame: dict[int, list[dict[str, str]]] = {}
        for row in pairs:
            frame_id = _int(row, "frame_id")
            if frame_id not in frame_lookup:
                raise DatasetError("配对引用不存在的 staging frame")
            left_capture, right_capture = frame_lookup[frame_id]
            accepted = _bool(row, "accepted")
            position = {
                axis: _float(row, axis, optional=not accepted)
                for axis in ("x_rig_m", "y_rig_m", "z_rig_m")
            }
            writer.append(
                "stereo_pairs",
                {
                    "pair_id": _int(row, "pair_id"),
                    "left_frame_index": frame_id,
                    "right_frame_index": frame_id,
                    "left_candidate_id": _int(row, "left_candidate_id"),
                    "right_candidate_id": _int(row, "right_candidate_id"),
                    "left_capture_ns": left_capture,
                    "right_capture_ns": right_capture,
                    "delta_t_lr_ns": right_capture - left_capture,
                    "disparity_px": _float(row, "disparity_px", optional=True),
                    "epipolar_error_px": _float(row, "epipolar_error_px", optional=True),
                    **position,
                    "cov_xx_m2": None,
                    "cov_xy_m2": None,
                    "cov_xz_m2": None,
                    "cov_yy_m2": None,
                    "cov_yz_m2": None,
                    "cov_zz_m2": None,
                    "accepted": accepted,
                    "reject_reason": None if accepted else row["reject_reason"],
                },
            )
            if accepted:
                accepted_by_frame.setdefault(frame_id, []).append(row)

        sample_index = 0
        previous_timestamp_ns = -1
        for frame_id in sorted(accepted_by_frame):
            candidates = accepted_by_frame[frame_id]
            # 单球轨迹每个曝光时刻只保留一个直接观测；所有接受配对仍在 stereo_pairs。
            chosen = max(
                candidates,
                key=lambda row: (
                    _float(row, "semantic_confidence", optional=True) or 0.0,
                    -(_float(row, "pair_score", optional=True) or 0.0),
                ),
            )
            left_capture, right_capture = frame_lookup[frame_id]
            timestamp_ns = round((left_capture + right_capture) / 2) - origin_ns
            if timestamp_ns <= previous_timestamp_ns:
                raise DatasetError("接受的直接观测时间不严格递增")
            previous_timestamp_ns = timestamp_ns
            writer.append(
                "trajectory",
                {
                    "sample_index": sample_index,
                    "timestamp_s": timestamp_ns / 1.0e9,
                    "left_frame_index": frame_id,
                    "right_frame_index": frame_id,
                    "x_m": _float(chosen, "x_rig_m"),
                    "y_m": _float(chosen, "y_rig_m"),
                    "z_m": _float(chosen, "z_rig_m"),
                    "confidence": _float(chosen, "semantic_confidence", optional=True),
                    "reprojection_error_px": _float(chosen, "epipolar_error_px", optional=True),
                    "cov_xx_m2": None,
                    "cov_xy_m2": None,
                    "cov_xz_m2": None,
                    "cov_yy_m2": None,
                    "cov_yz_m2": None,
                    "cov_zz_m2": None,
                },
            )
            sample_index += 1

        mapping = _fit_host_clock_mapping(
            frames,
            [left for left, _ in capture_times],
            camera_clock,
        )
        if mapping is not None:
            writer.append("clock_mappings", mapping)
            for row, (left_capture, right_capture) in zip(frames, capture_times):
                callback_ns = _int(row, "callback_timestamp_ns")
                frame_id = _int(row, "frame_id")
                for camera_id, capture_ns in ((left_id, left_capture), (right_id, right_capture)):
                    writer.append(
                        "processing_timings",
                        {
                            "event_id": f"callback-{camera_id}-{frame_id}",
                            "stage": "legacy_frame_callback",
                            "input_type": "camera_frame",
                            "input_id": f"{camera_id}:{frame_id}",
                            "capture_reference_ns": capture_ns,
                            "start_timestamp_ns": callback_ns,
                            "done_timestamp_ns": callback_ns,
                            "clock_domain": "nx-realtime",
                            "queue_depth": None,
                            "valid": True,
                            "source": pipeline_id,
                        },
                    )

        environment = _environment_record(
            context.get("conditions", {}),
            origin_ns,
            camera_clock,
        )
        if environment is not None:
            writer.append("environment", environment)

        audit_sources = (
            ("capture_stage_frames", directory / "frames.csv", "audit/stage_frames.csv"),
            ("capture_stage_detections", directory / "detections.csv", "audit/stage_detections.csv"),
            ("capture_stage_pairs", directory / "pairs.csv", "audit/stage_pairs.csv"),
            ("capture_stage_status", directory / "status.json", "audit/stage_status.json"),
            ("capture_context", directory / "context.json", "audit/context.json"),
            ("legacy_pipeline_config", directory / "pipeline.yaml", "audit/pipeline.yaml"),
        )
        for label, source, filename in audit_sources:
            writer.add_raw_source(label, source, filename=filename)
        if raw_mode != "none":
            for row in frames:
                frame_id = _int(row, "frame_id")
                for role, column in (("left", "raw_left"), ("right", "raw_right")):
                    relative = row[column]
                    if not relative and raw_mode == "all":
                        raise DatasetError("raw_frames 启用但 staging 缺少图像路径")
                    if not relative:
                        continue
                    source = (directory / relative).resolve()
                    if directory not in source.parents:
                        raise DatasetError("staging 原始图像路径逃逸")
                    writer.add_raw_source(
                        f"{role}_{frame_id:08d}",
                        source,
                        filename=f"rectified/{role}/{frame_id:08d}.pgm",
                    )
        return writer.finalize()


def run_capture(
    config: ResolvedConfig,
    *,
    trial_id: str,
    session_id: str,
    ball_id: str,
    duration_s: float,
    conditions: Mapping[str, Any] | None = None,
    notes: str = "",
    dataset_root: str | Path | None = None,
    keep_stage: bool = False,
    dry_run: bool = False,
    project_root: str | Path | None = None,
) -> Path:
    root = find_project_root(project_root)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise DatasetError("duration_s 必须是正的有限秒数")
    capture_root = (
        Path(dataset_root).resolve()
        if dataset_root is not None
        else (root / config.data["recording"]["dataset_root"]).resolve()
    )
    staging_root = capture_root / ".capture-staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{trial_id}-", dir=staging_root))
    # mkdtemp 已创建目录，prepare 需要独占创建；只移除这个刚创建且已解析的叶目录。
    stage.rmdir()
    context = build_capture_context(
        config,
        trial_id=trial_id,
        session_id=session_id,
        ball_id=ball_id,
        conditions=conditions,
        notes=notes,
        project_root=root,
    )
    prepare_capture_stage(stage, context)
    frame_limit = max(1, round(duration_s * config.data["runtime"]["nominal_fps"]))
    legacy_config = write_legacy_pipeline_config(
        stage / "pipeline.yaml",
        config,
        capture_stage=stage,
        frame_limit=frame_limit,
        project_root=root,
    )
    executable = (root / config.data["adapters"]["legacy_pipeline"]["executable"]).resolve()
    command = [str(executable), "--config", str(legacy_config)]
    if dry_run:
        return stage
    try:
        completed = subprocess.run(command, cwd=executable.parent, check=False)
        if completed.returncode != 0:
            raise DatasetError(f"旧 NX 管线退出码为 {completed.returncode}，staging={stage}")
        published = finalize_capture_stage(stage, capture_root, config, project_root=root)
    except Exception:
        keep_stage = True
        raise
    finally:
        if not keep_stage and stage.exists():
            shutil.rmtree(stage)
    return published

