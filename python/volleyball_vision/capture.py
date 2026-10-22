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


STAGE_SCHEMA_VERSION = 4
STAGE_FRAME_COLUMNS = [
    "frame_id",
    "host_capture_timestamp_ns",
    "callback_timestamp_ns",
    "left_receive_monotonic_ns",
    "right_receive_monotonic_ns",
    "left_receive_realtime_ns",
    "right_receive_realtime_ns",
    "left_sdk_host_timestamp_raw",
    "right_sdk_host_timestamp_raw",
    "left_device_timestamp_ns",
    "right_device_timestamp_ns",
    "stereo_timestamp_residual_ns",
    "trigger_sequence",
    "trigger_scheduled_monotonic_ns",
    "trigger_edge_monotonic_ns",
    "trigger_edge_realtime_ns",
    "trigger_write_uncertainty_ns",
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


def _select_direct_pairs(
    pairs: list[dict[str, str]],
) -> dict[int, dict[str, str]]:
    """为单球轨迹的每个曝光时刻选择唯一直接观测。"""

    pair_ids: set[int] = set()
    candidates: dict[int, list[dict[str, str]]] = {}
    for row in pairs:
        pair_id = _int(row, "pair_id")
        frame_id = _int(row, "frame_id")
        if pair_id < 0 or frame_id < 0:
            raise DatasetError("staging pair_id/frame_id 不得为负")
        if pair_id in pair_ids:
            raise DatasetError(f"staging 出现重复 pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if not _bool(row, "accepted"):
            continue
        confidence = _float(row, "semantic_confidence")
        score = _float(row, "pair_score")
        if confidence is None or not 0.0 <= confidence <= 1.0:
            raise DatasetError(
                "accepted staging pair 的 semantic_confidence 必须在 [0,1]"
            )
        assert score is not None
        position = {
            axis: _float(row, axis)
            for axis in ("x_rig_m", "y_rig_m", "z_rig_m")
        }
        if position["z_rig_m"] is None or position["z_rig_m"] <= 0.0:
            raise DatasetError("accepted staging pair 的 z_rig_m 必须大于 0")
        candidates.setdefault(frame_id, []).append(row)

    return {
        frame_id: min(
            rows,
            key=lambda row: (
                -float(row["semantic_confidence"]),
                float(row["pair_score"]),
                int(row["pair_id"]),
            ),
        )
        for frame_id, rows in candidates.items()
    }


def _uint32_is_next(previous: int, current: int) -> bool:
    """按相机 uint32 计数器语义判断连续性，允许自然回绕。"""

    if not 0 <= previous <= 0xFFFFFFFF or not 0 <= current <= 0xFFFFFFFF:
        return False
    return ((current - previous) & 0xFFFFFFFF) == 1


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
    verify_model: bool = True,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root)
    entry, artifact = resolve_model(config, project_root=root)
    expected = entry.get("sha256")
    weights_sha256: str | None
    if artifact.is_file():
        weights_sha256 = sha256_file(artifact)
        if expected and expected != weights_sha256:
            raise DatasetError("模型文件 SHA-256 与 models/manifest.yaml 不一致")
    else:
        weights_sha256 = expected
    if verify_model:
        if not artifact.is_file():
            raise DatasetError(f"模型文件不存在: {artifact}")
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
    context = {
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
        "exposure_timestamp": {
            "shutter_type": config.data["cameras"]["sync"]["shutter_type"],
            "source": config.data["cameras"]["sync"]["exposure_timestamp_source"],
            "delay_status": config.data["cameras"]["sync"][
                "exposure_delay_status"
            ],
            "delay_ns": config.data["cameras"]["sync"]["exposure_delay_ns"],
            "delay_uncertainty_ns": config.data["cameras"]["sync"][
                "exposure_delay_uncertainty_ns"
            ],
            "observation_timestamp": "exposure_midpoint",
        },
        "perception_pipeline": pipeline,
    }
    if not verify_model:
        # 预演允许目标机 artifact 尚未安装，但该 staging 永远不能发布为真实数据。
        context["planning_only"] = True
    return context


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


def _fit_clock_mapping(
    source_times_ns: list[int],
    reference_times_ns: list[int],
    *,
    source_clock: str,
    reference_clock: str,
    mapping_id: str,
    source: str,
    validity_source_times_ns: Iterable[int] = (),
) -> dict[str, Any] | None:
    points = [
        (source_time, reference_time)
        for source_time, reference_time in zip(source_times_ns, reference_times_ns)
        if source_time > 0
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
    all_source = [source_time for source_time, _ in points]
    all_source.extend(value for value in validity_source_times_ns if value > 0)
    return {
        "mapping_id": mapping_id,
        "source_clock_domain": source_clock,
        "reference_clock_domain": reference_clock,
        "source_anchor_ns": source_anchor,
        "reference_anchor_ns": reference_anchor,
        "drift_ppm": drift_ppm,
        "uncertainty_ns": max(100_000.0, float(max(residuals, default=0))),
        "valid_from_source_ns": min(all_source),
        "valid_to_source_ns": max(all_source),
        "source": source,
    }


def _map_timestamp(timestamp_ns: int, mapping: Mapping[str, Any]) -> int:
    scale = 1.0 + float(mapping["drift_ppm"]) * 1.0e-6
    return int(mapping["reference_anchor_ns"]) + round(
        (timestamp_ns - int(mapping["source_anchor_ns"])) * scale
    )


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
    if context.get("planning_only") is True:
        raise DatasetError("预演 staging 不得发布为真实 trial")
    if context.get("stage_schema_version") != STAGE_SCHEMA_VERSION:
        raise DatasetError("capture staging schema 版本不兼容")
    if status.get("stage_schema_version") != STAGE_SCHEMA_VERSION:
        raise DatasetError("capture status schema 版本不兼容")
    if not status.get("complete"):
        raise DatasetError(f"采集未完整结束: {status.get('fatal_error') or 'unknown'}")
    if context.get("config_sha256") != config.fingerprint_sha256:
        raise DatasetError("staging 使用的配置指纹与 finalize 配置不一致")
    expected_exposure = {
        "shutter_type": config.data["cameras"]["sync"]["shutter_type"],
        "source": config.data["cameras"]["sync"]["exposure_timestamp_source"],
        "delay_status": config.data["cameras"]["sync"][
            "exposure_delay_status"
        ],
        "delay_ns": config.data["cameras"]["sync"]["exposure_delay_ns"],
        "delay_uncertainty_ns": config.data["cameras"]["sync"][
            "exposure_delay_uncertainty_ns"
        ],
        "observation_timestamp": "exposure_midpoint",
    }
    if context.get("exposure_timestamp") != expected_exposure:
        raise DatasetError("staging 曝光时间语义与 finalize 配置不一致")
    for counter in ("dropped_frames", "copy_failures", "write_failures"):
        if int(status.get(counter, 0)) != 0:
            raise DatasetError(f"采集存在 {counter}={status[counter]}，fail_trial 禁止发布")

    frames = _read_csv_exact(directory / "frames.csv", STAGE_FRAME_COLUMNS)
    detections = _read_csv_exact(directory / "detections.csv", STAGE_DETECTION_COLUMNS)
    pairs = _read_csv_exact(directory / "pairs.csv", STAGE_PAIR_COLUMNS)
    if not frames:
        raise DatasetError("采集没有任何完整双目帧")
    try:
        frames_captured = int(status["frames_captured"])
        frames_written = int(status["frames_written"])
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError("capture status 缺少有效帧计数") from error
    if frames_captured != len(frames) or frames_written != len(frames):
        raise DatasetError(
            "capture status 帧计数与 frames.csv 不一致: "
            f"captured={frames_captured}, written={frames_written}, rows={len(frames)}"
        )
    frame_ids = [_int(row, "frame_id") for row in frames]
    if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
        raise DatasetError("staging frame_id 必须严格递增")
    if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
        raise DatasetError("staging 出现管线帧缺口，fail_trial 禁止发布")
    left_counters = [_int(row, "left_frame_counter") for row in frames]
    right_counters = [_int(row, "right_frame_counter") for row in frames]
    if any(left != right for left, right in zip(left_counters, right_counters)):
        raise DatasetError("左右 FrameCounter 不一致")
    if any(
        not _uint32_is_next(left, right)
        for left, right in zip(left_counters, left_counters[1:])
    ):
        raise DatasetError("硬件 FrameCounter 出现缺口，fail_trial 禁止发布")

    host_times = [_int(row, "host_capture_timestamp_ns") for row in frames]
    callback_times = [_int(row, "callback_timestamp_ns") for row in frames]
    left_receive_monotonic = [
        _int(row, "left_receive_monotonic_ns") for row in frames
    ]
    right_receive_monotonic = [
        _int(row, "right_receive_monotonic_ns") for row in frames
    ]
    left_receive_realtime = [
        _int(row, "left_receive_realtime_ns") for row in frames
    ]
    right_receive_realtime = [
        _int(row, "right_receive_realtime_ns") for row in frames
    ]
    left_device_times = [_int(row, "left_device_timestamp_ns") for row in frames]
    right_device_times = [_int(row, "right_device_timestamp_ns") for row in frames]
    timestamp_series = (
        host_times,
        callback_times,
        left_receive_monotonic,
        right_receive_monotonic,
        left_receive_realtime,
        right_receive_realtime,
        left_device_times,
        right_device_times,
    )
    if any(value <= 0 for values in timestamp_series for value in values):
        raise DatasetError("采集时间戳必须是正整数纳秒")
    for name, values in (
        ("host_capture_timestamp_ns", host_times),
        ("callback_timestamp_ns", callback_times),
        ("left_receive_monotonic_ns", left_receive_monotonic),
        ("right_receive_monotonic_ns", right_receive_monotonic),
        ("left_receive_realtime_ns", left_receive_realtime),
        ("right_receive_realtime_ns", right_receive_realtime),
        ("left_device_timestamp_ns", left_device_times),
        ("right_device_timestamp_ns", right_device_times),
    ):
        if any(right <= left for left, right in zip(values, values[1:])):
            raise DatasetError(f"{name} 必须严格递增")
    if any(callback < capture for capture, callback in zip(host_times, callback_times)):
        raise DatasetError("callback_timestamp_ns 不得早于 host_capture_timestamp_ns")

    trigger_sequences = [_int(row, "trigger_sequence") for row in frames]
    scheduled_monotonic = [
        _int(row, "trigger_scheduled_monotonic_ns") for row in frames
    ]
    edge_monotonic = [_int(row, "trigger_edge_monotonic_ns") for row in frames]
    edge_realtime = [_int(row, "trigger_edge_realtime_ns") for row in frames]
    write_uncertainty = [
        _int(row, "trigger_write_uncertainty_ns") for row in frames
    ]
    if any(value <= 0 for value in trigger_sequences):
        raise DatasetError("trigger_sequence 必须为正整数")
    if any(
        (sequence & 0xFFFFFFFF) != counter
        for sequence, counter in zip(trigger_sequences, left_counters)
    ):
        raise DatasetError("trigger_sequence 必须与相机 FrameCounter 对齐")
    if any(
        right != left + 1
        for left, right in zip(trigger_sequences, trigger_sequences[1:])
    ):
        raise DatasetError("GPIO 触发序列出现缺口")
    for name, values in (
        ("trigger_scheduled_monotonic_ns", scheduled_monotonic),
        ("trigger_edge_monotonic_ns", edge_monotonic),
        ("trigger_edge_realtime_ns", edge_realtime),
    ):
        if any(value <= 0 for value in values):
            raise DatasetError(f"{name} 必须为正整数纳秒")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise DatasetError(f"{name} 必须严格递增")
    if any(value < 0 for value in write_uncertainty):
        raise DatasetError("trigger_write_uncertainty_ns 不得为负")
    if any(
        edge < scheduled
        for edge, scheduled in zip(edge_monotonic, scheduled_monotonic)
    ):
        raise DatasetError("GPIO 上升沿时间不得早于计划时间")
    if any(edge > host for edge, host in zip(edge_realtime, host_times)):
        raise DatasetError("GPIO 上升沿不得晚于相机主机采集时间")
    for role, receive_mono, receive_real in (
        ("left", left_receive_monotonic, left_receive_realtime),
        ("right", right_receive_monotonic, right_receive_realtime),
    ):
        if any(
            receive < edge
            for receive, edge in zip(receive_mono, edge_monotonic)
        ):
            raise DatasetError(f"{role} USB 单调时钟到达时间早于 GPIO 上升沿")
        if any(
            receive < edge
            for receive, edge in zip(receive_real, edge_realtime)
        ):
            raise DatasetError(f"{role} USB 实时时钟到达时间早于 GPIO 上升沿")
    if any(
        callback < max(left_receive, right_receive)
        for callback, left_receive, right_receive in zip(
            callback_times,
            left_receive_realtime,
            right_receive_realtime,
        )
    ):
        raise DatasetError("callback_timestamp_ns 不得早于左右 USB 帧到达时间")

    frame_period_ns = 1.0e9 / config.data["runtime"]["nominal_fps"]
    period_tolerance_ns = max(
        frame_period_ns * 0.25,
        float(config.data["cameras"]["sync"]["estimated_error_ns"] or 0.0) * 4.0,
    )
    for index in range(1, len(frames)):
        trigger_period = edge_monotonic[index] - edge_monotonic[index - 1]
        camera_period = left_device_times[index] - left_device_times[index - 1]
        if abs(trigger_period - frame_period_ns) > period_tolerance_ns:
            raise DatasetError("GPIO 触发周期偏离配置帧率，疑似调度异常或漏触发")
        if abs(camera_period - trigger_period) > period_tolerance_ns:
            raise DatasetError("相机设备时间与 GPIO 触发序列不一致，疑似漏曝光")

    configured_sync_error = config.data["cameras"]["sync"]["estimated_error_ns"]
    if configured_sync_error is not None:
        residuals = [abs(_int(row, "stereo_timestamp_residual_ns")) for row in frames]
        if any(value > configured_sync_error for value in residuals):
            raise DatasetError(
                "双目同步残差超过 cameras.sync.estimated_error_ns: "
                f"max={max(residuals)}, limit={configured_sync_error}"
            )

    root = Path(dataset_root).resolve()
    ensure_dataset_calibration(root, config, project_root=project_root)
    left_id = config.data["cameras"]["left"]["id"]
    right_id = config.data["cameras"]["right"]["id"]
    hardware_times = [_shared_capture_timestamps(row) for row in frames]
    camera_clock = config.data["cameras"]["sync"]["clock_domain"]
    hardware_midpoints = [
        round((left + right) / 2) for left, right in hardware_times
    ]
    trigger_mapping = _fit_clock_mapping(
        edge_monotonic,
        hardware_midpoints,
        source_clock="nx-monotonic",
        reference_clock=camera_clock,
        mapping_id="nx-gpio-monotonic-to-camera-0001",
        source="nx-gpio-rising-edge-affine-fit",
        validity_source_times_ns=(
            *scheduled_monotonic,
            *left_receive_monotonic,
            *right_receive_monotonic,
            *(
                edge - uncertainty
                for edge, uncertainty in zip(edge_monotonic, write_uncertainty)
            ),
            *(
                edge + uncertainty
                for edge, uncertainty in zip(edge_monotonic, write_uncertainty)
            ),
        ),
    )
    realtime_mapping = _fit_clock_mapping(
        edge_realtime,
        hardware_midpoints,
        source_clock="nx-realtime",
        reference_clock=camera_clock,
        mapping_id="nx-realtime-to-camera-0001",
        source="paired-gpio-clock-read-affine-fit",
        validity_source_times_ns=(
            *host_times,
            *callback_times,
            *left_receive_realtime,
            *right_receive_realtime,
        ),
    )
    if trigger_mapping is None or realtime_mapping is None:
        raise DatasetError("无法建立 NX GPIO 到相机设备时钟的映射")
    configured_delay_uncertainty = expected_exposure["delay_uncertainty_ns"]
    edge_uncertainty_ns = max(write_uncertainty, default=0) + (
        0
        if configured_delay_uncertainty is None
        else int(configured_delay_uncertainty)
    )
    trigger_mapping["uncertainty_ns"] = (
        float(trigger_mapping["uncertainty_ns"]) + edge_uncertainty_ns
    )
    realtime_mapping["uncertainty_ns"] = (
        float(realtime_mapping["uncertainty_ns"]) + edge_uncertainty_ns
    )
    exposure_delay_ns = int(expected_exposure["delay_ns"])
    exposure_start_times = [
        _map_timestamp(timestamp, trigger_mapping) + exposure_delay_ns
        for timestamp in edge_monotonic
    ]
    left_receive_times = [
        _map_timestamp(timestamp, trigger_mapping)
        for timestamp in left_receive_monotonic
    ]
    right_receive_times = [
        _map_timestamp(timestamp, trigger_mapping)
        for timestamp in right_receive_monotonic
    ]
    inference_done_times = [
        _map_timestamp(timestamp, realtime_mapping)
        for timestamp in callback_times
    ]
    if any(
        right <= left
        for left, right in zip(exposure_start_times, exposure_start_times[1:])
    ):
        raise DatasetError("映射后的曝光开始时间必须严格递增")
    # 全局快门在整幅图像上同时积分。对于运动球心，直接观测的有效时间取
    # 曝光区间中点，而不是触发沿或曝光开始。曝光配置单位为微秒，这里四舍
    # 五入到整数纳秒；左右曝光时长分别计算，避免未来配置不同时静默混用。
    left_midpoint_offset_ns = round(
        float(config.data["cameras"]["left"]["exposure_us"]) * 500.0
    )
    right_midpoint_offset_ns = round(
        float(config.data["cameras"]["right"]["exposure_us"]) * 500.0
    )
    capture_times = [
        (
            exposure_start + left_midpoint_offset_ns,
            exposure_start + right_midpoint_offset_ns,
        )
        for exposure_start in exposure_start_times
    ]
    origin_ns = round(sum(capture_times[0]) / 2)
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
            "position_unit": "m",
            "time_unit": "s",
        },
        "capture": {
            "nominal_fps": config.data["runtime"]["nominal_fps"],
            "sync_method": config.data["cameras"]["sync"]["method"],
            "estimated_sync_error_s": max(
                float(config.data["cameras"]["sync"]["estimated_error_ns"] or 0.0),
                float(trigger_mapping["uncertainty_ns"]),
                float(realtime_mapping["uncertainty_ns"]),
            )
            * 1.0e-9,
            "exposure_timing": {
                "shutter_type": expected_exposure["shutter_type"],
                "timestamp_source": expected_exposure["source"],
                "delay_status": expected_exposure["delay_status"],
                "delay_ns": expected_exposure["delay_ns"],
                "delay_uncertainty_ns": expected_exposure[
                    "delay_uncertainty_ns"
                ],
                "observation_timestamp": expected_exposure[
                    "observation_timestamp"
                ],
            },
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
    accepted_by_frame = _select_direct_pairs(pairs)
    accepted_pair_ids = {
        _int(row, "pair_id") for row in accepted_by_frame.values()
    }
    accepted_detection_keys = {
        (frame_id, role, _int(row, f"{role}_candidate_id"))
        for frame_id, row in accepted_by_frame.items()
        for role in ("left", "right")
    }
    raw_mode = config.data["recording"]["raw_frames"]
    with TrialPackageWriter(root, metadata, project_root=project_root) as writer:
        for (
            row,
            (left_hardware, right_hardware),
            exposure_start,
            (left_capture, right_capture),
            left_receive,
            right_receive,
            inference_done,
        ) in zip(
            frames,
            hardware_times,
            exposure_start_times,
            capture_times,
            left_receive_times,
            right_receive_times,
            inference_done_times,
        ):
            frame_id = _int(row, "frame_id")
            common = {
                "frame_index": frame_id,
                "clock_domain": camera_clock,
                "exposure_start_ns": exposure_start,
                "roi_x": 0,
                "roi_y": 0,
                "roi_width": config.data["rectification"]["output_resolution"][0],
                "roi_height": config.data["rectification"]["output_resolution"][1],
                "pixel_format": "rectified_gray8",
                "dropped": False,
                "drop_reason": None,
            }
            for (
                camera_id,
                hardware_timestamp,
                capture_timestamp,
                receive_timestamp,
                camera_config,
            ) in (
                (
                    left_id,
                    left_hardware,
                    left_capture,
                    left_receive,
                    config.data["cameras"]["left"],
                ),
                (
                    right_id,
                    right_hardware,
                    right_capture,
                    right_receive,
                    config.data["cameras"]["right"],
                ),
            ):
                writer.append(
                    "camera_frames",
                    {
                        **common,
                        "camera_id": camera_id,
                        "hardware_timestamp_ns": hardware_timestamp,
                        "capture_timestamp_ns": capture_timestamp,
                        "receive_timestamp_ns": receive_timestamp,
                        "inference_done_timestamp_ns": inference_done,
                        "exposure_us": camera_config["exposure_us"],
                        "gain_db": camera_config["gain_db"],
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
            # 仍解析旧值以拒绝损坏的 staging；最终选择权属于直接配对。
            _bool(row, "selected")
            frame_id = _int(row, "frame_id")
            candidate_id = _int(row, "candidate_id")
            assert (
                width is not None
                and height is not None
                and cx is not None
                and cy is not None
            )
            writer.append(
                "detections_2d",
                {
                    "camera_id": left_id if role == "left" else right_id,
                    "frame_index": frame_id,
                    "candidate_id": candidate_id,
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
                    "selected": (
                        frame_id, role, candidate_id
                    ) in accepted_detection_keys,
                    "source": pipeline_id,
                },
            )

        for row in pairs:
            frame_id = _int(row, "frame_id")
            if frame_id not in frame_lookup:
                raise DatasetError("配对引用不存在的 staging frame")
            left_capture, right_capture = frame_lookup[frame_id]
            staging_accepted = _bool(row, "accepted")
            pair_id = _int(row, "pair_id")
            accepted = staging_accepted and pair_id in accepted_pair_ids
            position = {
                axis: _float(row, axis) if accepted else None
                for axis in ("x_rig_m", "y_rig_m", "z_rig_m")
            }
            writer.append(
                "stereo_pairs",
                {
                    "pair_id": pair_id,
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
                    "reject_reason": (
                        None
                        if accepted
                        else "not_selected"
                        if staging_accepted
                        else row["reject_reason"]
                    ),
                },
            )

        sample_index = 0
        previous_timestamp_ns = -1
        for frame_id in sorted(accepted_by_frame):
            chosen = accepted_by_frame[frame_id]
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

        writer.append("clock_mappings", trigger_mapping)
        writer.append("clock_mappings", realtime_mapping)
        for row, (left_capture, right_capture), left_receive, right_receive in zip(
            frames,
            capture_times,
            left_receive_monotonic,
            right_receive_monotonic,
        ):
            callback_ns = _int(row, "callback_timestamp_ns")
            frame_id = _int(row, "frame_id")
            for camera_id, capture_ns in (
                (left_id, left_capture),
                (right_id, right_capture),
            ):
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
            edge_ns = _int(row, "trigger_edge_monotonic_ns")
            uncertainty_ns = _int(row, "trigger_write_uncertainty_ns")
            edge_start_ns = edge_ns - uncertainty_ns // 2
            edge_done_ns = edge_start_ns + uncertainty_ns
            writer.append(
                "processing_timings",
                {
                    "event_id": f"gpio-trigger-{_int(row, 'trigger_sequence')}",
                    "stage": "gpio_rising_edge_write",
                    "input_type": "trigger_event",
                    "input_id": str(_int(row, "trigger_sequence")),
                    "capture_reference_ns": left_capture,
                    "start_timestamp_ns": edge_start_ns,
                    "done_timestamp_ns": edge_done_ns,
                    "clock_domain": "nx-monotonic",
                    "queue_depth": None,
                    "valid": True,
                    "source": "nx-gpio-pwm",
                },
            )
            for camera_id, receive_ns, capture_ns in (
                (left_id, left_receive, left_capture),
                (right_id, right_receive, right_capture),
            ):
                writer.append(
                    "processing_timings",
                    {
                        "event_id": f"usb-receive-{camera_id}-{frame_id}",
                        "stage": "camera_frame_receive",
                        "input_type": "camera_frame",
                        "input_id": f"{camera_id}:{frame_id}",
                        "capture_reference_ns": capture_ns,
                        "start_timestamp_ns": receive_ns,
                        "done_timestamp_ns": receive_ns,
                        "clock_domain": "nx-monotonic",
                        "queue_depth": None,
                        "valid": True,
                        "source": "hik-mvs-usb",
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
    if config.data["runtime"]["mode"] != "capture":
        raise DatasetError("volleyball capture 只接受 capture profile")
    if not config.data["recording"]["enabled"]:
        raise DatasetError("capture profile 必须启用 recording")
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
        verify_model=not dry_run,
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

