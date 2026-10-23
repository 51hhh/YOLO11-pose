"""主线 v6 staging 的结构、采集时序和硬件事实校验。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capture_stage_io import (
    load_json_object,
    parse_bool,
    parse_float,
    parse_int,
    read_csv_exact,
    uint32_is_next,
)
from .config import ResolvedConfig, find_project_root
from .dataset import DatasetError
from .hashing import sha256_file
from .mainline_stage_contract import (
    MAINLINE_ASSOCIATION_COLUMNS,
    MAINLINE_DETECTION_COLUMNS,
    MAINLINE_FRAME_COLUMNS,
    MAINLINE_OBSERVATION_COLUMNS,
    MAINLINE_STAGE_SCHEMA_VERSION,
)
from .runtime import mainline_executable


@dataclass(frozen=True, slots=True)
class ValidatedMainlineStage:
    directory: Path
    context: dict[str, Any]
    expected_exposure: dict[str, Any]
    frames: list[dict[str, str]]
    detections: list[dict[str, str]]
    associations: list[dict[str, str]]
    observations: list[dict[str, str]]
    camera_ids: dict[str, str]
    frame_by_key: dict[tuple[str, int], dict[str, str]]
    frame_pairs: dict[int, dict[str, dict[str, str]]]
    indexes: tuple[int, ...]
    paired_rows: tuple[tuple[dict[str, str], dict[str, str]], ...]
    edge_monotonic: tuple[int, ...]
    sync_limit_ns: float | None
    reference_clock: str
    left_device_clock: str
    right_device_clock: str
    capture_clock: str
    monotonic_clock: str


def optional_int(row: Mapping[str, str], key: str) -> int | None:
    if row.get(key, "") == "":
        return None
    return parse_int(row, key)


def optional_float_group(
    row: Mapping[str, str], fields: Iterable[str]
) -> tuple[float, ...] | None:
    names = tuple(fields)
    values = tuple(parse_float(row, name, optional=True) for name in names)
    present = tuple(value is not None for value in values)
    if any(present) and not all(present):
        raise DatasetError(f"staging 字段组 {names} 必须全部填写或全部留空")
    if not any(present):
        return None
    return tuple(float(value) for value in values if value is not None)


def safe_stage_source(directory: Path, relative: str) -> Path:
    if not relative:
        raise DatasetError("mainline staging 原始图像路径为空")
    source = (directory / relative).resolve()
    if source == directory or directory not in source.parents:
        raise DatasetError("mainline staging 原始图像路径逃逸")
    if not source.is_file():
        raise DatasetError(f"mainline staging 原始图像不存在: {relative}")
    return source


def expected_exposure(config: ResolvedConfig) -> dict[str, Any]:
    sync = config.data["cameras"]["sync"]
    return {
        "shutter_type": sync["shutter_type"],
        "source": sync["exposure_timestamp_source"],
        "delay_status": sync["exposure_delay_status"],
        "delay_ns": sync["exposure_delay_ns"],
        "delay_uncertainty_ns": sync["exposure_delay_uncertainty_ns"],
        "observation_timestamp": "exposure_midpoint",
    }


def _nonnegative_int(
    row: Mapping[str, str], key: str, *, positive: bool = False
) -> int:
    value = parse_int(row, key)
    if value < 0 or (positive and value == 0):
        qualifier = "正" if positive else "非负"
        raise DatasetError(f"staging 字段 {key} 必须是{qualifier}整数")
    return value


def _status_count(status: Mapping[str, Any], key: str) -> int:
    value = status.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetError(f"mainline status.{key} 必须是非负整数")
    return value


def _validate_writer_pgm(path: Path, width: int, height: int) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    if path.stat().st_size != len(header) + width * height:
        raise DatasetError(f"mainline PGM 尺寸与 staging 行不一致: {path}")
    with path.open("rb") as handle:
        if handle.read(len(header)) != header:
            raise DatasetError(f"mainline PGM 头部与 staging 行不一致: {path}")


def _constant(rows: Iterable[Mapping[str, str]], key: str) -> str:
    values = {row[key] for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise DatasetError(f"mainline staging 字段 {key} 必须是同一非空值")
    return next(iter(values))


def _validate_context_and_tables(
    directory: Path, config: ResolvedConfig, project_root: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    context = load_json_object(directory / "context.json")
    status = load_json_object(directory / "status_v6.json")
    if context.get("planning_only") is True:
        raise DatasetError("预演 staging 不得发布为真实 trial")
    if context.get("stage_schema_version") != MAINLINE_STAGE_SCHEMA_VERSION:
        raise DatasetError("mainline capture staging schema 版本不兼容")
    if status.get("stage_schema_version") != MAINLINE_STAGE_SCHEMA_VERSION:
        raise DatasetError("mainline capture status schema 版本不兼容")
    if status.get("complete") is not True or status.get("fatal_error") is not None:
        raise DatasetError(
            f"mainline 采集未完整结束: {status.get('fatal_error') or 'unknown'}"
        )
    if context.get("config_sha256") != config.fingerprint_sha256:
        raise DatasetError("mainline staging 配置指纹与 finalize 配置不一致")
    if context.get("calibration_id") != config.data["calibration"]["id"]:
        raise DatasetError("mainline staging 标定 ID 与 finalize 配置不一致")
    if context.get("calibration_source_sha256") != config.data["calibration"][
        "sha256"
    ]:
        raise DatasetError("mainline staging 标定哈希与 finalize 配置不一致")
    if context.get("code_worktree_clean") is False:
        raise DatasetError("脏工作树产生的 mainline staging 不得发布")
    executable_sha256 = context.get("runtime_executable_sha256")
    if executable_sha256 is not None:
        if (
            not isinstance(executable_sha256, str)
            or len(executable_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in executable_sha256
            )
        ):
            raise DatasetError("mainline staging 原生可执行文件哈希无效")
        executable = mainline_executable(project_root, config)
        if executable.is_file() and sha256_file(executable) != executable_sha256:
            raise DatasetError("mainline staging 原生可执行文件哈希与当前文件不一致")
    exposure = expected_exposure(config)
    if context.get("exposure_timestamp") != exposure:
        raise DatasetError("mainline staging 曝光时间语义与 finalize 配置不一致")
    partials = sorted(path.relative_to(directory) for path in directory.rglob("*.partial"))
    if partials:
        raise DatasetError(f"mainline staging 仍含未完成文件: {partials}")

    frames = read_csv_exact(directory / "frames_v6.csv", MAINLINE_FRAME_COLUMNS)
    detections = read_csv_exact(
        directory / "detections_v6.csv", MAINLINE_DETECTION_COLUMNS
    )
    associations = read_csv_exact(
        directory / "associations_v6.csv", MAINLINE_ASSOCIATION_COLUMNS
    )
    observations = read_csv_exact(
        directory / "observations_v6.csv", MAINLINE_OBSERVATION_COLUMNS
    )
    if not frames or len(frames) % 2 != 0:
        raise DatasetError("mainline staging 必须至少包含一对完整双目帧")
    expected_counts = {
        "camera_frames_written": len(frames),
        "stereo_frames_written": len(frames) // 2,
        "detections_written": len(detections),
        "associations_written": len(associations),
        "observations_written": len(observations),
        "raw_images_written": len(frames),
    }
    for key, expected in expected_counts.items():
        actual = _status_count(status, key)
        if actual != expected:
            raise DatasetError(
                f"mainline status.{key}={actual} 与 staging 实际计数 {expected} 不一致"
            )
    return context, exposure, frames, detections, associations, observations


def _validate_frame(
    row: dict[str, str],
    *,
    directory: Path,
    config: ResolvedConfig,
    camera_ids: Mapping[str, str],
    exposure: Mapping[str, Any],
) -> tuple[str, int, Path]:
    role = row["camera_role"]
    if role not in camera_ids or row["camera_id"] != camera_ids[role]:
        raise DatasetError("mainline 帧的相机角色或 ID 与配置不一致")
    frame_index = _nonnegative_int(row, "frame_index")
    if parse_bool(row, "dropped") or row["drop_reason"]:
        raise DatasetError("mainline fail_trial staging 不得包含 dropped 帧")

    width = _nonnegative_int(row, "image_width", positive=True)
    height = _nonnegative_int(row, "image_height", positive=True)
    stride = _nonnegative_int(row, "image_stride_bytes", positive=True)
    size = _nonnegative_int(row, "image_size_bytes", positive=True)
    roi = tuple(
        _nonnegative_int(row, name, positive=name in {"roi_width", "roi_height"})
        for name in ("roi_x", "roi_y", "roi_width", "roi_height")
    )
    expected_roi = config.data["cameras"][role]["roi"]
    if roi != (
        expected_roi["x"],
        expected_roi["y"],
        expected_roi["width"],
        expected_roi["height"],
    ):
        raise DatasetError("mainline 帧 ROI 与冻结配置不一致")
    if width != roi[2] or height != roi[3] or stride < width or size < stride * height:
        raise DatasetError("mainline 图像布局与帧 ROI 不一致")
    if row["image_pixel_format"] != row["pixel_format"]:
        raise DatasetError("mainline 图像缓冲格式与帧元数据不一致")
    if row["image_memory_kind"] not in {"host", "host_pinned"}:
        raise DatasetError("mainline staging 只能发布显式复制后的主机图像")
    raw_source = safe_stage_source(directory, row["raw_image"])
    _validate_writer_pgm(raw_source, width, height)

    exposure_us = parse_float(row, "exposure_us")
    gain_db = parse_float(row, "gain_db")
    if exposure_us is None or gain_db is None:
        raise DatasetError("mainline 曝光和增益必须存在")
    camera = config.data["cameras"][role]
    if exposure_us <= 0.0 or exposure_us != float(camera["exposure_us"]):
        raise DatasetError("mainline 实际曝光与冻结配置不一致")
    if gain_db != float(camera["gain_db"]):
        raise DatasetError("mainline 实际增益与冻结配置不一致")
    exposure_start = _nonnegative_int(row, "exposure_start_ns")
    capture = _nonnegative_int(row, "capture_timestamp_ns")
    if capture != exposure_start + round(exposure_us * 500.0):
        raise DatasetError("mainline 曝光中心不等于曝光开始加半个曝光时长")
    if (
        row["exposure_start_clock_domain"] != row["capture_clock_domain"]
        or row["receive_clock_domain"] != row["capture_clock_domain"]
    ):
        raise DatasetError("mainline 曝光、采集与 USB 实时到达必须属于同一时钟")
    receive = _nonnegative_int(row, "receive_timestamp_ns")
    if receive < capture:
        raise DatasetError("mainline USB 到达时间早于曝光中心")
    inference_done = optional_int(row, "inference_done_timestamp_ns")
    if inference_done is not None:
        if row["inference_done_clock_domain"] != row["capture_clock_domain"]:
            raise DatasetError("mainline 推理完成时间与采集时间不属于同一时钟")
        if inference_done < receive:
            raise DatasetError("mainline 推理完成时间早于 USB 到达")
    elif row["inference_done_clock_domain"]:
        raise DatasetError("mainline 空推理时间不得携带时钟域")

    receive_mono = _nonnegative_int(row, "receive_monotonic_ns")
    edge_mono = _nonnegative_int(row, "trigger_edge_monotonic_ns")
    scheduled_mono = _nonnegative_int(row, "trigger_scheduled_monotonic_ns")
    edge_real = _nonnegative_int(row, "trigger_edge_realtime_ns")
    uncertainty = _nonnegative_int(row, "trigger_write_uncertainty_ns")
    if uncertainty // 2 > edge_mono:
        raise DatasetError("mainline GPIO 写入不确定区间产生了负时间戳")
    if scheduled_mono > edge_mono or edge_mono > receive_mono:
        raise DatasetError("mainline GPIO 计划、上升沿与 USB 到达单调时序无效")
    if edge_real > exposure_start or edge_real > receive:
        raise DatasetError("mainline GPIO 实时时钟上升沿晚于曝光或 USB 到达")
    if exposure_start != edge_real + int(exposure["delay_ns"]):
        raise DatasetError("mainline 曝光开始与 GPIO 上升沿延迟配置不一致")
    if (
        row["trigger_scheduled_monotonic_clock_domain"]
        != row["trigger_edge_monotonic_clock_domain"]
        or row["receive_monotonic_clock_domain"]
        != row["trigger_edge_monotonic_clock_domain"]
    ):
        raise DatasetError("mainline GPIO 与 USB 单调时间必须属于同一时钟")
    if row["trigger_edge_realtime_clock_domain"] != row["capture_clock_domain"]:
        raise DatasetError("mainline GPIO 实时时钟与采集时钟不一致")
    mapping_valid = parse_bool(row, "capture_timestamp_mapping_valid")
    capture_uncertainty = optional_int(row, "capture_timestamp_uncertainty_ns")
    if exposure["delay_status"] == "measured":
        if not mapping_valid or capture_uncertainty is None:
            raise DatasetError("实测曝光延迟必须提供有效映射和时间不确定度")
        if capture_uncertainty != uncertainty + int(exposure["delay_uncertainty_ns"]):
            raise DatasetError("mainline 曝光时间不确定度与配置不一致")
    elif mapping_valid or capture_uncertainty is not None:
        raise DatasetError("未测量曝光延迟不得伪报有效映射或有限不确定度")

    counters = tuple(
        _nonnegative_int(row, name)
        for name in ("sensor_frame_number", "sensor_frame_counter", "sensor_trigger_index")
    )
    trigger_sequence = _nonnegative_int(row, "trigger_sequence")
    if (
        frame_index != trigger_sequence
        or (trigger_sequence & 0xFFFFFFFF) != counters[1]
    ):
        raise DatasetError(
            "mainline 帧索引、展开后的 GPIO 序号与 FrameCounter 不一致"
        )
    if any(value > 0xFFFFFFFF for value in counters):
        raise DatasetError("mainline 相机水印计数超出 uint32")
    _nonnegative_int(row, "hardware_timestamp_ns")
    parse_int(row, "stereo_hardware_timestamp_residual_ns")
    return role, frame_index, raw_source


def validate_mainline_stage(
    stage: str | Path,
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> ValidatedMainlineStage:
    directory = Path(stage).resolve()
    root = find_project_root(project_root)
    context, exposure, frames, detections, associations, observations = (
        _validate_context_and_tables(directory, config, root)
    )
    cameras = config.data["cameras"]
    camera_ids = {"left": cameras["left"]["id"], "right": cameras["right"]["id"]}
    frame_by_key: dict[tuple[str, int], dict[str, str]] = {}
    frame_pairs: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    referenced_raw: set[Path] = set()
    for row in frames:
        role, frame_index, raw_source = _validate_frame(
            row,
            directory=directory,
            config=config,
            camera_ids=camera_ids,
            exposure=exposure,
        )
        key = (role, frame_index)
        if key in frame_by_key:
            raise DatasetError(f"mainline staging 出现重复帧: {key}")
        if raw_source in referenced_raw:
            raise DatasetError("mainline 多帧引用了同一原始图像")
        frame_by_key[key] = row
        frame_pairs[frame_index][role] = row
        referenced_raw.add(raw_source)

    indexes = tuple(sorted(frame_pairs))
    if any(set(frame_pairs[index]) != {"left", "right"} for index in indexes):
        raise DatasetError("mainline staging 的每个 frame_index 必须恰有左右两帧")
    if any(right != left + 1 for left, right in zip(indexes, indexes[1:])):
        raise DatasetError("mainline staging 出现帧索引缺口")
    if set((directory / "raw" / "mainline").rglob("*.pgm")) != referenced_raw:
        raise DatasetError("mainline staging 原始图像集合与 frames_v6.csv 不一致")

    paired_rows = tuple(
        (frame_pairs[index]["left"], frame_pairs[index]["right"])
        for index in indexes
    )
    shared_fields = (
        "trigger_sequence",
        "trigger_scheduled_monotonic_ns",
        "trigger_scheduled_monotonic_clock_domain",
        "trigger_edge_monotonic_ns",
        "trigger_edge_monotonic_clock_domain",
        "trigger_edge_realtime_ns",
        "trigger_edge_realtime_clock_domain",
        "trigger_write_uncertainty_ns",
        "stereo_hardware_timestamp_residual_ns",
    )
    for left, right in paired_rows:
        if any(left[key] != right[key] for key in shared_fields):
            raise DatasetError("mainline 左右帧的共同触发事实不一致")
        if left["sensor_frame_counter"] != right["sensor_frame_counter"]:
            raise DatasetError("mainline 左右 FrameCounter 不一致")

    for role in ("left", "right"):
        role_rows = [frame_pairs[index][role] for index in indexes]
        for key in (
            "hardware_timestamp_ns",
            "capture_timestamp_ns",
            "receive_timestamp_ns",
            "receive_monotonic_ns",
        ):
            values = [parse_int(row, key) for row in role_rows]
            if any(right <= left for left, right in zip(values, values[1:])):
                raise DatasetError(f"mainline {role}.{key} 必须严格递增")
        for key in ("sensor_frame_number", "sensor_frame_counter", "sensor_trigger_index"):
            values = [parse_int(row, key) for row in role_rows]
            if any(
                not uint32_is_next(left, right)
                for left, right in zip(values, values[1:])
            ):
                raise DatasetError(f"mainline {role}.{key} 出现计数缺口")

    trigger_sequences = [parse_int(left, "trigger_sequence") for left, _ in paired_rows]
    if any(right != left + 1 for left, right in zip(trigger_sequences, trigger_sequences[1:])):
        raise DatasetError("mainline GPIO 触发序列出现缺口")
    edge_monotonic = tuple(
        parse_int(left, "trigger_edge_monotonic_ns") for left, _ in paired_rows
    )
    if any(right <= left for left, right in zip(edge_monotonic, edge_monotonic[1:])):
        raise DatasetError("mainline GPIO 上升沿必须严格递增")
    frame_period_ns = 1.0e9 / config.data["runtime"]["nominal_fps"]
    period_tolerance_ns = max(
        frame_period_ns * 0.25,
        float(cameras["sync"]["estimated_error_ns"] or 0.0) * 4.0,
    )
    for position in range(1, len(paired_rows)):
        trigger_period = edge_monotonic[position] - edge_monotonic[position - 1]
        device_period = (
            parse_int(paired_rows[position][0], "hardware_timestamp_ns")
            - parse_int(paired_rows[position - 1][0], "hardware_timestamp_ns")
        )
        if abs(trigger_period - frame_period_ns) > period_tolerance_ns:
            raise DatasetError("mainline GPIO 触发周期偏离配置帧率")
        if abs(device_period - trigger_period) > period_tolerance_ns:
            raise DatasetError("mainline 相机设备时间与 GPIO 触发序列不一致")
    sync_limit = cameras["sync"]["estimated_error_ns"]
    if sync_limit is not None and any(
        abs(parse_int(left, "stereo_hardware_timestamp_residual_ns")) > sync_limit
        for left, _ in paired_rows
    ):
        raise DatasetError("mainline 双目硬件时间残差超过配置上限")

    left_clock = _constant((left for left, _ in paired_rows), "hardware_clock_domain")
    right_clock = _constant((right for _, right in paired_rows), "hardware_clock_domain")
    if left_clock == right_clock:
        raise DatasetError("mainline 左右原始设备时钟域必须独立记录")
    capture_clock = _constant(frames, "capture_clock_domain")
    monotonic_clock = _constant(frames, "receive_monotonic_clock_domain")
    if capture_clock == monotonic_clock:
        raise DatasetError("mainline 实时与单调时钟域不得混用")
    return ValidatedMainlineStage(
        directory=directory,
        context=context,
        expected_exposure=exposure,
        frames=frames,
        detections=detections,
        associations=associations,
        observations=observations,
        camera_ids=camera_ids,
        frame_by_key=frame_by_key,
        frame_pairs=dict(frame_pairs),
        indexes=indexes,
        paired_rows=paired_rows,
        edge_monotonic=edge_monotonic,
        sync_limit_ns=sync_limit,
        reference_clock=cameras["sync"]["clock_domain"],
        left_device_clock=left_clock,
        right_device_clock=right_clock,
        capture_clock=capture_clock,
        monotonic_clock=monotonic_clock,
    )
