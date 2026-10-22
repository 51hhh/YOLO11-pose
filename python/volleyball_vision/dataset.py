"""真实双目采集数据包 2.1.0 的流式写入和独立校验。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, TextIO

import jsonschema

from .config import find_project_root


class DatasetError(ValueError):
    """数据包不满足采集契约。"""


@dataclass(frozen=True, slots=True)
class TrialValidationSummary:
    trial_id: str
    table_rows: Mapping[str, int]
    raw_files: int


@dataclass(frozen=True, slots=True)
class DatasetValidationSummary:
    root: Path
    trials: tuple[TrialValidationSummary, ...]
    calibrations: int
    partial_directories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _AcceptedPair:
    position_m: tuple[float, float, float]
    covariance_m2: tuple[float, float, float, float, float, float] | None
    epipolar_error_px: float | None


_COVARIANCE_3D_FIELDS = (
    "cov_xx_m2",
    "cov_xy_m2",
    "cov_xz_m2",
    "cov_yy_m2",
    "cov_yz_m2",
    "cov_zz_m2",
)
_CAMERA_TIMESTAMP_SEMANTICS = {
    "stereo_midpoint_capture",
    "left_capture",
    "right_capture",
    "hardware_trigger",
}


def _contract_root(project_root: str | Path | None = None) -> Path:
    root = find_project_root(project_root)
    return root / "contracts" / "capture" / "v2.1.0"


def load_table_contract(project_root: str | Path | None = None) -> dict[str, Any]:
    path = _contract_root(project_root) / "tables.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetError(f"缺少文件: {path}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"JSON 无效 {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetError(f"JSON 顶层必须是对象: {path}")
    return payload


def _validate_json_schema(path: Path, schema_path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    schema = _load_json(schema_path)
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        details = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            details.append(f"{location}: {error.message}")
        raise DatasetError(
            f"{path} schema 校验失败:\n" + "\n".join(f"  - {item}" for item in details)
        )
    return payload


def _safe_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{field_name} 必须是非空相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise DatasetError(f"{field_name} 必须留在 trial 目录内: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: str, *, field_name: str, allow_blank: bool = False) -> float | None:
    if value == "" and allow_blank:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{field_name} 必须是数字，实际为 {value!r}") from error
    if not math.isfinite(parsed):
        raise DatasetError(f"{field_name} 必须为有限数")
    return parsed


def _integer(value: str, *, field_name: str, allow_blank: bool = False) -> int | None:
    if value == "" and allow_blank:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{field_name} 必须是整数，实际为 {value!r}") from error
    if parsed < 0:
        raise DatasetError(f"{field_name} 不得为负")
    return parsed


def _boolean(value: str, *, field_name: str) -> bool:
    if value in {"1", "true", "True"}:
        return True
    if value in {"0", "false", "False"}:
        return False
    raise DatasetError(f"{field_name} 必须是布尔值，实际为 {value!r}")


def _optional_finite_group(
    row: Mapping[str, str],
    fields: tuple[str, ...],
    *,
    prefix: str,
) -> tuple[float, ...] | None:
    values = [row[field] for field in fields]
    if all(value == "" for value in values):
        return None
    if any(value == "" for value in values):
        raise DatasetError(f"{prefix}: {fields[0]}..{fields[-1]} 必须整组填写或留空")
    parsed = tuple(
        _finite(value, field_name=f"{prefix}.{field}")
        for field, value in zip(fields, values)
    )
    assert all(value is not None for value in parsed)
    return tuple(float(value) for value in parsed)


def _covariance3(
    row: Mapping[str, str],
    *,
    prefix: str,
) -> tuple[float, float, float, float, float, float] | None:
    values = _optional_finite_group(
        row,
        _COVARIANCE_3D_FIELDS,
        prefix=prefix,
    )
    if values is None:
        return None
    xx, xy, xz, yy, yz, zz = values
    scale = max(abs(value) for value in values)
    if min(xx, yy, zz) < 0.0:
        raise DatasetError(f"{prefix}: 3D 协方差对角线不得为负")
    if scale == 0.0:
        return (xx, xy, xz, yy, yz, zz)
    xx, xy, xz, yy, yz, zz = (
        value / scale for value in (xx, xy, xz, yy, yz, zz)
    )
    second_order_minors = (
        xx * yy - xy * xy,
        xx * zz - xz * xz,
        yy * zz - yz * yz,
    )
    if min(second_order_minors) < -1.0e-9:
        raise DatasetError(f"{prefix}: 3D 协方差不是半正定矩阵")
    determinant = (
        xx * yy * zz
        + 2.0 * xy * xz * yz
        - xx * yz * yz
        - yy * xz * xz
        - zz * xy * xy
    )
    if determinant < -1.0e-9:
        raise DatasetError(f"{prefix}: 3D 协方差不是半正定矩阵")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
    )


def _covariance2(
    row: Mapping[str, str],
    *,
    prefix: str,
) -> tuple[float, float, float] | None:
    fields = ("cov_uu_px2", "cov_uv_px2", "cov_vv_px2")
    values = _optional_finite_group(row, fields, prefix=prefix)
    if values is None:
        return None
    uu, uv, vv = values
    if uu < 0.0 or vv < 0.0:
        raise DatasetError(f"{prefix}: 2D 协方差对角线不得为负")
    scale = max(abs(value) for value in values)
    if scale > 0.0:
        normalized_uu = uu / scale
        normalized_uv = uv / scale
        normalized_vv = vv / scale
    else:
        normalized_uu = normalized_uv = normalized_vv = 0.0
    if normalized_uu * normalized_vv - normalized_uv * normalized_uv < -1.0e-9:
        raise DatasetError(f"{prefix}: 2D 协方差不是半正定矩阵")
    return (uu, uv, vv)


def _same_measurement(
    actual: tuple[float, ...] | None,
    expected: tuple[float, ...] | None,
) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return len(actual) == len(expected) and all(
        math.isclose(lhs, rhs, rel_tol=1.0e-12, abs_tol=1.0e-12)
        for lhs, rhs in zip(actual, expected)
    )


def _read_table(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise DatasetError(
                    f"{path} 表头不匹配\n"
                    f"  expected={expected_columns}\n  actual={reader.fieldnames}"
                )
            rows = list(reader)
    except FileNotFoundError as error:
        raise DatasetError(f"缺少表文件: {path}") from error
    return rows


def _matrix_translation(transform: Any, *, field_name: str) -> tuple[float, float, float]:
    try:
        return (
            float(transform[0][3]),
            float(transform[1][3]),
            float(transform[2][3]),
        )
    except (IndexError, TypeError, ValueError) as error:
        raise DatasetError(f"{field_name} 必须是 4x4 齐次变换") from error


def validate_calibration(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    calibration_path = Path(path)
    schema_path = _contract_root(project_root) / "calibration.schema.json"
    payload = _validate_json_schema(calibration_path, schema_path)
    cameras = payload["cameras"]
    roles = payload["camera_roles"]
    camera_ids = [camera["camera_id"] for camera in cameras]
    if roles["left"] == roles["right"]:
        raise DatasetError("标定左右相机角色不得相同")
    if camera_ids != [roles["left"], roles["right"]]:
        raise DatasetError("标定 cameras 必须按 camera_roles.left/right 顺序保存")
    if cameras[0]["clock_domain"] != cameras[1]["clock_domain"]:
        raise DatasetError("左右相机必须共享可审计的采集时钟域")
    left = _matrix_translation(
        cameras[0]["transform_body_from_camera"],
        field_name="left.transform_body_from_camera",
    )
    right = _matrix_translation(
        cameras[1]["transform_body_from_camera"],
        field_name="right.transform_body_from_camera",
    )
    baseline = math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(left, right)))
    if not math.isclose(baseline, float(payload["baseline_m"]), abs_tol=1.0e-9):
        raise DatasetError("标定 baseline_m 与两相机外参不一致")
    return payload


def _pipeline_fingerprint(pipeline: Mapping[str, Any]) -> str:
    canonical = dict(pipeline)
    canonical.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def with_pipeline_fingerprint(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """返回带有确定性指纹的流水线元数据。"""
    result = dict(pipeline)
    result["fingerprint_sha256"] = _pipeline_fingerprint(result)
    return result


def _mapped_timestamp_ns(
    source_ns: int,
    source_clock: str,
    reference_clock: str,
    mappings: Mapping[str, list[dict[str, str]]],
) -> int:
    if source_clock == reference_clock:
        return source_ns
    candidates = mappings.get(source_clock, [])
    matched: list[dict[str, str]] = []
    for mapping in candidates:
        lower = _integer(
            mapping["valid_from_source_ns"],
            field_name="clock_mappings.valid_from_source_ns",
            allow_blank=True,
        )
        upper = _integer(
            mapping["valid_to_source_ns"],
            field_name="clock_mappings.valid_to_source_ns",
            allow_blank=True,
        )
        if (lower is None or source_ns >= lower) and (upper is None or source_ns <= upper):
            matched.append(mapping)
    if len(matched) != 1:
        raise DatasetError(
            f"时钟 {source_clock!r} 的时间 {source_ns} 需要惟一映射到 {reference_clock!r}"
        )
    mapping = matched[0]
    if mapping["reference_clock_domain"] != reference_clock:
        raise DatasetError("时钟映射必须直接指向轨迹参考时钟")
    source_anchor = int(mapping["source_anchor_ns"])
    reference_anchor = int(mapping["reference_anchor_ns"])
    drift_ppm = float(mapping["drift_ppm"])
    return reference_anchor + round((source_ns - source_anchor) * (1.0 + drift_ppm * 1.0e-6))


def _validate_trial_tables(
    directory: Path,
    metadata: Mapping[str, Any],
    calibration: Mapping[str, Any],
    table_contract: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for name, spec in table_contract["tables"].items():
        declaration = metadata["trajectory"] if name == "trajectory" else metadata["sidecars"].get(name)
        if declaration is None:
            if spec["required"]:
                raise DatasetError(f"缺少必需表声明: {name}")
            continue
        if declaration["file"] != spec["file"]:
            raise DatasetError(f"{name}.file 必须为 {spec['file']!r}")
        if declaration["columns"] != spec["columns"]:
            raise DatasetError(f"{name}.columns 与 2.1.0 契约不一致")
        rows = _read_table(directory / spec["file"], spec["columns"])
        if name != "trajectory" and declaration["row_count"] != len(rows):
            raise DatasetError(f"{name}.row_count 与 CSV 实际行数不一致")
        if name == "trajectory" and declaration.get("sample_count") != len(rows):
            raise DatasetError("trajectory.sample_count 与 CSV 实际行数不一致")
        tables[name] = rows

    roles = calibration["camera_roles"]
    camera_ids = {roles["left"], roles["right"]}
    exposure_timing = metadata["capture"].get("exposure_timing")
    if exposure_timing is not None:
        if (
            exposure_timing["delay_status"] == "assumed_zero_unmeasured"
            and (
                exposure_timing["delay_ns"] != 0
                or exposure_timing["delay_uncertainty_ns"] is not None
            )
        ):
            raise DatasetError(
                "未测量曝光延迟必须记录 delay_ns=0 且 delay_uncertainty_ns=null"
            )
    frames: dict[tuple[str, int], dict[str, str]] = {}
    last_frame_index: dict[str, int] = {}
    last_capture_ns: dict[str, int] = {}
    for line, row in enumerate(tables["camera_frames"], start=2):
        prefix = f"camera_frames.csv:{line}"
        camera_id = row["camera_id"]
        if camera_id not in camera_ids:
            raise DatasetError(f"{prefix}: camera_id 不属于标定左右相机")
        frame_index = _integer(row["frame_index"], field_name=f"{prefix}.frame_index")
        capture_ns = _integer(row["capture_timestamp_ns"], field_name=f"{prefix}.capture_timestamp_ns")
        assert frame_index is not None and capture_ns is not None
        key = (camera_id, frame_index)
        if key in frames:
            raise DatasetError(f"{prefix}: 重复 camera/frame 键 {key}")
        if camera_id in last_frame_index and frame_index <= last_frame_index[camera_id]:
            raise DatasetError(f"{prefix}: frame_index 必须严格递增")
        if camera_id in last_capture_ns and capture_ns <= last_capture_ns[camera_id]:
            raise DatasetError(f"{prefix}: capture_timestamp_ns 必须严格递增")
        last_frame_index[camera_id] = frame_index
        last_capture_ns[camera_id] = capture_ns
        roi_values = tuple(
            _integer(row[name], field_name=f"{prefix}.{name}")
            for name in ("roi_x", "roi_y", "roi_width", "roi_height")
        )
        assert all(value is not None for value in roi_values)
        if roi_values[2] == 0 or roi_values[3] == 0:
            raise DatasetError(f"{prefix}: ROI 宽高必须大于 0")
        timestamps = [
            _integer(row[name], field_name=f"{prefix}.{name}", allow_blank=name not in {"hardware_timestamp_ns", "capture_timestamp_ns"})
            for name in ("exposure_start_ns", "capture_timestamp_ns", "receive_timestamp_ns", "inference_done_timestamp_ns")
        ]
        known = [item for item in timestamps if item is not None]
        if any(right < left for left, right in zip(known, known[1:])):
            raise DatasetError(f"{prefix}: 曝光/采集/接收/推理时间顺序无效")
        if exposure_timing is not None:
            exposure_start_ns = timestamps[0]
            capture_timestamp_ns = timestamps[1]
            exposure_us = _finite(
                row["exposure_us"], field_name=f"{prefix}.exposure_us"
            )
            if exposure_start_ns is None or capture_timestamp_ns is None:
                raise DatasetError(
                    f"{prefix}: 声明曝光时间模型后必须提供曝光开始和采集时间"
                )
            if exposure_us is None or exposure_us <= 0.0:
                raise DatasetError(f"{prefix}: exposure_us 必须为正数")
            if (
                exposure_timing["shutter_type"] == "global"
                and exposure_timing["observation_timestamp"]
                == "exposure_midpoint"
            ):
                expected_capture_ns = exposure_start_ns + round(
                    float(exposure_us) * 500.0
                )
                if capture_timestamp_ns != expected_capture_ns:
                    raise DatasetError(
                        f"{prefix}: 全局快门 capture_timestamp_ns "
                        "必须等于曝光开始加半个曝光时长"
                    )
        dropped = _boolean(row["dropped"], field_name=f"{prefix}.dropped")
        if dropped and not row["drop_reason"]:
            raise DatasetError(f"{prefix}: 丢帧必须记录 drop_reason")
        frames[key] = row

    pipeline_ids: set[str] = set()
    for pipeline in metadata["perception_pipelines"]:
        pipeline_id = pipeline["pipeline_id"]
        if pipeline_id in pipeline_ids:
            raise DatasetError(f"重复 perception pipeline ID: {pipeline_id}")
        if pipeline["fingerprint_sha256"] != _pipeline_fingerprint(pipeline):
            raise DatasetError(f"感知流水线指纹不匹配: {pipeline_id}")
        pipeline_ids.add(pipeline_id)

    candidates: dict[tuple[str, int, int], dict[str, str]] = {}
    selected_candidates: set[tuple[str, int, int]] = set()
    for line, row in enumerate(tables["detections_2d"], start=2):
        prefix = f"detections_2d.csv:{line}"
        camera_id = row["camera_id"]
        frame_index = _integer(row["frame_index"], field_name=f"{prefix}.frame_index")
        candidate_id = _integer(row["candidate_id"], field_name=f"{prefix}.candidate_id")
        assert frame_index is not None and candidate_id is not None
        if (camera_id, frame_index) not in frames:
            raise DatasetError(f"{prefix}: 引用不存在的相机帧")
        key = (camera_id, frame_index, candidate_id)
        if key in candidates:
            raise DatasetError(f"{prefix}: 重复候选 ID")
        if row["source"] not in pipeline_ids:
            raise DatasetError(f"{prefix}: source 未引用已指纹的感知流水线")
        frame_row = frames[(camera_id, frame_index)]
        if _boolean(
            frame_row["dropped"],
            field_name=f"{prefix}.source_frame_dropped",
        ):
            raise DatasetError(f"{prefix}: 检测不得引用 dropped 帧")
        width = int(frame_row["roi_width"])
        height = int(frame_row["roi_height"])
        u_px = _finite(row["u_px"], field_name=f"{prefix}.u_px")
        v_px = _finite(row["v_px"], field_name=f"{prefix}.v_px")
        assert u_px is not None and v_px is not None
        if not 0.0 <= u_px < width or not 0.0 <= v_px < height:
            raise DatasetError(f"{prefix}: 检测中心必须位于来源图像内")
        radius = _finite(
            row["radius_px"],
            field_name=f"{prefix}.radius_px",
            allow_blank=True,
        )
        if radius is not None and (
            radius <= 0.0 or radius > max(width, height)
        ):
            raise DatasetError(f"{prefix}: radius_px 必须大于 0 且不超过图像尺度")
        bounding_box = _optional_finite_group(
            row,
            ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
            prefix=prefix,
        )
        if bounding_box is not None:
            bbox_x, bbox_y, bbox_w, bbox_h = bounding_box
            if (
                bbox_w <= 0.0
                or bbox_h <= 0.0
                or bbox_w > width
                or bbox_h > height
            ):
                raise DatasetError(
                    f"{prefix}: bbox 宽高必须大于 0 且不超过图像尺寸"
                )
            if not (
                bbox_x < width
                and bbox_y < height
                and bbox_x + bbox_w > 0.0
                and bbox_y + bbox_h > 0.0
            ):
                raise DatasetError(f"{prefix}: bbox 必须与来源图像相交")
            if not (
                bbox_x <= u_px <= bbox_x + bbox_w
                and bbox_y <= v_px <= bbox_y + bbox_h
            ):
                raise DatasetError(f"{prefix}: 检测中心必须位于 bbox 内")
        _optional_finite_group(
            row,
            ("du_dt_pxps", "dv_dt_pxps"),
            prefix=prefix,
        )
        _covariance2(row, prefix=prefix)
        score = _finite(row["score"], field_name=f"{prefix}.score", allow_blank=True)
        if score is not None and not 0.0 <= score <= 1.0:
            raise DatasetError(f"{prefix}: score 必须在 [0,1]")
        if _boolean(row["selected"], field_name=f"{prefix}.selected"):
            selected_candidates.add(key)
        candidates[key] = row

    accepted_pairs: dict[tuple[int, int], _AcceptedPair] = {}
    accepted_candidates: set[tuple[str, int, int]] = set()
    pair_ids: set[int] = set()
    for line, row in enumerate(tables["stereo_pairs"], start=2):
        prefix = f"stereo_pairs.csv:{line}"
        pair_id = _integer(row["pair_id"], field_name=f"{prefix}.pair_id")
        left_frame = _integer(row["left_frame_index"], field_name=f"{prefix}.left_frame_index")
        right_frame = _integer(row["right_frame_index"], field_name=f"{prefix}.right_frame_index")
        left_candidate = _integer(row["left_candidate_id"], field_name=f"{prefix}.left_candidate_id")
        right_candidate = _integer(row["right_candidate_id"], field_name=f"{prefix}.right_candidate_id")
        assert None not in (pair_id, left_frame, right_frame, left_candidate, right_candidate)
        if pair_id in pair_ids:
            raise DatasetError(f"{prefix}: 重复 pair_id")
        pair_ids.add(pair_id)
        left_key = (roles["left"], left_frame, left_candidate)
        right_key = (roles["right"], right_frame, right_candidate)
        if left_key not in candidates or right_key not in candidates:
            raise DatasetError(f"{prefix}: 候选引用不存在或左右角色错误")
        left_capture = _integer(
            row["left_capture_ns"],
            field_name=f"{prefix}.left_capture_ns",
        )
        right_capture = _integer(
            row["right_capture_ns"],
            field_name=f"{prefix}.right_capture_ns",
        )
        assert left_capture is not None and right_capture is not None
        if left_capture != int(frames[(roles["left"], left_frame)]["capture_timestamp_ns"]):
            raise DatasetError(f"{prefix}: left_capture_ns 与左帧不一致")
        if right_capture != int(frames[(roles["right"], right_frame)]["capture_timestamp_ns"]):
            raise DatasetError(f"{prefix}: right_capture_ns 与右帧不一致")
        delta = _finite(row["delta_t_lr_ns"], field_name=f"{prefix}.delta_t_lr_ns")
        if not math.isclose(float(delta), right_capture - left_capture, abs_tol=0.5):
            raise DatasetError(f"{prefix}: delta_t_lr_ns 不等于右减左曝光时间")
        disparity = _finite(
            row["disparity_px"],
            field_name=f"{prefix}.disparity_px",
            allow_blank=True,
        )
        epipolar_error = _finite(
            row["epipolar_error_px"],
            field_name=f"{prefix}.epipolar_error_px",
            allow_blank=True,
        )
        if epipolar_error is not None and epipolar_error < 0.0:
            raise DatasetError(f"{prefix}: epipolar_error_px 不得为负")
        position = _optional_finite_group(
            row,
            ("x_rig_m", "y_rig_m", "z_rig_m"),
            prefix=prefix,
        )
        covariance = _covariance3(row, prefix=prefix)
        accepted = _boolean(row["accepted"], field_name=f"{prefix}.accepted")
        if accepted:
            if position is None or row["reject_reason"]:
                raise DatasetError(f"{prefix}: 接受配对必须有 3D 且不得有拒绝理由")
            if position[2] <= 0.0:
                raise DatasetError(f"{prefix}: 接受配对的 z_rig_m 必须大于 0")
            if disparity is None or disparity <= 0.0:
                raise DatasetError(f"{prefix}: 接受配对必须有正视差")
            frame_pair = (left_frame, right_frame)
            if frame_pair in accepted_pairs:
                raise DatasetError(f"{prefix}: 同一左右帧只能有一个 accepted 配对")
            accepted_pairs[frame_pair] = _AcceptedPair(
                position_m=(position[0], position[1], position[2]),
                covariance_m2=covariance,
                epipolar_error_px=epipolar_error,
            )
            accepted_candidates.update((left_key, right_key))
        elif not row["reject_reason"]:
            raise DatasetError(f"{prefix}: 拒绝配对必须有 reject_reason")

    if selected_candidates != accepted_candidates:
        missing = sorted(accepted_candidates - selected_candidates)
        unexpected = sorted(selected_candidates - accepted_candidates)
        raise DatasetError(
            "detections_2d.selected 必须与 accepted 配对引用完全一致: "
            f"missing={missing}, unexpected={unexpected}"
        )

    mappings_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    mapping_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    reference_clock = metadata["capture"]["trajectory_timebase"]["clock_domain"]
    for line, row in enumerate(tables["clock_mappings"], start=2):
        prefix = f"clock_mappings.csv:{line}"
        source_clock = row["source_clock_domain"]
        if source_clock == row["reference_clock_domain"]:
            raise DatasetError(f"{prefix}: source/reference 时钟不得相同")
        if row["reference_clock_domain"] != reference_clock:
            raise DatasetError(f"{prefix}: 映射必须直接指向轨迹时钟")
        lower_value = _integer(row["valid_from_source_ns"], field_name=f"{prefix}.valid_from", allow_blank=True)
        upper_value = _integer(row["valid_to_source_ns"], field_name=f"{prefix}.valid_to", allow_blank=True)
        lower = -math.inf if lower_value is None else float(lower_value)
        upper = math.inf if upper_value is None else float(upper_value)
        if upper < lower:
            raise DatasetError(f"{prefix}: 映射有效区间倒置")
        for old_lower, old_upper in mapping_intervals[source_clock]:
            if max(lower, old_lower) <= min(upper, old_upper):
                raise DatasetError(f"{prefix}: 同源时钟映射有效区间重叠")
        mapping_intervals[source_clock].append((lower, upper))
        mappings_by_source[source_clock].append(row)

    origin_ns = int(metadata["capture"]["trajectory_timebase"]["origin_ns"])
    semantics = metadata["capture"]["trajectory_timebase"]["timestamp_semantics"]
    previous_sample = -1
    previous_time_ns = -1
    referenced_pairs: set[tuple[int, int]] = set()
    for line, row in enumerate(tables["trajectory"], start=2):
        prefix = f"trajectory.csv:{line}"
        sample = _integer(row["sample_index"], field_name=f"{prefix}.sample_index")
        assert sample is not None
        timestamp_s = _finite(
            row["timestamp_s"],
            field_name=f"{prefix}.timestamp_s",
        )
        assert timestamp_s is not None
        timestamp_ns = round(timestamp_s * 1.0e9)
        if sample <= previous_sample or timestamp_ns <= previous_time_ns:
            raise DatasetError(f"{prefix}: sample_index 和 timestamp_s 必须严格递增")
        previous_sample = sample
        previous_time_ns = timestamp_ns
        position = tuple(
            _finite(row[field], field_name=f"{prefix}.{field}")
            for field in ("x_m", "y_m", "z_m")
        )
        assert all(value is not None for value in position)
        trajectory_position = tuple(float(value) for value in position)
        if trajectory_position[2] <= 0.0:
            raise DatasetError(f"{prefix}: z_m 必须大于 0")
        confidence = _finite(
            row["confidence"],
            field_name=f"{prefix}.confidence",
            allow_blank=True,
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise DatasetError(f"{prefix}: confidence 必须在 [0,1]")
        reprojection_error = _finite(
            row["reprojection_error_px"],
            field_name=f"{prefix}.reprojection_error_px",
            allow_blank=True,
        )
        if reprojection_error is not None and reprojection_error < 0.0:
            raise DatasetError(f"{prefix}: reprojection_error_px 不得为负")
        trajectory_covariance = _covariance3(row, prefix=prefix)
        left_text = row["left_frame_index"]
        right_text = row["right_frame_index"]
        if (left_text == "") != (right_text == ""):
            raise DatasetError(f"{prefix}: 左右帧引用必须同时存在或同时留空")
        if semantics in _CAMERA_TIMESTAMP_SEMANTICS and not left_text:
            raise DatasetError(f"{prefix}: 相机时基轨迹必须引用左右帧")
        if left_text:
            left_index = _integer(
                left_text,
                field_name=f"{prefix}.left_frame_index",
            )
            right_index = _integer(
                right_text,
                field_name=f"{prefix}.right_frame_index",
            )
            assert left_index is not None and right_index is not None
            left_frame_row = frames.get((roles["left"], left_index))
            right_frame_row = frames.get((roles["right"], right_index))
            if left_frame_row is None or right_frame_row is None:
                raise DatasetError(f"{prefix}: 引用不存在的左右帧")
            frame_pair = (left_index, right_index)
            accepted_pair = accepted_pairs.get(frame_pair)
            if accepted_pair is None:
                raise DatasetError(f"{prefix}: 轨迹样本必须引用已接受的直接配对")
            if frame_pair in referenced_pairs:
                raise DatasetError(f"{prefix}: accepted 配对只能生成一个轨迹样本")
            referenced_pairs.add(frame_pair)
            if not _same_measurement(
                trajectory_position,
                accepted_pair.position_m,
            ):
                raise DatasetError(f"{prefix}: 3D 位置与 accepted 配对不一致")
            if not _same_measurement(
                trajectory_covariance,
                accepted_pair.covariance_m2,
            ):
                raise DatasetError(f"{prefix}: 协方差与 accepted 配对不一致")
            if not _same_measurement(
                None if reprojection_error is None else (reprojection_error,),
                None
                if accepted_pair.epipolar_error_px is None
                else (accepted_pair.epipolar_error_px,),
            ):
                raise DatasetError(
                    f"{prefix}: 重投影误差与 accepted 配对极线误差不一致"
                )
            left_ns = _mapped_timestamp_ns(
                int(left_frame_row["capture_timestamp_ns"]),
                left_frame_row["clock_domain"],
                reference_clock,
                mappings_by_source,
            )
            right_ns = _mapped_timestamp_ns(
                int(right_frame_row["capture_timestamp_ns"]),
                right_frame_row["clock_domain"],
                reference_clock,
                mappings_by_source,
            )
            if semantics == "stereo_midpoint_capture":
                expected = round((left_ns + right_ns) / 2) - origin_ns
            elif semantics == "hardware_trigger":
                if left_ns != right_ns:
                    raise DatasetError(
                        f"{prefix}: hardware_trigger 要求左右 capture_timestamp_ns 相同"
                    )
                expected = left_ns - origin_ns
            elif semantics == "left_capture":
                expected = left_ns - origin_ns
            elif semantics == "right_capture":
                expected = right_ns - origin_ns
            else:
                expected = timestamp_ns
            if abs(timestamp_ns - expected) > 1:
                raise DatasetError(f"{prefix}: timestamp_s 与宣布的轨迹时基不一致")

    if semantics in _CAMERA_TIMESTAMP_SEMANTICS:
        missing_trajectory = sorted(set(accepted_pairs) - referenced_pairs)
        if missing_trajectory:
            raise DatasetError(
                "accepted 配对必须各自生成一个直接轨迹样本: "
                f"missing={missing_trajectory}"
            )

    for line, row in enumerate(tables["processing_timings"], start=2):
        prefix = f"processing_timing.csv:{line}"
        start = int(row["start_timestamp_ns"])
        done = int(row["done_timestamp_ns"])
        if done < start:
            raise DatasetError(f"{prefix}: done_timestamp_ns 早于 start_timestamp_ns")
        timing_clock = row["clock_domain"]
        _mapped_timestamp_ns(start, timing_clock, reference_clock, mappings_by_source)
        _mapped_timestamp_ns(done, timing_clock, reference_clock, mappings_by_source)
        if row["input_type"] == "camera_frame":
            try:
                camera_id, frame_index_text = row["input_id"].rsplit(":", 1)
                frame_index = int(frame_index_text)
            except (ValueError, TypeError) as error:
                raise DatasetError(f"{prefix}: camera_frame input_id 必须为 camera_id:index") from error
            frame = frames.get((camera_id, frame_index))
            if frame is None:
                raise DatasetError(f"{prefix}: 处理记录引用不存在的帧")
            expected_reference = _mapped_timestamp_ns(
                int(frame["capture_timestamp_ns"]),
                frame["clock_domain"],
                reference_clock,
                mappings_by_source,
            )
            if int(row["capture_reference_ns"]) != expected_reference:
                raise DatasetError(f"{prefix}: capture_reference_ns 与帧曝光时间映射不一致")
    return tables


def validate_trial(
    trial_directory: str | Path,
    *,
    dataset_root: str | Path | None = None,
    project_root: str | Path | None = None,
    verify_hashes: bool = True,
    allow_staging_name: bool = False,
) -> TrialValidationSummary:
    directory = Path(trial_directory).resolve()
    root = Path(dataset_root).resolve() if dataset_root else directory.parent.parent
    metadata = _validate_json_schema(
        directory / "metadata.json",
        _contract_root(project_root) / "metadata.schema.json",
    )
    if directory.name != metadata["trial_id"] and not allow_staging_name:
        raise DatasetError("trial 目录名必须等于 metadata.trial_id")
    calibration_path = root / "calibrations" / f"{metadata['calibration_id']}.json"
    calibration = validate_calibration(calibration_path, project_root=project_root)
    table_contract = load_table_contract(project_root)
    tables = _validate_trial_tables(directory, metadata, calibration, table_contract)

    seen_labels: set[str] = set()
    seen_paths: set[str] = set()
    for entry in metadata["raw_manifest"]:
        label = entry["label"]
        relative = str(_safe_relative_path(entry["relative_path"], field_name="raw_manifest.relative_path"))
        if label in seen_labels or relative in seen_paths:
            raise DatasetError("原始文件 label 和 relative_path 必须唯一")
        seen_labels.add(label)
        seen_paths.add(relative)
        path = directory / relative
        if not path.is_file():
            raise DatasetError(f"原始文件不存在: {relative}")
        if path.stat().st_size != entry["size_bytes"]:
            raise DatasetError(f"原始文件大小不匹配: {relative}")
        if verify_hashes and _sha256(path) != entry["sha256"]:
            raise DatasetError(f"原始文件 SHA-256 不匹配: {relative}")
    if metadata["source_files"] != {
        entry["label"]: entry["relative_path"] for entry in metadata["raw_manifest"]
    }:
        raise DatasetError("source_files 必须与 raw_manifest 的 label/path 完全一致")
    return TrialValidationSummary(
        trial_id=metadata["trial_id"],
        table_rows={name: len(rows) for name, rows in tables.items()},
        raw_files=len(metadata["raw_manifest"]),
    )


def validate_dataset(
    dataset_root: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_hashes: bool = True,
) -> DatasetValidationSummary:
    root = Path(dataset_root).resolve()
    calibration_dir = root / "calibrations"
    trials_dir = root / "trials"
    if not calibration_dir.is_dir() or not trials_dir.is_dir():
        raise DatasetError("数据集必须包含 calibrations/ 和 trials/")
    calibrations = sorted(calibration_dir.glob("*.json"))
    for path in calibrations:
        validate_calibration(path, project_root=project_root)
    partial = tuple(
        path.name for path in sorted(trials_dir.iterdir())
        if path.is_dir() and path.name.startswith(".")
    )
    trials = tuple(
        validate_trial(
            path,
            dataset_root=root,
            project_root=project_root,
            verify_hashes=verify_hashes,
        )
        for path in sorted(trials_dir.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    )
    return DatasetValidationSummary(
        root=root,
        trials=trials,
        calibrations=len(calibrations),
        partial_directories=partial,
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


@dataclass
class _OpenTable:
    handle: TextIO
    writer: csv.DictWriter
    columns: list[str]
    row_count: int = 0


class TrialPackageWriter:
    """把一次试验流式写入临时目录，校验后原子发布。"""

    def __init__(
        self,
        dataset_root: str | Path,
        metadata: Mapping[str, Any],
        *,
        project_root: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.project_root = find_project_root(project_root)
        self.metadata = dict(metadata)
        trial_id = self.metadata.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or any(character in trial_id for character in "/\\"):
            raise DatasetError("metadata.trial_id 无效")
        self.trial_id = trial_id
        self.trials_root = self.dataset_root / "trials"
        self.trials_root.mkdir(parents=True, exist_ok=True)
        (self.dataset_root / "calibrations").mkdir(parents=True, exist_ok=True)
        self.destination = self.trials_root / self.trial_id
        if self.destination.exists():
            raise FileExistsError(self.destination)
        self.stage = Path(tempfile.mkdtemp(prefix=f".{self.trial_id}-", suffix=".partial", dir=self.trials_root))
        self._contract = load_table_contract(self.project_root)
        self._tables: dict[str, _OpenTable] = {}
        self._raw_manifest: list[dict[str, Any]] = []
        self._closed = False
        self._published = False
        try:
            for name, spec in self._contract["tables"].items():
                if not spec["required"]:
                    continue
                path = self.stage / spec["file"]
                handle = path.open("x", encoding="utf-8", newline="")
                writer = csv.DictWriter(handle, fieldnames=spec["columns"], lineterminator="\n")
                writer.writeheader()
                self._tables[name] = _OpenTable(handle, writer, list(spec["columns"]))
        except Exception:
            self.abort()
            raise

    def append(self, table: str, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise DatasetError("写入器已关闭")
        if table not in self._tables:
            raise DatasetError(f"未打开的必需表: {table}")
        state = self._tables[table]
        extra = sorted(set(record) - set(state.columns))
        if extra:
            raise DatasetError(f"{table} 包含契约外字段: {extra}")
        row = {column: _csv_value(record.get(column)) for column in state.columns}
        state.writer.writerow(row)
        state.row_count += 1

    def add_raw_source(
        self,
        label: str,
        source: str | Path,
        *,
        filename: str | None = None,
    ) -> Path:
        if self._closed:
            raise DatasetError("写入器已关闭")
        if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in label):
            raise DatasetError(f"原始文件 label 无效: {label!r}")
        if any(entry["label"] == label for entry in self._raw_manifest):
            raise DatasetError(f"重复原始文件 label: {label}")
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise DatasetError(f"原始源文件不存在: {source_path}")
        safe_name = filename or source_path.name
        relative = _safe_relative_path(f"raw/{safe_name}", field_name="raw filename")
        destination = self.stage.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise DatasetError(f"重复原始文件路径: {relative}")
        shutil.copyfile(source_path, destination)
        self._raw_manifest.append(
            {
                "label": label,
                "relative_path": str(relative),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
        return destination

    def finalize(self) -> Path:
        if self._closed:
            raise DatasetError("写入器已关闭")
        for state in self._tables.values():
            state.handle.flush()
            os.fsync(state.handle.fileno())
            state.handle.close()
        self._closed = True

        payload = dict(self.metadata)
        payload["schema_version"] = "2.1.0"
        payload["capture_layout"] = "vehicle_stereo_v2"
        trajectory_state = self._tables["trajectory"]
        payload["trajectory"] = {
            "file": self._contract["tables"]["trajectory"]["file"],
            "sample_count": trajectory_state.row_count,
            "columns": trajectory_state.columns,
        }
        payload["sidecars"] = {
            name: {
                "file": self._contract["tables"][name]["file"],
                "row_count": state.row_count,
                "columns": state.columns,
            }
            for name, state in self._tables.items()
            if name != "trajectory"
        }
        payload["raw_manifest"] = list(self._raw_manifest)
        payload["source_files"] = {
            entry["label"]: entry["relative_path"] for entry in self._raw_manifest
        }
        metadata_path = self.stage / "metadata.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            validate_trial(
                self.stage,
                dataset_root=self.dataset_root,
                project_root=self.project_root,
                verify_hashes=True,
                allow_staging_name=True,
            )
        except Exception:
            self.abort()
            raise
        if self.destination.exists():
            raise FileExistsError(self.destination)
        os.rename(self.stage, self.destination)
        self._published = True
        return self.destination

    def abort(self) -> None:
        for state in self._tables.values():
            if not state.handle.closed:
                state.handle.close()
        self._closed = True
        if not self._published and self.stage.exists():
            shutil.rmtree(self.stage)

    def __enter__(self) -> "TrialPackageWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None or not self._published:
            self.abort()

