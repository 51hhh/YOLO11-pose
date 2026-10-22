"""把旧 OpenCV 双目标定转换为采集契约 2.1.0。

实时管线保存的是校正后灰度图，因此数据集标定描述校正相机模型：
畸变为零、内参来自 P1/P2、rig 原点位于左目校正光心。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .config import ConfigError, ResolvedConfig, find_project_root
from .dataset import DatasetError, validate_calibration


@dataclass(frozen=True, slots=True)
class OpenCvStereoCalibration:
    image_width: int
    image_height: int
    baseline_m: float
    rms_error_px: float
    projection_left: tuple[float, ...]
    projection_right: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(text: str, key: str) -> float:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^#\s]+)", text)
    if not match:
        raise DatasetError(f"OpenCV 标定缺少字段: {key}")
    try:
        return float(match.group(1))
    except ValueError as error:
        raise DatasetError(f"OpenCV 标定字段不是数字: {key}") from error


def _matrix(text: str, key: str, rows: int, cols: int) -> tuple[float, ...]:
    pattern = (
        rf"(?ms)^{re.escape(key)}:\s*!!opencv-matrix\s*$"
        rf".*?^\s*rows:\s*{rows}\s*$"
        rf".*?^\s*cols:\s*{cols}\s*$"
        rf".*?^\s*data:\s*\[(.*?)\]"
    )
    match = re.search(pattern, text)
    if not match:
        raise DatasetError(f"OpenCV 标定缺少 {rows}x{cols} 矩阵: {key}")
    try:
        values = tuple(float(value.strip()) for value in match.group(1).split(","))
    except ValueError as error:
        raise DatasetError(f"OpenCV 标定矩阵包含无效数字: {key}") from error
    if len(values) != rows * cols or not all(math.isfinite(value) for value in values):
        raise DatasetError(f"OpenCV 标定矩阵尺寸或数值无效: {key}")
    return values


def load_opencv_stereo_calibration(path: str | Path) -> OpenCvStereoCalibration:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise DatasetError(f"OpenCV 标定文件不存在: {source}") from error
    width = int(_scalar(text, "image_width"))
    height = int(_scalar(text, "image_height"))
    baseline_mm = _scalar(text, "baseline")
    rms = _scalar(text, "rms_error")
    if width <= 0 or height <= 0 or baseline_mm <= 0 or rms < 0:
        raise DatasetError("OpenCV 标定的尺寸、基线或 RMS 无效")
    return OpenCvStereoCalibration(
        image_width=width,
        image_height=height,
        baseline_m=baseline_mm / 1000.0,
        rms_error_px=rms,
        projection_left=_matrix(text, "projection_left", 3, 4),
        projection_right=_matrix(text, "projection_right", 3, 4),
    )


def _intrinsic_from_projection(values: tuple[float, ...]) -> list[list[float]]:
    return [
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]],
    ]


def _transform_with_x(x_m: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x_m],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_capture_calibration(
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    root = find_project_root(project_root)
    calibration_config = config.data["calibration"]
    if calibration_config["format"] != "opencv_yaml":
        raise ConfigError("当前转换器只接受 calibration.format=opencv_yaml")
    source = root / calibration_config["file"]
    stereo = load_opencv_stereo_calibration(source)
    expected_size = list(config.data["rectification"]["output_resolution"])
    if expected_size != [stereo.image_width, stereo.image_height]:
        raise DatasetError(
            f"校正输出尺寸 {expected_size} 与标定 "
            f"{[stereo.image_width, stereo.image_height]} 不一致"
        )
    cameras = config.data["cameras"]
    clock_domain = cameras["sync"]["clock_domain"]
    if created_at_utc is None:
        created_at_utc = calibration_config["created_at_utc"]
    common = {
        "image_size_px": [stereo.image_width, stereo.image_height],
        "distortion_model": "rectified_pinhole",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rms_reprojection_error_px": stereo.rms_error_px,
        "clock_domain": clock_domain,
        "extrinsic_covariance": None,
    }
    return {
        "schema_version": "2.1.0",
        "calibration_type": "vehicle_mounted_stereo",
        "calibration_id": calibration_config["id"],
        "created_at_utc": created_at_utc,
        "body_frame_id": calibration_config["rig_frame"],
        "camera_roles": {
            "left": cameras["left"]["id"],
            "right": cameras["right"]["id"],
        },
        "estimated_sync_error_s": (
            None
            if cameras["sync"]["estimated_error_ns"] is None
            else cameras["sync"]["estimated_error_ns"] * 1.0e-9
        ),
        "cameras": [
            {
                **common,
                "camera_id": cameras["left"]["id"],
                "camera_matrix": _intrinsic_from_projection(stereo.projection_left),
                "transform_body_from_camera": _transform_with_x(0.0),
            },
            {
                **common,
                "camera_id": cameras["right"]["id"],
                "camera_matrix": _intrinsic_from_projection(stereo.projection_right),
                "transform_body_from_camera": _transform_with_x(stereo.baseline_m),
            },
        ],
        "baseline_m": stereo.baseline_m,
        "notes": (
            "校正后 rectified_gray8 相机模型；源 OpenCV 标定 "
            f"{source.relative_to(root)} sha256={_sha256(source)}。"
            "右目设备时间戳在采集时映射到左目设备时钟。"
        ),
    }


def ensure_dataset_calibration(
    dataset_root: str | Path,
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> Path:
    root = Path(dataset_root).resolve()
    destination = root / "calibrations" / f"{config.data['calibration']['id']}.json"
    payload = build_capture_calibration(config, project_root=project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        # created_at_utc 也是配置中的固定资产属性；任何变化都意味着
        # 标定 ID 被复用。
        if existing != payload:
            raise DatasetError(f"同名标定已存在但内容不同，必须使用新 calibration_id: {destination}")
    else:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    validate_calibration(destination, project_root=project_root)
    return destination

