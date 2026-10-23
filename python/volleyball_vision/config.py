"""可追溯、可校验的分层配置加载器。

合并规则是契约的一部分：字典递归合并，标量覆盖，列表整体替换。
每个最终叶子值都保留来源文件，便于 explain 和数据集指纹审计。
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema
import yaml

from .triangulation_config import triangulation_semantic_errors


class ConfigError(ValueError):
    """配置文件、合并或语义不满足契约。"""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError(
                f"{loader.name}:{key_node.start_mark.line + 1}: 配置键必须是字符串"
            )
        if key in mapping:
            raise ConfigError(
                f"{loader.name}:{key_node.start_mark.line + 1}: 重复配置键 {key!r}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    profile: str
    data: Mapping[str, Any]
    provenance: Mapping[str, str]
    layers: tuple[Path, ...]
    fingerprint_sha256: str

    def value(self, dotted_key: str) -> Any:
        current: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ConfigError(f"配置键不存在: {dotted_key}")
            current = current[part]
        return current

    def source(self, dotted_key: str) -> str:
        if dotted_key in self.provenance:
            return self.provenance[dotted_key]
        prefix = dotted_key + "."
        sources = sorted(
            {source for key, source in self.provenance.items() if key.startswith(prefix)}
        )
        if not sources:
            raise ConfigError(f"配置键不存在: {dotted_key}")
        return ", ".join(sources)


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "schema" / "runtime.schema.json").is_file():
            return candidate
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "configs" / "schema" / "runtime.schema.json").is_file():
        return package_root
    raise ConfigError("无法定位项目根目录")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            loader = _UniqueKeyLoader(handle)
            loader.name = str(path)
            try:
                payload = loader.get_single_data()
            finally:
                loader.dispose()
    except yaml.YAMLError as error:
        raise ConfigError(f"YAML 解析失败 {path}: {error}") from error
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"配置顶层必须是对象: {path}")
    return payload


def _record_leaves(
    value: Any,
    source: str,
    provenance: dict[str, str],
    prefix: str = "",
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else key
            _record_leaves(nested, source, provenance, path)
    else:
        provenance[prefix] = source


def _drop_provenance_subtree(provenance: dict[str, str], prefix: str) -> None:
    for key in tuple(provenance):
        if key == prefix or key.startswith(prefix + "."):
            del provenance[key]


def _merge(
    target: dict[str, Any],
    overlay: Mapping[str, Any],
    source: str,
    provenance: dict[str, str],
    prefix: str = "",
) -> None:
    for key, value in overlay.items():
        path = f"{prefix}.{key}" if prefix else key
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, Mapping)
        ):
            _merge(target[key], value, source, provenance, path)
            continue
        target[key] = copy.deepcopy(value)
        _drop_provenance_subtree(provenance, path)
        _record_leaves(value, source, provenance, path)


def _parse_override(expression: str) -> tuple[list[str], Any]:
    if "=" not in expression:
        raise ConfigError(f"--set 需要 key=value，实际为: {expression!r}")
    raw_key, raw_value = expression.split("=", 1)
    parts = raw_key.split(".")
    if not raw_key or any(not part for part in parts):
        raise ConfigError(f"--set 键无效: {raw_key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigError(f"--set 值无效: {raw_value!r}") from error
    return parts, value


def _apply_override(
    target: dict[str, Any],
    expression: str,
    provenance: dict[str, str],
) -> None:
    parts, value = _parse_override(expression)
    current: dict[str, Any] = target
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise ConfigError(
                f"--set 只能覆盖已有键，不存在: {'.'.join(parts)}"
            )
        current = current[part]
    leaf = parts[-1]
    if leaf not in current:
        raise ConfigError(
            f"--set 只能覆盖已有键，不存在: {'.'.join(parts)}"
        )
    current[leaf] = value
    dotted = ".".join(parts)
    _drop_provenance_subtree(provenance, dotted)
    _record_leaves(value, f"CLI --set {expression}", provenance, dotted)


def _format_schema_error(error: jsonschema.ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{path}: {error.message}"


def _non_finite_number_paths(value: Any, prefix: str = "") -> list[str]:
    """列出 YAML 可表达、但规范 JSON 配置不允许的 NaN/Infinity。"""

    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_non_finite_number_paths(nested, path))
        return paths
    if isinstance(value, list):
        paths = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_non_finite_number_paths(nested, path))
        return paths
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix or "<root>"]
    return []


def _validate_schema(root: Path, payload: Mapping[str, Any]) -> None:
    non_finite = _non_finite_number_paths(payload)
    if non_finite:
        lines = "\n".join(f"  - {path}: 数值必须是有限数" for path in non_finite)
        raise ConfigError(f"配置 schema 校验失败:\n{lines}")
    schema_path = root / "configs" / "schema" / "runtime.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        lines = "\n".join(f"  - {_format_schema_error(error)}" for error in errors)
        raise ConfigError(f"配置 schema 校验失败:\n{lines}")


def _validate_semantics(root: Path, payload: Mapping[str, Any]) -> None:
    errors: list[str] = []
    cameras = payload["cameras"]
    left = cameras["left"]
    right = cameras["right"]
    if left["id"] == right["id"]:
        errors.append("cameras.left.id 与 cameras.right.id 必须不同")
    if left["serial"] == right["serial"]:
        errors.append("cameras.left.serial 与 cameras.right.serial 必须不同")
    if left["index"] == right["index"]:
        errors.append("cameras.left.index 与 cameras.right.index 必须不同")
    if left["resolution"] != right["resolution"]:
        errors.append("当前管线要求左右相机分辨率一致")
    if left["resolution"] != payload["rectification"]["output_resolution"]:
        errors.append("相机分辨率必须与 rectification.output_resolution 一致")
    frame_period_us = 1_000_000.0 / payload["runtime"]["nominal_fps"]
    if left["exposure_us"] >= frame_period_us or right["exposure_us"] >= frame_period_us:
        errors.append("曝光时间必须小于帧周期")
    if payload["stereo"]["min_depth_m"] >= payload["stereo"]["max_depth_m"]:
        errors.append("stereo.min_depth_m 必须小于 stereo.max_depth_m")
    for error in triangulation_semantic_errors(
        payload["stereo"]["triangulation"]
    ):
        errors.append(f"stereo.triangulation: {error}")
    if payload["recording"]["schema_version"] != "2.1.0":
        errors.append("新采集只允许 recording.schema_version=2.1.0")
    mode = payload["runtime"]["mode"]
    overflow = payload["runtime"]["queue"]["overflow_policy"]
    recording = payload["recording"]
    publisher = payload["publisher"]
    detector = payload["detector"]
    sync = cameras["sync"]
    transport = cameras["transport"]
    triangulation_backend = payload["stereo"]["triangulation"]["backend"]
    if mode in {"capture", "replay"} and overflow != "block":
        errors.append(f"{mode} 模式不得丢数据，overflow_policy 必须为 block")
    if mode == "runtime" and overflow not in {"drop_oldest", "block"}:
        errors.append("runtime 模式的 overflow_policy 必须为 drop_oldest 或 block")
    if recording["enabled"] and not recording["dataset_root"]:
        errors.append("录制启用时 recording.dataset_root 不得为空")
    if publisher["enabled"]:
        if publisher["backend"] == "none":
            errors.append("publisher.enabled=true 时 backend 不得为 none")
        if not publisher["topic"]:
            errors.append("publisher.enabled=true 时 topic 不得为空")
        if not publisher["frame_id"]:
            errors.append("publisher.enabled=true 时 frame_id 不得为空")
        elif publisher["frame_id"] != payload["calibration"]["rig_frame"]:
            errors.append(
                "publisher.frame_id 必须与 calibration.rig_frame 一致；"
                "坐标系变换不能只靠改名"
            )
    elif publisher["backend"] != "none" or publisher["topic"]:
        errors.append("publisher 禁用时 backend 必须为 none 且 topic 必须为空")

    live_mode = mode in {"runtime", "capture"}
    if live_mode:
        if cameras["backend"] != "hik_mvs":
            errors.append("runtime/capture 模式只支持 cameras.backend=hik_mvs")
        if payload["platform"]["accelerator"] not in {"cuda", "cuda_dla"}:
            errors.append("runtime/capture 模式必须使用 CUDA 加速器")
        if sync["method"] != "hardware_trigger":
            errors.append("runtime/capture 模式必须使用 hardware_trigger")
        if sync["estimated_error_ns"] is None:
            errors.append("runtime/capture 模式必须提供 estimated_error_ns")
        expected_sync = {
            "trigger_source": "Line0",
            "trigger_activation": "RisingEdge",
            "frame_identity_source": "hik_frame_spec_info",
            "shutter_type": "global",
            "exposure_timestamp_source": "nx_gpio_rising_edge",
        }
        for key, expected in expected_sync.items():
            if sync[key] != expected:
                label = (
                    f"{expected} (FrameSpecInfo)"
                    if key == "frame_identity_source"
                    else expected
                )
                errors.append(f"runtime/capture 要求 cameras.sync.{key}={label}")
    else:
        replay_backends = {
            "cameras.backend": cameras["backend"],
            "rectification.backend": payload["rectification"]["backend"],
            "detector.backend": detector["backend"],
            "stereo.triangulation.backend": triangulation_backend,
            "cameras.sync.method": sync["method"],
        }
        for key, actual in replay_backends.items():
            if actual != "recorded":
                errors.append(f"replay 模式要求 {key}=recorded")

    if mode == "capture":
        if not recording["enabled"]:
            errors.append("capture 模式必须启用 recording")
        if recording["loss_policy"] != "fail_trial":
            errors.append("capture 模式必须使用 recording.loss_policy=fail_trial")
        if recording["raw_frames"] != "all":
            errors.append("capture 模式必须保存 recording.raw_frames=all")
    else:
        if recording["enabled"]:
            errors.append("非 capture 模式不得启用 recording")
        if recording["raw_frames"] != "none":
            errors.append("非 capture 模式要求 recording.raw_frames=none")

    if sync["exposure_timestamp_source"] == "nx_gpio_rising_edge":
        if sync["method"] != "hardware_trigger":
            errors.append("nx_gpio_rising_edge 只允许用于 hardware_trigger")
        if sync["trigger_activation"] != "RisingEdge":
            errors.append("nx_gpio_rising_edge 要求 trigger_activation=RisingEdge")
        if sync["shutter_type"] != "global":
            errors.append("当前曝光中心时间模型要求 shutter_type=global")
    if sync["exposure_delay_status"] == "assumed_zero_unmeasured":
        if sync["exposure_delay_ns"] != 0:
            errors.append("未测量曝光延迟时 exposure_delay_ns 必须明确按 0 假设")
        if sync["exposure_delay_uncertainty_ns"] is not None:
            errors.append("未测量曝光延迟时 exposure_delay_uncertainty_ns 必须为 null")
    elif sync["exposure_delay_status"] == "measured":
        if sync["exposure_delay_uncertainty_ns"] is None:
            errors.append("实测曝光延迟必须提供 exposure_delay_uncertainty_ns")
    if cameras["backend"] == "hik_mvs":
        if transport["interface"] not in {"usb3", "gige"}:
            errors.append("hik_mvs 必须声明 usb3 或 gige 传输")
        if transport["grab_strategy"] != "one_by_one":
            errors.append("hik_mvs 双目必须使用 one_by_one 保留每侧到达顺序")
        if left["pixel_format"] != "bayer8" or right["pixel_format"] != "bayer8":
            errors.append("当前海康适配器固定输出 BayerRG8，左右 pixel_format 必须为 bayer8")
        if left["exposure_us"] != right["exposure_us"]:
            errors.append("当前海康适配器只支持左右相同 exposure_us")
        if left["gain_db"] != right["gain_db"]:
            errors.append("当前海康适配器只支持左右相同 gain_db")
        expected_roi = {
            "x": 0,
            "y": 0,
            "width": left["resolution"][0],
            "height": left["resolution"][1],
        }
        if left["roi"] != expected_roi or right["roi"] != expected_roi:
            errors.append("当前海康适配器只支持左右相同的零偏移全幅 ROI")
        if payload["rectification"]["backend"] != "cuda":
            errors.append("当前 NX 主线只支持 rectification.backend=cuda")
        if payload["stereo"]["triangulation"]["backend"] != "cuda":
            errors.append("当前 NX 主线只支持 stereo.triangulation.backend=cuda")
        if detector["backend"] != "tensorrt":
            errors.append("当前 NX 主线只支持 detector.backend=tensorrt")
        if detector["input_resolution"][0] != detector["input_resolution"][1]:
            errors.append("当前 TensorRT 检测器要求正方形 input_resolution")
        if sync["gpio_chip"] is None or sync["gpio_line"] is None:
            errors.append("hardware_trigger 必须配置 gpio_chip/gpio_line")
        if transport["embedded_info_clear_rows"] <= 0:
            errors.append("FrameSpecInfo 开启时必须配置水印像素清除行数")
    elif transport["interface"] != "recorded" or transport["grab_strategy"] != "recorded":
        errors.append("recorded 相机必须使用 recorded transport")

    manifest_path = root / "models" / "manifest.yaml"
    manifest = _load_yaml(manifest_path)
    model_id = payload["detector"]["model_id"]
    models = manifest.get("models", {})
    if model_id not in models:
        errors.append(f"detector.model_id {model_id!r} 未在 models/manifest.yaml 定义")
    elif mode in {"runtime", "capture"}:
        entry = models[model_id]
        if not isinstance(entry, Mapping):
            errors.append(f"模型清单 {model_id!r} 必须是对象")
        else:
            expected_format = {
                "tensorrt": "tensorrt_engine",
            }.get(detector["backend"])
            checks = (
                ("artifact_format", expected_format),
                ("input_resolution", detector["input_resolution"]),
                ("input_format", detector["input_format"]),
                ("target", payload["platform"]["id"]),
                ("accelerator", payload["platform"]["accelerator"]),
            )
            for field, expected in checks:
                if entry.get(field) != expected:
                    errors.append(
                        f"模型清单 {model_id}.{field}={entry.get(field)!r} "
                        f"与运行配置 {expected!r} 不一致"
                    )
    if errors:
        raise ConfigError("配置语义校验失败:\n" + "\n".join(f"  - {e}" for e in errors))


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_profile(
    profile: str,
    *,
    project_root: str | Path | None = None,
    overrides: Iterable[str] = (),
) -> ResolvedConfig:
    root = find_project_root(project_root)
    profile_path = Path(profile)
    if not profile_path.suffix:
        profile_path = root / "configs" / "profiles" / f"{profile}.yaml"
    elif not profile_path.is_absolute():
        profile_path = (root / profile_path).resolve()
    profile_path = profile_path.resolve()
    profiles_root = (root / "configs" / "profiles").resolve()
    if profile_path.parent != profiles_root:
        raise ConfigError("配置 profile 必须位于 configs/profiles")
    profile_payload = _load_yaml(profile_path)
    allowed_profile_keys = {"schema_version", "profile", "description", "layers"}
    unknown = sorted(set(profile_payload) - allowed_profile_keys)
    if unknown:
        raise ConfigError(f"profile 含未知键: {unknown}")
    if profile_payload.get("schema_version") != 1:
        raise ConfigError("profile.schema_version 必须为 1")
    name = profile_payload.get("profile")
    if not isinstance(name, str) or not name:
        raise ConfigError("profile.profile 必须为非空字符串")
    raw_layers = profile_payload.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ConfigError("profile.layers 必须为非空列表")
    if len(set(raw_layers)) != len(raw_layers):
        raise ConfigError("profile.layers 不得重复")

    configs_root = (root / "configs").resolve()
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    paths: list[Path] = []
    for raw_path in raw_layers:
        if not isinstance(raw_path, str) or not raw_path:
            raise ConfigError("profile.layers 项必须为非空字符串")
        layer_path = (configs_root / raw_path).resolve()
        if configs_root not in layer_path.parents:
            raise ConfigError(f"配置层逃逸 configs 目录: {raw_path}")
        layer = _load_yaml(layer_path)
        _merge(merged, layer, str(layer_path.relative_to(root)), provenance)
        paths.append(layer_path)
    for expression in overrides:
        _apply_override(merged, expression, provenance)
    _validate_schema(root, merged)
    _validate_semantics(root, merged)
    fingerprint = hashlib.sha256(canonical_json(merged).encode("utf-8")).hexdigest()
    return ResolvedConfig(
        profile=name,
        data=merged,
        provenance=provenance,
        layers=tuple(paths),
        fingerprint_sha256=fingerprint,
    )


def flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result

