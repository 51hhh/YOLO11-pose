"""可追溯、可校验的分层配置加载器。

合并规则是契约的一部分：字典递归合并，标量覆盖，列表整体替换。
每个最终叶子值都保留来源文件，便于 explain 和数据集指纹审计。
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema
import yaml


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


def _validate_schema(root: Path, payload: Mapping[str, Any]) -> None:
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
    if payload["recording"]["schema_version"] != "2.1.0":
        errors.append("新采集只允许 recording.schema_version=2.1.0")
    mode = payload["runtime"]["mode"]
    overflow = payload["runtime"]["queue"]["overflow_policy"]
    if mode in {"capture", "replay"} and overflow != "block":
        errors.append(f"{mode} 模式不得丢数据，overflow_policy 必须为 block")
    if mode == "runtime" and overflow not in {"drop_oldest", "block"}:
        errors.append("runtime 模式的 overflow_policy 必须为 drop_oldest 或 block")
    if payload["recording"]["enabled"] and not payload["recording"]["dataset_root"]:
        errors.append("录制启用时 recording.dataset_root 不得为空")

    manifest_path = root / "models" / "manifest.yaml"
    manifest = _load_yaml(manifest_path)
    model_id = payload["detector"]["model_id"]
    models = manifest.get("models", {})
    if model_id not in models:
        errors.append(f"detector.model_id {model_id!r} 未在 models/manifest.yaml 定义")
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

