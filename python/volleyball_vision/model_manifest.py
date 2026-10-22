"""模型清单解析与不可变指纹。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, ResolvedConfig, find_project_root


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(
    config: ResolvedConfig,
    *,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    root = find_project_root(project_root)
    payload = yaml.safe_load((root / "models" / "manifest.yaml").read_text(encoding="utf-8"))
    model_id = config.data["detector"]["model_id"]
    try:
        entry = dict(payload["models"][model_id])
    except (KeyError, TypeError) as error:
        raise ConfigError(f"模型清单缺少 {model_id!r}") from error
    artifact = (root / entry["artifact"]).resolve()
    if root not in artifact.parents:
        raise ConfigError("模型 artifact 不得逃逸项目目录")
    return entry, artifact

