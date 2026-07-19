#!/usr/bin/env python3
"""Dataset manifest helpers for trajectory fusion experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class DatasetClip:
    csv: Path
    metadata: Path | None = None
    split: str = "train"
    name: str = ""


def _load_manifest_data(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return dict(data) if isinstance(data, dict) else {"clips": data}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return dict(data or {}) if isinstance(data, dict) else {"clips": data or []}
    except ImportError:
        return _parse_simple_yaml_manifest(text)


def _parse_simple_yaml_manifest(text: str) -> Dict[str, Any]:
    """Parse the flat clips list used in the wiki without requiring PyYAML."""

    clips: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    in_clips = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "clips:":
            in_clips = True
            continue
        if not in_clips:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                clips.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if stripped:
                key, value = _split_key_value(stripped)
                current[key] = value
            continue
        if current is None:
            continue
        key, value = _split_key_value(stripped)
        current[key] = value
    if current:
        clips.append(current)
    return {"clips": clips}


def _split_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"manifest line lacks ':' separator: {line}")
    key, value = line.split(":", 1)
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return key.strip(), value


def _resolve(base: Path, value: object | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def load_manifest(path: str | Path) -> List[DatasetClip]:
    """Load clips from a dataset manifest.

    Expected YAML shape:

    clips:
      - csv: traj_p0p1_001.csv
        metadata: traj_p0p1_001.metadata.yaml
        split: train
    """

    manifest_path = Path(path)
    data = _load_manifest_data(manifest_path)
    raw_clips = data.get("clips", [])
    if not isinstance(raw_clips, list):
        raise ValueError("manifest 'clips' must be a list")

    clips: List[DatasetClip] = []
    base = manifest_path.parent
    for index, item in enumerate(raw_clips):
        if not isinstance(item, dict):
            raise ValueError(f"manifest clip #{index} must be a mapping")
        csv_path = _resolve(base, item.get("csv"))
        if csv_path is None:
            raise ValueError(f"manifest clip #{index} missing csv")
        metadata_path = _resolve(base, item.get("metadata"))
        clips.append(
            DatasetClip(
                csv=csv_path,
                metadata=metadata_path,
                split=str(item.get("split") or "train"),
                name=str(item.get("name") or csv_path.stem),
            )
        )
    return clips


def is_manifest_path(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in {".yaml", ".yml", ".json"}
