#!/usr/bin/env python3
"""CSV and metadata I/O for trajectory fusion datasets."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_metadata_value(value: str) -> Any:
    value = value.strip()
    if value == "" or value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_metadata(path: str | Path | None) -> Dict[str, Any]:
    """Read optional weak-label metadata.

    PyYAML is used when present. A minimal key/value parser is kept as a
    dependency-free fallback for the flat metadata files recommended in the
    wiki.
    """

    if path is None:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    text = meta_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return dict(loaded or {}) if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    metadata: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_metadata_value(value)
    return metadata


def find_metadata_for_csv(path: str | Path) -> Path | None:
    """Find a sidecar metadata file for a trajectory CSV."""

    csv_path = Path(path)
    candidates = (
        csv_path.with_suffix(".metadata.yaml"),
        csv_path.with_suffix(".metadata.yml"),
        csv_path.parent / "metadata.yaml",
        csv_path.parent / "metadata.yml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def derive_frame_summary_path(path: str | Path) -> Path:
    """Derive the recorder sidecar path matching the C++ recorder."""

    csv_path = Path(path)
    if csv_path.name.endswith(".csv"):
        return csv_path.with_name(csv_path.name[:-4] + ".frames.csv")
    return Path(str(csv_path) + ".frames.csv")


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    """Read CSV rows while tolerating accidental NUL bytes in log files."""

    data = Path(path).read_bytes().replace(b"\x00", b"")
    text = data.decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(text)))


def iter_extended_rows(path: str | Path) -> Iterable[Dict[str, str]]:
    """Yield rows from a future schema.md-compatible CSV file."""

    yield from read_csv_rows(path)
