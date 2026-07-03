"""Output helpers for offline volleyball keypoint probes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import cv2

from offline_volleyball_probe_roi import MatchResult


def write_keypoint_summary(
    out_dir: Path,
    summary: Dict[str, object],
    rows: Sequence[Dict[str, object]],
) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not rows:
        (out_dir / "summary.csv").write_text("", encoding="utf-8")
        return
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(image_paths: Sequence[Path], out_path: Path) -> None:
    sheets = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            sheets.append(img)
    if not sheets:
        return

    target_w = sheets[0].shape[1]
    normalized = []
    for img in sheets:
        if img.shape[1] != target_w:
            scale = target_w / img.shape[1]
            img = cv2.resize(img, (target_w, max(1, round(img.shape[0] * scale))))
        normalized.append(img)
    cv2.imwrite(str(out_path), cv2.vconcat(normalized))


def write_match_contact_sheets(
    out_dir: Path,
    results: Sequence[MatchResult],
) -> None:
    match_paths = [out_dir / f"{res.name}_matches.png" for res in results]
    zoom_paths = [out_dir / f"{res.name}_matches_zoom.png" for res in results]
    _write_contact_sheet(match_paths, out_dir / "contact_sheet.png")
    _write_contact_sheet(zoom_paths, out_dir / "zoom_contact_sheet.png")
