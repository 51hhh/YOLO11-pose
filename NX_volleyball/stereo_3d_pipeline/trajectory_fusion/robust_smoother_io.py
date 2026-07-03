"""CSV output helpers for robust smoother results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def write_output(path: str | Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    preferred = [
        "frame_id",
        "timestamp",
        "track_id",
        "x",
        "y",
        "z",
        "smooth_x",
        "smooth_y",
        "smooth_z",
        "smooth_vx",
        "smooth_vy",
        "smooth_vz",
        "smooth_sigma_z",
        "z_mono",
        "z_stereo",
        "depth_method",
        "confidence",
    ]
    extras = [key for key in rows[0].keys() if key not in preferred]
    fieldnames = preferred + extras
    with Path(path).open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
