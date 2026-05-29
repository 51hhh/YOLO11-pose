"""CSV data loading and continuous segment splitting."""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Frame:
    """Single detection frame with all 16 CSV columns."""
    frame_id: int
    timestamp: float
    has_detection: bool
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    det_confidence: float
    z_mono: float
    z_stereo: float
    disparity: float
    stereo_conf: float
    depth_method: str
    obs_x: float
    obs_y: float
    obs_z: float


@dataclass
class Segment:
    """Continuous detection frame sequence."""
    frames: List[Frame] = field(default_factory=list)
    source_file: str = ""

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def frame_ids(self) -> np.ndarray:
        return np.array([f.frame_id for f in self.frames])

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.timestamp for f in self.frames])

    @property
    def obs_xyz(self) -> np.ndarray:
        return np.array([[f.obs_x, f.obs_y, f.obs_z] for f in self.frames])

    @property
    def obs_z_depth(self) -> np.ndarray:
        return np.array([f.obs_z for f in self.frames])


def parse_frame(row: dict) -> Optional[Frame]:
    """Parse a CSV row dict into a Frame. Returns None if parsing fails."""
    try:
        has_det = row['has_detection'].strip().lower() in ('1', 'true', 'yes')
        return Frame(
            frame_id=int(row['frame_id']),
            timestamp=float(row['timestamp']),
            has_detection=has_det,
            bbox_cx=float(row['bbox_cx']),
            bbox_cy=float(row['bbox_cy']),
            bbox_w=float(row['bbox_w']),
            bbox_h=float(row['bbox_h']),
            det_confidence=float(row['det_confidence']),
            z_mono=float(row['z_mono']),
            z_stereo=float(row['z_stereo']),
            disparity=float(row['disparity']),
            stereo_conf=float(row['stereo_conf']),
            depth_method=row['depth_method'].strip(),
            obs_x=float(row['obs_x']),
            obs_y=float(row['obs_y']),
            obs_z=float(row['obs_z']),
        )
    except (KeyError, ValueError):
        return None


def load_csv(filepath: str) -> List[Frame]:
    """Load all detection frames from a single CSV file."""
    frames = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = parse_frame(row)
            if frame is not None and frame.has_detection and frame.obs_z > 0.01:
                frames.append(frame)
    return frames


def segment_frames(frames: List[Frame], gap_threshold: int = 5,
                   min_length: int = 10) -> List[List[Frame]]:
    """Split frames into continuous segments.
    
    A new segment starts when frame_id gap > gap_threshold.
    Only segments with >= min_length frames are kept.
    """
    if not frames:
        return []

    segments = []
    current = [frames[0]]

    for i in range(1, len(frames)):
        gap = frames[i].frame_id - frames[i - 1].frame_id
        if gap > gap_threshold:
            if len(current) >= min_length:
                segments.append(current)
            current = [frames[i]]
        else:
            current.append(frames[i])

    if len(current) >= min_length:
        segments.append(current)

    return segments


def load_dataset(data_dir: str, prefix: str = "") -> List[Segment]:
    """Load all CSV files matching prefix from data_dir.
    
    Args:
        data_dir: Path to directory containing CSV files.
        prefix: Filter files containing this group prefix (e.g., "0", "1", "2").
                Matches pattern "_{prefix}_" in filename.
                Empty string means load all CSV files.
    
    Returns:
        List of Segment objects from all matching files.
    """
    all_segments = []
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if prefix:
        csv_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.csv') and f"_{prefix}_" in f
        ])
    else:
        csv_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.csv')
        ])

    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        frames = load_csv(filepath)
        segments = segment_frames(frames)
        for seg_frames in segments:
            seg = Segment(frames=seg_frames, source_file=csv_file)
            all_segments.append(seg)

    return all_segments


def load_all_datasets(data_dir: str) -> dict:
    """Load datasets grouped by prefix (0, 1, 2).
    
    Returns dict with keys '0', '1', '2' and 'all'.
    """
    result = {}
    all_segments = []
    
    for prefix in ['0', '1', '2']:
        segs = load_dataset(data_dir, prefix)
        result[prefix] = segs
        all_segments.extend(segs)
    
    # Also load any files not starting with 0/1/2
    remaining = load_dataset(data_dir, "")
    result['all'] = remaining if remaining else all_segments
    
    return result
