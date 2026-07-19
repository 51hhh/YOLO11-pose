#!/usr/bin/env python3
"""
排球轨迹 CSV 回放节点 — 终端交互 + ROS2/RViz 可视化

默认接入当前落点方案:
  bbox_center + d0 反投影 -> Student-t EKF -> RK4 落点
并发布:
  /ball/realtime
  /ball/actual_path
  /ball/filtered_pos
  /ball/filtered_path
  /ball/predicted_path
  /ball/landing
  /ball/ground_plane
  /ball/landing_marker

用法:
  # 新数据集 + 当前 EKF 落点
  python3 replay_trajectory.py \\
      --csv .../p1_dy_regression_20260710_042552/traj.csv \\
      --camera-height 0.50

  # 指定 track
  python3 replay_trajectory.py --csv traj.csv --camera-height 0.50
  python3 replay_trajectory.py --csv traj.csv --max-range-m 10
  # 默认连续拼接全部 track；仅调试时才 --track-id / --auto-track
  python3 replay_trajectory.py --csv traj.csv --track-id 3 --camera-height 0.50

  # 旧滤波器兼容
  python3 replay_trajectory.py --csv traj.csv --filter imm --ground-height 0.50
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker


# ---------------------------------------------------------------------------
# Paths / optional landing pipeline import
# ---------------------------------------------------------------------------

_THIS_DIR = FsPath(__file__).resolve().parent
_NX_ROOT = _THIS_DIR.parent
_PIPELINE_ROOT = _NX_ROOT / "stereo_3d_pipeline"
_DEFAULT_LANDING_CFG = (
    _PIPELINE_ROOT / "trajectory_fusion" / "configs" / "landing_pipeline_bbox_ekf.json"
)

if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

try:
    from trajectory_fusion.landing_pipeline import LandingPipeline
    from trajectory_fusion.landing_pipeline.config import load_pipeline_config
    from trajectory_fusion.landing_pipeline.physics import (
        DRAG_K,
        G,
        as_unit,
        height_above_ground,
        rollout_landing,
    )

    HAS_LANDING_PIPELINE = True
except Exception as exc:  # pragma: no cover - runtime environment dependent
    HAS_LANDING_PIPELINE = False
    _LANDING_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(row: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    val = row.get(key, None)
    if val is None:
        return default
    if isinstance(val, str) and val.strip() == "":
        return default
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _safe_int(row: Dict[str, Any], key: str, default: int = -1) -> int:
    val = _safe_float(row, key, float(default))
    if val is None:
        return default
    return int(val)


def cam_to_rviz(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Camera(x right, y down, z forward) -> RViz/vision_world(x right, y forward, z up)."""
    return float(x), float(z), float(-y)


def sample_rollout_path(
    p0: Sequence[float],
    v0: Sequence[float],
    *,
    cd: float,
    g_hat: Sequence[float],
    ground_h: float,
    t_land: float,
    n_points: int = 24,
    dt: float = 0.008,
) -> List[np.ndarray]:
    """Sample ballistic path points from now until landing."""
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    g = as_unit(g_hat)
    t_land = max(float(t_land), dt)
    n_points = max(int(n_points), 2)
    targets = [t_land * i / (n_points - 1) for i in range(n_points)]

    def acc(vv: np.ndarray) -> np.ndarray:
        return G * g - DRAG_K * cd * np.linalg.norm(vv) * vv

    out: List[np.ndarray] = []
    t = 0.0
    ti = 0
    out.append(p.copy())
    ti = 1
    h_prev = height_above_ground(p, g, ground_h)

    while t < t_land + dt and ti < len(targets):
        step = min(dt, targets[ti] - t, t_land - t + 1e-9)
        if step <= 1e-9:
            break
        k1v = acc(v)
        k1p = v
        k2v = acc(v + 0.5 * step * k1v)
        k2p = v + 0.5 * step * k1v
        k3v = acc(v + 0.5 * step * k2v)
        k3p = v + 0.5 * step * k2v
        k4v = acc(v + step * k3v)
        k4p = v + step * k3v
        p = p + (step / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        v = v + (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t += step
        h = height_above_ground(p, g, ground_h)
        if h <= 0.0 and h_prev > 0.0:
            frac = h_prev / max(h_prev - h, 1e-9)
            p = p + (frac - 1.0) * step * v
            out.append(p.copy())
            break
        h_prev = h
        while ti < len(targets) and t + 1e-9 >= targets[ti]:
            out.append(p.copy())
            ti += 1
    if not out:
        out = [np.asarray(p0, dtype=float).copy()]
    return out


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    has_detection: bool
    obs_x: float
    obs_y: float
    obs_z: float
    track_id: int = -1
    row: Optional[Dict[str, str]] = None


def _row_has_detection(row: Dict[str, str]) -> bool:
    if "has_detection" in row:
        return int(float(row["has_detection"] or 0)) == 1
    if "confidence" in row:
        conf = _safe_float(row, "confidence", 0.0) or 0.0
        return conf > 0.0
    return True


def _row_xyz(row: Dict[str, str]) -> Tuple[float, float, float]:
    if "obs_x" in row:
        x = _safe_float(row, "obs_x", 0.0) or 0.0
        y = _safe_float(row, "obs_y", 0.0) or 0.0
        z = _safe_float(row, "obs_z", 0.0) or 0.0
    else:
        x = _safe_float(row, "x", 0.0) or 0.0
        y = _safe_float(row, "y", 0.0) or 0.0
        z = _safe_float(row, "z", 0.0) or 0.0
    return float(x), float(y), float(z)


def _frame_from_row(row: Dict[str, str], has_track: bool) -> Frame:
    x, y, z = _row_xyz(row)
    return Frame(
        frame_id=_safe_int(row, "frame_id", -1),
        timestamp=float(_safe_float(row, "timestamp", 0.0) or 0.0),
        has_detection=_row_has_detection(row),
        obs_x=x,
        obs_y=y,
        obs_z=z,
        track_id=_safe_int(row, "track_id", -1) if has_track else -1,
        row=row,
    )


def _row_in_range(row: Dict[str, str], max_range_m: float) -> bool:
    """Keep detection if camera-depth z is within max_range_m.

    max_range_m <= 0 disables filtering.
    """
    if max_range_m is None or max_range_m <= 0:
        return True
    _, _, z = _row_xyz(row)
    # invalid/unknown depth: drop when filter enabled (avoid far ghosts with bad z)
    if not math.isfinite(z) or z <= 0.01:
        return False
    return float(z) <= float(max_range_m)


def load_csv(
    path: str,
    track_id: Optional[int] = None,
    auto_track: bool = False,
    continuous: bool = True,
    stitch_gap_s: float = 0.25,
    stitch_jump_m: float = 2.5,
    max_range_m: float = 0.0,
) -> List[Frame]:
    """Load trajectory CSV.

    Default for multi-track dual-yolo logs:
      continuous timeline, stitching detections across track_id changes.
    Optional:
      --track-id N   only one track
      --auto-track   pick one flying track (legacy)
      --max-range-m  drop detections with camera z beyond this (0=off)
      continuous=False with no track filter keeps raw row order of all detections
    """
    raw_rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(dict(row))
    if not raw_rows:
        return []

    has_track = "track_id" in raw_rows[0]
    n_raw_det = sum(1 for r in raw_rows if _row_has_detection(r))
    raw_rows = [r for r in raw_rows if (not _row_has_detection(r)) or _row_in_range(r, max_range_m)]
    # keep only detections after range filter for playback
    raw_rows = [r for r in raw_rows if _row_has_detection(r)]
    n_kept = len(raw_rows)
    if max_range_m and max_range_m > 0:
        print(
            f"  距离过滤: max_range_m={float(max_range_m):.2f} m  "
            f"detections {n_raw_det} -> {n_kept} (剔除 {n_raw_det - n_kept})"
        )
    if not raw_rows:
        return []

    # 1) Explicit single track
    if track_id is not None and has_track:
        frames = [
            _frame_from_row(r, has_track)
            for r in raw_rows
            if _safe_int(r, "track_id", -1) == int(track_id) and _row_has_detection(r)
        ]
        frames.sort(key=lambda fr: (fr.timestamp, fr.frame_id, fr.track_id))
        print(f"  单 track 模式: track_id={track_id}, frames={len(frames)}")
        return frames

    # 2) Legacy auto single flying track
    if auto_track and has_track:
        selected = _auto_select_track(raw_rows)
        if selected is not None:
            print(f"  自动单 track 模式: track_id={selected}")
            frames = [
                _frame_from_row(r, has_track)
                for r in raw_rows
                if _safe_int(r, "track_id", -1) == int(selected) and _row_has_detection(r)
            ]
            frames.sort(key=lambda fr: (fr.timestamp, fr.frame_id, fr.track_id))
            print(f"  frames={len(frames)}")
            return frames

    # 3) Continuous multi-track: stitch by time
    det_rows = [r for r in raw_rows if _row_has_detection(r)]
    if not det_rows:
        return []

    # Group by timestamp (and frame_id if present) so simultaneous tracks compete.
    buckets: Dict[Tuple[int, float], List[Dict[str, str]]] = {}
    for r in det_rows:
        fid = _safe_int(r, "frame_id", -1)
        t = float(_safe_float(r, "timestamp", 0.0) or 0.0)
        buckets.setdefault((fid, t), []).append(r)

    keys = sorted(buckets.keys(), key=lambda k: (k[1], k[0]))
    frames: List[Frame] = []
    prev_pos: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    prev_tid: Optional[int] = None
    n_switch = 0

    for key in keys:
        cands = buckets[key]
        chosen = None
        if len(cands) == 1 or prev_pos is None:
            # Prefer larger motion / nearer-camera later by simple score if first
            if len(cands) == 1:
                chosen = cands[0]
            else:
                # bootstrap: prefer smaller z (nearer) with higher confidence
                def boot_score(r):
                    x, y, z = _row_xyz(r)
                    conf = _safe_float(r, "confidence", 0.0) or 0.0
                    return conf * 2.0 - 0.05 * z
                chosen = max(cands, key=boot_score)
        else:
            best_s = -1e18
            for r in cands:
                x, y, z = _row_xyz(r)
                p = np.array([x, y, z], dtype=float)
                dt = 0.0 if prev_t is None else max(float(key[1] - prev_t), 1e-3)
                dist = float(np.linalg.norm(p - prev_pos))
                tid = _safe_int(r, "track_id", -1)
                conf = _safe_float(r, "confidence", 0.0) or 0.0
                # Soft preference: same track, continuity, approaching camera.
                same = 1.0 if (prev_tid is not None and tid == prev_tid) else 0.0
                approach = max(0.0, float(prev_pos[2] - z))
                # Reject insane jumps unless no better candidate later.
                jump_pen = 0.0
                if dist > stitch_jump_m and same < 0.5:
                    jump_pen = 10.0 + dist
                s = 5.0 * same + 2.0 * conf + 0.8 * approach - 1.5 * dist - jump_pen
                # mild preference to continue shortly after prev_t
                if dt > stitch_gap_s and same < 0.5:
                    s -= 2.0 * (dt - stitch_gap_s)
                if s > best_s:
                    best_s = s
                    chosen = r
        if chosen is None:
            continue
        fr = _frame_from_row(chosen, has_track)
        if prev_tid is not None and fr.track_id != prev_tid:
            n_switch += 1
        frames.append(fr)
        prev_pos = np.array([fr.obs_x, fr.obs_y, fr.obs_z], dtype=float)
        prev_t = fr.timestamp
        prev_tid = fr.track_id

    tids = sorted({fr.track_id for fr in frames})
    print(
        f"  连续时间线模式: frames={len(frames)} tracks_used={tids} "
        f"track_switches={n_switch} (gap<={stitch_gap_s}s jump<={stitch_jump_m}m soft)"
    )
    return frames


def _auto_select_track(rows: List[Dict[str, str]]) -> Optional[int]:
    """Pick the track that looks most like a flying ball (legacy)."""
    by: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for row in rows:
        if not _row_has_detection(row):
            continue
        tid = _safe_int(row, "track_id", -1)
        t = _safe_float(row, "timestamp")
        z = _safe_float(row, "z")
        y = _safe_float(row, "y")
        if t is None or z is None or y is None:
            continue
        by.setdefault(tid, []).append((t, z, y, _safe_float(row, "x", 0.0) or 0.0))

    best_tid = None
    best_score = -1.0
    for tid, pts in by.items():
        if len(pts) < 20:
            continue
        pts = sorted(pts, key=lambda p: p[0])
        ts = np.array([p[0] for p in pts])
        zs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        dur = float(ts[-1] - ts[0])
        if dur < 0.35:
            continue
        z0 = float(np.median(zs[: max(3, len(zs) // 10)]))
        z1 = float(np.median(zs[-max(3, len(zs) // 10) :]))
        dz = z0 - z1
        y_span = float(np.max(ys) - np.min(ys))
        score = max(dz, 0.0) * 2.0 + y_span + 0.15 * len(pts) + 0.5 * dur
        if dz < 1.0:
            score *= 0.25
        if score > best_score:
            best_score = score
            best_tid = tid
    if best_tid is None and by:
        best_tid = max(by.items(), key=lambda kv: len(kv[1]))[0]
    return best_tid


class ReplayNode(Node):
    def __init__(
        self,
        frames: List[Frame],
        speed: float,
        loop: bool,
        *,
        mode: str = "landing_ekf",
        filter_name: Optional[str] = None,
        ground_height: float = 0.50,
        camera_height: float = 0.50,
        config_path: Optional[str] = None,
        landing_config: Optional[str] = None,
        enable_residual: bool = False,
        g_hat: Optional[Sequence[float]] = None,
        ground_h: Optional[float] = None,
    ):
        super().__init__("trajectory_replay")
        self.frames = frames
        self.speed = speed
        self.loop = loop
        self.idx = 0
        self.paused = False
        self.last_publish_time = 0.0

        self.mode = mode
        self.filter_name = filter_name
        self.camera_height = float(camera_height)
        # legacy Y-down ground plane for old filter + RViz cube height
        self.ground_height = float(ground_height if ground_height is not None else camera_height)

        self.filt = None
        self.landing_pipe = None
        self.gravity_vec = np.array([0.0, 9.81, 0.0], dtype=float)
        self.g_hat = np.array(g_hat if g_hat is not None else [0.0, 1.0, 0.0], dtype=float)
        self.ground_h = float(ground_h if ground_h is not None else -self.camera_height)
        self.cd = 0.10

        self.filtered_positions: List[np.ndarray] = []
        self.actual_positions: List[np.ndarray] = []
        self.landing_point: Optional[np.ndarray] = None
        self.time_to_land: Optional[float] = None
        self._last_filter_ts: Optional[float] = None
        self._init_obs_buffer: List[Tuple[float, np.ndarray]] = []
        self._filter_initialized = False
        self._outlier_gate = 2.0
        self._last_source = "none"
        self._last_track_id: Optional[int] = None
        self.track_switch_reset = True

        if self.mode == "landing_ekf":
            self._init_landing_pipeline(landing_config, enable_residual)
        elif filter_name:
            self._init_filter(filter_name, config_path)

        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pub_realtime = self.create_publisher(PointStamped, "/ball/realtime", best_effort)
        self.pub_path = self.create_publisher(Path, "/ball/actual_path", best_effort)
        self.pub_filtered_pos = self.create_publisher(PointStamped, "/ball/filtered_pos", best_effort)
        self.pub_filtered_path = self.create_publisher(Path, "/ball/filtered_path", best_effort)
        self.pub_predicted_path = self.create_publisher(Path, "/ball/predicted_path", best_effort)
        self.pub_landing = self.create_publisher(PointStamped, "/ball/landing", reliable)
        self.pub_ground_plane = self.create_publisher(Marker, "/ball/ground_plane", reliable)
        self.pub_landing_marker = self.create_publisher(Marker, "/ball/landing_marker", reliable)
        self.pub_ball_marker = self.create_publisher(Marker, "/ball/ball_marker", best_effort)

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_landing_pipeline(self, config_path: Optional[str], enable_residual: bool) -> None:
        if not HAS_LANDING_PIPELINE:
            raise RuntimeError(
                f"无法导入 landing_pipeline: {_LANDING_IMPORT_ERROR}\n"
                f"请确认路径: {_PIPELINE_ROOT}"
            )
        cfg_path = FsPath(config_path) if config_path else _DEFAULT_LANDING_CFG
        cfg = load_pipeline_config(
            cfg_path if cfg_path.exists() else None,
            enable_residual=enable_residual,
            use_runtime_d0=True,
        )
        # Dataset camera height = 50cm: plane offset for height_above_ground.
        # height(p=0) = -ground_h ≈ camera_height.
        cfg.ground_h = float(self.ground_h)
        if hasattr(self, "g_hat") and self.g_hat is not None:
            # keep config g_hat unless user overrides via CLI later
            pass
        # Prefer fitted g_hat from config JSON if present; else Y-down.
        self.g_hat = np.asarray(cfg.g_hat, dtype=float).reshape(3)
        self.g_hat = self.g_hat / max(np.linalg.norm(self.g_hat), 1e-12)
        cfg.g_hat = self.g_hat.tolist()
        cfg.residual.enabled = bool(enable_residual)
        self.cd = float(getattr(cfg.ekf, "cd", 0.10))
        self.landing_pipe = LandingPipeline(cfg)
        self.landing_pipe.ekf.g_hat = self.g_hat.copy()
        self.landing_pipe.ekf.ground_h = float(self.ground_h)
        print("  模式: landing_ekf (bbox + d0 + Student-t EKF + RK4)")
        print(f"  config: {cfg_path if cfg_path.exists() else 'defaults'}")
        print(f"  d0={cfg.d0:.3f}  Cd={self.cd:.3f}  residual={cfg.residual.enabled}")
        print(f"  g_hat={self.g_hat.tolist()}")
        print(f"  camera_height={self.camera_height:.3f}m  ground_h={self.ground_h:.3f}")
        print(f"  RViz ground z≈{-self.ground_height:.3f} (cam y-down)")

    def _init_filter(self, filter_name: str, config_path: Optional[str] = None) -> None:
        ta_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trajectory_analysis"))
        if ta_dir not in sys.path:
            sys.path.insert(0, ta_dir)
        from filters import create_filter

        if config_path is None:
            config_path = os.path.join(ta_dir, "config.yaml")
        params: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            physics = config.get("physics", {})
            gv = physics.get("gravity_vec", [0.0, 9.81, 0.0])
            self.gravity_vec = np.array(gv, dtype=float)
            filters_cfg = config.get("filters", {})
            lookup_name = filter_name[7:] if filter_name.startswith("robust_") else filter_name
            params = dict(filters_cfg.get(lookup_name, {}))
        self.filt = create_filter(filter_name, **params)
        self.filt.reset()
        print(f"  滤波器: {filter_name} (参数: {params})")
        print(f"  落地平面: y = {self.ground_height:.3f}m")

    def _reset_predictor(self, clear_paths: bool = True) -> None:
        if self.landing_pipe is not None:
            self.landing_pipe.reset()
        if self.filt is not None:
            self.filt.reset()
        if clear_paths:
            self.filtered_positions.clear()
            self.actual_positions.clear()
        self.landing_point = None
        self.time_to_land = None
        self._last_filter_ts = None
        self._init_obs_buffer.clear()
        self._filter_initialized = False
        self._last_source = "none"
        self._last_track_id = None
        self._last_track_id: Optional[int] = None
        self.track_switch_reset = True

    def destroy_node(self):
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass
        super().destroy_node()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def get_key(self) -> Optional[str]:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                ch3 = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                if ch2 == "[":
                    if ch3 == "C":
                        return "RIGHT"
                    if ch3 == "D":
                        return "LEFT"
                return None
            return ch
        return None

    def handle_input(self) -> bool:
        key = self.get_key()
        if key is None:
            return True
        if key == "q":
            return False
        if key == " ":
            self.paused = not self.paused
        elif key in ("RIGHT", "d"):
            self.seek(1)
        elif key in ("LEFT", "a"):
            self.seek(-1)
        elif key in (".", "w"):
            self.seek(10)
        elif key in (",", "s"):
            self.seek(-10)
        elif key in ("+", "="):
            self.speed = min(self.speed * 2.0, 32.0)
        elif key == "-":
            self.speed = max(self.speed / 2.0, 0.05)
        elif key == "r":
            self.idx = 0
            self._reset_predictor()
            self.paused = False
            print("\n  ↺ 重头播放")
        return True

    def seek(self, delta: int) -> None:
        """Step frames without always replaying from the start.

        self.idx means "next frame to process" (same as the play loop).
        - Forward: continue from current state, only process new frames.
        - Backward: rebuild causal EKF/filter state from frame 0.
        """
        if not self.frames or delta == 0:
            return
        n = len(self.frames)
        if delta > 0:
            end = min(self.idx + int(delta), n)
            while self.idx < end:
                self.publish_current()
                self.idx += 1
        else:
            target_next = max(0, self.idx + int(delta))
            self._reset_predictor()
            self.idx = 0
            while self.idx < target_next:
                self.publish_current()
                self.idx += 1
        self.paused = True
        self.last_publish_time = time.monotonic()

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------
    def _stamp_now(self) -> Time:
        return self.get_clock().now().to_msg()

    def _publish_point(self, pub, xyz_cam: Sequence[float], stamp: Time) -> None:
        rx, ry, rz = cam_to_rviz(xyz_cam[0], xyz_cam[1], xyz_cam[2])
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "vision_world"
        msg.point.x = rx
        msg.point.y = ry
        msg.point.z = rz
        pub.publish(msg)

    def _publish_path(self, pub, pts_cam: Sequence[np.ndarray], stamp: Time) -> None:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "vision_world"
        for p in pts_cam:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = "vision_world"
            rx, ry, rz = cam_to_rviz(p[0], p[1], p[2])
            pose.pose.position.x = rx
            pose.pose.position.y = ry
            pose.pose.position.z = rz
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        pub.publish(path)

    def _publish_ground_plane(self, stamp: Time) -> None:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "vision_world"
        marker.ns = "ground"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 6.0
        marker.pose.position.z = -self.ground_height  # cam y-down -> rviz z-up
        marker.pose.orientation.w = 1.0
        marker.scale.x = 8.0
        marker.scale.y = 14.0
        marker.scale.z = 0.01
        marker.color.r = 0.2
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 0.25
        self.pub_ground_plane.publish(marker)

    def _publish_sphere(self, pub, xyz_cam: Sequence[float], stamp: Time, *, ns: str, mid: int,
                        rgba: Sequence[float], scale: float) -> None:
        rx, ry, rz = cam_to_rviz(xyz_cam[0], xyz_cam[1], xyz_cam[2])
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "vision_world"
        marker.ns = ns
        marker.id = mid
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = rx
        marker.pose.position.y = ry
        marker.pose.position.z = rz
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        pub.publish(marker)

    # ------------------------------------------------------------------
    # Core per-frame
    # ------------------------------------------------------------------
    def publish_current(self, advance: bool = True) -> None:
        if not self.frames:
            return
        frame = self.frames[min(self.idx, len(self.frames) - 1)]
        stamp = self._stamp_now()

        obs_cam = None
        if frame.has_detection and np.linalg.norm([frame.obs_x, frame.obs_y, frame.obs_z]) > 0.01:
            obs_cam = np.array([frame.obs_x, frame.obs_y, frame.obs_z], dtype=float)

        # Landing pipeline may produce a better observation from disparity.
        if self.landing_pipe is not None and frame.row is not None:
            self._process_landing_ekf(frame, stamp)
        elif self.filt is not None and frame.has_detection and obs_cam is not None:
            # legacy filter path uses CSV xyz
            self._publish_point(self.pub_realtime, obs_cam, stamp)
            self.actual_positions.append(obs_cam.copy())
            self._publish_path(self.pub_path, self.actual_positions, stamp)
            self._publish_sphere(
                self.pub_ball_marker, obs_cam, stamp, ns="ball", mid=0,
                rgba=(1.0, 0.85, 0.1, 0.95), scale=0.21,
            )
            self._process_filter(frame, stamp)
        else:
            if obs_cam is not None:
                self._publish_point(self.pub_realtime, obs_cam, stamp)
                self.actual_positions.append(obs_cam.copy())
                self._publish_path(self.pub_path, self.actual_positions, stamp)
                self._publish_sphere(
                    self.pub_ball_marker, obs_cam, stamp, ns="ball", mid=0,
                    rgba=(1.0, 0.85, 0.1, 0.95), scale=0.21,
                )
            self._publish_ground_plane(stamp)

    def _process_landing_ekf(self, frame: Frame, stamp: Time) -> None:
        assert self.landing_pipe is not None
        assert frame.row is not None

        # Reset filter state on large time gap or track switch, but keep actual path
        # so RViz shows continuous recorded trajectory across IDs.
        need_reset = False
        reason = ""
        if self._last_filter_ts is not None:
            gap = frame.timestamp - self._last_filter_ts
            if gap > 0.35:
                need_reset = True
                reason = f"gap={gap:.3f}s"
        if (
            self.track_switch_reset
            and self._last_track_id is not None
            and frame.track_id >= 0
            and frame.track_id != self._last_track_id
        ):
            need_reset = True
            reason = (reason + "+" if reason else "") + f"track {self._last_track_id}->{frame.track_id}"
        if need_reset:
            self.landing_pipe.reset()
            self.filtered_positions.clear()  # filtered path restarts per segment
            self.landing_point = None
            self.time_to_land = None
            # keep self.actual_positions continuous
            # small stdout note only when paused/stepping would spam; keep quiet.

        out = self.landing_pipe.update_row(frame.row)
        self._last_filter_ts = frame.timestamp
        self._last_track_id = frame.track_id
        if out is None:
            # still try raw xyz for trail if available
            if frame.has_detection and np.linalg.norm([frame.obs_x, frame.obs_y, frame.obs_z]) > 0.01:
                obs = np.array([frame.obs_x, frame.obs_y, frame.obs_z], dtype=float)
                self._publish_point(self.pub_realtime, obs, stamp)
                self.actual_positions.append(obs)
                self._publish_path(self.pub_path, self.actual_positions, stamp)
            self._publish_ground_plane(stamp)
            return

        pos = np.asarray(out.position, dtype=float)
        vel = np.asarray(out.velocity, dtype=float)
        land = np.asarray(out.landing, dtype=float)
        self._last_source = out.source
        self.time_to_land = float(out.time_to_land)
        self.landing_point = land.copy()
        self.filtered_positions.append(pos.copy())
        self.actual_positions.append(pos.copy())

        # realtime / actual / filtered
        self._publish_point(self.pub_realtime, pos, stamp)
        self._publish_point(self.pub_filtered_pos, pos, stamp)
        self._publish_path(self.pub_path, self.actual_positions, stamp)
        self._publish_path(self.pub_filtered_path, self.filtered_positions, stamp)
        self._publish_sphere(
            self.pub_ball_marker, pos, stamp, ns="ball", mid=0,
            rgba=(1.0, 0.85, 0.1, 0.95), scale=0.21,
        )

        # landing + predicted path
        self._publish_point(self.pub_landing, land, stamp)
        self._publish_sphere(
            self.pub_landing_marker, land, stamp, ns="landing", mid=1,
            rgba=(1.0, 0.15, 0.1, 0.95), scale=0.28,
        )
        path_pts = sample_rollout_path(
            pos,
            vel,
            cd=self.cd,
            g_hat=self.g_hat,
            ground_h=self.ground_h,
            t_land=max(float(out.time_to_land), 0.05),
            n_points=28,
            dt=0.008,
        )
        # ensure last point is landing
        if path_pts:
            path_pts[-1] = land.copy()
        self._publish_path(self.pub_predicted_path, path_pts, stamp)
        self._publish_ground_plane(stamp)

    def _process_filter(self, frame: Frame, stamp: Time) -> None:
        """Legacy trajectory_analysis filter path."""
        obs = np.array([frame.obs_x, frame.obs_y, frame.obs_z], dtype=float)
        if np.linalg.norm(obs) < 0.01:
            return

        if self._last_filter_ts is not None:
            gap = frame.timestamp - self._last_filter_ts
            if gap > 0.3:
                self.filt.reset()
                self._init_obs_buffer.clear()
                self._filter_initialized = False
                self.filtered_positions.clear()
                self.landing_point = None
                self._last_filter_ts = None

        if not self._filter_initialized:
            self._init_obs_buffer.append((frame.timestamp, obs.copy()))
            if len(self._init_obs_buffer) >= 3:
                self._physics_init()
            else:
                self.filt.update(obs[0], obs[1], obs[2])
                self._last_filter_ts = frame.timestamp
                return
            if not self._filter_initialized:
                return

        dt = frame.timestamp - self._last_filter_ts
        if dt > 0:
            self.filt.predict(dt)

        predicted_state = self.filt.get_state()
        pred_pos = np.array([predicted_state.x, predicted_state.y, predicted_state.z], dtype=float)
        if np.linalg.norm(obs - pred_pos) > self._outlier_gate:
            self._last_filter_ts = frame.timestamp
            return

        state = self.filt.update(obs[0], obs[1], obs[2])
        self._last_filter_ts = frame.timestamp
        pos = np.array([state.x, state.y, state.z], dtype=float)
        vel = np.array([state.vx, state.vy, state.vz], dtype=float)
        self.filtered_positions.append(pos)
        self._publish_point(self.pub_filtered_pos, pos, stamp)
        self._publish_path(self.pub_filtered_path, self.filtered_positions, stamp)
        self._predict_landing_legacy(pos, vel, stamp)

    def _physics_init(self) -> None:
        buf = self._init_obs_buffer
        t0 = buf[0][0]
        p0 = buf[0][1]
        g = self.gravity_vec
        rows = []
        b = []
        for t, obs_i in buf[1:]:
            dt_i = t - t0
            if dt_i < 1e-4:
                continue
            rows.append(dt_i * np.eye(3))
            b_i = obs_i - p0 - 0.5 * g * dt_i ** 2
            b.append(b_i)
        if not rows:
            return
        A = np.vstack(rows)
        bb = np.concatenate(b)
        v0, *_ = np.linalg.lstsq(A, bb, rcond=None)
        # seed filter
        self.filt.update(p0[0], p0[1], p0[2])
        if hasattr(self.filt, "x") and hasattr(self.filt.x, "__len__") and len(self.filt.x) >= 6:
            self.filt.x[3:6] = v0
        elif hasattr(self.filt, "_models"):
            for m_x in self.filt._models:
                if hasattr(m_x, "__len__") and len(m_x) >= 6:
                    m_x[3:6] = v0
        self._filter_initialized = True
        self._last_filter_ts = buf[-1][0]
        for _, obs in buf:
            self.filtered_positions.append(obs.copy())

    def _predict_landing_legacy(self, pos, vel, stamp: Time) -> None:
        gy = self.gravity_vec[1]
        y0 = pos[1]
        vy = vel[1]
        a = 0.5 * gy
        b = vy
        c = y0 - self.ground_height
        disc = b * b - 4 * a * c
        if disc < 0 or abs(a) < 1e-10:
            return
        sqrt_d = math.sqrt(disc)
        cands = [t for t in [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)] if t > 0.01]
        if not cands:
            return
        t_land = min(cands)
        if t_land > 3.0:
            return
        landing = np.array(
            [pos[0] + vel[0] * t_land, self.ground_height, pos[2] + vel[2] * t_land],
            dtype=float,
        )
        self.landing_point = landing
        self.time_to_land = t_land
        self._publish_point(self.pub_landing, landing, stamp)
        self._publish_sphere(
            self.pub_landing_marker, landing, stamp, ns="landing", mid=1,
            rgba=(1.0, 0.15, 0.1, 0.95), scale=0.28,
        )
        path_pts = []
        for i in range(21):
            t = t_land * i / 20.0
            path_pts.append(
                np.array(
                    [
                        pos[0] + vel[0] * t,
                        pos[1] + vel[1] * t + 0.5 * gy * t * t,
                        pos[2] + vel[2] * t,
                    ],
                    dtype=float,
                )
            )
        self._publish_path(self.pub_predicted_path, path_pts, stamp)
        self._publish_ground_plane(stamp)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        total = len(self.frames)
        print(f"\n{'=' * 60}")
        print(f"  轨迹回放: {total} 帧, 速度 {self.speed:.1f}x, mode={self.mode}")
        print(f"  相机高度: {self.camera_height:.2f} m  |  ground_h={self.ground_h:.3f}")
        print(f"{'=' * 60}")
        print("  Space=暂停  ←/→=±1帧  ,/.=±10帧  +/-=加减速  r=重头  q=退出")
        print(f"{'=' * 60}\n")
        print("  RViz Fixed Frame: vision_world")
        print("  Topics:")
        print("    /ball/realtime          当前球点")
        print("    /ball/actual_path       实际轨迹")
        print("    /ball/filtered_path     滤波轨迹")
        print("    /ball/predicted_path    预测飞行曲线")
        print("    /ball/landing           预测落点")
        print("    /ball/landing_marker    落点球体")
        print("    /ball/ball_marker       当前球体")
        print("    /ball/ground_plane      地面")
        print()

        try:
            while rclpy.ok():
                if not self.handle_input():
                    break
                if not self.paused:
                    now = time.monotonic()
                    if self.idx < total:
                        if self.idx > 0:
                            dt = self.frames[self.idx].timestamp - self.frames[self.idx - 1].timestamp
                            if not math.isfinite(dt) or dt <= 0:
                                dt = 1.0 / 60.0
                        else:
                            dt = 1.0 / 60.0
                        wait = dt / max(self.speed, 1e-6)
                        if now - self.last_publish_time >= wait:
                            self.publish_current()
                            self.last_publish_time = now
                            self.idx += 1
                            f = self.frames[min(self.idx, total - 1)]
                            d = "●" if f.has_detection else "○"
                            landing_str = ""
                            if self.landing_point is not None:
                                lp = self.landing_point
                                ttl = self.time_to_land if self.time_to_land is not None else float("nan")
                                landing_str = (
                                    f" tid={f.track_id} L({lp[0]:.2f},{lp[2]:.2f}) ttl={ttl:.2f}s src={self._last_source}"
                                )
                            sys.stdout.write(
                                f"\r  ▶ [{self.idx:>5}/{total}] {self.speed:.1f}x "
                                f"{d} tid={f.track_id} ({f.obs_x:.2f},{f.obs_y:.2f},{f.obs_z:.2f}){landing_str}   "
                            )
                            sys.stdout.flush()
                    else:
                        if self.loop:
                            self.idx = 0
                            self._reset_predictor()
                            print("\n  ↻ 循环重播...")
                        else:
                            print("\n  ■ 播放完毕 (Space 继续停在末帧, r 重头, q 退出)")
                            self.paused = True
                else:
                    show_i = min(max(self.idx - 1, 0), total - 1) if self.idx > 0 else 0
                    f = self.frames[show_i]
                    d = "●" if f.has_detection else "○"
                    landing_str = ""
                    if self.landing_point is not None:
                        lp = self.landing_point
                        ttl = self.time_to_land if self.time_to_land is not None else float("nan")
                        landing_str = (
                            f" L({lp[0]:.2f},{lp[2]:.2f}) ttl={ttl:.2f}s src={self._last_source}"
                        )
                    sys.stdout.write(
                        f"\r  ⏸ [{show_i+1:>5}/{total}] {self.speed:.1f}x "
                        f"{d} ({f.obs_x:.2f},{f.obs_y:.2f},{f.obs_z:.2f}){landing_str}   "
                    )
                    sys.stdout.flush()
                    time.sleep(0.05)
                time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n  退出回放")


def _parse_vec3(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return [float(p) for p in parts]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="排球轨迹 CSV 回放 + 落点预测 RViz 可视化")
    parser.add_argument("--csv", required=True, help="traj.csv 路径")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--track-id", type=int, default=None, help="只播放指定 track_id（调试用）")
    parser.add_argument(
        "--auto-track",
        action="store_true",
        help="旧行为：自动只选一个飞行 track（默认关闭）",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="按时间连续拼接多 track（默认开启）",
    )
    parser.add_argument(
        "--no-continuous",
        action="store_true",
        help="关闭连续拼接（需配合 --auto-track 或 --track-id）",
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=0.0,
        help="手动指定时才剔除：相机深度 z 超过该值的检测点(m)。默认 0=不剔除",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="landing_ekf",
        choices=["landing_ekf", "legacy_filter", "raw"],
        help="landing_ekf=当前方案; legacy_filter=旧 filters; raw=只播原始点",
    )
    parser.add_argument("--filter", type=str, default=None, help="旧滤波器名（隐含 mode=legacy_filter）")
    parser.add_argument("--config", type=str, default=None, help="旧滤波器 config.yaml")
    parser.add_argument(
        "--landing-config",
        type=str,
        default=str(_DEFAULT_LANDING_CFG),
        help="landing_pipeline JSON 配置",
    )
    parser.add_argument("--residual", action="store_true", help="启用 TinyGRU 落点残差（可选）")

    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.50,
        help="相机离地高度 (m)。当前数据集默认 0.50m",
    )
    parser.add_argument(
        "--ground-height",
        type=float,
        default=None,
        help="旧 Y-down 地面坐标；默认等于 camera-height",
    )
    parser.add_argument(
        "--ground-h",
        type=float,
        default=None,
        help="landing_pipeline 的 ground_h；默认 -camera-height",
    )
    parser.add_argument(
        "--g-hat",
        type=_parse_vec3,
        default=None,
        help="重力方向单位向量, 如 0,1,0 或拟合值",
    )
    args = parser.parse_args(argv)

    if args.filter and args.mode == "landing_ekf":
        args.mode = "legacy_filter"

    camera_height = float(args.camera_height)
    ground_height = float(args.ground_height) if args.ground_height is not None else camera_height
    ground_h = float(args.ground_h) if args.ground_h is not None else -camera_height

    continuous = not bool(getattr(args, "no_continuous", False))
    if args.track_id is not None:
        continuous = False
    frames = load_csv(
        args.csv,
        track_id=args.track_id,
        auto_track=bool(args.auto_track),
        continuous=continuous and not bool(args.auto_track),
        max_range_m=float(args.max_range_m),
    )
    if not frames:
        print(f"错误: CSV 为空或读取失败: {args.csv}")
        return 1
    print(f"  加载 {len(frames)} 帧 from {args.csv}")
    if float(args.max_range_m) > 0:
        print(f"  max_range_m={float(args.max_range_m):.2f} m（已剔除更远检测）")
    else:
        print("  max_range_m=off（不按距离剔除）")

    if not rclpy.ok():
        rclpy.init()

    node = ReplayNode(
        frames,
        args.speed,
        args.loop,
        mode=args.mode,
        filter_name=args.filter,
        ground_height=ground_height,
        camera_height=camera_height,
        config_path=args.config,
        landing_config=args.landing_config,
        enable_residual=args.residual,
        g_hat=args.g_hat,
        ground_h=ground_h,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
