#!/usr/bin/env python3
"""Config rendering helpers for NX algorithm matrix tests."""

from __future__ import annotations

import re
from pathlib import Path

from nx_algorithm_cases import MODE_KEYS, Case


def set_yaml_bool(text: str, key: str, value: bool) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)(true|false)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{'true' if value else 'false'}{match.group(3)}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing bool key: {key}")
    return new


def set_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]*)(.*)$", re.M)

    def repl(match: re.Match[str]) -> str:
        suffix = match.group(3)
        if suffix.startswith("#"):
            suffix = " " + suffix
        return f"{match.group(1)}{value}{suffix}"

    new, count = pattern.subn(repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"missing scalar key: {key}")
    return new


def set_depth_mode(text: str, key: str, value: bool) -> str:
    if key not in MODE_KEYS:
        raise RuntimeError(f"unknown depth mode: {key}")
    return set_yaml_bool(text, key, value)


def disable_all_depth_modes(text: str) -> str:
    for key in sorted(MODE_KEYS):
        text = set_depth_mode(text, key, False)
    return text


def set_neural_enabled(text: str, value: bool) -> str:
    pattern = re.compile(
        r"(^neural_feature_matching:\n(?:^[ \t].*\n)*?^[ \t]*enabled:\s*)(true|false)(.*)$",
        re.M,
    )
    replacement = rf"\g<1>{'true' if value else 'false'}\3"
    text, _ = pattern.subn(replacement, text, count=1)
    return text


def render_neural_block(case: Case, neural_model_dir: Path) -> str:
    use_lightglue = str(case.neural_backend == "superpoint_lightglue").lower()
    extractor_engine_path = ""
    if case.neural_engine:
        extractor_engine_path = str(neural_model_dir / case.neural_engine)
    return f"""neural_feature_matching:
  enabled: true
  backend: "{case.neural_backend}"
  extractor_engine_path: "{extractor_engine_path}"
  matcher_engine_path: ""
  fused_engine_path: ""
  roi_size: {case.roi_size}
  top_k: {case.top_k}
  descriptor_dim: {case.descriptor_dim}
  min_matches: {case.neural_min_matches}
  max_y_error_px: {case.neural_max_y_error_px}
  max_disp_delta_px: {case.neural_max_disp_delta_px}
  final_disp_gate_px: {case.neural_final_disp_gate_px}
  min_score: {case.neural_min_score}
  use_lightglue: {use_lightglue}
"""


def upsert_neural_block(text: str, block: str) -> str:
    pattern = re.compile(r"^neural_feature_matching:\n(?:^[ \t].*\n?)*", re.M)
    new, count = pattern.subn(block.rstrip() + "\n", text, count=1)
    if count:
        return new
    return text.rstrip() + "\n\n" + block


def prepare_config(
    base: str,
    case: Case,
    out_dir: Path,
    config_dir: Path,
    neural_model_dir: Path,
) -> Path:
    text = base
    text = re.sub(r"(\nros2:\n\s*)enable:\s*true", r"\1enable: false", text, count=1)
    text = disable_all_depth_modes(text)
    text = set_yaml_bool(text, "subpixel_enabled", False)
    text = set_yaml_bool(text, "fallback_epipolar_search", False)
    text = set_neural_enabled(text, False)
    text = re.sub(
        r'output_path:\s*"dual_yolo_observation_data\.csv"',
        f'output_path: "{out_dir / (case.name + ".csv")}"',
        text,
        count=1,
    )
    for mode, value in case.modes.items():
        text = set_depth_mode(text, mode, value)
    if case.subpixel_enabled is not None:
        text = set_yaml_bool(text, "subpixel_enabled", case.subpixel_enabled)
    for key, value in case.yaml_scalars.items():
        text = set_yaml_scalar(text, key, value)
    if case.neural_backend:
        text = upsert_neural_block(text, render_neural_block(case, neural_model_dir))
    cfg = config_dir / f"{case.name}.yaml"
    cfg.write_text(text)
    return cfg
