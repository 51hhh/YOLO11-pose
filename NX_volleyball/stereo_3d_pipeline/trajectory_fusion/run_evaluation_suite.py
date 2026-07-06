#!/usr/bin/env python3
"""Run the standard trajectory-fusion evaluation suite for recorded CSV clips."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .analyze_candidate_consistency import (
        analyze_candidate_clips,
        write_method_csv as write_candidate_method_csv,
        write_pairwise_csv as write_candidate_pairwise_csv,
    )
    from .check_dataset import analyze_dataset
    from .dataset import find_metadata_for_csv, load_legacy_sequences, read_metadata
    from .evaluate_calibrated_smoother import apply_calibrated_smoother
    from .evaluate_fusion import _read as read_eval_rows
    from .evaluate_fusion import build_report
    from .manifest import DatasetClip, is_manifest_path, load_manifest
    from .robust_smoother import SmootherConfig, smooth_sequence, smooth_sequence_rts, write_output
except ImportError:  # pragma: no cover - direct script execution
    from analyze_candidate_consistency import (
        analyze_candidate_clips,
        write_method_csv as write_candidate_method_csv,
        write_pairwise_csv as write_candidate_pairwise_csv,
    )
    from check_dataset import analyze_dataset
    from dataset import find_metadata_for_csv, load_legacy_sequences, read_metadata
    from evaluate_calibrated_smoother import apply_calibrated_smoother
    from evaluate_fusion import _read as read_eval_rows
    from evaluate_fusion import build_report
    from manifest import DatasetClip, is_manifest_path, load_manifest
    from robust_smoother import SmootherConfig, smooth_sequence, smooth_sequence_rts, write_output


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "clip"


def resolve_clips(inputs: Sequence[str], metadata: str | None = None) -> List[DatasetClip]:
    if len(inputs) == 1 and is_manifest_path(inputs[0]):
        if metadata:
            raise SystemExit("--metadata cannot be used with a manifest")
        return load_manifest(inputs[0])
    if metadata and len(inputs) != 1:
        raise SystemExit("--metadata can only be used with one CSV")
    return [
        DatasetClip(
            csv=Path(item),
            metadata=Path(metadata) if metadata else None,
            split="eval",
            name=Path(item).stem,
        )
        for item in inputs
    ]


def _metadata_path_for_clip(clip: DatasetClip) -> Path | None:
    return clip.metadata or find_metadata_for_csv(clip.csv)


def _write_json(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _evaluate_csv(csv_path: Path, metadata_path: Path | None, output_json: Path) -> Dict[str, Any]:
    rows = read_eval_rows(csv_path)
    metadata = read_metadata(metadata_path)
    report = build_report(rows, metadata)
    _write_json(output_json, report)
    return report


def _run_robust_smoother(
    clip: DatasetClip,
    output_csv: Path,
    *,
    metadata_path: Path | None,
    gravity_y: float,
    use_online_position: bool,
    use_static_known_z: bool,
    rts: bool = False,
) -> Dict[str, Any]:
    cfg = SmootherConfig(
        gravity_y=gravity_y,
        use_online_position=use_online_position,
        use_static_known_z=use_static_known_z,
    )
    all_rows: List[Dict[str, float]] = []
    metrics: List[Dict[str, float]] = []
    smoother = smooth_sequence_rts if rts else smooth_sequence
    for sequence in load_legacy_sequences(clip.csv, metadata_path=metadata_path):
        rows, seq_metrics = smoother(sequence, cfg)
        all_rows.extend(rows)
        metrics.append(seq_metrics)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_output(output_csv, all_rows)
    return {"rows": len(all_rows), "sequences": metrics}


def run_suite(
    inputs: Sequence[str],
    output_dir: str | Path,
    *,
    metadata: str | None = None,
    checkpoint: str | Path | None = None,
    calibration: str | Path | None = None,
    device: str = "cpu",
    gravity_y: float = 9.81,
    use_online_position: bool = False,
    use_static_known_z: bool = False,
    include_rts_smoother: bool = True,
    include_candidate_consistency: bool = True,
    candidate_reference: str = "auto",
) -> Dict[str, Any]:
    clips = resolve_clips(inputs, metadata)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    used_names: Dict[str, int] = {}
    suite_report: Dict[str, Any] = {
        "output_dir": str(root),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "calibration": str(calibration) if calibration else None,
        "config": {
            "gravity_y": gravity_y,
            "use_online_position": use_online_position,
            "use_static_known_z": use_static_known_z,
            "include_rts_smoother": include_rts_smoother,
            "include_candidate_consistency": include_candidate_consistency,
            "candidate_reference": candidate_reference,
        },
        "clips": [],
    }

    for index, clip in enumerate(clips):
        base_name = _safe_stem(clip.name or clip.csv.stem)
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        name = base_name if count == 0 else f"{base_name}_{count + 1}"
        clip_dir = root / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = _metadata_path_for_clip(clip)

        check_json = clip_dir / "check_dataset.json"
        check_report = analyze_dataset(clip.csv, metadata_path=metadata_path)
        _write_json(check_json, check_report)

        raw_eval_json = clip_dir / "raw_eval.json"
        raw_report = _evaluate_csv(clip.csv, metadata_path, raw_eval_json)

        candidate_consistency_json = None
        candidate_consistency_csv = None
        candidate_pairwise_csv = None
        if include_candidate_consistency:
            candidate_report = analyze_candidate_clips(
                [
                    DatasetClip(
                        csv=clip.csv,
                        metadata=metadata_path,
                        split=clip.split,
                        name=name,
                    )
                ],
                reference=candidate_reference,
            )
            candidate_consistency_json = clip_dir / "candidate_consistency.json"
            candidate_consistency_csv = clip_dir / "candidate_consistency.csv"
            candidate_pairwise_csv = clip_dir / "candidate_pairwise.csv"
            _write_json(candidate_consistency_json, candidate_report)
            write_candidate_method_csv(candidate_consistency_csv, candidate_report)
            write_candidate_pairwise_csv(candidate_pairwise_csv, candidate_report)

        robust_csv = clip_dir / "robust_smooth.csv"
        robust_summary = _run_robust_smoother(
            clip,
            robust_csv,
            metadata_path=metadata_path,
            gravity_y=gravity_y,
            use_online_position=use_online_position,
            use_static_known_z=use_static_known_z,
        )
        robust_eval_json = clip_dir / "robust_smooth_eval.json"
        robust_report = _evaluate_csv(robust_csv, metadata_path, robust_eval_json)

        clip_report: Dict[str, Any] = {
            "index": index,
            "name": name,
            "csv": str(clip.csv),
            "metadata": str(metadata_path) if metadata_path else None,
            "split": clip.split,
            "check_dataset_json": str(check_json),
            "raw_eval_json": str(raw_eval_json),
            "robust_smooth_csv": str(robust_csv),
            "robust_smooth_eval_json": str(robust_eval_json),
            "check_rows": check_report.get("rows", 0),
            "raw_track_count": raw_report.get("track_count", 0),
            "robust_rows": robust_summary["rows"],
            "robust_track_count": robust_report.get("track_count", 0),
        }
        if include_rts_smoother:
            robust_rts_csv = clip_dir / "robust_rts_smooth.csv"
            robust_rts_summary = _run_robust_smoother(
                clip,
                robust_rts_csv,
                metadata_path=metadata_path,
                gravity_y=gravity_y,
                use_online_position=use_online_position,
                use_static_known_z=use_static_known_z,
                rts=True,
            )
            robust_rts_eval_json = clip_dir / "robust_rts_smooth_eval.json"
            robust_rts_report = _evaluate_csv(robust_rts_csv, metadata_path, robust_rts_eval_json)
            clip_report.update(
                {
                    "robust_rts_smooth_csv": str(robust_rts_csv),
                    "robust_rts_smooth_eval_json": str(robust_rts_eval_json),
                    "robust_rts_rows": robust_rts_summary["rows"],
                    "robust_rts_track_count": robust_rts_report.get("track_count", 0),
                }
            )
        if include_candidate_consistency:
            clip_report.update(
                {
                    "candidate_consistency_json": str(candidate_consistency_json),
                    "candidate_consistency_csv": str(candidate_consistency_csv),
                    "candidate_pairwise_csv": str(candidate_pairwise_csv),
                }
            )

        if calibration:
            calibrated_csv = clip_dir / "calibrated_smoother.csv"
            calibrated_json = clip_dir / "calibrated_smoother_apply.json"
            calibrated_report = apply_calibrated_smoother(
                input_csv=clip.csv,
                calibration_path=calibration,
                output_csv=calibrated_csv,
                metadata_path=metadata_path,
                smoother_cfg=SmootherConfig(
                    gravity_y=gravity_y,
                    use_online_position=use_online_position,
                    use_static_known_z=use_static_known_z,
                ),
            )
            _write_json(calibrated_json, calibrated_report)
            calibrated_eval_json = clip_dir / "calibrated_smoother_eval.json"
            _evaluate_csv(calibrated_csv, metadata_path, calibrated_eval_json)
            clip_report.update(
                {
                    "calibrated_smoother_csv": str(calibrated_csv),
                    "calibrated_smoother_apply_json": str(calibrated_json),
                    "calibrated_smoother_eval_json": str(calibrated_eval_json),
                }
            )

        if checkpoint:
            try:
                from .evaluate_reliability_checkpoint import apply_checkpoint
                from .evaluate_reliability_smoother import apply_reliability_smoother
            except ImportError:  # pragma: no cover - direct script execution
                from evaluate_reliability_checkpoint import apply_checkpoint
                from evaluate_reliability_smoother import apply_reliability_smoother

            direct_csv = clip_dir / "reliability_direct.csv"
            direct_json = clip_dir / "reliability_direct_apply.json"
            direct_report = apply_checkpoint(
                input_csv=clip.csv,
                checkpoint_path=checkpoint,
                output_csv=direct_csv,
                metadata_path=metadata_path,
                device=device,
            )
            _write_json(direct_json, direct_report)
            direct_eval_json = clip_dir / "reliability_direct_eval.json"
            _evaluate_csv(direct_csv, metadata_path, direct_eval_json)

            smoother_csv = clip_dir / "reliability_smoother.csv"
            smoother_json = clip_dir / "reliability_smoother_apply.json"
            smoother_report = apply_reliability_smoother(
                input_csv=clip.csv,
                checkpoint_path=checkpoint,
                output_csv=smoother_csv,
                metadata_path=metadata_path,
                device=device,
                smoother_cfg=SmootherConfig(
                    gravity_y=gravity_y,
                    use_online_position=use_online_position,
                    use_static_known_z=use_static_known_z,
                ),
            )
            _write_json(smoother_json, smoother_report)
            smoother_eval_json = clip_dir / "reliability_smoother_eval.json"
            _evaluate_csv(smoother_csv, metadata_path, smoother_eval_json)

            clip_report.update(
                {
                    "reliability_direct_csv": str(direct_csv),
                    "reliability_direct_apply_json": str(direct_json),
                    "reliability_direct_eval_json": str(direct_eval_json),
                    "reliability_smoother_csv": str(smoother_csv),
                    "reliability_smoother_apply_json": str(smoother_json),
                    "reliability_smoother_eval_json": str(smoother_eval_json),
                }
            )

        suite_report["clips"].append(clip_report)

    _write_json(root / "suite_summary.json", suite_report)
    return suite_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV file(s), or one dataset manifest YAML/JSON")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--metadata", help="Optional metadata YAML for a single CSV")
    parser.add_argument("--checkpoint", help="Optional ReliabilityNet checkpoint")
    parser.add_argument("--calibration", help="Optional per-method calibration JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gravity-y", type=float, default=9.81)
    parser.add_argument("--use-online-position", action="store_true")
    parser.add_argument(
        "--use-static-known-z",
        dest="use_static_known_z",
        action="store_true",
        help="Use static known_z metadata as a smoother update. Off by default to avoid label leakage in evaluation.",
    )
    parser.add_argument(
        "--no-static-known-z",
        dest="use_static_known_z",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--candidate-reference", default="auto")
    parser.add_argument("--skip-rts-smoother", action="store_true")
    parser.add_argument("--skip-candidate-consistency", action="store_true")
    parser.set_defaults(use_static_known_z=False)
    args = parser.parse_args()

    report = run_suite(
        inputs=args.inputs,
        output_dir=args.output_dir,
        metadata=args.metadata,
        checkpoint=args.checkpoint,
        calibration=args.calibration,
        device=args.device,
        gravity_y=args.gravity_y,
        use_online_position=args.use_online_position,
        use_static_known_z=args.use_static_known_z,
        include_rts_smoother=not args.skip_rts_smoother,
        include_candidate_consistency=not args.skip_candidate_consistency,
        candidate_reference=args.candidate_reference,
    )
    print(f"wrote suite for {len(report['clips'])} clip(s) to {report['output_dir']}")
    for clip in report["clips"]:
        print(
            "clip={name} rows={rows} robust_rows={robust_rows} rts_rows={rts_rows} dir={directory}".format(
                name=clip["name"],
                rows=clip["check_rows"],
                robust_rows=clip["robust_rows"],
                rts_rows=clip.get("robust_rts_rows", 0),
                directory=Path(clip["robust_smooth_csv"]).parent,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
