#!/usr/bin/env python3
"""Reproduce rejected Step 1 fix trials for failure analysis.

This script intentionally does not change the production fusion classes. It
records why covariance-rate correction and all-track trimmed alignment were
rejected after the first real-NMEA benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.benchmark_gps_fusion import evaluate_run
from data_tools.gps_sources import (
    GPSReplay,
    load_nmea_replay,
    load_reference_trajectory,
    odometry_proxy_from_recording,
    rigid_align_2d,
    wrap_angle,
)
from pipelines.localization_ekf import LocalizationFusion


class CovarianceRateTrialFusion(LocalizationFusion):
    """Rejected trial: rate=max(3, 1.5*sqrt(trace(P_map)))."""

    def _advance_global_correction(self, dt: float) -> None:
        angle_delta = float(
            wrap_angle(self.target_map_to_odom[2] - self.map_to_odom[2])
        )
        angle_limit = self.config.max_rotation_correction_rps * dt
        angle_step = float(np.clip(angle_delta, -angle_limit, angle_limit))
        if angle_step:
            global_position = self.global_pose[:2]
            self.map_to_odom[2] = float(
                wrap_angle(self.map_to_odom[2] + angle_step)
            )
            self.map_to_odom[:2] = global_position - self._rotate(
                self.local_pose[:2], self.map_to_odom[2]
            )

        translation_delta = self.target_map_to_odom[:2] - self.map_to_odom[:2]
        distance = float(np.linalg.norm(translation_delta))
        rate = max(
            self.config.max_translation_correction_mps,
            1.5 * float(np.sqrt(np.trace(self.map_covariance))),
        )
        maximum = rate * dt
        if distance > maximum > 0:
            translation_delta *= maximum / distance
        self.map_to_odom[:2] += translation_delta


def summarize_residual(values: np.ndarray) -> dict:
    return {
        "samples": len(values),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
    }


def transform_replay(
    raw_replay: GPSReplay,
    rotation: np.ndarray,
    translation: np.ndarray,
    label: str,
) -> GPSReplay:
    measurements = []
    for measurement in raw_replay.measurements:
        if measurement.has_position:
            position = rotation @ measurement.position + translation
            measurement = replace(
                measurement,
                x=float(position[0]),
                y=float(position[1]),
            )
        measurements.append(measurement)
    return GPSReplay(
        measurements,
        metadata={
            "alignment": label,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    )


def alignment_trial(recording: Path) -> dict:
    timestamps, reference, _ = load_reference_trajectory(recording)
    raw_replay = load_nmea_replay(recording, align_to_reference=False)
    quality4 = [
        measurement
        for measurement in raw_replay.measurements
        if measurement.fix_quality == 4
        and measurement.has_position
        and timestamps[0] <= measurement.timestamp <= timestamps[-1]
    ]
    source = np.array([measurement.position for measurement in quality4])
    measurement_times = np.array(
        [measurement.timestamp for measurement in quality4]
    )
    target = np.column_stack(
        (
            np.interp(measurement_times, timestamps, reference[:, 0]),
            np.interp(measurement_times, timestamps, reference[:, 1]),
        )
    )
    raw_rotation, raw_translation = rigid_align_2d(source, target)
    raw_residual = np.linalg.norm(
        source @ raw_rotation.T + raw_translation - target,
        axis=1,
    )
    keep = raw_residual <= np.percentile(raw_residual, 90)
    trimmed_rotation, trimmed_translation = rigid_align_2d(
        source[keep], target[keep]
    )
    trimmed_residual = np.linalg.norm(
        source @ trimmed_rotation.T + trimmed_translation - target,
        axis=1,
    )
    return {
        "raw_replay": raw_replay,
        "timestamps": measurement_times,
        "keep": keep,
        "raw_rotation": raw_rotation,
        "raw_translation": raw_translation,
        "trimmed_rotation": trimmed_rotation,
        "trimmed_translation": trimmed_translation,
        "raw_residual": raw_residual,
        "trimmed_residual": trimmed_residual,
        "summary": {
            "raw_fit_all_points": summarize_residual(raw_residual),
            "trimmed_fit_all_points": summarize_residual(trimmed_residual),
            "trimmed_fit_inliers_only": summarize_residual(
                trimmed_residual[keep]
            ),
        },
    }


def run_trial(
    recording: Path,
    replay: GPSReplay,
    fusion_type: type[LocalizationFusion],
    seed: int,
) -> dict:
    timestamps, reference, _ = load_reference_trajectory(recording)
    replay.seek(float(timestamps[0]))
    fusion = fusion_type()
    output_timestamps = []
    predicted = []
    local = []
    aligned_reference = []
    for index, odometry in enumerate(
        odometry_proxy_from_recording(recording, seed=seed),
        start=1,
    ):
        for gps_measurement in replay.pop_until(odometry.timestamp):
            fusion.process_gps(gps_measurement)
        fusion.process_odometry(odometry)
        output_timestamps.append(odometry.timestamp)
        predicted.append(fusion.global_pose)
        local.append(fusion.local_pose)
        aligned_reference.append(reference[index])
    return evaluate_run(
        fusion,
        replay,
        np.asarray(output_timestamps),
        np.asarray(predicted),
        np.asarray(local),
        np.asarray(aligned_reference),
    )


def selected_metrics(report: dict) -> dict:
    return {
        "relock_max_error_after_2s_m": report["relock"][
            "max_error_after_2s_m"
        ],
        "max_correction_induced_discontinuity_m": report["local_continuity"][
            "max_correction_induced_discontinuity_m"
        ],
        "trajectory_median_error_m": report["trajectory_error_m"]["median"],
        "trajectory_p95_error_m": report["trajectory_error_m"]["p95"],
    }


def quality_zero_audit(recording: Path) -> dict:
    timestamps, _, _ = load_reference_trajectory(recording)
    replay = load_nmea_replay(recording)
    measurements = [
        measurement
        for measurement in replay.measurements
        if timestamps[0] <= measurement.timestamp <= timestamps[-1]
    ]
    runs = []
    length = 0
    for measurement in measurements:
        if measurement.fix_quality == 0:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return {
        "quality_zero_run_lengths_messages": runs,
        "episodes_before_debounce": len(runs),
        "episodes_confirmed_by_debounce_3": sum(run >= 3 for run in runs),
        "episodes_removed_by_debounce_3": sum(run < 3 for run in runs),
    }


def git_revision() -> dict:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "head_commit": (
            result.stdout.strip() if result.returncode == 0 else None
        ),
        "worktree_dirty": (
            bool(status.stdout.strip()) if status.returncode == 0 else None
        ),
    }


def run(args: argparse.Namespace) -> dict:
    recording = args.dataset / "recording_2021-02-25_13-39-06"
    if not recording.is_dir():
        raise FileNotFoundError(recording)
    alignment = alignment_trial(recording)

    baseline = run_trial(
        recording,
        load_nmea_replay(recording, align_to_reference=True),
        LocalizationFusion,
        args.seed,
    )
    adaptive_raw = run_trial(
        recording,
        transform_replay(
            alignment["raw_replay"],
            alignment["raw_rotation"],
            alignment["raw_translation"],
            "raw_rigid_all_quality4",
        ),
        CovarianceRateTrialFusion,
        args.seed,
    )
    adaptive_trimmed = run_trial(
        recording,
        transform_replay(
            alignment["raw_replay"],
            alignment["trimmed_rotation"],
            alignment["trimmed_translation"],
            "trimmed_p90_rigid_all_quality4",
        ),
        CovarianceRateTrialFusion,
        args.seed,
    )
    baseline_metrics = selected_metrics(baseline)
    raw_metrics = selected_metrics(adaptive_raw)
    trimmed_metrics = selected_metrics(adaptive_trimmed)
    relock_effect = (
        "improves"
        if raw_metrics["relock_max_error_after_2s_m"]
        < baseline_metrics["relock_max_error_after_2s_m"]
        else "worsens"
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": git_revision(),
        "recording": recording.name,
        "seed": args.seed,
        "purpose": (
            "Failure-analysis evidence for rejected covariance-rate and "
            "all-track trimmed-alignment proposals"
        ),
        "debounce_audit": quality_zero_audit(recording),
        "alignment_trials": alignment["summary"],
        "fusion_trials": {
            "production_baseline": baseline_metrics,
            "covariance_rate_raw_alignment": {
                "formula": "max(3.0, 1.5 * sqrt(trace(P_map)))",
                **raw_metrics,
            },
            "covariance_rate_trimmed_p90_alignment": {
                "formula": "max(3.0, 1.5 * sqrt(trace(P_map)))",
                **trimmed_metrics,
            },
        },
        "conclusion": [
            "Debounce=3 removes only quality-zero runs shorter than 3 messages.",
            f"Covariance-rate trial {relock_effect} re-lock versus the current worktree baseline, violates 0.5 m no-jump, and misses B8.",
            "Trimmed alignment worsens all-point p95/RMSE and does not improve re-lock.",
            "These trials are rejected prototypes, not production benchmark claims.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark_trials.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output / "alignment_residuals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "kept_by_raw_p90",
                "raw_fit_residual_m",
                "trimmed_fit_residual_m",
            ]
        )
        writer.writerows(
            zip(
                alignment["timestamps"],
                alignment["keep"].astype(int),
                alignment["raw_residual"],
                alignment["trimmed_residual"],
            )
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/data/4seasons"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gps-fusion-trials"),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
