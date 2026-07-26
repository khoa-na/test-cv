#!/usr/bin/env python3
"""Calibration/holdout audit cho NMEA alignment của 4Seasons.

Không dùng holdout để fit hay chọn phương án. Benchmark so rigid fit 2D hiện
tại với transform chain chính thức, quét clock offset trên calibration split,
và định lượng camera/IMU lever arm như một ablation riêng.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_tools import fourseasons
from data_tools.gps_sources import (
    GPSReplay,
    load_nmea_replay,
    load_reference_trajectory,
    rigid_align_2d,
)


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(len(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "max_m": float(np.max(values)),
    }


def quality4_positions(
    replay: GPSReplay, reference_timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    measurements = [
        measurement
        for measurement in replay.measurements
        if measurement.fix_quality == 4
        and measurement.has_position
        and reference_timestamps[0]
        <= measurement.timestamp
        <= reference_timestamps[-1]
    ]
    return (
        np.array([item.timestamp for item in measurements]),
        np.array([item.position for item in measurements]),
    )


def interpolate_reference(
    query_timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
    reference_xy: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            np.interp(
                query_timestamps, reference_timestamps, reference_xy[:, 0]
            ),
            np.interp(
                query_timestamps, reference_timestamps, reference_xy[:, 1]
            ),
        )
    )


def quaternion_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def camera_lever_corrected(
    gps_imu_xy: np.ndarray,
    query_timestamps: np.ndarray,
    reference_poses: np.ndarray,
    camera_imu_translation: np.ndarray,
) -> np.ndarray:
    """GPS/IMU origin -> camera center, dùng orientation GT chỉ cho ablation."""
    reference_timestamps = reference_poses[:, 0]
    right = np.searchsorted(reference_timestamps, query_timestamps)
    right = np.clip(right, 1, len(reference_timestamps) - 1)
    left = right - 1
    use_right = (
        np.abs(reference_timestamps[right] - query_timestamps)
        < np.abs(query_timestamps - reference_timestamps[left])
    )
    nearest = np.where(use_right, right, left)
    correction = np.array(
        [
            quaternion_rotation_matrix(reference_poses[index, 4:8])
            @ camera_imu_translation
            for index in nearest
        ]
    )
    return gps_imu_xy - correction[:, :2]


def evaluate_candidate(
    source: np.ndarray,
    timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
    reference_xy: np.ndarray,
    calibration_indices: np.ndarray,
    holdout_indices: np.ndarray,
    offsets: np.ndarray,
    *,
    fit_rigid: bool,
) -> dict:
    trials = []
    for offset in offsets:
        target = interpolate_reference(
            timestamps + offset, reference_timestamps, reference_xy
        )
        rotation = np.eye(2)
        translation = np.zeros(2)
        if fit_rigid:
            rotation, translation = rigid_align_2d(
                source[calibration_indices], target[calibration_indices]
            )
        predicted = source @ rotation.T + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        trials.append(
            {
                "offset_s": float(offset),
                "objective_calibration_median_m": float(
                    np.median(residual[calibration_indices])
                ),
                "rotation": rotation,
                "translation": translation,
                "residual": residual,
            }
        )
    selected = min(
        trials,
        key=lambda item: (
            item["objective_calibration_median_m"],
            abs(item["offset_s"]),
        ),
    )
    residual = selected["residual"]
    zero = min(trials, key=lambda item: abs(item["offset_s"]))
    return {
        "selected_offset_s": selected["offset_s"],
        "selection_objective": "calibration median position error",
        "calibration": summarize(residual[calibration_indices]),
        "holdout": summarize(residual[holdout_indices]),
        "zero_offset_calibration": summarize(
            zero["residual"][calibration_indices]
        ),
        "zero_offset_holdout": summarize(zero["residual"][holdout_indices]),
        "rotation": selected["rotation"].tolist(),
        "translation": selected["translation"].tolist(),
        "_residual": residual,
    }


def render_residuals(
    output: Path,
    timestamps: np.ndarray,
    calibration_count: int,
    candidates: dict,
) -> None:
    relative_time = timestamps - timestamps[0]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for name, candidate in candidates.items():
        axis.plot(
            relative_time,
            candidate["_residual"],
            linewidth=1,
            label=name,
        )
    boundary = relative_time[calibration_count - 1]
    axis.axvline(boundary, color="black", linestyle="--", label="cal/holdout")
    axis.set_xlabel("Time from first RTK fix (s)")
    axis.set_ylabel("Position residual (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict:
    recording = args.dataset / args.recording
    reference_timestamps, reference_xy, _ = load_reference_trajectory(recording)
    reference_poses = fourseasons.load_reference_poses(recording)
    raw_replay = load_nmea_replay(recording, alignment_mode="raw_enu")
    chain_replay = load_nmea_replay(
        recording, alignment_mode="transform_chain"
    )
    timestamps, raw_positions = quality4_positions(
        raw_replay, reference_timestamps
    )
    chain_timestamps, chain_positions = quality4_positions(
        chain_replay, reference_timestamps
    )
    if not np.array_equal(timestamps, chain_timestamps):
        raise ValueError("Raw ENU và transform-chain không cùng sample")
    if len(timestamps) < 10:
        raise ValueError("Không đủ quality-4 sample để chia calibration/holdout")

    calibration_count = int(np.floor(len(timestamps) * args.calibration_fraction))
    if calibration_count < 3 or len(timestamps) - calibration_count < 3:
        raise ValueError("Calibration/holdout split cần ít nhất 3 sample mỗi bên")
    calibration_indices = np.arange(calibration_count)
    holdout_indices = np.arange(calibration_count, len(timestamps))
    offsets = np.arange(
        args.offset_min_s,
        args.offset_max_s + args.offset_step_s * 0.5,
        args.offset_step_s,
    )
    candidates = {
        "rigid_enu_calibration_only": evaluate_candidate(
            raw_positions,
            timestamps,
            reference_timestamps,
            reference_xy,
            calibration_indices,
            holdout_indices,
            offsets,
            fit_rigid=True,
        ),
        "official_transform_chain": evaluate_candidate(
            chain_positions,
            timestamps,
            reference_timestamps,
            reference_xy,
            calibration_indices,
            holdout_indices,
            offsets,
            fit_rigid=False,
        ),
    }

    transforms = fourseasons.load_transformations(recording)
    gps_imu_lever = np.asarray(transforms["transform_gps_imu"])[:3, 3]
    camera_imu_translation = np.asarray(
        transforms["transform_cam_imu"]
    )[:3, 3]
    chain_offset = candidates["official_transform_chain"][
        "selected_offset_s"
    ]
    lever_positions = camera_lever_corrected(
        chain_positions,
        timestamps + chain_offset,
        reference_poses,
        camera_imu_translation,
    )
    target = interpolate_reference(
        timestamps + chain_offset, reference_timestamps, reference_xy
    )
    lever_residual = np.linalg.norm(lever_positions - target, axis=1)
    lever_ablation = {
        "gps_imu_lever_arm_m": float(np.linalg.norm(gps_imu_lever)),
        "camera_imu_lever_arm_m": float(
            np.linalg.norm(camera_imu_translation)
        ),
        "uses_reference_orientation": True,
        "eligible_for_fusion_selection": False,
        "reason": (
            "Reference orientation is used only to bound the lever-arm effect; "
            "using it in fusion would leak ground truth."
        ),
        "calibration": summarize(lever_residual[calibration_indices]),
        "holdout": summarize(lever_residual[holdout_indices]),
    }

    selectable = {
        name: candidate
        for name, candidate in candidates.items()
        if name != "lever_arm_ablation"
    }
    selected_name = min(
        selectable,
        key=lambda name: (
            selectable[name]["calibration"]["median_m"],
            abs(selectable[name]["selected_offset_s"]),
        ),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    render_residuals(
        args.output / "alignment_residuals.png",
        timestamps,
        calibration_count,
        candidates,
    )
    with (args.output / "alignment_residuals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "split",
                *[f"{name}_residual_m" for name in candidates],
            ]
        )
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                [
                    timestamp,
                    "calibration" if index < calibration_count else "holdout",
                    *[
                        candidates[name]["_residual"][index]
                        for name in candidates
                    ],
                ]
            )

    for candidate in candidates.values():
        candidate.pop("_residual")
    report = {
        "recording": recording.name,
        "protocol": {
            "split": "chronological",
            "calibration_fraction": args.calibration_fraction,
            "calibration_samples": calibration_count,
            "holdout_samples": len(timestamps) - calibration_count,
            "holdout_used_for_fit_or_selection": False,
            "offset_grid_s": {
                "min": args.offset_min_s,
                "max": args.offset_max_s,
                "step": args.offset_step_s,
            },
        },
        "candidates": candidates,
        "lever_arm_ablation": lever_ablation,
        "selected_for_round5": {
            "name": selected_name,
            "time_offset_s": selectable[selected_name][
                "selected_offset_s"
            ],
            "basis": "lowest calibration median among deployable candidates",
        },
        "limitations": [
            "Reference trajectory combines stereo VIO and RTK-GNSS.",
            "Official Transformations.txt may itself be dataset-calibrated.",
            "Chronological holdout is reported but never used for selection.",
            "Lever-arm ablation uses GT orientation and is not deployable.",
        ],
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(".cache/data/4seasons")
    )
    parser.add_argument(
        "--recording", default="recording_2021-02-25_13-39-06"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gps-alignment-fix3"),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--offset-min-s", type=float, default=-1.0)
    parser.add_argument("--offset-max-s", type=float, default=1.0)
    parser.add_argument("--offset-step-s", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.calibration_fraction < 1:
        parser.error("--calibration-fraction phải nằm trong (0,1)")
    if args.offset_step_s <= 0 or args.offset_min_s > args.offset_max_s:
        parser.error("offset grid không hợp lệ")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
