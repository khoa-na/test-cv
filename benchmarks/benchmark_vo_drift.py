#!/usr/bin/env python3
"""Benchmark B1: stereo VO relative drift trên cửa sổ 500 m."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from data_tools.gps_sources import load_reference_trajectory, wrap_angle
from data_tools.imu_yaw import IMUYawConfig, load_imu_yaw_integrator
from data_tools.stereo_odometry import run_stereo_odometry
from pipelines.stereo_vo import StereoVO, StereoVOConfig


SEQUENCES = {
    "office_loop_1": "recording_2020-03-24_17-36-22",
    "neighborhood_4": "recording_2020-12-22_11-54-24",
    "garage_2": "recording_2021-02-25_13-39-06",
    # Bước 3 dựng landmark map từ garage_3, nên drift của chính nó là trần chất
    # lượng map — không thể suy ra từ garage_2.
    "garage_3": "recording_2021-05-10_19-15-19",
}


def interpolate_reference(
    recording: Path,
    timestamps: np.ndarray,
) -> np.ndarray:
    reference_timestamps, reference_xy, reference_heading = (
        load_reference_trajectory(recording)
    )
    if (
        timestamps[0] < reference_timestamps[0]
        or timestamps[-1] > reference_timestamps[-1]
    ):
        valid = (timestamps >= reference_timestamps[0]) & (
            timestamps <= reference_timestamps[-1]
        )
        if not np.all(valid):
            raise ValueError("VO timestamp nằm ngoài miền reference pose")
    return np.column_stack(
        (
            np.interp(timestamps, reference_timestamps, reference_xy[:, 0]),
            np.interp(timestamps, reference_timestamps, reference_xy[:, 1]),
            np.interp(
                timestamps,
                reference_timestamps,
                reference_heading,
            ),
        )
    )


def normalize_trajectory(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    origin = poses[0]
    delta = poses[:, :2] - origin[:2]
    cosine, sine = np.cos(origin[2]), np.sin(origin[2])
    normalized_xy = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        )
    )
    return np.column_stack(
        (
            normalized_xy,
            wrap_angle(poses[:, 2] - origin[2]),
        )
    )


def integrate_measurements(frames: list) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    poses = [np.zeros(3, dtype=np.float64)]
    last_velocity = np.zeros(3, dtype=np.float64)
    for frame in frames[1:]:
        pose = poses[-1]
        measurement = frame.measurement
        if measurement is not None:
            last_velocity = np.array(
                [
                    measurement.dx / measurement.dt,
                    measurement.dy / measurement.dt,
                    measurement.dtheta / measurement.dt,
                ]
            )
            relative_motion = np.array(
                [measurement.dx, measurement.dy, measurement.dtheta]
            )
        else:
            dt = frame.timestamp - timestamps[len(poses) - 1]
            relative_motion = last_velocity * dt
        pose = StereoVO.integrate_relative(pose, *relative_motion)
        poses.append(pose.copy())
    return timestamps, np.asarray(poses)


def drift_windows(
    vo_poses: np.ndarray,
    reference_poses: np.ndarray,
    *,
    window_length_m: float = 500.0,
) -> list[dict]:
    if len(vo_poses) != len(reference_poses):
        raise ValueError("VO/reference phải cùng số pose")
    segment_lengths = np.linalg.norm(
        np.diff(reference_poses[:, :2], axis=0),
        axis=1,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    windows = []
    start_index = 0
    while start_index < len(cumulative) - 1:
        target = cumulative[start_index] + window_length_m
        end_index = int(np.searchsorted(cumulative, target))
        if end_index >= len(cumulative):
            break
        path_length = float(cumulative[end_index] - cumulative[start_index])
        vo_relative = normalize_trajectory(
            vo_poses[[start_index, end_index]]
        )[-1]
        reference_relative = normalize_trajectory(
            reference_poses[[start_index, end_index]]
        )[-1]
        translation_error = float(
            np.linalg.norm(vo_relative[:2] - reference_relative[:2])
        )
        rotation_error = float(
            abs(wrap_angle(vo_relative[2] - reference_relative[2]))
        )
        windows.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "path_length_m": path_length,
                "translation_error_m": translation_error,
                "translation_drift_ratio": translation_error / path_length,
                "rotation_error_deg": float(np.rad2deg(rotation_error)),
                "rotation_drift_deg_per_100m": float(
                    np.rad2deg(rotation_error) * 100.0 / path_length
                ),
            }
        )
        start_index = end_index
    return windows


def summarize_windows(windows: list[dict]) -> dict:
    if not windows:
        return {
            "count": 0,
            "median_translation_drift_ratio": None,
            "p95_translation_drift_ratio": None,
            "pass_rate_le_5_percent": None,
        }
    values = np.array(
        [window["translation_drift_ratio"] for window in windows]
    )
    return {
        "count": len(windows),
        "median_translation_drift_ratio": float(np.median(values)),
        "p95_translation_drift_ratio": float(np.percentile(values, 95)),
        "pass_rate_le_5_percent": float(np.mean(values <= 0.05)),
        "max_rotation_drift_deg_per_100m": float(
            max(window["rotation_drift_deg_per_100m"] for window in windows)
        ),
    }


def summarize_cases(cases: dict[str, dict]) -> dict:
    windows = [
        (name, index, window)
        for name, case in cases.items()
        for index, window in enumerate(case["windows"], start=1)
    ]
    failed = [
        {
            "sequence": name,
            "window_index": index,
            "translation_drift_ratio": window[
                "translation_drift_ratio"
            ],
            "translation_error_m": window["translation_error_m"],
            "path_length_m": window["path_length_m"],
        }
        for name, index, window in windows
        if window["translation_drift_ratio"] > 0.05
    ]
    passed = len(windows) - len(failed)
    return {
        "total_windows": len(windows),
        "passed_windows": passed,
        "failed_windows": len(failed),
        "pass_rate_le_5_percent": (
            passed / len(windows) if windows else None
        ),
        "all_windows_pass": bool(windows) and not failed,
        "failures": failed,
    }


def refresh_existing_report(
    report_path: Path,
    dataset: Path,
    *,
    yaw_source: str,
) -> dict:
    """Bổ sung aggregate/calibration metadata mà không chạy lại VO.

    Chỉ tái sử dụng các metric đã có trong artifact; IMU gate được tính lại
    từ raw ``imu.txt`` và timestamp đầu reference nên vẫn có provenance.
    """
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report.get("cases", {})
    if not cases:
        raise ValueError("Artifact cũ không có cases")
    for name, case in cases.items():
        if yaw_source != "imu":
            case["imu_yaw_calibration"] = None
            continue
        recording_name = case.get("recording") or SEQUENCES.get(name)
        if recording_name is None:
            raise ValueError(f"Không xác định được recording cho {name}")
        recording = dataset / recording_name
        reference_timestamps, _, _ = load_reference_trajectory(recording)
        case["imu_yaw_calibration"] = load_imu_yaw_integrator(
            recording, float(reference_timestamps[0])
        ).metadata()
    report["summary_all_sequences"] = summarize_cases(cases)
    report["artifact_refresh"] = {
        "reused_existing_vo_metrics": True,
        "recomputed_fields": [
            "summary_all_sequences",
            "cases.*.imu_yaw_calibration",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def render_overlay(
    path: Path,
    vo_poses: np.ndarray,
    reference_poses: np.ndarray,
) -> None:
    canvas = np.full((800, 1000, 3), 245, dtype=np.uint8)
    values = np.vstack((vo_poses[:, :2], reference_poses[:, :2]))
    minimum = values.min(axis=0)
    extent = np.maximum(values.max(axis=0) - minimum, 1.0)
    scale = min(900.0 / extent[0], 700.0 / extent[1])

    def project(points: np.ndarray) -> np.ndarray:
        pixels = (points - minimum) * scale + np.array([50.0, 50.0])
        pixels[:, 1] = canvas.shape[0] - pixels[:, 1]
        return np.rint(pixels).astype(np.int32)

    cv2.polylines(
        canvas,
        [project(reference_poses[:, :2])],
        False,
        (100, 100, 100),
        2,
    )
    cv2.polylines(
        canvas,
        [project(vo_poses[:, :2])],
        False,
        (40, 80, 220),
        2,
    )
    cv2.putText(
        canvas,
        "gray=reference, red=stereo VO",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def benchmark_recording(
    recording: Path,
    calibration_dir: Path,
    output: Path,
    name: str,
    *,
    config: StereoVOConfig,
    start_frame: int = 0,
    max_frames: int | None = None,
    render: bool = True,
    yaw_source: str = "visual",
    imu_config: IMUYawConfig | None = None,
) -> dict:
    reference_timestamps, _, _ = load_reference_trajectory(recording)
    frames = run_stereo_odometry(
        recording,
        calibration_dir,
        config=config,
        start_frame=start_frame,
        max_frames=max_frames,
        timestamp_min=float(reference_timestamps[0]),
        timestamp_max=float(reference_timestamps[-1]),
        yaw_source=yaw_source,
        imu_config=imu_config,
    )
    imu_metadata = None
    if yaw_source == "imu":
        imu_metadata = load_imu_yaw_integrator(
            recording,
            frames[0].timestamp,
            imu_config,
        ).metadata()
    timestamps, vo_poses = integrate_measurements(frames)
    reference_poses = interpolate_reference(recording, timestamps)
    normalized_reference = normalize_trajectory(reference_poses)
    windows = drift_windows(vo_poses, normalized_reference)
    valid_count = sum(frame.measurement is not None for frame in frames[1:])
    attempted = max(len(frames) - 1, 1)
    processing_seconds = sum(frame.processing_seconds for frame in frames)
    valid_vo_distance = 0.0
    valid_reference_distance = 0.0
    for frame in frames[1:]:
        measurement = frame.measurement
        if measurement is None:
            continue
        valid_vo_distance += float(np.hypot(measurement.dx, measurement.dy))
        start_time = measurement.timestamp - measurement.dt
        start_xy = np.array(
            [
                np.interp(start_time, timestamps, reference_poses[:, 0]),
                np.interp(start_time, timestamps, reference_poses[:, 1]),
            ]
        )
        end_xy = np.array(
            [
                np.interp(
                    measurement.timestamp,
                    timestamps,
                    reference_poses[:, 0],
                ),
                np.interp(
                    measurement.timestamp,
                    timestamps,
                    reference_poses[:, 1],
                ),
            ]
        )
        valid_reference_distance += float(np.linalg.norm(end_xy - start_xy))
    cumulative_error = float(
        np.linalg.norm(vo_poses[-1, :2] - normalized_reference[-1, :2])
    )
    result = {
        "recording": recording.name,
        "frames": len(frames),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "reference_path_length_m": float(
            np.linalg.norm(
                np.diff(normalized_reference[:, :2], axis=0),
                axis=1,
            ).sum()
        ),
        "vo_path_length_m": float(
            np.linalg.norm(np.diff(vo_poses[:, :2], axis=0), axis=1).sum()
        ),
        "path_length_ratio": (
            float(
                np.linalg.norm(np.diff(vo_poses[:, :2], axis=0), axis=1).sum()
                / np.linalg.norm(
                    np.diff(normalized_reference[:, :2], axis=0),
                    axis=1,
                ).sum()
            )
            if len(frames) > 1
            else None
        ),
        "valid_updates": valid_count,
        "imu_yaw_updates": sum(
            frame.measurement is not None
            and frame.measurement.source == "stereo_vo_imu_yaw"
            for frame in frames
        ),
        "imu_yaw_calibration": imu_metadata,
        "dropout_rate": 1.0 - valid_count / attempted,
        "valid_update_distance_ratio": (
            valid_vo_distance / valid_reference_distance
            if valid_reference_distance > 0
            else None
        ),
        "reference_distance_not_covered_by_updates_m": float(
            max(
                np.linalg.norm(
                    np.diff(normalized_reference[:, :2], axis=0),
                    axis=1,
                ).sum()
                - valid_reference_distance,
                0.0,
            )
        ),
        "processing_fps": len(frames) / processing_seconds,
        "cumulative_translation_error_m": cumulative_error,
        "windows": windows,
        "summary_500m": summarize_windows(windows),
    }
    if render:
        render_overlay(
            output / f"{name}.png",
            vo_poses,
            normalized_reference,
        )
    return result


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
        default=Path("artifacts/vo-drift"),
    )
    parser.add_argument(
        "--sequence",
        choices=sorted(SEQUENCES),
        action="append",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--yaw-source",
        choices=("visual", "imu"),
        default="imu",
    )
    parser.add_argument(
        "--gyro-gate-override",
        type=float,
        help=(
            "CHẨN ĐOÁN, không phải config nộp bài: ép ngưỡng "
            "stationary_gyro_p95_rad_s để chạy phản chứng trên sequence bị "
            "cổng chặn. Mặc định giữ 0.02."
        ),
    )
    parser.add_argument(
        "--bias-calibration-mode",
        choices=("recording_start", "quietest_window"),
        default="recording_start",
        help=(
            "CHẨN ĐOÁN: quietest_window quét cả recording tìm cửa sổ 2 s yên "
            "nhất thay vì mặc định lấy 2 s đầu. Bài nộp dùng recording_start."
        ),
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help=(
            "Chỉ bổ sung aggregate + IMU calibration metadata vào "
            "output/benchmark.json; không chạy lại VO."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.refresh_existing:
        return refresh_existing_report(
            args.output / "benchmark.json",
            args.dataset,
            yaw_source=args.yaw_source,
        )
    names = args.sequence or list(SEQUENCES)
    config = StereoVOConfig()
    defaults = IMUYawConfig()
    imu_config = IMUYawConfig(
        stationary_gyro_p95_rad_s=(
            args.gyro_gate_override
            if args.gyro_gate_override is not None
            else defaults.stationary_gyro_p95_rad_s
        ),
        bias_calibration_mode=args.bias_calibration_mode,
    )
    cases = {}
    for name in names:
        recording = args.dataset / SEQUENCES[name]
        if not recording.is_dir():
            raise FileNotFoundError(recording)
        cases[name] = benchmark_recording(
            recording,
            args.dataset / "calibration",
            args.output,
            name,
            config=config,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            render=not args.no_render,
            yaw_source=args.yaw_source,
            imu_config=imu_config,
        )
    report = {
        "method": (
            "ORB stereo depth + temporal PnP; "
            f"yaw_source={args.yaw_source}"
        ),
        "gyro_gate_override": args.gyro_gate_override,
        "bias_calibration_mode": args.bias_calibration_mode,
        "config": asdict(config),
        "metric": (
            "Relative start-to-end SE(2) error per non-overlapping "
            "500 m reference-path window; no scale fit"
        ),
        "reference_usage": "Offline evaluator only; never passed to StereoVO",
        "summary_all_sequences": summarize_cases(cases),
        "cases": cases,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run(parse_args())
    for name, case in report["cases"].items():
        print(
            name,
            "fps=",
            round(case["processing_fps"], 2),
            "dropout=",
            round(case["dropout_rate"], 4),
            "path_ratio=",
            round(case["path_length_ratio"], 4),
            "B1=",
            case["summary_500m"],
        )


if __name__ == "__main__":
    main()
