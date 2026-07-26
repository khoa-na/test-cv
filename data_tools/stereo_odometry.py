"""Adapter causal từ recording 4Seasons sang StereoVO."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2

from data_tools.fourseasons import load_calibration
from data_tools.gps_sources import OdometryMeasurement
from data_tools.imu_yaw import load_imu_yaw_integrator
from pipelines.stereo_vo import StereoVO, StereoVOConfig


@dataclass(frozen=True)
class StereoFrame:
    frame_id: str
    timestamp: float
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class StereoOdometryFrame:
    frame_id: str
    timestamp: float
    measurement: OdometryMeasurement | None
    processing_seconds: float


def stereo_frames(recording: str | Path) -> list[StereoFrame]:
    """Join cam0/cam1/times bằng frame-id dạng chuỗi, không qua float."""
    recording = Path(recording)
    timestamps = {}
    for line in (recording / "times.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            timestamps[parts[0]] = float(parts[1])
    left = {
        path.stem: path
        for path in (recording / "undistorted_images" / "cam0").glob("*.png")
    }
    right = {
        path.stem: path
        for path in (recording / "undistorted_images" / "cam1").glob("*.png")
    }
    frame_ids = sorted(
        timestamps.keys() & left.keys() & right.keys(),
        key=lambda item: timestamps[item],
    )
    if not frame_ids:
        raise ValueError(f"{recording} không có stereo frame đồng bộ")
    return [
        StereoFrame(
            frame_id=frame_id,
            timestamp=timestamps[frame_id],
            left_path=left[frame_id],
            right_path=right[frame_id],
        )
        for frame_id in frame_ids
    ]


def run_stereo_odometry(
    recording: str | Path,
    calibration_dir: str | Path,
    *,
    config: StereoVOConfig | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    timestamp_min: float | None = None,
    timestamp_max: float | None = None,
    yaw_source: str = "visual",
) -> list[StereoOdometryFrame]:
    recording = Path(recording)
    frames = stereo_frames(recording)
    if timestamp_min is not None:
        frames = [frame for frame in frames if frame.timestamp >= timestamp_min]
    if timestamp_max is not None:
        frames = [frame for frame in frames if frame.timestamp <= timestamp_max]
    if not frames:
        raise ValueError("Không còn stereo frame trong miền timestamp yêu cầu")
    if start_frame < 0 or start_frame >= len(frames):
        raise ValueError("start_frame nằm ngoài recording")
    selected = frames[start_frame:]
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames phải >= 2")
        selected = selected[:max_frames]
    vo = StereoVO(load_calibration(Path(calibration_dir)), config)
    if yaw_source not in {"visual", "imu"}:
        raise ValueError(f"yaw_source không hỗ trợ: {yaw_source}")
    imu_yaw = (
        load_imu_yaw_integrator(recording, frames[0].timestamp)
        if yaw_source == "imu"
        else None
    )
    output = []
    for frame in selected:
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise ValueError(f"Không đọc được stereo frame {frame.frame_id}")
        started = perf_counter()
        measurement = vo.process(left, right, frame.timestamp)
        if measurement is not None and imu_yaw is not None:
            imu_heading = imu_yaw.delta_heading(
                measurement.timestamp - measurement.dt,
                measurement.timestamp,
            )
            if imu_heading is not None:
                measurement = replace(
                    measurement,
                    dtheta=imu_heading,
                    source="stereo_vo_imu_yaw",
                )
        elapsed = perf_counter() - started
        output.append(
            StereoOdometryFrame(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                measurement=measurement,
                processing_seconds=elapsed,
            )
        )
    return output
