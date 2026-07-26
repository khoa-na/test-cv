"""Gyro yaw-rate adapter cho 4Seasons, có stationary bias gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def quaternion_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError("Quaternion extrinsic không hợp lệ")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def load_camera_imu_rotation(recording: str | Path) -> np.ndarray:
    lines = Path(recording, "Transformations.txt").read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("# TS_cam_imu"):
            values = np.fromstring(lines[index + 1], sep=",")
            if len(values) != 7:
                break
            return quaternion_to_rotation(values[3:7])
    raise ValueError("Không tìm thấy TS_cam_imu trong Transformations.txt")


@dataclass(frozen=True)
class IMUYawConfig:
    bias_calibration_seconds: float = 2.0
    stationary_gyro_p95_rad_s: float = 0.02


class IMUYawIntegrator:
    """Tích phân heading vehicle từ gyro đã đổi sang camera frame."""

    def __init__(
        self,
        timestamps: np.ndarray,
        gyro_imu: np.ndarray,
        rotation_camera_imu: np.ndarray,
        calibration_start: float,
        config: IMUYawConfig | None = None,
    ) -> None:
        self.config = config or IMUYawConfig()
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        gyro = np.asarray(gyro_imu, dtype=np.float64)
        if (
            self.timestamps.ndim != 1
            or gyro.shape != (len(self.timestamps), 3)
            or len(self.timestamps) < 2
            or np.any(np.diff(self.timestamps) <= 0)
        ):
            raise ValueError("IMU timestamps/gyro không hợp lệ")
        gyro_camera = gyro @ np.asarray(
            rotation_camera_imu,
            dtype=np.float64,
        ).T
        calibration = (
            (self.timestamps >= calibration_start)
            & (
                self.timestamps
                <= calibration_start + self.config.bias_calibration_seconds
            )
        )
        if calibration.sum() < 10:
            raise ValueError("Không đủ IMU sample để calibrate gyro bias")
        stationary_score = float(
            np.percentile(np.linalg.norm(gyro_camera[calibration], axis=1), 95)
        )
        self.stationary_calibration_passed = (
            stationary_score <= self.config.stationary_gyro_p95_rad_s
        )
        self.stationary_score_rad_s = stationary_score
        self.bias_camera_y_rad_s = float(
            np.median(gyro_camera[calibration, 1])
        )
        vehicle_yaw_rate = -(
            gyro_camera[:, 1] - self.bias_camera_y_rad_s
        )
        intervals = np.diff(self.timestamps)
        self.integrated_heading = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    0.5
                    * (vehicle_yaw_rate[:-1] + vehicle_yaw_rate[1:])
                    * intervals
                ),
            )
        )

    def delta_heading(self, start: float, end: float) -> float | None:
        if (
            not self.stationary_calibration_passed
            or start < self.timestamps[0]
            or end > self.timestamps[-1]
            or end <= start
        ):
            return None
        values = np.interp(
            [start, end],
            self.timestamps,
            self.integrated_heading,
        )
        return float(values[1] - values[0])

    def metadata(self) -> dict:
        return {
            "low_angular_rate_gate_passed": (
                self.stationary_calibration_passed
            ),
            "gyro_norm_p95_rad_s": self.stationary_score_rad_s,
            "gyro_norm_p95_threshold_rad_s": (
                self.config.stationary_gyro_p95_rad_s
            ),
            "bias_camera_y_rad_s": self.bias_camera_y_rad_s,
            "calibration_seconds": self.config.bias_calibration_seconds,
        }


def load_imu_yaw_integrator(
    recording: str | Path,
    calibration_start: float,
    config: IMUYawConfig | None = None,
) -> IMUYawIntegrator:
    recording = Path(recording)
    imu = np.loadtxt(recording / "imu.txt")
    return IMUYawIntegrator(
        imu[:, 0] / 1e9,
        imu[:, 1:4],
        load_camera_imu_rotation(recording),
        calibration_start,
        config,
    )
