"""Nguồn GPS và odometry test-double cho pipeline localization.

Reference pose chỉ được đọc ở đây để tạo dữ liệu mô phỏng/proxy offline và
để căn frame cho benchmark. Pipeline localization chỉ nhận các measurement
đã được tạo, không đọc ground truth trong lúc chạy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from data_tools import fourseasons


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Chuẩn hóa góc về [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class GPSMeasurement:
    timestamp: float
    x: float | None
    y: float | None
    fix_quality: int
    satellites: int
    hdop: float | None
    source: str = "gps"
    mode: str = "real"

    @property
    def has_position(self) -> bool:
        return (
            self.x is not None
            and self.y is not None
            and np.isfinite(self.x)
            and np.isfinite(self.y)
        )

    @property
    def position(self) -> np.ndarray:
        if not self.has_position:
            raise ValueError("GPS measurement không có vị trí hợp lệ")
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class OdometryMeasurement:
    """Relative SE(2) motion trong vehicle frame tại frame trước."""

    timestamp: float
    dt: float
    dx: float
    dy: float
    dtheta: float
    translation_std: float
    rotation_std: float
    source: str = "proxy"

    @property
    def v_meas(self) -> float:
        return self.dx / self.dt

    @property
    def omega_meas(self) -> float:
        return self.dtheta / self.dt


@dataclass(frozen=True)
class ScenarioSegment:
    start: float
    end: float
    mode: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Scenario segment phải có end > start")
        if self.mode not in {"good", "degraded", "lost", "recovering"}:
            raise ValueError(f"GPS mode không hỗ trợ: {self.mode}")


class GPSReplay:
    """Causal replay: chỉ trả measurement có timestamp <= current_time."""

    def __init__(
        self, measurements: Iterable[GPSMeasurement], metadata: dict | None = None
    ) -> None:
        self.measurements = sorted(measurements, key=lambda item: item.timestamp)
        self.metadata = metadata or {}
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def seek(self, timestamp: float) -> None:
        """Bỏ measurement cũ hơn timestamp, không replay lịch sử vào frame đầu."""
        while (
            self._index < len(self.measurements)
            and self.measurements[self._index].timestamp < timestamp
        ):
            self._index += 1

    def pop_until(self, current_time: float) -> list[GPSMeasurement]:
        ready = []
        while (
            self._index < len(self.measurements)
            and self.measurements[self._index].timestamp <= current_time
        ):
            ready.append(self.measurements[self._index])
            self._index += 1
        return ready


def _epoch_from_seconds_of_day(seconds_of_day: float, reference_epoch: float) -> float:
    day = np.floor(reference_epoch / 86400.0) * 86400.0
    timestamp = day + seconds_of_day
    if timestamp - reference_epoch > 43200:
        timestamp -= 86400
    elif reference_epoch - timestamp > 43200:
        timestamp += 86400
    return float(timestamp)


def geodetic_to_ecef(
    latitude_deg: np.ndarray, longitude_deg: np.ndarray, altitude_m: np.ndarray
) -> np.ndarray:
    """WGS84 geodetic -> ECEF, vectorized."""
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    altitude = np.asarray(altitude_m, dtype=np.float64)
    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    normal = semi_major / np.sqrt(1 - eccentricity_sq * np.sin(latitude) ** 2)
    x = (normal + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (normal + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (normal * (1 - eccentricity_sq) + altitude) * np.sin(latitude)
    return np.column_stack((x, y, z))


def geodetic_to_enu(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    datum_latitude_deg: float,
    datum_longitude_deg: float,
) -> np.ndarray:
    """WGS84 lat/lon -> local ENU; GGA altitude không cần cho benchmark 2D."""
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    ecef = geodetic_to_ecef(latitude, longitude, np.zeros_like(latitude))
    datum = geodetic_to_ecef(
        np.array([datum_latitude_deg]),
        np.array([datum_longitude_deg]),
        np.array([0.0]),
    )[0]
    delta = ecef - datum
    lat0 = np.deg2rad(datum_latitude_deg)
    lon0 = np.deg2rad(datum_longitude_deg)
    rotation = np.array(
        [
            [-np.sin(lon0), np.cos(lon0), 0],
            [
                -np.sin(lat0) * np.cos(lon0),
                -np.sin(lat0) * np.sin(lon0),
                np.cos(lat0),
            ],
            [
                np.cos(lat0) * np.cos(lon0),
                np.cos(lat0) * np.sin(lon0),
                np.sin(lat0),
            ],
        ],
        dtype=np.float64,
    )
    return delta @ rotation.T


def rigid_align_2d(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit rotation + translation target ~= R @ source + t, không fit scale."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source và target phải cùng shape [N,2]")
    if len(source) < 3:
        raise ValueError("Cần ít nhất 3 điểm để fit rigid alignment")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def quaternion_camera_heading(quaternions: np.ndarray) -> np.ndarray:
    """Heading của camera +Z (trục nhìn) chiếu lên mặt phẳng world XY."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    x, y, z, w = quaternions.T
    forward_x = 2 * (x * z + y * w)
    forward_y = 2 * (y * z - x * w)
    return np.unwrap(np.arctan2(forward_y, forward_x))


def load_reference_trajectory(
    recording: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = Path(recording)
    poses = fourseasons.load_reference_poses(rec)
    scale = fourseasons.load_gnss_scale(rec)
    timestamps = poses[:, 0].astype(np.float64)
    xy = poses[:, 1:3].astype(np.float64) * scale
    heading = quaternion_camera_heading(poses[:, 4:8])
    return timestamps, xy, heading


def create_odometry_proxy(
    timestamps: np.ndarray,
    xy: np.ndarray,
    heading: np.ndarray,
    *,
    seed: int = 7,
    distance_scale_bias: float = 0.015,
    yaw_bias_per_meter: float = 2e-4,
    translation_noise_std: float = 0.002,
    rotation_noise_std: float = 3e-4,
) -> list[OdometryMeasurement]:
    """Tạo relative odometry causal từ trajectory dùng làm test double."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    if timestamps.ndim != 1 or xy.shape != (len(timestamps), 2):
        raise ValueError("timestamps [N] và xy [N,2] không hợp lệ")
    if heading.shape != timestamps.shape or len(timestamps) < 2:
        raise ValueError("heading phải có shape [N] và cần ít nhất 2 pose")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps phải tăng nghiêm ngặt")

    rng = np.random.default_rng(seed)
    output = []
    for index in range(1, len(timestamps)):
        dt = float(timestamps[index] - timestamps[index - 1])
        world_delta = xy[index] - xy[index - 1]
        previous_heading = heading[index - 1]
        cosine, sine = np.cos(previous_heading), np.sin(previous_heading)
        vehicle_delta = np.array(
            [
                cosine * world_delta[0] + sine * world_delta[1],
                -sine * world_delta[0] + cosine * world_delta[1],
            ]
        )
        distance = float(np.linalg.norm(vehicle_delta))
        vehicle_delta *= 1.0 + distance_scale_bias
        vehicle_delta += rng.normal(0.0, translation_noise_std, size=2)
        dtheta = float(wrap_angle(heading[index] - heading[index - 1]))
        dtheta += yaw_bias_per_meter * distance
        dtheta += float(rng.normal(0.0, rotation_noise_std))
        output.append(
            OdometryMeasurement(
                timestamp=float(timestamps[index]),
                dt=dt,
                dx=float(vehicle_delta[0]),
                dy=float(vehicle_delta[1]),
                dtheta=float(wrap_angle(dtheta)),
                translation_std=translation_noise_std,
                rotation_std=rotation_noise_std,
            )
        )
    return output


def odometry_proxy_from_recording(
    recording: str | Path, **kwargs
) -> list[OdometryMeasurement]:
    timestamps, xy, heading = load_reference_trajectory(recording)
    return create_odometry_proxy(timestamps, xy, heading, **kwargs)


def load_nmea_replay(
    recording: str | Path, *, align_to_reference: bool = False
) -> GPSReplay:
    """Load GGA và tùy chọn rigid-align ENU vào reference frame cho benchmark."""
    rec = Path(recording)
    frame_epoch = float(fourseasons.load_times(rec)[0, 1])
    rows = [row for row in fourseasons.load_nmea_gga(rec) if row["utc"] is not None]
    valid = [row for row in rows if row["lat"] is not None and row["lon"] is not None]
    if not valid:
        raise ValueError(f"{rec} không có GGA position hợp lệ")
    datum_row = next((row for row in valid if row["fix_quality"] == 4), valid[0])
    latitudes = np.array([row["lat"] for row in valid])
    longitudes = np.array([row["lon"] for row in valid])
    enu = geodetic_to_enu(
        latitudes, longitudes, datum_row["lat"], datum_row["lon"]
    )[:, :2]
    valid_index = {id(row): index for index, row in enumerate(valid)}

    rotation = np.eye(2)
    translation = np.zeros(2)
    alignment = "raw_enu"
    alignment_error = None
    if align_to_reference:
        ref_timestamps, ref_xy, _ = load_reference_trajectory(rec)
        fit_rows = [
            row
            for row in valid
            if row["fix_quality"] == 4
            and ref_timestamps[0]
            <= _epoch_from_seconds_of_day(row["utc"], frame_epoch)
            <= ref_timestamps[-1]
        ]
        if len(fit_rows) < 3:
            raise ValueError("Không đủ RTK quality-4 overlap để align NMEA")
        fit_timestamps = np.array(
            [
                _epoch_from_seconds_of_day(row["utc"], frame_epoch)
                for row in fit_rows
            ]
        )
        fit_source = np.array([enu[valid_index[id(row)]] for row in fit_rows])
        fit_target = np.column_stack(
            (
                np.interp(fit_timestamps, ref_timestamps, ref_xy[:, 0]),
                np.interp(fit_timestamps, ref_timestamps, ref_xy[:, 1]),
            )
        )
        rotation, translation = rigid_align_2d(fit_source, fit_target)
        alignment = "reference_pose_rigid_2d_offline"
        residual = np.linalg.norm(
            fit_source @ rotation.T + translation - fit_target, axis=1
        )
        alignment_error = {
            "quality4_samples": len(fit_rows),
            "median_m": float(np.median(residual)),
            "p95_m": float(np.percentile(residual, 95)),
            "rmse_m": float(np.sqrt(np.mean(residual**2))),
        }

    measurements = []
    for row in rows:
        position = None
        if row["lat"] is not None and row["lon"] is not None:
            raw = enu[valid_index[id(row)]]
            position = rotation @ raw + translation
        measurements.append(
            GPSMeasurement(
                timestamp=_epoch_from_seconds_of_day(row["utc"], frame_epoch),
                x=None if position is None else float(position[0]),
                y=None if position is None else float(position[1]),
                fix_quality=row["fix_quality"],
                satellites=row["satellites"],
                hdop=row["hdop"],
                source="nmea_gga",
                mode="real",
            )
        )
    return GPSReplay(
        measurements,
        metadata={
            "alignment": alignment,
            "datum_latitude": datum_row["lat"],
            "datum_longitude": datum_row["lon"],
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "alignment_error": alignment_error,
        },
    )


def simulate_gps(
    timestamps: np.ndarray,
    xy: np.ndarray,
    *,
    rate_hz: float = 5.0,
    segments: Iterable[ScenarioSegment] = (),
    seed: int = 11,
) -> GPSReplay:
    """Sinh GPS có kiểm soát từ reference trajectory cho unit test/benchmark."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if rate_hz <= 0:
        raise ValueError("rate_hz phải > 0")
    if xy.shape != (len(timestamps), 2):
        raise ValueError("xy phải có shape [N,2]")
    segments = list(segments)
    rng = np.random.default_rng(seed)
    sample_times = np.arange(timestamps[0], timestamps[-1], 1.0 / rate_hz)
    truth = np.column_stack(
        (
            np.interp(sample_times, timestamps, xy[:, 0]),
            np.interp(sample_times, timestamps, xy[:, 1]),
        )
    )
    measurements = []
    mode_counts: dict[str, int] = {}
    for timestamp, position in zip(sample_times, truth):
        mode = "good"
        for segment in segments:
            if segment.start <= timestamp < segment.end:
                mode = segment.mode
                break
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if mode == "lost":
            continue
        if mode == "good":
            quality, satellites, hdop, sigma = 4, 12, 0.8, 0.08
        elif mode == "degraded":
            quality, satellites, hdop, sigma = 1, 3, 7.0, 4.0
        else:
            quality, satellites, hdop, sigma = 1, 8, 1.5, 1.0
        noisy = position + rng.normal(0.0, sigma, size=2)
        if mode == "degraded" and mode_counts[mode] % 10 == 0:
            noisy += np.array([15.0, -12.0])
        measurements.append(
            GPSMeasurement(
                timestamp=float(timestamp),
                x=float(noisy[0]),
                y=float(noisy[1]),
                fix_quality=quality,
                satellites=satellites,
                hdop=hdop,
                source="simulated_reference_pose",
                mode=mode,
            )
        )
    return GPSReplay(
        measurements,
        metadata={
            "alignment": "reference_frame",
            "seed": seed,
            "rate_hz": rate_hz,
            "segments": [segment.__dict__ for segment in segments],
        },
    )
