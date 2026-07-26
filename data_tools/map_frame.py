"""Common map frame cho Bước 3: ENU 3D neo bằng RTK quality-4, datum cố định.

Bước 1–2 chạy mỗi recording trong frame riêng (ENU với datum của chính nó, hoặc
reference frame của chính nó). Landmark DB cần **một** frame dùng chung cho cả
mapping traversal lẫn query traversal, nếu không map dựng ở traversal này vô
nghĩa với traversal kia.

Hai đường tách bạch:

- *Production*: fusion nhận GPS trong ENU với datum chung
  (``gps_sources.load_nmea_replay(..., datum=...)``). Không đọc reference pose.
- *Evaluation*: reference pose của mỗi recording được fit sang chính ENU đó,
  bằng chính các fix quality-4 **của recording đó**. Fit chỉ dùng calibration
  split; holdout chỉ để báo cáo. Toàn bộ module này là evaluation-side.

Fit là rigid 3D (Kabsch, không scale — translation reference đã nhân GNSS scale).
Chiều fit là ``reference -> ENU``; đừng nhầm với vòng 3–5 fit ``NMEA -> reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_tools import fourseasons
from data_tools.gps_sources import (
    _epoch_from_seconds_of_day,
    geodetic_to_ecef,
    quaternion_camera_heading,
    wrap_angle,
)

CALIBRATION_SPLIT = 0.6


@dataclass(frozen=True)
class Datum:
    latitude: float
    longitude: float
    altitude: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)


@dataclass(frozen=True)
class MapFrameFit:
    rotation: np.ndarray
    translation: np.ndarray
    datum: Datum
    calibration_samples: int
    holdout_samples: int
    calibration_residual: dict
    holdout_residual: dict

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points phải có shape [N,3]")
        return points @ self.rotation.T + self.translation

    @property
    def yaw(self) -> float:
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))

    def metadata(self) -> dict:
        return {
            "direction": "reference_xyz -> common_enu",
            "fit": "rigid_3d_kabsch_no_scale",
            "datum": {
                "latitude": self.datum.latitude,
                "longitude": self.datum.longitude,
                "altitude_m": self.datum.altitude,
            },
            "split": "chronological",
            "calibration_fraction": CALIBRATION_SPLIT,
            "calibration_samples": self.calibration_samples,
            "holdout_samples": self.holdout_samples,
            "calibration_residual": self.calibration_residual,
            "holdout_residual": self.holdout_residual,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "reference_usage": "evaluation only",
        }


def quality4_rows(recording: Path) -> list[dict]:
    return [
        row
        for row in fourseasons.load_nmea_gga(recording)
        if row["utc"] is not None
        and row["lat"] is not None
        and row["fix_quality"] == 4
    ]


def first_quality4_datum(recording: str | Path) -> Datum:
    rows = quality4_rows(Path(recording))
    if not rows:
        raise ValueError(f"{recording} không có fix quality-4 để làm datum")
    row = rows[0]
    altitude = row["ellipsoid_altitude_m"]
    return Datum(row["lat"], row["lon"], 0.0 if altitude is None else altitude)


def geodetic_to_enu_3d(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    altitude_m: np.ndarray,
    datum: Datum,
) -> np.ndarray:
    """WGS84 -> ENU 3D quanh datum; giữ cao độ, khác ``geodetic_to_enu`` 2D."""
    ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
    origin = geodetic_to_ecef(
        np.array([datum.latitude]),
        np.array([datum.longitude]),
        np.array([datum.altitude]),
    )[0]
    latitude = np.deg2rad(datum.latitude)
    longitude = np.deg2rad(datum.longitude)
    rotation = np.array(
        [
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [
                -np.sin(latitude) * np.cos(longitude),
                -np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            ],
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
        ],
        dtype=np.float64,
    )
    return (ecef - origin) @ rotation.T


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 3D không scale: target ≈ R @ source + t."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source và target phải cùng shape [N,3]")
    if len(source) < 3:
        raise ValueError("Cần ít nhất 3 điểm để fit rigid 3D")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return rotation, target_center - rotation @ source_center


def _residual_stats(residual: np.ndarray) -> dict:
    if not len(residual):
        return {"count": 0}
    return {
        "count": int(len(residual)),
        "median_m": float(np.median(residual)),
        "p95_m": float(np.percentile(residual, 95)),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
    }


def reference_xyz(recording: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """timestamps, XYZ mét (đã nhân GNSS scale), heading camera trong frame reference."""
    poses = fourseasons.load_reference_poses(recording)
    scale = fourseasons.load_gnss_scale(recording)
    return (
        poses[:, 0].astype(np.float64),
        poses[:, 1:4].astype(np.float64) * scale,
        quaternion_camera_heading(poses[:, 4:8]),
    )


def fit_reference_to_map(
    recording: str | Path,
    datum: Datum,
    *,
    calibration_fraction: float = CALIBRATION_SPLIT,
) -> MapFrameFit:
    """Fit reference XYZ -> common ENU bằng quality-4 của chính recording đó."""
    recording = Path(recording)
    rows = quality4_rows(recording)
    if len(rows) < 6:
        raise ValueError(f"{recording} không đủ quality-4 để fit map frame")
    epoch = float(fourseasons.load_times(recording)[0, 1])
    timestamps, xyz, _ = reference_xyz(recording)
    fix_times = np.array(
        [_epoch_from_seconds_of_day(row["utc"], epoch) for row in rows]
    )
    inside = (fix_times >= timestamps[0]) & (fix_times <= timestamps[-1])
    rows = [row for row, keep in zip(rows, inside) if keep]
    fix_times = fix_times[inside]
    if len(rows) < 6:
        raise ValueError(f"{recording} không đủ quality-4 nằm trong reference")
    target = geodetic_to_enu_3d(
        np.array([row["lat"] for row in rows]),
        np.array([row["lon"] for row in rows]),
        np.array(
            [
                0.0 if row["ellipsoid_altitude_m"] is None else row["ellipsoid_altitude_m"]
                for row in rows
            ]
        ),
        datum,
    )
    source = np.column_stack(
        [np.interp(fix_times, timestamps, xyz[:, axis]) for axis in range(3)]
    )
    split = int(len(fix_times) * calibration_fraction)
    if split < 3:
        raise ValueError("Calibration split quá nhỏ để fit")
    rotation, translation = kabsch(source[:split], target[:split])
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return MapFrameFit(
        rotation=rotation,
        translation=translation,
        datum=datum,
        calibration_samples=split,
        holdout_samples=len(fix_times) - split,
        calibration_residual=_residual_stats(residual[:split]),
        holdout_residual=_residual_stats(residual[split:]),
    )


def reference_in_map_frame(
    recording: str | Path, fit: MapFrameFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """timestamps, XYZ trong map frame, heading đã xoay theo yaw của fit."""
    timestamps, xyz, heading = reference_xyz(Path(recording))
    return timestamps, fit.apply(xyz), wrap_angle(heading + fit.yaw)
