"""Loader cho 4Seasons dataset (TUM CVG).

Cấu trúc một recording:
    times.txt        frame_id(ns)  timestamp(s)  exposure(ms)
    result.txt       timestamp(s)  tx ty tz qx qy qz qw   (VIO reference pose, chưa nhân scale)
    GNSSPoses.txt    frame_id(ns), tx, ty, tz, qx, qy, qz, qw, scale, fusion_quality[, v3]
    imu.txt          timestamp(ns) gx gy gz ax ay az
    septentrio.nmea  NMEA thật từ receiver (GGA: fix quality, số vệ tinh, HDOP)
    undistorted_images/cam0|cam1/<frame_id>.png   (800x400, mono)
    Transformations.txt  chuỗi transform + "GNSS scale" (nhân vào translation để ra mét)

Calibration (thư mục calibration/):
    undistorted_calib_0.txt   "Pinhole fx fy cx cy 0" + kích thước — cho ảnh undistorted
    undistorted_calib_stereo.txt  transform cam0->cam1 (baseline)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_times(rec: Path) -> np.ndarray:
    """[N,2]: frame_id(ns), timestamp(s)."""
    raw = np.loadtxt(rec / "times.txt", usecols=(0, 1))
    return raw


def load_reference_poses(rec: Path) -> np.ndarray:
    """[N,8]: timestamp(s), tx, ty, tz, qx, qy, qz, qw. Translation chưa nhân GNSS scale."""
    return np.loadtxt(rec / "result.txt")


def load_gnss_scale(rec: Path) -> float:
    text = (rec / "Transformations.txt").read_text()
    return float(text.split("GNSS scale")[1].split()[0].replace(",", ""))


def load_imu(rec: Path) -> np.ndarray:
    """[M,7]: timestamp(ns), gyro xyz (rad/s), accel xyz (m/s^2)."""
    return np.loadtxt(rec / "imu.txt")


def load_nmea_gga(rec: Path) -> list[dict]:
    """GGA thật từ receiver: utc(s trong ngày), lat, lon, fix_quality, satellites, hdop.

    Message không có fix (quality 0) vẫn được trả về — đó chính là tín hiệu
    mất GPS mà GPS integrity monitor cần thấy.
    """
    out = []
    for line in (rec / "septentrio.nmea").read_text(errors="ignore").splitlines():
        if "GGA" not in line:
            continue
        p = line.split(",")
        if len(p) < 9 or not p[6]:
            continue
        utc = None
        if p[1]:
            utc = int(p[1][:2]) * 3600 + int(p[1][2:4]) * 60 + float(p[1][4:])
        lat = lon = None
        if p[2] and p[4]:
            lat = int(p[2][:2]) + float(p[2][2:]) / 60
            if p[3] == "S":
                lat = -lat
            lon = int(p[4][:3]) + float(p[4][3:]) / 60
            if p[5] == "W":
                lon = -lon
        out.append({
            "utc": utc,
            "lat": lat,
            "lon": lon,
            "fix_quality": int(p[6]),
            "satellites": int(p[7]) if p[7] else 0,
            "hdop": float(p[8]) if p[8] else None,
        })
    return out


def load_calibration(calib_dir: Path) -> dict:
    """Intrinsics pinhole cho ảnh undistorted + baseline stereo (mét)."""
    intr = {}
    for cam in (0, 1):
        parts = (calib_dir / f"undistorted_calib_{cam}.txt").read_text().split()
        # "Pinhole fx fy cx cy 0" theo convention DSO: giá trị chuẩn hoá nếu <=1
        fx, fy, cx, cy = map(float, parts[1:5])
        intr[f"cam{cam}"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    # ma trận 4x4 cam1 <- cam0; baseline = |translation|
    T = np.loadtxt(calib_dir / "undistorted_calib_stereo.txt")
    assert T.shape == (4, 4)
    intr["T_cam1_cam0"] = T
    intr["baseline_m"] = float(np.linalg.norm(T[:3, 3]))
    return intr


def frame_paths(rec: Path, cam: str = "cam0") -> list[Path]:
    return sorted((rec / "undistorted_images" / cam).glob("*.png"))


def load_recording(rec: str | Path) -> dict:
    """Gói toàn bộ metadata một recording (không load ảnh)."""
    rec = Path(rec)
    times = load_times(rec)
    poses = load_reference_poses(rec)
    scale = load_gnss_scale(rec)
    return {
        "path": rec,
        "times": times,
        "poses": poses,          # translation * scale = mét
        "gnss_scale": scale,
        "frames_cam0": frame_paths(rec, "cam0"),
        "frames_cam1": frame_paths(rec, "cam1"),
    }
