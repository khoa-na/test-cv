"""Smoke test loader 4Seasons trên recording đã tải (skip nếu chưa có data)."""

from pathlib import Path

import numpy as np
import pytest

from data_tools import fourseasons as fs

ROOT = Path(".cache/data/4seasons")
REC = ROOT / "recording_2021-02-25_13-39-06"  # parking_garage_2_train


pytestmark = pytest.mark.skipif(not REC.exists(), reason="4Seasons chưa tải")


def test_times_and_frames_align():
    times = fs.load_times(REC)
    frames = fs.frame_paths(REC)
    assert len(times) == len(frames)
    # frame_id trong times khớp tên file ảnh
    assert int(times[0, 0]) == int(frames[0].stem)


def test_reference_poses_metric_length():
    rec = fs.load_recording(REC)
    xyz = rec["poses"][:, 1:4] * rec["gnss_scale"]
    length = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    # garage_2 đo được ~852 m
    assert 700 < length < 1000


def test_nmea_has_real_gps_loss():
    gga = fs.load_nmea_gga(REC)
    quals = {g["fix_quality"] for g in gga}
    assert 0 in quals, "garage phải có message mất fix"
    assert any(q >= 4 for q in quals), "ngoài trời phải có RTK fix"
    positioned = next(row for row in gga if row["fix_quality"] == 4)
    assert positioned["altitude_msl_m"] is not None
    assert positioned["geoid_separation_m"] is not None
    assert positioned["ellipsoid_altitude_m"] == pytest.approx(
        positioned["altitude_msl_m"] + positioned["geoid_separation_m"]
    )


def test_transformations_match_dataset_convention():
    transforms = fs.load_transformations(REC)
    assert transforms["gnss_scale"] == pytest.approx(1.042780)
    for name in (
        "transform_s_as",
        "transform_cam_imu",
        "transform_w_gpsw",
        "transform_gps_imu",
        "transform_e_gpsw",
    ):
        transform = transforms[name]
        assert transform.shape == (4, 4)
        assert np.allclose(
            transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-6
        )
    assert np.linalg.norm(
        transforms["transform_gps_imu"][:3, 3]
    ) == pytest.approx(0.0)


def test_calibration():
    calib = fs.load_calibration(ROOT / "calibration")
    assert calib["cam0"]["fx"] > 100
    assert 0.2 < calib["baseline_m"] < 0.5


def test_imu_rate():
    imu = fs.load_imu(REC)
    dt = np.median(np.diff(imu[:2000, 0])) / 1e9
    assert 1e-4 < dt < 1e-3  # ~2 kHz
