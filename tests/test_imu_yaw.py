"""Tests cho gyro yaw fallback và stationary calibration gate."""

from __future__ import annotations

import numpy as np
import pytest

from data_tools.imu_yaw import (
    IMUYawConfig,
    IMUYawIntegrator,
    quaternion_to_rotation,
)


def moving_start_gyro() -> tuple[np.ndarray, np.ndarray]:
    """Xe đang quay suốt 3 s đầu rồi mới yên: hình dạng của garage_3."""
    timestamps = np.linspace(0.0, 12.0, 1201)
    gyro = np.zeros((len(timestamps), 3))
    gyro[:, 1] = 0.005
    gyro[timestamps < 3.0, 1] += 0.1
    return timestamps, gyro


def test_quaternion_identity_rotation():
    np.testing.assert_allclose(
        quaternion_to_rotation(np.array([0.0, 0.0, 0.0, 1.0])),
        np.eye(3),
        atol=1e-12,
    )


def test_stationary_bias_is_removed_before_yaw_integration():
    timestamps = np.linspace(0.0, 4.0, 401)
    gyro = np.zeros((len(timestamps), 3))
    gyro[:, 1] = 0.005
    gyro[timestamps > 2.0, 1] -= 0.1
    integrator = IMUYawIntegrator(
        timestamps,
        gyro,
        np.eye(3),
        calibration_start=0.0,
    )
    assert integrator.stationary_calibration_passed
    assert integrator.bias_camera_y_rad_s == pytest.approx(0.005)
    assert integrator.metadata()["low_angular_rate_gate_passed"]
    assert integrator.delta_heading(2.0, 3.0) == pytest.approx(
        0.1,
        abs=1e-3,
    )


def test_moving_calibration_window_disables_imu_override():
    timestamps = np.linspace(0.0, 4.0, 401)
    gyro = np.zeros((len(timestamps), 3))
    gyro[:, 1] = 0.03
    integrator = IMUYawIntegrator(
        timestamps,
        gyro,
        np.eye(3),
        calibration_start=0.0,
    )
    assert not integrator.stationary_calibration_passed
    assert integrator.delta_heading(2.0, 3.0) is None


def test_recording_start_mode_absorbs_real_motion_into_the_bias():
    timestamps, gyro = moving_start_gyro()
    integrator = IMUYawIntegrator(
        timestamps, gyro, np.eye(3), calibration_start=0.0
    )
    # Bias thật là 0.005; cửa sổ đầu nuốt luôn 0.1 rad/s xe đang quay.
    assert integrator.bias_camera_y_rad_s == pytest.approx(0.105)
    assert not integrator.stationary_calibration_passed


def test_quietest_window_mode_recovers_the_bias_from_later_in_the_recording():
    timestamps, gyro = moving_start_gyro()
    integrator = IMUYawIntegrator(
        timestamps,
        gyro,
        np.eye(3),
        calibration_start=0.0,
        config=IMUYawConfig(bias_calibration_mode="quietest_window"),
    )
    assert integrator.bias_camera_y_rad_s == pytest.approx(0.005)
    assert integrator.stationary_calibration_passed
    assert integrator.calibration_start >= 3.0
    assert integrator.metadata()["calibration_mode"] == "quietest_window"


def test_quietest_window_still_obeys_the_gate_when_nothing_is_quiet():
    # garage_3: 0/172 cửa sổ đạt ngưỡng. Bias trục y sạch không cứu được, vì
    # cổng chấm trên gyro norm và trục x ồn suốt băng.
    timestamps = np.linspace(0.0, 12.0, 1201)
    gyro = np.zeros((len(timestamps), 3))
    gyro[:, 0] = 0.03
    gyro[:, 1] = 0.005
    integrator = IMUYawIntegrator(
        timestamps,
        gyro,
        np.eye(3),
        calibration_start=0.0,
        config=IMUYawConfig(bias_calibration_mode="quietest_window"),
    )
    assert integrator.bias_camera_y_rad_s == pytest.approx(0.005)
    assert not integrator.stationary_calibration_passed
    assert integrator.delta_heading(5.0, 6.0) is None


def test_unknown_calibration_mode_is_rejected():
    timestamps, gyro = moving_start_gyro()
    with pytest.raises(ValueError, match="bias_calibration_mode"):
        IMUYawIntegrator(
            timestamps,
            gyro,
            np.eye(3),
            calibration_start=0.0,
            config=IMUYawConfig(bias_calibration_mode="whatever"),
        )
