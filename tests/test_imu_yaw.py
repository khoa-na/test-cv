"""Tests cho gyro yaw fallback và stationary calibration gate."""

from __future__ import annotations

import numpy as np
import pytest

from data_tools.imu_yaw import IMUYawIntegrator, quaternion_to_rotation


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
