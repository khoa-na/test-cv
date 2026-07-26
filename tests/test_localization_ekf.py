"""Unit tests cho GPS integrity + local/global fusion Bước 1."""

from __future__ import annotations

import numpy as np
import pytest

from data_tools.gps_sources import (
    GPSMeasurement,
    GPSReplay,
    OdometryMeasurement,
    create_odometry_proxy,
)
from pipelines.localization_ekf import (
    GPSIntegrityMonitor,
    GPSState,
    LocalOdometryEKF,
    LocalizationFusion,
    UTurnDetector,
)


def gps(
    timestamp: float,
    x: float | None = 0.0,
    y: float | None = 0.0,
    *,
    quality: int = 4,
    satellites: int = 12,
    hdop: float | None = 0.8,
) -> GPSMeasurement:
    return GPSMeasurement(
        timestamp=timestamp,
        x=x,
        y=y,
        fix_quality=quality,
        satellites=satellites,
        hdop=hdop,
        source="test",
    )


def odom(
    timestamp: float,
    *,
    dt: float = 0.1,
    dx: float = 0.1,
    dy: float = 0.0,
    dtheta: float = 0.0,
) -> OdometryMeasurement:
    return OdometryMeasurement(
        timestamp=timestamp,
        dt=dt,
        dx=dx,
        dy=dy,
        dtheta=dtheta,
        translation_std=0.002,
        rotation_std=0.0003,
        source="test",
    )


def make_good(fusion: LocalizationFusion, start: float = 0.0) -> None:
    for index in range(5):
        fusion.process_gps(gps(start + index * 0.1))
    assert fusion.integrity.state == GPSState.GOOD


def test_predict_straight_and_covariance_grows():
    ekf = LocalOdometryEKF()
    initial_position_variance = ekf.covariance[0, 0]
    for index in range(100):
        ekf.step(odom((index + 1) * 0.1))
    assert ekf.pose[0] == pytest.approx(10.0, abs=0.08)
    assert ekf.pose[1] == pytest.approx(0.0, abs=1e-6)
    assert ekf.covariance[0, 0] > initial_position_variance


def test_gps_changes_global_target_not_local_and_nis_rejects_outlier():
    fusion = LocalizationFusion()
    fusion.process_odometry(odom(0.1))
    local_before = fusion.local_pose.copy()
    first = fusion.process_gps(gps(0.1, 10.0, -3.0))
    assert first["accepted"]
    np.testing.assert_allclose(fusion.local_pose, local_before)
    target_before = fusion.target_map_to_odom.copy()

    accepted = fusion.process_gps(gps(0.2, 11.0, -3.0))
    assert accepted["accepted"]
    assert not np.allclose(fusion.target_map_to_odom, target_before)
    np.testing.assert_allclose(fusion.local_pose, local_before)

    target_before = fusion.target_map_to_odom.copy()
    rejected = fusion.process_gps(gps(0.3, 60.0, -3.0))
    assert not rejected["accepted"]
    assert rejected["nis"] > fusion.config.normal_nis_threshold
    np.testing.assert_allclose(fusion.target_map_to_odom, target_before)


def test_direct_loss_timeout_and_degraded_recovery_hysteresis():
    direct = GPSIntegrityMonitor(initial_state=GPSState.GOOD)
    transition = direct.observe(gps(1.0, None, None, quality=0, satellites=0), False)
    assert transition is not None
    assert transition.current == GPSState.LOST

    timeout = GPSIntegrityMonitor(initial_state=GPSState.GOOD)
    timeout.observe(gps(1.0), True)
    assert timeout.tick(2.4) is None
    transition = timeout.tick(2.6)
    assert transition is not None
    assert transition.reason == "receiver_timeout"

    monitor = GPSIntegrityMonitor(initial_state=GPSState.GOOD)
    transition = monitor.observe(gps(1.0, satellites=3, hdop=5.1), False)
    assert transition is not None
    assert transition.current == GPSState.DEGRADED
    for index in range(4):
        monitor.observe(gps(1.1 + index * 0.1, hdop=1.2), True)
        assert monitor.state == GPSState.DEGRADED
    monitor.observe(gps(1.5, hdop=1.2), True)
    assert monitor.state == GPSState.GOOD


def test_recovering_rejects_transition_to_degraded_not_lost():
    monitor = GPSIntegrityMonitor()
    monitor.observe(gps(1.0), True)
    assert monitor.state == GPSState.RECOVERING
    for index in range(3):
        monitor.observe(gps(1.1 + index * 0.1), False)
    assert monitor.state == GPSState.DEGRADED


def test_latched_pose_is_immutable_during_error_episode():
    fusion = LocalizationFusion()
    make_good(fusion)
    fusion.process_odometry(odom(0.6))
    fusion.process_gps(gps(0.6, satellites=3, hdop=6.0))
    assert fusion.integrity.state == GPSState.DEGRADED
    latched = fusion.latched_global_pose.copy()
    latched_timestamp = fusion.latched_timestamp
    fusion.process_odometry(odom(0.7))
    fusion.process_gps(gps(0.7, None, None, quality=0, satellites=0))
    fusion.process_odometry(odom(0.8))
    np.testing.assert_allclose(fusion.latched_global_pose, latched)
    assert fusion.latched_timestamp == latched_timestamp


def test_recovery_correction_is_rate_limited_and_local_has_no_jump():
    fusion = LocalizationFusion()
    make_good(fusion)
    fusion.process_gps(gps(0.6, None, None, quality=0, satellites=0))
    for index in range(20):
        fusion.process_odometry(odom(0.7 + index * 0.1))
    before_global = fusion.global_pose.copy()
    before_local = fusion.local_pose.copy()
    recovery_position = before_global[:2] + np.array([4.0, 0.0])
    result = fusion.process_gps(
        gps(2.7, float(recovery_position[0]), float(recovery_position[1]), quality=1)
    )
    assert result["accepted"]
    fusion.process_odometry(odom(2.8))
    global_delta = fusion.global_pose[:2] - before_global[:2]
    local_delta = fusion.local_pose[:2] - before_local[:2]
    correction_discontinuity = np.linalg.norm(global_delta - local_delta)
    assert correction_discontinuity <= (
        fusion.config.max_translation_correction_mps * 0.1 + 1e-6
    )


def test_yaw_correction_rotates_about_current_pose_without_position_jump():
    fusion = LocalizationFusion()
    fusion.local.state[0] = 100.0
    fusion.target_map_to_odom[2] = 0.2
    global_before = fusion.global_pose.copy()
    local_before = fusion.local_pose.copy()
    fusion.process_odometry(odom(0.1, dx=0.0))
    correction = np.linalg.norm(fusion.global_pose[:2] - global_before[:2])
    assert correction <= fusion.config.max_translation_correction_mps * 0.1 + 1e-6
    np.testing.assert_allclose(fusion.local_pose[:2], local_before[:2], atol=1e-6)


def test_uturn_synthetic_and_straight_false_positive():
    detector = UTurnDetector()
    detection = None
    threshold_crossing = None
    for timestamp in np.arange(0.0, 5.01, 0.1):
        heading = np.pi * min(timestamp / 4.0, 1.0)
        if threshold_crossing is None and heading >= np.deg2rad(150):
            threshold_crossing = timestamp
        detection = detection or detector.update(timestamp, heading)
    assert detection is not None
    assert detection["timestamp"] - threshold_crossing <= 0.1

    straight = UTurnDetector()
    assert all(
        straight.update(timestamp, 0.01 * np.sin(timestamp)) is None
        for timestamp in np.arange(0.0, 20.0, 0.1)
    )


def test_odometry_proxy_is_reproducible_signed_and_wraps_angle():
    timestamps = np.array([0.0, 1.0, 2.0])
    xy = np.array([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    heading = np.deg2rad(np.array([179.0, -179.0, -178.0]))
    kwargs = {
        "seed": 3,
        "distance_scale_bias": 0.0,
        "yaw_bias_per_meter": 0.0,
        "translation_noise_std": 0.0,
        "rotation_noise_std": 0.0,
    }
    first = create_odometry_proxy(timestamps, xy, heading, **kwargs)
    second = create_odometry_proxy(timestamps, xy, heading, **kwargs)
    assert first == second
    assert first[0].v_meas > 0  # heading ~180°, world -X là chạy tiến
    assert first[0].dtheta == pytest.approx(np.deg2rad(2.0))

    reverse = create_odometry_proxy(
        timestamps,
        xy,
        np.zeros(3),
        **kwargs,
    )
    assert reverse[0].v_meas < 0


def test_recovery_accepts_reasonable_innovation_but_rejects_extreme_outlier():
    reasonable = LocalizationFusion()
    make_good(reasonable)
    reasonable.process_gps(gps(0.6, None, None, quality=0, satellites=0))
    result = reasonable.process_gps(gps(1.0, 4.0, 0.0, quality=1, hdop=1.0))
    assert result["accepted"]
    assert reasonable.integrity.state == GPSState.RECOVERING

    extreme = LocalizationFusion()
    make_good(extreme)
    extreme.process_gps(gps(0.6, None, None, quality=0, satellites=0))
    result = extreme.process_gps(gps(1.0, 50.0, 0.0, quality=1, hdop=1.0))
    assert not result["accepted"]


def test_gps_replay_is_causal():
    replay = GPSReplay([gps(1.0), gps(2.0), gps(3.0)])
    assert replay.pop_until(0.9) == []
    assert [item.timestamp for item in replay.pop_until(2.1)] == [1.0, 2.0]
    assert replay.pop_until(2.9) == []
    assert [item.timestamp for item in replay.pop_until(3.0)] == [3.0]
    replay.reset()
    replay.seek(2.0)
    assert [item.timestamp for item in replay.pop_until(2.0)] == [2.0]
