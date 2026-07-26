import math

import pytest

from ros2.localization_node import (
    NAVSAT_TO_GGA,
    FusionBridge,
    self_check,
    yaw_to_quaternion,
)


def test_no_fix_status_never_produces_a_position():
    bridge = FusionBridge()
    bridge.on_gps(1000.0, 49.0, 11.0, 400.0, -1)
    assert NAVSAT_TO_GGA[-1] == 0
    # Datum chỉ được chốt bởi fix hợp lệ, không phải bởi bản tin NO_FIX.
    assert bridge.snapshot()["datum"] is None


def test_datum_latches_on_first_valid_fix_and_stays():
    bridge = FusionBridge()
    bridge.on_gps(1000.0, 49.0, 11.0, 400.0, 2)
    bridge.on_gps(1001.0, 49.1, 11.1, 410.0, 2)
    datum = bridge.snapshot()["datum"]
    assert datum["latitude"] == pytest.approx(49.0)
    assert datum["source"] == "first_valid_fix"


def test_dropout_advances_clock_without_fabricating_motion():
    bridge = FusionBridge()
    bridge.on_gps(1000.0, 49.0, 11.0, 400.0, 2)
    bridge.on_odometry(1000.1, 0.5, 0.0, 0.0, 0.1)
    moved = bridge.snapshot()["odom"]["x"]
    bridge.on_odometry_dropout(1000.2, 0.1)
    after = bridge.snapshot()
    # predict_only giữ vận tốc nên vị trí tiến, nhưng không có measurement mới.
    assert after["odom"]["x"] >= moved
    assert after["timestamp"] == pytest.approx(1000.2)


def test_covariance_row_major_places_xy_and_yaw_correctly():
    bridge = FusionBridge()
    bridge.on_gps(1000.0, 49.0, 11.0, 400.0, 2)
    bridge.on_odometry(1000.1, 0.5, 0.0, 0.0, 0.1)
    flat = bridge.odometry_covariance_row_major()
    matrix = bridge.snapshot()["position_covariance"]
    assert len(flat) == 36
    assert flat[0] == pytest.approx(matrix[0][0])
    assert flat[7] == pytest.approx(matrix[1][1])
    assert flat[14] == 1e6 and flat[21] == 1e6 and flat[28] == 1e6
    assert flat[35] > 0.0


def test_zero_dt_odometry_is_rejected():
    bridge = FusionBridge()
    with pytest.raises(ValueError):
        bridge.on_odometry(1000.0, 0.5, 0.0, 0.0, 0.0)


def test_yaw_to_quaternion_round_trips():
    for yaw in (0.0, 0.5, -1.2, math.pi / 2):
        _, _, z, w = yaw_to_quaternion(yaw)
        assert 2.0 * math.atan2(z, w) == pytest.approx(yaw)


def test_self_check_reports_gps_loss():
    assert self_check()["gps_state"] in {"DEGRADED", "LOST", "RECOVERING"}
