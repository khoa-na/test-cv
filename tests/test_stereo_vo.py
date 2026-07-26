"""Geometry và dropout contract cho stereo visual odometry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from data_tools.gps_sources import OdometryMeasurement
from pipelines.localization_ekf import GPSState, LocalizationFusion
from pipelines.stereo_vo import StereoVO, StereoVOConfig


CALIBRATION = {
    "cam0": {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 400.0,
        "cy": 200.0,
    },
    "cam1": {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 400.0,
        "cy": 200.0,
    },
    "baseline_m": 0.3,
    "T_cam1_cam0": np.array(
        [
            [1.0, 0.0, 0.0, -0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def project(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = points @ rotation.T + translation
    pixels = np.empty((len(points), 2), dtype=np.float64)
    pixels[:, 0] = 500.0 * transformed[:, 0] / transformed[:, 2] + 400.0
    pixels[:, 1] = 500.0 * transformed[:, 1] / transformed[:, 2] + 200.0
    return pixels


def synthetic_points(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        (
            rng.uniform(-3.0, 3.0, 100),
            rng.uniform(-1.5, 1.5, 100),
            rng.uniform(6.0, 20.0, 100),
        )
    )


def textured_stereo(seed: int = 9, disparity: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, size=(400, 800), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0)
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]
    return left, right


def test_depth_and_disparity_bounds_come_from_calibration():
    vo = StereoVO(CALIBRATION)
    assert StereoVO.depth_from_disparity(15.0, 500.0, 0.3) == pytest.approx(
        10.0
    )
    assert vo.min_disparity_px == pytest.approx(2.5)
    assert vo.max_disparity_px == pytest.approx(150.0)


def test_pnp_is_inverted_to_positive_forward_ego_motion():
    vo = StereoVO(CALIBRATION)
    points = synthetic_points()
    image_points = project(
        points,
        np.eye(3),
        np.array([0.0, 0.0, -1.0]),
    )
    estimate = vo._estimate_motion(points, image_points)
    assert estimate is not None
    dx, dy, dtheta = vo.camera_motion_to_se2(
        estimate.rotation,
        estimate.translation,
    )
    assert dx == pytest.approx(1.0, abs=1e-5)
    assert dy == pytest.approx(0.0, abs=1e-5)
    assert dtheta == pytest.approx(0.0, abs=1e-5)


def test_pure_left_rotation_has_positive_vehicle_heading():
    vo = StereoVO(CALIBRATION)
    points = synthetic_points()
    expected_heading = np.deg2rad(12.0)
    ego_rotation = rotation_y(-expected_heading)
    pnp_rotation = ego_rotation.T
    image_points = project(points, pnp_rotation, np.zeros(3))
    estimate = vo._estimate_motion(points, image_points)
    assert estimate is not None
    dx, dy, dtheta = vo.camera_motion_to_se2(
        estimate.rotation,
        estimate.translation,
    )
    assert dx == pytest.approx(0.0, abs=1e-5)
    assert dy == pytest.approx(0.0, abs=1e-5)
    assert dtheta == pytest.approx(expected_heading, abs=1e-5)


def test_pitch_does_not_leak_into_se2_heading():
    pitch = np.deg2rad(10.0)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    dx, dy, dtheta = StereoVO.camera_motion_to_se2(
        rotation_x,
        np.array([0.0, 0.2, 1.0]),
    )
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(0.0)
    assert dtheta == pytest.approx(0.0)


def test_blank_frames_return_none_without_throwing():
    vo = StereoVO(CALIBRATION)
    blank = np.zeros((400, 800), dtype=np.uint8)
    assert vo.process(blank, blank, 1.0) is None
    assert vo.process(blank, blank, 1.033) is None


def test_pnp_ransac_is_deterministic():
    vo = StereoVO(CALIBRATION)
    points = synthetic_points()
    image_points = project(
        points,
        np.eye(3),
        np.array([0.1, -0.05, -0.5]),
    )
    first = vo._estimate_motion(points, image_points)
    second = vo._estimate_motion(points, image_points)
    assert first is not None and second is not None
    np.testing.assert_allclose(first.rotation, second.rotation)
    np.testing.assert_allclose(first.translation, second.translation)
    np.testing.assert_array_equal(first.inlier_indices, second.inlier_indices)


def test_temporal_matching_recovers_after_one_featureless_frame():
    config = StereoVOConfig(
        min_pnp_points=8,
        forward_backward_interval=0,
    )
    vo = StereoVO(CALIBRATION, config)
    left, right = textured_stereo()
    blank = np.zeros_like(left)
    assert vo.process(left, right, 1.0) is None
    assert vo.process(blank, blank, 1.033) is None
    assert vo.process(left, right, 1.066) is None
    recovered = vo.process(left, right, 1.099)
    assert recovered is not None
    assert recovered.dt == pytest.approx(0.033)
    assert abs(recovered.dx) < 1e-3
    assert abs(recovered.dy) < 1e-3
    assert abs(recovered.dtheta) < 1e-3


def test_predict_only_advances_without_zero_velocity_measurement():
    fusion = LocalizationFusion()
    measurement = OdometryMeasurement(
        timestamp=0.1,
        dt=0.1,
        dx=0.1,
        dy=0.0,
        dtheta=0.0,
        translation_std=0.002,
        rotation_std=0.0003,
        source="test",
    )
    fusion.process_odometry(measurement)
    velocity_before = float(fusion.local.state[3])
    covariance_before = fusion.local.covariance.copy()
    pose_before = fusion.local_pose.copy()

    pose = fusion.predict_only(
        timestamp=2.0,
        dt=0.1,
        translation_process_std=0.05,
        rotation_process_std=0.01,
    )

    assert fusion.local.state[3] == pytest.approx(velocity_before)
    assert fusion.local_pose[0] > pose_before[0]
    assert np.trace(fusion.local.covariance) > np.trace(covariance_before)
    assert fusion.pose_log[-1]["timestamp"] == 2.0
    assert fusion.integrity.state == GPSState.LOST
    np.testing.assert_allclose(pose, fusion.global_pose)
