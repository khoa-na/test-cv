"""Test landmark DB + association."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipelines.landmark_db import (
    LandmarkConfig,
    LandmarkDatabase,
    LandmarkEntry,
    LandmarkMatcher,
    build_database,
    global_descriptor,
    select_keyframes,
)
from pipelines.stereo_vo import StereoVO

CALIBRATION = {
    "cam0": {"fx": 400.0, "fy": 400.0, "cx": 400.0, "cy": 200.0},
    "baseline_m": 0.3,
}


def textured_image(seed: int, size: tuple[int, int] = (400, 800)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, size=size, dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


# --- 1. Descriptor -------------------------------------------------------


def test_descriptor_is_deterministic_and_brightness_invariant():
    image = textured_image(0)
    first = global_descriptor(image)
    second = global_descriptor(image.copy())
    brighter = np.clip(image.astype(np.float32) * 0.6 + 40, 0, 255).astype(np.uint8)

    assert np.array_equal(first, second)
    assert float(first @ global_descriptor(brighter)) > 0.95
    assert float(first @ global_descriptor(textured_image(1))) < 0.5


def test_descriptor_is_l2_normalized():
    assert float(np.linalg.norm(global_descriptor(textured_image(2)))) == pytest.approx(
        1.0, abs=1e-5
    )


# --- 2. Retrieval --------------------------------------------------------


def make_entry(entry_id: int, image: np.ndarray, position, heading=0.0) -> LandmarkEntry:
    return LandmarkEntry(
        id=entry_id,
        landmark_class="keyframe_orb",
        position=np.array([position[0], position[1], 0.0]),
        heading=heading,
        descriptor=global_descriptor(image).astype(np.float32),
        keypoints=np.zeros((0, 2), np.float32),
        points_3d=np.zeros((0, 3), np.float32),
        orb_desc=np.zeros((0, 32), np.uint8),
        t_first=float(entry_id),
        t_last=float(entry_id),
        n_obs=1,
        covariance=np.eye(2) * 0.25,
    )


def test_retrieval_ranks_the_matching_place_first():
    images = [textured_image(seed) for seed in range(6)]
    database = LandmarkDatabase(
        [make_entry(index, image, (index * 3.0, 0.0)) for index, image in enumerate(images)]
    )
    matcher = LandmarkMatcher(database, CALIBRATION)
    noisy = np.clip(
        images[3].astype(np.int16) + np.random.default_rng(9).integers(-8, 8, images[3].shape),
        0,
        255,
    ).astype(np.uint8)

    result = matcher.query(noisy, noisy)

    assert result.ranked_entry_ids[0] == 3


# --- 3. Sequence consistency --------------------------------------------


def test_sequence_scoring_prefers_in_order_candidate():
    images = [textured_image(seed) for seed in range(8)]
    database = LandmarkDatabase(
        [make_entry(index, image, (index * 3.0, 0.0)) for index, image in enumerate(images)]
    )
    matcher = LandmarkMatcher(database, CALIBRATION)
    blank = np.zeros((400, 800), np.uint8)

    # Đi qua 4,5 rồi tới 6: candidate 6 được sequence ủng hộ.
    for index in (4, 5):
        matcher.query(images[index], blank)
    scores = matcher._sequence_scores(database.descriptors @ global_descriptor(images[6]))
    ranked = matcher.query(images[6], blank).ranked_entry_ids

    assert ranked[0] == 6
    assert scores[6] > scores[2]


# --- 4. Geometric verification ------------------------------------------


def synthetic_scene(rng: np.random.Generator, count: int = 120) -> np.ndarray:
    return np.column_stack(
        (
            rng.uniform(-6.0, 6.0, count),
            rng.uniform(-2.0, 2.0, count),
            rng.uniform(6.0, 20.0, count),
        )
    ).astype(np.float32)


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    camera = points @ rotation.T + translation
    fx, fy = CALIBRATION["cam0"]["fx"], CALIBRATION["cam0"]["fy"]
    cx, cy = CALIBRATION["cam0"]["cx"], CALIBRATION["cam0"]["cy"]
    return np.column_stack(
        (cx + fx * camera[:, 0] / camera[:, 2], cy + fy * camera[:, 1] / camera[:, 2])
    ).astype(np.float32)


def test_verification_recovers_known_base_pose():
    rng = np.random.default_rng(3)
    points = synthetic_scene(rng)
    descriptors = rng.integers(0, 255, size=(len(points), 32), dtype=np.uint8)
    # Camera query tiến 1.5 m và xoay trái 8 độ so với DB keyframe.
    forward, yaw = 1.5, np.deg2rad(8.0)
    rotation_motion = np.array(
        [
            [np.cos(-yaw), 0.0, np.sin(-yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(-yaw), 0.0, np.cos(-yaw)],
        ]
    )
    translation_motion = np.array([0.0, 0.0, forward])
    rotation_pnp = rotation_motion.T
    translation_pnp = -rotation_pnp @ translation_motion
    pixels = project(points, rotation_pnp, translation_pnp)

    entry = LandmarkEntry(
        id=7,
        landmark_class="keyframe_orb",
        position=np.array([10.0, -4.0, 0.0]),
        heading=np.deg2rad(30.0),
        descriptor=np.zeros(2048, np.float32),
        keypoints=pixels,
        points_3d=points,
        orb_desc=descriptors,
        t_first=0.0,
        t_last=0.0,
        n_obs=1,
        covariance=np.eye(2) * 0.25,
    )
    matcher = LandmarkMatcher(LandmarkDatabase([entry]), CALIBRATION)

    pose, inliers, rms, depth = matcher._verify(entry, pixels, descriptors)
    expected = StereoVO.integrate_relative(
        np.array([10.0, -4.0, np.deg2rad(30.0)]), forward, 0.0, yaw
    )

    assert inliers >= 15
    assert rms < 1.0
    assert np.allclose(pose[:2], expected[:2], atol=0.05)
    assert abs(pose[2] - expected[2]) < np.deg2rad(1.0)
    assert depth > 0


def test_verification_rejects_unrelated_pair():
    rng = np.random.default_rng(4)
    points = synthetic_scene(rng)
    entry = LandmarkEntry(
        id=1,
        landmark_class="keyframe_orb",
        position=np.zeros(3),
        heading=0.0,
        descriptor=np.zeros(2048, np.float32),
        keypoints=project(points, np.eye(3), np.zeros(3)),
        points_3d=points,
        orb_desc=rng.integers(0, 255, size=(len(points), 32), dtype=np.uint8),
        t_first=0.0,
        t_last=0.0,
        n_obs=1,
        covariance=np.eye(2) * 0.25,
    )
    matcher = LandmarkMatcher(LandmarkDatabase([entry]), CALIBRATION)
    other_descriptors = rng.integers(0, 255, size=(len(points), 32), dtype=np.uint8)

    assert matcher._verify(entry, entry.keypoints, other_descriptors) is None


# --- 5. Keyframe rule ----------------------------------------------------


def test_keyframes_follow_distance_not_time():
    fast = np.column_stack(
        (np.arange(0, 20, 1.0), np.zeros(20), np.zeros(20))
    )  # 1 m/frame
    slow = np.column_stack(
        (np.arange(0, 2.0, 0.1), np.zeros(20), np.zeros(20))
    )  # 0.1 m/frame

    assert select_keyframes(fast, distance_m=2.0, rotation_rad=1.0) == [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
    ]
    assert select_keyframes(slow, distance_m=2.0, rotation_rad=1.0) == [0]


def test_keyframe_triggers_on_rotation_without_translation():
    poses = np.column_stack(
        (np.zeros(10), np.zeros(10), np.linspace(0.0, np.deg2rad(90.0), 10))
    )

    assert len(select_keyframes(poses, distance_m=100.0, rotation_rad=np.deg2rad(15.0))) > 4


# --- 6. Serialize --------------------------------------------------------


@dataclass
class FakeFrame:
    timestamp: float
    left_path: str
    right_path: str


def test_serialize_round_trip_preserves_every_field(tmp_path: Path):
    rng = np.random.default_rng(5)
    entries = [
        LandmarkEntry(
            id=index,
            landmark_class="keyframe_orb",
            position=np.array([index * 2.0, 1.0, 0.0]),
            heading=0.3 * index,
            descriptor=rng.random(2048).astype(np.float32),
            keypoints=rng.random((10 + index, 2)).astype(np.float32),
            points_3d=rng.random((10 + index, 3)).astype(np.float32),
            orb_desc=rng.integers(0, 255, size=(10 + index, 32), dtype=np.uint8),
            t_first=float(index),
            t_last=float(index),
            n_obs=1,
            covariance=np.eye(2) * (0.5 + index),
        )
        for index in range(3)
    ]
    database = LandmarkDatabase(entries, {"point_frame": "cam0_optical"})
    path = tmp_path / "db.npz"
    database.save(path)
    loaded = LandmarkDatabase.load(path)

    assert loaded.metadata == database.metadata
    for original, restored in zip(entries, loaded.entries):
        assert original.id == restored.id
        assert original.landmark_class == restored.landmark_class
        assert np.array_equal(original.position, restored.position)
        assert original.heading == pytest.approx(restored.heading)
        assert np.array_equal(original.descriptor, restored.descriptor)
        assert np.array_equal(original.keypoints, restored.keypoints)
        assert np.array_equal(original.points_3d, restored.points_3d)
        assert np.array_equal(original.orb_desc, restored.orb_desc)
        assert original.n_obs == restored.n_obs
        assert np.array_equal(original.covariance, restored.covariance)


# --- 7. Spatial prior ----------------------------------------------------


def test_spatial_prior_excludes_far_candidates():
    images = [textured_image(seed) for seed in range(5)]
    database = LandmarkDatabase(
        [make_entry(index, image, (index * 40.0, 0.0)) for index, image in enumerate(images)]
    )
    matcher = LandmarkMatcher(database, CALIBRATION)

    result = matcher.query(
        images[4],
        images[4],
        prior_position=np.array([0.0, 0.0]),
        prior_covariance=np.eye(2) * 0.01,
    )

    assert result.candidate_count == 1
    assert result.ranked_entry_ids[0] == 0  # entry đúng bị prior loại, không biến mất khỏi metric
    assert not result.used_full_database


def test_prior_radius_is_clipped_by_configuration():
    matcher = LandmarkMatcher(LandmarkDatabase([]), CALIBRATION, LandmarkConfig())

    assert matcher.prior_radius(np.eye(2) * 1e-9) == pytest.approx(10.0)
    assert matcher.prior_radius(np.eye(2) * 1e6) == pytest.approx(50.0)


def test_reacquisition_drops_prior_after_consecutive_misses():
    images = [textured_image(seed) for seed in range(4)]
    database = LandmarkDatabase(
        [make_entry(index, image, (index * 40.0, 0.0)) for index, image in enumerate(images)]
    )
    config = LandmarkConfig(reacquisition_after_misses=3)
    matcher = LandmarkMatcher(database, CALIBRATION, config)
    blank = np.zeros((400, 800), np.uint8)
    prior = dict(prior_position=np.array([0.0, 0.0]), prior_covariance=np.eye(2) * 0.01)

    for _ in range(3):
        result = matcher.query(blank, blank, **prior)
        assert not result.used_full_database
    assert matcher.query(blank, blank, **prior).used_full_database


# --- 8. Measurement covariance ------------------------------------------


def test_measurement_covariance_adds_each_component_once():
    entry = make_entry(0, textured_image(6), (0.0, 0.0))
    config = LandmarkConfig()
    matcher = LandmarkMatcher(LandmarkDatabase([entry]), CALIBRATION, config)

    covariance, parts = matcher.measurement_covariance(entry, inliers=40, rms=1.0, median_depth=12.0)
    expected = (
        entry.covariance[0, 0]
        + config.sigma_model_m**2
        + parts["sigma_geometry_m"] ** 2
        + config.lever_arm_m**2
    )

    assert covariance[0, 0] == pytest.approx(expected)
    assert parts["sigma_geometry_m"] == pytest.approx(config.min_sigma_geometry_m)


def test_build_database_uses_causal_poses_only(tmp_path: Path):
    # Ảnh phải là cặp stereo thật: right = left dịch ngang 20 px → disparity 20
    # px nằm trong gate [fx·b/60, fx·b/1] = [2, 120] px, cho depth ≈ 6 m.
    left_images = {index: textured_image(index) for index in range(6)}
    right_images = {
        index: np.roll(image, -20, axis=1) for index, image in left_images.items()
    }
    frames = [FakeFrame(float(index), f"L{index}", f"R{index}") for index in range(6)]
    poses = np.column_stack((np.arange(6) * 2.5, np.zeros(6), np.zeros(6)))
    covariances = np.stack([np.eye(2) * 0.01] * 6)

    def read(path):
        index = int(str(path)[1:])
        return left_images[index] if str(path).startswith("L") else right_images[index]

    database = build_database(
        frames,
        poses,
        covariances,
        CALIBRATION,
        read_image=read,
    )

    assert len(database) == 6
    assert database.metadata["point_frame"] == "cam0_optical"
    assert database.metadata["pose_frame"] == "base_link_se2"
    assert all(entry.position[2] == 0.0 for entry in database.entries)
    assert all(entry.n_obs == 1 for entry in database.entries)
