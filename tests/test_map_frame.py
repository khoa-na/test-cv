"""Test common map frame của Bước 3 (STEP3 mục 2, test 9–10)."""

from __future__ import annotations

import numpy as np
import pytest

from data_tools.map_frame import Datum, geodetic_to_enu_3d, kabsch


def test_kabsch_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3)) * 10.0
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([12.0, -5.0, 3.0])
    target = source @ rotation.T + translation

    fitted_rotation, fitted_translation = kabsch(source, target)

    assert np.allclose(fitted_rotation, rotation, atol=1e-9)
    assert np.allclose(fitted_translation, translation, atol=1e-9)


def test_kabsch_does_not_absorb_scale():
    """Fit là rigid: dữ liệu bị scale phải để lại residual, không âm thầm khớp."""
    rng = np.random.default_rng(1)
    source = rng.normal(size=(40, 3)) * 10.0
    target = source * 1.10

    rotation, translation = kabsch(source, target)
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)

    assert residual.mean() > 0.5


def test_kabsch_rejects_reflection():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mirrored = source * np.array([1.0, 1.0, -1.0])

    rotation, _ = kabsch(source, mirrored)

    assert np.linalg.det(rotation) > 0


def test_enu_3d_keeps_altitude_and_orients_axes():
    datum = Datum(48.0, 11.0, 500.0)
    north = geodetic_to_enu_3d(
        np.array([48.001]), np.array([11.0]), np.array([500.0]), datum
    )[0]
    east = geodetic_to_enu_3d(
        np.array([48.0]), np.array([11.001]), np.array([500.0]), datum
    )[0]
    up = geodetic_to_enu_3d(
        np.array([48.0]), np.array([11.0]), np.array([510.0]), datum
    )[0]

    assert north[1] > 100.0 and abs(north[0]) < 1.0
    assert east[0] > 60.0 and abs(east[1]) < 1.0
    # Khác geodetic_to_enu 2D: cao độ không bị bỏ.
    assert up[2] == pytest.approx(10.0, abs=1e-3)
    assert np.linalg.norm(up[:2]) < 1e-3


def test_map_frame_direction_is_reference_to_enu():
    """Gọi ngược chiều phải hỏng — bảo vệ khỏi lặp lại chiều fit vòng 3–5."""
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(30, 3)) * 20.0
    angle = np.deg2rad(-110.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([-1250.0, 340.0, 5.0])
    enu = reference @ rotation.T + translation

    forward_rotation, forward_translation = kabsch(reference, enu)
    reversed_rotation, reversed_translation = kabsch(enu, reference)

    forward = np.linalg.norm(
        reference @ forward_rotation.T + forward_translation - enu, axis=1
    )
    reversed_applied = np.linalg.norm(
        reference @ reversed_rotation.T + reversed_translation - enu, axis=1
    )

    assert forward.max() < 1e-6
    assert reversed_applied.max() > 100.0
