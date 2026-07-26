"""Tests cho metric B1, tách khỏi chất lượng feature trên data thật."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.benchmark_vo_drift import (
    drift_windows,
    integrate_measurements,
    normalize_trajectory,
    summarize_cases,
)
from data_tools.gps_sources import OdometryMeasurement


def straight_poses(distance_m: float, samples: int = 1001) -> np.ndarray:
    return np.column_stack(
        (
            np.linspace(0.0, distance_m, samples),
            np.zeros(samples),
            np.zeros(samples),
        )
    )


def test_exact_relative_trajectory_has_zero_drift():
    reference = straight_poses(1000.0)
    windows = drift_windows(reference.copy(), reference)
    assert len(windows) == 2
    assert all(
        window["translation_drift_ratio"] == pytest.approx(0.0)
        for window in windows
    )


def test_metric_does_not_carry_previous_window_offset():
    reference = straight_poses(1000.0)
    vo = reference.copy()
    vo[500:, 0] += 30.0
    windows = drift_windows(vo, reference)
    assert windows[0]["translation_drift_ratio"] == pytest.approx(0.06)
    assert windows[1]["translation_drift_ratio"] == pytest.approx(0.0)


def test_scale_error_is_not_aligned_away():
    reference = straight_poses(500.0, samples=501)
    vo = reference.copy()
    vo[:, 0] *= 0.9
    windows = drift_windows(vo, reference)
    assert len(windows) == 1
    assert windows[0]["translation_drift_ratio"] == pytest.approx(0.1)


def test_normalize_removes_only_initial_se2_gauge():
    heading = np.deg2rad(30.0)
    poses = np.array(
        [
            [10.0, -4.0, heading],
            [10.0 + 2.0 * np.cos(heading), -4.0 + 2.0 * np.sin(heading), heading],
        ]
    )
    normalized = normalize_trajectory(poses)
    np.testing.assert_allclose(normalized[0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(normalized[1], np.array([2.0, 0.0, 0.0]), atol=1e-12)


def test_dropout_uses_last_vo_velocity_for_dead_reckoning():
    measurement = OdometryMeasurement(
        timestamp=0.1,
        dt=0.1,
        dx=1.0,
        dy=0.0,
        dtheta=0.0,
        translation_std=0.01,
        rotation_std=0.001,
        source="test",
    )
    frames = [
        SimpleNamespace(timestamp=0.0, measurement=None),
        SimpleNamespace(timestamp=0.1, measurement=measurement),
        SimpleNamespace(timestamp=0.2, measurement=None),
    ]
    _, poses = integrate_measurements(frames)
    assert poses[-1, 0] == pytest.approx(2.0)


def test_aggregate_summary_lists_failed_window():
    cases = {
        "a": {
            "windows": [
                {
                    "translation_drift_ratio": 0.04,
                    "translation_error_m": 20.0,
                    "path_length_m": 500.0,
                },
                {
                    "translation_drift_ratio": 0.06,
                    "translation_error_m": 30.0,
                    "path_length_m": 500.0,
                },
            ]
        }
    }
    summary = summarize_cases(cases)
    assert summary["total_windows"] == 2
    assert summary["passed_windows"] == 1
    assert not summary["all_windows_pass"]
    assert summary["failures"][0]["sequence"] == "a"
    assert summary["failures"][0]["window_index"] == 2
