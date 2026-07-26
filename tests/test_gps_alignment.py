from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_gps_alignment import (
    evaluate_candidate,
    quality4_positions,
)
from data_tools.gps_sources import load_nmea_replay, load_reference_trajectory


ROOT = Path(".cache/data/4seasons")
GARAGE = ROOT / "recording_2021-02-25_13-39-06"


def test_offset_selection_never_uses_holdout_for_rigid_fit():
    timestamps = np.arange(10, dtype=np.float64)
    reference_timestamps = timestamps.copy()
    reference_xy = np.column_stack((timestamps, timestamps**2))
    source = reference_xy.copy()
    source[6:] += 1000.0
    calibration = np.arange(6)
    holdout = np.arange(6, 10)
    result = evaluate_candidate(
        source,
        timestamps,
        reference_timestamps,
        reference_xy,
        calibration,
        holdout,
        np.array([0.0]),
        fit_rigid=True,
    )
    assert result["calibration"]["max_m"] < 1e-9
    assert result["holdout"]["median_m"] > 100


@pytest.mark.skipif(not GARAGE.exists(), reason="4Seasons chưa tải")
def test_official_chain_has_metric_scale_without_reference_fit():
    reference_timestamps, _, _ = load_reference_trajectory(GARAGE)
    replay = load_nmea_replay(GARAGE, alignment_mode="transform_chain")
    timestamps, positions = quality4_positions(replay, reference_timestamps)
    assert len(timestamps) > 500
    assert np.linalg.norm(np.ptp(positions, axis=0)) > 10
    assert replay.metadata["rotation"] == [[1.0, 0.0], [0.0, 1.0]]
    assert replay.metadata["translation"] == [0.0, 0.0]
    assert replay.metadata["transform_chain"]["gps_imu_lever_arm_m"] == 0.0


@pytest.mark.skipif(not GARAGE.exists(), reason="4Seasons chưa tải")
def test_replay_applies_calibrated_clock_offset():
    original = load_nmea_replay(GARAGE, alignment_mode="transform_chain")
    shifted = load_nmea_replay(
        GARAGE,
        alignment_mode="transform_chain",
        time_offset_s=0.1,
    )
    assert shifted.measurements[0].timestamp == pytest.approx(
        original.measurements[0].timestamp + 0.1
    )
