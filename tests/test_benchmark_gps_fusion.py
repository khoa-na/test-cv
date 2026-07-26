"""Tests cho metric benchmark GPS fusion; reference chỉ dùng offline."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.benchmark_gps_fusion import stable_relock_metric


def test_stable_metric_rejects_single_frame_below_threshold():
    timestamps = np.arange(0.0, 3.1, 0.1)
    errors = np.full_like(timestamps, 8.0)
    errors[10] = 4.0

    result = stable_relock_metric(timestamps, errors, anchor_time=1.0)

    assert result["time_to_stable_5m_seconds"] is None
    assert result["error_at_stable_m"] is None


def test_stable_metric_requires_one_continuous_second():
    timestamps = np.arange(0.0, 4.1, 0.1)
    errors = np.full_like(timestamps, 8.0)
    errors[timestamps >= 1.5] = 4.5

    result = stable_relock_metric(timestamps, errors, anchor_time=1.0)

    assert result["time_to_stable_5m_seconds"] == pytest.approx(0.5)
    assert result["error_at_stable_m"] == pytest.approx(4.5)


def test_stable_metric_is_bounded_to_ten_seconds():
    timestamps = np.arange(0.0, 13.1, 0.1)
    errors = np.full_like(timestamps, 8.0)
    errors[timestamps >= 11.1] = 4.0

    result = stable_relock_metric(timestamps, errors, anchor_time=1.0)

    assert result["time_to_stable_5m_seconds"] is None
    assert result["error_at_stable_m"] is None
