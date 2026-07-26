import numpy as np
import pytest

from benchmarks.benchmark_garage_localization import (
    SEGMENT_SECONDS,
    error_profile,
    suppress_position,
    summarize,
)
from data_tools.gps_sources import GPSMeasurement


def test_suppress_position_keeps_the_message_but_drops_the_fix():
    original = GPSMeasurement(
        timestamp=10.0, x=1.0, y=2.0, fix_quality=4, satellites=16, hdop=0.9
    )
    muted = suppress_position(original)
    assert muted.x is None and muted.y is None
    assert muted.fix_quality == 0
    assert muted.has_position is False
    # Timestamp và metadata phải giữ: receiver vẫn phát, chỉ là không có fix.
    assert muted.timestamp == 10.0
    assert muted.satellites == 16
    assert muted.mode == "artificial_outage"


def test_error_profile_interpolates_reference_at_pose_timestamps():
    reference_timestamps = np.array([0.0, 10.0])
    reference_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    timestamps = np.array([0.0, 5.0, 10.0])
    poses = np.array([[0.0, 3.0, 0.0], [5.0, 0.0, 0.0], [10.0, -4.0, 0.0]])
    errors = error_profile(timestamps, poses, reference_timestamps, reference_xy)
    assert errors == pytest.approx([3.0, 0.0, 4.0])


def test_summarize_splits_into_fixed_length_segments():
    timestamps = np.arange(0.0, 3 * SEGMENT_SECONDS, 1.0)
    errors = np.where(timestamps < SEGMENT_SECONDS, 1.0, 5.0)
    result = summarize(errors, timestamps)
    assert len(result["by_30s_segment"]) == 3
    assert result["by_30s_segment"][0]["median_m"] == pytest.approx(1.0)
    assert result["by_30s_segment"][1]["median_m"] == pytest.approx(5.0)
    # Gate #2 cấm kết luận bằng median toàn tuyến, nên curve phải luôn có mặt.
    assert {"p45", "p50", "p55", "p95"} <= set(result["percentile_curve"])


def test_summarize_handles_an_empty_segment():
    assert summarize(np.array([]), np.array([])) == {"samples": 0}
