import numpy as np

from benchmarks.benchmark_uturn import ground_truth_turns, match


def synthetic_heading(
    segments: list[tuple[float, float]], rate_hz: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Nối các đoạn (duration_s, sweep_deg) thành heading liên tục."""
    timestamps = [0.0]
    heading = [0.0]
    for duration, sweep in segments:
        count = int(round(duration * rate_hz))
        step = np.deg2rad(sweep) / count
        for _ in range(count):
            timestamps.append(timestamps[-1] + 1.0 / rate_hz)
            heading.append(heading[-1] + step)
    return np.asarray(timestamps), np.asarray(heading)


def test_ground_truth_keeps_uturn_and_drops_corner():
    timestamps, heading = synthetic_heading(
        [(5.0, 0.0), (6.0, 180.0), (5.0, 0.0), (3.0, 90.0), (5.0, 0.0)]
    )
    turns = ground_truth_turns(timestamps, heading)
    assert len(turns) == 1
    assert abs(turns[0]["sweep_degrees"] - 180.0) < 5.0


def test_ground_truth_drops_slow_sweep_below_rate_gate():
    # Quét 180 deg nhưng trải 60 s -> 3 deg/s, dưới ngưỡng 5 deg/s.
    timestamps, heading = synthetic_heading([(5.0, 0.0), (60.0, 180.0)])
    assert ground_truth_turns(timestamps, heading) == []


def test_match_counts_duplicate_as_false_positive():
    truths = [{"start": 10.0, "end": 16.0}]
    result = match(
        truths,
        [{"timestamp": 15.0}, {"timestamp": 17.0}],
    )
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_match_reports_missed_turn_outside_tolerance():
    truths = [{"start": 10.0, "end": 16.0}, {"start": 40.0, "end": 46.0}]
    result = match(truths, [{"timestamp": 15.0}, {"timestamp": 60.0}])
    assert result["false_negatives"] == 1
    assert result["missed_turns"][0]["start"] == 40.0
    assert result["unmatched_detections"][0]["label"] == "false_positive"
