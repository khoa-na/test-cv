"""B3 — phát hiện U-turn trên heading do chính hệ ước lượng.

Ground truth sinh từ reference pose bằng thuật toán ĐỘC LẬP với detector:
phân đoạn theo turn rate rồi tích phân heading trong từng đoạn. Detector
production dùng cửa sổ trượt trên heading của `LocalizationFusion.global_pose`
và không bao giờ đọc reference.

Định nghĩa GT (chốt trước khi chạy, không đổi sau khi thấy KPI):
  Một U-turn thật = đoạn xoay liên tục cực đại thoả cả ba:
    - |turn rate đã làm mượt 1 s| > 5 deg/s
    - kéo dài > 2 s
    - |heading quét được trong đoạn| >= 150 deg

Khớp detection với GT: detection tính là TP nếu timestamp rơi trong
[segment_start, segment_end + 2.0 s]. Mỗi đoạn GT khớp tối đa một detection;
detection thứ hai trong cùng đoạn tính FP (bắn trùng).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_gps_fusion import run_fusion
from data_tools.gps_sources import (
    load_nmea_replay,
    load_reference_trajectory,
    wrap_angle,
)
from pipelines.localization_ekf import UTurnDetector

SEQUENCES = {
    "garage_2": "recording_2021-02-25_13-39-06",
    "garage_3": "recording_2021-05-10_19-15-19",
    "office_loop_1": "recording_2020-03-24_17-36-22",
    "neighborhood_4": "recording_2020-12-22_11-54-24",
}

GT_MIN_TURN_RATE_DEG_S = 5.0
GT_MIN_DURATION_S = 2.0
GT_MIN_SWEEP_DEG = 150.0
MATCH_TOLERANCE_S = 2.0


def unwrap_heading(heading: np.ndarray) -> np.ndarray:
    steps = np.concatenate([[0.0], wrap_angle(np.diff(heading))])
    return float(heading[0]) + np.cumsum(steps)


def ground_truth_turns(
    timestamps: np.ndarray, heading: np.ndarray
) -> list[dict]:
    """Đoạn xoay liên tục quét >= 150 deg, đo trên reference pose."""
    unwrapped = unwrap_heading(heading)
    dt = np.diff(timestamps)
    rate = np.rad2deg(np.diff(unwrapped)) / np.maximum(dt, 1e-6)
    window = max(1, int(round(1.0 / float(np.median(dt)))))
    smooth = np.convolve(rate, np.ones(window) / window, mode="same")

    turning = np.abs(smooth) > GT_MIN_TURN_RATE_DEG_S
    edges = np.diff(turning.astype(int))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if turning[0]:
        starts = np.concatenate([[0], starts])
    if turning[-1]:
        ends = np.concatenate([ends, [len(turning)]])

    turns = []
    for start, end in zip(starts, ends):
        duration = float(timestamps[end] - timestamps[start])
        sweep = float(np.rad2deg(unwrapped[end] - unwrapped[start]))
        if duration <= GT_MIN_DURATION_S or abs(sweep) < GT_MIN_SWEEP_DEG:
            continue
        turns.append(
            {
                "start": float(timestamps[start]),
                "end": float(timestamps[end]),
                "duration_s": duration,
                "sweep_degrees": sweep,
            }
        )
    return turns


def detect(timestamps: np.ndarray, heading: np.ndarray) -> list[dict]:
    detector = UTurnDetector()
    events = []
    for stamp, yaw in zip(timestamps, heading):
        event = detector.update(float(stamp), float(yaw))
        if event is not None:
            events.append(event)
    return events


def match(truths: list[dict], detections: list[dict]) -> dict:
    matched_truth: dict[int, int] = {}
    labels = []
    for index, event in enumerate(detections):
        stamp = event["timestamp"]
        hit = next(
            (
                truth_index
                for truth_index, truth in enumerate(truths)
                if truth["start"] <= stamp <= truth["end"] + MATCH_TOLERANCE_S
            ),
            None,
        )
        if hit is None:
            labels.append("false_positive")
        elif hit in matched_truth:
            labels.append("duplicate")
        else:
            matched_truth[hit] = index
            labels.append("true_positive")

    true_positives = len(matched_truth)
    false_positives = sum(label != "true_positive" for label in labels)
    false_negatives = len(truths) - true_positives

    latencies = [
        detections[matched_truth[truth_index]]["timestamp"]
        - truths[truth_index]["start"]
        for truth_index in matched_truth
    ]

    return {
        "ground_truth_turns": len(truths),
        "detections": len(detections),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": (
            true_positives / len(detections) if detections else float("nan")
        ),
        "recall": true_positives / len(truths) if truths else float("nan"),
        "detection_latency_s": {
            "median": float(np.median(latencies)) if latencies else None,
            "max": float(np.max(latencies)) if latencies else None,
        },
        "missed_turns": [
            truths[index]
            for index in range(len(truths))
            if index not in matched_truth
        ],
        "unmatched_detections": [
            {**detections[index], "label": label}
            for index, label in enumerate(labels)
            if label != "true_positive"
        ],
    }


def evaluate(
    name: str, recording: Path, calibration: Path, max_vo_frames: int | None
) -> dict:
    reference_timestamps, _, reference_heading = load_reference_trajectory(
        recording
    )
    truths = ground_truth_turns(reference_timestamps, reference_heading)

    replay = load_nmea_replay(recording, alignment_mode="reference_rigid")
    report, timestamps, predicted, local, _ = run_fusion(
        recording,
        replay,
        seed=7,
        odom_source="vo",
        calibration_dir=calibration,
        max_vo_frames=max_vo_frames,
        yaw_source="imu",
    )

    system = match(truths, detect(timestamps, predicted[:, 2]))
    # Kênh phụ: heading local frame, không chịu xoay từ map->odom correction.
    local_only = match(truths, detect(timestamps, local[:, 2]))
    # Trần lý thuyết: detector chạy trực tiếp trên reference heading.
    oracle = match(
        truths, detect(reference_timestamps, reference_heading)
    )

    return {
        "recording": recording.name,
        "ground_truth": {
            "source": "reference pose (evaluation only)",
            "uses_ground_truth": True,
            "turns": truths,
        },
        "system_global_heading": system,
        "local_heading": local_only,
        "reference_heading_oracle": oracle,
        "odometry": report["odometry"],
        "vo_frames": int(len(timestamps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(".cache/data/4seasons")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/uturn-b3")
    )
    parser.add_argument(
        "--sequence", choices=sorted(SEQUENCES), action="append"
    )
    parser.add_argument("--max-vo-frames", type=int)
    args = parser.parse_args()

    calibration = args.dataset / "calibration"
    names = args.sequence or ["garage_2", "garage_3"]
    cases = {}
    for name in names:
        recording = args.dataset / SEQUENCES[name]
        if not recording.is_dir():
            raise FileNotFoundError(recording)
        cases[name] = evaluate(
            name, recording, calibration, args.max_vo_frames
        )
        summary = cases[name]["system_global_heading"]
        print(
            f"{name}: GT {summary['ground_truth_turns']}  "
            f"TP {summary['true_positives']}  "
            f"FP {summary['false_positives']}  "
            f"FN {summary['false_negatives']}  "
            f"precision {summary['precision']:.3f}  "
            f"recall {summary['recall']:.3f}"
        )

    report = {
        "kpi": {
            "threshold_degrees": 150.0,
            "window_seconds": 8.0,
            "note": (
                "Ngưỡng và cửa sổ chốt ở Bước 1, giữ nguyên sau khi thấy KPI."
            ),
        },
        "ground_truth_definition": {
            "min_turn_rate_deg_s": GT_MIN_TURN_RATE_DEG_S,
            "min_duration_s": GT_MIN_DURATION_S,
            "min_sweep_deg": GT_MIN_SWEEP_DEG,
            "match_tolerance_s": MATCH_TOLERANCE_S,
        },
        "cases": cases,
        "limitations": [
            "U-turn thật chỉ xuất hiện ở hai tuyến garage; office_loop_1 "
            "(3776 m) và neighborhood_4 (2195 m) có 0 U-turn nên không đóng "
            "góp mẫu dương.",
            "GT sinh từ reference pose (stereo VIO + RTK-GNSS), không phải "
            "nhãn người.",
            "Detector đọc heading của hệ; reference chỉ dùng để chấm.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nartifact: {args.output / 'benchmark.json'}")


if __name__ == "__main__":
    main()
