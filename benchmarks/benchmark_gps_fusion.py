#!/usr/bin/env python3
"""Benchmark GPS integrity + EKF bằng NMEA thật và kịch bản kiểm soát."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_tools.gps_sources import (
    GPSMeasurement,
    GPSReplay,
    ScenarioSegment,
    load_nmea_replay,
    load_reference_trajectory,
    odometry_proxy_from_recording,
    simulate_gps,
    wrap_angle,
)
from pipelines.localization_ekf import GPSState, LocalizationFusion


STATE_COLORS = {
    GPSState.GOOD.value: (30, 180, 30),
    GPSState.DEGRADED.value: (0, 180, 255),
    GPSState.LOST.value: (30, 30, 220),
    GPSState.RECOVERING.value: (220, 120, 20),
}


def condition_starts(
    replay: GPSReplay, start: float, end: float, timeout: float
) -> list[dict]:
    measurements = [
        measurement
        for measurement in replay.measurements
        if start <= measurement.timestamp <= end
    ]
    events = []
    was_bad = False
    previous_time = None
    for measurement in measurements:
        if previous_time is not None and measurement.timestamp - previous_time > timeout:
            events.append(
                {
                    "timestamp": previous_time + timeout,
                    "kind": "receiver_timeout",
                }
            )
        bad = (
            measurement.fix_quality == 0
            or not measurement.has_position
            or measurement.hdop is None
            or measurement.hdop > 5
            or measurement.satellites < 4
        )
        if bad and not was_bad:
            events.append(
                {
                    "timestamp": measurement.timestamp,
                    "kind": (
                        "fix_quality_zero"
                        if measurement.fix_quality == 0
                        else "quality_degraded"
                    ),
                }
            )
        was_bad = bad
        previous_time = measurement.timestamp
    return events


def summarize_acceptance(gps_log: list[dict]) -> dict:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in gps_log:
        if row["quality_good"]:
            grouped[row["state_before"]].append(row["accepted"])
    return {
        state: {
            "measurements": len(values),
            "accepted": int(sum(values)),
            "accept_rate": float(np.mean(values)) if values else None,
        }
        for state, values in grouped.items()
    }


def metric_at_time(
    timestamps: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    target_time: float,
) -> float | None:
    index = int(np.searchsorted(timestamps, target_time))
    if index >= len(timestamps):
        return None
    return float(np.linalg.norm(predicted[index, :2] - reference[index]))


def stable_relock_metric(
    timestamps: np.ndarray,
    errors: np.ndarray,
    anchor_time: float,
    *,
    threshold_m: float = 5.0,
    stable_duration_seconds: float = 1.0,
    max_window_seconds: float = 10.0,
) -> dict:
    """Đo lần đầu error GT ổn định dưới threshold trong cửa sổ hữu hạn."""
    start_index = int(np.searchsorted(timestamps, anchor_time))
    deadline = anchor_time + max_window_seconds
    for index in range(start_index, len(timestamps)):
        candidate_time = float(timestamps[index])
        stable_end = candidate_time + stable_duration_seconds
        if stable_end > deadline:
            break
        if errors[index] > threshold_m:
            continue
        end_index = int(np.searchsorted(timestamps, stable_end))
        if end_index >= len(timestamps) or timestamps[end_index] > deadline:
            continue
        if np.all(errors[index : end_index + 1] <= threshold_m):
            return {
                "time_to_stable_5m_seconds": candidate_time - anchor_time,
                "error_at_stable_m": float(errors[index]),
            }
    return {
        "time_to_stable_5m_seconds": None,
        "error_at_stable_m": None,
    }


def path_length_between(
    timestamps: np.ndarray, reference: np.ndarray, start: float, end: float
) -> float:
    mask = (timestamps >= start) & (timestamps <= end)
    points = reference[mask]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def evaluate_run(
    fusion: LocalizationFusion,
    replay: GPSReplay,
    timestamps: np.ndarray,
    predicted: np.ndarray,
    local: np.ndarray,
    reference: np.ndarray,
) -> dict:
    transitions = [
        {
            "timestamp": item.timestamp,
            "previous": item.previous.value,
            "current": item.current.value,
            "reason": item.reason,
        }
        for item in fusion.integrity.transitions
    ]
    events = condition_starts(
        replay,
        float(timestamps[0]),
        float(timestamps[-1]),
        fusion.integrity.config.timeout_seconds,
    )
    handovers = []
    for event in events:
        state_at_event = GPSState.LOST.value
        for transition in transitions:
            if transition["timestamp"] > event["timestamp"]:
                break
            state_at_event = transition["current"]
        if state_at_event in {"DEGRADED", "LOST"}:
            handovers.append(
                {
                    **event,
                    "detected_state": state_at_event,
                    "latency_seconds": 0.0,
                }
            )
            continue
        match = next(
            (
                transition
                for transition in transitions
                if event["timestamp"] <= transition["timestamp"] <= event["timestamp"] + 2.5
                and transition["current"] in {"DEGRADED", "LOST"}
            ),
            None,
        )
        handovers.append(
            {
                **event,
                "detected_state": None if match is None else match["current"],
                "latency_seconds": (
                    None
                    if match is None
                    else match["timestamp"] - event["timestamp"]
                ),
            }
        )

    relocks = []
    active_loss = None
    recovering_time = None
    trajectory_errors = np.linalg.norm(predicted[:, :2] - reference, axis=1)
    for transition in transitions:
        if transition["current"] == "LOST" and active_loss is None:
            active_loss = transition["timestamp"]
            recovering_time = None
        elif (
            active_loss is not None
            and transition["current"] == "RECOVERING"
            and recovering_time is None
        ):
            recovering_time = transition["timestamp"]
        elif active_loss is not None and transition["current"] == "GOOD":
            recovery_anchor = next(
                (
                    row["timestamp"]
                    for row in fusion.gps_log
                    if row["timestamp"] >= active_loss
                    and row["state_before"] == GPSState.LOST.value
                    and row["state_after"] == GPSState.RECOVERING.value
                    and row["accepted"]
                ),
                recovering_time,
            )
            error = metric_at_time(
                timestamps,
                predicted,
                reference,
                transition["timestamp"] + 2.0,
            )
            stable = {
                "time_to_stable_5m_seconds": None,
                "error_at_stable_m": None,
            }
            quality_counts = {}
            if recovery_anchor is not None:
                stable = stable_relock_metric(
                    timestamps,
                    trajectory_errors,
                    recovery_anchor,
                )
                quality_counts = dict(
                    sorted(
                        Counter(
                            str(item.fix_quality)
                            for item in replay.measurements
                            if recovery_anchor
                            <= item.timestamp
                            <= recovery_anchor + 10.0
                        ).items()
                    )
                )
            relocks.append(
                {
                    "lost_timestamp": active_loss,
                    "recovering_timestamp": recovering_time,
                    "recovery_anchor_timestamp": recovery_anchor,
                    "good_timestamp": transition["timestamp"],
                    "error_after_2s_m": error,
                    **stable,
                    "gps_quality_counts_10s": quality_counts,
                }
            )
            active_loss = None
            recovering_time = None

    dropouts = []
    for index, transition in enumerate(transitions):
        if transition["current"] != "LOST":
            continue
        next_recovery = next(
            (
                item
                for item in transitions[index + 1 :]
                if item["current"] == "RECOVERING"
            ),
            None,
        )
        if next_recovery is None:
            continue
        end = next_recovery["timestamp"]
        error = metric_at_time(timestamps, predicted, reference, end)
        distance = path_length_between(
            timestamps, reference, transition["timestamp"], end
        )
        dropouts.append(
            {
                "start": transition["timestamp"],
                "end": end,
                "duration_seconds": end - transition["timestamp"],
                "end_error_m": error,
                "path_length_m": distance,
                "drift_ratio": (
                    error / distance
                    if error is not None and distance > 1e-6
                    else None
                ),
            }
        )

    correction_discontinuities = []
    first_good_time = next(
        (
            transition["timestamp"]
            for transition in transitions
            if transition["current"] == "GOOD"
        ),
        float(timestamps[0]),
    )
    for index in range(1, len(predicted)):
        if timestamps[index] < first_good_time:
            continue
        local_delta = local[index, :2] - local[index - 1, :2]
        heading_offset = float(
            wrap_angle(predicted[index - 1, 2] - local[index - 1, 2])
        )
        cosine, sine = np.cos(heading_offset), np.sin(heading_offset)
        expected = np.array(
            [
                cosine * local_delta[0] - sine * local_delta[1],
                sine * local_delta[0] + cosine * local_delta[1],
            ]
        )
        actual = predicted[index, :2] - predicted[index - 1, :2]
        correction_discontinuities.append(float(np.linalg.norm(actual - expected)))

    return {
        "frames": len(timestamps),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "trajectory_error_m": {
            "median": float(np.median(trajectory_errors)),
            "p95": float(np.percentile(trajectory_errors, 95)),
            "final": float(trajectory_errors[-1]),
        },
        "handover": {
            "events": handovers,
            "max_latency_seconds": (
                max(
                    item["latency_seconds"]
                    for item in handovers
                    if item["latency_seconds"] is not None
                )
                if any(item["latency_seconds"] is not None for item in handovers)
                else None
            ),
            "within_2_seconds": (
                all(
                    item["latency_seconds"] is not None
                    and item["latency_seconds"] <= 2.0
                    for item in handovers
                )
                if handovers
                else None
            ),
        },
        "relock": {
            "events": relocks,
            "max_error_after_2s_m": (
                max(
                    item["error_after_2s_m"]
                    for item in relocks
                    if item["error_after_2s_m"] is not None
                )
                if any(item["error_after_2s_m"] is not None for item in relocks)
                else None
            ),
            "max_time_to_stable_5m_seconds": (
                max(
                    item["time_to_stable_5m_seconds"]
                    for item in relocks
                    if item["time_to_stable_5m_seconds"] is not None
                )
                if any(
                    item["time_to_stable_5m_seconds"] is not None
                    for item in relocks
                )
                else None
            ),
            "max_error_at_stable_m": (
                max(
                    item["error_at_stable_m"]
                    for item in relocks
                    if item["error_at_stable_m"] is not None
                )
                if any(
                    item["error_at_stable_m"] is not None
                    for item in relocks
                )
                else None
            ),
            "all_stable_within_10s": (
                all(
                    item["time_to_stable_5m_seconds"] is not None
                    for item in relocks
                )
                if relocks
                else None
            ),
        },
        "dropouts": dropouts,
        "local_continuity": {
            "max_correction_induced_discontinuity_m": (
                max(correction_discontinuities)
                if correction_discontinuities
                else 0.0
            ),
            "within_0_5_m": (
                max(correction_discontinuities, default=0.0) < 0.5
            ),
        },
        "nis_by_state": summarize_acceptance(fusion.gps_log),
        "transitions": transitions,
    }


def run_fusion(
    recording: Path, replay: GPSReplay, *, seed: int
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps, reference, _ = load_reference_trajectory(recording)
    odometry = odometry_proxy_from_recording(recording, seed=seed)
    replay.seek(float(timestamps[0]))
    fusion = LocalizationFusion()
    output_timestamps = []
    predicted = []
    local = []
    aligned_reference = []
    for index, measurement in enumerate(odometry, start=1):
        for gps_measurement in replay.pop_until(measurement.timestamp):
            fusion.process_gps(gps_measurement)
        fusion.process_odometry(measurement)
        output_timestamps.append(measurement.timestamp)
        predicted.append(fusion.global_pose)
        local.append(fusion.local_pose)
        aligned_reference.append(reference[index])
    arrays = (
        np.asarray(output_timestamps),
        np.asarray(predicted),
        np.asarray(local),
        np.asarray(aligned_reference),
    )
    report = evaluate_run(fusion, replay, *arrays)
    report["recording"] = recording.name
    report["gps_source"] = replay.metadata
    report["recovery_events"] = [
        {
            **event,
            "reference_error_m": metric_at_time(
                arrays[0],
                arrays[1],
                arrays[3],
                event["timestamp"],
            ),
        }
        for event in fusion.recovery_log
    ]
    report["reference_usage"] = (
        "Offline only: odometry proxy, simulated GPS, ENU rigid alignment and evaluation"
    )
    return report, *arrays


def render_trajectory(
    path: Path,
    predicted: np.ndarray,
    reference: np.ndarray,
    states: list[str],
) -> None:
    canvas = np.full((800, 1000, 3), 245, dtype=np.uint8)
    points = np.vstack((predicted[:, :2], reference))
    minimum = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - minimum, 1.0)
    scale = min(900 / extent[0], 700 / extent[1])

    def project(values: np.ndarray) -> np.ndarray:
        pixels = (values - minimum) * scale + np.array([50, 50])
        pixels[:, 1] = canvas.shape[0] - pixels[:, 1]
        return np.rint(pixels).astype(np.int32)

    reference_pixels = project(reference)
    predicted_pixels = project(predicted[:, :2])
    cv2.polylines(canvas, [reference_pixels], False, (100, 100, 100), 2)
    for index in range(1, len(predicted_pixels)):
        cv2.line(
            canvas,
            tuple(predicted_pixels[index - 1]),
            tuple(predicted_pixels[index]),
            STATE_COLORS.get(states[index], (0, 0, 0)),
            2,
        )
    cv2.putText(
        canvas,
        "gray=reference, green=GOOD, yellow=DEGRADED, red=LOST, blue=RECOVERING",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def simulated_segments(timestamps: np.ndarray) -> list[ScenarioSegment]:
    start, end = float(timestamps[0]), float(timestamps[-1])
    duration = end - start
    return [
        ScenarioSegment(start + 0.25 * duration, start + 0.32 * duration, "degraded"),
        ScenarioSegment(start + 0.45 * duration, start + 0.62 * duration, "lost"),
        ScenarioSegment(start + 0.62 * duration, start + 0.68 * duration, "recovering"),
    ]


def benchmark_case(
    recording: Path,
    replay: GPSReplay,
    output: Path,
    name: str,
    seed: int,
) -> dict:
    report, _, predicted, _, reference = run_fusion(recording, replay, seed=seed)
    # State được tái dựng từ transition để không giữ object fusion trong API.
    states = []
    current = GPSState.LOST.value
    transitions = iter(report["transitions"])
    transition = next(transitions, None)
    timestamps, _, _ = load_reference_trajectory(recording)
    for timestamp in timestamps[1:]:
        while transition is not None and transition["timestamp"] <= timestamp:
            current = transition["current"]
            transition = next(transitions, None)
        states.append(current)
    render_trajectory(output / f"{name}.png", predicted, reference, states)
    return report


def run(args: argparse.Namespace) -> dict:
    root = args.dataset
    garage_2 = root / "recording_2021-02-25_13-39-06"
    garage_3 = root / "recording_2021-05-10_19-15-19"
    neighborhood = root / "recording_2020-12-22_11-54-24"
    for recording in (garage_2, garage_3, neighborhood):
        if not recording.is_dir():
            raise FileNotFoundError(recording)

    real_replay = load_nmea_replay(garage_2, align_to_reference=True)
    garage_timestamps, garage_xy, _ = load_reference_trajectory(garage_3)
    neighborhood_timestamps, neighborhood_xy, _ = load_reference_trajectory(
        neighborhood
    )
    cases = {
        "garage_2_real_nmea": benchmark_case(
            garage_2, real_replay, args.output, "garage_2_real_nmea", args.seed
        ),
        "garage_3_simulated": benchmark_case(
            garage_3,
            simulate_gps(
                garage_timestamps,
                garage_xy,
                segments=simulated_segments(garage_timestamps),
                seed=args.seed,
            ),
            args.output,
            "garage_3_simulated",
            args.seed,
        ),
        "neighborhood_simulated": benchmark_case(
            neighborhood,
            simulate_gps(
                neighborhood_timestamps,
                neighborhood_xy,
                segments=simulated_segments(neighborhood_timestamps),
                seed=args.seed + 1,
            ),
            args.output,
            "neighborhood_simulated",
            args.seed,
        ),
    }
    report = {
        "method": "local odometry EKF + GPS integrity + smoothed map->odom",
        "kpi": {
            "handover_target_seconds": 2.0,
            "relock_target_error_m": 5.0,
            "correction_discontinuity_target_m": 0.5,
        },
        "cases": cases,
        "limitations": [
            "4Seasons reference pose combines stereo VIO and RTK-GNSS.",
            "NMEA-to-reference rigid alignment is offline evaluation setup, not online localization.",
            "Simulated GPS validates software transitions and does not replace real receiver evidence.",
            "Odometry is a noisy reference-pose proxy until stereo VO replaces the same API in Step 2.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/data/4seasons"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gps-fusion"),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    for name, case in report["cases"].items():
        print(
            name,
            "handover=",
            case["handover"]["max_latency_seconds"],
            "relock=",
            case["relock"]["max_error_after_2s_m"],
            "jump=",
            case["local_continuity"]["max_correction_induced_discontinuity_m"],
        )


if __name__ == "__main__":
    main()
