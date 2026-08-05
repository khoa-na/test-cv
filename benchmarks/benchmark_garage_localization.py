"""B5 — localization trong bãi đỗ có mái, nhiều tầng, GPS bị chặn.

Ba cấu hình dùng chung một segment vào/ra hầm đã chốt ở gate #1
(`artifacts/garage-pair-audit/segments.json`), không trim theo error:

  A  VO + NMEA thật, KHÔNG landmark. Baseline.
  B  A + landmark correction (DB dựng từ garage_3).
  C  B nhưng GPS bị cắt nhân tạo: giữ fix đến ngay trước cửa hầm để latch, sau
     đó replay quality-0 suốt đoạn có mái. Kịch bản thuần visual.

A/B đo outage THẬT (35,4 s). C là đoạn cắt dài hơn có kiểm soát (50,4 s). Báo
cáo phải phân biệt hai loại.

Gate #2 đã chứng minh median toàn tuyến của recording này mong manh (p45 5,49 m
vs p55 9,96 m), nên script bắt buộc in error theo từng đoạn 30 s và full
percentile curve. Kết luận A/B/C chỉ bằng median toàn tuyến là sai.

Landmark đi vào EKF qua đúng hook `update_position` mà GPS dùng. Không sửa
phương trình EKF nào.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_gps_fusion import stable_relock_metric
from benchmarks.benchmark_landmark_reid import (
    keyframe_indices_causal,
    run_causal_pipeline,
)
from data_tools.gps_sources import (
    GPSMeasurement,
    load_nmea_replay,
    load_reference_trajectory,
)
from data_tools.imu_yaw import load_imu_yaw_integrator
from data_tools.map_frame import (
    first_quality4_datum,
    fit_reference_to_map,
    reference_in_map_frame,
)
from data_tools.stereo_odometry import load_calibration, stereo_frames
from pipelines.landmark_db import LandmarkConfig, LandmarkMatcher, build_database
from pipelines.localization_ekf import FusionConfig, GPSState, LocalizationFusion
from pipelines.stereo_vo import StereoVO, StereoVOConfig

MAPPING = "recording_2021-05-10_19-15-19"
QUERY = "recording_2021-02-25_13-39-06"
SEGMENT_FILE = Path("artifacts/garage-pair-audit/segments.json")
# Sàn đăng ký liên-traversal đo ở gate #1; in cạnh mọi error, production không đọc.
REGISTRATION_FLOOR_M = 0.349
SEGMENT_SECONDS = 30.0
PERCENTILES = (5, 25, 45, 50, 55, 75, 90, 95, 99)


def load_segment() -> dict:
    data = json.loads(SEGMENT_FILE.read_text(encoding="utf-8"))
    if data["recording"] != QUERY:
        raise ValueError("segments.json không thuộc query recording")
    return data


def cache_visual_odometry(recording: Path, calibration_dir: Path) -> list:
    """Chạy VO một lần rồi dùng lại cho cả ba cấu hình.

    VO chỉ phụ thuộc ảnh, không phụ thuộc chính sách GPS/landmark, nên chạy lại
    ba lần là lãng phí và còn tạo cơ hội cho ba cấu hình lệch nhau vì lý do
    ngoài chính sách.
    """
    frames = stereo_frames(recording)
    vo = StereoVO(load_calibration(calibration_dir), StereoVOConfig())
    imu_yaw = load_imu_yaw_integrator(recording, frames[0].timestamp)
    cached = []
    for frame in frames:
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
        measurement = vo.process(left, right, frame.timestamp)
        if measurement is not None:
            heading = imu_yaw.delta_heading(
                measurement.timestamp - measurement.dt, measurement.timestamp
            )
            if heading is not None:
                measurement = replace(
                    measurement, dtheta=heading, source="stereo_vo_imu_yaw"
                )
        cached.append((frame, measurement))
    return cached


def suppress_position(measurement: GPSMeasurement) -> GPSMeasurement:
    """Bản tin quality-0: receiver còn phát nhưng không có fix."""
    return replace(measurement, x=None, y=None, fix_quality=0, mode="artificial_outage")


def run_configuration(
    name: str,
    cached: list,
    replay,
    matcher: LandmarkMatcher | None,
    keyframes: set[int],
    segment: dict,
    *,
    artificial_outage: bool,
    fusion_config: FusionConfig | None = None,
) -> dict:
    fusion = LocalizationFusion(fusion_config)
    enter = segment["covered_segment"]["enter_timestamp"]
    exit_time = segment["covered_segment"]["exit_timestamp"]

    timestamps, poses, states = [], [], []
    landmark_attempts = 0
    landmark_matched = 0
    landmark_accepted = 0
    landmark_nis_rejected = 0
    accepted_inside = 0
    attempts_inside = 0
    previous_timestamp = float(cached[0][0].timestamp)

    for index, (frame, measurement) in enumerate(cached):
        stamp = float(frame.timestamp)
        for gps in replay.pop_until(stamp):
            if artificial_outage and enter <= gps.timestamp <= exit_time:
                gps = suppress_position(gps)
            fusion.process_gps(gps)

        if measurement is None:
            dt = 1.0 / 30.0 if index == 0 else stamp - previous_timestamp
            fusion.predict_only(
                stamp,
                dt,
                translation_process_std=0.05,
                rotation_process_std=np.deg2rad(1.0),
            )
        else:
            fusion.process_odometry(measurement)
        previous_timestamp = stamp

        if matcher is not None and index in keyframes:
            inside = enter <= stamp <= exit_time
            landmark_attempts += 1
            attempts_inside += inside
            left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
            result = matcher.query(
                left,
                right,
                prior_position=fusion.global_pose[:2],
                prior_covariance=fusion.global_position_covariance(),
            )
            if result.match is not None:
                landmark_matched += 1
                # Cùng hook GPS dùng; gate NIS bật, không nới cho landmark.
                accepted, _ = fusion.update_position(
                    result.match.position, result.match.covariance, gate=True
                )
                landmark_accepted += accepted
                landmark_nis_rejected += not accepted
                accepted_inside += accepted and inside

        timestamps.append(stamp)
        poses.append(fusion.global_pose.copy())
        states.append(fusion.integrity.state.value)

    return {
        "configuration": name,
        "timestamps": np.asarray(timestamps),
        "poses": np.asarray(poses),
        "states": states,
        "landmark": {
            "queries": landmark_attempts,
            "verified_matches": landmark_matched,
            "accepted": landmark_accepted,
            "nis_rejected": landmark_nis_rejected,
            "no_verified_match": landmark_attempts - landmark_matched,
            "queries_inside_cover": attempts_inside,
            "accepted_inside_cover": accepted_inside,
        },
        "recovery_log": fusion.recovery_log,
    }


def error_profile(
    timestamps: np.ndarray,
    poses: np.ndarray,
    reference_timestamps: np.ndarray,
    reference_xy: np.ndarray,
) -> np.ndarray:
    interpolated = np.column_stack(
        [
            np.interp(timestamps, reference_timestamps, reference_xy[:, axis])
            for axis in range(2)
        ]
    )
    return np.linalg.norm(poses[:, :2] - interpolated, axis=1)


def summarize(errors: np.ndarray, timestamps: np.ndarray) -> dict:
    if not len(errors):
        return {"samples": 0}
    segments = []
    start = timestamps[0]
    while start < timestamps[-1]:
        mask = (timestamps >= start) & (timestamps < start + SEGMENT_SECONDS)
        if mask.any():
            segments.append(
                {
                    "t_offset_s": float(start - timestamps[0]),
                    "samples": int(mask.sum()),
                    "median_m": float(np.median(errors[mask])),
                    "p95_m": float(np.percentile(errors[mask], 95)),
                }
            )
        start += SEGMENT_SECONDS
    return {
        "samples": int(len(errors)),
        "median_m": float(np.median(errors)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "percentile_curve": {
            f"p{value}": float(np.percentile(errors, value)) for value in PERCENTILES
        },
        "by_30s_segment": segments,
        "registration_floor_m": REGISTRATION_FLOOR_M,
    }


def evaluate(run: dict, reference: tuple, segment: dict) -> dict:
    reference_timestamps, reference_xy = reference
    timestamps, poses = run["timestamps"], run["poses"]
    errors = error_profile(timestamps, poses, reference_timestamps, reference_xy)

    enter = segment["covered_segment"]["enter_timestamp"]
    exit_time = segment["covered_segment"]["exit_timestamp"]
    inside = (timestamps >= enter) & (timestamps <= exit_time)

    inside_path = 0.0
    if inside.sum() > 1:
        interpolated = np.column_stack(
            [
                np.interp(timestamps[inside], reference_timestamps, reference_xy[:, axis])
                for axis in range(2)
            ]
        )
        inside_path = float(
            np.sum(np.linalg.norm(np.diff(interpolated, axis=0), axis=1))
        )
    exit_error = float(errors[inside][-1]) if inside.any() else None

    # Metric B8 áp lên thời điểm ra khỏi mái: GPS quay lại đúng lúc này, nên
    # exit_time chính là anchor re-lock. Cùng định nghĩa với benchmark_gps_fusion.
    after_2s = None
    probe = exit_time + 2.0
    if timestamps[-1] >= probe:
        after_2s = float(np.interp(probe, timestamps, errors))
    relock = {
        "anchor_timestamp": exit_time,
        "error_after_2s_m": after_2s,
        **stable_relock_metric(timestamps, errors, exit_time),
    }

    states_inside = [
        state for state, keep in zip(run["states"], inside) if keep
    ]
    landmark = run["landmark"]
    coverage = (
        landmark["accepted_inside_cover"] / landmark["queries_inside_cover"]
        if landmark["queries_inside_cover"]
        else None
    )

    return {
        "configuration": run["configuration"],
        "whole_track": summarize(errors, timestamps),
        "inside_cover": summarize(errors[inside], timestamps[inside]),
        "cover_segment": {
            "enter_timestamp": enter,
            "exit_timestamp": exit_time,
            "duration_s": exit_time - enter,
            "reference_path_length_m": inside_path,
            "exit_error_m": exit_error,
            "exit_drift_percent": (
                100.0 * exit_error / inside_path if inside_path > 0 and exit_error else None
            ),
            "gps_states_seen": sorted(set(states_inside)),
        },
        "relock_after_exit": relock,
        "landmark": {
            **landmark,
            "accepted_coverage_inside": coverage,
        },
        "recovery_events": [
            {k: v for k, v in event.items() if k != "target"}
            for event in run["recovery_log"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/garage-localization")
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--consensus-sweep",
        action="store_true",
        help=(
            "Thêm các cấu hình chỉ đổi recovery_consensus_count/radius trên nền "
            "cấu hình A, để đo đường đánh đổi của B8. Không đổi canonical config."
        ),
    )
    args = parser.parse_args()

    mapping = args.dataset / MAPPING
    query = args.dataset / QUERY
    calibration_dir = args.dataset / "calibration"
    segment = load_segment()
    datum = first_quality4_datum(mapping)
    config = LandmarkConfig()

    mapping_run = run_causal_pipeline(
        mapping, calibration_dir, datum, max_frames=args.max_frames
    )
    database = build_database(
        mapping_run["frames"],
        mapping_run["poses"],
        mapping_run["covariances"],
        mapping_run["calibration"],
        config=config,
        metadata={"mapping_recording": mapping.name},
    )

    cached = cache_visual_odometry(query, calibration_dir)
    if args.max_frames:
        cached = cached[: args.max_frames]

    poses_only = np.zeros((len(cached), 3))
    scratch = LocalizationFusion()
    previous = float(cached[0][0].timestamp)
    for index, (frame, measurement) in enumerate(cached):
        if measurement is None:
            scratch.predict_only(
                float(frame.timestamp),
                1.0 / 30.0 if index == 0 else float(frame.timestamp) - previous,
                translation_process_std=0.05,
                rotation_process_std=np.deg2rad(1.0),
            )
        else:
            scratch.process_odometry(measurement)
        previous = float(frame.timestamp)
        poses_only[index] = scratch.global_pose
    keyframes = set(keyframe_indices_causal(poses_only, config))

    fit = fit_reference_to_map(query, datum)
    reference_timestamps, reference_xyz, _ = reference_in_map_frame(query, fit)
    reference = (reference_timestamps, reference_xyz[:, :2])

    # Canonical config, cộng phần quét ngưỡng consensus nếu được yêu cầu. VO đã
    # cache nên mỗi cấu hình thêm gần như miễn phí.
    defaults = FusionConfig()
    configurations = [
        ("A_baseline_no_landmark", False, False, None),
        ("B_landmark", True, False, None),
        ("C_landmark_artificial_outage", True, True, None),
    ]
    if args.consensus_sweep:
        # Giữ nguyên cấu hình A và chỉ đổi cổng consensus, để tách riêng ảnh
        # hưởng của recovery policy khỏi ảnh hưởng của landmark.
        for count, radius in ((2, 4.0), (1, 4.0), (3, 8.0), (1, 8.0)):
            configurations.append(
                (
                    f"sweep_count{count}_radius{radius:g}",
                    False,
                    False,
                    replace(
                        defaults,
                        recovery_consensus_count=count,
                        recovery_consensus_radius_m=radius,
                    ),
                )
            )
    results = {}
    for name, use_landmark, artificial, fusion_config in configurations:
        replay = load_nmea_replay(
            query, alignment_mode="raw_enu", datum=datum.as_tuple()
        )
        replay.seek(float(cached[0][0].timestamp))
        matcher = (
            LandmarkMatcher(database, mapping_run["calibration"], config)
            if use_landmark
            else None
        )
        run = run_configuration(
            name, cached, replay, matcher, keyframes, segment,
            artificial_outage=artificial,
            fusion_config=fusion_config,
        )
        results[name] = evaluate(run, reference, segment)
        inside = results[name]["inside_cover"]
        cover = results[name]["cover_segment"]
        if not inside.get("samples"):
            print(f"{name:32s} chưa chạy tới đoạn có mái (smoke test?)")
            continue
        drift = cover["exit_drift_percent"]
        print(
            f"{name:32s} inside median {inside['median_m']:6.2f} m  "
            f"p95 {inside['p95_m']:6.2f} m  "
            f"exit drift {'n/a' if drift is None else f'{drift:5.2f}%'}  "
            f"landmark accepted {results[name]['landmark']['accepted']}"
        )
        relock = results[name]["relock_after_exit"]
        after = relock["error_after_2s_m"]
        stable = relock["time_to_stable_5m_seconds"]
        print(
            f"{'':32s} relock: error@2s "
            f"{'n/a' if after is None else f'{after:.2f} m'}  "
            f"stable<5m "
            f"{'không đạt trong 10 s' if stable is None else f'{stable:.2f} s'}"
        )

    report = {
        "kpi": {"pass": "demo hoạt động", "excellent": "quantitative report"},
        "query_recording": query.name,
        "mapping_recording": mapping.name,
        "database_keyframes": len(database),
        "segment": segment,
        "alignment": "raw_enu + datum chung (production Bước 3)",
        "registration_floor_m": REGISTRATION_FLOOR_M,
        "configurations": results,
        "limitations": [
            "A và B đo outage GPS THẬT (35,4 s). C cắt GPS nhân tạo trên toàn "
            "đoạn có mái (50,4 s) để dựng kịch bản thuần visual.",
            "Segment vào/ra hầm chốt ở gate #1 bằng exposure + ảnh biên, không "
            "trim theo error.",
            "Median toàn tuyến của recording này mong manh (gate #2: p45 5,49 m "
            "vs p55 9,96 m); dùng percentile curve và error theo đoạn 30 s.",
            "Dataset không có lux metadata nên không claim '<10 lux' hay IR.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nartifact: {args.output / 'benchmark.json'}")


if __name__ == "__main__":
    main()
