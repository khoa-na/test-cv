"""B7 — throughput của stack localization trên CPU.

Đo trên cùng một cửa sổ frame cố định để so được giữa các cấu hình thread.
Tách thời gian từng tầng vì con số tổng không cho biết còn bao nhiêu ngân sách
cho tầng landmark của Bước 3.

Chỉ chạy khi máy rảnh; render video hay benchmark khác chạy song song sẽ làm
số này sai lệch.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_landmark_reid import run_causal_pipeline
from data_tools.gps_sources import (
    load_nmea_replay,
    load_reference_trajectory,
    wrap_angle,
)
from data_tools.imu_yaw import load_imu_yaw_integrator
from data_tools.map_frame import first_quality4_datum
from data_tools.stereo_odometry import load_calibration, stereo_frames
from pipelines.landmark_db import (
    LandmarkConfig,
    LandmarkDatabase,
    LandmarkMatcher,
    build_database,
)
from pipelines.localization_ekf import LocalizationFusion, UTurnDetector
from pipelines.stereo_vo import StereoVO, StereoVOConfig

SEQUENCES = {
    "garage_2": "recording_2021-02-25_13-39-06",
    "garage_3": "recording_2021-05-10_19-15-19",
}


def percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(np.max(array)),
    }


def measure(
    recording: Path,
    calibration: Path,
    *,
    threads: int,
    frame_count: int,
    start_frame: int,
    yaw_source: str,
    database: LandmarkDatabase | None = None,
) -> dict:
    cv2.setNumThreads(threads)
    frames = stereo_frames(recording)[start_frame : start_frame + frame_count]
    if len(frames) < 2:
        raise ValueError("cần ít nhất 2 frame")

    timestamps, _, _ = load_reference_trajectory(recording)
    replay = load_nmea_replay(recording, alignment_mode="reference_rigid")
    replay.seek(float(timestamps[0]))

    vo = StereoVO(load_calibration(calibration), StereoVOConfig())
    imu_yaw = (
        load_imu_yaw_integrator(recording, frames[0].timestamp)
        if yaw_source == "imu"
        else None
    )
    fusion = LocalizationFusion()
    detector = UTurnDetector()
    landmark_config = LandmarkConfig()
    matcher = (
        LandmarkMatcher(database, load_calibration(calibration), landmark_config)
        if database is not None
        else None
    )

    read_ms: list[float] = []
    vo_ms: list[float] = []
    fusion_ms: list[float] = []
    landmark_ms: list[float] = []
    total_ms: list[float] = []
    dropouts = 0
    previous_timestamp = float(frames[0].timestamp)
    # Landmark chạy ở nhịp keyframe (mỗi 2 m), không phải mỗi frame — đó là hành
    # vi thật của hệ. Anchor cập nhật nhân quả trong chính vòng đo.
    keyframe_anchor = None
    landmark_queries = 0

    for index, frame in enumerate(frames):
        started = time.perf_counter()
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
        after_read = time.perf_counter()

        measurement = vo.process(left, right, frame.timestamp)
        if measurement is not None and imu_yaw is not None:
            imu_heading = imu_yaw.delta_heading(
                measurement.timestamp - measurement.dt, measurement.timestamp
            )
            if imu_heading is not None:
                measurement = replace(
                    measurement, dtheta=imu_heading, source="stereo_vo_imu_yaw"
                )
        after_vo = time.perf_counter()

        for gps_measurement in replay.pop_until(frame.timestamp):
            fusion.process_gps(gps_measurement)
        if measurement is None:
            dropouts += 1
            dt = (
                1.0 / 30.0
                if index == 0
                else float(frame.timestamp - previous_timestamp)
            )
            fusion.predict_only(
                frame.timestamp,
                dt,
                translation_process_std=0.05,
                rotation_process_std=np.deg2rad(1.0),
            )
        else:
            fusion.process_odometry(measurement)
        detector.update(frame.timestamp, float(fusion.global_pose[2]))
        after_fusion = time.perf_counter()

        pose = fusion.global_pose
        is_keyframe = keyframe_anchor is None or (
            float(np.linalg.norm(pose[:2] - keyframe_anchor[:2]))
            >= landmark_config.keyframe_distance_m
            or abs(float(wrap_angle(pose[2] - keyframe_anchor[2])))
            >= landmark_config.keyframe_rotation_rad
        )
        if matcher is not None and is_keyframe:
            keyframe_anchor = pose.copy()
            landmark_queries += 1
            result = matcher.query(
                left,
                right,
                prior_position=pose[:2],
                prior_covariance=fusion.global_position_covariance(),
            )
            if result.match is not None:
                fusion.update_position(
                    result.match.position, result.match.covariance, gate=True
                )
        elif matcher is not None and keyframe_anchor is None:
            keyframe_anchor = pose.copy()
        finished = time.perf_counter()
        previous_timestamp = float(frame.timestamp)

        read_ms.append((after_read - started) * 1000.0)
        vo_ms.append((after_vo - after_read) * 1000.0)
        fusion_ms.append((after_fusion - after_vo) * 1000.0)
        landmark_ms.append((finished - after_fusion) * 1000.0)
        total_ms.append((finished - started) * 1000.0)

    pipeline_ms = [
        vo + fuse + mark
        for vo, fuse, mark in zip(vo_ms, fusion_ms, landmark_ms)
    ]
    query_ms = [value for value in landmark_ms if value > 1.0]
    return {
        "threads": threads,
        "frames": len(frames),
        "vo_dropouts": dropouts,
        "landmark": {
            "enabled": matcher is not None,
            "database_keyframes": len(database) if database is not None else 0,
            "queries": landmark_queries,
            "query_rate_per_frame": landmark_queries / max(len(frames), 1),
            # Amortized: chi phí landmark trải trên MỌI frame localization
            # là đại lượng theo frame. ms/query là số phụ, quy ước đo cấm dùng
            # nó thay cho FPS.
            "amortized_ms_per_frame": float(np.mean(landmark_ms)) if landmark_ms else 0.0,
            "ms_per_query": percentiles(query_ms) if query_ms else None,
        },
        "stage_latency": {
            "image_read": percentiles(read_ms),
            "stereo_vo": percentiles(vo_ms),
            "fusion_and_events": percentiles(fusion_ms),
            "landmark": percentiles(landmark_ms),
            "total_with_read": percentiles(total_ms),
        },
        "fps": {
            # Camera driver cấp frame trong hệ thật, nên FPS chính không tính
            # thời gian đọc PNG từ đĩa.
            #
            # Số nộp B7 là throughput = frame / tổng thời gian. Landmark chỉ chạy
            # ở nhịp keyframe nên frame TRUNG VỊ không có chi phí landmark; lấy
            # median sẽ giấu sạch một module khỏi con số throughput.
            "throughput": len(frames) / (float(np.sum(pipeline_ms)) / 1000.0),
            "pipeline_median": 1000.0 / float(np.median(pipeline_ms)),
            "pipeline_p95": 1000.0 / float(np.percentile(pipeline_ms, 95)),
            "with_disk_read_median": 1000.0 / float(np.median(total_ms)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument("--sequence", choices=sorted(SEQUENCES), default="garage_2")
    parser.add_argument("--output", type=Path, default=Path("artifacts/system-fps-b7"))
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--start-frame", type=int, default=300)
    parser.add_argument("--yaw-source", choices=("visual", "imu"), default="imu")
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--landmark-db",
        type=Path,
        default=Path("artifacts/system-fps-b7/landmark_db.npz"),
        help="DB dựng từ garage_3; tạo tự động nếu chưa có.",
    )
    parser.add_argument(
        "--no-landmark",
        action="store_true",
        help="Bỏ tầng landmark; giữ để so với vòng đo trước khi có Bước 3.",
    )
    args = parser.parse_args()

    recording = args.dataset / SEQUENCES[args.sequence]
    calibration = args.dataset / "calibration"

    database = None
    if not args.no_landmark:
        if args.landmark_db.exists():
            database = LandmarkDatabase.load(args.landmark_db)
        else:
            mapping = args.dataset / SEQUENCES["garage_3"]
            print(f"dựng landmark DB từ {mapping.name} (một lần)...")
            datum = first_quality4_datum(mapping)
            mapping_run = run_causal_pipeline(mapping, calibration, datum)
            database = build_database(
                mapping_run["frames"],
                mapping_run["poses"],
                mapping_run["covariances"],
                mapping_run["calibration"],
                config=LandmarkConfig(),
                metadata={"mapping_recording": mapping.name},
            )
            args.landmark_db.parent.mkdir(parents=True, exist_ok=True)
            database.save(args.landmark_db)
        print(f"landmark DB: {len(database)} keyframes")

    load_before = os.getloadavg()
    if load_before[0] > 1.0:
        print(
            f"cảnh báo: load average {load_before[0]:.2f} trước khi đo, "
            "FPS sẽ thấp hơn thực tế"
        )
    runs = []
    for threads in args.threads:
        result = measure(
            recording,
            calibration,
            threads=threads,
            frame_count=args.frames,
            start_frame=args.start_frame,
            yaw_source=args.yaw_source,
            database=database,
        )
        runs.append(result)
        print(
            f"threads {threads:2d}  throughput {result['fps']['throughput']:5.1f} FPS"
            f"  (latency median {result['fps']['pipeline_median']:5.1f})"
            f"  vo {result['stage_latency']['stereo_vo']['median_ms']:5.2f} ms"
            f"  fusion {result['stage_latency']['fusion_and_events']['median_ms']:5.3f} ms"
            f"  landmark {result['landmark']['amortized_ms_per_frame']:5.2f} ms/frame"
        )

    best = max(runs, key=lambda item: item["fps"]["throughput"])
    report = {
        "kpi": {"target_fps": 15.0, "good_fps": 20.0},
        "headline_metric": (
            "fps.throughput = frame / tổng thời gian, gồm cả tầng landmark. "
            "Median che chi phí landmark vì landmark chỉ chạy ở nhịp keyframe."
        ),
        "sequence": args.sequence,
        "recording": recording.name,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "opencv": cv2.__version__,
            # Máy bận làm FPS tụt ~10%; ghi lại để đọc artifact biết số đo trong
            # điều kiện nào, không phải tin lời mô tả.
            "load_average_before": load_before,
            "load_average_after": os.getloadavg(),
        },
        "runs": runs,
        "headline": {
            "threads": best["threads"],
            "throughput_fps": best["fps"]["throughput"],
            "landmark_included": database is not None,
            "passes_target": best["fps"]["throughput"] >= 15.0,
            "reaches_good": best["fps"]["throughput"] >= 20.0,
        },
        "notes": [
            "FPS chính không tính thời gian đọc PNG từ đĩa; hệ thật nhận frame "
            "từ camera driver. Cột with_disk_read_median cho biết chi phí đó.",
            "Stack pothole perception được đo riêng bởi benchmark_stereo_yolo "
            "và không nằm trong throughput localization này.",
            (
                "Vòng đo gồm stereo VO, EKF + integrity, U-turn detector và "
                "landmark query ở nhịp keyframe."
                if database is not None
                else "Chạy với --no-landmark: tầng landmark KHÔNG nằm trong số này."
            ),
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nheadline: {best['fps']['throughput']:.1f} FPS @ {best['threads']} threads"
        f" (landmark {'in' if database is not None else 'OUT'})"
        f"  -> {args.output / 'benchmark.json'}"
    )


if __name__ == "__main__":
    main()
