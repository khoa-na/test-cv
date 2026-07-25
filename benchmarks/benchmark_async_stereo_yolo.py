#!/usr/bin/env python3
"""Compare sequential and two-stage CPU throughput on one stereo group."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from benchmarks.benchmark_fan_stereo import CALIBRATION, paired_images
from pipelines.stereo_yolo_pipeline import StereoYOLOPipeline


def cpu_time() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def summarize_run(
    wall_seconds: float, cpu_seconds: float, latencies_ms: list[float]
) -> dict:
    logical_cpus = os.cpu_count() or 1
    core_equivalents = cpu_seconds / wall_seconds
    return {
        "frames": len(latencies_ms),
        "throughput_fps": len(latencies_ms) / wall_seconds,
        "latency_p50_ms": float(np.median(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "cpu_core_equivalents": core_equivalents,
        "total_cpu_percent": 100 * core_equivalents / logical_cpus,
    }


def output_signature(report: dict) -> tuple:
    return tuple(
        (
            round(item.get("depth_mm", -1), 5),
            round(item.get("area_mm2", -1), 5),
            tuple(round(value, 3) for value in item["bbox"]),
        )
        for item in report["potholes"]
    )


def run_sequential(
    pipeline: StereoYOLOPipeline, frames: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[dict, list[tuple]]:
    started = time.perf_counter()
    cpu_started = cpu_time()
    latencies = []
    signatures = []
    for left, right in frames:
        frame_started = time.perf_counter()
        report, _ = pipeline.predict(left, right)
        latencies.append((time.perf_counter() - frame_started) * 1000)
        signatures.append(output_signature(report))
    wall_seconds = time.perf_counter() - started
    return (
        summarize_run(wall_seconds, cpu_time() - cpu_started, latencies),
        signatures,
    )


def run_staged(
    pipeline: StereoYOLOPipeline,
    frames: list[tuple[np.ndarray, np.ndarray]],
    queue_depth: int,
) -> tuple[dict, list[tuple]]:
    pending: deque = deque()
    latencies = []
    signatures = []
    next_index = 0
    started = time.perf_counter()
    cpu_started = cpu_time()

    with (
        ThreadPoolExecutor(max_workers=1) as detector_worker,
        ThreadPoolExecutor(max_workers=1) as geometry_worker,
    ):

        def submit(index: int) -> None:
            left, right = frames[index]
            pending.append(
                (
                    time.perf_counter(),
                    left,
                    detector_worker.submit(pipeline.detect, left),
                    geometry_worker.submit(pipeline.estimate_geometry, left, right),
                )
            )

        while next_index < min(queue_depth, len(frames)):
            submit(next_index)
            next_index += 1

        while pending:
            submitted, left, detection_future, geometry_future = pending.popleft()
            result, _ = detection_future.result()
            geometry_state = geometry_future.result()
            report, _, _ = pipeline.fuse(left, result, geometry_state)
            latencies.append((time.perf_counter() - submitted) * 1000)
            signatures.append(output_signature(report))
            if next_index < len(frames):
                submit(next_index)
                next_index += 1

    wall_seconds = time.perf_counter() - started
    summary = summarize_run(
        wall_seconds, cpu_time() - cpu_started, latencies
    )
    summary["queue_depth"] = queue_depth
    summary["drop_ratio"] = 0.0
    return summary, signatures


def load_frames(dataset: Path, group: str, repeats: int) -> list:
    root = dataset / "dataset" if (dataset / "dataset").is_dir() else dataset
    frames = []
    for left_path, right_path in paired_images(root / group):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            raise FileNotFoundError(f"Cannot read {left_path} or {right_path}")
        frames.append((left, right))
    return frames * repeats


def run(args: argparse.Namespace) -> dict:
    frames = load_frames(args.dataset, args.group, args.repeats)
    calibration = CALIBRATION[args.group]
    pipeline = StereoYOLOPipeline(
        detector_path=args.detector,
        focal_px=calibration["focal_px"],
        baseline_mm=calibration["baseline_mm"],
        metric_scale=args.metric_scale,
        image_scale=args.scale,
        num_disparities=args.num_disparities,
        confidence=args.confidence,
        opencv_threads=args.opencv_threads,
        min_alignment_iou=args.min_alignment_iou,
        area_quantile=args.area_quantile,
        area_scale=args.area_scale,
    )
    for _ in range(args.warmup):
        pipeline.predict(*frames[0])

    sequential_runs = []
    staged_runs = []
    equivalent = True
    for round_index in range(args.rounds):
        runners = (
            (run_sequential, run_staged)
            if round_index % 2 == 0
            else (run_staged, run_sequential)
        )
        results = {}
        for runner in runners:
            if runner is run_staged:
                summary, signatures = runner(
                    pipeline, frames, args.queue_depth
                )
                results["staged"] = (summary, signatures)
            else:
                summary, signatures = runner(pipeline, frames)
                results["sequential"] = (summary, signatures)
        sequential_runs.append(results["sequential"][0])
        staged_runs.append(results["staged"][0])
        equivalent &= results["sequential"][1] == results["staged"][1]

    sequential_fps = float(
        np.median([item["throughput_fps"] for item in sequential_runs])
    )
    staged_fps = float(
        np.median([item["throughput_fps"] for item in staged_runs])
    )
    result = {
        "method": "bounded two-stage detector + stereo CPU pipeline",
        "group": args.group,
        "opencv_threads": args.opencv_threads,
        "logical_cpus": os.cpu_count() or 1,
        "sequential": {
            "median_throughput_fps": sequential_fps,
            "runs": sequential_runs,
        },
        "staged": {
            "median_throughput_fps": staged_fps,
            "runs": staged_runs,
        },
        "throughput_gain_percent": 100 * (staged_fps / sequential_fps - 1),
        "outputs_equivalent": equivalent,
        "note": "Offline benchmark: input is demand-driven, so no frames are dropped.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(".cache/data/fan-stereo-pothole")
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("models/pothole_yolo26n_seg.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/async-stereo-yolo/benchmark.json"),
    )
    parser.add_argument("--group", choices=CALIBRATION, default="model2")
    parser.add_argument("--metric-scale", type=float, default=0.8334711918061039)
    parser.add_argument("--area-scale", type=float, default=1.3755604448201877)
    parser.add_argument("--scale", type=float, default=0.3125)
    parser.add_argument("--num-disparities", type=int, default=112)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-alignment-iou", type=float, default=0.1)
    parser.add_argument("--area-quantile", type=float, default=0.986)
    parser.add_argument("--opencv-threads", type=int, default=4)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
