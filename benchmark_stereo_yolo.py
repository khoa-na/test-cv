#!/usr/bin/env python3
"""Benchmark YOLO + stereo fusion on all official Fan stereo pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark_fan_stereo import (
    CALIBRATION,
    paired_images,
    read_binary_ply_xyz,
    robust_z_extent,
)
from stereo_yolo_pipeline import StereoYOLOPipeline


def ground_truth_depth(model_dir: Path) -> float:
    depths = [
        robust_z_extent(read_binary_ply_xyz(path))
        for path in sorted((model_dir / "gt").glob("*.ply"))
    ]
    if not depths:
        raise ValueError(f"No laser scans found in {model_dir}")
    return float(np.median(depths))


def summarize(rows: list[dict]) -> dict:
    test = [row for row in rows if row["model"] in {"model2", "model3"}]
    fused = [row for row in test if row["fusion_success"]]
    errors = np.asarray([row["relative_error"] for row in fused])
    latencies = np.asarray([row["total_ms"] for row in rows])
    return {
        "pairs": len(rows),
        "detection_coverage": float(np.mean([row["detections"] > 0 for row in rows])),
        "fusion_coverage": float(np.mean([row["fusion_success"] for row in rows])),
        "strong_alignment_coverage": float(
            np.mean([row["strong_alignment"] for row in rows])
        ),
        "held_out": {
            "pairs": len(test),
            "fusion_coverage": len(fused) / len(test),
            "strong_alignment_coverage": float(
                np.mean([row["strong_alignment"] for row in test])
            ),
            "median_relative_error": float(np.median(errors)) if errors.size else None,
            "mean_relative_error": float(np.mean(errors)) if errors.size else None,
            "within_15_percent_of_fused": float(np.mean(errors <= 0.15))
            if errors.size
            else 0.0,
            "within_8_percent_of_fused": float(np.mean(errors <= 0.08))
            if errors.size
            else 0.0,
            "end_to_end_within_15_percent": float(
                np.mean(
                    [
                        row["fusion_success"]
                        and row["strong_alignment"]
                        and row["relative_error"] <= 0.15
                        for row in test
                    ]
                )
            ),
        },
        "performance": {
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "median_fps": float(1000 / np.median(latencies)),
            "min_fps": float(1000 / np.max(latencies)),
            "pairs_at_least_15_fps": float(np.mean(1000 / latencies >= 15)),
        },
    }


def run(args: argparse.Namespace) -> dict:
    dataset = args.dataset / "dataset" if (args.dataset / "dataset").is_dir() else args.dataset
    first_model_dir = dataset / "model1"
    first_left, first_right = paired_images(first_model_dir)[0]
    warm_left = cv2.imread(str(first_left))
    warm_right = cv2.imread(str(first_right))
    pipeline = StereoYOLOPipeline(
        args.detector,
        CALIBRATION["model1"]["focal_px"],
        CALIBRATION["model1"]["baseline_mm"],
        args.metric_scale,
        args.scale,
        args.num_disparities,
        args.confidence,
        args.opencv_threads,
    )
    for _ in range(args.warmup):
        pipeline.predict(warm_left, warm_right)

    args.output.mkdir(parents=True, exist_ok=True)
    failure_dir = args.output / "failures"
    failure_dir.mkdir(exist_ok=True)
    for stale_image in failure_dir.glob("*.jpg"):
        stale_image.unlink()
    rows = []
    for model_name, calibration in CALIBRATION.items():
        model_dir = dataset / model_name
        truth = ground_truth_depth(model_dir)
        pipeline.focal_px = calibration["focal_px"]
        pipeline.baseline_mm = calibration["baseline_mm"]
        for left_path, right_path in paired_images(model_dir):
            left = cv2.imread(str(left_path))
            right = cv2.imread(str(right_path))
            try:
                timings = []
                for _ in range(args.repeats):
                    report, annotated = pipeline.predict(left, right)
                    timings.append(report["latency"]["total_ms"])
                total_ms = float(np.median(timings))
                valid = [
                    pothole for pothole in report["potholes"] if "depth_mm" in pothole
                ]
                best = max(valid, key=lambda item: item["confidence"]) if valid else None
                relative_error = (
                    abs(best["depth_mm"] - truth) / truth if best is not None else None
                )
                fusion_success = best is not None
                strong_alignment = (
                    best is not None
                    and best["detection_residual_iou"] >= args.min_alignment_iou
                )
                status = (
                    "no_detection"
                    if report["count"] == 0
                    else "fusion_failed"
                    if not fusion_success
                    else "weak_alignment"
                    if not strong_alignment
                    else "depth_error"
                    if relative_error > 0.15
                    else "success"
                )
                row = {
                    "model": model_name,
                    "pair": left_path.stem[1:],
                    "ground_truth_mm": truth,
                    "status": status,
                    "detections": report["count"],
                    "fusion_success": fusion_success,
                    "strong_alignment": strong_alignment,
                    "confidence": best["confidence"] if best else None,
                    "depth_mm": best["depth_mm"] if best else None,
                    "relative_error": relative_error,
                    "area_cm2": best["area_cm2"] if best else None,
                    "residual_coverage": best["residual_coverage"] if best else None,
                    "detection_residual_iou": best["detection_residual_iou"]
                    if best
                    else None,
                    "total_ms": total_ms,
                    "fps": 1000 / total_ms,
                    "error": "",
                }
            except (ValueError, RuntimeError) as error:
                annotated = left
                row = {
                    "model": model_name,
                    "pair": left_path.stem[1:],
                    "ground_truth_mm": truth,
                    "status": "pipeline_error",
                    "detections": 0,
                    "fusion_success": False,
                    "strong_alignment": False,
                    "confidence": None,
                    "depth_mm": None,
                    "relative_error": None,
                    "area_cm2": None,
                    "residual_coverage": None,
                    "detection_residual_iou": None,
                    "total_ms": float("nan"),
                    "fps": float("nan"),
                    "error": str(error),
                }
            rows.append(row)
            if row["status"] != "success":
                cv2.imwrite(
                    str(failure_dir / f"{model_name}_{left_path.stem}.jpg"),
                    annotated,
                )
            print(
                f"{model_name}/{left_path.stem}: {row['status']} "
                f"{row['fps']:.1f} FPS"
            )

    result = {
        "method": "YOLO26n-seg ONNX + StereoSGBM road-disparity fusion",
        "calibration_group": "model1",
        "test_groups": ["model2", "model3"],
        "metric_scale": args.metric_scale,
        "strong_alignment_definition": (
            f"detection-residual IoU >= {args.min_alignment_iou}; "
            "heuristic proxy, not labeled-mask IoU"
        ),
        **summarize(rows),
        "limitations": [
            "Depth ground truth is the per-model laser z-extent proxy.",
            "Area has no independent ground truth in this benchmark.",
            "Only three physical potholes are represented.",
        ],
    }
    with (args.output / "rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/data/fan-stereo-pothole"),
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("models/pothole_yolo26n_seg.onnx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/stereo-yolo-benchmark"),
    )
    parser.add_argument("--metric-scale", type=float, default=0.8334711918061039)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-alignment-iou", type=float, default=0.1)
    parser.add_argument("--opencv-threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
