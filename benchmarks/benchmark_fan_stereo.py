#!/usr/bin/env python3
"""Benchmark the CPU stereo baseline on Fan et al.'s pothole dataset.

Model 1 calibrates one global metric scale. Models 2 and 3 remain held out.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from pipelines.stereo_sgbm import (
    compute_disparity,
    fit_road_disparity,
    measure_pothole,
    segment_residual,
)


CALIBRATION = {
    "model1": {"focal_px": 1384.24964, "baseline_mm": 119.4380934},
    "model2": {"focal_px": 1384.24964, "baseline_mm": 119.4380934},
    "model3": {"focal_px": 1391.96062, "baseline_mm": 119.4931030},
}


def read_binary_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header_lines: list[str] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Only binary little-endian PLY is supported: {path}")
        vertex_line = next(line for line in header_lines if line.startswith("element vertex "))
        count = int(vertex_line.split()[-1])
        vertex_start = header_lines.index(vertex_line)
        properties = [
            line.split()[-1]
            for line in header_lines[vertex_start + 1 :]
            if line.startswith("property ")
        ]
        if properties[:6] != ["x", "y", "z", "red", "green", "blue"]:
            raise ValueError(f"Unexpected PLY vertex layout: {path}")

        record = struct.Struct("<fffBBB")
        raw = file.read(count * record.size)
        if len(raw) != count * record.size:
            raise ValueError(f"Truncated PLY data: {path}")
        points = np.empty((count, 3), dtype=np.float32)
        for index, values in enumerate(record.iter_unpack(raw)):
            points[index] = values[:3]
        return points


def robust_z_extent(points: np.ndarray) -> float:
    return float(np.percentile(points[:, 2], 99.5) - np.percentile(points[:, 2], 0.5))


def paired_images(model_dir: Path) -> list[tuple[Path, Path]]:
    left_dir = model_dir / "left images"
    right_dir = model_dir / "right images"
    left = sorted(left_dir.glob("L*.png"), key=lambda path: int(path.stem[1:]))
    right_by_id = {int(path.stem[1:]): path for path in right_dir.glob("R*.png")}
    pairs = [(path, right_by_id[int(path.stem[1:])]) for path in left]
    if not pairs:
        raise ValueError(f"No stereo pairs found in {model_dir}")
    return pairs


def benchmark_pair(
    left_path: Path,
    right_path: Path,
    focal_px: float,
    baseline_mm: float,
    image_scale: float,
    num_disparities: int,
) -> dict[str, float]:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise ValueError(f"Cannot read stereo pair {left_path}, {right_path}")

    started = time.perf_counter()
    disparity, sgbm_ms = compute_disparity(
        left, right, image_scale, num_disparities
    )
    road = fit_road_disparity(disparity)
    residual = road - disparity
    mask, threshold = segment_residual(residual, disparity > 0)
    measurement = measure_pothole(
        disparity, road, mask, focal_px * image_scale, baseline_mm
    )
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "raw_depth_mm": measurement["depth_mm_p90"],
        "area_px": measurement["area_pixels_scaled"],
        "threshold_px": threshold,
        "valid_fraction": float(np.mean(disparity > 0)),
        "sgbm_ms": sgbm_ms,
        "total_ms": total_ms,
    }


def run(args: argparse.Namespace) -> dict:
    dataset = args.dataset / "dataset" if (args.dataset / "dataset").is_dir() else args.dataset
    rows: list[dict] = []
    model_summary: dict[str, dict] = {}
    for model_name, calibration in CALIBRATION.items():
        model_dir = dataset / model_name
        laser_depths = [
            robust_z_extent(read_binary_ply_xyz(path))
            for path in sorted((model_dir / "gt").glob("*.ply"))
        ]
        if not laser_depths:
            raise ValueError(f"No laser scans found in {model_dir}")
        ground_truth = float(np.median(laser_depths))

        model_rows = []
        for left_path, right_path in paired_images(model_dir):
            row = {
                "model": model_name,
                "pair": left_path.stem[1:],
                "ground_truth_mm": ground_truth,
                **benchmark_pair(
                    left_path,
                    right_path,
                    calibration["focal_px"],
                    calibration["baseline_mm"],
                    args.scale,
                    args.num_disparities,
                ),
            }
            rows.append(row)
            model_rows.append(row)
        model_summary[model_name] = {
            "pairs": len(model_rows),
            "laser_scans": len(laser_depths),
            "ground_truth_mm": ground_truth,
            "raw_depth_median_mm": float(
                np.median([row["raw_depth_mm"] for row in model_rows])
            ),
        }

    metric_scale = (
        model_summary["model1"]["ground_truth_mm"]
        / model_summary["model1"]["raw_depth_median_mm"]
    )
    for row in rows:
        row["depth_mm"] = row["raw_depth_mm"] * metric_scale
        row["relative_error"] = (
            abs(row["depth_mm"] - row["ground_truth_mm"]) / row["ground_truth_mm"]
        )
    for model_name, summary in model_summary.items():
        selected = [row for row in rows if row["model"] == model_name]
        predicted = float(np.median([row["depth_mm"] for row in selected]))
        summary["depth_median_mm"] = predicted
        summary["relative_error"] = (
            abs(predicted - summary["ground_truth_mm"]) / summary["ground_truth_mm"]
        )

    test_rows = [row for row in rows if row["model"] in {"model2", "model3"}]
    test_errors = np.asarray([row["relative_error"] for row in test_rows])
    latencies = np.asarray([row["total_ms"] for row in rows])
    result = {
        "method": "StereoSGBM + robust road-disparity plane + residual segmentation",
        "calibration_group": "model1",
        "test_groups": ["model2", "model3"],
        "metric_scale": metric_scale,
        "ground_truth_definition": (
            "Per-model median of laser scan robust z extents: p99.5(z)-p0.5(z)"
        ),
        "model_summary": model_summary,
        "held_out": {
            "views": len(test_rows),
            "mean_relative_error": float(np.mean(test_errors)),
            "median_relative_error": float(np.median(test_errors)),
            "within_15_percent": float(np.mean(test_errors <= 0.15)),
            "within_8_percent": float(np.mean(test_errors <= 0.08)),
        },
        "performance": {
            "pairs": len(rows),
            "median_latency_ms": float(np.median(latencies)),
            "median_fps": float(1000 / np.median(latencies)),
        },
        "limitations": [
            "Laser truth is group-level because each physical pothole has multiple views and scans.",
            "The z-extent proxy is not the paper's ICP closest-point reconstruction metric.",
            "The model1 scale must be replaced by verified rectified-camera calibration in production.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fan_stereo_rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "fan_stereo_benchmark.json").write_text(
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
        "--output",
        type=Path,
        default=Path("artifacts/fan-stereo-benchmark"),
    )
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--num-disparities", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
