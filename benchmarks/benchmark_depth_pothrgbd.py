import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from data_tools.audit_pothrgbd import find_dataset_root, index_files, sample_id
from pipelines.depth_inference import DepthAnythingONNX
from pipelines.pothole_pipeline import estimate_geometry


def split_name(key: str, test_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if value < test_fraction else "calibration"


def polygon_masks(label_path: Path, shape: tuple[int, int]) -> list[np.ndarray]:
    height, width = shape
    masks = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        coordinates = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
        points = np.rint(coordinates * [width, height]).astype(np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return masks


def geometry_ready_samples(root: Path, min_valid_fraction: float = 0.8) -> list[tuple]:
    image_paths = sorted((root / "images").glob("*"))
    depth_paths = sorted((root / "depths").glob("*.npy"))
    label_paths = sorted((root / "labels").glob("*.txt"))
    images = index_files(image_paths)
    depths = index_files(depth_paths)
    labels = index_files(label_paths)
    samples = []
    for key in sorted(set(images) & set(depths) & set(labels)):
        if not (len(images[key]) == len(depths[key]) == len(labels[key]) == 1):
            continue
        image = cv2.imread(str(images[key][0]))
        if image is None or image.shape[:2] != (480, 640):
            continue
        sensor_depth = np.load(depths[key][0], allow_pickle=False)
        valid = (sensor_depth > 0) & (sensor_depth < np.iinfo(sensor_depth.dtype).max)
        if np.count_nonzero(valid) / valid.size < min_valid_fraction:
            continue
        samples.append((key, images[key][0], depths[key][0], labels[key][0]))
    return samples


def fit_scale(
    records: list[dict], target_key: str, prediction_key: str, min_target: float
) -> float:
    ratios = [
        record[target_key] / record[prediction_key]
        for record in records
        if record["split"] == "calibration"
        and record[target_key] >= min_target
        and record[prediction_key] > 1e-6
    ]
    if not ratios:
        raise ValueError("Không đủ calibration samples để fit depth scale")
    return float(np.median(ratios))


def evaluate(
    records: list[dict],
    target_key: str,
    prediction_key: str,
    scale: float,
    min_target: float,
) -> dict:
    test = [record for record in records if record["split"] == "test"]
    eligible = [
        record
        for record in test
        if record[target_key] >= min_target and record[prediction_key] > 1e-6
    ]
    errors = np.array(
        [
            abs(scale * record[prediction_key] - record[target_key]) / record[target_key]
            for record in eligible
        ],
        dtype=np.float64,
    )
    return {
        "test_instances": len(test),
        "evaluated_instances": len(eligible),
        "coverage": len(eligible) / len(test) if test else 0.0,
        "mean_absolute_relative_error": float(errors.mean()) if errors.size else None,
        "median_absolute_relative_error": float(np.median(errors)) if errors.size else None,
        "within_15_percent": float(np.mean(errors <= 0.15)) if errors.size else None,
        "within_8_percent": float(np.mean(errors <= 0.08)) if errors.size else None,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    root = find_dataset_root(args.dataset)
    samples = geometry_ready_samples(root, args.min_valid_fraction)
    if args.limit:
        samples = samples[: args.limit]
    model = DepthAnythingONNX(args.model, args.size, args.threads)
    records = []
    latencies = []

    for sample_index, (key, image_path, depth_path, label_path) in enumerate(samples, 1):
        image = cv2.imread(str(image_path))
        sensor_depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
        sensor_depth[(sensor_depth <= 0) | (sensor_depth >= 65535)] = np.nan
        started = time.perf_counter()
        predicted_depth = model.predict(image)
        latencies.append((time.perf_counter() - started) * 1000)
        for instance_index, mask in enumerate(polygon_masks(label_path, sensor_depth.shape)):
            try:
                gt = estimate_geometry(mask, sensor_depth, depth_direction=1)
                positive = estimate_geometry(mask, predicted_depth, depth_direction=1)
                negative = estimate_geometry(mask, predicted_depth, depth_direction=-1)
            except ValueError:
                continue
            records.append(
                {
                    "sample_id": key,
                    "instance": instance_index,
                    "split": split_name(key, args.test_fraction, args.seed),
                    "gt_depth": gt["relative_depth"],
                    "predicted_positive": positive["relative_depth"],
                    "predicted_negative": negative["relative_depth"],
                    "gt_depth_ratio": gt["relative_depth"] / max(abs(gt["road_depth_median"]), 1e-6),
                    "predicted_positive_ratio": positive["relative_depth"]
                    / max(abs(positive["road_depth_median"]), 1e-6),
                    "predicted_negative_ratio": negative["relative_depth"]
                    / max(abs(negative["road_depth_median"]), 1e-6),
                    "gt_area_pixels": gt["area_pixels"],
                }
            )
        if args.progress and (sample_index % args.progress == 0 or sample_index == len(samples)):
            print(f"{sample_index}/{len(samples)} images", flush=True)

    evaluations = {}
    for mode, target_key, suffix, min_target in (
        ("raw_sensor_depth", "gt_depth", "", args.min_gt_depth),
        ("ground_plane_normalized", "gt_depth_ratio", "_ratio", args.min_gt_depth_ratio),
    ):
        candidates = {}
        for direction, base_key in ((1, "predicted_positive"), (-1, "predicted_negative")):
            prediction_key = base_key + suffix
            scale = fit_scale(records, target_key, prediction_key, min_target)
            calibration_metrics = evaluate(
                [
                    {**record, "split": "test"}
                    for record in records
                    if record["split"] == "calibration"
                ],
                target_key,
                prediction_key,
                scale,
                min_target,
            )
            candidates[direction] = {
                "prediction_key": prediction_key,
                "scale": scale,
                "calibration_error": calibration_metrics["mean_absolute_relative_error"],
            }
        direction = min(candidates, key=lambda value: candidates[value]["calibration_error"])
        selected = candidates[direction]
        evaluations[mode] = {
            "depth_direction": direction,
            "scale": selected["scale"],
            "calibration_mean_absolute_relative_error": selected["calibration_error"],
            "test": evaluate(
                records,
                target_key,
                selected["prediction_key"],
                selected["scale"],
                min_target,
            ),
        }
    report = {
        "dataset": str(root.resolve()),
        "model": str(args.model.resolve()),
        "depth_type": "relative_scaled_to_realsense_raw_units",
        "metric_unit_verified": False,
        "settings": {
            "input_size": args.size,
            "threads": args.threads,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "min_valid_fraction": args.min_valid_fraction,
            "min_gt_depth": args.min_gt_depth,
            "min_gt_depth_ratio": args.min_gt_depth_ratio,
        },
        "counts": {
            "images": len(samples),
            "instances": len(records),
            "calibration_instances": sum(record["split"] == "calibration" for record in records),
            "test_instances": sum(record["split"] == "test" for record in records),
        },
        "evaluations": evaluations,
        "latency": {
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": 1000.0 / float(np.median(latencies)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output / "instances.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Depth Anything against PothRGBD")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/depth-benchmark"))
    parser.add_argument("--size", type=int, default=196)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--min-gt-depth", type=float, default=5.0)
    parser.add_argument("--min-gt-depth-ratio", type=float, default=0.005)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
