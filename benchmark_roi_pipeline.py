import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from audit_pothrgbd import find_dataset_root
from benchmark_depth_pothrgbd import geometry_ready_samples, polygon_masks, split_name
from depth_regressor_inference import DepthRegressorONNX
from pothole_pipeline import estimate_geometry


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 0.0


def greedy_match(
    predicted: list[np.ndarray], target: list[np.ndarray], threshold: float
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (mask_iou(predicted[pred_index], target[target_index]), pred_index, target_index)
            for pred_index in range(len(predicted))
            for target_index in range(len(target))
        ),
        reverse=True,
    )
    matches = []
    used_predictions, used_targets = set(), set()
    for iou, pred_index, target_index in candidates:
        if iou < threshold:
            break
        if pred_index in used_predictions or target_index in used_targets:
            continue
        matches.append((pred_index, target_index, iou))
        used_predictions.add(pred_index)
        used_targets.add(target_index)
    return matches


def error_metrics(errors: list[float]) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "instances": len(values),
        "mean_relative_error": float(values.mean()) if values.size else None,
        "median_relative_error": float(np.median(values)) if values.size else None,
        "within_15_percent": float(np.mean(values <= 0.15)) if values.size else None,
        "within_8_percent": float(np.mean(values <= 0.08)) if values.size else None,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    root = find_dataset_root(args.dataset)
    samples = [
        sample
        for sample in geometry_ready_samples(root)
        if split_name(sample[0], args.test_fraction, args.seed) == "test"
    ]
    if args.limit:
        samples = samples[: args.limit]
    detector = YOLO(str(args.detector), task="segment")
    regressor = DepthRegressorONNX(
        args.depth_model,
        args.depth_size,
        args.depth_threads,
        args.reliable_min,
        args.reliable_max,
    )

    warm_image = cv2.imread(str(samples[0][1]))
    detector.predict(
        warm_image,
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=0.5,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )
    warm_mask = polygon_masks(samples[0][3], warm_image.shape[:2])[0]
    regressor.predict(warm_image, warm_mask)

    total_gt = total_predictions = matched = eligible_gt = 0
    depth_errors, reliable_depth_errors, area_errors = [], [], []
    latencies, rows = [], []
    for sample_index, (key, image_path, depth_path, label_path) in enumerate(samples, 1):
        image = cv2.imread(str(image_path))
        sensor_depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
        sensor_depth[(sensor_depth <= 0) | (sensor_depth >= 65535)] = np.nan
        targets = polygon_masks(label_path, image.shape[:2])

        started = time.perf_counter()
        result = detector.predict(
            image,
            imgsz=args.imgsz,
            conf=args.confidence,
            iou=0.5,
            device="cpu",
            retina_masks=True,
            verbose=False,
        )[0]
        predictions = []
        if result.masks is not None:
            for raw_mask in result.masks.data.cpu().numpy():
                mask = raw_mask > 0.5
                if mask.shape != image.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                predictions.append(mask)
        matches = greedy_match(predictions, targets, args.match_iou)
        total_gt += len(targets)
        total_predictions += len(predictions)
        matched += len(matches)

        for pred_index, target_index, iou in matches:
            try:
                gt = estimate_geometry(
                    targets[target_index], sensor_depth, depth_direction=1
                )
                prediction = regressor.predict(image, predictions[pred_index])
            except ValueError:
                continue
            target_ratio = gt["relative_depth"] / max(
                abs(gt["road_depth_median"]), 1e-6
            )
            if (
                gt["relative_depth"] < args.min_depth
                or target_ratio < args.min_depth_ratio
            ):
                continue
            eligible_gt += 1
            depth_error = (
                abs(prediction["relative_depth"] - gt["relative_depth"])
                / gt["relative_depth"]
            )
            area_error = (
                abs(
                    np.count_nonzero(predictions[pred_index])
                    - np.count_nonzero(targets[target_index])
                )
                / np.count_nonzero(targets[target_index])
            )
            depth_errors.append(depth_error)
            area_errors.append(area_error)
            if prediction["depth_reliable"]:
                reliable_depth_errors.append(depth_error)
            rows.append(
                {
                    "sample_id": key,
                    "mask_iou": iou,
                    "gt_depth": gt["relative_depth"],
                    "predicted_depth": prediction["relative_depth"],
                    "depth_relative_error": depth_error,
                    "area_relative_error": area_error,
                    "depth_reliable": prediction["depth_reliable"],
                }
            )
        latencies.append((time.perf_counter() - started) * 1000)
        if args.progress and (
            sample_index % args.progress == 0 or sample_index == len(samples)
        ):
            print(f"{sample_index}/{len(samples)} images", flush=True)

    median_ms = float(np.median(latencies))
    report = {
        "dataset": str(root.resolve()),
        "detector": str(args.detector.resolve()),
        "depth_model": str(args.depth_model.resolve()),
        "settings": {
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "match_iou": args.match_iou,
            "reliable_depth_range": [args.reliable_min, args.reliable_max],
        },
        "detection": {
            "images": len(samples),
            "gt_instances": total_gt,
            "predicted_instances": total_predictions,
            "matched_instances": matched,
            "gt_match_coverage": matched / total_gt if total_gt else 0.0,
            "prediction_match_precision": matched / total_predictions
            if total_predictions
            else 0.0,
        },
        "depth": {
            "eligible_matched_instances": eligible_gt,
            "all_matched": error_metrics(depth_errors),
            "reliable_only": {
                **error_metrics(reliable_depth_errors),
                "coverage_of_eligible_matches": len(reliable_depth_errors) / eligible_gt
                if eligible_gt
                else 0.0,
            },
        },
        "relative_area": error_metrics(area_errors),
        "latency": {
            "median_ms": median_ms,
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": 1000.0 / median_ms,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if rows:
        with (args.output / "instances.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO + ROI depth on PothRGBD")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/roi-pipeline-benchmark"))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--depth-size", type=int, default=128)
    parser.add_argument("--depth-threads", type=int, default=6)
    parser.add_argument("--reliable-min", type=float, default=32.46)
    parser.add_argument("--reliable-max", type=float, default=59.48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-depth", type=float, default=5.0)
    parser.add_argument("--min-depth-ratio", type=float, default=0.005)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
