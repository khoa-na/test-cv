#!/usr/bin/env python3
"""Fuse YOLO pothole masks with stereo road-disparity geometry."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ops

from stereo_sgbm import (
    compute_disparity,
    fit_road_disparity,
    measure_pothole,
    segment_residual,
)


def fuse_mask(
    detection_mask: np.ndarray,
    residual_mask: np.ndarray,
    min_residual_coverage: float = 0.2,
) -> np.ndarray:
    overlap = detection_mask.astype(bool) & residual_mask.astype(bool)
    residual_area = np.count_nonzero(residual_mask)
    if residual_area < 20 or np.count_nonzero(overlap) / residual_area < min_residual_coverage:
        raise ValueError("YOLO mask không khớp stereo residual")
    return residual_mask.astype(bool)


def road_surface_area_mm2(
    mask: np.ndarray,
    road_disparity: np.ndarray,
    focal_px: float,
    baseline_mm: float,
) -> float:
    height, width = road_disparity.shape
    mask = mask.astype(bool)
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        raise ValueError("Mask quá nhỏ để tính diện tích")
    x1, x2 = max(0, xs.min() - 1), min(width, xs.max() + 2)
    y1, y2 = max(0, ys.min() - 1), min(height, ys.max() + 2)
    y, x = np.mgrid[y1:y2, x1:x2]
    disparity = np.maximum(road_disparity[y1:y2, x1:x2], 1e-6)
    z = focal_px * baseline_mm / disparity
    points = np.stack(
        (
            (x - (width - 1) / 2) * baseline_mm / disparity,
            (y - (height - 1) / 2) * baseline_mm / disparity,
            z,
        ),
        axis=-1,
    )
    step_x = np.gradient(points, axis=1)
    step_y = np.gradient(points, axis=0)
    pixel_area = np.linalg.norm(np.cross(step_x, step_y), axis=-1)
    return float(np.sum(pixel_area[mask[y1:y2, x1:x2]]))


class StereoYOLOPipeline:
    def __init__(
        self,
        detector_path: Path,
        focal_px: float,
        baseline_mm: float,
        metric_scale: float = 1.0,
        image_scale: float = 0.35,
        num_disparities: int = 128,
        confidence: float = 0.25,
        opencv_threads: int = 8,
    ):
        self.detector = YOLO(str(detector_path), task="segment")
        cv2.setNumThreads(min(opencv_threads, os.cpu_count() or 1))
        self.focal_px = focal_px
        self.baseline_mm = baseline_mm
        self.metric_scale = metric_scale
        self.image_scale = image_scale
        self.num_disparities = num_disparities
        self.confidence = confidence

    def predict(self, left: np.ndarray, right: np.ndarray) -> tuple[dict, np.ndarray]:
        started = time.perf_counter()
        result = self.detector.predict(
            left,
            imgsz=512,
            conf=self.confidence,
            iou=0.5,
            device="cpu",
            retina_masks=False,
            verbose=False,
        )[0]
        detection_ms = (time.perf_counter() - started) * 1000
        disparity_started = time.perf_counter()
        disparity, sgbm_ms = compute_disparity(
            left, right, self.image_scale, self.num_disparities
        )
        disparity_total_ms = (time.perf_counter() - disparity_started) * 1000

        road = fit_road_disparity(disparity)
        residual = road - disparity
        residual_mask, residual_threshold = segment_residual(
            residual, disparity > 0
        )
        geometry_ready = time.perf_counter()

        annotated = left.copy()
        potholes = []
        if result.masks is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            masks = (
                ops.scale_masks(
                    result.masks.data[:, None],
                    disparity.shape,
                    mode="nearest",
                )[:, 0]
                .cpu()
                .numpy()
            )
            for index, (raw_mask, box, score) in enumerate(
                zip(masks, boxes, confidences)
            ):
                detection_mask = raw_mask > 0.5
                try:
                    geometry_mask = fuse_mask(detection_mask, residual_mask)
                    intersection = np.count_nonzero(detection_mask & residual_mask)
                    union = np.count_nonzero(detection_mask | residual_mask)
                    geometry = measure_pothole(
                        disparity,
                        road,
                        geometry_mask,
                        self.focal_px * self.image_scale * self.metric_scale,
                        self.baseline_mm,
                    )
                    area_mm2 = road_surface_area_mm2(
                        geometry_mask,
                        road,
                        self.focal_px * self.image_scale * self.metric_scale,
                        self.baseline_mm,
                    )
                    measurement = {
                        "depth_mm": geometry["depth_mm_p90"],
                        "area_mm2": area_mm2,
                        "area_cm2": area_mm2 / 100,
                        "valid_depth_pixels": geometry["valid_depth_pixels"],
                        "residual_coverage": intersection
                        / np.count_nonzero(residual_mask),
                        "detection_residual_iou": intersection / union,
                        "metric_calibrated": self.metric_scale != 1.0,
                    }
                except ValueError as error:
                    measurement = {"error": str(error), "metric_calibrated": False}

                potholes.append(
                    {
                        "id": index,
                        "confidence": float(score),
                        "bbox": [float(value) for value in box],
                        **measurement,
                    }
                )
                x1, y1, x2, y2 = box.astype(int)
                color = (0, 200, 0) if "error" not in measurement else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{score:.2f}"
                if "depth_mm" in measurement:
                    label += f" {measurement['depth_mm']:.1f}mm {measurement['area_cm2']:.1f}cm2"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        finished = time.perf_counter()
        total_ms = (finished - started) * 1000
        return {
            "potholes": potholes,
            "count": len(potholes),
            "depth_type": "stereo_road_disparity",
            "residual_threshold_px": residual_threshold,
            "latency": {
                "detection_ms": detection_ms,
                "sgbm_ms": sgbm_ms,
                "disparity_total_ms": disparity_total_ms,
                "geometry_ready_ms": (geometry_ready - started) * 1000,
                "total_ms": total_ms,
                "fps": 1000 / total_ms,
            },
        }, annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + stereo pothole geometry")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--focal", type=float, required=True)
    parser.add_argument("--baseline-mm", type=float, required=True)
    parser.add_argument("--metric-scale", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--opencv-threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/stereo-yolo"))
    args = parser.parse_args()

    left = cv2.imread(str(args.left))
    right = cv2.imread(str(args.right))
    if left is None or right is None:
        raise FileNotFoundError("Không đọc được stereo pair")
    pipeline = StereoYOLOPipeline(
        args.detector,
        args.focal,
        args.baseline_mm,
        args.metric_scale,
        args.scale,
        args.num_disparities,
        args.confidence,
        args.opencv_threads,
    )
    for _ in range(args.warmup):
        pipeline.predict(left, right)
    report, annotated = pipeline.predict(left, right)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "result.jpg"), annotated)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
