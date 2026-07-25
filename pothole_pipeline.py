import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from depth_inference import DepthAnythingONNX, colorize_depth


def fit_road_plane(depth: np.ndarray, ring: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(ring & np.isfinite(depth))
    if x.size < 30:
        raise ValueError("Không đủ depth pixels quanh ổ gà để fit mặt đường")
    h, w = depth.shape
    design = np.column_stack((x / max(w - 1, 1), y / max(h - 1, 1), np.ones(x.size)))
    values = depth[y, x]
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    keep = np.abs(residuals - np.median(residuals)) <= np.percentile(
        np.abs(residuals - np.median(residuals)), 80
    )
    coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
    return coefficients


def estimate_geometry(mask: np.ndarray, depth: np.ndarray, ring_width: int = 15) -> dict:
    mask = mask.astype(bool)
    if mask.shape != depth.shape:
        raise ValueError("Mask và depth phải cùng kích thước")
    if np.count_nonzero(mask) < 20:
        raise ValueError("Mask quá nhỏ để đo")
    kernel_size = ring_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
    road_plane = fit_road_plane(depth, ring)
    valid_inside = mask & np.isfinite(depth)
    inside_y, inside_x = np.nonzero(valid_inside)
    h, w = depth.shape
    expected_road = (
        road_plane[0] * inside_x / max(w - 1, 1)
        + road_plane[1] * inside_y / max(h - 1, 1)
        + road_plane[2]
    )
    residual = depth[inside_y, inside_x] - expected_road
    local_values = depth[(mask | ring) & np.isfinite(depth)]
    local_range = max(float(np.percentile(local_values, 95) - np.percentile(local_values, 5)), 1e-6)
    relative_depth = max(0.0, float(np.percentile(residual, 90)))
    area_pixels = int(np.count_nonzero(mask))
    return {
        "relative_depth": relative_depth,
        "depth_contrast": relative_depth / local_range,
        "relative_area": area_pixels / mask.size,
        "area_pixels": area_pixels,
        "road_depth_median": float(np.median(expected_road)),
        "valid_depth_pixels": int(np.count_nonzero(valid_inside)),
        "units": {
            "depth": "relative",
            "area": "image_fraction",
            "metric_calibrated": False,
        },
    }


def severity(geometry: dict) -> str:
    if geometry["depth_contrast"] >= 0.25 or geometry["relative_area"] >= 0.08:
        return "severe"
    if geometry["depth_contrast"] >= 0.10 or geometry["relative_area"] >= 0.03:
        return "moderate"
    return "minor"


class PotholePipeline:
    def __init__(
        self,
        detector_path: Path,
        depth_path: Path,
        depth_size: int = 196,
        confidence: float = 0.25,
        depth_threads: int = 6,
    ):
        self.detector = YOLO(str(detector_path), task="segment")
        self.depth = DepthAnythingONNX(depth_path, depth_size, depth_threads)
        self.confidence = confidence

    def predict(self, image: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
        started = time.perf_counter()
        result = self.detector.predict(
            image,
            imgsz=512,
            conf=self.confidence,
            iou=0.5,
            device="cpu",
            retina_masks=True,
            verbose=False,
        )[0]
        after_detection = time.perf_counter()
        depth = self.depth.predict(image)
        after_depth = time.perf_counter()

        annotated = image.copy()
        potholes = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for index, (raw_mask, box, score) in enumerate(zip(masks, boxes, confidences)):
                mask = raw_mask > 0.5
                if mask.shape != depth.shape:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                try:
                    geometry = estimate_geometry(mask, depth)
                except ValueError as error:
                    geometry = {"error": str(error)}
                    level = "unknown"
                else:
                    level = severity(geometry)
                potholes.append(
                    {
                        "id": index,
                        "confidence": float(score),
                        "bbox": [float(value) for value in box],
                        "severity": level,
                        "severity_calibrated": False,
                        **geometry,
                    }
                )
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(
                    annotated,
                    f"{score:.2f} {level}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

        finished = time.perf_counter()
        latency = {
            "detection_ms": (after_detection - started) * 1000,
            "depth_ms": (after_depth - after_detection) * 1000,
            "fusion_ms": (finished - after_depth) * 1000,
            "total_ms": (finished - started) * 1000,
        }
        latency["sequential_fps"] = 1000 / latency["total_ms"]
        return {
            "potholes": potholes,
            "count": len(potholes),
            "latency": latency,
            "depth_type": "relative",
            "metric_calibrated": False,
        }, annotated, depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Pothole segmentation + relative depth/area")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pipeline"))
    parser.add_argument("--depth-size", type=int, default=196)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")
    pipeline = PotholePipeline(
        args.detector, args.depth_model, args.depth_size, args.confidence
    )
    for _ in range(args.warmup):
        pipeline.predict(image)
    report, annotated, depth = pipeline.predict(image)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "result.jpg"), annotated)
    cv2.imwrite(str(args.output / "depth.jpg"), colorize_depth(depth))
    np.save(args.output / "relative_depth.npy", depth)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
