import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def compute_disparity(
    left: np.ndarray, right: np.ndarray, scale: float = 0.5, num_disparities: int = 256
) -> tuple[np.ndarray, float]:
    if left.shape != right.shape:
        raise ValueError("Ảnh stereo phải cùng kích thước")
    if num_disparities < 16 or num_disparities % 16:
        raise ValueError("num_disparities phải là bội số của 16")
    size = (round(left.shape[1] * scale), round(left.shape[0] * scale))
    left_gray = cv2.resize(
        cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    )
    right_gray = cv2.resize(
        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    )
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        disp12MaxDiff=2,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    started = time.perf_counter()
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16
    return disparity, (time.perf_counter() - started) * 1000


def fit_road_disparity(
    disparity: np.ndarray, stride: int = 16, iterations: int = 4, keep_fraction: float = 0.8
) -> np.ndarray:
    valid = np.isfinite(disparity) & (disparity > 0)
    y, x = np.nonzero(valid)
    if x.size < 100:
        raise ValueError("Không đủ valid disparity để fit mặt đường")
    height, width = disparity.shape
    x, y = x[::stride], y[::stride]
    design = np.column_stack(
        (x / max(width - 1, 1), y / max(height - 1, 1), np.ones(x.size))
    )
    values = disparity[y, x]
    for _ in range(iterations):
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        errors = values - design @ coefficients
        centered = np.abs(errors - np.median(errors))
        keep = centered <= np.quantile(centered, keep_fraction)
        design, values = design[keep], values[keep]
    yy, xx = np.mgrid[:height, :width]
    return (
        coefficients[0] * xx / max(width - 1, 1)
        + coefficients[1] * yy / max(height - 1, 1)
        + coefficients[2]
    ).astype(np.float32)


def segment_residual(
    residual: np.ndarray, valid: np.ndarray, quantile: float = 0.995
) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(residual[valid], quantile))
    candidate = (valid & (residual >= threshold)).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    if count < 2:
        raise ValueError("Không tìm thấy residual component")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest, threshold


def measure_pothole(
    disparity: np.ndarray,
    road_disparity: np.ndarray,
    mask: np.ndarray,
    focal_px: float,
    baseline_mm: float,
) -> dict:
    valid = mask & (disparity > 0) & (road_disparity > 0)
    if np.count_nonzero(valid) < 20:
        raise ValueError("Không đủ disparity trong pothole mask")
    actual_depth = focal_px * baseline_mm / disparity[valid]
    road_depth = focal_px * baseline_mm / road_disparity[valid]
    depression = actual_depth - road_depth
    return {
        "depth_mm_p50": float(np.percentile(depression, 50)),
        "depth_mm_p90": float(np.percentile(depression, 90)),
        "depth_mm_p95": float(np.percentile(depression, 95)),
        "road_distance_mm": float(np.median(road_depth)),
        "area_pixels_scaled": int(np.count_nonzero(mask)),
        "valid_depth_pixels": int(np.count_nonzero(valid)),
    }


def colorize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values[valid], (1, 99))
    normalized = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-inspired StereoSGBM pothole depth")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--focal", type=float, required=True, help="Focal length ở full resolution")
    parser.add_argument("--baseline-mm", type=float, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument("--residual-quantile", type=float, default=0.995)
    parser.add_argument("--output", type=Path, default=Path("artifacts/stereo-sgbm"))
    args = parser.parse_args()

    left, right = cv2.imread(str(args.left)), cv2.imread(str(args.right))
    if left is None or right is None:
        raise FileNotFoundError("Không đọc được stereo pair")
    started = time.perf_counter()
    disparity, sgbm_ms = compute_disparity(
        left, right, args.scale, args.num_disparities
    )
    valid = disparity > 0
    road_disparity = fit_road_disparity(disparity)
    residual = road_disparity - disparity
    mask, threshold = segment_residual(residual, valid, args.residual_quantile)
    geometry = measure_pothole(
        disparity,
        road_disparity,
        mask,
        args.focal * args.scale,
        args.baseline_mm,
    )
    total_ms = (time.perf_counter() - started) * 1000

    args.output.mkdir(parents=True, exist_ok=True)
    disparity_preview = colorize(disparity, valid)
    residual_preview = colorize(np.maximum(residual, 0), valid)
    annotated = cv2.resize(left, (disparity.shape[1], disparity.shape[0]))
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(args.output / "disparity.jpg"), disparity_preview)
    cv2.imwrite(str(args.output / "residual.jpg"), residual_preview)
    cv2.imwrite(str(args.output / "mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / "annotated.jpg"), annotated)
    report = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "method": "StereoSGBM + robust road disparity transformation",
        "calibration": {
            "focal_px_full_resolution": args.focal,
            "baseline_mm": args.baseline_mm,
            "scale": args.scale,
            "metric_status": "approximate_until_rectified_focal_is_verified",
        },
        "disparity": {
            "shape": list(disparity.shape),
            "valid_fraction": float(np.mean(valid)),
            "residual_threshold": threshold,
        },
        "geometry": geometry,
        "latency": {
            "sgbm_ms": sgbm_ms,
            "total_ms": total_ms,
            "sgbm_fps": 1000 / sgbm_ms,
            "total_fps": 1000 / total_ms,
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
