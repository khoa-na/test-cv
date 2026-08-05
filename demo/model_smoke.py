#!/usr/bin/env python3
"""Run a dataset-free CPU smoke inference with the tracked ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

CONFIG_ROOT = Path(tempfile.gettempdir()) / "pothole-portfolio-config"
(CONFIG_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CONFIG_ROOT / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CONFIG_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT / "ultralytics"))


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "pothole_yolo26n_seg.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_road(size: int) -> np.ndarray:
    """Create a deterministic textured road image for an interface smoke test."""
    rng = np.random.default_rng(20260805)
    vertical = np.linspace(82, 126, size, dtype=np.float32)[:, None, None]
    image = np.broadcast_to(vertical, (size, size, 3)).copy()
    image += rng.normal(0, 7, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    center = (size // 2, int(size * 0.68))
    axes = (int(size * 0.19), int(size * 0.075))
    cv2.ellipse(image, center, axes, -4, 0, 360, (35, 38, 40), -1)
    cv2.ellipse(image, center, axes, -4, 0, 360, (145, 145, 140), 3)
    cv2.line(image, (0, int(size * 0.57)), (size, int(size * 0.53)), (112, 112, 108), 2)
    return image


def run(model_path: Path, image_path: Path | None, size: int, output: Path | None) -> dict:
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if image_path is None:
        source = synthetic_road(size)
        source_description = "deterministic synthetic road"
    else:
        source = cv2.imread(str(image_path))
        if source is None:
            raise ValueError(f"Could not read image: {image_path}")
        source_description = str(image_path)

    started = time.perf_counter()
    model = YOLO(str(model_path), task="segment")
    results = model.predict(source=source, imgsz=size, device="cpu", verbose=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    result = results[0]

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result.plot()):
            raise RuntimeError(f"Could not write output image: {output}")

    boxes = 0 if result.boxes is None else len(result.boxes)
    masks = 0 if result.masks is None else len(result.masks)
    return {
        "status": "ok",
        "purpose": "interface smoke test; not an accuracy benchmark",
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
        },
        "input": {
            "source": source_description,
            "shape": list(source.shape),
            "inference_size": size,
        },
        "output": {
            "boxes": boxes,
            "masks": masks,
            "annotated_image": str(output) if output is not None else None,
        },
        "elapsed_ms_including_model_load": elapsed_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.model, args.image, args.size, args.output), indent=2))


if __name__ == "__main__":
    main()
