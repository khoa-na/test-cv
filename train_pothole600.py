#!/usr/bin/env python3
"""Prepare Pothole-600 masks and fine-tune YOLO26n-seg."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def find_dataset_root(path: Path) -> Path:
    for candidate in (path, path / "pothole600"):
        if (candidate / "training/rgb").is_dir() and (candidate / "training/label").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Không tìm thấy Pothole-600 bên dưới {path}")


def mask_to_labels(mask_path: Path, min_area: int, epsilon: float) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    height, width = mask.shape
    # ponytail: YOLO polygons cannot represent holes; external contours fit these binary masks.
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        polygon = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        normalized = polygon.astype(float) / np.array([width, height])
        coordinates = " ".join(f"{value:.8f}" for value in np.clip(normalized, 0, 1).ravel())
        labels.append(f"0 {coordinates}")
    return labels


def prepare_dataset(source: Path, output: Path, min_area: int, epsilon: float) -> Path:
    dataset = output / "dataset"
    totals = {}
    for source_split, target_split in (("training", "train"), ("validation", "val"), ("testing", "test")):
        image_dir = dataset / "images" / target_split
        label_dir = dataset / "labels" / target_split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        images = instances = 0
        for image_path in sorted((source / source_split / "rgb").glob("*.png")):
            link = image_dir / image_path.name
            if not link.exists():
                link.symlink_to(image_path.resolve())
            labels = mask_to_labels(source / source_split / "label" / image_path.name, min_area, epsilon)
            (label_dir / f"{image_path.stem}.txt").write_text("\n".join(labels), encoding="utf-8")
            images += 1
            instances += len(labels)
        totals[target_split] = (images, instances)

    yaml_path = dataset / "pothole600-seg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "pothole"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split, (images, instances) in totals.items():
        print(f"{split}: {images} images, {instances} instances")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Thư mục Pothole-600 đã giải nén")
    parser.add_argument("--output", type=Path, default=Path("artifacts/yolo26n-seg"))
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--weights", type=Path, help="Detection weights used to initialize the segmentation backbone")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Ví dụ: 0 cho GPU, cpu cho CPU; mặc định tự chọn")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--polygon-epsilon", type=float, default=0.002)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.min_area < 1:
        parser.error("epochs, batch, min-area phải > 0 và imgsz phải >= 32")
    if not 0 <= args.polygon_epsilon <= 0.1:
        parser.error("polygon-epsilon phải nằm trong khoảng 0..0.1")

    source = find_dataset_root(args.dataset)
    output = args.output.resolve()
    dataset_yaml = prepare_dataset(source, output, args.min_area, args.polygon_epsilon)
    if args.prepare_only:
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
        cache=True,
        seed=42,
        deterministic=True,
        project=str(output),
        name="train",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
