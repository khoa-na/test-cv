#!/usr/bin/env python3
"""Convert PothRGBD sang dataset YOLO-seg (images/labels train-val + YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from benchmarks.benchmark_depth_pothrgbd import split_name
from data_tools.audit_pothrgbd import find_dataset_root, index_files


def valid_label(label_path: Path) -> bool:
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        coordinates = values[1:]
        if (
            int(values[0]) != 0
            or len(coordinates) < 6
            or len(coordinates) % 2
            or not all(0.0 <= value <= 1.0 for value in coordinates)
        ):
            return False
    return True


def convert(source: Path, output: Path, val_fraction: float, seed: int) -> Path:
    root = find_dataset_root(source)
    images = index_files(sorted((root / "images").glob("*")))
    labels = index_files(sorted((root / "labels").glob("*.txt")))

    totals = {"train": [0, 0], "val": [0, 0]}
    skipped: list[str] = []
    for split in totals:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for key in sorted(set(images) & set(labels)):
        if len(images[key]) != 1 or len(labels[key]) != 1:
            skipped.append(f"{key}: pairing không duy nhất")
            continue
        image_path, label_path = images[key][0], labels[key][0]
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (480, 640):
            skipped.append(f"{key}: ảnh hỏng hoặc bị crop khác 480x640")
            continue
        if not valid_label(label_path):
            skipped.append(f"{key}: label không phải polygon YOLO-seg hợp lệ")
            continue
        # ponytail: val trùng test split của depth benchmark để không leak chéo task.
        split = "val" if split_name(key, val_fraction, seed) == "test" else "train"
        link = output / "images" / split / image_path.name
        if not link.exists():
            link.symlink_to(image_path.resolve())
        target = output / "labels" / split / f"{image_path.stem}.txt"
        target.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
        totals[split][0] += 1
        totals[split][1] += sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())

    yaml_path = output / "pothrgbd-seg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "pothole"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split, (count, instances) in totals.items():
        print(f"{split}: {count} images, {instances} instances")
    print(f"skipped: {len(skipped)}")
    for reason in skipped:
        print(f"  - {reason}")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PothRGBD sang YOLO-seg dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Thư mục PothRGBD đã giải nén")
    parser.add_argument("--output", type=Path, default=Path(".cache/data/pothrgbd-yoloseg"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction phải nằm trong khoảng (0, 1)")
    convert(args.dataset, args.output.resolve(), args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
