#!/usr/bin/env python3
"""Build the exact Pothole-600 + PothRGBD dataset manifest used for training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def split_counts(dataset: Path, split: str) -> dict[str, int]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    images = sorted(path for pattern in IMAGE_PATTERNS for path in image_dir.glob(pattern))
    labels = sorted(label_dir.glob("*.txt"))
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images/labels for {dataset} split {split}")
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)[:5]
        missing_images = sorted(label_stems - image_stems)[:5]
        raise ValueError(
            f"Unpaired {dataset}/{split}: missing labels={missing_labels}, "
            f"missing images={missing_images}"
        )
    instances = sum(
        sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
        for path in labels
    )
    return {"images": len(images), "instances": instances}


def build_manifest(pothole600: Path, pothrgbd: Path, output: Path) -> dict:
    pothole600 = pothole600.resolve()
    pothrgbd = pothrgbd.resolve()
    counts = {
        "pothole600_train": split_counts(pothole600, "train"),
        "pothole600_val": split_counts(pothole600, "val"),
        "pothole600_test": split_counts(pothole600, "test"),
        "pothrgbd_train": split_counts(pothrgbd, "train"),
        "pothrgbd_val_excluded": split_counts(pothrgbd, "val"),
    }
    manifest = {
        "path": "/",
        "train": [
            str(pothole600 / "images" / "train"),
            str(pothrgbd / "images" / "train"),
        ],
        "val": str(pothole600 / "images" / "val"),
        "test": str(pothole600 / "images" / "test"),
        "names": {0: "pothole"},
        "portfolio_manifest": {
            "split_policy": (
                "Pothole-600 train plus PothRGBD deterministic train subset; "
                "Pothole-600 validation/test remain evaluation-only"
            ),
            "counts": counts,
            "training_images": (
                counts["pothole600_train"]["images"]
                + counts["pothrgbd_train"]["images"]
            ),
            "training_instances": (
                counts["pothole600_train"]["instances"]
                + counts["pothrgbd_train"]["instances"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pothole600",
        type=Path,
        default=Path("artifacts/yolo26n-seg/dataset"),
    )
    parser.add_argument(
        "--pothrgbd",
        type=Path,
        default=Path(".cache/data/pothrgbd-yoloseg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/data/pothole600-pothrgbd-seg.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.pothole600, args.pothrgbd, args.output)
    summary = manifest["portfolio_manifest"]
    print(
        f"training: {summary['training_images']} images, "
        f"{summary['training_instances']} instances\nmanifest: {args.output}"
    )


if __name__ == "__main__":
    main()
