#!/usr/bin/env python3
"""Train the documented Pothole-600 + PothRGBD segmentation recipe."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


BASE_MODEL = "yolo26n-seg.pt"
BASE_MODEL_SHA256 = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_default_base_model(model: object, requested: str) -> None:
    if requested != BASE_MODEL:
        return
    candidates = [Path(requested)]
    checkpoint_path = getattr(model, "ckpt_path", None)
    if checkpoint_path:
        candidates.append(Path(checkpoint_path))
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Ultralytics loaded {BASE_MODEL}, but its file cannot be verified")
    actual = sha256(resolved)
    if actual != BASE_MODEL_SHA256:
        raise ValueError(
            f"Base-model SHA-256 mismatch: expected {BASE_MODEL_SHA256}, got {actual}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(".cache/data/pothole600-pothrgbd-seg.yaml"),
    )
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/yolo26n-seg-merged")
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--copy-paste", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {args.data}. Run "
            "python -m data_tools.build_combined_seg_dataset first."
        )
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs and batch must be positive; imgsz must be at least 32")
    if not 0.0 <= args.copy_paste <= 1.0:
        raise ValueError("copy-paste must be between 0 and 1")

    from ultralytics import YOLO

    model = YOLO(args.model, task="segment")
    verify_default_base_model(model, args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
        cache=True,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        cls_remap=True,
        close_mosaic=10,
        copy_paste=args.copy_paste,
        project=str(args.output),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
