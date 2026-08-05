"""A1 — receipt máy đọc được cho con số headline của Phần A.

Report trích 89,8% box mAP@0.5 nhưng lần chạy trước chỉ để lại PNG (PR curve,
confusion matrix). Không có file nào ghi model SHA, split, phiên bản thư viện
hay cấu hình chạy, nên con số không tự truy vết được. Script này đóng đúng
khoảng trống đó.

Chạy trên Pothole-600 official test split (180 ảnh / 196 instance). Split này
không tham gia gradient update, nhưng đã được dùng để đối chiếu nhiều
export/checkpoint, nên receipt ghi rõ điều đó thay vì gọi nó là test hoàn toàn
chưa quan sát.

Ngoài metric, script lưu các mẫu xếp theo IoU để đóng khoảng trống bằng chứng
định tính của A4: false negative, false positive và lỗi biên mask — thay vì ba
batch đầu vốn toàn ca thành công.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_YAML = Path(".cache/data/pothole600-pothrgbd-seg.yaml")
IOU_MATCH = 0.5


def file_digest(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def git_receipt() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": dirty}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Đọc polygon YOLO-seg và quy về bbox tuyệt đối."""
    if not label_path.exists():
        return np.zeros((0, 4))
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coordinates = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        boxes.append(
            [
                coordinates[:, 0].min(), coordinates[:, 1].min(),
                coordinates[:, 0].max(), coordinates[:, 1].max(),
            ]
        )
    return np.asarray(boxes) if boxes else np.zeros((0, 4))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def collect_failures(model, images: list[Path], labels_dir: Path, confidence: float):
    """Xếp hạng ảnh theo loại lỗi để lưu mẫu đại diện, không lấy ba batch đầu."""
    records = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = load_boxes(labels_dir / f"{image_path.stem}.txt", width, height)
        result = model.predict(image, conf=confidence, verbose=False)[0]
        predicted = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None and len(result.boxes)
            else np.zeros((0, 4))
        )
        overlap = iou_matrix(predicted, truth)
        matched_truth = set()
        matched_pred = set()
        if overlap.size:
            for pred_index in np.argsort(-overlap.max(axis=1)):
                truth_index = int(np.argmax(overlap[pred_index]))
                if (
                    overlap[pred_index, truth_index] >= IOU_MATCH
                    and truth_index not in matched_truth
                ):
                    matched_truth.add(truth_index)
                    matched_pred.add(int(pred_index))
        false_negative = len(truth) - len(matched_truth)
        false_positive = len(predicted) - len(matched_pred)
        boundary = (
            float(np.min([overlap[p].max() for p in matched_pred]))
            if matched_pred
            else None
        )
        records.append(
            {
                "image": image_path.name,
                "truth": len(truth),
                "predicted": len(predicted),
                "false_negatives": false_negative,
                "false_positives": false_positive,
                "worst_matched_iou": boundary,
            }
        )
    return records


def select_samples(records) -> dict:
    picks = {}
    by_fn = [r for r in records if r["false_negatives"] > 0]
    by_fp = [r for r in records if r["false_positives"] > 0]
    by_boundary = [r for r in records if r["worst_matched_iou"] is not None]
    if by_fn:
        picks["false_negative"] = max(by_fn, key=lambda r: r["false_negatives"])
    if by_fp:
        picks["false_positive"] = max(by_fp, key=lambda r: r["false_positives"])
    if by_boundary:
        picks["mask_boundary"] = min(by_boundary, key=lambda r: r["worst_matched_iou"])
    return picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector", type=Path, default=Path("models/pothole_yolo26n_seg.onnx")
    )
    parser.add_argument(
        "--onnx", type=Path, default=Path("models/pothole_yolo26n_seg.onnx")
    )
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument(
        "--images", type=Path, default=Path("artifacts/yolo26n-seg/dataset/images/test")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("artifacts/yolo26n-seg/dataset/labels/test")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/portfolio-detection/a1.json")
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.25)
    # ONNX phải chạy CPU: onnxruntime trong môi trường này thiếu libcudart, và
    # số deployment của report vốn đo bằng ONNX Runtime CPU.
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cpu" if args.detector.suffix == ".onnx" else ""

    # torch phải nạp trước onnxruntime: onnxruntime-gpu linh động liên kết
    # libcudart mà torch mới là thứ nạp nó vào process.
    import torch
    import onnxruntime
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.detector), task="segment")
    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        verbose=False,
        plots=False,
        project=".cache/benchmark-output",
        name="a1",
        exist_ok=True,
        device=args.device,
    )

    images = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    records = collect_failures(model, images, args.labels, args.confidence)
    picks = select_samples(records)

    report = {
        "kpi": "A1 — box mAP@0.5 >= 80% (đạt), >= 85% (xuất sắc)",
        "purpose": (
            "Receipt máy đọc được cho số headline Phần A; lần chạy trước chỉ "
            "để lại PNG nên số không tự truy vết được."
        ),
        "receipt": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_receipt(),
            "source": {
                **file_digest(Path(__file__)),
                "path": "benchmarks/benchmark_a1_receipt.py",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "onnxruntime": onnxruntime.__version__,
                "onnxruntime_gpu": package_version("onnxruntime-gpu"),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        },
        "model": {
            "evaluated": file_digest(args.detector),
            "deployed": file_digest(args.onnx) if args.onnx.exists() else None,
            "same_artifact": (
                args.onnx.exists()
                and args.detector.resolve() == args.onnx.resolve()
            ),
        },
        "dataset": {
            "name": "Pothole-600 official test split",
            "data_yaml": str(args.data),
            "split": "test",
            "images": len(images),
            "note": (
                "Split không tham gia gradient update, nhưng đã được dùng để "
                "đối chiếu nhiều export/checkpoint, nên không gọi là test "
                "hoàn toàn chưa quan sát."
            ),
        },
        "run_config": {
            "imgsz": args.imgsz,
            "confidence_for_failure_ranking": args.confidence,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        },
        "metrics": {
            "box_map50": float(metrics.box.map50),
            "box_map": float(metrics.box.map),
            "box_precision": float(metrics.box.mp),
            "box_recall": float(metrics.box.mr),
            "mask_map50": float(metrics.seg.map50),
            "mask_map": float(metrics.seg.map),
            "mask_precision": float(metrics.seg.mp),
            "mask_recall": float(metrics.seg.mr),
        },
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
        "failure_evidence": {
            "note": (
                "Mẫu chọn bằng xếp hạng theo IoU trên toàn split, không phải "
                "ba batch đầu (vốn toàn ca thành công). Receipt chỉ lưu metadata "
                "để file máy đọc được gọn; dataset nguồn được phát hành MIT."
            ),
            "images_with_false_negative": sum(
                1 for r in records if r["false_negatives"] > 0
            ),
            "images_with_false_positive": sum(
                1 for r in records if r["false_positives"] > 0
            ),
            "selected": picks,
        },
        "limitations": [
            "Pothole-600 test split đã được xem khi chọn export/checkpoint, "
            "nên đây là evaluation split chứ không phải unseen test.",
            "Latency của validation bao gồm protocol khác stereo throughput; "
            "không dùng số này thay benchmark pipeline trong README.",
            "Receipt chỉ lưu tên file và số đo failure; người dùng phải tự tải "
            "Pothole-600 để xem raw frame.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    m = report["metrics"]
    print(
        f"box  mAP@0.5 {m['box_map50']:.4f}  mAP@0.5:0.95 {m['box_map']:.4f}\n"
        f"mask mAP@0.5 {m['mask_map50']:.4f}  mAP@0.5:0.95 {m['mask_map']:.4f}\n"
        f"ranked failure cases: {sorted(picks)}\n"
        f"artifact: {args.output}"
    )


if __name__ == "__main__":
    main()
