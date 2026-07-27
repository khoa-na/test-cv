"""Module A trên một bộ dữ liệu độc lập hoàn toàn.

Mọi con số Module A hiện tại (box mAP@0.5 89,8%, mask 87,1%) đo trên val split
của chính họ dữ liệu đã train: Pothole-600 + PothRGBD. Bộ này thì model chưa
thấy một frame nào — Mendeley 5bwfg4v4cd (CC BY 4.0, Indonesia), 123 clip test,
mỗi clip 48 frame 1080x1080 kèm một clip mask làm ground truth từng frame.

Camera đặt cách mặt đường ~130 cm nhìn xuống, gần miền PothRGBD, nên đây là
phép thử chuyển miền chứ không phải thử một bài toán khác.

mAP tính bằng chính `YOLO.val()` với imgsz=512 như lúc train, nên số so trực
tiếp được với 89,8%/87,1%. Không có ngưỡng nào chỉnh theo kết quả.

Cảnh báo về ground truth: mask được phát hành dưới dạng mp4 nén mất mát, nên
biên mask có nhiễu nén. Script nhị phân hoá ở >127 và ghi lại tỉ lệ pixel rơi
vào vùng xám 64..192 để người đọc tự đánh giá mức nhiễu này.
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARCHIVE = Path(".cache/data/pothole-video-id/pothole_video.zip")
EXTRACT_ROOT = Path(".cache/data/pothole-video-id")
# Số đang báo cáo, đo trên val cùng họ dữ liệu train.
IN_DOMAIN = {"box_map50": 0.898, "box_map": 0.563, "mask_map50": 0.871, "mask_map": 0.525}
MIN_POLYGON_POINTS = 6
MIN_INSTANCE_PIXELS = 200


def ensure_test_split(archive: Path, root: Path) -> Path:
    rgb_dir = root / "pothole_video" / "test" / "rgb"
    if rgb_dir.is_dir() and len(list(rgb_dir.glob("*.mp4"))) >= 123:
        return root / "pothole_video" / "test"
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.startswith("pothole_video/test/") and name.endswith(".mp4")
        ]
        bundle.extractall(root, members=members)
    return root / "pothole_video" / "test"


def clip_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Tách instance bằng connected components rồi lấy contour ngoài."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    result = []
    for label in range(1, count):
        blob = (labels == label).astype(np.uint8)
        if int(blob.sum()) < MIN_INSTANCE_PIXELS:
            continue
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        if len(approx) >= MIN_POLYGON_POINTS:
            result.append(approx)
    return result


def build_dataset(split_dir: Path, output: Path, stride: int) -> dict:
    images = output / "images" / "val"
    labels = output / "labels" / "val"
    for folder in (images, labels):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    clips = sorted((split_dir / "rgb").glob("*.mp4"))
    written, instances, empty, grey_pixels, total_pixels = 0, 0, 0, 0, 0
    for clip in clips:
        rgb_frames = clip_frames(clip)
        mask_frames = clip_frames(split_dir / "mask" / clip.name)
        for index in range(0, min(len(rgb_frames), len(mask_frames)), stride):
            grey = cv2.cvtColor(mask_frames[index], cv2.COLOR_BGR2GRAY)
            grey_pixels += int(np.count_nonzero((grey > 64) & (grey < 192)))
            total_pixels += grey.size
            mask = grey > 127
            shapes = polygons(mask)
            stem = f"{clip.stem}_{index:03d}"
            cv2.imwrite(str(images / f"{stem}.jpg"), rgb_frames[index])
            height, width = mask.shape
            lines = []
            for shape in shapes:
                flat = shape.astype(np.float64)
                flat[:, 0] /= width
                flat[:, 1] /= height
                flat = np.clip(flat, 0.0, 1.0)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in flat.reshape(-1)))
            (labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            written += 1
            instances += len(lines)
            empty += int(not lines)

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        f"path: {output.resolve()}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n  0: pothole\n",
        encoding="utf-8",
    )
    return {
        "clips": len(clips),
        "frames": written,
        "frame_stride": stride,
        "gt_instances": instances,
        "frames_without_instance": empty,
        "mask_compression_grey_fraction": grey_pixels / total_pixels if total_pixels else None,
        "data_yaml": str(data_yaml),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--extract-root", type=Path, default=EXTRACT_ROOT)
    parser.add_argument(
        "--detector", type=Path, default=Path("artifacts/final/pothole_yolo26n_seg.pt")
    )
    parser.add_argument(
        "--dataset-output", type=Path, default=Path(".cache/data/pothole-video-id/yoloseg")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cross-dataset-pothole")
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help=(
            "Lấy mỗi stride frame. Frame liền nhau trong clip 2 s gần như trùng "
            "nhau nên giữ hết chỉ phồng số mẫu chứ không thêm thông tin. Chốt "
            "trước khi chạy, không chỉnh theo kết quả."
        ),
    )
    args = parser.parse_args()

    split_dir = ensure_test_split(args.archive, args.extract_root)
    dataset = build_dataset(split_dir, args.dataset_output, args.stride)
    print(json.dumps(dataset, indent=2, ensure_ascii=False))

    from ultralytics import YOLO

    metrics = YOLO(str(args.detector)).val(
        data=dataset["data_yaml"],
        imgsz=args.imgsz,
        split="val",
        verbose=False,
        plots=False,
        project=str(args.output),
        name="val",
        exist_ok=True,
    )
    measured = {
        "box_map50": float(metrics.box.map50),
        "box_map": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
        "mask_map50": float(metrics.seg.map50),
        "mask_map": float(metrics.seg.map),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
    }
    report = {
        "kpi": "mAP@0.5 >= 80% (đề bài, đo trên val cùng miền)",
        "question": "Model giữ được bao nhiêu khi sang bộ dữ liệu chưa từng thấy?",
        "detector": str(args.detector),
        "imgsz": args.imgsz,
        "source_dataset": {
            "name": "Mendeley 5bwfg4v4cd v2 - pothole video dataset",
            "license": "CC BY 4.0",
            "country": "Indonesia",
            "split_used": "test",
            "camera": "top-down, ~130 cm above road",
        },
        "dataset_build": dataset,
        "cross_dataset": measured,
        "in_domain_reference": IN_DOMAIN,
        "retention": {
            key: (measured[key] / IN_DOMAIN[key] if IN_DOMAIN.get(key) else None)
            for key in ("box_map50", "box_map", "mask_map50", "mask_map")
        },
        "limitations": [
            "Ground truth phát hành dưới dạng mp4 nén mất mát; biên mask có nhiễu "
            "nén. Tỉ lệ pixel xám 64..192 ghi trong dataset_build.",
            "Instance tách bằng connected components trên mask nhị phân, nên hai "
            "ổ gà dính nhau bị đếm thành một.",
            "Bộ này quay ở Indonesia, không phải TP.HCM. Không thay thế được yêu "
            "cầu quay thực địa của đề bài.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nbox  mAP@0.5 {measured['box_map50']:.3f} "
        f"(cùng miền {IN_DOMAIN['box_map50']:.3f})\n"
        f"mask mAP@0.5 {measured['mask_map50']:.3f} "
        f"(cùng miền {IN_DOMAIN['mask_map50']:.3f})\n"
        f"artifact: {args.output / 'benchmark.json'}"
    )


if __name__ == "__main__":
    main()
