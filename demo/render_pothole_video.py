"""Video demo Phần A trên footage video thật, có ground truth từng frame.

Bộ Fan chỉ có 27 cặp stereo tĩnh nên `render_part_a.py` phải giữ mỗi cặp vài
giây. Bộ này là video liên tục 24fps, và quan trọng hơn: model chưa từng thấy
một frame nào của nó. Contour vàng là dự đoán, xanh lá là ground truth, người
xem tự đối chiếu.

Clip lấy theo thứ tự số từ test split, không chọn theo kết quả.

Đề yêu cầu quay tại TP.HCM. Không thực hiện được trong thời gian test, nên
video render từ dataset công khai và banner ghi rõ nguồn.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.pothole_pipeline import PotholePipeline

BANNER_LINES = (
    "Source: Mendeley 5bwfg4v4cd (CC BY 4.0), Indonesia.",
    "Rendered, NOT self-recorded HCMC footage. Model never saw this dataset.",
)
LEGEND = "yellow = predicted mask   green box = predicted bbox   magenta = ground truth"
TRUTH_COLOR = (255, 0, 255)
FRAME = 720
BANNER_HEIGHT = 46
STATUS_HEIGHT = 92


def put(image, text, origin, *, scale=0.5, color=(235, 235, 235), thickness=1):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
        cv2.LINE_AA,
    )


def clip_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame
    capture.release()


def compose(rgb, truth_mask, report, annotated, clip_id, index, total):
    canvas = np.zeros((BANNER_HEIGHT + FRAME + STATUS_HEIGHT, FRAME, 3), np.uint8)
    canvas[:BANNER_HEIGHT] = (28, 28, 28)
    for row, line in enumerate(BANNER_LINES):
        put(canvas, line, (10, 18 + row * 15), scale=0.42, color=(150, 200, 255))

    view = cv2.resize(annotated, (FRAME, FRAME))
    truth = cv2.resize(truth_mask.astype(np.uint8), (FRAME, FRAME),
                       interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, TRUTH_COLOR, 2)
    canvas[BANNER_HEIGHT:BANNER_HEIGHT + FRAME] = view

    status = canvas[BANNER_HEIGHT + FRAME:]
    status[:] = (22, 22, 22)
    latency = report["latency"]
    put(status, f"clip {clip_id}  frame {index + 1}/{total}", (10, 22), scale=0.55)
    put(status, LEGEND, (10, 44), scale=0.4, color=(170, 170, 170))
    put(
        status,
        f"detections {report['count']}   "
        f"{latency['total_ms']:.0f} ms/frame   "
        f"{latency['sequential_fps']:.1f} FPS",
        (10, 66),
        scale=0.5,
        color=(96, 216, 216),
    )
    depths = [
        f"#{p['id']} conf {p['confidence']:.2f} {p['severity']}"
        for p in report["potholes"][:3]
    ]
    put(status, "   ".join(depths) or "no detection", (10, 86), scale=0.45,
        color=(170, 170, 170))
    return canvas


def render(dataset: Path, detector: Path, depth_model: Path, output: Path,
           clips: int, **kwargs) -> dict:
    rgb_dir = dataset / "test" / "rgb"
    mask_dir = dataset / "test" / "mask"
    paths = sorted(rgb_dir.glob("*.mp4"))[:clips]
    if not paths:
        raise FileNotFoundError(f"Không có clip nào trong {rgb_dir}")

    pipeline = PotholePipeline(detector, depth_model, **kwargs)
    first = next(clip_frames(paths[0]))
    pipeline.predict(first)  # warmup, nếu không frame đầu kéo FPS xuống

    writer, per_clip, total_frames = None, [], 0
    for path in paths:
        truth_frames = list(clip_frames(mask_dir / path.name))
        frames = list(clip_frames(path))
        counts, intersection, union = [], 0, 0
        for index, frame in enumerate(frames):
            report, annotated = pipeline.predict(frame)
            truth = (
                cv2.cvtColor(truth_frames[index], cv2.COLOR_BGR2GRAY) > 127
                if index < len(truth_frames)
                else np.zeros(frame.shape[:2], bool)
            )
            canvas = compose(frame, truth, report, annotated, path.stem, index,
                             len(frames))
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), 24.0,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
            counts.append(report["count"])
            predicted = np.zeros(frame.shape[:2], bool)
            for pothole in report["potholes"]:
                x1, y1, x2, y2 = (int(v) for v in pothole["bbox"])
                predicted[max(y1, 0):y2, max(x1, 0):x2] = True
            intersection += int(np.count_nonzero(predicted & truth))
            union += int(np.count_nonzero(predicted | truth))
            total_frames += 1
        per_clip.append({
            "clip": path.stem,
            "frames": len(frames),
            "mean_detections": float(np.mean(counts)),
            "bbox_iou_vs_truth": intersection / union if union else None,
        })
    if writer is not None:
        writer.release()
    return {"clips": per_clip, "frames": total_frames, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/data/pothole-video-id/pothole_video"),
    )
    parser.add_argument(
        "--detector", type=Path, default=Path("artifacts/final/pothole_yolo26n_seg.pt")
    )
    parser.add_argument(
        "--depth-model",
        type=Path,
        # Bản -context nhận 4 kênh (RGB + mask), khớp preprocess_roi. README
        # dùng bản này; artifacts/depth-regressor/ là bản 3 kênh cũ.
        default=Path("artifacts/depth-regressor-context/pothole_depth_regressor.onnx"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo/part_a_video.mp4"))
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = render(
        args.dataset, args.detector, args.depth_model, args.output, args.clips,
        confidence=args.confidence,
    )
    ious = [c["bbox_iou_vs_truth"] for c in summary["clips"]
            if c["bbox_iou_vs_truth"] is not None]
    summary["bbox_iou_median"] = float(np.median(ious)) if ious else None
    summary["note"] = (
        "IoU tính trên bbox dự đoán so với mask GT, nên là chặn dưới của "
        "segmentation IoU. Clip lấy theo thứ tự số, không chọn theo kết quả."
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nvideo: {args.output}")


if __name__ == "__main__":
    main()
