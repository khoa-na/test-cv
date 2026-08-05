"""Video demo Phần A — phát hiện ổ gà + đo độ sâu/diện tích bằng stereo.

Bộ Fan chỉ có 27 cặp stereo tĩnh, không phải video liên tục, nên mỗi cặp được
giữ vài giây. Ground truth hiển thị cạnh số đo để người xem tự đối chiếu.

Không có footage thực địa tự quay trong time-box, nên video render từ dataset
công khai và banner ghi rõ nguồn.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_fan_stereo import CALIBRATION, paired_images
from benchmarks.benchmark_stereo_yolo import ground_truth_area, ground_truth_depth
from pipelines.stereo_yolo_pipeline import StereoYOLOPipeline

BANNER = (
    "Source: Fan et al. stereo pothole dataset. Rendered, NOT self-recorded"
    " HCMC footage. Held-out = model2/model3; model1 used for calibration."
)
FRAME_WIDTH = 1280
BANNER_HEIGHT = 34
STATUS_HEIGHT = 128


def put(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def error_color(error: float | None) -> tuple[int, int, int]:
    if error is None:
        return (120, 120, 200)
    if error <= 0.08:
        return (96, 216, 96)
    if error <= 0.15:
        return (64, 208, 240)
    return (72, 72, 232)


def render(
    dataset: Path,
    detector: Path,
    output: Path,
    seconds_per_pair: float,
    fps: float,
    **pipeline_kwargs,
) -> dict:
    dataset = dataset / "dataset" if (dataset / "dataset").is_dir() else dataset
    pipeline = StereoYOLOPipeline(
        detector_path=detector,
        focal_px=CALIBRATION["model1"]["focal_px"],
        baseline_mm=CALIBRATION["model1"]["baseline_mm"],
        **pipeline_kwargs,
    )
    # Cùng warmup với benchmark_stereo_yolo, nếu không FPS trên video thấp hơn
    # số trong báo cáo chỉ vì chi phí lần gọi đầu.
    warm_left, warm_right = paired_images(dataset / "model1")[0]
    for _ in range(3):
        pipeline.predict(cv2.imread(str(warm_left)), cv2.imread(str(warm_right)))

    image_height = None
    writer = None
    output.parent.mkdir(parents=True, exist_ok=True)
    hold = max(1, int(round(seconds_per_pair * fps)))
    rendered = 0
    detected = 0

    for model_name, calibration in CALIBRATION.items():
        model_dir = dataset / model_name
        truth_depth = ground_truth_depth(model_dir)
        truth_area = ground_truth_area(model_dir)
        pipeline.focal_px = calibration["focal_px"]
        pipeline.baseline_mm = calibration["baseline_mm"]
        held_out = model_name != "model1"

        for left_path, right_path in paired_images(model_dir):
            left = cv2.imread(str(left_path))
            right = cv2.imread(str(right_path))
            report, annotated = pipeline.predict(left, right)

            scaled = cv2.resize(
                annotated,
                (
                    FRAME_WIDTH,
                    int(round(annotated.shape[0] * FRAME_WIDTH / annotated.shape[1])),
                ),
            )
            if writer is None:
                image_height = scaled.shape[0]
                height = BANNER_HEIGHT + image_height + STATUS_HEIGHT
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (FRAME_WIDTH, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Không mở được VideoWriter: {output}")
            elif scaled.shape[0] != image_height:
                scaled = cv2.resize(scaled, (FRAME_WIDTH, image_height))

            valid = [item for item in report["potholes"] if "depth_mm" in item]
            best = max(valid, key=lambda item: item["confidence"]) if valid else None
            depth_error = (
                abs(best["depth_mm"] - truth_depth) / truth_depth if best else None
            )
            area_error = (
                abs(best["area_mm2"] - truth_area) / truth_area if best else None
            )
            detected += best is not None

            canvas = np.full(
                (BANNER_HEIGHT + image_height + STATUS_HEIGHT, FRAME_WIDTH, 3),
                18,
                dtype=np.uint8,
            )
            cv2.rectangle(canvas, (0, 0), (FRAME_WIDTH, BANNER_HEIGHT), (52, 40, 32), -1)
            put(canvas, BANNER, (10, 22), scale=0.42, color=(210, 210, 210))
            canvas[BANNER_HEIGHT : BANNER_HEIGHT + image_height] = scaled

            base = BANNER_HEIGHT + image_height
            tag = "HELD-OUT" if held_out else "CALIBRATION"
            put(
                canvas,
                f"{model_name} / {left_path.stem}   [{tag}]",
                (12, base + 26),
                scale=0.55,
                color=(235, 235, 235) if held_out else (150, 190, 240),
                thickness=2,
            )
            latency = report["latency"]["total_ms"]
            # Benchmark lấy median 3 lần lặp mỗi cặp; ở đây chỉ chạy một lần nên
            # số thấp hơn median 17,8 FPS trong báo cáo. Ghi rõ để khỏi mâu thuẫn.
            put(
                canvas,
                f"detections: {report['count']}    "
                f"latency {latency:6.1f} ms    "
                f"{1000.0 / max(latency, 1e-6):4.1f} FPS (CPU, single-shot)",
                (12, base + 54),
            )
            if best is None:
                put(
                    canvas,
                    "no stereo-fused pothole in this pair",
                    (12, base + 84),
                    color=(120, 120, 200),
                )
            else:
                put(
                    canvas,
                    f"depth  {best['depth_mm']:7.1f} mm   GT {truth_depth:7.1f} mm"
                    f"   err {depth_error * 100:5.1f}%",
                    (12, base + 84),
                    color=error_color(depth_error),
                )
                put(
                    canvas,
                    f"area {best['area_mm2'] / 100.0:8.1f} cm2  GT"
                    f" {truth_area / 100.0:8.1f} cm2  err {area_error * 100:5.1f}%",
                    (12, base + 110),
                    color=error_color(area_error),
                )
                if best["fallback_applied"]:
                    put(
                        canvas,
                        "residual fallback",
                        (660, base + 84),
                        color=(64, 208, 240),
                    )
            put(
                canvas,
                "green <=8%   cyan <=15%   red >15%",
                (900, base + 110),
                scale=0.42,
                color=(150, 150, 150),
            )

            for _ in range(hold):
                writer.write(canvas)
            rendered += 1

    if writer is None:
        raise RuntimeError("Không có cặp stereo nào được render")
    writer.release()
    return {
        "pairs": rendered,
        "fused": detected,
        "duration_s": rendered * hold / fps,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(".cache/data/fan-stereo-pothole")
    )
    parser.add_argument(
        "--detector", type=Path, default=Path("models/pothole_yolo26n_seg.onnx")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/demo/part_a.mp4")
    )
    parser.add_argument("--seconds-per-pair", type=float, default=2.4)
    parser.add_argument("--fps", type=float, default=25.0)
    # Giữ đúng cấu hình production của benchmark_stereo_yolo.
    parser.add_argument("--metric-scale", type=float, default=0.8334711918061039)
    parser.add_argument("--scale", type=float, default=0.3125)
    parser.add_argument("--num-disparities", type=int, default=112)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-alignment-iou", type=float, default=0.1)
    parser.add_argument("--area-quantile", type=float, default=0.986)
    parser.add_argument("--area-scale", type=float, default=1.3755604448201877)
    parser.add_argument("--yolo-area-scale", type=float, default=1.0)
    parser.add_argument("--opencv-threads", type=int, default=4)
    args = parser.parse_args()

    summary = render(
        args.dataset,
        args.detector,
        args.output,
        args.seconds_per_pair,
        args.fps,
        metric_scale=args.metric_scale,
        image_scale=args.scale,
        num_disparities=args.num_disparities,
        confidence=args.confidence,
        opencv_threads=args.opencv_threads,
        min_alignment_iou=args.min_alignment_iou,
        area_quantile=args.area_quantile,
        area_scale=args.area_scale,
        yolo_area_scale=args.yolo_area_scale,
    )
    print(
        f"{summary['pairs']} pairs ({summary['fused']} fused) -> "
        f"{summary['duration_s']:.1f}s  {summary['output']}"
    )


if __name__ == "__main__":
    main()
