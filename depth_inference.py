import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    if size < 14 or size % 14:
        raise ValueError("Depth input size phải là bội số của 14")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    return ((resized - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if not finite.size:
        raise ValueError("Depth map không có giá trị hữu hạn")
    low, high = np.percentile(finite, (2, 98))
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


class DepthAnythingONNX:
    def __init__(self, model_path: Path, size: int = 224, threads: int = 0):
        if size < 14 or size % 14:
            raise ValueError("Depth input size phải là bội số của 14")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.size = size

    def predict(self, image: np.ndarray) -> np.ndarray:
        prediction = self.session.run(
            None, {self.input_name: preprocess(image, self.size)}
        )[0][0]
        return cv2.resize(
            prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Depth Anything V2 ONNX CPU inference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/depth"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")
    model = DepthAnythingONNX(args.model, args.size, args.threads)
    for _ in range(args.warmup):
        model.predict(image)

    latencies = []
    depth = None
    for _ in range(args.runs):
        started = time.perf_counter()
        depth = model.predict(image)
        latencies.append((time.perf_counter() - started) * 1000)
    assert depth is not None

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "relative_depth.npy", depth)
    cv2.imwrite(str(args.output / "relative_depth.jpg"), colorize_depth(depth))
    median_ms = float(np.median(latencies))
    report = {
        "model": str(args.model.resolve()),
        "image": str(args.image.resolve()),
        "input_size": args.size,
        "runs": args.runs,
        "latency_ms_median": median_ms,
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / median_ms,
        "output_shape": list(depth.shape),
        "output_min": float(depth.min()),
        "output_max": float(depth.max()),
        "depth_type": "relative",
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
