from pathlib import Path

import cv2
import numpy as np
# onnxruntime-gpu liên kết động libcudart; trên máy không cài CUDA runtime
# hệ thống, thư viện đó chỉ có trong wheel của torch. Nạp torch trước để nó đưa
# libcudart vào process, nếu không "import onnxruntime" chết ngay lúc import.
try:  # pragma: no cover - phụ thuộc môi trường
    import torch  # noqa: F401
except ImportError:
    pass
import onnxruntime as ort


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_features(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if not x.size:
        raise ValueError("Mask rỗng")
    height, width = mask.shape
    x1, x2, y1, y2 = x.min(), x.max() + 1, y.min(), y.max() + 1
    box_area = max((x2 - x1) * (y2 - y1), 1)
    return np.array(
        [
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
            mask.sum() / mask.size,
            mask.sum() / box_area,
        ],
        dtype=np.float32,
    )


def roi_crop(
    image: np.ndarray, points: np.ndarray, size: int, margin: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    y, x = np.nonzero(mask)
    x1, x2, y1, y2 = x.min(), x.max() + 1, y.min(), y.max() + 1
    pad_x, pad_y = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, x2 = max(0, x1 - pad_x), min(image.shape[1], x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(image.shape[0], y2 + pad_y)
    rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    crop_mask = cv2.resize(
        mask[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST
    )
    return rgb, crop_mask


def preprocess_roi(
    image: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = mask.astype(bool)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Mask rỗng")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    crop, crop_mask = roi_crop(image, points, size)
    crop = crop.astype(np.float32) / 255.0
    tensor = np.concatenate(
        (((crop - MEAN) / STD).transpose(2, 0, 1), crop_mask[None].astype(np.float32))
    )
    return tensor[None].astype(np.float32), mask_features(mask)[None]


class DepthRegressorONNX:
    def __init__(
        self,
        model_path: Path,
        size: int = 128,
        threads: int = 6,
        reliable_min: float = 32.46,
        reliable_max: float = 59.48,
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.size = size
        self.reliable_min = reliable_min
        self.reliable_max = reliable_max

    def predict(self, image: np.ndarray, mask: np.ndarray) -> dict:
        image_input, geometry_input = preprocess_roi(image, mask, self.size)
        depth, normalized_depth = self.session.run(
            None, {"image": image_input, "geometry": geometry_input}
        )[0][0]
        return {
            "relative_depth": float(depth),
            "normalized_depth": float(normalized_depth),
            "depth_reliable": bool(self.reliable_min <= depth <= self.reliable_max),
            "reliability_rule": "validation_prediction_range",
        }
