import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from benchmarks.benchmark_depth_pothrgbd import geometry_ready_samples, polygon_masks
from data_tools.audit_pothrgbd import find_dataset_root
from pipelines.depth_regressor_inference import MEAN, STD, mask_features, roi_crop
from pipelines.pothole_pipeline import estimate_geometry


def split_name(key: str, seed: int = 42) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
    return "test" if value < 0.2 else "validation" if value < 0.3 else "train"


def build_records(root: Path, seed: int, min_depth: float, min_ratio: float) -> list[dict]:
    records = []
    for key, image_path, depth_path, label_path in geometry_ready_samples(root):
        sensor_depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
        sensor_depth[(sensor_depth <= 0) | (sensor_depth >= 65535)] = np.nan
        for mask in polygon_masks(label_path, sensor_depth.shape):
            try:
                geometry = estimate_geometry(mask, sensor_depth, depth_direction=1)
            except ValueError:
                continue
            depth = geometry["relative_depth"]
            ratio = depth / max(abs(geometry["road_depth_median"]), 1e-6)
            if depth < min_depth or ratio < min_ratio:
                continue
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            records.append(
                {
                    "sample_id": key,
                    "image": image_path,
                    "points": max(contours, key=cv2.contourArea).reshape(-1, 2),
                    "geometry": mask_features(mask),
                    "target": np.array([depth, ratio], dtype=np.float32),
                    "split": split_name(key, seed),
                }
            )
    return records


class PotholeDepthDataset(Dataset):
    def __init__(self, records: list[dict], size: int, augment: bool = False):
        self.records = records
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imread(str(record["image"]))
        crop, mask = roi_crop(image, record["points"], self.size)
        crop = crop.astype(np.float32) / 255.0
        if self.augment:
            if np.random.random() < 0.5:
                crop = crop[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            crop = np.clip(crop * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        crop = np.concatenate(
            (((crop - MEAN) / STD).transpose(2, 0, 1), mask[None].astype(np.float32))
        )
        return (
            torch.from_numpy(crop),
            torch.from_numpy(record["geometry"]),
            torch.from_numpy(np.log(record["target"])),
        )


class DepthRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        first = backbone.features[0][0]
        four_channel = nn.Conv2d(
            4,
            first.out_channels,
            first.kernel_size,
            first.stride,
            first.padding,
            bias=False,
        )
        with torch.no_grad():
            four_channel.weight[:, :3] = first.weight
            four_channel.weight[:, 3:] = first.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = four_channel
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.head = nn.Sequential(
            nn.Linear(576 + 6, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual = torch.flatten(self.avgpool(self.features(image)), 1)
        return self.head(torch.cat((visual, geometry), dim=1))


class ExportModel(nn.Module):
    def __init__(self, model: DepthRegressor, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.model(image, geometry) * self.target_std + self.target_mean)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    errors = np.abs(prediction - target) / target
    return {
        "instances": len(target),
        "raw_depth": {
            "mean_relative_error": float(errors[:, 0].mean()),
            "median_relative_error": float(np.median(errors[:, 0])),
            "within_15_percent": float(np.mean(errors[:, 0] <= 0.15)),
            "within_8_percent": float(np.mean(errors[:, 0] <= 0.08)),
        },
        "normalized_depth": {
            "mean_relative_error": float(errors[:, 1].mean()),
            "median_relative_error": float(np.median(errors[:, 1])),
            "within_15_percent": float(np.mean(errors[:, 1] <= 0.15)),
            "within_8_percent": float(np.mean(errors[:, 1] <= 0.08)),
        },
    }


def predict(model, loader, device, target_mean, target_std):
    predictions, targets = [], []
    model.eval()
    with torch.inference_mode():
        for image, geometry, target in loader:
            output = model(image.to(device), geometry.to(device)).cpu()
            predictions.append(torch.exp(output * target_std + target_mean).numpy())
            targets.append(torch.exp(target).numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pothole ROI scalar depth regressor")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/depth-regressor"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-depth", type=float, default=5.0)
    parser.add_argument("--min-depth-ratio", type=float, default=0.005)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = find_dataset_root(args.dataset)
    records = build_records(root, args.seed, args.min_depth, args.min_depth_ratio)
    splits = {name: [record for record in records if record["split"] == name] for name in ("train", "validation", "test")}
    if any(not split for split in splits.values()):
        raise ValueError("Train/validation/test split rỗng")
    target_logs = np.log(np.stack([record["target"] for record in splits["train"]]))
    target_mean = torch.tensor(target_logs.mean(axis=0), dtype=torch.float32)
    target_std = torch.tensor(target_logs.std(axis=0), dtype=torch.float32)

    loaders = {
        name: DataLoader(
            PotholeDepthDataset(split, args.size, augment=name == "train"),
            batch_size=args.batch,
            shuffle=name == "train",
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, split in splits.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    mean_device, std_device = target_mean.to(device), target_std.to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best.pt"
    best_error, stale = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, geometry, target in loaders["train"]:
            image, geometry, target = image.to(device), geometry.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image, geometry), (target - mean_device) / std_device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_prediction, val_target = predict(model, loaders["validation"], device, target_mean, target_std)
        val_metrics = metrics(val_prediction, val_target)
        val_error = val_metrics["raw_depth"]["median_relative_error"]
        print(
            f"epoch {epoch:03d}: loss={np.mean(losses):.4f} "
            f"val_raw_median={val_error:.4f} "
            f"val_ratio_median={val_metrics['normalized_depth']['median_relative_error']:.4f}",
            flush=True,
        )
        if val_error < best_error:
            best_error, stale = val_error, 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_prediction, test_target = predict(model, loaders["test"], device, target_mean, target_std)
    test_metrics = metrics(test_prediction, test_target)
    export_model = ExportModel(model.cpu().eval(), target_mean, target_std)
    onnx_path = args.output / "pothole_depth_regressor.onnx"
    dummy_image = torch.zeros(1, 4, args.size, args.size)
    dummy_geometry = torch.zeros(1, 6)
    torch.onnx.export(
        export_model,
        (dummy_image, dummy_geometry),
        onnx_path,
        input_names=["image", "geometry"],
        output_names=["depth"],
        dynamic_axes={"image": {0: "batch"}, "geometry": {0: "batch"}, "depth": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    image, geometry, _ = next(iter(loaders["test"]))
    samples = min(len(image), 8)
    ort_output = session.run(
        None, {"image": image[:samples].numpy(), "geometry": geometry[:samples].numpy()}
    )[0]
    validation_model = DepthRegressor(pretrained=False)
    validation_model.load_state_dict(
        torch.load(best_path, map_location="cpu", weights_only=True)
    )
    validation_export = ExportModel(
        validation_model.eval(), target_mean, target_std
    ).eval()
    with torch.inference_mode():
        torch_output = validation_export(image[:samples], geometry[:samples]).numpy()
    onnx_max_error = float(np.max(np.abs(ort_output - torch_output)))
    onnx_max_relative_error = float(
        np.max(np.abs(ort_output - torch_output) / np.maximum(np.abs(torch_output), 1e-6))
    )
    if onnx_max_relative_error > 1e-3:
        raise RuntimeError(f"ONNX export lệch PyTorch: {onnx_max_relative_error:.6f}")

    latencies = []
    inputs = {"image": image[:1].numpy(), "geometry": geometry[:1].numpy()}
    for _ in range(5):
        session.run(None, inputs)
    for _ in range(100):
        started = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - started) * 1000)
    report = {
        "dataset": str(root.resolve()),
        "model": "MobileNetV3-Small ROI + mask geometry",
        "device": str(device),
        "settings": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
        "counts": {name: len(split) for name, split in splits.items()},
        "test": test_metrics,
        "onnx": {
            "path": str(onnx_path.resolve()),
            "max_absolute_export_error": onnx_max_error,
            "max_relative_export_error": onnx_max_relative_error,
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps_single_roi": 1000.0 / float(np.median(latencies)),
        },
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
