import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SAMPLE_ID = re.compile(r"^(\d{8}_\d{6})")


def find_dataset_root(path: Path) -> Path:
    candidates = [path, *path.glob("*")]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in ("images", "depths", "labels")):
            return candidate
    raise FileNotFoundError(f"Không tìm thấy images/depths/labels trong {path}")


def sample_id(path: Path) -> str:
    match = SAMPLE_ID.match(path.name)
    if not match:
        raise ValueError(f"Tên file không có timestamp ID: {path.name}")
    return match.group(1)


def index_files(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in paths:
        key = sample_id(path)
        index.setdefault(key, []).append(path)
    return index


def audit_labels(paths: list[Path]) -> dict:
    classes: Counter[int] = Counter()
    instances = 0
    point_counts: list[int] = []
    invalid: list[str] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                values = [float(value) for value in line.split()]
                class_id = int(values[0])
                coordinates = values[1:]
                valid = (
                    len(coordinates) >= 6
                    and len(coordinates) % 2 == 0
                    and all(0.0 <= value <= 1.0 for value in coordinates)
                )
                if not valid:
                    raise ValueError
            except (ValueError, IndexError):
                invalid.append(f"{path.name}:{line_number}")
                continue
            classes[class_id] += 1
            instances += 1
            point_counts.append(len(coordinates) // 2)
    return {
        "instances": instances,
        "classes": dict(sorted(classes.items())),
        "invalid_lines": invalid,
        "polygon_points_median": float(np.median(point_counts)) if point_counts else None,
    }


def audit_dataset(path: Path) -> dict:
    root = find_dataset_root(path)
    image_paths = sorted((root / "images").glob("*"))
    depth_paths = sorted((root / "depths").glob("*.npy"))
    label_paths = sorted((root / "labels").glob("*.txt"))
    images = index_files(image_paths)
    depths = index_files(depth_paths)
    labels = index_files(label_paths)
    all_ids = set(images) | set(depths) | set(labels)

    image_shapes: Counter[str] = Counter()
    unreadable_images: list[str] = []
    nonstandard_images: list[str] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable_images.append(image_path.name)
        else:
            image_shapes[f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"] += 1
            if image.shape[:2] != (480, 640):
                nonstandard_images.append(image_path.name)

    depth_shapes: Counter[str] = Counter()
    depth_dtypes: Counter[str] = Counter()
    valid_fractions: list[float] = []
    per_image_medians: list[float] = []
    invalid_depths: list[str] = []
    low_valid_depths: list[str] = []
    saturated_pixels = 0
    global_min: int | None = None
    global_max: int | None = None
    for depth_path in depth_paths:
        try:
            depth = np.load(depth_path, allow_pickle=False)
        except (OSError, ValueError):
            invalid_depths.append(depth_path.name)
            continue
        depth_shapes[f"{depth.shape[0]}x{depth.shape[1]}" if depth.ndim == 2 else str(depth.shape)] += 1
        depth_dtypes[str(depth.dtype)] += 1
        valid_mask = np.isfinite(depth) & (depth > 0)
        if np.issubdtype(depth.dtype, np.integer):
            saturated = depth == np.iinfo(depth.dtype).max
            saturated_pixels += int(np.count_nonzero(saturated))
            valid_mask &= ~saturated
        valid = depth[valid_mask]
        valid_fractions.append(float(valid.size / depth.size))
        if valid.size / depth.size < 0.8:
            low_valid_depths.append(depth_path.name)
        if valid.size:
            per_image_medians.append(float(np.median(valid)))
            current_min, current_max = int(valid.min()), int(valid.max())
            global_min = current_min if global_min is None else min(global_min, current_min)
            global_max = current_max if global_max is None else max(global_max, current_max)

    metadata = [
        str(candidate.relative_to(root))
        for candidate in root.rglob("*")
        if candidate.is_file()
        and any(word in candidate.name.lower() for word in ("calib", "intrinsic", "depth_scale", "split"))
    ]
    complete_ids = set(images) & set(depths) & set(labels)
    unambiguous_ids = {
        key for key in complete_ids if len(images[key]) == len(depths[key]) == len(labels[key]) == 1
    }
    ambiguous_ids = sorted(complete_ids - unambiguous_ids)
    nonstandard_ids = {sample_id(Path(name)) for name in nonstandard_images}
    return {
        "dataset_root": str(root.resolve()),
        "counts": {
            "images": len(image_paths),
            "depths": len(depth_paths),
            "labels": len(label_paths),
            "candidate_triplets": sum(
                min(len(images[key]), len(depths[key]), len(labels[key])) for key in complete_ids
            ),
            "unambiguous_triplets": len(unambiguous_ids),
            "geometry_ready_triplets": len(unambiguous_ids - nonstandard_ids),
            "ambiguous_triplets": sum(
                min(len(images[key]), len(depths[key]), len(labels[key])) for key in ambiguous_ids
            ),
        },
        "pairing": {
            "missing_images": sorted(all_ids - set(images)),
            "missing_depths": sorted(all_ids - set(depths)),
            "missing_labels": sorted(all_ids - set(labels)),
            "ambiguous_ids": ambiguous_ids,
        },
        "images": {
            "shapes": dict(image_shapes),
            "unreadable": unreadable_images,
            "nonstandard_480x640": nonstandard_images,
        },
        "depth": {
            "shapes": dict(depth_shapes),
            "dtypes": dict(depth_dtypes),
            "invalid_files": invalid_depths,
            "below_80_percent_valid": low_valid_depths,
            "valid_fraction_min": min(valid_fractions) if valid_fractions else None,
            "valid_fraction_median": float(np.median(valid_fractions)) if valid_fractions else None,
            "valid_fraction_max": max(valid_fractions) if valid_fractions else None,
            "nonzero_min": global_min,
            "nonzero_max": global_max,
            "saturated_65535_pixels": saturated_pixels,
            "per_image_nonzero_median": float(np.median(per_image_medians)) if per_image_medians else None,
        },
        "labels": audit_labels(label_paths),
        "metadata_files": metadata,
        "assessment": {
            "raw_depth_candidate": depth_dtypes == Counter({"uint16": len(depth_paths)})
            and depth_shapes == Counter({"480x640": len(depth_paths)}),
            "metric_unit_verified": bool(metadata),
            "note": (
                "Depth uint16 đồng bộ có thể là raw RealSense; cần depth_scale/intrinsics "
                "từ nguồn tác giả trước khi quy đổi chắc chắn sang mét và m²."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PothRGBD RGB/depth/segmentation pairing")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
