from pathlib import Path

import yaml

from data_tools.build_combined_seg_dataset import build_manifest
from training.train_combined_detector import BASE_MODEL_SHA256


def write_split(root: Path, split: str, count: int) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(count):
        (image_dir / f"{index}.png").write_bytes(b"image")
        (label_dir / f"{index}.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8"
        )


def test_build_manifest_tracks_exact_split_counts(tmp_path: Path) -> None:
    pothole600 = tmp_path / "pothole600"
    pothrgbd = tmp_path / "pothrgbd"
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        write_split(pothole600, split, count)
    for split, count in (("train", 3), ("val", 1)):
        write_split(pothrgbd, split, count)

    output = tmp_path / "combined.yaml"
    manifest = build_manifest(pothole600, pothrgbd, output)

    assert output.is_file()
    assert manifest["portfolio_manifest"]["training_images"] == 5
    assert manifest["portfolio_manifest"]["training_instances"] == 5
    assert manifest["val"].endswith("pothole600/images/val")
    assert all("pothrgbd/images/val" not in path for path in manifest["train"])


def test_canonical_training_recipe_matches_tracked_model() -> None:
    recipe_path = Path(__file__).resolve().parents[1] / "training/combined_recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

    assert recipe["base_model"]["sha256"] == BASE_MODEL_SHA256
    assert recipe["dataset"]["training_images"] == 1008
    assert recipe["dataset"]["training_instances"] == 1092
    assert recipe["train"]["batch"] == 16
    assert recipe["train"]["patience"] == 0
    assert recipe["train"]["copy_paste"] == 0.3
