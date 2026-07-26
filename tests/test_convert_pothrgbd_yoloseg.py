import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from data_tools.convert_pothrgbd_yoloseg import convert


def write_sample(root: Path, key: str, label: str, shape=(480, 640)) -> None:
    image_name = f"{key}_color_png.rf.hash"
    cv2.imwrite(str(root / "images" / f"{image_name}.jpg"), np.zeros((*shape, 3), np.uint8))
    np.save(root / "depths" / f"{key}_depth.npy", np.full((480, 640), 1000, np.uint16))
    (root / "labels" / f"{image_name}.txt").write_text(label, encoding="utf-8")


class ConvertPothRGBDYoloSegTest(unittest.TestCase):
    def test_converts_valid_and_skips_bad_samples(self):
        polygon = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("images", "depths", "labels"):
                (root / name).mkdir()
            for day in range(1, 21):
                write_sample(root, f"2025030{day % 10}_1200{day:02d}", polygon)
            write_sample(root, "20250310_130000", polygon, shape=(400, 640))
            write_sample(root, "20250310_130001", "0 0.1 0.1 0.9 1.5 0.9 0.9\n")
            output = root / "yoloseg"

            yaml_path = convert(root, output, val_fraction=0.2, seed=42)

            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            self.assertEqual(config["names"], {0: "pothole"})
            train = sorted((output / "images" / "train").glob("*.jpg"))
            val = sorted((output / "images" / "val").glob("*.jpg"))
            self.assertEqual(len(train) + len(val), 20)
            self.assertTrue(val, "split hash phải đưa ít nhất một mẫu vào val")
            for image_path in train + val:
                label = image_path.parents[2] / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
                self.assertEqual(label.read_text(encoding="utf-8"), polygon)


if __name__ == "__main__":
    unittest.main()
