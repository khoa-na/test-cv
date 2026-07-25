import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_tools.audit_pothrgbd import audit_dataset


class AuditPothRGBDTest(unittest.TestCase):
    def test_complete_rgb_depth_label_triplet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("images", "depths", "labels"):
                (root / name).mkdir()
            key = "20250227_135438"
            image_name = f"{key}_color_png.rf.hash"
            cv2.imwrite(str(root / "images" / f"{image_name}.jpg"), np.zeros((480, 640, 3), np.uint8))
            np.save(root / "depths" / f"{key}_depth.npy", np.full((480, 640), 1000, np.uint16))
            (root / "labels" / f"{image_name}.txt").write_text(
                "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
            )

            report = audit_dataset(root)

            self.assertEqual(report["counts"]["candidate_triplets"], 1)
            self.assertEqual(report["counts"]["unambiguous_triplets"], 1)
            self.assertTrue(report["assessment"]["raw_depth_candidate"])
            self.assertFalse(report["assessment"]["metric_unit_verified"])
            self.assertEqual(report["labels"]["instances"], 1)


if __name__ == "__main__":
    unittest.main()
