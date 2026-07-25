import unittest

import cv2
import numpy as np

from pothole_pipeline import estimate_geometry, regression_severity, severity


class PotholePipelineTest(unittest.TestCase):
    def test_depth_is_measured_against_sloped_road(self):
        height, width = 120, 160
        y, x = np.mgrid[:height, :width]
        depth = (1.0 + x * 0.001 + y * 0.002).astype(np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (80, 60), 20, 1, -1)
        depth[mask > 0] += 0.2

        geometry = estimate_geometry(mask, depth, ring_width=10)

        self.assertAlmostEqual(geometry["relative_depth"], 0.2, places=3)
        self.assertEqual(geometry["area_pixels"], int(mask.sum()))
        self.assertFalse(geometry["units"]["metric_calibrated"])
        self.assertEqual(severity(geometry), "severe")

    def test_mask_and_depth_shapes_must_match(self):
        with self.assertRaises(ValueError):
            estimate_geometry(np.ones((10, 10)), np.ones((11, 10)))

    def test_regression_severity_uses_depth_and_area(self):
        self.assertEqual(
            regression_severity({"normalized_depth": 0.01, "relative_area": 0.01}),
            "minor",
        )
        self.assertEqual(
            regression_severity({"normalized_depth": 0.03, "relative_area": 0.01}),
            "moderate",
        )
        self.assertEqual(
            regression_severity({"normalized_depth": 0.06, "relative_area": 0.01}),
            "severe",
        )


if __name__ == "__main__":
    unittest.main()
