import unittest

import numpy as np

from benchmark_fan_stereo import robust_z_extent
from stereo_sgbm import fit_road_disparity, measure_pothole


class StereoSGBMTest(unittest.TestCase):
    def test_road_fit_rejects_pothole_outlier_and_recovers_depth(self):
        height, width = 120, 160
        y, x = np.mgrid[:height, :width]
        road = (60 + 10 * x / width + 80 * y / height).astype(np.float32)
        disparity = road.copy()
        mask = np.zeros((height, width), dtype=bool)
        mask[90:110, 70:100] = True
        disparity[mask] -= 8

        fitted = fit_road_disparity(disparity, stride=1)
        geometry = measure_pothole(disparity, fitted, mask, 700, 120)

        self.assertLess(float(np.median(np.abs(fitted - road))), 0.1)
        self.assertGreater(geometry["depth_mm_p90"], 20)

    def test_robust_z_extent_rejects_extreme_outliers(self):
        z = np.concatenate([np.linspace(0, 40, 1000), [-1000, 1000]])
        points = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])

        self.assertGreater(robust_z_extent(points), 39)
        self.assertLess(robust_z_extent(points), 41)


if __name__ == "__main__":
    unittest.main()
