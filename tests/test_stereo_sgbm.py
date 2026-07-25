import unittest

import numpy as np

from benchmarks.benchmark_fan_stereo import robust_z_extent
from pipelines.stereo_sgbm import (
    expand_residual_mask,
    fit_road_disparity,
    measure_pothole,
)


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

    def test_residual_seed_expands_to_connected_pothole(self):
        residual = np.zeros((100, 100), dtype=np.float32)
        residual[30:70, 30:70] = 5
        residual[45:55, 45:55] = 10
        seed = residual == 10

        mask, _ = expand_residual_mask(
            residual, np.ones_like(seed), seed, quantile=0.85
        )

        self.assertEqual(np.count_nonzero(mask), 1600)


if __name__ == "__main__":
    unittest.main()
