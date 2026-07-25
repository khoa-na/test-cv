import unittest

import numpy as np

from benchmark_roi_pipeline import greedy_match, mask_iou
from benchmark_depth_pothrgbd import evaluate, fit_scale, split_name


class DepthBenchmarkTest(unittest.TestCase):
    def test_scale_is_fit_only_from_calibration_records(self):
        records = [
            {
                "split": "calibration",
                "gt_depth": value * 10,
                "prediction": value,
            }
            for value in (1.0, 2.0, 3.0)
        ]
        records.append({"split": "test", "gt_depth": 1000.0, "prediction": 1.0})

        scale = fit_scale(records, "gt_depth", "prediction", min_target=0)
        metrics = evaluate(records, "gt_depth", "prediction", scale, min_target=0)

        self.assertEqual(scale, 10.0)
        self.assertAlmostEqual(metrics["mean_absolute_relative_error"], 0.99)

    def test_split_is_stable(self):
        first = [split_name(str(index), 0.2, 42) for index in range(100)]
        second = [split_name(str(index), 0.2, 42) for index in range(100)]

        self.assertEqual(first, second)
        self.assertIn("test", first)
        self.assertIn("calibration", first)

    def test_greedy_mask_matching_is_one_to_one(self):
        first = np.zeros((10, 10), dtype=bool)
        first[1:5, 1:5] = True
        second = np.zeros((10, 10), dtype=bool)
        second[5:9, 5:9] = True

        self.assertEqual(mask_iou(first, first), 1.0)
        self.assertEqual(
            greedy_match([first, second], [first, second], threshold=0.5),
            [(1, 1, 1.0), (0, 0, 1.0)],
        )


if __name__ == "__main__":
    unittest.main()
