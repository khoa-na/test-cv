import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
