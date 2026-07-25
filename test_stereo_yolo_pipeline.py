import unittest

import numpy as np

from benchmark_stereo_yolo import summarize
from stereo_yolo_pipeline import fuse_mask, road_surface_area_mm2


class StereoYOLOPipelineTest(unittest.TestCase):
    def test_detection_gates_residual_mask(self):
        detection = np.zeros((20, 20), dtype=bool)
        residual = np.zeros((20, 20), dtype=bool)
        detection[4:16, 4:16] = True
        residual[8:14, 8:14] = True

        np.testing.assert_array_equal(fuse_mask(detection, residual), residual)

    def test_front_parallel_area_matches_pixel_footprint(self):
        disparity = np.full((20, 20), 60, dtype=np.float32)
        mask = np.zeros_like(disparity, dtype=bool)
        mask[5:15, 5:15] = True

        area = road_surface_area_mm2(mask, disparity, focal_px=600, baseline_mm=120)

        self.assertAlmostEqual(area, 100 * (120 / 60) ** 2, places=3)

    def test_benchmark_counts_fusion_failures_as_end_to_end_failures(self):
        rows = [
            {
                "model": "model2",
                "detections": 1,
                "fusion_success": True,
                "strong_alignment": True,
                "relative_error": 0.05,
                "total_ms": 50,
            },
            {
                "model": "model3",
                "detections": 1,
                "fusion_success": False,
                "strong_alignment": False,
                "relative_error": None,
                "total_ms": 100,
            },
        ]

        result = summarize(rows)

        self.assertEqual(result["held_out"]["fusion_coverage"], 0.5)
        self.assertEqual(result["held_out"]["end_to_end_within_15_percent"], 0.5)


if __name__ == "__main__":
    unittest.main()
