import unittest

import numpy as np
import torch

from pipelines.depth_regressor_inference import preprocess_roi, roi_crop
from training.train_depth_regressor import DepthRegressor, split_name


class DepthRegressorTest(unittest.TestCase):
    def test_split_groups_same_timestamp(self):
        self.assertEqual(split_name("20250305_074730"), split_name("20250305_074730"))

    def test_masked_crop_has_expected_shape(self):
        image = np.full((40, 60, 3), 127, dtype=np.uint8)
        points = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], dtype=np.int32)
        crop, mask = roi_crop(image, points, 32)
        self.assertEqual(crop.shape, (32, 32, 3))
        self.assertEqual(mask.shape, (32, 32))
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[10:30, 10:40] = 1
        image_input, geometry_input = preprocess_roi(image, full_mask, 32)
        self.assertEqual(image_input.shape, (1, 4, 32, 32))
        self.assertEqual(geometry_input.shape, (1, 6))

    def test_model_outputs_two_depth_targets(self):
        model = DepthRegressor(pretrained=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 4, 32, 32), torch.zeros(2, 6))
        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
