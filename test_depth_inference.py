import unittest

import numpy as np

from depth_inference import colorize_depth, preprocess


class DepthInferenceTest(unittest.TestCase):
    def test_preprocess_and_colorize(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        tensor = preprocess(image, 196)
        preview = colorize_depth(np.arange(48 * 64, dtype=np.float32).reshape(48, 64))

        self.assertEqual(tensor.shape, (1, 3, 196, 196))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(preview.shape, (48, 64, 3))

    def test_size_must_be_multiple_of_fourteen(self):
        with self.assertRaises(ValueError):
            preprocess(np.zeros((10, 10, 3), dtype=np.uint8), 225)


if __name__ == "__main__":
    unittest.main()
