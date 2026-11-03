import unittest
import numpy as np
from src.geometry import assess, corners, project
from src.evaluate import fit_threshold, summarize


class PlanarTests(unittest.TestCase):
    def test_known_translation_and_no_support(self):
        xy = np.array([(x, y) for x in range(10, 110, 20) for y in range(10, 110, 20)], np.float32)
        h = np.array([[1, 0, 12], [0, 1, -6], [0, 0, 1]], dtype=float)
        result = assess(xy, project(h, xy), (120, 120), (120, 120), h, 3)
        self.assertLess(result["corner_error"], 1e-4)
        self.assertEqual(result["inlier_ratio"], 1)
        self.assertGreater(result["coverage"], 0)
        missing = assess(xy[:3], xy[:3], (120, 120), (120, 120), h, 3)
        self.assertIsNone(missing["corner_error"])
        self.assertEqual(missing["matches"], 3)

    def test_calibration_and_denominators(self):
        scores = [{"matches": n, "corner_error": e} for n, e in [(12, 1), (12, 2), (8, 9), (1, None)]]
        self.assertEqual(fit_threshold(scores, "matches", .9, 5), 12)
        metrics = summarize(scores, "matches", 8, 5)
        self.assertEqual((metrics["pairs"], metrics["accepted"], metrics["false_accepts"]), (4, 3, 1))
        self.assertAlmostEqual(metrics["coverage"], .75)
        self.assertEqual(corners((21, 14)).tolist()[2], [20, 13])


if __name__ == "__main__":
    unittest.main()
