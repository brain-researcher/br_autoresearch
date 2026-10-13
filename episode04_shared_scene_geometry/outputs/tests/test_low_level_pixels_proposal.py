"""Three analytic checks on generated pixels only; no source/image-file reads."""
from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from low_level_pixels_proposal import descriptor, displayed_rgb, low_level_rdm


class PixelMomentTests(unittest.TestCase):
    def test_constant_color_norms_and_distances(self):
        black = np.zeros((1000, 1000, 3), dtype=np.uint8)
        white = np.full_like(black, 255)
        red = black.copy()
        red[:, :, 0] = 255
        vectors = np.stack([descriptor(black), descriptor(white), descriptor(red)])
        self.assertEqual(vectors.shape, (3, 3150))
        # Means alone contribute: three channels / six statistics = 1/2.
        np.testing.assert_allclose(np.sum(vectors**2, axis=1), [0, .5, 1/6], atol=1e-13)
        expected_squared = [[0, .5, 1/6], [.5, 0, 1/3], [1/6, 1/3, 0]]
        np.testing.assert_allclose(low_level_rdm(vectors)**2, expected_squared, atol=1e-13)

    def test_spatial_block_contrast_against_analytic_distance(self):
        alternating = ((np.arange(1000)[:, None]+np.arange(1000)[None, :]) % 2 * 255).astype(np.uint8)
        checker = np.repeat(alternating[:, :, None], 3, axis=2)
        half = np.zeros_like(checker)
        half[:, 500:] = 255
        # Both images have identical global RGB mean/SD. Checker blocks all have
        # mean=SD=1/2. At grid5, 4/5 of half-image blocks are pure; at10/20 all
        # are pure. Squared distance = (4/5 + 1 + 1) / 12 = 7/30.
        value = low_level_rdm(np.stack([descriptor(checker), descriptor(half)]))[0, 1]
        self.assertAlmostEqual(value**2, 7/30, places=12)

    def test_alpha_background_endpoints_and_rounding(self):
        rgba = np.empty((1000, 1000, 4), dtype=np.uint8)
        rgba[:, :, :3] = [0, 255, 37]
        for section, alpha in enumerate((0, 1, 128, 254, 255)):
            rgba[:, section*200:(section+1)*200, 3] = alpha
        encoded = BytesIO()
        Image.fromarray(rgba).save(encoded, format='PNG')
        result = displayed_rgb(encoded.getvalue())
        expected = ([128, 128, 128], [127, 128, 128], [64, 192, 82],
                    [1, 255, 37], [0, 255, 37])
        for section, color in enumerate(expected):
            np.testing.assert_array_equal(result[10, section*200+10], color)


if __name__ == '__main__':
    unittest.main()
