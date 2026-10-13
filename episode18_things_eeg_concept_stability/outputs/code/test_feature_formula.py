"""Focused regression for the corrected low-level dimension and constant field."""
import unittest

import numpy as np

from feature_extractors import gabor_response, low_level, require_compute


class LowLevelFormulaTest(unittest.TestCase):
    def test_Gabor_reflection_matches_direct_convolution_at_image_edges(self):
        require_compute()
        from scipy.ndimage import convolve
        gray = np.zeros((13, 17), dtype=np.float64)
        gray[0, 0], gray[0, 1], gray[-1, -1], gray[7, 0] = 1., .25, -.5, .75
        frequency, theta = .25, np.pi/8
        sigma = .56/frequency
        radius = int(np.ceil(3*sigma))
        y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
        kernel = np.exp(-(x*x+y*y)/(2*sigma*sigma))*np.exp(2j*np.pi*frequency*(x*np.cos(theta)+y*np.sin(theta)))
        kernel -= kernel.mean()
        kernel /= np.sqrt(np.sum(np.abs(kernel)**2))
        direct = convolve(gray.astype(np.complex128), kernel, mode="reflect")
        np.testing.assert_allclose(gabor_response(gray, kernel), direct, rtol=1e-12, atol=1e-12)

    def test_constant_blue_has_known_moments_no_texture_and_851_entries(self):
        pixels = np.zeros((8, 16, 3), dtype=np.uint8)
        pixels[..., 2] = 255
        features = low_level(pixels)
        self.assertEqual(features.shape, (851,))
        np.testing.assert_allclose(features[:96].reshape(16, 6),
                                   np.tile([0, 0, 0, 0, 1, 0], (16, 1)), atol=1e-7)
        histograms = features[96:144].reshape(3, 16)
        self.assertEqual(histograms[0, 0], 1.)
        self.assertEqual(histograms[1, 0], 1.)
        self.assertEqual(histograms[2, -1], 1.)
        np.testing.assert_allclose(histograms.sum(axis=1), 1.)
        # A spatially constant field has no non-DC power, texture or HOG.
        np.testing.assert_allclose(features[144:848], 0., atol=1e-6)
        np.testing.assert_allclose(features[-3:], [2., .0722, 0.], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
