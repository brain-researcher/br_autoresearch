"""Three small source-fidelity checks; run only in coordinator-owned compute.

No real images/features/assets, full VGG or generator model is loaded. The
private fixed-loop check uses a scalar generator and the real SGD optimizer;
it is not runtime qualification or a shortened empirical reconstruction.
"""
import unittest

import numpy as np
import torch
from torch import nn

import reconstruction_core as core


class ReconstructionCoreTests(unittest.TestCase):
    def test_source_channel_SD_scaling_dtype_owner_and_nonfinite_refusals(self):
        centered = np.array([[[0., 2.], [4., 6.]],
                             [[10., 14.], [18., 22.]]], dtype=np.float64)
        decoder_mean = np.broadcast_to(np.array([10., 100.])[:, None, None], (2, 2, 2))[None].copy()
        decoded = centered[None] + decoder_mean
        supplied_std = np.array([2., 4.], dtype=np.float64)
        result = core._scale_feature(decoded, decoder_mean, supplied_std)
        # Independent sample-SD oracle for the two channels: sqrt(20/3)
        # and sqrt(80/3). Their mean, rather than flattened SD, is required.
        denominator = (np.sqrt(20. / 3.) + np.sqrt(80. / 3.)) / 2.
        expected = ((centered - 9.5) / denominator * 3. + 9.5)[None] + decoder_mean
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(result.dtype, np.float64)
        self.assertFalse(np.allclose(result,
            ((centered - 9.5) / centered.std(ddof=1) * 3. + 9.5)[None] + decoder_mean))
        np.testing.assert_array_equal(decoded, centered[None] + decoder_mean)

        fc = np.array([[1., 3., 5., 7.]], dtype=np.float32)
        fc_mean = np.full_like(fc, 10.)
        expected_fc = ((fc[0] - np.float32(4.)) / np.float32(np.sqrt(20. / 3.)))
        expected_fc = (expected_fc * np.float32(2.) + np.float32(4.))[None] + fc_mean
        actual_fc = core._scale_feature(fc + fc_mean, fc_mean, np.array([2.], dtype=np.float32))
        np.testing.assert_array_equal(actual_fc, expected_fc)
        self.assertEqual(actual_fc.dtype, np.float32)
        for std in (np.zeros(1), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(std=std), self.assertRaises(core.ReconstructionUnavailable):
                core._scale_feature(decoded, decoder_mean, std)
        with self.assertRaisesRegex(core.ReconstructionUnavailable, "decoded_feature_scale"):
            core._scale_feature(decoder_mean.copy(), decoder_mean, supplied_std)

        features = {layer: np.array([[i + 1., i + 2.]], dtype=np.float64)
                    for i, layer in enumerate(core.LAYERS)}
        weights = core._layer_weights(features)
        norms = np.asarray([np.sqrt((i + 1.)**2 + (i + 2.)**2)
                            for i in range(19)], dtype=np.float32)
        oracle = np.reciprocal(np.square(norms))
        oracle /= oracle.sum()
        np.testing.assert_array_equal(list(weights.values()), oracle)
        for value in (0., np.nan, np.inf):
            malformed = dict(features, fc8=np.full((1, 2), value))
            with self.subTest(value=value), self.assertRaises(core.ReconstructionUnavailable):
                core._layer_weights(malformed)

        kwargs = dict(decoder_training_means={}, public_feature_std={}, encoder=None,
            generator=None, latent_upper_bound=np.ones(4096), method="content_loss",
            source="sub-03", target="sub-01", decoder_owner="sub-03", budget_check=lambda: None)
        with self.assertRaisesRegex(core.ReconstructionUnavailable, "target_decoder_mean_owner"):
            core.reconstruct({}, **kwargs)
        kwargs["decoder_owner"] = "sub-01"
        with self.assertRaisesRegex(core.ReconstructionUnavailable, "all_nineteen"):
            core.reconstruct({}, **kwargs)

    def test_literal_DIST_unweighted_term_reverse_sum_and_gradient(self):
        # Independent literal coefficients in source's reverse layer order.
        alpha_reverse = (
            .06661523244761258, .03871265134313895, .000629031742843134,
            .5310432634795051, .01975314483213067, .714010791024532,
            .08536182104218124, .030798346318926202, .004025735147052829,
            .0021716504774059618, .02880295296139471, .014169279688225732,
            .00019573287900056505, .0004887923569668929, .006857140440209977,
            .08084213863904581, .00024056214287883663, .003886371646003732,
            .009952859626673973)
        beta_reverse = (
            .008304840607742099, .044481711593671994, .038457933646483915,
            .0012780195483159135, .0018775814111698145, .5074163077203029,
            .002337825161420017, .7100372437615771, .5166895849277143,
            .03998274022264576, .04328555659354602, .024733951474856346,
            .0004859871528150426, .039778524165843814, .0002639605292406699,
            .02472305546171304, .12888847991806807, .008627502425502372,
            .000865427897168344)
        self.assertEqual(tuple(core.ALPHA[layer] for layer in reversed(core.LAYERS)), alpha_reverse)
        self.assertEqual(tuple(core.BETA[layer] for layer in reversed(core.LAYERS)), beta_reverse)
        x = torch.tensor([-2., -1., 0., 1., 2., 4.], dtype=torch.float64, requires_grad=True)
        targets, activations, weights = {}, {}, {}
        expected_terms, incorrectly_weighted = {}, {}
        for i, layer in enumerate(core.LAYERS):
            shape = (1, 6) if layer.startswith("fc") else (1, 2, 1, 3)
            activation = (x * (1. + i / 20.) + i / 7.).reshape(shape)
            target = torch.tensor([3., 1., -1., 4., 2., 0.], dtype=torch.float64).reshape(shape) + i / 13.
            activations[layer], targets[layer] = activation, target
            weight = np.float32((i + 1) / 190.)
            weights[layer] = weight
            groups = 1 if layer.startswith("fc") else 2
            rows_x, rows_y = activation.reshape(groups, -1), target.reshape(groups, -1)
            means, covariances = [], []
            for row_x, row_y in zip(rows_x, rows_y):
                count = row_x.numel()
                mx, my = row_x.sum() / count, row_y.sum() / count
                vx = torch.dot(row_x - mx, row_x - mx) / count
                vy = torch.dot(row_y - my, row_y - my) / count
                cov = torch.dot(row_x, row_y) / count - mx * my
                means.append((2 * mx * my + 1e-6) / (mx * mx + my * my + 1e-6))
                covariances.append((2 * cov + 1e-6) / (vx + vy + 1e-6))
            index = 18 - i
            dist = -343639 * (sum(means) / groups * alpha_reverse[index]
                              + sum(covariances) / groups * beta_reverse[index])
            delta = (activation - target).flatten()
            mse = torch.dot(delta, delta)
            expected_terms[layer] = mse * weight + dist
            incorrectly_weighted[layer] = (mse + dist) * weight
        access_order = []
        class RecordingDict(dict):
            def __getitem__(self, key):
                access_order.append(key)
                return super().__getitem__(key)
        actual = core._objective(RecordingDict(activations), targets, weights, lambda: None)
        expected, wrong = 0., 0.
        for layer in reversed(core.LAYERS):
            expected = expected + expected_terms[layer]
            wrong = wrong + incorrectly_weighted[layer]
        self.assertEqual(access_order, list(reversed(core.LAYERS)))
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-8)
        actual_gradient = torch.autograd.grad(actual, x, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, x)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-8)
        self.assertGreater(abs(float(actual.detach() - wrong.detach())), 1000.)
        identical = torch.tensor([[1., -2., 4.]], dtype=torch.float32)
        expected_identical = -(core.ALPHA["fc8"] + core.BETA["fc8"]) * 343639
        self.assertAlmostEqual(float(core._dist(identical, identical, "fc8")), expected_identical, delta=.02)

    def test_fixed_200_step_latent_oracle_prefinal_output_crop_pixels_and_cancel(self):
        class ScalarGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls, self.first = 0, None
            def forward(self, latent):
                if self.first is None:
                    self.first = latent.detach().cpu().numpy().copy()
                self.calls += 1
                return latent.mean().reshape(1, 1, 1, 1).expand(1, 3, 256, 256)
        generator = ScalarGenerator()
        upper = np.full(4096, 1000., dtype=np.float64)
        upper[0], upper[1] = .01, 0.
        initial = np.random.RandomState(42).normal(0, 1, 4096).astype(np.float32)
        random_state = np.random.get_state()
        np.testing.assert_array_equal(core._initial_latent(), initial)
        np.testing.assert_array_equal(core._initial_latent(), initial)
        current_state = np.random.get_state()
        self.assertEqual(random_state[0], current_state[0])
        np.testing.assert_array_equal(random_state[1], current_state[1])
        self.assertEqual(random_state[2:], current_state[2:])
        expected_value, velocity = initial.copy(), np.zeros(4096, dtype=np.float32)
        for iteration in range(200):
            before_last_update = expected_value.mean(dtype=np.float32)
            learning_rate = 2. + iteration * (1e-10 - 2.) / 200
            # The scalar generator and objective below give -1/4096 in every
            # coordinate, hence the normalized gradient is exactly -1.
            velocity = np.float32(.9) * velocity - np.float32(1.)
            expected_value = expected_value - np.float32(learning_rate) * velocity
            expected_value = np.float32(.99) * expected_value
            expected_value = np.minimum(np.maximum(expected_value, 0.), upper).astype(np.float32)
        actual = core._iterate(generator, nn.Identity(), lambda image: -image[0, 0, 0, 0],
                               upper, torch.device("cpu"), lambda: None)
        self.assertEqual(generator.calls, 200)
        np.testing.assert_array_equal(generator.first[0], initial)
        self.assertGreater(generator.first[0, 0], upper[0])
        self.assertLess(generator.first[0, 1], 0.)
        self.assertEqual(actual.shape, (3, 224, 224))
        np.testing.assert_allclose(actual, before_last_update, rtol=3e-6, atol=3e-6)
        self.assertGreater(abs(float(actual[0, 0, 0] - expected_value.mean())), .01)
        self.assertAlmostEqual(core._learning_rate(199), .0100000000995, places=13)
        self.assertNotEqual(core._learning_rate(199), 1e-10)

        grid = torch.arange(256 * 256, dtype=torch.float32).reshape(1, 1, 256, 256)
        cropped = core._crop_generated(grid.expand(1, 3, 256, 256))
        self.assertEqual(float(cropped[0, 0, 0, 0]), 16 * 256 + 16)
        self.assertEqual(float(cropped[0, 2, -1, -1]), 239 * 256 + 239)
        bgr = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        raw, pixels = core._postprocess(bgr)
        expected_raw = np.moveaxis(bgr[[2, 1, 0]], 0, -1) + np.array([123., 117., 104.], dtype=np.float32)
        ordered = np.sort(expected_raw.ravel())
        low = .66 * ordered[0] + .34 * ordered[1]
        high = .34 * ordered[-2] + .66 * ordered[-1]
        clipped = np.clip(expected_raw, low, high)
        expected_pixels = np.uint8((clipped - clipped.min()) * (255. / (clipped.max() - clipped.min())))
        np.testing.assert_array_equal(raw, expected_raw)
        np.testing.assert_array_equal(pixels, expected_pixels)
        self.assertFalse(raw.flags.writeable)
        self.assertFalse(pixels.flags.writeable)
        constant_bgr = np.broadcast_to(np.array([-97., -110., -116.], dtype=np.float32)[:, None, None], (3, 2, 3))
        constant_raw, constant_pixels = core._postprocess(constant_bgr)
        np.testing.assert_array_equal(constant_raw, np.full((2, 3, 3), 7.))
        self.assertFalse(constant_pixels.any())

        checks = [0]
        def cancel():
            checks[0] += 1
            if checks[0] == 3:
                raise TimeoutError("fixture cancellation")
        interrupted = ScalarGenerator()
        with self.assertRaisesRegex(TimeoutError, "fixture cancellation"):
            core._iterate(interrupted, nn.Identity(), lambda image: -image[0, 0, 0, 0],
                          upper, torch.device("cpu"), cancel)
        self.assertEqual(interrupted.calls, 1)


if __name__ == "__main__":
    unittest.main()
