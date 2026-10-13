"""One CUDA-specific numerical check, executable only in the selected GPU job.

No empirical data. The pinned tiny model and existing unequal-layer fixture
compare GPU sparse arithmetic against the independent CPU dense expression.
"""
import copy
import os
from pathlib import Path
import socket
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_loss import content_loss_step
from content_pair_cuda_measurement import CACHE_MODE, _cached_error_details
from schedule import load_pinned_converter_factory
from test_content_loss import dense_reference, fixture


class ContentCudaTests(unittest.TestCase):
    def test_pinned_converter_cuda_sparse_step_matches_CPU_dense_autograd(self):
        code = "fixed_layer_pass_order_required"
        self.assertEqual(_cached_error_details(ValueError(code), CACHE_MODE), {"error_code": code})
        self.assertEqual(_cached_error_details(ValueError(code), "bounded_column_stripe"), {})
        for error in (ValueError("PRIVATE_VALUES_AND_PATHS"), ValueError(code + "\nPRIVATE_VALUES"),
                      ValueError(code, "PRIVATE_VALUES"), RuntimeError(code)):
            self.assertEqual(_cached_error_details(error, CACHE_MODE), {})
        self.assertTrue(os.environ.get("SLURM_JOB_ID", "").isdigit())
        self.assertNotIn("-ln", socket.gethostname())
        self.assertTrue(torch.cuda.is_available(), "CUDA check cannot pass by skipping the selected device")
        self.assertEqual(torch.cuda.device_count(), 1)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        previous = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32)
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            factory = load_pinned_converter_factory()
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(42).get_state())
                reference = factory(9, 5)
            model = copy.deepcopy(reference).to("cuda:0")
            for actual, expected in zip(model.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(actual.detach().cpu(), expected.detach()))
            layers, blocks = fixture()
            source = torch.arange(27, dtype=torch.float32).reshape(3, 9) / 7 - .6
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=5e-5, betas=(.5, .999))
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(.5, .999))
            reference_optimizer.zero_grad(set_to_none=True)
            converted = reference(source)
            converted.retain_grad()
            expected_loss, expected_norms, expected_weights = dense_reference(converted, layers, blocks)
            expected_loss.backward()
            reference_optimizer.step()
            with torch.autocast(device_type="cuda", enabled=False):
                actual = content_loss_step(model, optimizer, source.to("cuda:0"), layers,
                    lambda layer: iter(blocks[layer.layer]), expected_rows=3,
                    max_block_units=2, row_chunk_size=2, backend="sparse")
            torch.cuda.synchronize(0)
            for observed, expected in ((actual.loss.cpu(), expected_loss.detach()),
                                       (actual.norms.cpu(), expected_norms.detach()),
                                       (actual.layer_weights.cpu(), expected_weights.detach()),
                                       (actual.response_adjoint.cpu(), converted.grad)):
                torch.testing.assert_close(observed, expected, rtol=2e-6, atol=2e-6)
            for observed, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(observed.grad.cpu(), expected.grad, rtol=3e-6, atol=2e-6)
                torch.testing.assert_close(observed.detach().cpu(), expected.detach(), rtol=2e-6, atol=2e-6)
                self.assertEqual(float(optimizer.state[observed]["step"]), 1.)
            self.assertFalse(actual.bitwise_source_equivalence)
            for records in blocks.values():
                for block in records:
                    self.assertIsNone(block.weights.grad)
                    self.assertIsNone(block.bias.grad)
                    self.assertIsNone(block.standardized_features.grad)
        finally:
            torch.set_float32_matmul_precision(previous[0])
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
            torch.backends.cudnn.allow_tf32 = previous[2]


if __name__ == "__main__":
    unittest.main()
