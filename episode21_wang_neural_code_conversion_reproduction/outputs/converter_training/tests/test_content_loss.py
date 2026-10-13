"""Tiny CPU numerical regressions. Prepared only; run in coordinated Slurm."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from content_loss import ActiveLayer, LossBlock, content_loss_and_adjoint, content_loss_step


def fixture():
    layers = (ActiveLayer("conv_fixture", 2, 3), ActiveLayer("fc_fixture", None, 1))
    blocks = {
        "conv_fixture": [
            LossBlock(0, torch.tensor([[3, 0], [1, 4]], dtype=torch.int64),
                      torch.tensor([[.5, -.25], [.75, .125]], dtype=torch.float32),
                      torch.tensor([.1, -.2], dtype=torch.float32),
                      torch.tensor([[1., 2.], [2., -1.], [.5, 3.]], dtype=torch.float32)),
            LossBlock(2, torch.tensor([[4, 2]], dtype=torch.int64),
                      torch.tensor([[1.25, -.5]], dtype=torch.float32),
                      torch.tensor([.3], dtype=torch.float32),
                      torch.tensor([[.5], [1.5], [-2.]], dtype=torch.float32)),
        ],
        "fc_fixture": [
            LossBlock(0, torch.tensor([[2, 0]], dtype=torch.int64),
                      torch.tensor([[.2, .6]], dtype=torch.float32),
                      torch.tensor([-.4], dtype=torch.float32),
                      torch.tensor([[2.], [4.], [1.]], dtype=torch.float32)),
        ],
    }
    return layers, blocks


def dense_reference(converted, layers, blocks):
    """Independent complete-layer dense autograd expression (no chunk engine)."""
    norms, errors = [], []
    for active in layers:
        targets = torch.cat([block.standardized_features for block in blocks[active.layer]], dim=1)
        weights = torch.zeros((converted.shape[1], active.units), dtype=torch.float32)
        bias = torch.cat([block.bias for block in blocks[active.layer]])
        for block in blocks[active.layer]:
            for local in range(block.weights.shape[0]):
                for selected in range(block.weights.shape[1]):
                    weights[block.eligible_columns[local, selected], block.unit_start + local] = block.weights[local, selected]
        norms.append(torch.linalg.vector_norm(targets))
        errors.append(((converted @ weights + bias - targets) ** 2).sum())
    norms = torch.stack(norms).to(torch.float32)
    layer_weights = norms.square().reciprocal()
    layer_weights = layer_weights / layer_weights.sum()
    loss = sum(weight * error for weight, error in zip(layer_weights, errors))
    return loss, norms, layer_weights


class CountingConverter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 5))
        self.forward_calls = 0
        self.backward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        output = self.net(value)
        output.register_hook(self._backward)
        return output

    def _backward(self, gradient):
        self.backward_calls += 1
        return gradient


class CountingAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=5e-5, betas=(.5, .999))
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


class ContentLossTests(unittest.TestCase):
    def test_unequal_complete_layers_match_dense_forward_and_input_gradient(self):
        layers, blocks = fixture()
        values = torch.tensor([[.2, -.3, .4, .1, .5], [.8, .2, -.1, .9, .3],
                               [-.5, .7, .6, -.2, 1.]], dtype=torch.float32)
        reference_input = values.clone().requires_grad_(True)
        expected_loss, expected_norms, expected_weights = dense_reference(reference_input, layers, blocks)
        expected_loss.backward()
        calls = []

        def provider(active):
            calls.append((active.layer, active.channel))
            return iter(blocks[active.layer])

        for backend in ("sparse", "dense"):
            for row_chunk in (1, 2, 3):
                with self.subTest(backend=backend, row_chunk=row_chunk):
                    calls.clear()
                    actual = content_loss_and_adjoint(values, layers, provider, expected_rows=3,
                        max_block_units=2, row_chunk_size=row_chunk, backend=backend)
                    torch.testing.assert_close(actual.loss, expected_loss.detach(), rtol=2e-6, atol=2e-6)
                    torch.testing.assert_close(actual.response_adjoint, reference_input.grad, rtol=2e-6, atol=2e-6)
                    torch.testing.assert_close(actual.norms, expected_norms, rtol=2e-6, atol=2e-6)
                    torch.testing.assert_close(actual.layer_weights, expected_weights, rtol=2e-6, atol=2e-6)
                    self.assertEqual(calls, [(layer.layer, layer.channel) for layer in layers] * 2)
                    self.assertFalse(actual.loss.requires_grad)
                    self.assertFalse(actual.response_adjoint.requires_grad)
                    self.assertFalse(actual.bitwise_source_equivalence)
                    # Mean-MSE would additionally divide these unequal layers by 9 and 3.
                    mean_loss = (actual.layer_weights * actual.layer_squared_errors /
                                 torch.tensor([9., 3.], dtype=torch.float32)).sum()
                    self.assertGreater(abs(float(actual.loss - mean_loss)), .1)

    def test_single_full_batch_backward_and_Adam_step_match_dense_autograd(self):
        layers, blocks = fixture()
        torch.manual_seed(19)
        model = CountingConverter()
        reference = copy.deepcopy(model)
        optimizer, reference_optimizer = CountingAdam(model.parameters()), CountingAdam(reference.parameters())
        source = torch.tensor([[1., .2, -.3], [-.4, .6, .8], [.7, -.1, .5]], dtype=torch.float32)
        reference_optimizer.zero_grad(set_to_none=True)
        expected, _, _ = dense_reference(reference(source), layers, blocks)
        expected.backward()
        reference_optimizer.step()
        actual = content_loss_step(model, optimizer, source, layers, lambda layer: iter(blocks[layer.layer]),
            expected_rows=3, max_block_units=2, row_chunk_size=2, backend="sparse")
        self.assertEqual((model.forward_calls, model.backward_calls, optimizer.steps), (1, 1, 1))
        torch.testing.assert_close(actual.loss, expected.detach(), rtol=2e-6, atol=2e-6)
        for observed, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(observed.grad, wanted.grad, rtol=3e-6, atol=2e-6)
            torch.testing.assert_close(observed, wanted, rtol=2e-6, atol=2e-6)
            self.assertEqual(float(optimizer.state[observed]["step"]), 1.)
        for records in blocks.values():
            for block in records:
                self.assertIsNone(block.weights.grad)
                self.assertIsNone(block.bias.grad)
                self.assertIsNone(block.standardized_features.grad)

    def test_invalid_complete_norm_or_incomplete_layer_cannot_update_model(self):
        layers, blocks = fixture()
        for failure in ("zero_norm", "missing_units", "frozen_target_gradient"):
            with self.subTest(failure=failure):
                changed = {key: list(value) for key, value in blocks.items()}
                first = changed["fc_fixture"][0]
                if failure == "zero_norm":
                    changed["fc_fixture"] = [replace(first, standardized_features=torch.zeros_like(first.standardized_features))]
                elif failure == "missing_units":
                    changed["conv_fixture"] = changed["conv_fixture"][:1]
                else:
                    changed["fc_fixture"] = [replace(first, standardized_features=first.standardized_features.clone().requires_grad_(True))]
                model = CountingConverter()
                original = [parameter.detach().clone() for parameter in model.parameters()]
                optimizer = CountingAdam(model.parameters())
                with self.assertRaises(ValueError):
                    content_loss_step(model, optimizer, torch.ones((3, 3)), layers,
                        lambda layer: iter(changed[layer.layer]), expected_rows=3,
                        max_block_units=2, row_chunk_size=2, backend="dense")
                self.assertEqual((model.backward_calls, optimizer.steps), (0, 0))
                self.assertFalse(optimizer.state)
                for parameter, before in zip(model.parameters(), original):
                    torch.testing.assert_close(parameter, before, rtol=0, atol=0)

    def test_missing_rows_invalid_axis_or_duplicate_voxel_refuse(self):
        layers, blocks = fixture()
        first = blocks["conv_fixture"][0]
        malformed = (
            replace(first, unit_start=1),
            replace(first, standardized_features=first.standardized_features[:2]),
            replace(first, eligible_columns=torch.tensor([[5, 0], [1, 4]], dtype=torch.int64)),
            replace(first, eligible_columns=torch.tensor([[3, 3], [1, 4]], dtype=torch.int64)),
            replace(first, weights=first.weights.clone().requires_grad_(True)),
        )
        for bad in malformed:
            changed = {**blocks, "conv_fixture": [bad, blocks["conv_fixture"][1]]}
            with self.subTest(block=bad), self.assertRaises(ValueError):
                content_loss_and_adjoint(torch.ones((3, 5)), layers,
                    lambda layer: iter(changed[layer.layer]), expected_rows=3,
                    max_block_units=2, row_chunk_size=2, backend="sparse")
        with self.assertRaises(ValueError):
            content_loss_and_adjoint(torch.ones((2, 5)), layers,
                lambda layer: iter(blocks[layer.layer]), expected_rows=3,
                max_block_units=2, row_chunk_size=2, backend="dense")


if __name__ == "__main__":
    unittest.main()
