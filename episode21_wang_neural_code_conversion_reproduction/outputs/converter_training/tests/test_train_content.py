"""Focused loop tests; run only in coordinator-owned Slurm compute.

The numerical engine is tested separately. Here a tiny mocked step isolates
the source's optimizer/LR order, complete row support and no-success-on-abort.
No real response, feature, decoder or checkpoint file is accessed.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_content as train


class ContentTrainingLoopTests(unittest.TestCase):
    def test_all_1024_steps_keep_full_rows_and_tagged_lr_order(self):
        source = torch.zeros((4712, 4), dtype=torch.float32)
        model = torch.nn.Linear(4, 4)
        rates, active_counts, reports = [], [], []

        def step(actual_model, optimizer, actual_source, active, provider, **kwargs):
            self.assertIs(actual_model, model)
            self.assertIs(actual_source, source)
            self.assertEqual(kwargs["expected_rows"], 4712)
            self.assertEqual(optimizer.defaults["betas"], (0.5, 0.999))
            rates.append(optimizer.param_groups[0]["lr"])
            active_counts.append(sum(item.units for item in active))
            optimizer.zero_grad(set_to_none=True)
            for parameter in actual_model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return SimpleNamespace(loss=torch.tensor(2.0))

        with mock.patch.object(train, "content_loss_step", side_effect=step):
            result = train.fit_content_pair(model, source, train.make_channel_schedule(),
                lambda active: (), check=lambda: None, max_block_units=64,
                row_chunk_size=32, backend="sparse", on_iteration=reports.append)
        self.assertEqual(len(rates), 1024)
        self.assertEqual(active_counts, [151096] * 1024)
        self.assertEqual(rates[:513], [0.00005] * 513)
        self.assertAlmostEqual(rates[513], 0.00005 * 511 / 512)
        self.assertAlmostEqual(rates[-1], 0.00005 / 512)
        self.assertEqual(reports[-1]["completed_iterations"], 1024)
        self.assertEqual(result.iterations, 1024)
        self.assertEqual(result.rows, 4712)
        self.assertFalse(result.bitwise_source_equivalence)

    def test_incomplete_row_support_cannot_enter_optimizer(self):
        with mock.patch.object(train, "content_loss_step") as engine:
            with self.assertRaisesRegex(ValueError, "4712"):
                train.fit_content_pair(torch.nn.Linear(4, 4), torch.zeros((10, 4)),
                    train.make_channel_schedule(), lambda active: (), check=lambda: None,
                    max_block_units=64, row_chunk_size=32, backend="sparse")
            engine.assert_not_called()

    def test_deadline_failure_propagates_without_completing_or_restarting(self):
        checks, steps = [], []

        def check():
            checks.append(True)
            if len(checks) == 3:
                raise TimeoutError("allocation deadline")

        def step(model, optimizer, *args, **kwargs):
            steps.append(True)
            optimizer.step()
            return SimpleNamespace(loss=torch.tensor(1.0))

        with mock.patch.object(train, "content_loss_step", side_effect=step):
            with self.assertRaisesRegex(TimeoutError, "allocation deadline"):
                train.fit_content_pair(torch.nn.Linear(4, 4), torch.zeros((4712, 4)),
                    train.make_channel_schedule(), lambda active: (), check=check,
                    max_block_units=64, row_chunk_size=32, backend="dense")
        self.assertEqual(len(steps), 1)


if __name__ == "__main__":
    unittest.main()
