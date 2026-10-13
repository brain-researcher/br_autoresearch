"""Three focused measurement-composition checks; coordinator compute only.

Reuse the existing tiny-width canonical panel and full 4,712-row fixture.
The actual driver and 1,024-step loop run; a tiny Adam step replaces numerical
loss work, and the measurement callback must stop them after four updates.
This does not measure production throughput or access empirical inputs.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import content_pair_measurement as measurement
import test_content_pair_driver as driver_fixtures


class ContentPairMeasurementTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with driver_fixtures.ContentPairDriverTests().fixture() as base, ExitStack() as stack:
            self.assertEqual(base.source.axis.participant, measurement.SOURCE)
            pair = measurement.driver.interface.bind_content_pair(base.panel.manifest,
                base.source, target=measurement.TARGET, budget=base.budget)
            context = SimpleNamespace(allowed=True)
            requests, steps = [], []

            def resolver(actual_context, **request):
                self.assertIs(actual_context, context)
                self.assertEqual(request["operation"], "measure_content_pair_four_updates")
                self.assertEqual((request["source"], request["target"]), ("sub-03", "sub-01"))
                self.assertIs(request["budget"], base.budget)
                requests.append(request)
                if not context.allowed:
                    raise PermissionError("fixture measurement admission denied")

            evidence = measurement.inputs.LossCorrectionEvidence(
                measurement.inputs.CORRECTION_RELATIVE, "46881096",
                measurement.inputs.CORRECTED_LOSS, "fixture_numpy", "fixture_torch")
            identity = (base.budget.allocation["job_id"], base.budget.allocation["hostname"])
            bound = measurement.inputs.BoundContentInputs(context, resolver, base.budget,
                identity, evidence, base.panel, pair)
            prepare = mock.Mock(return_value=bound)
            provider = object()

            def step(model, optimizer, values, active, supplied_provider, **limits):
                self.assertEqual(values.shape, (4712, len(base.source.axis.eligible_native_ids)))
                self.assertEqual(len(active), 19)
                self.assertEqual(sum(item.units for item in active), 151096)
                self.assertIs(supplied_provider, provider)
                self.assertEqual(limits, dict(expected_rows=4712, max_block_units=256,
                    row_chunk_size=512, backend="sparse"))
                self.assertEqual(optimizer.param_groups[0]["lr"], .00005)
                self.assertEqual(optimizer.param_groups[0]["betas"], (.5, .999))
                optimizer.zero_grad(set_to_none=True)
                for parameter in model.parameters():
                    parameter.grad = torch.zeros_like(parameter)
                optimizer.step()
                steps.append(tuple((value.layer, value.channel) for value in active))
                # The wrapper must never copy the incoming loss into its report.
                return SimpleNamespace(loss=torch.tensor(987654.125))

            # Deterministic operational counters, not measured fixture speed.
            ticks = []
            def snapshot():
                tick = len(ticks)
                ticks.append(tick)
                return {"monotonic": float(tick), "user_seconds": tick * .4,
                    "system_seconds": tick * .1, "peak_rss_bytes": 1000 + tick * 100,
                    "kernel_io": {key: tick * 10 for key in measurement._IO_KEYS}}

            stack.enter_context(mock.patch.object(measurement, "_snapshot", side_effect=snapshot))
            correction = stack.enter_context(mock.patch.object(measurement.inputs, "_read_loss_correction",
                                                               return_value=evidence))
            stack.enter_context(mock.patch.object(measurement.driver.interface, "content_block_provider",
                                                  return_value=provider))
            step_call = stack.enter_context(mock.patch.object(measurement.driver.training,
                "content_loss_step", side_effect=step))
            artifact = stack.enter_context(mock.patch.object(measurement.driver.final,
                "prepare_final_content_artifact"))
            publisher = stack.enter_context(mock.patch.object(measurement.driver.final,
                "write_final_content_artifact"))
            yield SimpleNamespace(base=base, context=context, resolver=resolver, requests=requests,
                bound=bound, prepare=prepare, steps=steps, step=step, step_call=step_call,
                artifact=artifact, publisher=publisher, correction=correction)

    def run_measurement(self, fixture, **changes):
        options = dict(require_admitted_context=fixture.resolver, prepare_inputs=fixture.prepare,
            budget=fixture.base.budget, reserved_final_path=fixture.base.path)
        options.update(changes)
        return measurement.measure_content_pair(fixture.context, **options)

    def test_real_driver_and_loop_stop_after_four_updates_without_final_artifact_or_loss(self):
        with self.fixture() as fixture:
            result = self.run_measurement(fixture)
            self.assertEqual(result["status"], "complete_measurement")
            self.assertEqual(result["completed_updates"], 4)
            self.assertEqual(fixture.step_call.call_count, 4)
            self.assertEqual(fixture.steps, [fixture.base.panel.channel_schedule.blocks(index)
                                            for index in range(4)])
            self.assertEqual(result["production_iterations"], 1024)
            self.assertFalse(result["production_fit_completed"])
            self.assertFalse(result["model_published"])
            self.assertFalse(result["resumable"])
            fixture.prepare.assert_called_once_with()
            fixture.correction.assert_called_once_with()
            fixture.artifact.assert_not_called()
            fixture.publisher.assert_not_called()
            self.assertFalse(fixture.base.path.exists())
            self.assertFalse(fixture.base.path.with_name(fixture.base.path.name + ".partial").exists())

            self.assertEqual(len(fixture.requests), 2)
            options = fixture.requests[0]["execution_options"]
            self.assertEqual((options["block_units"], options["row_chunk_size"], options["updates"]), (256, 512, 4))
            with self.assertRaises(TypeError):
                options["updates"] = 1024
            self.assertEqual([row["elapsed_seconds"] for row in result["updates"]], [1.] * 4)
            self.assertEqual([row["includes_pair_setup_and_first_Adam_allocation"]
                              for row in result["updates"]], [True, False, False, False])
            self.assertEqual([row["logical_feature_bytes"] for row in result["updates"]], [5695714816] * 4)
            self.assertEqual(result["completed_update_logical_feature_bytes"], 22782859264)
            self.assertIsNone(result["partial_update_logical_feature_bytes"])
            self.assertIsNone(result["error_type"])
            self.assertEqual(result["actual_versions"]["torch"], str(torch.__version__))
            text = json.dumps(result, allow_nan=False)
            self.assertNotIn("training_loss", text)
            self.assertNotIn("987654", text)
            self.assertNotIn("state_dict", text)

    def test_initial_admission_and_reserved_path_refuse_before_prepare_and_revocation_before_driver(self):
        for defect in ("denied", "boolean", "occupied", "revoked", "wrong_context"):
            with self.subTest(defect=defect), self.fixture() as fixture:
                changes = {}
                if defect == "denied":
                    fixture.context.allowed = False
                elif defect == "boolean":
                    changes["require_admitted_context"] = lambda *args, **kwargs: True
                elif defect == "occupied":
                    fixture.base.path.write_bytes(b"preserved")
                elif defect == "revoked":
                    def prepare():
                        fixture.context.allowed = False
                        return fixture.bound
                    fixture.prepare.side_effect = prepare
                else:
                    fixture.prepare.return_value = replace(fixture.bound, execution_context=object())
                if defect in ("denied", "boolean", "occupied"):
                    error = {"denied": PermissionError, "boolean": ValueError, "occupied": FileExistsError}[defect]
                    with self.assertRaises(error):
                        self.run_measurement(fixture, **changes)
                    fixture.prepare.assert_not_called()
                else:
                    result = self.run_measurement(fixture)
                    self.assertEqual(result["status"], "incomplete")
                    self.assertEqual(result["completed_updates"], 0)
                    self.assertEqual(result["error_type"], "PermissionError" if defect == "revoked" else "ValueError")
                fixture.step_call.assert_not_called()
                fixture.artifact.assert_not_called()
                fixture.publisher.assert_not_called()
                if defect == "occupied":
                    self.assertEqual(fixture.base.path.read_bytes(), b"preserved")

    def test_interrupted_update_reports_only_completed_operational_prefix_without_retry(self):
        with self.fixture() as fixture:
            attempts = []
            def interrupted(*args, **kwargs):
                attempts.append(True)
                if len(attempts) == 3:
                    raise TimeoutError("DO_NOT_REPORT_SCIENTIFIC_VALUES_987654")
                return fixture.step(*args, **kwargs)
            fixture.step_call.side_effect = interrupted
            result = self.run_measurement(fixture)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["error_type"], "TimeoutError")
            self.assertEqual(result["completed_updates"], 2)
            self.assertEqual(result["completed_update_logical_feature_bytes"], 2 * 5695714816)
            self.assertEqual(len(attempts), 3)
            self.assertIsNone(result["partial_update_logical_feature_bytes"])
            self.assertNotIn("DO_NOT_REPORT", json.dumps(result))
            self.assertNotIn("987654", json.dumps(result))
            fixture.prepare.assert_called_once_with()
            fixture.artifact.assert_not_called()
            fixture.publisher.assert_not_called()
            self.assertFalse(fixture.base.path.exists())
            self.assertFalse(fixture.base.path.with_name(fixture.base.path.name + ".partial").exists())

        # A limit observed in the callback follows a committed Adam/LR update.
        # It remains incomplete operationally, but must retain that update count.
        with self.fixture() as fixture:
            previous_check = fixture.base.budget.check
            def expires_after_update():
                previous_check()
                if fixture.steps:
                    raise TimeoutError("post-update deadline")
            fixture.base.budget.check = expires_after_update
            result = self.run_measurement(fixture)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["error_type"], "TimeoutError")
            self.assertEqual(result["completed_updates"], 1)
            self.assertEqual(result["completed_update_logical_feature_bytes"], 5695714816)
            self.assertEqual(fixture.step_call.call_count, 1)
            fixture.artifact.assert_not_called()
            fixture.publisher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
