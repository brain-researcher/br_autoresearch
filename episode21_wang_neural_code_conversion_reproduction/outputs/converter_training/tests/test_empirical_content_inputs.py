"""Three source-free input-composition fixtures; coordinator-owned compute only.

Reuse the tested driver's tiny canonical panel and fabricated 4,712-row roles.
The report fixture follows run_converter_loss_correction.main's public-source
report shape; it is not new scientific evidence or an empirical permission.
Actual data/manifest loaders are observed at their I/O boundaries. No real
source responses, features, coefficients, correction artifact or grant is read.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import empirical_content_inputs as inputs
import test_content_pair_driver as driver_fixtures


def correction_fixture():
    # Fields originate in run_converter_loss_correction.py and the public
    # feature_protocol.converter_conformance result, not a fabricated grant.
    loss_prefix = "float32_norms_of_complete_active_blocks_inverse_squared_then_normalize_sum_to_one"
    return {
        "astra_analysis_id": "pipeline_conformance",
        "status": "passed_source_sum_loss_correction", "job_id": "46881096",
        "hostname": "fixture-only", "elapsed_seconds": .25,
        "scope": "public_source_and_small_analytic_fixture_only",
        "historical_report": "outputs/pipeline_conformance/attempt-008/conformance_report.json",
        "historical_lock": "outputs/protocol_lock.yaml",
        "corrects_only": ["converter_conformance.loss",
            "resolved_protocol_table.content_converter_contract.content_loss.exact_weighting_layer_aggregation_and_numerical_stability"],
        "previous_loss_claim": loss_prefix + "_times_per_block_mean_MSE_sum_over_layers",
        "converter_conformance": {
            "passed": True, "fixture_only": True,
            "hidden_width": "source_input_width//2_for_both_hidden_layers",
            "normalization": {"kind": "InstanceNorm1d", "num_features": 1, "eps": 1e-5,
                "affine": False, "track_running_stats": False,
                "axes": "per_sample_across_hidden_positions_of_N_1_hidden_tensor_biased_variance"},
            "activation": "LeakyReLU_negative_slope_0.2",
            "loss": loss_prefix + "_times_per_block_sum_squared_error_sum_over_layers",
            "loss_reduction": "sum",
            "loss_fixture": {"caller_constructor_and_argument_verified": True,
                "expected_sum_loss": 48 / 7, "observed_loss": 48 / 7,
                "counterfactual_mean_loss": 16 / 7,
                "expected_gradient_per_element": {"conv": 8 / 7, "fc": 12 / 7},
                "all_prediction_gradients_verified": True},
            "feature_normalization": "source_training_image_features_use_target_decoder_training_mean_and_ddof1_std_per_unit",
            "degenerate_active_norm": "stop_affected_fit_no_added_epsilon_or_block_exclusion",
            "runtime": {"numpy": "fixture-numpy", "torch": "fixture-torch"},
            "torch_threads": 1, "real_images_or_neural_values_read": False,
            "OOD_access": False, "empirical_fitting_scoring": False,
        },
        "decision_basis": "outputs/protocol_decisions_2026-09-30.yaml:adopted_tagged_choices.content_converter",
        "historical_artifacts_preserved": True, "decoder_fits_and_VGG_features_affected": False,
        "empirical_fitting_scoring_authorized": False, "neural_or_image_values_read": False,
        "OOD_access": False, "parent_charged_once_to": "EP21",
        "terminal_accounting_by_coordinator": True,
    }


class EmpiricalContentInputsTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with driver_fixtures.ContentPairDriverTests().fixture() as base, ExitStack() as stack:
            report_path = base.path.with_name("fixture_loss_correction.json")
            report_path.write_text(json.dumps(correction_fixture()))
            context = SimpleNamespace(operations={"bind_content_inputs"})
            events, requests = [], []

            def resolver(supplied, **request):
                self.assertIs(supplied, context)
                self.assertEqual((request["source"], request["target"]),
                                 (base.source.axis.participant, base.target))
                self.assertIs(request["budget"], base.budget)
                requests.append(request)
                events.append("admit:" + request["operation"])
                if request["operation"] not in context.operations:
                    raise PermissionError("fixture context does not admit operation")

            real_correction = inputs._read_loss_correction
            def correction():
                events.append("correction")
                return real_correction()

            def manifest(*args, **kwargs):
                events.append("manifest")
                return base.panel.manifest

            def prepare(*args, **kwargs):
                events.append("prepare")
                return base.panel

            def source(*args, **kwargs):
                events.append("source")
                return base.source

            stack.enter_context(mock.patch.object(inputs, "CORRECTION_PATH", report_path))
            correction_call = stack.enter_context(mock.patch.object(inputs, "_read_loss_correction", side_effect=correction))
            manifest_call = stack.enter_context(mock.patch.object(inputs.interface.assembly, "load_complete_manifest", side_effect=manifest))
            prepare_call = stack.enter_context(mock.patch.object(inputs.driver, "prepare_content_panel", side_effect=prepare))
            source_call = stack.enter_context(mock.patch.object(inputs.interface, "load_source_training", side_effect=source))
            yield SimpleNamespace(base=base, context=context, resolver=resolver, events=events,
                requests=requests, report_path=report_path, correction=correction_call,
                manifest=manifest_call, prepare=prepare_call, source=source_call)

    def bind(self, fixture, **changes):
        options = dict(require_admitted_context=fixture.resolver,
            source=fixture.base.source.axis.participant, target=fixture.base.target,
            budget=fixture.base.budget)
        options.update(changes)
        return inputs.bind_content_inputs(fixture.context, **options)

    def test_exact_existing_reader_composition_and_admission_before_payloads(self):
        with self.fixture() as fixture:
            bound = self.bind(fixture)
            self.assertEqual(fixture.events, ["admit:bind_content_inputs", "correction",
                "manifest", "prepare", "admit:bind_content_inputs", "source", "admit:bind_content_inputs"])
            fixture.manifest.assert_called_once_with(
                inputs.interface.assembly.BASE / inputs.interface.assembly.MANIFEST,
                budget=fixture.base.budget)
            fixture.prepare.assert_called_once_with(fixture.base.panel.manifest, budget=fixture.base.budget)
            fixture.source.assert_called_once_with(fixture.base.panel.manifest,
                participant=fixture.base.source.axis.participant,
                expected_image_ids=fixture.base.source.image_ids, budget=fixture.base.budget)
            self.assertIs(bound.panel, fixture.base.panel)
            self.assertIs(bound.source, fixture.base.source)
            self.assertEqual(bound.target, fixture.base.target)
            self.assertEqual(bound.source.normalized_X.shape[0], 4712)
            self.assertEqual(len(bound.panel.axes), 5)
            self.assertEqual(len(bound.panel.initializations), 20)
            self.assertEqual(len(bound.panel.channel_schedule.channels_by_layer), 19)
            self.assertFalse(set(bound.source.image_ids).intersection(
                bound.pair.target_normalization.training_image_ids.tolist()))
            self.assertEqual(bound.loss_correction.torch_version, "fixture-torch")
            self.assertTrue(all(request["execution_options"] is None for request in fixture.requests))

    def test_missing_authority_stale_loss_or_role_conflict_refuses_before_source_read(self):
        for defect in ("missing_resolver", "decoder_boolean", "denied", "mean_loss",
                       "unchecked_gradient", "missing_report", "overlap", "revoked_after_prepare"):
            with self.subTest(defect=defect), self.fixture() as fixture:
                changes = {}
                if defect == "missing_resolver":
                    changes["require_admitted_context"] = None
                elif defect == "decoder_boolean":
                    changes["require_admitted_context"] = lambda *args, **kwargs: True
                elif defect == "denied":
                    fixture.context.operations.clear()
                elif defect == "missing_report":
                    fixture.report_path.unlink()
                elif defect in ("mean_loss", "unchecked_gradient"):
                    record = correction_fixture()
                    if defect == "mean_loss":
                        record["converter_conformance"]["loss"] = record["previous_loss_claim"]
                        record["converter_conformance"]["loss_reduction"] = "mean"
                    else:
                        record["converter_conformance"]["loss_fixture"]["all_prediction_gradients_verified"] = False
                    fixture.report_path.write_text(json.dumps(record))
                elif defect == "overlap":
                    ids = list(fixture.base.panel.training_image_ids)
                    ids[inputs.schedule.PARTICIPANTS.index(fixture.base.target)] = fixture.base.source.image_ids
                    fixture.prepare.side_effect = None
                    fixture.prepare.return_value = replace(fixture.base.panel, training_image_ids=tuple(ids))
                else:
                    def prepare(*args, **kwargs):
                        fixture.context.operations.clear()
                        return fixture.base.panel
                    fixture.prepare.side_effect = prepare
                expected = (PermissionError if defect in ("denied", "revoked_after_prepare")
                            else FileNotFoundError if defect == "missing_report" else ValueError)
                with self.assertRaises(expected):
                    self.bind(fixture, **changes)
                fixture.source.assert_not_called()
                if defect not in ("overlap", "revoked_after_prepare"):
                    fixture.manifest.assert_not_called()
                    fixture.prepare.assert_not_called()

    def test_measurement_cannot_publish_and_full_adapter_preserves_driver_limits(self):
        with self.fixture() as fixture:
            bound = self.bind(fixture)
            options = dict(device="cpu", block_units=7, row_chunk_size=37, backend="sparse",
                           final_path=fixture.base.path, maximum_bytes=2_000_000)
            with mock.patch.object(inputs.driver, "train_and_publish_content_pair") as publish:
                with self.assertRaises(PermissionError):
                    inputs.train_and_publish_full_pair(bound, **options)
                publish.assert_not_called()
                fixture.context.operations.add("train_and_publish_full_pair")
                result = inputs.train_and_publish_full_pair(bound, **options)
                self.assertIs(result, publish.return_value)
                publish.assert_called_once_with(bound.panel, bound.source, target=bound.target,
                    budget=bound.budget, on_iteration=None, **options)
                request = fixture.requests[-1]
                self.assertEqual(request["operation"], "train_and_publish_full_pair")
                self.assertEqual(dict(request["execution_options"]), options)
                with self.assertRaises(TypeError):
                    request["execution_options"]["maximum_bytes"] = 1
                publish.reset_mock()
                # Existing evidence cannot disappear after binding and still
                # allow the full driver to start; no retry or success fallback.
                fixture.report_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    inputs.train_and_publish_full_pair(bound, **options)
                publish.assert_not_called()
            self.assertFalse(fixture.base.path.exists())


if __name__ == "__main__":
    unittest.main()
