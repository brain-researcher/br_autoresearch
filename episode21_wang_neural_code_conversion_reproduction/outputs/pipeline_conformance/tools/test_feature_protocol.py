"""Focused feature-context regressions; stand-ins never enter episode outputs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_protocol as protocol
import public_runtime_harness as public


class TaggedContentLossCallerTests(unittest.TestCase):
    SOURCE = """\
def converter_training(model, real_A, blocks, inputs, layers, path, indices):
    criterion_mse = torch.nn.MSELoss(reduction='sum')
    for iteration in range(1024):
        loss = chunk_process_batch(model, real_A, blocks, inputs, layers, criterion_mse, path, indices)
    return loss
"""

    def test_pinned_sum_constructor_and_actual_call_argument_are_bound(self):
        self.assertEqual(protocol.tagged_content_loss_reduction(self.SOURCE), "sum")

    def test_mean_default_or_other_constructor_is_rejected(self):
        for replacement in ("torch.nn.MSELoss()", "torch.nn.MSELoss(reduction='mean')",
                            "torch.nn.L1Loss(reduction='sum')",
                            "torch.nn.MSELoss(reduction='sum', size_average=True)"):
            source = self.SOURCE.replace("torch.nn.MSELoss(reduction='sum')", replacement)
            with self.subTest(replacement=replacement), self.assertRaisesRegex(
                    RuntimeError, "tagged_sum_MSE_constructor_required"):
                protocol.tagged_content_loss_reduction(source)

    def test_different_or_inline_default_criterion_at_call_is_rejected(self):
        for replacement in ("other_criterion", "torch.nn.MSELoss()", "criterion_mse, extra"):
            source = self.SOURCE.replace("layers, criterion_mse, path", "layers, " + replacement + ", path")
            with self.subTest(replacement=replacement), self.assertRaisesRegex(
                    RuntimeError, "tagged_loss_caller_criterion_argument_required"):
                protocol.tagged_content_loss_reduction(source)

    def test_criterion_reassignment_mutation_or_missing_caller_is_rejected(self):
        for changed in (
            self.SOURCE.replace("    for iteration", "    criterion_mse = torch.nn.MSELoss()\n    for iteration"),
            self.SOURCE.replace("    for iteration", "    criterion_mse.reduction = 'mean'\n    for iteration"),
            self.SOURCE.replace("converter_training", "unrelated_function"),
        ):
            with self.subTest(source=changed), self.assertRaisesRegex(RuntimeError, "tagged_loss_"):
                protocol.tagged_content_loss_reduction(changed)


class FeatureContextTests(unittest.TestCase):
    def test_loaded_module_versions_are_compared_with_like_provenance(self):
        modules = {"numpy": "1.26.3", "torch": "2.4.0a0"}
        self.assertEqual(protocol.validate_loaded_runtime(
            {"runtime": modules}, {"runtime": modules}), modules)

    def test_genuine_runtime_change_or_missing_evidence_is_rejected(self):
        public_versions = {"numpy": "1.26.3", "torch": "2.4.0a0"}
        for versions in ({"numpy": "1.26.4", "torch": "2.4.0a0"},
                         {"numpy": "1.26.3", "torch": "2.5.0"}, {},
                         {"numpy": "1.26.3"}):
            with self.subTest(versions=versions), self.assertRaisesRegex(
                    RuntimeError, "public_and_feature_runtime_mismatch"):
                protocol.validate_loaded_runtime({"runtime": public_versions},
                                                 {"runtime": versions})

    def setUp(self):
        self.auth = {"schema_version": "ep21.feature_only_execution_authorization.v1",
            "episode_id": "ep21", "approved": True, "purpose": "regular_vgg19_derivative_export",
            "resources": protocol.RESOURCES, "allow_protocol_lock_completion": True,
            "allow_regular_image_values_after_valid_lock": True,
            "includes_public_runtime_conformance": True,
            "neural_execution_authorized": False, "permitted_image_count": 24681,
            "permitted_image_roles": ["subject_unique", "tau_train", "tau_test"],
            **{k: False for k in protocol.CLOSED}, "scientist_response": "TEST_ONLY_STAND_IN"}
        self.report = {"schema_version": "ep21.public_runtime_conformance.v3",
            "outcome": public.OUTCOME, "scope": public.SCOPE, "resources": public.RESOURCES,
            "job_id": "TEST_ONLY_JOB", "strict_load": True, "pre_relu_non_inplace": True,
            "decoder_penalized_bias_structured_prediction": True,
            "historical_Caffe_feature_parity": "unverified", **{k: False for k in public.BOUNDARY_FIELDS}}
        self.alignment = {"TEST_ONLY_ALIGNMENT": True}
        self.lock = {"schema_version": "ep21.protocol_lock.v1", "episode_id": "ep21",
            "state": "valid_prospective_lock", "module_states": {"laion_feature_preparation": "runnable_locked"},
            "neural_execution_authorized": False, "OOD_access_authorized": False,
            "neural_or_image_values_used_to_resolve": False,
            "regular_image_binding": self.alignment, "public_runtime_report_job": "TEST_ONLY_JOB",
            "converter_conformance": {"passed": True},
            "feature_recipe": {"layers": list(protocol.LAYERS), "shapes": [list(s) for s in protocol.SHAPES],
                "dtype": "float32", "checkpoint_members": public.PUBLIC_MEMBERS,
                "input_rule": "RGB_direct_BICUBIC_224_BGR_CHW_float32_0_255_minus_qualified_channel_means"}}

    def test_exact_feature_only_scope_is_accepted_without_neural_authority(self):
        self.assertIs(protocol.validate_feature_lock(self.lock, self.auth, self.report, self.alignment), self.lock)
        self.assertFalse(protocol.FeatureExportContext(Path("TEST_ONLY"), (), Path("TEST_ONLY"), Path("TEST_ONLY")).neural_capable)

    def test_source_only_or_synthetic_authority_cannot_open_features(self):
        for schema in ("ep21.synthetic_adapter_fixture.v1", "public_runtime_conformance"):
            with self.assertRaisesRegex(RuntimeError, "real_feature_only_authorization"):
                protocol.validate_authorization({**self.auth, "schema_version": schema})

    def test_neural_ood_fitting_scoring_reconstruction_authority_rejected(self):
        for k in protocol.CLOSED:
            with self.subTest(k=k), self.assertRaisesRegex(RuntimeError, "feature_only_boundary"):
                protocol.validate_authorization({**self.auth, k: True})

    def test_unapproved_or_changed_parent_resource_stops(self):
        with self.assertRaises(RuntimeError):
            protocol.validate_authorization({**self.auth, "approved": False})
        with self.assertRaisesRegex(RuntimeError, "feature_resource"):
            protocol.validate_authorization({**self.auth, "resources": {**protocol.RESOURCES, "minutes": 481}})

    def test_explicit_tighter_walltime_keeps_all_other_resource_caps(self):
        protocol.validate_authorization({**self.auth, "resources": {**protocol.RESOURCES, "minutes": 479}})
        with self.assertRaisesRegex(RuntimeError, "feature_resource"):
            protocol.validate_authorization({**self.auth, "resources": {**protocol.RESOURCES, "minutes": 479, "gpus": 2}})

    def test_failed_public_or_unfinished_lock_stops(self):
        with self.assertRaisesRegex(RuntimeError, "public_report_not_passed"):
            protocol.validate_feature_lock(self.lock, self.auth, {**self.report, "strict_load": False}, self.alignment)
        with self.assertRaisesRegex(RuntimeError, "valid_real_protocol_lock"):
            protocol.validate_feature_lock({**self.lock, "state": "draft"}, self.auth, self.report, self.alignment)

    def test_changed_alignment_or_public_provenance_stops(self):
        for update in ({"regular_image_binding": {"TEST_ONLY_CHANGED": True}}, {"public_runtime_report_job": "OTHER_TEST_ONLY_JOB"}):
            with self.assertRaises(RuntimeError):
                protocol.validate_feature_lock({**self.lock, **update}, self.auth, self.report, self.alignment)

    def test_missing_converter_conformance_or_closed_feature_module_stops(self):
        for update in ({"converter_conformance": {"passed": False}}, {"module_states": {"laion_feature_preparation": "not_evaluable_locked"}}):
            with self.assertRaises(RuntimeError):
                protocol.validate_feature_lock({**self.lock, **update}, self.auth, self.report, self.alignment)

    def test_different_layers_precision_asset_or_input_rule_stops(self):
        for k, v in (("layers", ["TEST_ONLY_OTHER_LAYER"]), ("dtype", "float16"),
                     ("checkpoint_members", {}), ("input_rule", "TEST_ONLY_OTHER_INPUT")):
            changed = copy.deepcopy(self.lock); changed["feature_recipe"][k] = v
            with self.subTest(k=k), self.assertRaisesRegex(RuntimeError, "frozen_feature_recipe"):
                protocol.validate_feature_lock(changed, self.auth, self.report, self.alignment)

    def test_metadata_publication_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "TEST_ONLY_lock.yaml"
            protocol.publish_json(p, {"TEST_ONLY": True})
            self.assertEqual(json.loads(p.read_text()), {"TEST_ONLY": True})
            self.assertEqual(p.stat().st_mode & 0o777, 0o400)
            with self.assertRaisesRegex(RuntimeError, "do_not_overwrite"):
                protocol.publish_json(p, {"TEST_ONLY": False})


if __name__ == "__main__":
    unittest.main()
