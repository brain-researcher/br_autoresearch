"""Three focused runner checks; coordinated compute only, no real payloads.

Use temporary admission records, invented tiny typed inputs and the existing
completed-fit fixture. Only fitting is mocked in the successful driver path;
the distinct artifact preparation, publication and reload are exercised.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import brain_loss as original
import brain_loss_constant as brain
import empirical_brain_inputs as inputs
import final_brain_constant_artifact as final
import run_brain_constant_pair as runner
import test_final_brain_artifact as old_fixtures


EXPECTED_WIDTHS = {"sub-01": 12295, "sub-03": 12897, "sub-05": 10891,
                   "sub-06": 12023, "sub-07": 12057}
EXPECTED_RESOURCES = dict(cpus=3, memory_gib=16, minutes=360, gpus=0,
    deadline_seconds=21000, rss_limit_bytes=16106127360,
    durable_bytes=2147483648, scratch_bytes=268435456, final_artifact_bytes=805306368)


def grant():
    return dict(schema_version="ep21.brain_constant_panel.v1", authorized=True,
        state="separately_regenerated_adapted_training_lock",
        normalization_policy=brain.NORMALIZATION_POLICY, inherited_lock="outputs/protocol_lock.yaml",
        operation="train_and_publish_brain_constant_pair", training_images=897, cv_fits=50, final_fits=1,
        original_pair_order=[list(pair) for pair in original.PAIR_ORDER], native_widths=EXPECTED_WIDTHS,
        resources=EXPECTED_RESOURCES, fresh_amended_fits=20, original_artifact_reuse=False,
        production_training=True, model_publication=True, scoring=False,
        tau_test_access=False, OOD_access=False, automatic_retry=False,
        approval_source="invented prospective fixture authority")


@contextmanager
def admission_files():
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        base = Path(temporary).resolve()
        stack.enter_context(mock.patch.object(runner, "BASE", base))
        stack.enter_context(mock.patch.object(runner, "PROFILE", base / "grant.json"))
        stack.enter_context(mock.patch.object(runner, "GUARD", base / "guard.json"))
        runner.PROFILE.write_text(json.dumps(grant()))
        runner.GUARD.write_text(json.dumps(dict(kind="ep21_brain_constant_discriminator",
            status="completed", guard_satisfied=True, first_fold_only=True,
            fits_performed=0, tau_test_accessed=False)))
        yield base


class BrainConstantRunnerTests(unittest.TestCase):
    def fixture(self, resolver=None):
        helper = old_fixtures.FinalBrainArtifactTests()
        old, budget = helper.fixture(), helper.budget()
        def state(axis, count):
            mean, sd = getattr(old, axis + "_mean"), getattr(old, axis + "_std")
            return brain.NormalizationState(mean, sd, sd.copy(), np.zeros(sd.shape, dtype=bool), count)
        completed = brain.ConstantBrainLossFit(old.summary, old.image_ids,
            old.source_native_voxel_ids, old.target_native_voxel_ids, old.matrix,
            tuple(state("source", 897 - count) for count in original.FOLD_SIZES),
            tuple(state("target", 897 - count) for count in original.FOLD_SIZES),
            state("source", 897), state("target", 897))
        responses = []
        for person, axis in ((old.summary.source, old.source_native_voxel_ids),
                             (old.summary.target, old.target_native_voxel_ids)):
            values = np.zeros((897, len(axis)), dtype=np.float32)
            values.setflags(write=False)
            responses.append(original.TauTrainResponses(person, old.image_ids, axis, values))
        identity = budget.allocation["job_id"], budget.allocation["hostname"]
        bound = inputs.BoundBrainInputs(object(), resolver or (lambda *args, **kw: None), budget,
            identity, original.make_common_folds(old.image_ids), *responses, (3,) * 897, (4,) * 897)
        return bound, completed

    def test_full_panel_admission_binds_resources_clocks_and_refuses_before_reader(self):
        with admission_files() as base:
            stages = tuple(runner.stage(index) for index in range(20))
            self.assertEqual(tuple((s.source, s.target) for s in stages), original.PAIR_ORDER)
            self.assertEqual(len({s.final_path for s in stages}), 20)
            for index, selected in enumerate(stages):
                self.assertEqual(selected.pair_index, index)
                self.assertEqual(selected.source_native_voxels, EXPECTED_WIDTHS[selected.source])
                self.assertEqual(selected.target_native_voxels, EXPECTED_WIDTHS[selected.target])
                self.assertEqual(selected.resources, EXPECTED_RESOURCES)
                self.assertEqual(selected.final_path.parent, selected.run_dir)
                self.assertEqual(selected.final_path.name, "final_brain_constant_pair.pt")
                self.assertEqual(selected.run_dir.parent, base / "outputs/runs/brain_constant_training")
            for invalid in (-1, 20, True):
                with self.assertRaises(runner.original.AdmissionError):
                    runner.stage(invalid)
            inactive = grant()
            inactive["authorized"] = False
            runner.PROFILE.write_text(json.dumps(inactive))
            with mock.patch.object(runner, "allocation") as allocation, \
                    mock.patch.object(inputs, "bind_brain_inputs") as reader:
                with self.assertRaises(runner.original.AdmissionError):
                    runner.run_once(4)
                allocation.assert_not_called()
                reader.assert_not_called()
            runner.PROFILE.write_text(json.dumps(grant()))
            selected = stages[4]
            selected.run_dir.mkdir(parents=True)
            scratch = base / "scratch" / selected.scratch_name
            scratch.mkdir(parents=True)
            now = 1800000000
            environment = {**{key: "1" for key in runner.original.THREADS},
                "SLURM_JOB_ID": "12345678", "SLURM_CPUS_PER_TASK": "3", "SLURM_MEM_PER_NODE": "16384",
                "SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_CPUS_PER_NODE": "3", "SLURM_NTASKS": "1",
                "SLURM_JOB_PARTITION": "russpold", "SLURM_JOB_START_TIME": str(now - 100),
                "SLURM_JOB_END_TIME": str(now + 21500), "CUDA_VISIBLE_DEVICES": "",
                "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": "", "SLURM_GPUS": "0",
                "SLURM_GPUS_ON_NODE": "0", "L_SCRATCH_JOB": str(scratch.parent), "TMPDIR": str(scratch)}
            with mock.patch.dict(os.environ, environment), \
                    mock.patch.object(runner.socket, "gethostname", return_value="sh04-fixture.stanford.edu"), \
                    mock.patch.object(runner.time, "time", return_value=now):
                context, budget = runner.admit(4)
                self.assertEqual(context.allocation["deadline_epoch"], now + 20900)
                self.assertEqual(context.allocation["job_end_epoch"], now + 21500)
                self.assertEqual(context.allocation["cpus"], 3)
                self.assertEqual(context.allocation["memory_gib"], 16)
                for end in (now + 21505, now + 30000):
                    with mock.patch.dict(os.environ, {"SLURM_JOB_END_TIME": str(end)}):
                        slack = runner.allocation()
                    self.assertEqual(slack["deadline_epoch"], now - 100 + 21000)
                with mock.patch.dict(os.environ, {"SLURM_JOB_END_TIME": str(now + 1000)}):
                    shorter = runner.allocation()
                self.assertEqual(shorter["deadline_epoch"], now + 940)
                arguments = dict(source=selected.source, target=selected.target, budget=budget)
                runner.resolver(context, operation="bind_brain_inputs", execution_options=None, **arguments)
                runner.resolver(context, operation=runner.OPERATION, execution_options=dict(
                    final_path=selected.final_path, maximum_bytes=EXPECTED_RESOURCES["final_artifact_bytes"]),
                    **arguments)
                # The real adapter must encounter denial before its metadata or
                # response reader when handed an unauthorized operation resolver.
                def unauthorized(context, **kwargs):
                    return runner.resolver(context, **{**kwargs, "operation": "evaluate_tau_test"})
                with mock.patch.object(inputs, "_metadata") as metadata, \
                        mock.patch.object(inputs, "_read_participant") as reader:
                    with self.assertRaisesRegex(runner.original.AdmissionError, "operation_not_authorized"):
                        inputs.bind_brain_inputs(context, require_admitted_context=unauthorized, **arguments)
                    metadata.assert_not_called()
                    reader.assert_not_called()
                for changes in ({"SLURM_CPUS_PER_TASK": "2"}, {"SLURM_MEM_PER_NODE": "7168"},
                                {"OMP_NUM_THREADS": "2"}, {"CUDA_VISIBLE_DEVICES": "0"},
                                {"SLURM_JOB_START_TIME": str(now + 1)},
                                {"SLURM_JOB_END_TIME": str(now)}):
                    with self.subTest(changes=changes), mock.patch.dict(os.environ, changes):
                        with self.assertRaises(runner.original.AdmissionError):
                            runner.allocation()
                changed = grant()
                changed["tau_test_access"] = True
                runner.PROFILE.write_text(json.dumps(changed))
                with self.assertRaises(runner.original.AdmissionError):
                    runner.resolver(context, operation="bind_brain_inputs", execution_options=None, **arguments)

    def test_typed_driver_preserves_folds_axes_and_distinct_final_publication(self):
        events = []
        def resolver(context, **kwargs):
            self.assertEqual(kwargs["operation"], "train_and_publish_brain_constant_pair")
            self.assertEqual((kwargs["source"], kwargs["target"]), original.PAIR_ORDER[5])
            self.assertEqual(dict(kwargs["execution_options"]), dict(
                final_path=path, maximum_bytes=EXPECTED_RESOURCES["final_artifact_bytes"]))
            events.append("admit")
        bound, completed = self.fixture(resolver)
        def fitted(source, target, folds, **kwargs):
            self.assertIs(source, bound.source)
            self.assertIs(target, bound.target)
            self.assertIs(folds, bound.folds)
            self.assertEqual(kwargs["source_native_ids"], completed.source_native_voxel_ids)
            self.assertEqual(kwargs["target_native_ids"], completed.target_native_voxel_ids)
            self.assertIs(kwargs["check"], bound.budget.check)
            events.append("fit")
            return completed
        actual_prepare, actual_write = final.prepare_final_brain_constant_artifact, final.write_final_brain_constant_artifact
        def prepare(*args, **kwargs):
            self.assertEqual(kwargs["expected_pair"], original.PAIR_ORDER[5])
            self.assertEqual(kwargs["authenticated_tau_train_image_ids"], completed.image_ids)
            self.assertEqual(kwargs["source_native_ids"], completed.source_native_voxel_ids)
            self.assertEqual(kwargs["target_native_ids"], completed.target_native_voxel_ids)
            events.append("prepare")
            return actual_prepare(*args, **kwargs)
        def write(*args, **kwargs):
            self.assertEqual(events[-1], "admit")
            events.append("publish")
            return actual_write(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_brain_constant_pair.pt"
            with mock.patch.object(brain, "fit_brain_pair", side_effect=fitted) as fitting, \
                    mock.patch.object(final, "prepare_final_brain_constant_artifact", side_effect=prepare), \
                    mock.patch.object(final, "write_final_brain_constant_artifact", side_effect=write):
                published, summary = runner.train_and_publish(bound, final_path=path,
                    maximum_bytes=EXPECTED_RESOURCES["final_artifact_bytes"])
            fitting.assert_called_once()
            self.assertEqual(events, ["admit", "fit", "admit", "prepare", "admit", "publish"])
            self.assertEqual((published, summary), (path, completed.summary))
            restored = final.load_final_brain_constant_artifact(path, budget=bound.budget,
                maximum_bytes=EXPECTED_RESOURCES["final_artifact_bytes"])
            self.assertEqual(restored["kind"], "ep21_final_brain_constant_pair")
            self.assertEqual(len(restored["fold_target_normalization"]), 5)
            np.testing.assert_array_equal(restored["matrix"].numpy(), completed.matrix)

    def test_denial_cancellation_and_existing_partial_never_start_or_repeat_fit(self):
        selected = runner.stage(4)
        for code in ("nonfinite_required_training_response", "nonfinite_or_negative_required_training_scale",
                     "zero_sd_nonconstant_required_training_column", "zero_sd_constant_mean_mismatch"):
            for role in ("source", "target"):
                for fold in range(5):
                    details = runner.failure_details(brain.BrainLossNotEvaluable(
                        f"{code}:{role}_fold_{fold}"), selected)
                    self.assertEqual(details, {"brain_loss_failure": dict(reason_code=code,
                        role=role, context="fold_training", fold=fold)})
                self.assertEqual(runner.failure_details(brain.BrainLossNotEvaluable(
                    f"{code}:{role}_final"), selected), {"brain_loss_failure": dict(
                        reason_code=code, role=role, context="final")})
        for exc in (ValueError("zero_sd_constant_mean_mismatch:target_final"),
                    brain.BrainLossNotEvaluable("zero_sd_constant_mean_mismatch:target_final:private-extra"),
                    brain.BrainLossNotEvaluable("unregistered free text")):
            self.assertEqual(runner.failure_details(exc, selected), {})
        for defect in ("denied", "cancelled_before_fit", "allocation", "final", "partial", "byte_cap", "untyped"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "final.pt"
                def resolver(context, **kwargs):
                    if defect == "denied":
                        raise PermissionError("binding only")
                    if defect == "cancelled_before_fit":
                        raise KeyboardInterrupt("cancelled")
                bound, _ = self.fixture(resolver)
                if defect == "allocation":
                    bound = replace(bound, allocation_identity=("other-job", "other-host"))
                if defect in ("final", "partial"):
                    retained = path if defect == "final" else path.with_name("final.pt.partial")
                    retained.write_bytes(b"preserved")
                if defect == "untyped":
                    bound = object()
                expected = {"denied": PermissionError, "cancelled_before_fit": KeyboardInterrupt}.get(
                    defect, runner.original.AdmissionError)
                cap = 1 if defect == "byte_cap" else EXPECTED_RESOURCES["final_artifact_bytes"]
                with mock.patch.object(brain, "fit_brain_pair") as fitting, \
                        mock.patch.object(final, "write_final_brain_constant_artifact") as publication:
                    with self.assertRaises(expected):
                        runner.train_and_publish(bound, final_path=path, maximum_bytes=cap)
                    fitting.assert_not_called()
                    publication.assert_not_called()
                if defect in ("final", "partial"):
                    self.assertEqual(retained.read_bytes(), b"preserved")
        with tempfile.TemporaryDirectory() as temporary:
            bound, _ = self.fixture()
            path = Path(temporary) / "cancelled.pt"
            with mock.patch.object(brain, "fit_brain_pair", side_effect=TimeoutError("deadline")) as fitting, \
                    mock.patch.object(final, "write_final_brain_constant_artifact") as publication:
                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    runner.train_and_publish(bound, final_path=path,
                        maximum_bytes=EXPECTED_RESOURCES["final_artifact_bytes"])
                fitting.assert_called_once()
                publication.assert_not_called()
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("cancelled.pt.partial").exists())


if __name__ == "__main__":
    unittest.main()
