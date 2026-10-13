"""Focused participant-state fixtures; prepare only, run inside future Slurm work."""
import csv
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import participant_training_state as state
import prepare_decoder_states as parent


class ParticipantStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state.prefix.load_libraries()
        cls.np = state.prefix.np

    def authority(self):
        resource = {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 120,
                    "durable_bytes": 4_000_000_000, "scratch_bytes": 2_000_000_000}
        return dict(state.FULL_AUTH_EXPECTED, reuse_completed_units=51328,
                    state_build_resources=dict(resource), assembly_resources=dict(resource),
                    resources={"worker_jobs": 75, "state_build_jobs": 1, "assembly_jobs": 1,
                               "maximum_concurrent_jobs": 8, "maximum_allocated_CPU_seconds": 2_200_000,
                               "durable_bytes": 500_000_000_000, "peak_transient_bytes": 16_000_000_000})

    def test_full_authority_cannot_be_substituted_by_consumed_or_layer_approval(self):
        with mock.patch.object(state.prefix, "read_json") as read:
            for path in (state.prefix.AUTH, state.retained.AUTH,
                         state.BASE / "outputs/decoder_training_layer_authorization.json"):
                with self.assertRaisesRegex(state.Stop, "separate_fixed_full_stage"):
                    state.authorize_full(path)
            read.assert_not_called()
        auth = self.authority()
        with mock.patch.object(state.prefix, "read_json", return_value=auth):
            self.assertIs(state.authorize_full(state.FULL_AUTH), auth)
        for changed in ({"approved": False}, {"approved": 1}, {"participants": ["sub-01"]},
                        {"allow_tau_train_values": True}, {"allow_original_neural_values_for_state_build": ["sub-01"]}):
            with mock.patch.object(state.prefix, "read_json", return_value=dict(auth, **changed)):
                with self.assertRaisesRegex(state.Stop, "explicit_full_stage"):
                    state.authorize_full(state.FULL_AUTH)
        with mock.patch.object(state, "authorize_full", side_effect=state.Stop("unapproved")), \
                mock.patch.object(state, "admit_participant") as metadata, \
                mock.patch.object(state.prefix, "load_libraries") as libraries:
            for call in (state.prepare_state, state.load_shared):
                with self.assertRaisesRegex(state.Stop, "unapproved"):
                    call("sub-03", state.FULL_AUTH, mock.Mock())
            metadata.assert_not_called()
            libraries.assert_not_called()

    def test_one_shared_state_parent_not_four_jobs_and_no_sub01_rebuild(self):
        auth = self.authority()
        auth["resources"]["state_build_jobs"] = 4
        with mock.patch.object(state.prefix, "read_json", return_value=auth):
            with self.assertRaisesRegex(state.Stop, "finite_aggregate"):
                state.authorize_full(state.FULL_AUTH)
        auth = self.authority()
        auth["state_build_resources"]["minutes"] = True
        with mock.patch.object(state.prefix, "read_json", return_value=auth):
            with self.assertRaisesRegex(state.Stop, "explicit_finite"):
                state.authorize_full(state.FULL_AUTH)
        with mock.patch.object(state, "authorize_full", return_value=self.authority()), \
                mock.patch.object(state, "require_compute") as compute, \
                mock.patch.object(state, "admit_participant") as metadata:
            with self.assertRaisesRegex(state.Stop, "sub01_state_is_retained"):
                state.prepare_state("sub-01", state.FULL_AUTH, mock.Mock())
            compute.assert_not_called()
            metadata.assert_not_called()
        self.assertEqual(state.state_directory("sub-01"), state.prefix.OUTPUT)
        self.assertEqual(state.state_directory("sub-07"), state.FULL_ROOT / "shared/sub-07")

    def test_participant_repetition_join_keeps_stored_order_and_rejects_contamination(self):
        rows = [{"output_row_index": str(index), "participant": "sub-05", "role": "subject_unique",
                 "image_id": image, "session": "session", "beta_volume_index": str(index)}
                for index, image in enumerate(("b", "a", "c", "b", "a"))]
        self.assertEqual(state.group_trials(rows, ["a", "b", "c"], "sub-05", 5), [[1, 4], [0, 3], [2]])
        for field, value, error in (("participant", "sub-03", "participant_trial"),
                                     ("role", "tau_train", "participant_trial"),
                                     ("image_id", "unregistered", "participant_trial"),
                                     ("beta_volume_index", "0", "duplicate_native")):
            bad = [dict(row) for row in rows]
            bad[1][field] = value
            with self.assertRaisesRegex(state.Stop, error):
                state.group_trials(bad, ["a", "b", "c"], "sub-05", 5)
        with self.assertRaisesRegex(state.Stop, "all_training_images"):
            state.group_trials(rows, ["a", "b", "c", "missing"], "sub-05", 5)
        with self.assertRaisesRegex(state.Stop, "all_training_images"):
            state.group_trials(rows, ["a", "b", "c"], "sub-05", 6)

    def test_feature_index_requires_own_complete_4712_image_binding(self):
        selected = [(10000 + index, {"image_id": "image-" + str(index)}) for index in range(4712)]
        fields = ["output_row_index", "regular_HDF5_row", "image_id", "participant", "role"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.tsv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(fields)
                writer.writerows((index, ordinal, row["image_id"], "sub-06", "subject_unique")
                                 for index, (ordinal, row) in enumerate(selected))
            self.assertEqual(state.feature_index(path, "sub-06", selected), [row["image_id"] for _, row in selected])
            with self.assertRaisesRegex(state.Stop, "participant_feature"):
                state.feature_index(path, "sub-03", selected)
            with self.assertRaisesRegex(state.Stop, "all_4712"):
                state.feature_index(path, "sub-06", selected[:-1])

    def arrays(self):
        np = self.np
        values = np.arange(5, dtype=np.float32)[:, None] + np.arange(502, dtype=np.float32)[None, :]
        values[:, 500:] = 10
        flags = np.packbits(np.ones((5, 502), dtype=np.uint8), axis=1, bitorder="little")
        native = np.arange(1000, 1502, dtype=np.int64)
        groups = [[0, 3], [1], [2, 4]]
        return values, flags, native, groups

    def test_state_math_reuses_source_helpers_and_preserves_500_case_native_ids(self):
        np = self.np
        values, flags, native, groups = self.arrays()
        with mock.patch.object(state.prefix, "aggregate_images", wraps=state.prefix.aggregate_images) as aggregate, \
                mock.patch.object(state.prefix, "normalize_neural", wraps=state.prefix.normalize_neural) as normalize, \
                mock.patch.object(state.prefix, "source_gram", wraps=state.prefix.source_gram) as gram:
            result = state.build_arrays(values, flags, native, groups, lambda: None)
        self.assertIs(aggregate.call_args.args[0], values)
        self.assertIs(aggregate.call_args.args[1], flags)
        self.assertIs(aggregate.call_args.args[2], groups)
        aggregate.assert_called_once()
        normalize.assert_called_once()
        gram.assert_called_once()
        np.testing.assert_array_equal(result["image_repetition_counts"], [2, 1, 2])
        np.testing.assert_array_equal(result["eligible_native_voxel_ids"], native[:500])
        np.testing.assert_array_equal(result["eligible_mask"], [True] * 500 + [False, False])
        self.assertEqual(result["normalized_X"].shape, (3, 500))
        self.assertEqual(result["penalized_Gram"].shape, (501, 501))
        self.assertEqual(result["normalized_X"].dtype, np.dtype("float32"))
        self.assertEqual(result["penalized_Gram"].dtype, np.dtype("float32"))
        self.assertEqual(float(result["penalized_Gram"][-1, -1]), 103.)
        # These are the selected participant's original native IDs, not rank IDs.
        self.assertEqual(int(result["eligible_native_voxel_ids"][0]), 1000)

    def test_npz_header_admission_bounds_all_arrays_before_loading(self):
        np = self.np
        values, flags, native, groups = self.arrays()
        arrays = state.build_arrays(values, flags, native, groups, lambda: None)
        arrays["training_image_ids"] = np.asarray(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binding = state.StateBinding("sub-03", directory, directory, ("a", "b", "c"),
                                         (2, 1, 2), 5, 502, 500)
            np.savez(binding.state_path, **arrays)
            state._array_headers(binding)
            with self.assertRaisesRegex(state.Stop, "state_array_shape"):
                state._array_headers(replace(binding, eligible_count=501))
            altered = dict(arrays, normalized_X=arrays["normalized_X"].astype(np.float64))
            np.savez(binding.state_path, **altered)
            with self.assertRaisesRegex(state.Stop, "state_source_float32"):
                state._array_headers(binding)
            np.savez(binding.state_path, **dict(arrays, unauthorized=np.ones(1)))
            with self.assertRaisesRegex(state.Stop, "exact_state_arrays"):
                state._array_headers(binding)

    def test_loader_uses_completed_own_state_without_reaggregation_or_gram_recompute(self):
        np = self.np
        values, flags, native, groups = self.arrays()
        arrays = state.build_arrays(values, flags, native, groups, lambda: None)
        arrays["training_image_ids"] = np.asarray(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binding = state.StateBinding("sub-03", directory, directory, ("a", "b", "c"),
                                         (2, 1, 2), 5, 502, 500)
            np.savez(binding.state_path, **arrays)
            with mock.patch.object(state, "authorize_full", return_value=self.authority()), \
                    mock.patch.object(state, "require_compute"), \
                    mock.patch.object(state, "admit_participant", return_value={"lock": {}}), \
                    mock.patch.object(state, "_admit_state_binding", return_value=binding), \
                    mock.patch.object(state, "_runtime"), \
                    mock.patch.object(state, "prepare_state") as build, \
                    mock.patch.object(state.prefix, "source_gram") as recompute:
                x, augmented, gram, eligible, actual = state.load_shared("sub-03", state.FULL_AUTH, mock.Mock())
                build.assert_not_called()
                recompute.assert_not_called()
                np.testing.assert_array_equal(x, arrays["normalized_X"])
                np.testing.assert_array_equal(gram, arrays["penalized_Gram"])
                np.testing.assert_array_equal(eligible, native[:500])
                np.testing.assert_array_equal(augmented[:, -1], np.ones(3, dtype=np.float32))
                self.assertIs(actual, binding)

    def run_parent_fixture(self, temporary, fail_participant=None):
        root = Path(temporary)
        output, manifest = root / "state_parent", root / "work_manifest.json"
        calls = []
        auth = self.authority()
        allocation = {"job_id": "123", "hostname": "compute", "allocated_cpus": 1,
                      "memory_gib": 7, "gpus": 0, "time_limit_seconds": 7200,
                      "job_end_epoch": time.time() + 7200}
        def prepare(participant, authorization, budget):
            calls.append((participant, authorization, budget))
            if participant == fail_participant:
                raise state.Stop("fixture_state_failure")
            return state.StateBinding(participant, root / "shared" / participant, root / "features" / participant,
                                      tuple("image-" + str(i) for i in range(4712)), (4,) * 4712,
                                      state.SUPPORT[participant][0], state.SUPPORT[participant][1], 500)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(parent, "OUTPUT", output))
            stack.enter_context(mock.patch.object(parent.contract, "MANIFEST", manifest))
            stack.enter_context(mock.patch.object(Path, "cwd", return_value=state.BASE))
            stack.enter_context(mock.patch.object(state, "authorize_full", return_value=auth))
            stack.enter_context(mock.patch.object(state.prefix, "read_json", return_value={"fixture": True}))
            validate = stack.enter_context(mock.patch.object(parent.contract, "validate_work_manifest", return_value={}))
            verify = stack.enter_context(mock.patch.object(parent.contract, "verify_allocation", return_value=allocation))
            stack.enter_context(mock.patch.object(state, "state_directory", side_effect=lambda p: root / "shared" / p))
            stack.enter_context(mock.patch.object(state, "prepare_state", side_effect=prepare))
            result = parent.run(SimpleNamespace(authorization=state.FULL_AUTH, manifest=manifest, max_seconds=600))
            validate.assert_called_once_with({"fixture": True}, auth)
            verify.assert_called_once_with(auth["state_build_resources"])
        return result, calls, json.loads((output / "state_parent_status.json").read_text())

    def test_state_parent_shares_one_budget_and_scheduler_admission_across_four_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, record = self.run_parent_fixture(temporary)
        self.assertEqual(result, 0)
        self.assertEqual([row[0] for row in calls], list(state.NEW_PARTICIPANTS))
        self.assertTrue(all(row[2] is calls[0][2] for row in calls))
        self.assertEqual(record["status"], "complete_four_participant_training_states_only")
        self.assertEqual(record["participants_completed"], list(state.NEW_PARTICIPANTS))
        self.assertEqual(record["participants_remaining"], [])
        self.assertFalse(record["sub01_state_recomputed"])
        self.assertEqual(record["accounting"]["state_build_jobs_consumed"], 1)

    def test_state_parent_stops_after_first_failure_and_retains_completed_state_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, calls, record = self.run_parent_fixture(temporary, "sub-05")
        self.assertEqual(result, 2)
        self.assertEqual([row[0] for row in calls], ["sub-03", "sub-05"])
        self.assertEqual(record["participants_completed"], ["sub-03"])
        self.assertEqual(record["participants_remaining"], ["sub-05", "sub-06", "sub-07"])
        self.assertEqual(record["active_participant"], "sub-05")
        self.assertEqual(record["error"], "fixture_state_failure")
        self.assertFalse(record["automatic_retry"])

    def test_state_parent_rejects_unapproved_stage_before_manifest_or_allocation(self):
        with mock.patch.object(Path, "cwd", return_value=state.BASE), \
                mock.patch.object(state, "authorize_full", side_effect=state.Stop("not_approved")), \
                mock.patch.object(parent.contract, "validate_work_manifest") as manifest, \
                mock.patch.object(parent.contract, "verify_allocation") as allocation, \
                mock.patch.object(state, "prepare_state") as build:
            with self.assertRaisesRegex(state.Stop, "not_approved"):
                parent.run(SimpleNamespace(authorization=state.FULL_AUTH, manifest=parent.contract.MANIFEST, max_seconds=None))
            manifest.assert_not_called()
            allocation.assert_not_called()
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
