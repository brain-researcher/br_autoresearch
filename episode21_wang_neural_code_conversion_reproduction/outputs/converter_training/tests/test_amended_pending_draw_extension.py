"""Pending-only clock extension admission; invented files and allocations."""
from contextlib import contextmanager
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import test_amended_full_readout_admission as fixture
import run_amended_full_readout as live
import run_amended_pending_draw_extension as extension


def selection():
    return {"schema_version": "ep21.pending_draw_clock_extension.v1",
        "selection_id": "pending_draw_extension001", "authorized": True,
        "approval_source": "invented_fixture_only", "original_array_job_id": extension.ARRAY,
        "batch_indices": list(extension.INDICES), "resources": dict(extension.RESOURCES),
        "held_pending_children": {str(i): {"array_job_id": extension.ARRAY,
            "batch_index": i, "state": "PENDING", "reason": "JobHeldUser", "elapsed_seconds": 0}
            for i in extension.INDICES}}


@contextmanager
def files(index=40):
    with fixture.files("draw", index) as original:
        root = original.stage.root
        selected = root / "pending_draw_extension001.json"
        with mock.patch.object(extension.caller, "BASE", original.root), \
                mock.patch.object(extension, "ROOT", root), \
                mock.patch.object(extension, "SELECTION", selected):
            selected.write_text(json.dumps(selection()))
            baseline = fixture.profile()
            baseline["resources"]["draw"].update(cpus=3, memory_gib=16, minutes=1440,
                deadline_seconds=84600, rss_limit_bytes=15 * 1024**3)
            original.stage.profile_path.write_text(json.dumps(baseline))
            binding = {**original.binding, "draw": {
                **original.binding["draw"], "array_job_id": extension.ARRAY},
                "finalize": {"job_id": "4000", "afterok_job_id": extension.ARRAY}}
            (root / "dependency_bindings.json").write_text(json.dumps(binding))
            resources = {**baseline["resources"]["draw"], "minutes": 2880, "deadline_seconds": 171000}
            stage = extension.caller._stage("draw", index, resources)
            f = SimpleNamespace(stage=stage, root=original.root, scratch=original.scratch,
                binding=binding, original=original, selection=selected, baseline=baseline)
            with fixture.prior.machine(f) as actual, mock.patch.dict(os.environ, {
                    "SLURM_ARRAY_JOB_ID": extension.ARRAY, "SLURM_ARRAY_TASK_COUNT": "63",
                    "SLURM_ARRAY_TASK_MIN": "0", "SLURM_ARRAY_TASK_MAX": "62",
                    "SLURM_ARRAY_TASK_STEP": "1"}):
                actual.allocation({"ArrayJobId": extension.ARRAY})
                yield f, actual


class PendingDrawExtensionTests(unittest.TestCase):
    def test_only_clock_changes_with_original_paths_bindings_and_live_profile_preserved(self):
        caller = extension.caller
        self.assertIsNot(caller, live)
        self.assertIsNot(caller._profile, live._profile)
        for index in (40, 61):
            with self.subTest(index=index), files(index) as (f, actual):
                original_text = f.stage.profile_path.read_text()
                context, budget = caller.admit("draw", index)
                expected = {phase: dict(value) for phase, value in f.baseline["resources"].items()}
                expected["draw"].update(minutes=2880, deadline_seconds=171000)
                self.assertEqual(context.profile["resources"], expected)
                self.assertEqual(budget.remaining_seconds(), 170880)
                original_stage = live._stage("draw", index, f.baseline["resources"]["draw"])
                self.assertEqual(context.stage.run_dir, original_stage.run_dir)
                self.assertEqual(context.stage.stage_h5, original_stage.stage_h5)
                self.assertEqual(context.stage.batch_path, original_stage.batch_path)
                self.assertEqual(caller._options(context.stage), live._options(original_stage))
                self.assertEqual(context.binding, live._bindings(original_stage, context.allocation))
                self.assertEqual(live._profile(original_stage)["resources"], f.baseline["resources"])
                self.assertEqual(f.stage.profile_path.read_text(), original_text)
                actual.allocation({"ArrayJobId": extension.ARRAY,
                    "EndTime": fixture.prior.stamp(fixture.prior.NOW + 600)})
                self.assertEqual(caller.admit("draw", index)[1].remaining_seconds(), 540)

    def test_unselected_scope_changed_evidence_and_wrong_clock_refuse_before_operation(self):
        caller = extension.caller
        with files() as (f, actual), mock.patch.object(caller, "_load_operation") as load:
            for phase, index in (("decode", None), ("finalize", None), ("draw", 39), ("draw", 62)):
                with self.subTest(phase=phase, index=index), self.assertRaises(caller.AdmissionError):
                    caller.run_once(phase, index)
            with mock.patch.dict(os.environ, {"SLURM_ARRAY_JOB_ID": "1234"}):
                with self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 40)
            for changed in ({"authorized": False}, {"batch_indices": list(range(40, 61))},
                    {"original_array_job_id": "1234"}, {"resources": {**extension.RESOURCES, "cpus": 4}}):
                f.selection.write_text(json.dumps({**selection(), **changed}))
                with self.subTest(changed=changed), self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 40)
            for changed in ({"state": "RUNNING"}, {"reason": "Priority"},
                    {"elapsed_seconds": 1}, {"elapsed_seconds": False}):
                chosen = selection()
                chosen["held_pending_children"]["40"].update(changed)
                f.selection.write_text(json.dumps(chosen))
                with self.subTest(changed=changed), self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 40)
            f.selection.write_text(json.dumps(selection()))
            actual.allocation({"ArrayJobId": extension.ARRAY, "TimeLimit": "1-00:00:00"})
            with self.assertRaises(caller.AdmissionError):
                caller.run_once("draw", 40)
            actual.allocation({"ArrayJobId": extension.ARRAY})
            changed = json.loads(f.stage.profile_path.read_text())
            changed["resources"]["draw"]["deadline_seconds"] = 84000
            f.stage.profile_path.write_text(json.dumps(changed))
            with self.assertRaises(caller.AdmissionError):
                caller.run_once("draw", 40)
            load.assert_not_called()

    def test_extended_original_publication_remains_compatible_with_all63_finalization(self):
        caller = extension.caller
        with files() as (f, actual), mock.patch.object(caller, "_configure_file_policy"), \
                mock.patch.object(caller, "_require_file_policy"):
            def operation(context, **kwargs):
                self.assertEqual(kwargs["batch_index"], 40)
                kwargs["require_admitted_context"](context, operation=caller.OPERATIONS["draw"],
                    source=None, target=None, budget=kwargs["budget"],
                    execution_options=caller._options(context.stage))
                context.stage.batch_path.write_text("{}")
                return fixture.PhaseResult("draw", context.stage.publication_paths)

            with mock.patch.object(caller, "_load_operation",
                    return_value=(operation, fixture.PhaseResult, {"fixture": True})):
                self.assertEqual(caller.run_once("draw", 40), 0)
            report = json.loads((f.stage.run_dir / "report.json").read_text())
            self.assertEqual(report["array_job_id"], extension.ARRAY)
            self.assertEqual(report["status"], "complete_draw")
            self.assertTrue(report["readers_closed"])
            claim = json.loads((f.stage.run_dir / "claim.json").read_text())
            self.assertEqual(claim["allocation"]["minutes"], 2880)
            self.assertFalse(claim["automatic_retry"])

        with fixture.files("finalize") as f, fixture.prior.machine(f):
            binding = {**f.binding, "draw": {**f.binding["draw"], "array_job_id": extension.ARRAY},
                "finalize": {"job_id": "4000", "afterok_job_id": extension.ARRAY}}
            (f.stage.root / "dependency_bindings.json").write_text(json.dumps(binding))
            for index in range(63):
                path = f.stage.root / "draws" / f"batch_{index:03d}" / "report.json"
                value = report if index == 40 else json.loads(path.read_text())
                value = {**value, "array_job_id": extension.ARRAY,
                    "allocation_identity": {"job_id": str(3100 + index), "hostname": "sh04-fixture"}}
                path.write_text(json.dumps(value))
            bound = live._bindings(f.stage, {"job_id": "4000"})
            live._completed(f.stage, bound)
            (f.stage.root / "draws/batch_061/batch_status.txt").write_text("phase_exit=124\nstatus=closed\n")
            with self.assertRaises(live.AdmissionError):
                live._completed(f.stage, bound)


if __name__ == "__main__":
    unittest.main()
