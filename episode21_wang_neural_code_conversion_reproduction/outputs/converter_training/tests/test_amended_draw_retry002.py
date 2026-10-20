"""Second retry identity and two-array completion, with invented metadata."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import test_amended_full_readout_admission as fixture
import run_amended_draw_retry as first
from run_amended_draw_retry002 import adapter as retry


def selection():
    return {"authorized": True, "attempt": "attempt002", "batch_indices": list(range(8, 16)),
        "array_job_id": "7000", "original_decode_job_id": "1000",
        "original_array_job_id": "3000", "approval_source": "invented_fixture_only",
        "failed_original_children": {str(i): {"state": "FAILED", "exit_code": "124:0"}
            for i in range(8, 16)}}


@contextmanager
def retry_files(index):
    with fixture.files("draw", index) as original:
        root = original.stage.root
        retry_root = root / "draw_retries/attempt002"
        selected = retry_root / "selection.json"
        with mock.patch.object(retry.caller, "BASE", original.root), \
                mock.patch.object(retry, "ROOT", root), \
                mock.patch.object(retry, "RETRY_ROOT", retry_root), \
                mock.patch.object(retry, "SELECTION", selected):
            retry_root.mkdir(parents=True)
            selected.write_text(json.dumps(selection()))
            (original.stage.run_dir / "batch_status.txt").write_text("batch_exit=124\nstatus=closed\n")
            resources = retry.profile(retry.stage("draw", index))["resources"]["draw"]
            stage = retry.stage("draw", index, resources)
            stage.run_dir.mkdir()
            scratch = original.root / stage.scratch_name
            scratch.mkdir()
            f = SimpleNamespace(stage=stage, root=original.root, scratch=scratch,
                binding=original.binding, original=original, selection=selected)
            with fixture.prior.machine(f) as actual, mock.patch.dict(os.environ, {
                    "SLURM_ARRAY_JOB_ID": "7000", "SLURM_ARRAY_TASK_COUNT": "8",
                    "SLURM_ARRAY_TASK_MIN": "8", "SLURM_ARRAY_TASK_MAX": "15"}):
                actual.allocation({"ArrayJobId": "7000"})
                yield f


class SecondDrawRetryTests(unittest.TestCase):
    def test_second_attempt_reuses_science_and_preserves_first_adapter(self):
        caller = retry.caller
        self.assertIsNot(first.caller, caller)
        self.assertEqual(first.ATTEMPT, "attempt001")
        self.assertEqual(first.INDICES, (0, 1, 62))
        for index in (8, 15):
            with self.subTest(index=index), retry_files(index) as f, \
                    mock.patch.object(caller, "_configure_file_policy"), \
                    mock.patch.object(caller, "_require_file_policy"):
                original_status = (f.original.stage.run_dir / "batch_status.txt").read_text()
                context, budget = caller.admit("draw", index)
                self.assertEqual(context.stage.stage_h5, f.original.stage.stage_h5)
                self.assertEqual(context.stage.batch_path, f.original.stage.batch_path)
                self.assertIn("attempt002", str(context.stage.run_dir))
                self.assertIn("attempt002", context.stage.scratch_name)
                self.assertEqual(context.profile["draw_retry_selection"], selection())
                self.assertEqual(budget.remaining_seconds(), 170880)

                def operation(ctx, **kwargs):
                    self.assertEqual(kwargs["batch_index"], index)
                    self.assertEqual(kwargs["stage_h5"], f.original.stage.stage_h5)
                    kwargs["require_admitted_context"](ctx, operation=caller.OPERATIONS["draw"],
                        source=None, target=None, budget=kwargs["budget"],
                        execution_options=caller._options(ctx.stage))
                    ctx.stage.batch_path.write_text("{}")
                    return fixture.PhaseResult("draw", ctx.stage.publication_paths)

                with mock.patch.object(caller, "_load_operation",
                        return_value=(operation, fixture.PhaseResult, {"fixture": True})):
                    self.assertEqual(caller.run_once("draw", index), 0)
                report = json.loads((f.stage.run_dir / "report.json").read_text())
                self.assertEqual(report["status"], "complete_draw")
                self.assertEqual(report["array_job_id"], "7000")
                self.assertTrue(report["readers_closed"])
                self.assertEqual((f.original.stage.run_dir / "batch_status.txt").read_text(), original_status)

    def test_unselected_indices_and_changed_attempt_refuse(self):
        for index in (0, 1, 7, 16, 62):
            with self.assertRaises(retry.caller.AdmissionError):
                retry.stage("draw", index)
        with retry_files(8) as f, mock.patch.object(retry.caller, "_load_operation") as load:
            chosen = selection(); chosen["attempt"] = "attempt001"
            f.selection.write_text(json.dumps(chosen))
            with self.assertRaises(retry.caller.AdmissionError):
                retry.caller.run_once("draw", 8)
            load.assert_not_called()

    def test_finalizer_joins_two_disjoint_retries_and_all_original_successes(self):
        runner = fixture.runner
        with fixture.files("finalize") as f, fixture.prior.machine(f):
            choices = [selection(), {**selection(), "attempt": "attempt001",
                "batch_indices": [0, 1, 62], "array_job_id": "6000",
                "failed_original_children": {str(i): {"state": "FAILED",
                    "exit_code": "2:0" if i == 62 else "124:0"} for i in (0, 1, 62)}}]
            choices.reverse()
            (f.stage.root / "draw_retry_bindings.json").write_text(
                json.dumps({"attempts": [v["attempt"] for v in choices]}))
            updated = {**f.binding, "finalize": {"job_id": "4000",
                "afterany_job_id": "3000", "afterok_job_ids": ["6000", "7000"]}}
            (f.stage.root / "dependency_bindings.json").write_text(json.dumps(updated))
            for chosen in choices:
                retry_root = f.stage.root / "draw_retries" / chosen["attempt"]
                retry_root.mkdir(parents=True)
                (retry_root / "selection.json").write_text(json.dumps(chosen))
                for index in chosen["batch_indices"]:
                    original = runner._stage("draw", index, fixture.profile()["resources"]["draw"])
                    code = chosen["failed_original_children"][str(index)]["exit_code"].split(":")[0]
                    (original.run_dir / "batch_status.txt").write_text(f"batch_exit={code}\nstatus=closed\n")
                    stage = replace(original, run_dir=retry_root / f"batch_{index:03d}")
                    fixture.closed_report(stage, str(int(chosen["array_job_id"]) + index + 100), chosen["array_job_id"])
                    report_path = stage.run_dir / "report.json"
                    report = json.loads(report_path.read_text())
                    report["published_paths"] = [str(stage.batch_path)]
                    report_path.write_text(json.dumps(report))
            binding = runner._bindings(f.stage, {"job_id": "4000"})
            runner._completed(f.stage, binding)
            selected = f.stage.root / "draw_retries/attempt002/selection.json"
            overlapping = selection(); overlapping["batch_indices"][0] = 0
            selected.write_text(json.dumps(overlapping))
            with self.assertRaises(runner.AdmissionError):
                runner._bindings(f.stage, {"job_id": "4000"})
            selected.write_text(json.dumps(selection()))
            report_path = f.stage.root / "draw_retries/attempt002/batch_008/report.json"
            original = report_path.read_text()
            for change in ({"array_job_id": "6000"}, {"status": "incomplete"}):
                report_path.write_text(json.dumps({**json.loads(original), **change}))
                with self.assertRaises(runner.AdmissionError):
                    runner._completed(f.stage, binding)
            report_path.write_text(original)
            (f.stage.root / "draws/batch_061/report.json").unlink()
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, binding)


if __name__ == "__main__":
    unittest.main()
