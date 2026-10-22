"""Gapped third retry and three-attempt finalization; invented metadata only."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import test_amended_full_readout_admission as fixture
import run_amended_draw_retry as first
from run_amended_draw_retry002 import adapter as second
from run_amended_draw_retry003 import adapter as retry


INDICES = (*range(16, 27), 28)


def selection(attempt="attempt003", indices=INDICES, array="8000"):
    return {"authorized": True, "attempt": attempt, "batch_indices": list(indices),
        "array_job_id": array, "original_decode_job_id": "1000",
        "original_array_job_id": "3000", "approval_source": "invented_fixture_only",
        "failed_original_children": {str(i): {"state": "FAILED",
            "exit_code": "2:0" if i == 62 else "124:0"} for i in indices}}


@contextmanager
def retry_files(index):
    with fixture.files("draw", index) as original:
        root = original.stage.root
        retry_root = root / "draw_retries/attempt003"
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
                    "SLURM_ARRAY_JOB_ID": "8000", "SLURM_ARRAY_TASK_COUNT": "12",
                    "SLURM_ARRAY_TASK_MIN": "16", "SLURM_ARRAY_TASK_MAX": "28"}):
                actual.allocation({"ArrayJobId": "8000"})
                yield f


class ThirdDrawRetryTests(unittest.TestCase):
    def test_gapped_retry_keeps_canonical_outputs_and_earlier_attempts(self):
        caller = retry.caller
        self.assertIsNot(caller, first.caller)
        self.assertIsNot(caller, second.caller)
        self.assertEqual((first.ATTEMPT, first.INDICES), ("attempt001", (0, 1, 62)))
        self.assertEqual((second.ATTEMPT, second.INDICES), ("attempt002", tuple(range(8, 16))))
        self.assertEqual(retry.INDICES, INDICES)
        script = (Path(__file__).resolve().parents[1] / "finding_a_draw_retry003.sbatch").read_text()
        for directive in ("--array=16-26,28%3", "--time=2-00:00:00",
                "--cpus-per-task=3", "--mem=16G"):
            self.assertIn("#SBATCH " + directive + "\n", script)
        for index in (16, 26, 28):
            with self.subTest(index=index), retry_files(index) as f, \
                    mock.patch.object(caller, "_configure_file_policy"), \
                    mock.patch.object(caller, "_require_file_policy"):
                original_status = (f.original.stage.run_dir / "batch_status.txt").read_text()
                context, budget = caller.admit("draw", index)
                self.assertEqual(context.stage.stage_h5, f.original.stage.stage_h5)
                self.assertEqual(context.stage.batch_path, f.original.stage.batch_path)
                self.assertIn("attempt003", str(context.stage.run_dir))
                self.assertIn("attempt003", context.stage.scratch_name)
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
                self.assertEqual(report["array_job_id"], "8000")
                self.assertTrue(report["readers_closed"])
                self.assertEqual((f.original.stage.run_dir / "batch_status.txt").read_text(), original_status)

    def test_gap_wrong_shape_and_changed_selection_refuse_before_operation(self):
        for index in (0, 15, 27, 29, 62):
            with self.subTest(index=index), self.assertRaises(retry.caller.AdmissionError):
                retry.stage("draw", index)
        with retry_files(28) as f, mock.patch.object(retry.caller, "_load_operation") as load:
            for changed in ({"SLURM_ARRAY_TASK_COUNT": "13"},
                    {"SLURM_ARRAY_TASK_MIN": "15"}, {"SLURM_ARRAY_TASK_MAX": "27"},
                    {"SLURM_ARRAY_JOB_ID": "7000"}, {"SLURM_ARRAY_TASK_ID": "27"}):
                with self.subTest(changed=changed), mock.patch.dict(os.environ, changed):
                    with self.assertRaises(retry.caller.AdmissionError):
                        retry.caller.run_once("draw", 28)
            for changed in ({"attempt": "attempt002"}, {"batch_indices": list(range(16, 29))}):
                f.selection.write_text(json.dumps({**selection(), **changed}))
                with self.subTest(changed=changed), self.assertRaises(retry.caller.AdmissionError):
                    retry.caller.run_once("draw", 28)
            load.assert_not_called()

    def test_finalizer_joins_three_retries_and_requires_all_63_closed_batches(self):
        runner = fixture.runner
        with fixture.files("finalize") as f, fixture.prior.machine(f):
            choices = [selection("attempt001", (0, 1, 62), "6000"),
                selection("attempt002", tuple(range(8, 16)), "7000"), selection()]
            (f.stage.root / "draw_retry_bindings.json").write_text(
                json.dumps({"attempts": [v["attempt"] for v in choices]}))
            updated = {**f.binding, "finalize": {"job_id": "4000",
                "afterany_job_id": "3000", "afterok_job_ids": ["6000", "7000", "8000"]}}
            binding_path = f.stage.root / "dependency_bindings.json"
            binding_path.write_text(json.dumps(updated))
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
            self.assertEqual(sum(len(v["batch_indices"]) for v in binding["draw_retries"].values()), 23)
            runner._completed(f.stage, binding)

            # Every retry remains an explicit successful scheduler prerequisite.
            binding_path.write_text(json.dumps({**updated, "finalize": {
                **updated["finalize"], "afterok_job_ids": ["6000", "7000"]}}))
            with self.assertRaises(runner.AdmissionError):
                runner._bindings(f.stage, {"job_id": "4000"})
            binding_path.write_text(json.dumps(updated))
            selected = f.stage.root / "draw_retries/attempt003/selection.json"
            overlapping = selection(); overlapping["batch_indices"][0] = 8
            selected.write_text(json.dumps(overlapping))
            with self.assertRaises(runner.AdmissionError):
                runner._bindings(f.stage, {"job_id": "4000"})
            selected.write_text(json.dumps(selection()))

            report_path = f.stage.root / "draw_retries/attempt003/batch_028/report.json"
            original = report_path.read_text()
            for change in ({"array_job_id": "7000"}, {"status": "incomplete"},
                    {"readers_closed": False}, {"published_paths": []}):
                report_path.write_text(json.dumps({**json.loads(original), **change}))
                with self.subTest(change=change), self.assertRaises(runner.AdmissionError):
                    runner._completed(f.stage, binding)
            report_path.write_text(original)
            # Batch 27 is deliberately retained from the original array.
            gap_status = f.stage.root / "draws/batch_027/batch_status.txt"
            original = gap_status.read_text()
            gap_status.write_text("phase_exit=124\nstatus=closed\n")
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, binding)
            gap_status.write_text(original)
            runner._completed(f.stage, binding)


if __name__ == "__main__":
    unittest.main()
