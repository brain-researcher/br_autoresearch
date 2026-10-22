"""Batch-37 metadata-read repair, with invented operational records only."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import test_amended_full_readout_admission as fixture
from test_amended_draw_retry003 import selection as prior_selection
import run_amended_draw_retry as first
from run_amended_draw_retry002 import adapter as second
from run_amended_draw_retry003 import adapter as third
from run_amended_draw_retry004 import adapter as retry


def selection():
    value = prior_selection("attempt004", (37,), "9000")
    value["failed_original_children"]["37"] = {"state": "FAILED", "exit_code": "1:0",
        "error_type": "JSONDecodeError", "original_job_id": "3137"}
    return value


def failed_original(stage):
    report = {"phase": "draw", "batch_index": 37, "status": "incomplete",
        "model_policy": fixture.runner.MODEL_POLICY, "error_type": "JSONDecodeError",
        "failed_stage": "execute_phase", "artifacts_published": None,
        "published_paths": None, "readers_closed": False, "resumable": False,
        "allocation_identity": {"job_id": "3137"}}
    (stage.run_dir / "report.json").write_text(json.dumps(report))
    (stage.run_dir / "batch_status.txt").write_text("phase_exit=1\nbatch_exit=1\nstatus=closed\n")
    return report


@contextmanager
def retry_files():
    with fixture.files("draw", 37) as original:
        root = original.stage.root
        retry_root = root / "draw_retries/attempt004"
        selected = retry_root / "selection.json"
        with mock.patch.object(retry.caller, "BASE", original.root), \
                mock.patch.object(retry, "ROOT", root), \
                mock.patch.object(retry, "RETRY_ROOT", retry_root), \
                mock.patch.object(retry, "SELECTION", selected):
            retry_root.mkdir(parents=True)
            selected.write_text(json.dumps(selection()))
            failed_original(original.stage)
            resources = retry.profile(retry.stage("draw", 37))["resources"]["draw"]
            stage = retry.stage("draw", 37, resources)
            stage.run_dir.mkdir()
            scratch = original.root / stage.scratch_name
            scratch.mkdir()
            f = SimpleNamespace(stage=stage, root=original.root, scratch=scratch,
                binding=original.binding, original=original, selection=selected)
            with fixture.prior.machine(f) as actual, mock.patch.dict(os.environ, {
                    "SLURM_ARRAY_JOB_ID": "9000", "SLURM_ARRAY_TASK_COUNT": "1",
                    "SLURM_ARRAY_TASK_MIN": "37", "SLURM_ARRAY_TASK_MAX": "37"}):
                actual.allocation({"ArrayJobId": "9000"})
                yield f


class FourthDrawRetryTests(unittest.TestCase):
    def test_batch37_retry_preserves_failure_and_original_science(self):
        caller = retry.caller
        for prior in (first, second, third):
            self.assertIsNot(caller, prior.caller)
            self.assertEqual(prior.failure_exit_code(37), "124")
            self.assertEqual(prior.failure_exit_code(62), "2")
        self.assertEqual(retry.INDICES, (37,))
        script = (Path(__file__).resolve().parents[1] / "finding_a_draw_retry004.sbatch").read_text()
        for directive in ("--array=37%1", "--time=2-00:00:00", "--cpus-per-task=3", "--mem=16G"):
            self.assertIn("#SBATCH " + directive + "\n", script)
        with retry_files() as f, mock.patch.object(caller, "_configure_file_policy"), \
                mock.patch.object(caller, "_require_file_policy"):
            preserved = {name: (f.original.stage.run_dir / name).read_text()
                for name in ("report.json", "batch_status.txt")}
            context, budget = caller.admit("draw", 37)
            self.assertEqual(context.stage.stage_h5, f.original.stage.stage_h5)
            self.assertEqual(context.stage.batch_path, f.original.stage.batch_path)
            self.assertEqual(budget.remaining_seconds(), 170880)
            self.assertEqual(retry.failure_exit_code(37), "1")

            def operation(ctx, **kwargs):
                self.assertEqual(kwargs["batch_index"], 37)
                kwargs["require_admitted_context"](ctx, operation=caller.OPERATIONS["draw"],
                    source=None, target=None, budget=kwargs["budget"],
                    execution_options=caller._options(ctx.stage))
                ctx.stage.batch_path.write_text("{}")
                return fixture.PhaseResult("draw", ctx.stage.publication_paths)

            with mock.patch.object(caller, "_load_operation",
                    return_value=(operation, fixture.PhaseResult, {"fixture": True})):
                self.assertEqual(caller.run_once("draw", 37), 0)
            report = json.loads((f.stage.run_dir / "report.json").read_text())
            self.assertEqual(report["status"], "complete_draw")
            self.assertEqual(report["array_job_id"], "9000")
            self.assertTrue(report["readers_closed"])
            for name, text in preserved.items():
                self.assertEqual((f.original.stage.run_dir / name).read_text(), text)

    def test_exact_diagnosis_and_closed_exit1_required_before_operation(self):
        caller = retry.caller
        for index in (16, 36, 38, 62):
            with self.subTest(index=index), self.assertRaises(caller.AdmissionError):
                retry.stage("draw", index)
        with retry_files() as f, mock.patch.object(caller, "_load_operation") as load:
            for change in ({"exit_code": "124:0"}, {"error_type": "ValueError"},
                    {"original_job_id": "3138"}, {"state": "RUNNING"}):
                chosen = selection()
                chosen["failed_original_children"]["37"].update(change)
                f.selection.write_text(json.dumps(chosen))
                with self.subTest(change=change), self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 37)
            f.selection.write_text(json.dumps(selection()))
            path = f.original.stage.run_dir / "report.json"
            original = path.read_text()
            for change in ({"error_type": "ValueError"}, {"failed_stage": "load_operation"},
                    {"readers_closed": True}, {"resumable": True}, {"artifacts_published": True},
                    {"published_paths": []}, {"batch_index": 38}, {"status": "complete_draw"}):
                path.write_text(json.dumps({**json.loads(original), **change}))
                with self.subTest(change=change), self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 37)
            path.write_text(original)
            status = f.original.stage.run_dir / "batch_status.txt"
            status.write_text("phase_exit=1\nbatch_exit=1\nstatus=running\n")
            with self.assertRaises(caller.AdmissionError):
                caller.run_once("draw", 37)
            load.assert_not_called()

    def test_partial_binding_read_fails_closed_and_atomic_replacement_does_not_resume(self):
        caller = retry.caller
        with retry_files() as f, mock.patch.object(caller, "_configure_file_policy"), \
                mock.patch.object(caller, "_require_file_policy"):
            binding = f.stage.root / "dependency_bindings.json"
            original = binding.read_text()
            staged = binding.with_name("dependency_bindings.new.json")
            with staged.open("w") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
            # A prepared replacement leaves current readers on complete JSON.
            caller.admit("draw", 37)

            def operation(ctx, **kwargs):
                binding.write_text('{"decode":')  # Reproduce an interrupted in-place metadata write.
                kwargs["budget"]._next_policy_check = 0.
                kwargs["require_admitted_context"](ctx, operation=caller.OPERATIONS["draw"],
                    source=None, target=None, budget=kwargs["budget"],
                    execution_options=caller._options(ctx.stage))
                self.fail("partial metadata must not admit further work")

            with mock.patch.object(caller, "_load_operation",
                    return_value=(operation, fixture.PhaseResult, {"fixture": True})):
                self.assertEqual(caller.run_once("draw", 37), 1)
            report = json.loads((f.stage.run_dir / "report.json").read_text())
            self.assertEqual(report["error_type"], "JSONDecodeError")
            self.assertEqual(report["failed_stage"], "execute_phase")
            self.assertFalse(report["readers_closed"])
            self.assertFalse(f.stage.batch_path.exists())
            os.replace(staged, binding)
            caller.admit("draw", 37)
            with mock.patch.object(caller, "_load_operation") as load:
                with self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 37)  # A closed attempt cannot silently resume.
                load.assert_not_called()

    def test_finalizer_joins_four_attempts_and_checks_metadata_failure_and_all63(self):
        runner = fixture.runner
        with fixture.files("finalize") as f, fixture.prior.machine(f):
            choices = [prior_selection("attempt001", (0, 1, 62), "6000"),
                prior_selection("attempt002", tuple(range(8, 16)), "7000"),
                prior_selection(), selection()]
            (f.stage.root / "draw_retry_bindings.json").write_text(
                json.dumps({"attempts": [v["attempt"] for v in choices]}))
            updated = {**f.binding, "finalize": {"job_id": "4000", "afterany_job_id": "3000",
                "afterok_job_ids": ["6000", "7000", "8000", "9000"]}}
            (f.stage.root / "dependency_bindings.json").write_text(json.dumps(updated))
            for chosen in choices:
                retry_root = f.stage.root / "draw_retries" / chosen["attempt"]
                retry_root.mkdir(parents=True)
                (retry_root / "selection.json").write_text(json.dumps(chosen))
                for index in chosen["batch_indices"]:
                    original = runner._stage("draw", index, fixture.profile()["resources"]["draw"])
                    if index == 37:
                        failed_original(original)
                    else:
                        code = chosen["failed_original_children"][str(index)]["exit_code"].split(":")[0]
                        (original.run_dir / "batch_status.txt").write_text(f"batch_exit={code}\nstatus=closed\n")
                    stage = replace(original, run_dir=retry_root / f"batch_{index:03d}")
                    fixture.closed_report(stage, str(int(chosen["array_job_id"]) + index + 100), chosen["array_job_id"])
                    report_path = stage.run_dir / "report.json"
                    report = json.loads(report_path.read_text())
                    report["published_paths"] = [str(stage.batch_path)]
                    report_path.write_text(json.dumps(report))
            bound = runner._bindings(f.stage, {"job_id": "4000"})
            self.assertEqual(sum(len(v["batch_indices"]) for v in bound["draw_retries"].values()), 24)
            runner._completed(f.stage, bound)
            chosen = f.stage.root / "draw_retries/attempt004/selection.json"
            invalid = selection(); invalid["batch_indices"] = [38]
            invalid["failed_original_children"] = {"38": invalid["failed_original_children"]["37"]}
            chosen.write_text(json.dumps(invalid))
            with self.assertRaises(runner.AdmissionError):
                runner._bindings(f.stage, {"job_id": "4000"})
            chosen.write_text(json.dumps(selection()))
            failure = f.stage.root / "draws/batch_037/report.json"
            original = failure.read_text()
            failure.write_text(json.dumps({**json.loads(original), "error_type": "ValueError"}))
            with self.assertRaises(runner.AdmissionError):
                runner._bindings(f.stage, {"job_id": "4000"})
            failure.write_text(original)
            retry_status = f.stage.root / "draw_retries/attempt004/batch_037/batch_status.txt"
            retry_status.write_text("phase_exit=1\nstatus=closed\n")
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, bound)
            retry_status.write_text("phase_exit=0\nstatus=closed\n")
            runner._completed(f.stage, bound)
            (f.stage.root / "draws/batch_061/batch_status.txt").write_text("phase_exit=1\nstatus=closed\n")
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, bound)


if __name__ == "__main__":
    unittest.main()
