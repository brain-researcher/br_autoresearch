"""Failed-batch recovery and full-63 completion; invented metadata on compute."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import test_amended_full_readout_admission as fixture
import run_amended_draw_retry as retry

caller = retry.caller


def selection():
    return {"authorized": True, "attempt": "attempt001", "batch_indices": [0, 1, 62],
        "array_job_id": "6000", "original_decode_job_id": "1000",
        "original_array_job_id": "3000", "approval_source": "invented_fixture_only",
        "failed_original_children": {str(i): {"state": "FAILED",
            "exit_code": "2:0" if i == 62 else "124:0"} for i in retry.INDICES}}


@contextmanager
def retry_files(index=0):
    with fixture.files("draw", index) as original:
        root = original.stage.root
        retry_root = root / "draw_retries/attempt001"
        selected = retry_root / "selection.json"
        with mock.patch.object(caller, "BASE", original.root), \
                mock.patch.object(retry, "ROOT", root), \
                mock.patch.object(retry, "RETRY_ROOT", retry_root), \
                mock.patch.object(retry, "SELECTION", selected):
            retry_root.mkdir(parents=True)
            selected.write_text(json.dumps(selection()))
            failed = original.stage.run_dir / "batch_status.txt"
            failed.write_text(f"batch_exit={'2' if index == 62 else '124'}\nstatus=closed\n")
            resources = retry.profile(retry.stage("draw", index))["resources"]["draw"]
            stage = retry.stage("draw", index, resources)
            stage.run_dir.mkdir()
            scratch = original.root / stage.scratch_name
            scratch.mkdir()
            yield SimpleNamespace(stage=stage, root=original.root, scratch=scratch,
                binding=original.binding, original=original, selection=selected)


@contextmanager
def machine(f):
    with fixture.prior.machine(f) as actual, mock.patch.dict(os.environ, {
            "SLURM_ARRAY_JOB_ID": "6000", "SLURM_ARRAY_TASK_COUNT": "3"}):
        actual.allocation({"ArrayJobId": "6000"})
        yield actual


class DrawRetryTests(unittest.TestCase):
    def test_fixed_science_and_fresh_paths_reuse_successful_decode(self):
        self.assertIsNot(fixture.runner, caller)
        self.assertIsNot(fixture.runner._stage, retry.stage)
        for index in retry.INDICES:
            with self.subTest(index=index), retry_files(index) as f, machine(f), \
                    mock.patch.object(caller, "_configure_file_policy"), \
                    mock.patch.object(caller, "_require_file_policy"):
                before = (f.original.stage.run_dir / "batch_status.txt").read_text()
                context, budget = caller.admit("draw", index)
                self.assertEqual(context.stage.stage_h5, f.original.stage.stage_h5)
                self.assertEqual(context.stage.batch_path, f.original.stage.batch_path)
                self.assertNotEqual(context.stage.run_dir, f.original.stage.run_dir)
                self.assertEqual(context.stage.resources["deadline_seconds"], 171000)
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
                self.assertEqual(report["array_job_id"], "6000")
                self.assertTrue(report["readers_closed"])
                self.assertEqual((f.original.stage.run_dir / "batch_status.txt").read_text(), before)

    def test_wrong_identity_science_or_successful_original_refuses_before_loader(self):
        for phase, index in (("decode", None), ("finalize", None), ("draw", 2)):
            with self.assertRaises(caller.AdmissionError):
                retry.stage(phase, index)
        for mode in ("wrong_array", "wrong_task", "changed_science", "completed_original"):
            with self.subTest(mode=mode), retry_files() as f, machine(f), \
                    mock.patch.object(caller, "_load_operation") as load:
                if mode == "wrong_array":
                    os.environ["SLURM_ARRAY_JOB_ID"] = "6001"
                elif mode == "wrong_task":
                    os.environ["SLURM_ARRAY_TASK_ID"] = "1"
                elif mode == "changed_science":
                    value = fixture.profile(); value["decoded_slots"] = 703
                    f.original.stage.profile_path.write_text(json.dumps(value))
                else:
                    (f.original.stage.run_dir / "batch_status.txt").write_text("batch_exit=0\nstatus=closed\n")
                with self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 0)
                load.assert_not_called()

    def test_no_overwrite_and_revoked_selection_stops_execution(self):
        for suffix in ("", ".partial"):
            with self.subTest(suffix=suffix), retry_files() as f, machine(f), \
                    mock.patch.object(caller, "_load_operation") as load:
                Path(str(f.stage.batch_path) + suffix).write_text("preserved")
                with self.assertRaises(caller.AdmissionError):
                    caller.run_once("draw", 0)
                load.assert_not_called()
        with retry_files() as f, machine(f), mock.patch.object(caller, "_require_file_policy"):
            context, budget = caller.admit("draw", 0)
            chosen = selection(); chosen["authorized"] = False
            f.selection.write_text(json.dumps(chosen))
            with self.assertRaises(caller.AdmissionError):
                budget.check_policy(caller.OPERATIONS["draw"])

    def test_finalizer_requires_all_effective_batches_and_never_falls_back(self):
        runner = fixture.runner
        with fixture.files("finalize") as f, fixture.prior.machine(f):
            selected = selection()
            retry_root = f.stage.root / "draw_retries/attempt001"
            retry_root.mkdir(parents=True)
            (retry_root / "selection.json").write_text(json.dumps(selected))
            (f.stage.root / "draw_retry_bindings.json").write_text(json.dumps({"attempts": ["attempt001"]}))
            updated = {**f.binding, "finalize": {"job_id": "4000",
                "afterany_job_id": "3000", "afterok_job_ids": ["6000"]}}
            (f.stage.root / "dependency_bindings.json").write_text(json.dumps(updated))
            for index in retry.INDICES:
                original = runner._stage("draw", index, fixture.profile()["resources"]["draw"])
                (original.run_dir / "batch_status.txt").write_text(
                    f"batch_exit={'2' if index == 62 else '124'}\nstatus=closed\n")
                stage = replace(original, run_dir=retry_root / f"batch_{index:03d}")
                fixture.closed_report(stage, str(6100 + index), "6000")
                path = stage.run_dir / "report.json"
                report = json.loads(path.read_text())
                report["published_paths"] = [str(stage.batch_path)]
                path.write_text(json.dumps(report))
            binding = runner._bindings(f.stage, {"job_id": "4000"})
            runner._completed(f.stage, binding)
            report_path = retry_root / "batch_062/report.json"
            original = report_path.read_text()
            for change in ({"status": "incomplete"}, {"array_job_id": "3000"},
                    {"published_paths": ["wrong.json"]}, {"readers_closed": False}):
                report_path.write_text(json.dumps({**json.loads(original), **change}))
                with self.assertRaises(runner.AdmissionError):
                    runner._completed(f.stage, binding)
            report_path.unlink()
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, binding)
            report_path.write_text(original)
            # Even an omitted untouched batch prevents the full finalizer.
            untouched = f.stage.root / "draws/batch_061/report.json"
            untouched.unlink()
            with self.assertRaises(runner.AdmissionError):
                runner._completed(f.stage, binding)


if __name__ == "__main__":
    unittest.main()
