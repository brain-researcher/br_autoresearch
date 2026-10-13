"""Three stdlib-only production admission/composition checks; Slurm execution only.

Mocked operations and temporary files only. Resource fixture durations are not
production selections; the actual authorization retains unset runtime limits.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_content_pair as runner


class ContentPairAdmissionTests(unittest.TestCase):
    def profile(self):
        resources = {**runner.FIXED_RESOURCES, "cpus": 3, "memory_gib": 16, "minutes": 1440,
                     "deadline_seconds": 84600, "allocated_cpu_hours": 72, "rss_limit_bytes": 15 * 1024**3}
        return {**runner.FIXED, "resources": resources, "input_scope": dict(runner.INPUT_SCOPE),
                "authorized": True, "approval_source": "fixture_only_not_a_production_selection"}

    def context_budget(self, root):
        scratch = root / "scratch"
        scratch.mkdir()
        context = runner.ContentPairContext(self.profile(), {
            "job_id": os.environ.get("SLURM_JOB_ID", ""), "hostname": socket.gethostname().split(".")[0],
            "job_end_epoch": time.time() + 86400}, scratch)
        return context, runner.ContentPairBudget(context)

    def test_inactive_or_unset_resources_stop_before_allocation_and_exact_operation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "proposal.json"
            with mock.patch.object(runner, "PROFILE", profile_path), \
                    mock.patch.object(runner, "_allocation") as allocation, \
                    mock.patch.object(runner, "_load_operations") as load:
                profile = self.profile()
                profile.update(authorized=False, approval_source=None)
                profile_path.write_text(json.dumps(profile))
                with self.assertRaisesRegex(runner.AdmissionError, "profile_not_activated"):
                    runner.run_once()
                for field, value in (("minutes", None), ("deadline_seconds", None),
                                     ("allocated_cpu_hours", None), ("allocated_cpu_hours", float("nan")),
                                     ("allocated_cpu_hours", float("inf")), ("allocated_cpu_hours", 71),
                                     ("cpus", True), ("minutes", 10081), ("deadline_seconds", 86400),
                                     ("rss_limit_bytes", 17 * 1024**3)):
                    with self.subTest(resource=field, value=value):
                        profile = self.profile()
                        profile["resources"][field] = value
                        profile_path.write_text(json.dumps(profile))
                        with self.assertRaises(runner.AdmissionError):
                            runner.run_once()
                for change in ({"iterations": 4}, {"training_images": 897}, {"pair_index": 0},
                               {"production_training": False}, {"input_scope": {**runner.INPUT_SCOPE, "tau": True}}):
                    profile_path.write_text(json.dumps({**self.profile(), **change}))
                    with self.assertRaises(runner.AdmissionError):
                        runner.run_once()
                allocation.assert_not_called()
                load.assert_not_called()
                profile_path.write_text(json.dumps(self.profile()))
                self.assertEqual(runner._profile(), self.profile())
            context, budget = self.context_budget(root)
            with mock.patch.object(runner, "RUN_DIR", root), mock.patch.object(runner, "_profile", return_value=context.profile):
                arguments = dict(source=runner.SOURCE, target=runner.TARGET, budget=budget)
                options = runner._execution_options(context.profile)
                runner.resolver(context, operation="bind_content_inputs", execution_options=None, **arguments)
                runner.resolver(context, operation="train_and_publish_full_pair", execution_options=options, **arguments)
                for change in ({"operation": "measure_content_pair_four_updates"},
                               {"operation": "train_and_publish_content_pair"}, {"source": runner.TARGET},
                               {"execution_options": {**options, "updates": 4}},
                               {"execution_options": {**options, "maximum_bytes": 805306368}},
                               {"execution_options": {**options, "final_path": root / "other.pt"}},
                               {"execution_options": {**options, "backend": "dense"}}):
                    with self.subTest(change=change), self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, **{**arguments, "operation": "train_and_publish_full_pair",
                                                  "execution_options": options, **change})

    def test_actual_allocation_must_match_selected_finite_resources(self):
        now = datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()
        end = datetime.fromtimestamp(now + 86400, tz=timezone.utc).replace(tzinfo=None).isoformat()
        env = {**{key: "1" for key in runner.THREADS}, "SLURM_JOB_ID": "12345678",
               "SLURM_CPUS_PER_TASK": "3", "SLURM_MEM_PER_NODE": "16384", "SLURM_NTASKS": "1",
               "SLURM_NNODES": "1", "CUDA_VISIBLE_DEVICES": "", "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": "",
               "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": ""}
        fields = {"JobId": "12345678", "NodeList": "sh04-fixture", "JobState": "RUNNING",
                  "NumCPUs": "3", "NumTasks": "1", "NumNodes": "1", "Partition": "russpold",
                  "TimeLimit": "1-00:00:00", "MinMemoryNode": "16G", "AllocTRES": "cpu=3,node=1,mem=16G",
                  "EndTime": end}
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(mock.patch.object(runner, "RUN_DIR", root))
            stack.enter_context(mock.patch.object(runner.socket, "gethostname", return_value="sh04-fixture.stanford.edu"))
            stack.enter_context(mock.patch.object(runner.time, "time", return_value=now))
            stack.enter_context(mock.patch.dict(os.environ, env))
            def allocation(changes):
                (root / "allocation.txt").write_text(" ".join(f"{key}={value}" for key, value in {**fields, **changes}.items()))
                return runner._allocation(self.profile())
            for limit in ("24:00:00", "1-00:00:00"):
                self.assertEqual(allocation({"TimeLimit": limit})["minutes"], 1440)
            for change in ({"TimeLimit": "2-00:00:00"}, {"NumCPUs": "1"}, {"MinMemoryNode": "7G"},
                           {"AllocTRES": "cpu=3,node=1,mem=16G,gres/gpu=1"}, {"NodeList": "other-node"},
                           {"EndTime": "2026-10-11T23:00:00"}, {"EndTime": "2026-10-12T00:00:00-07:00"}):
                with self.subTest(change=change), self.assertRaises(runner.AdmissionError):
                    allocation(change)
            with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}), self.assertRaises(runner.AdmissionError):
                allocation({})
            scratch = root / "ep21_content_pair_attempt001"
            scratch.mkdir()
            with mock.patch.dict(os.environ, {"L_SCRATCH_JOB": str(root), "TMPDIR": str(scratch)}):
                self.assertEqual(runner._scratch(), scratch)
            with mock.patch.dict(os.environ, {"L_SCRATCH_JOB": str(root), "TMPDIR": str(root)}), \
                    self.assertRaises(runner.AdmissionError):
                runner._scratch()

    def test_one_full_pair_claim_bounded_progress_and_redacted_failure(self):
        for outcome in ("complete", "wrong_pair", "incomplete_fit", "fit_failure", "publication_failure", "closeout_failure"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, progress_path = root / "final_content_pair.pt", root / "progress.jsonl"
                context, budget = self.context_budget(root)
                if outcome == "closeout_failure":
                    budget.check_storage = mock.Mock(side_effect=OSError("sensitive_closeout_text"))
                source = SimpleNamespace(axis=SimpleNamespace(participant="sub-03", eligible_native_ids=range(12897)),
                    image_ids=range(4712), normalized_X=SimpleNamespace(shape=(4712, 12897)))
                initial = SimpleNamespace(pair_index=0 if outcome == "wrong_pair" else 4,
                    source="sub-03", target="sub-01", input_width=12897, output_width=12294)
                panel = SimpleNamespace(axes=range(5), initializations=[initial] * 20,
                    channel_schedule=SimpleNamespace(channels_by_layer=[range(1024)] * 19))
                bound = SimpleNamespace(source=source, target="sub-01", panel=panel,
                    pair=SimpleNamespace(target_axis=SimpleNamespace(eligible_native_ids=range(12294))))
                def bind(actual_context, **kwargs):
                    self.assertIs(actual_context, context)
                    self.assertIs(kwargs["budget"], budget)
                    self.assertIs(kwargs["require_admitted_context"], runner.resolver)
                    self.assertEqual((kwargs["source"], kwargs["target"]), ("sub-03", "sub-01"))
                    self.assertTrue((root / "claim.json").is_file())
                    return bound
                def train(actual_bound, *, on_iteration, **kwargs):
                    self.assertIs(actual_bound, bound)
                    self.assertEqual(kwargs, {"device": "cpu", "block_units": 256, "row_chunk_size": 512,
                        "backend": "sparse", "final_path": path, "maximum_bytes": 1073741824})
                    count = 2 if outcome in ("fit_failure", "incomplete_fit") else 1024
                    for completed in range(1, count + 1):
                        on_iteration({"completed_iterations": completed, "training_loss": "sensitive_loss_value",
                                      "learning_rate_used": "sensitive_rate_value"})
                    if outcome == "fit_failure":
                        raise RuntimeError("sensitive_scientific_exception")
                    if outcome == "publication_failure":
                        path.with_name(path.name + ".partial").write_bytes(b"fixture_partial")
                        raise RuntimeError("sensitive_publication_exception")
                    path.write_bytes(b"fixture_final_model")
                    return SimpleNamespace(source="sub-03", target="sub-01", pair_index=4, path=path,
                        completion=SimpleNamespace(iterations=1023 if outcome == "incomplete_fit" else 1024,
                            rows=4712, backend="sparse", last_step_loss_before_update="sensitive_private_loss"))
                with mock.patch.object(runner, "RUN_DIR", root), mock.patch.object(runner, "FINAL", path), \
                        mock.patch.object(runner, "PROGRESS", progress_path), \
                        mock.patch.object(runner, "admit", return_value=(context, budget)), \
                        mock.patch.object(runner, "_load_operations", return_value=(bind, mock.Mock(side_effect=train),
                            {"python": "fixture", "numpy": "fixture", "torch": "fixture"})) as loader:
                    result = runner.run_once()
                    self.assertEqual(result, 0 if outcome == "complete" else 1)
                    report_text = (root / "report.json").read_text()
                    progress_text = progress_path.read_text()
                    self.assertNotIn("sensitive", report_text + progress_text)
                    self.assertNotIn("training_loss", report_text + progress_text)
                    report = json.loads(report_text)
                    records = [json.loads(line) for line in progress_text.splitlines()]
                    observed = 0 if outcome == "wrong_pair" else 2 if outcome in ("fit_failure", "incomplete_fit") else 1024
                    self.assertEqual(report["observed_completed_updates"], observed)
                    self.assertEqual([record["completed_update"] for record in records], list(range(1, observed + 1)))
                    self.assertTrue(all(set(record) == {"completed_update", "elapsed_wall_seconds"} for record in records))
                    self.assertLessEqual(progress_path.stat().st_size, 262144)
                    successful_fit = outcome in ("complete", "closeout_failure")
                    self.assertEqual(report["completed_updates"], 1024 if successful_fit else None)
                    self.assertEqual(report["model_published"], True if successful_fit else False if outcome == "wrong_pair" else None)
                    self.assertEqual(report["published_path"], str(path) if successful_fit else None)
                    self.assertEqual(loader.return_value[1].call_count, 0 if outcome == "wrong_pair" else 1)
                    if outcome == "publication_failure":
                        self.assertEqual(path.with_name(path.name + ".partial").read_bytes(), b"fixture_partial")
                    with self.assertRaises((runner.AdmissionError, FileExistsError)):
                        runner.run_once()
                    self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
