"""Three mocked production admission/composition checks; Slurm execution only.

Durations below are fixture values, not production selections. No CUDA,
scientific array, input payload or model fitting is performed by these checks.
"""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_content_cuda_pair as runner
from test_content_cuda_admission import fake_torch


NOW = 2_000_000_000.


def profile():
    return {**runner.FIXED, "authorized": True,
        "approval_source": "fixture_only_not_a_production_selection",
        "input_scope": dict(runner.INPUT_SCOPE),
        "resources": {**runner.FIXED_RESOURCES, "minutes": 1440, "deadline_seconds": 84600,
                      "allocated_cpu_hours": 48, "allocated_gpu_hours": 24}}


def allocation(changes=None):
    def stamp(epoch):
        return datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None).isoformat()
    fields = {"JobId": "123", "JobState": "RUNNING", "NodeList": "sh03-12n01",
        "NumCPUs": "2", "NumTasks": "1", "NumNodes": "1", "Partition": "gpu",
        "TimeLimit": "1-00:00:00", "MinMemoryNode": "16G",
        "AllocTRES": "cpu=2,mem=16G,node=1,gres/gpu=1,gres/gpu:fixture=1",
        "StartTime": stamp(NOW - 120), "EndTime": stamp(NOW + 86280)}
    fields.update(changes or {})
    return " ".join(f"{key}={value}" for key, value in fields.items())


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary).resolve()
        run, local = root / "run", root / "local"
        run.mkdir()
        local.mkdir()
        scratch = local / runner.SCRATCH_NAME
        scratch.mkdir()
        grant = root / "profile.json"
        grant.write_text(json.dumps(profile()))
        (run / "allocation.txt").write_text(allocation())
        for name, path in (("PROFILE", grant), ("RUN_DIR", run),
                           ("FINAL", run / "final_content_pair.pt"), ("PROGRESS", run / "progress.jsonl")):
            stack.enter_context(mock.patch.object(runner, name, path))
        env = {**{key: "1" for key in runner.THREADS}, "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "16384",
            "SLURM_GPUS": "1", "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_GPUS": "0",
            "CUDA_VISIBLE_DEVICES": "0", "L_SCRATCH_JOB": str(local), "TMPDIR": str(scratch)}
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(runner.socket, "gethostname", return_value="sh03-12n01.int"))
        stack.enter_context(mock.patch.object(runner.time, "time", return_value=NOW))
        stack.enter_context(mock.patch.object(runner.time, "monotonic", return_value=1000.))
        stack.enter_context(mock.patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1024)))
        torch = fake_torch()
        torch.cuda.total = 32 * 1024**3
        torch.cuda.get_device_properties = lambda device: SimpleNamespace(
            total_memory=torch.cuda.total, name=runner.FIXED["device_name"])
        stack.enter_context(mock.patch.dict(sys.modules, {"torch": torch}))
        yield SimpleNamespace(root=root, run=run, grant=grant, scratch=scratch, torch=torch)


class ContentCudaPairAdmissionTests(unittest.TestCase):
    def test_inactive_unset_or_changed_production_grant_precedes_imports_and_exact_operation(self):
        self.assertEqual(runner.PROFILE, runner.BASE / "outputs/content_cuda_pair_training_authorization.json")
        self.assertEqual(runner.RUN_DIR, runner.BASE / "outputs/runs/content_pair_training/cuda001")
        with fixture() as f, mock.patch.object(runner, "_allocation") as actual, \
                mock.patch.object(runner, "_initialize_cuda") as cuda, \
                mock.patch.object(runner, "_load_operations") as load:
            inactive = {**profile(), "authorized": False, "approval_source": None}
            f.grant.write_text(json.dumps(inactive))
            with self.assertRaisesRegex(runner.AdmissionError, "profile_not_activated"):
                runner.run_once()
            bad_resources = [(key, None) for key in ("minutes", "deadline_seconds", "allocated_cpu_hours", "allocated_gpu_hours")]
            bad_resources += [("allocated_cpu_hours", float("nan")), ("allocated_gpu_hours", float("inf")),
                ("allocated_cpu_hours", 47), ("allocated_gpu_hours", 23), ("minutes", True),
                ("minutes", 2881), ("deadline_seconds", 86400), ("cpus", 3), ("gpus", True),
                ("rss_limit_bytes", 16 * 1024**3), ("gpu_memory_limit_bytes", 21 * 1024**3)]
            for key, value in bad_resources:
                changed = profile()
                changed["resources"][key] = value
                f.grant.write_text(json.dumps(changed))
                with self.subTest(resource=key, value=value), self.assertRaises(runner.AdmissionError):
                    runner.run_once()
            for change in ({"operation": "measure_content_pair_cuda_cached_four_updates"},
                           {"iterations": 4}, {"training_images": 897}, {"pair_index": 0},
                           {"target_eligible_voxels": 12295}, {"maximum_retained_bytes": 4294967297},
                           {"device_name": "other GPU"}, {"scheduler_constraint": "GPU_MEM:32GB"},
                           {"tf32": True}, {"input_scope": {**runner.INPUT_SCOPE, "tau": True}}):
                f.grant.write_text(json.dumps({**profile(), **change}))
                with self.assertRaises(runner.AdmissionError):
                    runner.run_once()
            actual.assert_not_called()
            cuda.assert_not_called()
            load.assert_not_called()
            self.assertFalse((f.run / "claim.json").exists())
        with fixture() as f:
            context, budget = runner.admit()
            common = dict(source=runner.SOURCE, target=runner.TARGET, budget=budget)
            options = runner._execution_options()
            self.assertEqual(set(options), {"device", "block_units", "row_chunk_size", "backend",
                "final_path", "maximum_bytes", "feature_read_mode", "maximum_channel_bytes", "maximum_retained_bytes"})
            runner.resolver(context, operation="bind_content_inputs", execution_options=None, **common)
            runner.resolver(context, operation=runner.FIXED["operation"], execution_options=options, **common)
            invalid = [(name, options) for name in ("measure_content_pair_cuda_four_updates",
                "measure_content_pair_cuda_cached_four_updates", "train_and_publish_full_pair")]
            invalid += [(runner.FIXED["operation"], {**options, key: value}) for key, value in (
                ("device", "cpu"), ("backend", "dense"), ("maximum_bytes", 1),
                ("maximum_channel_bytes", 1073741825), ("maximum_retained_bytes", 4294967297),
                ("final_path", f.root / "measurement.pt"), ("updates", 4))]
            for operation, chosen in invalid:
                with self.subTest(operation=operation), self.assertRaises(runner.AdmissionError):
                    runner.resolver(context, operation=operation, execution_options=chosen, **common)

    def test_actual_GPU_limits_and_StartTime_deadline_cannot_reset_after_checks(self):
        with fixture() as f:
            for limit in ("24:00:00", "1-00:00:00"):
                (f.run / "allocation.txt").write_text(allocation({"TimeLimit": limit}))
                context, budget = runner.admit()
                self.assertEqual(context.allocation["minutes"], 1440)
                self.assertEqual(budget.remaining_seconds(), 84480)
            self.assertEqual(f.torch.cuda.fractions[-1], (20 / 32, 0))
            self.assertFalse(f.torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(f.torch.backends.cudnn.allow_tf32)
            for change in ({"NumCPUs": "3"}, {"MinMemoryNode": "7G"}, {"Partition": "russpold"},
                           {"TimeLimit": "02:00:00"}, {"AllocTRES": "cpu=2,mem=16G,node=1,gres/gpu=2"},
                           {"NodeList": "other-node"}, {"JobId": "124"},
                           {"StartTime": "2033-05-18T04:33:20-07:00"}):
                (f.run / "allocation.txt").write_text(allocation(change))
                with self.assertRaises(runner.AdmissionError), mock.patch.object(runner, "_initialize_cuda") as cuda:
                    runner.admit()
                cuda.assert_not_called()
            (f.run / "allocation.txt").write_text(allocation())
            for key, value in (("CUDA_VISIBLE_DEVICES", "0,1"), ("OMP_NUM_THREADS", "2"),
                               ("SLURM_GPUS_ON_NODE", "2"), ("SLURM_CPUS_PER_TASK", "3")):
                with mock.patch.dict(os.environ, {key: value}), self.assertRaises(runner.AdmissionError):
                    runner.admit()
            for key, value in (("count", 2), ("total", 22 * 1024**3 - 1)):
                with mock.patch.object(f.torch.cuda, key, value), self.assertRaises(runner.AdmissionError):
                    runner.admit()
            with mock.patch.object(f.torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                    total_memory=32 * 1024**3, name="Tesla V100-PCIE-32GB")), \
                    self.assertRaisesRegex(runner.AdmissionError, "measured_V100"):
                runner.admit()
            context, budget = runner.admit()
            for counter in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
                with mock.patch.object(f.torch.cuda, counter, 20 * 1024**3 + 1), self.assertRaises(MemoryError):
                    budget.check()
            with mock.patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=15 * 1024**2 + 1)), \
                    self.assertRaises(MemoryError):
                budget.check()
            with mock.patch.object(runner.time, "time", return_value=NOW + 600), \
                    mock.patch.object(runner.time, "monotonic", return_value=1600.):
                _, later_budget = runner.admit()
                self.assertEqual(later_budget.deadline, budget.deadline)
                self.assertEqual(later_budget.absolute_deadline, budget.absolute_deadline)
                with redirect_stdout(io.StringIO()) as output, mock.patch.object(runner, "_load_operations") as load:
                    self.assertEqual(runner.main(["--remaining-seconds"]), 0)
                    self.assertEqual(output.getvalue(), "83880\n")
                    load.assert_not_called()
                    self.assertFalse((f.run / "claim.json").exists())
            with mock.patch.object(runner.time, "time", return_value=budget.absolute_deadline), self.assertRaises(TimeoutError):
                budget.check()
            end = datetime.fromtimestamp(NOW + 120, timezone.utc).replace(tzinfo=None).isoformat()
            (f.run / "allocation.txt").write_text(allocation({"EndTime": end}))
            self.assertEqual(runner.admit()[0].allocation["deadline_epoch"], NOW + 60)
            with mock.patch.dict(os.environ, {"TMPDIR": str(f.scratch.parent)}), self.assertRaises(runner.AdmissionError):
                runner._scratch()

    def test_full_pair_composition_exclusive_outputs_and_count_only_progress(self):
        for outcome in ("complete", "wrong_pair", "incomplete_fit", "fit_failure", "publication_failure", "closeout_failure"):
            with self.subTest(outcome=outcome), fixture() as f:
                context, budget = runner.admit()
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
                path = f.run / "final_content_pair.pt"
                def bind(actual_context, **kwargs):
                    self.assertIs(actual_context, context)
                    self.assertIs(kwargs["budget"], budget)
                    self.assertIs(kwargs["require_admitted_context"], runner.resolver)
                    self.assertEqual((kwargs["source"], kwargs["target"]), ("sub-03", "sub-01"))
                    self.assertTrue((f.run / "claim.json").is_file())
                    return bound
                def train(actual_bound, *, on_iteration, **kwargs):
                    self.assertIs(actual_bound, bound)
                    self.assertEqual(kwargs, {"device": "cuda:0", "block_units": 256, "row_chunk_size": 512,
                        "backend": "sparse", "final_path": path, "maximum_bytes": 1073741824})
                    print("sensitive_printed_loss")
                    print("sensitive_printed_loss", file=sys.stderr)
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
                with mock.patch.object(runner, "admit", return_value=(context, budget)), \
                        mock.patch.object(runner, "_load_operations", return_value=(bind, mock.Mock(side_effect=train),
                            {"python": "fixture", "numpy": "fixture", "torch": "fixture"})) as loader:
                    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                        self.assertEqual(runner.run_once(), 0 if outcome == "complete" else 1)
                    report_text = (f.run / "report.json").read_text()
                    progress_text = (f.run / "progress.jsonl").read_text()
                    self.assertNotIn("sensitive", report_text + progress_text + out.getvalue() + err.getvalue())
                    self.assertNotIn("training_loss", report_text + progress_text)
                    report = json.loads(report_text)
                    records = [json.loads(line) for line in progress_text.splitlines()]
                    observed = 0 if outcome == "wrong_pair" else 2 if outcome in ("fit_failure", "incomplete_fit") else 1024
                    self.assertEqual(report["observed_completed_updates"], observed)
                    self.assertEqual([record["completed_update"] for record in records], list(range(1, observed + 1)))
                    self.assertTrue(all(set(record) == {"completed_update", "elapsed_wall_seconds"} for record in records))
                    self.assertLessEqual((f.run / "progress.jsonl").stat().st_size, 262144)
                    published = outcome in ("complete", "closeout_failure")
                    self.assertEqual(report["completed_updates"], 1024 if published else None)
                    self.assertEqual(report["model_published"], True if published else False if outcome == "wrong_pair" else None)
                    self.assertEqual(report["published_path"], str(path) if published else None)
                    self.assertEqual(loader.return_value[1].call_count, 0 if outcome == "wrong_pair" else 1)
                    if outcome == "publication_failure":
                        self.assertEqual(path.with_name(path.name + ".partial").read_bytes(), b"fixture_partial")
                    with self.assertRaises((runner.AdmissionError, FileExistsError)):
                        runner.run_once()
                    self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
