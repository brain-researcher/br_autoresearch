"""Three stdlib mocked admission checks; no CUDA, payloads or scientific fits."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from itertools import permutations
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
import run_content_cuda_remaining as runner
from test_content_cuda_admission import fake_torch


NOW = 2_000_000_000.


def profile(stage):
    return {**runner._fixed(stage), "authorized": True,
        "approval_source": "fixture_only_not_empirical_authority",
        "resources": dict(runner.RESOURCES), "input_scope": runner._input_scope(stage)}


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None).isoformat()


def allocation(changes=None):
    fields = {"JobId": "123", "JobState": "RUNNING", "NodeList": "sh03-12n01",
        "NumCPUs": "2", "NumTasks": "1", "NumNodes": "1", "Partition": "gpu",
        "TimeLimit": "2-00:00:00", "MinMemoryNode": "16G",
        "AllocTRES": "cpu=2,mem=16G,node=1,gres/gpu=1,gres/gpu:fixture=1",
        "StartTime": stamp(NOW - 120), "EndTime": stamp(NOW + 172680)}
    fields.update(changes or {})
    return " ".join(f"{key}={value}" for key, value in fields.items())


@contextmanager
def fixture(name):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary).resolve()
        stack.enter_context(mock.patch.object(runner, "BASE", root))
        stage = runner._stage(name)
        stage.run_dir.mkdir(parents=True)
        stage.profile_path.write_text(json.dumps(profile(stage)))
        (stage.run_dir / "allocation.txt").write_text(allocation())
        local = root / "local"
        scratch = local / stage.scratch_name
        scratch.mkdir(parents=True)
        # Other stages and all four submitted source03 paths stay untouched.
        histories = []
        for other in runner.STAGES.values():
            if other is stage:
                continue
            other.run_dir.mkdir(parents=True)
            path = other.run_dir / "claim.json"
            path.write_bytes(b"historical_other_stage")
            histories.append(path)
        old = root / "outputs/runs/content_pair_training/cuda001/claim.json"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"historical_pair4")
        histories.append(old)
        for target in ("05", "06", "07"):
            old = root / f"outputs/runs/content_pair_training/sub03_to_sub{target}_cuda001/claim.json"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"historical_source03_roster")
            histories.append(old)
        snapshots = {path: path.read_bytes() for path in histories}
        env = {**{key: "1" for key in runner.THREADS}, "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "16384", "SLURM_GPUS": "1",
            "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_GPUS": "0", "CUDA_VISIBLE_DEVICES": "0",
            "L_SCRATCH_JOB": str(local), "TMPDIR": str(scratch)}
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
        yield SimpleNamespace(root=root, stage=stage, run=stage.run_dir, grant=stage.profile_path,
                              scratch=scratch, torch=torch)
        if any(path.read_bytes() != before for path, before in snapshots.items()):
            raise AssertionError("another stage's history changed")


class ContentCudaRemainingAdmissionTests(unittest.TestCase):
    def test_fixed_pairs_deny_changed_profiles_before_allocation_imports_and_cross_stage_options(self):
        widths = {"sub-01": 12294, "sub-03": 12897, "sub-05": 10891, "sub-06": 12023, "sub-07": 12057}
        expected = {f"remaining-{i:03d}": (source, target, i, widths[source], widths[target])
                    for i, (source, target) in enumerate(permutations(tuple(widths), 2)) if source != "sub-03"}
        self.assertEqual(len(expected), 16)
        self.assertEqual(set(runner.STAGES), set(expected))
        self.assertEqual(runner.RESOURCES, {
            "cpus": 2, "memory_gib": 16, "minutes": 2880, "gpus": 1,
            "deadline_seconds": 169200, "allocated_cpu_hours": 96, "allocated_gpu_hours": 48,
            "rss_limit_bytes": 16106127360, "minimum_device_bytes": 23622320128,
            "gpu_memory_limit_bytes": 21474836480, "durable_bytes": 2147483648,
            "scratch_bytes": 1073741824, "final_artifact_bytes": 1073741824})
        self.assertEqual(runner.FIXED["schema_version"], "ep21.content_cuda_remaining_training.v1")
        self.assertEqual(runner.OPERATION, "train_and_publish_cached_cuda_remaining_pair")
        for name, (source, target, index, source_width, width) in expected.items():
            stage = runner._stage(name)
            self.assertEqual((stage.source, stage.target, stage.pair_index, stage.source_width, stage.target_width),
                             (source, target, index, source_width, width))
            self.assertEqual(stage.profile_path, runner.BASE / f"outputs/content_cuda_remaining_{index:03d}_authorization.json")
            self.assertEqual(stage.run_dir, runner.BASE /
                             f"outputs/runs/content_pair_training/sub{source[-2:]}_to_sub{target[-2:]}_cuda001")
            self.assertEqual(stage.scratch_name, f"ep21_content_cuda_remaining{index:03d}")
            self.assertEqual(runner._input_scope(stage), {
                "normalization_participants": list(widths), "source_training": source,
                "target_decoder": target, "source_features": source,
                "target_response_matrix": False, "tau": False, "OOD": False})
            with self.subTest(stage=name), fixture(name) as f, \
                    mock.patch.object(runner, "_allocation") as actual, \
                    mock.patch.object(runner, "_initialize_cuda") as cuda, \
                    mock.patch.object(runner, "_load_operations") as load:
                changes = [{"authorized": False}, {"stage": "remaining-004"}, {"pair_index": 4},
                    {"source": "sub-03"}, {"source_eligible_voxels": source_width - 1},
                    {"target": source}, {"target_eligible_voxels": width - 1}, {"iterations": 4},
                    {"operation": "train_and_publish_cached_cuda_roster_pair"},
                    {"operation": "train_and_publish_cached_cuda_full_pair"},
                    {"operation": "measure_content_pair_cuda_cached_four_updates"}, {"tf32": True},
                    {"maximum_retained_bytes": 4294967297},
                    {"input_scope": {**runner._input_scope(f.stage), "tau": True}},
                    {"input_scope": {**runner._input_scope(f.stage), "source_features": "sub-03"}},
                    {"input_scope": {**runner._input_scope(f.stage), "source_training": "sub-03"}},
                    {"input_scope": {**runner._input_scope(f.stage), "target_response_matrix": True}}]
                changes += [{"resources": {**runner.RESOURCES, key: value}} for key, value in
                    (("cpus", 3), ("minutes", 1440), ("deadline_seconds", 172800),
                     ("allocated_cpu_hours", 95), ("allocated_gpu_hours", None), ("gpus", True),
                     ("rss_limit_bytes", 16 * 1024**3), ("final_artifact_bytes", 1))]
                for change in changes:
                    f.grant.write_text(json.dumps({**profile(f.stage), **change}))
                    with self.assertRaises(runner.AdmissionError):
                        runner.run_once(name)
                with self.assertRaises(runner.AdmissionError):
                    runner.run_once("remaining-005")
                actual.assert_not_called()
                cuda.assert_not_called()
                load.assert_not_called()
                self.assertFalse((f.run / "claim.json").exists())
            with fixture(name) as f:
                context, budget = runner.admit(name)
                common = dict(source=source, target=target, budget=budget)
                options = runner._execution_options(f.stage)
                self.assertEqual(len(options), 9)
                runner.resolver(context, operation="bind_content_inputs", execution_options=None, **common)
                runner.resolver(context, operation=runner.OPERATION, execution_options=options, **common)
                for other in runner.STAGES.values():
                    if other is f.stage:
                        continue
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, operation=runner.OPERATION, execution_options=options,
                            **{**common, "source": other.source, "target": other.target})
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, operation=runner.OPERATION,
                            execution_options=runner._execution_options(other), **common)
                for operation in ("train_and_publish_cached_cuda_full_pair", "train_and_publish_cached_cuda_roster_pair",
                                  "measure_content_pair_cuda_cached_four_updates", "score"):
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, operation=operation, execution_options=options, **common)
                for key, value in (("maximum_bytes", 1), ("device", "cpu"), ("backend", "dense"),
                                   ("maximum_retained_bytes", 4294967297), ("updates", 4)):
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, operation=runner.OPERATION,
                            execution_options={**options, key: value}, **common)
                f.grant.write_text(json.dumps({**profile(f.stage), "authorized": False}))
                with self.assertRaises(runner.AdmissionError):
                    runner.resolver(context, operation="bind_content_inputs", execution_options=None, **common)

    def test_actual_GPU_and_host_caps_and_absolute_StartTime_survive_new_process_admission(self):
        for name in runner.STAGES:
            with self.subTest(stage=name), fixture(name) as f:
                context, budget = runner.admit(name)
                self.assertEqual(budget.remaining_seconds(), 169080)
                self.assertEqual(f.torch.cuda.fractions[-1], (20 / 32, 0))
                self.assertFalse(f.torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(f.torch.backends.cudnn.allow_tf32)
                for change in ({"NumCPUs": "3"}, {"MinMemoryNode": "7G"}, {"Partition": "russpold"},
                               {"TimeLimit": "1-00:00:00"}, {"JobId": "124"}, {"NodeList": "other"},
                               {"AllocTRES": "cpu=2,mem=16G,node=1,gres/gpu=2"}):
                    (f.run / "allocation.txt").write_text(allocation(change))
                    with mock.patch.object(runner, "_initialize_cuda") as cuda, self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                    cuda.assert_not_called()
                (f.run / "allocation.txt").write_text(allocation())
                for key, value in (("CUDA_VISIBLE_DEVICES", "0,1"), ("SLURM_GPUS_ON_NODE", "2"),
                                   ("SLURM_CPUS_PER_TASK", "3"), ("OMP_NUM_THREADS", "2")):
                    with mock.patch.dict(os.environ, {key: value}), self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                for key, value in (("count", 2), ("total", 22 * 1024**3 - 1)):
                    with mock.patch.object(f.torch.cuda, key, value), self.assertRaises(runner.AdmissionError):
                        runner.admit(name)
                with mock.patch.object(f.torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                        total_memory=32 * 1024**3, name="Tesla V100-PCIE-32GB")), self.assertRaises(runner.AdmissionError):
                    runner.admit(name)
                for counter in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
                    with mock.patch.object(f.torch.cuda, counter, 20 * 1024**3 + 1), self.assertRaises(MemoryError):
                        budget.check()
                with mock.patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=15 * 1024**2 + 1)), \
                        self.assertRaises(MemoryError):
                    budget.check()
                with mock.patch.object(runner.time, "time", return_value=NOW + 600), \
                        mock.patch.object(runner.time, "monotonic", return_value=1600.):
                    _, later = runner.admit(name)
                    self.assertEqual(later.deadline, budget.deadline)
                    self.assertEqual(later.absolute_deadline, budget.absolute_deadline)
                    with redirect_stdout(io.StringIO()) as out, mock.patch.object(runner, "_load_operations") as load:
                        self.assertEqual(runner.main(["--stage", name, "--remaining-seconds"]), 0)
                        self.assertEqual(out.getvalue(), "168480\n")
                        load.assert_not_called()
                        self.assertFalse((f.run / "claim.json").exists())
                with mock.patch.object(runner.time, "time", return_value=budget.absolute_deadline), self.assertRaises(TimeoutError):
                    budget.check()
                (f.run / "allocation.txt").write_text(allocation({"EndTime": stamp(NOW + 120)}))
                self.assertEqual(runner.admit(name)[0].allocation["deadline_epoch"], NOW + 60)
                with mock.patch.dict(os.environ, {"TMPDIR": str(f.scratch.parent)}), self.assertRaises(runner.AdmissionError):
                    runner._scratch(f.stage)

    def test_full_pair_dispatch_publication_failure_and_private_values_are_not_reported(self):
        for name in runner.STAGES:
            for outcome in ("complete", "wrong_pair", "incomplete_fit", "fit_failure", "publication_failure", "closeout_failure"):
                with self.subTest(stage=name, outcome=outcome), fixture(name) as f:
                    context, budget = runner.admit(name)
                    stage = f.stage
                    if outcome == "closeout_failure":
                        budget.check_storage = mock.Mock(side_effect=OSError("sensitive_closeout"))
                    source = SimpleNamespace(axis=SimpleNamespace(participant=stage.source, eligible_native_ids=range(stage.source_width)),
                        image_ids=range(4712), normalized_X=SimpleNamespace(shape=(4712, stage.source_width)))
                    initial = SimpleNamespace(pair_index=4 if outcome == "wrong_pair" else stage.pair_index,
                        source=stage.source, target=stage.target, input_width=stage.source_width,
                        hidden_width=stage.source_width // 2, output_width=stage.target_width)
                    panel = SimpleNamespace(axes=range(5), initializations=[initial] * 20,
                        channel_schedule=SimpleNamespace(channels_by_layer=[range(1024)] * 19))
                    bound = SimpleNamespace(source=source, target=stage.target, panel=panel,
                        pair=SimpleNamespace(target_axis=SimpleNamespace(eligible_native_ids=range(stage.target_width))))
                    def bind(actual, **kwargs):
                        self.assertIs(actual, context)
                        self.assertIs(kwargs["budget"], budget)
                        self.assertIs(kwargs["require_admitted_context"], runner.resolver)
                        self.assertEqual((kwargs["source"], kwargs["target"]), (stage.source, stage.target))
                        self.assertTrue((f.run / "claim.json").is_file())
                        return bound
                    def train(actual, *, on_iteration, **kwargs):
                        self.assertIs(actual, bound)
                        self.assertEqual(kwargs, {"stage": name, **runner._training_options(stage)})
                        print("sensitive_printed_loss")
                        print("sensitive_printed_loss", file=sys.stderr)
                        count = 2 if outcome in ("fit_failure", "incomplete_fit") else 1024
                        for completed in range(1, count + 1):
                            on_iteration({"completed_iterations": completed, "training_loss": "sensitive_loss"})
                        if outcome == "fit_failure":
                            raise RuntimeError("sensitive_exception")
                        if outcome == "publication_failure":
                            stage.final_path.with_name(stage.final_path.name + ".partial").write_bytes(b"partial")
                            raise RuntimeError("sensitive_publication")
                        stage.final_path.write_bytes(b"fixture_model")
                        return SimpleNamespace(source=stage.source, target=stage.target, pair_index=stage.pair_index,
                            path=stage.final_path, completion=SimpleNamespace(rows=4712, backend="sparse",
                            iterations=1023 if outcome == "incomplete_fit" else 1024, private_loss="sensitive"))
                    fit = mock.Mock(side_effect=train)
                    with mock.patch.object(runner, "admit", return_value=(context, budget)), \
                            mock.patch.object(runner, "_load_operations", return_value=(bind, fit,
                                {"python": "fixture", "numpy": "fixture", "torch": "fixture"})) as loader:
                        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                            self.assertEqual(runner.run_once(name), 0 if outcome == "complete" else 1)
                        text = (f.run / "report.json").read_text()
                        progress = (f.run / "progress.jsonl").read_text()
                        self.assertNotIn("sensitive", text + progress + out.getvalue() + err.getvalue())
                        self.assertNotIn("training_loss", text + progress)
                        report = json.loads(text)
                        records = [json.loads(line) for line in progress.splitlines()]
                        observed = 0 if outcome == "wrong_pair" else 2 if outcome in ("fit_failure", "incomplete_fit") else 1024
                        self.assertEqual(report["observed_completed_updates"], observed)
                        self.assertEqual((report["stage"], report["source"], report["target"], report["pair_index"]),
                                         (name, stage.source, stage.target, stage.pair_index))
                        self.assertEqual([r["completed_update"] for r in records], list(range(1, observed + 1)))
                        self.assertTrue(all(set(r) == {"completed_update", "elapsed_wall_seconds"} for r in records))
                        self.assertLessEqual((f.run / "progress.jsonl").stat().st_size, 262144)
                        published = outcome in ("complete", "closeout_failure")
                        self.assertEqual(report["completed_updates"], 1024 if published else None)
                        self.assertEqual(report["model_published"], True if published else False if outcome == "wrong_pair" else None)
                        self.assertEqual(report["published_path"], str(stage.final_path) if published else None)
                        self.assertEqual(fit.call_count, 0 if outcome == "wrong_pair" else 1)
                        if outcome == "publication_failure":
                            self.assertEqual(stage.final_path.with_name(stage.final_path.name + ".partial").read_bytes(), b"partial")
                        with self.assertRaises((runner.AdmissionError, FileExistsError)):
                            runner.run_once(name)
                        self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
