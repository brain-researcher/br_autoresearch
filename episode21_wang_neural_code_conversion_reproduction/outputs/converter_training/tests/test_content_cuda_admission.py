"""Three stdlib-only admission/control-flow fixtures; Slurm execution only.

CUDA is an explicit stub here. These checks do not qualify GPU arithmetic or
replace the separate real-device loss/adjoint fixture required by the batch.
"""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import wraps
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
import run_content_cuda_measurement as entry


def profile(stage_name="cuda001"):
    return {**entry._stage(stage_name).fixed, "authorized": True, "approval_source": "fixture_separate_CUDA_approval",
            "resources": dict(entry.RESOURCES), "input_scope": dict(entry.INPUT_SCOPE)}


def allocation():
    end = datetime.fromtimestamp(2_000_007_000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    start = datetime.fromtimestamp(1_999_999_880, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return ("JobId=123 JobState=RUNNING NodeList=sh03-12n01 NumCPUs=2 NumTasks=1 NumNodes=1 "
        "Partition=gpu TimeLimit=02:00:00 MinMemoryNode=16G "
        "AllocTRES=cpu=2,mem=16G,node=1,gres/gpu=1,gres/gpu:fixture=1 StartTime=" + start + " EndTime=" + end + "\n")


class FakeCuda:
    def __init__(self):
        self.available, self.count, self.current = True, 1, 0
        self.total = 24 * 1024**3
        self.allocated = self.reserved = self.peak_allocated = self.peak_reserved = 0
        self.fractions = []

    def is_available(self): return self.available
    def device_count(self): return self.count
    def set_device(self, device): self.current = device
    def current_device(self): return self.current
    def get_device_properties(self, device):
        return SimpleNamespace(total_memory=self.total, name="fixture CUDA device")
    def set_per_process_memory_fraction(self, fraction, device): self.fractions.append((fraction, device))
    def memory_allocated(self, device): return self.allocated
    def memory_reserved(self, device): return self.reserved
    def max_memory_allocated(self, device): return self.peak_allocated
    def max_memory_reserved(self, device): return self.peak_reserved


def fake_torch():
    calls = []
    return SimpleNamespace(cuda=FakeCuda(), calls=calls,
        set_num_threads=lambda value: calls.append(("threads", value)),
        set_num_interop_threads=lambda value: calls.append(("interop", value)),
        set_float32_matmul_precision=lambda value: calls.append(("precision", value)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                 cudnn=SimpleNamespace(allow_tf32=True)),
        is_autocast_enabled=lambda: False)


@contextmanager
def fixture(stage_name="cuda001"):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        original, cached, repaired, local = root / "original", root / "cached", root / "repaired", root / "local"
        original.mkdir()
        cached.mkdir()
        repaired.mkdir()
        local.mkdir()
        for name, value in (("RUN_DIR", original), ("PROFILE", root / "original.json"),
                            ("CACHE_RUN_DIR", cached), ("CACHE_PROFILE", root / "cached.json"),
                            ("CACHE_REPAIR_RUN_DIR", repaired), ("CACHE_REPAIR_PROFILE", root / "repaired.json")):
            stack.enter_context(mock.patch.object(entry, name, value))
        for name in entry.STAGES:
            stage = entry._stage(name)
            stage.profile_path.write_text(json.dumps(profile(name)))
        stage = entry._stage(stage_name)
        other = entry._stage("cache-001" if stage_name == "cuda001" else "cuda001")
        histories = []
        for name in entry.STAGES:
            if name != stage_name:
                previous = entry._stage(name)
                history = previous.run_dir / "historical.txt"
                history.write_text("preserve completed or selected other stage " + name)
                histories.append((previous, history, history.read_bytes(), previous.profile_path.read_bytes()))
        run, proposal = stage.run_dir, stage.profile_path
        scratch = local / stage.scratch_name
        scratch.mkdir()
        (run / "allocation.txt").write_text(allocation())
        environment = {**{key: "1" for key in entry.THREADS}, "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "16384",
            "SLURM_GPUS": "1", "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_GPUS": "0",
            "CUDA_VISIBLE_DEVICES": "0", "L_SCRATCH_JOB": str(local), "TMPDIR": str(scratch)}
        stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
        stack.enter_context(mock.patch.object(entry.socket, "gethostname", return_value="sh03-12n01.int"))
        stack.enter_context(mock.patch.object(entry.time, "time", return_value=2_000_000_000.))
        stack.enter_context(mock.patch.object(entry.time, "monotonic", return_value=1000.))
        stack.enter_context(mock.patch.object(entry.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1024)))
        torch = fake_torch()
        stack.enter_context(mock.patch.dict(sys.modules, {"torch": torch}))
        yield SimpleNamespace(run=run, proposal=proposal, scratch=scratch, torch=torch,
            stage=stage, other=other, histories=histories, argv=["--stage", stage_name])


def options(run, stage_name="cuda001"):
    value = {"device": "cuda:0", "block_units": 256, "row_chunk_size": 512, "backend": "sparse",
        "updates": 4, "source_rows": 4712, "layer_count": 19,
        "final_path": run / "reserved_final_model.pt", "maximum_bytes": 1}
    if entry._stage(stage_name).cached:
        value.update(feature_read_mode="bounded_channel_two_pass_cache",
                     maximum_channel_bytes=1073741824, maximum_retained_bytes=4294967296)
    return value


def all_stages(test):
    @wraps(test)
    def parameterized(self):
        for stage_name in entry.STAGES:
            with self.subTest(stage=stage_name):
                test(self, stage_name)
    return parameterized


class CUDAAdmissionTests(unittest.TestCase):
    @all_stages
    def test_inactive_or_changed_grant_refuses_before_CUDA_allocation_or_payload(self, stage_name):
        self.assertEqual(entry._stage().name, "cuda001")
        cached = entry._stage("cache-001")
        self.assertEqual(cached.profile_path, entry.BASE / "outputs/content_cuda_cache_authorization.json")
        self.assertEqual(cached.run_dir, entry.BASE / "outputs/runs/content_resource_measurement/cuda_cache001")
        self.assertEqual(cached.scratch_name, "ep21_content_cuda_cache001")
        repaired = entry._stage("cache-002")
        self.assertEqual(repaired.profile_path, entry.BASE / "outputs/content_cuda_cache_repair_authorization.json")
        self.assertEqual(repaired.run_dir, entry.BASE / "outputs/runs/content_resource_measurement/cuda_cache002")
        self.assertEqual(repaired.scratch_name, "ep21_content_cuda_cache002")
        self.assertEqual(repaired.fixed, cached.fixed)
        with fixture(stage_name) as f, mock.patch.object(entry, "_initialize_cuda") as cuda, \
                mock.patch.object(entry, "_load_operations") as operations:
            valid = profile(stage_name)
            cases = [{**valid, "authorized": False},
                     {**valid, "device": "cpu"},
                     {**valid, "input_scope": {**entry.INPUT_SCOPE, "tau": True}},
                     {**valid, "resources": {**entry.RESOURCES, "gpus": True}},
                     {**valid, "feature_read_mode": "unknown_cache"}]
            if f.stage.cached:
                cases.extend(({**valid, "maximum_channel_bytes": 1073741825},
                              {**valid, "maximum_retained_bytes": 4294967297},
                              {**valid, "operation": entry.FIXED["operation"]}))
            else:
                cases.append({**valid, "feature_stripe_bytes": 67108865})
            for value in cases:
                f.proposal.write_text(json.dumps(value))
                with mock.patch.object(entry, "_allocation") as actual:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(entry.main(f.argv), 2)
                    actual.assert_not_called()
            cuda.assert_not_called()
            operations.assert_not_called()
            self.assertFalse((f.run / "claim.json").exists())
            self.assertFalse((f.run / "report.json").exists())
        with fixture(stage_name) as f, mock.patch.object(entry, "_load_operations") as operations:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(entry.main(f.argv + ["--check-admission"]), 0)
            report = json.loads(output.getvalue())
            self.assertFalse(report["payload_read"])
            self.assertFalse(report["attempt_claimed"])
            self.assertEqual(report["stage"], stage_name)
            self.assertEqual(report["device"]["visible_device_count"], 1)
            self.assertEqual(f.torch.cuda.fractions, [(20 / 24, 0)])
            self.assertEqual(f.torch.calls, [("threads", 1), ("interop", 1), ("precision", "highest")])
            self.assertFalse(f.torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(f.torch.backends.cudnn.allow_tf32)
            operations.assert_not_called()
            self.assertFalse((f.run / "claim.json").exists())

        # Stub only the deferred imports: the selected wrapper must actually
        # deliver the fixed mode to the core, while the old call stays unchanged.
        bind, measure = mock.Mock(), mock.Mock()
        modules = {"empirical_content_inputs": SimpleNamespace(bind_content_inputs=bind),
                   "content_pair_cuda_measurement": SimpleNamespace(measure_content_pair_cuda=measure),
                   "decoder_interface": SimpleNamespace(FEATURE_STRIPE_BYTES=67108864)}
        with mock.patch.dict(sys.modules, modules):
            loaded_bind, loaded_measure = entry._load_operations(entry._stage(stage_name))
            self.assertIs(loaded_bind, bind)
            loaded_measure("fixture_context", reserved_final_path=Path("/fixture.pt"))
            expected = {"reserved_final_path": Path("/fixture.pt")}
            if entry._stage(stage_name).cached:
                expected["feature_read_mode"] = "bounded_channel_two_pass_cache"
            measure.assert_called_once_with("fixture_context", **expected)

    @all_stages
    def test_actual_GPU_allocation_device_and_budget_match_the_fixed_operation(self, stage_name):
        with fixture(stage_name) as f:
            for memory in ("16G", "16384M", "16384"):
                (f.run / "allocation.txt").write_text(allocation().replace("MinMemoryNode=16G", "MinMemoryNode=" + memory))
                context, budget = entry.admit(stage_name)
                self.assertEqual(budget.deadline, 7180. if f.stage.allocation_start_deadline else 7300.)
                self.assertEqual(context.allocation["gpus"], 1)
            for old, new in (("JobId=123", "JobId=124"), ("NumCPUs=2", "NumCPUs=3"),
                    ("Partition=gpu", "Partition=russpold"), ("MinMemoryNode=16G", "MinMemoryNode=7G"),
                    ("gres/gpu=1", "gres/gpu=2"), ("gres/gpu:fixture=1", "gres/gpu:fixture=2"),
                    ("TimeLimit=02:00:00", "TimeLimit=01:00:00"),
                    ("NodeList=sh03-12n01", "NodeList=sh04-ln04")):
                (f.run / "allocation.txt").write_text(allocation().replace(old, new))
                with self.subTest(defect=new), mock.patch.object(entry, "_initialize_cuda") as cuda:
                    with self.assertRaises(entry.AdmissionError):
                        entry.admit(stage_name)
                    cuda.assert_not_called()
            (f.run / "allocation.txt").write_text(allocation())
            for name, wrong in (("CUDA_VISIBLE_DEVICES", "0,1"), ("SLURM_GPUS_ON_NODE", "2"),
                                ("SLURM_CPUS_PER_TASK", "3"), ("OMP_NUM_THREADS", "2")):
                with mock.patch.dict(os.environ, {name: wrong}), mock.patch.object(entry, "_initialize_cuda") as cuda:
                    with self.assertRaises(entry.AdmissionError):
                        entry.admit(stage_name)
                    cuda.assert_not_called()
            if f.stage.allocation_start_deadline:
                original_context, original_budget = entry.admit(stage_name)
                with mock.patch.object(entry.time, "time", return_value=2_000_001_000.), \
                        mock.patch.object(entry.time, "monotonic", return_value=2000.):
                    later_context, later_budget = entry.admit(stage_name)
                    self.assertEqual(later_budget.deadline, original_budget.deadline)
                    self.assertEqual(later_budget.absolute_deadline, 2_000_006_180.)
                    self.assertEqual(later_budget.remaining_seconds(), 5180)
                    with redirect_stdout(io.StringIO()) as out, mock.patch.object(entry, "_load_operations") as load:
                        self.assertEqual(entry.main(f.argv + ["--remaining-seconds"]), 0)
                        self.assertEqual(out.getvalue(), "5180\n")
                        load.assert_not_called()
                        self.assertFalse((f.run / "claim.json").exists())
                for changed_time in (1_999_999_879., 2_000_006_180.):
                    with mock.patch.object(entry.time, "time", return_value=changed_time), \
                            self.assertRaises(entry.AdmissionError):
                        entry.admit(stage_name)
                end = datetime.fromtimestamp(2_000_000_120, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                record = allocation().rsplit(" EndTime=", 1)[0] + " EndTime=" + end + "\n"
                (f.run / "allocation.txt").write_text(record)
                self.assertEqual(entry.admit(stage_name)[0].allocation["deadline_epoch"], 2_000_000_060.)
                (f.run / "allocation.txt").write_text(allocation())
            else:
                with mock.patch.object(entry.time, "time", return_value=2_000_001_000.):
                    with self.assertRaisesRegex(entry.AdmissionError, "remaining"):
                        entry.admit(stage_name)
            for name, wrong in (("count", 2), ("total", entry.RESOURCES["minimum_device_bytes"] - 1)):
                with mock.patch.object(f.torch.cuda, name, wrong):
                    with self.assertRaises(entry.AdmissionError):
                        entry.admit(stage_name)
            context, budget = entry.admit(stage_name)
            common = dict(source=entry.SOURCE, target=entry.TARGET, budget=budget)
            self.assertIsNone(entry.resolver(context, operation="bind_content_inputs", execution_options=None, **common))
            operation = f.stage.fixed["operation"]
            chosen = options(f.run, stage_name)
            self.assertIsNone(entry.resolver(context, operation=operation, execution_options=chosen, **common))
            invalid = [("train_and_publish_content_pair", chosen),
                    (f.other.fixed["operation"], chosen),
                    (operation, {**chosen, "device": "cpu"}),
                    (operation, {**chosen, "updates": 1024}),
                    ("bind_content_inputs", {})]
            invalid.extend((operation, {**chosen, "final_path": previous.run_dir / "reserved_final_model.pt"})
                           for previous, *_ in f.histories)
            if f.stage.cached:
                invalid.extend((operation, {**chosen, key: value}) for key, value in (
                    ("maximum_channel_bytes", 1073741825), ("maximum_retained_bytes", 4294967297),
                    ("feature_read_mode", "bounded_column_stripe")))
            for operation, chosen in invalid:
                with self.assertRaises(entry.AdmissionError):
                    entry.resolver(context, operation=operation, execution_options=chosen, **common)
            for counter in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
                with mock.patch.object(f.torch.cuda, counter, entry.RESOURCES["gpu_memory_limit_bytes"] + 1):
                    with self.assertRaises(MemoryError):
                        budget.check()
            with mock.patch.object(entry.time, "monotonic", return_value=budget.deadline):
                with self.assertRaises(TimeoutError):
                    budget.check()
            with mock.patch.object(entry.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=15 * 1024**2 + 1)):
                with self.assertRaises(MemoryError):
                    budget.check()
            for previous, *_ in f.histories:
                with mock.patch.dict(os.environ, {"TMPDIR": str(f.scratch.parent / previous.scratch_name)}), \
                        self.assertRaises(entry.AdmissionError):
                    entry._scratch(f.stage)
            f.proposal.write_text(json.dumps({**profile(stage_name), "authorized": False}))
            with self.assertRaises(entry.AdmissionError):
                entry.resolver(context, operation="bind_content_inputs", execution_options=None, **common)

    @all_stages
    def test_single_claim_redacts_outcomes_and_preserves_completed_counts_after_closeout_failure(self, stage_name):
        sentinel = "PRIVATE_LOSS_AND_RESPONSE_VALUE"
        with fixture(stage_name) as f:
            def fail(context, **kwargs):
                print(sentinel)
                print(sentinel, file=sys.stderr)
                raise RuntimeError(sentinel)
            with mock.patch.object(entry, "_load_operations", return_value=(mock.Mock(), fail)) as loader:
                with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(entry.main(f.argv), 1)
                text = (f.run / "report.json").read_text()
                self.assertNotIn(sentinel, text + out.getvalue() + err.getvalue())
                self.assertEqual(json.loads(text)["error_type"], "RuntimeError")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(entry.main(f.argv), 2)
                loader.assert_called_once()
                self.assertEqual((f.run / "report.json").read_text(), text)
                self.assertFalse((f.run / "reserved_final_model.pt").exists())
                for previous, history, history_before, grant_before in f.histories:
                    self.assertEqual(history.read_bytes(), history_before)
                    self.assertEqual(previous.profile_path.read_bytes(), grant_before)
                    self.assertFalse((previous.run_dir / "claim.json").exists())
        for closeout_failure in (False, True):
            with self.subTest(closeout_failure=closeout_failure), fixture(stage_name) as f:
                bound = object()
                def bind(context, **kwargs):
                    kwargs["require_admitted_context"](context, operation="bind_content_inputs",
                        source=kwargs["source"], target=kwargs["target"], budget=kwargs["budget"], execution_options=None)
                    return bound
                def measure(context, **kwargs):
                    kwargs["require_admitted_context"](context, operation=f.stage.fixed["operation"],
                        source=entry.SOURCE, target=entry.TARGET, budget=kwargs["budget"],
                        execution_options=options(f.run, stage_name))
                    self.assertIs(kwargs["prepare_inputs"](), bound)
                    self.assertEqual(kwargs["reserved_final_path"], f.run / "reserved_final_model.pt")
                    return {"status": "complete_measurement", "completed_updates": 4,
                            "production_fit_completed": False, "model_published": False}
                with mock.patch.object(entry, "_load_operations", return_value=(bind, measure)), \
                        mock.patch.object(entry.MeasurementBudget, "check_storage",
                            side_effect=[None, OSError("fixture storage failure") if closeout_failure else None]):
                    self.assertEqual(entry.run_once(stage_name), 1 if closeout_failure else 0)
                report = json.loads((f.run / "report.json").read_text())
                self.assertEqual(report["completed_updates"], 4)
                self.assertEqual(report["status"], "incomplete" if closeout_failure else "complete_measurement")
                self.assertFalse(report["model_published"])
                self.assertEqual(report["stage"], stage_name)
                if f.stage.cached:
                    self.assertEqual(report["feature_read_mode"], "bounded_channel_two_pass_cache")
                    self.assertEqual((report["maximum_channel_bytes"], report["maximum_retained_bytes"]),
                                     (1073741824, 4294967296))
                    self.assertNotIn("feature_stripe_bytes", report)


if __name__ == "__main__":
    unittest.main()
