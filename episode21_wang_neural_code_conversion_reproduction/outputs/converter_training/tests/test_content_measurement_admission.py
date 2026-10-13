"""Source-free admission fixtures; run only in coordinator-owned compute."""
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
import run_content_measurement as entry


def profile(stage="original"):
    return {**entry.FIXED, **dict(entry._stage(stage).extra_scope),
            "authorized": True, "approval_source": "fixture-only-separate-approval",
            "resources": dict(entry.RESOURCES), "input_scope": dict(entry.INPUT_SCOPE),
            "purpose": "descriptive metadata does not enlarge authority"}


def snapshot():
    end = datetime.fromtimestamp(2_000_007_000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return ("JobId=123 JobState=RUNNING NodeList=sh04-04n21 NumCPUs=3 NumTasks=1 NumNodes=1 "
            "Partition=russpold TimeLimit=02:00:00 MinMemoryNode=16G "
            "AllocTRES=cpu=3,mem=16G,node=1,billing=3 EndTime=" + end + "\n")


@contextmanager
def fixture(stage="original"):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        run, local = root / "run", root / "local"
        run.mkdir()
        local.mkdir()
        scratch = local / entry._stage(stage).scratch_name
        scratch.mkdir()
        proposal = root / "proposal.json"
        proposal.write_text(json.dumps(profile(stage)))
        (run / "allocation.txt").write_text(snapshot())
        names = {"original": ("RUN_DIR", "PROFILE"),
                 "column-stripe-001": ("STRIPE_RUN_DIR", "STRIPE_PROFILE"),
                 "column-stripe-002": ("STRIPE_REPAIR_RUN_DIR", "STRIPE_REPAIR_PROFILE")}[stage]
        for name, value in zip(names, (run, proposal)):
            stack.enter_context(mock.patch.object(entry, name, value))
        environment = {**{key: "1" for key in entry.THREADS}, "SLURM_JOB_ID": "123",
                       "SLURM_CPUS_PER_TASK": "3", "SLURM_MEM_PER_NODE": "16384",
                       "L_SCRATCH_JOB": str(local), "TMPDIR": str(scratch), "CUDA_VISIBLE_DEVICES": ""}
        stack.enter_context(mock.patch.dict(os.environ, environment, clear=True))
        stack.enter_context(mock.patch.object(entry.socket, "gethostname", return_value="sh04-04n21.int"))
        stack.enter_context(mock.patch.object(entry.time, "time", return_value=2_000_000_000))
        stack.enter_context(mock.patch.object(entry.time, "monotonic", return_value=1000.0))
        stack.enter_context(mock.patch.object(entry.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=1024)))
        yield SimpleNamespace(root=root, run=run, scratch=scratch, proposal=proposal)


def options(run):
    return {"device": "cpu", "block_units": 256, "row_chunk_size": 512, "backend": "sparse",
            "updates": 4, "source_rows": 4712, "layer_count": 19,
            "final_path": run / "reserved_final_model.pt", "maximum_bytes": 1}


class AdmissionTests(unittest.TestCase):
    def test_column_stripe_requires_its_own_scope_scratch_and_destination(self):
        for stage in ("column-stripe-001", "column-stripe-002"):
            with self.subTest(stage=stage), fixture(stage) as f:
                old_profile, old_run = entry.PROFILE, entry.RUN_DIR
                other = entry._stage("column-stripe-002" if stage.endswith("001") else "column-stripe-001")
                with mock.patch.object(entry, "_read_text", wraps=entry._read_text) as reads:
                    context, budget = entry.admit(stage=stage)
                self.assertEqual(context.stage.name, stage)
                self.assertEqual(context.stage.run_dir, f.run)
                self.assertEqual(context.scratch, f.scratch)
                self.assertNotEqual(context.stage.profile_path, other.profile_path)
                self.assertNotEqual(context.stage.run_dir, other.run_dir)
                self.assertNotEqual(context.stage.scratch_name, other.scratch_name)
                self.assertEqual([call.args[0] for call in reads.call_args_list],
                                 [f.proposal, f.run / "allocation.txt"])
                self.assertEqual((entry.PROFILE, entry.RUN_DIR), (old_profile, old_run))
                self.assertEqual(entry._stage().name, "original")
                self.assertEqual(context.profile["feature_read_mode"], "bounded_column_stripe")
                self.assertEqual(context.profile["feature_stripe_bytes"], 67108864)
                common = dict(source=entry.SOURCE, target=entry.TARGET, budget=budget)
                entry.resolver(context, operation=entry.FIXED["operation"],
                               execution_options=options(f.run), **common)
                for wrong_run in (old_run, other.run_dir):
                    with self.assertRaises(entry.AdmissionError):
                        entry.resolver(context, operation=entry.FIXED["operation"],
                                       execution_options=options(wrong_run), **common)

                # The old grant cannot authorize either stripe stage. Both
                # optimization fields are checked rather than descriptive.
                for replacement in (profile(),
                                    {**profile(stage), "feature_read_mode": "arbitrary"},
                                    {**profile(stage), "feature_stripe_bytes": 67108865},
                                    {**profile(stage), "feature_stripe_bytes": True}):
                    f.proposal.write_text(json.dumps(replacement))
                    with mock.patch.object(entry, "_allocation") as allocation:
                        with self.assertRaises(entry.AdmissionError):
                            entry.admit(stage=stage)
                        allocation.assert_not_called()
                f.proposal.write_text(json.dumps(profile(stage)))
                for wrong_scratch in ("ep21_content_measurement_attempt002", other.scratch_name):
                    with mock.patch.dict(os.environ, {"TMPDIR": str(f.scratch.parent / wrong_scratch)}):
                        with self.assertRaisesRegex(entry.AdmissionError, "scratch"):
                            entry.admit(stage=stage)

    def test_column_stripe_cli_claim_report_and_single_attempt_preserve_original_stage(self):
        for stage in ("column-stripe-001", "column-stripe-002"):
            with self.subTest(stage=stage), fixture(stage) as f:
                bound = object()

                def bind(context, **kwargs):
                    self.assertEqual(context.stage.name, stage)
                    kwargs["require_admitted_context"](context, operation="bind_content_inputs",
                        source=kwargs["source"], target=kwargs["target"], budget=kwargs["budget"],
                        execution_options=None)
                    return bound

                def measure(context, **kwargs):
                    kwargs["require_admitted_context"](context, operation=entry.FIXED["operation"],
                        source=entry.SOURCE, target=entry.TARGET, budget=kwargs["budget"],
                        execution_options=options(f.run))
                    self.assertIs(kwargs["prepare_inputs"](), bound)
                    self.assertEqual(kwargs["reserved_final_path"], f.run / "reserved_final_model.pt")
                    return {"status": "complete_measurement", "completed_updates": 4,
                            "model_published": False, "production_fit_completed": False}

                with mock.patch.object(entry, "_load_operations", return_value=(bind, measure)) as loader:
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(entry.main(["--stage", stage, "--check-admission"]), 0)
                    self.assertEqual(json.loads(output.getvalue())["stage"], stage)
                    self.assertFalse((f.run / "claim.json").exists())
                    loader.assert_not_called()
                    self.assertEqual(entry.main(["--stage", stage]), 0)
                    loader.assert_called_once()
                    saved_report = (f.run / "report.json").read_bytes()
                    for name in ("claim.json", "report.json"):
                        value = json.loads((f.run / name).read_text())
                        self.assertEqual(value["stage"], stage)
                        self.assertEqual(value["feature_read_mode"], "bounded_column_stripe")
                        self.assertEqual(value["feature_stripe_bytes"], 67108864)
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(entry.main(["--stage", stage]), 2)
                    loader.assert_called_once()
                    self.assertEqual((f.run / "report.json").read_bytes(), saved_report)
                    self.assertFalse((f.run / "reserved_final_model.pt").exists())
                    # An arbitrary path cannot become a selected stage, and a
                    # rejected selector cannot reach admission/scientific imports.
                    with mock.patch.object(entry, "admit") as admission, redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            entry.main(["--stage", str(f.proposal)])
                        admission.assert_not_called()
                self.assertEqual(entry._stage().name, "original")
        # Default invocation remains the original grant/run identity.
        with fixture() as f:
            context, _ = entry.admit()
            self.assertEqual(context.stage.run_dir, f.run)
            self.assertEqual(context.stage.name, "original")
            self.assertNotIn("feature_read_mode", context.profile)

    def test_inactive_profile_refuses_before_allocation_claim_or_scientific_import(self):
        with fixture() as f, mock.patch.object(entry, "_load_operations") as loader:
            record = profile()
            record["authorized"] = False
            f.proposal.write_text(json.dumps(record))
            (f.run / "allocation.txt").unlink()
            with self.assertRaisesRegex(entry.AdmissionError, "profile_not_activated"):
                entry.admit()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main([]), 2)
            loader.assert_not_called()
            self.assertFalse((f.run / "claim.json").exists())
            self.assertFalse((f.run / "report.json").exists())

    def test_verified_allocation_resources_roles_and_format_equivalence(self):
        with fixture() as f:
            for memory in ("16G", "16384M", "16384"):
                text = snapshot().replace("MinMemoryNode=16G", "MinMemoryNode=" + memory)
                (f.run / "allocation.txt").write_text(text.replace("02:00:00", "2:00:00"))
                context, budget = entry.admit()
                self.assertEqual(context.allocation["memory_gib"], 16)
                self.assertEqual(budget.deadline, 7300)
            for old, new in (("JobId=123", "JobId=124"), ("NumCPUs=3", "NumCPUs=2"),
                             ("NodeList=sh04-04n21", "NodeList=sh04-ln04"),
                             ("MinMemoryNode=16G", "MinMemoryNode=7G"),
                             ("Partition=russpold", "Partition=bigmem"),
                             ("TimeLimit=02:00:00", "TimeLimit=01:00:00"),
                             ("billing=3", "billing=3,gres/gpu=1")):
                (f.run / "allocation.txt").write_text(snapshot().replace(old, new))
                with self.subTest(new=new), self.assertRaises(entry.AdmissionError):
                    entry.admit()
            (f.run / "allocation.txt").write_text(snapshot())
            with mock.patch.object(entry.time, "time", return_value=2_000_001_000):
                with self.assertRaisesRegex(entry.AdmissionError, "remaining"):
                    entry.admit()
            with mock.patch.dict(os.environ, {"SLURM_JOB_GPUS": "0"}):
                with self.assertRaises(entry.AdmissionError):
                    entry.admit()
            record = profile()
            record["input_scope"]["source_training"] = "sub-01"
            f.proposal.write_text(json.dumps(record))
            with self.assertRaises(entry.AdmissionError):
                entry.admit()

    def test_resolver_only_bind_and_fixed_four_updates_and_revocation(self):
        with fixture() as f:
            context, budget = entry.admit()
            common = dict(source=entry.SOURCE, target=entry.TARGET, budget=budget)
            self.assertIsNone(entry.resolver(context, operation="bind_content_inputs", execution_options=None, **common))
            self.assertIsNone(entry.resolver(context, operation=entry.FIXED["operation"], execution_options=options(f.run), **common))
            for operation, selected in (("train_and_publish_full_pair", options(f.run)),
                                        (entry.FIXED["operation"], {**options(f.run), "updates": 1024}),
                                        ("bind_content_inputs", {})):
                with self.assertRaises(entry.AdmissionError):
                    entry.resolver(context, operation=operation, execution_options=selected, **common)
            record = profile()
            record["authorized"] = False
            f.proposal.write_text(json.dumps(record))
            with self.assertRaises(entry.AdmissionError):
                entry.resolver(context, operation="bind_content_inputs", execution_options=None, **common)

    def test_once_only_failure_is_redacted_and_preserved(self):
        with fixture() as f:
            sentinel = "PRIVATE_EMPIRICAL_VALUE_DO_NOT_PRINT"
            def fail(*args, **kwargs):
                raise RuntimeError(sentinel)
            with mock.patch.object(entry, "_load_operations", return_value=(mock.Mock(), fail)) as loader:
                self.assertEqual(entry.run_once(), 1)
                loader.assert_called_once()
                report = (f.run / "report.json").read_text()
                self.assertNotIn(sentinel, report)
                self.assertEqual(json.loads(report)["error_type"], "RuntimeError")
                with self.assertRaises(FileExistsError):
                    entry.run_once()
                loader.assert_called_once()
                self.assertEqual((f.run / "report.json").read_text(), report)
                self.assertFalse((f.run / "reserved_final_model.pt").exists())

    def test_budget_caps_check_mode_and_successful_composition(self):
        with fixture() as f:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(["--check-admission"]), 0)
            self.assertFalse((f.run / "claim.json").exists())
            context, budget = entry.admit()
            with mock.patch.object(entry.time, "monotonic", return_value=7300):
                with self.assertRaises(TimeoutError):
                    budget.check()
            with mock.patch.object(entry.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=15 * 1024**2 + 1)):
                with self.assertRaises(MemoryError):
                    budget.check()
            # A tiny substituted cap exercises refusal inside the batch file cap.
            scratch_file = f.scratch / "too_big"
            scratch_file.write_bytes(b"123456789")
            with mock.patch.dict(entry.RESOURCES, {"scratch_bytes": 8}):
                with self.assertRaises(entry.AdmissionError):
                    budget.check_storage()
            scratch_file.unlink()
            with mock.patch.dict(entry.RESOURCES, {"durable_bytes": 8}):
                with self.assertRaises(entry.AdmissionError):
                    entry._publish(f.run / "refused.json", {"value": 1})
            self.assertFalse((f.run / "refused.json").exists())
            bound = object()
            def bind(received, **kwargs):
                kwargs["require_admitted_context"](received, operation="bind_content_inputs", source=kwargs["source"],
                    target=kwargs["target"], budget=kwargs["budget"], execution_options=None)
                return bound
            def measure(received, **kwargs):
                kwargs["require_admitted_context"](received, operation=entry.FIXED["operation"],
                    source=entry.SOURCE, target=entry.TARGET, budget=kwargs["budget"], execution_options=options(f.run))
                self.assertIs(kwargs["prepare_inputs"](), bound)
                return {"status": "complete_measurement", "completed_updates": 4,
                        "model_published": False, "production_fit_completed": False}
            with mock.patch.object(entry, "_load_operations", return_value=(bind, measure)):
                self.assertEqual(entry.run_once(), 0)
            self.assertEqual(json.loads((f.run / "report.json").read_text())["completed_updates"], 4)
            self.assertFalse((f.run / "reserved_final_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
