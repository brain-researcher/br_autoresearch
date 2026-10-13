"""Three stdlib-only admission fixtures; no scientific imports or payloads."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_finding_a_readout as runner


RESOURCES = {"cpus": 3, "memory_gib": 16, "minutes": 4320, "gpus": 0,
    "deadline_seconds": 255600, "rss_limit_bytes": 16106127360,
    "durable_bytes": 51539607552, "scratch_bytes": 4294967296, "collection_bytes": 42949672960}
NOW = datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()


def profile():
    return {**runner.FIXED, "resources": dict(RESOURCES), "input_scope": deepcopy(runner.INPUT_SCOPE),
            "authorized": True, "approval_source": "fixture_explicit_readout_selection"}


def stamp(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None).isoformat()


@contextmanager
def files():
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "BASE", Path(directory).resolve()):
        stage = runner._stage()
        stage.run_dir.mkdir(parents=True)
        stage.profile_path.write_text(json.dumps(profile()))
        (stage.run_dir / "batch_status.txt").write_text("\n".join(key + "=0" for key in runner.CHECK_RECORDS) + "\n")
        scratch = runner.BASE / stage.scratch_name
        scratch.mkdir()
        history = runner.BASE / "outputs/runs/content_pair_training/cuda001/history.txt"
        history.parent.mkdir(parents=True)
        history.write_text("existing original-model history")
        yield SimpleNamespace(stage=stage, scratch=scratch, root=runner.BASE, history=history)


@contextmanager
def machine(f):
    env = {**{key: "1" for key in runner.core.THREADS}, "SLURM_JOB_ID": "98765432",
           "SLURM_CPUS_PER_TASK": "3", "SLURM_MEM_PER_NODE": "16384", "SLURM_NTASKS": "1",
           "SLURM_NNODES": "1", "CUDA_VISIBLE_DEVICES": "", "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": "",
           "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": "", "L_SCRATCH_JOB": str(f.root), "TMPDIR": str(f.scratch)}
    fields = {"JobId": env["SLURM_JOB_ID"], "NodeList": "sh04-fixture", "JobState": "RUNNING",
              "NumCPUs": "3", "NumTasks": "1", "NumNodes": "1", "Partition": "russpold",
              "TimeLimit": "3-00:00:00", "MinMemoryNode": "16G", "AllocTRES": "cpu=3,node=1,mem=16G",
              "StartTime": stamp(NOW - 120), "EndTime": stamp(NOW + 259080)}

    def allocation(changes=None):
        (f.stage.run_dir / "allocation.txt").write_text(" ".join(
            f"{key}={value}" for key, value in {**fields, **(changes or {})}.items()))

    allocation()
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(runner.core.socket, "gethostname", return_value="sh04-fixture.stanford.edu"), \
            mock.patch.object(runner.time, "time", return_value=NOW) as wall, \
            mock.patch.object(runner.time, "monotonic", return_value=1000.) as mono, \
            mock.patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=0)) as usage:
        yield SimpleNamespace(allocation=allocation, wall=wall, mono=mono, usage=usage)


class FindingAReadoutAdmissionTests(unittest.TestCase):
    def test_fixed_scope_checks_and_operation_refusals_before_science(self):
        self.assertEqual(runner.RESOURCES, RESOURCES)
        self.assertEqual(runner.OPERATIONS, ("run_finding_a_first_readout", "bind_finding_a_inputs",
                                            "read_finding_a_tau_test", "read_finding_a_truth"))
        with files() as f:
            self.assertEqual(f.stage.profile_path, f.root / "outputs/finding_a_first_readout_authorization.json")
            self.assertEqual(f.stage.run_dir, f.root / "outputs/runs/finding_a_feature_decoding/first001")
            self.assertEqual(f.stage.final_path.name, "finding_a_collection.h5")
            original_profile = profile()
            changes = [{**original_profile, "authorized": False}, {**original_profile, "source": "sub-01"},
                {**original_profile, "target": "sub-05"}, {**original_profile, "draw_batch_size": 8},
                {**original_profile, "image_bootstrap_draws": 999}, {**original_profile, "model_policy": "amended"},
                {**original_profile, "resources": {**RESOURCES, "collection_bytes": 42949672961}},
                {**original_profile, "input_scope": {**runner.INPUT_SCOPE, "target_tau_test": True}}]
            with mock.patch.object(runner.core, "_allocation") as allocation, \
                    mock.patch.object(runner, "_load_operation") as load, \
                    mock.patch.object(runner, "_configure_file_policy") as policy:
                for changed in changes:
                    f.stage.profile_path.write_text(json.dumps(changed))
                    with self.assertRaises(runner.AdmissionError):
                        runner.run_once()
                allocation.assert_not_called(); load.assert_not_called(); policy.assert_not_called()
                self.assertFalse((f.stage.run_dir / "claim.json").exists())
            f.stage.profile_path.write_text(json.dumps(original_profile))
            with machine(f), mock.patch.object(runner, "_load_operation") as load, \
                    mock.patch.object(runner, "_configure_file_policy") as configure:
                complete = (f.stage.run_dir / "batch_status.txt").read_text()
                for altered in (complete.replace("batched_inference_checks_exit=0\n", ""),
                                complete.replace("readout_admission_checks_exit=0", "readout_admission_checks_exit=1"),
                                complete + "empirical_input_checks_exit=0\n"):
                    (f.stage.run_dir / "batch_status.txt").write_text(altered)
                    with self.assertRaises(runner.AdmissionError):
                        runner.run_once()
                load.assert_not_called(); configure.assert_not_called()
                (f.stage.run_dir / "batch_status.txt").write_text(complete)
            with machine(f) as actual, mock.patch.object(runner, "_require_file_policy"):
                context, budget = runner.admit()
                options = {"collection_path": f.stage.final_path, "maximum_bytes": RESOURCES["collection_bytes"]}
                request = dict(operation=runner.FIXED["operation"], source="sub-03", target="sub-01",
                               budget=budget, execution_options=options)
                for operation in runner.OPERATIONS:
                    runner.resolver(context, **{**request, "operation": operation,
                        "execution_options": options if operation == runner.FIXED["operation"] else None})
                for altered in ({"operation": "train_and_publish_cached_cuda_full_pair"},
                                {"source": "sub-01"}, {"target": "sub-05"},
                                {"execution_options": {**options, "collection_path": f.scratch / "other.h5"}},
                                {"execution_options": {**options, "maximum_bytes": 42949672961}},
                                {"operation": "read_finding_a_tau_test", "execution_options": {}}):
                    with self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, **{**request, **altered})
                other = runner.ReadoutContext(context.profile, context.allocation, context.scratch, context.stage)
                with self.assertRaises(runner.AdmissionError):
                    runner.resolver(other, **request)
                # Metadata revalidation is bounded; cheap budget checks always run.
                with mock.patch.object(runner, "_profile", wraps=runner._profile) as reread, \
                        mock.patch.object(budget, "check_storage", wraps=budget.check_storage) as storage:
                    for tick in (1000.1, 1000.5, 1000.9):
                        actual.mono.return_value = tick
                        runner.resolver(context, **request)
                    reread.assert_not_called(); storage.assert_not_called()
                    actual.mono.return_value = 1001.1
                    runner.resolver(context, **request)
                    self.assertEqual((reread.call_count, storage.call_count), (1, 1))
                f.stage.profile_path.write_text(json.dumps({**original_profile, "authorized": False}))
                actual.mono.return_value = 1002.2
                with self.assertRaises(runner.AdmissionError):
                    runner.resolver(context, **request)

    def test_actual_allocation_absolute_clock_caps_and_precheck_cli(self):
        with files() as f, machine(f) as actual:
            for value in ("3-00:00:00", "72:00:00"):
                actual.allocation({"TimeLimit": value})
                context, budget = runner.admit()
                self.assertEqual(budget.remaining_seconds(), 255480)
                self.assertEqual(budget.absolute_deadline, NOW + 255480)
            for changed in ({"NumCPUs": "2"}, {"MinMemoryNode": "7G"},
                            {"AllocTRES": "cpu=3,node=1,mem=16G,gres/gpu=1"},
                            {"NodeList": "different-node"}, {"TimeLimit": "2-00:00:00"},
                            {"StartTime": stamp(NOW + 1)}, {"StartTime": stamp(NOW - 255601)},
                            {"EndTime": stamp(NOW + 60)}):
                actual.allocation(changed)
                with self.assertRaises(runner.AdmissionError):
                    runner.admit()
            actual.allocation({"EndTime": stamp(NOW + 600)})
            self.assertEqual(runner.admit()[1].remaining_seconds(), 540)
            actual.allocation()
            for env in ({"OMP_NUM_THREADS": "2"}, {"SLURM_MEM_PER_NODE": "7168"},
                        {"CUDA_VISIBLE_DEVICES": "0"}, {"TMPDIR": str(f.root / "wrong")}):
                with mock.patch.dict(os.environ, env), self.assertRaises(runner.AdmissionError):
                    runner.admit()
            context, budget = runner.admit()
            actual.usage.return_value = SimpleNamespace(ru_maxrss=RESOURCES["rss_limit_bytes"] // 1024 + 1)
            with self.assertRaises(MemoryError):
                budget.check()
            actual.usage.return_value = SimpleNamespace(ru_maxrss=0)
            (f.stage.run_dir / "batch_status.txt").unlink()
            with mock.patch.object(runner, "_load_operation") as load, \
                    mock.patch.object(runner, "_configure_file_policy") as configure:
                for elapsed in (0, 600):
                    actual.wall.return_value, actual.mono.return_value = NOW + elapsed, 1000. + elapsed
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(runner.main(["--remaining-seconds"]), 0)
                    self.assertEqual(output.getvalue(), str(255480 - elapsed) + "\n")
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(runner.main(["--check-admission"]), 0)
                self.assertFalse(json.loads(output.getvalue())["payload_read"])
                load.assert_not_called(); configure.assert_not_called()
                self.assertFalse((f.stage.run_dir / "claim.json").exists())
            late = runner.admit()[1]
            self.assertEqual(late.absolute_deadline, budget.absolute_deadline)
            self.assertEqual(late.deadline, budget.deadline)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(["--collection-path", "/unselected.h5"])
        # Only the admitted standalone child establishes its process policy.
        state = {"soft": 32 * 1024**2, "hard": resource.RLIM_INFINITY, "signal": signal.SIG_IGN}

        def set_limit(which, values):
            self.assertEqual(which, resource.RLIMIT_FSIZE)
            state["soft"], state["hard"] = values

        def set_signal(which, value):
            self.assertEqual(which, signal.SIGXFSZ)
            state["signal"] = value

        with mock.patch.object(runner.resource, "getrlimit", side_effect=lambda which: (state["soft"], state["hard"])), \
                mock.patch.object(runner.resource, "setrlimit", side_effect=set_limit) as limit, \
                mock.patch.object(runner.signal, "getsignal", side_effect=lambda which: state["signal"]), \
                mock.patch.object(runner.signal, "signal", side_effect=set_signal) as disposition:
            runner._configure_file_policy()
            self.assertEqual(state["soft"], 42949672960)
            self.assertEqual(state["signal"], signal.SIG_DFL)
            self.assertEqual((limit.call_count, disposition.call_count), (1, 1))
            state["hard"] = 1024
            with self.assertRaises(runner.AdmissionError):
                runner._configure_file_policy()
            self.assertEqual((limit.call_count, disposition.call_count), (1, 1))

    def test_single_attempt_composition_redaction_and_publication_facts(self):
        for mode in ("complete", "failure", "post_publication_failure", "wrong_paths"):
            with self.subTest(mode=mode), files() as f, machine(f), ExitStack() as stack:
                history = f.history.read_bytes()
                stack.enter_context(mock.patch.object(runner, "_configure_file_policy"))
                stack.enter_context(mock.patch.object(runner, "_require_file_policy"))

                def operation(context, *, require_admitted_context, budget, collection_path, maximum_bytes):
                    self.assertTrue((f.stage.run_dir / "claim.json").is_file())
                    self.assertIs(require_admitted_context, runner.resolver)
                    self.assertEqual((collection_path, maximum_bytes), (f.stage.final_path, 42949672960))
                    for selected in runner.OPERATIONS:
                        require_admitted_context(context, operation=selected, source="sub-03", target="sub-01", budget=budget,
                            execution_options={"collection_path": collection_path, "maximum_bytes": maximum_bytes}
                            if selected == runner.FIXED["operation"] else None)
                    if mode == "failure":
                        f.stage.final_path.with_name(f.stage.final_path.name + ".partial").write_bytes(b"retained partial")
                        raise TimeoutError("fixture empirical values must not appear")
                    if mode == "wrong_paths":
                        return (f.scratch / "other.h5",) * 3
                    for path in f.stage.publication_paths:
                        path.write_bytes(b"mock published artifact")
                    if mode == "post_publication_failure":
                        budget.check_storage = mock.Mock(side_effect=MemoryError("fixture closeout limit"))
                    return f.stage.publication_paths

                load = stack.enter_context(mock.patch.object(runner, "_load_operation", return_value=(operation, {"fixture": "stdlib"})))
                self.assertEqual(runner.run_once(), 0 if mode == "complete" else 1)
                report = json.loads((f.stage.run_dir / "report.json").read_text())
                self.assertEqual(report["quantitative_finding"], "not_evaluable")
                self.assertEqual(report["comparability"], "not_evaluable")
                self.assertFalse(report["model_training_performed"])
                self.assertNotIn("fixture empirical", json.dumps(report))
                self.assertNotIn("fixture closeout", json.dumps(report))
                if mode in ("complete", "post_publication_failure"):
                    self.assertTrue(report["artifacts_published"])
                    self.assertEqual((report["decoded_slots"], report["unavailable_slots"]), (38, 817))
                    self.assertEqual(report["published_paths"], list(map(str, f.stage.publication_paths)))
                else:
                    self.assertIsNone(report["artifacts_published"])
                if mode != "complete":
                    self.assertEqual(report["status"], "incomplete")
                if mode == "failure":
                    self.assertTrue(report["resource_limit_failure"])
                    self.assertTrue(f.stage.final_path.with_name(f.stage.final_path.name + ".partial").is_file())
                if mode == "post_publication_failure":
                    self.assertEqual(report["failed_stage"], "closeout")
                with self.assertRaises(runner.AdmissionError):
                    runner.run_once()
                self.assertEqual(load.call_count, 1)
                self.assertEqual(f.history.read_bytes(), history)


if __name__ == "__main__":
    unittest.main()
