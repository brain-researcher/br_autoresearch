"""Three source-free admission fixtures; run only in the coordinator's Slurm job.

Both separately granted roster stages use temporary operational metadata and
mocked fitting/publication. No scientific module or empirical input is loaded.
"""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
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
import run_brain_pair_roster as runner


EXPECTED = {
    "roster-006": (6, "sub-06", 12023, "sub03_to_sub06_001", "006"),
    "roster-007": (7, "sub-07", 12057, "sub03_to_sub07_001", "007"),
}
RESOURCES = {
    "cpus": 3, "memory_gib": 16, "minutes": 1440, "gpus": 0,
    "deadline_seconds": 84600, "rss_limit_bytes": 16106127360,
    "durable_bytes": 2147483648, "scratch_bytes": 268435456,
    "final_artifact_bytes": 805306368,
}


def profile(stage):
    return {**runner.core.FIXED, "source": stage.source, "target": stage.target,
            "source_native_voxels": stage.source_native_voxels,
            "target_native_voxels": stage.target_native_voxels,
            "resources": dict(RESOURCES),
            "input_scope": {**runner.core.INPUT_SCOPE, "source_tau_train": stage.source,
                            "target_tau_train": stage.target},
            "authorized": True, "approval_source": "fixture_separate_pair_authority"}


@contextmanager
def stage_files(name):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        with mock.patch.object(runner, "BASE", root):
            for identity in EXPECTED:
                stage = runner._stage(identity)
                stage.run_dir.mkdir(parents=True)
                stage.profile_path.write_text(json.dumps(profile(stage)))
            selected = runner._stage(name)
            other = runner._stage("roster-007" if name == "roster-006" else "roster-006")
            history = other.run_dir / "historical.txt"
            history.write_text("separate attempt must remain untouched")
            yield SimpleNamespace(root=root, stage=selected, other=other,
                run=selected.run_dir, final=selected.final_path, history=history,
                history_before=history.read_bytes(), grant_before=other.profile_path.read_bytes())


def context_budget(f):
    scratch = f.root / f.stage.scratch_name
    scratch.mkdir()
    now = time.time()
    allocation = {"job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname().split(".")[0], "job_start_epoch": now - 600,
        "job_end_epoch": now + 85800, "deadline_epoch": now + 84000,
        "cpus": 3, "memory_gib": 16, "minutes": 1440, "gpus": 0,
        "partition": "russpold"}
    context = runner.BrainPairContext(profile(f.stage), allocation, scratch, f.stage)
    return context, runner.BrainPairBudget(context)


class BrainPairRosterTests(unittest.TestCase):
    def test_fixed_pair_grants_refuse_before_allocation_and_operation_import(self):
        self.assertEqual(tuple(runner.STAGES), tuple(EXPECTED))
        with self.assertRaises(runner.core.AdmissionError):
            runner.admit("next-memory-002")
        for name, (index, target, width, stem, suffix) in EXPECTED.items():
            with self.subTest(stage=name), stage_files(name) as f:
                stage = f.stage
                self.assertEqual((stage.source, stage.target, stage.pair_index), ("sub-03", target, index))
                self.assertEqual((stage.source_native_voxels, stage.target_native_voxels), (12897, width))
                self.assertEqual(stage.profile_path, f.root / f"outputs/brain_pair_roster_{suffix}_authorization.json")
                self.assertEqual(stage.run_dir, f.root / "outputs/runs/brain_pair_training" / stem)
                self.assertEqual(stage.final_path, stage.run_dir / "final_brain_pair.pt")
                self.assertEqual(stage.scratch_name, "ep21_brain_pair_" + stem)
                self.assertTrue(stage.allocation_start_deadline)
                self.assertEqual(stage.resources, RESOURCES)
                original = profile(stage)
                invalid = [
                    {**original, "authorized": False, "approval_source": None},
                    {**original, "target": f.other.target},
                    {**original, "source_native_voxels": 12896},
                    {**original, "target_native_voxels": f.other.target_native_voxels},
                    {**original, "training_images": 896},
                    {**original, "cv_fits": 49},
                    {**original, "input_scope": {**original["input_scope"], "tau_test": True}},
                    {**original, "input_scope": {**original["input_scope"], "target_tau_train": f.other.target}},
                    {**original, "resources": {**RESOURCES, "cpus": 2}},
                    {**original, "resources": {**RESOURCES, "deadline_seconds": 84601}},
                ]
                with mock.patch.object(runner, "_allocation") as allocation, \
                        mock.patch.object(runner, "_scratch") as scratch, \
                        mock.patch.object(runner, "_load_operations") as load:
                    for changed in invalid:
                        stage.profile_path.write_text(json.dumps(changed))
                        with self.subTest(changed=changed), self.assertRaises(runner.core.AdmissionError):
                            runner.run_once(name)
                    allocation.assert_not_called()
                    scratch.assert_not_called()
                    load.assert_not_called()
                    self.assertFalse((f.run / "claim.json").exists())
                stage.profile_path.write_text(json.dumps(original))
                context, budget = context_budget(f)
                arguments = dict(source=stage.source, target=stage.target, budget=budget)
                options = {"final_path": f.final, "maximum_bytes": 805306368}
                runner.resolver(context, operation="bind_brain_inputs", execution_options=None, **arguments)
                runner.resolver(context, operation="train_and_publish_brain_pair", execution_options=options, **arguments)
                for changed in ({"operation": "train_and_publish_cached_cuda_full_pair"},
                                {"source": stage.target}, {"target": f.other.target},
                                {"execution_options": {**options, "final_path": f.other.final_path}},
                                {"execution_options": {**options, "maximum_bytes": 805306369}},
                                {"execution_options": {**options, "resume": True}}):
                    with self.subTest(changed=changed), self.assertRaises(runner.core.AdmissionError):
                        runner.resolver(context, **{**arguments, "operation": "train_and_publish_brain_pair",
                                                  "execution_options": options, **changed})
                other_context = runner.BrainPairContext(context.profile, context.allocation, context.scratch, f.other)
                with self.assertRaises(runner.core.AdmissionError):
                    runner.resolver(other_context, operation="bind_brain_inputs", execution_options=None, **arguments)

    def test_actual_allocation_and_cli_preserve_shared_absolute_deadline(self):
        now = datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()
        def stamp(epoch):
            return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None).isoformat()
        env = {**{key: "1" for key in runner.core.THREADS}, "SLURM_JOB_ID": "87654321",
               "SLURM_CPUS_PER_TASK": "3", "SLURM_MEM_PER_NODE": "16384",
               "SLURM_NTASKS": "1", "SLURM_NNODES": "1", "CUDA_VISIBLE_DEVICES": "",
               "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": "", "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": ""}
        fields = {"JobId": "87654321", "NodeList": "sh04-fixture", "JobState": "RUNNING",
                  "NumCPUs": "3", "NumTasks": "1", "NumNodes": "1", "Partition": "russpold",
                  "TimeLimit": "1-00:00:00", "MinMemoryNode": "16G", "AllocTRES": "cpu=3,node=1,mem=16G",
                  "StartTime": stamp(now - 120), "EndTime": stamp(now + 86280)}
        for name in EXPECTED:
            with self.subTest(stage=name), stage_files(name) as f, ExitStack() as stack:
                scratch = f.root / f.stage.scratch_name
                scratch.mkdir()
                stack.enter_context(mock.patch.dict(os.environ, {**env, "L_SCRATCH_JOB": str(f.root), "TMPDIR": str(scratch)}))
                stack.enter_context(mock.patch.object(runner.core.socket, "gethostname", return_value="sh04-fixture.stanford.edu"))
                wall = stack.enter_context(mock.patch.object(runner.core.time, "time", return_value=now))
                mono = stack.enter_context(mock.patch.object(runner.core.time, "monotonic", return_value=1000.))
                def write_allocation(changes=None):
                    (f.run / "allocation.txt").write_text(" ".join(
                        f"{key}={value}" for key, value in {**fields, **(changes or {})}.items()))
                for limit in ("1-00:00:00", "24:00:00"):
                    write_allocation({"TimeLimit": limit})
                    context, budget = runner.admit(name)
                    self.assertEqual((context.allocation["cpus"], context.allocation["memory_gib"]), (3, 16))
                    self.assertEqual(budget.remaining_seconds(), 84480)
                    self.assertEqual(budget.absolute_deadline, now + 84480)
                for changed in ({"NumCPUs": "2"}, {"MinMemoryNode": "7G"},
                                {"AllocTRES": "cpu=2,node=1,mem=16G"},
                                {"AllocTRES": "cpu=3,node=1,mem=16G,gres/gpu=1"},
                                {"NodeList": "different-node"}, {"TimeLimit": "23:59:59"},
                                {"StartTime": stamp(now + 1)}, {"StartTime": stamp(now - 84601)},
                                {"EndTime": stamp(now + 60)}):
                    write_allocation(changed)
                    with self.subTest(changed=changed), self.assertRaises(runner.core.AdmissionError):
                        runner.admit(name)
                write_allocation()
                for changed in ({"SLURM_CPUS_PER_TASK": "2"}, {"SLURM_MEM_PER_NODE": "7168"},
                                {"OMP_NUM_THREADS": "2"}, {"TMPDIR": str(f.root / f.other.scratch_name)}):
                    with mock.patch.dict(os.environ, changed), self.assertRaises(runner.core.AdmissionError):
                        runner.admit(name)
                first_context, first_budget = runner.admit(name)
                for elapsed in (0, 600):
                    wall.return_value, mono.return_value = now + elapsed, 1000. + elapsed
                    with mock.patch.object(runner, "_load_operations") as load, redirect_stdout(io.StringIO()) as out:
                        self.assertEqual(runner.main(["--stage", name, "--remaining-seconds"]), 0)
                        self.assertEqual(out.getvalue(), str(84480 - elapsed) + "\n")
                        load.assert_not_called()
                    self.assertFalse((f.run / "claim.json").exists())
                later_context, later_budget = runner.admit(name)
                self.assertEqual(later_context.allocation["deadline_epoch"], first_context.allocation["deadline_epoch"])
                self.assertEqual(later_budget.deadline, first_budget.deadline)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as refused:
                    runner.main(["--stage", name, "--remaining-seconds", "--deadline-seconds", "90000"])
                self.assertEqual(refused.exception.code, 2)
                write_allocation({"EndTime": stamp(now + 720)})
                self.assertEqual(runner.admit(name)[1].remaining_seconds(), 60)
                with mock.patch.object(runner.core.resource, "getrusage",
                        return_value=SimpleNamespace(ru_maxrss=16106127360 // 1024 + 1)), self.assertRaises(MemoryError):
                    later_budget.check()
                wall.return_value = later_budget.absolute_deadline
                with self.assertRaises(TimeoutError):
                    later_budget.check()

    def test_exclusive_full_scope_composition_and_conservative_pair_bound_reports(self):
        cases = ("complete", "wrong_role", "wrong_native_width", "wrong_participant",
                 "wrong_rows", "wrong_pair_index", "wrong_publication_target", "fit_failure", "closeout_failure")
        for name in EXPECTED:
            for outcome in cases:
                with self.subTest(stage=name, outcome=outcome), stage_files(name) as f:
                    context, budget = context_budget(f)
                    responses = []
                    for participant, width in ((f.stage.source, 12897), (f.stage.target, f.stage.target_native_voxels)):
                        target = participant == f.stage.target
                        if target and outcome == "wrong_native_width":
                            width -= 1
                        rows = 896 if target and outcome == "wrong_rows" else 897
                        responses.append(SimpleNamespace(
                            participant=f.other.target if target and outcome == "wrong_participant" else participant,
                            role="tau_test" if target and outcome == "wrong_role" else "tau_train",
                            image_ids=range(rows), native_voxel_ids=range(width),
                            values=SimpleNamespace(shape=(rows, width))))
                    bound = SimpleNamespace(source=responses[0], target=responses[1])
                    def bind(actual_context, **kwargs):
                        self.assertIs(actual_context, context)
                        self.assertTrue((f.run / "claim.json").is_file())
                        self.assertEqual((kwargs["source"], kwargs["target"]), (f.stage.source, f.stage.target))
                        self.assertIs(kwargs["budget"], budget)
                        self.assertIs(kwargs["require_admitted_context"], runner.resolver)
                        runner.resolver(context, operation="bind_brain_inputs", source=f.stage.source,
                                        target=f.stage.target, budget=budget, execution_options=None)
                        return bound
                    def train(actual_bound, **kwargs):
                        self.assertIs(actual_bound, bound)
                        self.assertEqual(kwargs, {"final_path": f.final, "maximum_bytes": 805306368})
                        runner.resolver(context, operation="train_and_publish_brain_pair", source=f.stage.source,
                                        target=f.stage.target, budget=budget, execution_options=kwargs)
                        if outcome == "fit_failure":
                            raise RuntimeError("sensitive_values_and_paths_must_remain_redacted")
                        f.final.write_bytes(b"mock_final_model")
                        if outcome == "closeout_failure":
                            budget.check_storage = mock.Mock(side_effect=OSError("private closeout detail"))
                        return SimpleNamespace(source=f.stage.source,
                            target=f.other.target if outcome == "wrong_publication_target" else f.stage.target,
                            pair_index=f.other.pair_index if outcome == "wrong_pair_index" else f.stage.pair_index,
                            path=f.final, summary=SimpleNamespace(available=True, completed_fits=51))
                    training = mock.Mock(side_effect=train)
                    with mock.patch.object(runner, "admit", return_value=(context, budget)), \
                            mock.patch.object(runner, "_load_operations", return_value=(bind, training,
                                {"python": "fixture", "numpy": "fixture", "torch": "fixture"})) as load:
                        self.assertEqual(runner.run_once(name), 0 if outcome == "complete" else 1)
                        report_text = (f.run / "report.json").read_text()
                        report = json.loads(report_text)
                        claim = json.loads((f.run / "claim.json").read_text())
                        for record in (claim, report):
                            self.assertEqual((record["pair_stage"], record["pair_index"], record["source"], record["target"]),
                                             (name, f.stage.pair_index, "sub-03", f.stage.target))
                        known_publication = outcome in ("complete", "closeout_failure")
                        scope_failure = outcome in ("wrong_role", "wrong_native_width", "wrong_participant", "wrong_rows")
                        self.assertEqual(report["completed_fits"], 51 if known_publication else None)
                        self.assertEqual(report["model_published"], True if known_publication else False if scope_failure else None)
                        self.assertEqual(report["published_path"], str(f.final) if known_publication else None)
                        self.assertEqual(training.call_count, 0 if scope_failure else 1)
                        self.assertEqual(report["status"], "complete_brain_pair" if outcome == "complete" else "incomplete")
                        self.assertNotIn("sensitive_values", report_text)
                        self.assertNotIn("private closeout", report_text)
                        self.assertNotIn("brain_loss_failure", report)
                        with self.assertRaises((runner.core.AdmissionError, FileExistsError)):
                            runner.run_once(name)
                        self.assertEqual(load.call_count, 1)
                    self.assertEqual(f.history.read_bytes(), f.history_before)
                    self.assertEqual(f.other.profile_path.read_bytes(), f.grant_before)
                    self.assertFalse(f.other.final_path.exists())
                    self.assertFalse((f.other.run_dir / "claim.json").exists())


if __name__ == "__main__":
    unittest.main()
