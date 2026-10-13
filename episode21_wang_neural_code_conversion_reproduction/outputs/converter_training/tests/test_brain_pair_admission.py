"""Three parameterized stdlib fixtures; coordinator Slurm execution only.

Temporary operational metadata and mocked operations exercise all four fixed stages;
no scientific import, numerical fitting or real payload read is performed.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
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
import run_brain_pair as runner


def profile(stage):
    return {**runner.FIXED, "source": stage.source, "target": stage.target,
            "source_native_voxels": stage.source_native_voxels,
            "target_native_voxels": stage.target_native_voxels,
            "resources": stage.resources,
            "input_scope": {**runner.INPUT_SCOPE, "source_tau_train": stage.source,
                            "target_tau_train": stage.target},
            "authorized": True, "approval_source": "fixture_prospective_authorization"}


@contextmanager
def stage_files(stage_name):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary).resolve()
        old, new, memory = root / "original", root / "next", root / "memory"
        repair = root / "memory_repair"
        old.mkdir()
        new.mkdir()
        memory.mkdir()
        repair.mkdir()
        for name, value in (("PROFILE", root / "original.json"), ("RUN_DIR", old),
                            ("FINAL", old / "final_brain_pair.pt"),
                            ("NEXT_PROFILE", root / "next.json"), ("NEXT_RUN_DIR", new),
                            ("MEMORY_PROFILE", root / "memory.json"), ("MEMORY_RUN_DIR", memory),
                            ("MEMORY_REPAIR_PROFILE", root / "memory_repair.json"),
                            ("MEMORY_REPAIR_RUN_DIR", repair)):
            stack.enter_context(mock.patch.object(runner, name, value))
        for name in runner.STAGES:
            selected = runner._stage(name)
            selected.profile_path.write_text(json.dumps(profile(selected)))
        selected = runner._stage(stage_name)
        other = runner._stage("next-001" if stage_name == "original" else "original")
        histories = []
        for name in runner.STAGES:
            if name != stage_name:
                previous = runner._stage(name)
                history = previous.run_dir / "historical.txt"
                history.write_text("preserved unrelated attempt " + name)
                histories.append((previous, history, history.read_bytes(), previous.profile_path.read_bytes()))
        yield SimpleNamespace(root=root, stage=selected, other=other, histories=histories,
                              run=selected.run_dir, path=selected.final_path)


class BrainPairAdmissionTests(unittest.TestCase):
    def context_budget(self, root, stage):
        scratch = root / "budget_scratch"
        scratch.mkdir()
        now = time.time()
        allocation = {"job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": socket.gethostname().split(".")[0], "job_end_epoch": now + 85800,
            "cpus": stage.cpus, "memory_gib": stage.memory_gib}
        if stage.allocation_start_deadline:
            allocation.update(job_start_epoch=now - 600, deadline_epoch=now + 84000)
        context = runner.BrainPairContext(profile(stage), allocation, scratch, stage)
        return context, runner.BrainPairBudget(context)

    def test_inactive_profile_precedes_allocation_imports_and_resolver_is_exact(self):
        self.assertEqual(runner._stage().target, "sub-01")
        selected = runner._stage("next-001")
        self.assertEqual((selected.source, selected.target, selected.pair_index), ("sub-03", "sub-05", 5))
        self.assertEqual((selected.source_native_voxels, selected.target_native_voxels), (12897, 10891))
        self.assertEqual(selected.run_dir, runner.BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_001")
        self.assertEqual(selected.profile_path, runner.BASE / "outputs/brain_pair_next_authorization.json")
        self.assertEqual(selected.scratch_name, "ep21_brain_pair_sub03_to_sub05_001")
        memory = runner._stage("next-memory-001")
        self.assertEqual((memory.source, memory.target, memory.pair_index), ("sub-03", "sub-05", 5))
        self.assertEqual((memory.source_native_voxels, memory.target_native_voxels), (12897, 10891))
        self.assertEqual(memory.profile_path, runner.BASE / "outputs/brain_pair_memory_authorization.json")
        self.assertEqual(memory.run_dir, runner.BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_mem16_001")
        self.assertEqual(memory.scratch_name, "ep21_brain_pair_sub03_to_sub05_mem16_001")
        self.assertEqual(memory.resources, {**runner.RESOURCES, "cpus": 2, "memory_gib": 16,
                                           "rss_limit_bytes": 16106127360})
        repair = runner._stage("next-memory-002")
        self.assertEqual((repair.source, repair.target, repair.pair_index), ("sub-03", "sub-05", 5))
        self.assertEqual((repair.source_native_voxels, repair.target_native_voxels), (12897, 10891))
        self.assertEqual(repair.profile_path, runner.BASE / "outputs/brain_pair_memory_repair_authorization.json")
        self.assertEqual(repair.run_dir, runner.BASE / "outputs/runs/brain_pair_training/sub03_to_sub05_mem16_002")
        self.assertEqual(repair.scratch_name, "ep21_brain_pair_sub03_to_sub05_mem16_002")
        self.assertEqual(repair.resources, {**memory.resources, "cpus": 3})
        self.assertEqual(runner._stage().resources, selected.resources)
        self.assertEqual((runner.RESOURCES["cpus"], runner.RESOURCES["memory_gib"],
                          runner.RESOURCES["rss_limit_bytes"]), (1, 7, 6979321856))
        with self.assertRaises(runner.AdmissionError):
            runner.admit("sub03_to_arbitrary_participant")
        for stage_name in runner.STAGES:
            with self.subTest(stage=stage_name), stage_files(stage_name) as f:
                changed = profile(f.stage)
                changed.update(authorized=False, approval_source=None)
                f.stage.profile_path.write_text(json.dumps(changed))
                with mock.patch.object(runner, "_allocation") as allocation, \
                        mock.patch.object(runner, "_load_operations") as load:
                    with self.assertRaisesRegex(runner.AdmissionError, "profile_not_activated"):
                        runner.admit(stage_name)
                    allocation.assert_not_called()
                    load.assert_not_called()
                    invalid = [
                        {**profile(f.stage), "source_native_voxels": 12896},
                        {**profile(f.stage), "target": f.other.target},
                        {**profile(f.stage), "input_scope": {**profile(f.stage)["input_scope"], "tau_test": True}},
                        {**profile(f.stage), "resources": {**f.stage.resources, "cpus": True}},
                        {**profile(f.stage), "resources": {**f.stage.resources,
                            "rss_limit_bytes": 6979321856 if f.stage.rss_limit_bytes == 16106127360
                                               else 16106127360}},
                    ]
                    for value in invalid:
                        f.stage.profile_path.write_text(json.dumps(value))
                        with self.assertRaises(runner.AdmissionError):
                            runner._profile(f.stage)
                f.stage.profile_path.write_text(json.dumps(profile(f.stage)))
                context, budget = self.context_budget(f.root, f.stage)
                options = dict(final_path=f.path, maximum_bytes=f.stage.resources["final_artifact_bytes"])
                arguments = dict(source=f.stage.source, target=f.stage.target, budget=budget)
                runner.resolver(context, operation="bind_brain_inputs", execution_options=None, **arguments)
                runner.resolver(context, operation="train_and_publish_brain_pair", execution_options=options, **arguments)
                for change in ({"operation": "measure_content_pair_four_updates"},
                               {"source": f.stage.target}, {"target": f.other.target},
                               {"execution_options": {**options, "maximum_bytes": 1}},
                               {"execution_options": {**options, "final_path": f.other.final_path}}):
                    with self.subTest(change=change), self.assertRaises(runner.AdmissionError):
                        runner.resolver(context, **{**arguments, "operation": "train_and_publish_brain_pair",
                                                  "execution_options": options, **change})

    def test_allocation_accepts_both_full_day_formats_and_refuses_mismatch(self):
        fixed_now = datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp()
        def stamp(epoch):
            return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None).isoformat()
        base_env = {**{key: "1" for key in runner.THREADS}, "SLURM_JOB_ID": "12345678", "SLURM_NTASKS": "1",
               "SLURM_NNODES": "1", "CUDA_VISIBLE_DEVICES": "", "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": "",
               "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": ""}
        base_fields = {"JobId": "12345678", "NodeList": "sh04-fixture", "JobState": "RUNNING",
                  "NumTasks": "1", "NumNodes": "1", "Partition": "russpold",
                  "TimeLimit": "1-00:00:00",
                  "StartTime": stamp(fixed_now - 120), "EndTime": stamp(fixed_now + 86280)}
        for stage_name in runner.STAGES:
            with self.subTest(stage=stage_name), stage_files(stage_name) as f, ExitStack() as stack:
                cpus, memory_gib = f.stage.cpus, f.stage.memory_gib
                env = {**base_env, "SLURM_CPUS_PER_TASK": str(cpus),
                       "SLURM_MEM_PER_NODE": str(memory_gib * 1024)}
                fields = {**base_fields, "NumCPUs": str(cpus), "MinMemoryNode": f"{memory_gib}G",
                          "AllocTRES": f"cpu={cpus},node=1,mem={memory_gib}G"}
                stack.enter_context(mock.patch.object(runner.socket, "gethostname", return_value="sh04-fixture.stanford.edu"))
                wall = stack.enter_context(mock.patch.object(runner.time, "time", return_value=fixed_now))
                monotonic = stack.enter_context(mock.patch.object(runner.time, "monotonic", return_value=1000.))
                stack.enter_context(mock.patch.dict(os.environ, env))
                def allocation(changes):
                    (f.run / "allocation.txt").write_text(" ".join(
                        f"{key}={value}" for key, value in {**fields, **changes}.items()))
                    return runner._allocation(f.stage)
                for limit in ("24:00:00", "1-00:00:00"):
                    admitted = allocation({"TimeLimit": limit})
                    self.assertEqual((admitted["minutes"], admitted["cpus"], admitted["memory_gib"]),
                                     (1440, cpus, memory_gib))
                wrong_cpus = 2 if cpus != 2 else 1
                wrong_memory = 7 if memory_gib != 7 else 16
                invalid = [{"TimeLimit": "23:59:59"}, {"NumCPUs": str(wrong_cpus)},
                           {"MinMemoryNode": f"{wrong_memory}G"},
                           {"AllocTRES": f"cpu={wrong_cpus},node=1,mem={memory_gib}G"},
                           {"AllocTRES": f"cpu={cpus},node=1,mem={wrong_memory}G"},
                           {"AllocTRES": fields["AllocTRES"] + ",gres/gpu=1"}, {"NodeList": "other-node"},
                           {"EndTime": "2026-10-11T00:00:00-07:00"}]
                if stage_name == "original":
                    invalid.append({"EndTime": stamp(fixed_now + 84600)})
                else:
                    invalid.extend(({"StartTime": stamp(fixed_now + 1)},
                                    {"StartTime": stamp(fixed_now - 84601)},
                                    {"StartTime": "2026-10-09T23:58:00-07:00"},
                                    {"EndTime": stamp(fixed_now + 60)}))
                for change in invalid:
                    with self.subTest(change=change), self.assertRaises(runner.AdmissionError):
                        allocation(change)
                with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}), self.assertRaises(runner.AdmissionError):
                    allocation({})
                for changed_env in ({"SLURM_CPUS_PER_TASK": str(wrong_cpus)},
                                    {"SLURM_MEM_PER_NODE": str(wrong_memory * 1024)}):
                    with mock.patch.dict(os.environ, changed_env), self.assertRaises(runner.AdmissionError):
                        allocation({})
                scratch = f.root / f.stage.scratch_name
                scratch.mkdir()
                with mock.patch.dict(os.environ, {"L_SCRATCH_JOB": str(f.root), "TMPDIR": str(scratch)}):
                    self.assertEqual(runner._scratch(f.stage), scratch)
                with mock.patch.dict(os.environ, {"L_SCRATCH_JOB": str(f.root),
                                                   "TMPDIR": str(f.root / f.other.scratch_name)}), \
                        self.assertRaises(runner.AdmissionError):
                    runner._scratch(f.stage)
                admitted = allocation({})
                context = runner.BrainPairContext(profile(f.stage), admitted, scratch, f.stage)
                budget = runner.BrainPairBudget(context)
                expected = 84480 if f.stage.allocation_start_deadline else 84600
                self.assertEqual(budget.remaining_seconds(), expected)
                self.assertEqual(budget.resources, f.stage.resources)
                with mock.patch.object(runner.resource, "getrusage",
                        return_value=SimpleNamespace(ru_maxrss=f.stage.rss_limit_bytes // 1024)):
                    budget.check()
                with mock.patch.object(runner.resource, "getrusage",
                        return_value=SimpleNamespace(ru_maxrss=f.stage.rss_limit_bytes // 1024 + 1)), \
                        self.assertRaises(MemoryError):
                    budget.check()
                with mock.patch.object(runner, "admit", return_value=(context, budget)) as admission, \
                        mock.patch.object(runner, "_load_operations") as load, redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(runner.main(["--stage", stage_name, "--remaining-seconds"]), 0)
                    self.assertEqual(out.getvalue(), str(expected) + "\n")
                    admission.assert_called_once_with(stage_name)
                    load.assert_not_called()
                    self.assertFalse((f.run / "claim.json").exists())
                if f.stage.allocation_start_deadline:
                    self.assertEqual(admitted["deadline_epoch"], fixed_now + 84480)
                    self.assertEqual(allocation({"EndTime": stamp(fixed_now + 120)})["deadline_epoch"], fixed_now + 60)
                    wall.return_value, monotonic.return_value = fixed_now + 600, 1600.
                    later_allocation = allocation({})
                    later_context = runner.BrainPairContext(profile(f.stage), later_allocation, scratch, f.stage)
                    later_budget = runner.BrainPairBudget(later_context)
                    self.assertEqual(later_budget.remaining_seconds(), 83880)
                    self.assertEqual(later_budget.deadline, budget.deadline)
                    self.assertEqual(later_budget.absolute_deadline, admitted["deadline_epoch"])
                    wall.return_value = admitted["deadline_epoch"]
                    with self.assertRaises(TimeoutError):
                        later_budget.check()

    def test_one_claim_full_scope_and_safe_success_or_failure_report(self):
        class BrainLossNotEvaluable(RuntimeError):
            pass
        scale_code = "zero_or_nonfinite_required_training_scale"
        cases = [
            ("complete", None, None), ("wrong_native_width", None, None),
            ("wrong_role", None, None), ("wrong_publication_pair", None, None),
            ("fit_failure", RuntimeError("sensitive_CV_values_must_not_be_reported"), None),
            ("closeout_failure", None, None),
            ("known_scale", BrainLossNotEvaluable(scale_code + ":target_fold_0"),
             {"reason_code": scale_code, "role": "target", "context": "fold_training", "fold": 0}),
            ("known_final", BrainLossNotEvaluable(scale_code + ":source_final"),
             {"reason_code": scale_code, "role": "source", "context": "final"}),
            ("known_normalization", BrainLossNotEvaluable("nonfinite_required_standardized_response:source_validation"),
             {"reason_code": "nonfinite_required_standardized_response", "role": "source", "context": "validation"}),
            ("known_input", BrainLossNotEvaluable("nonfinite_required_tau_train_response:sub-03"),
             {"reason_code": "nonfinite_required_tau_train_response", "role": "source"}),
            ("known_solve", BrainLossNotEvaluable("required_float32_primal_solve_failed"),
             {"reason_code": "required_float32_primal_solve_failed"}),
            ("known_mse", BrainLossNotEvaluable("nonfinite_required_validation_MSE:fold_4:alpha_1400"),
             {"reason_code": "nonfinite_required_validation_MSE", "fold": 4, "alpha": 1400}),
            ("unknown", BrainLossNotEvaluable("sensitive_CV_unknown_reason"), None),
            ("malicious_suffix", BrainLossNotEvaluable(scale_code + ":target_fold_0\nsensitive_CV=123"), None),
            ("invalid_fold", BrainLossNotEvaluable(scale_code + ":target_fold_5"), None),
            ("invalid_alpha", BrainLossNotEvaluable("nonfinite_required_validation_MSE:fold_0:alpha_621"), None),
            ("wrong_class", RuntimeError(scale_code + ":target_fold_0"), None),
            ("extra_args", BrainLossNotEvaluable(scale_code + ":target_fold_0", "sensitive_CV=123"), None),
        ]
        for stage_name in runner.STAGES:
            selected = runner._stage(stage_name)
            other = runner._stage("next-001" if stage_name == "original" else "original")
            scoped_cases = cases + [
                ("known_target", BrainLossNotEvaluable("nonfinite_required_tau_train_response:" + selected.target),
                 {"reason_code": "nonfinite_required_tau_train_response", "role": "target"}),
                ("other_target", BrainLossNotEvaluable("nonfinite_required_tau_train_response:" + other.target), None),
            ]
            for outcome, fit_error, expected_details in scoped_cases:
                with self.subTest(stage=stage_name, outcome=outcome), stage_files(stage_name) as f:
                    context, budget = self.context_budget(f.root, f.stage)
                    if outcome == "closeout_failure":
                        budget.check_storage = mock.Mock(side_effect=OSError("fixture closeout failure"))
                    responses = []
                    for person, width in ((f.stage.source, f.stage.source_native_voxels),
                                          (f.stage.target, f.stage.target_native_voxels)):
                        is_target = person == f.stage.target
                        if outcome == "wrong_native_width" and is_target:
                            width -= 1
                        role = "tau_test" if outcome == "wrong_role" and is_target else "tau_train"
                        responses.append(SimpleNamespace(participant=person, role=role, image_ids=tuple(range(897)),
                            native_voxel_ids=range(width), values=SimpleNamespace(shape=(897, width))))
                    bound = SimpleNamespace(source=responses[0], target=responses[1])
                    def bind(actual_context, **kwargs):
                        self.assertIs(actual_context, context)
                        self.assertTrue((f.run / "claim.json").is_file())
                        self.assertIs(kwargs["budget"], budget)
                        self.assertEqual((kwargs["source"], kwargs["target"]), (f.stage.source, f.stage.target))
                        return bound
                    def train(actual_bound, **kwargs):
                        self.assertIs(actual_bound, bound)
                        self.assertEqual(kwargs, {"final_path": f.path, "maximum_bytes": 805306368})
                        if fit_error is not None:
                            raise fit_error
                        f.path.write_bytes(b"fixture_final_artifact")
                        return SimpleNamespace(source=f.stage.source,
                            target=f.other.target if outcome == "wrong_publication_pair" else f.stage.target, path=f.path,
                            summary=SimpleNamespace(available=True, completed_fits=51))
                    with mock.patch.object(runner, "admit", return_value=(context, budget)), \
                            mock.patch.dict(sys.modules, {"brain_loss": SimpleNamespace(BrainLossNotEvaluable=BrainLossNotEvaluable)}), \
                            mock.patch.object(runner, "_load_operations", return_value=(bind, mock.Mock(side_effect=train),
                                {"python": "fixture", "numpy": "fixture", "torch": "fixture"})) as loader:
                        # Original helpers remain valid for the independently admitted diagnostic.
                        if stage_name == "original" and outcome == "complete":
                            runner._bound_scope(bound)
                            self.assertEqual(runner._brain_failure_details(BrainLossNotEvaluable(
                                "nonfinite_required_tau_train_response:sub-01"))["brain_loss_failure"]["role"], "target")
                            self.assertEqual(runner._brain_failure_details(BrainLossNotEvaluable(
                                "nonfinite_required_tau_train_response:sub-05")), {})
                        result = runner.run_once(stage_name)
                        self.assertEqual(result, 0 if outcome == "complete" else 1)
                        report_text = (f.run / "report.json").read_text()
                        report = json.loads(report_text)
                        claim = json.loads((f.run / "claim.json").read_text())
                        self.assertEqual((report["source"], report["target"], report["pair_index"]),
                                         (f.stage.source, f.stage.target, f.stage.pair_index))
                        self.assertEqual(claim["pair_stage"], stage_name)
                        self.assertNotIn("sensitive_CV", report_text)
                        if expected_details is None:
                            self.assertNotIn("brain_loss_failure", report)
                        else:
                            self.assertEqual(report["brain_loss_failure"], expected_details)
                        succeeded = outcome in ("complete", "closeout_failure")
                        uncertain_publication = fit_error is not None or outcome == "wrong_publication_pair"
                        self.assertEqual(report["completed_fits"], 51 if succeeded else None)
                        self.assertEqual(report["model_published"], True if succeeded else None if uncertain_publication else False)
                        self.assertEqual(report["published_path"], str(f.path) if succeeded else None)
                        for key in ("elapsed_wall_seconds", "process_cpu_seconds", "peak_RSS_bytes"):
                            self.assertIn(key, report)
                        expected_calls = 0 if outcome in ("wrong_native_width", "wrong_role") else 1
                        self.assertEqual(loader.return_value[1].call_count, expected_calls)
                        with self.assertRaises((runner.AdmissionError, FileExistsError)):
                            runner.run_once(stage_name)
                        self.assertEqual(loader.call_count, 1)
                    for previous, history, history_before, grant_before in f.histories:
                        self.assertEqual(history.read_bytes(), history_before)
                        self.assertEqual(previous.profile_path.read_bytes(), grant_before)
                        self.assertFalse(previous.final_path.exists())
                        self.assertFalse((previous.run_dir / "claim.json").exists())


if __name__ == "__main__":
    unittest.main()
