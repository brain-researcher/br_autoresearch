"""Two stdlib admission/composition checks; invented files, coordinator compute.

Reuse the original machine/allocation fixture, not its production resources.
These explicit resource values are test metadata, never a selected grant.
"""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
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
import run_amended_full_readout as runner
import test_finding_a_full_readout_admission as prior


def profile():
    mib, gib = 1024**2, 1024**3
    common = dict(gpus=0, collection_bytes=64*mib, aggregate_bytes=128*mib,
                  durable_bytes=mib, scratch_bytes=mib, batch_bytes=mib)
    resources = {"decode": dict(common, cpus=3, memory_gib=16, minutes=120,
                               deadline_seconds=6900, rss_limit_bytes=15*gib),
                 "draw": dict(common, cpus=2, memory_gib=7, minutes=60,
                              deadline_seconds=3300, rss_limit_bytes=6*gib),
                 "finalize": dict(common, cpus=1, memory_gib=7, minutes=60,
                                  deadline_seconds=3300, rss_limit_bytes=6*gib)}
    return deepcopy({**runner.FIXED, "resources": resources, "input_scope": runner.INPUT_SCOPE,
        "maximum_concurrent_draw_tasks": 2, "authorized": True, "approval_source": "invented_fixture_only"})


def closed_report(stage, job_id, array_job_id=None):
    stage.run_dir.mkdir(parents=True, exist_ok=True)
    (stage.run_dir / "report.json").write_text(json.dumps(dict(phase=stage.phase,
        batch_index=stage.batch_index, status="complete_"+stage.phase, artifacts_published=True,
        model_policy=runner.MODEL_POLICY, decoded_slots=855, unavailable_slots=0,
        allocation_identity=dict(job_id=job_id, hostname="sh04-fixture"),
        array_job_id=array_job_id, readers_closed=stage.phase == "draw")))
    checks = "amended_readout_checks_exit=0\namended_readout_check_methods=8\n" if stage.phase == "decode" else ""
    (stage.run_dir / "batch_status.txt").write_text(checks+"phase_exit=0\nstatus=closed\n")


@contextmanager
def files(phase="decode", index=None):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "BASE", Path(directory).resolve()):
        selected = profile()
        stage = runner._stage(phase, index, selected["resources"][phase])
        stage.run_dir.mkdir(parents=True)
        (stage.root / "draw_batches").mkdir(exist_ok=True)
        stage.profile_path.write_text(json.dumps(selected))
        binding = {"decode": {"job_id": "1000", "afterok_job_ids": [str(2000+i) for i in range(20)],
                              "completed_model_job_ids": [str(2000+i) for i in range(20,40)]},
                   "draw": {"array_job_id": "3000", "afterok_job_id": "1000"},
                   "finalize": {"job_id": "4000", "afterok_job_id": "3000"}}
        (stage.root / "dependency_bindings.json").write_text(json.dumps(binding))
        decode = runner._stage("decode", resources=selected["resources"]["decode"])
        decode.run_dir.mkdir(exist_ok=True)
        (decode.run_dir / "batch_status.txt").write_text("amended_readout_checks_exit=0\namended_readout_check_methods=8\n")
        if phase != "decode":
            closed_report(decode, "1000")
            stage.stage_h5.write_bytes(b"invented-stage")
            stage.stage_h5.with_suffix(".decoded_point.json").write_text("{}")
        if phase == "finalize":
            for batch in range(63):
                item = runner._stage("draw", batch, selected["resources"]["draw"])
                closed_report(item, str(3100+batch), "3000")
                item.batch_path.write_text("{}")
        scratch = runner.BASE / stage.scratch_name
        scratch.mkdir()
        yield SimpleNamespace(stage=stage, binding=binding, scratch=scratch, root=runner.BASE)


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    artifacts: tuple
    model_policy: str = runner.MODEL_POLICY


class AmendedFullReadoutAdmissionTests(unittest.TestCase):
    def test_distinct_grant_resources_all40_jobs_and_phase_scope_precede_readers(self):
        for phase, index in (("decode", None), ("draw", 62), ("finalize", None)):
            with self.subTest(phase=phase), files(phase, index) as f:
                good = profile()
                invalid_resources = deepcopy(good["resources"])
                invalid_resources["decode"]["rss_limit_bytes"] = 16*1024**3
                for change in ({"authorized": False}, {"decoded_slots": 703},
                        {"model_policy": "original_locked"}, {"resources": invalid_resources},
                        {"resources": {}}, {"maximum_concurrent_draw_tasks": True}):
                    f.stage.profile_path.write_text(json.dumps({**good, **change}))
                    with mock.patch.object(runner.core, "_allocation") as allocation, \
                            mock.patch.object(runner, "_load_operation") as load:
                        with self.assertRaises(runner.AdmissionError):
                            runner.run_once(phase, index)
                        allocation.assert_not_called(); load.assert_not_called()
                f.stage.profile_path.write_text(json.dumps(good))
                with prior.machine(f) as actual, mock.patch.object(runner, "_require_file_policy"):
                    context, budget = runner.admit(phase, index)
                    self.assertEqual(context.stage.resources, good["resources"][phase])
                    self.assertEqual(budget.remaining_seconds(), good["resources"][phase]["deadline_seconds"]-120)
                    actual.allocation({"EndTime": prior.stamp(prior.NOW+600)})
                    self.assertEqual(runner.admit(phase,index)[1].remaining_seconds(),540)
                    actual.allocation()
                    actual.rss.return_value = SimpleNamespace(ru_maxrss=good["resources"][phase]["rss_limit_bytes"]//1024+1)
                    with self.assertRaises(MemoryError):
                        budget.check()
                    actual.rss.return_value = SimpleNamespace(ru_maxrss=0)
                    request = dict(operation=runner.OPERATIONS[phase], source=None, target=None,
                        budget=budget, execution_options=runner._options(context.stage))
                    runner.resolver(context, **request)
                    self.assertEqual(request["execution_options"]["model_policy"], runner.MODEL_POLICY)
                    for old_operation in ("prepare_full_finding_a_models", "read_full_finding_a_brain_model",
                                          "run_full_finding_a_decoded_point", "train_and_publish_brain_constant_pair"):
                        with self.assertRaises(runner.AdmissionError):
                            runner.resolver(context, **{**request, "operation":old_operation,"execution_options":None})
                    if phase == "decode":
                        for source,target in runner.PAIRS:
                            runner.resolver(context, operation="read_amended_full_finding_a_brain_model",
                                source=source,target=target,budget=budget,execution_options=None)
                            runner.resolver(context, operation="read_full_finding_a_content_model",
                                source=source,target=target,budget=budget,execution_options=None)
                    else:
                        with self.assertRaises(runner.AdmissionError):
                            runner.resolver(context, operation="bind_full_finding_a_inputs",
                                source=None,target=None,budget=budget,execution_options=None)
                    for mode in ("missing", "duplicate", "array_parent_shorthand"):
                        broken = deepcopy(f.binding)
                        if mode == "missing":
                            broken["decode"]["completed_model_job_ids"].pop()
                        elif mode == "duplicate":
                            broken["decode"]["completed_model_job_ids"][-1] = broken["decode"]["afterok_job_ids"][0]
                        else:
                            broken["decode"]["afterok_job_ids"] = ["2999"]
                        (f.stage.root/"dependency_bindings.json").write_text(json.dumps(broken))
                        with self.assertRaises(runner.AdmissionError):
                            runner.admit(phase,index)
                    (f.stage.root/"dependency_bindings.json").write_text(json.dumps(f.binding))
                    checks = f.stage.root/"decode/batch_status.txt"
                    saved = checks.read_text()
                    for changed in (saved.replace("methods=8", "methods=6"),
                                    saved.replace("checks_exit=0", "checks_exit=1"),
                                    saved+"amended_readout_checks_exit=0\n"):
                        checks.write_text(changed)
                        with mock.patch.object(runner, "_load_operation") as load:
                            with self.assertRaises(runner.AdmissionError):
                                runner.run_once(phase,index)
                            load.assert_not_called()
                    checks.write_text(saved+"astra_exit=1\n")
                    runner._checks_completed(f.stage)  # Tooling status cannot veto scientific execution.
                    if phase != "decode":
                        prior_path = f.stage.root/"decode/report.json"
                        record = json.loads(prior_path.read_text())
                        prior_path.write_text(json.dumps({**record,"model_policy":"original_locked"}))
                        with self.assertRaises(runner.AdmissionError):
                            runner._completed(f.stage,context.binding)
                        prior_path.write_text(json.dumps(record))
                    if phase == "finalize":
                        child = f.stage.root/"draws/batch_062/report.json"
                        record = json.loads(child.read_text())
                        for changed in ({"readers_closed":False}, {"model_policy":"original_locked"},
                                        {"allocation_identity":{"job_id":"3100"}}):
                            child.write_text(json.dumps({**record,**changed}))
                            with self.assertRaises(runner.AdmissionError):
                                runner._completed(f.stage,context.binding)
                        child.write_text(json.dumps(record))
                        # An actual bound array element can have its array's
                        # numeric job ID. Unique child jobs plus exact task
                        # indices are required, not unequal parent/child IDs.
                        child.write_text(json.dumps({**record,"allocation_identity":{"job_id":"3000"}}))
                        runner._completed(f.stage,context.binding)
                        child.write_text(json.dumps(record))
                    with mock.patch.object(runner,"_load_operation") as load, \
                            mock.patch.object(runner,"_configure_file_policy") as configure, redirect_stdout(io.StringIO()):
                        args = ["--phase",phase]+([] if index is None else ["--batch-index",str(index)])
                        self.assertEqual(runner.main(args+["--check-admission"]),0)
                        load.assert_not_called(); configure.assert_not_called()
        cap = profile()["resources"]["decode"]["collection_bytes"]
        stage = runner._stage("decode",resources=profile()["resources"]["decode"])
        with mock.patch.object(runner.resource,"getrlimit",side_effect=[(0,resource.RLIM_INFINITY),(cap,resource.RLIM_INFINITY)]), \
                mock.patch.object(runner.resource,"setrlimit") as setlimit, \
                mock.patch.object(runner.signal,"signal") as setsignal, \
                mock.patch.object(runner.signal,"getsignal",return_value=signal.SIG_DFL):
            runner._configure_file_policy(stage)
            setlimit.assert_called_once_with(resource.RLIMIT_FSIZE,(cap,resource.RLIM_INFINITY))
            setsignal.assert_called_once_with(signal.SIGXFSZ,signal.SIG_DFL)

    def test_typed_phase_policy_publication_fresh_attempts_and_conservative_failures(self):
        for phase,index in (("decode",None),("draw",62),("finalize",None)):
            for mode in ("success","old_tuple","wrong_policy","failure","closeout_failure"):
                with self.subTest(phase=phase,mode=mode), files(phase,index) as f, prior.machine(f):
                    def operation(context, **kwargs):
                        self.assertTrue((f.stage.run_dir/"claim.json").is_file())
                        self.assertIs(kwargs["require_admitted_context"],runner.resolver)
                        self.assertNotIn("model_policy",kwargs)  # Policy is bound by resolver, not a public override.
                        if phase == "decode":
                            self.assertEqual(kwargs["authenticated_tau_train_image_ids"],("897-fixture-identities",))
                        else:
                            self.assertNotIn("authenticated_tau_train_image_ids",kwargs)
                        if phase == "finalize":
                            self.assertIsNone(kwargs["require_readers_closed"]())
                        if mode == "failure":
                            raise TimeoutError("private_scientific_value_not_for_report")
                        for path in f.stage.publication_paths:
                            path.write_bytes(b"invented-output")
                        if mode == "closeout_failure":
                            kwargs["budget"].check = mock.Mock(side_effect=MemoryError("private_size"))
                        if mode == "old_tuple":
                            return f.stage.publication_paths
                        return PhaseResult(phase,f.stage.publication_paths,
                            "original_locked" if mode == "wrong_policy" else runner.MODEL_POLICY)

                    with mock.patch.object(runner,"_configure_file_policy"), \
                            mock.patch.object(runner,"_require_file_policy"), \
                            mock.patch.object(runner,"_tau_train_ids",return_value=("897-fixture-identities",)) as ids, \
                            mock.patch.object(runner,"_load_operation",return_value=(operation,PhaseResult,{"fixture":True})) as load:
                        self.assertEqual(runner.run_once(phase,index),0 if mode == "success" else 1)
                        self.assertEqual(ids.call_count,int(phase == "decode"))
                        report = json.loads((f.stage.run_dir/"report.json").read_text())
                        self.assertNotIn("private",json.dumps(report))
                        self.assertEqual(report["model_policy"],runner.MODEL_POLICY)
                        self.assertFalse(report["model_training_performed"])
                        self.assertFalse(report["automatic_retry"])
                        self.assertEqual(report["comparability"],"not_evaluable")
                        if mode == "success":
                            self.assertEqual(report["status"],"complete_"+phase)
                            self.assertEqual((report["decoded_slots"],report["unavailable_slots"]),(855,0))
                            self.assertTrue(report["artifacts_published"])
                            self.assertEqual(report["readers_closed"],phase == "draw")
                        elif mode == "closeout_failure":
                            self.assertEqual(report["status"],"incomplete")
                            self.assertTrue(report["artifacts_published"])
                            self.assertTrue(report["resource_limit_failure"])
                        else:
                            self.assertIsNone(report["artifacts_published"])
                        with self.assertRaises(runner.AdmissionError):
                            runner.run_once(phase,index)
                        self.assertEqual(load.call_count,1)
        with files() as f, prior.machine(f), mock.patch.object(runner,"_load_operation") as load:
            Path(str(f.stage.stage_h5)+".partial").write_bytes(b"earlier-stopped-attempt")
            with self.assertRaisesRegex(runner.AdmissionError,"existing_amended_readout_phase_no_retry"):
                runner.run_once("decode")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
