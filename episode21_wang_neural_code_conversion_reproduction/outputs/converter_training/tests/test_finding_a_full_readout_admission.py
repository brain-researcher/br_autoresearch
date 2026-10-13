"""Three stdlib-only phase admission/composition fixtures; no scientific inputs."""
from contextlib import ExitStack, contextmanager, redirect_stdout
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
import run_finding_a_full_readout as runner


NOW = datetime(2026, 10, 11, tzinfo=timezone.utc).timestamp()


def profile():
    return deepcopy({**runner.FIXED, "resources": runner.RESOURCES, "input_scope": runner.INPUT_SCOPE,
                     "authorized": True, "approval_source": "explicit_fixture_only"})


def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None).isoformat()


def closed_report(stage, job_id, array_job_id=None):
    stage.run_dir.mkdir(parents=True, exist_ok=True)
    result = {"phase": stage.phase, "batch_index": stage.batch_index,
        "status": "complete_" + stage.phase, "artifacts_published": True,
        "allocation_identity": {"job_id": job_id, "hostname": "sh04-fixture"},
        "array_job_id": array_job_id, "readers_closed": stage.phase == "draw"}
    (stage.run_dir / "report.json").write_text(json.dumps(result))
    status = "" if stage.phase != "decode" else "".join(key + "=0\n" for key in runner.CHECK_RECORDS)
    (stage.run_dir / "batch_status.txt").write_text(status + "phase_exit=0\nstatus=closed\n")


@contextmanager
def files(phase="decode", index=None):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "BASE", Path(directory).resolve()):
        stage = runner._stage(phase, index)
        stage.run_dir.mkdir(parents=True)
        (stage.root / "draw_batches").mkdir(exist_ok=True)
        stage.profile_path.write_text(json.dumps(profile()))
        binding = {"decode": {"job_id": "1000", "afterok_job_ids": [str(2000 + i) for i in range(20)],
                              "completed_model_job_ids": [str(2000 + i) for i in range(20, 32)]},
                   "draw": {"array_job_id": "3000", "afterok_job_id": "1000"},
                   "finalize": {"job_id": "4000", "afterok_job_id": "3000"}}
        (stage.root / "dependency_bindings.json").write_text(json.dumps(binding))
        decode = runner._stage("decode")
        decode.run_dir.mkdir(exist_ok=True)
        (decode.run_dir / "batch_status.txt").write_text("".join(key + "=0\n" for key in runner.CHECK_RECORDS))
        if phase != "decode":
            closed_report(decode, "1000")
            stage.stage_h5.write_bytes(b"published-fixture-stage")
            stage.stage_h5.with_suffix(".decoded_point.json").write_text("{}")
        if phase == "finalize":
            for batch in range(63):
                prior = runner._stage("draw", batch)
                closed_report(prior, str(3100 + batch), "3000")
                prior.batch_path.write_text("{}")
        scratch = runner.BASE / stage.scratch_name
        scratch.mkdir()
        yield SimpleNamespace(stage=stage, binding=binding, scratch=scratch, root=runner.BASE)


@contextmanager
def machine(f):
    phase, index, resources = f.stage.phase, f.stage.batch_index, f.stage.resources
    job = str(3100 + index) if phase == "draw" else f.binding[phase]["job_id"]
    env = {**{key: "1" for key in runner.core.THREADS}, "SLURM_JOB_ID": job,
        "SLURM_CPUS_PER_TASK": str(resources["cpus"]), "SLURM_MEM_PER_NODE": str(resources["memory_gib"] * 1024),
        "SLURM_NTASKS": "1", "SLURM_NNODES": "1", "CUDA_VISIBLE_DEVICES": "",
        "SLURM_GPUS": "", "SLURM_GPUS_ON_NODE": "", "SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": "",
        "SLURM_ARRAY_JOB_ID": "3000" if phase == "draw" else "",
        "SLURM_ARRAY_TASK_ID": str(index) if phase == "draw" else "",
        "SLURM_ARRAY_TASK_COUNT": "63" if phase == "draw" else "",
        "SLURM_ARRAY_TASK_MIN": "0" if phase == "draw" else "",
        "SLURM_ARRAY_TASK_MAX": "62" if phase == "draw" else "",
        "SLURM_ARRAY_TASK_STEP": "1" if phase == "draw" else "",
        "L_SCRATCH_JOB": str(f.root), "TMPDIR": str(f.scratch)}
    fields = {"JobId": job, "NodeList": "sh04-fixture", "JobState": "RUNNING",
        "NumCPUs": str(resources["cpus"]), "NumTasks": "1", "NumNodes": "1", "Partition": "russpold",
        "TimeLimit": str(resources["minutes"] // 60) + ":00:00", "MinMemoryNode": str(resources["memory_gib"]) + "G",
        "AllocTRES": f"cpu={resources['cpus']},node=1,mem={resources['memory_gib']}G",
        "StartTime": stamp(NOW - 120), "EndTime": stamp(NOW - 120 + resources["minutes"] * 60)}
    if phase == "draw":
        fields.update(ArrayJobId="3000", ArrayTaskId=str(index))

    def allocation(changes=None):
        (f.stage.run_dir / "allocation.txt").write_text(" ".join(
            f"{key}={value}" for key, value in {**fields, **(changes or {})}.items()))

    allocation()
    with mock.patch.dict(os.environ, env), \
            mock.patch.object(runner.core.socket, "gethostname", return_value="sh04-fixture.stanford.edu"), \
            mock.patch.object(runner.time, "time", return_value=NOW) as wall, \
            mock.patch.object(runner.time, "monotonic", return_value=1000.) as mono, \
            mock.patch.object(runner.resource, "getrusage", return_value=SimpleNamespace(ru_maxrss=0)) as rss:
        yield SimpleNamespace(allocation=allocation, wall=wall, mono=mono, rss=rss)


class FullFindingAReadoutAdmissionTests(unittest.TestCase):
    def test_phase_scope_dependencies_and_checks_precede_scientific_access(self):
        self.assertEqual(len(runner.CHECK_RECORDS), 10)
        for phase, index in (("decode", None), ("draw", 0), ("draw", 62), ("finalize", None)):
            with self.subTest(phase=phase, index=index), files(phase, index) as f:
                good = profile()
                for modified in ({**good, "authorized": False}, {**good, "decoded_slots": 702},
                                 {**good, "model_policy": "modified"},
                                 {**good, "input_scope": {**good["input_scope"], "draw": {"models": True}}}):
                    f.stage.profile_path.write_text(json.dumps(modified))
                    with mock.patch.object(runner.core, "_allocation") as allocate, \
                            mock.patch.object(runner, "_load_operation") as load, \
                            mock.patch.object(runner, "_configure_file_policy") as configure:
                        with self.assertRaises(runner.AdmissionError):
                            runner.run_once(phase, index)
                        allocate.assert_not_called(); load.assert_not_called(); configure.assert_not_called()
                f.stage.profile_path.write_text(json.dumps(good))
                with machine(f), mock.patch.object(runner, "_require_file_policy"):
                    context, budget = runner.admit(phase, index)
                    request = dict(operation=runner.OPERATIONS[phase], source=None, target=None,
                                   budget=budget, execution_options=runner._options(f.stage))
                    runner.resolver(context, **request)
                    bad = [{"operation": "train_brain_pair"}, {"source": "sub-03"},
                           {"execution_options": {**request["execution_options"], "arbitrary_option": True}}]
                    if phase != "decode":
                        bad.extend({"operation": operation, "execution_options": None} for operation in (
                            "bind_full_finding_a_inputs", "prepare_full_finding_a_models", "read_full_finding_a_truth"))
                    for changed in bad:
                        with self.assertRaises(runner.AdmissionError):
                            runner.resolver(context, **{**request, **changed})
                    if phase == "decode":
                        for operation, source, target in (
                            ("bind_full_finding_a_inputs", None, None), ("prepare_full_finding_a_models", None, None),
                            ("read_full_finding_a_truth", None, None), ("read_full_finding_a_tau_test", "sub-07", None),
                            ("read_full_finding_a_content_model", "sub-01", "sub-03"),
                            ("read_full_finding_a_brain_model", "sub-05", "sub-07")):
                            runner.resolver(context, **{**request, "operation": operation, "source": source,
                                                       "target": target, "execution_options": None})
                        with self.assertRaises(runner.AdmissionError):
                            runner.resolver(context, **{**request, "operation": "read_full_finding_a_brain_model",
                                "source": "sub-01", "target": "sub-03", "execution_options": None})
                    for altered in ("overlap", "missing", "duplicate"):
                        broken = deepcopy(f.binding)
                        if altered == "overlap":
                            broken["decode"]["completed_model_job_ids"][-1] = broken["decode"]["afterok_job_ids"][0]
                        elif altered == "missing":
                            broken["decode"]["completed_model_job_ids"].pop()
                        else:
                            broken["decode"]["afterok_job_ids"][-1] = broken["decode"]["afterok_job_ids"][0]
                        (f.stage.root / "dependency_bindings.json").write_text(json.dumps(broken))
                        with self.assertRaises(runner.AdmissionError):
                            runner.admit(phase, index)
                    for empty_field in ("afterok_job_ids", "completed_model_job_ids"):
                        all_one_kind = deepcopy(f.binding)
                        other = "completed_model_job_ids" if empty_field == "afterok_job_ids" else "afterok_job_ids"
                        all_one_kind["decode"][other] += all_one_kind["decode"][empty_field]
                        all_one_kind["decode"][empty_field] = []
                        (f.stage.root / "dependency_bindings.json").write_text(json.dumps(all_one_kind))
                        runner.admit(phase, index)
                    (f.stage.root / "dependency_bindings.json").write_text(json.dumps(f.binding))
                    checks = f.stage.root / "decode/batch_status.txt"
                    saved = checks.read_text()
                    for changed in (saved.replace("astra_exit=0\n", ""), saved + "draw_batch_checks_exit=0\n",
                                    saved.replace("full_model_checks_exit=0", "full_model_checks_exit=1")):
                        checks.write_text(changed)
                        with mock.patch.object(runner, "_load_operation") as load, \
                                mock.patch.object(runner, "_configure_file_policy") as configure:
                            with self.assertRaises(runner.AdmissionError):
                                runner.run_once(phase, index)
                            load.assert_not_called(); configure.assert_not_called()
                    checks.write_text(saved)
                    if phase == "draw":
                        for changed in ({"SLURM_ARRAY_TASK_ID": str(1 - index)}, {"SLURM_ARRAY_TASK_COUNT": "62"},
                                        {"SLURM_ARRAY_TASK_MIN": "1"}, {"SLURM_ARRAY_TASK_MAX": "63"},
                                        {"SLURM_ARRAY_TASK_STEP": "2"}):
                            with mock.patch.dict(os.environ, changed), self.assertRaises(runner.AdmissionError):
                                runner.admit(phase, index)
                        f.stage.stage_h5.with_suffix(".finalization_started.json").write_text("{}")
                        with self.assertRaises(runner.AdmissionError):
                            runner._completed(f.stage, context.binding)
                    if phase == "finalize":
                        prior = runner._stage("draw", 62).run_dir / "report.json"
                        original = json.loads(prior.read_text())
                        for change in ({"readers_closed": False}, {"status": "incomplete"}, {"batch_index": 61},
                                       {"array_job_id": "other"}, {"allocation_identity": {"job_id": "3100"}}):
                            prior.write_text(json.dumps({**original, **change}))
                            with self.assertRaises(runner.AdmissionError):
                                runner._completed(f.stage, context.binding)
                        prior.write_text(json.dumps(original))
                        runner._completed(f.stage, context.binding)
        for phase, index in (("draw", None), ("draw", True), ("draw", 63), ("decode", 0), ("finalize", 0)):
            with self.assertRaises(runner.AdmissionError):
                runner._stage(phase, index)

    def test_absolute_clocks_fatal_policy_inode_accounting_and_metadata_only_cli(self):
        for phase, index, cpus, gib, minutes, seconds, rss in (
            ("decode", None, 3, 16, 4320, 255600, 16106127360),
            ("draw", 62, 3, 16, 1440, 84600, 16106127360),
            ("finalize", None, 1, 7, 120, 6900, 6979321856)):
            with files(phase, index) as f, machine(f) as actual:
                limits = f.stage.resources
                self.assertEqual((limits["cpus"], limits["memory_gib"], limits["minutes"],
                                  limits["deadline_seconds"], limits["rss_limit_bytes"]),
                                 (cpus, gib, minutes, seconds, rss))
                self.assertEqual((limits["collection_bytes"], limits["aggregate_bytes"], limits["durable_bytes"],
                                  limits["scratch_bytes"], limits["batch_bytes"]),
                                 (824633720832, 833223655424, 67108864, 4294967296, 33554432))
                self.assertEqual(runner.admit(phase, index)[1].remaining_seconds(), seconds - 120)
                for changed in ({"NumCPUs": str(cpus + 1)}, {"MinMemoryNode": "32G"},
                                {"AllocTRES": f"cpu={cpus},node=1,mem={gib}G,gres/gpu=1"},
                                {"StartTime": stamp(NOW + 1)}, {"StartTime": stamp(NOW - seconds - 1)}):
                    actual.allocation(changed)
                    with self.assertRaises(runner.AdmissionError):
                        runner.admit(phase, index)
                actual.allocation({"EndTime": stamp(NOW + 600)})
                self.assertEqual(runner.admit(phase, index)[1].remaining_seconds(), 540)
                actual.allocation()
                context, budget = runner.admit(phase, index)
                actual.rss.return_value = SimpleNamespace(ru_maxrss=rss // 1024 + 1)
                with self.assertRaises(MemoryError):
                    budget.check()
                actual.rss.return_value = SimpleNamespace(ru_maxrss=0)
                # File policy is configured only in execution, never these metadata commands.
                (f.stage.root / "decode/batch_status.txt").unlink()
                args = ["--phase", phase] + ([] if index is None else ["--batch-index", str(index)])
                with mock.patch.object(runner, "_configure_file_policy") as configure, \
                        mock.patch.object(runner, "_load_operation") as load:
                    for elapsed in (0, 300):
                        actual.wall.return_value, actual.mono.return_value = NOW + elapsed, 1000. + elapsed
                        with redirect_stdout(io.StringIO()) as output:
                            self.assertEqual(runner.main(args + ["--remaining-seconds"]), 0)
                        self.assertEqual(output.getvalue(), str(seconds - 120 - elapsed) + "\n")
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(runner.main(args + ["--check-admission"]), 0)
                    configure.assert_not_called(); load.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "stage.h5").write_bytes(b"0123456789")
            os.link(root / "stage.h5", root / "final.h5")
            (root / "metadata.json").write_bytes(b"{}")
            self.assertEqual(runner._bytes_unique(root, 12), 12)
            with self.assertRaises(runner.AdmissionError):
                runner._bytes_unique(root, 11)
            (root / "different.h5").write_bytes(b"0123456789")
            with self.assertRaises(runner.AdmissionError):
                runner._bytes_unique(root, 12)
            (root / "different.h5").unlink()
            (root / "alias").symlink_to(root / "stage.h5")
            with self.assertRaises(runner.AdmissionError):
                runner._bytes_unique(root, 100)
        cap = 824633720832
        with mock.patch.object(runner.resource, "getrlimit", side_effect=[(0, resource.RLIM_INFINITY),
                                                                         (cap, resource.RLIM_INFINITY)]), \
                mock.patch.object(runner.resource, "setrlimit") as setlimit, \
                mock.patch.object(runner.signal, "signal") as setsignal, \
                mock.patch.object(runner.signal, "getsignal", return_value=signal.SIG_DFL):
            runner._configure_file_policy()
            setlimit.assert_called_once_with(resource.RLIMIT_FSIZE, (cap, resource.RLIM_INFINITY))
            setsignal.assert_called_once_with(signal.SIGXFSZ, signal.SIG_DFL)
        with mock.patch.object(runner.resource, "getrlimit", return_value=(cap, cap)), \
                mock.patch.object(runner.signal, "getsignal", return_value=signal.SIG_IGN), \
                self.assertRaises(runner.AdmissionError):
            runner._require_file_policy()

    def test_phase_composition_exclusive_attempts_and_conservative_failure_records(self):
        for phase, index in (("decode", None), ("draw", 62), ("finalize", None)):
            for mode in ("success", "failure", "wrong_path", "closeout_failure"):
                with self.subTest(phase=phase, mode=mode), files(phase, index) as f, machine(f):
                    calls = []

                    def operation(context, **kwargs):
                        self.assertTrue((f.stage.run_dir / "claim.json").is_file())
                        self.assertIs(kwargs["require_admitted_context"], runner.resolver)
                        calls.append(kwargs)
                        if phase == "decode":
                            self.assertEqual(kwargs["authenticated_tau_train_image_ids"], ("admitted-897-metadata",))
                            self.assertEqual(kwargs["stage_h5"], f.stage.stage_h5)
                            self.assertEqual(kwargs["maximum_bytes"], 824633720832)
                        elif phase == "draw":
                            for key, value in runner._options(f.stage).items():
                                self.assertEqual(kwargs[key], value)
                            self.assertNotIn("authenticated_tau_train_image_ids", kwargs)
                        else:
                            self.assertIsNone(kwargs["require_readers_closed"]())
                            self.assertNotIn("authenticated_tau_train_image_ids", kwargs)
                        if mode == "failure":
                            raise ValueError("private_scientific_value_must_not_be_reported")
                        for path in f.stage.publication_paths:
                            path.write_bytes(b"fixture-output")
                        if mode == "closeout_failure":
                            kwargs["budget"].check = mock.Mock(side_effect=MemoryError("private size"))
                        if mode == "wrong_path":
                            return f.stage.root / "wrong.json" if phase == "draw" else (f.stage.root / "wrong.h5",)
                        return f.stage.publication_paths[0] if phase == "draw" else f.stage.publication_paths

                    with mock.patch.object(runner, "_configure_file_policy"), \
                            mock.patch.object(runner, "_require_file_policy"), \
                            mock.patch.object(runner, "_tau_train_ids", return_value=("admitted-897-metadata",)) as ids, \
                            mock.patch.object(runner, "_load_operation", return_value=(operation, {"fixture": True})) as load:
                        code = runner.run_once(phase, index)
                        self.assertEqual(code, 0 if mode == "success" else 1)
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(ids.call_count, int(phase == "decode"))
                        report = json.loads((f.stage.run_dir / "report.json").read_text())
                        self.assertNotIn("private", json.dumps(report))
                        self.assertFalse(report["model_training_performed"])
                        self.assertEqual(report["quantitative_finding"], "not_evaluable")
                        if mode == "success":
                            self.assertEqual(report["status"], "complete_" + phase)
                            self.assertTrue(report["artifacts_published"])
                            self.assertEqual(report["readers_closed"], phase == "draw")
                            self.assertEqual((report["decoded_slots"], report["unavailable_slots"]), (703, 152))
                        elif mode == "closeout_failure":
                            self.assertEqual(report["status"], "incomplete")
                            self.assertTrue(report["artifacts_published"])
                            self.assertTrue(report["resource_limit_failure"])
                        else:
                            self.assertIsNone(report["artifacts_published"])
                            self.assertFalse(report["readers_closed"])
                        with self.assertRaises(runner.AdmissionError):
                            runner.run_once(phase, index)
                        self.assertEqual(load.call_count, 1)
                        self.assertTrue((f.stage.run_dir / "claim.json").is_file())
        # The metadata authenticator reuses the existing source-role function and lock.
        with files() as f, machine(f), mock.patch.object(runner, "_require_file_policy"):
            context, budget = runner.admit("decode")
            ids = tuple(f"image-{index:04d}" for index in range(897))
            protocol = SimpleNamespace(image_roles=mock.Mock(return_value=([],
                {"tau_train": [(i, {"image_id": value}) for i, value in enumerate(reversed(ids))]},
                {"existing_alignment": True}, {})))
            lock_path = f.root / "outputs/protocol_lock.yaml"
            lock = {"state": "valid_prospective_lock", "job_id": "46424955",
                    "regular_image_binding": {"existing_alignment": True}}
            lock_path.write_text(json.dumps(lock))
            with mock.patch.dict(sys.modules, {"feature_protocol": protocol}):
                self.assertEqual(runner._tau_train_ids(context, budget), ids)
                lock["regular_image_binding"] = {"other": True}
                lock_path.write_text(json.dumps(lock))
                with self.assertRaises(runner.AdmissionError):
                    runner._tau_train_ids(context, budget)
            protocol.image_roles.assert_called()


if __name__ == "__main__":
    unittest.main()
