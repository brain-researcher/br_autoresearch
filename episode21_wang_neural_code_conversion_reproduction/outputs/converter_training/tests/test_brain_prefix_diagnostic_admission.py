"""Two parameterized stdlib fixtures; coordinator Slurm execution only.

Both parents and scientific operations are explicit stubs. These checks open
no tau data. The CPU parent's actual resource admission has its own fixtures.
"""
import builtins
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_brain_prefix_diagnostic as entry


def profile(stage_name="original"):
    return {**entry.FIXED,
            "parent_authorization": entry._stage(stage_name).parent_authorization,
            "authorized": True, "approval_source": "fixture_separate_tau_diagnostic_approval",
            "resources": dict(entry.RESOURCES), "input_scope": dict(entry.INPUT_SCOPE)}


@contextmanager
def fixture(stage_name="original"):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary).resolve()
        old_run, new_run = root / "attempt001", root / "attempt002"
        old_run.mkdir()
        new_run.mkdir()
        historical = old_run / "status.txt"
        historical.write_text("parent_job=47360846\ndiagnostic_exit=2\nstatus=closed\n")
        old_grant, new_grant = root / "old.json", root / "new.json"
        for key, value in (("PROFILE", old_grant), ("RUN_DIR", old_run),
                           ("RECONSTRUCTION_PROFILE", new_grant),
                           ("RECONSTRUCTION_RUN_DIR", new_run)):
            stack.enter_context(mock.patch.object(entry, key, value))
        old_grant.write_text(json.dumps(profile("original")))
        new_grant.write_text(json.dumps(profile("reconstruction-002")))
        selected = entry._stage(stage_name)
        other = entry._stage("reconstruction-002" if stage_name == "original" else "original")
        parent_context = object()
        parent_budget = SimpleNamespace(context=parent_context,
            allocation={"job_id": "123", "hostname": "sh03-12n01"},
            check=mock.Mock(), check_storage=mock.Mock())
        stack.enter_context(mock.patch.object(entry.time, "monotonic", return_value=1000.))
        stack.enter_context(mock.patch.object(entry.resource, "getrusage",
            return_value=SimpleNamespace(ru_maxrss=1024)))
        cuda_admit = stack.enter_context(mock.patch.object(entry.parent, "admit",
            return_value=(parent_context, parent_budget)))
        cpu_admit = mock.Mock(return_value=(parent_context, parent_budget))
        stack.enter_context(mock.patch.dict(sys.modules,
            {"run_reconstruction_qualification": SimpleNamespace(admit=cpu_admit)}))
        yield SimpleNamespace(run=selected.run_dir, grant=selected.profile_path,
            selected=selected, other=other, historical=historical,
            parent_context=parent_context, parent_budget=parent_budget,
            cuda_admit=cuda_admit, cpu_admit=cpu_admit)


def bound_stub(context, resolver, budget):
    # Shapes/identities only: no response array is constructed or read.
    def response(participant, width):
        return SimpleNamespace(participant=participant, role="tau_train",
            image_ids=tuple("image_%04d" % i for i in range(897)),
            native_voxel_ids=range(width), values=SimpleNamespace(shape=(897, width)))
    return SimpleNamespace(source=response("sub-03", 12897),
        target=response("sub-01", 12295), execution_context=context,
        require_admitted_context=resolver, budget=budget)


class BrainPrefixAdmissionTests(unittest.TestCase):
    def test_separate_tau_grant_refuses_before_parent_or_scientific_imports(self):
        self.assertEqual(entry._stage().name, "original")
        self.assertEqual(entry._stage().run_dir,
                         entry.BASE / "outputs/runs/brain_pair_prefix_diagnostic/attempt001")
        self.assertEqual(entry._stage("reconstruction-002").run_dir,
                         entry.BASE / "outputs/runs/brain_pair_prefix_diagnostic/attempt002")
        self.assertEqual(entry._stage("reconstruction-002").profile_path,
                         entry.BASE / "outputs/brain_prefix_diagnostic_reconstruction_authorization.json")
        with self.assertRaises(entry.parent.AdmissionError):
            entry.run_once("caller_supplied_other_path")
        real_import = builtins.__import__
        forbidden_imports = []

        def guarded_import(name, *args, **kwargs):
            if name in ("empirical_brain_inputs", "brain_prefix_diagnostic", "brain_loss",
                        "run_reconstruction_qualification"):
                forbidden_imports.append(name)
                raise AssertionError("parent or scientific import preceded separate admission")
            return real_import(name, *args, **kwargs)

        for stage_name in entry.STAGES:
            valid = profile(stage_name)
            invalid = [
                {**valid, "authorized": False}, {**valid, "approval_source": ""},
                {**valid, "operation": "train_and_publish_brain_pair"},
                {**valid, "source": "sub-05"}, {**valid, "fits": 1},
                {**valid, "input_scope": {**entry.INPUT_SCOPE, "tau_test": True}},
                {**valid, "resources": {**entry.RESOURCES, "deadline_seconds": 301}},
                {**valid, "parent_authorization": profile(
                    "reconstruction-002" if stage_name == "original" else "original"
                )["parent_authorization"]},
            ]
            with fixture(stage_name) as f, mock.patch("builtins.__import__", side_effect=guarded_import):
                for changed in invalid:
                    with self.subTest(stage=stage_name, changed=changed):
                        f.grant.write_text(json.dumps(changed))
                        with self.assertRaises(entry.parent.AdmissionError):
                            entry.run_once(stage_name)
                        f.cuda_admit.assert_not_called()
                        f.cpu_admit.assert_not_called()
                        f.parent_budget.check.assert_not_called()
                        self.assertFalse((f.run / "claim.json").exists())
                        self.assertFalse((f.run / "report.json").exists())
                        self.assertFalse((f.other.run_dir / "claim.json").exists())
        self.assertEqual(forbidden_imports, [])

    def test_resolver_and_composition_retain_diagnostic_only_outcomes(self):
        for stage_name in entry.STAGES:
            with self.subTest(stage=stage_name), fixture(stage_name) as f:
                context = entry.DiagnosticContext(entry._profile(f.selected), f.parent_context, f.selected)
                budget = entry.DiagnosticBudget(context, f.parent_budget)
                admitted = dict(source="sub-03", target="sub-01", budget=budget,
                                execution_options=None)
                for operation in ("bind_brain_inputs", "diagnose_brain_first_fold"):
                    self.assertIsNone(entry.resolver(context, operation=operation, **admitted))
                for changed in (
                    {"operation": "train_and_publish_brain_pair"},
                    {"operation": "score_brain_pair"},
                    {"source": "sub-05"}, {"target": "sub-03"},
                    {"execution_options": {"fits": 1}},
                    {"budget": SimpleNamespace(context=context)},
                ):
                    with self.subTest(changed=changed), self.assertRaises(entry.parent.AdmissionError):
                        entry.resolver(context, **{**admitted,
                            "operation": "diagnose_brain_first_fold", **changed})
                for other_context in (f.parent_context,
                        entry.DiagnosticContext(context.profile, f.parent_context, f.selected)):
                    with self.assertRaises(entry.parent.AdmissionError):
                        entry.resolver(other_context, operation="bind_brain_inputs", **admitted)
                with mock.patch.object(f.parent_budget, "context", object()):
                    with self.assertRaises(entry.parent.AdmissionError):
                        entry.resolver(context, operation="bind_brain_inputs", **admitted)
                forged = entry.DiagnosticStage(stage_name, f.other.profile_path,
                    f.other.run_dir, f.selected.parent_authorization)
                forged_context = entry.DiagnosticContext(context.profile, f.parent_context, forged)
                forged_budget = entry.DiagnosticBudget(forged_context, f.parent_budget)
                with self.assertRaises(entry.parent.AdmissionError):
                    entry.resolver(forged_context, operation="bind_brain_inputs",
                                   **{**admitted, "budget": forged_budget})
                f.grant.write_text(json.dumps({**profile(stage_name), "authorized": False}))
                with self.assertRaises(entry.parent.AdmissionError):
                    entry.resolver(context, operation="diagnose_brain_first_fold", **admitted)

        class BrainLossNotEvaluable(RuntimeError):
            pass

        sentinel = "PRIVATE_NEURAL_VALUE_MUST_NOT_APPEAR"
        cases = (
            ("not_localized", None, 0),
            ("localized", BrainLossNotEvaluable(
                "zero_or_nonfinite_required_training_scale:source_fold_0"), 0),
            ("incomplete", RuntimeError(sentinel), 1),
        )
        for stage_name in entry.STAGES:
            for status, error, expected_exit in cases:
                with self.subTest(stage=stage_name, status=status), fixture(stage_name) as f:
                    calls = []
                    history_before = f.historical.read_bytes()
                    other_grant_before = f.other.profile_path.read_bytes()

                    def bind(context, *, require_admitted_context, source, target, budget):
                        self.assertTrue((f.run / "claim.json").exists())
                        self.assertIs(require_admitted_context, entry.resolver)
                        require_admitted_context(context, operation="bind_brain_inputs",
                            source=source, target=target, budget=budget, execution_options=None)
                        calls.append("bind")
                        return bound_stub(context, require_admitted_context, budget)

                    def diagnose(bound):
                        calls.append("diagnose")
                        for operation in ("diagnose_brain_first_fold", "bind_brain_inputs"):
                            bound.require_admitted_context(bound.execution_context,
                                operation=operation, source=bound.source.participant,
                                target=bound.target.participant, budget=bound.budget, execution_options=None)
                        print(sentinel)
                        print(sentinel, file=sys.stderr)
                        if error is not None:
                            raise error
                        return {"status": "not_localized", "completed_prefix": "first_fold_normalization",
                                "fold": 0, "fits_performed": 0, "scores_produced": False,
                                "model_published": False}

                    modules = {
                        "empirical_brain_inputs": SimpleNamespace(bind_brain_inputs=bind),
                        "brain_prefix_diagnostic": SimpleNamespace(diagnose_brain_first_fold=diagnose),
                        "brain_loss": SimpleNamespace(BrainLossNotEvaluable=BrainLossNotEvaluable),
                    }
                    with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()) as out, \
                            redirect_stderr(io.StringIO()) as err:
                        self.assertEqual(entry.run_once(stage_name), expected_exit)
                        if stage_name == "original":
                            f.cuda_admit.assert_called_once_with()
                            f.cpu_admit.assert_not_called()
                        else:
                            f.cpu_admit.assert_called_once_with(component="brain_prefix")
                            f.cuda_admit.assert_not_called()
                        report_text = (f.run / "report.json").read_text()
                        report = json.loads(report_text)
                        claim = json.loads((f.run / "claim.json").read_text())
                        self.assertEqual(report["status"], status)
                        self.assertEqual(report["fits_performed"], 0)
                        self.assertIs(report["scores_produced"], False)
                        self.assertIs(report["model_published"], False)
                        self.assertEqual(report["diagnostic_stage"], stage_name)
                        self.assertEqual(claim["parent_authorization"], f.selected.parent_authorization)
                        self.assertIs(claim["additional_allocation_charge"], False)
                        self.assertNotIn(sentinel, report_text + out.getvalue() + err.getvalue())
                        if status == "localized":
                            self.assertEqual(report["brain_loss_failure"]["reason_code"],
                                             "zero_or_nonfinite_required_training_scale")
                        else:
                            self.assertNotIn("brain_loss_failure", report)
                        if error is not None:
                            self.assertEqual(report["failed_stage"], "first_fold_normalization")
                        with self.assertRaises(entry.parent.AdmissionError):
                            entry.run_once(stage_name)
                        self.assertEqual(calls, ["bind", "diagnose"])
                    self.assertEqual(f.historical.read_bytes(), history_before)
                    self.assertEqual(f.other.profile_path.read_bytes(), other_grant_before)
                    self.assertFalse((f.other.run_dir / "claim.json").exists())
                    self.assertFalse((f.other.run_dir / "report.json").exists())
                    expected_names = {"claim.json", "report.json"}
                    if stage_name == "original":
                        expected_names.add("status.txt")
                    self.assertEqual({path.name for path in f.run.iterdir()}, expected_names)


if __name__ == "__main__":
    unittest.main()
