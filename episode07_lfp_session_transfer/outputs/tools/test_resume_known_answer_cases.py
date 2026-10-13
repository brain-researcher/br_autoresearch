"""Two bounded continuation/accounting checks; coordinated Slurm execution only.

Fits and case construction are fabricated mocks. These tests never qualify a
selected model or read empirical outcomes, and never repeat the recovery suite.
"""
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from finalist_controls import resolve_finalist
from regression import RegressionNotConverged
import known_answer_cases as fixed
import resume_known_answer_cases as repair


def check():
    pass


def finalists():
    return (resolve_finalist("adaptive_07", stream="history_assisted"),
            resolve_finalist("adaptive_03", stream="today_only"))


def prior_report(specs):
    return {"state": "unresolved_qualification", "stage": "member_fit",
        "known_answer_recovery_complete": False, "held_out_opened": False,
        "scientific_checks_complete": False, "evidence_stage": "generated_known_answer_only",
        "selection": repair.SELECTION, "animal": "Mihili", "target_day": "20140304",
        "cases": list(fixed.CASES), "seed": fixed.SEED, "internal_limit_seconds": 600,
        "allocated_cpus": 1, "numerical_threads": 1,
        "attempted_member_fits": 3, "completed_member_predictions": 2,
        "wall_seconds": repair.PRIOR_WALL_SECONDS, "process_cpu_seconds": repair.PRIOR_CPU_SECONDS,
        "finalists": {value.stream: value.candidate_id for value in specs},
        "case": "stable", "region": "M1", "stream": "today_only", "member_id": "starting_09",
        "error_type": "RegressionNotConverged", "error": repair.PRIOR_ERROR,
        "feasibility": repair._json(fixed.qualify_finalists(specs)),
        "results": [{"case": "stable", "region": "M1", "stream": "history_assisted",
            "finalist_id": "adaptive_07", "oracle_h_amplitude": 4., "oracle_beta": 8.,
            "prediction_beta": 7.99402421327968, "nrmse": .0011962451159235926,
            "slope_error": .005975786720320286, "slope_limit": 1.6, "passed": True}]}


@dataclass(frozen=True)
class Prediction:
    values: object


def case_fixture(name, *, check):
    cells = []
    for region in ("M1", "PMd"):
        for stream in fixed.STREAMS:
            identity = (name, region, stream)
            cells.append(fixed.CaseCell(region, stream, identity,
                SimpleNamespace(identity=identity), None, None, None))
    return fixed.KnownAnswerCase(name, tuple(cells))


def result_fixture(cell, prediction, *, case, finalist):
    oracle = tuple(fixed.member_oracle(member, case, stream=cell.stream) for member in finalist.members)
    amplitude = math.fsum(member.weight * value[0] for member, value in zip(finalist.members, oracle))
    beta = math.fsum(member.weight * value[1] for member, value in zip(finalist.members, oracle))
    return fixed.RecoveryResult(case, cell.region, cell.stream, finalist.candidate_id,
                                amplitude, beta, beta, 0., 0., .20 * max(abs(beta), 1.), True)


def mock_execution(stack, calls, *, numerical_failure=False, scientific_failure=False):
    def fit(days, *, recipe, check, weight_recipe=None):
        calls.append((days, recipe, weight_recipe))
        if numerical_failure:
            raise RegressionNotConverged(100000, .01, gram_kkt_error=.01, tolerance=1e-5)
        return SimpleNamespace(recipe=recipe)
    def predict(fitted, development, *, check):
        return Prediction(np.ones((1, 1, 1)))
    def evaluate(cell, prediction, *, case, finalist):
        result = result_fixture(cell, prediction, case=case, finalist=finalist)
        return replace(result, nrmse=.25, passed=False) if scientific_failure else result
    stack.enter_context(patch.object(fixed, "generate_case", side_effect=case_fixture))
    stack.enter_context(patch.object(fixed, "_fitted_diagnostics", return_value=None))
    stack.enter_context(patch.object(fixed, "_prediction_values", side_effect=lambda cell, pred: pred.values))
    stack.enter_context(patch.object(fixed, "evaluate_prediction", side_effect=evaluate))
    stack.enter_context(patch("primary_model.fit_primary_model", side_effect=fit))
    stack.enter_context(patch("primary_model.predict_primary_development", side_effect=predict))
    stack.enter_context(patch("weighted_primary_model.fit_weighted_primary_model", side_effect=fit))
    stack.enter_context(patch("weighted_primary_model.predict_weighted_primary_development", side_effect=predict))


def add_timing(report, wall=100., cpu=90.):
    report.update(continuation_wall_seconds=wall, continuation_process_cpu_seconds=cpu,
                  cumulative_wall_seconds=repair.PRIOR_WALL_SECONDS + wall,
                  cumulative_wall_charged_seconds=14. + wall,
                  cumulative_process_cpu_seconds=repair.PRIOR_CPU_SECONDS + cpu)
    return report


class KnownAnswerContinuationTests(unittest.TestCase):
    def test_remaining_roster_single_failed_retry_exact_reuse_and_solver_only_override(self):
        specs = finalists()
        prior = prior_report(specs)
        untouched = copy.deepcopy(prior)
        calls, events = [], []
        with ExitStack() as stack:
            mock_execution(stack, calls)
            report = repair.run_remaining_cases(specs, prior, check=check, progress=events.append)
        self.assertEqual(prior, untouched)
        self.assertEqual(len(calls), 30)
        self.assertEqual((report["attempted_member_fits"], report["completed_member_predictions"]), (33, 32))
        self.assertEqual(report["new_member_fit_attempts"], 30)
        self.assertEqual(len(report["results"]), 16)
        self.assertEqual(report["results"][0], prior["results"][0])
        self.assertEqual((events[0]["attempted_member_fits"], events[0]["completed_member_predictions"]), (3, 2))
        expected_keys = [key for key in repair._keys(specs) if key[:3] != repair.REUSED_CELL]
        changed = 0
        for key, (identity, received, weighting) in zip(expected_keys, calls):
            self.assertEqual(identity, key[:3])
            member = repair._member(specs, key)
            self.assertEqual(weighting, member.weight_recipe)
            if key[2:] == ("today_only", "starting_09"):
                changed += 1
                self.assertEqual(received.regression.max_iterations, 100000)
                self.assertEqual(replace(received, regression=replace(received.regression, max_iterations=1000)),
                                 member.recipe)
            else:
                self.assertEqual(received, member.recipe)
        self.assertEqual(changed, 8)
        ledger = report["member_task_ledger"]
        self.assertEqual(sum(record["reused_completed_prediction"] for record in ledger), 2)
        self.assertEqual([tuple(record["key"]) for record in ledger if len(record["attempts"]) == 2],
                         [repair.FAILED_KEY])
        repair.validate_completed_report(add_timing(report), prior, specs)
        repair.validate_completed_report(add_timing(copy.deepcopy(report), wall=586.), prior, specs)

    def test_admission_and_completed_validation_reject_mutation_and_any_new_failure_stops(self):
        specs, calls = finalists(), []
        prior = prior_report(specs)
        with ExitStack() as stack:
            mock_execution(stack, calls)
            valid = add_timing(repair.run_remaining_cases(specs, prior, check=check))
        invalid_reports = []
        for name, value in (("seed", 1), ("attempted_member_fits", 32), ("new_member_fit_attempts", 31),
                            ("completed_member_predictions", 31), ("prior_wall_charged_seconds", 0.)):
            bad = copy.deepcopy(valid)
            bad[name] = value
            invalid_reports.append(bad)
        bad = copy.deepcopy(valid)
        bad["results"][1] = copy.deepcopy(bad["results"][0])
        invalid_reports.append(bad)
        bad = copy.deepcopy(valid)
        bad["results"][0]["nrmse"] = 0.
        invalid_reports.append(bad)
        bad = copy.deepcopy(valid)
        bad["finalist_specs"]["today_only"]["members"][0]["recipe"]["regression"]["tolerance"] = .01
        invalid_reports.append(bad)
        bad = copy.deepcopy(valid)
        bad["solver_override"]["to"] = 1000000
        invalid_reports.append(bad)
        bad = copy.deepcopy(valid)
        bad["member_task_ledger"][2]["attempts"].append(copy.deepcopy(bad["member_task_ledger"][2]["attempts"][-1]))
        invalid_reports.append(bad)
        invalid_reports.append(add_timing(copy.deepcopy(valid), wall=586.001))
        for bad in invalid_reports:
            with self.assertRaises(ValueError):
                repair.validate_completed_report(bad, prior, specs)
        for change in ({"state": "scientific_recovery_failed"}, {"attempted_member_fits": 4},
                       {"seed": 1}, {"results": prior["results"] * 2}):
            bad_prior = copy.deepcopy(prior)
            bad_prior.update(change)
            with self.assertRaises(ValueError):
                repair.run_remaining_cases(specs, bad_prior, check=check)
        calls, events = [], []
        with ExitStack() as stack:
            mock_execution(stack, calls, numerical_failure=True)
            with self.assertRaises(RegressionNotConverged):
                repair.run_remaining_cases(specs, prior, check=check, progress=events.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[-1]["state"], "unresolved_qualification")
        self.assertEqual((events[-1]["attempted_member_fits"], events[-1]["completed_member_predictions"]), (4, 2))
        self.assertEqual(events[-1]["results"], prior["results"])
        calls = []
        with ExitStack() as stack:
            mock_execution(stack, calls, scientific_failure=True)
            failed = repair.run_remaining_cases(specs, prior, check=check)
        self.assertEqual(len(calls), 2)
        self.assertEqual(failed["state"], "scientific_recovery_failed")
        self.assertFalse(failed["known_answer_recovery_complete"])
        self.assertEqual((failed["attempted_member_fits"], failed["completed_member_predictions"]), (5, 4))


if __name__ == "__main__":
    unittest.main()
