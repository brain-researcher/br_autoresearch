"""Focused repair-aware MZ menu admission tests; execute only on compute."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_repaired_menu_wave as pool
import test_mz_mvc_convergence_repair as fixtures


class RepairedMenuWaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ConvergenceRepairTests.setUpClass()
        cls.runner_fixture = fixtures.ConvergenceRepairTests(
            "test_first_half_exhaustion_freshly_fits_both_halves_with_original_baseline_replay")
        cls.menu_fixture = cls.runner_fixture.menu_fixture
        cls.fixture, cls.spec = cls.runner_fixture.fixture, cls.runner_fixture.spec

    def panels(self, index):
        if index == 42:
            return dict(zip(("baseline", *pool.saved.RIVALS, "failed", "previous"), self.menu_fixture.reports()))
        base = self.runner_fixture.source(index)
        base["job_id"] = "70000111"
        reports = {"baseline": base}
        original_template = self.menu_fixture.reports()[1]
        for model in pool.saved.RIVALS:
            report = deepcopy(original_template)
            report.update(fitted_model=model, replicate_index=index, seed=pool.saved.cell_seed("MZ", index),
                          source_dir=str(pool.repair.saved.source_directory(index)), source_finished_utc=base["finished_utc"],
                          completed_rivals=[model], fits=[self.fixture.fit(model, train, held) for train, held in pool.saved.HALVES],
                          participant_scores={person: {**row, model: -2.05} for person, row in base["participant_scores"].items()})
            reports[model] = report
        if index in pool.ORIGINAL_INDICES:
            return reports
        reuse = index in pool.REVERSE_INDICES
        failed = self.runner_fixture.failed(index, reuse)
        runtime, metadata = pool.repair.repair_context(index, self.spec, failed)
        metadata["source_and_preserved_fit_validated_against_actual_cohort"] = True
        fits = []
        for half, (train, held) in enumerate(pool.saved.HALVES):
            if reuse and half == 0:
                fit = deepcopy(failed["fits"][0])
                fit.update(elapsed_seconds=.002, repair_execution="reused_validated_original_fit", fit_maximum_sweeps=500,
                           original_fit_elapsed_seconds=failed["fits"][0]["elapsed_seconds"])
                metadata.update(reused_first_fit_materialization_elapsed_seconds=.002,
                                reused_first_fit_rescore_elapsed_seconds=.01, reused_first_fit_rescored=True)
            else:
                fit = self.fixture.fit("MVC", train, held)
                fit.update(objective_history=[float(value) for value in range(1067, 1, -1)] + [2.],
                           elapsed_seconds=20., repair_execution="fresh_fit", fit_maximum_sweeps=1500)
            fits.append(fit)
        reports["MVC"].update(
            phase=pool.repair.PHASE, job_id="46597025" if index in (2, 3) else "70000222",
            source_report="single_process_recovery.json", failed_report=str(pool.repair.failed_path(index)),
            source_job_id=base["job_id"], source_baseline_validated=True, repair_context_validated=True,
            runtime_implementation=runtime, numerical_repair=metadata, fits=fits,
            simulation_copied=False, old_synthetic_responses_used=False,
            selected_replicates=[index], selected_models=["MVC"], fixed_cohort_count=200,
            component_scope="single_MVC_numerical_repair", wave_selected=False, retry_wave_selected=False,
            automatic_retry=False, full_menu_wave_selected=False,
            internal_seconds=6600 if reuse else 12600, elapsed_seconds=1000.,
            cumulative_process_elapsed_seconds=failed["elapsed_seconds"] + 1000.)
        reports["failed"] = failed
        return reports

    def technical(self, panels, *, unvalidated=False):
        report = panels["MVC"]
        report.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                      error="RuntimeError: Internal whole-process deadline exceeded", completed_rivals=[], diagnostics={})
        report.pop("participant_scores")
        if unvalidated:
            report.update(repair_context_validated=False, source_baseline_validated=False, source_job_id=None,
                          saved_scores_replayed=False, fits=[], scored_primary_directions=[],
                          numerical_repair={"kind": "maximum_sweeps_only", "original_maximum_sweeps": 500,
                              "runtime_maximum_sweeps": 1500, "nonconverged_checkpoint_state_reused": False,
                              "source_and_preserved_fit_validated_against_actual_cohort": False})
            for key in ("implementation", "runtime_implementation", "source_finished_utc", "cumulative_process_elapsed_seconds"):
                report.pop(key)
        else:
            reuse = report["replicate_index"] in pool.REVERSE_INDICES
            report["fits"] = report["fits"][:2 if reuse else 1]
            report["scored_primary_directions"] = [{"training_runs": [1, 4], "scoring_runs": [2, 3]}] if reuse else []
            report["fits"][-1].update(converged=False, objective_history=[float(value) for value in range(1500, 0, -1)])
        return panels

    def invoke(self, indices, panels, *, fresh=46597540, reverse=46597546, missing=(), output=None):
        allowed = {pool.IMPLEMENTATION: json.dumps(self.spec)}
        for index, reports in panels.items():
            paths = pool.input_paths(index, fresh, reverse)
            allowed.update({paths[name]: json.dumps(report) for name, report in reports.items()})
        reads = []
        original = pool.saved.validate_rival
        snapshots = deepcopy(panels)

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in missing:
                raise FileNotFoundError(path)
            return allowed[path]

        with patch.object(Path, "read_text", reader), patch.object(pool, "require_compute"), redirect_stdout(io.StringIO()):
            try:
                if output is None:
                    result = pool.pool(indices, self.spec, fresh, reverse)
                else:
                    args = ["--terminal-indices", ",".join(map(str, indices)), "--implementation", str(pool.IMPLEMENTATION),
                            "--output", str(output)]
                    if fresh is not None:
                        args += ["--fresh-array-job-id", str(fresh)]
                    if reverse is not None:
                        args += ["--reverse-array-job-id", str(reverse)]
                    self.assertEqual(pool.main(args), 0)
                    result = None
            finally:
                self.assertIs(pool.saved.validate_rival, original)
        self.assertEqual(panels, snapshots)
        return json.loads(output.read_text()) if output is not None else result, reads

    def test_exact_original_initial_and_two_array_paths_preserve_held_and_unselected_sources(self):
        self.assertEqual(tuple(pool.input_paths(0).values()), pool.wave.input_paths(0))
        self.assertEqual(tuple(pool.input_paths(42).values()), pool.repaired42.INPUT_REPORTS)
        for index in (92, 105):
            self.assertEqual(tuple(pool.input_paths(index).values()), pool.wave.input_paths(index))
        for index in (2, 3):
            self.assertIn(f"repair-cases-46597025/replicate{index}/", str(pool.input_paths(index)["MVC"]))
        for index, array in ((4, 46597540), (11, 46597546), (199, 46597540)):
            paths = pool.input_paths(index, 46597540, 46597546)
            self.assertIn(f"repair-wave-{array}/replicate{index}/", str(paths["MVC"]))
            self.assertEqual(paths["failed"], pool.wave.input_paths(index)[-1])
            self.assertEqual(tuple(paths[name] for name in ("baseline", "MC", "MV", "MVZ")), pool.wave.input_paths(index)[:-1])
        self.assertIsNone(pool.input_paths(1, 46597540, 46597546))
        self.assertIsNone(pool.input_paths(4))
        self.assertIsNone(pool.input_paths(11, 46597540))
        for index in (-1, 200, True):
            with self.assertRaises(ValueError):
                pool.input_paths(index)
        with patch.object(Path, "read_text", side_effect=AssertionError("No selected admitted source")):
            result = pool.pool([1, 4, 11], self.spec)
        self.assertEqual(result["diagnostic_counts"]["unknown_menus"], 200)

    def test_both_repair_modes_reuse_nine_contrasts_and_correct_original_runtime_fit_validation(self):
        for index in (2, 3):
            panels = self.panels(index)
            snapshot = deepcopy(panels)
            original = pool.saved.validate_rival
            with patch.object(pool.fixed, "nested_contrast", wraps=pool.fixed.nested_contrast) as contrasts, \
                    patch.object(pool.fixed, "validate_fit", wraps=pool.fixed.validate_fit) as fits:
                result = pool.build_repaired_summary(*(panels[name] for name in ("baseline", *pool.saved.RIVALS, "failed")), self.spec, index)
            self.assertEqual(panels, snapshot)
            self.assertIs(pool.saved.validate_rival, original)
            self.assertEqual(contrasts.call_count, 9)
            self.assertEqual([(call.args[1], call.args[2]) for call in contrasts.call_args_list], list(pool.fixed.EDGES))
            self.assertEqual([call.args[4]["optimizer"]["maximum_sweeps"] for call in fits.call_args_list[-2:]],
                             [500, 1500] if index == 3 else [1500, 1500])
            self.assertEqual(result["numerical_repair"], panels["MVC"]["numerical_repair"])
            self.assertEqual(result["cumulative_process_elapsed_seconds"], panels["failed"]["elapsed_seconds"] + 1000.)
            self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ"])

    def test_pool_preserves_original_builders_repaired42_and_fixed200_counts(self):
        panels = {index: self.panels(index) for index in (2, 3, 42, 92, 105)}
        panels[0] = {name: {"original_zero_panel": name} for name in pool.input_paths(0)}
        zero_summary = {"maximal_eligible_models_diagnostic": ["MZ"], "multiple_incomparable_eligible_models": False,
                        "nested_comparisons": {}, "family_selection": "not_evaluated"}
        with patch.object(pool.fixed, "validate_base", return_value={}), \
                patch.object(pool.fixed, "validate_addon"), \
                patch.object(pool.fixed, "build_summary", return_value=zero_summary) as zero, \
                patch.object(pool.repaired42, "build_summary", wraps=pool.repaired42.build_summary) as forty_two:
            result, reads = self.invoke([0, 2, 3, 42, 92, 105], panels)
        zero.assert_called_once()
        forty_two.assert_called_once()
        self.assertEqual(result["completed_indices"], [0, 2, 3, 42, 92, 105])
        self.assertEqual(result["diagnostic_counts"]["completed_menus"], 6)
        self.assertEqual(result["diagnostic_counts"]["unknown_menus"], 194)
        self.assertEqual(len(result["cohorts"]), 200)
        self.assertEqual(result["cohorts"][1]["reason"], "MC1_interrupted_retry_held")
        self.assertFalse(any("replicate1/" in str(path) for path in reads))
        self.assertEqual(result["cohorts"][42]["summary"]["phase"], pool.repaired42.PHASE)
        for field in pool.repair.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")

    def test_missing_and_returned_technical_repairs_remain_unknown_with_costs_and_prefix(self):
        panels = {2: self.technical(self.panels(2)), 3: self.technical(self.panels(3)), 4: self.panels(4)}
        result, reads = self.invoke([2, 3, 4], panels, missing=(pool.input_paths(4, 46597540, 46597546)["MC"],))
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["diagnostic_counts"]["unknown_menus"], 200)
        for index in (2, 3):
            attempt = result["cohorts"][index]["technical_MVC"]
            self.assertEqual(attempt["elapsed_seconds"], 1000.)
            self.assertEqual(len(attempt["returned_fits"]), 2 if index == 3 else 1)
            self.assertEqual(attempt["cumulative_process_elapsed_seconds"], panels[index]["failed"]["elapsed_seconds"] + 1000.)
            self.assertNotIn("summary", result["cohorts"][index])
        panels = {2: self.technical(self.panels(2), unvalidated=True)}
        result, reads = self.invoke([2], panels)
        self.assertEqual(result["cohorts"][2]["technical_MVC"]["returned_fits"], [])
        self.assertNotIn("cumulative_process_elapsed_seconds", result["cohorts"][2]["technical_MVC"])
        result, reads = self.invoke([2], {2: self.panels(2)}, missing=(pool.input_paths(2)["failed"],))
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["cohorts"][2]["missing_reports"], [str(pool.input_paths(2)["failed"])])

    def test_malformed_successful_repair_cannot_hide_behind_another_missing_component(self):
        changes = (
            lambda p: p["MVC"].update(phase="original_phase"),
            lambda p: p["MVC"].update(job_id="other"),
            lambda p: p["MVC"].update(source_finished_utc="different"),
            lambda p: p["MVC"].update(source_job_id="other"),
            lambda p: p["MVC"]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda p: p["MVC"]["numerical_repair"].update(nonconverged_checkpoint_state_reused=True),
            lambda p: p["MVC"].update(cumulative_process_elapsed_seconds=3100.),
            lambda p: p["MVC"]["fits"][0]["parameters"][0].__setitem__(0, p["MVC"]["fits"][0]["parameters"][0][0] + .001),
            lambda p: p["MVC"]["numerical_repair"].update(reused_first_fit_original_elapsed_seconds=1.),
            lambda p: p["MVC"].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda p: p["MVC"]["participant_scores"]["p000"].update(M0=0.),
            lambda p: p["MVC"].update(full_gate="pass"),
            lambda p: p["failed"].update(source_finished_utc="different"),
        )
        for number, change in enumerate(changes):
            panels = self.panels(3)
            change(panels)
            with self.subTest(change=number), self.assertRaises(ValueError):
                self.invoke([3], {3: panels}, missing=(pool.input_paths(3)["MC"],))
        panels = self.panels(2)
        panels["MVC"]["phase"] = "wrong"
        with self.assertRaises(ValueError):
            self.invoke([2], {2: panels}, missing=(pool.input_paths(2)["failed"],))

    def test_technical_identity_replay_prefix_and_claim_failures_are_not_silently_counted_unknown(self):
        changes = (
            lambda p: p["MVC"].update(participant_scores={}),
            lambda p: p["MVC"].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda p: p["MVC"].update(scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}]),
            lambda p: p["MVC"].update(elapsed_seconds=-1.),
        )
        for number, change in enumerate(changes):
            panels = self.technical(self.panels(2))
            change(panels)
            with self.subTest(change=number), self.assertRaises(ValueError):
                self.invoke([2], {2: panels})

    def test_cli_explicit_arrays_compute_original_implementation_and_fresh_output(self):
        for value in ("", "0", "-1", "array"):
            with self.assertRaises(argparse.ArgumentTypeError):
                pool.array_id(value)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pool.json"
            args = ["--terminal-indices", "2", "--implementation", str(pool.IMPLEMENTATION), "--output", str(output)]
            with patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
                with patch.object(pool, "require_compute", side_effect=RuntimeError("compute")), self.assertRaises(RuntimeError):
                    pool.main(args)
                with patch.object(pool, "require_compute"), self.assertRaises(ValueError):
                    pool.main(["--terminal-indices", "2", "--implementation", "wrong.json", "--output", str(output)])
            result, reads = self.invoke([4, 11], {index: self.panels(index) for index in (4, 11)}, output=output)
            self.assertEqual(result["completed_indices"], [4, 11])
            self.assertEqual((result["fresh_array_job_id"], result["reverse_array_job_id"]), (46597540, 46597546))
            with patch.object(pool, "require_compute"), patch.object(Path, "read_text", side_effect=AssertionError("No read")), self.assertRaises(FileExistsError):
                pool.main(args)


if __name__ == "__main__":
    unittest.main()
