"""Focused fixed-source indexed-null menu pooling tests; compute only."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz_null_indexed_menu as pool
import test_mz_null_baseline_wave as baseline_fixtures
import test_mz_null_repaired_menu as pilot_fixtures
import test_mz_null_completed_baseline_rival as completion_fixtures
import test_mz_null_additional_completed_rival as additional_fixtures


class IndexedNullMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline_fixtures.NullBaselineWaveTests.setUpClass()
        cls.baseline_fixture = baseline_fixtures.NullBaselineWaveTests("test_exact_source_mapping_preserves_pilot_and_two_array_ranges")
        cls.fixture = cls.baseline_fixture.fixture
        cls.spec, cls.people, cls.truth = cls.fixture.spec, cls.fixture.people, cls.fixture.truth
        pilot_fixtures.NullRepairedMenuTests.setUpClass()
        cls.pilot_fixture = pilot_fixtures.NullRepairedMenuTests(
            "test_original_nine_contrasts_and_mixed_fit_validation_preserve_null_claim_boundary")
        completion_fixtures.CompletedNullBaselineRivalTests.setUpClass()
        cls.completion_fixture = completion_fixtures.CompletedNullBaselineRivalTests(
            "test_exact_completion_and_preserved_original_fit_lineage_are_required")
        cls.additional_fixture = additional_fixtures.AdditionalCompletedNullRivalTests(
            "test_exact_two_completed_identities_and_shared_cost_lineage_are_preserved")
        cls.additional_fixture.completed_fixture = cls.completion_fixture

    def reports(self, index):
        if index == 0:
            result = dict(zip(("baseline", *pool.menu.RIVALS, "failed_MVC"), self.pilot_fixture.reports()))
            result["baseline"]["nested_comparison"] = pool.menu.fixed.nested_contrast(
                result["baseline"]["participant_scores"], "MZ", "M0")
            return result
        failed, lineage = None, None
        if index in pool.UNADMITTED:
            base, failed = self.additional_fixture.sources(index)
            lineage = pool.additional.validate_completed_baseline(index, base, failed, self.truth, self.people, self.spec)
        elif index == 35:
            failed = self.completion_fixture.source_fixture.source()
            base = self.completion_fixture.completed(failed)
            lineage = pool.completed35.validate_completed_baseline(base, failed, self.truth, self.people, self.spec)
        else:
            base = self.baseline_fixture.report(index)
            base["job_id"] = "70000013"
        result = {"baseline": base}
        if failed is not None:
            result["failed_baseline"] = failed
        directory = pool.simple.source_directory(index)
        for model, report in zip(pool.menu.RIVALS, self.pilot_fixture.menu_fixture.reports()[1:]):
            replacement = index in pool.UNADMITTED or (index == 35 and model != "MVC")
            phase = pool.mvc_runner.PHASE if model == "MVC" else (
                pool.completed35.PHASE if replacement else (pool.simple.PHASE if model in ("MC", "MV") else pool.mvz_runner.PHASE))
            source_report = str(pool.completed35.COMPLETION_REPORT) if index == 35 else (
                str(directory / pool.baseline.indexed.REPORT_NAME) if model == "MVC" else pool.baseline.indexed.REPORT_NAME)
            if index in pool.UNADMITTED:
                phase = pool.additional.PHASE
                source_report = str(pool.additional.source_paths(index)[1])
            report.update(phase=phase, replicate_index=index, seed=1300037 + 10007 * index,
                          job_id="46589557" if index == 35 and model != "MVC" else "70000017", source_dir=str(directory),
                          source_report=source_report, source_job_id=str(base["job_id"]), source_finished_utc=base["finished_utc"],
                          source_baseline_validated=True, selected_replicates=[index], selected_models=[model],
                          fixed_cohort_count=200, component_scope=f"{model}_only", wave_selected=not replacement,
                          full_menu_wave_selected=False, elapsed_seconds=100.,
                          internal_seconds=8400 if model == "MVC" else (3300 if model == "MVZ" else 2400),
                          participant_scores={person: {**row, model: -2.05} for person, row in base["participant_scores"].items()})
            if replacement:
                report.update(retry_wave_selected=False, simulation_copied=False, baseline_lineage=lineage)
            else:
                report.update(wave_scope=f"{model}_component_only", retry_selected=False)
            if index in pool.UNADMITTED:
                report.update(job_id="70000017" if model == "MVC" else "46595106",
                              wave_scope="single_completed_baseline_component", retry_selected=False,
                              automatic_retry=False, early_kill="not_evaluated",
                              baseline_validation_maximum_sweeps=500,
                              fitted_maximum_sweeps=1000 if model == "MVC" else 500)
            for fit in report["fits"]:
                fit["elapsed_seconds"] = 40.
            if model == "MVC":
                report.update(runtime_implementation=pool.mvc_runner.runtime_specification(self.spec),
                              numerical_override=pool.expected_override(), automatic_retry=False,
                              simulation_copied=False, baseline_lineage=lineage, early_kill="not_evaluated")
                for fit in report["fits"]:
                    fit["objective_history"] = [float(value) for value in range(650, 1, -1)] + [2.]
            result[model] = report
        return result

    def bindings(self):
        return (pool.menu.validate_rival, pool.menu.null.validate_source, pool.menu.null.REPLICATE_INDEX,
                pool.menu.null.SEED, pool.menu.null.SOURCE, self.additional_fixture.bindings())

    def invoke(self, selected, panels, *, missing=(), cli_output=None, additional_array=None):
        allowed = {pool.IMPLEMENTATION: json.dumps(self.spec)}
        events, reads, loads = {}, [], []
        for index, reports in panels.items():
            paths = pool.input_paths(index, additional_array)
            allowed.update({paths[name]: json.dumps(report) for name, report in reports.items()})
            directory = pool.menu.null.SOURCE if index == 0 else pool.simple.source_directory(index)
            allowed[directory / "generating_parameters.json"] = json.dumps(self.truth)
            events[directory / "simulated_events.csv.gz"] = self.people
        original = self.bindings()
        snapshots = deepcopy(panels)

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in missing:
                raise FileNotFoundError(path)
            return allowed[path]

        def loader(path):
            loads.append(path)
            return events[path]

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(pool.menu.null.shared, "load_cohort", side_effect=loader))
            stack.enter_context(patch.object(pool, "require_compute"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                if cli_output is None:
                    result = pool.pool(selected, self.spec, additional_array)
                else:
                    args = ["--terminal-indices", ",".join(map(str, selected)),
                            "--implementation", str(pool.IMPLEMENTATION), "--output", str(cli_output)]
                    if additional_array is not None:
                        args.extend(["--additional-mvc-array-job-id", str(additional_array)])
                    self.assertEqual(pool.main(args), 0)
                    result = None
            finally:
                self.assertEqual(self.bindings(), original)
        self.assertEqual(panels, snapshots)
        if cli_output is not None:
            result = json.loads(cli_output.read_text())
        return result, reads, loads

    def test_exact_selected_paths_cover_pilot_group8_both_baseline_ranges_and_replacements(self):
        self.assertEqual(tuple(pool.input_paths(0).values()), pool.pilot_menu.INPUT_REPORTS)
        for index in (1, 4, 5, 29, 32, 33, 35, 199):
            paths = pool.input_paths(index)
            base_array = 46579924 if index <= 4 else 46580387
            expected = pool.completed35.COMPLETION_REPORT if index == 35 else pool.RUNS / (
                f"20261005-M0-null-baseline-array-{base_array}/replicate{index}/mz_null_replicate.json")
            self.assertEqual(paths["baseline"], expected)
            simple_array = 46583757 if 29 <= index <= 32 else 46584095
            mvz_array = 46581137 if index <= 4 else (46584696 if 29 <= index <= 32 else 46585182)
            for model, array in (("MC", simple_array), ("MV", simple_array), ("MVZ", mvz_array)):
                if index == 35:
                    self.assertIn(f"null35-{model}-completed-baseline-46589557", str(paths[model]))
                else:
                    self.assertIn(f"null-{model}-indexed-array-{array}/replicate{index}/", str(paths[model]))
            self.assertIn(f"MVC-indexed-array-{46591849 if index == 1 else 46592261}/replicate{index}/", str(paths["MVC"]))
        for index in (108, 176, -1, 200, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                pool.input_paths(index)
        for index in (108, 176):
            paths = pool.input_paths(index, 70000031)
            self.assertEqual(paths["baseline"], pool.additional.source_paths(index)[1])
            self.assertEqual(paths["failed_baseline"], pool.simple.source_directory(index) / "mz_null_replicate.json")
            self.assertEqual(paths["MVC"], pool.RUNS / (
                f"20261005-M0-null-MVC-additional-array-70000031/replicate{index}/mz_null_additional_completed_rival.json"))
            for model in ("MC", "MV", "MVZ"):
                self.assertEqual(paths[model], pool.RUNS / (
                    f"20261005-M0-null{index}-{model}-completed-baseline-46595106/mz_null_additional_completed_rival.json"))
        for index in (0, 1, 29, 35, 199):
            self.assertEqual(pool.input_paths(index, 70000031), pool.input_paths(index))
        for array in (0, -1, True, "70000031"):
            with self.subTest(array=array), self.assertRaises(ValueError):
                pool.input_paths(108, array)

    def test_indexed_builder_reuses_nine_contrasts_and_original500_rivals_runtime1000_MVC(self):
        reports = self.reports(5)
        before = deepcopy(reports)
        original = self.bindings()
        with patch.object(pool.menu.fixed, "nested_contrast", wraps=pool.menu.fixed.nested_contrast) as contrasts, \
                patch.object(pool.menu.fixed, "validate_fit", wraps=pool.menu.fixed.validate_fit) as fits:
            result = pool.build_indexed_summary(reports["baseline"], {model: reports[model] for model in pool.menu.RIVALS},
                                               self.spec, self.truth, self.people, 5)
        self.assertEqual(self.bindings(), original)
        self.assertEqual(reports, before)
        # One existing source diagnostic is verified before the unchanged nine
        # menu contrasts. There is no new selector or numerical formula.
        self.assertEqual(contrasts.call_count, 10)
        self.assertEqual([(call.args[1], call.args[2]) for call in contrasts.call_args_list[-9:]], list(pool.menu.fixed.EDGES))
        self.assertEqual([(call.args[1], call.args[4]["optimizer"]["maximum_sweeps"]) for call in fits.call_args_list],
                         [("M0", 500), ("MZ", 500), ("M0", 500), ("MZ", 500),
                          ("MC", 500), ("MC", 500), ("MV", 500), ("MV", 500),
                          ("MVZ", 500), ("MVZ", 500), ("MVC", 1000), ("MVC", 1000)])
        self.assertEqual(result["replicate_index"], 5)
        self.assertEqual(len(result["nested_comparisons"]), 9)
        self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ"])
        self.assertEqual(result["family_selection"], "not_evaluated")

    def test_pilot_and_index35_use_existing_repair_lineage_validators_without_relabeling(self):
        panels = {index: self.reports(index) for index in (0, 35)}
        with patch.object(pool.pilot_menu, "build_summary", wraps=pool.pilot_menu.build_summary) as pilot_builder, \
                patch.object(pool.completed35, "validate_completed_baseline", wraps=pool.completed35.validate_completed_baseline) as lineage:
            result, reads, loads = self.invoke([0, 35], panels)
        self.assertEqual(result["completed_indices"], [0, 35])
        self.assertEqual(pilot_builder.call_count, 1)
        self.assertGreaterEqual(lineage.call_count, 1)
        self.assertEqual(result["cohorts"][0]["summary"]["phase"], pool.pilot_menu.PHASE)
        self.assertEqual(result["cohorts"][35]["summary"]["baseline_lineage"]["completion_job_id"], "46586603")
        self.assertIn(pool.completed35.FAILED_REPORT, reads)
        self.assertIn(pool.completed35.completion.SOURCE / "simulated_events.csv.gz", loads)
        self.assertNotIn(pool.mvc_runner.QUALIFYING_NULL0_REPORT, [path for path in reads if path not in pool.input_paths(0).values()])

    def test_fixed200_counts_M0_only_and_incomparable_diagnostics_without_scientific_rates(self):
        panels = {index: self.reports(index) for index in (1, 29)}
        base = panels[1]["baseline"]
        for row in base["participant_scores"].values():
            row["MZ"] = row["M0"]
        base["nested_comparison"] = pool.menu.fixed.nested_contrast(base["participant_scores"], "MZ", "M0")
        for model in pool.menu.RIVALS:
            for person, row in base["participant_scores"].items():
                panels[1][model]["participant_scores"][person].update(row)
        for row in panels[29]["MC"]["participant_scores"].values():
            row["MC"] = -1.97
        result, reads, loads = self.invoke([1, 29], panels)
        self.assertEqual(result["diagnostic_counts"], {"complete_menus": 2, "M0_only_maximal_diagnostics": 1,
                         "multiple_incomparable_diagnostics": 1, "unknown_cohorts": 198})
        self.assertEqual(len(result["cohorts"]), 200)
        self.assertEqual(len(result["unresolved_indices"]), 198)
        for field in pool.CLAIM_FIELDS:
            self.assertEqual(result[field], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["episode_terminal"])
        self.assertFalse(result["retry_selected"])

    def test_missing_technical_and_unadmitted_cohorts_stay_unknown_without_loading_unselected_sources(self):
        panels = {index: self.reports(index) for index in (1, 4, 5, 35)}
        technical = panels[4]["MVC"]
        technical.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                         error="RuntimeError: Nonconverged fit is technical; it is not a recovery failure",
                         fits=technical["fits"][:1], scored_primary_directions=[], completed_rivals=[], diagnostics={})
        technical.pop("participant_scores")
        technical["fits"][0].update(converged=False, objective_history=list(range(1000, 0, -1)))
        panels[5]["baseline"] = self.baseline_fixture.report(5, technical=True)
        completion = panels[35]["baseline"]
        completion.update(status="technical_baseline_completion_failure", pairwise_diagnostic_complete=False,
                          error="RuntimeError: completion interrupted", fits=completion["fits"][:3],
                          scored_fit_prefix=completion["scored_fit_prefix"][:3],
                          scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
        completion.pop("participant_scores")
        completion.pop("nested_comparison")
        missing = (pool.input_paths(1)["MVZ"],)
        result, reads, loads = self.invoke([1, 4, 5, 35, 108, 176], panels, missing=missing)
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        self.assertEqual(result["cohorts"][4]["technical_components"]["MVC"]["returned_fits"][0]["returned_sweeps"], 1000)
        self.assertEqual(result["cohorts"][4]["technical_components"]["MVC"]["elapsed_seconds"], 100.)
        self.assertEqual(result["cohorts"][5]["reason"], "technical_baseline_failure")
        self.assertEqual(result["cohorts"][35]["reason"], "technical_baseline_completion_failure")
        for index in (108, 176):
            self.assertEqual(result["cohorts"][index]["reason"], "baseline_completion_and_downstream_components_not_admitted")
        self.assertFalse(any("replicate108" in str(path) or "replicate176" in str(path) for path in reads + loads))
        self.assertNotIn(pool.input_paths(5)["MC"], reads)
        with patch.object(Path, "read_text", side_effect=AssertionError("No selected admitted sources")):
            self.assertEqual(pool.pool([108, 176], self.spec)["diagnostic_counts"]["unknown_cohorts"], 200)

    def test_malformed_completed_panels_abort_even_when_another_component_is_missing(self):
        mutations = (
            lambda reports: reports["MVC"].update(phase=pool.menu.null.REPORT_NAME),
            lambda reports: reports["MVC"]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda reports: reports["MVC"]["numerical_override"].update(nonconverged_checkpoint_state_reused=True),
            lambda reports: reports["MC"].update(source_finished_utc="different"),
            lambda reports: reports["MC"].update(source_job_id="46579924"),
            lambda reports: reports["MVC"]["participant_scores"]["p000"].update(M0=-2.),
            lambda reports: reports["MVC"].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda reports: reports["MVC"].update(full_gate="pass"),
            lambda reports: reports["MC"]["fits"][0].update(objective_history=[1.] * 501),
            lambda reports: reports["MVC"].update(scored_primary_directions=[]),
        )
        for number, mutate in enumerate(mutations):
            reports = self.reports(1)
            mutate(reports)
            with self.subTest(mutation=number), self.assertRaises(ValueError):
                self.invoke([1], {1: reports}, missing=(pool.input_paths(1)["MVZ"],))
        reports = self.reports(35)
        reports["MC"]["baseline_lineage"]["completion_job_id"] = "other"
        with self.assertRaises(ValueError):
            self.invoke([35], {35: reports})

    def test_technical_status_cannot_hide_successful_scores_or_wrong_identity(self):
        for mutation in ("scores", "identity", "cost", "scored_failed_fit"):
            reports = self.reports(1)
            bad = reports["MVC"]
            bad.update(status="technical_rival_failure", rival_diagnostics_complete=False, error="RuntimeError: interrupted",
                       completed_rivals=[], diagnostics={}, scored_primary_directions=[], fits=bad["fits"][:1])
            if mutation != "scores":
                bad.pop("participant_scores")
            if mutation == "identity":
                bad["replicate_index"] = 2
            elif mutation == "cost":
                bad["elapsed_seconds"] = -1.
            elif mutation == "scored_failed_fit":
                bad["fits"][0]["converged"] = False
                bad["scored_primary_directions"] = [{"training_runs": [1, 4], "scoring_runs": [2, 3]}]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.invoke([1], {1: reports})

    def test_validated_source_replay_failure_without_completion_timestamp_remains_unknown(self):
        for index, model in ((1, "MC"), (35, "MVZ")):
            reports = self.reports(index)
            report = reports[model]
            report.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                          error="ValueError: saved scores exceed replay tolerance", saved_scores_replayed=False,
                          fits=[], scored_primary_directions=[], completed_rivals=[], diagnostics={})
            for field in ("participant_scores", "source_finished_utc", "implementation",
                          "saved_score_replay_max_absolute_discrepancy", "saved_score_replay_absolute_tolerance"):
                report.pop(field)
            with self.subTest(index=index, model=model):
                result, reads, loads = self.invoke([index], {index: reports})
                self.assertEqual(result["completed_indices"], [])
                self.assertEqual(result["cohorts"][index]["technical_components"][model]["returned_fits"], [])
                self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
                report["source_finished_utc"] = "different"
                with self.assertRaises(ValueError):
                    self.invoke([index], {index: reports})

    def test_optional_additional_admission_preserves_lineage_nine_contrasts_and_mixed_specs(self):
        panels = {index: self.reports(index) for index in (108, 176)}
        with patch.object(pool.additional, "validate_completed_baseline", wraps=pool.additional.validate_completed_baseline) as sources, \
                patch.object(pool.menu.fixed, "validate_fit", wraps=pool.menu.fixed.validate_fit) as fits:
            result, reads, loads = self.invoke([108, 176], panels, additional_array=70000031)
        self.assertEqual(result["additional_mvc_array_job_id"], 70000031)
        self.assertEqual(result["completed_indices"], [108, 176])
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 198)
        self.assertEqual({call.args[0] for call in sources.call_args_list}, {108, 176})
        self.assertEqual({(call.args[1], call.args[4]["optimizer"]["maximum_sweeps"]) for call in fits.call_args_list},
                         {("M0", 500), ("MZ", 500), ("MC", 500), ("MV", 500), ("MVZ", 500), ("MVC", 1000)})
        for index in (108, 176):
            summary = result["cohorts"][index]["summary"]
            self.assertEqual(len(summary["nested_comparisons"]), 9)
            self.assertEqual(summary["additional_mvc_array_job_id"], 70000031)
            self.assertEqual(summary["baseline_lineage"]["completion_job_id"], "46591918")
            self.assertEqual(summary["baseline_lineage"]["original_worker_job_id"], pool.additional.additional.SOURCES[index][0])
            self.assertIn(pool.input_paths(index, 70000031)["failed_baseline"], reads)
            self.assertIn(pool.simple.source_directory(index) / "simulated_events.csv.gz", loads)
            for field in pool.CLAIM_FIELDS:
                self.assertEqual(summary[field], "not_evaluated")
            with self.assertRaises(ValueError):
                pool.build_indexed_summary(panels[index]["baseline"], {model: panels[index][model] for model in pool.menu.RIVALS},
                    self.spec, self.truth, self.people, index, panels[index]["failed_baseline"])

    def test_additional_admission_rejects_wrong_completion_lineage_phase_model_spec_and_replay(self):
        mutations = (
            lambda reports: reports["baseline"].update(job_id="46586603"),
            lambda reports: reports["failed_baseline"].update(job_id="46580478"),
            lambda reports: reports["baseline"]["fits"][0].update(elapsed_seconds=0.),
            lambda reports: reports["MC"].update(phase=pool.completed35.PHASE),
            lambda reports: reports["MC"].update(job_id="46589557"),
            lambda reports: reports["MV"].update(source_report=str(pool.completed35.COMPLETION_REPORT)),
            lambda reports: reports["MC"].update(source_finished_utc="different"),
            lambda reports: reports["MVZ"].update(baseline_lineage={}),
            lambda reports: reports["MC"].update(baseline_validation_maximum_sweeps=1000),
            lambda reports: reports["MVC"].update(fitted_maximum_sweeps=500),
            lambda reports: reports["MVC"]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda reports: reports["MVC"].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda reports: reports["MVC"]["participant_scores"]["p000"].update(M0=-2.),
        )
        for index in (108, 176):
            for number, mutate in enumerate(mutations):
                reports = self.reports(index)
                mutate(reports)
                with self.subTest(index=index, mutation=number), self.assertRaises(ValueError):
                    self.invoke([index], {index: reports}, additional_array=70000031)

    def test_additional_missing_and_technical_attempts_remain_unknown_with_consumed_costs(self):
        for model in ("MC", "MVC"):
            reports = self.reports(108)
            report = reports[model]
            report.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                          error="ValueError: saved scores exceed replay tolerance", saved_scores_replayed=False,
                          fits=[], scored_primary_directions=[], completed_rivals=[], diagnostics={})
            report.pop("participant_scores")
            report.pop("source_finished_utc")
            with self.subTest(model=model):
                result, reads, loads = self.invoke([108], {108: reports}, additional_array=70000031,
                    missing=(pool.input_paths(108, 70000031)["MVZ"],))
                self.assertEqual(result["completed_indices"], [])
                self.assertEqual(result["cohorts"][108]["technical_components"][model]["elapsed_seconds"], 100.)
                self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        for index in (108, 176):
            reports = self.reports(index)
            report = reports["baseline"]
            report.update(status="technical_baseline_completion_failure", pairwise_diagnostic_complete=False,
                          error="RuntimeError: completion interrupted", fits=report["fits"][:3],
                          scored_fit_prefix=report["scored_fit_prefix"][:3],
                          scored_primary_directions=[{"training_runs": [1, 4], "scoring_runs": [2, 3]}])
            report.pop("participant_scores")
            report.pop("nested_comparison")
            result, reads, loads = self.invoke([index], {index: reports}, additional_array=70000031)
            self.assertEqual(result["cohorts"][index]["reason"], "technical_baseline_completion_failure")
            self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
            self.assertNotIn(pool.input_paths(index, 70000031)["MC"], reads)

    def test_cli_explicit_terminal_selection_compute_implementation_and_fresh_output_guards(self):
        for value in ("", "1,1", "-1", "200", "all"):
            with self.subTest(value=value), self.assertRaises(Exception):
                pool.baseline.terminal_indices(value)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pool.json"
            with patch.object(Path, "read_text", side_effect=AssertionError("No source read")):
                args = ["--terminal-indices", "1", "--implementation", str(pool.IMPLEMENTATION), "--output", str(output)]
                with patch.object(pool, "require_compute", side_effect=RuntimeError("Compute required")), self.assertRaises(RuntimeError):
                    pool.main(args)
                with patch.object(pool, "require_compute"), self.assertRaises(ValueError):
                    pool.main(["--terminal-indices", "1", "--implementation", str(Path(directory) / "wrong.json"), "--output", str(output)])
            result, reads, loads = self.invoke([1], {1: self.reports(1)}, cli_output=output)
            self.assertEqual(result["completed_indices"], [1])
            self.assertIsNone(result["additional_mvc_array_job_id"])
            self.assertEqual(reads[0], pool.IMPLEMENTATION)
            with patch.object(pool, "require_compute"), patch.object(Path, "read_text", side_effect=AssertionError("No read")), self.assertRaises(FileExistsError):
                pool.main(args)
            result, reads, loads = self.invoke([108], {108: self.reports(108)},
                cli_output=Path(directory) / "additional.json", additional_array=70000031)
            self.assertEqual(result["completed_indices"], [108])
            self.assertEqual(result["additional_mvc_array_job_id"], 70000031)
            for value in ("0", "-1", "not-an-array"):
                with patch.object(Path, "read_text", side_effect=AssertionError("No source read")), \
                        redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    pool.main([*args, "--additional-mvc-array-job-id", value])


if __name__ == "__main__":
    unittest.main()
