"""Focused optional runtime1500 null-repair admission checks; compute only."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mz_null_mvc_repair_admission as admission
import test_mz_null_indexed_mvc_repair as fixtures


class NullMVCRepairAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IndexedNullMVCRepairTests.setUpClass()
        cls.fixture = fixtures.IndexedNullMVCRepairTests(
            "test_two_fresh1500_fits_follow_original500_replay_preserve_last_bits_and_cost_once")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def reports(self, index=43, failure=None):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, scored, checkpoints, reads, source, failed = self.fixture.exercise(
                Path(directory) / "attempt", index=index, failure=failure)
        report["job_id"] = "70000444"
        return report, failed, source

    def validate(self, reports, index=43):
        report, failed, source = reports
        before = deepcopy((report, failed, source, self.spec))
        bindings = self.fixture.bindings()
        try:
            return admission.validate_repair(report, failed, source, self.spec,
                                             source["participant_scores"], self.people, index)
        finally:
            self.assertEqual((report, failed, source, self.spec), before)
            self.assertEqual(self.fixture.bindings(), bindings)

    def test_only_explicit_selected_index_array_pairs_change_paths(self):
        mapping = admission.repair_arrays("43:46599566, 61:70000611,168:70000618")
        self.assertEqual(mapping, {43: 46599566, 61: 70000611, 168: 70000618})
        self.assertIsNone(admission.input_paths(43, None))
        self.assertIsNone(admission.input_paths(61, {43: 46599566}))
        self.assertIsNone(admission.input_paths(1, mapping))
        for index, array in mapping.items():
            paths = admission.input_paths(index, mapping)
            self.assertEqual(paths["MVC"], admission.RUNS /
                f"20261005-M0-null-MVC-repair-array-{array}/replicate{index}/mz_null_indexed_mvc_repair.json")
            self.assertEqual(paths["failed_MVC"], fixtures.repair.failed_path(index))
        for value in ("", "43", "43:0", "43:-1", "42:7000", "43:7,43:8", "43:7:8", "all:7"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                admission.repair_arrays(value)
        for mapping in ({True: 7}, {43: True}, {43: "7"}, {44: 7}, []):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                admission.validate_mapping(mapping)

    def test_each_real_producer_shape_requires_two_fresh1500_fits_and_exact_baseline_last_bits(self):
        for index in admission.INDICES:
            reports = self.reports(index)
            original_validate = fixtures.repair.admission.menu.fixed.validate_fit
            with self.subTest(index=index), patch.object(fixtures.repair.admission.menu.fixed,
                    "validate_fit", wraps=original_validate) as validate:
                scores = self.validate(reports, index)
            self.assertEqual(scores, reports[0]["participant_scores"])
            self.assertEqual([call.args[4]["optimizer"]["maximum_sweeps"] for call in validate.call_args_list], [1500, 1500])
            for person, row in reports[2]["participant_scores"].items():
                self.assertEqual({key: scores[person][key] for key in row}, row)

    def test_wrong_identity_lineage_specs_cost_replay_or_claim_cannot_enter_completed_scores(self):
        original = self.reports()
        changes = (
            lambda reports: reports[0].update(phase=fixtures.repair.indexed.PHASE),
            lambda reports: reports[0].update(replicate_index=61),
            lambda reports: reports[0].update(job_id="other"),
            lambda reports: reports[0].update(source_report="wrong.json"),
            lambda reports: reports[0].update(source_job_id="other"),
            lambda reports: reports[0].update(source_finished_utc="different"),
            lambda reports: reports[0].update(failed_report="wrong.json"),
            lambda reports: reports[1].update(job_id="other"),
            lambda reports: reports[2].update(job_id="other"),
            lambda reports: reports[0]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda reports: reports[0]["numerical_override"].update(runtime_maximum_sweeps=1000),
            lambda reports: reports[0]["numerical_repair"].update(previous_process_elapsed_seconds=1.),
            lambda reports: reports[0]["numerical_repair"].update(nonconverged_checkpoint_state_reused=True),
            lambda reports: reports[0].update(cumulative_process_elapsed_seconds=0.),
            lambda reports: reports[0].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda reports: reports[0]["participant_scores"]["p000"].update(M0=-2.),
            lambda reports: reports[0]["fits"][0].update(objective_history=[1.] * 1501),
            lambda reports: reports[0].update(full_gate="pass"),
            lambda reports: reports[0].update(automatic_retry=True),
        )
        for position, change in enumerate(changes):
            reports = deepcopy(original)
            change(reports)
            with self.subTest(change=position), self.assertRaises((ValueError, RuntimeError)):
                self.validate(reports)
        for field, value in (("status", "running"), ("rival_diagnostics_complete", False),
                             ("source_baseline_validated", False), ("repair_context_validated", False),
                             ("completed_rivals", []), ("scored_primary_directions", [])):
            reports = deepcopy(original)
            reports[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(reports)

    def test_technical_returned_prefixes_preserve_costs_without_score_admission(self):
        for failure, fit_count in (("source_timeout", 0), ("replay", 0), ("first_fit", 1),
                                   ("second_fit", 2), ("finalization_timeout", 2)):
            reports = self.reports(61, failure)
            with self.subTest(failure=failure):
                evidence = self.validate(reports, 61)
                self.assertEqual(evidence["status"], "technical_rival_failure")
                self.assertEqual(len(evidence["returned_fits"]), fit_count)
                self.assertNotIn("participant_scores", evidence)
                self.assertEqual(evidence["elapsed_seconds"], reports[0]["elapsed_seconds"])
                if failure == "source_timeout":
                    self.assertNotIn("cumulative_process_elapsed_seconds", evidence)
                else:
                    self.assertEqual(evidence["cumulative_process_elapsed_seconds"],
                                     reports[1]["elapsed_seconds"] + reports[0]["elapsed_seconds"])
                if fit_count == 2:
                    self.assertEqual(evidence["returned_fits"][0]["returned_sweeps"], 1100)

    def test_missing_prior_stays_unadmitted_and_malformed_technical_attempt_is_rejected(self):
        reports = list(self.reports())
        reports[1] = None
        self.assertIsNone(self.validate(reports))
        reports[0]["phase"] = "wrong"
        with self.assertRaises(ValueError):
            self.validate(reports)
        original = self.reports(168, "second_fit")
        for field, value in (("participant_scores", {}), ("elapsed_seconds", -1.),
                             ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                             ("scored_primary_directions", [])):
            reports = deepcopy(original)
            reports[0][field] = value
            # An empty scored prefix is not inherently invalid after a returned
            # first fit. Make the decisive invalid case a scored failed second fit.
            if field == "scored_primary_directions":
                reports[0][field] = [{"training_runs": list(train), "scoring_runs": list(held)}
                                     for train, held in fixtures.repair.indexed.HALVES]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(reports, 168)

    def pool_panels(self, index, failure=None, mapped=True):
        report, failed, source = self.reports(index, failure)
        panels = self.fixture.source_fixture.reports(index)
        panels["baseline"] = source
        for model in ("MC", "MV", "MVZ"):
            panels[model].update(source_job_id=source["job_id"], source_finished_utc=source["finished_utc"])
        panels["MVC"] = report if mapped else failed
        if mapped:
            panels["failed_MVC"] = failed
        return panels

    def invoke_pool(self, selected, panels, mapping=None, *, missing=(), output=None):
        pool = fixtures.repair.admission
        allowed = {pool.IMPLEMENTATION: json.dumps(self.spec)}
        for index, reports in panels.items():
            paths = pool.input_paths(index, mvc_repair_arrays=mapping)
            allowed.update({paths[name]: json.dumps(report) for name, report in reports.items()})
            allowed[pool.simple.source_directory(index) / "generating_parameters.json"] = json.dumps(self.fixture.truth)
        preserved = deepcopy(panels)
        functions = (pool.rival_identity, pool.technical_rival, pool.expected_override, pool.menu.validate_rival)
        runtime = pool.mvc_runner.RUNTIME_SWEEPS
        reads = []

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path in missing:
                raise FileNotFoundError(path)
            return allowed[path]

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_text", reader))
            stack.enter_context(patch.object(pool.menu.null.shared, "load_cohort", return_value=self.people))
            stack.enter_context(patch.object(pool, "require_compute"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                if output is None:
                    result = pool.pool(selected, self.spec, mvc_repair_arrays=mapping)
                else:
                    args = ["--terminal-indices", ",".join(map(str, selected)), "--implementation", str(pool.IMPLEMENTATION),
                            "--output", str(output)]
                    if mapping is not None:
                        args += ["--mvc-repair-arrays", ",".join(f"{index}:{array}" for index, array in mapping.items())]
                    self.assertEqual(pool.main(args), 0)
                    result = None
            finally:
                self.assertEqual((pool.rival_identity, pool.technical_rival, pool.expected_override, pool.menu.validate_rival), functions)
                self.assertEqual(pool.mvc_runner.RUNTIME_SWEEPS, runtime)
        self.assertEqual(panels, preserved)
        return json.loads(output.read_text()) if output is not None else result, reads

    def test_optional_pool_cli_admits_each_mapped_repair_with_original_nine_contrasts_and_fixed200(self):
        pool = fixtures.repair.admission
        mapping = {43: 46599566, 61: 46599991, 168: 46599991}
        panels = {index: self.pool_panels(index) for index in mapping}
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke_pool(list(mapping), panels, mapping, output=Path(directory) / "summary.json")
        self.assertEqual(result["completed_indices"], [43, 61, 168])
        self.assertEqual(result["mvc_repair_arrays"], {str(index): array for index, array in mapping.items()})
        self.assertEqual(result["diagnostic_counts"]["complete_menus"], 3)
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 197)
        for index, array in mapping.items():
            summary = result["cohorts"][index]["summary"]
            self.assertEqual(set(summary["nested_comparisons"]), {f"{model}-{nested}" for model, nested in pool.menu.fixed.EDGES})
            self.assertEqual(summary["mvc_repair_array_job_id"], array)
            self.assertEqual(summary["mvc_runtime_implementation"]["optimizer"]["maximum_sweeps"], 1500)
            self.assertEqual(summary["cumulative_mvc_process_elapsed_seconds"], panels[index]["MVC"]["cumulative_process_elapsed_seconds"])
            for field in pool.CLAIM_FIELDS:
                self.assertEqual(summary[field], "not_evaluated")
            self.assertIn(admission.input_paths(index, mapping)["failed_MVC"], reads)
            self.assertIn(admission.input_paths(index, mapping)["MVC"], reads)

    def test_omitted_mapping_keeps_original_technical_source_and_unselected_repairs_are_not_read(self):
        pool = fixtures.repair.admission
        panels = {43: self.pool_panels(43, mapped=False)}
        result, reads = self.invoke_pool([43], panels)
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["mvc_repair_arrays"], {})
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        self.assertIn(fixtures.repair.failed_path(43), reads)
        self.assertFalse(any("MVC-repair-array" in str(path) for path in reads))
        with patch.object(Path, "read_text", side_effect=AssertionError("No selected repair")):
            self.assertEqual(pool.pool([], self.spec, mvc_repair_arrays={43: 46599566})["completed_indices"], [])
        panels[43]["MVC"] = self.reports()[0]
        with self.assertRaises(ValueError):
            self.invoke_pool([43], panels)

    def test_pool_technical_missing_and_malformed_repairs_never_contribute_complete_menus(self):
        pool = fixtures.repair.admission
        mapping = {43: 46599566, 61: 46599991, 168: 46599991}
        panels = {43: self.pool_panels(43, "second_fit"), 61: self.pool_panels(61), 168: self.pool_panels(168)}
        missing = (pool.input_paths(61, mvc_repair_arrays=mapping)["failed_MVC"],
                   pool.input_paths(168, mvc_repair_arrays=mapping)["MVZ"])
        result, reads = self.invoke_pool(list(mapping), panels, mapping, missing=missing)
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        evidence = result["cohorts"][43]["technical_components"]["MVC"]
        self.assertEqual(len(evidence["returned_fits"]), 2)
        self.assertEqual(evidence["cumulative_process_elapsed_seconds"], panels[43]["MVC"]["cumulative_process_elapsed_seconds"])
        self.assertEqual(result["cohorts"][61]["missing_reports"], [str(missing[0])])
        panels[168]["MVC"]["participant_scores"]["p000"]["M0"] += 1e-10
        with self.assertRaises(ValueError):
            self.invoke_pool(list(mapping), panels, mapping, missing=missing)
        with self.assertRaises(ValueError):
            pool.build_indexed_summary(panels[43]["baseline"], {model: panels[43][model] for model in pool.menu.RIVALS},
                self.spec, self.fixture.truth, self.people, 43, mvc_repair_arrays=mapping, failed_mvc=panels[43]["failed_MVC"])


if __name__ == "__main__":
    unittest.main()
