"""Focused prepared reverse-completion admission tests; execute only on compute."""
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

import mz_null_mvc_reverse_admission as admission
import test_mz_null_indexed_mvc_reverse_completion as fixtures


class NullMVCReverseAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NullMVCReverseCompletionTests.setUpClass()
        cls.fixture = fixtures.NullMVCReverseCompletionTests(
            "test_one_fresh_reverse_after_baseline_replay_and_preserved_first_rescore_with_cost_once")
        cls.spec, cls.people = cls.fixture.spec, cls.fixture.people

    def reports(self, index=183, failure=None):
        with tempfile.TemporaryDirectory() as directory:
            report, calls, scores, validations, checkpoints, reads, source, failed = self.fixture.exercise(
                Path(directory) / "new", index=index, failure=failure)
        report["job_id"] = "70000183"
        return report, failed, source

    def validate(self, reports, index=183):
        report, failed, source = reports
        before = deepcopy((report, failed, source, self.spec))
        bindings = self.fixture.bindings()
        old_fit_spec = fixtures.repair.admission.fit_spec
        try:
            return admission.validate_completion(report, failed, source, self.spec,
                                                 source["participant_scores"], self.people, index)
        finally:
            self.assertEqual((report, failed, source, self.spec), before)
            self.assertEqual(self.fixture.bindings(), bindings)
            self.assertIs(fixtures.repair.admission.fit_spec, old_fit_spec)

    def test_only_explicit_selected_reverse_array_pairs_admit_new_paths(self):
        mapping = admission.reverse_arrays("183:46600829, 31:46601362,12:70000012,37:70000037,109:70000109,135:70000135,140:70000140,159:70000159,164:70000164,188:70000188")
        self.assertEqual(mapping, {183: 46600829, 31: 46601362, 12: 70000012, 37: 70000037,
                                   109: 70000109, 135: 70000135, 140: 70000140, 159: 70000159, 164: 70000164, 188: 70000188})
        self.assertIsNone(admission.input_paths(183, None))
        self.assertIsNone(admission.input_paths(31, {183: 46600829}))
        self.assertIsNone(admission.input_paths(43, mapping))
        for index, array in mapping.items():
            paths = admission.input_paths(index, mapping)
            self.assertEqual(paths["MVC"], admission.RUNS /
                f"20261005-M0-null-MVC-reverse-completion-array-{array}/replicate{index}/mz_null_indexed_mvc_reverse_completion.json")
            self.assertEqual(paths["failed_MVC"], fixtures.repair.failed_path(index))
        for value in ("", "183", "183:0", "31:-1", "43:7000", "183:7,183:8", "31:7:8", "all:7"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                admission.reverse_arrays(value)
        for mapping in ({True: 7}, {31: True}, {183: "7"}, {168: 7}, []):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                admission.validate_mapping(mapping)

    def test_both_real_producer_shapes_require_mixed1000_1500_and_exact_saved_baseline_scores(self):
        for index in admission.INDICES:
            reports = self.reports(index)
            fixed = fixtures.repair.admission.menu.fixed
            with self.subTest(index=index), patch.object(fixed, "validate_fit", wraps=fixed.validate_fit) as validate:
                scores = self.validate(reports, index)
            self.assertEqual(scores, reports[0]["participant_scores"])
            self.assertEqual([call.args[4]["optimizer"]["maximum_sweeps"] for call in validate.call_args_list][-2:], [1000, 1500])
            expected = ("Coordinator separately selects each diagnosed null31/183 reverse completion; no pilot reread or wave selection"
                        if index in (31, 183) else
                        "Coordinator separately selects each diagnosed indexed null reverse completion; no pilot reread or wave selection")
            self.assertEqual(reports[0]["numerical_override"]["qualification_boundary"], expected)
            for person, row in reports[2]["participant_scores"].items():
                self.assertEqual({key: scores[person][key] for key in row}, row)

    def test_wrong_identity_source_specs_replay_or_claims_cannot_supply_scores(self):
        original = self.reports()
        changes = (
            lambda rows: rows[0].update(phase=fixtures.repair.indexed.PHASE),
            lambda rows: rows[0].update(replicate_index=31),
            lambda rows: rows[0].update(job_id="other"),
            lambda rows: rows[0].update(source_report="wrong.json"),
            lambda rows: rows[0].update(source_job_id="other"),
            lambda rows: rows[0].update(source_finished_utc="different"),
            lambda rows: rows[0].update(failed_report="wrong.json"),
            lambda rows: rows[1].update(job_id="other"),
            lambda rows: rows[2].update(job_id="other"),
            lambda rows: rows[0]["runtime_implementation"]["optimizer"].update(subject_ftol=1e-8),
            lambda rows: rows[0]["numerical_override"].update(fresh_fit_directions=[]),
            lambda rows: rows[0]["numerical_repair"].update(nonconverged_checkpoint_state_reused=True),
            lambda rows: rows[0].update(saved_score_replay_max_absolute_discrepancy=2e-12),
            lambda rows: rows[0]["participant_scores"]["p000"].update(M0=-2.),
            lambda rows: rows[0]["fits"][1].update(objective_history=[1.] * 1501),
            lambda rows: rows[0].update(full_gate="pass"),
            lambda rows: rows[0].update(automatic_retry=True),
        )
        for position, change in enumerate(changes):
            reports = deepcopy(original)
            change(reports)
            with self.subTest(change=position), self.assertRaises((ValueError, RuntimeError)):
                self.validate(reports)

    def test_preserved_first_contents_rescoring_and_single_counted_cost_are_required(self):
        original = self.reports(31)
        changes = (
            lambda rows: rows[0]["fits"][0]["objective_history"].__setitem__(0, 270.),
            lambda rows: rows[0]["fits"][0].update(parameter_names=[]),
            lambda rows: rows[0].update(first_half_fit_reused=False),
            lambda rows: rows[0].update(first_half_rescored=False),
            lambda rows: rows[0].update(fresh_second_half_fit_called=False),
            lambda rows: rows[0].update(first_half_rescore_elapsed_seconds=-1.),
            lambda rows: rows[0].update(first_half_copy_elapsed_seconds=rows[0]["fits"][0]["elapsed_seconds"] + 1.),
            lambda rows: rows[0]["per_half_provenance"][0].update(validation_maximum_sweeps=500),
            lambda rows: rows[0]["numerical_repair"].update(first_half_original_fit_elapsed_seconds=0.),
            lambda rows: rows[0].update(cumulative_process_elapsed_seconds=
                rows[0]["cumulative_process_elapsed_seconds"] + rows[1]["fits"][0]["elapsed_seconds"]),
        )
        for position, change in enumerate(changes):
            reports = deepcopy(original)
            change(reports)
            with self.subTest(change=position), self.assertRaises((ValueError, RuntimeError)):
                self.validate(reports, 31)

    def test_technical_source_replay_rescore_fit_and_finalization_prefixes_remain_unknown(self):
        for failure, fit_count in (("source_timeout", 0), ("replay", 0), ("rescore", 1),
                                   ("fit_timeout", 1), ("nonconvergence", 2), ("finalization_timeout", 2)):
            reports = self.reports(31, failure)
            with self.subTest(failure=failure):
                evidence = self.validate(reports, 31)
                self.assertEqual(evidence["status"], "technical_rival_failure")
                self.assertEqual(len(evidence["returned_fits"]), fit_count)
                self.assertNotIn("participant_scores", evidence)
                self.assertEqual(evidence["elapsed_seconds"], reports[0]["elapsed_seconds"])
                if failure == "source_timeout":
                    self.assertNotIn("cumulative_process_elapsed_seconds", evidence)
                else:
                    self.assertEqual(evidence["cumulative_process_elapsed_seconds"],
                                     reports[1]["elapsed_seconds"] + reports[0]["elapsed_seconds"])
                if fit_count:
                    self.assertEqual(evidence["returned_fits"][0]["returned_sweeps"], 269)

    def test_missing_prior_stays_unadmitted_and_malformed_technical_claims_abort(self):
        reports = list(self.reports())
        reports[1] = None
        self.assertIsNone(self.validate(reports))
        reports[0]["phase"] = "wrong"
        with self.assertRaises(ValueError):
            self.validate(reports)
        original = self.reports(183, "nonconvergence")
        for field, value in (("participant_scores", {}), ("elapsed_seconds", -1.),
                             ("first_half_rescored", False), ("first_half_copy_elapsed_seconds", None),
                             ("saved_score_replay_max_absolute_discrepancy", 2e-12),
                             ("scored_primary_directions", [{"training_runs": list(train), "scoring_runs": list(held)}
                                                           for train, held in fixtures.repair.indexed.HALVES])):
            reports = deepcopy(original)
            reports[0][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, RuntimeError)):
                self.validate(reports)


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
            paths = pool.input_paths(index, mvc_reverse_arrays=mapping)
            allowed.update({paths[name]: json.dumps(report) for name, report in reports.items()})
            allowed[pool.simple.source_directory(index) / "generating_parameters.json"] = json.dumps(self.fixture.truth)
        before = deepcopy(panels)
        functions = (pool.rival_identity, pool.technical_rival, pool.expected_override, pool.fit_spec, pool.menu.validate_rival)
        bindings, reads = self.fixture.bindings(), []

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
                    result = pool.pool(selected, self.spec, mvc_reverse_arrays=mapping)
                else:
                    args = ["--terminal-indices", ",".join(map(str, selected)), "--implementation", str(pool.IMPLEMENTATION),
                            "--output", str(output)]
                    if mapping is not None:
                        args += ["--mvc-reverse-arrays", ",".join(f"{index}:{array}" for index, array in mapping.items())]
                    self.assertEqual(pool.main(args), 0)
                    result = None
            finally:
                self.assertEqual((pool.rival_identity, pool.technical_rival, pool.expected_override, pool.fit_spec, pool.menu.validate_rival), functions)
                self.assertEqual(self.fixture.bindings(), bindings)
        self.assertEqual(panels, before)
        return json.loads(output.read_text()) if output is not None else result, reads

    def test_pool_cli_admits_explicit_mixed_spec_reverses_with_unchanged_nine_contrasts_fixed200(self):
        pool = fixtures.repair.admission
        mapping = {183: 46600829, 31: 46601362, 12: 70000012, 109: 70000109}
        panels = {index: self.pool_panels(index) for index in mapping}
        with tempfile.TemporaryDirectory() as directory:
            result, reads = self.invoke_pool(list(mapping), panels, mapping, output=Path(directory) / "summary.json")
        self.assertEqual(result["completed_indices"], [12, 31, 109, 183])
        self.assertEqual(result["mvc_reverse_arrays"], {str(index): array for index, array in mapping.items()})
        self.assertEqual(result["mvc_repair_arrays"], {})
        self.assertEqual(result["diagnostic_counts"]["complete_menus"], 4)
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 196)
        for index, array in mapping.items():
            summary = result["cohorts"][index]["summary"]
            self.assertEqual(set(summary["nested_comparisons"]), {f"{model}-{nested}" for model, nested in pool.menu.fixed.EDGES})
            self.assertEqual(summary["mvc_reverse_array_job_id"], array)
            self.assertEqual([row["validation_maximum_sweeps"] for row in summary["mvc_per_half_provenance"]], [1000, 1500])
            self.assertEqual(summary["cumulative_mvc_process_elapsed_seconds"], panels[index]["MVC"]["cumulative_process_elapsed_seconds"])
            self.assertEqual(summary["mvc_first_half_copy_elapsed_seconds"], panels[index]["MVC"]["fits"][0]["elapsed_seconds"])
            for field in pool.CLAIM_FIELDS:
                self.assertEqual(summary[field], "not_evaluated")
            self.assertIn(admission.input_paths(index, mapping)["failed_MVC"], reads)
            self.assertIn(admission.input_paths(index, mapping)["MVC"], reads)

    def test_omitted_mapping_preserves_original_failure_and_never_reads_unselected_reverse(self):
        pool = fixtures.repair.admission
        panels = {183: self.pool_panels(183, mapped=False)}
        result, reads = self.invoke_pool([183], panels)
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["mvc_reverse_arrays"], {})
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        self.assertIn(fixtures.repair.failed_path(183), reads)
        self.assertFalse(any("MVC-reverse-completion-array" in str(path) for path in reads))
        with patch.object(Path, "read_text", side_effect=AssertionError("No selected reverse")):
            empty = pool.pool([], self.spec, mvc_repair_arrays={43: 46599566}, mvc_reverse_arrays={183: 46600829})
            self.assertEqual(empty["completed_indices"], [])
        original43 = pool.input_paths(43, mvc_repair_arrays={43: 46599566})
        self.assertEqual(pool.input_paths(43, mvc_repair_arrays={43: 46599566}, mvc_reverse_arrays={183: 46600829}), original43)
        panels[183]["MVC"] = self.reports()[0]
        with self.assertRaises(ValueError):
            self.invoke_pool([183], panels)

    def test_pool_technical_missing_and_malformed_reverses_remain_unknown_or_abort(self):
        pool = fixtures.repair.admission
        mapping = {183: 46600829, 31: 46601362, 12: 70000012}
        panels = {183: self.pool_panels(183, "nonconvergence"), 31: self.pool_panels(31), 12: self.pool_panels(12)}
        missing = (pool.input_paths(31, mvc_reverse_arrays=mapping)["failed_MVC"],
                   pool.input_paths(12, mvc_reverse_arrays=mapping)["MVZ"])
        result, reads = self.invoke_pool(list(mapping), panels, mapping, missing=missing)
        self.assertEqual(result["completed_indices"], [])
        self.assertEqual(result["diagnostic_counts"]["unknown_cohorts"], 200)
        evidence = result["cohorts"][183]["technical_components"]["MVC"]
        self.assertEqual(len(evidence["returned_fits"]), 2)
        self.assertEqual(evidence["cumulative_process_elapsed_seconds"], panels[183]["MVC"]["cumulative_process_elapsed_seconds"])
        self.assertEqual(result["cohorts"][31]["missing_reports"], [str(missing[0])])
        panels[12]["MVC"]["fits"][0]["objective_history"][0] += 1.
        with self.assertRaises(ValueError):
            self.invoke_pool(list(mapping), panels, mapping, missing=missing)
        with self.assertRaises(ValueError):
            pool.build_indexed_summary(panels[183]["baseline"], {model: panels[183][model] for model in pool.menu.RIVALS},
                self.spec, self.fixture.truth, self.people, 183, mvc_reverse_arrays=mapping, failed_mvc=panels[183]["failed_MVC"])


if __name__ == "__main__":
    unittest.main()
