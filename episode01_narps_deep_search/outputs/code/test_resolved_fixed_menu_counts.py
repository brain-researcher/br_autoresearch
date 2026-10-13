"""Authored qualified-contrast fixtures; no likelihoods, fits or source events."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import summarize_resolved_fixed_menu_counts as counts


class ResolvedMenuCountsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = counts.load_helpers()
        cls.spec = json.loads(counts.IMPLEMENTATION.read_text())
        import run_twice_mz_menu_pilot as arithmetic
        cls.arithmetic = arithmetic

    def scores(self, kind):
        values = dict.fromkeys(self.helpers.fixed.MODELS, -1.)
        if kind == "MZ":
            values["MZ"] = -.97
        elif kind == "MV":
            values["MV"] = -.97
        elif kind == "MVZ":
            values.update(MV=-.975, MZ=-.975, MVZ=-.94)
        elif kind == "MVC":
            values.update(MV=-.975, MC=-.975, MVC=-.94)
        elif kind == "incomparable":
            values.update(MZ=-.97, MC=-.97)
        elif kind != "M0":
            raise ValueError(kind)
        return {f"p{index:03}": {"version": "EI" if index < 54 else "ER", **values} for index in range(108)}

    def row(self, cell, index, kind):
        expected = counts.expected_identity(cell, index, self.helpers)
        summary = {"analysis_id": "behavioral_mechanism_kill_test", "phase": expected["phase"],
            "generating_model": counts.GENERATORS[cell], "replicate_index": index, "seed": expected["seed"],
            "source_dir": expected["source_dir"], "source_finished_utc": "2026-10-05T10:00:00+00:00",
            "implementation": deepcopy(self.spec), "participants": 108, "runs": 432, "trials": 27648,
            "model_menu": list(self.helpers.fixed.MODELS),
            "family_selection": "not_evaluated", "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated",
            "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False}
        scores = self.scores(kind)
        summary.update(self.arithmetic.menu_diagnostics(scores, SimpleNamespace(menu=self.helpers.fixed)))
        # Actual indexed MZ summaries omit this panel; null summaries and MZ0
        # retain it. The new interpretation consumes only their stored contrasts.
        if cell == "M0_null" or index == 0:
            summary["participant_scores"] = scores
        return {"replicate_index": index, "seed": expected["seed"], "input_reports": expected["paths"],
                "status": "nested_diagnostics_completed" if cell == "MZ_SESOI" else "complete_menu_diagnostic", "summary": summary}

    def refresh(self, document, cell):
        complete = [row for row in document["cohorts"] if row["status"] != "unknown"]
        unknown = [row["replicate_index"] for row in document["cohorts"] if row["status"] == "unknown"]
        maxima = Counter(tuple(row["summary"]["maximal_eligible_models_diagnostic"]) for row in complete)
        document["completed_indices"] = [row["replicate_index"] for row in complete]
        if cell == "MZ_SESOI":
            document["unknown_indices"] = unknown
            document["diagnostic_counts"] = {"completed_menus": len(complete), "unknown_menus": len(unknown),
                "maximal_eligible_model_sets_diagnostic": [{"models": list(models), "menus": count} for models, count in sorted(maxima.items())],
                "menus_with_multiple_incomparable_eligible_models": sum(count for models, count in maxima.items() if len(models) > 1)}
        else:
            document["unresolved_indices"] = unknown
            document["diagnostic_counts"] = {"complete_menus": len(complete), "M0_only_maximal_diagnostics": maxima[("M0",)],
                "multiple_incomparable_diagnostics": sum(count for models, count in maxima.items() if len(models) > 1),
                "unknown_cohorts": len(unknown)}

    def documents(self):
        documents = {}
        for cell in counts.SOURCES:
            document = {"analysis_id": "behavioral_mechanism_kill_test",
                "phase": self.helpers.mz.PHASE if cell == "MZ_SESOI" else self.helpers.null.PHASE,
                "generating_model": counts.GENERATORS[cell], "implementation": deepcopy(self.spec),
                "fixed_cohort_count": 200, "selected_terminal_indices": [i for i in range(200) if cell != "MZ_SESOI" or i != 1],
                "family_selection": "not_evaluated", "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated",
                "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
                "cohorts": [self.row(cell, i, counts.GENERATORS[cell]) if cell != "MZ_SESOI" or i != 1 else
                    {"replicate_index": 1, "status": "unknown", "reason": "MC1_interrupted_retry_held"} for i in range(200)]}
            if cell == "MZ_SESOI":
                document.update(fresh_array_job_id=counts.MZ_FRESH, reverse_array_job_id=counts.MZ_REVERSE)
            else:
                document.update(additional_mvc_array_job_id=counts.NULL_ADDITIONAL,
                    mvc_repair_arrays={str(i): array for i, array in counts.NULL_REPAIRS.items()},
                    mvc_reverse_arrays={str(i): array for i, array in counts.NULL_REVERSES.items()})
            self.refresh(document, cell)
            documents[cell] = document
        return documents

    def invoke(self, documents, *, expiry=None, missing=None):
        before, reads, now = deepcopy(documents), [], [100.]
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
        real_reader, real_writer = Path.read_text, counts.write_report

        def reader(path, *args, **kwargs):
            reads.append(path)
            if path == counts.IMPLEMENTATION:
                return json.dumps(self.spec)
            for cell, selected in counts.SOURCES.items():
                if path == selected:
                    if cell == missing:
                        raise FileNotFoundError(str(path))
                    if expiry == "second_read" and cell == "M0_null":
                        now[0] = 341.
                    return json.dumps(documents[cell], allow_nan=False)
            raise AssertionError(f"Only the two preserved summaries and original specification may be read: {path}")

        def writer(path, result):
            real_writer(path, result)
            if expiry == "publication" and result["status"] == "resolved_counts_completed":
                now[0] = 341.

        def alarm(seconds):
            if expiry == "cleanup" and seconds == 0:
                now[0] = 341.

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory) / "resolved.json"
            for context in (patch.object(counts, "require_compute"), patch.object(counts, "load_helpers", return_value=self.helpers),
                    patch.dict(os.environ, {key: "1" for key in counts.THREAD_ENV}), patch.object(Path, "read_text", reader),
                    patch.object(counts.time, "monotonic", side_effect=lambda: now[0]),
                    patch.object(counts.signal, "alarm", side_effect=alarm), patch.object(counts, "write_report", side_effect=writer),
                    redirect_stdout(io.StringIO())):
                stack.enter_context(context)
            code = counts.main(["--implementation", str(counts.IMPLEMENTATION), "--output", str(output), "--max-seconds", "240"])
            result = json.loads(real_reader(output))
        self.assertEqual(documents, before)
        self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
        return result, code, reads

    def assert_scope(self, result):
        self.assertEqual(result["full_gate"], "not_evaluated")
        self.assertEqual(result["incomparable_arbitration"], "not_evaluated")
        self.assertIsNone(result["full_recovery_pass"])
        self.assertFalse(result["paired_statistics_independently_recomputed"])
        self.assertFalse(result["participant_score_panels_consumed"])
        for field in ("new_fits", "new_simulation", "saved_scores_rescored", "bootstrap_performed", "confidence_intervals", "automatic_retry"):
            self.assertFalse(result[field])
        for cell in result["cells"].values():
            self.assertEqual(cell["known_correct"] + cell["known_incorrect"] + cell["unknown"], 200)
            self.assertEqual([row["replicate_index"] for row in cell["cohorts"]], list(range(200)))

    def test_pure_resolver_simple_null_nested_joint_and_incomparable_sets(self):
        for kind, model, mechanisms in (("MZ", "MZ", ["z"]), ("M0", "M0", []),
                ("MVZ", "MVZ", ["eta", "z"]), ("MVC", "MVC", ["c", "eta"]), ("incomparable", None, None)):
            with self.subTest(kind=kind):
                source = self.arithmetic.menu_diagnostics(self.scores(kind),
                    SimpleNamespace(menu=self.helpers.fixed))["nested_comparisons"]
                before = deepcopy(source)
                result = counts.resolve_contrasts(source, self.helpers)
                self.assertEqual((result["resolved_model"], result["resolved_mechanisms"]), (model, mechanisms))
                self.assertEqual(len(result["nested_comparisons"]), 9)
                self.assertEqual(source, before)
                if model is None:
                    self.assertEqual(result["maximal_eligible_models_diagnostic"], ["MZ", "MC"])

    def test_fixed199_plus_unknown_and_null200_are_counted_without_excluding_held_index(self):
        documents = self.documents()
        self.assertNotIn("participant_scores", documents["MZ_SESOI"]["cohorts"][2]["summary"])
        # An optional panel is not an input to this derived interpretation.
        documents["M0_null"]["cohorts"][0]["summary"]["participant_scores"] = "not consumed"
        result, code, reads = self.invoke(documents)
        self.assertEqual(code, 0)
        self.assertEqual(reads, [counts.IMPLEMENTATION, *counts.SOURCES.values()])
        mz, null = result["cells"]["MZ_SESOI"], result["cells"]["M0_null"]
        self.assertEqual((mz["known_correct"], mz["known_incorrect"], mz["unknown"]), (199, 0, 1))
        self.assertEqual(mz["correct_fraction_bounds"], [.995, 1.])
        self.assertEqual(mz["cohorts"][1]["decision_reason"], "MC1_interrupted_retry_held")
        self.assertEqual(mz["cohorts"][2]["nested_comparisons"],
                         documents["MZ_SESOI"]["cohorts"][2]["summary"]["nested_comparisons"])
        self.assertEqual(null["known_correct"], 200)
        for component in null["false_component_counts"].values():
            self.assertEqual((component["known_false_selections"], component["unknown"], component["fraction_bounds"]), (0, 0, [0., 0.]))
        self.assert_scope(result)

    def test_incorrect_incomparable_and_technical_missing_rows_keep_conservative_fixed_bounds(self):
        documents = self.documents()
        mz, null = documents["MZ_SESOI"], documents["M0_null"]
        mz["cohorts"][2] = self.row("MZ_SESOI", 2, "MVZ")
        mz["cohorts"][3] = self.row("MZ_SESOI", 3, "incomparable")
        for index, reason in ((4, "technical_component_failure"), (5, "selected_report_missing")):
            mz["cohorts"][index] = {"replicate_index": index, "status": "unknown", "reason": reason,
                "input_reports": counts.expected_identity("MZ_SESOI", index, self.helpers)["paths"]}
        for index, kind in ((2, "MV"), (3, "MZ"), (4, "MVC"), (5, "incomparable")):
            null["cohorts"][index] = self.row("M0_null", index, kind)
        self.refresh(mz, "MZ_SESOI")
        self.refresh(null, "M0_null")
        result, code, _ = self.invoke(documents)
        self.assertEqual(code, 0)
        mz, null = result["cells"]["MZ_SESOI"], result["cells"]["M0_null"]
        self.assertEqual((mz["known_correct"], mz["known_incorrect"], mz["unknown"], mz["incomparable_count"]), (195, 1, 4, 1))
        self.assertEqual(mz["correct_fraction_bounds"], [.975, .995])
        self.assertEqual((null["known_correct"], null["known_incorrect"], null["unknown"]), (196, 3, 1))
        for category, expected in (("valuation", 2), ("bias", 2), ("any_component", 3)):
            component = null["false_component_counts"][category]
            self.assertEqual(component["known_false_selections"], expected)
            self.assertEqual(component["fraction_bounds"], [expected / 200, (expected + 1) / 200])
        self.assert_scope(result)

    def test_altered_contrast_identity_support_margins_eligibility_or_source_are_rejected(self):
        source = self.row("MZ_SESOI", 2, "MZ")["summary"]
        expected = counts.expected_identity("MZ_SESOI", 2, self.helpers)
        mutations = (lambda r: r["nested_comparisons"]["MZ-M0"].update(mean_gain_nats_per_trial=-.1),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(participant_SE=-.1),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(participant_SE=float("nan")),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(mean_gain_nats_per_trial=float("inf")),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(model="MC"),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(nested_baseline="MV"),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(participants=107),
            lambda r: r["nested_comparisons"].pop("MVC-MC"),
            lambda r: r["nested_comparisons"]["MZ-M0"].update(qualifies_nested_comparison=False),
            lambda r: r["nested_comparisons"]["MZ-M0"]["requirements"].update(mean_gain_at_least_0p01=False),
            lambda r: r["nested_comparison_eligibility"].update(MZ=False),
            lambda r: r.update(maximal_eligible_models_diagnostic=["MC"]),
            lambda r: r["implementation"]["optimizer"].update(maximum_sweeps=1500),
            lambda r: r["nested_comparisons"]["MZ-M0"]["version_means"].update(EI=float("nan")),
            lambda r: r.update(seed=123), lambda r: r.update(source_dir="/different/source"),
            lambda r: r.update(phase="MZ0_fixed_menu_nested_diagnostics"))
        for mutation in mutations:
            changed = deepcopy(source)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                counts.validate_saved_menu(changed, "MZ_SESOI", 2, expected, self.spec, self.helpers)
        # The policy uses an inclusive 0.01 margin but a strict two-SE margin.
        for mean, se, qualifies in ((.01, .004, True), (.01, .005, False), (.009, 0., False)):
            changed = deepcopy(source)
            contrast = changed["nested_comparisons"]["MZ-M0"]
            contrast.update(mean_gain_nats_per_trial=mean, participant_SE=se,
                requirements={"mean_gain_at_least_0p01": mean >= .01,
                    "mean_gain_greater_than_two_participant_SE": mean > 2 * se},
                qualifies_nested_comparison=qualifies)
            result = counts.resolve_contrasts(changed["nested_comparisons"], self.helpers)
            self.assertEqual(result["nested_comparison_eligibility"]["MZ"], qualifies)

    def test_duplicate_or_missing_indices_array_substitution_and_stored_counts_are_rejected(self):
        document = self.documents()["M0_null"]
        for mutation in (lambda r: r["cohorts"].pop(), lambda r: r["cohorts"][1].update(replicate_index=0),
                lambda r: r.update(additional_mvc_array_job_id=1),
                lambda r: r["cohorts"][0]["input_reports"].update(MVC="/different/attempt.json"),
                lambda r: r["diagnostic_counts"].update(M0_only_maximal_diagnostics=199)):
            changed = deepcopy(document)
            mutation(changed)
            with self.assertRaises(ValueError):
                counts.validate_cell(changed, "M0_null", self.spec, self.helpers)

    def test_missing_summary_and_mid_read_expiry_publish_partial_evidence_without_final_bounds(self):
        documents = self.documents()
        for options in ({"missing": "M0_null"}, {"expiry": "second_read"}):
            with self.subTest(options=options):
                result, code, _ = self.invoke(documents, **options)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "technical_resolved_count_failure")
                self.assertEqual(result["cells"]["MZ_SESOI"]["known_correct"], 199)
                self.assertEqual(result["cells"]["M0_null"]["unknown"], 200)
                for cell in result["cells"].values():
                    self.assertFalse(cell["source_validation_complete"])
                    self.assertIsNone(cell["correct_fraction_bounds"])
                self.assert_scope(result)

    def test_publication_and_handler_cleanup_deadlines_withhold_final_qualification(self):
        documents = self.documents()
        for expiry in ("publication", "cleanup"):
            with self.subTest(expiry=expiry):
                result, code, _ = self.invoke(documents, expiry=expiry)
                self.assertEqual(code, 1)
                self.assertEqual(result["status"], "technical_resolved_count_failure")
                self.assertEqual(result["cells"]["MZ_SESOI"]["known_correct"], 199)
                self.assertEqual(result["cells"]["M0_null"]["known_correct"], 200)
                self.assertTrue(all(cell["counts_provisional"] for cell in result["cells"].values()))
                self.assertTrue(all(cell["correct_fraction_bounds"] is None for cell in result["cells"].values()))
                self.assert_scope(result)


if __name__ == "__main__":
    unittest.main()
