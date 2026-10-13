"""Source-free scientific/numerical fixtures for the adopted EP17 pilot."""

from collections import Counter
import csv
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import concept_role_assignment as method
import run_concept_role_assignment as runner
import reconcile_released_events as core


def concept(number):
    return method.Concept(number, f"concept_{number}", number, 5, 2)


def lag_row(repetition, seconds="", days="", rank=0, index=0, session=1):
    return {"source_repetition": str(repetition), "delay_seconds": str(seconds), "delay_days": str(days),
            "retained_acquisition_rank": str(rank), "glmsingle_trial_index_zero_based": str(index),
            "session": str(session), "atypical": "False", "not_for_memory": "False"}


def toy_raw(values):
    fields = [method.Field("fixture/value", "acquisition", Fraction(1))]
    return method.RawFeatures([concept(i + 1) for i in range(len(values))], fields,
                              [[Decimal(value)] for value in values], {})


class FeatureTests(unittest.TestCase):
    def test_lag_first_history_gap_and_conflicting_sentinels(self):
        self.assertEqual(method.lag_class(lag_row(1))[0], "expected_first")
        # An observed retained-first row can have an omitted earlier exposure.
        self.assertEqual(method.lag_class(lag_row(4, "12.5", "0"))[:2], ("within_seconds", Decimal("12.5")))
        self.assertEqual(method.lag_class(lag_row(2, "0", "3"))[:2], ("between_days", Decimal(3)))
        for item, reason in (
            (lag_row(2, "0", "0"), "both_zero_sentinels"),
            (lag_row(2, "2", "1"), "both_positive_delays"),
            (lag_row(2, "", "0"), "one_unknown_delay"),
            (lag_row(2), "nonfirst_both_unknown"),
            (lag_row(1, "1", "0"), "delay_on_source_first_exposure"),
        ):
            label, value, actual_reason = method.lag_class(item)
            self.assertEqual((label, value, actual_reason), ("unresolved", None, reason))

    def test_negative_or_nonfinite_delay_is_invalid_not_missing(self):
        for token in ("-1", "NaN", "Infinity"):
            with self.assertRaises(core.Refusal):
                method.lag_class(lag_row(2, token, "0"))

    def test_equal_image_weights_override_unequal_repeat_counts(self):
        images = {
            "b": [lag_row(1, rank=4, index=2, session=2),
                  lag_row(2, "0", "2", rank=6, index=3, session=2),
                  lag_row(3, "0", "2", rank=8, index=4, session=2),
                  lag_row(4, "0", "2", rank=10, index=5, session=2)],
            "a": [lag_row(1, rank=0, index=0), lag_row(2, "9", "0", rank=2, index=1)],
        }
        with localcontext() as context:
            context.prec, context.rounding = 80, ROUND_HALF_EVEN
            values, reasons, _ = method.participant_features(images, [1, 2], 11)
            reversed_input = {name: list(reversed(rows)) for name, rows in reversed(list(images.items()))}
            reordered = method.participant_features(reversed_input, [1, 2], 11)[0]
            self.assertEqual(values, reordered)
            self.assertEqual(values[:4], list(map(Decimal, ("0.5", "0.5", "0.4", "0.12"))))
            self.assertEqual(values[4:8], list(map(Decimal, ("0.375", "0.25", "0.375", "0"))))
            self.assertEqual(values[8], Decimal(10).ln())
            self.assertEqual(values[9], 0)
            self.assertEqual(values[10], Decimal(3).ln())
            self.assertEqual(values[11], 0)
            self.assertEqual(dict(reasons), {})

    def test_identical_nonterminating_log_moment_has_exact_zero_variance(self):
        with localcontext() as context:
            context.prec, context.rounding = 80, ROUND_HALF_EVEN
            mean, variance = method.log_moments([(Decimal(2), Fraction(1, 7))] * 7)
            self.assertEqual(mean, Decimal(3).ln())
            self.assertEqual(variance, Decimal(0))
            self.assertEqual(method.log_moments([]), (None, None))

    def test_family_and_participant_weights_and_unmapped_multilabel(self):
        concepts = [concept(1), concept(2)]
        rows = []
        for subject in method.SUBJECTS:
            for c in concepts:
                for image in ("a", "b"):
                    for rep in (1, 2):
                        item = lag_row(rep, "" if rep == 1 else "5", "" if rep == 1 else "0", rep - 1, rep - 1)
                        item.update({"participant": subject, "things_category_nr": str(c.cid), "eligible_image": "True",
                                     "expected_provider_image_path": f"{c.uid}/{image}"})
                        rows.append(item)
        memberships = {1: [1, 1] + [0] * 51, 2: [0] * 53}
        raw = method.build_features(concepts, [f"label_{i}" for i in range(53)], memberships, rows,
                                    {s: [1, 2] for s in method.SUBJECTS}, {s: 10 for s in method.SUBJECTS})
        families, participants = {}, {}
        for f in raw.fields:
            families[f.family] = families.get(f.family, Fraction()) + f.weight
            if f.participant:
                key = f.family, f.participant
                participants[key] = participants.get(key, Fraction()) + f.weight
        self.assertEqual(set(families.values()), {Fraction(1, 4)})
        self.assertEqual(set(participants.values()), {Fraction(1, 16)})
        self.assertEqual(raw.values[0][:3], [1, 1, 0])
        self.assertEqual(raw.values[0][53], 0)
        self.assertEqual(raw.values[1][53], 1)
        self.assertEqual(len(raw.fields), 104)


class PrecisionTests(unittest.TestCase):
    def test_exact_constants_and_wholly_missing_moments_are_zero_columns(self):
        with localcontext() as context:
            context.prec = 80
            constant = Decimal(3).ln()
        fields = [method.Field("s/within_log_mean", "lag", Fraction(1, 2)),
                  method.Field("s/between_log_mean", "lag", Fraction(1, 2))]
        raw = method.RawFeatures([concept(i) for i in (1, 2, 3)], fields,
                                [[constant, None], [None, None], [constant, None]], {})
        prepared = method.prepare(raw)
        self.assertEqual(prepared.q, [[0, 0], [0, 0], [0, 0]])
        self.assertEqual(prepared.imputed[1][0], constant)
        self.assertIsNone(prepared.raw[1][0])
        self.assertTrue(prepared.field_details[1]["wholly_unobserved"])
        self.assertEqual(prepared.field_details[0]["parent_variance"], "0")

    def test_imputation_and_scaling_use_only_parent_concepts(self):
        fields = [method.Field("s/within_log_mean", "lag", Fraction(1))]
        raw = method.RawFeatures([concept(i) for i in (1, 2, 3, 4)], fields,
                                [[Decimal(1)], [None], [Decimal(3)], [Decimal(1000)]], {})
        prepared = method.prepare(raw, {1, 2, 3})
        self.assertEqual(prepared.imputed[1][0], Decimal(2))
        self.assertEqual(prepared.field_details[0]["parent_mean"], "2")
        self.assertEqual([c.cid for c in prepared.concepts], [1, 2, 3])

    def test_half_even_quantum_and_negative_ties(self):
        values = ["0.00005", "0.00015", "0.00025", "-0.00005", "-0.00015"]
        self.assertEqual([method.quantize_z(Decimal(value)) for value in values], [0, 2, 2, 0, -2])

    def test_missing_nonlag_feature_and_bad_identifier_refuse(self):
        raw = toy_raw([1, 2])
        raw.values[0][0] = None
        with self.assertRaises(core.Refusal):
            method.prepare(raw)
        with self.assertRaises(core.Refusal):
            method.Concept(1, "bad\tname", 1, 5, 1).identity()


class ExactAssignmentTests(unittest.TestCase):
    def test_gram_delta_and_cache_agree_with_full_objective_after_every_swap(self):
        q = [[10**40, 2], [0, -3], [-10**40, 5], [7, 1], [-8, -2], [3, 6]]
        weights, sizes, labels = [2, 3], [2, 2, 2], [0, 0, 1, 1, 2, 2]
        gram = method.gram_matrix(q, weights)
        cache = method.caches(gram, labels, 3)
        for a, b in ((0, 2), (1, 4), (3, 5), (0, 3)):
            if labels[a] == labels[b]:
                continue
            before = method.objective(q, weights, labels, sizes)
            delta = method.swap_delta(gram, cache, labels, sizes, a, b)
            sums = method.group_feature_sums(q, labels, 3)
            self.assertEqual(delta, method.direct_delta(q, weights, sums, labels, sizes, a, b))
            method.apply_swap(gram, cache, labels, a, b)
            self.assertEqual(method.objective(q, weights, labels, sizes) - before, delta)
            self.assertEqual(cache, method.caches(gram, labels, 3))
            self.assertEqual(Counter(labels), {0: 2, 1: 2, 2: 2})
        self.assertGreater(abs(gram[0][0]), 2**63)

    def test_unequal_capacity_delta_matches_independent_rational_loss(self):
        q = [[-4, 2], [-1, 5], [2, 1], [3, -3], [7, 0], [8, 4]]
        weights, sizes, labels = [2, 7], [4, 1, 1], [0, 0, 0, 0, 1, 2]
        gram, cache = method.gram_matrix(q, weights), None
        cache = method.caches(gram, labels, 3)
        def loss(labels):
            result = Fraction()
            for j, weight in enumerate(weights):
                overall = Fraction(sum(row[j] for row in q), len(q))
                for group, n in enumerate(sizes):
                    mean = Fraction(sum(row[j] for row, label in zip(q, labels) if label == group), n)
                    result += weight * Fraction(n, len(q)) * (mean - overall) ** 2
            return result
        changed = list(labels)
        changed[0], changed[4] = changed[4], changed[0]
        # J omits only a constant and multiplies L by N*lcm(sizes).
        self.assertEqual(method.swap_delta(gram, cache, labels, sizes, 0, 4), (loss(changed) - loss(labels)) * 24)

    def test_hash_start_pair_order_and_capacity_are_reproducible(self):
        concepts = [concept(i) for i in range(1, 9)]
        first = method.initialize(concepts, [4, 2, 2], method.NAMESPACE)
        self.assertEqual(first, method.initialize(concepts, [4, 2, 2], method.NAMESPACE))
        self.assertEqual(Counter(first), {0: 4, 1: 2, 2: 2})
        pairs = method.pair_order(concepts, method.NAMESPACE)
        self.assertEqual(len(pairs), 28)
        self.assertEqual(set(pairs), {(a, b) for a in range(8) for b in range(a + 1, 8)})
        self.assertEqual(pairs, method.pair_order(concepts, method.NAMESPACE))

    def test_finite_sweep_cap_is_distinct_from_zero_swap_convergence(self):
        raw = toy_raw([0, 0, 0, 0])
        labels = method.initialize(raw.concepts, [2, 2], method.NAMESPACE)
        raw.values = [[Decimal(10 * label)] for label in labels]
        prepared = method.prepare(raw)
        with patch.object(method, "MAX_SWEEPS", 1):
            result, report = method.optimize(prepared, [2, 2], method.NAMESPACE)
        self.assertTrue(report["fixed_sweep_budget_exhausted"])
        self.assertFalse(report["pairwise_local_optimum"])
        self.assertLess(report["final_J"], report["initial_J"])
        self.assertEqual(Counter(result), {0: 2, 1: 2})
        _, converged = method.optimize(prepared, [2, 2], method.NAMESPACE)
        self.assertTrue(converged["pairwise_local_optimum"])
        self.assertEqual(converged["sweeps"][-1]["accepted_swaps"], 0)
        self.assertFalse(converged["global_optimum_claimed"])

    def test_equal_loss_swaps_rejected_and_deadline_raises(self):
        prepared = method.prepare(toy_raw([1, 1, 1, 1]))
        initial = method.initialize(prepared.concepts, [2, 2], method.NAMESPACE)
        labels, report = method.optimize(prepared, [2, 2], method.NAMESPACE)
        self.assertEqual(labels, initial)
        self.assertEqual(len(report["sweeps"]), 1)
        def expired():
            raise core.Refusal("attempt_deadline")
        with self.assertRaises(core.Refusal):
            method.optimize(prepared, [2, 2], method.NAMESPACE, expired)

    def test_child_block_api_never_receives_fold_labels(self):
        raw = toy_raw([1, 2, 3, 4])
        calls = []
        real_prepare = method.prepare
        def fake_prepare(argument, parent_ids):
            calls.append((argument, parent_ids))
            return real_prepare(argument, parent_ids)
        outer_labels = [0, 1]
        def fake_optimize(prepared, sizes, namespace, check_deadline):
            return (list(outer_labels) if namespace.endswith("/outer-fold") else [0, 1]), {}
        with patch.object(method, "prepare", side_effect=fake_prepare), \
             patch.object(method, "optimize", side_effect=fake_optimize) as optimize:
            first = method.assign_children(raw, {1, 2}, {3, 4})
            outer_labels.reverse()
            changed = method.assign_children(raw, {1, 2}, {3, 4})
        self.assertEqual([call[1] for call in calls], [{1, 2}, {1, 2}, {3, 4}] * 2)
        self.assertEqual([call.args[2] for call in optimize.call_args_list],
                         [method.NAMESPACE + suffix for suffix in ("/outer-fold", "/development-block", "/audit-block")] * 2)
        self.assertNotEqual(first["outer_folds"], changed["outer_folds"])
        self.assertEqual(first["development_blocks"], changed["development_blocks"])
        self.assertEqual(first["audit_blocks"], changed["audit_blocks"])
        self.assertNotIn("fold", " ".join(inspect.signature(method.assign_children).parameters))


class PilotPublicationTests(unittest.TestCase):
    def test_raw_feature_artifact_keeps_unobserved_cells_blank_and_no_roles(self):
        raw = method.RawFeatures([concept(1), concept(2)],
                                [method.Field("s/within_log_mean", "lag", Fraction(1))],
                                [[Decimal(2)], [None]], {})
        prepared = method.prepare(raw)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            runner.write_feature_artifact(out, raw, prepared)
            with (out / "raw_metadata_features.csv").open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(records[1]["s/within_log_mean"], "")
            definition = json.loads((out / "feature_definition.json").read_text())
            self.assertFalse(definition["roles_assigned"])
            self.assertFalse(definition["metadata_assignments_complete"])
            self.assertFalse(definition["source_roles_complete"])

    def test_deadline_after_publication_and_completion_clears_success_flags(self):
        raw = toy_raw([1, 2])
        real_complete = runner.complete_pilot
        def interrupted_completion(report, concepts, fields):
            real_complete(report, concepts, fields)
            raise core.Refusal("attempt_deadline")
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "load_inputs", return_value=()), \
             patch.object(method, "build_features", return_value=raw), \
             patch.object(runner, "pilot_measurements", return_value={"unbuffered_full_operation_cost_estimate_seconds": {}}), \
             patch.object(runner, "complete_pilot", side_effect=interrupted_completion):
            out = Path(temporary)
            self.assertEqual(runner.run_pilot(out), 2)
            self.assertTrue((out / "raw_metadata_features.csv").is_file())
            report = json.loads((out / "pilot_report.json").read_text())
            for field in ("pilot_complete", "metadata_feature_preparation_complete", "metadata_assignments_complete",
                          "source_roles_complete", "roles_assigned", "folds_created", "blocks_created"):
                self.assertFalse(report[field])
            self.assertEqual(report["status"], "stopped")


class FullAssignmentTests(unittest.TestCase):
    def test_balance_report_uses_exact_quantized_loss_and_preserves_coverage(self):
        fields = [method.Field("s/within_log_mean", "lag", Fraction(1))]
        raw = method.RawFeatures([concept(i) for i in range(1, 5)], fields,
                                [[Decimal(1)], [None], [Decimal(3)], [Decimal(4)]], {})
        prepared = method.prepare(raw)
        labels = [0, 0, 1, 1]
        report = method.balance_summary(prepared, labels, [2, 2], ["a", "b"])
        groups = report["field_residuals"][0]["groups"]
        self.assertEqual([(group["observed_concepts"], group["imputed_concepts"]) for group in groups], [(1, 1), (2, 0)])
        expected = sum(Fraction(1, 2) * Fraction(group["quantized_residual_numerator"],
                                               group["quantized_residual_denominator"]) ** 2 for group in groups)
        loss = report["family_losses"]["lag"]
        self.assertEqual(Fraction(loss["quantized_loss_numerator"], loss["quantized_loss_denominator"]), expected)
        self.assertTrue(report["imputed_values_are_not_observed_delays"])
        self.assertTrue(report["assignment_selection_used_only_quantized_objective"])

    def test_pilot_raw_decimal_and_missingness_roundtrip_without_recomputation(self):
        value = Decimal("1.23456789012345678901234567890123456789")
        concepts = [method.Concept(i, f"concept_{i}", i, 6 if i <= 34 else 5, 6 if i <= 34 else 5)
                    for i in range(1, 721)]
        fields = [method.Field("s/within_log_mean", "lag", Fraction(1, 2)),
                  method.Field("s/between_log_mean", "lag", Fraction(1, 2))]
        raw = method.RawFeatures(concepts, fields, [[value if i % 2 else None, None] for i in range(720)], {})
        prepared = method.prepare(raw)
        pilot = {"status": "completed", "pilot_complete": True, "metadata_feature_preparation_complete": True,
                 "metadata_assignments_complete": False, "roles_assigned": False, "source_roles_complete": False,
                 "errors": [], "concept_count": 720, "independent_delta_checks": 4096,
                 "source_based_swaps_accepted": 0, "initialization_restored": True,
                 "source_inputs": [str(path) for path in runner.INPUT_PATHS], "field_count": 2, "feature_count": 2}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            runner.write_feature_artifact(out, raw, prepared)
            files = {runner.PILOT / "pilot_report.json": json.dumps(pilot).encode(),
                     runner.PILOT / "feature_definition.json": (out / "feature_definition.json").read_bytes(),
                     runner.PILOT / "raw_metadata_features.csv": (out / "raw_metadata_features.csv").read_bytes()}
            for path, flag in ((runner.QUALIFIED / "qualification_report.json", "eligibility_complete"),
                               (runner.CHRONOLOGY / "chronology_report.json", "source_chronology_complete")):
                files[path] = json.dumps({"status": "completed", flag: True, "errors": [],
                                         "roles_assigned": False, "source_roles_complete": False}).encode()
            with patch.object(runner.stage, "bounded_read", side_effect=lambda path: files[path]):
                loaded, details = runner.load_pilot_features()
            self.assertEqual(loaded.values, raw.values)
            self.assertEqual(loaded.values[1][0], value)
            self.assertIsNone(loaded.values[0][0])
            self.assertEqual(method.prepare(loaded).field_details, details)

    def inherited_fixture(self):
        c = concept(1)
        image = {"provider_image_path": "images_fmri/concept_1/a.jpg", "image_name": "a.jpg",
                 "concept_name": c.uid, "things_category_nr": "1", "provider_category_nr": "1",
                 "provider_exemplar_nr": "1", "condition": "exp", "eligible_image": "False",
                 "common_to_four": "True", "failure_reasons": "fewer_than_two_retained_presentations",
                 **{f"retained_count_{subject}": "1" for subject in method.SUBJECTS}}
        assignment = {1: {"role": "development", "outer_fold": "fold01", "development_block": "block03", "audit_block": ""}}
        image_rows, by_path = runner.attach_image_roles([image], [c], assignment)
        rows = [{"participant": subject, "glmsingle_trial_index_zero_based": "7", "things_category_nr": "1",
                 "expected_provider_image_path": image["provider_image_path"], "source_image_basename": "a.jpg",
                 "concept_name": c.uid, "provider_category_nr": "1", "provider_exemplar_nr": "1", "condition": "exp",
                 "eligible_image": "False", "atypical": "True", "delay_days": "2.5", "source_order": "7",
                 "retained_acquisition_rank": "3", "repeat_half": ""} for subject in method.SUBJECTS]
        return image, image_rows, by_path, rows, assignment

    def test_ineligible_image_and_all_presentations_inherit_whole_concept_role(self):
        image, images, by_path, rows, assignments = self.inherited_fixture()
        output = runner.attach_presentation_roles(rows, by_path, assignments)
        self.assertEqual(images[0]["role"], "development")
        self.assertEqual(images[0]["eligible_image"], "False")
        self.assertEqual({key: images[0][key] for key in image}, image)
        for original, augmented in zip(rows, output):
            self.assertEqual({key: augmented[key] for key in original}, original)
            self.assertEqual({key: augmented[key] for key in runner.ROLE_FIELDS}, assignments[1])

    def test_changed_image_anchor_or_duplicate_presentation_refuses_inheritance(self):
        _, _, by_path, rows, assignments = self.inherited_fixture()
        changed = [dict(row) for row in rows]
        changed[0]["eligible_image"] = "True"
        with self.assertRaises(core.Refusal):
            runner.attach_presentation_roles(changed, by_path, assignments)
        with self.assertRaises(core.Refusal):
            runner.attach_presentation_roles(rows + [dict(rows[0])], by_path, assignments)

    def test_failed_fourth_assignment_never_reaches_map_publication(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "load_pilot_features", return_value=(toy_raw([1, 2]), [])), \
             patch.object(runner, "execute_assignments", side_effect=core.Refusal("fourth_assignment_deadline")), \
             patch.object(runner, "publish_assignments") as publish:
            out = Path(temporary)
            self.assertEqual(runner.run_full(out), 2)
            publish.assert_not_called()
            self.assertEqual({path.name for path in out.iterdir()}, {"assignment_report.json"})
            report = json.loads((out / "assignment_report.json").read_text())
            self.assertFalse(report["metadata_assignments_complete"])
            self.assertFalse(report["roles_assigned"])
            self.assertFalse(report["source_roles_complete"])

    def test_full_completion_deadline_clears_roles_and_keeps_handoff_closed(self):
        real_complete = runner.complete_full
        def interrupted_completion(report):
            real_complete(report)
            raise core.Refusal("attempt_deadline")
        def partial_publication(out, *args):
            (out / "concept_assignments.csv").write_text("things_category_nr,role\n1,development\n")
            return {"concepts": 1}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "load_pilot_features", return_value=(toy_raw([1, 2]), [])), \
             patch.object(runner, "execute_assignments", return_value=({1: {"role": "development"}}, {"assignments": {}})), \
             patch.object(runner, "publish_assignments", side_effect=partial_publication), \
             patch.object(runner, "complete_full", side_effect=interrupted_completion):
            out = Path(temporary)
            self.assertEqual(runner.run_full(out), 2)
            report = json.loads((out / "assignment_report.json").read_text())
            for field in ("metadata_assignments_complete", "roles_assigned", "folds_created", "blocks_created",
                          "source_roles_complete", "physical_role_isolation_verified", "neural_handoffs_materialized"):
                self.assertFalse(report[field])
            self.assertTrue((out / "concept_assignments.csv").exists())
            self.assertEqual(report["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
