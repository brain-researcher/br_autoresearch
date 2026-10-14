"""Two bounded composition regressions on existing fixtures; compute only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import continuous_fold_runner as folds
import continuous_tuning as tuning
import production_baseline_controller as baseline
import production_selected_design_cohort as cohort
import streamed_tuning_adapter as adapter
import test_continuous_tuning as original
from test_production_baseline_controller import FixtureBudget, plan_fixture, completed, write
from test_streamed_tuning_adapter import fixture, FINGERPRINTS, RESOURCES


CHOICE = tuning.BaselineChoice(6, 3)
MATCH = tuning.MatchedIndices(6, 6, (6,) * 8)
SELECTIONS = ((CHOICE, MATCH),) * 6


def diagnostic_fixture():
    cache, recipe, _, _ = fixture()
    value = adapter.outer_capacity_and_match(cache, CHOICE, recipe, lambda: None,
        resources=RESOURCES, diagnostic_recipe_id=cohort.RECIPE_ID,
        frozen_partition_fingerprints=FINGERPRINTS)
    return tuple(replace(value, selected_design=replace(value.selected_design, outer_index=i))
                 for i in range(6))


class SelectedDesignCohortTests(unittest.TestCase):
    def test_six_real_streamed_checks_use_saved_choices_without_candidate_fits(self):
        sequences = original.fixture()
        person = SimpleNamespace(sequences=sequences,
            inputs=SimpleNamespace(channel_order=original.CHANNELS), prepare_training=None)
        caches = [fixture(i)[0] for i in range(6)]
        recipe = fixture()[1]
        kept = []
        # Existing preparation is independently tested; isolate this new
        # composition while retaining the actual six streamed factorizations.
        with patch.object(cohort, "make_tuning_recipe", return_value=recipe), \
             patch.object(cohort, "solver_limits", return_value=original.LIMITS), \
             patch.object(tuning, "_prepare", side_effect=caches) as preparation, \
             patch.object(tuning, "run_tuned_participant", side_effect=AssertionError("no fixed-panel entry")), \
             patch.object(folds, "_fit", side_effect=AssertionError("no candidate fitting")), \
             patch.object(tuning, "_score", side_effect=AssertionError("no candidate scoring")):
            result = cohort.selected_person(person, SELECTIONS, FINGERPRINTS, {}, RESOURCES,
                check=lambda: None, retain=lambda outer, value: kept.append((outer, value)))
        self.assertEqual([call.args[1] for call in preparation.call_args_list],
                         [folds._split(i) for i in range(6)])
        self.assertEqual([outer for outer, _ in kept], list(range(6)))
        self.assertTrue(all(value.match == MATCH for _, value in kept))
        self.assertEqual(result["status"], "selected_design_checks_pass")
        self.assertFalse(result["false_bank_fully_qualified"])
        self.assertFalse(result["EEG_scored"])
        changed = ((CHOICE, tuning.MatchedIndices(6, 7, (6,) * 8)),) + SELECTIONS[1:]
        with patch.object(cohort, "make_tuning_recipe", return_value=recipe), \
             patch.object(cohort, "solver_limits", return_value=original.LIMITS), \
             patch.object(tuning, "_prepare", return_value=caches[0]):
            with self.assertRaisesRegex(ValueError, "no_penalty_fallback"):
                cohort.selected_person(person, changed, FINGERPRINTS, {}, RESOURCES,
                    check=lambda: None, retain=lambda *_: self.fail("changed match was published"))

    def test_original_frozen_artifacts_require_all30_all180_and_preserve_no_release(self):
        numerical = diagnostic_fixture()
        summary = adapter.assemble_outer_checks(numerical)
        budget = FixtureBudget()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            people = [f"person_{index:02d}" for index in range(30)]
            development, release_path, plan_path = base/"development.json", base/"release.json", base/"plan.json"
            write(development, {"role": "development", "participants": people})
            candidate_dirs = {}
            for person in people:
                candidate = base / ("candidate_" + person)
                candidate.mkdir()
                candidate_dirs[person] = str(candidate)
                write(candidate/"candidate_bank.json", {"participant_id": person,
                                                        "primary_bank_fingerprints": FINGERPRINTS})
            write(release_path, {"development_manifest": str(development), "false_candidate_dirs": candidate_dirs})
            plan = baseline.validate_plan(plan_fixture(release_path))
            write(plan_path, plan)
            search, output = base/"search", base/"selected"
            output.mkdir()
            baseline.initialize_search(str(plan_path), str(search), budget=budget)
            core = {"status": "complete_30_person_core_calibration", "plan_id": baseline.digest(plan),
                "trial_id": baseline.CORE_ID, "recipe_id": baseline.digest(plan["core_calibration_recipe"]),
                "participants": people, "participant_count": 30, "inner_splits": 900, "outer_splits": 180,
                "audit_access": False, "concept_or_false_scores_computed": False,
                "meaningful_margins": {"G": .1, "T": .1, "delta_C": .005, "delta_P": .0025,
                                       "delta_Q": .025, "delta_T": .025}, "positive_reference_gain": True}
            write(search/(baseline.CORE_ID+".complete.json"), core)
            for trial in plan["trials"][:22]:
                write(search/(trial["trial_id"]+".complete.json"), completed(plan, trial, 1., people))
            selected = plan["trials"][0]
            for person in people:
                directory = search / (selected["trial_id"] + "__" + person)
                directory.mkdir()
                write(directory/"complete.json", {"status": "complete_participant_baseline",
                    "plan_id": baseline.digest(plan), "trial_id": selected["trial_id"],
                    "recipe_id": baseline.digest(selected["recipe"]), "participant_id": person,
                    "inner_splits": 30, "outer_splits": 6,
                    "concept_or_false_scores_computed": False, "audit_access": False})
                for outer in range(6):
                    baseline._split_record(directory/("outer_%02d_result.json" % outer),
                        {"split": folds._split(outer), "choice": CHOICE, "match": MATCH}, budget)
            baseline.freeze_baseline(str(search), budget=budget)
            frozen = cohort.bind_frozen_baseline(str(search), budget=budget)
            self.assertEqual(frozen.trial, selected)
            self.assertEqual(cohort.saved_outer_selections(frozen, people[0], budget=budget), SELECTIONS)
            records = {}
            for person in people:
                directory = output/person
                directory.mkdir()
                for outer, value in enumerate(numerical):
                    baseline._split_record(directory/("outer_%02d.json" % outer), value, budget)
                records[person] = baseline._plain({"status": "complete_participant_selected_design", "scope": cohort.SCOPE,
                    "freeze_id": frozen.freeze_id, "plan_id": frozen.settings["plan_id"],
                    "trial_id": selected["trial_id"], "recipe_id": baseline.digest(selected["recipe"]),
                    "participant_id": person, "frozen_selections": SELECTIONS,
                    "partition_fingerprints": FINGERPRINTS, "result": summary,
                    "nuisance_preparation_performed": True, "concept_or_false_scores_computed": False,
                    "full_false_bank_qualified": False, "scientific_release_authorized": False,
                    "audit_access": False, "automatic_retry": False})
                if person != people[-1]:
                    write(directory/"complete.json", records[person])
            with self.assertRaisesRegex(ValueError, "bounded_metadata_file_required"):
                cohort.finalize(str(search), str(output), budget=budget)
            self.assertFalse((output/"cohort_complete.json").exists())
            write(output/people[-1]/"complete.json", records[people[-1]])
            changed = deepcopy(records[people[0]])
            changed["freeze_id"] = "0" * 64
            write(output/people[0]/"complete.json", changed)
            with self.assertRaisesRegex(ValueError, "binding_differs"):
                cohort.finalize(str(search), str(output), budget=budget)
            self.assertFalse((output/"cohort_complete.json").exists())
            write(output/people[0]/"complete.json", records[people[0]])
            changed = deepcopy(records[people[1]])
            changed["result"]["outer_folds"][-1]["outer_index"] = 0
            write(output/people[1]/"complete.json", changed)
            with self.assertRaisesRegex(ValueError, "coverage_differs"):
                cohort.finalize(str(search), str(output), budget=budget)
            write(output/people[1]/"complete.json", records[people[1]])
            result = cohort.finalize(str(search), str(output), budget=budget)
            self.assertEqual((result["participant_count"], result["outer_folds"], result["false_matches"]), (30, 180, 1440))
            self.assertEqual(result["status"], "all30_selected_design_pass")
            self.assertFalse(result["scientific_release_authorized"])
            final_path = output/"cohort_complete.json"
            preserved = final_path.read_bytes()
            with self.assertRaises(FileExistsError):
                cohort.finalize(str(search), str(output), budget=budget)
            self.assertEqual(final_path.read_bytes(), preserved)
            final = json.loads(preserved)
            self.assertFalse(final["full_false_bank_qualified"])
            self.assertFalse(final["concept_or_false_scores_computed"])
            self.assertFalse(final["audit_access"])


if __name__ == "__main__":
    unittest.main()
