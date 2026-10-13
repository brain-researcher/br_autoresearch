"""Focused record-only outer-stability cases; no neural data or fits.

Run only in coordinated Slurm. Reuse scalar fixture builders, not the existing
projector or nomination suites. These tests do not authenticate score chronology.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import component_nomination as nomination
import component_outer_stability as core
from test_component_nomination import comparison, panels, UnopenedRecords


def records(choices=None, animal="Chewie-L"):
    roster = nomination.TARGET_SESSIONS[animal]
    if choices is None:
        choices = ("C", "C", "C") if animal == "Mihili" else ("C", "C", "C", "D")
    nominated, evaluated = {}, {}
    for session, choice in zip(roster, choices):
        gains = {component: ((.08, .08) if component == choice else (-.04, -.04))
                 for component in nomination.COMPONENTS}
        values = panels(animal, gains)
        nominated[session] = nomination.nominate_outer_component(target_animal=animal,
            target_development_sessions=roster, left_out_session=session,
            comparisons={key: values[key] for key in roster if key != session})
        evaluated[session] = comparison(session, choice, (.001, .002))
    return nominated, evaluated


def assess(nominated, evaluated, animal="Chewie-L"):
    return core.assess_outer_stability(target_animal=animal,
        target_development_sessions=nomination.TARGET_SESSIONS[animal],
        nominations=nominated, left_out_comparisons=evaluated)


class ComponentOuterStabilityTests(unittest.TestCase):
    def test_all_three_mihili_or_three_of_four_chewie_without_preservation_claim(self):
        for animal in nomination.TARGET_SESSIONS:
            nominated, evaluated = records(animal=animal)
            first = nomination.TARGET_SESSIONS[animal][0]
            evaluated[first] = comparison(first, "C", (-.2, -.1))
            result = assess(nominated, evaluated, animal)
            self.assertEqual((result.status, result.stable_component), ("stable_selection", "C"))
            self.assertEqual(result.nomination_counts,
                (("C", 3), ("D", 0 if animal == "Mihili" else 1), ("E", 0)))
            self.assertEqual(len(result.folds), 3 if animal == "Mihili" else 4)
            self.assertTrue(all(len(fold.nomination.supplied_sessions) == len(result.folds) - 1
                                for fold in result.folds))
            self.assertTrue(result.no_tied_fold)
            self.assertTrue(result.no_left_out_sign_contradiction)
            self.assertFalse(result.establishes_component_claim)
            self.assertEqual(result.folds[0].regional_signs, (-1, -1))
            # Small positive and negative gains are legitimate stable-selection
            # records even though they do not establish preserved prediction.
            self.assertEqual(result.folds[0].comparison.status, "failed")
            self.assertEqual(result.folds[-1].comparison.status, "unresolved")
            self.assertIs(result.folds[-1].nomination, nominated[result.folds[-1].left_out_session])
            self.assertIs(core.bind_outer_assessment(result, target_animal=animal,
                target_development_sessions=nomination.TARGET_SESSIONS[animal]), result)

    def test_two_two_tie_and_fourth_nominees_sign_conflict_each_prevent_stability(self):
        # Two of three Mihili folds is insufficient under the adopted rule.
        nominated, evaluated = records(("C", "C", "D"), animal="Mihili")
        result = assess(nominated, evaluated, animal="Mihili")
        self.assertEqual(result.status, "unstable_selection")
        self.assertEqual(result.failed_conditions, ("same_component_in_at_least_three",))
        roster = nomination.TARGET_SESSIONS["Chewie-L"]
        nominated, evaluated = records(("C", "C", "D", "D"))
        result = assess(nominated, evaluated)
        self.assertEqual(result.failed_conditions, ("same_component_in_at_least_three",))
        nominated, evaluated = records()
        fourth = roster[-1]
        tied = panels(gains={"C": (.02, .02), "D": (.01, .01), "E": (-.04, -.04)})
        nominated[fourth] = nomination.nominate_outer_component(target_animal="Chewie-L",
            target_development_sessions=roster, left_out_session=fourth,
            comparisons={key: tied[key] for key in roster if key != fourth})
        evaluated[fourth] = None
        result = assess(nominated, evaluated)
        self.assertEqual(result.status, "unstable_selection")
        self.assertTrue(result.same_component_in_at_least_three)
        self.assertIn("no_tied_fold", result.failed_conditions)
        self.assertEqual(result.folds[-1].nomination.tied_components, ("C", "D"))
        for gains in ((.1, -.1), (0., -.1)):
            nominated, evaluated = records()
            evaluated[fourth] = comparison(fourth, "D", gains)
            result = assess(nominated, evaluated)
            self.assertEqual(result.status, "unstable_selection")
            self.assertEqual(result.failed_conditions, ("no_left_out_sign_contradiction",))
            self.assertFalse(result.folds[-1].signs_agree)

    def test_missing_and_unscoreable_evidence_never_imply_sign_agreement(self):
        nominated, evaluated = records()
        session = nomination.TARGET_SESSIONS["Chewie-L"][0]
        for record, reason in ((None, "missing_evaluation"),
                (comparison(session, "C", (.1, .1), scoreable=False), "unscoreable_evaluation")):
            evaluated[session] = record
            result = assess(nominated, evaluated)
            self.assertEqual(result.status, "unresolved_outer_stability")
            self.assertEqual(result.failed_conditions, ())
            self.assertEqual(result.unresolved_evidence, ((session, reason),))
            self.assertIsNone(result.no_left_out_sign_contradiction)
            self.assertIsNone(result.stable_component)
        # A recorded no-nominee fold cannot admit any selected evaluation.
        roster = nomination.TARGET_SESSIONS["Chewie-L"]
        values = panels(gains={component: (.1, -.1) for component in nomination.COMPONENTS})
        nominated[session] = nomination.nominate_outer_component(target_animal="Chewie-L",
            target_development_sessions=roster, left_out_session=session,
            comparisons={key: values[key] for key in roster if key != session})
        with self.assertRaises(core.OuterStabilityError):
            assess(nominated, evaluated)
        evaluated[session] = None
        result = assess(nominated, evaluated)
        self.assertIn((session, "no_prefold_nominee"), result.unresolved_evidence)
        # Three repeated nominations plus a fourth unscoreable nomination
        # cannot turn an unassessed sign condition into a positive finding.
        nominated, evaluated = records()
        fourth = roster[-1]
        values = panels()
        values[roster[0]]["E"] = comparison(roster[0], "E", (.1, .1), scoreable=False)
        nominated[fourth] = nomination.nominate_outer_component(target_animal="Chewie-L",
            target_development_sessions=roster, left_out_session=fourth,
            comparisons={key: values[key] for key in roster if key != fourth})
        evaluated[fourth] = None
        result = assess(nominated, evaluated)
        self.assertEqual(result.status, "unresolved_outer_stability")
        self.assertTrue(result.same_component_in_at_least_three)
        self.assertEqual(result.failed_conditions, ())

    def test_wrong_fold_component_role_or_forged_nomination_is_refused_before_use(self):
        nominated, evaluated = records()
        roster = nomination.TARGET_SESSIONS["Chewie-L"]
        with self.assertRaises(core.OuterStabilityError):
            assess(UnopenedRecords({**nominated, "Chewie_CO_20161006.mat": None}), evaluated)
        with self.assertRaises(core.OuterStabilityError):
            assess(nominated, UnopenedRecords({key: None for key in roster[:-1]}))
        forged = dict(nominated)
        forged[roster[0]] = replace(forged[roster[0]], nominee="E")
        with self.assertRaises(core.OuterStabilityError):
            assess(forged, UnopenedRecords(evaluated))
        for replacement in (evaluated[roster[1]], comparison(roster[0], "E", (.1, .1)),
                            {component: None for component in nomination.COMPONENTS}):
            changed = {**evaluated, roster[0]: replacement}
            with self.assertRaises(core.OuterStabilityError):
                assess(nominated, changed)
        wrong_role = evaluated[roster[0]]
        wrong_role = replace(wrong_role, m1=replace(wrong_role.m1,
            design=replace(wrong_role.m1.design, role="calibration")))
        with self.assertRaises(core.OuterStabilityError):
            assess(nominated, {**evaluated, roster[0]: wrong_role})
        recorded = assess(nominated, evaluated)
        # Final nomination binds derived state to the actual fold evidence,
        # not just a nominal stable-family string or the number of folds.
        for changed in (replace(recorded, stable_component="E"),
                        replace(recorded, status="unresolved_outer_stability"),
                        replace(recorded, folds=recorded.folds[::-1]),
                        replace(recorded, folds=(replace(recorded.folds[0], signs_agree=False),)
                            + recorded.folds[1:])):
            with self.assertRaises(core.OuterStabilityError):
                core.bind_outer_assessment(changed, target_animal="Chewie-L",
                                          target_development_sessions=roster)


if __name__ == "__main__":
    unittest.main()
