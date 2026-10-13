"""Focused nomination decisions from declared scalar score records, not fits.

No neural array, source file, bootstrap or repeated projector suite is used.
Run only in the coordinator's Slurm allocation; checks are initially unrun.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import component_nomination as nomination
from component_scoring import ComponentScore, FrozenDesign, RuleScore, compare_regions


def region_score(session, region, component, gain, *, scoreable=True):
    design = FrozenDesign(session, "evaluation", ("right", "left"),
        ("r1", "r2", "l1"), ("right", "right", "left"), (5, 6),
        (f"{region}-axis1", f"{region}-axis2"))
    if not scoreable:
        return ComponentScore(region, component, design, .001, "unscoreable",
            "target_energy_below_gate", .0001, (), None, None, None)
    rules = (RuleScore("correct", 1. - gain, gain), RuleScore("pooled", 1., 0.),
             RuleScore("swapped", 1.1, -.1))
    return ComponentScore(region, component, design, .001, "scoreable", None,
                          1., rules, gain, gain + .1, gain)


def comparison(session, component, gains, *, scoreable=True):
    return compare_regions(m1=region_score(session, "M1", component, gains[0], scoreable=scoreable),
                           pmd=region_score(session, "PMd", component, gains[1], scoreable=scoreable))


def panels(animal="Chewie-L", gains=None):
    gains = gains or {"C": (.05, .05), "D": (.01, .01), "E": (-.02, -.02)}
    return {session: {component: comparison(session, component, gains[component])
                      for component in nomination.COMPONENTS}
            for session in nomination.TARGET_SESSIONS[animal]}


def outer(values, animal="Chewie-L"):
    roster = nomination.TARGET_SESSIONS[animal]
    return nomination.nominate_outer_component(target_animal=animal,
        target_development_sessions=roster, left_out_session=roster[-1],
        comparisons={session: values[session] for session in roster[:-1]})


def outer_assessment(animal="Chewie-L", *, choices=None, missing=()):
    """Independent scalar records for the complete registered fold roster.

    These fixtures test binding and decision arithmetic, not actual chronology
    or whether models could produce these component scores.
    """
    import component_outer_stability as stability
    roster = nomination.TARGET_SESSIONS[animal]
    choices = choices or ("C",) * len(roster)
    nominated, evaluated = {}, {}
    for session, choice in zip(roster, choices):
        values = panels(animal, {component: ((.08, .08) if component == choice else (-.04, -.04))
                                  for component in nomination.COMPONENTS})
        nominated[session] = nomination.nominate_outer_component(target_animal=animal,
            target_development_sessions=roster, left_out_session=session,
            comparisons={key: values[key] for key in roster if key != session})
        evaluated[session] = None if session in missing else comparison(session, choice, (.03, .04))
    return stability.assess_outer_stability(target_animal=animal,
        target_development_sessions=roster, nominations=nominated, left_out_comparisons=evaluated)


DEFAULT_ASSESSMENT = object()


def final(values, animal="Chewie-L", *, assessment=DEFAULT_ASSESSMENT):
    if assessment is DEFAULT_ASSESSMENT:
        assessment = outer_assessment(animal)
    return nomination.nominate_final_component(target_animal=animal,
        target_development_sessions=nomination.TARGET_SESSIONS[animal],
        outer_assessment=assessment, comparisons=values)


class UnopenedRecords(dict):
    def __getitem__(self, key):
        raise AssertionError("rejected session/chronology must precede obtaining records")


class ComponentNominationTests(unittest.TestCase):
    def test_mixed_sign_rival_does_not_veto_but_numerical_missingness_does(self):
        values = panels(gains={"C": (-.1, -.1), "D": (.8, -.1), "E": (-.3, -.3)})
        result = outer(values)
        self.assertEqual((result.status, result.nominee), ("nominated", "C"))
        candidates = {candidate.component: candidate for candidate in result.candidates}
        self.assertTrue(candidates["C"].eligible)
        self.assertEqual(candidates["C"].median_gain, -.1)
        self.assertFalse(candidates["D"].eligible)
        self.assertEqual(candidates["D"].reason, "regional_sign_disagreement")
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.records[0][1][1].m1.regional_gain, .8)
        session = result.supplied_sessions[0]
        values[session]["D"] = comparison(session, "D", (.8, -.1), scoreable=False)
        unresolved = outer(values)
        self.assertEqual(unresolved.status, "unresolved_numerical_support")
        self.assertIsNone(unresolved.nominee)
        self.assertEqual(unresolved.unscoreable_sessions, (session,))
        # Zero versus negative is still a sign disagreement, despite the
        # session-level failure rule accepting both nonpositive values.
        zeros = outer(panels(gains={"C": (0., 0.), "D": (0., -.1), "E": (-.3, -.3)}))
        self.assertEqual(zeros.nominee, "C")
        self.assertFalse(zeros.candidates[1].eligible)
        # Mihili has only two prefold sessions: both are required, including
        # every rival's numerical support. Its median is the middle-pair mean.
        mihili = panels("Mihili")
        roster = nomination.TARGET_SESSIONS["Mihili"]
        self.assertEqual(roster, ("Mihili_CO_20140217.mat", "Mihili_CO_20140218.mat",
                                  "Mihili_CO_20140304.mat"))
        mihili[roster[0]]["C"] = comparison(roster[0], "C", (.09, .09))
        result = outer(mihili, "Mihili")
        self.assertEqual(result.supplied_sessions, roster[:-1])
        self.assertEqual(result.nominee, "C")
        self.assertAlmostEqual(result.candidates[0].median_gain, .07)
        mihili[roster[1]]["E"] = comparison(roster[1], "E", (-.02, -.02), scoreable=False)
        result = outer(mihili, "Mihili")
        self.assertEqual(result.status, "unresolved_numerical_support")
        self.assertIsNone(result.nominee)

    def test_final_nomination_requires_common_three_not_three_per_candidate(self):
        values = panels()
        roster = nomination.TARGET_SESSIONS["Chewie-L"]
        values[roster[0]]["C"] = comparison(roster[0], "C", (.05, .05), scoreable=False)
        # A mixed-sign rival on the excluded numerical-support session is
        # retained, but cannot change eligibility on the common three.
        values[roster[0]]["D"] = comparison(roster[0], "D", (.8, -.1))
        result = final(values)
        self.assertEqual(result.nominee, "C")
        self.assertEqual(result.ranking_sessions, roster[1:])
        self.assertEqual(len(result.records), 4)
        self.assertTrue(result.candidates[1].eligible)
        different_sets = panels()
        for session, component in zip(roster[:3], nomination.COMPONENTS):
            different_sets[session][component] = comparison(session, component, (.1, .1), scoreable=False)
        unsupported = final(different_sets)
        self.assertEqual(unsupported.status, "unresolved_numerical_support")
        self.assertIsNone(unsupported.nominee)
        self.assertEqual(unsupported.unscoreable_sessions, roster[:3])
        complete = final(panels())
        self.assertEqual(complete.ranking_sessions, roster)  # Do not drop the fourth common session.
        mihili = panels("Mihili")
        roster = nomination.TARGET_SESSIONS["Mihili"]
        self.assertEqual(final(mihili, "Mihili").ranking_sessions, roster)
        mihili[roster[-1]]["E"] = comparison(roster[-1], "E", (-.02, -.02), scoreable=False)
        unsupported = final(mihili, "Mihili")
        self.assertEqual(unsupported.status, "unresolved_numerical_support")
        self.assertEqual(unsupported.ranking_sessions, ())

    def test_median_selection_fixed_tie_margin_and_no_eligible_candidate(self):
        values = panels(gains={"C": (.02, .02), "D": (.05, .05), "E": (-.02, -.02)})
        session = nomination.TARGET_SESSIONS["Chewie-L"][2]
        values[session]["C"] = comparison(session, "C", (.8, .8))
        self.assertEqual(outer(values).nominee, "D")  # C wins only under an incorrect mean.
        tied = outer(panels(gains={"C": (.02, .02), "D": (.01, .01), "E": (-.02, -.02)}))
        self.assertEqual((tied.status, tied.tied_components), ("tied", ("C", "D")))
        self.assertIsNone(tied.nominee)
        beyond = outer(panels(gains={"C": (.021, .021), "D": (.01, .01), "E": (-.02, -.02)}))
        self.assertEqual(beyond.nominee, "C")
        none = outer(panels(gains={component: (.1, -.1) for component in nomination.COMPONENTS}))
        self.assertEqual(none.status, "no_eligible_candidate")
        self.assertIsNone(none.nominee)

    def test_wrong_sessions_chronology_and_record_binding_refuse(self):
        roster = nomination.TARGET_SESSIONS["Chewie-L"]
        for invalid in (UnopenedRecords({session: None for session in roster}),
                        UnopenedRecords({roster[0]: None, roster[1]: None, "Chewie_CO_20161006.mat": None})):
            with self.assertRaises(nomination.ComponentNominationError):
                nomination.nominate_outer_component(target_animal="Chewie-L",
                    target_development_sessions=roster, left_out_session=roster[3], comparisons=invalid)
        assessed = outer_assessment()
        # Neither a completion flag nor forged summaries can stand in for
        # the exact fold evidence. Refusal must precede any final-panel read.
        for invalid in (None, True, 1, outer_assessment("Mihili"),
                        replace(assessed, stable_component="D"),
                        replace(assessed, folds=assessed.folds[:-1]),
                        replace(assessed, folds=(replace(assessed.folds[0],
                            nomination=replace(assessed.folds[0].nomination, nominee="D")),)
                            + assessed.folds[1:])):
            with self.assertRaises(nomination.ComponentNominationError):
                final(UnopenedRecords({session: None for session in roster}), assessment=invalid)
        with self.assertRaises(nomination.ComponentNominationError):
            nomination.nominate_outer_component(target_animal="Chewie-L",
                target_development_sessions=roster[:-1], left_out_session=roster[0], comparisons={})
        wrong = panels()
        wrong[roster[0]]["C"] = wrong[roster[1]]["C"]
        with self.assertRaises(nomination.ComponentNominationError):
            outer(wrong)
        forged = panels()
        forged[roster[0]]["C"] = replace(forged[roster[0]]["C"], equal_region_gain=.9)
        with self.assertRaises(nomination.ComponentNominationError):
            outer(forged)
        mihili = nomination.TARGET_SESSIONS["Mihili"]
        for invalid_roster in (mihili + ("Mihili_CO_20140306.mat",), mihili[:-1],
                               (mihili[0], mihili[0], mihili[-1])):
            with self.assertRaises(nomination.ComponentNominationError):
                nomination.nominate_outer_component(target_animal="Mihili",
                    target_development_sessions=invalid_roster, left_out_session=mihili[-1],
                    comparisons=UnopenedRecords({session: None for session in mihili[:-1]}))

    def test_bidirectional_agreement_is_only_one_required_claim_condition(self):
        chewie, mihili = final(panels()), final(panels("Mihili"), "Mihili")
        result = nomination.compare_final_nominations(mihili_to_chewie=chewie, chewie_to_mihili=mihili)
        self.assertEqual((result.status, result.component), ("same_untied_final_nominee", "C"))
        self.assertFalse(result.establishes_component_claim)
        self.assertTrue(chewie.outer_stability_evaluated)
        self.assertEqual(chewie.outer_assessment.stable_component, "C")
        changed_values = panels("Mihili", {"C": (.01, .01), "D": (.1, .1), "E": (-.1, -.1)})
        changed = final(changed_values, "Mihili",
                        assessment=outer_assessment("Mihili", choices=("D", "D", "D")))
        self.assertEqual(nomination.compare_final_nominations(mihili_to_chewie=chewie,
            chewie_to_mihili=changed).status, "directional_disagreement")
        mismatched = final(changed_values, "Mihili")
        self.assertEqual((mismatched.status, mismatched.ranking_nominee, mismatched.nominee),
                         ("outer_family_mismatch", "D", None))
        first = nomination.TARGET_SESSIONS["Mihili"][0]
        for assessment, status in ((outer_assessment("Mihili", choices=("C", "C", "D")),
                                    "unstable_outer_selection"),
                                   (outer_assessment("Mihili", missing=(first,)),
                                    "unresolved_outer_stability")):
            blocked = final(panels("Mihili"), "Mihili", assessment=assessment)
            self.assertEqual((blocked.status, blocked.ranking_nominee, blocked.nominee),
                             (status, "C", None))
            self.assertEqual(nomination.compare_final_nominations(mihili_to_chewie=chewie,
                chewie_to_mihili=blocked).status, "unresolved_nomination")
        for forged in (replace(mismatched, status="nominated", nominee="D"),
                       replace(mihili, outer_assessment=None),
                       replace(mihili, ranking_sessions=mihili.ranking_sessions[:-1]),
                       replace(mihili, candidates=changed.candidates)):
            with self.assertRaises(nomination.ComponentNominationError):
                nomination.compare_final_nominations(mihili_to_chewie=chewie,
                                                      chewie_to_mihili=forged)
        with self.assertRaises(nomination.ComponentNominationError):
            nomination.compare_final_nominations(mihili_to_chewie=mihili, chewie_to_mihili=chewie)
        with self.assertRaises(nomination.ComponentNominationError):
            nomination.compare_final_nominations(mihili_to_chewie=outer(panels()), chewie_to_mihili=mihili)


if __name__ == "__main__":
    unittest.main()
