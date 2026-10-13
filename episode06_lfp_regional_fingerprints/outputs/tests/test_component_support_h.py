"""Five focused scalar H cases; no real profiles, neural arrays or source reads.

Run only in the coordinator's Slurm allocation. The existing scalar nomination
fixtures provide selection records; this does not rerun projector/fit suites.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import component_nomination as nomination
import component_support_h as core
from test_component_nomination import final, outer, outer_assessment, panels, UnopenedRecords


def support(session, reliability=0., distance=4., component="C"):
    return core.CalibrationSupport(session, component, "calibration", reliability, distance, True)


def references(keys):
    values = (-3., -2., -1., 1., 2., 3.) if len(keys) == 6 else (-3., -2., -1., 0., 1., 2., 3.)
    return {key: support(key, value, value + 4.) for key, value in zip(keys, values)}


def agreement():
    return nomination.compare_final_nominations(
        mihili_to_chewie=final(panels("Chewie-L"), "Chewie-L"),
        chewie_to_mihili=final(panels("Mihili"), "Mihili"))


class UnreadScalars(core.CalibrationSupport):
    def __getattribute__(self, name):
        if name in ("reliability", "profile_distance", "bandwise_distance_scale_valid"):
            raise AssertionError("all session/component/role metadata must bind before scalar access")
        return super().__getattribute__(name)


class ComponentSupportTests(unittest.TestCase):
    def test_fixed_arithmetic_inclusive_thresholds_and_negative_reliability(self):
        selected = outer(panels())
        session = selected.left_out_session
        keys = tuple(key for key in core.DEVELOPMENT_SESSIONS if key != session)
        refs = references(keys)
        scale = core.MAD_FACTOR * 2.
        for h, expected in ((.25, "preserved"), (-.25, "failed"), (0., None)):
            result = core.predict_outer_support(nomination_result=selected,
                target=support(session, h * scale), references=refs)
            self.assertEqual(result.support_index, h)
            self.assertEqual(result.prediction, expected)
            self.assertEqual(result.status, "predicted" if expected else "abstained")
            self.assertEqual((result.reliability_median, result.distance_median), (0., 4.))
            self.assertEqual((result.reliability_scale, result.distance_scale), (scale, scale))
            self.assertFalse(result.establishes_component_claim)
        # Equal input weights: raising distance by the same standardized amount
        # cancels a reliability increase, without fitting an outcome coefficient.
        result = core.predict_outer_support(nomination_result=selected,
            target=support(session, scale, 4. + scale), references=refs)
        self.assertAlmostEqual(result.support_index, 0.)
        self.assertIsNone(result.prediction)

    def test_other_six_and_all_seven_references_keep_canonical_roles(self):
        self.assertEqual(len(core.DEVELOPMENT_SESSIONS), 7)
        self.assertNotIn("Mihili_CO_20140306.mat", core.DEVELOPMENT_SESSIONS)
        for animal in nomination.TARGET_SESSIONS:
            selected = outer(panels(animal), animal)
            session = selected.left_out_session
            keys = tuple(key for key in core.DEVELOPMENT_SESSIONS if key != session)
            result = core.predict_outer_support(nomination_result=selected,
                target=support(session), references=references(keys))
            self.assertEqual(result.reference_sessions, keys)
            self.assertEqual(len(result.reference_sessions), 6)
            self.assertNotIn(session, result.reference_sessions)
            counts = tuple(sum(key.startswith(prefix) for key in result.reference_sessions)
                           for prefix in ("Mihili_", "Chewie_"))
            self.assertEqual(counts, (2, 4) if animal == "Mihili" else (3, 3))
            # The two/four reference mix still uses the pooled session median.
            self.assertEqual((result.reliability_median, result.distance_median), (0., 4.))
        frozen = agreement()
        for animal, sessions in core.RESERVED_SESSIONS.items():
            for session in sessions:
                result = core.predict_reserved_support(agreement=frozen, target_session=session,
                    target=support(session), references=references(core.DEVELOPMENT_SESSIONS))
                self.assertEqual(result.reference_sessions, core.DEVELOPMENT_SESSIONS)
                self.assertEqual(result.target_animal, animal)
                self.assertEqual(result.reliability_scale, core.MAD_FACTOR * 2.)
                self.assertEqual(result.distance_scale, core.MAD_FACTOR * 2.)

    def test_missing_invalid_and_zero_scale_support_abstains_without_imputation(self):
        selected = outer(panels())
        session = selected.left_out_session
        keys = tuple(key for key in core.DEVELOPMENT_SESSIONS if key != session)
        refs = references(keys)
        missing = core.predict_outer_support(nomination_result=selected, target=object(),
            references=UnopenedRecords({key: None for key in keys[:-1]}))
        self.assertEqual(missing.status, "abstained")
        self.assertEqual(missing.missing_reference_sessions, (keys[-1],))
        cases = [
            (None, refs),
            (support(session), {**refs, keys[0]: None}),
            (replace(support(session), reliability=float("nan")), refs),
            (replace(support(session), profile_distance=float("inf")), refs),
            (replace(support(session), profile_distance=-.1), refs),
            (replace(support(session), bandwise_distance_scale_valid=False), refs),
            (support(session), {**refs, keys[0]: replace(refs[keys[0]], reliability=None)}),
            (support(session), {**refs, keys[0]: replace(refs[keys[0]], bandwise_distance_scale_valid=None)}),
            (support(session), {key: replace(value, reliability=1.) for key, value in refs.items()}),
            (support(session), {key: replace(value, profile_distance=1.) for key, value in refs.items()}),
            # Finite inputs with nonfinite derived scale must also abstain.
            (support(session), {key: replace(value, reliability=(-1.5e308 if i < 3 else 1.5e308))
                               for i, (key, value) in enumerate(refs.items())}),
        ]
        for target, reference in cases:
            with self.subTest(target=target):
                result = core.predict_outer_support(nomination_result=selected,
                    target=target, references=reference)
                self.assertEqual(result.status, "abstained")
                self.assertIsNone(result.prediction)
                self.assertIsNone(result.support_index)
                self.assertTrue(result.reasons)

    def test_no_nominee_never_reads_support_inputs_or_promotes_final_agreement(self):
        tied = {"C": (.02, .02), "D": (.01, .01), "E": (-.02, -.02)}
        selected = outer(panels(gains=tied))
        result = core.predict_outer_support(nomination_result=selected,
            target=object(), references=object())
        self.assertEqual(result.status, "no_nominee")
        self.assertIsNone(result.support_index)
        frozen = nomination.compare_final_nominations(
            mihili_to_chewie=final(panels(gains=tied)),
            chewie_to_mihili=final(panels("Mihili"), "Mihili"))
        result = core.predict_reserved_support(agreement=frozen,
            target_session=core.RESERVED_SESSIONS["Mihili"][0], target=object(), references=object())
        self.assertEqual(result.status, "no_nominee")
        self.assertIsNone(result.prediction)
        self.assertFalse(result.establishes_component_claim)
        first = nomination.TARGET_SESSIONS["Mihili"][0]
        blocked_finals = (
            final(panels("Mihili", {"C": (.01, .01), "D": (.1, .1), "E": (-.1, -.1)}), "Mihili"),
            final(panels("Mihili"), "Mihili",
                  assessment=outer_assessment("Mihili", choices=("C", "C", "D"))),
            final(panels("Mihili"), "Mihili",
                  assessment=outer_assessment("Mihili", missing=(first,))),
        )
        self.assertEqual(tuple(value.status for value in blocked_finals),
            ("outer_family_mismatch", "unstable_outer_selection", "unresolved_outer_stability"))
        for value in blocked_finals:
            frozen = nomination.compare_final_nominations(mihili_to_chewie=final(panels()),
                                                          chewie_to_mihili=value)
            result = core.predict_reserved_support(agreement=frozen,
                target_session=core.RESERVED_SESSIONS["Mihili"][0],
                target=object(), references=object())
            self.assertEqual(result.status, "no_nominee")

    def test_forbidden_keys_and_wrong_binding_are_rejected_before_scalar_access(self):
        selected = outer(panels())
        session = selected.left_out_session
        keys = tuple(key for key in core.DEVELOPMENT_SESSIONS if key != session)
        unread = {key: UnreadScalars(key, "C", "calibration", 0., 4., True) for key in keys}
        for extra in (session, core.RESERVED_SESSIONS["Chewie-L"][0],
                      "Mihili_CO_20140306.mat", "unknown_session"):
            with self.assertRaises(core.SupportPredictionError):
                core.predict_outer_support(nomination_result=selected, target=object(),
                    references=UnopenedRecords({**unread, extra: None}))
        for wrong in (replace(support(keys[-1]), role="evaluation"),
                      replace(support(keys[-1]), component="D"),
                      replace(support(keys[-1]), session_id=keys[0])):
            with self.assertRaises(core.SupportPredictionError):
                core.predict_outer_support(nomination_result=selected,
                    target=support(session), references={**unread, keys[-1]: wrong})
        with self.assertRaises(core.SupportPredictionError):
            core.predict_outer_support(nomination_result=selected,
                target=replace(support(session), role="training"), references=unread)
        with self.assertRaises(core.SupportPredictionError):
            core.predict_outer_support(nomination_result=final(panels()),
                target=object(), references=object())
        with self.assertRaises(core.SupportPredictionError):
            core.predict_reserved_support(agreement=replace(agreement(), component="E"),
                target_session=core.RESERVED_SESSIONS["Mihili"][0], target=object(), references=object())
        frozen = agreement()
        value = frozen.chewie_to_mihili
        assessment = value.outer_assessment
        # Keep superficial direction/family identities unchanged while
        # changing the retained evidence. Every case must fail before H input
        # inspection; a fabricated same-family agreement grants no prediction.
        for forged_final in (replace(value, outer_assessment=None),
                replace(value, outer_assessment=replace(assessment, stable_component="D")),
                replace(value, outer_assessment=replace(assessment, folds=assessment.folds[:-1])),
                replace(value, records=value.records[:-1]),
                replace(value, nominee="D")):
            with self.assertRaises(core.SupportPredictionError):
                core.predict_reserved_support(
                    agreement=replace(frozen, chewie_to_mihili=forged_final),
                    target_session=core.RESERVED_SESSIONS["Mihili"][0],
                    target=object(), references=object())


if __name__ == "__main__":
    unittest.main()
