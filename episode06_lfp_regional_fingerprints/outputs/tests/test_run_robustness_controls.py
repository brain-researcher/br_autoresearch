"""One omission/source-MAD boundary check; coordinated compute only."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from component_scoring import FrozenDesign
from primary_profile import BlockScore, ProfileScore, paired_profile
from profile_classifier import PairedSessionProfiles
import run_robustness_controls as runner


def check():
    pass


def fixture(animal, session, left, right, indices=(0,)):
    blocks = tuple(runner.initial.BLOCKS[index] for index in indices)
    design = FrozenDesign(session, "evaluation", tuple(range(8)), tuple(range(8)),
                          tuple(range(8)), (5, 6), ("M1_pc0", "M1_pc1"))
    m1 = ProfileScore("M1", design, 100., tuple(BlockScore(block, 100.*(1-value), value, None)
                                              for block, value in zip(blocks, left)))
    pmd = ProfileScore("PMd", replace(design, axis_ids=("PMd_pc0", "PMd_pc1")), 100.,
                       tuple(BlockScore(block, 100.*(1-value), value, None) for block, value in zip(blocks, right)))
    return PairedSessionProfiles(animal, paired_profile(m1, pmd, check=check))


class RobustnessControlTests(unittest.TestCase):
    def test_source_MAD_refits_after_each_whole_session_omission_and_undefined_fold_stays_visible(self):
        amplitudes = ((.1, .3, .5), (.2, .4, .6, .8))
        panels = tuple(tuple(fixture(animal, session, (a,), (-a,)) for session, a in zip(roster, values))
                       for animal, roster, values in zip(runner.references.ANIMALS, runner.initial.ROSTERS, amplitudes))
        result = runner.evaluate_leaveout(panels, kind="magnitude", check=check)
        self.assertTrue(result["complete"])
        self.assertEqual(result["fit_attempts"], 14)
        expected_omissions = [(animal, session) for animal, roster in zip(
            runner.references.ANIMALS, runner.initial.ROSTERS) for session in roster]
        self.assertEqual([(row["omitted_animal"], row["omitted_session"]) for row in result["omissions"]],
                         expected_omissions)
        # Independent medians of the remaining absolute scalar observations:
        # each regional pair contributes +/-a. Target omissions leave the
        # source scale and coefficients identical to every other target omission.
        expected_scales = ((.4, .5), (.3, .5), (.2, .5),
                           (.3, .6), (.3, .6), (.3, .4), (.3, .4))
        for slot, expected in zip(result["omissions"], expected_scales):
            self.assertTrue(slot["complete"])
            for source, (direction, scale) in enumerate(zip(slot["directions"], expected)):
                self.assertEqual(direction["balanced_accuracy"], 1.)
                self.assertAlmostEqual(direction["fit"]["magnitude_center"], 0.)
                self.assertAlmostEqual(direction["fit"]["magnitude_scale"], scale*1.4826)
                self.assertNotIn(slot["omitted_session"], direction["source_sessions"]+direction["target_sessions"])
                self.assertEqual(direction["source_sessions"], [session for session in runner.initial.ROSTERS[source]
                                                               if session != slot["omitted_session"]])
                self.assertEqual(len(direction["predictions"]), 2*len(direction["target_sessions"]))
        for index in range(4, 7):
            self.assertEqual(result["omissions"][index]["directions"][0]["fit"]["coefficients"],
                             result["omissions"][3]["directions"][0]["fit"]["coefficients"])

        # Full six source rows have positive MAD; omitting the last source
        # pair leaves {0,0,.5,0}, whose median absolute deviation is exactly0.
        changed = tuple(fixture("Mihili", session, left, right) for session, left, right in zip(
            runner.initial.ROSTERS[0], ((0.,), (.5,), (.6,)), ((0.,), (0.,), (-.6,))))
        unavailable = runner.evaluate_leaveout((changed, panels[1]), kind="magnitude", check=check)
        self.assertFalse(unavailable["complete"])
        self.assertEqual(unavailable["fit_attempts"], 5)
        self.assertEqual(len(unavailable["omissions"]), 7)
        self.assertTrue(all(slot["complete"] for slot in unavailable["omissions"][:2]))
        self.assertIn("source_magnitude_MAD", unavailable["unavailable_reason"])
        self.assertIsNone(unavailable["omissions"][2]["directions"][0])
        self.assertTrue(all(slot["unavailable_reason"] == "not_attempted_after_unavailable_omission"
                            for slot in unavailable["omissions"][3:]))

        # The independent prototype branch retains the same omissions and
        # perfectly separated opposite five-band shapes; no scalar scale is fitted.
        shapes = tuple(tuple(fixture(animal, session, (.8, -.8, 0., 0., 0.),
            (-.8, .8, 0., 0., 0.), runner.LOW5) for session in roster)
            for animal, roster in zip(runner.references.ANIMALS, runner.initial.ROSTERS))
        prototype = runner.evaluate_leaveout(shapes, kind="prototype", check=check)
        self.assertTrue(prototype["complete"])
        self.assertEqual(prototype["fit_attempts"], 14)
        self.assertTrue(all(direction["balanced_accuracy"] == 1. for slot in prototype["omissions"]
                            for direction in slot["directions"]))
        self.assertTrue(all("magnitude_scale" not in direction["fit"] for slot in prototype["omissions"]
                            for direction in slot["directions"]))


if __name__ == "__main__":
    unittest.main()
