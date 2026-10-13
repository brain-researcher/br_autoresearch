"""Two focused paired-profile control checks; coordinated compute only."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bandwise_prediction import BANDS, FeatureBlock
from component_scoring import FrozenDesign
from primary_profile import BlockScore, ProfileScore, paired_profile
import profile_classifier as classifiers
import profile_controls as core


BLOCKS = tuple(FeatureBlock(f"band_{i}", (band,)) for i, band in enumerate(BANDS[:6]))
RECIPE = classifiers.ClassifierRecipe("cosine_prototype")


def check():
    pass


def panels():
    result = []
    for animal, roster in zip(core.ANIMALS, core.ROSTERS):
        panel = []
        for session in roster:
            design = FrozenDesign(session, "evaluation", tuple(range(8)), tuple(range(8)),
                                  tuple(range(8)), (5, 6), ("M1_pc0", "M1_pc1"))
            # Centered opposite shapes with exactly one distinguished band.
            # 3- and4-session shuffled source prototypes can never cancel:
            # uniform coverage of all six peak locations would be required.
            # A common -5 offset keeps all synthetic score/error pairs within
            # the physical S<=1 range without changing either centered shape.
            m1, pmd = (0., -6., -6., -6., -6., -6.), (-10., -4., -4., -4., -4., -4.)
            left = ProfileScore("M1", design, 100., tuple(
                BlockScore(block, 100. * (1 - value), value, None) for block, value in zip(BLOCKS, m1)))
            right = ProfileScore("PMd", replace(design, axis_ids=("PMd_pc0", "PMd_pc1")), 100., tuple(
                BlockScore(block, 100. * (1 - value), value, None) for block, value in zip(BLOCKS, pmd)))
            panel.append(classifiers.PairedSessionProfiles(animal, paired_profile(left, right, check=check)))
        result.append(tuple(panel))
    return tuple(result)


def evaluate(value):
    return core.evaluate_profile_controls(*value, mihili_sessions=core.ROSTERS[0],
        chewie_sessions=core.ROSTERS[1], recipe=RECIPE, check=check)


class ProfileControlsTests(unittest.TestCase):
    def test_all_seven_omissions_refit_both_directions_and_failure_does_not_hide_or_contaminate_other_control(self):
        supplied = panels()
        original_fit = classifiers.fit_profile_classifier
        with patch.object(classifiers, "fit_profile_classifier", wraps=original_fit) as fitted:
            result = evaluate(supplied)
        leaveout = result["leave_one_out"]
        self.assertTrue(leaveout["complete"])
        self.assertEqual((leaveout["fit_attempts"], result["band_permutation"]["fit_attempts"]), (14, 2))
        self.assertEqual(fitted.call_count, 16)
        self.assertTrue(result["signed_agreement_original"]["raw_all_positive"])
        expected_omissions = tuple((animal, session) for animal, roster in zip(core.ANIMALS, core.ROSTERS)
                                   for session in roster)
        self.assertEqual(tuple((row["omitted_animal"], row["omitted_session"]) for row in leaveout["omissions"]),
                         expected_omissions)
        for index, row in enumerate(leaveout["omissions"]):
            self.assertTrue(row["complete"])
            self.assertEqual(row["equal_direction_mean_balanced_accuracy"], 1.)
            for source, direction in enumerate(row["directions"]):
                wanted_source = tuple(session for session in core.ROSTERS[source]
                                      if session != row["omitted_session"])
                wanted_target = tuple(session for session in core.ROSTERS[1 - source]
                                      if session != row["omitted_session"])
                call = fitted.call_args_list[2 * index + source]
                self.assertEqual(call.kwargs["expected_sessions"], wanted_source)
                self.assertEqual(tuple(value.profile.session_id for value in call.args[0]), wanted_source)
                self.assertEqual(tuple(direction["source_sessions"]), wanted_source)
                self.assertEqual(tuple(direction["target_sessions"]), wanted_target)
                self.assertEqual(len(direction["predictions"]), 2 * len(wanted_target))
                self.assertEqual(direction["balanced_accuracy"], 1.)
                self.assertNotIn(row["omitted_session"], direction["source_sessions"] + direction["target_sessions"])

        calls = 0
        def fail_first_only(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise classifiers.ClassificationUnavailable("one undefined omission")
            return original_fit(*args, **kwargs)
        with patch.object(classifiers, "fit_profile_classifier", side_effect=fail_first_only):
            partial = evaluate(supplied)
        self.assertFalse(partial["leave_one_out"]["complete"])
        self.assertEqual(partial["leave_one_out"]["fit_attempts"], 1)
        self.assertEqual(len(partial["leave_one_out"]["omissions"]), 7)
        self.assertTrue(all(row["equal_direction_mean_balanced_accuracy"] is None
                            for row in partial["leave_one_out"]["omissions"]))
        self.assertEqual(partial["leave_one_out"]["unavailable_reason"], "one undefined omission")
        self.assertTrue(partial["band_permutation"]["complete"])
        self.assertEqual(calls, 3)
        bad = replace(supplied[0][0].profile.m1.blocks[0], score=None)
        missing_region = replace(supplied[0][0].profile.m1, blocks=(bad,) + supplied[0][0].profile.m1.blocks[1:])
        missing = replace(supplied[0][0], profile=replace(supplied[0][0].profile, m1=missing_region))
        with patch.object(classifiers, "fit_profile_classifier", side_effect=AssertionError("must validate all support")):
            unavailable = evaluate(((missing,) + supplied[0][1:], supplied[1]))
        self.assertFalse(unavailable["leave_one_out"]["complete"])
        self.assertFalse(unavailable["band_permutation"]["complete"])
        self.assertEqual(unavailable["leave_one_out"]["fit_attempts"], 0)
        self.assertEqual(unavailable["band_permutation"]["fit_attempts"], 0)

    def test_seeded_band_mapping_moves_paired_payloads_preserves_labels_and_matches_peak_overlap_oracle(self):
        supplied = panels()
        permuted, ledger = core.permute_paired_band_labels(*supplied,
            mihili_sessions=core.ROSTERS[0], chewie_sessions=core.ROSTERS[1], check=check)
        repeated, repeated_ledger = core.permute_paired_band_labels(*supplied,
            mihili_sessions=core.ROSTERS[0], chewie_sessions=core.ROSTERS[1], check=check)
        self.assertEqual((permuted, ledger), (repeated, repeated_ledger))
        peaks, row_number = [[], []], 0
        for animal_index, (before_panel, after_panel) in enumerate(zip(supplied, permuted)):
            for position, before, after in zip(core.ORIGINAL_POSITIONS[animal_index], before_panel, after_panel):
                record = ledger[row_number]
                row_number += 1
                self.assertEqual(record["seed_entropy"], [6006202, animal_index, position])
                self.assertEqual(record["session_id"], before.profile.session_id)
                indices = tuple(record["destination_to_source_indices"])
                self.assertEqual(sorted(indices), list(range(6)))
                expected_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([6006202, animal_index, position])))
                self.assertEqual(indices, tuple(expected_rng.permutation(6)))
                self.assertEqual(after.profile.blocks, BLOCKS)
                for previous, current in ((before.profile.m1, after.profile.m1), (before.profile.pmd, after.profile.pmd)):
                    self.assertEqual(current.design, previous.design)
                    self.assertEqual(current.residual_energy, previous.residual_energy)
                    self.assertEqual(tuple(score.block for score in current.blocks), BLOCKS)
                    self.assertEqual(tuple(score.score for score in current.blocks),
                                     tuple(previous.blocks[index].score for index in indices))
                    self.assertEqual(tuple(score.squared_error for score in current.blocks),
                                     tuple(previous.blocks[index].squared_error for index in indices))
                self.assertEqual(after.profile.m1_minus_pmd,
                    tuple(a.score - b.score for a, b in zip(after.profile.m1.blocks, after.profile.pmd.blocks)))
                self.assertEqual(before.profile.m1_minus_pmd, (10., -2., -2., -2., -2., -2.))
                peaks[animal_index].append(indices.index(0))
        # For v=6*one_hot(peak)-1, a source prototype's dot with target v
        # has sign 36*source_peak_count -6*source_session_count. With3/4
        # source sessions, correct transfer means the target peak occurs at
        # least once in the source. This oracle requires no fitted classifier.
        expected = [sum(peak in set(peaks[source]) for peak in peaks[1 - source]) / len(peaks[1 - source])
                    for source in range(2)]
        result = evaluate(supplied)
        band = result["band_permutation"]
        self.assertTrue(band["complete"])
        self.assertEqual([direction["balanced_accuracy"] for direction in band["directions"]], expected)
        self.assertEqual(band["equal_direction_mean_balanced_accuracy"], .5 * sum(expected))
        self.assertEqual(band["interpretation"], "sensitivity_no_automatic_chance_veto")
        self.assertEqual(result["signed_agreement_original"]["mihili_raw_mean"], (10., -2., -2., -2., -2., -2.))
        self.assertIsNotNone(band["signed_agreement_permuted"])
        with self.assertRaises(ValueError):
            core.permute_paired_band_labels(supplied[0][::-1], supplied[1],
                mihili_sessions=core.ROSTERS[0][::-1], chewie_sessions=core.ROSTERS[1], check=check)


if __name__ == "__main__":
    unittest.main()
