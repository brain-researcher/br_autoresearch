"""Known-answer M2 alignment arithmetic; no fit or empirical qualification."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from development_lopo import (
    DEVELOPMENT_IDS, M2_VARIANTS, FoldContext, FoldPrediction, LOPORun,
    M2Prediction, MatchedWarpPrediction,
)
from m2_alignment_signature import m2_alignment_points
from registered_endpoint_metrics import DOMAINS


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError("forbidden map conversion")


class M2AlignmentSignatureTests(unittest.TestCase):
    def fixture(self, deleted_id=None):
        retained = tuple(person for person in DEVELOPMENT_IDS if person != deleted_id)
        u = np.tile([-1., 1.], 104)
        pattern = np.arange(1., 19.)[:, None]*u/18
        folds = []
        for index, person in enumerate(retained):
            context = FoldContext(person, tuple(value for value in retained if value != person), deleted_id)
            G = (index+1)*pattern+np.arange(18.)[:, None]
            gain = 1+index/10
            # Distinct genuine half signals expose wrong 1x3 / 2x4 averaging.
            parts = np.stack([G+factor*(index-4)*pattern+quarter*10
                              for quarter, factor in enumerate((1., 3., 2., 6.))])
            m2 = tuple(M2Prediction(name, gain*(G+(variant+1)*pattern), np.eye(18), {})
                       for variant, name in enumerate(M2_VARIANTS))
            controls = tuple(MatchedWarpPrediction(name, draw,
                gain*(G+(variant+1)*(draw-2)*pattern/5), np.eye(18), {})
                for variant, name in enumerate(M2_VARIANTS) for draw in range(8))
            folds.append(FoldPrediction(context, G, np.eye(18), 1., 1., G, np.eye(18),
                                        np.eye(2), gain, parts,
                                        m2_predictions=m2, matched_warp_predictions=controls))
        return LOPORun(retained, deleted_id, tuple(folds)), pattern

    def test_signed_domain_weighted_two_half_equal_person_known_answer(self):
        run, pattern = self.fixture()
        result = m2_alignment_points(run)
        energy = 0.
        offset = 0
        for domain in DOMAINS:
            energy += sum(np.mean(pattern[row]**2) for row in range(offset, offset+len(domain)))/(7*len(domain))
            offset += len(domain)
        self.assertNotAlmostEqual(energy, np.mean(pattern**2))
        for variant, name in enumerate(M2_VARIANTS):
            point = result["variants"][name]
            expected = np.asarray([(variant+1)*(index-4)*3*energy*.7
                                   for index in range(12)])
            # Observed-half factors average to3; draw factors average to.3.
            np.testing.assert_allclose(point["participant_alignment_advantage"], expected, atol=1e-12)
            np.testing.assert_allclose(point["advantage_by_participant_and_half"],
                expected[:, None]*np.asarray([2., 4.])[None]/3, atol=1e-12)
            self.assertAlmostEqual(point["alignment_advantage"], expected.mean(), places=12)
            self.assertLess(point["participant_alignment_advantage"][0], 0)
            self.assertEqual(point["control_alignment_by_participant_draw_and_half"].shape, (12, 8, 2))
        self.assertEqual(result["biological_n"], 12)
        self.assertFalse(result["native_signature_certified"])
        self.assertFalse(result["half_field_stability_certified"])
        self.assertFalse(result["permutation_inference"])

    def test_constant_offsets_and_common_prediction_baseline_cancel(self):
        run, _ = self.fixture()
        first = m2_alignment_points(run)
        altered = []
        for fold in run.folds:
            # Add the same spatially varying pattern to parent and all nulls.
            # This changes absolute alignment but cannot change the advantage.
            common = (fold.gain-1)*fold.G
            m2 = tuple(replace(value, maps=value.maps+common+123.) for value in fold.m2_predictions)
            controls = tuple(replace(value, maps=value.maps+common-37.) for value in fold.matched_warp_predictions)
            altered.append(replace(fold, m2_predictions=m2, matched_warp_predictions=controls,
                                   observed_partitions=fold.observed_partitions+91.))
        second = m2_alignment_points(replace(run, folds=tuple(altered)))
        for name in M2_VARIANTS:
            np.testing.assert_allclose(first["variants"][name]["participant_alignment_advantage"],
                                       second["variants"][name]["participant_alignment_advantage"], atol=1e-11)
        self.assertNotAlmostEqual(first["variants"][M2_VARIANTS[0]]["mean_fitted_alignment"],
                                  second["variants"][M2_VARIANTS[0]]["mean_fitted_alignment"])
        # Gain is removed before alignment: varying the common positive gain
        # changes neither the primary statistic nor each person's weight.
        rescaled = []
        for index, fold in enumerate(run.folds):
            factor = 2+index/3
            rescaled.append(replace(fold, gain=factor*fold.gain,
                m2_predictions=tuple(replace(value, maps=factor*value.maps) for value in fold.m2_predictions),
                matched_warp_predictions=tuple(replace(value, maps=factor*value.maps)
                                               for value in fold.matched_warp_predictions)))
        third = m2_alignment_points(replace(run, folds=tuple(rescaled)))
        for name in M2_VARIANTS:
            np.testing.assert_allclose(first["variants"][name]["participant_alignment_advantage"],
                                       third["variants"][name]["participant_alignment_advantage"], atol=1e-12)

    def test_complete_nested_identities_and_missing_draws_refuse_before_maps(self):
        run, _ = self.fixture(DEVELOPMENT_IDS[2])
        result = m2_alignment_points(run)
        self.assertEqual(result["biological_n"], 11)
        self.assertNotIn(DEVELOPMENT_IDS[2], result["participant_ids"])
        poisoned = tuple(replace(fold, G=Poison(), observed_partitions=Poison()) for fold in run.folds)
        final = poisoned[-1]
        invalid = poisoned[:-1]+(replace(final, matched_warp_predictions=final.matched_warp_predictions[:-1]),)
        with self.assertRaisesRegex(ValueError, "all eight"):
            m2_alignment_points(replace(run, folds=invalid))
        invalid = poisoned[:-1]+(replace(final, context=replace(final.context, training_ids=DEVELOPMENT_IDS)),)
        with self.assertRaisesRegex(ValueError, "exclusions"):
            m2_alignment_points(replace(run, folds=invalid))
        with self.assertRaisesRegex(ValueError, "ordered development folds"):
            m2_alignment_points(replace(run, folds=run.folds[:-1]))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            m2_alignment_points(replace(run, folds=(replace(run.folds[0], gain=0.),)+run.folds[1:]))
        bad = run.folds[0].m2_predictions[0]
        bad = replace(bad, maps=np.full((18, 208), np.nan))
        fold = replace(run.folds[0], m2_predictions=(bad,)+run.folds[0].m2_predictions[1:])
        with self.assertRaisesRegex(ValueError, "finite"):
            m2_alignment_points(replace(run, folds=(fold,)+run.folds[1:]))


if __name__ == "__main__":
    unittest.main()
