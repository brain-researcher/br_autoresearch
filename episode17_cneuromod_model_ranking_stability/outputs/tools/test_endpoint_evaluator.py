"""Focused endpoint-composition checks; fabricated sums, no study payloads."""
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np

import endpoint_evaluator as core


def fixture(phase="development"):
    nblocks, nconcepts = (24, 20) if phase == "development" else (12, 10)
    blocks = []
    for block in range(nblocks):
        # Different exemplar counts per block distinguish pooling from a mean
        # of block R² and prevent accidental second multiplication by counts.
        concepts = tuple(f"c{block:02}_{i:02}" for i in range(nconcepts)
                         for _ in range(1 + block % 3))
        images = tuple(f"i{block:02}_{i:03}" for i in range(len(concepts)))
        blocks.append(core.ConceptBlock(f"block-{block:02}", images, concepts))
    blocks = tuple(blocks)
    columns = np.arange(512, dtype=int) * 2 + 3
    bins = np.repeat(np.arange(1, 9), 64)
    reliable = (np.arange(512) % 64 < 32).astype(float) / 256.
    supports, errors, truths = {}, {}, {}
    for person_index, person in enumerate(core.SUBJECTS):
        for roi_index, roi in enumerate(core.ROIS):
            group = (person, roi)
            inclusion_counts = np.where(np.arange(512) % 64 < 32, 20, 12)
            denominators = np.where(np.arange(256) % 2 == 0, .1, .2)
            support = core.FrozenSupport(columns.copy(), bins.copy(), inclusion_counts / 8192.,
                reliable.copy(), columns[::2].copy(), denominators,
                {name: core.GainStrata(columns[:256].copy(), columns[256:].copy())
                 for name in core.H_COMPONENTS})
            supports[group] = support
            a, b, sst, truth = {}, {}, {}, {}
            for stage_index, stage in enumerate(core.STAGES):
                # Both models have negative raw R². Their retained difference
                # and the normalized endpoint remain perfectly well defined.
                image_count = np.array([len(item.image_ids) for item in blocks])[:, None]
                denominator = image_count * (1 + np.arange(nblocks)[:, None]) * (1 + np.arange(512)[None, :] / 1024.)
                advantage = (.12 * (stage_index + 1) + .03 * person_index + .02 * roi_index
                             + .001 * (stage_index + 1) * np.arange(nblocks)[:, None]
                             + .04 * (stage_index + 1) * np.arange(512)[None, :] / 512.)
                a[stage], b[stage] = (2. - advantage) * denominator, 2. * denominator
                sst[stage], truth[stage] = denominator, advantage
            errors[group] = core.PairedBlockErrors(("DINO_R50", "SUP_R50_V1"), columns.copy(), blocks, a, b, sst)
            truths[group] = truth
    return supports, errors, truths


def direct_reference(supports, errors, truths, people, blocks):
    """Independent known-advantage calculation, with named endpoint formulae."""
    rows = np.empty((4, 9))
    for index, person in enumerate(core.SUBJECTS):
        rois = []
        for roi in core.ROIS:
            group = (person, roi)
            d = {}
            for stage in core.STAGES:
                denominator = errors[group].sst[stage]
                d[stage] = np.sum(truths[group][stage] * denominator * blocks[:, None], axis=0) / np.sum(denominator * blocks[:, None], axis=0)
            bc, cd = d["C"] - d["B"], d["D"] - d["C"]
            lower = np.arange(512) % 64 < 32
            rois.append([d["B"].mean(), d["C"].mean(), d["D"].mean(),
                         .625 * d["D"][lower].mean() + .375 * d["D"][~lower].mean(), d["D"][lower].mean(),
                         d["D"][::2].mean(), (10 * d["D"][::4].mean() + 5 * d["D"][2::4].mean()) / 2,
                         bc[256:].mean() - bc[:256].mean(), cd[256:].mean() - cd[:256].mean()])
        rows[index] = np.mean(rois, axis=0)
    return (people / people.sum()) @ rows, rows


class EndpointCompositionTests(unittest.TestCase):
    def test_complete_family_pools_image_sums_and_retains_equal_group_weights(self):
        supports, errors, truths = fixture()
        family = core.EndpointFamily(supports)
        statistic = family.bind("development", errors, pair_id="P1")
        people = np.array([2, 0, 1, 1])
        blocks = np.array([1, 3] + [0] * 22)
        expected, rows = direct_reference(supports, errors, truths, people, blocks)
        np.testing.assert_allclose(statistic(people, blocks), expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(statistic.participant_effects(blocks), rows, rtol=1e-13, atol=1e-13)
        self.assertEqual(statistic.component_ids, core.PRIMARY + core.H_COMPONENTS)
        self.assertEqual(family.directed_edges, core.EDGES)
        self.assertGreater(expected[6], 1.)
        self.assertGreater(abs(expected[3] - expected[2]), 1e-4)
        self.assertGreater(abs(expected[6] - expected[5] / .15), .1)
        # An equal-weight mean of block ratios is the wrong estimand here.
        wrong = np.mean(truths[core.GROUPS[0]]["B"][:2], axis=(0, 1))
        self.assertGreater(abs(rows[0, 0] - wrong), .005)
        leave_person = np.array([1, 1, 1, 0])
        leave_block = np.ones(24, dtype=int)
        leave_block[1] = 0
        wanted, _ = direct_reference(supports, errors, truths, leave_person, leave_block)
        np.testing.assert_allclose(statistic(leave_person, leave_block), wanted, atol=1e-13)

    def test_beta_companion_is_equal_block_mean_not_pooled_primary_change(self):
        supports, errors, truths = fixture("audit")
        statistic = core.EndpointFamily(supports).bind("audit", errors, pair_id="P1")
        people, blocks = np.ones(4, dtype=int), np.ones(12, dtype=int)
        expected = np.mean([np.mean(truths[group]["C"] - truths[group]["B"])
                            for group in core.GROUPS])
        self.assertAlmostEqual(statistic.beta_block_mean("B_to_C", people, blocks), expected, places=13)
        pooled = statistic(people, blocks)
        self.assertGreater(abs(expected - (pooled[1] - pooled[0])), .001)
        np.testing.assert_allclose(statistic.group_beta_block_changes(core.GROUPS[0], "B_to_C"),
                                   truths[core.GROUPS[0]]["C"] - truths[core.GROUPS[0]]["B"], atol=1e-14)

    def test_calibration_inapplicability_freezes_family_before_error_binding(self):
        supports, errors, _ = fixture()
        key = core.GROUPS[0]
        strata = dict(supports[key].beta_strata)
        strata["H_BC"] = core.InapplicableGain("undefined_calibration_gain")
        supports[key] = replace(supports[key], V_NC=np.array([], dtype=int),
                                V_NC_ceiling_denominator=np.array([], dtype=float), beta_strata=strata)
        family = core.EndpointFamily(supports)
        self.assertEqual(family.primary_endpoint_ids, core.PRIMARY[:5])
        self.assertEqual(family.h_component_ids, ("H_CD",))
        self.assertIn(("B_Aall_raw", "C_Aall_raw"), family.directed_edges)
        self.assertEqual(family.calibration_inapplicability["ceiling_empty_groups"], (key,))
        statistic = family.bind("development", errors, pair_id="P1")
        self.assertEqual(statistic(np.ones(4, dtype=int), np.ones(24, dtype=int)).shape, (6,))
        with self.assertRaisesRegex(core.EndpointError, "normalized_family_inapplicable"):
            statistic.crossed_bounds("F_raw_normalized")
        with self.assertRaisesRegex(core.EndpointError, "beta_explanation_inapplicable"):
            statistic.group_beta_block_changes(key, "B_to_C")
        strata["H_BC"] = None
        with self.assertRaisesRegex(core.EndpointError, "gain_disposition_unresolved"):
            core.EndpointFamily(supports)

    def test_bound_arrays_and_family_cannot_change_through_original_inputs(self):
        supports, errors, _ = fixture()
        family = core.EndpointFamily(supports)
        statistic = family.bind("development", errors, pair_id="P1")
        people, blocks = np.ones(4, dtype=int), np.ones(24, dtype=int)
        before = statistic(people, blocks)
        supports[core.GROUPS[0]].A_N_weights[:] = 0
        supports[core.GROUPS[0]].beta_strata.clear()
        errors[core.GROUPS[0]].sse_a["D"][:] = 0
        np.testing.assert_array_equal(statistic(people, blocks), before)
        with self.assertRaises(FrozenInstanceError):
            family.component_ids = ()
        with self.assertRaises(FrozenInstanceError):
            statistic.family = None
        with self.assertRaises(ValueError):
            statistic.participant_effects(blocks).setflags(write=True)

    def test_misaligned_or_invalid_inputs_refuse_without_subset_rescue(self):
        supports, errors, _ = fixture()
        family, key = core.EndpointFamily(supports), core.GROUPS[0]
        with self.assertRaisesRegex(core.EndpointError, "eight_participant_ROIs_required"):
            family.bind("development", {group: item for group, item in errors.items() if group != key}, pair_id="P1")
        altered = dict(errors)
        altered[key] = replace(errors[key], model_order=("SUP_R50_V1", "DINO_R50"))
        with self.assertRaisesRegex(core.EndpointError, "fixed_pair_sign_or_model_identity_changed"):
            family.bind("development", altered, pair_id="P1")
        with self.assertRaisesRegex(core.EndpointError, "fixed_pair_sign_or_model_identity_changed"):
            family.bind("development", errors, pair_id="P2")
        with self.assertRaisesRegex(core.EndpointError, "registered_pair_required"):
            family.bind("development", errors, pair_id="new_pair")
        altered[key] = replace(errors[key], column_ids=errors[key].column_ids[::-1])
        with self.assertRaisesRegex(core.EndpointError, "column_identity_changed"):
            family.bind("development", altered, pair_id="P1")
        changed_blocks = list(errors[key].blocks)
        changed_blocks[0] = replace(changed_blocks[0], image_ids=("different-image",) + changed_blocks[0].image_ids[1:])
        altered[key] = replace(errors[key], blocks=tuple(changed_blocks))
        with self.assertRaisesRegex(core.EndpointError, "alignment_changed"):
            family.bind("development", altered, pair_id="P1")
        changed_blocks[1] = replace(changed_blocks[1], concept_ids=(changed_blocks[0].concept_ids[0],) + changed_blocks[1].concept_ids[1:])
        altered[key] = replace(errors[key], blocks=tuple(changed_blocks))
        with self.assertRaisesRegex(core.EndpointError, "whole_concept_block_assignment_changed"):
            family.bind("development", altered, pair_id="P1")
        for value, pattern in ((0., "sst_invalid"), (np.nan, "array_nonfinite")):
            sst = {stage: array.copy() for stage, array in errors[key].sst.items()}
            sst["B"][0, 0] = value
            altered[key] = replace(errors[key], sst=sst)
            with self.assertRaisesRegex(core.EndpointError, pattern):
                family.bind("development", altered, pair_id="P1")
        altered[key] = replace(errors[key], sse_b={"B": errors[key].sse_b["B"]})
        with self.assertRaisesRegex(core.EndpointError, "complete_B_C_D_required"):
            family.bind("development", altered, pair_id="P1")
        statistic = family.bind("development", errors, pair_id="P1")
        for people, blocks in (([1., 1., 1., 1.], np.ones(24, dtype=int)),
                               ([0, 0, 0, 0], np.ones(24, dtype=int)),
                               ([1, 1, 1, 1], np.array([1] * 23 + [-1]))):
            with self.assertRaises(core.EndpointError):
                statistic(people, blocks)

    def test_crossed_engine_receives_complete_frozen_families_and_callback(self):
        supports, errors, truths = fixture("audit")
        statistic = core.EndpointFamily(supports).bind("audit", errors, pair_id="P1")
        people, blocks = np.array([1, 2, 0, 1]), np.array([0, 2] + [1] * 10)
        expected, _ = direct_reference(supports, errors, truths, people, blocks)
        calls = []

        def inference(callback, ids, scales, phase, *, primary_endpoint_ids):
            calls.append((ids, scales, phase, primary_endpoint_ids))
            wanted = expected if "D_VNC_normalized" in ids else np.delete(expected, 6)
            np.testing.assert_allclose(callback(people, blocks), wanted, atol=1e-13)
            return "delegated-to-existing-inference"

        # Reuse the already qualified 10,000-draw engine. This checks only the
        # newly composed callback/family boundary, without another bootstrap suite.
        with patch.object(core.uncertainty, "crossed_bounds", side_effect=inference):
            self.assertEqual(statistic.crossed_bounds(), "delegated-to-existing-inference")
            statistic.crossed_bounds("F_raw")
        self.assertEqual(calls[0][0], core.PRIMARY + core.H_COMPONENTS)
        self.assertEqual(calls[0][3], core.PRIMARY)
        self.assertEqual(calls[0][1][6], "ceiling_normalized_R2")
        self.assertEqual(calls[0][2], "audit")
        self.assertEqual(calls[1][3], core.PRIMARY[:-1])
        self.assertNotIn("H_BC", calls[1][3])
        with self.assertRaisesRegex(core.EndpointError, "inference_family_invalid"):
            statistic.crossed_bounds("selected_best_endpoints")


if __name__ == "__main__":
    unittest.main()
