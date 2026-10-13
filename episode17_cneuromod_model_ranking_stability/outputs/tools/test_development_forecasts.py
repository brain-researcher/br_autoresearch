"""Focused forecast-composition checks on fabricated sums, never study arrays.

The finite nine-column OLS calls exercise the already adopted numerical core;
no candidate models, extra forecast criteria or empirical claims are added.
Run only in the coordinator's actual Slurm allocation with one numerical thread.
"""
from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np

import development_forecasts as core
import endpoint_evaluator as endpoint
import measurement_core as measurement


def fixture():
    # Deliberately nonlexical metadata order: first12/last12 cannot be inferred
    # by sorting labels. Image counts vary independently of equal-block targets.
    blocks = tuple(endpoint.ConceptBlock(f"metadata-{(7 * block) % 24:02}",
        tuple(f"image-{block}-{item}" for item in range(20 * (1 + block % 3))),
        tuple(f"concept-{block}-{concept}" for concept in range(20)
              for _ in range(1 + block % 3))) for block in range(24))
    supports, errors, designs, truth = {}, {}, {}, {}
    for person_index, person in enumerate(endpoint.SUBJECTS):
        for roi_index, roi in enumerate(endpoint.ROIS):
            group = person, roi
            width = 512 + 64 * person_index + 64 * roi_index
            bin_width = width // 8
            columns = 3 + 2 * np.arange(width, dtype=int)
            bins = np.repeat(np.arange(1, 9), bin_width)
            gain = np.linspace(-1., 1., width)
            design = measurement.beta_gain_design(gain, bins, columns)
            designs[group] = core.ColumnBoundGainDesign(columns.copy(), design)
            counts = np.tile(np.full(bin_width, 1024 // bin_width), 8)
            for bin_index in range(8):
                start = bin_index * bin_width
                counts[start:start + 1024 % bin_width] += 1
            reliable = (np.arange(width) % bin_width < 32).astype(float) / 256.
            denominator = np.where(np.arange(width // 2) % 2 == 0, .1, .2)
            supports[group] = endpoint.FrozenSupport(columns, bins, counts / 8192., reliable,
                columns[::2], denominator,
                {name: endpoint.GainStrata(columns[design.low_indices], columns[design.high_indices])
                 for name in endpoint.H_COMPONENTS})
            block = np.arange(1, 25, dtype=float)[:, None]
            voxel = np.arange(width, dtype=float)[None, :] / width
            baseline = np.broadcast_to(.12 + .01 * person_index + .02 * roi_index + .07 * voxel, (24, width))
            # Opposing left/right errors distinguish ROI-before-absolute-error
            # from the incorrect mean of per-ROI absolute errors.
            bc = (.020 if roi_index == 0 else -.015) * block + .01 * gain[None, :] + .003 * person_index
            cd = (.012 if roi_index == 0 else -.007) * block + .02 * gain[None, :] + .002 * person_index
            d = {"B": baseline, "C": baseline + bc, "D": baseline + bc + cd}
            image_counts = np.array([len(item.image_ids) for item in blocks])[:, None]
            sst = image_counts * block * (1. + voxel)
            errors[group] = endpoint.PairedBlockErrors(("DINO_R50", "SUP_R50_V1"), columns, blocks,
                {stage: (2. - d[stage]) * sst for stage in endpoint.STAGES},
                {stage: 2. * sst for stage in endpoint.STAGES},
                {stage: sst.copy() for stage in endpoint.STAGES})
            truth[group] = d
    family = endpoint.EndpointFamily(supports)
    return family.bind("development", errors, pair_id="P1"), supports, errors, designs, truth


def build(statistic, edge, designs=None, resource=lambda event: None, order=None):
    return core.build_development_forecast(statistic, edge,
        reference_block_order=tuple(block.block_id for block in statistic.blocks) if order is None else order,
        gain_designs=designs, resource_check=resource)


class DevelopmentForecastTests(unittest.TestCase):
    def test_fixed_weight_forecasts_use_pooled_D_then_equal_roi_and_person_means(self):
        statistic, supports, errors, _, truth = fixture()
        for edge, limit in (("A_N_to_R_N", .02), ("raw_to_ceiling_normalized", .05)):
            expected = {}
            for group in endpoint.GROUPS:
                support, denominator = supports[group], errors[group].sst["D"]
                advantage = np.sum(truth[group]["D"] * denominator, axis=0) / np.sum(denominator, axis=0)
                if edge == "A_N_to_R_N":
                    expected[group] = (support.R_N_weights - support.A_N_weights) @ advantage
                else:
                    selected = advantage[::2]
                    expected[group] = np.mean(selected / support.V_NC_ceiling_denominator) - np.mean(selected)
            result = build(statistic, edge)
            per_person = np.array([np.mean([expected[person, roi] for roi in endpoint.ROIS])
                                   for person in endpoint.SUBJECTS])
            self.assertTrue(result.eligible)
            self.assertEqual(result.target, "aggregate_edge_change")
            self.assertEqual(result.absolute_error_limit, limit)
            self.assertAlmostEqual(result.value, np.mean(per_person), places=13)
            np.testing.assert_allclose(result.participant_forecasts, per_person, atol=1e-13)
            self.assertIs(result.endpoint_family, statistic.family)
            self.assertEqual(result.pair_id, "P1")
            self.assertFalse(result.group_fits)
            self.assertIsNone(result.crossfit_absolute_errors)
            with self.assertRaises(ValueError):
                result.participant_forecasts.setflags(write=True)
        # This estimand remains the mean of voxel ratios, not a ratio of means.
        group = endpoint.GROUPS[0]
        denominator = errors[group].sst["D"]
        d = np.sum(truth[group]["D"] * denominator, axis=0) / np.sum(denominator, axis=0)
        wrong = np.mean(d[::2]) / np.mean(supports[group].V_NC_ceiling_denominator) - np.mean(d[::2])
        self.assertGreater(abs(expected[group] - wrong), .05)

    def test_beta_crossfit_uses_fixed_halves_and_roi_before_absolute_error(self):
        statistic, _, _, designs, truth = fixture()
        for edge, before, after in (("B_to_C", "B", "C"), ("C_to_D", "C", "D")):
            result = build(statistic, edge, designs)
            observed_groups, predicted_groups = {}, {}
            for group in endpoint.GROUPS:
                observed_groups[group] = (truth[group][after] - truth[group][before]).mean(axis=1)
                predicted_groups[group] = np.r_[np.repeat(observed_groups[group][12:].mean(), 12),
                                                np.repeat(observed_groups[group][:12].mean(), 12)]
            observed = np.array([np.mean([observed_groups[person, roi] for roi in endpoint.ROIS], axis=0)
                                 for person in endpoint.SUBJECTS])
            predicted = np.array([np.mean([predicted_groups[person, roi] for roi in endpoint.ROIS], axis=0)
                                  for person in endpoint.SUBJECTS])
            expected_errors = np.abs(observed - predicted)
            np.testing.assert_allclose(result.crossfit_observed, observed, atol=1e-13)
            np.testing.assert_allclose(result.crossfit_predicted, predicted, atol=1e-13)
            np.testing.assert_allclose(result.crossfit_absolute_errors, expected_errors, atol=1e-13)
            wrong = np.array([np.mean([np.abs(observed_groups[person, roi] - predicted_groups[person, roi])
                                      for roi in endpoint.ROIS], axis=0) for person in endpoint.SUBJECTS])
            self.assertGreater(np.max(np.abs(wrong - expected_errors)), .1)
            self.assertAlmostEqual(result.value, observed.mean(), places=13)
            self.assertEqual(result.absolute_error_limit, .02)
            self.assertEqual(result.target, "equal_block_mean_stage_change")
            self.assertIn("pass_not_adjudicated", result.eligibility_reason)
            self.assertEqual(set(result.group_fits), set(endpoint.GROUPS))
            primary = statistic(np.ones(4, dtype=int), np.ones(24, dtype=int))
            pooled_change = primary[statistic.component_ids.index(after + "_Aall_raw")] - primary[
                statistic.component_ids.index(before + "_Aall_raw")]
            self.assertGreater(abs(result.value - pooled_change), .005)
            for group in endpoint.GROUPS:
                np.testing.assert_array_equal(result.gain_designs[group].column_ids, designs[group].column_ids)
                self.assertAlmostEqual(result.group_fits[group].development_block_mean_forecast,
                                       observed_groups[group].mean(), places=13)
                with self.assertRaises(ValueError):
                    result.group_fits[group].coefficients.setflags(write=True)
            with self.assertRaises(FrozenInstanceError):
                result.value = 0.

    def test_reference_order_role_columns_and_gain_strata_cannot_be_repaired(self):
        statistic, _, errors, designs, _ = fixture()
        order = tuple(block.block_id for block in statistic.blocks)
        with self.assertRaisesRegex(ValueError, "frozen_reference_block_order_changed"):
            build(statistic, "B_to_C", designs, order=tuple(sorted(order)))
        with self.assertRaisesRegex(ValueError, "explicit_reference_block_order_required"):
            core.build_development_forecast(statistic, "B_to_C", reference_block_order=None,
                                            gain_designs=designs, resource_check=lambda event: None)
        audit_errors = {group: replace(entry, blocks=tuple(
            endpoint.ConceptBlock(f"audit-{block}", tuple(f"audit-image-{block}-{i}" for i in range(10)),
                                  tuple(f"audit-concept-{block}-{i}" for i in range(10))) for block in range(12)),
            sse_a={stage: values[:12] for stage, values in entry.sse_a.items()},
            sse_b={stage: values[:12] for stage, values in entry.sse_b.items()},
            sst={stage: values[:12] for stage, values in entry.sst.items()}) for group, entry in errors.items()}
        audit = statistic.family.bind("audit", audit_errors, pair_id="P1")
        with self.assertRaisesRegex(ValueError, "development_statistic_required"):
            build(audit, "A_N_to_R_N")
        group = endpoint.GROUPS[0]
        with self.assertRaisesRegex(ValueError, "eight_calibration_gain_designs_required"):
            build(statistic, "B_to_C", {key: value for key, value in designs.items() if key != group})
        bad_columns = {**designs, group: replace(designs[group], column_ids=designs[group].column_ids[::-1])}
        with self.assertRaisesRegex(ValueError, "gain_design_original_columns_changed"):
            build(statistic, "B_to_C", bad_columns)
        design = designs[group].design
        bad_strata = {**designs, group: replace(designs[group], design=replace(design,
            low_indices=design.high_indices, high_indices=design.low_indices))}
        with self.assertRaisesRegex(ValueError, "frozen_gain_strata_changed"):
            build(statistic, "B_to_C", bad_strata)
        matrix = design.matrix.copy()
        # Preserve all eight one-hot bin counts and the gain/strata, but move
        # two native columns between bins. It is a different frozen operator.
        other_bin = int(np.flatnonzero(matrix[:, 1])[0])
        matrix[[0, other_bin], :8] = matrix[[other_bin, 0], :8]
        bad_bins = {**designs, group: replace(designs[group], design=replace(design, matrix=matrix))}
        with self.assertRaisesRegex(ValueError, "frozen_spatial_bin_assignment_changed"):
            build(statistic, "B_to_C", bad_bins)

    def test_calibration_inapplicability_and_size_control_are_explicit_not_zero(self):
        statistic, supports, errors, _, _ = fixture()
        size = build(statistic, "A_all_to_A_N")
        self.assertFalse(size.eligible)
        self.assertIsNone(size.value)
        self.assertIn("no_adopted", size.eligibility_reason)
        group = endpoint.GROUPS[0]
        support = supports[group]
        changed = {**supports, group: replace(support,
            beta_strata={**support.beta_strata, "H_BC": endpoint.InapplicableGain("gain_design_rank_deficient")},
            V_NC=np.array([], dtype=int), V_NC_ceiling_denominator=np.array([], dtype=float))}
        inapplicable = endpoint.EndpointFamily(changed).bind("development", errors, pair_id="P1")
        with patch.object(measurement, "fit_beta_forecast", new=lambda *args: self.fail("inapplicable fit attempted")):
            for edge, reason in (("B_to_C", "calibration_beta_gain_inapplicable"),
                                 ("raw_to_ceiling_normalized", "calibration_ceiling_support_inapplicable")):
                result = build(inapplicable, edge)
                self.assertFalse(result.eligible)
                self.assertEqual(result.eligibility_reason, reason)
                self.assertIsNone(result.value)
        self.assertIn("B_Aall_raw", inapplicable.primary_endpoint_ids)
        self.assertIn("C_Aall_raw", inapplicable.primary_endpoint_ids)

    def test_resource_admission_precedes_existing_OLS_and_frozen_design_is_a_snapshot(self):
        statistic, _, _, designs, _ = fixture()
        marker = RuntimeError("OLS budget exhausted")
        operations = []

        def deny(event):
            operations.append((event.operation, event.when))
            if event.operation == "fit_beta_forecast" and event.when == "before":
                raise marker

        with patch.object(measurement, "fit_beta_forecast", new=lambda *args: self.fail("OLS preceded admission")), \
             self.assertRaises(RuntimeError) as caught:
            build(statistic, "B_to_C", designs, resource=deny)
        self.assertIs(caught.exception, marker)
        self.assertNotIn(("complete", "after"), operations)
        with self.assertRaisesRegex(ValueError, "external_resource_callback_required"):
            build(statistic, "B_to_C", designs, resource=None)
        result = build(statistic, "B_to_C", designs)
        group = endpoint.GROUPS[0]
        before = result.gain_designs[group].design.matrix.copy()
        designs[group].design.matrix.setflags(write=True)
        designs[group].design.matrix[0, 0] = 999.
        designs[group].column_ids[0] = 999
        np.testing.assert_array_equal(result.gain_designs[group].design.matrix, before)
        self.assertNotEqual(result.gain_designs[group].column_ids[0], 999)
        with self.assertRaises(ValueError):
            result.gain_designs[group].design.matrix.setflags(write=True)

    def test_existing_OLS_rank_failure_is_not_an_inapplicability_or_partial_forecast(self):
        statistic, _, _, designs, _ = fixture()
        group = endpoint.GROUPS[0]
        old = designs[group].design
        # Malformed supplied design: its gain duplicates one bin intercept.
        # The existing numerical core must refuse it, not drop this group.
        matrix = old.matrix.copy()
        matrix[:, -1] = matrix[:, 0]
        bad = {**designs, group: replace(designs[group], design=replace(old,
            matrix=matrix, standardized_gain=matrix[:, -1]))}
        operations = []
        with self.assertRaisesRegex(ValueError, "gain_OLS_invalid"):
            build(statistic, "B_to_C", bad, resource=lambda event: operations.append(event.operation))
        self.assertIn("fit_beta_forecast", operations)
        self.assertNotIn("complete", operations)


if __name__ == "__main__":
    unittest.main()
