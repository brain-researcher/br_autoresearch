"""Focused original-axis/applicability fixtures; coordinator-owned compute only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import derive_private_operators as method


def fixture():
    """Eight groups with gapped original IDs; every bin leaves two spare voxels."""
    maps, supports, structural, products = {}, {}, {}, []
    local_count = 8 * 34
    for subject in method.measurement.SUBJECTS:
        axis = np.concatenate((2 * np.arange(local_count), 1000 + 2 * np.arange(local_count)))
        position = np.tile(np.arange(local_count), 2)
        base = .01 * (position % 7)
        gain_bc, gain_cd = .02 * (position % 5), .03 * (position % 11)
        for stage, fisher in (("B", base), ("C", base + gain_bc), ("D", base + gain_bc + gain_cd)):
            prefix = f"{subject}_{stage}_"
            maps[prefix + "column_ids"] = axis.copy()
            maps[prefix + "response_eligible"] = np.ones(len(axis), dtype=bool)
            maps[prefix + "reliability_valid"] = np.ones(len(axis), dtype=bool)
            maps[prefix + "fisher_z"] = fisher.copy()
            maps[prefix + "ceiling_valid"] = np.ones(len(axis), dtype=bool)
            maps[prefix + "noise_ceiling"] = np.full(len(axis), .5)
            maps[prefix + "ceiling_denominator"] = np.full(len(axis), .5)
            if stage == "D":
                maps[prefix + "noise_ceiling"][::local_count] = -.25
                maps[prefix + "ceiling_denominator"][::local_count] = .10
                maps[prefix + "ceiling_valid"][3::local_count] = False
                maps[prefix + "noise_ceiling"][3::local_count] = np.nan
                maps[prefix + "ceiling_denominator"][3::local_count] = np.nan
            products.append({"participant": subject, "stage": stage, "column_ids": axis.tolist(),
                             "path": str(method.calibration.COPY_PRIVATE / method.calibration.ROLE / subject / f"{stage}.h5")})
        for index, roi in enumerate(method.measurement.ROIS):
            prefix = f"{subject}_{roi}_"
            ids = axis[index * local_count:(index + 1) * local_count].copy()
            bins = np.asarray([f"BIN{i // 34 + 1:02d}" for i in range(local_count)])
            bank = np.concatenate([ids[b * 34:b * 34 + 32] for b in range(8)])
            reliable = np.concatenate([ids[b * 34 + 2:(b + 1) * 34] for b in range(8)])
            counts = np.where(np.isin(ids, bank), 32, 0).astype(np.int64)
            supports.update({prefix + "column_ids": ids, prefix + "spatial_bins": bins,
                             prefix + "A_all": ids.copy(), prefix + "A_N_banks": np.tile(bank, (32, 1)),
                             prefix + "A_N_inclusion_counts": counts, prefix + "R_N": reliable,
                             prefix + "V_NC": np.delete(ids, 3)})
            structural[f"{subject}_{roi}"] = [[int(ident), str(bin_name), method.support.ROI_LABELS[roi]]
                                             for ident, bin_name in zip(ids, bins)]
    return {"products": products, "binding": {"structural_columns": structural}}, maps, supports


def completed_report():
    return {"analysis_id": "calibration_support", "mode": "full", "status": "completed", "errors": [],
            "calibration_support_complete": True, "derived_arrays_verified": True,
            "private_attempt": str(method.SOURCE_PRIVATE), "full_copy_attempt": str(method.calibration.COPY_PRIVATE),
            "selected_original_columns": 8558, "eligible_image_count": 605, "products_planned": 12,
            "planned_physical_stripes": 3258, "completed_physical_stripes": 3258}


class PrivateOperatorTests(unittest.TestCase):
    def test_original_column_join_and_strata_refuse_permuted_stage_or_wrong_role(self):
        plan, maps, supports = fixture()
        with patch.object(method.calibration.columns.h5py, "File", side_effect=AssertionError("no raw HDF5 access")):
            arrays, decisions = method.derive(plan, maps, supports)
        prefix = "sub-01_left_"
        axis = supports[prefix + "A_all"]
        gain = maps["sub-01_C_fisher_z"][:len(axis)] - maps["sub-01_B_fisher_z"][:len(axis)]
        order = np.lexsort((axis, gain))
        np.testing.assert_array_equal(arrays[prefix + "B_to_C_low_columns"], axis[order[:len(axis) // 2]])
        np.testing.assert_array_equal(arrays[prefix + "B_to_C_high_columns"], axis[order[len(axis) // 2:]])
        self.assertTrue(decisions["H_component_calibration_applicability"]["H_BC"])
        changed = copy.deepcopy(maps)
        changed["sub-01_C_column_ids"] = changed["sub-01_C_column_ids"][::-1]
        with self.assertRaises(method.core.Refusal):
            method.derive(plan, changed, supports)
        changed_plan = copy.deepcopy(plan)
        changed_plan["products"][0]["path"] = changed_plan["products"][0]["path"].replace("support_calibration", "development")
        with self.assertRaisesRegex(method.core.Refusal, "private_stage_axis_or_role_changed"):
            method.derive(changed_plan, maps, supports)

    def test_undefined_or_constant_gain_is_private_inapplicability_without_voxel_or_endpoint_loss(self):
        plan, maps, supports = fixture()
        maps["sub-01_B_reliability_valid"][0] = False
        maps["sub-01_B_fisher_z"][0] = np.nan
        count = len(supports["sub-02_left_A_all"])
        maps["sub-02_C_fisher_z"][:count] = maps["sub-02_B_fisher_z"][:count]
        arrays, decisions = method.derive(plan, maps, supports)
        group = decisions["groups"]["sub-01_left"]
        self.assertEqual(group["beta_explanations"]["B_to_C"]["reason"], "undefined_calibration_gain")
        self.assertFalse(group["beta_explanations"]["B_to_C"]["applicable"])
        self.assertTrue(group["beta_explanations"]["C_to_D"]["applicable"])
        self.assertEqual(decisions["groups"]["sub-02_left"]["beta_explanations"]["B_to_C"]["reason"],
                         "calibration_gain_variance_zero")
        self.assertFalse(decisions["H_component_calibration_applicability"]["H_BC"])
        self.assertTrue(decisions["H_component_calibration_applicability"]["H_CD"])
        for subject, roi in method.GROUP_KEYS:
            prefix = f"{subject}_{roi}_"
            np.testing.assert_array_equal(arrays[prefix + "A_all"], supports[prefix + "A_all"])
            self.assertTrue(all(decisions["groups"][f"{subject}_{roi}"]["endpoint_calibration_prerequisites"].values()))
        self.assertNotIn("sub-01_left_B_to_C_design", arrays)
        self.assertTrue(np.isnan(arrays["sub-01_left_B_to_C_gain"][0]))

    def test_saved_bank_weights_and_negative_ceiling_reaggregate_on_exact_original_axes(self):
        plan, maps, supports = fixture()
        arrays, decisions = method.derive(plan, maps, supports)
        prefix = "sub-01_right_"
        axis = arrays[prefix + "A_all"]
        # Private, source-free arithmetic targets test the axis join; the adapter
        # itself never accepts a model score or calls a forecast fitter.
        values = np.cos(np.arange(len(axis)) * .3)
        lookup = {int(ident): i for i, ident in enumerate(axis)}
        anatomical = np.mean([np.mean([values[lookup[int(ident)]] for ident in bank])
                              for bank in supports[prefix + "A_N_banks"]])
        reliable = np.mean([values[lookup[int(ident)]] for ident in supports[prefix + "R_N"]])
        self.assertAlmostEqual(float(arrays[prefix + "A_N_weights"] @ values), anatomical)
        self.assertAlmostEqual(float(arrays[prefix + "R_N_weights"] @ values), reliable)
        self.assertAlmostEqual(float(arrays[prefix + "A_N_to_R_N_operator"] @ values), reliable - anatomical)
        vnc = arrays[prefix + "V_NC"]
        self.assertIn(axis[0], vnc)  # Negative finite ceiling is retained.
        self.assertNotIn(axis[3], vnc)
        denominator = arrays[prefix + "V_NC_ceiling_denominator"]
        self.assertEqual(denominator[0], .1)
        np.testing.assert_array_equal(arrays[prefix + "V_NC_positive_ceiling_operator"], 1. / denominator)
        self.assertEqual(arrays[prefix + "V_NC_positive_ceiling_operator"][0], 10.)
        self.assertEqual(decisions["groups"]["sub-01_right"]["ceiling_operator"]["positive_g_key"],
                         prefix + "V_NC_positive_ceiling_operator")
        selected = np.asarray([values[lookup[int(ident)]] for ident in vnc])
        self.assertAlmostEqual(float(arrays[prefix + "raw_to_ceiling_normalized_difference_weights"] @ selected),
                               float(np.mean(selected / denominator) - np.mean(selected)))
        self.assertGreater(float(arrays[prefix + "V_NC_normalized_weights"].sum()), 1.)
        changed = copy.deepcopy(supports)
        changed[prefix + "V_NC"] = axis.copy()
        with self.assertRaisesRegex(method.core.Refusal, "frozen_V_NC_changed"):
            method.derive(plan, maps, changed)

    def test_completed_source_required_and_runtime_keeps_applicability_out_of_public_report(self):
        with self.assertRaisesRegex(method.core.Refusal, "completed_private_calibration_attempt001_required"):
            method.validate_source_report({**completed_report(), "status": "running"})
        plan, maps, supports = fixture()
        maps["sub-01_B_reliability_valid"][0] = False
        maps["sub-01_B_fisher_z"][0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir(mode=0o700)
            private = cache / "private/calibration_operators/attempt001"
            public = root / "public/attempt001"
            with patch.object(method, "PRIVATE", private), patch.object(method, "PUBLIC", public), \
                 patch.object(method.calibration.original, "CACHE_PARENT", cache), \
                 patch.object(method.calibration.original, "load_json", return_value=completed_report()), \
                 patch.object(method, "load_inputs", return_value=(plan, maps, supports)), \
                 patch.object(method.calibration.columns.h5py, "File", side_effect=AssertionError("no raw HDF5 access")):
                self.assertEqual(method.run(), 0)
            report_text = (public / "operator_report.json").read_text()
            report = json.loads(report_text)
            self.assertTrue(report["operator_artifacts_complete"])
            self.assertFalse(report["reader_handoffs_activated"])
            self.assertFalse(report["candidate_scoring_authorized"])
            self.assertFalse(report["raw_neural_values_read"])
            for token in ("H_BC", "H_CD", "undefined_calibration_gain", "endpoint_calibration_prerequisites", "residual_gain_variance"):
                self.assertNotIn(token, report_text)
            private_report = json.loads((private / "applicability.json").read_text())
            self.assertFalse(private_report["H_component_calibration_applicability"]["H_BC"])
            self.assertEqual((private / "applicability.json").stat().st_mode & 0o7777, 0o600)
            self.assertEqual((private / "operators.npz").stat().st_mode & 0o7777, 0o600)


if __name__ == "__main__":
    unittest.main()
