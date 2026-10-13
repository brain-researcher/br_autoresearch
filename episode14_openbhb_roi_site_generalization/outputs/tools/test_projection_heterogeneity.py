#!/usr/bin/env python3
"""Small exact-reconstruction checks; run only in the owned Slurm allocation."""
import os
import socket
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import analyze_projection_heterogeneity as analysis


def row(person, age, raw_error, retained_error, rescaled_error, domain="a", view=0, support=True):
    return {"id": person, "age": float(age), "domain": domain, "view": view,
            "on_common_support": support,
            "predictions": {"raw": age + raw_error, "retained": age + retained_error,
                            "rescaled": age + rescaled_error, "restored": age + raw_error}}


class ReconstructionChecks(unittest.TestCase):
    def test_correction_identity_with_unequal_appearance_counts(self):
        rows = [row("repeat", 40, 1, 0, 1), row("repeat", 40, -2, 0, 1, view=1),
                row("single", 50, 3, 0, 1)]
        actual = analysis.metrics(rows)
        self.assertAlmostEqual(actual["models"]["raw"]["mae"], 2.25)
        self.assertAlmostEqual(actual["models"]["raw"]["mse"], 5.75)
        self.assertAlmostEqual(actual["models"]["raw"]["centered_mae"], 1.75)
        item = actual["contrasts"]["retained_minus_raw"]
        self.assertAlmostEqual(item["delta_mse"], -5.75)
        self.assertAlmostEqual(item["error_correction_alignment"], -11.5)
        self.assertAlmostEqual(item["correction_energy"], 5.75)
        self.assertAlmostEqual(item["delta_centered_mse"], item["centered_error_correction_alignment"] + item["centered_correction_energy"])
        restored = actual["contrasts"]["restored_minus_raw"]
        self.assertTrue(all(value == 0 for value in restored.values()))

    def test_leave_domain_out_recomputes_domain_centers(self):
        rows = [row("repeat", 40, 1, 0, 1), row("repeat", 40, -2, 0, 1, view=1),
                row("single", 50, 3, 0, 1), row("elsewhere", 30, 5, 3, 3, domain="b")]
        report = analysis.analyze_rows(rows)
        pair = "rescaled_minus_raw"
        self.assertAlmostEqual(report["domain_macro"]["contrasts"][pair]["delta_mae"], -1.625)
        omitted_a = report["leave_one_domain_out"]["a"]["influence"][pair]
        omitted_b = report["leave_one_domain_out"]["b"]["influence"][pair]
        self.assertAlmostEqual(omitted_a["delta_mae"]["leave_one_domain_out"], -2.0)
        self.assertAlmostEqual(omitted_b["delta_mae"]["leave_one_domain_out"], -1.25)
        self.assertAlmostEqual(omitted_a["delta_centered_mae"]["leave_one_domain_out"], 0.0)
        self.assertAlmostEqual(omitted_b["delta_centered_mae"]["leave_one_domain_out"], -1.75)
        self.assertEqual(set(report["leave_one_domain_out"]), {"a", "b"})

    def test_support_decomposition_keeps_appearance_reweighting(self):
        rows = [row("repeat", 20, 0, 0, 1, support=True),
                row("repeat", 20, 0, 0, 5, view=1, support=False),
                row("single", 30, 0, 0, 3, support=True),
                row("uncovered", 40, 0, 0, 2, domain="b", support=False)]
        support = analysis.analyze_rows(rows)["common_support"]
        self.assertEqual(support["covered_domains"], ["a"])
        self.assertEqual(support["uncovered_domains"], ["b"])
        self.assertAlmostEqual(support["domains"]["a"]["supported_original_weight_mass"], 0.75)
        observed = support["additive_loss_contrast_difference_decomposition"]["rescaled_minus_raw"]["delta_mae"]
        self.assertAlmostEqual(observed["domain_reweighting"], 0.5)
        self.assertAlmostEqual(observed["within_domain_restriction_original_weights"], -2 / 3)
        self.assertAlmostEqual(observed["appearance_reweighting"], -1 / 3)
        self.assertAlmostEqual(observed["common_minus_all"], -0.5)
        conditional = support["domains"]["a"]["supported_conditional_original_weights"]["models"]["rescaled"]["mae"]
        rebalanced = support["domains"]["a"]["supported_rebalanced_participants"]["models"]["rescaled"]["mae"]
        self.assertAlmostEqual(conditional, 7 / 3)
        self.assertAlmostEqual(rebalanced, 2.0)
        self.assertNotEqual(conditional, rebalanced)

    def test_age_bin_edges_and_outside_ages_are_preserved(self):
        self.assertEqual(analysis.age_bin(0), "0-10")
        self.assertEqual(analysis.age_bin(9.999), "0-10")
        self.assertEqual(analysis.age_bin(10), "10-20")
        self.assertEqual(analysis.age_bin(100), "90-100")
        self.assertEqual(analysis.age_bin(100.01), "outside_0_100")
        self.assertEqual(analysis.age_bin(-1), "outside_0_100")
        rows = [row("low", -1, 1, 0, 0), row("high", 100, 2, 1, 1),
                row("outside", 101, 3, 2, 2)]
        result = analysis.analyze_rows(rows)
        self.assertEqual(result["participants"], 3)
        self.assertEqual(result["age_bins"]["outside_0_100"]["domains"]["a"]["participants"], 2)
        self.assertEqual(result["age_bins"]["90-100"]["domains"]["a"]["participants"], 1)
        self.assertEqual(result["age_bins"]["0-10"]["domains"]["a"]["status"], "empty")
        self.assertEqual(result["common_support"]["uncovered_domains"], [])


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Run only in the coordinator's actual Slurm allocation")
    with threadpool_limits(limits=1):
        unittest.main(verbosity=2)
