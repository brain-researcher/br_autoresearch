"""Focused comparator arithmetic tests; run only in root's compute allocation.

These explicit arrays are interface fixtures, not simulated scientific data or
replacement NPZ files. No solver, input record or scientific artifact is read.
"""
import copy
from contextlib import redirect_stdout
import io
import os
import socket
import unittest

import numpy as np

import fast_solver_fastpath_compare as compare


def fixture():
    ends = np.array([5e-6, 62.5e-6, .004])
    starts = np.r_[0., ends[:-1]]
    holds = np.zeros((3, 24))
    work = np.arange(1, 25, dtype=float).reshape(3, 8)*1e-15
    ref = dict(t_start=starts, t_end=ends, hold_mean=holds, hold_end=holds.copy(),
               source_mean=np.zeros((3, 64)), work_fields=np.array(compare.WORK_FIELDS), work=work,
               conv_start=starts.copy(), conv_end=ends.copy(), conv_adc=np.zeros(3, dtype=int),
               pixel_conv=np.array([[0, 1, 2]], dtype=int))
    new = dict(t_start=starts.copy(), t_end=ends.copy(), hold_mean=holds.copy(), hold_end=holds.copy(),
               source_mean=np.full((3, 64), np.nan), work_fields=np.array(compare.WORK_FIELDS),
               window_names=np.array([name for name, _ in compare.WINDOWS]),
               window_work=np.cumsum(work, axis=0),
               # Deliberately incomplete interval work must never determine acceptance.
               work=np.full((3, 8), np.nan))
    checkpoints = dict(slots=np.array([0, 2, 25, 1600]), times=np.r_[0., ends],
                       states=np.zeros((4, compare.STATE_COUNT)))
    audit = dict(available=True, stored_start=0., stored_end=2e-15, input_work=3e-15,
                 heat=1e-15, supplied_charge_work=0., residual=0., relative_residual=0.,
                 method="independent audit fixture", upstream_quadrature_orders=[16, 32],
                 downstream_quadrature_order=16,
                 quadrature_signed_difference=dict(input_work=0., heat=0.),
                 quadrature_absolute_difference_sum=dict(input_work=0., heat=0.),
                 downstream_order_doubling=dict(available=False, reason="original 16-point kernel"),
                 residual_acceptance=dict(available=False, reason="diagnostic only"))
    diagnostic = dict(available=True, segments=1, sweeps=2, max_sweeps=2,
                      max_wr_change=.1, max_rep_pad=.2, max_rep_source=.3, clip_events=0)
    report = dict(status="complete",
                  window_energy_audit={name: copy.deepcopy(audit) for name, _ in compare.WINDOWS},
                  window_solver_diagnostics={name: diagnostic.copy() for name, _ in compare.WINDOWS})
    return ref, new, checkpoints, copy.deepcopy(checkpoints), report


def evaluate(items):
    ref, new, ref_ck, new_ck, report = items
    return compare.compare_arrays(ref, new, ref_ck, new_ck, solver_report=report,
                                  thresholds=np.array([-1e-7, 0., 1e-7]),
                                  levels=np.array([-1.5e-7, -.5e-7, .5e-7, 1.5e-7]))


class FastpathComparatorTests(unittest.TestCase):
    def test_main_help_exits_without_payload_or_paths(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exit_status:
            compare.main(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        for option in ("--reference", "--solver", "--output"):
            self.assertIn(option, output.getvalue())

    def test_original_absolute_and_relative_tolerances(self):
        self.assertEqual((compare.STATE_ATOL, compare.STATE_RTOL), (1e-8, 1e-5))
        self.assertEqual((compare.WORK_ATOL, compare.WORK_RTOL), (1e-18, 1e-5))
        zero = np.zeros(1)
        self.assertTrue(compare.voltage_comparison(zero, np.array([1e-8]))["passed"])
        self.assertFalse(compare.voltage_comparison(zero, np.array([np.nextafter(1e-8, np.inf)]))["passed"])
        for factor, expected in ((.99, True), (1.01, False)):
            self.assertEqual(compare.voltage_comparison(np.ones(1),
                np.array([1.+factor*(1e-8+1e-5)]))["passed"], expected)
        zero_work = np.zeros(8)
        actual = zero_work.copy()
        actual[0] = 1e-18
        self.assertTrue(compare.physical_work_comparison(zero_work, actual)[compare.WORK_FIELDS[0]]["passed"])
        actual[0] = np.nextafter(1e-18, np.inf)
        self.assertFalse(compare.physical_work_comparison(zero_work, actual)[compare.WORK_FIELDS[0]]["passed"])
        reference_work = np.full(8, 1e-12)
        for factor, expected in ((.99, True), (1.01, False)):
            actual = reference_work+factor*(1e-18+1e-5*reference_work)
            self.assertEqual(all(x["passed"] for x in compare.physical_work_comparison(reference_work, actual).values()), expected)

    def test_threshold_tie_one_code_and_one_third_lsb_boundary(self):
        thresholds = np.array([-1e-7, 0., 1e-7])
        levels = np.array([-1.5e-7, -.5e-7, .5e-7, 1.5e-7])
        pixels = np.array([[0, 1, 2]])
        reference = np.zeros(3)  # An exact tie goes to the upper code.
        codes, result = compare.code_pixel_comparison(reference, np.array([-1e-12, 0., 0.]), thresholds, levels, pixels)
        self.assertTrue(codes["passed"])
        self.assertEqual(codes["differ_allowed"], 1)
        self.assertTrue(result["passed"])
        codes, result = compare.code_pixel_comparison(reference, np.array([-1e-12, -1e-12, 0.]), thresholds, levels, pixels)
        self.assertTrue(codes["passed"])
        self.assertFalse(result["passed"])
        codes, _ = compare.code_pixel_comparison(np.full(3, 5e-8), np.full(3, 1.5e-7), thresholds, levels, pixels)
        self.assertFalse(codes["passed"])  # Reference is not within voltage tolerance of a threshold.
        codes, _ = compare.code_pixel_comparison(reference, np.full(3, -2e-7), thresholds, levels, pixels)
        self.assertFalse(codes["passed"])  # Two-code differences are never admitted.
        with self.assertRaisesRegex(ValueError, "nonfinite_conversion_means"):
            compare.code_pixel_comparison(reference, np.array([np.nan, 0., 0.]), thresholds, levels, pixels)

    def test_complete_window_work_only_and_physical_failure(self):
        items = fixture()
        result = evaluate(items)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["required_windows"], ["5us", "dwell", "frame"])
        self.assertTrue(result["windows"]["frame"]["solver"]["available"])
        self.assertFalse(result["windows"]["frame"]["source_mean"]["available"])
        self.assertFalse(result["windows"]["frame"]["per_interval_solver"]["available"])
        items[1]["window_work"][2, 0] += 1e-12
        result = evaluate(items)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["available_window_failure"])
        self.assertEqual(result["missing_windows"], [])
        self.assertEqual(result["invalid_windows"], [])
        self.assertEqual(result["windows"]["frame"]["status"], "physical_tolerance_failure")

    def test_partial_reference_and_invalid_required_evidence_are_distinct(self):
        items = fixture()
        items[2]["slots"] = items[2]["slots"][:-1]
        result = evaluate(items)
        self.assertEqual(result["missing_windows"], ["frame"])
        self.assertEqual(result["invalid_windows"], [])
        self.assertFalse(result["available_window_failure"])
        self.assertFalse(result["qualified"])
        for window in ("5us", "dwell"):
            self.assertTrue(result["windows"][window]["passed"])
        items = fixture()
        items[1]["window_names"] = items[1]["window_names"][:-1]
        items[1]["window_work"] = items[1]["window_work"][:-1]
        result = evaluate(items)
        self.assertEqual(result["invalid_windows"], ["frame"])
        self.assertFalse(result["qualified"])
        items = fixture()
        items[1]["window_work"][2, 0] = np.nan
        self.assertEqual(evaluate(items)["invalid_windows"], ["frame"])
        items = fixture()
        items[1]["t_start"][1] += 1e-8
        self.assertEqual(evaluate(items)["invalid_windows"], ["dwell", "frame"])
        items = fixture()
        items[0]["conv_start"][0] = 1e-6
        result = evaluate(items)
        self.assertEqual(result["invalid_windows"], ["5us", "dwell", "frame"])
        self.assertFalse(result["qualified"])  # Incomplete apertures cannot disappear from the code test.

    def test_finite_independent_audit_completeness_without_new_residual_margin(self):
        for mutation in ("missing", "nonfinite", "missing_quadrature", "nonfinite_quadrature"):
            items = fixture()
            audit = items[4]["window_energy_audit"]["frame"]
            if mutation == "missing":
                del items[4]["window_energy_audit"]["frame"]
            elif mutation == "nonfinite":
                audit["residual"] = np.nan
            elif mutation == "missing_quadrature":
                del audit["quadrature_signed_difference"]
            else:
                audit["quadrature_absolute_difference_sum"]["heat"] = np.nan
            result = evaluate(items)
            self.assertTrue(result["all_physical_windows_passed"])
            self.assertFalse(result["available_window_failure"])
            self.assertFalse(result["complete_window_energy_audits_available"])
            self.assertFalse(result["qualified"])
        items = fixture()
        items[4]["window_energy_audit"]["frame"].update(stored_end=102e-15, residual=100e-15,
                                                        relative_residual=100./102.)
        result = evaluate(items)
        self.assertTrue(result["qualified"])  # Finite residuals are reported; no new tolerance is invented.
        self.assertEqual(result["windows"]["frame"]["energy_audit"]["residual"], 100e-15)
        items[4]["status"] = "failed"
        self.assertFalse(evaluate(items)["qualified"])


if __name__ == "__main__":
    if "-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("compute allocation required; root owns the test session")
    unittest.main()
