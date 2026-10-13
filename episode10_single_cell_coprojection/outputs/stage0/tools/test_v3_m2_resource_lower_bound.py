"""Check the resource bound against the existing combined validator and timer."""
from types import SimpleNamespace
import signal
import time
import unittest
from unittest.mock import patch

import diagnose_v3_m2_resource_lower_bound as diagnostic
import run_v3_m2_observed_search_v1 as observed


class ResourceBoundTests(unittest.TestCase):
    def test_lower_bound_agrees_with_original_combined_formula(self):
        names = (
            "synthetic_membership_generation", "fresh_M0_replay_refit", "fresh_M1_replay_refit",
            "shared_12_fold_transform_and_offset_cache_setup", "reference_12_fold_trial",
            "core_12_fold_trial", "optional_trigger_diagnostic", "selected_full_development_refit_pair",
            "controller_selection_and_aggregate_report", "low_rank_12_fold_trial", "hybrid_12_fold_trial",
        )
        zero = dict(wall_seconds=0.0, cpu_seconds=0.0, scratch_gib=0.0, peak_rss_gib=0.0)
        scale = {"measured_units": {name: dict(zero) for name in names}}
        falsifier = {"measured_units": {name: dict(zero) for name in diagnostic.timing.REQUIRED_UNITS}}
        unit = falsifier["measured_units"][diagnostic.timing.UNIT_PERMUTATION]
        unit["wall_seconds"] = 284.0
        observed._validate_combined_resource_ceiling(scale, falsifier)
        unit["wall_seconds"] = 300.0
        with self.assertRaisesRegex(observed.ObservedSearchRunnerError, "cap 80, coverage 16 exceeds cpu_core_hours"):
            observed._validate_combined_resource_ceiling(scale, falsifier)
        self.assertEqual(diagnostic.projected_core_hour_lower_bound(300.0), 1900.0)

    def test_interrupted_prefix_restores_timer_without_accepting_fit(self):
        fixture = SimpleNamespace(cells=(), candidates=None, geometry=None)
        old_handler = signal.getsignal(signal.SIGALRM)
        with patch.object(diagnostic, "PREFIX_DEADLINE_SECONDS", 0.01), \
             patch.object(diagnostic.timing.permutation_m1, "fit_permuted_outer_m1", side_effect=lambda *a, **k: time.sleep(1)), \
             patch.object(diagnostic.timing, "validate_permutation_accounting") as validate:
            lower, elapsed, interrupted = diagnostic.measure_prefix(fixture, None)
        self.assertTrue(interrupted)
        self.assertGreaterEqual(elapsed, lower)
        self.assertGreater(lower, 0)
        validate.assert_not_called()
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), old_handler)

    def test_unexpected_fit_error_is_not_a_resource_failure(self):
        fixture = SimpleNamespace(cells=(), candidates=None, geometry=None)
        old_handler = signal.getsignal(signal.SIGALRM)
        with patch.object(diagnostic.timing.permutation_m1, "fit_permuted_outer_m1", side_effect=ValueError("fit failure")):
            with self.assertRaisesRegex(ValueError, "fit failure"):
                diagnostic.measure_prefix(fixture, None)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), old_handler)


if __name__ == "__main__":
    unittest.main()
