"""Conditional full-label gate regression; no real prompt/model reads."""
import unittest

import run_all_name_text_features as full


class FullLabelGateTests(unittest.TestCase):
    def record(self):
        return {"status": "first64_name_text_cost_pilot_completed", "completed_concepts": 64,
                "selected_concepts": 64, "unique_concepts_inferred": 64, "batch_size": 32,
                "qualification": str(full.pilot.QUALIFICATION), "prompt_source": str(full.pilot.PROMPTS),
                "projected_full1854_encoding_seconds": 10., "load_seconds": 1., "max_rss_kib": 1024000}

    def test_only_completed_same_scope_pilot_can_qualify_for_manual_invocation(self):
        full.require_successful_pilot(self.record(), 600)
        for key, value in (("status", "running"), ("completed_concepts", 63), ("batch_size", 16),
                           ("prompt_source", "unselected"), ("qualification", "unselected")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Completed fixed64"):
                full.require_successful_pilot({**self.record(), key: value}, 600)

    def test_missing_nonfinite_or_over_budget_measurements_cannot_release(self):
        for key, value in (("projected_full1854_encoding_seconds", None),
                           ("projected_full1854_encoding_seconds", float("nan")),
                           ("projected_full1854_encoding_seconds", 599.),
                           ("load_seconds", -1.), ("max_rss_kib", 7 * 1024**2)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Measured pilot"):
                full.require_successful_pilot({**self.record(), key: value}, 600)


if __name__ == "__main__":
    unittest.main()
