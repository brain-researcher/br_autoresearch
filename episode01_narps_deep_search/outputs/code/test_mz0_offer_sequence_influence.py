"""Focused saved-score weighting, membership and finite-scope checks; Slurm only."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import summarize_mz0_offer_sequence_influence as diagnostic


def fixture():
    primary = {"phase": "MZ_SESOI_fitted_recovery_component", "generating_model": "MZ",
        "replicate_index": 0, "seed": 910031, "participants": 108,
        "status": "necessary_primary_comparison_completed", "participant_scores": {}}
    schedule = {"analysis_id": "public_asset_qualification", "job_id": "46424502",
        "source_event_table": "outputs/runs/public_asset_qualification/20261003-attempt01/original_events.csv.gz",
        "choice_RT_neural_values_used": False, "grouping": "exact_whole_participant_ordered_run_trial_gain_loss",
        "offer_only_template_counts": {"EI": 8, "ER": 8}, "expected_nominal_templates_per_version": 8,
        "matches_documented_eight_offer_sequences": True, "groups": []}
    members = []
    for version in diagnostic.VERSIONS:
        person_index = 0
        for group_index, count in enumerate((1, 2, 3, 4, 5, 6, 7, 26), 1):
            group = f"{version}-offers-{group_index:02d}"
            people = []
            gain = (1. if version == "EI" else 2.) if group_index == 1 else (
                -.5 if version == "EI" and group_index == 2 else 0.)
            for _ in range(count):
                person_index += 1
                person = f"{version}-{person_index:03d}"
                people.append(person)
                primary["participant_scores"][person] = {"version": version, "M0": -3., "MZ": -3. + gain}
                members.append({"participant": person, "task_version": version, "offer_sequence_id": group})
            schedule["groups"].append({"offer_sequence_id": group, "participants": people})
    return primary, schedule, members


class OfferSequenceInfluenceTests(unittest.TestCase):
    def test_unequal_groups_keep_participant_weight_and_all_single_and_pair_omissions(self):
        inputs = fixture()
        preserved = deepcopy(inputs)
        report = diagnostic.new_report()
        diagnostic.fill_report(report, *inputs, lambda: None, lambda: None)
        self.assertEqual(inputs, preserved)
        self.assertEqual(report["progress"], {"template_groups": 16, "single_omissions": 16, "pair_omissions": 120})
        full = report["full_panel_reference"]
        self.assertEqual(full["participants_by_version"], {"EI": 54, "ER": 54})
        self.assertEqual(full["mean_gain_nats_per_trial"], 2/108)
        self.assertEqual(full["version_means"], {"EI": 0., "ER": 2/54})
        self.assertNotEqual(full["mean_gain_nats_per_trial"], 2.5/16, "Never weight unequal templates equally")
        groups = {row["offer_sequence_id"]: row for row in report["template_groups"]}
        self.assertEqual((groups["EI-offers-02"]["participants"], groups["EI-offers-02"]["mean_gain_nats_per_trial"]), (2, -.5))
        singles = {tuple(row["omitted_offer_sequence_ids"]): row for row in report["single_omissions"]}
        one = singles["EI-offers-01",]
        self.assertEqual((one["participants"], one["mean_gain_nats_per_trial"]), (107, 1/107))
        self.assertEqual(one["version_means"], {"EI": -1/53, "ER": 2/54})
        self.assertEqual(one["version_mean_positive"], {"EI": False, "ER": True})
        pairs = {tuple(row["omitted_offer_sequence_ids"]): row for row in report["pair_omissions"]}
        self.assertEqual(len(pairs), 120)
        self.assertEqual(Counter(row["pair_type"] for row in pairs.values()), {"within_version": 56, "cross_version": 64})
        within = pairs["EI-offers-01", "EI-offers-02"]
        self.assertEqual((within["participants"], within["mean_gain_nats_per_trial"]), (105, 2/105))
        cross = pairs["EI-offers-01", "ER-offers-01"]
        self.assertEqual((cross["participants"], cross["mean_gain_nats_per_trial"]), (106, -1/106))
        self.assertEqual(cross["participants_by_version"], {"EI": 53, "ER": 53})
        self.assertEqual(cross["version_mean_positive"], {"EI": False, "ER": False})
        self.assertEqual(cross["mean_change_from_full_panel"], -1/106 - 2/108)
        self.assertNotIn("participant_SE", full)
        self.assertEqual(report["cluster_intervals"], "not_evaluated")
        self.assertEqual(report["full_gate"], "not_evaluated")

    def test_missing_duplicate_and_mismatched_memberships_or_invalid_saved_scores_refuse(self):
        mutations = (
            lambda p, s, m: m.pop(),
            lambda p, s, m: m.append(deepcopy(m[0])),
            lambda p, s, m: m[0].update(task_version="ER"),
            lambda p, s, m: m[0].update(offer_sequence_id="ER-offers-01"),
            lambda p, s, m: s["groups"][0]["participants"].append(s["groups"][0]["participants"][0]),
            lambda p, s, m: s["groups"].pop(),
            lambda p, s, m: p["participant_scores"]["EI-001"].update(MZ=float("nan")),
            lambda p, s, m: p.update(status="technical_failure"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                inputs = fixture()
                mutation(*inputs)
                with self.assertRaises(ValueError):
                    diagnostic.validate_inputs(*inputs)

    def test_whole_cap_preserves_actual_partial_progress_and_closes_late_success(self):
        for stage, expected_pairs in (("pairs", 16), ("final_publication", 120)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                episode = Path(directory)
                output = episode / "outputs/result.json"
                now = [100.]
                inputs = fixture()
                preserved = deepcopy(inputs)
                write = diagnostic.write_report
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

                def writer(path, report):
                    write(path, report)
                    if ((stage == "pairs" and report["progress"]["pair_omissions"] == 16)
                            or (stage == "final_publication" and report["diagnostic_complete"])):
                        now[0] = 161.

                with patch.object(diagnostic, "EPISODE", episode), patch.object(diagnostic, "require_compute"), \
                        patch.object(diagnostic, "read_inputs", return_value=inputs), \
                        patch.object(diagnostic.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(diagnostic, "write_report", side_effect=writer):
                    report = diagnostic.run(output)
                    with self.assertRaises(FileExistsError):
                        diagnostic.run(output)
                self.assertEqual(inputs, preserved)
                self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)
                self.assertEqual(json.loads(output.read_text()), report)
                self.assertEqual(report["status"], "technical_offer_sequence_influence_failure")
                self.assertFalse(report["diagnostic_complete"])
                self.assertEqual(report["progress"]["pair_omissions"], expected_pairs)
                self.assertEqual(len(report["pair_omissions"]), expected_pairs)
                self.assertEqual(report["progress"]["single_omissions"], 16)
                self.assertIn("deadline", report["error"])
                self.assertGreaterEqual(report["elapsed_seconds"], 0.)
                self.assertFalse(report["models_refitted"])
                self.assertEqual(report["full_gate"], "not_evaluated")


if __name__ == "__main__":
    unittest.main()
