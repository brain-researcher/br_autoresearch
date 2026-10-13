"""Source-free checks for fixed condition/run partitions and role-member planning."""
from collections import Counter
import copy
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import build_condition_partition_plan as plan
import join_source_metadata as metadata
from test_source_metadata import design_fixture, encode_table


def encode_members(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=metadata.PLAN_COLUMNS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def example(multiple_regressors=False):
    ids = [f"sub-{i:02}" for i in range(24)]
    records, summaries = [], {}
    for session in metadata.SESSIONS:
        source = design_fixture(session)
        if multiple_regressors and session == "ses-s1":
            # Preserve all736 members but make one condition use two actual
            # regressors per run, including literal reg_id0, not a shifted ID.
            for run in range(1, 17):
                target = next(row for row in source if row["run"] == run and row["cond_name"] == "AOnly00")
                duplicate = next(row for row in source if row["run"] == run and row["reg_id"] == 0)
                duplicate.update({key: value for key, value in target.items() if key != "reg_id"})
        for pid in ids:
            rows = metadata.parse_reginfo(encode_table(source), pid, session)
            records.extend(rows)
            if pid == ids[0]:
                summaries[session] = metadata.validate_session(rows, Counter(row["beta_member"] for row in rows),
                                                               metadata.EXPECTED_BETAS[session])
    joined = {"metadata_join_complete": True, "beta_or_anatomy_voxel_elements_decoded": False,
        "Task_B_image_streams_opened": 0, "regressor_members_table": "regressor_members.tsv",
        "participants": [{"participant_id": pid, "beta_members_joined": dict(metadata.EXPECTED_BETAS)} for pid in ids],
        "condition_multiplicity_by_session": summaries,
        "condition_design": metadata.validate_task_sets(summaries["ses-s1"], summaries["ses-s2"])}
    flags = {"Task_B_values_read": False, "physical_handoffs_created": False,
             "scientific_margins_activated": False, "candidate_scoring_authorized": False}
    report = {"status": "complete_12_12_role_assignment", "enumeration_complete": True,
        "participant_roles_published": True, "assignments_enumerated": 2704156, **flags}
    roles = {"participants": 24, "enumeration_complete": True, "assignments_enumerated": 2704156,
        "audit_ids": ids[:12], "development_ids": ids[12:], **flags}
    return joined, report, roles, records


class ConditionPartitionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joined, cls.report, cls.roles, cls.records = example()
        cls.payload = encode_members(cls.records)
        cls.result, cls.rows = plan.compile_plan(cls.joined, cls.report, cls.roles, cls.payload)

    def test_every_original_member_and_whole_person_role_are_preserved(self):
        self.assertEqual(len(self.rows), 36480)
        originals = {(r["participant_id"], r["session"], r["run"], r["reg_id"]): r for r in self.records}
        for row in self.rows:
            original = originals[(row["participant_id"], row["session"], row["run"], row["reg_id"])]
            self.assertEqual({key: row[key] for key in metadata.PLAN_COLUMNS}, original)
            audit = row["participant_id"] in self.roles["audit_ids"]
            self.assertEqual(row["participant_role"], "audit" if audit else "development")
            expected = ("audit_calibration" if row["session"] == "ses-s1" else "sealed_audit") if audit else "development"
            self.assertEqual(row["planned_handoff"], expected)
            if row["reg_id"] == 0:
                self.assertTrue(row["beta_member"].endswith("reg-00_beta.nii"))
                self.assertEqual(row["endpoint_use"], "instruction_excluded")
                self.assertEqual(row["condition_map_included"], 0)
        self.assertEqual(self.result["member_counts_by_planned_handoff"],
                         {"audit_calibration": 8832, "sealed_audit": 9408, "development": 18240})
        self.assertEqual(sum(row["endpoint_use"] == "Task_B_primary" for row in self.rows), 24*18*16)

    def test_exact_halves_quarters_domain_weights_and_unresolved_geometry_semantics(self):
        design = self.result["condition_design"]
        self.assertEqual([len(design[key]) for key in ("Task_A_conditions", "Task_B_conditions",
                         "shared_conditions_postdecision_Task_B_diagnostic_only", "Task_B_only_conditions")], [29, 32, 14, 18])
        for groups, length in ((design["map_halves"], 8), (design["geometry_partitions"], 4)):
            self.assertEqual(sorted(run for group in groups for run in group), list(range(1, 17)))
            self.assertTrue(all(len(group) == length for group in groups))
        self.assertEqual(design["primary_geometry_partition_pairs"], [[1, 2], [3, 4]])
        for domain, conditions in metadata.DOMAINS.items():
            values = [Fraction(**design["condition_weights"][condition]) for condition in conditions]
            self.assertEqual(sum(values), Fraction(1, 7))
            chosen = [row for row in self.rows if row["primary_domain"] == domain]
            self.assertTrue(all(Fraction(row["primary_condition_weight_numerator"], row["primary_condition_weight_denominator"])
                                == Fraction(1, 7*len(conditions)) for row in chosen))
        self.assertIsNone(self.result["geometry_pair_weights"]["numerical_weights"])
        self.assertIsNone(self.result["geometry_pair_weights"]["diagonal_and_symmetric_entry_convention"])
        self.assertFalse(self.result["geometry_pair_weights"]["activated"])
        self.assertIsNone(self.result["geometry_sensitivity_pairings"]["pairings"])

    def test_source_regressor_multiplicity_gets_arithmetic_within_run_weight(self):
        joined, report, roles, records = example(multiple_regressors=True)
        result, rows = plan.compile_plan(joined, report, roles, encode_members(records))
        chosen = [row for row in rows if row["participant_id"] == "sub-00"
                  and row["session"] == "ses-s1" and row["cond_name"] == "AOnly00"]
        self.assertEqual(len(chosen), 32)
        for run in range(1, 17):
            members = [row for row in chosen if row["run"] == run]
            self.assertEqual(len(members), 2)
            self.assertIn(0, [row["reg_id"] for row in members])
            self.assertEqual([row["regressors_in_condition_run"] for row in members], [2, 2])
            self.assertEqual([Fraction(row["within_run_weight_numerator"], row["within_run_weight_denominator"])
                              for row in members], [Fraction(1, 2), Fraction(1, 2)])
        self.assertEqual(result["map_construction"]["map_half_run_weight"], {"numerator": 1, "denominator": 8})
        self.assertEqual(result["map_construction"]["geometry_partition_run_weight"], {"numerator": 1, "denominator": 4})

    def test_bad_member_identity_partition_instruction_and_duplicate_fail(self):
        ids = [item["participant_id"] for item in self.joined["participants"]]
        changes = ({"half": 2}, {"geometry_partition": 2}, {"task_set": "B"},
                   {"instruction": 0}, {"beta_member": "../closed.nii"}, {"run": 17})
        for change in changes:
            bad = list(self.records)
            bad[0] = {**bad[0], **change}
            with self.subTest(change=change), self.assertRaises(plan.PlanFailure):
                plan.parse_members(encode_members(bad), ids)
        bad = list(self.records)
        bad[1] = dict(bad[0])
        with self.assertRaises(plan.PlanFailure):
            plan.parse_members(encode_members(bad), ids)
        with self.assertRaises(plan.PlanFailure):
            plan.parse_members(encode_members(self.records[:-1]), ids)

    def test_partial_or_changed_roles_and_source_design_cannot_publish_a_subset(self):
        report = copy.deepcopy(self.report)
        report["enumeration_complete"] = False
        with self.assertRaises(plan.PlanFailure):
            plan.validate_prerequisites(self.joined, report, self.roles)
        roles = copy.deepcopy(self.roles)
        roles["audit_ids"][-1] = roles["development_ids"][0]
        with self.assertRaises(plan.PlanFailure):
            plan.validate_prerequisites(self.joined, self.report, roles)
        roles = copy.deepcopy(self.roles)
        roles["Task_B_values_read"] = True
        with self.assertRaises(plan.PlanFailure):
            plan.validate_prerequisites(self.joined, self.report, roles)
        joined = copy.deepcopy(self.joined)
        joined["condition_multiplicity_by_session"]["ses-s1"]["AOnly00"]["regressors_per_run"][0] = 2
        with self.assertRaisesRegex(plan.PlanFailure, "multiplicity"):
            plan.compile_plan(joined, self.report, self.roles, self.payload)

    def test_no_handoff_scoring_private_value_or_independence_claim_is_activated(self):
        for key in ("physical_handoffs_created", "reader_isolation_established", "audit_calibration_activated",
                    "source_archive_or_image_opened", "participant_numeric_values_exported",
                    "scientific_margins_activated", "candidate_scoring_authorized", "independent_run_errors_established"):
            self.assertIs(self.result[key], False)
        self.assertEqual(self.result["audit_Task_B_open_count"], 0)
        self.assertEqual(self.result["planned_handoffs"]["audit_calibration"]["activation"], "only_after_panel_lock")
        self.assertEqual(self.result["planned_handoffs"]["sealed_audit"]["activation"], "once_after_prediction_lock")
        self.assertNotIn("aggregate_balance_objective", self.result)
        self.assertNotIn("selected_Task_A_header", self.result)
        self.assertTrue(self.result["member_table_scope"].endswith("anatomy_manifest_not_created"))

    def test_loader_reads_only_four_exact_bounded_metadata_paths(self):
        expected = {"metadata": self.joined, "members": self.payload, "role_report": self.report, "roles": self.roles}
        root = Path("/fixture/outputs")
        payloads = {root/relative: expected[key] if key == "members" else json.dumps(expected[key]).encode()
                    for key, (relative, _) in plan.INPUTS.items()}
        reads = []
        def read(path, bound, report):
            reads.append((path, bound))
            return payloads[path]
        with patch.object(plan, "bounded_read", side_effect=read):
            result = plan.load_inputs(root, {"text_bytes_read": 0})
        self.assertEqual(result, expected)
        self.assertEqual(reads, [(root/relative, bound) for relative, bound in plan.INPUTS.values()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"metadata.txt"
            path.write_bytes(b"12345")
            counter = {"text_bytes_read": 0}
            with self.assertRaisesRegex(plan.PlanFailure, "exceeds_bound"):
                plan.bounded_read(path, 4, counter)
            self.assertEqual(counter["text_bytes_read"], 5)

    def test_complete_publication_preserves_rows_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan.publish_plan(self.result, self.rows, output)
            published = json.loads((output/plan.OUTPUT_FILES[0]).read_text())
            self.assertEqual(published, self.result)
            with (output/plan.OUTPUT_FILES[1]).open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 36480)
            self.assertEqual(rows[0]["reg_id"], "0")
            before = (output/plan.OUTPUT_FILES[0]).read_bytes()
            with self.assertRaises(FileExistsError):
                plan.publish_plan(self.result, self.rows, output)
            self.assertEqual((output/plan.OUTPUT_FILES[0]).read_bytes(), before)

    def test_incomplete_or_partial_publication_has_no_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(plan.PlanFailure):
                plan.publish_plan(self.result, self.rows[:-1], output)
            self.assertEqual(list(output.iterdir()), [])
            original_link = plan.os.link
            calls = []
            def fail_second(source, target):
                calls.append(target)
                if len(calls) == 2:
                    raise OSError("fixture publication interruption")
                original_link(source, target)
            with patch.object(plan.os, "link", side_effect=fail_second), self.assertRaises(OSError):
                plan.publish_plan(self.result, self.rows, output)
            self.assertTrue((output/plan.OUTPUT_FILES[0]).is_file())
            self.assertFalse((output/plan.OUTPUT_FILES[1]).exists())
            self.assertFalse((output/"condition_partition_plan_report.json").exists())
            self.assertFalse(any(path.name.endswith(".partial") for path in output.iterdir()))
            self.assertEqual(self.result["publication_requirement"], "both_plan_files_require_complete_terminal_report")

    def test_deadline_after_file_publication_refuses_complete_terminal_state(self):
        # Both metadata files may exist when the bound expires. Consumers must
        # still reject them without a successful terminal report; no retry.
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)/"outputs"
            (outputs/"tools").mkdir(parents=True)
            destination = outputs/"runs/source_qualification/attempt"
            inputs = dict(metadata=self.joined, role_report=self.report, roles=self.roles, members=self.payload)
            with patch.object(plan, "__file__", str(outputs/"tools/build_condition_partition_plan.py")), \
                 patch.object(sys, "argv", ["build_condition_partition_plan.py",
                     "--output-dir", str(destination), "--max-seconds", "120"]), \
                 patch.dict(plan.os.environ, {"SLURM_JOB_ID": "fixture"}), \
                 patch.object(plan.socket, "gethostname", return_value="fixture-compute"), \
                 patch.object(plan.signal, "signal"), patch.object(plan.signal, "alarm"), \
                 patch.object(plan, "load_inputs", return_value=inputs), \
                 patch.object(plan, "compile_plan", return_value=(self.result, self.rows)), \
                 patch.object(plan.time, "monotonic", side_effect=[0., 0., 120., 120.]), \
                 patch("builtins.print"):
                code = plan.main()
            self.assertEqual(code, 1)
            self.assertTrue(all((destination/name).is_file() for name in plan.OUTPUT_FILES))
            report = json.loads((destination/"condition_partition_plan_report.json").read_text())
            self.assertEqual(report["status"], "condition_partition_plan_failure")
            self.assertIs(report["plan_complete"], False)
            self.assertIs(report["automatic_retry"], False)


if __name__ == "__main__":
    unittest.main()
