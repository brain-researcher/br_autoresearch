from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ep12_executor.coverage_successor import (
    SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
    CoverageSuccessorError,
    derive_error_budget,
)
from ep12_executor.production_readiness import (
    REQUIRED_READINESS_CHECK_IDS,
    ReadinessError,
    claim_is_permitted,
    evaluate_production_readiness,
    verify_contract_roles,
    verify_full_synthetic_qualification,
    write_readiness_artifacts,
)
from ep12_executor.runtime import EPISODE_ROOT
from ep12_executor.scientific_qualification import (
    _accept_cell,
    _cell_summary,
    _write_csv,
    _write_markdown,
    qualification_cells,
)
from ep12_executor.scientific_synthetic import SCENARIO_TRUTH
from ep12_executor.yaml_subset import load_yaml_subset


class ProductionReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qualification_spec_path = (
            EPISODE_ROOT
            / "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
        )
        cls.qualification_spec = load_yaml_subset(cls.qualification_spec_path)
        cls.expected_cells = qualification_cells(
            "qualification", cls.qualification_spec
        )
        cls.expected_repetitions = int(
            cls.qualification_spec["run_tiers"][
                "qualification_replicates_per_core_error_cell"
            ]
        )
        cls.expected_whole_types = int(
            cls.qualification_spec["run_tiers"]["whole_types_per_replicate"]
        )
        oracle_spec = cls.qualification_spec["stochastic_oracle"]
        cls.coverage_error_budget = derive_error_budget(
            familywise_confidence_level=float(
                oracle_spec["familywise_confidence_level"]
            ),
            cell_count=len(cls.expected_cells),
            planned_look_count=len(
                oracle_spec["tiers"]["qualification"][
                    "planned_oracle_replicates_per_cell"
                ]
            ),
        )

    def _write_complete_qualification(self, root: Path) -> Path:
        directory = root / "qualification"
        directory.mkdir()
        records_by_cell: dict[str, list[dict]] = {
            cell.cell_id: [] for cell in self.expected_cells
        }
        records = directory / "replicate_results.jsonl"
        with records.open("w", encoding="utf-8") as handle:
            for cell in self.expected_cells:
                for replicate_index in range(self.expected_repetitions):
                    coverage_success = replicate_index < self.expected_repetitions - 3
                    record = {
                        "cell_id": cell.cell_id,
                        "scenario": cell.scenario,
                        "truth_family": SCENARIO_TRUTH[cell.scenario],
                        "replicate_index": replicate_index,
                        "terminal": "unresolved",
                        "intervals": {
                            "labels": [
                                "left_to_right",
                                "right_to_left",
                                "aggregate",
                            ],
                            "estimate": [0.0, 0.0, 0.0],
                            "standard_error": [0.1, 0.1, 0.1],
                            "critical_max_t": 1.0,
                            "lower": (
                                [-1.0, -1.0, -1.0]
                                if coverage_success
                                else [1.0, 1.0, 1.0]
                            ),
                            "upper": (
                                [1.0, 1.0, 1.0]
                                if coverage_success
                                else [2.0, 2.0, 2.0]
                            ),
                            "margin": 0.02,
                            "bootstrap_replicates": 1999,
                        },
                        "controls": {
                            "endpoint_side_permutation_p": 0.0,
                            "endpoint_mass_control_pass": False,
                            "leave_one_type_out_directional_gain": [
                                [0.0, 0.0]
                                for _ in range(self.expected_whole_types)
                            ],
                            "influence_control_pass": True,
                            "all_pass": False,
                        },
                        "type_records": [
                            {
                                "synthetic_type_index": type_index,
                                "deltas": [0.0, 0.0],
                                "source_fit_metadata": {},
                            }
                            for type_index in range(self.expected_whole_types)
                        ],
                        "numerical_failures": [],
                        "wall_seconds": 0.01,
                        "cpu_seconds": 0.005,
                    }
                    records_by_cell[cell.cell_id].append(record)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        oracle_enclosure = {
            "lower": [-0.001, -0.001, -0.001],
            "upper": [0.001, 0.001, 0.001],
            "precision_met": True,
            "maximum_half_width": 0.001,
            "estimand": "infinite_design_expected_fixed_finite_procedure_gain",
            "independent_of_outer_qualification": True,
        }
        oracle_path = root / "stochastic_oracle_aggregate.json"
        oracle_report = {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "scientific_gate": True,
            "all_cells_precision_met": True,
            "cells": [
                {
                    "cell": {"cell_id": cell.cell_id},
                    "oracle_enclosure": dict(oracle_enclosure),
                }
                for cell in self.expected_cells
            ],
            "error_budget": asdict(self.coverage_error_budget),
            "real_connectivity_accessed": False,
            "final_connectivity_accessed": False,
        }
        oracle_path.write_text(json.dumps(oracle_report), encoding="utf-8")
        summaries = []
        for cell in self.expected_cells:
            summary = _cell_summary(
                cell,
                records_by_cell[cell.cell_id],
                oracle_enclosure=oracle_enclosure,
                binomial_confidence_level=(
                    self.coverage_error_budget.binomial_readout_confidence_level
                ),
            )
            passed, reasons = _accept_cell(
                summary, self.qualification_spec["acceptance"]
            )
            summary["passed"] = passed
            summary["failure_reasons"] = reasons
            self.assertTrue(passed, (cell.cell_id, reasons))
            summaries.append(summary)
        cost = {
            "cost_gate_passed": True,
            "measured_type_direction_triplets": (
                len(self.expected_cells)
                * self.expected_repetitions
                * self.expected_whole_types
                * 2
            ),
        }
        stability = {"passed": True}
        report = {
            "tier": "qualification",
            "qualification_scope": "actual_T_U_M_generated_data_only",
            "contract_configuration": True,
            "scientific_contract_enforced": True,
            "qualification_passed": True,
            "cells": summaries,
            "settings": {
                "repetitions_per_cell": self.expected_repetitions,
                "whole_types_per_replicate": self.expected_whole_types,
            },
            "cost_profile": cost,
            "integration_stability": stability,
            "stochastic_oracle_aggregate_path": str(oracle_path),
            "stochastic_oracle_error_budget": asdict(self.coverage_error_budget),
            "real_connectivity_accessed": False,
            "real_connectivity_access_authorized": "development_only",
        }
        (directory / "cost_profile.json").write_text(
            json.dumps(cost), encoding="utf-8"
        )
        (directory / "integration_stability.json").write_text(
            json.dumps(stability), encoding="utf-8"
        )
        _write_csv(directory / "result_table.csv", summaries)
        _write_markdown(directory / "result_table.md", summaries, report)
        (directory / "qualification_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return directory

    def _verify_qualification(self, directory: Path, root: Path) -> dict:
        return verify_full_synthetic_qualification(
            qualification_dir=directory,
            qualification_spec_path=self.qualification_spec_path,
            outputs_root=root,
        )

    def _publication_report(self, *, passed: bool) -> dict:
        failed_id = REQUIRED_READINESS_CHECK_IDS[1]
        checks = []
        for check_id in REQUIRED_READINESS_CHECK_IDS:
            check_passed = passed or check_id != failed_id
            checks.append(
                {
                    "check_id": check_id,
                    "passed": check_passed,
                    **(
                        {"evidence": {"verified": True}}
                        if check_passed
                        else {"reason": "synthetic prerequisite is incomplete"}
                    ),
                }
            )
        failed = [] if passed else [failed_id]
        return {
            "status": "pass" if passed else "revise",
            "episode_id": "ep12",
            "policy_id": "ep12_malecns_type_sufficiency_v1",
            "gate_scope": "readiness_for_observed_post36_tail_only",
            "checks": checks,
            "failed_foundational_checks": failed,
            "production_readiness_passed": passed,
            "claim_guards": {
                "observed_post36_tail_launch": {
                    "permitted": passed,
                    "missing": failed,
                }
            },
            "observed_outcome_values_read": False,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }

    def test_complete_full_qualification_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-qualification-") as name:
            root = Path(name)
            result = self._verify_qualification(
                self._write_complete_qualification(root), root
            )
        self.assertEqual(
            result["replicate_record_count"],
            len(self.expected_cells) * self.expected_repetitions,
        )

    def test_tampered_passing_cell_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-summary-tamper-") as name:
            root = Path(name)
            directory = self._write_complete_qualification(root)
            report_path = directory / "qualification_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["cells"][0]["false_M_rate"] = 0.5
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ReadinessError, "summaries differ"):
                self._verify_qualification(directory, root)

    def test_imprecise_stochastic_oracle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-oracle-tamper-") as name:
            root = Path(name)
            directory = self._write_complete_qualification(root)
            report = json.loads(
                (directory / "qualification_report.json").read_text(encoding="utf-8")
            )
            oracle_path = Path(report["stochastic_oracle_aggregate_path"])
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            oracle["all_cells_precision_met"] = False
            oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
            with self.assertRaisesRegex(
                CoverageSuccessorError, "not qualification-ready"
            ):
                self._verify_qualification(directory, root)

    def test_scientifically_failing_error_or_coverage_records_are_rejected(self) -> None:
        target_cell_id = next(
            cell.cell_id
            for cell in self.expected_cells
            if cell.scenario == "adequate_template"
            and cell.acceptance_role == "core_operating_characteristic"
        )
        for mutation in ("false_M", "coverage"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-scientific-failure-"
            ) as name:
                root = Path(name)
                directory = self._write_complete_qualification(root)
                records_path = directory / "replicate_results.jsonl"
                records = [
                    json.loads(line)
                    for line in records_path.read_text(encoding="utf-8").splitlines()
                ]
                for record in records:
                    if record["cell_id"] != target_cell_id:
                        continue
                    if mutation == "false_M":
                        record["terminal"] = "M_supported_before_full_search_null"
                    else:
                        record["intervals"]["lower"] = [1.0, 1.0, 1.0]
                        record["intervals"]["upper"] = [2.0, 2.0, 2.0]
                records_path.write_text(
                    "".join(
                        json.dumps(record, sort_keys=True) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ReadinessError, "recomputed qualification acceptance failed"
                ):
                    self._verify_qualification(directory, root)

    def test_readiness_publication_allows_only_monotone_revise_to_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-publication-") as name:
            root = Path(name)
            output = root / "readiness"
            revise = self._publication_report(passed=False)
            passed = self._publication_report(passed=True)
            write_readiness_artifacts(output, revise, outputs_root=root)
            write_readiness_artifacts(output, passed, outputs_root=root)
            write_readiness_artifacts(output, passed, outputs_root=root)
            self.assertEqual(
                json.loads(
                    (output / "production_readiness.json").read_text(encoding="utf-8")
                ),
                passed,
            )

    def test_passing_readiness_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-pass-lock-") as name:
            root = Path(name)
            output = root / "readiness"
            passed = self._publication_report(passed=True)
            write_readiness_artifacts(output, passed, outputs_root=root)
            conflicting = json.loads(json.dumps(passed))
            conflicting["checks"][0]["evidence"] = {"verified": "different"}
            with self.assertRaisesRegex(
                ReadinessError, "only advance from revise to pass"
            ):
                write_readiness_artifacts(output, conflicting, outputs_root=root)
            with self.assertRaisesRegex(
                ReadinessError, "only advance from revise to pass"
            ):
                write_readiness_artifacts(
                    output,
                    self._publication_report(passed=False),
                    outputs_root=root,
                )

    def test_readiness_publication_rejects_nonmonotone_or_unauthenticated_updates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-conflict-") as name:
            root = Path(name)
            output = root / "readiness"
            revise = self._publication_report(passed=False)
            write_readiness_artifacts(output, revise, outputs_root=root)
            conflicting_revise = json.loads(json.dumps(revise))
            conflicting_revise["checks"][0]["evidence"] = {"verified": "different"}
            with self.assertRaisesRegex(
                ReadinessError, "only advance from revise to pass"
            ):
                write_readiness_artifacts(
                    output, conflicting_revise, outputs_root=root
                )
            wrong_identity = self._publication_report(passed=True)
            wrong_identity["policy_id"] = "another_policy"
            with self.assertRaisesRegex(ReadinessError, "identity differs"):
                write_readiness_artifacts(output, wrong_identity, outputs_root=root)

        with tempfile.TemporaryDirectory(prefix="ep12-readiness-invalid-") as name:
            root = Path(name)
            malformed = self._publication_report(passed=True)
            malformed["checks"] = []
            with self.assertRaisesRegex(ReadinessError, "nonempty list"):
                write_readiness_artifacts(
                    root / "readiness", malformed, outputs_root=root
                )

    def test_report_only_partial_qualification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-readiness-partial-") as name:
            root = Path(name)
            directory = root / "qualification"
            directory.mkdir()
            (directory / "qualification_report.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(ReadinessError, "missing qualification artifact"):
                self._verify_qualification(directory, root)

    def test_wrong_tier_or_replicate_count_is_rejected(self) -> None:
        for mutation, pattern in (("tier", "tier is not full"), ("replicates", "replicate count")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-wrong-tier-"
            ) as name:
                root = Path(name)
                directory = self._write_complete_qualification(root)
                report_path = directory / "qualification_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if mutation == "tier":
                    report["tier"] = "smoke"
                else:
                    report["cells"][0]["replicates"] -= 1
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ReadinessError, pattern):
                    self._verify_qualification(directory, root)

    def test_malformed_replicate_scientific_records_are_rejected(self) -> None:
        cases = (
            ("terminal", "valid nonfailure terminal"),
            ("intervals", "intervals must be a mapping"),
            ("controls", "controls must be a mapping"),
            ("type_indices", "type indices are incomplete or duplicated"),
            ("deltas", "two directional deltas"),
            ("source_fit_metadata", "source fit metadata must be a mapping"),
        )
        for mutation, pattern in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-record-shape-"
            ) as name:
                root = Path(name)
                directory = self._write_complete_qualification(root)
                records_path = directory / "replicate_results.jsonl"
                lines = records_path.read_text(encoding="utf-8").splitlines()
                record = json.loads(lines[0])
                if mutation == "terminal":
                    record["terminal"] = "numerical_failure"
                elif mutation == "intervals":
                    record["intervals"] = []
                elif mutation == "controls":
                    record["controls"] = []
                elif mutation == "type_indices":
                    record["type_records"][0]["synthetic_type_index"] = 1
                elif mutation == "deltas":
                    record["type_records"][0]["deltas"] = [0.0]
                else:
                    record["type_records"][0]["source_fit_metadata"] = []
                lines[0] = json.dumps(record, sort_keys=True)
                records_path.write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ReadinessError, pattern):
                    self._verify_qualification(directory, root)

    def test_result_table_replicates_and_passed_are_validated(self) -> None:
        for mutation, pattern in (
            ("replicates", "wrong replicate count"),
            ("passed", "failed cell"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-result-table-"
            ) as name:
                root = Path(name)
                directory = self._write_complete_qualification(root)
                table_path = directory / "result_table.csv"
                with table_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = reader.fieldnames
                    rows = list(reader)
                if mutation == "replicates":
                    rows[0]["replicates"] = str(self.expected_repetitions - 1)
                else:
                    rows[0]["passed"] = "False"
                with table_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ReadinessError, pattern):
                    self._verify_qualification(directory, root)

    def test_missing_cost_or_stability_gate_is_rejected(self) -> None:
        for key, pattern in (
            ("cost_profile", "cost profile"),
            ("integration_stability", "integration stability"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory(
                prefix="ep12-readiness-missing-gate-"
            ) as name:
                root = Path(name)
                directory = self._write_complete_qualification(root)
                report_path = directory / "qualification_report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report.pop(key)
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ReadinessError, pattern):
                    self._verify_qualification(directory, root)

    def test_contract_roles_preserve_development_and_final_boundaries(self) -> None:
        executor = EPISODE_ROOT / "outputs" / "executor"
        evidence = verify_contract_roles(
            prefix_contract_path=executor / "SCIENTIFIC_CONTRACT.yaml",
            active_contract_path=executor / "SCIENTIFIC_CONTRACT_POST36.yaml",
        )
        self.assertFalse(evidence["final_connectivity_accessed"])

    def test_current_readiness_stays_revise_for_scientific_qualification(self) -> None:
        outputs = EPISODE_ROOT / "outputs"
        executor = outputs / "executor"
        run_root = outputs / "development_run002"
        report = evaluate_production_readiness(
            qualification_dir=outputs / "prelaunch" / "scientific_qualification_dev007",
            observed_run_root=run_root,
            observed_finalization_dir=run_root / "observed_finalization",
            falsifier_plan_path=executor / "OBSERVED_POST36_TAIL_PLAN.json",
        )
        checks = {row["check_id"]: row for row in report["checks"]}
        self.assertEqual(report["status"], "revise")
        self.assertFalse(checks["full_synthetic_qualification"]["passed"])
        self.assertTrue(checks["observed_36_trial_provisional_chain"]["passed"])
        self.assertTrue(checks["post36_falsifier_plan"]["passed"])
        self.assertFalse(claim_is_permitted(report, "observed_tail"))
        self.assertFalse(claim_is_permitted(report, "final_connectivity"))


if __name__ == "__main__":
    unittest.main()
