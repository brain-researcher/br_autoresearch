"""Production dispatcher for the seven composite EP12 falsifier controls."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as parquet
from scipy import sparse

from .adaptive_search import validate_development_configuration
from .control_trial import (
    run_control_development_trial,
    verify_development_trial_artifacts,
)
from .falsifier_controls import (
    COUNTABLE_COMPOSITE_CONTROLS,
    FalsifierControlError,
    composite_control_receipt,
    cross_side_alignment_permutation,
    endpoint_branch_inventory,
    implementation_spec,
    leave_one_type_influence,
    make_branch_manifest,
    neuron_influence_curves,
    nuisance_block_columns,
    partner_family_inventory,
)
from .policy import EpisodePolicy, digest_bytes, digest_object
from .runtime import atomic_json


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return normalized[:48] or "branch"


def _fit_branch_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one independent branch in a spawned, one-thread worker process."""

    suite = ProductionFalsifierSuite(
        snapshot_dir=Path(str(payload["snapshot_dir"])),
        policy=EpisodePolicy.load(Path(str(payload["policy_path"]))),
        contract_path=Path(str(payload["contract_path"])),
        contract_provenance=payload.get("contract_provenance"),
    )
    result = suite._run_fit_branch(**dict(payload["branch_arguments"]))
    # The authenticated artifacts and branch manifest remain on scratch.  The
    # parent needs only compact aggregation metadata across the process pipe.
    result["report"] = {
        "wall_seconds": float(result["report"].get("wall_seconds", 0.0)),
        "cpu_seconds": float(result["report"].get("cpu_seconds", 0.0)),
    }
    return result


class ProductionFalsifierSuite:
    """Execute complete branch inventories and return one authenticated receipt."""

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        policy: EpisodePolicy,
        contract_path: Path | None = None,
        contract_provenance: Mapping[str, Any] | None = None,
    ):
        self.snapshot_dir = snapshot_dir.resolve()
        self.policy = policy
        self.contract_path = (
            contract_path.resolve(strict=True)
            if contract_path is not None
            else (Path(__file__).resolve().parents[1] / "SCIENTIFIC_CONTRACT.yaml")
        )
        self.contract_provenance = (
            dict(contract_provenance) if contract_provenance is not None else None
        )
        self.spec = implementation_spec()

    def _worker_count(self, branch_count: int) -> int:
        try:
            allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        except ValueError as error:
            raise FalsifierControlError("SLURM_CPUS_PER_TASK is not an integer") from error
        maximum = int(self.policy.cpu_cores_per_trial_maximum)
        if allocation < 1 or allocation > maximum or maximum > 32:
            raise FalsifierControlError(
                "control branch worker allocation violates the frozen 32-core limit"
            )
        return min(allocation, max(1, int(branch_count)))

    def _inventories(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        vocabulary = json.loads(
            (self.snapshot_dir / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
        )
        hierarchy_path = self.snapshot_dir / "partner_type_hierarchy.json"
        if not hierarchy_path.is_file():
            raise FalsifierControlError("partner hierarchy is required by composite controls")
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        matrix = sparse.load_npz(
            self.snapshot_dir / "development_outgoing_counts.npz"
        ).tocsr()
        column_mass = np.asarray(matrix.sum(axis=0)).ravel()[None, :]
        return (
            partner_family_inventory(hierarchy, vocabulary),
            endpoint_branch_inventory(vocabulary, column_mass),
        )

    def _branch_config(
        self,
        proposal: Mapping[str, Any],
        *,
        index: int,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(proposal["configuration"])
        config.update(dict(overrides or {}))
        config["trial_id"] = f"{proposal['trial_id']}__branch_{index:03d}"
        validate_development_configuration(self.policy, config)
        return config

    def _run_fit_branch(
        self,
        *,
        proposal: Mapping[str, Any],
        branch_root: Path,
        index: int,
        branch_id: str,
        overrides: Mapping[str, Any] | None = None,
        control_spec: Mapping[str, Any] | None = None,
        capacity_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self._branch_config(proposal, index=index, overrides=overrides)
        safe = f"{index:03d}_{_slug(branch_id)}"
        result_root = branch_root / safe
        trial_dir = result_root / config["trial_id"]
        spec = {"branch_id": branch_id, **dict(control_spec or {})}
        report = run_control_development_trial(
            output_dir=trial_dir,
            snapshot_dir=self.snapshot_dir,
            configuration=config,
            control_spec=spec,
            contract_path=self.contract_path,
            contract_provenance=self.contract_provenance,
        )
        config_hash = digest_object(config)
        verified = verify_development_trial_artifacts(
            result_root=result_root,
            configuration=config,
            configuration_hash=config_hash,
        )
        scientific_configuration = {
            key: config[key]
            for key in sorted(config)
            if key not in {"trial_id", "stage"}
        }
        executed_configuration = report.get("configuration", {})
        capacity_signature = {
            "scientific_configuration": scientific_configuration,
            "modeled_partner_vocabulary_sha256": executed_configuration.get(
                "modeled_partner_vocabulary_sha256"
            ),
            "collapsed_partner_count": executed_configuration.get(
                "collapsed_partner_count"
            ),
        }
        complete_rows = report.get("complete_type_records", [])
        common_triplet_capacity = bool(complete_rows) and all(
            set(
                row.get("source_fit_metadata", {})
                .get(side, {})
                .get("parameter_counts", {})
            )
            == {"T", "U", "M"}
            for row in complete_rows
            for side in ("left", "right")
        )
        capacity_ok = (
            common_triplet_capacity
            and (
                capacity_reference is None
                or capacity_signature == dict(capacity_reference)
            )
        )
        transforms = list(report.get("control_transforms", []))
        applicability = bool(
            int(report["aggregate"].get("valid_type_count", 0)) > 0
            and int(report["aggregate"].get("technical_failure_count", -1)) == 0
        )
        conservation = all(
            transform.get("row_totals_preserved", True) is True
            and transform.get("total_mass_preserved", True) is True
            for transform in transforms
            if transform.get("row_mass_preservation_required", True) is not False
        )
        artifacts = {
            "trial_report.json": digest_bytes((trial_dir / "trial_report.json").read_bytes()),
            "manifest.json": digest_bytes((trial_dir / "manifest.json").read_bytes()),
            "result_table.csv": digest_bytes((trial_dir / "result_table.csv").read_bytes()),
            "type_direction_receipts.jsonl": digest_bytes(
                (trial_dir / "type_direction_receipts.jsonl").read_bytes()
            ),
        }
        manifest = make_branch_manifest(
            control_id=str(proposal["registered_falsifier_id"]),
            branch_id=branch_id,
            branch_kind="complete_triplet_fit",
            complete_triplet_sha256=verified["trial_report_sha256"],
            applicability_checks_passed=applicability,
            conservation_checks_passed=conservation,
            capacity_checks_passed=capacity_ok,
            artifacts=artifacts,
            details={
                "configuration_sha256": config_hash,
                "capacity_signature": capacity_signature,
                "control_spec_sha256": report["control_spec_sha256"],
                "transforms": transforms,
            },
        )
        atomic_json(result_root / "branch_manifest.json", manifest)
        return {
            "branch_id": branch_id,
            "manifest": manifest,
            "report": report,
            "verified": verified,
            "capacity_signature": capacity_signature,
            "result_root": result_root,
            "trial_dir": trial_dir,
        }

    def _run_fit_branches(
        self,
        *,
        proposal: Mapping[str, Any],
        branch_root: Path,
        branch_specs: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Run independent fit branches concurrently, preserving plan order."""

        specs = [dict(value) for value in branch_specs]
        if not specs:
            return [], 1
        workers = self._worker_count(len(specs))
        if workers == 1:
            results = [
                self._run_fit_branch(
                    proposal=proposal,
                    branch_root=branch_root,
                    **spec,
                )
                for spec in specs
            ]
        else:
            payloads = [
                {
                    "snapshot_dir": str(self.snapshot_dir),
                    "policy_path": str(self.policy.path),
                    "contract_path": str(self.contract_path),
                    "contract_provenance": self.contract_provenance,
                    "branch_arguments": {
                        "proposal": dict(proposal),
                        "branch_root": branch_root,
                        **spec,
                    },
                }
                for spec in specs
            ]
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as pool:
                futures = [pool.submit(_fit_branch_worker, payload) for payload in payloads]
                results = [future.result() for future in futures]
        for result in results:
            checkpoint_path = Path(result["result_root"]) / "branch_manifest.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint != result["manifest"]:
                raise FalsifierControlError(
                    f"branch checkpoint disagrees with aggregate: {result['branch_id']}"
                )
        return results, workers

    def _analytic_manifest(
        self,
        *,
        control_id: str,
        branch_id: str,
        baseline: Mapping[str, Any],
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        return make_branch_manifest(
            control_id=control_id,
            branch_id=branch_id,
            branch_kind="analytic_from_complete_triplet",
            complete_triplet_sha256=baseline["verified"]["trial_report_sha256"],
            applicability_checks_passed=True,
            conservation_checks_passed=True,
            capacity_checks_passed=True,
            artifacts={
                "baseline_trial_report.json": baseline["verified"][
                    "trial_report_sha256"
                ]
            },
            details=details,
        )

    def execute(
        self,
        *,
        proposal: Mapping[str, Any],
        attempt_dir: Path,
    ) -> dict[str, Any]:
        started_wall = time.perf_counter()
        control_id = str(proposal["registered_falsifier_id"])
        if control_id not in COUNTABLE_COMPOSITE_CONTROLS:
            raise FalsifierControlError(f"not a composite control: {control_id}")
        branch_root = attempt_dir / "control_branches"
        branch_root.mkdir(parents=True, exist_ok=False)
        families: list[dict[str, Any]] = []
        endpoints: list[dict[str, Any]] = []
        if control_id in {
            "nuisance_and_partner_block_ablations",
            "leave_one_development_type_and_partner_family_influence",
        }:
            families, _ = self._inventories()
        elif control_id == "status_endpoint_anatomy_and_missingness_checks":
            _, endpoints = self._inventories()
        index = 1
        baseline = self._run_fit_branch(
            proposal=proposal,
            branch_root=branch_root,
            index=index,
            branch_id="weighted_baseline",
        )
        fit_runs = [baseline]
        max_workers_used = 1
        index += 1
        expected: list[str] = []
        manifests: list[dict[str, Any]] = []

        if control_id == "cross_side_component_alignment_permutation":
            records = [
                {
                    "provider_type": row["provider_type"],
                    "left": row["source_fit_metadata"]["left"][
                        "M_component_partner_contrasts"
                    ],
                    "right": row["source_fit_metadata"]["right"][
                        "M_component_partner_contrasts"
                    ],
                }
                for row in baseline["report"]["complete_type_records"]
            ]
            alignment = cross_side_alignment_permutation(
                records, seed=int(proposal["transform_seed"])
            )
            for branch in alignment["branches"]:
                branch_id = str(branch["branch_id"])
                expected.append(branch_id)
                manifests.append(
                    self._analytic_manifest(
                        control_id=control_id,
                        branch_id=branch_id,
                        baseline=baseline,
                        details=branch,
                    )
                )

        elif control_id == "binary_weighted_vocabulary_and_rank_sensitivities":
            expected.append("weighted_baseline")
            manifests.append(baseline["manifest"])
            branch_specs = [
                {
                    "index": index,
                    "branch_id": "binary",
                    "control_spec": {"matrix_transform": "binary"},
                }
            ]
            index += 1
            for vocabulary in self.policy.grammar_choices["partner_vocabulary"]:
                for rank in self.policy.grammar_choices["rank"]:
                    branch_id = f"vocabulary::{vocabulary}__rank::{rank}"
                    branch_specs.append(
                        {
                            "index": index,
                            "branch_id": branch_id,
                            "overrides": {
                                "partner_vocabulary": vocabulary,
                                "rank": rank,
                            },
                        }
                    )
                    index += 1
            branches, workers = self._run_fit_branches(
                proposal=proposal,
                branch_root=branch_root,
                branch_specs=branch_specs,
            )
            max_workers_used = max(max_workers_used, workers)
            fit_runs.extend(branches)
            for branch in branches:
                expected.append(str(branch["branch_id"]))
                manifests.append(branch["manifest"])

        elif control_id == "nuisance_and_partner_block_ablations":
            width = 22 + (
                0
                if int(proposal["configuration"]["nuisance_spline_df"]) == 0
                else 6 * (int(proposal["configuration"]["nuisance_spline_df"]) - 1)
            )
            blocks = nuisance_block_columns(
                spline_df=int(proposal["configuration"]["nuisance_spline_df"]),
                design_width=width,
            )
            declared = set(map(str, proposal["configuration"]["nuisance"]))
            branch_specs: list[dict[str, Any]] = []
            for block_id in sorted(declared):
                if block_id not in blocks:
                    raise FalsifierControlError(f"declared nuisance block unavailable: {block_id}")
                branch_id = f"nuisance::{block_id}"
                branch_specs.append(
                    {
                        "index": index,
                        "branch_id": branch_id,
                        "control_spec": {"nuisance_ablation_block": block_id},
                    }
                )
                index += 1
            for family in families:
                branch_id = str(family["branch_id"])
                branch_specs.append(
                    {
                        "index": index,
                        "branch_id": branch_id,
                        "control_spec": {
                            "partner_pool_keys": family["raw_partner_keys"]
                        },
                    }
                )
                index += 1
            branches, workers = self._run_fit_branches(
                proposal=proposal,
                branch_root=branch_root,
                branch_specs=branch_specs,
            )
            max_workers_used = max(max_workers_used, workers)
            fit_runs.extend(branches)
            for branch in branches:
                expected.append(str(branch["branch_id"]))
                manifests.append(branch["manifest"])

        elif control_id == "capacity_matched_noise_controls":
            expected.append("weighted_baseline")
            manifests.append(baseline["manifest"])
            noise = self._run_fit_branch(
                proposal=proposal,
                branch_root=branch_root,
                index=index,
                branch_id="pooled_multinomial_noise",
                control_spec={
                    "matrix_transform": "pooled_multinomial_noise",
                    "transform_seed": int(proposal["transform_seed"]),
                },
                capacity_reference=baseline["capacity_signature"],
            )
            fit_runs.append(noise)
            expected.append("pooled_multinomial_noise")
            manifests.append(noise["manifest"])

        elif control_id == "leave_one_development_type_and_partner_family_influence":
            influence = leave_one_type_influence(
                baseline["report"]["complete_type_records"]
            )
            for row in influence["leave_one_type_curve"]:
                branch_id = f"leave_one_type::{row['omitted_provider_type']}"
                expected.append(branch_id)
                manifests.append(
                    self._analytic_manifest(
                        control_id=control_id,
                        branch_id=branch_id,
                        baseline=baseline,
                        details=row,
                    )
                )
            branch_specs = []
            for family in families:
                branch_id = str(family["branch_id"])
                branch_specs.append(
                    {
                        "index": index,
                        "branch_id": branch_id,
                        "control_spec": {
                            "partner_pool_keys": family["raw_partner_keys"]
                        },
                    }
                )
                index += 1
            branches, workers = self._run_fit_branches(
                proposal=proposal,
                branch_root=branch_root,
                branch_specs=branch_specs,
            )
            max_workers_used = max(max_workers_used, workers)
            fit_runs.extend(branches)
            for branch in branches:
                expected.append(str(branch["branch_id"]))
                manifests.append(branch["manifest"])

        elif control_id == "few_type_and_high_strength_neuron_concentration_check":
            type_curve = leave_one_type_influence(
                baseline["report"]["complete_type_records"]
            )
            neuron_curves = neuron_influence_curves(
                baseline["report"]["complete_type_records"]
            )
            analytic = {
                "leave_one_type_curve": type_curve,
                "leave_one_neuron_curve": neuron_curves["leave_one_neuron_curve"],
                "high_strength_cumulative_curve": neuron_curves[
                    "high_strength_cumulative_curve"
                ],
            }
            for branch_id, details in analytic.items():
                expected.append(branch_id)
                manifests.append(
                    self._analytic_manifest(
                        control_id=control_id,
                        branch_id=branch_id,
                        baseline=baseline,
                        details={branch_id: details, "new_concentration_cutoff": None},
                    )
                )

        elif control_id == "status_endpoint_anatomy_and_missingness_checks":
            width = 22 + (
                0
                if int(proposal["configuration"]["nuisance_spline_df"]) == 0
                else 6 * (int(proposal["configuration"]["nuisance_spline_df"]) - 1)
            )
            blocks = nuisance_block_columns(
                spline_df=int(proposal["configuration"]["nuisance_spline_df"]),
                design_width=width,
            )
            branch_specs = []
            for block_id in sorted(blocks):
                branch_id = f"covariate::{block_id}"
                branch_specs.append(
                    {
                        "index": index,
                        "branch_id": branch_id,
                        "control_spec": {"nuisance_ablation_block": block_id},
                    }
                )
                index += 1
            for endpoint in endpoints:
                branch_id = str(endpoint["branch_id"])
                branch_specs.append(
                    {
                        "index": index,
                        "branch_id": branch_id,
                        "control_spec": {
                            "endpoint_pool_bin": endpoint["endpoint_bin"]
                        },
                    }
                )
                index += 1
            branches, workers = self._run_fit_branches(
                proposal=proposal,
                branch_root=branch_root,
                branch_specs=branch_specs,
            )
            max_workers_used = max(max_workers_used, workers)
            fit_runs.extend(branches)
            for branch in branches:
                expected.append(str(branch["branch_id"]))
                manifests.append(branch["manifest"])
        else:  # pragma: no cover - guarded by membership above
            raise FalsifierControlError(f"unhandled composite control: {control_id}")

        receipt = composite_control_receipt(
            control_id=control_id,
            expected_branch_ids=expected,
            branches=manifests,
            implementation_spec_sha256=self.spec["spec_sha256"],
        )
        receipt.update(
            {
                "schema_version": "ep12.post36_control_receipt.v1",
                "implementation": "ProductionFalsifierSuite",
                "transforms": [
                    {
                        "transform": "complete_prespecified_control_branch",
                        "branch_id": branch_id,
                    }
                    for branch_id in expected
                ],
                "applicability_checks_passed": True,
                "conservation_checks_passed": True,
                "capacity_checks_passed": True,
                "max_concurrent_fit_workers": max_workers_used,
                "branch_checkpoints_authenticated": True,
            }
        )
        receipt["receipt_sha256"] = digest_object(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        return {
            "control_receipt": receipt,
            "representative": baseline,
            "branch_root": branch_root,
            "resource_totals": {
                "wall_seconds": time.perf_counter() - started_wall,
                "branch_wall_seconds_sum": sum(
                    float(branch["report"].get("wall_seconds", 0.0))
                    for branch in fit_runs
                ),
                "cpu_seconds": sum(
                    float(branch["report"].get("cpu_seconds", 0.0))
                    for branch in fit_runs
                ),
                "max_concurrent_fit_workers": max_workers_used,
            },
        }
