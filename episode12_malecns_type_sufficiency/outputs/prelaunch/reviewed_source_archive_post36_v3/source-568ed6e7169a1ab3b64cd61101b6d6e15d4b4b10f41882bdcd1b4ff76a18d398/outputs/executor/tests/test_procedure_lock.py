from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.contract_provenance import ACTIVE_EPOCH, PREFIX_EPOCH
from ep12_executor.null_controller import freeze_null_controller_program
from ep12_executor.mode_support_inference import infer_mode_support
from ep12_executor.full_search_runtime import (
    RUNTIME_BENCHMARK_SCHEMA,
    RUNTIME_BENCHMARK_SCOPE,
    RUNTIME_BENCHMARK_ROLE,
    TIMING_CORPUS_SCHEMA,
)
from ep12_executor.policy import EpisodePolicy, canonical_json, digest_object
from ep12_executor.post_lock_runtime_materializer import freeze_resource_plan
from ep12_executor.procedure_lock import (
    ARCHIVE_SCHEMA_VERSION,
    CONTROL_INVENTORY_SCHEMA_VERSION,
    FITTED_NULL_FALSIFIER_ID,
    NOVELTY_AUDIT_FALSIFIER_ID,
    PRE_NULL_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SOURCE_BINDING_SCHEMA_VERSION,
    ProcedureLockError,
    build_procedure_lock,
    verify_procedure_lock,
    write_procedure_lock,
)
from ep12_executor.procedure_lock_dispatch import (
    CANONICAL_RUNTIME_PROVIDER,
    ProcedureDispatchError,
    _default_job_resolver,
    _default_submitter,
    dispatch_procedure_lock,
    freeze_scheduler_job_status,
)
from ep12_executor.resource_authorization import (
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_SCOPE,
    AUTHORIZATION_SOURCE,
    PROJECTION_FIELDS,
)


EPISODE_ROOT = Path(__file__).resolve().parents[3]
PREFIX_CONTRACT_SHA256 = "1" * 64
CONTRACT_SHA256 = "7" * 64
TRANSITION_SEMANTIC_SHA256 = "8" * 64
TRANSITION_FILE_SHA256 = "9" * 64
ATTESTATION_SHA256 = "2" * 64
ROLE_SHA256 = "3" * 64
SCHEDULE_FILE_SHA256 = "4" * 64
LEDGER_SHA256 = "5" * 64
LEDGER_OBJECT_SHA256 = "6" * 64


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def configuration(trial_id: str, stage: str, choice: int) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "stage": stage,
        "synthetic_scientific_choice": choice,
    }


def signature(config: dict[str, object]) -> str:
    return digest_object(
        {key: value for key, value in config.items() if key not in {"trial_id", "stage"}}
    )


def comparison_record(
    trial_id: str,
    stage: str,
    choice: int,
    *,
    eligible: bool,
    score: float,
) -> dict[str, object]:
    config = configuration(trial_id, stage, choice)
    active = stage == "post36_tail"
    return {
        "trial_id": trial_id,
        "configuration": config,
        "configuration_sha256": digest_object(config),
        "scientific_signature": signature(config),
        "source_stage": stage,
        "selection_eligible": eligible,
        "status": "completed",
        "valid_complete_T_U_M_triplet": True,
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "technical_failure_count": 0,
        "objective": {
            "bilateral_floor": score,
            "bilateral": score + 0.01,
            "direction_disagreement": 0.01,
            "types_both_above_margin": 8,
        },
        "result_table_sha256": sha(f"result:{trial_id}"),
        "contract_epoch": ACTIVE_EPOCH if active else PREFIX_EPOCH,
        "scientific_contract_sha256": (
            CONTRACT_SHA256 if active else PREFIX_CONTRACT_SHA256
        ),
        **(
            {
                "contract_transition_semantic_sha256": (
                    TRANSITION_SEMANTIC_SHA256
                ),
                "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            }
            if active
            else {}
        ),
    }


class ProcedureLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_freeze_patch = mock.patch(
            "ep12_executor.procedure_lock_dispatch._source_freeze_identity",
            return_value={
                "manifest_path": "/tmp/ep12-procedure-test-freeze.json",
                "manifest_file_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "file_count": 50,
            },
        )
        self.source_freeze_patch.start()

    def tearDown(self) -> None:
        self.source_freeze_patch.stop()

    def test_dispatch_submitter_ignores_adversarial_sbatch_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="88101\n", stderr="")
        poison = {
            "SBATCH_ARRAY_INX": "0",
            "SBATCH_EXPORT": "ALL",
            "SBATCH_PARTITION": "wrong-partition",
            "SBATCH_WRAP": "touch /tmp/should-not-run",
            "SLURM_EXPORT_ENV": "ALL",
        }
        with mock.patch.dict(os.environ, poison, clear=False), mock.patch(
            "ep12_executor.procedure_lock_dispatch.subprocess.run",
            return_value=completed,
        ) as run:
            self.assertEqual(_default_submitter(["sbatch", "--parsable"]), "88101")
        environment = run.call_args.kwargs["env"]
        self.assertIn("PATH", environment)
        self.assertTrue(
            set(environment)
            <= {"PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE"}
        )
        self.assertFalse(
            any(name.startswith(("SBATCH_", "SLURM_")) for name in environment)
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")

    @staticmethod
    def refresh_pre_manifest(inputs: dict[str, object]) -> None:
        manifest = inputs["pre_null_manifest"]
        manifest["mode_support_inference_sha256"] = inputs[
            "mode_support_result"
        ]["inference_sha256"]
        manifest["control_inventory_sha256"] = digest_object(
            inputs["control_result_inventory"]
        )
        core = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
        manifest["bundle_sha256"] = digest_object(core)

    def fixture(self, *, weak_mode: bool = False) -> dict[str, object]:
        policy = self.policy
        observed_ids = [
            value
            for value in policy.required_falsifiers
            if value
            not in {FITTED_NULL_FALSIFIER_ID, NOVELTY_AUDIT_FALSIFIER_ID}
        ]
        self.assertEqual(len(observed_ids), 9)
        initial_ids = [f"initial_{index:02d}" for index in range(policy.minimum_valid_trials)]
        records = [
            comparison_record(
                trial_id,
                "initial_minimum",
                index,
                eligible=True,
                score=0.30 + index / 10_000,
            )
            for index, trial_id in enumerate(initial_ids)
        ]
        slots: list[dict[str, object]] = []
        for index, falsifier_id in enumerate(observed_ids):
            trial_id = f"control_{index:02d}"
            record = comparison_record(
                trial_id,
                "post36_tail",
                100 + index,
                eligible=False,
                score=0.99,
            )
            records.append(record)
            slots.append(
                self.slot(record, 37 + index, falsifier_id=falsifier_id, eligible=False)
            )
        selected_record = comparison_record(
            "tail_challenger",
            "post36_tail",
            999,
            eligible=True,
            score=0.90,
        )
        records.append(selected_record)
        slots.append(self.slot(selected_record, 46, falsifier_id=None, eligible=True))
        schedule = {
            "schema_version": "ep12.post36_realized_schedule.v2",
            "policy_sha256": policy.policy_hash,
            "plan_sha256": sha("plan"),
            "controller_identity_sha256": sha("controller"),
            "stop_reason": "frozen_patience_reached",
            "stop_valid_trial_ordinal": 46,
            "completed_promotable_search": True,
            "promotion_gates": {
                "minimum_search": True,
                "patience": True,
                "falsifier_fraction": True,
                "all_controls": True,
            },
            "incumbent_trial_id": selected_record["trial_id"],
            "incumbent_configuration_sha256": selected_record["configuration_sha256"],
            "realized_slots": slots,
            "realized_slot_count": len(slots),
            "realized_controller_program_sha256": digest_object(slots),
            "ledger_record_count": 40,
            "ledger_head_hash": sha("ledger-head"),
            "ledger_hash": LEDGER_OBJECT_SHA256,
            "non_trial_required_gates": [
                FITTED_NULL_FALSIFIER_ID,
                NOVELTY_AUDIT_FALSIFIER_ID,
            ],
            "non_trial_required_gates_completed": False,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
            "contract_epoch": ACTIVE_EPOCH,
            "scientific_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
        }
        schedule_manifest = {
            "schema_version": "ep12.post36_realized_schedule_manifest.v2",
            "files": {
                "realized_schedule.json": SCHEDULE_FILE_SHA256,
                "ledger.jsonl": LEDGER_SHA256,
            },
            "ledger_head_hash": schedule["ledger_head_hash"],
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "contract_epoch": ACTIVE_EPOCH,
            "scientific_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
        }
        archive = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "observed_attestation_sha256": ATTESTATION_SHA256,
            "initial_trial_ids": initial_ids,
            "records": records,
            "selected_trial_id": selected_record["trial_id"],
            "selected_configuration_sha256": selected_record[
                "configuration_sha256"
            ],
            "source_evidence": {"synthetic_fixture_sha256": sha("source-evidence")},
            "contract_partition": {
                "initial_prefix": {
                    "contract_epoch": PREFIX_EPOCH,
                    "scientific_contract_sha256": PREFIX_CONTRACT_SHA256,
                    "trial_ids": initial_ids,
                },
                "post36_tail": {
                    "contract_epoch": ACTIVE_EPOCH,
                    "scientific_contract_sha256": CONTRACT_SHA256,
                    "trial_ids": [str(slot["trial_id"]) for slot in slots],
                },
                "contract_transition_semantic_sha256": (
                    TRANSITION_SEMANTIC_SHA256
                ),
                "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            },
            "selected_source_contract_epoch": selected_record["contract_epoch"],
            "selected_source_contract_sha256": selected_record[
                "scientific_contract_sha256"
            ],
            "complete": True,
            "final_connectivity_accessed": False,
        }
        n = 16
        offset = np.linspace(-0.002, 0.002, n)
        gain = np.column_stack((0.08 + offset, 0.075 - offset))
        if weak_mode:
            gain = np.column_stack(
                (np.array([0.0, 0.05] * 8), np.array([0.05, 0.0] * 8))
            )
        mode = infer_mode_support(
            comparison_trial_id=str(selected_record["trial_id"]),
            configuration_sha256=str(selected_record["configuration_sha256"]),
            contract_sha256=CONTRACT_SHA256,
            result_table_sha256=str(selected_record["result_table_sha256"]),
            meaningful_margin=policy.meaningful_margin,
            prevalence_parameter_rank_definition=(
                "synthetic_fitted_component_parameter_rank"
            ),
            whole_provider_type_ids=[f"type_{index:02d}" for index in range(n)],
            prediction_gain_by_type_direction=gain,
            component_symmetric_kl_by_type_direction=np.ones((n, 2, 1)) * 0.7,
            matched_contrast_cosine_by_type=np.ones((n, 1)) * 0.8,
            partner_contrast_by_type_direction=np.ones((n, 2, 1)) * 0.5,
            effective_memberships_by_direction_component=np.tile(
                np.asarray([[[12, 11], [10, 13]]]), (n, 1, 1)
            ),
            fitted_parameter_ranks_by_direction_component=np.tile(
                np.asarray([[[3, 3], [3, 3]]]), (n, 1, 1)
            ),
        )
        mode.update(
            {
                "selected_source_contract_epoch": selected_record["contract_epoch"],
                "selected_source_contract_sha256": selected_record[
                    "scientific_contract_sha256"
                ],
                "refit_contract_epoch": ACTIVE_EPOCH,
                "contract_transition_semantic_sha256": (
                    TRANSITION_SEMANTIC_SHA256
                ),
                "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            }
        )
        mode["inference_sha256"] = digest_object(
            {key: value for key, value in mode.items() if key != "inference_sha256"}
        )
        controls: list[dict[str, object]] = []
        for falsifier_id in policy.required_falsifiers:
            if falsifier_id == FITTED_NULL_FALSIFIER_ID:
                controls.append(
                    {
                        "falsifier_id": falsifier_id,
                        "status": "pending_full_search_null",
                        "applicable": None,
                        "control_passed": None,
                        "evidence_sha256s": [sha("registered:fitted-null")],
                        "final_connectivity_accessed": False,
                    }
                )
            elif falsifier_id == NOVELTY_AUDIT_FALSIFIER_ID:
                controls.append(
                    {
                        "falsifier_id": falsifier_id,
                        "status": "pending_post_result_novelty_audit",
                        "applicable": None,
                        "control_passed": None,
                        "evidence_sha256s": [sha("registered:novelty")],
                        "final_connectivity_accessed": False,
                    }
                )
            else:
                evidence = sorted(
                    str(slot["control_receipt_sha256"])
                    for slot in slots
                    if slot["registered_falsifier_id"] == falsifier_id
                )
                controls.append(
                    {
                        "falsifier_id": falsifier_id,
                        "status": "passed",
                        "applicable": True,
                        "control_passed": True,
                        "evidence_sha256s": evidence,
                        "final_connectivity_accessed": False,
                    }
                )
        inventory = {
            "schema_version": CONTROL_INVENTORY_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "controls": controls,
            "pending_registrations": {},
            "final_connectivity_accessed": False,
        }
        generator_selection = {
            "schema_version": "ep12.fitted_null_generator_selection.v1",
            "complete": True,
            "development_only": True,
            "selected_generator_by_type_side": {
                "type_a|left": "T",
                "type_a|right": "U",
            },
            "selected_source_contract_epoch": selected_record["contract_epoch"],
            "selected_source_contract_sha256": selected_record[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "refit_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            "final_connectivity_accessed": False,
        }
        plan = json.loads(
            (EPISODE_ROOT / "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json").read_text(
                encoding="utf-8"
            )
        )
        controller_program = freeze_null_controller_program(
            policy=policy,
            coverage_configurations=coverage_configurations(policy),
            post36_tail_plan=plan,
            countable_falsifier_ids=plan["required_falsifier_classification"][
                "countable_observed_trial_ids"
            ],
            non_trial_required_falsifier_ids=[
                row["falsifier_id"] for row in plan["non_trial_required_gates"]
            ],
        )
        inventory["pending_registrations"][FITTED_NULL_FALSIFIER_ID] = {
            "status": "pending_full_search_null",
            "null_controller_program_sha256": controller_program["program_sha256"],
            "observed_schedule_replayed": False,
        }
        controller_file_sha256 = sha("controller-program-file")
        archive_sha256 = digest_object(archive)
        generator_selection_sha256 = digest_object(generator_selection)
        callback_data_sha256 = sha("callback-data")
        source_core = {
            "schema_version": RUNTIME_SOURCE_BINDING_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "prefix_contract_sha256": PREFIX_CONTRACT_SHA256,
            "active_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            "selected_source_contract_epoch": selected_record["contract_epoch"],
            "selected_source_contract_sha256": selected_record[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "selected_trial_id": selected_record["trial_id"],
            "selected_configuration_sha256": selected_record[
                "configuration_sha256"
            ],
            "comparison_archive_sha256": archive_sha256,
            "null_generator_selection_sha256": generator_selection_sha256,
            "null_controller_program_sha256": controller_program["program_sha256"],
            "callback_data_sha256": callback_data_sha256,
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        source_binding = {
            **source_core,
            "source_binding_sha256": digest_object(source_core),
        }
        pre_core = {
            "schema_version": PRE_NULL_MANIFEST_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "prefix_contract_sha256": PREFIX_CONTRACT_SHA256,
            "active_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            "selected_source_contract_epoch": selected_record["contract_epoch"],
            "selected_source_contract_sha256": selected_record[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "selected_trial_id": selected_record["trial_id"],
            "selected_configuration_sha256": selected_record[
                "configuration_sha256"
            ],
            "comparison_archive_sha256": archive_sha256,
            "mode_support_inference_sha256": mode["inference_sha256"],
            "control_inventory_sha256": digest_object(inventory),
            "generator_selection_sha256": generator_selection_sha256,
            "null_controller_program_sha256": controller_program["program_sha256"],
            "null_controller_program_file_sha256": controller_file_sha256,
            "runtime_source_binding_sha256": source_binding[
                "source_binding_sha256"
            ],
            "callback_data_sha256": callback_data_sha256,
            "model_state_sha256": sha("model-state"),
            "snapshot_binding_sha256": sha("snapshot-binding"),
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        pre_manifest = {**pre_core, "bundle_sha256": digest_object(pre_core)}
        return {
            "policy": policy,
            "prefix_contract_sha256": PREFIX_CONTRACT_SHA256,
            "active_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": TRANSITION_SEMANTIC_SHA256,
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            "observed_attestation_sha256": ATTESTATION_SHA256,
            "development_role_manifest_sha256": ROLE_SHA256,
            "comparison_archive": archive,
            "mode_support_result": mode,
            "realized_schedule": schedule,
            "realized_schedule_manifest": schedule_manifest,
            "realized_schedule_file_sha256": SCHEDULE_FILE_SHA256,
            "realized_ledger_file_sha256": LEDGER_SHA256,
            "realized_ledger_object_sha256": LEDGER_OBJECT_SHA256,
            "control_result_inventory": inventory,
            "null_generator_selection": generator_selection,
            "null_controller_program": controller_program,
            "null_controller_program_file_sha256": controller_file_sha256,
            "pre_null_manifest": pre_manifest,
            "pre_null_manifest_file_sha256": sha("pre-null-manifest-file"),
            "runtime_source_binding": source_binding,
        }

    def slot(
        self,
        record: dict[str, object],
        ordinal: int,
        *,
        falsifier_id: str | None,
        eligible: bool,
    ) -> dict[str, object]:
        trial_id = str(record["trial_id"])
        return {
            "trial_id": trial_id,
            "valid_trial_ordinal": ordinal,
            "slot_sha256": sha(f"slot:{trial_id}"),
            "slot_kind": "promotable_challenger" if eligible else "mandatory_falsifier",
            "configuration": record["configuration"],
            "configuration_sha256": record["configuration_sha256"],
            "scientific_signature": record["scientific_signature"],
            "proposal_context_hash": sha(f"context:{trial_id}"),
            "lineage": {"parent_trial_id": "initial_35"},
            "registered_falsifier_id": falsifier_id,
            "promotion_eligible": eligible,
            "improved": eligible,
            "disposition": "promoted" if eligible else "control_complete",
            "control_receipt_sha256": sha(f"control:{trial_id}"),
            "durable_bundle_manifest_sha256": sha(f"bundle:{trial_id}"),
            "contract_epoch": ACTIVE_EPOCH,
            "scientific_contract_sha256": CONTRACT_SHA256,
            "contract_transition_semantic_sha256": (
                TRANSITION_SEMANTIC_SHA256
            ),
            "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
        }

    def resource_plan(self) -> dict[str, object]:
        plan = json.loads(
            (EPISODE_ROOT / "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json")
            .read_text(encoding="utf-8")
        )
        controller = freeze_null_controller_program(
            policy=self.policy,
            coverage_configurations=coverage_configurations(self.policy),
            post36_tail_plan=plan,
            countable_falsifier_ids=plan["required_falsifier_classification"][
                "countable_observed_trial_ids"
            ],
            non_trial_required_falsifier_ids=[
                row["falsifier_id"] for row in plan["non_trial_required_gates"]
            ],
        )
        typed = [f"typed::partner_{index:02d}" for index in range(36)]
        endpoints = [
            "untyped_status::Other::unknown",
            "untyped_status::Traced::complete",
            "untyped_status::Other::orphan",
            "untyped_status::Assign::candidate",
            "missing_annotation",
        ]
        vocabulary = typed + endpoints
        hierarchy = {
            f"partner_{index:02d}": {
                "subclass": f"subclass_{index:02d}",
                "class": f"class_{min(index, 34):02d}",
            }
            for index in range(36)
        }
        benchmark_core = {
            "schema_version": RUNTIME_BENCHMARK_SCHEMA,
            "method": "production_fitted_null_sampler_and_T_U_M_callback_base_fit",
            "worker_count": 32,
            "benchmark_replicate_indices": [0, 1, 2],
            "benchmark_replicate_ids": ["null_001", "null_002", "null_003"],
            "benchmark_replicate_seeds_sha256": sha("benchmark-seeds"),
            "max_timing_trial_id": "coverage_017",
            "max_timing_configuration_sha256": sha("max-configuration"),
            "design_sha256": sha("design"),
            "generator_manifest_sha256": sha("generators"),
            "expansion_plan_sha256": sha("expansion"),
            "callback_identity_sha256": sha("callback"),
            "elapsed_wall_seconds": [10.0, 12.0, 11.0],
            "measured_parallel_mean_wall_seconds": 11.0,
            "measured_parallel_max_wall_seconds": 12.0,
            "complete_T_U_M_branches_measured": 3,
            "benchmark_scope": RUNTIME_BENCHMARK_SCOPE,
            "benchmark_role": RUNTIME_BENCHMARK_ROLE,
            "projection_is_upper_bound": False,
            "observed_timing_corpus_sha256": sha("timing-corpus"),
            "observed_timing_corpus_trial_count": 36,
            "scientific_scores_or_objectives_persisted": False,
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        benchmark = {
            **benchmark_core,
            "receipt_sha256": digest_object(benchmark_core),
        }
        timing = {
            "method": (
                "production_parallel_benchmark_with_authenticated_"
                "observed_timing_corpus_maxima"
            ),
            "observed_timing_corpus_schema": TIMING_CORPUS_SCHEMA,
            "observed_timing_corpus_trial_count": 36,
            "observed_timing_corpus_sha256": sha("timing-corpus"),
            "observed_attestation_sha256": sha("observed-attestation"),
            "observed_attestation_manifest_file_sha256": sha("attestation-file"),
            "observed_serial_mean_wall_seconds": 10.0,
            "observed_serial_max_wall_seconds": 20.0,
            "observed_serial_mean_cpu_seconds_per_branch": 4.5,
            "observed_serial_max_cpu_seconds_per_branch": 9.0,
            "max_timing_trial_id": "coverage_017",
            "max_busy_timing_trial_id": "coverage_017",
            "legacy_cost_projection_file_sha256": sha("legacy-cost"),
            "legacy_cost_projection_used_for_projection": False,
            "max_timing_report_file_sha256": sha("max-timing"),
            "max_busy_timing_report_file_sha256": sha("max-timing"),
            "parallelism_scope": (
                "provider_types_within_one_complete_T_U_M_branch"
            ),
            "control_branch_execution": "sequential",
            "scientific_outcomes_persisted": False,
            "runtime_benchmark_receipt": benchmark,
        }
        return freeze_resource_plan(
            policy=self.policy,
            controller_program=controller,
            raw_vocabulary=vocabulary,
            raw_counts=np.ones((1, len(vocabulary)), dtype=np.int64),
            partner_hierarchy=hierarchy,
            covariates_by_spline_df={
                0: np.zeros((1, 22)),
                3: np.zeros((1, 34)),
                5: np.zeros((1, 46)),
            },
            timing_basis=timing,
        )

    def runtime_bundle(
        self, root: Path, lock: dict[str, object], lock_file: Path
    ) -> tuple[Path, str, dict[str, object]]:
        bundle = root / "runtime_bundle"
        bundle.mkdir()
        resource_plan = self.resource_plan()
        resource_plan_bytes = (
            canonical_json(resource_plan) + "\n"
        ).encode("utf-8")
        (bundle / "resource_plan.json").write_bytes(resource_plan_bytes)
        lock_file_sha256 = hashlib.sha256(lock_file.read_bytes()).hexdigest()
        manifest_core = {
            "schema_version": "ep12.synthetic_runtime_bundle_for_dispatch_test.v1",
            "procedure_lock_sha256": sha("runtime-procedure-lock"),
            "upstream_procedure_lock_sha256": lock["procedure_lock_sha256"],
            "upstream_procedure_lock_file_sha256": lock_file_sha256,
            "resource_plan_sha256": resource_plan["plan_sha256"],
            "files": {
                "resource_plan.json": hashlib.sha256(
                    resource_plan_bytes
                ).hexdigest(),
            },
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        manifest = {
            **manifest_core,
            "bundle_sha256": digest_object(manifest_core),
        }
        manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
        (bundle / "runtime_bundle.json").write_text(
            manifest_bytes.decode("utf-8"),
            encoding="utf-8",
        )
        pending_dir = root / "pending"
        pending_dir.mkdir()
        pending_core = {
            "schema_version": "ep12.resource_authorization_pending.v1",
            "action": "resource_authorization_pending",
            "reason": (
                "explicit_exact_resource_authorization_required_before_"
                "full_null_dispatch"
            ),
            "auto_launch_full_null": False,
            "runtime_bundle_sha256": manifest["bundle_sha256"],
            "runtime_bundle_manifest_file_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "resource_plan_sha256": resource_plan["plan_sha256"],
            "resource_plan_file_sha256": hashlib.sha256(
                resource_plan_bytes
            ).hexdigest(),
            "scratch_stage_receipt_sha256": sha("scratch-stage"),
            "procedure_lock_sha256": lock["procedure_lock_sha256"],
            "procedure_lock_file_sha256": lock_file_sha256,
            **{field: resource_plan[field] for field in PROJECTION_FIELDS},
            "program_search_count": resource_plan["program_search_count"],
            "synthetic_null_search_count": resource_plan[
                "synthetic_null_search_count"
            ],
            "full_search_null_submitted": False,
            "jobs_submitted": [],
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        pending = {
            **pending_core,
            "receipt_sha256": digest_object(pending_core),
        }
        pending_path = pending_dir / "resource_authorization_pending.json"
        pending_bytes = (canonical_json(pending) + "\n").encode("utf-8")
        pending_path.write_bytes(pending_bytes)
        authorization_dir = root / "resource_authorization"
        authorization_dir.mkdir()
        authorization_core = {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "episode_id": "ep12",
            "action": "authorize_full_search_null_dispatch",
            "authorization_source": AUTHORIZATION_SOURCE,
            "scope": AUTHORIZATION_SCOPE,
            "allow_full_search_null_dispatch": True,
            "runtime_bundle_sha256": manifest["bundle_sha256"],
            "runtime_bundle_manifest_file_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
            "resource_plan_sha256": resource_plan["plan_sha256"],
            "resource_plan_file_sha256": hashlib.sha256(
                resource_plan_bytes
            ).hexdigest(),
            "pending_receipt_sha256": pending["receipt_sha256"],
            "pending_receipt_file_sha256": hashlib.sha256(
                pending_bytes
            ).hexdigest(),
            "procedure_lock_sha256": lock["procedure_lock_sha256"],
            "procedure_lock_file_sha256": lock_file_sha256,
            **{field: resource_plan[field] for field in PROJECTION_FIELDS},
            "program_search_count": resource_plan["program_search_count"],
            "synthetic_null_search_count": resource_plan[
                "synthetic_null_search_count"
            ],
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        authorization = {
            **authorization_core,
            "authorization_sha256": digest_object(authorization_core),
        }
        authorization_path = authorization_dir / "full_search_null_authorization.json"
        authorization_bytes = (
            canonical_json(authorization) + "\n"
        ).encode("utf-8")
        authorization_path.write_bytes(authorization_bytes)
        return bundle, str(manifest["bundle_sha256"]), {
            "resource_authorization_path": authorization_path,
            "resource_authorization_file_sha256": hashlib.sha256(
                authorization_bytes
            ).hexdigest(),
            "resource_authorization_pending_path": pending_path,
            "source_freeze_manifest": root / "synthetic_source_freeze.json",
            "source_freeze_manifest_sha256": "a" * 64,
        }

    def test_supported_mode_locks_exact_winner_and_allows_null(self) -> None:
        inputs = self.fixture()
        lock = build_procedure_lock(**inputs)
        verify_procedure_lock(lock, policy=self.policy)
        self.assertTrue(lock["procedure_locked"])
        self.assertTrue(lock["scientific_support"])
        self.assertTrue(lock["null_eligible"])
        self.assertEqual(lock["selected_comparison"]["trial_id"], "tail_challenger")
        self.assertEqual(
            len(lock["required_control_results"]),
            len(self.policy.required_falsifiers),
        )
        self.assertFalse(lock["final_connectivity_accessed"])
        with tempfile.TemporaryDirectory() as directory:
            first = write_procedure_lock(Path(directory), lock)
            (Path(directory) / "manifest.json").unlink()
            second = write_procedure_lock(Path(directory), lock)
            self.assertEqual(first, second)

    def test_weak_support_is_interpretive_and_still_requires_full_null(self) -> None:
        lock = build_procedure_lock(**self.fixture(weak_mode=True))
        self.assertTrue(lock["procedure_locked"])
        self.assertFalse(lock["scientific_support"])
        self.assertTrue(lock["null_eligible"])
        self.assertEqual(lock["scientific_support_state"], "unresolved")
        self.assertIn(
            "simultaneous_mode_support_not_established",
            lock["unresolved_reasons"],
        )

    def test_missing_control_fails_closed_without_inventing_a_result(self) -> None:
        inputs = self.fixture()
        inventory = inputs["control_result_inventory"]
        inventory["controls"].pop(3)
        self.refresh_pre_manifest(inputs)
        lock = build_procedure_lock(**inputs)
        self.assertFalse(lock["procedure_locked"])
        self.assertFalse(lock["null_eligible"])
        self.assertEqual(len(lock["missing_required_control_results"]), 1)
        missing = lock["missing_required_control_results"][0]
        normalized = {
            record["falsifier_id"]: record
            for record in lock["required_control_results"]
        }
        self.assertEqual(normalized[missing]["status"], "missing")

    def test_missing_mode_diagnostic_still_requires_the_full_null(self) -> None:
        inputs = self.fixture()
        selected = inputs["comparison_archive"]["records"][-1]
        inputs["mode_support_result"] = infer_mode_support(
            comparison_trial_id=str(selected["trial_id"]),
            configuration_sha256=str(selected["configuration_sha256"]),
            contract_sha256=CONTRACT_SHA256,
            result_table_sha256=str(selected["result_table_sha256"]),
            meaningful_margin=self.policy.meaningful_margin,
            prevalence_parameter_rank_definition="frozen_but_unavailable",
            whole_provider_type_ids=["type_a", "type_b"],
            prediction_gain_by_type_direction=None,
            component_symmetric_kl_by_type_direction=None,
            matched_contrast_cosine_by_type=None,
            partner_contrast_by_type_direction=None,
            effective_memberships_by_direction_component=None,
            fitted_parameter_ranks_by_direction_component=None,
        )
        inputs["mode_support_result"].update(
            {
                "selected_source_contract_epoch": selected["contract_epoch"],
                "selected_source_contract_sha256": selected[
                    "scientific_contract_sha256"
                ],
                "refit_contract_epoch": ACTIVE_EPOCH,
                "contract_transition_semantic_sha256": (
                    TRANSITION_SEMANTIC_SHA256
                ),
                "contract_transition_file_sha256": TRANSITION_FILE_SHA256,
            }
        )
        inputs["mode_support_result"]["inference_sha256"] = digest_object(
            {
                key: value
                for key, value in inputs["mode_support_result"].items()
                if key != "inference_sha256"
            }
        )
        self.refresh_pre_manifest(inputs)
        lock = build_procedure_lock(**inputs)
        self.assertTrue(lock["procedure_locked"])
        self.assertTrue(lock["null_eligible"])
        self.assertFalse(lock["scientific_support"])
        self.assertIn("mode_support_inputs_incomplete", lock["unresolved_reasons"])

    def test_schedule_and_selected_result_tampering_fail_closed(self) -> None:
        inputs = self.fixture()
        inputs["realized_schedule"]["realized_slots"][0]["promotion_eligible"] = True
        with self.assertRaisesRegex(ProcedureLockError, "controller program digest"):
            build_procedure_lock(**inputs)
        inputs = self.fixture()
        inputs["mode_support_result"]["configuration_sha256"] = "f" * 64
        with self.assertRaisesRegex(ProcedureLockError, "digest mismatch"):
            build_procedure_lock(**inputs)

    def test_omitted_comparison_and_lock_tamper_fail_closed(self) -> None:
        inputs = self.fixture()
        inputs["comparison_archive"]["records"].pop()
        with self.assertRaisesRegex(ProcedureLockError, "exactly cover"):
            build_procedure_lock(**inputs)
        lock = build_procedure_lock(**self.fixture())
        lock["null_eligible"] = False
        with self.assertRaisesRegex(ProcedureLockError, "digest mismatch"):
            verify_procedure_lock(lock, policy=self.policy)

    def test_eligible_lock_submits_exact_array_and_afterok_aggregate_once(self) -> None:
        lock = build_procedure_lock(**self.fixture())
        commands: list[list[str]] = []

        def submit(command: object) -> str:
            commands.append(list(command))
            return "41001;testcluster" if len(commands) == 1 else "41002"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_dir = root / "lock"
            write_procedure_lock(lock_dir, lock)
            script = root / "run_full_search_null.sbatch"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            main_wrapper_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
            aggregate_script = root / "run_full_search_null_aggregate.sbatch"
            aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
            bundle, bundle_sha256, authorization = self.runtime_bundle(
                root, lock, lock_dir / "procedure_lock.json"
            )
            pending_path = authorization["resource_authorization_pending_path"]
            pending_before = pending_path.read_bytes()
            dispatch_dir = root / "dispatch"
            receipt = dispatch_procedure_lock(
                lock=lock,
                policy=self.policy,
                output_dir=dispatch_dir,
                procedure_lock_path=lock_dir / "procedure_lock.json",
                full_search_null_script=script,
                full_search_null_aggregate_script=aggregate_script,
                runtime_provider=CANONICAL_RUNTIME_PROVIDER,
                runtime_bundle=bundle,
                runtime_bundle_sha256=bundle_sha256,
                full_null_work_root=root / "scratch" / "full-null",
                full_null_durable_root=root / "durable" / "full-null",
                submitter=submit,
                job_resolver=lambda _: None,
                **authorization,
            )
            (dispatch_dir / "manifest.json").unlink()
            replay = dispatch_procedure_lock(
                lock=lock,
                policy=self.policy,
                output_dir=dispatch_dir,
                procedure_lock_path=lock_dir / "procedure_lock.json",
                full_search_null_script=script,
                full_search_null_aggregate_script=aggregate_script,
                runtime_provider=CANONICAL_RUNTIME_PROVIDER,
                runtime_bundle=bundle,
                runtime_bundle_sha256=bundle_sha256,
                full_null_work_root=root / "scratch" / "full-null",
                full_null_durable_root=root / "durable" / "full-null",
                submitter=lambda _: self.fail("idempotent replay resubmitted a job"),
                job_resolver=lambda _: self.fail("completed replay queried scheduler"),
                **authorization,
            )
            self.assertTrue((dispatch_dir / "manifest.json").is_file())
            progress = json.loads(
                (dispatch_dir / "dispatch_progress.json").read_text(encoding="utf-8")
            )
            release = json.loads(
                (pending_path.parent / "resource_authorization_release.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(pending_path.read_bytes(), pending_before)
        self.assertEqual(receipt, replay)
        self.assertEqual(progress["launch"], receipt["launch"])
        self.assertEqual(progress["launch_sha256"], receipt["launch_sha256"])
        self.assertTrue(
            {
                "runtime_provider", "full_search_null_script",
                "full_search_null_script_sha256", "runtime_bundle",
                "full_search_null_aggregate_script",
                "full_search_null_aggregate_script_sha256",
                "runtime_bundle_sha256", "procedure_lock_path",
                "procedure_lock_sha256", "procedure_lock_file_sha256",
                "full_null_work_root", "full_null_durable_root", "array_spec",
                "aggregate_resources",
                "aggregate_main_wrapper_sha256_export",
            }
            <= set(progress["launch"])
        )
        expected_main_export = (
            f"EP12_FULL_NULL_MAIN_WRAPPER_SHA256={main_wrapper_sha256}"
        )
        self.assertEqual(
            receipt["launch"]["aggregate_main_wrapper_sha256_export"],
            expected_main_export,
        )
        self.assertEqual(
            receipt["jobs_submitted"][1][
                "aggregate_main_wrapper_sha256_export"
            ],
            expected_main_export,
        )
        self.assertEqual(receipt["action"], "full_search_null_submitted")
        self.assertEqual(len(commands), 2)
        self.assertIn("--array=0-98", commands[0])
        self.assertTrue(
            any("EP12_FULL_NULL_BUNDLE_SHA256=" in value for value in commands[0])
        )
        self.assertEqual(commands[0][-1], "replicate")
        self.assertIn("--dependency=afterok:41001", commands[1])
        self.assertIn("--cpus-per-task=1", commands[1])
        self.assertIn("--mem=32G", commands[1])
        self.assertIn("--time=12:00:00", commands[1])
        self.assertIn("--no-requeue", commands[1])
        aggregate_export = next(
            value for value in commands[1] if value.startswith("--export=")
        )
        self.assertIn(expected_main_export, aggregate_export)
        replicate_export = next(
            value for value in commands[0] if value.startswith("--export=")
        )
        self.assertNotIn("EP12_FULL_NULL_MAIN_WRAPPER_SHA256=", replicate_export)
        self.assertIn("run_full_search_null_aggregate.sbatch", commands[1][-2])
        self.assertEqual(commands[1][-1], "aggregate")
        release_core = {
            key: value for key, value in release.items() if key != "release_sha256"
        }
        self.assertEqual(release["release_sha256"], digest_object(release_core))
        self.assertEqual(release["dispatch_sha256"], receipt["dispatch_sha256"])
        self.assertEqual(release["dispatch_job_ids"], ["41001", "41002"])
        self.assertTrue(release["pending_receipt_retained"])
        self.assertEqual(
            release["resource_plan_sha256"], receipt["resource_plan_sha256"]
        )
        self.assertFalse(receipt["episode_terminal_emitted"])

    def test_eligible_dispatch_rejects_missing_or_projection_mismatched_authorization(
        self,
    ) -> None:
        lock = build_procedure_lock(**self.fixture())
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_dir = root / "lock"
            write_procedure_lock(lock_dir, lock)
            script = root / "run_full_search_null.sbatch"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            aggregate_script = root / "run_full_search_null_aggregate.sbatch"
            aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
            bundle, bundle_sha256, authorization = self.runtime_bundle(
                root, lock, lock_dir / "procedure_lock.json"
            )
            arguments = {
                "lock": lock,
                "policy": self.policy,
                "output_dir": root / "dispatch",
                "procedure_lock_path": lock_dir / "procedure_lock.json",
                "full_search_null_script": script,
                "full_search_null_aggregate_script": aggregate_script,
                "runtime_provider": CANONICAL_RUNTIME_PROVIDER,
                "runtime_bundle": bundle,
                "runtime_bundle_sha256": bundle_sha256,
                "full_null_work_root": root / "scratch" / "full-null",
                "full_null_durable_root": root / "durable" / "full-null",
                "submitter": lambda command: commands.append(list(command)) or "1",
                "job_resolver": lambda _: None,
            }
            with self.assertRaisesRegex(
                ProcedureDispatchError, "explicit exact resource authorization"
            ):
                dispatch_procedure_lock(**arguments)

            authorization_path = authorization["resource_authorization_path"]
            changed = json.loads(authorization_path.read_text(encoding="utf-8"))
            field = "projected_maximum_program_allocated_core_hours"
            changed[field] = changed[field] + 1.0
            changed_core = {
                key: value
                for key, value in changed.items()
                if key != "authorization_sha256"
            }
            changed["authorization_sha256"] = digest_object(changed_core)
            changed_bytes = (canonical_json(changed) + "\n").encode("utf-8")
            authorization_path.write_bytes(changed_bytes)
            changed_authorization = {
                **authorization,
                "resource_authorization_file_sha256": hashlib.sha256(
                    changed_bytes
                ).hexdigest(),
            }
            with self.assertRaisesRegex(
                ProcedureDispatchError, "exact generated resource plan"
            ):
                dispatch_procedure_lock(**arguments, **changed_authorization)
        self.assertEqual(commands, [])

    def test_aggregate_lost_ack_waits_for_visibility_without_resubmission(self) -> None:
        lock = build_procedure_lock(**self.fixture())
        commands: list[list[str]] = []
        accepted: dict[str, str] = {}
        recovery_visible = False
        interrupted = False

        def submit(command: object) -> str:
            nonlocal interrupted
            commands.append(list(command))
            if len(commands) == 1:
                return "42001"
            if not interrupted:
                interrupted = True
                name = next(
                    value.split("=", 1)[1]
                    for value in commands[-1]
                    if value.startswith("--job-name=")
                )
                accepted[name] = "42002"
                raise RuntimeError("synthetic aggregate submission interruption")
            self.fail("uncertain aggregate was submitted twice")

        def resolve(name: str) -> str | None:
            return accepted.get(name) if recovery_visible else None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_dir = root / "lock"
            write_procedure_lock(lock_dir, lock)
            script = root / "run_full_search_null.sbatch"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            aggregate_script = root / "run_full_search_null_aggregate.sbatch"
            aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
            bundle, bundle_sha256, authorization = self.runtime_bundle(
                root, lock, lock_dir / "procedure_lock.json"
            )
            arguments = {
                "lock": lock,
                "policy": self.policy,
                "output_dir": root / "dispatch",
                "procedure_lock_path": lock_dir / "procedure_lock.json",
                "full_search_null_script": script,
                "full_search_null_aggregate_script": aggregate_script,
                "runtime_provider": CANONICAL_RUNTIME_PROVIDER,
                "runtime_bundle": bundle,
                "runtime_bundle_sha256": bundle_sha256,
                "full_null_work_root": root / "scratch" / "full-null",
                "full_null_durable_root": root / "durable" / "full-null",
                "submitter": submit,
                "job_resolver": resolve,
                "job_status_resolver": lambda job_id, name: (
                    freeze_scheduler_job_status(
                        job_id=job_id,
                        job_name=name,
                        parent_state="PENDING",
                    )
                ),
                **authorization,
            }
            with self.assertRaisesRegex(RuntimeError, "synthetic aggregate"):
                dispatch_procedure_lock(**arguments)
            progress = json.loads(
                (root / "dispatch/dispatch_progress.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(progress["state"], "aggregate_submission_uncertain")
            self.assertEqual(progress["array_job_id"], "42001")
            self.assertIsNone(progress["aggregate_job_id"])
            with self.assertRaisesRegex(
                ProcedureDispatchError, "aggregate submission remains uncertain"
            ):
                dispatch_procedure_lock(**arguments)
            self.assertEqual(len(commands), 2)
            recovery_visible = True
            receipt = dispatch_procedure_lock(**arguments)
        self.assertEqual(receipt["jobs_submitted"][0]["job_id"], "42001")
        self.assertEqual(receipt["jobs_submitted"][1]["job_id"], "42002")
        self.assertEqual(sum("--array=0-98" in command for command in commands), 1)
        self.assertEqual(len(commands), 2)

    def test_scheduler_recovery_names_prevent_duplicate_array_and_aggregate(self) -> None:
        lock = build_procedure_lock(**self.fixture())
        for interrupted_role in ("replicate", "aggregate"):
            with self.subTest(interrupted_role=interrupted_role), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lock_dir = root / "lock"
                write_procedure_lock(lock_dir, lock)
                script = root / "run_full_search_null.sbatch"
                script.write_text("#!/bin/bash\n", encoding="utf-8")
                aggregate_script = root / "run_full_search_null_aggregate.sbatch"
                aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
                bundle, bundle_sha256, authorization = self.runtime_bundle(
                    root, lock, lock_dir / "procedure_lock.json"
                )
                accepted: dict[str, str] = {}
                commands: list[list[str]] = []
                interrupted = False
                recovery_visible = False

                def submit(command: object) -> str:
                    nonlocal interrupted
                    normalized = list(command)
                    commands.append(normalized)
                    role = str(normalized[-1])
                    progress = json.loads(
                        (root / "dispatch/dispatch_progress.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        progress["state"],
                        (
                            "array_submission_uncertain"
                            if role == "replicate"
                            else "aggregate_submission_uncertain"
                        ),
                    )
                    name = next(
                        value.split("=", 1)[1]
                        for value in normalized
                        if value.startswith("--job-name=")
                    )
                    job_id = "43001" if role == "replicate" else "43002"
                    accepted[name] = job_id
                    if role == interrupted_role and not interrupted:
                        interrupted = True
                        raise RuntimeError(f"synthetic {role} client interruption")
                    return job_id

                arguments = {
                    "lock": lock,
                    "policy": self.policy,
                    "output_dir": root / "dispatch",
                    "procedure_lock_path": lock_dir / "procedure_lock.json",
                    "full_search_null_script": script,
                    "full_search_null_aggregate_script": aggregate_script,
                    "runtime_provider": CANONICAL_RUNTIME_PROVIDER,
                    "runtime_bundle": bundle,
                    "runtime_bundle_sha256": bundle_sha256,
                    "full_null_work_root": root / "scratch" / "full-null",
                    "full_null_durable_root": root / "durable" / "full-null",
                    "submitter": submit,
                    "job_resolver": lambda name: (
                        accepted.get(name) if recovery_visible else None
                    ),
                    "job_status_resolver": lambda job_id, name: (
                        freeze_scheduler_job_status(
                            job_id=job_id,
                            job_name=name,
                            parent_state="RUNNING",
                        )
                    ),
                    **authorization,
                }
                with self.assertRaisesRegex(RuntimeError, interrupted_role):
                    dispatch_procedure_lock(**arguments)
                with self.assertRaisesRegex(
                    ProcedureDispatchError, "submission remains uncertain"
                ):
                    dispatch_procedure_lock(**arguments)
                command_count = len(commands)
                recovery_visible = True
                receipt = dispatch_procedure_lock(**arguments)
                self.assertEqual(
                    [row["job_id"] for row in receipt["jobs_submitted"]],
                    ["43001", "43002"],
                )
                self.assertEqual(
                    sum(command[-1] == "replicate" for command in commands), 1
                )
                self.assertEqual(
                    sum(command[-1] == "aggregate" for command in commands), 1
                )
                self.assertGreaterEqual(len(commands), command_count)

    @mock.patch("ep12_executor.procedure_lock_dispatch.subprocess.run")
    def test_scheduler_name_recovery_rejects_cross_source_ambiguity(
        self, run: mock.Mock
    ) -> None:
        name = "ep12-null-r-frozen-token"
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, stdout=f"44001|{name}|zijiao\n", stderr=""
            ),
            subprocess.CompletedProcess(
                [], 0, stdout=f"44002|{name}|zijiao\n", stderr=""
            ),
        ]
        with mock.patch(
            "ep12_executor.procedure_lock_dispatch._effective_user",
            return_value="zijiao",
        ), self.assertRaisesRegex(ProcedureDispatchError, "multiple scheduler jobs"):
            _default_job_resolver(name)

    def test_terminal_failed_array_retries_only_missing_checkpoints_crash_safely(self) -> None:
        lock = build_procedure_lock(**self.fixture())
        commands: list[list[str]] = []
        accepted: dict[str, str] = {}
        interrupted_retry_array = False
        interrupted_retry_aggregate = False

        def submit(command: object) -> str:
            nonlocal interrupted_retry_array, interrupted_retry_aggregate
            normalized = list(command)
            commands.append(normalized)
            role = str(normalized[-1])
            if len(commands) == 1:
                return "51001"
            if len(commands) == 2:
                return "51002"
            name = next(
                value.split("=", 1)[1]
                for value in normalized
                if value.startswith("--job-name=")
            )
            generation = 1 if len(commands) <= 4 else 2
            progress = json.loads(
                (
                    Path(arguments["output_dir"])
                    / f"dispatch_recovery_progress_{generation:03d}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                progress["state"],
                (
                    "retry_array_submission_uncertain"
                    if role == "replicate"
                    else "retry_aggregate_submission_uncertain"
                ),
            )
            if role == "replicate" and not interrupted_retry_array:
                interrupted_retry_array = True
                accepted[name] = "51003"
                raise RuntimeError("synthetic retry submission interruption")
            if role == "aggregate" and not interrupted_retry_aggregate:
                interrupted_retry_aggregate = True
                accepted[name] = "51004"
                raise RuntimeError("synthetic retry aggregate interruption")
            return "51005" if role == "replicate" else "51006"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_dir = root / "lock"
            write_procedure_lock(lock_dir, lock)
            script = root / "run_full_search_null.sbatch"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            aggregate_script = root / "run_full_search_null_aggregate.sbatch"
            aggregate_script.write_text("#!/bin/bash\n", encoding="utf-8")
            bundle, bundle_sha256, authorization = self.runtime_bundle(
                root, lock, lock_dir / "procedure_lock.json"
            )
            work_root = root / "scratch" / "full-null"
            arguments = {
                "lock": lock,
                "policy": self.policy,
                "output_dir": root / "dispatch",
                "procedure_lock_path": lock_dir / "procedure_lock.json",
                "full_search_null_script": script,
                "full_search_null_aggregate_script": aggregate_script,
                "runtime_provider": CANONICAL_RUNTIME_PROVIDER,
                "runtime_bundle": bundle,
                "runtime_bundle_sha256": bundle_sha256,
                "full_null_work_root": work_root,
                "full_null_durable_root": root / "durable" / "full-null",
                "submitter": submit,
                "job_resolver": lambda name: accepted.get(name),
                **authorization,
            }
            receipt = dispatch_procedure_lock(**arguments)
            runtime_manifest = json.loads(
                (bundle / "runtime_bundle.json").read_text(encoding="utf-8")
            )
            checkpoint_core = {
                "complete": True,
                "replicate_index": 0,
                "replicate_id": "null_001",
                "procedure_lock_sha256": runtime_manifest[
                    "procedure_lock_sha256"
                ],
                "synthetic_checkpoint": True,
            }
            checkpoint = {
                **checkpoint_core,
                "checkpoint_sha256": digest_object(checkpoint_core),
            }
            checkpoint_path = work_root / "replicates/null_001/checkpoint.json"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_text(
                json.dumps(checkpoint, sort_keys=True) + "\n", encoding="utf-8"
            )

            array_name = receipt["jobs_submitted"][0]["job_name"]
            aggregate_name = receipt["jobs_submitted"][1]["job_name"]

            def status(job_id: str, name: str) -> dict[str, object]:
                if job_id == "51001":
                    self.assertEqual(name, array_name)
                    return freeze_scheduler_job_status(
                        job_id=job_id,
                        job_name=name,
                        parent_state="FAILED",
                        task_states={
                            index: "COMPLETED" if index == 0 else "FAILED"
                            for index in range(self.policy.null_replicates)
                        },
                    )
                if job_id == "51002":
                    self.assertEqual(name, aggregate_name)
                    return freeze_scheduler_job_status(
                        job_id=job_id,
                        job_name=name,
                        parent_state="CANCELLED",
                    )
                self.assertIn(job_id, {"51003", "51004"})
                return freeze_scheduler_job_status(
                    job_id=job_id,
                    job_name=name,
                    parent_state="RUNNING" if job_id == "51003" else "PENDING",
                )

            recovery_arguments = {
                **arguments,
                "job_status_resolver": status,
                "recover_terminal_failure": True,
                "checkpoint_verifier": lambda index: (
                    checkpoint if index == 0 else None
                ),
            }
            with self.assertRaisesRegex(RuntimeError, "retry submission"):
                dispatch_procedure_lock(**recovery_arguments)
            with self.assertRaisesRegex(RuntimeError, "retry aggregate"):
                dispatch_procedure_lock(**recovery_arguments)
            recovery = dispatch_procedure_lock(**recovery_arguments)
            replay = dispatch_procedure_lock(
                **{
                    **recovery_arguments,
                    "submitter": lambda _: self.fail(
                        "recovery receipt replay resubmitted a job"
                    ),
                    "job_resolver": lambda _: self.fail(
                        "recovery receipt replay queried scheduler"
                    ),
                    "job_status_resolver": status,
                }
            )
            def failed_retry_status(job_id: str, name: str) -> dict[str, object]:
                if job_id == "51003":
                    return freeze_scheduler_job_status(
                        job_id=job_id,
                        job_name=name,
                        parent_state="FAILED",
                        task_states={index: "FAILED" for index in range(1, 99)},
                    )
                self.assertEqual(job_id, "51004")
                return freeze_scheduler_job_status(
                    job_id=job_id,
                    job_name=name,
                    parent_state="CANCELLED",
                )

            second_generation = dispatch_procedure_lock(
                **{
                    **recovery_arguments,
                    "job_status_resolver": failed_retry_status,
                }
            )

        self.assertEqual(recovery, replay)
        self.assertEqual(recovery["retry_array_spec"], "1-98")
        self.assertEqual(recovery["retry_indices"], list(range(1, 99)))
        self.assertEqual(recovery["retry_array_job_id"], "51003")
        self.assertEqual(recovery["retry_aggregate_job_id"], "51004")
        self.assertEqual(recovery["generation"], 1)
        self.assertEqual(second_generation["generation"], 2)
        self.assertEqual(
            second_generation["previous_recovery_sha256"],
            recovery["recovery_sha256"],
        )
        self.assertEqual(second_generation["retry_array_job_id"], "51005")
        self.assertEqual(second_generation["retry_aggregate_job_id"], "51006")
        self.assertEqual(
            recovery["aggregate_main_wrapper_sha256_export"],
            recovery["launch"]["aggregate_main_wrapper_sha256_export"],
        )
        self.assertEqual(
            second_generation["aggregate_main_wrapper_sha256_export"],
            second_generation["launch"][
                "aggregate_main_wrapper_sha256_export"
            ],
        )
        self.assertFalse(recovery["blind_duplicate_submission"])
        self.assertEqual(sum(command[-1] == "replicate" for command in commands), 3)
        self.assertIn("--array=1-98", commands[2])
        self.assertIn("--dependency=afterok:51003", commands[3])
        self.assertIn("--array=1-98", commands[4])
        self.assertIn("--dependency=afterok:51005", commands[5])
        for command in (commands[1], commands[3], commands[5]):
            self.assertIn("--cpus-per-task=1", command)
            self.assertIn("--mem=32G", command)
            self.assertIn("--time=12:00:00", command)
            self.assertIn("--no-requeue", command)
            self.assertIn("run_full_search_null_aggregate.sbatch", command[-2])
            aggregate_export = next(
                value for value in command if value.startswith("--export=")
            )
            self.assertIn(
                recovery["launch"]["aggregate_main_wrapper_sha256_export"],
                aggregate_export,
            )

    def test_failed_mechanical_control_emits_status_without_policy_terminal(self) -> None:
        inputs = self.fixture(weak_mode=True)
        observed = next(
            record
            for record in inputs["control_result_inventory"]["controls"]
            if record["falsifier_id"]
            not in {FITTED_NULL_FALSIFIER_ID, NOVELTY_AUDIT_FALSIFIER_ID}
        )
        observed["status"] = "failed"
        observed["control_passed"] = False
        self.refresh_pre_manifest(inputs)
        lock = build_procedure_lock(**inputs)
        self.assertTrue(lock["procedure_locked"])
        self.assertFalse(lock["null_eligible"])
        with tempfile.TemporaryDirectory() as directory:
            receipt = dispatch_procedure_lock(
                lock=lock,
                policy=self.policy,
                output_dir=Path(directory),
                procedure_lock_path=Path(directory) / "not-needed.json",
                full_search_null_script=Path(directory) / "not-needed.sbatch",
                runtime_provider=None,
                submitter=lambda _: self.fail("noneligible lock submitted a job"),
            )
        self.assertEqual(receipt["action"], "null_not_eligible")
        self.assertEqual(receipt["jobs_submitted"], [])
        self.assertFalse(receipt["episode_terminal_emitted"])
        self.assertFalse(receipt["final_connectivity_accessed"])

    def test_unlocked_or_invalid_provider_never_submits(self) -> None:
        inputs = self.fixture()
        inputs["control_result_inventory"]["controls"].pop()
        self.refresh_pre_manifest(inputs)
        unlocked = build_procedure_lock(**inputs)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProcedureDispatchError, "unlocked"):
                dispatch_procedure_lock(
                    lock=unlocked,
                    policy=self.policy,
                    output_dir=Path(directory),
                    procedure_lock_path=Path(directory) / "lock.json",
                    full_search_null_script=Path(directory) / "null.sbatch",
                    runtime_provider=None,
                    submitter=lambda _: self.fail("unlocked procedure submitted a job"),
                )
        lock = build_procedure_lock(**self.fixture())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ProcedureDispatchError, "runtime provider"):
                dispatch_procedure_lock(
                    lock=lock,
                    policy=self.policy,
                    output_dir=Path(directory),
                    procedure_lock_path=Path(directory) / "lock.json",
                    full_search_null_script=Path(directory) / "null.sbatch",
                    runtime_provider="bad,value",
                    submitter=lambda _: self.fail("invalid provider submitted a job"),
                )


if __name__ == "__main__":
    unittest.main()
