from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ep12_executor.full_search_runtime as runtime_module
import ep12_executor.post_lock_runtime_materializer as materializer_module
from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.fitted_null import (
    FittedNullDesign,
    TypePooledExpansionPlan,
    null_seed_binding,
)
from ep12_executor.full_search_runtime import (
    FullSearchRuntimeError,
    RUNTIME_BENCHMARK_SCHEMA,
    TIMING_CORPUS_SCHEMA,
    freeze_selected_fit_attestation,
    load_runtime_bundle,
    materialize_runtime_bundle,
    stage_runtime_bundle,
)
from ep12_executor.null_controller import freeze_null_controller_program
from ep12_executor.null_trial_runner import (
    ProductionFullSearchCallback,
    SELECTION_PROGRAM_SHA256,
)
from ep12_executor.policy import EpisodePolicy, digest_bytes, digest_object
from ep12_executor.post_lock_runtime_materializer import (
    PostLockRuntimeMaterializerError,
    _timing_evidence,
    freeze_resource_plan,
)
from ep12_executor.scientific_models import FittedPredictiveModel, helmert_ilr_basis


EPISODE_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
PLAN_PATH = EPISODE_ROOT / "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json"
ROLE_SHA256 = "cd" * 32
SOURCE_SHA256 = "ef" * 32
UPSTREAM_SHA256 = "ab" * 32
UPSTREAM_FILE_SHA256 = "12" * 32
GENERATOR_SELECTION_SHA256 = "34" * 32


def _program(policy: EpisodePolicy) -> dict:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    return freeze_null_controller_program(
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


def _state() -> tuple[
    FittedNullDesign,
    dict[tuple[str, str], FittedPredictiveModel],
    dict[int, np.ndarray],
    dict[str, dict[str, str]],
]:
    design = FittedNullDesign.create(
        neuron_ids=[101, 102, 103, 104],
        focal_types=["A", "A", "A", "A"],
        sides=["L", "L", "R", "R"],
        covariates=np.asarray([[-1.0], [-0.5], [0.5], [1.0]], dtype=float),
        raw_counts=np.asarray(
            [[12, 4, 2, 1], [10, 5, 1, 2], [11, 3, 4, 1], [13, 2, 3, 2]],
            dtype=np.int64,
        ),
        raw_vocabulary=[
            "typed::x", "typed::y", "untyped_status::Traced::complete",
            "missing_annotation",
        ],
        collapsed_vocabulary=[
            "typed::x", "other_typed", "untyped", "missing_annotation",
        ],
        raw_to_collapsed=[0, 1, 2, 3],
        collapsed_typed_bins=["typed::x", "other_typed"],
    )
    dimension = len(design.collapsed_vocabulary) - 1
    basis = helmert_ilr_basis(len(design.collapsed_vocabulary))
    coefficients = np.zeros((design.covariates.shape[1] + 1, dimension))
    coefficients[0] = np.asarray([1.0, 0.0, -0.5, -0.5]) @ basis
    model = FittedPredictiveModel(
        model_id="T",
        basis=basis,
        nuisance_coefficients=coefficients,
        latent_offsets=np.zeros((1, dimension)),
        log_weights=np.zeros(1),
        prior_means=np.zeros((1, dimension)),
        prior_covariances=np.eye(dimension)[None, :, :] * 1e-10,
        prior_weights=np.ones(1),
        integration_draws=1,
        integration_seed=7,
        parameter_count=int(coefficients.size),
        diagnostics={},
        pseudocount=0.5,
    )
    generators = {("A", "L"): model, ("A", "R"): model}
    coordinate = np.linspace(-0.5, 0.5, len(design.neuron_ids))
    covariates: dict[int, np.ndarray] = {}
    for spline_df, width in ((0, 22), (3, 34), (5, 46)):
        value = np.zeros((len(design.neuron_ids), width), dtype=np.float64)
        value[:, 0] = coordinate
        covariates[spline_df] = value
    hierarchy = {
        "x": {"subclass": "alpha", "class": "sensory"},
        "y": {"subclass": "beta", "class": "sensory"},
    }
    return design, generators, covariates, hierarchy


def _production_inventory() -> tuple[list[str], np.ndarray, dict[str, dict[str, str]]]:
    """Return a non-outcome fixture with production's 71 families/6 endpoints."""

    typed = [f"typed::partner_{index:02d}" for index in range(36)]
    endpoints = [
        "untyped_status::Other::unknown",
        "untyped_status::Traced::complete",
        "untyped_status::Other::orphan",
        "untyped_status::Assign::candidate",
        "missing_annotation",
    ]
    vocabulary = typed + endpoints
    counts = np.ones((1, len(vocabulary)), dtype=np.int64)
    hierarchy = {
        f"partner_{index:02d}": {
            "subclass": f"subclass_{index:02d}",
            "class": f"class_{min(index, 34):02d}",
        }
        for index in range(36)
    }
    return vocabulary, counts, hierarchy


def _benchmark_receipt(
    *,
    design_sha256: str,
    generator_manifest_sha256: str,
    expansion_plan_sha256: str,
    callback_identity_sha256: str,
    elapsed: tuple[float, float, float] = (10.0, 12.0, 11.0),
    timing_corpus_sha256: str = "89" * 32,
) -> dict:
    core = {
        "schema_version": RUNTIME_BENCHMARK_SCHEMA,
        "method": "production_fitted_null_sampler_and_T_U_M_callback_base_fit",
        "worker_count": 32,
        "benchmark_replicate_indices": [0, 1, 2],
        "benchmark_replicate_ids": ["null_001", "null_002", "null_003"],
        "benchmark_replicate_seeds_sha256": "45" * 32,
        "max_timing_trial_id": "coverage_017",
        "max_timing_configuration_sha256": "67" * 32,
        "design_sha256": design_sha256,
        "generator_manifest_sha256": generator_manifest_sha256,
        "expansion_plan_sha256": expansion_plan_sha256,
        "callback_identity_sha256": callback_identity_sha256,
        "elapsed_wall_seconds": list(elapsed),
        "measured_parallel_mean_wall_seconds": sum(elapsed) / len(elapsed),
        "measured_parallel_max_wall_seconds": max(elapsed),
        "complete_T_U_M_branches_measured": 3,
        "benchmark_scope": runtime_module.RUNTIME_BENCHMARK_SCOPE,
        "benchmark_role": runtime_module.RUNTIME_BENCHMARK_ROLE,
        "projection_is_upper_bound": False,
        "observed_timing_corpus_sha256": timing_corpus_sha256,
        "observed_timing_corpus_trial_count": 36,
        "scientific_scores_or_objectives_persisted": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def _timing(
    receipt: dict,
    *,
    serial_mean: float = 10.0,
    serial_max: float = 20.0,
    serial_cpu: float = 9.0,
) -> dict:
    return {
        "method": (
            "production_parallel_benchmark_with_authenticated_"
            "observed_timing_corpus_maxima"
        ),
        "observed_timing_corpus_schema": TIMING_CORPUS_SCHEMA,
        "observed_timing_corpus_trial_count": 36,
        "observed_timing_corpus_sha256": receipt[
            "observed_timing_corpus_sha256"
        ],
        "observed_attestation_sha256": "9a" * 32,
        "observed_attestation_manifest_file_sha256": "9b" * 32,
        "observed_serial_mean_wall_seconds": serial_mean,
        "observed_serial_max_wall_seconds": serial_max,
        "observed_serial_mean_cpu_seconds_per_branch": serial_cpu / 2.0,
        "observed_serial_max_cpu_seconds_per_branch": serial_cpu,
        "max_timing_trial_id": "coverage_017",
        "max_busy_timing_trial_id": "coverage_017",
        "legacy_cost_projection_file_sha256": "56" * 32,
        "legacy_cost_projection_used_for_projection": False,
        "max_timing_report_file_sha256": "78" * 32,
        "max_busy_timing_report_file_sha256": "78" * 32,
        "parallelism_scope": "provider_types_within_one_complete_T_U_M_branch",
        "control_branch_execution": "sequential",
        "scientific_outcomes_persisted": False,
        "runtime_benchmark_receipt": receipt,
    }


def _authenticated_timing_fixture(
    root: Path,
    *,
    program: dict,
) -> dict:
    outputs = root / "outputs"
    run_root = outputs / "development_run002"
    cost_path = outputs / "development_run001" / "real_cost_projection.json"
    cost_path.parent.mkdir(parents=True)
    cost_path.write_text(
        json.dumps(
            {
                "schema_version": "ep12.real_development_cost_projection.v1",
                "source_trial_wall_seconds": 999999.0,
                "source_trial_cpu_seconds": 888888.0,
            }
        ),
        encoding="utf-8",
    )
    max_configuration = dict(program["coverage_configurations"][0])
    max_trial_id = str(max_configuration["trial_id"])
    trial_ids = [max_trial_id] + [
        f"timing_{index:03d}" for index in range(2, 37)
    ]
    timing_values: dict[str, tuple[float, float]] = {}
    rows: dict[str, dict] = {}
    report_paths: dict[str, Path] = {}
    for index, trial_id in enumerate(trial_ids):
        if index == 0:
            wall_seconds, cpu_seconds = 200.0, 100.0
        elif index == 1:
            wall_seconds, cpu_seconds = 150.0, 190.0
        else:
            wall_seconds, cpu_seconds = 10.0 + index, 9.0 + index
        configuration = {}
        if trial_id == max_trial_id:
            configuration = {
                **max_configuration,
                "configuration_sha256": digest_object(max_configuration),
            }
        report = {
            "schema_version": "ep12.development_trial.v1",
            "trial_id": trial_id,
            "status": "completed",
            "data_role": "development_only",
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "configuration": configuration,
            # Outcome-shaped data demonstrate that no such values are emitted.
            "aggregate": {"objective": 123456.0},
            "final_connectivity_accessed": False,
        }
        report_path = run_root / "observed" / trial_id / "trial_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(
            json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_sha256 = digest_bytes(report_path.read_bytes())
        rows[trial_id] = {
            "trial_id": trial_id,
            "source_trial_directory": f"observed/{trial_id}",
            "source_files_sha256": {"trial_report.json": report_sha256},
        }
        timing_values[trial_id] = (wall_seconds, cpu_seconds)
        report_paths[trial_id] = report_path
    attestation = {
        "attestation_manifest_sha256": "aa" * 32,
        "attestation_sha256": "bb" * 32,
        "verified_trial_count": len(rows),
        "_trial_rows_by_id": rows,
    }
    return {
        "outputs": outputs,
        "run_root": run_root,
        "cost_path": cost_path,
        "trial_ids": trial_ids,
        "timing_values": timing_values,
        "report_paths": report_paths,
        "max_trial_id": max_trial_id,
        "max_configuration": max_configuration,
        "attestation": attestation,
    }


class FullSearchRuntimeAdaptiveTests(unittest.TestCase):
    def test_authenticated_timing_corpus_uses_independent_full_maxima(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _authenticated_timing_fixture(
                Path(temporary), program=program
            )
            with (
                patch.object(
                    materializer_module,
                    "EPISODE_OUTPUTS_ROOT",
                    fixture["outputs"],
                ),
                patch.object(
                    materializer_module,
                    "verify_attestation_root",
                    return_value=fixture["attestation"],
                ),
            ):
                evidence, configuration = _timing_evidence(
                    fixture["cost_path"],
                    fixture["report_paths"][fixture["max_trial_id"]],
                    run_root=fixture["run_root"],
                    expected_observed_attestation_manifest_sha256="aa" * 32,
                    expected_trial_ids=fixture["trial_ids"],
                    policy=policy,
                    controller_program=program,
                )
            self.assertEqual(
                evidence["observed_timing_corpus_trial_count"], 36
            )
            self.assertEqual(evidence["observed_serial_max_wall_seconds"], 200.0)
            self.assertEqual(
                evidence["observed_serial_max_cpu_seconds_per_branch"], 190.0
            )
            self.assertEqual(evidence["max_timing_trial_id"], fixture["max_trial_id"])
            self.assertEqual(evidence["max_busy_timing_trial_id"], "timing_002")
            self.assertFalse(evidence["legacy_cost_projection_used_for_projection"])
            self.assertFalse(evidence["scientific_outcomes_persisted"])
            self.assertNotIn("objective", json.dumps(evidence).lower())
            self.assertEqual(configuration, fixture["max_configuration"])

    def test_authenticated_timing_corpus_rejects_stale_max_and_bad_hash(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _authenticated_timing_fixture(
                Path(temporary), program=program
            )
            stale_id = "timing_003"
            common = {
                "run_root": fixture["run_root"],
                "expected_observed_attestation_manifest_sha256": "aa" * 32,
                "expected_trial_ids": fixture["trial_ids"],
                "policy": policy,
                "controller_program": program,
            }
            with (
                patch.object(
                    materializer_module,
                    "EPISODE_OUTPUTS_ROOT",
                    fixture["outputs"],
                ),
                patch.object(
                    materializer_module,
                    "verify_attestation_root",
                    return_value=fixture["attestation"],
                ),
                self.assertRaises(PostLockRuntimeMaterializerError),
            ):
                _timing_evidence(
                    fixture["cost_path"],
                    fixture["report_paths"][stale_id],
                    **common,
                )
            bad_attestation = json.loads(json.dumps(fixture["attestation"]))
            bad_attestation["_trial_rows_by_id"][fixture["max_trial_id"]][
                "source_files_sha256"
            ]["trial_report.json"] = "00" * 32
            with (
                patch.object(
                    materializer_module,
                    "EPISODE_OUTPUTS_ROOT",
                    fixture["outputs"],
                ),
                patch.object(
                    materializer_module,
                    "verify_attestation_root",
                    return_value=bad_attestation,
                ),
                self.assertRaises(PostLockRuntimeMaterializerError),
            ):
                _timing_evidence(
                    fixture["cost_path"],
                    fixture["report_paths"][fixture["max_trial_id"]],
                    **common,
                )

    def test_resource_plan_and_runtime_bundle_round_trip(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        design, generators, covariates, hierarchy = _state()
        selected = dict(program["coverage_configurations"][0])
        attestation = freeze_selected_fit_attestation(
            controller_program=program,
            selected_trial_id=selected["trial_id"],
            selected_configuration=selected,
            development_role_manifest_sha256=ROLE_SHA256,
            source_selection_artifact_sha256=SOURCE_SHA256,
            upstream_procedure_lock_sha256=UPSTREAM_SHA256,
            null_generator_selection_sha256=GENERATOR_SELECTION_SHA256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = root / "outputs"
            scratch = root / "scratch"
            outputs.mkdir()
            scratch.mkdir()
            bundle_path = outputs / "runtime"
            artifact_root = scratch / "work"
            with (
                patch.object(runtime_module, "EPISODE_OUTPUTS_ROOT", outputs),
                patch.object(runtime_module, "EPISODE_SCRATCH_ROOT", scratch),
            ):
                settings = {"integration_draws": 2, "source_cv_folds": 2}
                probe = root / "callback_probe"
                runtime_module._write_callback_data(
                    probe,
                    design=design,
                    covariates=covariates,
                    hierarchy=hierarchy,
                    settings=settings,
                )
                probe_inputs, _ = runtime_module._load_callback_data(
                    probe,
                    design=design,
                    policy=policy,
                    artifact_root=root / "probe_artifacts",
                )
                callback = ProductionFullSearchCallback(
                    policy=policy, inputs=probe_inputs
                )
                expansion = TypePooledExpansionPlan.build(design)
                binding = null_seed_binding(
                    design,
                    generators,
                    expansion,
                    selection_sha256=SELECTION_PROGRAM_SHA256,
                )
                receipt = _benchmark_receipt(
                    design_sha256=design.design_sha256,
                    generator_manifest_sha256=binding[
                        "generator_manifest_sha256"
                    ],
                    expansion_plan_sha256=expansion.plan_sha256,
                    callback_identity_sha256=digest_object(callback.identity()),
                )
                inventory_vocabulary, inventory_counts, inventory_hierarchy = (
                    _production_inventory()
                )
                resource = freeze_resource_plan(
                    policy=policy,
                    controller_program=program,
                    raw_vocabulary=inventory_vocabulary,
                    raw_counts=inventory_counts,
                    partner_hierarchy=inventory_hierarchy,
                    covariates_by_spline_df=covariates,
                    timing_basis=_timing(receipt),
                )
                self.assertEqual(resource["requested_cpus_per_replicate"], 32)
                self.assertEqual(
                    resource["minimum_complete_T_U_M_fit_count"], 308
                )
                self.assertEqual(
                    resource["maximum_complete_T_U_M_fit_count"], 690
                )
                self.assertEqual(resource["program_search_count"], 100)
                self.assertFalse(resource["cpu_core_hour_ceiling_enforced"])
                self.assertEqual(
                    resource["runtime_benchmark_scope"],
                    runtime_module.RUNTIME_BENCHMARK_SCOPE,
                )
                self.assertEqual(
                    resource["runtime_benchmark_role"],
                    "authenticated_planning_estimate",
                )
                self.assertFalse(resource["projection_is_upper_bound"])
                self.assertEqual(
                    resource["actual_per_trial_runtime_enforcement"],
                    runtime_module.PER_TRIAL_RUNTIME_ENFORCEMENT,
                )
                self.assertEqual(
                    resource["actual_cumulative_runtime_enforcement"],
                    runtime_module.CUMULATIVE_RUNTIME_ENFORCEMENT,
                )
                self.assertEqual(
                    resource["wall_budget_exhaustion_handling"],
                    runtime_module.WALL_BUDGET_EXHAUSTION_HANDLING,
                )
                self.assertEqual(
                    resource["projected_minimum_replicate_busy_core_hours"],
                    308 * 9.0 / 3600.0,
                )
                self.assertLessEqual(
                    resource["projected_maximum_replicate_wall_hours"], 120
                )
                self.assertEqual(
                    resource["maximum_model_family_fit_count"],
                    3 * resource["maximum_complete_T_U_M_fit_count"],
                )
                self.assertEqual(
                    resource[
                        "projected_maximum_replicate_allocated_core_hours"
                    ],
                    32 * resource["projected_maximum_replicate_wall_hours"],
                )
                self.assertLessEqual(
                    resource["projected_maximum_replicate_busy_core_hours"],
                    resource[
                        "projected_maximum_replicate_allocated_core_hours"
                    ],
                )
                manifest = materialize_runtime_bundle(
                    output_dir=bundle_path,
                    policy=policy,
                    design=design,
                    generators=generators,
                    covariates_by_spline_df=covariates,
                    partner_hierarchy=hierarchy,
                    fit_settings=settings,
                    controller_program=program,
                    selection_attestation=attestation,
                    null_generator_selection_sha256=GENERATOR_SELECTION_SHA256,
                    development_role_manifest_sha256=ROLE_SHA256,
                    upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                    upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                    resource_plan=resource,
                )
                loaded = load_runtime_bundle(
                    bundle_path,
                    policy=policy,
                    artifact_root=artifact_root,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                    upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                )
                self.assertEqual(
                    loaded.controller_program["program_sha256"],
                    program["program_sha256"],
                )
                self.assertEqual(
                    manifest["resource_plan_sha256"], resource["plan_sha256"]
                )
                staged_bundle = scratch / "staged_runtime"
                stage_receipt = stage_runtime_bundle(
                    durable_bundle_dir=bundle_path,
                    scratch_bundle_dir=staged_bundle,
                    policy=policy,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                    upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                )
                self.assertEqual(
                    stage_receipt["scratch_bundle_dir"], str(staged_bundle)
                )
                self.assertEqual(
                    stage_runtime_bundle(
                        durable_bundle_dir=bundle_path,
                        scratch_bundle_dir=staged_bundle,
                        policy=policy,
                        expected_bundle_sha256=manifest["bundle_sha256"],
                        upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                        upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                    ),
                    stage_receipt,
                )
                collision_stage = scratch / "collision_stage"
                collision_temporary = scratch / (
                    f".{collision_stage.name}.staging."
                    f"{manifest['bundle_sha256'][:16]}.{runtime_module.os.getpid()}."
                    f"{'a' * 32}"
                )
                collision_temporary.mkdir()
                collision_marker = collision_temporary / "foreign-owner.txt"
                collision_marker.write_text("do not remove\n", encoding="utf-8")
                with patch.object(
                    runtime_module.secrets,
                    "token_hex",
                    side_effect=["a" * 32, "b" * 32],
                ):
                    collision_receipt = stage_runtime_bundle(
                        durable_bundle_dir=bundle_path,
                        scratch_bundle_dir=collision_stage,
                        policy=policy,
                        expected_bundle_sha256=manifest["bundle_sha256"],
                        upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                        upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                    )
                self.assertTrue(collision_marker.is_file())
                self.assertEqual(
                    collision_receipt["scratch_bundle_dir"], str(collision_stage)
                )

                raced_stage = scratch / "raced_stage"
                real_rename = runtime_module.os.rename
                competitor_published = False

                def publish_competitor(source: object, destination: object) -> None:
                    nonlocal competitor_published
                    if Path(destination) == raced_stage and not competitor_published:
                        competitor_published = True
                        runtime_module.shutil.copytree(
                            bundle_path, raced_stage, symlinks=True
                        )
                        raise OSError(
                            runtime_module.errno.EEXIST,
                            "concurrent stage already published",
                        )
                    real_rename(source, destination)

                with patch.object(
                    runtime_module.os, "rename", side_effect=publish_competitor
                ):
                    raced_receipt = stage_runtime_bundle(
                        durable_bundle_dir=bundle_path,
                        scratch_bundle_dir=raced_stage,
                        policy=policy,
                        expected_bundle_sha256=manifest["bundle_sha256"],
                        upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                        upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                    )
                self.assertTrue(competitor_published)
                self.assertEqual(
                    raced_receipt["scratch_bundle_dir"], str(raced_stage)
                )
                self.assertFalse(any(
                    path.name.startswith(f".{raced_stage.name}.staging.")
                    for path in scratch.iterdir()
                ))
                staged_runtime = load_runtime_bundle(
                    staged_bundle,
                    policy=policy,
                    artifact_root=artifact_root,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                    upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                )
                self.assertEqual(
                    staged_runtime.controller_program["program_sha256"],
                    program["program_sha256"],
                )
                resource_path = bundle_path / "resource_plan.json"
                payload = json.loads(resource_path.read_text(encoding="utf-8"))
                payload["provider_type_workers"] = 31
                resource_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(FullSearchRuntimeError):
                    load_runtime_bundle(
                        bundle_path,
                        policy=policy,
                        artifact_root=artifact_root,
                        expected_bundle_sha256=manifest["bundle_sha256"],
                        upstream_procedure_lock_sha256=UPSTREAM_SHA256,
                        upstream_procedure_lock_file_sha256=UPSTREAM_FILE_SHA256,
                    )

    def test_resource_plan_rejects_over_12h_control_projection(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        _, _, covariates, _ = _state()
        vocabulary, counts, hierarchy = _production_inventory()
        receipt = _benchmark_receipt(
            design_sha256="01" * 32,
            generator_manifest_sha256="02" * 32,
            expansion_plan_sha256="03" * 32,
            callback_identity_sha256="04" * 32,
        )
        with self.assertRaises(FullSearchRuntimeError):
            freeze_resource_plan(
                policy=policy,
                controller_program=program,
                raw_vocabulary=vocabulary,
                raw_counts=counts,
                partner_hierarchy=hierarchy,
                covariates_by_spline_df=covariates,
                timing_basis=_timing(
                    receipt, serial_mean=10.0, serial_max=24 * 3600.0
                ),
            )

    def test_resource_plan_rejects_benchmark_corpus_mismatch(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        _, _, covariates, _ = _state()
        vocabulary, counts, hierarchy = _production_inventory()
        receipt = _benchmark_receipt(
            design_sha256="01" * 32,
            generator_manifest_sha256="02" * 32,
            expansion_plan_sha256="03" * 32,
            callback_identity_sha256="04" * 32,
        )
        timing = _timing(receipt)
        timing["observed_timing_corpus_sha256"] = "fe" * 32
        with self.assertRaises(FullSearchRuntimeError):
            freeze_resource_plan(
                policy=policy,
                controller_program=program,
                raw_vocabulary=vocabulary,
                raw_counts=counts,
                partner_hierarchy=hierarchy,
                covariates_by_spline_df=covariates,
                timing_basis=timing,
            )

    def test_resource_plan_rejects_false_benchmark_or_enforcement_claims(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        _, _, covariates, _ = _state()
        vocabulary, counts, hierarchy = _production_inventory()
        receipt = _benchmark_receipt(
            design_sha256="01" * 32,
            generator_manifest_sha256="02" * 32,
            expansion_plan_sha256="03" * 32,
            callback_identity_sha256="04" * 32,
        )
        false_upper_bound = dict(receipt)
        false_upper_bound["projection_is_upper_bound"] = True
        false_upper_bound["receipt_sha256"] = digest_object({
            key: value
            for key, value in false_upper_bound.items()
            if key != "receipt_sha256"
        })
        with self.assertRaisesRegex(
            FullSearchRuntimeError, "benchmark violates"
        ):
            freeze_resource_plan(
                policy=policy,
                controller_program=program,
                raw_vocabulary=vocabulary,
                raw_counts=counts,
                partner_hierarchy=hierarchy,
                covariates_by_spline_df=covariates,
                timing_basis=_timing(false_upper_bound),
            )

        plan = freeze_resource_plan(
            policy=policy,
            controller_program=program,
            raw_vocabulary=vocabulary,
            raw_counts=counts,
            partner_hierarchy=hierarchy,
            covariates_by_spline_df=covariates,
            timing_basis=_timing(receipt),
        )
        changed = dict(plan)
        changed["actual_cumulative_runtime_enforcement"] = "not_enforced"
        changed["plan_sha256"] = digest_object({
            key: value
            for key, value in changed.items()
            if key != "plan_sha256"
        })
        with self.assertRaisesRegex(
            FullSearchRuntimeError, "violates frozen limits"
        ):
            runtime_module.verify_resource_plan(changed, policy=policy)


if __name__ == "__main__":
    unittest.main()
