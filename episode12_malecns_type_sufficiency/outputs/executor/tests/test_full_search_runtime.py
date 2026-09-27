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
    TIMING_CORPUS_KIND,
    build_selected_fit_binding,
    load_runtime_bundle,
    materialize_runtime_bundle,
    stage_runtime_bundle,
)
from ep12_executor.null_controller import freeze_null_controller_program
from ep12_executor.null_trial_runner import (
    ProductionFullSearchCallback,
    SELECTION_PROGRAM_SHA256,
)
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.post_lock_runtime_materializer import (
    PostLockRuntimeMaterializerError,
    _timing_evidence,
    freeze_resource_plan,
)
from ep12_executor.scientific_models import FittedPredictiveModel, helmert_ilr_basis


EPISODE_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
PLAN_PATH = EPISODE_ROOT / "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json"
ROLE_IDENTITY = "cd" * 32
UPSTREAM_LOCK_IDENTITY = "ab" * 32
GENERATOR_SELECTION_IDENTITY = "34" * 32


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
            "typed::x",
            "typed::y",
            "untyped_status::Traced::complete",
            "missing_annotation",
        ],
        collapsed_vocabulary=[
            "typed::x",
            "other_typed",
            "untyped",
            "missing_annotation",
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


def _benchmark(
    *,
    design_sha256: str,
    generator_manifest_sha256: str,
    expansion_plan_sha256: str,
    callback_identity_sha256: str,
    elapsed: tuple[float, float, float] = (10.0, 12.0, 11.0),
) -> dict:
    return {
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
        "observed_timing_corpus_sha256": "89" * 32,
        "observed_timing_corpus_trial_count": 36,
        "scientific_scores_or_objectives_persisted": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }


def _timing(benchmark: dict, *, serial_max: float = 20.0) -> dict:
    return {
        "method": (
            "production_parallel_benchmark_with_complete_"
            "observed_timing_corpus_maxima"
        ),
        "observed_timing_corpus_kind": TIMING_CORPUS_KIND,
        "observed_timing_corpus_trial_count": 36,
        "observed_timing_corpus_sha256": benchmark[
            "observed_timing_corpus_sha256"
        ],
        "observed_serial_mean_wall_seconds": 10.0,
        "observed_serial_max_wall_seconds": serial_max,
        "observed_serial_mean_cpu_seconds_per_branch": 4.5,
        "observed_serial_max_cpu_seconds_per_branch": 9.0,
        "max_timing_trial_id": "coverage_017",
        "max_busy_timing_trial_id": "coverage_017",
        "legacy_cost_projection_superseded": True,
        "legacy_cost_projection_used_for_projection": False,
        "parallelism_scope": "provider_types_within_one_complete_T_U_M_branch",
        "control_branch_execution": "sequential",
        "scientific_outcomes_persisted": False,
        "runtime_benchmark": benchmark,
    }


class FullSearchRuntimeTests(unittest.TestCase):
    def test_complete_timing_corpus_uses_independent_maxima(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        configurations = program["coverage_configurations"]
        self.assertEqual(len(configurations), 18)
        trial_ids = [str(item["trial_id"]) for item in configurations] + [
            f"timing_extra_{index:03d}" for index in range(1, 19)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary) / "outputs"
            run_root = outputs / "development"
            run_root.mkdir(parents=True)
            rows: dict[str, dict] = {}
            report_paths: dict[str, Path] = {}
            for index, trial_id in enumerate(trial_ids):
                configuration = configurations[min(index, len(configurations) - 1)]
                trial_dir = run_root / trial_id
                trial_dir.mkdir()
                report_path = trial_dir / "trial_report.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "trial_id": trial_id,
                            "configuration": {
                                **configuration,
                                "configuration_sha256": digest_object(configuration),
                            },
                            "final_connectivity_accessed": False,
                        }
                    ),
                    encoding="utf-8",
                )
                wall = 200.0 if index == 4 else 20.0 + index
                cpu = 190.0 if index == 8 else 10.0 + index
                rows[trial_id] = {
                    "trial_report_path": str(report_path),
                    "wall_seconds": wall,
                    "cpu_seconds": cpu,
                }
                report_paths[trial_id] = report_path
            max_wall_id = trial_ids[4]
            with (
                patch.object(materializer_module, "EPISODE_OUTPUTS_ROOT", outputs),
                patch.object(
                    materializer_module,
                    "load_observed_trial_status",
                    return_value={"rows_by_id": rows},
                ),
            ):
                evidence, selected = _timing_evidence(
                    report_paths[max_wall_id],
                    run_root=run_root,
                    expected_trial_ids=list(rows),
                    policy=policy,
                    controller_program=program,
                )
                with self.assertRaises(PostLockRuntimeMaterializerError):
                    _timing_evidence(
                        report_paths[trial_ids[3]],
                        run_root=run_root,
                        expected_trial_ids=list(rows),
                        policy=policy,
                        controller_program=program,
                    )
            self.assertEqual(
                evidence["method"],
                "production_parallel_benchmark_with_complete_"
                "observed_timing_corpus_maxima",
            )
            self.assertEqual(evidence["observed_timing_corpus_trial_count"], 36)
            self.assertEqual(evidence["observed_serial_max_wall_seconds"], 200.0)
            self.assertEqual(evidence["observed_serial_max_cpu_seconds_per_branch"], 190.0)
            self.assertEqual(selected, configurations[4])
            self.assertNotIn("attest", json.dumps(evidence).lower())
            self.assertNotIn("file_sha256", json.dumps(evidence))

    def test_runtime_round_trip_and_idempotent_scratch_stage(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        design, generators, covariates, hierarchy = _state()
        selected = dict(program["coverage_configurations"][0])
        selected_binding = build_selected_fit_binding(
            controller_program=program,
            selected_trial_id=selected["trial_id"],
            selected_configuration=selected,
            development_role_manifest_sha256=ROLE_IDENTITY,
            upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
            null_generator_selection_sha256=GENERATOR_SELECTION_IDENTITY,
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
                callback = ProductionFullSearchCallback(policy=policy, inputs=probe_inputs)
                expansion = TypePooledExpansionPlan.build(design)
                binding = null_seed_binding(
                    design,
                    generators,
                    expansion,
                    selection_sha256=SELECTION_PROGRAM_SHA256,
                )
                benchmark = _benchmark(
                    design_sha256=design.design_sha256,
                    generator_manifest_sha256=binding["generator_manifest_sha256"],
                    expansion_plan_sha256=expansion.plan_sha256,
                    callback_identity_sha256=digest_object(callback.identity()),
                )
                vocabulary, counts, inventory_hierarchy = _production_inventory()
                resource = freeze_resource_plan(
                    policy=policy,
                    controller_program=program,
                    raw_vocabulary=vocabulary,
                    raw_counts=counts,
                    partner_hierarchy=inventory_hierarchy,
                    covariates_by_spline_df=covariates,
                    timing_basis=_timing(benchmark),
                )
                self.assertEqual(resource["array_replicates"], 99)
                self.assertEqual(resource["minimum_complete_T_U_M_fit_count"], 308)
                self.assertEqual(resource["maximum_complete_T_U_M_fit_count"], 690)
                self.assertLessEqual(
                    resource["projected_maximum_replicate_wall_hours"], 120
                )
                self.assertNotIn("schema_version", resource)
                self.assertNotIn("plan_sha256", resource)
                manifest = materialize_runtime_bundle(
                    output_dir=bundle_path,
                    policy=policy,
                    design=design,
                    generators=generators,
                    covariates_by_spline_df=covariates,
                    partner_hierarchy=hierarchy,
                    fit_settings=settings,
                    controller_program=program,
                    selected_fit_binding=selected_binding,
                    null_generator_selection_sha256=GENERATOR_SELECTION_IDENTITY,
                    development_role_manifest_sha256=ROLE_IDENTITY,
                    upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                    resource_plan=resource,
                )
                self.assertNotIn("files", manifest)
                self.assertNotIn("upstream_procedure_lock_file_sha256", manifest)
                self.assertTrue((bundle_path / "selected_fit_binding.json").is_file())
                self.assertFalse((bundle_path / "selected_fit_attestation.json").exists())
                self.assertFalse((bundle_path / "runtime_benchmark_receipt.json").exists())
                loaded = load_runtime_bundle(
                    bundle_path,
                    policy=policy,
                    artifact_root=artifact_root,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                )
                self.assertEqual(
                    loaded.controller_program["program_sha256"],
                    program["program_sha256"],
                )
                staged_bundle = scratch / "staged_runtime"
                first = stage_runtime_bundle(
                    durable_bundle_dir=bundle_path,
                    scratch_bundle_dir=staged_bundle,
                    policy=policy,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                )
                second = stage_runtime_bundle(
                    durable_bundle_dir=bundle_path,
                    scratch_bundle_dir=staged_bundle,
                    policy=policy,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                )
                self.assertEqual(first, second)
                self.assertNotIn("receipt_sha256", first)
                self.assertNotIn("file_inventory_sha256", first)

                resource_path = bundle_path / "resource_plan.json"
                changed = json.loads(resource_path.read_text(encoding="utf-8"))
                changed["human_note"] = "extra metadata is allowed"
                resource_path.write_text(json.dumps(changed), encoding="utf-8")
                load_runtime_bundle(
                    bundle_path,
                    policy=policy,
                    artifact_root=artifact_root,
                    expected_bundle_sha256=manifest["bundle_sha256"],
                    upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                )
                changed["provider_type_workers"] = 31
                resource_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(FullSearchRuntimeError):
                    load_runtime_bundle(
                        bundle_path,
                        policy=policy,
                        artifact_root=artifact_root,
                        expected_bundle_sha256=manifest["bundle_sha256"],
                        upstream_procedure_lock_sha256=UPSTREAM_LOCK_IDENTITY,
                    )

    def test_resource_plan_rejects_control_over_wall_limit(self) -> None:
        policy = EpisodePolicy.load(POLICY_PATH)
        program = _program(policy)
        _, _, covariates, _ = _state()
        vocabulary, counts, hierarchy = _production_inventory()
        benchmark = _benchmark(
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
                timing_basis=_timing(benchmark, serial_max=24 * 3600.0),
            )


if __name__ == "__main__":
    unittest.main()
