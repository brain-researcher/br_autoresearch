"""Production T/U/M callback and falsifier registry for fitted-null replay.

All fits call the same :func:`scientific_models.fit_triplet` implementation as
the observed development runner.  Falsifier handlers transform only the
already generated development-role sample and execute their complete frozen
branch inventories; none accepts an asserted receipt.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .adaptive_search import validate_development_configuration
from .development_trial import _collapse_vocabulary
from .falsifier_controls import (
    ablate_nuisance_block,
    component_partner_contrasts,
    composite_control_receipt,
    cross_side_alignment_permutation,
    endpoint_branch_inventory,
    implementation_spec,
    leave_one_type_influence,
    make_branch_manifest,
    neuron_influence_curves,
    nuisance_block_columns,
    partner_family_inventory,
    pooled_multinomial_noise,
)
from .falsifier_transforms import (
    margin_preserving_switch_randomization,
    pool_partner_columns,
    shuffle_partner_identities_on_side,
    shuffle_side_membership,
)
from .fitted_null import FittedNullSample
from .full_search_null import (
    CALLBACK_IDENTITY_SCHEMA,
    CALLBACK_PROTOCOL,
    FINAL_ARTIFACT_SCHEMA,
    MODEL_TRIPLET,
    freeze_final_result,
    freeze_fit_score_result,
)
from .model_artifacts import ModelArtifactError, atomic_json
from .policy import EpisodePolicy, digest_bytes, digest_object
from .scientific_models import FittedTriplet, fit_triplet, seed_from


CALLBACK_VERSION = "ep12.production_full_search_callback.v2"
TRIAL_ARTIFACT_SCHEMA = "ep12.full_search_null_trial_artifact.v2"
FULL_NULL_WORKER_LIMIT = 32
NUMERIC_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SELECTION_PROGRAM = {
    "schema_version": "ep12.full_search_null_selection_program.v1",
    "eligible": "promotion_eligible_complete_T_U_M_trials_only",
    "ranking": [
        "maximum_bilateral_directional_floor",
        "maximum_bilateral_mean",
        "minimum_direction_disagreement",
        "maximum_type_count_with_both_directions_above_frozen_margin",
        "lexicographic_configuration_sha256",
    ],
    "reported_statistic": "selected_mean_bilateral_M_minus_best_fair_T_or_U",
    "target_side_model_selection": False,
}
SELECTION_PROGRAM_SHA256 = digest_object(SELECTION_PROGRAM)

COUNTABLE_FALSIFIERS = (
    "strength_and_margin_preserving_graph_randomization",
    "cross_side_component_alignment_permutation",
    "shuffled_partner_and_valid_side_controls",
    "binary_weighted_vocabulary_and_rank_sensitivities",
    "nuisance_and_partner_block_ablations",
    "capacity_matched_noise_controls",
    "leave_one_development_type_and_partner_family_influence",
    "few_type_and_high_strength_neuron_concentration_check",
    "status_endpoint_anatomy_and_missingness_checks",
)


class NullTrialRunnerError(RuntimeError):
    """A production null trial or required control did not complete."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise NullTrialRunnerError("trial artifact is not canonical JSON") from error


def callback_code_identity() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "null_trial_runner.py",
        "full_search_null.py",
        "full_search_runtime.py",
        "null_controller.py",
        "fitted_null.py",
        "model_artifacts.py",
        "policy.py",
        "scientific_models.py",
        "development_trial.py",
        "falsifier_controls.py",
        "falsifier_transforms.py",
    )
    return {name: digest_bytes((package / name).read_bytes()) for name in names}


@dataclass(frozen=True)
class NullTrialInputs:
    """Authenticated, non-final metadata needed to refit generated samples."""

    raw_vocabulary: tuple[str, ...]
    covariates_by_spline_df: Mapping[int, np.ndarray]
    partner_hierarchy: Mapping[str, Mapping[str, Any]]
    fit_settings: Mapping[str, Any]
    artifact_root: Path
    callback_data_sha256: str


def _collapse(
    raw_counts: np.ndarray,
    focal_types: np.ndarray,
    raw_vocabulary: Sequence[str],
    configuration: Mapping[str, Any],
    hierarchy: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    collapsed, vocabulary, _ = _collapse_vocabulary(
        sparse.csr_matrix(np.asarray(raw_counts, dtype=np.int64)),
        np.asarray(focal_types, dtype=object),
        list(map(str, raw_vocabulary)),
        strategy=str(configuration["partner_vocabulary"]),
        minimum_development_type_count=int(
            configuration["partner_minimum_development_type_count"]
        ),
        rare_partner_mass_fraction=float(
            configuration["rare_partner_pooling_mass_fraction"]
        ),
        hierarchy_depth=int(configuration["partner_hierarchy_depth"]),
        partner_hierarchy={key: dict(value) for key, value in hierarchy.items()},
    )
    names = tuple(map(str, vocabulary))
    retained = set(names)
    hierarchy_field = (
        "subclass"
        if int(configuration["partner_hierarchy_depth"]) == 1
        else "class"
    )
    mapping: list[int] = []
    for raw_key in raw_vocabulary:
        if raw_key in retained:
            target = raw_key
        elif raw_key.startswith("typed::") and str(
            configuration["partner_vocabulary"]
        ) == "bounded_hierarchical_partner_categories":
            provider_type = raw_key.removeprefix("typed::")
            label = str(hierarchy.get(provider_type, {}).get(hierarchy_field, "")).strip()
            target = f"hierarchy_{int(configuration['partner_hierarchy_depth'])}::{label or 'unclassified'}"
        elif raw_key.startswith("typed::"):
            target = "other_typed"
        elif raw_key == "missing_annotation":
            target = "missing_annotation"
        elif raw_key.startswith("untyped_status::"):
            _, status, status_label = raw_key.split("::", 2)
            normalized = f"{status} {status_label}".lower()
            if "orphan" in normalized or "leaves" in normalized:
                target = "fragment"
            elif status in {"Assign", "Anchor"} or any(
                token in normalized
                for token in ("hard to trace", "partially traced", "anchor", "assign")
            ):
                target = "proofreading"
            elif status == "Traced":
                target = "untyped"
            else:
                target = "unknown"
        else:
            target = "out_of_vocabulary"
        if target not in retained:
            raise NullTrialRunnerError(f"collapsed vocabulary omits target bin {target}")
        mapping.append(names.index(target))
    dense = collapsed.toarray().astype(np.int64, copy=False)
    check = np.zeros_like(dense)
    for raw_index, collapsed_index in enumerate(mapping):
        check[:, collapsed_index] += raw_counts[:, raw_index]
    if not np.array_equal(dense, check):
        raise NullTrialRunnerError("trial collapse mapping disagrees with production collapse")
    return dense, names, np.asarray(mapping, dtype=np.int64)


def _pooled_counts(
    raw_counts: np.ndarray,
    raw_to_collapsed: np.ndarray,
    selected_raw_columns: Sequence[int],
    collapsed_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected = set(map(int, selected_raw_columns))
    if not selected:
        raise NullTrialRunnerError("retained-mass pooling selected no raw columns")
    augmented = np.column_stack(
        [raw_counts, np.zeros(raw_counts.shape[0], dtype=np.int64)]
    )
    pooled_raw, receipt = pool_partner_columns(
        augmented,
        sorted(selected),
        pool_column=raw_counts.shape[1],
    )
    result = np.zeros((raw_counts.shape[0], collapsed_width + 1), dtype=np.int64)
    for raw_index, collapsed_index in enumerate(raw_to_collapsed):
        result[:, int(collapsed_index)] += pooled_raw[:, raw_index]
    result[:, collapsed_width] = pooled_raw[:, raw_counts.shape[1]]
    if not np.array_equal(result.sum(axis=1), raw_counts.sum(axis=1)):
        raise NullTrialRunnerError("explicit retained-mass pooling changed row totals")
    return result, receipt.as_dict()


def _fit_options(
    configuration: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    required = {"integration_draws", "source_cv_folds"}
    if not required <= set(settings):
        raise NullTrialRunnerError("locked fitter settings are incomplete")
    return {
        "integration_draws": int(settings["integration_draws"]),
        "folds": int(settings["source_cv_folds"]),
        "latent_rank": int(configuration["rank"]),
        "ridge": float(configuration["regularization"]),
        "pseudocount": float(configuration["composition_pseudocount"]),
        "representation": str(configuration["representation"]),
        "covariance_mode": str(configuration["covariance"]),
        "covariance_rank": int(configuration["covariance_rank"]),
        "component_candidates": (int(configuration["component_count"]),),
    }


def _direction_record(
    *,
    fit: FittedTriplet,
    target_counts: np.ndarray,
    target_covariates: np.ndarray,
    target_ids: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = fit.score_target(target_counts, target_covariates)
    t_values = np.asarray(scored["T"], dtype=float)
    u_values = np.asarray(scored["U"], dtype=float)
    m_values = np.asarray(scored["M"], dtype=float)
    reference = (
        u_values if str(scored["reference_model_id"]).startswith("U") else t_values
    )
    delta = m_values - reference
    neurons = [
        {
            "body_id": str(body_id),
            "strength": int(counts.sum()),
            "T": float(t_score),
            "U": float(u_score),
            "M": float(m_score),
            "delta": float(delta_score),
        }
        for body_id, counts, t_score, u_score, m_score, delta_score in zip(
            target_ids,
            target_counts,
            t_values,
            u_values,
            m_values,
            delta,
        )
    ]
    summary = {
        "T": float(np.mean(t_values)),
        "U": float(np.mean(u_values)),
        "M": float(np.mean(m_values)),
        "delta": float(np.mean(delta)),
        "reference_model_id": str(scored["reference_model_id"]),
        "null_generator_model_id": str(scored["null_generator_model_id"]),
    }
    return summary, neurons


def _fit_counts(
    *,
    counts: np.ndarray,
    covariates: np.ndarray,
    focal_types: np.ndarray,
    sides: np.ndarray,
    neuron_ids: np.ndarray,
    configuration: Mapping[str, Any],
    settings: Mapping[str, Any],
    seed_material: Sequence[Any],
    meaningful_margin: float,
    fit_workers: int = 1,
) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    covariate_values = np.asarray(covariates, dtype=float)
    if values.ndim != 2 or covariate_values.shape[0] != values.shape[0]:
        raise NullTrialRunnerError("trial counts/covariates have incompatible rows")
    if np.any(values < 0) or not np.all(np.isfinite(covariate_values)):
        raise NullTrialRunnerError("trial arrays are negative or nonfinite")
    options = _fit_options(configuration, settings)
    component_count = int(configuration["component_count"])
    minimum_source = component_count * int(settings["source_cv_folds"])
    if isinstance(fit_workers, bool) or not isinstance(fit_workers, int):
        raise NullTrialRunnerError("fit worker count must be an integer")
    if not 1 <= fit_workers <= FULL_NULL_WORKER_LIMIT:
        raise NullTrialRunnerError("fit worker count is outside the frozen 32-core limit")
    if fit_workers > 1:
        invalid_threads = {
            name: os.environ.get(name)
            for name in NUMERIC_THREAD_ENVIRONMENT
            if os.environ.get(name) != "1"
        }
        if invalid_threads:
            raise NullTrialRunnerError(
                "parallel null fitting requires every numeric thread environment "
                f"to equal 1: {invalid_threads}"
            )
    records: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    provider_jobs: list[
        tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for provider_type in sorted(set(map(str, focal_types))):
        type_mask = np.asarray(focal_types == provider_type)
        left = type_mask & (sides == "L")
        right = type_mask & (sides == "R")
        left_counts = values[left]
        right_counts = values[right]
        reason: str | None = None
        if len(left_counts) < minimum_source or len(right_counts) < minimum_source:
            reason = "source_side_below_frozen_support"
        elif np.any(left_counts.sum(axis=1) <= 0) or np.any(
            right_counts.sum(axis=1) <= 0
        ):
            reason = "zero_observed_target_count"
        if reason is not None:
            unscorable.append(
                {
                    "provider_type": provider_type,
                    "reason_code": reason,
                    "left_neurons": int(left.sum()),
                    "right_neurons": int(right.sum()),
                }
            )
            continue
        provider_jobs.append(
            (provider_type, left, right, left_counts, right_counts)
        )

    def fit_provider(
        job: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> dict[str, Any]:
        provider_type, left, right, left_counts, right_counts = job
        base_seed = seed_from(*seed_material, provider_type)
        try:
            left_fit = fit_triplet(
                left_counts,
                covariate_values[left],
                seed=seed_from(base_seed, "left-source"),
                **options,
            )
            right_fit = fit_triplet(
                right_counts,
                covariate_values[right],
                seed=seed_from(base_seed, "right-source"),
                **options,
            )
            left_summary, right_neurons = _direction_record(
                fit=left_fit,
                target_counts=right_counts,
                target_covariates=covariate_values[right],
                target_ids=neuron_ids[right],
            )
            right_summary, left_neurons = _direction_record(
                fit=right_fit,
                target_counts=left_counts,
                target_covariates=covariate_values[left],
                target_ids=neuron_ids[left],
            )
            left_contrasts = component_partner_contrasts(
                basis=left_fit.M.basis,
                component_means=left_fit.M.prior_means,
                component_weights=left_fit.M.prior_weights,
            )
            right_contrasts = component_partner_contrasts(
                basis=right_fit.M.basis,
                component_means=right_fit.M.prior_means,
                component_weights=right_fit.M.prior_weights,
            )
        except Exception as error:
            raise NullTrialRunnerError(
                f"T/U/M fit failed for provider type {provider_type}: {error}"
            ) from error
        return {
            "provider_type": provider_type,
            "left_neurons": int(left.sum()),
            "right_neurons": int(right.sum()),
            "left_to_right_delta": left_summary["delta"],
            "right_to_left_delta": right_summary["delta"],
            "left_to_right_scores": {
                key: left_summary[key] for key in ("T", "U", "M")
            },
            "right_to_left_scores": {
                key: right_summary[key] for key in ("T", "U", "M")
            },
            "target_neuron_scores": {
                "left_to_right": right_neurons,
                "right_to_left": left_neurons,
            },
            "source_fit_metadata": {
                "left": {
                    "U": left_fit.U.model_id,
                    "M": left_fit.M.model_id,
                    "reference": left_fit.reference_model_id,
                    "null_generator": left_fit.null_generator_model_id,
                },
                "right": {
                    "U": right_fit.U.model_id,
                    "M": right_fit.M.model_id,
                    "reference": right_fit.reference_model_id,
                    "null_generator": right_fit.null_generator_model_id,
                },
            },
            "left": left_contrasts,
            "right": right_contrasts,
        }

    provider_workers = min(fit_workers, max(1, len(provider_jobs)))
    if provider_workers == 1:
        records = [fit_provider(job) for job in provider_jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=provider_workers,
            thread_name_prefix="ep12-null-provider",
        ) as pool:
            # Ordered map makes every scientific artifact worker invariant.
            records = list(pool.map(fit_provider, provider_jobs))
    if not records:
        raise NullTrialRunnerError("trial has no valid provider-type comparison")
    directional = np.asarray(
        [
            [record["left_to_right_delta"], record["right_to_left_delta"]]
            for record in records
        ],
        dtype=float,
    )
    objective = {
        "left_to_right": float(np.mean(directional[:, 0])),
        "right_to_left": float(np.mean(directional[:, 1])),
        "bilateral": float(np.mean(directional)),
        "bilateral_floor": float(
            min(np.mean(directional[:, 0]), np.mean(directional[:, 1]))
        ),
        "direction_disagreement": float(
            abs(np.mean(directional[:, 0]) - np.mean(directional[:, 1]))
        ),
        "types_both_above_margin": int(
            np.sum(np.all(directional > float(meaningful_margin), axis=1))
        ),
    }
    if not all(math.isfinite(float(value)) for value in objective.values()):
        raise NullTrialRunnerError("trial produced a nonfinite selection objective")
    result = {
        "objective": objective,
        "valid_type_count": len(records),
        "unscorable_type_count": len(unscorable),
        "unscorable": unscorable,
        "records": records,
        "complete_T_U_M_result": True,
        "result_sha256": "pending",
    }
    result["result_sha256"] = digest_object(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


class ProductionFullSearchCallback:
    """Lock-bound callback that performs real fits and complete controls."""

    def __init__(self, *, policy: EpisodePolicy, inputs: NullTrialInputs):
        self.policy = policy
        self.inputs = inputs
        self._objectives: dict[tuple[int, str], dict[str, Any]] = {}
        self._handlers: dict[
            str,
            Callable[
                [FittedNullSample, Mapping[str, Any], np.ndarray, np.ndarray],
                tuple[dict[str, Any], dict[str, Any]],
            ],
        ] = {
            "strength_and_margin_preserving_graph_randomization": self._strength_control,
            "cross_side_component_alignment_permutation": self._alignment_control,
            "shuffled_partner_and_valid_side_controls": self._shuffle_control,
            "binary_weighted_vocabulary_and_rank_sensitivities": self._binary_control,
            "nuisance_and_partner_block_ablations": self._ablation_control,
            "capacity_matched_noise_controls": self._noise_control,
            "leave_one_development_type_and_partner_family_influence": self._leave_one_control,
            "few_type_and_high_strength_neuron_concentration_check": self._concentration_control,
            "status_endpoint_anatomy_and_missingness_checks": self._status_control,
        }
        if tuple(self._handlers) != COUNTABLE_FALSIFIERS:
            raise NullTrialRunnerError("falsifier registry order or coverage changed")
        if not self.inputs.raw_vocabulary:
            raise NullTrialRunnerError("callback raw vocabulary is empty")
        if not self.inputs.partner_hierarchy:
            raise NullTrialRunnerError("callback partner hierarchy is unavailable")
        for spline_df in self.policy.grammar_choices["nuisance_spline_df"]:
            value = int(spline_df)
            if value not in self.inputs.covariates_by_spline_df:
                raise NullTrialRunnerError(
                    f"callback lacks frozen covariates for spline df {value}"
                )

    def identity(self) -> Mapping[str, Any]:
        implementation_sha256 = digest_object(
            {
                "code": callback_code_identity(),
                "callback_data_sha256": self.inputs.callback_data_sha256,
                "selection_program": SELECTION_PROGRAM,
                "falsifier_implementation_spec": implementation_spec(),
            }
        )
        return {
            "schema_version": CALLBACK_IDENTITY_SCHEMA,
            "protocol": CALLBACK_PROTOCOL,
            "callback_id": CALLBACK_VERSION,
            "callback_version": "1",
            "implementation_sha256": implementation_sha256,
            "selection_program_sha256": SELECTION_PROGRAM_SHA256,
            "supported_model_triplet": list(MODEL_TRIPLET),
            "supported_falsifier_ids": list(self._handlers),
            "development_only": True,
            "final_connectivity_accessed": False,
            "nested_null_searches_supported": False,
        }

    def _arrays(
        self, sample: FittedNullSample, configuration: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        validate_development_configuration(self.policy, configuration)
        spline_df = int(configuration["nuisance_spline_df"])
        covariates = np.asarray(
            self.inputs.covariates_by_spline_df[spline_df], dtype=float
        )
        rows = len(sample.neuron_ids)
        if covariates.shape[0] != rows:
            raise NullTrialRunnerError("callback covariates differ from sample rows")
        focal_types = np.asarray(sample.focal_types, dtype=object)
        sides = np.asarray(sample.sides, dtype=object)
        neuron_ids = np.asarray(sample.neuron_ids, dtype=object)
        return covariates, focal_types, sides, neuron_ids

    def _fit_raw(
        self,
        *,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        raw_counts: np.ndarray,
        sides: np.ndarray,
        branch_id: str,
        configuration: Mapping[str, Any] | None = None,
        covariates_override: np.ndarray | None = None,
        fit_workers: int | None = None,
    ) -> dict[str, Any]:
        config = dict(configuration or trial["configuration"])
        covariates, focal_types, _, neuron_ids = self._arrays(sample, config)
        if covariates_override is not None:
            covariates = np.asarray(covariates_override, dtype=float)
        counts, _, _ = _collapse(
            np.asarray(raw_counts, dtype=np.int64),
            focal_types,
            self.inputs.raw_vocabulary,
            config,
            self.inputs.partner_hierarchy,
        )
        return self._fit_direct(
            sample=sample,
            trial=trial,
            counts=counts,
            covariates=covariates,
            focal_types=focal_types,
            sides=sides,
            neuron_ids=neuron_ids,
            branch_id=branch_id,
            configuration=config,
            fit_workers=fit_workers,
        )

    def _fit_direct(
        self,
        *,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        counts: np.ndarray,
        covariates: np.ndarray,
        focal_types: np.ndarray,
        sides: np.ndarray,
        neuron_ids: np.ndarray,
        branch_id: str,
        configuration: Mapping[str, Any],
        fit_workers: int | None = None,
    ) -> dict[str, Any]:
        if fit_workers is None:
            fit_workers = self._provider_worker_count()
        return _fit_counts(
            counts=counts,
            covariates=covariates,
            focal_types=focal_types,
            sides=sides,
            neuron_ids=neuron_ids,
            configuration=configuration,
            settings=self.inputs.fit_settings,
            seed_material=(
                "ep12-full-search-null-fit",
                sample.replicate_seed,
                trial["configuration_sha256"],
                branch_id,
            ),
            meaningful_margin=self.policy.meaningful_margin,
            fit_workers=fit_workers,
        )

    def _base_fit(
        self, sample: FittedNullSample, trial: Mapping[str, Any]
    ) -> dict[str, Any]:
        _, _, sides, _ = self._arrays(sample, trial["configuration"])
        return self._fit_raw(
            sample=sample,
            trial=trial,
            raw_counts=sample.raw_counts,
            sides=sides,
            branch_id="primary",
        )

    def _provider_worker_count(self) -> int:
        """Return the bounded Slurm allocation for provider-type fits."""

        try:
            allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        except ValueError as error:
            raise NullTrialRunnerError(
                "SLURM_CPUS_PER_TASK is not an integer"
            ) from error
        maximum = int(self.policy.cpu_cores_per_trial_maximum)
        if allocation < 1 or maximum < 1 or allocation > maximum or maximum > 32:
            raise NullTrialRunnerError(
                "null-control worker allocation violates the frozen 32-core limit"
            )
        if allocation > 1:
            invalid_threads = {
                name: os.environ.get(name)
                for name in NUMERIC_THREAD_ENVIRONMENT
                if os.environ.get(name) != "1"
            }
            if invalid_threads:
                raise NullTrialRunnerError(
                    "parallel null fitting requires every numeric thread environment "
                    f"to equal 1: {invalid_threads}"
                )
        return allocation

    def _execute_fit_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        branch_id = str(job.get("branch_id", ""))
        arguments = job.get("arguments")
        if (
            not branch_id
            or job.get("kind") not in {"direct", "raw"}
            or not isinstance(arguments, Mapping)
            or arguments.get("branch_id") != branch_id
        ):
            raise NullTrialRunnerError("null-control fit job is malformed")
        try:
            if job["kind"] == "raw":
                return self._fit_raw(**dict(arguments))
            return self._fit_direct(**dict(arguments))
        except Exception as error:
            if isinstance(error, NullTrialRunnerError):
                raise
            raise NullTrialRunnerError(
                f"null-control fit branch failed: {branch_id}: {error}"
            ) from error

    def _run_fit_jobs(
        self, jobs: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Run branches in order; each branch parallelizes over provider types."""

        normalized = [dict(job) for job in jobs]
        if not normalized:
            return [], 1
        branch_ids = [str(job.get("branch_id", "")) for job in normalized]
        if not all(branch_ids) or len(branch_ids) != len(set(branch_ids)):
            raise NullTrialRunnerError("null-control fit branch ids are not unique")
        # Branch-level parallelism would nest with the provider-type pool and
        # oversubscribe the 32-core allocation.  Serial branch execution also
        # makes the worst-case resource model exact: one complete branch at a
        # time, with every allocated core available inside that branch.
        results = [self._execute_fit_job(job) for job in normalized]
        return results, 1

    def _manifest(
        self,
        *,
        control_id: str,
        branch_id: str,
        result: Mapping[str, Any],
        details: Mapping[str, Any] | None = None,
        applicability: bool = True,
        conservation: bool = True,
        capacity: bool = True,
    ) -> dict[str, Any]:
        return make_branch_manifest(
            control_id=control_id,
            branch_id=branch_id,
            branch_kind="executed_T_U_M_or_registered_analytic_branch",
            complete_triplet_sha256=str(result["result_sha256"]),
            applicability_checks_passed=applicability,
            conservation_checks_passed=conservation,
            capacity_checks_passed=capacity,
            artifacts={"fit_result_sha256": str(result["result_sha256"])},
            details=details,
        )

    def _receipt(
        self,
        control_id: str,
        expected: Sequence[str],
        branches: Sequence[Mapping[str, Any]],
        *,
        fit_branch_count: int,
        analytic_branch_count: int,
        max_concurrent_fit_workers: int,
    ) -> dict[str, Any]:
        if (
            fit_branch_count < 1
            or analytic_branch_count < 0
            or max_concurrent_fit_workers < 1
            or max_concurrent_fit_workers > fit_branch_count
        ):
            raise NullTrialRunnerError("null-control execution accounting is invalid")
        receipt = composite_control_receipt(
            control_id=control_id,
            expected_branch_ids=list(expected),
            branches=list(branches),
            implementation_spec_sha256=implementation_spec()["spec_sha256"],
        )
        receipt.update(
            {
                "fit_execution_backend": "serial_branches_provider_type_thread_pool",
                "fit_branch_count": int(fit_branch_count),
                "analytic_branch_count": int(analytic_branch_count),
                "max_concurrent_fit_workers": int(max_concurrent_fit_workers),
                "max_provider_fit_workers": self._provider_worker_count(),
                "prespecified_result_order_preserved": True,
            }
        )
        receipt["receipt_sha256"] = digest_object(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        return receipt

    def _strength_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        strata = np.asarray(
            [f"{provider_type}::{side}" for provider_type, side in zip(focal_types, sides)],
            dtype=object,
        )
        transformed, receipt = margin_preserving_switch_randomization(
            sample.raw_counts,
            strata,
            seed=seed_from(sample.replicate_seed, trial["trial_id"], "switch"),
        )
        passed = bool(
            receipt.row_totals_preserved
            and receipt.column_totals_preserved
            and receipt.total_mass_preserved
            and receipt.input_sha256 != receipt.output_sha256
            and int(receipt.details.get("completed_switches", 0)) > 0
        )
        if not passed:
            raise NullTrialRunnerError("margin-preserving control failed conservation")
        result = self._fit_raw(
            sample=sample,
            trial=trial,
            raw_counts=transformed,
            sides=sides,
            branch_id="margin_preserving_switch",
        )
        return result, {
            "falsifier_id": trial["falsifier_id"],
            "control_passed": True,
            "transforms": [receipt.as_dict()],
            "complete_T_U_M_result_sha256": result["result_sha256"],
            "fit_execution_backend": "serial_branches_provider_type_thread_pool",
            "fit_branch_count": 1,
            "analytic_branch_count": 0,
            "max_concurrent_fit_workers": 1,
            "max_provider_fit_workers": self._provider_worker_count(),
        }

    def _shuffle_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        typed = [
            index
            for index, value in enumerate(self.inputs.raw_vocabulary)
            if value.startswith("typed::")
        ]
        seed = seed_from(sample.replicate_seed, trial["trial_id"], "shuffle")
        counts, count_receipt = shuffle_partner_identities_on_side(
            sample.raw_counts, sides, typed, shuffled_side="R", seed=seed
        )
        shuffled_sides, side_receipt = shuffle_side_membership(
            sides, focal_types, seed=(seed + 1) % (2**32)
        )
        passed = bool(
            count_receipt.row_totals_preserved
            and count_receipt.total_mass_preserved
            and count_receipt.input_sha256 != count_receipt.output_sha256
            and int(side_receipt["changed_row_count"]) > 0
        )
        if not passed:
            raise NullTrialRunnerError("partner/side shuffle was inapplicable")
        result = self._fit_raw(
            sample=sample,
            trial=trial,
            raw_counts=counts,
            sides=np.asarray(shuffled_sides, dtype=object),
            branch_id="partner_and_side_shuffle",
        )
        return result, {
            "falsifier_id": trial["falsifier_id"],
            "control_passed": True,
            "transforms": [count_receipt.as_dict(), side_receipt],
            "complete_T_U_M_result_sha256": result["result_sha256"],
            "fit_execution_backend": "serial_branches_provider_type_thread_pool",
            "fit_branch_count": 1,
            "analytic_branch_count": 0,
            "max_concurrent_fit_workers": 1,
            "max_provider_fit_workers": self._provider_worker_count(),
        }

    def _alignment_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del focal_types, sides
        result = self._base_fit(sample, trial)
        alignment = cross_side_alignment_permutation(
            result["records"],
            seed=seed_from(sample.replicate_seed, trial["trial_id"], "alignment"),
        )
        expected = [str(branch["branch_id"]) for branch in alignment["branches"]]
        branches = [
            self._manifest(
                control_id=str(trial["falsifier_id"]),
                branch_id=str(branch["branch_id"]),
                result=result,
                details={"alignment_branch_sha256": digest_object(branch)},
            )
            for branch in alignment["branches"]
        ]
        return result, self._receipt(
            str(trial["falsifier_id"]),
            expected,
            branches,
            fit_branch_count=1,
            analytic_branch_count=len(expected),
            max_concurrent_fit_workers=1,
        )

    def _noise_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config = dict(trial["configuration"])
        covariates, _, _, neuron_ids = self._arrays(sample, config)
        counts, _, _ = _collapse(
            sample.raw_counts,
            focal_types,
            self.inputs.raw_vocabulary,
            config,
            self.inputs.partner_hierarchy,
        )
        noise, transform = pooled_multinomial_noise(
            counts,
            seed=seed_from(sample.replicate_seed, trial["trial_id"], "noise"),
        )
        expected = ["weighted_baseline", "pooled_multinomial_noise"]
        jobs = [
            {
                "kind": "direct",
                "branch_id": branch_id,
                "arguments": {
                    "sample": sample,
                    "trial": trial,
                    "counts": branch_counts,
                    "covariates": covariates,
                    "focal_types": focal_types,
                    "sides": sides,
                    "neuron_ids": neuron_ids,
                    "branch_id": branch_id,
                    "configuration": config,
                },
            }
            for branch_id, branch_counts in zip(expected, (counts, noise))
        ]
        (baseline, generated), workers = self._run_fit_jobs(jobs)
        branches = [
            self._manifest(
                control_id=str(trial["falsifier_id"]),
                branch_id=expected[0],
                result=baseline,
            ),
            self._manifest(
                control_id=str(trial["falsifier_id"]),
                branch_id=expected[1],
                result=generated,
                details=transform,
                conservation=bool(transform["row_totals_preserved"]),
                capacity=bool(transform["same_shape"]),
            ),
        ]
        return baseline, self._receipt(
            str(trial["falsifier_id"]),
            expected,
            branches,
            fit_branch_count=len(jobs),
            analytic_branch_count=0,
            max_concurrent_fit_workers=workers,
        )

    def _binary_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        control_id = str(trial["falsifier_id"])
        expected: list[str] = []
        jobs: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        scheduled = dict(trial["configuration"])
        weighted = np.asarray(sample.raw_counts, dtype=np.int64)
        binary = (np.asarray(sample.raw_counts) > 0).astype(np.int64)

        # The frozen implementation specification defines exactly fourteen
        # fitted branches: one scheduled weighted baseline, one scheduled
        # binary sensitivity, and twelve weighted vocabulary/rank branches.
        # It does not define a full binary vocabulary/rank Cartesian product.
        branch_specs: list[tuple[str, np.ndarray, dict[str, Any]]] = [
            (
                "weighted_baseline",
                weighted,
                {
                    "count_mode": "weighted",
                    "partner_vocabulary": scheduled["partner_vocabulary"],
                    "rank": int(scheduled["rank"]),
                },
            ),
            (
                "binary",
                binary,
                {
                    "count_mode": "binary",
                    "partner_vocabulary": scheduled["partner_vocabulary"],
                    "rank": int(scheduled["rank"]),
                },
            ),
        ]
        for vocabulary in self.policy.grammar_choices["partner_vocabulary"]:
            for rank in self.policy.grammar_choices["rank"]:
                branch_specs.append(
                    (
                        f"vocabulary::{vocabulary}__rank::{int(rank)}",
                        weighted,
                        {
                            "count_mode": "weighted",
                            "partner_vocabulary": vocabulary,
                            "rank": int(rank),
                        },
                    )
                )

        for branch_id, source, detail in branch_specs:
            config = dict(scheduled)
            config["partner_vocabulary"] = detail["partner_vocabulary"]
            config["rank"] = int(detail["rank"])
            validate_development_configuration(self.policy, config)
            expected.append(branch_id)
            details.append(detail)
            jobs.append(
                {
                    "kind": "raw",
                    "branch_id": branch_id,
                    "arguments": {
                        "sample": sample,
                        "trial": trial,
                        "raw_counts": source,
                        "sides": sides,
                        "branch_id": branch_id,
                        "configuration": config,
                    },
                }
            )
        results, workers = self._run_fit_jobs(jobs)
        branches: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for branch_id, detail, result in zip(expected, details, results):
            mode = str(detail["count_mode"])
            vocabulary = detail["partner_vocabulary"]
            rank = int(detail["rank"])
            if branch_id == "weighted_baseline":
                selected = result
            branches.append(
                self._manifest(
                    control_id=control_id,
                    branch_id=branch_id,
                    result=result,
                    details=detail,
                )
            )
        if selected is None:
            raise NullTrialRunnerError("weighted control omitted the scheduled branch")
        return selected, self._receipt(
            control_id,
            expected,
            branches,
            fit_branch_count=len(jobs),
            analytic_branch_count=0,
            max_concurrent_fit_workers=workers,
        )

    def _family_fit_jobs(
        self,
        *,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        config = dict(trial["configuration"])
        covariates, _, _, neuron_ids = self._arrays(sample, config)
        collapsed, _, raw_to_collapsed = _collapse(
            sample.raw_counts,
            focal_types,
            self.inputs.raw_vocabulary,
            config,
            self.inputs.partner_hierarchy,
        )
        by_key = {
            key: index for index, key in enumerate(self.inputs.raw_vocabulary)
        }
        expected: list[str] = []
        jobs: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for family in partner_family_inventory(
            self.inputs.partner_hierarchy, self.inputs.raw_vocabulary
        ):
            selected = [by_key[key] for key in family["raw_partner_keys"]]
            pooled, transform = _pooled_counts(
                sample.raw_counts,
                raw_to_collapsed,
                selected,
                collapsed.shape[1],
            )
            branch_id = str(family["branch_id"])
            expected.append(branch_id)
            jobs.append(
                {
                    "kind": "direct",
                    "branch_id": branch_id,
                    "arguments": {
                        "sample": sample,
                        "trial": trial,
                        "counts": pooled,
                        "covariates": covariates,
                        "focal_types": focal_types,
                        "sides": sides,
                        "neuron_ids": neuron_ids,
                        "branch_id": branch_id,
                        "configuration": config,
                    },
                }
            )
            metadata.append(
                {
                    "details": {"family": family, "transform": transform},
                    "conservation": bool(transform["row_totals_preserved"]),
                }
            )
        return expected, jobs, metadata

    def _ablation_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        control_id = str(trial["falsifier_id"])
        config = dict(trial["configuration"])
        covariates, _, _, neuron_ids = self._arrays(sample, config)
        counts, _, _ = _collapse(
            sample.raw_counts,
            focal_types,
            self.inputs.raw_vocabulary,
            config,
            self.inputs.partner_hierarchy,
        )
        jobs: list[dict[str, Any]] = [
            {
                "kind": "direct",
                "branch_id": "ablation_reference",
                "arguments": {
                    "sample": sample,
                    "trial": trial,
                    "counts": counts,
                    "covariates": covariates,
                    "focal_types": focal_types,
                    "sides": sides,
                    "neuron_ids": neuron_ids,
                    "branch_id": "ablation_reference",
                    "configuration": config,
                },
            }
        ]
        expected: list[str] = []
        metadata: list[dict[str, Any]] = []
        for block_id in self.policy.nuisance_terms:
            transformed, receipt = ablate_nuisance_block(
                covariates,
                spline_df=int(config["nuisance_spline_df"]),
                block_id=str(block_id),
            )
            branch_id = f"nuisance::{block_id}"
            expected.append(branch_id)
            metadata.append({"details": receipt, "conservation": True})
            jobs.append(
                {
                    "kind": "direct",
                    "branch_id": branch_id,
                    "arguments": {
                        "sample": sample,
                        "trial": trial,
                        "counts": counts,
                        "covariates": transformed,
                        "focal_types": focal_types,
                        "sides": sides,
                        "neuron_ids": neuron_ids,
                        "branch_id": branch_id,
                        "configuration": config,
                    },
                }
            )
        family_expected, family_jobs, family_metadata = self._family_fit_jobs(
            sample=sample,
            trial=trial,
            focal_types=focal_types,
            sides=sides,
        )
        expected.extend(family_expected)
        metadata.extend(family_metadata)
        jobs.extend(family_jobs)
        results, workers = self._run_fit_jobs(jobs)
        baseline = results[0]
        branches = [
            self._manifest(
                control_id=control_id,
                branch_id=branch_id,
                result=result,
                details=branch_metadata["details"],
                conservation=bool(branch_metadata["conservation"]),
            )
            for branch_id, branch_metadata, result in zip(
                expected, metadata, results[1:]
            )
        ]
        return baseline, self._receipt(
            control_id,
            expected,
            branches,
            fit_branch_count=len(jobs),
            analytic_branch_count=0,
            max_concurrent_fit_workers=workers,
        )

    def _leave_one_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        control_id = str(trial["falsifier_id"])
        jobs: list[dict[str, Any]] = [
            {
                "kind": "raw",
                "branch_id": "primary",
                "arguments": {
                    "sample": sample,
                    "trial": trial,
                    "raw_counts": sample.raw_counts,
                    "sides": sides,
                    "branch_id": "primary",
                },
            }
        ]
        family_expected, family_jobs, family_metadata = self._family_fit_jobs(
            sample=sample,
            trial=trial,
            focal_types=focal_types,
            sides=sides,
        )
        jobs.extend(family_jobs)
        results, workers = self._run_fit_jobs(jobs)
        baseline = results[0]
        influence = leave_one_type_influence(baseline["records"])
        expected: list[str] = []
        branches: list[dict[str, Any]] = []
        for point in influence["leave_one_type_curve"]:
            branch_id = f"leave_type::{point['omitted_provider_type']}"
            expected.append(branch_id)
            branches.append(
                self._manifest(
                    control_id=control_id,
                    branch_id=branch_id,
                    result=baseline,
                    details=point,
                )
            )
        expected.extend(family_expected)
        branches.extend(
            self._manifest(
                control_id=control_id,
                branch_id=branch_id,
                result=result,
                details=branch_metadata["details"],
                conservation=bool(branch_metadata["conservation"]),
            )
            for branch_id, branch_metadata, result in zip(
                family_expected, family_metadata, results[1:]
            )
        )
        return baseline, self._receipt(
            control_id,
            expected,
            branches,
            fit_branch_count=len(jobs),
            analytic_branch_count=len(influence["leave_one_type_curve"]),
            max_concurrent_fit_workers=workers,
        )

    def _concentration_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del focal_types, sides
        control_id = str(trial["falsifier_id"])
        baseline = self._base_fit(sample, trial)
        type_curve = leave_one_type_influence(baseline["records"])
        neuron_curves = neuron_influence_curves(baseline["records"])
        details = {
            "type_curve": type_curve,
            "neuron_curves": neuron_curves,
            "new_concentration_cutoff": None,
        }
        expected = ["complete_type_curve", "complete_neuron_curve", "complete_strength_curve"]
        branches = [
            self._manifest(
                control_id=control_id,
                branch_id=branch_id,
                result=baseline,
                details=details,
            )
            for branch_id in expected
        ]
        return baseline, self._receipt(
            control_id,
            expected,
            branches,
            fit_branch_count=1,
            analytic_branch_count=len(expected),
            max_concurrent_fit_workers=1,
        )

    def _status_control(
        self,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        focal_types: np.ndarray,
        sides: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        control_id = str(trial["falsifier_id"])
        config = dict(trial["configuration"])
        covariates, _, _, neuron_ids = self._arrays(sample, config)
        counts, _, raw_to_collapsed = _collapse(
            sample.raw_counts,
            focal_types,
            self.inputs.raw_vocabulary,
            config,
            self.inputs.partner_hierarchy,
        )
        jobs: list[dict[str, Any]] = [
            {
                "kind": "direct",
                "branch_id": "status_reference",
                "arguments": {
                    "sample": sample,
                    "trial": trial,
                    "counts": counts,
                    "covariates": covariates,
                    "focal_types": focal_types,
                    "sides": sides,
                    "neuron_ids": neuron_ids,
                    "branch_id": "status_reference",
                    "configuration": config,
                },
            }
        ]
        expected: list[str] = []
        metadata: list[dict[str, Any]] = []
        blocks = nuisance_block_columns(
            spline_df=int(config["nuisance_spline_df"]),
            design_width=int(covariates.shape[1]),
        )
        for block_id in blocks:
            transformed, receipt = ablate_nuisance_block(
                covariates,
                spline_df=int(config["nuisance_spline_df"]),
                block_id=block_id,
            )
            branch_id = f"covariate::{block_id}"
            expected.append(branch_id)
            metadata.append({"details": receipt, "conservation": True})
            jobs.append(
                {
                    "kind": "direct",
                    "branch_id": branch_id,
                    "arguments": {
                        "sample": sample,
                        "trial": trial,
                        "counts": counts,
                        "covariates": transformed,
                        "focal_types": focal_types,
                        "sides": sides,
                        "neuron_ids": neuron_ids,
                        "branch_id": branch_id,
                        "configuration": config,
                    },
                }
            )
        for endpoint in endpoint_branch_inventory(
            self.inputs.raw_vocabulary, sample.raw_counts
        ):
            pooled, transform = _pooled_counts(
                sample.raw_counts,
                raw_to_collapsed,
                endpoint["raw_columns"],
                counts.shape[1],
            )
            branch_id = str(endpoint["branch_id"])
            expected.append(branch_id)
            metadata.append(
                {
                    "details": {"endpoint": endpoint, "transform": transform},
                    "conservation": bool(transform["row_totals_preserved"]),
                }
            )
            jobs.append(
                {
                    "kind": "direct",
                    "branch_id": branch_id,
                    "arguments": {
                        "sample": sample,
                        "trial": trial,
                        "counts": pooled,
                        "covariates": covariates,
                        "focal_types": focal_types,
                        "sides": sides,
                        "neuron_ids": neuron_ids,
                        "branch_id": branch_id,
                        "configuration": config,
                    },
                }
            )
        results, workers = self._run_fit_jobs(jobs)
        baseline = results[0]
        branches = [
            self._manifest(
                control_id=control_id,
                branch_id=branch_id,
                result=result,
                details=branch_metadata["details"],
                conservation=bool(branch_metadata["conservation"]),
            )
            for branch_id, branch_metadata, result in zip(
                expected, metadata, results[1:]
            )
        ]
        return baseline, self._receipt(
            control_id,
            expected,
            branches,
            fit_branch_count=len(jobs),
            analytic_branch_count=0,
            max_concurrent_fit_workers=workers,
        )

    @staticmethod
    def _verify_context(sample: FittedNullSample, context: Mapping[str, Any]) -> None:
        required = {
            "callback_protocol", "procedure_lock_sha256",
            "controller_program_sha256",
            "replicate_index", "replicate_id", "replicate_seed",
            "sample_receipt_sha256",
            "initial_ledger_empty", "nested_null_searches", "development_only",
            "final_connectivity_accessed",
        }
        if set(context) != required:
            raise NullTrialRunnerError("full-search callback context differs from schema")
        if not (
            context["callback_protocol"] == CALLBACK_PROTOCOL
            and isinstance(context["controller_program_sha256"], str)
            and len(context["controller_program_sha256"]) == 64
            and int(context["replicate_index"]) == sample.replicate_index
            and context["replicate_id"] == sample.replicate_id
            and int(context["replicate_seed"]) == sample.replicate_seed
            and isinstance(context["sample_receipt_sha256"], str)
            and len(context["sample_receipt_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in context["sample_receipt_sha256"]
            )
            and context["initial_ledger_empty"] is True
            and int(context["nested_null_searches"]) == 0
            and context["development_only"] is True
            and context["final_connectivity_accessed"] is False
        ):
            raise NullTrialRunnerError("full-search callback context is not locked")

    def fit_score(
        self,
        *,
        sample: FittedNullSample,
        trial: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._verify_context(sample, context)
        configuration = dict(trial["configuration"])
        validate_development_configuration(self.policy, configuration)
        if digest_object(configuration) != trial["configuration_sha256"]:
            raise NullTrialRunnerError("scheduled configuration hash changed")
        covariates, focal_types, sides, _ = self._arrays(sample, configuration)
        del covariates
        falsifier_id = trial["falsifier_id"]
        if falsifier_id is None:
            primary = self._base_fit(sample, trial)
            control: dict[str, Any] = {
                "falsifier_id": None,
                "control_passed": True,
                "identity_branch": True,
            }
        else:
            handler = self._handlers.get(str(falsifier_id))
            if handler is None:
                raise NullTrialRunnerError(f"unsupported falsifier callback: {falsifier_id}")
            primary, control = handler(sample, trial, focal_types, sides)
            if control.get("control_passed") is not True:
                raise NullTrialRunnerError("falsifier callback did not pass execution gates")
        executed_complete_fit_count = int(control.get("fit_branch_count", 1))
        if executed_complete_fit_count < 1:
            raise NullTrialRunnerError(
                "control receipt lacks positive complete T/U/M fit accounting"
            )
        objective = dict(primary["objective"])
        artifact = {
            "schema_version": TRIAL_ARTIFACT_SCHEMA,
            "callback_identity_sha256": digest_object(self.identity()),
            "replicate_index": sample.replicate_index,
            "replicate_id": sample.replicate_id,
            "replicate_seed": sample.replicate_seed,
            "valid_trial_ordinal": int(trial["valid_trial_ordinal"]),
            "trial_id": str(trial["trial_id"]),
            "configuration_sha256": str(trial["configuration_sha256"]),
            "primary_fit": primary,
            "control": control,
            "objective": objective,
            "complete_T_U_M_result": True,
            "executed_complete_T_U_M_fit_count": executed_complete_fit_count,
            "executed_model_family_fit_count": executed_complete_fit_count
            * len(MODEL_TRIPLET),
            "development_only": True,
            "final_connectivity_accessed": False,
            "nested_null_searches": 0,
        }
        artifact["artifact_sha256"] = digest_object(artifact)
        path = (
            self.inputs.artifact_root
            / "callback_artifacts"
            / sample.replicate_id
            / (
                f"{int(trial['valid_trial_ordinal']):03d}_"
                f"{trial['configuration_sha256']}.json"
            )
        )
        try:
            atomic_json(path, artifact)
        except ModelArtifactError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise NullTrialRunnerError("cannot reconcile callback artifact") from error
            if existing != artifact:
                raise NullTrialRunnerError("callback artifact changed on resume")
        artifact_file_sha256 = digest_bytes(path.read_bytes())
        self._objectives[(sample.replicate_index, str(trial["trial_id"]))] = objective
        return freeze_fit_score_result(
            replicate_index=sample.replicate_index,
            replicate_id=sample.replicate_id,
            trial=trial,
            objective=objective,
            executed_complete_T_U_M_fit_count=executed_complete_fit_count,
            fit_score_artifact_path=path.relative_to(
                self.inputs.artifact_root
            ).as_posix(),
            fit_score_artifact_sha256=artifact_file_sha256,
        )

    def finalize_search(
        self,
        *,
        sample: FittedNullSample,
        schedule: Mapping[str, Any],
        trial_results: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._verify_context(sample, context)
        trials = schedule.get("trials")
        if not isinstance(trials, list) or len(trials) != len(trial_results):
            raise NullTrialRunnerError("finalization lacks the complete realized schedule")
        eligible: list[tuple[tuple[Any, ...], Mapping[str, Any], Mapping[str, Any]]] = []
        for trial, result in zip(trials, trial_results):
            if (
                result.get("trial_id") != trial.get("trial_id")
                or result.get("configuration_sha256") != trial.get("configuration_sha256")
            ):
                raise NullTrialRunnerError("finalization trial sequence changed")
            if trial.get("promotion_eligible") is not True:
                continue
            objective = result.get("objective")
            if not isinstance(objective, Mapping) or float(result["selection_statistic"]) != float(
                objective["bilateral"]
            ):
                raise NullTrialRunnerError("eligible trial objective is unavailable")
            key = (
                -float(objective["bilateral_floor"]),
                -float(objective["bilateral"]),
                float(objective["direction_disagreement"]),
                -int(objective["types_both_above_margin"]),
                str(trial["configuration_sha256"]),
            )
            eligible.append((key, trial, result))
        if not eligible:
            raise NullTrialRunnerError("realized schedule has no eligible completed trial")
        _, selected_trial, selected_result = min(eligible, key=lambda value: value[0])
        path = (
            self.inputs.artifact_root
            / "callback_artifacts"
            / sample.replicate_id
            / "selection.json"
        )
        artifact_core = {
            "schema_version": FINAL_ARTIFACT_SCHEMA,
            "replicate_index": sample.replicate_index,
            "replicate_id": sample.replicate_id,
            "executed_trial_count": len(trials),
            "realized_schedule_sha256": schedule["schedule_sha256"],
            "selected_trial_id": str(selected_trial["trial_id"]),
            "selected_configuration_sha256": str(
                selected_trial["configuration_sha256"]
            ),
            "selected_statistic": float(selected_result["selection_statistic"]),
            "selection_program_sha256": SELECTION_PROGRAM_SHA256,
            "complete_controller_rerun": True,
            "initial_ledger_empty": True,
            "development_only": True,
            "final_connectivity_accessed": False,
        }
        artifact = {
            **artifact_core,
            "artifact_sha256": digest_object(artifact_core),
        }
        try:
            atomic_json(path, artifact)
        except ModelArtifactError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise NullTrialRunnerError("cannot reconcile final selection artifact") from error
            if existing != artifact:
                raise NullTrialRunnerError("final selection artifact changed on resume")
        final = freeze_final_result(
            replicate_index=sample.replicate_index,
            replicate_id=sample.replicate_id,
            executed_trial_count=len(trials),
            selected_trial_id=str(selected_trial["trial_id"]),
            selected_configuration_sha256=str(
                selected_trial["configuration_sha256"]
            ),
            selected_statistic=float(selected_result["selection_statistic"]),
            selection_program_sha256=SELECTION_PROGRAM_SHA256,
            final_selection_artifact_path=path.relative_to(
                self.inputs.artifact_root
            ).as_posix(),
            final_selection_artifact_sha256=digest_bytes(path.read_bytes()),
        )
        return _json_copy(final)
