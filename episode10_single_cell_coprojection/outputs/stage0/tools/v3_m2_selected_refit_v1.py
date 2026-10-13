#!/usr/bin/env python3
"""Named in-process selected full-development refit for EP10 v3 M2.

This module is entered only by the observed runner after an ordinary terminal
selection.  It keeps the exact live terminal/session authority, freshly runs
the frozen full-development M1 selection and refit, prepares the physical
rank-104 M2 objective, performs the selected family's cold/warm pair without
reopening selection, and writes one mode-0600 completion report.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m1_real_v1 as real_m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import load_v3_m2_observed_inputs_v1 as input_loader  # noqa: E402
import v3_m2_observed_session_v1 as observed  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


REPORT_FILENAME = "selected_refit_report.json"
CHECKPOINT_DIRECTORY = "selected_refit_checkpoints"
REPEAT_OBJECTIVE_TOLERANCE = 1.0e-12
REPEAT_CANDIDATE_PROBABILITY_TOLERANCE = 1.0e-10
REPEAT_PAIR_CONTRAST_TOLERANCE = 1.0e-10
DOMINANCE_RATIO_LIMIT = 0.50
DIRECTION_STABILITY_MINIMUM = 9
MOP_GAIN_FLOOR_BITS_PER_CELL = -0.01
OCTANT_LABELS = tuple(
    f"{x}{y}{z}" for x in range(2) for y in range(2) for z in range(2)
)


class SelectedRefitError(RuntimeError):
    """The terminal-selected full-development refit failed closed."""


class _FullObjective:
    """Opaque registered result of the one authorized fresh-M1 builder."""

    __slots__ = (
        "_seal",
        "theta",
        "prepared",
        "selected_lambda",
        "audit_evidence",
        "__weakref__",
    )

    def __init__(
        self,
        theta: np.ndarray,
        prepared: m2.PreparedM2,
        selected_lambda: float,
        audit_evidence: Mapping[str, Any],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _FULL_OBJECTIVE_SEAL:
            raise SelectedRefitError(
                "full objectives require the authorized fresh-M1 builder"
            )
        self._seal = _seal
        self.theta = theta
        self.prepared = prepared
        self.selected_lambda = selected_lambda
        self.audit_evidence = MappingProxyType(dict(audit_evidence))

    def __copy__(self):
        raise SelectedRefitError("full objectives cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise SelectedRefitError("full objectives cannot be copied")

    def __reduce__(self):
        raise SelectedRefitError("full objectives cannot be serialized")


@dataclass(frozen=True)
class _FullObjectiveBinding:
    authority_reference: weakref.ReferenceType[
        observed.SelectedRefitProcedureAuthorization
    ]
    owner_reference: weakref.ReferenceType[executor.CoreTrialExecutor]
    terminal_reference: weakref.ReferenceType[observed.TerminalSelectionAuthorization]
    selected_lambda: float
    theta_digest: bytes
    prepared_digest: bytes
    audit_digest: bytes


_FULL_OBJECTIVE_SEAL = object()
_FULL_OBJECTIVE_RECORDS: dict[
    int,
    tuple[weakref.ReferenceType[_FullObjective], _FullObjectiveBinding],
] = {}


def _register_full_objective(
    full: _FullObjective,
    binding: _FullObjectiveBinding,
) -> None:
    identity = id(full)

    def remove(reference: weakref.ReferenceType[_FullObjective]) -> None:
        current = _FULL_OBJECTIVE_RECORDS.get(identity)
        if current is not None and current[0] is reference:
            _FULL_OBJECTIVE_RECORDS.pop(identity, None)

    _FULL_OBJECTIVE_RECORDS[identity] = (weakref.ref(full, remove), binding)


def _full_objective_authority_record(
    full: object,
    authority: object,
    owner: object,
    terminal: object,
) -> tuple[np.ndarray, m2.PreparedM2, Mapping[str, Any], bytes]:
    """Private observed-session handshake for one builder-issued token."""

    if type(full) is not _FullObjective or full._seal is not _FULL_OBJECTIVE_SEAL:
        raise SelectedRefitError("full objective token is invalid")
    record = _FULL_OBJECTIVE_RECORDS.get(id(full))
    if record is None or record[0]() is not full:
        raise SelectedRefitError("full objective token was not registered")
    binding = record[1]
    if (
        binding.authority_reference() is not authority
        or binding.owner_reference() is not owner
        or binding.terminal_reference() is not terminal
        or full.selected_lambda != binding.selected_lambda
        or binding.theta_digest != executor._deep_integrity_digest(full.theta)
        or binding.prepared_digest != executor._deep_integrity_digest(full.prepared)
        or binding.audit_digest
        != executor._deep_integrity_digest(full.audit_evidence)
    ):
        raise SelectedRefitError("full objective token changed after issuance")
    digest = executor._authority_binding_digest(
        "selected_refit_full_objective_token_v1",
        id(full),
        id(authority),
        id(owner),
        id(terminal),
        binding.selected_lambda.hex(),
        binding.theta_digest,
        binding.prepared_digest,
        binding.audit_digest,
    )
    return full.theta, full.prepared, full.audit_evidence, digest


@dataclass(frozen=True)
class _FitPair:
    cold: Any
    warm: Any
    specification: Any
    initializer_configuration: controller.Configuration | None
    # Hybrid receipts and their precursor are otherwise weak-only once the
    # solver helper returns.  Keep that authority chain alive through report
    # validation and callback completion.
    authority_chain: tuple[Any, ...] = ()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink():
        raise SelectedRefitError("selected-refit report already exists")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise SelectedRefitError("selected-refit report mode changed after write")
    except SelectedRefitError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise SelectedRefitError("selected-refit report write failed closed") from error
    finally:
        try:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass


def _prepare_checkpoint_root(root: Path) -> Path:
    path = root / CHECKPOINT_DIRECTORY
    if path.parent != root or path.exists() or path.is_symlink():
        raise SelectedRefitError("selected-refit checkpoint route is not pristine")
    try:
        path.mkdir(mode=0o700)
    except OSError as error:
        raise SelectedRefitError("selected-refit checkpoint root cannot be created") from error
    return path


def _fresh_full_m1(
    owner: executor.CoreTrialExecutor,
    terminal: observed.TerminalSelectionAuthorization,
    authority: observed.SelectedRefitProcedureAuthorization,
    output_root: Path,
) -> _FullObjective:
    observed._begin_selected_refit_full_m1_request(owner, terminal, authority)
    try:
        checkpoint_root = _prepare_checkpoint_root(output_root)
        cells, moments, candidates, geometry, pairs = (
            observed._selected_refit_source_request(
                owner,
                terminal,
                authority,
            )
        )
        selection = real_m1.select_outer_lambdas(
            cells,
            moments,
            candidates,
            geometry,
            checkpoint_root,
        )
        if (
            len(selection.warm_cache) != math.comb(real_m1.EXPECTED_GROUPS, 2)
            or selection.boundary_count >= 6
        ):
            raise SelectedRefitError(
                "fresh full-M1 tuning failed its frozen inventory"
            )
        full = real_m1.tune_and_refit_full_development(
            cells,
            moments,
            candidates,
            geometry,
            checkpoint_root,
            selection.warm_cache,
        )
        full_indexes = tuple(range(executor.EXPECTED_PRODUCTION_ROWS))
        bundle = full.full_refit
        selected_lambda = float(full.selected_lambda)
        leave_two = tuple(
            (key, item)
            for key, item in full.warm_cache.items()
            if key.startswith("leave_two__")
        )
        leave_one = tuple(
            (key, item)
            for key, item in full.warm_cache.items()
            if key.startswith("leave_one__")
        )
        all_bundles = tuple(item for _, item in (*leave_two, *leave_one)) + (
            bundle,
        )
        expected_paths: dict[str, tuple[int, ...]] = {}
        for left_index, left in enumerate(m1.GROUP_ORDER[:-1]):
            for right in m1.GROUP_ORDER[left_index + 1 :]:
                expected_paths[f"leave_two__{left}__{right}"] = tuple(
                    index
                    for index, cell in enumerate(cells)
                    if cell.group not in {left, right}
                )
        for held_out in m1.GROUP_ORDER:
            expected_paths[f"leave_one__{held_out}"] = tuple(
                index
                for index, cell in enumerate(cells)
                if cell.group != held_out
            )
        observed_paths = {
            key: item.indexes for key, item in (*leave_two, *leave_one)
        }
        if (
            len(full.warm_cache)
            != math.comb(real_m1.EXPECTED_GROUPS, 2) + real_m1.EXPECTED_GROUPS
            or len(leave_two) != 66
            or len(leave_one) != 12
            or set(selection.warm_cache) != {key for key, _ in leave_two}
            or any(
                selection.warm_cache.get(key) is not item
                for key, item in leave_two
            )
            or len({id(item) for item in all_bundles}) != 79
            or any(item.checkpoint_reused is not False for item in all_bundles)
            or any(item.key != key for key, item in (*leave_two, *leave_one))
            or observed_paths != expected_paths
            or bundle.key != "full_refit"
            or bundle.indexes != full_indexes
            or selected_lambda not in m1.LAMBDAS_DESCENDING
            or selected_lambda not in bundle.fits
        ):
            raise SelectedRefitError("fresh full-M1 refit binding is incomplete")
        optimizer = bundle.fits[selected_lambda]
        theta = np.asarray(optimizer.theta, dtype=float)
        if (
            theta is not optimizer.theta
            or theta.shape != (m1.PARAMETER_DIMENSION,)
            or not np.all(np.isfinite(theta))
        ):
            raise SelectedRefitError("fresh full-M1 coefficient vector is malformed")
        _, prepared = m2.fit_residual_transform(
            cells,
            moments,
            candidates,
            geometry,
            pairs,
            full_indexes,
            expected_rank=executor.EXPECTED_PRODUCTION_RANK,
        )
        m1_prepared = bundle.prepared
        transform_matches = {
            "cell_object_identities": (
                len(prepared.cells) == len(cells)
                and all(
                    prepared_cell is source_cell
                    for prepared_cell, source_cell in zip(
                        prepared.cells, cells, strict=True
                    )
                )
            ),
            "design_matrix": np.array_equal(prepared.m1_data.x, m1_prepared.x),
            "geometry_centers": np.array_equal(
                prepared.m1_data.geometry_centers,
                m1_prepared.geometry_centers,
            ),
            "geometry_scales": np.array_equal(
                prepared.m1_data.geometry_scales,
                m1_prepared.geometry_scales,
            ),
            "equal_group_weights": np.array_equal(
                prepared.m1_data.weights,
                m1_prepared.weights,
            ),
            "soma_mean": np.array_equal(
                prepared.transform.design_transform.soma_mean,
                m1_prepared.design_transform.soma_mean,
            ),
            "soma_scale": np.array_equal(
                prepared.transform.design_transform.soma_scale,
                m1_prepared.design_transform.soma_scale,
            ),
            "M2_geometry_scales": np.array_equal(
                prepared.transform.geometry_scales,
                m1_prepared.geometry_scales,
            ),
        }
        ranks = dict(
            prepared.transform.diagnostics.get(
                "rank_at_tau_over_10_tau_ten_tau", {}
            )
        )
        if (
            prepared.transform.rank != executor.EXPECTED_PRODUCTION_RANK
            or any(value is not True for value in transform_matches.values())
            or ranks
            != {
                "tau_over_10": executor.EXPECTED_PRODUCTION_RANK,
                "tau": executor.EXPECTED_PRODUCTION_RANK,
                "ten_tau": executor.EXPECTED_PRODUCTION_RANK,
            }
        ):
            raise SelectedRefitError(
                "fresh full-M1 coordinates differ from the rank-104 M2 objective"
            )
        path_evidence = [
            {
                "key": key,
                "kind": "leave_two" if key.startswith("leave_two__") else "leave_one",
                "training_indexes": list(item.indexes),
                "bundle_identity": id(item),
                "checkpoint_reused": item.checkpoint_reused,
            }
            for key, item in sorted((*leave_two, *leave_one))
        ]
        audit_evidence = {
            "leave_two_path_count": len(leave_two),
            "leave_one_path_count": len(leave_one),
            "zero_full_refit_count": 1,
            "outer_zero_refit_count": 0,
            "all_79_bundle_identities_unique": True,
            "all_checkpoints_fresh": True,
            "outer_boundary_count": int(selection.boundary_count),
            "training_paths": path_evidence,
            "full_refit": {
                "key": bundle.key,
                "training_indexes": list(bundle.indexes),
                "bundle_identity": id(bundle),
                "checkpoint_reused": bundle.checkpoint_reused,
            },
            "M1_M2_full_transform_exact_matches": transform_matches,
            "rank_diagnostics": {
                "rank_at_tau_over_10_tau_ten_tau": ranks,
                "base_rank_tolerance": float(
                    prepared.transform.diagnostics["rank_tolerance"]
                ),
                "rank_stable": bool(
                    prepared.transform.diagnostics["rank_stable"]
                ),
                "rank": int(prepared.transform.diagnostics["rank"]),
                "expected_rank": int(
                    prepared.transform.diagnostics["expected_rank"]
                ),
                "expected_rank_match": bool(
                    prepared.transform.diagnostics["expected_rank_match"]
                ),
            },
            "transform_arrays": {
                "M1_design_matrix": np.asarray(m1_prepared.x).tolist(),
                "M2_embedded_M1_design_matrix": np.asarray(
                    prepared.m1_data.x
                ).tolist(),
                "M1_geometry_centers": np.asarray(
                    m1_prepared.geometry_centers
                ).tolist(),
                "M2_embedded_M1_geometry_centers": np.asarray(
                    prepared.m1_data.geometry_centers
                ).tolist(),
                "M1_geometry_scales": np.asarray(
                    m1_prepared.geometry_scales
                ).tolist(),
                "M2_embedded_M1_geometry_scales": np.asarray(
                    prepared.m1_data.geometry_scales
                ).tolist(),
                "M1_equal_group_weights": np.asarray(m1_prepared.weights).tolist(),
                "M2_embedded_M1_equal_group_weights": np.asarray(
                    prepared.m1_data.weights
                ).tolist(),
                "M1_soma_mean": np.asarray(
                    m1_prepared.design_transform.soma_mean
                ).tolist(),
                "M2_soma_mean": np.asarray(
                    prepared.transform.design_transform.soma_mean
                ).tolist(),
                "M1_soma_scale": np.asarray(
                    m1_prepared.design_transform.soma_scale
                ).tolist(),
                "M2_soma_scale": np.asarray(
                    prepared.transform.design_transform.soma_scale
                ).tolist(),
                "M2_geometry_scales": np.asarray(
                    prepared.transform.geometry_scales
                ).tolist(),
                "M2_projection_F_to_H": np.asarray(
                    prepared.transform.projection_f_to_h
                ).tolist(),
                "M2_pair_residual_scales": np.asarray(
                    prepared.transform.pair_residual_scales
                ).tolist(),
                "M2_range_basis": np.asarray(
                    prepared.transform.range_basis
                ).tolist(),
                "M2_range_projector": np.asarray(
                    prepared.transform.range_projector
                ).tolist(),
                "M2_residual_gram": np.asarray(
                    prepared.transform.residual_gram
                ).tolist(),
                "M2_scaled_residual_gram": np.asarray(
                    prepared.transform.scaled_residual_gram
                ).tolist(),
            },
        }
        result = _FullObjective(
            theta,
            prepared,
            selected_lambda,
            audit_evidence,
            _seal=_FULL_OBJECTIVE_SEAL,
        )
        _register_full_objective(
            result,
            _FullObjectiveBinding(
                authority_reference=weakref.ref(authority),
                owner_reference=weakref.ref(owner),
                terminal_reference=weakref.ref(terminal),
                selected_lambda=selected_lambda,
                theta_digest=executor._deep_integrity_digest(theta),
                prepared_digest=executor._deep_integrity_digest(prepared),
                audit_digest=executor._deep_integrity_digest(
                    result.audit_evidence
                ),
            ),
        )
        observed._complete_selected_refit_full_m1_request(
            owner,
            terminal,
            authority,
            result,
        )
        return result
    except BaseException:
        observed._burn_selected_refit_procedure(authority)
        raise


def _fit_selected_pair(
    owner: executor.CoreTrialExecutor,
    terminal: observed.TerminalSelectionAuthorization,
    authority: observed.SelectedRefitProcedureAuthorization,
    configuration: controller.Configuration,
    full: _FullObjective,
) -> _FitPair:
    candidates = owner.prepared.candidates
    geometry = owner.prepared.geometry
    pairs = owner.prepared.pairs
    maximum_iterations = owner.maximum_iterations

    def run_stage(stage: str, prior: object | None, callback: Any) -> Any:
        began = False
        try:
            observed._begin_selected_refit_fit_stage(
                owner,
                terminal,
                authority,
                full,
                stage,
                prior_result=prior,
            )
            began = True
            result = callback()
            observed._finish_selected_refit_fit_stage(authority, stage, result)
            began = False
            return result
        except BaseException:
            if began:
                try:
                    observed._finish_selected_refit_fit_stage(
                        authority, stage, None
                    )
                except BaseException:
                    observed._burn_selected_refit_procedure(authority)
            else:
                observed._burn_selected_refit_procedure(authority)
            raise

    if configuration.family in controller.CORE_FAMILIES:
        ratio = float(configuration.alpha) if configuration.family == "elastic_net" else 1.0
        maximum = executor._validate_lambda_maximum(
            m2.fold_lambda_maximum(
                full.theta,
                full.prepared,
                candidates,
                geometry,
                pairs,
                configuration.family,
                l1_ratio=ratio,
            )
        )
        penalty = executor.penalty_from_configuration(configuration, maximum)
        cold = run_stage(
            "cold",
            None,
            lambda: m2.fit_m2(
                full.theta,
                full.prepared,
                candidates,
                geometry,
                pairs,
                penalty.specification,
                initial=None,
                maximum_iterations=maximum_iterations,
            ),
        )
        warm = run_stage(
            "warm",
            cold,
            lambda: m2.fit_m2(
                full.theta,
                full.prepared,
                candidates,
                geometry,
                pairs,
                penalty.specification,
                initial=cold.beta,
                maximum_iterations=maximum_iterations,
            ),
        )
        return _FitPair(cold, warm, penalty.specification, None)

    if configuration.family == "low_rank":
        if configuration.rank is None:
            raise SelectedRefitError("selected low-rank configuration lacks rank")
        scales = optional.fold_low_rank_scales(
            full.theta,
            full.prepared,
            candidates,
            geometry,
            pairs,
        )
        specification = optional.LowRankSpec(
            int(configuration.rank),
            float(configuration.rho),
            float(scales.lambda_reference),
        )
        cold = run_stage(
            "cold",
            None,
            lambda: optional.fit_low_rank(
                full.theta,
                full.prepared,
                candidates,
                geometry,
                pairs,
                specification,
                maximum_iterations=maximum_iterations,
            ),
        )
        warm = run_stage(
            "warm",
            cold,
            lambda: optional.refit_low_rank_from_converged(
                full.theta,
                full.prepared,
                candidates,
                geometry,
                pairs,
                specification,
                cold,
                maximum_iterations=maximum_iterations,
            ),
        )
        return _FitPair(cold, warm, specification, None)

    if configuration.family != "sparse_plus_low_rank":
        raise SelectedRefitError("terminal selected an unsupported M2 family")
    if configuration.rank is None or configuration.sparse_rho is None:
        raise SelectedRefitError("selected hybrid configuration is incomplete")
    try:
        receipt = optional.authorize_terminal_full_development_hybrid(
            owner,
            terminal,
            authority,
            full,
        )
        receipt_binding = optional._require_terminal_full_development_hybrid_receipt(
            receipt
        )
        specification = receipt_binding.target_specification_reference()
        initializer = receipt_binding.initializer_configuration_reference()
        if specification is None or initializer is None:
            raise SelectedRefitError(
                "selected hybrid receipt lost its prescribed configuration"
            )
        # Both authority bindings deliberately retain only weak references.
        # Keep the prescribed precursor strongly live until cold and warm have
        # consumed it; a bare call result can otherwise disappear immediately.
        precursor = optional.fit_terminal_full_development_hybrid_precursor(
            receipt,
            maximum_iterations=maximum_iterations,
        )
        cold = optional.fit_terminal_full_development_hybrid_cold(
            receipt,
            maximum_sweeps=maximum_iterations,
        )
        warm = optional.fit_terminal_full_development_hybrid_warm(
            receipt,
            maximum_sweeps=maximum_iterations,
        )
        if precursor is None:
            raise SelectedRefitError("selected hybrid precursor disappeared")
    except BaseException:
        observed._burn_selected_refit_procedure(authority)
        raise
    result = _FitPair(
        cold,
        warm,
        specification,
        initializer,
        (receipt, precursor),
    )
    try:
        _require_hybrid_fit_pair_authority(result)
    except BaseException:
        observed._burn_selected_refit_procedure(authority)
        raise
    return result


def _require_hybrid_fit_pair_authority(
    fit_pair: _FitPair,
) -> Any:
    """Reauthenticate the exact receipt/precursor/cold/warm chain."""

    if len(fit_pair.authority_chain) != 2:
        raise SelectedRefitError(
            "selected hybrid fit lost its retained authority chain"
        )
    receipt, precursor = fit_pair.authority_chain
    try:
        receipt_binding = (
            optional._require_terminal_full_development_hybrid_receipt(receipt)
        )
        cold_binding = optional._require_registered_hybrid_result(
            fit_pair.cold, "production"
        )
        warm_binding = optional._require_registered_hybrid_result(
            fit_pair.warm, "production"
        )
    except optional.LowRankError as error:
        raise SelectedRefitError(
            "selected hybrid authority chain failed final reauthentication"
        ) from error
    if (
        receipt_binding.phase != "warm_ready"
        or receipt_binding.selected_anchor_reference is None
        or receipt_binding.selected_anchor_reference() is not precursor
        or receipt_binding.cold_result_reference is None
        or receipt_binding.cold_result_reference() is not fit_pair.cold
        or receipt_binding.warm_result_reference is None
        or receipt_binding.warm_result_reference() is not fit_pair.warm
        or cold_binding.selected_anchor_reference() is not precursor
        or warm_binding.selected_anchor_reference() is not precursor
        or cold_binding.production_receipt_reference is None
        or cold_binding.production_receipt_reference() is not receipt
        or warm_binding.production_receipt_reference is None
        or warm_binding.production_receipt_reference() is not receipt
    ):
        raise SelectedRefitError(
            "selected hybrid receipt and result chronology changed"
        )
    return receipt_binding


def _group_pair_contrasts(
    beta: np.ndarray,
    full: _FullObjective,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros(m2.PAIR_DIMENSION, dtype=float)
    group_sums = {
        group: np.zeros(m2.PAIR_DIMENSION, dtype=float) for group in m1.GROUP_ORDER
    }
    group_counts = {group: 0 for group in m1.GROUP_ORDER}
    for cell_index, cell in enumerate(full.prepared.cells):
        fitted = m2.candidate_distribution(
            beta,
            full.theta,
            full.prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        baseline = m2.candidate_distribution(
            zero,
            full.theta,
            full.prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        indicators = pairs.basis(cell.k).indicators.astype(float, copy=False)
        group_sums[cell.group] += (
            fitted.probabilities @ indicators
            - baseline.probabilities @ indicators
        )
        group_counts[cell.group] += 1
    if any(group_counts[group] == 0 for group in m1.GROUP_ORDER):
        raise SelectedRefitError("full refit pair contrasts missed a group")
    matrix = np.vstack(
        [group_sums[group] / float(group_counts[group]) for group in m1.GROUP_ORDER]
    )
    aggregate = np.mean(matrix, axis=0)
    if matrix.shape != (12, m2.PAIR_DIMENSION) or not np.all(np.isfinite(matrix)):
        raise SelectedRefitError("full refit pair contrasts are malformed")
    return matrix, aggregate


@dataclass(frozen=True)
class _OuterAudit:
    group_gains: Mapping[str, float]
    leave_one_gains: Mapping[str, float]
    aggregate_gain: float
    dominance_ratio: float
    outer_pair_vector: tuple[float, ...]
    selected_contrast: float
    selected_direction: str
    selected_group_contributions: Mapping[str, float]
    leave_one_contrasts: Mapping[str, float]
    stable_count: int
    raw_cells: tuple[Mapping[str, Any], ...]
    octant_diagnostic: Mapping[str, Any]


def _repeat_evidence(
    full: _FullObjective,
    fit_pair: _FitPair,
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    selected_pair_index: int,
) -> tuple[float, float, Mapping[str, Any]]:
    """Return exhaustive cold/warm prediction witnesses and both maxima."""

    cold_beta = np.asarray(fit_pair.cold.beta, dtype=float)
    warm_beta = np.asarray(fit_pair.warm.beta, dtype=float)
    maximum_probability_error = -1.0
    maximum_probability_witness: Mapping[str, Any] | None = None
    candidate_visits = 0
    for cell_index, cell in enumerate(full.prepared.cells):
        cold = m2.candidate_distribution(
            cold_beta,
            full.theta,
            full.prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        warm = m2.candidate_distribution(
            warm_beta,
            full.theta,
            full.prepared,
            cell_index,
            candidates,
            geometry,
            pairs,
        )
        difference = np.abs(cold.probabilities - warm.probabilities)
        candidate_index = int(np.argmax(difference))
        error = float(difference[candidate_index])
        candidate_visits += int(difference.size)
        if error > maximum_probability_error:
            membership = candidates.basis(cell.k).membership[candidate_index]
            maximum_probability_error = error
            maximum_probability_witness = {
                "cell_index": cell_index,
                "cell_key": str(cell.identifier),
                "K": int(cell.k),
                "candidate_index": candidate_index,
                "candidate_members": [
                    int(index) for index in np.flatnonzero(membership)
                ],
                "cold_probability": float(cold.probabilities[candidate_index]),
                "warm_probability": float(warm.probabilities[candidate_index]),
                "absolute_difference": error,
            }
    expected_visits = sum(
        candidates.basis(cell.k).count for cell in full.prepared.cells
    )
    if (
        maximum_probability_witness is None
        or not math.isfinite(maximum_probability_error)
        or candidate_visits != expected_visits
    ):
        raise SelectedRefitError(
            "cold/warm candidate comparison did not visit every candidate"
        )

    cold_group_pairs, cold_pairs = _group_pair_contrasts(
        cold_beta, full, candidates, geometry, pairs
    )
    warm_group_pairs, warm_pairs = _group_pair_contrasts(
        warm_beta, full, candidates, geometry, pairs
    )
    errors = np.abs(cold_pairs - warm_pairs)
    if not 0 <= selected_pair_index < m2.PAIR_DIMENSION:
        raise SelectedRefitError("selected repeat-pair index is outside 0..119")
    maximum_pair_index = int(np.argmax(errors))
    maximum_pair_error = float(errors[maximum_pair_index])
    selected_pair_error = float(errors[selected_pair_index])
    evidence = {
        "candidate_probability_comparisons": candidate_visits,
        "candidate_probability_maximum_argmax": maximum_probability_witness,
        "cold_equal_group_pair_contrasts_all_120": cold_pairs.tolist(),
        "warm_equal_group_pair_contrasts_all_120": warm_pairs.tolist(),
        "pair_contrast_absolute_errors_all_120": errors.tolist(),
        "pair_contrast_maximum_argmax": {
            "pair_index": maximum_pair_index,
            "pair": controller.PAIR_NAMES[maximum_pair_index],
            "cold_value": float(cold_pairs[maximum_pair_index]),
            "warm_value": float(warm_pairs[maximum_pair_index]),
            "absolute_difference": maximum_pair_error,
        },
        "outer_selected_pair_repeat": {
            "pair_index": selected_pair_index,
            "pair": controller.PAIR_NAMES[selected_pair_index],
            "cold_value": float(cold_pairs[selected_pair_index]),
            "warm_value": float(warm_pairs[selected_pair_index]),
            "absolute_difference": selected_pair_error,
        },
        "cold_group_pair_contrasts_12_by_120": cold_group_pairs.tolist(),
        "warm_group_pair_contrasts_12_by_120": warm_group_pairs.tolist(),
    }
    return maximum_probability_error, maximum_pair_error, evidence


def _outer_crossfit_audit(
    owner: executor.CoreTrialExecutor,
    terminal: observed.TerminalSelectionAuthorization,
    winner: controller.TrialResult,
    candidate_selection: Mapping[str, Any],
    full: _FullObjective,
    position_strata: object,
) -> _OuterAudit:
    """Recompute every scientific completion diagnostic from outer-CV rows."""

    execution = observed._selected_refit_outer_execution(owner, terminal)
    folds = tuple(execution.folds)
    if len(folds) != executor.EXPECTED_FOLDS:
        raise SelectedRefitError("outer execution does not contain 12 folds")
    by_index: dict[int, executor.CellTrialResult] = {}
    group_gains: dict[str, float] = {}
    for ordinal, (group, fold) in enumerate(
        zip(m1.GROUP_ORDER, folds, strict=True), start=1
    ):
        if (
            fold.spec.ordinal != ordinal
            or fold.spec.held_out_group != group
            or group in group_gains
            or not fold.cells
        ):
            raise SelectedRefitError("outer execution fold chronology changed")
        gains: list[float] = []
        for cell_result in fold.cells:
            if (
                type(cell_result) is not executor.CellTrialResult
                or cell_result.group != group
                or cell_result.global_index in by_index
            ):
                raise SelectedRefitError("outer execution cell inventory changed")
            gain = (
                float(cell_result.m2_log_probability_nats)
                - float(cell_result.m1_log_probability_nats)
            ) / m1.LN2
            if not math.isfinite(gain):
                raise SelectedRefitError("outer execution contains a nonfinite gain")
            gains.append(gain)
            by_index[cell_result.global_index] = cell_result
        group_gain = math.fsum(gains) / len(gains)
        if group_gain != float(fold.gain_bits_per_cell):
            raise SelectedRefitError("outer fold gain is not recomputable from cells")
        group_gains[group] = group_gain
    if tuple(sorted(by_index)) != tuple(range(executor.EXPECTED_PRODUCTION_ROWS)):
        raise SelectedRefitError("outer execution did not score all 194 rows once")

    outer_cells = tuple(by_index[index] for index in range(len(by_index)))
    for index, (source, cell_result) in enumerate(
        zip(full.prepared.cells, outer_cells, strict=True)
    ):
        if (
            cell_result.global_index != index
            or cell_result.cell_key != str(source.identifier)
            or cell_result.group != str(source.group)
            or cell_result.k != int(source.k)
        ):
            raise SelectedRefitError(
                "outer execution rows differ from the full objective source"
            )

    rows = tuple(execution.pair_contrast_rows)
    row_by_key = {row.cell_key: row for row in rows}
    if len(rows) != executor.EXPECTED_PRODUCTION_ROWS or len(row_by_key) != len(rows):
        raise SelectedRefitError("outer pair rows are incomplete or duplicated")
    for cell_result in outer_cells:
        row = row_by_key.get(cell_result.cell_key)
        if (
            row is None
            or row.group != cell_result.group
            or row.k != cell_result.k
            or row.values != cell_result.pair_probability_contrasts
        ):
            raise SelectedRefitError("outer score rows and pair rows are not aligned")
    selected_again = dict(controller.select_pair_contrast(rows))
    selection_fields = (
        "pair",
        "pair_index",
        "equal_group_probability_contrast",
        "direction",
        "selection_rule",
        "analysis_label",
        "biological_promotion",
    )
    if any(
        candidate_selection.get(field) != selected_again.get(field)
        for field in selection_fields
    ):
        raise SelectedRefitError("candidate-pair selection is not reproducible")
    outer_pair_vector = executor._aggregate_pair_contrasts(rows)
    if outer_pair_vector != tuple(execution.equal_group_pair_contrasts):
        raise SelectedRefitError("outer equal-group pair vector changed")

    aggregate_gain = math.fsum(
        group_gains[group] for group in m1.GROUP_ORDER
    ) / len(m1.GROUP_ORDER)
    if aggregate_gain != float(winner.gain_bits_per_cell):
        raise SelectedRefitError("outer cell gains do not reproduce the winner score")
    absolute_sum = math.fsum(abs(group_gains[group]) for group in m1.GROUP_ORDER)
    dominance_ratio = (
        0.0
        if absolute_sum == 0.0
        else max(abs(group_gains[group]) for group in m1.GROUP_ORDER)
        / absolute_sum
    )
    leave_one_gains = {
        group: math.fsum(
            group_gains[other] for other in m1.GROUP_ORDER if other != group
        )
        / (len(m1.GROUP_ORDER) - 1)
        for group in m1.GROUP_ORDER
    }

    grouped_rows = {
        group: tuple(row for row in rows if row.group == group)
        for group in m1.GROUP_ORDER
    }
    if any(not grouped_rows[group] for group in m1.GROUP_ORDER):
        raise SelectedRefitError("outer pair rows omit a group")
    group_pair_vectors = {
        group: tuple(
            math.fsum(float(row.values[index]) for row in grouped_rows[group])
            / len(grouped_rows[group])
            for index in range(m2.PAIR_DIMENSION)
        )
        for group in m1.GROUP_ORDER
    }
    rebuilt_pairs = tuple(
        math.fsum(group_pair_vectors[group][index] for group in m1.GROUP_ORDER)
        / len(m1.GROUP_ORDER)
        for index in range(m2.PAIR_DIMENSION)
    )
    if rebuilt_pairs != outer_pair_vector:
        raise SelectedRefitError("outer pair vector is not equal-group recomputable")
    pair_index = candidate_selection.get("pair_index")
    if type(pair_index) is not int or not 0 <= pair_index < m2.PAIR_DIMENSION:
        raise SelectedRefitError("terminal candidate selection lacks its pair index")
    selected_contrast = float(outer_pair_vector[pair_index])
    selected_direction = (
        "positive"
        if selected_contrast > 0.0
        else "negative"
        if selected_contrast < 0.0
        else "zero"
    )
    if (
        selected_again["pair"] != controller.PAIR_NAMES[pair_index]
        or selected_again["direction"] != selected_direction
        or selected_again["equal_group_probability_contrast"] != selected_contrast
    ):
        raise SelectedRefitError("outer selected-pair identity or direction changed")
    direction_sign = (
        1 if selected_contrast > 0.0 else -1 if selected_contrast < 0.0 else 0
    )
    contributions = {
        group: float(group_pair_vectors[group][pair_index])
        for group in m1.GROUP_ORDER
    }
    leave_one_contrasts = {
        group: math.fsum(
            contributions[other] for other in m1.GROUP_ORDER if other != group
        )
        / (len(m1.GROUP_ORDER) - 1)
        for group in m1.GROUP_ORDER
    }
    stable_count = sum(
        direction_sign != 0 and value * direction_sign > 0.0
        for value in leave_one_contrasts.values()
    )

    try:
        position_strata = input_loader._require_position_strata(
            position_strata, tuple(full.prepared.cells)
        )
    except input_loader.ObservedInputError as error:
        raise SelectedRefitError(
            "pre-outcome position strata lost source alignment"
        ) from error
    medians = getattr(position_strata, "medians_xyz", None)
    strata_keys = getattr(position_strata, "cell_keys", None)
    strata_groups = getattr(position_strata, "groups", None)
    strata_octants = getattr(position_strata, "octants", None)
    if (
        type(medians) is not tuple
        or len(medians) != 3
        or any(type(value) is not float or not math.isfinite(value) for value in medians)
        or type(strata_keys) is not tuple
        or type(strata_groups) is not tuple
        or type(strata_octants) is not tuple
        or len(strata_keys) != executor.EXPECTED_PRODUCTION_ROWS
        or len(strata_groups) != executor.EXPECTED_PRODUCTION_ROWS
        or len(strata_octants) != executor.EXPECTED_PRODUCTION_ROWS
        or any(octant not in OCTANT_LABELS for octant in strata_octants)
        or any(
            (strata_keys[index], strata_groups[index])
            != (cell_result.cell_key, cell_result.group)
            for index, cell_result in enumerate(outer_cells)
        )
    ):
        raise SelectedRefitError("pre-outcome position strata lost source alignment")
    cell_gains = tuple(
        (
            float(cell_result.m2_log_probability_nats)
            - float(cell_result.m1_log_probability_nats)
        )
        / m1.LN2
        for cell_result in outer_cells
    )
    group_octant: list[Mapping[str, Any]] = []
    octant_summaries: list[Mapping[str, Any]] = []
    material_octants: list[str] = []
    for octant in OCTANT_LABELS:
        nonempty_group_means: list[float] = []
        pooled: list[float] = []
        for group in m1.GROUP_ORDER:
            values = [
                cell_gains[index]
                for index in range(len(cell_gains))
                if strata_groups[index] == group and strata_octants[index] == octant
            ]
            mean = None if not values else math.fsum(values) / len(values)
            group_octant.append(
                {
                    "group": group,
                    "octant": octant,
                    "count": len(values),
                    "mean_outer_CV_M2_minus_M1_gain_bits_per_cell": mean,
                }
            )
            if mean is not None:
                nonempty_group_means.append(mean)
                pooled.extend(values)
        primary = (
            None
            if not nonempty_group_means
            else math.fsum(nonempty_group_means) / len(nonempty_group_means)
        )
        pooled_mean = None if not pooled else math.fsum(pooled) / len(pooled)
        if primary is not None and primary < MOP_GAIN_FLOOR_BITS_PER_CELL:
            material_octants.append(octant)
        octant_summaries.append(
            {
                "octant": octant,
                "cell_count": len(pooled),
                "nonempty_group_count": len(nonempty_group_means),
                "primary_equal_nonempty_group_gain_bits_per_cell": primary,
                "descriptive_pooled_cell_gain_bits_per_cell": pooled_mean,
            }
        )
    octant_diagnostic = {
        "medians_xyz_fixed_pre_outcome": [float(value) for value in medians],
        "tie_rule": "bit=1 iff coordinate >= exact full-development median",
        "octant_order": list(OCTANT_LABELS),
        "primary_rule": (
            "primary occupied-octant gain = fsum of nonempty group×octant means "
            "divided by count of nonempty groups"
        ),
        "empty_strata": "explicit null",
        "group_by_octant_all_96": group_octant,
        "octant_summaries": octant_summaries,
        "material_boundary_bits_per_cell": MOP_GAIN_FLOOR_BITS_PER_CELL,
        "material_octants_below_boundary": material_octants,
        "scope_limitation": bool(material_octants),
        "scope_limitation_status": (
            "scope_limited" if material_octants else "no_scope_limitation"
        ),
        "diagnostic_only_no_filter_reselection_or_pass_gate": True,
    }
    raw_cells = tuple(
        {
            "global_index": int(cell_result.global_index),
            "cell_key": cell_result.cell_key,
            "group": cell_result.group,
            "K": int(cell_result.k),
            "M1_log_probability_nats": float(
                cell_result.m1_log_probability_nats
            ),
            "M2_log_probability_nats": float(
                cell_result.m2_log_probability_nats
            ),
            "gain_bits_per_cell": cell_gains[index],
            "pre_outcome_octant": strata_octants[index],
        }
        for index, cell_result in enumerate(outer_cells)
    )
    return _OuterAudit(
        group_gains=group_gains,
        leave_one_gains=leave_one_gains,
        aggregate_gain=aggregate_gain,
        dominance_ratio=dominance_ratio,
        outer_pair_vector=outer_pair_vector,
        selected_contrast=selected_contrast,
        selected_direction=selected_direction,
        selected_group_contributions=contributions,
        leave_one_contrasts=leave_one_contrasts,
        stable_count=stable_count,
        raw_cells=raw_cells,
        octant_diagnostic=octant_diagnostic,
    )


def _completion_payload(
    owner: executor.CoreTrialExecutor,
    terminal: observed.TerminalSelectionAuthorization,
    winner: controller.TrialResult,
    candidate_selection: Mapping[str, Any],
    configuration: controller.Configuration,
    full: _FullObjective,
    fit_pair: _FitPair,
    position_strata: object,
) -> Mapping[str, Any]:
    if configuration.family == "sparse_plus_low_rank":
        _require_hybrid_fit_pair_authority(fit_pair)
    elif fit_pair.authority_chain:
        raise SelectedRefitError(
            "selected nonhybrid fit carries hybrid authority state"
        )
    objective_error = abs(float(fit_pair.cold.objective) - float(fit_pair.warm.objective))
    outer = _outer_crossfit_audit(
        owner,
        terminal,
        winner,
        candidate_selection,
        full,
        position_strata,
    )
    pair_index = int(candidate_selection["pair_index"])
    probability_error, pair_error, repeat_evidence = _repeat_evidence(
        full,
        fit_pair,
        owner.prepared.candidates,
        owner.prepared.geometry,
        owner.prepared.pairs,
        pair_index,
    )
    gates = {
        "repeat_objective": objective_error <= REPEAT_OBJECTIVE_TOLERANCE,
        "repeat_candidate_probability": probability_error
        <= REPEAT_CANDIDATE_PROBABILITY_TOLERANCE,
        "repeat_identified_pair_contrast": pair_error
        <= REPEAT_PAIR_CONTRAST_TOLERANCE,
        "no_single_group_dominance": outer.dominance_ratio
        <= DOMINANCE_RATIO_LIMIT,
        "selected_pair_direction_stability": outer.stable_count
        >= DIRECTION_STABILITY_MINIMUM,
        "MOp_source_family": outer.aggregate_gain
        >= MOP_GAIN_FLOOR_BITS_PER_CELL,
    }
    status = "pass" if all(gates.values()) else "failed_completion_gate"
    return {
        "status": status,
        "terminal_authorization_reauthenticated": True,
        "terminal_winner_valid_index": int(winner.trial.valid_index),
        "selected_configuration_key": configuration.key,
        "selected_configuration_family": configuration.family,
        "selected_pair_index": pair_index,
        "selected_pair": candidate_selection.get("pair"),
        "outer_selected_pair_direction": outer.selected_direction,
        "outer_selected_pair_contrast": outer.selected_contrast,
        "fresh_M1": {
            "selected_lambda": full.selected_lambda,
            **dict(full.audit_evidence),
        },
        "full_development_M2": {
            "cells": len(full.prepared.cells),
            "groups": len(m1.GROUP_ORDER),
            "residual_rank": full.prepared.transform.rank,
            "initialization": "frozen_deterministic_then_exact_converged_result",
            "hybrid_precursor_configuration_key": (
                None
                if fit_pair.initializer_configuration is None
                else fit_pair.initializer_configuration.key
            ),
        },
        "repeat_metrics": {
            "cold_objective": float(fit_pair.cold.objective),
            "warm_objective": float(fit_pair.warm.objective),
            "objective_absolute_difference": objective_error,
            "objective_tolerance": REPEAT_OBJECTIVE_TOLERANCE,
            "candidate_probability_maximum_absolute_difference": probability_error,
            "candidate_probability_tolerance": REPEAT_CANDIDATE_PROBABILITY_TOLERANCE,
            "identified_pair_contrast_maximum_absolute_difference": pair_error,
            "identified_pair_contrast_tolerance": REPEAT_PAIR_CONTRAST_TOLERANCE,
            **repeat_evidence,
        },
        "outer_cross_fitted_execution": {
            "source": "authenticated_terminal_winner_outer_cross_fit",
            "cells": list(outer.raw_cells),
            "equal_group_pair_contrasts_all_120": list(
                outer.outer_pair_vector
            ),
            "selected_pair_group_contributions": dict(
                outer.selected_group_contributions
            ),
        },
        "groupwise_M2_minus_M1_gain_bits_per_cell": dict(outer.group_gains),
        "leave_one_group_M2_minus_M1_gain_bits_per_cell": dict(
            outer.leave_one_gains
        ),
        "leave_one_group_selected_pair_contrasts": dict(
            outer.leave_one_contrasts
        ),
        "no_single_group_dominance_ratio": outer.dominance_ratio,
        "no_single_group_dominance_limit": DOMINANCE_RATIO_LIMIT,
        "direction_stable_leave_one_group_count": outer.stable_count,
        "direction_stability_minimum": DIRECTION_STABILITY_MINIMUM,
        "aggregate_MOp_M2_minus_M1_gain_bits_per_cell": outer.aggregate_gain,
        "aggregate_MOp_gain_floor_bits_per_cell": MOP_GAIN_FLOOR_BITS_PER_CELL,
        "target_blind_position_octant_diagnostic": dict(
            outer.octant_diagnostic
        ),
        "completion_gates": gates,
        "selection_reopened": False,
        "opaque_authority_serialized": False,
        "final_unassigned_swc_or_m0_full_opened": False,
    }


def _commit_completion_report(
    output_root: Path,
    owner: executor.CoreTrialExecutor,
    terminal: observed.TerminalSelectionAuthorization,
    authority: observed.SelectedRefitProcedureAuthorization,
    configuration: controller.Configuration,
    full: _FullObjective,
    fit_pair: _FitPair,
    payload: Mapping[str, Any],
) -> None:
    """Final all-family reauthentication immediately before atomic commit."""

    try:
        observed._require_selected_refit_full_objective(
            owner,
            terminal,
            authority,
            full,
        )
        observed._require_selected_refit_fit_pair(
            authority,
            fit_pair.cold,
            fit_pair.warm,
        )
        if configuration.family == "sparse_plus_low_rank":
            _require_hybrid_fit_pair_authority(fit_pair)
        elif fit_pair.authority_chain:
            raise SelectedRefitError(
                "selected nonhybrid report carries hybrid authority state"
            )
        _atomic_json(output_root / REPORT_FILENAME, payload)
    except BaseException:
        observed._burn_selected_refit_procedure(authority)
        raise


class SelectedRefitCoordinator:
    """Named callback retaining only the fixed same-job output route."""

    __slots__ = ("_output_root",)

    def __init__(
        self,
        output_root: str | os.PathLike[str],
    ) -> None:
        self._output_root = Path(os.path.abspath(os.fspath(output_root)))

    def __call__(self, context: object) -> str:
        components = getattr(context, "components", None)
        terminal = getattr(context, "terminal", None)
        authority = getattr(context, "selected_refit_authority", None)
        supplied_winner = getattr(context, "authorized_winner", None)
        candidate_selection = getattr(context, "candidate_selection", None)
        position_strata = getattr(context, "position_strata", None)
        output_root = getattr(context, "output_root", None)
        if (
            components is None
            or type(terminal) is not observed.TerminalSelectionAuthorization
            or type(authority)
            is not observed.SelectedRefitProcedureAuthorization
            or type(supplied_winner) is not controller.TrialResult
            or not isinstance(candidate_selection, Mapping)
            or position_strata is None
            or type(output_root) is not Path
            or output_root != self._output_root
            or not output_root.is_dir()
            or output_root.is_symlink()
        ):
            raise SelectedRefitError("selected-refit callback context is malformed")
        owner = getattr(components, "owner", None)
        if type(owner) is not executor.CoreTrialExecutor:
            raise SelectedRefitError("selected-refit callback owner is malformed")
        try:
            # The builder claims ``full_m1_in_flight`` before it performs any
            # deep reauthentication or creates a checkpoint directory.
            full = _fresh_full_m1(owner, terminal, authority, output_root)
            winner = observed.require_selected_refit_authorization(owner, terminal)
            if winner is not supplied_winner:
                raise SelectedRefitError(
                    "selected-refit callback winner identity changed"
                )
            configuration = winner.trial.configuration
            if type(configuration) is not controller.Configuration:
                raise SelectedRefitError(
                    "selected-refit terminal configuration changed"
                )
            fit_pair = _fit_selected_pair(
                owner,
                terminal,
                authority,
                configuration,
                full,
            )
            # Reauthenticate the builder-issued objective after the warm solve,
            # before any metric is computed or report is committed.  A solver
            # that mutates the prepared objective while fitting therefore burns
            # the one-shot procedure instead of producing completion evidence.
            observed._require_selected_refit_full_objective(
                owner,
                terminal,
                authority,
                full,
            )
            payload = _completion_payload(
                owner,
                terminal,
                winner,
                candidate_selection,
                configuration,
                full,
                fit_pair,
                position_strata,
            )
            _commit_completion_report(
                output_root,
                owner,
                terminal,
                authority,
                configuration,
                full,
                fit_pair,
                payload,
            )
            if payload["status"] != "pass":
                raise SelectedRefitError(
                    "selected full-development refit failed a gate"
                )
            return REPORT_FILENAME
        except BaseException:
            observed._burn_selected_refit_procedure(authority)
            raise


__all__ = (
    "REPORT_FILENAME",
    "SelectedRefitCoordinator",
    "SelectedRefitError",
)
