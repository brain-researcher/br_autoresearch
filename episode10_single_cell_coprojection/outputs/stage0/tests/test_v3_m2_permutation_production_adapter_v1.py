#!/usr/bin/env python3
"""Target-free authority tests for the named whole-set-permutation adapter."""

from __future__ import annotations

import copy
import dataclasses
import gc
import inspect
import math
import sys
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(TEST_PATH.parent) not in sys.path:
    sys.path.insert(0, str(TEST_PATH.parent))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import test_v3_m2_permutation_m1_v1 as m1_tests  # noqa: E402
import v3_m2_permutation_adapter_v1 as mapping  # noqa: E402
import v3_m2_permutation_m1_v1 as permutation_m1  # noqa: E402
import v3_m2_permutation_production_adapter_v1 as production  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


class _Token:
    pass


@dataclasses.dataclass
class _LedgerAttempt:
    ordinal: int
    trial: controller.Trial
    outcome: str
    error_type: str | None
    execution_capability: production.PermutationAttemptCapability
    execution_capability_digest: bytes
    production_attempt_ordinal: int


def _optimizer(beta: np.ndarray | None = None) -> m2.M2OptimizerResult:
    vector = np.zeros(m2.PAIR_DIMENSION) if beta is None else np.asarray(beta, dtype=float)
    return m2.M2OptimizerResult(
        beta=vector.copy(),
        objective=0.0,
        smooth_objective=0.0,
        penalty_value=0.0,
        iterations=0,
        lipschitz=1.0,
        kkt_residual=0.0,
        relative_objective_change=0.0,
        range_residual=0.0,
        status="pass",
        telemetry={"target_free_fixture": True},
    )


def _transform(offset: executor.FixedM1Offset) -> m2.ResidualTransform:
    basis = np.eye(m2.PAIR_DIMENSION)
    return m2.ResidualTransform(
        design_transform=offset.design_transform,
        geometry_scales=offset.geometry_scales,
        projection_f_to_h=np.zeros((m1.PARAMETER_DIMENSION, m2.PAIR_DIMENSION)),
        pair_residual_scales=np.ones(m2.PAIR_DIMENSION),
        range_basis=basis,
        range_projector=basis.copy(),
        residual_gram=basis.copy(),
        scaled_residual_gram=basis.copy(),
        diagnostics={"target_free_fixture": True},
    )


def _prepared(
    cells: tuple[m1.Cell, ...],
    offset: executor.FixedM1Offset,
    transform: m2.ResidualTransform,
) -> m2.PreparedM2:
    count = len(cells)
    return m2.PreparedM2(
        m1.PreparedData(
            cells=cells,
            x=np.zeros((count, m1.DESIGN_DIMENSION)),
            geometry_centers=np.zeros((count, m1.GEOMETRY_DIMENSION)),
            geometry_scales=offset.geometry_scales,
            weights=np.full(count, 1.0 / count),
            design_transform=offset.design_transform,
        ),
        transform,
    )


def _cache(
    execution: permutation_m1.PermutationM1Execution,
) -> executor.PreparedTrialCache:
    folds = []
    for spec in executor.logo_fold_specs(execution.permuted_cells):
        offset = execution.fixed_outer_offsets[spec.held_out_group]
        transform = _transform(offset)
        training_cells = tuple(
            execution.permuted_cells[index] for index in spec.training_indexes
        )
        validation_cells = tuple(
            execution.permuted_cells[index] for index in spec.validation_indexes
        )
        baseline = executor._readonly_copy(
            execution.baseline_log_probability_nats[list(spec.validation_indexes)],
            shape=(len(validation_cells),),
            label="target-free permutation baseline",
        )
        pair_baseline = np.zeros((len(validation_cells), m2.PAIR_DIMENSION))
        pair_baseline.setflags(write=False)
        folds.append(
            executor.PreparedFold(
                spec=spec,
                fixed_m1=offset,
                training=_prepared(training_cells, offset, transform),
                validation=_prepared(validation_cells, offset, transform),
                baseline_log_probability_nats=baseline,
                baseline_pair_probabilities=pair_baseline,
            )
        )
    cache = executor.PreparedTrialCache(
        execution.candidates,
        execution.geometry,
        m2.PairFeatureLibrary(execution.candidates),
        tuple(folds),
        executor.SYNTHETIC_TEST_MODE,
        None,
        tuple((cell.identifier, cell.group) for cell in execution.permuted_cells),
        None,
        _seal=executor._PREPARED_CACHE_SEAL,
    )
    return executor._register_prepared_cache(cache)


def _numerical(
    cache: executor.PreparedTrialCache,
    trial: controller.Trial,
    maximum_iterations: int,
) -> tuple[tuple[production.PermutationFoldExecution, ...], controller.TrialResult]:
    del maximum_iterations
    assert trial.configuration is not None
    maximum = m2.LambdaMaximumResult(
        1.0, np.nextafter(1.0, 0.0), 1.0, 0.0, 0
    )
    penalty = executor.penalty_from_configuration(trial.configuration, maximum)
    folds = []
    for prepared in cache.folds:
        fit = _optimizer()
        scores = []
        for local, (global_index, cell) in enumerate(
            zip(
                prepared.spec.validation_indexes,
                prepared.validation.cells,
                strict=True,
            )
        ):
            baseline = float(prepared.baseline_log_probability_nats[local])
            observed = baseline if cell.k in {0, m1.PANEL_SIZE} else baseline + 0.01
            scores.append(
                production.PermutationCellScore(
                    global_index=global_index,
                    cell_key=cell.identifier,
                    group=cell.group,
                    k=cell.k,
                    m1_log_probability_nats=baseline,
                    m2_log_probability_nats=observed,
                    gain_bits=(observed - baseline) / math.log(2.0),
                )
            )
        folds.append(
            production.PermutationFoldExecution(
                spec=prepared.spec,
                fixed_m1=prepared.fixed_m1,
                transform=prepared.training.transform,
                penalty=penalty,
                fit=fit,
                gain_bits_per_cell=math.fsum(score.gain_bits for score in scores)
                / len(scores),
                active_pair_degrees=0,
                cells=tuple(scores),
            )
        )
    result = controller.TrialResult(
        trial,
        math.fsum(fold.gain_bits_per_cell for fold in folds) / len(folds),
        0,
    )
    return tuple(folds), result


class WholeSetPermutationProductionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.m1_fixture = m1_tests.PermutationM1Tests()
        self.m1_fixture.setUp()
        self.cells = self.m1_fixture.cells
        self.candidates = self.m1_fixture.candidates
        self.geometry = self.m1_fixture.geometry
        self.trial = m1_tests._trial()
        self.session = _Token()
        self.authority = _Token()
        self.cache_authority = _Token()
        self.adapter = production._create_synthetic_qualification_adapter(
            self.cells,
            self.candidates,
            self.geometry,
            session=self.session,
            authority=self.authority,
            cache_authority=self.cache_authority,
        )
        self.fit_calls: list[object] = []
        self.cache_calls: list[object] = []
        self.numerical_calls: list[object] = []

    def _fit_m1(
        self,
        cells,
        assignment,
        candidates,
        geometry,
        *,
        identity_prefix,
    ):
        self.fit_calls.append((assignment, identity_prefix))
        with (
            mock.patch.object(
                permutation_m1, "_fit_warm_path", self.m1_fixture._warm
            ),
            mock.patch.object(
                permutation_m1, "_fit_zero_outer", self.m1_fixture._zero
            ),
            mock.patch.object(
                permutation_m1.real_m1,
                "_production_predictor",
                self.m1_fixture._predict,
            ),
        ):
            return permutation_m1.fit_permuted_outer_m1(
                cells,
                assignment,
                candidates,
                geometry,
                identity_prefix=identity_prefix,
            )

    def _prepare_cache(self, execution, candidates, geometry, **kwargs):
        self.cache_calls.append((execution, candidates, geometry, kwargs))
        return _cache(execution)

    def _run_numerical(self, cache, trial, maximum_iterations):
        self.numerical_calls.append((cache, trial, maximum_iterations))
        return _numerical(cache, trial, maximum_iterations)

    @contextmanager
    def _numerical_patches(self):
        with (
            mock.patch.object(production, "_FIT_PERMUTED_M1", self._fit_m1),
            mock.patch.object(
                production, "_PREPARE_PERMUTED_CACHE", self._prepare_cache
            ),
            mock.patch.object(
                production, "_CONFIGURED_M2_RUNNER", self._run_numerical
            ),
        ):
            yield

    def _issue(
        self,
        *,
        session_ordinal: int = 1,
        null_replicate_index: int | None = None,
    ) -> production.PermutationAttemptCapability:
        return production._issue_permutation_attempt_capability(
            self.adapter,
            self.session,
            self.authority,
            self.trial,
            null_replicate_index=null_replicate_index,
            session_attempt_ordinal=session_ordinal,
            session_version=session_ordinal - 1,
        )

    def _ledger(
        self,
        capability: production.PermutationAttemptCapability,
        outcome: str,
    ) -> _LedgerAttempt:
        record = production._lookup_capability(capability)
        return _LedgerAttempt(
            ordinal=record.session_attempt_ordinal,
            trial=self.trial,
            outcome=outcome,
            error_type=record.error_type,
            execution_capability=capability,
            execution_capability_digest=record.binding_digest,
            production_attempt_ordinal=record.attempt_ordinal,
        )

    def test_pre_m1_mapping_exact_66_plus_12_cache_and_detailed_record(self) -> None:
        capability = self._issue()
        state = production._require_adapter(self.adapter)
        mapping_record = state.mappings[self.trial.valid_index]
        self.assertEqual(self.fit_calls, [])
        self.assertIs(
            production._lookup_capability(capability).assignment_reference(),
            mapping_record.assignment,
        )

        with self._numerical_patches():
            observed = production.execute_permutation_attempt(
                self.adapter, capability
            )

        self.assertEqual(len(self.fit_calls), 1)
        self.assertIs(self.fit_calls[0][0], mapping_record.assignment)
        self.assertEqual(len(self.m1_fixture.warm_calls), 66)
        self.assertEqual(len({key for key, _ in self.m1_fixture.warm_calls}), 66)
        self.assertEqual(len(self.m1_fixture.zero_calls), 12)
        self.assertEqual(len({key for key, _, _ in self.m1_fixture.zero_calls}), 12)
        self.assertEqual(len(self.cache_calls), 1)
        self.assertEqual(len(self.numerical_calls), 1)
        self.assertIs(observed.trial, self.trial)
        self.assertIs(observed.configuration, self.trial.configuration)
        self.assertIs(observed.assignment, mapping_record.assignment)
        self.assertEqual(len(observed.folds), 12)
        scores = tuple(score for fold in observed.folds for score in fold.cells)
        self.assertEqual(len(scores), 194)
        self.assertEqual({score.global_index for score in scores}, set(range(194)))
        self.assertEqual(len({score.cell_key for score in scores}), 194)
        self.assertEqual(len(observed.m1_fit_identities), 12)
        self.assertFalse(observed.selector_eligible)
        self.assertIs(
            production._require_registered_execution(self.adapter, observed),
            observed.result,
        )
        producer, validator = production._adapter_producer_and_validator(self.adapter)
        self.assertTrue(validator(producer, observed.result))
        self.assertTrue(
            validator.validate_execution_detail(
                producer, observed.result, observed
            )
        )
        self.assertFalse(
            production._validate_named_external_execution_detail(
                producer, validator, observed.result, observed
            )
        )
        self.assertFalse(
            production._validate_named_external_execution_detail(
                producer, validator, observed.result, object()
            )
        )
        self.assertNotIn(
            "callback", inspect.signature(production.execute_permutation_attempt).parameters
        )
        self.assertFalse(observed.m1_execution.full_development_refit_run)
        with self.assertRaisesRegex(
            production.PermutationProductionError, "exact fresh M1 execution"
        ):
            production._validate_permuted_cache(
                observed.m1_execution,
                observed.prepared_cache,
                observed.prepared_cache,
            )

        first = observed.prepared_cache.folds[0]
        wrong_training = tuple(
            self.cells[index] for index in first.spec.training_indexes
        )
        bad_first = dataclasses.replace(
            first,
            training=_prepared(
                wrong_training,
                first.fixed_m1,
                first.training.transform,
            ),
        )
        bad_cache = executor.PreparedTrialCache(
            observed.prepared_cache.candidates,
            observed.prepared_cache.geometry,
            observed.prepared_cache.pairs,
            (bad_first, *observed.prepared_cache.folds[1:]),
            observed.prepared_cache.mode,
            observed.prepared_cache.execution_inventory,
            observed.prepared_cache.cell_inventory,
            observed.prepared_cache.expected_rank,
            _seal=executor._PREPARED_CACHE_SEAL,
        )
        executor._register_prepared_cache(bad_cache)
        with self.assertRaisesRegex(
            production.PermutationProductionError,
            "M1 offset, transform, or own baseline",
        ):
            production._validate_permuted_cache(
                observed.m1_execution,
                bad_cache,
                self.cache_authority,
            )

        prepared_fold = observed.prepared_cache.folds[0]
        observed_fold = observed.folds[0]
        forged_penalty = dataclasses.replace(
            observed_fold.penalty,
            penalty_kind="forged_kind",
            specification=m2.PenaltySpec("group_lasso", 999.0, 1.0),
        )
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError,
            "penalty mapping",
        ):
            production._validate_fold_detail(
                prepared_fold,
                dataclasses.replace(observed_fold, penalty=forged_penalty),
                self.trial.configuration,
                1,
            )

        score_position = next(
            index
            for index, score in enumerate(observed_fold.cells)
            if score.global_index in {0, 1}
        )
        score = observed_fold.cells[score_position]
        forged_score = dataclasses.replace(
            score, global_index=bool(score.global_index)
        )
        forged_scores = list(observed_fold.cells)
        forged_scores[score_position] = forged_score
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError,
            "held-out score",
        ):
            production._validate_fold_detail(
                prepared_fold,
                dataclasses.replace(observed_fold, cells=tuple(forged_scores)),
                self.trial.configuration,
                1,
            )

    def test_consume_burn_retry_same_mapping_and_commit_chronology(self) -> None:
        first = self._issue()
        first_record = production._lookup_capability(first)
        assignment = first_record.assignment_reference()

        def fail_after_detail_registration(*args, **kwargs):
            del args, kwargs
            self.assertEqual(first_record.phase, "consumed")
            raise KeyboardInterrupt("target-free failure")

        with (
            self._numerical_patches(),
            mock.patch.object(
                production,
                "_require_registered_execution",
                side_effect=fail_after_detail_registration,
            ),
        ):
            with self.assertRaises(KeyboardInterrupt):
                production.execute_permutation_attempt(self.adapter, first)
        self.assertEqual(first_record.phase, "burned_unacked")
        self.assertIsNone(first_record.m1_execution_reference)
        self.assertIsNone(first_record.prepared_cache_reference)
        self.assertIsNone(first_record.execution_reference)
        self.assertIsNone(first_record.execution_digest)
        terminal = production._completed_permutation_attempt_binding(
            self.adapter,
            self.session,
            self.authority,
            first,
            "error",
        )
        self.assertEqual(terminal[2], "builtins.KeyboardInterrupt")
        self.assertIsNone(terminal[3])
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "already consumed"
        ):
            production.execute_permutation_attempt(self.adapter, first)
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "not ledger-terminal"
        ):
            self._issue(session_ordinal=2)

        failed_attempt = self._ledger(first, "invalid")
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError,
            "differs from permutation capability",
        ):
            production._acknowledge_permutation_attempt(
                self.adapter,
                self.session,
                self.authority,
                first,
                dataclasses.replace(
                    failed_attempt,
                    error_type="forged.Error",
                ),
            )
        production._acknowledge_permutation_attempt(
            self.adapter,
            self.session,
            self.authority,
            first,
            failed_attempt,
        )
        second = self._issue(session_ordinal=2)
        self.assertIsNot(second, first)
        self.assertIs(
            production._lookup_capability(second).assignment_reference(), assignment
        )
        with self._numerical_patches():
            execution = production.execute_permutation_attempt(self.adapter, second)
        valid_attempt = self._ledger(second, "valid")
        production._acknowledge_permutation_attempt(
            self.adapter,
            self.session,
            self.authority,
            second,
            valid_attempt,
        )
        self.assertEqual(
            production._require_acknowledged_permutation_attempt(
                self.adapter,
                self.session,
                self.authority,
                second,
                valid_attempt,
            )[0],
            2,
        )
        with self.assertRaises(production.PermutationExecutionAuthorityError):
            production._acknowledge_permutation_attempt(
                self.adapter,
                self.session,
                self.authority,
                second,
                valid_attempt,
            )
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "cannot be attempted"
        ):
            self._issue(session_ordinal=3)
        self.assertIs(
            production._require_registered_execution(self.adapter, execution),
            execution.result,
        )

    def test_wrong_context_null_route_mutation_and_dead_cache_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "session/authority"
        ):
            production._issue_permutation_attempt_capability(
                self.adapter,
                _Token(),
                self.authority,
                self.trial,
                null_replicate_index=None,
                session_attempt_ordinal=1,
                session_version=0,
            )
        capability = self._issue(null_replicate_index=1)
        assignment = production._lookup_capability(capability).assignment_reference()
        self.assertEqual(assignment.rng_route, "null_replay")
        self.assertEqual(assignment.null_replicate_index, 1)

        other_session = _Token()
        other_authority = _Token()
        other_cache = _Token()
        other_adapter = production._create_synthetic_qualification_adapter(
            self.cells,
            self.candidates,
            self.geometry,
            session=other_session,
            authority=other_authority,
            cache_authority=other_cache,
        )
        with (
            mock.patch.object(
                production,
                "_FIT_PERMUTED_M1",
                side_effect=AssertionError("wrong adapter reached M1"),
            ),
            self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "another adapter",
            ),
        ):
            production.execute_permutation_attempt(other_adapter, capability)
        self.assertEqual(production._lookup_capability(capability).phase, "issued")

        with mock.patch.object(
            mapping.PermutationAssignment,
            "apply",
            side_effect=AssertionError("rebound mapping reached M1"),
        ):
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "mapping functions were rebound",
            ):
                production.execute_permutation_attempt(self.adapter, capability)
        self.assertEqual(production._lookup_capability(capability).phase, "issued")

        expected_cells = mapping.apply_permutation(self.cells, assignment)
        wrong_first = dataclasses.replace(
            expected_cells[0],
            observed_index=self.cells[0].observed_index,
        )
        if wrong_first.observed_index == expected_cells[0].observed_index:
            wrong_first = dataclasses.replace(
                expected_cells[0],
                identifier="wrong_recipient",
            )
        with self.assertRaisesRegex(
            production.PermutationProductionError,
            "recipient-to-donor mapping",
        ):
            production._validate_applied_mapping(
                self.cells,
                assignment,
                (wrong_first, *expected_cells[1:]),
            )

        bool_position = next(
            index
            for index, donor_index in enumerate(assignment.recipient_to_donor)
            if self.cells[donor_index].observed_index in {0, 1}
        )
        bool_index_cell = dataclasses.replace(
            expected_cells[bool_position],
            observed_index=bool(expected_cells[bool_position].observed_index),
        )
        bool_cells = list(expected_cells)
        bool_cells[bool_position] = bool_index_cell
        with self.assertRaisesRegex(
            production.PermutationProductionError,
            "recipient-to-donor mapping",
        ):
            production._validate_applied_mapping(
                self.cells,
                assignment,
                tuple(bool_cells),
            )

        cloned_trial = dataclasses.replace(self.trial)
        state = production._require_adapter(self.adapter)
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "frozen trial or mapping"
        ):
            production._require_mapping_record(
                state,
                cloned_trial,
                1,
            )
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError,
            "exact range",
        ):
            production._require_mapping_record(state, self.trial, True)
        forged_coverage_trial = dataclasses.replace(
            self.trial,
            coverage_binding=object(),
        )
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError,
            "outside controller core families",
        ):
            production._require_trial(forged_coverage_trial)

        source = self.cells[0]
        original_identifier = source.identifier
        object.__setattr__(source, "identifier", "mutated_source")
        try:
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError, "source cells changed"
            ):
                production._require_adapter(self.adapter)
        finally:
            object.__setattr__(source, "identifier", original_identifier)

        dead_session = _Token()
        dead_authority = _Token()
        cache_token = _Token()
        dead_adapter = production._create_synthetic_qualification_adapter(
            self.cells,
            self.candidates,
            self.geometry,
            session=dead_session,
            authority=dead_authority,
            cache_authority=cache_token,
        )
        # Keep session/authority alive long enough to isolate the cache producer.
        dead_state = production._lookup_adapter(dead_adapter)
        session = dead_state.session_reference()
        authority = dead_state.authority_reference()
        del cache_token
        gc.collect()
        self.assertIsNotNone(session)
        self.assertIsNotNone(authority)
        with self.assertRaisesRegex(
            production.PermutationExecutionAuthorityError, "source producer"
        ):
            production._require_adapter(dead_adapter)

        with self.assertRaises(production.PermutationExecutionAuthorityError):
            copy.copy(capability)

    def test_commit_baseexception_rolls_back_and_allows_exact_retry(self) -> None:
        first = self._issue()
        first_record = production._lookup_capability(first)
        assignment = first_record.assignment_reference()
        original_weak = production._weak

        def fail_commit(value, label):
            if label == "permutation result":
                raise KeyboardInterrupt("commit interruption")
            return original_weak(value, label)

        with (
            self._numerical_patches(),
            mock.patch.object(production, "_weak", side_effect=fail_commit),
        ):
            with self.assertRaises(KeyboardInterrupt):
                production.execute_permutation_attempt(self.adapter, first)
        self.assertEqual(first_record.phase, "burned_unacked")
        self.assertEqual(first_record.completion, "error")
        self.assertIsNone(first_record.execution_reference)
        state = production._require_adapter(self.adapter)
        self.assertNotIn(self.trial.valid_index, state.result_records)
        terminal = production._completed_permutation_attempt_binding(
            self.adapter,
            self.session,
            self.authority,
            first,
            "error",
        )
        self.assertEqual(terminal[2:], ("builtins.KeyboardInterrupt", None))

        failed_attempt = self._ledger(first, "resource")
        production._acknowledge_permutation_attempt(
            self.adapter,
            self.session,
            self.authority,
            first,
            failed_attempt,
        )
        retry = self._issue(session_ordinal=2)
        self.assertIsNot(retry, first)
        self.assertIs(
            production._lookup_capability(retry).assignment_reference(), assignment
        )

    def test_fixed_runner_covers_all_controller_permutation_families(self) -> None:
        capability = self._issue()
        assignment = production._lookup_capability(capability).assignment_reference()
        assert assignment is not None
        m1_execution = self._fit_m1(
            self.cells,
            assignment,
            self.candidates,
            self.geometry,
            identity_prefix="qualification.fixed.runner",
        )
        cache = _cache(m1_execution)
        maximum = m2.LambdaMaximumResult(
            1.0, np.nextafter(1.0, 0.0), 1.0, 0.0, 0
        )
        baseline_by_validation = {
            id(fold.validation): fold.baseline_log_probability_nats
            for fold in cache.folds
        }

        def evaluate(beta, theta, prepared, *args, **kwargs):
            del beta, theta, args, kwargs
            baseline = baseline_by_validation[id(prepared)]
            return m2.M2Evaluation(
                0.0,
                np.zeros(m2.PAIR_DIMENSION),
                baseline.copy(),
                np.zeros((len(baseline), m1.PANEL_SIZE)),
                0,
            )

        families = (
            controller.Configuration("lasso", 0.1),
            controller.Configuration("elastic_net", 0.1, alpha=0.5),
            controller.Configuration("group_lasso", 0.1),
        )
        expected = (
            ("lasso", 1.0),
            ("elastic_net", 0.5),
            ("group_lasso", 1.0),
        )
        for configuration, (kind, ratio) in zip(families, expected, strict=True):
            trial = dataclasses.replace(self.trial, configuration=configuration)
            fit_calls = []

            def fit(*args, **kwargs):
                fit_calls.append((args, kwargs))
                return _optimizer()

            with (
                mock.patch.object(
                    production,
                    "_CORE_FOLD_LAMBDA_MAXIMUM",
                    return_value=maximum,
                ),
                mock.patch.object(production, "_CORE_FIT_M2", side_effect=fit),
                mock.patch.object(
                    production, "_CORE_EVALUATE_M2", side_effect=evaluate
                ),
            ):
                folds, result = production._execute_configured_core_m2(
                    cache, trial, 1
                )
            self.assertEqual(len(folds), 12)
            self.assertEqual(len(fit_calls), 12)
            self.assertEqual(result.gain_bits_per_cell, 0.0)
            for args, _ in fit_calls:
                specification = args[5]
                self.assertEqual(specification.kind, kind)
                self.assertEqual(specification.l1_ratio, ratio)

        with mock.patch.object(m2, "fit_m2", side_effect=AssertionError):
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError, "were rebound"
            ):
                production._require_fixed_production_functions()
        with mock.patch.object(mapping, "_require_trial", return_value=self.trial):
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "mapping functions were rebound",
            ):
                production._require_trial(self.trial)
        with mock.patch.object(mapping, "_canonical_cells", return_value=((), (), (), ())):
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "mapping functions were rebound",
            ):
                production._create_synthetic_qualification_adapter(
                    self.cells,
                    self.candidates,
                    self.geometry,
                    session=_Token(),
                    authority=_Token(),
                    cache_authority=_Token(),
                )

    def test_exact_observed_hook_contract_and_rebinding_fail_closed(self) -> None:
        names = production._OBSERVED_HOOK_NAMES
        callbacks = {}
        for name in names:
            def accept(*arguments, _name=name):
                del arguments, _name
                return True

            callbacks[name] = accept

        events = []

        def resolve(name):
            return callbacks[name]

        def request(name, *arguments, expected_callback=None):
            self.assertIs(expected_callback, callbacks[name])
            events.append((name, arguments))
            return expected_callback(*arguments)

        owner = _Token()
        source_cache = _Token()
        capability = _Token()
        binding_digest = b"exact-binding"
        ledger_attempt = _Token()
        with (
            mock.patch.object(production, "_observed_hook", resolve),
            mock.patch.object(production, "_OBSERVED_HOOK", resolve),
            mock.patch.object(production, "_observed_request", request),
            mock.patch.object(production, "_OBSERVED_REQUEST", request),
        ):
            hooks = production._request_observed_bind(
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
            )
            production._request_observed_capability(
                hooks,
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                7,
                11,
            )
            production._request_observed_execute(
                hooks,
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                capability,
                binding_digest,
                7,
                11,
            )
            production._request_observed_ack(
                hooks,
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                capability,
                binding_digest,
                3,
                7,
                11,
                "success",
                None,
                ledger_attempt,
            )
            callbacks[production._OBSERVED_CAPABILITY_HOOK] = lambda *args: True
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "changed after binding",
            ):
                production._request_observed_capability(
                    hooks,
                    self.session,
                    self.authority,
                    self.adapter,
                    owner,
                    source_cache,
                    self.trial,
                    self.trial.coverage_binding,
                    None,
                    8,
                    12,
                )

        self.assertEqual([event[0] for event in events], list(names))
        self.assertEqual(
            events[0][1],
            (self.session, self.authority, self.adapter, owner, source_cache),
        )
        self.assertEqual(
            events[1][1],
            (
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                7,
                11,
            ),
        )
        self.assertEqual(
            events[2][1],
            (
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                capability,
                binding_digest,
                7,
                11,
            ),
        )
        self.assertEqual(
            events[3][1],
            (
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                self.trial.coverage_binding,
                None,
                capability,
                binding_digest,
                3,
                7,
                11,
                "success",
                None,
                ledger_attempt,
            ),
        )

        with mock.patch.object(production, "_observed_request", lambda *args: True):
            with self.assertRaisesRegex(
                production.PermutationExecutionAuthorityError,
                "were rebound",
            ):
                production._require_fixed_production_functions()

    def test_mutated_detail_and_dead_adapter_producer_fail_closed(self) -> None:
        capability = self._issue()
        with self._numerical_patches():
            execution = production.execute_permutation_attempt(
                self.adapter, capability
            )
        object.__setattr__(execution, "selector_eligible", True)
        try:
            with self.assertRaises(production.PermutationExecutionAuthorityError):
                production._require_registered_execution(self.adapter, execution)
        finally:
            object.__setattr__(execution, "selector_eligible", False)
        self.assertIs(
            production._require_registered_execution(self.adapter, execution),
            execution.result,
        )

        producer, validator = production._adapter_producer_and_validator(self.adapter)
        result = execution.result
        reference = weakref.ref(self.adapter)
        del producer
        del self.adapter
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(validator(_Token(), result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
