#!/usr/bin/env python3
"""Lightweight target-free tests for the named Rademacher production layer."""

from __future__ import annotations

import copy
import dataclasses
import gc
import inspect
import math
import pickle
import sys
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import v3_m2_rademacher_adapter_v1 as primitive  # noqa: E402
import v3_m2_rademacher_production_adapter_v1 as production  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


class _Token:
    pass


@dataclasses.dataclass
class _FeatureToken:
    candidates: m1.CandidateLibrary
    seed_route: primitive.RademacherSeedRoute


@dataclasses.dataclass
class _FitToken:
    beta: np.ndarray


@dataclasses.dataclass
class _RademacherFoldToken:
    features: object


@dataclasses.dataclass
class _LedgerAttempt:
    ordinal: int
    trial: controller.Trial
    outcome: str
    error_type: str | None
    execution_capability: production.RademacherAttemptCapability
    execution_capability_digest: bytes
    production_attempt_ordinal: int


def _coverage_binding() -> controller.CoverageBinding:
    return controller.CoverageBinding(
        tuple((index,) for index in range(controller.COVERAGE_TRIALS)),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0", 0),
        tuple("0x0.0p+0" for _ in range(controller.PAIR_COUNT)),
    )


def _trial(
    configuration: controller.Configuration | None = None,
    *,
    valid_index: int = 17,
    seed: int = 2026092801,
    binding: controller.CoverageBinding | None = None,
) -> controller.Trial:
    return controller.Trial(
        valid_index,
        "reference",
        "rademacher_noise",
        "postcoverage",
        True,
        seed=seed,
        cycle=1,
        matched_configuration=(
            controller.Configuration("lasso", 0.1)
            if configuration is None
            else configuration
        ),
        coverage_binding=_coverage_binding() if binding is None else binding,
    )


def _low_rank_result(
    configuration: controller.Configuration,
) -> optional.LowRankOptimizerResult:
    assert configuration.rank is not None
    rank = int(configuration.rank)
    return optional.LowRankOptimizerResult(
        state=optional.LowRankState(
            np.eye(m1.PANEL_SIZE, dtype=float)[:, :rank],
            np.zeros(rank),
        ),
        beta=np.zeros(m2.PAIR_DIMENSION),
        objective=0.0,
        smooth_objective=0.0,
        penalty_value=0.0,
        iterations=1,
        gradient_infinity=0.0,
        relative_objective_change=0.0,
        status="pass",
        telemetry={
            "objective_mode": "production",
            "rank": rank,
            "rho": float(configuration.rho),
            "lambda_reference": 1.0,
        },
    )


def _hybrid_coverage() -> tuple[
    controller.CoverageEvidence,
    tuple[controller.TrialResult, ...],
]:
    core = tuple(
        controller.TrialResult(trial, 0.01, int(trial.role == "association"))
        for trial in controller.coverage_trials()
    )
    contrasts = [0.0] * controller.PAIR_COUNT
    contrasts[1] = 0.6
    contrasts[7] = 0.3
    contrasts[11] = 0.1
    evidence = controller.bind_coverage_evidence(
        core,
        controller.OptionalDiagnostics(0.40, 16.0, 0.80, 9),
        contrasts,
    )
    low_rank_trials = controller.coverage_trials(
        coverage_evidence=evidence
    )[controller.COVERAGE_TRIALS : controller.COVERAGE_TRIALS + 8]
    anchors = tuple(
        controller.TrialResult(
            trial,
            (
                0.08
                if trial.configuration is not None
                and trial.configuration.rank == 2
                and trial.configuration.rho == 0.1
                else 0.02
            ),
            1,
        )
        for trial in low_rank_trials
    )
    return evidence, anchors


class RademacherProductionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            source,
            _,
            _,
            cls.candidates,
            cls.geometry,
            cls.pairs,
            _,
        ) = m2.synthetic_k2_problem(rows=24)
        rows = []
        for index, (identifier, group) in enumerate(
            controller.FROZEN_CELL_INVENTORY.entries
        ):
            template = source[index % len(source)]
            rows.append(
                dataclasses.replace(
                    template,
                    identifier=identifier,
                    group=group,
                    observed_index=index % math.comb(m1.PANEL_SIZE, 2),
                )
            )
        cls.cells = tuple(rows)
        cls.moments = tuple(
            cls.geometry.uniform_moments(cell) for cell in cls.cells
        )

    def setUp(self) -> None:
        self.session = _Token()
        self.authority = _Token()
        self.cache = _Token()
        self.trial = _trial()
        self.runner_calls = 0
        self.fail_next: BaseException | None = None
        self.feature_calls: list[tuple[Any, Any]] = []

        def feature_factory(candidates, route):
            self.feature_calls.append((candidates, route))
            return _FeatureToken(candidates, route)

        self.adapter = production._create_synthetic_qualification_adapter(
            self.cache,
            self.cells,
            self.moments,
            self.candidates,
            self.geometry,
            self.pairs,
            session=self.session,
            authority=self.authority,
            feature_factory=feature_factory,
            numerical_runner=self._runner,
        )

    def _runner(self, capability, record, state):
        self.runner_calls += 1
        self.assertEqual(record.phase, "consumed")
        if self.fail_next is not None:
            error = self.fail_next
            self.fail_next = None
            raise error
        trial = record.trial_reference()
        assert trial is not None and trial.matched_configuration is not None
        folds = []
        for spec in executor.logo_fold_specs(state.source_cells):
            cells = []
            for global_index in spec.validation_indexes:
                cell = state.source_cells[global_index]
                baseline = 0.0
                observed = 0.01
                cells.append(
                    production.RademacherCellScore(
                        global_index=global_index,
                        cell_key=cell.identifier,
                        group=cell.group,
                        k=cell.k,
                        m1_log_probability_nats=baseline,
                        m2_log_probability_nats=observed,
                        gain_bits=(observed - baseline) / math.log(2.0),
                    )
                )
            gain = math.fsum(cell.gain_bits for cell in cells) / len(cells)
            offset = executor.FixedM1Offset(
                held_out_group=spec.held_out_group,
                training_indexes=spec.training_indexes,
                theta=np.zeros(m1.PARAMETER_DIMENSION),
                design_transform=m1.DesignTransform(np.zeros(9), np.ones(9)),
                geometry_scales=np.ones(m1.GEOMETRY_DIMENSION),
                fit_identity=f"synthetic_rademacher_{spec.ordinal}",
            )
            folds.append(
                production.RademacherFoldExecution(
                    spec=spec,
                    fixed_m1=offset,
                    rademacher_fit=_Token(),
                    validation=_Token(),
                    specification=trial.matched_configuration,
                    precursor_configuration=None,
                    precursor_specification=None,
                    precursor_fit=None,
                    precursor_receipt=None,
                    fit=_FitToken(np.zeros(m2.PAIR_DIMENSION)),
                    gain_bits_per_cell=gain,
                    active_pair_degrees=0,
                    cells=tuple(cells),
                )
            )
        result = controller.TrialResult(
            trial,
            math.fsum(fold.gain_bits_per_cell for fold in folds) / len(folds),
            0,
        )
        return tuple(folds), result

    def _issue(
        self,
        *,
        trial: controller.Trial | None = None,
        session_ordinal: int = 1,
        null_replicate_index: int | None = None,
    ) -> production.RademacherAttemptCapability:
        return production._issue_rademacher_attempt_capability(
            self.adapter,
            self.session,
            self.authority,
            self.trial if trial is None else trial,
            None,
            (),
            (),
            null_replicate_index=null_replicate_index,
            session_attempt_ordinal=session_ordinal,
            session_version=session_ordinal - 1,
        )

    @staticmethod
    def _ledger(capability, outcome):
        record = production._lookup_capability(capability)
        trial = record.trial_reference()
        assert trial is not None
        return _LedgerAttempt(
            ordinal=record.session_attempt_ordinal,
            trial=trial,
            outcome=outcome,
            error_type=record.error_type,
            execution_capability=capability,
            execution_capability_digest=record.binding_digest,
            production_attempt_ordinal=record.attempt_ordinal,
        )

    def test_exact_dictionary_shared_across_12_folds_and_194_scores(self) -> None:
        capability = self._issue()
        record = production._lookup_capability(capability)
        features = record.dictionary_reference()
        execution = production.execute_rademacher_attempt(
            self.adapter, capability
        )
        self.assertEqual(self.runner_calls, 1)
        self.assertEqual(len(self.feature_calls), 1)
        self.assertIs(execution.features, features)
        self.assertIs(execution.seed_route, features.seed_route)
        self.assertEqual(execution.seed_route.mode, "target_free_qualification")
        self.assertEqual(len(execution.folds), 12)
        self.assertTrue(
            all(execution.features is features for _ in execution.folds)
        )
        scores = tuple(score for fold in execution.folds for score in fold.cells)
        self.assertEqual(len(scores), 194)
        self.assertEqual({score.global_index for score in scores}, set(range(194)))
        self.assertEqual(len({score.cell_key for score in scores}), 194)
        self.assertFalse(execution.selector_eligible)
        self.assertTrue(execution.trial.falsifier)
        self.assertIs(
            production._require_registered_execution(self.adapter, execution),
            execution.result,
        )
        producer, validator = production._adapter_producer_and_validator(
            self.adapter
        )
        self.assertTrue(validator(producer, execution.result))
        self.assertTrue(
            validator.validate_execution_detail(
                producer, execution.result, execution
            )
        )
        self.assertFalse(
            validator.validate_execution_detail(
                producer, execution.result, None
            )
        )
        self.assertFalse(
            production._validate_named_external_execution_detail(
                producer, validator, execution.result, execution
            )
        )
        self.assertNotIn(
            "callback",
            inspect.signature(
                production.create_rademacher_reference_adapter
            ).parameters,
        )

    def test_consume_burn_fully_qualified_error_and_retry_same_dictionary(self) -> None:
        first = self._issue()
        first_record = production._lookup_capability(first)
        dictionary = first_record.dictionary_reference()
        self.fail_next = KeyboardInterrupt("target-free interruption")
        with self.assertRaises(KeyboardInterrupt):
            production.execute_rademacher_attempt(self.adapter, first)
        self.assertEqual(first_record.phase, "burned_unacked")
        self.assertEqual(first_record.error_type, "builtins.KeyboardInterrupt")
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "already consumed",
        ):
            production.execute_rademacher_attempt(self.adapter, first)
        terminal = production._completed_rademacher_attempt_binding(
            self.adapter,
            self.session,
            self.authority,
            first,
            "error",
        )
        self.assertEqual(terminal[2], "builtins.KeyboardInterrupt")
        failed = self._ledger(first, "resource")
        production._acknowledge_rademacher_attempt(
            self.adapter,
            self.session,
            self.authority,
            first,
            failed,
        )
        retry = self._issue(session_ordinal=2)
        self.assertIs(
            production._lookup_capability(retry).dictionary_reference(),
            dictionary,
        )
        self.assertEqual(len(self.feature_calls), 1)
        execution = production.execute_rademacher_attempt(self.adapter, retry)
        succeeded = self._ledger(retry, "valid")
        production._acknowledge_rademacher_attempt(
            self.adapter,
            self.session,
            self.authority,
            retry,
            succeeded,
        )
        self.assertEqual(
            production._require_acknowledged_rademacher_attempt(
                self.adapter,
                self.session,
                self.authority,
                retry,
                succeeded,
            )[0],
            2,
        )
        self.assertIs(execution.features, dictionary)
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "cannot be committed twice",
        ):
            self._issue(session_ordinal=3)

    def test_null_route_retry_context_seed_and_coverage_mutation_fail_closed(self) -> None:
        capability = self._issue(null_replicate_index=1)
        record = production._lookup_capability(capability)
        route = record.route_reference()
        self.assertEqual(route.mode, primitive.NULL_REPLAY_MODE)
        self.assertEqual(route.null_replicate_seed, controller.NULL_SEEDS[0])
        self.assertEqual(
            route.spawn_key,
            (
                primitive.NULL_STAGE_CODE_M2_FIT,
                primitive.NONAPPLICABLE_FOLD_ORDINAL,
                self.trial.valid_index,
                primitive.PURPOSE_CODE_RADEMACHER_DESIGN,
            ),
        )
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "session/authority",
        ):
            production._issue_rademacher_attempt_capability(
                self.adapter,
                _Token(),
                self.authority,
                self.trial,
                None,
                (),
                (),
                null_replicate_index=1,
                session_attempt_ordinal=2,
                session_version=1,
            )

        binding = self.trial.coverage_binding
        assert binding is not None
        original = binding.contrast_signature
        object.__setattr__(binding, "contrast_signature", tuple("bad" for _ in original))
        try:
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "live context",
            ):
                production._require_capability(capability, require_current=True)
        finally:
            object.__setattr__(binding, "contrast_signature", original)

        first_cell = self.cells[0]
        original_identifier = first_cell.identifier
        object.__setattr__(first_cell, "identifier", "mutated_source")
        try:
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "source cells",
            ):
                production._require_adapter(self.adapter)
        finally:
            object.__setattr__(first_cell, "identifier", original_identifier)

        first_moment = self.moments[0]
        original_moment = float(first_moment.geometry_mean[0])
        first_moment.geometry_mean[0] = np.nextafter(original_moment, math.inf)
        try:
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "source cells, moments",
            ):
                production._require_adapter(self.adapter)
        finally:
            first_moment.geometry_mean[0] = original_moment

        self.fail_next = RuntimeError("burn for route mismatch")
        with self.assertRaises(RuntimeError):
            production.execute_rademacher_attempt(self.adapter, capability)
        failed = self._ledger(capability, "invalid")
        production._acknowledge_rademacher_attempt(
            self.adapter,
            self.session,
            self.authority,
            capability,
            failed,
        )
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "frozen objective or dictionary",
        ):
            self._issue(session_ordinal=2, null_replicate_index=2)

    def test_cross_adapter_trial_clone_detail_clone_and_replay_rejected(self) -> None:
        capability = self._issue()
        other_session = _Token()
        other_authority = _Token()
        other_cache = _Token()
        other = production._create_synthetic_qualification_adapter(
            other_cache,
            self.cells,
            self.moments,
            self.candidates,
            self.geometry,
            self.pairs,
            session=other_session,
            authority=other_authority,
            feature_factory=lambda candidates, route: _FeatureToken(candidates, route),
            numerical_runner=self._runner,
        )
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "another adapter",
        ):
            production.execute_rademacher_attempt(other, capability)
        self.assertEqual(production._lookup_capability(capability).phase, "issued")

        cloned_trial = dataclasses.replace(self.trial)
        state = production._require_adapter(self.adapter)
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "frozen objective or dictionary",
        ):
            production._dictionary_record(state, cloned_trial, None)

        execution = production.execute_rademacher_attempt(
            self.adapter, capability
        )
        clone = dataclasses.replace(execution)
        with self.assertRaises(production.RademacherExecutionAuthorityError):
            production._require_registered_execution(self.adapter, clone)
        original_folds = execution.folds
        object.__setattr__(
            execution,
            "folds",
            (dataclasses.replace(execution.folds[0]), *execution.folds[1:]),
        )
        try:
            with self.assertRaises(production.RademacherExecutionAuthorityError):
                production._require_registered_execution(self.adapter, execution)
        finally:
            object.__setattr__(execution, "folds", original_folds)
        self.assertIs(
            production._require_registered_execution(self.adapter, execution),
            execution.result,
        )
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "already consumed",
        ):
            production.execute_rademacher_attempt(self.adapter, capability)
        for operation in (
            lambda: copy.copy(capability),
            lambda: copy.deepcopy(capability),
            lambda: pickle.dumps(capability),
        ):
            with self.assertRaises(production.RademacherExecutionAuthorityError):
                operation()

    def test_matched_family_inventory_and_v3_hybrid_precursor_derivation(self) -> None:
        families = (
            controller.Configuration("lasso", 0.1),
            controller.Configuration("elastic_net", 0.1, alpha=0.5),
            controller.Configuration("group_lasso", 0.1),
            controller.Configuration("low_rank", 0.1, rank=2),
        )
        evidence, anchors = _hybrid_coverage()
        optional_nodes = controller.bound_optional_anchor_configurations(
            evidence, anchors
        )
        target = optional_nodes[8]
        families = (*families, target)
        for index, configuration in enumerate(families, start=30):
            cycle = 1 if index < 34 else 2
            trial = controller.Trial(
                index,
                "reference",
                "rademacher_noise",
                "postcoverage",
                True,
                seed=2026092800 + index,
                cycle=cycle,
                matched_configuration=configuration,
                coverage_binding=evidence.binding,
            )
            self.assertIs(production._require_trial(trial), trial)
            self.assertIs(trial.matched_configuration, configuration)

        derived = production._OPTIONAL_DERIVE_PRECURSOR(
            evidence, anchors, target
        )
        selected = controller.choose_incumbent(anchors).trial.configuration
        self.assertIs(derived, selected)
        different_rank = next(
            configuration
            for configuration in optional_nodes[8:]
            if configuration.rank != selected.rank
        ) if any(
            configuration.rank != selected.rank
            for configuration in optional_nodes[8:]
        ) else controller.Configuration(
            "sparse_plus_low_rank",
            target.rho,
            rank=3,
            sparse_rho=target.sparse_rho,
            sparse_edges=target.sparse_edges,
        )
        prescribed = production._OPTIONAL_DERIVE_PRECURSOR(
            evidence, anchors, different_rank
        )
        self.assertEqual(prescribed.family, "low_rank")
        self.assertEqual(prescribed.rank, different_rank.rank)
        self.assertEqual(prescribed.rho, 0.1)
        self.assertIn(
            "_OPTIONAL_FIT_LOW_RANK",
            inspect.getsource(production._run_production_numerical),
        )
        self.assertIn(
            "production_receipt=precursor_receipt",
            inspect.getsource(production._run_production_numerical),
        )

    def test_noise_objective_recomputes_core_low_rank_and_hybrid_normalizers(self) -> None:
        fixed = executor.FixedM1Offset(
            m1.GROUP_ORDER[0],
            tuple(range(1, 194)),
            np.zeros(m1.PARAMETER_DIMENSION),
            m1.DesignTransform(np.zeros(9), np.ones(9)),
            np.ones(m1.GEOMETRY_DIMENSION),
            "normalizer_fixture",
        )
        prepared = _Token()
        features = _FeatureToken(
            self.candidates,
            primitive.RademacherSeedRoute("target_free_qualification", 101),
        )
        maximum = m2.LambdaMaximumResult(
            2.0, np.nextafter(2.0, 0.0), 2.0, 0.0, 1
        )
        scales = optional.LowRankFoldScales(
            lambda_reference=3.0,
            raw_gradient_spectral_norm=3.0,
            smooth_lipschitz_upper_bound=1.0,
            null_gradient=np.zeros(m2.PAIR_DIMENSION),
            null_gradient_matrix=np.zeros((m1.PANEL_SIZE, m1.PANEL_SIZE)),
        )
        core_calls = []
        scale_calls = []

        def lambda_maximum(*args, **kwargs):
            core_calls.append((args, kwargs))
            return maximum

        def low_rank_scales(*args, **kwargs):
            scale_calls.append((args, kwargs))
            return scales

        with (
            mock.patch.object(
                production,
                "_CORE_LAMBDA_MAXIMUM",
                side_effect=lambda_maximum,
            ),
            mock.patch.object(
                production,
                "_OPTIONAL_SCALES",
                side_effect=low_rank_scales,
            ),
        ):
            core_configurations = (
                controller.Configuration("lasso", 0.1),
                controller.Configuration("elastic_net", 0.1, alpha=0.5),
                controller.Configuration("group_lasso", 0.1),
            )
            mappings = tuple(
                production._core_specification(
                    fixed,
                    prepared,
                    self.candidates,
                    self.geometry,
                    features,
                    configuration,
                )
                for configuration in core_configurations
            )
            low = production._low_rank_specification(
                fixed,
                prepared,
                self.candidates,
                self.geometry,
                features,
                controller.Configuration("low_rank", 0.1, rank=2),
            )
            evidence, anchors = _hybrid_coverage()
            hybrid_configuration = controller.bound_optional_anchor_configurations(
                evidence, anchors
            )[8]
            hybrid = production._hybrid_specification(
                fixed,
                prepared,
                self.candidates,
                self.geometry,
                features,
                hybrid_configuration,
            )
        self.assertEqual([item.penalty_kind for item in mappings], [
            "lasso",
            "elastic_net",
            "group_lasso",
        ])
        self.assertEqual([item.specification.strength for item in mappings], [
            0.2,
            0.2,
            0.2,
        ])
        self.assertEqual(low.lambda_reference, 3.0)
        self.assertEqual(hybrid.lambda_reference, 3.0)
        self.assertEqual(hybrid.core_lasso_lambda_maximum, 2.0)
        self.assertEqual(len(core_calls), 4)
        self.assertEqual(len(scale_calls), 2)
        self.assertTrue(all(call[0][4] is features for call in core_calls))
        self.assertTrue(all(call[0][4] is features for call in scale_calls))

    def test_precursor_receipt_binds_fold_config_fit_and_dictionary(self) -> None:
        evidence, anchors = _hybrid_coverage()
        target = controller.bound_optional_anchor_configurations(
            evidence, anchors
        )[8]
        trial = _trial(target, binding=evidence.binding)
        capability = self._issue(trial=trial)
        record = production._lookup_capability(capability)
        record.phase = "consumed"
        initializer = production._OPTIONAL_DERIVE_PRECURSOR(
            evidence, anchors, target
        )
        precursor_spec = optional.LowRankSpec(
            int(initializer.rank), float(initializer.rho), 1.0
        )
        hybrid_spec = optional.HybridSpec(
            int(target.rank),
            float(target.rho),
            float(target.sparse_rho),
            1.0,
            1.0,
            tuple(m2.PAIR_INDEX[index] for index in target.sparse_edges),
        )
        precursor_fit = _low_rank_result(initializer)
        spec = executor.FoldSpec(1, m1.GROUP_ORDER[0], tuple(range(1, 194)), (0,))
        physical_fold = executor.PreparedFold(
            spec=spec,
            fixed_m1=executor.FixedM1Offset(
                spec.held_out_group,
                spec.training_indexes,
                np.zeros(m1.PARAMETER_DIMENSION),
                m1.DesignTransform(np.zeros(9), np.ones(9)),
                np.ones(m1.GEOMETRY_DIMENSION),
                "receipt_fold_1",
            ),
            training=_Token(),
            validation=_Token(),
            baseline_log_probability_nats=np.zeros(1),
            baseline_pair_probabilities=np.zeros((1, m2.PAIR_DIMENSION)),
        )
        rfit = _RademacherFoldToken(record.dictionary_reference())
        receipt = production._issue_precursor_receipt(
            capability,
            physical_fold,
            rfit,
            evidence,
            anchors,
            target,
            initializer,
            precursor_spec,
            hybrid_spec,
            precursor_fit,
        )
        fold = production.RademacherFoldExecution(
            spec=spec,
            fixed_m1=physical_fold.fixed_m1,
            rademacher_fit=rfit,
            validation=_Token(),
            specification=hybrid_spec,
            precursor_configuration=initializer,
            precursor_specification=precursor_spec,
            precursor_fit=precursor_fit,
            precursor_receipt=receipt,
            fit=_FitToken(np.zeros(m2.PAIR_DIMENSION)),
            gain_bits_per_cell=0.0,
            active_pair_degrees=0,
            cells=(),
        )
        binding = production._require_precursor_receipt(
            receipt, fold, capability
        )
        self.assertIs(binding.precursor_fit_reference(), precursor_fit)
        wrong_fit = dataclasses.replace(precursor_fit)
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "binding changed",
        ):
            production._require_precursor_receipt(
                receipt,
                dataclasses.replace(fold, precursor_fit=wrong_fit),
                capability,
            )
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "binding changed",
        ):
            production._require_precursor_receipt(
                receipt,
                dataclasses.replace(
                    fold,
                    precursor_specification=dataclasses.replace(precursor_spec),
                ),
                capability,
            )

    def test_dead_cache_adapter_and_off_chronology_bind_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "rejects synthetic",
        ):
            production._bind_to_observed_session(
                self.adapter,
                self.session,
                self.authority,
                object(),
            )

        session = _Token()
        authority = _Token()
        cache = _Token()
        adapter = production._create_synthetic_qualification_adapter(
            cache,
            self.cells,
            self.moments,
            self.candidates,
            self.geometry,
            self.pairs,
            session=session,
            authority=authority,
            feature_factory=lambda candidates, route: _FeatureToken(candidates, route),
            numerical_runner=self._runner,
        )
        del cache
        gc.collect()
        with self.assertRaisesRegex(
            production.RademacherExecutionAuthorityError,
            "source producer",
        ):
            production._require_adapter(adapter)

        capability = self._issue()
        execution = production.execute_rademacher_attempt(
            self.adapter, capability
        )
        producer, validator = production._adapter_producer_and_validator(
            self.adapter
        )
        reference = weakref.ref(self.adapter)
        del producer
        del self.adapter
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(validator(_Token(), execution.result))

    def test_pristine_unbound_factory_exposes_validator_then_exact_owner_cache(self) -> None:
        cache = object.__new__(executor.PreparedTrialCache)
        object.__setattr__(cache, "candidates", self.candidates)
        object.__setattr__(cache, "geometry", self.geometry)
        object.__setattr__(cache, "pairs", self.pairs)
        object.__setattr__(cache, "mode", executor.SEALED_PRODUCTION_MODE)
        owner = object.__new__(executor.CoreTrialExecutor)
        owner.prepared = cache
        owner.maximum_iterations = executor.FROZEN_MAXIMUM_ITERATIONS

        with (
            mock.patch.object(
                executor,
                "_require_prepared_cache",
                return_value=cache,
            ),
            mock.patch.object(
                production,
                "_source_cells",
                return_value=self.cells,
            ),
            mock.patch.object(production, "_require_source_binding"),
        ):
            adapter = production.create_rademacher_reference_adapter(
                owner,
                self.cells,
                self.moments,
            )
            state = production._lookup_adapter(adapter)
            self.assertIsNone(state.session_reference)
            self.assertIsNone(state.authority_reference)
            self.assertIsNone(state.observed_hooks)
            producer, validator = production._adapter_producer_and_validator(adapter)
            self.assertIs(producer, adapter)
            self.assertIs(state.validator, validator)
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "live session authority",
            ):
                production._require_adapter(adapter)

            session = _Token()
            authority = _Token()
            hooks = tuple(object() for _ in production._OBSERVED_HOOK_NAMES)
            state.session_reference = weakref.ref(session)
            state.authority_reference = weakref.ref(authority)
            state.observed_hooks = hooks
            with mock.patch.object(
                production,
                "_require_observed_hooks",
                return_value=hooks,
            ):
                exact_owner, exact_cache = production._adapter_owner_and_cache(adapter)
                self.assertIs(exact_owner, owner)
                self.assertIs(exact_cache, cache)
                replacement = object.__new__(executor.PreparedTrialCache)
                owner.prepared = replacement
                try:
                    with self.assertRaisesRegex(
                        production.RademacherExecutionAuthorityError,
                        "exact production owner/cache",
                    ):
                        production._adapter_owner_and_cache(adapter)
                finally:
                    owner.prepared = cache

    def test_observed_hook_contract_is_exact_and_callback_rebinding_fails_closed(self) -> None:
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

        owner = object.__new__(executor.CoreTrialExecutor)
        source_cache = object.__new__(executor.PreparedTrialCache)
        capability = _Token()
        binding_digest = b"exact-rademacher-binding"
        ledger_attempt = _Token()
        coverage = _Token()
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
                coverage,
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
                coverage,
                None,
                capability,
                binding_digest,
            )
            production._request_observed_ack(
                hooks,
                self.session,
                self.authority,
                self.adapter,
                owner,
                source_cache,
                self.trial,
                coverage,
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
                production.RademacherExecutionAuthorityError,
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
                    coverage,
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
                coverage,
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
                coverage,
                None,
                capability,
                binding_digest,
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
                coverage,
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

    def test_fixed_function_rebinding_is_detected(self) -> None:
        production._require_fixed_production_functions()
        with mock.patch.object(
            primitive,
            "fit_rademacher_fold",
            side_effect=AssertionError("rebound solver must not run"),
        ):
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "were rebound",
            ):
                production._require_fixed_production_functions()
        with mock.patch.object(production, "_observed_request", lambda *args: True):
            with self.assertRaisesRegex(
                production.RademacherExecutionAuthorityError,
                "were rebound",
            ):
                production._require_fixed_production_functions()


if __name__ == "__main__":
    unittest.main(verbosity=2)
