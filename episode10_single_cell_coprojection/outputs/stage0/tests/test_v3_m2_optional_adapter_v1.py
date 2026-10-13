#!/usr/bin/env python3
"""Target-free authority tests for the named v3 optional-family adapter."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
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

import fit_v3_m2 as core  # noqa: E402
import fit_v3_m2_optional as optional  # noqa: E402
import test_v3_m2_trial_executor_v1 as executor_tests  # noqa: E402
import v3_m2_observed_session_v1 as observed  # noqa: E402
import v3_m2_optional_adapter_v1 as adapter_module  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


class OptionalAdapterAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        executor_tests.SharedExecutorTests.setUpClass()
        cls.source_cache = executor_tests.SharedExecutorTests.cache

    @classmethod
    def _fresh_cache(cls) -> executor.PreparedTrialCache:
        source = cls.source_cache
        return executor._register_prepared_cache(
            executor.PreparedTrialCache(
                source.candidates,
                source.geometry,
                source.pairs,
                source.folds,
                source.mode,
                source.expected_rank,
                source.cell_inventory,
                source.execution_inventory,
                _seal=executor._PREPARED_CACHE_SEAL,
            )
        )

    @contextmanager
    def _production_fixture(self):
        cache = self._fresh_cache()
        owner = executor.CoreTrialExecutor(cache, maximum_iterations=1)
        with (
            mock.patch.object(
                executor,
                "_require_prepared_cache",
                side_effect=lambda supplied: supplied,
            ),
            mock.patch.object(
                executor,
                "SEALED_PRODUCTION_MODE",
                executor.SYNTHETIC_TEST_MODE,
            ),
        ):
            adapter = adapter_module.create_optional_family_adapter(owner)
            session = observed.start_observed_session(owner)
            authority = observed.issue_optional_family_authority(session, adapter)
            yield owner, session, adapter, authority

    @staticmethod
    def _core_execution(
        owner: executor.CoreTrialExecutor,
        trial: controller.Trial,
        authorization=None,
    ) -> executor.TrialExecution:
        del authorization
        contrasts = [0.0] * controller.PAIR_COUNT
        if trial.valid_index == 2:
            contrasts[1] = 0.6
            contrasts[7] = 0.3
            contrasts[11] = 0.1
        result = controller.TrialResult(
            trial,
            0.05 if trial.valid_index == 2 else 0.01,
            1 if trial.role == "association" else 0,
        )
        execution = executor.TrialExecution(
            trial=trial,
            folds=(),
            pair_contrast_rows=(),
            equal_group_pair_contrasts=tuple(contrasts),
            result=result,
        )
        executor._register_owned_execution(owner, execution)
        return execution

    @staticmethod
    def _dummy_low_rank_result(
        configuration: controller.Configuration,
        fold_ordinal: int,
    ) -> optional.LowRankOptimizerResult:
        assert configuration.rank is not None
        rank = int(configuration.rank)
        state = optional.LowRankState(
            np.eye(core.m1.PANEL_SIZE, dtype=float)[:, :rank],
            np.zeros(rank, dtype=float),
        )
        return optional.LowRankOptimizerResult(
            state=state,
            beta=np.zeros(core.PAIR_DIMENSION, dtype=float),
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
                "fold_ordinal": fold_ordinal,
            },
        )

    @classmethod
    def _optional_execution(
        cls,
        record: adapter_module._CapabilityRecord,
    ) -> adapter_module.OptionalTrialExecution:
        trial = record.trial_reference()
        cache = record.cache_reference()
        session = record.session_reference()
        owner = record.owner_reference()
        authority = record.authority_reference()
        assert (
            trial is not None
            and cache is not None
            and session is not None
            and owner is not None
            and authority is not None
            and trial.configuration is not None
        )
        configuration = trial.configuration
        gain = 0.08 if (
            configuration.family == "low_rank"
            and configuration.rank == 2
            and configuration.rho == 0.1
        ) else 0.02
        folds = ()
        if configuration.family == "low_rank":
            folds = tuple(
                adapter_module.OptionalFoldTrialExecution(
                    spec=prepared_fold.spec,
                    specification=optional.LowRankSpec(
                        int(configuration.rank),
                        float(configuration.rho),
                        1.0,
                    ),
                    initializer_configuration=None,
                    initializer_fit=None,
                    hybrid_authority=None,
                    hybrid_receipt=None,
                    fit=cls._dummy_low_rank_result(
                        configuration,
                        prepared_fold.spec.ordinal,
                    ),
                    gain_bits_per_cell=gain,
                    active_pair_degrees=0,
                    cells=(),
                )
                for prepared_fold in cache.folds
            )
        result = controller.TrialResult(trial, gain, 0)
        return adapter_module.OptionalTrialExecution(
            trial=trial,
            folds=folds,
            pair_contrast_rows=(),
            equal_group_pair_contrasts=(0.0,) * controller.PAIR_COUNT,
            result=result,
            mode=executor.SEALED_PRODUCTION_MODE,
            session_identity=id(session),
            owner_identity=id(owner),
            cache_identity=id(cache),
            authority_identity=id(authority),
            attempt_ordinal=record.optional_attempt_ordinal,
        )

    @staticmethod
    def _drive_core(
        owner: executor.CoreTrialExecutor,
        session: observed.ControllerRunSession,
    ) -> None:
        diagnostics = controller.OptionalDiagnostics(0.40, 16.0, 0.80, 9)
        with (
            mock.patch.object(
                owner,
                "_execute_trial",
                side_effect=lambda trial, authorization=None: (
                    OptionalAdapterAuthorityTests._core_execution(
                        owner,
                        trial,
                        authorization,
                    )
                ),
            ),
            mock.patch.object(
                observed.diagnostic,
                "diagnose_optional_families",
                return_value=mock.Mock(diagnostics=diagnostics),
            ),
        ):
            for expected_index in range(1, controller.COVERAGE_TRIALS + 1):
                result = session.run_next_core()
                assert result.trial.valid_index == expected_index
        state = observed._lookup_session(session)
        assert state.coverage_evidence is not None
        assert state.coverage_evidence.activation.sparse_plus_low_rank

    def test_named_binding_wrong_session_cache_and_no_generic_production(self) -> None:
        other_cache = self._fresh_cache()
        other_owner = executor.CoreTrialExecutor(other_cache, maximum_iterations=1)
        with self._production_fixture() as (owner, session, adapter, authority):
            other_adapter = adapter_module.create_optional_family_adapter(other_owner)
            other_session = observed.start_observed_session(other_owner)
            other_authority = observed.issue_optional_family_authority(
                other_session,
                other_adapter,
            )
            with self.assertRaises(observed.ObservedSessionError):
                observed.issue_optional_family_authority(session, other_adapter)
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_external(other_authority)
            with self.assertRaises(observed.ObservedSessionError):
                observed._issue_adapter_authority(
                    session,
                    lambda trial: trial,
                    mode="production",
                    scopes=("optional",),
                )
            self.assertEqual(session.valid_trial_count, 0)
            self.assertEqual(session.invalid_attempt_count, 0)
            self.assertTrue(all((owner, adapter, authority)))

    def test_sealed_external_receipt_rejects_duck_validator_detail_shim(self) -> None:
        with self._production_fixture() as (owner, session, _adapter, _authority):
            self._drive_core(owner, session)
            trial = session.next_trial
            result = controller.TrialResult(trial, 0.0, 0)

            class DuckProducer:
                pass

            class DuckValidator:
                def __call__(self, producer, supplied_result):
                    del producer
                    return supplied_result is result

                def validate_execution_detail(
                    self,
                    producer,
                    supplied_result,
                    execution_detail,
                ):
                    del producer, execution_detail
                    return supplied_result is result

            producer = DuckProducer()
            validator = DuckValidator()
            with self.assertRaisesRegex(
                executor.CoreTrialExecutorError,
                "exact named validator",
            ):
                executor.authenticate_external_execution(
                    owner,
                    producer,
                    validator,
                    result,
                    execution_detail=object(),
                )
            forged_receipt = executor.ExternalExecutionReceipt(
                result,
                execution_detail=object(),
                _seal=executor._EXTERNAL_EXECUTION_RECEIPT_SEAL,
            )
            executor._register_external_receipt_identity(
                forged_receipt,
                (
                    weakref.ref(owner),
                    weakref.ref(owner.prepared),
                    weakref.ref(producer),
                    weakref.ref(validator),
                    executor._deep_integrity_digest(forged_receipt),
                ),
            )
            with self.assertRaisesRegex(
                executor.CoreTrialExecutorError,
                "exact named validator",
            ):
                executor._require_external_execution_receipt(
                    owner,
                    forged_receipt,
                )

    def test_failed_retry_fresh_token_detailed_receipt_and_replay(self) -> None:
        with self._production_fixture() as (owner, session, adapter, authority):
            self._drive_core(owner, session)
            calls = 0

            def execute(record_capability, record):
                nonlocal calls
                del record_capability
                calls += 1
                if calls == 1:
                    raise RuntimeError("target-free optional solver failure")
                return self._optional_execution(record)

            with (
                mock.patch.object(
                    adapter_module,
                    "_execute_numerical_attempt",
                    side_effect=execute,
                ),
                mock.patch.object(
                    adapter_module,
                    "_require_registered_execution",
                    side_effect=lambda _adapter, execution: execution.result,
                ),
            ):
                with self.assertRaises(observed.ObservedSessionError):
                    session.run_next_optional(authority)
                expected_trial = session.next_trial
                result = session.run_next_optional(authority)
                self.assertIs(result.trial, expected_trial)
                state = observed._lookup_session(session)
                failed, succeeded = state.attempts[-2:]
                self.assertIsNot(
                    failed.execution_capability,
                    succeeded.execution_capability,
                )
                self.assertEqual(failed.production_attempt_ordinal, 1)
                self.assertEqual(succeeded.production_attempt_ordinal, 2)
                self.assertEqual(session.invalid_attempt_count, 1)
                receipt = state.history[-1]
                self.assertIsInstance(receipt, executor.ExternalExecutionReceipt)
                self.assertIsInstance(
                    receipt.execution_detail,
                    adapter_module.OptionalTrialExecution,
                )
                self.assertIs(receipt.execution_detail.result, receipt.result)
                producer, validator = adapter_module._adapter_producer_and_validator(
                    adapter
                )
                with self.assertRaises(executor.CoreTrialExecutorError):
                    executor.authenticate_external_execution(
                        owner,
                        producer,
                        validator,
                        receipt.result,
                    )
                replay = succeeded.execution_capability
                assert replay is not None
                for operation in (
                    lambda: copy.copy(replay),
                    lambda: copy.deepcopy(replay),
                    lambda: pickle.dumps(replay),
                ):
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        operation()
                with self.assertRaises(
                    adapter_module.OptionalExecutionAuthorityError
                ):
                    adapter_module._execute_optional_attempt(adapter, replay)

    def test_concurrent_issue_allows_one_exact_capability(self) -> None:
        with self._production_fixture() as (owner, session, adapter, authority):
            self._drive_core(owner, session)
            state = observed._lookup_session(session)
            trial = state.schedule[len(state.results)]
            evidence = state.coverage_evidence
            assert evidence is not None
            state.in_flight = True
            state.optional_capability_issuance_open = True
            barrier = threading.Barrier(2)

            def issue():
                barrier.wait()
                try:
                    return adapter_module._issue_optional_attempt_capability(
                        adapter,
                        session,
                        authority,
                        owner,
                        trial,
                        evidence,
                        tuple(state.history),
                        tuple(state.results),
                        len(state.attempts) + 1,
                        state.version,
                    )
                except Exception as error:
                    return error

            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    issued = tuple(pool.map(lambda _: issue(), range(2)))
            finally:
                state.optional_capability_issuance_open = False
                state.in_flight = False
            capabilities = tuple(
                value
                for value in issued
                if type(value) is adapter_module.OptionalAttemptCapability
            )
            failures = tuple(
                value
                for value in issued
                if type(value) is adapter_module.OptionalExecutionAuthorityError
            )
            self.assertEqual(len(capabilities), 1)
            self.assertEqual(len(failures), 1)

    def test_hybrid_authority_rejects_wrong_fold_initializer_spec_and_target(self) -> None:
        with self._production_fixture() as (owner, session, adapter, authority):
            self._drive_core(owner, session)
            with (
                mock.patch.object(
                    adapter_module,
                    "_execute_numerical_attempt",
                    side_effect=lambda capability, record: self._optional_execution(record),
                ),
                mock.patch.object(
                    adapter_module,
                    "_require_registered_execution",
                    side_effect=lambda _adapter, execution: execution.result,
                ),
            ):
                for _ in range(8):
                    session.run_next_optional(authority)

                checks: dict[str, bool] = {}

                def inspect_hybrid(capability, record):
                    evidence, score_results, executions = (
                        adapter_module._low_rank_coverage(record)
                    )
                    target = record.trial_reference().configuration
                    assert target is not None
                    initializer = optional.derive_hybrid_low_rank_initializer_configuration(
                        evidence,
                        score_results,
                        target,
                    )
                    source = next(
                        execution
                        for execution in executions
                        if execution.trial.configuration is initializer
                    )
                    fold = owner.prepared.folds[0]
                    fit = source.folds[0].fit
                    assert type(fit) is optional.LowRankOptimizerResult
                    specification = optional.HybridSpec(
                        int(target.rank),
                        float(target.rho),
                        float(target.sparse_rho),
                        1.0,
                        1.0,
                        tuple(core.PAIR_INDEX[index] for index in target.sparse_edges),
                    )
                    hybrid_authority = adapter_module._issue_hybrid_solver_authority(
                        capability,
                        fold,
                        evidence,
                        score_results,
                        target,
                        initializer,
                        fit,
                        specification,
                    )
                    checks["valid"] = bool(
                        adapter_module._require_hybrid_solver_authority(
                            hybrid_authority,
                            evidence,
                            score_results,
                            fit,
                            target,
                            specification,
                            initializer,
                        )
                    )

                    class Receipt:
                        pass

                    receipt = Receipt()
                    selected_configuration = controller.choose_incumbent(
                        score_results
                    ).trial.configuration
                    assert selected_configuration is not None
                    receipt_binding = optional._ProductionHybridBinding(
                        coverage_reference=weakref.ref(evidence),
                        low_rank_result_references=tuple(
                            weakref.ref(result) for result in score_results
                        ),
                        selected_anchor_reference=weakref.ref(fit),
                        coverage_signature=optional._coverage_evidence_signature(
                            evidence
                        ),
                        low_rank_result_signatures=tuple(
                            optional._controller_result_signature(result)
                            for result in score_results
                        ),
                        selected_anchor_signature=(
                            optional._low_rank_result_integrity_signature(fit)
                        ),
                        selected_configuration_key=selected_configuration.key,
                        selected_rank=int(selected_configuration.rank),
                        selected_rho=float(selected_configuration.rho),
                        localized_edge_indexes=tuple(evidence.localized.edges),
                        target_configuration_reference=weakref.ref(target),
                        target_configuration_signature=(
                            optional._controller_configuration_signature(target)
                        ),
                        target_specification_reference=weakref.ref(specification),
                        target_specification_signature=(
                            optional._hybrid_specification_signature(specification)
                        ),
                        initializer_configuration_key=initializer.key,
                        initializer_rule_version=(
                            optional.HYBRID_INITIALIZER_RULE_VERSION
                        ),
                        session_authority_reference=weakref.ref(hybrid_authority),
                        session_authority_digest=b"test-session-authority",
                    )
                    fit_binding = optional._HybridResultBinding(
                        objective=mock.sentinel.objective_binding,
                        objective_mode="production",
                        selected_anchor_reference=weakref.ref(fit),
                        selected_anchor_signature=(
                            optional._low_rank_result_integrity_signature(fit)
                        ),
                        production_receipt_reference=weakref.ref(receipt),
                        result_signature=(),
                    )
                    fold_execution = adapter_module.OptionalFoldTrialExecution(
                        spec=fold.spec,
                        specification=specification,
                        initializer_configuration=initializer,
                        initializer_fit=fit,
                        hybrid_authority=hybrid_authority,
                        hybrid_receipt=receipt,
                        fit=mock.sentinel.hybrid_fit,
                        gain_bits_per_cell=0.0,
                        active_pair_degrees=0,
                        cells=(),
                    )
                    adapter_module._require_exact_hybrid_execution_binding(
                        fold_execution,
                        target,
                        specification,
                        fit_binding,
                        receipt_binding,
                    )
                    wrong_receipt = Receipt()
                    adversarial_folds = (
                        dataclasses.replace(
                            fold_execution,
                            initializer_fit=dataclasses.replace(fit),
                        ),
                        dataclasses.replace(
                            fold_execution,
                            hybrid_receipt=wrong_receipt,
                        ),
                        dataclasses.replace(
                            fold_execution,
                            hybrid_authority=object(),
                        ),
                        dataclasses.replace(
                            fold_execution,
                            initializer_configuration=dataclasses.replace(
                                initializer
                            ),
                        ),
                    )
                    for adversarial_fold in adversarial_folds:
                        with self.assertRaises(
                            adapter_module.OptionalExecutionAuthorityError
                        ):
                            adapter_module._require_exact_hybrid_execution_binding(
                                adversarial_fold,
                                target,
                                specification,
                                fit_binding,
                                receipt_binding,
                            )
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        adapter_module._require_exact_hybrid_execution_binding(
                            fold_execution,
                            dataclasses.replace(target),
                            specification,
                            fit_binding,
                            receipt_binding,
                        )
                    cloned_results = (
                        dataclasses.replace(score_results[0]),
                        *score_results[1:],
                    )
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        adapter_module._require_hybrid_solver_authority(
                            hybrid_authority,
                            evidence,
                            cloned_results,
                            fit,
                            target,
                            specification,
                            initializer,
                        )
                    with mock.patch.object(
                        adapter_module,
                        "_low_rank_coverage",
                        return_value=(evidence, cloned_results, executions),
                    ):
                        with self.assertRaises(
                            adapter_module.OptionalExecutionAuthorityError
                        ):
                            adapter_module._require_hybrid_solver_authority(
                                hybrid_authority,
                                evidence,
                                score_results,
                                fit,
                                target,
                                specification,
                                initializer,
                            )
                    wrong_fold_fit = source.folds[1].fit
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        adapter_module._require_hybrid_solver_authority(
                            hybrid_authority,
                            evidence,
                            score_results,
                            wrong_fold_fit,
                            target,
                            specification,
                            initializer,
                        )
                    changed_specification = dataclasses.replace(
                        specification,
                        rho_sparse=0.001
                        if specification.rho_sparse != 0.001
                        else 0.003,
                    )
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        adapter_module._require_hybrid_solver_authority(
                            hybrid_authority,
                            evidence,
                            score_results,
                            fit,
                            target,
                            changed_specification,
                            initializer,
                        )
                    cloned_target = dataclasses.replace(target)
                    with self.assertRaises(
                        adapter_module.OptionalExecutionAuthorityError
                    ):
                        adapter_module._require_hybrid_solver_authority(
                            hybrid_authority,
                            evidence,
                            score_results,
                            fit,
                            cloned_target,
                            specification,
                            initializer,
                        )
                    raise RuntimeError("stop after target-free authority checks")

                with mock.patch.object(
                    adapter_module,
                    "_execute_numerical_attempt",
                    side_effect=inspect_hybrid,
                ):
                    with self.assertRaises(observed.ObservedSessionError):
                        session.run_next_optional(authority)
                self.assertTrue(checks.get("valid"))
                self.assertEqual(session.invalid_attempt_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
