#!/usr/bin/env python3
"""Target-free adversarial tests for the chronology-owning observed session."""

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import pickle
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(TEST_PATH.parent) not in sys.path:
    sys.path.insert(0, str(TEST_PATH.parent))

import test_v3_m2_trial_executor_v1 as executor_tests  # noqa: E402
import test_v3_m2_permutation_m1_v1 as permutation_m1_tests  # noqa: E402
import test_v3_m2_permutation_production_adapter_v1 as permutation_production_tests  # noqa: E402
import fit_v3_m2 as fit_m2  # noqa: E402
import fit_v3_m2_optional as fit_optional  # noqa: E402
import v3_m2_observed_session_v1 as observed  # noqa: E402
import v3_m2_permutation_production_adapter_v1 as permutation_production  # noqa: E402
import v3_m2_rademacher_production_adapter_v1 as rademacher_production  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


class _WeakObject:
    pass


@dataclasses.dataclass
class _WeakResult:
    value: float


class ScriptedCallback:
    def __init__(
        self,
        branch: str = "none",
        *,
        gain_mode: str = "flat",
        core_winner: str = "trial2",
        failures: int = 0,
        resource: bool = False,
        raw_result: bool = False,
        rademacher_gain: float = 0.0,
    ) -> None:
        self.branch = branch
        self.gain_mode = gain_mode
        self.core_winner = core_winner
        self.failures = failures
        self.resource = resource
        self.raw_result = raw_result
        self.rademacher_gain = rademacher_gain
        self.calls: list[controller.Trial] = []
        self.session = None
        self.authority = None
        self.reenter = False
        self.reentry_errors: list[type[BaseException]] = []

    def _gain(self, trial: controller.Trial) -> float:
        if trial.kind == "rademacher_noise":
            return self.rademacher_gain
        if trial.role == "reference" or trial.falsifier:
            return 0.0
        configuration = trial.configuration
        gain = 0.01
        if self.core_winner == "trial2" and trial.valid_index == 2:
            gain = 0.05
        if self.core_winner == "tie" and trial.valid_index in {2, 3}:
            gain = 0.06
        if (
            self.core_winner == "elastic"
            and configuration is not None
            and configuration.family == "elastic_net"
            and configuration.alpha == 0.5
            and configuration.rho == 0.1
        ):
            gain = 0.06
        if (
            configuration is not None
            and configuration.family == "low_rank"
            and configuration.rank == 2
            and configuration.rho == 0.1
        ):
            gain = 0.08
        if (
            configuration is not None
            and trial.kind == "anchor"
            and configuration.family == "sparse_plus_low_rank"
            and configuration.rho == 0.001
            and configuration.sparse_rho == 0.001
        ):
            gain = 0.12
        if (
            self.gain_mode == "cap"
            and trial.kind == "successor"
            and trial.phase == "patience"
        ):
            gain = 0.20 + 0.003 * trial.valid_index
        return gain

    def _contrasts(self, trial: controller.Trial) -> tuple[float, ...]:
        values = [0.0] * controller.PAIR_COUNT
        if trial.valid_index == 2 and self.branch == "low":
            values[:] = [1.0] * controller.PAIR_COUNT
        elif trial.valid_index == 2 and self.branch == "hybrid":
            values[1] = 0.5
            values[7] = 0.3
            values[11] = 0.2
        return tuple(values)

    def __call__(self, trial: controller.Trial):
        self.calls.append(trial)
        if self.reenter:
            assert self.session is not None and self.authority is not None
            actions = (
                lambda: self.session.run_next_synthetic_qualification(self.authority),
                lambda: self.session.seal_resource_stop("nested"),
                lambda: observed.authorize_terminal_selection(self.session),
                lambda: observed.issue_synthetic_qualification_authority(
                    self.session,
                    ScriptedCallback(),
                    scopes=("core",),
                ),
            )
            for action in actions:
                try:
                    action()
                except BaseException as error:  # exact adversarial capture
                    self.reentry_errors.append(type(error))
                else:  # pragma: no cover - must fail closed
                    raise AssertionError("nested session action unexpectedly succeeded")
            self.reenter = False
        if self.resource:
            raise observed.ObservedResourceStop("synthetic resource stop")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("synthetic invalid attempt")
        result = controller.TrialResult(
            trial,
            self._gain(trial),
            1 if trial.role == "association" else 0,
        )
        if self.raw_result:
            return result
        return observed.SyntheticTrialOutcome(result, self._contrasts(trial))


class ObservedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        executor_tests.SharedExecutorTests.setUpClass()
        cls.cache = executor_tests.SharedExecutorTests.cache
        cls._terminal_fixtures = {}

    @classmethod
    def _fresh_cache(cls):
        source = cls.cache
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

    @classmethod
    def _new_session(
        cls,
        branch: str = "none",
        *,
        gain_mode: str = "flat",
        core_winner: str = "trial2",
        failures: int = 0,
        resource: bool = False,
        raw_result: bool = False,
        rademacher_gain: float = 0.0,
        scopes=("core", "optional", "falsifier"),
    ):
        owner = executor.CoreTrialExecutor(cls.cache, maximum_iterations=1)
        session = observed.start_synthetic_qualification_session(owner)
        callback = ScriptedCallback(
            branch,
            gain_mode=gain_mode,
            core_winner=core_winner,
            failures=failures,
            resource=resource,
            raw_result=raw_result,
            rademacher_gain=rademacher_gain,
        )
        diagnostic = controller.OptionalDiagnostics(
            0.39 if branch == "none" else 0.40,
            16.0,
            0.80,
            9,
        )
        authority = observed.issue_synthetic_qualification_authority(
            session,
            callback,
            scopes=scopes,
            diagnostic_override=diagnostic,
        )
        callback.session = session
        callback.authority = authority
        return owner, session, callback, authority

    @staticmethod
    def _drive(session, callback, authority) -> None:
        expected_count = (
            controller.MAXIMUM_VALID_TRIALS
            if callback.gain_mode == "cap"
            else {"none": 54, "low": 62, "hybrid": 66}[callback.branch]
        )
        for valid_index in range(1, expected_count + 1):
            result = session.run_next_synthetic_qualification(authority)
            assert result.trial.valid_index == valid_index

    @classmethod
    def _terminal_fixture(cls, name: str):
        if name not in cls._terminal_fixtures:
            if name == "ordinary":
                fixture = cls._new_session("none", core_winner="elastic")
            elif name == "hybrid":
                fixture = cls._new_session("hybrid")
            elif name == "cap":
                fixture = cls._new_session("hybrid", gain_mode="cap")
            else:  # pragma: no cover
                raise AssertionError(name)
            owner, session, callback, authority = fixture
            cls._drive(session, callback, authority)
            terminal = observed.authorize_terminal_selection(session)
            cls._terminal_fixtures[name] = (*fixture, terminal)
        return cls._terminal_fixtures[name]

    def test_all_branch_prefixes_and_earliest_atomic_stop(self) -> None:
        expected_counts = {"none": 54, "low": 62, "hybrid": 66}
        retained = []
        for branch, expected_count in expected_counts.items():
            owner, session, callback, authority = self._new_session(branch)
            retained.append((owner, session, callback, authority))
            with self.assertRaises(observed.ObservedSessionError):
                observed.authorize_terminal_selection(session)
            for valid_index in range(1, expected_count + 1):
                result = session.run_next_synthetic_qualification(authority)
                self.assertEqual(result.trial.valid_index, valid_index)
            self.assertEqual(session.status, observed.ORDINARY_PATIENCE_STOP)
            self.assertEqual(session.valid_trial_count, expected_count)
            self.assertEqual(session.scientific_conclusion, observed.ORDINARY_CONCLUSION)
            self.assertIsNone(session.next_trial)
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_synthetic_qualification(authority)
            self.assertEqual(len(callback.calls), expected_count)

        # A terminal capability is once-only even after its object is collected.
        owner, session, callback, authority = retained[0]
        terminal = observed.authorize_terminal_selection(session)
        self.assertIs(
            observed.require_selected_refit_authorization(owner, terminal),
            observed.require_terminal_selection(owner, terminal),
        )
        with self.assertRaises(observed.ObservedSessionError):
            observed.authorize_terminal_selection(session)
        del terminal
        gc.collect()
        with self.assertRaises(observed.ObservedSessionError):
            observed.authorize_terminal_selection(session)
        retained[0] = (owner, session, None, authority)
        del callback
        gc.collect()
        with self.assertRaises(observed.ObservedSessionError):
            session.status

    def test_exact_cap_is_incomplete_and_not_refit_authority(self) -> None:
        owner, session, callback, authority, terminal = self._terminal_fixture("cap")
        self.assertEqual(session.valid_trial_count, controller.MAXIMUM_VALID_TRIALS)
        self.assertEqual(session.status, observed.INCOMPLETE_BUDGET_STOP)
        self.assertEqual(
            terminal.scientific_conclusion,
            observed.INCOMPLETE_CONCLUSION,
        )
        winner = observed.require_terminal_selection(owner, terminal)
        self.assertIs(winner, terminal.winner_result)
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_selected_refit_authorization(owner, terminal)
        self.assertIsNone(session.next_trial)

    def test_terminal_selection_requires_frozen_rademacher_margin(self) -> None:
        blocked = self._new_session(
            "none",
            core_winner="trial2",
            rademacher_gain=0.049,
        )
        blocked_owner, blocked_session, blocked_callback, blocked_authority = blocked
        self._drive(blocked_session, blocked_callback, blocked_authority)
        with self.assertRaises(observed.ObservedSessionError):
            observed.authorize_terminal_selection(blocked_session)

        boundary = self._new_session(
            "none",
            core_winner="trial2",
            rademacher_gain=0.048,
        )
        boundary_owner, boundary_session, boundary_callback, boundary_authority = boundary
        self._drive(boundary_session, boundary_callback, boundary_authority)
        terminal = observed.authorize_terminal_selection(boundary_session)
        self.assertIs(
            observed.require_selected_refit_authorization(boundary_owner, terminal),
            observed.require_terminal_selection(boundary_owner, terminal),
        )
        self.assertTrue(blocked_owner is not None and blocked_callback.calls)

    def test_retry_resource_raw_and_reentry_fail_closed(self) -> None:
        owner, session, callback, authority = self._new_session(failures=12)
        expected = session.next_trial
        for attempt in range(controller.MAX_INVALID_RETRIES):
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_synthetic_qualification(authority)
            if attempt == 0:
                with self.assertRaises(observed.ObservedSessionError):
                    observed.issue_synthetic_qualification_authority(
                        session,
                        ScriptedCallback(),
                        scopes=("core",),
                        diagnostic_override=controller.OptionalDiagnostics(
                            0.40, 16.0, 0.80, 9
                        ),
                    )
            if attempt + 1 < controller.MAX_INVALID_RETRIES:
                self.assertIs(session.next_trial, expected)
        self.assertEqual(session.status, observed.TECHNICAL_RETRY_STOP)
        self.assertEqual(session.invalid_attempt_count, 12)
        self.assertEqual(session.valid_trial_count, 0)
        with self.assertRaises(observed.ObservedSessionError):
            observed.authorize_terminal_selection(session)
        with self.assertRaises(observed.ObservedSessionError):
            observed.issue_synthetic_qualification_authority(
                session, ScriptedCallback(), scopes=("core",)
            )

        _, resource_session, _, resource_authority = self._new_session(resource=True)
        with self.assertRaises(observed.ObservedResourceStop):
            resource_session.run_next_synthetic_qualification(resource_authority)
        self.assertEqual(resource_session.status, observed.TECHNICAL_RESOURCE_STOP)
        with self.assertRaises(observed.ObservedSessionError):
            observed.authorize_terminal_selection(resource_session)

        _, raw_session, _, raw_authority = self._new_session(raw_result=True)
        with self.assertRaises(observed.ObservedSessionError):
            raw_session.run_next_synthetic_qualification(raw_authority)
        self.assertEqual(raw_session.status, observed.TECHNICAL_INTEGRITY_STOP)
        self.assertEqual(raw_session.invalid_attempt_count, 0)

        _, nested, nested_callback, nested_authority = self._new_session()
        nested_callback.reenter = True
        nested.run_next_synthetic_qualification(nested_authority)
        self.assertEqual(nested.valid_trial_count, 1)
        self.assertEqual(nested.invalid_attempt_count, 0)
        self.assertEqual(len(nested_callback.calls), 1)
        self.assertEqual(
            nested_callback.reentry_errors,
            [observed.ObservedSessionError] * 4,
        )

    def test_authority_scope_cross_session_liveness_and_production_once(self) -> None:
        _, session, callback, core_authority = self._new_session(scopes=("core",))
        other_callback = ScriptedCallback()
        falsifier_authority = observed.issue_synthetic_qualification_authority(
            session,
            other_callback,
            scopes=("falsifier",),
        )
        with self.assertRaises(observed.ObservedSessionError):
            observed.issue_synthetic_qualification_authority(
                session, ScriptedCallback(), scopes=("core",)
            )
        with self.assertRaises(observed.ObservedSessionError):
            session.run_next_synthetic_qualification(falsifier_authority)
        self.assertEqual(session.invalid_attempt_count, 0)

        _, other_session, other_cb, other_auth = self._new_session()
        with self.assertRaises(observed.ObservedSessionError):
            session.run_next_synthetic_qualification(other_auth)

        dead_owner, dead_session, dead_callback, dead_authority = self._new_session()
        del dead_callback
        gc.collect()
        with self.assertRaises(observed.ObservedSessionError):
            dead_session.run_next_synthetic_qualification(dead_authority)

        # Once an external receipt exists, collecting its opaque authority
        # invalidates every later read instead of leaving a usable producer.
        _, issued_session, issued_callback, issued_authority = self._new_session()
        for _ in range(17):
            issued_session.run_next_synthetic_qualification(issued_authority)
        self.assertEqual(issued_session.valid_trial_count, 17)
        issued_callback.authority = None
        del issued_authority
        gc.collect()
        with self.assertRaises(observed.ObservedSessionError):
            issued_session.status

        production_cache = self._fresh_cache()
        first_owner = executor.CoreTrialExecutor(production_cache, maximum_iterations=1)
        second_owner = executor.CoreTrialExecutor(production_cache, maximum_iterations=1)
        self.assertFalse(hasattr(observed, "_production_cache_sessions"))
        with mock.patch.object(
            executor,
            "_require_prepared_cache",
            side_effect=lambda cache: cache,
        ), mock.patch.object(
            executor,
            "SEALED_PRODUCTION_MODE",
            executor.SYNTHETIC_TEST_MODE,
        ):
            production = observed.start_observed_session(first_owner)
            self.assertIsNotNone(production.next_trial)
            with self.assertRaises(observed.ObservedSessionError):
                observed.start_observed_session(second_owner)
            with self.assertRaises(observed.ObservedSessionError):
                observed._issue_adapter_authority(
                    production,
                    ScriptedCallback(),
                    mode="production",
                    scopes=("falsifier",),
                )

        # Retain live values through the end of the test.
        self.assertTrue(all((callback, core_authority, other_callback, other_cb, dead_owner)))

    def test_append_only_attempt_schedule_and_history_integrity(self) -> None:
        _, session, callback, authority = self._new_session(failures=1)
        with self.assertRaises(observed.ObservedSessionError):
            session.run_next_synthetic_qualification(authority)
        session.run_next_synthetic_qualification(authority)
        state = observed._lookup_session(session)

        original_attempt = state.attempts[0]
        state.attempts[0] = dataclasses.replace(original_attempt, ordinal=99)
        with self.assertRaises(observed.ObservedSessionError):
            session.status
        state.attempts[0] = original_attempt

        original_count = state.invalid_counts_at_results[0]
        state.invalid_counts_at_results[0] = original_count + 1
        with self.assertRaises(observed.ObservedSessionError):
            session.status
        state.invalid_counts_at_results[0] = original_count

        original_trial = state.schedule[1]
        object.__setattr__(original_trial, "valid_index", 99)
        with self.assertRaises(observed.ObservedSessionError):
            session.status
        object.__setattr__(original_trial, "valid_index", 2)

        original_entry = state.history[0]
        state.history[0] = dataclasses.replace(original_entry)
        with self.assertRaises(observed.ObservedSessionError):
            session.status
        state.history[0] = original_entry
        self.assertEqual(session.valid_trial_count, 1)
        self.assertTrue(callback.calls and authority is not None)

    def test_invalid_ledger_rollback_and_integrity_override_are_atomic(self) -> None:
        _, session, callback, authority = self._new_session()
        state = observed._lookup_session(session)
        trial = state.schedule[0]
        with mock.patch.object(
            observed,
            "_acknowledge_capability",
            side_effect=RuntimeError("target-free acknowledgment failure"),
        ):
            with self.assertRaises(RuntimeError):
                observed._record_invalid(
                    state,
                    trial,
                    id(authority),
                    RuntimeError("target-free attempt failure"),
                )
        self.assertEqual(state.invalid_attempts, 0)
        self.assertEqual(state.version, 0)
        self.assertEqual(state.attempts, [])
        self.assertEqual(state.attempt_digests, [])

        state.status = observed.TECHNICAL_RETRY_STOP
        prior_version = state.version
        observed._seal_integrity_stop(state)
        self.assertEqual(state.status, observed.TECHNICAL_INTEGRITY_STOP)
        self.assertEqual(state.version, prior_version + 1)
        self.assertIsNone(state.scientific_conclusion)
        self.assertTrue(callback is not None)

    def test_production_cache_claim_raw_scout_and_retry_capabilities(self) -> None:
        core = controller.coverage_trials()
        scouted_cache = self._fresh_cache()
        scouted_owner = executor.CoreTrialExecutor(
            scouted_cache, maximum_iterations=1
        )
        with mock.patch.object(
            executor,
            "_require_prepared_cache",
            side_effect=lambda cache: cache,
        ), mock.patch.object(
            executor,
            "SEALED_PRODUCTION_MODE",
            executor.SYNTHETIC_TEST_MODE,
        ):
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                scouted_owner.execute(core[0])
            with self.assertRaises(observed.ObservedSessionError):
                observed.start_observed_session(scouted_owner)

        claimed_cache = self._fresh_cache()
        owner = executor.CoreTrialExecutor(claimed_cache, maximum_iterations=1)
        other_owner = executor.CoreTrialExecutor(claimed_cache, maximum_iterations=1)
        with mock.patch.object(
            executor,
            "_require_prepared_cache",
            side_effect=lambda cache: cache,
        ), mock.patch.object(
            executor,
            "SEALED_PRODUCTION_MODE",
            executor.SYNTHETIC_TEST_MODE,
        ):
            session = observed.start_observed_session(owner)
            forged = object.__new__(executor.ProductionExecutionCapability)
            unissued = executor.ProductionExecutionCapability(
                _seal=executor._PRODUCTION_EXECUTION_CAPABILITY_SEAL
            )
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                owner.execute(core[0])
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                other_owner.execute(core[0])
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                owner.execute(core[0], _session_capability=forged)
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                owner.execute(core[0], _session_capability=unissued)
            state = observed._lookup_session(session)
            forged_claim = executor.ProductionCacheClaim(
                _seal=executor._PRODUCTION_CACHE_CLAIM_SEAL
            )
            for claim in (
                forged_claim,
                object.__new__(executor.ProductionCacheClaim),
            ):
                with self.assertRaises(executor.ProductionExecutionAuthorityError):
                    executor._require_production_cache_claim(
                        owner,
                        session,
                        claim,
                        0,
                        allow_pending=False,
                    )
            self.assertEqual(session.invalid_attempt_count, 0)
            self.assertEqual(session.valid_trial_count, 0)

            with mock.patch.object(
                owner,
                "_execute_trial",
                side_effect=RuntimeError("synthetic solver failure"),
            ):
                with self.assertRaises(observed.ObservedSessionError):
                    session.run_next_core()
            self.assertEqual(session.invalid_attempt_count, 1)
            self.assertEqual(session.valid_trial_count, 0)
            observed_result = session.run_next_core()
            self.assertEqual(observed_result.trial, core[0])
            self.assertEqual(session.invalid_attempt_count, 1)
            self.assertEqual(session.valid_trial_count, 1)

            state = observed._lookup_session(session)
            first, second = state.attempts
            self.assertIsNot(first.execution_capability, second.execution_capability)
            self.assertEqual(first.production_attempt_ordinal, 1)
            self.assertEqual(second.production_attempt_ordinal, 2)
            replay = second.execution_capability
            assert replay is not None
            for operation in (
                lambda: copy.copy(replay),
                lambda: copy.deepcopy(replay),
                lambda: pickle.dumps(replay),
            ):
                with self.assertRaises(executor.ProductionExecutionAuthorityError):
                    operation()
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                owner.execute(second.trial, _session_capability=replay)
            self.assertEqual(session.invalid_attempt_count, 1)
            self.assertEqual(session.valid_trial_count, 1)
            with self.assertRaises(observed.ObservedSessionError):
                observed.start_observed_session(other_owner)

    def test_production_capability_concurrent_issue_branch_and_owner_binding(self) -> None:
        core = controller.coverage_trials()
        claimed_cache = self._fresh_cache()
        owner = executor.CoreTrialExecutor(claimed_cache, maximum_iterations=1)
        other_owner = executor.CoreTrialExecutor(claimed_cache, maximum_iterations=1)
        with mock.patch.object(
            executor,
            "_require_prepared_cache",
            side_effect=lambda cache: cache,
        ), mock.patch.object(
            executor,
            "SEALED_PRODUCTION_MODE",
            executor.SYNTHETIC_TEST_MODE,
        ):
            session = observed.start_observed_session(owner)
            state = observed._lookup_session(session)
            current_trial = state.schedule[0]
            state.in_flight = True
            state.capability_issuance_open = True
            barrier = threading.Barrier(2)

            def issue():
                barrier.wait()
                try:
                    return executor._issue_production_execution_capability(
                        owner,
                        session,
                        state.production_claim,
                        current_trial,
                        None,
                    )
                except Exception as error:
                    return error

            with ThreadPoolExecutor(max_workers=2) as pool:
                issued = tuple(pool.map(lambda _: issue(), range(2)))
            capabilities = tuple(
                value
                for value in issued
                if type(value) is executor.ProductionExecutionCapability
            )
            failures = tuple(
                value
                for value in issued
                if type(value) is executor.ProductionExecutionAuthorityError
            )
            self.assertEqual(len(capabilities), 1)
            self.assertEqual(len(failures), 1)
            capability = capabilities[0]
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                owner.execute(state.schedule[1], _session_capability=capability)
            with self.assertRaises(executor.ProductionExecutionAuthorityError):
                other_owner.execute(current_trial, _session_capability=capability)
            state.capability_issuance_open = False
            state.in_flight = False

    def test_exact_named_falsifier_adapters_bind_through_live_session_hooks(self) -> None:
        m1_fixture = permutation_m1_tests.PermutationM1Tests()
        m1_fixture.setUp()
        execution = m1_fixture._fit()
        cache = permutation_production_tests._cache(execution)
        owner = executor.CoreTrialExecutor(
            cache,
            maximum_iterations=executor.FROZEN_MAXIMUM_ITERATIONS,
        )
        object.__setattr__(cache, "mode", executor.SEALED_PRODUCTION_MODE)
        object.__setattr__(
            cache,
            "expected_rank",
            executor.EXPECTED_PRODUCTION_RANK,
        )
        object.__setattr__(
            cache,
            "execution_inventory",
            executor.FrozenExecutionInventory.from_tsv(),
        )

        def physical_provenance(*args, **kwargs):
            del args, kwargs
            return SimpleNamespace(
                physical_rank=executor.EXPECTED_PRODUCTION_RANK,
            )

        def identity_cache(current):
            return current

        @dataclasses.dataclass
        class FeatureToken:
            candidates: object
            seed_route: object

        def authenticate_features(value):
            if type(value) is not FeatureToken:
                raise rademacher_production.RademacherExecutionAuthorityError(
                    "target-free feature token changed"
                )
            return value

        def numerical_runner(capability, record, adapter_state):
            del capability, adapter_state
            trial = record.trial_reference()
            if type(trial) is not controller.Trial:
                raise AssertionError("Rademacher trial died inside integration probe")
            return (), controller.TrialResult(trial, 0.0, 0)

        def registered_execution(adapter, execution_detail):
            if type(execution_detail) is not rademacher_production.RademacherTrialExecution:
                raise rademacher_production.RademacherExecutionAuthorityError(
                    "target-free execution detail changed"
                )
            binding = rademacher_production._lookup_execution(execution_detail)
            if (
                binding.adapter_reference() is not adapter
                or binding.execution_digest
                != executor._deep_integrity_digest(execution_detail)
            ):
                raise rademacher_production.RademacherExecutionAuthorityError(
                    "target-free execution binding changed"
                )
            return execution_detail.result

        @dataclasses.dataclass
        class PermutedM1Token:
            assignment: object
            candidates: object
            geometry: object
            permuted_cells: tuple[object, ...]
            fixed_outer_offsets: dict[str, object]
            selected_lambdas: dict[str, float]
            warm_path_count: int = 66
            zero_refit_count: int = 12
            full_development_refit_run: bool = False

        def fit_permuted_m1(
            cells,
            assignment,
            candidates,
            geometry,
            *,
            identity_prefix,
        ):
            del identity_prefix
            offsets = {
                fold.spec.held_out_group: fold.fixed_m1 for fold in cache.folds
            }
            return PermutedM1Token(
                assignment,
                candidates,
                geometry,
                assignment.apply(cells),
                offsets,
                {group: 1.0 for group in fit_m2.m1.GROUP_ORDER},
            )

        def require_permuted_m1(execution_detail):
            if type(execution_detail) is not PermutedM1Token:
                raise permutation_production.PermutationExecutionAuthorityError(
                    "target-free permutation M1 token changed"
                )
            return execution_detail

        def prepare_permuted_cache(*args, **kwargs):
            del args, kwargs
            return cache

        def configured_permutation_runner(prepared, trial, maximum_iterations):
            del prepared, maximum_iterations
            return (), controller.TrialResult(trial, 0.0, 0)

        def registered_permutation_execution(adapter, execution_detail):
            if type(execution_detail) is not permutation_production.PermutationTrialExecution:
                raise permutation_production.PermutationExecutionAuthorityError(
                    "target-free permutation execution detail changed"
                )
            binding = permutation_production._lookup_execution(execution_detail)
            if (
                binding.adapter_reference() is not adapter
                or binding.execution_digest
                != executor._deep_integrity_digest(execution_detail)
            ):
                raise permutation_production.PermutationExecutionAuthorityError(
                    "target-free permutation execution binding changed"
                )
            return execution_detail.result

        with ExitStack() as stack:
            patches = (
                mock.patch.object(
                    executor,
                    "_require_prepared_cache",
                    new=identity_cache,
                ),
                mock.patch.object(
                    permutation_production,
                    "_REQUIRE_PREPARED_CACHE",
                    new=identity_cache,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_PHYSICAL_PROVENANCE",
                    physical_provenance,
                ),
                mock.patch.object(
                    fit_m2,
                    "_require_physical_fold_provenance",
                    physical_provenance,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_FEATURE_FACTORY",
                    FeatureToken,
                ),
                mock.patch.object(
                    rademacher_production.rademacher,
                    "RademacherFeatureLibrary",
                    FeatureToken,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_FEATURE_AUTHENTICATE",
                    authenticate_features,
                ),
                mock.patch.object(
                    rademacher_production.rademacher,
                    "_require_issued_features",
                    authenticate_features,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_run_production_numerical",
                    numerical_runner,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_PRODUCTION_NUMERICAL_RUNNER",
                    numerical_runner,
                ),
                mock.patch.object(
                    rademacher_production,
                    "_require_registered_execution",
                    registered_execution,
                ),
                mock.patch.object(
                    permutation_production,
                    "_FIT_PERMUTED_M1",
                    fit_permuted_m1,
                ),
                mock.patch.object(
                    permutation_production.permutation_m1,
                    "fit_permuted_outer_m1",
                    fit_permuted_m1,
                ),
                mock.patch.object(
                    permutation_production,
                    "_REQUIRE_PERMUTED_M1_EXECUTION",
                    require_permuted_m1,
                ),
                mock.patch.object(
                    permutation_production.permutation_m1,
                    "_require_execution",
                    require_permuted_m1,
                ),
                mock.patch.object(
                    permutation_production,
                    "_PREPARE_PERMUTED_CACHE",
                    prepare_permuted_cache,
                ),
                mock.patch.object(
                    permutation_production.permutation_m1,
                    "prepare_permuted_m2_cache",
                    prepare_permuted_cache,
                ),
                mock.patch.object(
                    permutation_production,
                    "_CONFIGURED_M2_RUNNER",
                    configured_permutation_runner,
                ),
                mock.patch.object(
                    permutation_production,
                    "_execute_configured_core_m2",
                    configured_permutation_runner,
                ),
                mock.patch.object(
                    permutation_production,
                    "_validate_permuted_cache",
                    side_effect=lambda m1_execution, prepared, source: prepared,
                ),
                mock.patch.object(
                    permutation_production,
                    "_require_registered_execution",
                    registered_permutation_execution,
                ),
            )
            for patcher in patches:
                stack.enter_context(patcher)
            permutation_adapter = (
                permutation_production.create_whole_set_permutation_adapter(owner)
            )
            cells = permutation_production._source_cells(cache)
            moments = tuple(cache.geometry.uniform_moments(cell) for cell in cells)
            rademacher_adapter = (
                rademacher_production.create_rademacher_reference_adapter(
                    owner,
                    cells,
                    moments,
                )
            )
            session = observed.start_observed_session(owner)
            permutation_authority = observed.issue_whole_set_permutation_authority(
                session,
                permutation_adapter,
            )
            rademacher_authority = observed.issue_rademacher_reference_authority(
                session,
                rademacher_adapter,
            )
            self.assertEqual(session.valid_trial_count, 0)
            self.assertEqual(session.invalid_attempt_count, 0)
            self.assertEqual(session.status, observed.RUNNING)
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_falsifier(permutation_authority)
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_falsifier(rademacher_authority)
            self.assertEqual(session.valid_trial_count, 0)

            def execute_core(trial, *, authorization=None):
                del authorization
                gain = (
                    0.05
                    if trial.valid_index == 2
                    else 0.01
                    if trial.role == "association"
                    else 0.0
                )
                result = controller.TrialResult(
                    trial,
                    gain,
                    1 if trial.role == "association" else 0,
                )
                core_execution = executor.TrialExecution(
                    trial,
                    (),
                    (),
                    (0.0,) * controller.PAIR_COUNT,
                    result,
                )
                executor._register_owned_execution(owner, core_execution)
                return core_execution

            diagnostic_result = SimpleNamespace(
                diagnostics=controller.OptionalDiagnostics(0.0, 1.0, 0.0, 0),
            )
            with mock.patch.object(
                owner,
                "_execute_trial",
                side_effect=execute_core,
            ), mock.patch.object(
                observed.diagnostic,
                "diagnose_optional_families",
                return_value=diagnostic_result,
            ), mock.patch.object(
                observed,
                "_diagnostic_rows",
                return_value=(),
            ):
                for expected_index in range(1, controller.COVERAGE_TRIALS + 1):
                    result = session.run_next_core()
                    self.assertEqual(result.trial.valid_index, expected_index)

            self.assertEqual(session.next_trial.kind, "rademacher_noise")
            rademacher_result = session.run_next_falsifier(rademacher_authority)
            self.assertEqual(rademacher_result.trial.valid_index, 17)
            self.assertEqual(session.valid_trial_count, 17)
            session_state = observed._lookup_session(session)
            self.assertEqual(session_state.rademacher_production_attempts, 1)
            self.assertIsInstance(
                session_state.history[-1].execution_detail,
                rademacher_production.RademacherTrialExecution,
            )
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_falsifier(permutation_authority)

            for expected_index in range(18, 21):
                result = session.run_next_falsifier(rademacher_authority)
                self.assertEqual(result.trial.valid_index, expected_index)
            with mock.patch.object(
                owner,
                "_execute_trial",
                side_effect=execute_core,
            ):
                for expected_index in range(21, 28):
                    result = session.run_next_core()
                    self.assertEqual(result.trial.valid_index, expected_index)
            self.assertEqual(session.next_trial.kind, "permutation")
            with self.assertRaises(observed.ObservedSessionError):
                session.run_next_falsifier(rademacher_authority)
            permutation_result = session.run_next_falsifier(permutation_authority)
            self.assertEqual(permutation_result.trial.valid_index, 28)
            self.assertEqual(session.valid_trial_count, 28)
            self.assertEqual(session_state.rademacher_production_attempts, 4)
            self.assertEqual(session_state.permutation_production_attempts, 1)
            self.assertIsInstance(
                session_state.history[-1].execution_detail,
                permutation_production.PermutationTrialExecution,
            )

            with mock.patch.object(
                observed,
                "_rademacher_adapter_capability_request",
                lambda *args: True,
            ):
                with self.assertRaises(observed.ObservedSessionError):
                    _ = session.status

            self.assertEqual(session.status, observed.RUNNING)
            self.assertTrue(
                permutation_authority is not None
                and rademacher_authority is not None
            )

    def test_terminal_forgery_nested_mutation_and_other_owner(self) -> None:
        owner, session, callback, authority, terminal = self._terminal_fixture("ordinary")
        winner = observed.require_terminal_selection(owner, terminal)
        self.assertIs(
            observed.require_selected_refit_authorization(owner, terminal),
            winner,
        )
        with self.assertRaises(observed.ObservedSessionError):
            observed.TerminalSelectionAuthorization()
        forged = observed.TerminalSelectionAuthorization(
            _seal=observed._TERMINAL_AUTHORIZATION_SEAL
        )
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_terminal_selection(owner, forged)
        for operation in (
            lambda: copy.copy(terminal),
            lambda: copy.deepcopy(terminal),
            lambda: pickle.dumps(terminal),
        ):
            with self.assertRaises(observed.ObservedSessionError):
                operation()

        old_gain = winner.gain_bits_per_cell
        object.__setattr__(winner, "gain_bits_per_cell", old_gain + 1.0)
        with self.assertRaises(observed.ObservedSessionError):
            terminal.winner_result
        object.__setattr__(winner, "gain_bits_per_cell", old_gain)

        old_iterations = owner.maximum_iterations
        owner.maximum_iterations = old_iterations + 1
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_terminal_selection(owner, terminal)
        owner.maximum_iterations = old_iterations

        state = observed._lookup_session(session)
        unexecuted = state.schedule[-1]
        old_index = unexecuted.valid_index
        object.__setattr__(unexecuted, "valid_index", old_index + 1)
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_terminal_selection(owner, terminal)
        object.__setattr__(unexecuted, "valid_index", old_index)

        other_owner = executor.CoreTrialExecutor(self.cache, maximum_iterations=1)
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_terminal_selection(other_owner, terminal)
        self.assertIs(observed.require_terminal_selection(owner, terminal), winner)
        self.assertTrue(callback.calls and authority is not None)

    def test_every_configuration_field_and_winner_are_bound(self) -> None:
        owner, session, callback, authority, terminal = self._terminal_fixture("hybrid")
        winner = observed.require_terminal_selection(owner, terminal)
        configuration = winner.trial.configuration
        self.assertIsNotNone(configuration)
        assert configuration is not None
        self.assertEqual(configuration.family, "sparse_plus_low_rank")
        self.assertEqual(configuration.rank, 2)
        self.assertTrue(configuration.sparse_edges)

        replacements = {
            "family": "lasso",
            "rho": 1.0,
            "alpha": 0.5,
            "rank": 1,
            "sparse_rho": 1.0,
            "sparse_edges": tuple(sorted((*configuration.sparse_edges, 17))),
        }
        for field, changed in replacements.items():
            original = getattr(configuration, field)
            object.__setattr__(configuration, field, changed)
            with self.subTest(field=field):
                with self.assertRaises(observed.ObservedSessionError):
                    observed.require_terminal_selection(owner, terminal)
            object.__setattr__(configuration, field, original)

        state = observed._lookup_session(session)
        challenger = state.results[1]
        old_gain = challenger.gain_bits_per_cell
        object.__setattr__(challenger, "gain_bits_per_cell", winner.gain_bits_per_cell + 1.0)
        with self.assertRaises(observed.ObservedSessionError):
            observed.require_terminal_selection(owner, terminal)
        object.__setattr__(challenger, "gain_bits_per_cell", old_gain)
        self.assertIs(observed.require_terminal_selection(owner, terminal), winner)

        ordinary_owner, _, _, _, ordinary_terminal = self._terminal_fixture("ordinary")
        elastic = ordinary_terminal.selected_configuration
        self.assertEqual(elastic.family, "elastic_net")
        for field, changed in (("rho", 1.0), ("alpha", 0.25)):
            original = getattr(elastic, field)
            object.__setattr__(elastic, field, changed)
            with self.assertRaises(observed.ObservedSessionError):
                observed.require_terminal_selection(ordinary_owner, ordinary_terminal)
            object.__setattr__(elastic, field, original)

        tie_owner, tie_session, tie_callback, tie_authority = self._new_session(
            "none", core_winner="tie"
        )
        self._drive(tie_session, tie_callback, tie_authority)
        tie_terminal = observed.authorize_terminal_selection(tie_session)
        expected_tie_winner = controller.choose_incumbent(
            tuple(observed._lookup_session(tie_session).results)
        )
        self.assertIs(
            observed.require_terminal_selection(tie_owner, tie_terminal),
            expected_tie_winner,
        )
        self.assertTrue(callback.calls and authority is not None)

    @staticmethod
    def _selected_refit_stage_fixture(
        phase: str,
    ) -> tuple[
        object,
        object,
        object,
        object,
        observed._SelectedRefitBinding,
        object,
        object,
    ]:
        owner = _WeakObject()
        terminal = _WeakObject()
        session = _WeakObject()
        winner = _WeakObject()
        winner.trial = SimpleNamespace(
            configuration=SimpleNamespace(family="sparse_plus_low_rank")
        )
        authority = observed.SelectedRefitProcedureAuthorization(
            _seal=observed._SELECTED_REFIT_AUTHORIZATION_SEAL
        )
        binding = observed._SelectedRefitBinding(
            session_reference=weakref.ref(session),
            owner_reference=weakref.ref(owner),
            terminal_reference=weakref.ref(terminal),
            winner_reference=weakref.ref(winner),
            source=object(),
            phase=phase,
            hybrid_reserved=True,
        )
        observed._register_selected_refit(authority, binding)
        full = _WeakObject()
        receipt = _WeakObject()
        binding.hybrid_receipt_reference = weakref.ref(receipt)
        binding.hybrid_receipt_guard = receipt
        binding.hybrid_receipt_digest = b"receipt"
        # Keep every weakly referenced fixture object alive in the return value.
        return owner, terminal, authority, full, binding, winner, receipt

    def test_selected_refit_wrong_receipt_claims_burns_and_cannot_retry(
        self,
    ) -> None:
        (
            owner,
            terminal,
            authority,
            full,
            binding,
            winner,
            receipt,
        ) = self._selected_refit_stage_fixture("full_m1_ready")
        wrong_receipt = _WeakObject()
        with (
            mock.patch.object(
                observed,
                "_selected_refit_binding",
                return_value=(binding, winner),
            ),
            mock.patch.object(
                observed,
                "_require_selected_refit_full_objective",
                return_value=b"full",
            ),
        ):
            with self.assertRaisesRegex(
                observed.ObservedSessionError, "exact reserved receipt"
            ):
                observed._begin_selected_refit_fit_stage(
                    owner,
                    terminal,
                    authority,
                    full,
                    "precursor",
                    hybrid_receipt=wrong_receipt,
                )
            self.assertEqual(binding.phase, "burned")
            with self.assertRaisesRegex(
                observed.ObservedSessionError, "off chronology"
            ):
                observed._begin_selected_refit_fit_stage(
                    owner,
                    terminal,
                    authority,
                    full,
                    "precursor",
                    hybrid_receipt=receipt,
                )
            self.assertEqual(binding.phase, "burned")

    def test_selected_refit_issuance_is_permanent_after_gc_burn_or_complete(
        self,
    ) -> None:
        for terminal_state in ("live", "garbage_collected", "burned", "complete"):
            with self.subTest(terminal_state=terminal_state):
                owner = _WeakObject()
                terminal = _WeakObject()
                session = _WeakObject()
                winner = _WeakObject()
                candidates = _WeakObject()
                geometry = _WeakObject()
                pairs = _WeakObject()
                owner.prepared = SimpleNamespace(
                    candidates=candidates,
                    geometry=geometry,
                    pairs=pairs,
                )
                source = observed._SelectedRefitSource(
                    cells=(),
                    moments=(),
                    candidates_reference=weakref.ref(candidates),
                    geometry_reference=weakref.ref(geometry),
                    pairs_reference=weakref.ref(pairs),
                    cells_digest=executor._deep_integrity_digest(()),
                    moments_digest=executor._deep_integrity_digest(()),
                )
                state = SimpleNamespace(
                    selected_refit_source=source,
                    selected_refit_authority_issued=False,
                    selected_refit_authority_reference=None,
                )
                snapshot = SimpleNamespace(session_reference=weakref.ref(session))
                with (
                    mock.patch.object(
                        observed,
                        "require_selected_refit_authorization",
                        return_value=winner,
                    ),
                    mock.patch.object(
                        observed, "_lookup_terminal", return_value=snapshot
                    ),
                    mock.patch.object(
                        observed, "_require_session", return_value=state
                    ),
                ):
                    authority = observed.authorize_selected_refit_procedure(
                        owner, terminal
                    )
                    if terminal_state == "live":
                        for operation in (
                            lambda: copy.copy(authority),
                            lambda: copy.deepcopy(authority),
                            lambda: pickle.dumps(authority),
                        ):
                            with self.assertRaises(observed.ObservedSessionError):
                                operation()
                    elif terminal_state == "garbage_collected":
                        reference = weakref.ref(authority)
                        del authority
                        gc.collect()
                        self.assertIsNone(reference())
                    elif terminal_state == "burned":
                        observed._burn_selected_refit_procedure(authority)
                        self.assertEqual(
                            observed._lookup_selected_refit(authority).phase,
                            "burned",
                        )
                    else:
                        binding = observed._lookup_selected_refit(authority)
                        full = _WeakObject()
                        cold = _WeakResult(0.0)
                        warm = _WeakResult(1.0)
                        binding.full_objective_reference = weakref.ref(full)
                        binding.full_objective_guard = full
                        binding.cold_reference = weakref.ref(cold)
                        binding.cold_guard = cold
                        binding.cold_digest = executor._deep_integrity_digest(cold)
                        binding.warm_reference = weakref.ref(warm)
                        binding.warm_guard = warm
                        binding.warm_digest = executor._deep_integrity_digest(warm)
                        binding.phase = "warm_ready"
                        with mock.patch.object(
                            observed,
                            "_require_selected_refit_full_objective",
                            return_value=b"full",
                        ):
                            observed._complete_selected_refit_procedure(
                                authority, "selected_refit_report.json"
                            )
                        self.assertEqual(binding.phase, "complete")
                    with self.assertRaisesRegex(
                        observed.ObservedSessionError, "already issued"
                    ):
                        observed.authorize_selected_refit_procedure(
                            owner, terminal
                        )
                    self.assertTrue(state.selected_refit_authority_issued)

        with self.assertRaises(observed.ObservedSessionError):
            observed.SelectedRefitProcedureAuthorization()

    def test_full_m1_begin_claims_before_deep_failure_and_burns(self) -> None:
        (
            owner,
            terminal,
            authority,
            _full,
            binding,
            _winner,
            _receipt,
        ) = self._selected_refit_stage_fixture("issued_for_full_m1")
        with mock.patch.object(
            observed,
            "_selected_refit_binding",
            side_effect=observed.ObservedSessionError("deep source failure"),
        ):
            with self.assertRaisesRegex(
                observed.ObservedSessionError, "deep source failure"
            ):
                observed._begin_selected_refit_full_m1_request(
                    owner, terminal, authority
                )
            self.assertEqual(binding.phase, "burned")
            with self.assertRaisesRegex(
                observed.ObservedSessionError, "duplicated or is off chronology"
            ):
                observed._begin_selected_refit_full_m1_request(
                    owner, terminal, authority
                )
            self.assertEqual(binding.phase, "burned")

    def test_selected_refit_duplicate_full_m1_begin_burns_ready_objective(
        self,
    ) -> None:
        (
            owner,
            terminal,
            authority,
            full,
            binding,
            _winner,
            _receipt,
        ) = self._selected_refit_stage_fixture("full_m1_ready")
        binding.full_objective_reference = weakref.ref(full)
        binding.full_objective_guard = full
        with self.assertRaisesRegex(
            observed.ObservedSessionError,
            "duplicated or is off chronology",
        ):
            observed._begin_selected_refit_full_m1_request(
                owner, terminal, authority
            )
        self.assertEqual(binding.phase, "burned")
        self.assertIsNone(binding.full_objective_guard)
        with self.assertRaises(observed.ObservedSessionError):
            observed._begin_selected_refit_fit_stage(
                owner,
                terminal,
                authority,
                full,
                "precursor",
                hybrid_receipt=binding.hybrid_receipt_guard,
            )
        self.assertEqual(binding.phase, "burned")

    def test_selected_refit_final_pair_reauth_detects_post_warm_mutation(
        self,
    ) -> None:
        (
            _owner,
            _terminal,
            authority,
            _full,
            binding,
            _winner,
            _receipt,
        ) = self._selected_refit_stage_fixture("warm_ready")
        cold = _WeakResult(1.0)
        warm = _WeakResult(2.0)
        binding.cold_reference = weakref.ref(cold)
        binding.cold_guard = cold
        binding.cold_digest = executor._deep_integrity_digest(cold)
        binding.warm_reference = weakref.ref(warm)
        binding.warm_guard = warm
        binding.warm_digest = executor._deep_integrity_digest(warm)
        observed._require_selected_refit_fit_pair(authority, cold, warm)
        warm.value = 3.0
        with self.assertRaisesRegex(
            observed.ObservedSessionError,
            "changed before report commit",
        ):
            observed._require_selected_refit_fit_pair(authority, cold, warm)

    def test_selected_refit_guards_survive_callback_return_until_complete(
        self,
    ) -> None:
        (
            owner,
            terminal,
            authority,
            full,
            binding,
            _winner,
            receipt,
        ) = self._selected_refit_stage_fixture("warm_ready")
        precursor = _WeakResult(1.0)
        cold = _WeakResult(2.0)
        warm = _WeakResult(3.0)
        binding.full_objective_reference = weakref.ref(full)
        binding.full_objective_guard = full
        binding.precursor_reference = weakref.ref(precursor)
        binding.precursor_guard = precursor
        binding.precursor_digest = executor._deep_integrity_digest(precursor)
        binding.cold_reference = weakref.ref(cold)
        binding.cold_guard = cold
        binding.cold_digest = executor._deep_integrity_digest(cold)
        binding.warm_reference = weakref.ref(warm)
        binding.warm_guard = warm
        binding.warm_digest = executor._deep_integrity_digest(warm)
        references = tuple(
            weakref.ref(value) for value in (full, receipt, precursor, cold, warm)
        )
        del full, receipt, precursor, cold, warm
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        receipt_binding = SimpleNamespace(
            phase="warm_ready",
            selected_anchor_reference=references[2],
            cold_result_reference=references[3],
            warm_result_reference=references[4],
        )
        result_binding = SimpleNamespace(
            selected_anchor_reference=references[2],
            production_receipt_reference=references[1],
        )
        with (
            mock.patch.object(
                observed,
                "_require_selected_refit_full_objective",
                return_value=b"full",
            ) as require_full,
            mock.patch.object(
                fit_optional,
                "_require_terminal_full_development_hybrid_receipt",
                return_value=receipt_binding,
            ),
            mock.patch.object(
                fit_optional,
                "_require_registered_hybrid_result",
                return_value=result_binding,
            ) as require_result,
        ):
            observed._complete_selected_refit_procedure(
                authority, "selected_refit_report.json"
            )
        require_full.assert_called_once_with(
            owner,
            terminal,
            authority,
            references[0](),
        )
        self.assertEqual(require_result.call_count, 2)
        self.assertEqual(binding.phase, "complete")
        self.assertTrue(all(reference() is not None for reference in references))

    def test_selected_refit_completion_receipt_mutation_burns_all_guards(
        self,
    ) -> None:
        (
            owner,
            terminal,
            authority,
            full,
            binding,
            _winner,
            _receipt,
        ) = self._selected_refit_stage_fixture("warm_ready")
        precursor = _WeakResult(1.0)
        cold = _WeakResult(2.0)
        warm = _WeakResult(3.0)
        binding.full_objective_reference = weakref.ref(full)
        binding.full_objective_guard = full
        binding.precursor_reference = weakref.ref(precursor)
        binding.precursor_guard = precursor
        binding.precursor_digest = executor._deep_integrity_digest(precursor)
        binding.cold_reference = weakref.ref(cold)
        binding.cold_guard = cold
        binding.cold_digest = executor._deep_integrity_digest(cold)
        binding.warm_reference = weakref.ref(warm)
        binding.warm_guard = warm
        binding.warm_digest = executor._deep_integrity_digest(warm)
        with (
            mock.patch.object(
                observed,
                "_require_selected_refit_full_objective",
                return_value=b"full",
            ),
            mock.patch.object(
                fit_optional,
                "_require_terminal_full_development_hybrid_receipt",
                side_effect=fit_optional.LowRankError("mutated spec"),
            ),
        ):
            with self.assertRaisesRegex(
                observed.ObservedSessionError,
                "receipt reauthentication",
            ):
                observed._complete_selected_refit_procedure(
                    authority, "selected_refit_report.json"
                )
        self.assertEqual(binding.phase, "burned")
        self.assertIsNone(binding.full_objective_guard)
        self.assertIsNone(binding.hybrid_receipt_guard)
        self.assertIsNone(binding.precursor_guard)
        self.assertIsNone(binding.cold_guard)
        self.assertIsNone(binding.warm_guard)

    def test_selected_refit_substituted_or_mutated_prior_burns_and_cannot_retry(
        self,
    ) -> None:
        for stage, ready_phase, field in (
            ("cold", "precursor_ready", "precursor"),
            ("warm", "cold_ready", "cold"),
        ):
            with self.subTest(stage=stage, attack="substituted_identity"):
                (
                    owner,
                    terminal,
                    authority,
                    full,
                    binding,
                    winner,
                    receipt,
                ) = self._selected_refit_stage_fixture(ready_phase)
                prior = _WeakResult(1.0)
                setattr(binding, f"{field}_reference", weakref.ref(prior))
                setattr(binding, f"{field}_guard", prior)
                setattr(
                    binding,
                    f"{field}_digest",
                    executor._deep_integrity_digest(prior),
                )
                substitute = _WeakResult(1.0)
                with (
                    mock.patch.object(
                        observed,
                        "_selected_refit_binding",
                        return_value=(binding, winner),
                    ),
                    mock.patch.object(
                        observed,
                        "_require_selected_refit_full_objective",
                        return_value=b"full",
                    ),
                ):
                    with self.assertRaises(observed.ObservedSessionError):
                        observed._begin_selected_refit_fit_stage(
                            owner,
                            terminal,
                            authority,
                            full,
                            stage,
                            hybrid_receipt=receipt,
                            prior_result=substitute,
                        )
                    self.assertEqual(binding.phase, "burned")
                    with self.assertRaisesRegex(
                        observed.ObservedSessionError, "off chronology"
                    ):
                        observed._begin_selected_refit_fit_stage(
                            owner,
                            terminal,
                            authority,
                            full,
                            stage,
                            hybrid_receipt=receipt,
                            prior_result=prior,
                        )

            with self.subTest(stage=stage, attack="mutated_exact_result"):
                (
                    owner,
                    terminal,
                    authority,
                    full,
                    binding,
                    winner,
                    receipt,
                ) = self._selected_refit_stage_fixture(ready_phase)
                prior = _WeakResult(1.0)
                setattr(binding, f"{field}_reference", weakref.ref(prior))
                setattr(binding, f"{field}_guard", prior)
                setattr(
                    binding,
                    f"{field}_digest",
                    executor._deep_integrity_digest(prior),
                )
                prior.value = 2.0
                with (
                    mock.patch.object(
                        observed,
                        "_selected_refit_binding",
                        return_value=(binding, winner),
                    ),
                    mock.patch.object(
                        observed,
                        "_require_selected_refit_full_objective",
                        return_value=b"full",
                    ),
                ):
                    with self.assertRaises(observed.ObservedSessionError):
                        observed._begin_selected_refit_fit_stage(
                            owner,
                            terminal,
                            authority,
                            full,
                            stage,
                            hybrid_receipt=receipt,
                            prior_result=prior,
                        )
                    self.assertEqual(binding.phase, "burned")


if __name__ == "__main__":
    unittest.main()
