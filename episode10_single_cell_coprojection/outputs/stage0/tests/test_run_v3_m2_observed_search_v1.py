#!/usr/bin/env python3
"""Target-free tests for the sealed EP10 v3 observed-search runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_v3_m2_observed_search_v1 as runner  # noqa: E402


def _write_selected_refit_report(context: runner.PostselectionContext) -> str:
    payload = {
        "status": "pass",
        "terminal_authorization_reauthenticated": True,
        "terminal_winner_valid_index": (
            context.authorized_winner.trial.valid_index
        ),
        "selected_configuration_key": (
            context.terminal.selected_configuration.key
        ),
        "selected_pair_index": context.candidate_selection["pair_index"],
        "completion_gates": {
            name: True for name in runner.POSTSELECTION_REQUIRED_GATES
        },
        "opaque_authority_serialized": False,
        "final_unassigned_swc_or_m0_full_opened": False,
    }
    path = context.output_root / runner.POSTSELECTION_RESULT_FILENAME
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return runner.POSTSELECTION_RESULT_FILENAME


def _measurement(*, status: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "wall_seconds": 1.0,
        "cpu_seconds": 1.0,
        "peak_rss_gib": 1.0,
        "scratch_gib": 0.0,
    }
    if status:
        result["status"] = "pass"
    return result


def _prerequisites() -> dict[str, dict[str, object]]:
    scale_units = {
        name: _measurement(status=False) for name in runner.SCALE_REQUIRED_UNITS
    }
    falsifier_units = {
        name: _measurement(status=True) for name in runner.FALSIFIER_REQUIRED_UNITS
    }
    gates = {
        "cpu_core_hours": True,
        "gpu_hours": True,
        "wall_clock_hours": True,
        "parallel_cores": True,
        "aggregate_memory_GiB": True,
        "scratch_GiB": True,
        "per_task_measured_RSS_within_initial_request": True,
    }
    projection = {
        "observed_plus_99_replays_allocated_cpu_core_hours": 100.0,
        "gpu_hours": 0.0,
        "observed_plus_four_replay_waves_wall_clock_hours": 10.0,
        "parallel_cores": 64,
        "aggregate_memory_GiB": 256.0,
        "projected_scratch_GiB_no_deletion_credit": 10.0,
        "ceiling_gates": dict(gates),
        "all_ceiling_gates_pass": True,
    }
    optional_names = {
        "optional_trigger_diagnostic",
        "low_rank_population_recovery",
        "sparse_plus_low_rank_population_recovery",
    }
    return {
        "core": {
            "status": "pass",
            "overall_gate_status": "pass",
            "mode": "full",
            "stage": "v3_M2_target_free_frozen_core_contract_qualification",
            "gate_inventory": {
                "all_required_gates_present_and_passed": True,
                "required_gate_count": 8,
            },
            "actual_K_opened": False,
            "real_target_membership_or_M2_effects_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        },
        "actual_k": {
            "status": "pass",
            "stage": "v3_M2_actual_K_only_residual_rank_gate_v1",
            "rows": 194,
            "groups": 12,
            "actual_K_opened": True,
            "realized_positive_set_membership_opened": False,
            "final_or_unassigned_opened": False,
            "rank_gate": {
                "status": "pass",
                "training_folds_evaluated": 12,
                "training_folds_passed": 12,
                "expected_rank": 104,
                "rank_stable_in_every_fold": True,
            },
        },
        "m1_adequacy": {
            "status": "pass",
            "gate": "v3_M2_preaccess_M1_adequacy",
            "development_cells": 194,
            "development_groups": 12,
            "target_panel_size": 16,
            "M1_summary_and_outer_predictions_opened": True,
            "real_M2_outcomes_opened": False,
            "final_or_unassigned_opened": False,
            "equal_group_M1_minus_uniform_log_score_gain_bits_per_cell": 0.1,
            "equal_group_marginal_Brier_M1": 0.2,
            "equal_group_marginal_Brier_fixed_K_uniform": 0.3,
            "maximum_absolute_marginal_sum_K_error": 0.0,
        },
        "optional": {
            "status": "pass",
            "overall_gate_status": "pass",
            "stage": "v3_M2_target_free_optional_family_qualification_v1",
            "gate_inventory": {
                "all_required_gates_present_and_passed": True,
                "required_gate_count": 3,
            },
            "gates": {name: {"status": "pass"} for name in optional_names},
            "real_target_membership_or_M2_effects_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        },
        "scale": {
            "status": "pass_pending_sacct_and_remaining_budget_reconciliation",
            "stage": "v3_M2_membership_blind_full_search_scale_benchmark_v1",
            "policy_ceiling_projection_available": True,
            "policy_ceiling_projection_pass": True,
            "missing_unit_timings_imputed": False,
            "aggregate_only_report": True,
            "real_target_membership_or_M2_effects_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
            "measured_units": scale_units,
            "resource_projections": {
                "52_trial_fully_triggered_branch": dict(projection),
                "80_trial_cap": dict(projection),
            },
        },
        "falsifier_timing": {
            "status": "pass",
            "report_kind": "v3_m2_v4_target_free_falsifier_timing_supplement_v1",
            "aggregate_only": True,
            "fixture": {
                "source": "deterministic_in_memory_synthetic",
                "rows": 194,
                "groups": 12,
                "folds": 12,
                "expected_residual_rank": 104,
                "real_outcomes_opened": False,
                "protected_final_unassigned_or_SWC_opened": False,
                "existing_M0_M1_or_M2_result_opened": False,
            },
            "measured_units": falsifier_units,
            "rademacher_witness": {
                "frozen_seed_count": 7,
                "folds_per_probe": 12,
                "one_dictionary_shared_across_each_probe": True,
            },
            "permutation_witness": {
                "mapping_count": 1,
                "configured_M2_folds": 12,
                "own_fresh_M1_baseline_used": True,
                "full_development_M1_fit_run": False,
            },
        },
    }


class _FakeSession:
    def __init__(self) -> None:
        core = SimpleNamespace(
            valid_index=1,
            role="reference",
            kind="reference_anchor",
            phase="coverage",
            falsifier=False,
            cycle=None,
            seed=None,
            configuration=None,
        )
        low_rank_configuration = runner.controller.Configuration(
            "low_rank", 0.1, rank=1
        )
        optional = SimpleNamespace(
            valid_index=2,
            role="association",
            kind="anchor",
            phase="extension_coverage",
            falsifier=False,
            cycle=None,
            seed=None,
            configuration=low_rank_configuration,
        )
        core_configuration = runner.controller.Configuration("group_lasso", 0.1)
        permutation = SimpleNamespace(
            valid_index=3,
            role="association",
            kind="permutation",
            phase="postcoverage",
            falsifier=True,
            cycle=2,
            seed=2026092813,
            configuration=core_configuration,
        )
        rademacher = SimpleNamespace(
            valid_index=4,
            role="reference",
            kind="rademacher_noise",
            phase="postcoverage",
            falsifier=True,
            cycle=2,
            seed=2026092801,
            configuration=None,
        )
        self.trials = [core, optional, permutation, rademacher]
        self.index = 0
        self.routes: list[str] = []

    @property
    def next_trial(self) -> object | None:
        return None if self.index == len(self.trials) else self.trials[self.index]

    @property
    def status(self) -> str:
        return "ordinary_patience_stop" if self.next_trial is None else "running"

    @property
    def valid_trial_count(self) -> int:
        return self.index

    @property
    def invalid_attempt_count(self) -> int:
        return 0

    def _run(self, route: str) -> object:
        trial = self.next_trial
        assert trial is not None
        self.routes.append(route)
        self.index += 1
        return SimpleNamespace(
            trial=trial,
            valid=True,
            gain_bits_per_cell=0.01 * self.index,
            active_pair_degrees=self.index,
        )

    def run_next_core(self) -> object:
        return self._run("core")

    def run_next_optional(self, authority: object) -> object:
        del authority
        return self._run("optional")

    def run_next_falsifier(self, authority: object) -> object:
        return self._run(f"falsifier:{authority}")


class _FakeTerminal:
    def __init__(self, session: _FakeSession) -> None:
        configuration = runner.controller.Configuration("group_lasso", 0.1)
        trial = SimpleNamespace(valid_index=3, configuration=configuration)
        self.disposition = "ordinary_patience_stop"
        self.scientific_conclusion = "ordinary_search_complete"
        self.winner_result = SimpleNamespace(
            trial=trial,
            gain_bits_per_cell=0.03,
            active_pair_degrees=3,
        )
        self.selected_configuration = configuration
        self.valid_trial_count = session.valid_trial_count
        self.invalid_attempt_count = session.invalid_attempt_count


class _LedgerFailureSession:
    def __init__(self, *, ledgered_invalid: bool) -> None:
        self.trial = runner.controller.coverage_trials()[0]
        self.ledgered_invalid = ledgered_invalid
        self._status = runner.observed.RUNNING
        self._invalid_attempt_count = 0
        self.ledger_state = SimpleNamespace(invalid_attempts=0, attempts=[])

    @property
    def next_trial(self) -> object:
        return self.trial

    @property
    def status(self) -> str:
        return self._status

    @property
    def valid_trial_count(self) -> int:
        return 0

    @property
    def invalid_attempt_count(self) -> int:
        return self._invalid_attempt_count

    def run_next_core(self) -> object:
        underlying = ValueError("sealed numerical detail")
        if self.ledgered_invalid:
            self._invalid_attempt_count = 1
            self._status = runner.observed.TECHNICAL_RETRY_STOP
            self.ledger_state.invalid_attempts = 1
            self.ledger_state.attempts.append(
                SimpleNamespace(
                    trial=self.trial,
                    outcome="invalid",
                    invalid_attempts_after=1,
                    error_type="builtins.ValueError",
                )
            )
            raise runner.observed.ObservedSessionError(
                "numerical wrapper"
            ) from underlying
        self._status = runner.observed.TECHNICAL_INTEGRITY_STOP
        raise runner.observed.ObservedSessionError(
            "preflight integrity wrapper"
        ) from underlying


class _ResourceFailureSession:
    def __init__(self) -> None:
        self.trial = runner.controller.coverage_trials()[0]
        self._status = runner.observed.RUNNING
        self.ledger_state = SimpleNamespace(invalid_attempts=0, attempts=[])

    @property
    def next_trial(self) -> object:
        return self.trial

    @property
    def status(self) -> str:
        return self._status

    @property
    def valid_trial_count(self) -> int:
        return 0

    @property
    def invalid_attempt_count(self) -> int:
        return 0

    def run_next_core(self) -> object:
        self._status = runner.observed.TECHNICAL_RESOURCE_STOP
        self.ledger_state.attempts.append(
            SimpleNamespace(
                trial=self.trial,
                outcome="resource",
                invalid_attempts_after=0,
                error_type=(
                    "v3_m2_observed_session_v1.ObservedResourceStop"
                ),
            )
        )
        raise runner.observed.ObservedResourceStop("sealed resource detail")


class _SyntheticOwner:
    def __init__(self, prepared: object) -> None:
        self.prepared = prepared


def _real_core_execution_fixture(
    trial: object,
    *,
    gain: float = 0.0,
) -> tuple[object, object]:
    group_counts = runner.m1.GROUP_SIZES
    self_indexes: list[tuple[str, tuple[int, ...]]] = []
    cell_inventory: list[tuple[str, str]] = []
    cursor = 0
    for group in runner.m1.GROUP_ORDER:
        count = int(group_counts[group])
        indexes = tuple(range(cursor, cursor + count))
        self_indexes.append((group, indexes))
        cell_inventory.extend((f"cell_{index:03d}", group) for index in indexes)
        cursor += count
    assert cursor == runner.EXPECTED_ROWS

    prepared_folds = []
    execution_folds = []
    pair_rows = []
    zero_contrasts = (0.0,) * runner.controller.PAIR_COUNT
    all_indexes = tuple(range(runner.EXPECTED_ROWS))
    for ordinal, (group, validation_indexes) in enumerate(self_indexes, start=1):
        validation_set = set(validation_indexes)
        training_indexes = tuple(
            index for index in all_indexes if index not in validation_set
        )
        spec = runner.executor.FoldSpec(
            ordinal,
            group,
            training_indexes,
            validation_indexes,
        )
        fixed_m1 = runner.executor.FixedM1Offset(
            group,
            training_indexes,
            object(),
            object(),
            object(),
            f"fit_{group}",
        )
        prepared_folds.append(
            runner.executor.PreparedFold(
                spec,
                fixed_m1,
                object(),
                object(),
                object(),
                object(),
            )
        )
        cells = tuple(
            runner.executor.CellTrialResult(
                index,
                f"cell_{index:03d}",
                group,
                2,
                0.0,
                float(gain),
                zero_contrasts,
            )
            for index in validation_indexes
        )
        execution_folds.append(
            runner.executor.FoldTrialResult(
                spec,
                None,
                None,
                float(gain),
                0,
                cells,
            )
        )
        pair_rows.extend(
            runner.controller.make_pair_contrast_row(
                cell.cell_key,
                cell.group,
                cell.k,
                zero_contrasts,
            )
            for cell in cells
        )

    prepared = runner.executor.PreparedTrialCache(
        object(),
        object(),
        object(),
        tuple(prepared_folds),
        runner.executor.SYNTHETIC_TEST_MODE,
        None,
        tuple(cell_inventory),
        None,
        _seal=runner.executor._PREPARED_CACHE_SEAL,
    )
    runner.executor._register_prepared_cache(prepared)
    owner = _SyntheticOwner(prepared)
    result = runner.controller.TrialResult(trial, float(gain), 0)
    execution = runner.executor.TrialExecution(
        trial,
        tuple(execution_folds),
        tuple(pair_rows),
        zero_contrasts,
        result,
    )
    runner.executor._register_owned_execution(owner, execution)
    return owner, execution


class PrerequisiteTests(unittest.TestCase):
    def test_six_existing_reports_pass_without_hash_or_schema_layer(self) -> None:
        validated = runner.validate_prerequisite_reports(_prerequisites())
        self.assertEqual(set(validated.payloads), set(runner.REQUIRED_PREREQUISITES))
        self.assertTrue(all(value == "mapping" for value in validated.source_kinds.values()))

    def test_resource_inventory_and_protected_flags_fail_closed(self) -> None:
        reports = _prerequisites()
        reports["scale"]["measured_units"].pop("core_12_fold_trial")
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner.validate_prerequisite_reports(reports)

        reports = _prerequisites()
        reports["actual_k"]["final_or_unassigned_opened"] = True
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner.validate_prerequisite_reports(reports)

    def test_untriggered_optional_branch_uses_length_16_envelope(self) -> None:
        reports = _prerequisites()
        expensive = (
            reports["falsifier_timing"]["measured_units"]
            ["one_outer_only_permutation_M1_plus_configured_M2_fit"]
        )
        expensive["wall_seconds"] = 284.0

        # The former length-28 accounting passed cap 80 because it charged
        # only 52 expensive adaptive slots after 12 cheap optional trials.
        old_total_wall = 4.0 + 1.0 + 15.0 + 8.0 + 4.0 + 52.0 * 284.0
        replay_prefix_wall = 3.0
        old_cpu_core_hours = (
            2.0
            * (
                old_total_wall
                + 99.0 * (replay_prefix_wall + old_total_wall)
            )
            * 2.0
            / 3600.0
        )
        self.assertLess(
            old_cpu_core_hours,
            runner.RESOURCE_CEILINGS["cpu_core_hours"],
        )

        # With optionals untriggered, seven slots are branch-eligible
        # Rademacher probes and the other 57 cap-80 slots are charged at the
        # expensive permutation/association maximum, so reconciliation fails.
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner.validate_prerequisite_reports(reports)

    def test_route_rejection_is_component_based_not_falsifier_substring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "falsifier_timing_report.json"
            allowed.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                runner._read_report_path(allowed, "allowed"),
                {},
            )
            forbidden = root / "final" / "report.json"
            forbidden.parent.mkdir()
            forbidden.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(runner.ObservedSearchRunnerError):
                runner._read_report_path(forbidden, "forbidden")


class ComponentBindingTests(unittest.TestCase):
    def test_all_named_adapters_bind_while_session_is_pristine(self) -> None:
        calls: list[str] = []
        cache = object()
        owner = object()
        optional = object()
        permutation = object()
        rademacher = object()
        session = object()

        def step(name: str, value: object):
            def inner(*args: object, **kwargs: object) -> object:
                del args, kwargs
                calls.append(name)
                return value

            return inner

        with (
            mock.patch.object(
                runner.executor,
                "prepare_trial_cache",
                side_effect=step("prepare", cache),
            ),
            mock.patch.object(
                runner.executor,
                "CoreTrialExecutor",
                side_effect=step("owner", owner),
            ),
            mock.patch.object(
                runner.optional_adapter,
                "create_optional_family_adapter",
                side_effect=step("create_optional", optional),
            ),
            mock.patch.object(
                runner.permutation_adapter,
                "create_whole_set_permutation_adapter",
                side_effect=step("create_permutation", permutation),
            ),
            mock.patch.object(
                runner.rademacher_adapter,
                "create_rademacher_reference_adapter",
                side_effect=step("create_rademacher", rademacher),
            ),
            mock.patch.object(
                runner.observed,
                "start_observed_session",
                side_effect=step("start_session", session),
            ),
            mock.patch.object(
                runner.observed,
                "issue_optional_family_authority",
                side_effect=step("bind_optional", "optional_authority"),
            ),
            mock.patch.object(
                runner.observed,
                "issue_whole_set_permutation_authority",
                side_effect=step("bind_permutation", "permutation_authority"),
            ),
            mock.patch.object(
                runner.observed,
                "issue_rademacher_reference_authority",
                side_effect=step("bind_rademacher", "rademacher_authority"),
            ),
        ):
            result = runner.build_observed_search(
                cells=(object(),) * 194,
                moments=(object(),) * 194,
                candidates=object(),
                geometry=object(),
                pairs=object(),
                fixed_m1_offsets={group: object() for group in runner.m1.GROUP_ORDER},
                execution_inventory=object(),
            )
        self.assertIs(result.owner, owner)
        self.assertEqual(
            calls,
            [
                "prepare",
                "owner",
                "create_optional",
                "create_permutation",
                "create_rademacher",
                "start_session",
                "bind_optional",
                "bind_permutation",
                "bind_rademacher",
            ],
        )

    def test_numeric_serializer_rejects_unregistered_detail_type(self) -> None:
        trial = runner.controller.coverage_trials()[0]
        result = runner.controller.TrialResult(trial, 0.0, 0)
        components = SimpleNamespace()
        with (
            mock.patch.object(
                runner,
                "_authenticated_execution",
                return_value=object(),
            ),
            self.assertRaises(runner.ObservedSearchRunnerError),
        ):
            runner.serialize_committed_execution_detail(components, result)

    def test_runner_rejects_duck_or_relabelled_position_strata(self) -> None:
        cells = tuple(
            SimpleNamespace(
                identifier=f"cell_{index:03d}",
                group=runner.m1.GROUP_ORDER[index % len(runner.m1.GROUP_ORDER)],
                soma_um=(float(index), float(index % 7), float(index % 11)),
            )
            for index in range(runner.EXPECTED_ROWS)
        )
        exact = runner.input_loader._freeze_position_strata(cells)
        self.assertIs(runner._validate_position_strata(exact, cells), exact)
        duck = SimpleNamespace(
            medians_xyz=exact.medians_xyz,
            cell_keys=exact.cell_keys,
            groups=exact.groups,
            octants=exact.octants,
        )
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner._validate_position_strata(duck, cells)
        relabelled = runner.input_loader.PositionStrata(
            exact.medians_xyz,
            exact.cell_keys,
            exact.groups,
            ("111" if exact.octants[0] != "111" else "000",)
            + exact.octants[1:],
        )
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner._validate_position_strata(relabelled, cells)

    def test_real_core_fold_serializes_m1_from_exact_prepared_fold(self) -> None:
        trial = runner.controller.coverage_trials()[0]
        owner, execution = _real_core_execution_fixture(trial)
        state = SimpleNamespace(
            history=[execution],
            results=[execution.result],
            schedule=[trial],
            coverage_evidence=None,
        )
        components = SimpleNamespace(owner=owner, session=object())
        with mock.patch.object(
            runner.observed,
            "_require_session",
            return_value=state,
        ):
            detail = runner.serialize_committed_execution_detail(
                components,
                execution.result,
            )
        self.assertEqual(detail["execution_kind"], "core")
        self.assertEqual(len(detail["folds"]), runner.EXPECTED_FOLDS)
        self.assertEqual(
            [fold["fixed_m1"]["fit_identity"] for fold in detail["folds"]],
            [f"fit_{group}" for group in runner.m1.GROUP_ORDER],
        )

    def test_trial16_serializes_after_real_coverage_binding(self) -> None:
        trials = runner.controller.coverage_trials()
        owner, execution = _real_core_execution_fixture(trials[-1], gain=1.0)
        raw_results = [
            runner.controller.TrialResult(trial, 0.0, 0)
            for trial in trials[:-1]
        ] + [execution.result]
        history = [
            runner.executor.TrialExecution(
                trial,
                (),
                (),
                (0.0,) * runner.controller.PAIR_COUNT,
                result,
            )
            for trial, result in zip(
                trials[:-1],
                raw_results[:-1],
                strict=True,
            )
        ] + [execution]
        state = SimpleNamespace(
            owner=owner,
            history=history,
            results=list(raw_results),
            schedule=list(trials),
            synthetic_diagnostic=runner.controller.OptionalDiagnostics(
                0.0,
                1.0,
                0.0,
                0,
            ),
            coverage_evidence=None,
            coverage_evidence_digest=None,
        )
        runner.observed._bind_coverage(state)
        bound_result = state.results[-1]
        self.assertIsNot(bound_result, execution.result)
        self.assertIs(
            bound_result,
            state.coverage_evidence.core_results[-1],
        )
        components = SimpleNamespace(owner=owner, session=object())
        with mock.patch.object(
            runner.observed,
            "_require_session",
            return_value=state,
        ):
            detail = runner.serialize_committed_execution_detail(
                components,
                bound_result,
            )
        self.assertEqual(detail["trial"]["valid_index"], 16)
        self.assertTrue(detail["result"]["coverage_binding_present"])
        self.assertEqual(len(detail["folds"]), runner.EXPECTED_FOLDS)

    def test_invalid_attempt_has_separate_sealed_numeric_ledger_record(self) -> None:
        trial = runner.controller.coverage_trials()[0]
        session = SimpleNamespace(
            status=runner.observed.RUNNING,
            valid_trial_count=0,
            invalid_attempt_count=1,
        )
        record = runner._invalid_numeric_attempt_record(
            1,
            trial,
            "builtins.ValueError",
            session,
        )
        self.assertEqual(record["outcome"], "invalid")
        self.assertEqual(record["failure_class"], "builtins.ValueError")
        self.assertFalse(record["numerical_execution_detail_available"])
        self.assertFalse(record["opaque_authorities_or_capabilities_serialized"])
        self.assertNotIn("synthetic failure", json.dumps(record))

    def test_hybrid_configuration_key_and_edges_are_absent_from_aggregates(self) -> None:
        configuration = runner.controller.Configuration(
            "sparse_plus_low_rank",
            0.1,
            rank=1,
            sparse_rho=0.1,
            sparse_edges=(117, 119),
        )
        trial = runner.controller.Trial(
            2,
            "association",
            "anchor",
            "extension_coverage",
            False,
            configuration,
        )
        result = runner.controller.TrialResult(trial, 0.01, 2)
        session = SimpleNamespace(
            status=runner.observed.RUNNING,
            valid_trial_count=2,
            invalid_attempt_count=0,
        )
        terminal = SimpleNamespace(
            disposition=runner.observed.ORDINARY_PATIENCE_STOP,
            scientific_conclusion=runner.observed.ORDINARY_CONCLUSION,
            winner_result=result,
            selected_configuration=configuration,
            valid_trial_count=2,
            invalid_attempt_count=0,
        )
        prerequisites = runner.ValidatedPrerequisiteReports(
            payloads=_prerequisites(),
            source_kinds={
                name: "mapping" for name in runner.REQUIRED_PREREQUISITES
            },
        )
        aggregate_payloads = (
            runner._trial_record(2, trial, result, session),
            runner._invalid_attempt_record(
                2,
                trial,
                "builtins.ValueError",
                session,
            ),
            runner._terminal_summary(
                terminal,
                job_id="1",
                attempt_count=2,
                prerequisites=prerequisites,
                candidate_selection={"status": "selected"},
                selected_refit_authorized=True,
                callback_requested=False,
                callback_invoked=False,
                callback_status="not_requested",
                callback_result_linkage=None,
            ),
        )
        emitted = json.dumps(aggregate_payloads)
        self.assertNotIn(configuration.key, emitted)
        self.assertNotIn("configuration_key", emitted)
        self.assertNotIn("sparse_edges", emitted)
        self.assertNotIn("__edges_117_119", emitted)
        self.assertIn("sparse_plus_low_rank", emitted)


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.position_strata = object()
        patcher = mock.patch.object(
            runner,
            "_validate_position_strata",
            return_value=self.position_strata,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _root(self, scratch: Path, job_id: str) -> Path:
        return (
            scratch
            / "br_autoresearch"
            / "episode10_single_cell_coprojection"
            / f"{runner.OUTPUT_PREFIX}{job_id}"
        )

    def test_exact_cap_terminal_is_explicitly_incomplete_and_not_refit_authorized(self) -> None:
        configuration = runner.controller.Configuration("group_lasso", 0.1)
        terminal = SimpleNamespace(
            disposition=runner.observed.INCOMPLETE_BUDGET_STOP,
            scientific_conclusion=runner.observed.INCOMPLETE_CONCLUSION,
            winner_result=SimpleNamespace(
                trial=SimpleNamespace(valid_index=80),
                gain_bits_per_cell=0.01,
                active_pair_degrees=2,
            ),
            selected_configuration=configuration,
            valid_trial_count=80,
            invalid_attempt_count=0,
        )
        prerequisites = runner.ValidatedPrerequisiteReports(
            payloads=_prerequisites(),
            source_kinds={name: "mapping" for name in runner.REQUIRED_PREREQUISITES},
        )
        candidate = {"status": "not_run_incomplete_terminal"}
        summary = runner._terminal_summary(
            terminal,
            job_id="1",
            attempt_count=80,
            prerequisites=prerequisites,
            candidate_selection=candidate,
            selected_refit_authorized=False,
            callback_requested=False,
            callback_invoked=False,
            callback_status="not_requested",
            callback_result_linkage=None,
        )
        entries = [
            {
                "valid_index": index,
                "attempt_ordinal": index,
                "numeric_execution_file": f"numeric/committed_trial_{index:04d}.json",
            }
            for index in range(1, 81)
        ]
        manifest = runner._terminal_manifest(
            terminal,
            job_id="1",
            attempt_paths=tuple(Path(f"attempt_{index:04d}.json") for index in range(1, 81)),
            execution_entries=entries,
            invalid_execution_entries=(),
            coverage_rehydration={"coverage_length": 28},
            candidate_selection=candidate,
            selected_refit_authorized=False,
            callback_requested=False,
            callback_invoked=False,
            callback_status="not_requested",
            callback_result_linkage=None,
        )
        self.assertEqual(summary["status"], "incomplete_terminal")
        self.assertFalse(summary["selected_refit_authorized"])
        self.assertEqual(manifest["status"], "incomplete_terminal")
        self.assertFalse(manifest["selected_refit_authorized"])

        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner._terminal_manifest(
                terminal,
                job_id="1",
                attempt_paths=tuple(
                    Path(f"attempt_{index:04d}.json") for index in range(1, 80)
                ),
                execution_entries=entries,
                invalid_execution_entries=(),
                coverage_rehydration={"coverage_length": 28},
                candidate_selection=candidate,
                selected_refit_authorized=False,
                callback_requested=False,
                callback_invoked=False,
                callback_status="not_requested",
                callback_result_linkage=None,
            )
        with self.assertRaises(runner.ObservedSearchRunnerError):
            runner._terminal_manifest(
                terminal,
                job_id="1",
                attempt_paths=tuple(
                    Path(f"attempt_{index:04d}.json") for index in range(1, 81)
                ),
                execution_entries=entries,
                invalid_execution_entries=({"attempt_ordinal": 81},),
                coverage_rehydration={"coverage_length": 28},
                candidate_selection=candidate,
                selected_refit_authorized=False,
                callback_requested=False,
                callback_invoked=False,
                callback_status="not_requested",
                callback_result_linkage=None,
            )

    def test_incomplete_terminal_persists_diagnostics_but_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "24680"
            session = _FakeSession()
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority="OA",
                permutation_authority="PA",
                rademacher_authority="RA",
            )
            configuration = runner.controller.Configuration("group_lasso", 0.1)
            terminal = SimpleNamespace(
                disposition=runner.observed.INCOMPLETE_BUDGET_STOP,
                scientific_conclusion=runner.observed.INCOMPLETE_CONCLUSION,
                winner_result=SimpleNamespace(
                    trial=SimpleNamespace(valid_index=4),
                    gain_bits_per_cell=0.04,
                    active_pair_degrees=4,
                ),
                selected_configuration=configuration,
                valid_trial_count=4,
                invalid_attempt_count=0,
            )
            callback = mock.Mock()
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_terminal_selection",
                    return_value=terminal,
                ),
                mock.patch.object(
                    runner.observed,
                    "require_selected_refit_authorization",
                ) as require_refit,
                mock.patch.object(
                    runner,
                    "serialize_committed_execution_detail",
                    side_effect=lambda components, result: {
                        "record_kind": "test_numeric_execution",
                        "valid_index": result.trial.valid_index,
                    },
                ),
                mock.patch.object(
                    runner,
                    "_coverage_rehydration_record",
                    return_value={"coverage_length": 16},
                ),
                self.assertRaises(runner.ObservedSearchRunnerError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=callback,
                )
            callback.assert_not_called()
            require_refit.assert_not_called()
            root = self._root(scratch, job_id)
            summary = json.loads(
                (root / "terminal_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "terminal_manifest.json").read_text(encoding="utf-8")
            )
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "incomplete_terminal")
            self.assertEqual(manifest["status"], "incomplete_terminal")
            self.assertEqual(failure["stage"], "incomplete_terminal")
            self.assertFalse(summary["selected_refit_authorized"])
            self.assertFalse(summary["postselection_callback_requested"])
            self.assertFalse(summary["postselection_callback_invoked"])
            self.assertEqual(
                summary["postselection_callback_status"],
                "not_requested",
            )

    def test_ledgered_numerical_invalid_uses_underlying_failure_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "13579"
            session = _LedgerFailureSession(ledgered_invalid=True)
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority=object(),
                permutation_authority=object(),
                rademacher_authority=object(),
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "_require_session",
                    return_value=session.ledger_state,
                ),
                self.assertRaises(runner.observed.ObservedSessionError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=mock.Mock(),
                )
            root = self._root(scratch, job_id)
            numeric = json.loads(
                (root / "numeric" / "invalid_attempt_0001.json").read_text(
                    encoding="utf-8"
                )
            )
            aggregate = json.loads(
                (root / "attempt_0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(numeric["failure_class"], "builtins.ValueError")
            self.assertEqual(aggregate["failure_class"], "builtins.ValueError")
            self.assertNotIn("numerical wrapper", json.dumps(numeric))

    def test_preflight_wrapper_is_not_serialized_as_committed_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "97531"
            session = _LedgerFailureSession(ledgered_invalid=False)
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority=object(),
                permutation_authority=object(),
                rademacher_authority=object(),
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "_require_session",
                    return_value=session.ledger_state,
                ),
                self.assertRaises(runner.observed.ObservedSessionError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=mock.Mock(),
                )
            root = self._root(scratch, job_id)
            self.assertFalse((root / "attempt_0001.json").exists())
            self.assertEqual(tuple((root / "numeric").iterdir()), ())
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["completed_attempt_records"], 0)
            self.assertEqual(failure["completed_numeric_execution_records"], 0)

    def test_committed_resource_attempt_is_persisted_then_rethrown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "86420"
            session = _ResourceFailureSession()
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority=object(),
                permutation_authority=object(),
                rademacher_authority=object(),
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "_require_session",
                    return_value=session.ledger_state,
                ),
                self.assertRaises(runner.observed.ObservedResourceStop),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=mock.Mock(),
                )
            root = self._root(scratch, job_id)
            aggregate = json.loads(
                (root / "attempt_0001.json").read_text(encoding="utf-8")
            )
            numeric = json.loads(
                (root / "numeric" / "resource_attempt_0001.json").read_text(
                    encoding="utf-8"
                )
            )
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate["status"], "attempt_resource_stop")
            self.assertEqual(numeric["outcome"], "resource")
            self.assertEqual(failure["completed_attempt_records"], 1)
            self.assertEqual(failure["completed_numeric_execution_records"], 1)
            self.assertNotIn("sealed resource detail", json.dumps(aggregate))

    def test_serial_dispatch_atomic_aggregate_records_and_authorized_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "12345"
            session = _FakeSession()
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority="OA",
                permutation_authority="PA",
                rademacher_authority="RA",
            )
            events: list[str] = []
            selected_refit_authority = object()

            def authorize(owner: object, terminal: object) -> object:
                del owner
                events.append("authorized")
                return terminal.winner_result

            def callback(context: runner.PostselectionContext) -> str:
                self.assertIs(context.components, components)
                self.assertEqual(context.authorized_winner.trial.valid_index, 3)
                self.assertIs(
                    context.selected_refit_authority,
                    selected_refit_authority,
                )
                self.assertIs(context.position_strata, self.position_strata)
                self.assertEqual(events, ["authorized"])
                pending_summary = json.loads(
                    (
                        self._root(scratch, job_id) / "terminal_summary.json"
                    ).read_text(encoding="utf-8")
                )
                pending_manifest = json.loads(
                    (
                        self._root(scratch, job_id) / "terminal_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertTrue(pending_summary["postselection_callback_requested"])
                self.assertFalse(pending_summary["postselection_callback_invoked"])
                self.assertEqual(
                    pending_summary["postselection_callback_status"],
                    "pending",
                )
                self.assertEqual(
                    pending_manifest["postselection_callback_status"],
                    "pending",
                )
                events.append("callback")
                return _write_selected_refit_report(context)

            terminal = _FakeTerminal(session)
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_terminal_selection",
                    side_effect=lambda supplied: _FakeTerminal(supplied),
                ),
                mock.patch.object(
                    runner.observed,
                    "require_selected_refit_authorization",
                    side_effect=authorize,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_selected_refit_procedure",
                    return_value=selected_refit_authority,
                ) as issue_selected_refit,
                mock.patch.object(
                    runner.observed,
                    "_complete_selected_refit_procedure",
                ) as complete_selected_refit,
                mock.patch.object(
                    runner,
                    "serialize_committed_execution_detail",
                    side_effect=lambda components, result: {
                        "record_kind": "test_numeric_execution",
                        "valid_index": result.trial.valid_index,
                    },
                ),
                mock.patch.object(
                    runner,
                    "_select_winner_pair",
                    return_value={
                        "pair": "A--B",
                        "pair_index": 0,
                        "equal_group_probability_contrast": 0.1,
                        "direction": "positive",
                    },
                ),
                mock.patch.object(
                    runner,
                    "_coverage_rehydration_record",
                    return_value={"coverage_length": 16},
                ),
            ):
                result = runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=callback,
                )
            del terminal
            self.assertEqual(
                session.routes,
                ["core", "optional", "falsifier:PA", "falsifier:RA"],
            )
            self.assertEqual(events, ["authorized", "callback"])
            issue_selected_refit.assert_called_once()
            complete_selected_refit.assert_called_once_with(
                selected_refit_authority,
                runner.POSTSELECTION_RESULT_FILENAME,
            )
            self.assertTrue(result.postselection_callback_invoked)
            self.assertEqual(
                result.postselection_result_linkage,
                runner.POSTSELECTION_RESULT_FILENAME,
            )
            self.assertEqual(len(result.attempt_paths), 4)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "ordinary_terminal")
            self.assertTrue(summary["aggregate_only"])
            self.assertTrue(summary["selected_refit_authorized"])
            self.assertTrue(summary["postselection_callback_requested"])
            self.assertTrue(summary["postselection_callback_invoked"])
            self.assertEqual(summary["postselection_callback_status"], "complete")
            self.assertEqual(
                summary["postselection_result_linkage"],
                runner.POSTSELECTION_RESULT_FILENAME,
            )
            self.assertEqual(
                summary["candidate_selection_status"],
                "selected_in_sealed_terminal_manifest",
            )
            self.assertNotIn("A--B", json.dumps(summary))
            self.assertNotIn("equal_group_probability_contrast", json.dumps(summary))
            self.assertEqual(len(result.execution_paths), 4)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["winner_valid_index"], 3)
            self.assertEqual(len(manifest["committed_executions"]), 4)
            self.assertEqual(manifest["candidate_selection"]["pair"], "A--B")
            self.assertTrue(manifest["postselection_callback_invoked"])
            self.assertEqual(manifest["postselection_callback_status"], "complete")
            self.assertEqual(
                manifest["postselection_result_linkage"],
                runner.POSTSELECTION_RESULT_FILENAME,
            )
            emitted = "\n".join(
                path.read_text(encoding="utf-8") for path in result.attempt_paths
            )
            self.assertNotIn("cell_key", emitted)
            self.assertNotIn("membership", emitted)

    def test_callback_failure_preserves_pending_terminal_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "11223"
            session = _FakeSession()
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority="OA",
                permutation_authority="PA",
                rademacher_authority="RA",
            )
            selected_refit_authority = object()

            def authorize(owner: object, terminal: object) -> object:
                del owner
                return terminal.winner_result

            def callback(context: runner.PostselectionContext) -> str:
                del context
                raise RuntimeError("sealed callback detail")

            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_terminal_selection",
                    side_effect=lambda supplied: _FakeTerminal(supplied),
                ),
                mock.patch.object(
                    runner.observed,
                    "require_selected_refit_authorization",
                    side_effect=authorize,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_selected_refit_procedure",
                    return_value=selected_refit_authority,
                ),
                mock.patch.object(
                    runner.observed,
                    "_burn_selected_refit_procedure",
                ) as burn_selected_refit,
                mock.patch.object(
                    runner,
                    "serialize_committed_execution_detail",
                    side_effect=lambda components, result: {
                        "record_kind": "test_numeric_execution",
                        "valid_index": result.trial.valid_index,
                    },
                ),
                mock.patch.object(
                    runner,
                    "_select_winner_pair",
                    return_value={
                        "pair": "A--B",
                        "pair_index": 0,
                        "equal_group_probability_contrast": 0.1,
                        "direction": "positive",
                    },
                ),
                mock.patch.object(
                    runner,
                    "_coverage_rehydration_record",
                    return_value={"coverage_length": 16},
                ),
                self.assertRaises(RuntimeError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=callback,
                )
            root = self._root(scratch, job_id)
            summary = json.loads(
                (root / "terminal_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "terminal_manifest.json").read_text(encoding="utf-8")
            )
            failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["postselection_callback_status"], "pending")
            self.assertFalse(summary["postselection_callback_invoked"])
            self.assertEqual(manifest["postselection_callback_status"], "pending")
            self.assertFalse(manifest["postselection_callback_invoked"])
            self.assertEqual(failure["stage"], "postselection_callback")
            self.assertNotIn("sealed callback detail", json.dumps(failure))
            burn_selected_refit.assert_called_once_with(selected_refit_authority)

    def test_nonempty_job_root_refuses_ambiguous_resume_before_component_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "98765"
            root = self._root(scratch, job_id)
            root.mkdir(parents=True)
            (root / "attempt_0001.json").write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(runner, "build_observed_search") as build,
                self.assertRaises(runner.ObservedSearchRunnerError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=root,
                    job_id=job_id,
                    postselection_callback=mock.Mock(),
                )
            build.assert_not_called()

    def test_callback_never_runs_when_refit_authorization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "scratch"
            scratch.mkdir()
            job_id = "54321"
            session = _FakeSession()
            # Start terminal so the test reaches only authorization/callback.
            session.index = len(session.trials)
            components = runner.ObservedSearchComponents(
                owner=object(),
                session=session,
                optional=object(),
                permutation=object(),
                rademacher=object(),
                optional_authority=object(),
                permutation_authority=object(),
                rademacher_authority=object(),
            )
            callback = mock.Mock()
            with (
                mock.patch.dict(
                    os.environ,
                    {"SCRATCH": str(scratch), "SLURM_JOB_ID": job_id},
                    clear=False,
                ),
                mock.patch.object(
                    runner,
                    "build_observed_search",
                    return_value=components,
                ),
                mock.patch.object(
                    runner.observed,
                    "authorize_terminal_selection",
                    return_value=_FakeTerminal(session),
                ),
                mock.patch.object(
                    runner.observed,
                    "require_selected_refit_authorization",
                    side_effect=runner.observed.ObservedSessionError("not ordinary"),
                ),
                self.assertRaises(runner.observed.ObservedSessionError),
            ):
                runner.run_observed_search(
                    cells=(object(),) * 194,
                    moments=(object(),) * 194,
                    candidates=object(),
                    geometry=object(),
                    pairs=object(),
                    position_strata=self.position_strata,
                    fixed_m1_offsets={
                        group: object() for group in runner.m1.GROUP_ORDER
                    },
                    prerequisite_reports=_prerequisites(),
                    output_root=self._root(scratch, job_id),
                    job_id=job_id,
                    postselection_callback=callback,
                )
            callback.assert_not_called()
            failure = json.loads(
                (self._root(scratch, job_id) / "failure.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(failure["stage"], "terminal_selection")
            self.assertEqual(failure["exception_type"], "ObservedSessionError")


if __name__ == "__main__":
    unittest.main()
