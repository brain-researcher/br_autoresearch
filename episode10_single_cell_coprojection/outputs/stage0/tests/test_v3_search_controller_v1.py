#!/usr/bin/env python3
"""Focused tests for the pure EP10 v3 search-controller grammar."""

from __future__ import annotations

import dataclasses
import copy
import importlib.util
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v3_search_controller_v1.py"
SPEC = importlib.util.spec_from_file_location("v3_search_controller_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c
SPEC.loader.exec_module(c)


def _anchor_gain(trial):
    configuration = trial.configuration
    if (
        configuration is not None
        and configuration.family == "low_rank"
        and configuration.rank == 2
        and configuration.rho == 0.1
    ):
        return 0.03
    return 0.01 - 0.00001 * trial.valid_index


def core_results():
    return tuple(
        c.TrialResult(trial, _anchor_gain(trial), trial.valid_index % 5)
        for trial in c.coverage_trials()
    )


def unbound_core(results):
    scored = tuple(results)
    if scored and scored[-1].trial.valid_index == c.COVERAGE_TRIALS:
        scored = (*scored[:-1], dataclasses.replace(scored[-1], coverage_binding=None))
    return scored


def no_trigger_evidence(core=None):
    scored = core_results() if core is None else unbound_core(core)
    return c.bind_coverage_evidence(
        scored,
        c.OptionalDiagnostics(0.39, 16.0, 0.80, 9),
        (0.0,) * c.PAIR_COUNT,
    )


def low_rank_evidence(core=None):
    scored = core_results() if core is None else unbound_core(core)
    return c.bind_coverage_evidence(
        scored,
        c.OptionalDiagnostics(0.40, 16.0, 0.80, 9),
        (1.0,) * c.PAIR_COUNT,
    )


def optional_evidence(core=None):
    scored = core_results() if core is None else unbound_core(core)
    diagnostic = c.OptionalDiagnostics(0.40, 16.0, 0.80, 9)
    contrasts = [0.0] * 120
    contrasts[1] = 0.5
    contrasts[7] = 0.3
    contrasts[11] = 0.2
    evidence = c.bind_coverage_evidence(scored, diagnostic, tuple(contrasts))
    staged = c.coverage_trials(coverage_evidence=evidence)
    low_results = [
        c.TrialResult(
            trial,
            0.03
            if trial.configuration is not None
            and trial.configuration.rank == 2
            and trial.configuration.rho == 0.1
            else 0.01,
            4,
        )
        for trial in staged[16:24]
    ]
    return evidence, tuple(low_results)


def branch(*, optional=False, optional_branch=None):
    if optional_branch is None:
        optional_branch = "hybrid" if optional else "none"
    if optional_branch not in {"none", "low_rank", "hybrid"}:
        raise AssertionError("unknown test branch")
    core = core_results()
    if optional_branch == "hybrid":
        evidence, low_results = optional_evidence(core)
        coverage = c.coverage_trials(
            coverage_evidence=evidence,
            low_rank_anchor_results=low_results,
        )
    elif optional_branch == "low_rank":
        evidence = low_rank_evidence(core)
        coverage = c.coverage_trials(coverage_evidence=evidence)
    else:
        evidence = no_trigger_evidence(core)
        coverage = c.coverage_trials(coverage_evidence=evidence)
    coverage_results = [*evidence.core_results]
    coverage_results.extend(
        c.TrialResult(trial, _anchor_gain(trial), trial.valid_index % 5)
        for trial in coverage[16:]
    )
    tried = c._real_tried_configurations(coverage_results)
    one = c.successor_configurations(
        coverage_results,
        tried,
        count=7,
        coverage_evidence=evidence,
    )
    coverage_length = len(coverage)
    one_results = [
        c.TrialResult(
            c.Trial(
                coverage_length + 5 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                value,
                cycle=1,
                coverage_binding=evidence.binding,
            ),
            0.02 - 0.0001 * index,
            index % 4,
        )
        for index, value in enumerate(one)
    ]
    two = c.successor_configurations(
        (*coverage_results, *one_results),
        (*tried, *one),
        count=7,
        coverage_evidence=evidence,
    )
    trials = c.minimum_schedule(
        one,
        two,
        coverage_results=coverage_results,
        cycle_one_results=one_results,
        coverage_evidence=evidence,
    )
    results = list(coverage_results)
    for trial in trials[coverage_length:]:
        if trial.kind == "successor" and trial.cycle == 1:
            gain = 0.02 - 0.0001 * one.index(trial.configuration)
        else:
            gain = 0.01 if trial.role == "association" and not trial.falsifier else 0.0
        results.append(c.TrialResult(trial, gain, 4))
    return trials, results, one, two, evidence


def cycles():
    _, _, one, two, _ = branch()
    return one, two


def assert_substituted_evidence_rejected_everywhere(
    schedule, results, one, two, substituted_evidence
):
    coverage_length = next(
        index
        for index, trial in enumerate(schedule)
        if trial.kind == "rademacher_noise"
    )
    compact_cycle_one = (
        *results[:coverage_length],
        *results[coverage_length + 4 : coverage_length + 11],
    )
    checks = {
        "schedule_without_results": lambda: c.validate_schedule(
            schedule, coverage_evidence=substituted_evidence
        ),
        "schedule_with_results": lambda: c.validate_schedule(
            schedule,
            result_history=results,
            coverage_evidence=substituted_evidence,
        ),
        "history": lambda: c.validate_result_history(
            results, coverage_evidence=substituted_evidence
        ),
        "stop": lambda: c.stop_decision(results, 0, substituted_evidence),
        "patience": lambda: c.patience_streak(results, substituted_evidence),
        "decision_one": lambda: c.incumbent_challenger_decision(
            results[: coverage_length + 12],
            1,
            coverage_evidence=substituted_evidence,
        ),
        "decision_two": lambda: c.incumbent_challenger_decision(
            results, 2, coverage_evidence=substituted_evidence
        ),
        "successor_one": lambda: c.successor_configurations(
            results[:coverage_length], coverage_evidence=substituted_evidence
        ),
        "successor_two": lambda: c.successor_configurations(
            compact_cycle_one, coverage_evidence=substituted_evidence
        ),
        "continuation": lambda: c.continuation_block(
            coverage_length + 25,
            results,
            coverage_length=coverage_length,
            coverage_evidence=substituted_evidence,
        ),
        "minimum": lambda: c.minimum_schedule(
            one,
            two,
            coverage_results=results[:coverage_length],
            cycle_one_results=results[
                coverage_length + 4 : coverage_length + 11
            ],
            coverage_evidence=substituted_evidence,
        ),
    }
    for name, check in checks.items():
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError(f"substituted evidence escaped {name}")


def forge_trial_binding(trial, binding):
    forged = copy.copy(trial)
    object.__setattr__(forged, "coverage_binding", binding)
    return forged


def assert_poisoned_postcore_history_rejected_everywhere(
    schedule, results, one, two, evidence, binding
):
    coverage_length = next(
        index
        for index, trial in enumerate(schedule)
        if trial.kind == "rademacher_noise"
    )
    poisoned_schedule = list(schedule)
    poisoned_schedule[coverage_length] = forge_trial_binding(
        poisoned_schedule[coverage_length], binding
    )
    poisoned_results = list(results)
    poisoned_results[coverage_length] = dataclasses.replace(
        poisoned_results[coverage_length], trial=poisoned_schedule[coverage_length]
    )
    compact = [
        *results[:coverage_length],
        *results[coverage_length + 4 : coverage_length + 11],
    ]
    compact[-1] = dataclasses.replace(
        compact[-1], trial=forge_trial_binding(compact[-1].trial, binding)
    )
    checks = {
        "schedule_without_results": lambda: c.validate_schedule(
            poisoned_schedule, coverage_evidence=evidence
        ),
        "schedule_with_results": lambda: c.validate_schedule(
            poisoned_schedule,
            result_history=poisoned_results,
            coverage_evidence=evidence,
        ),
        "history": lambda: c.validate_result_history(
            poisoned_results, coverage_evidence=evidence
        ),
        "stop": lambda: c.stop_decision(poisoned_results, 0, evidence),
        "patience": lambda: c.patience_streak(poisoned_results, evidence),
        "decision_one": lambda: c.incumbent_challenger_decision(
            poisoned_results[: coverage_length + 12],
            1,
            coverage_evidence=evidence,
        ),
        "decision_two": lambda: c.incumbent_challenger_decision(
            poisoned_results, 2, coverage_evidence=evidence
        ),
        "successor": lambda: c.successor_configurations(
            compact, coverage_evidence=evidence
        ),
        "continuation": lambda: c.continuation_block(
            coverage_length + 25,
            poisoned_results,
            coverage_length=coverage_length,
            coverage_evidence=evidence,
        ),
        "minimum": lambda: c.minimum_schedule(
            one,
            two,
            coverage_results=results[:coverage_length],
            cycle_one_results=compact[-7:],
            coverage_evidence=evidence,
        ),
    }
    for name, check in checks.items():
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError(f"poisoned post-core binding escaped {name}")


def mismatched_hybrid_history():
    """Build the former bypass: rank-8 scores with scheduled rank-2 hybrids."""
    evidence, low_results = optional_evidence()
    coverage = c.coverage_trials(
        coverage_evidence=evidence,
        low_rank_anchor_results=low_results,
    )
    coverage_results = list(evidence.core_results)
    for trial in coverage[16:]:
        gain = _anchor_gain(trial)
        configuration = trial.configuration
        if 17 <= trial.valid_index <= 24:
            assert configuration is not None and configuration.family == "low_rank"
            gain = 1.0 if configuration.rank == 8 else -1.0
        coverage_results.append(c.TrialResult(trial, gain, 4))
    assert c.best_low_rank_anchor_rank(coverage_results[16:24], evidence) == 8
    assert {
        result.trial.configuration.rank for result in coverage_results[24:28]
    } == {2}

    tried = c._real_tried_configurations(coverage_results)
    one = c._successor_configurations_unchecked(
        coverage_results, tried, count=7
    )
    one_results = tuple(
        c.TrialResult(
            c.Trial(
                33 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                configuration,
                cycle=1,
                coverage_binding=evidence.binding,
            ),
            0.02 - 0.0001 * index,
            4,
        )
        for index, configuration in enumerate(one)
    )
    two = c._successor_configurations_unchecked(
        (*coverage_results, *one_results), (*tried, *one), count=7
    )
    schedule = coverage + c.postcoverage_trials(
        one,
        two,
        start_valid_index=29,
        coverage_evidence=evidence,
    )
    results = list(coverage_results)
    for trial in schedule[28:]:
        if trial.kind == "successor" and trial.cycle == 1:
            gain = 0.02 - 0.0001 * one.index(trial.configuration)
        else:
            gain = 0.01 if trial.role == "association" and not trial.falsifier else 0.0
        results.append(c.TrialResult(trial, gain, 4))
    return schedule, results, evidence


def fabricated_hybrid_edge_history():
    """Recompute a self-consistent hybrid history around fabricated edge 0."""
    _, legitimate_results, _, _, evidence = branch(optional=True)
    coverage_results = list(legitimate_results[:28])
    for index in range(24, 28):
        configuration = coverage_results[index].trial.configuration
        assert configuration is not None
        assert configuration.family == "sparse_plus_low_rank"
        fabricated = dataclasses.replace(configuration, sparse_edges=(0,))
        coverage_results[index] = dataclasses.replace(
            coverage_results[index],
            trial=dataclasses.replace(
                coverage_results[index].trial, configuration=fabricated
            ),
        )
    one = c._successor_configurations_unchecked(
        coverage_results,
        c._real_tried_configurations(coverage_results),
        count=7,
    )
    one_results = tuple(
        c.TrialResult(
            c.Trial(
                33 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                configuration,
                cycle=1,
                coverage_binding=evidence.binding,
            ),
            0.02 - 0.0001 * index,
            4,
        )
        for index, configuration in enumerate(one)
    )
    two = c._successor_configurations_unchecked(
        (*coverage_results, *one_results),
        c._real_tried_configurations((*coverage_results, *one_results)),
        count=7,
    )
    schedule = tuple(result.trial for result in coverage_results) + c.postcoverage_trials(
        one,
        two,
        start_valid_index=29,
        coverage_evidence=evidence,
    )
    results = list(coverage_results)
    for trial in schedule[28:]:
        if trial.kind == "successor" and trial.cycle == 1:
            gain = 0.02 - 0.0001 * one.index(trial.configuration)
        else:
            gain = 0.01 if trial.role == "association" and not trial.falsifier else 0.0
        results.append(c.TrialResult(trial, gain, 4))
    return schedule, results, one, two, evidence


def extended_results(last_valid_index: int):
    base_trials, base_results, _, _, evidence = branch()
    trials = list(base_trials)
    results = list(base_results)
    tried = list(c._real_tried_configurations(results))
    for start in range(41, last_valid_index + 1, 5):
        block = c.continuation_block(
            start, results, tried, coverage_evidence=evidence
        )
        for trial in block:
            if trial.valid_index > last_valid_index:
                break
            trials.append(trial)
            results.append(
                c.TrialResult(
                    trial,
                    0.01
                    if trial.role == "association" and not trial.falsifier
                    else 0.0,
                    4,
                )
            )
            if trial.kind == "successor" and trial.configuration is not None:
                tried.append(trial.configuration)
    return trials, results, evidence


def test_anchor_inventory_and_boundaries() -> None:
    anchors = c.anchor_configurations()
    assert len(anchors) == 15
    assert {value.family for value in anchors} == {
        "lasso",
        "elastic_net",
        "group_lasso",
    }
    assert {value.alpha for value in anchors if value.family == "elastic_net"} == {
        0.25,
        0.5,
        0.75,
    }
    for family in ("lasso", "elastic_net", "group_lasso"):
        values = [value.rho for value in anchors if value.family == family]
        assert 1.0 in values and 0.001 in values


def test_frozen_inventory_matches_embedded_exact_identity_tuple() -> None:
    assert len(c.EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES) == 194
    assert (
        c.FROZEN_CELL_INVENTORY.entries
        == c.EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES
    )


def test_minimum_schedule_exact_counts() -> None:
    trials, _, one, two, _ = branch()
    assert len(trials) == 40
    assert sum(value.role == "reference" for value in trials) == 8
    assert sum(value.role == "association" for value in trials) == 32
    assert sum(value.falsifier for value in trials[16:]) == 10
    assert math.isclose(10 / 24, 5 / 12)
    assert sum(value.kind == "successor" and value.cycle == 1 for value in trials) == 7
    assert sum(value.kind == "successor" and value.cycle == 2 for value in trials) == 7
    noise = [value for value in trials if value.kind == "rademacher_noise"]
    assert [value.matched_configuration for value in noise[:4]] == list(one[:4])
    assert [value.matched_configuration for value in noise[4:]] == list(two[:3])
    assert {value.configuration.family for value in trials if value.kind == "permutation"} == {
        "lasso",
        "elastic_net",
        "group_lasso",
    }


def test_successors_are_unique_and_untried() -> None:
    one, two = cycles()
    anchors = {value.key for value in c.anchor_configurations()}
    assert not anchors.intersection(value.key for value in one)
    assert not anchors.intersection(value.key for value in two)
    assert not {value.key for value in one}.intersection(value.key for value in two)


def test_incumbent_tie_order() -> None:
    trials = c.coverage_trials()[1:4]
    results = [
        c.TrialResult(trials[0], 0.1, 3),
        c.TrialResult(trials[1], 0.1, 2),
        c.TrialResult(trials[2], 0.1, 2),
    ]
    winner = c.choose_incumbent(results)
    assert winner.trial.configuration == min(
        (result.trial.configuration for result in results),
        key=c.configuration_order_key,
    )


def test_optional_anchor_inventory() -> None:
    low = c.low_rank_anchor_configurations()
    assert len(low) == 8
    assert {value.rank for value in low} == set(c.LOW_RANKS)
    evidence, low_results = optional_evidence()
    hybrid = c.bound_optional_anchor_configurations(
        evidence, low_results
    )
    assert len(hybrid) == 12
    assert sum(value.family == "sparse_plus_low_rank" for value in hybrid) == 4
    assert {
        value.rank for value in hybrid if value.family == "sparse_plus_low_rank"
    } == {2}
    assert {
        value.sparse_edges for value in hybrid if value.family == "sparse_plus_low_rank"
    } == {(1, 7)}


def test_optional_trigger_requires_diffuse_and_localized_evidence() -> None:
    diffuse = c.OptionalDiagnostics(0.40, 16.0, 0.80, 9)
    broad = c.localized_sparse_selection([1.0] * 120)
    assert c.optional_activation(diffuse, broad) == c.OptionalActivation(True, False)
    evidence, _ = optional_evidence()
    localized = evidence.localized
    assert localized.edges == (1, 7) and localized.eligible
    assert c.optional_activation(diffuse, localized) == c.OptionalActivation(True, True)
    not_diffuse = c.OptionalDiagnostics(0.39, 16.0, 0.80, 9)
    assert c.optional_activation(not_diffuse, localized) == c.OptionalActivation(False, False)
    try:
        c.OptionalDiagnostics(0.40, 121.0, 0.80, 9)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("impossible participation ratio accepted")


def test_optional_diagnostic_cosine_count_is_an_exact_integer() -> None:
    for value in (0, 12):
        assert (
            c.OptionalDiagnostics(0.40, 16.0, 0.80, value)
            .leave_one_group_cosines_at_least_0_70
            == value
        )
    for value in (-1, 13, False, True, 9.0, 9.5):
        try:
            c.OptionalDiagnostics(0.40, 16.0, 0.80, value)
        except c.ControllerError:
            pass
        else:
            raise AssertionError(f"non-integer/invalid cosine count accepted: {value!r}")


def test_every_postcore_trial_and_trial16_result_carry_one_binding() -> None:
    for branch_name in ("none", "low_rank", "hybrid"):
        schedule, results, _, _, evidence = branch(optional_branch=branch_name)
        assert results[15].coverage_binding == evidence.binding
        assert all(result.coverage_binding is None for result in results[:15])
        assert all(
            trial.coverage_binding == evidence.binding
            for trial in schedule
            if trial.valid_index >= 17
        )
        coverage_length = next(
            index
            for index, trial in enumerate(schedule)
            if trial.kind == "rademacher_noise"
        )
        block = c.continuation_block(
            coverage_length + 25,
            results,
            coverage_length=coverage_length,
            coverage_evidence=evidence,
        )
        assert all(trial.coverage_binding == evidence.binding for trial in block)


def test_same_branch_diagnostic_replacement_rejected_everywhere() -> None:
    schedule, results, one, two, evidence = branch()
    substituted = c.bind_coverage_evidence(
        unbound_core(evidence.core_results),
        c.OptionalDiagnostics(0.38, 16.0, 0.80, 9),
        evidence.equal_group_pair_contrasts,
    )
    assert substituted.activation == evidence.activation
    assert substituted.binding != evidence.binding
    assert_substituted_evidence_rejected_everywhere(
        schedule, results, one, two, substituted
    )


def test_scaled_same_edge_contrasts_rejected_everywhere() -> None:
    schedule, results, one, two, evidence = branch(optional=True)
    scaled = tuple(2.0 * value for value in evidence.equal_group_pair_contrasts)
    substituted = c.bind_coverage_evidence(
        unbound_core(evidence.core_results), evidence.diagnostic, scaled
    )
    assert substituted.localized.edges == evidence.localized.edges == (1, 7)
    assert substituted.activation == evidence.activation
    assert substituted.binding != evidence.binding
    assert_substituted_evidence_rejected_everywhere(
        schedule, results, one, two, substituted
    )


def test_wrong_core_evidence_rejected_everywhere() -> None:
    schedule, results, one, two, evidence = branch()
    wrong_core = list(unbound_core(evidence.core_results))
    wrong_core[0] = dataclasses.replace(
        wrong_core[0], gain_bits_per_cell=wrong_core[0].gain_bits_per_cell + 0.0001
    )
    substituted = c.bind_coverage_evidence(
        wrong_core, evidence.diagnostic, evidence.equal_group_pair_contrasts
    )
    assert substituted.activation == evidence.activation
    assert substituted.binding != evidence.binding
    assert_substituted_evidence_rejected_everywhere(
        schedule, results, one, two, substituted
    )


def test_missing_and_changed_postcore_binding_rejected_everywhere() -> None:
    schedule, results, one, two, evidence = branch()
    try:
        dataclasses.replace(schedule[16], coverage_binding=None)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("a newly constructed post-core trial omitted binding")

    assert_poisoned_postcore_history_rejected_everywhere(
        schedule, results, one, two, evidence, None
    )
    replacement = c.bind_coverage_evidence(
        unbound_core(evidence.core_results),
        c.OptionalDiagnostics(0.38, 16.0, 0.80, 9),
        evidence.equal_group_pair_contrasts,
    )
    assert_poisoned_postcore_history_rejected_everywhere(
        schedule, results, one, two, evidence, replacement.binding
    )


def test_schedule_only_rejects_binding_poison_at_every_postcore_position() -> None:
    schedule, results, evidence = extended_results(45)
    replacement = c.bind_coverage_evidence(
        unbound_core(evidence.core_results),
        c.OptionalDiagnostics(0.38, 16.0, 0.80, 9),
        evidence.equal_group_pair_contrasts,
    )
    assert replacement.binding != evidence.binding
    assert any(
        trial.valid_index >= 43 and trial.kind == "successor"
        for trial in schedule
    )
    for position in range(c.COVERAGE_TRIALS, len(schedule)):
        changed = list(schedule)
        changed[position] = dataclasses.replace(
            changed[position], coverage_binding=replacement.binding
        )
        missing = list(schedule)
        missing[position] = forge_trial_binding(missing[position], None)
        for label, poisoned in (("changed", changed), ("missing", missing)):
            try:
                c.validate_schedule(poisoned, coverage_evidence=evidence)
            except c.ControllerError:
                pass
            else:
                raise AssertionError(
                    f"schedule-only validation accepted {label} binding at "
                    f"valid index {schedule[position].valid_index}"
                )


def test_trial16_binding_and_core_float_bits_are_mandatory() -> None:
    schedule, results, one, two, evidence = branch()
    missing = list(results)
    missing[15] = dataclasses.replace(missing[15], coverage_binding=None)
    changed_evidence = c.bind_coverage_evidence(
        unbound_core(evidence.core_results),
        c.OptionalDiagnostics(0.38, 16.0, 0.80, 9),
        evidence.equal_group_pair_contrasts,
    )
    changed = list(results)
    changed[15] = dataclasses.replace(
        changed[15], coverage_binding=changed_evidence.binding
    )
    for poisoned in (missing, changed):
        checks = (
            lambda poisoned=poisoned: c.validate_schedule(
                schedule, result_history=poisoned, coverage_evidence=evidence
            ),
            lambda poisoned=poisoned: c.validate_result_history(
                poisoned, coverage_evidence=evidence
            ),
            lambda poisoned=poisoned: c.stop_decision(poisoned, 0, evidence),
            lambda poisoned=poisoned: c.patience_streak(poisoned, evidence),
            lambda poisoned=poisoned: c.incumbent_challenger_decision(
                poisoned[:28], 1, coverage_evidence=evidence
            ),
            lambda poisoned=poisoned: c.successor_configurations(
                poisoned[:16], coverage_evidence=evidence
            ),
            lambda poisoned=poisoned: c.continuation_block(
                41, poisoned, coverage_evidence=evidence
            ),
            lambda poisoned=poisoned: c.minimum_schedule(
                one,
                two,
                coverage_results=poisoned[:16],
                cycle_one_results=poisoned[20:27],
                coverage_evidence=evidence,
            ),
        )
        for check in checks:
            try:
                check()
            except c.ControllerError:
                pass
            else:
                raise AssertionError("trial-16 binding substitution was accepted")

    signed_core = list(core_results())
    signed_core[0] = dataclasses.replace(signed_core[0], gain_bits_per_cell=0.0)
    signed_evidence = c.bind_coverage_evidence(
        signed_core,
        c.OptionalDiagnostics(0.39, 16.0, 0.80, 9),
        (0.0,) * c.PAIR_COUNT,
    )
    signed_history = list(signed_evidence.core_results)
    signed_history[0] = dataclasses.replace(
        signed_history[0], gain_bits_per_cell=-0.0
    )
    try:
        c.validate_available_result_prefix(signed_history, signed_evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("bit-distinct signed-zero core score was accepted")


def test_schedule_without_results_checks_embedded_binding_or_fails_closed() -> None:
    for branch_name in ("none", "low_rank"):
        schedule, _, _, _, evidence = branch(optional_branch=branch_name)
        c.validate_schedule(schedule, coverage_evidence=evidence)
        replacement = c.bind_coverage_evidence(
            unbound_core(evidence.core_results),
            c.OptionalDiagnostics(0.38, 16.0, 0.80, 9),
            evidence.equal_group_pair_contrasts,
        )
        try:
            c.validate_schedule(schedule, coverage_evidence=replacement)
        except c.ControllerError:
            pass
        else:
            raise AssertionError("schedule ignored its embedded evidence binding")

    hybrid_schedule, _, _, _, hybrid_evidence = branch(optional=True)
    try:
        c.validate_schedule(hybrid_schedule, coverage_evidence=hybrid_evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("score-selected hybrid schedule validated without scores")


def test_triggered_coverage_shifts_minimum_and_patience() -> None:
    schedule, results, _, _, evidence = branch(optional=True)
    assert len(schedule) == 52
    assert all(trial.kind == "anchor" for trial in schedule[16:28])
    assert schedule[28].kind == "rademacher_noise"
    assert c.stop_decision(results, 0, evidence) == "continue"
    parent_results = list(results)
    tried = list(c._real_tried_configurations(parent_results))
    for start in range(53, 78, 5):
        block = c.continuation_block(
            start,
            parent_results,
            tried,
            coverage_length=28,
            coverage_evidence=evidence,
        )
        for trial in block:
            parent_results.append(
                c.TrialResult(
                    trial,
                    0.01 if trial.kind == "successor" else 0.0,
                    4,
                )
            )
            if trial.kind == "successor":
                tried.append(trial.configuration)
    tail = c.continuation_block(
        78,
        parent_results,
        tried,
        coverage_length=28,
        coverage_evidence=evidence,
    )
    assert [trial.valid_index for trial in tail] == [78, 79, 80]


def test_two_incumbent_challenger_decisions() -> None:
    _, results, _, _, evidence = branch()
    first = c.incumbent_challenger_decision(
        results[:28], 1, coverage_evidence=evidence
    )
    second = c.incumbent_challenger_decision(
        results, 2, coverage_evidence=evidence
    )
    assert first.winner == first.challenger
    assert second.incumbent.gain_bits_per_cell == 0.02


def test_strict_trial_and_null_validation() -> None:
    for bad_index in (0, 100):
        try:
            c.new_null_replay(bad_index)
        except c.ControllerError:
            pass
        else:
            raise AssertionError("invalid null index accepted")
    try:
        c.Configuration("lasso", 0.2)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("off-grid rho accepted")
    try:
        c.Trial(1, "reference", "unknown", "coverage", False)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("unknown trial kind accepted")
    try:
        c.NullReplayState(1, c.NULL_SEEDS[0], M1_refit_required=False)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("null replay accepted a disabled required branch")


def test_adversarial_schedule_and_decision_rejections() -> None:
    trials, results, _, _, evidence = branch()
    swapped = list(trials)
    left = swapped[1].configuration
    right = swapped[2].configuration
    swapped[1] = dataclasses.replace(swapped[1], configuration=right)
    swapped[2] = dataclasses.replace(swapped[2], configuration=left)
    try:
        c.validate_schedule(swapped, coverage_evidence=evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("swapped core anchor order accepted")

    poisoned_phase = list(trials)
    poisoned_phase[1] = dataclasses.replace(
        poisoned_phase[1], phase="extension_coverage"
    )
    try:
        c.validate_schedule(poisoned_phase, coverage_evidence=evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("core anchor with an extension phase accepted")

    optional_trials, optional_results, _, _, optional_evidence_state = branch(
        optional=True
    )
    poisoned_optional_phase = list(optional_trials)
    poisoned_optional_phase[16] = dataclasses.replace(
        poisoned_optional_phase[16], phase="coverage"
    )
    try:
        c.validate_schedule(
            poisoned_optional_phase,
            result_history=optional_results,
            coverage_evidence=optional_evidence_state,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("optional anchor with a core-coverage phase accepted")

    poisoned_cycle = list(trials)
    poisoned_cycle[20] = dataclasses.replace(poisoned_cycle[20], cycle=2)
    try:
        c.validate_schedule(poisoned_cycle, coverage_evidence=evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("cycle-one successor relabeled as cycle two accepted")

    try:
        dataclasses.replace(trials[20], phase="patience")
    except c.ControllerError:
        pass
    else:
        raise AssertionError("postcoverage successor with patience phase constructed")

    anchors = c.anchor_configurations()
    bad_one_results = tuple(
        c.TrialResult(
            c.Trial(
                21 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                configuration,
                cycle=1,
                coverage_binding=evidence.binding,
            ),
            0.01,
            4,
        )
        for index, configuration in enumerate(anchors[:7])
    )
    try:
        c.minimum_schedule(
            anchors[:7],
            anchors[7:14],
            coverage_results=results[:16],
            cycle_one_results=bad_one_results,
            coverage_evidence=evidence,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("already-tried anchors accepted as successors")

    continuation_trials, continuation_results, continuation_evidence = extended_results(45)
    malformed = list(continuation_trials)
    malformed[42] = dataclasses.replace(
        malformed[42], configuration=c.anchor_configurations()[0]
    )
    try:
        c.validate_schedule(
            malformed, coverage_evidence=continuation_evidence
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("duplicate continuation configuration accepted")

    try:
        c.incumbent_challenger_decision(
            results[:27], 1, coverage_evidence=evidence
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("early incomplete incumbent decision accepted")

    inconsistent = list(results)
    target = [
        index
        for index, result in enumerate(inconsistent)
        if result.trial.kind == "successor" and result.trial.cycle == 1
    ][-1]
    inconsistent[target] = dataclasses.replace(
        inconsistent[target], gain_bits_per_cell=1.0
    )
    try:
        c.validate_result_history(inconsistent, coverage_evidence=evidence)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("score-inconsistent successor history accepted")


def test_optional_rank_is_bound_to_actual_coverage_scores() -> None:
    _, results, one, two, evidence = branch(optional=True)
    coverage_results = list(results[:28])
    for index in range(16, 24):
        configuration = coverage_results[index].trial.configuration
        assert configuration is not None and configuration.family == "low_rank"
        coverage_results[index] = dataclasses.replace(
            coverage_results[index],
            gain_bits_per_cell=1.0 if configuration.rank == 8 else -1.0,
        )
    cycle_one_results = tuple(results[32:39])
    try:
        c.minimum_schedule(
            one,
            two,
            coverage_results=coverage_results,
            cycle_one_results=cycle_one_results,
            coverage_evidence=evidence,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("hybrid rank from a separate score stream was accepted")


def test_mismatched_hybrid_rank_rejected_by_every_history_decision() -> None:
    schedule, results, evidence = mismatched_hybrid_history()
    checks = (
        lambda: c.validate_schedule(schedule, coverage_evidence=evidence),
        lambda: c.validate_schedule(
            schedule, result_history=results, coverage_evidence=evidence
        ),
        lambda: c.validate_result_history(results, coverage_evidence=evidence),
        lambda: c.stop_decision(results, 0, evidence),
        lambda: c.incumbent_challenger_decision(
            results[:40], 1, coverage_evidence=evidence
        ),
        lambda: c.incumbent_challenger_decision(
            results, 2, coverage_evidence=evidence
        ),
        lambda: c.patience_streak(results, evidence),
        lambda: c.continuation_block(
            53,
            results,
            c._real_tried_configurations(results),
            coverage_length=28,
            coverage_evidence=evidence,
        ),
        lambda: c.successor_configurations(
            results[:28],
            c._real_tried_configurations(results[:28]),
            count=7,
            coverage_evidence=evidence,
        ),
    )
    for check in checks:
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError("mismatched hybrid rank escaped a history decision")


def test_triggered_extensions_cannot_be_skipped_by_any_public_path() -> None:
    none_schedule, none_results, none_one, none_two, _ = branch()
    low_evidence = low_rank_evidence(none_results[:16])
    low_bound_results = [*low_evidence.core_results, *none_results[16:]]
    assert c.validate_available_result_prefix(
        low_bound_results[:16], low_evidence
    ) is None
    assert c.patience_streak(low_bound_results[:16], low_evidence) == 0
    assert (
        c.stop_decision(low_bound_results[:16], 0, low_evidence)
        == "continue_minimum_not_met"
    )
    low_skip_checks = (
        lambda: c.validate_schedule(
            none_schedule,
            result_history=low_bound_results,
            coverage_evidence=low_evidence,
        ),
        lambda: c.validate_result_history(
            low_bound_results, coverage_evidence=low_evidence
        ),
        lambda: c.stop_decision(low_bound_results, 0, low_evidence),
        lambda: c.patience_streak(low_bound_results, low_evidence),
        lambda: c.incumbent_challenger_decision(
            low_bound_results[:28], 1, coverage_evidence=low_evidence
        ),
        lambda: c.continuation_block(
            41,
            low_bound_results,
            c._real_tried_configurations(low_bound_results),
            coverage_evidence=low_evidence,
        ),
        lambda: c.successor_configurations(
            low_bound_results[:16],
            c._real_tried_configurations(low_bound_results[:16]),
            count=7,
            coverage_evidence=low_evidence,
        ),
        lambda: c.minimum_schedule(
            none_one,
            none_two,
            coverage_results=low_bound_results[:16],
            cycle_one_results=low_bound_results[20:27],
            coverage_evidence=low_evidence,
        ),
    )

    low_schedule, low_results, low_one, low_two, _ = branch(
        optional_branch="low_rank"
    )
    hybrid_evidence, _ = optional_evidence(low_results[:16])
    hybrid_staged = c.coverage_trials(coverage_evidence=hybrid_evidence)
    hybrid_bound_low = [*hybrid_evidence.core_results]
    hybrid_bound_low.extend(
        c.TrialResult(trial, result.gain_bits_per_cell, result.active_pair_degrees)
        for trial, result in zip(hybrid_staged[16:24], low_results[16:24])
    )
    hybrid_bound_results = [*hybrid_bound_low, *low_results[24:]]
    assert c.validate_available_result_prefix(
        hybrid_bound_results[:24], hybrid_evidence
    ) is None
    assert c.patience_streak(hybrid_bound_results[:24], hybrid_evidence) == 0
    assert (
        c.stop_decision(hybrid_bound_results[:24], 0, hybrid_evidence)
        == "continue_minimum_not_met"
    )
    hybrid_skip_checks = (
        lambda: c.validate_schedule(
            low_schedule,
            result_history=hybrid_bound_results,
            coverage_evidence=hybrid_evidence,
        ),
        lambda: c.validate_result_history(
            hybrid_bound_results, coverage_evidence=hybrid_evidence
        ),
        lambda: c.stop_decision(hybrid_bound_results, 0, hybrid_evidence),
        lambda: c.patience_streak(hybrid_bound_results, hybrid_evidence),
        lambda: c.incumbent_challenger_decision(
            hybrid_bound_results[:36], 1, coverage_evidence=hybrid_evidence
        ),
        lambda: c.continuation_block(
            49,
            hybrid_bound_results,
            c._real_tried_configurations(hybrid_bound_results),
            coverage_length=24,
            coverage_evidence=hybrid_evidence,
        ),
        lambda: c.successor_configurations(
            hybrid_bound_results[:24],
            c._real_tried_configurations(hybrid_bound_results[:24]),
            count=7,
            coverage_evidence=hybrid_evidence,
        ),
        lambda: c.minimum_schedule(
            low_one,
            low_two,
            coverage_results=hybrid_bound_results[:24],
            cycle_one_results=hybrid_bound_results[28:35],
            coverage_evidence=hybrid_evidence,
        ),
    )
    for check in (*low_skip_checks, *hybrid_skip_checks):
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError("mandatory optional coverage was skipped")


def test_every_postcore_public_path_requires_coverage_evidence() -> None:
    schedule, results, one, two, _ = branch()
    checks = (
        lambda: c.validate_available_result_prefix(results[:16]),
        lambda: c.validate_schedule(schedule),
        lambda: c.validate_result_history(results),
        lambda: c.stop_decision(results, 0),
        lambda: c.patience_streak(results),
        lambda: c.incumbent_challenger_decision(results[:28], 1),
        lambda: c.continuation_block(
            41, results, c._real_tried_configurations(results)
        ),
        lambda: c.successor_configurations(
            results[:16],
            c._real_tried_configurations(results[:16]),
            count=7,
        ),
        lambda: c.minimum_schedule(
            one,
            two,
            coverage_results=results[:16],
            cycle_one_results=results[20:27],
        ),
    )
    for check in checks:
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError("post-core path accepted omitted coverage evidence")
    try:
        c.CoverageEvidence(
            tuple(results[:16]),
            c.OptionalDiagnostics(0.39, 16.0, 0.80, 9),
            [0.0] * c.PAIR_COUNT,
            results[15].coverage_binding,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("mutable coverage contrasts were accepted")


def test_fabricated_hybrid_edges_rejected_by_every_public_path() -> None:
    schedule, results, one, two, evidence = fabricated_hybrid_edge_history()
    checks = (
        lambda: c.validate_available_result_prefix(results, evidence),
        lambda: c.validate_schedule(
            schedule, result_history=results, coverage_evidence=evidence
        ),
        lambda: c.validate_result_history(results, coverage_evidence=evidence),
        lambda: c.stop_decision(results, 0, evidence),
        lambda: c.patience_streak(results, evidence),
        lambda: c.incumbent_challenger_decision(
            results[:40], 1, coverage_evidence=evidence
        ),
        lambda: c.incumbent_challenger_decision(
            results, 2, coverage_evidence=evidence
        ),
        lambda: c.continuation_block(
            53,
            results,
            c._real_tried_configurations(results),
            coverage_length=28,
            coverage_evidence=evidence,
        ),
        lambda: c.successor_configurations(
            results[:28],
            c._real_tried_configurations(results[:28]),
            count=7,
            coverage_evidence=evidence,
        ),
        lambda: c.minimum_schedule(
            one,
            two,
            coverage_results=results[:28],
            cycle_one_results=results[32:39],
            coverage_evidence=evidence,
        ),
    )
    for check in checks:
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError("fabricated hybrid sparse edges escaped evidence binding")


def test_cycle_relabel_rejected_by_every_history_dependent_path() -> None:
    schedule, results, one, two, evidence = branch()
    target = max(
        (
            (index, result)
            for index, result in enumerate(results)
            if result.trial.kind == "successor" and result.trial.cycle == 1
        ),
        key=lambda item: item[1].gain_bits_per_cell,
    )[0]
    assert results[target].trial.valid_index == 21
    poisoned = list(results)
    poisoned[target] = dataclasses.replace(
        poisoned[target],
        trial=dataclasses.replace(poisoned[target].trial, cycle=2),
    )
    checks = (
        lambda: c.validate_schedule(
            tuple(result.trial for result in poisoned),
            coverage_evidence=evidence,
        ),
        lambda: c.validate_schedule(
            tuple(result.trial for result in poisoned),
            result_history=poisoned,
            coverage_evidence=evidence,
        ),
        lambda: c.validate_result_history(poisoned, coverage_evidence=evidence),
        lambda: c.stop_decision(poisoned, 0, evidence),
        lambda: c.stop_decision(poisoned[:28], 0, evidence),
        lambda: c.incumbent_challenger_decision(
            poisoned[:28], 1, coverage_evidence=evidence
        ),
        lambda: c.incumbent_challenger_decision(
            poisoned, 2, coverage_evidence=evidence
        ),
        lambda: c.patience_streak(poisoned, evidence),
        lambda: c.patience_streak(poisoned[:28], evidence),
        lambda: c.continuation_block(
            41,
            poisoned,
            c._real_tried_configurations(poisoned),
            coverage_evidence=evidence,
        ),
        lambda: c.successor_configurations(
            poisoned,
            c._real_tried_configurations(poisoned),
            count=7,
            coverage_evidence=evidence,
        ),
        lambda: c.successor_configurations(
            poisoned[:28],
            c._real_tried_configurations(poisoned[:28]),
            count=7,
            coverage_evidence=evidence,
        ),
        lambda: c.minimum_schedule(
            one,
            two,
            coverage_results=poisoned[:16],
            cycle_one_results=poisoned[20:27],
            coverage_evidence=evidence,
        ),
    )
    for check in checks:
        try:
            check()
        except c.ControllerError:
            pass
        else:
            raise AssertionError("cycle relabel escaped a history-dependent path")


def test_continuation_requires_exact_next_valid_index() -> None:
    _, results, _, _, evidence = branch()
    tried = c._real_tried_configurations(results)
    for skipped_start in (46, 51, 76):
        try:
            c.continuation_block(
                skipped_start,
                results,
                tried,
                coverage_evidence=evidence,
            )
        except c.ControllerError:
            pass
        else:
            raise AssertionError("continuation builder accepted a skipped block")
    assert c.continuation_block(
        41, results, tried, coverage_evidence=evidence
    )[0].valid_index == 41

    _, results_through_45, evidence_through_45 = extended_results(45)
    tried_through_45 = c._real_tried_configurations(results_through_45)
    assert (
        c.continuation_block(
            46,
            results_through_45,
            tried_through_45,
            coverage_evidence=evidence_through_45,
        )[0].valid_index
        == 46
    )


def test_public_builders_bind_tried_inventory_and_decision_time() -> None:
    _, results, _, _, evidence = branch()
    coverage_results = results[:16]
    try:
        c.successor_configurations(
            coverage_results,
            (),
            count=7,
            coverage_evidence=evidence,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("successor builder trusted an empty tried inventory")
    assert len(
        c.successor_configurations(
            coverage_results, count=7, coverage_evidence=evidence
        )
    ) == 7

    for tail_length in (1, 4, 5, 7, 11, 13, 16, 22, 24):
        parent = results[: 16 + tail_length]
        try:
            c.successor_configurations(
                parent,
                c._real_tried_configurations(parent),
                count=7,
                coverage_evidence=evidence,
            )
        except c.ControllerError:
            pass
        else:
            raise AssertionError("successor builder accepted an off-time history")

    try:
        c.continuation_block(
            41, results, (), coverage_evidence=evidence
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("continuation builder trusted an empty tried inventory")
    assert c.continuation_block(
        41, results, coverage_evidence=evidence
    )[0].valid_index == 41


def test_malformed_early_coverage_never_reaches_early_return() -> None:
    _, results, _, _, evidence = branch()
    poisoned = list(results)
    configuration = poisoned[1].trial.configuration
    assert configuration is not None
    poisoned[1] = dataclasses.replace(
        poisoned[1],
        trial=c.Trial(
            2,
            "association",
            "successor",
            "postcoverage",
            False,
            configuration,
            cycle=1,
        ),
    )
    for prefix_length in (2, 15, 28, 40):
        prefix = poisoned[:prefix_length]
        checks = (
            lambda prefix=prefix: c.validate_available_result_prefix(
                prefix, evidence if len(prefix) >= 16 else None
            ),
            lambda prefix=prefix: c.patience_streak(
                prefix, evidence if len(prefix) >= 16 else None
            ),
            lambda prefix=prefix: c.stop_decision(
                prefix, 0, evidence if len(prefix) >= 16 else None
            ),
            lambda prefix=prefix: c.stop_decision(
                prefix, 13, evidence if len(prefix) >= 16 else None
            ),
        )
        for check in checks:
            try:
                check()
            except c.ControllerError:
                pass
            else:
                raise AssertionError("malformed early coverage reached an early return")


def test_legitimate_partial_histories_all_optional_branches() -> None:
    expected_coverage = {"none": 16, "low_rank": 24, "hybrid": 28}
    for branch_name, coverage_length in expected_coverage.items():
        _, results, _, _, evidence = branch(optional_branch=branch_name)
        for prefix_length in range(1, coverage_length + 1):
            bound_evidence = evidence if prefix_length >= 16 else None
            c.validate_available_result_prefix(
                results[:prefix_length], bound_evidence
            )
            assert c.patience_streak(
                results[:prefix_length], bound_evidence
            ) == 0
            assert (
                c.stop_decision(
                    results[:prefix_length], 0, bound_evidence
                )
                == "continue_minimum_not_met"
            )
        for tail_length in (0, 1, 4, 5, 11, 12, 13, 15, 16, 22, 24):
            prefix = results[: coverage_length + tail_length]
            assert c._validate_partial_result_history(
                prefix, coverage_evidence=evidence
            ) == coverage_length

        coverage_results = results[:coverage_length]
        cycle_one = c.successor_configurations(
            coverage_results,
            c._real_tried_configurations(coverage_results),
            count=7,
            coverage_evidence=evidence,
        )
        compact_cycle_one = (
            *coverage_results,
            *results[coverage_length + 4 : coverage_length + 11],
        )
        cycle_two = c.successor_configurations(
            compact_cycle_one,
            c._real_tried_configurations(compact_cycle_one),
            count=7,
            coverage_evidence=evidence,
        )
        assert cycle_one == tuple(
            result.trial.configuration
            for result in results[coverage_length + 4 : coverage_length + 11]
        )
        assert cycle_two == tuple(
            result.trial.configuration
            for result in results[coverage_length + 15 : coverage_length + 22]
        )
        c.incumbent_challenger_decision(
            results[: coverage_length + 12],
            1,
            coverage_evidence=evidence,
        )
        c.incumbent_challenger_decision(
            results, 2, coverage_evidence=evidence
        )
        c.validate_result_history(results, coverage_evidence=evidence)
        assert c.patience_streak(results, evidence) == 0
        assert c.stop_decision(results, 0, evidence) == "continue"
        block = c.continuation_block(
            coverage_length + 25,
            results,
            c._real_tried_configurations(results),
            coverage_length=coverage_length,
            coverage_evidence=evidence,
        )
        assert block[0].valid_index == coverage_length + 25


def test_continuation_preserves_ceiling_and_falsifier_fraction() -> None:
    trials, results, evidence = extended_results(80)
    c.validate_schedule(trials, coverage_evidence=evidence)
    c.validate_result_history(results, coverage_evidence=evidence)
    assert sum(value.role == "reference" for value in trials) == 8
    assert sum(value.role == "association" for value in trials) == 72
    assert sum(value.falsifier for value in trials[16:]) >= math.ceil(0.4 * 64)
    real_keys = [
        value.configuration.key
        for value in trials
        if value.kind in {"anchor", "successor"} and value.configuration is not None
    ]
    assert len(real_keys) == len(set(real_keys))
    exhaustion = [
        trial.seed
        for trial in trials
        if trial.kind == "permutation"
        and trial.seed is not None
        and trial.seed >= 2026092837
    ]
    assert exhaustion == list(range(2026092837, 2026092837 + len(exhaustion)))
    assert c.stop_decision(results, 0, evidence) == "incomplete_budget_stop"


def test_patience_earliest_stop_and_reset() -> None:
    _, baseline, evidence = extended_results(54)
    assert c.patience_streak(baseline, evidence) == 14
    assert c.stop_decision(baseline, 0, evidence) == "ordinary_patience_stop"
    improved = list(baseline)
    target = next(
        index
        for index, result in enumerate(improved[40:], start=40)
        if result.trial.kind == "successor"
    )
    improved[target] = c.TrialResult(improved[target].trial, 0.023, 4)
    assert c.patience_streak(improved, evidence) < 14


def test_invalid_retry_ceiling() -> None:
    assert c.stop_decision([], 0) == "continue_minimum_not_met"
    assert c.stop_decision([], 11) == "continue_minimum_not_met"
    assert c.stop_decision([], 12) == "technical_stop_invalid_retry_ceiling"
    assert c.stop_decision([], 13) == "technical_stop_invalid_retry_ceiling"


def test_null_seed_inventory() -> None:
    seeds = c.null_replay_seeds()
    assert len(seeds) == 99
    assert len(set(seeds)) == 99
    assert seeds[0] == 2026093001 and seeds[-1] == 2026093099
    replay = c.new_null_replay(1)
    assert replay.trial_history == ()
    assert replay.M0_refit_required and replay.M1_refit_required
    assert replay.optional_trigger_replay_required
    first = c.seedsequence_substream(
        1, "M1_refit", 1, 0, "model_fit_or_initialization"
    )
    repeat = c.seedsequence_substream(
        1, "M1_refit", 1, 0, "model_fit_or_initialization"
    )
    other = c.seedsequence_substream(
        1, "M2_fit", 1, 1, "model_fit_or_initialization"
    )
    assert first == repeat and first != other and len(first) == 4
    categorical_folds = {
        c.seedsequence_substream(
            1, "M1_refit", fold, 0, "fitted_M1_categorical_sampling"
        )
        for fold in (0, 1, 79)
    }
    assert len(categorical_folds) == 3


def test_integrated_empty_history_replay_all_optional_branches() -> None:
    expected_coverage = {"none": 16, "low_rank": 24, "hybrid": 28}
    for replicate, branch_name in enumerate(expected_coverage, start=1):
        result = c.run_target_free_replay_conformance(replicate, branch_name)
        assert result["started_empty"] is True
        assert result["coverage_length"] == expected_coverage[branch_name]
        assert result["disposition"] == "ordinary_patience_stop"
        assert result["all_required_branches_replayed"] is True
        assert result["fold_ordinals_exercised"] == (0, 1, 79)
        assert result["purpose_routes_exercised"] == c.REPLAY_PURPOSE_ROUTES
        assert ("M1_refit", "fitted_M1_categorical_sampling") in result[
            "purpose_routes_exercised"
        ]
        assert result["substreams_derived"] == 3 * (result["valid_trials"] + 5)
        assert result["substream_determinism_verified"] is True
        assert result == c.run_target_free_replay_conformance(
            replicate, branch_name
        )


def test_candidate_selector_equal_groups_and_lexicographic_tie() -> None:
    rows = []
    group_ordinals = {group: 0 for group in c.GROUP_SIZES}
    for cell_key, group in c.FROZEN_CELL_INVENTORY.entries:
        index = group_ordinals[group]
        group_ordinals[group] += 1
        values = [0.0] * 120
        k = 1 if index == 0 else 2
        if k >= 2:
            values[2] = -0.3
            values[4] = 0.3
        rows.append(c.make_pair_contrast_row(cell_key, group, k, tuple(values)))
    assert group_ordinals == c.GROUP_SIZES
    selected = c.select_pair_contrast(rows)
    assert selected["pair"] == c.PAIR_NAMES[2]  # first frozen index on abs tie
    assert selected["pair_index"] == 2
    assert selected["direction"] == "negative"
    assert selected["biological_promotion"] is False

    mutable_values = [0] * c.PAIR_COUNT
    canonical = c.make_pair_contrast_row(
        "canonical_copy", "212074", 2, mutable_values
    )
    assert type(canonical.values) is tuple
    assert all(type(value) is float for value in canonical.values)
    mutable_values[0] = 1.0
    assert canonical.values == (0.0,) * c.PAIR_COUNT

    def assert_selection_rejected(candidate_rows, label):
        try:
            c.select_pair_contrast(candidate_rows)
        except c.ControllerError:
            pass
        else:
            raise AssertionError(label)

    try:
        c.PairContrastRow(
            rows[0].cell_key,
            rows[0].group,
            rows[0].k,
            rows[0].values,
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("direct PairContrastRow construction was accepted")

    replaced_row = dataclasses.replace(rows[0])
    poisoned = list(rows)
    poisoned[0] = replaced_row
    assert_selection_rejected(
        poisoned,
        "dataclasses.replace forged a controller-issued pair-contrast row",
    )

    double_mutated = c.make_pair_contrast_row(
        rows[0].cell_key,
        rows[0].group,
        1,
        (0.0,) * c.PAIR_COUNT,
    )
    object.__setattr__(double_mutated, "k", 2)
    object.__setattr__(
        double_mutated,
        "values",
        (0.5,) + (0.0,) * (c.PAIR_COUNT - 1),
    )
    poisoned = list(rows)
    poisoned[0] = double_mutated
    assert_selection_rejected(
        poisoned,
        "structurally valid K/value double mutation escaped issuance integrity",
    )

    mutable_row = c.make_pair_contrast_row(
        rows[0].cell_key,
        rows[0].group,
        rows[0].k,
        rows[0].values,
    )
    object.__setattr__(mutable_row, "values", [0.0] * c.PAIR_COUNT)
    poisoned = list(rows)
    poisoned[0] = mutable_row
    assert_selection_rejected(
        poisoned,
        "post-init mutable pair-contrast vector was accepted",
    )

    fractional_k = c.make_pair_contrast_row(
        rows[0].cell_key,
        rows[0].group,
        rows[0].k,
        rows[0].values,
    )
    object.__setattr__(fractional_k, "k", 1.5)
    poisoned = list(rows)
    poisoned[0] = fractional_k
    assert_selection_rejected(poisoned, "post-init fractional K was accepted")

    nonzero_boundary = c.make_pair_contrast_row(
        rows[0].cell_key,
        rows[0].group,
        1,
        (0.0,) * c.PAIR_COUNT,
    )
    object.__setattr__(
        nonzero_boundary,
        "values",
        (0.5,) + (0.0,) * (c.PAIR_COUNT - 1),
    )
    poisoned = list(rows)
    poisoned[0] = nonzero_boundary
    assert_selection_rejected(
        poisoned,
        "post-init nonzero K=1 boundary contrast was accepted",
    )

    original_inventory = c.FROZEN_CELL_INVENTORY.entries
    try:
        object.__setattr__(
            c.FROZEN_CELL_INVENTORY,
            "entries",
            (*original_inventory[:-1], ("mutated_inventory", original_inventory[-1][1])),
        )
        assert_selection_rejected(
            rows,
            "mutated in-memory frozen inventory was accepted",
        )
    finally:
        object.__setattr__(
            c.FROZEN_CELL_INVENTORY,
            "entries",
            original_inventory,
        )
    assert c.select_pair_contrast(rows)["pair_index"] == 2

    duplicate = list(rows)
    duplicate[-1] = c.make_pair_contrast_row(
        duplicate[0].cell_key,
        duplicate[-1].group,
        duplicate[-1].k,
        duplicate[-1].values,
    )
    try:
        c.select_pair_contrast(duplicate)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("duplicate cross-fitted cell accepted")

    replaced = list(rows)
    replaced[-1] = c.make_pair_contrast_row(
        "prospective_inventory_replacement",
        replaced[-1].group,
        replaced[-1].k,
        replaced[-1].values,
    )
    try:
        c.select_pair_contrast(replaced)
    except c.ControllerError:
        pass
    else:
        raise AssertionError("same-group replacement outside frozen inventory accepted")

    try:
        c.make_pair_contrast_row(
            "K16", "212074", 16, (0.5,) + (0.0,) * 119
        )
    except c.ControllerError:
        pass
    else:
        raise AssertionError("nonzero K=16 contrast accepted")

    for malformed_k in (False, True, 1.0, 1.5, "1"):
        try:
            c.make_pair_contrast_row(
                f"malformed_K_{malformed_k!r}",
                "212074",
                malformed_k,
                (0.0,) * 120,
            )
        except c.ControllerError:
            pass
        else:
            raise AssertionError(f"non-integer K accepted: {malformed_k!r}")


def run() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"v3 search controller tests: PASS ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    run()
