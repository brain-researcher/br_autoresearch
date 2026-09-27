"""Outcome-blind adaptive controller used inside every EP12 null replicate.

The frozen object in this module is a *program*, never an observed trajectory.
It contains the fixed coverage rows, proposal grammar, budgets, and post-36
slot templates.  A replicate begins with an empty record list and derives all
adaptive configurations, promotions, patience arithmetic, and its stopping
point only from that replicate's fitted-null outcomes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .adaptive_search import validate_development_configuration
from .adaptive_successors import _ranking_key, _scientific_signature, _successors
from .policy import EpisodePolicy, digest_object


REALIZED_SCHEMA = "ep12.adaptive_null_realized_schedule.v1"
OBJECTIVE_FIELDS = (
    "left_to_right",
    "right_to_left",
    "bilateral",
    "bilateral_floor",
    "direction_disagreement",
    "types_both_above_margin",
)


class NullControllerError(RuntimeError):
    """The frozen program or a replicate transition is invalid."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NullControllerError("controller metadata is not canonical JSON") from error


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value).decode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NullControllerError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NullControllerError(f"{label} must be a mapping")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return digest_object({key: item for key, item in value.items() if key != field})


def _validated_frozen_tail_plan(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the exact separately frozen tail plan, never a rehashed substitute."""

    try:
        from observed_tail_plan import build_plan, validate_plan
    except (ImportError, OSError) as error:
        raise NullControllerError("frozen observed-tail plan module is unavailable") from error
    expected = build_plan()
    candidate = expected if value is None else _copy(value)
    try:
        # ``None`` avoids a second filesystem policy read; policy identity is
        # checked independently below and by the controller program itself.
        validate_plan(candidate, policy_path=None)
    except (TypeError, ValueError) as error:
        raise NullControllerError(
            "post-36 template is not the exact frozen declarative plan"
        ) from error
    return _copy(expected)


def _project_tail_slots(
    raw_slots: Sequence[Mapping[str, Any]],
    *,
    countable: Sequence[str],
    policy: EpisodePolicy,
) -> list[dict[str, Any]]:
    """Project the observed-plan slots into the executable null slot schema."""

    _require(
        len(raw_slots) == policy.maximum_valid_trials - policy.minimum_valid_trials,
        "post-36 template does not cover every allowed ordinal",
    )
    slots: list[dict[str, Any]] = []
    for expected, raw in enumerate(
        raw_slots, start=policy.minimum_valid_trials + 1
    ):
        slot = _mapping(raw, f"post-36 slot {expected}")
        ordinal = int(slot.get("valid_trial_ordinal", -1))
        kind = str(slot.get("slot_kind", ""))
        falsifier_id = slot.get("falsifier_id")
        eligible = slot.get("promotion_eligible")
        _require(ordinal == expected, "post-36 template ordinals are not contiguous")
        _require(
            (
                kind == "mandatory_falsifier"
                and falsifier_id in countable
                and eligible is False
            )
            or (
                kind == "adaptive_challenger"
                and falsifier_id is None
                and eligible is True
            ),
            f"post-36 slot {ordinal} has an invalid classification",
        )
        core = {
            "valid_trial_ordinal": ordinal,
            "trial_id": f"null_tail_{ordinal:03d}",
            "slot_kind": kind,
            "falsifier_id": falsifier_id,
            "promotion_eligible": bool(eligible),
            "counts_toward_post_coverage_denominator": True,
            "counts_toward_post_coverage_falsifier_numerator": (
                kind == "mandatory_falsifier"
            ),
        }
        slots.append({**core, "slot_sha256": digest_object(core)})
    return slots


def _budget_contract(policy: EpisodePolicy) -> dict[str, Any]:
    raw = _mapping(policy.raw.get("budgets"), "policy budgets")
    return {
        "minimum_valid_trials": policy.minimum_valid_trials,
        "maximum_valid_trials": policy.maximum_valid_trials,
        "patience_valid_trials": policy.patience_valid_trials,
        "patience_active_after_trial": policy.patience_active_after_trial,
        "coverage_valid_trials": policy.coverage_minimum,
        "post_coverage_falsifier_fraction_min": policy.falsifier_fraction_minimum,
        "minimum_adaptive_successor_cycles": policy.minimum_adaptive_cycles,
        "minimum_incumbent_challenger_decisions": policy.minimum_incumbent_decisions,
        "cpu_cores_per_trial_max": policy.cpu_cores_per_trial_maximum,
        "memory_gib_per_trial_max": policy.memory_gib_per_trial_maximum,
        "wall_hours_per_trial_max": policy.wall_hours_per_trial_maximum,
        "wall_clock_hour_ceiling": float(raw["wall_clock_hour_ceiling"]),
        "gpu_count": int(raw["gpu_count"]),
        "null_replicates": policy.null_replicates,
    }


def _grammar_contract(policy: EpisodePolicy) -> dict[str, Any]:
    return {
        "model_triplet": list(policy.model_triplet),
        "nuisance": list(policy.nuisance_terms),
        "choices": {
            key: list(values) for key, values in policy.grammar_choices.items()
        },
        "proposal_operator": "one_policy_ordered_grammar_field_change",
        "ranking": [
            "bilateral_floor_descending",
            "bilateral_descending",
            "direction_disagreement_ascending",
            "types_both_above_margin_descending",
            "configuration_sha256_ascending",
        ],
    }


def freeze_null_controller_program(
    *,
    policy: EpisodePolicy,
    coverage_configurations: Sequence[Mapping[str, Any]],
    post36_tail_plan: Mapping[str, Any],
    countable_falsifier_ids: Sequence[str],
    non_trial_required_falsifier_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze an outcome-blind executable program, not observed decisions."""

    coverage = [_copy(value) for value in coverage_configurations]
    _require(
        len(coverage) == policy.coverage_minimum == 18,
        "null controller requires exactly 18 fixed coverage configurations",
    )
    for ordinal, configuration in enumerate(coverage, start=1):
        validate_development_configuration(policy, configuration)
        _require(
            configuration.get("stage") == "coverage",
            f"coverage configuration {ordinal} is not a coverage row",
        )
    _require(
        len({_scientific_signature(value) for value in coverage}) == len(coverage),
        "coverage prefix repeats a scientific configuration",
    )
    countable = [str(value) for value in countable_falsifier_ids]
    non_trial = [str(value) for value in non_trial_required_falsifier_ids]
    _require(
        len(set(countable)) == len(countable)
        and len(set(non_trial)) == len(non_trial)
        and not (set(countable) & set(non_trial))
        and set(countable) | set(non_trial) == set(policy.required_falsifiers),
        "controller falsifier classes do not partition SEARCH_POLICY",
    )
    _require(
        "fitted_T_U_null_full_revised_search_rerun" in non_trial,
        "nested full-search null must remain a non-trial gate",
    )
    plan = _validated_frozen_tail_plan(
        _mapping(post36_tail_plan, "post-36 template plan")
    )
    raw_slots = plan.get("all_slots")
    _require(isinstance(raw_slots, list), "post-36 template lacks all_slots")
    slots = _project_tail_slots(raw_slots, countable=countable, policy=policy)
    grammar = _grammar_contract(policy)
    budgets = _budget_contract(policy)
    core = {
        "policy_sha256": policy.policy_hash,
        "initial_ledger": "empty_for_each_replicate",
        "coverage_configurations": coverage,
        "coverage_configuration_sha256": [
            digest_object(value) for value in coverage
        ],
        "adaptive_prefix": [
            {
                "cycle": 1,
                "successor_count": 9,
                "parent_scope": "all_18_coverage_outcomes",
                "seen_scope": "all_18_coverage_configurations",
            },
            {
                "cycle": 2,
                "successor_count": 9,
                "parent_scope": "cycle_1_outcomes_only",
                "seen_scope": "all_27_prior_configurations",
            },
        ],
        "post36_slots": slots,
        "countable_falsifier_ids": countable,
        "non_trial_required_falsifier_ids": non_trial,
        "grammar": grammar,
        "grammar_sha256": digest_object(grammar),
        "budgets": budgets,
        "budgets_sha256": digest_object(budgets),
        "source_post36_plan_sha256": str(plan["plan_sha256"]),
        "source_post36_plan_object_sha256": digest_object(plan),
        "source_post36_slot_projection_sha256": digest_object(slots),
        "observed_realized_schedule_embedded": False,
        "observed_promotion_markers_embedded": False,
        "null_outcomes_drive_proposals_promotions_and_stop": True,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    program = {**core, "program_sha256": digest_object(core)}
    verify_null_controller_program(program, policy=policy)
    return program


def verify_null_controller_program(
    program: Mapping[str, Any], *, policy: EpisodePolicy
) -> dict[str, Any]:
    program = _mapping(program, "null controller program")
    required = {
        "policy_sha256", "initial_ledger",
        "coverage_configurations", "coverage_configuration_sha256",
        "adaptive_prefix", "post36_slots", "countable_falsifier_ids",
        "non_trial_required_falsifier_ids", "grammar", "grammar_sha256",
        "budgets", "budgets_sha256", "source_post36_plan_sha256",
        "source_post36_plan_object_sha256",
        "source_post36_slot_projection_sha256",
        "observed_realized_schedule_embedded",
        "observed_promotion_markers_embedded",
        "null_outcomes_drive_proposals_promotions_and_stop", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "program_sha256",
    }
    _require(required.issubset(program), "null controller program is incomplete")
    _require(program["program_sha256"] == _self_hash(program, "program_sha256"), "controller program self-hash mismatch")
    _require(program["policy_sha256"] == policy.policy_hash, "controller program policy mismatch")
    _require(program["initial_ledger"] == "empty_for_each_replicate", "null ledger is not empty by contract")
    _require(
        program["observed_realized_schedule_embedded"] is False
        and program["observed_promotion_markers_embedded"] is False
        and program["null_outcomes_drive_proposals_promotions_and_stop"] is True,
        "controller program embeds observed decisions",
    )
    _require(
        program["development_only"] is True
        and program["final_connectivity_accessed"] is False
        and program["final_connectivity_access_authorized"] is False,
        "controller program crosses the final-data firewall",
    )
    coverage = program["coverage_configurations"]
    hashes = program["coverage_configuration_sha256"]
    _require(
        isinstance(coverage, list)
        and len(coverage) == policy.coverage_minimum == 18
        and isinstance(hashes, list)
        and hashes == [digest_object(value) for value in coverage],
        "controller coverage prefix is incomplete or changed",
    )
    for configuration in coverage:
        validate_development_configuration(policy, configuration)
        _require(configuration.get("stage") == "coverage", "controller coverage stage changed")
    expected_adaptive = [
        {"cycle": 1, "successor_count": 9, "parent_scope": "all_18_coverage_outcomes", "seen_scope": "all_18_coverage_configurations"},
        {"cycle": 2, "successor_count": 9, "parent_scope": "cycle_1_outcomes_only", "seen_scope": "all_27_prior_configurations"},
    ]
    _require(program["adaptive_prefix"] == expected_adaptive, "9+9 adaptive prefix changed")
    countable = list(map(str, program["countable_falsifier_ids"]))
    non_trial = list(map(str, program["non_trial_required_falsifier_ids"]))
    _require(
        set(countable) | set(non_trial) == set(policy.required_falsifiers)
        and not (set(countable) & set(non_trial)),
        "controller falsifier partition changed",
    )
    slots = program["post36_slots"]
    _require(
        isinstance(slots, list)
        and len(slots) == policy.maximum_valid_trials - policy.minimum_valid_trials,
        "controller post-36 slots are incomplete",
    )
    for ordinal, slot in enumerate(slots, start=policy.minimum_valid_trials + 1):
        core = {key: value for key, value in slot.items() if key != "slot_sha256"}
        _require(
            int(slot.get("valid_trial_ordinal", -1)) == ordinal
            and slot.get("trial_id") == f"null_tail_{ordinal:03d}"
            and slot.get("slot_sha256") == digest_object(core),
            "controller post-36 slot identity changed",
        )
        kind = slot.get("slot_kind")
        _require(
            (kind == "mandatory_falsifier" and slot.get("falsifier_id") in countable and slot.get("promotion_eligible") is False and slot.get("counts_toward_post_coverage_falsifier_numerator") is True)
            or (kind == "adaptive_challenger" and slot.get("falsifier_id") is None and slot.get("promotion_eligible") is True and slot.get("counts_toward_post_coverage_falsifier_numerator") is False),
            f"controller slot classification changed at {ordinal}",
        )
    frozen_plan = _validated_frozen_tail_plan()
    frozen_slots = _project_tail_slots(
        frozen_plan["all_slots"], countable=countable, policy=policy
    )
    _require(
        program["source_post36_plan_sha256"] == frozen_plan["plan_sha256"]
        and program["source_post36_plan_object_sha256"]
        == digest_object(frozen_plan)
        and program["source_post36_slot_projection_sha256"]
        == digest_object(frozen_slots)
        and slots == frozen_slots,
        "controller is not bound to the exact frozen post-36 plan object",
    )
    grammar = _grammar_contract(policy)
    budgets = _budget_contract(policy)
    _require(program["grammar"] == grammar and program["grammar_sha256"] == digest_object(grammar), "controller grammar changed")
    _require(program["budgets"] == budgets and program["budgets_sha256"] == digest_object(budgets), "controller budgets changed")
    return _copy(program)


def normalize_objective(value: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(value, "trial objective")
    _require(set(value) == set(OBJECTIVE_FIELDS), "trial objective fields differ")
    result = {
        "left_to_right": float(value["left_to_right"]),
        "right_to_left": float(value["right_to_left"]),
        "bilateral": float(value["bilateral"]),
        "bilateral_floor": float(value["bilateral_floor"]),
        "direction_disagreement": float(value["direction_disagreement"]),
        "types_both_above_margin": int(value["types_both_above_margin"]),
    }
    _require(
        all(
            item == item and abs(item) != float("inf")
            for key, item in result.items()
            if key != "types_both_above_margin"
        )
        and result["types_both_above_margin"] >= 0,
        "trial objective is nonfinite or invalid",
    )
    return result


def _record_from_step(step: Mapping[str, Any]) -> dict[str, Any]:
    proposal = _mapping(step.get("proposal"), "step proposal")
    fit = _mapping(step.get("fit_result"), "step fit result")
    disposition = _mapping(step.get("disposition"), "step disposition")
    return {
        "trial_id": str(proposal["trial_id"]),
        "configuration": _copy(proposal["configuration"]),
        "configuration_sha256": str(proposal["configuration_sha256"]),
        "objective": normalize_objective(fit["objective"]),
        "promotion_eligible": proposal["promotion_eligible"] is True,
        "improved": disposition["improved"] is True,
        "valid_trial_ordinal": int(proposal["valid_trial_ordinal"]),
        "slot_kind": str(proposal["slot_kind"]),
        "falsifier_id": proposal.get("falsifier_id"),
    }


def records_from_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [_record_from_step(step) for step in steps]
    _require(
        [row["valid_trial_ordinal"] for row in result]
        == list(range(1, len(result) + 1)),
        "replicate steps are not a contiguous empty-ledger trajectory",
    )
    return result


def _post36_challenger(
    *,
    policy: EpisodePolicy,
    ordinal: int,
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return the exact observed-path successor, or ``None`` if exhausted."""

    eligible_records = [
        dict(value) for value in records if value["promotion_eligible"]
    ]
    _require(eligible_records, "post-36 proposal has no incumbent archive")
    try:
        configurations, lineages = _successors(
            policy=policy,
            cycle=ordinal,
            ranked_parents=sorted(eligible_records, key=_ranking_key),
            seen_configurations=[
                dict(value["configuration"]) for value in eligible_records
            ],
            count=1,
        )
    except ValueError as error:
        message = str(error)
        if not (
            message.startswith("only ")
            and message.endswith(" unseen successors were available")
        ):
            raise
        return None
    return dict(configurations[0]), dict(lineages[0])


def propose_next_trial(
    *,
    program: Mapping[str, Any],
    policy: EpisodePolicy,
    prior_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive exactly one next proposal from this null replicate's outcomes."""

    verify_null_controller_program(program, policy=policy)
    records = records_from_steps(prior_steps)
    ordinal = len(records) + 1
    _require(ordinal <= policy.maximum_valid_trials, "controller has no ordinal after maximum")
    lineage: dict[str, Any]
    if ordinal <= policy.coverage_minimum:
        configuration = dict(program["coverage_configurations"][ordinal - 1])
        lineage = {"rule": "fixed_coverage", "coverage_index": ordinal - 1}
        slot_kind = "coverage"
        falsifier_id = None
        eligible = True
        slot_hash = None
    elif ordinal <= 27:
        parents = [dict(value) for value in records[:18]]
        configurations, lineages = _successors(
            policy=policy,
            cycle=1,
            ranked_parents=sorted(parents, key=_ranking_key),
            seen_configurations=[dict(value["configuration"]) for value in parents],
            count=9,
        )
        index = ordinal - 19
        configuration = dict(configurations[index])
        lineage = {**dict(lineages[index]), "rule": "null_outcome_adaptive_cycle_1"}
        slot_kind = "adaptive_challenger"
        falsifier_id = None
        eligible = True
        slot_hash = None
    elif ordinal <= 36:
        prefix = [dict(value) for value in records[:27]]
        parents = [dict(value) for value in records[18:27]]
        configurations, lineages = _successors(
            policy=policy,
            cycle=2,
            ranked_parents=sorted(parents, key=_ranking_key),
            seen_configurations=[dict(value["configuration"]) for value in prefix],
            count=9,
        )
        index = ordinal - 28
        configuration = dict(configurations[index])
        lineage = {**dict(lineages[index]), "rule": "null_outcome_adaptive_cycle_2"}
        slot_kind = "adaptive_challenger"
        falsifier_id = None
        eligible = True
        slot_hash = None
    else:
        slot = program["post36_slots"][ordinal - 37]
        slot_kind = str(slot["slot_kind"])
        falsifier_id = slot["falsifier_id"]
        eligible = slot["promotion_eligible"] is True
        slot_hash = str(slot["slot_sha256"])
        eligible_records = [dict(value) for value in records if value["promotion_eligible"]]
        _require(eligible_records, "post-36 proposal has no incumbent archive")
        incumbent = min(eligible_records, key=_ranking_key)
        if slot_kind == "mandatory_falsifier":
            configuration = dict(incumbent["configuration"])
            lineage = {
                "rule": "current_null_incumbent_registered_falsifier",
                "parent_trial_id": incumbent["trial_id"],
                "parent_configuration_sha256": incumbent["configuration_sha256"],
                "registered_falsifier_id": falsifier_id,
            }
        else:
            successor = _post36_challenger(
                policy=policy, ordinal=ordinal, records=records
            )
            _require(
                successor is not None,
                "admissible configuration space is exhausted",
            )
            configuration, successor_lineage = successor
            lineage = {
                **successor_lineage,
                "rule": "null_outcome_post36_challenger",
            }
    trial_id = (
        str(configuration["trial_id"])
        if ordinal <= 36
        else f"null_tail_{ordinal:03d}"
    )
    configuration["trial_id"] = trial_id
    configuration["stage"] = "coverage" if ordinal <= 18 else "adaptive"
    validate_development_configuration(policy, configuration)
    core = {
        "valid_trial_ordinal": ordinal,
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": digest_object(configuration),
        "scientific_signature": _scientific_signature(configuration),
        "slot_kind": slot_kind,
        "falsifier_id": falsifier_id,
        "promotion_eligible": bool(eligible),
        "lineage": lineage,
        "slot_template_sha256": slot_hash,
        "proposal_context_sha256": digest_object(
            [step.get("step_sha256") for step in prior_steps]
        ),
        "derived_from_null_outcomes_only": True,
    }
    return {**core, "proposal_sha256": digest_object(core)}


def disposition_for_result(
    *,
    proposal: Mapping[str, Any],
    objective: Mapping[str, Any],
    prior_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    objective = normalize_objective(objective)
    records = records_from_steps(prior_steps)
    eligible_records = [dict(value) for value in records if value["promotion_eligible"]]
    candidate = {
        "trial_id": str(proposal["trial_id"]),
        "configuration": _copy(proposal["configuration"]),
        "configuration_sha256": str(proposal["configuration_sha256"]),
        "objective": objective,
    }
    if not eligible_records:
        improved = proposal.get("promotion_eligible") is True
        incumbent = candidate
    else:
        previous = min(eligible_records, key=_ranking_key)
        improved = bool(
            proposal.get("promotion_eligible") is True
            and _ranking_key(candidate) < _ranking_key(previous)
        )
        incumbent = candidate if improved else previous
    core = {
        "promotion_eligible": proposal.get("promotion_eligible") is True,
        "improved": improved,
        "disposition": "incumbent" if improved else "retired",
        "incumbent_trial_id": incumbent["trial_id"],
        "incumbent_configuration_sha256": incumbent["configuration_sha256"],
        "ranking_key_sha256": digest_object(list(_ranking_key(candidate))),
    }
    return {**core, "disposition_sha256": digest_object(core)}


def stop_status(
    *,
    program: Mapping[str, Any],
    policy: EpisodePolicy,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = records_from_steps(steps)
    ordinal = len(records)
    denominator = max(0, ordinal - policy.coverage_minimum)
    controls = [
        str(value["falsifier_id"])
        for value in records
        if value["slot_kind"] == "mandatory_falsifier"
    ]
    numerator = len(controls)
    completed = sorted(set(controls))
    # The frozen observed controller opens the patience epoch at trial 36.
    # Prefix promotions choose the incumbent but do not move that epoch;
    # only promotion-eligible tail improvements reset it.
    tail_improvements = [
        value["valid_trial_ordinal"]
        for value in records
        if value["improved"] and value["valid_trial_ordinal"] > 36
    ]
    last_improvement = max([36, *tail_improvements]) if ordinal >= 36 else ordinal
    fraction_gate = (
        denominator > 0
        and numerator * 5 >= denominator * 2
        and abs(policy.falsifier_fraction_minimum - 0.4) < 1e-12
    )
    ids_gate = set(completed) == set(program["countable_falsifier_ids"])
    trailing = ordinal - last_improvement if last_improvement else ordinal
    patience = bool(
        ordinal >= policy.minimum_valid_trials
        and ordinal >= policy.patience_active_after_trial
        and fraction_gate
        and ids_gate
        and trailing >= policy.patience_valid_trials
    )
    exhausted = False
    exhaustion_probe_sha256: str | None = None
    next_ordinal = ordinal + 1
    if (
        not patience
        and ordinal >= policy.minimum_valid_trials
        and next_ordinal <= policy.maximum_valid_trials
        and program["post36_slots"][next_ordinal - 37]["slot_kind"]
        == "adaptive_challenger"
    ):
        probe_core = {
            "next_valid_trial_ordinal": next_ordinal,
            "eligible_archive": [
                {
                    "trial_id": value["trial_id"],
                    "configuration_sha256": value["configuration_sha256"],
                    "objective": value["objective"],
                }
                for value in records
                if value["promotion_eligible"]
            ],
        }
        exhaustion_probe_sha256 = digest_object(probe_core)
        exhausted = _post36_challenger(
            policy=policy,
            ordinal=next_ordinal,
            records=records,
        ) is None
    # Preserve the frozen controller's terminal priority: the hard maximum is
    # evaluated before patience.  Cumulative wall-budget exhaustion is handled
    # by the execution layer because it is authenticated from persisted step
    # timings, and is likewise evaluated before accepting a patience stop.
    if ordinal >= policy.maximum_valid_trials:
        action, reason = "stop", "valid_trials_max_reached"
    elif patience:
        action, reason = "stop", "patience_exhausted"
    elif exhausted:
        action, reason = "stop", "admissible_configuration_space_exhausted"
    else:
        action, reason = "continue", None
    core = {
        "valid_trial_count": ordinal,
        "post_coverage_valid_trial_count": denominator,
        "counted_post_coverage_falsifier_count": numerator,
        "completed_countable_falsifier_ids": completed,
        "falsifier_fraction_gate": fraction_gate,
        "all_countable_falsifier_ids_gate": ids_gate,
        "last_promotion_eligible_improvement_ordinal": last_improvement,
        "trailing_valid_trials_without_eligible_improvement": trailing,
        "patience_actionable": patience,
        "admissible_configuration_space_exhausted": exhausted,
        "exhaustion_probe_sha256": exhaustion_probe_sha256,
        "action": action,
        "reason": reason,
    }
    return {**core, "stop_status_sha256": digest_object(core)}


def freeze_realized_null_schedule(
    *,
    program: Mapping[str, Any],
    policy: EpisodePolicy,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze only the trajectory realized by this fitted-null replicate."""

    status = stop_status(program=program, policy=policy, steps=steps)
    _require(status["action"] == "stop", "cannot freeze a running null trajectory")
    records = records_from_steps(steps)
    trials = []
    for record, step in zip(records, steps):
        proposal = step["proposal"]
        trials.append({
            "valid_trial_ordinal": record["valid_trial_ordinal"],
            "trial_id": record["trial_id"],
            "configuration": record["configuration"],
            "configuration_sha256": record["configuration_sha256"],
            "slot_kind": record["slot_kind"],
            "falsifier_id": record["falsifier_id"],
            "promotion_eligible": record["promotion_eligible"],
            "null_promotion_eligible_improvement": record["improved"],
            "proposal_sha256": proposal["proposal_sha256"],
            "fit_result_sha256": step["fit_result"]["result_sha256"],
            "step_sha256": step["step_sha256"],
        })
    core = {
        "schema_version": REALIZED_SCHEMA,
        "policy_sha256": policy.policy_hash,
        "program_sha256": program["program_sha256"],
        "trial_count": len(trials),
        "trials": trials,
        "stop": status,
        "initial_ledger_empty": True,
        "observed_schedule_replayed": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    return {**core, "schedule_sha256": digest_object(core)}


__all__ = [
    "NullControllerError",
    "OBJECTIVE_FIELDS",
    "REALIZED_SCHEMA",
    "disposition_for_result",
    "freeze_null_controller_program",
    "freeze_realized_null_schedule",
    "normalize_objective",
    "propose_next_trial",
    "records_from_steps",
    "stop_status",
    "verify_null_controller_program",
]
