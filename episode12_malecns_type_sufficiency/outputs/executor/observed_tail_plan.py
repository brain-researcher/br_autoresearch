"""Frozen, outcome-blind plan for EP12 valid trials 37 through 96.

This module is deliberately outside :mod:`ep12_executor`.  Existing Slurm
jobs hash or import files in that package, so the post-36 plan must not alter
their import graph or code identity.  The plan contains selectors and control
identities, never observed connectivity values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "ep12.observed_post36_tail_plan.v1"
POLICY_SHA256 = "d64c5c9c56b70f3312472e056b9394f519107eda268902ceb586771d435d8b8d"
EXPECTED_PLAN_SHA256 = "cf5251ae6ec1fe42a56a0f98fbbadab96a819405180c4b68adb557dea5555bba"

COVERAGE_VALID_TRIALS = 18
PRE_TAIL_VALID_TRIALS = 36
MINIMUM_VALID_TRIALS = 36
MAXIMUM_VALID_TRIALS = 96
PATIENCE_VALID_TRIALS = 16
PATIENCE_ACTIVE_AFTER_TRIAL = 36
FALSIFIER_FRACTION_NUMERATOR = 2
FALSIFIER_FRACTION_DENOMINATOR = 5

ALL_REQUIRED_FALSIFIERS = (
    "fitted_T_U_null_full_revised_search_rerun",
    "strength_and_margin_preserving_graph_randomization",
    "cross_side_component_alignment_permutation",
    "shuffled_partner_and_valid_side_controls",
    "binary_weighted_vocabulary_and_rank_sensitivities",
    "nuisance_and_partner_block_ablations",
    "capacity_matched_noise_controls",
    "leave_one_development_type_and_partner_family_influence",
    "few_type_and_high_strength_neuron_concentration_check",
    "status_endpoint_anatomy_and_missingness_checks",
    "forbidden_group_instance_post_result_novelty_audit",
)

# These nine controls can be realized as valid, complete development T/U/M
# comparisons and therefore can enter the post-coverage trial denominator and
# falsifier numerator.  The other two policy items are separate operations.
COUNTABLE_OBSERVED_FALSIFIERS = ALL_REQUIRED_FALSIFIERS[1:-1]

NON_TRIAL_REQUIRED_GATES = (
    {
        "falsifier_id": "fitted_T_U_null_full_revised_search_rerun",
        "controller_operation": "run_full_search_null",
        "counts_as_observed_valid_trial": False,
        "counts_toward_post_coverage_falsifier_fraction": False,
        "reason": (
            "SEARCH_POLICY defines this as exactly 99 complete controller reruns "
            "after null eligibility, not one observed T/U/M trial"
        ),
    },
    {
        "falsifier_id": "forbidden_group_instance_post_result_novelty_audit",
        "controller_operation": "post_result_novelty_audit",
        "counts_as_observed_valid_trial": False,
        "counts_toward_post_coverage_falsifier_fraction": False,
        "reason": (
            "provider group and instance are forbidden for search and may be used "
            "only in a post-result novelty audit"
        ),
    },
)

EPISODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
DEFAULT_PLAN_PATH = Path(__file__).resolve().with_name(
    "OBSERVED_POST36_TAIL_PLAN.json"
)


class TailPlanError(ValueError):
    """The frozen tail plan or a proposed realization is inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def required_falsifier_count(post_coverage_valid_count: int) -> int:
    """Return ceil(0.40 * count) using exact integer arithmetic."""

    if post_coverage_valid_count < 0:
        raise TailPlanError("post-coverage valid count cannot be negative")
    numerator = post_coverage_valid_count * FALSIFIER_FRACTION_NUMERATOR
    return (
        numerator + FALSIFIER_FRACTION_DENOMINATOR - 1
    ) // FALSIFIER_FRACTION_DENOMINATOR


def minimum_added_falsifiers(
    *,
    prior_post_coverage_valid: int,
    prior_counted_falsifiers: int,
) -> int:
    """Solve the fraction gate without floating-point rounding."""

    if prior_post_coverage_valid < 0 or prior_counted_falsifiers < 0:
        raise TailPlanError("prior counts cannot be negative")
    if prior_counted_falsifiers > prior_post_coverage_valid:
        raise TailPlanError("falsifiers cannot exceed post-coverage valid trials")
    added = 0
    while (
        (prior_counted_falsifiers + added) * FALSIFIER_FRACTION_DENOMINATOR
        < (prior_post_coverage_valid + added) * FALSIFIER_FRACTION_NUMERATOR
    ):
        added += 1
    return added


def _selector(slot_kind: str) -> dict[str, Any]:
    common = {
        "ledger_prefix": "all_scored_valid_trials_through_previous_valid_ordinal",
        "final_type_information_allowed": False,
        "final_connectivity_accessed": False,
        "tie_rule": "frozen_development_only_archive_order_then_configuration_sha256",
    }
    if slot_kind == "mandatory_falsifier":
        return {
            **common,
            "base_configuration_rule": (
                "best_promotion_eligible_complete_comparison_in_ledger_prefix"
            ),
            "transform_rule": "registered_falsifier_id_exactly_as_declared_in_slot",
            "target_data_role": "development_only",
        }
    if slot_kind == "adaptive_challenger":
        return {
            **common,
            "configuration_rule": (
                "next_unseen_one_operator_grammar_neighbor_from_ranked_"
                "promotion_eligible_archive"
            ),
            "proposal_context_hash_required": True,
            "prior_scored_outcome_reference_required": True,
            "target_data_role": "development_only",
        }
    raise TailPlanError(f"unknown slot kind: {slot_kind}")


def _realization_evidence(slot_kind: str) -> list[str]:
    evidence = [
        "valid_complete_T_U_M_triplet",
        "scored_record_committed",
        "zero_technical_failures",
        "configuration_and_slot_hashes_match",
        "development_roles_only",
        "final_connectivity_accessed_false",
    ]
    if slot_kind == "mandatory_falsifier":
        evidence.extend(
            [
                "registered_falsifier_id_matches",
                "falsifier_control_receipt_passed",
                "control_conservation_and_applicability_checks_passed",
            ]
        )
    else:
        evidence.extend(
            [
                "proposal_committed_before_execution",
                "complete_visible_ledger_prefix_hash_matches",
                "unseen_admissible_grammar_configuration",
            ]
        )
    return evidence


def _make_slot(
    *,
    valid_trial_ordinal: int,
    phase: str,
    slot_kind: str,
    falsifier_id: str | None,
    falsifier_cycle_index: int | None,
    cumulative_falsifiers: int,
    exercised_falsifiers: set[str],
) -> dict[str, Any]:
    post_coverage = valid_trial_ordinal - COVERAGE_VALID_TRIALS
    required = required_falsifier_count(post_coverage)
    fraction_gate = (
        cumulative_falsifiers * FALSIFIER_FRACTION_DENOMINATOR
        >= post_coverage * FALSIFIER_FRACTION_NUMERATOR
    )
    countable_gate = set(COUNTABLE_OBSERVED_FALSIFIERS).issubset(
        exercised_falsifiers
    )
    patience_window_available = valid_trial_ordinal - PRE_TAIL_VALID_TRIALS
    patience_may_be_checked = (
        valid_trial_ordinal >= PATIENCE_ACTIVE_AFTER_TRIAL
        and valid_trial_ordinal >= MINIMUM_VALID_TRIALS
        and fraction_gate
        and countable_gate
        and patience_window_available >= PATIENCE_VALID_TRIALS
    )
    promotion_eligible = slot_kind == "adaptive_challenger"
    slot: dict[str, Any] = {
        "valid_trial_ordinal": valid_trial_ordinal,
        "trial_id": f"observed_tail_{valid_trial_ordinal:03d}",
        "phase": phase,
        "slot_kind": slot_kind,
        "falsifier_id": falsifier_id,
        "falsifier_cycle_index": falsifier_cycle_index,
        "promotion_eligible": promotion_eligible,
        "promotion_eligibility_reason": (
            "primary-development unseen grammar challenger"
            if promotion_eligible
            else "control or transformed-data result cannot replace the primary incumbent"
        ),
        "counts_toward_post_coverage_denominator": True,
        "counts_toward_post_coverage_falsifier_numerator": (
            slot_kind == "mandatory_falsifier"
        ),
        "selector": _selector(slot_kind),
        "required_realization_evidence": _realization_evidence(slot_kind),
        "cumulative_if_every_prior_slot_is_valid": {
            "post_coverage_valid_trial_count": post_coverage,
            "counted_falsifier_trial_count": cumulative_falsifiers,
            "minimum_counted_falsifiers_for_fraction": required,
            "falsifier_fraction_exact": f"{cumulative_falsifiers}/{post_coverage}",
            "falsifier_fraction_gate_passes": fraction_gate,
            "all_countable_falsifier_ids_exercised": countable_gate,
            "valid_trials_since_trial_36": patience_window_available,
            "patience_may_be_evaluated": patience_may_be_checked,
            "patience_would_be_exhausted_if_no_promotion_eligible_improvement_"
            "after_trial_36": patience_may_be_checked,
        },
    }
    slot["slot_sha256"] = digest_object(slot)
    return slot


def _build_slots() -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    exercised: set[str] = set()
    falsifier_count = 0

    # Trials 37--48 establish the exact fraction and exercise every countable
    # control before any promotion-eligible proposal.  Trials 49--50 are the
    # first two primary-development challengers.  Trials 51--52 continue the
    # frozen policy-order control cycle, restoring 14/34 at the first point
    # where the post-36 patience window may be evaluated.
    for ordinal in range(PRE_TAIL_VALID_TRIALS + 1, PRE_TAIL_VALID_TRIALS + 13):
        falsifier_id = COUNTABLE_OBSERVED_FALSIFIERS[
            falsifier_count % len(COUNTABLE_OBSERVED_FALSIFIERS)
        ]
        cycle_index = falsifier_count // len(COUNTABLE_OBSERVED_FALSIFIERS) + 1
        falsifier_count += 1
        exercised.add(falsifier_id)
        slots.append(
            _make_slot(
                valid_trial_ordinal=ordinal,
                phase="initial_falsifier_patience_tail",
                slot_kind="mandatory_falsifier",
                falsifier_id=falsifier_id,
                falsifier_cycle_index=cycle_index,
                cumulative_falsifiers=falsifier_count,
                exercised_falsifiers=exercised,
            )
        )

    for ordinal in range(PRE_TAIL_VALID_TRIALS + 13, PRE_TAIL_VALID_TRIALS + 15):
        slots.append(
            _make_slot(
                valid_trial_ordinal=ordinal,
                phase="initial_promotable_challengers",
                slot_kind="adaptive_challenger",
                falsifier_id=None,
                falsifier_cycle_index=None,
                cumulative_falsifiers=falsifier_count,
                exercised_falsifiers=exercised,
            )
        )

    for ordinal in range(PRE_TAIL_VALID_TRIALS + 15, PRE_TAIL_VALID_TRIALS + 17):
        falsifier_id = COUNTABLE_OBSERVED_FALSIFIERS[
            falsifier_count % len(COUNTABLE_OBSERVED_FALSIFIERS)
        ]
        cycle_index = falsifier_count // len(COUNTABLE_OBSERVED_FALSIFIERS) + 1
        falsifier_count += 1
        exercised.add(falsifier_id)
        slots.append(
            _make_slot(
                valid_trial_ordinal=ordinal,
                phase="initial_falsifier_patience_tail",
                slot_kind="mandatory_falsifier",
                falsifier_id=falsifier_id,
                falsifier_cycle_index=cycle_index,
                cumulative_falsifiers=falsifier_count,
                exercised_falsifiers=exercised,
            )
        )

    # Reserve slots preserve the 2/5 gate at every prefix.  A falsifier is
    # inserted exactly when the next valid trial would otherwise make the
    # cumulative numerator too small; all other reserve slots are admissible
    # primary-development challengers that may reset patience.
    for ordinal in range(PRE_TAIL_VALID_TRIALS + 17, MAXIMUM_VALID_TRIALS + 1):
        post_coverage = ordinal - COVERAGE_VALID_TRIALS
        must_be_falsifier = falsifier_count < required_falsifier_count(post_coverage)
        if must_be_falsifier:
            falsifier_id = COUNTABLE_OBSERVED_FALSIFIERS[
                falsifier_count % len(COUNTABLE_OBSERVED_FALSIFIERS)
            ]
            cycle_index = falsifier_count // len(COUNTABLE_OBSERVED_FALSIFIERS) + 1
            falsifier_count += 1
            exercised.add(falsifier_id)
            slot_kind = "mandatory_falsifier"
        else:
            falsifier_id = None
            cycle_index = None
            slot_kind = "adaptive_challenger"
        slots.append(
            _make_slot(
                valid_trial_ordinal=ordinal,
                phase="reserve_through_policy_maximum",
                slot_kind=slot_kind,
                falsifier_id=falsifier_id,
                falsifier_cycle_index=cycle_index,
                cumulative_falsifiers=falsifier_count,
                exercised_falsifiers=exercised,
            )
        )
    return slots


def build_plan() -> dict[str, Any]:
    """Build the complete declarative plan without reading run artifacts."""

    slots = _build_slots()
    initial = slots[:PATIENCE_VALID_TRIALS]
    reserve = slots[PATIENCE_VALID_TRIALS:]
    reserve_falsifiers = sum(
        slot["counts_toward_post_coverage_falsifier_numerator"] for slot in reserve
    )
    reserve_promotable = sum(slot["promotion_eligible"] for slot in reserve)
    initial_minimum = minimum_added_falsifiers(
        prior_post_coverage_valid=PRE_TAIL_VALID_TRIALS - COVERAGE_VALID_TRIALS,
        prior_counted_falsifiers=0,
    )
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "outcome-blind observed-development tail after valid trial 36; no final "
            "connectivity access and no scientific outcome values embedded"
        ),
        "policy_identity": {
            "path": "SEARCH_POLICY.yaml",
            "sha256": POLICY_SHA256,
        },
        "prefix_contract": {
            "coverage_boundary_valid_trial": COVERAGE_VALID_TRIALS,
            "pre_tail_valid_trial_count": PRE_TAIL_VALID_TRIALS,
            "pre_tail_post_coverage_valid_count": (
                PRE_TAIL_VALID_TRIALS - COVERAGE_VALID_TRIALS
            ),
            "pre_tail_counted_falsifier_count": 0,
            "pre_tail_counted_falsifier_count_must_be_verified_from_control_"
            "receipts_not_outcome_values": True,
            "branch_coverage_and_two_successor_cycles_must_be_verified": True,
            "failed_invalid_or_readiness_rejected_attempt_advances_valid_ordinal": False,
        },
        "policy_arithmetic": {
            "minimum_valid_trials": MINIMUM_VALID_TRIALS,
            "maximum_valid_trials": MAXIMUM_VALID_TRIALS,
            "patience_valid_trials": PATIENCE_VALID_TRIALS,
            "patience_active_after_trial": PATIENCE_ACTIVE_AFTER_TRIAL,
            "post_coverage_falsifier_fraction": {
                "numerator": FALSIFIER_FRACTION_NUMERATOR,
                "denominator": FALSIFIER_FRACTION_DENOMINATOR,
                "exact_gate": "counted_falsifiers * 5 >= post_coverage_valid_trials * 2",
                "rounding_for_required_count": "ceiling",
            },
            "minimum_added_all_falsifier_trials_needed_from_trial_36": initial_minimum,
            "earliest_fraction_gate_trial_under_initial_tail": (
                PRE_TAIL_VALID_TRIALS + initial_minimum
            ),
            "earliest_all_countable_ids_trial_under_initial_tail": (
                PRE_TAIL_VALID_TRIALS + len(COUNTABLE_OBSERVED_FALSIFIERS)
            ),
            "earliest_qualified_patience_trial_if_no_eligible_improvement": (
                PRE_TAIL_VALID_TRIALS + PATIENCE_VALID_TRIALS
            ),
            "patience_action_rule": (
                "patience can stop only when minimum/coverage prerequisites, the exact "
                "post-coverage fraction, every countable falsifier id, and 16 trailing "
                "valid trials without a promotion-eligible improvement all pass"
            ),
        },
        "required_falsifier_classification": {
            "all_policy_ids": list(ALL_REQUIRED_FALSIFIERS),
            "countable_observed_trial_ids": list(COUNTABLE_OBSERVED_FALSIFIERS),
            "classification_adds_no_new_scientific_threshold": True,
        },
        "non_trial_required_gates": [dict(value) for value in NON_TRIAL_REQUIRED_GATES],
        "initial_tail": {
            "first_valid_trial_ordinal": initial[0]["valid_trial_ordinal"],
            "last_valid_trial_ordinal": initial[-1]["valid_trial_ordinal"],
            "slot_count": len(initial),
            "counted_falsifier_slot_count": sum(
                slot["counts_toward_post_coverage_falsifier_numerator"]
                for slot in initial
            ),
            "promotion_eligible_slot_count": sum(
                slot["promotion_eligible"] for slot in initial
            ),
            "all_countable_ids_exercised_if_every_slot_realizes": True,
            "fraction_at_end_exact": "14/34",
        },
        "reserve_tail": {
            "first_valid_trial_ordinal": reserve[0]["valid_trial_ordinal"],
            "last_valid_trial_ordinal": reserve[-1]["valid_trial_ordinal"],
            "slot_count": len(reserve),
            "counted_falsifier_slot_count": reserve_falsifiers,
            "promotion_eligible_slot_count": reserve_promotable,
            "scheduling_rule": (
                "insert a policy-order falsifier exactly when an adaptive challenger "
                "would make cumulative counted_falsifiers/post_coverage_valid_trials "
                "fall below 2/5"
            ),
            "fraction_at_policy_maximum_exact": "32/78",
        },
        "execution_contract": {
            "valid_trial_ordinal_semantics": (
                "ordinal advances only after all required_realization_evidence passes"
            ),
            "promotion_eligible_false_results_may_replace_incumbent": False,
            "only_promotion_eligible_improvement_resets_patience": True,
            "patience_is_never_actionable_before_fraction_and_countable_id_gates": True,
            "maximum_or_resource_stop_with_failed_promotion_gates_is_not_a_"
            "completed_promotable_search": True,
            "non_trial_required_gates_remain_required_after_observed_tail": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        },
        "all_slots": slots,
    }
    plan["plan_sha256"] = digest_object(plan)
    return plan


def assess_patience(
    *,
    valid_trial_ordinal: int,
    counted_post_coverage_falsifiers: int,
    completed_countable_falsifier_ids: Iterable[str],
    last_promotion_eligible_improvement_ordinal: int,
) -> dict[str, Any]:
    """Evaluate only frozen mechanical gates; no scientific values are read."""

    if not PRE_TAIL_VALID_TRIALS <= valid_trial_ordinal <= MAXIMUM_VALID_TRIALS:
        raise TailPlanError("valid trial ordinal is outside the post-36 plan")
    if not 0 <= last_promotion_eligible_improvement_ordinal <= valid_trial_ordinal:
        raise TailPlanError("last improvement ordinal is inconsistent")
    post_coverage = valid_trial_ordinal - COVERAGE_VALID_TRIALS
    if not 0 <= counted_post_coverage_falsifiers <= post_coverage:
        raise TailPlanError("counted falsifier count is inconsistent")
    completed = set(completed_countable_falsifier_ids)
    unknown = completed - set(COUNTABLE_OBSERVED_FALSIFIERS)
    if unknown:
        raise TailPlanError(f"unknown countable falsifier ids: {sorted(unknown)}")
    fraction_gate = (
        counted_post_coverage_falsifiers * FALSIFIER_FRACTION_DENOMINATOR
        >= post_coverage * FALSIFIER_FRACTION_NUMERATOR
    )
    countable_gate = set(COUNTABLE_OBSERVED_FALSIFIERS).issubset(completed)
    trailing = valid_trial_ordinal - last_promotion_eligible_improvement_ordinal
    trailing_gate = trailing >= PATIENCE_VALID_TRIALS
    valid_trials_since_minimum = valid_trial_ordinal - PRE_TAIL_VALID_TRIALS
    since_minimum_gate = valid_trials_since_minimum >= PATIENCE_VALID_TRIALS
    minimum_gate = valid_trial_ordinal >= MINIMUM_VALID_TRIALS
    active_gate = valid_trial_ordinal >= PATIENCE_ACTIVE_AFTER_TRIAL
    actionable = (
        minimum_gate
        and active_gate
        and fraction_gate
        and countable_gate
        and trailing_gate
        and since_minimum_gate
    )
    failed = [
        name
        for name, passed in (
            ("minimum_valid_trials", minimum_gate),
            ("patience_active_after_trial", active_gate),
            ("post_coverage_falsifier_fraction", fraction_gate),
            ("all_countable_falsifier_ids", countable_gate),
            ("trailing_valid_trials_without_eligible_improvement", trailing_gate),
            ("valid_trials_since_trial_36", since_minimum_gate),
        )
        if not passed
    ]
    return {
        "valid_trial_ordinal": valid_trial_ordinal,
        "post_coverage_valid_trial_count": post_coverage,
        "counted_post_coverage_falsifiers": counted_post_coverage_falsifiers,
        "required_falsifier_count": required_falsifier_count(post_coverage),
        "falsifier_fraction_gate": fraction_gate,
        "all_countable_falsifier_ids_gate": countable_gate,
        "trailing_valid_trials_without_eligible_improvement": trailing,
        "trailing_patience_gate": trailing_gate,
        "valid_trials_since_trial_36": valid_trials_since_minimum,
        "valid_trials_since_trial_36_gate": since_minimum_gate,
        "patience_actionable": actionable,
        "failed_gates": failed,
    }


def validate_plan(
    payload: Mapping[str, Any],
    *,
    policy_path: Path | None = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """Fail closed unless the artifact equals the frozen generated plan."""

    candidate = dict(payload)
    if candidate != build_plan():
        raise TailPlanError("tail plan differs from the frozen declarative plan")
    if candidate.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise TailPlanError("tail plan hash differs from the frozen expected hash")
    if policy_path is not None and not policy_path.is_file():
        raise TailPlanError("SEARCH_POLICY.yaml is missing")
    slots = candidate["all_slots"]
    for slot in slots:
        unhashed = dict(slot)
        recorded = unhashed.pop("slot_sha256")
        if digest_object(unhashed) != recorded:
            raise TailPlanError(f"slot hash mismatch: {slot.get('trial_id')}")
    return {
        "schema_version": candidate["schema_version"],
        "plan_sha256": candidate["plan_sha256"],
        "slot_count": len(slots),
        "initial_tail_slot_count": candidate["initial_tail"]["slot_count"],
        "reserve_tail_slot_count": candidate["reserve_tail"]["slot_count"],
        "final_connectivity_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-observed-tail-plan")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--verify", type=Path, metavar="PLAN_JSON")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.emit:
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return 0
    payload = json.loads(arguments.verify.read_text(encoding="utf-8"))
    print(
        json.dumps(
            validate_plan(payload, policy_path=arguments.policy),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
