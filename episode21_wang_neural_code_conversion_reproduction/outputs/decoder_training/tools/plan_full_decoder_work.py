"""Unactivated full decoder assignment planning; no scientific payload I/O.

The supplied completed reports must describe closed allocations and exact
committed coverage. Planned spans never count as retained fits. This module
does not submit jobs, create authority, build states or fit any decoder.
Run its checks only in a coordinated substantive Slurm allocation.
"""
from copy import deepcopy

import decoder_coverage as coverage

require = coverage.require


def remaining_after(retained):
    """Subtract every actually retained interval, including exact partial fits."""
    retained = tuple(retained)
    required = {span.key: span for span in coverage.complete_spans()}
    grouped = {}
    for span in retained:
        span.validate()
        require(span.key in required, "retained_identity_outside_full_target")
        grouped.setdefault(span.key, []).append(span)
    remaining = []
    for key, whole in required.items():
        cursor = 0
        for old in sorted(grouped.get(key, ()), key=lambda item: item.start):
            require(old.start >= cursor, "retained_coverage_overlap_or_duplicate")
            if old.start > cursor:
                remaining.append(coverage.Span(*key, cursor, old.start))
            cursor = old.stop
        if cursor < whole.stop:
            remaining.append(coverage.Span(*key, cursor, whole.stop))
    require(coverage.verify_complete_coverage((*retained, *remaining)) == coverage.TOTAL,
            "full_target_coverage_required")
    return tuple(remaining)


def plan_from_spans(retained, *, max_units=1_003_520, minutes=480):
    """Pure prospective assignment calculation, not admission of saved results.

    Production callers must use build_plan(), which first validates terminal
    reports. The exact partition and physical Slurm job IDs are assigned only
    after separate approval; these spending scenarios confer no authority.
    """
    require(type(minutes) is int and 10 <= minutes <= 480,
            "finite_prospective_work_item_ceiling_required")
    require(type(max_units) is int and 50176 <= max_units <= 1_003_520,
            "prospective_whole_channel_packing_limit_required")
    retained = tuple(retained)
    remaining = remaining_after(retained)
    items = []
    for ordinal, spans in enumerate(coverage.pack_work(remaining, max_units=max_units), 1):
        count = sum(span.count for span in spans)
        items.append({
            "work_item_id": f"work_{ordinal:04d}",
            "participant": spans[0].participant,
            "spans": [span.record() for span in spans],
            "units_requested": count,
            "resources": {"cpus": 1, "memory_gib": 7, "gpus": 0,
                          "minutes": minutes,
                          # Dense late-layer work items can contain thousands
                          # of channel groups. Both request and final status
                          # repeat span identities; reserve bytes per span.
                          "durable_bytes": count * 6045 + len(spans) * (1_048_576 + 4096) + 65_536,
                          "scratch_bytes": 2_000_000_000},
        })
    planned = sum(item["units_requested"] for item in items)
    retained_count = sum(span.count for span in retained)
    require(retained_count + planned == coverage.TOTAL, "every_required_unit_assigned_once")
    # Four new participant states share one sequential parent. Sub01's existing
    # state is reused. Assembly follows all fits, so phase budgets do not imply
    # simultaneous state, fit and assembly allocations.
    state_resources = {"cpus": 1, "memory_gib": 7, "gpus": 0,
                       "minutes": 240, "durable_bytes": 8_000_000_000,
                       "scratch_bytes": 2_000_000_000}
    assembly_resources = {"cpus": 1, "memory_gib": 7, "gpus": 0,
                          "minutes": 240, "durable_bytes": 1_000_000_000,
                          "scratch_bytes": 2_000_000_000}
    durable = sum(item["resources"]["durable_bytes"] for item in items)
    durable += state_resources["durable_bytes"] + assembly_resources["durable_bytes"]
    require(durable <= 500_000_000_000, "prospective_full_stage_storage_envelope_exceeded")
    return {
        "schema_version": "ep21.decoder_work_manifest.v1",
        "planning_only": True,
        "execution_authorized": False,
        "proposed_budget_mode": "actual_allocated_with_full_outstanding_reservations",
        "proposed_additional_CPU_seconds": 500 * 3600,
        "full_reservation_scenario_is_not_dispatch_authority": True,
        "included_layer_reservation_and_journal_must_fit_within_storage_envelope": True,
        "continuation_cause_admission": "not_established_by_identity_planning",
        "participants": list(coverage.PARTICIPANTS),
        "layers": list(coverage.prefix.LAYERS),
        "total_required_units": coverage.TOTAL,
        "training_role": "subject_unique", "training_images": 4712,
        "numerical_block_units": 64,
        "retained_units": retained_count, "remaining_units": planned,
        "maximum_units_per_work_item": max_units,
        "work_items": items,
        "state_participants_to_build": list(coverage.PARTICIPANTS[1:]),
        "state_build_resources": state_resources,
        "assembly_resources": assembly_resources,
        "prospective_aggregate": {
            "maximum_fit_concurrency": min(8, len(items)),
            "maximum_fit_cpus": min(8, len(items)),
            "maximum_fit_memory_gib": min(8, len(items)) * 7,
            "maximum_fit_peak_scratch_bytes": min(8, len(items)) * 2_000_000_000,
            "maximum_parent_jobs": len(items) + 2,
            "maximum_allocated_cpu_seconds": (len(items) * minutes + 480) * 60,
            "maximum_new_durable_bytes": durable,
            "storage_envelope_bytes": 500_000_000_000,
            "charge_each_parent_once": True,
            "phase_order": ["four_new_states_in_one_parent", "fit_work_items", "assembly"],
        },
        "automatic_retry": False, "automatic_extension": False,
        "predictions_or_scores_authorized": False,
        "tau_or_OOD_authorized": False,
        "complete_decoder": False,
    }


def build_plan(retained_entries, *, max_units=1_003_520, minutes=480):
    """Validate supplied terminal metadata before calculating pending identities.

    No directory discovery, fallback report or live-producer observation is
    performed. Uncertain publication requires a separately resolved accounting
    decision; it must not turn into permission to replay possibly completed rows.
    The eventual assembler independently checks actual fragment payloads.
    Exact retained coverage does not make an unresolved numerical/source failure
    runnable; the worker separately admits each terminal continuation cause.
    """
    from decoder_execution_contract import read_retained_coverage

    entries = deepcopy(list(retained_entries))
    retained = read_retained_coverage(entries)
    plan = plan_from_spans(retained, max_units=max_units, minutes=minutes)
    plan["retained"] = entries
    plan["retained_payloads_verified_by_planner"] = False
    return plan
