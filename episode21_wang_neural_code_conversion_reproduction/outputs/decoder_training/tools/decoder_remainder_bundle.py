"""One explicit serial work6/work13 remainder bundle; metadata operations only.

The array anchor is work6/index5, but both immutable logical remainder IDs
are registered before submission. One physical parent reserves 7,200 CPU
seconds and is charged once. This module performs no I/O or dispatch and
does not replace the full plan/predecessor validation in its caller.
"""
import re


FIELD = "grouped_remainder_work_items"
MEMBERS = ("work_0006", "work_0013")
ANCHOR = MEMBERS[0]
ARRAY_INDEX = 5
RESERVED_CPU_SECONDS = 7200
PLAN_FIELD = "reviewed_deadline_remainder"
PLAN_RELATIVE = "outputs/decoder_training/full_continuation_001/deadline_remainders_002.json"
REVIEW_RELATIVE = "outputs/decoder_training/full_continuation_001/reviewed_deadline_partials_002.json"


class BundleError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BundleError(message)


def validate_bundle(batch):
    """Validate the exact grouping, separately from full scientific plan checks."""
    require(isinstance(batch, dict) and batch.get(FIELD) == list(MEMBERS)
            and batch.get("phase") == "workers"
            and batch.get("manual_worker_refill") is True
            and not any(key in batch for key in ("reviewed_deadline_partials",
                "reviewed_pre_admission_replacement", "reviewed_startup_replacement")),
            "only_explicit_manual_work6_work13_remainder_bundle_supported")
    rows = batch.get("allocations")
    require(rows == [{"logical_id": ANCHOR, "array_index": ARRAY_INDEX,
                      "reserved_CPU_seconds": RESERVED_CPU_SECONDS}]
            and type(rows[0]["array_index"]) is int
            and type(rows[0]["reserved_CPU_seconds"]) is int,
            "bundle_requires_one_exact_index5_7200_second_reservation")
    plan = batch.get(PLAN_FIELD)
    require(isinstance(plan, dict)
            and plan.get("schema_version") == "ep21.decoder_deadline_remainders.v1"
            and plan.get("reviewed_partials") == REVIEW_RELATIVE
            and plan.get("automatic_retry") is False
            and plan.get("preserve_original_fragments") is True
            and plan.get("scope") == "one_manual_exact_remainder_attempt_per_listed_work_item",
            "bundle_requires_explicit_cumulative_plan002")
    items = plan.get("work_items")
    require(isinstance(items, list) and all(isinstance(item, dict) for item in items)
            and [item.get("work_item_id") for item in items]
                == ["work_0005", "work_0006", "work_0007", "work_0013"],
            "bundle_requires_unchanged_ordered_plan002_work_ids")
    for work, index, units in (("work_0006", 5, 57408), ("work_0013", 12, 14976)):
        item = next(item for item in items if item["work_item_id"] == work)
        resources = item.get("resources", {})
        require(type(item.get("array_index")) is int and item["array_index"] == index
                and item.get("participant") == "sub-01"
                and type(item.get("units_requested")) is int and item["units_requested"] == units
                and all(type(resources.get(key)) is int and resources[key] == value
                        for key, value in {"cpus": 1, "memory_gib": 7, "gpus": 0,
                                           "minutes": 480, "scratch_bytes": 2_000_000_000}.items()),
                "bundle_preserves_member_identity_and_original_resource_maxima")
    return MEMBERS


def work_ids_for_job(job, batches):
    """Return logical members without creating another physical accounting row."""
    batch = batches.get(job.get("batch_id"), {})
    if FIELD not in batch:
        return (job["logical_id"],)
    members = validate_bundle(batch)
    accepted = batch.get("accepted_id")
    require(isinstance(accepted, str) and re.fullmatch(r"[1-9][0-9]*", accepted)
            and batch.get("not_submitted") is False
            and job.get("phase") == "workers" and job.get("logical_id") == ANCHOR
            and job.get("job_id") == f"{accepted}_{ARRAY_INDEX}"
            and type(job.get("reserved_CPU_seconds")) is int
            and job["reserved_CPU_seconds"] == RESERVED_CPU_SECONDS,
            "bundle_member_requires_exact_accepted_physical_anchor")
    return members


def admit_member(account, item, allocation, environ):
    """Admit one named child after the caller verifies actual Slurm resources.

The environment remains the real array index5 for both children. The second
child's logical index12 is an immutable plan identity, never a Slurm alias.
"""
    parent, task, raw = (environ.get(key, "") for key in
                        ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID"))
    require(isinstance(parent, str) and re.fullmatch(r"[1-9][0-9]*", parent)
            and task == str(ARRAY_INDEX)
            and isinstance(raw, str) and re.fullmatch(r"[1-9][0-9]*", raw)
            and allocation.get("job_id") == raw,
            "bundle_requires_real_index5_array_and_allocation_identity")
    job = account["jobs"].get(f"{parent}_{task}")
    require(job is not None and job.get("terminal") is None,
            "bundle_requires_outstanding_registered_physical_parent")
    batch = account["batches"].get(job.get("batch_id"), {})
    require(FIELD in batch and item.get("work_item_id") in work_ids_for_job(job, account["batches"]),
            "requested_work_is_not_a_registered_bundle_member")
    planned = next(value for value in batch[PLAN_FIELD]["work_items"]
                   if value["work_item_id"] == item["work_item_id"])
    require(set(item) <= set(planned) | {"committed_spans"}
            and all(item.get(key) == value for key, value in planned.items()),
            "bundle_member_must_match_exact_registered_plan_entry")
    require(all(type(allocation.get(key)) is int and allocation[key] == value
                for key, value in {"allocated_cpus": 1, "memory_gib": 7, "gpus": 0}.items())
            and type(allocation.get("time_limit_seconds")) is int
            and 0 < allocation["time_limit_seconds"] <= RESERVED_CPU_SECONDS,
            "bundle_actual_allocation_exceeds_one_CPU_7GiB_two_hours")
    actual, reserved, cap = (account.get("actual_CPU_seconds"),
                             account.get("outstanding_reserved_CPU_seconds"),
                             account.get("budget", {}).get("cpu_seconds"))
    require(all(type(value) is int for value in (actual, reserved, cap))
            and actual >= 0 and reserved >= RESERVED_CPU_SECONDS
            and actual + reserved <= cap
            and not account.get("uncertain_batches") and not account.get("rejected_batch"),
            "bundle_parent_is_unfunded_or_submission_history_unresolved")
    return job
