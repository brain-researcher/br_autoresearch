"""One prospective user-authorized parallelism amendment; metadata only.

Original authority, manifests, roles, CPU/durable limits and past intents stay
unchanged. A physical worker retains one complete channel bank at a time.
The fixed 64 MiB headroom also bounds its serial small startup metadata.
No scheduler calls, payload access, dispatch or automatic retry occur here.
"""
from datetime import datetime

import decoder_coverage as coverage
import decoder_remainder_bundle as bundle

EVENT = "parallelism_amendment"
HEADROOM = 64 * 1024**2
FIXED_POLICY = {
    "schema_version": "ep21.decoder_parallelism_amendment.v1",
    "authorized_by": "user",
    "maximum_concurrent_jobs": 42,
    "scratch_headroom_bytes": HEADROOM,
    "scratch_rule": "max_declared_complete_channel_bank_per_physical_worker_plus_headroom",
    "maximum_allocated_CPU_seconds": 500 * 3600,
    "durable_bytes": 500_000_000_000,
    "peak_transient_bytes": 16_000_000_000,
    "original_authority_and_manifest_unchanged": True,
    "scientific_fit_roles_and_kernels_unchanged": True,
    "automatic_retry": False,
}


class ParallelismError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ParallelismError(reason)


def validate_policy(policy):
    require(isinstance(policy, dict) and set(policy) == set(FIXED_POLICY)
            and all(type(policy[key]) is type(value) and policy[key] == value
                    for key, value in FIXED_POLICY.items()),
            "exact_user_parallelism_policy_required")


def effective_maximum(authority, policy):
    if policy is None:
        return min(8, authority["resources"]["maximum_concurrent_jobs"])
    validate_policy(policy)
    return policy["maximum_concurrent_jobs"]


def work_scratch_bytes(manifest, work_ids):
    """Complete bank size, even for a partial spatial interval; bundle is serial."""
    require(isinstance(work_ids, (tuple, list)) and work_ids
            and all(isinstance(work, str) for work in work_ids)
            and len(set(work_ids)) == len(work_ids), "explicit_unique_physical_worker_members_required")
    items = {item["work_item_id"]: item for item in manifest["work_items"]}
    require(len(items) == len(manifest["work_items"]) and all(work in items for work in work_ids),
            "parallelism_members_must_belong_to_immutable_manifest")
    largest = 0
    for work in work_ids:
        item = items[work]
        require(isinstance(item.get("spans"), list) and item["spans"],
                "declared_channel_spans_required_for_scratch_reservation")
        for record in item["spans"]:
            require(isinstance(record, dict) and set(record) == {
                "participant", "layer", "channel", "spatial_Fortran_interval"},
                "exact_declared_channel_identity_required")
            interval = record["spatial_Fortran_interval"]
            require(isinstance(interval, list) and len(interval) == 2,
                    "exact_declared_channel_interval_required")
            span = coverage.Span(record["participant"], record["layer"], record["channel"], *interval).validate()
            require(span.participant == item["participant"], "work_participant_must_match_declared_channel")
            shape = span.shape
            units = shape[1] * shape[2] if len(shape) == 3 else shape[0]
            amount = coverage.prefix.IMAGES * units * 4
            require(amount <= item["resources"]["scratch_bytes"],
                    "derived_bank_exceeds_original_per_worker_cap")
            largest = max(largest, amount)
    return largest + HEADROOM


def outstanding_scratch_bytes(jobs, batches, manifest, policy):
    validate_policy(policy)
    total = 0
    for job in jobs.values():
        if job["terminal"] is not None:
            continue
        require(job["phase"] == "workers" and job.get("batch_id") in batches
                and batches[job["batch_id"]].get("accepted_id") is not None,
                "parallelism_requires_known_outstanding_worker_parents")
        total += work_scratch_bytes(manifest, bundle.work_ids_for_job(job, batches))
    return total


def validate_reservation(row, jobs, batches, manifest, policy):
    """Count accepted outstanding parents and all proposed physical parents."""
    validate_policy(policy)
    require(row.get("phase") == "workers" and row.get("manual_worker_refill") is True,
            "parallelism_only_for_explicit_manual_worker_admission")
    outstanding = [job for job in jobs.values() if job["terminal"] is None]
    allocations = row.get("allocations")
    require(isinstance(allocations, list) and allocations,
            "parallelism_requires_nonempty_physical_allocations")
    require(len(outstanding) + len(allocations) <= policy["maximum_concurrent_jobs"],
            "parallelism_outstanding_plus_new_exceeds_42_worker_cap")
    scratch = outstanding_scratch_bytes(jobs, batches, manifest, policy)
    if bundle.FIELD in row:
        scratch += work_scratch_bytes(manifest, bundle.validate_bundle(row))
    else:
        for allocation in allocations:
            scratch += work_scratch_bytes(manifest, (allocation["logical_id"],))
    require(scratch <= policy["peak_transient_bytes"],
            "parallelism_outstanding_plus_new_scratch_exceeds_16GB")
    return scratch


def make_event(*, observed_utc, user_instruction):
    return {"event": EVENT, "observed_utc": observed_utc,
            "user_instruction": user_instruction, "policy": dict(FIXED_POLICY)}


def validate_event(event, authority, manifest, jobs, batches):
    required = {"event", "observed_utc", "user_instruction", "policy"}
    require(isinstance(event, dict) and set(event) in (required, required | {"utc"})
            and event["event"] == EVENT and isinstance(event["user_instruction"], str)
            and 0 < len(event["user_instruction"].strip()) <= 4096
            and isinstance(event["observed_utc"], str), "explicit_dated_user_parallelism_amendment_required")
    try:
        observed = datetime.fromisoformat(event["observed_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParallelismError("dated_parallelism_amendment_required") from exc
    require(observed.utcoffset() is not None, "timezone_bearing_parallelism_amendment_required")
    policy = event["policy"]
    validate_policy(policy)
    caps = authority["resources"]
    require(caps.get("maximum_concurrent_jobs") == 8
            and all(type(caps.get(key)) is int and caps[key] == policy[key] for key in
                    ("maximum_allocated_CPU_seconds", "durable_bytes", "peak_transient_bytes")),
            "parallelism_preserves_original_500h_500GB_16GB_caps")
    require(all(batch.get("accepted_id") is not None and batch.get("not_submitted") is False
                for batch in batches.values()), "parallelism_requires_reconciled_accepted_history")
    require(not any(job["phase"] == "assembly" for job in jobs.values()),
            "parallelism_does_not_amend_assembly_authority")
    require(sum(job["terminal"] is None for job in jobs.values()) <= policy["maximum_concurrent_jobs"]
            and outstanding_scratch_bytes(jobs, batches, manifest, policy) <= policy["peak_transient_bytes"],
            "existing_workers_exceed_amended_parallelism_envelope")
    return dict(policy)
