"""Explicit immutable remainder plans for diagnosed deadline stops.

Pure standard-library metadata operations. No reads, scheduling, retries,
normalization, numerical fitting or authority changes occur on import/call.
The original 75 assignments, original fragments and every CPU charge persist.
"""
from datetime import datetime
from pathlib import PurePosixPath

import reviewed_deadline_partials as partials
import decoder_remainder_bundle as bundle
import reviewed_bundle_startup_replacement as bundle_startup


FIELD = "reviewed_deadline_remainder"
ORIGINAL_MANIFEST_RELATIVE = partials.ROOT + "/work_manifest.json"
PLAN_RELATIVE = partials.ROOT + "/deadline_remainders_001.json"
REVIEW_RELATIVE = partials.ROOT + "/reviewed_deadline_partials_001.json"
WORK_IDS = ("work_0005", "work_0006", "work_0007")
PLAN_RELATIVE_002 = partials.ROOT + "/deadline_remainders_002.json"
REVIEW_RELATIVE_002 = partials.ROOT + "/reviewed_deadline_partials_002.json"
WORK_IDS_002 = WORK_IDS + ("work_0013",)
# Only observed October7 deadline stops; not a generic retry route.
KNOWN = {
    "work_0005": (4, "46839670_4", "46840029", 246656),
    "work_0006": (5, "46839670_5", "46840030", 57408),
    "work_0007": (6, "46839670_6", "46840031", 51776),
    "work_0013": (12, "46881096_12", "46881096", 14976),
}
METADATA_BYTES = 64*1024**2
EXTRA_DURABLE_BYTES = 2_368_108_032
EXTRA_DURABLE_BYTES_002 = 2_546_718_336
require = partials.require


def plan_relative(plan):
    require(isinstance(plan, dict) and plan.get("reviewed_partials") in
            (REVIEW_RELATIVE, REVIEW_RELATIVE_002), "explicit_known_remainder_plan_required")
    return PLAN_RELATIVE if plan["reviewed_partials"] == REVIEW_RELATIVE else PLAN_RELATIVE_002


def plans_compatible(left, right):
    """The cumulative supplement retains every item of the immutable first plan."""
    left_path, right_path = plan_relative(left), plan_relative(right)
    if left_path == right_path:
        return left == right
    old, new = (left, right) if left_path == PLAN_RELATIVE else (right, left)
    fixed = set(old) - {"reviewed_partials", "work_items"}
    return (set(new) == set(old) and all(old[key] == new[key] for key in fixed)
            and new["work_items"][:len(old["work_items"])] == old["work_items"])


def latest_published_review(account):
    reviews = [batch[partials.FIELD] for batch in account["batches"].values()
               if partials.FIELD in batch and batch["accepted_id"] is not None]
    require(reviews, "remainder_requires_unchanged_published_partial_review")
    return reviews[-1]


def reserved_durable_bytes(batches):
    """Reserve each immutable physical output once, including unaccepted intents."""
    items = {}
    for batch in batches.values():
        if FIELD not in batch:
            continue
        for item in batch[FIELD]["work_items"]:
            work = item["work_item_id"]
            require(work not in items or items[work] == item, "published_remainder_item_is_immutable")
            items[work] = item
    return sum(item["resources"]["durable_bytes"] for item in items.values())


def remainder_resources(units, spans):
    return {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 480,
            "durable_bytes": units*6045+spans*1024**2+METADATA_BYTES,
            "scratch_bytes": 2_000_000_000}


def validate_plan_structure(plan, authority, original_manifest):
    supplemental = plan_relative(plan) == PLAN_RELATIVE_002
    work_ids = WORK_IDS_002 if supplemental else WORK_IDS
    exact = {"schema_version": "ep21.decoder_deadline_remainders.v1",
             "original_work_manifest": ORIGINAL_MANIFEST_RELATIVE,
             "reviewed_partials": REVIEW_RELATIVE_002 if supplemental else REVIEW_RELATIVE,
             "automatic_retry": False,
             "preserve_original_fragments": True,
             "scope": "one_manual_exact_remainder_attempt_per_listed_work_item"}
    require(isinstance(plan, dict) and set(plan) == set(exact) | {"work_items"}
            and all(type(plan[key]) is type(value) and plan[key] == value for key, value in exact.items()),
            "exact_known_deadline_remainder_plan_required")
    items = plan["work_items"]
    require(isinstance(items, list) and len(items) == len(work_ids)
            and all(isinstance(item, dict) for item in items)
            and [item.get("work_item_id") for item in items] == list(work_ids),
            "only_ordered_diagnosed_deadline_remainders_required")
    original = original_manifest["work_items"]
    require(len(original) == authority["resources"]["worker_jobs"] == 75,
            "unchanged_75_logical_assignments_required")
    fields = {"work_item_id", "participant", "array_index", "predecessor_job_id",
              "predecessor_allocation_job_id", "predecessor_status", "output_directory",
              "spans", "units_requested", "resources"}
    for item in items:
        work = item["work_item_id"]
        index, predecessor, allocation, units = KNOWN[work]
        generation = "002" if work == "work_0013" else "001"
        parent = original[index]
        require(set(item) == fields and parent["work_item_id"] == work
                and type(item["array_index"]) is int and item["array_index"] == index
                and item["participant"] == parent["participant"] == "sub-01"
                and item["predecessor_job_id"] == predecessor
                and item["predecessor_allocation_job_id"] == allocation
                and item["predecessor_status"] == f"{partials.ROOT}/shards/{work}/work_item_status.json"
                and item["output_directory"] == f"{partials.ROOT}/remainder_shards_{generation}/{work}",
                "remainder_identity_or_exclusive_path_mismatch")
        spans, requested = partials._spans(item["spans"]), partials._spans(parent["spans"])
        require(len(spans) == (20 if work == "work_0013" else 5)
                and list(spans) == [key for key in requested if key in spans]
                and all(key in requested and requested[key][0] <= begin < end == requested[key][1]
                        for key, (begin, end) in spans.items())
                and type(item["units_requested"]) is int and item["units_requested"] == units
                and sum(end-begin for begin, end in spans.values()) == units,
                "only_exact_sized_remaining_intervals_may_be_assigned")
        expected = remainder_resources(units, len(spans))
        require(isinstance(item["resources"], dict) and item["resources"] == expected
                and all(type(value) is int for value in item["resources"].values()),
                "fixed_remainder_resources_and_conservative_storage_required")
    extra = sum(item["resources"]["durable_bytes"] for item in items)
    require(extra == (EXTRA_DURABLE_BYTES_002 if supplemental else EXTRA_DURABLE_BYTES),
            "exact_extra_remainder_storage_reservation_required")
    # Keep the original whole-manifest reservations, including unfilled HDF5
    # space. This does not speculate about reclaiming any closed shard's cap.
    original_storage = (sum(item["resources"]["durable_bytes"] for item in original)
        + authority["state_build_resources"]["durable_bytes"]
        + authority["assembly_resources"]["durable_bytes"] + METADATA_BYTES
        + authority["closed_included_layer_storage"]["charged_bytes"])
    caps = authority["resources"]
    require(original_storage+extra <= caps["durable_bytes"] <= 500_000_000_000
            and caps["maximum_concurrent_jobs"] <= 8
            and 2_000_000_000*caps["maximum_concurrent_jobs"] <= caps["peak_transient_bytes"],
            "remainder_storage_or_concurrent_scratch_exceeds_original_cap")
    return {item["work_item_id"]: item for item in items}


def _published_review(account, review):
    prior = [batch[partials.FIELD] for batch in account["batches"].values()
             if partials.FIELD in batch and batch["accepted_id"] is not None]
    require(prior and any(value == review for value in prior),
            "remainder_requires_unchanged_published_partial_review")
    latest = prior[-1]
    fixed = set(review) - {"reviewed_utc", "partials"}
    require(latest == review or (
        set(latest) == set(review) and all(latest[key] == review[key] for key in fixed)
        and len(latest["partials"]) > len(review["partials"])
        and latest["partials"][:len(review["partials"])] == review["partials"]
        and datetime.fromisoformat(latest["reviewed_utc"].replace("Z", "+00:00"))
            > datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))),
        "remainder_requires_preserved_published_partial_review")


def validate_plan(plan, authority, original_manifest, reviewed_partials,
                  predecessor_records, account, *, base_directory):
    items = validate_plan_structure(plan, authority, original_manifest)
    _published_review(account, reviewed_partials)
    # Reuse exact review/terminal validation without selecting/retrying any of
    # the reviewed jobs. This structural call grants no worker admission.
    reviewed = partials.validate_intent({"phase": "workers", "manual_worker_refill": True,
        "allocations": [], partials.FIELD: reviewed_partials}, account["jobs"], original_manifest)
    require(set(entry["work_item_id"] for entry in reviewed.values()) == set(items)
            and set(predecessor_records) == set(items), "exact_planned_predecessor_records_required")
    base = PurePosixPath(base_directory)
    result = {}
    for work, item in items.items():
        entry = reviewed[item["predecessor_job_id"]]
        record = predecessor_records[work]
        original = original_manifest["work_items"][item["array_index"]]
        partials.validate_closed_record(entry, record, original,
            authorization_path=str(base / "outputs/decoder_training_full_authorization.json"),
            manifest_path=str(base / ORIGINAL_MANIFEST_RELATIVE),
            shared_state_path=str(base / "outputs/decoder_training/retained_prefix_001/shared_training_state.npz"))
        require(item["spans"] == record["remaining_spans"]
                and item["units_requested"] == original["units_requested"]-record["completed_units"],
                "remainder_must_equal_exact_closed_uncommitted_intervals")
        result[work] = {**item, "committed_spans": record["committed_spans"]}
    return result


def validate_remainder_intent(batch, jobs, batches, authority, original_manifest):
    """One explicit extra physical attempt, charged normally, per named work."""
    require(batch.get("phase") == "workers" and batch.get("manual_worker_refill") is True
            and partials.FIELD not in batch and "reviewed_pre_admission_replacement" not in batch,
            "remainder_requires_its_own_explicit_manual_worker_intent")
    plan = batch.get(FIELD)
    items = validate_plan_structure(plan, authority, original_manifest)
    reviews = [other[partials.FIELD] for other in batches.values()
               if partials.FIELD in other and other["accepted_id"] is not None]
    matching = [review for review in reviews
                if [entry["work_item_id"] for entry in review["partials"]] == list(items)]
    require(matching, "remainder_requires_published_review")
    review = matching[-1]
    _published_review({"batches": batches}, review)
    reviewed = partials.validate_intent({"phase": "workers", "manual_worker_refill": True,
        "allocations": [], partials.FIELD: review}, jobs, original_manifest)
    require(set(reviewed) == {KNOWN[work][1] for work in items},
            "exact_planned_published_predecessors_required")
    allocations = batch.get("allocations")
    require(isinstance(allocations, list) and allocations
            and len({row["logical_id"] for row in allocations}) == len(allocations),
            "explicit_unique_remainder_selection_required")
    if bundle.FIELD in batch:
        selected = bundle.validate_bundle(batch)
    else:
        selected = [allocation["logical_id"] for allocation in allocations]
        for allocation in allocations:
            work = allocation["logical_id"]
            require(work in items and allocation == {"logical_id": work,
                "array_index": items[work]["array_index"], "reserved_CPU_seconds": 28800},
                "remainder_must_keep_original_index_and_full_eight_hour_reservation")
        require("work_0013" not in selected, "short_work13_remainder_requires_substantive_bundle")
    superseded = {other[field]["predecessor_job_id"] for other in batches.values()
                  for field in ("reviewed_pre_admission_replacement", "reviewed_startup_replacement")
                  if field in other and other["accepted_id"] is not None}
    superseded.update(bundle_startup.superseded_jobs(batches))
    proposed_startup = bundle_startup.validate_intent(batch, jobs, batches)
    if proposed_startup is not None:
        superseded.add(proposed_startup)
    for work in selected:
        predecessors = [job for job in jobs.values() if work in bundle.work_ids_for_job(job, batches)
                        and job["job_id"] not in superseded]
        require(len(predecessors) == 1 and predecessors[0]["job_id"] == items[work]["predecessor_job_id"]
                and predecessors[0]["terminal"] is not None,
                "one_original_closed_attempt_required_before_exact_remainder")
        require(not any(FIELD in other and (work in other.get(bundle.FIELD, ())
                         or any(row["logical_id"] == work for row in other["allocations"]))
                        and not (other.get("accepted_id") is not None
                                 and f'{other["accepted_id"]}_{bundle.ARRAY_INDEX}' in superseded)
                        for other in batches.values()), "one_manual_remainder_attempt_only_no_retry")
    require(all(plans_compatible(other[FIELD], plan) for other in batches.values() if FIELD in other),
            "published_remainder_plan_is_immutable")


def registered_remainder_job(account, plan, work_item_id, *, require_terminal=False):
    wanted = next((item for item in plan["work_items"] if item["work_item_id"] == work_item_id), None)
    require(wanted is not None, "registered_remainder_must_belong_to_selected_plan")
    superseded = bundle_startup.superseded_jobs(account["batches"])
    candidates = [job for job in account["jobs"].values()
        if job["job_id"] not in superseded
        and job["phase"] == "workers" and work_item_id in bundle.work_ids_for_job(job, account["batches"])
        and job["batch_id"] in account["batches"]
        and FIELD in account["batches"][job["batch_id"]]
        and plans_compatible(account["batches"][job["batch_id"]][FIELD], plan)
        and wanted in account["batches"][job["batch_id"]][FIELD]["work_items"]
        and account["batches"][job["batch_id"]]["accepted_id"] is not None]
    require(len(candidates) == 1, "one_exact_accepted_remainder_allocation_required")
    job = candidates[0]
    if require_terminal:
        terminal = job["terminal"]
        require(terminal is not None and terminal["state"] == "COMPLETED"
                and terminal["exit_code"] == "0:0" and terminal["allocated_cpus"] == 1,
                "remainder_assembly_requires_successfully_charged_terminal_allocation")
    else:
        require(job["terminal"] is None, "remainder_worker_allocation_must_be_outstanding")
    return job


def validate_completed_remainder(item, record, registered_job, *, base_directory,
                                 remainder_plan_relative=PLAN_RELATIVE, registered_work_items=None):
    base = PurePosixPath(base_directory)
    terminal = registered_job["terminal"]
    require(remainder_plan_relative in (PLAN_RELATIVE, PLAN_RELATIVE_002)
            and (item["work_item_id"] in WORK_IDS or remainder_plan_relative == PLAN_RELATIVE_002),
            "completed_remainder_requires_its_registered_plan")
    members = ((registered_job["logical_id"],) if registered_work_items is None else registered_work_items)
    grouped = members == bundle.MEMBERS
    require(members == (registered_job["logical_id"],) or grouped
            and registered_job["logical_id"] == bundle.ANCHOR
            and registered_job.get("reserved_CPU_seconds") == bundle.RESERVED_CPU_SECONDS,
            "completed_remainder_requires_registered_logical_membership")
    require(not grouped or record.get(bundle.FIELD) == list(bundle.MEMBERS),
            "completed_bundle_child_requires_explicit_common_parent_membership")
    require(terminal is not None and terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
            and terminal["allocated_cpus"] == 1
            and item["work_item_id"] in members,
            "completed_remainder_requires_exact_charged_job")
    allocation = record.get("allocation", {})
    require(record.get("schema_version") == "ep21.decoder_work_item.v1"
            and record.get("status") == "complete_work_item_only"
            and record.get("work_item_id") == item["work_item_id"]
            and record.get("participant") == item["participant"]
            and record.get("resources") == item["resources"]
            and allocation.get("job_id") == terminal["allocation_job_id"]
            and allocation.get("allocated_cpus") == 1 and allocation.get("memory_gib") == 7
            and allocation.get("gpus") == 0
            and (type(allocation.get("time_limit_seconds")) is int
                 and 0 < allocation["time_limit_seconds"] <= bundle.RESERVED_CPU_SECONDS
                 if grouped else allocation.get("time_limit_seconds") == 28800)
            and record.get("authorization") == str(base / "outputs/decoder_training_full_authorization.json")
            and record.get("work_manifest") == str(base / ORIGINAL_MANIFEST_RELATIVE)
            and record.get("remainder_plan") == str(base / remainder_plan_relative)
            and record.get("predecessor_job_id") == item["predecessor_job_id"]
            and record.get("predecessor_status") == item["predecessor_status"]
            and record.get("shared_state_reference") == str(base / "outputs/decoder_training/retained_prefix_001/shared_training_state.npz")
            and record.get("fragment") == "channels.h5", "completed_remainder_identity_mismatch")
    require(record.get("requested_spans") == record.get("committed_spans") == item["spans"]
            and record.get("remaining_spans") == []
            and type(record.get("completed_units")) is int
            and record.get("units_requested") == record["completed_units"] == item["units_requested"]
            and record.get("remaining_interval_status") == "exact_uncommitted_coverage"
            and record.get("completed_units_is_lower_bound") is False
            and "uncertain_publication" not in record
            and isinstance(record.get("closeout_errors"), list)
            and all(isinstance(error, dict) and error.get("affects_fragment") is False
                    for error in record["closeout_errors"]), "completed_remainder_must_cover_exact_fresh_spans")
    require(record.get("training_role") == "subject_unique" and record.get("training_images") == 4712
            and record.get("original_neural_values_read") is False
            and record.get("retained_neural_state_read") is True
            and record.get("original_budget_consumption_preserved") is True
            and all(record.get(key) is False for key in ("predictions_or_scores_computed", "tau_train_values_read",
                "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized", "automatic_retry")),
            "completed_remainder_must_preserve_decoder_only_scope")
