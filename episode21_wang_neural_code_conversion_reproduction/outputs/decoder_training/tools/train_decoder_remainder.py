"""One manually admitted exact deadline or reviewed I/O remainder; no dispatch.

Reuses the existing fit_span kernel, channel staging/store and closeout. It
never opens the original partial fragment for writing or changes its work
manifest. This entrypoint needs separate exact accepted journal registration.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import decoder_batch_accounting as accounting
import decoder_deadline_remainders as remainders
import decoder_remainder_bundle as bundle
import reviewed_io_quarantine as io_quarantine
import train_decoder_work_item as original

contract, state, layer, prefix = original.contract, original.state, original.layer, original.prefix
require, Stop = original.require, original.Stop


def remainder_runtime_resources(manifest, work_item, resources, admitted_policy):
    """Bind the physical scratch reservation to the complete original work item.

    A remainder's smaller interval does not shrink the full channel bank, and
    a prior serial bundle keeps its original admission and resource record.
    """
    if admitted_policy is None:
        return dict(resources)
    selected = [item for item in manifest["work_items"] if item["work_item_id"] == work_item]
    require(len(selected) == 1, "one_original_work_item_for_remainder_scratch_required")
    full_spans = tuple(contract.parse_span(row) for row in selected[0]["spans"])
    return original.runtime_resources(resources, full_spans, admitted_policy)


def admit_io_recovery(review_path, authority, manifest):
    """Read only the fixed review, closed operational evidence and journal.

    The old failed HDF5 files are never opened or used as fitting inputs.
    This helper grants no new submission; the exact accepted worker is bound
    separately below before source metadata or shared training state is read.
    """
    import reviewed_io_recovery as recovery
    review_path = Path(review_path).absolute()
    require(review_path == prefix.BASE / recovery.REVIEW_RELATIVE
            and review_path.is_file() and not review_path.is_symlink()
            and review_path.resolve() == review_path,
            "fixed_exact_IO_recovery_review_required")
    review = prefix.read_json(review_path)
    planned = recovery.validate_review_structure(review, manifest)
    with accounting.locked_events(contract.ROOT / "submission.jsonl", shared=True) as (events, _):
        account = accounting.replay(events, authority, manifest)
    records, reports = {}, {}
    for work, item in planned.items():
        paths = [item["predecessor_status"], item["inspection"]["report"]]
        for relative in paths:
            path = prefix.BASE / relative
            require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
                    "unchanged_closed_IO_metadata_required")
        records[work] = prefix.read_json(prefix.BASE / item["predecessor_status"])
        report_path = item["inspection"]["report"]
        if report_path not in reports:
            reports[report_path] = prefix.read_json(prefix.BASE / report_path)
    items = recovery.validate_review(review, manifest, account["jobs"], records, reports,
                                     base_directory=str(prefix.BASE))
    return review, items, account


def admit_io_recovery_allocation(authority, manifest, review, account, item, allocation):
    """Bind current exact array task and its already admitted effective cap."""
    import reviewed_io_recovery as recovery
    job = recovery.registered_remainder_job(account, review, item["work_item_id"])
    registered_batch = account["batches"][job["batch_id"]]
    require(registered_batch.get(recovery.FIELD) == review and bundle.FIELD not in registered_batch
            and os.environ.get("SLURM_ARRAY_TASK_ID") == str(item["array_index"])
            and job["job_id"] == f"{os.environ.get('SLURM_ARRAY_JOB_ID')}_{item['array_index']}",
            "exact_registered_IO_remainder_allocation_required")
    policy = contract.admit_registered_allocation(authority, manifest, item["work_item_id"], allocation)
    require(policy == {"effective_scratch_bytes": item["effective_scratch_bytes"],
                       "parallelism_amendment_id": "parallelism_amendment"}
            and type(policy.get("effective_scratch_bytes")) is int,
            "accepted_IO_remainder_effective_scratch_policy_required")
    return job, registered_batch, policy


def run(args):
    require(Path.cwd().resolve() == prefix.BASE, "exact_episode_working_directory_required")
    authority = state.authorize_full(args.authorization)
    require(Path(args.manifest).absolute() == contract.MANIFEST and not contract.MANIFEST.is_symlink(),
            "unchanged_full_manifest_required")
    io_path = getattr(args, "io_recovery_review", None)
    require((args.remainder_plan is None) != (io_path is None), "one_explicit_remainder_route_required")
    io_mode = io_path is not None
    plan_path = Path(io_path if io_mode else args.remainder_plan).absolute()
    if not io_mode:
        require(plan_path in (prefix.BASE / remainders.PLAN_RELATIVE, prefix.BASE / remainders.PLAN_RELATIVE_002)
                and plan_path.is_file()
                and not plan_path.is_symlink(), "fixed_exact_remainder_plan_required")
    manifest = prefix.read_json(contract.MANIFEST)
    contract.validate_work_manifest(manifest, authority)
    if io_mode:
        plan, items, account = admit_io_recovery(plan_path, authority, manifest)
    else:
        plan = prefix.read_json(plan_path)
        require(prefix.BASE / remainders.plan_relative(plan) == plan_path, "remainder_plan_content_identity_mismatch")
        planned = remainders.validate_plan_structure(plan, authority, manifest)
        with accounting.locked_events(contract.ROOT / "submission.jsonl", shared=True) as (events, _):
            account = accounting.replay(events, authority, manifest)
        review_path = prefix.BASE / plan["reviewed_partials"]
        require(review_path.is_file() and not review_path.is_symlink(), "fixed_original_partial_review_required")
        review = prefix.read_json(review_path)
        records = {}
        for work, item in planned.items():
            path = prefix.BASE / item["predecessor_status"]
            require(path.is_file() and not path.is_symlink()
                    and (path.parent / "channels.h5").is_file()
                    and not (path.parent / "channels.h5").is_symlink(), "original_partial_must_remain_preserved")
            records[work] = prefix.read_json(path)
        items = remainders.validate_plan(plan, authority, manifest, review, records, account,
                                         base_directory=str(prefix.BASE))
    require(args.work_item in items, "explicit_diagnosed_remainder_work_required")
    item = items[args.work_item]
    resources = item["resources"]
    allocation = contract.verify_allocation(resources)
    if io_mode:
        job, registered_batch, admitted_policy = admit_io_recovery_allocation(
            authority, manifest, plan, account, item, allocation)
        grouped = False
    else:
        job = remainders.registered_remainder_job(account, plan, args.work_item)
        registered_batch = account["batches"][job["batch_id"]]
        grouped = bundle.FIELD in registered_batch
        admitted_policy = None
        if grouped:
            require(bundle.admit_member(account, item, allocation, os.environ) == job,
                    "exact_grouped_remainder_allocation_required")
        else:
            require(os.environ.get("SLURM_ARRAY_TASK_ID") == str(item["array_index"])
                    and job["job_id"] == f"{os.environ.get('SLURM_ARRAY_JOB_ID')}_{item['array_index']}",
                    "exact_remainder_array_registration_required")
            admitted_policy = contract.admit_registered_allocation(authority, manifest, args.work_item, allocation)
    effective_resources = remainder_runtime_resources(manifest, args.work_item, resources, admitted_policy)
    budget = contract.WorkBudget(allocation, effective_resources, args.max_seconds)
    spans = tuple(contract.parse_span(row) for row in item["spans"])
    output = prefix.BASE / item["output_directory"]
    require(not output.exists() and not output.is_symlink(), "remainder_output_must_be_fresh")
    if io_mode:
        require(output.resolve() == output.absolute(), "exclusive_IO_remainder_path_required")
    metadata = state.admit_participant(item["participant"])
    scratch_root = Path(os.environ.get("L_SCRATCH_JOB", ""))
    require(scratch_root.is_absolute() and scratch_root.is_dir(), "node_local_job_scratch_required")
    scratch = scratch_root / ("ep21-remainder-001-" + args.work_item)
    require(not scratch.exists(), "fresh_remainder_scratch_required")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.parent.is_symlink(), "exclusive_remainder_parent_required")
    output.mkdir(mode=0o700)
    count = item["units_requested"]
    record = {"schema_version": "ep21.decoder_work_item.v1", "status": "admitted_pending",
        "work_item_id": args.work_item, "participant": item["participant"],
        "authorization": str(Path(args.authorization).absolute()), "work_manifest": str(contract.MANIFEST),
        ("io_recovery_review" if io_mode else "remainder_plan"): str(plan_path),
        "predecessor_job_id": item["predecessor_job_id"],
        "predecessor_status": item["predecessor_status"],
        "allocation": allocation, "resources": resources, "max_work_seconds": budget.seconds,
        "training_role": "subject_unique", "training_images": 4712,
        "units_requested": count, "completed_units": 0,
        "requested_spans": item["spans"], "committed_spans": [], "remaining_spans": item["spans"],
        "remaining_interval_status": "exact_uncommitted_coverage", "completed_units_is_lower_bound": False,
        "original_neural_values_read": False, "retained_neural_state_read": False,
        "regular_feature_values_read": False, "predictions_or_scores_computed": False,
        "tau_train_values_read": False, "tau_test_values_read": False, "OOD_access": False,
        "complete_decoder": False, "successor_authorized": False, "automatic_retry": False,
        "original_budget_consumption_preserved": True, "fragment": "channels.h5",
        "scratch_peak_live_bytes": 0, "scratch_bytes_written": 0,
        "scratch_cap_interpretation": "peak_simultaneously_retained_transient_bytes",
        "block_timings": {"normalization_correlation_RHS_and_exact500_solve_seconds": 0.,
                          "select_solve_seconds": 0., "publication_seconds": 0.}, "channel_timings": []}
    if grouped:
        record[bundle.FIELD] = list(bundle.MEMBERS)
    if registered_batch.get(io_quarantine.CLEAN_WORK5_FIELD) is True:
        record[io_quarantine.CLEAN_WORK5_FIELD] = True
    if admitted_policy is not None:
        record.update(effective_scratch_bytes=effective_resources["scratch_bytes"],
                      parallelism_amendment_id=admitted_policy["parallelism_amendment_id"])
    prefix.publish_json(output / "requested_units.json", record, budget)
    bank = container = None
    stores, error, complete = [], None, False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame):
        raise Stop("work_item_deadline_reached" if signum == signal.SIGALRM else "termination_requested")
    for sig in previous:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, budget.deadline-time.monotonic()))
    try:
        scratch.mkdir(mode=0o700)
        os.environ["TMPDIR"] = str(scratch)
        prefix.load_libraries()
        runtime = {"python": sys.version.split()[0], "numpy": prefix.np.__version__, "h5py": prefix.h5py.__version__}
        require(all(runtime[key] == metadata["lock"]["runtime"][key] for key in runtime),
                "locked_decoder_runtime_mismatch")
        record["runtime"] = runtime
        corrmat = prefix.source_corrmat()
        started = time.monotonic()
        record["retained_neural_state_read"] = True
        x, augmented, gram, native, binding = state.load_shared(item["participant"], args.authorization, budget)
        record["shared_state_load_seconds"] = time.monotonic()-started
        record["shared_state_reference"] = str(binding.directory / "shared_training_state.npz")
        record["training_row_binding_reference"] = str(binding.directory / "training_row_binding.json")
        record["training_repetitions"] = binding.training_repetitions
        record["eligible_voxels"] = len(native)
        largest_bank = max(4712*math.prod(span.shape[1:] if len(span.shape) == 3 else span.shape)*4 for span in spans)
        estimate = gram.nbytes+augmented.nbytes+3*x.nbytes+largest_bank+3*1024**3
        record["conservative_memory_planning_bytes"] = estimate
        require(estimate < resources["memory_gib"]*1024**3, "planned_arrays_exceed_memory_cap")
        require(largest_bank <= effective_resources["scratch_bytes"], "work_item_scratch_cap")
        budget.room(count*6045+len(spans)*1_048_576)
        fragment = output / "channels.h5"
        container = prefix.h5py.File(fragment, "x")
        container.attrs.update(schema_version="ep21.decoder_work_item_fragment.v1", participant=item["participant"],
                               work_item_id=args.work_item, complete_decoder=False, units_requested=count)
        for span in spans:
            budget.check()
            detail = {**span.record(), "loading_seconds": 0., "fitting_seconds": 0.}
            record["channel_timings"].append(detail)
            started = time.monotonic()
            record["regular_feature_values_read"] = True
            try:
                if admitted_policy is None:
                    bank = layer.stage_channel(metadata["feature_root"], scratch, span, budget)
                else:
                    bank = layer.stage_channel(metadata["feature_root"], scratch, span, budget,
                                               scratch_bytes=effective_resources["scratch_bytes"])
            finally:
                detail["loading_seconds"] = time.monotonic()-started
                if (scratch / "channel.float32").exists():
                    record["scratch_peak_live_bytes"] = max(record["scratch_peak_live_bytes"],
                                                             (scratch / "channel.float32").stat().st_size)
            record["scratch_bytes_written"] += bank.nbytes
            store = layer.ChannelStore(container, fragment, span, budget)
            stores.append(store)
            started = time.monotonic()
            try:
                original.fit_span(bank, span, x, augmented, gram, native, corrmat, store, budget, record["block_timings"])
            finally:
                detail["fitting_seconds"] = time.monotonic()-started
            require(store.committed_end == span.stop, "complete_requested_channel_interval_required")
            bank._mmap.close()
            bank = None
            (scratch / "channel.float32").unlink()
            print(json.dumps({"work_item_id": args.work_item, "remainder": True, "channel_complete": span.record(),
                              "new_units": sum(value.committed_end-value.span.start for value in stores)}), flush=True)
        complete = sum(value.committed_end-value.span.start for value in stores) == count
        require(complete, "complete_exact_remainder_required")
        record["status"] = "all_units_committed_pending_container_finalization"
    except Exception as exc:
        error = exc
        record.update(status="partial_work_item_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        by_key = {store.span.key: store for store in stores}
        record["committed_spans"], record["remaining_spans"] = [], []
        for span in spans:
            store = by_key.get(span.key)
            cursor = span.start if store is None else store.committed_end
            if cursor > span.start:
                record["committed_spans"].append(contract.coverage.Span(*span.key, span.start, cursor).record())
            if cursor < span.stop:
                record["remaining_spans"].append(contract.coverage.Span(*span.key, cursor, span.stop).record())
            if store is not None and store.uncertain_publication is not None:
                record["uncertain_publication"] = store.uncertain_publication
                record["remaining_interval_status"] = "not_authoritative_for_repair_due_to_uncertain_marker_publication"
                record["completed_units_is_lower_bound"] = True
        record["closeout_errors"] = layer.finalize_files(container, bank, scratch, output / "channels.h5", budget)
        if error is None and any(item["affects_fragment"] for item in record["closeout_errors"]):
            error = Stop("work_item_fragment_finalization_failed")
            record.update(status="all_units_committed_fragment_finalization_failed", error_type=type(error).__name__, error=str(error))
        if complete and error is None:
            record["status"] = "complete_work_item_only"
        record["completed_units"] = sum(store.committed_end-store.span.start for store in stores)
        record["elapsed_seconds"] = time.monotonic()-budget.started
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_feature_bytes_read"] = budget.logical_read_bytes
        record["durable_bytes_before_final_status"] = sum(budget.files.values())
        record["scratch_bytes_written_is_lower_bound"] = error is not None
        record["accounting"] = {"charge_parent_allocation_once": True,
                                "maximum_CPU_seconds": allocation["time_limit_seconds"],
                                "prior_consumption_preserved": True, "terminal_accounting_by_coordinator": True}
        prefix.publish_json(output / "work_item_status.json", record, budget, closing=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest", required=True)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument("--remainder-plan")
    route.add_argument("--io-recovery-review")
    import reviewed_io_recovery as recovery
    parser.add_argument("--work-item", required=True, choices=(*remainders.WORK_IDS_002, *recovery.WORK_IDS))
    parser.add_argument("--max-seconds", type=int)
    try:
        return run(parser.parse_args())
    except Exception as exc:
        print(json.dumps({"status": "remainder_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
