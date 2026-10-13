"""Guarded full-EP21 worker: one exact authorized manifest assignment, no dispatch.

All five participants, all 19 layers and true convolutional/FC tails are
supported. A distinct full-stage approval and complete work manifest must be
admitted before numerical imports or payload reads. This neither builds state
implicitly nor enables predictions, converters, evaluation or another job.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import decoder_execution_contract as contract
import participant_training_state as state

layer, prefix = contract.layer, contract.prefix
require, Stop = prefix.require, prefix.Stop
WorkBudget, verify_allocation = contract.WorkBudget, contract.verify_allocation
SCRATCH_OVERHEAD_BYTES = 64 * 1024**2


def largest_channel_bank_bytes(spans):
    """One 4,712-image float32 channel, including unrequested spatial positions."""
    require(bool(spans), "explicit_work_item_spans_required")
    return max(4712 * math.prod(span.shape[1:] if len(span.shape) == 3 else span.shape) * 4
               for span in spans)


def runtime_resources(resources, spans, admitted_policy):
    """Apply only the accepted batch's additive scratch policy; never edit manifest.

    Legacy admissions return None. Amended batches reserve one largest channel
    bank plus 64 MiB, without changing CPU, memory, walltime or durable output.
    """
    effective = dict(contract.validate_resources(resources))
    if admitted_policy is None:
        return effective
    require(isinstance(admitted_policy, dict)
            and set(admitted_policy) == {"effective_scratch_bytes", "parallelism_amendment_id"}
            and admitted_policy["parallelism_amendment_id"] == "parallelism_amendment",
            "accepted_parallelism_scratch_policy_required")
    cap = admitted_policy["effective_scratch_bytes"]
    expected = largest_channel_bank_bytes(spans) + SCRATCH_OVERHEAD_BYTES
    require(type(cap) is int and cap == expected and cap <= resources["scratch_bytes"],
            "exact_assignment_bank_plus_overhead_scratch_cap_required")
    effective["scratch_bytes"] = cap
    return effective


def fit_span(bank, span, x, augmented, gram, native, corrmat, store, budget, timing):
    """Unchanged float32 kernels, both frozen source branches, genuine tails."""
    np = prefix.np
    require(len(native) >= 500, "fewer_than_500_voxels_not_evaluable")
    for begin in range(span.start, span.stop, 64):
        end = min(begin + 64, span.stop)
        budget.check()
        targets = np.ascontiguousarray(bank[:, begin:end], dtype=np.float32)
        started = time.monotonic()
        try:
            prepared = prefix.prepare_layer_math(x, augmented, gram, targets, corrmat)
        finally:
            timing["normalization_correlation_RHS_and_exact500_solve_seconds"] += time.monotonic() - started
        del targets
        if "bad_units" in prepared:
            raise Stop(prepared["error"] + f":{span.key}:spatial_F="
                       + ",".join(str(begin + index) for index in prepared["bad_units"]))
        expected = ("source_exactly_500_native_order_no_selection" if len(native) == 500
                    else "source_greater_than_500_with_native_ID_ties")
        require(prepared["branch"] == expected, "participant_source_branch_mismatch")
        rows = []
        try:
            for index in range(end - begin):
                budget.check()
                if len(native) == 500:
                    selected_ids = native
                    weights, bias = prepared["solved"][:500, index], prepared["solved"][500, index]
                    elapsed = 0.  # The block solve is timed above, not charged twice.
                else:
                    started = time.monotonic()
                    try:
                        selected, weights, bias = prefix.solve_selected(
                            gram, prepared["right"][index], prepared["correlations"][index], native)
                    finally:
                        elapsed = time.monotonic() - started
                        timing["select_solve_seconds"] += elapsed
                    selected_ids = native[selected]
                rows.append((selected_ids, weights, bias, prepared["means"][0, index],
                             prepared["scales"][0, index], elapsed))
        finally:
            if rows:
                started = time.monotonic()
                try:
                    store.commit(begin, rows)
                finally:
                    timing["publication_seconds"] += time.monotonic() - started
        del prepared, rows


def fit_span_with_publication_policy(bank, span, x, augmented, gram, native, corrmat,
                                     store, budget, timing, *, publication_blocks=1):
    """Group writes only; retain the exact 64-column numerical fit sequence.

    The original ChannelStore remains the authority for durable coverage and
    uncertain markers. The optional buffer owns no persistent cursor. Ordinary
    numerical/deadline exceptions flush solved rows; publication failures are
    latched and never retried by the final flush.
    """
    require(type(publication_blocks) is int and publication_blocks in (1, 4),
            "explicit_supported_publication_block_count_required")
    if publication_blocks == 1:
        return fit_span(bank, span, x, augmented, gram, native, corrmat, store, budget, timing)
    from buffered_publication import BufferedPublication
    buffered = BufferedPublication(store)
    try:
        return fit_span(bank, span, x, augmented, gram, native, corrmat, buffered, budget, timing)
    finally:
        started = time.monotonic()
        try:
            buffered.flush()
        finally:
            timing["publication_seconds"] += time.monotonic() - started


def run(args):
    require(Path.cwd().resolve() == prefix.BASE, "exact_episode_working_directory_required")
    authority = state.authorize_full(args.authorization)
    publication_blocks = getattr(args, "publication_blocks", 1)
    require(type(publication_blocks) is int and publication_blocks in (1, 4),
            "explicit_supported_publication_block_count_required")
    require(Path(args.manifest).absolute() == contract.MANIFEST and not contract.MANIFEST.is_symlink(),
            "fixed_full_stage_work_manifest_required")
    manifest = prefix.read_json(contract.MANIFEST)
    assignments = contract.validate_work_manifest(manifest, authority)
    require(args.work_item in assignments, "explicit_authorized_work_item_required")
    item = assignments[args.work_item]
    spans = tuple(contract.parse_span(row) for row in item["spans"])
    resources = item["resources"]
    allocation = verify_allocation(resources)
    admitted_policy = contract.admit_registered_allocation(authority, manifest, args.work_item, allocation)
    effective_resources = runtime_resources(resources, spans, admitted_policy)
    budget = WorkBudget(allocation, effective_resources, args.max_seconds)
    output = contract.ROOT / "shards" / args.work_item
    require(not output.exists() and not output.is_symlink(), "existing_work_item_cannot_replay")
    metadata = state.admit_participant(item["participant"])
    scratch_root = Path(os.environ.get("L_SCRATCH_JOB", ""))
    require(scratch_root.is_absolute() and scratch_root.is_dir(), "node_local_job_scratch_required")
    scratch = scratch_root / ("ep21-decoder-" + args.work_item)
    require(not scratch.exists(), "fresh_work_item_scratch_required")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Exclusive ownership before loading any fitted arrays.
    count = sum(span.count for span in spans)
    record = {"schema_version": "ep21.decoder_work_item.v1", "status": "admitted_pending",
              "work_item_id": args.work_item, "participant": item["participant"],
              "authorization": str(Path(args.authorization).absolute()), "work_manifest": str(contract.MANIFEST),
              "allocation": allocation, "resources": resources, "max_work_seconds": budget.seconds,
              "numerical_block_units": 64, "publication_blocks": publication_blocks,
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
                                "select_solve_seconds": 0., "publication_seconds": 0.},
              "channel_timings": []}
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
    signal.setitimer(signal.ITIMER_REAL, max(.001, budget.deadline - time.monotonic()))
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
        record["shared_state_load_seconds"] = time.monotonic() - started
        record["shared_state_reference"] = str(binding.directory / "shared_training_state.npz")
        record["training_row_binding_reference"] = str(binding.directory / "training_row_binding.json")
        record["training_repetitions"] = binding.training_repetitions
        record["eligible_voxels"] = len(native)
        largest_bank = largest_channel_bank_bytes(spans)
        publication_extra = 8 * 1024**2 if publication_blocks == 4 else 0
        estimate = gram.nbytes + augmented.nbytes + 3 * x.nbytes + largest_bank + 3 * 1024**3 + publication_extra
        record["publication_buffer_memory_allowance_bytes"] = publication_extra
        record["conservative_memory_planning_bytes"] = estimate
        require(estimate < resources["memory_gib"] * 1024**3, "planned_arrays_exceed_memory_cap")
        require(largest_bank <= effective_resources["scratch_bytes"], "work_item_scratch_cap")
        budget.room(count * 6045 + len(spans) * 1_048_576)
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
                # The only large transient writer is this one channel mmap.
                # State, target math and publication buffers stay in memory;
                # fitted HDF5 and JSON records are written to durable output.
                if admitted_policy is None:
                    bank = layer.stage_channel(metadata["feature_root"], scratch, span, budget)
                else:
                    bank = layer.stage_channel(metadata["feature_root"], scratch, span, budget,
                                               scratch_bytes=effective_resources["scratch_bytes"])
            finally:
                detail["loading_seconds"] = time.monotonic() - started
                if (scratch / "channel.float32").exists():
                    record["scratch_peak_live_bytes"] = max(record["scratch_peak_live_bytes"],
                                                           (scratch / "channel.float32").stat().st_size)
            record["scratch_bytes_written"] += bank.nbytes
            store = layer.ChannelStore(container, fragment, span, budget)
            stores.append(store)
            started = time.monotonic()
            try:
                fit_span_with_publication_policy(bank, span, x, augmented, gram, native, corrmat,
                    store, budget, record["block_timings"], publication_blocks=publication_blocks)
            finally:
                detail["fitting_seconds"] = time.monotonic() - started
            require(store.committed_end == span.stop, "complete_requested_channel_interval_required")
            bank._mmap.close()
            bank = None
            (scratch / "channel.float32").unlink()
            print(json.dumps({"work_item_id": args.work_item, "channel_complete": span.record(),
                              "new_units": sum(s.committed_end - s.span.start for s in stores)}), flush=True)
        complete = sum(store.committed_end - store.span.start for store in stores) == count
        require(complete, "complete_fixed_work_item_required")
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
            record.update(status="all_units_committed_fragment_finalization_failed",
                          error_type=type(error).__name__, error=str(error))
        if complete and error is None:
            record["status"] = "complete_work_item_only"
        record["completed_units"] = sum(store.committed_end - store.span.start for store in stores)
        record["elapsed_seconds"] = time.monotonic() - budget.started
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_feature_bytes_read"] = budget.logical_read_bytes
        record["durable_bytes_before_final_status"] = sum(budget.files.values())
        record["scratch_bytes_written_is_lower_bound"] = error is not None
        record["accounting"] = {"charge_parent_allocation_once": True,
                                "maximum_CPU_seconds": resources["minutes"] * 60,
                                "prior_consumption_preserved": True, "terminal_accounting_by_coordinator": True}
        prefix.publish_json(output / "work_item_status.json", record, budget, closing=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--work-item", required=True)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--publication-blocks", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "work_item_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
