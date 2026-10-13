"""UNACTIVATED EP21 sub-01 conv1_1 channels 1–63 continuation.

One separately approved 1 CPU/7 GiB/24-hour job; no submission or retry here.
Requires a new exact authority before scientific imports or empirical reads.
The consumed channel-0 authority cannot activate this entrypoint.
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

import decoder_coverage as coverage
import train_channel_continuation as prior

prefix = prior.prefix
require, Stop = prefix.require, prefix.Stop
BASE = prefix.BASE
AUTH = BASE / "outputs/decoder_training_layer_authorization.json"
OUTPUT = BASE / "outputs/decoder_training/layer_continuation_001"
RESOURCES = {"jobs": 1, "cpus": 1, "memory_gib": 7, "minutes": 1440, "gpus": 0,
             "durable_bytes": 25_000_000_000, "scratch_bytes": 2_000_000_000}
SPANS = coverage.next_layer_spans()
UNITS = 3_161_088
MAX_SECONDS = 86100
AUTH_EXPECTED = {
    "schema_version": "ep21.decoder_layer_authorization.v1", "episode_id": "ep21",
    "purpose": "retained_sub01_conv1_1_channels1_through63", "approved": True,
    "participant": "sub-01", "training_role": "subject_unique", "training_images": 4712,
    "layer": "conv1_1", "channel_interval": [1, 64], "spatial_Fortran_interval": [0, 50176],
    "units_requested": UNITS, "numerical_block_units": 64, "image_batch": 64,
    "resources": RESOURCES, "internal_seconds": MAX_SECONDS, "reuse_completed_units": 51328,
    "scratch_cap_interpretation": "peak_simultaneously_retained_transient_bytes",
    "reuse_shared_state": "outputs/decoder_training/retained_prefix_001/shared_training_state.npz",
    "reuse_protocol_lock": "outputs/protocol_lock.yaml",
    "retained_reports": ["outputs/decoder_training/retained_prefix_001/prefix_status.json",
                         "outputs/decoder_training/continuation_001/continuation_status.json"],
    "output": "outputs/decoder_training/layer_continuation_001",
    "allow_retained_neural_state": True, "allow_original_neural_values": False,
    "allow_regular_feature_values": True, "allow_decoder_fitting": True,
    "feature_inputs": [str((prefix.FEATURES / name).relative_to(BASE)) for name in prefix.FEATURE_FILES],
    "feature_value_scope": "sub01_conv1_1_channels1_through63_planes_only",
    "allow_tau_train_values": False, "allow_tau_test_values": False, "allow_ood": False,
    "allow_prediction_or_scoring": False, "allow_converters": False,
    "allow_reconstruction": False, "allow_external_sites": False,
    "automatic_retry": False, "automatic_extension": False, "successor_authorized": False,
    "original_budget_consumption_preserved": True,
}


def authorize(path):
    require(Path(path).absolute() == AUTH and not AUTH.is_symlink(), "fixed_layer_authorization_path_required")
    record = prefix.read_json(AUTH)
    require(all(json.dumps(record.get(k), sort_keys=True) == json.dumps(v, sort_keys=True)
                for k, v in AUTH_EXPECTED.items()), "explicit_new_layer_authorization_required")
    return record


def admit_completed(ids):
    groups = prior.admit_prefix(ids)
    record = prefix.read_json(prior.OUTPUT / "continuation_status.json")
    expected = {"schema_version": "ep21.decoder_continuation.v1",
                "status": "complete_continuation_tranche_only", "participant": "sub-01",
                "training_role": "subject_unique", "training_images": 4712,
                "training_repetitions": 18843, "layer": "conv1_1", "channel": 0,
                "units_requested": 50112, "completed_units": 50112,
                "requested_spatial_Fortran_interval": [64, 50176],
                "committed_spatial_Fortran_interval": [64, 50176],
                "remaining_interval_status": "exact_uncommitted_coverage",
                "completed_units_is_lower_bound": False, "fragment": "conv1_1_channel0.h5",
                "retained_prefix_completed_units": 1216, "complete_decoder": False,
                "predictions_or_scores_computed": False, "tau_train_values_read": False,
                "tau_test_values_read": False, "OOD_access": False, "successor_authorized": False}
    require(all(record.get(key) == value for key, value in expected.items())
            and "uncertain_publication" not in record
            and record.get("shared_state_reference") == str(prior.PRIOR / "shared_training_state.npz"),
            "completed_channel0_exact_coverage_required")
    require(all(path.is_file() and not path.is_symlink() for path in
                (prior.PRIOR / "conv1_1.h5", prior.OUTPUT / "conv1_1_channel0.h5")),
            "retained_layer_fragment_references_must_exist")
    require(sum(span.count for span in SPANS) == UNITS, "next_layer_unit_count_required")
    retained = [span for span in coverage.retained_spans() if span.layer == "conv1_1"]
    require(coverage.verify_complete_coverage([*retained, *SPANS], participants=("sub-01",),
                                             layers=("conv1_1",)) == 3_211_264,
            "retained_plus_new_complete_layer_required")
    return groups


class Budget(prior.Budget):
    def room(self, extra):
        require(sum(self.files.values()) + extra <= RESOURCES["durable_bytes"], "new_layer_durable_byte_cap")


def stage_channel(feature_root, scratch, span, budget, *, scratch_bytes=None):
    """A single channel bank; finite-mask offsets use C order, targets use F order.

    Full workers pass their admitted effective cap explicitly. Legacy callers
    retain the original 2-GB bound. No new disk cache or second bank is created.
    """
    np, h5py = prefix.np, prefix.h5py
    shape = span.shape
    size = math.prod(shape[1:]) if len(shape) == 3 else shape[0]
    amount = prefix.IMAGES * size * 4
    cap = RESOURCES["scratch_bytes"] if scratch_bytes is None else scratch_bytes
    require(type(cap) is int and 0 < cap <= RESOURCES["scratch_bytes"], "explicit_bounded_channel_scratch_cap")
    require(amount <= cap, "channel_scratch_cap")
    free = os.statvfs(scratch)
    require(free.f_bavail * free.f_frsize >= amount + 1_048_576, "node_local_capacity_required")
    path = scratch / "channel.float32"
    require(not path.exists(), "fresh_channel_bank_required")
    bank = np.memmap(path, dtype="<f4", mode="w+", shape=(prefix.IMAGES, size))
    try:
        with h5py.File(feature_root / "features.h5", "r") as source, \
                h5py.File(feature_root / "finite_packed.h5", "r") as finite_source:
            data, finite = source[span.layer], finite_source[span.layer]
            require(data.shape == (prefix.IMAGES,) + shape and data.dtype.str == "<f4"
                    and data.chunks is None and data.compression is None
                    and finite.shape == (prefix.IMAGES, (math.prod(shape) + 7) // 8)
                    and finite.dtype.str == "|u1", "canonical_channel_headers_required")
            bit_start, bit_stop = span.channel * size, (span.channel + 1) * size
            first_byte, last_byte = bit_start // 8, (bit_stop + 7) // 8
            for first in range(0, prefix.IMAGES, 64):
                budget.check()
                last = min(first + 64, prefix.IMAGES)
                packed = finite[first:last, first_byte:last_byte]
                valid = np.unpackbits(packed, axis=1, bitorder="little")[:, bit_start % 8:bit_start % 8 + size]
                if len(shape) == 3:
                    plane = data[first:last, span.channel, :, :]
                    arranged = plane.transpose(0, 2, 1).reshape(last - first, size)
                    valid_f = valid.reshape((last - first,) + shape[1:]).transpose(0, 2, 1).reshape(last - first, size)
                else:
                    plane = data[first:last, :]
                    arranged, valid_f = plane, valid
                bad = np.argwhere((valid_f[:, span.start:span.stop] == 0)
                                  | ~np.isfinite(arranged[:, span.start:span.stop]))
                if bad.size:
                    image, offset = map(int, bad[0])
                    raise Stop(f"nonfinite_required_target:{span.key}:image={first + image}:spatial_F={span.start + offset}")
                bank[first:last] = arranged
                budget.logical_read_bytes += plane.nbytes + packed.nbytes
                del plane, arranged, packed, valid, valid_f, bad
        bank.flush()
        return bank
    except BaseException:
        bank._mmap.close()
        raise


class ChannelStore:
    """One group in one fresh work-item container; the parent owns its lifetime."""
    def __init__(self, container, path, span, budget):
        np = prefix.np
        span.validate()
        budget.room(span.count * 6045 + 1_048_576)
        self.container, self.path, self.span, self.budget = container, path, span, budget
        self.committed_end = span.start
        self.uncertain_publication = None
        self.file = container.create_group(f"{span.layer}/channel_{span.channel:03d}")
        out, count = self.file, span.count
        out.attrs.update(participant=span.participant, layer=span.layer, channel=span.channel,
                         spatial_Fortran_start=span.start, spatial_Fortran_stop=span.stop,
                         complete_decoder=False, state_codes="0=incomplete,2=durably_complete",
                         source_fragment_axes="unit_by_ranked_selected_native_voxel; bias_separate")
        out.create_dataset("state", (count,), dtype="u1", fillvalue=0)
        out.create_dataset("selected_native_voxel_ids", (count, 500), dtype="<i8", fillvalue=-1)
        out.create_dataset("weights", (count, 500), dtype="<f4", fillvalue=np.nan)
        for name in ("bias", "target_mean", "target_std"):
            out.create_dataset(name, (count,), dtype="<f4", fillvalue=np.nan)
        out.create_dataset("solve_seconds", (count,), dtype="<f8", fillvalue=np.nan)
        spatial = np.arange(span.start, span.stop, dtype=np.int64)
        out.create_dataset("spatial_Fortran_index", data=spatial)
        if len(span.shape) == 3:
            channels, height, width = span.shape
            full_f = span.channel + channels * spatial
            full_c = span.channel * height * width + (spatial % height) * width + spatial // height
        else:
            full_f, full_c = spatial, spatial
        out.create_dataset("full_layer_Fortran_index", data=full_f)
        out.create_dataset("full_layer_C_index", data=full_c)
        prefix.fsync_h5(container)
        budget.note(path)

    def commit(self, begin, rows):
        np, out = prefix.np, self.file
        require(begin == self.committed_end and rows and begin + len(rows) <= self.span.stop,
                "disjoint_ascending_new_units_required")
        start, stop = begin - self.span.start, begin - self.span.start + len(rows)
        require(bool((out["state"][start:stop] == 0).all()), "completed_rows_cannot_replay")
        for ids, weights, bias, mean, std, elapsed in rows:
            require(ids.shape == weights.shape == (500,) and len(np.unique(ids)) == 500
                    and np.isfinite(weights).all() and np.isfinite([bias, mean, std, elapsed]).all()
                    and std > 0 and elapsed >= 0, "valid_solved_row_required")
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
        try:
            out["selected_native_voxel_ids"][start:stop] = np.stack([r[0] for r in rows])
            out["weights"][start:stop] = np.stack([r[1] for r in rows])
            for index, name in enumerate(("bias", "target_mean", "target_std", "solve_seconds"), 2):
                out[name][start:stop] = [r[index] for r in rows]
            prefix.fsync_h5(self.container)
            phase = "completion_marker_assignment"
            try:
                out["state"][start:stop] = 2
                phase = "completion_marker_fsync"
                prefix.fsync_h5(self.container)
            except Exception as exc:
                self.uncertain_publication = {**self.span.record(),
                    "spatial_Fortran_interval": [begin, begin + len(rows)],
                    "phase": phase, "error_type": type(exc).__name__}
                raise
            self.committed_end = begin + len(rows)
            self.budget.note(self.path)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


def fit_blocks(bank, span, x, augmented, gram, native, corrmat, store, budget, timing):
    np = prefix.np
    for begin in range(span.start, span.stop, 64):
        end = min(begin + 64, span.stop)
        budget.check()
        started = time.monotonic()
        targets = np.ascontiguousarray(bank[:, begin:end], dtype=np.float32)
        try:
            prepared = prefix.prepare_layer_math(x, augmented, gram, targets, corrmat)
        finally:
            timing["normalization_correlation_RHS_seconds"] += time.monotonic() - started
        del targets
        if "bad_units" in prepared:
            raise Stop(prepared["error"] + f":{span.key}:spatial_F="
                       + ",".join(str(begin + i) for i in prepared["bad_units"]))
        require(prepared["branch"] == "source_greater_than_500_with_native_ID_ties",
                "retained_sub01_source_branch_required")
        rows = []
        try:
            for index in range(end - begin):
                budget.check()
                started = time.monotonic()
                try:
                    selected, weights, bias = prefix.solve_selected(
                        gram, prepared["right"][index], prepared["correlations"][index], native)
                finally:
                    elapsed = time.monotonic() - started
                    timing["select_solve_seconds"] += elapsed
                rows.append((native[selected], weights, bias, prepared["means"][0, index],
                             prepared["scales"][0, index], elapsed))
        finally:
            if rows:
                started = time.monotonic()
                try:
                    store.commit(begin, rows)
                finally:
                    timing["publication_seconds"] += time.monotonic() - started


def finalize_files(container, bank, scratch, fragment_path, budget):
    """Try every ordinary closure action; preserve evidence if one fails.

    Fragment failures prevent a new complete-layer manifest. Reconstructible
    scratch cleanup failures are recorded separately from fitted-row validity.
    """
    errors = []
    def attempt(stage, action, affects_fragment):
        try:
            action()
        except Exception as exc:
            errors.append({"stage": stage, "affects_fragment": affects_fragment,
                           "error_type": type(exc).__name__, "error": str(exc)})
    if container is not None:
        attempt("fragment_fsync", lambda: prefix.fsync_h5(container), True)
        attempt("fragment_close", container.close, True)
        attempt("fragment_accounting", lambda: budget.note(fragment_path), True)
    if bank is not None:
        attempt("channel_bank_close", bank._mmap.close, False)
    attempt("channel_bank_unlink", lambda: (scratch / "channel.float32").unlink(missing_ok=True), False)
    def remove_scratch():
        if scratch.exists():
            scratch.rmdir()
    attempt("task_scratch_rmdir", remove_scratch, False)
    return errors


def run(args):
    require(Path.cwd().resolve() == BASE and 0 < args.max_seconds <= MAX_SECONDS,
            "exact_episode_and_bounded_seconds_required")
    authorize(args.authorization)
    allocation = prior.verify_allocation()
    require(not OUTPUT.exists(), "existing_attempt_no_replay_or_overwrite")
    budget = Budget(allocation, args.max_seconds)
    lock, selected, _, feature_root, feature_admission = prefix.admit_metadata()
    ids = prefix.feature_index(feature_root / "image_index.tsv", selected)
    groups = admit_completed(ids)
    scratch_root = Path(os.environ.get("L_SCRATCH_JOB", ""))
    require(scratch_root.is_absolute() and scratch_root.is_dir(), "node_local_job_scratch_required")
    scratch = scratch_root / "ep21-decoder-layer-continuation-001"
    require(not scratch.exists(), "fresh_node_local_scratch_required")
    OUTPUT.mkdir(mode=0o700)
    record = {"schema_version": "ep21.decoder_layer_continuation.v1", "status": "admitted_pending",
              "authorization": str(AUTH), "allocation": allocation, "resources": RESOURCES,
              "max_work_seconds": budget.seconds, "participant": "sub-01", "training_role": "subject_unique",
              "training_images": 4712, "training_repetitions": 18843, "layer": "conv1_1",
              "units_requested": UNITS, "completed_units": 0, "retained_completed_units": 51328,
              "requested_spans": [span.record() for span in SPANS], "committed_spans": [],
              "remaining_spans": [span.record() for span in SPANS],
              "remaining_interval_status": "exact_uncommitted_coverage", "completed_units_is_lower_bound": False,
              "shared_state_reference": str(prior.PRIOR / "shared_training_state.npz"),
              "feature_admission": feature_admission, "original_neural_values_read": False,
              "retained_neural_state_read": False, "regular_feature_values_read": False,
              "predictions_or_scores_computed": False, "tau_train_values_read": False,
              "tau_test_values_read": False, "OOD_access": False, "complete_decoder": False,
              "complete_layer": False, "successor_authorized": False, "automatic_retry": False,
              "original_budget_consumption_preserved": True, "fragment": "channels.h5",
              "scratch_peak_live_bytes": 0, "scratch_bytes_written": 0,
              "scratch_cap_interpretation": "peak_simultaneously_retained_transient_bytes",
              "block_timings": {"normalization_correlation_RHS_seconds": 0., "select_solve_seconds": 0.,
                                "publication_seconds": 0.}, "channel_timings": []}
    prefix.publish_json(OUTPUT / "requested_units.json", record, budget)
    bank = container = None
    stores, error, pending_manifest = [], None, None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame):
        raise Stop("layer_deadline_reached" if signum == signal.SIGALRM else "termination_requested")
    for sig in handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, budget.deadline - time.monotonic()))
    try:
        scratch.mkdir(mode=0o700)
        os.environ["TMPDIR"] = str(scratch)
        prefix.load_libraries()
        runtime = {"python": sys.version.split()[0], "numpy": prefix.np.__version__, "h5py": prefix.h5py.__version__}
        require(all(runtime[k] == lock["runtime"][k] for k in runtime), "locked_decoder_runtime_mismatch")
        record["runtime"] = runtime
        corrmat = prefix.source_corrmat()
        started = time.monotonic()
        record["retained_neural_state_read"] = True
        x, augmented, gram, native = prior.load_shared(ids, groups, budget)
        record["shared_state_load_seconds"] = time.monotonic() - started
        estimate = gram.nbytes + augmented.nbytes + 3 * x.nbytes + prior.BANK_BYTES + 3 * 1024**3
        record["conservative_memory_planning_bytes"] = estimate
        require(estimate < RESOURCES["memory_gib"] * 1024**3, "planned_arrays_exceed_memory_cap")
        budget.room(UNITS * 6045 + 63 * 1_048_576)
        fragment_path = OUTPUT / "channels.h5"
        container = prefix.h5py.File(fragment_path, "x")
        container.attrs.update(schema_version="ep21.decoder_channel_groups.v1", participant="sub-01",
                               complete_decoder=False, units_requested=UNITS)
        for span in SPANS:
            budget.check()
            detail = {**span.record(), "loading_seconds": 0., "fitting_seconds": 0.}
            record["channel_timings"].append(detail)
            started = time.monotonic()
            record["regular_feature_values_read"] = True
            try:
                bank = stage_channel(feature_root, scratch, span, budget)
            finally:
                detail["loading_seconds"] = time.monotonic() - started
                # A partial bank's logical size bounds its live occupancy;
                # it does not prove that its full contents were written.
                if (scratch / "channel.float32").exists():
                    record["scratch_peak_live_bytes"] = max(record["scratch_peak_live_bytes"],
                                                           (scratch / "channel.float32").stat().st_size)
            record["scratch_bytes_written"] += bank.nbytes
            store = ChannelStore(container, fragment_path, span, budget)
            stores.append(store)
            started = time.monotonic()
            try:
                fit_blocks(bank, span, x, augmented, gram, native, corrmat, store, budget, record["block_timings"])
            finally:
                detail["fitting_seconds"] = time.monotonic() - started
            require(store.committed_end == span.stop, "complete_channel_required")
            bank._mmap.close()
            bank = None
            (scratch / "channel.float32").unlink()
            # This is execution progress, never a predictive score.
            print(json.dumps({"channel_complete": span.channel, "new_units": sum(s.committed_end - s.span.start for s in stores)}), flush=True)
        require(sum(store.committed_end - store.span.start for store in stores) == UNITS,
                "complete_fixed_layer_tranche_required")
        pending_manifest = {"schema_version": "ep21.complete_layer_manifest.v1", "participant": "sub-01",
                    "layer": "conv1_1", "shape": [64, 224, 224], "complete_layer": True,
                    "complete_decoder": False, "units": 3211264,
                    "shared_state": record["shared_state_reference"],
                    "fragments": [
                        {"path": str(prior.PRIOR / "conv1_1.h5"), "format": "retained_prefix",
                         "channel": 0, "spatial_Fortran_interval": [0, 64]},
                        {"path": str(prior.OUTPUT / "conv1_1_channel0.h5"), "format": "continuation_fragment",
                         "channel": 0, "spatial_Fortran_interval": [64, 50176]},
                        *[{"path": str(fragment_path), "group": f"conv1_1/channel_{span.channel:03d}",
                           **span.record()} for span in SPANS]],
                    "source_fragment_axes": "unit_by_ranked_selected_native_voxel; bias_separate",
                    "training_role": "subject_unique", "training_images": 4712,
                    "prediction_or_converter_authorized": False}
        record["status"] = "all_units_committed_pending_container_finalization"
    except Exception as exc:
        error = exc
        record.update(status="partial_layer_continuation_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        by_key = {store.span.key: store for store in stores}
        record["committed_spans"], record["remaining_spans"] = [], []
        for span in SPANS:
            store = by_key.get(span.key)
            cursor = span.start if store is None else store.committed_end
            if cursor > span.start:
                record["committed_spans"].append(coverage.Span(*span.key, span.start, cursor).record())
            if cursor < span.stop:
                record["remaining_spans"].append(coverage.Span(*span.key, cursor, span.stop).record())
            if store is not None and store.uncertain_publication is not None:
                record["uncertain_publication"] = store.uncertain_publication
                record["remaining_interval_status"] = "not_authoritative_for_repair_due_to_uncertain_marker_publication"
                record["completed_units_is_lower_bound"] = True
        record["closeout_errors"] = finalize_files(container, bank, scratch, OUTPUT / "channels.h5", budget)
        if error is None and any(item["affects_fragment"] for item in record["closeout_errors"]):
            error = Stop("fragment_finalization_failed")
            record.update(status="all_units_committed_fragment_finalization_failed",
                          error_type=type(error).__name__, error=str(error))
        if pending_manifest is not None and error is None:
            try:
                prefix.publish_json(OUTPUT / "complete_layer_manifest.json", pending_manifest, budget, closing=True)
                record.update(status="complete_layer_continuation_only", complete_layer=True)
            except Exception as exc:
                error = exc
                record.update(status="all_units_committed_layer_manifest_publication_failed",
                              error_type=type(exc).__name__, error=str(exc))
        record["completed_units"] = sum(store.committed_end - store.span.start for store in stores)
        record["elapsed_seconds"] = time.monotonic() - budget.started
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_feature_bytes_read"] = budget.logical_read_bytes
        record["durable_bytes_before_final_status"] = sum(budget.files.values())
        record["scratch_bytes_written_is_lower_bound"] = error is not None
        record["channel_bank_bytes"] = prior.BANK_BYTES
        record["accounting"] = {"charge_parent_allocation_once": True, "maximum_CPU_hours": 24,
                                "prior_consumption_preserved": True, "terminal_accounting_by_coordinator": True}
        prefix.publish_json(OUTPUT / "continuation_status.json", record, budget, closing=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "layer_continuation_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
