"""Approved fixed EP21 continuation; exact separate authority is required.

Reuses immutable retained_prefix_001 state and source math. Fits only sub-01,
conv1_1, channel 0, spatial Fortran [64, 50176). No prediction or scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import zipfile

import train_retained_prefix as prefix

BASE = prefix.BASE
PRIOR = prefix.OUTPUT
OUTPUT = BASE / "outputs/decoder_training/continuation_001"
AUTH = BASE / "outputs/decoder_training_continuation_authorization.json"
START, END, BLOCK, IMAGE_BATCH = 64, 50176, 64, 64
HEIGHT, WIDTH, CHANNELS = 224, 224, 64
NATIVE, ELIGIBLE = 12295, 12294
BANK_BYTES = prefix.IMAGES * END * 4
ALLOCATION_SECONDS = 86400
MAX_SECONDS = 86100
RESOURCES = {"jobs": 1, "cpus": 1, "memory_gib": 7, "minutes": 1440, "gpus": 0,
             "durable_bytes": 1_000_000_000, "scratch_bytes": 2_000_000_000}
AUTH_EXPECTED = {
    "schema_version": "ep21.decoder_continuation_authorization.v1", "episode_id": "ep21",
    "purpose": "retained_sub01_conv1_1_channel0_continuation_001", "approved": True,
    "participant": "sub-01", "training_role": "subject_unique", "training_images": 4712,
    "layer": "conv1_1", "channel": 0, "spatial_Fortran_interval": [START, END],
    "numerical_block_units": BLOCK, "image_batch": IMAGE_BATCH, "resources": RESOURCES,
    "internal_seconds": MAX_SECONDS, "reuse_completed_prefix_units": 1216,
    "reuse_shared_state": "outputs/decoder_training/retained_prefix_001/shared_training_state.npz",
    "reuse_protocol_lock": "outputs/protocol_lock.yaml",
    "output": "outputs/decoder_training/continuation_001",
    "allow_retained_neural_state": True, "allow_original_neural_values": False,
    "allow_regular_feature_values": True, "allow_decoder_fitting": True,
    "feature_inputs": [str((prefix.FEATURES / n).relative_to(BASE)) for n in prefix.FEATURE_FILES],
    "feature_value_scope": "sub-01_conv1_1_channel_0_planes_only",
    "allow_tau_train_values": False, "allow_tau_test_values": False, "allow_ood": False,
    "allow_prediction_or_scoring": False, "allow_converters": False,
    "allow_reconstruction": False, "allow_external_sites": False,
    "automatic_retry": False, "automatic_extension": False, "successor_authorized": False,
    "original_budget_consumption_preserved": True,
}
require, Stop = prefix.require, prefix.Stop


def authorize(path):
    require(Path(path).absolute() == AUTH and not AUTH.is_symlink(), "fixed_authorization_path_required")
    record = prefix.read_json(AUTH)
    # JSON comparison also rejects 1 in place of true and booleans as resource counts.
    require(all(json.dumps(record.get(k), sort_keys=True) == json.dumps(v, sort_keys=True)
                for k, v in AUTH_EXPECTED.items()), "explicit_continuation_authorization_required")
    return record


def verify_allocation():
    """The newly approved 24-hour allocation; the consumed prefix stays 2-hour."""
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and "-ln" not in host, "actual_compute_allocation_required")
    require(all(os.environ.get(name) == "1" for name in prefix.THREADS), "six_thread_caps_must_equal_one")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168", "one_CPU_7GiB_task_required")
    result = subprocess.run(["scontrol", "show", "job", job, "-o"], check=True,
                            capture_output=True, text=True, timeout=30)
    fields = dict(token.split("=", 1) for token in result.stdout.split() if "=" in token)
    require(fields.get("JobId") == job and fields.get("JobState") == "RUNNING"
            and fields.get("NumCPUs") == "1" and fields.get("NumNodes") == "1"
            and fields.get("NodeList") == host, "actual_single_CPU_node_binding_required")
    tres = dict(token.split("=", 1) for token in fields.get("AllocTRES", "").split(",") if "=" in token)
    require(tres.get("cpu") == "1" and tres.get("mem") in ("7G", "7168M")
            and not any(key.startswith("gres/gpu") for key in tres), "actual_allocation_resource_mismatch")
    limit = prefix.seconds(fields["TimeLimit"])
    require(0 < limit <= ALLOCATION_SECONDS, "24_hour_maximum_allocation_required")
    end = int(os.environ.get("SLURM_JOB_END_TIME", "0"))
    require(end > time.time() + 300, "allocation_closure_reserve_required")
    return {"job_id": job, "hostname": host, "allocated_cpus": 1,
            "memory_gib": 7, "gpus": 0, "time_limit_seconds": limit,
            "job_end_epoch": end}


def admit_prefix(ids):
    record = prefix.read_json(PRIOR / "prefix_status.json")
    require(record.get("schema_version") == "ep21.retained_decoder_prefix.v1"
            and record.get("status") == "complete_retained_prefix_only"
            and record.get("participant") == "sub-01" and record.get("training_role") == "subject_unique"
            and record.get("training_images") == 4712 and record.get("training_repetitions") == 18843
            and record.get("units_requested") == record.get("completed_units") == 1216
            and record.get("eligible_voxels") == ELIGIBLE
            and record.get("shared_state") == "shared_training_state.npz"
            and record.get("feature_admission", {}).get("mode") == "approved_complete_regular_export",
            "completed_fixed_prefix_required")
    require(all(record.get(k) is False for k in ("predictions_or_scores_computed", "tau_train_values_read",
                "tau_test_values_read", "OOD_access", "complete_decoder", "successor_authorized")),
            "prefix_role_boundaries_required")
    expected = {layer: [dict(spec, status="complete") for spec in prefix.unit_specs(layer, shape)]
                for layer, shape in zip(prefix.LAYERS, prefix.SHAPES)}
    require(record.get("units") == expected, "all_1216_exact_completed_prefix_identities_required")
    binding = prefix.read_json(PRIOR / "training_row_binding.json")
    groups = binding.get("preserved_trial_rows_per_image", [])
    require(binding.get("training_image_ids") == ids and binding.get("role") == "subject_unique"
            and binding.get("participant") == "sub-01" and binding.get("image_mean_weighting") == "equal_images"
            and len(groups) == prefix.IMAGES and all(groups)
            and all(type(i) is int for group in groups for i in group)
            and sorted(i for group in groups for i in group) == list(range(18843)),
            "retained_image_repetition_binding_required")
    return groups


def blocks(start=START, end=END):
    require(START <= start < end <= END, "no_prefix_replay_or_out_of_scope_units")
    for begin in range(start, end, BLOCK):
        yield begin, min(begin + BLOCK, end)


def coordinates(spatial):
    require(START <= spatial < END, "no_prefix_replay_or_out_of_scope_unit")
    h, w = spatial % HEIGHT, spatial // HEIGHT
    return (0, h, w), CHANNELS * spatial, h * WIDTH + w


class Budget(prefix.Budget):
    def check(self):
        require(time.monotonic() < self.deadline, "continuation_deadline_reached")
        # A process guard, not a claim about Slurm RSS or free allocation memory.
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 6.5 * 1024**3,
                "worker_RSS_closure_threshold")

    def room(self, extra):
        require(sum(self.files.values()) + extra <= RESOURCES["durable_bytes"], "durable_byte_cap")


def shared_headers(path):
    """Bound array headers before allocating, using only this exact retained NPZ."""
    np = prefix.np
    shapes = {"normalized_X": (4712, ELIGIBLE), "penalized_Gram": (NATIVE, NATIVE),
              "all_native_voxel_ids": (NATIVE,), "eligible_native_voxel_ids": (ELIGIBLE,),
              "eligible_mask": (NATIVE,), "neural_mean": (1, NATIVE), "neural_std": (1, NATIVE),
              "image_repetition_counts": (4712,), "training_image_ids": (4712,)}
    require(path.stat().st_size < 1_000_000_000, "retained_state_size_mismatch")
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {k + ".npy" for k in shapes}, "exact_retained_state_arrays_required")
        for key, shape in shapes.items():
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "retained_state_array_header_version")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                actual, fortran, dtype = reader(stream)
                require(actual == shape and not dtype.hasobject, "retained_state_array_header_mismatch")
                if key in ("normalized_X", "penalized_Gram", "neural_mean", "neural_std"):
                    require(dtype.str == "<f4", "retained_float32_required")


def load_shared(ids, groups, budget):
    np = prefix.np
    path = PRIOR / "shared_training_state.npz"
    shared_headers(path)
    with np.load(path, allow_pickle=False) as state:
        require(state["training_image_ids"].tolist() == ids
                and state["image_repetition_counts"].tolist() == list(map(len, groups)),
                "retained_state_image_repetitions_mismatch")
        native, eligible, mask = (state[k] for k in
                                  ("all_native_voxel_ids", "eligible_native_voxel_ids", "eligible_mask"))
        require(native.dtype.kind in "iu" and eligible.dtype.kind in "iu" and mask.dtype.kind == "b"
                and bool((native[1:] > native[:-1]).all()) and int(mask.sum()) == ELIGIBLE
                and np.array_equal(native[mask], eligible), "retained_native_voxel_order_mismatch")
        x, gram = state["normalized_X"], state["penalized_Gram"]
    budget.check()
    # Existing Gram is reused unchanged; only its corresponding bias-augmented X
    # is reconstructed. Neither original neural values nor prefix coefficients open.
    augmented = np.hstack((x, np.ones((len(x), 1), dtype=np.float32)))
    return x, augmented, gram, eligible


def stage_channel(feature_root, scratch, budget):
    np, h5py = prefix.np, prefix.h5py
    require(BANK_BYTES <= RESOURCES["scratch_bytes"], "scratch_byte_cap")
    free = os.statvfs(scratch)
    require(free.f_bavail * free.f_frsize >= BANK_BYTES + 1_048_576, "node_local_capacity_required")
    path = scratch / "conv1_1_channel0.float32"
    require(not path.exists(), "scratch_bank_no_overwrite")
    bank = np.memmap(path, dtype="<f4", mode="w+", shape=(prefix.IMAGES, END))
    try:
        with h5py.File(feature_root / "features.h5", "r") as source, h5py.File(feature_root / "finite_packed.h5", "r") as flags:
            data, finite = source["conv1_1"], flags["conv1_1"]
            require(data.shape == (4712, 64, 224, 224) and data.dtype.str == "<f4"
                    and data.chunks is None and data.compression is None
                    and finite.shape == (4712, (64 * END + 7) // 8) and finite.dtype.str == "|u1",
                    "canonical_channel_headers_required")
            for first in range(0, prefix.IMAGES, IMAGE_BATCH):
                budget.check()
                last = min(first + IMAGE_BATCH, prefix.IMAGES)
                plane = data[first:last, 0, :, :]
                packed = finite[first:last, :END // 8]
                # Channel 0 occupies full-C [0,50176); bit order is little.
                valid = np.unpackbits(packed, axis=1, bitorder="little").reshape(-1, HEIGHT, WIDTH)
                arranged = plane.transpose(0, 2, 1).reshape(last - first, END)
                valid_f = valid.transpose(0, 2, 1).reshape(last - first, END)
                bad = np.argwhere((valid_f[:, START:] == 0) | ~np.isfinite(arranged[:, START:]))
                if bad.size:
                    image, offset = map(int, bad[0])
                    raise Stop(f"nonfinite_required_target:image={first + image}:spatial_F={START + offset}")
                bank[first:last] = arranged
                budget.logical_read_bytes += plane.nbytes + packed.nbytes
                del plane, packed, valid, arranged, valid_f, bad
        bank.flush()
        return bank
    except BaseException:
        bank._mmap.close()
        raise


class Fragment:
    """One fresh fixed-scope HDF5; state=2 only after the data fsync completes."""
    def __init__(self, path, budget, start=START, end=END):
        np, h5py = prefix.np, prefix.h5py
        require(START <= start < end <= END, "fragment_scope_mismatch")
        budget.room((end - start) * (500 * 12 + 41) + 1_048_576)
        self.path, self.budget, self.start, self.end = path, budget, start, end
        self.committed_end = start
        self.uncertain_publication = None
        self.file = h5py.File(path, "x")
        out, count = self.file, end - start
        out.attrs.update(schema_version="ep21.decoder_continuation_fragment.v1", participant="sub-01",
                         layer="conv1_1", channel=0, spatial_Fortran_start=start,
                         spatial_Fortran_stop=end, complete_decoder=False,
                         state_codes="0=incomplete,2=durably_complete",
                         source_fragment_axes="unit_by_ranked_selected_native_voxel; bias_separate")
        out.create_dataset("state", (count,), dtype="u1", fillvalue=0)
        out.create_dataset("selected_native_voxel_ids", (count, 500), dtype="<i8", fillvalue=-1)
        out.create_dataset("weights", (count, 500), dtype="<f4", fillvalue=np.nan)
        for name in ("bias", "target_mean", "target_std"):
            out.create_dataset(name, (count,), dtype="<f4", fillvalue=np.nan)
        out.create_dataset("solve_seconds", (count,), dtype="<f8", fillvalue=np.nan)
        spatial = np.arange(start, end, dtype=np.int64)
        out.create_dataset("spatial_Fortran_index", data=spatial)
        out.create_dataset("full_layer_Fortran_index", data=CHANNELS * spatial)
        out.create_dataset("full_layer_C_index", data=(spatial % HEIGHT) * WIDTH + spatial // HEIGHT)
        prefix.fsync_h5(out)
        budget.note(path)

    def commit(self, begin, rows):
        np = prefix.np
        require(begin == self.committed_end and rows and begin + len(rows) <= self.end,
                "disjoint_ascending_new_units_required")
        start, stop = begin - self.start, begin + len(rows) - self.start
        require(bool((self.file["state"][start:stop] == 0).all()), "completed_rows_immutable")
        for ids, weights, bias, mean, std, elapsed in rows:
            require(ids.shape == weights.shape == (500,) and len(np.unique(ids)) == 500
                    and np.isfinite(weights).all() and np.isfinite([bias, mean, std, elapsed]).all()
                    and std > 0 and elapsed >= 0, "valid_solved_row_required")
        # Only this bounded transaction defers the ordinary deadline. Kernel and
        # external limits remain active; a killed block without markers is incomplete.
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM, signal.SIGTERM})
        try:
            out = self.file
            out["selected_native_voxel_ids"][start:stop] = np.stack([r[0] for r in rows])
            out["weights"][start:stop] = np.stack([r[1] for r in rows])
            for index, name in enumerate(("bias", "target_mean", "target_std", "solve_seconds"), 2):
                out[name][start:stop] = [r[index] for r in rows]
            prefix.fsync_h5(out)
            phase = "completion_marker_assignment"
            try:
                out["state"][start:stop] = 2
                phase = "completion_marker_fsync"
                prefix.fsync_h5(out)
            except Exception as exc:
                # A later close may persist these markers. Preserve the evidence;
                # do not erase them, retry the write, or call their units unfitted.
                self.uncertain_publication = {
                    "spatial_Fortran_interval": [begin, begin + len(rows)],
                    "phase": phase, "error_type": type(exc).__name__}
                raise
            self.committed_end = begin + len(rows)
            self.budget.note(self.path)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)

    def close(self):
        self.file.close()


def fit_blocks(bank, x, augmented, gram, native, corrmat, fragment, budget, timing=None):
    np = prefix.np
    if timing is None:
        timing = {"normalization_correlation_RHS_seconds": 0., "select_solve_seconds": 0.,
                  "publication_seconds": 0.}
    for begin, end in blocks():
        budget.check()
        setup_started = time.monotonic()
        targets = np.ascontiguousarray(bank[:, begin:end], dtype=np.float32)
        try:
            math = prefix.prepare_layer_math(x, augmented, gram, targets, corrmat)
        finally:
            timing["normalization_correlation_RHS_seconds"] += time.monotonic() - setup_started
        del targets
        if "bad_units" in math:
            raise Stop(math["error"] + ":spatial_F=" + ",".join(str(begin + i) for i in math["bad_units"]))
        require(math["branch"] == "source_greater_than_500_with_native_ID_ties", "retained_source_branch_required")
        solved_rows = []
        try:
            for index in range(end - begin):
                budget.check()
                started = time.monotonic()
                try:
                    selected, weights, bias = prefix.solve_selected(gram, math["right"][index],
                                                                    math["correlations"][index], native)
                finally:
                    elapsed = time.monotonic() - started
                    timing["select_solve_seconds"] += elapsed
                solved_rows.append((native[selected], weights, bias, math["means"][0, index],
                                    math["scales"][0, index], elapsed))
        finally:
            # Preserve completed solves even when the ordinary deadline interrupts
            # a later unit. No automatic reopening or partial-fragment replay exists.
            if solved_rows:
                started = time.monotonic()
                try:
                    fragment.commit(begin, solved_rows)
                finally:
                    timing["publication_seconds"] += time.monotonic() - started
        del math, solved_rows


def run(args):
    require(Path.cwd().resolve() == BASE, "exact_episode_working_directory_required")
    require(type(args.max_seconds) is int and 0 < args.max_seconds <= MAX_SECONDS, "bounded_internal_seconds_required")
    authorize(args.authorization)  # Before numerical imports or any empirical artifact.
    allocation = verify_allocation()
    require(not OUTPUT.exists(), "existing_attempt_no_replay_or_overwrite")
    budget = Budget(allocation, args.max_seconds)
    lock, selected, _, feature_root, feature_admission = prefix.admit_metadata()
    ids = prefix.feature_index(feature_root / "image_index.tsv", selected)
    groups = admit_prefix(ids)
    scratch_root = Path(os.environ.get("L_SCRATCH_JOB", ""))
    require(scratch_root.is_absolute() and scratch_root.is_dir(), "explicit_node_local_job_scratch_required")
    scratch = scratch_root / "ep21-decoder-continuation-001"
    require(not scratch.exists(), "fresh_node_local_scratch_required")
    OUTPUT.mkdir(mode=0o700)
    record = {"schema_version": "ep21.decoder_continuation.v1", "status": "admitted_pending",
              "authorization": str(AUTH), "allocation": allocation, "resources": RESOURCES,
              "max_work_seconds": budget.seconds, "participant": "sub-01", "training_role": "subject_unique",
              "training_images": 4712, "training_repetitions": 18843, "layer": "conv1_1", "channel": 0,
              "requested_spatial_Fortran_interval": [START, END], "units_requested": END - START,
              "retained_prefix_reference": str(PRIOR), "retained_prefix_completed_units": 1216,
              "shared_state_reference": str(PRIOR / "shared_training_state.npz"),
              "training_row_binding_reference": str(PRIOR / "training_row_binding.json"),
              "committed_spatial_Fortran_interval": [START, START], "remaining_spatial_Fortran_interval": [START, END],
              "remaining_interval_status": "exact_uncommitted_coverage",
              "completed_units_is_lower_bound": False, "repair_requires_new_authorization": True,
              "feature_admission": feature_admission, "original_neural_values_read": False,
              "retained_neural_state_read": False, "regular_feature_values_read": False,
              "predictions_or_scores_computed": False, "tau_train_values_read": False,
              "tau_test_values_read": False, "OOD_access": False, "complete_decoder": False,
              "successor_authorized": False, "automatic_retry": False,
              "original_budget_consumption_preserved": True, "fragment": "conv1_1_channel0.h5",
              "memory_limit_is_not_fit_guarantee": True, "bank_bytes": BANK_BYTES}
    record["block_timings"] = {"normalization_correlation_RHS_seconds": 0.,
                               "select_solve_seconds": 0., "publication_seconds": 0.}
    prefix.publish_json(OUTPUT / "requested_units.json", record, budget)
    bank = fragment = None
    error = None
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame):
        raise Stop("continuation_deadline_reached" if signum == signal.SIGALRM else "termination_requested")
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
        record["retained_neural_state_read"] = True
        started = time.monotonic()
        try:
            x, augmented, gram, native = load_shared(ids, groups, budget)
        finally:
            record["shared_state_load_and_bias_augmentation_seconds"] = time.monotonic() - started
        # Conservative explicit large-array accounting includes a fully resident
        # bank, two source-centered X temporaries and 3 GiB for unmodelled memory.
        estimate = gram.nbytes + augmented.nbytes + 3 * x.nbytes + BANK_BYTES + 3 * 1024**3
        record["conservative_memory_planning_bytes"] = estimate
        require(estimate < RESOURCES["memory_gib"] * 1024**3, "planned_arrays_exceed_memory_cap")
        record["regular_feature_values_read"] = True
        started = time.monotonic()
        bank = stage_channel(feature_root, scratch, budget)
        record["channel_loading_seconds"] = time.monotonic() - started
        fragment = Fragment(OUTPUT / record["fragment"], budget)
        started = time.monotonic()
        try:
            fit_blocks(bank, x, augmented, gram, native, corrmat, fragment, budget, record["block_timings"])
        finally:
            record["fitting_and_publication_seconds"] = time.monotonic() - started
        require(fragment.committed_end == END, "fixed_tranche_coverage_required")
        record["status"] = "complete_continuation_tranche_only"
    except Exception as exc:
        error = exc
        record.update(status="partial_continuation_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if fragment is not None:
            record["committed_spatial_Fortran_interval"] = [START, fragment.committed_end]
            record["remaining_spatial_Fortran_interval"] = [fragment.committed_end, END]
            if fragment.uncertain_publication is not None:
                record["uncertain_publication"] = fragment.uncertain_publication
                record["remaining_interval_status"] = "not_authoritative_for_repair_due_to_uncertain_marker_publication"
                record["completed_units_is_lower_bound"] = True
            fragment.close()
        if bank is not None:
            bank._mmap.close()
        bank_path = scratch / "conv1_1_channel0.float32"
        record["scratch_bytes_written_by_worker"] = bank_path.stat().st_size if bank_path.exists() else 0
        if bank_path.exists():
            bank_path.unlink()  # Reconstructible, task-owned staging only.
        if scratch.exists():
            scratch.rmdir()
        record["completed_units"] = record["committed_spatial_Fortran_interval"][1] - START
        record["elapsed_seconds"] = time.monotonic() - budget.started
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_input_bytes_from_channel"] = budget.logical_read_bytes
        record["durable_bytes_before_final_status"] = sum(budget.files.values())
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
        result = run(args)
    except Exception as exc:
        print(json.dumps({"status": "continuation_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2
    print(json.dumps({"status": "continuation_tranche_complete" if result == 0 else "continuation_stopped",
                      "report": str(OUTPUT / "continuation_status.json"), "complete_decoder": False,
                      "no_retry_or_successor": True}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
