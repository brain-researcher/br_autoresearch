"""Separate bounded cost measurement for the first complete symmetric core grid.

Reuse the initialized original30-person inputs and fixed sub07 recipe. Prepare
the original inner(0,1) training state once; complete all13 penalties x61 scalar
fits before scoring either image-validation side. Preserve original sequence,
E3, PCA, nuisance, whitening, weighting and solver conventions through existing
helpers. Private checkpoints retain coefficients; public records contain only
cost/count/status information. This is not a full participant, search trial,
scientific result, grant, automatic retry or reuse of the stopped old attempt.
Only --resume-checkpoints selects reuse of this separate stage's completed
scalars; it requires fresh public attempt records and the same actual prepared
training state. A complete grid cannot be restarted through this entrypoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import core_reference_paired_inner as paired
import production_baseline_controller as controller
import production_core_reference as core
from training_transforms import CommonSensorSpace


def _owned_process_pss(pid):
    """Read the same proportional-resident metric on old and new kernels."""
    try:
        source = open(f"/proc/{pid}/smaps_rollup", "rb")
        cap, rollup = 65536, True
    except FileNotFoundError:
        source = open(f"/proc/{pid}/smaps", "rb")
        cap, rollup = 16*1024**2, False
    with source:
        payload = source.read(cap+1)
    core.require(len(payload) <= cap, "owned_process_PSS_record_byte_cap")
    lines = payload.splitlines()
    core.require(len(lines) <= 262144, "owned_process_PSS_record_line_cap")
    entries = [line.split() for line in lines if line.startswith(b"Pss:")]
    core.require(bool(entries) and (not rollup or len(entries) == 1),
                 "owned_process_PSS_measurement_required")
    core.require(all(len(row) == 3 and row[2] == b"kB" and row[1].isdigit() for row in entries),
                 "owned_process_PSS_units_required")
    return sum(int(row[1])*1024 for row in entries)


PERSON = "sub-07"
FIRST_SPLIT = (0, 1)
PUBLIC_PROGRESS_CAP = 1048576
FINAL_RESERVE = 65536


def _retain_inner(private_dir, result, budget):
    path = private_dir / ("inner_%02d_%02d.json" %
                          (result.split.outer_index, result.split.inner_index))
    if not path.exists():
        controller._split_record(path, result, budget)
        return
    # A manually selected continuation never overwrites an already complete side.
    saved = controller._json(path, budget.record)
    record = saved["record"]
    core.require(record["split"] == controller._plain(asdict(result.split))
                 and record["training_sequence_ids"] == list(result.training_sequence_ids)
                 and record["heldout_sequence_ids"] == list(result.heldout_sequence_ids)
                 and saved["array_file"] == path.with_suffix(".npz").name
                 and path.with_suffix(".npz").stat().st_size == saved["array_bytes"],
                 "completed_inner_record_binding_changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        core.require(set(arrays.files) == {"array_0000", "array_0001", "array_0002"},
                     "complete_original_inner_arrays_required")
        for i, field in enumerate(("image_sequence_losses", "nuisance_sequence_losses", "normalized_image_losses")):
            core.require(record[field]["array"] == "array_%04d" % i
                         and np.array_equal(arrays["array_%04d" % i], getattr(result, field)),
                         "retained_inner_result_changed")


def _checkpoint_fit(*args, **kwargs):
    from core_reference_scalar_checkpoint import fit_checkpointed
    return fit_checkpointed(*args, **kwargs)


def _prepared_state_signature(cache, sensor, check):
    """Bind resumed coefficients to actual prepared numbers, not source names.

    Hash once per rebuilt training state, streaming array bytes in <=1MiB
    chunks. No extra copy of the complete operator/target matrix is created.
    This identity stays in private checkpoint metadata.
    """
    state = hashlib.sha256()
    def token(value):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        state.update(len(encoded).to_bytes(8, "little"))
        state.update(encoded)
    def visit(value):
        check()
        if isinstance(value, np.ndarray):
            token(("array", value.dtype.str, value.shape))
            if value.size == 0:
                return
            if value.flags.c_contiguous:
                raw = memoryview(value).cast("B")
                for offset in range(0, len(raw), 1048576):
                    check()
                    state.update(raw[offset:offset+1048576])
            else:
                for chunk in np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                                       order="C", buffersize=131072):
                    check()
                    state.update(chunk.tobytes(order="C"))
        elif is_dataclass(value):
            token(("dataclass", type(value).__name__, [field.name for field in fields(value)]))
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            token(("sequence", len(value)))
            for item in value:
                visit(item)
        else:
            token(value.item() if isinstance(value, np.generic) else value)
    design = cache.train_designs["image"]
    visit(("original_prepared_core_image_v1", design.shape, sensor.channel_order, sensor.basis))
    for sequence in design.sequences:
        visit((sequence.sequence_id, sequence.sample_times, sequence.nuisance, tuple(sequence.terms)))
        for name, term in sequence.terms.items():
            visit((name, term.basis_dimension, term.column_slice.start, term.column_slice.stop,
                   term.features, term.kernels))
    visit((cache.targets, cache.weights, cache.whitener))
    builder = cache.preparation.build_designs
    if isinstance(builder, core.CoreDesignBuilder):
        visit(builder.image_transform)
    return state.hexdigest()


def _reserve_checkpoint_storage(private_dir, columns, recipe, budget):
    # Arrays in the scalar adapter: p×61 float64 coefficients, p penalties,
    # 61 bool flags and four61-element int64/float64 diagnostic vectors.
    # Its bounded metadata plus ZIP/NPY header allowance is included explicitly.
    per_checkpoint = columns*62*8 + 61*(1+4*8) + 65536 + 16384
    core.require(per_checkpoint <= recipe["npz_byte_cap"], "complete_scalar_checkpoint_fits_original_record_cap")
    pending_bytes = 0
    for path in private_dir.glob("image_penalty_*.npz.pending-*"):
        budget.check()
        core.require(path.is_file() and not path.is_symlink(), "owned_partial_checkpoint_file_required")
        pending_bytes += path.stat().st_size
        budget.admit_write(pending_bytes)
    # Thirteen final checkpoint records plus one simultaneous atomic rewrite;
    # previous orphan writes are preserved and charged in addition. This is
    # retained-storage admission, not accumulated I/O across793 replacements.
    reservation = 14*per_checkpoint + pending_bytes
    budget.admit_write(reservation)
    budget.written += reservation
    return per_checkpoint


def fit_first_pair(person, recipe, *, private_dir, budget, workers,
                   settings_id, on_stage, on_scalar, scalar_fit=None):
    """Full original grid; only the scalar execution/checkpoint layer differs."""
    recipe = core.validate_recipe(recipe)
    core.require(person.participant_id == PERSON and type(workers) is int and 1 <= workers <= 4,
                 "fixed_sub07_and_at_most_four_scalar_workers_required")
    scalar_fit = _checkpoint_fit if scalar_fit is None else scalar_fit
    sensor = CommonSensorSpace(person.inputs.channel_order)
    limits = folds.SolverLimits(**recipe["solver"])
    sequences = folds._roster(person.sequences, sensor, budget.check)
    split = folds._split(*FIRST_SPLIT)
    cache = core._prepare_core(sequences, split, sensor, recipe, person.prepare_training,
                              limits, budget.check, on_stage)
    on_stage("binding_actual_prepared_training_state")
    prepared_id = _prepared_state_signature(cache, sensor, budget.check)
    checkpoint_cap = _reserve_checkpoint_storage(private_dir, cache.train_designs["image"].shape[1],
                                                recipe, budget)
    on_stage("complete13_by61_image_scalar_grid")
    fits = []
    for index, ridge in enumerate(tuning.RIDGE_GRID):
        budget.check()
        binding = {"settings_id": settings_id, "recipe_id": controller.digest(recipe),
                   "participant_id": PERSON, "split": asdict(split),
                   "model": "image", "penalty_index": index, "ridge": float(ridge),
                   "prepared_state_sha256": prepared_id}
        fit = scalar_fit(cache.train_designs["image"], cache.targets, cache.weights,
            tuning._grid_penalty(cache, "image", index), limits, budget.check,
            checkpoint=private_dir / ("image_penalty_%02d.npz" % index), binding=binding,
            workers=workers, record_byte_cap=checkpoint_cap,
            progress=lambda row, index=index: on_scalar(index, row))
        core.require(fit.coefficients.shape == (cache.train_designs["image"].shape[1], 61)
                     and len(fit.stop_codes) == len(fit.iterations) == 61,
                     "complete_original61_scalar_fit_required")
        fits.append(fit)
    core.require(len(fits) == 13, "complete_original13_penalty_grid_required")
    # No image predictions or image scores occur until every scalar is retained.
    on_stage("both_original_image_validation_sides")
    convention = SimpleNamespace(normalization=recipe["normalization"])
    counterpart = paired._counterpart(cache, sequences, sensor, convention, budget.check)
    results = []
    for view in (cache, counterpart):
        losses = folds._readonly([tuning._score(view, "image", fit, budget.check) for fit in fits])
        normalized = folds._readonly(tuning.normalize_losses(losses, view.nuisance_losses,
                                                            normalization=convention.normalization))
        results.append(controller.CoreInnerResult(view.split,
            tuple(row.sequence_id for row in view.training),
            tuple(row.sequence_id for row in view.heldout), losses, view.nuisance_losses, normalized))
    for result in results:
        _retain_inner(private_dir, result, budget)
    return tuple(results)


def _allocation(budget, workers):
    budget.require_compute_threads()
    core.require(budget.threads == 1 and type(workers) is int and 1 <= workers <= 4,
                 "one_BLAS_thread_per_scalar_process_required")
    core.require(os.environ.get("SLURM_CPUS_PER_TASK") == "4"
                 and os.environ.get("SLURM_MEM_PER_NODE") == "15360"
                 and not os.environ.get("SLURM_JOB_GPUS")
                 and not os.environ.get("SLURM_STEP_GPUS"), "selected_four_CPU_15GiB_no_GPU_required")
    start, end = int(os.environ["SLURM_JOB_START_TIME"]), int(os.environ["SLURM_JOB_END_TIME"])
    # Slurm's exported actual end-start can differ from nominal TimeLimit by
    # a few seconds; the selected two-hour request is coordinator-verified.
    core.require(start <= time.time() < end and start < end
                 and budget.deadline <= min(start+7140, end-60, controller.LAST_DEADLINE-10),
                 "unchanged_episode_and_two_hour_allocation_clocks_required")


def run_cost(initialized_root, private_dir, public_dir, *, budget, workers=4,
             aggregate_pss_limit_bytes, resume_checkpoints=False):
    _allocation(budget, workers)
    root, private, public = map(controller._path, (initialized_root, private_dir, public_dir))
    core.require(private != root and root not in private.parents and private not in root.parents
                 and public != root and root not in public.parents and public not in root.parents
                 and private != public and private not in public.parents and public not in private.parents,
                 "fresh_separate_private_and_public_cost_paths_required")
    core.require(type(resume_checkpoints) is bool and not public.exists(), "fresh_public_cost_attempt_required")
    core.require(private.is_dir() if resume_checkpoints else not private.exists(),
                 "explicit_checkpoint_continuation_or_fresh_private_stage_required")
    core.require(type(aggregate_pss_limit_bytes) is int and 0 < aggregate_pss_limit_bytes <= 12*1024**3,
                 "explicit_aggregate_PSS_guard_within_15GiB_allocation_required")
    inputs = core.bind_initialized(root, budget)
    core.require(inputs.people[0] == PERSON, "original_first_frozen_person_required")
    binding = {"participant_id": PERSON, "settings_id": controller.digest(inputs.settings),
        "recipe_id": controller.digest(inputs.recipe), "original_initialized_root": str(root),
        "fresh_private_directory": str(private), "requested_penalties": 13,
        "contrasts_per_penalty": 61, "requested_image_scalar_fits": 793,
        "symmetric_split_indices": [[0,1],[1,0]]}
    if resume_checkpoints:
        core.require(not (private/"complete.json").exists(), "already_completed_grid_refuses_continuation")
        original = controller._json(private/"settings.json", budget.record)
        core.require(all(original.get(key) == value for key, value in binding.items()),
                     "original_checkpoint_stage_binding_changed")
        # The13 final coefficient files are admitted separately as a complete
        # fixed-grid reservation once their actual column dimension is known.
        known = [private/"settings.json"]
        for stem in ("inner_00_01", "inner_01_00"):
            known.extend(private/(stem+suffix) for suffix in (".json", ".npz", ".json.partial", ".npz.partial"))
        retained = sum(path.stat().st_size for path in known if path.exists())
        budget.admit_write(retained)
        budget.written += retained
    else:
        private.mkdir(mode=0o700, parents=False, exist_ok=False)
    public.mkdir(mode=0o700, parents=False, exist_ok=False)
    report = {"status": "incomplete_first_symmetric_core_grid", **binding,
        "job_id": os.environ.get("SLURM_JOB_ID"), "deadline_epoch": budget.deadline,
        "workers": workers, "BLAS_threads_per_process": 1, "requested_penalties": 13,
        "aggregate_pss_limit_bytes": aggregate_pss_limit_bytes,
        "aggregate_pss_observed_peak_bytes": 0,
        "aggregate_memory_guard": "parent_and_owned_workers_PSS_sampled_at_budget_checks_not_hard_limit",
        "contrasts_per_penalty": 61, "requested_image_scalar_fits": 793,
        "completed_image_scalar_fits": 0, "completed_image_validation_sides": 0,
        "completed_penalties": 0, "last_stage": "loading_original_sub07",
        "explicit_checkpoint_continuation": resume_checkpoints,
        "reused_image_scalar_fits": 0,
        "scientific_release_authorized": False, "full_participant_completed": False,
        "audit_access": False, "automatic_retry": False, "old_partial_fit_reused": False}
    core.require(budget.durable-budget.written >= FINAL_RESERVE+PUBLIC_PROGRESS_CAP,
                 "bounded_operational_reserve_required")
    budget.durable -= FINAL_RESERVE
    started, progress_bytes = time.monotonic(), 0
    owner_pid, worker_pids, last_memory_check = os.getpid(), set(), -np.inf
    original_check = budget.check
    def checked():
        nonlocal last_memory_check
        original_check()
        # Workers inherit the callback under fork; only their owner sums PSS.
        if os.getpid() != owner_pid or time.monotonic()-last_memory_check < .25:
            return
        total = 0
        for pid in (owner_pid, *sorted(worker_pids)):
            try:
                pss = _owned_process_pss(pid)
            except FileNotFoundError:
                core.require(pid != owner_pid, "owner_memory_measurement_required")
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise RuntimeError("live_owned_worker_PSS_measurement_unavailable")
            total += pss
        last_memory_check = time.monotonic()
        report["aggregate_pss_observed_peak_bytes"] = max(report["aggregate_pss_observed_peak_bytes"], total)
        if total > aggregate_pss_limit_bytes:
            raise MemoryError("aggregate_owned_process_PSS_guard_reached")
    budget.check = checked
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")
    def stage(value):
        budget.check()
        report["last_stage"] = value
    def scalar(index, row):
        nonlocal progress_bytes
        if row.get("event") == "workers_started":
            supplied = row.get("worker_pids")
            core.require(isinstance(supplied, list) and len(supplied) <= workers
                         and len(set(supplied)) == len(supplied)
                         and all(type(pid) is int and pid > 0 and pid != owner_pid for pid in supplied),
                         "bounded_owned_scalar_worker_PIDs_required")
            worker_pids.update(supplied)
        elif row.get("event") == "workers_finished":
            worker_pids.clear()
        elif row.get("event") == "checkpoint_loaded":
            reused = row.get("reused_scalars")
            core.require(type(reused) is int and 0 <= reused <= 61, "bounded_reused_scalar_count_required")
            report["reused_image_scalar_fits"] += reused
        # Exact scalar checkpoint metadata; never stringify arbitrary solver output.
        allowed = {key: row[key] for key in
                   ("event", "column", "completed_scalars", "required_scalars", "reused_scalars", "stop_code",
                    "iterations", "elapsed_seconds", "reused", "workers", "worker_peak_rss_bytes")
                   if key in row}
        value = {"penalty_index": index, **allowed}
        text = (json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode()
        core.require(len(text) <= 1024 and progress_bytes+len(text) <= PUBLIC_PROGRESS_CAP,
                     "bounded_scalar_progress_required")
        budget.admit_write(len(text))
        with (public / "scalar_progress.jsonl").open("ab") as target:
            os.chmod(public / "scalar_progress.jsonl", 0o600)
            target.write(text)
            target.flush()
        budget.written += len(text)
        progress_bytes += len(text)
        completed = row.get("completed_scalars")
        if type(completed) is int and 0 <= completed <= 61:
            report["completed_image_scalar_fits"] = index*61+completed
            report["completed_penalties"] = index+(completed == 61)
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        if not resume_checkpoints:
            controller._publish_json(private / "settings.json", report, budget)
        inputs.check = budget.check
        inputs.images.check = budget.check
        person = inputs.load_participant(PERSON)
        result = fit_first_pair(person, inputs.recipe, private_dir=private, budget=budget, workers=workers,
            settings_id=report["settings_id"], on_stage=stage, on_scalar=scalar)
        core.require(tuple((row.split.outer_index, row.split.inner_index) for row in result) == ((0,1),(1,0)),
                     "both_original_symmetric_validation_sides_required")
        completion = {**report, "status": "complete_first_symmetric_core_grid", "completed_penalties": 13,
                      "completed_image_scalar_fits": 793, "completed_image_validation_sides": 2}
        controller._publish_json(private / "complete.json", completion, budget)
        report.update(completion)
    except Exception as error:
        report["error_class"] = type(error).__name__
    finally:
        budget.check = original_check
        for sig, original in previous.items():
            signal.signal(sig, original)
        report["elapsed_seconds"] = time.monotonic()-started
        report["progress_bytes"] = progress_bytes
        # Reuse the original bounded closeout mechanism even after soft refusal.
        core._operational_finish(public, report, budget)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialized-root", required=True)
    parser.add_argument("--private-dir", required=True)
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--workers", type=int, choices=(1,2,3,4), default=4)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, required=True)
    parser.add_argument("--durable-byte-limit", type=int, required=True)
    parser.add_argument("--record-byte-limit", type=int, required=True)
    parser.add_argument("--aggregate-pss-limit-bytes", type=int, required=True)
    parser.add_argument("--resume-checkpoints", action="store_true",
                        help="Explicit coordinator-selected continuation; never automatic")
    args = parser.parse_args(argv)
    try:
        budget = controller.RunBudget(deadline_epoch=args.deadline_epoch,
            rss_limit_bytes=args.rss_limit_bytes, durable_byte_limit=args.durable_byte_limit,
            record_byte_limit=args.record_byte_limit, threads=1)
        report = run_cost(args.initialized_root, args.private_dir, args.public_dir,
                          budget=budget, workers=args.workers,
                          aggregate_pss_limit_bytes=args.aggregate_pss_limit_bytes,
                          resume_checkpoints=args.resume_checkpoints)
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 0 if report["status"] == "complete_first_symmetric_core_grid" else 1
    except Exception as error:
        print(json.dumps({"status": "scalar_cost_refused", "error_class": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
