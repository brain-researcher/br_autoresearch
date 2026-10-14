"""Stage 1 item 1: O1 planner and O2 readout against the unmodified legacy code.

Development tests (not counted). Part A runs the legacy streaming hold
planner restricted to windows 0..W-1 (the B-family field bound is constant,
asserted bitwise) for a fast precheck and the O2 streaming-readout comparison
with legacy hold means. Part B runs the full legacy chain once: the complete
16-s streaming hold plan (first pass) and bind_hold_windows (second pass);
every window's fragments are compared, and windows 0-2, 20 seeded random
windows and the final window are compared record by record (canonical byte
stream and legacy window digest), plus the eager legacy readout. Planning and
readout only: no electrical forward simulation. One terminal JSON always.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace


def _write_json(path, value):
    try:
        encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=repr) + "\n"
    except (TypeError, ValueError) as error:
        encoded = json.dumps(dict(status="report_serialization_failed", error=str(error)[:2000],
                                  summary=repr(value)[:20000]), indent=2) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _failure(phase, error):
    return dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:2000],
                traceback=traceback.format_exc()[-6000:])


def _differences(left, right):
    keys = sorted(set(left) | set(right))
    return {key: (repr(left.get(key, "<missing>")), repr(right.get(key, "<missing>")))
            for key in keys if left.get(key, "<missing>") != right.get(key, "<missing>")}


def part_a(case, plan, readout, clock, *, windows, seed, retained):
    """Restricted legacy chain: O1 precheck (windows 0-2) and O2 streaming readout."""
    import numpy as np
    from fast_io_exact import CYCLE_SECONDS, float_bits, shift
    from fast_io_legacy_reference import (canonical_window_bytes, compare_readout, compare_splits,
        cursor_for, legacy_fragment_check, legacy_window_digest, run_streaming_readout,
        synthetic_hold_means, validator_context, window_ends_check)
    from fast_io_readout import CompactRecordWriter, WORK_FIELDS, read_window, readouts_bitwise_equal
    from streaming_input_plan import bind_hold_windows
    from streaming_midpoint_hold import compile_streaming_midpoint_hold
    t, check = plan.template, clock.check
    result = {}
    end = float(shift(windows - 1, CYCLE_SECONDS))
    restricted_bound = plan.bound_contact.derivative_bound_volts_per_second(0.0, end)
    result["restricted_field_bound_bitwise_equal_full_record"] = bool(
        float_bits(restricted_bound) == float_bits(plan.lipschitz))
    if not result["restricted_field_bound_bitwise_equal_full_record"]:
        raise AssertionError("restricted support changes the field bound; Part A is not a legacy proxy")
    clock.set("A: legacy streaming hold plan restricted to windows 0..W-1")
    restricted = compile_streaming_midpoint_hold(case.raw_time, case.contacts,
        lambda: (action for k in range(windows) for action in case.schedule.physical_actions_for_cycle(k)),
        support=(0.0, end), epsilon_input_volts=1e-8, max_intervals=40000000, max_work=100000000,
        check=check, noise_hold_bound=case.noise.hold_bound)
    result["restricted_hold_report"] = dict(restricted.report)
    clock.set("A: legacy windows via bind_hold_windows and schedule.readout_window")
    factory = bind_hold_windows(restricted, lambda: (case.schedule.readout_window(k) for k in range(windows)),
                                max_windows=windows, max_window_fragments=200000, check=check)
    legacy = list(factory())
    result["legacy_window_bounds_exact"] = window_ends_check(legacy)
    plans = [plan.window(k) for k in range(windows)]
    legacy_context = validator_context(field_provider=restricted.field_provider, array=case.array,
        noise=case.noise, limits=case.limits, input_bindings=case.schedule.input_bindings, check=check)
    new_context = validator_context(field_provider=plan.field_provider, array=case.array, noise=case.noise,
        limits=case.limits, input_bindings=case.schedule.input_bindings, check=check)
    clock.set("A: O1 precheck windows 0-2 (canonical stream and legacy window digest)")
    precheck, events = [], 0
    for k in range(min(3, windows)):
        cursor = cursor_for(k, events, events - 1)
        old = legacy_window_digest(legacy_context, legacy[k], cursor)
        new = legacy_window_digest(new_context, plan.legacy_window(k, plans[k]), cursor)
        precheck.append(dict(window=k, digest_equal=old["digest"] == new["digest"],
                             canonical_equal=old["canonical"] == new["canonical"],
                             canonical_bytes=len(old["canonical"]),
                             fragments=legacy_fragment_check(legacy[k], plans[k], t)))
        events += len(legacy[k].physical_intervals)
    result["o1_precheck"] = precheck
    clock.set("A: O2 legacy streaming readout with legacy hold means")
    bounds0, retained_means = retained
    grid0 = plans[0].split_bounds
    retained_match = (len(bounds0) <= len(grid0) - 1 and all(
        float_bits(a) == float_bits(grid0[j]) and float_bits(b) == float_bits(grid0[j + 1])
        for j, (a, b) in enumerate(bounds0)))
    result["retained_attempt004_hold_means_used"] = bool(retained_match)
    result["retained_attempt004_intervals"] = len(bounds0)
    q = readout.quantizers[0]
    means = [synthetic_hold_means(plans[k], thresholds=q.thresholds, levels=q.levels, input_min=q.input_min_volts,
                                  input_max=q.input_max_volts, n_adc=len(t.adc_ids), seed=seed * 10007 + k,
                                  retained=retained_means if (k == 0 and retained_match) else None)
             for k in range(windows)]
    per_window, executed_report = run_streaming_readout(legacy, input_ids=t.input_ids, input_adc=t.input_adc,
        hold_means=lambda k: means[k], splitter=SimpleNamespace(_limits=case.limits), validate_only=False,
        check=check)
    new_readouts = [readout.read_window(plans[k], means[k]) for k in range(windows)]
    comparisons = []
    for k in range(windows):
        problems = compare_readout(per_window[k], new_readouts[k], t, completed=plans[k].conv_completed)
        problems += compare_splits(per_window[k]["bounds"], per_window[k]["active"], plans[k], t.input_ids)
        codes = new_readouts[k].code
        comparisons.append(dict(window=k, problems=problems, raw_values=len(per_window[k]["raw"]),
            observations=len(per_window[k]["observations"]),
            clipped_conversions=int(new_readouts[k].raw_clipped.sum()),
            distinct_codes=int(len(np.unique(codes))),
            near_threshold_conversions=int(np.sum(np.abs(new_readouts[k].aperture_mean
                - q.thresholds[np.clip(codes - 1, 0, len(q.thresholds) - 1)]) <= 4 * np.spacing(
                np.abs(new_readouts[k].aperture_mean) + 1e-300)))))
    result["o2_streaming"] = comparisons
    result["o2_streaming_executed_report"] = executed_report
    clock.set("A: metadata-only legacy readout totals versus closed form")
    _, dry_report = run_streaming_readout(legacy, input_ids=t.input_ids, input_adc=t.input_adc,
        hold_means=None, splitter=SimpleNamespace(_limits=case.limits), validate_only=True, check=check)
    closed = plan.readout_totals(windows)
    result["prefix_readout_totals_equal"] = closed == dry_report
    result["prefix_readout_totals_differences"] = _differences(closed, dry_report)
    clock.set("A: compact writer round trip")
    scratch = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    path = scratch / f"fast_io_record_{os.environ.get('SLURM_JOB_ID', 'local')}.h5"
    digests = dict(template_digest=t.digest, field_descriptor_digest=plan.field_descriptor_digest,
                   noise_descriptor_digest=plan.noise_descriptor_digest)
    writer = CompactRecordWriter(path, template=t, plan_digests=digests, windows=windows,
                                 accounting_columns=("energy_balance_residual_joule",))
    rng = np.random.Generator(np.random.PCG64(seed))
    cumulative = dict.fromkeys(WORK_FIELDS, 0.)
    try:
        for k in range(windows):
            window_energy = {name: float(value) for name, value in zip(WORK_FIELDS, rng.normal(0, 1e-15, len(WORK_FIELDS)))}
            cumulative = {name: cumulative[name] + window_energy[name] for name in WORK_FIELDS}
            writer.write_window(new_readouts[k], energy_window=window_energy, energy_cumulative=cumulative,
                                accounting=dict(energy_balance_residual_joule=0.0))
        writer.finish(status="complete", metadata=dict(test="stage1_round_trip_not_scientific"))
    finally:
        writer.close()
    size = path.stat().st_size
    round_trip = [readouts_bitwise_equal(read_window(path, plan, k), new_readouts[k]) for k in range(windows)]
    path.unlink()
    result["writer_round_trip_differences"] = round_trip
    result["writer_bytes_per_window"] = size / windows
    result["writer_projected_record_bytes"] = size / windows * 4000
    clock.set("A: batched field evaluation (informational for O6)")
    keys = plans[0].hold_key
    batched = (plan.bound_contact._modes @ case.raw_time.drivers_uv(keys)) * 1e-6
    single = np.stack([plan.field_provider(float(key)).ravel() for key in keys[:256]], axis=1)
    result["batched_field_bitwise_equal_per_key_first_256"] = bool(
        np.array_equal(float_bits(batched[:, :256]), float_bits(single)))
    result["batched_field_max_abs_difference_volts"] = float(np.max(np.abs(batched[:, :256] - single)))
    passed = (all(item["digest_equal"] and item["canonical_equal"] and item["fragments"]["equal"]
                  for item in precheck)
              and all(not item["problems"] for item in comparisons)
              and result["prefix_readout_totals_equal"] and not any(round_trip)
              and result["legacy_window_bounds_exact"])
    result["passed"] = bool(passed)
    return result


def part_b(case, plan, readout, clock, *, selected, seed, readout_windows):
    """Full legacy chain: O1 acceptance on selected windows, fragments of all windows."""
    import numpy as np
    from fast_io_exact import CYCLE_COUNT, CYCLE_SECONDS, shift
    from fast_io_legacy_reference import (compare_readout, compare_splits, cursor_for, legacy_fragment_check,
        legacy_window_digest, run_eager_readout, synthetic_hold_means, validator_context)
    from streaming_acquisition_readout import ReadoutWindow
    from streaming_input_plan import bind_hold_windows
    from streaming_midpoint_hold import compile_streaming_midpoint_hold
    t, check = plan.template, clock.check
    result = {}
    clock.set("B: legacy full-record streaming hold plan (first pass)")
    started = time.monotonic()
    full = compile_streaming_midpoint_hold(case.raw_time, case.contacts, case.schedule.physical_action_factory,
        support=case.schedule.acquisition_window, epsilon_input_volts=1e-8, max_intervals=40000000,
        max_work=100000000, check=check, noise_hold_bound=case.noise.hold_bound)
    result["legacy_first_pass_seconds"] = time.monotonic() - started
    legacy_report = dict(full.report)
    result["legacy_hold_report"] = legacy_report
    result["hold_report_equal"] = legacy_report == plan.totals["hold"]
    result["hold_report_differences"] = _differences(legacy_report, plan.totals["hold"])
    retained_path = case_paths().get("attempt002_report")
    with open(retained_path, encoding="utf-8") as stream:
        retained = json.load(stream)["wrapper"]["hold"]
    result["attempt002_retained_hold_report_equal"] = retained == plan.totals["hold"]
    result["attempt002_retained_hold_report_differences"] = _differences(retained, plan.totals["hold"])
    selected = sorted(set(selected))
    legacy_context = validator_context(field_provider=full.field_provider, array=case.array, noise=case.noise,
        limits=case.limits, input_bindings=case.schedule.input_bindings, check=check)
    new_context = validator_context(field_provider=plan.field_provider, array=case.array, noise=case.noise,
        limits=case.limits, input_bindings=case.schedule.input_bindings, check=check)
    q = readout.quantizers[0]

    def readout_factory():
        for k in range(CYCLE_COUNT):
            if k in selected_set:
                yield case.schedule.readout_window(k)
            else:
                # Only index/start/end are used by bind_hold_windows; fragments
                # of these windows are compared, their readout records are not.
                yield ReadoutWindow(k, float(shift(k, 0.0)), float(shift(k, CYCLE_SECONDS)), (), (), (), (), ())

    selected_set = set(selected)
    clock.set("B: legacy second pass with per-window comparison")
    started = time.monotonic()
    factory = bind_hold_windows(full, readout_factory, max_windows=4000, max_window_fragments=200000, check=check)
    events = fragments = windows = 0
    fragment_failures, records = [], []
    for window in factory():
        k = window.index
        check()
        plan_window = plan.window(k)
        check_result = legacy_fragment_check(window, plan_window, t)
        if not check_result["equal"] and len(fragment_failures) < 20:
            fragment_failures.append(dict(window=k, **check_result))
        if k in selected_set:
            readout_window = window.readout_window
            bounds_exact = (readout_window.index, readout_window.start, readout_window.end) == (
                k, float(shift(k, 0.0)), float(shift(k, CYCLE_SECONDS)))
            cursor = cursor_for(k, events, events - 1)
            old = legacy_window_digest(legacy_context, window, cursor)
            new = legacy_window_digest(new_context, plan.legacy_window(k, plan_window), cursor)
            item = dict(window=k, readout_bounds_exact=bounds_exact, digest_equal=old["digest"] == new["digest"],
                        canonical_equal=old["canonical"] == new["canonical"],
                        canonical_bytes=len(old["canonical"]),
                        canonical_sha256=hashlib.sha256(old["canonical"]).hexdigest(),
                        legacy_window_digest=old["digest"].hex(), fragments_equal=check_result["equal"],
                        conversions=len(readout_window.conversions),
                        observations=len(readout_window.observations))
            if k >= readout_windows:
                means = synthetic_hold_means(plan_window, thresholds=q.thresholds, levels=q.levels,
                    input_min=q.input_min_volts, input_max=q.input_max_volts, n_adc=len(t.adc_ids),
                    seed=seed * 10007 + k)
                eager = run_eager_readout(window, input_ids=t.input_ids, input_adc=t.input_adc,
                                          hold_means=means, splitter=SimpleNamespace(_limits=case.limits))
                new_readout = readout.read_window(plan_window, means)
                problems = compare_readout(eager, new_readout, t, eager_order=True)
                problems += compare_splits(eager["bounds"], None, plan_window, t.input_ids)
                item["o2_eager_problems"] = problems
                item["o2_eager_clipped"] = int(new_readout.raw_clipped.sum())
            records.append(item)
            del old, new
        events += len(window.physical_intervals)
        fragments += len(window.physical_intervals)
        windows += 1
    result["legacy_second_pass_seconds"] = time.monotonic() - started
    result["windows_compared"] = windows
    result["fragment_failures"] = fragment_failures
    result["records"] = records
    totals = plan.totals
    result["exact_totals"] = dict(
        windows=(windows, totals["windows"]), physical_input_events=(events, totals["physical_input_events"]),
        physical_fragments=(fragments, totals["physical_fragments"]))
    result["exact_totals_equal"] = all(a == b for a, b in result["exact_totals"].values())
    o2_late = [item for item in records if "o2_eager_problems" in item]
    passed = (result["hold_report_equal"] and result["attempt002_retained_hold_report_equal"]
              and not fragment_failures and windows == CYCLE_COUNT and result["exact_totals_equal"]
              and all(item["digest_equal"] and item["canonical_equal"] and item["fragments_equal"]
                      and item["readout_bounds_exact"] for item in records)
              and all(not item["o2_eager_problems"] for item in o2_late)
              and len(records) == len(selected))
    result["passed"] = bool(passed)
    return result


def case_paths():
    from fast_io_case import EPISODE_ROOT
    return dict(attempt002_report=EPISODE_ROOT / "outputs/runs/model_specification/"
                "conditional_preflight_cost_attempt002/preflight_cost_report.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--random-windows", type=int, default=20)
    parser.add_argument("--readout-windows", type=int, default=8)
    parser.add_argument("--skip-full-chain", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "equivalence_report.json"
    if not run_dir.is_dir() or report_path.exists():
        raise RuntimeError("supply an existing run directory without an equivalence report")
    from fast_io_case import COMPONENT_REPORT, PhaseClock, assert_not_advanced, build_case, build_plan, compute_context, peak_rss_kib
    clock = PhaseClock(args.max_seconds)
    report = dict(status="failed", scope="stage1_O1_O2_equivalence_development_test_not_counted",
                  seed=args.seed, electrical_forward_simulation=False, parts={})

    def interrupt(signum, frame):
        raise TimeoutError(f"equivalence run interrupted by signal {signum}")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, args.max_seconds))
    try:
        report["allocation"] = compute_context()
        import numpy as np
        from fast_io_legacy_reference import load_retained_hold_means
        from fast_io_readout import VectorizedReadout
        case = build_case(clock)
        plan, totals = build_plan(case, clock)
        report["closed_form_totals"] = totals
        readout = VectorizedReadout(plan.template)
        rng = np.random.Generator(np.random.PCG64(args.seed))
        random_windows = sorted(int(k) for k in rng.choice(np.arange(3, 4000), args.random_windows, replace=False))
        # Windows where record IDs gain a decimal digit inside the window: the
        # legacy string-sorted supports and publication ties change order there.
        digit_windows = [3, 22, 38, 222, 384, 2224, 3846]
        selected = [0, 1, 2] + random_windows + digit_windows + [3999]
        report["selected_windows"] = dict(fixed=[0, 1, 2], random=random_windows, digit_transition=digit_windows,
            final=[3999], rule="numpy PCG64(seed).choice(arange(3, 4000), random_windows, replace=False)")
        retained = load_retained_hold_means(COMPONENT_REPORT)
        try:
            report["parts"]["A"] = part_a(case, plan, readout, clock, windows=args.readout_windows,
                                          seed=args.seed, retained=retained)
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            report["parts"]["A"] = dict(passed=False, failure=_failure(clock.phase, error))
        if not args.skip_full_chain:
            report["parts"]["B"] = part_b(case, plan, readout, clock, selected=selected, seed=args.seed,
                                          readout_windows=args.readout_windows)
        assert_not_advanced(case)
        parts = report["parts"]
        report["status"] = ("passed" if all(part.get("passed") for part in parts.values())
                            and (args.skip_full_chain or "B" in parts) else "failed")
    except BaseException as error:
        report["failure"] = _failure(clock.phase, error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report["timings_seconds"] = clock.close()
    report["rss_by_phase"] = clock.rss
    report["elapsed_seconds"] = clock.elapsed
    report["peak_rss_kib"] = peak_rss_kib()
    _write_json(report_path, report)
    print(json.dumps(dict(event="fast_io_equivalence_terminal", status=report["status"],
                          elapsed_seconds=report["elapsed_seconds"], report=str(report_path))), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
