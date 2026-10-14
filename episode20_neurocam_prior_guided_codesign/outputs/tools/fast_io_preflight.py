"""Measure the new O1 preflight for the full 16-second B2/2001 schedule.

Planning only: legacy setup (world, circuit, template, noise binding), then
typed template arrays, descriptor digests, the vectorized full-record proof
with closed-form holds, the unchanged legacy per-window validators on windows
0-2 and 3999, and scalar-bound spot checks. No writer, electrical advance,
readout value or score. Writes one exclusive terminal JSON on every exit path.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


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


def dict_differences(left, right):
    keys = sorted(set(left) | set(right))
    return [key for key in keys if left.get(key, "<missing>") != right.get(key, "<missing>")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "preflight_report.json"
    if not run_dir.is_dir() or report_path.exists():
        raise RuntimeError("supply an existing run directory without a preflight report")

    from fast_io_case import PhaseClock, build_case, build_plan, compute_context, peak_rss_kib, assert_not_advanced
    clock = PhaseClock(args.max_seconds)
    report = dict(status="preflight_failed", scope="O1_new_preflight_full_16_second_schedule_planning_only",
                  electrical_state_advanced=False, writer_created=False, readout_values_produced=False)

    def interrupt(signum, frame):
        raise TimeoutError(f"preflight interrupted by signal {signum}")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, args.max_seconds))
    try:
        report["allocation"] = compute_context()
        case = build_case(clock)
        setup_seconds = clock.elapsed
        rss_before_planner = peak_rss_kib()
        planner_started = time.monotonic()
        plan, totals = build_plan(case, clock)
        clock.set("O1 legacy validators on template windows 0-2 and final window 3999")
        from fast_io_planner import legacy_window_validation, scalar_bound_spot_check
        legacy = legacy_window_validation(plan, array=case.array, noise=case.noise, limits=case.limits,
                                          input_bindings=case.schedule.input_bindings)
        clock.set("O1 scalar-bound spot checks")
        spot = scalar_bound_spot_check(plan, (0, 1, 2, 3999))
        clock.set("O1 descriptor re-verification and totals")
        prefix = plan.readout_totals(3)
        if prefix != legacy["dry_readout_report"]:
            raise AssertionError("closed-form prefix readout totals differ from the legacy dry readout: "
                                 + repr({key: (prefix.get(key), legacy["dry_readout_report"].get(key))
                                         for key in dict_differences(prefix, legacy["dry_readout_report"])}))
        if spot["mismatches"]:
            raise AssertionError(f"scalar-bound spot check mismatches: {spot['mismatches']}")
        if not plan.verify_descriptors(case.raw_time, case.contacts):
            raise AssertionError("field or noise descriptors changed during planning")
        assert_not_advanced(case)
        planner_seconds = time.monotonic() - planner_started
        report.update(status="preflight_complete", totals=totals, legacy_validation=dict(
            window_digests=legacy["window_digests"],
            maximum_window_numeric_bytes=legacy["maximum_window_numeric_bytes"],
            prefix_dry_readout_report_matches_closed_form=True,
            cursor_after_prefix=legacy["cursor_after_prefix"]),
            scalar_bound_spot_check=spot, setup_seconds=setup_seconds,
            planner_seconds=planner_seconds, peak_rss_kib_before_planner=rss_before_planner,
            legacy_reference_seconds=dict(attempt001_hold_preparation=960.5817,
                attempt001_whole_plan_preflight_unfinished=4987.7855,
                attempt002_hold_preparation=968.538697, attempt002_whole_plan_preflight_unfinished=4979.448215))
    except BaseException as error:
        report["failure"] = dict(phase=clock.phase, exception_type=type(error).__name__,
                                 message=str(error)[:2000], traceback=traceback.format_exc()[-6000:])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    report["timings_seconds"] = clock.close()
    report["rss_by_phase"] = clock.rss
    report["elapsed_seconds"] = clock.elapsed
    report["peak_rss_kib"] = peak_rss_kib()
    _write_json(report_path, report)
    print(json.dumps(dict(event="fast_io_preflight_terminal", status=report["status"],
                          elapsed_seconds=report["elapsed_seconds"], report=str(report_path))), flush=True)
    return 0 if report["status"] == "preflight_complete" else 2


if __name__ == "__main__":
    sys.exit(main())
