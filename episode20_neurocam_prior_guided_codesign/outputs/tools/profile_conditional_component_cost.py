"""Profile the unchanged full-dimensional first-two-actions component command.

Prepared entrypoint only; invoking it requires a separately selected compute
attempt and an external 300-second timeout. The delegated component retains its
290-second inclusive work timer, original numerical/RSS limits, input realization
and exclusive settings/result files. Profiling overhead is inside that timer.
No physical function, equation or input is replaced. Profile times diagnose work
distribution and cannot be treated as unprofiled throughput.
"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import resource
import sys
import time


TOOLS_ROOT = Path(__file__).resolve().parent
EPISODE_ROOT = TOOLS_ROOT.parents[1]
MAX_PROFILE_BYTES = 2 * 1024**2
PROFILE_NAME = "component_function_profile.json"
SECTION_LIMITS = dict(ep20_functions=900, numpy_scipy_functions=300,
    compiled_numeric_calls=200, other_hot_functions=100)
NUMERIC_MARKERS = ("numpy", "scipy", "ndarray", "ufunc", "SuperLU", "fblas",
    "flapack", "umath", "multiarray", "_sparsetools")


def _profile_report(profiler, *, allocation, run_dir, component_exit_code,
                    component_exception, component_elapsed, wrapper_started):
    stats = pstats.Stats(profiler)
    sections = {name: [] for name in SECTION_LIMITS}
    for (filename, lineno, name), (primitive, calls, self_seconds, cumulative, _) in stats.stats.items():
        if filename.startswith(str(TOOLS_ROOT) + os.sep):
            section = "ep20_functions"
        elif "/numpy/" in filename or "/scipy/" in filename:
            section = "numpy_scipy_functions"
        elif filename == "~" and any(marker in name for marker in NUMERIC_MARKERS):
            section = "compiled_numeric_calls"
        else:
            section = "other_hot_functions"
        sections[section].append(dict(filename=filename, lineno=lineno, function=name,
            primitive_calls=primitive, total_calls=calls, self_seconds=self_seconds,
            cumulative_seconds=cumulative))
    available = {name: len(rows) for name, rows in sections.items()}
    for name, rows in sections.items():
        rows.sort(key=lambda row: (-row["cumulative_seconds"], -row["self_seconds"],
            row["filename"], row["lineno"], row["function"]))
        del rows[SECTION_LIMITS[name]:]
    return dict(scope="profiled_original_component_engineering_cost_only",
        allocation=allocation, delegated_entrypoint=str(TOOLS_ROOT / "run_conditional_component_cost.py"),
        delegated_arguments=["--first-two-actions", "--radau-preconditioner", "contact_block",
            "--run-dir", str(run_dir)],
        component_report=str(run_dir / "component_cost_report.json"),
        component_exit_code=component_exit_code, component_exception=component_exception,
        profiled_component_elapsed_seconds=component_elapsed,
        wrapper_elapsed_seconds_before_profile_serialization=time.monotonic() - wrapper_started,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        component_inclusive_work_seconds=290., required_external_timeout_seconds=300.,
        unchanged_component_maximum_peak_rss_kib=13 * 1024**2 // 2,
        profiled_total_calls=stats.total_calls, profiled_primitive_calls=stats.prim_calls,
        total_profiled_self_seconds=stats.total_tt,
        section_selection="highest_cumulative_then_self_time_within_each_disjoint_section",
        available_functions_by_section=available, section_row_limits=SECTION_LIMITS,
        functions=sections,
        interpretation_limits=[
            "cProfile overhead consumes the unchanged component time and RSS allowance",
            "component setup and scientific imports are included; this is not steady-state solver throughput",
            "cumulative times nest and are nonadditive across functions and sections",
            "only calls visible to cProfile appear; untraced compiled kernels remain in their caller time",
            "self time excludes traced callees but does not identify every underlying compiled operation",
            "contact preparation, for_step factor/cache handling, apply, C/G products, Krylov, forcing and physical work remain distinguishable by their original function keys",
            "a timeout can leave only a partial component frontier; the original component report is authoritative",
            "profiled elapsed times are not directly comparable to unprofiled attempt throughput"],
        calibrated_reference=False, candidate_scored=False, full_record_completed=False)


def _write_profile(path, report):
    # Limits ordinarily leave ample room. If long function names exhaust the
    # byte bound, keep the hottest half of each section and report the omission.
    report["byte_limit_truncation_applied"] = False
    while True:
        report["retained_functions_by_section"] = {
            name: len(rows) for name, rows in report["functions"].items()}
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if len(encoded.encode("utf-8")) <= MAX_PROFILE_BYTES:
            break
        if not any(report["functions"].values()):
            raise ValueError("profile metadata alone exceeds its two MiB limit")
        for rows in report["functions"].values():
            del rows[len(rows) // 2:]
        report["byte_limit_truncation_applied"] = True
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    wrapper_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # The imported boundary helper contains only standard-library definitions.
    # Verify this exact shell before enabling profiling or numerical imports.
    from run_conditional_forward_cost import _compute_context
    allocation = _compute_context()
    if os.environ.get("BLIS_NUM_THREADS") != "1":
        raise RuntimeError("BLIS_NUM_THREADS must also be 1 before numerical imports")
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs") or not run_dir.is_dir()
            or any((run_dir / name).exists() for name in
                ("settings.json", "component_cost_report.json", PROFILE_NAME))):
        raise RuntimeError("supply a fresh existing episode run directory; outputs are exclusive")
    from run_conditional_component_cost import main as component_main
    component_arguments = ["--first-two-actions", "--radau-preconditioner", "contact_block",
        "--run-dir", str(run_dir)]
    profiler = cProfile.Profile()
    component_exit_code = component_exception = None
    component_started = time.monotonic()
    try:
        profiler.enable()
        component_exit_code = component_main(component_arguments)
        return component_exit_code
    except BaseException as error:
        component_exception = dict(exception_type=type(error).__name__, message=str(error)[:1000])
        raise
    finally:
        profiler.disable()
        component_elapsed = time.monotonic() - component_started
        try:
            report = _profile_report(profiler, allocation=allocation, run_dir=run_dir,
                component_exit_code=component_exit_code, component_exception=component_exception,
                component_elapsed=component_elapsed, wrapper_started=wrapper_started)
            _write_profile(run_dir / PROFILE_NAME, report)
            print(json.dumps(dict(event="component_profile_retained", path=str(run_dir / PROFILE_NAME),
                component_exit_code=component_exit_code,
                elapsed_seconds=time.monotonic() - wrapper_started)), flush=True)
        except BaseException as error:
            # Ancillary serialization must not replace the original command's
            # exit code or exception. Its settings and result remain untouched.
            print(json.dumps(dict(event="component_profile_not_retained",
                exception_type=type(error).__name__, message=str(error)[:1000],
                component_exit_code=component_exit_code)), file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
