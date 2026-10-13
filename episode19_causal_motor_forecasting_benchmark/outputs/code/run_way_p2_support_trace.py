#!/usr/bin/env python3
"""One fixed WAY P2/S1 support pass with durable, value-preserving stage traces."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time


OUTPUTS = Path(__file__).resolve().parents[1]
RUNS = OUTPUTS / "runs/source_views_and_timing"
MANIFEST = RUNS / "job_46425841_development_view/development_view.json"
REPORT_NAME = "p2_support_trace.json"
MAX_SECONDS = 180
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, report):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def execution_guard(args):
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("This selected pass requires exactly one allocated CPU")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise RuntimeError("All six numerical thread counts must be one")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        raise ValueError("Whole-process cap must be positive and at most 180 seconds")
    if RUNS.resolve() not in args.output_dir.resolve().parents:
        raise ValueError("Output must be a fresh EP19 source_views_and_timing directory")
    expected = (Path(os.environ["SCRATCH"]) / "br_autoresearch" /
                "episode19_causal_motor_forecasting_benchmark" /
                "development_view_job_46425841")
    if args.development_view.resolve() != expected.resolve():
        raise ValueError("Only the existing job_46425841 development view is selected")


@contextmanager
def stage_bindings(panel, stage):
    """Wrap entry/return only; pass original arguments and results unchanged."""
    names = ("loadmat", "select_columns", "decode_fixed_half_level_cue",
             "sensor_summaries", "fit_pilot_noise")
    originals = {name: getattr(panel, name) for name in names}
    selection = 0

    def traced_load(*args, **kwargs):
        return stage("MAT_hs_load", originals["loadmat"], *args, **kwargs)

    def traced_select(*args, **kwargs):
        nonlocal selection
        labels = ("select_five_EMG_channels", "select_nine_hand_position_channels",
                  "select_ParticipantLED")
        if selection >= len(labels):
            raise RuntimeError("Unexpected additional channel-selection call")
        label = labels[selection]
        selection += 1
        return stage(label, originals["select_columns"], *args, **kwargs)

    def traced_cue(*args, **kwargs):
        return stage("fixed_half_level_cue_decode", originals["decode_fixed_half_level_cue"],
                     *args, **kwargs)

    def traced_summaries(*args, **kwargs):
        return stage("unchanged_sensor_summaries", originals["sensor_summaries"],
                     *args, **kwargs)

    def traced_support(*args, **kwargs):
        if kwargs.get("support_only") is not True:
            raise RuntimeError("Only the existing support-only branch is selected")
        return stage("unchanged_stationarity_support", originals["fit_pilot_noise"],
                     *args, **kwargs)

    replacements = (traced_load, traced_select, traced_cue, traced_summaries, traced_support)
    try:
        for name, replacement in zip(names, replacements):
            setattr(panel, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(panel, name, original)


def run(args):
    started = time.monotonic()
    cpu_started = time.process_time()
    execution_guard(args)
    deadline = started + args.max_seconds
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / REPORT_NAME
    report = {
        "analysis_id": "source_views_and_timing",
        "phase": "fixed_P2_S1_instrumented_conditional_support",
        "status": "running", "complete": False,
        "started_utc": utc_now(), "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
        "allocated_cpus": 1, "thread_environment": {key: os.environ[key] for key in THREADS},
        "internal_deadline_seconds": args.max_seconds,
        "manifest": str(MANIFEST), "participant": 2, "series": 1,
        "source_role": "fit_and_inner_selection", "source_manifest_validated": False,
        "source": str(args.development_view / "way/fit_and_inner_selection/HS_P2_S1.mat"),
        "source_unit_cue_threshold": .5, "settings": None, "primary_paths": None,
        "stages": [], "result": None,
        "MAT_load_entry_recorded": False, "MAT_load_returned": False,
        "EEG_array_incidentally_deserialized": None,
        "EEG_access_note": "An unfinished whole-hs load may already have decoded EEG; unknown until return.",
        "stage_started_means": "Checkpoint immediately before a call; not proof of bytes read or call completion.",
        "P1_source_payload_reopened": False, "other_participant_payload_access": False,
        "feedback_or_held_out_payload_access": False,
        "noise_thresholds_estimated_or_frozen": False,
        "onset_or_risk_labels_computed": False, "EEG_features_or_forecast_scores_computed": False,
        "waveforms_reported": False, "scientific_labels_usable": False,
        "known_return_authenticated": False, "combined_onset_timing_bound_ms": None,
        "selected_file_polarity_supported_by_official_durations": False,
        "valid_forecast_trials": 0, "automatic_retry": False, "follow_on_selected": False,
        "bindings_restored": False, "signal_handlers_restored": False,
        "cost_accounting": "Shared parent is charged once by the coordinator under EP04; no additional parent charge here.",
    }

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Selected P2/S1 whole-process deadline reached")

    def publish():
        report.update(elapsed_seconds=time.monotonic() - started,
                      process_cpu_seconds=time.process_time() - cpu_started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write(destination, report)

    def stage(name, function, *call_args, **call_kwargs):
        check()
        item = {"stage": name, "status": "started", "started_utc": utc_now(),
                "started_elapsed_seconds": time.monotonic() - started,
                "call_returned": False}
        report["stages"].append(item)
        if name == "MAT_hs_load":
            report["MAT_load_entry_recorded"] = True
        publish()
        check()
        stage_started, stage_cpu = time.monotonic(), time.process_time()
        try:
            value = function(*call_args, **call_kwargs)
            item["call_returned"] = True
            if name == "MAT_hs_load":
                report["MAT_load_returned"] = True
                report["EEG_array_incidentally_deserialized"] = (
                    isinstance(value.get("hs"), dict) and "eeg" in value["hs"])
            elif name == "fixed_half_level_cue_decode":
                report["returned_cue_metadata"] = value[1]
            elif name == "unchanged_stationarity_support":
                report["returned_stationary_support"] = value[1]
            elif name == "existing_one_file_P2_S1":
                report["result"] = value
            check()
            item["status"] = "returned"
        except BaseException as error:
            item.update(status="interrupted_or_failed", error_type=type(error).__name__,
                        error=str(error))
            raise
        finally:
            item.update(elapsed_seconds=time.monotonic() - stage_started,
                        process_cpu_seconds=time.process_time() - stage_cpu)
            publish()
        check()
        return value

    def stop(signum, _frame):
        raise TimeoutError(f"Selected P2/S1 pass received signal {signum}")

    prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    success = False
    try:
        for sig in prior_handlers:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
        publish()
        check()

        def admit_source():
            manifest = json.loads(MANIFEST.read_text())
            matching = [row for row in manifest["files"]
                        if row["source"] == "way" and row["member"] == "HS_P2_S1.mat"]
            source = Path(report["source"])
            if (len(matching) != 1 or matching[0]["role"] != "fit_and_inner_selection"
                    or Path(matching[0]["destination"]).resolve() != source.resolve()
                    or not 0 < matching[0]["bytes"] <= 80 * 1024**2
                    or source.stat().st_size != matching[0]["bytes"]):
                raise ValueError("Exact existing P2/S1 fitting manifest identity/size mismatch")
            report.update(source_manifest_validated=True, source_bytes=matching[0]["bytes"])
            return source

        source = stage("P2_manifest_and_source_size_admission", admit_source)

        def import_panel():
            import check_way_rest_support_panel
            return check_way_rest_support_panel

        panel = stage("existing_numpy_scipy_reader_import", import_panel)
        settings = panel.PilotSettings()
        report.update(settings=settings.record(), primary_paths=panel.PATH_NAMES)
        try:
            with stage_bindings(panel, stage):
                report["result"] = stage("existing_one_file_P2_S1", panel.one_file,
                                         source, 2, settings)
        finally:
            report["bindings_restored"] = True
        check()
        report.update(status="complete_P2_S1_conditional_support_only", complete=True,
                      finished_utc=utc_now())
        publish()
        check()
        success = True
    except BaseException as error:
        report.update(status="P2_S1_support_pass_incomplete", complete=False,
                      error_type=type(error).__name__, error=str(error), finished_utc=utc_now())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
        report["signal_handlers_restored"] = True
        if time.monotonic() >= deadline:
            success = False
            report.update(status="P2_S1_support_pass_incomplete", complete=False,
                          error_type="TimeoutError", error="Whole-process deadline reached during finalization",
                          finished_utc=utc_now())
        publish()
        # A returned final write is part of the cap; retain completed stages but
        # correct the final claim if publication itself used the remaining time.
        if success and time.monotonic() >= deadline:
            success = False
            report.update(status="P2_S1_support_pass_incomplete", complete=False,
                          error_type="TimeoutError", error="Whole-process deadline reached during final publication",
                          finished_utc=utc_now())
            publish()
    print(json.dumps({key: report[key] for key in
                      ("status", "complete", "elapsed_seconds", "MAT_load_returned",
                       "source_manifest_validated", "signal_handlers_restored")}))
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
