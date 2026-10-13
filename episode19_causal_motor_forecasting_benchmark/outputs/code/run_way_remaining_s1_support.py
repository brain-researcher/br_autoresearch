#!/usr/bin/env python3
"""Fixed remaining WAY S1 support: P3--P12, no P1/P2 reread or follow-on."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

# This module has only standard-library imports; its scientific imports occur
# inside its run function, which is not called here.
from run_way_p2_support_trace import MANIFEST, RUNS, THREADS, atomic_write, stage_bindings


PARTICIPANTS = tuple(range(3, 13))
REPORT_NAME = "remaining_s1_support.json"
MAX_SECONDS = 600


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    started, cpu_started = time.monotonic(), time.process_time()
    if (not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname()
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or any(os.environ.get(key) != "1" for key in THREADS)):
        raise RuntimeError("Use one actual compute CPU and all six thread limits equal to one")
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= MAX_SECONDS:
        raise ValueError("Whole-process cap must be positive and at most 600 seconds")
    expected = (Path(os.environ["SCRATCH"]) / "br_autoresearch" /
                "episode19_causal_motor_forecasting_benchmark/development_view_job_46425841")
    if args.development_view.resolve() != expected.resolve():
        raise ValueError("Only the preserved job_46425841 development view is selected")
    if RUNS.resolve() not in args.output_dir.resolve().parents:
        raise ValueError("Use a fresh EP19 source_views_and_timing output directory")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    deadline = started + args.max_seconds
    report = {
        "analysis_id": "source_views_and_timing", "phase": "remaining_fixed_S1_conditional_support",
        "status": "running", "complete": False, "started_utc": utc_now(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ.get("SLURM_STEP_ID"),
        "host": socket.gethostname(), "allocated_cpus": 1,
        "thread_environment": {key: os.environ[key] for key in THREADS},
        "internal_deadline_seconds": args.max_seconds, "manifest": str(MANIFEST),
        "original_fixed_panel_participants": list(range(1, 13)),
        "selected_participants": list(PARTICIPANTS), "series": 1,
        "source_role": "fit_and_inner_selection", "all_selected_sources_admitted": False,
        "preserved_reports_not_reread": {
            "1": str(RUNS / "20261003-peripheral-pilot02/peripheral_pilot_report.json"),
            "2": str(RUNS / "20261005-P2-support-46658777/p2_support_trace.json"),
        },
        "P1_or_P2_payload_reopened": False,
        "settings": None, "primary_paths": None, "source_unit_cue_threshold": .5,
        "preparation_stages": [], "current_participant": None,
        "results": [{"participant": person, "series": 1, "status": "not_started",
                     "complete": False, "source_manifest_validated": False,
                     "source": str(args.development_view / "way/fit_and_inner_selection" /
                                   f"HS_P{person}_S1.mat"),
                     "report": str(args.output_dir / f"P{person}_S1/support_trace.json"),
                     "stages": [], "result": None,
                     "MAT_load_entry_recorded": False, "MAT_load_returned": False,
                     "EEG_array_incidentally_deserialized": None,
                     "bindings_restored": False} for person in PARTICIPANTS],
        "stage_started_means": "Checkpoint before a call, not proof of bytes read or completion.",
        "unfinished_MAT_access": "An interrupted whole-hs load may already have decoded EEG; extent is unknown.",
        "successful_subset_selected": False, "all_participant_outcomes_retained": True,
        "feedback_or_held_out_payload_access": False, "waveforms_reported": False,
        "noise_thresholds_estimated_or_frozen": False, "onset_or_risk_labels_computed": False,
        "EEG_features_or_forecast_scores_computed": False, "scientific_labels_usable": False,
        "known_return_authenticated": False, "combined_onset_timing_bound_ms": None,
        "selected_file_polarity_supported_by_official_durations": False,
        "valid_forecast_trials": 0, "automatic_retry": False, "follow_on_selected": False,
        "signal_handlers_restored": False,
        "cost_accounting": "Coordinator owns shared allocation accounting; do not charge its parent again here.",
    }
    active = None

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Remaining S1 support whole-process deadline reached")

    def publish_panel():
        report.update(elapsed_seconds=time.monotonic() - started,
                      process_cpu_seconds=time.process_time() - cpu_started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write(args.output_dir / REPORT_NAME, report)

    def publish_active():
        if active is None:
            publish_panel()
        else:
            active["updated_elapsed_seconds"] = time.monotonic() - started
            atomic_write(Path(active["report"]), active)

    def stage(name, function, *values, **keywords):
        check()
        item = {"stage": name, "status": "started", "call_returned": False,
                "started_utc": utc_now(), "started_elapsed_seconds": time.monotonic() - started}
        target = report["preparation_stages"] if active is None else active["stages"]
        target.append(item)
        if active is not None and name == "MAT_hs_load":
            active["MAT_load_entry_recorded"] = True
        publish_active()
        check()
        begin, cpu_begin = time.monotonic(), time.process_time()
        try:
            value = function(*values, **keywords)
            item["call_returned"] = True
            if active is not None:
                if name == "MAT_hs_load":
                    active["MAT_load_returned"] = True
                    active["EEG_array_incidentally_deserialized"] = (
                        isinstance(value.get("hs"), dict) and "eeg" in value["hs"])
                elif name == "fixed_half_level_cue_decode":
                    active["returned_cue_metadata"] = value[1]
                elif name == "unchanged_stationarity_support":
                    active["returned_stationary_support"] = value[1]
                elif name == "existing_one_file":
                    active["result"] = value
            check()
            item["status"] = "returned"
        except BaseException as error:
            item.update(status="interrupted_or_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            item.update(elapsed_seconds=time.monotonic() - begin,
                        process_cpu_seconds=time.process_time() - cpu_begin)
            publish_active()
        check()
        return value

    def stop(signum, _frame):
        raise TimeoutError(f"Remaining S1 support interrupted by signal {signum}")

    prior_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    success = False
    try:
        for sig in prior_handlers:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
        publish_panel()
        manifest = stage("read_original_development_manifest",
                         lambda: json.loads(MANIFEST.read_text()))
        for row in report["results"]:
            def admit(row=row):
                name = f"HS_P{row['participant']}_S1.mat"
                matches = [entry for entry in manifest["files"]
                           if entry["source"] == "way" and entry["member"] == name]
                source = Path(row["source"])
                if (len(matches) != 1 or matches[0]["role"] != "fit_and_inner_selection"
                        or Path(matches[0]["destination"]).resolve() != source.resolve()
                        or not 0 < matches[0]["bytes"] <= 80 * 1024**2
                        or source.stat().st_size != matches[0]["bytes"]):
                    raise ValueError(f"Exact existing fitting manifest identity/size mismatch for {name}")
                row.update(source_manifest_validated=True, source_bytes=matches[0]["bytes"])
            stage(f"admit_P{row['participant']}_S1", admit)
        report["all_selected_sources_admitted"] = True

        def import_panel():
            import check_way_rest_support_panel
            return check_way_rest_support_panel

        panel = stage("existing_numpy_scipy_reader_import", import_panel)
        settings = panel.PilotSettings()
        report.update(settings=settings.record(), primary_paths=panel.PATH_NAMES)
        for row in report["results"]:
            check()
            active = row
            Path(row["report"]).parent.mkdir(exist_ok=False)
            file_started, file_cpu = time.monotonic(), time.process_time()
            row.update(status="running", started_utc=utc_now())
            report["current_participant"] = row["participant"]
            publish_panel()
            publish_active()
            try:
                with stage_bindings(panel, stage):
                    stage("existing_one_file", panel.one_file,
                          Path(row["source"]), row["participant"], settings)
                row["bindings_restored"] = True
                check()
                row.update(status="complete_conditional_support_only", complete=True)
            except BaseException as error:
                row.update(status="file_processing_failure", complete=False,
                           error_type=type(error).__name__, error=str(error))
                if isinstance(error, (TimeoutError, KeyboardInterrupt, SystemExit)):
                    raise
                # A file-specific technical failure cannot silently remove this
                # person or stop the other fixed files while time remains.
            finally:
                row.update(bindings_restored=True, finished_utc=utc_now(),
                           elapsed_seconds=time.monotonic() - file_started,
                           process_cpu_seconds=time.process_time() - file_cpu)
                publish_active()
                publish_panel()
            check()
            active = None
        report["current_participant"] = None
        success = all(row["complete"] for row in report["results"])
        report.update(status=("complete_remaining10_conditional_support_panel" if success
                              else "panel_with_file_failures_all_outcomes_retained"), complete=success)
        publish_panel()
        check()
    except BaseException as error:
        success = False
        report.update(status="remaining_S1_panel_incomplete", complete=False,
                      error_type=type(error).__name__, error=str(error))
        if active is not None and active["status"] == "complete_conditional_support_only":
            # This file's publication/check has not returned within the cap.
            # Preserve its returned scientific result without a completion claim.
            active.update(status="file_processing_failure", complete=False,
                          error_type=type(error).__name__, error=str(error))
            publish_active()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
        report["signal_handlers_restored"] = True
        for row in report["results"]:
            if row["status"] == "not_started":
                row["status"] = "not_completed_before_failure"
        if time.monotonic() >= deadline:
            success = False
            report.update(status="remaining_S1_panel_incomplete", complete=False,
                          error_type="TimeoutError", error="Deadline reached during finalization")
        report["finished_utc"] = utc_now()
        publish_panel()
        if success and time.monotonic() >= deadline:
            success = False
            report.update(status="remaining_S1_panel_incomplete", complete=False,
                          error_type="TimeoutError", error="Deadline reached during final publication")
            publish_panel()
    print(json.dumps({key: report[key] for key in
                      ("status", "complete", "elapsed_seconds", "all_selected_sources_admitted")}))
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
