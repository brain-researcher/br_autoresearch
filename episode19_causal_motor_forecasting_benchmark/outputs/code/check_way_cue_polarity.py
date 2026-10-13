#!/usr/bin/env python3
"""Compare fitting-series official LED metadata with raw LED edges only.

This source-semantics check is never a predictor/event-label export. It does
not authenticate completed return, a production detector, or total timing.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time
import zipfile

import numpy as np
from scipy.io import loadmat

from pilot_way_peripheral_onset import select_columns
from read_way_cue_metadata import read_cues


def residual_summary(observed, reference):
    residual_ms = (observed - reference) * 1000.
    return {"count": len(residual_ms), "signed_min_ms": float(np.min(residual_ms)),
            "signed_median_ms": float(np.median(residual_ms)),
            "signed_max_ms": float(np.max(residual_ms)),
            "max_absolute_ms": float(np.max(np.abs(residual_ms))),
            "mean_absolute_ms": float(np.mean(np.abs(residual_ms)))}


def deadline(signum, frame):
    raise TimeoutError("180-second cue metadata comparison bound reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("coordinator-owned compute allocation required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("all numeric thread counts must be one")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(180)
    source = Path(os.environ["OAK"]) / "users/zijiao/br_autoresearch_data/steward_acquisition/ep19_public_sources/archives/way/P1.zip"
    report = {
        "analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "status": "cue_metadata_comparison_started", "archive": str(source), "member": "WS_P1_S1.mat",
        "allowed_WS_numeric_fields": ["ws.win.trial_start_time", "ws.win.LEDon", "ws.win.LEDoff"],
        "source_role": "P1_S1_fit_and_inner_selection", "held_out_payload_access": False,
        "feedback_payload_access": False, "WS_waveform_arrays_decoded": False,
        "WS_bytes_including_skipped_waveform_bytes_decompressed": False,
        "HS_nested_EEG_incidentally_deserialized": False, "EEG_features_or_scores_computed": False,
        "provider_movement_events_used": False, "provider_LED_metadata_used_as_predictors": False,
        "production_decoder_changed": False, "online_risk_contract_qualified": False,
        "known_end_return_phase_authenticated": False, "scientific_labels_usable": False,
        "total_onset_uncertainty_verified": False, "combined_worst_case_timing_bound_ms": None,
        "comparison_threshold_source_units": .5,
        "comparison_has_no_offset_fitting_or_event_subsetting": True,
        "time_convention": "HS_first_sample_t0; official_HS_event_time=trial_start_time+LED_relative_time",
        "official_conversion_reference": "https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_GetEventsInHS.m",
    }
    exit_code = 0
    try:
        with zipfile.ZipFile(source) as archive:
            info = archive.getinfo("WS_P1_S1.mat")
            if info.file_size != 50960640:
                raise ValueError("Selected WS member size differs from the existing source inventory")
            with archive.open(info) as stream:
                payload = stream.read(50960641)
        if len(payload) != 50960640:
            raise ValueError("Incomplete selected WS member")
        report["WS_bytes_including_skipped_waveform_bytes_decompressed"] = True
        trials, reader_report = read_cues(payload)
        del payload
        report["field_reader"] = reader_report
        official_on = np.array([trial["trial_start_time"] + trial["LEDon"] for trial in trials])
        official_off = np.array([trial["trial_start_time"] + trial["LEDoff"] for trial in trials])
        if not (np.all(official_on < official_off) and np.all(np.diff(official_on) > 0)
                and np.all(official_off[:-1] < official_on[1:])):
            raise ValueError("Official LED intervals are not ordered, nonoverlapping complete pairs")
        report["official_LED_pair_count"] = len(trials)
        report["all_official_LEDon_relative_times_equal_two_seconds"] = all(trial["LEDon"] == 2 for trial in trials)
        del trials
        hs_source = args.development_view / "way/fit_and_inner_selection/HS_P1_S1.mat"
        report["HS_source"] = str(hs_source)
        content = loadmat(hs_source, variable_names=["hs"], simplify_cells=True)
        hs = content["hs"]
        report["HS_nested_EEG_incidentally_deserialized"] = True
        if int(hs["participant"]) != 1 or int(hs["series"]) != 1 or float(hs["misc"]["samplingrate"]) != 500.:
            raise ValueError("Selected HS identity or LED sampling rate differs")
        led = select_columns(hs["misc"], ["ParticipantLED"])[:, 0].copy()
        del hs, content
        if not np.all(np.isfinite(led)):
            raise ValueError("Unknown LED samples prevent a complete direct edge comparison")
        above = led >= .5
        up = (np.flatnonzero(~above[:-1] & above[1:]) + 1) / 500.
        down = (np.flatnonzero(above[:-1] & ~above[1:]) + 1) / 500.
        report["raw_upcrossing_count"] = len(up)
        report["raw_downcrossing_count"] = len(down)
        if len(up) == len(down) == len(official_on):
            report["higher_level_equals_illuminated_comparison"] = {
                "up_vs_official_on": residual_summary(up, official_on),
                "down_vs_official_off": residual_summary(down, official_off),
            }
            report["opposite_polarity_comparison"] = {
                "down_vs_official_on": residual_summary(down, official_on),
                "up_vs_official_off": residual_summary(up, official_off),
            }
            report["status"] = "complete_fitting_file_LED_polarity_comparison"
        else:
            report["status"] = "unresolved_LED_count_mismatch_no_subset_or_matching_fit"
    except Exception as error:
        report.update(status="cue_metadata_comparison_execution_failure", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output_dir / "cue_polarity_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "max_rss_kib",
                                                 "WS_waveform_arrays_decoded", "held_out_payload_access")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
