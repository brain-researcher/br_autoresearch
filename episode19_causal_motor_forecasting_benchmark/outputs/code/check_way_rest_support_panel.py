#!/usr/bin/env python3
"""Fixed S1 panel for all12 WAY people, reusing P1 and reading only fitting files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np
from scipy.io import loadmat

from peripheral_onset import (PATH_NAMES, PilotSettings, decode_fixed_half_level_cue,
                              fit_pilot_noise, sensor_summaries)
from pilot_way_peripheral_onset import select_columns


MAX_SOURCE_BYTES = 80 * 1024**2


def one_file(source, participant, settings):
    """Whole classic MAT decoding is incidental; no EEG feature or score is made."""
    content = loadmat(source, variable_names=["hs"], simplify_cells=True)
    hs = content["hs"]
    del hs["eeg"]
    if int(hs["participant"]) != participant or int(hs["series"]) != 1:
        raise ValueError("Fixed fitting panel source identity mismatch")
    rates = {key: float(hs[key]["samplingrate"]) for key in ("emg", "kin", "misc")}
    if rates != {"emg": 4000., "kin": 500., "misc": 500.}:
        raise ValueError("Unexpected native sensor sampling rates")
    emg = select_columns(hs["emg"], ["Anterior Deltoid", "Brachoradial", "Flexor Digitorum",
                         "Common Extensor Digitorum", "First Dorsal Interosseus"])
    hand = select_columns(hs["kin"], [f"P{axis}{sensor} - position {axis} sensor {sensor}"
                          for sensor in (2, 3, 4) for axis in ("x", "y", "z")]).reshape(-1, 3, 3)
    led = select_columns(hs["misc"], ["ParticipantLED"])[:, 0]
    if len(led) != len(hand) or abs(len(emg)/4000. - len(hand)/500.) > .002:
        raise ValueError("Native peripheral durations disagree beyond one 500Hz tick")
    del hs, content
    cue, cue_report = decode_fixed_half_level_cue(led)
    features = sensor_summaries(emg, hand, settings)
    _, support = fit_pilot_noise(features, emg, hand, cue, settings, support_only=True)
    return {"participant": participant, "series": 1, "status": "complete_conditional_support_only",
            "source": str(source), "source_payload_reused": False,
            "EEG_array_incidentally_deserialized": True,
            "selected_file_polarity_supported_by_official_durations": False,
            "primary_nonfinite_values": int(np.sum(~np.isfinite(emg)) + np.sum(~np.isfinite(hand))),
            "cue": cue_report, "stationary_noise": support}


def deadline(signum, frame):
    raise TimeoutError("Selected fitting-only panel deadline reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=240)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use root's verified compute allocation")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("All numeric thread counts must be one")
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("Deadline must be between1 and540seconds")
    outputs = Path(__file__).resolve().parents[1]
    if outputs / "runs/source_views_and_timing" not in args.output_dir.resolve().parents:
        raise ValueError("Output must be a new EP19 source_views_and_timing attempt")
    manifest = json.loads((outputs / "runs/source_views_and_timing/job_46425841_development_view/development_view.json").read_text())
    settings = PilotSettings()
    reused_path = outputs / "runs/source_views_and_timing/20261003-peripheral-pilot02/peripheral_pilot_report.json"
    reused = json.loads(reused_path.read_text())
    if (reused["settings"] != settings.record() or reused["status"] != "peripheral_pilot_unresolved_noise_support"
            or reused["stationary_noise"]["thresholds_frozen_for_scoring"]
            or reused["source_unit_cue_threshold"] != .5):
        raise ValueError("P1_S1 reuse does not match the prospectively adopted panel")
    sources = []
    for participant in range(2, 13):
        name = f"HS_P{participant}_S1.mat"
        matches = [row for row in manifest["files"] if row["source"] == "way" and row["member"] == name]
        source = args.development_view / "way/fit_and_inner_selection" / name
        if (len(matches) != 1 or matches[0]["role"] != "fit_and_inner_selection"
                or Path(matches[0]["destination"]).resolve() != source.resolve()
                or not 0 < matches[0]["bytes"] <= MAX_SOURCE_BYTES
                or source.stat().st_size != matches[0]["bytes"]):
            raise ValueError("Exact fitting-role source manifest mismatch")
        sources.append((participant, source, matches[0]["bytes"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    report = {"analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "settings": settings.record(), "primary_paths": PATH_NAMES, "fixed_participants": list(range(1, 13)),
              "fixed_series": 1, "source_role": "fit_and_inner_selection",
              "input_bytes_remaining11": sum(size for _, _, size in sources),
              "internal_deadline_seconds": args.max_seconds, "remaining11_source_reads_attempted": 0,
              "P1_source_payload_reopened": False, "source_unit_cue_threshold": .5,
              "noise_thresholds_estimated_or_frozen": False, "successful_subset_selected": False,
              "EEG_features_or_forecast_scores_computed": False, "onset_or_risk_labels_computed": False,
              "feedback_or_held_out_payload_access": False, "known_return_authenticated": False,
              "scientific_labels_usable": False, "combined_onset_timing_bound_ms": None,
              "results": [{"participant": 1, "series": 1, "status": "reused_complete_conditional_support_only",
                           "source_payload_reused": True, "existing_report": str(reused_path),
                           "selected_file_polarity_supported_by_official_durations": True,
                           "primary_nonfinite_values": reused["source_sensor_support"]["primary_nonfinite_values"],
                           "cue": reused["cue"], "stationary_noise": reused["stationary_noise"]}]}
    exit_code = 0
    try:
        for participant, source, _ in sources:
            report["remaining11_source_reads_attempted"] += 1
            file_started = time.monotonic()
            try:
                row = one_file(source, participant, settings)
            except TimeoutError:
                raise
            except Exception as error:
                row = {"participant": participant, "series": 1, "status": "file_processing_failure",
                       "source": str(source), "error_type": type(error).__name__, "error": str(error),
                       "classic_hs_load_attempted_including_incidental_EEG": True}
                exit_code = 1
            row["elapsed_seconds"] = time.monotonic() - file_started
            report["results"].append(row)
        report["status"] = "complete_all12_conditional_support_panel" if exit_code == 0 else "panel_with_file_failures_all_results_retained"
    except Exception as error:
        report.update(status="panel_incomplete", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        completed = {row["participant"] for row in report["results"]}
        for participant in range(1, 13):
            if participant not in completed:
                report["results"].append({"participant": participant, "series": 1, "status": "not_completed_before_failure"})
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output_dir / "rest_support_panel.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "remaining11_source_reads_attempted", "elapsed_seconds", "max_rss_kib")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
