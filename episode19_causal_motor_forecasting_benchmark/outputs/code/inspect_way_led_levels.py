#!/usr/bin/env python3
"""Describe ParticipantLED in one fitting HS file; select no cue/risk decoder.

Threshold fractions use the full file's range for retrospective diagnostics
only. They are not causally available scaling, authenticated on/off states,
onset labels, or a production threshold. No EEG feature or event table is used.
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

import numpy as np
from scipy.io import loadmat

from pilot_way_peripheral_onset import select_columns


QUANTILES = [0., .001, .01, .1, .25, .5, .75, .9, .99, .999, 1.]
THRESHOLD_FRACTIONS = [.25, .5, .75]


def quantile_summary(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "quantiles": None}
    return {"count": len(values), "quantiles": np.quantile(values, QUANTILES).tolist()}


def edge_summary(values, threshold, rate):
    finite = np.isfinite(values)
    state = np.where(finite, (values >= threshold).astype(np.int8), -1)
    up = np.flatnonzero((state[:-1] == 0) & (state[1:] == 1)) + 1
    down = np.flatnonzero((state[:-1] == 1) & (state[1:] == 0)) + 1
    starts = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
    ends = np.r_[starts[1:], len(state)]
    complete = (starts > 0) & (ends < len(state))
    durations = {}
    for code, name in ((0, "below_threshold"), (1, "at_or_above_threshold"), (-1, "nonfinite")):
        selected = complete & (state[starts] == code)
        durations[name] = quantile_summary((ends[selected] - starts[selected]) / rate)
    return {
        "threshold_source_units": float(threshold),
        "upcrossing_count": len(up), "downcrossing_count": len(down),
        "upcrossing_first20_sample_indices": up[:20].tolist(),
        "downcrossing_first20_sample_indices": down[:20].tolist(),
        "upcrossing_first20_times_seconds": (up[:20] / rate).tolist(),
        "downcrossing_first20_times_seconds": (down[:20] / rate).tolist(),
        "upcrossing_interval_seconds": quantile_summary(np.diff(up) / rate),
        "downcrossing_interval_seconds": quantile_summary(np.diff(down) / rate),
        "complete_constant_comparison_run_duration_seconds": durations,
        "first_and_last_recording_runs_omitted_from_duration_summary": True,
        "transitions_into_or_out_of_nonfinite_samples_not_counted_as_edges": True,
        "on_off_polarity_assigned": False,
    }


def deadline(signum, frame):
    raise TimeoutError("120-second LED diagnostic bound reached")


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
    signal.alarm(120)
    report = {
        "analysis_id": "source_views_and_timing", "status": "LED_diagnostic_started",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
        "quantile_probabilities": QUANTILES, "range_threshold_fractions": THRESHOLD_FRACTIONS,
        "diagnostic_thresholds_use_full_file_range_not_online_scaling": True,
        "waveform_samples_emitted": False, "distribution_values_and_edge_summaries_only": True,
        "nested_modalities_incidentally_deserialized": False,
        "only_ParticipantLED_values_analyzed": True, "EEG_features_or_scores_computed": False,
        "held_out_access": False, "feedback_access": False, "provider_event_tables_opened": False,
        "numeric_cue_polarity_authenticated": False, "known_end_return_phase_authenticated": False,
        "online_risk_contract_qualified": False, "scientific_labels_usable": False,
        "total_onset_uncertainty_verified": False, "combined_worst_case_timing_bound_ms": None,
        "cue_decoder_changed_or_selected": False,
        "source_methods": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/",
        "official_event_code": "https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m",
    }
    exit_code = 0
    try:
        source = args.development_view / "way/fit_and_inner_selection/HS_P1_S1.mat"
        report["source"] = str(source)
        content = loadmat(source, variable_names=["hs"], simplify_cells=True)
        hs = content["hs"]
        report["nested_modalities_incidentally_deserialized"] = True
        if int(hs["participant"]) != 1 or int(hs["series"]) != 1:
            raise ValueError("selected fitting source identity mismatch")
        rate = float(hs["misc"]["samplingrate"])
        if rate != 500.:
            raise ValueError("expected native ParticipantLED sampling rate of 500 Hz")
        led = select_columns(hs["misc"], ["ParticipantLED"])[:, 0].copy()
        del hs, content
        # Other nested arrays, including EEG, are no longer referenced or used.
        report["sampling_rate_hz"] = rate
        report["sample_time_origin"] = "first_released_sample_is_zero; no_absolute_clock_claim"
        finite = led[np.isfinite(led)]
        report["sample_count"] = len(led)
        report["nonfinite_count"] = int(len(led) - len(finite))
        if not len(finite):
            raise ValueError("no finite ParticipantLED values")
        levels, counts = np.unique(finite, return_counts=True)
        report["finite_distinct_levels"] = len(levels)
        report["amplitude_source_units"] = quantile_summary(finite)
        # A full histogram is small for the observed 95 levels; cap future output.
        keep = np.arange(len(levels)) if len(levels) <= 256 else np.sort(np.argsort(-counts, kind="stable")[:256])
        report["level_histogram"] = [{"value": float(levels[i]), "count": int(counts[i])} for i in keep]
        report["histogram_truncated_to_256_most_frequent_levels"] = len(levels) > 256
        report["largest_adjacent_level_gaps"] = []
        if len(levels) > 1:
            gaps = np.diff(levels)
            for index in np.argsort(-gaps, kind="stable")[:5]:
                report["largest_adjacent_level_gaps"].append({
                    "lower_level": float(levels[index]), "upper_level": float(levels[index + 1]),
                    "gap": float(gaps[index]), "fraction_of_full_range": float(gaps[index] / (levels[-1] - levels[0])),
                    "samples_at_or_below_lower_level": int(np.sum(counts[:index + 1])),
                    "samples_at_or_above_upper_level": int(np.sum(counts[index + 1:])),
                })
            report["retrospective_range_threshold_diagnostics"] = [
                dict(range_fraction=fraction, **edge_summary(led, levels[0] + fraction * (levels[-1] - levels[0]), rate))
                for fraction in THRESHOLD_FRACTIONS]
        report["status"] = "LED_distribution_and_edges_described_no_decoder_selected"
    except Exception as error:
        report.update(status="LED_diagnostic_execution_failure", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output_dir / "led_diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "max_rss_kib",
                                                 "cue_decoder_changed_or_selected", "held_out_access")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
