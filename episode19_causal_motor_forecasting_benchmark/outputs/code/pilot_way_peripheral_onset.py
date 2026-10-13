#!/usr/bin/env python3
"""One fitting-file peripheral pilot, with its new focused tests in one run."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time
import unittest

import numpy as np
from scipy.io import loadmat

from peripheral_onset import (PATH_NAMES, PilotSettings, decode_fixed_half_level_cue,
                              detect_risk, fit_pilot_noise, sensor_summaries)


def select_columns(group, requested):
    names = [str(name).strip() for name in np.asarray(group["names"], dtype=object).reshape(-1)]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source sensor names")
    signal_array = np.asarray(group["sig"])
    if signal_array.ndim != 2 or signal_array.shape[1] != len(names):
        raise ValueError("source sensor names and dimensions disagree")
    if any(name not in names for name in requested):
        raise ValueError("required primary sensor missing")
    return np.asarray(signal_array[:, [names.index(name) for name in requested]], dtype=float)


def deadline(_signum, _frame):
    raise TimeoutError("selected 540-second maximum pilot deadline reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("coordinator-owned compute allocation required")
    if any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise RuntimeError("all numeric thread counts must be one")
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("deadline must be at most 540 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    settings = PilotSettings()
    report = {"analysis_id": "source_views_and_timing", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(), "settings": settings.record(),
              "primary_paths": PATH_NAMES, "held_out_access": False, "EEG_features_or_forecast_scores_computed": False,
              "EEG_array_incidentally_deserialized": False, "waveforms_reported": False,
              "valid_forecast_trials": 0, "feedback_calls": 0, "total_onset_uncertainty_verified": False,
              "scientific_margins_frozen": False, "production_detector_frozen": False,
              "numeric_cue_polarity_authenticated": False, "known_end_return_phase_authenticated": False,
              "selected_fitting_file_higher_level_polarity_supported": True,
              "polarity_evidence_scope": "HS_P1_S1_all34_raw_high_durations_equal_official_LEDoff_minus_LEDon",
              "provider_annotation_absolute_time_origin_resolved": False,
              "WS_metadata_or_timestamps_read_by_this_pilot": False,
              "source_unit_cue_threshold": .5,
              "online_risk_contract_qualified": False, "scientific_labels_usable": False,
              "candidate_cue_assumption": "fixed0.5_higher_is_illuminated_supported_for_P1_S1; completed_return_unqualified",
              "development_parameters_fit_before_fixed_parameter_replay": True,
              "replay_is_out_of_sample_scientific_evaluation": False}
    exit_code = 0
    try:
        tests_out = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/"tests"), pattern="test_peripheral_onset.py")
        result = unittest.TextTestRunner(stream=tests_out, verbosity=2).run(suite)
        (args.output_dir/"tests.txt").write_text(tests_out.getvalue())
        report["new_component_tests"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}
        if not result.wasSuccessful() or result.testsRun == 0:
            raise RuntimeError("new detector tests failed before source access")
        source = args.development_view/"way/fit_and_inner_selection/HS_P1_S1.mat"
        report["source"] = str(source)
        content = loadmat(source, variable_names=["hs"], simplify_cells=True)
        hs = content["hs"]
        report["EEG_array_incidentally_deserialized"] = True
        # Classic MAT nests every modality in hs: deserialize honestly, then
        # discard EEG before computing sensor statistics, thresholds or labels.
        eeg = hs.pop("eeg")
        report["discarded_EEG_shape"] = list(np.shape(eeg["sig"]))
        del eeg
        hs.pop("name", None)
        if int(hs["participant"]) != 1 or int(hs["series"]) != 1:
            raise ValueError("selected fitting source identity mismatch")
        rates = {name: float(hs[name]["samplingrate"]) for name in ("emg", "kin", "misc")}
        if rates != {"emg": 4000., "kin": 500., "misc": 500.}:
            raise ValueError("representative native sampling rates changed")
        emg = select_columns(hs["emg"], ["Anterior Deltoid", "Brachoradial", "Flexor Digitorum", "Common Extensor Digitorum", "First Dorsal Interosseus"])
        position_names = [f"P{axis}{sensor} - position {axis} sensor {sensor}" for sensor in (2, 3, 4) for axis in ("x", "y", "z")]
        hand = select_columns(hs["kin"], position_names).reshape(-1, 3, 3)
        led = select_columns(hs["misc"], ["ParticipantLED"])[:, 0]
        if len(led) != len(hand) or abs(len(emg)/4000. - len(hand)/500.) > .002:
            raise ValueError("native peripheral durations disagree beyond one 500Hz tick")
        other = select_columns(hs["kin"], [f"P{axis}1 - position {axis} sensor 1" for axis in ("x", "y", "z")] +
                               [f"F{axis}{plate} - force {axis.lower()} plate {plate}" for axis in ("X", "Y", "Z") for plate in (1, 2)])
        report["source_sensor_support"] = {"sampling_rates_hz": rates, "emg_samples": len(emg), "peripheral_ticks": len(hand),
                  "primary_nonfinite_values": int(np.sum(~np.isfinite(emg)) + np.sum(~np.isfinite(hand))),
                  "object_and_force_nonfinite_values": int(np.sum(~np.isfinite(other))),
                  "object_and_force_may_replace_primary_paths": False,
                  "object_and_force_consistency_qualification_complete": False,
                  "acquisition_clipping_limits_known": False, "explicit_time_or_sync_channel_present_in_hs": False,
                  "equal_duration_proves_clock_alignment": False}
        del other, hs, content
        cue, cue_report = decode_fixed_half_level_cue(led)
        cue_report["selected_fitting_file_higher_level_polarity_supported"] = True
        cue_report["provider_annotation_absolute_time_origin_resolved"] = False
        report["cue"] = cue_report
        if cue is None:
            report["status"] = "peripheral_pilot_unresolved_cue"
        else:
            features = sensor_summaries(emg, hand, settings)
            thresholds, noise_report = fit_pilot_noise(features, emg, hand, cue, settings)
            report["stationary_noise"] = noise_report
            if thresholds is None:
                report["status"] = "peripheral_pilot_unresolved_noise_support"
            else:
                events, _ = detect_risk(features, cue, *thresholds, settings)
                events["selected_fitting_file_higher_level_polarity_supported"] = True
                report["conditional_candidate_cue_replay"] = events
                report["status"] = "peripheral_pilot_conditional_candidate_cue_replay_not_qualified"
        report["timing_components_ms"] = {
            "publication_reported_cross_system_alignment_bound": 2.,
            "publication_alignment_directly_remeasured": False,
            "primary_common_grid_sample_interval": 2.,
            "decision_grid_interval_not_onset_timestamp_rounding": 50.,
            "confirmation_lookahead_label_only_not_timestamp_shift": 40.,
            "EMG_past_center_and_RMS_history_span_not_delay_bound": 519.75,
            "motion_derivative_and_trailing_summary_history_span_not_delay_bound": 20.,
            "unknown_acquisition_response_bound": None,
            "unknown_total_resampling_and_filter_response_bound": None,
            "unknown_threshold_crossing_uncertainty_bound": None,
            "combined_worst_case_bound": None,
            "required_combined_bound_strictly_less_than": 300.,
            "source_methods": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/"}
    except Exception as error:
        report.update(status="peripheral_pilot_execution_failure", error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output_dir/"peripheral_pilot_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "max_rss_kib", "EEG_array_incidentally_deserialized", "held_out_access", "EEG_features_or_forecast_scores_computed")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
