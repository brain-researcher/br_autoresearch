#!/usr/bin/env python3
"""Development header/marker geometry and one prospective FIR's coefficients.

Authorized inputs must be supplied by the coordinator: the explicit frozen
development manifest and recorded BIDS root. Open only its 30 participants'
exact .vhdr/.vmrk leaves through the existing reader; .eeg is stat-only. Never
open an event TSV, event/image join, EEG/image array, audit manifest or archive.
All recorded E3 intervals are reported, including repeated-image sequences.
An E1 count of 309 is descriptive, NOT a substitute for the frozen main-event
identity join. This report does not prepare EEG, adopt a filter, release a
reader, infer main sequence identities or certify a scientific comparison.

The one coefficient calculation is the prospective 129-tap, 100-Hz,
Kaiser-beta8.6 causal FIR, with origin0, 128-sample past support and 64-sample
linear-phase delay. Numerical frequency response, not EEG, informs the filter
decision. No search over alternative filters or outcome-driven selection.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
from scipy.signal import firwin, freqz

from brainvision_sequence_reader import bind_development_paths, read_recording_metadata
from false_partition_search import BudgetStop, THREAD_ENV, atomic_json, budget
from feature_extractors import require_compute
from prepare_false_partition_inputs import read_json, require


def coefficient_summary(check):
    check()
    taps = firwin(129, 100., window=("kaiser", 8.6), pass_zero="lowpass", scale=True, fs=1000.)
    require(taps.shape == (129,) and np.isfinite(taps).all())
    # Uniform grid includes both DC and Nyquist; add the exact passband edge.
    frequencies, response = freqz(taps, worN=2**18+1, fs=1000., include_nyquist=True)
    edge_frequencies, edge_response = freqz(taps, worN=np.array([80., 100., 125.]), fs=1000.)
    gain = np.abs(response)
    require(np.isfinite(gain).all() and np.isfinite(edge_response).all())
    pass_gain = np.r_[gain[frequencies <= 80.], abs(edge_response[0])]
    stop_gain = np.r_[gain[frequencies >= 125.], abs(edge_response[2])]
    # Multiplication by exp(+i*64*w) preserves magnitude. Its derivative has
    # norm at most sum |(k-64)h[k]|. This bounds the unobserved frequency-grid
    # intervals, independent of an EEG spectrum. Report rounding separately.
    derivative = float(np.sum(np.abs((np.arange(129)-64)*taps)))
    spacing = float(np.max(np.diff(frequencies)))
    between_grid = derivative*(2*np.pi/1000.)*spacing/2
    stop_max = float(stop_gain.max())
    pass_error = float(np.max(np.abs(pass_gain-1.)))
    check()
    return {"recipe_status": "prospective_not_adopted", "design": "scipy.signal.firwin",
        "numtaps": 129, "cutoff_half_amplitude_hz": 100., "window": ["kaiser", 8.6],
        "scale": True, "source_sample_rate_hz": 1000, "target_sample_rate_hz": 250,
        "fir_origin_sample": 0, "detrend": "none", "decimation_phase_mod4": 0,
        "operation_order": "detrend_then_fir_then_decimate", "input_sample_unit": "V",
        "past_support_samples": 128, "future_support_samples": 0, "group_delay_samples": 64,
        "coefficients_float64": taps.tolist(),
        "coefficients_little_endian_float64_sha256": hashlib.sha256(taps.astype("<f8").tobytes()).hexdigest(),
        "dc_gain": float(taps.sum()), "maximum_symmetry_error": float(np.max(abs(taps-taps[::-1]))),
        "passband_hz": [0., 80.], "passband_grid_minimum_gain": float(pass_gain.min()),
        "passband_grid_maximum_gain": float(pass_gain.max()), "passband_grid_maximum_absolute_gain_error": pass_error,
        "stopband_hz": [125., 500.], "stopband_grid_maximum_gain": stop_max,
        "stopband_grid_minimum_attenuation_db": float(-20*np.log10(stop_max)),
        "exact_edge_gains": dict(zip(map(str, edge_frequencies.tolist()), np.abs(edge_response).tolist())),
        "frequency_grid_maximum_spacing_hz": spacing,
        "analytic_between_grid_magnitude_variation_bound": between_grid,
        "passband_absolute_error_envelope_ignoring_roundoff": pass_error+between_grid,
        "stopband_gain_envelope_ignoring_roundoff": stop_max+between_grid,
        "stopband_attenuation_envelope_db_ignoring_roundoff": float(-20*np.log10(stop_max+between_grid)),
        "numerical_roundoff_included_in_analytic_envelope": False,
        "primary_response_horizon_ms": [0, 800], "filter_delay_compensation": False,
        "false_partition_primary_context_horizon_samples": 928,
        "scientific_filter_acceptance_decision": None}


def participant_geometry(metadata, check):
    cues = tuple(row for row in metadata.markers
                 if row.marker_type == "Event" and re.fullmatch(r"E\s*3", row.description))
    starts = [row.sample_zero_based for row in cues]
    require(cues and len(set(starts)) == len(starts))
    stimulus = np.asarray([row.sample_one_based-1 for row in metadata.stimulus_markers], dtype=np.int64)
    require(np.all(np.diff(stimulus) > 0))
    segments = metadata.recording_segments
    segment_starts = [segment.start_sample for segment in segments]
    rows = []
    for index, cue in enumerate(cues):
        check()
        start = cue.sample_zero_based
        next_cue = starts[index+1] if index+1 < len(starts) else metadata.recording_sample_count
        segment = segments[bisect_right(segment_starts, start)-1]
        selected = stimulus[np.searchsorted(stimulus, start):np.searchsorted(stimulus, next_cue)]
        row = {"e3_marker_id": cue.marker_id, "e3_zero_based_sample": start,
            "next_e3_or_recording_end_sample": next_cue, "recording_segment_id": segment.segment_id,
            "recording_segment_end_sample": segment.stop_sample, "e1_count": len(selected),
            "main_sequence_identity_inferred": False}
        if len(selected):
            first, last = int(selected[0]), int(selected[-1])
            required_stop = last+801  # Half-open interval includes last E1 +800ms.
            score_first = first-800
            first_retained_grid = score_first + (-score_first) % 4
            last_retained_grid = last+800 - (last+800) % 4
            row.update(first_e1_zero_based_sample=first, last_e1_zero_based_sample=last,
                e3_to_first_e1_samples=first-start, required_stop_sample=required_stop,
                next_e3_margin_samples=next_cue-required_stop,
                segment_end_margin_samples=segment.stop_sample-required_stop,
                entire_e1_interval_in_same_segment=last < segment.stop_sample,
                last_plus_800ms_fits=required_stop <= min(next_cue, segment.stop_sample),
                proposed_negative_800ms_score_start_sample=score_first,
                negative_800ms_start_has_full_causal128_support=score_first >= start+128,
                proposed_recording_phase0_first_scored_sample=first_retained_grid,
                proposed_recording_phase0_last_scored_sample=last_retained_grid,
                proposed_complete_score_grid_count=max(0, (last_retained_grid-first_retained_grid)//4+1))
        rows.append(row)
    nonempty = [row for row in rows if row["e1_count"]]
    return {"participant_id": metadata.paths.participant_id,
        "sampling_hz": metadata.sampling_hz, "recording_samples_from_binary_stat": metadata.recording_sample_count,
        "recorded_channel_order": list(metadata.channel_order), "channel_resolutions": list(metadata.resolutions),
        "channel_units": list(metadata.source_units), "volts_per_stored_unit": list(metadata.volts_per_stored_unit),
        "recording_segments": len(segments), "e3_intervals": rows,
        "nonempty_e3_interval_count": len(nonempty), "empty_e3_interval_count": len(rows)-len(nonempty),
        "stimuli_before_first_e3": int(np.searchsorted(stimulus, starts[0])),
        "e1_counts_by_e3_interval": dict(sorted(Counter(str(row["e1_count"]) for row in rows).items())),
        "e3_to_first_e1_gap_counts": dict(sorted(Counter(str(row["e3_to_first_e1_samples"])
            for row in rows if row["e1_count"]).items())),
        # Empty E3 intervals have no first/last E1 to assess. Preserve their
        # rows/count separately; no nonempty intervals means unknown, not pass.
        "all_nonempty_intervals_fit_last_plus_800ms":
            all(row["last_plus_800ms_fits"] for row in nonempty) if nonempty else None,
        "all_nonempty_intervals_support_full_negative_800ms":
            all(row["negative_800ms_start_has_full_causal128_support"] for row in nonempty) if nonempty else None,
        "frozen_main_event_join_checked": False, "EEG_amplitudes_read": False}


def all_observed(values):
    """A known failure dominates unknown; missing observations never pass."""
    values = tuple(values)
    return False if any(value is False for value in values) else None if not values or any(value is None for value in values) else True


def inspect(manifest_path, bids_root, check):
    manifest = read_json(manifest_path)
    people = manifest["participants"]
    require(manifest["role"] == "development" and len(people) == len(set(people)) == 30
            and people == sorted(people) and manifest["event_count"] == 30*24648)
    results = []
    for person in people:
        check()
        paths = bind_development_paths(development_manifest=manifest_path, bids_root=bids_root, participant_id=person)
        metadata = read_recording_metadata(paths, header_byte_limit=8*1024**2,
                                          marker_byte_limit=128*1024**2, budget_check=check)
        results.append(participant_geometry(metadata, check))
        del metadata
    common = set(results[0]["recorded_channel_order"])
    for result in results[1:]:
        common.intersection_update(result["recorded_channel_order"])
    ordered = [channel for channel in results[0]["recorded_channel_order"] if channel in common]
    return {"status": "development_metadata_inspection_complete", "participants": results,
        "observed_common_channel_order_using_first_development_record": ordered,
        "observed_common_channel_count": len(ordered), "previously_frozen_channel_order_checked": False,
        "all_sampling_rates_1000_hz": all(result["sampling_hz"] == 1000 for result in results),
        "empty_e3_interval_count": sum(result["empty_e3_interval_count"] for result in results),
        "participants_with_no_nonempty_e3_intervals":
            [result["participant_id"] for result in results if not result["nonempty_e3_interval_count"]],
        "all_nonempty_recorded_e3_intervals_fit_proposed_positive_800ms":
            all_observed(result["all_nonempty_intervals_fit_last_plus_800ms"] for result in results),
        "all_nonempty_recorded_e3_intervals_support_proposed_negative_800ms":
            all_observed(result["all_nonempty_intervals_support_full_negative_800ms"] for result in results),
        "main_sequence_identity_release": False, "EEG_amplitudes_read": False,
        "behavior_event_values_read": False, "audit_inputs_read": False,
        "filter_or_span_recipe_adopted": False, "coefficients": coefficient_summary(check)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--rss-byte-cap", type=int, required=True)
    args = parser.parse_args(argv)
    owned, started, operational = False, time.time(), {"status": "started"}
    try:
        require_compute()
        require(all(os.environ.get(name) == "1" for name in THREAD_ENV))
        require(all(path.is_absolute() and ".." not in path.parts
                    for path in (args.development_manifest, args.bids_root, args.output)))
        os.umask(0o077)
        args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        owned = True
        atomic_json(args.output/"settings.json", {"development_manifest": str(args.development_manifest),
            "bids_root": str(args.bids_root), "deadline": args.deadline, "rss_byte_cap": args.rss_byte_cap,
            "durable_byte_cap": 16*1024**2, "EEG_access": "binary_stat_only_no_open",
            "event_TSV_or_join_access": False}, cap=64*1024)
        with budget(args.deadline, args.rss_byte_cap) as check:
            report = inspect(args.development_manifest, args.bids_root, check)
            check()
            atomic_json(args.output/"metadata_and_fir.json", report, cap=8*1024**2)
        operational.update(status="complete_metadata_and_coefficient_inspection", participants=30,
                           filter_or_span_recipe_adopted=False)
        return 0
    except BudgetStop as error:
        operational.update(status="resource_incomplete", reason=str(error))
        return 75
    except BaseException as error:
        operational.update(status="stopped_error", error_type=type(error).__name__)
        return 1
    finally:
        if owned:
            operational.update(elapsed_seconds=time.time()-started,
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                finished_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(args.output/"operational.json", operational, cap=64*1024)


if __name__ == "__main__":
    raise SystemExit(main())
