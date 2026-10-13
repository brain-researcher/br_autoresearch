"""Development-only WAY peripheral detector; arrays supplied by a trusted reader.

This module accepts no filenames, EEG, annotations, or forecast scores. The
noise recipe is a prospective pilot, not a frozen scientific onset definition.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PilotSettings:
    sample_rate_hz: int = 500
    emg_rate_hz: int = 4000
    emg_past_center_ms: int = 500
    summary_ms: int = 20
    confirmation_ms: int = 40
    entry_stillness_ms: int = 500
    candidate_rest_ms: int = 1000
    motion_stationarity_ratio_max: float = 3.0
    emg_quarter_rms_ratio_max: float = 2.0
    emg_mean_drift_ratio_max: float = 3.0
    minimum_rest_windows: int = 8
    minimum_rest_windows_per_parity: int = 4
    minimum_stationary_fraction: float = 0.5
    high_noise_mad_multiplier: float = 6.0
    low_noise_mad_multiplier: float = 3.0
    validation_any_path_high_fraction_max: float = 0.05
    decision_interval_ms: int = 50

    def record(self):
        return asdict(self)


PATH_NAMES = ["emg_anterior_deltoid", "emg_brachoradial", "emg_flexor_digitorum",
              "emg_common_extensor_digitorum", "emg_first_dorsal_interosseus",
              "index_finger_p2", "thumb_p3", "wrist_p4"]


def trailing_mean(values, width):
    """Finite causal mean; no padding, full finite history required after gaps.

    Equivalent to the existing CausalFIR with width equal coefficients, using
    cumulative sums for the native 4 kHz source. The independent test compares
    it with that already-qualified stateful kernel.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or width < 1:
        raise ValueError("matrix and positive finite window required")
    out = np.full_like(values, np.nan)
    if len(values) < width:
        return out
    valid = np.isfinite(values)
    sums = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(np.where(valid, values, 0.), axis=0)])
    counts = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)])
    complete = counts[width:] - counts[:-width] == width
    selected = (sums[width:] - sums[:-width]) / width
    out[width - 1:] = np.where(complete, selected, np.nan)
    return out


def sensor_summaries(emg, hand_position, settings=PilotSettings()):
    """Return five EMG RMS and three backward-speed means on native 500 Hz ticks.

    hand_position has shape (ticks, three sensors, XYZ). Native source units
    are retained. EMG at tick i uses sample 8*i and earlier only.
    """
    emg = np.array(emg, dtype=np.float64, copy=True)
    hand = np.array(hand_position, dtype=np.float64, copy=True)
    if emg.ndim != 2 or emg.shape[1] != 5 or hand.ndim != 3 or hand.shape[1:] != (3, 3):
        raise ValueError("all five EMG and three XYZ hand paths are required")
    ratio = settings.emg_rate_hz // settings.sample_rate_hz
    if ratio * settings.sample_rate_hz != settings.emg_rate_hz or (len(hand)-1)*ratio >= len(emg):
        raise ValueError("native sampling grid does not cover every peripheral tick")
    # A missing primary path resets all sensor histories. An EMG gap between
    # common ticks is known only at the next tick, never the preceding one.
    bad_hand = np.flatnonzero(~np.all(np.isfinite(hand), axis=(1, 2)))
    bad_emg = np.flatnonzero(~np.all(np.isfinite(emg), axis=1))
    emg[bad_emg] = np.nan
    emg[bad_hand * ratio] = np.nan
    common_bad_emg = (bad_emg + ratio - 1) // ratio
    common_bad_emg = common_bad_emg[common_bad_emg < len(hand)]
    hand[bad_hand] = np.nan
    hand[common_bad_emg] = np.nan
    # Propagate a detected common-grid gap back to all EMG paths at that tick.
    emg[common_bad_emg * ratio] = np.nan
    ncenter = settings.emg_rate_hz * settings.emg_past_center_ms // 1000
    nsummary = settings.emg_rate_hz * settings.summary_ms // 1000
    center = trailing_mean(emg, ncenter)
    # Strictly past mean, without global/session normalization.
    residual = np.full_like(emg, np.nan)
    residual[1:] = emg[1:] - center[:-1]
    envelope = np.sqrt(np.maximum(trailing_mean(residual * residual, nsummary), 0.))
    velocity = np.full((len(hand), 3), np.nan)
    velocity[1:] = np.linalg.norm(np.diff(hand, axis=0), axis=2) * settings.sample_rate_hz
    motion = trailing_mean(velocity, settings.sample_rate_hz * settings.summary_ms // 1000)
    return np.column_stack([envelope[np.arange(len(hand)) * ratio], motion])


def decode_two_level_cue(signal):
    """Describe two recorded levels and propose conditional candidate states.

    Nonfinite samples remain unknown. An analogue/noisy multilevel signal is
    unresolved, not silently thresholded or reconstructed from future events.
    Numeric higher=illuminated/start is NOT authenticated by two levels.
    """
    signal = np.asarray(signal, dtype=np.float64)
    levels, counts = np.unique(signal[np.isfinite(signal)], return_counts=True)
    report = {"finite_level_count": int(len(levels)), "unknown_samples": int(np.sum(~np.isfinite(signal))),
              "numeric_polarity_authenticated": False, "known_start_cue_authenticated": False,
              "known_end_return_phase_authenticated": False,
              "conditional_assumption": "higher_recorded_level_is_start_on_lower_is_end_off_not_source_authenticated"}
    if len(levels) != 2:
        report["status"] = "unresolved_not_two_exact_levels"
        return None, report
    report.update(status="two_levels_observed_conditional_candidate_states_only", levels=levels.tolist(), level_counts=counts.tolist())
    cue = np.where(np.isfinite(signal), (signal == levels[1]).astype(np.int8), -1)
    return cue, report


def decode_fixed_half_level_cue(signal):
    """Pointwise0.5 source-unit comparison for the selected fitting-file pilot.

    No full-recording amplitude estimate or timestamp correction enters state.
    Missing values remain unknown. Source polarity support is a property of the
    caller's selected fitting file, not proof supplied by this numeric function.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("one ParticipantLED channel is required")
    finite = np.isfinite(signal)
    cue = np.where(finite, (signal >= .5).astype(np.int8), -1)
    return cue, {
        "status": "fixed_half_level_candidate_states_for_conditional_pilot",
        "threshold_source_units": .5, "comparison": "at_or_above",
        "unknown_samples": int(np.sum(~finite)),
        "whole_file_normalization_or_timestamp_shift": False,
        "numeric_polarity_authenticated_by_decoder": False,
        "known_end_return_phase_authenticated": False,
        "scientific_labels_usable": False,
    }


def rest_window_stationarity(emg, hand, settings=PilotSettings()):
    """Use all paths and fixed quarter stability criteria, never quietest ranks."""
    if not np.all(np.isfinite(emg)) or not np.all(np.isfinite(hand)):
        return False, ["nonfinite_primary_sensor"]
    reasons = []
    hand_q = np.array_split(hand, 4)
    means = np.asarray([q.mean(axis=0) for q in hand_q])
    pooled = np.sqrt(sum(np.sum((q-m)**2, axis=(0, 2)) for q, m in zip(hand_q, means)) / len(hand))
    displacement = np.max(np.linalg.norm(means - hand.mean(axis=0), axis=2), axis=0)
    for path in np.flatnonzero(displacement > settings.motion_stationarity_ratio_max * pooled):
        reasons.append(PATH_NAMES[5+path] + "_quarter_motion_drift")
    emg_q = np.array_split(emg, 4)
    emg_means = np.asarray([q.mean(axis=0) for q in emg_q])
    rms = np.asarray([np.sqrt(np.mean((q-m)**2, axis=0)) for q, m in zip(emg_q, emg_means)])
    pooled_sd = np.sqrt(np.mean(rms*rms, axis=0))
    for path in range(5):
        if np.min(rms[:, path]) <= 0 or np.max(rms[:, path]) > settings.emg_quarter_rms_ratio_max * np.min(rms[:, path]):
            reasons.append(PATH_NAMES[path] + "_quarter_noise_nonstationary_or_zero")
        if np.max(np.abs(emg_means[:, path]-emg[:, path].mean())) > settings.emg_mean_drift_ratio_max * pooled_sd[path]:
            reasons.append(PATH_NAMES[path] + "_quarter_mean_drift")
    return not reasons, reasons


def fit_pilot_noise(features, emg, hand, cue, settings=PilotSettings(), *, support_only=False):
    """Pre-candidate-edge noise windows do not by themselves prove rest/return."""
    rises = np.flatnonzero((cue[1:] == 1) & (cue[:-1] == 0)) + 1
    width = settings.candidate_rest_ms * settings.sample_rate_hz // 1000
    # The entire 500 ms centering + 20 ms RMS history must lie in this window.
    warmup = (settings.emg_past_center_ms + settings.summary_ms) * settings.sample_rate_hz // 1000
    ratio = settings.emg_rate_hz // settings.sample_rate_hz
    windows, fit, validation = [], [], []
    eligible = 0
    for cue_index, stop in enumerate(rises):
        begin = int(stop)-width
        row = {"cue_index": cue_index, "cue_sample": int(stop), "partition": "noise_fit" if cue_index % 2 == 0 else "noise_validation"}
        if begin < 0 or not np.all(cue[begin:stop] == 0):
            row.update(eligible=False, stationary=False, reasons=["incomplete_one_second_observed_lower_level"])
            windows.append(row)
            continue
        eligible += 1
        accepted, reasons = rest_window_stationarity(emg[begin*ratio:int(stop)*ratio], hand[begin:stop], settings)
        values = features[begin+warmup:stop]
        if not np.all(np.isfinite(values)):
            accepted = False
            reasons.append("incomplete_finite_summary_history")
        row.update(eligible=True, stationary=accepted, reasons=reasons, summary_samples=int(len(values)))
        windows.append(row)
        if accepted:
            (fit if cue_index % 2 == 0 else validation).append(values)
    accepted_count = len(fit) + len(validation)
    report = {"candidate_higher_level_edges": int(len(rises)), "eligible_windows": eligible, "stationary_windows": accepted_count,
              "fit_windows": len(fit), "validation_windows": len(validation), "windows": windows,
              "candidate_windows_prove_physiological_rest": False, "thresholds_frozen_for_scoring": False,
              "candidate_windows_require_cue_and_rest_interpretation": True}
    enough = (accepted_count >= settings.minimum_rest_windows and len(fit) >= settings.minimum_rest_windows_per_parity
              and len(validation) >= settings.minimum_rest_windows_per_parity and eligible > 0
              and accepted_count / eligible >= settings.minimum_stationary_fraction)
    if not enough:
        report["status"] = "unresolved_stationary_support"
        return None, report
    if support_only:
        report["status"] = "stationary_support_sufficient_thresholds_not_fit"
        return None, report
    train = np.concatenate(fit)
    center = np.median(train, axis=0)
    # Normal-consistent robust MAD; zero values never receive an invented floor.
    scale = 1.4826 * np.median(np.abs(train-center), axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        report.update(status="unresolved_zero_or_nonfinite_noise_scale", unresolved_paths=[PATH_NAMES[i] for i in np.flatnonzero(~np.isfinite(scale) | (scale <= 0))])
        return None, report
    low = center + settings.low_noise_mad_multiplier * scale
    high = center + settings.high_noise_mad_multiplier * scale
    held_noise = np.concatenate(validation)
    fraction = float(np.mean(np.any(held_noise >= high, axis=1)))
    report.update(noise_median=center.tolist(), noise_scaled_mad=scale.tolist(), low_threshold=low.tolist(),
                  high_threshold=high.tolist(), fit_summary_samples=len(train), validation_summary_samples=len(held_noise),
                  validation_any_path_high_fraction=fraction)
    if fraction > settings.validation_any_path_high_fraction_max:
        report["status"] = "unresolved_validation_noise_crossing_rate"
        return None, report
    report["status"] = "pilot_noise_support_pass_not_full_onset_qualification"
    return (low, high), report


def earliest_accepted_crossing(accepted):
    """A later confirmation cannot replace an earlier accepted crossing label."""
    return min(accepted, key=lambda row: (row["crossing_sample"], row["path_index"])) if accepted else None


def detect_risk(features, cue, low, high, settings=PilotSettings()):
    """Conditional candidate-cue replay; not qualified online risk.

    The prospective arithmetic implements armed/pending behavior but the pilot
    does not establish source cue polarity or the known end/return phase.
    Its state, exposure and sensor crossings cannot become scientific labels.
    """
    features = np.asarray(features, dtype=float)
    cue = np.asarray(cue)
    low, high = np.asarray(low), np.asarray(high)
    if features.ndim != 2 or features.shape[1] != 8 or cue.shape != (len(features),):
        raise ValueError("all eight primary paths and aligned cue are required")
    if low.shape != (8,) or high.shape != (8,) or not np.all(np.isfinite([low, high])) or np.any(high <= low):
        raise ValueError("finite separated high/low thresholds required for all paths")
    if not np.all(np.isin(cue, [-1, 0, 1])):
        raise ValueError("candidate states must be unknown, lower or higher level")
    still_needed = settings.entry_stillness_ms * settings.sample_rate_hz // 1000
    dwell = settings.confirmation_ms * settings.sample_rate_hz // 1000
    stride = settings.decision_interval_ms * settings.sample_rate_hz // 1000
    risk = np.zeros(len(features), dtype=bool)
    suspended = np.zeros(len(features), dtype=bool)
    rows, current = [], None
    still, previous_cue, armed = 0, -1, False
    pending = {}
    suspended_episode = False

    def close(reason, stop):
        nonlocal current, pending, armed, suspended_episode
        if current is not None:
            current.update(close_reason=reason, close_sample=int(stop))
            if pending:
                current["unresolved_crossings"].extend({"path_index": p, "crossing_sample": s, "reason": reason} for p, s in pending.items())
            onset = earliest_accepted_crossing(current["accepted_crossings"])
            ambiguous_earlier = onset is not None and any(r["crossing_sample"] <= onset["crossing_sample"] for r in current["unresolved_crossings"])
            current["candidate_sensor_onset"] = dict(onset) if onset is not None and not ambiguous_earlier else None
            current["earlier_unresolved_crossing_prevents_label"] = ambiguous_earlier
            rows.append(current)
        current, pending, armed, suspended_episode = None, {}, False, False

    for tick, values in enumerate(features):
        valid = bool(np.all(np.isfinite(values))) and cue[tick] >= 0
        if not valid:
            close("sensor_or_cue_gap_unknown", tick)
            still, previous_cue = 0, -1
            continue
        if cue[tick] == 0 and current is not None:
            close("observed_lower_level_not_authenticated_end_return", tick)
        if cue[tick] == 1 and previous_cue == 0:
            close("new_candidate_higher_level_edge", tick)
            current = {"candidate_start_edge_sample": tick, "accepted_crossings": [], "rejected_crossings": [], "unresolved_crossings": [],
                       "conditional_candidate_risk_decisions": 0, "pending_or_requalification_decisions": 0}
        was_still = still >= still_needed
        quiet = bool(np.all(values <= low))
        still = still + 1 if quiet else 0
        if current is not None and not current["accepted_crossings"]:
            if not armed and not pending and was_still:
                # State at this tick uses prior stillness; a current crossing
                # immediately suspends this decision and all pending decisions.
                armed = True
                suspended_episode = False
            if armed or pending:
                for path in range(8):
                    if path not in pending and values[path] >= high[path]:
                        pending[path] = tick
                if pending:
                    suspended_episode = True
            for path, start in list(pending.items()):
                if values[path] < low[path]:
                    current["rejected_crossings"].append({"path_index": path, "crossing_sample": start, "rejection_sample": tick})
                    del pending[path]
                elif tick - start >= dwell:
                    current["accepted_crossings"].append({"path_index": path, "crossing_sample": start, "confirmation_sample": tick})
                    del pending[path]
            if suspended_episode and not pending:
                armed = False
            if armed and not suspended_episode and not current["accepted_crossings"]:
                risk[tick] = True
            if suspended_episode:
                suspended[tick] = True
            if tick % stride == 0:
                current["conditional_candidate_risk_decisions"] += int(risk[tick])
                current["pending_or_requalification_decisions"] += int(suspended[tick])
        elif current is not None:
            # Resolve any already-started paths after a first confirmation.
            # No new path or second movement may rearm this cue.
            for path, start in list(pending.items()):
                if values[path] < low[path]:
                    current["rejected_crossings"].append({"path_index": path, "crossing_sample": start, "rejection_sample": tick})
                    del pending[path]
                elif tick - start >= dwell:
                    current["accepted_crossings"].append({"path_index": path, "crossing_sample": start, "confirmation_sample": tick})
                    del pending[path]
        previous_cue = int(cue[tick])
    close("administrative_series_end", len(features))
    for row in rows:
        if row["candidate_sensor_onset"] is not None:
            row["candidate_sensor_onset"]["seconds"] = row["candidate_sensor_onset"]["crossing_sample"] / settings.sample_rate_hz
    return {"candidate_cue_segments": rows, "candidate_start_edges_processed": len(rows),
            "candidate_sensor_onsets_with_confirmation": sum(r["candidate_sensor_onset"] is not None for r in rows),
            "cue_polarity_authenticated": False, "known_end_return_phase_authenticated": False,
            "online_risk_contract_qualified": False, "scientific_labels_usable": False,
            "conditional_candidate_risk_decisions": int(risk[::stride].sum()), "pending_or_requalification_decisions": int(suspended[::stride].sum()),
            "conditional_candidate_risk_seconds": float(risk.sum()/settings.sample_rate_hz),
            "pending_or_requalification_seconds": float(suspended.sum()/settings.sample_rate_hz)}, risk
