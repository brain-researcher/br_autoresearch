#!/usr/bin/env python3
"""EP19 post hoc raw-onset timing diagnostic; save in outputs/code/.

Python 3.12, numpy, and the supplied existing modules.
Run only in an actual Slurm compute-node shell.
The orchestrator must enter the declaration in the timing handoff before launch.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import socket
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector_readers as R
import detector_summaries as S
import detector_freeze as F
from run_detector_freeze import sp_load


EXPECTED_LOW = 0.08509930134406786
EXPECTED_HIGH = 0.16721586763828217
EXPECTED_FULL = {
    "cues": 4432,
    "accepted": 4189,
    "runs": 155,
    "start_tones": 4432,
    "still_windows": 3661,
    "fit_windows": 1901,
    "validation_windows": 1760,
}

# Locked design assumptions, including the post-smoke changes.
KS = (3, 5, 8)
SEED = 20261017
FIXED_MS = 12
MARGIN_MS = 288
SEARCH_MS = 3000
LOOKAHEAD_MS = 100
PERSIST_MS = 40
OCCUPANCY = 0.80
PRIMARY_QUIET_MS = 200
EARLIEST_QUIET_MS = 40
MIN_NOISE_WINDOWS = 20

BACKGROUND_MS = 2800
WARMUP_MS = 900
STILL_MS = 1900
END_TONE_OFFSET_MS = 1000
CONTROL_SEARCH_START_MS = 1420
CONTROL_S_MS = 1920
T0_MS = 1920
SHAM_L_MS = 2696                 # Add 3 ms for the 1-kHz detector clock.
MAX_BACKGROUNDS = 32
MIN_BACKGROUNDS = 20
PROVIDER_TICK_TOLERANCE = 1e-6

CONTROL_SPECS = {
    "linear": (80, 160, 272, 288, 304, 320, 400, 500),
    "quadratic": (80, 160, 272, 288, 304, 320, 400, 500, 600),
    "pulse_ramp": (288, 304, 320, 400, 600),
}
SEEN_SMOKE_PARTICIPANTS = ("sub-02", "sub-15", "sub-22")

PROVIDER_COLUMNS = {
    "acc_start_tick", "onset_minus_accstart_ms",
    "trial_start_minus_cue_ms", "acc_flag", "discard",
}
CUE_COLUMNS = (
    "participant", "session", "run", "cue",
    "onset_tick", "start_known_tick",
)

DECLARATION = """\
1. POST HOC revision after smoke job 48227088: sub-02/sub-15/sub-22 outputs were seen; those people remain in the full run. All added numerical/operational choices are assumptions.
2. Scope is development self-paced accelerometer/TRIG only; no held-out people, EEG features, provider-derived selection, or changes to frozen labels/thresholds.
3. Accepted events have finite frozen onset_tick L; S=start_known_tick. Provider column values are excluded from estimation and controls.
4. Native r(t)=||a(t)-mean(a[t-520,t-20))|| in g; baseline endpoints are half-open, and the current sample is unsmoothed.
5. Reproduce frozen stationary-window membership; use the final 480 native milliseconds/window for person median m and s=1.4826 MAD.
6. Require >=20 stationary windows and finite positive s. Primary theta=m+5s; locked sensitivities use k=3 and 8; unusable references give no_valid_data.
7. Main search is B=max(S-500,L-3000) through L+100 ms with complete validity; qualification is retrospective, but every reported R<=L.
8. A 40-ms window qualifies with >=80% samples strictly above theta. Primary quiet means no qualifying window starts in the specified half-open interval; its required window support must be observed.
9. Separate primary bouts at >=200 ms of quiet window starts; choose the bout whose envelope reaches L. Anchor 200 ms immediately before its first qualifying-window start; R is that window's first exceedance.
10. Keep the earlier descriptive sensitivity unchanged: earliest qualifying bout, initial 40-ms all-sample quiet, and 40-ms all-sample-quiet separators; report disagreements, never use it for decisions.
11. Missing validity gives no_valid_data; missing primary anchor/bout gives indeterminate. Determinate lag bounds equal L-R; unresolved upper is infinity, lower zero except L-candidate for an observed label-reaching primary bout.
12. Control backgrounds are nonoverlapping 2.8-s native segments inside [latest preceding classified end-tone onset+1 s, classified start-tone onset); retain the latest qualifying segment per interval. Require full frozen validity; screen only the final 1.9 s with 100-ms block means within 0.02 g of their mean.
13. Relative control timeline: valid-only warm-up [0,900), still part [900,2800), search B=1420, S=T0=1920 ms; sham L=2696 ms at 250 Hz or 2699 ms at 1 kHz, with 100-ms lookahead. Sham pre-label spans are 1276/1279 ms.
14. Seed 20261017 selects <=32 eligible backgrounds/person by reservoir sampling; require >=20 and report interval/eligible/selected counts. Background selection never uses event estimates or provider values.
15. Inject random-direction linear, quadratic, and 0.04-g/120-ms pulse plus quadratic growth beginning 180 ms after T0. Detectable-lag targets: linear/quadratic {80,160,272,288,304,320,400,500}; quadratic adds 600; pulse+ramp {288,304,320,400,600} ms.
16. Calibrate amplitude per person/family using only noiseless profiles, theta5 and frozen LOW/HIGH; search threshold intervals/breakpoints for exact L_nl-T_det targets, cap growth at L_nl+100, reverify equality and support, including T_det-200>=B. Targets never exceed 600; infeasible cells remain uncovered; close control_templates.csv before noisy controls.
17. L is first HIGH with that and the next ten detector samples >=LOW. Control truth is actual noisy L-T_det, where T_det is the first persistent noiseless-component exceedance; require the same noiseless label-reaching primary bout. Physical L-T0 is descriptive; k=3/8 reuse theta5-calibrated profiles.
18. Uncovered/unsupported/disconnected controls count against >=95% usable coverage per person/family; require >=20 distinct backgrounds in each required actual-truth band.
19. At k=5 require >=90% determinate >=288-ms classification for true lag [288,320), >=95% for [320,650], and >=95% determinate <288-ms classification for [0,200) in linear/quadratic controls.
20. Shams use the declared shorter control search and lookahead; any qualifying pre-label departure, missing initial 40 ms of quiet window starts, or invalid search is an alarm; require <=5% alarms/person.
21. All participants must pass k=5 controls; other-k controls are descriptive. Report per-k sham fractions in noise_reference.csv; inadequate locked coverage means underidentified without retuning.
22. Use nearest-rank empirical p95 censoring bounds over all events; report determinate-only/per-person p95, event totals >=300 ms after adding 12 ms, and cross-k long-lag events.
23. Failed primary controls, >5% unresolved primary events at any k, p95 bounds straddling 288 ms, differing classifications across k, or participant-list execution => UNDERIDENTIFIED.
24. Otherwise k=5 p95 lower+12>=300 => FAIL_EMPIRICAL_P95; upper+12<300 => BELOW_EMPIRICAL_P95_ONLY.
25. Also report k=5 metrics restricted to control-passing people, explicitly descriptive and unable to replace the all-participant decision.
26. Failure weakens the 188-ms p95 proxy and 300-ms timing argument; below-limit weakens the large-p95-lag alternative only under these assumptions. Neither certifies a worst-case bound or grants scoring clearance.
27. Flag 188 separately: exceeded if lower+12>188, conditionally compatible if upper+12<=188, otherwise unresolved.
28. Close Stage 1 tables, Stage 2 calibration/controls, and Stage 3 summary before Stage 4 provider access. Only then report R-AccStartIndex and L-AccStartIndex without flag filtering; provider ticks round only within 1e-6 tick of an integer.
29. Bounds address unresolved observations, not sampling confidence. People are independent human units; synthetic realism, shorter-control representativeness, persistence, drift handling and bout association remain assumptions.
"""

LOG_TEMPLATE = """\
## {stamp} Pacific — source_views_and_timing — self-paced raw-onset lag test pre-declared
POST HOC revision after smoke job 48227088; sub-02/sub-15/sub-22 outputs were seen and those people remain in the full development run.
Development accelerometer/TRIG only; frozen labels/thresholds unchanged; provider column values excluded until Stage 4.
Keep native [t-520,t-20) departure, person median+{{3,5,8}}*1.4826 MAD, >=20 still windows, 80%/40-ms qualification, and main search max(S-500,L-3000)..L+100.
Primary 200-ms quiet/separation means no qualifying window starts; anchor ends at the first qualifying-window start, with R its first exceedance. Earliest strict-sample-quiet sensitivity stays descriptive.
Controls: latest nonoverlapping 2.8-s segment per end+1-s->start interval; all valid, final 1.9 s still. B=1420, S=T0=1920, sham L=2696/2699 ms; seed 20261017, <=32/person, >=20 required; report eligible counts.
Keep exact noiseless detectable-lag calibration, targets <=600 ms, growth cap L_nl+100, support checks and uncovered cells; close calibration table before noisy controls.
Keep >=95% usable coverage, >=20 backgrounds/band, near >=90%, long/fast >=95%, sham <=5%; sham initial quiet now uses qualifying-window starts.
Keep all-person control gate, censoring bounds, 12 ms, threshold agreement and >5% unresolved rule; smoke always UNDERIDENTIFIED. Otherwise empirical p95 FAIL/BELOW_ONLY; no worst-case/188-ms certification or scoring clearance.
Close Stage 3 before provider access; provider ticks round only within 1e-6 tick. Full assumptions/rules accompany code; control-passing subsets and provider agreement remain descriptive.
"""

ESTIMATE_FIELDS = (
    "status", "reason", "R_tick", "lag_ms", "lag_lower_ms", "lag_upper_ms",
    "candidate_tick", "candidate_window_start_tick", "bout_end_tick", "n_bouts",
)
RAW_FIELDS = (
    "participant", "session", "run", "cue", "fs", "k",
    "L_tick", "S_tick", "theta_g",
    "search_start_tick", "search_end_tick", "search_capped",
    "anchor_start_tick", "anchor_end_tick",
    "initial_quiet_40ms", "earliest_initial_quiet_40ms",
    "any_qualifying_before_L",
    *ESTIMATE_FIELDS,
    *(f"earliest_{name}" for name in ESTIMATE_FIELDS),
)
NOISE_FIELDS = (
    "participant", "fs", "runs", "start_tones", "still_windows",
    "fit_windows", "validation_windows", "reference_samples",
    "median_g", "mad_g", "sigma_g",
    "theta_k3_g", "theta_k5_g", "theta_k8_g", "usable",
    "control_intervals", "control_intervals_with_room",
    "control_eligible", "control_selected",
    "sham_alarm_fraction_k3", "sham_alarm_fraction_k5",
    "sham_alarm_fraction_k8",
)
TEMPLATE_FIELDS = (
    "participant", "fs", "family", "target_detectable_lag_ms",
    "calibration_status", "calibration_reason",
    "theta_k5_g", "amplitude", "amplitude_units",
    "T0_tick", "S_tick", "search_start_tick", "L_nl_tick", "T_det_k5_tick",
    "noiseless_physical_lag_ms", "noiseless_detectable_lag_k5_ms",
    "growth_stop_tick", "calibration_candidate_kind",
    "family_candidates_evaluated",
)
CONTROL_FIELDS = (
    "participant", "fs", "background_id", "background_session",
    "background_run", "background_start_tick", "background_end_tick",
    "background_still_start_tick", "background_tone_tick",
    "background_previous_end_tone_tick", "background_interval_start_tick",
    "kind", "family", "target_detectable_lag_ms",
    "calibration_status", "calibration_reason",
    "noiseless_physical_lag_ms", "noiseless_detectable_lag_k5_ms",
    "L_nl_tick", "growth_stop_tick", "scale_coefficient",
    "direction_x", "direction_y", "direction_z", "k",
    "control_status", "T0_tick", "S_tick", "T_det_tick", "L_tick",
    "true_lag_ms", "sub_noise_lead_ms", "T_det_minus_T0_ms",
    "truth_label_continuous", "truth_primary_R_tick",
    "status", "reason", "R_tick", "recovered_lag_ms", "n_bouts",
    "candidate_window_start_tick", "anchor_start_tick", "anchor_end_tick",
    "search_start_tick", "search_end_tick", "initial_quiet_40ms",
    "earliest_status", "earliest_R_tick", "earliest_recovered_lag_ms",
    "sham_alarm",
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, fields, rows, *, replace=False):
    with Path(path).open(
        "w" if replace else "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def require_slurm(workers):
    host = socket.gethostname()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in host.lower():
        raise RuntimeError(
            "Actual Slurm compute-node shell required: SLURM_JOB_ID must "
            "be set and hostname must lack '-ln'."
        )
    if os.environ.get("SLURM_PROCID", "0") != "0":
        raise RuntimeError("Launch one coordinating Python process.")
    if workers < 1:
        raise ValueError("--workers must be positive")
    declared = (
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_CPUS_ON_NODE")
    )
    if not declared or not re.fullmatch(r"\d+", declared):
        raise RuntimeError("An explicit numeric Slurm CPU limit is required.")
    limit = int(declared)
    if hasattr(os, "sched_getaffinity"):
        limit = min(limit, len(os.sched_getaffinity(0)))
    if workers > limit:
        raise ValueError(f"{workers} workers exceed CPU limit {limit}.")
    return host, limit


def projected_csv(path, columns, *, allow_provider=False):
    """Project selected fields; excluded provider values are never used.

    CSV tokenization traverses the physical record. Excluded values are not
    indexed, converted, tested, or retained in the projected output.
    """
    if not allow_provider and PROVIDER_COLUMNS.intersection(columns):
        raise RuntimeError("Provider access attempted before Stage 4")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV headers")
        indices = [header.index(name) for name in columns]
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError("Malformed CSV record")
            yield {name: row[index] for name, index in zip(columns, indices)}


def optional_int(value):
    """Strict parser for non-provider integer columns."""
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "null", "na"):
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    if number != math.floor(number):
        raise ValueError(f"Expected integer tick/index, got {value!r}")
    return int(number)


def required_int(value):
    answer = optional_int(value)
    if answer is None:
        raise ValueError("Required integer is missing")
    return answer


def provider_tick(value):
    """Stage 4 only: tolerate at most 1e-6 tick of serialization error."""
    text = str(value).strip()
    if text.lower() in ("", "nan", "none", "null", "na"):
        return None
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite provider tick: {value!r}")
    rounded = round(number)
    if abs(number - rounded) > PROVIDER_TICK_TOLERANCE:
        raise ValueError(f"Provider tick is not within 1e-6 of an integer: {value!r}")
    return int(rounded)


def event_key(row):
    return (
        str(row["participant"]), str(row["session"]),
        int(row["run"]), str(row["cue"]),
    )


def stem_key(stem):
    match = re.fullmatch(
        r"(sub-\d+)_(ses-\d+)_task-reachingandgrasping_run-(\d+)_eeg",
        Path(stem).name,
    )
    if match is None:
        raise ValueError(f"Unexpected development stem: {stem}")
    return match.group(1), match.group(2), int(match.group(3))


def native_residual(acc, fs):
    """Exact half-open [t-520,t-20) native baseline."""
    acc = np.asarray(acc, dtype=np.float64)
    width = int(round(0.500 * fs))
    exclusion = int(round(0.020 * fs))
    past_mean = S.trailing_mean(acc, width)
    baseline = np.full_like(acc, np.nan)
    # trailing_mean includes its own endpoint; the baseline endpoint excludes it.
    shift = exclusion + 1
    if len(acc) > shift:
        baseline[shift:] = past_mean[:-shift]
    return acc - baseline


def native_statistic(acc, fs):
    return np.linalg.norm(native_residual(acc, fs), axis=1)


def native_validity(raw_r, detector_valid, fs):
    if fs == 250.0:
        indices = np.arange(len(raw_r))
    elif fs == 1000.0:
        indices = (np.arange(len(raw_r)) - 3) // 4
    else:
        raise ValueError("Unsupported native rate")
    supported = (indices >= 0) & (indices < len(detector_valid))
    out = np.zeros(len(raw_r), dtype=bool)
    out[supported] = detector_valid[indices[supported]]
    return out & np.isfinite(raw_r)


def still_screen(acc, valid, begin, end):
    """The supplied stationary-window screen, unchanged."""
    if begin < 0 or end > len(acc) or end <= begin:
        return False
    if (end - begin) % F.SP_BLOCK_SAMPLES:
        return False
    if not np.all(valid[begin:end]):
        return False
    section = acc[begin:end]
    blocks = section.reshape(-1, F.SP_BLOCK_SAMPLES, 3).mean(axis=1)
    spread = np.linalg.norm(blocks - section.mean(axis=0), axis=1)
    return bool(np.max(spread) <= F.SP_STILL_BLOCK_G)


def latest_background_end(
    ticks, acc_prefix, bad_prefix, previous_end_tick, start_tick,
    last_eligible_end,
):
    """Latest eligible detector-grid segment in one end-to-start interval.

    Evaluates all candidate endpoints vectorially, with a fixed 19-block loop.
    The full 700 detector samples must be valid; only the final 475 samples
    undergo the stillness screen. A detector block j has raw support beginning
    at tick 4*j for either native rate.
    """
    total_n = BACKGROUND_MS // 4
    still_n = STILL_MS // 4
    interval_start = int(previous_end_tick) + END_TONE_OFFSET_MS
    first_begin = max((interval_start + 3) // 4, last_eligible_end)
    first_end = first_begin + total_n
    last_end = int(np.searchsorted(ticks, int(start_tick), side="left"))
    if last_end < first_end:
        return None, False

    ends = np.arange(last_end, first_end - 1, -1, dtype=np.int64)
    valid = bad_prefix[ends] - bad_prefix[ends - total_n] == 0
    ends = ends[valid]
    if not len(ends):
        return None, True

    means = (acc_prefix[ends] - acc_prefix[ends - still_n]) / still_n
    keep = np.ones(len(ends), dtype=bool)
    for block in range(still_n // F.SP_BLOCK_SAMPLES):
        begin = ends - still_n + block * F.SP_BLOCK_SAMPLES
        block_mean = (
            acc_prefix[begin + F.SP_BLOCK_SAMPLES] - acc_prefix[begin]
        ) / F.SP_BLOCK_SAMPLES
        keep &= (
            np.linalg.norm(block_mean - means, axis=1)
            <= F.SP_STILL_BLOCK_G
        )
    good = np.flatnonzero(keep)
    return (int(ends[good[0]]) if len(good) else None), True


def qualifying_windows(values, theta, width):
    """q[j] describes the complete width-sample window starting at j."""
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Qualification received nonfinite samples")
    hot = values > theta
    if len(values) < width:
        return hot, np.zeros(0, dtype=bool), np.empty(0, int), np.empty(0, int)
    cumulative = np.r_[0, np.cumsum(hot, dtype=np.int64)]
    need = int(math.ceil(OCCUPANCY * width - 1e-12))
    q = cumulative[width:] - cumulative[:-width] >= need
    starts = np.flatnonzero(q)
    if not len(starts):
        return hot, q, starts, np.empty(0, dtype=np.int64)
    hot_indices = np.flatnonzero(hot)
    first_hot = hot_indices[np.searchsorted(hot_indices, starts)]
    return hot, q, starts, first_hot


def grouped_bouts(hot, starts, first_hot, width, quiet_width, *, window_quiet):
    """Rows: first exceedance, last qualifying-window end, first window start."""
    if not len(starts):
        return np.empty((0, 3), dtype=np.int64)

    if window_quiet:
        # Between successive qualifying starts, diff-1 starts are quiet.
        first = np.r_[0, np.flatnonzero(np.diff(starts) > quiet_width) + 1]
    else:
        # Unchanged descriptive sensitivity: every sample quiet.
        edges = np.flatnonzero(np.diff(np.r_[0, (~hot).astype(np.int8), 0]))
        quiet_starts, quiet_ends = edges[::2], edges[1::2]
        reset_ends = quiet_ends[quiet_ends - quiet_starts >= quiet_width]
        groups = np.searchsorted(reset_ends, first_hot, side="right")
        first = np.flatnonzero(np.r_[True, np.diff(groups) != 0])

    last = np.r_[first[1:] - 1, len(starts) - 1]
    return np.column_stack((
        first_hot[first], starts[last] + width - 1, starts[first],
    )).astype(np.int64)


def empty_estimate():
    return {
        "status": "indeterminate", "reason": "",
        "R_tick": "", "lag_ms": "",
        "lag_lower_ms": 0.0, "lag_upper_ms": float("inf"),
        "candidate_tick": "", "candidate_window_start_tick": "",
        "bout_end_tick": "", "n_bouts": 0,
    }


def record_candidate(result, bout, begin, dt, L):
    tick = int((begin + int(bout[0])) * dt)
    result.update(
        candidate_tick=tick,
        candidate_window_start_tick=int((begin + int(bout[2])) * dt),
        bout_end_tick=int((begin + int(bout[1])) * dt),
        lag_lower_ms=float(L - tick),
    )


def resolve_candidate(result, reason):
    lag = float(result["lag_lower_ms"])
    result.update(
        status="determinate", reason=reason,
        R_tick=int(result["candidate_tick"]),
        lag_ms=lag, lag_lower_ms=lag, lag_upper_ms=lag,
    )


def estimate(raw_r, native_ok, fs, L, start_known, theta):
    dt = int(round(1000 / fs))
    start = max(int(start_known) - 500, int(L) - SEARCH_MS)
    finish = int(L) + LOOKAHEAD_MS
    primary, earliest = empty_estimate(), empty_estimate()
    common = {
        "search_start_tick": start,
        "search_end_tick": finish,
        "search_capped": int(int(L) - SEARCH_MS > int(start_known) - 500),
        "anchor_start_tick": "", "anchor_end_tick": "",
        "initial_quiet_40ms": "", "earliest_initial_quiet_40ms": "",
        "any_qualifying_before_L": "",
    }

    def packed():
        return {
            **common, **primary,
            **{f"earliest_{key}": value for key, value in earliest.items()},
        }

    def invalidate(reason):
        for result in (primary, earliest):
            result.update(status="no_valid_data", reason=reason)
        return packed()

    if not math.isfinite(theta):
        return invalidate("unusable_noise_reference")

    begin = (start + dt - 1) // dt
    end = finish // dt + 1
    width = PERSIST_MS // dt
    short_quiet = EARLIEST_QUIET_MS // dt
    anchor_width = PRIMARY_QUIET_MS // dt
    if begin < 0 or end > len(raw_r) or end - begin < anchor_width + width:
        return invalidate("incomplete_search_or_lookahead_support")
    if not np.all(native_ok[begin:end]):
        return invalidate("validity_gap_including_lookahead")

    values = raw_r[begin:end]
    label_index = int(L) // dt - begin
    hot, q, starts, first_hot = qualifying_windows(values, theta, width)

    # q has enough observed support for every start tested here.
    initial_window_quiet = len(q) >= short_quiet and not np.any(q[:short_quiet])
    initial_sample_quiet = bool(np.all(values[:short_quiet] <= theta))
    common["initial_quiet_40ms"] = int(initial_window_quiet)
    common["earliest_initial_quiet_40ms"] = int(initial_sample_quiet)
    common["any_qualifying_before_L"] = int(np.any(first_hot <= label_index))

    # Unchanged descriptive earliest-bout sensitivity.
    old_bouts = grouped_bouts(
        hot, starts, first_hot, width, short_quiet, window_quiet=False
    )
    old_bouts = old_bouts[old_bouts[:, 0] <= label_index]
    earliest["n_bouts"] = int(len(old_bouts))
    if len(old_bouts):
        record_candidate(earliest, old_bouts[0], begin, dt, L)
    if not initial_sample_quiet:
        earliest["reason"] = "missing_initial_40ms_quiet_left_censored"
    elif not len(old_bouts):
        earliest["reason"] = "no_persistent_departure_at_or_before_label"
    else:
        resolve_candidate(
            earliest,
            "multiple_bouts_earliest_retained" if len(old_bouts) > 1 else "single_bout",
        )

    bouts = grouped_bouts(
        hot, starts, first_hot, width, anchor_width, window_quiet=True
    )
    bouts = bouts[bouts[:, 0] <= label_index]
    primary["n_bouts"] = int(len(bouts))
    reaching = bouts[bouts[:, 1] >= label_index]
    if len(reaching) > 1:
        raise RuntimeError("More than one separated primary bout spans L")
    if not len(reaching):
        primary["reason"] = (
            "no_persistent_departure_at_or_before_label"
            if not len(bouts) else "no_bout_reaches_label"
        )
        return packed()

    bout = reaching[0]
    record_candidate(primary, bout, begin, dt, L)

    # Anchor in WINDOW-START coordinates, excluding the first qualifying start.
    # Testing [R-200,R) instead would let this window invalidate its own anchor.
    first_window = int(bout[2])
    anchor_begin = first_window - anchor_width
    anchored = (
        anchor_begin >= 0
        and first_window <= len(q)
        and not np.any(q[anchor_begin:first_window])
    )
    if not anchored:
        primary["reason"] = "missing_200ms_quiet_anchor_left_censored"
        return packed()

    common["anchor_start_tick"] = int((begin + anchor_begin) * dt)
    common["anchor_end_tick"] = int((begin + first_window) * dt)
    resolve_candidate(
        primary,
        "label_continuous_bout_prior_bouts_ignored"
        if len(bouts) > 1 else "single_label_continuous_bout",
    )
    return packed()


def first_detectable_tick(noiseless_r, fs, theta):
    if not math.isfinite(theta):
        return None
    dt = int(round(1000 / fs))
    begin = T0_MS // dt
    values = noiseless_r[begin:]
    if not len(values) or not np.all(np.isfinite(values)):
        return None
    _, _, starts, first_hot = qualifying_windows(values, theta, PERSIST_MS // dt)
    return int((begin + first_hot[0]) * dt) if len(starts) else None


def reference_theta(reference, k):
    return (
        float(reference[f"theta_k{k}_g"])
        if reference["usable"] else float("nan")
    )


def stage1_person(task):
    participant, stems, cues, background_dir = task
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, int(participant.split("-")[1]), 1])
    )
    references, backgrounds = [], []
    counts = {
        "runs": 0, "start_tones": 0, "still_windows": 0,
        "fit_windows": 0, "validation_windows": 0,
        "control_intervals": 0, "control_intervals_with_room": 0,
        "control_eligible": 0,
    }
    expected_fs = 1000.0 if participant == "sub-15" else 250.0

    for stem in stems:
        run, acc, ticks, dev, valid, bursts, _ = sp_load(stem)
        if run.participant != participant or run.fs != expected_fs:
            raise ValueError("Participant/rate mismatch")
        counts["runs"] += 1
        if (
            participant == "sub-22" and run.session == "ses-01"
            and run.run in (6, 7) and np.any(valid)
        ):
            raise ValueError("Known flat-axis run contains valid detector samples")

        raw_r = native_statistic(run.acc_g, run.fs)
        factor = int(round(run.fs / R.SP_DETECTOR_HZ))
        starts = sorted(
            [b for b in bursts if b.get("tone") and b["class"] == "start"],
            key=lambda b: b["onset_tick"],
        )
        end_tones = np.asarray(sorted(
            int(b["onset_tick"]) for b in bursts
            if b.get("tone") and b["class"] == "end"
        ), dtype=np.int64)
        counts["start_tones"] += len(starts)

        # Prefix sums support vectorized background searches. Invalid data
        # cannot be accepted because the full-segment bad-count must be zero.
        acc_prefix = np.vstack([
            np.zeros((1, 3)),
            np.cumsum(np.where(np.isfinite(acc), acc, 0.0), axis=0),
        ])
        bad_prefix = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
        last_eligible_end = 0

        for start_index, burst in enumerate(starts):
            tone_tick = int(burst["onset_tick"])
            end = int(np.searchsorted(ticks, tone_tick))
            begin = end - F.SP_NOISE_SAMPLES

            # Noise reference membership remains exactly the frozen rule.
            if still_screen(acc, valid, begin, end):
                counts["still_windows"] += 1
                split = "fit_windows" if start_index % 2 == 0 else "validation_windows"
                counts[split] += 1
                native_begin, native_end = begin * factor, end * factor
                ref_begin = native_begin + int(round(0.520 * run.fs))
                values = raw_r[ref_begin:native_end]
                if (
                    len(values) != int(round(0.480 * run.fs))
                    or not np.all(np.isfinite(values))
                ):
                    raise ValueError("Native reference support disagrees with frozen window")
                references.append(values.copy())

            # Background eligibility is independent of event estimates and
            # whether the final second immediately before this tone was accepted.
            previous_index = int(np.searchsorted(end_tones, tone_tick, side="left")) - 1
            if previous_index < 0:
                continue
            previous_end_tick = int(end_tones[previous_index])
            counts["control_intervals"] += 1
            selected_end, had_room = latest_background_end(
                ticks, acc_prefix, bad_prefix, previous_end_tick, tone_tick,
                last_eligible_end,
            )
            counts["control_intervals_with_room"] += int(had_room)
            if selected_end is None:
                continue

            selected_begin = selected_end - BACKGROUND_MS // 4
            first_native, last_native = selected_begin * factor, selected_end * factor
            segment = run.acc_g[first_native:last_native]
            first_tick = first_native * run.ticks_per_sample
            final_tick = last_native * run.ticks_per_sample
            interval_start = previous_end_tick + END_TONE_OFFSET_MS
            if (
                len(segment) != int(BACKGROUND_MS * run.fs / 1000)
                or not np.all(np.isfinite(segment))
                or first_tick < interval_start
                or final_tick > tone_tick
                or final_tick - first_tick != BACKGROUND_MS
            ):
                raise ValueError("Control background violates interval/native support")

            # Make all eligible candidates nonoverlapping, before reservoir selection.
            last_eligible_end = selected_end
            counts["control_eligible"] += 1
            item = {
                "acc": segment.copy(),
                "session": run.session, "run": run.run,
                "start_tick": first_tick, "end_tick": final_tick,
                "tone_tick": tone_tick,
                "previous_end_tone_tick": previous_end_tick,
                "interval_start_tick": interval_start,
            }
            if len(backgrounds) < MAX_BACKGROUNDS:
                backgrounds.append(item)
            else:
                replacement = int(rng.integers(counts["control_eligible"]))
                if replacement < MAX_BACKGROUNDS:
                    backgrounds[replacement] = item

    values = np.concatenate(references) if references else np.empty(0)
    median = float(np.median(values)) if len(values) else float("nan")
    mad = float(np.median(np.abs(values - median))) if len(values) else float("nan")
    sigma = 1.4826 * mad
    usable = bool(
        counts["still_windows"] >= MIN_NOISE_WINDOWS
        and math.isfinite(median) and math.isfinite(sigma) and sigma > 0
    )
    reference = {
        "participant": participant, "fs": expected_fs, **counts,
        "reference_samples": int(len(values)),
        "median_g": median, "mad_g": mad, "sigma_g": sigma,
        "usable": usable, "control_selected": len(backgrounds),
    }
    for k in KS:
        reference[f"theta_k{k}_g"] = median + k * sigma if usable else float("nan")
        reference[f"sham_alarm_fraction_k{k}"] = ""

    backgrounds.sort(key=lambda item: (item["session"], item["run"], item["tone_tick"]))
    cache = Path(background_dir) / f"{participant}.npz"
    length = int(BACKGROUND_MS * expected_fs / 1000)
    arrays = {
        "acc": (
            np.stack([item["acc"] for item in backgrounds])
            if backgrounds else np.empty((0, length, 3))
        ),
        "fs": np.asarray(expected_fs),
        "session": np.asarray([item["session"] for item in backgrounds], dtype="U32"),
    }
    for field in (
        "run", "start_tick", "end_tick", "tone_tick",
        "previous_end_tone_tick", "interval_start_tick",
    ):
        arrays[field] = np.asarray([item[field] for item in backgrounds], dtype=np.int64)
    np.savez_compressed(cache, **arrays)

    cues_by_run = defaultdict(list)
    for cue in cues:
        cues_by_run[(cue["session"], cue["run"])].append(cue)

    estimates = []
    for stem in stems:
        _, session, number = stem_key(stem)
        events = cues_by_run.get((session, number), [])
        if not events:
            continue
        run, acc, ticks, dev, valid, bursts, _ = sp_load(stem)
        raw_r = native_statistic(run.acc_g, run.fs)
        native_ok = native_validity(raw_r, valid, run.fs)
        phase = 3 if run.fs == 1000.0 else 0

        for cue in events:
            L, start_known = cue["L_tick"], cue["S_tick"]
            if L % 4 != phase:
                raise ValueError("Frozen onset is not on its availability clock")
            if L < start_known or L > (run.frames - 1) * run.ticks_per_sample:
                raise ValueError("Frozen onset is outside its temporal range")
            for k in KS:
                theta = reference_theta(reference, k)
                estimates.append({
                    "participant": participant, "session": session,
                    "run": number, "cue": cue["cue"], "fs": run.fs, "k": k,
                    "L_tick": L, "S_tick": start_known, "theta_g": theta,
                    **estimate(raw_r, native_ok, run.fs, L, start_known, theta),
                })
    return reference, estimates, str(cache)


def control_features(acc, fs):
    reduced, ticks = R.reduce_to_detector_rate(acc, fs)
    departure = S.self_paced_departure(reduced)
    sd = R.trailing_sd(reduced, R.SP_DETECTOR_HZ)
    valid = np.isfinite(departure) & np.all(sd >= F.SP_FLAT_SD_G, axis=1)
    raw_r = native_statistic(acc, fs)
    return raw_r, native_validity(raw_r, valid, fs), departure, ticks, valid


def frozen_crossing(departure, ticks, low, high, valid=None):
    """HIGH at j; LOW at j and the following ten samples (40 ms)."""
    width = F.SP_CONFIRM + 1
    if len(departure) < width:
        return None
    ok = np.isfinite(departure) & (departure >= low)
    if valid is not None:
        ok &= valid
    counts = np.r_[0, np.cumsum(ok, dtype=np.int64)]
    sustained = counts[width:] - counts[:-width] == width
    length = len(sustained)
    candidates = np.flatnonzero(
        sustained
        & (departure[:length] >= high)
        & (ticks[:length] >= CONTROL_S_MS)
    )
    return int(ticks[candidates[0]]) if len(candidates) else None


def control_support(L, T_det, fs):
    dt = int(round(1000 / fs))
    last_tick = BACKGROUND_MS - dt
    B = max(CONTROL_S_MS - 500, int(L) - SEARCH_MS)
    return bool(
        L >= CONTROL_S_MS
        and B >= WARMUP_MS + 520
        and T_det <= L
        and T_det - PRIMARY_QUIET_MS >= B
        and L + LOOKAHEAD_MS <= last_tick
    )


def template_parts(fs, family, stop_tick=None):
    n = int(BACKGROUND_MS * fs / 1000)
    dt = int(round(1000 / fs))
    ticks = np.arange(n, dtype=np.int64) * dt
    active = ticks >= T0_MS
    u = np.where(active, (ticks - T0_MS + dt) / 1000.0, 0.0)
    growth_ticks = ticks if stop_tick is None else np.minimum(ticks, stop_tick)
    growth_u = np.where(active, (growth_ticks - T0_MS + dt) / 1000.0, 0.0)
    pulse = np.zeros(n, dtype=np.float64)
    if family == "linear":
        unit = growth_u
    elif family == "quadratic":
        unit = growth_u * growth_u
    elif family == "pulse_ramp":
        inside = active & (u <= 0.120)
        pulse[inside] = 0.04 * np.sin(np.pi * u[inside] / 0.120) ** 2
        unit = np.maximum(growth_u - 0.180, 0.0) ** 2
    else:
        raise ValueError(f"Unknown family: {family}")
    return pulse, unit


def template_signal(fs, family, amplitude, stop_tick):
    pulse, unit = template_parts(fs, family, stop_tick)
    return pulse + amplitude * unit


def signed_detector_component(signal, fs):
    """Linear coefficients for calibration; final verification uses supplied summary."""
    component = np.zeros((len(signal), 3), dtype=np.float64)
    component[:, 0] = signal
    reduced, ticks = R.reduce_to_detector_rate(component, fs)
    recent = S.trailing_mean(reduced, S.SP_SUMMARY_SAMPLES)
    base = S.trailing_mean(reduced, S.SP_BASE_SAMPLES)
    signed = np.full(len(reduced), np.nan)
    lag = S.SP_SUMMARY_SAMPLES
    signed[lag:] = recent[lag:, 0] - base[:-lag, 0]
    return signed, ticks


def amplitude_candidates(raw_pulse, raw_unit, dev_pulse, dev_unit, theta, low, high):
    """Cover positive-amplitude threshold intervals and their breakpoints."""
    roots = []
    for pulse, unit, levels in (
        (raw_pulse, raw_unit, (theta,)),
        (dev_pulse, dev_unit, (low, high)),
    ):
        use = np.isfinite(pulse) & np.isfinite(unit) & (unit != 0)
        p, u = pulse[use], unit[use]
        for level in levels:
            for sign in (-1.0, 1.0):
                values = (sign * level - p) / u
                values = values[np.isfinite(values) & (values > 0)]
                if len(values):
                    roots.append(values)

    if not roots:
        return [(1.0, "constant_threshold_pattern")]
    roots = np.unique(np.concatenate(roots))
    tail = float(roots[-1] + max(1.0, abs(float(roots[-1]))))
    if not math.isfinite(tail) or tail <= roots[-1]:
        raise ValueError("Nonfinite amplitude partition")

    bounds = np.r_[0.0, roots, tail]
    left, right = bounds[:-1], bounds[1:]
    middle = left + (right - left) / 2.0
    valid = (
        np.isfinite(middle) & (middle > 0)
        & (middle > left) & (middle < right)
    )
    middle, left, right = middle[valid], left[valid], right[valid]
    relative_width = (right - left) / np.maximum(
        np.abs(middle), np.finfo(np.float64).tiny
    )
    order = np.argsort(-relative_width, kind="stable")
    candidates = [
        (float(middle[index]), "threshold_interval_midpoint") for index in order
    ]
    candidates.extend((float(value), "threshold_breakpoint") for value in roots)
    return candidates


def noiseless_features(signal, fs, theta, low, high):
    component = np.zeros((len(signal), 3), dtype=np.float64)
    component[:, 0] = signal
    reduced, ticks = R.reduce_to_detector_rate(component, fs)
    departure = S.self_paced_departure(reduced)
    raw_r = native_statistic(component, fs)
    return (
        raw_r,
        frozen_crossing(departure, ticks, low, high),
        first_detectable_tick(raw_r, fs, theta),
    )


def calibrate_family(participant, fs, family, targets, theta, low, high):
    rows = {
        target: {
            "participant": participant, "fs": fs, "family": family,
            "target_detectable_lag_ms": target,
            "calibration_status": "uncovered",
            "calibration_reason": "no_exact_target_with_declared_support",
            "theta_k5_g": theta, "amplitude": "",
            "amplitude_units": "g/s" if family == "linear" else "g/s^2",
            "T0_tick": T0_MS, "S_tick": CONTROL_S_MS,
            "search_start_tick": CONTROL_SEARCH_START_MS,
            "L_nl_tick": "", "T_det_k5_tick": "",
            "noiseless_physical_lag_ms": "",
            "noiseless_detectable_lag_k5_ms": "",
            "growth_stop_tick": "", "calibration_candidate_kind": "",
            "family_candidates_evaluated": 0,
        }
        for target in targets
    }
    if not math.isfinite(theta) or theta <= 0:
        for row in rows.values():
            row["calibration_reason"] = "unusable_k5_noise_reference"
        return list(rows.values())

    pulse, unit = template_parts(fs, family)
    raw_pulse = native_residual(pulse[:, None], fs)[:, 0]
    raw_unit = native_residual(unit[:, None], fs)[:, 0]
    dev_pulse, ticks = signed_detector_component(pulse, fs)
    dev_unit, other_ticks = signed_detector_component(unit, fs)
    if not np.array_equal(ticks, other_ticks):
        raise RuntimeError("Noiseless component clocks disagree")
    candidates = amplitude_candidates(
        raw_pulse, raw_unit, dev_pulse, dev_unit, theta, low, high
    )

    remaining = set(targets)
    evaluations = 0
    for amplitude, candidate_kind in candidates:
        evaluations += 1
        L = frozen_crossing(np.abs(dev_pulse + amplitude * dev_unit), ticks, low, high)
        if L is None:
            continue
        raw_r = np.abs(raw_pulse + amplitude * raw_unit)
        T_det = first_detectable_tick(raw_r, fs, theta)
        if T_det is None or not control_support(L, T_det, fs):
            continue
        target = int(L - T_det)
        if target not in remaining:
            continue

        stop_tick = int(L + LOOKAHEAD_MS)
        signal = template_signal(fs, family, amplitude, stop_tick)
        _, checked_L, checked_T = noiseless_features(signal, fs, theta, low, high)
        if (
            checked_L is None or checked_T is None
            or checked_L != L or checked_T != T_det
            or checked_L - checked_T != target
            or stop_tick != checked_L + LOOKAHEAD_MS
            or not control_support(checked_L, checked_T, fs)
        ):
            continue

        rows[target].update(
            calibration_status="calibrated",
            calibration_reason="exact_detectable_target_verified_after_growth_cap",
            amplitude=float(amplitude),
            L_nl_tick=int(checked_L), T_det_k5_tick=int(checked_T),
            noiseless_physical_lag_ms=int(checked_L - T0_MS),
            noiseless_detectable_lag_k5_ms=int(checked_L - checked_T),
            growth_stop_tick=stop_tick,
            calibration_candidate_kind=candidate_kind,
        )
        remaining.remove(target)
        if not remaining:
            break

    for row in rows.values():
        row["family_candidates_evaluated"] = evaluations
    return [rows[target] for target in targets]


def stage2_calibrate_person(task):
    participant, fs, theta, low, high = task
    rows = []
    for family, targets in CONTROL_SPECS.items():
        rows.extend(calibrate_family(participant, fs, family, targets, theta, low, high))
    return rows


def estimate_control_fields(est):
    return {
        "status": est["status"], "reason": est["reason"],
        "R_tick": est["R_tick"], "recovered_lag_ms": est["lag_ms"],
        "n_bouts": est["n_bouts"],
        "candidate_window_start_tick": est["candidate_window_start_tick"],
        "anchor_start_tick": est.get("anchor_start_tick", ""),
        "anchor_end_tick": est.get("anchor_end_tick", ""),
        "search_start_tick": est.get("search_start_tick", ""),
        "search_end_tick": est.get("search_end_tick", ""),
        "initial_quiet_40ms": est.get("initial_quiet_40ms", ""),
        "earliest_status": est["earliest_status"],
        "earliest_R_tick": est["earliest_R_tick"],
        "earliest_recovered_lag_ms": est["earliest_lag_ms"],
    }


def missing_control_estimate(reason="no_frozen_crossing"):
    result = empty_estimate()
    result["reason"] = reason
    result.update({
        f"earliest_{key}": value for key, value in empty_estimate().items()
    })
    result["earliest_reason"] = reason
    return result


def stage2_person(task):
    participant, cache_path, reference, low, high, calibration_rows = task
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, int(participant.split("-")[1]), 2])
    )
    fs = float(reference["fs"])
    templates = {}

    for calibration in calibration_rows:
        key = (calibration["family"], calibration["target_detectable_lag_ms"])
        if key in templates:
            raise ValueError("Duplicate calibration cell")
        if calibration["calibration_status"] != "calibrated":
            templates[key] = None
            continue
        signal = template_signal(
            fs, calibration["family"], float(calibration["amplitude"]),
            int(calibration["growth_stop_tick"]),
        )
        noiseless_r, L_nl, T_det5 = noiseless_features(
            signal, fs, reference_theta(reference, 5), low, high
        )
        if (
            L_nl is None or T_det5 is None
            or L_nl != calibration["L_nl_tick"]
            or T_det5 != calibration["T_det_k5_tick"]
            or L_nl - T_det5 != calibration["target_detectable_lag_ms"]
            or calibration["growth_stop_tick"] != L_nl + LOOKAHEAD_MS
            or not control_support(L_nl, T_det5, fs)
        ):
            raise RuntimeError("Locked calibration did not reproduce")
        templates[key] = signal, noiseless_r

    expected_keys = {
        (family, target)
        for family, targets in CONTROL_SPECS.items() for target in targets
    }
    if set(templates) != expected_keys:
        raise ValueError("Missing or extra locked calibration cells")

    rows = []
    with np.load(cache_path, allow_pickle=False) as cache:
        if float(cache["fs"].item()) != fs:
            raise ValueError("Background/calibration rate mismatch")
        for background_id, background in enumerate(cache["acc"]):
            first_tick = int(cache["start_tick"][background_id])
            provenance = {
                "participant": participant, "fs": fs,
                "background_id": background_id,
                "background_session": str(cache["session"][background_id]),
                "background_run": int(cache["run"][background_id]),
                "background_start_tick": first_tick,
                "background_end_tick": int(cache["end_tick"][background_id]),
                "background_still_start_tick": first_tick + WARMUP_MS,
                "background_tone_tick": int(cache["tone_tick"][background_id]),
                "background_previous_end_tone_tick": int(
                    cache["previous_end_tone_tick"][background_id]
                ),
                "background_interval_start_tick": int(
                    cache["interval_start_tick"][background_id]
                ),
            }

            raw_r, native_ok, departure, ticks, valid = control_features(background, fs)
            sham_L = SHAM_L_MS + (3 if fs == 1000.0 else 0)
            for k in KS:
                est = estimate(
                    raw_r, native_ok, fs, sham_L, CONTROL_S_MS,
                    reference_theta(reference, k),
                )
                alarm = (
                    est["status"] == "no_valid_data"
                    or est["any_qualifying_before_L"] == 1
                    or est["initial_quiet_40ms"] != 1
                )
                rows.append({
                    **provenance, "kind": "sham", "family": "sham", "k": k,
                    "S_tick": CONTROL_S_MS, "L_tick": sham_L,
                    "control_status": "sham",
                    **estimate_control_fields(est),
                    "sham_alarm": int(alarm),
                })

            for calibration in calibration_rows:
                family = calibration["family"]
                target = calibration["target_detectable_lag_ms"]
                template = templates[(family, target)]
                # Consume a direction for uncovered cells too.
                direction = rng.normal(size=3)
                length = float(np.linalg.norm(direction))
                if length == 0:
                    raise RuntimeError("Zero synthetic direction")
                direction /= length
                cell = {
                    **provenance, "kind": "injected", "family": family,
                    "target_detectable_lag_ms": target,
                    "calibration_status": calibration["calibration_status"],
                    "calibration_reason": calibration["calibration_reason"],
                    "noiseless_physical_lag_ms": calibration["noiseless_physical_lag_ms"],
                    "noiseless_detectable_lag_k5_ms": calibration["noiseless_detectable_lag_k5_ms"],
                    "L_nl_tick": calibration["L_nl_tick"],
                    "growth_stop_tick": calibration["growth_stop_tick"],
                    "scale_coefficient": calibration["amplitude"],
                    "direction_x": float(direction[0]),
                    "direction_y": float(direction[1]),
                    "direction_z": float(direction[2]),
                    "T0_tick": T0_MS, "S_tick": CONTROL_S_MS, "sham_alarm": "",
                }

                if template is None:
                    for k in KS:
                        rows.append({
                            **cell, "k": k, "control_status": "uncovered_template",
                            **estimate_control_fields(missing_control_estimate(
                                "uncovered_noiseless_template"
                            )),
                        })
                    continue

                signal, noiseless_r = template
                injected = background + signal[:, None] * direction
                raw_r, native_ok, departure, ticks, valid = control_features(injected, fs)
                L = frozen_crossing(departure, ticks, low, high, valid)
                noiseless_ok = np.isfinite(noiseless_r)

                for k in KS:
                    theta = reference_theta(reference, k)
                    T_det = first_detectable_tick(noiseless_r, fs, theta)
                    truth_primary, truth_continuous = None, None
                    if L is None:
                        est = missing_control_estimate()
                        control_status = "no_crossing"
                    else:
                        est = estimate(raw_r, native_ok, fs, L, CONTROL_S_MS, theta)
                        if T_det is not None:
                            truth_primary = estimate(
                                noiseless_r, noiseless_ok, fs, L, CONTROL_S_MS, theta
                            )
                            truth_continuous = bool(
                                truth_primary["status"] == "determinate"
                                and truth_primary["R_tick"] == T_det
                            )
                        if L < T0_MS:
                            control_status = "pre_injection_crossing"
                        elif T_det is None:
                            control_status = "no_detectable_component_onset"
                        elif T_det > L:
                            control_status = "detectable_component_onset_after_label"
                        elif not control_support(L, T_det, fs):
                            control_status = "insufficient_declared_temporal_support"
                        elif not truth_continuous:
                            control_status = "truth_not_label_continuous"
                        else:
                            control_status = "ok"

                    rows.append({
                        **cell, "k": k, "control_status": control_status,
                        "T_det_tick": "" if T_det is None else T_det,
                        "L_tick": "" if L is None else L,
                        "true_lag_ms": "" if L is None or T_det is None else L - T_det,
                        # Retained historical field name: physical onset-to-label lag.
                        "sub_noise_lead_ms": "" if L is None else L - T0_MS,
                        "T_det_minus_T0_ms": "" if T_det is None else T_det - T0_MS,
                        "truth_label_continuous": (
                            "" if truth_continuous is None else int(truth_continuous)
                        ),
                        "truth_primary_R_tick": (
                            "" if truth_primary is None else truth_primary["R_tick"]
                        ),
                        **estimate_control_fields(est),
                    })
    return rows


def nearest_rank(values, q=0.95):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not len(values):
        return None
    return float(values[max(0, int(math.ceil(q * len(values))) - 1)])


def distribution(values, *, include_values=False):
    values = [float(value) for value in values]
    result = {
        "n": len(values),
        "median_ms": float(np.median(values)) if values else None,
        "p95_ms": nearest_rank(values),
    }
    if include_values:
        result["values_ms"] = values
    return result


def control_band(rows, lower, upper, positive, *, inclusive_upper=False):
    selected = [
        row for row in rows
        if row["control_status"] == "ok"
        and float(row["true_lag_ms"]) >= lower
        and (
            float(row["true_lag_ms"]) <= upper if inclusive_upper
            else float(row["true_lag_ms"]) < upper
        )
    ]
    successes = sum(
        row["status"] == "determinate"
        and (
            float(row["recovered_lag_ms"]) >= MARGIN_MS if positive
            else float(row["recovered_lag_ms"]) < MARGIN_MS
        )
        for row in selected
    )
    return {
        "n": len(selected),
        "backgrounds": len({row["background_id"] for row in selected}),
        "successes": int(successes),
        "fraction": successes / len(selected) if selected else None,
    }


def assess_controls(rows, participants, calibration_rows):
    by_k = {}
    for k in KS:
        people = {}
        for participant in participants:
            group = [
                row for row in rows
                if row["participant"] == participant and row["k"] == k
            ]
            shams = [row for row in group if row["kind"] == "sham"]
            alarm_fraction = (
                sum(int(row["sham_alarm"]) for row in shams) / len(shams)
                if shams else None
            )
            person_pass = bool(
                len(shams) >= MIN_BACKGROUNDS
                and alarm_fraction is not None and alarm_fraction <= 0.05
            )
            families = {}
            for family, targets in CONTROL_SPECS.items():
                trials = [row for row in group if row["family"] == family]
                cells = [
                    row for row in calibration_rows
                    if row["participant"] == participant and row["family"] == family
                ]
                if len(cells) != len(targets):
                    raise ValueError("Incomplete calibration table")
                expected_trials = len(shams) * len(targets)
                if len(trials) != expected_trials:
                    raise ValueError("Missing controls, including uncovered cells")

                usable = sum(row["control_status"] == "ok" for row in trials)
                coverage = usable / len(trials) if trials else 0.0
                near = control_band(trials, 288, 320, True)
                long = control_band(trials, 320, 650, True, inclusive_upper=True)
                checks = [
                    coverage >= 0.95,
                    near["backgrounds"] >= MIN_BACKGROUNDS,
                    near["fraction"] is not None and near["fraction"] >= 0.90,
                    long["backgrounds"] >= MIN_BACKGROUNDS,
                    long["fraction"] is not None and long["fraction"] >= 0.95,
                ]
                fast = None
                if family in ("linear", "quadratic"):
                    fast = control_band(trials, 0, 200, False)
                    checks.extend([
                        fast["backgrounds"] >= MIN_BACKGROUNDS,
                        fast["fraction"] is not None and fast["fraction"] >= 0.95,
                    ])
                status_counts = defaultdict(int)
                for row in trials:
                    status_counts[row["control_status"]] += 1
                passed = bool(all(checks))
                families[family] = {
                    "pass": passed,
                    "trials": len(trials),
                    "expected_trials_including_uncovered": expected_trials,
                    "usable_crossing_fraction": coverage,
                    "control_status_counts": dict(status_counts),
                    "calibration_cells": len(cells),
                    "calibrated_cells": sum(
                        row["calibration_status"] == "calibrated" for row in cells
                    ),
                    "uncovered_targets_ms": [
                        row["target_detectable_lag_ms"] for row in cells
                        if row["calibration_status"] != "calibrated"
                    ],
                    "near": near, "long": long, "fast": fast,
                    "descriptive_sub_noise_lead_L_minus_T0": distribution([
                        row["sub_noise_lead_ms"] for row in trials
                        if row.get("sub_noise_lead_ms", "") != ""
                    ]),
                    "descriptive_subthreshold_Tdet_minus_T0": distribution([
                        row["T_det_minus_T0_ms"] for row in trials
                        if row.get("T_det_minus_T0_ms", "") != ""
                    ]),
                }
                person_pass = person_pass and passed

            people[participant] = {
                "pass": bool(person_pass),
                "sham_n": len(shams),
                "sham_alarm_fraction": alarm_fraction,
                "families": families,
            }
        by_k[str(k)] = {
            "pass": bool(people) and all(item["pass"] for item in people.values()),
            "participants": people,
        }

    return {
        "primary_k": 5,
        "primary_pass": by_k["5"]["pass"],
        "result": "PASS" if by_k["5"]["pass"] else "FAIL",
        "truth": "actual noisy-control L minus person/k-specific noiseless T_det",
        "calibration_table": "control_templates.csv",
        "calibration_table_closed_before_noisy_controls": True,
        "calibration_targets": "exact noiseless L_nl minus person-specific T_det at k=5",
        "k3_k8_reuse_k5_templates": True,
        "background_ms": BACKGROUND_MS,
        "valid_only_warmup_ms": WARMUP_MS,
        "screened_still_ms": STILL_MS,
        "control_search_start_ms": CONTROL_SEARCH_START_MS,
        "control_S_ms": CONTROL_S_MS,
        "T0_ms": T0_MS,
        "sham_prelabel_span_ms_by_fs": {"250": 1276, "1000": 1279},
        "by_k": by_k,
        "interpretation": (
            "Engineering criteria, not independent-trial binomial confidence "
            "statements. Profiles reuse backgrounds. Uncovered cells count "
            "against coverage. These shorter searches do not directly validate "
            "the maximum three-second real-event search. No retuning follows."
        ),
    }


def metrics(rows, *, prefix=""):
    if not rows:
        raise ValueError("No accepted events in summary group")
    lower = [float(row[prefix + "lag_lower_ms"]) for row in rows]
    upper = [float(row[prefix + "lag_upper_ms"]) for row in rows]
    determinate = [row for row in rows if row[prefix + "status"] == "determinate"]
    lags = [float(row[prefix + "lag_ms"]) for row in determinate]
    qlower, qupper = nearest_rank(lower), nearest_rank(upper)
    classification = (
        "FAIL_EMPIRICAL_P95" if qlower >= MARGIN_MS else
        "BELOW_EMPIRICAL_P95_ONLY" if qupper < MARGIN_MS else
        "STRADDLES_MARGIN"
    )
    reasons = defaultdict(int)
    for row in rows:
        reasons[row[prefix + "reason"]] += 1
    return {
        "n": len(rows),
        "determinate": len(determinate),
        "indeterminate": sum(row[prefix + "status"] == "indeterminate" for row in rows),
        "no_valid_data": sum(row[prefix + "status"] == "no_valid_data" for row in rows),
        "unresolved_fraction": 1 - len(determinate) / len(rows),
        "multiple_bouts": sum(int(row[prefix + "n_bouts"]) > 1 for row in rows),
        "reason_counts": dict(reasons),
        "p95_determinate_ms": nearest_rank(lags),
        "p95_lower_ms": qlower,
        "p95_upper_ms": qupper if math.isfinite(qupper) else None,
        "p95_upper_unbounded": not math.isfinite(qupper),
        "determinate_total_ge_300_count": sum(lag + FIXED_MS >= 300 for lag in lags),
        "known_total_ge_300_lower_count": sum(lag + FIXED_MS >= 300 for lag in lower),
        "possible_total_ge_300_upper_count": sum(lag + FIXED_MS >= 300 for lag in upper),
        "classification": classification,
    }


def estimator_disagreements(rows):
    both = [
        row for row in rows
        if row["status"] == "determinate" and row["earliest_status"] == "determinate"
    ]
    changed_R = [
        row for row in both if int(row["R_tick"]) != int(row["earliest_R_tick"])
    ]
    status_changed = sum(row["status"] != row["earliest_status"] for row in rows)
    return {
        "n": len(rows),
        "both_determinate": len(both),
        "both_determinate_different_R": len(changed_R),
        "different_status": int(status_changed),
        "different_status_or_determinate_R": int(status_changed + len(changed_R)),
        "primary_R_minus_earliest_R_ms": distribution([
            int(row["R_tick"]) - int(row["earliest_R_tick"]) for row in both
        ]),
        "unresolved_pairs_are_not_assumed_equal_onsets": True,
    }


def make_decision(raw_rows, controls, participants, smoke):
    by_k, per_person = {}, {}
    earliest_by_k, earliest_per_person, differences = {}, {}, {}
    for k in KS:
        group = [row for row in raw_rows if row["k"] == k]
        by_k[str(k)] = metrics(group)
        per_person[str(k)] = {
            person: metrics([row for row in group if row["participant"] == person])
            for person in participants
        }
        earliest_by_k[str(k)] = metrics(group, prefix="earliest_")
        earliest_per_person[str(k)] = {
            person: metrics(
                [row for row in group if row["participant"] == person],
                prefix="earliest_",
            )
            for person in participants
        }
        differences[str(k)] = estimator_disagreements(group)

    reasons = []
    if smoke:
        reasons.append("participant_list_smoke_run")
    if not controls["primary_pass"]:
        reasons.append("all_participant_primary_controls_not_passed")
    if any(item["unresolved_fraction"] > 0.05 for item in by_k.values()):
        reasons.append("over_5_percent_unresolved_primary_events_at_some_k")
    classes = {item["classification"] for item in by_k.values()}
    if "STRADDLES_MARGIN" in classes:
        reasons.append("primary_p95_censoring_interval_straddles_margin")
    if len(classes) != 1:
        reasons.append("primary_threshold_sensitivity_disagreement")
    outcome = "UNDERIDENTIFIED" if reasons else by_k["5"]["classification"]

    long_sets = [
        {
            event_key(row) for row in raw_rows
            if row["k"] == k and row["status"] == "determinate"
            and float(row["lag_ms"]) >= MARGIN_MS
        }
        for k in KS
    ]
    stable_long = sorted(set.intersection(*long_sets))
    main = by_k["5"]
    if main["p95_lower_ms"] + FIXED_MS > 188:
        proxy188 = "EMPIRICAL_LOWER_BOUND_EXCEEDS_188"
    elif not main["p95_upper_unbounded"] and main["p95_upper_ms"] + FIXED_MS <= 188:
        proxy188 = "EMPIRICALLY_COMPATIBLE_CONDITIONAL_ON_IDENTIFICATION"
    else:
        proxy188 = "UNRESOLVED"

    passing = [
        person for person in participants
        if controls["by_k"]["5"]["participants"][person]["pass"]
    ]
    restricted = [
        row for row in raw_rows if row["k"] == 5 and row["participant"] in passing
    ]
    restricted_metrics = metrics(restricted) if restricted else None

    return {
        "outcome": outcome, "reasons": reasons,
        "primary_estimator": "label_continuous_bout_with_200ms_window_start_quiet_anchor",
        "primary_k": 5, "margin_lag_ms": MARGIN_MS, "fixed_terms_ms": FIXED_MS,
        "quantile": "nearest-rank empirical p95",
        "by_k": by_k, "per_person": per_person,
        "earliest_sensitivity": {
            "used_for_decision": False,
            "quiet_definition": "unchanged strict all-sample quiet",
            "by_k": earliest_by_k,
            "per_person": earliest_per_person,
            "primary_vs_earliest_by_k": differences,
        },
        "control_passing_participant_subset": {
            "descriptive_only": True, "used_for_decision": False,
            "participants": passing,
            "excluded_participants": [p for p in participants if p not in passing],
            "primary_k": 5, "metrics": restricted_metrics,
            "classification": (
                restricted_metrics["classification"]
                if restricted_metrics else "NO_CONTROL_PASSING_PARTICIPANTS"
            ),
            "interpretation": (
                "Selected-person description; cannot replace the all-person "
                "control gate, establish generalization, or grant scoring clearance."
            ),
        },
        "188_ms_proxy": proxy188,
        "cross_k_long_lag_event_count": len(stable_long),
        "cross_k_long_lag_event_keys": stable_long,
        "worst_case_bound_certified": False,
        "scoring_clearance": "NOT_GRANTED_BY_THIS_TEST",
        "interpretation": {
            "FAIL_EMPIRICAL_P95": (
                "Empirical p95 fails under the estimator/control assumptions, "
                "weakening the 188-ms proxy and the 300-ms timing argument."
            ),
            "BELOW_EMPIRICAL_P95_ONLY": (
                "Weakens the large-p95-lag alternative only under these "
                "assumptions; no worst-case certification or reinstatement of 188 ms."
            ),
            "UNDERIDENTIFIED": (
                "The declared diagnostic cannot resolve the empirical timing "
                "criterion. Control limitations do not establish real-label failure."
            ),
        },
        "uncertainty": (
            "Censoring bounds, not sampling confidence intervals. Development-only, "
            "post hoc evidence; the three observed smoke participants remain included."
        ),
    }


def stage4_provider_comparison(cue_path, onset_path, summary_path, output_path):
    if not Path(summary_path).is_file():
        raise RuntimeError("Stage 3 must be closed before provider access")
    with Path(summary_path).open(encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot.get("provider_access_has_not_occurred") is not True:
        raise RuntimeError("Missing pre-provider Stage 3 snapshot")

    with Path(onset_path).open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    selected_keys = {event_key(row) for row in saved}
    provider = {}
    columns = ("participant", "session", "run", "cue", "acc_start_tick")
    for row in projected_csv(cue_path, columns, allow_provider=True):
        key = event_key(row)
        if key not in selected_keys:
            continue
        if key in provider:
            raise ValueError("Duplicate provider join key")
        provider[key] = provider_tick(row["acc_start_tick"])

    labels = {event_key(row): required_int(row["L_tick"]) for row in saved}
    label_differences = [
        L - provider[key] for key, L in labels.items() if provider.get(key) is not None
    ]
    by_k = {}
    for k in KS:
        raw_differences, matched_label_differences = [], []
        for row in saved:
            if int(row["k"]) != k or row["status"] != "determinate":
                continue
            acc_start = provider.get(event_key(row))
            if acc_start is None:
                continue
            raw_differences.append(required_int(row["R_tick"]) - acc_start)
            matched_label_differences.append(required_int(row["L_tick"]) - acc_start)
        by_k[str(k)] = {
            "R_minus_AccStartIndex": distribution(raw_differences, include_values=True),
            "L_minus_AccStartIndex_same_rows": distribution(
                matched_label_differences, include_values=True
            ),
        }

    write_json(output_path, {
        "stage": "post_lock_agreement_only",
        "lock_method": "single_coordinator_stage_order",
        "decision_summary_written_before_provider_access": True,
        "decision_summary": Path(summary_path).name,
        "onset_table": Path(onset_path).name,
        "provider_field_used": "acc_start_tick",
        "provider_tick_rounding_tolerance": PROVIDER_TICK_TOLERANCE,
        "provider_flag_filtering": False,
        "accepted_events": len(labels),
        "provider_missing_events": sum(provider.get(key) is None for key in labels),
        "L_minus_AccStartIndex_all_available": distribution(
            label_differences, include_values=True
        ),
        "L_minus_AccStartIndex_ge_288_fraction": (
            float(np.mean(np.asarray(label_differences) >= MARGIN_MS))
            if label_differences else None
        ),
        "by_k": by_k,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--participants", help="Comma-separated development IDs; smoke only")
    args = parser.parse_args()

    host, cpu_limit = require_slurm(args.workers)
    requested = (
        [p.strip() for p in args.participants.split(",") if p.strip()]
        if args.participants is not None else list(R.SP_DEVELOPMENT)
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Participant list is empty or duplicated")
    if set(requested) - set(R.SP_DEVELOPMENT):
        raise ValueError("Only supplied development participants may be processed")
    participants = [p for p in R.SP_DEVELOPMENT if p in set(requested)]
    smoke = args.participants is not None

    freeze_dir, output_dir = args.freeze_dir.resolve(), args.output_dir.resolve()
    if output_dir == freeze_dir or freeze_dir in output_dir.parents:
        raise ValueError("Output must not be inside the frozen run directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a fresh or empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Written before scientific inputs or diagnostic output.
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M")
    with (output_dir / "predeclaration.log").open("x", encoding="utf-8") as handle:
        handle.write(LOG_TEMPLATE.format(stamp=stamp))
    with (output_dir / "predeclaration.txt").open("x", encoding="utf-8") as handle:
        handle.write(DECLARATION)
    background_dir = output_dir / "control_backgrounds"
    background_dir.mkdir()

    if (
        F.SP_CONFIRM != 10 or F.SP_NOISE_SAMPLES != 250
        or F.SP_BLOCK_SAMPLES != 25 or S.SP_SUMMARY_SAMPLES != 5
        or S.SP_BASE_SAMPLES != 125 or R.SP_DETECTOR_HZ != 250
        or F.SP_STILL_BLOCK_G != 0.02 or F.SP_FLAT_SD_G != 1e-4
    ):
        raise ValueError("Imported constants differ from supplied contract")
    if (
        WARMUP_MS + STILL_MS != BACKGROUND_MS
        or CONTROL_S_MS - 500 != CONTROL_SEARCH_START_MS
        or CONTROL_SEARCH_START_MS < WARMUP_MS + 520
        or T0_MS - PRIMARY_QUIET_MS < CONTROL_SEARCH_START_MS
        or T0_MS < CONTROL_S_MS
        or SHAM_L_MS + LOOKAHEAD_MS != BACKGROUND_MS - 4
        or max(max(targets) for targets in CONTROL_SPECS.values()) > 600
    ):
        raise RuntimeError("Inconsistent locked control timeline")

    with (freeze_dir / "detector_settings.json").open(encoding="utf-8") as handle:
        settings = json.load(handle)["self_paced"]
    low, high = float(settings["low_g"]), float(settings["high_g"])
    if not (
        math.isclose(low, EXPECTED_LOW, rel_tol=0, abs_tol=1e-14)
        and math.isclose(high, EXPECTED_HIGH, rel_tol=0, abs_tol=1e-14)
    ):
        raise ValueError("Thresholds differ from supplied freeze")

    cue_path = freeze_dir / "self_paced_cues.csv"
    projected = list(projected_csv(cue_path, CUE_COLUMNS))
    cues, seen = [], set()
    accepted_all = 0
    for row in projected:
        if row["participant"] not in R.SP_DEVELOPMENT:
            raise ValueError("Cue table contains non-development participant")
        row["run"] = required_int(row["run"])
        key = event_key(row)
        if key in seen:
            raise ValueError("Duplicate cue key")
        seen.add(key)
        L = optional_int(row["onset_tick"])
        if L is None:
            continue
        accepted_all += 1
        if row["participant"] in participants:
            cues.append({
                "participant": row["participant"], "session": row["session"],
                "run": row["run"], "cue": row["cue"], "L_tick": L,
                "S_tick": required_int(row["start_known_tick"]),
            })
    if len(projected) != EXPECTED_FULL["cues"] or accepted_all != EXPECTED_FULL["accepted"]:
        raise ValueError("Frozen cue/accepted counts do not match supplied freeze")
    if not cues:
        raise ValueError("No accepted onsets for selected participants")

    all_stems = sorted(R.development_stems())
    if len(all_stems) != EXPECTED_FULL["runs"]:
        raise ValueError("Development run count differs from 155")
    stems_by_person, stem_keys = defaultdict(list), set()
    for stem in all_stems:
        key = stem_key(stem)
        if key[0] not in R.SP_DEVELOPMENT or key in stem_keys:
            raise ValueError("Unexpected/duplicate development run")
        stem_keys.add(key)
        if key[0] in participants:
            stems_by_person[key[0]].append(stem)
    for cue in cues:
        if (cue["participant"], cue["session"], cue["run"]) not in stem_keys:
            raise ValueError("Accepted cue has no development raw run")
    if any(not any(c["participant"] == p for c in cues) for p in participants):
        raise ValueError("Selected participant has no accepted onsets")

    write_json(output_dir / "run_manifest.json", {
        "hostname": host, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "workers": args.workers, "cpu_limit": cpu_limit,
        "participants": participants,
        "scope": "smoke" if smoke else "full_development",
        "freeze_dir": str(freeze_dir),
        "seed": SEED, "frozen_low_g": low, "frozen_high_g": high,
        "revision_is_post_hoc": True,
        "observed_smoke_job": "48227088",
        "observed_smoke_participants": SEEN_SMOKE_PARTICIPANTS,
        "observed_smoke_participants_remain_in_full_run": True,
        "new_choices_are_assumptions": True,
        "provider_separation": "single_coordinator_stage_order",
        "control_timeline_ms": {
            "background": BACKGROUND_MS, "valid_only_warmup": WARMUP_MS,
            "screened_still": STILL_MS, "search_start": CONTROL_SEARCH_START_MS,
            "S": CONTROL_S_MS, "T0": T0_MS,
            "sham_L_250": SHAM_L_MS, "sham_L_1000": SHAM_L_MS + 3,
            "lookahead": LOOKAHEAD_MS,
        },
    })

    context = mp.get_context("spawn")
    references, raw_rows, caches = [], [], {}
    tasks = [
        (
            p, stems_by_person[p],
            [cue for cue in cues if cue["participant"] == p], str(background_dir),
        )
        for p in participants
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(stage1_person, task): task[0] for task in tasks}
        for future in as_completed(futures):
            reference, rows, cache = future.result()
            references.append(reference)
            raw_rows.extend(rows)
            caches[reference["participant"]] = cache
            print(
                f"Stage 1: {reference['participant']}; "
                f"eligible backgrounds={reference['control_eligible']}; "
                f"selected={reference['control_selected']}",
                flush=True,
            )

    references.sort(key=lambda row: row["participant"])
    raw_rows.sort(key=lambda row: (*event_key(row), row["k"]))
    if len(raw_rows) != len(cues) * len(KS):
        raise ValueError("Missing accepted onset/threshold pairs")
    if not smoke:
        for field in (
            "runs", "start_tones", "still_windows", "fit_windows", "validation_windows",
        ):
            actual = sum(int(row[field]) for row in references)
            if actual != EXPECTED_FULL[field]:
                raise ValueError(
                    f"Frozen reproduction mismatch for {field}: "
                    f"{actual} versus {EXPECTED_FULL[field]}"
                )

    onset_path, noise_path = output_dir / "raw_onsets.csv", output_dir / "noise_reference.csv"
    write_csv(onset_path, RAW_FIELDS, raw_rows)
    write_csv(noise_path, NOISE_FIELDS, references)

    # Stage 2a receives neither real-event estimates nor real-background arrays.
    calibration_rows = []
    tasks = [
        (row["participant"], float(row["fs"]), reference_theta(row, 5), low, high)
        for row in references
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(stage2_calibrate_person, task): task[0] for task in tasks}
        for future in as_completed(futures):
            calibration_rows.extend(future.result())
            print(f"Stage 2 calibration: {futures[future]}", flush=True)

    calibration_rows.sort(key=lambda row: (
        row["participant"], row["family"], row["target_detectable_lag_ms"]
    ))
    if len(calibration_rows) != len(participants) * sum(map(len, CONTROL_SPECS.values())):
        raise RuntimeError("Missing calibration rows")
    write_csv(output_dir / "control_templates.csv", TEMPLATE_FIELDS, calibration_rows)
    print("Calibration table closed before noisy controls", flush=True)

    control_rows = []
    tasks = [
        (
            row["participant"], caches[row["participant"]], row, low, high,
            [cell for cell in calibration_rows if cell["participant"] == row["participant"]],
        )
        for row in references
    ]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(stage2_person, task): task[0] for task in tasks}
        for future in as_completed(futures):
            control_rows.extend(future.result())
            print(f"Stage 2 noisy controls: {futures[future]}", flush=True)

    control_rows.sort(key=lambda row: (
        row["participant"], row["background_id"], row["kind"], row["family"],
        row.get("target_detectable_lag_ms", -1), row["k"],
    ))
    write_csv(output_dir / "positive_control.csv", CONTROL_FIELDS, control_rows)
    controls = assess_controls(control_rows, participants, calibration_rows)
    write_json(output_dir / "positive_control_summary.json", controls)

    # Complete sham-report columns only; never refit noise references.
    for reference in references:
        for k in KS:
            fraction = controls["by_k"][str(k)]["participants"][
                reference["participant"]
            ]["sham_alarm_fraction"]
            reference[f"sham_alarm_fraction_k{k}"] = "" if fraction is None else fraction
    write_csv(noise_path, NOISE_FIELDS, references, replace=True)

    decision = make_decision(raw_rows, controls, participants, smoke)
    summary = {
        "scope": "smoke" if smoke else "full_development",
        "participants": participants, "accepted_events": len(cues),
        "revision_is_post_hoc": True, "observed_smoke_job": "48227088",
        "observed_smoke_participants": SEEN_SMOKE_PARTICIPANTS,
        "observed_smoke_participants_remain_in_full_run": True,
        "noise_counts": {
            key: sum(int(row[key]) for row in references)
            for key in (
                "runs", "start_tones", "still_windows", "fit_windows", "validation_windows",
            )
        },
        "background_counts_by_person": {
            row["participant"]: {
                key: int(row[key]) for key in (
                    "control_intervals", "control_intervals_with_room",
                    "control_eligible", "control_selected",
                )
            }
            for row in references
        },
        "control_template_counts": {
            "cells": len(calibration_rows),
            "calibrated": sum(
                row["calibration_status"] == "calibrated" for row in calibration_rows
            ),
            "uncovered": sum(
                row["calibration_status"] != "calibrated" for row in calibration_rows
            ),
        },
        "positive_control_primary_pass": controls["primary_pass"],
        "decision": decision,
        "lock_method": "single_coordinator_stage_order",
        "stage_order": [
            "1: estimates and noise references written",
            "2a: noiseless calibration table written and closed",
            "2b: noisy controls written; noise CSV sham columns completed",
            "3: this decision summary written and closed",
            "4: provider values accessed afterward for separate agreement output",
        ],
        "provider_access_has_not_occurred": True,
        "stage3_snapshot_not_rewritten_after_provider_access": True,
    }
    summary_path = output_dir / "raw_onset_summary.json"
    write_json(summary_path, summary)
    print(f"Stage 3 locked: {decision['outcome']}", flush=True)

    stage4_provider_comparison(
        cue_path, onset_path, summary_path, output_dir / "post_lock_comparison.json"
    )
    print(
        f"Stage 4 complete; result={decision['outcome']}; "
        f"controls={controls['result']}; output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
