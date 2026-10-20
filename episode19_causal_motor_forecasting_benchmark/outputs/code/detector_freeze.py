"""EP19 detector freeze: thresholds from development stationary noise, cues, risk state.

Rules: outputs/component_details.md section 1.2 (written before any detector
output, October 16, 2026). This module holds the threshold, cue and risk-state
logic; ``run_detector_freeze.py`` drives development data through it.
"""
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- frozen rules
QUANTILE = 99.9                  # scale: sigma_q = q99.9 - median of fit still-noise ticks
RUNGS = (1.0, 1.5, 2.0, 3.0, 4.0)  # low = m + c*sigma_q, high = m + 2c*sigma_q
FIT_WINDOW_CROSSING_FRACTION = 0.01  # allowed fit windows with a confirmed crossing: max(1, 1%)
WAY_STILL_SPREAD_CM = 0.5
WAY_CLEARANCE_CM = 15.0
WAY_NOISE_TICKS = 500            # [rise - 1 s, rise)
WAY_CONFIRM = 20                 # 40 ms at 500 Hz
WAY_QUIET = 250                  # 500 ms
WAY_GRID = 25                    # 50 ms
REF_TOLERANCE_MULTIPLIER = 3.0   # return references: 3 x q99 |deviation| over fit still ticks
REF_FLOORS = {"position_cm": 0.1, "motion_cm_s": 0.5, "force_n": 0.02}

SP_NOISE_SAMPLES = 250           # 1 s at 250 Hz before a start-tone onset
SP_BLOCK_SAMPLES = 25            # 100-ms blocks for the stillness screen
SP_STILL_BLOCK_G = 0.02
SP_CONFIRM = 10                  # 40 ms at 250 Hz
SP_QUIET = 125                   # 500 ms at 250 Hz
SP_GRID_TICKS = 50               # 50 ms on the 1-kHz tick clock
SP_FLAT_SD_G = 1e-4              # trailing 1-s axis SD below this = flat axis = missing
TONE_ENVELOPE_S = 0.100
TONE_REFERENCE_BLOCKS = 10
TONE_ON_RATIO = 4.0
TONE_MERGE_S = 0.100
TONE_CONFIRM_S = 0.250
SCHEDULE_S = {"start_to_end": 12.0, "end_to_start": 7.08, "same": 19.08}
SCHEDULE_TOL_S = {"start_to_end": 0.4, "end_to_start": 0.4, "same": 0.5}

STATE_CODES = {"unknown": 0, "unarmed": 1, "at_risk": 2, "pending": 3}


# ------------------------------------------------------------- noise and fits
def confirmed_crossing(x, low, high, confirm):
    """True if a 1-D noise trace crosses >= high at c and stays >= low at c+1..c+confirm.

    Same confirmation as the risk stream (accepted at c + confirm). A crossing still
    unresolved at the window end counts as confirmed (conservative).
    """
    i, n = 0, len(x)
    while i < n:
        if x[i] >= high:
            j = i + 1
            while j < n and j <= i + confirm and x[j] >= low:
                j += 1
            if j > i + confirm or j >= n:
                return True
            i = j
        else:
            i += 1
    return False


def quiet_entry(window, low, quiet):
    """All paths <= low over the final `quiet` samples of the window."""
    tail = window[-quiet:]
    return len(tail) == quiet and bool(np.all(tail <= low))


@dataclass
class PathFit:
    median: float
    sigma_q: float
    rung: float
    low: float
    high: float
    fit_windows_with_crossing: int
    allowed: int
    resolved: bool
    at_top_rung_without_meeting_criterion: bool


def fit_thresholds(fit_windows, confirm):
    """Per-path quantile-scaled hysteresis thresholds with the rung criterion (fit windows only)."""
    fit = np.concatenate(fit_windows)
    out = []
    allowed = max(1, int(np.floor(FIT_WINDOW_CROSSING_FRACTION * len(fit_windows))))
    for p in range(fit.shape[1]):
        m = float(np.median(fit[:, p]))
        sq = float(np.percentile(fit[:, p], QUANTILE) - m)
        if not np.isfinite(sq) or sq <= 0:
            out.append(PathFit(m, sq, np.nan, np.nan, np.nan, -1, allowed, False, False))
            continue
        chosen = None
        for c in RUNGS:
            low, high = m + c * sq, m + 2 * c * sq
            count = sum(confirmed_crossing(w[:, p], low, high, confirm) for w in fit_windows)
            chosen = (c, low, high, count)
            if count <= allowed:
                break
        c, low, high, count = chosen
        out.append(PathFit(m, sq, c, low, high, int(count), allowed, True, count > allowed))
    return out


def validate_thresholds(windows, low, high, confirm, quiet, rate_hz):
    """Held-out-parity noise check: any-path confirmed crossings and quiet-entry availability."""
    if not windows:
        return {"windows": 0}
    any_cross = 0
    per_path = np.zeros(len(low), dtype=int)
    entry = 0
    for w in windows:
        hits = [confirmed_crossing(w[:, p], low[p], high[p], confirm) for p in range(len(low))]
        per_path += np.array(hits, dtype=int)
        any_cross += any(hits)
        entry += quiet_entry(w, low, quiet)
    minutes = sum(len(w) for w in windows) / rate_hz / 60.0
    return {"windows": len(windows), "noise_minutes": minutes,
            "windows_with_any_path_crossing": int(any_cross),
            "any_path_crossing_fraction": any_cross / len(windows),
            "any_path_crossings_per_minute": any_cross / minutes,
            "per_path_windows_with_crossing": per_path.tolist(),
            "quiet_entry_available_fraction": entry / len(windows)}


# ------------------------------------------------------------ TRIG tone cues
def detect_tones(trig_uv, fs):
    """Causal tone bursts on |dTRIG| with tone-duration confirmation and schedule classes.

    Returns bursts as dicts with native-sample onset/end, confirmation sample (tones) or
    rejection sample (short bursts), and class 'start'/'end'/'unclassified' (tones) or
    'rejected'. Each class is decided from information available by its confirmation.
    """
    if not np.all(np.isfinite(trig_uv)):
        raise ValueError("nonfinite TRIG requires an explicit cue-evidence gap")
    d = np.abs(np.diff(trig_uv, prepend=trig_uv[0]))
    w = int(round(TONE_ENVELOPE_S * fs))
    c = np.concatenate([[0.0], np.cumsum(d)])
    env = np.full(len(d), np.nan)
    env[w - 1:] = (c[w:] - c[:-w]) / w
    block = int(fs)
    nblocks = len(d) // block
    medians = np.median(d[:nblocks * block].reshape(nblocks, block), axis=1) if nblocks else np.zeros(0)
    ref = np.full(len(d), np.nan)
    for j in range(2, nblocks + 1):
        ref[j * block:(j + 1) * block] = np.median(medians[max(0, j - TONE_REFERENCE_BLOCKS):j])
    with np.errstate(invalid="ignore"):
        hot = env > TONE_ON_RATIO * ref
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hot.astype(np.int8), [0]])))
    merge = int(round(TONE_MERGE_S * fs))
    need = int(np.ceil(TONE_CONFIRM_S * fs - 1e-9))   # first whole sample count reaching 0.25 s
    bursts = []
    for a, z in zip(edges[::2], edges[1::2]):
        if bursts and a - bursts[-1]["end"] < merge:
            bursts[-1]["end"] = int(z)
        else:
            bursts.append({"onset": int(a), "end": int(z)})
        merged = bursts[-1]
        if "confirmed_at" not in merged and merged["end"] - merged["onset"] >= need:
            # The span is established only once the hot run that completes it has been observed.
            merged["confirmed_at"] = max(merged["onset"] + need, int(a))
    for b in bursts:
        b["duration_s"] = (b["end"] - b["onset"]) / fs
        if b["end"] - b["onset"] >= need:
            b["tone"] = True
            b["known"] = b.pop("confirmed_at")
        else:
            b["tone"] = False
            b["known"] = min(b["end"] + merge, len(d))
            b["class"] = "rejected"
    tones = [b for b in bursts if b["tone"]]

    def scheduled(gap_samples, kind):
        return abs(gap_samples - int(round(SCHEDULE_S[kind] * fs))) <= int(round(SCHEDULE_TOL_S[kind] * fs))

    last_classified, previous = None, None
    for b in tones:
        t = b["onset"]
        cls = "unclassified"
        if last_classified is not None:
            gap = t - last_classified["onset"]
            if last_classified["class"] == "start" and scheduled(gap, "start_to_end"):
                cls = "end"
            elif last_classified["class"] == "end" and scheduled(gap, "end_to_start"):
                cls = "start"
            elif scheduled(gap, "same"):
                cls = last_classified["class"]
        if cls == "unclassified" and previous is not None and previous is not last_classified:
            # Interval to the previous (unclassified) tone: 12.0 s only follows a start,
            # 7.08 s only follows an end.
            gap = t - previous["onset"]
            if scheduled(gap, "start_to_end"):
                cls = "end"
            elif scheduled(gap, "end_to_start"):
                cls = "start"
        b["class"] = cls
        if cls != "unclassified":
            last_classified = b
        previous = b
    return bursts, {"reference_median_uV": float(np.nanmedian(ref)) if np.isfinite(ref).any() else None}


# ----------------------------------------------------------- generic stream
@dataclass
class CueResult:
    start: int                   # sample index where the cue opened
    stop: int                    # exclusive cut sample (end, gap or segment end)
    stop_reason: str
    status: str                  # supported / not_supported / unknown
    onset: tuple | None          # (crossing_sample, known_sample)
    onset_path: int | None
    accepted: list
    rejected: list
    pending_at_stop: dict
    unknown_from: int | None     # earliest unresolved possible onset as of the cut (sample)


def run_segment(values, low, high, confirm, quiet, starts, cuts, evidence_at=None):
    """Replay one uninterrupted segment with the WayRiskStream state rules, generically.

    values (m, P) finite; ``starts`` maps sample -> return-support status; ``cuts`` is a
    set of samples where an open cue is cut BEFORE that sample is adjudicated;
    ``evidence_at`` optionally maps a cut sample to an earlier sample whose
    pre-adjudication state is the cue's evidence (self-paced tone onset).
    Returns per-sample state codes (after each sample), per-sample cue index and cues.
    """
    m, npath = values.shape
    state = np.full(m, STATE_CODES["unarmed"], dtype=np.int8)
    cue_of = np.full(m, -1, dtype=np.int32)
    cues = []
    cue = None
    snapshots = {}
    evidence_at = evidence_at or {}
    wanted_snapshots = set(evidence_at.values())

    def close(i, reason):
        nonlocal cue
        if cue is None:
            return
        snap = snapshots.pop(evidence_at[i], None) if i in evidence_at else None
        if snap is None:
            snap = (dict(cue["pending"]), list(cue["accepted"]), cue["onset"], cue["surveillance"],
                    list(cue["rejected"]))
        pending, accepted, onset, surveillance, rejected = snap
        if not surveillance:
            unknown = cue["start"]
        elif onset is None and pending:
            unknown = min(pending.values())
            if accepted:
                unknown = min(unknown, min(c for c, _, _ in accepted))
        else:
            unknown = None
        path = None
        if onset is not None:
            path = min(accepted)[1]
        cues.append(CueResult(cue["start"], i, reason, cue["status"], onset, path, accepted,
                              rejected, pending, unknown))
        cue = None

    for i in range(m):
        if i in cuts:
            close(i, "cut")
        if i in starts:
            close(i, "restart")
            status = starts[i]
            cue = {"start": i, "status": status, "surveillance": status == "supported",
                   "armed": status == "supported", "requalifying": False, "quiet": 0,
                   "pending": {}, "accepted": [], "onset": None, "rejected": []}
        if cue is None:
            continue
        if i in wanted_snapshots:
            snapshots[i] = (dict(cue["pending"]), list(cue["accepted"]), cue["onset"], cue["surveillance"],
                            list(cue["rejected"]))
        cue_of[i] = len(cues)
        if not cue["surveillance"]:
            state[i] = STATE_CODES["unknown"] if cue["status"] == "unknown" else STATE_CODES["unarmed"]
            continue
        v = values[i]
        if cue["requalifying"] and not cue["pending"] and not cue["accepted"] and cue["quiet"] >= quiet:
            cue["armed"], cue["requalifying"] = True, False
        if not cue["accepted"]:
            for p in range(npath):
                if p not in cue["pending"] and v[p] >= high[p]:
                    cue["pending"][p] = i
                    cue["armed"] = False
                    cue["requalifying"] = True
                    cue["quiet"] = 0
        had_pending = bool(cue["pending"])
        for p, c in list(cue["pending"].items()):
            if v[p] < low[p]:
                cue["rejected"].append((c, p, i))
                del cue["pending"][p]
            elif i - c >= confirm:
                cue["accepted"].append((c, p, i))
                del cue["pending"][p]
        if cue["accepted"]:
            c = min(a[0] for a in cue["accepted"])
            if cue["onset"] is None and not any(o <= c for o in cue["pending"].values()):
                cue["onset"] = (c, i)
            cue["armed"] = False
            cue["quiet"] = 0
        elif cue["pending"]:
            cue["quiet"] = 0
        elif cue["requalifying"]:
            if had_pending:
                cue["quiet"] = 0
            q = bool(np.all(v <= low))
            cue["quiet"] = cue["quiet"] + 1 if q else 0
        if cue["accepted"]:
            state[i] = STATE_CODES["unarmed"] if cue["onset"] is not None else STATE_CODES["pending"]
        else:
            state[i] = STATE_CODES["at_risk"] if cue["armed"] else STATE_CODES["pending"]
    close(m, "segment_end")
    return state, cue_of, cues


def retrospective_departure(values, low, crossing, floor_sample=0):
    """Earliest start of a continuous supra-low run (any path) that reaches the crossing sample.

    Diagnostic only (looks backward from an accepted onset). Returns (sample, path).
    """
    best, best_path = crossing, None
    for p in range(values.shape[1]):
        if not values[crossing, p] > low[p]:
            continue
        j = crossing
        while j - 1 >= floor_sample and values[j - 1, p] > low[p]:
            j -= 1
        if j < best or best_path is None:
            best, best_path = j, p
    return best, best_path
