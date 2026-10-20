"""EP19 development readers that apply the October 16 resolved source facts.

WAY: P1-P12 series 1-7 HS files from the job-46425841 development view. Cues are
HS ``ParticipantLED > 0.5`` (lit); per-lift weight (g) and surface code (1 silk,
2 suede, 3 sandpaper) come from HS ``env`` from the LED rise; positions are cm,
forces N. Position-derived model inputs lag decision time by 40 ms. Provider
``HandStart`` is read only from ``AllLifts`` rows of series 1-7 (agreement only).

Self-paced: the 15 development participants. microvolts = stored value x header
resolution; 1 g = 1.4465e6 microvolts; vector norms, not signed axes; a flat axis
is missing. Provider ``AccStartIndex`` is read only for agreement.

Closed and never opened here: WAY series 8-9 and their AllLifts rows, the eight
held-out self-paced participants, and EEG (HS loads decode it incidentally; it is
discarded unread). Source facts:
outputs/source_timing_handoff.md#october-16-source-fact-classification-current-supersedes-the-open-items-below
"""
import io
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

WAY_RATE_HZ = 500
EMG_RATE_HZ = 4000
EMG_PER_TICK = EMG_RATE_HZ // WAY_RATE_HZ
WAY_PARTICIPANTS = tuple(range(1, 13))
WAY_SERIES = tuple(range(1, 8))
POSITION_INPUT_LAG_TICKS = 20          # 40 ms: position-derived model inputs only
LED_LIT_LEVEL = 0.5
ENV_READ_OFFSET_TICKS = 5              # env switches to the upcoming lift within +-10 ms of the rise
WEIGHT_GRAMS = (165.0, 330.0, 660.0)
SURFACE_NAMES = {1: "silk", 2: "suede", 3: "sandpaper"}

SP_DEVELOPMENT = ("sub-02", "sub-03", "sub-04", "sub-06", "sub-08", "sub-09", "sub-11", "sub-12",
                  "sub-14", "sub-15", "sub-17", "sub-18", "sub-20", "sub-21", "sub-22")
SP_HELD_OUT = ("sub-01", "sub-05", "sub-07", "sub-10", "sub-13", "sub-16", "sub-19", "sub-23")
G_MICROVOLT = 1.4465e6
SP_TICK_HZ = 1000                      # integer 1-kHz tick clock for 250-Hz and 1-kHz runs
SP_DETECTOR_HZ = 250                   # 1-kHz accelerometer is causally reduced to 250 Hz

_SCRATCH_VIEW = os.path.join(os.environ.get("SCRATCH", ""), "br_autoresearch",
                             "episode19_causal_motor_forecasting_benchmark", "development_view_job_46425841")
WAY_VIEW = os.path.join(_SCRATCH_VIEW, "way")
SP_VIEW = os.path.join(_SCRATCH_VIEW, "self_paced", "development")
_ARCHIVES = os.path.join(os.environ.get("OAK", ""), "users/zijiao/br_autoresearch_data/steward_acquisition/"
                         "ep19_public_sources/archives")
WAY_ARCHIVES = os.path.join(_ARCHIVES, "way")
SP_ARCHIVE = os.path.join(_ARCHIVES, "freewill", "Freewill_EEG_Reaching_Grasping.zip")


def _names(values):
    return [str(v).strip() for v in np.atleast_1d(values)]


def _column(sub, name):
    return np.asarray(sub.sig, dtype=np.float64)[:, _names(sub.names).index(name)]


def rising_falling(lit):
    lit = np.asarray(lit, dtype=bool)
    rises = np.flatnonzero(lit[1:] & ~lit[:-1]) + 1
    falls = np.flatnonzero(~lit[1:] & lit[:-1]) + 1
    return rises, falls


@dataclass
class WaySeries:
    participant: int
    series: int
    role: str
    n: int
    emg: np.ndarray            # (>= 8n, 5) native 4-kHz samples; EMG sample 8i belongs to tick i
    emg_names: list
    positions_cm: np.ndarray   # (n, 4, 3): P1 object, P2 index, P3 thumb, P4 wrist; XYZ, +z up
    forces_n: np.ndarray       # (n, 2, 3): plates 1-2, FX FY FZ, raw (rest offset ~ -0.08 N)
    led: np.ndarray
    lit: np.ndarray
    rises: np.ndarray
    falls: np.ndarray
    weight_env: np.ndarray
    surface_env: np.ndarray


def load_way_series(participant, series):
    """Load one development HS series; series 8-9 are refused."""
    if participant not in WAY_PARTICIPANTS or series not in WAY_SERIES:
        raise ValueError("only WAY P1-P12 series 1-7 are development data")
    role = "fit_and_inner_selection" if series <= 6 else "aggregate_development_feedback"
    path = os.path.join(WAY_VIEW, role, f"HS_P{participant}_S{series}.mat")
    hs = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["hs"]
    if float(hs.kin.samplingrate) != WAY_RATE_HZ or float(hs.emg.samplingrate) != EMG_RATE_HZ:
        raise ValueError("unexpected WAY sampling rates")
    kin = hs.kin
    n = int(np.asarray(kin.sig).shape[0])
    positions = np.stack([np.stack([_column(kin, f"P{axis}{sensor} - position {axis} sensor {sensor}")
                                    for axis in "xyz"], axis=1) for sensor in range(1, 5)], axis=1)
    forces = np.stack([np.stack([_column(kin, f"F{axis}{plate} - force {axis.lower()} plate {plate}")
                                 for axis in "XYZ"], axis=1) for plate in (1, 2)], axis=1)
    emg = np.asarray(hs.emg.sig, dtype=np.float64)
    if emg.shape[1] != 5 or (n - 1) * EMG_PER_TICK >= len(emg):
        raise ValueError("all five EMG channels covering every tick are required")
    led = _column(hs.misc, "ParticipantLED")
    lit = led > LED_LIT_LEVEL
    rises, falls = rising_falling(lit)
    out = WaySeries(participant, series, role, n, emg, _names(hs.emg.names), positions, forces, led, lit,
                    rises, falls, _column(hs.env, "Weight"), _column(hs.env, "Surface"))
    del hs  # EEG decoded by loadmat is discarded unread
    return out


def lift_conditions(way):
    """Per LED rise: weight (g) and surface code read at rise + 10 ms, when env has switched."""
    rows = []
    for rise in way.rises:
        tick = int(rise) + ENV_READ_OFFSET_TICKS
        if tick >= way.n:
            rows.append({"rise_tick": int(rise), "available_tick": None, "weight_g": None,
                         "surface_code": None, "resolved": False})
            continue
        weight = float(way.weight_env[tick])
        surface = float(way.surface_env[tick])
        grams = min(WEIGHT_GRAMS, key=lambda g: abs(g - weight))
        code = int(round(surface))
        resolved = abs(grams - weight) <= 5.0 and code in SURFACE_NAMES and abs(surface - code) <= 0.1
        rows.append({"rise_tick": int(rise), "available_tick": tick, "weight_g": grams if resolved else None,
                     "surface_code": code if resolved else None, "resolved": bool(resolved)})
    return rows


def lagged(values, ticks=POSITION_INPUT_LAG_TICKS):
    """Value at decision t is the sample at t - ticks; earlier decisions are unavailable (NaN)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full_like(values, np.nan)
    out[ticks:] = values[:-ticks]
    return out


def past_no_contact_force_baseline(way, clearance_cm=15.0, window_ticks=250):
    """Causal plate-force baseline: mean over the latest pre-rise interval with both digits >= 15 cm.

    The baseline formed at rise r uses [r-250, r) and applies from r until the next
    qualifying rise; before the first one it is unknown (NaN). Net force = raw - baseline.
    """
    baseline = np.full(way.forces_n.shape, np.nan)
    p = way.positions_cm
    clear = np.minimum(np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                       np.linalg.norm(p[:, 2] - p[:, 0], axis=1)) >= clearance_cm
    applied = []
    for rise in way.rises:
        begin = int(rise) - window_ticks
        if begin < 0 or not np.all(clear[begin:rise]):
            continue
        baseline[rise:] = way.forces_n[begin:rise].mean(axis=0)
        applied.append(int(rise))
    return way.forces_n - baseline, baseline, applied


def way_handstart_rows(participant):
    """Provider HandStart (agreement only) for series 1-7 rows; other rows are dropped unread.

    HS time (s) = StartTime - 2 + tHandStart - 0.002, StartTime being the stored
    trial_start_time (HS time of LED onset).
    """
    with zipfile.ZipFile(os.path.join(WAY_ARCHIVES, f"P{participant}.zip")) as archive:
        raw = archive.read(f"P{participant}_AllLifts.mat")
    content = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    del raw
    record = next(v for k, v in content.items()
                  if not k.startswith("__") and hasattr(v, "AllLifts") and hasattr(v, "ColNames"))
    names = _names(record.ColNames)
    table = np.asarray(record.AllLifts, dtype=np.float64)
    keep = table[:, names.index("Run")] <= 7
    table = table[keep]   # series 8-9 rows are dropped before any other column is read
    del record, content
    columns = {name: table[:, names.index(name)] for name in ("Run", "Lift", "StartTime", "tHandStart")}
    hs_time = columns["StartTime"] - 2.0 + columns["tHandStart"] - 0.002
    return [{"series": int(s), "lift": int(l), "start_time": float(st), "t_handstart": float(th),
             "handstart_hs_s": float(h)}
            for s, l, st, th, h in zip(columns["Run"], columns["Lift"], columns["StartTime"],
                                       columns["tHandStart"], hs_time)]


# ----------------------------------------------------------------- self-paced

def read_vhdr(path):
    section, info, channels = None, {}, []
    with open(path, encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            if line.startswith("["):
                section = line.strip()
                continue
            if "=" not in line or line.startswith(";"):
                continue
            key, value = line.split("=", 1)
            if section in ("[Common Infos]", "[Binary Infos]"):
                info[key.strip()] = value.strip()
            elif section == "[Channel Infos]" and key.startswith("Ch"):
                parts = value.split(",")
                channels.append((parts[0], float(parts[2]) if len(parts) > 2 and parts[2] else 1.0))
    return info, channels


@dataclass
class SelfPacedRun:
    participant: str
    session: str
    run: int
    stem: str
    fs: float
    frames: int
    ticks_per_sample: int      # on the 1-kHz tick clock
    trig_uv: np.ndarray
    acc_g: np.ndarray          # (frames, 3) X Y Z in g


def development_stems():
    stems = []
    for participant in SP_DEVELOPMENT:
        for session in sorted(os.listdir(os.path.join(SP_VIEW, participant))):
            folder = os.path.join(SP_VIEW, participant, session, "eeg")
            for name in sorted(os.listdir(folder)):
                if name.endswith("_eeg.vhdr"):
                    stems.append(os.path.join(folder, name[:-5]))
    return stems


def load_self_paced_run(stem):
    base = os.path.basename(stem)
    match = re.match(r"(sub-\d+)_(ses-\d+)_task-reachingandgrasping_run-(\d+)_eeg$", base)
    if match is None or match.group(1) not in SP_DEVELOPMENT:
        raise ValueError("only the 15 development participants may be read")
    info, channels = read_vhdr(stem + ".vhdr")
    if info.get("BinaryFormat", "IEEE_FLOAT_32") != "IEEE_FLOAT_32" or \
            info.get("DataOrientation", "MULTIPLEXED") != "MULTIPLEXED":
        raise ValueError("unexpected BrainVision binary layout")
    names = [c[0] for c in channels]
    resolution = np.array([c[1] for c in channels])
    count = int(info["NumberOfChannels"])
    fs = 1e6 / float(info["SamplingInterval"])
    if fs not in (250.0, 1000.0):
        raise ValueError("unexpected sampling rate")
    data = np.fromfile(stem + ".eeg", dtype="<f4")
    if data.size % count:
        raise ValueError("binary size is not a multiple of the channel count")
    data = data.reshape(-1, count)
    trig = data[:, names.index("TRIG")].astype(np.float64) * resolution[names.index("TRIG")]
    acc = np.column_stack([data[:, names.index(a)].astype(np.float64) * resolution[names.index(a)]
                           for a in "XYZ"]) / G_MICROVOLT
    frames = len(data)
    del data  # EEG/EOG columns are discarded unread
    return SelfPacedRun(match.group(1), match.group(2), int(match.group(3)), base, fs, frames,
                        int(SP_TICK_HZ // fs), trig, acc)


def reduce_to_detector_rate(acc, fs):
    """Causal reduction to 250 Hz: 1-kHz runs use the mean of samples 4k..4k+3 at sample 4k+3.

    Returns the reduced (m, 3) array and the 1-kHz tick at which each reduced
    sample becomes available (250-Hz runs: tick 4k).
    """
    if fs == SP_DETECTOR_HZ:
        return acc, np.arange(len(acc)) * (SP_TICK_HZ // SP_DETECTOR_HZ)
    if fs != 1000.0:
        raise ValueError("unsupported rate")
    m = len(acc) // 4
    reduced = acc[:4 * m].reshape(m, 4, 3).mean(axis=1)
    return reduced, np.arange(m) * 4 + 3


def trailing_sd(values, width):
    """Trailing population SD over `width` samples (NaN until full)."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    if len(values) < width:
        return out
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    pad = np.zeros((1,) + values.shape[1:])
    c1 = np.concatenate([pad, np.cumsum(safe, axis=0)])
    c2 = np.concatenate([pad, np.cumsum(safe * safe, axis=0)])
    counts = np.concatenate([np.zeros((1,) + values.shape[1:], dtype=np.int64), np.cumsum(finite, axis=0)])
    mean = (c1[width:] - c1[:-width]) / width
    var = (c2[width:] - c2[:-width]) / width - mean * mean
    complete = counts[width:] - counts[:-width] == width
    out[width - 1:] = np.where(complete, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return out


def derivative_onsets(participant, session, raw_frames_by_run):
    """Provider TrialStartIndex/TrialEndIndex/AccStartIndex (agreement only) per raw run.

    Derivative runs are bound to raw runs by unique frame count (October 16 rule);
    MATLAB string columns are not decodable by scipy. Indices are returned as stored
    (1-based samples of the bound run) together with the derivative rate.
    """
    if participant not in SP_DEVELOPMENT:
        raise ValueError("held-out derivatives stay closed")
    member = (f"derivatives/matfiles/{participant}/{session}/"
              f"{participant}_{session}_task-reachingandgrasping_eeg.mat")
    with zipfile.ZipFile(SP_ARCHIVE) as archive:
        raw = archive.read(member)
    record = sio.loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["data"]
    del raw
    table = record.dataTable
    header = [str(h) for h in table[0]]
    rows = table[1:-1] if str(table[-1][0]).strip() in ("", "[]") else table[1:]
    raw_eeg = record.rawEEG
    if isinstance(raw_eeg, np.ndarray) and raw_eeg.dtype != object:
        blocks = [raw_eeg.shape]                     # a single run is not wrapped in a cell
    else:
        blocks = [np.shape(b) for b in np.atleast_1d(raw_eeg)]
    del raw_eeg
    fs = float(record.fs)
    del record
    frames = [max(shape) for shape in blocks]
    binding = {}
    for k, count in enumerate(frames, start=1):
        matches = [run for run, raw_count in raw_frames_by_run.items() if raw_count == count]
        binding[k] = matches[0] if len(matches) == 1 else None
    def number(value):
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return float("nan")
        return float(array[0]) if array.size == 1 else float("nan")

    out = []
    for row in rows:
        values = {name: row[header.index(name)] for name in header
                  if name not in ("EEGFileNumber", "Comments")}
        derivative_run = number(values["Run"])
        if not np.isfinite(derivative_run):
            continue
        derivative_run = int(derivative_run)
        out.append({"derivative_run": derivative_run, "raw_run": binding.get(derivative_run),
                    "trial": number(values["TrialNumber"]), "discard": number(values["TrialDiscardIndex"]),
                    "trial_start_index": number(values["TrialStartIndex"]),
                    "trial_end_index": number(values["TrialEndIndex"]),
                    "acc_start_index": number(values["AccStartIndex"]),
                    "acc_flag": number(values["AccDetectionFlag"]),
                    "trig_start_flag": number(values["TRIGSTARTDetectionFlag"]),
                    "trig_end_flag": number(values["TRIGENDDetectionFlag"])})
    return {"fs": fs, "derivative_frames": frames, "binding": binding, "rows": out}
