"""Past-only 20-ms sensor summaries for the EP19 onset/stillness detectors.

WAY paths (500-Hz ticks), in this order: five EMG envelopes, then index (P2),
thumb (P3) and wrist (P4) speed. EMG at tick i uses native 4-kHz samples up to
8i: residual from the strictly past 500-ms mean, RMS over the trailing 20 ms.
Speed at tick i is the 20-ms backward displacement |x(i) - x(i-10)| / 0.02 s.

Self-paced path (250-Hz detector samples): departure norm |mean of the latest
20 ms - mean of the preceding 500 ms| of the 3-axis accelerometer vector, in g.
No centered filter, future padding or whole-record normalization is used.
"""
import numpy as np

WAY_PATHS = ("emg_1", "emg_2", "emg_3", "emg_4", "emg_5", "index_p2_speed", "thumb_p3_speed",
             "wrist_p4_speed")
EMG_CENTER_SAMPLES = 2000      # 500 ms at 4 kHz, strictly past
EMG_SUMMARY_SAMPLES = 80       # 20 ms at 4 kHz
SPEED_LAG_TICKS = 10           # 20 ms at 500 Hz
WAY_SUMMARY_SUPPORT_TICKS = 10  # raw support of a WAY summary beyond its own tick (20 ms)
SP_SUMMARY_SAMPLES = 5         # 20 ms at 250 Hz
SP_BASE_SAMPLES = 125          # 500 ms at 250 Hz, ending where the 20-ms window begins
SP_SUMMARY_SUPPORT_SAMPLES = 5


def trailing_mean(values, width):
    """Causal mean of the last `width` samples; NaN until complete or after a nonfinite sample."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    if len(values) < width:
        return out
    valid = np.isfinite(values)
    pad = np.zeros((1,) + values.shape[1:])
    sums = np.concatenate([pad, np.cumsum(np.where(valid, values, 0.0), axis=0)])
    counts = np.concatenate([pad, np.cumsum(valid, axis=0)])
    complete = counts[width:] - counts[:-width] == width
    out[width - 1:] = np.where(complete, (sums[width:] - sums[:-width]) / width, np.nan)
    return out


def way_summaries(emg, positions_cm, n_ticks):
    """(n_ticks, 8) summaries; rows before complete history are NaN."""
    emg = np.asarray(emg, dtype=np.float64)
    center = trailing_mean(emg, EMG_CENTER_SAMPLES)
    residual = np.full_like(emg, np.nan)
    residual[1:] = emg[1:] - center[:-1]
    envelope = np.sqrt(np.maximum(trailing_mean(residual * residual, EMG_SUMMARY_SAMPLES), 0.0))
    ticks = np.arange(n_ticks)
    out = np.full((n_ticks, 8), np.nan)
    out[:, :5] = envelope[ticks * 8]
    hand = np.asarray(positions_cm, dtype=np.float64)[:, 1:4]
    speed = np.full((n_ticks, 3), np.nan)
    speed[SPEED_LAG_TICKS:] = np.linalg.norm(hand[SPEED_LAG_TICKS:] - hand[:-SPEED_LAG_TICKS], axis=2) / 0.02
    out[:, 5:] = speed
    return out


def object_velocity(positions_cm, lag=SPEED_LAG_TICKS):
    """Backward 20-ms per-axis velocity of the object sensor P1, cm/s."""
    p1 = np.asarray(positions_cm, dtype=np.float64)[:, 0]
    out = np.full(p1.shape, np.nan)
    out[lag:] = (p1[lag:] - p1[:-lag]) / (lag / 500.0)
    return out


def self_paced_departure(acc_g):
    """(m,) departure norm in g at 250-Hz detector samples; NaN before 520 ms of history."""
    acc = np.asarray(acc_g, dtype=np.float64)
    recent = trailing_mean(acc, SP_SUMMARY_SAMPLES)
    base = trailing_mean(acc, SP_BASE_SAMPLES)
    out = np.full(len(acc), np.nan)
    lag = SP_SUMMARY_SAMPLES
    out[lag:] = np.linalg.norm(recent[lag:] - base[:-lag], axis=1)
    return out
