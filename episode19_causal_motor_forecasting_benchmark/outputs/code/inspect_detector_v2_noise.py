#!/usr/bin/env python3
"""EP19 round-1 noise-only inspection. Python 3.12; numpy plus supplied modules.

No detector is selected or frozen here. No real movement onset, event crossing,
forecast label, provider comparison, or EEG feature is computed by this script.

IMPORTANT ACCESS LIMITATION:
The required, existing sp_load reader loads whole raw runs and computes whole-run
accelerometer summaries and TRIG classifications. It cannot enforce physical
pre-start-only I/O. Execution requires --allow-incidental-full-run-load.
Additional calculations below use only declared pre-start noise windows,
their preceding diagnostic support, and end+1-s -> start background intervals.

All new probe grids, stricter-screen diagnostics, numerical tolerances, and
noise-replay conventions are declared inspection assumptions, not source facts.
Outputs require a new directory and never replace existing artifacts.
"""

from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
import re
import socket
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector_freeze as F
import detector_readers as R
import detector_summaries as S
import self_paced_raw_onset_test as V4
from run_detector_freeze import sp_load


PEOPLE = (
    "sub-02", "sub-03", "sub-04", "sub-06", "sub-08",
    "sub-09", "sub-11", "sub-12", "sub-14", "sub-15",
    "sub-17", "sub-18", "sub-20", "sub-21", "sub-22",
)
SPLITS = ("fit", "validation", "all")
RUNGS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
LEVELS = np.arange(4, 61, 2, dtype=np.float64) / 1000.0

TAIL_START = 130
TAIL_N = 120
TAIL_LEVEL_G = 0.03
STRICT_BLOCK_N = 5
BACKGROUND_N = 700
STILL_N = 475
SAFE_STILL_OFFSET_N = 130
QUIET_N = 125
# Final 1.9 s: 520 ms feature warm-up, 500 ms quiet history,
# then 880 ms of synthetic-cue surveillance.
SAFE_BACKGROUND_N = STILL_N - SAFE_STILL_OFFSET_N  # 345
REPLAY_N = SAFE_BACKGROUND_N - QUIET_N            # 220

EXPECTED = {
    "runs": 155,
    "start_tones": 4432,
    "windows": 3661,
    "fit": 1901,
    "validation": 1760,
}
EXPECTED_LOW = 0.08509930134406786
EXPECTED_HIGH = 0.16721586763828217

EXPECTED_BACKGROUND_COUNTS = dict(zip(PEOPLE, (
    153, 261, 60, 61, 191, 159, 174, 110, 236, 224,
    80, 150, 197, 327, 162,
)))
EXPECTED_THETAS = dict(zip(PEOPLE, (
    (0.0098, 0.0139, 0.0201),
    (0.0075, 0.0106, 0.0152),
    (0.0158, 0.0225, 0.0327),
    (0.0101, 0.0143, 0.0205),
    (0.0114, 0.0158, 0.0225),
    (0.0082, 0.0116, 0.0166),
    (0.0071, 0.0099, 0.0141),
    (0.0092, 0.0130, 0.0187),
    (0.0090, 0.0127, 0.0183),
    (0.0090, 0.0127, 0.0182),
    (0.0125, 0.0175, 0.0250),
    (0.0121, 0.0171, 0.0246),
    (0.0096, 0.0136, 0.0195),
    (0.0078, 0.0109, 0.0155),
    (0.0126, 0.0176, 0.0253),
)))

INSPECTION_DECLARATION = """\
1. Round-1 inspection only: no detector selection, real-event estimation, labels, provider access, or EEG feature use.
2. Only the 15 named development people; supplied development_stems and sp_load readers; whole-run loader access requires explicit acknowledgment.
3. Reproduce frozen stationary membership/parity; primary corrected X support is window indices 130:250; index 129 is separately identified.
4. Inspect quantile and MAD scales with c={0.5,0.75,1,1.5,2,3,4}; all probes are reported without selecting a rung.
5. Inspect levels 0.004:0.002:0.060 g, a 0.03-g tail diagnostic, and a 20-ms/0.02-g stricter screen without changing membership.
6. Reproduce native v4 references from final 480 ms, using both parities only for that reference; detector probe scales use pooled fit X only.
7. Retain every eligible v4 background, with no reservoir subsampling; assign diagnostic parity from its upcoming classified start index.
8. Background replay uses only support-complete X: 520-ms warm-up, 500-ms quiet history, then one synthetic cue for the remaining 880 ms.
9. D1/D2 noise replays retain one-event disarming and rejection/requalification; report pending truncation separately from confirmed false events.
10. Write fresh JSON/CSV only; rounded-reference and count checks must pass before these outputs inform round 2.
"""


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(safe_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def require_slurm(workers):
    host = socket.gethostname()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in host.lower():
        raise RuntimeError("An actual Slurm compute-node shell is required.")
    if os.environ.get("SLURM_PROCID", "0") != "0":
        raise RuntimeError("Launch one coordinating Python process.")
    declared = os.environ.get("SLURM_CPUS_PER_TASK")
    if not declared or not re.fullmatch(r"\d+", declared):
        raise RuntimeError("SLURM_CPUS_PER_TASK must specify an explicit CPU limit.")
    limit = int(declared)
    if hasattr(os, "sched_getaffinity"):
        limit = min(limit, len(os.sched_getaffinity(0)))
    if workers < 1 or workers > min(8, limit):
        raise RuntimeError(f"Requested workers={workers}; usable CPU limit={limit}.")
    return host, limit


def check_imported_contract():
    if tuple(R.SP_DEVELOPMENT) != PEOPLE:
        raise RuntimeError("Development participant contract differs.")
    expected_constants = (
        F.SP_NOISE_SAMPLES == 250,
        F.SP_BLOCK_SAMPLES == 25,
        F.SP_STILL_BLOCK_G == 0.02,
        F.SP_FLAT_SD_G == 1e-4,
        F.SP_CONFIRM == 10,
        F.SP_QUIET == 125,
        F.SP_GRID_TICKS == 50,
        S.SP_SUMMARY_SAMPLES == 5,
        S.SP_BASE_SAMPLES == 125,
        R.SP_DETECTOR_HZ == 250,
        V4.BACKGROUND_MS == 2800,
        V4.STILL_MS == 1900,
        V4.WARMUP_MS == 900,
        V4.END_TONE_OFFSET_MS == 1000,
    )
    if not all(expected_constants):
        raise RuntimeError("Imported constants differ from the supplied appendix.")


def stem_identity(stem):
    match = re.fullmatch(
        r"(sub-\d+)_(ses-\d+)_task-reachingandgrasping_run-(\d+)_eeg",
        Path(stem).name,
    )
    if match is None or match.group(1) not in PEOPLE:
        raise ValueError(f"Unexpected development stem: {stem}")
    return match.group(1), match.group(2), int(match.group(3))


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite values entered a retained noise distribution.")
    median = float(np.median(values))
    mad_sd = float(1.4826 * np.median(np.abs(values - median)))
    qs = np.percentile(values, (90, 99, 99.5, 99.9, 99.99), method="linear")
    return {
        "n": int(len(values)),
        "median_g": median,
        "mad_sd_g": mad_sd,
        "q90_g": float(qs[0]),
        "q99_g": float(qs[1]),
        "q99_5_g": float(qs[2]),
        "q99_9_g": float(qs[3]),
        "q99_99_g": float(qs[4]),
        "max_g": float(np.max(values)),
    }


def concatenate(parts):
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def inspect_person(task):
    """Read one run at a time. Retain only small pre-start noise arrays."""
    participant, stems = task
    require_slurm(1)
    windows, backgrounds, run_rows = [], [], []
    expected_fs = 1000.0 if participant == "sub-15" else 250.0

    for stem in stems:
        run, acc, ticks, dev, valid, bursts, _ = sp_load(stem)
        if run.participant != participant or run.fs != expected_fs:
            raise RuntimeError("Participant or native-rate mismatch.")
        factor = int(round(run.fs / 250.0))
        phase = 3 if factor == 4 else 0

        starts = sorted(
            (b for b in bursts if b.get("tone") and b["class"] == "start"),
            key=lambda b: b["onset_tick"],
        )
        end_tones = np.asarray(sorted(
            int(b["onset_tick"]) for b in bursts
            if b.get("tone") and b["class"] == "end"
        ), dtype=np.int64)

        run_row = {
            "participant": participant,
            "session": run.session,
            "run": run.run,
            "fs": run.fs,
            "start_tones": len(starts),
            "fit": 0,
            "validation": 0,
            "background_intervals": 0,
            "background_intervals_with_room": 0,
            "backgrounds": 0,
        }
        last_eligible_end = 0

        for start_index, burst in enumerate(starts):
            split = "fit" if start_index % 2 == 0 else "validation"
            tone_tick = int(burst["onset_tick"])
            end = int(np.searchsorted(ticks, tone_tick, side="left"))
            begin = end - F.SP_NOISE_SAMPLES

            # Exactly the frozen window rule, as implemented in the appendix.
            if V4.still_screen(acc, valid, begin, end):
                section = acc[begin:end]
                x = np.asarray(dev[begin:end], dtype=np.float64).copy()
                native_section = run.acc_g[begin * factor:end * factor]
                native_r = V4.native_statistic(native_section, run.fs)[
                    int(round(0.520 * run.fs)):
                ].copy()
                r250 = V4.native_statistic(section, 250.0)[TAIL_START:].copy()

                if (
                    len(x) != 250 or len(native_r) != 120 * factor
                    or len(r250) != 120
                    or not np.all(np.isfinite(x))
                    or not np.all(np.isfinite(native_r))
                    or not np.all(np.isfinite(r250))
                ):
                    raise RuntimeError("Retained window support is inconsistent.")
                if begin < 129:
                    raise RuntimeError("Frozen validity lacks diagnostic prehistory.")

                center = section.mean(axis=0)
                strict_blocks = section.reshape(-1, STRICT_BLOCK_N, 3).mean(axis=1)
                strict_spread = float(np.max(
                    np.linalg.norm(strict_blocks - center, axis=1)
                ))
                early = x[:129] > TAIL_LEVEL_G
                boundary = bool(x[129] > TAIL_LEVEL_G)
                tail = x[130:] > TAIL_LEVEL_G
                safe_any = boundary or bool(np.any(tail))
                early_any = bool(np.any(early))
                pre_mean = acc[begin - 129:begin].mean(axis=0)

                meta = {
                    "participant": participant,
                    "session": run.session,
                    "run": run.run,
                    "fs": run.fs,
                    "start_index": start_index,
                    "split": split,
                    "start_tone_tick": tone_tick,
                    "window_raw_start_tick": int(begin * 4),
                    "window_raw_stop_tick": int(end * 4),
                    "max_x_g": float(np.max(x)),
                    "max_early_x_g": float(np.max(x[:129])),
                    "boundary_x_g": float(x[129]),
                    "max_tail_x_g": float(np.max(x[130:])),
                    "hot_window": int(early_any or safe_any),
                    "early_hot": int(early_any),
                    "safe_hot": int(safe_any),
                    "tail_hot": int(np.any(tail)),
                    "early_only_hot": int(early_any and not safe_any),
                    "safe_only_hot": int(safe_any and not early_any),
                    "early_and_safe_hot": int(early_any and safe_any),
                    "boundary_hot": int(boundary),
                    "early_hot_samples": int(np.sum(early)),
                    "boundary_hot_samples": int(boundary),
                    "tail_hot_samples": int(np.sum(tail)),
                    "strict_pass": int(strict_spread <= 0.02),
                    "strict_max_block_deviation_g": strict_spread,
                    "raw_max_sample_deviation_g": float(np.max(
                        np.linalg.norm(section - center, axis=1)
                    )),
                    "prewindow_mean_offset_g": float(np.linalg.norm(pre_mean - center)),
                }
                windows.append({
                    "meta": meta, "x": x, "native_r": native_r, "r250": r250,
                })
                run_row[split] += 1

            # Background membership does not depend on acceptance of that 1-s window.
            previous_index = int(
                np.searchsorted(end_tones, tone_tick, side="left")
            ) - 1
            if previous_index < 0:
                continue
            previous_end_tick = int(end_tones[previous_index])
            interval_start = previous_end_tick + 1000
            run_row["background_intervals"] += 1

            # Translate the v4 search to this pre-start interval. This avoids
            # constructing additional whole-run accelerometer prefix sums.
            offset = max(0, (interval_start + 3) // 4)
            local_end = end
            if local_end < offset:
                continue
            interval_acc = acc[offset:local_end]
            interval_valid = valid[offset:local_end]
            local_ticks = ticks[offset:local_end] - 4 * offset
            acc_prefix = np.vstack((
                np.zeros((1, 3)),
                np.cumsum(
                    np.where(np.isfinite(interval_acc), interval_acc, 0.0),
                    axis=0,
                ),
            ))
            bad_prefix = np.r_[0, np.cumsum(~interval_valid, dtype=np.int64)]
            selected_local_end, had_room = V4.latest_background_end(
                local_ticks,
                acc_prefix,
                bad_prefix,
                previous_end_tick - 4 * offset,
                tone_tick - 4 * offset,
                max(0, last_eligible_end - offset),
            )
            run_row["background_intervals_with_room"] += int(had_room)
            if selected_local_end is None:
                continue

            selected_end = offset + int(selected_local_end)
            selected_begin = selected_end - BACKGROUND_N
            first_tick = selected_begin * 4
            stop_tick = selected_end * 4
            native_section = run.acc_g[
                selected_begin * factor:selected_end * factor
            ]
            if (
                first_tick < interval_start or stop_tick > tone_tick
                or stop_tick - first_tick != 2800
                or len(native_section) != 700 * factor
                or not np.all(np.isfinite(native_section))
                or not np.all(valid[selected_begin:selected_end])
            ):
                raise RuntimeError("Background violates v4 interval/support rules.")

            last_eligible_end = selected_end
            safe_x = dev[selected_end - SAFE_BACKGROUND_N:selected_end].copy()
            if len(safe_x) != 345 or not np.all(np.isfinite(safe_x)):
                raise RuntimeError("Incorrect support-complete background X.")
            meta = {
                "participant": participant,
                "session": run.session,
                "run": run.run,
                "fs": run.fs,
                "start_index": start_index,
                "split": split,
                "previous_end_tone_tick": previous_end_tick,
                "start_tone_tick": tone_tick,
                "interval_start_tick": interval_start,
                "background_start_tick": first_tick,
                "background_stop_tick": stop_tick,
                "screened_still_start_tick": first_tick + 900,
                "safe_x_start_tick": first_tick + 1420 + phase,
                "virtual_cue_tick": first_tick + 1920 + phase,
            }
            backgrounds.append({"meta": meta, "x": safe_x})
            run_row["backgrounds"] += 1

        if (
            participant == "sub-22" and run.session == "ses-01"
            and run.run in (6, 7)
            and (run_row["fit"] + run_row["validation"] + run_row["backgrounds"])
        ):
            raise RuntimeError("A known flat-axis run contributed retained noise.")

        run_rows.append(run_row)
        del run, acc, ticks, dev, valid, bursts

    return {
        "participant": participant,
        "fs": expected_fs,
        "windows": windows,
        "backgrounds": backgrounds,
        "runs": run_rows,
    }


def d2_noise_replay(values, low, high, supported=True):
    """One synthetic cue; counters/states only, never real-event labels.

    Pending starts at LOW. HIGH starts a ten-future-sample confirmation clock.
    Any drop below LOW rejects. Surveillance continues during requalification,
    matching the supplied rule that a crossing can confirm while requalifying.
    """
    values = np.asarray(values)
    states = np.full(len(values), F.STATE_CODES["unarmed"], dtype=np.int8)
    starts = confirmed = rejected = 0
    pending = False
    high_at = None
    armed = bool(supported)
    requalifying = False
    quiet = 0

    if not supported:
        return {
            "states": states, "starts": 0, "confirmed": 0, "rejected": 0,
            "open_pending": 0, "open_low_only": 0, "open_high_seen": 0,
        }

    for i, value in enumerate(values):
        if requalifying and not pending and quiet >= QUIET_N:
            armed, requalifying = True, False

        if not pending and value >= low:
            pending = True
            high_at = None
            starts += 1
            armed = False
            requalifying = True
            quiet = 0

        had_pending = pending
        if pending:
            if value < low:
                pending = False
                high_at = None
                rejected += 1
            else:
                if high_at is None and value >= high:
                    high_at = i
                if high_at is not None and i - high_at >= F.SP_CONFIRM:
                    confirmed = 1
                    pending = False
                    high_at = None
                    # States were initialized unarmed; no rearming after acceptance.
                    break

        if pending:
            quiet = 0
        elif requalifying:
            if had_pending:
                quiet = 0
            quiet = quiet + 1 if value <= low else 0

        states[i] = (
            F.STATE_CODES["at_risk"] if armed else F.STATE_CODES["pending"]
        )

    return {
        "states": states,
        "starts": starts,
        "confirmed": confirmed,
        "rejected": rejected,
        "open_pending": int(pending),
        "open_low_only": int(pending and high_at is None),
        "open_high_seen": int(pending and high_at is not None),
    }


def d1_noise_replay(values, low, high, supported):
    states, _, cues = F.run_segment(
        np.asarray(values)[:, None],
        np.asarray([low]), np.asarray([high]),
        F.SP_CONFIRM, F.SP_QUIET,
        {0: "supported" if supported else "not_supported"},
        set(),
    )
    if len(cues) != 1:
        raise RuntimeError("Synthetic D1 replay did not produce exactly one cue.")
    cue = cues[0]
    confirmed = len(cue.accepted)
    rejected = len(cue.rejected)
    pending = len(cue.pending_at_stop)
    return {
        "states": states,
        "starts": confirmed + rejected + pending,
        "confirmed": confirmed,
        "rejected": rejected,
        "open_pending": pending,
        "open_low_only": 0,
        "open_high_seen": pending,
    }


def has_quiet_run(values, low):
    quiet = np.asarray(values) <= low
    prefix = np.r_[0, np.cumsum(quiet, dtype=np.int64)]
    return bool(np.any(prefix[QUIET_N:] - prefix[:-QUIET_N] == QUIET_N))


CROSS_COUNTS = (
    "windows", "v1_hit_windows", "d2_confirmed_windows",
    "d2_open_pending_windows", "d2_low_only_pending_windows",
    "d2_high_seen_pending_windows", "d2_hit_windows",
)
BG_COUNTS = (
    "backgrounds", "any_quiet_backgrounds", "fixed_entry_backgrounds",
    "candidate_starts", "confirmed_false_onsets", "rejected_candidates",
    "open_pending_at_end", "open_low_only_at_end",
    "grid_points", "at_risk_grid_points", "pending_grid_points",
    "supported_grid_points", "evaluation_ms", "supported_ms",
)


def probe_person(task):
    data, thresholds = task
    participant = data["participant"]
    cross_rows, bg_rows = [], []

    for threshold in thresholds:
        low, high = threshold["low_g"], threshold["high_g"]
        cc = {split: dict.fromkeys(CROSS_COUNTS, 0) for split in SPLITS}
        bc = {
            (model, split): dict.fromkeys(BG_COUNTS, 0)
            for model in ("D1", "D2") for split in SPLITS
        }

        for window in data["windows"]:
            x = window["x"][TAIL_START:]
            v1_hit = int(F.confirmed_crossing(x, low, high, F.SP_CONFIRM))
            d2 = d2_noise_replay(x, low, high)
            # Actual confirmation / HIGH-seen end censoring must agree with v1.
            if v1_hit != int(bool(d2["confirmed"] or d2["open_high_seen"])):
                raise RuntimeError("D1/D2 noise confirmation semantics disagree.")
            facts = {
                "windows": 1,
                "v1_hit_windows": v1_hit,
                "d2_confirmed_windows": d2["confirmed"],
                "d2_open_pending_windows": d2["open_pending"],
                "d2_low_only_pending_windows": d2["open_low_only"],
                "d2_high_seen_pending_windows": d2["open_high_seen"],
                "d2_hit_windows": int(bool(d2["confirmed"] or d2["open_pending"])),
            }
            for split in (window["meta"]["split"], "all"):
                for key, value in facts.items():
                    cc[split][key] += value

        for background in data["backgrounds"]:
            x, meta = background["x"], background["meta"]
            any_quiet = has_quiet_run(x, low)
            fixed_entry = bool(np.all(x[:QUIET_N] <= low))
            values = x[QUIET_N:]
            if len(values) != REPLAY_N:
                raise RuntimeError("Incorrect synthetic surveillance length.")

            first_tick = int(meta["virtual_cue_tick"])
            stop_tick = int(meta["background_stop_tick"])
            grid = np.arange(((first_tick + 49) // 50) * 50, stop_tick, 50)
            sample_indices = ((grid - first_tick) // 4).astype(np.int64)
            if np.any(sample_indices < 0) or np.any(sample_indices >= len(values)):
                raise RuntimeError("Invalid synthetic grid projection.")
            duration_ms = stop_tick - first_tick

            for model in ("D1", "D2"):
                replay = (
                    d1_noise_replay(values, low, high, fixed_entry)
                    if model == "D1"
                    else d2_noise_replay(values, low, high, fixed_entry)
                )
                grid_states = replay["states"][sample_indices]
                facts = {
                    "backgrounds": 1,
                    "any_quiet_backgrounds": int(any_quiet),
                    "fixed_entry_backgrounds": int(fixed_entry),
                    "candidate_starts": replay["starts"],
                    "confirmed_false_onsets": replay["confirmed"],
                    "rejected_candidates": replay["rejected"],
                    "open_pending_at_end": replay["open_pending"],
                    "open_low_only_at_end": replay["open_low_only"],
                    "grid_points": int(len(grid)),
                    "at_risk_grid_points": int(np.sum(
                        grid_states == F.STATE_CODES["at_risk"]
                    )),
                    "pending_grid_points": int(np.sum(
                        grid_states == F.STATE_CODES["pending"]
                    )),
                    "supported_grid_points": int(len(grid)) if fixed_entry else 0,
                    "evaluation_ms": int(duration_ms),
                    "supported_ms": int(duration_ms) if fixed_entry else 0,
                }
                for split in (meta["split"], "all"):
                    for key, value in facts.items():
                        bc[(model, split)][key] += value

        for split in SPLITS:
            cross_rows.append({
                "participant": participant,
                "threshold_id": threshold["threshold_id"],
                "split": split,
                **cc[split],
            })
            for model in ("D1", "D2"):
                bg_rows.append({
                    "participant": participant,
                    "threshold_id": threshold["threshold_id"],
                    "model": model,
                    "split": split,
                    **bc[(model, split)],
                })
    return cross_rows, bg_rows


def add_pooled(rows, grouping, count_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in grouping)].append(row)
    pooled = []
    for key, members in sorted(groups.items()):
        pooled.append({
            "participant": "POOL",
            **dict(zip(grouping, key)),
            **{field: sum(row[field] for row in members) for field in count_fields},
        })
    return rows + pooled


def finish_cross(row):
    minutes = row["windows"] * 0.480 / 60.0
    return {
        **row,
        "noise_minutes": minutes,
        "v1_fraction": fraction(row["v1_hit_windows"], row["windows"]),
        "d2_fraction": fraction(row["d2_hit_windows"], row["windows"]),
        "v1_flagged_windows_per_minute": fraction(row["v1_hit_windows"], minutes),
        "d2_flagged_windows_per_minute": fraction(row["d2_hit_windows"], minutes),
    }


def finish_background(row):
    minutes = row["evaluation_ms"] / 60000.0
    supported_minutes = row["supported_ms"] / 60000.0
    return {
        **row,
        "evaluation_minutes": minutes,
        "supported_minutes": supported_minutes,
        "any_quiet_fraction": fraction(
            row["any_quiet_backgrounds"], row["backgrounds"]
        ),
        "fixed_entry_fraction": fraction(
            row["fixed_entry_backgrounds"], row["backgrounds"]
        ),
        "candidate_starts_per_minute": fraction(row["candidate_starts"], minutes),
        "confirmed_false_onsets_per_minute": fraction(
            row["confirmed_false_onsets"], minutes
        ),
        "candidate_starts_per_supported_minute": fraction(
            row["candidate_starts"], supported_minutes
        ),
        "confirmed_false_onsets_per_supported_minute": fraction(
            row["confirmed_false_onsets"], supported_minutes
        ),
        "at_risk_grid_fraction": fraction(
            row["at_risk_grid_points"], row["grid_points"]
        ),
        "at_risk_grid_fraction_given_supported_entry": fraction(
            row["at_risk_grid_points"], row["supported_grid_points"]
        ),
        "pending_grid_fraction": fraction(
            row["pending_grid_points"], row["grid_points"]
        ),
    }


def distribution_tables(data):
    distribution_rows, level_rows, tail_rows = [], [], []
    groups = [(person["participant"], person["windows"]) for person in data]
    groups.append(("POOL", [w for person in data for w in person["windows"]]))

    for participant, windows in groups:
        for split in SPLITS:
            selected = [
                w for w in windows
                if split == "all" or w["meta"]["split"] == split
            ]
            arrays = {
                "X_full": concatenate([w["x"] for w in selected]),
                "X_tail": concatenate([w["x"][130:] for w in selected]),
                "X_early": concatenate([w["x"][:129] for w in selected]),
                "X_tail_strict_pass": concatenate([
                    w["x"][130:] for w in selected if w["meta"]["strict_pass"]
                ]),
                "X_tail_strict_fail": concatenate([
                    w["x"][130:] for w in selected if not w["meta"]["strict_pass"]
                ]),
                "r_native": concatenate([w["native_r"] for w in selected]),
                "r_250_unsmoothed": concatenate([w["r250"] for w in selected]),
            }
            for statistic, values in arrays.items():
                stats = describe(values)
                row = {
                    "participant": participant, "split": split,
                    "statistic": statistic, "windows": len(selected), **stats,
                }
                if statistic.startswith("r_") and stats["n"]:
                    for k in (3, 5, 8):
                        row[f"theta_k{k}_g"] = (
                            stats["median_g"] + k * stats["mad_sd_g"]
                        )
                distribution_rows.append(row)
                for level in LEVELS:
                    above = int(np.sum(values > level))
                    level_rows.append({
                        "participant": participant, "split": split,
                        "statistic": statistic, "level_g": float(level),
                        "n": len(values), "above": above,
                        "fraction_above": fraction(above, len(values)),
                    })

            metas = [w["meta"] for w in selected]
            fields = (
                "hot_window", "early_hot", "safe_hot", "tail_hot",
                "early_only_hot", "safe_only_hot", "early_and_safe_hot",
                "boundary_hot", "early_hot_samples", "boundary_hot_samples",
                "tail_hot_samples", "strict_pass",
            )
            totals = {key: sum(m[key] for m in metas) for key in fields}
            hot = totals["hot_window"]
            tail_rows.append({
                "participant": participant, "split": split,
                "windows": len(selected), **totals,
                "hot_windows_strict_pass": sum(
                    m["hot_window"] and m["strict_pass"] for m in metas
                ),
                "tail_hot_windows_strict_pass": sum(
                    m["tail_hot"] and m["strict_pass"] for m in metas
                ),
                "tail_hot_windows_strict_fail": sum(
                    m["tail_hot"] and not m["strict_pass"] for m in metas
                ),
                "early_only_fraction_of_hot_windows": fraction(
                    totals["early_only_hot"], hot
                ),
                "safe_hot_fraction_of_hot_windows": fraction(
                    totals["safe_hot"], hot
                ),
            })
    return distribution_rows, level_rows, tail_rows


def make_thresholds(data):
    fit_full = [
        w["x"][:, None]
        for person in data for w in person["windows"]
        if w["meta"]["split"] == "fit"
    ]
    fit_tail = concatenate([w[130:, 0] for w in fit_full])
    if not len(fit_tail):
        raise RuntimeError("No fit-window X samples.")
    stats = describe(fit_tail)
    median = stats["median_g"]
    scales = {
        "quantile": stats["q99_9_g"] - median,
        "mad": stats["mad_sd_g"],
    }
    thresholds = []
    for scale_name, scale in scales.items():
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"Invalid candidate scale: {scale_name}.")
        for rung in RUNGS:
            thresholds.append({
                "threshold_id": f"{scale_name}_c{rung:g}",
                "scale": scale_name, "rung": rung,
                "median_g": median, "scale_g": scale,
                "low_g": median + rung * scale,
                "high_g": median + 2.0 * rung * scale,
            })
    original_fit = F.fit_thresholds(fit_full, F.SP_CONFIRM)[0]
    return thresholds, dict(original_fit.__dict__)


def feasibility_table(thresholds, references):
    rows = []
    # For a steady scalar linear acceleration ramp:
    # X gain = 260 ms, independent of the causal 1-kHz block-mean phase.
    x_gain_s = ((S.SP_SUMMARY_SAMPLES + S.SP_BASE_SAMPLES) / 2.0) / 250.0
    for reference in references:
        fs = reference["fs"]
        width = int(round(0.500 * fs))
        exclusion = int(round(0.020 * fs))
        native_gain_s = (exclusion + (width + 1) / 2.0) / fs
        ratio_x_to_r = x_gain_s / native_gain_s
        theta = reference["theta_k5_g"]
        for threshold in thresholds:
            for model in ("D1", "D2"):
                label_level = threshold["high_g"] if model == "D1" else threshold["low_g"]
                upper_r = label_level / ratio_x_to_r
                possible = theta < upper_r
                plateau = (theta + upper_r) / 2.0 if possible else None
                rows.append({
                    "participant": reference["participant"],
                    "fs": fs,
                    "threshold_id": threshold["threshold_id"],
                    "model": model,
                    "theta_k5_g": theta,
                    "label_threshold_x_g": label_level,
                    "equal_statistic_comparison_theta_lt_label_level": theta < label_level,
                    "steady_ramp_x_gain_s": x_gain_s,
                    "steady_ramp_native_r_gain_s": native_gain_s,
                    "steady_ramp_x_to_r_ratio": ratio_x_to_r,
                    "admissible_r_plateau_lower_open_g": theta,
                    "admissible_r_plateau_upper_open_g": upper_r,
                    "steady_ramp_plateau_interval_nonempty": possible,
                    "example_r_plateau_g": plateau,
                    "example_x_plateau_g": (
                        plateau * ratio_x_to_r if possible else None
                    ),
                    "example_acceleration_slope_g_per_s": (
                        plateau / native_gain_s if possible else None
                    ),
                    "finite_duration_control_feasibility_established": False,
                })
    return rows


def compact_summary(data, distributions, tails, thresholds, crosses, bgs, report):
    dist = {
        (r["participant"], r["split"], r["statistic"]): r
        for r in distributions
    }
    cross = {
        (r["participant"], r["split"], r["threshold_id"]): r for r in crosses
    }
    bg = {
        (r["participant"], r["split"], r["threshold_id"], r["model"]): r
        for r in bgs
    }

    def f(value):
        return "NA" if value is None else f"{value:.4g}"

    lines = [
        "INSPECTION ONLY; no detector selected; no event labels produced.",
        "COUNTS " + json.dumps(report["counts"], sort_keys=True),
        "CHECKS " + json.dumps(report["checks"], sort_keys=True),
        "V1_NOISE_REPRODUCTION " + json.dumps(report["v1_noise_fit"], sort_keys=True),
    ]
    for person in data:
        p = person["participant"]
        full = dist[(p, "all", "X_full")]
        fit = dist[(p, "fit", "X_tail")]
        val = dist[(p, "validation", "X_tail")]
        native = dist[(p, "all", "r_native")]
        nf = sum(w["meta"]["split"] == "fit" for w in person["windows"])
        nv = len(person["windows"]) - nf
        lines.append(
            f"PERSON {p} windows={nf}/{nv} backgrounds={len(person['backgrounds'])} "
            f"Xfull_m/q999={f(full.get('median_g'))}/{f(full.get('q99_9_g'))} "
            f"Xtail_fit_m/MAD/q999={f(fit.get('median_g'))}/"
            f"{f(fit.get('mad_sd_g'))}/{f(fit.get('q99_9_g'))} "
            f"Xtail_val_q999={f(val.get('q99_9_g'))} "
            f"theta3/5/8={f(native.get('theta_k3_g'))}/"
            f"{f(native.get('theta_k5_g'))}/{f(native.get('theta_k8_g'))}"
        )
    for split in ("fit", "validation"):
        for statistic in ("X_full", "X_tail", "r_native", "r_250_unsmoothed"):
            lines.append(
                "POOL_DIST " + json.dumps(dist[("POOL", split, statistic)], sort_keys=True)
            )
        tail = next(r for r in tails if r["participant"] == "POOL" and r["split"] == split)
        lines.append("POOL_TAIL " + json.dumps(tail, sort_keys=True))

    for threshold in thresholds:
        tid = threshold["threshold_id"]
        pf = cross[("POOL", "fit", tid)]
        pv = cross[("POOL", "validation", tid)]
        worst1 = max(cross[(p, "fit", tid)]["v1_fraction"] for p in PEOPLE)
        worst2 = max(cross[(p, "fit", tid)]["d2_fraction"] for p in PEOPLE)
        risk1 = min(bg[(p, "fit", tid, "D1")]["at_risk_grid_fraction"] for p in PEOPLE)
        risk2 = min(bg[(p, "fit", tid, "D2")]["at_risk_grid_fraction"] for p in PEOPLE)
        lines.append(
            f"PROBE {tid} low/high={f(threshold['low_g'])}/{f(threshold['high_g'])} "
            f"fit_D1/D2={pf['v1_hit_windows']}/{pf['d2_hit_windows']}/{pf['windows']} "
            f"val_D1/D2={pv['v1_hit_windows']}/{pv['d2_hit_windows']}/{pv['windows']} "
            f"max_person_fit_D1/D2={f(worst1)}/{f(worst2)} "
            f"min_person_fit_risk_D1/D2={f(risk1)}/{f(risk2)}"
        )

    # Full per-person quantile-ladder diagnostics, still <200 total stdout lines.
    for threshold in thresholds:
        if threshold["scale"] != "quantile":
            continue
        tid = threshold["threshold_id"]
        for p in PEOPLE:
            a, v = cross[(p, "fit", tid)], cross[(p, "validation", tid)]
            d1, d2 = bg[(p, "fit", tid, "D1")], bg[(p, "fit", tid, "D2")]
            v1, v2 = bg[(p, "validation", tid, "D1")], bg[(p, "validation", tid, "D2")]
            lines.append(
                f"DETAIL {tid} {p} "
                f"Wfit_D1/D2/n={a['v1_hit_windows']}/{a['d2_hit_windows']}/{a['windows']} "
                f"Wval_D1/D2/n={v['v1_hit_windows']}/{v['d2_hit_windows']}/{v['windows']} "
                f"Bfit/n={d1['backgrounds']} any/entry={f(d1['any_quiet_fraction'])}/"
                f"{f(d1['fixed_entry_fraction'])} risk1/2={f(d1['at_risk_grid_fraction'])}/"
                f"{f(d2['at_risk_grid_fraction'])} starts_pm1/2="
                f"{f(d1['candidate_starts_per_minute'])}/{f(d2['candidate_starts_per_minute'])} "
                f"false_pm1/2={f(d1['confirmed_false_onsets_per_minute'])}/"
                f"{f(d2['confirmed_false_onsets_per_minute'])} "
                f"Bval/n={v1['backgrounds']} entry={f(v1['fixed_entry_fraction'])} "
                f"risk1/2={f(v1['at_risk_grid_fraction'])}/{f(v2['at_risk_grid_fraction'])}"
            )
    lines.extend((
        f"STATUS {report['status']} elapsed_s={report['elapsed_s']:.1f}",
        f"OUTPUT {report['output_dir']}",
        "Full distributions, level fractions, tail attribution, probe rates, "
        "background metrics, and plateau feasibility are in JSON/CSV.",
    ))
    if len(lines) >= 200:
        raise RuntimeError("Compact summary exceeded its declared line limit.")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--allow-incidental-full-run-load", action="store_true",
        help="Acknowledge required sp_load whole-run I/O and summaries.",
    )
    args = parser.parse_args()

    host, cpu_limit = require_slurm(args.workers)
    check_imported_contract()
    if not args.allow_incidental_full_run_load:
        raise SystemExit(
            "No source data opened. The required sp_load reads/summarizes whole runs. "
            "Obtain agreement on that scope, then supply "
            "--allow-incidental-full-run-load."
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inspection_declaration.txt").open("x", encoding="utf-8") as handle:
        handle.write(INSPECTION_DECLARATION)
    started = time.monotonic()
    write_json(output / "manifest.json", {
        "purpose": "round_1_noise_only_inspection_no_selection",
        "hostname": host,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "workers": args.workers,
        "cpu_limit": cpu_limit,
        "python": sys.version,
        "numpy": np.__version__,
        "participants": PEOPLE,
        "whole_run_loader_access_acknowledged": True,
        "loader_reads_full_multiplexed_binary": True,
        "eeg_features_used": False,
        "provider_functions_called": False,
        "event_tables_or_label_files_opened": False,
        "new_inspection_choices_are_assumptions": True,
        "quantile_method": "numpy percentile method=linear",
        "level_grid_g": LEVELS,
        "threshold_probe_rungs": RUNGS,
        "corrected_X_indices": "[130,250)",
        "first_support_complete_X_index": 129,
        "native_reference_pooling": (
            "Sample pooled within each person; combined parities reproduce v4. "
            "Pooled native-rate summaries are sample weighted, so a 1-kHz "
            "window contributes four times as many samples as a 250-Hz window."
        ),
        "noise_replay": {
            "screened_still_ms": 1900,
            "feature_warmup_ms": 520,
            "initial_quiet_ms": 500,
            "nominal_surveillance_ms": 880,
            "grid": "Original run's integer 50-ms grid",
            "quiet_anywhere_scope": "Support-complete X in final 1380 ms",
            "unsupported_fixed_entry": "Unarmed for the entire synthetic cue",
            "one_false_onset_maximum_per_background": True,
            "rates": "Both all-surveillance and supported-surveillance denominators",
        },
        "per_person_frozen_window_count_constants": (
            "Not supplied in prompt; emitted for external comparison, not invented."
        ),
    })

    stems = sorted(R.development_stems())
    grouped = {p: [] for p in PEOPLE}
    seen = set()
    for stem in stems:
        identity = stem_identity(stem)
        if identity in seen:
            raise RuntimeError("Duplicate development run.")
        seen.add(identity)
        grouped[identity[0]].append(stem)
    if len(stems) != EXPECTED["runs"] or any(not grouped[p] for p in PEOPLE):
        raise RuntimeError("Development run inventory does not match the prompt.")

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        data = list(pool.map(inspect_person, [(p, grouped[p]) for p in PEOPLE]))

    run_rows = [r for person in data for r in person["runs"]]
    window_rows = [w["meta"] for person in data for w in person["windows"]]
    background_rows = [b["meta"] for person in data for b in person["backgrounds"]]
    distributions, levels, tails = distribution_tables(data)
    thresholds, v1_fit = make_thresholds(data)

    references = []
    dist_lookup = {
        (row["participant"], row["split"], row["statistic"]): row
        for row in distributions
    }
    for person in data:
        p = person["participant"]
        native = dist_lookup[(p, "all", "r_native")]
        if native["n"] == 0 or native["mad_sd_g"] <= 0:
            raise RuntimeError(f"Unusable native noise reference for {p}.")
        row = {
            "participant": p, "fs": person["fs"],
            "windows": len(person["windows"]),
            "fit": sum(w["meta"]["split"] == "fit" for w in person["windows"]),
            "validation": sum(
                w["meta"]["split"] == "validation" for w in person["windows"]
            ),
            "backgrounds": len(person["backgrounds"]),
            "expected_backgrounds": EXPECTED_BACKGROUND_COUNTS[p],
            "median_g": native["median_g"],
            "mad_sd_g": native["mad_sd_g"],
        }
        for k, expected in zip((3, 5, 8), EXPECTED_THETAS[p]):
            row[f"theta_k{k}_g"] = native[f"theta_k{k}_g"]
            row[f"expected_rounded_theta_k{k}_g"] = expected
            row[f"theta_k{k}_difference_from_rounded_g"] = (
                native[f"theta_k{k}_g"] - expected
            )
        references.append(row)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        probe_results = list(pool.map(
            probe_person, [(person, thresholds) for person in data]
        ))
    crosses = [row for result in probe_results for row in result[0]]
    bgs = [row for result in probe_results for row in result[1]]
    crosses = [
        finish_cross(row) for row in add_pooled(
            crosses, ("threshold_id", "split"), CROSS_COUNTS
        )
    ]
    bgs = [
        finish_background(row) for row in add_pooled(
            bgs, ("threshold_id", "model", "split"), BG_COUNTS
        )
    ]

    # Attach the maximum individual crossing fraction to every pooled row.
    for row in crosses:
        if row["participant"] != "POOL":
            continue
        members = [
            r for r in crosses
            if r["participant"] in PEOPLE
            and r["threshold_id"] == row["threshold_id"]
            and r["split"] == row["split"]
        ]
        for model, field in (("D1", "v1_fraction"), ("D2", "d2_fraction")):
            usable = [r for r in members if r[field] is not None]
            worst = max(usable, key=lambda r: r[field]) if usable else None
            row[f"max_person_{model}_fraction"] = None if worst is None else worst[field]
            row[f"max_person_{model}_participant"] = None if worst is None else worst["participant"]

    counts = {
        "runs": len(run_rows),
        "start_tones": sum(r["start_tones"] for r in run_rows),
        "windows": len(window_rows),
        "fit": sum(r["split"] == "fit" for r in window_rows),
        "validation": sum(r["split"] == "validation" for r in window_rows),
        "backgrounds": len(background_rows),
    }
    # v4 theta values in the prompt are rounded to four decimal places.
    rounding_tolerance = 0.00005 + 1e-9
    checks = {
        "global_frozen_counts": all(counts[k] == v for k, v in EXPECTED.items()),
        "v1_full_window_low_high": (
            math.isclose(v1_fit["low"], EXPECTED_LOW, rel_tol=0, abs_tol=1e-10)
            and math.isclose(v1_fit["high"], EXPECTED_HIGH, rel_tol=0, abs_tol=1e-10)
        ),
        "all_person_v4_background_counts": all(
            r["backgrounds"] == r["expected_backgrounds"] for r in references
        ),
        "all_person_rounded_v4_thetas": all(
            abs(r[f"theta_k{k}_difference_from_rounded_g"]) <= rounding_tolerance
            for r in references for k in (3, 5, 8)
        ),
        "all_person_reference_windows_ge_20": all(
            r["windows"] >= 20 for r in references
        ),
        "all_person_fit_and_validation_windows_present": all(
            r["fit"] > 0 and r["validation"] > 0 for r in references
        ),
        "all_person_background_parities_present": all(
            any(b["meta"]["split"] == split for b in person["backgrounds"])
            for person in data for split in ("fit", "validation")
        ),
    }
    report = {
        "status": "INSPECTION_COMPLETE" if all(checks.values()) else "REPRODUCTION_FAILED",
        "counts": counts,
        "checks": checks,
        "v1_noise_fit": v1_fit,
        "theta_comparison_absolute_tolerance_g": rounding_tolerance,
        "thresholds_selected": False,
        "real_movement_onsets_computed": False,
        "forecast_labels_computed": False,
        "provider_access": False,
        "held_out_access": False,
        "physical_pre_start_only_io": False,
        "additional_analysis_restricted_to_declared_pre_start_spans": True,
        "per_person_count_comparison": (
            "Compare noise_reference.csv fit/validation/windows with the v1 "
            "settings. Those per-person constants were not included in the prompt."
        ),
        "elapsed_s": time.monotonic() - started,
        "output_dir": str(output),
    }

    outputs = {
        "run_counts.csv": run_rows,
        "window_diagnostics.csv": window_rows,
        "background_inventory.csv": background_rows,
        "distributions.csv": distributions,
        "level_fractions.csv": levels,
        "tail_attribution.csv": tails,
        "noise_reference.csv": references,
        "threshold_probes.csv": thresholds,
        "window_crossings.csv": crosses,
        "background_replay.csv": bgs,
        "plateau_feasibility.csv": feasibility_table(thresholds, references),
    }
    for name, rows in outputs.items():
        write_csv(output / name, rows)
    write_json(output / "inspection_summary.json", report)

    lines = compact_summary(
        data, distributions, tails, thresholds, crosses, bgs, report
    )
    with (output / "pasteback.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
