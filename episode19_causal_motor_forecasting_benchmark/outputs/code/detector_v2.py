"""EP19 detector v2: noise-only calibration, closures, and generic D2 replay."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import detector_freeze as F
import detector_readers as R
import detector_summaries as S
import run_detector_freeze as RF
import inspect_detector_v2_noise as I
import self_paced_raw_onset_test as V4

PEOPLE = tuple(R.SP_DEVELOPMENT)
MODELS = ("D2", "D1")
ARRAY_NAMES = ("tick", "state", "cue", "category", "observed_bins", "reason")
CUE_FIELDS = (
    "source", "participant", "session", "run", "cue", "cue_tick",
    "start_known_tick", "end_tick", "stop_reason", "support",
    "support_reasons", "onset_tick", "known_tick", "rt_ms",
    "confirmation_ms", "departure_ms", "rejected_crossings",
    "unresolved_at_end",
)
SOURCE_FILES = (
    "detector_v2.py", "run_detector_v2.py", "self_paced_raw_onset_test_v5.py",
    "detector_freeze.py", "run_detector_freeze.py", "detector_readers.py",
    "detector_summaries.py", "forecast_labels.py",
    "classical_forecast_backend.py", "self_paced_raw_onset_test.py",
    "inspect_detector_v2_noise.py",
)
PROTECTED_NAMES = {
    "detector_freeze_job_48219799", "raw_onset_test_job_48230171",
}


def require_slurm(workers=1):
    return I.require_slurm(workers)


def fresh(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError(f"Output symlink refused: {path}")
    path = path.resolve()
    if PROTECTED_NAMES.intersection(path.parts):
        raise RuntimeError("Outputs may not enter a preserved v1/v4 directory")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Nonempty output refused: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def text_file(path, text):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path, value):
    I.write_json(path, value)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_npz(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {name: digest(CODE / name) for name in SOURCE_FILES}


def artifact_hashes(folder):
    folder = Path(folder)
    return {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p != folder / "closed.json"
    }


def close(folder, stage, **metadata):
    folder = Path(folder)
    write_json(folder / "closed.json", {
        "stage": stage, "complete": True, "metadata": metadata,
        "files": artifact_hashes(folder), "sources": source_hashes(),
    })


def require_closed(folder, stage):
    folder = Path(folder)
    record = read_json(folder / "closed.json")
    if record.get("stage") != stage or record.get("complete") is not True:
        raise RuntimeError(f"Wrong/incomplete closure: {folder}")
    if record["files"] != artifact_hashes(folder):
        raise RuntimeError(f"Closed artifacts changed: {folder}")
    if record["sources"] != source_hashes():
        raise RuntimeError("Code changed after stage closure; use a fresh run")
    return record["metadata"]


def pool_map(function, tasks, workers):
    require_slurm(workers)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        return list(pool.map(function, tasks))


def participants(text):
    if text is None:
        return list(PEOPLE)
    requested = [p.strip() for p in text.split(",") if p.strip()]
    if (not requested or len(set(requested)) != len(requested)
            or set(requested) - set(PEOPLE)):
        raise ValueError("Require unique development participant IDs")
    return [p for p in PEOPLE if p in requested]


def scope(args):
    return {
        "participants": participants(args.participants),
        "participant_list": args.participants is not None,
        "health_only": bool(args.health_only),
    }


def check_scope(metadata, requested):
    for key in ("participants", "participant_list", "health_only"):
        if metadata.get(key) != requested[key]:
            raise RuntimeError(f"Stage scope mismatch: {key}")


def stems_by_person():
    I.check_imported_contract()
    stems = sorted(R.development_stems())
    groups = {p: [] for p in PEOPLE}
    seen = set()
    for stem in stems:
        key = I.stem_identity(stem)
        if key in seen:
            raise RuntimeError("Duplicate development run")
        seen.add(key)
        groups[key[0]].append(stem)
    if len(stems) != 155 or any(not groups[p] for p in PEOPLE):
        raise RuntimeError("Development inventory must contain exactly 155 runs")
    return groups


def event_key(row):
    return V4.event_key(row)


def run_segment_d2(values, low, high, confirm, quiet, starts, cuts,
                   evidence_at=None):
    """F.run_segment interface; pending_at_stop maps path to LOW crossing.

    Snapshots are pre-adjudication facts. HIGH clocks remain internal;
    externally visible pending and unknown bounds always use LOW crossings.
    """
    values = np.asarray(values)
    low, high = np.asarray(low), np.asarray(high)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("D2 requires finite (sample,path) values")
    m, npath = values.shape
    if low.shape != (npath,) or high.shape != (npath,):
        raise ValueError("D2 threshold shape mismatch")
    if not np.all(np.isfinite(low) & np.isfinite(high) & (high >= low)):
        raise ValueError("Invalid D2 thresholds")
    state = np.full(m, F.STATE_CODES["unarmed"], np.int8)
    cue_of = np.full(m, -1, np.int32)
    cues, snapshots = [], {}
    cue = None
    evidence_at = evidence_at or {}
    wanted = set(evidence_at.values())

    def snapshot():
        return (
            dict(cue["pending"]), list(cue["accepted"]), cue["onset"],
            cue["surveillance"], list(cue["rejected"]),
        )

    def finish(i, reason):
        nonlocal cue
        if cue is None:
            return
        snap = snapshots.pop(evidence_at[i], None) if i in evidence_at else None
        pending, accepted, onset, surveillance, rejected = (
            snapshot() if snap is None else snap
        )
        if not surveillance:
            unknown = cue["start"]
        elif onset is None and pending:
            unknown = min(pending.values())
            if accepted:
                unknown = min(unknown, min(c for c, _, _ in accepted))
        else:
            unknown = None
        path = min(accepted)[1] if onset is not None else None
        cues.append(F.CueResult(
            cue["start"], i, reason, cue["status"], onset, path,
            accepted, rejected, pending, unknown,
        ))
        cue = None

    for i in range(m):
        if i in cuts:
            finish(i, "cut")
        if i in starts:
            finish(i, "restart")
            status = starts[i]
            cue = {
                "start": i, "status": status,
                "surveillance": status == "supported",
                "armed": status == "supported",
                "requalifying": False, "quiet": 0,
                "pending": {}, "high_at": {}, "accepted": [],
                "onset": None, "rejected": [],
            }
        if cue is None:
            continue
        if i in wanted:
            snapshots[i] = snapshot()
        cue_of[i] = len(cues)
        if not cue["surveillance"]:
            state[i] = F.STATE_CODES[
                "unknown" if cue["status"] == "unknown" else "unarmed"
            ]
            continue

        v = values[i]
        if (cue["requalifying"] and not cue["pending"]
                and not cue["accepted"] and cue["quiet"] >= quiet):
            cue["armed"], cue["requalifying"] = True, False
        if not cue["accepted"]:
            for p in range(npath):
                if p not in cue["pending"] and v[p] >= low[p]:
                    cue["pending"][p] = i
                    cue["high_at"][p] = None
                    cue["armed"] = False
                    cue["requalifying"] = True
                    cue["quiet"] = 0

        had_pending = bool(cue["pending"])
        for p, crossing in list(cue["pending"].items()):
            if v[p] < low[p]:
                cue["rejected"].append((crossing, p, i))
                del cue["pending"][p]
                del cue["high_at"][p]
                continue
            if cue["high_at"][p] is None and v[p] >= high[p]:
                cue["high_at"][p] = i
            high_at = cue["high_at"][p]
            if high_at is not None and i - high_at >= confirm:
                cue["accepted"].append((crossing, p, i))
                del cue["pending"][p]
                del cue["high_at"][p]

        if cue["accepted"]:
            crossing = min(a[0] for a in cue["accepted"])
            if (cue["onset"] is None
                    and not any(c <= crossing for c in cue["pending"].values())):
                cue["onset"] = (crossing, i)
            cue["armed"] = False
            cue["quiet"] = 0
        elif cue["pending"]:
            cue["quiet"] = 0
        elif cue["requalifying"]:
            if had_pending:
                cue["quiet"] = 0
            cue["quiet"] = cue["quiet"] + 1 if np.all(v <= low) else 0

        if cue["accepted"]:
            state[i] = F.STATE_CODES[
                "unarmed" if cue["onset"] is not None else "pending"
            ]
        else:
            state[i] = F.STATE_CODES["at_risk" if cue["armed"] else "pending"]

    finish(m, "segment_end")
    return state, cue_of, cues


def check_d2_contract():
    low, high = np.array([1.0]), np.array([2.0])
    x = np.array([0, 1.1, 1.2, 2.1, 1.1, 1.1, 0])[:, None]
    state, _, cues = run_segment_d2(x, low, high, 2, 3,
                                    {0: "supported"}, set())
    assert cues[0].onset == (1, 5)
    assert np.all(state[1:5] == F.STATE_CODES["pending"])
    _, _, cues = run_segment_d2(x, low, high, 2, 3,
                                {0: "supported"}, {6}, {6: 4})
    assert cues[0].onset is None and cues[0].unknown_from == 1
    y = np.array([0, 1.1, 0.5, 2.1, 1.1, 1.1])[:, None]
    _, _, cues = run_segment_d2(y, low, high, 2, 3,
                                {0: "supported"}, set())
    assert cues[0].onset == (3, 5) and len(cues[0].rejected) == 1
    y = np.array([0, 1.1, 0, 0, 0, 0])[:, None]
    state, _, _ = run_segment_d2(y, low, high, 2, 3,
                                 {0: "supported"}, set())
    assert state[4] == F.STATE_CODES["pending"]
    assert state[5] == F.STATE_CODES["at_risk"]
    y = np.array([[0, 0], [1.1, 0], [1.1, 2.1],
                  [1.1, 1.1], [1.1, 1.1], [0, 1.1]])
    _, _, cues = run_segment_d2(y, np.ones(2), np.full(2, 2.0),
                                2, 3, {0: "supported"}, set())
    assert cues[0].onset == (2, 5)
    return 5


def noise_probe(task):
    data, thresholds = task
    require_slurm(1)
    result = I.probe_person(task)
    # Verify that inspection availability uses the implemented generic D2.
    for threshold in thresholds:
        low, high = threshold["low_g"], threshold["high_g"]
        for background in data["backgrounds"]:
            x = background["x"]
            supported = bool(np.all(x[:125] <= low))
            expected = I.d2_noise_replay(x[125:], low, high, supported)
            state, _, cues = run_segment_d2(
                x[125:, None], np.array([low]), np.array([high]),
                F.SP_CONFIRM, F.SP_QUIET,
                {0: "supported" if supported else "not_supported"}, set(),
            )
            if (not np.array_equal(state, expected["states"])
                    or int(cues[0].onset is not None) != expected["confirmed"]):
                raise RuntimeError("Generic D2 differs from noise replay")
    return result


def fit_settings(folder, workers):
    """All fifteen people; no event table, labels or provider inputs."""
    groups = stems_by_person()
    data = pool_map(I.inspect_person, [(p, groups[p]) for p in PEOPLE], workers)
    windows = [w for person in data for w in person["windows"]]
    runs = [r for person in data for r in person["runs"]]
    counts = {
        "runs": len(runs),
        "start_tones": sum(r["start_tones"] for r in runs),
        "windows": len(windows),
        "fit": sum(w["meta"]["split"] == "fit" for w in windows),
        "validation": sum(w["meta"]["split"] == "validation" for w in windows),
        "backgrounds": sum(len(p["backgrounds"]) for p in data),
    }
    if any(counts[k] != v for k, v in I.EXPECTED.items()):
        raise RuntimeError(f"Frozen noise count mismatch: {counts}")
    if counts["backgrounds"] != 2545:
        raise RuntimeError("Expected 2545 eligible backgrounds")

    fit_full = [w["x"][:, None] for w in windows
                if w["meta"]["split"] == "fit"]
    fit_tail = [w[130:] for w in fit_full]
    original = F.fit_thresholds(fit_full, F.SP_CONFIRM)[0]
    if not (math.isclose(original.low, V4.EXPECTED_LOW, abs_tol=1e-10, rel_tol=0)
            and math.isclose(original.high, V4.EXPECTED_HIGH,
                             abs_tol=1e-10, rel_tol=0)):
        raise RuntimeError("Full-window v1 thresholds did not reproduce")
    fitted = F.fit_thresholds(fit_tail, F.SP_CONFIRM)[0]
    if not fitted.resolved:
        raise RuntimeError("Corrected-support quantile scale is unusable")

    thresholds = [{
        "threshold_id": f"quantile_c{c:g}", "scale": "quantile", "rung": c,
        "median_g": fitted.median, "scale_g": fitted.sigma_q,
        "low_g": fitted.median + c * fitted.sigma_q,
        "high_g": fitted.median + 2 * c * fitted.sigma_q,
    } for c in F.RUNGS]

    probes = pool_map(noise_probe, [(p, thresholds) for p in data], workers)
    cross = [r for result in probes for r in result[0]]
    background = [r for result in probes for r in result[1]]
    cross = [I.finish_cross(r) for r in I.add_pooled(
        cross, ("threshold_id", "split"), I.CROSS_COUNTS)]
    background = [I.finish_background(r) for r in I.add_pooled(
        background, ("threshold_id", "model", "split"), I.BG_COUNTS)]
    cx = {(r["participant"], r["split"], r["threshold_id"]): r for r in cross}
    bx = {(r["participant"], r["split"], r["threshold_id"], r["model"]): r
          for r in background}

    history, chosen, final_guards = [], thresholds[-1], {}
    for threshold in thresholds:
        c, tid = threshold["rung"], threshold["threshold_id"]
        if c < fitted.rung:
            continue
        pooled = cx[("POOL", "fit", tid)]
        pooled_ok = pooled["v1_hit_windows"] <= max(
            1, math.floor(.01 * pooled["windows"]))
        guards = {}
        for model in MODELS:
            causes = []
            if not pooled_ok:
                causes.append("pooled_fit_crossing_criterion")
            for person in PEOPLE:
                row = cx[(person, "fit", tid)]
                if (not row["windows"] or row["v1_hit_windows"]
                        > max(1, math.floor(.05 * row["windows"]))):
                    causes.append(f"{person}:fit_crossing_ceiling")
            pool_bg = bx[("POOL", "fit", tid, model)]
            for field in ("fixed_entry_fraction", "at_risk_grid_fraction"):
                if pool_bg[field] is None or pool_bg[field] < .90:
                    causes.append(f"pooled:{field}")
            for person in PEOPLE:
                risk = bx[(person, "fit", tid, model)]["at_risk_grid_fraction"]
                if risk is None or risk < .75:
                    causes.append(f"{person}:at_risk_floor")
            guards[model] = {"eligible": not causes, "failed_guards": causes}
        history.append({"rung": c, "pooled_criterion": pooled_ok,
                        "candidates": guards})
        chosen, final_guards = threshold, guards
        if all(g["eligible"] for g in guards.values()):
            break

    references = []
    for person in data:
        p = person["participant"]
        native = I.concatenate([w["native_r"] for w in person["windows"]])
        median = float(np.median(native))
        mad = float(np.median(np.abs(native - median)))
        sigma = 1.4826 * mad
        if len(person["backgrounds"]) != I.EXPECTED_BACKGROUND_COUNTS[p]:
            raise RuntimeError(f"Background count mismatch: {p}")
        reference = {
            "participant": p, "fs": person["fs"],
            "runs": len(person["runs"]),
            "start_tones": sum(r["start_tones"] for r in person["runs"]),
            "still_windows": len(person["windows"]),
            "fit_windows": sum(w["meta"]["split"] == "fit"
                               for w in person["windows"]),
            "validation_windows": sum(w["meta"]["split"] == "validation"
                                      for w in person["windows"]),
            "control_eligible": len(person["backgrounds"]),
            "reference_samples": len(native), "median_g": median,
            "mad_g": mad, "sigma_g": sigma,
            "usable": len(person["windows"]) >= 20 and sigma > 0,
        }
        for k, expected in zip(V4.KS, I.EXPECTED_THETAS[p]):
            reference[f"theta_k{k}_g"] = median + k * sigma
            if abs(median + k * sigma - expected) > .000050001:
                raise RuntimeError(f"Native reference mismatch: {p}, k={k}")
        if not reference["usable"]:
            raise RuntimeError(f"Unusable native reference: {p}")
        references.append(reference)

    settings = {
        "counts": counts, "participants": PEOPLE,
        "corrected_support": "[130,250)",
        "base_v1_rule_fit": fitted.__dict__,
        "v1_full_window_reproduction": original.__dict__,
        "shared_threshold": chosen, "guard_history": history,
        "shared_rung_escalation_is_an_assumption": True,
        "validation_used_for_selection": False,
        "candidates": {
            model: {**chosen, **final_guards[model]} for model in MODELS
        },
        "references": references,
        "provider_access": False,
    }
    folder = Path(folder)
    write_json(folder / "detector_settings.json", settings)
    write_csv(folder / "run_counts.csv", runs)
    write_csv(folder / "window_inventory.csv", [w["meta"] for w in windows])
    write_csv(folder / "background_inventory.csv",
              [b["meta"] for p in data for b in p["backgrounds"]])
    write_csv(folder / "window_crossings.csv", cross)
    write_csv(folder / "background_replay.csv", background)
    write_csv(folder / "thresholds.csv", thresholds)
    write_csv(folder / "noise_reference.csv", references)
    save_npz(
        folder / "noise_samples.npz",
        fit_full=np.concatenate(fit_full)[:, 0],
        fit_corrected=np.concatenate(fit_tail)[:, 0],
    )
    return settings
