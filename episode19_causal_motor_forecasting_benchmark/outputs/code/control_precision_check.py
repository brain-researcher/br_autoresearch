#!/usr/bin/env python3
"""EP19 control precision; fixed v5 gates and descriptive onset-slope strata."""
from __future__ import annotations

import os
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import csv
import gzip
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import premature_r_diagnostic as P
from premature_r_diagnostic import D, V4, V5, RF

DECLARATION = P.DECLARATION
DEV = (
    "dev_linear_q10", "dev_linear_q50",
    "dev_quadratic_q10", "dev_quadratic_q50",
)
ALL_FAMILIES = V5.FAMILIES + DEV
CURVE_TARGETS = tuple(t for t in V5.CORE if 272 <= t <= 320)
REPORTED_V5 = {
    "sub-02": {
        "sham_alarm_fraction_rounded": 0.070,
        "unanchorable_backgrounds": 2,
        "retained_as_reported_v5_result": True,
    }
}


def read_cells(path):
    rows = D.read_csv(path)
    boolean = {"core", "gated_family"} | {f"truth_ok_k{k}" for k in V5.KS}
    integer = {
        "target_detectable_lag_ms", "T0_tick", "S_tick", "search_start_tick",
        "L_tick", "growth_stop_tick",
    } | {f"T_det_k{k}_tick" for k in V5.KS}
    for row in rows:
        for key in boolean:
            if row[key] not in ("True", "False"):
                raise RuntimeError(f"Invalid closed boolean: {key}")
            row[key] = row[key] == "True"
        for key in integer:
            row[key] = "" if row[key] == "" else V4.required_int(row[key])
        row["fs"] = float(row["fs"])
    return rows


def background_key(row):
    return row["session"], int(row["run"]), int(row["start_tone_tick"])


def inventories(people, *, expanded):
    inventory = D.read_csv(P.DETECTOR / "settings/background_inventory.csv")
    selected = D.read_csv(P.V5RUN / "calibration/selected_backgrounds.csv")
    if len(inventory) != 2545 or len(selected) != 480:
        raise RuntimeError("Closed background inventory count differs")
    if {r["participant"] for r in inventory} != set(D.PEOPLE):
        raise RuntimeError("Background person inventory differs")
    output = {}
    for p in people:
        full = [r for r in inventory if r["participant"] == p]
        lookup = {background_key(r): r for r in full}
        if len(lookup) != len(full):
            raise RuntimeError("Duplicate background inventory key")
        first = sorted(
            [r for r in selected if r["participant"] == p],
            key=lambda r: int(r["background_id"]))
        if [int(r["background_id"]) for r in first] != list(range(32)):
            raise RuntimeError("Selected IDs must be exactly 0..31")
        used, ordered = set(), []
        for row in first:
            key = background_key(row)
            if key not in lookup or key in used:
                raise RuntimeError("Selected background missing/duplicated")
            source = lookup[key]
            for field in source:
                if str(source[field]) != str(row[field]):
                    raise RuntimeError("Selected/inventory provenance differs")
            used.add(key)
            ordered.append(dict(source))
        if expanded:
            ordered.extend(
                dict(r) for r in sorted(full, key=background_key)
                if background_key(r) not in used)
        output[p] = [
            {**row, "background_id": bid} for bid, row in enumerate(ordered)]
    return output


def cache_all(task):
    """Supplied v5 loader logic with ordered inventory and no reservoir cap."""
    p, stems, inventory, folder = task
    D.require_slurm(1)
    by_run = defaultdict(list)
    for bid, item in enumerate(inventory):
        by_run[(item["session"], int(item["run"]))].append((bid, item))
    segments = [None] * len(inventory)
    fs_expected = 1000.0 if p == "sub-15" else 250.0
    for stem in stems:
        _, session, run_number = V4.stem_key(stem)
        needed = by_run.get((session, run_number), [])
        if not needed:
            continue
        run, _, _, _, valid, _, _ = RF.sp_load(stem)
        if run.fs != fs_expected:
            raise RuntimeError("Background native rate mismatch")
        factor = int(run.fs / 250)
        for bid, item in needed:
            first = int(item["background_start_tick"])
            stop = int(item["background_stop_tick"])
            a, b = first // 4, stop // 4
            if first % 4 or stop - first != 2800 or not np.all(valid[a:b]):
                raise RuntimeError("Closed background support differs")
            section = run.acc_g[a * factor:b * factor].copy()
            if len(section) != int(2.8 * run.fs) or not np.all(np.isfinite(section)):
                raise RuntimeError("Invalid native background")
            segments[bid] = section
    if any(segment is None for segment in segments):
        raise RuntimeError("Background lacks its native run")
    D.save_npz(
        Path(folder) / f"{p}_backgrounds.npz",
        acc=np.stack(segments), fs=np.asarray(fs_expected))
    return {"participant": p, "backgrounds": len(segments)}


def split_control_rows(reference, calibration, inventory, template_folder,
                       background_folder, families, *, limit=None):
    """Loading wrapper for the sole noisy-control implementation."""
    p = reference["participant"]
    with np.load(Path(template_folder) / f"{p}_templates.npz",
                 allow_pickle=False) as saved:
        templates = {family: saved[family].copy() for family in families}
    with np.load(Path(background_folder) / f"{p}_backgrounds.npz",
                 allow_pickle=False) as saved:
        fs, backgrounds = float(saved["fs"]), saved["acc"]
    if fs != float(reference["fs"]) or len(backgrounds) != len(inventory):
        raise RuntimeError("Cache inventory mismatch")
    yield from array_control_rows(
        reference, calibration, inventory, templates, backgrounds, families, limit)


def array_control_rows(reference, calibration, inventory, templates,
                       backgrounds, families, limit=None):
    """Used by both regressions and every expanded/health control output."""
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    pid = int(p.split("-")[1])
    rngs = {
        2: np.random.default_rng(np.random.SeedSequence([V4.SEED, pid, 2])),
        4: np.random.default_rng(np.random.SeedSequence([V4.SEED, pid, 4])),
    }
    cells = {f: [c for c in calibration if c["family"] == f] for f in families}
    count = len(backgrounds) if limit is None else min(limit, len(backgrounds))
    for bid in range(count):
        item, background = inventory[bid], backgrounds[bid]
        if int(item["background_id"]) != bid:
            raise RuntimeError("Background ordering changed")
        provenance = {
            "participant": p, "fs": fs, "background_id": bid,
            "background_session": item["session"],
            "background_run": int(item["run"]),
            "background_start_tick": int(item["background_start_tick"]),
            "background_end_tick": int(item["background_stop_tick"]),
        }
        raw, ok, _, _, _ = V4.control_features(background, fs)
        for position in V5.SHAMS:
            L = position + (3 if fs == 1000 else 0)
            for k in V5.KS:
                est = V5.estimate(
                    raw, ok, fs, L, 1920, V4.reference_theta(reference, k))
                yield {
                    **provenance, "kind": "sham", "family": "sham",
                    "core": False, "k": k, "L_tick": L, "S_tick": 1920,
                    "control_status": "sham", **V4.estimate_control_fields(est),
                    "sham_alarm": int(
                        est["status"] in ("determinate", "no_valid_data")
                        or est["reason"] == "missing_200ms_quiet_anchor_left_censored"),
                    "v4_any_departure_sham_alarm": int(
                        est["status"] == "no_valid_data"
                        or est["any_qualifying_before_L"] == 1
                        or est["initial_quiet_40ms"] != 1),
                }
        for family in families:
            direction = rngs[2 if family in V5.FAMILIES else 4].normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm == 0:
                raise RuntimeError("Zero random direction")
            direction /= norm
            injected = background + templates[family][:, None] * direction
            raw, ok, _, _, _ = V4.control_features(injected, fs)
            for cell in cells[family]:
                for k in V5.KS:
                    td = cell[f"T_det_k{k}_tick"]
                    L = "" if cell["L_tick"] == "" else int(cell["L_tick"])
                    if cell["calibration_status"] != "calibrated":
                        status = "uncovered_template"
                        est = V4.missing_control_estimate(cell["calibration_reason"])
                    elif td == "":
                        status = "undefined_noiseless_truth"
                        est = V4.missing_control_estimate(status)
                    elif not cell[f"truth_ok_k{k}"]:
                        status = "truth_not_associated"
                        est = V4.missing_control_estimate(status)
                    else:
                        est = V5.estimate(
                            raw, ok, fs, L, 1920, V4.reference_theta(reference, k))
                        status = (
                            "invalid_search_support"
                            if est["status"] == "no_valid_data" else "ok")
                    yield {
                        **provenance, "kind": "injected", "family": family,
                        "core": cell["core"], "gated_family": family in V5.GATED,
                        "target_detectable_lag_ms": cell["target_detectable_lag_ms"],
                        "calibration_status": cell["calibration_status"],
                        "calibration_reason": cell["calibration_reason"],
                        "k": k, "L_tick": L, "S_tick": 1920,
                        "T0_tick": cell["T0_tick"], "T_det_tick": td,
                        "true_lag_ms": "" if td == "" or L == "" else L - int(td),
                        "control_status": status,
                        "direction_x": float(direction[0]),
                        "direction_y": float(direction[1]),
                        "direction_z": float(direction[2]),
                        **V4.estimate_control_fields(est),
                    }


def trial_key(row):
    return (
        int(row["background_id"]), row["kind"], row["family"],
        int(row["k"]), V4.required_int(row["L_tick"]),
    )


def exact_field(a, b):
    a, b = "" if a is None else str(a), "" if b is None else str(b)
    if a == b:
        return True
    if not a or not b or a in ("True", "False") or b in ("True", "False"):
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def first32_check(task):
    reference, calibration, inventory, template_folder, background_folder = task
    D.require_slurm(1)
    p = reference["participant"]
    old = {}
    with (P.V5RUN / "controls/positive_control.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        for row in reader:
            if row["participant"] == p:
                key = trial_key(row)
                if key in old:
                    raise RuntimeError("Duplicate original control key")
                old[key] = row
    expected = 32 * (
        len(V5.SHAMS) * len(V5.KS)
        + len(V5.FAMILIES) * len(V5.TARGETS) * len(V5.KS))
    if len(old) != expected:
        raise RuntimeError("Original per-person control row count differs")
    seen = set()
    for row in split_control_rows(
        reference, calibration, inventory, template_folder, background_folder,
        V5.FAMILIES, limit=32
    ):
        key = trial_key(row)
        if key not in old or key in seen or set(row) - set(fields):
            raise RuntimeError(f"First-32 regression key/field mismatch: {p}")
        seen.add(key)
        for field in fields:
            if not exact_field(old[key][field], row.get(field, "")):
                raise RuntimeError(f"First-32 regression mismatch: {p}, field={field}")
    if seen != set(old):
        raise RuntimeError("First-32 regression omitted rows")
    return {"participant": p, "pass": True, "rows": len(seen), "fields": len(fields)}


def fit_development(reference, family, target_g):
    """Nearest searched eligible slope; no monotonicity assumption for g(A)."""
    fs = float(reference["fs"])
    theta = V4.reference_theta(reference, 5)
    dt = int(round(1000 / fs))
    ticks = np.arange(2800 // dt) * dt
    availability_phase = 3 if fs == 1000 else 0
    power = 2 if "quadratic" in family else 1
    attempts, phase_best = [], []
    base_info = {
        "target_rise_per_s": target_g,
        "achieved_rise_per_s": None, "relative_error": None,
        "grid_amplitudes_per_phase": 4000, "local_refinements": 30,
        "positive_bracket_floor_ratio": 2.0 ** -40,
    }
    if not P.finite(target_g) or target_g <= 0:
        return None, {
            **base_info, "reason": "nonpositive_or_missing_development_slope",
            "phase_attempts": attempts,
        }
    target_g = float(target_g)

    for phi in ((0,) if fs == 250 else range(4)):
        T0 = 1920 + phi
        unit = np.maximum((ticks - T0) / 1000, 0) ** power

        def measure(amplitude):
            signal = amplitude * unit
            raw = np.abs(V5.scalar_residual(signal, fs))
            td = V5.first_detectable(raw, fs, theta, T0)
            if td is None:
                return None, None
            g = P.rise_rate(raw, np.isfinite(raw), fs, td, theta)
            return g, td

        upper = theta
        bracketed = False
        for _ in range(40):
            g, _ = measure(upper)
            if P.finite(g) and g >= target_g:
                bracketed = True
                break
            upper *= 2
        if not bracketed:
            attempts.append({"phase": phi, "status": "not_bracketed"})
            continue
        lower = upper * 2.0 ** -40

        def candidate(amplitude):
            amplitude = float(amplitude)
            g, td = measure(amplitude)
            if not P.finite(g) or td is None or td % 4 != availability_phase:
                return None
            cap = td + 700
            if cap > int(ticks[-1]) or cap % dt:
                return None
            signal = amplitude * unit
            signal[ticks > cap] = signal[cap // dt]
            raw = np.abs(V5.scalar_residual(signal, fs))
            if V5.first_detectable(raw, fs, theta, T0) != td:
                return None
            achieved = P.rise_rate(raw, np.isfinite(raw), fs, td, theta)
            if not P.finite(achieved):
                return None
            return {
                "T0_tick": T0, "T_det_k5_tick": td, "growth_stop_tick": cap,
                "amplitude": amplitude, "amplitude_units": f"g/s^{power}",
                "target_rise_per_s": target_g,
                "achieved_rise_per_s": achieved,
                "relative_error": abs(achieved - target_g) / target_g,
                "chosen_phase_ms": phi,
            }

        def score(item):
            return (
                abs(item["achieved_rise_per_s"] - target_g),
                item["amplitude"], item["chosen_phase_ms"],
            )

        grid = np.geomspace(lower, upper, 4000)
        best, best_index = None, None
        for index, amplitude in enumerate(grid):
            item = candidate(amplitude)
            if item is not None and (best is None or score(item) < score(best)):
                best, best_index = item, index
        if best is None:
            attempts.append({
                "phase": phi, "status": "no_phase_and_cap_valid_candidate",
                "bracket": [lower, upper],
            })
            continue

        left = float(grid[max(0, best_index - 1)])
        right = float(grid[min(len(grid) - 1, best_index + 1)])
        # Bisect both local sides; retain the best eligible point, not a root.
        for _ in range(30):
            center = best["amplitude"]
            points = sorted(set((
                left, (left + center) / 2, center,
                (center + right) / 2, right,
            )))
            evaluated = [best]
            for amplitude in points:
                if amplitude != center:
                    item = candidate(amplitude)
                    if item is not None:
                        evaluated.append(item)
            best = min(evaluated, key=score)
            center = best["amplitude"]
            index = points.index(center)
            left = points[max(0, index - 1)]
            right = points[min(len(points) - 1, index + 1)]
        phase_best.append(best)
        attempts.append({
            "phase": phi, "status": "searched",
            "bracket": [lower, upper],
            "best_amplitude": best["amplitude"],
            "achieved_rise_per_s": best["achieved_rise_per_s"],
            "relative_error": best["relative_error"],
        })

    if not phase_best:
        return None, {
            **base_info, "reason": "slope_unattainable_within_5pct",
            "phase_attempts": attempts,
        }
    best = min(phase_best, key=lambda item: (
        abs(item["achieved_rise_per_s"] - target_g),
        item["amplitude"], item["chosen_phase_ms"]))
    info = {**base_info, **best, "phase_attempts": attempts}
    if best["relative_error"] > .05:
        return None, {**info, "reason": "slope_unattainable_within_5pct"}
    unit = np.maximum((ticks - best["T0_tick"]) / 1000, 0) ** power
    signal = best["amplitude"] * unit
    cap = best["growth_stop_tick"]
    signal[ticks > cap] = signal[cap // dt]
    return signal, info


def development_cells(reference, family, signal, info):
    fs, p = float(reference["fs"]), reference["participant"]
    dt = int(round(1000 / fs))
    last_tick = 2800 - dt
    phase = 3 if fs == 1000 else 0
    if signal is None:
        return [{
            "participant": p, "fs": fs, "family": family,
            "gated_family": False, "core": target in V5.CORE,
            "target_detectable_lag_ms": target,
            "T0_tick": "", "S_tick": 1920, "search_start_tick": 1420,
            "L_tick": "", "growth_stop_tick": "",
            "calibration_status": "uncovered",
            "calibration_reason": info["reason"],
            **{f"T_det_k{k}_tick": "" for k in V5.KS},
            **{f"truth_ok_k{k}": False for k in V5.KS},
        } for target in V5.TARGETS]
    T0, detected = info["T0_tick"], info["T_det_k5_tick"]
    raw = np.abs(V5.scalar_residual(signal, fs))
    ok = np.isfinite(raw)
    detected_by_k = {
        k: V5.first_detectable(raw, fs, V4.reference_theta(reference, k), T0)
        for k in V5.KS
    }
    output = []
    for target in V5.TARGETS:
        L = detected + target
        reasons = []
        if L % 4 != phase:
            reasons.append("exact_target_off_detector_availability_clock")
        if L < 1920 or L + 100 > last_tick:
            reasons.append("insufficient_declared_temporal_support")
        truths = {}
        for k in V5.KS:
            td = detected_by_k[k]
            truth_ok = False
            if td is not None and not reasons:
                est = V5.estimate(
                    raw, ok, fs, L, 1920, V4.reference_theta(reference, k))
                truth_ok = est["status"] == "determinate" and est["R_tick"] == td
            truths[f"T_det_k{k}_tick"] = "" if td is None else td
            truths[f"truth_ok_k{k}"] = bool(truth_ok)
            if k == 5 and not reasons and not truth_ok:
                reasons.append("noiseless_association_or_anchor_failed")
        if L - detected != target:
            raise RuntimeError("Exact target arithmetic differs")
        output.append({
            "participant": p, "fs": fs, "family": family,
            "gated_family": False, "core": target in V5.CORE,
            "target_detectable_lag_ms": target,
            "T0_tick": T0, "S_tick": 1920, "search_start_tick": 1420,
            "L_tick": L, "growth_stop_tick": info["growth_stop_tick"],
            "calibration_status": "uncovered" if reasons else "calibrated",
            "calibration_reason": ";".join(reasons) if reasons
            else "exact_phase_support_anchor_and_association_verified",
            **truths,
        })
    return output


def strata_worker(task):
    reference, slopes, folder = task
    D.require_slurm(1)
    p, fs = reference["participant"], float(reference["fs"])
    with np.load(P.V5RUN / "calibration" / f"{p}_templates.npz",
                 allow_pickle=False) as saved:
        arrays = {family: saved[family].copy() for family in V5.FAMILIES}
    cells, parameters = [], []
    for family in DEV:
        quantile = "q10" if family.endswith("q10") else "q50"
        signal, info = fit_development(reference, family, slopes[quantile])
        cells.extend(development_cells(reference, family, signal, info))
        arrays[family] = np.zeros(int(2.8 * fs)) if signal is None else signal
        parameters.append({
            "participant": p, "family": family, "descriptive_only": True,
            "calibrated": signal is not None, **info,
        })
    D.save_npz(Path(folder) / f"{p}_templates.npz", **arrays)
    return cells, parameters


def strata_stage(args):
    root = P.output_root(args, "control_precision", first=True)
    prd = P.check_stage(args.prd_dir / "features", "prd_features", args)
    P.check_stage(args.prd_dir / "known_positive", "prd_known_positive", args)
    if prd["known_positive_sha256"] != D.digest(
        args.prd_dir / "known_positive/closed.json"
    ):
        raise RuntimeError("PRD positive-control predecessor changed")
    folder = D.fresh(root / "strata")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    refs, _, _, _, _ = P.inputs()
    people = P.scope(args)["participants"]
    refs = [r for r in refs if r["participant"] in people]
    original = read_cells(P.V5RUN / "calibration/control_templates.csv")
    inventory = inventories(people, expanded=not args.health_only)

    if args.health_only:
        template_folder = background_folder = P.V5RUN / "calibration"
    else:
        groups = D.stems_by_person()
        D.pool_map(
            cache_all,
            [(p, groups[p], inventory[p], str(folder)) for p in people],
            args.workers)
        template_folder = D.fresh(folder / "regression_templates")
        background_folder = folder
        for reference in refs:
            p = reference["participant"]
            with np.load(P.V5RUN / "calibration" / f"{p}_templates.npz",
                         allow_pickle=False) as saved:
                arrays = {f: saved[f].copy() for f in V5.FAMILIES}
            D.save_npz(template_folder / f"{p}_templates.npz", **arrays)

    regression = D.pool_map(
        first32_check,
        [(r, [c for c in original if c["participant"] == r["participant"]],
          inventory[r["participant"]], str(template_folder), str(background_folder))
         for r in refs], args.workers)
    if not all(r["pass"] for r in regression):
        raise RuntimeError("First-32 reproduction failed")
    D.write_json(folder / "regression.json", {
        "pass": True, "people": regression,
        "expanded_noisy_controls_have_not_run": True,
        "implementation": "array_control_rows",
    })

    if args.health_only:
        D.write_json(folder / "health.json", {
            "participants": len(people),
            "backgrounds": {p: len(inventory[p]) for p in people},
            "regression_status": "PASS",
        })
    else:
        event_rows = D.read_csv(args.prd_dir / "features/event_features.csv")
        slopes = {}
        for p in people:
            source = [
                r for r in event_rows if r["participant"] == p
                and r["set"] in ("tail", "reference")
            ]
            values = [
                float(r["rise_per_s"]) for r in source
                if P.finite(r.get("rise_per_s"))
            ]
            quantiles = np.quantile(values, [.1, .5]) if values else [None, None]
            by_set = {}
            for group in ("tail", "reference"):
                x = [
                    float(r["rise_per_s"]) for r in source
                    if r["set"] == group and P.finite(r.get("rise_per_s"))
                ]
                by_set[group] = {
                    "n": len(x),
                    "q10": float(np.quantile(x, .1)) if x else None,
                    "q50": float(np.quantile(x, .5)) if x else None,
                }
            slopes[p] = {
                "n": len(values), "missing": len(source) - len(values),
                "q10": None if quantiles[0] is None else float(quantiles[0]),
                "q50": None if quantiles[1] is None else float(quantiles[1]),
                "by_set": by_set,
            }
        results = D.pool_map(
            strata_worker,
            [(r, slopes[r["participant"]], str(folder)) for r in refs],
            args.workers)
        cells = original + [c for result in results for c in result[0]]
        D.write_json(folder / "slopes.json", slopes)
        D.write_json(folder / "control_templates.json", cells)
        D.write_csv(folder / "control_templates.csv", cells)
        D.write_json(
            folder / "template_parameters.json",
            [row for result in results for row in result[1]])
        D.write_json(folder / "background_inventory.json", inventory)
        D.write_csv(
            folder / "background_inventory.csv",
            [r for p in people for r in inventory[p]])
    P.close_stage(
        folder, "cpc_strata", args, regression_pass=True,
        prd_features_sha256=D.digest(args.prd_dir / "features/closed.json"),
        provider_access_has_not_occurred=True)
    print("CPC strata and first-32 regression closed", flush=True)


def weights(n, seed_key, pid):
    rng = np.random.default_rng(np.random.SeedSequence([P.SEED, pid, seed_key]))
    return rng.multinomial(n, np.full(n, 1 / n), size=2000)


def fraction_metric(rows, n_backgrounds, boot_weights, select, success):
    numerator = np.zeros(n_backgrounds, dtype=float)
    denominator = np.zeros(n_backgrounds, dtype=float)
    for row in rows:
        if select(row):
            bid = int(row["background_id"])
            denominator[bid] += 1
            numerator[bid] += bool(success(row))
    total = int(np.sum(denominator))
    if not total:
        return {"estimate": None, "ci95": [None, None], "n": 0, "backgrounds": 0}
    bottom, top = boot_weights @ denominator, boot_weights @ numerator
    boot = top[bottom > 0] / bottom[bottom > 0]
    return {
        "estimate": float(np.sum(numerator) / total),
        "ci95": np.quantile(boot, [.025, .975], method="inverted_cdf").tolist(),
        "n": total, "backgrounds": int(np.count_nonzero(denominator)),
    }


def family_metrics(rows, n, boot):
    core = [r for r in rows if r["core"]]
    answer = {
        "coverage": fraction_metric(
            core, n, boot, lambda r: True, lambda r: r["control_status"] == "ok")
    }
    for name, low, high, positive, inclusive in (
        ("near", 288, 320, True, False),
        ("long", 320, 650, True, True),
        ("fast", 0, 200, False, False),
    ):
        def select(row, low=low, high=high, inclusive=inclusive):
            if row["control_status"] != "ok":
                return False
            truth = float(row["true_lag_ms"])
            return truth >= low and (truth <= high if inclusive else truth < high)

        def success(row, positive=positive):
            return row["status"] == "determinate" and (
                float(row["recovered_lag_ms"]) >= 288 if positive
                else float(row["recovered_lag_ms"]) < 288)

        answer[name] = fraction_metric(core, n, boot, select, success)
    return answer


def relation(metric, gate):
    if metric["backgrounds"] < 20 or not all(P.finite(v) for v in metric["ci95"]):
        return "UNCOVERED"
    low, high = metric["ci95"]
    if high < gate:
        return "BELOW_GATE"
    if low > gate:
        return "ABOVE_GATE"
    return "CROSSES_GATE"


def error_distribution(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"n_traces": 0}
    q = np.quantile(x, [.05, .25, .5, .75, .95])
    return {
        "n_traces": len(x), "median_ms": float(q[2]),
        "IQR_ms": float(q[3] - q[1]), "p5_ms": float(q[0]), "p95_ms": float(q[4]),
        "P_le_minus4": float(np.mean(x <= -4)),
        "P_le_minus8": float(np.mean(x <= -8)),
        "P_le_minus12": float(np.mean(x <= -12)),
        "P_ge_plus4": float(np.mean(x >= 4)),
        "P_ge_plus8": float(np.mean(x >= 8)),
    }


def precision_details(rows, n, boot):
    canonical = {
        int(r["background_id"]): float(r["recovered_lag_ms"]) - float(r["true_lag_ms"])
        for r in rows
        if int(r["target_detectable_lag_ms"]) == 320
        and r["control_status"] == "ok" and r["status"] == "determinate"
    }
    curves = {
        str(target): fraction_metric(
            rows, n, boot,
            lambda r, target=target: (
                int(r["target_detectable_lag_ms"]) == target
                and r["control_status"] == "ok"),
            lambda r: (
                r["status"] == "determinate"
                and float(r["recovered_lag_ms"]) >= 288))
        for target in CURVE_TARGETS
    }
    near = [
        r for r in rows if r["core"] and r["control_status"] == "ok"
        and 288 <= float(r["true_lag_ms"]) < 320
    ]
    observed = implied = disagreement = missing = unresolved = 0
    for row in near:
        bid = int(row["background_id"])
        actual = (
            row["status"] == "determinate"
            and float(row["recovered_lag_ms"]) >= 288)
        predicted = (
            bid in canonical
            and float(row["true_lag_ms"]) + canonical[bid] >= 288)
        observed += actual
        implied += predicted
        disagreement += actual != predicted
        missing += bid not in canonical
        unresolved += row["status"] != "determinate"
    total = len(near)
    varied, compared = set(), set()
    for row in rows:
        bid = int(row["background_id"])
        if (bid in canonical and row["core"] and row["control_status"] == "ok"
                and row["status"] == "determinate"):
            compared.add(bid)
            error = float(row["recovered_lag_ms"]) - float(row["true_lag_ms"])
            if error != canonical[bid]:
                varied.add(bid)
    mechanism = {
        "canonical_target_ms": 320, "canonical_coverage": len(canonical) / n,
        "near_rows": total,
        "observed_near": observed / total if total else None,
        "implied_near": implied / total if total else None,
        "prediction_discrepancy": disagreement / total if total else None,
        "unresolved_near_fraction": unresolved / total if total else None,
        "missing_canonical_near_rows": missing,
        "traces_compared_across_targets": len(compared),
        "traces_with_target_dependent_error": len(varied),
        "implied_denominator": "same usable near rows; missing canonical error predicts failure",
    }
    if not total:
        classification = "UNDERIDENTIFIED"
    elif (
        len(canonical) / n >= .95 and disagreement / total <= .01
        and unresolved / total <= .01 and abs(implied - observed) / total <= .01
    ):
        classification = "TIMING_ERROR_CONSISTENT"
    elif unresolved:
        classification = "ASSOCIATION_OR_ANCHOR_CONTRIBUTES"
    else:
        classification = "TARGET_DEPENDENCE_OR_INSUFFICIENT_CANONICAL_COVERAGE"
    mechanism["classification"] = classification
    errors = list(canonical.values())
    return {
        "lag_error": error_distribution(errors),
        "onset_error": error_distribution([-error for error in errors]),
        "curve": curves, "CP2": mechanism,
        "unanchorable_backgrounds": len({
            int(r["background_id"]) for r in rows
            if r["reason"] == "missing_200ms_quiet_anchor_left_censored"
        }),
        "status_reason_counts": dict(Counter(
            f"{r['control_status']}:{r['status']}:{r['reason']}" for r in rows)),
    }


def write_gzip_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with gzip.open(path, "xt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def controls_worker(task):
    reference, calibration, inventory, cache_folder, output_folder, health = task
    D.require_slurm(1)
    p = reference["participant"]
    families = V5.FAMILIES if health else ALL_FAMILIES
    iterator = split_control_rows(
        reference, calibration, inventory, cache_folder, cache_folder, families)
    if health:
        rows, counts = [], Counter()
        for row in iterator:
            rows.append({key: row[key] for key in (
                "participant", "background_id", "kind", "family", "k",
                "control_status", "status", "reason")})
            counts[f"{row['kind']}:{row['control_status']}:{row['status']}"] += 1
        write_gzip_rows(Path(output_folder) / f"{p}_health.csv.gz", rows)
        return {"participant": p, "rows": len(rows), "counts": dict(counts)}

    rows = list(iterator)  # One person's rows only; never returned to the parent.
    write_gzip_rows(Path(output_folder) / f"{p}_trials.csv.gz", rows)
    original = [r for r in rows if r["family"] in V5.FAMILIES or r["kind"] == "sham"]
    first32 = [r for r in original if int(r["background_id"]) < 32]
    old_gates = V5.assess_controls(first32, [p])
    all_gates = V5.assess_controls(original, [p])
    n, pid = len(inventory), int(p.split("-")[1])
    boot, old_boot = weights(n, 20, pid), weights(32, 21, pid)
    precision = {}
    for k in V5.KS:
        group = [r for r in rows if r["k"] == k]
        families_out = {}
        for family in families:
            trials = [r for r in group if r["family"] == family]
            metrics = family_metrics(trials, n, boot)
            old_trials = [r for r in trials if int(r["background_id"]) < 32]
            old_metrics = (
                family_metrics(old_trials, 32, old_boot)
                if family in V5.FAMILIES else None)
            gates = {"coverage": .95, "near": .90, "long": .95}
            if family in ("linear", "quadratic"):
                gates["fast"] = .95
            families_out[family] = {
                "gated": family in V5.GATED,
                "metrics": metrics, "v5_metrics": old_metrics,
                "CP1": {
                    name: relation(metrics[name], gate) for name, gate in gates.items()
                } if family in V5.GATED else {},
                "CP3": relation(metrics["near"], .90) if family in DEV else None,
                **precision_details(trials, n, boot),
            }
        shams = [r for r in group if r["kind"] == "sham"]
        old_shams = [r for r in shams if int(r["background_id"]) < 32]
        sham_metric = fraction_metric(
            shams, n, boot, lambda r: True, lambda r: r["sham_alarm"] == 1)
        old_sham_metric = fraction_metric(
            old_shams, 32, old_boot, lambda r: True,
            lambda r: r["sham_alarm"] == 1)
        precision[str(k)] = {
            "families": families_out,
            "CP4": {
                "all_background_sham": sham_metric, "v5_sham": old_sham_metric,
                "relation_to_0.05": relation(sham_metric, .05),
                "alarm_reason_counts": dict(Counter(
                    f"{r['status']}:{r['reason']}" for r in shams if r["sham_alarm"])),
                "unanchorable_sham_backgrounds": len({
                    r["background_id"] for r in shams
                    if r["reason"] == "missing_200ms_quiet_anchor_left_censored"
                }),
                "unanchorable_any_control_backgrounds": len({
                    r["background_id"] for r in group
                    if r["reason"] == "missing_200ms_quiet_anchor_left_censored"
                }),
            },
        }
    return {
        "participant": p, "backgrounds": n, "trial_rows": len(rows),
        "v5": {
            str(k): old_gates["by_k"][str(k)]["participants"][p] for k in V5.KS},
        "all_background_original_gates": {
            str(k): all_gates["by_k"][str(k)]["participants"][p] for k in V5.KS},
        "precision": precision,
    }


def population_summary(results):
    output = {}
    for family in ALL_FAMILIES:
        new, old, paired = [], [], []
        for result in results:
            item = result["precision"]["5"]["families"][family]
            value = item["metrics"]["near"]["estimate"]
            if P.finite(value):
                new.append(value)
            old_item = item["v5_metrics"]
            old_value = old_item["near"]["estimate"] if old_item is not None else None
            if P.finite(old_value):
                old.append(old_value)
            if P.finite(value) and P.finite(old_value):
                paired.append(value - old_value)
        curves = {
            str(target): P.bootstrap_people([
                r["precision"]["5"]["families"][family]["curve"][str(target)]["estimate"]
                for r in results
            ], mean=True)
            for target in CURVE_TARGETS
        }
        output[family] = {
            "all_near": P.bootstrap_people(new, mean=True),
            "v5_near": P.bootstrap_people(old, mean=True),
            "paired_near_difference": P.bootstrap_people(paired, mean=True),
            "curve": curves,
        }
    return output


def controls_stage(args):
    root = P.output_root(args, "control_precision", first=False)
    metadata = P.check_stage(root / "strata", "cpc_strata", args)
    P.check_stage(args.prd_dir / "features", "prd_features", args)
    feature_hash = D.digest(args.prd_dir / "features/closed.json")
    if metadata["prd_features_sha256"] != feature_hash or not metadata["regression_pass"]:
        raise RuntimeError("CPC predecessor/regression boundary differs")
    refs, _, _, _, _ = P.inputs()
    people = P.scope(args)["participants"]
    refs = [r for r in refs if r["participant"] in people]
    folder = D.fresh(root / "controls")
    D.text_file(folder / "predeclaration.md", DECLARATION)
    if args.health_only:
        calibration = read_cells(P.V5RUN / "calibration/control_templates.csv")
        inventory = inventories(people, expanded=False)
        cache = P.V5RUN / "calibration"
    else:
        calibration = D.read_json(root / "strata/control_templates.json")
        inventory = D.read_json(root / "strata/background_inventory.json")
        cache = root / "strata"
    results = D.pool_map(
        controls_worker,
        [(r, [c for c in calibration if c["participant"] == r["participant"]],
          inventory[r["participant"]], str(cache), str(folder), args.health_only)
         for r in refs], args.workers)
    if args.health_only:
        D.write_json(folder / "health.json", {"people": results})
    else:
        cp1, cp2 = Counter(), Counter()
        for result in results:
            for family in V5.GATED:
                item = result["precision"]["5"]["families"][family]
                cp1[item["CP1"]["near"]] += 1
                cp2[item["CP2"]["classification"]] += 1
        summary = {
            "primary_k": 5, "descriptive_k": [3, 8],
            "people": {r["participant"]: r for r in results},
            "population": population_summary(results),
            "CP1_k5_near_counts": dict(cp1),
            "CP2_k5_gated_counts": dict(cp2),
            "reported_v5_retained": REPORTED_V5,
            "diagnostic_all_background_k5_gate_pass": all(
                r["all_background_original_gates"]["5"]["pass"] for r in results),
            "v5_outcomes_unchanged": {"D2": "UNDERIDENTIFIED", "D1": "UNDERIDENTIFIED"},
            "acceptance_rule_changed": False, "detector_frozen": False,
            "provider_access_has_not_occurred": True,
            "uncertainty": {
                "within_person": "2000 background bootstrap resamples; conditional precision",
                "population": "10000 equal-weight person bootstrap resamples",
            },
            "next_design_implications": {
                "BELOW_GATE": "The unchanged synthetic gate has a systematic precision limitation; detector changes alone do not alter these controls.",
                "CROSSES_GATE": "Sampling precision remains inconclusive; consider a prospectively larger acceptance sample.",
                "ABOVE_GATE": "The larger diagnostic sample lies above the gate; the original v5 outcome remains unchanged.",
                "CP3": "Slope strata assess conditional template resolution, not proof of real-onset validity.",
                "CP4": "Retain and inspect sham/anchor failures independently of near sensitivity.",
            },
        }
        D.write_json(folder / "decision.json", summary)
    P.close_stage(
        folder, "cpc_controls", args,
        strata_sha256=D.digest(root / "strata/closed.json"),
        prd_features_sha256=feature_hash,
        provider_access_has_not_occurred=True)
    print("CPC controls closed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("strata", "controls"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prd-dir", type=Path, required=True)
    parser.add_argument("--participants")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    D.require_slurm(args.workers)
    P.scope(args)
    P.code_hashes()
    {"strata": strata_stage, "controls": controls_stage}[args.stage](args)


if __name__ == "__main__":
    main()
