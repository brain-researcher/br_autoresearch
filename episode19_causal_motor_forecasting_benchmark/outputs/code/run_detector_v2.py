#!/usr/bin/env python3
"""Provider-free EP19 detector v2 settings, labels, and v1 regression."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector_v2 as D
from detector_v2 import F, R, RF, V4
from classical_forecast_backend import GroupIdentity
from forecast_labels import AcceptedOnset, Followup, RiskDecision, label_for_decision
from self_paced_raw_onset_test_v5 import DECLARATION


def replay_run(loaded, model, low, high):
    """The supplied sp_participant's non-provider logic, one run at a time."""
    run, acc, ticks, dev, valid, bursts, tone_info = loaded
    participant, session = run.participant, run.session
    m, values = len(dev), dev[:, None]
    if m == 0:
        raise RuntimeError("Empty detector stream")
    tones = [b for b in bursts if b.get("tone")]
    tone_rows = [{
        "participant": participant, "session": session, "run": run.run,
        "onset_s": b["onset"] / run.fs, "duration_s": b["duration_s"],
        "class": b["class"], "known_s": b["known"] / run.fs,
    } for b in bursts]

    starts, cuts, evidence, start_meta = {}, set(), {}, {}
    last_classified = None
    for b in tones:
        kn = int(np.searchsorted(ticks, b["known_tick"]))
        on = int(np.searchsorted(ticks, b["onset_tick"]))
        if kn >= m:
            continue
        cuts.add(kn)
        evidence[kn] = on
        if b["class"] == "start":
            S_tick = b["known_tick"]
            lo = int(np.searchsorted(ticks, S_tick - 500))
            reasons = []
            if last_classified is None or last_classified["class"] != "end":
                status = "unknown"
                reasons.append("previous_end_not_observed")
            elif last_classified["onset_tick"] >= S_tick - 500:
                status = "unknown"
                reasons.append("end_inside_return_window")
            elif lo <= 0 or not np.all(valid[lo:kn]):
                status = "unknown"
                reasons.append("sensor_history_incomplete")
            elif np.any(dev[lo:kn] > low[0]):
                status = "not_supported"
                reasons.append("departure_exceeds_low_threshold")
            else:
                status = "supported"
            starts[kn] = status
            start_meta[kn] = (b, status, reasons)
        if b["class"] != "unclassified":
            last_classified = b

    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(np.int8), 0]))
    state_all = np.full(m, F.STATE_CODES["unknown"], np.int8)
    cue_all = np.full(m, -1, np.int32)
    all_cues, endpoint_cuts = [], {}
    replay = F.run_segment if model == "D1" else D.run_segment_d2

    for a, bnd in zip(edges[::2], edges[1::2]):
        st = {k - a: v for k, v in starts.items() if a <= k < bnd}
        ct = {k - a for k in cuts if a <= k < bnd}
        ev = {k - a: max(v, a) - a for k, v in evidence.items()
              if a <= k < bnd}
        boundary_tick = int(ticks[bnd]) if bnd < m else int(ticks[bnd - 1]) + 1
        tail = [
            x for x in bursts if ticks[a] <= x["onset_tick"] < boundary_tick
            and (x["known_tick"] >= boundary_tick
                 or (x.get("tone")
                     and np.searchsorted(ticks, x["known_tick"]) >= bnd))
        ]
        if tail:
            x = min(tail, key=lambda y: y["onset_tick"])
            ev[bnd - a] = int(np.searchsorted(ticks, x["onset_tick"])) - a
            endpoint_cuts[bnd] = (
                int(x["onset_tick"]),
                "tone_" + x["class"] if x["known_tick"] < boundary_tick
                else "unresolved_burst",
            )
        state, cue_of, cues = replay(
            values[a:bnd], low, high, F.SP_CONFIRM, F.SP_QUIET, st, ct, ev)
        state_all[a:bnd] = state
        base = len(all_cues)
        cue_all[a:bnd] = np.where(cue_of >= 0, cue_of + base, -1)
        all_cues.extend((a, bnd, c) for c in cues)

    grid_ticks = np.arange(0, int(ticks[-1]) + 1, F.SP_GRID_TICKS)
    idx = np.searchsorted(ticks, grid_ticks, side="right") - 1
    g_state = np.where(
        idx >= 0, state_all[np.maximum(idx, 0)], F.STATE_CODES["unknown"]
    ).astype(np.int8)
    g_cue = np.where(
        idx >= 0, cue_all[np.maximum(idx, 0)], -1).astype(np.int32)
    for b in bursts:
        inside = ((grid_ticks >= b["onset_tick"])
                  & (grid_ticks < b["known_tick"]) & (g_cue >= 0))
        g_state[inside & (g_state == F.STATE_CODES["at_risk"])] = \
            F.STATE_CODES["pending"]

    g_cat = np.full(len(grid_ticks), RF.NOT_AT_RISK, np.int8)
    g_obs = np.full(len(grid_ticks), -1, np.int8)
    g_reason = np.zeros(len(grid_ticks), np.int8)
    out_cue = np.full(len(grid_ticks), -1, np.int16)
    cue_rows = []
    for ci, (a, bnd, c) in enumerate(all_cues):
        kn_abs = c.start + a
        meta = start_meta.get(kn_abs)
        tone, status, reasons = meta if meta else (None, c.status, [])
        identity = GroupIdentity(
            "self_paced", participant, f"{session}/run-{run.run:04d}", f"cue-{ci}")
        if c.stop_reason == "cut":
            closing = next(
                b for b in tones
                if int(np.searchsorted(ticks, b["known_tick"])) == c.stop + a)
            stop_tick = int(closing["onset_tick"])
            stop_reason = "tone_" + closing["class"]
        else:
            boundary = c.stop + a
            stop_tick = (int(ticks[boundary]) if boundary < m
                         else int(ticks[boundary - 1]) + 1)
            stop_reason = "gap" if boundary < m else "run_end"
            if bnd in endpoint_cuts and endpoint_cuts[bnd][0] > int(ticks[kn_abs]):
                stop_tick, stop_reason = endpoint_cuts[bnd]
        unknown = None if c.unknown_from is None else int(ticks[c.unknown_from + a])
        onset = None
        if c.onset is not None:
            onset = (int(ticks[c.onset[0] + a]), int(ticks[c.onset[1] + a]))
            if onset[0] >= stop_tick:
                onset = None
        accepted = None if onset is None else AcceptedOnset(identity, *onset)
        sel = np.flatnonzero(g_cue == ci)
        out_cue[sel] = ci
        for gi in sel:
            if g_state[gi] != F.STATE_CODES["at_risk"] or grid_ticks[gi] >= stop_tick:
                if g_state[gi] == F.STATE_CODES["at_risk"]:
                    g_state[gi] = F.STATE_CODES["pending"]
                continue
            label = label_for_decision(
                RiskDecision(identity, int(grid_ticks[gi]), "at_risk"),
                Followup(identity, stop_tick, unknown), accepted,
                ticks_per_second=R.SP_TICK_HZ, grid_origin_tick=0,
            )
            g_cat[gi], g_obs[gi], g_reason[gi] = RF.label_tuple(label)

        row = {
            "source": "self_paced", "participant": participant,
            "session": session, "run": run.run, "cue": ci,
            "cue_tick": int(tone["onset_tick"]) if tone else None,
            "start_known_tick": int(ticks[kn_abs]),
            "end_tick": stop_tick, "stop_reason": stop_reason,
            "support": status, "support_reasons": ";".join(reasons),
            "onset_tick": None, "known_tick": None, "rt_ms": None,
            "confirmation_ms": None, "departure_ms": None,
            "rejected_crossings": len(c.rejected),
            "unresolved_at_end": bool(unknown is not None and status == "supported"),
        }
        if onset is not None:
            dep, _ = F.retrospective_departure(values[a:bnd], low, c.onset[0])
            row.update(
                onset_tick=onset[0], known_tick=onset[1],
                rt_ms=onset[0] - row["cue_tick"] if row["cue_tick"] is not None else None,
                confirmation_ms=onset[1] - onset[0],
                departure_ms=onset[0] - int(ticks[dep + a]) + RF.SP_SUPPORT_MS[run.fs],
            )
        cue_rows.append(row)

    seen_starts = {a + c.start for a, _, c in all_cues}
    missing = [(k, meta) for k, meta in sorted(start_meta.items())
               if k not in seen_starts]
    for extra, (k, (tone, _, reasons)) in enumerate(missing, start=len(all_cues)):
        cue_rows.append({
            "source": "self_paced", "participant": participant,
            "session": session, "run": run.run, "cue": extra,
            "cue_tick": int(tone["onset_tick"]),
            "start_known_tick": int(ticks[k]), "support": "unknown",
            "support_reasons": ";".join([*reasons, "outside_valid_segment"]),
            "onset_tick": None,
        })
    return cue_rows, (
        grid_ticks, g_state, out_cue, g_cat, g_obs, g_reason
    ), tone_rows


def label_person(task):
    participant, stems, configs = task
    D.require_slurm(1)
    result = {
        model: {"participant": participant, "cues": [], "grids": {}, "tones": []}
        for model in configs
    }
    for stem in stems:
        loaded = RF.sp_load(stem)
        run = loaded[0]
        if run.participant != participant:
            raise RuntimeError("Participant mismatch")
        for model, config in configs.items():
            cues, grid, tones = replay_run(
                loaded, model, np.array([config["low_g"]]),
                np.array([config["high_g"]]))
            out = result[model]
            out["cues"].extend(cues)
            out["tones"].extend(tones)
            out["grids"][f"{run.session}/run-{run.run:04d}"] = grid
    return result


def generated_arrays(person):
    return {
        f"{key.replace('/', '_')}_{name}": arr
        for key, grid in person["grids"].items()
        for name, arr in zip(D.ARRAY_NAMES, grid)
    }


def run_regression(args, folder):
    checks = D.check_d2_contract()
    freeze = args.freeze_dir.resolve()
    frozen = D.read_json(freeze / "detector_settings.json")["self_paced"]
    if (float(frozen["low_g"]) != V4.EXPECTED_LOW
            or float(frozen["high_g"]) != V4.EXPECTED_HIGH):
        raise RuntimeError("Unexpected frozen v1 thresholds")
    groups = D.stems_by_person()
    config = {"D1": {"low_g": V4.EXPECTED_LOW, "high_g": V4.EXPECTED_HIGH}}
    results = D.pool_map(
        label_person, [(p, groups[p], config) for p in D.PEOPLE], args.workers)
    generated = [r["D1"] for r in results]
    expected_rows = list(V4.projected_csv(
        freeze / "self_paced_cues.csv", D.CUE_FIELDS))
    actual_rows = [c for p in generated for c in p["cues"]]
    expected = {D.event_key(r): r for r in expected_rows}
    actual = {D.event_key(r): r for r in actual_rows}
    if (len(expected) != len(expected_rows) or len(actual) != len(actual_rows)
            or set(actual) != set(expected) or len(actual) != 4432):
        raise RuntimeError("v1 cue-key/count regression failed")
    for key in expected:
        for field in D.CUE_FIELDS:
            value = actual[key].get(field)
            encoded = "" if value is None else str(value)
            if encoded != expected[key][field]:
                raise RuntimeError(f"v1 cue regression failed: {key}, {field}")
    if sum(r.get("onset_tick") is not None for r in actual_rows) != 4189:
        raise RuntimeError("v1 onset-count regression failed")
    arrays_checked = 0
    for person in generated:
        arrays = generated_arrays(person)
        with np.load(
            freeze / "labels" / f"self_paced_{person['participant']}.npz",
            allow_pickle=False,
        ) as saved:
            if set(saved.files) != set(arrays):
                raise RuntimeError("v1 label-array keys differ")
            for key, value in arrays.items():
                if (saved[key].dtype != value.dtype
                        or not np.array_equal(saved[key], value)):
                    raise RuntimeError(
                        f"v1 label regression failed: {person['participant']}, {key}")
                arrays_checked += 1
    report = {
        "pass": True, "cues": 4432, "accepted_onsets": 4189,
        "arrays_checked": arrays_checked, "d2_synthetic_contract_checks": checks,
        "provider_values_used": False,
    }
    D.write_json(folder / "regression.json", report)
    D.close(folder, "regression_detector_v1", **report)
    print("Detector regression PASS", flush=True)


def write_candidate(folder, model, people, requested, config):
    label_dir = D.fresh(folder / "labels")
    cues = [c for p in people for c in p["cues"]]
    if len({D.event_key(c) for c in cues}) != len(cues):
        raise RuntimeError("Duplicate candidate cue")
    if len(requested["participants"]) == 15 and len(cues) != 4432:
        raise RuntimeError("Full candidate must retain 4432 cues")
    for person in people:
        D.save_npz(
            label_dir / f"self_paced_{person['participant']}.npz",
            **generated_arrays(person))
    D.write_csv(folder / "self_paced_cues.csv", cues, D.CUE_FIELDS)
    D.write_csv(folder / "self_paced_tones.csv",
                [t for p in people for t in p["tones"]])
    exposure = [
        RF.exposure_row("self_paced", p["participant"], p["grids"], p["cues"], 1000)
        for p in people
    ]
    D.write_csv(folder / "exposure_by_participant.csv", exposure)
    totals = {
        field: sum(row[field] for row in exposure)
        for field in exposure[0] if field not in ("source", "participant")
    }
    summary = {
        **requested, "candidate": model, "settings": config,
        "exposure": totals, "provider_access": False,
    }
    if not requested["health_only"]:
        summary["historical_backward_low_plus_summary_support_ms"] = RF.pct(
            [r.get("departure_ms") for r in cues])
        summary["historical_diagnostic_is_mechanical_for_D2"] = model == "D2"
    D.write_json(folder / "label_summary.json", summary)
    D.close(folder, f"labels_{model}", **requested, candidate=model)
    return cues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("settings", "labels", "regression"),
                        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--participants")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    D.require_slurm(args.workers)
    requested = D.scope(args)

    if args.stage == "settings":
        if args.participants is not None:
            raise ValueError("Settings always use all fifteen people; omit --participants")
        root = D.fresh(args.output_dir)
        folder = D.fresh(root / "settings")
        D.text_file(folder / "predeclaration.md", DECLARATION)
        settings = D.fit_settings(folder, args.workers)
        D.close(folder, "settings", participants=list(D.PEOPLE))
        print(f"Settings closed; counts={settings['counts']}", flush=True)
        return

    if args.stage == "regression":
        if args.freeze_dir is None or args.participants is not None:
            raise ValueError("Regression requires --freeze-dir and all fifteen people")
        folder = D.fresh(args.output_dir)
        D.text_file(folder / "predeclaration.md", DECLARATION)
        run_regression(args, folder)
        return

    if args.test_dir is None:
        raise ValueError("Labels require --test-dir")
    root, test = args.output_dir.resolve(), args.test_dir.resolve()
    D.require_closed(root / "settings", "settings")
    control_meta = D.require_closed(test / "controls", "controls")
    D.check_scope(control_meta, requested)
    settings_hash = D.digest(root / "settings" / "closed.json")
    if control_meta["settings_sha256"] != settings_hash:
        raise RuntimeError("Controls belong to different settings")
    settings = D.read_json(root / "settings" / "detector_settings.json")
    stage = D.fresh(root / "labels_stage")
    D.text_file(stage / "predeclaration.md", DECLARATION)
    folders = {model: D.fresh(root / model) for model in D.MODELS}
    groups = D.stems_by_person()
    results = D.pool_map(
        label_person,
        [(p, groups[p], settings["candidates"])
         for p in requested["participants"]],
        args.workers,
    )
    cue_sets = {}
    for model in D.MODELS:
        cue_sets[model] = write_candidate(
            folders[model], model, [r[model] for r in results],
            requested, settings["candidates"][model])
    first = {D.event_key(r): r for r in cue_sets["D1"]}
    second = {D.event_key(r): r for r in cue_sets["D2"]}
    if set(first) != set(second):
        raise RuntimeError("Candidate cue enumeration differs")
    for key in first:
        for field in ("cue_tick", "start_known_tick", "known_tick"):
            if first[key].get(field) != second[key].get(field):
                raise RuntimeError(f"Shared-threshold candidate mismatch: {key}, {field}")
        if ((first[key].get("onset_tick") is None)
                != (second[key].get("onset_tick") is None)):
            raise RuntimeError("Shared thresholds produced different confirmed events")
    D.close(
        stage, "labels", **requested, settings_sha256=settings_hash,
        controls_sha256=D.digest(test / "controls" / "closed.json"),
        candidate_closures={
            model: D.digest(root / model / "closed.json") for model in D.MODELS},
    )
    print("Both candidate label stages closed", flush=True)


if __name__ == "__main__":
    main()
