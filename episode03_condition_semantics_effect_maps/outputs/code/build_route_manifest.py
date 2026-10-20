#!/usr/bin/env python3
"""Build EP03's per-route manifest and the proposed methods-only orientation table.

Inputs are EP03's own records only: the route census (job 48185964), the
proposed condition-role table and the event-edit review. No producer file,
map, header, PNG, group report or score is read. The orientation rules are the
proposal in outputs/orientation_freeze_proposal.md; nothing here freezes them.

Rules, in precedence order, for a comparison with positive set P and negative
set N (weights summing to zero): R2 task minus control, R3 higher minus lower,
R4 more positive minus less positive valence, R5 declared category order
(standardised labels, alphabetical). Single-sign contrasts: R1 events minus the
implicit baseline, R6 duration (response-time) regressors, R7 parametric
regressors (higher minus lower value of the documented quantity).
"""

import argparse
import csv
import json
import os
import re
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path

LEVEL = {"control": 1, "ordinal": 2, "valence": 3, "category": 4}
RULE = {1: "R2_task_minus_control", 2: "R3_higher_minus_lower", 3: "R4_valence", 4: "R5_category_order"}
GOAL_RULES = {1, 2}  # GOAL's own examples: task minus control, higher minus lower
MID_ROUTES = {"ds003858/task-MIDmb4", "ds003858/task-MIDmb8", "ds003858/task-MIDsingle",
              "ds004920/task-mid", "ds005012/task-mid"}
# Decisions fixed before any map access (log 2026-10-05 and 2026-10-16 15:28).
FROZEN_BART = {c: 1 for c in ("pumpsvcontrol", "explodevcontrol", "pumpscashvcontrol",
                              "pumpsexplodevcontrol", "parapumpsvcontrol", "pumpspara",
                              "controlpara", "cashpara", "explodepara")}
TOL = 1e-9


def mid_levels(cond):
    """Factor levels of a monetary-incentive-delay trial type."""
    name = cond.split(".", 1)[-1].lower()
    lv = {}
    if name == "probe":
        return {"phase": "probe"}
    lv["phase"] = ("cue" if name.startswith("cue_") else "fix" if name.startswith("fix_")
                   else "fb" if name.startswith(("fb_", "feedback_")) else None)
    if "neutral" in name or "nomoneystake" in name:
        lv["incentive"] = "neutral"
    elif any(k in name for k in ("win", "gain", "loss")):
        lv["incentive"] = "incentive"
        lv["valence"] = "loss" if "loss" in name else "win"
        if "large" in name:
            lv["magnitude"] = "large"
        elif "small" in name:
            lv["magnitude"] = "small"
    if lv["phase"] == "fb":
        lv["accuracy"] = "hit" if "hit" in name else "miss" if "miss" in name else None
    return {k: v for k, v in lv.items() if v is not None}


def route_roles(roles, route, conds):
    spec = roles["routes"].get(route)
    if route in MID_ROUTES:
        tmpl = roles["routes"]["_mid_template"]
        return tmpl["factors"], {c: mid_levels(c) for c in conds}
    if not spec:
        return {}, {}
    return spec.get("factors", {}), spec.get("conditions", {})


def factor_value(factors, fname, level):
    f = factors[fname]
    return f["levels"].get(level) if level is not None else None


def orient_comparison(factors, croles, pos, neg):
    """Apply R2-R5 by precedence; returns (sign, level, evidence codes, note, one-sided levels)."""
    notes, one_sided = [], []
    for lvl in (1, 2, 3, 4):
        determinate, mixed = [], []
        for fname, f in factors.items():
            if LEVEL[f["type"]] != lvl:
                continue
            pv = [factor_value(factors, fname, croles.get(c, {}).get(fname)) for c in pos]
            nv = [factor_value(factors, fname, croles.get(c, {}).get(fname)) for c in neg]
            if set(pv) == set(nv):
                continue  # does not discriminate
            if None in pv or None in nv:
                notes.append(f"{fname}: not defined on both sides")
                one_sided.append(lvl)
                continue  # not comparable; does not block, but a lower-level decision becomes assumed
            if f["type"] == "control":
                if all(v == "task" for v in pv) and all(v == "control" for v in nv):
                    determinate.append((fname, 1))
                elif all(v == "control" for v in pv) and all(v == "task" for v in nv):
                    determinate.append((fname, -1))
                else:
                    mixed.append(fname)
            elif f["type"] in ("ordinal", "valence"):
                if min(pv) > max(nv):
                    determinate.append((fname, 1))
                elif max(pv) < min(nv):
                    determinate.append((fname, -1))
                else:
                    mixed.append(fname)
            else:
                if max(pv) < min(nv):
                    determinate.append((fname, 1))
                elif min(pv) > max(nv):
                    determinate.append((fname, -1))
                else:
                    mixed.append(fname)
        if determinate:
            signs = {s for _, s in determinate}
            if len(signs) > 1:
                return None, lvl, set(), f"conflict at {RULE[lvl]}: {determinate}", one_sided
            ev = {factors[f]["evidence"] for f, _ in determinate}
            if mixed:
                notes.append(f"mixed at the same level, ignored: {mixed}")
            used = ", ".join(f for f, _ in determinate)
            return (signs.pop(), lvl, ev, f"{RULE[lvl]} on {used}" + (f"; {'; '.join(notes)}" if notes else ""),
                    one_sided)
        if mixed:
            return None, lvl, set(), f"mixed levels of {mixed} at {RULE[lvl]}", one_sided
    return None, None, set(), "no documented factor discriminates the two sides" + (
        f"; {'; '.join(notes)}" if notes else ""), one_sided


def convolved(model):
    """Regressor names (or prefixes) passed to Convolve in the first node."""
    names = set()
    node = (model or {}).get("nodes", [{}])[0]
    for ins in (node.get("transformations") or {}).get("instructions", []):
        if ins.get("name") == "Convolve":
            names.update(ins.get("input") or [])
    return names


def is_convolved(cond, names):
    return any(cond == n or (n.endswith("*") and cond.startswith(n[:-1])) for n in names)


def duration_regressors(model):
    out = set()
    node = (model or {}).get("nodes", [{}])[0]
    for ins in (node.get("transformations") or {}).get("instructions", []):
        if ins.get("name") == "Assign" and ins.get("target_attr") == "duration":
            out.update(ins.get("target") or [])
    return out


def classify(row, model, roles):
    d = row["definitions"][0]
    net = defaultdict(float)
    for c, w in zip(d["condition_list"], d["weights"]):
        net[c] += float(w)
    net = {c: w for c, w in net.items() if abs(w) > TOL}
    out = {"conditions": d["condition_list"], "weights": d["weights"], "flags": []}
    if not net:
        out.update(cls="null", rule="none", s=None, status="ineligible", note="weights cancel on every regressor")
        return out, net
    durs = duration_regressors(model)
    kinds = {c: "ind" if c.startswith("trial_type.") else "dur" if c in durs else "pm" for c in net}
    conv = convolved(model)
    unconv = [c for c in net if not is_convolved(c, conv)]
    if not conv:
        out["flags"].append("no_convolve_instruction")
    elif unconv:
        out["flags"].append("regressor_not_convolved:" + "|".join(unconv))
    factors, croles = route_roles(roles, row["route"], list(net))
    pos = [c for c, w in net.items() if w > 0]
    neg = [c for c, w in net.items() if w < 0]
    kset = set(kinds.values())
    if not neg or not pos:
        sign = 1 if pos else -1
        if kset == {"ind"}:
            out.update(cls="E_event_vs_implicit_baseline", rule="R1_event_minus_baseline", s=sign,
                       status="proposed_rule", note="modelled event(s) minus the implicit baseline")
        elif kset == {"dur"}:
            out.update(cls="D_duration_regressor", rule="R6_duration", s=sign, status="proposed_rule",
                       note="response-time-duration regressor: longer minus shorter")
        elif kset == {"pm"}:
            info = [croles.get(c, {}) for c in net]
            if any("_dir" not in i for i in info):
                out.update(cls="P_parametric", rule="R7_parametric", s=None, status="unresolved",
                           note="direction of a parametric regressor is undocumented")
            else:
                dirs = {i["_dir"] for i in info}
                constr = {i["_constr"] for i in info}
                if len(dirs) > 1 or "U" in constr:
                    out.update(cls="P_parametric", rule="R7_parametric", s=None, status="unresolved",
                               note="signed quantity without a documented pole" if "U" in constr
                               else "parametric directions disagree")
                else:
                    status = "resolved_goal_rule" if constr <= {"E", "X"} else "assumed_goal_rule"
                    out.update(cls="P_parametric", rule="R7_parametric", s=sign * dirs.pop(), status=status,
                               note="higher minus lower value of: " + "; ".join(i.get("_quantity", "") for i in info),
                               construction="".join(sorted(constr)))
        else:
            out.update(cls="mixed_single_sign", rule="none", s=None, status="unresolved",
                       note="mixes event, duration and parametric regressors")
        return out, net
    total, absum = sum(net.values()), sum(abs(w) for w in net.values())
    if abs(total) > 0.02 * absum:
        out.update(cls="U_unbalanced", rule="none", s=None, status="unresolved",
                   note=f"two-signed weights sum to {total:.3g}")
        return out, net
    if len(kset) > 1:
        out.update(cls="C_comparison", rule="none", s=None, status="unresolved",
                   note="compares regressors of different kinds")
        return out, net
    constr = set()
    if kset == {"pm"}:
        info = [croles.get(c, {}) for c in net]
        if any("_dir" not in i for i in info) or len({i["_dir"] for i in info}) > 1:
            out.update(cls="C_comparison", rule="none", s=None, status="unresolved",
                       note="parametric directions undocumented or unequal")
            return out, net
        constr = {i["_constr"] for i in info}
    sign, lvl, ev, note, one_sided = orient_comparison(factors, croles, pos, neg)
    if sign is None:
        out.update(cls="C_comparison", rule="none", s=None, status="unresolved", note=note)
        return out, net
    higher_one_sided = any(l < lvl for l in one_sided)
    if higher_one_sided:
        out["flags"].append("higher_precedence_factor_one_sided")
    assumed = bool(ev & {"P", "I"}) or bool(constr - {"E", "X"}) or higher_one_sided
    if lvl in GOAL_RULES:
        status = "assumed_goal_rule" if assumed else "resolved_goal_rule"
    else:
        status = "proposed_rule_assumed_roles" if assumed else "proposed_rule"
    out.update(cls="C_comparison", rule=RULE[lvl], s=sign, status=status, note=note,
               evidence="".join(sorted(ev)), construction="".join(sorted(constr)) or None)
    return out, net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--roles", type=Path, required=True)
    ap.add_argument("--edits", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    prov = json.loads((args.census_dir / "route_provenance.json").read_text())
    maps = json.loads((args.census_dir / "dataset_maps.json").read_text())
    models = json.loads((args.census_dir / "embedded_models.json").read_text())
    other = json.loads((args.census_dir / "embedded_models_other_map_commits.json").read_text())
    roles = json.loads(args.roles.read_text())
    edits = json.loads(args.edits.read_text())
    routes = {r["route"]: r for r in prov["routes"]}
    mismatch = edits["name_definition_mismatches"]

    rows, seen = [], {}
    for m in sorted(maps, key=lambda x: (x["route"], x["session"] or "", x["contrast"])):
        r = routes[m["route"]]
        primary = next((c["sha"] for c in r["map_commits"] if c["dataset_z_maps"] ==
                        max(cc["dataset_z_maps"] for cc in r["map_commits"])), None)
        model = other.get(m["route"], {}).get(m["map_commit"]) if m["map_commit"] != primary else None
        model = model or models[m["route"]]
        o, net = classify(m, model, roles)
        neg_row = dict(m, definitions=[dict(m["definitions"][0], weights=[-w for w in m["definitions"][0]["weights"]])])
        o_rev, _ = classify(neg_row, model, roles)
        if (o["s"] is None) != (o_rev["s"] is None) or (o["s"] is not None and o_rev["s"] != -o["s"]) \
                or o["status"] != o_rev["status"]:
            o["flags"].append("reversal_inconsistent")
        key = (m["route"], m["session"], tuple(sorted((c, round(w, 9)) for c, w in net.items())))
        if net and key in seen:
            o.update(status="ineligible", rule="none", s=None, note=f"same definition as {seen[key]}")
            o["flags"].append("duplicate_definition")
        elif net:
            seen[key] = m["contrast"]
        ck = f"{m['route']}:{m['contrast']}"
        if ck in mismatch:
            o["flags"].append("name_definition_mismatch")
        if m["route"] == "ds000001/task-balloonanalogrisktask" and m["contrast"] in FROZEN_BART:
            if o["s"] == FROZEN_BART[m["contrast"]] and o["status"] == "resolved_goal_rule":
                o["status"] = "resolved_frozen"
            else:
                o["flags"].append("disagrees_with_frozen_decision")
        d = m["definitions"][0]
        nread = r["design_matrices"]["read"]
        if "design_runs_with_all_conditions" in d and d["design_runs_with_all_conditions"] < nread:
            o["flags"].append(f"conditions_absent_in_{nread - d['design_runs_with_all_conditions']}_of_{nread}_designs")
        if m["inputs"] < 3:
            o["flags"].append(f"n_inputs_{m['inputs']}")
        rows.append({
            "route": m["route"], "dataset_id": m["dataset_id"], "task": m["task"],
            "partition_status": r["partition_status"], "session": m["session"] or "",
            "contrast": m["contrast"], "map_commit": m["map_commit"], "map_commit_date": m["map_commit_date"][:10],
            "n_inputs": m["inputs"], "n_status": m["n_status"],
            "definition": " ".join(f"{w:+g}*{c}" for c, w in zip(o["conditions"], o["weights"])),
            "class": o["cls"], "rule": o["rule"], "s": o["s"] if o["s"] is not None else "",
            "status": o["status"], "evidence": o.get("evidence") or "", "construction": o.get("construction") or "",
            "note": o["note"], "flags": ";".join(o["flags"])})

    # Per-route manifest.
    by_route = defaultdict(list)
    for x in rows:
        by_route[x["route"]].append(x)
    manifest = []
    for name, r in sorted(routes.items()):
        mrows = by_route.get(name, [])
        b, sp, ev = r.get("build", {}), r.get("spec", {}), r.get("events", {})
        rev = edits["accessions"].get(r["dataset_id"], {})
        configured = r.get("configured_subjects")
        n_inputs = sorted({x["n_inputs"] for x in mrows})
        status_counts = Counter(x["status"] for x in mrows)
        fn = ev.get("function") or {}
        first_fn = next((h["date"][:10] for h in ev.get("function_history", []) if h["body_hash"]), None)
        rep = (r.get("report") or {}).get("parsed", [])
        commits = r.get("map_commits", [])
        psha = max(commits, key=lambda c: c["dataset_z_maps"])["sha"] if commits else None
        rep_written = all(p["changed_in_map_commit"] for p in rep
                          if p["path"] == (r.get("report") or {}).get("model_source") and p.get("map_commit") == psha)
        flags = sorted({f.split(":")[0] for x in mrows for f in x["flags"].split(";") if f})
        if r["dataset_id"] in edits.get("timing_assumptions", {}):
            flags.append("timing_assumption")
        manifest.append({
            "route": name, "dataset_id": r["dataset_id"], "task": r["task"],
            "partition_status": r["partition_status"], "dataset_z_maps": r["dataset_z_maps_in_head"],
            "map_commits": "; ".join(f"{c['sha'][:8]} {c['date'][:10]} ({c['dataset_z_maps']}) {c['subject'][:24]}"
                                     for c in r.get("map_commits", [])),
            "build": b.get("fitlins_version") or "", "estimator": b.get("estimator") or "",
            "smoothing": b.get("smoothing") or "", "drop_missing": b.get("drop_missing"),
            "drift_model": b.get("drift_model") or "",
            "build_status": ("resolved" if b.get("fitlins_version") else "no_dataset_maps" if not mrows else "unresolved"),
            "build_read_from": ("map commit" if not b.get("read_from_later_unannex_copy") else
                                f"later in-git copy ({b.get('read_from_commit_subject')})") if b else "",
            "spec_class": sp.get("difference_class_vs_local") or "",
            "spec_matches_version_at_run": sp.get("matches_version_current_at_map_commit"),
            "spec_status": ("" if not mrows else "resolved" if sp.get("embedded_model_found") and rep_written
                            else "assumed: report not rewritten by the map commit" if sp.get("embedded_model_found")
                            else "unresolved"),
            "configured_n": len(configured) if configured else "",
            "fitted_n_assumed": "/".join(str(n) for n in n_inputs),
            "n_status": ("" if not mrows else "assumed: every input effect map rewritten by the map commit; df check pending"
                         if all(x["n_status"] == "all_inputs_rewritten_with_map" for x in mrows) else "unresolved"),
            "preproc_events": ev.get("preproc_events"), "edit_function_first_committed": first_fn or "",
            "edit_status": rev.get("status", ""), "edit_kind": rev.get("kind", ""),
            "orientation_resolved": status_counts["resolved_frozen"] + status_counts["resolved_goal_rule"],
            "orientation_assumed_goal_rule": status_counts["assumed_goal_rule"],
            "orientation_proposed_rule": status_counts["proposed_rule"] + status_counts["proposed_rule_assumed_roles"],
            "orientation_unresolved": status_counts["unresolved"],
            "orientation_ineligible": status_counts["ineligible"],
            "flags": ";".join(flags)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fname, data in (("orientation_by_map.tsv", rows), ("route_manifest.tsv", manifest)):
        with open(args.output_dir / fname, "w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    summary = {
        "analysis_id": "source_handoff_qualification", "check": "route_manifest_and_orientation_proposal",
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "inputs": {k: str(v) for k, v in vars(args).items() if k != "output_dir"},
        "routes": len(manifest), "routes_with_maps": sum(1 for x in manifest if x["dataset_z_maps"]),
        "dataset_z_maps": len(rows),
        "status_counts": dict(Counter(x["status"] for x in rows)),
        "rule_counts": {f"{r} s={sg}": n for (r, sg), n in
                        Counter((x["rule"], x["s"]) for x in rows if x["rule"] != "none").most_common()},
        "class_counts": dict(Counter(x["class"] for x in rows)),
        "by_partition": {p: dict(Counter(x["status"] for x in rows if x["partition_status"] == p))
                         for p in ("included", "excluded")},
        "flag_counts": dict(Counter(f.split(":")[0] for x in rows for f in x["flags"].split(";") if f)),
        "frozen_bart_agreement": sum(1 for x in rows if x["status"] == "resolved_frozen"),
        "reversal_inconsistent": sum("reversal_inconsistent" in x["flags"] for x in rows),
        "maps_or_scores_opened": False, "elapsed_seconds": round(time.time() - t0, 2)}
    (args.output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
