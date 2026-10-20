#!/usr/bin/env python3
"""Per-route execution-record census of EP03's NeuroEffect candidate routes.

For each FitLins task route of the candidate accessions with shared analyses,
recover from read-only git history (`git log` / `git show` only) and text files:

- the data commit that last wrote each Dataset-level Z map, and whether that
  commit also rewrote every effect map of the node feeding the Dataset node
  (the N/contributor evidence used for ds000001 on 2026-10-16);
- the runtime FitLins version and parameters: the task `dataset_description.json`
  as committed with the maps;
- the model embedded in the FitLins report at that commit, compared with the
  local producer spec and with every committed version of that spec;
- producer event edits: the basic-details flags and the dataset's function in
  `modify_events.py` current at the run date, plus a cell-level comparison of
  used vs OpenNeuro events wherever the producer's input copy survives;
- methods-only task documentation (task bold/events sidecars) for orientation.

Opens no image, NIfTI header, PNG, group report or score. Annexed map paths are
seen only as git paths, blob ids and symlink names. Standard library only.
"""

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import re
import socket
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

CODE_SUB = "code/openneuro_glmfitlins"
CONFOUND = re.compile(r"^(trans_|rot_|cosine|a_comp_cor|t_comp_cor|c_comp_cor|w_comp_cor|framewise|dvars|"
                      r"std_dvars|rmsd|global_signal|csf|white_matter|non_steady_state|motion_outlier|drift)")
MODIFY = "scripts/prep_report_py/modify_events.py"
_CAMEL = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")


# ----------------------------------------------------------------- git access
def git(gitdir, *args):
    """Run a read-only git command (log/show) and return text."""
    res = subprocess.run(["git", "-c", "core.quotepath=off", f"--git-dir={gitdir}", *args],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode:
        raise RuntimeError(f"git {args[0]} failed in {gitdir}: {res.stderr.decode()[:300]}")
    return res.stdout.decode("utf-8", "replace")


def show(gitdir, commit, path, cache=None):
    key = (str(gitdir), commit, path)
    if cache is not None and key in cache:
        return cache[key]
    try:
        text = git(gitdir, "show", f"{commit}:{path}")
    except RuntimeError:
        text = None
    if cache is not None:
        cache[key] = text
    return text


def committed_text(gitdir, commits, idx, path, cache):
    """Text of `path` at commit idx; if annexed there, the first later in-git copy."""
    text = show(gitdir, commits[idx]["sha"], path, cache)
    if text and text.lstrip().startswith(("../", ".git/annex")):
        for j in range(idx - 1, -1, -1):
            ch = commits[j]["changes"].get(path)
            if ch and ch[0] != "D" and ch[1] == "100644":
                return show(gitdir, commits[j]["sha"], path, cache), j
        return None, None
    return text, idx


def history(gitdir):
    """First-parent commits, newest first, each with its raw path changes."""
    out = git(gitdir, "log", "--first-parent", "-m", "--no-renames", "--raw", "--no-abbrev",
              "--format=@@%H%x09%ct%x09%an%x09%s")
    commits = []
    for line in out.splitlines():
        if line.startswith("@@"):
            sha, ct, author, subject = (line[2:].split("\t", 3) + [""])[:4]
            commits.append({"sha": sha, "time": int(ct), "author": author,
                            "subject": subject, "changes": {}})
        elif line.startswith(":") and commits:
            meta, path = line.split("\t", 1)
            _, new_mode, _, new_sha, status = meta[1:].split()
            commits[-1]["changes"][path] = (status[0], new_mode, new_sha)
    return commits


def file_versions(gitdir, path):
    """Commits (newest first) that changed `path`: [(sha, committer time)]."""
    try:
        out = git(gitdir, "log", "--format=%H%x09%ct", "--", path)
    except RuntimeError:
        return []
    return [(s, int(t)) for s, t in (ln.split("\t") for ln in out.splitlines() if ln)]


def blob_sha(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts)) if ts else None


# ------------------------------------------------------------ model handling
def snake(key):
    return _CAMEL.sub(r"_\1", key).lower()


def normalize(obj, parent=None):
    """pybids convert_JSON: CamelCase keys -> snake_case; Replace maps are data."""
    if isinstance(obj, dict):
        if parent == "replace":
            return {k: normalize(v) for k, v in obj.items()}
        return {snake(k): normalize(v, snake(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v, parent) for v in obj]
    return obj


def diff_paths(a, b, path="", out=None, limit=40):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in report' if k not in a else 'missing in spec'}")
            else:
                diff_paths(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: report={json.dumps(a)[:120]} spec={json.dumps(b)[:120]}")
    return out


def strip_report_defaults(a, b):
    """Drop report-only keys whose values are pybids defaults (False/None/empty)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: strip_report_defaults(v, b.get(k)) for k, v in a.items()
                if k in b or v not in (False, None, [], {})}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [strip_report_defaults(x, y) for x, y in zip(a, b)]
    return a


def spec_difference_class(emb, local):
    if emb == local:
        return "exact", []
    e2 = strip_report_defaults(emb, local)
    if e2 == local:
        return "pybids_defaults_only", diff_paths(emb, local)
    top = sorted(k for k in set(e2) | set(local) if e2.get(k) != local.get(k))
    if top == ["input"]:
        return "input_filter_only", diff_paths(e2, local)
    if "nodes" in top:
        en, ln = e2.get("nodes", []), local.get("nodes", [])
        if len(en) == len(ln) and all(
                {k: v for k, v in a.items() if k != "contrasts"} == {k: v for k, v in b.items() if k != "contrasts"}
                for a, b in zip(en, ln)):
            return ("contrasts_differ" if top == ["nodes"] else "contrasts_and_input_differ"), diff_paths(e2, local)
    return "model_differs", diff_paths(e2, local)


def camel_node(name):
    words = name.split("_")
    return words[0] + "".join(w.title() for w in words[1:])


def norm_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def report_model(text):
    """Model JSON and participant line embedded in a FitLins report."""
    if not text:
        return None, None, None
    m = re.search(r"<summary>\s*Model specification\s*</summary>\s*<pre>(.*?)</pre>", text, re.S)
    model, err = None, None
    if m:
        try:
            model = json.loads(html.unescape(m.group(1)))
        except ValueError as exc:
            err = f"embedded model not JSON: {exc}"
    else:
        err = "no embedded model block"
    p = re.search(r"<li>\s*Participants\s*\((\d+)\)\s*:\s*([^<]*)</li>", text)
    parts = {"count": int(p.group(1)), "labels": [s.strip() for s in p.group(2).split(",") if s.strip()]} if p else None
    designs = sorted(set(re.findall(r"(sub-[^/\"'\s]+?_design\.svg)", text)))
    return model, err, {"participants": parts, "design_matrix_figures": len(designs)}


def entities(basename):
    stem = basename[:-7] if basename.endswith(".nii.gz") else basename.rsplit(".", 1)[0]
    parts = stem.split("_")
    ents = {"_suffix": parts[-1]}
    for p in parts[:-1]:
        if "-" in p:
            k, v = p.split("-", 1)
            ents[k] = v
    return ents


# ------------------------------------------------------- producer code facts
_TREES = {}


def function_source(module_text, name):
    """Source of the last top-level definition of `name` (the one Python binds)."""
    key = hashlib.sha1(module_text.encode()).hexdigest()
    if key not in _TREES:
        try:
            _TREES[key] = ast.parse(module_text)
        except SyntaxError as exc:
            _TREES[key] = exc
    tree = _TREES[key]
    if isinstance(tree, SyntaxError):
        return None, f"syntax error: {tree}", 0
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if not defs:
        return None, None, 0
    return ast.get_source_segment(module_text, defs[-1]), None, len(defs)


def body_hash(src):
    return hashlib.sha1(re.sub(r"\s+", " ", src).strip().encode()).hexdigest()[:12] if src else None


def called_names(src):
    names = set()
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                f = node.func
                names.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", "?"))
    except SyntaxError:
        pass
    return sorted(names)


def at_time(versions, ts):
    """Newest version committed at or before ts (versions newest first)."""
    for sha, ct in versions:
        if ct <= ts:
            return sha
    return None


# ------------------------------------------------------------ events checks
def num(text):
    text = (text or "").strip()
    if text.lower() in ("", "n/a", "nan", "na", "none"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return None


def read_tsv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def compare_events(used_path, src_path):
    ucols, used = read_tsv(used_path)
    scols, src = read_tsv(src_path)
    out = {"added_columns": [c for c in ucols if c not in scols],
           "removed_columns": [c for c in scols if c not in ucols],
           "rows_used": len(used), "rows_source": len(src), "cell_changes": Counter(),
           "trial_types_used": Counter(r.get("trial_type") for r in used) if "trial_type" in ucols else None,
           "trial_types_source": Counter(r.get("trial_type") for r in src) if "trial_type" in scols else None}
    if len(used) == len(src):
        for u, s in zip(used, src):
            for col in scols:
                if col not in ucols:
                    continue
                sv, uv = s.get(col), u.get(col)
                if sv == uv:
                    continue
                sn, un = num(sv), num(uv)
                if sn is not None and un is not None:
                    if math.isnan(sn) and math.isnan(un):
                        continue
                    if math.isnan(sn) and un == 0.0:
                        out["cell_changes"][f"{col}:nan_to_0"] += 1
                    elif not (math.isnan(sn) or math.isnan(un)) and abs(sn - un) <= 1e-9:
                        continue
                    else:
                        out["cell_changes"][f"{col}:value"] += 1
                else:
                    out["cell_changes"][f"{col}:text"] += 1
    return out


def events_files(root, task):
    pat = re.compile(rf"_task-{re.escape(task)}(_[^/]*)?_events\.tsv$")
    found = []
    if not root.is_dir():
        return found
    for sub in sorted(e.name for e in os.scandir(root) if e.name.startswith("sub-")):
        dirs = [root / sub / "func"]
        sdir = root / sub
        if sdir.is_dir():
            dirs += [sdir / s.name / "func" for s in os.scandir(sdir) if s.name.startswith("ses-")]
        for d in dirs:
            if d.is_dir():
                found += [d / f.name for f in os.scandir(d) if pat.search(f.name)]
    return found


def route_event_comparison(input_root, raw_root, task):
    files = events_files(input_root, task)
    agg = {"files": len(files), "identical": 0, "missing_source": 0, "row_count_changed": 0,
           "unreadable": 0, "cell_changes": Counter(), "added_columns": Counter(),
           "removed_columns": Counter(), "trial_types_used": Counter(), "trial_types_source": Counter()}
    for f in files:
        rel = f.relative_to(input_root)
        src = raw_root / rel
        try:
            if not src.exists():
                agg["missing_source"] += 1
                continue
            c = compare_events(f, src)
        except Exception:  # broken annex link or unreadable text; counted, not retried
            agg["unreadable"] += 1
            continue
        agg["cell_changes"].update(c["cell_changes"])
        agg["added_columns"].update(c["added_columns"])
        agg["removed_columns"].update(c["removed_columns"])
        if c["trial_types_used"]:
            agg["trial_types_used"].update(c["trial_types_used"])
        if c["trial_types_source"]:
            agg["trial_types_source"].update(c["trial_types_source"])
        if c["rows_used"] != c["rows_source"]:
            agg["row_count_changed"] += 1
        elif not (c["cell_changes"] or c["added_columns"] or c["removed_columns"]):
            agg["identical"] += 1
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in agg.items()}


# ----------------------------------------------------- methods-only sidecars
def methods_docs(raw_root, task):
    docs = {}

    def load(path):
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    bold = load(raw_root / f"task-{task}_bold.json")
    events = load(raw_root / f"task-{task}_events.json")
    if bold is None or events is None:  # fall back to the first subject's sidecars
        for f in events_files(raw_root, task)[:1]:
            stem = f.name[: -len("_events.tsv")]
            if events is None:
                events = load(f.with_name(stem + "_events.json"))
            if bold is None:
                cand = sorted(f.parent.glob(stem.split("_run-")[0] + "*_bold.json"))
                bold = load(cand[0]) if cand else None
    if isinstance(bold, dict):
        docs["bold"] = {k: str(bold[k])[:3000] for k in ("TaskName", "TaskDescription", "CogAtlasID",
                                                          "CogPOID", "Instructions") if k in bold}
    if isinstance(events, dict):
        text = json.dumps(events)
        docs["events_sidecar"] = events if len(text) <= 20000 else {"truncated": text[:20000]}
    dd = load(raw_root / "dataset_description.json")
    if isinstance(dd, dict):
        docs["dataset"] = {k: dd[k] for k in ("Name", "ReferencesAndLinks", "DatasetDOI", "HowToAcknowledge")
                           if k in dd}
    return docs


# ------------------------------------------------------------ route analysis
def node_chain(model):
    nodes = model.get("nodes", []) if model else []
    by_name = {n.get("name"): n for n in nodes}
    edges = model.get("edges") if model else None
    source = {}
    if edges:
        for e in edges:
            source[e.get("destination")] = e.get("source")
    else:
        for prev, cur in zip(nodes, nodes[1:]):
            source[cur.get("name")] = prev.get("name")
    return nodes, by_name, source


def contrast_definitions(model):
    """Every named contrast or dummy contrast, keyed by normalized name."""
    defs = defaultdict(list)
    nodes, _, _ = node_chain(model)
    for node in nodes:
        level, nname = node.get("level"), node.get("name")
        for c in node.get("contrasts", []) or []:
            defs[norm_name(c.get("name"))].append({
                "node": nname, "level": level, "kind": "explicit", "name": c.get("name"),
                "condition_list": c.get("condition_list"), "weights": c.get("weights"),
                "test": c.get("test")})
        dummy = node.get("dummy_contrasts")
        if dummy:
            conds = dummy.get("conditions") if isinstance(dummy, dict) else None
            if conds is None:
                xs = (node.get("model") or {}).get("x") or []
                conds = [x for x in xs if isinstance(x, str)]
            for cond in conds:
                defs[norm_name(cond)].append({
                    "node": nname, "level": level, "kind": "dummy", "name": cond,
                    "condition_list": [cond], "weights": [1],
                    "test": dummy.get("test") if isinstance(dummy, dict) else None,
                    "dummy_condition_list_explicit": isinstance(dummy, dict) and "conditions" in dummy})
    return defs


def analyse_route(ds, task, repo, commits, last, head_paths, code_git, fallback_code_git,
                  input_root, cache, producer_cache):
    tdir = f"task-{task}"
    prefix = tdir + "/"
    paths = [p for p in head_paths if p.startswith(prefix)]
    route = {"dataset_id": ds, "task": task, "route": f"{ds}/{tdir}"}

    # Dataset-level Z maps: stat-z statmaps without a subject entity under node-*.
    dmaps = []
    for p in paths:
        parts = p.split("/")
        if len(parts) not in (3, 4) or not parts[1].startswith("node-"):
            continue
        if len(parts) == 4 and not parts[2].startswith("ses-"):
            continue
        ents = entities(parts[-1])
        if ents.get("stat") == "z" and ents["_suffix"] == "statmap" and "sub" not in ents:
            dmaps.append((p, parts[1], ents))
    route["dataset_z_maps_in_head"] = len(dmaps)

    # Worktree agreement with HEAD for those maps (symlink names only).
    wt = Counter()
    for p, _, _ in dmaps:
        full = repo / p
        if os.path.islink(full):
            target = os.readlink(full).encode()
            wt["matches_head" if blob_sha(target) == last[p][3] else "differs_from_head"] += 1
        elif full.exists():
            wt["regular_file_not_checked"] += 1
        else:
            wt["absent_in_worktree"] += 1
    route["worktree_vs_head_dataset_z"] = dict(wt)

    map_commits = Counter(last[p][0] for p, _, _ in dmaps)
    route["map_commits"] = [{"index": i, "sha": commits[i]["sha"][:12], "date": iso(commits[i]["time"]),
                             "subject": commits[i]["subject"], "author": commits[i]["author"],
                             "dataset_z_maps": n} for i, n in sorted(map_commits.items())]
    if not dmaps:
        route["status"] = "no_dataset_z_maps"
        return route, []
    primary = min(map_commits, key=lambda i: (-map_commits[i], i))  # most maps; newest on ties
    pc = commits[primary]
    psha, ptime = pc["sha"], pc["time"]

    # Runtime build: dataset_description.json as committed with the maps.
    dd_path = f"{tdir}/dataset_description.json"
    dd_text, dd_idx = committed_text(repo / ".git", commits, primary, dd_path, cache)
    dd = None
    if dd_text:
        try:
            dd = json.loads(dd_text)
        except ValueError:
            dd = None
    pipe = (dd or {}).get("PipelineDescription", {})
    params = pipe.get("Parameters", {}) or {}
    route["build"] = {
        "fitlins_version": pipe.get("Version"), "estimator": params.get("estimator"),
        "smoothing": params.get("smoothing"), "space": params.get("space"),
        "drop_missing": params.get("drop_missing"), "participant_label": params.get("participant_label"),
        "desc_label": params.get("desc_label"), "drift_model": params.get("drift_model"),
        "model_arg": params.get("model"), "derivatives": params.get("derivatives"),
        "ignore": params.get("ignore"), "error_ts": params.get("error_ts"),
        "description_changed_in_map_commit": dd_path in pc["changes"],
        "read_from_commit": commits[dd_idx]["sha"][:12] if dd_idx is not None else None,
        "read_from_later_unannex_copy": dd_idx is not None and dd_idx != primary,
        "read_from_commit_subject": commits[dd_idx]["subject"] if dd_idx is not None else None,
    }
    try:
        route["build"]["worktree_description_matches_map_commit"] = (repo / dd_path).read_text() == dd_text
    except OSError:
        route["build"]["worktree_description_matches_map_commit"] = None
    versions = []
    for i, c in enumerate(commits):
        if dd_path in c["changes"] and c["changes"][dd_path][0] != "D":
            t = show(repo / ".git", c["sha"], dd_path, cache)
            try:
                v = json.loads(t).get("PipelineDescription", {}).get("Version") if t else None
            except ValueError:
                v = "annexed_or_unparsable"
            versions.append({"sha": c["sha"][:12], "date": iso(c["time"]), "version": v})
    route["build"]["description_history"] = versions

    # Report: model embedded in the FitLins report committed with the maps.
    reports = sorted(p for p in paths if p.startswith(f"{tdir}/reports/") and p.endswith(".html")
                     and p.count("/") == 2)
    reports.sort(key=lambda p: (not os.path.basename(p).startswith("model-"), p))
    rep = {"reports_in_head": reports, "parsed": []}
    models = {}  # map-commit index -> (model, report path)
    for idx in sorted(map_commits):
        for rp in reports:
            text, src_idx = committed_text(repo / ".git", commits, idx, rp, cache)
            m, err, extra = report_model(text)
            rep["parsed"].append({
                "path": rp, "map_commit": commits[idx]["sha"][:12],
                "read_from_commit": commits[src_idx]["sha"][:12] if src_idx is not None else None,
                "read_from_commit_subject": commits[src_idx]["subject"] if src_idx is not None else None,
                "changed_in_map_commit": rp in commits[idx]["changes"], "model_error": err, **(extra or {})})
            if idx not in models and m is not None:
                models[idx] = (m, rp)
    if primary not in models:  # no parsable model at the map commit: fall back to HEAD, flagged
        for rp in reports:
            m, _, _ = report_model(show(repo / ".git", "HEAD", rp, cache))
            if m is not None:
                models[primary] = (m, rp)
                rep["model_from_head_not_map_commit"] = True
                break
    model = models.get(primary, (None, None))[0]
    rep["model_source"] = models.get(primary, (None, None))[1]
    route["report"] = rep
    route["_embedded_model"] = model
    route["_other_commit_models"] = {commits[i]["sha"][:12]: mm for i, (mm, _) in models.items() if i != primary}

    # Spec match: local producer spec and its committed history.
    spec = {"embedded_model_found": model is not None}
    model_arg = params.get("model") or f"{ds}-{task}_specs"
    spec_file = os.path.basename(model_arg)
    spec_file = spec_file if spec_file.endswith(".json") else spec_file + ".json"
    spec_rel = f"statsmodel_specs/{ds}/{spec_file}"
    code_dir = repo / CODE_SUB
    spec["spec_path"] = f"{CODE_SUB}/{spec_rel}"
    gitdir = code_git if code_git else fallback_code_git
    spec["producer_clone"] = str(gitdir)
    local = None
    try:
        local = json.loads((code_dir / spec_rel).read_text())
    except (OSError, ValueError) as exc:
        spec["local_spec_error"] = repr(exc)[:200]
    emb = normalize(model) if model is not None else None
    if local is not None and emb is not None:
        spec["matches_local"] = normalize(local) == emb
        spec["difference_class_vs_local"], diffs = spec_difference_class(emb, normalize(local))
        if diffs:
            spec["differences_vs_local"] = diffs
        if spec["difference_class_vs_local"] not in ("exact", "pybids_defaults_only"):
            spec["local_contrasts"] = [
                {"node": n.get("name"), "contrasts": n.get("contrasts")} for n in normalize(local).get("nodes", [])]
        spec["other_map_commit_class_vs_local"] = {
            sha: spec_difference_class(normalize(mm), normalize(local))[0]
            for sha, mm in route["_other_commit_models"].items()}
    hist = file_versions(gitdir, spec_rel)
    matches = []
    for sha, ct in hist:
        t = show(gitdir, sha, spec_rel, producer_cache)
        try:
            ok = emb is not None and t is not None and spec_difference_class(
                emb, normalize(json.loads(t)))[0] in ("exact", "pybids_defaults_only")
        except ValueError:
            ok = False
        if ok:
            matches.append({"sha": sha[:12], "date": iso(ct)})
    spec["committed_versions"] = len(hist)
    spec["matching_committed_versions"] = matches
    at_run = at_time(hist, ptime)
    spec["version_current_at_map_commit"] = at_run[:12] if at_run else None
    spec["matches_version_current_at_map_commit"] = bool(at_run and any(m["sha"] == at_run[:12] for m in matches))
    spec["latest_version_matches"] = bool(hist and matches and matches[0]["sha"] == hist[0][0][:12])
    route["spec"] = spec

    # Producer event edits current at the map commit.
    route["events"] = producer_event_facts(ds, task, gitdir, ptime, producer_cache)
    used_root = input_root / ds
    if used_root.is_dir():
        route["events"]["used_vs_openneuro"] = route_event_comparison(used_root, repo / "sourcedata/raw", task)
    else:
        route["events"]["used_vs_openneuro"] = "producer input copy not retained"

    # Inputs to the Dataset node and their rewrite status.
    nodes, by_name, source = node_chain(model)
    dir_to_node = {f"node-{camel_node(n.get('name', ''))}": n for n in nodes}
    inputs = defaultdict(list)  # (node dir, contrast, session) -> [(subject, path, commit idx)]
    level_rewrites = Counter()
    for p in paths:
        parts = p.split("/")
        if len(parts) < 4 or not parts[1].startswith("node-") or not parts[2].startswith("sub-"):
            continue
        ents = entities(parts[-1])
        if ents.get("stat") != "effect" or ents["_suffix"] != "statmap":
            continue
        inputs[(parts[1], ents.get("contrast"), ents.get("ses"))].append((ents.get("sub"), p, last[p][0]))
        level_rewrites[(parts[1], last[p][0] == primary)] += 1
    route["effect_maps_by_node"] = {
        nd: {"in_head": level_rewrites[(nd, True)] + level_rewrites[(nd, False)],
             "rewritten_by_primary_map_commit": level_rewrites[(nd, True)]}
        for nd in sorted({k[0] for k in level_rewrites})}

    configured = ((model or {}).get("input") or {}).get("subject")
    route["configured_subjects"] = configured

    # Run-level design matrices: header lines only (regressor names).
    run_dir = f"node-{camel_node(nodes[0].get('name', ''))}" if nodes else None
    design_cols, design = Counter(), {"files": 0, "read": 0, "unreadable": 0, "rewritten_with_primary": 0}
    design_sets = []
    for p in paths:
        parts = p.split("/")
        if run_dir and len(parts) >= 4 and parts[1] == run_dir and parts[2].startswith("sub-") \
                and parts[-1].endswith("_design.tsv"):
            design["files"] += 1
            design["rewritten_with_primary"] += last[p][0] == primary
            try:
                with open(repo / p) as handle:
                    cols = handle.readline().rstrip("\n").split("\t")
                design["read"] += 1
                design_sets.append(set(cols))
                design_cols.update(c for c in cols if not CONFOUND.match(c))
            except OSError:
                design["unreadable"] += 1
    design["non_confound_columns"] = dict(sorted(design_cols.items()))
    route["design_matrices"] = design
    rows = []
    for p, ndir, ents in sorted(dmaps):
        idx = last[p][0]
        node = dir_to_node.get(ndir)
        src_name = source.get(node.get("name")) if node else None
        src_dir = f"node-{camel_node(src_name)}" if src_name else None
        src_level = by_name.get(src_name, {}).get("level") if src_name else None
        contrast, ses = ents.get("contrast"), ents.get("ses")
        group_by = (node or {}).get("group_by")
        cand = []
        if src_dir:
            if ses is not None:
                cand = inputs.get((src_dir, contrast, ses), [])
            else:
                cand = [x for (nd, c, s), lst in inputs.items() if nd == src_dir and c == contrast for x in lst]
        per_sub = Counter(s for s, _, _ in cand)
        rewritten = [x for x in cand if x[2] == idx]
        newer = [x for x in cand if x[2] < idx]
        older_commits = sorted({commits[x[2]]["sha"][:12] for x in cand if x[2] > idx})
        if not cand:
            n_status = "no_input_effect_maps_found"
        elif newer:
            n_status = "inputs_changed_after_map"
        elif len(rewritten) == len(cand):
            n_status = "all_inputs_rewritten_with_map"
        elif not rewritten:
            n_status = "no_inputs_rewritten_with_map"
        else:
            n_status = "partially_rewritten"
        stats_same = {}
        for stat in ("t", "effect", "variance", "p"):
            q = p.replace("_stat-z_statmap", f"_stat-{stat}_statmap")
            stats_same[stat] = (last[q][0] == idx) if q in last and last[q][1] != "D" else None
        mdl = models.get(idx, models.get(primary, (None, None)))[0]
        d = contrast_definitions(mdl).get(norm_name(contrast), []) if mdl else []
        for dd_ in d:
            if dd_.get("level", "").lower() == "run" and design_sets and idx == primary:
                conds = [c for c in dd_.get("condition_list") or [] if isinstance(c, str)]
                dd_["design_runs_with_all_conditions"] = sum(all(c in ds_ for c in conds) for ds_ in design_sets)
                dd_["design_runs_missing"] = dict(Counter(c for ds_ in design_sets for c in conds if c not in ds_))
        subjects = sorted(per_sub)
        rows.append({
            "route": route["route"], "dataset_id": ds, "task": task, "node": ndir,
            "node_level": (node or {}).get("level"), "group_by": group_by, "session": ses,
            "contrast": contrast, "path": p, "map_commit": commits[idx]["sha"][:12],
            "map_commit_date": iso(commits[idx]["time"]), "is_primary_commit": idx == primary,
            "definition_from_commit": commits[idx]["sha"][:12] if idx in models else commits[primary]["sha"][:12],
            "other_stats_same_commit": stats_same, "source_node": src_dir, "source_level": src_level,
            "inputs": len(cand), "input_subjects": len(subjects), "max_inputs_per_subject": max(per_sub.values()) if per_sub else 0,
            "inputs_rewritten_with_map": len(rewritten), "inputs_from_older_commits": older_commits,
            "n_status": n_status,
            "missing_configured": sorted(set(configured) - set(subjects)) if configured and cand else None,
            "extra_vs_configured": sorted(set(subjects) - set(configured)) if configured and cand else None,
            "subjects": subjects, "definitions": d})
    route["status"] = "censused"
    return route, rows


def producer_event_facts(ds, task, gitdir, ptime, cache):
    out = {}
    bd_rel = f"statsmodel_specs/{ds}/{ds}_basic-details.json"
    bd_hist = file_versions(gitdir, bd_rel)
    bd_sha = at_time(bd_hist, ptime) or (bd_hist[-1][0] if bd_hist else None)
    out["basic_details_version"] = bd_sha[:12] if bd_sha else None
    out["basic_details_versions"] = len(bd_hist)
    bd_text = show(gitdir, bd_sha, bd_rel, cache) if bd_sha else None
    tinfo = None
    if bd_text:
        try:
            tasks = json.loads(bd_text).get("Tasks", {})
            key = next((k for k in tasks if k.lower() == task.lower()), None)
            if key is None:
                pref = sorted((k for k in tasks if task.lower().startswith(k.lower())), key=len)
                key = pref[-1] if pref else None
                out["basic_details_task_key_by_prefix"] = key
            tinfo = tasks.get(key) if key else None
        except ValueError:
            tinfo = None
    if tinfo:
        out.update({k: tinfo.get(k) for k in ("preproc_events", "dummy_volumes", "task_runs", "task_sessions",
                                               "trial_type_values", "column_names", "cite_links", "bold_volumes")})
    else:
        out["basic_details_task_entry"] = "absent"

    key = ("modify_hist", str(gitdir))
    if key not in cache:
        cache[key] = file_versions(gitdir, MODIFY)
    mod_hist = cache[key]
    at_run = at_time(mod_hist, ptime)
    bodies = []
    for sha, ct in reversed(mod_hist):  # oldest first
        text = show(gitdir, sha, MODIFY, cache)
        src, err, ndefs = function_source(text, ds) if text else (None, "unreadable", 0)
        bodies.append({"sha": sha, "time": ct, "hash": body_hash(src), "src": src, "defs": ndefs})
    out["modify_events_version_at_map_commit"] = at_run[:12] if at_run else None
    changes, prev = [], "start"
    for b in bodies:
        if b["hash"] != prev:
            changes.append({"sha": b["sha"][:12], "date": iso(b["time"]), "body_hash": b["hash"]})
            prev = b["hash"]
    out["function_history"] = changes
    cur = next((b for b in bodies if b["sha"] == at_run), None)
    if cur and cur["src"]:
        first = cur
        for b in reversed([b for b in bodies if b["time"] <= cur["time"]]):
            if b["hash"] != cur["hash"]:
                break
            first = b
        later = sorted({b["hash"] for b in bodies if b["time"] > cur["time"] and b["hash"] != cur["hash"]})
        out["function"] = {
            "name": ds, "definitions_in_module": cur["defs"], "body_hash": cur["hash"],
            "body_unchanged_since": iso(first["time"]), "body_unchanged_since_commit": first["sha"][:12],
            "later_different_bodies": len([h for h in later if h]), "calls": called_names(cur["src"]),
            "mentions_task": task.lower() in cur["src"].lower(), "source": cur["src"]}
    else:
        out["function"] = None
    head = bodies[-1] if bodies else None
    out["function_at_latest_commit_hash"] = head["hash"] if head else None
    out["applied"] = bool(out.get("preproc_events")) and out["function"] is not None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--analysis-root", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these accessions")
    args = ap.parse_args()
    t0 = time.time()
    cand = json.loads(args.candidates.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fallback = args.analysis_root / "ds000001" / CODE_SUB / ".git"
    cache, producer_cache = {}, {}
    routes, maps, methods, errors = [], [], {}, []
    for row in cand["candidates"]:
        ds = row["dataset_id"]
        if args.only and ds not in args.only:
            continue
        if not row["fitlins_analysis_dir_present"]:
            continue
        repo = args.analysis_root / ds
        t1 = time.time()
        try:
            commits = history(repo / ".git")
            merges = len(git(repo / ".git", "log", "--merges", "--format=%H").split())
        except Exception as exc:
            errors.append({"dataset_id": ds, "stage": "history", "error": repr(exc)[:300]})
            continue
        last = {}
        for i, c in enumerate(commits):
            for p, (st, mode, sha) in c["changes"].items():
                if p not in last:
                    last[p] = (i, st, mode, sha)
        head_paths = [p for p, v in last.items() if v[1] != "D"]
        code_git = repo / CODE_SUB / ".git"
        code_git = code_git if code_git.is_dir() else None
        code_head = None
        if code_git:
            try:
                code_head = git(code_git, "log", "-1", "--format=%H %ct").split()
            except RuntimeError:
                code_head = None
        for task_dir in row["fitlins_task_routes"]:
            task = task_dir[len("task-"):]
            try:
                route, rows = analyse_route(ds, task, repo, commits, last, head_paths, code_git, fallback,
                                            args.input_root, cache, producer_cache)
            except Exception as exc:
                errors.append({"dataset_id": ds, "task": task, "stage": "route", "error": repr(exc)[:500]})
                continue
            route.update({"partition_status": row["partition_status"], "repo_commits": len(commits),
                          "repo_merge_commits": merges, "repo_head": commits[0]["sha"][:12] if commits else None,
                          "producer_clone_head": code_head[0][:12] if code_head else None,
                          "producer_clone_present": code_git is not None,
                          "census_dataset_z_filenames": sum((row["z_maps_by_task_and_node"] or {}).get(task_dir, {}).values())})
            routes.append(route)
            maps += rows
            try:
                methods[route["route"]] = methods_docs(repo / "sourcedata/raw", task)
            except Exception as exc:
                methods[route["route"]] = {"error": repr(exc)[:200]}
        print(f"{ds}: {len(commits)} commits, {time.time() - t1:.1f}s", flush=True)

    # Embedded models and function sources go to their own files.
    models = {r["route"]: r.pop("_embedded_model", None) for r in routes}
    other_models = {r["route"]: r.pop("_other_commit_models") for r in routes if r.get("_other_commit_models")}
    for r in routes:
        r.pop("_other_commit_models", None)
    functions = {}
    for r in routes:
        f = (r.get("events") or {}).get("function")
        if f and f.get("source"):
            functions.setdefault(f"{r['dataset_id']}:{f['body_hash']}", f.pop("source"))
        elif f:
            f.pop("source", None)
    report = {
        "analysis_id": "source_handoff_qualification", "check": "route_execution_record_census",
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "inputs": {k: str(v) for k, v in vars(args).items() if k != "output_dir"},
        "git_commands": ["log", "show"], "images_headers_pngs_group_reports_or_scores_opened": False,
        "routes": routes, "errors": errors, "elapsed_seconds": time.time() - t0}
    (args.output_dir / "route_provenance.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (args.output_dir / "dataset_maps.json").write_text(json.dumps(maps, indent=1, default=str) + "\n")
    (args.output_dir / "methods_sources.json").write_text(json.dumps(methods, indent=1, default=str) + "\n")
    (args.output_dir / "modify_events_functions.json").write_text(json.dumps(functions, indent=1) + "\n")
    (args.output_dir / "embedded_models.json").write_text(json.dumps(models, indent=1) + "\n")
    (args.output_dir / "embedded_models_other_map_commits.json").write_text(json.dumps(other_models, indent=1) + "\n")
    print(json.dumps({"routes": len(routes), "dataset_z_maps": len(maps), "errors": errors,
                      "elapsed_seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
