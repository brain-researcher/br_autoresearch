#!/usr/bin/env python3
"""EP16 single sealed T9 evaluation (case-series amendment, 2026-10-16). Not to be run before freezing.

Sealed mode verifies the frozen specification, selected panel, T11 positive control, checks, code,
runtime inputs and T9 archive identity before claiming the next write-once attemptNN directory.
A restart is allowed only when every previous attempt has a failure record confirming that no
primary-results file was written. Protected T9 fields are read only after the opened record exists.

Rehearsal mode (--rehearsal-feature-dir or --rehearsal-from-archive, development participant only)
runs the same evaluation code on development data and writes to a rehearsal directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import tarfile
import tempfile
import time

import numpy as np

import ep16_branches as br
import ep16_control as pc
import ep16_roles
from ep16_data import (assemble, load_feature_dir, packet_summary, passing_component_days, realized_design,
                       structural_records)
from ep16_panel import (Panel, adjusted_contrast, bootstrap, combine, conclusion, observed_decision,
                        one_sided_bound, primary_statistics, robust_contrast)
from ep16_roles import SealedEvaluation, authorize_protected_read
from extract_development_features import per_electrode_arrays
from mat5_neural import read_session_windows
from mat5_structural import field_vector

EPISODE_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "runs" / "t9_evaluation"
FROZEN_VALUES = ("ridge_lambda_sbp", "ridge_lambda_tx_4_5", "ridge_lambda_sbp_frozen_scaling",
                 "J_min", "D_min", "raw_score_decline_margin")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, payload):
    Path(path).write_text(json.dumps(jsonable(payload), indent=1) + "\n")


def record_failure(output_dir, stage, error):
    payload = {"stage": stage, "error": repr(error),
               "no_results_written": not (Path(output_dir) / "primary_results.json").exists(),
               "utc": datetime.now(timezone.utc).isoformat()}
    try:
        write(Path(output_dir) / "failure.json", payload)
    except OSError as failure_error:
        error.add_note(f"Could not write failure.json: {failure_error!r}")


# ----------------------------------------------------------------------------- freeze and seal
def panel_choice_errors(panel):
    if not isinstance(panel, dict):
        return ["selected panel must be an object"]
    problems = []
    if panel.get("normalization") not in ("block", "none"):
        problems.append("panel normalization must be block or none")
    if panel.get("readiness") not in ("strict", "screened"):
        problems.append("panel readiness must be strict or screened")
    if panel.get("supervised_32") not in br.SUPERVISED:
        problems.append("panel supervised_32 is not a registered supervised method")
    if panel.get("emulator") not in ("E1_dropout", "E2_theta0", "E2_theta0_floor"):
        problems.append("panel emulator is not a selectable frozen emulator")
    unlabelled = panel.get("unlabelled")
    allowed = (("past_only", {"history": 16}), ("past_only", {"history": 64}),
               ("whole_session", {}), ("covariance", {"alpha": 0.5}),
               ("covariance", {"alpha": 0.9}), ("subspace", {"k_sub": 8}))
    if (not isinstance(unlabelled, dict) or set(unlabelled) != {"method", "settings"}
            or not isinstance(unlabelled.get("settings"), dict)
            or not any(unlabelled["method"] == method and unlabelled["settings"] == settings
                       for method, settings in allowed)):
        problems.append("panel unlabelled method/settings are not a registered choice")
    return problems


def frozen_choice_errors(spec, panel, control):
    """Return inconsistencies that prevent freezing or opening the selected evaluation."""
    problems = panel_choice_errors(panel)
    if not isinstance(spec, dict) or not isinstance(control, dict):
        return problems + ["specification and positive control must be objects"]
    if not isinstance(panel, dict):
        return problems
    decisions = spec.get("scientist_decisions")
    if not isinstance(decisions, dict):
        problems.append("scientist_decisions must record normalization and readiness")
    else:
        for key in ("normalization", "readiness"):
            if key not in decisions or decisions[key] != panel.get(key):
                problems.append(f"scientist_decisions.{key} differs from the selected panel")

    def pending(value):
        if isinstance(value, dict):
            return any((key == "proposed_pending_scientist" and bool(item)) or pending(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(pending(item) for item in value)
        return False

    if pending(spec):
        problems.append("specification has non-empty proposed_pending_scientist")
    if control.get("participant") != "T11":
        problems.append("positive control must identify participant T11")
    for key in ("normalization", "readiness"):
        if key not in control or control[key] != panel.get(key):
            problems.append(f"T11 positive-control {key} differs from the selected panel")
    uncertainty = spec.get("uncertainty")
    if not isinstance(uncertainty, dict):
        problems.append("uncertainty settings are missing")
    else:
        for key in ("draws", "sensitivity_draws"):
            value = uncertainty.get(key)
            if type(value) is not int or value <= 0:
                problems.append(f"uncertainty.{key} must be a positive integer")
        if type(uncertainty.get("seed")) is not int or uncertainty["seed"] < 0:
            problems.append("uncertainty.seed must be a nonnegative integer")
        if control.get("draws") != uncertainty.get("draws"):
            problems.append("T11 positive-control draws differ from uncertainty.draws")
    values, recorded = spec.get("values_from_T11"), control.get("values")
    if not isinstance(values, dict) or not isinstance(recorded, dict):
        problems.append("specification and positive control must record T11-derived values")
    else:
        for key in FROZEN_VALUES:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"values_from_T11.{key} is not fixed and finite")
        for key in FROZEN_VALUES:  # metadata keys (evidence, timestamps) need not match
            if key not in recorded or recorded[key] != values.get(key):
                problems.append(f"T11 positive-control value {key} differs from the specification")
    return problems


def verify_manifest(manifest_path):
    from freeze_t9_evaluation import CODE, RUNTIME_INPUTS

    problems = []
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object")
        pointers = ("specification", "selected_panel", "positive_control", "synthetic_checks")
        for key in pointers:
            if not isinstance(manifest.get(key), str) or not manifest[key]:
                raise ValueError(f"Manifest is missing {key}")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest files must be a nonempty list")
        listed = []
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]
                    or not isinstance(entry.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]) is None):
                raise ValueError("Malformed manifest file entry")
            path = Path(entry["path"]).resolve()
            listed.append(path)
            if sha256(path) != entry["sha256"].lower():
                problems.append(f"{entry['path']}: hash differs")
        if len(listed) != len(set(listed)):
            problems.append("manifest files contain duplicate paths")
        required = {Path(manifest[key]).resolve() for key in pointers}
        required.update((EPISODE_ROOT / "outputs" / "code" / name).resolve() for name in CODE)
        required.update((EPISODE_ROOT / path).resolve() for path in RUNTIME_INPUTS)
        required.update((EPISODE_ROOT / "outputs" / "config" / name).resolve()
                        for name in ("analysis_definition.json", "schedule_support.json", "source_fact_rules.json"))
        required.add(Path(__file__).resolve().with_name("freeze_t9_evaluation.py"))
        missing = required - set(listed)
        if missing:
            problems.append("manifest omits required files: " + ", ".join(str(p) for p in sorted(missing)))
        spec = json.loads(Path(manifest["specification"]).read_text())
        control = json.loads(Path(manifest["positive_control"]).read_text())
        panel = json.loads(Path(manifest["selected_panel"]).read_text())
        problems.extend(frozen_choice_errors(spec, panel, control))
        if not isinstance(control, dict) or control.get("pass") is not True:
            problems.append("known-positive control did not pass on T11")
        if (not isinstance(panel, dict) or panel.get("status") != "selected_by_development_search"
                or panel.get("rehearsal_only")):
            problems.append("selected panel is not a development-search selection")
        checks = Path(manifest["synthetic_checks"]).read_text()
        if not re.search(r"^OK\b", checks, flags=re.M) or re.search(r"^FAILED", checks, flags=re.M) or "skipped" in checks:
            problems.append("synthetic checks did not all run and pass")
        archive = manifest.get("t9_archive")
        if (not isinstance(archive, dict) or not isinstance(archive.get("path"), str)
                or not Path(archive["path"]).is_absolute() or Path(archive["path"]).name != "decoding_T9.tar.gz"
                or type(archive.get("size")) is not int or archive["size"] <= 0
                or not isinstance(archive.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", archive["sha256"]) is None):
            problems.append("manifest lacks a valid T9 archive path, size and SHA-256")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        raise SystemExit(f"Freeze manifest check failed: {error}") from error
    if problems:
        raise SystemExit("Freeze manifest check failed: " + "; ".join(problems))
    return manifest, spec, panel, control


def verify_archive(manifest):
    """Verify the acquired archive before claiming an evaluation attempt; hash in 1 MiB chunks."""
    expected = manifest["t9_archive"]
    archive = Path(expected["path"])
    try:
        if archive.stat().st_size != expected["size"]:
            raise SystemExit("T9 archive byte size differs from the acquisition manifest")
        if sha256(archive) != expected["sha256"].lower():
            raise SystemExit("T9 archive SHA-256 differs from the acquisition manifest")
    except OSError as error:
        raise SystemExit(f"Cannot verify the T9 archive: {error}") from error


def open_seal(output_dir, manifest_path):
    """Verify the freeze and archive, exclusively claim the next attempt, then issue the T9 token."""
    manifest, _, _, _ = verify_manifest(manifest_path)
    verify_archive(manifest)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    earlier = sorted(p for p in EVALUATION_ROOT.iterdir() if p.is_dir() and p.name.startswith("attempt"))
    for path in earlier:
        match = re.fullmatch(r"attempt([0-9]{2,})", path.name)
        if (path.is_symlink() or match is None or int(match.group(1)) < 1
                or path.name != f"attempt{int(match.group(1)):02d}"):
            raise SystemExit(f"Invalid existing attempt directory: {path}")
        try:
            failure = json.loads((path / "failure.json").read_text())
        except (OSError, ValueError) as error:
            raise SystemExit(f"Earlier attempt has no readable failure record: {path}") from error
        if (not isinstance(failure, dict) or failure.get("no_results_written") is not True
                or (path / "primary_results.json").exists()):
            raise SystemExit(f"Earlier attempt does not permit a restart: {path}")
    output_dir = Path(output_dir)
    expected_name = f"attempt{len(earlier) + 1:02d}"
    if output_dir.parent.resolve() != EVALUATION_ROOT.resolve() or output_dir.name != expected_name:
        raise SystemExit(f"The T9 output directory must be {EVALUATION_ROOT / expected_name}")
    output_dir.mkdir(exist_ok=False)  # exclusive mkdir is the atomic claim
    try:
        nonce = secrets.token_hex(16)
        record = {"participant": "T9", "nonce": nonce, "opened_utc": datetime.now(timezone.utc).isoformat(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
                  "freeze_manifest": str(Path(manifest_path).resolve()), "freeze_manifest_sha256": sha256(manifest_path)}
        with open(output_dir / ep16_roles.OPENED_RECORD, "x") as handle:
            handle.write(json.dumps(record, indent=1) + "\n")
        return SealedEvaluation(ep16_roles._ISSUER, "T9", output_dir, nonce)
    except BaseException as error:
        record_failure(output_dir, "open_seal", error)
        raise


def read_archive_sessions(source_root, participant, days, token, scratch_dir):
    """Role-checked in-memory session arrays from an archive; no feature cache is written."""
    authorize_protected_read(participant, token)
    archive = Path(source_root) / "source" / f"decoding_{participant}.tar.gz"
    escaped = re.escape(participant)
    pattern = re.compile(rf"decoding/{escaped}/{escaped}_day_([0-9]+)_decoding\.mat")
    wanted, out, seen = set(days), {}, set()
    Path(scratch_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=f"ep16-{participant.lower()}-", dir=scratch_dir) as private, tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            match = pattern.fullmatch(member.name.removeprefix("./"))
            if not member.isfile() or match is None:
                continue
            day = int(match.group(1))
            if day in seen:
                raise ValueError(f"Duplicate {participant} day member: {day}")
            seen.add(day)
            if day not in wanted:
                continue
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read archive member {member.name}")
            with extracted, tempfile.TemporaryFile(dir=private) as opaque:
                shutil.copyfileobj(extracted, opaque, length=1 << 20)
                values, f, index, _ = read_session_windows(opaque, participant, token=token)
            electrodes = values["electrodes"]
            ids = field_vector(electrodes, "electrode_id").astype(np.int64)
            out[day] = {"trial_start": index["starts"], "window_inside_block": index["window_inside_block"],
                        "trial_end": index["ends"], "trial_block": index["trial_blocks"],
                        "target_start": np.asarray(values["trial_targets"], dtype=np.float64),
                        "cursor_start": f["cursor_start"], "sbp": f["sbp"], "tx_4_5": f["tx_4_5"],
                        "sbp_odd_even": f["sbp_odd_even"], "electrode_id": ids,
                        "electrode_array": per_electrode_arrays(electrodes, len(ids))}
    if set(out) != wanted:
        raise SystemExit(f"{participant} sessions missing: {sorted(wanted - set(out))}")
    return out


# ----------------------------------------------------------------------------- evaluation
def primary_bounds(draws):
    return {"adjusted": one_sided_bound(draws["adjusted"], 0.0125, "lower"),
            "robust": one_sided_bound(draws["robust"], 0.0125, "lower"),
            "fractional_local": one_sided_bound(draws["fractional_local"], 0.0125, "upper"),
            "raw_transported": one_sided_bound(draws["raw_transported"], 0.0125, "upper"),
            "nonestimable": {k: int((~np.isfinite(v)).sum()) for k, v in draws.items()}}


def readiness_nodes(sessions, panel_choice, lam, J_min):
    """Strict keeps every node; screened removes sessions without finite observed J > J_min."""
    rule = panel_choice.get("readiness")
    if rule == "strict":
        return sessions, []
    if rule != "screened":
        raise ValueError(f"Unknown readiness rule {rule!r}")
    J = Panel(sessions, "sbp", lam, J_min).tables["fit"]["J"]
    keep = np.isfinite(J) & (J > J_min)
    return [s for s, ok in zip(sessions, keep) if ok], [s.day for s, ok in zip(sessions, keep) if not ok]


def evaluate(sessions, packets, design, spec, panel_choice, args, out_dir, array_labels,
             frozen_scaling_sessions=None, progress=None):
    progress = {} if progress is None else progress
    out_dir = Path(out_dir)
    progress["stage"] = "packets_and_design"
    values = spec["values_from_T11"]
    J_min, D_min = values["J_min"], values["D_min"]
    raw_margin, lam = values["raw_score_decline_margin"], values["ridge_lambda_sbp"]
    support_config = json.loads((EPISODE_ROOT / "outputs/config/schedule_support.json").read_text())
    report = {"packets": packet_summary(packets), "values_from_T11": values, "selected_panel": panel_choice, "timing_seconds": {}}
    states = {"extra_loss": "unresolved", "local_signal": "unresolved", "recording": "not_evaluated", "repair": "not_evaluated"}
    pc_flags = {"extra_loss": False, "local_signal": False}
    nodes, removed = readiness_nodes(sessions, panel_choice, lam, J_min)
    design = realized_design(nodes, support_config)
    report["realized_design"] = {k: v for k, v in design.items() if k != "contrast_node_shares_by_day"}
    report["readiness_screen"] = {"rule": panel_choice["readiness"], "removed_days": removed}
    write(out_dir / "packets_and_design.json", {
        k: report[k] for k in ("packets", "values_from_T11", "selected_panel", "realized_design", "readiness_screen")})
    if not design["conditional_numerical_pass"]:
        reason = "realized design fails the unchanged component gates"
        report["primary"] = {"status": "not_evaluated", "reason": reason}
        report["known_positive_control"] = {"status": "not_evaluated", "reason": reason}
        report["recording"] = {"status": "not_evaluated"}
        report["repair"] = {"status": "not_evaluated"}
        report["sensitivities"] = {"status": "not_evaluated", "reason": reason}
        for stage, payload in (("primary_results", report["primary"]),
                               ("known_positive_control", report["known_positive_control"]),
                               ("branches", {"recording": report["recording"], "repair": report["repair"], "decisions": states}),
                               ("sensitivities", report["sensitivities"])):
            progress["stage"] = stage
            write(out_dir / f"{stage}.json", payload)
        label, readings = conclusion(states, pc_flags)
        return {**report, "conclusion": label, "readings": readings, "decisions": states, "positive_control_flags": pc_flags}

    progress["stage"] = "primary_results"
    panel = Panel(nodes, "sbp", lam, J_min)
    observed = panel.observed()
    point, info = primary_statistics(observed, J_min)
    stat = lambda d: primary_statistics(d, J_min)[0]
    t0 = time.monotonic()
    draws = bootstrap(panel, stat, args.draws, seed=spec["uncertainty"]["seed"], workers=args.workers)
    np.savez_compressed(out_dir / "draws_primary.npz", **draws)
    b = primary_bounds(draws)
    report["timing_seconds"]["primary"] = time.monotonic() - t0
    states["extra_loss"] = combine([observed_decision(point["adjusted"], b["adjusted"], 0.10, "lower"),
                                    observed_decision(point["robust"], b["robust"], 0.10, "lower")])
    states["local_signal"] = combine([observed_decision(point["fractional_local"], b["fractional_local"], -0.15, "upper"),
                                      observed_decision(point["raw_transported"], b["raw_transported"], raw_margin, "upper")])
    report["primary"] = {"status": "evaluated", "point": point, "info": info, "bounds": b, "decisions": dict(states)}
    write(out_dir / "primary_results.json", report["primary"])

    # Known-positive control rerun on this participant's design (same injections, rules and draws).
    progress["stage"] = "known_positive_control"
    t0 = time.monotonic()
    control = {}
    inj_R, report_R = pc.extra_loss_injection(panel)
    inj_J, calibration = pc.local_signal_injection(panel)
    for name, variant in (("extra_loss", inj_R), ("local_signal", inj_J)):
        if variant.get("status") == "unresolved":
            control[name] = {"status": "unresolved", "reason": variant["reason"]}
            continue
        control[name] = {"status": "evaluated", "point": primary_statistics(panel.observed(variant), J_min)[0],
                         "bounds": primary_bounds(bootstrap(panel, stat, args.draws, seed=16, workers=args.workers, variant=variant))}
    control["extra_loss_pass"] = ({"pass": False, "status": "unresolved", "reason": inj_R["reason"]}
                                 if inj_R.get("status") == "unresolved" else
                                 pc.extra_loss_pass(point, control["extra_loss"]["point"], control["extra_loss"]["bounds"]))
    control["local_signal_pass"] = ({"pass": False, "status": "unresolved", "reason": inj_J["reason"]}
                                   if inj_J.get("status") == "unresolved" else
                                   pc.local_signal_pass(point, control["local_signal"]["point"],
                                                        control["local_signal"]["bounds"], raw_margin))
    control["local_signal_calibration"], control["extra_loss_injection"] = calibration, report_R
    pc_flags = {"extra_loss": control["extra_loss_pass"]["pass"], "local_signal": control["local_signal_pass"]["pass"]}
    report["known_positive_control"] = control
    report["timing_seconds"]["positive_control"] = time.monotonic() - t0
    write(out_dir / "known_positive_control.json", control)

    # Conditional mechanism branches: all tables and scores remain behind the extra-loss gate.
    progress["stage"] = "branches"
    emulator, method = panel_choice["emulator"], panel_choice["supervised_32"]
    long_rows = np.flatnonzero(panel.tables["long"])
    run_branches = states["extra_loss"] == "supported" or args.rehearsal_force_branches
    report["recording"] = {"status": "not_evaluated"}
    report["repair"] = {"status": "not_evaluated"}
    if run_branches:
        t0 = time.monotonic()
        em = br.emulator_tables(panel, emulator, array_labels, 20, seed=16)
        table, active = br.repair_table(panel, method)  # near and long edges for recalibrated contrasts
        rec_point, rec_info = br.recording_statistics(observed, em, J_min)
        rep_point, rep_info = br.repair_statistics(observed, table, J_min, D_min)
        report["recording"] = {"status": "evaluated", "point": rec_point, "info": rec_info}
        report["repair"] = {"status": "evaluated", "point": rep_point, "info": {**rep_info, "bound_active": active}}
        rec = bootstrap(panel, lambda d: br.recording_statistics(d, em, J_min)[0], args.draws, seed=161, workers=args.workers)
        rep = bootstrap(panel, lambda d: br.repair_statistics(d, table, J_min, D_min)[0], args.draws, seed=162, workers=args.workers)
        rb = {"identity_upper": one_sided_bound(rec["identity_max_reproduction_error"], 0.025, "upper"),
              "separation_lower": one_sided_bound(rec["random_minus_identity_max_error"], 0.025, "lower")}
        pb = {"recovery_lower": one_sided_bound(rep["recovery_fraction"], 0.025, "lower"),
              "gain_lower": one_sided_bound(rep["relative_gain"], 0.025, "lower")}
        report["recording"]["bounds"], report["repair"]["bounds"] = rb, pb
        report["timing_seconds"]["branches"] = time.monotonic() - t0
        if states["extra_loss"] == "supported":
            states["recording"] = combine([
                observed_decision(rec_point["identity_max_reproduction_error"], rb["identity_upper"], 0.10, "upper"),
                observed_decision(rec_point["random_minus_identity_max_error"], rb["separation_lower"], 0.05, "lower")])
            states["repair"] = combine([
                observed_decision(rep_point["recovery_fraction"], pb["recovery_lower"], 0.50, "lower"),
                observed_decision(rep_point["relative_gain"], pb["gain_lower"], 0.10, "lower")])
    write(out_dir / "branches.json", {"recording": report["recording"], "repair": report["repair"],
                                      "decisions": dict(states),
                                      "rehearsal_forced": bool(run_branches and states["extra_loss"] != "supported")})
    label, readings = conclusion(states, pc_flags)
    report.update(decisions=states, positive_control_flags=pc_flags, conclusion=label, readings=readings)

    # Mandatory sensitivities and controls (claim-narrowing only; sensitivity_draws where bounded).
    progress["stage"] = "sensitivities"
    t0 = time.monotonic()
    sens, sd = {}, args.sensitivity_draws

    def bounded(sub_panel, seed):
        return {"sessions": len(sub_panel.sessions), "point": primary_statistics(sub_panel.observed(), J_min)[0],
                "bounds": primary_bounds(bootstrap(sub_panel, stat, sd, seed=seed, workers=args.workers))}

    sens["tx_4_5"] = bounded(Panel(nodes, "tx_4_5", values["ridge_lambda_tx_4_5"], J_min), 163)
    sens["gap_14_365"] = bounded(Panel(nodes, "sbp", lam, J_min, near=(1, 14), long_min=365), 164)
    sens["reverse_pairs"] = bounded(Panel(nodes, "sbp", lam, J_min, direction="reverse"), 165)
    pure = [s for s in nodes if s.configuration_pure]
    try:
        pure_ok = len(pure) >= 9 and realized_design(pure, support_config)["conditional_numerical_pass"]
    except (ValueError, IndexError, np.linalg.LinAlgError):
        pure_ok = False
    sens["configuration_pure"] = bounded(Panel(pure, "sbp", lam, J_min), 166) if pure_ok else {
        "sessions": len(pure), "status": "configuration-pure design fails the component gates"}
    if frozen_scaling_sessions is not None:
        frozen_nodes = [s for s in frozen_scaling_sessions if s.day in {n.day for n in nodes}]
        lam_frozen = values["ridge_lambda_sbp_frozen_scaling"]
        frozen_panel = Panel(frozen_nodes, "sbp", lam_frozen, J_min)
        sens["frozen_source_only_scaling"] = {
            **bounded(frozen_panel, 169), "lambda": lam_frozen,
            "J_median": float(np.median(frozen_panel.tables["fit"]["J"]))}
    else:
        sens["frozen_source_only_scaling"] = {"status": "not_evaluated", "reason": "frozen-scaling sessions unavailable"}
    sens["survivor_electrodes"] = ("identical_to_primary" if all(s.available.all() for s in nodes)
                                   else "not_identical_requires_common_subset_refit")
    sens["shifted_label_null_J_q99"] = float(np.quantile(br.shifted_label_null(panel), 0.99))
    sens["permutation_null_J_q99"] = float(np.quantile(br.permutation_null(panel), 0.99))
    leak, valid_half = br.half_window_leakage(panel)
    sens["random_bin_leakage_inflation"] = float(np.mean(leak - valid_half))
    sens["random_trial_split_minus_block_split_L"] = float(np.nanmean(br.random_trial_split(panel, np.random.default_rng(167))
                                                                      - observed.local["L"]))
    long_obs = np.flatnonzero(observed.long)

    def long_gain(losses):
        gain = observed.edge_scores(losses, "target", long_obs) - observed.F[long_obs]
        finite = np.isfinite(gain)
        return (float(np.mean(gain[finite])) if finite.any() else None), int(finite.sum())

    selected = panel_choice["unlabelled"]
    unlabelled = (("selected", selected["method"], selected["settings"]),
                  ("past_only_H16", "past_only", {"history": 16}),
                  ("past_only_H64", "past_only", {"history": 64}),
                  ("whole_session", "whole_session", {}))
    report["unlabelled_reporting"] = {
        "past_only_finite_history_rule": "at least max(2, H//2) finite history values per channel; otherwise source-scaling fallback",
        "past_only_label": ("history_restricted_restandardization_of_block_normalized_features_not_literal_past_only"
                            if panel_choice["normalization"] == "block" else "past_only"),
        "methods": {}}
    cached = {}
    for tag, m, settings in unlabelled:
        key = (m, tuple(sorted(settings.items())))
        if key not in cached:
            cached[key] = long_gain(br.unlabelled_table(panel, m, rows=long_rows, **settings))
        gain, count = cached[key]
        sens[f"unlabelled_{tag}_long_mean_gain"] = gain
        report["unlabelled_reporting"]["methods"][tag] = {"method": m, "settings": settings,
                                                         "finite_long_pairs": count, "long_pairs": len(long_obs)}

    sens["shuffled_label_32_repair"] = {"status": "not_evaluated"}
    sens["emulator_theta0.5"] = {"status": "not_evaluated"}
    sens["repair_curves"] = {"status": "not_evaluated"}
    if run_branches:
        shuffled, _ = br.repair_table(panel, method, rows=long_rows, shuffle_seed=168)
        sens["shuffled_label_32_repair"] = br.repair_statistics(observed, shuffled, J_min, D_min)[0]
        if emulator.startswith("E2"):
            sens["emulator_theta0.5"] = br.recording_statistics(
                observed, br.emulator_tables(panel, "E2_theta0.5", array_labels, 20, seed=16), J_min)[0]
        else:
            sens["emulator_theta0.5"] = {"status": "not_evaluated", "reason": "selected emulator is E1_dropout"}
        curves = {"status": "evaluated", "method": method, "long_mean_gain_over_F": {}, "finite_long_pairs": {}}
        for n in (16, 32, 64):
            curve_table = table if n == 32 else br.repair_table(panel, method, n=n, rows=long_rows)[0]
            gain, count = long_gain(curve_table)
            curves["long_mean_gain_over_F"][n] = gain
            curves["finite_long_pairs"][n] = count
        Lt, Bt, Jt = observed.edge_target_scores()
        A32 = observed.edge_scores(table)
        recalibrated = {"adjusted": None, "robust": None}
        if (len(A32) and np.isfinite(A32).all() and np.isfinite(Lt).all()
                and np.isfinite(Jt).all() and np.all(Jt > J_min)):
            R_rec = (Lt - A32) / Jt
            recalibrated["adjusted"] = adjusted_contrast(observed.src, observed.tgt, observed.long, R_rec)
            recalibrated["robust"] = robust_contrast(observed.tgt, observed.long, R_rec)
        curves["recalibrated_32_near_long_contrasts"] = recalibrated
        curves["recalibrated_definition"] = "R_rec = (L_t - A_st(32))/J_t on all near and long edges"
        curves["near_pairs"], curves["long_pairs"] = int((~observed.long).sum()), int(observed.long.sum())
        sens["repair_curves"] = curves
    report["sensitivities"] = sens
    report["timing_seconds"]["sensitivities"] = time.monotonic() - t0
    write(out_dir / "sensitivities.json", {"sensitivities": sens, "unlabelled_reporting": report["unlabelled_reporting"],
                                          "draws": sd})
    return report


def main(argv=None):
    from freeze_t9_evaluation import RUNTIME_INPUTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", "1")))
    parser.add_argument("--draws", type=int, default=9999)
    parser.add_argument("--sensitivity-draws", type=int, default=1999)
    rehearsal_group = parser.add_mutually_exclusive_group()
    rehearsal_group.add_argument("--rehearsal-feature-dir", type=Path, help="development rehearsal only; never T9")
    rehearsal_group.add_argument("--rehearsal-from-archive", action="store_true", help="read development sessions from their archive")
    parser.add_argument("--rehearsal-participant", default="T11")
    parser.add_argument("--rehearsal-spec", type=Path, default=Path("outputs/config/analysis_specification.json"))
    parser.add_argument("--rehearsal-panel", type=Path)
    parser.add_argument("--rehearsal-force-branches", action="store_true", help="rehearsal only: bootstrap the branches regardless")
    args = parser.parse_args(argv)
    rehearsal = args.rehearsal_feature_dir is not None or args.rehearsal_from_archive
    if args.workers <= 0 or args.draws <= 0 or args.sensitivity_draws <= 0:
        parser.error("--workers, --draws and --sensitivity-draws must be positive")
    if not rehearsal and (args.freeze_manifest is None or args.source_root is None):
        parser.error("sealed mode requires --freeze-manifest and --source-root")
    if not rehearsal and args.rehearsal_force_branches:
        parser.error("--rehearsal-force-branches is for development rehearsals only")
    if rehearsal and args.rehearsal_panel is None:
        parser.error("a rehearsal requires --rehearsal-panel")
    if args.rehearsal_from_archive and args.source_root is None:
        parser.error("--rehearsal-from-archive requires --source-root")
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise SystemExit("Compute allocation required")
    started = time.monotonic()
    args.output_dir = args.output_dir.absolute()
    if args.output_dir.is_symlink():
        parser.error("--output-dir must not be a symlink")
    if rehearsal and (args.output_dir.resolve() == EVALUATION_ROOT.resolve()
                      or EVALUATION_ROOT.resolve() in args.output_dir.resolve().parents):
        parser.error("rehearsals must write outside the T9 evaluation root")
    structural, support = [EPISODE_ROOT / path for path in RUNTIME_INPUTS]
    support_config = json.loads((EPISODE_ROOT / "outputs/config/schedule_support.json").read_text())
    if rehearsal:
        participant = args.rehearsal_participant
        authorize_protected_read(participant)  # development only; T9 is refused here
        spec = json.loads(args.rehearsal_spec.read_text())
        panel_choice = json.loads(args.rehearsal_panel.read_text())
        problems = panel_choice_errors(panel_choice)
        if problems:
            parser.error("; ".join(problems))
        if args.rehearsal_from_archive:
            archive = args.source_root / "source" / f"decoding_{participant}.tar.gz"
            if not archive.is_file():
                parser.error(f"Development archive is missing: {archive}")
        mode = "rehearsal_on_development_data"
    else:
        manifest, spec, panel_choice, _ = verify_manifest(args.freeze_manifest)
        if args.draws != spec["uncertainty"]["draws"]:
            parser.error("--draws must equal the frozen uncertainty.draws")
        if args.sensitivity_draws != spec["uncertainty"]["sensitivity_draws"]:
            parser.error("--sensitivity-draws must equal the frozen uncertainty.sensitivity_draws")
        archive = args.source_root / "source" / "decoding_T9.tar.gz"
        if archive.resolve() != Path(manifest["t9_archive"]["path"]).resolve():
            parser.error("--source-root does not identify the frozen T9 archive")
        participant = "T9"
        mode = "sealed_single_T9_evaluation"
    days = passing_component_days(support, participant)
    records = structural_records(structural, participant, days)  # load before sealing
    normalization = panel_choice["normalization"]
    scratch = args.scratch_dir.resolve()
    output = args.output_dir.resolve()
    if scratch == output or output in scratch.parents:
        parser.error("--scratch-dir must be outside --output-dir")
    try:
        args.scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryFile(dir=args.scratch_dir) as probe:
            probe.write(b"\0")
            probe.flush()
    except OSError as error:
        parser.error(f"--scratch-dir is not writable: {error}")

    progress = {"stage": "features"}
    if rehearsal:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        token = None
    else:
        token = open_seal(args.output_dir, args.freeze_manifest)
    try:
        if rehearsal and not args.rehearsal_from_archive:
            features = load_feature_dir(args.rehearsal_feature_dir, participant, days)
        else:
            features = read_archive_sessions(args.source_root, participant, days, token, args.scratch_dir)
        progress["stage"] = "packets_and_design"
        sessions, packets, axis = assemble(records, features, normalization=normalization)
        frozen = assemble(records, features, normalization="none")[0] if normalization != "none" else sessions
        design = realized_design(sessions, support_config)
        array_labels = np.array([a for a, _, _ in axis])
        report = evaluate(sessions, packets, design, spec, panel_choice, args, args.output_dir, array_labels, frozen, progress)
        progress["stage"] = "evaluation_report"
        report["normalization"] = normalization
        report.update(mode=mode, participant=participant, slurm_job_id=os.environ["SLURM_JOB_ID"], hostname=socket.gethostname(),
                      draws=args.draws, sensitivity_draws=args.sensitivity_draws, workers=args.workers,
                      finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        with open(args.output_dir / "evaluation_report.json", "x") as handle:
            handle.write(json.dumps(jsonable(report), indent=1) + "\n")
        print(json.dumps({"mode": mode, "conclusion": report.get("conclusion"), "elapsed": round(report["elapsed_seconds"], 1)}))
    except BaseException as error:
        record_failure(args.output_dir, progress["stage"], error)
        raise


if __name__ == "__main__":
    main()
