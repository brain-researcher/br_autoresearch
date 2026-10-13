#!/usr/bin/env python3
"""EP01 v6 original-event qualification; never fit a model or read BOLD values.

The field and category mapping is a source-documentation decision, supplied
explicitly rather than guessed from numeric response codes or neural results.
Run the command in a Slurm allocation. Inputs are only read, and each attempt
uses a new output directory so a failed attempt cannot replace earlier evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tarfile


class SourceError(ValueError):
    pass


MISSING = {"", "n/a", "na", "nan", "none"}
CATEGORIES = {"strong_accept": (1, 1), "weak_accept": (1, 0),
              "weak_reject": (0, 0), "strong_reject": (0, 1),
              "no_response": (-1, -1)}
EVENT_FIELDS = ["participant", "task_version", "run", "trial_order", "onset",
                "gain", "loss", "response_category", "choice", "strength",
                "response_time", "no_response", "sequence_template"]


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Source qualification must run inside a verified Slurm allocation")


def read_tsv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(f"Invalid numeric {name}: {value!r}") from exc
    if not math.isfinite(value):
        raise SourceError(f"Nonfinite {name}")
    return value


def normalize_trial(row, mapping, participant, version, run, index):
    """Apply only the documented source mapping; retain deadline misses."""
    fields = mapping["fields"]
    gain = number(row[fields["gain"]], "gain")
    loss = number(row[fields["loss"]], "loss") * mapping["loss_multiplier"]
    onset = number(row[fields["onset"]], "onset")
    if gain <= 0 or loss <= 0 or onset < 0:
        raise SourceError("Offers must be positive magnitudes and onsets nonnegative")
    if "duration" in fields and number(row[fields["duration"]], "duration") != 4.0:
        raise SourceError("Original event duration does not match the four-second offer/deadline")
    source_response = row[fields["response"]].strip()
    if source_response not in mapping["response_categories"]:
        raise SourceError(f"Unmapped source response {source_response!r}")
    category = mapping["response_categories"][source_response]
    if category not in CATEGORIES:
        raise SourceError(f"Unknown canonical response category {category!r}")
    choice, strength = CATEGORIES[category]
    raw_rt = row[fields["response_time"]].strip()
    if category == "no_response":
        # A source may explicitly encode a missed response with a sentinel RT.
        # It is not promoted to a response or recoded as rejection.
        if raw_rt.lower() not in MISSING:
            rt_value = number(raw_rt, "miss RT")
            if rt_value not in mapping.get("no_response_rt_values", []):
                raise SourceError(f"Undocumented missed-response RT {raw_rt!r}")
        rt = ""
    else:
        rt = number(raw_rt, "response_time") * mapping.get("rt_seconds_multiplier", 1.0)
        if rt <= 0 or rt > 4.0:
            raise SourceError(f"Responded trial RT outside (0, 4] seconds: {rt}")
    return dict(participant=participant, task_version=version, run=run,
                trial_order=index, onset=onset, gain=gain, loss=loss,
                response_category=category, choice=choice, strength=strength,
                response_time=rt, no_response=int(choice == -1),
                sequence_template="")


def qualify_events(source_root, mapping):
    participants_path = source_root / "participants.tsv"
    participants = read_tsv(participants_path)
    if len(participants) != 108:
        raise SourceError(f"Expected 108 participant rows; observed {len(participants)}")
    groups = mapping["task_versions"]
    group_field = mapping["participant_group_field"]
    all_events, inventory, participant_versions = [], [], {}
    seen = set()
    for participant_row in participants:
        participant = participant_row["participant_id"]
        if not re.fullmatch(r"sub-[A-Za-z0-9]+", participant) or participant in seen:
            raise SourceError(f"Invalid or duplicate participant identifier {participant!r}")
        seen.add(participant)
        value = participant_row[group_field]
        if value not in groups:
            raise SourceError(f"Undocumented task version {value!r}")
        version = groups[value]
        if version not in {"EI", "ER"}:
            raise SourceError(f"Unknown canonical task version {version!r}")
        participant_versions[participant] = version
        # Positive allowlist: only this participant's original MGT event tables.
        paths = sorted((source_root / participant / "func").glob(
            f"{participant}_task-MGT_run-*_events_ORIGINAL.tsv"))
        if len(paths) != 4:
            raise SourceError(f"{participant}: expected four original event files, got {len(paths)}")
        participant_events, runs = [], set()
        for path in paths:
            match = re.search(r"_run-(\d+)_events_ORIGINAL\.tsv$", path.name)
            run = int(match.group(1))
            if run not in {1, 2, 3, 4} or run in runs:
                raise SourceError(f"{participant}: invalid or repeated run {run}")
            runs.add(run)
            rows = read_tsv(path)
            if len(rows) != 64:
                raise SourceError(f"{path.name}: expected 64 trials, observed {len(rows)}")
            if not set(mapping["fields"].values()).issubset(rows[0]):
                raise SourceError(f"{path.name}: required documented fields are absent")
            events = [normalize_trial(row, mapping, participant, version, run, i + 1)
                      for i, row in enumerate(rows)]
            if any(b["onset"] <= a["onset"] for a, b in zip(events, events[1:])):
                raise SourceError(f"{path.name}: trial onsets are not increasing")
            participant_events.extend(events)
            inventory.append(dict(participant=participant, run=run, path=str(path),
                                  trials=len(events), missing_responses=sum(e["no_response"] for e in events)))
        offers = {(e["gain"], e["loss"]) for e in participant_events}
        if len(offers) != 256:
            raise SourceError(f"{participant}: expected 256 distinct offer pairs, got {len(offers)}")
        gains = {g for g, _ in offers}
        losses = {loss for _, loss in offers}
        expected_gains = set(range(10, 41, 2)) if version == "EI" else set(range(5, 21))
        if gains != expected_gains or losses != set(range(5, 21)):
            raise SourceError(f"{participant}: offer levels disagree with documented {version} schedule")
        all_events.extend(participant_events)
    version_counts = Counter(participant_versions.values())
    if version_counts != {"EI": 54, "ER": 54}:
        raise SourceError(f"Expected EI=54 and ER=54, observed {dict(version_counts)}")
    # Schedule identity uses only offer order and onsets, never choice or RT.
    # Keep exact observed schedules; report deviations from eight rather than
    # coarsening timing or merging sequences after observing model outcomes.
    schedules = defaultdict(dict)
    by_participant = defaultdict(list)
    for event in all_events:
        by_participant[event["participant"]].append(event)
    for participant in sorted(by_participant):
        events = by_participant[participant]
        version = events[0]["task_version"]
        schedule = tuple((e["run"], e["trial_order"], e["gain"], e["loss"], e["onset"])
                         for e in sorted(events, key=lambda e: (e["run"], e["trial_order"])))
        if schedule not in schedules[version]:
            schedules[version][schedule] = f"{version}-{len(schedules[version]) + 1:02d}"
        for event in events:
            event["sequence_template"] = schedules[version][schedule]
    return all_events, inventory, {v: len(s) for v, s in schedules.items()}


def describe_fmriprep(root, participants):
    """File availability only; no neural, confound, mask or image arrays read."""
    records = []
    for participant in sorted(participants):
        for run in range(1, 5):
            entry = {"participant": participant, "run": run}
            for label, suffix in (("bold", "desc-preproc_bold.nii.gz"),
                                  ("mask", "desc-brain_mask.nii.gz"),
                                  ("confounds", "desc-confounds_timeseries.tsv")):
                # fMRIPrep derivatives can include a named pipeline directory.
                bases = [root / participant / "func", root / "fmriprep" / participant / "func"]
                found = []
                for base in bases:
                    for path in base.glob(f"{participant}_task-MGT_run-*_{suffix}"):
                        match = re.search(r"_run-(\d+)_", path.name)
                        if not match or int(match.group(1)) != run:
                            continue
                        if label != "confounds" and "space-MNI152NLin2009cAsym_res-2_" not in path.name:
                            continue
                        if path.is_file():
                            found.append(str(path))
                entry[label] = found
            records.append(entry)
    return records


def describe_team_archive(path):
    """Read archive headers only; never extract maps or calculate team scores."""
    relevant = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            lower = member.name.lower()
            if member.isfile() and ("metadata" in lower or "unthresh" in lower or
                                    "collection" in lower or "hypoth" in lower):
                relevant.append({"member": member.name, "bytes": member.size})
    return relevant


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--fmriprep-root", type=Path)
    parser.add_argument("--team-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory for each attempt")
    args.output_dir.mkdir(parents=True)
    report = {"analysis_id": "public_asset_qualification", "policy_version": 6,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "source_root": str(args.source_root), "source_mapping": str(args.source_mapping),
              "behavioral_model_fit": False, "bold_values_read": False, "team_values_read": False}
    try:
        mapping = json.loads(args.source_mapping.read_text())
        if mapping.get("loss_multiplier") not in {-1, 1} or not mapping.get("documentation"):
            raise SourceError("An explicit documented sign/category mapping is required")
        events, files, schedules = qualify_events(args.source_root, mapping)
        with gzip.open(args.output_dir / "original_events.csv.gz", "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(events)
        report.update(participant_behavior_ready=True, participants=108, runs=len(files),
                      trials=len(events), no_responses=sum(e["no_response"] for e in events),
                      response_counts=dict(Counter(e["response_category"] for e in events)),
                      exact_schedule_counts=schedules, expected_schedule_counts={"EI": 8, "ER": 8},
                      event_files=files, event_table="original_events.csv.gz")
        if args.fmriprep_root:
            records = describe_fmriprep(args.fmriprep_root, {e["participant"] for e in events})
            report["fmriprep_files"] = records
            report["fmriprep_unique_complete_runs"] = sum(
                all(len(row[k]) == 1 for k in ("bold", "mask", "confounds")) for row in records)
        if args.team_archive:
            report["team_archive"] = str(args.team_archive)
            report["team_archive_relevant_headers"] = describe_team_archive(args.team_archive)
        report["status"] = "behavioral_source_qualified"
        report["remaining_before_behavioral_kill_test"] = [
            "instantiate_and_qualify_hierarchical_deadline_DDM_and_both_mandatory_rivals",
            "record_recovery_regimes_priors_contaminant_channel_and_replicate_count_before_scoring"]
        if schedules != {"EI": 8, "ER": 8}:
            report["remaining_before_behavioral_kill_test"].append(
                "resolve_exact_schedule_identity_against_the_eight_source_templates_without_outcome_based_merging")
    except Exception as exc:
        report.update(status="source_qualification_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "asset_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k in {
            "analysis_id", "status", "participants", "runs", "trials", "no_responses", "error"}}, indent=2))


if __name__ == "__main__":
    main()
