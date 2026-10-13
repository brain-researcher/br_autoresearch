#!/usr/bin/env python3
"""Resolve nominal offer schedules without using choice, RT or neural values.

The first source pass intentionally kept exact measured onsets, so scanner
timing differences could make nominally repeated schedules appear unique.
This follow-up uses ordered offer identities only and reports all timing
differences instead of rounding them to manufacture eight groups.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import os
from pathlib import Path

from qualify_assets import require_compute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.output_dir.exists():
        raise FileExistsError("Use a new attempt directory")
    by_participant = defaultdict(list)
    with gzip.open(args.events, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            # Deliberately retain only design covariates from the source row.
            by_participant[row["participant"]].append({
                "run": int(row["run"]), "trial": int(row["trial_order"]),
                "gain": float(row["gain"]), "loss": float(row["loss"]),
                "onset": float(row["onset"]), "version": row["task_version"]})
    groups = {"EI": {}, "ER": {}}
    for participant in sorted(by_participant):
        rows = sorted(by_participant[participant], key=lambda r: (r["run"], r["trial"]))
        if len(rows) != 256 or len({(r["run"], r["trial"]) for r in rows}) != 256:
            raise ValueError("A full four-run participant design is required")
        versions = {r["version"] for r in rows}
        if len(versions) != 1:
            raise ValueError("Participant's task version is inconsistent")
        version = versions.pop()
        key = tuple((r["run"], r["trial"], r["gain"], r["loss"]) for r in rows)
        if key not in groups[version]:
            groups[version][key] = {"offer_sequence_id": f"{version}-offers-{len(groups[version])+1:02d}",
                                   "participants": [], "onsets": []}
        groups[version][key]["participants"].append(participant)
        groups[version][key]["onsets"].append([r["onset"] for r in rows])
    report = {"analysis_id": "public_asset_qualification", "job_id": os.environ["SLURM_JOB_ID"],
              "source_event_table": str(args.events), "choice_RT_neural_values_used": False,
              "grouping": "exact_whole_participant_ordered_run_trial_gain_loss",
              "offer_only_template_counts": {v: len(g) for v, g in groups.items()},
              "expected_nominal_templates_per_version": 8, "groups": []}
    memberships = []
    for version, collection in groups.items():
        for group in collection.values():
            onsets = group.pop("onsets")
            group["maximum_absolute_onset_difference_seconds"] = max(
                max(values)-min(values) for values in zip(*onsets))
            normalized = [[t-row[(j//64)*64] for j, t in enumerate(row)] for row in onsets]
            group["maximum_difference_after_run_onset_alignment_seconds"] = max(
                max(values)-min(values) for values in zip(*normalized))
            report["groups"].append(group)
            memberships.extend({"participant": p, "task_version": version,
                                "offer_sequence_id": group["offer_sequence_id"]}
                               for p in group["participants"])
    report["matches_documented_eight_offer_sequences"] = all(len(g) == 8 for g in groups.values())
    report["interpretation"] = (
        "Eight exact ordered offer sequences per version; measured onsets remain participant-specific. "
        "The attached memberships are a design-only candidate for the registered template-cluster sensitivity."
        if report["matches_documented_eight_offer_sequences"] else
        "The offered sequence identities do not recover the documented eight templates. "
        "Provider schedule assignments are still needed; do not merge or relabel groups using outcomes.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"schedule_support.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output_dir/"offer_sequence_membership.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["participant", "task_version", "offer_sequence_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(memberships)
    print(json.dumps({"offer_only_template_counts": report["offer_only_template_counts"],
                      "matches_documented_eight_offer_sequences": report["matches_documented_eight_offer_sequences"]}))


if __name__ == "__main__":
    main()
