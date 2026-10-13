#!/usr/bin/env python3
"""Qualify EP17 source-history ordering from public projected metadata only."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
import os
from pathlib import Path
import signal
import socket
import time

import qualify_source_roles as stage
import reconcile_released_events as core


QUALIFICATION = core.EPISODE / "outputs/runs/source_qualification/attempt002"
OUTPUT_PARENT = core.EPISODE / "outputs/runs/source_chronology"
INTERNAL_SECONDS = 120
RUN_WIDTH = 60
KNOWN_SUB01_SES17 = (1, 2, 3, 5, 6, 4)
RUN_FIELDS = ("participant", "session", "run", "component_id", "component_coordinates_qualified",
              "component_source_history_start", "global_source_history_start", "global_order_qualified")
ADDED_FIELDS = ("eligible_image", "retained_repeat_rank", "repeat_half", "ordinal_status", "component_id",
                "component_source_history_position", "global_source_history_position", "retained_acquisition_rank")


class Issues:
    def __init__(self):
        self.counts = Counter()
        self.examples = []

    def add(self, code, **coordinates):
        self.counts[code] += 1
        if len(self.examples) < 50:
            self.examples.append({"code": code, **coordinates})


def load_eligibility(rows):
    report = json.loads(stage.bounded_read(QUALIFICATION / "qualification_report.json"))
    core.require(report.get("status") == "completed" and report.get("eligibility_complete") is True and
                 report.get("eligible_720_concept_universe_verified") is True and
                 report.get("qualified_concept_count") == 720 and report.get("errors") == [] and
                 report.get("source_roles_complete") is False and report.get("roles_assigned") is False and
                 report.get("source_metadata_attempt") == str(core.METADATA) and
                 report.get("event_projection_attempt") == str(stage.EVENTS), "completed_eligibility_required")
    _, images = stage.table_rows(stage.bounded_read(QUALIFICATION / "image_qualification.csv"), stage.IMAGE_FIELDS)
    observed, counts = {}, defaultdict(Counter)
    for row in rows:
        path = row["expected_provider_image_path"]
        identity = (row["source_image_basename"], row["concept_name"], row["things_category_nr"],
                    row["provider_category_nr"], row["provider_exemplar_nr"], row["condition"])
        core.require(observed.setdefault(path, identity) == identity, "observed_identity_conflict")
        counts[path][row["participant"]] += 1
    labels = {}
    for item in images:
        path = item["provider_image_path"]
        core.require(path in observed and path not in labels and item["eligible_image"] in ("True", "False"),
                     "eligibility_image_scope_changed")
        core.require(tuple(item[field] for field in stage.IDENTITY_FIELDS[1:]) == observed[path],
                     "eligibility_image_identity_changed")
        core.require(all(core.integer(item[f"retained_count_{s}"], "retained_count") == counts[path][s]
                         for s in stage.SUBJECTS), "eligibility_retained_counts_changed")
        labels[path] = item["eligible_image"] == "True"
        core.require(not labels[path] or all(counts[path][s] >= 2 for s in stage.SUBJECTS),
                     "qualified_image_lacks_required_repeats")
    core.require(set(labels) == set(observed) and sum(labels.values()) == report.get("qualified_image_count"),
                 "eligibility_projection_incomplete")
    return labels


def repeat_metadata(rows, eligible, issues):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["participant"], row["expected_provider_image_path"])].append(row)
    mapped, edges = {}, defaultdict(list)
    valid_groups, invalid_subjects = set(), set()
    stats = Counter()
    for group_key, group in sorted(groups.items()):
        subject, image = group_key
        values = [None if row["source_repetition"] == "" else
                  core.integer(row["source_repetition"], "source_repetition") for row in group]
        valid = all(value is not None and value > 0 for value in values) and len(set(values)) == len(values)
        reason = "valid"
        if not valid:
            reason = "unknown_or_nonunique_source_ordinal"
        else:
            ordered = sorted(zip(values, group), key=lambda pair: pair[0])
            per_run = defaultdict(list)
            for _, row in ordered:
                per_run[stage.run_key(row)].append(core.integer(row["source_order"], "source_order"))
            if any(any(a >= b for a, b in zip(order, order[1:])) for order in per_run.values()):
                valid = False
                reason = "ordinal_within_run_order_conflict"
        if not valid:
            invalid_subjects.add(subject)
            issues.add(reason, participant=subject, image_path=image)
            for row in group:
                mapped[stage.key(row)] = {"retained_repeat_rank": "", "repeat_half": "", "ordinal_status": reason}
            continue
        valid_groups.add(group_key)
        for rank, (_, row) in enumerate(ordered):
            mapped[stage.key(row)] = {"retained_repeat_rank": rank,
                                     "repeat_half": 1 + rank % 2 if eligible[image] else "",
                                     "ordinal_status": "valid"}
        for (previous_ordinal, previous), (current_ordinal, current) in zip(ordered, ordered[1:]):
            if current_ordinal != previous_ordinal + 1:
                stats["skipped_nonconsecutive_source_ordinal_pairs"] += 1
                continue
            if current["trials_since_lastrep"] == "":
                stats["skipped_unknown_trial_gap_pairs"] += 1
                continue
            gap = core.integer(current["trials_since_lastrep"], "trials_since_lastrep")
            if gap <= 0:
                issues.add("nonpositive_history_trial_gap", participant=subject,
                           trial_index=stage.key(current)[1])
                invalid_subjects.add(subject)
                continue
            delta = gap + core.integer(previous["source_order"], "source_order") - core.integer(current["source_order"], "source_order")
            edges[subject].append((stage.run_key(previous), stage.run_key(current), delta,
                                   stage.key(previous)[1], stage.key(current)[1]))
    half_complete = True
    for image, selected in eligible.items():
        if selected:
            for subject in stage.SUBJECTS:
                key = subject, image
                if key not in valid_groups or len(groups.get(key, [])) < 2:
                    half_complete = False
                    issues.add("qualified_image_half_unresolved", participant=subject, image_path=image)
    return mapped, edges, groups, valid_groups, invalid_subjects, half_complete, dict(stats)


def solve_runs(subject, nodes, edges, issues):
    """Integer potentials; lexical order names components, never global time."""
    adjacency = {node: [] for node in nodes}
    for before, after, delta, _, _ in edges:
        core.require(before in adjacency and after in adjacency, "constraint_run_outside_roster")
        adjacency[before].append((after, delta))
        adjacency[after].append((before, -delta))
    potentials, component_of, components = {}, {}, []
    for origin in sorted(nodes):
        if origin in potentials:
            continue
        number = len(components)
        component = []
        queue = deque([origin])
        potentials[origin] = 0
        component_of[origin] = number
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor, delta in adjacency[node]:
                if neighbor not in potentials:
                    potentials[neighbor] = potentials[node] + delta
                    component_of[neighbor] = number
                    queue.append(neighbor)
        components.append(component)
    invalid = set()
    for before, after, delta, old_index, new_index in edges:
        if potentials[after] - potentials[before] != delta:
            invalid.add(component_of[before])
            issues.add("inconsistent_run_offset_constraint", participant=subject,
                       previous_trial_index=old_index, current_trial_index=new_index)
    for number, component in enumerate(components):
        shift = min(potentials[node] for node in component)
        for node in component:
            potentials[node] -= shift
        ordered = sorted(component, key=lambda node: (potentials[node], node))
        for previous, current in zip(ordered, ordered[1:]):
            if potentials[current] < potentials[previous] + RUN_WIDTH:
                invalid.add(number)
                issues.add("overlapping_run_intervals", participant=subject,
                           previous_session_run=list(previous[1:]), current_session_run=list(current[1:]))
    exception_status = "not_applicable"
    if subject == "sub-01":
        known_nodes = [(subject, 17, run) for run in KNOWN_SUB01_SES17]
        if not all(node in potentials for node in known_nodes):
            exception_status = "unresolved"
            issues.add("documented_exception_run_missing", participant=subject)
        elif len({component_of[node] for node in known_nodes}) != 1 or component_of[known_nodes[0]] in invalid:
            exception_status = "unresolved"
        else:
            actual = tuple(node[2] for node in sorted(known_nodes, key=lambda node: potentials[node]))
            exception_status = "matched" if actual == KNOWN_SUB01_SES17 else "contradiction"
            if exception_status == "contradiction":
                issues.add("documented_run_order_exception_disagrees", participant=subject)
                invalid.add(component_of[known_nodes[0]])
    if len(components) != 1:
        issues.add("disconnected_run_graph", participant=subject, components=len(components))
    qualified = len(components) == 1 and not invalid and exception_status in ("matched", "not_applicable")
    runs = {}
    for node in sorted(nodes):
        component_valid = component_of[node] not in invalid
        runs[node] = {"participant": subject, "session": node[1], "run": node[2],
                      "component_id": f"{subject}:component{component_of[node]:03d}",
                      "component_coordinates_qualified": component_valid,
                      "component_source_history_start": potentials[node] if component_valid else "",
                      "global_source_history_start": potentials[node] if qualified else "",
                      "global_order_qualified": qualified}
    summary = {"participant": subject, "retained_runs": len(nodes), "constraint_edges": len(edges),
               "connected_components": len(components), "invalid_components": len(invalid),
               "known_run_order_exception": exception_status, "global_order_qualified": qualified}
    return runs, summary


def chronology(rows, eligible):
    issues = Issues()
    mapped, edges, groups, valid_groups, invalid_subjects, half_complete, stats = repeat_metadata(rows, eligible, issues)
    runs, participants = {}, []
    for subject in stage.SUBJECTS:
        nodes = {stage.run_key(row) for row in rows if row["participant"] == subject}
        core.require(bool(nodes), "missing_chronology_participant")
        subject_runs, summary = solve_runs(subject, nodes, edges[subject], issues)
        # Compare image histories within every qualified component, including
        # disconnected graphs. Different components have no comparable origin.
        contradicted_components = set()
        for group_key, group in groups.items():
            if group_key[0] != subject or group_key not in valid_groups:
                continue
            ordered = sorted(group, key=lambda row: mapped[stage.key(row)]["retained_repeat_rank"])
            component_positions = defaultdict(list)
            for row in ordered:
                record = subject_runs[stage.run_key(row)]
                if record["component_coordinates_qualified"]:
                    component_positions[record["component_id"]].append(
                        record["component_source_history_start"] + core.integer(row["source_order"], "source_order"))
            bad_components = {component for component, positions in component_positions.items()
                              if any(a >= b for a, b in zip(positions, positions[1:]))}
            if bad_components:
                contradicted_components.update(bad_components)
                invalid_subjects.add(subject)
                issues.add("ordinal_component_order_conflict", participant=subject, image_path=group_key[1])
                if eligible[group_key[1]]:
                    half_complete = False
                for row in group:
                    mapped[stage.key(row)].update({"retained_repeat_rank": "", "repeat_half": "",
                                                  "ordinal_status": "ordinal_component_order_conflict"})
        for record in subject_runs.values():
            if record["component_id"] in contradicted_components:
                record["component_coordinates_qualified"] = False
                record["component_source_history_start"] = ""
        summary["ordinal_conflicted_components"] = len(contradicted_components)
        summary["invalid_components"] += len(contradicted_components)
        if subject in invalid_subjects or contradicted_components:
            summary["global_order_qualified"] = False
        if not summary["global_order_qualified"]:
            for record in subject_runs.values():
                record["global_source_history_start"] = ""
                record["global_order_qualified"] = False
        runs.update(subject_runs)
        participants.append(summary)
    rank = {}
    for subject in stage.SUBJECTS:
        selected = [row for row in rows if row["participant"] == subject]
        if next(x for x in participants if x["participant"] == subject)["global_order_qualified"]:
            selected.sort(key=lambda row: runs[stage.run_key(row)]["global_source_history_start"] +
                          core.integer(row["source_order"], "source_order"))
            rank.update({stage.key(row): index for index, row in enumerate(selected)})
    output = []
    for row in rows:
        record = runs[stage.run_key(row)]
        within_run = core.integer(row["source_order"], "source_order")
        component_start, global_start = record["component_source_history_start"], record["global_source_history_start"]
        output.append({**row, "eligible_image": eligible[row["expected_provider_image_path"]], **mapped[stage.key(row)],
                       "component_id": record["component_id"],
                       "component_source_history_position": component_start + within_run if component_start != "" else "",
                       "global_source_history_position": global_start + within_run if global_start != "" else "",
                       "retained_acquisition_rank": rank.get(stage.key(row), "")})
    return output, list(runs.values()), {
        "retained_half_qualification_complete": half_complete,
        "global_chronology_complete": all(x["global_order_qualified"] for x in participants),
        "all_observed_ordinals_qualified": not invalid_subjects and not any(issues.counts[code] for code in (
            "inconsistent_run_offset_constraint", "overlapping_run_intervals", "documented_run_order_exception_disagrees")),
        "participants": participants, "issue_counts": dict(issues.counts), "issue_examples": issues.examples,
        "issue_examples_maximum": 50, **stats,
    }


def run_attempt(out):
    started = time.monotonic()
    report = {"analysis_id": "source_chronology", "status": "running", "source_chronology_complete": False,
              "retained_half_qualification_complete": False, "global_chronology_complete": False,
              "source_roles_complete": False, "roles_assigned": False, "folds_created": False,
              "neural_values_read": False, "behavioral_outcomes_read": False, "image_values_read": False,
              "model_features_read": False, "candidate_scoring_run": False, "audit_opened": False,
              "full_exposure_history_complete": False, "reconstructed_positions_are_elapsed_time": False,
              "source_metadata_attempt": str(core.METADATA), "event_projection_attempt": str(stage.EVENTS),
              "eligibility_attempt": str(QUALIFICATION), "events_commit": core.EVENT_COMMIT,
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 180,
              "input_file_byte_cap": stage.FILE_CAP, "network_requests": 0, "automatic_retries": 0,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "errors": []}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    try:
        rows = stage.load_sources()
        eligible = load_eligibility(rows)
        presentations, runs, result = chronology(rows, eligible)
        report.update(result)
        report["preserved_presentations"] = len(presentations)
        report["preserved_runs"] = len(runs)
        fields = (*core.OUTPUT_FIELDS, "concept_name", "condition", *ADDED_FIELDS)
        stage.write_table(out / "presentation_chronology.csv", fields, presentations)
        stage.write_table(out / "run_chronology.csv", RUN_FIELDS, runs)
        core.require(time.monotonic() < started + INTERNAL_SECONDS, "attempt_deadline")
        if result["retained_half_qualification_complete"] and result["global_chronology_complete"]:
            report.update({"source_chronology_complete": True, "status": "completed"})
        else:
            report.update({"status": "unresolved", "source_chronology_complete": False})
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report.update({"status": "stopped", "source_chronology_complete": False,
                       "retained_half_qualification_complete": False, "global_chronology_complete": False})
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        report["errors"].append({"code": "source_chronology_processing_failure"})
        report.update({"status": "stopped", "source_chronology_complete": False,
                       "retained_half_qualification_complete": False, "global_chronology_complete": False})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - started
    with (out / "chronology_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "retained_half_qualification_complete": report["retained_half_qualification_complete"],
                      "global_chronology_complete": report["global_chronology_complete"],
                      "source_chronology_complete": report["source_chronology_complete"]}))
    return 0 if report["status"] == "completed" and report["source_chronology_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(),
                 "requires_coordinator_owned_compute")
    out = args.out.resolve()
    core.require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    return run_attempt(out)


if __name__ == "__main__":
    raise SystemExit(main())
