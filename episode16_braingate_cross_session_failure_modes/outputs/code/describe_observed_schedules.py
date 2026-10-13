#!/usr/bin/env python3
"""EP16 structural-only target recurrence and observed transition diagnostic.

The sole data input is the existing authorized structural JSONL. Retains every
declared trial and its row identity. Uses only the previous roundoff descriptor
for alignment; no proposed scientific tolerance/support threshold is applied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import cKDTree

from describe_schedule_design import correspondence, geometry
from describe_structural_support import array_labels
from mat5_structural import field_vector


def write_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def transition_rows(counts):
    return [{"source_target_id": int(a), "destination_target_id": int(b),
             "count": int(count)} for (a, b), count in sorted(counts.items())]


def hub_candidates(edges):
    """Observed edges only: a hub does not identify a task or forbidden edges."""
    nonself = [(a, b) for a, b in edges if a != b]
    if not nonself:
        return []
    common = set(nonself[0])
    for edge in nonself[1:]:
        common.intersection_update(edge)
    return sorted(common)


def prepare(record, block_stream):
    fields, summary = record["fields"], record["summary"]
    shape = geometry(fields["trial_targets"])
    if shape is None:
        raise ValueError("Degenerate target layout requires explicit diagnosis")
    blocks = field_vector(fields["trials"], "block_id").astype(np.int64)
    starts = field_vector(fields["trials"], "trial_start_index").astype(np.int64)
    ends = field_vector(fields["trials"], "trial_end_index").astype(np.int64)
    declared_blocks = field_vector(fields["blocks"], "block_id").astype(np.int64)
    if not len(blocks) == len(starts) == len(ends) == len(shape["inverse"]):
        raise ValueError("Structural trial vectors differ in length")
    if len(set(declared_blocks.tolist())) != len(declared_blocks):
        raise ValueError("Duplicate declared block identity")
    if not set(blocks).issubset(set(declared_blocks)):
        raise ValueError("Trial refers to an undeclared block")
    identity = {"participant": summary["participant"], "day": summary["post_implant_day"],
                "physical_array_set": array_labels(fields["electrodes"])}
    all_transitions, block_sets, block_summaries = Counter(), [], []
    retained = []
    for block in sorted(declared_blocks.tolist()):
        rows = np.flatnonzero(blocks == block)
        rows = rows[np.argsort(starts[rows], kind="stable")]
        retained.extend(rows.tolist())
        sequence = shape["inverse"][rows].tolist()
        counts = Counter(sequence)
        transitions = Counter(zip(sequence[:-1], sequence[1:]))
        all_transitions.update(transitions)
        target_set = tuple(sorted(counts))
        block_sets.append(target_set)
        detail = dict(identity, block_id=block, declared_trials=len(rows),
            original_trial_rows=rows.tolist(), trial_start_indices=starts[rows].tolist(),
            trial_end_indices=ends[rows].tolist(), target_id_sequence=sequence,
            observed_target_ids=list(target_set),
            target_counts={str(key): value for key, value in sorted(counts.items())},
            directed_transition_counts=transition_rows(transitions),
            repeated_target_ids=sum(count > 1 for count in counts.values()),
            singleton_target_ids=sum(count == 1 for count in counts.values()),
            all_trial_targets_distinct=bool(sequence) and len(counts) == len(sequence),
            self_transition_count=sum(count for (a, b), count in transitions.items() if a == b),
            observed_hub_candidates=hub_candidates(transitions),
            target_motion_within_trial_observable=False,
            recurrence_does_not_establish_task_identity_or_size_independence=True)
        write_row(block_stream, detail)
        block_summaries.append({key: detail[key] for key in
            ("block_id", "declared_trials", "observed_target_ids", "repeated_target_ids",
             "singleton_target_ids", "all_trial_targets_distinct", "observed_hub_candidates")})
    if sorted(retained) != list(range(len(starts))):
        raise ValueError("Original trial identity not preserved exactly once")
    # A nearest-neighbor spacing diagnostic, not a clustering/eligibility rule.
    distances, _ = cKDTree(shape["points"]).query(shape["points"], k=2)
    nearest = distances[:, 1]
    if not np.isfinite(nearest).all() or np.any(nearest <= 0):
        raise ValueError("Nonpositive spacing after exact target deduplication")
    output = dict(identity, declared_trials=len(starts), retained_trials=len(retained),
        declared_blocks=len(declared_blocks), exact_unique_targets=len(shape["points"]),
        normalized_unique_targets=shape["points"].tolist(),
        normalization_center=shape["center"], normalization_rms_radius=shape["rms_radius"],
        roundoff_descriptor_tolerance=shape["tolerance"],
        normalized_minimum_target_spacing=float(nearest.min()),
        normalized_nearest_neighbor_spacing_quantiles={str(q): float(np.quantile(nearest, q))
            for q in (0.0, 0.01, 0.05, 0.5, 1.0)},
        target_ids_with_nearest_neighbor_within_roundoff_descriptor=int(np.sum(nearest <= shape["tolerance"])),
        distinct_observed_block_target_sets=len(set(block_sets)),
        all_blocks_cover_session_targets=all(len(s) == len(shape["points"]) for s in block_sets),
        block_set_differences_mean="observed_coverage_difference_not_proof_of_layout_change",
        blocks=block_summaries, pooled_within_block_transition_counts=transition_rows(all_transitions),
        observed_hub_candidates=hub_candidates(all_transitions),
        within_trial_stability="not_observable_from_one_target_point_per_trial",
        size_dependence="unresolved_not_measured_not_a_positive_size_dependence_finding",
        schedule_eligibility_adjudicated=False)
    return {"output": output, "geometry": shape, "transitions": all_transitions,
            "block_sets": block_sets, "identity": identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned compute allocation required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    sessions, counts = [], Counter()
    with args.structural_jsonl.open() as source, (args.output_dir / "observed_blocks.jsonl").open("x") as blocks:
        for line in source:
            if line.strip():
                session = prepare(json.loads(line), blocks)
                sessions.append(session)
                counts[session["identity"]["participant"]] += 1
    sessions.sort(key=lambda item: (item["identity"]["participant"], item["identity"]["day"]))
    identities = [(s["identity"]["participant"], s["identity"]["day"]) for s in sessions]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate session identity")
    strata = defaultdict(list)
    with (args.output_dir / "observed_sessions.jsonl").open("x") as output:
        for session in sessions:
            write_row(output, session["output"])
            ident = session["identity"]
            strata[(ident["participant"], ident["physical_array_set"],
                    len(session["geometry"]["points"]))].append(session)
    aligned, transition_differences, layout_differences = 0, 0, 0
    with (args.output_dir / "aligned_session_pairs.jsonl").open("x") as output:
        for stratum, members in sorted(strata.items()):
            for index, left in enumerate(members):
                for right in members[index + 1:]:
                    mapping = correspondence(left["geometry"], right["geometry"])
                    if mapping is None:
                        continue
                    inverse = {int(dst): src for src, dst in enumerate(mapping)}
                    right_edges = {(inverse[a], inverse[b]) for a, b in right["transitions"]}
                    left_edges = set(left["transitions"])
                    left_only, right_only = sorted(left_edges - right_edges), sorted(right_edges - left_edges)
                    left_sets = set(left["block_sets"])
                    right_sets = {tuple(sorted(inverse[v] for v in row)) for row in right["block_sets"]}
                    aligned += 1
                    transition_differences += bool(left_only or right_only)
                    layout_differences += left_sets != right_sets
                    write_row(output, {"participant": stratum[0], "physical_array_set": stratum[1],
                        "source_day": left["identity"]["day"], "target_day": right["identity"]["day"],
                        "source_to_target_coordinate_ids": mapping.tolist(),
                        "maximum_normalized_matched_point_error": float(np.max(np.linalg.norm(
                            left["geometry"]["points"] - right["geometry"]["points"][mapping], axis=1))),
                        "alignment_rule": "previous_roundoff_descriptor_only_no_new_scientific_tolerance",
                        "observed_transition_support_equal": not (left_only or right_only),
                        "transitions_observed_only_in_source": left_only,
                        "transitions_observed_only_in_target_in_source_ids": right_only,
                        "observed_block_target_set_families_equal": left_sets == right_sets,
                        "block_target_sets_only_in_source": sorted(left_sets - right_sets),
                        "block_target_sets_only_in_target_in_source_ids": sorted(right_sets - left_sets),
                        "observed_compatibility": "identical_observed_support" if
                            not (left_only or right_only) and left_sets == right_sets else "observed_support_difference",
                        "difference_is_proven_forbidden_transition_conflict": False,
                        "sampling_absence_vs_protocol_change": "unresolved",
                        "size_dependence": "unresolved_not_observed",
                        "schedule_eligibility_adjudicated": False})
    report = {"analysis_id": "structural_support", "phase": "observed_schedule_diagnostic",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "elapsed_seconds": time.monotonic() - begin,
        "complete": True, "sessions": len(sessions), "session_counts": dict(sorted(counts.items())),
        "declared_trials": sum(s["output"]["declared_trials"] for s in sessions),
        "retained_trials": sum(s["output"]["retained_trials"] for s in sessions),
        "blocks": sum(s["output"]["declared_blocks"] for s in sessions),
        "aligned_session_pairs": aligned, "pairs_with_observed_transition_support_differences": transition_differences,
        "pairs_with_observed_block_coverage_differences": layout_differences,
        "source_structural_view": str(args.structural_jsonl),
        "scientific_coordinate_tolerance_activated": False, "scientific_support_limits_activated": False,
        "target_coordinates_merged_or_rounded": False, "roles_assigned": False, "trials_selected": False,
        "neural_cursor_impedance_values_decoded": False, "raw_source_files_opened": False,
        "size_dependence_adjudicated": False, "forbidden_transition_rules_inferred_from_absence": False,
        "schedule_eligibility_adjudicated": False,
        "limitations": ["One target point per trial cannot establish within-trial target stability.",
            "Exact-coordinate recurrence does not identify tasks or prove size independence.",
            "Absent sampled targets/edges are not proof of a changed or forbidden schedule.",
            "Roundoff-only pair alignment is a descriptor, not scientific schedule qualification.",
            "Minimum spacing can reflect near duplicates; no spacing threshold is applied."]}
    (args.output_dir / "observed_schedule_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
