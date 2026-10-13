#!/usr/bin/env python3
"""Run EP17's selected metadata pilot or finite full concept assignment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, DecimalException
from fractions import Fraction
import json
from pathlib import Path, PurePosixPath
import os
import resource
import signal
import socket
import time

import concept_role_assignment as method
import qualify_source_chronology as chronology
import qualify_source_roles as stage
import reconcile_released_events as core

QUALIFIED = chronology.QUALIFICATION
CHRONOLOGY = core.EPISODE / "outputs/runs/source_chronology/attempt001"
OUTPUT_PARENT = core.EPISODE / "outputs/runs/concept_role_assignment"
PILOT = OUTPUT_PARENT / "pilot001"
INTERNAL_SECONDS = 300
FULL_INTERNAL_SECONDS = 1800
PILOT_CHECKS = 4096
UPDATE_PAIRS_PER_SIZE = 256
INPUT_PATHS = (
    core.METADATA / "metadata_report.json", stage.EVENTS / "reconciliation_report.json",
    core.METADATA / "run_metadata.csv", core.METADATA / "presentation_identity.csv",
    stage.EVENTS / "event_metadata.csv", QUALIFIED / "qualification_report.json",
    QUALIFIED / "image_qualification.csv", QUALIFIED / "concept_qualification.csv",
    QUALIFIED / "taxonomy_crosswalk.csv", CHRONOLOGY / "chronology_report.json",
    CHRONOLOGY / "presentation_chronology.csv", CHRONOLOGY / "run_chronology.csv",
)
FULL_INPUT_PATHS = (PILOT / "pilot_report.json", PILOT / "feature_definition.json", PILOT / "raw_metadata_features.csv",
                    QUALIFIED / "qualification_report.json", CHRONOLOGY / "chronology_report.json",
                    QUALIFIED / "image_qualification.csv", CHRONOLOGY / "presentation_chronology.csv")
ROLE_NAMES = ("development", "support_calibration", "sealed_audit")
ROLE_FIELDS = ("role", "outer_fold", "development_block", "audit_block")
CONCEPT_PREFIX = ("things_category_nr", "uniqueID", "provider_category_nr", "original_common_image_count", "qualified_image_count")


def load_inputs():
    """Read each fixed metadata file once and release duplicate row tables."""
    base = stage.load_sources()
    eligible = chronology.load_eligibility(base)
    _, concept_rows = stage.table_rows(stage.bounded_read(QUALIFIED / "concept_qualification.csv"), stage.CONCEPT_FIELDS)
    header, taxonomy_rows = stage.table_rows(stage.bounded_read(QUALIFIED / "taxonomy_crosswalk.csv"))
    core.require(header[:2] == ["things_category_nr", "uniqueID"] and len(header) == 55,
                 "taxonomy_projection_fields_changed")
    taxonomy = {}
    for item in taxonomy_rows:
        cid = core.integer(item["things_category_nr"], "things_category_nr")
        core.require(cid not in taxonomy, "duplicate_taxonomy_concept")
        membership = [core.integer(item[label], "taxonomy_membership") for label in header[2:]]
        core.require(all(value in (0, 1) for value in membership), "taxonomy_not_binary")
        taxonomy[cid] = (item["uniqueID"], membership)
    del taxonomy_rows
    report = json.loads(stage.bounded_read(CHRONOLOGY / "chronology_report.json"))
    core.require(report.get("status") == "completed" and report.get("errors") == [] and
                 all(report.get(flag) is True for flag in (
                     "source_chronology_complete", "retained_half_qualification_complete",
                     "global_chronology_complete", "all_observed_ordinals_qualified")) and
                 report.get("preserved_presentations") == 49980 and report.get("preserved_runs") == 833 and
                 report.get("eligibility_attempt") == str(QUALIFIED) and
                 report.get("source_metadata_attempt") == str(core.METADATA) and
                 report.get("event_projection_attempt") == str(stage.EVENTS) and
                 report.get("roles_assigned") is False and report.get("source_roles_complete") is False,
                 "completed_fixed_chronology_required")
    _, run_rows = stage.table_rows(stage.bounded_read(CHRONOLOGY / "run_chronology.csv"), chronology.RUN_FIELDS)
    run_map = {}
    for item in run_rows:
        key = stage.run_key(item)
        core.require(key not in run_map and item["global_order_qualified"] == "True" and
                     item["component_coordinates_qualified"] == "True" and
                     item["global_source_history_start"] == item["component_source_history_start"],
                     "unqualified_chronology_run")
        run_map[key] = item
    core.require(set(run_map) == {stage.run_key(row) for row in base}, "chronology_run_roster_changed")
    fields = (*core.OUTPUT_FIELDS, "concept_name", "condition", *chronology.ADDED_FIELDS)
    _, rows = stage.table_rows(stage.bounded_read(CHRONOLOGY / "presentation_chronology.csv"), fields)
    core.require(len(rows) == len(base), "chronology_row_count_changed")
    expected = {stage.key(row): row for row in base}
    seen, image_identity, image_counts = set(), {}, defaultdict(Counter)
    by_person, images_by_concept = defaultdict(list), defaultdict(set)
    for row in rows:
        key, run_key = stage.key(row), stage.run_key(row)
        core.require(key in expected and key not in seen and run_key in run_map, "chronology_coordinate_changed")
        core.require(all(row[field] == value for field, value in expected[key].items()),
                     "chronology_source_projection_changed")
        image = row["expected_provider_image_path"]
        cid = core.integer(row["things_category_nr"], "things_category_nr")
        identity = (cid, row["concept_name"], core.integer(row["provider_category_nr"], "provider_category_nr"))
        core.require(image_identity.setdefault(image, identity) == identity and
                     PurePosixPath(image).parent.name == identity[1] and
                     row["eligible_image"] == str(eligible[image]) and row["ordinal_status"] == "valid",
                     "chronology_eligibility_identity_changed")
        position = core.integer(row["global_source_history_position"], "global_position")
        core.require(position == core.integer(run_map[run_key]["global_source_history_start"], "global_start") +
                     core.integer(row["source_order"], "source_order") and
                     row["component_source_history_position"] == row["global_source_history_position"],
                     "chronology_position_changed")
        retained_rank = core.integer(row["retained_repeat_rank"], "retained_repeat_rank")
        core.require(retained_rank >= 0 and row["repeat_half"] ==
                     (str(1 + retained_rank % 2) if eligible[image] else ""), "repeat_half_mapping_changed")
        by_person[row["participant"]].append(row)
        image_counts[image][row["participant"]] += 1
        images_by_concept[cid].add(image)
        seen.add(key)
    core.require(seen == set(expected), "chronology_projection_incomplete")
    del base, expected, run_rows
    sessions, totals = {}, {}
    for subject in method.SUBJECTS:
        selected = sorted(by_person[subject], key=lambda row: int(row["global_source_history_position"]))
        core.require(len(selected) == stage.RUN_COUNTS[subject] * 60 and
                     [int(row["retained_acquisition_rank"]) for row in selected] == list(range(len(selected))) and
                     len({row["global_source_history_position"] for row in selected}) == len(selected),
                     "acquisition_rank_scope_changed")
        sessions[subject] = sorted({int(row["session"]) for row in selected})
        totals[subject] = len(selected)
    concepts = []
    for item in concept_rows:
        cid = core.integer(item["things_category_nr"], "things_category_nr")
        original = sum(all(image_counts[image][s] > 0 for s in method.SUBJECTS) for image in images_by_concept[cid])
        qualified = sum(eligible[image] for image in images_by_concept[cid])
        core.require(cid in taxonomy and taxonomy[cid][0] == item["uniqueID"] == item["concept_name"] and
                     item["eligible_concept"] == "True" and item["original_five_six_status"] == str(original) and
                     int(item["original_common_image_count"]) == original and int(item["qualified_image_count"]) == qualified and
                     int(item["taxonomy_membership_count"]) == sum(taxonomy[cid][1]) and
                     item["taxonomy_unmapped"] == str(not any(taxonomy[cid][1])), "concept_projection_changed")
        provider_id = core.integer(item["provider_category_nr"], "provider_category_nr")
        core.require(all(image_identity[image] == (cid, item["uniqueID"], provider_id) for image in images_by_concept[cid]),
                     "concept_image_identity_changed")
        concepts.append(method.Concept(cid, item["uniqueID"], provider_id, original, qualified))
    core.require(len(concepts) == len({c.cid for c in concepts}) == 720 and
                 {c.cid for c in concepts} == set(taxonomy) == set(images_by_concept) and
                 sum(eligible.values()) == 3634 and sum(not any(item[1]) for item in taxonomy.values()) == 128,
                 "adopted_eligible_universe_changed")
    return concepts, header[2:], {cid: item[1] for cid, item in taxonomy.items()}, rows, sessions, totals


def pilot_measurements(prepared, check_deadline):
    times = {}
    started = time.perf_counter()
    denominator, weights = method.integer_weights(prepared.fields)
    labels = method.initialize(prepared.concepts, [480, 120, 120], method.NAMESPACE)
    gram = method.gram_matrix(prepared.q, weights)
    times["role_gram_seconds"] = time.perf_counter() - started
    check_deadline()
    started = time.perf_counter()
    pairs = method.pair_order(prepared.concepts, method.NAMESPACE)
    times["role_pair_order_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    cache = method.caches(gram, labels, 3)
    sums = method.group_feature_sums(prepared.q, labels, 3)
    initial_J = method.objective(prepared.q, weights, labels, [480, 120, 120])
    times["role_direct_cache_objective_seconds"] = time.perf_counter() - started
    check_deadline()
    selected = []
    for a, b in pairs:
        if labels[a] != labels[b]:
            selected.append((a, b))
            if len(selected) == PILOT_CHECKS:
                break
    core.require(len(selected) == PILOT_CHECKS, "pilot_pair_check_count_changed")
    started = time.perf_counter()
    deltas = [method.swap_delta(gram, cache, labels, [480, 120, 120], a, b) for a, b in selected]
    times["cached_delta_checks_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    for (a, b), delta in zip(selected, deltas):
        check_deadline()
        core.require(delta == method.direct_delta(prepared.q, weights, sums, labels, [480, 120, 120], a, b),
                     "pilot_independent_delta_disagreement")
    times["independent_delta_checks_seconds"] = time.perf_counter() - started
    original_labels = list(labels)
    update_timings = []
    for n in (720, 480, 120):
        # Prefixes benchmark the same arbitrary-integer arithmetic at each
        # parent size. They are not candidate child assignments or optima.
        cropped = [vector[:n] for vector in gram[:n]]
        temporary_labels = labels[:n]
        temporary_cache = method.caches(cropped, temporary_labels, 3)
        baseline_cache = [vector[:] for vector in temporary_cache]
        benchmark_pairs = [(a, b) for a, b in pairs if b < n and temporary_labels[a] != temporary_labels[b]][:UPDATE_PAIRS_PER_SIZE]
        core.require(len(benchmark_pairs) == UPDATE_PAIRS_PER_SIZE, "pilot_cache_benchmark_pairs_missing")
        started = time.perf_counter()
        for a, b in benchmark_pairs:
            check_deadline()
            method.apply_swap(cropped, temporary_cache, temporary_labels, a, b)
            method.apply_swap(cropped, temporary_cache, temporary_labels, a, b)
        elapsed = time.perf_counter() - started
        core.require(temporary_labels == labels[:n] and temporary_cache == baseline_cache,
                     "temporary_cache_benchmark_not_restored")
        count = 2 * UPDATE_PAIRS_PER_SIZE * n
        update_timings.append({"parent_size": n, "temporary_apply_undo_pairs": UPDATE_PAIRS_PER_SIZE,
                               "per_concept_cache_updates": count, "seconds": elapsed,
                               "seconds_per_concept_cache_update": elapsed / count,
                               "benchmark_only_not_a_child_assignment": True})
    core.require(labels == original_labels and cache == method.caches(gram, labels, 3),
                 "pilot_initialization_changed")
    check_deadline()
    gram_ratio = (720 * 721 + 2 * 480 * 481 + 120 * 121) / (720 * 721)
    pair_ratio = method.MAX_PAIR_VISITS / 8 / len(pairs)
    rates = {item["parent_size"]: item["seconds_per_concept_cache_update"] for item in update_timings}
    update_counts = {720: 1490918400, 480: 882892800, 120: 6854400}
    core.require(sum(update_counts.values()) == method.MAX_CACHE_UPDATES, "full_cache_bound_changed")
    cost = {
        "four_gram_preparations": times["role_gram_seconds"] * gram_ratio,
        "four_pair_orders": times["role_pair_order_seconds"] * pair_ratio,
        "all_pair_visits_as_cross_group_checks": method.MAX_PAIR_VISITS * times["cached_delta_checks_seconds"] / PILOT_CHECKS,
        "all_possible_accepted_swap_cache_updates": sum(update_counts[n] * rates[n] for n in update_counts),
        "all_initial_and_eight_sweep_cache_objective_replays": times["role_direct_cache_objective_seconds"] * gram_ratio * 9,
    }
    return {"independent_delta_checks": PILOT_CHECKS, "source_based_swaps_accepted": 0,
            "initialization_restored": True, "initial_J": initial_J, "integer_weight_denominator": denominator,
            "feature_count": len(prepared.fields), "timings": times, "cache_update_timings": update_timings,
            "full_maximum_pair_visits": method.MAX_PAIR_VISITS,
            "full_maximum_per_concept_cache_updates": method.MAX_CACHE_UPDATES,
            "full_cache_update_counts_by_parent_size": update_counts,
            "unbuffered_full_operation_cost_estimate_seconds": cost,
            "cost_estimates_are_measured_projections_not_guaranteed_walltime_bounds": True}


def write_feature_artifact(out, raw, prepared):
    names = [field.name for field in raw.fields]
    prefix = ["things_category_nr", "uniqueID", "provider_category_nr", "original_common_image_count", "qualified_image_count"]
    with (out / "raw_metadata_features.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(prefix + names)
        for concept, values in zip(raw.concepts, raw.values):
            writer.writerow([concept.cid, concept.uid, concept.provider_id, concept.original_count, concept.qualified_count] +
                            ["" if value is None else str(value) for value in values])
    core.require((out / "raw_metadata_features.csv").stat().st_size <= stage.FILE_CAP, "raw_feature_artifact_exceeds_cap")
    definitions = {"analysis_id": "concept_role_assignment", "artifact": "raw_metadata_features",
                   "roles_assigned": False, "metadata_assignments_complete": False,
                   "source_roles_complete": False, "decimal_precision": method.PRECISION,
                   "rounding": "ROUND_HALF_EVEN", "standardized_quantum": "0.0001",
                   "blank_cells": "unobserved conditional log-lag moments; never zero delays",
                   "imputation_applies_separately_to_each_assignment_parent": True,
                   "fields": [{"name": f.name, "family": f.family, "participant": f.participant,
                               "weight_numerator": f.weight.numerator, "weight_denominator": f.weight.denominator}
                              for f in raw.fields],
                   "role_parent_field_preparation": prepared.field_details,
                   "metadata_diagnostics": raw.diagnostics,
                   "source_inputs": [str(path) for path in INPUT_PATHS]}
    with (out / "feature_definition.json").open("x", encoding="utf-8") as stream:
        json.dump(definitions, stream, indent=2, allow_nan=False)
        stream.write("\n")


def complete_pilot(report, concept_count, field_count):
    report.update({"status": "completed", "metadata_feature_preparation_complete": True,
                   "pilot_complete": True, "concept_count": concept_count, "field_count": field_count})


def run_pilot(out):
    start = time.monotonic()
    report = {"analysis_id": "concept_role_assignment", "mode": "pilot", "status": "running",
              "metadata_feature_preparation_complete": False, "pilot_complete": False,
              "metadata_assignments_complete": False, "source_roles_complete": False,
              "roles_assigned": False, "folds_created": False, "blocks_created": False,
              "neural_values_read": False, "behavioral_outcomes_read": False, "image_values_read": False,
              "model_features_read": False, "audit_opened": False,
              "internal_seconds": INTERNAL_SECONDS, "external_seconds_required": 360,
              "requested_memory_gib": 6, "input_file_byte_cap": stage.FILE_CAP,
              "source_inputs": [str(path) for path in INPUT_PATHS], "source_input_file_count": len(INPUT_PATHS),
              "network_requests": 0, "automatic_retries": 0, "full_assignment_selected": False,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "errors": []}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(INTERNAL_SECONDS)
    def check_deadline():
        core.require(time.monotonic() < start + INTERNAL_SECONDS, "attempt_deadline")
    try:
        began = time.perf_counter()
        inputs = load_inputs()
        report["input_validation_seconds"] = time.perf_counter() - began
        check_deadline()
        began = time.perf_counter()
        raw = method.build_features(*inputs)
        del inputs
        report["raw_metadata_feature_seconds"] = time.perf_counter() - began
        check_deadline()
        began = time.perf_counter()
        prepared = method.prepare(raw)
        report["role_parent_preparation_seconds"] = time.perf_counter() - began
        check_deadline()
        report.update(pilot_measurements(prepared, check_deadline))
        cost = report["unbuffered_full_operation_cost_estimate_seconds"]
        cost["input_and_raw_feature_preparation_if_repeated"] = report["input_validation_seconds"] + report["raw_metadata_feature_seconds"]
        cost["four_parent_scalings"] = report["role_parent_preparation_seconds"] * (1800 / 720)
        report["conservative_full_planning_seconds"] = 2 * sum(cost.values()) + 60
        report["planning_factor"] = 2
        report["planning_fixed_seconds"] = 60
        report["full_execution_automatically_authorized"] = False
        check_deadline()
        write_feature_artifact(out, raw, prepared)
        check_deadline()
        complete_pilot(report, len(raw.concepts), len(raw.fields))
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report.update({"status": "stopped", "metadata_feature_preparation_complete": False, "pilot_complete": False})
    except (OSError, ValueError, TypeError, KeyError, IndexError, DecimalException, OverflowError):
        report["errors"].append({"code": "concept_assignment_pilot_processing_failure"})
        report.update({"status": "stopped", "metadata_feature_preparation_complete": False, "pilot_complete": False})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with (out / "pilot_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "pilot_complete": report["pilot_complete"],
                      "roles_assigned": False, "elapsed_seconds": report["elapsed_seconds"],
                      "peak_rss_kib": report["peak_rss_kib"]}))
    return 0 if report["status"] == "completed" and report["pilot_complete"] and report["metadata_feature_preparation_complete"] else 2


def load_pilot_features():
    """Reuse exact completed raw summaries, including unobserved-cell masks."""
    report = json.loads(stage.bounded_read(PILOT / "pilot_report.json"))
    core.require(report.get("status") == "completed" and report.get("pilot_complete") is True and
                 report.get("metadata_feature_preparation_complete") is True and
                 report.get("metadata_assignments_complete") is False and report.get("roles_assigned") is False and
                 report.get("source_roles_complete") is False and report.get("errors") == [] and
                 report.get("concept_count") == 720 and report.get("independent_delta_checks") == 4096 and
                 report.get("source_based_swaps_accepted") == 0 and report.get("initialization_restored") is True and
                 report.get("source_inputs") == [str(path) for path in INPUT_PATHS], "completed_exact_feature_pilot_required")
    definition = json.loads(stage.bounded_read(PILOT / "feature_definition.json"))
    core.require(definition.get("roles_assigned") is False and definition.get("metadata_assignments_complete") is False and
                 definition.get("source_roles_complete") is False and definition.get("decimal_precision") == 80 and
                 definition.get("rounding") == "ROUND_HALF_EVEN" and definition.get("standardized_quantum") == "0.0001" and
                 definition.get("source_inputs") == [str(path) for path in INPUT_PATHS], "prepared_feature_conventions_changed")
    fields = [method.Field(item["name"], item["family"], Fraction(item["weight_numerator"], item["weight_denominator"]),
                           item["participant"]) for item in definition["fields"]]
    core.require(len(fields) == report.get("field_count") == report.get("feature_count") and
                 len({field.name for field in fields}) == len(fields) and
                 sum((field.weight for field in fields), Fraction()) == 1 and all(field.weight > 0 for field in fields),
                 "prepared_feature_fields_changed")
    _, records = stage.table_rows(stage.bounded_read(PILOT / "raw_metadata_features.csv"), (*CONCEPT_PREFIX, *(f.name for f in fields)))
    concepts, values = [], []
    for item in records:
        concept = method.Concept(int(item["things_category_nr"]), item["uniqueID"], int(item["provider_category_nr"]),
                                 int(item["original_common_image_count"]), int(item["qualified_image_count"]))
        concept.identity()
        core.require(concept.original_count in (5, 6) and 1 <= concept.qualified_count <= concept.original_count,
                     "prepared_concept_exemplars_changed")
        concepts.append(concept)
        vector = [None if item[field.name] == "" else Decimal(item[field.name]) for field in fields]
        core.require(all(value is None or value.is_finite() for value in vector), "nonfinite_prepared_feature")
        values.append(vector)
    core.require(len(concepts) == len({c.cid for c in concepts}) == len({c.uid for c in concepts}) == 720 and
                 concepts == sorted(concepts, key=lambda c: c.key()) and sum(c.qualified_count for c in concepts) == 3634,
                 "prepared_concept_universe_changed")
    for path, flag in ((QUALIFIED / "qualification_report.json", "eligibility_complete"),
                       (CHRONOLOGY / "chronology_report.json", "source_chronology_complete")):
        prerequisite = json.loads(stage.bounded_read(path))
        core.require(prerequisite.get("status") == "completed" and prerequisite.get(flag) is True and
                     prerequisite.get("errors") == [] and prerequisite.get("roles_assigned") is False and
                     prerequisite.get("source_roles_complete") is False, "source_prerequisite_no_longer_complete")
    return method.RawFeatures(concepts, fields, values, definition["metadata_diagnostics"]), definition["role_parent_field_preparation"]


def execute_assignments(raw, expected_role_preparation, check_deadline):
    role_prepared = method.prepare(raw)
    core.require(role_prepared.field_details == expected_role_preparation, "reloaded_raw_feature_preparation_changed")
    check_deadline()
    role_labels, role_report = method.optimize(role_prepared, [480, 120, 120], method.NAMESPACE, check_deadline)
    role_by_id = dict(zip((c.cid for c in role_prepared.concepts), role_labels))
    core.require(Counter(role_labels) == {0: 480, 1: 120, 2: 120}, "role_capacities_changed")
    development = {cid for cid, label in role_by_id.items() if label == 0}
    audit = {cid for cid, label in role_by_id.items() if label == 2}
    children = method.assign_children(raw, development, audit, check_deadline)
    check_deadline()
    balance = {"roles": method.balance_summary(role_prepared, role_labels, [480, 120, 120], ROLE_NAMES)}
    reports = {"roles": role_report}
    partitions = {"roles": (role_by_id, ROLE_NAMES)}
    for name, child in children.items():
        prepared = child["prepared"]
        labels = [child["labels"][c.cid] for c in prepared.concepts]
        prefix = "fold" if name == "outer_folds" else "block"
        names = [f"{prefix}{i + 1:02d}" for i in range(len(child["sizes"]))]
        balance[name] = method.balance_summary(prepared, labels, child["sizes"], names)
        reports[name] = child["report"]
        partitions[name] = child["labels"], names
        check_deadline()
    assignments = {}
    for concept in raw.concepts:
        cid, role = concept.cid, role_by_id[concept.cid]
        assignments[cid] = {"role": ROLE_NAMES[role],
                            "outer_fold": f"fold{children['outer_folds']['labels'][cid] + 1:02d}" if role == 0 else "",
                            "development_block": f"block{children['development_blocks']['labels'][cid] + 1:02d}" if role == 0 else "",
                            "audit_block": f"block{children['audit_blocks']['labels'][cid] + 1:02d}" if role == 2 else ""}
    # Descriptive exceptions remain outside every optimizer objective.
    flag_counts = raw.diagnostics["schedule_flag_counts_by_concept_participant"]
    for name, (labels_by_id, names) in partitions.items():
        counts = {group: {subject: Counter() for subject in method.SUBJECTS} for group in names}
        for key, flags in flag_counts.items():
            subject, cid = key.split(":")
            cid = int(cid)
            if cid in labels_by_id:
                counts[names[labels_by_id[cid]]][subject].update(flags)
        balance[name]["schedule_flag_counts_by_group_participant"] = counts
    return assignments, {"assignments": reports, "balance": balance,
                         "development_blocks_received_outer_fold_labels": False,
                         "quantized_objective_only_selected_moves": True}


def attach_image_roles(images, concepts, assignments):
    concept_by_id = {concept.cid: concept for concept in concepts}
    output, by_path = [], {}
    for row in images:
        cid = core.integer(row["things_category_nr"], "things_category_nr")
        image = row["provider_image_path"]
        core.require(cid in assignments and cid in concept_by_id and image not in by_path, "image_assignment_identity_changed")
        concept = concept_by_id[cid]
        core.require(row["concept_name"] == concept.uid == PurePosixPath(image).parent.name and
                     int(row["provider_category_nr"]) == concept.provider_id and row["eligible_image"] in ("True", "False") and
                     row["common_to_four"] in ("True", "False"), "image_concept_anchor_changed")
        augmented = {**row, **assignments[cid]}
        by_path[image] = augmented
        output.append(augmented)
    return output, by_path


def attach_presentation_roles(rows, images_by_path, assignments):
    output, coordinates = [], set()
    counts = defaultdict(Counter)
    for row in rows:
        image = row["expected_provider_image_path"]
        cid = core.integer(row["things_category_nr"], "things_category_nr")
        key = stage.key(row)
        core.require(image in images_by_path and cid in assignments and key not in coordinates, "presentation_assignment_identity_changed")
        anchor = images_by_path[image]
        core.require(row["things_category_nr"] == anchor["things_category_nr"] and
                     row["concept_name"] == anchor["concept_name"] and row["source_image_basename"] == anchor["image_name"] and
                     row["provider_category_nr"] == anchor["provider_category_nr"] and
                     row["provider_exemplar_nr"] == anchor["provider_exemplar_nr"] and
                     row["condition"] == anchor["condition"] and row["eligible_image"] == anchor["eligible_image"],
                     "presentation_image_anchor_changed")
        output.append({**row, **assignments[cid]})
        counts[image][row["participant"]] += 1
        coordinates.add(key)
    for image, anchor in images_by_path.items():
        core.require(all(counts[image][subject] == int(anchor[f"retained_count_{subject}"]) for subject in method.SUBJECTS),
                     "inherited_presentation_counts_changed")
    return output


def publish_assignments(out, raw, assignments, balances, check_deadline):
    """Reached only after all four optimizers finish; report is the barrier."""
    core.require(len(assignments) == 720 and set(assignments) == {c.cid for c in raw.concepts}, "incomplete_concept_assignment")
    _, images = stage.table_rows(stage.bounded_read(QUALIFIED / "image_qualification.csv"), stage.IMAGE_FIELDS)
    image_rows, image_by_path = attach_image_roles(images, raw.concepts, assignments)
    core.require(len(image_rows) == 4320 and sum(row["eligible_image"] == "True" for row in image_rows) == 3634,
                 "inherited_image_roster_changed")
    concept_counts = defaultdict(Counter)
    for row in image_rows:
        cid = int(row["things_category_nr"])
        concept_counts[cid]["common"] += row["common_to_four"] == "True"
        concept_counts[cid]["qualified"] += row["eligible_image"] == "True"
    core.require(all(concept_counts[c.cid]["common"] == c.original_count and
                     concept_counts[c.cid]["qualified"] == c.qualified_count for c in raw.concepts),
                 "reused_feature_image_counts_changed")
    fields = (*core.OUTPUT_FIELDS, "concept_name", "condition", *chronology.ADDED_FIELDS)
    _, rows = stage.table_rows(stage.bounded_read(CHRONOLOGY / "presentation_chronology.csv"), fields)
    presentations = attach_presentation_roles(rows, image_by_path, assignments)
    core.require(len(presentations) == 49980 and len({stage.run_key(row) for row in presentations}) == 833 and
                 Counter(row["participant"] for row in presentations) ==
                 Counter({subject: stage.RUN_COUNTS[subject] * 60 for subject in method.SUBJECTS}),
                 "inherited_presentation_roster_changed")
    check_deadline()
    concept_rows = [{"things_category_nr": c.cid, "uniqueID": c.uid, "provider_category_nr": c.provider_id,
                     "original_common_image_count": c.original_count, "qualified_image_count": c.qualified_count,
                     **assignments[c.cid]} for c in raw.concepts]
    stage.write_table(out / "concept_assignments.csv", (*CONCEPT_PREFIX, *ROLE_FIELDS), concept_rows)
    stage.write_table(out / "image_assignments.csv", (*stage.IMAGE_FIELDS, *ROLE_FIELDS), image_rows)
    stage.write_table(out / "presentation_assignments.csv", (*fields, *ROLE_FIELDS), presentations)
    with (out / "balance_report.json").open("x", encoding="utf-8") as stream:
        json.dump(balances, stream, indent=2, allow_nan=False)
        stream.write("\n")
    check_deadline()
    return {"concepts": len(concept_rows), "images": len(image_rows), "presentations": len(presentations), "runs": 833}


def complete_full(report):
    report.update({"status": "completed", "metadata_assignments_complete": True,
                   "roles_assigned": True, "folds_created": True, "blocks_created": True})


def run_full(out):
    start = time.monotonic()
    report = {"analysis_id": "concept_role_assignment", "mode": "full", "status": "running",
              "metadata_assignments_complete": False, "source_roles_complete": False,
              "roles_assigned": False, "folds_created": False, "blocks_created": False,
              "neural_handoffs_materialized": False, "physical_role_isolation_verified": False,
              "neural_values_read": False, "behavioral_outcomes_read": False, "image_values_read": False,
              "model_features_read": False, "audit_opened": False, "pilot_attempt": str(PILOT),
              "raw_metadata_features_reused_without_recomputation": True,
              "internal_seconds": FULL_INTERNAL_SECONDS, "external_seconds_required": 1860,
              "requested_memory_gib": 6, "input_file_byte_cap": stage.FILE_CAP,
              "source_inputs": [str(path) for path in FULL_INPUT_PATHS], "source_input_file_count": len(FULL_INPUT_PATHS),
              "network_requests": 0, "automatic_retries": 0, "maximum_sweeps_per_assignment": 8,
              "maximum_pair_visits": method.MAX_PAIR_VISITS, "maximum_per_concept_cache_updates": method.MAX_CACHE_UPDATES,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "errors": []}
    signal.signal(signal.SIGALRM, core.deadline_alarm)
    signal.alarm(FULL_INTERNAL_SECONDS)
    def check_deadline():
        core.require(time.monotonic() < start + FULL_INTERNAL_SECONDS, "attempt_deadline")
    try:
        raw, expected_preparation = load_pilot_features()
        check_deadline()
        assignments, balances = execute_assignments(raw, expected_preparation, check_deadline)
        check_deadline()
        report["assignment_searches"] = balances["assignments"]
        report["preserved_metadata"] = publish_assignments(out, raw, assignments, balances, check_deadline)
        report["role_counts"] = dict(Counter(item["role"] for item in assignments.values()))
        report["development_blocks_received_outer_fold_labels"] = False
        check_deadline()
        complete_full(report)
    except core.Refusal as error:
        report["errors"].append(error.detail)
        report.update({"status": "stopped", "metadata_assignments_complete": False,
                       "roles_assigned": False, "folds_created": False, "blocks_created": False})
    except (OSError, ValueError, TypeError, KeyError, IndexError, DecimalException, OverflowError):
        report["errors"].append({"code": "concept_assignment_processing_failure"})
        report.update({"status": "stopped", "metadata_assignments_complete": False,
                       "roles_assigned": False, "folds_created": False, "blocks_created": False})
    finally:
        signal.alarm(0)
    report["elapsed_seconds"] = time.monotonic() - start
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["downstream_use_requires_this_terminal_report_completed"] = True
    with (out / "assignment_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "metadata_assignments_complete": report["metadata_assignments_complete"],
                      "source_roles_complete": False, "elapsed_seconds": report["elapsed_seconds"],
                      "peak_rss_kib": report["peak_rss_kib"]}))
    return 0 if report["status"] == "completed" and report["metadata_assignments_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    core.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname(), "requires_coordinator_owned_compute")
    out = args.out.resolve()
    core.require(out.parent == OUTPUT_PARENT.resolve() and not out.exists(), "fresh_episode_attempt_required")
    out.mkdir(parents=True)
    return run_pilot(out) if args.mode == "pilot" else run_full(out)


if __name__ == "__main__":
    raise SystemExit(main())
