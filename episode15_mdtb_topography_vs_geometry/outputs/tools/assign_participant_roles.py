#!/usr/bin/env python3
"""Trusted exact 12/12 assignment; never print participant numeric inputs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import io
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import time


SOURCE_RELATIVE = "data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0/source"
PRIVATE_RELATIVE = "cache/task_a_private_reliability/attempt001/task_a_reliability.csv"
PRIVATE_FIELD = "Task_A_median_centered_spatial_correlation"
CATEGORY_FIELDS = ("sex", "native_english", "ses-rest")
MISSING = {"", "n/a", "na", "nan", "none"}
EXPECTED_ASSIGNMENTS = 2704156
TABLE_BOUND = 1024 * 1024
OBJECTIVES = (
    "maximum_categorical_level_count_imbalance",
    "sum_categorical_level_count_imbalance",
    "standardized_age_mean_difference", "standardized_age_Wasserstein_distance",
    "standardized_Task_A_reliability_mean_difference",
    "standardized_Task_A_reliability_Wasserstein_distance",
)


class AssignmentFailure(ValueError):
    """Only fixed codes are permitted in public failure reports."""
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def bounded_read(path, bound, report=None):
    with Path(path).open("rb") as stream:
        payload = stream.read(bound + 1)
    if report is not None:
        report["text_bytes_read"] += len(payload)
    if len(payload) > bound:
        raise AssignmentFailure("text_input_exceeds_bound")
    return payload


def require_lineage_review(confirmed, section, log_path, report=None):
    if not confirmed or not section or "\n" in section or section.strip() != section:
        raise AssignmentFailure("documented_lineage_review_required")
    lines = bounded_read(log_path, 4 * TABLE_BOUND, report).decode("utf-8").splitlines()
    heading = "## " + section
    positions = [i for i, line in enumerate(lines) if line == heading]
    if len(positions) != 1:
        raise AssignmentFailure("lineage_review_section_not_unique")
    start = positions[0] + 1
    stop = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    if not any(line.strip() for line in lines[start:stop]):
        raise AssignmentFailure("lineage_review_section_empty")
    return "outputs/experiment_log.md#" + section


def cohort_ids(ids, expected=24):
    ids = list(ids)
    if (len(ids) != expected or len(set(ids)) != expected or ids != sorted(ids)
            or any(not re.fullmatch(r"sub-\d{2}", pid) for pid in ids)):
        raise AssignmentFailure("complete_unique_ordered_cohort_required")
    return ids


def table_rows(payload, fields, ids, delimiter):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter=delimiter)
    if len(reader.fieldnames or []) != len(fields) or set(reader.fieldnames or []) != set(fields):
        raise AssignmentFailure("unexpected_private_input_schema")
    rows = list(reader)
    if (len(rows) != len(ids) or any(None in row or any(v is None for v in row.values()) for row in rows)
            or Counter(row.get("participant_id") for row in rows) != Counter(ids)):
        raise AssignmentFailure("private_input_cohort_mismatch")
    return {row["participant_id"]: row for row in rows}


def exact_number(text):
    value = text.strip()
    if value.lower() in MISSING or len(value) > 128:
        raise AssignmentFailure("missing_or_invalid_numeric_input")
    try:
        decimal = Decimal(value)
        if not decimal.is_finite() or not math.isfinite(float(decimal)):
            raise AssignmentFailure("missing_or_invalid_numeric_input")
        # Construction is exact and does not use Decimal arithmetic/context.
        return Fraction(decimal)
    except (InvalidOperation, ValueError, OverflowError):
        raise AssignmentFailure("missing_or_invalid_numeric_input") from None


def parse_private_inputs(demographic_payload, reliability_payload, ids):
    demographics = table_rows(demographic_payload, ("participant_id", "age") + CATEGORY_FIELDS, ids, "\t")
    reliability = table_rows(reliability_payload, ("participant_id", PRIVATE_FIELD), ids, ",")
    categories = []
    for field in CATEGORY_FIELDS:
        values = []
        for pid in ids:
            value = demographics[pid][field].strip()
            if len(value) > 128 or any(ord(char) < 32 for char in value):
                raise AssignmentFailure("invalid_categorical_input")
            # Tagged levels distinguish missingness from every literal code.
            values.append(("missing", "") if value.lower() in MISSING else ("observed", value))
        categories.append(values)
    ages = [exact_number(demographics[pid]["age"]) for pid in ids]
    reliabilities = [exact_number(reliability[pid][PRIVATE_FIELD]) for pid in ids]
    return categories, ages, reliabilities


def common_integers(values):
    scale = math.lcm(*(value.denominator for value in values))
    scaled = [value.numerator * (scale // value.denominator) for value in values]
    # Remove a common offset exactly; differences and variance are unchanged.
    origin = min(scaled)
    return tuple(value - origin for value in scaled)


def wasserstein_numerator(values, audit, development):
    return sum(abs(a - d) for a, d in zip(sorted(values[i] for i in audit),
                                         sorted(values[i] for i in development)))


def enumerate_roles(ids, categories, ages, reliabilities, on_progress=lambda count: None):
    """Generic even-N kernel for focused fixtures; main requires exactly24."""
    n = len(ids)
    if n < 2 or n % 2 or ids != sorted(ids) or len(set(ids)) != n:
        raise AssignmentFailure("invalid_enumeration_cohort")
    if (len(ages) != n or len(reliabilities) != n or len(categories) != 3
            or any(len(values) != n for values in categories)):
        raise AssignmentFailure("incomplete_enumeration_inputs")
    half = n // 2
    category_masks = []
    for values in categories:
        for level in sorted(set(values)):
            mask = sum(1 << i for i, value in enumerate(values) if value == level)
            category_masks.append((mask, mask.bit_count()))
    age = common_integers(ages)
    reliability = common_integers(reliabilities)
    age_total, reliability_total = sum(age), sum(reliability)
    age_constant = not any(age)
    reliability_constant = not any(reliability)
    best, best_audit = None, None
    count = 0
    for audit in combinations(range(n), half):
        count += 1
        if count % 4096 == 0:
            on_progress(count)
        mask = sum(1 << i for i in audit)
        differences = [abs(2 * (mask & level_mask).bit_count() - total)
                       for level_mask, total in category_masks]
        prefix = (max(differences), sum(differences))
        if best is not None and prefix > best[:2]:
            continue
        prefix += (abs(2 * sum(age[i] for i in audit) - age_total),)
        if best is not None and prefix > best[:3]:
            continue
        development = tuple(i for i in range(n) if not mask & (1 << i))
        prefix += (0 if age_constant else wasserstein_numerator(age, audit, development),)
        if best is not None and prefix > best[:4]:
            continue
        prefix += (abs(2 * sum(reliability[i] for i in audit) - reliability_total),)
        if best is not None and prefix > best[:5]:
            continue
        score = prefix + (0 if reliability_constant else wasserstein_numerator(reliability, audit, development),)
        # combinations visits sorted audit-ID tuples in lexicographic order.
        if best is None or score < best:
            best, best_audit = score, audit
    if count != math.comb(n, half) or best_audit is None:
        raise AssignmentFailure("incomplete_exhaustive_enumeration")
    on_progress(count)
    return {"audit_indices": best_audit, "objective_integers": best,
            "assignments_enumerated": count, "enumeration_complete": True,
            "age_integers": age, "reliability_integers": reliability}


def standardized(numerator, values):
    n = len(values)
    variance_numerator = n * sum(value * value for value in values) - sum(values) ** 2
    if variance_numerator == 0:
        if numerator != 0:
            raise AssignmentFailure("inconsistent_zero_spread_objective")
        return 0.0
    # numerator/(n/2) divided by population SD; rational squaring avoids
    # overflow and only the final bounded standardized aggregate is rounded.
    squared = Fraction((numerator * n) ** 2, (n // 2) ** 2 * variance_numerator)
    return math.sqrt(float(squared))


def public_roles(ids, categories, selected):
    cohort_ids(ids)
    if (selected.get("enumeration_complete") is not True
            or selected.get("assignments_enumerated") != EXPECTED_ASSIGNMENTS):
        raise AssignmentFailure("complete_enumeration_required_for_publication")
    audit = tuple(selected["audit_indices"])
    if len(audit) != 12 or len(set(audit)) != 12 or audit != tuple(sorted(audit)) or any(i < 0 or i >= 24 for i in audit):
        raise AssignmentFailure("invalid_final_partition")
    development = tuple(i for i in range(24) if i not in audit)
    score = selected["objective_integers"]
    objective = [score[0], score[1]] + [standardized(score[i], selected[key]) for i, key in (
        (2, "age_integers"), (3, "age_integers"),
        (4, "reliability_integers"), (5, "reliability_integers"))]
    counts = {}
    for field, values in zip(CATEGORY_FIELDS, categories):
        counts[field] = [{"level_kind": level[0], "level": level[1],
                          "audit_count": sum(values[i] == level for i in audit),
                          "development_count": sum(values[i] == level for i in development)}
                         for level in sorted(set(values))]
    return {
        "analysis_id": "role_assignment", "participants": 24,
        "audit_ids": [ids[i] for i in audit], "development_ids": [ids[i] for i in development],
        "assignments_enumerated": EXPECTED_ASSIGNMENTS, "enumeration_complete": True,
        "aggregate_balance_objective": dict(zip(OBJECTIVES, objective)), "categorical_counts": counts,
        "selection_arithmetic": "exact_serialized_decimal_rationals_scaled_to_integers",
        "numeric_standardization": "all24_population_SD_constant_terms_zero",
        "Wasserstein_order": 1, "mean_difference": "absolute", "tie": "lexicographically_first_sorted_audit_ID_tuple",
        "participant_numeric_inputs_exported": False, "role_specific_numeric_summaries_exported": False,
        "Task_B_values_read": False, "physical_handoffs_created": False,
        "scientific_margins_activated": False, "candidate_scoring_authorized": False,
    }


def publish_roles(payload, output):
    if (payload.get("enumeration_complete") is not True
            or payload.get("assignments_enumerated") != EXPECTED_ASSIGNMENTS):
        raise AssignmentFailure("complete_enumeration_required_for_publication")
    final = Path(output) / "participant_roles.json"
    temporary = Path(output) / ".participant_roles.json.partial"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)


def validate_prerequisites(joined, private_report, source, private_path):
    if (joined.get("metadata_join_complete") is not True
            or joined.get("all_selected_Task_A_headers_equal_native_cerebellar_mask_grid") is not True
            or Path(joined["source"]).resolve() != source.resolve()):
        raise AssignmentFailure("completed_source_metadata_required")
    ids = cohort_ids(sorted(row["participant_id"] for row in joined["participants"]))
    if joined["demographic_completeness"].get("age_available_for_future_role_assignment") is not True:
        raise AssignmentFailure("complete_age_metadata_required")
    required_true = ("private_artifact_published", "all24_reliability_computable", "all24_half_maps_finite_and_nonconstant")
    if (private_report.get("status") != "complete_all24_private_Task_A_reliability"
            or any(private_report.get(key) is not True for key in required_true)
            or any(private_report.get(key) != 24 for key in ("participants_expected", "participants_completed", "private_reliability_records"))
            or private_report.get("fixed_support_points") != 208
            or private_report.get("support_changed") is not False
            or private_report.get("Task_B_values_read") is not False
            or private_report.get("Task_B_image_streams_opened") != 0
            or Path(private_report["private_artifact_path"]).resolve() != private_path.resolve()):
        raise AssignmentFailure("complete_private_reliability_preparation_required")
    return ids


def deadline(signum, frame):
    raise TimeoutError("Fixed role assignment deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lineage-review-complete", action="store_true")
    parser.add_argument("--lineage-review-section", required=True)
    parser.add_argument("--max-seconds", type=int, default=540)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute allocation")
    if not 1 <= args.max_seconds <= 540:
        raise ValueError("Internal deadline must be between one and 540 seconds")
    outputs = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.parent != outputs / "runs/role_assignment":
        raise ValueError("A fresh attempt under this episode's role_assignment directory is required")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "analysis_id": "role_assignment", "status": "started", "operation": "lineage_review_gate",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "internal_deadline_seconds": args.max_seconds, "expected_assignments": EXPECTED_ASSIGNMENTS,
        "assignments_enumerated": 0, "enumeration_complete": False, "participant_roles_published": False,
        "interrupted_enumeration_count_is_last_4096_tuple_checkpoint": True,
        "text_bytes_read": 0, "Task_B_values_read": False, "image_streams_opened": 0,
        "participant_numeric_values_exported": False, "scientific_margins_activated": False,
        "candidate_scoring_authorized": False, "physical_handoffs_created": False,
        "private_boundary": "same_UID_logical_trusted_assigner_not_OS_seal", "automatic_retry": False,
        "preparation_scratch_budget_bytes": 2 * 1024 ** 3,
    }
    exit_code = 0
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        report["lineage_review_reference"] = require_lineage_review(
            args.lineage_review_complete, args.lineage_review_section, outputs / "experiment_log.md", report)
        report["operation"] = "completed_operational_prerequisites"
        stage = outputs / "runs/source_qualification"
        joined = json.loads(bounded_read(stage / "metadata_join001/source_metadata.json", 8 * TABLE_BOUND, report))
        private_report = json.loads(bounded_read(stage / "private_task_a001/private_task_a_reliability_report.json", TABLE_BOUND, report))
        source = Path(os.environ["OAK"]) / SOURCE_RELATIVE
        private_path = outputs / PRIVATE_RELATIVE
        ids = validate_prerequisites(joined, private_report, source, private_path)
        report["operation"] = "trusted_demographic_and_private_reliability_read"
        categories, ages, reliabilities = parse_private_inputs(
            bounded_read(source / "participants.tsv", TABLE_BOUND, report),
            bounded_read(private_path, TABLE_BOUND, report), ids)
        report["operation"] = "exhaustive_12_12_lexicographic_assignment"
        selected = enumerate_roles(ids, categories, ages, reliabilities,
                                   lambda count: report.update(assignments_enumerated=count))
        report["enumeration_complete"] = True
        payload = public_roles(ids, categories, selected)
        payload["lineage_review_reference"] = report["lineage_review_reference"]
        report["operation"] = "complete_partition_publication"
        publish_roles(payload, output)
        del categories, ages, reliabilities, selected
        report["status"] = "complete_12_12_role_assignment"
    except Exception as error:
        report.update(status="role_assignment_failure", failure_category=type(error).__name__,
                      failure_code=error.code if isinstance(error, AssignmentFailure) else "bounded_assignment_or_publication_failure")
        exit_code = 1
    finally:
        signal.alarm(0)
        report["participant_roles_published"] = (output / "participant_roles.json").is_file()
        report.update(elapsed_seconds=time.monotonic() - started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "role_assignment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "assignments_enumerated", "participant_roles_published",
                                                 "elapsed_seconds", "max_rss_kib")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
