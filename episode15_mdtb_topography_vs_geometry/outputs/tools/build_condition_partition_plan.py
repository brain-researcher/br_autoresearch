#!/usr/bin/env python3
"""Compile fixed EP15 condition, run-partition and role labels from public metadata."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import time

from join_source_metadata import (DOMAINS, EXPECTED_BETAS, PLAN_COLUMNS, SESSIONS,
                                  beta_member, validate_session, validate_task_sets)


INPUTS = {
    "metadata": ("runs/source_qualification/metadata_join001/source_metadata.json", 8*1024**2),
    "members": ("runs/source_qualification/metadata_join001/regressor_members.tsv", 32*1024**2),
    "role_report": ("runs/role_assignment/attempt001/role_assignment_report.json", 1024**2),
    "roles": ("runs/role_assignment/attempt001/participant_roles.json", 1024**2),
}
INTEGER_FIELDS = ("run", "reg_id", "instruction", "common", "half", "geometry_partition")
EXTRA_COLUMNS = ("participant_role", "planned_handoff", "endpoint_use", "condition_map_included",
    "regressors_in_condition_run", "within_run_weight_numerator", "within_run_weight_denominator",
    "primary_domain", "primary_condition_weight_numerator", "primary_condition_weight_denominator")
MEMBER_COLUMNS = [*PLAN_COLUMNS, *EXTRA_COLUMNS]
STATUS = "complete_condition_partition_member_plan"
EXPECTED_ASSIGNMENTS = 2704156
EXPECTED_ROWS = 24*sum(EXPECTED_BETAS.values())
HALVES = [list(range(1, 9)), list(range(9, 17))]
QUARTERS = [list(range(start, start+4)) for start in (1, 5, 9, 13)]
GEOMETRY_PAIRS = [[1, 2], [3, 4]]
OUTPUT_FILES = ("condition_partition_plan.json", "role_member_plan.tsv")


class PlanFailure(ValueError):
    pass


def bounded_read(path, bound, report):
    with Path(path).open("rb") as stream:
        payload = stream.read(bound+1)
    report["text_bytes_read"] += len(payload)
    if len(payload) > bound:
        raise PlanFailure("metadata_input_exceeds_bound")
    return payload


def load_inputs(outputs, report):
    """Exactly four existing public metadata files; never follow recorded source paths."""
    values = {}
    for key, (relative, bound) in INPUTS.items():
        payload = bounded_read(Path(outputs)/relative, bound, report)
        values[key] = payload if key == "members" else json.loads(payload)
    return values


def validate_prerequisites(metadata, role_report, roles):
    if (metadata.get("metadata_join_complete") is not True
            or metadata.get("beta_or_anatomy_voxel_elements_decoded") is not False
            or metadata.get("Task_B_image_streams_opened") != 0
            or metadata.get("regressor_members_table") != "regressor_members.tsv"):
        raise PlanFailure("completed_Task_B_blind_metadata_required")
    participants = metadata["participants"]
    ids = sorted(row["participant_id"] for row in participants)
    if (len(ids) != 24 or len(set(ids)) != 24
            or any(not re.fullmatch(r"sub-\d{2}", pid) for pid in ids)
            or any(row["beta_members_joined"] != EXPECTED_BETAS for row in participants)):
        raise PlanFailure("complete_original_24_person_metadata_required")
    if (role_report.get("status") != "complete_12_12_role_assignment"
            or role_report.get("enumeration_complete") is not True
            or role_report.get("participant_roles_published") is not True
            or role_report.get("assignments_enumerated") != EXPECTED_ASSIGNMENTS
            or roles.get("enumeration_complete") is not True
            or roles.get("assignments_enumerated") != EXPECTED_ASSIGNMENTS
            or roles.get("participants") != 24):
        raise PlanFailure("complete_exhaustive_role_publication_required")
    for item in (role_report, roles):
        if any(item.get(key) is not False for key in ("Task_B_values_read", "physical_handoffs_created",
                "scientific_margins_activated", "candidate_scoring_authorized")):
            raise PlanFailure("unchanged_pre_handoff_access_stage_required")
    audit, development = roles["audit_ids"], roles["development_ids"]
    if (len(audit) != 12 or len(development) != 12 or audit != sorted(set(audit))
            or development != sorted(set(development)) or set(audit)&set(development)
            or sorted(audit+development) != ids):
        raise PlanFailure("complete_disjoint_fixed_12_12_roles_required")
    return ids, {pid: "audit" if pid in audit else "development" for pid in ids}


def parse_members(payload, ids, check=lambda: None):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    if reader.fieldnames != PLAN_COLUMNS:
        raise PlanFailure("original_member_table_columns_required")
    grouped, seen = defaultdict(list), set()
    for index, source in enumerate(reader):
        if index % 1024 == 0:
            check()
        if index >= EXPECTED_ROWS or None in source or any(value is None for value in source.values()):
            raise PlanFailure("unexpected_or_incomplete_member_row")
        row = dict(source)
        try:
            for key in INTEGER_FIELDS:
                row[key] = int(row[key])
        except ValueError as error:
            raise PlanFailure("noninteger_member_identity_or_partition") from error
        pid, session = row["participant_id"], row["session"]
        run, reg = row["run"], row["reg_id"]
        key = (pid, session, run, reg)
        if (pid not in ids or session not in SESSIONS or row["task_set"] != SESSIONS[session]
                or not 1 <= run <= 16 or reg < 0 or key in seen
                or row["instruction"] not in (0, 1) or row["common"] not in (0, 1)
                or bool(row["instruction"]) != (row["task_code"] == "instrct")
                or row["half"] != (1 if run <= 8 else 2)
                or row["geometry_partition"] != (run-1)//4+1
                or row["beta_member"] != beta_member(pid, session, run, reg)
                or not row["task_name"] or not row["cond_name"]):
            raise PlanFailure("member_identity_or_frozen_partition_mismatch")
        seen.add(key)
        grouped[(pid, session)].append(row)
    if len(seen) != EXPECTED_ROWS or set(grouped) != {(pid, session) for pid in ids for session in SESSIONS}:
        raise PlanFailure("all_36480_members_and_48_sessions_required")
    return grouped


def compile_plan(metadata, role_report, roles, members, check=lambda: None):
    ids, role_by_id = validate_prerequisites(metadata, role_report, roles)
    grouped = parse_members(members, ids, check)
    design = None
    rows = []
    domain_by_condition = {condition: domain for domain, conditions in DOMAINS.items() for condition in conditions}
    for pid in ids:
        check()
        summaries = {}
        for session in SESSIONS:
            records = sorted(grouped[(pid, session)], key=lambda row: (row["run"], row["reg_id"]))
            # These counts refer only to the already joined member table. No
            # archive is reopened, and this is not a fresh source-presence claim.
            summaries[session] = validate_session(records, Counter(row["beta_member"] for row in records),
                                                   EXPECTED_BETAS[session])
            if summaries[session] != metadata["condition_multiplicity_by_session"][session]:
                raise PlanFailure("literal_condition_identity_or_multiplicity_changed")
            for record in records:
                row = dict(record)
                task_a, instruction = session == "ses-s1", bool(row["instruction"])
                role = role_by_id[pid]
                handoff = "development" if role == "development" else ("audit_calibration" if task_a else "sealed_audit")
                primary = not instruction and not task_a and row["cond_name"] in domain_by_condition
                if instruction:
                    use, multiplicity, weight = "instruction_excluded", 0, {"numerator": 0, "denominator": 1}
                else:
                    multiplicity = summaries[session][row["cond_name"]]["regressors_per_run"][row["run"]-1]
                    use = "Task_A_calibration" if task_a else ("Task_B_primary" if primary else "Task_B_shared_postdecision_diagnostic")
                    weight = {"numerator": 1, "denominator": multiplicity}
                domain = domain_by_condition[row["cond_name"]] if primary else ""
                row.update(participant_role=role, planned_handoff=handoff, endpoint_use=use,
                    condition_map_included=int(not instruction), regressors_in_condition_run=multiplicity,
                    within_run_weight_numerator=weight["numerator"], within_run_weight_denominator=weight["denominator"],
                    primary_domain=domain, primary_condition_weight_numerator=int(primary),
                    primary_condition_weight_denominator=7*len(DOMAINS[domain]) if primary else 1)
                rows.append(row)
        current = validate_task_sets(summaries["ses-s1"], summaries["ses-s2"])
        if (len(current["Task_A_conditions"]) != 29 or len(current["Task_B_conditions"]) != 32
                or current != metadata["condition_design"]
                or current["map_halves"] != HALVES or current["geometry_partitions"] != QUARTERS
                or current["primary_geometry_partition_pairs"] != GEOMETRY_PAIRS):
            raise PlanFailure("fixed_29_A_32_B_14_shared_18_B_only_design_required")
        if design is None:
            design = current
        elif current != design:
            raise PlanFailure("condition_design_differs_across_participants")
    weights = {condition: Fraction(value["numerator"], value["denominator"])
               for condition, value in design["condition_weights"].items()}
    if sum(weights.values(), Fraction()) != 1 or any(
            sum((weights[condition] for condition in conditions), Fraction()) != Fraction(1, 7)
            for conditions in DOMAINS.values()):
        raise PlanFailure("exact_seven_domain_condition_weights_required")
    plan = {"analysis_id": "condition_partition_plan", "metadata_only": True,
        "participants": 24, "development_ids": list(roles["development_ids"]), "audit_ids": list(roles["audit_ids"]),
        "condition_design": design, "condition_multiplicity_by_session": metadata["condition_multiplicity_by_session"],
        "member_table": OUTPUT_FILES[1], "member_rows": len(rows),
        "member_counts_by_planned_handoff": dict(Counter(row["planned_handoff"] for row in rows)),
        "noninstruction_member_counts_by_planned_handoff": dict(Counter(row["planned_handoff"] for row in rows if row["condition_map_included"])),
        "map_construction": {"within_run": "arithmetic_mean_of_all_condition_regressors",
            "map_half_run_weight": {"numerator": 1, "denominator": 8},
            "geometry_partition_run_weight": {"numerator": 1, "denominator": 4},
            "condition_maps_remain_separate": True, "source_reg_id_used_literally": True},
        "geometry_pair_weights": {"requirement": "domain_pair_balanced_full_signed_matrix",
            "numerical_weights": None, "diagonal_and_symmetric_entry_convention": None,
            "status": "unresolved_prospective_choice", "activated": False},
        "geometry_sensitivity_pairings": {"status": "not_selected_here", "pairings": None},
        "planned_handoffs": {
            "development": {"participant_role": "development", "contents": "anatomy_Task_A_Task_B",
                "reader": "development_worker", "activation": "after_required_reader_separation"},
            "audit_calibration": {"participant_role": "audit", "contents": "anatomy_Task_A",
                "reader": "calibrator", "activation": "only_after_panel_lock"},
            "sealed_audit": {"participant_role": "audit", "contents": "Task_B",
                "reader": "trusted_evaluator", "activation": "once_after_prediction_lock"}},
        "member_table_scope": "functional_beta_member_names_only; anatomy_manifest_not_created",
        "physical_handoffs_created": False, "reader_isolation_established": False,
        "audit_calibration_activated": False, "audit_Task_B_open_count": 0,
        "source_archive_or_image_opened": False, "participant_numeric_values_exported": False,
        "scientific_margins_activated": False, "candidate_scoring_authorized": False,
        "independent_run_errors_established": False,
        "publication_requirement": "both_plan_files_require_complete_terminal_report"}
    return plan, rows


def publish_plan(plan, rows, output):
    """Only complete validated metadata reaches publication; partial pairs are invalid."""
    output = Path(output)
    if plan["member_rows"] != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
        raise PlanFailure("complete_member_plan_required_for_publication")
    finals = [output/name for name in OUTPUT_FILES]
    if any(path.exists() for path in finals):
        raise FileExistsError("Existing condition/member plan cannot be overwritten")
    temporaries = [output/("."+name+".partial") for name in OUTPUT_FILES]
    try:
        with temporaries[0].open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with temporaries[1].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MEMBER_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        for temporary, final in zip(temporaries, finals):
            os.link(temporary, final)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=120)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified Slurm compute shell")
    if not 1 <= args.max_seconds <= 120:
        raise ValueError("The selected internal bound is at most 120 seconds")
    outputs = Path(__file__).resolve().parents[1]
    if args.output_dir.resolve().parent != (outputs/"runs/source_qualification").resolve():
        raise ValueError("Use a fresh attempt under this episode's source_qualification outputs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"analysis_id": "condition_partition_plan", "status": "started",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "internal_deadline_seconds": args.max_seconds, "text_bytes_read": 0,
        "input_files": [relative for relative, _ in INPUTS.values()],
        "member_rows": 0, "plan_complete": False, "physical_handoffs_created": False,
        "reader_isolation_established": False, "source_archive_or_image_opened": False,
        "Task_B_values_read": False, "participant_numeric_values_exported": False,
        "scientific_margins_activated": False, "candidate_scoring_authorized": False,
        "numeric_geometry_pair_weights_activated": False, "automatic_retry": False,
        "preparation_scratch_budget_bytes": 2*1024**3}
    def deadline(*_):
        raise TimeoutError("condition_partition_plan_deadline")
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    code = 0
    try:
        values = load_inputs(outputs, report)
        plan, rows = compile_plan(values["metadata"], values["role_report"], values["roles"], values["members"], check)
        check()
        publish_plan(plan, rows, args.output_dir)
        check()
        report.update(status=STATUS, plan_complete=True, member_rows=len(rows),
            participant_count=24, session_count=48,
            member_counts_by_planned_handoff=plan["member_counts_by_planned_handoff"],
            noninstruction_member_counts_by_planned_handoff=plan["noninstruction_member_counts_by_planned_handoff"],
            output_files=list(OUTPUT_FILES))
    except Exception as error:
        report.update(status="condition_partition_plan_failure", plan_complete=False, failure_category=type(error).__name__,
            failure_code=str(error) if isinstance(error, PlanFailure) else "bounded_metadata_compilation_or_publication_failure")
        code = 1
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        (args.output_dir/"condition_partition_plan_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if report["status"] != STATUS or report["plan_complete"] is not True:
        code = 1
    print(json.dumps({key: report[key] for key in ("status", "member_rows", "plan_complete", "elapsed_seconds", "max_rss_kib")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
