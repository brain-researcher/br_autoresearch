"""Manually bound draw-only retries; reuse the successful original decode.

The private caller instance changes execution paths/resources/bindings only.
Its original draw composition, canonical batch output and science validator
are reused. Importing this file does not alter the live original caller.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location(
    "_ep21_draw_retry_caller", Path(__file__).with_name("run_amended_full_readout.py"))
caller = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = caller
_spec.loader.exec_module(caller)
original_stage, original_profile = caller._stage, caller._profile
original_completed = caller._completed
ROOT = original_stage("draw", 0).root
ATTEMPT = "attempt001"
RETRY_ROOT = ROOT / "draw_retries" / ATTEMPT
SELECTION = RETRY_ROOT / "selection.json"
INDICES = (0, 1, 62)


def failure_exit_code(index):
    """Original retry diagnoses; a private adapter may bind a narrower repair."""
    return "2" if index == 62 else "124"


def selection():
    value = json.loads(caller._text(SELECTION))
    caller._require(value.get("authorized") is True
        and value.get("attempt") == ATTEMPT
        and value.get("batch_indices") == list(INDICES)
        and caller._job_id(value.get("array_job_id"))
        and isinstance(value.get("approval_source"), str)
        and bool(value["approval_source"].strip()),
        "activated_specific_draw_retry_selection_required")
    old = json.loads(caller._text(ROOT / "dependency_bindings.json"))
    caller._require(value.get("original_decode_job_id") == old["decode"]["job_id"]
        and value.get("original_array_job_id") == old["draw"]["array_job_id"]
        and value["array_job_id"] not in {
            old["decode"]["job_id"], old["draw"]["array_job_id"], old["finalize"]["job_id"]},
        "fresh_array_with_original_successful_decode_required")
    failures = value.get("failed_original_children", {})
    caller._require(set(failures) == {str(i) for i in INDICES}
        and all(failures[str(i)].get("state") == "FAILED"
            and failures[str(i)].get("exit_code") == failure_exit_code(i) + ":0"
            for i in INDICES), "diagnosed_original_failures_required")
    return value


def stage(phase, batch_index=None, resources=None):
    caller._require(phase == "draw" and type(batch_index) is int
        and batch_index in INDICES, "selected_draw_only_no_decode_or_finalizer")
    return caller.AmendedStage("draw", batch_index, SELECTION, ROOT,
        RETRY_ROOT / f"batch_{batch_index:03d}",
        f"ep21_draw_retry_{ATTEMPT}_{batch_index:03d}", True,
        () if resources is None else tuple(sorted(resources.items())))


def profile(selected_stage):
    chosen = selection()
    science = original_profile(original_stage("draw", selected_stage.batch_index))
    # Retain the entire original scope and every cap except the measured clock.
    resources = {phase: dict(values) for phase, values in science["resources"].items()}
    resources["draw"].update(minutes=2880, deadline_seconds=171000)
    resources = caller._resource_plan(resources)
    return {**science, "resources": resources, "draw_retry_selection": chosen}


def bindings(selected_stage, allocation):
    chosen = selection()
    old = json.loads(caller._text(ROOT / "dependency_bindings.json"))
    fields = dict(token.split("=", 1) for token in
        caller._text(selected_stage.run_dir / "allocation.txt").split() if "=" in token)
    expected = {"SLURM_ARRAY_JOB_ID": chosen["array_job_id"],
        "SLURM_ARRAY_TASK_ID": str(selected_stage.batch_index),
        "SLURM_ARRAY_TASK_COUNT": str(len(INDICES)),
        "SLURM_ARRAY_TASK_MIN": str(min(INDICES)),
        "SLURM_ARRAY_TASK_MAX": str(max(INDICES))}
    caller._require(all(os.environ.get(key) == value for key, value in expected.items())
        and fields.get("ArrayJobId") == chosen["array_job_id"]
        and fields.get("ArrayTaskId") == str(selected_stage.batch_index)
        and fields.get("JobId") == allocation["job_id"],
        "actual_selected_retry_array_child_required")
    return {"decode": old["decode"], "draw": {
        "array_job_id": chosen["array_job_id"],
        "completed_decode_job_id": old["decode"]["job_id"]}}


def completed(selected_stage, binding):
    original_completed(selected_stage, binding)
    # A retry can replace only an explicitly failed original execution.
    failed = ROOT / "draws" / f"batch_{selected_stage.batch_index:03d}" / "batch_status.txt"
    expected = failure_exit_code(selected_stage.batch_index)
    records = caller._status(failed, ("batch_exit", "status"))
    caller._require(records == {"batch_exit": [expected], "status": ["closed"]},
        "preserved_closed_failed_original_required")


# Only this freshly loaded caller instance is adapted. Existing worker modules
# and the original authorization JSON are never modified by a retry process.
caller._stage, caller._profile = stage, profile
caller._bindings, caller._completed = bindings, completed


if __name__ == "__main__":
    raise SystemExit(caller.main())
