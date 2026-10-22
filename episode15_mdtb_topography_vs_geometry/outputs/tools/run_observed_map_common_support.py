"""Adopted EP15 followup; every run requires its separate policy amendment.

Reuses the existing prediction freeze and original B-only role manifest. The
complete strict-availability scan is frozen with its observations before any
prediction loss. A failed later report can resume from that freeze without
reopening B. Original prediction, failure and access-history files are read-only.
"""
import argparse
from datetime import datetime
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from calibration_falsifiers import TaskBObservation
from observed_map_audit_context import AUDIT_IDS, CONDITIONS
from observed_map_common_support import (
    ADOPTION_KEY, FULL_SUPPORT, ROUTE_ID, require_common_adoption,
    scan_audit_support, validate_support_record,
)
from report_observed_map_common_support import report_common_support
import run_observed_map_audit as original


FOLLOWUP_ROOT = original.RUN_ROOT / "common_support001"
SUPPORT_DIR = FOLLOWUP_ROOT / "frozen_support"
RESULT_DIR = FOLLOWUP_ROOT / "evaluation"
EXECUTION_PATH = FOLLOWUP_ROOT / "execution.json"
OBSERVATION_BYTES = 4 * 1024 * 1024
_require, _bounded, _write_new = original._require, original._bounded, original._write_new


def actual_policy():
    policy = original.actual_policy()  # Compute boundary and actual episode path.
    require_common_adoption(policy)
    return policy


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.utcoffset() is not None, "timezone_aware_freeze_and_access_times_required")
    return parsed


def _origin(predictions, evaluation_manifest, amendment):
    prior = json.loads(_bounded(original.EXECUTION_PATH, original.STATE_BYTES))
    _require(prior.get("status") == "failed" and prior.get("attempts")
             and prior.get("binding", {}).get("prediction_dir") == str(original.PREDICTION_DIR)
             and prior["binding"].get("evaluation_manifest") == str(original._path(evaluation_manifest)),
             "preserved_failed_original_audit_and_identical_frozen_prediction_manifest_required")
    frozen, opened = predictions["frozen_at_utc"], prior["first_B_opening_utc"]
    _require(_timestamp(frozen) <= _timestamp(opened) <= _timestamp(amendment["adopted_at_utc"]),
             "original_predictions_precede_original_B_access_and_followup_adoption")
    return {"prediction_dir": str(original.PREDICTION_DIR), "prediction_frozen_at_utc": frozen,
            "original_execution_path": str(original.EXECUTION_PATH), "original_first_B_opening_utc": opened,
            "original_attempt_count": len(prior["attempts"]),
            "original188_audit_status": prior["status"]}


def _freeze_support(scan, *, origin, check, durable_byte_cap):
    _require(not SUPPORT_DIR.exists(), "common_support_is_already_frozen")
    record = {"status": "complete_strict_availability_intersection", "route_id": ROUTE_ID,
        "participant_ids": AUDIT_IDS, "condition_ids": CONDITIONS,
        "original_voxel_ids": scan.original_voxel_ids, "keep": scan.keep.tolist(),
        "retained_voxel_ids": tuple(np.asarray(scan.original_voxel_ids)[scan.keep].tolist()),
        "retained_count": int(scan.keep.sum()), "selected_member_counts": scan.selected_member_counts,
        "all_selected_members_scanned": True, "predictive_losses_computed": False,
        "frozen_at_utc": original._now(), "job_id": os.environ["SLURM_JOB_ID"],
        "selection_rule": "strict_positive_weight_stencil_valid_and_finite_in_every_required_selected_B_regressor",
        "model_or_loss_values_used_for_selection": False, "joint_across_all12_audit_people": True,
        "original": origin}
    validate_support_record(record, scan.original_voxel_ids)
    _require(len(scan.observations) == 12
             and tuple(item.participant_id for item in scan.observations) == AUDIT_IDS
             and all(tuple(item.task_b_condition_ids) == CONDITIONS
                     and tuple(item.voxel_ids) == scan.original_voxel_ids for item in scan.observations),
             "complete_observation_scan_required")
    observations = np.stack([np.asarray(item.task_b_partitions) for item in scan.observations])
    _require(observations.shape == (12, 4, 18, FULL_SUPPORT)
             and np.isfinite(observations[..., scan.keep]).all(),
             "every_retained_partition_map_must_be_finite")
    staging = Path(tempfile.mkdtemp(prefix=".support-incomplete-", dir=FOLLOWUP_ROOT))
    with (staging / "observations.npz").open("xb") as stream:
        np.savez(stream, observations=observations)
        stream.flush()
        os.fsync(stream.fileno())
    _write_new(staging / "support.json", (json.dumps(original._jsonable(record), allow_nan=False, indent=2) + "\n").encode())
    _require(sum((staging / name).stat().st_size for name in ("support.json", "observations.npz")) <= durable_byte_cap,
             "common_support_durable_byte_limit_reached")
    check()
    _require(not SUPPORT_DIR.exists(), "common_support_is_already_frozen")
    staging.rename(SUPPORT_DIR)


def _load_support(predictions, *, origin, check):
    check()
    record = json.loads(_bounded(SUPPORT_DIR / "support.json", original.METADATA_BYTES))
    keep = validate_support_record(record, predictions["voxel_ids"])
    _require(record.get("original") == origin, "frozen_support_must_bind_the_same_original_audit")
    raw = _bounded(SUPPORT_DIR / "observations.npz", OBSERVATION_BYTES)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        _require(len(entries) == 1 and entries[0].filename == "observations.npy"
                 and entries[0].file_size <= OBSERVATION_BYTES, "bounded_frozen_observations_only")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        values = np.array(archive["observations"], copy=True)
    _require(values.shape == (12, 4, 18, FULL_SUPPORT) and values.dtype.kind in "fiu"
             and np.isfinite(values[..., keep]).all(), "complete_finite_retained_observation_panel_required")
    observations = tuple(TaskBObservation(person, CONDITIONS, tuple(predictions["voxel_ids"]), values[index])
                         for index, person in enumerate(AUDIT_IDS))
    check()
    return record, observations


def _completed():
    record = json.loads(_bounded(RESULT_DIR / "report/observed_map_common_support.complete.json", original.ARRAY_BYTES))
    _require(record.get("status") in ("complete_exploratory_common_support_report", "scientific_stop_insufficient_common_support")
             and tuple(record.get("participant_ids", ())) == AUDIT_IDS
             and all((RESULT_DIR / name).is_file() for name in (
                 "report/3lines.md", "figures/3lines.md", "figures/observed_map_common_support.png",
                 "figures/observed_map_common_support.pdf", "access_history.json")),
             "complete_followup_report_figure_and_access_history_required")
    return {"status": record["status"], "result_dir": str(RESULT_DIR)}


def evaluate(*, evaluation_manifest, policy, check, durable_byte_cap, engineering_resume_reason=None):
    amendment = require_common_adoption(policy)  # Before every reader or output.
    _require(callable(check) and type(durable_byte_cap) is int and durable_byte_cap > 0,
             "explicit_resource_check_and_positive_durable_cap_required")
    _require(engineering_resume_reason is None or (isinstance(engineering_resume_reason, str)
             and 0 < len(engineering_resume_reason.strip()) <= 2000), "bounded_resume_reason_required")
    predictions = original.load_frozen_predictions(policy=policy, check=check)
    origin = _origin(predictions, evaluation_manifest, amendment)
    binding = {"original": origin, "evaluation_manifest": str(original._path(evaluation_manifest)),
               "adoption_key": ADOPTION_KEY, "adopted_at_utc": amendment["adopted_at_utc"]}
    FOLLOWUP_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    descriptor = os.open(EXECUTION_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = stream.read(original.STATE_BYTES + 1)
        _require(len(prior.encode()) <= original.STATE_BYTES, "bounded_execution_record_required")
        state = json.loads(prior) if prior else None
        if state is not None:
            _require(state.get("binding") == binding, "followup_resume_requires_identical_origin_and_manifest")
        if RESULT_DIR.exists():
            _require(state is not None, "published_followup_requires_its_access_record")
            result = _completed()
            state["status"] = "complete"
            original._save_state(stream, state)
            return result
        if state is None:
            _require(engineering_resume_reason is None and not SUPPORT_DIR.exists(),
                     "first_followup_is_not_a_resume_or_unrecorded_support")
            state = {"binding": binding, "attempts": [], "renewed_B_openings": [],
                     "separation": "procedural_same_account_not_technically_enforced"}
        else:
            _require(state.get("status") in ("failed", "incomplete") and engineering_resume_reason is not None,
                     "explicit_engineering_resume_required_after_prior_followup_attempt")
        attempt = {"started_utc": original._now(), "job_id": os.environ["SLURM_JOB_ID"],
                   "engineering_resume_reason": engineering_resume_reason,
                   "prior_Task_B_access_disclosed": True, "reuse_frozen_support": SUPPORT_DIR.exists()}
        state["attempts"].append(attempt)
        state["status"] = "incomplete"
        original._save_state(stream, state)
        try:
            if not SUPPORT_DIR.exists():
                opened = original._now()
                state["renewed_B_openings"].append({"opened_at_utc": opened, "job_id": os.environ["SLURM_JOB_ID"]})
                original._save_state(stream, state)  # Before even B metadata.
                scan = scan_audit_support(policy=policy, frozen_predictions=predictions,
                    **original.load_audit_inputs(evaluation_manifest, policy=policy, calibration=False, check=check), check=check)
                _freeze_support(scan, origin=origin, check=check, durable_byte_cap=durable_byte_cap)
                del scan
            record, observations = _load_support(predictions, origin=origin, check=check)
            _require(state["renewed_B_openings"]
                     and _timestamp(state["renewed_B_openings"][-1]["opened_at_utc"]) <= _timestamp(record["frozen_at_utc"]),
                     "complete_support_freeze_must_follow_recorded_B_access")
            attempt["support_frozen_at_utc"] = record["frozen_at_utc"]
            attempt["scoring_started_utc"] = original._now()
            original._save_state(stream, state)  # Durable freeze precedes scoring.
            staging = Path(tempfile.mkdtemp(prefix=".evaluation-incomplete-", dir=FOLLOWUP_ROOT))
            result = report_common_support(predictions, observations, support_record=record, policy=policy,
                output_root=staging / "report", figure_root=staging / "figures", check=check)
            attempt.update(status="complete", finished_utc=original._now())
            completed_state = dict(state, status="complete")
            _write_new(staging / "access_history.json", (json.dumps(completed_state, allow_nan=False, indent=2) + "\n").encode())
            names = ("report/observed_map_common_support.complete.json", "report/3lines.md", "figures/3lines.md",
                     "figures/observed_map_common_support.png", "figures/observed_map_common_support.pdf", "access_history.json")
            total = sum((staging / name).stat().st_size for name in names)
            total += sum((SUPPORT_DIR / name).stat().st_size for name in ("support.json", "observations.npz"))
            _require(total <= durable_byte_cap, "complete_followup_durable_byte_limit_reached")
            check()
            _require(not RESULT_DIR.exists(), "followup_result_already_published")
            staging.rename(RESULT_DIR)
            state.update(completed_state)
            original._save_state(stream, state)
            return {"status": result["status"], "result_dir": str(RESULT_DIR)}
        except BaseException as error:
            attempt.update(status="failed", error_class=type(error).__name__, finished_utc=original._now())
            state["status"] = "failed"
            original._save_state(stream, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--engineering-resume-reason")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-kib", type=int, required=True)
    parser.add_argument("--durable-byte-cap", type=int, required=True)
    args = parser.parse_args()
    policy = actual_policy()
    with original.resource_check(deadline_epoch=args.deadline_epoch, rss_limit_kib=args.rss_limit_kib) as check:
        result = evaluate(evaluation_manifest=args.evaluation_manifest, policy=policy, check=check,
            durable_byte_cap=args.durable_byte_cap, engineering_resume_reason=args.engineering_resume_reason)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
