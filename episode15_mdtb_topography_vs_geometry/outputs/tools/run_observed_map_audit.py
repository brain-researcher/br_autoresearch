"""Inactive two-stage audit runner; the actual episode policy must adopt it.

Prediction and evaluation paths are fixed. A complete A-only prediction bundle
is published before B is opened. One small execution record prevents another
output name from presenting repeated access as a new evaluation. All separation
is procedural within the same account, not technical blinding.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
import zipfile

import numpy as np

from observed_map_audit_context import AUDIT_IDS, require_adoption, validate_prediction_bundle
from observed_map_audit_readers import (
    AuditCalibrationBinding, AuditObservationBinding, FROZEN_SUPPORT_PATH,
    read_audit_calibration, read_audit_observations,
)
from run_development_point_panel import (
    MANIFEST_BYTES, MEMBER_BYTES, _bounded, _compute, _jsonable, _path,
    _write_new, load_released_inputs,
)


OUTPUTS = Path(__file__).resolve().parents[1]
POLICY_PATH = OUTPUTS.parent / "SEARCH_POLICY.yaml"
RUN_ROOT = OUTPUTS / "runs/observed_map_audit"
PREDICTION_DIR = RUN_ROOT / "predictions"
EVALUATION_DIR = RUN_ROOT / "evaluation"
EXECUTION_PATH = RUN_ROOT / "audit_execution.json"
ARRAY_KEYS = ("M0a", "M0b", "predictions", "condition_control_predictions",
              "donor_control_predictions", "random_basis_predictions",
              "matched_warp_predictions", "sampling_native_world")
ARRAY_BYTES = 128 * 1024 * 1024
METADATA_BYTES = 4 * 1024 * 1024
STATE_BYTES = 128 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(timezone.utc).isoformat()


def actual_policy():
    """No caller-supplied approval flag or alternative policy path."""
    _compute()
    import yaml
    policy = yaml.safe_load(_bounded(POLICY_PATH, METADATA_BYTES))
    require_adoption(policy)
    return policy


@contextmanager
def resource_check(*, deadline_epoch, rss_limit_kib):
    _require(math.isfinite(deadline_epoch) and deadline_epoch > time.time()
             and type(rss_limit_kib) is int and rss_limit_kib > 0,
             "finite_future_deadline_and_positive_memory_limit_required")
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "timer_already_active")
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}

    def check():
        _require(time.time() < deadline_epoch
                 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib,
                 "audit_attempt_resource_limit_reached")

    def stop(_signal, _frame):
        raise RuntimeError("audit_attempt_interrupted")

    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_epoch - time.time()))
        check()
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def load_audit_inputs(manifest_path, *, policy, calibration, check):
    require_adoption(policy)
    from build_condition_partition_plan import MEMBER_COLUMNS
    check()
    manifest = json.loads(_bounded(manifest_path, MANIFEST_BYTES))
    _require(isinstance(manifest, dict) and set(manifest) == {
        "release_bindings", "member_rows_path", "condition_multiplicity_by_session"},
        "explicit_single_role_audit_manifest_required")
    records = manifest["release_bindings"]
    fields = {"participant_id", "archive_path"}
    if calibration:
        fields |= {"anatomy_headers", "task_a_header"}
    _require(isinstance(records, list) and len(records) == 12
             and all(isinstance(row, dict) and set(row) == fields for row in records)
             and tuple(row["participant_id"] for row in records) == AUDIT_IDS,
             "all12_original_ordered_audit_bindings_required")
    cls = AuditCalibrationBinding if calibration else AuditObservationBinding
    bindings = tuple(cls(**dict(row, archive_path=_path(row["archive_path"]))) for row in records)
    table = csv.DictReader(io.StringIO(_bounded(manifest["member_rows_path"], MEMBER_BYTES)
                                     .decode("utf-8-sig")), delimiter="\t")
    _require(table.fieldnames == MEMBER_COLUMNS, "original_condition_member_columns_required")
    rows = tuple(table)
    session, handoff = (("ses-s1", "audit_calibration") if calibration else ("ses-s2", "sealed_audit"))
    _require(all(None not in row and all(value is not None for value in row.values())
        and row["participant_id"] in AUDIT_IDS and row["participant_role"] == "audit"
        and row["planned_handoff"] == handoff and row["session"] == session for row in rows),
        "only_requested_audit_role_metadata_permitted")
    summaries = manifest["condition_multiplicity_by_session"]
    _require(isinstance(summaries, dict) and session in summaries,
             "frozen_source_condition_multiplicities_required")
    check()
    return dict(release_bindings=bindings, member_rows=rows,
                condition_multiplicity_by_session={session: summaries[session]})


def _fixed_support(bundle):
    from finite_development_support import load_selection
    selection = load_selection(FROZEN_SUPPORT_PATH)
    _require(tuple(bundle["voxel_ids"]) == tuple(map(int, selection["selected_voxel_ids"])),
             "prediction_axis_must_equal_existing_development_frozen188_axis")


def _publish_predictions(bundle, *, policy, check, durable_byte_cap):
    bundle = validate_prediction_bundle(bundle, policy=policy, require_sampling_geometry=True)
    _fixed_support(bundle)
    _require(not PREDICTION_DIR.exists() and not EXECUTION_PATH.exists(),
             "prediction_freeze_already_exists_or_B_execution_started")
    arrays = {key: bundle[key] for key in ARRAY_KEYS}
    _require(sum(value.nbytes for value in arrays.values()) <= ARRAY_BYTES,
             "bounded_prediction_array_size_required")
    metadata = {key: value for key, value in bundle.items() if key not in ARRAY_KEYS}
    metadata["frozen_at_utc"] = _now()
    metadata["job_id"] = os.environ["SLURM_JOB_ID"]
    metadata["separation"] = "procedural_same_account_not_technically_enforced"
    encoded = (json.dumps(_jsonable(metadata), allow_nan=False, indent=2) + "\n").encode()
    _require(len(encoded) <= METADATA_BYTES, "bounded_prediction_metadata_required")
    staging = Path(tempfile.mkdtemp(prefix=".predictions-incomplete-", dir=RUN_ROOT))
    with (staging / "predictions.npz").open("xb") as destination:
        np.savez(destination, **arrays)
        destination.flush()
        os.fsync(destination.fileno())
    _write_new(staging / "metadata.json", encoded)
    _require(sum((staging / name).stat().st_size for name in ("predictions.npz", "metadata.json"))
             <= durable_byte_cap, "prediction_durable_byte_limit_reached")
    check()
    _require(not PREDICTION_DIR.exists() and not EXECUTION_PATH.exists(),
             "prediction_freeze_already_exists_or_B_execution_started")
    staging.rename(PREDICTION_DIR)
    return {"status": "complete_observed_map_audit_predictions", "prediction_dir": str(PREDICTION_DIR)}


def load_frozen_predictions(*, policy, check):
    require_adoption(policy)
    check()
    _require(PREDICTION_DIR.is_dir(), "complete_prediction_directory_required_before_B")
    metadata = json.loads(_bounded(PREDICTION_DIR / "metadata.json", METADATA_BYTES))
    _require(not set(metadata).intersection(ARRAY_KEYS), "arrays_must_be_in_numeric_bundle")
    raw = _bounded(PREDICTION_DIR / "predictions.npz", ARRAY_BYTES)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        _require(len(entries) == len(ARRAY_KEYS)
                 and {item.filename for item in entries} == {key + ".npy" for key in ARRAY_KEYS}
                 and sum(item.file_size for item in entries) <= ARRAY_BYTES,
                 "only_bounded_frozen_prediction_arrays_permitted")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        bundle = dict(metadata, **{key: np.array(archive[key], copy=True) for key in ARRAY_KEYS})
    checked = validate_prediction_bundle(bundle, policy=policy, require_sampling_geometry=True)
    _fixed_support(checked)
    check()
    return checked


def predict(*, development_manifest, calibration_manifest, policy, check, durable_byte_cap):
    require_adoption(policy)
    _require(not PREDICTION_DIR.exists() and not EXECUTION_PATH.exists(),
             "refitting_forbidden_after_prediction_freeze_or_B_opening")
    from development_handoff_reader import read_development_handoffs
    from finite_development_support import load_selection
    from observed_map_audit_predictions import fit_audit_predictions
    inputs = load_released_inputs(development_manifest, check=check)
    selection = load_selection(FROZEN_SUPPORT_PATH)
    development = read_development_handoffs(**inputs, check=check, derived_support=selection)
    calibration = read_audit_calibration(policy=policy,
        **load_audit_inputs(calibration_manifest, policy=policy, calibration=True, check=check),
        common_support=inputs["common_support"], derived_support=selection, check=check)
    _require(calibration.voxel_ids == development.voxel_ids, "identical_fixed_development_and_audit_axes_required")
    bundle = fit_audit_predictions(development, calibration.task_a_packets,
        common_support=inputs["common_support"], forward_header=development.qualified_forward_header,
        policy=policy, check=check)
    bundle.update(sampling_native_world=calibration.sampling_native_world,
                  sampling_mask_headers=calibration.sampling_mask_headers)
    return _publish_predictions(bundle, policy=policy, check=check, durable_byte_cap=durable_byte_cap)


def _save_state(stream, state):
    payload = json.dumps(state, allow_nan=False, indent=2) + "\n"
    _require(len(payload.encode()) <= STATE_BYTES, "bounded_execution_record_required")
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _completed_evaluation():
    path = EVALUATION_DIR / "report/observed_map_audit.complete.json"
    result = json.loads(_bounded(path, ARRAY_BYTES))
    _require(result.get("status") == "complete_observed_map_audit_report"
             and tuple(result.get("participant_ids", ())) == AUDIT_IDS,
             "published_complete_all12_evaluation_required")
    _require(all((EVALUATION_DIR / relative).is_file() for relative in (
        "report/3lines.md", "figures/3lines.md", "figures/observed_map_audit.png",
        "figures/observed_map_audit.pdf", "access_history.json")),
        "published_complete_report_and_figure_required")
    return {"status": "complete_observed_map_audit_report", "evaluation_dir": str(EVALUATION_DIR)}


def evaluate(*, evaluation_manifest, policy, check, durable_byte_cap, engineering_resume_reason=None):
    require_adoption(policy)
    predictions = load_frozen_predictions(policy=policy, check=check)  # Before B manifest.
    _require(engineering_resume_reason is None or (isinstance(engineering_resume_reason, str)
             and 0 < len(engineering_resume_reason.strip()) <= 2000), "bounded_resume_reason_required")
    binding = {"prediction_dir": str(PREDICTION_DIR), "evaluation_dir": str(EVALUATION_DIR),
               "evaluation_manifest": str(_path(evaluation_manifest))}
    # The one locked file is a scientific access record, not a general receipt tree.
    # O_APPEND would ignore _save_state's seek and concatenate JSON documents.
    # Create without truncation, then replace contents under the same file lock.
    descriptor = os.open(EXECUTION_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        prior = stream.read(STATE_BYTES + 1)
        _require(len(prior.encode()) <= STATE_BYTES, "bounded_execution_record_required")
        state = json.loads(prior) if prior else None
        if state is not None:
            _require(state.get("binding") == binding, "engineering_resume_must_use_identical_freeze_and_paths")
        if EVALUATION_DIR.exists():
            _require(state is not None, "published_evaluation_requires_its_prior_access_record")
            completed = _completed_evaluation()
            state.update(status="complete", completed_utc=state.get("completed_utc", _now()))
            _save_state(stream, state)
            return completed  # No B metadata or values are reopened.
        if state is not None:
            _require(state.get("status") in ("failed", "incomplete") and engineering_resume_reason is not None,
                     "explicit_same_procedure_engineering_resume_required_after_prior_B_access")
        else:
            _require(engineering_resume_reason is None, "first_execution_is_not_a_resume")
            state = {"binding": binding, "first_B_opening_utc": _now(), "attempts": [],
                     "separation": "procedural_same_account_not_technically_enforced"}
        attempt = {"started_utc": _now(), "job_id": os.environ["SLURM_JOB_ID"],
                   "engineering_resume_reason": engineering_resume_reason,
                   "prior_B_access_disclosed": bool(state["attempts"])}
        state["attempts"].append(attempt)
        state["status"] = "incomplete"
        _save_state(stream, state)  # Record opening before even reading the B manifest.
        staging = Path(tempfile.mkdtemp(prefix=".evaluation-incomplete-", dir=RUN_ROOT))
        attempt["staging_directory"] = str(staging)
        _save_state(stream, state)
        try:
            observations = read_audit_observations(policy=policy, frozen_predictions=predictions,
                **load_audit_inputs(evaluation_manifest, policy=policy, calibration=False, check=check), check=check)
            from report_observed_map_audit import report_audit_predictions
            result = report_audit_predictions(predictions, observations, output_root=staging / "report",
                figure_root=staging / "figures", policy=policy, check=check)
            attempt["finished_utc"] = _now()
            attempt["status"] = "complete"
            completed_state = dict(state, status="complete", completed_utc=_now())
            _write_new(staging / "access_history.json",
                (json.dumps(completed_state, allow_nan=False, indent=2) + "\n").encode())
            names = ("report/observed_map_audit.complete.json", "report/3lines.md", "figures/3lines.md",
                     "figures/observed_map_audit.png", "figures/observed_map_audit.pdf", "access_history.json")
            _require(sum((staging / name).stat().st_size for name in names) <= durable_byte_cap,
                     "evaluation_durable_byte_limit_reached")
            check()
            _require(not EVALUATION_DIR.exists(), "evaluation_already_published")
            staging.rename(EVALUATION_DIR)
            state.update(completed_state)
            _save_state(stream, state)
            return {"status": result["status"], "evaluation_dir": str(EVALUATION_DIR)}
        except BaseException as error:
            attempt.update(status="failed", finished_utc=_now(), error_class=type(error).__name__)
            state["status"] = "failed"
            _save_state(stream, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "evaluate"))
    parser.add_argument("--development-manifest")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--engineering-resume-reason")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-kib", type=int, required=True)
    parser.add_argument("--durable-byte-cap", type=int, required=True)
    args = parser.parse_args()
    policy = actual_policy()  # Hard compute and actual-adoption gates precede any input.
    _require(args.durable_byte_cap > 0, "positive_durable_byte_limit_required")
    RUN_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    with resource_check(deadline_epoch=args.deadline_epoch, rss_limit_kib=args.rss_limit_kib) as check:
        if args.stage == "predict":
            _require(args.development_manifest and args.calibration_manifest
                     and args.evaluation_manifest is None and args.engineering_resume_reason is None,
                     "prediction_stage_accepts_only_development_and_A_calibration_manifests")
            result = predict(development_manifest=args.development_manifest,
                calibration_manifest=args.calibration_manifest, policy=policy,
                check=check, durable_byte_cap=args.durable_byte_cap)
        else:
            _require(args.evaluation_manifest and args.development_manifest is None
                     and args.calibration_manifest is None, "evaluation_stage_cannot_refit_or_read_audit_A")
            result = evaluate(evaluation_manifest=args.evaluation_manifest, policy=policy,
                check=check, durable_byte_cap=args.durable_byte_cap,
                engineering_resume_reason=args.engineering_resume_reason)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
