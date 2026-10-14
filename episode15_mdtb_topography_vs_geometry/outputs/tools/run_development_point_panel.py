#!/usr/bin/env python3
"""Run EP15's adopted conditional observed-beta panel from a released view.

The coordinator owns release, the published-parameter/cache firewall, scientific
readiness and actual Slurm/resource admission. This entrypoint grants none of
them. The explicitly selected observed-beta route retains unresolved error
assumptions and describes observed-map loss only; its legacy signed statistics
remain assumption-conditional. This does not qualify the original reliable-
variation estimands. No smaller cohort/model cost experiment is added.

The manifest has exactly four fields, mirroring the existing reader arguments:
  release_bindings: twelve ordered {participant_id, archive_path,
                    anatomy_headers, task_a_header} objects;
  member_rows_path: development-only TSV with original MEMBER_COLUMNS;
  condition_multiplicity_by_session: the existing source-summary object;
  common_support_path: the unchanged four-array, 208-location support NPZ.
All paths are explicit absolute paths. Root publishes this view; this worker
never loads or filters mixed-role metadata and never discovers source paths.
Existing source metadata for trusted view population is anatomical_headers.json
under source_qualification/attempt001; source_metadata.json under
metadata_join001 (selected_Task_A_header.header and condition multiplicities);
condition_partition_plan001/role_member_plan.tsv; and
anatomical_support001/common_support.npz. The original files are NOT defaults.

The existing reader performs complete plan/header checks before voxel decoding.
The unchanged panel retains all twelve targets, sixteen variants, baselines,
matched controls and genuine-A falsifiers: 24 source fits and 48 calibration
panels, with no participant-deletion expansion. Complete observed-loss and
signed summaries are saved only after all fits/controls finish. Cost mode saves
the full result privately as point_summaries.withheld.json and emits operational
measurements only. A coordinator can later publish that same completed file
without refitting. Withholding is procedural, not OS isolation from the owner.
No ratios, winner, calibrated uncertainty, formal class decision or audit access.

Run on compute with one numerical thread, an explicit absolute deadline/RSS
guard, a fresh output directory and a fixed durable-byte allowance. No scratch
files or images are extracted. Slurm and an enclosing TERM/KILL timeout must
bound native operations and imports; Python guards do not bound native peak
memory. Costs are recorded for the actual complete route, not treated as prior
full-panel cost evidence. Prior passing reader/panel checks are reused.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time
import zipfile


OUTPUTS = Path(__file__).resolve().parents[1]
MANIFEST_BYTES = 8 * 1024**2
MEMBER_BYTES = 32 * 1024**2
SUPPORT_BYTES = 64 * 1024**2
REPORT_BYTES = 64 * 1024
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class ExecutionRefusal(ValueError):
    """Categorical operational failure; never include response values."""


def _require(condition):
    if not condition:
        raise ExecutionRefusal()


def _compute():
    _require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname())
    _require(all(os.environ.get(name) == "1" for name in THREAD_ENV))


def _path(value):
    path = Path(value)
    _require(path.is_absolute() and ".." not in path.parts)
    return path


def _bounded(path, limit):
    with _path(path).open("rb") as source:
        value = source.read(limit + 1)
    _require(len(value) <= limit)
    return value


def load_released_inputs(manifest_path, *, check):
    """Load only supplied role-filtered metadata and shared support, on compute.

    The manifest is a caller-owned release view, not proof of access authority.
    No container is opened here; the existing reader owns its strict checks.
    """
    _compute()
    import numpy as np
    from build_condition_partition_plan import MEMBER_COLUMNS
    from development_handoff_reader import DevelopmentReleaseBinding, DEVELOPMENT_IDS

    check()
    manifest = json.loads(_bounded(manifest_path, MANIFEST_BYTES))
    _require(isinstance(manifest, dict) and set(manifest) == {
        "release_bindings", "member_rows_path", "condition_multiplicity_by_session", "common_support_path"})
    records = manifest["release_bindings"]
    _require(isinstance(records, list) and len(records) == len(DEVELOPMENT_IDS))
    fields = {"participant_id", "archive_path", "anatomy_headers", "task_a_header"}
    _require(all(isinstance(row, dict) and set(row) == fields for row in records))
    _require(tuple(row["participant_id"] for row in records) == DEVELOPMENT_IDS)
    bindings = tuple(DevelopmentReleaseBinding(row["participant_id"], _path(row["archive_path"]),
        row["anatomy_headers"], row["task_a_header"]) for row in records)
    table = csv.DictReader(io.StringIO(_bounded(manifest["member_rows_path"], MEMBER_BYTES).decode("utf-8-sig")),
                           delimiter="\t")
    _require(table.fieldnames == MEMBER_COLUMNS)
    rows = tuple(table)
    _require(all(None not in row and all(value is not None for value in row.values())
        and row["participant_id"] in DEVELOPMENT_IDS and row["participant_role"] == "development"
        and row["planned_handoff"] == "development" for row in rows))
    check()
    support_payload = _bounded(manifest["common_support_path"], SUPPORT_BYTES)
    with zipfile.ZipFile(io.BytesIO(support_payload)) as archive:
        entries = archive.infolist()
        _require(len(entries) == 4 and sum(item.file_size for item in entries) <= SUPPORT_BYTES)
    with np.load(io.BytesIO(support_payload), allow_pickle=False) as source:
        support = {name: np.array(source[name], copy=True) for name in source.files}
    check()
    return dict(release_bindings=bindings, member_rows=rows,
        condition_multiplicity_by_session=manifest["condition_multiplicity_by_session"],
        common_support=support)


def run_released_development(manifest_path, *, check, stage, derived_support_path=None):
    """Return the existing full panel; callbacks carry operational stage only."""
    _compute()
    from development_handoff_reader import read_development_handoffs
    from proposed_development_point_panel import fit_development_point_panel

    stage("released_metadata")
    inputs = load_released_inputs(manifest_path, check=check)
    selection = None
    if derived_support_path is not None:
        from finite_development_support import load_selection
        selection = load_selection(derived_support_path)
    stage("development_mapping")
    development = read_development_handoffs(**inputs, check=check, derived_support=selection)
    _require(development.qualified_forward_header is not None)
    stage("complete_point_panel")
    panel = fit_development_point_panel(development, common_support=inputs["common_support"],
        forward_header=development.qualified_forward_header, check=check)
    check()
    return panel, dict(development.report)


def _jsonable(value):
    import numpy as np
    if isinstance(value, Mapping):
        _require(all(isinstance(key, str) for key in value))
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ExecutionRefusal()


def _encoded(value, limit):
    payload = (json.dumps(_jsonable(value), allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    _require(len(payload) <= limit)
    return payload


def _write_new(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())


def _exception_diagnostic(error):
    """Retain bounded traceback locations; never messages, source or locals."""
    from collections import deque
    chain, seen = [], set()
    while error is not None and len(chain) < 4 and id(error) not in seen:
        seen.add(id(error))
        frames, count = deque(maxlen=8), 0
        trace = error.__traceback__
        while trace is not None:
            code = trace.tb_frame.f_code
            frames.append(dict(file=Path(code.co_filename).name[:128],
                               function=code.co_name[:96], line=trace.tb_lineno))
            count += 1
            trace = trace.tb_next
        chain.append(dict(exception_class=(type(error).__module__ + "." + type(error).__name__)[:160],
                          frames=list(frames), omitted_outer_frames=max(0, count - len(frames))))
        error = error.__cause__ if error.__cause__ is not None else (
            None if error.__suppress_context__ else error.__context__)
    return dict(exception_chain=chain, additional_chain_omitted=error is not None,
                messages_source_and_locals_recorded=False)


def execute(*, manifest_path, output_dir, deadline_epoch, rss_limit_kib, durable_byte_cap,
            analysis_scope, publication_mode, derived_support_path=None):
    """Fresh-attempt CLI composition. An existing directory is never modified."""
    _compute()
    _require(analysis_scope == "conditional_observed_beta"
        and publication_mode in ("withheld_cost_measurement", "complete_descriptive_report"))
    _require(math.isfinite(deadline_epoch) and deadline_epoch > time.time()
        and type(rss_limit_kib) is int and rss_limit_kib > 0
        and type(durable_byte_cap) is int and durable_byte_cap > REPORT_BYTES)
    manifest_path, output_dir = _path(manifest_path), _path(output_dir)
    if derived_support_path is not None:
        derived_support_path = _path(derived_support_path)
        _require(derived_support_path == OUTPUTS / "runs/observed_beta_development/finite_support001/selection.npz")
    _require(OUTPUTS / "runs" in output_dir.resolve().parents)
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    output_dir.chmod(0o700)
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(status="started", phase="startup", job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path), point_summaries_published=False,
        derived_support_path=None if derived_support_path is None else str(derived_support_path),
        target_availability_used_for_common_support=derived_support_path is not None,
        analysis_scope=analysis_scope, publication_mode=publication_mode,
        route_id="observed_beta_development_descriptive",
        scientific_summary_withheld=publication_mode == "withheld_cost_measurement",
        private_summary_complete=False,
        stage_elapsed_seconds={}, scratch_bytes=0, complete_panel_required=True,
        audit_values_read=False, formal_class_decision=False, uncertainty_calibrated=False,
        separation="procedural_same_account_not_enforced_or_blinded",
        caps=dict(deadline_epoch=deadline_epoch, rss_limit_kib=rss_limit_kib,
                  durable_byte_cap=durable_byte_cap))
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    phase_start = started

    def check():
        _require(time.time() < deadline_epoch
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= rss_limit_kib)

    def stage(name):
        nonlocal phase_start
        now = time.monotonic()
        report["stage_elapsed_seconds"][report["phase"]] = now - phase_start
        report["phase"], phase_start = name, now
        check()

    def stop(_signal, _frame):
        raise ExecutionRefusal()

    code = 1
    try:
        _write_new(output_dir / "settings.json", _encoded(report, REPORT_BYTES // 2))
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_epoch - time.time()))
        panel, mapping = run_released_development(manifest_path, check=check, stage=stage,
            **({} if derived_support_path is None else {"derived_support_path": derived_support_path}))
        stage("complete_observed_beta_summary")
        from observed_beta_summary import observed_beta_summary
        _require(panel.ordinary_fit.deletions == ())
        observed = observed_beta_summary(panel.ordinary_fit.full, panel.calibration_falsifier_fit, check=check,
            **({} if derived_support_path is None else {"support_size": mapping["support_locations"]}))
        stage("complete_point_persistence")
        payload = _encoded(dict(analysis_scope=analysis_scope, observed_beta=observed,
            numerical_methods=dict(
                m4_default="sufficient_full_domain_curvature_and_first_order_gap",
                local_optimizer_nonconvergence="same_qualified_global_solver_for_the_canonical_family",
                local_optimizer_nonfinite_trial="same_qualified_global_solver_for_the_canonical_family",
                canonical_diagonal_on_curvature_refusal="global_profile_exclusion_or_balanced_scalar_dual_vertex_or_constrained_vertex_with_global_exclusion",
                canonical_block_families_on_curvature_refusal="independent_block_radial_global_exclusion",
                original_model_domain_rank_gap_and_operator_limits_retained=True,
                unresolved_certificate_stops_complete_panel=True,
                exact_global_optimizer_or_interval_arithmetic_proof=False),
            spatial_estimand=dict(locations=mapping["support_locations"],
                selection_path=None if derived_support_path is None else str(derived_support_path),
                target_Task_B_availability_used=derived_support_path is not None,
                availability_conditioned=derived_support_path is not None,
                comparison_with_original_208_location_loss_permitted=False),
            assumption_conditional_cross_partition_statistics=dict(
                interpretation="cross_partition_statistics_conditional_on_unresolved_error_assumptions",
                reliable_variation_established=False,
                ordinary=panel.ordinary_points,
                calibration_falsifiers=panel.calibration_falsifier_points,
                m2_alignment=panel.m2_alignment_points),
            claim_limit="descriptive_observed_beta_prediction_in_twelve_development_participants_only",
            recovery_ratios_or_class_decisions=False), durable_byte_cap - REPORT_BYTES)
        check()
        name = ("point_summaries.withheld.json" if publication_mode == "withheld_cost_measurement"
                else "point_summaries.json")
        partial, final = output_dir / (name + ".partial"), output_dir / name
        _write_new(partial, payload)
        check()
        os.link(partial, final)
        report["private_summary_complete"] = True
        report["point_summaries_published"] = publication_mode == "complete_descriptive_report"
        report["complete_summary_filename"] = name
        try:
            partial.unlink()
            report["partial_cleanup_complete"] = True
        except OSError:
            report["partial_cleanup_complete"] = False
        report.update(status=("completed_observed_beta_cost_panel" if report["scientific_summary_withheld"]
                              else "completed_observed_beta_descriptive_panel"), mapping=mapping,
            point_summary_bytes=len(payload), complete_panel=dict(participants=12,
                Task_B_only_conditions=18, partitions=4, locations=mapping["support_locations"], variants=16,
                source_fits=24, calibration_panels=48, random_basis_draws_per_target=8,
                matched_warp_draws_per_variant_per_target=8, matched_warp_variants=4,
                condition_control_targets=12, identity_control_targets=12))
        code = 0
    except BaseException as error:
        report["status"] = "stopped"
        report["error_code"] = report["phase"] + "_failed"
        report["exception_diagnostic"] = _exception_diagnostic(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report["stage_elapsed_seconds"][report["phase"]] = time.monotonic() - phase_start
        report.update(elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu_started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        _write_new(output_dir / "operational.json", _encoded(report, REPORT_BYTES // 2))
    return code if report["status"] in ("completed_observed_beta_cost_panel",
        "completed_observed_beta_descriptive_panel") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--derived-support", type=Path)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--durable-byte-cap", required=True, type=int)
    parser.add_argument("--analysis-scope", required=True, choices=("conditional_observed_beta",))
    parser.add_argument("--publication-mode", required=True,
                        choices=("withheld_cost_measurement", "complete_descriptive_report"))
    args = parser.parse_args(argv)
    try:
        code = execute(manifest_path=args.manifest, output_dir=args.output_dir,
            deadline_epoch=args.deadline_epoch, rss_limit_kib=args.rss_limit_kib,
            durable_byte_cap=args.durable_byte_cap, analysis_scope=args.analysis_scope,
            publication_mode=args.publication_mode, derived_support_path=args.derived_support)
    except BaseException:
        code = 1
    print(json.dumps({"status": "completed" if code == 0 else "stopped",
                      "analysis_id": "development_point_panel"}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
