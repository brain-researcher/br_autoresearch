#!/usr/bin/env python3
"""One unmocked primary evaluator composition using the existing invented fixture.

No acquired artifact, real neural/image value or empirical audit role is read.
This is a bounded software dry run, not a recovery study, scientific result,
method freeze or audit opening. All 606 invented images, 120 concepts, twelve
blocks and eight native group axes are retained. The registered 10000 draws
and adjudication run without patched kernels or supplied interval substitutes.
Existing secondary/component tests are reused, not rerun by this entrypoint.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import time

EPISODE = Path(__file__).resolve().parents[2]
REPORT_ROOT = EPISODE / "outputs/runs/prediction_qualification/evaluator_dry_run"


class DryRunRefusal(ValueError):
    pass


def require(condition):
    if not condition:
        raise DryRunRefusal()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _write(stream, report, cap):
    content = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    require(len(content) <= cap)
    stream.seek(0)
    stream.write(content)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def compose(resource_check):
    """Use actual application, complete crossed bounds and actual adjudicator."""
    import numpy as np
    import audit_adjudicator as adjudicator
    import audit_prediction as application
    import test_audit_prediction as fixture_source
    import uncertainty_core as uncertainty

    resource_check("construct_existing_invented_fixture")
    values = fixture_source.fixture()
    design, development, predictors, features, responses, relation, order, expected, expected_cka = values
    require(uncertainty.DRAWS == 10000 and uncertainty.SEED == 1729
            and uncertainty.ORDER_STATISTIC_ONE_BASED == 9501)
    resource_check("apply_frozen_invented_predictors")
    applied = application.apply_frozen_audit(design, development, predictors, features, responses, relation,
        block_order=order, secondary_checks=(), resource_check=resource_check)
    require(len(applied.design.image_ids) == 606 and len(applied.blocks) == 12
            and len(applied.columns) == 8 and applied.statistic.family is relation.endpoint_family)
    for model in expected:
        np.testing.assert_allclose(applied.cka_features[model], expected_cka[model], rtol=2e-13, atol=2e-13)
        for group in applied.columns:
            for stage in application.endpoint.STAGES:
                np.testing.assert_allclose(applied.predictions[model][group][stage], expected[model][group][stage],
                                           rtol=2e-13, atol=2e-13)
    resource_check("registered_complete_crossed_bounds")
    bounds = applied.statistic.crossed_bounds(relation.inference_family)
    require(bounds.component_ids == relation.component_ids and bounds.phase == "audit"
            and bool(np.isfinite(bounds.standard_error).all()) and bool((bounds.standard_error > 0).all()))
    resource_check("adjudicate_invented_primary_relation")
    result = adjudicator.adjudicate_primary_audit(relation, applied.statistic, bounds)
    require(result.pair_id == relation.pair_id and result.required_components == relation.required_components)
    resource_check("outcome_free_composition_complete")
    # Operational scope only: invented effects/bounds and gate outcomes are
    # deliberately not serialized as episode findings.
    return dict(images=606, concepts=120, concept_blocks=12, participant_ROI_groups=8,
                models=2, beta_stages=3, components=len(bounds.component_ids),
                bootstrap_draws=uncertainty.DRAWS, bootstrap_seed=uncertainty.SEED,
                bootstrap_order_statistic=uncertainty.ORDER_STATISTIC_ONE_BASED,
                frozen_prediction_reference_matched=True, complete_bounds_computed=True,
                adjudication_completed=True, secondary_scores_computed=False)


def _run(args):
    from run_private_development_features import Budget
    import private_stimulus_feature_pilot as storage

    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    directory = REPORT_ROOT / args.attempt
    require(not directory.exists() and not directory.is_symlink())
    storage._mkdir_private(directory, exclusive=True)
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="prediction_qualification", stage="outcome_free_evaluator_dry_run",
        attempt=args.attempt, status="running", started_utc=stamp(),
        hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        deadline_epoch=args.deadline_epoch, rss_limit_kib=args.rss_limit_kib,
        study_payloads_opened=False, scientific_audit_openings=0, error_code=None, composition=None)
    saved = {}

    def stop(_signal, _frame):
        raise DryRunRefusal()

    with (directory / "dry_run_report.json").open("x+b") as stream:
        _write(stream, report, args.report_byte_cap)
        try:
            for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
                saved[sig] = signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                require(os.environ.get(name) == "1")
            report["composition"] = compose(budget)
            report["status"] = "completed"
        except BaseException:
            report["status"], report["error_code"] = "stopped", "outcome_free_composition_failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=stamp(), total_pipeline_wall_seconds=time.monotonic() - started,
                total_pipeline_process_cpu_seconds=time.process_time() - cpu_started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                last_operation=budget.last_operation)
            _write(stream, report, args.report_byte_cap)
    return 0 if report["status"] == "completed" and report["composition"] is not None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--rss-limit-kib", required=True, type=int)
    parser.add_argument("--report-byte-cap", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname()
                and re.fullmatch(r"dry[0-9]{3,6}", args.attempt) is not None
                and math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time()
                and args.rss_limit_kib > 0 and args.report_byte_cap > 0)
        code = _run(args)
    except BaseException:
        code = 1
    print(json.dumps({"status": "completed" if code == 0 else "stopped", "attempt": args.attempt}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
