#!/usr/bin/env python3
"""EP17 complete development endpoint panel from one finished private fit.

Only an explicitly released completed fit, the same development responses and
the existing family001 calibration operators may enter. No fitting, image
inference, support selection, raw calibration, secondary scoring or audit read.
The complete four-pair/five-edge result stays private until coordinator review.
This supplies endpoint evidence; it does not adjudicate fifteen valid search
trials, complete adaptive coverage, select a relation or authorize an audit.

Both registered within-pair families run unchanged: F_raw for raw-only edges
and F_raw_normalized for the ceiling edge and full-contract relation. Each
uses the existing 10,000-draw PCG64(1729) engine. The registered family001 has
no eligible H components: beta primary edges remain, while their explanatory
forecasts stay explicitly unavailable. No post-score family change is allowed.

The coordinator owns release/resource admission. Required caps bound the
existing readers and publication; an actual Slurm allocation plus an external
timeout must bound native peak memory and operations between callbacks.
"""
from datetime import datetime, timezone
import argparse
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
REPORT_ROOT = EPISODE / "outputs/runs/development_endpoints"
PRIVATE_ROOT = EPISODE / "outputs/cache/private/development_endpoints"
FAMILY_ROOT = EPISODE / "outputs/runs/prediction_qualification/endpoint_family"
# Fixed metadata labels/order in concept_role_assignment_specification.md:226
# and run_concept_role_assignment.py:382. Never inferred from image-row order.
BLOCK_ORDER = tuple(f"block{index:02d}" for index in range(1, 25))
EDGE_IDS = ("B_to_C", "C_to_D", "A_all_to_A_N", "A_N_to_R_N", "raw_to_ceiling_normalized")
FAMILIES = ("F_raw", "F_raw_normalized")


class EndpointRunRefusal(ValueError):
    """Categorical operational failure; private values never enter exceptions."""


def require(condition):
    if not condition:
        raise EndpointRunRefusal()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value))
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    require(value is None or isinstance(value, (str, bool, int, float)))
    return value


def _write(stream, value, cap):
    payload = (json.dumps(_jsonable(value), sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    require(len(payload) <= cap)
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())
    return len(payload)


def _family_description(family):
    import endpoint_evaluator as endpoint
    reasons = family.calibration_inapplicability
    return dict(component_ids=list(family.component_ids), primary_endpoint_ids=list(family.primary_endpoint_ids),
        h_component_ids=list(family.h_component_ids), directed_edges=[list(edge) for edge in family.directed_edges],
        score_scales=list(family.score_scales),
        group_widths={f"{person}_{roi}": len(family._supports[person, roi].columns) for person, roi in endpoint.GROUPS},
        calibration_inapplicability=dict(ceiling_empty_groups=[list(group) for group in reasons["ceiling_empty_groups"]],
            gain_reasons=[dict(group=list(group), component=component, reason=reason)
                          for (group, component), reason in reasons["gain_reasons"].items()]))


def _bind_family_record(family, record, operator_attempt):
    import endpoint_evaluator as endpoint
    require(record.get("status") == "qualified_calibration_endpoint_family"
        and record.get("analysis_id") == "prediction_qualification"
        and all(record.get(key) is False for key in
                ("candidate_scores_read", "raw_calibration_responses_read", "audit_values_read"))
        and record.get("source_private_operator_attempt") == str(operator_attempt))
    description = _family_description(family)
    require(all(record.get(key) == value for key, value in description.items()))
    require(family.primary_endpoint_ids == endpoint.PRIMARY and family.h_component_ids == ()
            and family.directed_edges == endpoint.EDGES)
    return description


def evaluate_endpoint_panel(result, responses, family, *, block_order, resource_check, progress):
    """Reuse actual endpoint/forecast callers; never replace interval arithmetic."""
    import numpy as np
    import development_endpoints as development
    import development_forecasts as forecast
    import endpoint_evaluator as endpoint
    import uncertainty_core as uncertainty
    require(callable(resource_check) and callable(progress))
    require(family.primary_endpoint_ids == endpoint.PRIMARY and family.h_component_ids == ()
            and family.directed_edges == endpoint.EDGES)
    resource_check()
    composed = development.compose_development_endpoints(result, responses, family,
        block_order=block_order, resource_check=resource_check)
    entries, completed_families = {}, 0
    for pair, model_order in endpoint.PAIR_SIGNS.items():
        statistic = composed.statistics[pair]
        intervals = {}
        for name in FAMILIES:
            resource_check()
            progress("crossed_intervals", len(entries), completed_families)
            bounds = statistic.crossed_bounds(name)
            edges = tuple(edge for edge in family.directed_edges
                          if all(component in bounds.primary_endpoint_ids for component in edge))
            intervals[name] = dict(component_ids=bounds.component_ids, score_scales=bounds.score_scales,
                primary_endpoint_ids=bounds.primary_endpoint_ids, estimate=bounds.estimate,
                standard_error=bounds.standard_error, critical_value=bounds.critical_value,
                lower=bounds.lower, upper=bounds.upper, margins=bounds.margins,
                relation_decision=uncertainty.relation_decision(bounds, edges),
                classification_scope=("raw_only" if name == "F_raw" else "complete_eligible_primary_family"))
            completed_families += 1
            resource_check()
            progress("crossed_intervals", len(entries), completed_families)
        forecasts, edges = {}, {}
        for index, edge_id in enumerate(EDGE_IDS):
            resource_check()
            value = forecast.build_development_forecast(statistic, edge_id,
                reference_block_order=block_order, resource_check=resource_check)
            forecasts[edge_id] = dict(eligible=value.eligible, eligibility_reason=value.eligibility_reason,
                target=value.target, value=value.value, absolute_error_limit=value.absolute_error_limit,
                participant_forecasts=value.participant_forecasts,
                crossfit_predicted=value.crossfit_predicted, crossfit_observed=value.crossfit_observed,
                crossfit_absolute_errors=value.crossfit_absolute_errors)
            family_name = "F_raw_normalized" if edge_id == "raw_to_ceiling_normalized" else "F_raw"
            bounds = intervals[family_name]
            before, after = family.directed_edges[index]
            first, second = (bounds["component_ids"].index(item) for item in (before, after))
            edges[edge_id] = dict(before=before, after=after, interval_family=family_name,
                signed_endpoint_change=float(bounds["estimate"][second] - bounds["estimate"][first]),
                opposite_margin_crossing=bool(
                    (bounds["lower"][first] > bounds["margins"][first]
                     and bounds["upper"][second] < -bounds["margins"][second])
                    or (bounds["upper"][first] < -bounds["margins"][first]
                        and bounds["lower"][second] > bounds["margins"][second])),
                role="required_size_control" if edge_id == "A_all_to_A_N" else "registered_measurement_edge")
        entries[pair] = dict(model_order=model_order, promotion_eligible=pair != "F1",
            participant_ids=endpoint.SUBJECTS, participant_effects=statistic.participant_effects(np.ones(24, dtype=int)),
            interval_families=intervals, edges=edges, forecasts=forecasts)
        progress("pair_complete", len(entries), completed_families)
    resource_check()
    return dict(stage="complete_development_endpoint_panel", phase="development", complete_endpoint_panel=True,
        initial_coverage_complete=False, valid_trial_adjudication="pending_coordinator_review",
        candidate_selection=False, adaptive_search_complete=False, audit_access=False,
        secondary_correlation_or_CKA_computed=False,
        scope=dict(images=2423, concepts=480, concept_blocks=24, participants=4, ROIs=2,
                   beta_stages=3, fixed_pairs=4, promotion_pairs=3, edges_per_pair=5),
        frozen_block_order=tuple(block_order), endpoint_family=_family_description(family),
        bootstrap=dict(draws=uncertainty.DRAWS, seed=uncertainty.SEED,
            order_statistic_one_based=uncertainty.ORDER_STATISTIC_ONE_BASED,
            family_scope="within_registered_pair_not_experiment_wide_across_search",
            numerical_families=FAMILIES), pairs=entries,
        exposure_limits=dict(target_image_training_exposure="unknown_not_clean",
            EP17_EP18="correlated_design_exposed_not_independent_replication",
            causal_training_recipe_claim=False, new_participant_population_claim=False),
        separation="procedural_same_account_not_technically_enforced_or_blinded")


def _validate(args):
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname())
    require(re.fullmatch(r"panel[0-9]{3,6}", args.attempt) is not None
        and re.fullmatch(r"fit[0-9]{3,6}", args.fit_attempt) is not None
        and re.fullmatch(r"full[0-9]{3,6}", args.feature_attempt) is not None
        and args.family_attempt == "family001")
    require(all(os.environ.get(name) == "1" for name in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
         "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")))
    require(math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time())
    require(all(type(getattr(args, name)) is int and getattr(args, name) > 0 for name in
        ("rss_limit_kib", "metadata_byte_cap", "feature_byte_cap", "response_byte_cap",
         "operator_array_byte_cap", "checkpoint_byte_cap", "science_byte_cap", "report_byte_cap")))


def _run(args):
    import private_stimulus_feature_pilot as storage
    from run_private_development_features import Budget
    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    public, private = REPORT_ROOT / args.attempt, PRIVATE_ROOT / args.attempt
    require(all(not path.exists() and not path.is_symlink() for path in (public, private)))
    storage._mkdir_private(public, exclusive=True)
    storage._mkdir_private(private, exclusive=True)
    destination = private / "endpoint_panel.json"
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="initial_coverage", stage="development_endpoint_panel", attempt=args.attempt,
        fit_attempt=args.fit_attempt, feature_attempt=args.feature_attempt, family_attempt=args.family_attempt,
        status="started", started_utc=stamp(), job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
        private_summary_complete=False, scientific_output_released=False, audit_access=False,
        initial_coverage_complete=False, pairs_completed=0, bootstrap_families_completed=0,
        private_summary_path=str(destination), caps=vars(args))
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def stop(_signal, _frame):
        raise EndpointRunRefusal()

    with storage._exclusive_file(public / "endpoint_report.json") as stream:
        _write(stream, report, args.report_byte_cap)

        def check(event=None):
            budget(event)

        def stage(name):
            report["phase"] = name
            _write(stream, report, args.report_byte_cap)
            check()

        def progress(phase, pairs, families):
            report.update(phase=phase, pairs_completed=pairs, bootstrap_families_completed=families)
            _write(stream, report, args.report_byte_cap)
            check()

        try:
            for sig in saved:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            stage("scientific_imports")
            import complete_development_inputs as inputs
            import predictor_checkpoint as checkpoint
            import released_endpoint_family as family_loader
            import run_complete_development_prediction as fitter
            stage("completed_fit_and_frozen_family_binding")
            fit = inputs.existing._json(fitter.REPORT_ROOT / args.fit_attempt / "fit_report.json",
                                        args.metadata_byte_cap, check)
            checkpoint_path = fitter.PRIVATE_ROOT / args.fit_attempt / "complete_prediction.zip"
            require(fit.get("status") == "completed" and fit.get("checkpoint_published") is True
                and fit.get("stage") == "complete_nested_prediction" and fit.get("attempt") == args.fit_attempt
                and fit.get("feature_attempt") == args.feature_attempt
                and fit.get("checkpoint_path") == str(checkpoint_path)
                and fit.get("audit_access") is False)
            record = inputs.existing._json(FAMILY_ROOT / args.family_attempt / "family_report.json",
                                            args.metadata_byte_cap, check)
            family = family_loader.load_released_endpoint_family(metadata_byte_cap=args.metadata_byte_cap,
                array_byte_cap=args.operator_array_byte_cap, resource_check=check)
            _bind_family_record(family, record, family_loader.operators.PRIVATE)
            stage("complete_checkpoint_loading")
            result = checkpoint.load_prediction_checkpoint(checkpoint_path,
                max_bytes=args.checkpoint_byte_cap, resource_check=check)
            stage("same_complete_development_responses")
            source = inputs.load_complete_development_inputs(feature_attempt=args.feature_attempt,
                feature_byte_cap=args.feature_byte_cap, metadata_byte_cap=args.metadata_byte_cap,
                response_byte_cap=args.response_byte_cap, resource_check=check)
            require(source.complete_development_response_release is True and source.design == result.design)
            responses = source.responses
            report["loader_dimensions"] = dict(source.operational_dimensions)
            del source  # Existing feature bank is unused here; release it before endpoint work.
            stage("complete_endpoint_composition")
            science = evaluate_endpoint_panel(result, responses, family, block_order=BLOCK_ORDER,
                                               resource_check=check, progress=progress)
            del result, responses
            stage("complete_private_summary_publication")
            partial = private / "endpoint_panel.json.partial"
            with storage._exclusive_file(partial) as target:
                report["private_summary_bytes"] = _write(target, science, args.science_byte_cap)
            check()
            os.link(partial, destination)
            report["private_summary_complete"] = True
            try:
                partial.unlink()
                report["partial_cleanup_complete"] = True
            except OSError:
                report["partial_cleanup_complete"] = False
            report["status"] = "completed_private_endpoint_panel"
        except BaseException:
            report["status"], report["error_code"] = "stopped", report.get("phase", "startup") + "_failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=stamp(), elapsed_seconds=time.monotonic() - started,
                process_cpu_seconds=time.process_time() - cpu_started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                last_operation=budget.last_operation, private_summary_path_present=destination.exists())
            _write(stream, report, args.report_byte_cap)
    return 0 if report["status"] == "completed_private_endpoint_panel" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attempt", "fit-attempt", "feature-attempt", "family-attempt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    for name in ("rss-limit-kib", "metadata-byte-cap", "feature-byte-cap", "response-byte-cap",
                 "operator-array-byte-cap", "checkpoint-byte-cap", "science-byte-cap", "report-byte-cap"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args(argv)
    try:
        _validate(args)
        code = _run(args)
    except BaseException:
        code = 1
    print(json.dumps({"status": "completed" if code == 0 else "stopped",
                      "stage": "development_endpoint_panel"}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
