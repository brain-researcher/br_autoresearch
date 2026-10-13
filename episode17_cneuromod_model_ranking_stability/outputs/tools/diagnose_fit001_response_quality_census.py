#!/usr/bin/env python3
"""Complete fixed EP17 response-quality counts; no model or score is computed.

One explicitly selected fit001_diagnostic002 reopens the same released full001
development inputs. Every original column remains in all22 training,6 validation
and24 uncertainty partitions, for all8 groups and3 stages. The existing variance
qualification supplies categorical counts. Training-baseline SST uses exactly
the existing centered-training mean and residual-square sum formula; numerical
invalidity is counted rather than rescued. There are no predictions, SSEs,
R-squared values, comparisons, PCA fits, support changes or array exports.

Coordinator execution:1CPU/7GiB,<=15min,6GiB observed RSS guard and external
timeout. Completion is a quality census, never production qualification or
authorization to change a rule or refit. Original fit/diagnostic files stay intact.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_complete_development_prediction as fitting


EPISODE = Path(__file__).resolve().parents[2]
ATTEMPT = "fit001_diagnostic002"
OUTPUT = EPISODE / "outputs/runs/nested_prediction" / ATTEMPT


def _write(stream, report, cap):
    # Compact JSON keeps the complete partition census within its1MiB cap.
    data = (json.dumps(report, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fitting.require(len(data) <= cap)
    stream.seek(0)
    stream.write(data)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _source_frames(error):
    """Only code coordinates; never exception messages, locals or input paths."""
    frames, current = [], error.__traceback__
    while current is not None and len(frames) < 32:
        frames.append(dict(file=Path(current.tb_frame.f_code.co_filename).name,
                           function=current.tb_frame.f_code.co_name, line=current.tb_lineno))
        current = current.tb_next
    return frames[-12:]


def _variance_counts(values):
    import numpy as np
    import calibration_core as calibration
    result = calibration.qualify_variance(values)
    finite, moments, positive = result.finite_inputs, result.finite_moments, result.positive_variance
    nonpositive = finite & moments & ~positive
    exact_constant = finite & np.all(values == values[0], axis=0)
    counts = dict(valid_columns=int(result.valid.sum()),
        nonfinite_input_columns=int((~finite).sum()),
        nonfinite_moments_with_finite_inputs_columns=int((finite & ~moments).sum()),
        zero_or_underflow_variance_columns=int(nonpositive.sum()),
        positive_at_or_below_relative_threshold_columns=int(
            (finite & moments & positive & ~result.above_relative_threshold).sum()),
        exact_constant_zero_variance_columns=int((nonpositive & exact_constant).sum()),
        nonconstant_zero_or_underflow_variance_columns=int((nonpositive & ~exact_constant).sum()))
    fitting.require(sum(counts[key] for key in (
        "valid_columns", "nonfinite_input_columns", "nonfinite_moments_with_finite_inputs_columns",
        "zero_or_underflow_variance_columns", "positive_at_or_below_relative_threshold_columns")) == values.shape[1])
    return counts, ~result.valid


def _training_mean(values):
    import numpy as np
    import calibration_core as calibration
    # Same np.mean(float64) plus exact-constant branch as prediction_core.
    # Keep every column, including nonfinite results, for diagnostic counts.
    # No production path is changed to accept such a result.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        mean, _ = calibration.center(values)
    return mean


def _sst(values, baseline):
    import numpy as np
    # Exactly prediction_core.error_sums' denominator, without its whole-input
    # early refusal. No model prediction or candidate-error numerator exists.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        residual = values - baseline
        return np.sum(residual * residual, axis=0, dtype=np.float64)


def _sst_counts(sums):
    import numpy as np
    finite = np.isfinite(sums)
    counts = dict(positive_finite_columns=int((finite & (sums > 0)).sum()),
        nonpositive_finite_columns=int((finite & (sums <= 0)).sum()),
        nonfinite_columns=int((~finite).sum()))
    fitting.require(sum(counts.values()) == len(sums))
    return counts, ~(finite & (sums > 0))


def census(design, responses, *, check, progress):
    """Count complete fixed partitions with the original baseline per image."""
    import numpy as np
    import nested_prediction as nested
    fitting.require(callable(check) and callable(progress))
    training, validation, blocks = nested._row_indices(design)
    fitting.require((len(training), len(validation), len(blocks)) == (22, 6, 24))
    selection_sets = tuple(folds for folds in training if len(folds) in (5, 6))
    fitting.require(len(selection_sets) == 7 and set(responses) == set(nested.GROUPS))
    complete = []
    totals, variance_categories, sst_categories = Counter(), Counter(), Counter()
    for group in nested.GROUPS:
        panel = responses[group]
        fitting.require(panel.rows == design and set(panel.stages) == set(nested.STAGES))
        for stage in nested.STAGES:
            values = panel.stages[stage]
            fitting.require(values.ndim == 2 and values.shape == (2423, len(panel.column_ids)))
            width = values.shape[1]
            variances, denominators, means, scoring = [], [], {}, {}
            failed_variance = np.zeros(width, dtype=bool)
            failed_sst = np.zeros(width, dtype=bool)

            def event(operation, **metadata):
                check(dict(operation=operation, group=list(group), stage=stage, **metadata))

            partitions = tuple(("training", folds, rows) for folds, rows in training.items()) + tuple(
                ("validation", label, rows) for label, rows in validation.items()) + tuple(
                ("uncertainty_block", label, rows) for label, rows in blocks.items())
            for kind, label, rows in partitions:
                event("classify_variance", partition_kind=kind, partition_label=label,
                      shape=[len(rows), width])
                selected = values[rows]
                counts, invalid = _variance_counts(selected)
                failed_variance |= invalid
                entry = dict(kind=kind, label=label, image_count=len(rows), counts=counts)
                if kind == "training":
                    event("training_baseline_mean", training_folds=label, shape=[len(rows), width])
                    means[label] = _training_mean(selected)
                    entry["nonfinite_training_mean_columns"] = int((~np.isfinite(means[label])).sum())
                del selected
                variances.append(entry)
                variance_categories.update(counts)
                totals["variance_partitions"] += 1
                totals["variance_partition_column_failures"] += int(invalid.sum())
                totals[kind + "_variance_partitions_with_failure"] += int(invalid.any())

            def denominator(kind, label, sums, image_count, **metadata):
                nonlocal failed_sst
                counts, invalid = _sst_counts(sums)
                failed_sst |= invalid
                denominators.append(dict(kind=kind, label=label, image_count=image_count,
                                         counts=counts, **metadata))
                sst_categories.update(counts)
                totals["SST_partitions_including_explicit_pools"] += 1
                totals["SST_partition_column_failures_including_pools"] += int(invalid.sum())
                totals[kind + "_SST_partitions_with_failure"] += int(invalid.any())

            #30 unique inner partitions for outer selection, plus6 partitions
            # shared by final selection and the actual outer OOF baseline.
            for folds in training:
                for heldout, rows in validation.items():
                    if heldout in folds:
                        continue
                    kind = "inner_selection_partition" if len(folds) == 4 else "outer_and_final_selection_partition"
                    event("training_baseline_SST", partition_kind=kind, training_folds=folds,
                          validation_fold=heldout, shape=[len(rows), width])
                    sums = _sst(values[rows], means[folds])
                    scoring[folds, heldout] = sums
                    denominator(kind, heldout, sums, len(rows), training_folds=folds)
            fitting.require(len(scoring) == 36)

            #Same pooled denominators as _select/pooled_r2 for each of the six
            # outer-selection panels and final all-development selection.
            for selection in selection_sets:
                event("pooled_selection_SST", selection_folds=selection)
                pooled = np.zeros(width, dtype=np.float64)
                images = 0
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    for heldout in selection:
                        folds = tuple(label for label in selection if label != heldout)
                        pooled += scoring[folds, heldout]
                        images += len(validation[heldout])
                denominator("pooled_selection", selection, pooled, images)

            event("assemble_original_outer_baseline", shape=list(values.shape))
            baseline = np.empty_like(values)
            pooled_outer = np.zeros(width, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                for heldout, rows in validation.items():
                    folds = tuple(label for label in design.fold_labels if label != heldout)
                    baseline[rows] = means[folds]
                    pooled_outer += scoring[folds, heldout]
            denominator("pooled_outer_OOF", "all_development_images", pooled_outer, len(values))
            for label, rows in blocks.items():
                event("uncertainty_block_training_baseline_SST", block=label, shape=[len(rows), width])
                denominator("uncertainty_block", label, _sst(values[rows], baseline[rows]), len(rows))
            del baseline
            fitting.require(len(variances) == 52 and len(denominators) == 68)
            complete.append(dict(group=list(group), stage=stage, columns=width,
                variance_partitions=variances, training_baseline_SST_partitions=denominators,
                unique_columns_with_any_variance_failure=int(failed_variance.sum()),
                unique_columns_with_any_SST_failure=int(failed_sst.sum())))
            totals["group_stages"] += 1
            progress(len(complete))
    fitting.require(len(complete) == 24 and totals["variance_partitions"] == 1248
                    and totals["SST_partitions_including_explicit_pools"] == 1632)
    return dict(group_stages=complete, totals=dict(totals),
        variance_category_partition_column_occurrences=dict(variance_categories),
        SST_category_partition_column_occurrences_including_pools=dict(sst_categories),
        complete_fixed_response_census=True,
        variance_categories_are_mutually_exclusive_except_named_zero_subcounts=True,
        variance_relative_threshold=1e-12,
        SST_pool_notes="Six five-fold inner-selection pools and one six-fold final-selection pool; the separately labelled outer OOF pool has the same six training baselines as final selection.",
        failed_column_counts_are_not_column_removal=True, arrays_or_column_ids_exported=False,
        candidate_predictions_or_scores_computed=False, production_qualification_established=False,
        scientific_rule_amendment_adopted=False)


def _run(args):
    import private_stimulus_feature_pilot as storage
    from run_private_development_features import Budget
    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    fitting.require(not OUTPUT.exists() and not OUTPUT.is_symlink())
    storage._mkdir_private(OUTPUT, exclusive=True)
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="prediction_qualification", stage="complete_response_quality_census",
        attempt=ATTEMPT, original_attempt="fit001", feature_attempt="full001", status="started",
        started_utc=fitting.stamp(), job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
        model_fitting=False, PCA_fitting=False, private_inner_scoring=False, endpoint_scoring=False,
        raw_calibration_or_audit_response_access=False, response_magnitudes_recorded=False,
        original_attempt_modified=False, complete_fixed_response_census=False, group_stages_completed=0,
        caps=vars(args))
    phase, last_event = "scientific_imports", {}
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    fitting.require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def stop(_signal, _frame):
        raise fitting.FitRunRefusal()

    with storage._exclusive_file(OUTPUT / "diagnostic_report.json") as stream:
        _write(stream, report, args.report_byte_cap)

        def check(event=None):
            nonlocal last_event
            if event is not None:
                last_event = event if isinstance(event, dict) else fitting._event_description(event)
            budget(last_event.get("operation", "boundary"))

        def stage(name):
            nonlocal phase
            phase = name
            report["phase"] = name
            _write(stream, report, args.report_byte_cap)
            check()

        def progress(completed):
            report["group_stages_completed"] = completed
            _write(stream, report, args.report_byte_cap)
            check()

        try:
            for sig in saved:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            import complete_development_inputs as inputs
            stage("original_confirmed_refusal_binding")
            first = inputs.existing._json(fitting.REPORT_ROOT / "fit001_diagnostic001/diagnostic_report.json",
                                           args.metadata_byte_cap, check)
            fitting.require(first.get("attempt") == "fit001_diagnostic001"
                and first.get("status") == "completed_diagnosis"
                and first.get("response_variance_refusal_reproduced") is True
                and first.get("numerical_failure_code") == "response_partition_variance_invalid")
            stage("same_complete_development_input_loading")
            source = inputs.load_complete_development_inputs(feature_attempt="full001",
                feature_byte_cap=args.feature_byte_cap, metadata_byte_cap=args.metadata_byte_cap,
                response_byte_cap=args.response_byte_cap, resource_check=check)
            fitting.require(source.complete_development_response_release is True)
            report["input_loading_seconds"] = time.monotonic() - started
            report["loader_dimensions"] = dict(source.operational_dimensions)
            design, responses = source.design, source.responses
            del source
            stage("complete_fixed_partition_census")
            counted = census(design, responses, check=check, progress=progress)
            check()
            report.update(counted)
            report["status"] = "completed_quality_census"
        except BaseException as error:
            report.update(fitting._failure_category(error), exception_source_frames=_source_frames(error),
                          status="stopped", error_code=phase + "_failed")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=fitting.stamp(), elapsed_seconds=time.monotonic() - started,
                process_cpu_seconds=time.process_time() - cpu_started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                last_operation=budget.last_operation, last_resource_event=last_event)
            _write(stream, report, args.report_byte_cap)
    return 0 if report["status"] == "completed_quality_census" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    for flag in ("rss-limit-kib", "metadata-byte-cap", "feature-byte-cap", "response-byte-cap", "report-byte-cap"):
        parser.add_argument("--" + flag, type=int, required=True)
    args = parser.parse_args(argv)
    try:
        fitting.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname()
            and os.environ.get("SLURM_CPUS_PER_TASK", "").isdecimal()
            and int(os.environ["SLURM_CPUS_PER_TASK"]) >= 1)
        fitting.require(all(os.environ.get(name) == "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")))
        fitting.require(math.isfinite(args.deadline_epoch) and 0 < args.deadline_epoch - time.time() <= 900)
        fitting.require(all(type(getattr(args, name)) is int and getattr(args, name) > 0 for name in
            ("rss_limit_kib", "metadata_byte_cap", "feature_byte_cap", "response_byte_cap", "report_byte_cap")))
        fitting.require(args.rss_limit_kib <= 6 * 1024 * 1024 and args.report_byte_cap <= 1024 * 1024)
        code = _run(args)
    except BaseException as error:
        print(json.dumps(dict(status="stopped_before_report", **fitting._failure_category(error),
                              exception_source_frames=_source_frames(error))), flush=True)
        return 1
    print(json.dumps(dict(status="completed_quality_census" if code == 0 else "stopped", attempt=ATTEMPT)), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
