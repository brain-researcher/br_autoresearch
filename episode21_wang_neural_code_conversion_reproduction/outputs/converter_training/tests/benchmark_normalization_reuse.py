#!/usr/bin/env python3
"""Isolated array-only normalization prototype; run in coordinated Slurm.

No empirical readers, imported-source edits, scheduler calls or output files.
The prototype changes only float32 block normalization placement. It retains
all scalar draw arithmetic after row selection and all validation/cleanup.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import statistics
import sys
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True


class Budget:
    def __init__(self, seconds, rss_kib):
        self.deadline = time.monotonic() + seconds
        self.rss_kib = rss_kib

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("benchmark_90second_budget")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.rss_kib:
            raise MemoryError("benchmark_448MiB_RSS_budget")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def execute(budget, report):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import numpy as np
    import finding_a_batched_inference as batched
    import finding_a_metrics as metrics
    from test_finding_a_metrics import TEST_IDS, TRAIN_IDS, fixture, blocks
    from test_finding_a_batched_inference import equal, readonly
    budget.check()
    original_state = batched._LayerState
    checker = unittest.TestCase()

    class Cache:
        """One actual retained block object; no integer-id reuse or old bank."""
        def __init__(self):
            self.block = self.x = self.y = None
            self.preparations = self.maximum_bytes = 0

        def clear(self):
            self.block = self.x = self.y = None

        def get(self, block):
            if self.block is not block:
                self.clear()
                # Caller has already run the original _block validation.
                x = (block.predicted - block.training_mean) / block.training_std
                y = (block.truth - block.training_mean) / block.training_std
                require(x.dtype == y.dtype == np.dtype("float32"), "float32_normalization_boundary")
                x.setflags(write=False)
                y.setflags(write=False)
                self.block, self.x, self.y = block, x, y
                self.preparations += 1
                self.maximum_bytes = max(self.maximum_bytes, x.nbytes + y.nbytes)
            return self.x, self.y

    def prototype_class(cache):
        class ReusedNormalizationState(original_state):
            def consume(self, block, owner, image_ids, units, check):
                check()
                ids = metrics._block(block, owner, image_ids, units, self.seen)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    predicted, truth = block.predicted[self.draw], block.truth[self.draw]
                    raw_x, raw_y = predicted.astype(np.float64), truth.astype(np.float64)
                    _, _, px, py, pxy = metrics._centered(raw_x, raw_y, axis=0)
                    self.profile[ids] = metrics._correlations(px, py, pxy)
                    check()
                    normalized_x, normalized_y = cache.get(block)
                    x = normalized_x[self.draw].astype(np.float64)
                    y = normalized_y[self.draw].astype(np.float64)
                    bx, by, bxx, byy, bxy = metrics._centered(x, y, axis=1)
                    total = self.count + len(ids)
                    dx, dy = bx - self.mean_x, by - self.mean_y
                    factor = self.count * len(ids) / total
                    self.xx += bxx + dx * dx * factor
                    self.yy += byy + dy * dy * factor
                    self.xy += bxy + dx * dy * factor
                    self.mean_x += dx * (len(ids) / total)
                    self.mean_y += dy * (len(ids) / total)
                self.count = total
                self.training_mean[ids], self.training_std[ids] = block.training_mean, block.training_std
                self.seen[ids] = True
                check()
        return ReusedNormalizationState

    arguments = dict(method="content_loss", source="sub-03", target="sub-01", layer="fc8",
                     expected_image_ids=TEST_IDS, decoder_training_image_ids=TRAIN_IDS)

    def run(data, draws, prototype, callback=None):
        cache = Cache()
        check = budget.check if callback is None else callback
        manager = mock.patch.object(batched, "_LayerState", prototype_class(cache)) if prototype else nullcontext()
        try:
            with manager:
                scores = batched.score_layer_batch(data, image_indices=draws, check=check, **arguments)
            metadata = {"normalization_preparations": cache.preparations,
                        "maximum_additional_cache_bytes": cache.maximum_bytes}
            return scores, metadata
        finally:
            cache.clear()
            require(cache.block is cache.x is cache.y is None, "cache_released_at_call_exit")
            require(batched._LayerState is original_state, "production_class_restored")

    def frozen_blocks(*data, **kwargs):
        supplied = blocks(*data, **kwargs)
        for block in supplied:
            for name in ("full_layer_C_index", "predicted", "truth", "training_mean", "training_std"):
                getattr(block, name).setflags(write=False)
        return supplied

    schedule = np.random.RandomState(42).randint(0, 224, size=(16, 224))
    schedule[0] = np.arange(224)
    schedule[1] = np.arange(223, -1, -1)
    schedule[2] = 0
    schedule[3] = np.tile(np.array([1, 2]), 112)
    draws = readonly(schedule, np.int64)
    predicted, truth, mean, std = fixture()
    altered = tuple(value.copy() for value in (predicted, truth, mean, std))
    altered[1][:, 0] = altered[2][0]
    altered[1][0] = altered[2]
    altered[0][3, 17] = np.nan
    altered[0][4, 19] = np.finfo(np.float32).max
    altered[3][19] = np.float32(.125)
    reverse = np.roll(np.arange(1000, dtype=np.int64)[::-1], 147)
    report["checks"] = []
    for label, data, order, reversed_blocks in (
            ("finite", (predicted, truth, mean, std), None, False),
            ("constants_nan_float32_overflow_reversed_C", altered, reverse, True)):
        for width in (64, 113, 256):
            supplied = frozen_blocks(*data, width=width, order=order)
            if reversed_blocks:
                supplied.reverse()
            for count in ((16, 1, 8) if width == 64 else (16,)):
                indices = readonly(draws[:count], np.int64)
                baseline, _ = run(iter(supplied), indices, False)
                candidate, cache_stats = run(iter(supplied), indices, True)
                require(len(candidate) == len(baseline) == count, "complete_draw_count")
                for actual, expected in zip(candidate, baseline):
                    equal(checker, actual, expected)
                    for name in ("image_indices", "training_mean", "training_std"):
                        require(not getattr(actual, name).flags.writeable, "readonly_score_arrays")
                    for name in ("pattern", "profile"):
                        summary = getattr(actual, name)
                        require(not summary.correlations.flags.writeable and not summary.defined_mask.flags.writeable,
                                "readonly_metric_support")
                require(cache_stats["normalization_preparations"] == len(supplied), "one_normalization_per_block")
                require(cache_stats["maximum_additional_cache_bytes"] <= 2 * 224 * width * 4, "one_block_cache_bound")
                report["checks"].append({"case": label, "width": width, "draws": count, "all_fields_exact": True})
                del baseline, candidate
                budget.check()

    # Retain actual coverage validation and source cleanup under the prototype.
    supplied = frozen_blocks(predicted, truth, mean, std, width=64)
    for label, data in (("missing_block", supplied[:-1]), ("duplicate_block", supplied + supplied[:1])):
        for prototype in (False, True):
            closed = []
            def source():
                try:
                    yield from data
                finally:
                    closed.append(True)
            try:
                run(source(), readonly(draws[:1], np.int64), prototype)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid_coverage_not_refused")
            require(closed == [True], "source_closed_after_coverage_failure")
        report["checks"].append({"case": label, "baseline_and_prototype_refuse_and_close": True})
    cache = Cache()
    state = prototype_class(cache)(1000, readonly(draws[0], np.int64))
    bad_std = supplied[0].training_std.copy()
    bad_std[0] = 0
    try:
        state.consume(replace(supplied[0], training_std=bad_std), "sub-01", TEST_IDS, 1000, budget.check)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid_scale_not_refused")
    require(cache.preparations == 0 and cache.block is None, "cache_must_follow_original_validation")
    report["checks"].append({"case": "invalid_scale", "cache_unprepared_before_validation": True})
    for prototype in (False, True):
        started, closed = [], []
        def source():
            try:
                started.append(True)
                yield supplied[0]
                raise AssertionError("source_resumed_after_cancellation")
            finally:
                closed.append(True)
        def cancel():
            budget.check()
            if started:
                raise TimeoutError("fixture_cancellation")
        try:
            run(source(), readonly(draws[:1], np.int64), prototype, cancel)
        except TimeoutError as error:
            require(str(error) == "fixture_cancellation", "unexpected_timeout")
        else:
            raise AssertionError("cancellation_not_refused")
        require(closed == [True], "source_closed_after_cancellation")
    report["checks"].append({"case": "cancellation", "baseline_and_prototype_close": True})

    report["benchmarks"] = []
    benchmark_draws = readonly(np.random.RandomState(42).randint(0, 224, size=(16, 224)), np.int64)
    for width in (64, 113, 256):
        data = frozen_blocks(predicted, truth, mean, std, width=width)
        # Warm both paths equally, then alternate which one runs first.
        for prototype in (False, True):
            warm, _ = run(iter(data), benchmark_draws, prototype)
            del warm
        timings = {"current": [], "prototype": []}
        for repeat in range(7):
            for prototype in ((False, True) if repeat % 2 == 0 else (True, False)):
                budget.check()
                began = time.perf_counter()
                result, cache_stats = run(iter(data), benchmark_draws, prototype)
                elapsed = time.perf_counter() - began
                timings["prototype" if prototype else "current"].append(elapsed)
                del result
                budget.check()
        current, proposed = (statistics.median(timings[key]) for key in ("current", "prototype"))
        report["benchmarks"].append({"width": width, "blocks": len(data), "draws": 16, "repeats_per_path": 7,
            "seconds": timings, "current_median_seconds": current, "prototype_median_seconds": proposed,
            "speed_ratio_current_over_prototype": current / proposed,
            "maximum_additional_normalized_cache_bytes": 2 * 224 * width * 4,
            "current_block_referenced_array_bytes": 2 * 224 * width * 4 + 16 * width,
            "scope": "fabricated_complete_fc8_only_no_HDF_or_full_panel_timing"})
    budget.check()
    report.update(status="complete", checks_status="passed", comparison="exact_all_returned_fields_including_support",
                  normalization_dtype="float32_before_row_selection", cache_bound="one_actual_block_two_readonly_float32_arrays")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=90.)
    parser.add_argument("--rss-limit-kib", type=int, default=448 * 1024)
    args = parser.parse_args()
    report = {"kind": "normalization_reuse_array_only_benchmark", "status": "failed",
              "empirical_payloads_opened": False, "production_modules_modified_on_disk": False,
              "files_written": False, "job_id": os.environ.get("SLURM_JOB_ID")}
    began = time.monotonic()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    try:
        require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "Slurm_compute_only")
        require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 90, "finite_at_most_90_seconds")
        require(0 < args.rss_limit_kib <= 448 * 1024, "RSS_at_most_448MiB")
        require(all(os.environ.get(name) == "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
                "single_numerical_thread_required")
        require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "existing_timer_not_replaced")
        budget = Budget(args.max_seconds, args.rss_limit_kib)
        def stop(_signal, _frame):
            raise TimeoutError("benchmark_deadline_or_termination")
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        execute(budget, report)
    except BaseException as error:
        report.update(status="failed", checks_status=report.get("checks_status", "incomplete"),
                      error_type=type(error).__name__, error=str(error)[:2000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(elapsed_seconds=time.monotonic() - began,
                      peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
