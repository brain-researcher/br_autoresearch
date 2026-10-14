"""One separately admitted fresh L40S cached measurement, never production.

Only new attempt/device wiring lives here. Existing allocation/budget, cached
reader, full four-update scientific computation and numerical fixture remain
unchanged. No import-time CUDA initialization or empirical access occurs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os

import run_content_cuda_measurement as original

BASE = original.BASE
PROFILE = BASE / "outputs/content_cuda_l40s_measurement_authorization.json"
RUN_DIR = BASE / "outputs/runs/content_resource_measurement/cuda_l40s001"
SCRATCH_NAME = "ep21_content_cuda_l40s001"
DEVICE_NAME = "NVIDIA L40S"
CONSTRAINT = "GPU_SKU:L40S&GPU_MEM:48GB"
require = original._require


class L40SStage(original.MeasurementStage):
    @property
    def cached(self):
        return True

    @property
    def fixed(self):
        return {**super().fixed, "stage": "l40s-001", "device_name": DEVICE_NAME,
                "scheduler_constraint": CONSTRAINT}


def stage():
    return L40SStage("l40s-001", PROFILE, RUN_DIR, SCRATCH_NAME, True)


def admit():
    selected = stage()
    profile = original._profile(selected)
    context = original.MeasurementContext(profile, original._allocation(selected),
                                         original._scratch(selected), selected)
    budget = original.MeasurementBudget(context)
    budget.check_storage()
    original._initialize_cuda(budget)
    require(budget.device_info["name"] == DEVICE_NAME, "selected_L40S_device_required")
    return context, budget


def resolver(context, *, operation, source, target, budget, execution_options):
    require(isinstance(context, original.MeasurementContext)
            and isinstance(budget, original.MeasurementBudget)
            and budget.context is context and budget.allocation == context.allocation,
            "same_CUDA_measurement_context_required")
    require(context.stage == stage() and original._profile(stage()) == context.profile,
            "fresh_L40S_measurement_admission_changed")
    budget.check()
    require(budget.device_info["name"] == DEVICE_NAME, "selected_L40S_device_required")
    require((source, target) == (original.SOURCE, original.TARGET),
            "fixed_CUDA_measurement_pair_required")
    if operation == "bind_content_inputs":
        require(execution_options is None, "binding_has_no_execution_options")
    elif operation == context.stage.fixed["operation"]:
        expected = dict(device="cuda:0", block_units=256, row_chunk_size=512,
            backend="sparse", updates=4, source_rows=4712, layer_count=19,
            final_path=context.stage.run_dir / "reserved_final_model.pt", maximum_bytes=1,
            feature_read_mode="bounded_channel_two_pass_cache",
            maximum_channel_bytes=1073741824, maximum_retained_bytes=4294967296)
        require(execution_options is not None
                and original._same(dict(execution_options), expected),
                "fixed_CUDA_measurement_options_required")
    else:
        raise original.AdmissionError("operation_not_authorized")


def run_once():
    context, budget = admit()
    selected = context.stage
    reserved = selected.run_dir / "reserved_final_model.pt"
    require(not any(os.path.lexists(path) for path in (
        selected.run_dir / "claim.json", selected.run_dir / "report.json", reserved,
        reserved.with_name(reserved.name + ".partial"))), "existing_attempt_no_retry")
    original._publish(selected.run_dir / "claim.json", dict(
        operation=selected.fixed["operation"], stage=selected.name,
        allocation=context.allocation, device=budget.device_info,
        approval_source=context.profile["approval_source"], automatic_retry=False,
        production_training=False, model_publication=False), selected)
    report, interrupted = None, False
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            bind, measure = original._load_operations(selected)

            def prepare_inputs():
                return bind(context, require_admitted_context=resolver,
                    source=original.SOURCE, target=original.TARGET, budget=budget)

            report = measure(context, require_admitted_context=resolver,
                prepare_inputs=prepare_inputs, budget=budget, reserved_final_path=reserved)
        budget.check_storage()
        require(isinstance(report, dict) and report.get("status") in ("complete_measurement", "incomplete")
                and report.get("model_published") is False
                and report.get("production_fit_completed") is False,
                "operational_CUDA_measurement_report_required")
        if report["status"] == "complete_measurement":
            require(type(report.get("completed_updates")) is int and report["completed_updates"] == 4,
                    "all_four_measurement_updates_required")
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if not isinstance(report, dict):
            report = dict(kind="ep21_content_pair_cuda_resource_measurement",
                source=original.SOURCE, target=original.TARGET, completed_updates=None,
                production_fit_completed=False, model_published=False, resumable=False,
                allocation_identity={key: context.allocation[key] for key in ("job_id", "hostname")})
        report.update(status="incomplete", error_type=type(error).__name__,
                      failed_stage="L40S_measurement_entrypoint")
    report.update(stage=selected.name, admitted_device=budget.device_info,
        feature_read_mode=selected.fixed["feature_read_mode"],
        maximum_channel_bytes=selected.fixed["maximum_channel_bytes"],
        maximum_retained_bytes=selected.fixed["maximum_retained_bytes"])
    original._publish(selected.run_dir / "report.json", report, selected)
    return 130 if interrupted else 0 if report["status"] == "complete_measurement" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-admission", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.check_admission:
            context, budget = admit()
            print(json.dumps(dict(status="L40S_measurement_admitted",
                stage=context.stage.name, job_id=context.allocation["job_id"],
                device=budget.device_info, payload_read=False, attempt_claimed=False)))
            return 0
        return run_once()
    except BaseException as error:
        print(json.dumps(dict(status="L40S_measurement_entrypoint_refused", error_type=type(error).__name__)))
        return 130 if isinstance(error, KeyboardInterrupt) else 2


if __name__ == "__main__":
    raise SystemExit(main())
