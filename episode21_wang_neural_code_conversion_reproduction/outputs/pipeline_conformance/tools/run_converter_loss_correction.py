"""Validate the diagnosed tagged-loss correction inside a substantive EP21 job.

Reads public source and existing conformance metadata only. The historical
lock/report stay immutable; this report corrects their content-loss fields
without authorizing converter fitting, scoring, or another allocation.
"""
from __future__ import annotations

import os
from pathlib import Path
import socket
import time

import feature_protocol as protocol


OUTPUT = protocol.BASE / "outputs/pipeline_conformance/converter_loss_correction_001/report.json"
OLD_REPORT = "outputs/pipeline_conformance/attempt-008/conformance_report.json"
OLD_LOSS = "float32_norms_of_complete_active_blocks_inverse_squared_then_normalize_sum_to_one_times_per_block_mean_MSE_sum_over_layers"
NEW_LOSS = "float32_norms_of_complete_active_blocks_inverse_squared_then_normalize_sum_to_one_times_per_block_sum_squared_error_sum_over_layers"


def main():
    require = protocol.require
    job = os.environ.get("SLURM_JOB_ID", "")
    host = socket.gethostname().split(".")[0]
    require(Path.cwd().resolve() == protocol.BASE and job.isdigit() and "-ln" not in host,
            "actual_EP21_compute_shell_required")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "1"
            and os.environ.get("SLURM_MEM_PER_NODE") == "7168"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "selected_one_CPU_seven_GiB_CPU_only_parent_required")
    require(all(os.environ.get(key) == "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")),
        "one_thread_limits_required")
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "correction_report_cannot_replay")
    old = protocol.read_json(protocol.BASE / OLD_REPORT)
    require(old.get("converter_conformance", {}).get("loss") == OLD_LOSS,
            "diagnosed_historical_loss_required")
    public = protocol.read_json(protocol.BASE / "outputs/pipeline_conformance/attempt-007/public_runtime_conformance_report.json")
    started = time.monotonic()
    corrected = protocol.converter_conformance()
    protocol.validate_loaded_runtime(public, corrected)
    require(corrected.get("passed") is True and corrected.get("loss") == NEW_LOSS,
            "corrected_source_sum_fixture_required")
    report = {
        "astra_analysis_id": "pipeline_conformance",
        "status": "passed_source_sum_loss_correction",
        "job_id": job, "hostname": host,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "public_source_and_small_analytic_fixture_only",
        "historical_report": OLD_REPORT,
        "historical_lock": "outputs/protocol_lock.yaml",
        "corrects_only": [
            "converter_conformance.loss",
            "resolved_protocol_table.content_converter_contract.content_loss.exact_weighting_layer_aggregation_and_numerical_stability",
        ],
        "previous_loss_claim": OLD_LOSS,
        "converter_conformance": corrected,
        "decision_basis": "outputs/protocol_decisions_2026-09-30.yaml:adopted_tagged_choices.content_converter",
        "historical_artifacts_preserved": True,
        "decoder_fits_and_VGG_features_affected": False,
        "empirical_fitting_scoring_authorized": False,
        "neural_or_image_values_read": False,
        "OOD_access": False,
        "parent_charged_once_to": "EP21",
        "terminal_accounting_by_coordinator": True,
    }
    protocol.publish_json(OUTPUT, report)
    print(f"Converter loss correction passed; report={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
