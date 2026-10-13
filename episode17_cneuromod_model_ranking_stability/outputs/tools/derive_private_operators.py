#!/usr/bin/env python3
"""Finish calibration-defined operators from completed private artifacts only.

No raw neural/image reader, model feature, candidate score, development target,
reader release or audit is implemented. Run once in coordinated compute with
one CPU, 2 GiB, 120 seconds internally / 180 externally (+10 seconds kill),
and a 32 MiB new-output cap. All applicability decisions and maps stay private.
An inapplicable gain design never removes its beta primary endpoints or voxels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time
import zipfile

import numpy as np

import calibrate_private_roles as calibration
import measurement_core as measurement
import support_core as support

core = calibration.core
SOURCE_PRIVATE, SOURCE_PUBLIC = calibration.paths("full")
PRIVATE = calibration.original.CACHE_PARENT / "private/calibration_operators/attempt001"
PUBLIC = calibration.EPISODE / "outputs/runs/calibration_operators/attempt001"
INTERNAL_SECONDS = 120
OUTPUT_CAP = 32 * 1024**2
INPUT_CAP = 32 * 1024**2
GROUP_KEYS = tuple((s, r) for s in measurement.SUBJECTS for r in measurement.ROIS)
EDGES = {"B_to_C": ("B", "C"), "C_to_D": ("C", "D")}
STAGE_FIELDS = ("column_ids", "response_eligible", "reliability_valid", "fisher_z",
                "ceiling_valid", "noise_ceiling", "ceiling_denominator")
SUPPORT_FIELDS = ("column_ids", "spatial_bins", "A_all", "A_N_banks",
                  "A_N_inclusion_counts", "R_N", "V_NC")
GAIN_INAPPLICABILITY = {"calibration_gain_variance_zero", "residual_gain_variance_below_floor",
                       "gain_design_rank_deficient"}


def validate_source_report(report):
    core.require(report.get("analysis_id") == "calibration_support" and report.get("mode") == "full" and
                 report.get("status") == "completed" and report.get("errors") == [] and
                 report.get("calibration_support_complete") is True and report.get("derived_arrays_verified") is True and
                 report.get("private_attempt") == str(SOURCE_PRIVATE) and
                 report.get("full_copy_attempt") == str(calibration.COPY_PRIVATE) and
                 report.get("selected_original_columns") == 8558 and report.get("eligible_image_count") == 605 and
                 report.get("products_planned") == 12 and report.get("completed_physical_stripes") == 3258 and
                 report.get("planned_physical_stripes") == 3258,
                 "completed_private_calibration_attempt001_required")


def load_npz(path, required_keys, check):
    """Only fixed uncompressed NumPy artifacts from the completed builder."""
    size = calibration.copied.ordinary_private_file(path, SOURCE_PRIVATE)
    core.require(size <= INPUT_CAP, "private_operator_input_byte_cap")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        core.require(len(names) == len(set(names)) and len(entries) <= 512 and
                     all(entry.compress_type == zipfile.ZIP_STORED and not entry.is_dir() and
                         entry.file_size >= 0 for entry in entries) and
                     sum(entry.file_size for entry in entries) <= INPUT_CAP,
                     "private_operator_archive_representation_changed")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        core.require(set(required_keys).issubset(archive.files), "private_operator_artifact_keys_missing")
        for key in required_keys:
            check()
            value = archive[key]
            core.require(value.dtype.kind in "biufU" and value.nbytes <= INPUT_CAP,
                         "private_operator_array_representation_changed")
            result[key] = value
    return result


def load_inputs(check):
    validate_source_report(calibration.original.load_json(SOURCE_PUBLIC / "calibration_report.json"))
    path = SOURCE_PRIVATE / "read_plan.json"
    core.require(calibration.copied.ordinary_private_file(path, SOURCE_PRIVATE) <= INPUT_CAP,
                 "private_operator_input_byte_cap")
    plan = calibration.original.load_json(path)
    binding = plan["binding"]
    core.require(binding.get("full_copy_attempt") == str(calibration.COPY_PRIVATE) and
                 binding.get("role_assignment") == str(calibration.original.ASSIGNMENT) and
                 binding.get("structural_attempt") == str(calibration.STRUCTURAL),
                 "private_calibration_binding_changed")
    maps = load_npz(SOURCE_PRIVATE / "calibration_maps.npz",
                    [f"{subject}_{stage}_{name}" for subject in measurement.SUBJECTS
                     for stage in calibration.STAGES for name in STAGE_FIELDS], check)
    supports = load_npz(SOURCE_PRIVATE / "supports.npz",
                        [f"{subject}_{roi}_{name}" for subject, roi in GROUP_KEYS for name in SUPPORT_FIELDS], check)
    return plan, maps, supports


def ids(value, name, upper):
    raw = np.asarray(value)
    core.require(raw.ndim == 1 and raw.dtype.kind in "iu" and np.all(raw < upper) and
                 np.all(raw >= 0) and (len(raw) < 2 or np.all(raw[1:] > raw[:-1])), name + "_identity_changed")
    return raw.astype(np.int64, copy=False)


def vector(value, width, name, *, boolean=False):
    raw = np.asarray(value)
    core.require(raw.shape == (width,) and (raw.dtype.kind == "b" if boolean else raw.dtype.kind == "f"),
                 name + "_representation_changed")
    return raw


def bind_stage_axes(plan, maps):
    products = plan.get("products", [])
    core.require(len(products) == 12 and {(p["participant"], p["stage"]) for p in products} ==
                 {(s, stage) for s in measurement.SUBJECTS for stage in calibration.STAGES},
                 "private_calibration_product_roster_changed")
    result = {}
    for subject in measurement.SUBJECTS:
        for stage in calibration.STAGES:
            prefix = f"{subject}_{stage}_"
            item = next(p for p in products if (p["participant"], p["stage"]) == (subject, stage))
            axis = ids(maps[prefix + "column_ids"], "stage_original_columns", support.PARTICIPANT_COLUMNS[subject])
            core.require(len(axis) > 0 and np.array_equal(axis, np.asarray(item["column_ids"])) and
                         item["path"] == str(calibration.COPY_PRIVATE / calibration.ROLE / subject / f"{stage}.h5"),
                         "private_stage_axis_or_role_changed")
            fields = {"column_ids": axis}
            for name in STAGE_FIELDS[1:]:
                fields[name] = vector(maps[prefix + name], len(axis), name,
                                      boolean=name in ("response_eligible", "reliability_valid", "ceiling_valid"))
            core.require(np.isfinite(fields["fisher_z"][fields["reliability_valid"]]).all(),
                         "qualified_fisher_map_nonfinite")
            if stage != "B":
                core.require(np.array_equal(axis, result[(subject, "B")]["column_ids"]),
                             "cross_stage_original_axis_changed")
            result[(subject, stage)] = fields
    return result


def derive(plan, maps, supports, check=lambda: None):
    """Join original IDs; compute maps only, never accept a model advantage."""
    stage_maps = bind_stage_axes(plan, maps)
    frozen = plan["binding"]["structural_columns"]
    core.require(set(frozen) == {f"{s}_{r}" for s, r in GROUP_KEYS}, "eight_frozen_structural_groups_required")
    arrays, decisions = {}, {}
    for subject, roi in GROUP_KEYS:
        check()
        prefix = f"{subject}_{roi}_"
        original = ids(supports[prefix + "column_ids"], "support_original_columns", support.PARTICIPANT_COLUMNS[subject])
        spatial = np.asarray(supports[prefix + "spatial_bins"])
        expected = frozen[f"{subject}_{roi}"]
        core.require(len(original) > 0 and spatial.shape == original.shape and spatial.dtype.kind == "U" and
                     [[int(ident), str(bin_name), support.ROI_LABELS[roi]] for ident, bin_name in zip(original, spatial)] == expected,
                     "frozen_support_columns_or_bins_changed")
        global_axis = stage_maps[(subject, "B")]["column_ids"]
        lookup = {int(ident): index for index, ident in enumerate(global_axis)}
        core.require(set(original).issubset(lookup), "support_column_missing_from_stage_axis")
        take = np.asarray([lookup[int(ident)] for ident in original])
        eligible = np.logical_and.reduce([stage_maps[(subject, stage)]["response_eligible"][take]
                                         for stage in calibration.STAGES])
        a_all = ids(supports[prefix + "A_all"], "A_all", support.PARTICIPANT_COLUMNS[subject])
        core.require(np.array_equal(a_all, original[eligible]) and len(a_all) >= support.N,
                     "frozen_A_all_changed")
        roi_lookup = {int(ident): index for index, ident in enumerate(original)}
        a_take = np.asarray([roi_lookup[int(ident)] for ident in a_all])
        a_stage_take = take[a_take]
        a_bins = np.asarray([int(value[3:]) for value in spatial[a_take]], dtype=np.int64)
        core.require(set(a_bins.tolist()) == set(range(1, 9)) and
                     all(np.count_nonzero(a_bins == b) >= support.Q for b in range(1, 9)),
                     "frozen_A_all_bin_quota_changed")
        arrays[prefix + "A_all"] = a_all
        arrays[prefix + "A_all_spatial_bins"] = a_bins
        arrays[prefix + "A_all_weights"] = np.full(len(a_all), 1. / len(a_all))
        group = {"endpoint_calibration_prerequisites": {name: True for name in
                 ("B_Aall_raw", "C_Aall_raw", "D_Aall_raw", "D_AN_raw", "D_RN_raw")},
                 "beta_explanations": {}, "full_endpoint_eligibility_requires_separate_development_checks": True}
        for edge, (before, after) in EDGES.items():
            check()
            left, right = stage_maps[(subject, before)], stage_maps[(subject, after)]
            valid = left["reliability_valid"][a_stage_take] & right["reliability_valid"][a_stage_take]
            gain = np.full(len(a_all), np.nan)
            gain[valid] = right["fisher_z"][a_stage_take][valid] - left["fisher_z"][a_stage_take][valid]
            edge_prefix = prefix + edge + "_"
            arrays[edge_prefix + "gain"] = gain
            arrays[edge_prefix + "gain_defined"] = valid
            decision = {"applicable": False, "reason": None, "axis_key": prefix + "A_all",
                        "gain_key": edge_prefix + "gain", "beta_primary_endpoints_removed": False,
                        "frozen_columns_removed": False}
            if not np.all(valid) or not np.isfinite(gain).all():
                decision["reason"] = "undefined_calibration_gain"
            else:
                try:
                    design = measurement.beta_gain_design(gain, a_bins, a_all)
                except measurement.MeasurementError as error:
                    reason = str(error)
                    core.require(reason in GAIN_INAPPLICABILITY, "unexpected_gain_design_failure")
                    decision["reason"] = reason
                else:
                    arrays[edge_prefix + "standardized_gain"] = design.standardized_gain
                    arrays[edge_prefix + "design"] = design.matrix
                    arrays[edge_prefix + "low_columns"] = a_all[design.low_indices]
                    arrays[edge_prefix + "high_columns"] = a_all[design.high_indices]
                    arrays[edge_prefix + "residual_gain_variance"] = np.asarray([design.residual_gain_variance])
                    decision.update(applicable=True, reason="qualified_under_frozen_gain_rule")
            group["beta_explanations"][edge] = decision
        banks = np.asarray(supports[prefix + "A_N_banks"])
        core.require(banks.dtype.kind in "iu" and banks.shape == (len(support.BANKS), support.N) and
                     all(len(set(row)) == support.N and set(row).issubset(set(a_all)) for row in banks),
                     "frozen_anatomical_bank_changed")
        a_index = {int(ident): i for i, ident in enumerate(a_all)}
        counts = np.zeros(len(a_all), dtype=np.int64)
        for bank in banks:
            positions = np.asarray([a_index[int(ident)] for ident in bank])
            core.require(all(np.count_nonzero(a_bins[positions] == b) == support.Q for b in range(1, 9)),
                         "frozen_anatomical_bin_quota_changed")
            counts[positions] += 1
        saved_counts = np.asarray(supports[prefix + "A_N_inclusion_counts"])
        core.require(saved_counts.shape == original.shape and saved_counts.dtype.kind in "iu" and
                     np.array_equal(saved_counts[a_take], counts) and np.all(saved_counts[~eligible] == 0),
                     "frozen_anatomical_inclusion_counts_changed")
        reliable = ids(supports[prefix + "R_N"], "R_N", support.PARTICIPANT_COLUMNS[subject])
        core.require(len(reliable) == support.N and set(reliable).issubset(a_index), "frozen_reliable_support_changed")
        r_positions = np.asarray([a_index[int(ident)] for ident in reliable])
        core.require(all(np.count_nonzero(a_bins[r_positions] == b) == support.Q for b in range(1, 9)) and
                     np.all(stage_maps[(subject, "D")]["reliability_valid"][a_stage_take[r_positions]]),
                     "frozen_reliable_bin_quota_changed")
        anatomical_weights = counts.astype(float) / (len(support.BANKS) * support.N)
        reliable_weights = np.isin(a_all, reliable).astype(float) / support.N
        arrays[prefix + "A_N_weights"] = anatomical_weights
        arrays[prefix + "R_N_weights"] = reliable_weights
        arrays[prefix + "A_all_to_A_N_operator"] = anatomical_weights - arrays[prefix + "A_all_weights"]
        arrays[prefix + "A_N_to_R_N_operator"] = reliable_weights - anatomical_weights
        d = stage_maps[(subject, "D")]
        vnc = ids(supports[prefix + "V_NC"], "V_NC", support.PARTICIPANT_COLUMNS[subject])
        core.require(np.array_equal(vnc, a_all[d["ceiling_valid"][a_stage_take]]), "frozen_V_NC_changed")
        denominator = d["ceiling_denominator"][np.asarray([lookup[int(ident)] for ident in vnc], dtype=np.int64)]
        ceiling = d["noise_ceiling"][np.asarray([lookup[int(ident)] for ident in vnc], dtype=np.int64)]
        core.require(np.isfinite(ceiling).all() and np.isfinite(denominator).all() and
                     np.array_equal(denominator, np.maximum(ceiling, .10)), "frozen_ceiling_denominator_changed")
        arrays[prefix + "V_NC"] = vnc
        arrays[prefix + "V_NC_ceiling_denominator"] = denominator
        arrays[prefix + "V_NC_positive_ceiling_operator"] = 1. / denominator
        # An empty calibrated ceiling support is recorded privately, with no
        # positive-ceiling salvage and no division by its zero size.
        arrays[prefix + "V_NC_raw_weights"] = np.full(len(vnc), 1. / len(vnc)) if len(vnc) else np.empty(0)
        arrays[prefix + "V_NC_normalized_weights"] = (1. / denominator) / len(vnc) if len(vnc) else np.empty(0)
        arrays[prefix + "raw_to_ceiling_normalized_difference_weights"] = (1. / denominator - 1.) / len(vnc) if len(vnc) else np.empty(0)
        group["ceiling_operator"] = {
            "axis_key": prefix + "V_NC",
            "positive_g_key": prefix + "V_NC_positive_ceiling_operator",
            "positive_g_definition": "g(v)=1/max(NC_D(v),0.10);not_normalized_to_sum_one",
            "edge_difference_weights_key": prefix + "raw_to_ceiling_normalized_difference_weights",
            "edge_difference_weights_definition": "(g(v)-1)/number_of_frozen_V_NC_columns",
            "normalized_endpoint_weights_key": prefix + "V_NC_normalized_weights",
            "normalized_endpoint_weights_definition": "g(v)/number_of_frozen_V_NC_columns"}
        group["endpoint_calibration_prerequisites"].update(D_VNC_raw=bool(len(vnc)), D_VNC_normalized=bool(len(vnc)))
        decisions[f"{subject}_{roi}"] = group
    check()
    private_report = {"source_private_calibration": str(SOURCE_PRIVATE), "groups": decisions,
                      "H_component_calibration_applicability": {
                          "H_BC": all(item["beta_explanations"]["B_to_C"]["applicable"] for item in decisions.values()),
                          "H_CD": all(item["beta_explanations"]["C_to_D"]["applicable"] for item in decisions.values())},
                      "development_forecast_fitted": False, "candidate_scores_consumed": False,
                      "worker_release_or_final_endpoint_eligibility_established": False}
    return arrays, private_report


def run():
    core.require(not PRIVATE.exists() and not PRIVATE.is_symlink() and not PUBLIC.exists() and not PUBLIC.is_symlink(),
                 "private_operator_attempt_already_exists")
    PUBLIC.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"analysis_id": "calibration_support", "status": "running", "errors": [],
              "operator_artifacts_complete": False, "private_calibration_artifacts_read": False,
              "raw_neural_values_read": False, "stimulus_values_read": False, "candidate_scores_consumed": False,
              "development_forecast_fitted": False, "reader_handoffs_activated": False,
              "candidate_scoring_authorized": False, "scientific_audit_evaluation_count": 0,
              "source_private_calibration": str(SOURCE_PRIVATE), "private_attempt": str(PRIVATE),
              "cpus": 1, "memory_bound_bytes": calibration.MEMORY_BYTES, "internal_seconds": INTERNAL_SECONDS,
              "external_seconds_required": 180, "external_kill_grace_seconds": 10, "output_byte_cap": OUTPUT_CAP,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname()}
    def check():
        core.require(time.monotonic() - started < INTERNAL_SECONDS, "private_operator_deadline")
    def alarm(_signum, _frame):
        raise core.Refusal("private_operator_deadline")
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(INTERNAL_SECONDS)
    budget = None
    try:
        # Exposure flag is conservative if an archive read fails partway.
        validate_source_report(calibration.original.load_json(SOURCE_PUBLIC / "calibration_report.json"))
        report["private_calibration_artifacts_read"] = True
        plan, maps, supports = load_inputs(check)
        arrays, applicability = derive(plan, maps, supports, check)
        calibration.private_directory(PRIVATE, calibration.original.CACHE_PARENT)
        budget = calibration.copied.OutputBudget(0, cap=OUTPUT_CAP)
        calibration.save_arrays(PRIVATE / "operators.npz", arrays, budget, check)
        calibration.write_private(PRIVATE / "applicability.json",
                                  (json.dumps(applicability, sort_keys=True, allow_nan=False) + "\n").encode(), budget)
        check()
        report.update(status="completed", operator_artifacts_complete=True,
                      participant_ROI_groups_processed=len(GROUP_KEYS), beta_group_edges_processed=2 * len(GROUP_KEYS))
    except core.Refusal as error:
        report.update(status="stopped", errors=[{"code": error.detail["code"]}])
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, MemoryError, zipfile.BadZipFile):
        report.update(status="stopped", errors=[{"code": "private_operator_processing_failure"}])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    report.update(elapsed_seconds=time.monotonic() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  private_output_bytes=sum(budget.sizes.values()) if budget else 0)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    core.require(len(encoded.encode()) <= calibration.copied.REPORT_RESERVE, "private_operator_public_report_byte_cap")
    with (PUBLIC / "operator_report.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({key: report[key] for key in ("status", "operator_artifacts_complete", "elapsed_seconds", "peak_rss_kib")}))
    return 0 if report["status"] == "completed" else 2


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    core.require(os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname(), "verified_compute_allocation_required")
    core.require(os.environ.get("SLURM_CPUS_PER_TASK") == "1" and all(os.environ.get(name) == "1" for name in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
                 "single_cpu_thread_environment_required")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = calibration.MEMORY_BYTES if soft == resource.RLIM_INFINITY else min(soft, calibration.MEMORY_BYTES)
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
