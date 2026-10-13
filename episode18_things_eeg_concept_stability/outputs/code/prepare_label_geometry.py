#!/usr/bin/env python3
"""Finite label-only PCA/RBF preparation; no EEG, image or event-payload reader.

Code-only until bundled into a substantive coordinator-owned allocation. The
first necessary kernel is the cost pilot and remains its completed cache;
there is no minimum-runtime padding, repeated equivalent fit or successor job.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gc
from itertools import product
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from label_rbf import BANDWIDTH_MULTIPLIERS, fit_label_rbf_basis
from qualify_model_runtime import read_json, require, require_compute
from slot_capacity import RIDGE_GRID
from training_transforms import (FeaturePanelTransform, LABEL,
                                 RANK_GRID, fit_feature_transform)


EPISODE = Path(__file__).resolve().parents[2]
PREPARATION = EPISODE / "outputs/runs/feature_preparation"
NAMES = PREPARATION / "all_name_text_46843116"
HUMAN = PREPARATION / "20261003-human_metadata01"
STRUCTURAL = EPISODE / "outputs/runs/input_role_preparation/20261003-metadata01/structural_input_report.json"
FROZEN = EPISODE / "outputs/runs/input_role_preparation/20261003-freeze_metadata01/frozen_inputs_and_roles.json"
REGISTRY = EPISODE / "outputs/config/feature_registry.json"
CONCEPTS = 1854
MAX_SECONDS = 1800.
PROJECTION_FACTOR = 2.
FINISH_RESERVE_SECONDS = 30.
# The rank scale follows the registered float64 matrix-rank convention.
# The two reconstruction limits are fixed numerical checks, never EEG-tuned.
NUMERICS = dict(eigenvalue_rtol=CONCEPTS * np.finfo(np.float64).eps,
                eigenvalue_atol=0., reconstruction_rtol=1e-10,
                reconstruction_max_atol=1e-9)


def load_labels():
    registry, frozen, structural = map(read_json, (REGISTRY, FROZEN, STRUCTURAL))
    human_report = read_json(HUMAN / "human_metadata_report.json")
    name_report = read_json(NAMES / "label_feature_report.json")
    require(registry["label_panel"] == list(LABEL)
            and registry["training_only_transforms"]["rank_grid"] == list(RANK_GRID),
            "Use the unchanged registered label panel and ranks")
    require(frozen["roles_frozen"] and frozen["join_frozen"], "Frozen common inventory is required")
    participants = set(frozen["roles"]["development"] + frozen["roles"]["audit"])
    results = structural["participant_results"]
    require(len(participants) == len(results) == 46
            and {record["participant"] for record in results} == participants,
            "Structural concept evidence must cover exactly the frozen 46 participants")
    required_checks = ("main_blocks_0_to_11", "each_block_has_each_concept_once",
                       "1854_concepts_with_12_distinct_exemplars")
    require(all(record["checks"][name] for record in results for name in required_checks),
            "Every allowed eight/ten-block training set must contain the same 1854 concepts")
    # Other historical structural failures concerned sample timing and were
    # resolved by the frozen join. They are not silently relabelled as passing.
    require(human_report["status"] == "source_crosswalk_complete_no_missing_values"
            and human_report["frozen_main_concepts"] == CONCEPTS,
            "The completed human-label crosswalk is required")
    require(name_report["status"] == "all1854_label_text_features_completed"
            and name_report["completed_concepts"] == CONCEPTS
            and name_report["feature_shape"] == [CONCEPTS, 512],
            "Use the completed qualified name embeddings")
    ids = tuple(row["uniqueID"] for row in read_json(NAMES / "concept_ids_and_tokens.json"))
    require(len(ids) == len(set(ids)) == CONCEPTS, "Name-feature IDs must be complete and unique")
    names = np.load(NAMES / "concept_name_clip_text.npy", allow_pickle=False)
    require(names.shape == (CONCEPTS, 512) and names.dtype == np.float32 and np.isfinite(names).all(),
            "Qualified name features must remain finite float32[1854,512]")
    with (HUMAN / "human_features_untransformed.tsv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        properties = registry["features"]["human_properties_12"]["selected_columns"]
        categories = [name for name in reader.fieldnames if name.startswith("category53::")]
        require(len(properties) == 12 and len(categories) == 53
                and categories == ["category53::" + name for name in human_report["category_columns"]],
                "Preserve the registered property order and all source-ordered category columns")
        human = {}
        for row in reader:
            require(row["uniqueID"] not in human, "Duplicate human-feature concept ID")
            human[row["uniqueID"]] = [float(row[name]) for name in properties + categories]
    require(set(human) == set(ids), "Human and name features must have the exact same concept inventory")
    values = np.asarray([human[key] for key in ids], dtype=np.float64)
    require(np.isfinite(values).all(), "No imputation or concept removal is permitted")
    require(np.isin(values[:, 12:], [0., 1.]).all(), "Category membership must remain binary")
    return ids, {LABEL[0]: names, LABEL[1]: values[:, :12], LABEL[2]: values[:, 12:]}


def run(args):
    require_compute()
    require(os.environ.get("SLURM_MEM_PER_NODE") == "7168" and not os.environ.get("SLURM_JOB_GPUS"),
            "Use exactly one CPU, 7 GiB and no GPU")
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= MAX_SECONDS,
            "Finite runtime must be positive and no more than 30 minutes")
    output = args.output_dir.absolute()
    require(not output.exists() and not output.is_symlink()
            and output.parent.resolve() == PREPARATION.resolve()
            and output.name.startswith("label_geometry_"), "Use a fresh episode-local label_geometry attempt")
    output.mkdir()
    started = time.monotonic()
    paths = []
    report = dict(analysis_id="feature_preparation", phase="label_only_geometry_preparation", status="running",
                  job_id=os.environ["SLURM_JOB_ID"], hostname=socket.gethostname(),
                  started_utc=datetime.now(timezone.utc).isoformat(), max_seconds=args.max_seconds,
                  numeric_tolerances=NUMERICS, projection_safety_factor=PROJECTION_FACTOR,
                  finish_reserve_seconds=FINISH_RESERVE_SECONDS, kernels=[], rank_width_aliases=[],
                  repeated_equivalent_fits=False, EEG_or_image_or_event_payload_read=False,
                  frozen_feature_record_emitted=False, audit_seal_verified=False,
                  full_temporal_operator_approximation_qualified=False,
                  capacity_matching_or_penalty_selection_performed=False, automatic_successor=False)

    def write():
        report.update(elapsed_seconds=time.monotonic() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cached_array_bytes=sum(path.stat().st_size for path in paths))
        temporary = output / "label_geometry_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "label_geometry_report.json")

    def interrupt(signum, frame):
        raise TimeoutError(f"Finite preparation interrupted by signal {signum}")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupt)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        write()
        ids, families = load_labels()
        full = {name: fit_feature_transform(families[name], training_unit_ids=ids, requested_rank=128)
                for name in LABEL}
        arrays = {"concept_ids": np.asarray(ids)}
        for name, fit in full.items():
            for field in ("mean", "population_sd", "loadings", "singular_values", "zero_sd_columns"):
                arrays[name + "__" + field] = np.asarray(getattr(fit, field))
        pca_path = output / "label_pca_parameters.npz"
        np.savez(pca_path, **arrays)
        paths.append(pca_path)
        report["family_numerical_ranks"] = {name: fit.numerical_rank for name, fit in full.items()}
        report["family_rank_thresholds"] = {name: fit.rank_threshold for name, fit in full.items()}
        report["concepts"] = len(ids)
        report["training_unit_reuse"] = "Identical unique concepts in every eligible eight/ten-block training set; no held-out-only units"
        panels = {}
        for requested in product(RANK_GRID, repeat=3):
            transforms = tuple(replace(full[name], requested_rank=rank,
                                       effective_rank=min(rank, full[name].effective_rank),
                                       loadings=full[name].loadings[:min(rank, full[name].effective_rank)])
                               for name, rank in zip(LABEL, requested))
            effective = tuple(fit.effective_rank for fit in transforms)
            key = "r" + "_".join(map(str, effective))
            if key not in panels:
                panels[key] = FeaturePanelTransform("label", LABEL, transforms)
            for multiplier in BANDWIDTH_MULTIPLIERS:
                report["rank_width_aliases"].append(dict(requested_ranks=list(requested),
                    effective_ranks=list(effective), bandwidth_multiplier=multiplier,
                    geometry=key, kernel=key + "_bw" + str(multiplier).replace(".", "p")))
        report.update(requested_rank_triples=27, requested_rank_width_mappings=81,
                      distinct_geometries=len(panels), distinct_kernels=3 * len(panels))
        row_ids = {name: ids for name in LABEL}
        kernel_seconds = []
        for key, panel in panels.items():
            rows = panel.apply(families, row_ids=row_ids)
            geometry_path = output / (key + ".npz")
            np.savez(geometry_path, transformed_labels=rows)
            paths.append(geometry_path)
            for multiplier in BANDWIDTH_MULTIPLIERS:
                if kernel_seconds and (time.monotonic() - started + PROJECTION_FACTOR * max(kernel_seconds)
                                       + FINISH_RESERVE_SECONDS >= args.max_seconds):
                    report["status"] = "time_guard_partial"
                    write()
                    return 2
                kernel_started = time.monotonic()
                fit = fit_label_rbf_basis(rows, training_unit_ids=ids, label_panel_transform=panel,
                                          bandwidth_multiplier=multiplier, **NUMERICS)
                kernel_id = key + "_bw" + str(multiplier).replace(".", "p")
                path = output / (kernel_id + ".npz")
                np.savez(path, eigenvectors=fit.retained_eigenvectors, eigenvalues=fit.retained_eigenvalues,
                         complete_spectrum=fit.complete_eigenvalue_spectrum,
                         discarded_eigenvalues=fit.discarded_eigenvalues)
                paths.append(path)
                record = dict(kernel=kernel_id, geometry=key, bandwidth_multiplier=multiplier,
                              bandwidth=fit.bandwidth, median_positive_training_distance=fit.median_positive_training_distance,
                              effective_rank=fit.effective_rank, eigenvalue_threshold=fit.eigenvalue_threshold,
                              relative_frobenius_error=fit.reconstruction_relative_frobenius_error,
                              max_absolute_error=fit.reconstruction_max_absolute_error)
                discarded = float(np.max(np.abs(fit.discarded_eigenvalues), initial=0.))
                record.update(maximum_discarded_absolute_eigenvalue=discarded,
                              discarded_to_minimum_ridge_diagnostic=discarded / min(RIDGE_GRID))
                del fit
                gc.collect()
                elapsed = time.monotonic() - kernel_started
                kernel_seconds.append(elapsed)
                record["elapsed_seconds"] = elapsed
                report["kernels"].append(record)
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 7 * 1024**2,
                        "Observed memory must remain below the 7-GiB envelope")
                if len(kernel_seconds) == 1:
                    projected = (time.monotonic() - started + PROJECTION_FACTOR * elapsed
                                 * (report["distinct_kernels"] - 1) + FINISH_RESERVE_SECONDS)
                    report["pilot_kernel"] = kernel_id
                    report["projected_total_seconds_after_pilot"] = projected
                    if projected >= args.max_seconds:
                        report["status"] = "pilot_complete_projected_remainder_exceeds_bound"
                        write()
                        return 2
                write()
            del rows
        report["status"] = "label_geometry_cache_complete"
        write()
        return 0
    except TimeoutError as error:
        report.update(status="time_guard_partial", error=str(error))
        write()
        return 2
    except Exception as error:
        report.update(status="technical_failure", error=f"{type(error).__name__}: {error}")
        write()
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    raise SystemExit(run(parser.parse_args()))
