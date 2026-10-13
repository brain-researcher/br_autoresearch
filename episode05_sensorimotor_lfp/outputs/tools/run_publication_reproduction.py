#!/usr/bin/env python3
"""Run one exposed M1 session through unchanged pinned publication functions.

The transient MATLAB input contains only qualified 19-bin M1/velocity windows.
This is reference-code reproduction, not the later held-direction primary.
All source I/O, acquisition, fixtures and MATLAB execution require Slurm.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request

import numpy as np
from scipy.io import savemat

from source_adapter import M1_SESSIONS, M1Session, compute_only, write_json

ANALYSIS_REVISION = "cbda8e2e6106f5eb5ff98e18a689c595179ac5db"
TRIALDATA_REVISION = "d5e5eeb1af592cf88c03599df7433878f9d11bbe"


def acquire_code(destination, repository, revision):
    """Fetch a small pinned code archive, never a scientific data source."""
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar.gz"
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        root = members[0].name.split("/")[0]
        if not root.endswith(revision):
            raise ValueError("archive root does not identify the selected revision")
        for member in members:
            # Only ordinary files/directories; no execution, links or traversal.
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unexpected member in selected code archive")
        bundle.extractall(destination, members=members, filter="data")
    return destination / root


def export_selected_windows(source, table, session_id, destination):
    selected = next((s for s in M1_SESSIONS if f"{s[0]}_{s[2]}" == session_id), None)
    if selected is None:
        raise ValueError("session is outside the sixteen exposed M1 sources")
    with table.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["session_id"] == session_id]
    if not rows or any(row["publication_execution_support"] != "True" or
                       row["result"] != "R" for row in rows):
        raise ValueError("qualified successful whole-trial identities are required")
    indices = [int(row["source_trial_index_zero_based"]) for row in rows]
    if indices != sorted(set(indices)):
        raise ValueError("qualified trial identities must be unique and source ordered")
    fields = ("monkey", "date_time", "bin_size", "trial_id", "result", "tgtDir",
              "idx_movement_on", "M1_spikes", "M1_unit_guide", "M1_lfp",
              "M1_lfp_guide", "vel")
    trials = np.empty((1, len(rows)), dtype=[(name, object) for name in fields])
    lfp_guide = unit_guide = None
    logical_bytes = 0
    with M1Session(source, *selected) as session:
        for column, row in enumerate(rows):
            source_index = int(row["source_trial_index_zero_based"])
            record = session.window(source_index, -4, 14, spikes=True)
            meta = record["metadata"]
            if meta["trial_id"] != row["trial_id"] or meta["result"] != "R":
                raise ValueError("selected source identity no longer matches qualified table")
            for event in ("idx_startTime", "idx_tgtOnTime", "idx_goCueTime", "idx_movement_on", "idx_endTime"):
                if meta[event] != int(row[event]):
                    raise ValueError("selected source event differs from qualified table")
            for name in ("lfp", "velocity", "spikes"):
                values = record[name]
                if values.shape[0] != 19 or not np.isfinite(values).all():
                    raise ValueError(f"nonfinite or incomplete {name} window; no trial is silently dropped")
                logical_bytes += values.nbytes
            if np.any(record["spikes"] < 0):
                raise ValueError("negative released spike value cannot receive reference square root")
            if lfp_guide is None:
                lfp_guide, unit_guide = record["lfp_guide"], record["unit_guide"]
            elif not (np.array_equal(lfp_guide, record["lfp_guide"]) and
                      np.array_equal(unit_guide, record["unit_guide"])):
                raise ValueError("reference first-trial guide assumption fails within session")
            values = {
                "monkey": selected[0], "date_time": session.text(source_index, "date_time"),
                "bin_size": 0.03, "trial_id": float(source_index + 1), "result": "R",
                "tgtDir": meta["direction_radians"], "idx_movement_on": 5.0,
                "M1_spikes": record["spikes"], "M1_unit_guide": unit_guide,
                "M1_lfp": record["lfp"], "M1_lfp_guide": lfp_guide,
                "vel": record["velocity"],
            }
            for name, value in values.items():
                trials[name][0, column] = value
        source_trial_count = session.count
    # Reference velocity code uses physical channel IDs as within-band positions.
    # Verify that its literal indexing actually selects those physical channels.
    channels = np.intersect1d(unit_guide[:, 0], lfp_guide[:, 0])
    if (not len(channels) or not np.isfinite(channels).all()
            or np.any(channels < 1) or np.any(channels != np.floor(channels))):
        raise ValueError("reference matched-channel IDs must be finite positive one-based integers")
    bands = np.unique(lfp_guide[:, 2])
    if len(bands) != 9 or unit_guide.shape[0] < 10:
        raise ValueError("paper nine bands / ten latent dimensions unsupported")
    for band in bands:
        positions = np.flatnonzero(lfp_guide[:, 2] == band)
        if np.max(channels) > len(positions):
            raise ValueError("reference channel-as-position indexing is out of range")
        if not np.array_equal(lfp_guide[positions[channels.astype(int) - 1], 0], channels):
            raise ValueError("reference channel-as-position indexing disagrees with physical guide")
    savemat(destination, {"trial_data": trials}, do_compression=False, oned_as="row")
    return {"animal": selected[0], "implant": selected[1], "session_id": session_id,
            "source_trial_count": source_trial_count, "selected_trial_count": len(rows),
            "trial_ids": [row["trial_id"] for row in rows],
            "source_trial_indices_zero_based": indices, "matched_electrodes": channels.tolist(),
            "upper_band_hz": bands.tolist(), "logical_selected_array_bytes": logical_bytes,
            "transient_mat_bytes": destination.stat().st_size,
            "reference_positional_channel_rule_equals_physical_identity": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trial-table", type=Path, required=True)
    parser.add_argument("--session", choices=[f"{s[0]}_{s[2]}" for s in M1_SESSIONS], required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matlab", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    args.out = args.out.resolve()
    args.work = args.work.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before_blocks = resource.getrusage(resource.RUSAGE_SELF).ru_inblock
    report = {"analysis_id": "publication_reproduction", "status": "started",
              "session_id": args.session, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "analysis_revision": ANALYSIS_REVISION,
              "trialdata_compatibility_revision": TRIALDATA_REVISION,
              "trialdata_is_known_author_runtime_revision": False,
              "reference_functions_modified": False, "primary_search_started": False,
              "scientific_trial_count_increment": 0, "whole_paper_reproduction_established": False,
              "source_role": "exposed_infrastructure_reproduction",
              "pca_scope": "all_selected_trials_as_in_publication_not_primary_training_only",
              "held_future_source_accessed": False, "other_arrays_accessed": False,
              "publication_window_ticks": [-4, 14], "publication_folds_per_representation": 50,
              "publication_past_lags_ticks": [1, 2, 3], "polynomial_prediction": "Y_only_reference_quirk"}
    try:
        analysis = acquire_code(args.work / "analysis", "BeNeuroLab/Relationship_between_latent_dynamics_and_LFP", ANALYSIS_REVISION)
        trialdata = acquire_code(args.work / "trialdata", "NeuralAnalysis/TrialData", TRIALDATA_REVISION)
        subset = args.work / "selected_m1_windows.mat"
        report["selected_source"] = export_selected_windows(args.source, args.trial_table, args.session, subset)
        seed = 20261003 + [f"{s[0]}_{s[2]}" for s in M1_SESSIONS].index(args.session)
        config = {"analysis_root": str(analysis), "trialdata_root": str(trialdata),
                  "input_mat": str(subset), "output_dir": str(args.out), "seed": seed,
                  "session_id": args.session}
        config_path = args.work / "reference_config.json"
        write_json(config_path, config)
        env = dict(os.environ)
        env.update(EP05_REFERENCE_CONFIG=str(config_path), OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        # Paths are passed via an environment variable, never MATLAB code interpolation.
        with (args.out / "reference_stdout.log").open("x") as stream:
            run = subprocess.run([str(args.matlab), "-singleCompThread", "-batch",
                                  "ep05_reference_driver"], cwd=Path(__file__).resolve().parent,
                                 env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1500)
        report["matlab_returncode"] = run.returncode
        if run.returncode:
            raise RuntimeError("reference MATLAB execution failed; inspect retained stdout")
        with (args.out / "reference_result.json").open() as stream:
            report["reference_result"] = json.load(stream)
        report["status"] = "reference_session_completed"
        report["seed"] = seed
        report["rng_boundary"] = "fixed_for_new_reproduction_original_paper_draws_unarchived"
    except Exception as error:
        report["status"] = "technical_failure"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["python_ru_inblock_delta_cache_dependent_not_physical_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_inblock - before_blocks)
        write_json(args.out / "reproduction_report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "session_id", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
