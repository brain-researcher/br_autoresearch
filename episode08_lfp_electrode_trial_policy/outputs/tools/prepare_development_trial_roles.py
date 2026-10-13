#!/usr/bin/env python3
"""EP08 metadata-only development evaluation/fitting roles.

Reads only the completed source-support JSON; never opens a source MAT or
neural array. Execution materializes the coordinator-selected prospective
split, not a scoring authorization or filesystem access seal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import socket


DEVELOPMENT = (
    "Mihili_CO_20140217.mat", "Mihili_CO_20140218.mat",
    "Mihili_CO_20140303.mat", "Mihili_CO_20140304.mat",
    "Chewie_CO_20160929.mat", "Chewie_CO_20161005.mat",
    "Chewie_CO_20161006.mat", "Chewie_CO_20161007.mat",
)
ROLE_SEED_PREFIX = "ep08-development-trial-roles-v1|20261003"
EVALUATION_PER_DIRECTION = 5
FITTING_TOTALS = (32, 64, 96)


def prepare_roles(metadata):
    if metadata.get("neural_elements_read") is not False or metadata.get("scores_computed") is not False:
        raise ValueError("Use the completed outcome-blind source-support metadata")
    sessions = metadata["sessions"]
    filenames = [item["filename"] for item in sessions]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate source session identity")
    selected = {item["filename"]: item for item in sessions if item["session_role"] == "development"}
    if set(selected) != set(DEVELOPMENT):
        raise ValueError("Preserve the existing eight development sessions")

    output = []
    for filename in DEVELOPMENT:
        session = selected[filename]
        by_direction = [[] for _ in range(8)]
        seen = set()
        for trial in session["metadata_eligible_trials"]:
            index, direction = trial["trial_index_zero_based"], trial["direction_index"]
            if (type(index) is not int or not 0 <= index < session["trial_count"]
                    or index in seen or type(direction) is not int or not 0 <= direction < 8):
                raise ValueError(f"Invalid or duplicate trial identity in {filename}")
            seen.add(index)
            by_direction[direction].append(index)
        if [len(values) for values in by_direction] != session["eligible_metadata_count_by_direction"]:
            raise ValueError(f"Trial identities and declared support disagree in {filename}")

        pools, evaluation = [], []
        for direction, indices in enumerate(by_direction):
            ordered = sorted(indices)
            rng = random.Random()
            rng.seed(f"{ROLE_SEED_PREFIX}|{filename}|{direction}", version=2)
            rng.shuffle(ordered)
            if len(ordered) < EVALUATION_PER_DIRECTION + max(FITTING_TOTALS) // 8:
                raise ValueError(f"Declared evaluation and fitting support unavailable: {filename}, direction {direction}")
            evaluation.extend(ordered[:EVALUATION_PER_DIRECTION])
            pools.append(ordered[EVALUATION_PER_DIRECTION:])

        training_sets = {}
        for total in FITTING_TOTALS:
            per_direction = total // 8
            training = [index for pool in pools for index in pool[:per_direction]]
            folds = []
            for fold in range(4):
                validation = [index for pool in pools for position, index in enumerate(pool[:per_direction])
                              if position % 4 == fold]
                validation_set = set(validation)
                fitting = [index for index in training if index not in validation_set]
                folds.append({"fold_index": fold, "fitting_trials": fitting, "validation_trials": validation})
            training_sets[str(total)] = {"training_trials": training, "inner_folds": folds}
        if set(evaluation).intersection(index for pool in pools for index in pool):
            raise ValueError("Evaluation and fitting roles overlap")
        output.append({"filename": filename, "animal": session["animal"],
                       "session_role": "development", "eligible_trials": len(seen),
                       "evaluation_trials": evaluation, "fitting_pool_by_direction": pools,
                       "nested_training_sets": training_sets})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if args.output_dir.exists():
        raise FileExistsError("Preserve existing trial roles; use a new selected attempt")
    metadata = json.loads(args.metadata_report.read_text())
    sessions = prepare_roles(metadata)
    report = {"analysis_id": "source_support", "phase": "development_trial_roles",
              "recipe": "outputs/development_comparison_proposal.md",
              "source_metadata": str(args.metadata_report),
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "seed_prefix": ROLE_SEED_PREFIX, "random_algorithm": "Python3.12 random.Random seed version2; shuffle ascending indices",
              "evaluation_trials_per_direction": EVALUATION_PER_DIRECTION,
              "fitting_totals": list(FITTING_TOTALS), "sessions": sessions,
              "development_sessions": len(sessions), "evaluation_trials_total": 40 * len(sessions),
              "source_MAT_files_opened": 0, "neural_elements_read": False,
              "held_session_trial_roles_created": False, "held_session_payload_read": False,
              "scores_computed": False, "candidate_access_boundary_verified": False,
              "scientific_comparison_activated": False, "valid_evaluations_charged": 0}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "development_trial_roles.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "development_sessions": len(sessions),
                      "evaluation_trials_total": report["evaluation_trials_total"],
                      "neural_elements_read": False, "scores_computed": False}), flush=True)


if __name__ == "__main__":
    main()
