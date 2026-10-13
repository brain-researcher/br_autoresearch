"""Separate metadata-only initializer roles inside the immutable fitting whitelist."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import socket

from acquired_only_reader import CHECKPOINTS
from development_reader import selected_metadata
from prepare_development_trial_roles import DEVELOPMENT

ORDERS = (0, 1, 2)
RECIPE = "outputs/initializer_robustness_proposal.md"


def derive_role(source, original_role, records, order):
    filename = original_role["filename"]
    original = original_role["nested_training_sets"]["96"]["training_trials"]
    evaluation = original_role["evaluation_trials"]
    pools = original_role["fitting_pool_by_direction"]
    if (order not in ORDERS or filename not in DEVELOPMENT or source["filename"] != filename
            or source["session_role"] != "development" or original_role["session_role"] != "development"
            or len(original) != 96 or len(set(original)) != 96
            or len(evaluation) != 40 or len(set(evaluation)) != 40
            or set(original) & set(evaluation) or len(pools) != 8):
        raise ValueError("Exact original96 fitting/40 evaluation development roles required")
    queues, mappings = [], []
    for direction, pool in enumerate(pools):
        allowed = pool[:12]
        expected = {trial for trial in original if records[trial]["direction_index"] == direction}
        if (len(allowed) != 12 or len(set(allowed)) != 12 or set(allowed) != expected
                or any(type(trial) is not int for trial in allowed)):
            raise ValueError("Each original direction must supply exactly twelve whitelisted IDs")
        canonical = sorted(allowed)
        queue = canonical.copy()
        literal = f"ep08-initializer-robustness-order-v1|20261004|{filename}|{direction}|{order}"
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(queue)
        queues.append(queue)
        mappings.append({"direction": direction, "seed_literal": literal,
                         "canonical_fitting_trial_ids": canonical, "permuted_fitting_trial_ids": queue})
    result = copy.deepcopy(original_role)
    result["initializer_order"] = order
    result["fitting_pool_by_direction"] = queues
    result["nested_training_sets"] = {}
    for total in CHECKPOINTS:
        trials = [trial for queue in queues for trial in queue[:total//8]]
        folds = []
        for fold in range(4):
            valid = [trial for queue in queues for position, trial in enumerate(queue[:total//8])
                     if position % 4 == fold]
            folds.append({"fold_index": fold, "fitting_trials": [t for t in trials if t not in set(valid)],
                          "validation_trials": valid})
        result["nested_training_sets"][str(total)] = {"training_trials": trials, "inner_folds": folds}
    if set(result["nested_training_sets"]["96"]["training_trials"]) != set(original):
        raise ArithmeticError("New roles cannot change the original fitting whitelist")
    return result, mappings


def prepare_roles(metadata, original_roles):
    if (len(original_roles["sessions"]) != 8
            or {s["filename"] for s in original_roles["sessions"]} != set(DEVELOPMENT)):
        raise ValueError("All eight original development sessions required")
    sources = {name: selected_metadata(metadata, original_roles, name) for name in DEVELOPMENT}
    orders = []
    for order in ORDERS:
        sessions, mappings = [], []
        for filename in DEVELOPMENT:
            source, original, records = sources[filename]
            role, queues = derive_role(source, original, records, order)
            # The qualified validator checks all standard role prefixes/folds.
            selected_metadata(metadata, {"sessions": [role]}, filename)
            sessions.append(role)
            mappings.append({"filename": filename, "queues": queues})
        orders.append({"initializer_order": order, "sessions": sessions, "queue_records": mappings})
    return {"status": "completed_initializer_robustness_roles", "recipe": RECIPE,
            "initializer_orders": list(ORDERS), "orders": orders,
            "original_role_support": [{"filename": name,
                "fitting_trial_whitelist": sources[name][1]["nested_training_sets"]["96"]["training_trials"],
                "evaluation_trials": sources[name][1]["evaluation_trials"]} for name in DEVELOPMENT],
            "neural_elements_read": False, "source_MAT_files_opened": 0,
            "held_session_accessed": False, "valid_evaluations_charged": 0}


def validate_prepared(prepared, metadata, original_roles):
    expected = prepare_roles(metadata, original_roles)
    if any(prepared.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared roles must exactly preserve the frozen ordinary queues and original role boundary")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("metadata-report", "trial-roles", "output-dir"):
        parser.add_argument("--"+flag, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Coordinator-owned verified compute allocation required")
    if args.output_dir.exists():
        raise FileExistsError("Preserve old roles and select a fresh role attempt")
    report = prepare_roles(json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text()))
    report.update(source_metadata=str(args.metadata_report), original_trial_roles=str(args.trial_roles),
                  prepared_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                  slurm_job_id=os.environ["SLURM_JOB_ID"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"initializer_robustness_roles.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "neural_elements_read": False, "session_order_roles": 24}))


if __name__ == "__main__":
    main()
