"""One Slurm array element per remaining frozen development participant.

The mapping comes only from the original initialized 30-person binding and
its already released development manifest. Sub-07 has a separate live owner
and is excluded. This entrypoint never selects people using EEG or losses and
does not read any existing participant result. Root selects finite resources
and submits the array after the first complete-pair cost measurement.
"""
import argparse
import json
import os
from pathlib import Path

import production_baseline_controller as controller
import production_core_reference as core
from production_core_direct import AllocatedBudget, run_person


def selected_person(initialized_root, array_index):
    settings = controller._json(Path(initialized_root)/"settings.json", 32*1024**2)
    people = tuple(settings["participants"])
    core.require(len(people) == 30 and len(set(people)) == 30 and people == tuple(sorted(people))
        and people[0] == "sub-07", "original_complete_30_person_mapping_required")
    manifest = controller._json(Path(settings["release"]["development_manifest"]), 16*1024**2)
    core.require(manifest["role"] == "development" and tuple(manifest["participants"]) == people,
                 "mapping_must_match_frozen_released_development_role")
    remaining = tuple(person for person in people if person != "sub-07")
    core.require(type(array_index) is int and 0 <= array_index < len(remaining) == 29,
                 "one_of_29_remaining_development_indices_required")
    return remaining[array_index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialized-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--rss-limit-bytes", required=True, type=int)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
        durable_byte_limit=8*1024**3, record_byte_limit=16*1024**2, threads=args.threads)
    budget.require_compute_threads()
    core.require(bool(os.environ.get("SLURM_ARRAY_JOB_ID")), "coordinator_submitted_array_required")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    person = selected_person(args.initialized_root, index)
    print(json.dumps({"array_job_id": os.environ["SLURM_ARRAY_JOB_ID"], "array_index": index,
        "job_id": os.environ["SLURM_JOB_ID"], "participant_id": person,
        "mapping": "original_sorted_30_development_people_excluding_sub07",
        "audit_access": False}), flush=True)
    report = run_person(args.initialized_root, args.run_root, args.public_root,
        person, budget, args.chunk_rows)
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
