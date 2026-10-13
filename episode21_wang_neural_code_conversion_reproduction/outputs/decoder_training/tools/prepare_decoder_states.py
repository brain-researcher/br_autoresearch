"""Prepare all four missing EP21 participant states in one bounded parent job.

Unactivated: requires the separate full-stage authority and complete work
manifest. No submission, renewal, retry, partial-state replay or decoder fit.
Completed sub01 state is referenced by the full manifest and never rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import time

import decoder_execution_contract as contract
import participant_training_state as state

prefix = state.prefix
require, Stop = state.require, state.Stop
OUTPUT = contract.ROOT / "state_preparation_001"


def run(args):
    require(Path.cwd().resolve() == prefix.BASE, "exact_episode_working_directory_required")
    authority = state.authorize_full(args.authorization)
    require(Path(args.manifest).absolute() == contract.MANIFEST and not contract.MANIFEST.is_symlink(),
            "fixed_full_stage_work_manifest_required")
    manifest = prefix.read_json(contract.MANIFEST)
    contract.validate_work_manifest(manifest, authority)
    resources = authority["state_build_resources"]
    # One scheduler admission for the whole four-participant parent.
    allocation = contract.verify_allocation(resources)
    contract.admit_registered_allocation(authority, manifest, "states", allocation)
    budget = contract.WorkBudget(allocation, resources, args.max_seconds)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), "state_parent_no_replay_or_overwrite")
    require(all(not state.state_directory(participant).exists()
                and not state.state_directory(participant).is_symlink()
                for participant in state.NEW_PARTICIPANTS), "all_four_new_states_must_be_fresh")
    OUTPUT.mkdir(mode=0o700)
    record = {"schema_version": "ep21.decoder_state_parent.v1", "status": "admitted_pending",
              "authorization": str(state.FULL_AUTH), "work_manifest": str(contract.MANIFEST),
              "allocation": allocation, "resources": resources, "max_work_seconds": budget.seconds,
              "participants_requested": list(state.NEW_PARTICIPANTS), "participants_completed": [],
              "participants_remaining": list(state.NEW_PARTICIPANTS), "participant_states": [],
              "training_images_each": 4712, "training_role": "subject_unique",
              "sub01_state_reused_unchanged": True, "sub01_state_recomputed": False,
              "regular_feature_values_read": False, "tau_train_values_read": False,
              "tau_test_values_read": False, "OOD_access": False,
              "predictions_or_scores_computed": False, "decoder_units_fitted": 0,
              "complete_decoder": False, "automatic_retry": False, "successor_authorized": False,
              "original_budget_consumption_preserved": True}
    prefix.publish_json(OUTPUT / "state_parent_request.json", record, budget)
    error = None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    def stop(signum, frame):
        raise Stop("state_parent_deadline_reached" if signum == signal.SIGALRM else "termination_requested")
    for sig in previous:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, budget.deadline - time.monotonic()))
    try:
        for participant in state.NEW_PARTICIPANTS:
            budget.check()
            record["active_participant"] = participant
            binding = state.prepare_state(participant, args.authorization, budget)
            record["participant_states"].append({
                "participant": participant, "directory": str(binding.directory),
                "state_status": str(binding.directory / "state_status.json"),
                "training_images": len(binding.image_ids), "training_repetitions": binding.training_repetitions,
                "native_voxels": binding.native_count, "eligible_voxels": binding.eligible_count})
            record["participants_completed"].append(participant)
            record["participants_remaining"] = [p for p in state.NEW_PARTICIPANTS
                                                if p not in record["participants_completed"]]
            print(json.dumps({"participant_state_complete": participant,
                              "completed_states": len(record["participants_completed"])}), flush=True)
        require(record["participants_completed"] == list(state.NEW_PARTICIPANTS),
                "all_four_participant_states_required")
        record["status"] = "complete_four_participant_training_states_only"
        record.pop("active_participant", None)
    except Exception as exc:
        error = exc
        record.update(status="participant_state_parent_stopped", error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        record["elapsed_seconds"] = time.monotonic() - budget.started
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["logical_neural_input_bytes"] = budget.logical_read_bytes
        record["durable_bytes_before_parent_status"] = sum(budget.files.values())
        record["accounting"] = {"charge_parent_allocation_once": True,
                                "maximum_CPU_seconds": resources["minutes"] * 60,
                                "state_build_jobs_consumed": 1,
                                "prior_consumption_preserved": True, "terminal_accounting_by_coordinator": True}
        # Individual completed state records remain authoritative if a hard
        # external stop prevents this parent closeout. Never infer a restart.
        prefix.publish_json(OUTPUT / "state_parent_status.json", record, budget, closing=True)
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "state_parent_not_completed", "error_type": type(exc).__name__,
                          "error": str(exc), "no_retry_or_successor": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
