"""Refit only the sixteen changed directions from the one frozen selector."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import signal
import time

from production_core_direct import AllocatedBudget, atomic_json
from repeated_image_control import validate_recipe
from repeated_image_sequences import bind_participant, load_bound_participant
import report_repeated_selector_sensitivity as selector
from repeated_selector_refit import (SELECTOR_PATH, REFIT_ROOT, run_participant,
    validate_completed_person, validate_selection_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--operations", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, default=600 * 1024**2)
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch,
        rss_limit_bytes=args.rss_limit_bytes, durable_byte_limit=128 * 1024**2,
        record_byte_limit=16 * 1024**2, threads=1)
    budget.require_compute_threads()
    selector.require(Path(args.selector).resolve() == SELECTOR_PATH.resolve()
        and Path(args.output_root).resolve() == REFIT_ROOT.resolve(),
        "single already frozen selector and separate fixed refit root required")
    recipe = validate_recipe(selector._read(selector.RECIPE_PATH))
    frozen = selector._read(SELECTOR_PATH)
    selection = validate_selection_report(frozen, recipe=recipe)
    changed_people = tuple(person for person in selector.PEOPLE
        if any(row["selection_changed"] for row in selection[person]["directions"]))
    selector.require(len(changed_people) == 14 and frozen["changed_direction_count"] == 16,
        "the single frozen sixteen-direction change set is required")
    release = selector._read(selector.EPISODE / "outputs/config/development_eeg_release_20261012_001.json")
    manifest = selector._read(Path(release["development_manifest"]))
    selector.require(manifest["role"] == "development" and tuple(manifest["participants"]) == selector.PEOPLE,
        "unchanged original thirty-person development release required")
    operations = Path(args.operations)
    operations.mkdir(parents=True, exist_ok=False)
    REFIT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {"status": "started", "job_id": os.environ["SLURM_JOB_ID"],
        "changed_people": list(changed_people), "changed_direction_count": 16,
        "completed_changed_people": 0, "audit_access": False, "inner_retuning_performed": False}

    def progress(*messages, **fields):
        for message in messages:
            fields.update(message if isinstance(message, dict) else {"stage": str(message)})
        status.update(fields, elapsed_seconds=time.monotonic() - started)
        atomic_json(operations / "progress.json", status)
        budget.check()

    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, person_id in enumerate(changed_people):
            budget.check()
            original = selector._read(selector.INPUT_ROOT / person_id / "complete.json")
            directory = REFIT_ROOT / person_id
            complete_path = directory / "complete.json"
            progress(participant_id=person_id, stage="bind_original_changed_person")
            if complete_path.is_file():
                validate_completed_person(selector._read(complete_path),
                    selection_record=selection[person_id], original_control=original)
                progress(stage="reused_complete_person", completed_changed_people=index + 1)
                continue
            bound = bind_participant(**{key: release[key] for key in
                ("development_manifest", "bids_root", "frozen_channel_order", "coefficient_record")},
                participant_id=person_id, budget_check=budget.check)
            progress(stage="load_original_repeated_sequences")
            person = load_bound_participant(bound, budget_check=budget.check)
            run_participant(person, selection_record=selection[person_id], original_control=original,
                recipe=recipe, run_dir=directory, budget_check=budget.check, progress=progress)
            del person, bound, original
            gc.collect()
            progress(stage="changed_person_complete", completed_changed_people=index + 1)
        progress(status="complete_all_frozen_changed_directions", stage="all_sixteen_complete")
        return 0
    except (TimeoutError, MemoryError) as error:
        status.update(status="bounded_stop", error=type(error).__name__, detail=str(error))
        atomic_json(operations / "progress.json", status)
        return 75
    except Exception as error:
        status.update(status="failed", error=type(error).__name__, detail=str(error))
        atomic_json(operations / "progress.json", status)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
