"""Run one original development person's prescribed repeated-image control.

Metadata binds before voltage access. Completed inner/directional fits survive
bounded job stops; complete people are reused and all30 precede cohort release.
This entrypoint opens neither main-session model scores nor audit data.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from production_core_direct import AllocatedBudget, atomic_json
from repeated_image_sequences import bind_participant, load_bound_participant
from repeated_image_control import run_participant, validate_recipe


EPISODE = Path(__file__).resolve().parents[2]


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("bounded configuration required")
    with path.open() as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--participant-index", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--operations", required=True)
    parser.add_argument("--rss-limit-bytes", type=int, default=6*1024**3)
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch,
        rss_limit_bytes=args.rss_limit_bytes, durable_byte_limit=1024**3,
        record_byte_limit=16*1024**2, threads=1)
    budget.require_compute_threads()
    config = read_json(args.config)
    if (config["analysis_id"] != "repeated_image_signal_recovery"
            or config["data_role"] != "development_repeated_signal_recovery_only"
            or config["audit_access"] is not False
            or config["primary_or_candidate_use"] is not False):
        raise ValueError("prescribed development repeated-control role required")
    release = read_json(EPISODE / config["data_release"])
    recipe = read_json(EPISODE / config["core_recipe"])
    validate_recipe(recipe)
    manifest = read_json(release["development_manifest"])
    people = manifest["participants"]
    if (manifest["role"] != "development" or len(people) != 30
            or people != sorted(set(people)) or not 0 <= args.participant_index < 30):
        raise ValueError("original thirty-person development roster required")
    person_id = people[args.participant_index]
    run_dir = EPISODE / config["run_root"] / person_id
    operations = Path(args.operations)
    operations.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {"participant_id": person_id, "job_id": os.environ["SLURM_JOB_ID"],
        "analysis_id": config["analysis_id"], "audit_access": False,
        "concept_or_false_scores_computed": False, "status": "started"}

    def progress(*values, **fields):
        for value in values:
            if isinstance(value, dict):
                fields.update(value)
            else:
                fields["stage"] = str(value)
        status.update(fields, elapsed_seconds=time.monotonic()-start)
        atomic_json(operations / "progress.json", status)
        budget.check()

    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if (run_dir / "complete.json").is_file():
            saved = read_json(run_dir / "complete.json")
            if (saved.get("status") != "complete_repeated_image_control"
                    or saved.get("participant_id") != person_id
                    or saved.get("binding", {}).get("recipe") != recipe):
                raise ValueError("existing completed person does not match the selected control")
            progress(status="reused_complete_person")
            return 0
        progress(stage="bind_repeated_metadata_before_voltage_access")
        bound = bind_participant(**{key: release[key] for key in
            ("development_manifest", "bids_root", "frozen_channel_order", "coefficient_record")},
            participant_id=person_id, budget_check=budget.check)
        atomic_json(operations / "metadata.json", bound.metadata_summary)
        progress(stage="prepare_repeated_voltage_intervals")
        person = load_bound_participant(bound, budget_check=budget.check)
        progress(stage="fit_complete_odd_even_control")
        result = run_participant(person, recipe=recipe, run_dir=run_dir,
                                 budget_check=budget.check, progress=progress)
        if result.get("status") != "complete_repeated_image_control":
            raise ValueError("full repeated-control completion required")
        progress(status="complete_repeated_image_control", stage="participant_complete")
        return 0
    except (TimeoutError, MemoryError) as error:
        status.update(status="bounded_stop", error=type(error).__name__,
                      elapsed_seconds=time.monotonic()-start)
        atomic_json(operations / "progress.json", status)
        return 75
    except Exception as error:
        status.update(status="failed", error=type(error).__name__,
                      detail=str(error), elapsed_seconds=time.monotonic()-start)
        atomic_json(operations / "progress.json", status)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
