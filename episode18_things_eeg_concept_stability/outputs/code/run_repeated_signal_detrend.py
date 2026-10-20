"""One finite all-person fixed detrending diagnostic; no model or primary refit."""
import argparse
import os
from pathlib import Path
import signal
import time

from production_core_direct import AllocatedBudget, atomic_json
from repeated_image_sequences import bind_participant, load_bound_participant
from repeated_signal_detrend import run_participant, PROJECTION_RECIPE, SCHEMA
from report_repeated_image_control import DEVELOPMENT_PARTICIPANTS
from run_repeated_signal_diagnostic import read_json


EPISODE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--operations", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, default=6 * 1024**3)
    args = parser.parse_args()
    budget = AllocatedBudget(deadline_epoch=args.deadline_epoch,
        rss_limit_bytes=args.rss_limit_bytes, durable_byte_limit=1024**3,
        record_byte_limit=16 * 1024**2, threads=1)
    budget.require_compute_threads()
    config = read_json(args.config)
    if (config["analysis_id"] != "repeated_signal_detrend"
            or config["data_role"] != "development_repeated_signal_diagnosis_only"
            or config["primary_or_candidate_use"] is not False
            or config["audit_access"] is not False
            or config["polynomial_degree"] != 3
            or config["template_bin_width_ms"] != 20
            or config["positive_horizon_ms"] != [0, 800]
            or config["negative_horizon_ms"] != [-800, 0]
            or config["diagnostic_identity_shift"] != 37):
        raise ValueError("fixed development-only detrending diagnostic required")
    release = read_json(EPISODE / config["data_release"])
    manifest = read_json(release["development_manifest"])
    people = tuple(manifest["participants"])
    completed = read_json(EPISODE / config["original_report"])
    if (manifest["role"] != "development" or people != DEVELOPMENT_PARTICIPANTS
            or completed["status"] != "complete_30_person_repeated_signal_diagnostic_report"
            or tuple(completed["participants"]) != people
            or completed["audit_access"] is not False
            or completed["concept_or_false_scores_computed"] is not False):
        raise ValueError("original complete released thirty-person diagnostic required")
    operations = Path(args.operations)
    operations.mkdir(parents=True, exist_ok=False)
    root = EPISODE / config["run_root"]
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    status = {"analysis_id": config["analysis_id"], "job_id": os.environ["SLURM_JOB_ID"],
        "status": "started", "completed_people": 0, "audit_access": False,
        "concept_or_false_scores_computed": False}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.monotonic() - start)
        atomic_json(operations / "progress.json", status)
        budget.check()

    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")

    def valid_result(row, person_id):
        return (row.get("status") == "complete_repeated_signal_detrend"
            and row.get("schema") == SCHEMA and row.get("participant_id") == person_id
            and row.get("projection_recipe") == PROJECTION_RECIPE
            and row.get("audit_access") is False
            and row.get("concept_or_false_scores_computed") is False)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, person_id in enumerate(people):
            budget.check()
            directory = root / person_id
            directory.mkdir(exist_ok=True)
            complete_path = directory / "complete.json"
            progress(participant_id=person_id, stage="bind_detrend_person")
            if complete_path.is_file():
                saved = read_json(complete_path)
                if not valid_result(saved, person_id) or saved.get("configuration") != config:
                    raise ValueError("saved detrending person differs")
                progress(stage="reused_complete_person", completed_people=index + 1)
                continue
            bound = bind_participant(**{key: release[key] for key in
                ("development_manifest", "bids_root", "frozen_channel_order", "coefficient_record")},
                participant_id=person_id, budget_check=budget.check)
            atomic_json(operations / (person_id + "_metadata.json"), bound.metadata_summary)
            progress(stage="load_original_repeated_sequences")
            person = load_bound_participant(bound, budget_check=budget.check)
            progress(stage="fixed_degree3_positive_and_negative_templates")
            result = run_participant(person, budget_check=budget.check)
            if not valid_result(result, person_id):
                raise ValueError("diagnostic returned inconsistent person or data role")
            result.update(configuration=config, job_id=os.environ["SLURM_JOB_ID"])
            budget.check()
            atomic_json(complete_path, result)
            del person, bound, result
            progress(stage="participant_complete", completed_people=index + 1)
        progress(status="complete_all30_detrend", stage="all30_complete")
        return 0
    except (TimeoutError, MemoryError) as error:
        status.update(status="bounded_stop", error=type(error).__name__)
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
