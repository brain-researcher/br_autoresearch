"""Freeze and replay the authorized fixed EP13 package inside root-owned Slurm."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket

from target_certificate import evaluate, fixed_suite_checks, replay_certificate


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("orientation", "independent", "all"), default="all")
    args = parser.parse_args()
    hostname, job_id = socket.gethostname(), os.environ.get("SLURM_JOB_ID")
    if "-ln" in hostname or not job_id:
        parser.error("exact replay must run in the coordinator's verified Slurm allocation")
    source = Path(__file__).resolve().parent
    files = ["proof.md", "fixtures.json", "target_certificate.py", "independent_checker.py", "run_exact.py"]
    revisions = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}
    base = {"utc": datetime.now(timezone.utc).isoformat(), "hostname": hostname,
            "slurm_job_id": job_id, "python": platform.python_version(),
            "evidence_role": "exact implementation/orientation audit; universal proof is proof.md"}
    cases = json.loads((source / "fixtures.json").read_text())
    if args.phase in ("orientation", "all"):
        args.run_dir.mkdir(parents=True, exist_ok=False)
        save(args.run_dir / "package_lock.json", {**base, "source_directory": str(source), "sha256": revisions,
             "independent_checker_inputs": "fixture tuples only; no expected labels or main certificates"})
        own = [evaluate(case) for case in cases]
        coverage = fixed_suite_checks(cases, own)
        save(args.run_dir / "orientation_report.json", {**base, "analysis_id": "exact_orientation_suite",
             "coverage": coverage, "cases": own, "passed": True})
        if args.phase == "orientation":
            print(json.dumps({"analysis_id": "exact_orientation_suite", "passed": True, "run_dir": str(args.run_dir)}))
            return
    else:
        locked = json.loads((args.run_dir / "package_lock.json").read_text())
        assert locked["sha256"] == revisions, "package changed after lock; preserve attempt and start a repaired one"
        orientation = json.loads((args.run_dir / "orientation_report.json").read_text())
        assert orientation["passed"]
        own, coverage = orientation["cases"], orientation["coverage"]
    spec = importlib.util.spec_from_file_location("ep13_independent_checker", source / "independent_checker.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    other = []
    for case, expected in zip(cases, own):
        # A fresh JSON roundtrip isolates the checker's inputs from main results.
        result = checker.check_case(json.loads(json.dumps(case)))
        assert result["id"] == expected["id"] and result["valid"] == expected["valid"]
        if result["valid"]:
            assert result["identifiable"] == expected["identifiable"]
            assert result["Y"] == expected["Y"] and result["tau"] == expected["tau"]
            assert result["observation_identity"] and result["target_identity"] and result["certificate_valid"]
            assert replay_certificate(case, result["certificate"])
        other.append(result)
    assert revisions == {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}
    save(args.run_dir / "independent_replay_report.json", {**base,
         "analysis_id": "independent_certificate_replay", "cases": other,
         "all_input_decisions_and_exact_values_agree": True,
         "independent_certificates_replayed": True, "passed": True})
    save(args.run_dir / "completion.json", {**base, "local_outcome": "theorem_verified",
         "universal_evidence": str(source / "proof.md"), "valid_cases": coverage["valid_cases"],
         "invalid_inputs_rejected": coverage["rejected_inputs"],
         "canonical_acceptance": False, "new_empirical_or_biological_result": False})
    print(json.dumps({"local_outcome": "theorem_verified", "run_dir": str(args.run_dir),
                      "valid_cases": coverage["valid_cases"], "rejected_inputs": coverage["rejected_inputs"]}))


if __name__ == "__main__":
    main()
