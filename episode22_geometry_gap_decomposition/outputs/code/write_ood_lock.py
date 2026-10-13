#!/usr/bin/env python3
"""Write EP22's write-once OOD configuration lock (after validation passes)."""

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import ood_confirmation as oc

VALIDATION = oc.EP22 / "outputs/runs/ood_evaluator_validation/20261012-attempt02"


def main() -> int:
    preview = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # draft only; never the real lock path
    if preview is None and oc.LOCK.exists():
        raise SystemExit("lock already exists; it is write-once")
    review = json.loads(oc.EXPOSURE_REVIEW.read_text())
    coverage = json.loads((VALIDATION / "ood_coverage_check.json").read_text())
    dev = json.loads((VALIDATION / "dev_test_report.json").read_text())
    tests = (VALIDATION / "synthetic_tests.txt").read_text()
    reader = dev["neural"]["sub-03"]["reader"]["dev_reader_verification"]
    checks = {
        "exposure_review_pass": review["conclusion"] == "pass",
        "coverage_ok": coverage["coverage_ok"],
        "synthetic_tests_all_pass": tests.count("PASS") == 4 and "Error" not in tests,
        "dev_reader_bitwise_identical": all(v["bitwise_identical_to_EP04_handoff"] for v in reader.values()),
        "dev_test_code_matches_current": dev["code_sha256"] == oc.code_checksums(),
        "dev_test_settings_match_current": dev["settings"] == oc.SETTINGS,
    }
    if not all(checks.values()):
        raise SystemExit(f"validation incomplete: {checks}")
    # C2b retention rule (settings 'c2b_retention'): confirmatory only if the dev-test met the C2b rule for both VLMs.
    dev_c2b = dev["evaluation"]["C2"]["C2b_all_pairs"]
    c2b_confirmatory = all(dev_c2b[m]["passed"] is True for m in ("CLIP", "SigLIP2"))
    lock = {
        "c2b_confirmatory": c2b_confirmatory,
        "c2b_retention_basis": {m: {k: dev_c2b[m][k] for k in ("native_gap_mean", "swapped_gap_mean", "closure_mean",
                                                             "participants_meeting_rule", "evaluable", "passed")}
                                for m in ("CLIP", "SigLIP2")},
        "lock": "ep22_ood_confirmation", "write_once": True,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "status": "locked_pending_scientist_opening_approval",
        "code_sha256": oc.code_checksums(), "settings": oc.SETTINGS,
        "adopted_design_record": "outputs/experiment_log.md#2026-10-12-2255-utc--confirmation-design--scientist-adopted-the-restated-predictions-exposure-review-authorized",
        "separation": "procedural, same account (adopted 2026-10-12)",
        "inputs": {
            "category_table": str(oc.CATEGORY_TABLE), "category_table_sha256": oc.sha256(oc.CATEGORY_TABLE),
            "corrected_embeddings_download_record": json.loads((oc.CORRECTED / "download_record.json").read_text()),
            "c3_alpha_source": str(oc.D2_CELLS),
        },
        "pre_lock_evidence": {
            "exposure_review": str(oc.EXPOSURE_REVIEW), "exposure_review_sha256": oc.sha256(oc.EXPOSURE_REVIEW),
            "coverage_check": str(VALIDATION / "ood_coverage_check.json"),
            "dev_test_report": str(VALIDATION / "dev_test_report.json"),
            "synthetic_tests": str(VALIDATION / "synthetic_tests.txt"),
            "checks": checks,
        },
        "opening_rule": "One run of ood_confirmation.py --mode confirm after the scientist's explicit approval is logged; "
                        "no partial result is released; an exact software retry is allowed only if no score was produced.",
    }
    if preview is not None:
        if preview.resolve() == oc.LOCK.resolve():
            raise SystemExit("a preview may not be written to the lock path")
        lock["status"] = "PREVIEW_NOT_A_LOCK"
        preview.write_text(json.dumps(lock, indent=1))
        print(json.dumps({"preview": str(preview), "checks": checks}))
        return 0
    with open(oc.LOCK, "x") as f:
        json.dump(lock, f, indent=1)
    os.chmod(oc.LOCK, stat.S_IRUSR | stat.S_IRGRP)
    print(json.dumps({"lock": str(oc.LOCK), "sha256": oc.sha256(oc.LOCK), "checks": checks}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
