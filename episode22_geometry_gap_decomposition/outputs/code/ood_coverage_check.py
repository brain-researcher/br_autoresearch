#!/usr/bin/env python3
"""EP22 outcome-blind OOD coverage check: trial metadata and embeddings only, no beta value."""

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import ood_confirmation as oc


def main(out_path: Path) -> int:
    summary = json.loads((oc.EP04 / oc.METADATA / "source_metadata_report.json").read_text())
    arrays = json.loads((oc.EP04 / oc.ARRAYS).read_text())
    table = list(csv.DictReader(open(oc.CATEGORY_TABLE)))
    ood = sorted(r["image_id"] for r in table)
    index = {n: i for i, n in enumerate(ood)}
    dev_trials = Counter()
    with open(oc.EP04 / oc.METADATA / "development_trials.jsonl") as f:
        for line in f:
            r = json.loads(line)
            dev_trials[(r["participant"], r["session"])] += 1
    frozen_ood = {r["participant"]: r["trials"] for r in summary["trial_role_counts"] if r["role"] == "conditional_ood"}
    report = {"analysis_id": "ood_coverage_check", "created_utc": datetime.now(timezone.utc).isoformat(),
              "beta_values_read": False, "ood_images": len(ood), "participants": {}}
    ok = len(ood) == 371
    for p in oc.SETTINGS["participants"]:
        per_image, sessions_with_ood, problems = Counter(), 0, []
        for rec in sorted((r for r in summary["trial_files"] if r["participant"] == p), key=lambda r: r["session"]):
            with open(oc.SOURCE_ROOT / "dataset" / rec["trial_file"]) as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            beta = [b for b in arrays["betas"] if (b["participant"], b["session"]) == (p, rec["session"])]
            if len(rows) != rec["rows"] or len(beta) != 1 or beta[0]["header"]["shape"][3] != len(rows):
                problems.append(f"{rec['session']}: TSV rows / NIfTI axis mismatch")
            hits = [r["label"] for r in rows if r["label"] in index]
            per_image.update(hits)
            if hits:
                sessions_with_ood += 1
                if dev_trials[(p, rec["session"])] < 2:
                    problems.append(f"{rec['session']}: fewer than two development trials for standardization")
        hist = Counter(per_image[n] for n in ood)
        below_two = [n for n in ood if per_image[n] < 2]
        total = sum(per_image.values())
        entry = {"ood_presentations": total, "frozen_accounting_ood_trials": frozen_ood.get(p),
                 "matches_frozen_accounting": total == frozen_ood.get(p),
                 "presentations_per_image_histogram": {str(k): v for k, v in sorted(hist.items())},
                 "images_below_two_presentations": below_two,
                 "images_with_three_presentations": [n for n in ood if per_image[n] == 3],
                 "sessions_with_ood_trials": sessions_with_ood, "problems": problems}
        ok = ok and entry["matches_frozen_accounting"] and not below_two and not problems
        report["participants"][p] = entry
    report["coverage_ok"] = bool(ok)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({"coverage_ok": report["coverage_ok"],
                      "ood_presentations": {p: e["ood_presentations"] for p, e in report["participants"].items()}}))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
