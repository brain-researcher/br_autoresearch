#!/usr/bin/env python3
"""Profile only the fixed participant-assignment fields; do not assign roles."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
from functools import reduce
import json
import math
import os
from pathlib import Path
import socket


CATEGORICAL = ["gender", "native_english", "language_profile", "language_spoken_at_home"]
MISSING = {"", "n/a", "na", "nan", "none"}


def category(value):
    value = value.strip()
    return "__MISSING__" if value.lower() in MISSING else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    previous = json.loads(args.qualification_report.read_text())
    eligible = sorted(previous["potentially_eligible_participants"], key=lambda item: int(item.split("-")[1]))
    if len(eligible) != 46:
        raise ValueError("Expected fixed 46-person cohort")
    with (args.bids_root/"participants.tsv").open(newline="", encoding="utf-8-sig") as source:
        rows = {row["participant_id"]: row for row in csv.DictReader(source, delimiter="\t")}
    definitions = json.loads((args.bids_root/"participants.json").read_text())
    ages, missing_age, categorical = {}, [], {}
    for participant in eligible:
        raw = rows[participant]["age"].strip()
        if raw.lower() in MISSING:
            missing_age.append(participant)
        else:
            ages[participant] = Fraction(raw)
    for field in CATEGORICAL:
        counts = Counter(category(rows[participant][field]) for participant in eligible)
        categorical[field] = {"source_definition": definitions[field]["Description"],
                              "levels": dict(sorted(counts.items())),
                              "pooled_Bernoulli_SD_by_level": {
                                  level: math.sqrt((count/46)*(1-count/46)) for level, count in sorted(counts.items())},
                              "missing_count": counts.get("__MISSING__", 0)}
    denominator = math.lcm(*(value.denominator for value in ages.values())) if ages else None
    scaled = {participant: int(value*denominator) for participant, value in ages.items()} if ages else {}
    common_step = reduce(math.gcd, (abs(value-min(scaled.values())) for value in scaled.values())) if scaled else None
    age_values = [float(value) for value in ages.values()]
    mean = sum(age_values)/len(age_values) if age_values else None
    sample_sd = math.sqrt(sum((value-mean)**2 for value in age_values)/(len(age_values)-1)) if len(age_values) > 1 else None
    quartiles, start = [], 0
    for size in (12, 12, 11, 11):
        quartiles.append(eligible[start:start+size])
        start += size
    record = {"analysis_id": "input_role_preparation", "phase": "metadata_assignment_encoding_profile",
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "observed_utc": datetime.now(timezone.utc).isoformat(), "participants": 46,
              "age": {"source_definition": definitions["age"]["Description"],
                      "missing_participants": missing_age, "integer_values": denominator == 1,
                      "rational_common_denominator": denominator, "scaled_integer_resolution_gcd": common_step,
                      "range": [min(age_values), max(age_values)] if age_values else [],
                      "eligible_cohort_mean": mean, "eligible_cohort_sample_SD": sample_sd,
                      "sample_SD_formula": "sqrt(sum((age_i-mean_age)^2)/(N-1))",
                      "exact_rational_age_by_participant": {participant: str(value) for participant, value in ages.items()}},
              "categorical_fields": categorical, "acquisition_order_quartiles": quartiles,
              "proposed_encoding": {"age": "continuous; no imputation selected",
                                    "categorical": "all recorded nominal levels, exact labels after outer whitespace trim; missing is explicit category",
                                    "category_scale": "sqrt(p_j*(1-p_j)), p_j=count_j/46",
                                    "zero_variance": "drop and record, per policy",
                                    "standardized_imbalance": "(sum_audit(x_j)/16 - sum_development(x_j)/30) / eligible_cohort_scale_j",
                                    "same_quartile_indicators_enter_objective": True},
              "metadata_rows": [{"participant_id": participant, "age_source": rows[participant]["age"],
                                 **{field: category(rows[participant][field]) for field in CATEGORICAL}}
                                for participant in eligible],
              "encoding_frozen": False, "roles_assigned": False,
              "eeg_or_image_values_opened": False, "audit_seal_verified": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"role_metadata_profile.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in record.items() if key not in {"metadata_rows", "acquisition_order_quartiles", "age"}}, indent=2))
    print(json.dumps({"age": {key: value for key, value in record["age"].items() if key != "exact_rational_age_by_participant"}}, indent=2))


if __name__ == "__main__":
    main()
