#!/usr/bin/env python3
"""Exact three-tier EP18 metadata role proposal using integer MILP feasibility.

Objective values are Fractions: minimax squared standardized imbalance, sum of
squared standardized imbalances, then the lexicographically smallest audit ID
tuple. Squaring the nonnegative minimax preserves its ordering. No weighted
objective or approximate objective-equality tolerance is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class Feature:
    name: str
    group: str
    values: tuple
    variance: Fraction
    domain: tuple

    def square(self, count, audit_n):
        n = len(self.values)
        return Fraction((n*count-audit_n*sum(self.values))**2,
                        (audit_n*(n-audit_n))**2)/self.variance


def features_from_rows(rows, quartiles, quota, audit_n):
    ids = [row["participant_id"] for row in rows]
    n = len(rows)
    ages = tuple(int(Fraction(row["age_source"])) for row in rows)
    if any(Fraction(row["age_source"]).denominator != 1 for row in rows):
        raise ValueError("This frozen implementation requires the observed integer ages")
    age_var = Fraction(n*sum(value*value for value in ages)-sum(ages)**2, n*(n-1))
    features, dropped = [], []
    if age_var:
        domain = tuple(range(sum(sorted(ages)[:audit_n]), sum(sorted(ages)[-audit_n:])+1))
        features.append(Feature("age", "age", ages, age_var, domain))
    else:
        dropped.append("age")
    for field in ("gender", "native_english", "language_profile", "language_spoken_at_home"):
        for level in sorted({row[field] for row in rows}):
            values = tuple(int(row[field] == level) for row in rows)
            total = sum(values)
            variance = Fraction(total*(n-total), n*n)
            name = field+"="+level
            if not variance:
                dropped.append(name)
                continue
            domain = tuple(range(max(0, total-(n-audit_n)), min(total, audit_n)+1))
            features.append(Feature(name, field, values, variance, domain))
    constraints = []
    for index, group in enumerate(quartiles):
        values = tuple(int(participant in group) for participant in ids)
        total = sum(values)
        variance = Fraction(total*(n-total), n*n)
        features.append(Feature(f"acquisition_quartile={index+1}", "acquisition_quartile", values, variance, (quota,)))
        constraints.append((values, quota, quota))
    return ids, features, constraints, dropped


class ExactAssignment:
    def __init__(self, ids, features, constraints, audit_n, wall_seconds):
        self.ids, self.features, self.constraints = ids, features, list(constraints)
        self.audit_n = audit_n
        self.deadline = time.monotonic()+wall_seconds
        self.calls, self.certificates = 0, []
        self.constraints.append((tuple([1]*len(ids)), audit_n, audit_n))

    def remaining(self):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Exact optimizer wall bound reached; no role proposal is certified")
        return remaining

    def feasible(self, domains, fixed=None, label=""):
        self.remaining()
        constraints = list(self.constraints)
        for feature, domain in zip(self.features, domains):
            if not domain:
                self.certificates.append({"label": label, "status": "exact_integer_domain_empty",
                                          "feature": feature.name, "feasible": False})
                return None
            # A convex quadratic sublevel has a contiguous integer interval.
            if tuple(range(domain[0], domain[-1]+1)) != tuple(domain):
                raise ValueError("Noncontiguous integer feature domain")
            constraints.append((feature.values, domain[0], domain[-1]))
        for index, value in (fixed or {}).items():
            row = tuple(int(column == index) for column in range(len(self.ids)))
            constraints.append((row, value, value))
        matrix = np.asarray([row for row, _, _ in constraints], dtype=float)
        lower = np.asarray([low for _, low, _ in constraints], dtype=float)
        upper = np.asarray([high for _, _, high in constraints], dtype=float)
        result = milp(np.zeros(len(self.ids)), integrality=np.ones(len(self.ids)),
                      bounds=Bounds(np.zeros(len(self.ids)), np.ones(len(self.ids))),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": self.remaining(), "mip_rel_gap": 0.0, "threads": 1})
        self.calls += 1
        certificate = {"label": label, "status": int(result.status), "message": result.message,
                       "primal_objective": None if result.fun is None else float(result.fun),
                       "dual_bound": None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound),
                       "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)}
        self.certificates.append(certificate)
        if result.status == 2:
            return None
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"Uncertified MILP solve: {certificate}")
        gap = getattr(result, "mip_gap", None)
        if gap is not None and gap != 0:
            raise RuntimeError("MILP returned a nonzero gap")
        vector = tuple(int(round(value)) for value in result.x)
        if any(value not in (0, 1) for value in vector):
            raise RuntimeError("MILP returned nonbinary role membership")
        # Check feasibility in exact integer arithmetic, including every fixed bit.
        for coefficients, low, high in constraints:
            actual = sum(int(coefficient)*value for coefficient, value in zip(coefficients, vector))
            if not int(low) <= actual <= int(high):
                raise RuntimeError("MILP candidate fails exact integer constraint check")
        return vector

    def solve(self):
        levels = sorted({feature.square(count, self.audit_n) for feature in self.features for count in feature.domain})

        def domains_at(level):
            return [tuple(count for count in feature.domain if feature.square(count, self.audit_n) <= level)
                    for feature in self.features]

        low, high = -1, len(levels)-1
        if self.feasible(domains_at(levels[high]), label="minimax_initial_upper") is None:
            raise ValueError("No role assignment satisfies the fixed cohort/quartile/pair constraints")
        while high-low > 1:
            middle = (high+low)//2
            if self.feasible(domains_at(levels[middle]), label=f"minimax_level_{middle}") is None:
                low = middle
            else:
                high = middle
        optimum = levels[high]
        domains = domains_at(optimum)
        grouped = defaultdict(list)
        for index, feature in enumerate(self.features):
            grouped[feature.group].append(index)
        group_states = []
        for group, indices in grouped.items():
            states = []
            for counts in itertools.product(*(domains[index] for index in indices)):
                self.remaining()
                if group != "age" and sum(counts) != self.audit_n:
                    continue
                states.append(dict(zip(indices, counts)))
            # A dropped constant categorical family leaves no group and thus no
            # spurious sum constraint. Every retained family includes all levels.
            group_states.append(states)
        aggregates = []
        for pieces in itertools.product(*group_states):
            self.remaining()
            mapping = {index: count for piece in pieces for index, count in piece.items()}
            counts = tuple(mapping[index] for index in range(len(self.features)))
            objective = sum((feature.square(count, self.audit_n)
                             for feature, count in zip(self.features, counts)), Fraction())
            aggregates.append((objective, counts))
        aggregates.sort()
        tied, sum_optimum, rejected = [], None, 0
        for objective, counts in aggregates:
            if sum_optimum is not None and objective > sum_optimum:
                break
            point_domains = [(count,) for count in counts]
            vector = self.feasible(point_domains, label=f"sumsq_aggregate_{rejected+len(tied)}")
            if vector is None:
                rejected += 1
            else:
                sum_optimum = objective
                tied.append(counts)
        if not tied:
            raise RuntimeError("Minimax feasible but no aggregate state remained")
        fixed, lex_steps = {}, []
        for index, participant in enumerate(self.ids):
            self.remaining()
            candidate = dict(fixed)
            candidate[index] = 1
            witness = None
            for state in tied:
                witness = self.feasible([(count,) for count in state], candidate,
                                        label=f"lex_prefer_{participant}")
                if witness is not None:
                    break
            fixed[index] = 1 if witness is not None else 0
            lex_steps.append({"participant": participant, "audit": fixed[index],
                              "rule": "prefer inclusion of smallest remaining ID if any exact optimum remains feasible"})
        selected = tuple(fixed[index] for index in range(len(self.ids)))
        actual_counts = [sum(value*bit for value, bit in zip(feature.values, selected)) for feature in self.features]
        actual_objectives = [feature.square(count, self.audit_n) for feature, count in zip(self.features, actual_counts)]
        if max(actual_objectives) != optimum or sum(actual_objectives, Fraction()) != sum_optimum:
            raise AssertionError("Final role vector does not have both exact optimal objectives")
        final = self.feasible([(count,) for count in actual_counts], fixed, label="final_exact_role_vector")
        if final != selected:
            raise AssertionError("Final vector is not certified feasible")
        return {"audit_proposal": [participant for participant, bit in zip(self.ids, selected) if bit],
                "development_proposal": [participant for participant, bit in zip(self.ids, selected) if not bit],
                "minimax_squared_exact": str(optimum), "minimax_absolute": math.sqrt(float(optimum)),
                "minimax_previous_infeasible_squared_exact": str(levels[low]) if low >= 0 else None,
                "sumsq_exact": str(sum_optimum), "sumsq_lower_bound_exact": str(sum_optimum),
                "sumsq_upper_bound_exact": str(sum_optimum),
                "aggregate_states_in_minimax_domain": len(aggregates),
                "smaller_or_tied_infeasible_aggregate_states": rejected,
                "feasible_exact_optimum_aggregate_states": len(tied),
                "feature_audit_counts": dict(zip((feature.name for feature in self.features), actual_counts)),
                "lexicographic_steps": lex_steps, "milp_calls": self.calls,
                "solver_certificates": self.certificates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-profile", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=480)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("Use the coordinator's verified compute allocation")
    profile = json.loads(args.metadata_profile.read_text())
    if profile["participants"] != 46 or profile["age"]["missing_participants"]:
        raise ValueError("Frozen complete 46-person profile is required")
    rows = sorted(profile["metadata_rows"], key=lambda row: row["participant_id"])
    ids, features, constraints, dropped = features_from_rows(rows, profile["acquisition_order_quartiles"], 4, 16)
    constraints.append((tuple(int(participant in {"sub-49", "sub-50"}) for participant in ids), 1, 1))
    started = time.monotonic()
    result = ExactAssignment(ids, features, constraints, 16, args.wall_seconds).solve()
    result.update({"analysis_id": "input_role_preparation", "phase": "exact_metadata_role_proposal",
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                   "observed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic()-started,
                   "metadata_profile": str(args.metadata_profile), "zero_variance_fields_dropped": dropped,
                   "algorithm": "exact rational objective tiers; finite aggregate enumeration plus integer MILP feasibility; lexicographic feasibility",
                   "roles_frozen": False, "audit_seal_verified": False, "eeg_or_image_values_opened": False})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"role_proposal.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"solver_certificates", "lexicographic_steps"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
