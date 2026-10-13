"""EP17 adopted metadata features and exact finite concept assignment math."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import hashlib
from math import lcm

import reconcile_released_events as core

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-06")
PRECISION = 80
QUANTUM = 10000
MAX_SWEEPS = 8
NAMESPACE = "ep17-source-roles-v1"
MAX_PAIR_VISITS = 3967200
MAX_CACHE_UPDATES = 2380665600
LAG_CLASSES = ("expected_first", "within_seconds", "between_days", "unresolved")


@dataclass(frozen=True)
class Concept:
    cid: int
    uid: str
    provider_id: int
    original_count: int
    qualified_count: int

    def key(self):
        return self.cid, self.uid.encode("utf-8")

    def identity(self):
        core.require(self.cid > 0 and bool(self.uid) and
                     not any(c in self.uid for c in ("\t", "\n", "\r", "\0")), "invalid_concept_identity")
        return str(self.cid).encode("ascii") + b"\t" + self.uid.encode("utf-8")


@dataclass(frozen=True)
class Field:
    name: str
    family: str
    weight: Fraction
    participant: str = ""


@dataclass
class RawFeatures:
    concepts: list[Concept]
    fields: list[Field]
    values: list[list[Decimal | None]]
    diagnostics: dict


@dataclass
class Prepared:
    concepts: list[Concept]
    fields: list[Field]
    raw: list[list[Decimal | None]]
    imputed: list[list[Decimal]]
    q: list[list[int]]
    field_details: list[dict]


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def decimal_token(token):
    if token == "":
        return None
    value = Decimal(token)
    core.require(value.is_finite() and value >= 0, "invalid_projected_delay")
    return value


def lag_class(row):
    repetition = core.integer(row["source_repetition"], "source_repetition")
    core.require(repetition > 0, "invalid_source_ordinal")
    seconds, days = decimal_token(row["delay_seconds"]), decimal_token(row["delay_days"])
    if repetition == 1 and seconds is None and days is None:
        return "expected_first", None, ""
    if repetition > 1 and seconds is not None and days is not None:
        if seconds > 0 and days == 0:
            return "within_seconds", seconds, ""
        if days > 0 and seconds == 0:
            return "between_days", days, ""
    if repetition == 1:
        reason = "delay_on_source_first_exposure"
    elif seconds is None and days is None:
        reason = "nonfirst_both_unknown"
    elif seconds is None or days is None:
        reason = "one_unknown_delay"
    elif seconds == 0 and days == 0:
        reason = "both_zero_sentinels"
    else:
        reason = "both_positive_delays"
    return "unresolved", None, reason


def rational_moments(weighted):
    coverage = sum((weight for _, weight in weighted), Fraction())
    core.require(coverage > 0, "empty_rational_moment")
    first = weighted[0][0]
    if all(value == first for value, _ in weighted):
        return first, Fraction()
    mean = sum((value * weight for value, weight in weighted), Fraction()) / coverage
    variance = sum((weight * (value - mean) ** 2 for value, weight in weighted), Fraction()) / coverage
    return mean, variance


def log_moments(weighted):
    if not weighted:
        return None, None
    values = [((Decimal(1) + value).ln(), weight) for value, weight in weighted]
    first = values[0][0]
    if all(value == first for value, _ in values):
        return first, Decimal(0)
    coverage = sum((weight for _, weight in values), Fraction())
    mean = sum((value * decimal_fraction(weight) for value, weight in values), Decimal(0)) / decimal_fraction(coverage)
    variance = sum((decimal_fraction(weight) * (value - mean) ** 2 for value, weight in values), Decimal(0)) / decimal_fraction(coverage)
    core.require(mean.is_finite() and variance.is_finite() and variance >= 0, "invalid_log_lag_moments")
    return mean, variance


def participant_features(image_rows, sessions, total_rows):
    """Exact image/repetition weights; caller supplies one concept/person."""
    core.require(bool(image_rows) and total_rows > 1 and bool(sessions), "empty_concept_participant")
    session_mass = {session: Fraction() for session in sessions}
    classes = {name: Fraction() for name in LAG_CLASSES}
    lags = {"within_seconds": [], "between_days": []}
    positions, reasons, flags = [], Counter(), Counter()
    for image in sorted(image_rows, key=lambda value: value.encode("utf-8")):
        rows = sorted(image_rows[image], key=lambda r: int(r["glmsingle_trial_index_zero_based"]))
        core.require(len(rows) >= 2, "eligible_image_without_repeats")
        weight = Fraction(1, len(image_rows) * len(rows))
        for row in rows:
            session = core.integer(row["session"], "session")
            core.require(session in session_mass, "session_outside_fixed_roster")
            session_mass[session] += weight
            rank = core.integer(row["retained_acquisition_rank"], "retained_acquisition_rank")
            core.require(0 <= rank < total_rows, "invalid_acquisition_rank")
            positions.append((Fraction(rank, total_rows - 1), weight))
            label, value, reason = lag_class(row)
            classes[label] += weight
            if value is not None:
                lags[label].append((value, weight))
            if reason:
                reasons[reason] += 1
            for flag in ("atypical", "not_for_memory"):
                core.require(row[flag] in ("True", "False", ""), "invalid_projected_schedule_flag")
                flags[f"{flag}:{row[flag] or 'unknown'}"] += 1
    core.require(sum(session_mass.values(), Fraction()) == sum(classes.values(), Fraction()) == 1,
                 "concept_participant_weights_not_one")
    mean, variance = rational_moments(positions)
    output = [decimal_fraction(session_mass[s]) for s in sessions]
    output.extend((decimal_fraction(mean), decimal_fraction(variance)))
    output.extend(decimal_fraction(classes[name]) for name in LAG_CLASSES)
    output.extend(log_moments(lags["within_seconds"]))
    output.extend(log_moments(lags["between_days"]))
    return output, reasons, flags


def build_features(concepts, taxonomy_labels, memberships, rows, sessions, totals):
    """Rows are the previously qualified public chronology projection only."""
    concepts = sorted(concepts, key=lambda c: c.key())
    core.require(len({c.cid for c in concepts}) == len(concepts), "duplicate_feature_concept")
    for concept in concepts:
        concept.identity()
    fields = [Field(f"taxonomy/{name}", "taxonomy", Fraction(1, 216)) for name in taxonomy_labels]
    fields.extend((Field("taxonomy/unmapped", "taxonomy", Fraction(1, 216)),
                   Field("exemplars/original_six", "exemplars", Fraction(1, 8)),
                   Field("exemplars/qualified_count", "exemplars", Fraction(1, 8))))
    core.require(len(taxonomy_labels) == 53 and len(set(taxonomy_labels)) == 53,
                 "expected_frozen_taxonomy_fields")
    for subject in SUBJECTS:
        core.require(sessions[subject] == sorted(set(sessions[subject])) and bool(sessions[subject]),
                     "session_field_order_changed")
        fields.extend(Field(f"{subject}/session/{s}", "acquisition", Fraction(1, 32 * len(sessions[subject])), subject)
                      for s in sessions[subject])
        fields.extend(Field(f"{subject}/acquisition/{name}", "acquisition", Fraction(1, 64), subject)
                      for name in ("mean", "variance"))
        fields.extend(Field(f"{subject}/lag/{name}", "lag", Fraction(1, 128), subject)
                      for name in (*LAG_CLASSES, "within_log_mean", "within_log_variance",
                                   "between_log_mean", "between_log_variance"))
    core.require(sum((field.weight for field in fields), Fraction()) == 1, "feature_weights_not_one")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row["eligible_image"] == "True":
            grouped[int(row["things_category_nr"])][row["participant"]][row["expected_provider_image_path"]].append(row)
    values, all_reasons, all_flags = [], Counter(), {}
    with localcontext() as context:
        context.prec, context.rounding = PRECISION, ROUND_HALF_EVEN
        for concept in concepts:
            membership = memberships[concept.cid]
            core.require(len(membership) == 53 and all(value in (0, 1) for value in membership), "invalid_taxonomy_vector")
            core.require(concept.original_count in (5, 6) and concept.qualified_count >= 1,
                         "invalid_exemplar_feature")
            values.append([*(Decimal(value) for value in membership), Decimal(not any(membership)),
                           Decimal(concept.original_count == 6), Decimal(concept.qualified_count)])
        # Participant is the outer accumulation order; values remain concept rows.
        for subject in SUBJECTS:
            for concept, vector in zip(concepts, values):
                images = grouped[concept.cid][subject]
                core.require(len(images) == concept.qualified_count, "qualified_image_feature_scope_changed")
                feature, reasons, flags = participant_features(images, sessions[subject], totals[subject])
                vector.extend(feature)
                all_reasons.update(reasons)
                all_flags[f"{subject}:{concept.cid}"] = dict(flags)
    core.require(all(len(vector) == len(fields) for vector in values), "feature_width_changed")
    return RawFeatures(concepts, fields, values, {"unresolved_delay_reasons": dict(all_reasons),
                                                "schedule_flag_counts_by_concept_participant": all_flags})


def stable_mean(values):
    core.require(bool(values), "mean_of_empty_field")
    if all(value == values[0] for value in values):
        return values[0]
    return sum(values, Decimal(0)) / Decimal(len(values))


def quantize_z(value):
    return int((Decimal(QUANTUM) * value).to_integral_value(rounding=ROUND_HALF_EVEN))


def prepare(raw, parent_ids=None):
    """No fold/block labels accepted: only parent concept IDs and raw metadata."""
    selected = set(parent_ids) if parent_ids is not None else {c.cid for c in raw.concepts}
    pairs = [(c, vector) for c, vector in zip(raw.concepts, raw.values) if c.cid in selected]
    pairs.sort(key=lambda pair: pair[0].key())
    core.require(len(pairs) == len(selected) and len(pairs) >= 2, "invalid_assignment_parent")
    concepts = [c for c, _ in pairs]
    original = [list(vector) for _, vector in pairs]
    filled, q, details = [list(vector) for vector in original], [[] for _ in original], []
    with localcontext() as context:
        context.prec, context.rounding = PRECISION, ROUND_HALF_EVEN
        for j, field in enumerate(raw.fields):
            observed = [vector[j] for vector in original if vector[j] is not None]
            core.require(all(value.is_finite() for value in observed), "nonfinite_feature")
            core.require(len(observed) == len(original) or field.name.endswith(("_log_mean", "_log_variance")),
                         "unexpected_missing_feature")
            imputation = stable_mean(observed) if observed else Decimal(0)
            for vector in filled:
                if vector[j] is None:
                    vector[j] = imputation
            column = [vector[j] for vector in filled]
            constant = all(value == column[0] for value in column)
            mean = column[0] if constant else stable_mean(column)
            variance = Decimal(0) if constant else sum(((value - mean) ** 2 for value in column), Decimal(0)) / Decimal(len(column))
            core.require(constant or variance.is_finite() and variance > 0, "invalid_nonconstant_feature_variance")
            sd = Decimal(0) if constant else variance.sqrt()
            for vector, value in zip(q, column):
                quantized = 0 if constant else quantize_z((value - mean) / sd)
                vector.append(quantized)
            details.append({"name": field.name, "observed_concepts": len(observed),
                            "imputed_concepts": len(original) - len(observed), "wholly_unobserved": not observed,
                            "constant": constant, "imputation_value": str(imputation),
                            "parent_mean": str(mean), "parent_variance": str(variance)})
    return Prepared(concepts, raw.fields, original, filled, q, details)


def integer_weights(fields):
    denominator = lcm(*(field.weight.denominator for field in fields))
    return denominator, [field.weight.numerator * (denominator // field.weight.denominator) for field in fields]


def initialize(concepts, sizes, namespace):
    core.require(sum(sizes) == len(concepts) and all(size > 0 for size in sizes), "invalid_group_capacities")
    prefix = namespace.encode("utf-8") + b"\0init\0"
    order = sorted(range(len(concepts)), key=lambda i: (hashlib.sha256(prefix + concepts[i].identity()).digest(), concepts[i].key()))
    labels, begin = [-1] * len(concepts), 0
    for group, size in enumerate(sizes):
        for i in order[begin:begin + size]:
            labels[i] = group
        begin += size
    return labels


def pair_order(concepts, namespace):
    core.require(concepts == sorted(concepts, key=lambda c: c.key()), "concepts_not_canonical")
    prefix = namespace.encode("utf-8") + b"\0swap\0"
    identities = [concept.identity() for concept in concepts]
    ordered = [(hashlib.sha256(prefix + identities[a] + b"\0" + identities[b]).digest(), a, b)
               for a in range(len(concepts)) for b in range(a + 1, len(concepts))]
    ordered.sort()
    return [(a, b) for _, a, b in ordered]


def gram_matrix(q, weights):
    n = len(q)
    result = [[0] * n for _ in range(n)]
    weighted = [[value * weight for value, weight in zip(vector, weights)] for vector in q]
    for a in range(n):
        for b in range(a, n):
            value = sum(x * y for x, y in zip(weighted[a], q[b]))
            result[a][b] = result[b][a] = value
    return result


def group_feature_sums(q, labels, groups):
    sums = [[0] * len(q[0]) for _ in range(groups)]
    for vector, label in zip(q, labels):
        for j, value in enumerate(vector):
            sums[label][j] += value
    return sums


def objective(q, weights, labels, sizes):
    common = lcm(*sizes)
    sums = group_feature_sums(q, labels, len(sizes))
    return sum((common // size) * sum(w * value * value for w, value in zip(weights, vector))
               for size, vector in zip(sizes, sums))


def caches(gram, labels, groups):
    output = []
    for vector in gram:
        totals = [0] * groups
        for value, label in zip(vector, labels):
            totals[label] += value
        output.append(totals)
    return output


def swap_delta(gram, cache, labels, sizes, a, b):
    A, B = labels[a], labels[b]
    if A == B:
        return 0
    common = lcm(*sizes)
    h = gram[a][a] + gram[b][b] - 2 * gram[a][b]
    return ((common // sizes[A]) * (2 * (cache[b][A] - cache[a][A]) + h) +
            (common // sizes[B]) * (-2 * (cache[b][B] - cache[a][B]) + h))


def direct_delta(q, weights, sums, labels, sizes, a, b):
    """Independent O(fields) evaluation, without Gram/cache values."""
    A, B = labels[a], labels[b]
    if A == B:
        return 0
    common = lcm(*sizes)
    total = 0
    for j, weight in enumerate(weights):
        difference = q[b][j] - q[a][j]
        total += weight * ((common // sizes[A]) * ((sums[A][j] + difference) ** 2 - sums[A][j] ** 2) +
                           (common // sizes[B]) * ((sums[B][j] - difference) ** 2 - sums[B][j] ** 2))
    return total


def apply_swap(gram, cache, labels, a, b):
    A, B = labels[a], labels[b]
    core.require(A != B, "same_group_cache_update")
    for i, vector in enumerate(gram):
        difference = vector[b] - vector[a]
        cache[i][A] += difference
        cache[i][B] -= difference
    labels[a], labels[b] = B, A


def optimize(prepared, sizes, namespace, check_deadline=lambda: None):
    """Exactly one initialization; at most eight full fixed-order sweeps."""
    _, weights = integer_weights(prepared.fields)
    labels = initialize(prepared.concepts, sizes, namespace)
    gram = gram_matrix(prepared.q, weights)
    check_deadline()
    cache = caches(gram, labels, len(sizes))
    pairs = pair_order(prepared.concepts, namespace)
    check_deadline()
    value = initial = objective(prepared.q, weights, labels, sizes)
    sweeps, visits, updates = [], 0, 0
    for sweep in range(MAX_SWEEPS):
        accepted = 0
        for a, b in pairs:
            check_deadline()
            visits += 1
            if labels[a] == labels[b]:
                continue
            delta = swap_delta(gram, cache, labels, sizes, a, b)
            if delta < 0:
                apply_swap(gram, cache, labels, a, b)
                value += delta
                accepted += 1
                updates += len(labels)
        check_deadline()
        core.require(Counter(labels) == Counter({group: size for group, size in enumerate(sizes)}), "capacity_changed")
        core.require(value == objective(prepared.q, weights, labels, sizes) and cache == caches(gram, labels, len(sizes)),
                     "exact_objective_cache_disagreement")
        sweeps.append({"sweep": sweep + 1, "accepted_swaps": accepted, "J": value})
        if accepted == 0:
            break
    return labels, {"initial_J": initial, "final_J": value, "sweeps": sweeps, "pair_visits": visits,
                    "per_concept_cache_updates": updates, "pairwise_local_optimum": sweeps[-1]["accepted_swaps"] == 0,
                    "fixed_sweep_budget_exhausted": len(sweeps) == MAX_SWEEPS and sweeps[-1]["accepted_swaps"] > 0,
                    "global_optimum_claimed": False}


def assign_children(raw, development_ids, audit_ids, check_deadline=lambda: None):
    """Fold labels are never inputs to either block construction."""
    output = {}
    for name, parent, sizes, suffix in (
        ("outer_folds", development_ids, [80] * 6, "/outer-fold"),
        ("development_blocks", development_ids, [20] * 24, "/development-block"),
        ("audit_blocks", audit_ids, [10] * 12, "/audit-block"),
    ):
        prepared = prepare(raw, parent)
        labels, report = optimize(prepared, sizes, NAMESPACE + suffix, check_deadline)
        output[name] = {"labels": dict(zip((c.cid for c in prepared.concepts), labels)),
                        "report": report, "prepared": prepared, "sizes": sizes}
    return output


def balance_summary(prepared, labels, sizes, group_names):
    """Full metadata residuals; exact quantized losses, no selection threshold."""
    core.require(len(group_names) == len(sizes) and len(labels) == len(prepared.concepts) and
                 Counter(labels) == Counter({g: size for g, size in enumerate(sizes)}), "balance_capacity_changed")
    n = len(labels)
    indices = [[i for i, label in enumerate(labels) if label == g] for g in range(len(sizes))]
    family_quantized, family_raw = defaultdict(Fraction), defaultdict(Decimal)
    residuals = []
    with localcontext() as context:
        context.prec, context.rounding = PRECISION, ROUND_HALF_EVEN
        for j, field in enumerate(prepared.fields):
            parent_raw = stable_mean([vector[j] for vector in prepared.imputed])
            parent_q = Fraction(sum(vector[j] for vector in prepared.q), n * QUANTUM)
            variance = Decimal(prepared.field_details[j]["parent_variance"])
            sd = variance.sqrt() if variance else Decimal(0)
            groups = []
            for group, selected in enumerate(indices):
                raw_mean = stable_mean([prepared.imputed[i][j] for i in selected])
                raw_difference = raw_mean - parent_raw
                raw_standardized = raw_difference / sd if sd else Decimal(0)
                q_mean = Fraction(sum(prepared.q[i][j] for i in selected), sizes[group] * QUANTUM)
                q_difference = q_mean - parent_q
                mass = field.weight * Fraction(sizes[group], n)
                family_quantized[field.family] += mass * q_difference ** 2
                family_raw[field.family] += decimal_fraction(mass) * raw_standardized ** 2
                observed = sum(prepared.raw[i][j] is not None for i in selected)
                groups.append({"group": group_names[group], "concepts": sizes[group],
                               "observed_concepts": observed, "imputed_concepts": sizes[group] - observed,
                               "imputed_raw_mean": str(raw_mean), "imputed_raw_residual": str(raw_difference),
                               "unquantized_standardized_residual": str(raw_standardized),
                               "quantized_standardized_mean": str(decimal_fraction(q_mean)),
                               "quantized_standardized_residual": str(decimal_fraction(q_difference)),
                               "quantized_residual_numerator": q_difference.numerator,
                               "quantized_residual_denominator": q_difference.denominator})
            residuals.append({"field": field.name, "family": field.family,
                              "weight_numerator": field.weight.numerator, "weight_denominator": field.weight.denominator,
                              "parent_imputed_raw_mean": str(parent_raw),
                              "parent_quantized_standardized_mean": str(decimal_fraction(parent_q)),
                              "constant": prepared.field_details[j]["constant"],
                              "wholly_unobserved": prepared.field_details[j]["wholly_unobserved"], "groups": groups})
        losses = {family: {"quantized_loss_numerator": value.numerator, "quantized_loss_denominator": value.denominator,
                           "quantized_loss_decimal": str(decimal_fraction(value)),
                           "unquantized_standardized_loss_decimal": str(family_raw[family])}
                  for family, value in family_quantized.items()}
    return {"group_sizes": dict(zip(group_names, sizes)), "family_losses": losses, "field_residuals": residuals,
            "raw_residuals_use_declared_parent_imputation": True,
            "imputed_values_are_not_observed_delays": True, "assignment_selection_used_only_quantized_objective": True}
