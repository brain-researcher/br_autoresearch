"""Dormant two-member EP07 development-prediction composition; no file access.

The caller chooses two already evaluated members and one positive convex pair
of weights for the entire five-day/two-region panel. This function does not
choose members or weights, fit models, read targets, score predictions, or
average R2. It is not a final-opening interface. Feedback records must be the
caller-admitted completed pair records from allowed development feedback;
matching declared IDs/stream and complete available support does not authenticate
arbitrary dictionaries or prove their association with a file's contents.

Both input panels are fully validated and copied into immutable snapshots,
including common trial order, native neuron/time axes and exact calibration
baseline equality, before any weighted prediction is formed. Different days
retain their independent axes; no cross-day units are matched. Member order
and weights remain the same in every cell. Weights must sum to one as supplied,
without implicit normalization, fitting, clipping or per-setting exceptions.

The result is unscored and unpublished. Its NativePredictions/baseline tuples
can be passed to the existing CandidatePredictionWriter or separate scorer by
an authorized caller. It deliberately carries no completed-publication marker.
The caller must record the returned member IDs/weights with its candidate recipe.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Real

import numpy as np

import candidate_predictions as publication
from native_trials import NativePredictions, _snapshot


@dataclass(frozen=True)
class EnsemblePredictions:
    candidate_id: str
    stream: str
    member_ids: tuple[str, str]
    weights: tuple[float, float]
    predictions: tuple[NativePredictions, ...]
    calibration_means: tuple[NativePredictions, ...]


def _finite(value):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
            and math.isfinite(float(value)))


def _weights(values):
    if (not isinstance(values, tuple) or len(values) != 2
            or any(not _finite(value) or not 0 < value < 1 for value in values)):
        raise ValueError("two explicit finite positive global weights are required")
    result = tuple(float(value) for value in values)
    if math.fsum(result) != 1.0:
        raise ValueError("global ensemble weights must sum to one without normalization")
    return result


def _evaluated(record, candidate_id, stream):
    """Check supplied existing feedback, never infer success from predictions."""
    if (not isinstance(record, Mapping) or record.get("candidate_id") != candidate_id
            or record.get("state") != "paired_development_scored"
            or record.get("outcome_contact") is not True
            or not isinstance(record.get("streams"), Mapping)):
        raise PermissionError("each ensemble member requires its completed allowed development record")
    result = record["streams"].get(stream)
    if (not isinstance(result, Mapping) or result.get("complete") is not True
            or not _finite(result.get("global_selection_r2"))):
        raise PermissionError("the declared member stream has no complete available evaluation")
    regions = result.get("regions")
    if (not isinstance(regions, (list, tuple)) or len(regions) != 2
            or any(not isinstance(region, Mapping) for region in regions)
            or {region.get("region") for region in regions} != {"M1", "PMd"}):
        raise PermissionError("member evaluation must contain exactly both native regions")
    for region in regions:
        if (region.get("available") is not True or region.get("unavailable_reason") is not None
                or type(region.get("complete_new_days")) is not int
                or region["complete_new_days"] != 5
                or type(region.get("independent_animals")) is not int
                or region["independent_animals"] != 2
                or any(not _finite(region.get(name)) for name in
                       ("model_r2", "calibration_mean_r2", "improvement_over_mean"))):
            raise PermissionError("member evaluation lacks complete available five-day/two-animal support")


def _panel(member, check):
    record = member.check_record
    if (not isinstance(record, publication.PublicationCheck)
            or record.kind != publication.CHECK_KIND
            or type(record.complete_count) is not int
            or record.complete_count != len(publication.PANEL_KEYS)
            or not isinstance(member.predictions, tuple)
            or not isinstance(member.calibration_means, tuple)
            or len(member.predictions) != len(publication.PANEL_KEYS)
            or len(member.calibration_means) != len(publication.PANEL_KEYS)):
        raise ValueError("each member requires the complete published ten-setting panel")
    cells, sessions = {}, {}
    for prediction, baseline in zip(member.predictions, member.calibration_means):
        publication._check(check)
        prediction, baseline = publication._pair(prediction, baseline)
        key = prediction.animal, prediction.target_day, prediction.region
        if key in cells:
            raise ValueError("a member cannot duplicate a panel setting")
        publication._session_binding(sessions, prediction)
        cells[key] = prediction, baseline
    if set(cells) != set(publication.PANEL_KEYS):
        raise ValueError("all and only the fixed ten settings are required")
    return tuple(cells[key] for key in publication.PANEL_KEYS)


def _same_cell(left, right):
    a, baseline_a = left
    b, baseline_b = right
    for name in ("animal", "target_day", "region", "trial_ids", "calibration_trial_ids"):
        if getattr(a, name) != getattr(b, name):
            raise ValueError("ensemble member trial, calibration or setting identities differ")
    for name in ("directions", "bin_offsets", "unit_guide"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise ValueError("ensemble member native axes or directions differ")
    if not np.array_equal(baseline_a.values, baseline_b.values):
        raise ValueError("ensemble members must retain exactly the same calibration comparator")


def _weighted_values(left, right, weights):
    with np.errstate(over="ignore", invalid="ignore"):
        result = weights[0] * left + weights[1] * right
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite native ensemble prediction")
    return _snapshot(result)


def compose_ensemble(members, *, candidate_id, stream, member_ids,
                     feedback_records, weights=(.5, .5), check):
    """Compose exactly two evaluated same-stream members using fixed weights.

    ``feedback_records`` is an ordered pair of existing per-candidate records
    (the entries in an allowed development report's ``pairs`` list), matching
    ``member_ids``. Only the requested stream is required/inspected within each
    record. No complete-search/finalist status is inferred from these members.
    """
    publication._check(check)
    publication._identity(candidate_id, stream)
    if (not isinstance(member_ids, tuple) or len(member_ids) != 2
            or any(not isinstance(value, str) for value in member_ids)
            or len(set(member_ids)) != 2 or candidate_id in member_ids):
        raise ValueError("two distinct globally fixed members and a new candidate ID are required")
    for value in member_ids:
        publication._identity(value, stream)
    weights = _weights(weights)
    members = tuple(members)
    if (len(members) != 2 or any(not isinstance(member, publication.CandidatePredictions)
            or member.candidate_id != name or member.stream != stream
            for member, name in zip(members, member_ids))):
        raise ValueError("exactly the two declared same-stream member panels are required")
    if not isinstance(feedback_records, tuple) or len(feedback_records) != 2:
        raise PermissionError("both completed member feedback records must be supplied")
    # Fail before examining any prediction arrays when prior evaluation is absent.
    for name, record in zip(member_ids, feedback_records):
        _evaluated(record, name, stream)
    left, right = (_panel(member, check) for member in members)
    for a, b in zip(left, right):
        publication._check(check)
        _same_cell(a, b)
    # A mismatch in the final setting cannot leave earlier averaged predictions.
    predictions, baselines = [], []
    for (a, baseline), (b, _) in zip(left, right):
        publication._check(check)
        predictions.append(replace(a, values=_weighted_values(a.values, b.values, weights)))
        baselines.append(baseline)
    publication._check(check)
    return EnsemblePredictions(candidate_id, stream, member_ids, weights,
                               tuple(predictions), tuple(baselines))
