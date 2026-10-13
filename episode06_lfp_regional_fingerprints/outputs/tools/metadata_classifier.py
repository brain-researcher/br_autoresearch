"""Fixed EP06 recording/support-metadata control; no data or file reader.

The caller supplies the completed seven-session exporter report and owns
scientific admission and the compute deadline. Only log1p(native unit count),
log1p(training-finite electrode count), and log1p(training trial count) enter
the model. Names and identities bind paired rows and animal transfer only.
Training-finite electrode count is an exporter-recorded support measure, not
an acquisition-header-only measure or a neural reliability estimate.

Each direction uses source-column mean/SD (ddof=0), freezes source-constant
columns to zero for both animals, and fits mean logloss + .5*||beta||^2 with
an unpenalized intercept. The frozen profile-classifier Newton solver supplies
the same convergence rule. There is no target fit, row shape normalization,
tuning, label null, neural comparison, final-data opening, or acceptance rule.
Missing/invalid support or numerical failure makes the whole control
unavailable. Exact logit ties receive half credit, never a forced region.
"""
from collections.abc import Mapping
import math
from numbers import Integral

import numpy as np

from profile_classifier import ClassificationUnavailable, _fit_logistic


# Fixed development identities mirror the adopted exporter roster. Importing
# an exporter/reader is unnecessary: this primitive receives metadata only.
SESSIONS = (
    ("Mihili", "Mihili_CO_20140217.mat", 208),
    ("Mihili", "Mihili_CO_20140218.mat", 225),
    ("Mihili", "Mihili_CO_20140304.mat", 203),
    ("Chewie-L", "Chewie_CO_20160929.mat", 208),
    ("Chewie-L", "Chewie_CO_20161005.mat", 202),
    ("Chewie-L", "Chewie_CO_20161007.mat", 168),
    ("Chewie-L", "Chewie_CO_20161014.mat", 740),
)
FEATURES = ("native_units", "training_finite_electrodes", "training_trials")
REGIONS = ("M1", "PMd")
ANIMALS = ("Mihili", "Chewie-L")


class MetadataUnavailable(RuntimeError):
    """The entire fixed seven-session metadata comparison is unavailable."""


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource/cancellation check required")
    check()


def _count(value, minimum, maximum, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not minimum <= value <= maximum):
        raise MetadataUnavailable(f"invalid metadata count: {name}")
    return int(value)


def _metadata_rows(report, check):
    """Validate the entire roster before either source fit; read no arrays."""
    _check(check)
    if (not isinstance(report, Mapping)
            or report.get("kind") != "EP06_seven_session_development_views"
            or report.get("status") != "completed"
            or report.get("stage") != "preparation_only_no_scientific_scores"
            or report.get("reserved_sessions_opened") is not False
            or report.get("original_trials") != sum(row[2] for row in SESSIONS)):
        raise MetadataUnavailable("completed development-only exporter report required")
    records = report.get("sessions")
    if not isinstance(records, (list, tuple)) or len(records) != len(SESSIONS):
        raise MetadataUnavailable("all seven fixed paired development sessions required")
    identities, counts = [], []
    for (animal, filename, total), record in zip(SESSIONS, records):
        _check(check)
        if (not isinstance(record, Mapping) or record.get("filename") != filename
                or record.get("status") != "complete"):
            raise MetadataUnavailable("ordered development session identity/completion differs")
        roles = record.get("trial_counts")
        if not isinstance(roles, Mapping) or set(roles) != {"training", "calibration", "evaluation"}:
            raise MetadataUnavailable("explicit training/calibration/evaluation count roles required")
        # These are necessary aggregate bounds, not a reconstruction of the
        # exporter's direction-wise split or a claim to revalidate its arrays.
        role_counts = {role: _count(roles[role], minimum, total, role)
                       for role, minimum in (("training", 56), ("calibration", 32), ("evaluation", 32))}
        if sum(role_counts.values()) != total:
            raise MetadataUnavailable("all original trials must retain one declared role")
        regions = record.get("regions")
        if not isinstance(regions, (list, tuple)) or len(regions) != 2:
            raise MetadataUnavailable("both paired regional support records required")
        for region, regional in zip(REGIONS, regions):
            if not isinstance(regional, Mapping) or regional.get("region") != region:
                raise MetadataUnavailable("ordered M1/PMd support identities differ")
            native = _count(regional.get("native_electrodes"), 96, 96, "native_electrodes")
            units = _count(regional.get("native_units"), 1, 4096, "native_units")
            usable = _count(regional.get("training_finite_electrodes"), 15, native,
                            "training_finite_electrodes")
            identities.append((animal, filename[:-4], region))
            counts.append((units, usable, role_counts["training"]))
    return tuple(identities), np.log1p(np.asarray(counts, dtype=np.float64))


def _direction(identities, x, source_animal, check):
    _check(check)
    source_indices = [i for i, row in enumerate(identities) if row[0] == source_animal]
    target_indices = [i for i, row in enumerate(identities) if row[0] != source_animal]
    source, target = x[source_indices], x[target_indices]
    mean = np.mean(source, axis=0)
    constant = np.all(source == source[0], axis=0)
    sd = np.sqrt(np.mean((source - mean) ** 2, axis=0))
    sd[constant] = 0.  # Exact repeated input may have a rounded nonzero residual.
    active = ~constant
    if (not np.isfinite(mean).all() or not np.isfinite(sd).all()
            or np.any(sd[active] <= 0)):
        raise MetadataUnavailable("source metadata scaling unavailable")
    source_z, target_z = np.zeros_like(source), np.zeros_like(target)
    source_z[:, active] = (source[:, active] - mean[active]) / sd[active]
    target_z[:, active] = (target[:, active] - mean[active]) / sd[active]
    # Class labels are a separate source-only vector, not metadata features.
    source_labels = np.asarray([int(identities[i][2] == "M1") for i in source_indices], dtype=float)
    try:
        parameters, objective, iterations, gradient = _fit_logistic(source_z, source_labels, 1., check)
    except ClassificationUnavailable as error:
        raise MetadataUnavailable(f"source metadata fit unavailable: {error}") from error
    _check(check)
    logits = target_z @ parameters[:-1] + parameters[-1]
    if not np.isfinite(logits).all():
        raise MetadataUnavailable("nonfinite frozen metadata prediction")
    predictions = []
    for index, logit in zip(target_indices, logits):
        animal, session, region = identities[index]
        truth = int(region == "M1")
        predicted = None if logit == 0 else int(logit > 0)
        predictions.append({"animal": animal, "session_id": session, "region": region,
            "logit": float(logit), "true_label": truth, "predicted_label": predicted,
            "credit": .5 if predicted is None else float(predicted == truth)})
    recalls = [float(np.mean([row["credit"] for row in predictions if row["true_label"] == label]))
               for label in (0, 1)]
    paired = [{"session_id": predictions[i]["session_id"],
               "accuracy": .5 * (predictions[i]["credit"] + predictions[i + 1]["credit"])}
              for i in range(0, len(predictions), 2)]
    return {"source_animal": source_animal, "target_animal": predictions[0]["animal"],
        "source_sessions": [identities[i][1] for i in source_indices[::2]],
        "target_sessions": [row["session_id"] for row in paired],
        "source_rows": len(source_indices), "target_rows": len(target_indices),
        "source_mean": mean.tolist(), "source_sd_ddof0": sd.tolist(),
        "source_constant_columns": constant.tolist(), "coefficients": parameters[:-1].tolist(),
        "intercept": float(parameters[-1]), "objective": float(objective),
        "iterations": int(iterations), "maximum_absolute_gradient": float(gradient),
        "predictions": predictions, "paired_session_accuracy": paired,
        "class_0_recall": recalls[0], "class_1_recall": recalls[1],
        "balanced_accuracy": .5 * sum(recalls),
        "tied_rows": sum(row["predicted_label"] is None for row in predictions)}


def evaluate_metadata_classifier(export_report, *, check):
    """Return both frozen cross-animal predictions or raise whole-control unavailability.

    This function accepts an already-read completed exporter report. It never
    reads HDF, a source file, a scientific score or a neural outcome. The caller
    must execute scientific computation in its coordinated compute allocation.
    """
    identities, x = _metadata_rows(export_report, check)
    directions = [_direction(identities, x, animal, check) for animal in ANIMALS]
    _check(check)
    return {"kind": "EP06_recording_support_metadata_classifier", "status": "complete",
        "feature_order": list(FEATURES), "transform": "log1p_then_source_column_mean_sd_ddof0",
        "source_constant_rule": "zero_for_both_source_and_target", "penalty": 1.,
        "objective": "mean_logloss_plus_half_penalty_beta_squared_unpenalized_intercept",
        "label_definition": {"M1": 1, "PMd": 0}, "exact_tie_credit": .5,
        "directions": directions,
        "equal_direction_mean_balanced_accuracy": math.fsum(row["balanced_accuracy"] for row in directions) / 2.,
        "claim_scope": "fixed_development_recording_support_metadata_control_only",
        "scientific_acceptance": None}
