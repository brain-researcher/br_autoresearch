"""Pure array primitives for the registered EP03 comparison; no data reader.

These functions implement target normalization, canonical family prototypes and
the exact group-weighted loss ratio. They neither authorize target access nor
assign confidence intervals to dependent out-of-fold scores. Estimators,
inference thresholds and scientific acceptance are deliberately separate.
"""

import numpy as np


def normalize_targets(values, mask):
    z = np.asarray(values, dtype=np.float64)
    keep = np.asarray(mask)
    if z.ndim != 2 or keep.dtype != np.dtype(bool) or keep.shape != (z.shape[1],):
        raise ValueError("provide map-by-location values and an explicit boolean common mask")
    if keep.sum() < 2:
        raise ValueError("common mask needs at least two locations")
    selected = z[:, keep]
    if not np.isfinite(selected).all():
        raise ValueError("nonfinite target within the fixed mask; no per-map deletion")
    centered = selected - selected.mean(axis=1, keepdims=True)
    # Scale first to avoid squaring overflow/underflow without an epsilon floor.
    scale = np.max(np.abs(centered), axis=1, keepdims=True)
    if not np.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("constant or numerically invalid target on fixed support")
    unit = centered / scale
    return unit / np.linalg.norm(unit, axis=1, keepdims=True)


def _labels(labels, length, name):
    result = np.asarray(labels, dtype=str)
    if result.shape != (length,) or (result == "").any():
        raise ValueError(f"invalid {name} labels")
    return result


def family_prototypes(normalized_source_targets, groups, families, polarity, training_groups):
    z = np.asarray(normalized_source_targets, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("map-by-location targets required")
    group = _labels(groups, len(z), "group")
    family = _labels(families, len(z), "family")
    s = np.asarray(polarity)
    if s.shape != (len(z),) or not np.isin(s, [-1, 1]).all():
        raise ValueError("each map needs its adopted polarity")
    requested = set(training_groups)
    if not requested or not requested.issubset(set(group)):
        raise ValueError("training groups must be nonempty and present")
    train = np.isin(group, list(requested))
    if not np.isfinite(z[train]).all():
        raise ValueError("nonfinite training targets")
    # Do not inspect held-out target values when fitting the prototype.
    canonical = z[train] * s[train, None]
    train_group, train_family = group[train], family[train]
    result = {}
    for f in np.unique(train_family):
        present = train_family == f
        # A multi-contrast dataset contributes one family mean, not n votes.
        means = [canonical[present & (train_group == g)].mean(axis=0)
                 for g in np.unique(train_group[present])]
        result[f] = np.mean(means, axis=0)
    return result


def predictions(prototypes, families, polarity, residual=None):
    families = list(families)
    if not families or any(f not in prototypes for f in families):
        raise ValueError("unrepresented family; report unsupported, never use a zero baseline")
    base = np.stack([prototypes[f] for f in families])
    s = np.asarray(polarity)
    if s.shape != (len(base),) or not np.isin(s, [-1, 1]).all():
        raise ValueError("invalid requested polarity")
    r = np.zeros_like(base) if residual is None else np.asarray(residual, dtype=np.float64)
    if r.shape != base.shape or not np.isfinite(r).all() or not np.isfinite(base).all():
        raise ValueError("finite matching residual/prototype shapes required")
    return base * s[:, None], (base + r) * s[:, None]


def grouped_relative_gain(target, family_prediction, model_prediction, groups):
    arrays = [np.asarray(x, dtype=np.float64) for x in (target, family_prediction, model_prediction)]
    z, p, q = arrays
    if z.ndim != 2 or not z.size or any(a.shape != z.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("identical finite nonempty map/location shapes required")
    group = _labels(groups, len(z), "group")
    records = []
    for g in np.unique(group):
        rows = group == g
        lp = float(np.mean((z[rows] - p[rows]) ** 2))
        lq = float(np.mean((z[rows] - q[rows]) ** 2))
        if not np.isfinite(lp) or not np.isfinite(lq) or lp <= 0:
            raise ValueError("zero/invalid group prototype loss; no epsilon or omitted group")
        records.append({"group": str(g), "maps": int(rows.sum()), "family_mse": lp,
                        "model_mse": lq, "relative_gain": 1.0 - lq / lp})
    return {"groups": records, "mean_relative_gain": float(np.mean([r["relative_gain"] for r in records])),
            "uncertainty": "requires grouped resampling with refitting; no iid OOF interval",
            "scientific_acceptance": "not_evaluated"}
