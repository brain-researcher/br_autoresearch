import numpy as np
import pandas as pd

from outputs.code.fit_subject import aggregate, build_design
from outputs.code.aggregate_results import (
    gram_pack,
    region_metric,
    threshold_pair_metrics,
    weighted_metrics,
)


def test_equal_run_and_ivw_agree_for_equal_variance():
    effects = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    variances = np.ones_like(effects)
    ivw, equal, stabilized, floors = aggregate(effects, variances)
    np.testing.assert_allclose(ivw, equal)
    np.testing.assert_allclose(stabilized, equal)
    np.testing.assert_allclose(floors, 1.0)


def test_stabilization_caps_extreme_weight():
    rng = np.random.default_rng(3)
    effects = rng.normal(size=(4, 2000))
    variances = np.ones_like(effects)
    variances[0, 0] = 1e-30
    ivw, equal, stabilized, floors = aggregate(effects, variances)
    assert floors[0] > variances[0, 0]
    assert abs(stabilized[0] - equal[0]) < abs(ivw[0] - equal[0])


def test_actual_design_metadata_builds_finite_c1():
    design = build_design(
        __import__("pathlib").Path("inputs/fmriprep/sub-001/func/sub-001_task-MGT_run-1_desc-confounds_timeseries.tsv"),
        __import__("pathlib").Path("inputs/fitlins/task-MGT/node-runLevel/sub-001/sub-001_task-MGT_run-1_design.tsv"),
    )
    assert len(design) == 453
    assert np.isfinite(design.to_numpy()).all()
    assert "gain_demean" in design and "loss_demean" in design
    assert sum(name.startswith("cosine") for name in design.columns) == 6
    assert design.shape[1] == 36
    assert np.allclose(design["trans_x_derivative1"].iloc[0], 0.0)


def test_gram_metrics_equal_explicit_group_map():
    rng = np.random.default_rng(9)
    m0 = rng.normal(size=(7, 101))
    m8 = 0.7 * m0 + rng.normal(scale=0.3, size=m0.shape)
    region = rng.uniform(size=101) > 0.2
    weights = rng.uniform(size=7)
    weights /= weights.sum()
    explicit = region_metric(weights @ m0, weights @ m8, region)
    gram = weighted_metrics(gram_pack(m0, m8, region), weights)
    for key in explicit:
        np.testing.assert_allclose(gram[key][0], explicit[key], rtol=1e-11, atol=1e-12)


def test_threshold_pair_helper_emits_fixed_readouts():
    rng = np.random.default_rng(1)
    s0 = 2.0 + rng.normal(size=(12, 300))
    s8 = s0 + 0.05 * rng.normal(size=s0.shape)
    result = threshold_pair_metrics(s0, s8, None)
    assert result["df"] == 11
    assert result["matched_k"] > 0
    assert 0.0 <= result["fdr_jaccard"] <= 1.0
    assert 0.0 <= result["top_abs_t_jaccard"] <= 1.0


def test_affine_alignment_negative_control_is_numerically_stable():
    rng = np.random.default_rng(17)
    y0 = rng.normal(scale=0.01, size=10_000)
    control = 0.1 * np.std(y0) + 1.25 * y0
    result = region_metric(y0, control, np.ones(len(y0), dtype=bool))
    residual_rms_ratio = result["d_aligned"]
    assert abs(result["pearson_r"] - 1.0) <= 1e-12
    assert residual_rms_ratio <= 1e-10
