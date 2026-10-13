#!/usr/bin/env python3
"""Known-answer checks of the EP22 OOD evaluator's estimators and verdicts (synthetic data only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import ood_confirmation as oc  # noqa: E402

PARTS = oc.SETTINGS["participants"]
CONF, DESC = oc.SETTINGS["confirmatory_categories"], oc.SETTINGS["descriptive_categories"]
ROIS = oc.SETTINGS["primary_rois"] + [oc.SETTINGS["descriptive_roi"]]
SIZES = dict(zip(CONF + DESC, [40, 36, 30, 34, 8, 10, 12, 14, 16]))


def synthetic(kind, seed=0, d=48, voxels=60, noise=0.05):
    rng = np.random.default_rng(seed)
    cats, start = {}, 0
    for c in CONF + DESC:
        cats[c] = list(range(start, start + SIZES[c]))
        start += SIZES[c]
    n = start
    feats = {m: np.zeros((n, d)) for m in oc.SETTINGS["models"]}
    brain = np.zeros((n, voxels))
    for c, idx in cats.items():
        k = len(idx)
        z = rng.standard_normal((k, d)) * (np.arange(1, d + 1) ** -0.9)          # steep "DINOv2" spectrum
        z = z - z.mean(axis=0)
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        ref = u @ np.diag(s) @ vt + 0.3                                            # offset keeps rows away from zero
        rot = np.linalg.qr(rng.standard_normal((d, d)))[0]
        flat = u @ np.diag(s ** 0.3) @ vt @ rot + 0.3                             # same eigenvectors, flatter spectrum
        feats["DINOv2"][idx] = ref
        for m in ("CLIP", "PEcore", "SigLIP2"):
            feats[m][idx] = ref if kind == "identical" else flat
        w = rng.standard_normal((d, voxels))
        brain[idx] = (rng.standard_normal((k, voxels)) if kind == "random" else (u @ np.diag(s) @ vt) @ w) \
            + noise * rng.standard_normal((k, voxels))
    feats = {m: oc.l2(x) for m, x in feats.items()}
    results, bases = {}, {}
    for p in PARTS:
        pat = brain + noise * rng.standard_normal(brain.shape)
        results[p] = {"patterns": {roi: pat for roi in ROIS}, "eligible": np.ones(n, bool), "evc_eligible": np.ones(n, bool)}
        bases[p] = {m: {"mu": np.zeros(d), "v": np.eye(d), "s": np.ones(d), "alpha": 1.0} for m in oc.SETTINGS["models"]}
    return results, {p: feats for p in PARTS}, {p: cats for p in PARTS}, bases


def run(kind, permutations=200):
    r, f, c, b = synthetic(kind)
    return oc.evaluate(r, f, c, b, permutations, seed=7)


def test_spectral_only_difference():
    rep = run("spectral")
    assert rep["gate"]["passed"], rep["gate"]
    assert rep["verdict_status"] == "scored"
    c1 = rep["C1"]["CLIP"]
    assert c1["gap_mean"] < -0.02 and c1["classification"] == "spectral", c1
    within = rep["C2"]["within_category_detail"]["CLIP"]
    # Shared eigenvectors (up to L2 normalization): the spectrum carries most of the within-category gap.
    assert abs(within["swapped_gap_mean"]) <= 0.5 * abs(within["native_gap_mean"]), within
    assert abs(within["E_vec_mean"]) < abs(within["E_val_mean"]), within
    assert rep["C2"]["C2a_within_category"]["CLIP"]["passed"] and rep["C2"]["C2a_within_category"]["SigLIP2"]["passed"]
    b = rep["C2"]["C2b_all_pairs"]["CLIP"]
    assert set(b) >= {"native_gap_mean", "swapped_gap_mean", "closure_mean", "evaluable", "passed"}, b
    # Exact algebra: gap = E_val + E_vec in every participant.
    d = rep["C2"]["all_pairs_detail"]["SigLIP2"]
    assert all(abs(d["native_gap"][p] - d["E_val"][p] - d["E_vec"][p]) < 1e-12 for p in PARTS)
    assert rep["C3"]["CLIP"]["passed"] is False      # alpha = 1, identity basis: no change from native


def test_identical_models():
    rep = run("identical")
    c1 = rep["C1"]["CLIP"]
    assert abs(c1["gap_mean"]) < 1e-12 and c1["classification"] == "selection_only", c1
    assert rep["C2"]["C2a_within_category"]["CLIP"]["passed"] is False          # E_val == 0 is not < 0
    b = rep["C2"]["C2b_all_pairs"]["CLIP"]
    assert b["evaluable"] is False and b["passed"] is None, b


def test_random_brain_fails_gate():
    rep = run("random")
    assert not rep["gate"]["passed"] and rep["verdict_status"] == "uninformative_gate_failed", rep["gate"]


def test_permutations_deterministic():
    a, b = run("spectral", 100), run("spectral", 100)
    assert a["gate"]["null_threshold"] == b["gate"]["null_threshold"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
