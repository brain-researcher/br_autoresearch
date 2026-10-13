#!/usr/bin/env python3
"""Contract-relevant EP05 synthetic indexing, grouping and leakage checks.

No empirical source or publication outcome is read. Passing does not establish
publication fidelity or select the primary scientific configuration.
"""
import argparse
from pathlib import Path

import numpy as np

from method_components import DirectionTime, lodo_predictions, matched_electrode_mean, r2_sse
from source_adapter import compute_only, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    checks = {}
    labels = np.repeat(np.arange(8), 4)
    angles = labels * np.pi / 4
    times = np.arange(6) / 5
    baseline = np.empty((32, 6, 2))
    baseline[:, :, 0] = np.cos(angles)[:, None] * (0.4 + times) + 0.2 * np.sin(2 * angles)[:, None]
    baseline[:, :, 1] = np.sin(angles)[:, None] * (0.8 - times) - 0.3 * np.cos(2 * angles)[:, None]
    held = labels == 3
    fit = DirectionTime.fit(angles[~held], baseline[~held], 2, labels[~held])
    error = float(np.max(np.abs(fit.predict(angles[held]) - baseline[held])))
    assert error < 1e-12, "cyclic held-direction known answer failed"
    checks["held_direction_cyclic_known_answer"] = {"passed": True, "max_error": error}
    trial_deviation = np.tile([-1.5, -0.5, 0.5, 1.5], 8)
    features = (trial_deviation[:, None] * (0.5 + times))[..., None]
    target = baseline + features * np.array([0.8, -0.3])
    output = lodo_predictions(features, target, angles, labels,
                              harmonics=2, ridge_alpha=0.0, minimum_trials_per_direction=4)
    candidate_score = r2_sse(target, output["candidate"])
    assert abs(candidate_score - 1) < 1e-12
    checks["trial_specific_linear_known_answer"] = {"passed": True, "R2_SSE": candidate_score}
    changed = target.copy()
    changed[held] += 700.0
    repeated = lodo_predictions(features, changed, angles, labels,
                                harmonics=2, ridge_alpha=0.0, minimum_trials_per_direction=4)
    assert np.array_equal(output["candidate"][held], repeated["candidate"][held])
    assert np.array_equal(output["models"][3].feature_mean, repeated["models"][3].feature_mean)
    checks["held_direction_targets_do_not_fit_that_fold"] = {"passed": True}
    changed_features = features.copy()
    changed_features[held] += 1000.0
    feature_repeated = lodo_predictions(changed_features, target, angles, labels,
                                       harmonics=2, ridge_alpha=0.0, minimum_trials_per_direction=4)
    assert np.array_equal(output["models"][3].feature_mean, feature_repeated["models"][3].feature_mean)
    assert np.array_equal(output["models"][3].feature_scale, feature_repeated["models"][3].feature_scale)
    checks["held_direction_features_do_not_scale_that_fold"] = {"passed": True}
    derangement = np.arange(len(labels))
    for direction in range(8):
        indices = np.flatnonzero(labels == direction)
        derangement[indices] = np.roll(indices, 1)
    wrong_score = r2_sse(target, output["candidate"][derangement])
    assert wrong_score < candidate_score
    checks["whole_trial_pairing_detects_individual_signal"] = {"passed": True, "wrong_pair_R2_SSE": wrong_score}
    assert r2_sse(target, np.full_like(target, 100.0)) < 0
    checks["negative_R2_retained"] = {"passed": True}
    guide = np.array([[43, 0, 0], [2, 8, 12], [2, 0, 0], [43, 8, 12]])
    raw = np.array([[3.0, 100.0, 7.0, 200.0], [5.0, 300.0, 9.0, 400.0]])
    selected = matched_electrode_mean(raw, guide, 0, 0, [2, 43])
    assert np.array_equal(selected, np.array([[5.0], [7.0]]))
    checks["noncontiguous_electrode_join_uses_guide"] = {"passed": True}
    write_json(args.out, {"analysis_id": "publication_reproduction", "scope": "episode_method_components_only",
                          "empirical_data_read": False, "publication_fidelity_established": False,
                          "scientific_configuration_selected": False, "checks": checks, "passed": True})
    print("EP05 method component known-answer/indexing/leakage checks passed.")


if __name__ == "__main__":
    main()
