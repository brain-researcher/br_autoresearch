#!/usr/bin/env python3
"""Bounded initializer regression; no optimization or recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import run_mixed_support_population_h_recovery as h

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage0 = args.episode_root / "outputs/stage0"
    layout = h.full.core.load_metadata(stage0 / "whole_cortex_ssptr_inventory.csv")
    roles = h.full.panel_b_fixed_roles(layout)
    indices = np.asarray(roles.audit, dtype=int)
    basis = h.full.DesignBasis.from_support(
        layout.p_design[indices], h.full.equal_unit_weights(layout, indices)
    )
    checkpoint = stage0 / "population_saa_r128_three_fresh_C_model_state.npz"
    config = h.bootstrap_c_checkpoint_config(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as state:
        c_theta = state["frozen_theta"].copy()
    assert config.seed + 8192 == h.EMPIRICAL_PERTURBATION_SEED
    assert config.amplitude == 1.25
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="initializer-regression-", dir=args.scratch_dir
    ) as temporary:
        # One replicate bounds this code regression. It is not an H fit and
        # supplies no evidence that the proposed optimizer repair will work.
        store = h.population.SAAMemmapStore.create(
            Path(temporary), layout, basis, indices,
            amplitude=config.amplitude, maximum_replicates=1,
            seed=config.seed, stream=config.streams.fit,
        )
        legacy = h.h_initial_starts(store, c_theta, config.amplitude, 1)
        revised = h.h_initial_starts(
            store, c_theta, config.amplitude, 1,
            h.EMPIRICAL_PERTURBATION_POLICY,
        )
        repeated = h.h_initial_starts(
            store, c_theta, config.amplitude, 1,
            h.EMPIRICAL_PERTURBATION_POLICY,
        )
        assert tuple(source for source, _ in revised) == h.H_START_SOURCES
        np.testing.assert_array_equal(revised[0][1], legacy[0][1])
        np.testing.assert_array_equal(revised[1][1], legacy[1][1])
        for (_, theta), (_, again) in zip(revised, repeated):
            np.testing.assert_array_equal(theta, again)
        assert not np.array_equal(revised[2][1], legacy[2][1])

        pool = h.full.PopulationPool(
            data=store.score_data(0, 1, include_outcomes=True),
            single_design=basis.apply(layout.p_design[indices]),
            truth=store.truth, replicates=1,
        )
        codec = h.full.ParameterCodec("H", basis.rank)
        expected = h.full.make_initial_starts(
            codec, pool, config.amplitude, 3,
            h.EMPIRICAL_PERTURBATION_SEED,
            embedded_h_start=legacy[0][1],
        )
        for (_, theta), wanted in zip(revised, expected):
            np.testing.assert_array_equal(theta, wanted)

        # Independently reconstruct the declared empirical perturbation,
        # including projection away from unidentifiable base directions.
        blocks = h.full.empirical_blocks(codec, pool)
        projection = np.linalg.pinv(pool.single_design) @ pool.single_design
        rng = h.full.rng_for(20269193, 702, 1)
        for name, value in blocks.items():
            delta = rng.normal(
                0.0, 0.009 if name == "log_sigma" else 0.035,
                size=value.shape,
            )
            if name in {"base_support", "base_value"}:
                delta = projection @ delta
            blocks[name] = value + delta
        manual = h.full.clip_to_bounds(codec.pack(blocks), codec.bounds())
        np.testing.assert_array_equal(revised[2][1], manual)
        for _, theta in revised:
            assert np.isfinite(theta).all()
            assert all(lower <= x <= upper for x, (lower, upper)
                       in zip(theta, codec.bounds()))
        assert h.full.has_nonzero_group_signature(codec, revised[1][1])
        assert h.full.has_nonzero_group_signature(codec, revised[2][1])

        # Changing the diagnostic truth-based start must not affect the
        # empirical start. The outcome and empirical initializer stay fixed.
        original_truth_initializer = h.full.truth_based_blocks
        def altered_truth_initializer(*positional, **keywords):
            truth_blocks = original_truth_initializer(*positional, **keywords)
            truth_blocks["group_support"] = truth_blocks["group_support"] + 0.01
            return truth_blocks
        h.full.truth_based_blocks = altered_truth_initializer
        try:
            changed_truth = h.h_initial_starts(
                store, c_theta, config.amplitude, 1,
                h.EMPIRICAL_PERTURBATION_POLICY,
            )
        finally:
            h.full.truth_based_blocks = original_truth_initializer
        assert not np.array_equal(changed_truth[1][1], revised[1][1])
        np.testing.assert_array_equal(changed_truth[2][1], revised[2][1])

    assert h.h_attempt_settings(h.LEGACY_START_POLICY) == (
        "population_h_recovery_summary.json", 5601, 5701
    )
    assert h.h_attempt_settings(h.EMPIRICAL_PERTURBATION_POLICY) == (
        "population_h_empirical_perturbation_v1_summary.json", 5602, 5702
    )
    h.assert_h_streams_are_fresh(config.streams, 5602, 5702)
    try:
        h.assert_h_streams_are_fresh(config.streams, 5702, 5702)
    except ValueError:
        pass
    else:
        raise AssertionError("Anchor/verification collision was accepted")
    assert (h.H_MAIN_MAX_ITERATIONS, h.H_POLISH_MAX_ITERATIONS) == (1200, 600)
    actual = json.loads((stage0 / "population_h_recovery_summary.json").read_text())
    assert h.choose_status(actual["acceptance"]) == actual["status"]
    gates = dict.fromkeys(actual["acceptance"], True)
    for gate, expected_status in (
        ("all_scheduled_H_starts_avoid_iteration_cap",
         "revise_population_H_optimizer_iteration_cap"),
        ("fresh_best_two_H_scores_differ_below_threshold",
         "revise_population_H_fresh_start_stability_failed"),
    ):
        assert h.choose_status({**gates, gate: False}) == expected_status
    result = {
        "status": "pass_focused_initializer_regression",
        "uses_real_projection_outcomes": False,
        "optimizer_run": False,
        "bounded_fixture_replicates": 1,
        "unchanged_exact_C_and_truth_starts": True,
        "existing_helper_and_independent_perturbation_match": True,
        "empirical_start_independent_of_truth_initializer_change": True,
        "deterministic_distinct_bounded_nonzero_starts": True,
        "legacy_and_revision_artifact_names_and_streams_preserved": True,
        "caps_and_previous_failed_gate_statuses_preserved": True,
        "H_recovery_demonstrated": False,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
