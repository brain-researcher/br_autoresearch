#!/usr/bin/env python3
"""Validate EP20's nonbinding NeuroCam reference/example bundles.

The validator is intentionally standard-library-only so it can run with the
default Sherlock Python. It reads the provisioned JSON files and writes no
episode state or scientific artifact.
"""

import json
import math
from pathlib import Path


SCHEMA_VERSION = "ep20.neurocam_reference_bundle.v1"
BUNDLE_SPECS = {
    "neurocam_paper_direct_v1": {
        "filename": "neurocam_paper_direct_v1.json",
        "role": "documentary_reference_example",
        "estimation_only": False,
    },
    "neurocam_figure_derived_v1": {
        "filename": "neurocam_figure_derived_v1.json",
        "role": "derived_estimation_example",
        "estimation_only": True,
    },
}
ALLOWED_USES = {
    "documentary_context",
    "non_candidate_discriminating_simulation_sanity_check",
}
PROHIBITED_USES = {
    "prior_initialization",
    "model_parameter_prior",
    "search_space_initialization",
    "reference_model_calibration",
    "reference_model_qualification",
    "candidate_scoring",
    "launch_gate_lifting",
    "substitution_for_raw_iv_cv",
    "substitution_for_compact_model",
    "substitution_for_pdk_netlist_or_layout",
    "substitution_for_fabrication_outcomes",
}
FALSE_BOUNDARY_FIELDS = (
    "authoritative_for_reference_fit",
    "authoritative_for_reference_qualification",
    "candidate_score_eligible",
    "foundry_signoff",
    "scientific_outcome_imported",
    "has_raw_iv_cv",
    "has_compact_model",
    "has_pdk",
    "has_fabrication_outcomes",
)


def bundle_directory():
    episode_root = Path(__file__).resolve().parents[2]
    return episode_root / "inputs" / "reference_bundles"


def load_bundle(bundle_id):
    spec = BUNDLE_SPECS[bundle_id]
    path = bundle_directory() / spec["filename"]
    with path.open("r", encoding="utf-8") as stream:
        bundle = json.load(stream)
    if not isinstance(bundle, dict):
        raise ValueError("{} must contain a JSON object".format(path))
    return bundle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def by_name(rows):
    return {row["name"]: row for row in rows}


def validate_boundary(bundle, bundle_id, spec):
    require(bundle.get("schema_version") == SCHEMA_VERSION, "invalid schema_version")
    require(bundle.get("episode_id") == "ep20", "bundle must be scoped to ep20")
    require(bundle.get("bundle_id") == bundle_id, "bundle_id does not match filename")
    require(bundle.get("bundle_role") == spec["role"], "invalid bundle_role")
    require(bundle.get("reference_only") is True, "reference_only must be true")
    require(bundle.get("example_only") is True, "example_only must be true")
    require(
        bundle.get("estimation_only") is spec["estimation_only"],
        "invalid estimation_only flag",
    )
    require(
        bundle.get("launch_gate_effect") == "none", "launch gate effect must be none"
    )
    require(
        bundle.get("calibration_status") == "uncalibrated_reference",
        "bundle must remain uncalibrated",
    )
    for field in FALSE_BOUNDARY_FIELDS:
        require(bundle.get(field) is False, "{} must be false".format(field))
    require(set(bundle.get("allowed_uses", [])) == ALLOWED_USES, "allowed uses changed")
    require(
        set(bundle.get("prohibited_uses", [])) == PROHIBITED_USES,
        "prohibited uses changed",
    )
    contracts = bundle.get("canonical_role_contracts", {})
    require(
        contracts.get("anchor_manifest") == "../NEUROCAM_ANCHORS.yaml",
        "anchor manifest reference changed",
    )
    require(
        contracts.get("anchor_split_manifest") == "../REFERENCE_ANCHOR_SPLIT.yaml",
        "anchor split reference changed",
    )
    require(
        contracts.get("this_bundle_is_canonical") is False,
        "example cannot be canonical",
    )
    source = bundle.get("source", {})
    require(
        source.get("doi") == "10.1016/j.scib.2025.11.030",
        "source DOI changed",
    )
    require(source.get("pmid") == "41309324", "source PMID changed")
    require(
        source.get("source_artifact_included") is False,
        "bundle must not claim the source artifact is included",
    )


def validate_paper_direct(bundle):
    anchors = bundle.get("anchors")
    require(isinstance(anchors, list) and anchors, "paper-direct anchors are required")
    for anchor in anchors:
        require(
            anchor.get("evidence_tier") == "paper_direct", "paper anchor tier changed"
        )
        require(bool(anchor.get("source_anchor")), "paper anchor source is required")
        require(
            isinstance(anchor.get("assumptions"), list),
            "anchor assumptions must be a list",
        )
        require(bool(anchor.get("uncertainty")), "anchor uncertainty is required")


def validate_figure_derived(bundle, paper_bundle):
    source = bundle.get("source", {})
    require(
        source.get("manual_visual_estimates_included") is True,
        "manual estimates must be declared",
    )
    require(
        source.get("raw_digitized_trace_included") is False,
        "raw trace must remain absent",
    )

    estimates = bundle.get("estimates")
    require(isinstance(estimates, list) and estimates, "derived estimates are required")
    for estimate in estimates:
        require(
            estimate.get("evidence_tier")
            in {
                "physics_context_algebraic_estimate",
                "algebraic_schedule_estimate",
            },
            "derived estimate tier changed",
        )
        require(bool(estimate.get("source_anchor")), "estimate source is required")
        require(bool(estimate.get("assumptions")), "estimate assumptions are required")
        require(bool(estimate.get("uncertainty")), "estimate uncertainty is required")

    manual = bundle.get("manual_visual_estimates", {})
    require(
        manual.get("approximate_visual_reading_no_coordinate_trace") is True,
        "manual estimates need no-trace label",
    )
    require(
        manual.get("uncertainty", {}).get("absolute_at_least", 0) >= 0.03,
        "manual uncertainty is too small",
    )
    items = manual.get("items")
    require(isinstance(items, list) and items, "manual visual items are required")
    for item in items:
        require(
            item.get("evidence_tier") == "manual_visual_estimate", "manual tier changed"
        )
        require(
            item.get("approximate_visual_reading_no_coordinate_trace") is True,
            "manual item needs no-trace label",
        )
        require(
            item.get("uncertainty", {}).get("absolute", 0) >= 0.03,
            "manual item uncertainty is too small",
        )
        require(
            bool(item.get("forbidden_interpretation")),
            "manual item needs a forbidden interpretation",
        )

    anchors = by_name(paper_bundle["anchors"])
    derived = by_name(estimates)
    epsilon_0_f_per_cm = 8.8541878128e-14
    thickness_cm = 200e-7
    cox_values = sorted(
        epsilon_0_f_per_cm / (thickness_cm / eps_sinx + thickness_cm / 3.9) * 1e9
        for eps_sinx in (6.5, 8.0)
    )
    cox = derived["oxide_capacitance_areal"]["value"]
    require(
        math.isclose(cox["lower"], cox_values[0], abs_tol=0.1),
        "Cox lower estimate changed",
    )
    require(
        math.isclose(cox["upper"], cox_values[1], abs_tol=0.1),
        "Cox upper estimate changed",
    )

    mobility = anchors["field_effect_mobility"]["value"]
    threshold = anchors["threshold_voltage"]["value"]
    mu_cox_values = [mobility * value * 1e-9 for value in cox_values]
    mu_cox = derived["square_law_mu_cox_coefficient"]["value"]
    require(
        math.isclose(mu_cox["lower"], mu_cox_values[0], rel_tol=0.02),
        "mu*Cox lower estimate changed",
    )
    require(
        math.isclose(mu_cox["upper"], mu_cox_values[1], rel_tol=0.02),
        "mu*Cox upper estimate changed",
    )

    overdrive = 6.0 - threshold
    idsat_values = [0.5 * value * overdrive**2 * 1e6 for value in mu_cox_values]
    idsat = derived["ideal_saturation_current_per_unit_w_over_l"]["value"]
    require(
        math.isclose(idsat["lower"], idsat_values[0], abs_tol=0.02),
        "Idsat lower estimate changed",
    )
    require(
        math.isclose(idsat["upper"], idsat_values[1], abs_tol=0.02),
        "Idsat upper estimate changed",
    )

    ron_values = sorted(1.0 / (value * overdrive) for value in mu_cox_values)
    ron = derived["small_vds_on_resistance_per_unit_l_over_w"]["value"]
    require(
        math.isclose(ron["lower"], ron_values[0], rel_tol=0.04),
        "Ron lower estimate changed",
    )
    require(
        math.isclose(ron["upper"], ron_values[1], rel_tol=0.04),
        "Ron upper estimate changed",
    )

    cycle_seconds = 64 * 62.5e-6
    scan = derived["scan_cycle"]["value"]
    require(math.isclose(scan["cycle_ms"], cycle_seconds * 1e3), "scan cycle changed")
    require(
        math.isclose(scan["per_channel_rate_hz"], 1.0 / cycle_seconds),
        "scan rate changed",
    )
    require(
        math.isclose(scan["duty_cycle_percent"], 62.5e-6 / cycle_seconds * 100),
        "duty cycle changed",
    )

    tau = derived["one_pole_tau_upper_bound"]["value"]["upper_bound"]
    require(
        math.isclose(tau, 1.0 / (2 * math.pi * 1000), abs_tol=1e-6), "tau bound changed"
    )


def validate_all():
    bundles = {}
    for bundle_id, spec in BUNDLE_SPECS.items():
        bundle = load_bundle(bundle_id)
        validate_boundary(bundle, bundle_id, spec)
        bundles[bundle_id] = bundle
    validate_paper_direct(bundles["neurocam_paper_direct_v1"])
    validate_figure_derived(
        bundles["neurocam_figure_derived_v1"],
        bundles["neurocam_paper_direct_v1"],
    )
    return bundles


if __name__ == "__main__":
    validated = validate_all()
    print(json.dumps({"ok": True, "bundle_ids": sorted(validated)}, sort_keys=True))
