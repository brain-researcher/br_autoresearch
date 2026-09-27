"""Executable development-only adaptive-search plan for EP12."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from .policy import EpisodePolicy, digest_object
from .runtime import EPISODE_ROOT, atomic_json


BUDGET_OVERRIDE_PATH = (
    EPISODE_ROOT / "outputs" / "executor" / "DEVELOPMENT_BUDGET_OVERRIDE.yaml"
)


def validate_development_configuration(
    policy: EpisodePolicy,
    configuration: dict[str, Any],
) -> None:
    """Fail closed against the frozen grammar while allowing development data."""

    required = {
        "trial_id",
        "stage",
        "models",
        "partner_vocabulary",
        "partner_minimum_development_type_count",
        "rare_partner_pooling_mass_fraction",
        "partner_hierarchy_depth",
        "representation",
        "rank",
        "composition_pseudocount",
        "nuisance",
        "nuisance_spline_df",
        "covariance",
        "covariance_rank",
        "component_count",
        "regularization",
        "data_mode",
    }
    if set(configuration) != required:
        missing = sorted(required - set(configuration))
        extra = sorted(set(configuration) - required)
        raise ValueError(f"development configuration keys differ: missing={missing}, extra={extra}")
    if configuration["data_mode"] != "development":
        raise ValueError("real adaptive tasks may use development data only")
    if list(configuration["models"]) != list(policy.model_triplet):
        raise ValueError("every trial must fit the complete frozen T/U/M triplet")
    if list(configuration["nuisance"]) != list(policy.nuisance_terms):
        raise ValueError("all models must share the frozen nuisance terms")
    for key in (
        "partner_vocabulary",
        "partner_minimum_development_type_count",
        "rare_partner_pooling_mass_fraction",
        "partner_hierarchy_depth",
        "representation",
        "rank",
        "composition_pseudocount",
        "nuisance_spline_df",
        "covariance",
        "covariance_rank",
        "component_count",
        "regularization",
    ):
        if configuration[key] not in policy.grammar_choices[key]:
            raise ValueError(f"{key} is outside the SEARCH_POLICY grammar")
    if configuration["stage"] not in {"coverage", "adaptive"}:
        raise ValueError("development trial stage must be coverage or adaptive")


def coverage_configurations(policy: EpisodePolicy) -> list[dict[str, Any]]:
    """Build the frozen 18-row real covering array and spread secondary knobs."""

    vocabularies = policy.grammar_choices["partner_vocabulary"]
    representations = policy.grammar_choices["representation"]
    covariances = policy.grammar_choices["covariance"]
    components = policy.grammar_choices["component_count"]
    if not (len(vocabularies) == len(representations) == len(covariances) == 3):
        raise ValueError("coverage construction requires three 3-level primary factors")
    if len(components) != 2:
        raise ValueError("coverage construction requires the declared two K levels")

    rows: list[dict[str, Any]] = []
    for row_index, (vocabulary_index, representation_index, component_index) in enumerate(
        itertools.product(range(3), range(3), range(2))
    ):
        covariance_index = (
            vocabulary_index + representation_index + component_index
        ) % 3
        configuration = {
            "trial_id": f"coverage_{row_index + 1:03d}",
            "stage": "coverage",
            "models": list(policy.model_triplet),
            "partner_vocabulary": vocabularies[vocabulary_index],
            "partner_minimum_development_type_count": policy.grammar_choices[
                "partner_minimum_development_type_count"
            ][(vocabulary_index + 2 * representation_index + component_index) % 3],
            "rare_partner_pooling_mass_fraction": policy.grammar_choices[
                "rare_partner_pooling_mass_fraction"
            ][(2 * vocabulary_index + representation_index + component_index) % 3],
            "partner_hierarchy_depth": policy.grammar_choices[
                "partner_hierarchy_depth"
            ][(vocabulary_index + representation_index + component_index) % 2],
            "representation": representations[representation_index],
            "rank": policy.grammar_choices["rank"][
                (row_index + representation_index) % len(policy.grammar_choices["rank"])
            ],
            "composition_pseudocount": policy.grammar_choices[
                "composition_pseudocount"
            ][(vocabulary_index + representation_index + 2 * component_index) % 3],
            "nuisance": list(policy.nuisance_terms),
            "nuisance_spline_df": policy.grammar_choices["nuisance_spline_df"][
                (vocabulary_index + 2 * representation_index + component_index) % 3
            ],
            "covariance": covariances[covariance_index],
            "covariance_rank": policy.grammar_choices["covariance_rank"][
                (2 * vocabulary_index + representation_index + component_index) % 3
            ],
            "component_count": components[component_index],
            "regularization": policy.grammar_choices["regularization"][row_index % 6],
            "data_mode": "development",
        }
        validate_development_configuration(policy, configuration)
        rows.append(configuration)
    return rows


def coverage_proof(policy: EpisodePolicy, rows: list[dict[str, Any]]) -> dict[str, Any]:
    factors = ("partner_vocabulary", "representation", "covariance", "component_count")
    pair_results: dict[str, Any] = {}
    all_complete = True
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1 :]:
            observed = {(row[left], row[right]) for row in rows}
            expected = set(
                itertools.product(policy.grammar_choices[left], policy.grammar_choices[right])
            )
            complete = observed == expected
            all_complete &= complete
            pair_results[f"{left}__{right}"] = {
                "complete": complete,
                "observed_pair_count": len(observed),
                "expected_pair_count": len(expected),
            }
    return {"all_required_pairs_covered": all_complete, "pairs": pair_results}


def prepare_coverage_plan(*, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    allowed_output = (EPISODE_ROOT / "outputs").resolve()
    if output_dir != allowed_output and allowed_output not in output_dir.parents:
        raise ValueError("search plan must stay under this EP12 outputs root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"search plan directory is not empty: {output_dir}")
    policy_path = EPISODE_ROOT / "SEARCH_POLICY.yaml"
    policy = EpisodePolicy.load(policy_path)
    rows = coverage_configurations(policy)
    proof = coverage_proof(policy, rows)
    if not proof["all_required_pairs_covered"]:
        raise RuntimeError("real coverage plan does not cover every required pair")
    if not BUDGET_OVERRIDE_PATH.is_file():
        raise FileNotFoundError("the explicit development budget override is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_path = output_dir / "coverage_configurations.json"
    atomic_json(configuration_path, rows)
    report = {
        "phase": "observed_development_coverage",
        "configuration_count": len(rows),
        "policy_minimum_valid_trials": policy.minimum_valid_trials,
        "coverage_minimum": policy.coverage_minimum,
        "coverage_proof": proof,
        "configuration_sha256": [digest_object(row) for row in rows],
        "budget_interpretation": (
            "the original 1000-core-hour stop is disabled for completing the minimum "
            "full search; 2750 core-hours is the user-authorized planning scale"
        ),
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    atomic_json(output_dir / "search_plan.json", report)
    return report


def load_coverage_configuration(*, plan_dir: Path, index: int) -> dict[str, Any]:
    rows = json.loads(
        (plan_dir.resolve() / "coverage_configurations.json").read_text(encoding="utf-8")
    )
    if index < 0 or index >= len(rows):
        raise IndexError(f"coverage index {index} is outside 0..{len(rows) - 1}")
    policy = EpisodePolicy.load(EPISODE_ROOT / "SEARCH_POLICY.yaml")
    configuration = dict(rows[index])
    validate_development_configuration(policy, configuration)
    return configuration
