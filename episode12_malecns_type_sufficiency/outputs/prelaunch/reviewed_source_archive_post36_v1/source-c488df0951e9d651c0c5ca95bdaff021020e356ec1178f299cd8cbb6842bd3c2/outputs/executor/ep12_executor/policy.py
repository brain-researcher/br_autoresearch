"""EP12 policy loading and fail-closed grammar validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .yaml_subset import load_yaml_subset


class PolicyError(ValueError):
    """The policy cannot safely drive this executor."""


class ConfigurationError(ValueError):
    """A proposed trial is outside the frozen EP12 grammar."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_object(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def _at(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise PolicyError(f"missing policy field: {'.'.join(traversed)}")
        value = value[key]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


@dataclass(frozen=True)
class EpisodePolicy:
    path: Path
    raw: Mapping[str, Any]
    policy_hash: str
    policy_id: str
    episode_id: str
    model_triplet: tuple[str, ...]
    meaningful_margin: float
    minimum_valid_trials: int
    maximum_valid_trials: int
    patience_valid_trials: int
    patience_active_after_trial: int
    coverage_minimum: int
    falsifier_fraction_minimum: float
    minimum_adaptive_cycles: int
    minimum_incumbent_decisions: int
    null_replicates: int
    null_pass_threshold: float
    maximum_open_count: int
    cpu_core_hour_ceiling: float
    cpu_cores_per_trial_maximum: int
    memory_gib_per_trial_maximum: float
    wall_hours_per_trial_maximum: float
    gpu_hour_ceiling: float
    terminal_mapping: Mapping[str, str]
    grammar_choices: Mapping[str, tuple[Any, ...]]
    nuisance_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    required_falsifiers: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "EpisodePolicy":
        path = path.resolve()
        content = path.read_bytes()
        raw = load_yaml_subset(path)

        _require(
            _at(raw, "schema_version") == "autoresearch.adaptive_search_policy.v1",
            "unsupported search-policy schema",
        )
        _require(_at(raw, "episode_id") == "ep12", "executor is restricted to EP12")
        _require(_at(raw, "evidence_mode") == "adaptive_search", "wrong evidence mode")
        _require(
            _at(raw, "common_protocol_ref") == "../ADAPTIVE_SEARCH_PROTOCOL.md",
            "unexpected common protocol",
        )
        _require(
            _at(raw, "dataset_roles", "assignment_unit") == "whole_provider_type",
            "EP12 must split by whole provider type",
        )
        _require(
            _at(raw, "dataset_roles", "final_evaluation", "directions")
            == ["left_to_right", "right_to_left"],
            "both reciprocal final directions are required",
        )
        _require(
            _at(raw, "dataset_roles", "final_evaluation", "score_target_side_without_fit_or_refit")
            is True,
            "target-side refitting must be forbidden",
        )
        triplet = tuple(_at(raw, "grammar", "mandatory_model_triplet"))
        _require(
            triplet == ("T_type_template", "U_flexible_unimodal", "M_residual_modes"),
            "the complete T/U/M triplet is mandatory",
        )
        _require(
            _at(raw, "full_search_null", "replicates") == 99,
            "EP12 requires exactly 99 full-search null replicates",
        )
        _require(
            _at(raw, "full_search_null", "rerun_complete_controller") is True,
            "null replicates must rerun the complete controller",
        )
        _require(
            _at(raw, "audit", "configuration_lock_required") is True
            and _at(raw, "audit", "lock_hash_required") is True,
            "final evaluation requires a configuration lock",
        )
        _require(
            _at(raw, "audit", "audit_updates_search") is False,
            "final evaluation may not update search",
        )
        _require(
            _at(raw, "promotion", "incumbent_is_terminal") is False,
            "an incumbent is not a terminal result",
        )

        choice_names = (
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
        )
        grammar_choices: dict[str, tuple[Any, ...]] = {}
        for name in choice_names:
            choices = _at(raw, "grammar", name)
            _require(isinstance(choices, list) and choices, f"empty grammar choice: {name}")
            grammar_choices[name] = tuple(choices)

        terminal_mapping = _at(raw, "terminal_mapping")
        expected_terminals = {
            "candidate_ready_residual_modes",
            "closed_single_population_adequate_within_margin",
            "closed_unresolved",
            "closed_no_valid_comparison",
            "incomplete_search",
            "technical_failure",
        }
        _require(
            isinstance(terminal_mapping, Mapping)
            and set(terminal_mapping) == expected_terminals,
            "terminal mapping is incomplete or contains unknown classes",
        )

        return cls(
            path=path,
            raw=raw,
            policy_hash=digest_bytes(content),
            policy_id=str(_at(raw, "policy_id")),
            episode_id=str(_at(raw, "episode_id")),
            model_triplet=triplet,
            meaningful_margin=float(_at(raw, "decision_rules", "meaningful_margin", "value")),
            minimum_valid_trials=int(_at(raw, "budgets", "minimum_valid_trials")),
            maximum_valid_trials=int(_at(raw, "budgets", "maximum_valid_trials")),
            patience_valid_trials=int(_at(raw, "budgets", "patience_valid_trials")),
            patience_active_after_trial=int(_at(raw, "budgets", "patience_active_after_trial")),
            coverage_minimum=int(_at(raw, "branch_coverage", "minimum_valid_trials")),
            falsifier_fraction_minimum=float(
                _at(raw, "promotion", "post_coverage_falsifier_fraction_min")
            ),
            minimum_adaptive_cycles=int(
                _at(raw, "promotion", "minimum_adaptive_successor_cycles")
            ),
            minimum_incumbent_decisions=int(
                _at(raw, "promotion", "minimum_incumbent_challenger_decisions")
            ),
            null_replicates=int(_at(raw, "full_search_null", "replicates")),
            null_pass_threshold=float(_at(raw, "full_search_null", "pass_threshold")),
            maximum_open_count=int(_at(raw, "audit", "maximum_open_count")),
            cpu_core_hour_ceiling=float(_at(raw, "budgets", "cpu_core_hour_ceiling")),
            cpu_cores_per_trial_maximum=int(
                _at(raw, "budgets", "cpu_cores_per_trial_max")
            ),
            memory_gib_per_trial_maximum=float(
                _at(raw, "budgets", "memory_gib_per_trial_max")
            ),
            wall_hours_per_trial_maximum=float(
                _at(raw, "budgets", "wall_hours_per_trial_max")
            ),
            gpu_hour_ceiling=float(_at(raw, "budgets", "gpu_hour_ceiling")),
            terminal_mapping=dict(terminal_mapping),
            grammar_choices=grammar_choices,
            nuisance_terms=tuple(_at(raw, "grammar", "nuisance")),
            forbidden_terms=tuple(_at(raw, "grammar", "forbidden")),
            required_falsifiers=tuple(_at(raw, "required_falsifiers")),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        required = set(self.grammar_choices) | {"models", "nuisance", "data_mode"}
        missing = sorted(required - set(config))
        if missing:
            raise ConfigurationError(f"missing configuration fields: {', '.join(missing)}")
        extra = sorted(set(config) - required)
        if extra:
            raise ConfigurationError(
                f"unregistered configuration fields: {', '.join(extra)}"
            )
        if tuple(config["models"]) != self.model_triplet:
            raise ConfigurationError("every trial must fit the complete ordered T/U/M triplet")
        if tuple(config["nuisance"]) != self.nuisance_terms:
            raise ConfigurationError("all prespecified nuisance families must remain present")
        if config["data_mode"] != "synthetic":
            raise ConfigurationError("this qualified executor currently accepts synthetic data only")
        for name, choices in self.grammar_choices.items():
            if config[name] not in choices:
                raise ConfigurationError(
                    f"configuration field {name!r} is outside the frozen grammar"
                )

        forbidden_fragments = {
            "provider_group",
            "provider_instance",
            "final_outcome",
            "target_side_fit",
            "arbitrary_code",
        }

        def walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
            yield path, value
            if isinstance(value, Mapping):
                for key, child in value.items():
                    yield from walk(child, path + (str(key),))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    yield from walk(child, path + (str(index),))

        for path, value in walk(config):
            searchable = ".".join(path).lower()
            if isinstance(value, str):
                searchable += "." + value.lower()
            if any(fragment in searchable for fragment in forbidden_fragments):
                raise ConfigurationError(
                    f"forbidden information/operator requested at {'.'.join(path) or '<root>'}"
                )
            if isinstance(value, str) and value in self.forbidden_terms:
                raise ConfigurationError(
                    f"policy-forbidden operation requested at {'.'.join(path) or '<root>'}"
                )

    def contract_summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "model_triplet": list(self.model_triplet),
            "meaningful_margin": self.meaningful_margin,
            "minimum_valid_trials": self.minimum_valid_trials,
            "maximum_valid_trials": self.maximum_valid_trials,
            "coverage_minimum": self.coverage_minimum,
            "patience_valid_trials": self.patience_valid_trials,
            "null_replicates": self.null_replicates,
            "maximum_open_count": self.maximum_open_count,
            "real_data_enabled": False,
        }
