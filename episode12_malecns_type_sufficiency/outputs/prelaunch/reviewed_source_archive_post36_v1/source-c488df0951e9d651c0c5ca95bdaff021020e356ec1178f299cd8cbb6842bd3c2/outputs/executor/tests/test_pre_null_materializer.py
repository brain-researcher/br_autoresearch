from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ep12_executor.adaptive_search import coverage_configurations
from ep12_executor.contract_provenance import (
    ACTIVE_EPOCH,
    PREFIX_EPOCH,
    PROVENANCE_SCHEMA,
    RANK_RULE,
)
from ep12_executor.null_trial_runner import _collapse, _fit_options
from ep12_executor.policy import EpisodePolicy, digest_object
from ep12_executor.pre_null_materializer import (
    PreNullMaterializerError,
    fit_selected_mode_and_generators,
)
from ep12_executor.scientific_models import fit_triplet, seed_from
from ep12_executor.scientific_contract_transition import TRANSITION_RECEIPT_SCHEMA
from tests.test_contract_transition import dynamic_attestation_authority


EPISODE_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"
CONTRACT_SHA256 = "a1" * 32


class PreNullSelectedFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = EpisodePolicy.load(POLICY_PATH)
        cls.configuration = dict(coverage_configurations(cls.policy)[0])
        cls.configuration["trial_id"] = "synthetic_selected"
        cls.configuration["stage"] = "adaptive"
        cls.selected = {
            "trial_id": cls.configuration["trial_id"],
            "configuration": cls.configuration,
            "configuration_sha256": digest_object(cls.configuration),
            "contract_epoch": PREFIX_EPOCH,
            "scientific_contract_sha256": "b2" * 32,
        }
        cls.fit_settings = {"integration_draws": 4, "source_cv_folds": 2}
        cls.raw_vocabulary = [
            "typed::x",
            "typed::y",
            "untyped_status::Traced::complete",
            "missing_annotation",
        ]
        cls.hierarchy = {
            "x": {"subclass": "alpha", "class": "sensory"},
            "y": {"subclass": "beta", "class": "sensory"},
        }
        rng = np.random.default_rng(51023)
        ids: list[int] = []
        types: list[str] = []
        sides: list[str] = []
        rows: list[np.ndarray] = []
        for type_index, provider_type in enumerate(("A", "B")):
            for side_index, side in enumerate(("L", "R")):
                for neuron_index in range(6):
                    ids.append(10_000 + len(ids))
                    types.append(provider_type)
                    sides.append(side)
                    component = neuron_index % 2
                    base = np.asarray(
                        [
                            20 + 8 * component + 2 * type_index,
                            23 + 8 * (1 - component) + side_index,
                            4 + type_index,
                            2 + side_index,
                        ],
                        dtype=float,
                    )
                    rows.append(rng.poisson(base).astype(np.int64) + 1)
        cls.neuron_ids = ids
        cls.focal_types = types
        cls.sides = sides
        cls.raw_counts = np.vstack(rows)
        coordinate = np.linspace(-0.8, 0.8, len(ids))
        cls.covariates = {
            int(spline_df): np.column_stack(
                (
                    coordinate,
                    np.log1p(cls.raw_counts.sum(axis=1)) / 10.0,
                    np.asarray([0.0 if side == "L" else 1.0 for side in sides]),
                )
            )
            for spline_df in cls.policy.grammar_choices["nuisance_spline_df"]
        }

    def _durable_rows(self) -> list[dict[str, object]]:
        configuration = self.configuration
        focal_types = np.asarray(self.focal_types, dtype=object)
        sides = np.asarray(self.sides, dtype=object)
        collapsed, _, _ = _collapse(
            self.raw_counts,
            focal_types,
            self.raw_vocabulary,
            configuration,
            self.hierarchy,
        )
        covariates = self.covariates[int(configuration["nuisance_spline_df"])]
        options = _fit_options(configuration, self.fit_settings)
        rows: list[dict[str, object]] = []
        for provider_type in ("A", "B"):
            type_mask = focal_types == provider_type
            left = type_mask & (sides == "L")
            right = type_mask & (sides == "R")
            base = seed_from(
                "ep12-development", configuration["trial_id"], provider_type
            )
            left_fit = fit_triplet(
                collapsed[left],
                covariates[left],
                seed=seed_from(base, "left-source"),
                **options,
            )
            right_fit = fit_triplet(
                collapsed[right],
                covariates[right],
                seed=seed_from(base, "right-source"),
                **options,
            )
            rows.append(
                {
                    "provider_type": provider_type,
                    "left_to_right_delta": left_fit.score_target(
                        collapsed[right], covariates[right]
                    )["mean_delta_M_minus_reference"],
                    "right_to_left_delta": right_fit.score_target(
                        collapsed[left], covariates[left]
                    )["mean_delta_M_minus_reference"],
                }
            )
        return rows

    def _write_table(self, path: Path, *, tamper: float = 0.0) -> None:
        rows = self._durable_rows()
        rows[0]["left_to_right_delta"] = (
            float(rows[0]["left_to_right_delta"]) + tamper
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "provider_type",
                    "left_to_right_delta",
                    "right_to_left_delta",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _fit(self, table: Path, *, rank_rule: str):
        authority = dynamic_attestation_authority()
        return fit_selected_mode_and_generators(
            policy=self.policy,
            selected=self.selected,
            selected_result_table=table,
            contract_sha256=CONTRACT_SHA256,
            contract_provenance={
                "schema_version": PROVENANCE_SCHEMA,
                "prefix_contract": {
                    "epoch": PREFIX_EPOCH, "sha256": "b2" * 32,
                },
                "active_contract": {
                    "epoch": ACTIVE_EPOCH, "sha256": CONTRACT_SHA256,
                },
                "transition": {
                    "schema_version": TRANSITION_RECEIPT_SCHEMA,
                    "complete": True, "semantic_sha256": "c3" * 32,
                    "file_sha256": "d4" * 32, "spec_sha256": "e5" * 32,
                    "spec_file_sha256": "f6" * 32,
                    "observed_attestation_authority": authority,
                    "controller_final_receipt_required_before_irreversible_transition": True,
                    "authoritative_attestation_job_id": authority[
                        "authoritative_job_id"
                    ],
                    "obsolete_attestation_job_id": authority["obsolete_job_id"],
                },
                "prevalence_parameter_rank_definition": RANK_RULE,
                "operator_development_outcome_values_displayed": True,
                "values_used_for_transition_decisions": False,
                "final_connectivity_accessed": False,
            },
            numerical_replay_tolerance=1e-10,
            prevalence_parameter_rank_definition=rank_rule,
            neuron_ids=self.neuron_ids,
            focal_types=self.focal_types,
            sides=self.sides,
            raw_counts=self.raw_counts,
            raw_vocabulary=self.raw_vocabulary,
            partner_hierarchy=self.hierarchy,
            covariates_by_spline_df=self.covariates,
            fit_settings=self.fit_settings,
        )

    def test_exact_refit_emits_whole_type_evidence_and_per_side_generators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "result_table.csv"
            self._write_table(table)
            mode, selection, design, generators = self._fit(
                table,
                rank_rule="representation_latent_effective_rank",
            )
        self.assertTrue(mode["complete_input"])
        self.assertEqual(
            mode["whole_provider_type_inputs"]["whole_provider_type_ids"],
            ["A", "B"],
        )
        self.assertEqual(set(generators), {("A", "L"), ("A", "R"), ("B", "L"), ("B", "R")})
        self.assertEqual(len(selection["selected_generator_by_type_side"]), 4)
        self.assertEqual(len(design.neuron_ids), len(self.neuron_ids))
        self.assertFalse(mode["final_connectivity_accessed"])

    def test_durable_gain_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "result_table.csv"
            self._write_table(table, tamper=1e-3)
            with self.assertRaisesRegex(
                PreNullMaterializerError, "does not replay durable gains"
            ):
                self._fit(
                    table,
                    rank_rule="representation_latent_effective_rank",
                )

    def test_unfrozen_parameter_rank_definition_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "result_table.csv"
            self._write_table(table)
            with self.assertRaisesRegex(
                PreNullMaterializerError, "not the frozen post-36 choice"
            ):
                self._fit(table, rank_rule="not_a_frozen_scientific_definition")


if __name__ == "__main__":
    unittest.main()
