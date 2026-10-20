"""Focused EP16 seal/freeze checks using synthetic fixtures only.

Run from outputs/code:
    python3 -m unittest -v test_ep16_seal_freeze
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import ep16_data
import ep16_panel
import ep16_roles
import evaluate_t9_sealed as sealed


def valid_choices():
    """Return fresh, consistent synthetic specification, panel and control."""
    panel = {
        "status": "selected_by_development_search",
        "normalization": "block",
        "readiness": "strict",
        "supervised_32": sorted(sealed.br.SUPERVISED)[0],
        "emulator": "E1_dropout",
        "unlabelled": {"method": "past_only", "settings": {"history": 16}},
    }
    values = {
        "ridge_lambda_sbp": 1.0,
        "ridge_lambda_tx_4_5": 2.0,
        "ridge_lambda_sbp_frozen_scaling": 3.0,
        "J_min": 0.01,
        "D_min": 0.02,
        "raw_score_decline_margin": -0.05,
    }
    spec = {
        "scientist_decisions": {
            "normalization": panel["normalization"],
            "readiness": panel["readiness"],
        },
        "proposed_pending_scientist": [],
        "uncertainty": {"draws": 99, "sensitivity_draws": 19, "seed": 16},
        "values_from_T11": values,
    }
    control = {
        "participant": "T11",
        "pass": True,
        "normalization": panel["normalization"],
        "readiness": panel["readiness"],
        "draws": spec["uncertainty"]["draws"],
        "values": values.copy(),
    }
    return spec, panel, control


def write_json(path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class ChoiceValidationTests(unittest.TestCase):
    def test_complete_valid_choices_have_no_errors(self):
        spec, panel, control = valid_choices()
        self.assertEqual(sealed.panel_choice_errors(panel), [])
        self.assertEqual(sealed.frozen_choice_errors(spec, panel, control), [])

    def test_unregistered_panel_choices_are_rejected_by_both_validators(self):
        cases = (
            (
                "normalization",
                "blocks",
                "panel normalization must be block or none",
            ),
            (
                "readiness",
                "unknown",
                "panel readiness must be strict or screened",
            ),
            (
                "supervised_32",
                "__unregistered_supervised__",
                "panel supervised_32 is not a registered supervised method",
            ),
            (
                "emulator",
                "__unregistered_emulator__",
                "panel emulator is not a selectable frozen emulator",
            ),
            (
                "unlabelled",
                {"method": "past_only", "settings": {"history": 32}},
                "panel unlabelled method/settings are not a registered choice",
            ),
        )
        for key, value, expected in cases:
            with self.subTest(field=key):
                spec, panel, control = valid_choices()
                panel[key] = value
                if key in ("normalization", "readiness"):
                    # Isolate registration failure from consistency failures.
                    spec["scientist_decisions"][key] = value
                    control[key] = value
                self.assertIn(expected, sealed.panel_choice_errors(panel))
                self.assertIn(
                    expected, sealed.frozen_choice_errors(spec, panel, control)
                )

    def test_missing_scientist_decisions_is_rejected(self):
        spec, panel, control = valid_choices()
        del spec["scientist_decisions"]
        self.assertIn(
            "scientist_decisions must record normalization and readiness",
            sealed.frozen_choice_errors(spec, panel, control),
        )

    def test_nonempty_pending_decisions_are_rejected_recursively(self):
        fragments = (
            {"proposed_pending_scientist": ["normalization"]},
            {
                "metadata": {
                    "nested": {
                        "proposed_pending_scientist": {"readiness": "pending"}
                    }
                }
            },
            {
                "metadata": [
                    {"nested": [[{"proposed_pending_scientist": "pending"}]]}
                ]
            },
        )
        for index, fragment in enumerate(fragments):
            with self.subTest(location=index):
                spec, panel, control = valid_choices()
                spec.update(fragment)
                self.assertIn(
                    "specification has non-empty proposed_pending_scientist",
                    sealed.frozen_choice_errors(spec, panel, control),
                )

        spec, panel, control = valid_choices()
        spec["values_from_T11"]["evidence"] = {
            "proposed_pending_scientist": ["review"]
        }
        self.assertIn(
            "specification has non-empty proposed_pending_scientist",
            sealed.frozen_choice_errors(spec, panel, control),
        )

    def test_control_readiness_and_normalization_must_match(self):
        for key, value in (
            ("readiness", "screened"),
            ("normalization", "none"),
        ):
            with self.subTest(field=key):
                spec, panel, control = valid_choices()
                control[key] = value
                self.assertIn(
                    f"T11 positive-control {key} differs from the selected panel",
                    sealed.frozen_choice_errors(spec, panel, control),
                )

    def test_control_draws_must_match(self):
        spec, panel, control = valid_choices()
        control["draws"] += 1
        self.assertIn(
            "T11 positive-control draws differ from uncertainty.draws",
            sealed.frozen_choice_errors(spec, panel, control),
        )

    def test_each_frozen_value_must_match(self):
        for key in sealed.FROZEN_VALUES:
            with self.subTest(value=key):
                spec, panel, control = valid_choices()
                control["values"][key] = spec["values_from_T11"][key] + 1.0
                self.assertIn(
                    f"T11 positive-control value {key} differs from the specification",
                    sealed.frozen_choice_errors(spec, panel, control),
                )

    def test_extra_values_metadata_need_not_appear_in_control(self):
        spec, panel, control = valid_choices()
        spec["values_from_T11"].update(
            evidence={"source": "synthetic unit-test fixture"},
            fixed_utc="2000-01-01T00:00:00+00:00",
        )
        self.assertNotIn("evidence", control["values"])
        self.assertNotIn("fixed_utc", control["values"])
        self.assertEqual(sealed.frozen_choice_errors(spec, panel, control), [])


class SealRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(
            self.enterContext(
                tempfile.TemporaryDirectory(prefix="ep16-seal-freeze-")
            )
        )
        self.root = self.temp_root / "t9_evaluation"
        self.manifest = {"fixture": "synthetic manifest; no archive"}
        self.manifest_path = self.temp_root / "manifest.json"
        write_json(self.manifest_path, self.manifest)

        # Both modules must use the temporary anchor before any seal call.
        self.enterContext(patch.object(sealed, "EVALUATION_ROOT", self.root))
        self.enterContext(patch.object(ep16_roles, "EVALUATION_ROOT", self.root))
        self.verify_manifest = self.enterContext(
            patch.object(
                sealed,
                "verify_manifest",
                return_value=(self.manifest, {}, {}, {}),
            )
        )
        self.verify_archive = self.enterContext(
            patch.object(sealed, "verify_archive", return_value=None)
        )
        self.enterContext(
            patch.object(sealed.secrets, "token_hex", return_value="0" * 32)
        )

    def open_attempt(self, number):
        return sealed.open_seal(
            self.root / f"attempt{number:02d}", self.manifest_path
        )

    def write_failure(self, number, no_results_written):
        write_json(
            self.root / f"attempt{number:02d}" / "failure.json",
            {"no_results_written": no_results_written},
        )

    def test_first_attempt_must_be_attempt01(self):
        with self.assertRaisesRegex(SystemExit, r"must be .*attempt01$"):
            self.open_attempt(2)
        self.assertFalse((self.root / "attempt02").exists())
        self.assertFalse((self.root / "attempt01").exists())

    def test_first_attempt_creates_record_and_token_passes_check(self):
        token = self.open_attempt(1)
        record_path = self.root / "attempt01" / "t9_opened.json"

        self.assertIsInstance(token, ep16_roles.SealedEvaluation)
        self.assertEqual(token.output_dir, self.root / "attempt01")
        self.assertTrue(record_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["participant"], "T9")
        self.assertEqual(record["nonce"], token.nonce)
        self.assertIsNone(token.check())

        self.verify_manifest.assert_called_once_with(self.manifest_path)
        self.verify_archive.assert_called_once_with(self.manifest)

    def test_second_call_without_failure_record_is_refused(self):
        self.open_attempt(1)
        with self.assertRaisesRegex(SystemExit, "no readable failure record"):
            self.open_attempt(2)
        self.assertFalse((self.root / "attempt02").exists())

    def test_clean_failure_allows_attempt02_but_not_skipping_to_attempt03(self):
        self.open_attempt(1)
        self.write_failure(1, True)

        # Check the wrong number before creating attempt02.
        with self.assertRaisesRegex(SystemExit, r"must be .*attempt02$"):
            self.open_attempt(3)
        self.assertFalse((self.root / "attempt03").exists())

        token = self.open_attempt(2)
        self.assertEqual(token.output_dir, self.root / "attempt02")
        self.assertTrue((token.output_dir / "t9_opened.json").is_file())
        self.assertIsNone(token.check())

    def test_failure_reporting_results_written_refuses_restart(self):
        self.open_attempt(1)
        self.write_failure(1, False)

        with self.assertRaisesRegex(SystemExit, "does not permit a restart"):
            self.open_attempt(2)
        self.assertFalse((self.root / "attempt02").exists())

    def test_primary_results_presence_overrides_clean_failure_record(self):
        self.open_attempt(1)
        self.write_failure(1, True)
        write_json(self.root / "attempt01" / "primary_results.json", {})

        with self.assertRaisesRegex(SystemExit, "does not permit a restart"):
            self.open_attempt(2)
        self.assertFalse((self.root / "attempt02").exists())

    def test_check_rejects_paths_outside_direct_canonical_attempt_children(self):
        invalid_paths = (
            self.temp_root / "outside" / "attempt01",
            self.root / "nested" / "attempt01",
            self.root / "not_an_attempt",
            self.root / "attempt1",
            self.root / "attempt00",
            self.root / "attempt001",
        )
        for path in invalid_paths:
            with self.subTest(path=str(path.relative_to(self.temp_root))):
                path.mkdir(parents=True, exist_ok=True)
                nonce = "matching-test-nonce"
                # A matching record isolates the path-validation requirement.
                write_json(
                    path / ep16_roles.OPENED_RECORD,
                    {"participant": "T9", "nonce": nonce},
                )
                token = ep16_roles.SealedEvaluation(
                    ep16_roles._ISSUER, "T9", path, nonce
                )
                with self.assertRaisesRegex(
                    ep16_roles.RoleSealError, "direct attemptNN child"
                ):
                    token.check()


class GuardAndDecisionTests(unittest.TestCase):
    def test_readiness_nodes_rejects_unknown_rule(self):
        with self.assertRaisesRegex(ValueError, "Unknown readiness rule"):
            sealed.readiness_nodes(
                [], {"readiness": "unknown"}, lam=1.0, J_min=0.01
            )

    def test_assemble_rejects_unknown_normalization_with_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unknown normalization"):
            ep16_data.assemble({}, {}, normalization="blocks")

    def test_finite_one_sided_bounds(self):
        values = np.array([3.0, 1.0, 4.0, 2.0])
        # alpha * (B + 1) = 1, selecting the relevant extreme.
        self.assertEqual(
            ep16_panel.one_sided_bound(values, 0.2, "lower"), 1.0
        )
        self.assertEqual(
            ep16_panel.one_sided_bound(values, 0.2, "upper"), 4.0
        )

    def test_each_infinity_sign_is_nonestimable_in_both_bound_directions(self):
        # Keep signs separate: combining them could conceal a NaN-only guard.
        for nonfinite in (np.inf, -np.inf, np.nan):
            for side, expected in (("lower", -np.inf), ("upper", np.inf)):
                with self.subTest(nonfinite=nonfinite, side=side):
                    values = np.array([1.0, 2.0, 3.0, nonfinite])
                    self.assertEqual(
                        ep16_panel.one_sided_bound(values, 0.2, side),
                        expected,
                    )

    def test_nonestimable_observed_points_cannot_be_resolved_by_finite_bounds(self):
        for side, bound in (("lower", 1.0), ("upper", -1.0)):
            with self.subTest(side=side, point="finite"):
                self.assertEqual(
                    ep16_panel.observed_decision(bound, bound, 0.0, side),
                    "supported",
                )
            for point in (None, np.nan):
                with self.subTest(side=side, point=point):
                    self.assertEqual(
                        ep16_panel.observed_decision(point, bound, 0.0, side),
                        "unresolved",
                    )


if __name__ == "__main__":
    unittest.main()
