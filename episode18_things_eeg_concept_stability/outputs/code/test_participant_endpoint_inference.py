"""Three deterministic formula/boundary checks; no scientific simulations."""
from dataclasses import FrozenInstanceError, replace
import math
import unittest

import numpy as np
from scipy.special import stdtrit

import participant_endpoint_inference as inference


def inputs(role, margins=None):
    n = inference.ROLE_SIZES[role]
    margins = inference.Margins(2.5, .5) if margins is None else margins
    thresholds = (margins.delta_C,) + (margins.delta_P,) * 8
    return dict(role=role, participant_ids=tuple(f"{role}_{i}" for i in range(n)),
        primary={name: np.full(n, 3*value) for name, value in zip(inference.PRIMARY, thresholds)},
        positive={name: np.full(n, value+.125) for name, value in zip(inference.POSITIVE,
            (margins.delta_Q, margins.delta_T, 0.))},
        negative={name: np.zeros(n) for name in inference.NEGATIVE},
        linear_label=np.full(n, margins.delta_C+.125), margins=margins)


class ParticipantEndpointInferenceTests(unittest.TestCase):
    def test_registered_familywise_quantiles_and_participant_standard_errors(self):
        development = inference.infer_participant_endpoints(**inputs("development"))
        for role in ("development", "audit"):
            args = inputs(role)
            n = len(args["participant_ids"])
            centered = np.arange(n, dtype=float) - (n-1)/2
            for names, key in ((inference.PRIMARY, "primary"), (inference.POSITIVE, "positive"),
                               (inference.NEGATIVE, "negative")):
                args[key] = {name: .125*(index+1) + centered*.003*(index+1)
                             for index, name in enumerate(names)}
            args["linear_label"] = centered*.004+.25
            if role == "audit":
                args["development_evidence"] = development
            actual = inference.infer_participant_endpoints(**args)
            for result, names, values, tails in (
                (actual.primary_positive, inference.PRIMARY, args["primary"], 1),
                (actual.primary_classification, inference.PRIMARY, args["primary"], 2),
                (actual.positive_controls, inference.POSITIVE, args["positive"], 1),
                (actual.negative_controls, inference.NEGATIVE, args["negative"], 2),
                (actual.linear_label, inference.LINEAR, {inference.LINEAR[0]:args["linear_label"]}, 1)):
                tail = .05/(len(names)*tails)
                critical = float(stdtrit(n-1, 1-tail))
                self.assertEqual(result.endpoints, names)
                self.assertEqual(result.participant_count, n)
                self.assertEqual(result.degrees_of_freedom, n-1)
                self.assertEqual(result.tail_alpha, tail)
                self.assertAlmostEqual(result.critical_t, critical, places=12)
                for index, name in enumerate(names):
                    column = list(map(float, values[name]))
                    mean = math.fsum(column)/n
                    sd = math.sqrt(math.fsum((value-mean)**2 for value in column)/(n-1))
                    se = sd/math.sqrt(n)
                    self.assertAlmostEqual(result.mean[index], mean, places=13)
                    self.assertAlmostEqual(result.sample_sd[index], sd, places=13)
                    self.assertAlmostEqual(result.standard_error[index], se, places=13)
                    self.assertAlmostEqual(result.lower[index], mean-critical*se, places=12)
                    if tails == 2:
                        self.assertAlmostEqual(result.upper[index], mean+critical*se, places=12)
                    else:
                        self.assertIsNone(result.upper)
            # One-sided primary bounds must not silently use the two-sided cutoff.
            self.assertGreater(actual.primary_positive.lower[0], actual.primary_classification.lower[0])

    def test_strict_margin_edges_zero_variance_and_conditional_classifications(self):
        margins = inference.Margins(2.5, .5)  # C=.125, P=.0625, Q=.625, T=.125.
        dev_args = inputs("development", margins)
        dev_args["primary"][inference.PRIMARY[0]][:] = margins.delta_C
        dev_args["positive"][inference.POSITIVE[0]][:] = margins.delta_Q
        dev_args["negative"][inference.NEGATIVE[0]][:] = margins.delta_P
        boundary = inference.infer_participant_endpoints(**dev_args)
        self.assertEqual(boundary.primary_positive.sample_sd, (0.,)*9)
        self.assertEqual(boundary.primary_classification.lower, boundary.primary_classification.upper)
        self.assertEqual(boundary.numerical.development_mean_relation, "at_least_one_mean_at_most_margin")
        self.assertEqual(boundary.numerical.positive_lower_minus_threshold[0], 0.)
        self.assertEqual(boundary.numerical.negative_control_relation, "all_upper_bounds_at_most_margin")
        self.assertIn("feature_and_readout_sufficiency", boundary.numerical.diagnostic_rule_matches)
        self.assertIn("readout_insufficiency", boundary.numerical.diagnostic_rule_matches)
        dev_args["linear_label"][:] = margins.delta_C
        self.assertNotIn("readout_insufficiency",
            inference.infer_participant_endpoints(**dev_args).numerical.diagnostic_rule_matches)
        dev_args["negative"][inference.NEGATIVE[0]][:] = margins.delta_P+.125
        self.assertEqual(inference.infer_participant_endpoints(**dev_args).numerical.negative_control_relation,
                         "lower_bound_exceeds_margin")
        dev_args["negative"][inference.NEGATIVE[0]] = margins.delta_P + np.linspace(-.125,.125,30)
        self.assertEqual(inference.infer_participant_endpoints(**dev_args).numerical.negative_control_relation,
                         "upper_bound_exceeds_margin_without_artifact_bound")

        development = inference.infer_participant_endpoints(**inputs("development", margins))
        audit_args = inputs("audit", margins)
        audit_args["development_evidence"] = development
        audit_args["primary"][inference.PRIMARY[0]][:] = margins.delta_C
        audit_args["primary"][inference.PRIMARY[1]][:] = -margins.delta_P
        audit_args["primary"][inference.PRIMARY[2]][:] = -margins.delta_P-.125
        result = inference.infer_participant_endpoints(**audit_args)
        self.assertEqual(result.numerical.partition_equivalent_indices, (1,))
        self.assertEqual(result.numerical.partition_better_than_true_indices, (2,))
        self.assertEqual(result.numerical.audit_effect_rule_matches,
            ("new_participant_nonreplication", "readout_insufficiency",
             "feature_and_readout_sufficiency", "partition_nonspecificity"))
        dev_args = inputs("development", margins)
        dev_args["primary"][inference.PRIMARY[0]][:] = 2*margins.delta_C
        audit_args["development_evidence"] = inference.infer_participant_endpoints(**dev_args)
        self.assertNotIn("new_participant_nonreplication",
            inference.infer_participant_endpoints(**audit_args).numerical.audit_effect_rule_matches)
        strong = inputs("audit", margins)
        strong["development_evidence"] = development
        self.assertEqual(inference.infer_participant_endpoints(**strong).numerical.audit_effect_rule_matches,
                         ("primary_positive_lower_bound_conjunction",))

    def test_complete_families_rosters_calibration_and_no_missing_data_repair(self):
        floor = inference.Margins(-1., -2.)
        self.assertEqual((floor.delta_C,floor.delta_P,floor.delta_Q,floor.delta_T), (.001,.0005,.002,.001))
        self.assertEqual(floor.G, -1.)  # Numerical evidence does not invent positive calibration.
        with self.assertRaises(TypeError):
            inference.Margins(1., 1., delta_C=0.)
        with self.assertRaises(FrozenInstanceError):
            floor.delta_C = 0.
        for invalid in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                inference.Margins(invalid, 1.)
        for family in ("primary", "positive", "negative"):
            args = inputs("development")
            del args[family][next(iter(args[family]))]
            with self.assertRaisesRegex(ValueError, "complete_exact_endpoint_family_required"):
                inference.infer_participant_endpoints(**args)
        for invalid in (np.full(30,np.nan), np.full(30,np.inf), np.ones(29), np.ones(30,dtype=bool)):
            args = inputs("development")
            args["primary"][inference.PRIMARY[0]] = invalid
            with self.assertRaises(ValueError):
                inference.infer_participant_endpoints(**args)
        for people in (tuple(f"p{i}" for i in range(29)), ("same",)*30):
            args = inputs("development")
            args["participant_ids"] = people
            with self.assertRaisesRegex(ValueError, "complete_unique_assigned_participant_roster_required"):
                inference.infer_participant_endpoints(**args)
        development = inference.infer_participant_endpoints(**inputs("development"))
        for previous in (None, replace(development,margins=inference.Margins(3.,.5))):
            args = inputs("audit")
            args["development_evidence"] = previous
            with self.assertRaisesRegex(ValueError, "same_margin_disjoint_complete_development_evidence_required"):
                inference.infer_participant_endpoints(**args)
        args = inputs("audit")
        args["development_evidence"] = development
        args["participant_ids"] = development.participant_ids[:16]
        with self.assertRaisesRegex(ValueError, "same_margin_disjoint_complete_development_evidence_required"):
            inference.infer_participant_endpoints(**args)
        args = inputs("development")
        result = inference.infer_participant_endpoints(**args)
        old = result.primary_positive.mean
        args["primary"][inference.PRIMARY[0]][:] = -99
        self.assertEqual(result.primary_positive.mean, old)
        self.assertFalse(hasattr(result, "terminal_outcome"))
        self.assertFalse(hasattr(result, "scientific_release_authorized"))


if __name__ == "__main__":
    unittest.main()
