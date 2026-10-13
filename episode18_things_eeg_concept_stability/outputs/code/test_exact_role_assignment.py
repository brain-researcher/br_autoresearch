"""Exhaustive small-cohort oracle for all three role-assignment objective tiers."""
from fractions import Fraction
import itertools
import unittest

from exact_role_assignment import ExactAssignment, features_from_rows


class ExactRoleTests(unittest.TestCase):
    def test_matches_exhaustive_three_tier_optimum(self):
        rows = [{"participant_id": f"sub-{index+1:02d}", "age_source": str(age),
                 "gender": "F" if index % 2 else "M", "native_english": "Yes" if index < 5 else "No",
                 "language_profile": "Bilingual" if index % 3 else "Monolingual",
                 "language_spoken_at_home": "English" if index % 2 else "Chinese"}
                for index, age in enumerate([18, 19, 20, 21, 18, 19, 20, 21])]
        ids, features, constraints, _ = features_from_rows(rows, [[row["participant_id"] for row in rows[:4]],
                                                                  [row["participant_id"] for row in rows[4:]]], 2, 4)
        constraints.append((tuple(int(index in (6, 7)) for index in range(8)), 1, 1))
        brute = []
        for selected in itertools.combinations(range(8), 4):
            bits = tuple(int(index in selected) for index in range(8))
            if any(not low <= sum(value*bit for value, bit in zip(vector, bits)) <= high
                   for vector, low, high in constraints):
                continue
            scores = [feature.square(sum(value*bit for value, bit in zip(feature.values, bits)), 4) for feature in features]
            brute.append((max(scores), sum(scores, Fraction()), tuple(ids[index] for index in selected)))
        expected = min(brute)
        actual = ExactAssignment(ids, features, constraints, 4, 60).solve()
        self.assertEqual(Fraction(actual["minimax_squared_exact"]), expected[0])
        self.assertEqual(Fraction(actual["sumsq_exact"]), expected[1])
        self.assertEqual(tuple(actual["audit_proposal"]), expected[2])

    def test_retains_complement_levels_and_nonzero_quartile_imbalance(self):
        rows = [{"participant_id": f"sub-{index+1:02d}", "age_source": "20",
                 "gender": "F" if index < 4 else "M", "native_english": "Yes",
                 "language_profile": "One", "language_spoken_at_home": "English"} for index in range(7)]
        _, features, _, dropped = features_from_rows(rows, [[row["participant_id"] for row in rows[:4]],
                                                           [row["participant_id"] for row in rows[4:]]], 1, 2)
        gender = [feature for feature in features if feature.group == "gender"]
        self.assertEqual(len(gender), 2)
        self.assertEqual(gender[0].square(1, 2), gender[1].square(1, 2))
        self.assertTrue(all(feature.square(1, 2) > 0 for feature in features if feature.group == "acquisition_quartile"))
        self.assertIn("age", dropped)


if __name__ == "__main__":
    unittest.main()
