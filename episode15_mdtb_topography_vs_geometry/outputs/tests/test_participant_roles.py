"""Focused source-free fixtures; execute only in coordinator-owned compute."""
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import assign_participant_roles as roles


def ids(n):
    return [f"sub-{i:02d}" for i in range(1, n + 1)]


def categories(n):
    return [[("observed", "x")] * n for _ in range(3)]


def oracle(names, groups, age, reliability):
    """Independent direct Fraction objective without shared integer/pruning code."""
    n, half = len(names), len(names) // 2
    candidates = []
    for audit in combinations(range(n), half):
        development = tuple(i for i in range(n) if i not in audit)
        count_differences = []
        for field in groups:
            for level in set(field):
                count_differences.append(abs(sum(field[i] == level for i in audit)
                                             - sum(field[i] == level for i in development)))
        objective = [max(count_differences), sum(count_differences)]
        for values in (age, reliability):
            a = sorted(values[i] for i in audit)
            d = sorted(values[i] for i in development)
            objective += [abs(sum(a) / half - sum(d) / half),
                          sum(abs(x - y) for x, y in zip(a, d)) / half]
        candidates.append((tuple(objective), tuple(names[i] for i in audit), audit))
    return min(candidates)[2]


def private_tables(names, age=None, reliability=None, category="X"):
    age = age or [str(20 + i) for i in range(len(names))]
    reliability = reliability or ["0.125"] * len(names)
    demographic = "participant_id\tsex\tage\tnative_english\tses-rest\n" + "".join(
        f"{pid}\t{category}\t{a}\tY\tN\n" for pid, a in zip(names, age))
    private = "participant_id," + roles.PRIVATE_FIELD + "\n" + "".join(
        f"{pid},{r}\n" for pid, r in zip(names, reliability))
    return demographic.encode(), private.encode()


def complete_public_payload():
    n = 24
    selected = {
        "audit_indices": tuple(range(12)), "objective_integers": (0, 0, 0, 0, 0, 0),
        "assignments_enumerated": roles.EXPECTED_ASSIGNMENTS, "enumeration_complete": True,
        "age_integers": (0,) * n, "reliability_integers": (0,) * n,
    }
    return roles.public_roles(ids(n), categories(n), selected)


class ParticipantRoleTests(unittest.TestCase):
    def test_streaming_pruning_matches_independent_exhaustive_objectives(self):
        n = 8
        names = ids(n)
        groups = [[("observed", str((i + shift) % modulus)) for i in range(n)]
                  for shift, modulus in ((0, 2), (1, 3), (0, 2))]
        age = list(map(Fraction, (18, 24, 19, 37, 25, 41, 26, 33)))
        reliability = [Fraction(value, 100) for value in (13, -11, 42, 73, 6, 51, -4, 29)]
        progress = []
        selected = roles.enumerate_roles(names, groups, age, reliability, progress.append)
        self.assertEqual(selected["audit_indices"], oracle(names, groups, age, reliability))
        self.assertEqual(selected["assignments_enumerated"], math.comb(n, n // 2))
        self.assertEqual(progress[-1], 70)
        self.assertEqual(roles.EXPECTED_ASSIGNMENTS, math.comb(24, 12))

    def test_categorical_priority_and_age_before_reliability(self):
        names = ids(4)
        constant = [Fraction(0)] * 4
        groups = categories(4)
        groups[0] = [("observed", value) for value in ("A", "A", "B", "B")]
        # Category balance rules out the otherwise perfect {0,1} age split.
        age = list(map(Fraction, (0, 10, 4, 6)))
        selected = roles.enumerate_roles(names, groups, age, constant)
        self.assertEqual(selected["audit_indices"], (0, 3))
        self.assertEqual(selected["objective_integers"][:2], (0, 0))
        # With categories tied, age balance dominates opposing reliability.
        selected = roles.enumerate_roles(names, categories(4), list(map(Fraction, (0, 1, 2, 3))),
                                         list(map(Fraction, (0, 3, 1, 2))))
        self.assertEqual(selected["audit_indices"], (0, 3))

    def test_wasserstein_resolves_equal_means_and_has_known_value(self):
        values = tuple(map(Fraction, range(8)))
        selected = roles.enumerate_roles(ids(8), categories(8), values, [Fraction(0)] * 8)
        self.assertEqual(selected["audit_indices"], oracle(ids(8), categories(8), values, [Fraction(0)] * 8))
        # Both sums are14, but {0,1,6,7} has W1=2 and {0,2,5,7} has W1=1.
        self.assertEqual(selected["audit_indices"], (0, 2, 5, 7))
        self.assertEqual(roles.wasserstein_numerator((0, 2, 1, 3), (0, 1), (2, 3)), 2)
        self.assertAlmostEqual(roles.standardized(2, (0, 2, 1, 3)), 1 / math.sqrt(1.25))

    def test_exact_serialized_decimals_preserve_real_differences_and_complement_ties(self):
        values = [roles.exact_number(value) for value in (
            "1", "1.00000000000000000001", "1.00000000000000000002", "1.00000000000000000003")]
        self.assertEqual(roles.common_integers(values), (0, 1, 2, 3))
        selected = roles.enumerate_roles(ids(4), categories(4), values, [Fraction(0)] * 4)
        self.assertEqual(selected["audit_indices"], (0, 3))
        complement = (1, 2)
        self.assertEqual(sum(values[i] for i in selected["audit_indices"]), sum(values[i] for i in complement))
        self.assertEqual(selected["assignments_enumerated"], 6)
        self.assertEqual(roles.exact_number("1e-100"), Fraction(1, 10 ** 100))

    def test_zero_spread_missing_categories_and_missing_numeric_failure(self):
        names = ids(4)
        demographic, private = private_tables(names, age=["20"] * 4, category=" n/a ")
        groups, age, reliability = roles.parse_private_inputs(demographic, private, names)
        self.assertEqual(set(groups[0]), {("missing", "")})
        selected = roles.enumerate_roles(names, groups, age, reliability)
        self.assertEqual(selected["audit_indices"], (0, 1))
        self.assertEqual(selected["objective_integers"], (0, 0, 0, 0, 0, 0))
        self.assertEqual(roles.standardized(0, (0, 0, 0, 0)), 0)
        for bad in ("", "n/a", "NaN", "Infinity", "-inf", "1e999"):
            with self.subTest(bad=bad), self.assertRaises(roles.AssignmentFailure):
                roles.exact_number(bad)
        for which in ("age", "reliability"):
            kwargs = {which: ["1", "n/a", "1", "1"]}
            with self.subTest(which=which), self.assertRaises(roles.AssignmentFailure):
                roles.parse_private_inputs(*private_tables(names, **kwargs), names)
        with self.assertRaises(roles.AssignmentFailure):
            roles.parse_private_inputs(demographic.replace(b"sub-02", b"sub-01"), private, names)

    def test_lineage_gate_precedes_all_private_or_demographic_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = Path(temp) / "outputs"
            outputs.mkdir()
            log = outputs / "experiment_log.md"
            log.write_text("## Completed review\nLimited current inventory; uncertainty retained.\n")
            with self.assertRaises(roles.AssignmentFailure):
                roles.require_lineage_review(False, "Completed review", log)
            with self.assertRaises(roles.AssignmentFailure):
                roles.require_lineage_review(True, "Missing review", log)
            self.assertIn("Completed review", roles.require_lineage_review(True, "Completed review", log))
            attempt = outputs / "runs/role_assignment/blocked001"
            argv = ["assign", "--output-dir", str(attempt), "--lineage-review-section", "Completed review"]
            with patch.object(roles, "__file__", str(outputs / "tools/assign_participant_roles.py")), \
                    patch.object(sys, "argv", argv), patch.dict(roles.os.environ, {"SLURM_JOB_ID": "fixture"}), \
                    patch.object(roles.socket, "gethostname", return_value="compute-fixture"), \
                    patch.object(roles, "bounded_read", side_effect=AssertionError("No input may be read before explicit confirmation")) as reader, \
                    patch("builtins.print"):
                self.assertEqual(roles.main(), 1)
                reader.assert_not_called()
            self.assertFalse((attempt / "participant_roles.json").exists())

    def test_prerequisites_require_exact_complete_private_path_and_no_task_b(self):
        source, private_path = Path("/fixture/source"), Path("/fixture/private.csv")
        joined = {"metadata_join_complete": True, "all_selected_Task_A_headers_equal_native_cerebellar_mask_grid": True,
                  "source": str(source), "participants": [{"participant_id": pid} for pid in ids(24)],
                  "demographic_completeness": {"age_available_for_future_role_assignment": True}}
        report = {"status": "complete_all24_private_Task_A_reliability", "private_artifact_published": True,
                  "all24_reliability_computable": True, "all24_half_maps_finite_and_nonconstant": True,
                  "participants_expected": 24, "participants_completed": 24, "private_reliability_records": 24,
                  "fixed_support_points": 208, "support_changed": False, "Task_B_values_read": False,
                  "Task_B_image_streams_opened": 0, "private_artifact_path": str(private_path)}
        self.assertEqual(roles.validate_prerequisites(joined, report, source, private_path), ids(24))
        for key, bad in (("participants_completed", 23), ("Task_B_values_read", True),
                         ("private_artifact_path", "/fixture/other.csv"), ("support_changed", True)):
            with self.subTest(key=key), self.assertRaises(roles.AssignmentFailure):
                roles.validate_prerequisites(joined, dict(report, **{key: bad}), source, private_path)

    def test_only_complete_search_can_publish_and_atomic_failure_has_no_roles(self):
        payload = complete_public_payload()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            incomplete = dict(payload, assignments_enumerated=100, enumeration_complete=False)
            with self.assertRaises(roles.AssignmentFailure):
                roles.publish_roles(incomplete, output)
            with patch.object(roles.os, "link", side_effect=OSError("fixture publication failure")):
                with self.assertRaises(OSError):
                    roles.publish_roles(payload, output)
            self.assertEqual(list(output.iterdir()), [])
            roles.publish_roles(payload, output)
            first = (output / "participant_roles.json").read_bytes()
            with self.assertRaises(FileExistsError):
                roles.publish_roles(payload, output)
            self.assertEqual((output / "participant_roles.json").read_bytes(), first)
            self.assertFalse((output / ".participant_roles.json.partial").exists())
        # An interrupted streaming search returns no current-best assignment.
        def interrupted(*args):
            yield (0, 1)
            raise TimeoutError("fixture interruption")
        with patch.object(roles, "combinations", interrupted), self.assertRaises(TimeoutError):
            roles.enumerate_roles(ids(4), categories(4), [Fraction(0)] * 4, [Fraction(0)] * 4)

    def test_public_payload_contains_roles_and_aggregate_balance_only(self):
        payload = complete_public_payload()
        self.assertEqual(len(payload["audit_ids"]), 12)
        self.assertEqual(len(payload["development_ids"]), 12)
        self.assertEqual(set(payload["audit_ids"]) | set(payload["development_ids"]), set(ids(24)))
        self.assertEqual(set(payload["aggregate_balance_objective"]), set(roles.OBJECTIVES))
        self.assertFalse(payload["participant_numeric_inputs_exported"])
        self.assertFalse(payload["role_specific_numeric_summaries_exported"])
        self.assertFalse(payload["Task_B_values_read"])
        self.assertFalse(payload["physical_handoffs_created"])
        for forbidden in ("age_integers", "reliability_integers", "objective_integers", "participant_id", roles.PRIVATE_FIELD):
            self.assertNotIn('"' + forbidden + '"', json.dumps(payload))
        self.assertEqual(set(payload["categorical_counts"]), set(roles.CATEGORY_FIELDS))


if __name__ == "__main__":
    unittest.main()
