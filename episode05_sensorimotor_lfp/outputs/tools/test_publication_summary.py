"""One exact regression fixture for the new publication summary statistic."""
import math
from pathlib import Path
import unittest

from summarize_publication_reproduction import association, session_directory
from source_adapter import M1_SESSIONS


class PublicationAssociationTest(unittest.TestCase):
    def test_recovery_changes_only_the_exact_session_directory(self):
        first, remaining = Path("first_completed"), Path("attempt003")
        recovered = Path("attempt004/Chewie_20150313")
        changed = []
        for index, expected in enumerate(M1_SESSIONS):
            original = session_directory(index, expected, first, remaining)
            selected = session_directory(index, expected, first, remaining, recovered)
            if selected != original:
                changed.append((index, expected, selected))
        self.assertEqual(changed, [(12, ("Chewie", "Chewie-R", "20150313"), recovered)])
        self.assertEqual(session_directory(0, M1_SESSIONS[0], first, remaining, recovered), first)

    def test_exact_regression_with_orthogonal_residual(self):
        # x centered has sumsq20; residual [1,-1,-1,1] is orthogonal to x
        # and the intercept. Thus slope2, SSE4, SST84, F(1,2)=40 exactly.
        # F(1,2)'s survival at40 is 1-sqrt(20/21).
        got = association([-3, -1, 1, 3], [-5, -3, 1, 7])
        self.assertAlmostEqual(got["slope"], 2, places=13)
        self.assertAlmostEqual(got["intercept"], 0, places=13)
        self.assertAlmostEqual(got["r_squared"], 20 / 21, places=13)
        self.assertAlmostEqual(got["F_test_p"], 1 - math.sqrt(20 / 21), places=13)


if __name__ == "__main__":
    unittest.main()
