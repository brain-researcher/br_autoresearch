"""One focused eleven-session roster check; run only in a coordinated compute job."""
import unittest
from unittest.mock import patch

from source_support import (
    EP06_RESERVED_SESSIONS, ORIGINAL_SESSION_POSITIONS, PREPARATION_SESSIONS,
    SESSIONS,
)
from view_access import CandidateViews, EARLIER, NEW


class PreparationRosterTests(unittest.TestCase):
    def test_joint_exclusion_preserves_history_positions_and_refuses_reader_io(self):
        self.assertEqual(len(SESSIONS), 12)
        self.assertEqual(SESSIONS[4], ("Mihili", "20140306", "new"))
        self.assertEqual(PREPARATION_SESSIONS, SESSIONS[:4] + SESSIONS[5:])
        self.assertEqual(ORIGINAL_SESSION_POSITIONS[("Mihili", "20140307")], 6)
        self.assertEqual(EP06_RESERVED_SESSIONS, {
            ("Mihili", "20140303"), ("Mihili", "20140307"),
            ("Chewie", "20161006"), ("Chewie", "20161021"),
        })
        for animal, label in (("Mihili", "Mihili"), ("Chewie", "Chewie-L")):
            for role, roster in (("earlier", EARLIER), ("new", NEW)):
                self.assertEqual(roster[label], tuple(
                    day for name, day, day_role in PREPARATION_SESSIONS
                    if name == animal and day_role == role
                ))
        reader = CandidateViews("/unused-candidate-path")
        with patch("view_access.h5py.File") as open_file:
            for region in ("M1", "PMd"):
                for role in ("fit", "development_features"):
                    with self.assertRaises(ValueError):
                        next(reader.iter_records(role, animal="Mihili", target_day="20140306",
                                                 region=region, stream="history_assisted"))
            open_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
