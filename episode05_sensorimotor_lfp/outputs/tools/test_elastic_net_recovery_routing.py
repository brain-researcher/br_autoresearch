"""Fixed preserved/native bank routing without opening scientific artifacts."""
from pathlib import Path
import unittest
from unittest.mock import patch

import run_elastic_net_recovery_panel as recovery
from primary_reference import SESSIONS


class RecoveryRoutingTest(unittest.TestCase):
    def test_exact_eight_preserved_and_eight_native_sources_without_reads(self):
        old = Path("/synthetic/preserved_attempt001")
        new = Path("/synthetic/native_attempt002")
        preserved = {"Chewie_20150313", "Mihili_20140217", "Mihili_20140218", "Mihili_20140303",
                     "Mihili_20140304", "Mihili_20140306", "Chewie_20160929", "Chewie_20161005"}
        fresh = {"Mihili_20140307", "Chewie_20161006", "Chewie_20161007", "Chewie_20161014",
                 "Chewie_20161021", "Chewie_20150319", "Chewie_20150629", "Chewie_20150630"}
        self.assertEqual(set(SESSIONS), preserved | fresh)
        with patch.object(Path, "read_text", side_effect=AssertionError("No bank reads allowed")):
            sources = recovery.selected_bank_sources(old, new)
        self.assertEqual(list(sources), list(SESSIONS))
        self.assertEqual(len(set(sources.values())), 16)
        for session in SESSIONS:
            expected = (old if session in preserved else new)/session/"score_banks.json"
            self.assertEqual(sources[session], str(expected))
        self.assertEqual(sources["Mihili_20140307"], str(new/"Mihili_20140307/score_banks.json"))
        self.assertEqual(recovery.PILOT, "Mihili_20140307")

    def test_equal_roots_and_normalized_aliases_cannot_reuse_failed_paths(self):
        root = Path("/synthetic/preserved")
        for other in (root, root/"subdirectory/.."):
            with self.subTest(other=other), self.assertRaises(ValueError):
                recovery.selected_bank_sources(root, other)
        with patch.object(recovery, "NEW", recovery.PRESERVED):
            with self.assertRaises(ValueError):
                recovery.selected_bank_sources(root, Path("/synthetic/native"))

    def test_new_bank_requires_both_report_and_model_backend_identity(self):
        recipe, backend = recovery.RECIPES[0], recovery.BACKEND
        valid = {"solver_backend": backend, "banks": {recipe: {"solver_backend": backend}}}
        with patch.object(recovery, "validate_session") as validate:
            recovery.validate_native(valid, recovery.PILOT)
            validate.assert_called_once_with(valid, recovery.PILOT)
            for changed in ({"banks": {recipe: {"solver_backend": backend}}},
                            {"solver_backend": backend, "banks": {recipe: {}}}):
                with self.assertRaises(ValueError):
                    recovery.validate_native(changed, recovery.PILOT)


if __name__ == "__main__":
    unittest.main()
