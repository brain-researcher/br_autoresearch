"""Inherited-setgid and fresh-attempt fixtures; run only in a Slurm allocation."""
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import materialize_neural_roles as pilot
import materialize_neural_roles_full as full
import materialize_neural_roles_pilot002 as replacement
import neural_role_private_paths as method


class NeuralRolePrivatePathTests(unittest.TestCase):
    def test_new_directory_clears_inherited_setgid_before_content(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent.chmod(0o2770)
            witness = parent / "inherited_witness"
            witness.mkdir(mode=0o700)
            self.assertEqual(stat.S_IMODE(witness.stat().st_mode), 0o2700)
            target = parent / "new/child"
            method.private_directory(target, parent)
            self.assertEqual(stat.S_IMODE((parent / "new").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o2770)
            self.assertEqual(list(target.iterdir()), [])

    def test_existing_unexpected_directory_and_symlink_are_refused_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            unexpected = parent / "existing"
            unexpected.mkdir(mode=0o700)
            unexpected.chmod(0o2700)
            with self.assertRaises(method.core.Refusal):
                method.private_directory(unexpected / "child", parent)
            self.assertEqual(stat.S_IMODE(unexpected.stat().st_mode), 0o2700)
            self.assertFalse((unexpected / "child").exists())
            link = parent / "link"
            link.symlink_to(unexpected, target_is_directory=True)
            with self.assertRaises(method.core.Refusal):
                method.private_directory(link, parent)
            self.assertTrue(link.is_symlink())

    def test_pilot002_dispatch_changes_only_process_bindings_and_full_selection_is_explicit(self):
        original_private, original_public = pilot.PRIVATE, pilot.PUBLIC
        original_helper = pilot.private_directory
        def fake_main():
            self.assertEqual(pilot.PRIVATE.name, "pilot002")
            self.assertEqual(pilot.PUBLIC.name, "pilot002")
            self.assertIs(pilot.private_directory, method.private_directory)
            return 23
        with patch.multiple(pilot, PRIVATE=original_private, PUBLIC=original_public, private_directory=original_helper), \
             patch.object(pilot, "main", side_effect=fake_main):
            self.assertEqual(replacement.main(), 23)
        self.assertEqual((pilot.PRIVATE, pilot.PUBLIC), (original_private, original_public))
        with patch.multiple(full, PILOT_PRIVATE=full.PILOT_PRIVATE, PILOT_PUBLIC=full.PILOT_PUBLIC, PILOT_REPORT=full.PILOT_REPORT):
            full.select_pilot_attempt("pilot002")
            self.assertEqual(full.PILOT_PRIVATE.name, "pilot002")
            self.assertEqual(full.PILOT_REPORT.parent.name, "pilot002")
            self.assertEqual((pilot.PRIVATE, pilot.PUBLIC), (original_private, original_public))
            with self.assertRaises(full.core.Refusal):
                full.select_pilot_attempt("../pilot003")


if __name__ == "__main__":
    unittest.main()
