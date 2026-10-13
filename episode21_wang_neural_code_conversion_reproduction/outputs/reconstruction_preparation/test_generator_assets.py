"""Two focused local-ZIP checks; UNRUN, coordinator Slurm execution only."""
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

import generator_assets as assets


def fixture(directory, *, include_bounds=True):
    path = directory / "fixture.zip"
    contents = {
        assets.REQUIRED_MEMBERS["checkpoint"]: b"opaque fixture bytes, never a model",
        assets.REQUIRED_MEMBERS["latent_upper_bound"]: b"1 2 3\n",
        "unused/another_generator.pt": b"must not be extracted",
    }
    if not include_bounds:
        contents.pop(assets.REQUIRED_MEMBERS["latent_upper_bound"])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, value)
    payload = path.read_bytes()
    return path, contents, {"ARCHIVE_BYTES": len(payload), "ARCHIVE_MD5": hashlib.md5(payload).hexdigest()}


class GeneratorAssetTests(unittest.TestCase):
    def test_fixed_identity_and_exclusive_destination_refuse_before_extraction(self):
        self.assertEqual(assets.ARCHIVE_BYTES, 832_692_892)
        self.assertEqual(assets.ARCHIVE_MD5, "8be66997045859a49b1d2bcef6640490")
        self.assertEqual(assets.archive_tools.MAX_ARCHIVE_UNCOMPRESSED_BYTES, 2_000_000_000)
        self.assertEqual(assets.archive_tools.MAX_MEMBER_UNCOMPRESSED_BYTES, 1_500_000_000)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, _, pins = fixture(root)
            output = root / "new"
            with mock.patch.object(assets.archive_tools, "file_digest") as digest:
                with self.assertRaisesRegex(ValueError, "size differs"):
                    assets.extract_generator_assets(archive, output, budget_check=lambda: None)
                digest.assert_not_called()
            self.assertFalse(output.exists())
            with mock.patch.object(assets, "ARCHIVE_BYTES", pins["ARCHIVE_BYTES"]):
                with mock.patch.object(assets.archive_tools, "inspect_zip") as inspect:
                    with self.assertRaisesRegex(ValueError, "MD5 or size"):
                        assets.extract_generator_assets(archive, output, budget_check=lambda: None)
                    inspect.assert_not_called()
            self.assertFalse(output.exists())
            output.mkdir()
            retained = output / "historical.txt"
            retained.write_text("preserve this attempt")
            with mock.patch.object(assets.archive_tools, "regular_file_identity") as identity:
                with self.assertRaises(FileExistsError):
                    assets.extract_generator_assets(archive, output, budget_check=lambda: None)
                identity.assert_not_called()
            self.assertEqual(retained.read_text(), "preserve this attempt")
            missing_root = root / "missing_fixture"
            missing_root.mkdir()
            missing, _, missing_pins = fixture(missing_root, include_bounds=False)
            with mock.patch.multiple(assets, **missing_pins):
                with self.assertRaisesRegex(ValueError, "exact unique regular"):
                    assets.extract_generator_assets(missing, root / "missing_output", budget_check=lambda: None)
            self.assertFalse((root / "missing_output").exists())

    def test_only_two_members_publish_with_integrity_and_failed_partials_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive, contents, pins = fixture(root)
            checks = []
            with mock.patch.multiple(assets, **pins), mock.patch.object(assets, "CHUNK_BYTES", 5):
                output = root / "success"
                result = assets.extract_generator_assets(archive, output,
                    budget_check=lambda: checks.append(True))
                self.assertGreater(len(checks), 12)
                self.assertEqual({path.name for path in output.iterdir()},
                    {"generator_relu7.pt", "relu7_act_range_3x.txt"})
                self.assertTrue(result["required_members_complete"])
                for flag in ("model_deserialized", "latent_values_validated", "runtime_qualified",
                             "empirical_access_authorized"):
                    self.assertFalse(result[flag])
                for role, metadata in result["members"].items():
                    expected = contents[assets.REQUIRED_MEMBERS[role]]
                    self.assertEqual(Path(metadata["resolved_path"]).read_bytes(), expected)
                    self.assertEqual(metadata["sha256"], hashlib.sha256(expected).hexdigest())
                    self.assertEqual(metadata["extracted_bytes"], len(expected))
                # A callback failure after a real partial write must preserve
                # the partial and refuse a second attempt in the same folder.
                cancelled = root / "cancelled"
                partial = cancelled / "generator_relu7.pt.partial"
                calls_after_create = []
                def cancel():
                    if partial.exists():
                        calls_after_create.append(True)
                        if len(calls_after_create) == 3:
                            raise TimeoutError("fixture budget expired")
                with self.assertRaisesRegex(TimeoutError, "fixture budget expired"):
                    assets.extract_generator_assets(archive, cancelled, budget_check=cancel)
                self.assertTrue(partial.exists())
                self.assertGreater(partial.stat().st_size, 0)
                self.assertFalse((cancelled / "generator_relu7.pt").exists())
                with self.assertRaises(FileExistsError):
                    assets.extract_generator_assets(archive, cancelled, budget_check=lambda: None)
                # Independently force the recorded inventory digest to differ
                # while keeping the actual ZIP, size and MD5 unchanged.
                original_inspect = assets.archive_tools.inspect_zip
                def altered_inventory(*args, **kwargs):
                    inventory = original_inspect(*args, **kwargs)
                    for member in inventory["members"]:
                        if member["path"] == assets.REQUIRED_MEMBERS["checkpoint"]:
                            member["sha256"] = "0" * 64
                    return inventory
                corrupt_output = root / "integrity_failure"
                with mock.patch.object(assets.archive_tools, "inspect_zip", side_effect=altered_inventory):
                    with self.assertRaisesRegex(ValueError, "integrity check"):
                        assets.extract_generator_assets(archive, corrupt_output, budget_check=lambda: None)
                self.assertEqual((corrupt_output / "generator_relu7.pt.partial").read_bytes(),
                    contents[assets.REQUIRED_MEMBERS["checkpoint"]])
                self.assertFalse((corrupt_output / "generator_relu7.pt").exists())


if __name__ == "__main__":
    unittest.main()
