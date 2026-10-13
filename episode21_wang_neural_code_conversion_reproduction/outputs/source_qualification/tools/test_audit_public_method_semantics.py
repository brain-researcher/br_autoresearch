#!/usr/bin/env python3
"""Regression tests for the EP21 public method-semantics audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("audit_public_method_semantics.py")
SPEC = importlib.util.spec_from_file_location("audit_public_method_semantics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PublicMethodSemanticsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with audit.verified_archive(audit.SOURCE_ARCHIVE) as (archive, identity):
            cls.archive_identity = identity
            cls.members = audit.read_allowlisted_members(archive)

    def member_copy(self):
        return copy.deepcopy(self.members)

    def test_exact_archive_and_allowlist(self) -> None:
        self.assertEqual(self.archive_identity["md5"], audit.PINNED_ARCHIVE_MD5)
        self.assertEqual(
            self.archive_identity["sha256"], audit.PINNED_ARCHIVE_SHA256
        )

    def test_archive_reads_only_allowlisted_member_bodies(self) -> None:
        class RecordingArchive:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.read_names = []

            def infolist(self):
                return self.wrapped.infolist()

            def read(self, member):
                self.read_names.append(member.filename)
                return self.wrapped.read(member)

        with audit.verified_archive(audit.SOURCE_ARCHIVE) as (archive, _identity):
            recording = RecordingArchive(archive)
            observed = audit.read_allowlisted_members(recording)
        expected_names = {
            audit.ARCHIVE_PREFIX + relative
            for relative, _size, _digest in audit.MEMBER_SPECS.values()
        }
        self.assertEqual(set(recording.read_names), expected_names)
        self.assertEqual(len(recording.read_names), len(expected_names))
        self.assertEqual(set(observed), set(audit.MEMBER_SPECS))
        self.assertEqual(self.archive_identity["size_bytes"], audit.PINNED_ARCHIVE_SIZE)
        self.assertEqual(set(self.members), set(audit.MEMBER_SPECS))
        self.assertTrue(
            all(
                not value["path"].startswith("/")
                and "/results/" not in value["path"]
                and "/data/" not in value["path"]
                for value in self.members.values()
            )
        )

    def test_real_member_facts(self) -> None:
        facts = audit.derive_method_facts(self.members)
        decoder = facts["feature_decoder"]
        self.assertEqual(decoder["ridge_alpha"], 100)
        self.assertEqual(decoder["requested_voxels_per_feature_unit"], 500)
        self.assertEqual(decoder["training_response_std_ddof"], 1)
        self.assertEqual(len(decoder["layers"]), 19)
        self.assertFalse(decoder["explicit_intercept_option_in_script"])
        brain = facts["brain_loss"]
        self.assertEqual(brain["parameter_table"]["directed_pair_count"], 20)
        self.assertEqual(brain["fit_n_feat"], 0)
        self.assertFalse(brain["source_target_label_equality_assertion_in_tagged_fitter"])
        self.assertEqual(
            brain["parameter_table"]["unique_alpha_values"],
            [620, 640, 660, 740, 780, 800, 820, 1000, 1200, 1400],
        )
        self.assertTrue(
            facts["source_style_dyadic_bootstrap"]["conversion_bootstrap"]
            ["empty_or_unfillable_draw_has_no_bounded_failure_rule"]
        )

    def test_build_report_is_conservative_and_bound(self) -> None:
        report = audit.build_report(19)
        self.assertEqual(
            report["schema_version"], "ep21.public_method_semantics_audit.v1"
        )
        self.assertEqual(report["attempt"], 19)
        self.assertEqual(
            report["bindings"]["qualification_report"]["sha256"],
            audit.QUALIFICATION_SHA256,
        )
        self.assertEqual(
            report["bindings"]["neutral_conformance_report"]["sha256"],
            audit.NEUTRAL_CONFORMANCE_SHA256,
        )
        self.assertEqual(
            report["bindings"]["protocol_table"]["sha256"],
            audit.PROTOCOL_TABLE_SHA256,
        )
        self.assertFalse(report["protocol_effect"]["scientific_choice_adopted"])
        self.assertFalse(report["protocol_effect"]["protocol_lock_emitted"])
        self.assertFalse(report["protocol_effect"]["empirical_access_authorized"])
        self.assertFalse(report["protocol_effect"]["empirical_readiness"])
        self.assertTrue(all(value is False for value in report["boundary"].values()))
        self.assertIn("fastl2lir", report["retained_unresolved"])
        self.assertIn("bdpy", report["retained_unresolved"])

    def test_frozen_attempt_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "attempt 19"):
            audit.build_report(18)

    def test_feature_config_mutations_fail(self) -> None:
        config = self.members["feature_decoder_config"]["text"]
        for old, new in [
            ("alpha: 100", "alpha: 101"),
            ("VC:  500", "VC:  499"),
            ("image_index", "wrong_label"),
            ("  - fc8", "  - fc9"),
        ]:
            with self.subTest(old=old):
                with self.assertRaises(ValueError):
                    audit.parse_feature_config(config.replace(old, new, 1))

    def test_brain_parameter_mutations_fail(self) -> None:
        parameters = self.members["brain_parameters"]["text"]
        with self.assertRaises(ValueError):
            audit.parse_brain_parameters(parameters.replace("sub01,sub02", "sub01,sub03", 1))
        with self.assertRaises(ValueError):
            audit.parse_brain_parameters(parameters.replace(",6000,", ",5999,", 1))
        with self.assertRaises(ValueError):
            audit.parse_brain_parameters(parameters.replace(",VC,wholeVC", ",V1,wholeVC", 1))

    def test_method_semantic_mutations_fail(self) -> None:
        cases = [
            ("feature_decoder_train", "ddof=1", "ddof=0"),
            ("feature_decoder_predict", "y_pred = y_pred * y_norm + y_mean", "y_pred = y_pred"),
            ("brain_train", "n_feat=0", "n_feat=1"),
            ("brain_test", "np.ones((x.shape[0], 1))", "np.zeros((x.shape[0], 1))"),
            ("feature_evaluator", "np.nanmean(r_prof.flatten())", "np.mean(r_prof.flatten())"),
            ("pixel_evaluator", "pairwise_identification(recon_images, true_images)", "pattern_correlation(recon_images, true_images)"),
            ("dnn_evaluator", "self.image_size = [227, 227]", "self.image_size = [224, 224]"),
            ("conversion_bootstrap", "if source_sample != target_sample:", "if source_sample == target_sample:"),
            ("decoding_bootstrap", "combined_sample = combined_sample.head(sample_size)", "combined_sample = combined_sample.tail(sample_size)"),
            ("identification_bootstrap", "with Pool() as pool:", "with Pool(1) as pool:"),
        ]
        for key, old, new in cases:
            with self.subTest(key=key):
                members = self.member_copy()
                members[key]["text"] = members[key]["text"].replace(old, new, 1)
                with self.assertRaises(ValueError):
                    audit.derive_method_facts(members)

    def test_parser_never_executes_top_level_code(self) -> None:
        members = self.member_copy()
        members["feature_decoder_train"]["text"] += "\nraise RuntimeError('must not execute')\n"
        facts = audit.derive_method_facts(members)
        self.assertEqual(facts["feature_decoder"]["ridge_alpha"], 100)

    def test_allowlist_rejects_duplicate_and_hash_mutation(self) -> None:
        path, size, digest = audit.MEMBER_SPECS["brain_parameters"]
        full_path = audit.ARCHIVE_PREFIX + path
        payload = self.members["brain_parameters"]["text"].encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            duplicate_zip = pathlib.Path(directory) / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate_zip, "w") as archive:
                    archive.writestr(full_path, payload)
                    archive.writestr(full_path, payload)
            with zipfile.ZipFile(duplicate_zip) as archive:
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    audit.read_allowlisted_members(
                        archive, {"brain_parameters": (path, size, digest)}
                    )
            modified_zip = pathlib.Path(directory) / "modified.zip"
            with zipfile.ZipFile(modified_zip, "w") as archive:
                archive.writestr(full_path, payload[:-1] + b"X")
            with zipfile.ZipFile(modified_zip) as archive:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    audit.read_allowlisted_members(
                        archive, {"brain_parameters": (path, size, digest)}
                    )

    def test_bindings_reject_promotion_or_dependency_change(self) -> None:
        qualification = json.loads(audit.QUALIFICATION_REPORT.read_text())
        neutral = json.loads(audit.NEUTRAL_CONFORMANCE_REPORT.read_text())
        protocol = audit.PROTOCOL_TABLE.read_text()
        audit.validate_bindings(qualification, neutral, protocol)

        promoted = copy.deepcopy(neutral)
        promoted["protocol_lock_emitted"] = True
        with self.assertRaises(ValueError):
            audit.validate_bindings(qualification, promoted, protocol)

        rebound = copy.deepcopy(qualification)
        rebound["source_pipeline_assets"]["unmanifested_or_unpinned_gaps"][
            "main_bdpy_dependency"
        ]["state"] = "resolved"
        with self.assertRaises(ValueError):
            audit.validate_bindings(rebound, neutral, protocol)

        with self.assertRaises(ValueError):
            audit.validate_bindings(
                qualification,
                neutral,
                protocol.replace("protocol_lock_exists: false", "protocol_lock_exists: true"),
            )

    def test_stable_reader_rejects_symlink_hardlink_and_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            payload = b"bound-input"
            digest = hashlib.sha256(payload).hexdigest()
            regular = root / "regular"
            regular.write_bytes(payload)
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            with self.assertRaises(ValueError):
                audit.read_stable_bytes(
                    symlink, label="fixture", maximum_bytes=100, expected_sha256=digest
                )
            hardlink = root / "hardlink"
            os.link(regular, hardlink)
            with self.assertRaisesRegex(ValueError, "single-link"):
                audit.read_stable_bytes(
                    regular, label="fixture", maximum_bytes=100, expected_sha256=digest
                )
            hardlink.unlink()
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                audit.read_stable_bytes(
                    fifo, label="fixture", maximum_bytes=100, expected_sha256=digest
                )

    def test_stable_reader_rejects_symlinked_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            real = root / "real"
            real.mkdir()
            payload = b"bound-input"
            (real / "input").write_bytes(payload)
            (root / "alias").symlink_to(real, target_is_directory=True)
            with self.assertRaises((ValueError, OSError)):
                audit.read_stable_bytes(
                    root / "alias/input",
                    label="fixture",
                    maximum_bytes=100,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )

    def test_verified_archive_rejects_symlink_hardlink_fifo_and_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            regular = root / "fixture.zip"
            with zipfile.ZipFile(regular, "w") as archive:
                archive.writestr("safe.txt", b"safe")
            payload = regular.read_bytes()
            identity_patch = {
                "PINNED_ARCHIVE_SIZE": len(payload),
                "PINNED_ARCHIVE_MD5": hashlib.md5(payload).hexdigest(),
                "PINNED_ARCHIVE_SHA256": hashlib.sha256(payload).hexdigest(),
            }

            symlink = root / "alias.zip"
            symlink.symlink_to(regular)
            with mock.patch.multiple(audit, **identity_patch):
                with self.assertRaises((ValueError, OSError)):
                    with audit.verified_archive(symlink):
                        pass

            hardlink = root / "hard.zip"
            os.link(regular, hardlink)
            with mock.patch.multiple(audit, **identity_patch):
                with self.assertRaisesRegex(ValueError, "single-link"):
                    with audit.verified_archive(regular):
                        pass
            hardlink.unlink()

            fifo = root / "fifo.zip"
            os.mkfifo(fifo)
            with mock.patch.multiple(audit, **identity_patch):
                with self.assertRaises(ValueError):
                    with audit.verified_archive(fifo):
                        pass

            real_fstat = audit.os.fstat
            calls = 0

            class ChangedStat:
                def __init__(self, wrapped):
                    self.wrapped = wrapped

                def __getattr__(self, name):
                    if name == "st_ctime_ns":
                        return self.wrapped.st_ctime_ns + 1
                    return getattr(self.wrapped, name)

            def changed_second_fstat(descriptor):
                nonlocal calls
                calls += 1
                observed = real_fstat(descriptor)
                return ChangedStat(observed) if calls == 2 else observed

            with mock.patch.multiple(audit, **identity_patch):
                with mock.patch.object(audit.os, "fstat", side_effect=changed_second_fstat):
                    with self.assertRaisesRegex(ValueError, "changed while hashing"):
                        with audit.verified_archive(regular):
                            pass

    def test_atomic_publication_and_existing_target_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)
            report = {"schema_version": "fixture", "finite": True}
            identity = audit.atomic_publish_json(root, "report.json", report)
            target = root / "report.json"
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(), identity["sha256"]
            )
            with self.assertRaises(FileExistsError):
                audit.atomic_publish_json(root, "report.json", report)
            self.assertEqual(
                [path.name for path in root.iterdir()], ["report.json"]
            )

    def test_atomic_publication_refuses_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)
            outside = root / "outside"
            outside.write_text("keep")
            (root / "report.json").symlink_to(outside)
            with self.assertRaises(FileExistsError):
                audit.atomic_publish_json(root, "report.json", {"x": 1})
            self.assertEqual(outside.read_text(), "keep")

    def test_atomic_publication_handles_short_write_and_cleans_zero_write(self) -> None:
        real_write = audit.os.write
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)

            def short_write(descriptor, payload):
                return real_write(descriptor, payload[: max(1, len(payload) // 2)])

            with mock.patch.object(audit.os, "write", side_effect=short_write):
                audit.atomic_publish_json(root, "short.json", {"payload": "x" * 100})
            self.assertTrue((root / "short.json").is_file())

            with mock.patch.object(audit.os, "write", return_value=0):
                with self.assertRaisesRegex(OSError, "zero-progress"):
                    audit.atomic_publish_json(root, "zero.json", {"x": 1})
            self.assertFalse((root / "zero.json").exists())
            self.assertFalse(any("zero.json" in path.name for path in root.iterdir()))

    def test_atomic_publication_detects_directory_rebind(self) -> None:
        real_open_nofollow = audit.open_nofollow
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            output = base / "output"
            output.mkdir(mode=0o700)
            moved = base / "moved"
            output_open_count = 0

            def rebind_on_reopen(path, flags):
                nonlocal output_open_count
                if path == output:
                    output_open_count += 1
                    if output_open_count == 2:
                        output.rename(moved)
                        output.mkdir(mode=0o700)
                return real_open_nofollow(path, flags)

            with mock.patch.object(audit, "open_nofollow", side_effect=rebind_on_reopen):
                with self.assertRaisesRegex(ValueError, "directory pathname was rebound"):
                    audit.atomic_publish_json(output, "report.json", {"x": 1})
            self.assertFalse((moved / "report.json").exists())
            self.assertFalse((output / "report.json").exists())

    def test_atomic_publication_detects_final_name_replacement(self) -> None:
        real_read_all = audit._read_all_descriptor
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)
            read_count = 0

            def replace_after_held_read(descriptor, maximum_bytes):
                nonlocal read_count
                payload = real_read_all(descriptor, maximum_bytes)
                read_count += 1
                if read_count == 2:
                    target = root / "report.json"
                    target.unlink()
                    target.write_text("replacement")
                    target.chmod(0o600)
                return payload

            with mock.patch.object(
                audit, "_read_all_descriptor", side_effect=replace_after_held_read
            ):
                with self.assertRaisesRegex(ValueError, "final pathname verification"):
                    audit.atomic_publish_json(root, "report.json", {"x": 1})
            self.assertEqual((root / "report.json").read_text(), "replacement")

    def test_atomic_publication_detects_temp_swap_and_postlink_mutation(self) -> None:
        real_link = audit.os.link
        real_write = audit.os.write
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)

            def swap_temp_then_link(source, target, **kwargs):
                source_dir = kwargs["src_dir_fd"]
                audit.os.unlink(source, dir_fd=source_dir)
                replacement = audit.os.open(
                    source,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=source_dir,
                )
                try:
                    real_write(replacement, b"replacement")
                finally:
                    audit.os.close(replacement)
                return real_link(source, target, **kwargs)

            with mock.patch.object(audit.os, "link", side_effect=swap_temp_then_link):
                with self.assertRaisesRegex(ValueError, "does not name the held inode"):
                    audit.atomic_publish_json(root, "swapped.json", {"x": 1})
            self.assertFalse((root / "swapped.json").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)

            def mutate_after_link(source, target, **kwargs):
                result = real_link(source, target, **kwargs)
                descriptor = audit.os.open(
                    target,
                    os.O_WRONLY | os.O_TRUNC,
                    dir_fd=kwargs["dst_dir_fd"],
                )
                try:
                    real_write(descriptor, b"mutated")
                finally:
                    audit.os.close(descriptor)
                return result

            with mock.patch.object(audit.os, "link", side_effect=mutate_after_link):
                with self.assertRaises(ValueError):
                    audit.atomic_publish_json(root, "mutated.json", {"x": 1})
            self.assertFalse((root / "mutated.json").exists())
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
