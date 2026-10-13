from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import unittest
from unittest import mock

from ep12_executor.qualification_launch import (
    INTENT_RELATIVE,
    OUTPUT_RELATIVE,
    PROGRESS_RELATIVE,
    RECEIPT_RELATIVE,
    QualificationLaunchError,
    _HASH_TARGETS,
    _TRANSITION_IDENTITY_FIELDS,
    _validated_transition_authority,
    _default_job_resolver,
    _unique_resolved_job_id,
    launch_qualification003,
    read_qualification_launch_receipt,
)
from ep12_executor.policy import canonical_json, digest_object
from ep12_executor.synthetic import EPISODE_ROOT
from ep12_executor.source_freeze import SourceFreezeError
from ep12_executor.test_launch import (
    RECEIPT_RELATIVE as TEST_RECEIPT_RELATIVE,
    TEST_WRAPPER_RELATIVE,
    launch_executor_tests,
    read_test_launch_receipt,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DynamicTransitionQualificationConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.archive = {
            "archive_receipt_file_sha256": "11" * 32,
            "archive_receipt_sha256": "12" * 32,
            "manifest_file_sha256": "13" * 32,
            "manifest_sha256": "14" * 32,
            "tree_sha256": "15" * 32,
        }
        self.final = {
            "status": "absent_at_publication",
            "path": str(
                self.root / "outputs/prelaunch/post36_orchestration/receipt.json"
            ),
            "required_before_irreversible_transition": True,
        }
        self.authority = {
            "authoritative_job_id": "987654",
            "obsolete_job_id": "45332084",
            "obsolete_result_authoritative": False,
            "authoritative_result": True,
            "authority_sha256": "21" * 32,
            "replacement_terminal_identity_path": str(
                self.root
                / "outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"
            ),
            "post36_start_authorization_path": str(
                self.root
                / "outputs/prelaunch/post36_orchestration/start_authorization.json"
            ),
            "obsolete_cancellation_receipt_path": str(
                self.root
                / "outputs/prelaunch/replacement_attestation_launch/"
                "obsolete_cancellation/receipt.json"
            ),
            "controller_final_receipt": self.final,
            "source_archive": self.archive,
        }
        self.evidence = {
            "observed_attestation_authority": {"synthetic": "validated below"},
            "observed_attestation_authority_sha256": self.authority[
                "authority_sha256"
            ],
            "authoritative_attestation_job_id": "987654",
            "obsolete_attestation_job_id": "45332084",
            "controller_final_receipt_status": "absent_at_publication",
            "controller_final_receipt_required_before_irreversible_transition": True,
            "source_archive": self.archive,
            "source_archive_receipt_file_sha256": self.archive[
                "archive_receipt_file_sha256"
            ],
            "source_archive_receipt_sha256": self.archive[
                "archive_receipt_sha256"
            ],
            "source_archive_manifest_file_sha256": self.archive[
                "manifest_file_sha256"
            ],
            "source_archive_manifest_sha256": self.archive["manifest_sha256"],
            "source_archive_tree_sha256": self.archive["tree_sha256"],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validate(self) -> dict[str, object]:
        with mock.patch(
            "ep12_executor.qualification_launch.verify_observed_attestation_authority",
            return_value=self.authority,
        ):
            return _validated_transition_authority(
                self.evidence, episode_root=self.root
            )

    def test_dynamic_replacement_and_exact_old_cancellation_are_consumed(self) -> None:
        self.assertEqual(self._validate(), self.authority)

    def test_static_obsolete_job_cannot_become_authoritative(self) -> None:
        self.authority["authoritative_job_id"] = "45332084"
        self.evidence["authoritative_attestation_job_id"] = "45332084"
        with self.assertRaisesRegex(
            QualificationLaunchError, "misstates dynamic-R authority"
        ):
            self._validate()

    def test_top_start_authorization_path_mismatch_fails_closed(self) -> None:
        self.authority["post36_start_authorization_path"] = str(
            self.root / "wrong-start.json"
        )
        with self.assertRaisesRegex(
            QualificationLaunchError, "misstates dynamic-R authority"
        ):
            self._validate()


class QualificationLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "SEARCH_POLICY.yaml").write_text("schema: test\n", encoding="utf-8")
        frozen_spec = (
            EPISODE_ROOT / "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC.yaml"
        ).read_bytes()
        for index, relative in enumerate(_HASH_TARGETS.values(), start=1):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                frozen_spec
                if relative.name
                in {
                    "SYNTHETIC_QUALIFICATION_SPEC.yaml",
                    "SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml",
                }
                else f"source-{index}\n".encode("ascii")
            )
        test_wrapper = self.root / TEST_WRAPPER_RELATIVE
        test_wrapper.parent.mkdir(parents=True, exist_ok=True)
        test_wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.commands: list[list[str]] = []
        self.resolutions: list[str] = []
        self.transition_patch = mock.patch(
            "ep12_executor.qualification_launch._transition_identity",
            side_effect=self._transition_identity,
        )
        self.transition_patch.start()

    def tearDown(self) -> None:
        self.transition_patch.stop()
        self.temporary.cleanup()

    def _transition_identity(self, *, episode_root: Path, executor_root: Path) -> dict:
        value: dict[str, object] = {field: "a" * 64 for field in _TRANSITION_IDENTITY_FIELDS}
        value.update(
            {
                "transition_id": "ep12_post36_representation_rank_v1",
                "transition_manifest_path": str(
                    episode_root
                    / "outputs/prelaunch/scientific_contract_transition/manifest.json"
                ),
                "transition_receipt_path": str(
                    episode_root
                    / "outputs/prelaunch/scientific_contract_transition/receipt.json"
                ),
                "observed_trial_count": 36,
                "exact_36_attestation_set_matches_transition_evidence": True,
                "source_trial_reports_reverified": True,
                "operator_development_outcome_values_displayed": True,
                "programmatic_transition_extraction_outcome_fields_accessed": False,
                "values_used_for_transition_decisions": False,
                "final_connectivity_accessed": False,
                "active_contract_sha256": _file_hash(
                    episode_root / _HASH_TARGETS["scientific_contract_sha256"]
                ),
                "active_qualification_spec_sha256": _file_hash(
                    episode_root / _HASH_TARGETS["qualification_spec_sha256"]
                ),
                "active_scientific_models_sha256": _file_hash(
                    episode_root / _HASH_TARGETS["scientific_models_code_sha256"]
                ),
            }
        )
        self.assertEqual(executor_root, episode_root / "outputs/executor")
        return value

    def _submit(self, command: list[str]) -> str:
        progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
        self.assertEqual(progress["state"], "submission_uncertain")
        self.assertIsNone(progress["slurm_job_id"])
        self.commands.append(list(command))
        return "90123;sherlock"

    def _resolve_none(self, job_name: str) -> str | None:
        self.resolutions.append(job_name)
        return None

    def _launch_qualification(self, *, test_job_id: str | int, **kwargs: object) -> dict[str, object]:
        test_id = str(test_job_id)
        kwargs.pop("episode_root", None)
        test_receipt = self.root / TEST_RECEIPT_RELATIVE
        if test_receipt.exists():
            observed = read_test_launch_receipt(test_receipt, episode_root=self.root)
            self.assertEqual(observed["slurm_job_id"], test_id)
        else:
            launch_executor_tests(
                episode_root=self.root,
                submitter=lambda command: f"{test_id};sherlock",
                job_resolver=lambda name, user: None,
            )
        resolver = kwargs.pop("job_resolver")

        def adapt(name: str, user: str) -> dict[str, str] | None:
            result = resolver(name)  # type: ignore[operator]
            if result is None:
                return None
            return {"job_id": str(result), "job_name": name, "user": user}

        return launch_qualification003(
            test_job_id=test_id,
            episode_root=self.root,
            job_resolver=adapt,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_exact_submission_receipt_and_idempotent_reentry(self) -> None:
        receipt = self._launch_qualification(
            test_job_id="81234",
            episode_root=self.root,
            submitter=self._submit,
            job_resolver=self._resolve_none,
        )
        self.assertEqual(receipt["slurm_job_id"], "90123")
        self.assertEqual(receipt["test_job_id"], "81234")
        self.assertEqual(receipt["dependency"], "afterok:81234")
        self.assertEqual(receipt["tier"], "qualification")
        self.assertEqual(receipt["replicates_per_core_error_cell"], 72)
        self.assertEqual(receipt["whole_types_per_replicate"], 8)
        self.assertEqual(
            receipt["output_dir"], str((self.root / OUTPUT_RELATIVE).resolve())
        )
        self.assertFalse(receipt["real_connectivity_accessed"])
        self.assertFalse(receipt["final_connectivity_access_authorized"])
        for field, relative in _HASH_TARGETS.items():
            self.assertEqual(receipt[field], _file_hash(self.root / relative))

        self.assertEqual(len(self.commands), 1)
        command = self.commands[0]
        self.assertEqual(command[:2], ["sbatch", "--parsable"])
        self.assertIn(f"--job-name={receipt['slurm_job_name']}", command)
        self.assertIn("--dependency=afterok:81234", command)
        self.assertIn("--kill-on-invalid-dep=yes", command)
        export = next(value for value in command if value.startswith("--export="))
        self.assertIn("EP12_TIER=qualification", export)
        self.assertIn(f"EP12_OUTPUT_DIR={receipt['output_dir']}", export)
        self.assertIn("EP12_REPETITIONS=72", export)
        self.assertIn("EP12_WHOLE_TYPES=8", export)
        self.assertIn("EP12_TRANSITION_MANIFEST=", export)
        self.assertIn("EP12_ACTIVE_CONTRACT_SHA256=", export)
        self.assertNotIn("ALL", export)
        self.assertEqual(
            command[-1],
            str((self.root / _HASH_TARGETS["slurm_script_sha256"]).resolve()),
        )
        self.assertTrue((self.root / INTENT_RELATIVE).is_file())
        self.assertTrue((self.root / PROGRESS_RELATIVE).is_file())
        self.assertTrue((self.root / RECEIPT_RELATIVE).is_file())

        repeated = self._launch_qualification(
            test_job_id="81234",
            episode_root=self.root,
            submitter=lambda command: self.fail("must not resubmit"),
            job_resolver=lambda name: self.fail("must not query after receipt"),
        )
        self.assertEqual(repeated, receipt)
        self.assertEqual(len(self.commands), 1)

    def test_client_interruption_recovers_accepted_job_without_duplicate(self) -> None:
        accepted: list[list[str]] = []

        def disconnected_submit(command: list[str]) -> str:
            progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
            self.assertEqual(progress["state"], "submission_uncertain")
            self.assertIsNone(progress["slurm_job_id"])
            accepted.append(list(command))
            raise RuntimeError("client lost the accepted sbatch response")

        with self.assertRaisesRegex(RuntimeError, "lost the accepted"):
            self._launch_qualification(
                test_job_id=81235,
                episode_root=self.root,
                submitter=disconnected_submit,
                job_resolver=lambda name: None,
            )
        self.assertEqual(len(accepted), 1)
        progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
        self.assertEqual(progress["state"], "submission_uncertain")
        self.assertIsNone(progress["slurm_job_id"])

        for _ in range(2):
            with self.assertRaisesRegex(
                QualificationLaunchError, "refusing duplicate sbatch"
            ):
                self._launch_qualification(
                    test_job_id=81235,
                    episode_root=self.root,
                    submitter=lambda command: self.fail(
                        "uncertain retry must not duplicate"
                    ),
                    job_resolver=lambda name: None,
                )
        self.assertEqual(len(accepted), 1)
        progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
        self.assertEqual(progress["state"], "submission_uncertain")
        self.assertIsNone(progress["slurm_job_id"])

        recovered_names: list[str] = []

        def recover(job_name: str) -> str | None:
            recovered_names.append(job_name)
            return "90124"

        receipt = self._launch_qualification(
            test_job_id=81235,
            episode_root=self.root,
            submitter=lambda command: self.fail("recovery must not duplicate"),
            job_resolver=recover,
        )
        self.assertEqual(receipt["slurm_job_id"], "90124")
        self.assertEqual(recovered_names, [receipt["slurm_job_name"]])
        progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
        self.assertEqual(progress["state"], "submitted")
        self.assertEqual(progress["slurm_job_id"], "90124")

    def test_source_mutation_after_planned_intent_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            self._launch_qualification(
                test_job_id=81236,
                episode_root=self.root,
                submitter=lambda command: (_ for _ in ()).throw(
                    RuntimeError("disconnect")
                ),
                job_resolver=lambda name: None,
            )
        mutated = self.root / _HASH_TARGETS["scientific_models_code_sha256"]
        mutated.write_bytes(mutated.read_bytes() + b"mutation\n")
        with self.assertRaisesRegex(SourceFreezeError, "surface changed"):
            self._launch_qualification(
                test_job_id=81236,
                episode_root=self.root,
                submitter=lambda command: self.fail("must not submit after mutation"),
                job_resolver=lambda name: self.fail("must not query after mutation"),
            )

    def test_progress_tampering_and_nonfresh_output_fail_closed(self) -> None:
        (self.root / OUTPUT_RELATIVE).mkdir(parents=True)
        with self.assertRaisesRegex(QualificationLaunchError, "not fresh"):
            self._launch_qualification(
                test_job_id=81237,
                episode_root=self.root,
                submitter=self._submit,
                job_resolver=self._resolve_none,
            )
        (self.root / OUTPUT_RELATIVE).rmdir()

        with self.assertRaises(RuntimeError):
            self._launch_qualification(
                test_job_id=81237,
                episode_root=self.root,
                submitter=lambda command: (_ for _ in ()).throw(
                    RuntimeError("disconnect")
                ),
                job_resolver=lambda name: None,
            )
        path = self.root / PROGRESS_RELATIVE
        progress = json.loads(path.read_text())
        progress["state"] = "submitted"
        path.write_text(canonical_json(progress) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(QualificationLaunchError, "digest mismatch"):
            self._launch_qualification(
                test_job_id=81237,
                episode_root=self.root,
                submitter=lambda command: self.fail("must not submit corrupt progress"),
                job_resolver=lambda name: self.fail("must not query corrupt progress"),
            )

    def test_final_receipt_requires_durable_submitted_progress(self) -> None:
        self._launch_qualification(
            test_job_id=81238,
            episode_root=self.root,
            submitter=self._submit,
            job_resolver=self._resolve_none,
        )
        (self.root / PROGRESS_RELATIVE).unlink()
        with self.assertRaisesRegex(QualificationLaunchError, "lacks durable progress"):
            self._launch_qualification(
                test_job_id=81238,
                episode_root=self.root,
                submitter=lambda command: self.fail("must not resubmit"),
                job_resolver=lambda name: self.fail("must not query"),
            )

    def test_shared_receipt_verifier_rejects_noncanonical_and_mutated_fields(self) -> None:
        receipt = self._launch_qualification(
            test_job_id=81239,
            episode_root=self.root,
            submitter=self._submit,
            job_resolver=self._resolve_none,
        )
        path = self.root / RECEIPT_RELATIVE
        self.assertEqual(
            read_qualification_launch_receipt(path, episode_root=self.root),
            receipt,
        )
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(QualificationLaunchError, "not canonical JSON"):
            read_qualification_launch_receipt(path, episode_root=self.root)

        mutations = {
            "qualification_version": "qualification004",
            "whole_types_per_replicate": 9,
            "test_job_id": "81240",
            "dependency": "afterok:99999",
            "intent_sha256": "0" * 64,
            "slurm_job_name": "ep12-qual003-not-the-intent-name",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                changed = dict(receipt)
                changed[field] = replacement
                core = {
                    key: value
                    for key, value in changed.items()
                    if key != "launch_sha256"
                }
                changed["launch_sha256"] = digest_object(core)
                path.write_text(canonical_json(changed) + "\n")
                with self.assertRaises(QualificationLaunchError):
                    read_qualification_launch_receipt(path, episode_root=self.root)

        changed = dict(receipt)
        changed["unexpected"] = True
        core = {
            key: value for key, value in changed.items() if key != "launch_sha256"
        }
        changed["launch_sha256"] = digest_object(core)
        path.write_text(canonical_json(changed) + "\n")
        with self.assertRaisesRegex(QualificationLaunchError, "schema changed"):
            read_qualification_launch_receipt(path, episode_root=self.root)

    def test_all_launcher_artifact_paths_reject_symlink_components(self) -> None:
        prelaunch = self.root / "outputs/prelaunch"
        prelaunch.mkdir(parents=True)
        outside = self.root / "outside-artifact"
        outside.write_text("outside\n", encoding="utf-8")
        for relative in (
            OUTPUT_RELATIVE,
            RECEIPT_RELATIVE,
            INTENT_RELATIVE,
            PROGRESS_RELATIVE,
            Path("outputs/prelaunch/.full_synthetic_qualification_launch_qualification003.lock"),
        ):
            with self.subTest(relative=str(relative)):
                target = self.root / relative
                target.symlink_to(outside)
                try:
                    with self.assertRaisesRegex(
                        QualificationLaunchError, "symlink component"
                    ):
                        self._launch_qualification(
                            test_job_id=81241,
                            episode_root=self.root,
                            submitter=lambda command: self.fail("must not submit"),
                            job_resolver=lambda name: self.fail("must not query"),
                        )
                finally:
                    target.unlink()

    def test_source_intermediate_and_episode_root_symlinks_are_rejected(self) -> None:
        outside = self.root / "outside-source"
        source_dir = self.root / "outputs/executor/slurm"
        source_dir.rename(outside)
        source_dir.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(SourceFreezeError, "symlink component"):
            self._launch_qualification(
                test_job_id=81242,
                episode_root=self.root,
                submitter=lambda command: self.fail("must not submit"),
                job_resolver=lambda name: self.fail("must not query"),
            )

        alias = self.root.parent / f"{self.root.name}-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        try:
            with self.assertRaisesRegex(SourceFreezeError, "symlink component"):
                self._launch_qualification(
                    test_job_id=81242,
                    episode_root=alias,
                    submitter=lambda command: self.fail("must not submit"),
                    job_resolver=lambda name: self.fail("must not query"),
                )
        finally:
            alias.unlink()

    def test_exact_name_parser_rejects_ambiguity_and_ignores_steps(self) -> None:
        name = "ep12-qual003-deadbeef"
        self.assertEqual(
            _unique_resolved_job_id(
                f"90125|{name}\n90125.batch|{name}\n90126|other\n", name
            ),
            "90125",
        )
        with self.assertRaisesRegex(QualificationLaunchError, "multiple Slurm jobs"):
            _unique_resolved_job_id(f"90125|{name}\n90126|{name}\n", name)

    @mock.patch("ep12_executor.qualification_launch.subprocess.run")
    def test_default_recovery_uses_one_sanitized_sacct_query(self, run: mock.Mock) -> None:
        name = "ep12-qual003-feedface"
        user = pwd.getpwuid(os.getuid()).pw_name
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout=f"90127|{name}|{user}\n90127.batch|{name}|{user}\n", stderr=""
        )
        self.assertEqual(
            _default_job_resolver(name, user),
            {"job_id": "90127", "job_name": name, "user": user},
        )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "sacct")
        environment = run.call_args.kwargs["env"]
        self.assertFalse(
            any(name.startswith(("SBATCH_", "SLURM_", "EP12_")) for name in environment)
        )

    @mock.patch("ep12_executor.qualification_launch.subprocess.run")
    def test_default_recovery_accepts_unique_accounting_match(self, run: mock.Mock) -> None:
        name = "ep12-qual003-cafebabe"
        user = pwd.getpwuid(os.getuid()).pw_name
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout=f"90128|{name}|{user}\n", stderr=""
        )
        self.assertEqual(
            _default_job_resolver(name, user),
            {"job_id": "90128", "job_name": name, "user": user},
        )
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
