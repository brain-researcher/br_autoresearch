#!/share/software/user/open/python/3.12.1/bin/python3.12
"""Focused fixtures for canonical downstream runtime and qualification binding."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MERGED = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency/post36_staging_merged"
)
if str(MERGED) not in sys.path:
    sys.path.insert(0, str(MERGED))
import ep12_executor  # noqa: E402


def load_candidate(name: str, relative: str):
    full_name = f"ep12_executor.{name}"
    spec = importlib.util.spec_from_file_location(full_name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_candidate(
    "post36_downstream_receipt_bound_candidate",
    "ep12_executor/post36_downstream_receipt_bound.py",
)
adapter = load_candidate(
    "archive_nested_qualification_candidate",
    "ep12_executor/archive_nested_qualification.py",
)


archive_handoff = load_candidate(
    "archive_slurm_handoff_candidate",
    "ep12_executor/archive_slurm_handoff.py",
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DownstreamRuntimeVerifierTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep12-downstream-runtime-")
        base = Path(temporary.name)
        root = base / "episode"
        staged = base / "stage"
        executor = staged / "outputs/executor"
        package = executor / "ep12_executor"
        package.mkdir(parents=True)
        for name in (
            "post36_downstream_receipt_bound.py",
            "post36_downstream_launch.py",
            "post36_unattended_controller.py",
            "post36_receipt_bound.py",
        ):
            (package / name).write_text("# fixture\n", encoding="utf-8")
        archive_root = base / "archive"
        wrapper = archive_root / "outputs/executor/slurm/test_executor.sbatch"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
        archive = {
            "archived_root": str(archive_root),
            "archive_receipt_file_sha256": "1" * 64,
            "archive_receipt_sha256": "2" * 64,
            "manifest_file_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "tree_sha256": "5" * 64,
        }
        stage_receipt = (
            root
            / runtime.TRANSACTION_RELATIVE
            / "stages/synthetic_tests/receipt.json"
        )
        stage_receipt.parent.mkdir(parents=True)
        stage_receipt.write_bytes(b"{}\n")
        intent = root / runtime.INTENT_RELATIVE
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.write_bytes(b"{}\n")
        stage_result = base / "stage-result.json"
        stage_result.write_bytes(b"{}\n")
        arguments = ["argument-one", "argument-two"]
        stage = {
            "stage": "synthetic_tests",
            "slurm_job_id": "701",
            "slurm_job_name": "ep12-synthetic-tests-fixture",
            "submission_user": "zijiao",
            "predecessor_job_id": "700",
            "dependency": "afterok:700",
            "arguments": arguments,
            "wrapper": str(wrapper),
            "wrapper_sha256": sha(wrapper),
            "stage_receipt_sha256": "6" * 64,
            "release_authorization_file_sha256": "7" * 64,
            "release_authorization_sha256": "8" * 64,
            "release_receipt_file_sha256": "9" * 64,
            "release_receipt_sha256": "a" * 64,
        }
        controller = {
            "intent_sha256": "b" * 64,
            "replacement": {"archive": archive},
            "stages": {
                "publisher": {"slurm_job_id": "698"},
                "downstream": {"slurm_job_id": "700"},
            },
        }
        evidence = {
            "intent_path": str(intent),
            "intent_file_sha256": sha(intent),
            "intent_sha256": "c" * 64,
            "archive": archive,
            "stages": {"synthetic_tests": stage},
        }
        return (
            temporary, root, staged, executor, package, wrapper,
            stage_receipt, intent, stage_result, arguments, archive,
            controller, evidence,
        )

    def test_exact_runtime_binding_and_release_proof(self) -> None:
        fixture = self.fixture()
        (
            temporary, root, staged, executor, package, wrapper,
            stage_receipt, intent, stage_result, arguments, archive,
            controller, evidence,
        ) = fixture
        self.addCleanup(temporary.cleanup)
        stage_result_value = {
            "staged_root": str(staged),
            "upstream_archived_root": archive["archived_root"],
        }
        patches = (
            mock.patch.object(runtime, "EPISODE_ROOT", root),
            mock.patch.object(runtime, "__file__", str(package / "post36_downstream_receipt_bound.py")),
            mock.patch.object(runtime._downstream_module, "__file__", str(package / "post36_downstream_launch.py")),
            mock.patch.object(runtime._controller_module, "__file__", str(package / "post36_unattended_controller.py")),
            mock.patch.object(runtime._base_runtime_module, "__file__", str(package / "post36_receipt_bound.py")),
            mock.patch.object(runtime, "verify_post36_start_authorization", return_value=controller),
            mock.patch.object(runtime, "verify_post36_downstream_receipt", return_value=evidence),
            mock.patch.object(runtime, "_stage_result", return_value=stage_result_value),
            mock.patch.object(runtime, "_verify_ep12_import_origins", return_value=None),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            result = runtime.verify_downstream_stage_runtime(
                episode_root=root,
                orchestration_intent_path=intent,
                expected_intent_file_sha256=sha(intent),
                expected_intent_sha256="c" * 64,
                stage_receipt_path=stage_receipt,
                expected_stage="synthetic_tests",
                executed_wrapper=wrapper,
                slurm_job_id="701",
                slurm_job_name="ep12-synthetic-tests-fixture",
                bootstrap_stage_result_path=stage_result,
                staged_root=staged,
                wrapper_arguments=arguments,
            )
        self.assertTrue(result["all_stage_and_release_receipts_verified"])
        self.assertEqual(result["stage_receipt_file_sha256"], sha(stage_receipt))
        self.assertEqual(result["archive"], archive)

    def test_runtime_rejects_mutated_arguments(self) -> None:
        fixture = self.fixture()
        (
            temporary, root, staged, _executor, package, wrapper,
            stage_receipt, intent, stage_result, _arguments, archive,
            controller, evidence,
        ) = fixture
        self.addCleanup(temporary.cleanup)
        with (
            mock.patch.object(runtime, "EPISODE_ROOT", root),
            mock.patch.object(runtime, "__file__", str(package / "post36_downstream_receipt_bound.py")),
            mock.patch.object(runtime._downstream_module, "__file__", str(package / "post36_downstream_launch.py")),
            mock.patch.object(runtime._controller_module, "__file__", str(package / "post36_unattended_controller.py")),
            mock.patch.object(runtime._base_runtime_module, "__file__", str(package / "post36_receipt_bound.py")),
            mock.patch.object(runtime, "verify_post36_start_authorization", return_value=controller),
            mock.patch.object(runtime, "verify_post36_downstream_receipt", return_value=evidence),
            mock.patch.object(runtime, "_stage_result", return_value={"staged_root": str(staged)}),
            mock.patch.object(runtime, "_verify_ep12_import_origins", return_value=None),
        ):
            with self.assertRaisesRegex(
                runtime.Post36DownstreamReceiptBoundError,
                "positional arguments",
            ):
                runtime.verify_downstream_stage_runtime(
                    episode_root=root,
                    orchestration_intent_path=intent,
                    expected_intent_file_sha256=sha(intent),
                    expected_intent_sha256="c" * 64,
                    stage_receipt_path=stage_receipt,
                    expected_stage="synthetic_tests",
                    executed_wrapper=wrapper,
                    slurm_job_id="701",
                    slurm_job_name="ep12-synthetic-tests-fixture",
                    bootstrap_stage_result_path=stage_result,
                    staged_root=staged,
                    wrapper_arguments=["mutated"],
                )


class ArchiveNestedQualificationTests(unittest.TestCase):
    def test_adapter_binds_exact_test_dependency_transition_and_sources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-nested-qualification-") as temporary:
            base = Path(temporary)
            root = base / "episode"
            executor = base / "stage/outputs/executor"
            package = executor / "ep12_executor"
            package.mkdir(parents=True)
            adapter_file = package / "archive_nested_qualification.py"
            downstream_file = package / "post36_downstream_launch.py"
            adapter_file.write_text("# fixture\n", encoding="utf-8")
            downstream_file.write_text("# fixture\n", encoding="utf-8")
            contract = executor / "SCIENTIFIC_CONTRACT_POST36.yaml"
            wrapper = executor / "slurm/qualify_methods.sbatch"
            contract.write_text("schema_version: fixture\n", encoding="utf-8")
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
            source_targets = {
                "scientific_contract_sha256": Path(
                    "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml"
                ),
                "slurm_script_sha256": Path(
                    "outputs/executor/slurm/qualify_methods.sbatch"
                ),
            }
            archive = {"archived_root": str(base / "archive"), "tree_sha256": "1" * 64}
            controller = {
                "intent_sha256": "2" * 64,
                "replacement": {"archive": archive},
                "stages": {
                    "publisher": {"slurm_job_id": "801"},
                    "downstream": {"slurm_job_id": "802"},
                },
            }
            launch = (
                root
                / adapter.DOWNSTREAM_TRANSACTION_RELATIVE
                / "stages/scientific_qualification/receipt.json"
            )
            launch.parent.mkdir(parents=True)
            launch.write_bytes(b"{}\n")
            output = root / "outputs/prelaunch/scientific_qualification_qualification003"
            transition = {
                "transition_manifest_path": str(
                    root / "outputs/prelaunch/scientific_contract_transition/manifest.json"
                ),
                "transition_manifest_file_sha256": "3" * 64,
                "transition_receipt_sha256": "4" * 64,
            }
            arguments = ["unused"] * 17
            arguments[7] = str(root / adapter.DOWNSTREAM_INTENT_RELATIVE)
            arguments[10] = str(launch)
            arguments[11] = str(output)
            arguments[12] = "72"
            arguments[13] = "8"
            arguments[14] = transition["transition_manifest_path"]
            arguments[15] = transition["transition_manifest_file_sha256"]
            arguments[16] = transition["transition_receipt_sha256"]
            test = {"slurm_job_id": "803"}
            qualification = {
                "slurm_job_id": "804",
                "slurm_job_name": "ep12-scientific-qualification-fixture",
                "submission_user": "zijiao",
                "predecessor_job_id": "803",
                "dependency": "afterok:803",
                "arguments": arguments,
                "wrapper": str(base / "archive/outputs/executor/slurm/qualify_methods.sbatch"),
                "wrapper_sha256": sha(wrapper),
                "stage_receipt_sha256": "5" * 64,
                "release_authorization_file_sha256": "6" * 64,
                "release_authorization_sha256": "7" * 64,
                "release_receipt_file_sha256": "8" * 64,
                "release_receipt_sha256": "9" * 64,
            }
            downstream = {
                "intent_path": str(root / adapter.DOWNSTREAM_INTENT_RELATIVE),
                "intent_file_sha256": "a" * 64,
                "intent_sha256": "b" * 64,
                "archive": archive,
                "scientific_transition": transition,
                "stages": {
                    "synthetic_tests": test,
                    "scientific_qualification": qualification,
                },
            }
            with (
                mock.patch.object(adapter, "EPISODE_ROOT", root),
                mock.patch.object(adapter, "__file__", str(adapter_file)),
                mock.patch.object(adapter._downstream_module, "__file__", str(downstream_file)),
                mock.patch.object(adapter, "QUALIFICATION_HASH_TARGETS", source_targets),
                mock.patch.object(adapter, "verify_post36_start_authorization", return_value=controller),
                mock.patch.object(adapter, "verify_post36_downstream_receipt", return_value=downstream),
            ):
                evidence = adapter.verify_archive_nested_qualification_launch(
                    episode_root=root,
                    executor_root=executor,
                    launch_path=launch,
                )
            self.assertEqual(evidence["test_job_id"], "803")
            self.assertEqual(evidence["slurm_job_id"], "804")
            self.assertEqual(evidence["scientific_contract_transition"], transition)
            self.assertTrue(evidence["all_stage_and_release_receipts_verified"])
            view = adapter.qualification_launch_view(evidence)
            self.assertEqual(view["replicates_per_core_error_cell"], 72)
            self.assertEqual(view["scientific_contract_sha256"], sha(contract))


class ArchiveSlurmAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="ep12-archive-slurm-authority-"
        )
        self.root = Path(self.temporary.name) / "episode"
        self.root.mkdir()
        self.executor = Path(self.temporary.name) / "executor"
        self.archive = {
            "archived_root": str(Path(self.temporary.name) / "source-archive"),
            "archive_receipt_file_sha256": "1" * 64,
            "archive_receipt_sha256": "2" * 64,
            "manifest_file_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "tree_sha256": "5" * 64,
        }
        authorization_path = (
            self.root
            / "outputs/prelaunch/post36_orchestration/start_authorization.json"
        )
        self.start = {
            "intent_sha256": "a" * 64,
            "authorization_path": str(authorization_path),
            "authorization_file_sha256": "b" * 64,
            "start_authorization_sha256": "c" * 64,
            "replacement": {
                "slurm_job_id": "703",
                "archive": copy.deepcopy(self.archive),
            },
            "stages": {
                "publisher": {"slurm_job_id": "701"},
                "downstream": {"slurm_job_id": "702"},
            },
        }
        self.downstream = {
            "archive": copy.deepcopy(self.archive),
            "receipt_path": str(
                self.root
                / "outputs/prelaunch/post36_downstream_launch/receipt.json"
            ),
            "receipt_file_sha256": "6" * 64,
            "receipt_sha256": "7" * 64,
            "replacement_job_id": "703",
            "upstream_start_authorization": {
                "path": str(authorization_path),
                "file_sha256": "b" * 64,
                "semantic_sha256": "c" * 64,
            },
        }
        self.final = {
            "intent_sha256": "a" * 64,
            "receipt_path": str(
                self.root / archive_handoff.UPSTREAM_RECEIPT_RELATIVE
            ),
            "receipt_file_sha256": "d" * 64,
            "receipt_sha256": "e" * 64,
            "replacement": {
                "slurm_job_id": "703",
                "archive": copy.deepcopy(self.archive),
            },
            "start_authorization": {
                "path": str(authorization_path),
                "file_sha256": "b" * 64,
                "start_authorization_sha256": "c" * 64,
            },
        }
        self.final_path = self.root / archive_handoff.UPSTREAM_RECEIPT_RELATIVE

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def call_authority(
        self,
        kind: str,
        *,
        downstream: dict | None = None,
        final: dict | None = None,
    ):
        with (
            mock.patch.object(
                archive_handoff,
                "verify_post36_start_authorization",
                return_value=copy.deepcopy(self.start),
            ),
            mock.patch.object(
                archive_handoff,
                "verify_post36_downstream_receipt",
                return_value=copy.deepcopy(
                    self.downstream if downstream is None else downstream
                ),
            ),
            mock.patch.object(
                archive_handoff,
                "verify_post36_orchestration_receipt",
                return_value=copy.deepcopy(self.final if final is None else final),
            ) as final_verifier,
        ):
            result = archive_handoff._authority(
                self.root, self.executor, kind=kind
            )
        return result, final_verifier

    def test_final_receipt_is_optional_only_while_absent_before_procedure(self) -> None:
        authority, final_verifier = self.call_authority(
            "readiness_to_archive_tail"
        )
        self.assertIsNone(authority["final"])
        final_verifier.assert_not_called()

        with self.assertRaisesRegex(
            archive_handoff.ArchiveSlurmHandoffError,
            "procedure-lock boundary requires",
        ):
            self.call_authority("archive_tail_to_procedure_lock")

    def test_present_final_receipt_is_always_verified_and_exact(self) -> None:
        self.final_path.parent.mkdir(parents=True)
        self.final_path.write_bytes(b"{}\n")
        authority, final_verifier = self.call_authority(
            "archive_tail_to_next_tail"
        )
        self.assertEqual(authority["final"], self.final)
        final_verifier.assert_called_once_with(
            episode_root=self.root, receipt_path=self.final_path
        )

        stable_tail = archive_handoff._stable_authority(authority)
        self.assertEqual(
            stable_tail["controller_authority_kind"], "start_authorization"
        )
        self.assertEqual(
            stable_tail["controller_authority_path"],
            self.start["authorization_path"],
        )
        procedure, _ = self.call_authority("archive_tail_to_procedure_lock")
        stable_procedure = archive_handoff._stable_authority(procedure)
        self.assertEqual(
            stable_procedure["controller_authority_kind"], "final_top_receipt"
        )
        self.assertEqual(
            stable_procedure["controller_authority_path"],
            self.final["receipt_path"],
        )

        changed = copy.deepcopy(self.final)
        changed["start_authorization"]["file_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            archive_handoff.ArchiveSlurmHandoffError,
            "differs from immutable start authority",
        ):
            self.call_authority("archive_tail_to_next_tail", final=changed)

    def test_downstream_binds_exact_start_archive_and_dynamic_replacement(self) -> None:
        cases = {}
        changed = copy.deepcopy(self.downstream)
        changed["replacement_job_id"] = "704"
        cases["dynamic replacement"] = changed
        changed = copy.deepcopy(self.downstream)
        changed["upstream_start_authorization"]["semantic_sha256"] = "0" * 64
        cases["start authorization"] = changed
        changed = copy.deepcopy(self.downstream)
        changed["archive"]["tree_sha256"] = "0" * 64
        cases["archive"] = changed
        for label, downstream in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                archive_handoff.ArchiveSlurmHandoffError,
                "start authorization, archive, or replacement identity",
            ):
                self.call_authority(
                    "readiness_to_archive_tail", downstream=downstream
                )


if __name__ == "__main__":
    unittest.main()
