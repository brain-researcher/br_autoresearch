#!/share/software/user/open/python/3.12.1/bin/python3.12
"""Focused hostile-live-source tests for archive-bound downstream wrappers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"
PINNED_BOOTSTRAP = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency/post36_staging_merged/"
    "ep12_executor/source_archive_bootstrap.py"
)
CANONICAL_ROOT = (
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
CANONICAL_SCRATCH = (
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
FIXED_PYTHON = "/share/software/user/open/python/3.12.1/bin/python3.12"
BOOTSTRAP_SHA256 = (
    "65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa"
)
WRAPPERS = {
    "test_executor.sbatch": (11, "synthetic_tests"),
    "qualify_methods.sbatch": (17, "scientific_qualification"),
    "check_production_readiness.sbatch": (25, "production_readiness"),
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def surface_rule() -> dict[str, object]:
    return {
        "include": ["SEARCH_POLICY.yaml", "outputs/executor/**"],
        "source_root_layout": "exact_clean_episode_source_v1",
        "excluded_directory_names": [
            ".git", ".hg", ".mypy_cache", ".nox", ".pycache-check",
            ".pytest_cache", ".ruff_cache", ".svn", ".temp", ".tox",
            "__pycache__", "patch", "temp", "tmp",
        ],
        "excluded_file_names": [".DS_Store", ".coverage", "patch", "temp", "tmp"],
        "excluded_suffixes": [
            ".diff", ".orig", ".patch", ".pyc", ".pyo", ".rej", ".swo",
            ".swp", ".temp", ".tmp", "~",
        ],
        "reserved_archive_manifest": "source_archive_manifest.json",
        "symlinks_allowed": False,
        "special_files_allowed": False,
    }


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, directories, files in os.walk(path, topdown=False):
        for name in files:
            item = Path(current) / name
            if not item.is_symlink():
                item.chmod(0o600)
        for name in directories:
            item = Path(current) / name
            if not item.is_symlink():
                item.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def create_archive(parent: Path, files: dict[str, bytes]) -> dict[str, str]:
    parent.mkdir(mode=0o700, parents=True)
    directories: set[str] = set()
    for name in files:
        for candidate in Path(name).parents:
            if str(candidate) != ".":
                directories.add(candidate.as_posix())
    ordered_directories = sorted(directories)
    records = [
        {
            "path": name,
            "size_bytes": len(files[name]),
            "sha256": sha(files[name]),
            "archive_mode": "0440",
        }
        for name in sorted(files)
    ]
    rule = surface_rule()
    tree = {
        "surface_rule": rule,
        "directories": ordered_directories,
        "files": records,
    }
    tree_sha = sha(canonical(tree))
    archive = parent / ("source-" + tree_sha)
    archive.mkdir(mode=0o700)
    for name in ordered_directories:
        (archive / name).mkdir(mode=0o700)
    for name, encoded in files.items():
        target = archive / name
        target.write_bytes(encoded)
        target.chmod(0o440)
    manifest_core = {
        "schema_version": "ep12.reviewed_source_archive_manifest.v1",
        "archive_kind": "content_addressed_reviewed_executor_source",
        "surface_rule": rule,
        "directory_count": len(ordered_directories),
        "file_count": len(records),
        "total_source_bytes": sum(record["size_bytes"] for record in records),
        "directories": ordered_directories,
        "files": records,
        "tree_sha256": tree_sha,
        "source_files_read_once": True,
        "symlinks_allowed": False,
        "complete": True,
    }
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = sha(canonical(manifest_core))
    manifest_bytes = canonical(manifest) + b"\n"
    manifest_path = archive / "source_archive_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o440)
    receipt_core = {
        "schema_version": "ep12.reviewed_source_archive_receipt.v1",
        "archive_name": archive.name,
        "manifest_name": manifest_path.name,
        "manifest_file_sha256": sha(manifest_bytes),
        "manifest_sha256": manifest["manifest_sha256"],
        "tree_sha256": tree_sha,
        "directory_count": len(ordered_directories),
        "file_count": len(records),
        "total_source_bytes": manifest["total_source_bytes"],
        "complete": True,
    }
    receipt = dict(receipt_core)
    receipt["receipt_sha256"] = sha(canonical(receipt_core))
    receipt_bytes = canonical(receipt) + b"\n"
    receipt_path = parent / "source_archive_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    receipt_path.chmod(0o440)
    for name in reversed(ordered_directories):
        (archive / name).chmod(0o550)
    archive.chmod(0o550)
    return {
        "archive": str(archive),
        "receipt_file": sha(receipt_bytes),
        "receipt": receipt["receipt_sha256"],
        "manifest_file": sha(manifest_bytes),
        "manifest": manifest["manifest_sha256"],
        "tree": tree_sha,
    }


RUNTIME_STUB = b"""import json
if __name__ == "__main__":
    print(json.dumps({"verified": True}, sort_keys=True, separators=(",", ":")))
"""


class Fixture:
    def __init__(self, wrapper: str, stage: str) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="ep12-downstream-wrapper-"))
        self.root = self.base / "episode"
        self.scratch = self.base / "scratch"
        self.root.mkdir(mode=0o700)
        self.scratch.mkdir(mode=0o700)
        text = (SLURM / wrapper).read_text(encoding="utf-8")
        text = text.replace(CANONICAL_ROOT, str(self.root))
        text = text.replace(CANONICAL_SCRATCH, str(self.scratch))
        self.wrapper = self.base / wrapper
        self.wrapper.write_text(text, encoding="utf-8")
        self.wrapper.chmod(0o700)
        paths = {
            "synthetic_tests": (
                "outputs/prelaunch/post36_downstream_launch/intent.json",
                "outputs/prelaunch/post36_downstream_launch/stages/synthetic_tests/receipt.json",
            ),
            "scientific_qualification": (
                "outputs/prelaunch/post36_downstream_launch/intent.json",
                "outputs/prelaunch/post36_downstream_launch/stages/scientific_qualification/receipt.json",
            ),
            "production_readiness": (
                "outputs/prelaunch/post36_downstream_launch/intent.json",
                "outputs/prelaunch/post36_downstream_launch/stages/production_readiness/receipt.json",
            ),
        }
        intent_name, receipt_name = paths[stage]
        self.intent = self.root / intent_name
        self.receipt = self.root / receipt_name
        self.intent.parent.mkdir(mode=0o700, parents=True)
        self.receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.intent.write_bytes(b"{}\n")
        self.receipt.write_bytes(b"{}\n")

    def close(self) -> None:
        remove_tree(self.base)

    def common(self, identity: dict[str, str]) -> list[str]:
        return [
            str(self.root),
            identity["archive"],
            identity["receipt_file"],
            identity["receipt"],
            identity["manifest_file"],
            identity["manifest"],
            identity["tree"],
            str(self.intent),
            sha(self.intent.read_bytes()),
            "9" * 64,
            str(self.receipt),
        ]

    def run(
        self, arguments: list[str], *, hostile: Path
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "SLURM_JOB_ID": "990101",
            "SLURM_JOB_NAME": "ep12-archive-wrapper-test",
            "PATH": "/hostile/path",
            "LD_LIBRARY_PATH": "/hostile/library",
            "PYTHONPATH": str(hostile),
            "PYTHONHOME": str(self.base / "hostile-pythonhome"),
            "HOME": str(self.base / "hostile-home"),
            "USER": os.environ.get("USER", "zijiao"),
            "LOGNAME": os.environ.get("LOGNAME", "zijiao"),
        }
        return subprocess.run(
            ["/bin/bash", str(self.wrapper), *arguments],
            cwd=self.base,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )

    def hostile_package(self, names: tuple[str, ...]) -> tuple[Path, Path]:
        hostile = self.base / "hostile"
        package = hostile / "ep12_executor"
        package.mkdir(mode=0o700, parents=True)
        sentinel = self.base / "hostile-imported"
        code = (
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('bad')\n"
        )
        (package / "__init__.py").write_text(code, encoding="utf-8")
        for name in names:
            (package / name).write_text(code, encoding="utf-8")
        return hostile, sentinel


class ArchiveDownstreamWrapperTests(unittest.TestCase):
    maxDiff = None

    def test_syntax_and_frozen_contract(self) -> None:
        paths = [str(SLURM / name) for name in WRAPPERS]
        result = subprocess.run(
            ["/bin/bash", "-n", *paths],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, (count, stage) in WRAPPERS.items():
            with self.subTest(wrapper=name):
                text = (SLURM / name).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --export=NONE", text)
                self.assertIn(f"[[ $# -eq {count} ]]", text)
                self.assertIn(BOOTSTRAP_SHA256, text)
                self.assertIn(f"EP12_PYTHON={FIXED_PYTHON}", text)
                self.assertIn("export PATH=/usr/bin:/bin", text)
                self.assertIn(
                    "export LD_LIBRARY_PATH=/share/software/user/open/python/3.12.1/lib",
                    text,
                )
                self.assertIn(f"--expected-stage {stage}", text)
                self.assertIn("--wrapper-arguments", text)
                self.assertIn("-P -B", text)
                self.assertNotIn("45332084", text)
                self.assertNotIn("source_freeze", text)
                self.assertLess(
                    text.index(
                        '"${EP12_PYTHON}" -I -S "${EP12_BOOTSTRAP}" '
                        "verify-and-stage"
                    ),
                    text.index("-m ep12_executor"),
                )

        qualification_text = (SLURM / "qualify_methods.sbatch").read_text(
            encoding="utf-8"
        )
        for required in (
            '--episode-root "${EP12_ROOT}"',
            '--policy "${EP12_STAGED_ROOT}/SEARCH_POLICY.yaml"',
            '--contract "${EP12_STAGED_EXECUTOR}/SCIENTIFIC_CONTRACT_POST36.yaml"',
            '--qualification-spec "${EP12_STAGED_EXECUTOR}/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"',
        ):
            self.assertIn(required, qualification_text)

    def test_wrong_bootstrap_fails_before_execution(self) -> None:
        fixture = Fixture("test_executor.sbatch", "synthetic_tests")
        try:
            archive = fixture.base / ("source-" + "5" * 64)
            bootstrap = archive / "outputs/executor/ep12_executor/source_archive_bootstrap.py"
            bootstrap.parent.mkdir(mode=0o700, parents=True)
            sentinel = fixture.base / "bad-bootstrap-ran"
            bootstrap.write_text(
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            identity = {
                "archive": str(archive),
                "receipt_file": "1" * 64,
                "receipt": "2" * 64,
                "manifest_file": "3" * 64,
                "manifest": "4" * 64,
                "tree": "5" * 64,
            }
            hostile, hostile_sentinel = fixture.hostile_package(
                ("post36_downstream_receipt_bound.py",)
            )
            result = fixture.run(fixture.common(identity), hostile=hostile)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("archive bootstrap SHA-256 mismatch", result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertFalse(hostile_sentinel.exists())
        finally:
            fixture.close()

    def test_wrong_launch_intent_hash_fails_before_bootstrap(self) -> None:
        fixture = Fixture("test_executor.sbatch", "synthetic_tests")
        try:
            archive = fixture.base / ("source-" + "5" * 64)
            bootstrap = archive / "outputs/executor/ep12_executor/source_archive_bootstrap.py"
            bootstrap.parent.mkdir(mode=0o700, parents=True)
            sentinel = fixture.base / "bootstrap-ran"
            bootstrap.write_text(
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('bad')\n",
                encoding="utf-8",
            )
            identity = {
                "archive": str(archive),
                "receipt_file": "1" * 64,
                "receipt": "2" * 64,
                "manifest_file": "3" * 64,
                "manifest": "4" * 64,
                "tree": "5" * 64,
            }
            arguments = fixture.common(identity)
            arguments[8] = "0" * 64
            hostile, hostile_sentinel = fixture.hostile_package(
                ("post36_downstream_receipt_bound.py",)
            )
            result = fixture.run(arguments, hostile=hostile)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("launch-intent file SHA-256 mismatch", result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertFalse(hostile_sentinel.exists())
        finally:
            fixture.close()

    def test_group_writable_private_parent_fails_before_bootstrap(self) -> None:
        fixture = Fixture("test_executor.sbatch", "synthetic_tests")
        try:
            archive = fixture.base / ("source-" + "5" * 64)
            bootstrap = archive / "outputs/executor/ep12_executor/source_archive_bootstrap.py"
            bootstrap.parent.mkdir(mode=0o700, parents=True)
            bootstrap.write_bytes(PINNED_BOOTSTRAP.read_bytes())
            private = fixture.scratch / "archive_wrapper_tmp"
            private.mkdir(mode=0o700, parents=True)
            private.chmod(0o770)
            identity = {
                "archive": str(archive),
                "receipt_file": "1" * 64,
                "receipt": "2" * 64,
                "manifest_file": "3" * 64,
                "manifest": "4" * 64,
                "tree": "5" * 64,
            }
            hostile, sentinel = fixture.hostile_package(
                ("post36_downstream_receipt_bound.py",)
            )
            result = fixture.run(fixture.common(identity), hostile=hostile)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("group- or world-writable", result.stderr)
            self.assertFalse(sentinel.exists())
        finally:
            fixture.close()

    def test_executor_tests_ignore_hostile_live_source(self) -> None:
        fixture = Fixture("test_executor.sbatch", "synthetic_tests")
        try:
            marker = fixture.root / "authenticated-tests-ran"
            test_source = f"""import unittest
from pathlib import Path
from ep12_executor import post36_downstream_receipt_bound
class OriginTest(unittest.TestCase):
    def test_origin(self):
        origin = Path(post36_downstream_receipt_bound.__file__).resolve()
        self.assertIn("source-", str(origin))
        Path({str(marker)!r}).write_text(str(origin))
""".encode()
            identity = create_archive(
                fixture.base / "archive-publication",
                {
                    "SEARCH_POLICY.yaml": b"schema_version: test\n",
                    "outputs/executor/ep12_executor/__init__.py": b"",
                    "outputs/executor/ep12_executor/source_archive_bootstrap.py": PINNED_BOOTSTRAP.read_bytes(),
                    "outputs/executor/ep12_executor/post36_downstream_receipt_bound.py": RUNTIME_STUB,
                    "outputs/executor/tests/test_origin.py": test_source,
                },
            )
            hostile, sentinel = fixture.hostile_package(
                ("post36_downstream_receipt_bound.py",)
            )
            result = fixture.run(fixture.common(identity), hostile=hostile)
            self.assertEqual(
                result.returncode,
                0,
                "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr,
            )
            self.assertFalse(sentinel.exists())
            self.assertIn("source-" + identity["tree"], marker.read_text())
        finally:
            fixture.close()

    def test_qualification_ignores_hostile_live_source(self) -> None:
        fixture = Fixture("qualify_methods.sbatch", "scientific_qualification")
        try:
            marker = fixture.root / "authenticated-qualification-ran"
            main = f"""from pathlib import Path
from . import scientific_models, scientific_qualification, scientific_synthetic
Path({str(marker)!r}).write_text("|".join((
    str(Path(scientific_models.__file__).resolve()),
    str(Path(scientific_qualification.__file__).resolve()),
    str(Path(scientific_synthetic.__file__).resolve()),
)))
""".encode()
            identity = create_archive(
                fixture.base / "archive-publication",
                {
                    "SEARCH_POLICY.yaml": b"schema_version: test\n",
                    "outputs/executor/ep12_executor/__init__.py": b"",
                    "outputs/executor/ep12_executor/__main__.py": main,
                    "outputs/executor/ep12_executor/source_archive_bootstrap.py": PINNED_BOOTSTRAP.read_bytes(),
                    "outputs/executor/ep12_executor/post36_downstream_receipt_bound.py": RUNTIME_STUB,
                    "outputs/executor/ep12_executor/scientific_models.py": b"ORIGIN='stage'\n",
                    "outputs/executor/ep12_executor/scientific_qualification.py": b"ORIGIN='stage'\n",
                    "outputs/executor/ep12_executor/scientific_synthetic.py": b"ORIGIN='stage'\n",
                    "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml": b"schema_version: contract\n",
                    "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml": b"schema_version: qualification\n",
                },
            )
            hostile, sentinel = fixture.hostile_package(
                (
                    "__main__.py",
                    "post36_downstream_receipt_bound.py",
                    "scientific_models.py",
                    "scientific_qualification.py",
                    "scientific_synthetic.py",
                )
            )
            output = (
                fixture.root
                / "outputs/prelaunch/scientific_qualification_qualification003"
            )
            transition = (
                fixture.root
                / "outputs/prelaunch/scientific_contract_transition/manifest.json"
            )
            transition.parent.mkdir(mode=0o700, parents=True)
            transition.write_bytes(b"{}\n")
            arguments = fixture.common(identity) + [
                str(output), "72", "8", str(transition),
                sha(transition.read_bytes()), "7" * 64,
            ]
            result = fixture.run(arguments, hostile=hostile)
            self.assertEqual(
                result.returncode,
                0,
                "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr,
            )
            self.assertFalse(sentinel.exists())
            self.assertIn("source-" + identity["tree"], marker.read_text())
        finally:
            fixture.close()

    def test_archive_procedure_boundary_stops_at_resource_hold(self) -> None:
        archive_wrapper = (
            SLURM / "run_procedure_lock_from_archive.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "EP12_PROCEDURE_LOCK_LAUNCH_MODE=hold EP12_AUTO_LAUNCH_FULL_NULL=0",
            archive_wrapper,
        )
        self.assertNotIn("EP12_AUTO_LAUNCH_FULL_NULL=1", archive_wrapper)
        self.assertIn(
            '"${EP12_STAGED_EXECUTOR}/slurm/run_procedure_lock.sbatch"',
            archive_wrapper,
        )

        procedure = (
            PINNED_BOOTSTRAP.parents[1] / "slurm/run_procedure_lock.sbatch"
        ).read_text(encoding="utf-8")
        pending = '"action": "resource_authorization_pending"'
        hold_exit = (
            'if [[ "${EP12_AUTO_LAUNCH_FULL_NULL}" == "0" ]]; then\n'
            "    exit 0"
        )
        dispatch = "python3 -m ep12_executor.procedure_lock_dispatch"
        self.assertIn(pending, procedure)
        self.assertIn(hold_exit, procedure)
        self.assertIn(dispatch, procedure)
        self.assertLess(procedure.index(pending), procedure.index(hold_exit))
        self.assertLess(procedure.index(hold_exit), procedure.index(dispatch))

    def test_readiness_binds_dynamic_replacement_and_staged_modules(self) -> None:
        fixture = Fixture("check_production_readiness.sbatch", "production_readiness")
        try:
            marker = fixture.root / "authenticated-readiness-handoff"
            top_module = b"""import hashlib,json
from pathlib import Path
def verify_post36_start_authorization(*, episode_root, authorization_path=None):
    path = Path(authorization_path)
    value = json.loads(path.read_text())
    value["authorization_path"] = str(path)
    value["authorization_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value
"""
            handoff = f"""import json
from pathlib import Path
import sys
from ep12_executor.post36_unattended_controller import verify_post36_start_authorization
if __name__ == "__main__":
    arguments = sys.argv[1:]
    root = Path(arguments[arguments.index("--episode-root") + 1])
    authority_path = root / "outputs/prelaunch/post36_orchestration/start_authorization.json"
    evidence = verify_post36_start_authorization(
        episode_root=root, authorization_path=authority_path
    )
    replacement = evidence["replacement"]
    Path({str(marker)!r}).write_text(
        str(Path(__file__).resolve()) + "\\n"
        + " ".join(arguments) + "\\n"
        + str(replacement["slurm_job_id"]) + "\\n"
        + str(replacement["archive"]["archived_root"])
    )
""".encode()
            inert_main = b"if __name__ == '__main__':\n    pass\n"
            contract = b"schema_version: contract\n"
            qualification_spec = b"schema_version: qualification\n"
            scientific_models = b"ORIGIN='stage'\n"
            files = {
                "SEARCH_POLICY.yaml": b"schema_version: policy\n",
                "outputs/executor/ep12_executor/__init__.py": b"",
                "outputs/executor/ep12_executor/source_archive_bootstrap.py": PINNED_BOOTSTRAP.read_bytes(),
                "outputs/executor/ep12_executor/post36_downstream_receipt_bound.py": RUNTIME_STUB,
                "outputs/executor/ep12_executor/archive_nested_qualification.py": b"ORIGIN='stage'\n",
                "outputs/executor/ep12_executor/archive_slurm_handoff.py": handoff,
                "outputs/executor/ep12_executor/post36_unattended_controller.py": top_module,
                "outputs/executor/ep12_executor/scientific_transition_binding.py": inert_main,
                "outputs/executor/ep12_executor/production_readiness.py": inert_main,
                "outputs/executor/ep12_executor/scientific_models.py": scientific_models,
                "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml": contract,
                "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml": qualification_spec,
                "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json": b"{}\n",
                "outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml": b"schema_version: budget\n",
                "outputs/executor/slurm/run_post36_tail_from_archive.sbatch": b"#!/bin/bash\nexit 0\n",
            }
            identity = create_archive(
                fixture.base / "archive-publication", files
            )
            qualification = (
                fixture.root
                / "outputs/prelaunch/post36_downstream_launch/stages/scientific_qualification/receipt.json"
            )
            qualification.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            qualification.write_bytes(b"{}\n")
            qualification_dir = (
                fixture.root
                / "outputs/prelaunch/scientific_qualification_qualification003"
            )
            qualification_dir.mkdir(mode=0o700)
            transition = (
                fixture.root
                / "outputs/prelaunch/scientific_contract_transition/manifest.json"
            )
            transition.parent.mkdir(mode=0o700, parents=True)
            transition.write_bytes(b"{}\n")
            replacement_job_id = "9101"
            authority = (
                fixture.root
                / "outputs/prelaunch/post36_orchestration/start_authorization.json"
            )
            authority.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            authority_evidence = {
                "start_authorization_sha256": "8" * 64,
                "replacement": {
                    "slurm_job_id": replacement_job_id,
                    "archive": {
                        "archived_root": identity["archive"],
                        "archive_receipt_file_sha256": identity["receipt_file"],
                        "archive_receipt_sha256": identity["receipt"],
                        "manifest_file_sha256": identity["manifest_file"],
                        "manifest_sha256": identity["manifest"],
                        "tree_sha256": identity["tree"],
                    },
                },
                "obsolete_job_is_execution_dependency": False,
                "real_connectivity_accessed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            }
            authority.write_bytes(canonical(authority_evidence) + b"\n")
            authority_file_sha = sha(authority.read_bytes())
            hostile, sentinel = fixture.hostile_package(
                (
                    "post36_downstream_receipt_bound.py",
                    "archive_nested_qualification.py",
                    "archive_slurm_handoff.py",
                    "post36_unattended_controller.py",
                    "scientific_transition_binding.py",
                    "production_readiness.py",
                )
            )
            arguments = fixture.common(identity) + [
                str(qualification),
                str(qualification_dir),
                str(transition),
                sha(transition.read_bytes()),
                "1" * 64,
                "2" * 64,
                "3" * 64,
                sha(contract),
                sha(qualification_spec),
                sha(scientific_models),
                str(authority),
                authority_file_sha,
                "8" * 64,
                replacement_job_id,
            ]
            result = fixture.run(arguments, hostile=hostile)
            self.assertEqual(
                result.returncode,
                0,
                "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr,
            )
            self.assertFalse(sentinel.exists())
            text = marker.read_text()
            self.assertIn("source-" + identity["tree"], text)
            self.assertIn(replacement_job_id, text)
            self.assertIn("readiness_to_archive_tail", text)
            self.assertNotIn("source_freeze", text)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
