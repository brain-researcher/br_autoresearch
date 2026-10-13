#!/share/software/user/open/python/3.12.1/bin/python3.12
"""Black-box and structural tests for the four archive-bound post36 wrappers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
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
FIXED_LD = "/share/software/user/open/python/3.12.1/lib"
BOOTSTRAP_SHA256 = (
    "65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa"
)
PROBE_SHA256 = "eb46416bef86fcc0ffeb493ee772be1d68706e3e67f2ecbd4f708d22f9460a42"
SPEC_SHA256 = "76539bd7ede65609a56a493c22232c49b4f664d9a7bbd1d346522d07f63e57c2"
LEGACY_WRAPPER_SHA256 = (
    "308eb7eb45935c99500b320eabcfad63f6d60857a4f371e092859604eace06f2"
)

WRAPPERS = {
    "run_contract_invariance_probe_from_archive.sbatch": (12, "post36_probe"),
    "compare_contract_invariance_from_archive.sbatch": (14, "comparison"),
    "publish_scientific_transition_from_archive.sbatch": (15, "publisher"),
    "launch_post36_downstream_from_archive.sbatch": (12, "downstream"),
}

COMMON_ASSIGNMENTS = [
    "EP12_ROOT=$1",
    "EP12_ARCHIVE_ROOT=$2",
    "EP12_ARCHIVE_RECEIPT_FILE_SHA256=$3",
    "EP12_ARCHIVE_RECEIPT_SHA256=$4",
    "EP12_ARCHIVE_MANIFEST_FILE_SHA256=$5",
    "EP12_ARCHIVE_MANIFEST_SHA256=$6",
    "EP12_ARCHIVE_TREE_SHA256=$7",
    "EP12_ORCHESTRATION_INTENT=$8",
    "EP12_ORCHESTRATION_INTENT_FILE_SHA256=$9",
    "EP12_ORCHESTRATION_INTENT_SHA256=${10}",
    "EP12_STAGE_RECEIPT=${11}",
]


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


def source_surface_rule() -> dict[str, object]:
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


def make_writable_and_remove(path: Path) -> None:
    if not path.exists():
        return
    for current, directories, files in os.walk(path, topdown=False):
        for name in files:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                candidate.chmod(0o600)
        for name in directories:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                candidate.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def create_reviewed_archive(parent: Path, files: dict[str, bytes]) -> dict[str, str]:
    parent.mkdir(mode=0o700, parents=True)
    directories: set[str] = set()
    for name in files:
        relative = Path(name)
        for candidate in relative.parents:
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
    rule = source_surface_rule()
    tree = {
        "surface_rule": rule,
        "directories": ordered_directories,
        "files": records,
    }
    tree_sha256 = sha(canonical(tree))
    archive = parent / ("source-" + tree_sha256)
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
        "total_source_bytes": sum(item["size_bytes"] for item in records),
        "directories": ordered_directories,
        "files": records,
        "tree_sha256": tree_sha256,
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
        "tree_sha256": tree_sha256,
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
        "tree": tree_sha256,
    }


class WrapperFixture:
    def __init__(self, wrapper_name: str, stage: str) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="ep12-wrapper-test-"))
        self.root = self.base / "episode"
        self.scratch = self.base / "scratch"
        self.root.mkdir(mode=0o700)
        self.scratch.mkdir(mode=0o700)
        original = (SLURM / wrapper_name).read_text(encoding="utf-8")
        patched = original.replace(CANONICAL_ROOT, str(self.root))
        patched = patched.replace(CANONICAL_SCRATCH, str(self.scratch))
        self.wrapper = self.base / wrapper_name
        self.wrapper.write_text(patched, encoding="utf-8")
        self.wrapper.chmod(0o700)
        self.intent = (
            self.root / "outputs/prelaunch/post36_orchestration/intent.json"
        )
        self.stage_receipt = (
            self.root
            / "outputs/prelaunch/post36_orchestration/stages"
            / stage
            / "receipt.json"
        )
        self.intent.parent.mkdir(mode=0o700, parents=True)
        self.stage_receipt.parent.mkdir(mode=0o700, parents=True)
        self.intent.write_bytes(b"{}\n")
        self.stage_receipt.write_bytes(b"{}\n")

    def close(self) -> None:
        make_writable_and_remove(self.base)

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
            str(self.stage_receipt),
        ]

    def run(self, arguments: list[str], *, hostile: Path | None = None) -> subprocess.CompletedProcess[str]:
        environment = {
            "SLURM_JOB_ID": "990001",
            "SLURM_JOB_NAME": "ep12-test-archive-wrapper",
            "PATH": "/definitely/not/trusted",
            "LD_LIBRARY_PATH": "/definitely/not/trusted",
            "PYTHONPATH": str(hostile or (self.base / "absent-pythonpath")),
            "PYTHONHOME": str(self.base / "absent-pythonhome"),
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
            timeout=90,
            check=False,
        )


class ArchiveWrapperTests(unittest.TestCase):
    maxDiff = None

    def test_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", *[str(SLURM / name) for name in WRAPPERS]],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_frozen_positional_and_runtime_verifier_contracts(self) -> None:
        verifier_flags = [
            "--episode-root",
            "--orchestration-intent-path",
            "--expected-intent-file-sha256",
            "--expected-intent-sha256",
            "--stage-receipt-path",
            "--expected-stage",
            "--executed-wrapper",
            "--slurm-job-id",
            "--slurm-job-name",
            "--bootstrap-stage-result-path",
            "--staged-root",
            "--wrapper-arguments",
        ]
        for name, (count, stage) in WRAPPERS.items():
            with self.subTest(wrapper=name):
                text = (SLURM / name).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --export=NONE", text)
                self.assertIn(f"[[ $# -eq {count} ]]", text)
                positions = [text.index(item) for item in COMMON_ASSIGNMENTS]
                self.assertEqual(positions, sorted(positions))
                for flag in verifier_flags:
                    self.assertIn(flag, text)
                self.assertIn(f"--expected-stage {stage}", text)
                self.assertIn('--executed-wrapper "$0"', text)
                self.assertIn('--wrapper-arguments "$@"', text)
                self.assertIn(
                    "-P -B -m ep12_executor.post36_receipt_bound "
                    "verify-stage-runtime",
                    text,
                )

    def test_bootstrap_precedes_lmod_and_all_archive_imports(self) -> None:
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                text = (SLURM / name).read_text(encoding="utf-8")
                bootstrap = text.index(
                    '"${EP12_PYTHON}" -I -S "${EP12_BOOTSTRAP}" '
                    "verify-and-stage"
                )
                first_module_import = text.index("-m ep12_executor")
                self.assertLess(bootstrap, first_module_import)
                if name != "launch_post36_downstream_from_archive.sbatch":
                    self.assertLess(
                        bootstrap, text.index('source "${EP12_LMOD_INIT}"')
                    )
                    for module_name in (
                        "python/3.12.1",
                        "py-numpy/1.26.3_py312",
                        "py-scipy/1.12.0_py312",
                        "py-scikit-learn/1.5.1_py312",
                    ):
                        self.assertIn(module_name, text)
                    self.assertIn("EP12_MODULE_PYTHONPATH=", text)
                    self.assertIn("EP12_MODULE_LD_LIBRARY_PATH=", text)
                self.assertIn("export PATH=/usr/bin:/bin", text)
                self.assertIn(f"export LD_LIBRARY_PATH={FIXED_LD}", text)
                self.assertIn(f"EP12_PYTHON={FIXED_PYTHON}", text)
                self.assertIn("/usr/bin/readlink -f", text)
                self.assertNotIn("/usr/bin/python3", text)
                self.assertIn(BOOTSTRAP_SHA256, text)
                self.assertIn("__file__", text)
                self.assertIn("live-source import rejected", text)

    def test_downstream_coordinator_uses_authenticated_staged_executor(self) -> None:
        text = (
            SLURM / "launch_post36_downstream_from_archive.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn('--executor-root "${EP12_STAGED_EXECUTOR}"', text)
        self.assertNotIn("EP12_LIVE_EXECUTOR", text)

    def test_protected_probe_bytes_remain_pinned(self) -> None:
        for name in (
            "run_contract_invariance_probe_from_archive.sbatch",
            "compare_contract_invariance_from_archive.sbatch",
        ):
            text = (SLURM / name).read_text(encoding="utf-8")
            self.assertIn(PROBE_SHA256, text)
            self.assertIn(SPEC_SHA256, text)
            self.assertIn(LEGACY_WRAPPER_SHA256, text)

    def test_no_scheduler_poll_or_legacy_fallback_surface(self) -> None:
        forbidden = re.compile(
            r"\b(?:squeue|sacct|scontrol|scancel)\b|"
            r"full[-_]null|resource_authorization|"
            r"PYTHONPATH=.*EP12_ARCHIVE_ROOT"
        )
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                text = (SLURM / name).read_text(encoding="utf-8")
                self.assertIsNone(forbidden.search(text))

    def test_wrong_bootstrap_hash_fails_before_execution(self) -> None:
        fixture = WrapperFixture(
            "run_contract_invariance_probe_from_archive.sbatch", "post36_probe"
        )
        try:
            archive = fixture.base / ("source-" + "5" * 64)
            bootstrap = (
                archive
                / "outputs/executor/ep12_executor/source_archive_bootstrap.py"
            )
            bootstrap.parent.mkdir(mode=0o700, parents=True)
            sentinel = fixture.base / "bootstrap-executed"
            bootstrap.write_text(
                "from pathlib import Path\n"
                f"Path({str(sentinel)!r}).write_text('executed')\n",
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
            output = (
                fixture.root
                / "outputs/prelaunch/contract_transition_probe/post36_v1/probe.json"
            )
            result = fixture.run(fixture.common(identity) + [str(output)])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("archive bootstrap SHA-256 mismatch", result.stderr)
            self.assertFalse(sentinel.exists())
        finally:
            fixture.close()

    def test_wrong_archive_tree_digest_reaches_pinned_bootstrap_and_fails(self) -> None:
        fixture = WrapperFixture(
            "run_contract_invariance_probe_from_archive.sbatch", "post36_probe"
        )
        try:
            archive = fixture.base / "archive-publication" / ("source-" + "5" * 64)
            bootstrap = (
                archive
                / "outputs/executor/ep12_executor/source_archive_bootstrap.py"
            )
            bootstrap.parent.mkdir(mode=0o700, parents=True)
            shutil.copyfile(PINNED_BOOTSTRAP, bootstrap)
            self.assertEqual(sha(bootstrap.read_bytes()), BOOTSTRAP_SHA256)
            identity = {
                "archive": str(archive),
                "receipt_file": "1" * 64,
                "receipt": "2" * 64,
                "manifest_file": "3" * 64,
                "manifest": "4" * 64,
                "tree": "6" * 64,
            }
            output = (
                fixture.root
                / "outputs/prelaunch/contract_transition_probe/post36_v1/probe.json"
            )
            result = fixture.run(fixture.common(identity) + [str(output)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "archive root name is not expected content address",
                result.stderr,
            )
        finally:
            fixture.close()

    def test_wrong_canonical_path_and_argument_order_fail_closed(self) -> None:
        fixture = WrapperFixture(
            "run_contract_invariance_probe_from_archive.sbatch", "post36_probe"
        )
        try:
            identity = {
                "archive": str(fixture.base / ("source-" + "5" * 64)),
                "receipt_file": "1" * 64,
                "receipt": "2" * 64,
                "manifest_file": "3" * 64,
                "manifest": "4" * 64,
                "tree": "5" * 64,
            }
            output = (
                fixture.root
                / "outputs/prelaunch/contract_transition_probe/post36_v1/probe.json"
            )
            arguments = fixture.common(identity) + [str(output)]
            wrong_path = list(arguments)
            wrong_path[7] = str(fixture.root / "wrong-intent.json")
            result = fixture.run(wrong_path)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("orchestration intent path is not canonical", result.stderr)

            wrong_order = list(arguments)
            wrong_order[1], wrong_order[2] = wrong_order[2], wrong_order[1]
            result = fixture.run(wrong_order)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("archive root must be an absolute", result.stderr)
        finally:
            fixture.close()

    def test_hostile_live_source_cannot_override_authenticated_stage(self) -> None:
        fixture = WrapperFixture(
            "publish_scientific_transition_from_archive.sbatch", "publisher"
        )
        try:
            bootstrap = PINNED_BOOTSTRAP.read_bytes()
            runtime_verifier = b"""import json
if __name__ == "__main__":
    print(json.dumps({"verified": True}, sort_keys=True, separators=(",", ":")))
"""
            binding = b'ORIGIN = "authenticated-stage"\n'
            publisher = b"""from pathlib import Path
import argparse
import numpy
import scipy
import sklearn
from . import scientific_transition_binding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--executor-root", required=True)
    parser.add_argument("--synthetic-invariance", required=True)
    parser.add_argument("--replacement-terminal-identity", required=True)
    parser.add_argument("--post36-start-authorization", required=True)
    parser.add_argument("--obsolete-cancellation-receipt", required=True)
    args = parser.parse_args()
    marker = Path(args.episode_root) / "staged-import-ok.txt"
    versions = ",".join((numpy.__version__, scipy.__version__, sklearn.__version__))
    marker.write_text(str(Path(__file__).resolve()) + "\\n" + scientific_transition_binding.ORIGIN + "\\n" + versions + "\\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
            identity = create_reviewed_archive(
                fixture.base / "archive-publication",
                {
                    "SEARCH_POLICY.yaml": b"schema_version: test\n",
                    "outputs/executor/ep12_executor/__init__.py": b"",
                    "outputs/executor/ep12_executor/source_archive_bootstrap.py": bootstrap,
                    "outputs/executor/ep12_executor/post36_receipt_bound.py": runtime_verifier,
                    "outputs/executor/ep12_executor/scientific_transition_binding.py": binding,
                    "outputs/executor/ep12_executor/scientific_transition_publish.py": publisher,
                },
            )
            comparison = (
                fixture.root
                / "outputs/prelaunch/contract_transition_probe/"
                "comparison_v1/comparison.json"
            )
            terminal = (
                fixture.root
                / "outputs/prelaunch/replacement_attestation_launch/"
                "terminal_identity.json"
            )
            start_authorization = (
                fixture.root
                / "outputs/prelaunch/post36_orchestration/start_authorization.json"
            )
            cancellation = (
                fixture.root
                / "outputs/prelaunch/replacement_attestation_launch/"
                "obsolete_cancellation/receipt.json"
            )
            comparison.parent.mkdir(mode=0o700, parents=True)
            terminal.parent.mkdir(mode=0o700, parents=True)
            start_authorization.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            cancellation.parent.mkdir(mode=0o700, parents=True)
            comparison.write_bytes(b"{}\n")
            terminal.write_bytes(b"{}\n")

            hostile = fixture.base / "hostile"
            start_authorization.write_bytes(b"{}\n")
            cancellation.write_bytes(b"{}\n")
            hostile_package = hostile / "ep12_executor"
            hostile_package.mkdir(mode=0o700, parents=True)
            malicious_sentinel = fixture.base / "malicious-live-imported"
            hostile_code = (
                "from pathlib import Path\n"
                f"Path({str(malicious_sentinel)!r}).write_text('bad')\n"
            )
            (hostile_package / "__init__.py").write_text(
                hostile_code, encoding="utf-8"
            )
            for name in (
                "post36_receipt_bound.py",
                "scientific_transition_binding.py",
                "scientific_transition_publish.py",
            ):
                (hostile_package / name).write_text(hostile_code, encoding="utf-8")

            arguments = fixture.common(identity) + [
                str(comparison),
                str(terminal),
                str(start_authorization),
                str(cancellation),
            ]
            result = fixture.run(arguments, hostile=hostile)
            self.assertEqual(
                result.returncode,
                0,
                "stdout:\n" + result.stdout + "\nstderr:\n" + result.stderr,
            )
            self.assertFalse(malicious_sentinel.exists())
            marker = fixture.root / "staged-import-ok.txt"
            self.assertTrue(marker.is_file())
            marker_text = marker.read_text(encoding="utf-8")
            self.assertIn("source-" + identity["tree"], marker_text)
            self.assertIn("authenticated-stage", marker_text)
            self.assertIn("1.26.3,1.12.0,1.5.1", marker_text)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
