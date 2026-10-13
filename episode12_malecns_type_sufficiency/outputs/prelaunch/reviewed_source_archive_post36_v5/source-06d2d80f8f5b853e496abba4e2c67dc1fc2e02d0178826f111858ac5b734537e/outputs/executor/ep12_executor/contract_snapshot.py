"""Single-read, content-addressed snapshots for EP12 scientific contracts.

The observed 36-trial phase ran under an immutable v1 contract.  Post-36 code
must never parse one set of bytes and later record the digest of another.  This
module therefore reads a regular file once, parses those exact bytes, and can
fail closed if the path's content changes before an artifact is committed.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .yaml_subset import parse_yaml_subset_bytes


V1_CONTRACT_SHA256 = (
    "f7de6c70fac3f6690b909a47162c2c9183cc8747ce6fa7c2fc281c7e2363b006"
)
V1_QUALIFICATION_SPEC_SHA256 = (
    "e4a3e81b08b3c46b793823112758fbb811603e8ba4ea7bc9febfe2066ba5eac0"
)
V1_CONTRACT_SCHEMA = "ep12.scientific_contract.v1"
V1_CONTRACT_ID = "ep12_outgoing_tum_v1"
V1_QUALIFICATION_SPEC_SCHEMA = "ep12.synthetic_scientific_qualification.v1"


class ContractSnapshotError(ValueError):
    """A contract path is unsafe, changed, or has the wrong immutable identity."""


def _read_regular_file_no_follow(path: Path) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ContractSnapshotError(f"cannot open contract without following links: {absolute}") from exc
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode):
            raise ContractSnapshotError(f"contract is not a regular file: {absolute}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        content = b"".join(chunks)
        if len(content) != information.st_size:
            raise ContractSnapshotError(f"contract changed while it was read: {absolute}")
        return content
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class YamlContractSnapshot:
    """Exact bytes, parsed mapping, and digest captured in one operation."""

    path: Path
    content: bytes
    sha256: str
    document: Mapping[str, Any]

    def assert_unchanged(self) -> None:
        current = _read_regular_file_no_follow(self.path)
        if current != self.content:
            raise ContractSnapshotError(
                f"contract bytes changed after snapshot: {self.path}"
            )


def capture_yaml_contract(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> YamlContractSnapshot:
    """Capture a YAML contract from one read and optionally enforce its digest."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    content = _read_regular_file_no_follow(absolute)
    sha256 = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ContractSnapshotError(
            f"contract digest mismatch for {absolute}: expected {expected_sha256}, got {sha256}"
        )
    document = parse_yaml_subset_bytes(content)
    return YamlContractSnapshot(
        path=absolute,
        content=content,
        sha256=sha256,
        document=document,
    )


def assert_immutable_v1_contract(path: Path) -> YamlContractSnapshot:
    """Require the byte-exact scientific contract used by the observed trials."""

    snapshot = capture_yaml_contract(path, expected_sha256=V1_CONTRACT_SHA256)
    if (
        snapshot.document.get("schema_version") != V1_CONTRACT_SCHEMA
        or snapshot.document.get("contract_id") != V1_CONTRACT_ID
    ):
        raise ContractSnapshotError("immutable v1 scientific contract identity mismatch")
    return snapshot

def assert_immutable_v1_qualification_spec(path: Path) -> YamlContractSnapshot:
    """Require the byte-exact generated-data qualification spec paired with v1."""

    snapshot = capture_yaml_contract(
        path, expected_sha256=V1_QUALIFICATION_SPEC_SHA256
    )
    if (
        snapshot.document.get("schema_version")
        != V1_QUALIFICATION_SPEC_SCHEMA
        or snapshot.document.get("contract_ref") != "SCIENTIFIC_CONTRACT.yaml"
    ):
        raise ContractSnapshotError("immutable v1 qualification-spec identity mismatch")
    return snapshot
