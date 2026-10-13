"""Prepare the two pinned public generator members from a supplied local ZIP.

No download, CLI, model deserialization or empirical admission is provided.
The caller supplies an admitted compute allocation, an outer process deadline
and a budget callback. Existing bounded archive hashing/inspection helpers have
no internal callback: checks surround those stages; extraction checks each
4-MiB chunk. Existing directories refuse reuse, and any failed extraction keeps
its newly created directory and partial files for the caller's failure record.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

QUALIFICATION_TOOLS = Path(__file__).resolve().parents[1] / "source_qualification" / "tools"
if str(QUALIFICATION_TOOLS) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_TOOLS))
import audit_public_model_payloads as archive_tools


ARCHIVE_BYTES = 832_692_892
ARCHIVE_MD5 = "8be66997045859a49b1d2bcef6640490"
ARCHIVE_NAME = "bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip"
MEMBER_ROOT = "bvlc_reference_caffenet_generator_ILSVRC2012_Training"
REQUIRED_MEMBERS = {
    "checkpoint": MEMBER_ROOT + "/generator_relu7.pt",
    "latent_upper_bound": MEMBER_ROOT + "/relu7_act_range_3x.txt",
}
CHUNK_BYTES = archive_tools.CHUNK_BYTES


def extract_generator_assets(archive_path, extraction_dir, *, budget_check):
    """Authenticate one fixed archive and exclusively extract its two members.

    ``extraction_dir`` must not exist; its parent must already exist. The two
    source members are published by basename inside it, with their complete ZIP
    identities retained in the return record. No success record is returned
    until both extracted bytes match their authenticated inventory entries.
    Supplied paths and public-only execution authorization belong to the caller.
    """
    if not callable(budget_check):
        raise TypeError("explicit budget_check callback required")
    budget_check()
    archive_path = Path(os.path.abspath(os.fspath(archive_path)))
    extraction_dir = Path(os.path.abspath(os.fspath(extraction_dir)))
    if os.path.lexists(extraction_dir):
        raise FileExistsError("generator extraction directory already exists")

    identity = archive_tools.regular_file_identity(archive_path)
    if identity[2] != ARCHIVE_BYTES:
        raise ValueError("generator archive size differs from fixed pin")
    budget_check()
    digest, size = archive_tools.file_digest(archive_path, "md5")
    if size != ARCHIVE_BYTES or digest != ARCHIVE_MD5:
        raise ValueError("generator archive MD5 or size differs from fixed pin")
    if archive_tools.regular_file_identity(archive_path) != identity:
        raise ValueError("generator archive changed during authentication")
    budget_check()
    inventory = archive_tools.inspect_zip(archive_path, expected_identity=identity)
    budget_check()
    if inventory["unsafe_member_names"] or inventory["non_regular_member_names"]:
        raise ValueError("generator archive contains unsafe or non-regular members")
    selected = {}
    for role, member_path in REQUIRED_MEMBERS.items():
        matches = archive_tools.matching_path(inventory, member_path)
        if len(matches) != 1 or matches[0]["member_type"] != "regular_file":
            raise ValueError("exact unique regular generator member required: " + role)
        selected[role] = matches[0]

    # Keep the same validated archive open throughout copying. The original
    # helper rejects symlinks, non-regular files and multiple hard links.
    descriptor, copying_identity = archive_tools._open_regular(archive_path)
    with os.fdopen(descriptor, "rb", closefd=True) as raw:
        if copying_identity != identity:
            raise ValueError("generator archive changed before extraction")
        budget_check()
        extraction_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        partials = {}
        extracted = {}
        with zipfile.ZipFile(raw) as archive:
            for role, metadata in selected.items():
                budget_check()
                member_path = metadata["path"]
                final_path = extraction_dir / PurePosixPath(member_path).name
                partial_path = final_path.with_name(final_path.name + ".partial")
                info = archive.getinfo(member_path)
                if info.file_size != metadata["uncompressed_bytes"]:
                    raise ValueError("generator member size changed before extraction")
                copied = 0
                member_digest = hashlib.sha256()
                # Failure leaves this exclusive partial, never a falsely
                # published final file or a silently retried extraction.
                with archive.open(info, "r") as source, partial_path.open("xb") as destination:
                    os.fchmod(destination.fileno(), 0o600)
                    while True:
                        budget_check()
                        block = source.read(CHUNK_BYTES)
                        if not block:
                            break
                        copied += len(block)
                        if copied > metadata["uncompressed_bytes"]:
                            raise ValueError("generator member exceeded authenticated size")
                        destination.write(block)
                        member_digest.update(block)
                        budget_check()
                    destination.flush()
                    os.fsync(destination.fileno())
                    if (copied != metadata["uncompressed_bytes"]
                            or member_digest.hexdigest() != metadata["sha256"]):
                        raise ValueError("extracted generator member failed integrity check")
                    os.fchmod(destination.fileno(), 0o440)
                partials[role] = (partial_path, final_path)
                extracted[role] = {
                    **metadata,
                    "resolved_path": str(final_path.resolve(strict=False)),
                    "extracted_bytes": copied,
                }
        if archive_tools._identity(os.fstat(raw.fileno())) != identity:
            raise ValueError("generator archive changed while extracting")
        if archive_tools.regular_file_identity(archive_path) != identity:
            raise ValueError("generator archive path changed while extracting")
        budget_check()
        for partial_path, final_path in partials.values():
            budget_check()
            # link() refuses an existing destination; rename() would overwrite.
            # An interrupted publication retains its partial/complete files and
            # the directory will refuse reuse on a subsequent call.
            os.link(partial_path, final_path)
            partial_path.unlink()
        budget_check()

    return {
        "archive_path": str(archive_path.resolve(strict=True)),
        "archive_name": ARCHIVE_NAME,
        "archive_bytes": size,
        "archive_md5": digest,
        "archive_entry_count": inventory["entry_count"],
        "archive_file_count": inventory["file_count"],
        "archive_uncompressed_bytes": inventory["total_uncompressed_bytes"],
        "extraction_dir": str(extraction_dir.resolve(strict=True)),
        "members": extracted,
        "required_members_complete": True,
        "model_deserialized": False,
        "latent_values_validated": False,
        "runtime_qualified": False,
        "empirical_access_authorized": False,
    }
