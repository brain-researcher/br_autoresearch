#!/usr/bin/env python3
"""Authenticate the paper-cited official BVLC AlexNet asset without loading it."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Mapping, NamedTuple


ASTRA_ANALYSIS_ID = "source_qualification"
SCHEMA_VERSION = "ep21.bvlc_alexnet_paper_asset_audit.v1"
EXPECTED_ATTEMPT = 18
EXPECTED_QUALIFICATION_ATTEMPT = 17
EXPECTED_QUALIFICATION_SCHEMA = "ep21.source_qualification_report.v2"
EXPECTED_QUALIFICATION_SHA256 = (
    "98b9e5e30947f12484dc2df66d5ef06b666cb84f2e84cc01b8b0a1c609e07126"
)
EXPECTED_EPISODE_DIR = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode21_wang_neural_code_conversion_reproduction"
)
QUALIFICATION_REPORT_RELATIVE = Path(
    "outputs/source_qualification/attempt-017/qualification_report.json"
)
OUTPUT_RELATIVE = Path(
    "outputs/source_qualification/attempt-018/bvlc_alexnet_paper_asset_audit.json"
)
EXPECTED_CACHE_DIR = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode21_wang_neural_code_conversion_reproduction/public_model_assets"
)

BVLC_CAFFE_COMMIT = "9b891540183ddc834a02b2bd81b31afae71b2153"
BVLC_MODEL_TRAINING_CAFFE_COMMIT = "709dc15af4a06bebda027c1eb2b3f3e3375d5077"
README_URL = (
    "https://raw.githubusercontent.com/BVLC/caffe/"
    f"{BVLC_CAFFE_COMMIT}/models/bvlc_alexnet/readme.md"
)
DEPLOY_URL = (
    "https://raw.githubusercontent.com/BVLC/caffe/"
    f"{BVLC_CAFFE_COMMIT}/models/bvlc_alexnet/deploy.prototxt"
)
MODEL_URL = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
README_SHA256 = "a537febafb2bfc7d33d11e645a4eba6b873bf577a1a874214d3a5170a503e67e"
DEPLOY_SHA256 = "b564ba76416b4ac1e062c6621cbd80d9c19304ba152d9d791c75870f3182c4fa"
MODEL_SHA1 = "9116a64c0fbe4459d18f4bb6b56d647b63920377"
MODEL_SIZE_BYTES = 243_862_414
MODEL_FILENAME = "bvlc_alexnet.caffemodel"
MAX_METADATA_BYTES = 128 * 1024
MAX_QUALIFICATION_BYTES = 5 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class FileIdentity(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int
    mode: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _identity(info: os.stat_result) -> FileIdentity:
    return FileIdentity(
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        stat.S_IMODE(info.st_mode),
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_absolute_directory(path: Path, *, create_missing: bool) -> int:
    if not path.is_absolute() or path == Path("/"):
        raise ValueError(f"directory path must be a non-root absolute path: {path}")
    descriptor = os.open("/", _directory_flags())
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create_missing:
                    raise
                parent = os.fstat(descriptor)
                if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o022:
                    raise ValueError(
                        f"refusing to create below a non-private directory: {path}"
                    )
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            child_info = os.fstat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child)
                raise ValueError(f"path component is not a directory: {component}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class StableDirectory:
    """Hold a private directory open and detect path rebinding."""

    def __init__(self, path: Path, *, create: bool):
        if not path.is_absolute():
            raise ValueError(f"directory path must be absolute: {path}")
        self.path = path
        self.descriptor = _open_absolute_directory(path, create_missing=create)
        try:
            info = os.fstat(self.descriptor)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"path is not a real directory: {path}")
            if info.st_uid != os.getuid():
                raise ValueError(f"directory is not owned by the current user: {path}")
            os.fchmod(self.descriptor, 0o700)
            self.device = info.st_dev
            self.inode = info.st_ino
            self.ensure_bound()
        except BaseException:
            os.close(self.descriptor)
            raise

    def ensure_bound(self) -> None:
        info = os.fstat(self.descriptor)
        mode = stat.S_IMODE(info.st_mode)
        rebound = _open_absolute_directory(self.path, create_missing=False)
        try:
            path_info = os.fstat(rebound)
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or mode & 0o777 != 0o700
                or mode & (stat.S_ISUID | stat.S_ISVTX)
                or (info.st_dev, info.st_ino) != (self.device, self.inode)
                or (path_info.st_dev, path_info.st_ino) != (self.device, self.inode)
            ):
                raise ValueError(f"directory binding or permissions changed: {self.path}")
        finally:
            os.close(rebound)

    def close(self) -> None:
        os.close(self.descriptor)

    def __enter__(self) -> "StableDirectory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def _open_regular(path: Path) -> tuple[int, FileIdentity]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"path is not a single-link regular file: {path}")
        path_info = path.lstat()
        if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError(f"path identity changed while opening: {path}")
        return descriptor, _identity(info)
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_at(directory: StableDirectory, name: str) -> tuple[int, FileIdentity]:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("directory-relative filename is invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory.descriptor)
    try:
        info = os.fstat(descriptor)
        path_info = os.stat(
            name,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"cache entry is not a single-link regular file: {name}")
        if _identity(path_info) != _identity(info):
            raise ValueError(f"cache entry identity changed while opening: {name}")
        return descriptor, _identity(info)
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_beneath(
    base_directory: Path,
    relative_path: Path,
) -> tuple[int, int, FileIdentity]:
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
        or relative_path.name in {"", ".", ".."}
    ):
        raise ValueError("relative input path is invalid")
    directory = _open_absolute_directory(base_directory, create_missing=False)
    try:
        for component in relative_path.parts[:-1]:
            child = os.open(component, _directory_flags(), dir_fd=directory)
            child_info = os.fstat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child)
                raise ValueError(f"input ancestor is not a directory: {component}")
            os.close(directory)
            directory = child
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(relative_path.name, flags, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            entry = os.stat(
                relative_path.name,
                dir_fd=directory,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("input is not a single-link regular file")
            if _identity(info) != _identity(entry):
                raise ValueError("input identity changed while opening")
            return descriptor, directory, _identity(info)
        except BaseException:
            os.close(descriptor)
            raise
    except BaseException:
        os.close(directory)
        raise


def _require_path_identity(path: Path, identity: FileIdentity) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or _identity(info) != identity:
        raise ValueError(f"path identity changed: {path}")


def _require_name_identity(
    directory: StableDirectory,
    name: str,
    identity: FileIdentity,
) -> None:
    info = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or _identity(info) != identity:
        raise ValueError(f"directory entry identity changed: {name}")


def _digest_descriptor(descriptor: int, algorithm: str) -> tuple[str, int]:
    digest = hashlib.new(algorithm)
    total = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
        total += len(block)
    return digest.hexdigest(), total


def _write_all(
    descriptor: int,
    payload: bytes,
    *,
    write: Callable[[int, bytes | memoryview], int] = os.write,
) -> None:
    """Write every byte, including when the underlying write is short."""
    remaining = memoryview(payload)
    while remaining:
        written = write(descriptor, remaining)
        if written <= 0:
            raise OSError("write returned no progress")
        remaining = remaining[written:]


def verified_file(path: Path, algorithm: str, expected: str, size: int) -> FileIdentity:
    descriptor, before = _open_regular(path)
    try:
        observed, observed_size = _digest_descriptor(descriptor, algorithm)
        after = _identity(os.fstat(descriptor))
        if before != after:
            raise ValueError(f"file changed while hashing: {path}")
        _require_path_identity(path, after)
        if observed_size != size or observed != expected:
            raise ValueError(
                f"file identity mismatch for {path.name}: "
                f"size={observed_size}, {algorithm}={observed}"
            )
        return after
    finally:
        os.close(descriptor)


def read_stable_bytes(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[bytes, FileIdentity]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected SHA256 is not canonical")
    descriptor, before = _open_regular(path)
    try:
        if before.size <= 0 or before.size > maximum_bytes:
            raise ValueError(f"input has an invalid size: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, CHUNK_BYTES)
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise ValueError(f"input exceeded its size bound: {path}")
        after = _identity(os.fstat(descriptor))
        if before != after or total != before.size:
            raise ValueError(f"input changed while reading: {path}")
        _require_path_identity(path, after)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"input SHA256 mismatch: {observed}")
    return payload, after


def read_stable_relative_bytes(
    base_directory: Path,
    relative_path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> tuple[bytes, FileIdentity]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected SHA256 is not canonical")
    descriptor, parent, before = _open_regular_beneath(base_directory, relative_path)
    try:
        if before.size <= 0 or before.size > maximum_bytes:
            raise ValueError("relative input has an invalid size")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, CHUNK_BYTES)
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise ValueError("relative input exceeded its size bound")
        after = _identity(os.fstat(descriptor))
        entry = _identity(
            os.stat(
                relative_path.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
        )
        if before != after or entry != after or total != before.size:
            raise ValueError("relative input changed while reading")
    finally:
        os.close(descriptor)
        os.close(parent)

    rebound_descriptor, rebound_parent, rebound = _open_regular_beneath(
        base_directory,
        relative_path,
    )
    try:
        if rebound != after:
            raise ValueError("relative input path was rebound after reading")
    finally:
        os.close(rebound_descriptor)
        os.close(rebound_parent)
    payload = b"".join(chunks)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"relative input SHA256 mismatch: {observed}")
    return payload, after


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        return None


def _open_url(request: urllib.request.Request, timeout: int) -> Any:
    return urllib.request.build_opener(NoRedirectHandler()).open(request, timeout=timeout)


def _validate_url(url: str, *, metadata: bool) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("source URL must be credential-free and immutable")
    if metadata:
        if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
            raise ValueError("metadata must use pinned GitHub HTTPS content")
    elif (
        parsed.scheme != "http"
        or parsed.hostname != "dl.caffe.berkeleyvision.org"
        or parsed.path != f"/{MODEL_FILENAME}"
    ):
        raise ValueError("model URL differs from the official provider URL")


def fetch_pinned_text(
    url: str,
    expected_sha256: str,
    *,
    open_url: Callable[[urllib.request.Request, int], Any] = _open_url,
) -> str:
    _validate_url(url, metadata=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ep21-source-audit/1"})
    with open_url(request, 60) as response:
        status = response.status if hasattr(response, "status") else response.getcode()
        if status != 200:
            raise ValueError(f"unexpected metadata HTTP status: {status}")
        final_url = response.geturl()
        if final_url != url:
            raise ValueError("metadata request redirected or changed URL")
        payload = response.read(MAX_METADATA_BYTES + 1)
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError("metadata response exceeds the bounded size")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"metadata SHA256 mismatch: {observed}")
    return payload.decode("utf-8")


def parse_provider_readme(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("provider README lacks front matter")
    parts = text.split("---\n", 2)
    if len(parts) != 3 or parts[0] != "":
        raise ValueError("provider README front matter is incomplete")
    front_matter = parts[1]
    fields: dict[str, str] = {}
    for line in front_matter.splitlines():
        if ":" not in line:
            raise ValueError("malformed provider README front matter")
        key, value = line.split(":", 1)
        key = key.strip()
        if key in fields:
            raise ValueError(f"duplicate provider README field: {key}")
        fields[key] = value.strip()
    expected = {
        "name": "BAIR/BVLC AlexNet Model",
        "caffemodel": MODEL_FILENAME,
        "caffemodel_url": MODEL_URL,
        "license": "unrestricted",
        "sha1": MODEL_SHA1,
        "caffe_commit": BVLC_MODEL_TRAINING_CAFFE_COMMIT,
    }
    if fields != expected:
        raise ValueError(f"provider README fields differ from pins: {fields}")
    return fields


def _strip_prototxt_comments(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        quoted = False
        escaped = False
        retained: list[str] = []
        for character in line:
            if escaped:
                retained.append(character)
                escaped = False
                continue
            if character == "\\" and quoted:
                retained.append(character)
                escaped = True
                continue
            if character == '"':
                quoted = not quoted
                retained.append(character)
                continue
            if character == "#" and not quoted:
                break
            retained.append(character)
        if quoted:
            raise ValueError("deploy definition contains an unterminated quote")
        cleaned.append("".join(retained))
    return "\n".join(cleaned) + "\n"


def _layer_type(body: str) -> str:
    matches = re.findall(
        r'^\s*type:\s*(?:"([^"]+)"|([A-Z_]+))\s*$',
        body,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError("layer does not declare exactly one type")
    value = matches[0][0] or matches[0][1]
    legacy = {
        "CONVOLUTION": "Convolution",
        "INNER_PRODUCT": "InnerProduct",
        "INPUT": "Input",
    }
    return legacy.get(value, value)


def validate_qualification_facts(qualification: Mapping[str, Any]) -> dict[str, Any]:
    if qualification.get("schema_version") != EXPECTED_QUALIFICATION_SCHEMA:
        raise ValueError("qualification report schema changed")
    if qualification.get("astra_analysis_id") != ASTRA_ANALYSIS_ID:
        raise ValueError("qualification report analysis identity changed")
    if qualification.get("attempt") != EXPECTED_QUALIFICATION_ATTEMPT:
        raise ValueError("qualification report attempt changed")
    discrepancy = qualification.get("paper_code_discrepancy_register", {}).get(
        "alexnet_backend_and_input_size"
    )
    expected_discrepancy = {
        "paper": "official Caffe bvlc_alexnet with 224 by 224 inputs",
        "state": "recorded_unresolved_for_pipeline_conformance",
        "tagged_code": "bdpy PyTorch AlexNet port with direct 227 by 227 resize",
    }
    if discrepancy != expected_discrepancy:
        raise ValueError("qualification report AlexNet discrepancy changed")

    model_audit = qualification.get("source_pipeline_assets", {}).get(
        "public_model_payload_audit", {}
    )
    if (
        model_audit.get("schema_version") != "ep21.public_model_payload_audit.v2"
        or model_audit.get("attempt") != EXPECTED_QUALIFICATION_ATTEMPT
        or model_audit.get("checkpoint_deserialization_performed") is not False
        or model_audit.get("empirical_or_decoded_payload_accessed") is not False
        or model_audit.get("ood_or_ep04_payload_accessed") is not False
    ):
        raise ValueError("qualification report public-model boundary changed")
    alexnet = model_audit.get("evaluator_checkpoint_references", {}).get("alexnet", {})
    if (
        alexnet.get("state")
        != "lab_local_checkpoint_unmanifested_public_caffenet_not_substitutable"
        or alexnet.get("exact_public_bvlc_alexnet_named_member_verified") is not False
        or alexnet.get("caffenet_equivalence_claim_allowed") is not False
        or alexnet.get("lab_local_path")
        != "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt"
    ):
        raise ValueError("qualification report tagged AlexNet checkpoint state changed")
    caffenet = alexnet.get("public_caffenet_candidate_member", {})
    expected_caffenet = {
        "basename": "bvlc_reference_caffenet.pt",
        "path": "bvlc_reference_caffenet/bvlc_reference_caffenet.pt",
        "sha256": "24016c5822dfc778796f9410fc5d2d474bb451d86094747c8c5b1274f8bd9a00",
        "uncompressed_bytes": 243_866_815,
    }
    if any(caffenet.get(key) != value for key, value in expected_caffenet.items()):
        raise ValueError("qualification report public CaffeNet candidate changed")
    source_alexnet = model_audit.get("source_requirements", {}).get(
        "evaluator_branches", {}
    ).get("alexnet", {})
    if (
        source_alexnet.get("checkpoint_path") != alexnet.get("lab_local_path")
        or source_alexnet.get("checkpoint_provenance") != "lab_local_unchecksummed"
        or source_alexnet.get("image_size") != [227, 227]
        or source_alexnet.get("model_class") != "AlexNet"
        or source_alexnet.get("layer_map") != "alexnet"
    ):
        raise ValueError("qualification report tagged AlexNet requirements changed")
    return {
        "paper_description": discrepancy["paper"],
        "tagged_description": discrepancy["tagged_code"],
        "discrepancy_state": discrepancy["state"],
        "tagged_checkpoint": {
            "path": alexnet["lab_local_path"],
            "basename": "bvlc_alexnet.pt",
            "format": "PyTorch",
            "authenticated": False,
            "provenance": source_alexnet["checkpoint_provenance"],
        },
        "public_caffenet": {
            **expected_caffenet,
            "format": "PyTorch",
            "authenticated_as_caffenet_candidate": True,
            "authorized_as_alexnet_substitute": False,
        },
    }


def load_qualification_binding(
    episode_dir: Path,
    qualification_path: Path,
    *,
    expected_sha256: str = EXPECTED_QUALIFICATION_SHA256,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not episode_dir.is_absolute():
        raise ValueError("episode directory must be absolute")
    expected_path = episode_dir / QUALIFICATION_REPORT_RELATIVE
    if qualification_path.absolute() != expected_path:
        raise ValueError("qualification report path is not the canonical attempt-17 artifact")
    payload, identity = read_stable_relative_bytes(
        episode_dir,
        QUALIFICATION_REPORT_RELATIVE,
        expected_sha256=expected_sha256,
        maximum_bytes=MAX_QUALIFICATION_BYTES,
    )
    parsed = json.loads(payload.decode("utf-8", errors="strict"))
    if not isinstance(parsed, dict):
        raise ValueError("qualification report is not a JSON object")
    facts = validate_qualification_facts(parsed)
    binding = {
        "relative_path": str(QUALIFICATION_REPORT_RELATIVE),
        "sha256": expected_sha256,
        "size_bytes": identity.size,
        "schema_version": parsed["schema_version"],
        "analysis_id": parsed["astra_analysis_id"],
        "attempt": parsed["attempt"],
    }
    return parsed, facts, binding


def parse_deploy_prototxt(text: str) -> dict[str, Any]:
    text = _strip_prototxt_comments(text)
    network_names = re.findall(r'^name:\s*"([^"]+)"\s*$', text, flags=re.MULTILINE)
    if network_names != ["AlexNet"]:
        raise ValueError("deploy definition is not named AlexNet")
    layer_starts = list(re.finditer(r"^layers?\s*\{", text, flags=re.MULTILINE))
    layer_bodies: dict[str, str] = {}
    layer_types: dict[str, str] = {}
    for index, start in enumerate(layer_starts):
        stop = layer_starts[index + 1].start() if index + 1 < len(layer_starts) else len(text)
        body = text[start.start():stop]
        names = re.findall(r'^\s*name:\s*"([^"]+)"\s*$', body, flags=re.MULTILINE)
        if len(names) != 1:
            raise ValueError("deploy definition contains an unnamed layer")
        name = names[0]
        if name in layer_bodies:
            raise ValueError(f"deploy definition repeats layer name: {name}")
        layer_bodies[name] = body
        layer_types[name] = _layer_type(body)

    legacy_dimensions = [
        int(value)
        for value in re.findall(r"^input_dim:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    ]
    modern_dimensions: list[int] = []
    if "data" in layer_bodies and layer_types["data"] == "Input":
        input_matches = re.findall(
            r"input_param\s*\{\s*shape\s*:?\s*\{(?P<body>.*?)\}\s*\}",
            layer_bodies["data"],
            flags=re.DOTALL,
        )
        if len(input_matches) != 1:
            raise ValueError("Input layer does not contain exactly one input shape")
        modern_dimensions = [
            int(value) for value in re.findall(r"dim:\s*(\d+)", input_matches[0])
        ]
    if bool(legacy_dimensions) == bool(modern_dimensions):
        raise ValueError("deploy definition must use exactly one input-shape syntax")
    dimensions = legacy_dimensions or modern_dimensions
    if dimensions != [10, 3, 227, 227]:
        raise ValueError(f"unexpected official AlexNet input shape: {dimensions}")

    required = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"]
    if any(name not in layer_bodies for name in required):
        raise ValueError("deploy definition lacks one or more required AlexNet layers")
    expected_types = {
        **{name: "Convolution" for name in required[:5]},
        **{name: "InnerProduct" for name in required[5:]},
    }
    if any(layer_types[name] != expected for name, expected in expected_types.items()):
        raise ValueError("deploy definition has an unexpected required-layer type")
    group_layers = []
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
        groups = re.findall(
            r"^\s*group:\s*(\d+)\s*$",
            layer_bodies[name],
            flags=re.MULTILINE,
        )
        expected_groups = ["2"] if name in {"conv2", "conv4", "conv5"} else []
        if groups != expected_groups:
            raise ValueError(f"unexpected group declaration for {name}: {groups}")
        if groups:
            group_layers.append(name)
    if group_layers != ["conv2", "conv4", "conv5"]:
        raise ValueError(f"unexpected grouped-convolution layout: {group_layers}")
    return {
        "network_name": "AlexNet",
        "input_shape": dimensions,
        "feature_layers": required,
        "grouped_convolution_layers": group_layers,
    }


@contextlib.contextmanager
def cache_lock(directory: StableDirectory) -> Iterator[None]:
    lock_name = ".official-bvlc-alexnet-audit.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_name, flags, 0o600, dir_fd=directory.descriptor)
    try:
        info = os.fstat(descriptor)
        path_info = os.stat(
            lock_name,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise ValueError("cache lock is not a single-link regular file")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        directory.ensure_bound()
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _entry_exists(directory: StableDirectory, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _safe_unlink_owned_entry(
    directory: StableDirectory,
    name: str,
    device: int,
    inode: int,
) -> None:
    try:
        info = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (info.st_dev, info.st_ino) == (device, inode):
        os.unlink(name, dir_fd=directory.descriptor)


def _verified_cache_descriptor(
    directory: StableDirectory,
    name: str,
    expected_sha1: str,
    expected_size: int,
) -> tuple[int, FileIdentity]:
    descriptor, before = _open_regular_at(directory, name)
    try:
        observed, observed_size = _digest_descriptor(descriptor, "sha1")
        after = _identity(os.fstat(descriptor))
        if before != after:
            raise ValueError(f"cache entry changed while hashing: {name}")
        _require_name_identity(directory, name, after)
        if observed_size != expected_size or observed != expected_sha1:
            raise ValueError(
                f"cache entry identity mismatch for {name}: "
                f"size={observed_size}, sha1={observed}"
            )
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
        normalized = _identity(os.fstat(descriptor))
        _require_name_identity(directory, name, normalized)
        observed_again, size_again = _digest_descriptor(descriptor, "sha1")
        final = _identity(os.fstat(descriptor))
        if (
            normalized != final
            or observed_again != expected_sha1
            or size_again != expected_size
        ):
            raise ValueError(f"cache entry changed while normalizing: {name}")
        _require_name_identity(directory, name, final)
        return descriptor, final
    except BaseException:
        os.close(descriptor)
        raise


def download_or_reuse_model(
    cache_dir: Path,
    *,
    expected_size: int = MODEL_SIZE_BYTES,
    expected_sha1: str = MODEL_SHA1,
    open_url: Callable[[urllib.request.Request, int], Any] = _open_url,
) -> tuple[dict[str, Any], Path]:
    _validate_url(MODEL_URL, metadata=False)
    destination = cache_dir / MODEL_FILENAME
    with StableDirectory(cache_dir, create=True) as directory:
        with cache_lock(directory):
            if _entry_exists(directory, MODEL_FILENAME):
                descriptor, identity = _verified_cache_descriptor(
                    directory,
                    MODEL_FILENAME,
                    expected_sha1,
                    expected_size,
                )
                os.close(descriptor)
                directory.ensure_bound()
                return (
                    {
                        "cache_filename": MODEL_FILENAME,
                        "cache_reused_after_identity_check": True,
                        "payload_bytes_downloaded_this_attempt": 0,
                        "http_content_length_observed_this_attempt": None,
                        "observed_size_bytes": identity.size,
                        "observed_sha1": expected_sha1,
                        "cache_mode": f"{identity.mode:04o}",
                        "cache_nlink": identity.nlink,
                    },
                    destination,
                )

            temporary = f".{MODEL_FILENAME}.part.{secrets.token_hex(12)}"
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory.descriptor)
            downloaded = 0
            digest = hashlib.sha1()
            linked = False
            try:
                request = urllib.request.Request(
                    MODEL_URL,
                    headers={"User-Agent": "ep21-source-audit/1"},
                )
                with open_url(request, 300) as response:
                    status = (
                        response.status
                        if hasattr(response, "status")
                        else response.getcode()
                    )
                    if status != 200:
                        raise ValueError(f"unexpected model HTTP status: {status}")
                    if response.geturl() != MODEL_URL:
                        raise ValueError("model request redirected or changed URL")
                    lengths = response.headers.get_all("Content-Length") or []
                    if lengths != [str(expected_size)]:
                        raise ValueError(f"unexpected model Content-Length: {lengths}")
                    while True:
                        block = response.read(CHUNK_BYTES)
                        if not block:
                            break
                        if downloaded + len(block) > expected_size:
                            raise ValueError("model download exceeded its exact size pin")
                        _write_all(descriptor, block)
                        digest.update(block)
                        downloaded += len(block)
                if downloaded != expected_size or digest.hexdigest() != expected_sha1:
                    raise ValueError(
                        "downloaded model identity mismatch: "
                        f"size={downloaded}, sha1={digest.hexdigest()}"
                    )
                os.fchmod(descriptor, 0o440)
                os.fsync(descriptor)
                written_before = _identity(os.fstat(descriptor))
                _require_name_identity(directory, temporary, written_before)
                stored_sha1, stored_size = _digest_descriptor(descriptor, "sha1")
                written_identity = _identity(os.fstat(descriptor))
                if (
                    written_before != written_identity
                    or stored_size != expected_size
                    or stored_sha1 != expected_sha1
                ):
                    raise ValueError(
                        "stored temporary model identity mismatch: "
                        f"size={stored_size}, sha1={stored_sha1}"
                    )
                _require_name_identity(directory, temporary, written_identity)
                os.link(
                    temporary,
                    MODEL_FILENAME,
                    src_dir_fd=directory.descriptor,
                    dst_dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                linked = True
                linked_identity = _identity(os.fstat(descriptor))
                _require_name_identity(directory, temporary, linked_identity)
                _require_name_identity(directory, MODEL_FILENAME, linked_identity)
                if linked_identity.nlink != 2:
                    raise ValueError("published cache entry has unexpected link count")
                os.unlink(temporary, dir_fd=directory.descriptor)
                final = _identity(os.fstat(descriptor))
                _require_name_identity(directory, MODEL_FILENAME, final)
                if final.nlink != 1 or final.mode != 0o440:
                    raise ValueError("published cache entry has unsafe final metadata")
                final_sha1, final_size = _digest_descriptor(descriptor, "sha1")
                after_final_hash = _identity(os.fstat(descriptor))
                if (
                    after_final_hash != final
                    or final_size != expected_size
                    or final_sha1 != expected_sha1
                ):
                    raise ValueError(
                        "published cache bytes changed: "
                        f"size={final_size}, sha1={final_sha1}"
                    )
                _require_name_identity(directory, MODEL_FILENAME, after_final_hash)
                final = after_final_hash
                os.fsync(directory.descriptor)
                directory.ensure_bound()
            except BaseException:
                current = os.fstat(descriptor)
                if linked:
                    _safe_unlink_owned_entry(
                        directory,
                        MODEL_FILENAME,
                        current.st_dev,
                        current.st_ino,
                    )
                _safe_unlink_owned_entry(
                    directory,
                    temporary,
                    current.st_dev,
                    current.st_ino,
                )
                os.close(descriptor)
                raise
            else:
                os.close(descriptor)
            return (
                {
                    "cache_filename": MODEL_FILENAME,
                    "cache_reused_after_identity_check": False,
                    "payload_bytes_downloaded_this_attempt": downloaded,
                    "http_content_length_observed_this_attempt": expected_size,
                    "observed_size_bytes": final.size,
                    "observed_sha1": expected_sha1,
                    "cache_mode": f"{final.mode:04o}",
                    "cache_nlink": final.nlink,
                },
                destination,
            )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if Path(path.name).name != path.name or path.name in {"", ".", ".."}:
        raise ValueError("output filename is invalid")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with StableDirectory(path.parent, create=True) as directory:
        if _entry_exists(directory, path.name):
            raise FileExistsError(f"refusing to replace output: {path}")
        temporary = f".{path.name}.part.{secrets.token_hex(12)}"
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory.descriptor)
        linked = False
        try:
            _write_all(descriptor, encoded)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            written = _identity(os.fstat(descriptor))
            _require_name_identity(directory, temporary, written)
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
                follow_symlinks=False,
            )
            linked = True
            linked_identity = _identity(os.fstat(descriptor))
            _require_name_identity(directory, temporary, linked_identity)
            _require_name_identity(directory, path.name, linked_identity)
            if linked_identity.nlink != 2:
                raise ValueError("published report has unexpected link count")
            os.unlink(temporary, dir_fd=directory.descriptor)
            final = _identity(os.fstat(descriptor))
            _require_name_identity(directory, path.name, final)
            if final.nlink != 1 or final.mode != 0o600 or final.size != len(encoded):
                raise ValueError("published report has unsafe final metadata")
            observed, observed_size = _digest_descriptor(descriptor, "sha256")
            after_hash = _identity(os.fstat(descriptor))
            if (
                after_hash != final
                or observed_size != len(encoded)
                or observed != hashlib.sha256(encoded).hexdigest()
            ):
                raise ValueError("published report bytes changed")
            _require_name_identity(directory, path.name, after_hash)
            os.fsync(directory.descriptor)
            directory.ensure_bound()
        except BaseException:
            current = os.fstat(descriptor)
            if linked:
                _safe_unlink_owned_entry(
                    directory,
                    path.name,
                    current.st_dev,
                    current.st_ino,
                )
            _safe_unlink_owned_entry(
                directory,
                temporary,
                current.st_dev,
                current.st_ino,
            )
            os.close(descriptor)
            raise
        else:
            os.close(descriptor)


def build_report(
    *,
    started_at: str,
    completed_at: str,
    process_id: int,
    qualification_binding: Mapping[str, Any],
    qualification_facts: Mapping[str, Any],
    provider: Mapping[str, str],
    deploy: Mapping[str, Any],
    cache: Mapping[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    asset_distinctions = {
        "official_paper_cited_bvlc_alexnet": {
            "filename": MODEL_FILENAME,
            "format": "Caffe protobuf",
            "provider_published_sha1": MODEL_SHA1,
            "authenticated_as_official_provider_asset": True,
        },
        "tagged_evaluator_bvlc_alexnet": qualification_facts["tagged_checkpoint"],
        "public_bvlc_reference_caffenet": qualification_facts["public_caffenet"],
    }
    scientific_boundary = {
        "paper_cited_official_bvlc_caffe_asset_authenticated": True,
        "paper_description_from_bound_qualification": qualification_facts[
            "paper_description"
        ],
        "tagged_description_from_bound_qualification": qualification_facts[
            "tagged_description"
        ],
        "paper_reports_resize_pixels": [224, 224],
        "official_deploy_input_shape": deploy["input_shape"],
        "tagged_evaluator_reports_direct_resize_pixels": [227, 227],
        "tagged_pytorch_checkpoint_authenticated_by_this_audit": False,
        "official_caffemodel_equivalent_to_tagged_bvlc_alexnet_pt_verified": False,
        "official_caffemodel_satisfies_tagged_checkpoint_requirement": False,
        "caffemodel_to_pytorch_conversion_performed": False,
        "public_caffenet_substitution_authorized": False,
        "paper_caffe_vs_tagged_pytorch_backend_decision_resolved": False,
        "paper_text_224_vs_official_deploy_227_decision_resolved": False,
        "paper_faithful_pipeline_specification_complete": False,
        "paper_faithful_pipeline_runtime_ready": False,
        "remaining_actions": {
            "paper_route": (
                "Resolve paper-described 224 versus official-deploy 227; bind a native "
                "Caffe runtime or documented deterministic conversion; strict-load the "
                "official model; and lock mean, color, crop, layer, and activation conventions."
            ),
            "tagged_route": (
                "Authenticate the exact bvlc_alexnet.pt and strict-load it through the "
                "tagged bdpy implementation; this official Caffe asset cannot clear that "
                "checkpoint requirement."
            ),
            "fallback": "Lock Findings C and E not_evaluable.",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "astra_analysis_id": ASTRA_ANALYSIS_ID,
        "attempt": EXPECTED_ATTEMPT,
        "started_at": started_at,
        "completed_at": completed_at,
        "process_id": process_id,
        "outcome": "completed_official_bvlc_alexnet_provider_asset_authenticated",
        "scope": "paper_cited_official_public_bvlc_alexnet_provider_asset_only",
        "upstream_qualification_binding": dict(qualification_binding),
        "provider_metadata": {
            "repository": "https://github.com/BVLC/caffe",
            "repository_commit": BVLC_CAFFE_COMMIT,
            "readme_url": README_URL,
            "readme_sha256": README_SHA256,
            "deploy_url": DEPLOY_URL,
            "deploy_sha256": DEPLOY_SHA256,
            "declared_fields": dict(provider),
            "license_semantics": (
                "provider-declared string only; no DUA, Sherlock risk-classification, "
                "or legal determination is inferred"
            ),
        },
        "official_deploy_definition": dict(deploy),
        "model_asset": {
            "provider_url": MODEL_URL,
            "provider_transport": (
                "http_bytes_with_integrity_anchored_by_pinned_https_provider_metadata"
            ),
            "provider_published_sha1": MODEL_SHA1,
            "prepared_expected_size_bytes": MODEL_SIZE_BYTES,
            "prepared_size_basis": (
                "prior HTTP Content-Length observation; re-observed and byte-counted "
                "when a download occurs"
            ),
            "provider_metadata_declares_size_bytes": False,
            "cache_filename": cache_path.name,
            **dict(cache),
        },
        "asset_distinctions": asset_distinctions,
        "scientific_boundary": scientific_boundary,
        "access_boundary": {
            "qualification_input": str(QUALIFICATION_REPORT_RELATIVE),
            "public_metadata_hosts": ["raw.githubusercontent.com"],
            "public_model_host": "dl.caffe.berkeleyvision.org",
            "scratch_cache_relative_purpose": "public_model_assets_only",
            "attestation_basis": (
                "exact frozen code paths and identity-validated inputs/cache; no claim "
                "about unrelated external filesystem state"
            ),
        },
        "checkpoint_deserialization_performed": False,
        "tensor_or_device_context_created": False,
        "model_inference_performed": False,
        "empirical_or_ood_payload_accessed": False,
        "ep04_or_legacy_scratch_accessed": False,
        "feature_or_scoring_handoff_accessed": False,
        "protocol_lock_emitted": False,
        "empirical_readiness": "blocked",
        "next_analysis": "pipeline_conformance_after_scientist_choice_and_structural_handoff",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.attempt != EXPECTED_ATTEMPT:
        raise ValueError(f"this frozen supplemental audit requires attempt {EXPECTED_ATTEMPT}")

    expected_script = EXPECTED_EPISODE_DIR / OUTPUT_RELATIVE.parents[1] / "tools" / Path(__file__).name
    script_path = Path(__file__).resolve(strict=True)
    episode_descriptor = _open_absolute_directory(
        EXPECTED_EPISODE_DIR,
        create_missing=False,
    )
    os.close(episode_descriptor)
    canonical_episode = EXPECTED_EPISODE_DIR
    if script_path != expected_script:
        raise ValueError("runner is not executing from the frozen EP21 episode path")
    qualification_path = canonical_episode / QUALIFICATION_REPORT_RELATIVE
    output_path = canonical_episode / OUTPUT_RELATIVE

    started_at = utc_now()
    process_id = os.getpid()
    print(
        json.dumps(
            {
                "event": "ep21_bvlc_alexnet_asset_audit_started",
                "attempt": args.attempt,
                "pid": process_id,
                "started_at": started_at,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _, qualification_facts, qualification_binding = load_qualification_binding(
        canonical_episode,
        qualification_path,
    )
    readme_text = fetch_pinned_text(README_URL, README_SHA256)
    deploy_text = fetch_pinned_text(DEPLOY_URL, DEPLOY_SHA256)
    provider = parse_provider_readme(readme_text)
    deploy = parse_deploy_prototxt(deploy_text)
    cache, cache_path = download_or_reuse_model(EXPECTED_CACHE_DIR)
    completed_at = utc_now()
    report = build_report(
        started_at=started_at,
        completed_at=completed_at,
        process_id=process_id,
        qualification_binding=qualification_binding,
        qualification_facts=qualification_facts,
        provider=provider,
        deploy=deploy,
        cache=cache,
        cache_path=cache_path,
    )
    atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "event": "ep21_bvlc_alexnet_asset_audit_completed",
                "attempt": args.attempt,
                "output": str(OUTPUT_RELATIVE),
                "completed_at": completed_at,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
