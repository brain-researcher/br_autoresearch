#!/usr/bin/env python3
"""Authenticate the two public model archives relevant to EP21 identification.

This runner is intentionally limited to the pinned Wang V1.0.0 source archive
and two public, non-empirical Figshare model archives.  It never deserializes a
checkpoint and never reads a stimulus, neural, decoded-feature, OOD, or EP04
payload.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import verified_source_archive as source_archive_verifier


SOURCE_ARCHIVE_MD5 = "bb80dcc1a038737d6d8fc2c95b3e5586"
SOURCE_ARCHIVE_SIZE_BYTES = 23_110_908
SOURCE_ROOT = "KamitaniLab-InterSiteNeuralCodeConversion-edbff02/"
IDENTIFICATION_SCRIPT_SUFFIX = (
    "evaluation/identification_accuracy/recon_image_eval_dnn.py"
)
RECONSTRUCTION_MANIFEST_SUFFIX = "reconstruction/data/files.json"
VGG_CONFIG_SUFFIX = "reconstruction/config/encoder/vgg19.yaml"
VGG_CHECKPOINT_MEMBER = (
    "VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt"
)
CAFFENET_CHECKPOINT_MEMBER = (
    "bvlc_reference_caffenet/bvlc_reference_caffenet.pt"
)
VGG_LAYERS = [
    "conv1_1",
    "conv1_2",
    "conv2_1",
    "conv2_2",
    "conv3_1",
    "conv3_2",
    "conv3_3",
    "conv3_4",
    "conv4_1",
    "conv4_2",
    "conv4_3",
    "conv4_4",
    "conv5_1",
    "conv5_2",
    "conv5_3",
    "conv5_4",
    "fc6",
    "fc7",
    "fc8",
]
ASSETS = {
    "vgg19": {
        "article_id": 21564384,
        "figshare_file_id": 38225868,
        "filename": "VGG_ILSVRC_19_layers.zip",
        # Use the dedicated downloader host.  The human-facing host can return
        # an AWS-WAF 202 challenge with a zero-byte body to non-browser clients.
        "url": "https://ndownloader.figshare.com/files/38225868",
        "size_bytes": 533_710_127,
        "md5": "49f4d6dbf1024f33149352dfd309dc4d",
        "tagged_checkpoint_basename": "VGG_ILSVRC_19_layers.pt",
        "expected_checkpoint_member_path": VGG_CHECKPOINT_MEMBER,
    },
    "caffenet": {
        "article_id": 21564384,
        "figshare_file_id": 52398902,
        "filename": "bvlc_reference_caffenet.zip",
        "url": "https://ndownloader.figshare.com/files/52398902",
        "size_bytes": 226_699_018,
        "md5": "2e4fe2126c7ed13554e33d716a72c65f",
        "tagged_checkpoint_basename": "bvlc_alexnet.pt",
        "observed_candidate_member_path": CAFFENET_CHECKPOINT_MEMBER,
    },
}
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2_000_000_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 1_500_000_000
CHUNK_BYTES = 4 * 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


FileIdentity = tuple[int, int, int, int]


def _identity(info: os.stat_result) -> FileIdentity:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _open_regular(path: Path) -> tuple[int, FileIdentity]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open regular file: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"path is not a single-link regular file: {path}")
        return descriptor, _identity(info)
    except Exception:
        os.close(descriptor)
        raise


def regular_file_identity(path: Path) -> FileIdentity:
    descriptor, identity = _open_regular(path)
    os.close(descriptor)
    return identity


def file_digest(path: Path, algorithm: str) -> tuple[str, int]:
    descriptor, initial_identity = _open_regular(path)
    digest = hashlib.new(algorithm)
    byte_count = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            byte_count += len(chunk)
        final_identity = _identity(os.fstat(handle.fileno()))
    if final_identity != initial_identity or byte_count != initial_identity[2]:
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest(), byte_count


def stream_digest(handle: BinaryIO, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
    )


def inspect_zip(
    path: Path, expected_identity: FileIdentity | None = None
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    unsafe_names: list[str] = []
    non_regular_names: list[str] = []
    seen_names: set[str] = set()
    total_uncompressed = 0
    entry_count = 0
    descriptor, initial_identity = _open_regular(path)
    if expected_identity is not None and initial_identity != expected_identity:
        os.close(descriptor)
        raise ValueError(f"archive identity changed before inspection: {path}")
    with os.fdopen(descriptor, "rb", closefd=True) as raw_handle:
        with zipfile.ZipFile(raw_handle) as archive:
            for info in archive.infolist():
                entry_count += 1
                if info.filename in seen_names:
                    raise ValueError(f"duplicate archive member: {info.filename}")
                seen_names.add(info.filename)
                if not safe_member_name(info.filename):
                    unsafe_names.append(info.filename)
                    continue
                unix_mode = info.external_attr >> 16
                if info.is_dir():
                    if not stat.S_ISDIR(unix_mode):
                        non_regular_names.append(info.filename)
                    continue
                if not stat.S_ISREG(unix_mode):
                    non_regular_names.append(info.filename)
                    continue
                if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        f"archive member exceeds size bound: {info.filename}"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("archive exceeds total uncompressed-size bound")
                with archive.open(info, "r") as handle:
                    sha256 = stream_digest(handle)
                members.append(
                    {
                        "path": info.filename,
                        "basename": PurePosixPath(info.filename).name,
                        "member_type": "regular_file",
                        "unix_mode": f"{unix_mode:o}",
                        "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": sha256,
                    }
                )
        final_identity = _identity(os.fstat(raw_handle.fileno()))
    if final_identity != initial_identity:
        raise ValueError(f"archive changed while inspecting: {path}")
    return {
        "entry_count": entry_count,
        "file_count": len(members),
        "total_uncompressed_bytes": total_uncompressed,
        "unsafe_member_names": unsafe_names,
        "non_regular_member_names": non_regular_names,
        "members": sorted(members, key=lambda item: item["path"]),
    }


def matching_path(inventory: dict[str, Any], member_path: str) -> list[dict[str, Any]]:
    return [item for item in inventory["members"] if item["path"] == member_path]


def matching_basename(inventory: dict[str, Any], basename: str) -> list[dict[str, Any]]:
    return [item for item in inventory["members"] if item["basename"] == basename]


def _validate_cache_directory(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    info = os.lstat(cache_dir)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("cache directory must be a real directory")


@contextlib.contextmanager
def cache_lock(cache_dir: Path) -> Iterator[None]:
    _validate_cache_directory(cache_dir)
    lock_path = cache_dir / ".public-model-audit.lock"
    flags = (
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("cache lock is not a single-link regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _sanitized_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _validate_download_url(value: str, *, initial: bool) -> None:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("model download URL must be credential-free HTTPS")
    hostname = parsed.hostname.lower()
    if initial and hostname != "ndownloader.figshare.com":
        raise ValueError("model download must start at the dedicated Figshare host")
    if not initial and not (
        hostname == "ndownloader.figshare.com" or hostname.endswith(".amazonaws.com")
    ):
        raise ValueError(f"unexpected model-download redirect host: {hostname}")


class HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_download_url(new_url, initial=False)
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


DOWNLOAD_OPENER = urllib.request.build_opener(HttpsOnlyRedirectHandler())


def _open_download(request: urllib.request.Request, timeout: int) -> Any:
    return DOWNLOAD_OPENER.open(request, timeout=timeout)


def _chmod_verified(path: Path, identity: FileIdentity, mode: int) -> None:
    descriptor, observed = _open_regular(path)
    try:
        if observed != identity:
            raise ValueError(f"file identity changed before chmod: {path}")
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def download_or_reuse(
    asset: dict[str, Any], cache_dir: Path
) -> tuple[dict[str, Any], FileIdentity]:
    _validate_cache_directory(cache_dir)
    _validate_download_url(asset["url"], initial=True)
    destination = cache_dir / asset["filename"]
    reused = False
    observed_md5 = ""
    observed_size = 0
    destination_identity: FileIdentity | None = None
    if os.path.lexists(destination):
        destination_identity = regular_file_identity(destination)
        observed_size = destination_identity[2]
        if observed_size == asset["size_bytes"]:
            observed_md5, observed_size = file_digest(destination, "md5")
            reused = observed_md5 == asset["md5"]

    payload_bytes_downloaded = 0
    http_status: int | None = None
    sanitized_final_url: str | None = None
    supplied_content_length: int | None = None
    if not reused:
        request = urllib.request.Request(
            asset["url"],
            headers={
                "User-Agent": "EP21-public-model-payload-audit/2.0",
                "Accept-Encoding": "identity",
            },
        )
        temporary: Path | None = None
        try:
            digest = hashlib.md5()  # Supplier reproduction identity.
            with _open_download(request, timeout=180) as response:
                http_status = int(response.getcode())
                if http_status != 200:
                    raise ValueError(f"unexpected model-download HTTP status: {http_status}")
                final_url = response.geturl()
                _validate_download_url(final_url, initial=False)
                sanitized_final_url = _sanitized_url(final_url)
                lengths = response.headers.get_all("Content-Length") or []
                if not lengths:
                    raise ValueError("download lacks Content-Length")
                try:
                    parsed_lengths = [int(value) for value in lengths]
                except ValueError as error:
                    raise ValueError("invalid download Content-Length") from error
                if len(set(parsed_lengths)) != 1 or parsed_lengths[0] < 0:
                    raise ValueError("conflicting download Content-Length values")
                supplied_content_length = parsed_lengths[0]
                if supplied_content_length != asset["size_bytes"]:
                    raise ValueError(
                        "download Content-Length differs from the immutable pin"
                    )
                if response.headers.get("Transfer-Encoding") is not None:
                    raise ValueError("ambiguous transfer encoding on model download")
                with tempfile.NamedTemporaryFile(
                    "w+b",
                    prefix=f".{destination.name}.part-",
                    dir=cache_dir,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        if payload_bytes_downloaded + len(chunk) > asset["size_bytes"]:
                            raise ValueError("download exceeded its immutable size pin")
                        handle.write(chunk)
                        digest.update(chunk)
                        payload_bytes_downloaded += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.fchmod(handle.fileno(), 0o440)
            observed_md5 = digest.hexdigest()
            observed_size = payload_bytes_downloaded
            if observed_md5 != asset["md5"] or observed_size != asset["size_bytes"]:
                raise ValueError(
                    f"download identity mismatch for {asset['filename']}: "
                    f"md5={observed_md5}, bytes={observed_size}"
                )
            if temporary is None:
                raise RuntimeError("download temporary was not created")
            os.replace(temporary, destination)
            temporary = None
            destination_identity = regular_file_identity(destination)
        finally:
            if temporary is not None:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()
    if destination_identity is None:
        destination_identity = regular_file_identity(destination)
    _chmod_verified(destination, destination_identity, 0o440)
    destination_identity = regular_file_identity(destination)
    return (
        {
            "cache_filename": destination.name,
            "cache_reused_after_identity_check": reused,
            "cache_regular_file_verified": True,
            "cache_symlink_rejected": True,
            "cache_mode": "0440",
            "payload_bytes_downloaded_this_attempt": payload_bytes_downloaded,
            "observed_size_bytes": observed_size,
            "observed_md5": observed_md5,
            "size_matches": observed_size == asset["size_bytes"],
            "md5_matches": observed_md5 == asset["md5"],
            "download_http_status": http_status,
            "download_content_length": supplied_content_length,
            "sanitized_final_url": sanitized_final_url,
        },
        destination_identity,
    )


def _one(items: list[Any], label: str) -> Any:
    if len(items) != 1:
        raise ValueError(f"expected exactly one {label}, got {len(items)}")
    return items[0]


def _attribute_chain(node: ast.AST) -> str | None:
    names: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        names.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    names.append(current.id)
    return ".".join(reversed(names))


def _self_assignment(body: list[ast.stmt], attribute: str) -> ast.AST:
    values = [
        statement.value
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and _attribute_chain(statement.targets[0]) == f"self.{attribute}"
    ]
    return _one(values, f"assignment to self.{attribute}")


def _name_assignment(body: list[ast.stmt], name: str) -> ast.AST:
    values = [
        statement.value
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    ]
    return _one(values, f"assignment to {name}")


def _literal(value: ast.AST, label: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} is not a literal") from error


def _compare_encoder_name(node: ast.AST, value: str) -> bool:
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.left, ast.Name)
        and node.left.id == "encoder_name"
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == value
    )


def _parse_encoder_branch(body: list[ast.stmt], expected_model: str) -> dict[str, Any]:
    encoder = _self_assignment(body, "encoder")
    if not (
        isinstance(encoder, ast.Call)
        and isinstance(encoder.func, ast.Name)
        and encoder.func.id == expected_model
        and not encoder.args
        and not encoder.keywords
    ):
        raise ValueError(f"unexpected {expected_model} constructor")
    checkpoint = _literal(_name_assignment(body, "encoder_param_file"), "checkpoint")
    image_size = _literal(_self_assignment(body, "image_size"), "image size")
    mean_call = _self_assignment(body, "mean_image")
    if not (
        isinstance(mean_call, ast.Call)
        and _attribute_chain(mean_call.func) == "np.float32"
        and len(mean_call.args) == 1
        and not mean_call.keywords
    ):
        raise ValueError(f"unexpected {expected_model} mean conversion")
    mean = _literal(mean_call.args[0], "BGR mean")
    mapping_call = _self_assignment(body, "layer_mapping")
    if not (
        isinstance(mapping_call, ast.Call)
        and isinstance(mapping_call.func, ast.Name)
        and mapping_call.func.id == "layer_map"
        and len(mapping_call.args) == 1
        and not mapping_call.keywords
    ):
        raise ValueError(f"unexpected {expected_model} layer map")
    mapping = _literal(mapping_call.args[0], "layer map")
    direct_calls = [
        statement.value
        for statement in body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    if [_attribute_chain(call.func) for call in direct_calls] != [
        "self.encoder.to",
        "self.encoder.load_state_dict",
        "self.encoder.eval",
    ]:
        raise ValueError(f"unexpected {expected_model} setup-call order")
    load_call = direct_calls[1]
    if not (
        len(load_call.args) == 1
        and not load_call.keywords
        and isinstance(load_call.args[0], ast.Call)
        and _attribute_chain(load_call.args[0].func) == "torch.load"
        and len(load_call.args[0].args) == 1
        and not load_call.args[0].keywords
        and isinstance(load_call.args[0].args[0], ast.Name)
        and load_call.args[0].args[0].id == "encoder_param_file"
    ):
        raise ValueError(f"unexpected {expected_model} state-dict load")
    return {
        "model_class": expected_model,
        "checkpoint_path": checkpoint,
        "checkpoint_provenance": "lab_local_unchecksummed",
        "image_size": image_size,
        "bgr_mean": mean,
        "layer_map": mapping,
        "setup_call_order": ["to", "load_state_dict", "eval"],
    }


def _statement_dump(source: str) -> str:
    return ast.dump(ast.parse(source).body[0], include_attributes=False)


def _statement_index(body: list[ast.stmt], source: str) -> int:
    expected = _statement_dump(source)
    matches = [
        index
        for index, statement in enumerate(body)
        if ast.dump(statement, include_attributes=False) == expected
    ]
    return _one(matches, f"statement {source!r}")


def _validate_feature_extractor(initializer: ast.FunctionDef) -> bool:
    _statement_index(initializer.body, "self.layers = list(self.layer_mapping.keys())")
    value = _self_assignment(initializer.body, "feature_extractor")
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "FeatureExtractor"
        and [_attribute_chain(item) for item in value.args]
        == ["self.encoder", "self.layers", "self.layer_mapping"]
        and [(keyword.arg, _attribute_chain(keyword.value)) for keyword in value.keywords[:1]]
        == [("device", "self.device")]
        and len(value.keywords) == 2
        and value.keywords[1].arg == "detach"
        and isinstance(value.keywords[1].value, ast.Constant)
        and value.keywords[1].value.value is True
    ):
        raise ValueError("FeatureExtractor does not consume the branch mapping exactly")
    return True


def _parse_preprocessing(handler: ast.ClassDef) -> dict[str, bool]:
    activation = _one(
        [
            node
            for node in handler.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_activation"
        ],
        "DNNHandler.get_activation",
    )
    image_loop = _one(
        [
            node
            for node in ast.walk(activation)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "a_img"
        ],
        "per-image preprocessing loop",
    )
    resize_index = _statement_index(
        image_loop.body, "a_img = a_img.resize(self.image_size, Image.LANCZOS)"
    )
    array_index = _statement_index(image_loop.body, "x = np.asarray(a_img)")
    guarded = [
        (index, node)
        for index, node in enumerate(image_loop.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _attribute_chain(node.test.left) == "self.encoder_name"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.In)
        and len(node.test.comparators) == 1
        and _literal(node.test.comparators[0], "preprocessing encoder list")
        == ["AlexNet", "VGG19"]
    ]
    guard_index, guard = _one(guarded, "shared AlexNet/VGG19 preprocessing guard")
    bgr_index = _statement_index(
        guard.body, "x = np.transpose(x, (2, 0, 1))[::-1]"
    )
    mean_index = _statement_index(
        guard.body,
        "x = np.float32(x) - np.reshape(self.mean_image, (3, 1, 1))",
    )
    extraction_index = _statement_index(
        image_loop.body, "features = self.feature_extractor.run(x)"
    )
    if not (resize_index < array_index < guard_index < extraction_index):
        raise ValueError("tagged evaluator preprocessing order changed")
    if not bgr_index < mean_index:
        raise ValueError("BGR conversion must precede mean subtraction")
    return {
        "lanczos_resize_to_branch_image_size": True,
        "image_to_array": True,
        "rgb_to_bgr_chw": True,
        "float32_channel_mean_subtraction": True,
        "branch_feature_extractor_consumption": True,
    }


def parse_source_requirements(
    text: str, script_path: str, source_archive_md5: str
) -> dict[str, Any]:
    tree = ast.parse(text)
    handler = _one(
        [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DNNHandler"
        ],
        "DNNHandler class",
    )
    initializer = _one(
        [
            node
            for node in handler.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ],
        "DNNHandler.__init__",
    )
    top_if = _one(
        [
            node
            for node in initializer.body
            if isinstance(node, ast.If) and _compare_encoder_name(node.test, "AlexNet")
        ],
        "top-level AlexNet branch",
    )
    if not (
        len(top_if.orelse) == 1
        and isinstance(top_if.orelse[0], ast.If)
        and _compare_encoder_name(top_if.orelse[0].test, "VGG19")
    ):
        raise ValueError("VGG19 is not the direct alternate evaluator branch")
    alexnet = _parse_encoder_branch(top_if.body, "AlexNet")
    vgg19 = _parse_encoder_branch(top_if.orelse[0].body, "VGG19")
    expected = {
        "AlexNet": {
            "checkpoint_path": "/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt",
            "image_size": [227, 227],
            "bgr_mean": [104.0, 117.0, 123.0],
            "layer_map": "alexnet",
        },
        "VGG19": {
            "checkpoint_path": "/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt",
            "image_size": [224, 224],
            "bgr_mean": [104.0, 117.0, 123.0],
            "layer_map": "vgg19",
        },
    }
    for name, facts in (("AlexNet", alexnet), ("VGG19", vgg19)):
        for key, expected_value in expected[name].items():
            if facts[key] != expected_value:
                raise ValueError(f"unexpected {name} {key}: {facts[key]!r}")
    _validate_feature_extractor(initializer)
    preprocessing = _parse_preprocessing(handler)
    return {
        "source_archive_md5": source_archive_md5,
        "script_path": script_path,
        "script_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "evaluator_branches": {"alexnet": alexnet, "vgg19": vgg19},
        "preprocessing_assertions": preprocessing,
    }


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in tagged manifest: {key}")
        result[key] = value
    return result


def parse_reconstruction_manifest(text: str) -> dict[str, Any]:
    payload = json.loads(text, object_pairs_hook=_no_duplicate_object)
    recon_demo = payload.get("recon_demo", {})
    if recon_demo.get("save_in") != "./":
        raise ValueError("tagged reconstruction manifest save_in changed")
    files = recon_demo.get("files", [])
    expected = {
        "vgg19": ("VGG_ILSVRC_19_layers.zip", 38225868, ASSETS["vgg19"]["md5"]),
        "caffenet": (
            "bvlc_reference_caffenet.zip",
            52398902,
            ASSETS["caffenet"]["md5"],
        ),
    }
    parsed: dict[str, Any] = {}
    for asset_id, (name, file_id, md5) in expected.items():
        item = _one(
            [candidate for candidate in files if candidate.get("name") == name],
            f"manifest record for {name}",
        )
        url_path = urllib.parse.urlsplit(item.get("url", "")).path.rstrip("/")
        if url_path.split("/")[-1] != str(file_id):
            raise ValueError(f"manifest file ID changed for {name}")
        if item.get("md5sum") != md5:
            raise ValueError(f"manifest MD5 changed for {name}")
        if item.get("postproc") != [{"name": "unzip", "destination": "./models"}]:
            raise ValueError(f"manifest extraction destination changed for {name}")
        parsed[asset_id] = {
            "name": name,
            "figshare_file_id": file_id,
            "md5": md5,
            "save_in": "./",
            "extraction_destination": "./models",
        }
    return parsed


def parse_vgg_config(text: str) -> dict[str, Any]:
    def scalar(name: str) -> str:
        matches = re.findall(
            rf"^\s*{re.escape(name)}:\s*([^#\n]+?)\s*$", text, re.MULTILINE
        )
        return _one(matches, f"VGG config {name}")

    layers_match = re.search(r"^layers:\s*$([\s\S]*)", text, re.MULTILINE)
    if layers_match is None:
        raise ValueError("VGG config lacks layers")
    observed = {
        "name": scalar("name"),
        "parameters_file": scalar("parameters_file"),
        "input_image_shape": _literal(
            ast.parse(scalar("input_image_shape"), mode="eval").body,
            "VGG config input shape",
        ),
        "image_mean_file": scalar("image_mean_file"),
        "feature_std_file": scalar("feature_std_file"),
        "layers": re.findall(
            r"^\s*-\s+([A-Za-z0-9_]+)\s*$", layers_match.group(1), re.MULTILINE
        ),
    }
    expected = {
        "name": "vgg19",
        "parameters_file": "./data/models/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt",
        "input_image_shape": [224, 224, 3],
        "image_mean_file": "./data/models/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy",
        "feature_std_file": "./data/models/VGG_ILSVRC_19_layers/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat",
        "layers": VGG_LAYERS,
    }
    if observed != expected:
        raise ValueError("tagged VGG reconstruction configuration changed")
    observed["archive_member_paths"] = {
        "checkpoint": VGG_CHECKPOINT_MEMBER,
        "mean": "VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy",
        "feature_std": "VGG_ILSVRC_19_layers/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat",
    }
    return observed


def source_requirements(source_archive: Path) -> dict[str, Any]:
    exact_names = {
        "script": SOURCE_ROOT + IDENTIFICATION_SCRIPT_SUFFIX,
        "manifest": SOURCE_ROOT + RECONSTRUCTION_MANIFEST_SUFFIX,
        "vgg_config": SOURCE_ROOT + VGG_CONFIG_SUFFIX,
    }
    with source_archive_verifier.verified_zip(
        source_archive, SOURCE_ARCHIVE_MD5, SOURCE_ARCHIVE_SIZE_BYTES
    ) as verified:
        observed_md5 = verified.md5
        archive = verified.archive
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("tagged source archive has duplicate member names")
        for label, name in exact_names.items():
            if names.count(name) != 1 or not safe_member_name(name):
                raise ValueError(f"tagged source archive lacks exact {label} member")
        script_text = archive.read(exact_names["script"]).decode(
            "utf-8", errors="strict"
        )
        manifest_text = archive.read(exact_names["manifest"]).decode(
            "utf-8", errors="strict"
        )
        config_text = archive.read(exact_names["vgg_config"]).decode(
            "utf-8", errors="strict"
        )
    requirements = parse_source_requirements(
        script_text, exact_names["script"], observed_md5
    )
    requirements["tagged_reconstruction_binding"] = {
        "manifest_path": exact_names["manifest"],
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "manifest_assets": parse_reconstruction_manifest(manifest_text),
        "vgg_config_path": exact_names["vgg_config"],
        "vgg_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "vgg_config": parse_vgg_config(config_text),
    }
    return requirements


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.attempt <= 0:
        parser.error("--attempt must be positive")

    started_at = utc_now()
    process_id = os.getpid()
    print(
        json.dumps(
            {
                "event": "ep21_public_model_payload_audit_started",
                "pid": process_id,
                "attempt": args.attempt,
                "started_at": started_at,
            }
        ),
        flush=True,
    )
    requirements = source_requirements(args.source_archive)
    audited_assets: dict[str, Any] = {}
    total_downloaded = 0
    with cache_lock(args.cache_dir):
        for asset_id, pinned in ASSETS.items():
            cache, identity = download_or_reuse(pinned, args.cache_dir)
            archive_path = args.cache_dir / pinned["filename"]
            inventory = inspect_zip(archive_path, expected_identity=identity)
            exact_named_members = matching_basename(
                inventory, pinned["tagged_checkpoint_basename"]
            )
            audited_assets[asset_id] = {
                "pinned_identity": pinned,
                "cache": cache,
                "archive_inventory": inventory,
                "exact_tagged_checkpoint_named_members": exact_named_members,
                "exact_tagged_checkpoint_named_member_count": len(
                    exact_named_members
                ),
            }
            total_downloaded += cache["payload_bytes_downloaded_this_attempt"]

    vgg_required_paths = set(
        requirements["tagged_reconstruction_binding"]["vgg_config"][
            "archive_member_paths"
        ].values()
    )
    vgg_inventory_paths = {
        item["path"] for item in audited_assets["vgg19"]["archive_inventory"]["members"]
    }
    if not vgg_required_paths.issubset(vgg_inventory_paths):
        raise ValueError("VGG archive lacks one or more exact tagged config members")
    vgg_checkpoint_members = matching_path(
        audited_assets["vgg19"]["archive_inventory"], VGG_CHECKPOINT_MEMBER
    )
    caffenet_candidate_members = matching_path(
        audited_assets["caffenet"]["archive_inventory"],
        CAFFENET_CHECKPOINT_MEMBER,
    )
    vgg_public_exact = len(vgg_checkpoint_members) == 1
    alexnet_public_exact = (
        audited_assets["caffenet"][
            "exact_tagged_checkpoint_named_member_count"
        ]
        == 1
    )
    caffenet_candidate_exact = len(caffenet_candidate_members) == 1
    if not vgg_public_exact or not caffenet_candidate_exact:
        raise ValueError("public archive lacks an exact tagged manifest/config member")
    if alexnet_public_exact:
        raise ValueError("unexpected bvlc_alexnet.pt appeared in the CaffeNet archive")
    all_archives_valid = all(
        item["cache"]["md5_matches"]
        and item["cache"]["size_matches"]
        and item["archive_inventory"]["unsafe_member_names"] == []
        and item["archive_inventory"]["non_regular_member_names"] == []
        for item in audited_assets.values()
    )
    if not all_archives_valid:
        raise ValueError("one or more public model archives failed identity or safety")
    if not all(requirements["preprocessing_assertions"].values()):
        raise ValueError("tagged identification preprocessing assertions failed")

    report = {
        "schema_version": "ep21.public_model_payload_audit.v2",
        "astra_analysis_id": "source_qualification",
        "attempt": args.attempt,
        "started_at": started_at,
        "completed_at": utc_now(),
        "process_id": process_id,
        "scope": "two_pinned_public_non_empirical_model_archives",
        "source_requirements": requirements,
        "assets": audited_assets,
        "public_reconstruction_bindings": {
            "vgg19": {
                "state": "tagged_manifest_config_and_archive_members_authenticated",
                "checkpoint_member_verified": vgg_public_exact,
                "checkpoint_member": vgg_checkpoint_members[0],
                "required_member_paths_verified": sorted(vgg_required_paths),
            },
            "caffenet": {
                "state": "tagged_manifest_archive_and_candidate_member_authenticated",
                "candidate_checkpoint_member_verified": caffenet_candidate_exact,
                "candidate_checkpoint_member": caffenet_candidate_members[0],
            },
        },
        "evaluator_checkpoint_references": {
            "vgg19": {
                "state": "lab_local_copy_unchecksummed_public_member_same_name_only",
                "lab_local_path": requirements["evaluator_branches"]["vgg19"][
                    "checkpoint_path"
                ],
                "public_reconstruction_member": vgg_checkpoint_members[0],
                "lab_local_byte_identity_verified": False,
                "strict_bdpy_runtime_load_verified": False,
                "remaining_action": "Authenticate the lab-local evaluator copy or prospectively bind the public member, then validate a strict load in the locked runtime.",
            },
            "alexnet": {
                "state": "lab_local_checkpoint_unmanifested_public_caffenet_not_substitutable",
                "lab_local_path": requirements["evaluator_branches"]["alexnet"][
                    "checkpoint_path"
                ],
                "exact_public_bvlc_alexnet_named_member_verified": alexnet_public_exact,
                "public_caffenet_candidate_member": caffenet_candidate_members[0],
                "caffenet_equivalence_claim_allowed": False,
                "content_level_non_equivalence_verified_by_this_runner": False,
                "remaining_action": "Authenticate bvlc_alexnet.pt or lock AlexNet identification findings not_evaluable; do not substitute CaffeNet by convenience.",
            },
        },
        "payload_bytes_downloaded_this_attempt": total_downloaded,
        "empirical_or_decoded_payload_accessed": False,
        "ood_or_ep04_payload_accessed": False,
        "checkpoint_deserialization_performed": False,
        "access_boundary": {
            "allowed_payload_classes": [
                "pinned_public_source_archive",
                "two_pinned_public_non_empirical_model_archives",
            ],
            "cache_regular_file_and_symlink_controls_enforced": True,
            "archive_members_streamed_without_extraction": True,
            "boundary_attestation_basis": "runner code path and exact CLI/cache whitelist; no claim about external filesystem state beyond those validated paths",
        },
        "outcome": "completed_public_reconstruction_assets_authenticated_evaluator_checkpoints_unbound",
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "event": "ep21_public_model_payload_audit_completed",
                "pid": process_id,
                "attempt": args.attempt,
                "completed_at": report["completed_at"],
                "output": str(args.output),
                "outcome": report["outcome"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
