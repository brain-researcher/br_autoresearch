#!/usr/bin/env python3
"""Identity-stable verification for the pinned public EP21 source ZIP."""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
import zipfile
from pathlib import Path
from typing import Iterator, NamedTuple


CHUNK_BYTES = 4 * 1024 * 1024
FileIdentity = tuple[int, int, int, int, int]


class VerifiedZip(NamedTuple):
    archive: zipfile.ZipFile
    md5: str
    size: int


def _identity(info: os.stat_result) -> FileIdentity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _stream_md5(handle) -> tuple[str, int]:
    digest = hashlib.md5()  # Required release identity, not security use.
    byte_count = 0
    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


@contextlib.contextmanager
def verified_zip(
    path: Path, expected_md5: str, expected_size: int | None = None
) -> Iterator[VerifiedZip]:
    """Hash and parse one held regular-file descriptor without reopening a path."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open source archive: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"source archive is not a single-link regular file: {path}")
        initial_identity = _identity(info)
        if expected_size is not None and info.st_size != expected_size:
            raise ValueError(
                f"source archive size mismatch: expected {expected_size}, got {info.st_size}"
            )
        with os.fdopen(descriptor, "rb", closefd=True) as raw_handle:
            descriptor = -1
            observed_md5, byte_count = _stream_md5(raw_handle)
            hash_identity = _identity(os.fstat(raw_handle.fileno()))
            if hash_identity != initial_identity or byte_count != info.st_size:
                raise ValueError("source archive changed while hashing")
            if observed_md5 != expected_md5:
                raise ValueError(
                    f"source archive MD5 mismatch: expected {expected_md5}, got {observed_md5}"
                )
            raw_handle.seek(0)
            try:
                with zipfile.ZipFile(raw_handle) as archive:
                    yield VerifiedZip(archive, observed_md5, byte_count)
            finally:
                final_identity = _identity(os.fstat(raw_handle.fileno()))
                if final_identity != initial_identity:
                    raise ValueError("source archive changed during ZIP inspection")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
