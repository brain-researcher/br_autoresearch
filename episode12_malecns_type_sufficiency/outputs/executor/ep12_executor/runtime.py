"""Small shared runtime helpers for the live EP12 scientific executor."""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)


def atomic_json(path: Path, value: Any) -> None:
    """Write one JSON record atomically without creating an identity receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_once(path: Path, content: bytes, *, mode: int = 0o640) -> bool:
    """Atomically publish complete bytes without following or replacing links.

    The temporary inode is fully written and synced before a hard link publishes
    it at the destination.  An identical regular file is an idempotent retry;
    every other pre-existing destination is rejected.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | nofollow,
    )
    temporary_name = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"

    def existing_bytes() -> bytes:
        metadata = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise FileExistsError(f"refusing non-regular write-once target: {path}")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | nofollow,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(descriptor)

    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            mode,
            dir_fd=directory_fd,
        )
        try:
            owned_descriptor = descriptor
            descriptor = -1
            with os.fdopen(owned_descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if existing_bytes() != content:
                    raise FileExistsError(
                        f"refusing to replace a different write-once file: {path}"
                    )
                return False
            os.fsync(directory_fd)
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)
