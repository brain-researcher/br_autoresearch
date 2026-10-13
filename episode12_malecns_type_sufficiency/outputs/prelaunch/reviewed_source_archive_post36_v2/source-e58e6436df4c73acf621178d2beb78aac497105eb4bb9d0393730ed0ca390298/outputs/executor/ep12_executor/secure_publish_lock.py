"""Acquire a no-follow regular-file lease and hold it across ``exec``.

The full-null aggregate publishes one durable tree.  Opening its lock with a
shell redirection would truncate a symlink target before ``flock`` could
inspect it.  This helper opens the exact lock path with ``O_NOFOLLOW``, checks
that the opened inode is a regular file, acquires an exclusive lease, makes
the descriptor inheritable, and then replaces itself with the publication
command.  The same open file description (and therefore the lease) remains
live until that command exits.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import stat
from typing import Sequence


LOCK_FD_ENV = "EP12_SECURE_PUBLISH_LOCK_FD"
LOCK_PATH_ENV = "EP12_SECURE_PUBLISH_LOCK_PATH"


class SecurePublishLockError(RuntimeError):
    """The publication lease path or inherited descriptor is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SecurePublishLockError(message)


def acquire_publish_lock(path: Path) -> int:
    """Open and exclusively lease one non-symlink regular lock file."""

    raw = Path(path)
    _require(raw.is_absolute(), "publish lock path must be absolute")
    _require(".." not in raw.parts, "publish lock path may not contain traversal")
    parent = raw.parent.resolve(strict=True)
    lexical = Path(os.path.abspath(os.fspath(raw)))
    _require(lexical.parent == parent, "publish lock parent may not traverse a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags, 0o640)
    except OSError as exc:
        raise SecurePublishLockError(f"cannot safely open publish lock {lexical}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "publish lock is not a regular file")
        path_metadata = os.lstat(lexical)
        _require(not stat.S_ISLNK(path_metadata.st_mode), "publish lock is a symlink")
        _require(
            (metadata.st_dev, metadata.st_ino)
            == (path_metadata.st_dev, path_metadata.st_ino),
            "publish lock path changed while it was opened",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.set_inheritable(descriptor, True)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def verify_inherited_publish_lock(path: Path, descriptor: int) -> None:
    """Verify that ``descriptor`` still leases the exact regular lock inode."""

    _require(descriptor >= 0, "inherited publish lock descriptor is invalid")
    raw = Path(path)
    _require(raw.is_absolute(), "inherited publish lock path must be absolute")
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = os.lstat(raw)
    except OSError as exc:
        raise SecurePublishLockError(f"cannot verify inherited publish lock: {exc}") from exc
    _require(stat.S_ISREG(descriptor_metadata.st_mode), "inherited lock is not regular")
    _require(not stat.S_ISLNK(path_metadata.st_mode), "inherited lock path is a symlink")
    _require(
        (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        == (path_metadata.st_dev, path_metadata.st_ino),
        "inherited publish lock path no longer names the leased inode",
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SecurePublishLockError("inherited descriptor does not own the lease") from exc


def exec_under_publish_lock(path: Path, command: Sequence[str]) -> None:
    """Acquire ``path`` and ``exec`` ``command`` without releasing the lease."""

    _require(bool(command), "publish-lock exec requires a command")
    descriptor = acquire_publish_lock(path)
    environment = dict(os.environ)
    environment[LOCK_FD_ENV] = str(descriptor)
    environment[LOCK_PATH_ENV] = str(Path(path))
    try:
        os.execvpe(str(command[0]), list(command), environment)
    except Exception:
        os.close(descriptor)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-secure-publish-lock")
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    execute = subparsers.add_parser("exec")
    execute.add_argument("--lock-path", type=Path, required=True)
    execute.add_argument("command", nargs=argparse.REMAINDER)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--lock-path", type=Path, required=True)
    verify.add_argument("--fd", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command_name == "exec":
        command = list(arguments.command)
        if command and command[0] == "--":
            command = command[1:]
        exec_under_publish_lock(arguments.lock_path, command)
        raise AssertionError("exec returned unexpectedly")
    verify_inherited_publish_lock(arguments.lock_path, arguments.fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCK_FD_ENV",
    "LOCK_PATH_ENV",
    "SecurePublishLockError",
    "acquire_publish_lock",
    "exec_under_publish_lock",
    "verify_inherited_publish_lock",
]
