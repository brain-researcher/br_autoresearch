"""No-follow directory and lock primitives for EP12 durable transactions."""

from __future__ import annotations

import os
from pathlib import Path
import stat


class SecureTransactionError(RuntimeError):
    """A durable transaction directory or lock is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SecureTransactionError(message)


def open_secure_directory(
    path: Path, *, root: Path, create: bool = True, mode: int = 0o750
) -> tuple[Path, int]:
    """Open ``path`` beneath ``root`` without following any child symlink."""

    boundary = Path(os.path.abspath(os.fspath(root)))
    target = Path(os.path.abspath(os.fspath(path)))
    _require(boundary.is_absolute() and target.is_absolute(), "transaction path is not absolute")
    _require(boundary == boundary.resolve(strict=True), "transaction root traverses a symlink")
    try:
        relative = target.relative_to(boundary)
    except ValueError as exc:
        raise SecureTransactionError("transaction path escapes its root") from exc
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(boundary, flags)
    try:
        for component in relative.parts:
            _require(component not in {"", ".", ".."}, "unsafe transaction component")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                _require(create, f"missing transaction directory: {target}")
                os.mkdir(component, mode=mode, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            _require(stat.S_ISDIR(metadata.st_mode), "transaction component is not a directory")
            os.close(descriptor)
            descriptor = child
        return target, descriptor
    except SecureTransactionError:
        os.close(descriptor)
        raise
    except OSError as exc:
        os.close(descriptor)
        raise SecureTransactionError(
            f"cannot safely open transaction directory {target}: {exc}"
        ) from exc


def open_regular_lock_at(directory_fd: int, name: str, *, mode: int = 0o640) -> int:
    """Create/open one regular lock relative to an already safe directory fd."""

    _require("/" not in name and name not in {"", ".", ".."}, "unsafe lock name")
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise SecureTransactionError(f"cannot safely open transaction lock: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "transaction lock is not regular")
        _require(
            (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino),
            "transaction lock path changed while opening",
        )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


__all__ = ["SecureTransactionError", "open_regular_lock_at", "open_secure_directory"]
