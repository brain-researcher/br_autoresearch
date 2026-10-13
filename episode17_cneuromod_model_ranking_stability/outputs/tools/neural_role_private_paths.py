"""Create new EP17 private directories with exact 0700, including under setgid parents."""
from __future__ import annotations

import os
import stat

import reconcile_released_events as core


def private_directory(path, parent):
    core.require(path.is_relative_to(parent) and parent.is_dir() and not parent.is_symlink(), "private_parent_invalid")
    current = parent
    for part in path.relative_to(parent).parts:
        current /= part
        if not current.exists():
            current.mkdir(mode=0o700)
            # A setgid parent can add 02000 even when mkdir requested 0700.
            # Only normalize this call's new, owned directory, before content.
            descriptor = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                info = os.fstat(descriptor)
                core.require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(), "new_private_directory_owner_changed")
                os.fchmod(descriptor, 0o700)
            finally:
                os.close(descriptor)
        core.require(not current.is_symlink() and current.is_dir() and stat.S_IMODE(current.stat().st_mode) == 0o700,
                     "private_directory_permissions_invalid")
