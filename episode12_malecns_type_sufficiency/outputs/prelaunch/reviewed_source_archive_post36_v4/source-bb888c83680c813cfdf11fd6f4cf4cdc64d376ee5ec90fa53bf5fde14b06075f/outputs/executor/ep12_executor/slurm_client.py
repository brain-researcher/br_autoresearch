"""Minimal process environment for EP12 Slurm client commands.

Slurm treats environment variables such as ``SBATCH_*`` as command options.
Passing the login environment to ``sbatch`` would therefore let stale ambient
state override a source-frozen wrapper.  Scheduler clients need only an
executable search path and optional locale selection, so use a strict allowlist
for every subprocess invocation.
"""

from __future__ import annotations

import os
from typing import Mapping


_LOCALE_NAMES = frozenset({"LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE"})
_TRUSTED_CLIENT_PATH = "/usr/bin:/bin"


def sanitized_slurm_client_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with no Slurm/Sbatch option aliases."""

    ambient = os.environ if source is None else source
    # Sherlock's scheduler clients are the system binaries under /usr/bin.
    # Never let a caller-controlled PATH redirect a control-plane invocation
    # to an alias, function shim, virtual environment, or staged executable.
    clean = {"PATH": _TRUSTED_CLIENT_PATH}
    for name in sorted(_LOCALE_NAMES):
        value = ambient.get(name)
        if value is not None and "\x00" not in value:
            clean[name] = value
    return clean


__all__ = ["sanitized_slurm_client_environment"]
