#!/usr/bin/env bash
#SBATCH --job-name=ep10-target-env
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

target="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312"
cache_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/python_build_cache"

if [[ -e "$target" ]]; then
  printf 'refusing to replace existing environment: %s\n' "$target" >&2
  exit 1
fi

mkdir -p "$(dirname "$target")" "$cache_root/pip" "$cache_root/xdg"
export PIP_CACHE_DIR="$cache_root/pip"
export XDG_CACHE_HOME="$cache_root/xdg"
export PYTHONNOUSERSITE=1

ml purge
ml load devel python/3.12.1

python3 -m venv "$target"
"$target/bin/python" -m pip install --only-binary=:all: \
  'numpy==1.26.4' \
  'PyYAML==6.0.3' \
  'pynrrd==1.1.3' \
  'typing_extensions==4.16.0'

"$target/bin/python" - <<'PY'
from importlib.metadata import version
import pathlib
import sys

import nrrd
import numpy
import typing_extensions
import yaml

prefix = pathlib.Path(sys.prefix).resolve()
for module in (nrrd, numpy, typing_extensions, yaml):
    path = pathlib.Path(module.__file__).resolve()
    if prefix not in path.parents:
        raise RuntimeError(f"dependency escaped the clean environment: {path}")
print(f"python={sys.version.split()[0]}")
for distribution in ("numpy", "PyYAML", "pynrrd", "typing_extensions"):
    print(f"{distribution}={version(distribution)}")
PY

"$target/bin/python" -m pip check
"$target/bin/python" -m pip freeze
