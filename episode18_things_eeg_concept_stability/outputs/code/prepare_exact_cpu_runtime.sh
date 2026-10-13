#!/bin/bash
# Finite private-runtime correction. Invoke under the provisioning job's
# 2400-second timeout; never modify the stopped module environment.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* && "${SLURM_CPUS_PER_TASK:-}" == 1 ]]
: "${GROUP_HOME:?}" "${OAK:?}" "${L_SCRATCH_JOB:?}" "${EP18_PROVISION_RUN:?}"
cd "$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_environment="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_${SLURM_JOB_ID}"
[[ ! -e "$ep18_environment" && ! -L "$ep18_environment" ]]
[[ -d "$EP18_PROVISION_RUN" && ! -L "$EP18_PROVISION_RUN" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/tmp"
export PIP_CACHE_DIR="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/pip"
export XDG_CACHE_HOME="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/cache"
export HF_HOME="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/hf"
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_FIND_LINKS PIP_TRUSTED_HOST PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
python3.12 -B - <<'PY'
import platform
if platform.python_version() != '3.12.1':
    raise RuntimeError('The unchanged registry requires Python 3.12.1')
PY
python3.12 -m venv "$ep18_environment"
# Installing the two CPU wheels without dependencies first prevents a PyPI
# resolver from selecting a CUDA distribution. The second finite resolution
# fills their ordinary dependencies while constraining every selected version.
"$ep18_environment/bin/python" -B -m pip install --disable-pip-version-check \
  --no-cache-dir --only-binary=:all: --retries 0 --timeout 30 --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  --report "$EP18_PROVISION_RUN/torch_cpu_install.json" \
  'torch==2.4.1+cpu' 'torchvision==0.19.1+cpu'
"$ep18_environment/bin/python" -B -m pip install --disable-pip-version-check \
  --no-cache-dir --only-binary=:all: --retries 0 --timeout 30 \
  --index-url https://pypi.org/simple \
  --constraint outputs/config/cpu_runtime_requirements.txt \
  --requirement outputs/config/cpu_runtime_requirements.txt \
  --report "$EP18_PROVISION_RUN/runtime_install.json"
"$ep18_environment/bin/python" -B -m pip check
"$ep18_environment/bin/python" -B - <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path('outputs/code').resolve()))
import qualify_model_runtime as qualification

qualification.require_compute()
registry = qualification.read_json(qualification.REGISTRY)
versions = qualification.qualify_versions(registry)
environment = Path(sys.prefix).resolve()
for name in registry['runtime_recipe']['target_versions']:
    if not Path(versions[name]['path']).absolute().is_relative_to(environment):
        raise RuntimeError(f'Runtime package is outside the fresh private environment: {name}')
if versions['torch']['version'] != '2.4.1+cpu' or versions['torchvision']['version'] != '0.19.1+cpu':
    raise RuntimeError('Exact selected CPU distributions required')
record = {'analysis_id': 'feature_preparation', 'status': 'runtime_versions_qualified',
          'finished_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
          'job_id': os.environ['SLURM_JOB_ID'], 'environment': str(environment), 'runtime': versions,
          'installed_distributions': sorted((item.metadata['Name'], item.version)
                                            for item in importlib.metadata.distributions()),
          'model_assets_loaded': False, 'inference_performed': False,
          'image_or_EEG_access': False, 'audit_seal_established': False}
with (Path(os.environ['EP18_PROVISION_RUN']) / 'runtime_environment.json').open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
print(json.dumps(record, sort_keys=True), flush=True)
PY
