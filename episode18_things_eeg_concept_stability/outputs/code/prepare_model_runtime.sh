#!/bin/bash
# Historical first installation in46624388; the module route is STOPPED.
# The current asset map records the regex correction and actual Torch/vision
# release mismatch. Do not rerun this helper or infer a qualified runtime.
# Coordinator-only, invoked under an external 600-second Slurm step bound.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* && "${SLURM_CPUS_PER_TASK:-}" == 1 ]]
cd "$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_environment="$GROUP_HOME/users/zijiao/ep18_model_runtime_py312"
[[ ! -e "$ep18_environment" ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
export TMPDIR="$L_SCRATCH_JOB/ep18-runtime-tmp"
export XDG_CACHE_HOME="$L_SCRATCH_JOB/ep18-runtime-cache"
export HF_HOME="$L_SCRATCH_JOB/ep18-runtime-hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
module purge
module load devel math viz python/3.12.1 py-torchvision/0.19.1_py312 py-scikit-image/0.24.0_py312 py-safetensors/0.4.2_py312 py-regex/20247.24_py312
module load py-pillow/10.2.0_py312
python3.12 -m venv --system-site-packages "$ep18_environment"
"$ep18_environment/bin/python" -B - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

mapping = json.loads(Path('outputs/config/model_runtime_asset_map_proposal.json').read_text())
wheels = mapping['runtime_proposal']['additional_wheels']
assert len(wheels) == 6
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
    '--no-cache-dir', '--no-deps', '--only-binary=:all:'] +
    [entry['url'] + '#sha256=' + entry['sha256'] for entry in wheels], check=True)
PY
export PYTHONPATH="outputs/code${PYTHONPATH:+:$PYTHONPATH}"
"$ep18_environment/bin/python" -B - <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import qualify_model_runtime as qualification

qualification.require_compute()
versions = qualification.qualify_versions(qualification.read_json(qualification.REGISTRY))
record = {'analysis_id': 'feature_preparation', 'status': 'runtime_versions_qualified',
    'finished_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
    'job_id': os.environ['SLURM_JOB_ID'], 'runtime': versions,
    'model_assets_loaded': False, 'inference_performed': False,
    'image_or_EEG_access': False, 'audit_seal_established': False}
destination = Path('outputs/runs/feature_preparation') / f"20261005-runtime-environment-{os.environ['SLURM_JOB_ID']}" / 'runtime_environment.json'
with destination.open('x') as output:
    json.dump(record, output, indent=2)
    output.write('\n')
print(json.dumps(record, indent=2))
PY
