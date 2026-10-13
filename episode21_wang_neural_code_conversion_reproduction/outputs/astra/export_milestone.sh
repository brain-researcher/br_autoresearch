#!/usr/bin/env bash
# EP21-only export using the existing pinned ASTRA runtime.
set -euo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge >/dev/null 2>&1
module load devel math
module load python/3.12.1 py-numpy/2.2.6_py312 py-scipy/1.16.0_py312

task_astra_runtime="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules"
export PYTHONPATH="$task_astra_runtime/brain_researcher/src${PYTHONPATH:+:$PYTHONPATH}"
task_export_dir=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$task_astra_runtime/venv/bin/python" -B "$task_export_dir/export_milestone.py" "$@"
