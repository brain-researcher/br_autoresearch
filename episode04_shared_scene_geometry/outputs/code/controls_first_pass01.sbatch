#!/bin/bash
# Registered sensitivities plus retained encoding/null cost units after observed RSA.
#SBATCH --job-name=ep04-controls-first
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/controls_diagnostics/20261005-first01-%j.out
#SBATCH --error=outputs/runs/controls_diagnostics/20261005-first01-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep04_controls_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s\n' "$ep04_controls_host" "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}"
[[ "$ep04_controls_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 2 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scipy/1.12.0_py312 viz py-pillow/10.2.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
bash -n outputs/code/controls_first_pass01.sbatch
bash -n outputs/code/controls_first_pass_worker.sh
python3.12 -B - <<'PY'
import ast
from pathlib import Path
for name in ('run_native_sensitivities.py', 'run_identical_fold_encoding.py', 'run_registered_null_unit.py'):
    ast.parse((Path('outputs/code') / name).read_text())
for name in ('test_identical_fold_encoding.py', 'test_registered_null_unit.py'):
    ast.parse((Path('outputs/tests') / name).read_text())
PY
ep04_controls_run="outputs/runs/controls_diagnostics/20261005-first01-${SLURM_JOB_ID}"
mkdir "$ep04_controls_run"
ep04_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
timeout --signal=TERM --kill-after=10s 30s "$ep04_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep04_controls_run/astra.json" 2>&1
ep04_astra_exit=$?
set -e
printf 'astra_validation_exit=%s\n' "$ep04_astra_exit"
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  bash outputs/code/controls_first_pass_worker.sh sensitivities "$ep04_controls_run" \
  > "$ep04_controls_run/sensitivities.log" 2>&1 &
ep04_sensitivity_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
  bash outputs/code/controls_first_pass_worker.sh encoding_null "$ep04_controls_run" \
  > "$ep04_controls_run/encoding_null.log" 2>&1 &
ep04_diagnostic_pid=$!
set +e
wait "$ep04_sensitivity_pid"
ep04_sensitivity_exit=$?
wait "$ep04_diagnostic_pid"
ep04_diagnostic_exit=$?
set -e
printf 'sensitivities\t%s\nencoding_null\t%s\n' \
  "$ep04_sensitivity_exit" "$ep04_diagnostic_exit" > "$ep04_controls_run/worker_exit_status.tsv"
cat "$ep04_controls_run/worker_exit_status.tsv"
[[ "$ep04_sensitivity_exit" == 0 && "$ep04_diagnostic_exit" == 0 ]]
