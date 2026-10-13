#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvz-ready
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-ready-group-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVZ-ready-group-%A_%a.err

# Root selects only --array=8 --dependency=afterany:46580387_8.
# This is the previously unsubmitted MVZ component for indices29–32;
# the existing group1 attempt and other components remain untouched.
# Baseline-pool tests already passed in46583757; run the six indexed-MVZ
# tests here before the unchanged group's four cohorts/two fits each.
# Two3300+10-second rounds plus120+10 tests and30+10 ASTRA =6790<7200s.
# No submission, polling, retry or additional allocation occurs here.
set -euo pipefail
EP_NULL_MVZ_READY_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s GROUP=%s NTASKS=%s CPUS_PER_TASK=%s SOURCE=46580387_8 INDICES=29-32\n' \
  "$EP_NULL_MVZ_READY_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if (( $# != 0 )) || [[ -z "${SLURM_JOB_ID-}" || -z "${SLURM_ARRAY_JOB_ID-}" || \
      "$EP_NULL_MVZ_READY_HOST" == *-ln* || "${SLURM_ARRAY_TASK_ID-}" != 8 || \
      "${SLURM_NTASKS-}" != 2 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires selected array task8 on a compute node, two tasks, one CPU each and no script arguments\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_MVZ_READY_PYTHON=$(command -v python3.12)
"$EP_NULL_MVZ_READY_PYTHON" --version
bash -n outputs/code/mz_null_indexed_mvz_ready_group.sbatch
bash -n outputs/code/mz_null_indexed_mvz_groups.sbatch
EP_NULL_MVZ_READY_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-indexed-MVZ-ready-checks-${SLURM_ARRAY_JOB_ID}"
mkdir "$EP_NULL_MVZ_READY_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_MVZ_READY_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_indexed_rival.py -v \
  > "$EP_NULL_MVZ_READY_CHECKS/indexed_rival_tests.txt" 2>&1
printf 'NULL_INDEXED_MVZ_READY_TESTS_PASS\n'
EP_NULL_MVZ_READY_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_MVZ_READY_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null_indexed_mvz_ready_${SLURM_ARRAY_JOB_ID}.json"; then
  printf 'NULL_INDEXED_MVZ_READY_ASTRA_PASS\n'
else
  printf 'NULL_INDEXED_MVZ_READY_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
# Keep this allocation and its distinct startup log. At group8 the frozen
# implementation skips its group1 checks and group2 whole-baseline summary.
exec bash outputs/code/mz_null_indexed_mvz_groups.sbatch
