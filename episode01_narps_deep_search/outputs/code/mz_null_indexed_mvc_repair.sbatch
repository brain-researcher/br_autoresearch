#!/bin/bash -l
#SBATCH --job-name=ep01-null-mvc-repair
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-repair-%A_%a.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-repair-%A_%a.err

# Initial43 selected;61/168 await its checks/replay. Explicit43/61/168 only.
# Each has diagnosed first-half1000
# exhaustion in their terminal46592261 tasks. Both halves must be fresh; no completed
# first fit exists. Preserve original500 baselines and original failed MVC.
# Changed adapter tests and ASTRA share the substantive allocation.
# Maximum130+40+12610=12780s<13200s including timeout graces.
set -euo pipefail
EP_NULL_MVC_REPAIR_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s ARRAY=%s INDEX=%s CPUS=%s\n' \
  "$EP_NULL_MVC_REPAIR_HOST" "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
  "${SLURM_ARRAY_TASK_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL_MVC_REPAIR_HOST" == *-ln* || \
      -z "${SLURM_ARRAY_JOB_ID-}" || \
      "${SLURM_NTASKS-}" != 1 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
case "${SLURM_ARRAY_TASK_ID-}" in 43|61|168) ;; *) exit 2 ;; esac
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL_MVC_REPAIR_PYTHON=$(command -v python3.12)
"$EP_NULL_MVC_REPAIR_PYTHON" --version
bash -n outputs/code/mz_null_indexed_mvc_repair.sbatch
if [[ "$SLURM_ARRAY_TASK_ID" == 43 ]]; then
EP_NULL_MVC_REPAIR_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-repair-checks-${SLURM_ARRAY_JOB_ID}"
mkdir "$EP_NULL_MVC_REPAIR_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL_MVC_REPAIR_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_indexed_mvc_repair.py -v \
  > "$EP_NULL_MVC_REPAIR_CHECKS/tests.txt" 2>&1
printf 'NULL_INDEXED_MVC_REPAIR_TESTS_PASS\n'
EP_NULL_MVC_REPAIR_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL_MVC_REPAIR_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null43_mvc_repair_${SLURM_ARRAY_JOB_ID}.json"; then
  printf 'NULL_INDEXED_MVC_REPAIR_ASTRA_PASS\n'
else
  printf 'NULL_INDEXED_MVC_REPAIR_ASTRA_OPEN: coordinator must inspect output\n' >&2
fi
fi
exec timeout --signal=TERM --kill-after=10s 12600s "$EP_NULL_MVC_REPAIR_PYTHON" -B \
  outputs/code/run_mz_null_indexed_mvc_repair.py --replicate-index "$SLURM_ARRAY_TASK_ID" \
  --implementation outputs/code/behavioral_implementation.json \
  --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null-MVC-repair-array-${SLURM_ARRAY_JOB_ID}/replicate${SLURM_ARRAY_TASK_ID}" \
  --max-seconds 12600
