#!/bin/bash -l
#SBATCH --job-name=ep01-mz-other-partitions
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-other-partitions-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-other-partitions-%j.err

# Two finite sensitivity workers. Tests130 + ASTRA40 + parallel workers5410
# =5580<6000 seconds. No scheduler polling, automatic retry or new simulation.
set -euo pipefail
EP_MZ_PARTITION_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS=%s\n' "$EP_MZ_PARTITION_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MZ_PARTITION_HOST" == *-ln* || "${SLURM_NTASKS-}" != 2 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MZ_PARTITION_PYTHON=$(command -v python3.12)
"$EP_MZ_PARTITION_PYTHON" --version
bash -n outputs/code/mz_other_partitions.sbatch
EP_MZ_PARTITION_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-other-partitions-checks-${SLURM_JOB_ID}"
mkdir "$EP_MZ_PARTITION_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_MZ_PARTITION_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_other_partitions.py -v \
  > "$EP_MZ_PARTITION_CHECKS/tests.txt" 2>&1
printf 'MZ_OTHER_PARTITIONS_TESTS_PASS\n'
EP_MZ_PARTITION_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_MZ_PARTITION_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_MZ_PARTITION_CHECKS/astra_validation.json"; then
  printf 'MZ_OTHER_PARTITIONS_ASTRA_PASS\n'
else
  printf 'MZ_OTHER_PARTITIONS_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi

EP_MZ_PARTITION_PIDS=()
EP_MZ_PARTITION_NAMES=()
for EP_MZ_PARTITION in 12_34 13_24; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
    --output="$EP_MZ_PARTITION_CHECKS/${EP_MZ_PARTITION}.out" --error="$EP_MZ_PARTITION_CHECKS/${EP_MZ_PARTITION}.err" \
    timeout --signal=TERM --kill-after=10s 5400s "$EP_MZ_PARTITION_PYTHON" -B \
    outputs/code/run_mz_other_partitions.py --partition "$EP_MZ_PARTITION" \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-other-partitions-${SLURM_JOB_ID}/${EP_MZ_PARTITION}" \
    --max-seconds 5400 &
  EP_MZ_PARTITION_PIDS+=("$!")
  EP_MZ_PARTITION_NAMES+=("$EP_MZ_PARTITION")
done
EP_MZ_PARTITION_EXIT=0
for EP_MZ_PARTITION_SLOT in "${!EP_MZ_PARTITION_PIDS[@]}"; do
  if wait "${EP_MZ_PARTITION_PIDS[$EP_MZ_PARTITION_SLOT]}"; then
    printf 'MZ_OTHER_PARTITION_COMPLETED partition=%s\n' "${EP_MZ_PARTITION_NAMES[$EP_MZ_PARTITION_SLOT]}"
  else
    EP_MZ_PARTITION_EXIT=1
    printf 'MZ_OTHER_PARTITION_FAILED partition=%s: preserve attempt; no retry\n' "${EP_MZ_PARTITION_NAMES[$EP_MZ_PARTITION_SLOT]}" >&2
  fi
done
exit "$EP_MZ_PARTITION_EXIT"
