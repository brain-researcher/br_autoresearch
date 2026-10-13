#!/bin/bash -l
#SBATCH --job-name=ep01-mz-effect-levels
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-effect-levels-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-effect-levels-%j.err

# Two prospectively selected representative0 cells; no wave, polling or retry.
# Tests130 + ASTRA40 + two parallel exclusive workers5410 =5580<6000 seconds.
set -euo pipefail
EP_MZ_LEVEL_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS=%s\n' "$EP_MZ_LEVEL_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_MZ_LEVEL_HOST" == *-ln* || "${SLURM_NTASKS-}" != 2 || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_MZ_LEVEL_PYTHON=$(command -v python3.12)
"$EP_MZ_LEVEL_PYTHON" --version
bash -n outputs/code/mz_effect_level_representatives.sbatch
EP_MZ_LEVEL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-effect-level-checks-${SLURM_JOB_ID}"
mkdir "$EP_MZ_LEVEL_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_MZ_LEVEL_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_effect_level_recovery.py -v \
  > "$EP_MZ_LEVEL_CHECKS/tests.txt" 2>&1
printf 'MZ_EFFECT_LEVEL_TESTS_PASS\n'
EP_MZ_LEVEL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_MZ_LEVEL_ASTRA" validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_MZ_LEVEL_CHECKS/astra_validation.json"; then
  printf 'MZ_EFFECT_LEVEL_ASTRA_PASS\n'
else
  printf 'MZ_EFFECT_LEVEL_ASTRA_OPEN: coordinator must inspect retained output\n' >&2
fi

EP_MZ_LEVEL_PIDS=()
EP_MZ_LEVEL_NAMES=()
for EP_MZ_EFFECT_LEVEL in half twice; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
    --output="$EP_MZ_LEVEL_CHECKS/${EP_MZ_EFFECT_LEVEL}.out" --error="$EP_MZ_LEVEL_CHECKS/${EP_MZ_EFFECT_LEVEL}.err" \
    timeout --signal=TERM --kill-after=10s 5400s "$EP_MZ_LEVEL_PYTHON" -B \
    outputs/code/run_mz_effect_level_recovery.py --level "$EP_MZ_EFFECT_LEVEL" --replicate-index 0 \
    --design-events outputs/runs/behavioral_mechanism_kill_test/20261003-MC-SESOI-replicate000-attempt003/simulated_events.csv.gz \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-effect-level-representatives-${SLURM_JOB_ID}/${EP_MZ_EFFECT_LEVEL}/replicate0" \
    --max-seconds 5400 &
  EP_MZ_LEVEL_PIDS+=("$!")
  EP_MZ_LEVEL_NAMES+=("$EP_MZ_EFFECT_LEVEL")
done
EP_MZ_LEVEL_EXIT=0
for EP_MZ_LEVEL_SLOT in "${!EP_MZ_LEVEL_PIDS[@]}"; do
  if wait "${EP_MZ_LEVEL_PIDS[$EP_MZ_LEVEL_SLOT]}"; then
    printf 'MZ_EFFECT_LEVEL_COMPLETED level=%s index=0\n' "${EP_MZ_LEVEL_NAMES[$EP_MZ_LEVEL_SLOT]}"
  else
    EP_MZ_LEVEL_EXIT=1
    printf 'MZ_EFFECT_LEVEL_FAILED level=%s: preserve attempt; no retry\n' "${EP_MZ_LEVEL_NAMES[$EP_MZ_LEVEL_SLOT]}" >&2
  fi
done
exit "$EP_MZ_LEVEL_EXIT"
