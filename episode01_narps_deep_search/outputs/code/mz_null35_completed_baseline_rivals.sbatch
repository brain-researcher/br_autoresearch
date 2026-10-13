#!/bin/bash -l
#SBATCH --job-name=ep01-null35-rivals
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:25:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-completed-baseline-rivals-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-completed-baseline-rivals-%j.err

# Exact completed null35 baseline46586603 only. Coordinator confirms the old
# source-rejected MC/MV and MVZ steps terminal before this new selection.
# One1CPU worker runs MC then MV; the other runs MVZ. This aggregates short
# components while fitting on two CPUs. Every selected component is attempted
# once, independently of the other's failure. No baseline refit or resimulation.
# Tests120+10 + ASTRA30+10 + max(2*(2400+10),3300+10) =4990<5100s.
set -euo pipefail
EP_NULL35_RIVAL_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s NTASKS=%s CPUS_PER_TASK=%s INDEX=35 SOURCE_COMPLETION=46586603\n' \
  "$EP_NULL35_RIVAL_HOST" "${SLURM_JOB_ID-}" "${SLURM_NTASKS-}" "${SLURM_CPUS_PER_TASK-}"
if [[ -z "${SLURM_JOB_ID-}" || "$EP_NULL35_RIVAL_HOST" == *-ln* || "${SLURM_CPUS_PER_TASK-}" != 1 ]]; then
  printf 'Requires an actual compute allocation and one CPU per task\n' >&2
  exit 2
fi
cd "$SLURM_SUBMIT_DIR"

if [[ "${1-}" == --worker ]]; then
  if (( $# != 2 )) || [[ "${SLURM_NTASKS-}" != 1 ]]; then
    printf 'Worker requires one Slurm task and explicit MC_MV or MVZ mode\n' >&2
    exit 2
  fi
  case "$2" in
    MC_MV) EP_NULL35_RIVAL_MODELS=(MC MV); EP_NULL35_RIVAL_SECONDS=2400 ;;
    MVZ) EP_NULL35_RIVAL_MODELS=(MVZ); EP_NULL35_RIVAL_SECONDS=3300 ;;
    *) printf 'Expected MC_MV or MVZ\n' >&2; exit 2 ;;
  esac
  EP_NULL35_RIVAL_WORKER_EXIT=0
  for EP_NULL35_RIVAL_MODEL in "${EP_NULL35_RIVAL_MODELS[@]}"; do
    if timeout --signal=TERM --kill-after=10s "$EP_NULL35_RIVAL_SECONDS" \
        "$EP_NULL35_RIVAL_PYTHON" -B outputs/code/run_mz_null_completed_baseline_rival.py \
        --model "$EP_NULL35_RIVAL_MODEL" --implementation outputs/code/behavioral_implementation.json \
        --output-dir "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-${EP_NULL35_RIVAL_MODEL}-completed-baseline-${SLURM_JOB_ID}" \
        --max-seconds "$EP_NULL35_RIVAL_SECONDS" \
        > "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-${EP_NULL35_RIVAL_MODEL}-completed-baseline-${SLURM_JOB_ID}.out" \
        2> "outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-${EP_NULL35_RIVAL_MODEL}-completed-baseline-${SLURM_JOB_ID}.err"; then
      printf 'NULL35_COMPONENT_COMPLETED model=%s\n' "$EP_NULL35_RIVAL_MODEL"
    else
      EP_NULL35_RIVAL_WORKER_EXIT=1
      printf 'NULL35_COMPONENT_FAILED model=%s: preserve attempt; other selected models remain independent\n' \
        "$EP_NULL35_RIVAL_MODEL" >&2
    fi
  done
  exit "$EP_NULL35_RIVAL_WORKER_EXIT"
fi

if (( $# != 0 )) || [[ "${SLURM_NTASKS-}" != 2 ]]; then
  printf 'Parent requires exactly two one-CPU Slurm tasks and no arguments\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
EP_NULL35_RIVAL_PYTHON=$(command -v python3.12)
export EP_NULL35_RIVAL_PYTHON
"$EP_NULL35_RIVAL_PYTHON" --version
bash -n outputs/code/mz_null35_completed_baseline_rivals.sbatch
EP_NULL35_RIVAL_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-completed-baseline-rival-checks-${SLURM_JOB_ID}"
mkdir "$EP_NULL35_RIVAL_CHECKS"
timeout --signal=TERM --kill-after=10s 120s "$EP_NULL35_RIVAL_PYTHON" -B \
  -m unittest discover -s outputs/code -p test_mz_null_completed_baseline_rival.py -v \
  > "$EP_NULL35_RIVAL_CHECKS/tests.txt" 2>&1
printf 'NULL35_COMPLETED_BASELINE_RIVAL_TESTS_PASS\n'
EP_NULL35_RIVAL_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if timeout --signal=TERM --kill-after=10s 30s "$EP_NULL35_RIVAL_ASTRA" validate \
    outputs/astra/v0.0.14/astra.yaml --json \
    > "outputs/astra/v0.0.14/validation_20261005_null35_completed_baseline_rivals_${SLURM_JOB_ID}.json"; then
  printf 'NULL35_COMPLETED_BASELINE_RIVAL_ASTRA_PASS\n'
else
  printf 'NULL35_COMPLETED_BASELINE_RIVAL_ASTRA_OPEN: coordinator must inspect validation output\n' >&2
fi
EP_NULL35_RIVAL_PIDS=()
for EP_NULL35_RIVAL_MODE in MC_MV MVZ; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exclusive --exact --mem=2G --cpu-bind=cores \
    --output="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-${EP_NULL35_RIVAL_MODE}-worker-${SLURM_JOB_ID}.out" \
    --error="outputs/runs/behavioral_mechanism_kill_test/20261005-M0-null35-${EP_NULL35_RIVAL_MODE}-worker-${SLURM_JOB_ID}.err" \
    bash outputs/code/mz_null35_completed_baseline_rivals.sbatch --worker "$EP_NULL35_RIVAL_MODE" &
  EP_NULL35_RIVAL_PIDS+=("$!")
done
EP_NULL35_RIVAL_EXIT=0
for EP_NULL35_RIVAL_PID in "${EP_NULL35_RIVAL_PIDS[@]}"; do
  if ! wait "$EP_NULL35_RIVAL_PID"; then EP_NULL35_RIVAL_EXIT=1; fi
done
exit "$EP_NULL35_RIVAL_EXIT"
