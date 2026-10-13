#!/bin/bash -l
#SBATCH --job-name=ep01-gain-mz-urgency-trace
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-gain-reverse-and-trace-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-gain-reverse-and-trace-%j.err

# Independent4CPU common-gain repair and1CPU unchanged urgency-start trace.
# New checks190+70, ASTRA40, concurrent max3610, pool130 =4040<4500s.
set -euo pipefail
EP_MZ_BUNDLE_HOST=$(hostname)
printf 'HOST=%s SLURM_JOB_ID=%s CPUS_PER_TASK=%s\n' \
  "$EP_MZ_BUNDLE_HOST" "${SLURM_JOB_ID-}" "${SLURM_CPUS_PER_TASK-}"
if [[ $# != 0 || "$EP_MZ_BUNDLE_HOST" == *-ln* || -z "${SLURM_JOB_ID-}" \
      || "${SLURM_CPUS_PER_TASK-}" != 5 ]]; then exit 2; fi
cd "$SLURM_SUBMIT_DIR"
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="outputs/code:outputs/tests${PYTHONPATH:+:$PYTHONPATH}"
EP_MZ_BUNDLE_PYTHON=$(command -v python3.12)
"$EP_MZ_BUNDLE_PYTHON" --version
bash -n outputs/code/mz_common_gain_mz_reverse_and_trace.sbatch
EP_MZ_BUNDLE_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-MZ-reverse-repair-${SLURM_JOB_ID}"
EP_MZ_TRACE_ROOT="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimizer-trace-${SLURM_JOB_ID}"
EP_MZ_BUNDLE_CHECKS="outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-common-gain-MZ-reverse-checks-${SLURM_JOB_ID}"
mkdir "$EP_MZ_BUNDLE_ROOT" "$EP_MZ_BUNDLE_CHECKS"
mkdir "$EP_MZ_BUNDLE_ROOT/dir1"
printf 'stage\texit_status\n' > "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"

if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 180s "$EP_MZ_BUNDLE_PYTHON" -B -m unittest -v \
    test_mz_common_gain_mz_reverse_repair test_mz_common_gain_mz_reverse_pool \
    > "$EP_MZ_BUNDLE_CHECKS/common_gain_reverse_tests.txt" 2>&1; then
  EP_MZ_REPAIR_CHECK_STATUS=0
else
  EP_MZ_REPAIR_CHECK_STATUS=$?
fi
printf 'common_gain_reverse_tests\t%s\n' "$EP_MZ_REPAIR_CHECK_STATUS" \
  >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"

if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 60s "$EP_MZ_BUNDLE_PYTHON" -B -m unittest -v \
    test_mz_urgency_optimizer_trace > "$EP_MZ_BUNDLE_CHECKS/urgency_trace_tests.txt" 2>&1; then
  EP_MZ_TRACE_CHECK_STATUS=0
else
  EP_MZ_TRACE_CHECK_STATUS=$?
fi
printf 'urgency_trace_tests\t%s\n' "$EP_MZ_TRACE_CHECK_STATUS" \
  >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"

EP_MZ_BUNDLE_ASTRA="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 30s "$EP_MZ_BUNDLE_ASTRA" \
    validate outputs/astra/v0.0.14/astra.yaml --json \
    > "$EP_MZ_BUNDLE_CHECKS/astra_validation.json" 2> "$EP_MZ_BUNDLE_CHECKS/astra_validation.err"; then
  EP_MZ_ASTRA_STATUS=0
else
  EP_MZ_ASTRA_STATUS=$?
fi
printf 'astra_validation\t%s\n' "$EP_MZ_ASTRA_STATUS" >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"

EP_MZ_REPAIR_PID=""
EP_MZ_TRACE_PID=""
if [[ "$EP_MZ_REPAIR_CHECK_STATUS" == 0 ]]; then
  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 3600s "$EP_MZ_BUNDLE_PYTHON" -B \
    outputs/code/run_mz_common_gain_mz_reverse_repair.py \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_MZ_BUNDLE_ROOT/dir1/MZ" --max-seconds 3600 \
    > "$EP_MZ_BUNDLE_ROOT/dir1/MZ.out" 2> "$EP_MZ_BUNDLE_ROOT/dir1/MZ.err" &
  EP_MZ_REPAIR_PID=$!
else
  printf 'COMMON_GAIN_REPAIR_SKIPPED: focused checks failed\n' >&2
fi
if [[ "$EP_MZ_TRACE_CHECK_STATUS" == 0 ]]; then
  srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
    timeout --signal=TERM --kill-after=10s 1200s "$EP_MZ_BUNDLE_PYTHON" -B \
    outputs/code/run_mz_urgency_optimizer_trace.py \
    --source-dir outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001 \
    --implementation outputs/code/behavioral_implementation.json \
    --output-dir "$EP_MZ_TRACE_ROOT" --max-seconds 1200 \
    > "$EP_MZ_BUNDLE_CHECKS/urgency_trace_stdout.json" 2> "$EP_MZ_BUNDLE_CHECKS/urgency_trace_stderr.txt" &
  EP_MZ_TRACE_PID=$!
else
  printf 'URGENCY_TRACE_SKIPPED: focused checks failed\n' >&2
fi

EP_MZ_REPAIR_STATUS=$EP_MZ_REPAIR_CHECK_STATUS
EP_MZ_TRACE_STATUS=$EP_MZ_TRACE_CHECK_STATUS
EP_MZ_POOL_STATUS=1
if [[ -n "$EP_MZ_REPAIR_PID" ]]; then
  if wait "$EP_MZ_REPAIR_PID"; then EP_MZ_REPAIR_STATUS=0; else EP_MZ_REPAIR_STATUS=$?; fi
  printf 'common_gain_reverse_repair\t%s\n' "$EP_MZ_REPAIR_STATUS" \
    >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"
  if [[ "$EP_MZ_REPAIR_STATUS" == 0 ]]; then
    if srun --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --exclusive --exact --cpu-bind=cores \
        timeout --signal=TERM --kill-after=10s 120s "$EP_MZ_BUNDLE_PYTHON" -B \
        outputs/code/summarize_mz_common_gain_mz_reverse_repaired.py \
        --repair-dir "$EP_MZ_BUNDLE_ROOT/dir1/MZ" --repair-job-id "$SLURM_JOB_ID" \
        --implementation outputs/code/behavioral_implementation.json \
        --output "$EP_MZ_BUNDLE_ROOT/common_gain_mz_reverse_repaired_summary.json" \
        > "$EP_MZ_BUNDLE_ROOT/pool.out" 2> "$EP_MZ_BUNDLE_ROOT/pool.err"; then
      EP_MZ_POOL_STATUS=0
    else
      EP_MZ_POOL_STATUS=$?
    fi
    printf 'common_gain_reverse_pool\t%s\n' "$EP_MZ_POOL_STATUS" \
      >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"
  fi
fi
if [[ -n "$EP_MZ_TRACE_PID" ]]; then
  if wait "$EP_MZ_TRACE_PID"; then EP_MZ_TRACE_STATUS=0; else EP_MZ_TRACE_STATUS=$?; fi
  printf 'urgency_optimizer_trace\t%s\n' "$EP_MZ_TRACE_STATUS" \
    >> "$EP_MZ_BUNDLE_CHECKS/stage_exit_status.tsv"
fi
if [[ "$EP_MZ_REPAIR_STATUS" != 0 || "$EP_MZ_TRACE_STATUS" != 0 \
      || "$EP_MZ_POOL_STATUS" != 0 || "$EP_MZ_ASTRA_STATUS" != 0 ]]; then exit 1; fi
