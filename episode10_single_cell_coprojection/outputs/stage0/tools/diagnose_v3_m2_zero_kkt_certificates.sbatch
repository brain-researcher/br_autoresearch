#!/usr/bin/env bash
#SBATCH --job-name=ep10-m2-kkt-diag
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
runtime_log="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection/m2_zero_kkt_certificate_diagnostic_${SLURM_JOB_ID}.log"

test -x "$python_bin"
test -f "$project/outputs/stage0/tests/test_fit_v3_m2.py"
test ! -e "$runtime_log"

exec >>"$runtime_log" 2>&1
printf 'EP10 v3 M2 zero-KKT certificate diagnostic log: %s\n' "$runtime_log"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$project"
set +e
"$python_bin" -m unittest \
  outputs.stage0.tests.test_fit_v3_m2.M2CoreTests.test_tangential_midpoints_have_separation_certificates
tangential_status=$?
"$python_bin" -m unittest \
  outputs.stage0.tests.test_fit_v3_m2.M2CoreTests.test_deterministic_planted_probability_recovery_all_penalties
recovery_status=$?
set -e

if [[ "$tangential_status" -ne 1 || "$recovery_status" -ne 1 ]]; then
  printf 'ERROR: expected both unchanged reproducers to fail closed; statuses=%s,%s\n' \
    "$tangential_status" "$recovery_status"
  exit 2
fi

diagnostic_count="$(grep -c 'minimax_gradient_infinity=' "$runtime_log")"
if [[ "$diagnostic_count" -ne 3 ]]; then
  printf 'ERROR: expected 3 certificate diagnostic records, observed %s\n' \
    "$diagnostic_count"
  exit 2
fi

printf 'EP10 v3 M2 zero-KKT certificate diagnostic: PASS (%s records)\n' \
  "$diagnostic_count"
