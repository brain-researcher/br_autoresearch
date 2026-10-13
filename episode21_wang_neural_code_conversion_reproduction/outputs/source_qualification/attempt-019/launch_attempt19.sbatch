#!/bin/bash
#SBATCH --job-name=ep21-method-audit
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-019/slurm-%j.out

set -euo pipefail
umask 077
ulimit -c 0

export LC_ALL=C
export LANG=C
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
unset PYTHONHOME PYTHONPATH

module purge
module load python/3.12.1

ep21_python_bin="$(command -v python3.12)"
ep21_expected_python="/share/software/user/open/python/3.12.1/bin/python3.12"
ep21_runner="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/tools/audit_public_method_semantics.py"
ep21_expected_runner_sha="122f44866ba95a6eddf24a79d7a093dd4d748ed82b5f9fe09dec443aadc5ee46"
ep21_output="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-019/public_method_semantics_audit.json"

if [[ "$ep21_python_bin" != "$ep21_expected_python" ]]; then
  printf 'unexpected_python=%s\n' "$ep21_python_bin" >&2
  exit 64
fi
ep21_runner_identity="$(/usr/bin/sha256sum "$ep21_runner")"
if [[ "$ep21_runner_identity" != "$ep21_expected_runner_sha  $ep21_runner" ]]; then
  printf 'runner_identity_mismatch=%s\n' "$ep21_runner_identity" >&2
  exit 65
fi
if [[ -e "$ep21_output" || -L "$ep21_output" ]]; then
  printf 'refusing_existing_output=%s\n' "$ep21_output" >&2
  exit 66
fi

printf 'python_path=%s\n' "$ep21_python_bin"
"$ep21_python_bin" --version
printf 'runner_sha256=%s\n' "$ep21_expected_runner_sha"

exec "$ep21_python_bin" -I -B "$ep21_runner" --attempt 19
