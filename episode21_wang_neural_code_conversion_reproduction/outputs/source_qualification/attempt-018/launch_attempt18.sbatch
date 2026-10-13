#!/bin/bash
#SBATCH --job-name=ep21-bvlc-alexnet
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --chdir=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-018/slurm-%j.out

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
ep21_runner="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/tools/audit_bvlc_alexnet_paper_asset.py"
ep21_expected_runner_sha="2998cf99d3817eb2643e128ac3e45a84faff1ba791c4d2a27be965ffaf8cac7e"
ep21_output="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/source_qualification/attempt-018/bvlc_alexnet_paper_asset_audit.json"

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

exec "$ep21_python_bin" -I -B "$ep21_runner" --attempt 18
