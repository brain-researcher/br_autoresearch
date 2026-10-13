#!/bin/bash

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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  printf 'refusing_active_slurm_allocation=%s\n' "$SLURM_JOB_ID" >&2
  exit 64
fi

ep21_hostname="$(hostname)"
if [[ "${ep21_hostname%%.*}" != *-ln* ]]; then
  printf 'refusing_non_login_hostname=%s\n' "$ep21_hostname" >&2
  exit 65
fi

module purge
module load python/3.12.1

ep21_python_bin="$(command -v python3.12)"
ep21_expected_python="/share/software/user/open/python/3.12.1/bin/python3.12"
ep21_runner="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/pipeline_conformance/tools/run_adapter_conformance.py"
ep21_expected_runner_sha="bd3cc6a9f3ca07aefc9c5932e6ab3f644c8d1a4de3c14f541ff7d2f7c98c910f"
ep21_adapter="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/pipeline_conformance/tools/execution_adapter.py"
ep21_expected_adapter_sha="2e269d5fe53933523871a613e84ee621c6058b05a769085c943166cae70b8ed8"
ep21_output_dir="/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/pipeline_conformance/attempt-006"
ep21_output="$ep21_output_dir/adapter_conformance_report.json"

if [[ "$ep21_python_bin" != "$ep21_expected_python" ]]; then
  printf 'unexpected_python=%s\n' "$ep21_python_bin" >&2
  exit 66
fi
ep21_runner_identity="$(/usr/bin/sha256sum "$ep21_runner")"
if [[ "$ep21_runner_identity" != "$ep21_expected_runner_sha  $ep21_runner" ]]; then
  printf 'runner_identity_mismatch=%s\n' "$ep21_runner_identity" >&2
  exit 67
fi
ep21_adapter_identity="$(/usr/bin/sha256sum "$ep21_adapter")"
if [[ "$ep21_adapter_identity" != "$ep21_expected_adapter_sha  $ep21_adapter" ]]; then
  printf 'adapter_identity_mismatch=%s\n' "$ep21_adapter_identity" >&2
  exit 68
fi
if [[ -e "$ep21_output" || -L "$ep21_output" ]]; then
  printf 'refusing_existing_output=%s\n' "$ep21_output" >&2
  exit 69
fi
shopt -s nullglob
ep21_partials=("$ep21_output_dir"/.adapter_conformance_report.json.*.partial)
if (( ${#ep21_partials[@]} != 0 )); then
  printf 'refusing_existing_partial_count=%d\n' "${#ep21_partials[@]}" >&2
  exit 70
fi

printf 'execution_context=attended-login\n'
printf 'hostname=%s\n' "$ep21_hostname"
printf 'python_path=%s\n' "$ep21_python_bin"
"$ep21_python_bin" --version
printf 'runner_sha256=%s\n' "$ep21_expected_runner_sha"
printf 'adapter_sha256=%s\n' "$ep21_expected_adapter_sha"

exec "$ep21_python_bin" -I -B "$ep21_runner" \
  --attempt 6 \
  --execution-context attended-login \
  --max-workers 1
