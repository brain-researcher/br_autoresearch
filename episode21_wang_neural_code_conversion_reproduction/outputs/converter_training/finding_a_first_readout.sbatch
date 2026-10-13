#!/bin/bash
# One original partial-roster readout; no retry, continuation or successor.
# All children share the actual parent's absolute clock and one 216-CPUh cap.
#SBATCH --job-name=ep21-finding-a-first
#SBATCH --partition=russpold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterok:47370327
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/finding_a_feature_decoding/first001/slurm-%j.out
#SBATCH --error=outputs/runs/finding_a_feature_decoding/first001/slurm-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_readout_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s GPUS=%s\n' "$ep21_readout_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}" "${SLURM_GPUS_ON_NODE:-0}"
[[ "$ep21_readout_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 3 \
  && "$SLURM_MEM_PER_NODE" == 16384 && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
ep21_readout_base=$PWD
ep21_readout_run="$ep21_readout_base/outputs/runs/finding_a_feature_decoding/first001"
[[ -d "$ep21_readout_run" ]] || exit 2
umask 077
mkdir "$ep21_readout_run/batch_claim" || exit 2
( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nmaximum_allocated_CPU_hours=216\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep21_readout_run/batch_status.txt" )
ep21_finish() {
  ep21_readout_exit=$?
  trap - EXIT
  printf 'batch_exit=%s\nstatus=closed\n' "$ep21_readout_exit" >> "$ep21_readout_run/batch_status.txt"
  exit "$ep21_readout_exit"
}
trap ep21_finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ulimit -c 0
# Bound child logs/fixtures without reducing the hard cap. The admitted
# empirical runner later sets its exact 40-GiB soft cap and default SIGXFSZ.
ulimit -S -f 32768
TZ=UTC timeout --signal=TERM --kill-after=5s 15s scontrol show job -o "$SLURM_JOB_ID" \
  > "$ep21_readout_run/allocation.txt"
ep21_readout_allocation=$(< "$ep21_readout_run/allocation.txt")
[[ "$ep21_readout_allocation" =~ StartTime=([^[:space:]]+) ]] || exit 2
ep21_readout_start=$(date -u -d "${BASH_REMATCH[1]}" +%s)
[[ "$ep21_readout_allocation" =~ EndTime=([^[:space:]]+) ]] || exit 2
ep21_readout_end=$(date -u -d "${BASH_REMATCH[1]}" +%s)
ep21_readout_deadline=$((ep21_readout_start + 255600))
if (( ep21_readout_end - 60 < ep21_readout_deadline )); then
  ep21_readout_deadline=$((ep21_readout_end - 60))
fi
printf 'absolute_deadline_epoch=%s\n' "$ep21_readout_deadline" >> "$ep21_readout_run/batch_status.txt"
ep21_limit() {
  local requested=$1 available
  available=$((ep21_readout_deadline - $(date -u +%s) - 10))
  (( available > 0 )) || return 124
  if (( available < requested )); then requested=$available; fi
  printf '%s\n' "$requested"
}

source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 py-pytorch/2.4.1_py312
# Preserve the module-provided PYTHONPATH from this point onward.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
[[ -d "${L_SCRATCH_JOB:?}" ]] || exit 2
export TMPDIR="$L_SCRATCH_JOB/ep21_finding_a_first001"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" TORCH_HOME="$TMPDIR/torch"
ep21_readout_entry="$ep21_readout_base/outputs/converter_training/tools/run_finding_a_readout.py"
ep21_readout_rc=0
ep21_readout_limit=$(ep21_limit 60)
if timeout --signal=TERM --kill-after=5s "${ep21_readout_limit}s" python3.12 -B \
    "$ep21_readout_entry" --check-admission > "$ep21_readout_run/admission.txt" 2>&1; then :
else ep21_readout_rc=$?; fi
printf 'admission_exit=%s\n' "$ep21_readout_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_readout_rc" -eq 0 ]] || exit "$ep21_readout_rc"

ep21_check_module() {
  local module_name=$1 duration=$2 limit
  limit=$(ep21_limit "$duration") || return $?
  (
    cd "$ep21_readout_base/outputs/converter_training/tests"
    timeout --signal=TERM --kill-after=5s "${limit}s" python3.12 -B - "$module_name" <<'PYTEST'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYTEST
  )
}
# Four distinct receipts are required by the empirical runner. Each runs once.
ep21_input_rc=0
if ep21_check_module test_finding_a_empirical_inputs 180 \
    > "$ep21_readout_run/empirical_input_checks.txt" 2>&1; then :; else ep21_input_rc=$?; fi
printf 'empirical_input_checks_exit=%s\n' "$ep21_input_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_input_rc" -ne 137 ]] || exit 137
ep21_composition_rc=0
if ep21_check_module test_finding_a_first_readout 180 \
    > "$ep21_readout_run/readout_composition_checks.txt" 2>&1; then :; else ep21_composition_rc=$?; fi
printf 'readout_composition_checks_exit=%s\n' "$ep21_composition_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_composition_rc" -ne 137 ]] || exit 137
ep21_batched_rc=0
if ep21_check_module test_finding_a_batched_inference 300 \
    > "$ep21_readout_run/batched_inference_checks.txt" 2>&1; then :; else ep21_batched_rc=$?; fi
printf 'batched_inference_checks_exit=%s\n' "$ep21_batched_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_batched_rc" -ne 137 ]] || exit 137
ep21_admission_checks_rc=0
if ep21_check_module test_finding_a_readout_admission 180 \
    > "$ep21_readout_run/readout_admission_checks.txt" 2>&1; then :; else ep21_admission_checks_rc=$?; fi
printf 'readout_admission_checks_exit=%s\n' "$ep21_admission_checks_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_admission_checks_rc" -ne 137 ]] || exit 137

ep21_readout_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_readout_astra_deps="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
ep21_astra_check() {
  local limit
  limit=$(ep21_limit 30) || return $?
  env PYTHONPATH="$ep21_readout_astra/lib/python3.12/site-packages:$ep21_readout_astra_deps" \
    timeout --signal=TERM --kill-after=5s "${limit}s" "$ep21_readout_astra/bin/astra-tools" \
    validate outputs/astra/v0.0.14/astra.yaml --json
}
ep21_astra_rc=0
if ep21_astra_check > "$ep21_readout_run/astra_validation.json" 2>&1; then :; else ep21_astra_rc=$?; fi
printf 'astra_exit=%s\n' "$ep21_astra_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_astra_rc" -ne 137 ]] || exit 137

# Independent EP19 software companion: no source payloads, fit or score.
# Preserve its result separately; ordinary check/ASTRA failure does not gate EP21.
ep19_companion_base="${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
ep19_companion_run="$ep19_companion_base/outputs/runs/reference_panel/finding_a_companion001"
ep19_checks_rc=not_run
ep19_astra_rc=not_run
if [[ -d "$ep19_companion_run" ]] && mkdir "$ep19_companion_run/batch_claim" && ( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nempirical_access=false\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep19_companion_run/status.txt" ); then
ep19_checks_rc=0
ep21_readout_limit=$(ep21_limit 115)
if (
  cd "$ep19_companion_base/outputs/tests"
  timeout --signal=TERM --kill-after=5s "${ep21_readout_limit}s" python3.12 -B - <<'PYEP19'
import sys
import torch
import unittest
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
suite = unittest.defaultTestLoader.loadTestsFromName('test_reference_final_refit')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PYEP19
) > "$ep19_companion_run/checks.txt" 2>&1; then :; else ep19_checks_rc=$?; fi
printf 'checks_exit=%s\n' "$ep19_checks_rc" >> "$ep19_companion_run/status.txt"
if [[ "$ep19_checks_rc" -eq 137 ]]; then
  printf 'astra_exit=not_run\nstatus=closed\n' >> "$ep19_companion_run/status.txt"
  exit 137
fi
ep19_astra_rc=0
if ( cd "$ep19_companion_base" && ep21_astra_check ) \
    > "$ep19_companion_run/astra_validation.json" 2>&1; then :; else ep19_astra_rc=$?; fi
printf 'astra_exit=%s\nstatus=closed\n' "$ep19_astra_rc" >> "$ep19_companion_run/status.txt"
[[ "$ep19_astra_rc" -ne 137 ]] || exit 137
else
  printf 'ep19_companion_setup=unavailable\n' >> "$ep21_readout_run/batch_status.txt"
fi
printf 'ep19_checks_exit=%s\nep19_astra_exit=%s\n' "$ep19_checks_rc" "$ep19_astra_rc" \
  >> "$ep21_readout_run/batch_status.txt"

if (( ep21_input_rc != 0 || ep21_composition_rc != 0 || ep21_batched_rc != 0 \
      || ep21_admission_checks_rc != 0 || ep21_astra_rc != 0 )); then
  printf 'readout_exit=not_run\n' >> "$ep21_readout_run/batch_status.txt"
  exit 2
fi
ep21_readout_limit=$(ep21_limit 60)
ep21_remaining_rc=0
if ep21_runner_remaining=$(timeout --signal=TERM --kill-after=5s "${ep21_readout_limit}s" \
    python3.12 -B "$ep21_readout_entry" --remaining-seconds \
    2> "$ep21_readout_run/remaining_stderr.txt"); then :; else ep21_remaining_rc=$?; fi
printf 'remaining_admission_exit=%s\n' "$ep21_remaining_rc" >> "$ep21_readout_run/batch_status.txt"
[[ "$ep21_remaining_rc" -eq 0 && "$ep21_runner_remaining" =~ ^[0-9]+$ \
  && "$ep21_runner_remaining" -gt 0 ]] || exit 2
ep21_readout_limit=$(ep21_limit "$ep21_runner_remaining")
printf 'readout_timeout_seconds=%s\n' "$ep21_readout_limit" >> "$ep21_readout_run/batch_status.txt"
ep21_readout_rc=0
if timeout --signal=TERM --kill-after=5s "${ep21_readout_limit}s" python3.12 -u -B \
    "$ep21_readout_entry" > "$ep21_readout_run/execution.txt" 2>&1; then :
else ep21_readout_rc=$?; fi
printf 'readout_exit=%s\n' "$ep21_readout_rc" >> "$ep21_readout_run/batch_status.txt"
exit "$ep21_readout_rc"
