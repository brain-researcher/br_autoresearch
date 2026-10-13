#!/bin/bash
# Prospective first complete direct-surface panel; submission is recorded separately.
# One finite allocation, no requeue, successors or automated resubmission.
#SBATCH --job-name=ep20-dev-worlds
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/development_worlds001-%j.out
#SBATCH --error=outputs/runs/development_worlds001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep20_world_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep20_world_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep20_world_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
[[ -d "$TMPDIR" ]]
ep20_world_run="${PWD}/outputs/runs/development_worlds_job_${SLURM_JOB_ID}"
mkdir "$ep20_world_run"
printf 'job=%s\nparent_charged_once_to=EP20\nreference_fit=false\ncandidate_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep20_world_run/status.txt"
ep20_world_check_storage_log() {
  # Preserve the existing stop-on-storage-fault rule across optional children.
  LC_ALL=C grep -E -i -q -- '^[[:space:]]*(OSError|IOError|BrokenPipeError):[[:space:]]+\[Errno[[:space:]]+(5|28|108|122)\]|^[^{}]*:[[:space:]]*(Input/output error|No space left on device|Disk quota exceeded|Cannot send after transport endpoint shutdown)[[:space:]]*$' "$1"
  case $? in 1) return 0 ;; *) return 74 ;; esac
}

# New operator/integration checks only; earlier passing primitive suites are reused.
set +e
(
  cd outputs/tools || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v -f \
    test_factorized_field_power_scales test_development_focal_worlds test_development_worlds
) > "$ep20_world_run/tests.txt" 2>&1
ep20_world_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep20_world_checks_exit" >> "$ep20_world_run/status.txt"
[[ "$ep20_world_checks_exit" == 0 ]]

# Independent EP19 integration checks share this finite parent and its resource cap.
ep19_ensemble_root="${OAK}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
ep19_ensemble_run="$ep19_ensemble_root/outputs/runs/ensemble_checks_job_${SLURM_JOB_ID}"
mkdir "$ep19_ensemble_run"
printf 'job=%s\nparent_charged_once_to=EP20\nempirical_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep19_ensemble_run/status.txt"
set +e
(
  cd "$ep19_ensemble_root/outputs/tests" || exit 2
  timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v -f test_ensemble_forecast_backend
) > "$ep19_ensemble_run/tests.txt" 2>&1
ep19_ensemble_checks_exit=$?
set -e
ep20_world_check_storage_log "$ep19_ensemble_run/tests.txt" || exit 74
printf 'tests_exit=%s\n' "$ep19_ensemble_checks_exit" >> "$ep19_ensemble_run/status.txt"
printf 'optional_ep19_checks_exit=%s\n' "$ep19_ensemble_checks_exit" >> "$ep20_world_run/status.txt"

# Program validation is recorded; it does not certify a scientific result.
ep20_world_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_world_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
set +e
PYTHONPATH="$ep20_world_astra/lib/python3.12/site-packages:$ep20_world_dependencies" \
  timeout --signal=TERM --kill-after=5s 30s "$ep20_world_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep20_world_run/astra_validation.json" 2>&1
ep20_world_astra_exit=$?
(
  cd "$ep19_ensemble_root" || exit 2
  PYTHONPATH="$ep20_world_astra/lib/python3.12/site-packages:$ep20_world_dependencies" \
    timeout --signal=TERM --kill-after=5s 30s "$ep20_world_astra/bin/astra-tools" validate \
    outputs/astra/v0.0.14/astra.yaml --json
) > "$ep19_ensemble_run/astra_validation.json" 2>&1
ep19_ensemble_astra_exit=$?
set -e
ep20_world_check_storage_log "$ep20_world_run/astra_validation.json" || exit 74
ep20_world_check_storage_log "$ep19_ensemble_run/astra_validation.json" || exit 74
printf 'astra_exit=%s\n' "$ep19_ensemble_astra_exit" >> "$ep19_ensemble_run/status.txt"
printf 'astra_exit=%s\noptional_ep19_astra_exit=%s\n' \
  "$ep20_world_astra_exit" "$ep19_ensemble_astra_exit" >> "$ep20_world_run/status.txt"

# The 6,000-second scientific deadline leaves setup, checks and shutdown inside 2h.
# The driver retains complete groups and lists unresolved/remaining world identities.
set +e
timeout --signal=TERM --kill-after=5s 6040s python3.12 -B outputs/tools/development_worlds.py \
  --output-root "$ep20_world_run/panel" --max-seconds 6000 \
  > "$ep20_world_run/generation.txt" 2>&1
ep20_world_generation_exit=$?
set -e
printf 'generation_exit=%s\n' "$ep20_world_generation_exit" >> "$ep20_world_run/status.txt"
[[ "$ep20_world_generation_exit" == 0 ]] || exit "$ep20_world_generation_exit"

# Resume the already-adopted EP17 private pilot using fresh capacity evidence.
# This writes no neural values to the controller logs and opens no reader handoff.
# Capacity was assessed by root at 07:15:26 UTC from the current official quota
# and exact-parent df observations collected earlier in this same turn.
ep17_pilot_root="${OAK}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
ep17_pilot_capacity="$ep17_pilot_root/outputs/runs/role_materialization/capacity_observation001.json"
mkdir -p "$ep17_pilot_root/outputs/runs/role_materialization"
(
  umask 077
  set -o noclobber
  cat > "$ep17_pilot_capacity" <<EP17_CAPACITY
{
  "parent": "$ep17_pilot_root/outputs/cache",
  "slurm_job_id": "$SLURM_JOB_ID",
  "observed_at_utc": "2026-10-09T07:15:26Z",
  "quota_checked": true,
  "filesystem_available_bytes": 86288309837824,
  "quota_available_bytes": 77000000000000,
  "concurrent_reserved_bytes": 560000000000,
  "source_note": "Fresh official sh_quota without cached-data warning and exact-parent df, assessed by root; no directory scan."
}
EP17_CAPACITY
)
set +e
(
  umask 077
  ulimit -v 2097152
  cd "$ep17_pilot_root" || exit 2
  exec timeout --signal=TERM --kill-after=10s 180s python3.12 -B \
    outputs/tools/materialize_neural_roles.py --mode pilot \
    --capacity-observation outputs/runs/role_materialization/capacity_observation001.json
) > "$ep20_world_run/ep17_private_pilot.txt" 2>&1
ep17_pilot_exit=$?
set -e
ep20_world_check_storage_log "$ep20_world_run/ep17_private_pilot.txt" || exit 74
printf 'optional_ep17_pilot_exit=%s\n' "$ep17_pilot_exit" >> "$ep20_world_run/status.txt"
# The privacy-preserving pilot intentionally suppresses error details. Any
# nonzero result therefore stops the parent; no later scientific I/O follows.
exit "$ep17_pilot_exit"
