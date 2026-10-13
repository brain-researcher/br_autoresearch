#!/bin/bash
# Coordinator supplies a freshly discovered eligible partition at submission.
#SBATCH --job-name=ep18-public-provision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/feature_preparation/provision_exact_cpu_001-%j.out
#SBATCH --error=outputs/runs/feature_preparation/provision_exact_cpu_001-%j.err
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* && "${SLURM_CPUS_PER_TASK:-}" == 1 ]]
[[ "${SLURM_MEM_PER_NODE:-}" == 7168 && -z "${SLURM_JOB_GPUS:-}" ]]
: "${GROUP_HOME:?}" "${OAK:?}" "${L_SCRATCH_JOB:?}"
cd "$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export EP18_PROVISION_RUN="$PWD/outputs/runs/feature_preparation/exact_cpu_provision_${SLURM_JOB_ID}"
mkdir "$EP18_PROVISION_RUN"
export TMPDIR="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/tmp"
export PIP_CACHE_DIR="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/pip"
export XDG_CACHE_HOME="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/cache"
export HF_HOME="$L_SCRATCH_JOB/ep18-provision-${SLURM_JOB_ID}/hf"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME"
ep18_phase=initialization
ep18_started=$SECONDS
finish() {
  ep18_exit=$?
  trap - EXIT
  printf '{"phase":"%s","exit_code":%s,"elapsed_seconds":%s,"job_id":"%s","allocated_cpus":1,"memory_mib":7168,"maximum_seconds":7200,"automatic_retry":false}\n' \
    "$ep18_phase" "$ep18_exit" "$((SECONDS-ep18_started))" "$SLURM_JOB_ID" > "$EP18_PROVISION_RUN/provision_exit.json"
  exit "$ep18_exit"
}
trap finish EXIT
printf 'hostname=%s job_id=%s cpus=%s memory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE"
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
# Brief checks from separately scoped episodes share this substantive parent
# allocation serially. A sibling failure is recorded and does not stop EP18.
ep18_phase=shared_source_preparation
module load math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312
set +e
(
  cd "$OAK/users/zijiao/br_autoresearch/episode09_local_global_projection" || exit 2
  timeout --signal=TERM --kill-after=2s 10s python3.12 -B \
    outputs/tools/reconcile_brain236174_provider_alias.py \
    > "outputs/runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_${SLURM_JOB_ID}.json" \
    2> "outputs/runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_${SLURM_JOB_ID}.err"
)
printf 'EP09_provider_alias_check_exit=%s\n' "$?" | tee "$EP18_PROVISION_RUN/shared_checks.txt"
(
  cd "$OAK/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign" || exit 2
  ep18_ep20_checks="outputs/runs/coupled_rc_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep18_ep20_checks" || exit 2
  timeout --signal=TERM --kill-after=2s 60s python3.12 -B outputs/tools/test_coupled_rc.py -v \
    > "$ep18_ep20_checks/tests.txt" 2>&1
  ep18_ep20_check_exit=$?
  printf 'job=%s\nchecks_exit=%s\nparent_charged_once_to=EP18\n' \
    "$SLURM_JOB_ID" "$ep18_ep20_check_exit" > "$ep18_ep20_checks/status.txt"
  exit "$ep18_ep20_check_exit"
)
printf 'EP20_coupled_rc_check_exit=%s\n' "$?" | tee -a "$EP18_PROVISION_RUN/shared_checks.txt"
ep18_shared_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
for ep18_validation_episode in episode18_things_eeg_concept_stability episode20_neurocam_prior_guided_codesign; do
  (
    cd "$OAK/users/zijiao/br_autoresearch/$ep18_validation_episode" || exit 2
    timeout --signal=TERM --kill-after=2s 30s "$ep18_shared_astra" validate outputs/astra/v0.0.14/astra.yaml --json \
      > "$EP18_PROVISION_RUN/${ep18_validation_episode}_astra.json" 2>&1
  )
  printf 'ASTRA_episode=%s exit=%s\n' "$ep18_validation_episode" "$?" | tee -a "$EP18_PROVISION_RUN/shared_checks.txt"
done
set -e
ep18_phase=transfer_context_regression
timeout --signal=TERM --kill-after=2s 30s python3.12 -B -m unittest discover \
  -s outputs/code -p test_model_asset_transfer_guard.py -v \
  > "$EP18_PROVISION_RUN/transfer_context_tests.log" 2>&1
ep18_phase=exact_cpu_runtime_install
timeout --signal=TERM --kill-after=10s 2400s bash outputs/code/prepare_exact_cpu_runtime.sh \
  > "$EP18_PROVISION_RUN/runtime_setup.log" 2>&1
printf '%s\n' 'exact_cpu_runtime_install exit=0'
# Setup ran in a child shell. Clear any shared-check module paths here too
# before using its private environment for transfer and model qualification.
module purge
module load devel python/3.12.1
unset PYTHONPATH PYTHONHOME
ep18_environment="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_${SLURM_JOB_ID}"
export EP18_RUNTIME_ENV="$ep18_environment"
ep18_phase=public_asset_acquisition
timeout --signal=TERM --kill-after=10s 1800s "$ep18_environment/bin/python" -B outputs/code/acquire_model_assets.py \
  --asset-map outputs/config/model_runtime_asset_map_proposal.json --max-seconds 1800 \
  > "$EP18_PROVISION_RUN/model_asset_acquisition.log" 2>&1
printf '%s\n' 'public_asset_acquisition exit=0'
ep18_phase=offline_model_runtime_qualification
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
timeout --signal=TERM --kill-after=10s 1210s "$ep18_environment/bin/python" -B outputs/code/qualify_model_runtime.py \
  --asset-map outputs/config/model_runtime_asset_map_proposal.json \
  --output-dir "$EP18_PROVISION_RUN/model_runtime" --max-seconds 1200 \
  > "$EP18_PROVISION_RUN/model_runtime.log" 2>&1
ep18_phase=completed_public_asset_and_runtime_provisioning
printf '%s\n' 'offline_model_runtime_qualification exit=0; no feature inference or EEG access'
