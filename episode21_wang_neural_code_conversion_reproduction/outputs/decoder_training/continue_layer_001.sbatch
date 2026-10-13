#!/bin/bash
# UNACTIVATED: requires a separate exact scientist-approved authority file.
# Coordinator chooses a discovered eligible nonpreemptible partition once approved.
#SBATCH --job-name=ep21-layer-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/continue_layer001-%j.out
#SBATCH --error=outputs/decoder_training/continue_layer001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_layer_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_layer_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_layer_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_layer_authorization.json && ! -e outputs/decoder_training/layer_continuation_001 ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
# Separately scoped EP19 arithmetic/interface checks share this parent's limits.
# Their result is recorded independently and does not gate unrelated EP21 fits.
ep21_layer_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
set +e
(
  set -e
  cd "${OAK:?}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
  ep19_check_dir="outputs/runs/classical_backend_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep19_check_dir"
  set +e
  (
    cd outputs/tests || exit 2
    timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
      test_classical_forecast_backend test_non_neural_forecast_backend
  ) > "$ep19_check_dir/tests.txt" 2>&1
  ep19_check_exit=$?
  printf 'job=%s\nchecks_exit=%s\nparent_charged_once_to=EP21\n' \
    "$SLURM_JOB_ID" "$ep19_check_exit" > "$ep19_check_dir/status.txt"
  timeout --signal=TERM --kill-after=5s 30s "$ep21_layer_astra" validate \
    outputs/astra/v0.0.14/astra.yaml --json > "$ep19_check_dir/astra_validation.json" 2>&1
  ep19_astra_exit=$?
  printf 'astra_validation_exit=%s\n' "$ep19_astra_exit" >> "$ep19_check_dir/status.txt"
  printf 'EP19 checks_exit=%s astra_validation_exit=%s\n' "$ep19_check_exit" "$ep19_astra_exit"
  exit "$ep19_check_exit"
)
ep19_bundle_exit=$?
printf 'EP19_optional_bundle_exit=%s\n' "$ep19_bundle_exit"
# Source-free checks for the prepared full-stage code do not activate that stage.
# Keep failures separate from the required checks for this first-layer worker.
(
  set -e
  ep21_full_check_dir="${PWD}/outputs/decoder_training/full_pipeline_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep21_full_check_dir"
  cd outputs/decoder_training/tests
  set +e
  timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
    test_participant_training_state test_train_decoder_work_item \
    test_plan_full_decoder_work test_decoder_assembly test_submit_full_decoder_jobs \
    > "$ep21_full_check_dir/tests.txt" 2>&1
  ep21_full_check_exit=$?
  printf 'job=%s\nchecks_exit=%s\nempirical_payload_access=false\nparent_charged_once_to=EP21\n' \
    "$SLURM_JOB_ID" "$ep21_full_check_exit" > "$ep21_full_check_dir/status.txt"
  exit "$ep21_full_check_exit"
)
ep21_full_bundle_exit=$?
printf 'EP21_optional_full_pipeline_bundle_exit=%s\n' "$ep21_full_bundle_exit"
timeout --signal=TERM --kill-after=5s 30s "$ep21_layer_astra" validate \
  outputs/astra/v0.0.14/astra.yaml --json \
  > "outputs/decoder_training/continue_layer001-astra-${SLURM_JOB_ID}.json" 2>&1
ep21_layer_astra_exit=$?
set -e
printf 'EP21_astra_validation_exit=%s\n' "$ep21_layer_astra_exit"
# New loader/coverage/publication tests only, inside the substantive allocation.
timeout --signal=TERM --kill-after=10s 60s python3.12 -B -m unittest discover \
  -s outputs/decoder_training/tests -p test_train_layer_continuation.py -v
exec timeout --signal=TERM --kill-after=10s 86130s python3.12 -u -B \
  outputs/decoder_training/tools/train_layer_continuation.py \
  --authorization outputs/decoder_training_layer_authorization.json --max-seconds 86100
