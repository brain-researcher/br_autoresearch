#!/bin/bash
# UNACTIVATED full-stage preparation: one parent builds four states in order.
# The coordinator supplies a discovered partition after separate full approval.
#SBATCH --job-name=ep21-full-states
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_states001-%j.out
#SBATCH --error=outputs/decoder_training/full_states001-%j.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_states_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s\n' "$ep21_states_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_states_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 && "$SLURM_MEM_PER_NODE" == 7168 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
[[ -f outputs/decoder_training_full_authorization.json \
   && -f outputs/decoder_training/full_continuation_001/work_manifest.json ]]
source /share/software/user/open/lmod/lmod/init/bash
module purge
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"
# Required checks for the closed-layer storage repair run before new neural access.
ep21_state_check_dir="${PWD}/outputs/decoder_training/full_state_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_state_check_dir"
(
  cd outputs/decoder_training/tests
  timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
    test_decoder_closed_layer_storage
) > "$ep21_state_check_dir/storage_checks.txt" 2>&1
printf 'Required_closed_layer_storage_checks=passed\n'

# Separately owned, bounded EP20 component checks share this substantive parent.
# Their outcome does not gate unrelated EP21 state preparation.
set +e
(
  cd "${OAK}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign" || exit 2
  ep20_check_dir="outputs/runs/switched_line_checks_job_${SLURM_JOB_ID}"
  mkdir "$ep20_check_dir" || exit 2
  timeout --signal=TERM --kill-after=5s 60s python3.12 -B \
    outputs/tools/test_switched_line.py -v > "$ep20_check_dir/tests.txt" 2>&1
  ep20_test_exit=$?
  printf 'job=%s\nchecks_exit=%s\nparent_charged_once_to=EP21\n' \
    "$SLURM_JOB_ID" "$ep20_test_exit" > "$ep20_check_dir/status.txt"
  exit "$ep20_test_exit"
)
ep20_component_exit=$?
printf 'EP20_component_checks_exit=%s\n' "$ep20_component_exit"
ep21_state_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/astra-tools"
for ep21_checked_episode in \
  episode10_single_cell_coprojection \
  episode11_projection_types_vs_gradients \
  episode12_malecns_type_sufficiency \
  episode16_braingate_cross_session_failure_modes \
  episode20_neurocam_prior_guided_codesign \
  episode21_wang_neural_code_conversion_reproduction
do
  (
    cd "${OAK}/users/zijiao/br_autoresearch/${ep21_checked_episode}" || exit 2
    timeout --signal=TERM --kill-after=5s 30s "$ep21_state_astra" validate \
      outputs/astra/v0.0.14/astra.yaml --json \
      > "outputs/astra/v0.0.14/validation-adopted-${SLURM_JOB_ID}.json" 2>&1
  )
  printf 'ASTRA_episode=%s exit=%s\n' "$ep21_checked_episode" "$?"
done
set -e
# The Python budget also stops work 300 seconds before the actual allocation end.
exec timeout --signal=TERM --kill-after=10s 14130s python3.12 -u -B \
  outputs/decoder_training/tools/prepare_decoder_states.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --max-seconds 14100
