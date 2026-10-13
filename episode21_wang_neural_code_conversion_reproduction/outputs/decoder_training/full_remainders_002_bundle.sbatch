#!/bin/bash
# UNACTIVATED: one explicitly registered work6 + work13 remainder bundle.
# Both logical IDs keep their original intervals and separate fresh outputs.
# One physical parent is charged once; no dispatch, retry or successor here.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_remainders002_bundle-%A_%a.out
#SBATCH --error=outputs/decoder_training/full_remainders002_bundle-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_bundle_start_seconds=$SECONDS
ep21_bundle_host=$(hostname)
printf 'BUNDLE_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_bundle_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_bundle_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 && "$SLURM_ARRAY_TASK_ID" == 5 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
[[ -n "${L_SCRATCH_JOB:?}" && -d "$L_SCRATCH_JOB" ]]
export TMPDIR="$L_SCRATCH_JOB"

# Exercise the changed finite admission and assembly compatibility in this
# substantive allocation before either child receives empirical input.
ep21_bundle_checks="${PWD}/outputs/decoder_training/remainder_bundle_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_bundle_checks"
printf 'job=%s\nparent_charged_once_to=EP21\nlogical_members=work_0006,work_0013\n' \
  "$SLURM_JOB_ID" > "$ep21_bundle_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_decoder_deadline_remainders test_submit_deadline_remainders \
    test_decoder_remainder_assembly test_reviewed_deadline_partials \
    test_submit_full_decoder_jobs test_decoder_assembly \
    test_decoder_remainder_bundle test_decoder_remainder_supplement \
    test_reviewed_io_quarantine test_inspect_closed_io_fragments \
    test_reviewed_bundle_startup_replacement
) > "$ep21_bundle_checks/checks.txt" 2>&1
ep21_bundle_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_bundle_checks_exit" >> "$ep21_bundle_checks/status.txt"
[[ "$ep21_bundle_checks_exit" == 0 ]]
printf 'Required_exact_bundle_admission_checks=passed\n'

# Check only the five terminal I/O-failed fragments. Read-only inspection
# cannot turn uncertain coefficients into reusable coverage. Fresh fitting
# stops if this detects a new storage/read/close error or exhausts its bound.
ep21_inspection_end_epoch=$(python3.12 -B - <<'EP21_INSPECTION_END'
import sys
sys.path.insert(0, "outputs/decoder_training/tools")
import decoder_execution_contract as contract
resources = {"cpus": 1, "memory_gib": 7, "gpus": 0, "minutes": 120,
             "durable_bytes": 1000000, "scratch_bytes": 2000000000}
print(contract.verify_allocation(resources)["job_end_epoch"])
EP21_INSPECTION_END
)
set +e
timeout --signal=TERM --kill-after=5s 605s python3.12 -u -B \
  outputs/decoder_training/tools/inspect_closed_io_fragments.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --allocation-end-epoch "$ep21_inspection_end_epoch" --max-seconds 600 \
  > "$ep21_bundle_checks/closed_io_inspection.json" 2> "$ep21_bundle_checks/closed_io_inspection.err"
ep21_inspection_exit=$?
set -e
printf 'closed_io_inspection_exit=%s\n' "$ep21_inspection_exit" >> "$ep21_bundle_checks/status.txt"
[[ "$ep21_inspection_exit" == 0 ]]

# EP15's pending source-only checks share this substantive allocation. No MDTB
# values or reader release. Child runtime changes stay in its own shell and
# its result cannot veto the independent decoder fits.
set +e
timeout --signal=TERM --kill-after=5s 210s bash <<'EP15_PENDING_COMPONENTS'
set -uo pipefail
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-scikit-learn/1.5.1_py312 || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${OAK:?}/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry" || exit 2
ep15_pending_dir="outputs/runs/component_checks_job_${SLURM_JOB_ID}"
mkdir "$ep15_pending_dir" || exit 2
printf 'job=%s\nparent_charged_once_to=EP21\nempirical_execution=false\n' \
  "$SLURM_JOB_ID" > "$ep15_pending_dir/status.txt" || exit 2
exec > "$ep15_pending_dir/wrapper.txt" 2>&1 || exit 2
hostname
printf 'SLURM_JOB_ID=%s CPU_LIMIT=1 MEMORY_GIB=7\n' "$SLURM_JOB_ID"
(
  cd outputs/tests || exit 2
  timeout --foreground --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v \
    test_development_lopo test_local_parcel_prediction test_smooth_relocation \
    test_matched_random_warp test_local_isometry_variants test_paired_stretch_variants \
    test_anatomical_group_baseline test_matched_random_basis \
    test_local_functional_source test_calibration_falsifiers test_m2_alignment_signature
) > "$ep15_pending_dir/tests.txt" 2>&1
ep15_pending_exit=$?
printf 'tests_exit=%s\n' "$ep15_pending_exit" >> "$ep15_pending_dir/status.txt" || exit 2
ep15_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep15_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
PYTHONPATH="$ep15_astra/lib/python3.12/site-packages:$ep15_dependencies" \
  timeout --foreground --signal=TERM --kill-after=5s 30s "$ep15_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep15_pending_dir/astra_validation.json" 2>&1
ep15_astra_exit=$?
printf 'astra_exit=%s\n' "$ep15_astra_exit" >> "$ep15_pending_dir/status.txt"
[[ "$ep15_pending_exit" == 0 && "$ep15_astra_exit" == 0 ]]
EP15_PENDING_COMPONENTS
ep21_optional_ep15_exit=$?
set -e
printf 'optional_ep15_exit=%s\n' "$ep21_optional_ep15_exit" >> "$ep21_bundle_checks/status.txt"

# Changed EP21 program validation is documentation evidence, not a fit gate.
ep21_bundle_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_bundle_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
set +e
PYTHONPATH="$ep21_bundle_astra/lib/python3.12/site-packages:$ep21_bundle_dependencies" \
  timeout --signal=TERM --kill-after=5s 30s "$ep21_bundle_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep21_bundle_checks/astra_validation.json" 2>&1
ep21_bundle_astra_exit=$?
set -e
printf 'astra_exit=%s\n' "$ep21_bundle_astra_exit" >> "$ep21_bundle_checks/status.txt"

run_ep21_bundle_member() {
  local ep21_bundle_work=$1
  # The shared 6,900-second wrapper allowance includes modules and checks;
  # it never restarts for child2. Each worker also clips its internal deadline
  # to the independently verified actual Slurm EndTime minus 300 seconds.
  local ep21_bundle_remaining=$((6900 - (SECONDS - ep21_bundle_start_seconds)))
  [[ "$ep21_bundle_remaining" -gt 30 ]] || return 2
  printf 'Starting_bundle_member=%s shared_wrapper_seconds_remaining=%s\n' \
    "$ep21_bundle_work" "$ep21_bundle_remaining"
  set +e
  timeout --signal=TERM --kill-after=10s "${ep21_bundle_remaining}s" python3.12 -u -B \
    outputs/decoder_training/tools/train_decoder_remainder.py \
    --authorization outputs/decoder_training_full_authorization.json \
    --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
    --remainder-plan outputs/decoder_training/full_continuation_001/deadline_remainders_002.json \
    --work-item "$ep21_bundle_work" --max-seconds "$((ep21_bundle_remaining - 30))"
  local ep21_bundle_member_exit=$?
  set -e
  printf '%s_exit=%s\n' "$ep21_bundle_work" "$ep21_bundle_member_exit" >> "$ep21_bundle_checks/status.txt"
  return "$ep21_bundle_member_exit"
}

# The real Slurm index remains5 throughout. Logical work13 keeps index12 only
# in its original manifest and child status; no allocation identity is forged.
run_ep21_bundle_member work_0006
run_ep21_bundle_member work_0013
printf 'All_registered_bundle_members_completed=true\n' >> "$ep21_bundle_checks/status.txt"
