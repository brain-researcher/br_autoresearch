#!/bin/bash
# One explicitly admitted, exact work5 remainder under plan002.
# Original fragments and the separate five-fragment quarantine remain intact.
#SBATCH --job-name=ep21-full-workers-001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/decoder_training/full_remainder_work5_001-%A_%a.out
#SBATCH --error=outputs/decoder_training/full_remainder_work5_001-%A_%a.err
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
ep21_work5_host=$(hostname)
printf 'WORK5_REMAINDER_HOST=%s JOB=%s ARRAY=%s INDEX=%s CPUS=%s MEM_MIB=%s\n' \
  "$ep21_work5_host" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_JOB_ID:?}" \
  "${SLURM_ARRAY_TASK_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:?}"
[[ "$ep21_work5_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 \
   && "$SLURM_MEM_PER_NODE" == 7168 && "$SLURM_ARRAY_TASK_ID" == 4 ]]
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
ep21_work5_checks="${PWD}/outputs/decoder_training/work5_remainder_checks_job_${SLURM_JOB_ID}"
mkdir "$ep21_work5_checks"
printf 'job=%s\nparent_charged_once_to=EP21\nlogical_member=work_0005\n' \
  "$SLURM_JOB_ID" > "$ep21_work5_checks/status.txt"
set +e
(
  cd outputs/decoder_training/tests || exit 2
  timeout --signal=TERM --kill-after=5s 180s python3.12 -B -m unittest -v \
    test_work5_quarantine_remainder test_submit_deadline_remainders \
    test_decoder_remainder_supplement test_decoder_remainder_bundle test_worker_scratch_policy
) > "$ep21_work5_checks/checks.txt" 2>&1
ep21_work5_checks_exit=$?
set -e
printf 'required_checks_exit=%s\n' "$ep21_work5_checks_exit" >> "$ep21_work5_checks/status.txt"
[[ "$ep21_work5_checks_exit" == 0 ]]
printf 'Required_exact_work5_remainder_checks=passed\n'

# Document validation remains independent of fitting and inspection authority.
ep21_work5_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep21_work5_dependencies="${GROUP_HOME}/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/lib/python3.12/site-packages"
set +e
PYTHONPATH="$ep21_work5_astra/lib/python3.12/site-packages:$ep21_work5_dependencies" \
  timeout --signal=TERM --kill-after=5s 30s "$ep21_work5_astra/bin/astra-tools" validate \
  outputs/astra/v0.0.14/astra.yaml --json > "$ep21_work5_checks/astra_validation.json" 2>&1
ep21_work5_astra_exit=$?
set -e
printf 'astra_exit=%s\n' "$ep21_work5_astra_exit" >> "$ep21_work5_checks/status.txt"
# Three read-only scans share this substantive parent. Clean internal timeouts
# retain incomplete evidence and may continue; storage/admission failures stop
# only this affected parent. No inspected data enter the following fit.
set +e
bash outputs/decoder_training/selected_io_inspections_001.sh \
  > "$ep21_work5_checks/selected_inspections.txt" 2>&1
ep21_work5_inspection_exit=$?
set -e
printf 'selected_inspections_exit=%s\n' "$ep21_work5_inspection_exit" >> "$ep21_work5_checks/status.txt"
[[ "$ep21_work5_inspection_exit" == 0 ]]

# The runner verifies accepted registration, exact remaining spans, actual
# resources and allocation EndTime minus300s before reading fitted inputs.
exec timeout --signal=TERM --kill-after=10s 28530s python3.12 -u -B \
  outputs/decoder_training/tools/train_decoder_remainder.py \
  --authorization outputs/decoder_training_full_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --remainder-plan outputs/decoder_training/full_continuation_001/deadline_remainders_002.json \
  --work-item work_0005 --max-seconds 28500
