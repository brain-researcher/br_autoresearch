#!/bin/bash
# One attended saved-inner-loss selection comparison; no EEG fitting.
set -euo pipefail
[[ $# == 0 && "${SLURM_JOB_ID:-}" == 47586225 && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && -n "${SLURM_STEP_ID:-}" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
(( $(date +%s) + 240 < ${SLURM_JOB_END_TIME:?} ))
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep20_root="$OAK/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep18_run="$ep18_root/outputs/runs/repeated_selector_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_run"
ep18_local="${L_SCRATCH_JOB:?}/repeated-selector-${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
printf 'hostname=%s job=%s step=%s cpus=1 threads=1 companion_memory_mib=512\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" | tee "$ep18_run/allocation.txt"
cd "$ep20_root/outputs/tools"
if timeout --signal=TERM --kill-after=5s 60s python3.12 -B -m unittest -v \
  test_reference_s5_refinement_report.TestRefinementReport.test_distant_interpretation_distinguishes_ceiling_from_agreement_without_selecting_points \
  > "$ep20_root/outputs/runs/reference_s5/interpretation_fixture_${SLURM_JOB_ID}_${SLURM_STEP_ID}.txt" 2>&1; then
  printf 'EP20 interpretation fixture passed; distant outcomes unopened.\n'
else
  printf 'EP20 interpretation fixture failed; independent saved-inner comparison continues.\n'
fi
cd "$ep18_root/outputs/code"
/usr/bin/time -v -o "$ep18_run/resources.txt" \
  timeout --signal=TERM --kill-after=5s 180s python3.12 -u -B report_repeated_selector_sensitivity.py \
  --input-root "$ep18_root/outputs/runs/repeated_image_signal_recovery/control001" \
  --output-dir "$ep18_run/report" > "$ep18_run/stdout.txt" 2>&1
printf 'saved_inner_selector_report_complete=%s\n' "$ep18_run/report"
