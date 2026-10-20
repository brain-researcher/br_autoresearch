#!/bin/bash
# Attended finite companion in the current task's 3-CPU/16-GiB EP21 parent.
# Decode remains one thread with a 15-GiB RSS guard; this step is one CPU/512MiB.
set -euo pipefail
[[ $# == 0 && "${SLURM_JOB_ID:-}" == 47586225 && "$(hostname)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && -n "${SLURM_STEP_ID:-}" ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
ep18_root="${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep15_root="$OAK/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry"
ep18_run="$ep18_root/outputs/runs/repeated_identity_metadata/job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_run"
ep18_local="${L_SCRATCH_JOB:?}/identity-companion-${SLURM_STEP_ID}"
mkdir -m 700 "$ep18_local"
export TMPDIR="$ep18_local" MPLCONFIGDIR="$ep18_local/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312
printf 'hostname=%s job=%s step=%s cpus=1 threads=1 companion_memory_mib=512\n' \
  "$(hostname)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" | tee "$ep18_run/allocation.txt"
cd "$ep15_root/outputs/tools"
if /usr/bin/time -v -o "../runs/observed_map_audit_fixture_retest_${SLURM_JOB_ID}_${SLURM_STEP_ID}.resources.txt" \
  timeout --signal=TERM --kill-after=5s 90s python3.12 -B -m unittest -v \
  test_observed_map_audit_predictions test_observed_map_audit_readers_runner test_report_observed_map_audit \
  > "../runs/observed_map_audit_fixture_retest_${SLURM_JOB_ID}_${SLURM_STEP_ID}.txt" 2>&1; then
  printf 'EP15 repaired fixture tests passed; no audit access.\n'
else
  printf 'EP15 fixture failure recorded; independent metadata diagnosis continues.\n'
fi
cd "$ep18_root/outputs/code"
timeout --signal=TERM --kill-after=5s 90s python3.12 -B -m unittest -v test_repeated_identity_metadata \
  > "$ep18_run/checks.txt" 2>&1
ep18_deadline=$(($(date +%s)+300))
(( ep18_deadline < ${SLURM_JOB_END_TIME:?}-30 ))
/usr/bin/time -v -o "$ep18_run/resources.txt" \
  timeout --signal=TERM --kill-after=5s 315s python3.12 -u -B repeated_identity_metadata.py \
  --output-dir "$ep18_run/diagnosis" --deadline-epoch "$ep18_deadline" --rss-limit-bytes 402653184
printf 'all30_identity_metadata_diagnosis_complete\n'
