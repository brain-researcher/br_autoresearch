#!/bin/bash
set -euo pipefail
umask 077
[[ $# == 0 && ${SLURM_JOB_ID:?} == 47579378 && ${SLURM_CPUS_PER_TASK:?} == 1 ]]
ep18_diag_host=$(hostname)
[[ "$ep18_diag_host" != *-ln* && ${SLURM_STEP_ID:?} != batch && "$SLURM_STEP_ID" != extern ]]
ep18_diag_now=$(date -u +%s)
(( SLURM_JOB_END_TIME - 60 > ep18_diag_now + 120 && SLURM_JOB_START_TIME + 21000 > ep18_diag_now + 120 ))
cd "${OAK:?}/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability"
ep18_diag_run="$PWD/outputs/runs/feature_preparation/image_task005_diagnostic001"
mkdir -m 700 "$ep18_diag_run"
printf 'parent_job=%s\nstep_id=%s\nhostname=%s\nrequested_cpus=1\nrequested_memory_mib=512\nparent_charged_once_to=EP21\n' \
  "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$ep18_diag_host" >"$ep18_diag_run/status.txt"
trap 'ep18_diag_exit=$?; printf "exit=%s\nfinished_utc=%s\n" "$ep18_diag_exit" "$(date -u +%FT%TZ)" >>"$ep18_diag_run/status.txt"' EXIT
ulimit -c 0
ulimit -S -f 2048
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=''
export TMPDIR="${L_SCRATCH_JOB:?}/ep18-clip-diagnostic001"
mkdir -m 700 "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache"
timeout --signal=TERM --kill-after=5s 60s python3.12 -B outputs/code/diagnose_clip_task005_001.py \
  >"$ep18_diag_run/report.json" 2>"$ep18_diag_run/stderr.txt"
