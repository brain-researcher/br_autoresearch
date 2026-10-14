#!/bin/bash
# Run once inside the coordinator's >=30-minute, >=1CPU/7GiB allocation.
# No scheduler calls, background work, renewal or automatic successor.
set -euo pipefail
[[ $# -eq 1 && "$1" =~ ^attempt[0-9]+$ ]] || exit 2
ep06_host=$(hostname)
[[ "$ep06_host" != *-ln* && -n "${SLURM_JOB_ID:-}" ]] || exit 2
cd "${OAK:?}/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
umask 077
ep06_base="$PWD/outputs/runs/lmp_support_followup"
mkdir -p "$ep06_base" outputs/figures/lmp_support_followup
ep06_launch="$ep06_base/${1}_launch"
mkdir "$ep06_launch"
printf 'host=%s\njob=%s\nstarted_epoch=%s\n' "$ep06_host" "$SLURM_JOB_ID" "$(date +%s)" > "$ep06_launch/execution.txt"
export TMPDIR="${L_SCRATCH_JOB:?}/ep06_lmp_support_${1}"
mkdir "$TMPDIR"
ulimit -S -f 1048576
(cd outputs/tests && timeout --signal=TERM --kill-after=5s 120s python3.12 -B -m unittest -v test_lmp_support_followup) > "$ep06_launch/checks.txt" 2>&1
timeout --signal=TERM --kill-after=5s 1510s python3.12 -u -B outputs/tools/run_lmp_support_followup.py \
  --views "$SCRATCH/br_autoresearch/episode06_lfp_regional_fingerprints/development001/development_views.h5" \
  --attempt "$ep06_base/$1" --max-seconds 1500 --rss-gib 5 > "$ep06_launch/run.txt" 2>&1
timeout --signal=TERM --kill-after=5s 90s python3.12 -B outputs/tools/plot_lmp_support_followup.py \
  --report "$ep06_base/$1/report.json" --output "$PWD/outputs/figures/lmp_support_followup/$1" > "$ep06_launch/plot.txt" 2>&1
printf 'ended_epoch=%s\n' "$(date +%s)" >> "$ep06_launch/execution.txt"
