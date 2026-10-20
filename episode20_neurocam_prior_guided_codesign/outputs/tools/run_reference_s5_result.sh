#!/bin/bash
# Run only in the coordinator's attended, explicitly allocated 1-CPU compute
# step after all original records are complete. This script submits no job and
# queries no scheduler. Freeze independently requires all 36 original records,
# each durably complete at 4,000 frames/16 s; no live writer is modified.
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
s5_host=$(hostname)
[[ "$s5_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_STEP_NUM_TASKS:-}" == 1 ]] || exit 2
[[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]] || exit 2
# Bound this companion independently of the parent allocation's memory.
ulimit -S -v 524288
ulimit -c 0
# The parent may own a GPU. This CPU-only step must never expose it locally.
export CUDA_VISIBLE_DEVICES=''
s5_started=$(date +%s)
# Reserve the timeout's 10-second termination grace inside the 14-minute cap.
s5_deadline=$((s5_started + 830))
s5_parent_deadline=$((SLURM_JOB_END_TIME - 60))
if (( s5_parent_deadline < s5_deadline )); then s5_deadline=$s5_parent_deadline; fi
(( s5_deadline > s5_started )) || exit 124
s5_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
s5_base="$s5_root/outputs/runs/reference_s5"
s5_manifest="$s5_base/conditional_family.json"
s5_records="$s5_base/records"
s5_run="$s5_base/result_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
s5_figure="$s5_root/outputs/figures/conditional_s5_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
[[ -f "$s5_manifest" && -d "$s5_records" && ! -e "$s5_figure" ]] || exit 2
# Never overwrite an earlier freeze, report, or partial rendering.
mkdir -m 700 "$s5_run"
s5_phase=setup
s5_freeze_complete=false
s5_report_complete=false
s5_finish() {
  s5_exit=$?
  trap - EXIT
  printf '{"phase":"%s","exit_code":%s,"finished_utc":"%s","job_id":"%s","step_id":"%s","required_records":36,"frames_per_record":4000,"freeze_complete":%s,"report_complete":%s,"automatic_successor":false}\n' \
    "$s5_phase" "$s5_exit" "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" \
    "$s5_freeze_complete" "$s5_report_complete" > "$s5_run/terminal.json"
  exit "$s5_exit"
}
trap s5_finish EXIT
printf 'HOST=%s JOB=%s STEP=%s OUTPUT=%s\n' "$s5_host" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$s5_run"

source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312 viz py-matplotlib/3.8.3_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export TMPDIR="${L_SCRATCH_JOB:?}"
export MPLCONFIGDIR="$TMPDIR/ep20_s5_result_${SLURM_JOB_ID}_${SLURM_STEP_ID}_matplotlib"
mkdir -m 700 "$MPLCONFIGDIR"
cd "$s5_root/outputs/tools"

s5_phase=all36_complete_record_validation_and_nearest_only_freeze
printf 'PHASE=%s\n' "$s5_phase"
s5_left=$((s5_deadline - $(date +%s)))
(( s5_left > 0 )) || exit 124
/usr/bin/time -v -o "$s5_run/freeze_resources.txt" \
  timeout --signal=TERM --kill-after=10s "${s5_left}s" \
  python3.12 -u -B reference_s5_prediction.py freeze \
  --manifest "$s5_manifest" --records-root "$s5_records" \
  --output "$s5_run/nearest_only_freeze.json" \
  > "$s5_run/freeze_stdout.txt" 2>&1
s5_freeze_complete=true

# Only a successful full-grid freeze reaches the existing reporter. It emits
# a calibration-only result for an empty compatible set without opening any
# distant response or inspecting the paper/check arguments on that branch.
s5_phase=frozen_family_conditional_report
printf 'PHASE=%s\n' "$s5_phase"
s5_left=$((s5_deadline - $(date +%s)))
(( s5_left > 0 )) || exit 124
/usr/bin/time -v -o "$s5_run/report_resources.txt" \
  timeout --signal=TERM --kill-after=10s "${s5_left}s" \
  python3.12 -u -B reference_s5_report.py \
  --freeze "$s5_run/nearest_only_freeze.json" --output-dir "$s5_figure" \
  --paper-far-db -45 --check-max-db -42 \
  > "$s5_run/report_stdout.txt" 2>&1
s5_report_complete=true
s5_phase=complete
printf 'PHASE=%s REPORT=%s\n' "$s5_phase" "$s5_figure"
