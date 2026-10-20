#!/bin/bash
# Coordinator-owned attended 1-CPU Slurm step after the selected writers finish.
# No submission, polling, simulation, live-record edits, or automatic successor.
# The tested active freeze requires all 36+27+3+3 full records before the report
# computes any distant amplitude. Every nearest-compatible point is retained.
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
s5_host=$(hostname)
[[ "$s5_host" != *-ln* && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_STEP_NUM_TASKS:-}" == 1 ]] || exit 2
[[ "${SLURM_JOB_END_TIME:-}" =~ ^[0-9]+$ ]] || exit 2
# Memory is bounded by the coordinator's finite parent Slurm allocation.
# RSS does not establish the virtual-address space needed by mapped libraries.
ulimit -c 0
export CUDA_VISIBLE_DEVICES=''
s5_started=$(date +%s)
s5_deadline=$((s5_started + 830))
s5_parent_deadline=$((SLURM_JOB_END_TIME - 60))
if (( s5_parent_deadline < s5_deadline )); then s5_deadline=$s5_parent_deadline; fi
(( s5_deadline > s5_started )) || exit 124
s5_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
s5_base="$s5_root/outputs/runs/reference_s5"
s5_initial_freeze="$s5_base/result_job_47382257_step_1/nearest_only_freeze.json"
s5_interior="$s5_base/refinement_interior001"
s5_center="$s5_base/refinement_center001"
s5_r2_center="$s5_base/refinement_center002"
s5_run="$s5_base/result_all69_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
s5_figure="$s5_root/outputs/figures/conditional_s5_all69_job_${SLURM_JOB_ID}_step_${SLURM_STEP_ID}"
[[ -f "$s5_initial_freeze" && ! -e "$s5_figure" ]] || exit 2
for s5_round in "$s5_interior" "$s5_center" "$s5_r2_center"; do
  [[ -f "$s5_round/manifest.json" && -d "$s5_round/records" ]] || exit 2
done
# Never overwrite a freeze, completed report, or partial rendering.
mkdir -m 700 "$s5_run"
s5_phase=setup
s5_freeze_complete=false
s5_report_complete=false
s5_finish() {
  s5_exit=$?
  trap - EXIT
  printf '{"phase":"%s","exit_code":%s,"finished_utc":"%s","job_id":"%s","step_id":"%s","required_records":69,"frames_per_record":4000,"freeze_complete":%s,"report_complete":%s,"automatic_successor":false}\n' \
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
export MPLCONFIGDIR="$TMPDIR/ep20_s5_all69_${SLURM_JOB_ID}_${SLURM_STEP_ID}_matplotlib"
mkdir -m 700 "$MPLCONFIGDIR"
cd "$s5_root/outputs/tools"

# saved_summary verifies 4,000 committed frames and 16 s for each of all 69
# records. This phase estimates only the nearest anchors. The active CLI also
# rejects missing center001/center002 history and any truncated compatible set.
s5_phase=all69_complete_record_validation_and_nearest_only_freeze
printf 'PHASE=%s\n' "$s5_phase"
s5_left=$((s5_deadline - $(date +%s)))
(( s5_left > 0 )) || exit 124
/usr/bin/time -v -o "$s5_run/freeze_resources.txt" \
  timeout --signal=TERM --kill-after=10s "${s5_left}s" \
  python3.12 -u -B reference_s5_refinement_report.py freeze \
  --initial-freeze "$s5_initial_freeze" \
  --manifest "$s5_interior/manifest.json" --records-root "$s5_interior/records" \
  --centering-manifest "$s5_center/manifest.json" --centering-records-root "$s5_center/records" \
  --r2-centering-manifest "$s5_r2_center/manifest.json" --r2-centering-records-root "$s5_r2_center/records" \
  --output "$s5_run/nearest_only_freeze.json" \
  > "$s5_run/freeze_stdout.txt" 2>&1
s5_freeze_complete=true

# Only the successfully persisted all-69 freeze reaches distant evaluation.
# Existing complete-record sufficient statistics supply all 4,096 pads: no
# circuit simulation is restarted. The reporter retains every compatible
# point and both contact labels; the range is a finite conditional envelope,
# not a device CI, continuous-domain rejection, or hardware validation.
# If the compatible set is empty, the reporter leaves distant values unopened.
s5_phase=frozen_all69_conditional_report
printf 'PHASE=%s\n' "$s5_phase"
s5_left=$((s5_deadline - $(date +%s)))
(( s5_left > 0 )) || exit 124
/usr/bin/time -v -o "$s5_run/report_resources.txt" \
  timeout --signal=TERM --kill-after=10s "${s5_left}s" \
  python3.12 -u -B reference_s5_refinement_report.py report \
  --freeze "$s5_run/nearest_only_freeze.json" --output-dir "$s5_figure" \
  --paper-far-db -45 --check-max-db -42 \
  > "$s5_run/report_stdout.txt" 2>&1
s5_report_complete=true
s5_phase=complete
printf 'PHASE=%s REPORT=%s\n' "$s5_phase" "$s5_figure"
