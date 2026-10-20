#!/bin/bash
# Coordinator submits only after the original36-record nearest-only result and
# the selected interior001 manifest exist. No automatic report or successor.
#SBATCH --job-name=ep20-s5-interior
#SBATCH --partition=russpold,normal
#SBATCH --qos=normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-26%27
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/reference_s5/refinement_interior001/slurm-%A_%a.out
#SBATCH --error=outputs/runs/reference_s5/refinement_interior001/slurm-%A_%a.err
set -euo pipefail
umask 077
[[ $# -eq 0 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]] || exit 2
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 ]] || exit 2
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]] || exit 2
[[ "${SLURM_MEM_PER_NODE:-}" == 2048 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]] || exit 2
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|1[0-9]|2[0-6])$ ]] || exit 2
s5_task=$SLURM_ARRAY_TASK_ID
s5_structure="R$((s5_task / 9 + 1))"
s5_point=$((s5_task % 9))
printf -v s5_point_label '%02d' "$s5_point"
printf -v s5_task_label '%02d' "$s5_task"
s5_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
s5_base="$s5_root/outputs/runs/reference_s5/refinement_interior001"
s5_manifest="$s5_base/manifest.json"
s5_record="$s5_base/records/$s5_structure/point_$s5_point_label"
s5_operations="$s5_base/operations_array_${SLURM_ARRAY_JOB_ID}"
s5_run="$s5_operations/task_${s5_task_label}_job_${SLURM_JOB_ID}"
[[ -f "$s5_manifest" ]] || exit 2
mkdir -p -m 700 "$s5_operations"
mkdir -m 700 "$s5_run"
s5_phase=setup
s5_record_status=not_observed
s5_finish() {
  s5_exit=$?
  trap - EXIT
  printf '{"phase":"%s","exit_code":%s,"finished_utc":"%s","round_id":"interior001","array_job_id":"%s","array_task_id":%s,"job_id":"%s","structure":"%s","point":%s,"target_frames":4000,"record_status":"%s","completion_source":"record_operation/run_report.json","held_aside_access":false,"automatic_successor":false}\n' \
    "$s5_phase" "$s5_exit" "$(date -u +%FT%TZ)" "$SLURM_ARRAY_JOB_ID" \
    "$s5_task" "$SLURM_JOB_ID" "$s5_structure" "$s5_point" "$s5_record_status" \
    > "$s5_run/terminal.json"
  exit "$s5_exit"
}
trap s5_finish EXIT
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
cd "$s5_root/outputs/tools"
s5_deadline=$(( ${SLURM_JOB_END_TIME:?} - 120 ))
if [[ -e "$s5_record/frame_checkpoint.npz" ]]; then
  set -- --resume
else
  set --
fi
s5_left=$((s5_deadline - $(date +%s)))
(( s5_left > 60 )) || exit 124
s5_phase=full_record_nearest_only_refinement_point
/usr/bin/time -v -o "$s5_run/resources.txt" \
  timeout --signal=TERM --kill-after=30s "$((s5_left+30))s" \
  python3.12 -u -B reference_s5_refinement.py run \
  --manifest "$s5_manifest" --structure "$s5_structure" --point "$s5_point" \
  --record-dir "$s5_record" --run-dir "$s5_run/record_operation" \
  --stop-after-frame 4000 --deadline-seconds "$s5_left" --memory-mib 1536 \
  "$@" > "$s5_run/record_stdout.txt" 2>&1
s5_phase=read_completion_status
s5_record_status=$(python3.12 -B - "$s5_run/record_operation/run_report.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
status = report.get("status")
if status not in ("complete_record", "clean_prefix", "bounded_stop", "failed"):
    raise SystemExit("worker completion status is missing or unrecognized")
if status == "complete_record" and (
        report.get("target_frames") != 4000 or report.get("completed_frames_after") != 4000):
    raise SystemExit("complete_record lacks the full 4000-frame completion count")
print(status)
PY
)
case "$s5_record_status" in
  complete_record) s5_phase=full_record_complete_nearest_only ;;
  clean_prefix|bounded_stop) s5_phase=bounded_attempt_incomplete_consult_checkpoint ;;
  failed) s5_phase=worker_reported_failure; exit 2 ;;
esac
