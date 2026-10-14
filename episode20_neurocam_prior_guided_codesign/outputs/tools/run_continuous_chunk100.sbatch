#!/bin/bash
#SBATCH --job-name=ep20-continuous-100
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage2/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage2/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_reference="$ep20_root/outputs/runs/fast_forward_stage1/solver/ref_frame_standard"
ep20_seam="$ep20_root/outputs/runs/fast_forward_stage1/solver/restart_seam001/seam/seam_comparison.json"
ep20_prefix="$ep20_root/outputs/runs/fast_forward_stage2/solver/continuous_10frames001/equivalence.json"
ep20_run="$ep20_root/outputs/runs/fast_forward_stage2/solver/continuous_100frames001"
ep20_record="$ep20_root/outputs/runs/fast_forward_stage2/solver/continuous_record001"
mkdir "$ep20_run"
ep20_phase=setup
trap 'ep20_exit=$?; printf "{\"phase\":\"%s\",\"exit_code\":%s,\"finished_utc\":\"%s\",\"record_target_frames\":4000,\"stop_after_frame\":100,\"scientific_scoring\":false,\"automatic_successor\":false,\"clipping_equivalence_asserted\":false}\n" "$ep20_phase" "$ep20_exit" "$(date -u +%FT%TZ)" > "$ep20_run/terminal.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'host=%s\njob=%s\nallocated_cpus=%s\nmemory_mib=%s\nrecord_directory=%s\n' \
 "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_CPUS_PER_NODE" "$SLURM_MEM_PER_NODE" "$ep20_record" > "$ep20_run/status.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
cd "$ep20_root/outputs/tools"
ep20_start=${SLURM_JOB_START_TIME:?}
ep20_end=${SLURM_JOB_END_TIME:?}
# Use the actual allocation's deadline, including time already spent in setup.
# The last 10% remains available for interruption/clean close; 180 seconds of
# the internal window are reserved for the saved first-frame comparison.
ep20_deadline=$((ep20_start+9*(ep20_end-ep20_start)/10))
(( ep20_deadline <= ep20_end-60 )) || ep20_deadline=$((ep20_end-60))
ep20_phase=qualified_seam_and_ten_frame_predecessors
python3.12 -B - "$ep20_seam" "$ep20_prefix" <<'PREDECESSORS'
import json, sys
from pathlib import Path
seam, prefix = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
assert seam['qualified'] is True and seam['propagated_frames'] == 3
assert prefix['qualified'] is True and prefix['all_physical_windows_passed'] is True
PREDECESSORS
# Engineering continuation only: original offset-reference equivalence remains
# required before clipping qualification and production scientific claims.
# Measured ten-frame elapsed was 2412.44 s, with warm frames around 221--239 s.
# The record retains these first 100 frames for subsequent chronological chunks.
ep20_phase=first_hundred_frames_of_persistent_record
mkdir "$ep20_run/standard"
mkdir "$ep20_record"
ep20_left=$((ep20_deadline-$(date +%s)-180))
(( ep20_left > 60 ))
/usr/bin/time -v -o "$ep20_run/resources.txt" \
 timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B fast_solver_continuous.py \
 --run-dir "$ep20_run/standard" --record-dir "$ep20_record" \
 --frames 4000 --stop-after-frame 100 \
 --deadline-seconds "$ep20_left" --memory-limit-mib 6656 --cache-mib 1024 \
 > "$ep20_run/stdout.txt" 2>&1
ep20_phase=checkpointed_hundred_and_first_frame_equivalence
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B - \
 "$ep20_reference" "$ep20_record" "$ep20_run/standard/solver_report.json" "$ep20_run/equivalence.json" \
 > "$ep20_run/equivalence.txt" 2>&1 <<'COMPARE'
import json, sys
from pathlib import Path
from conditional_forward_recipe import expected_state_ids
from fast_solver_fastpath_compare import compare
reference, record, report_path, destination = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
assert report['status'] == 'checkpointed' and report['record_complete'] is False
assert report['record_target_frames'] == 4000
assert report['frames_completed'] == report['frames_this_run'] == 100
assert report['output_rows_written'] == 100
assert Path(report['checkpoint_path']) == record / 'frame_checkpoint.npz'
assert Path(report['checkpoint_path']).is_file()
result = compare(reference, record / 'first_frame', state_ids=expected_state_ids())
result.update(reference=str(reference), solver=str(record / 'first_frame'),
              comparator='fastpath_complete_window_work', retained_record_frames=100,
              record_complete=False, scientific_scoring=False, clipping_equivalence_asserted=False)
with destination.open('x') as stream:
    json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result, sort_keys=True, allow_nan=False))
raise SystemExit(0 if result['qualified'] else 2)
COMPARE
ep20_phase=checkpointed_at_frame100_no_scoring_no_successor
