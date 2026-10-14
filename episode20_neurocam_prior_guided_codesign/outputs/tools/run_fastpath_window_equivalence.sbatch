#!/bin/bash
#SBATCH --job-name=ep20-dwell-windows
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^3(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_base="$ep20_root/outputs/runs/fast_forward_stage1/solver"
ep20_run="$ep20_base/dwell_windows002"
mkdir "$ep20_run"
ep20_phase=setup
trap 'ep20_exit=$?; printf "{\"phase\":\"%s\",\"exit_code\":%s,\"finished_utc\":\"%s\",\"automatic_scaling_selected\":false}\n" "$ep20_phase" "$ep20_exit" "$(date -u +%FT%TZ)" > "$ep20_run/terminal.json"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'host=%s\njob=%s\nallocated_cpus=%s\nmemory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_CPUS_PER_NODE" "$SLURM_MEM_PER_NODE" > "$ep20_run/status.txt"
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
ep20_deadline=$((ep20_start+3240))
(( ep20_deadline <= ep20_end-60 )) || ep20_deadline=$((ep20_end-60))
(( ep20_deadline-ep20_start <= 3240 && ep20_deadline-$(date +%s) > 480 ))
ep20_phase=focused_checks
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B -m unittest -v \
 test_fast_solver_complex_forcing > "$ep20_run/checks.txt" 2>&1
ep20_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
env PYTHONPATH="$ep20_astra/lib/python3.12/site-packages:$ep20_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep20_astra/bin/astra-tools" validate "$ep20_root/outputs/astra/v0.0.14/astra.yaml" --json > "$ep20_run/astra.txt" 2>&1 || true
ep20_phase=repaired_first_dwell
mkdir "$ep20_run/prefix"
/usr/bin/time -v -o "$ep20_run/prefix/resources.txt" timeout --signal=TERM --kill-after=30s 290s \
 python3.12 -u -B fast_solver_fastpath.py --run-dir "$ep20_run/prefix" --slots 25 \
 --deadline-seconds 270 > "$ep20_run/prefix/stdout.txt" 2>&1
ep20_phase=prefix_equivalence
ep20_compare_exit=0
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B fast_solver_fastpath_compare.py \
 --reference "$ep20_base/ref_frame_standard" --solver "$ep20_run/prefix" \
 --output "$ep20_run/prefix/equivalence.json" > "$ep20_run/prefix/equivalence.txt" 2>&1 || ep20_compare_exit=$?
(( ep20_compare_exit == 2 ))
python3.12 -B - "$ep20_run/prefix/equivalence.json" <<'PREFIX'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report['missing_windows'] == ['frame']
assert not report['invalid_windows'] and not report['available_window_failure']
assert report['solver_status'] == 'complete' and report['qualified'] is False
assert report['windows']['frame']['missing_solver'] is True
assert report['windows']['frame']['missing_reference'] is False
for name in ('5us', 'dwell'):
    assert report['windows'][name]['passed'] is True
    assert report['windows'][name]['energy_audit']['available'] is True
PREFIX
ep20_phase=standard_frame
mkdir "$ep20_run/standard"
ep20_standard_start=$(date +%s)
ep20_left=$((ep20_deadline-ep20_standard_start-150))
(( ep20_left > 60 ))
/usr/bin/time -v -o "$ep20_run/standard/resources.txt" timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B fast_solver_fastpath.py --run-dir "$ep20_run/standard" --slots 1600 \
 --deadline-seconds "$ep20_left" > "$ep20_run/standard/stdout.txt" 2>&1
ep20_phase=standard_equivalence
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B fast_solver_fastpath_compare.py \
 --reference "$ep20_base/ref_frame_standard" --solver "$ep20_run/standard" \
 --output "$ep20_run/standard/equivalence.json" > "$ep20_run/standard/equivalence.txt" 2>&1
# Select the second complete frame only if twice the measured first-frame cost
# plus its comparison fits the same shared deadline. Never extend or resubmit.
ep20_now=$(date +%s)
ep20_offset_need=$((2*(ep20_now-ep20_standard_start)+150))
if (( ep20_deadline-ep20_now < ep20_offset_need )); then
  ep20_phase=complete_standard_offset_deferred_time
  exit 0
fi
ep20_phase=offset_frame
mkdir "$ep20_run/offset"
ep20_left=$((ep20_deadline-$(date +%s)-150))
/usr/bin/time -v -o "$ep20_run/offset/resources.txt" timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B fast_solver_fastpath.py --run-dir "$ep20_run/offset" --slots 1600 \
 --deadline-seconds "$ep20_left" --offsets-json "$ep20_base/offsets_fixture.json" > "$ep20_run/offset/stdout.txt" 2>&1
ep20_phase=offset_available_prefix_equivalence
ep20_compare_exit=0
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B fast_solver_fastpath_compare.py \
 --reference "$ep20_base/ref_frame_offsets" --solver "$ep20_run/offset" \
 --output "$ep20_run/offset/partial_equivalence.json" > "$ep20_run/offset/equivalence.txt" 2>&1 || ep20_compare_exit=$?
(( ep20_compare_exit == 2 ))
python3.12 -B - "$ep20_run/offset/partial_equivalence.json" <<'CHECK'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report['missing_windows'] == ['frame']
assert not report['invalid_windows'] and not report['available_window_failure']
assert report['solver_status'] == 'complete' and report['qualified'] is False
for name in ('5us', 'dwell'):
    assert report['windows'][name]['passed'] is True
    assert report['windows'][name]['energy_audit']['available'] is True
CHECK
ep20_phase=complete_frames_offset_reference_pending
