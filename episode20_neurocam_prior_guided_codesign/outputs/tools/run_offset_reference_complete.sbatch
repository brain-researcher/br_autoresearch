#!/bin/bash
#SBATCH --job-name=ep20-offset-reference
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=14:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_base="$ep20_root/outputs/runs/fast_forward_stage1/solver"
ep20_run="$ep20_base/ref_frame_offsets_full002"
mkdir "$ep20_run"
ep20_phase=setup
trap 'ep20_exit=$?; printf "phase=%s\nexit=%s\nfinished_utc=%s\n" "$ep20_phase" "$ep20_exit" "$(date -u +%FT%TZ)" >> "$ep20_run/status.txt"' EXIT
printf 'host=%s\njob=%s\nallocated_cpus=%s\nmemory_mib=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_CPUS_PER_NODE" "$SLURM_MEM_PER_NODE" > "$ep20_run/status.txt"
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
cd "$ep20_root/outputs/tools"
ep20_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
env PYTHONPATH="$ep20_astra/lib/python3.12/site-packages:$ep20_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep20_astra/bin/astra-tools" validate "$ep20_root/outputs/astra/v0.0.14/astra.yaml" --json > "$ep20_run/astra.txt" 2>&1 || true
ep20_start=${SLURM_JOB_START_TIME:?}
ep20_end=${SLURM_JOB_END_TIME:?}
ep20_now=$(date +%s)
ep20_left=$((ep20_start+(ep20_end-ep20_start)*9/10-ep20_now))
(( ep20_left > 60 && ep20_left <= 45360 ))
printf -v ep20_slots '%s,' 2 {25..1600..25}
ep20_slots=${ep20_slots%,}
ep20_phase=original_reference
/usr/bin/time -v -o "$ep20_run/resources.txt" timeout --signal=TERM --kill-after=60s "$((ep20_left+30))s" \
 python3.12 -u -B fast_solver_reference.py --run-dir "$ep20_run" --slots 1600 \
 --tolerance standard --offsets-json "$ep20_base/offsets_fixture.json" \
 --checkpoint-slots "$ep20_slots" --deadline-seconds "$ep20_left" > "$ep20_run/stdout.txt" 2>&1
ep20_phase=full_offset_equivalence
timeout --signal=TERM --kill-after=10s 120s python3.12 -u -B fast_solver_compare.py \
 --reference "$ep20_run" --solver "$ep20_base/dev_d3/step3_offsetsolve" \
 --output "$ep20_run/offset_equivalence.json" > "$ep20_run/offset_equivalence.txt" 2>&1
python3.12 -B - "$ep20_run" <<'CHECK'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); r = json.loads((root/'reference_report.json').read_text())
c = json.loads((root/'offset_equivalence.json').read_text())
assert r['status'] == 'complete' and r['completed_intervals'] == 6400
assert set(c['windows']) == {'5us', 'dwell', 'frame'}
passed = all(v.get('passed') is True for v in c['windows'].values())
(root/'equivalence_admission.json').write_text(json.dumps({'reference_complete': True,
 'required_windows': ['5us','dwell','frame'], 'all_windows_passed': passed,
 'automatic_scaling_selected': False}, indent=2)+'\n')
assert passed, 'At least one required scientific equivalence window failed'
CHECK
ep20_phase=complete
