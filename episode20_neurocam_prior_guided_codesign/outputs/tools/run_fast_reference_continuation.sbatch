#!/bin/bash
# Resume saved cycle-0 reference state; no automatic retry or resubmission.
# Usage: sbatch run_fast_reference_continuation.sbatch SOURCE_SUBDIR FRESH_SUBDIR
#SBATCH --job-name=ep20-ref-resume
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -uo pipefail
ep20_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s START=%s\n' "$ep20_host" "${SLURM_JOB_ID:?}" \
  "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:-unknown}" "$(date -Is)"
[[ "$ep20_host" != *-ln* && $# -eq 2 ]] || exit 2
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_base="$ep20_root/outputs/runs/fast_forward_stage1/solver"
ep20_run="$ep20_base/$2"
mkdir "$ep20_run" || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:-/tmp}"
ep20_start=${SLURM_JOB_START_TIME:?}
ep20_end=${SLURM_JOB_END_TIME:?}
ep20_limit=$((ep20_end-ep20_start))
ep20_now=$(date +%s)
ep20_left=$((ep20_start+ep20_limit*9/10-ep20_now))
((ep20_left > 60)) || exit 2
printf 'source=%s\njob=%s\ndeadline_seconds=%s\n' "$1" "$SLURM_JOB_ID" "$ep20_left" > "$ep20_run/status.txt"
cd "$ep20_root/outputs/tools" || exit 2
# Focused numerical-bound checks belong to this substantive continuation.
timeout --signal=TERM --kill-after=5s 30s python3.12 -B - > "$ep20_run/roundoff_checks.txt" 2>&1 <<'CHECK'
import numpy as np
from fast_solver_reference_continue import energy_replay_check
base = np.array([1e-12, 1e-8, 0.0])
assert energy_replay_check(base, base, 0)['exact']
within = base.copy()
for _ in range(8): within[:2] = np.nextafter(within[:2], np.inf)
assert energy_replay_check(base, within, 0)['passed']
outside = within.copy(); outside[0] = np.nextafter(outside[0], np.inf)
assert not energy_replay_check(base, outside, 0)['passed']
nonzero = base.copy(); nonzero[2] = np.nextafter(0.0, 1.0)
assert not energy_replay_check(base, nonzero, 0)['passed']
print('Exact identity, eight/nine ULP boundary and zero-reference checks passed.')
CHECK
[[ $? == 0 ]] || exit 2
ep20_astra="$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
env PYTHONPATH="$ep20_astra/lib/python3.12/site-packages:$ep20_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep20_astra/bin/astra-tools" validate "$ep20_root/outputs/astra/v0.0.14/astra.yaml" --json > "$ep20_run/astra.txt" 2>&1 || true
ep20_now=$(date +%s)
ep20_left=$((ep20_start+ep20_limit*9/10-ep20_now))
((ep20_left > 60)) || exit 2
/usr/bin/time -v -o "$ep20_run/resources.txt" \
  timeout --signal=TERM --kill-after=60s "$((ep20_left+30))s" \
  python3.12 -u -B fast_solver_reference_continue.py --source "$ep20_base/$1" \
    --run-dir "$ep20_run" --deadline-seconds "$ep20_left" > "$ep20_run/stdout.txt" 2>&1
ep20_exit=$?
printf 'exit=%s\nend=%s\n' "$ep20_exit" "$(date -Is)" >> "$ep20_run/status.txt"
if [[ ! -f "$ep20_run/reference_report.json" ]]; then
  printf '{"status":"failed","failure":"no terminal Python report","exit":%s,"slurm_job_id":"%s"}\n' \
    "$ep20_exit" "$SLURM_JOB_ID" > "$ep20_run/reference_report_fallback.json"
fi
if [[ "$ep20_exit" == 0 ]]; then
  # Finite saved-frame adjudication, with no polling, fitting or automatic scaling.
  timeout --signal=TERM --kill-after=10s 120s python3.12 -u -B fast_solver_compare.py \
    --reference "$ep20_run" --solver "$ep20_base/dev_d3/step3_offsetsolve" \
    --output "$ep20_run/offset_equivalence.json" > "$ep20_run/offset_equivalence.txt" 2>&1
  ep20_compare_exit=$?
  if [[ "$ep20_compare_exit" == 0 ]]; then
    python3.12 - "$ep20_run/offset_equivalence.json" <<'CHECK'
import json, sys
r = json.load(open(sys.argv[1]))
assert set(r['windows']) == {'5us', 'dwell', 'frame'}
assert all(v.get('passed') is True for v in r['windows'].values()), 'Offset equivalence incomplete or failed'
CHECK
    ep20_compare_exit=$?
  fi
  printf 'offset_comparison_exit=%s\n' "$ep20_compare_exit" >> "$ep20_run/status.txt"
  [[ "$ep20_compare_exit" == 0 ]] || ep20_exit=3
fi
exit "$ep20_exit"
