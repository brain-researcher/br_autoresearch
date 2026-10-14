#!/bin/bash
#SBATCH --job-name=ep20-continuous-10
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage2/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage2/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_reference="$ep20_root/outputs/runs/fast_forward_stage1/solver/ref_frame_standard"
ep20_predecessor="$ep20_root/outputs/runs/fast_forward_stage1/solver/continuous_frame005/equivalence.json"
ep20_run="$ep20_root/outputs/runs/fast_forward_stage2/solver/continuous_10frames001"
mkdir "$ep20_run"
ep20_phase=setup
trap 'ep20_exit=$?; printf "{\"phase\":\"%s\",\"exit_code\":%s,\"finished_utc\":\"%s\",\"scientific_scoring\":false,\"automatic_successor\":false,\"clipping_equivalence_asserted\":false}\n" "$ep20_phase" "$ep20_exit" "$(date -u +%FT%TZ)" > "$ep20_run/terminal.json"' EXIT
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
ep20_phase=qualified_standard_predecessor
python3.12 -B - "$ep20_predecessor" <<'PREDECESSOR'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report['qualified'] is True
assert report['all_physical_windows_passed'] is True
PREDECESSOR
# This is an engineering continuity measurement. The separate original
# offset-reference comparison remains required for clipping qualification.
# The 7-GiB request and 6656-MiB guard use measured 4.22-GiB peak RSS; the
# earlier blanket half-request headroom rule is not a numerical tolerance.
ep20_phase=continuous_ten_frames
mkdir "$ep20_run/standard"
ep20_left=$((ep20_deadline-$(date +%s)-150))
(( ep20_left > 60 ))
/usr/bin/time -v -o "$ep20_run/resources.txt" timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B fast_solver_continuous.py --run-dir "$ep20_run/standard" --frames 10 \
 --deadline-seconds "$ep20_left" --memory-limit-mib 6656 --cache-mib 1024 \
 > "$ep20_run/stdout.txt" 2>&1
ep20_phase=first_frame_equivalence
# The comparator CLI is restricted to Stage 1 directories; its existing
# array/archive API applies unchanged tolerances to this Stage 2 prefix.
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B - \
 "$ep20_reference" "$ep20_run/standard/first_frame" "$ep20_run/equivalence.json" \
 > "$ep20_run/equivalence.txt" 2>&1 <<'COMPARE'
import json, sys
from pathlib import Path
from conditional_forward_recipe import expected_state_ids
from fast_solver_fastpath_compare import compare
reference, solver, destination = map(Path, sys.argv[1:])
result = compare(reference, solver, state_ids=expected_state_ids())
result.update(reference=str(reference), solver=str(solver), comparator='fastpath_complete_window_work')
with destination.open('x') as stream:
    json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result, sort_keys=True, allow_nan=False))
raise SystemExit(0 if result['qualified'] else 2)
COMPARE
ep20_phase=complete_ten_frame_continuity_no_scoring
