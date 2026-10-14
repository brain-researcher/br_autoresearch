#!/bin/bash
#SBATCH --job-name=ep20-continuous-frame
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname)" != *-ln* ]]
[[ "${SLURM_JOB_CPUS_PER_NODE:-}" =~ ^1(\(x1\))?$ ]]
ep20_root="${OAK:?}/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign"
ep20_base="$ep20_root/outputs/runs/fast_forward_stage1/solver"
ep20_run="$ep20_base/continuous_frame005"
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
(( ep20_deadline-$(date +%s) > 480 ))
ep20_phase=predecessor_standard_equivalence
python3.12 -B - "$ep20_base/continuous_frame004/equivalence.json" <<'PREDECESSOR'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
assert report['qualified'] is True
assert report['all_physical_windows_passed'] is True
PREDECESSOR
# Both affected independent energy methods passed in audit_response002 on
# Slurm step 47377474.7. Reuse that evidence; candidate source remains frozen.
ep20_phase=changed_program_validation
ep20_astra="${GROUP_HOME:?}/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv"
ep20_deps="$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156"
env PYTHONPATH="$ep20_astra/lib/python3.12/site-packages:$ep20_deps/lib/python3.12/site-packages" timeout --signal=TERM --kill-after=5s 30s "$ep20_astra/bin/astra-tools" validate "$ep20_root/outputs/astra/v0.0.14/astra.yaml" --json > "$ep20_run/astra.txt" 2>&1 || true
ep20_phase=continuous_standard_frame
mkdir "$ep20_run/standard"
ep20_left=$((ep20_deadline-$(date +%s)-150))
(( ep20_left > 60 ))
/usr/bin/time -v -o "$ep20_run/resources.txt" timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B fast_solver_continuous.py --run-dir "$ep20_run/standard" --frames 1 \
 --deadline-seconds "$ep20_left" --memory-limit-mib 6656 --cache-mib 1024 \
 > "$ep20_run/stdout.txt" 2>&1
ep20_phase=continuous_first_frame_equivalence
timeout --signal=TERM --kill-after=5s 120s python3.12 -u -B fast_solver_fastpath_compare.py \
 --reference "$ep20_base/ref_frame_standard" --solver "$ep20_run/standard/first_frame" \
 --output "$ep20_run/equivalence.json" > "$ep20_run/equivalence.txt" 2>&1
ep20_phase=audit_arithmetic_comparison
timeout --signal=TERM --kill-after=5s 30s python3.12 -B - "$ep20_base/continuous_frame004/standard" "$ep20_run/standard" "$ep20_run/audit_comparison.json" <<'AUDIT'
import json, math, sys
from pathlib import Path
old_dir, new_dir, destination = map(Path, sys.argv[1:])
old = json.loads((old_dir / 'first_frame/solver_report.json').read_text())
new = json.loads((new_dir / 'first_frame/solver_report.json').read_text())
windows = {}
for name in ('5us', 'dwell', 'frame'):
    differences = {}
    for field in ('input_work', 'heat', 'residual', 'relative_residual'):
        before = old['window_energy_audit'][name][field]
        after = new['window_energy_audit'][name][field]
        assert math.isfinite(before) and math.isfinite(after)
        differences[field] = dict(before=before, after=after, difference=after-before)
    windows[name] = differences
result = dict(role='descriptive_audit_arithmetic_comparison_no_new_acceptance_threshold',
              previous=str(old_dir), current=str(new_dir), windows=windows)
for label, directory in (('previous', old_dir), ('current', new_dir)):
    report = json.loads((directory / 'solver_report.json').read_text())
    result[label + '_performance'] = {key: report[key] for key in
        ('elapsed_seconds', 'mean_frame_seconds', 'component_timings_seconds', 'peak_rss_kib')}
with destination.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
AUDIT
ep20_phase=continuous_function_profile
mkdir "$ep20_run/profile"
ep20_left=$((ep20_deadline-$(date +%s)-60))
(( ep20_left > 60 ))
/usr/bin/time -v -o "$ep20_run/profile_resources.txt" timeout --signal=TERM --kill-after=30s "$((ep20_left+20))s" \
 python3.12 -u -B -m cProfile -o "$ep20_run/function_profile.pstats" fast_solver_continuous.py \
 --run-dir "$ep20_run/profile" --frames 1 --deadline-seconds "$ep20_left" \
 --memory-limit-mib 6656 --cache-mib 1024 > "$ep20_run/profile_stdout.txt" 2>&1
python3.12 -B - "$ep20_run/function_profile.pstats" > "$ep20_run/function_profile.txt" <<'PROFILE'
import pstats, sys
stats = pstats.Stats(sys.argv[1]).strip_dirs()
stats.sort_stats('cumulative').print_stats(45)
stats.sort_stats('tottime').print_stats(45)
PROFILE
python3.12 -B - "$ep20_run/function_profile.pstats" > "$ep20_run/function_profile_callers.txt" <<'CALLERS'
import pstats, sys
stats = pstats.Stats(sys.argv[1]).strip_dirs().sort_stats('cumulative')
stats.print_callers('einsumfunc.py:1009')
stats.print_callees('upstream_energy_audit')
stats.print_stats('_audit_unit_response')
CALLERS
ep20_phase=complete_standard_integration_no_scaling
