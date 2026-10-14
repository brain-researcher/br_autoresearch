#!/bin/bash
# EP20 Stage 1 item 1 (planner/readout): development tests only, not counted.
# One finite allocation: unit tests, the new O1 preflight measurement and the
# O1/O2 legacy-equivalence run. No electrical forward simulation, no requeue,
# no successor or automatic resubmission. Selection is recorded beforehand in
# outputs/runs/fast_forward_stage1/planner_readout/RECORD.md.
#SBATCH --job-name=ep20-fastio-s1
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/planner_readout/slurm-%j.out
#SBATCH --error=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/planner_readout/slurm-%j.err
set -uo pipefail

job_started=$(date +%s)
time_limit_seconds=7200
deadline_seconds=$(( time_limit_seconds * 9 / 10 ))
host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s START=%s\n' "$host" "${SLURM_JOB_ID:-}" \
  "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

episode=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign
run_root="$episode/outputs/runs/fast_forward_stage1/planner_readout"
job_dir="$run_root/job_${SLURM_JOB_ID:-unknown}"
seed=20261012
tests_exit=not_run
preflight_exit=not_run
equivalence_exit=not_run
stage=startup

write_terminal() {
  local code=$1
  local now
  now=$(date +%s)
  [[ -d "$job_dir" ]] || return 0
  printf '{\n  "job_id": "%s",\n  "hostname": "%s",\n  "exit_code": %s,\n  "stage": "%s",\n  "tests_exit": "%s",\n  "preflight_exit": "%s",\n  "equivalence_exit": "%s",\n  "elapsed_seconds": %s,\n  "internal_deadline_seconds": %s,\n  "time_limit_seconds": %s,\n  "seed": %s,\n  "finished_utc": "%s",\n  "electrical_forward_simulation": false,\n  "counted_formal_run": false\n}\n' \
    "${SLURM_JOB_ID:-}" "$host" "$code" "$stage" "$tests_exit" "$preflight_exit" "$equivalence_exit" \
    "$(( now - job_started ))" "$deadline_seconds" "$time_limit_seconds" "$seed" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$job_dir/job_terminal.json.tmp" \
    && mv "$job_dir/job_terminal.json.tmp" "$job_dir/job_terminal.json"
}
trap 'code=$?; write_terminal "$code"' EXIT
trap 'stage="${stage}_terminated"; exit 143' TERM INT

[[ -n "${SLURM_JOB_ID:-}" && "$host" != *-ln* && "${SLURM_CPUS_PER_TASK:-}" == 1 ]] || exit 3
mkdir "$job_dir" || exit 4
cd "$episode" || exit 5

stage=environment
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
threads="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads" BLIS_NUM_THREADS="$threads"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:?}"
[[ -d "$TMPDIR" ]] || exit 6
module list > "$job_dir/modules.txt" 2>&1

remaining() { echo $(( deadline_seconds - ($(date +%s) - job_started) )); }

stage=unit_tests
budget=$(( $(remaining) < 900 ? $(remaining) : 900 ))
(
  cd outputs/tools || exit 2
  timeout --signal=TERM --kill-after=10s "${budget}s" python3.12 -B -m unittest -v \
    test_fast_io_exact test_fast_io_readout test_fast_io_planner
) > "$job_dir/tests.txt" 2>&1
tests_exit=$?
[[ "$tests_exit" == 0 ]] || exit 10

stage=new_preflight
budget=$(( $(remaining) - 60 ))
(( budget > 300 )) || exit 11
( cd outputs/tools && /usr/bin/time -v timeout --signal=TERM --kill-after=10s "$(( budget + 30 ))s" \
    python3.12 -B fast_io_preflight.py --run-dir "$job_dir" --max-seconds "$budget" ) \
  > "$job_dir/preflight_stdout.txt" 2> "$job_dir/preflight_resources.txt"
preflight_exit=$?

stage=equivalence
budget=$(( $(remaining) - 60 ))
(( budget > 600 )) || exit 12
( cd outputs/tools && /usr/bin/time -v timeout --signal=TERM --kill-after=10s "$(( budget + 30 ))s" \
    python3.12 -B fast_io_stage1_equivalence.py --run-dir "$job_dir" --max-seconds "$budget" \
    --seed "$seed" --random-windows 20 --readout-windows 8 ) \
  > "$job_dir/equivalence_stdout.txt" 2> "$job_dir/equivalence_resources.txt"
equivalence_exit=$?

stage=complete
[[ "$preflight_exit" == 0 && "$equivalence_exit" == 0 ]] || exit 20
exit 0
