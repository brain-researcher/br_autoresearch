#!/bin/bash
# Stage 1 development job for the fast solver (tests and/or full-size windows).
# Usage: sbatch [--time=...] run_fast_solver_dev.sbatch RUN_SUBDIR STEPS
#   STEPS: comma list of
#     tests[:Pattern]              unit tests (test_fast_solver_core, optional -k pattern)
#     solve:SLOTS:MODE[:GRADE]     fast solver on the first SLOTS actions (MODE ref_fed|exact)
#     compare:REFDIR:SOLVEDIR      acceptance comparison (paths relative to the solver run root)
#     offsetsolve:SLOTS:MODE:JSON  fast solver with a synthetic offset fixture
#     diff:REFDIR:SOLVEA:SOLVEB    report-only differences between two solver runs
# No requeue, no automatic resubmission. Internal deadlines at 90% of the limit.
#SBATCH --job-name=ep20-fs-dev
#SBATCH --partition=russpold,normal
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/solver/slurm-%j.out
#SBATCH --error=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/runs/fast_forward_stage1/solver/slurm-%j.err
set -uo pipefail
ep20_host=$(hostname)
printf 'HOST=%s JOB=%s CPUS=%s MEM_MIB=%s START=%s\n' "$ep20_host" "${SLURM_JOB_ID:?}" \
  "${SLURM_CPUS_PER_TASK:?}" "${SLURM_MEM_PER_NODE:-unknown}" "$(date -Is)"
[[ "$ep20_host" != *-ln* ]] || exit 2
[[ $# -eq 2 ]] || exit 2
ep20_root=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign
ep20_base="$ep20_root/outputs/runs/fast_forward_stage1/solver"
ep20_run="$ep20_base/$1"
mkdir "$ep20_run" || exit 2
source /share/software/user/open/lmod/lmod/init/bash
module purge
unset PYTHONPATH PYTHONHOME
module load devel python/3.12.1 math py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VECLIB_MAXIMUM_THREADS="$SLURM_CPUS_PER_TASK" BLIS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="${L_SCRATCH_JOB:-/tmp}"
ep20_limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
ep20_seconds=$(python3 -c "import sys;p=[int(x) for x in sys.argv[1].replace('-',':').split(':')];print(sum(v*m for v,m in zip(reversed(p),(1,60,3600,86400))))" "$ep20_limit")
ep20_start=$(date +%s)
ep20_deadline_at=$(( ep20_start + ep20_seconds * 9 / 10 ))
printf 'job=%s\nhost=%s\nlimit_seconds=%s\nsteps=%s\n' "$SLURM_JOB_ID" "$ep20_host" "$ep20_seconds" "$2" \
  > "$ep20_run/status.txt"
cd "$ep20_root/outputs/tools" || exit 2
ep20_overall=0
ep20_step_index=0
IFS=',' read -r -a ep20_steps <<< "$2"
for ep20_step in "${ep20_steps[@]}"; do
  ep20_step_index=$(( ep20_step_index + 1 ))
  ep20_now=$(date +%s)
  ep20_left=$(( ep20_deadline_at - ep20_now ))
  if (( ep20_left < 60 )); then
    printf 'step%s=%s skipped_no_time\n' "$ep20_step_index" "$ep20_step" >> "$ep20_run/status.txt"
    ep20_overall=3
    continue
  fi
  IFS=':' read -r -a ep20_part <<< "$ep20_step"
  ep20_kind="${ep20_part[0]}"
  ep20_out="$ep20_run/step${ep20_step_index}_${ep20_kind}"
  case "$ep20_kind" in
    tests)
      ep20_pattern=()
      [[ ${#ep20_part[@]} -ge 2 ]] && ep20_pattern=(-k "${ep20_part[1]}")
      FAST_SOLVER_TEST_JSON="$ep20_out.json" \
        timeout --signal=TERM --kill-after=20s "${ep20_left}s" \
        python3.12 -u -B -m unittest -v test_fast_solver_core ${ep20_pattern[@]+"${ep20_pattern[@]}"} \
        > "$ep20_out.txt" 2>&1
      ep20_exit=$?
      ;;
    solve|offsetsolve)
      mkdir "$ep20_out"
      ep20_extra=()
      [[ "$ep20_kind" == offsetsolve ]] && ep20_extra=(--offsets-json "${ep20_part[3]}")
      [[ "$ep20_kind" == solve && ${#ep20_part[@]} -ge 4 ]] && ep20_extra=(--grade "${ep20_part[3]}")
      /usr/bin/time -v -o "$ep20_out/resources.txt" \
        timeout --signal=TERM --kill-after=30s "${ep20_left}s" \
        python3.12 -u -B fast_solver_stage1.py --run-dir "$ep20_out" --slots "${ep20_part[1]}" \
          --mode "${ep20_part[2]}" --deadline-seconds "$(( ep20_left - 30 ))" \
          --checkpoint-slots "2,$(seq -s, 25 25 1600)" ${ep20_extra[@]+"${ep20_extra[@]}"} \
          > "$ep20_out/stdout.txt" 2>&1
      ep20_exit=$?
      ;;
    compare)
      timeout --signal=TERM --kill-after=20s "${ep20_left}s" \
        python3.12 -u -B fast_solver_compare.py --reference "$ep20_base/${ep20_part[1]}" \
          --solver "$ep20_base/${ep20_part[2]}" --output "$ep20_out.json" > "$ep20_out.txt" 2>&1
      ep20_exit=$?
      ;;
    diff)
      timeout --signal=TERM --kill-after=20s "${ep20_left}s" \
        python3.12 -u -B fast_solver_compare.py --reference "$ep20_base/${ep20_part[1]}" \
          --solver "$ep20_base/${ep20_part[2]}" --solver-b "$ep20_base/${ep20_part[3]}" \
          --output "$ep20_out.json" > "$ep20_out.txt" 2>&1
      ep20_exit=$?
      ;;
    *)
      ep20_exit=2
      ;;
  esac
  printf 'step%s=%s exit=%s end=%s\n' "$ep20_step_index" "$ep20_step" "$ep20_exit" "$(date -Is)" \
    >> "$ep20_run/status.txt"
  [[ "$ep20_exit" == 0 ]] || ep20_overall=1
done
printf 'overall=%s end=%s\n' "$ep20_overall" "$(date -Is)" >> "$ep20_run/status.txt"
printf '{"slurm_job_id": "%s", "host": "%s", "steps": "%s", "overall_exit": %s}\n' "$SLURM_JOB_ID" \
  "$ep20_host" "$2" "$ep20_overall" > "$ep20_run/terminal.json"
exit "$ep20_overall"
