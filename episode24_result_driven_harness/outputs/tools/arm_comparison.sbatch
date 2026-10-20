#!/bin/bash -l
# EP24 arm comparison job. Submit with an explicit time limit and a clean
# environment, writing the Slurm log into the attempt directory:
#   sbatch -p <partition> -t <hh:mm:ss> -c <cpus> --mem=<mem> --export=NONE \
#     --output=<run_dir>/slurm-%j.out arm_comparison.sbatch <mode> <run_dir> [driver options]
set -eo pipefail
mode=$1
run_dir=$2
shift 2
ep=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode24_result_driven_harness
# The rater must use its own login, not an inherited desktop-session environment.
for v in $(env | sed -n 's/^\(CLAUDE[A-Z_]*\|ANTHROPIC[A-Z_]*\|USE_LOCAL_OAUTH\|USE_STAGING_OAUTH\)=.*/\1/p'); do
  unset "$v"
done
ml devel nodejs/24.13.0
export PATH="$HOME/.local/bin:$PATH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=$mode start=$(date -Is)"
echo "python=$(command -v python3) $(python3 --version 2>&1) codex=$(codex --version 2>&1)"
python3 -I "$ep/outputs/tools/arm_comparison.py" "$mode" \
  --materials "$ep/outputs/arm_comparison" \
  --run-dir "$run_dir" \
  --scratch "$SCRATCH/br_autoresearch/episode24_result_driven_harness/work" "$@"
echo "end=$(date -Is)"
