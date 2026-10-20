#!/bin/bash -l
# EP24 attempt002: Codex picks each episode's resume-point entry, then the
# picks are verified and built into packets and key. Submit with --export=NONE:
#   sbatch -p <partition> -t <hh:mm:ss> -c 4 --mem=8G --export=NONE \
#     --output=<run_dir>/slurm-%j.out arm_comparison_attempt002_items.sbatch <run_dir> <materials_dir>
set -eo pipefail
ep=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode24_result_driven_harness
for v in $(env | sed -n 's/^\(CLAUDE[A-Z_]*\|ANTHROPIC[A-Z_]*\|USE_LOCAL_OAUTH\|USE_STAGING_OAUTH\)=.*/\1/p'); do
  unset "$v"
done
ml devel nodejs/24.13.0
export PATH="$HOME/.local/bin:$PATH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) codex=$(codex --version 2>&1)"
items="$ep/outputs/tools/arm_comparison_attempt002_items.py"
python3 -I "$items" extract --run-dir "$1" --materials "$2" \
  --scratch "$SCRATCH/br_autoresearch/episode24_result_driven_harness/work"
python3 -I "$items" build --run-dir "$1" --materials "$2"
echo "end=$(date -Is)"
