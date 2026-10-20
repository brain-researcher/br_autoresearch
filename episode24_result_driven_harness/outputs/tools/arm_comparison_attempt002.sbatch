#!/bin/bash -l
# EP24 attempt002 stage job. Submit with --export=NONE:
#   sbatch ... --output=<run_dir>/slurm-%j.out arm_comparison_attempt002.sbatch screen <run_dir>
#   sbatch ... --dependency=afterok:<screen job> ... arm_comparison_attempt002.sbatch test <run_dir>
# screen: old-arm screening draws, both raters, then the retention rule (fails if < 8 retained)
# test:   fresh draws for every arm on retained items, both raters, then the analysis
set -eo pipefail
stage=$1
run_dir=$2
ep=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode24_result_driven_harness
mat=$ep/outputs/arm_comparison/attempt002
for v in $(env | sed -n 's/^\(CLAUDE[A-Z_]*\|ANTHROPIC[A-Z_]*\|USE_LOCAL_OAUTH\|USE_STAGING_OAUTH\)=.*/\1/p'); do
  unset "$v"
done
ml devel nodejs/24.13.0
export PATH="$HOME/.local/bin:$PATH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} stage=$stage start=$(date -Is) codex=$(codex --version 2>&1)"
work="$SCRATCH/br_autoresearch/episode24_result_driven_harness/work"
drive() {
  python3 -I "$ep/outputs/tools/arm_comparison.py" "$1" --materials "$mat" --run-dir "$run_dir" \
    --scratch "$work" --plan "$2" "${@:3}"
}
if [ "$stage" = screen ]; then
  drive generate plan_screen.json
  drive rate plan_screen.json --rater claude
  drive rate plan_screen.json --rater astra
  python3 -I "$ep/outputs/tools/arm_comparison_attempt002_items.py" retain --run-dir "$run_dir" --materials "$mat"
else
  drive generate plan.json
  drive rate plan.json --rater claude
  drive rate plan.json --rater astra
  ml viz py-matplotlib/3.10.3_py312
  mkdir -p "$ep/outputs/figures/arm_comparison_attempt002"
  python3 -P "$ep/outputs/tools/arm_comparison_analysis.py" run --materials "$mat" --run-dir "$run_dir" \
    --raters ratings,ratings_astra --results "$ep/outputs/arm_comparison/results/attempt002" \
    --figure "$ep/outputs/figures/arm_comparison_attempt002/paired_targeting.png"
fi
echo "end=$(date -Is)"
