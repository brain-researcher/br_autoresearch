#!/bin/bash -l
# EP24 attempt003 stage job. Submit with --export=NONE and an explicit time limit:
#   sbatch ... --output=<run_dir>/slurm-%j.out arm_comparison_attempt003.sbatch <stage> <run_dir>
# items:  Codex picks and tags candidates; verified build of packets, key and clusters
# screen: old and directive screening draws, both raters, then the retention rule
# test:   fresh draws for every arm on retained items, both raters, then the
#         episode-level analysis
set -eo pipefail
stage=$1
run_dir=$2
ep=/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode24_result_driven_harness
mat=$ep/outputs/arm_comparison/attempt003
for v in $(env | sed -n 's/^\(CLAUDE[A-Z_]*\|ANTHROPIC[A-Z_]*\|USE_LOCAL_OAUTH\|USE_STAGING_OAUTH\)=.*/\1/p'); do
  unset "$v"
done
ml devel nodejs/24.13.0
export PATH="$HOME/.local/bin:$PATH"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} stage=$stage start=$(date -Is) codex=$(codex --version 2>&1)"
work="$SCRATCH/br_autoresearch/episode24_result_driven_harness/work"
items="$ep/outputs/tools/arm_comparison_attempt003_items.py"
drive() {
  python3 -I "$ep/outputs/tools/arm_comparison.py" "$1" --materials "$mat" --run-dir "$run_dir" \
    --scratch "$work" --plan "$2" "${@:3}"
}
case "$stage" in
  items)
    python3 -I "$items" extract --run-dir "$run_dir" --materials "$mat" --scratch "$work"
    python3 -I "$items" build --run-dir "$run_dir" --materials "$mat"
    ;;
  screen)
    drive generate plan_screen.json
    drive rate plan_screen.json --rater claude
    drive rate plan_screen.json --rater astra
    python3 -I "$items" retain --run-dir "$run_dir" --materials "$mat"
    ;;
  test|describe)
    # describe: the scientist's descriptive look at the retained gap items (effect sizes only)
    drive generate plan.json
    drive rate plan.json --rater claude
    drive rate plan.json --rater astra
    ml viz py-matplotlib/3.10.3_py312 > "$run_dir/ml_analysis.log" 2>&1
    mkdir -p "$ep/outputs/figures/arm_comparison_attempt003"
    mode=run
    [ "$stage" = describe ] && mode=describe
    python3 -P "$ep/outputs/tools/arm_comparison_analysis.py" "$mode" --materials "$mat" --run-dir "$run_dir" \
      --raters ratings,ratings_astra --clusters "$mat/clusters.json" \
      --results "$ep/outputs/arm_comparison/results/attempt003" \
      --figure "$ep/outputs/figures/arm_comparison_attempt003/paired_targeting.png"
    ;;
esac
echo "end=$(date -Is)"
