#!/usr/bin/env bash
#SBATCH --job-name=ep10-v3-m0
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=russpold,normal

set -euo pipefail
umask 007

: "${SCRATCH:?SCRATCH is required}"
: "${OAK:?OAK is required}"
: "${GROUP_HOME:?GROUP_HOME is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

ml purge
ml load devel python/3.12.1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python_bin="$GROUP_HOME/users/zijiao/envs/ep10_targets_py312/bin/python"
project="$OAK/users/zijiao/br_autoresearch/episode10_single_cell_coprojection"
stage0="$project/outputs/stage0"
engine="$stage0/tools/fit_v3_m0.py"
contract="$stage0/v3_m0_model_contract.yaml"
scratch_root="$SCRATCH/br_autoresearch/episode10_single_cell_coprojection"
target_root="$scratch_root/gao_mop_development_targets_v3_ep10_assets"
interface_root="$scratch_root/v3_model_interface_preflight_v1"
output_root="$scratch_root/v3_m0_fit_v1"

for required in \
  "$python_bin" \
  "$engine" \
  "$contract" \
  "$target_root/primary_16_target_calls.csv" \
  "$target_root/v3_gate_summary.json" \
  "$interface_root/v3_model_interface.csv" \
  "$interface_root/v3_model_interface_summary.json"; do
  test -f "$required"
done

if [[ -e "$output_root" ]]; then
  printf 'ERROR: refusing to overwrite M0 output: %s\n' "$output_root" >&2
  exit 3
fi

# Fail closed on the two immediately preceding development-only gates before
# the model engine is allowed to open the recorded-positive call table.
"$python_bin" - "$target_root/v3_gate_summary.json" "$interface_root/v3_model_interface_summary.json" <<'PY'
import json
import sys
from pathlib import Path

target = json.loads(Path(sys.argv[1]).read_text())
interface = json.loads(Path(sys.argv[2]).read_text())
if not (
    target.get("status") == "pass"
    and target.get("eligible_cells") == 194
    and target.get("final_or_unassigned_files_opened") is False
    and interface.get("status") == "pass"
    and interface.get("interface_rows") == 194
    and interface.get("projection_outcomes_opened_or_emitted") is False
    and interface.get("final_or_unassigned_files_opened") is False
):
    raise SystemExit("upstream v3 target/interface gate is not passed and sealed")
print("upstream-v3-gates-pass")
PY

"$python_bin" --version
"$python_bin" "$engine" --self-test --contract "$contract"
"$python_bin" "$engine" \
  --execute \
  --contract "$contract" \
  --target-root "$target_root" \
  --interface-root "$interface_root" \
  --output-root "$output_root" \
  --temperature-diagnostic
