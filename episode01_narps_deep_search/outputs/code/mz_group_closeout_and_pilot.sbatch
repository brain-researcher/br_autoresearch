#!/bin/bash -l
#SBATCH --job-name=ep01-mz-close
#SBATCH --partition=normal
#SBATCH --constraint=CPU_SKU:7502
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:35:00
#SBATCH --no-requeue
#SBATCH --dependency=afterok:46539590
#SBATCH --output=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-closeout-pilot-%j.out
#SBATCH --error=outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-closeout-pilot-%j.err

# Finite conditional scientific chain. Root submits only afterok:46539590;
# earlier fixed MZ0 and array46538652 are already terminal. No monitoring,
# resubmission, retry, interval calculation or follow-on draw is performed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if (( $# != 0 )); then
  printf 'This fixed MZ closeout chain takes no arguments\n' >&2
  exit 2
fi
module purge
module load devel python/3.12.1 math py-numpy/1.26.3 py-scikit-learn/1.5.1_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
EP_PYTHON=$(command -v python3.12)
hostname
printf 'SLURM_JOB_ID=%s CPUS=%s FIXED_DEPENDENCY=afterok:46539590\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK"
"$EP_PYTHON" --version
exec timeout --signal=TERM --kill-after=10s 5400s "$EP_PYTHON" -B - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("outputs/runs/behavioral_mechanism_kill_test")
source = root / "20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json"
reports = [source]
reports += [root / f"20261005-MZ-SESOI-array-46538652/replicate{i}/single_process_recovery.json" for i in range(1, 21)]
reports += [root / f"20261005-MZ-SESOI-array-46539590/replicate{i}/single_process_recovery.json" for i in range(21, 200)]
output = root / f"20261005-MZ-closeout-pilot-{os.environ['SLURM_JOB_ID']}"
output.mkdir()
implementation = "outputs/code/behavioral_implementation.json"
report_arguments = [argument for path in reports for argument in ("--report", str(path))]
subprocess.run([sys.executable, "-B", "outputs/code/summarize_single_process_recovery.py",
                "--model", "MZ", "--implementation", implementation,
                *report_arguments, "--output", str(output / "necessary_summary.json")], check=True)
subprocess.run([sys.executable, "-B", "outputs/code/summarize_mz_group_recovery.py",
                "--implementation", implementation,
                *report_arguments, "--output", str(output / "group_summary.json")], check=True)
group = json.loads((output / "group_summary.json").read_text())
ready = (group.get("group_point_bias_complete") is True
         and group.get("status") == "complete_fixed_cell_point_bias"
         and group.get("statistics_provisional") is False
         and group.get("planned_cell_replicates") == 200
         and group.get("valid_replicates") == 200
         and group.get("remaining_unique_valid_replicates") == 0
         and group.get("both_versions_relative_bias_pass") is True)
if not ready:
    print("PILOT NOT RUN: the complete200 point-bias criterion did not pass for both EI and ER.", flush=True)
    sys.exit(0)
print("Complete200 EI/ER point-bias criterion passed; executing only selected bootstrap draw0. No CI/coverage claim.", flush=True)
subprocess.run([sys.executable, "-B", "outputs/code/run_mz_bootstrap_pilot.py",
                "--source-report", str(source), "--implementation", implementation,
                "--output-dir", str(output / "bootstrap000"), "--max-seconds", "5400"], check=True)
PY
