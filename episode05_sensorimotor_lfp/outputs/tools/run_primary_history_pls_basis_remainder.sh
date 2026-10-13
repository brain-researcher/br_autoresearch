#!/bin/bash
# Selected from the completed 22.868-second pilot; use the owned compute shell.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" != *-ln* ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 1 ]]
EP_BASIS_PYTHON=$(command -v python3.12)
cd "$OAK/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp"
timeout --signal=TERM --kill-after=10s 1560s \
    "$EP_BASIS_PYTHON" -B outputs/tools/run_primary_history_pls_basis_panel.py remainder \
    --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
    --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
    --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
    --hw-bank-root outputs/runs/adaptive_search/primary_history_repeatability_weight_control_attempt002 \
    --k-bank-root outputs/runs/adaptive_search/primary_capacity_attempt001 \
    --blend-bank-root outputs/runs/adaptive_search/primary_history_blend_attempt001 \
    --svd-bank-root outputs/runs/adaptive_search/primary_covariance_svd_control_attempt002 \
    --rbf-bank-root outputs/runs/adaptive_search/primary_history_rbf_attempt001 \
    --pls-bank-root outputs/runs/adaptive_search/primary_history_pls_attempt001 \
    --hw-summary outputs/runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json \
    --k-summary outputs/runs/adaptive_search/primary_capacity_summary_attempt001/primary_capacity.json \
    --blend-summary outputs/runs/adaptive_search/primary_history_blend_summary_attempt001/primary_history_blend.json \
    --svd-summary outputs/runs/adaptive_search/primary_covariance_svd_summary_attempt001/primary_covariance_svd.json \
    --rbf-summary outputs/runs/adaptive_search/primary_history_rbf_summary_attempt001/primary_history_rbf.json \
    --pls-summary outputs/runs/adaptive_search/primary_history_pls_summary_attempt001/primary_history_pls.json \
    --bank-root outputs/runs/adaptive_search/primary_history_pls_basis_attempt001 \
    --summary-out outputs/runs/adaptive_search/primary_history_pls_basis_summary_attempt001 \
    --out outputs/runs/adaptive_search/primary_history_pls_basis_remainder_attempt001 \
    --workers 1 --max-seconds 1500
