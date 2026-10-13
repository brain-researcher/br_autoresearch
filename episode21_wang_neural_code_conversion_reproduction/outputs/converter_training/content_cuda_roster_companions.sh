# Sourced once by the original pair5 parent. No empirical access or submissions.
[[ "$ep21_roster_stage" == roster-005 ]] || exit 2
ep21_companion_h5_tests() {
  local directory=$1 seconds=$2 limit
  shift 2
  limit=$(ep21_roster_limit "$seconds") || return $?
  (
    ulimit -S -f 32768
    cd "$directory"
    env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 \
      PYTHONPATH="${GROUP_HOME:?}/users/zijiao/ep17_h5py_3_10_0_cp311" \
      timeout --signal=TERM --kill-after=5s "${limit}s" \
      "$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python" -B -m unittest -v "$@"
  )
}
ep21_companion_group() {
  local episode=$1 run=$2 label=$3 mode=$4 directory=$5 seconds=$6 result=0
  shift 6
  if [[ "$mode" == h5 ]]; then
    if ep21_companion_h5_tests "$directory" "$seconds" "$@" \
        > "$run/$label.txt" 2>&1; then :; else result=$?; fi
  else
    if ep21_roster_tests "$directory" "$seconds" "$@" \
        > "$run/$label.txt" 2>&1; then :; else result=$?; fi
  fi
  printf '%s_exit=%s\n' "$label" "$result" >> "$run/status.txt"
  if [[ "$result" -eq 137 ]]; then
    printf 'status=closed_parent_SIGKILL\n' >> "$run/status.txt"
    exit 137
  fi
  # Always enforce the original parent's clock, even after an ordinary failure.
  ep21_roster_limit 1 > /dev/null
}

ep17_roster_base="${OAK:?}/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability"
ep17_roster_run="$ep17_roster_base/outputs/runs/prediction_qualification/cuda_roster005_companion001"
if [[ -d "$ep17_roster_run" ]] && mkdir "$ep17_roster_run/batch_claim" && ( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charge_once=true\nempirical_access=false\nselected_unique_methods=49\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep17_roster_run/status.txt" ); then
  ep21_companion_group EP17 "$ep17_roster_run" endpoints_forecasts module "$ep17_roster_base/outputs/tools" 115 \
    test_development_endpoints test_development_forecasts
  ep21_companion_group EP17 "$ep17_roster_run" development_secondary module "$ep17_roster_base/outputs/tools" 175 \
    test_development_secondary
  ep21_companion_group EP17 "$ep17_roster_run" audit_prediction module "$ep17_roster_base/outputs/tools" 115 \
    test_audit_prediction
  ep21_companion_group EP17 "$ep17_roster_run" audit_secondary module "$ep17_roster_base/outputs/tools" 115 \
    test_audit_secondary
  (
    # This existing fixture chooses L_SCRATCH_JOB instead of TMPDIR. Keep its
    # disposable containers inside this parent's admitted1GiB scratch tree.
    export L_SCRATCH_JOB="$TMPDIR"
    ep21_companion_group EP17 "$ep17_roster_run" predictor_checkpoint module "$ep17_roster_base/outputs/tools" 115 \
      test_predictor_checkpoint
  )
  ep21_companion_group EP17 "$ep17_roster_run" development_responses h5 "$ep17_roster_base/outputs/tools" 115 \
    test_development_responses
  ep21_companion_group EP17 "$ep17_roster_run" prediction_cost module "$ep17_roster_base/outputs/tools" 115 \
    test_development_prediction_cost
  ep21_companion_group EP17 "$ep17_roster_run" development_features h5 "$ep17_roster_base/outputs/tools" 115 \
    test_development_features
  ep21_companion_group EP17 "$ep17_roster_run" audit_responses h5 "$ep17_roster_base/outputs/tools" 115 \
    test_audit_responses
  ep17_roster_astra_rc=0
  if ( cd "$ep17_roster_base" && ep21_roster_validate ) \
      > "$ep17_roster_run/astra_validation.json" 2>&1; then :; else ep17_roster_astra_rc=$?; fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep17_roster_astra_rc" >> "$ep17_roster_run/status.txt"
  [[ "$ep17_roster_astra_rc" -ne 137 ]] || exit 137
else
  printf 'EP17_companion_setup=unavailable\n' >> "$ep21_roster_run/status.txt"
fi

ep19_roster_base="${OAK:?}/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark"
ep19_roster_run="$ep19_roster_base/outputs/runs/reference_panel/cuda_roster005_prediction_companion001"
if [[ -d "$ep19_roster_run" ]] && mkdir "$ep19_roster_run/batch_claim" && ( set -o noclobber
  printf 'parent_episode=EP21\nparent_job=%s\nparent_charge_once=true\nempirical_access=false\nselected_methods=3\nstatus=started\n' \
    "$SLURM_JOB_ID" > "$ep19_roster_run/status.txt" ); then
  ep21_companion_group EP19 "$ep19_roster_run" development_prediction module "$ep19_roster_base/outputs/tests" 175 \
    test_reference_development_prediction
  ep19_roster_astra_rc=0
  if ( cd "$ep19_roster_base" && ep21_roster_validate ) \
      > "$ep19_roster_run/astra_validation.json" 2>&1; then :; else ep19_roster_astra_rc=$?; fi
  printf 'astra_exit=%s\nstatus=closed\n' "$ep19_roster_astra_rc" >> "$ep19_roster_run/status.txt"
  [[ "$ep19_roster_astra_rc" -ne 137 ]] || exit 137
else
  printf 'EP19_companion_setup=unavailable\n' >> "$ep21_roster_run/status.txt"
fi
ep21_roster_limit 1 > /dev/null
