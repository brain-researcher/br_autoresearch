#!/bin/bash
set -euo pipefail
[[ $# == 2 ]] || exit 2
ep04_control_mode=$1
ep04_control_run=$2
ep04_control_host=$(hostname)
printf 'HOST=%s JOB=%s STEP=%s CPUS=%s MODE=%s\n' "$ep04_control_host" \
  "${SLURM_JOB_ID:?}" "${SLURM_STEP_ID:?}" "${SLURM_CPUS_PER_TASK:?}" "$ep04_control_mode"
[[ "$ep04_control_host" != *-ln* && "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "${OAK:?}/users/zijiao/br_autoresearch/episode04_shared_scene_geometry"
ep04_control_observed=outputs/runs/observed_native_panel/20261005-observed01/observed_native_panel.json
ep04_control_low=outputs/runs/source_readiness/20261005-low_level_full01
case "$ep04_control_mode" in
  sensitivities)
    set +e
    timeout --signal=TERM --kill-after=10s 2460s python3.12 -u -B \
      outputs/code/run_native_sensitivities.py \
      --observed-report "$ep04_control_observed" --low-level-dir "$ep04_control_low" \
      --output-dir outputs/runs/native_sensitivities/20261005-all01 --max-seconds 2400
    ep04_sensitivity_status=$?
    timeout --signal=TERM --kill-after=10s 160s bash \
      "$OAK/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/tools/render_observed_outgoing_description.sh" \
      > "$ep04_control_run/ep12_description_figure.log" 2>&1
    ep12_figure_status=$?
    (
      cd "$OAK/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark" || exit 2
      timeout --signal=TERM --kill-after=10s 90s bash outputs/code/read_way_information_page01.sh
    ) > "$ep04_control_run/ep19_information_page01.log" 2>&1
    ep19_document_status=$?
    set -e
    printf 'sensitivities\t%s\nep12_description_figure\t%s\nep19_information_page01\t%s\n' \
      "$ep04_sensitivity_status" "$ep12_figure_status" "$ep19_document_status" > "$ep04_control_run/sensitivity_reporting_exit_status.tsv"
    cat "$ep04_control_run/sensitivity_reporting_exit_status.tsv"
    [[ "$ep04_sensitivity_status" == 0 && "$ep12_figure_status" == 0 && "$ep19_document_status" == 0 ]]
    ;;
  encoding_null)
    set +e
    timeout --signal=TERM --kill-after=10s 45s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_identical_fold_encoding.py -v \
      > "$ep04_control_run/encoding_checks.log" 2>&1
    ep04_encoding_checks=$?
    ep04_encoding_exit=125
    if [[ "$ep04_encoding_checks" == 0 ]]; then
      timeout --signal=TERM --kill-after=10s 1210s python3.12 -u -B \
        outputs/code/run_identical_fold_encoding.py --participant sub-01 --fold 0 \
        --observed-report "$ep04_control_observed" --low-level-dir "$ep04_control_low" \
        --output-dir outputs/runs/identical_fold_encoding/20261005-pilot01 --max-seconds 1200 \
        > "$ep04_control_run/encoding_pilot.log" 2>&1
      ep04_encoding_exit=$?
    fi
    timeout --signal=TERM --kill-after=10s 45s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_registered_null_unit.py -v \
      > "$ep04_control_run/null_checks.log" 2>&1
    ep04_null_checks=$?
    ep04_null_exit=125
    if [[ "$ep04_null_checks" == 0 ]]; then
      timeout --signal=TERM --kill-after=10s 1210s python3.12 -u -B \
        outputs/code/run_registered_null_unit.py --participant sub-01 --fold 0 \
        --permutation-range 0:256 --geometry-range 0:8 \
        --observed-report "$ep04_control_observed" --low-level-dir "$ep04_control_low" \
        --output-dir outputs/runs/registered_nulls/20261005-prefix01 --max-seconds 1200 \
        > "$ep04_control_run/null_prefix.log" 2>&1
      ep04_null_exit=$?
    fi
    timeout --signal=TERM --kill-after=10s 45s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_collect_registered_nulls.py -v \
      > "$ep04_control_run/null_collection_checks.log" 2>&1
    ep04_collection_checks=$?
    timeout --signal=TERM --kill-after=10s 45s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_collect_identical_fold_encoding.py -v \
      > "$ep04_control_run/encoding_collection_checks.log" 2>&1
    ep04_encoding_collection_checks=$?
    timeout --signal=TERM --kill-after=10s 45s python3.12 -B -m unittest discover \
      -s outputs/tests -p test_screen_registered_panel.py -v \
      > "$ep04_control_run/panel_screen_checks.log" 2>&1
    ep04_screen_checks=$?
    set -e
    printf 'encoding_checks\t%s\nencoding_pilot\t%s\nnull_checks\t%s\nnull_prefix\t%s\nnull_collection_checks\t%s\nencoding_collection_checks\t%s\npanel_screen_checks\t%s\n' \
      "$ep04_encoding_checks" "$ep04_encoding_exit" "$ep04_null_checks" "$ep04_null_exit" "$ep04_collection_checks" "$ep04_encoding_collection_checks" "$ep04_screen_checks" \
      > "$ep04_control_run/diagnostic_exit_status.tsv"
    cat "$ep04_control_run/diagnostic_exit_status.tsv"
    [[ "$ep04_encoding_checks" == 0 && "$ep04_encoding_exit" == 0 \
       && "$ep04_null_checks" == 0 && "$ep04_null_exit" == 0 && "$ep04_collection_checks" == 0 \
       && "$ep04_encoding_collection_checks" == 0 && "$ep04_screen_checks" == 0 ]]
    ;;
  *) exit 2 ;;
esac
