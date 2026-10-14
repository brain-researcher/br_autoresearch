#!/usr/bin/env python3
"""Released complete EP15 development predictions: one figure and three lines.

The coordinator records release in the existing experiment log before invoking
--release-completed-summary. This flag has no default and is not a new grant.
Operational completion is checked BEFORE reading the withheld payload. No
fitting, audit access, ranking, confidence interval or recovery claim occurs.
Runtime: Sherlock python/3.12.1, py-numpy/1.26.3_py312,
py-scipy/1.12.0_py312 and py-matplotlib/3.8.3_py312; one numerical thread.
The fitting Conda environment lacks matplotlib and is not the report runtime.
"""
import argparse
import json
import math
from pathlib import Path

import run_development_point_panel as runner


SCOPE = "conditional_observed_beta"
ROUTE = "observed_beta_development_descriptive"
SUMMARY_NAME = "point_summaries.withheld.json"
PANEL = dict(participants=12, Task_B_only_conditions=18, partitions=4, locations=188,
             variants=16, source_fits=24, calibration_panels=48,
             random_basis_draws_per_target=8, matched_warp_draws_per_variant_per_target=8,
             matched_warp_variants=4, condition_control_targets=12, identity_control_targets=12)
LIMITS = (
    "Out-of-fold Task-B-value prediction in twelve development participants, conditional on "
    "the fixed 188-location availability mask. Target Task B contributed availability to "
    "the support; its numerical responses did not fit that target's prediction. Participant "
    "points show variability, not independent fitted experiments or confidence intervals. "
    "LOPO training sets overlap and the identity-control donor cycle is fixed. Measurement "
    "error remains in observed beta loss. No winner, reliable-signal, recovery, superiority "
    "or mechanism claim; the original sealed-participant audit remains closed. Separation "
    "is procedural within the same account."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, max_bytes):
    require(path.is_file() and path.stat().st_size <= max_bytes, "missing or oversized JSON input")
    with path.open() as stream:
        value = json.load(stream)
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), "nonfinite JSON value")
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def completed_payload(run_dir):
    """Only operational metadata is opened until every completion guard passes."""
    operation = read_json(run_dir / "operational.json", runner.REPORT_BYTES)
    support = runner.OUTPUTS / "runs/observed_beta_development/finite_support001/selection.npz"
    expected = dict(status="completed_observed_beta_cost_panel", analysis_scope=SCOPE,
        route_id=ROUTE, publication_mode="withheld_cost_measurement",
        scientific_summary_withheld=True, private_summary_complete=True,
        point_summaries_published=False,
        complete_summary_filename=SUMMARY_NAME, complete_panel_required=True,
        audit_values_read=False, formal_class_decision=False, uncertainty_calibrated=False,
        target_availability_used_for_common_support=True,
        separation="procedural_same_account_not_enforced_or_blinded")
    require(all(operation.get(key) == value for key, value in expected.items()),
            "run is not a completed private observed-beta panel with the adopted boundaries")
    require(operation.get("complete_panel") == PANEL, "incomplete or different full panel")
    require(operation.get("derived_support_path") == str(support), "different spatial support")
    path = (run_dir / SUMMARY_NAME).resolve()
    require(path.parent == run_dir, "summary must remain in the selected run")
    require(path.stat().st_size == operation.get("point_summary_bytes"), "incomplete saved summary")
    return path, operation


def validate_summary(payload):
    """Validate the saved reporting axes and paired contrasts, without refitting."""
    import numpy as np
    from calibration_falsifiers import VARIANTS, identity_pairings
    from development_lopo import DEVELOPMENT_IDS, M2_VARIANTS, M3_VARIANTS
    from registered_endpoint_metrics import CONDITIONS, condition_weights

    require(set(payload) == {"analysis_scope", "observed_beta", "numerical_methods",
        "spatial_estimand", "assumption_conditional_cross_partition_statistics", "claim_limit",
        "recovery_ratios_or_class_decisions"}, "unknown completed-summary schema")
    require(payload["analysis_scope"] == SCOPE and payload["recovery_ratios_or_class_decisions"] is False
        and payload["claim_limit"] == "descriptive_observed_beta_prediction_in_twelve_development_participants_only",
        "different result scope or claim")
    spatial = payload["spatial_estimand"]
    require(spatial.get("locations") == 188 and spatial.get("availability_conditioned") is True
        and spatial.get("target_Task_B_availability_used") is True
        and spatial.get("comparison_with_original_208_location_loss_permitted") is False
        and spatial.get("selection_path") == str(runner.OUTPUTS / "runs/observed_beta_development/finite_support001/selection.npz"),
        "different availability-conditioned estimand")
    observed = payload["observed_beta"]
    require(set(observed) == {"analysis_scope", "route_id", "biological_n", "participant_ids",
        "condition_ids", "partition_run_ids", "voxel_ids", "condition_weights", "participant_weight",
        "partition_weight", "voxel_weight", "support_size", "contribution_axes", "contribution_definition",
        "aggregation", "interpretation", "uncertainty_calibrated", "formal_class_decision", "ordinary",
        "random_local_basis", "matched_random_warp", "calibration_controls"}, "unknown observed-beta schema")
    require(observed["analysis_scope"] == SCOPE and observed["route_id"] == ROUTE
        and observed["biological_n"] == 12 and observed["support_size"] == 188
        and observed["uncertainty_calibrated"] is False and observed["formal_class_decision"] is False,
        "different observed-beta panel")
    require(tuple(observed["participant_ids"]) == DEVELOPMENT_IDS
        and tuple(observed["condition_ids"]) == CONDITIONS
        and observed["partition_run_ids"] == [list(range(start, start+4)) for start in (1, 5, 9, 13)]
        and observed["contribution_axes"] == ["participant", "partition", "condition"]
        and len(observed["voxel_ids"]) == 188 and len(set(observed["voxel_ids"])) == 188,
        "different participant, condition, partition or location axes")
    require(observed["participant_weight"] == 1/12 and observed["partition_weight"] == 1/4
        and observed["voxel_weight"] == 1/188
        and np.array_equal(np.asarray(observed["condition_weights"]), condition_weights())
        and observed["contribution_definition"] == "mean_voxel_squared_error_after_separate_spatial_centering"
        and observed["aggregation"] == "equal_partitions_then_seven_domain_weighting_then_equal_participants"
        and observed["interpretation"] == "observed_beta_prediction_error_including_measurement_error",
        "different loss or weights")
    rows, arrays = {}, {}

    def add(name, value, difference=False):
        prefix = "mse_difference" if difference else "observed_mse"
        tensor_key, person_key, mean_key = "partition_condition_"+prefix, "participant_mean_"+prefix, "mean_"+prefix
        require(set(value) == {tensor_key, person_key, mean_key}, "unknown loss record")
        require(type(value[mean_key]) in (int, float), "saved loss mean must be numeric")
        tensor, people = np.asarray(value[tensor_key]), np.asarray(value[person_key])
        require(tensor.shape == (12, 4, 18) and people.shape == (12,)
            and tensor.dtype.kind in "fiu" and people.dtype.kind in "fiu"
            and np.isfinite(tensor).all() and np.isfinite(people).all(), "invalid loss axes or values")
        require(difference or np.all(tensor >= 0), "negative squared-error contribution")
        reduced = np.mean(np.sum(tensor*condition_weights()[None, None, :], axis=-1), axis=1)
        require(np.allclose(people, reduced, rtol=1e-12, atol=1e-12)
            and math.isclose(float(value[mean_key]), float(np.mean(people)), rel_tol=1e-12, abs_tol=1e-12),
            "saved means do not match their complete contributions")
        rows[name] = {"participant_values": people.tolist(), "mean": float(value[mean_key]),
                      "quantity": prefix}
        arrays[name] = tensor

    def difference(name, value, left, right):
        add(name, value, True)
        require(np.allclose(arrays[name], arrays[left]-arrays[right], rtol=1e-12, atol=1e-12),
                "paired contrast does not match its saved sources")

    ordinary = observed["ordinary"]
    require(set(ordinary) == {"source_training_ids", "losses", "model_minus_baseline"}, "unknown ordinary schema")
    expected_sources = [[person for person in DEVELOPMENT_IDS if person != target] for target in DEVELOPMENT_IDS]
    require(ordinary["source_training_ids"] == expected_sources, "ordinary source exclusions changed")
    require(set(ordinary["losses"]) == {"M0a", "M0b", *VARIANTS}
        and set(ordinary["model_minus_baseline"]) == set(VARIANTS), "incomplete model panel")
    for name in ("M0a", "M0b", *VARIANTS):
        add("ordinary/"+name, ordinary["losses"][name])
    for name in VARIANTS:
        contrasts = ordinary["model_minus_baseline"][name]
        require(set(contrasts) == {"model_minus_M0a", "model_minus_M0b"}, "missing baseline comparison")
        for baseline in ("M0a", "M0b"):
            difference("ordinary/"+name+"-"+baseline, contrasts["model_minus_"+baseline],
                       "ordinary/"+name, "ordinary/"+baseline)
    controls = observed["calibration_controls"]
    require(set(controls) == {"condition", "identity"}, "missing calibration control")
    donors = dict(identity_pairings())
    for label in ("condition", "identity"):
        control = controls[label]
        require(set(control) == {"source_training_ids", "true_packet_ids", "control_packet_ids", "variants", "comparison"},
                "unknown calibration-control schema")
        expected_donors = [donors[target] if label == "identity" else target for target in DEVELOPMENT_IDS]
        expected_training = ([[person for person in DEVELOPMENT_IDS if person not in (target, donors[target])]
                              for target in DEVELOPMENT_IDS] if label == "identity" else expected_sources)
        require(control["source_training_ids"] == expected_training
            and control["true_packet_ids"] == list(DEVELOPMENT_IDS)
            and control["control_packet_ids"] == expected_donors
            and control["comparison"] == ("donor_minus_true_within_paired_reduced_source" if label == "identity"
                                          else "relabeled_minus_true_within_ordinary_source")
            and set(control["variants"]) == set(VARIANTS), "changed paired-control source, cycle or panel")
        for name in VARIANTS:
            values, stem = control["variants"][name], label+"/"+name
            require(set(values) == {"true", "control", "control_minus_true"}, "unknown paired-loss schema")
            add(stem+"/true", values["true"])
            add(stem+"/control", values["control"])
            difference(stem+"/control_minus_true", values["control_minus_true"], stem+"/control", stem+"/true")
            if label == "condition":
                require(np.allclose(arrays[stem+"/true"], arrays["ordinary/"+name], rtol=1e-12, atol=1e-12),
                        "condition true loss must reuse its ordinary source")

    def draws(stem, value, true_name, extra=False):
        require(set(value) == ({"draw_losses", "mean_draw_loss", "mean_control_minus_true", "matched_true_variant"}
                              if extra else {"draw_losses", "mean_draw_loss", "mean_control_minus_true"}),
                "unknown matched-control schema")
        require(len(value["draw_losses"]) == 8, "all eight fixed draws are required")
        for index, draw in enumerate(value["draw_losses"]):
            add(stem+"/draw"+str(index), draw)
        add(stem+"/mean_draw_loss", value["mean_draw_loss"])
        require(np.allclose(arrays[stem+"/mean_draw_loss"], np.mean(
            [arrays[stem+"/draw"+str(index)] for index in range(8)], axis=0), rtol=1e-12, atol=1e-12),
            "matched losses must be averaged after per-draw scoring")
        difference(stem+"/control_minus_true", value["mean_control_minus_true"],
                   stem+"/mean_draw_loss", "ordinary/"+true_name)
    basis = observed["random_local_basis"]
    require(basis.get("matched_true_variant") == M3_VARIANTS[0], "changed random-basis comparison")
    draws("random_basis/"+M3_VARIANTS[0], basis, M3_VARIANTS[0], True)
    require(set(observed["matched_random_warp"]) == set(M2_VARIANTS), "incomplete matched-warp panel")
    for name in M2_VARIANTS:
        draws("matched_warp/"+name, observed["matched_random_warp"][name], name)
    return rows, tuple(VARIANTS), tuple(DEVELOPMENT_IDS)


def figure(rows, variants, output_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from development_lopo import M2_VARIANTS, M3_VARIANTS

    fig = plt.figure(figsize=(21, 13))
    grid = fig.add_gridspec(2, 4, height_ratios=(4, 1.65), hspace=.31, wspace=.29)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    names = ("M0a", "M0b", *variants)
    labels = [name.replace("_", " ") for name in names]
    offset = np.linspace(-.19, .19, 12)
    def points(ax, row, values, color="#256b91"):
        ax.scatter(values["participant_values"], row+offset, s=9, alpha=.4, color=color, linewidths=0)
        ax.scatter(values["mean"], row, marker="D", s=30, color=color, zorder=3)
    for index, name in enumerate(names):
        points(axes[0], index, rows["ordinary/"+name], "#333333" if index < 2 else "#256b91")
        if index >= 2:
            points(axes[1], index, rows["ordinary/"+name+"-M0b"])
            points(axes[2], index, rows["condition/"+name+"/control_minus_true"], "#9b5d2e")
            points(axes[3], index, rows["identity/"+name+"/control_minus_true"], "#774b8f")
    titles = ("Observed Task-B loss", "Model minus M0b", "Relabeled Task A minus true", "Donor Task A minus true")
    for index, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=11)
        ax.set_yticks(range(len(names)), labels if index == 0 else [""]*len(names), fontsize=8)
        ax.set_ylim(len(names)-.5, -.5)
        ax.set_xlabel("Observed beta MSE" if index == 0 else "Paired MSE difference")
        ax.grid(axis="x", alpha=.16)
        if index:
            ax.axvline(0, color="#777777", linewidth=.7)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
    ax = fig.add_subplot(grid[1, :])
    controls = [("matched_warp/"+name, name, "Matched warp: "+name.replace("_", " ")) for name in M2_VARIANTS]
    controls.append(("random_basis/"+M3_VARIANTS[0], M3_VARIANTS[0], "Random local basis: M3 reference"))
    for index, (stem, true_name, _) in enumerate(controls):
        points(ax, index, rows[stem+"/control_minus_true"], "#40785a")
        draw_means = [rows[stem+"/draw"+str(draw)]["mean"]-rows["ordinary/"+true_name]["mean"] for draw in range(8)]
        ax.scatter(draw_means, index+np.linspace(-.2, .2, 8), marker="x", s=19, color="#222222", alpha=.55)
    ax.set_yticks(range(5), [item[2] for item in controls], fontsize=9)
    ax.set_ylim(4.5, -.5)
    ax.axvline(0, color="#777777", linewidth=.7)
    ax.set_xlabel("Matched control minus true MSE; each of eight draw means also shown as ×")
    ax.set_title("Every fixed matched-random control", fontsize=11)
    ax.grid(axis="x", alpha=.16)
    fig.suptitle("EP15 · Prespecified out-of-fold Task-B-value prediction", fontsize=16, y=.985)
    fig.text(.5, .955, "12 development participants · 18 conditions · 188 availability-conditioned locations · all 16 model variants",
             ha="center", fontsize=10)
    fig.text(.5, .021, "Dots: participant values. Diamonds: equal-participant means. No confidence intervals. "
             "LOPO training overlaps; donor cycle is fixed. Original audit remains closed.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.245, right=.985, top=.92, bottom=.09)
    fig.savefig(output_dir / "observed_beta.png", dpi=170)
    fig.savefig(output_dir / "observed_beta.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-completed-summary", action="store_true")
    args = parser.parse_args(argv)
    require(args.release_completed_summary, "coordinator release flag is required before opening any result")
    runner._compute()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    require(runner.OUTPUTS in run_dir.parents and runner.OUTPUTS in output_dir.parents,
            "run and report paths must be within this episode's outputs")
    require(run_dir.is_dir() and output_dir.parent.is_dir() and not output_dir.exists(),
            "existing run and fresh report directory with existing parent required")
    path, operation = completed_payload(run_dir)
    payload = read_json(path, 32*1024**2)
    rows, variants, participants = validate_summary(payload)
    from development_lopo import M1_VARIANTS, M2_VARIANTS, M3_VARIANTS, M4_VARIANTS
    references = (M1_VARIANTS[0], M2_VARIANTS[0], M3_VARIANTS[0], M4_VARIANTS[0])
    line1 = ("Prespecified out-of-fold Task-B-value evaluation completed for all 12 development participants, "
             "18 conditions, 16 variants and fixed controls on 188 availability-conditioned locations.")
    line2 = ("Mean observed beta MSE: M0a="+format(rows["ordinary/M0a"]["mean"], ".6g")
             +", M0b="+format(rows["ordinary/M0b"]["mean"], ".6g")
             +"; fixed M1/M2/M3/M4 reference differences versus M0b (same order): "
             +", ".join(format(rows["ordinary/"+name+"-M0b"]["mean"], ".6g") for name in references)+".")
    line3 = ("Participant points describe variability; overlapping training and the fixed donor cycle preclude naive CIs. "
             "Target Task-B availability informed support; the original audit stays closed under procedural same-account separation, not technical enforcement; no winner or reliable-signal claim.")
    # Resolve the required figure dependency before creating a fresh output.
    import matplotlib
    # Create no report directory until access, completeness and data checks pass.
    output_dir.mkdir(mode=0o700)
    figure(rows, variants, output_dir)
    report = dict(analysis_scope=SCOPE, route_id=ROUTE, input_summary=str(path),
        source_job_id=operation["job_id"], coordinator_release_flag=True,
        complete_panel=PANEL, participant_ids=participants, model_order=variants,
        fixed_reference_order=references, interpretation=LIMITS,
        confidence_intervals_provided=False, original_audit_completed=False,
        numerical_methods=payload["numerical_methods"], summaries=rows,
        three_line_summary=[line1, line2, line3])
    with (output_dir / "observed_beta_report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    (output_dir / "3lines.md").write_text(line1+"\n"+line2+"\n"+line3+"\n")
    print(json.dumps({"status": "completed_released_descriptive_report", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
