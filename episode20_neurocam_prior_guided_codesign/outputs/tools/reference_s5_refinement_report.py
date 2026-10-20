"""Freeze all 69 selected S5 records before the active distant prediction.

Historical records retain their original manifest and paths. No distant values
are used in calibration; an empty compatible union never opens distant data.
Low-level helpers retain historical 63/66-record inspection, but active CLI
execution requires both calibration-only adaptive centering rounds.
Execute only in the coordinator's Slurm allocation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reference_s5_family import CONTACT_NAMES, STRUCTURES
from reference_s5_operator import PROBE_SITE, nearest_summary
from reference_s5_prediction import load_json, saved_summary, write_json
from reference_s5_refinement import validate_manifest as validate_refinement_manifest
from reference_s5_report import spatial_curves, validate_frozen as validate_initial_freeze


INITIAL, INTERIOR, CENTER, R2_CENTER = "initial001", "interior001", "center001", "center002"
ROUND_ORDER = (INITIAL, INTERIOR, CENTER, R2_CENTER)
STATUS = "complete_history_nearest_only_frozen_before_distant_evaluation"
IDENTITY = ("round_id", "structure", "point", "record_dir")


def compatible(nearest, criteria):
    source, gate = nearest["nearest_source"], nearest["nearest_gate"]
    return (source["status"] == gate["status"] == "finite"
            and np.isfinite([source["db"], gate["db"]]).all()
            and abs(source["db"] - criteria["source_target_db"]) <= criteria["absolute_error_max_db"]
            and abs(gate["db"] - criteria["gate_target_db"]) <= criteria["absolute_error_max_db"]
            and source["db"] > gate["db"])


def validate_frozen(frozen):
    """Validate the complete round/structure/point union before record access."""
    if (frozen.get("status") != STATUS
            or set(frozen["manifests"]) not in
            ({INITIAL, INTERIOR}, {INITIAL, INTERIOR, CENTER}, set(ROUND_ORDER))):
        raise ValueError("a complete initial/interior, optionally sequentially centered nearest-only freeze is required")
    manifests = frozen["manifests"]
    validate_refinement_manifest(manifests[INTERIOR])
    if CENTER in manifests:
        from reference_s5_centering import validate_manifest as validate_centering_manifest
        validate_centering_manifest(manifests[CENTER])
    if R2_CENTER in manifests:
        from reference_s5_r2_centering import validate_manifest as validate_r2_centering_manifest
        validate_r2_centering_manifest(manifests[R2_CENTER])
    old_rows = [{key: value for key, value in row.items() if key != "round_id"}
                for row in frozen["calibration"] if row["round_id"] == INITIAL]
    old_accepted = [{key: value for key, value in row.items() if key != "round_id"}
                    for row in frozen["compatible"] if row["round_id"] == INITIAL]
    validate_initial_freeze(dict(status="fit_frozen_before_distant_evaluation",
        manifest=manifests[INITIAL], calibration=old_rows, compatible=old_accepted))
    if any(manifest["calibration"] != manifests[INITIAL]["calibration"]
           for manifest in manifests.values()):
        raise ValueError("refinement changes the initial nearest-only tolerances")
    expected = {(round_id, structure, point) for round_id, manifest in manifests.items()
                for structure in STRUCTURES for point in range(len(manifest["points"]))}
    round_counts = {INITIAL: 36, INTERIOR: 27,
                    **({CENTER: 3} if CENTER in manifests else {}),
                    **({R2_CENTER: 3} if R2_CENTER in manifests else {})}
    total = sum(round_counts.values())
    if (len(expected) != total
            or frozen.get("required_grid_records", total) != total
            or frozen.get("round_counts", round_counts) != round_counts):
        raise ValueError("exactly 36 initial, 27 interior and 3 records per included centering round are required")
    seen, accepted, paths = set(), [], set()
    for row in frozen["calibration"]:
        key = (row["round_id"], row["structure"], row["point"])
        if key not in expected or key in seen or row["record_dir"] in paths:
            raise ValueError("repeated, missing or changed round/structure/point binding")
        seen.add(key)
        paths.add(row["record_dir"])
        round_id, structure, point = key
        manifest = manifests[round_id]
        if (row["parameters"] != manifest["points"][point]
                or row["named_templates"] != [f"{name}/{structure}" for name in CONTACT_NAMES]):
            raise ValueError("a row changes its parameters or contact alternatives")
        binding = json.loads(row["binding"])
        if (binding["manifest"] != manifest or binding["structure"] != structure
                or binding["parameters"] != row["parameters"]):
            raise ValueError("a row differs from its original record manifest binding")
        is_compatible = bool(compatible(row["nearest"], manifest["calibration"]))
        if row["compatible"] is not is_compatible:
            raise ValueError("compatibility differs from the unchanged nearest-only rule")
        if is_compatible:
            accepted.append({key: row[key] for key in IDENTITY})
    if seen != expected or len(frozen["calibration"]) != total:
        raise ValueError(f"all {total} original and refinement records must be frozen")
    identities = lambda rows: sorted(tuple(row[key] for key in IDENTITY) for row in rows)
    if identities(accepted) != identities(frozen["compatible"]):
        raise ValueError("compatible union must contain every accepted historical and new point")
    return manifests


def freeze_history(initial_freeze, manifest, records_root, *, centering_manifest=None,
                   centering_records_root=None, r2_centering_manifest=None,
                   r2_centering_records_root=None, record_loader=saved_summary):
    """Read only completed records, preserving every historical binding."""
    if (centering_manifest is None) != (centering_records_root is None):
        raise ValueError("centering manifest and records root must be supplied together")
    if (r2_centering_manifest is None) != (r2_centering_records_root is None):
        raise ValueError("R2-centering manifest and records root must be supplied together")
    if r2_centering_manifest is not None and centering_manifest is None:
        raise ValueError("center002 requires the retained center001 manifest and records")
    old_manifest = validate_initial_freeze(initial_freeze)
    validate_refinement_manifest(manifest)
    manifests = {INITIAL: old_manifest, INTERIOR: manifest}
    rounds = [(INTERIOR, manifest, records_root)]
    if centering_manifest is not None:
        from reference_s5_centering import validate_manifest as validate_centering_manifest
        validate_centering_manifest(centering_manifest)
        manifests[CENTER] = centering_manifest
        rounds.append((CENTER, centering_manifest, centering_records_root))
    if r2_centering_manifest is not None:
        from reference_s5_r2_centering import validate_manifest as validate_r2_centering_manifest
        validate_r2_centering_manifest(r2_centering_manifest)
        manifests[R2_CENTER] = r2_centering_manifest
        rounds.append((R2_CENTER, r2_centering_manifest, r2_centering_records_root))
    rows = []
    for old in initial_freeze["calibration"]:
        summary, clips, binding = record_loader(old["record_dir"], old_manifest,
                                                old["structure"], old["point"])
        if (binding != old["binding"] or nearest_summary(summary) != old["nearest"]
                or clips != old["raw_clipped_count"]):
            raise ValueError("an original record differs from its retained nearest-only freeze")
        rows.append(dict(old, round_id=INITIAL))
    for round_id, selected, root in rounds:
        for structure in STRUCTURES:
            for point, parameters in enumerate(selected["points"]):
                directory = Path(root) / structure / f"point_{point:02d}"
                summary, clips, binding = record_loader(directory, selected, structure, point)
                nearest = nearest_summary(summary)
                rows.append(dict(round_id=round_id, structure=structure, point=point,
                    record_dir=str(directory.resolve()), parameters=parameters, binding=binding,
                    nearest=nearest, compatible=bool(compatible(nearest, selected["calibration"])),
                    raw_clipped_count=clips,
                    named_templates=[f"{name}/{structure}" for name in CONTACT_NAMES]))
    round_counts = {round_id: len(selected["points"]) * len(STRUCTURES)
                    for round_id, selected in manifests.items()}
    selection = "nearest_only_after_initial36_no_compatible_points; distant_unopened"
    if CENTER in manifests:
        selection += ("; center001 adaptively chosen by nearest-gate interpolation of released R1 interior "
                      "points07/08 at their fixed source load; the same point is run for R1/R2/R3")
    if R2_CENTER in manifests:
        selection += ("; center002 is one final common triplet adaptively selected from completed R2 "
                      "nearest-only calibration, not an unlimited search; all three structures retained")
    selection += f"; all{sum(round_counts.values())} records required before distant access"
    frozen = dict(status=STATUS, manifests=manifests,
        calibration=rows, compatible=[{key: row[key] for key in IDENTITY}
                                     for row in rows if row["compatible"]],
        required_grid_records=sum(round_counts.values()), round_counts=round_counts,
        uncertainty="numerically_explored_parameter_envelope_not_device_CI",
        refinement_selection=selection)
    validate_frozen(frozen)
    return frozen


def distant_interpretation(entries):
    """Describe the existing one-sided decisions without selecting survivors."""
    if not entries:
        raise ValueError("distant interpretation requires the nonempty frozen compatible set")
    values = [row["far_maximum"] for row in entries]
    finite = [value for value in values if value["status"] == "finite"]
    zeros = sum(value["status"] == "negative_infinity_zero_numerator" for value in values)
    undefined = len(values) - len(finite) - zeros
    passes = sum(row["distant_check_passes"] for row in entries)
    zero = dict(db=None, status="negative_infinity_zero_numerator")
    lower = zero if zeros else min(finite, key=lambda value: value["db"]) if finite else None
    upper = max(finite, key=lambda value: value["db"]) if finite else zero if zeros else None
    def label(value):
        if value is None:
            return "undefined"
        return "−∞ (zero fitted amplitude)" if value["db"] is None else f"{value['db']:.3f}"
    interval = dict(minimum=lower, maximum=upper, finite_points=len(finite), complete=not undefined,
                    zero_amplitude_points=zeros, undefined_points=undefined,
                    interpretation="explored extrema, not a confidence interval or continuous-domain bound")
    text = f"[{label(lower)}, {label(upper)}] dB"
    if undefined:
        text = f"unavailable ({undefined} undefined distant maxima)"
        category = "undefined_distant_values"
        meaning = "Undefined distant maxima prevent a complete interpretation of the explored set."
    elif passes == len(entries):
        category = "all_compatible_points_pass_ceiling"
        meaning = ("The explored set is consistent with this one-sided ceiling; this does not establish "
                   "agreement with −45 dB or identify a mechanism.")
    elif passes:
        category = "nearest_calibration_leaves_distant_check_unresolved"
        meaning = "Nearest calibration does not determine the distant pass/fail result within this explored set."
    else:
        category = "all_compatible_points_fail_ceiling"
        meaning = ("The joint assumed operator/circuit and explored compatible points fail this distant check; "
                   "no single assumption is isolated.")
    return interval, text, dict(category=category, meaning=meaning)


def build_result(frozen, *, paper_far_db=None, check_max_db=None, record_loader=saved_summary):
    manifests = validate_frozen(frozen)
    round_counts = {round_id: len(manifest["points"]) * len(STRUCTURES)
                    for round_id, manifest in manifests.items()}
    total = sum(round_counts.values())
    per_structure = total // len(STRUCTURES)
    scope = "36 initial plus 27 nearest-only interior"
    if CENTER in manifests:
        scope += " plus 3 center001 records adaptively chosen from R1 nearest-only interpolation"
    if R2_CENTER in manifests:
        scope += " plus 3 center002 records from one final R2-guided nearest-only centering triplet"
    counts = {structure: sum(row["compatible"] for row in frozen["calibration"]
                              if row["structure"] == structure) for structure in STRUCTURES}
    result = dict(manifests=manifests, calibration=frozen["calibration"], entries=[],
        grid_records=total, round_counts=round_counts,
        grid_records_by_structure={structure: per_structure for structure in STRUCTURES},
        compatible_by_structure=counts,
        uncertainty="numerically_explored_compatible_parameter_envelope_not_device_CI",
        data_separation="same_account_procedural; previously known public summaries; not blind validation",
        contact_law_interpretation="E_area and E_sublinear give identical dry-air predictions; they are not independent evidence and this fixture cannot distinguish them",
        refinement_selection=frozen["refinement_selection"],
        numerical_floor="extreme negative dB can represent floating-point residues of near-zero fitted amplitudes; no physical attenuation precision is claimed",
        not_claimed=["complete_continuous_domain_coverage", "physical_model_rejection",
                     "independent_device_confidence_interval", "accepted_full_reference", "candidate_R2_ranking"])
    if not frozen["compatible"]:
        result.update(status=f"no_compatible_point_in_complete{total}_point_exploration",
            distant_evidence_accessed=False, interpretation=[
                f"Conditional dry-air S5 calibration retains all {total} full 16-second records: {scope}; E_area/E_sublinear give identical predictions, not independent evidence.",
                "No explored point meets both nearest-anchor ±3 dB tolerances and the source-above-gate rule; the distant check remains unopened.",
                "This finite-grid outcome does not reject the continuous parameter domain or physical hardware; extreme negative dB residues are not precise physical attenuation measurements."])
        return result, None
    if paper_far_db != -45. or check_max_db != -42.:
        raise ValueError("unchanged assigned distant paper summary and ceiling are required")
    lookup = {(row["round_id"], row["structure"], row["point"]): row for row in frozen["calibration"]}
    amplitudes_all = []
    for candidate in frozen["compatible"]:
        key = (candidate["round_id"], candidate["structure"], candidate["point"])
        row = lookup[key]
        summary, clips, binding = record_loader(row["record_dir"], manifests[row["round_id"]],
                                                row["structure"], row["point"])
        if (binding != row["binding"] or nearest_summary(summary) != row["nearest"]
                or clips != row["raw_clipped_count"]):
            raise ValueError("prediction record differs from its frozen calibration")
        amplitudes = summary.amplitudes(np.arange(4096))
        curves = spatial_curves(amplitudes)
        far = curves["far_maximum"]
        passes = (far["status"] == "negative_infinity_zero_numerator"
                  or (far["status"] == "finite" and far["db"] <= check_max_db))
        result["entries"].append(dict(**candidate, **curves,
            named_templates=row["named_templates"], raw_clipped_count=clips,
            distant_check_passes=passes))
        amplitudes_all.append(amplitudes)
    n_pass = sum(row["distant_check_passes"] for row in result["entries"])
    far_range, range_text, meaning = distant_interpretation(result["entries"])
    count_text = ", ".join(f"{name}: {counts[name]}/{per_structure}" for name in STRUCTURES)
    result.update(status="conditional_distant_prediction", distant_evidence_accessed=True,
        distant_maximum_range=far_range, distant_maximum_range_text=range_text,
        distant_check_interpretation=meaning,
        paper_far_db=paper_far_db, check_max_db=check_max_db, interpretation=[
            f"Conditional dry-air S5 retains {scope}, all full 16-second records; all compatible points were frozen ({count_text}); E_area/E_sublinear give identical predictions, not independent evidence.",
            f"Distant maximum range {range_text}; {n_pass}/{len(result['entries'])} compatible points satisfy the −42 dB ceiling over every pad at Manhattan distance ≥2; the held-aside public summary is approximately −45 dB.",
            meaning["meaning"] + " This finite conditional range is not a device confidence interval, continuous-domain exclusion, hardware validation or full-reference acceptance; known public summaries and same-account separation preclude blind-validation claims."])
    return result, np.stack(amplitudes_all)


def require_active_centering(frozen):
    """Admit the active 69-record route; 63/66-record helpers remain historical."""
    manifests = validate_frozen(frozen)
    if R2_CENTER not in manifests:
        raise ValueError("active conditional reporting requires all 69 records including center001 and center002")
    return frozen


def _calibration_panels(axes, result):
    colors = dict(R1="tab:blue", R2="tab:orange", R3="tab:green")
    criteria = result["manifests"][INITIAL]["calibration"]
    rounds = [round_id for round_id in ROUND_ORDER if round_id in result["manifests"]]
    labels = {INITIAL: "Initial", INTERIOR: "interior", CENTER: "center", R2_CENTER: "R2 center"}
    offset, ticks, separators, sections = 0, [], [], []
    for round_id in rounds:
        count = len(result["manifests"][round_id]["points"])
        if offset:
            separators.append(offset - .5)
        ticks.extend([offset, offset + (count - 1) // 2, offset + count - 1])
        sections.append(f"{labels[round_id]} point {offset}" if count == 1 else
                        f"{labels[round_id]} points {offset}–{offset + count - 1}")
        offset += count
    for axis, direction in zip(axes, ("source", "gate")):
        nonfinite = 0
        for index, structure in enumerate(STRUCTURES):
            rows = sorted((row for row in result["calibration"] if row["structure"] == structure),
                          key=lambda row: (rounds.index(row["round_id"]), row["point"]))
            for position, row in enumerate(rows):
                value = row["nearest"][f"nearest_{direction}"]
                x = position + (index - 1) * .2
                if value["status"] != "finite":
                    nonfinite += 1
                    axis.scatter([x], [-49], color=colors[structure], marker="x", s=24)
                    axis.annotate(value["status"], (x, -49), xytext=(0, 5),
                                  textcoords="offset points", rotation=90, fontsize=5)
                    continue
                residual = value["db"] - criteria[f"{direction}_target_db"]
                outside = residual < -50 or residual > 50
                y = (-49. if residual < -50 else 49.) if outside else residual
                marker = "v" if residual < -50 else "^" if residual > 50 else "o"
                axis.scatter([x], [y], color=colors[structure], marker=marker, s=22,
                    label=f"{structure} (both E contacts)" if position == 0 else None)
                if outside:
                    axis.annotate(f"{residual:.0f}", (x, y), xytext=(0, 5 if y < 0 else -9),
                                  textcoords="offset points", rotation=90, ha="center", fontsize=6)
        tolerance = criteria["absolute_error_max_db"]
        axis.axhspan(-tolerance, tolerance, color="grey", alpha=.18)
        axis.axhline(0., color="black", lw=.7)
        for separator in separators:
            axis.axvline(separator, color="grey", ls="--", lw=.8)
        axis.set(xlabel=" | ".join(sections), ylim=(-52, 52),
                 ylabel="Prediction − nearest anchor (dB)",
                 title=f"Nearest {direction}: all {result['grid_records']} records; ±50 dB zoom",
                 xticks=sorted(set(ticks)))
        axis.text(.01, .98, "Triangles: outside range, labelled actual residual\nExtreme negative residues do not establish physical attenuation precision",
                  transform=axis.transAxes, va="top", fontsize=6)
        if nonfinite:
            axis.text(.01, .04, f"{nonfinite} nonfinite ratios marked ×; explicit status in JSON",
                      transform=axis.transAxes, fontsize=6)
    axes[0].legend(fontsize=6, loc="center left")


def write_report(output_dir, result, amplitudes):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("report output directory must be fresh and empty")
    lines = result["interpretation"]
    if len(lines) != 3 or any(not line or "\n" in line for line in lines):
        raise ValueError("exactly three nonempty summary lines are required")
    write_json(out / "prediction.json", result)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if amplitudes is not None:
        rr, cc = np.indices((64, 64))
        np.savez(out / "per_pad_amplitudes.npz", amplitudes_volts=amplitudes,
            rows=rr.ravel(), columns=cc.ravel(),
            manhattan_distance=(abs(rr - PROBE_SITE[0]) + abs(cc - PROBE_SITE[1])).ravel(),
            round_ids=np.asarray([row["round_id"] for row in result["entries"]]),
            structures=np.asarray([row["structure"] for row in result["entries"]]),
            points=np.asarray([row["point"] for row in result["entries"]]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not result["entries"]:
        figure, calibration_axes = plt.subplots(1, 2, figsize=(13, 5))
        _calibration_panels(calibration_axes, result)
        figure.suptitle(f"No compatible point in all {result['grid_records']} explored records; distant evidence remains unopened")
    else:
        figure = plt.figure(figsize=(15, 9))
        grid = figure.add_gridspec(2, 6)
        _calibration_panels([figure.add_subplot(grid[0, :3]), figure.add_subplot(grid[0, 3:])], result)
        colors = dict(R1="tab:blue", R2="tab:orange", R3="tab:green")
        for index, (field, title) in enumerate((("all_pad_manhattan_curve", "All-pad shell maxima"),
                ("source_curve", "Source direction"), ("gate_curve", "Gate direction"))):
            axis = figure.add_subplot(grid[1, 2 * index:2 * index + 2])
            seen = set()
            for row in result["entries"]:
                key = row["round_id"], row["structure"]
                values = [value["db"] if value["status"] == "finite" else np.nan for value in row[field]]
                axis.plot(range(len(values)), values, color=colors[row["structure"]], alpha=.55,
                    ls={INITIAL: "-", INTERIOR: "--", CENTER: ":", R2_CENTER: "-."}[row["round_id"]],
                    label=" / ".join(key) if key not in seen else None)
                seen.add(key)
            axis.axhline(result["paper_far_db"], color="black", ls="--", label="Paper ≈−45 dB")
            axis.axhline(result["check_max_db"], color="grey", ls=":", label="Ceiling −42 dB")
            axis.set(xlabel="Pad separation", ylabel="Amplitude / driven output (dB)", title=title)
            if index == 0:
                axis.legend(fontsize=6)
        figure.suptitle("Conditional dry-air S5: all compatible explored points; no device confidence interval")
    caption = "; ".join(f"{key}: {result['round_counts'][key]}" for key in ROUND_ORDER
                        if key in result["round_counts"])
    if CENTER in result["manifests"]:
        caption += ". Calibration-only adaptive R1 centering"
    if R2_CENTER in result["manifests"]:
        caption += " and one final R2 centering triplet"
    caption += ".\nE_area/E_sublinear coincide in dry air; no independent contact-law evidence."
    if result["entries"]:
        caption += " Distant maxima: " + result["distant_maximum_range_text"] + "."
    caption += "\nFinite conditional exploration; no global exclusion, device confidence interval or hardware validation."
    figure.text(.5, .015, caption, ha="center", fontsize=7, wrap=True)
    figure.tight_layout(rect=(0., .06, 1., 1.))
    figure.savefig(out / "conditional_s5_refinement_prediction.png", dpi=180)
    figure.savefig(out / "conditional_s5_refinement_prediction.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--initial-freeze", required=True)
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--records-root", required=True)
    freeze.add_argument("--centering-manifest", required=True)
    freeze.add_argument("--centering-records-root", required=True)
    freeze.add_argument("--r2-centering-manifest", required=True)
    freeze.add_argument("--r2-centering-records-root", required=True)
    freeze.add_argument("--output", required=True)
    report = commands.add_parser("report")
    report.add_argument("--freeze", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--paper-far-db", type=float)
    report.add_argument("--check-max-db", type=float)
    args = parser.parse_args()
    from fast_solver_continuous import compute_context
    compute_context()
    if args.command == "freeze":
        if Path(args.output).exists():
            raise ValueError("combined freeze output must be new")
        value = freeze_history(load_json(args.initial_freeze), load_json(args.manifest), args.records_root,
            centering_manifest=load_json(args.centering_manifest),
            centering_records_root=args.centering_records_root,
            r2_centering_manifest=load_json(args.r2_centering_manifest),
            r2_centering_records_root=args.r2_centering_records_root)
        write_json(args.output, require_active_centering(value))
    else:
        out = Path(args.output_dir)
        if out.exists() and any(out.iterdir()):
            raise ValueError("report output directory must be fresh and empty")
        frozen = require_active_centering(load_json(args.freeze))
        result, amplitudes = build_result(frozen,
            paper_far_db=args.paper_far_db, check_max_db=args.check_max_db)
        write_report(out, result, amplitudes)


if __name__ == "__main__":
    main()
