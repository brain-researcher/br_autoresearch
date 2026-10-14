"""Report the frozen conditional S5 grid; never use distant evidence to refit.

This separate entrypoint leaves the live calibration imports untouched. It
requires the complete 3 x 12 nearest-only freeze. An empty compatible sample
produces a calibration-only report without opening records or distant evidence.
Run on compute with the existing numpy/scipy/h5py/matplotlib module environment.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from reference_s5_family import CONTACT_NAMES, STRUCTURES
from reference_s5_operator import PROBE_SITE, nearest_summary, ratio_db
from reference_s5_prediction import load_json, validate_manifest, write_json


def _valid_ratio(value):
    status = value.get("status")
    if status == "finite":
        return isinstance(value.get("db"), (int, float)) and math.isfinite(value["db"])
    return status in ("negative_infinity_zero_numerator", "undefined_zero_denominator") and value.get("db") is None


def validate_frozen(frozen):
    """Reject partial/search-selected subsets before opening any record."""
    if frozen.get("status") != "fit_frozen_before_distant_evaluation":
        raise ValueError("a completed nearest-only parameter freeze is required")
    manifest = frozen["manifest"]
    validate_manifest(manifest)
    expected = {(structure, point) for structure in STRUCTURES for point in range(len(manifest["points"]))}
    rows, seen, accepted = frozen["calibration"], set(), []
    criteria = manifest["calibration"]
    for row in rows:
        key = (row["structure"], row["point"])
        if key not in expected or key in seen:
            raise ValueError("frozen calibration repeats or changes a prescribed grid point")
        seen.add(key)
        structure, point = key
        if (row["parameters"] != manifest["points"][point]
                or row["named_templates"] != [f"{name}/{structure}" for name in CONTACT_NAMES]):
            raise ValueError("frozen calibration changes parameters or contact alternatives")
        binding = json.loads(row["binding"])
        if (binding["manifest"] != manifest or binding["structure"] != structure
                or binding["parameters"] != row["parameters"]):
            raise ValueError("frozen calibration record differs from its selected experiment")
        source, gate = row["nearest"]["nearest_source"], row["nearest"]["nearest_gate"]
        if not _valid_ratio(source) or not _valid_ratio(gate):
            raise ValueError("invalid nearest-observation ratio")
        compatible = (source["status"] == gate["status"] == "finite"
            and abs(source["db"] - criteria["source_target_db"]) <= criteria["absolute_error_max_db"]
            and abs(gate["db"] - criteria["gate_target_db"]) <= criteria["absolute_error_max_db"]
            and source["db"] > gate["db"])
        if row["compatible"] is not compatible:
            raise ValueError("compatibility differs from the unchanged nearest-only rule")
        if compatible:
            accepted.append(dict(structure=structure, point=point, record_dir=row["record_dir"]))
    if seen != expected or len(rows) != len(expected):
        raise ValueError("all 36 prescribed structural/grid records must be frozen")
    by_identity = lambda row: (row["structure"], row["point"], row["record_dir"])
    if sorted(map(by_identity, accepted)) != sorted(map(by_identity, frozen["compatible"])):
        raise ValueError("reported compatible sample is not the complete nearest-only compatible set")
    return manifest


def spatial_curves(amplitudes):
    """Include off-axis pads in the exact spatial statistic used by the check."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.shape != (4096,) or not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0):
        raise ValueError("finite nonnegative amplitude required for every physical pad")
    rr, cc = np.indices((64, 64))
    pr, pc = PROBE_SITE
    distance = (abs(rr - pr) + abs(cc - pc)).ravel()
    denominator = float(amplitudes[pr * 64 + pc])
    maximum = lambda mask: ratio_db(float(np.max(amplitudes[mask])), denominator)
    return dict(
        all_pad_manhattan_curve=[maximum(distance == d) for d in range(int(distance.max()) + 1)],
        source_curve=[maximum(((cc == pc) & (abs(rr - pr) == d)).ravel()) for d in range(33)],
        gate_curve=[maximum(((rr == pr) & (abs(cc - pc) == d)).ravel()) for d in range(33)],
        far_maximum=maximum(distance >= 2))


def build_result(frozen, *, paper_far_db=None, check_max_db=None, record_loader=None):
    manifest = validate_frozen(frozen)
    counts = {structure: sum(row["compatible"] for row in frozen["calibration"]
                              if row["structure"] == structure) for structure in STRUCTURES}
    result = dict(manifest=manifest, calibration=frozen["calibration"],
        grid_records=36, grid_records_by_structure={name: 12 for name in STRUCTURES},
        compatible_by_structure=counts, entries=[],
        uncertainty="numerically_explored_compatible_parameter_envelope_not_device_CI",
        data_separation="same_account_procedural; public paper summaries previously inspected, not blind or independent validation",
        calibration_source="complete nearest-only freeze after all 36 full 16-second records",
        exact_zero_plot="nonfinite dB omitted from curves and retained explicitly in numerical tables",
        not_claimed=["complete_continuous_domain_coverage", "physical_model_rejection",
                     "independent_device_confidence_interval", "accepted_full_reference", "candidate_R2_ranking"])
    count_text = ", ".join(f"{structure}: {counts[structure]}/12" for structure in STRUCTURES)
    if not frozen["compatible"]:
        # Do not load any response record, calculate far amplitudes, or inspect
        # the paper/check arguments on this branch. The sample can miss fits.
        result.update(status="no_compatible_point_in_completed_sampled_grid", distant_evidence_accessed=False,
            interpretation=[
                "Conditional dry-air S5 calibration completed all 36 full 16-second records (12 points per R1/R2/R3); both E contact labels share each air prediction.",
                "No sampled point met both nearest-anchor ±3 dB tolerances and the source-above-gate rule; the distant check was not opened.",
                "This finite-grid outcome does not reject the continuous parameter domain or physical hardware; further nearest-only exploration remains possible."])
        return result, None
    if paper_far_db != -45. or check_max_db != -42.:
        raise ValueError("nonempty prediction requires the assigned paper summary and unchanged distant ceiling")
    if record_loader is None:
        from reference_s5_prediction import saved_summary
        record_loader = saved_summary
    amplitudes_all = []
    lookup = {(row["structure"], row["point"]): row for row in frozen["calibration"]}
    for candidate in frozen["compatible"]:
        row = lookup[(candidate["structure"], candidate["point"])]
        summary, clips, binding = record_loader(candidate["record_dir"], manifest,
                                                candidate["structure"], candidate["point"])
        if binding != row["binding"] or nearest_summary(summary) != row["nearest"]:
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
    n_pass = sum(entry["distant_check_passes"] for entry in result["entries"])
    result.update(status="conditional_distant_prediction", distant_evidence_accessed=True,
        paper_far_db=paper_far_db, check_max_db=check_max_db,
        interpretation=[
            f"Conditional dry-air S5 used all 36 full 16-second records and froze the nearest-only compatible sample ({count_text}); both E contact labels share each air prediction.",
            f"{n_pass}/{len(result['entries'])} compatible sampled points satisfy the −42 dB ceiling for the maximum over every pad at Manhattan distance ≥2; the held-aside paper summary is approximately −45 dB.",
            "These are explored conditional predictions, not device confidence intervals or full-reference acceptance; previously known public summaries and same-account procedural separation preclude blind-validation claims."])
    return result, np.stack(amplitudes_all)


def _calibration_panels(axes, result):
    colors = dict(R1="tab:blue", R2="tab:orange", R3="tab:green")
    criteria = result["manifest"]["calibration"]
    for axis, direction in zip(axes, ("source", "gate")):
        missing = 0
        for index, structure in enumerate(STRUCTURES):
            rows = sorted((row for row in result["calibration"] if row["structure"] == structure),
                          key=lambda row: row["point"])
            values = [row["nearest"][f"nearest_{direction}"] for row in rows]
            residual = [value["db"] - criteria[f"{direction}_target_db"]
                        if value["status"] == "finite" else np.nan for value in values]
            missing += sum(value["status"] != "finite" for value in values)
            axis.scatter(np.arange(12) + (index - 1) * .18, residual,
                         color=colors[structure], s=19, label=structure + " (both E contacts)")
        tolerance = criteria["absolute_error_max_db"]
        axis.axhspan(-tolerance, tolerance, color="grey", alpha=.15)
        axis.axhline(0., color="black", lw=.7)
        axis.set(xlabel="Prespecified parameter point", ylabel="Prediction − nearest anchor (dB)",
                 title=f"Nearest {direction}: all 36 calibration records", xticks=range(12))
        if missing:
            axis.text(.02, .97, f"{missing} nonfinite ratios omitted; see table", va="top",
                      transform=axis.transAxes, fontsize=8)
    axes[0].legend(fontsize=7)


def write_report(output_dir, result, amplitudes):
    """Write one figure plus the exact three-line summary and its numeric data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("report output directory must be fresh and empty")
    lines = result["interpretation"]
    if len(lines) != 3 or any(not line or "\n" in line for line in lines):
        raise ValueError("scientific summary must contain exactly three nonempty lines")
    # Save scientific numbers before rendering; a plotting failure needs no refit.
    write_json(out / "prediction.json", result)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if amplitudes is not None:
        rr, cc = np.indices((64, 64))
        np.savez(out / "per_pad_amplitudes.npz", amplitudes_volts=amplitudes,
                 rows=rr.ravel(), columns=cc.ravel(),
                 manhattan_distance=(abs(rr - PROBE_SITE[0]) + abs(cc - PROBE_SITE[1])).ravel(),
                 structures=np.asarray([entry["structure"] for entry in result["entries"]]),
                 points=np.asarray([entry["point"] for entry in result["entries"]]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not result["entries"]:
        figure, calibration_axes = plt.subplots(1, 2, figsize=(11, 4))
        _calibration_panels(calibration_axes, result)
        figure.suptitle("No compatible point in this 36-point sample; distant evidence remains unopened")
    else:
        figure = plt.figure(figsize=(14, 8))
        grid = figure.add_gridspec(2, 6)
        calibration_axes = [figure.add_subplot(grid[0, :3]), figure.add_subplot(grid[0, 3:])]
        _calibration_panels(calibration_axes, result)
        axes = [figure.add_subplot(grid[1, start:start + 2]) for start in (0, 2, 4)]
        colors = dict(R1="tab:blue", R2="tab:orange", R3="tab:green")
        fields = ("all_pad_manhattan_curve", "source_curve", "gate_curve")
        titles = ("All pads: maximum in each Manhattan shell", "Source direction", "Gate direction")
        for axis, field, title in zip(axes, fields, titles):
            seen = set()
            for entry in result["entries"]:
                values = [value["db"] if value["status"] == "finite" else np.nan for value in entry[field]]
                label = entry["structure"] + " (both E contacts)" if entry["structure"] not in seen else None
                seen.add(entry["structure"])
                axis.plot(range(len(values)), values, color=colors[entry["structure"]], alpha=.55, label=label)
            maximum_distance = 64 if field == "all_pad_manhattan_curve" else 32
            axis.hlines(result["paper_far_db"], 2, maximum_distance, color="black", linestyles="--",
                        label="Held-aside paper ≈−45 dB")
            axis.hlines(result["check_max_db"], 2, maximum_distance, color="grey", linestyles=":",
                        label="Prespecified distant ceiling")
            axis.set(xlabel="Pad separation", ylabel="Amplitude / driven output (dB)",
                     title=title, xlim=(0, maximum_distance))
        axes[0].legend(fontsize=7)
        figure.suptitle("Conditional dry-air S5: every compatible sampled prediction; no device confidence interval")
    figure.tight_layout()
    figure.savefig(out / "conditional_s5_prediction.png", dpi=180)
    figure.savefig(out / "conditional_s5_prediction.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-far-db", type=float)
    parser.add_argument("--check-max-db", type=float)
    args = parser.parse_args()
    from fast_solver_continuous import compute_context
    compute_context()
    result, amplitudes = build_result(load_json(args.freeze), paper_far_db=args.paper_far_db,
                                      check_max_db=args.check_max_db)
    write_report(args.output_dir, result, amplitudes)


if __name__ == "__main__":
    main()
