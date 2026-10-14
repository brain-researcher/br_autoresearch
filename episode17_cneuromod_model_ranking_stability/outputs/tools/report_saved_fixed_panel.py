"""Adjudicate only the coordinator-released, completed EP17 panel001 summary.

No fits, responses, images, calibration arrays or audit inputs are opened.
Existing simultaneous intervals and margins are retained, with no new inference.
Run only in the coordinator's bounded Slurm allocation.
"""
import csv
import json
import math
import os
from pathlib import Path
import socket


def main():
    assert os.environ.get("SLURM_JOB_ID") and "-ln" not in socket.gethostname()
    root = Path(__file__).resolve().parents[2]
    source = root / "outputs/cache/private/development_endpoints/panel001/endpoint_panel.json"
    with source.open("rb") as stream:
        payload = stream.read(1048577)
    assert len(payload) <= 1048576
    panel = json.loads(payload)
    assert panel["complete_endpoint_panel"] and not panel["audit_access"]
    assert panel["phase"] == "development"
    assert panel["scope"] == dict(images=2423, concepts=480, concept_blocks=24,
        participants=4, ROIs=2, beta_stages=3, fixed_pairs=4, promotion_pairs=3, edges_per_pair=5)
    assert panel["bootstrap"]["draws"] == 10000 and panel["bootstrap"]["seed"] == 1729
    assert panel["bootstrap"]["order_statistic_one_based"] == 9501
    assert panel["frozen_block_order"] == [f"block{i:02d}" for i in range(1, 25)]
    expected = ["B_Aall_raw", "C_Aall_raw", "D_Aall_raw", "D_AN_raw", "D_RN_raw",
                "D_VNC_raw", "D_VNC_normalized"]
    edges = {"B_to_C": expected[0:2], "C_to_D": expected[1:3],
        "A_all_to_A_N": expected[2:4], "A_N_to_R_N": expected[3:5],
        "raw_to_ceiling_normalized": expected[5:7]}
    assert panel["endpoint_family"]["primary_endpoint_ids"] == expected
    assert panel["endpoint_family"]["h_component_ids"] == []
    assert set(panel["pairs"]) == {"P1", "P2", "P3", "F1"}
    rows, dispositions = [], {}
    for pair in ("P1", "P2", "P3", "F1"):
        value = panel["pairs"][pair]
        assert value["participant_ids"] == ["sub-01", "sub-02", "sub-03", "sub-06"]
        assert value["promotion_eligible"] == (pair != "F1")
        assert set(value["edges"]) == set(edges)
        for name, bounds in value["interval_families"].items():
            ids = expected if name == "F_raw_normalized" else expected[:6]
            assert name in {"F_raw", "F_raw_normalized"} and bounds["component_ids"] == ids
            margins = [.01] * 6 + ([.05] if name == "F_raw_normalized" else [])
            assert bounds["margins"] == margins
            for i, (estimate, lower, upper, se) in enumerate(zip(bounds["estimate"],
                    bounds["lower"], bounds["upper"], bounds["standard_error"])):
                assert all(math.isfinite(x) for x in (estimate, lower, upper, se))
                assert lower <= estimate <= upper and se > 0
                assert math.isclose(estimate, sum(row[i] for row in value["participant_effects"]) / 4,
                                    rel_tol=1e-10, abs_tol=1e-12)
        full = value["interval_families"]["F_raw_normalized"]
        equivalent = all(lo >= -margin and hi <= margin for lo, hi, margin in
                         zip(full["lower"], full["upper"], full["margins"]))
        if full["relation_decision"] == "practical_equivalence":
            assert equivalent
        for edge, (before, after) in edges.items():
            cell = value["edges"][edge]
            family = "F_raw_normalized" if edge == "raw_to_ceiling_normalized" else "F_raw"
            assert cell["before"] == before and cell["after"] == after and cell["interval_family"] == family
            bounds = value["interval_families"][family]
            a, b = (bounds["component_ids"].index(k) for k in (before, after))
            reversal = ((bounds["lower"][a] > bounds["margins"][a] and
                         bounds["upper"][b] < -bounds["margins"][b]) or
                        (bounds["upper"][a] < -bounds["margins"][a] and
                         bounds["lower"][b] > bounds["margins"][b]))
            assert reversal == cell["opposite_margin_crossing"]
            rows.append(dict(pair=pair, model_A=value["model_order"][0], model_B=value["model_order"][1],
                edge=edge, validity="valid_primary_comparison", before=before, after=after,
                before_effect=bounds["estimate"][a], before_lower=bounds["lower"][a],
                before_upper=bounds["upper"][a], after_effect=bounds["estimate"][b],
                after_lower=bounds["lower"][b], after_upper=bounds["upper"][b],
                reversal=reversal, forecast_eligible=value["forecasts"][edge]["eligible"],
                forecast_disposition=value["forecasts"][edge]["eligibility_reason"]))
        dispositions[pair] = full["relation_decision"]
    assert len(rows) == 20
    out = root / "outputs/runs/fixed_panel_adjudication/closeout001"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "all_twenty_comparisons.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(source=str(source), valid_primary_comparisons=20, fixed_panel_complete=True,
        original_adaptive_search_complete=False, pair_dispositions=dispositions,
        qualifying_reversals=sum(row["reversal"] for row in rows), audit_access=False,
        candidate_selected=False, no_refit=True, new_statistical_inference=False,
        job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
        limits=panel["exposure_limits"], separation=panel["separation"],
        interval_scope=panel["bootstrap"]["family_scope"])
    with (out / "adjudication.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    lines = ["# EP17 complete fixed-panel development result", "",
        "All20 registered primary comparisons are valid and complete under the October13 fixed-panel amendment. "
        "The original adaptive search is not claimed complete. All three trained-model pairs have simultaneous "
        "bounds within the registered practical-equivalence regions across the seven eligible endpoints. "
        "No registered edge shows a qualifying opposite-margin reversal. No primary candidate is selected and the audit stays closed.", "",
        "| Pair (A minus B) | Complete-family classification | Reliable-support raw R² difference | Simultaneous bounds |",
        "| --- | --- | ---: | --- |"]
    for pair in ("P1", "P2", "P3", "F1"):
        value = panel["pairs"][pair]
        b = value["interval_families"]["F_raw_normalized"]
        lines.append(f"| {pair}: {' − '.join(value['model_order'])} | {dispositions[pair]} | "
                     f"{b['estimate'][4]:.6f} | [{b['lower'][4]:.6f}, {b['upper'][4]:.6f}] |")
    lines += ["", "The table is one navigational endpoint; all20 comparisons and every endpoint are retained in "
        "`all_twenty_comparisons.csv` and the source summary. Raw-R² equivalence margin is0.01; "
        "ceiling-normalized margin is0.05. Small signed differences can be distinguishable from zero while remaining practically equivalent.", "",
        "F1 is the trained-versus-random falsifier and cannot be promoted. It is underidentified against the practical margins; "
        "this neither establishes equivalence nor proves no training benefit. Beta-edge primary comparisons are valid, "
        "but H-dependent explanatory forecasts remain inapplicable under the frozen calibration rule. "
        "The size-control edge has no adopted explanatory forecast. Correlation/CKA secondary analyses were not computed.", "",
        "Evidence is limited to four fixed participants, anatomical bilateral fusiform,480 development concepts, "
        "2423 images and24 concept blocks. The eight within-pair10000-draw families are not an experiment-wide "
        "multiplicity guarantee. Shared model pairs are not independent replications. Training-image exposure remains unknown; "
        "recipe differences cannot identify causal training effects. Separation is procedural under one account, not technically enforced. "
        "No claim of new-participant generalization or held-out audit confirmation follows.", "",
        "Reuse fit002 and panel001; this closeout performs no refit, resampling or audit access."]
    (out / "development_result.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 2, figsize=(11, 10), gridspec_kw={"width_ratios": [3, 2]})
    labels = ["B / anatomical", "C / anatomical", "D / anatomical", "D / matched count",
              "D / reliable support", "D / ceiling support"]
    for row, pair in enumerate(("P1", "P2", "P3", "F1")):
        value = panel["pairs"][pair]
        b = value["interval_families"]["F_raw_normalized"]
        for col, indices in enumerate((range(6), [6])):
            ax = axes[row, col]
            margin = .01 if col == 0 else .05
            ax.axvspan(-margin, margin, color="#e4edf4")
            ax.axvline(0, color="#666666", linewidth=.7)
            for j, i in enumerate(indices):
                ax.plot([b["lower"][i], b["upper"][i]], [j, j], color="#176585", linewidth=2)
                ax.plot(b["estimate"][i], j, "o", color="#176585", markersize=4)
            ax.set_yticks(list(range(len(indices))), labels if col == 0 else ["Normalized"])
            ax.invert_yaxis()
            ax.set_xlim((-.012, .043) if col == 0 else (-.06, .115))
            ax.set_title((pair + ": " + " − ".join(value["model_order"])) if col == 0 else dispositions[pair], fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 3:
                ax.set_xlabel("A − B raw R²" if col == 0 else "A − B ceiling-normalized R²")
    figure.suptitle("EP17: small trained-model differences on development concepts", fontsize=14)
    figure.text(.5, .008, "Existing within-pair simultaneous bounds; shaded practical-equivalence margins. "
                "Four fixed participants; audit unopened.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .03, 1, .96))
    destination = root / "outputs/figures/fixed_panel_development"
    destination.mkdir(parents=True, exist_ok=False)
    figure.savefig(destination / "complete_pair_intervals.png", dpi=160)
    figure.savefig(destination / "complete_pair_intervals.pdf")
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
