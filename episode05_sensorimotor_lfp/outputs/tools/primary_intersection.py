"""Fixed unit-guide intersection and eight ordinary count-matched HW supports."""
import numpy as np

from primary_capacity import score_bank
from primary_history_blend import validate_original_selections
from primary_matched_budget import fit_view
from primary_reference import ANIMALS, LAMBDAS, SESSIONS
from primary_temporal import aggregate_rows, directional_screens, select_paired_lambda
from qualify_primary_intersection import RECIPES, REFERENCE, REPLICATES, choose_pilot, guide_intersection, support_sets


def validate_description(row, records):
    session = row["session_id"]
    joined = guide_intersection(row["lfp_guide"], row["unit_guide_pairs"], row["original_electrode_ids"])
    if any(row[key] != value for key, value in joined.items()):
        raise ValueError("Exact guide-derived physical intersection required")
    ids = [r["trial_id"] for r in records]
    directions = [int(r["direction"]) for r in records]
    supports = support_sets(joined["original_electrode_ids"], joined["intersection_electrode_ids"], session)
    unique = {tuple(s["electrode_ids"]) for s in supports if not s["full_support"]}
    p = 4*joined["intersection_electrode_count"]
    if (session not in SESSIONS or row["animal"] != ANIMALS[session]
            or row["complete_trials"] != len(ids) or row["trial_ids"] != ids
            or row["trial_directions"] != directions
            or row["source_trial_indices_zero_based"] != [int(r["source_trial_index_zero_based"]) for r in records]
            or row["direction_counts"] != [directions.count(d) for d in range(8)]
            or any(directions.count(d) < 2 for d in range(8))
            or row["constant_guides_on_all_original_trials"] is not True
            or any(row[key] is not False for key in ("spike_values_read", "LFP_XY_values_read", "scores_computed"))
            or row["support_sets"] != supports or row["unique_new_fit_supports"] != len(unique)
            or row["workload_proxy"] != len(unique)*(105*len(ids)*p*p+8*p*p*p)):
        raise ValueError("Complete original identities, immutable queues and metadata-only workload required")
    return row


def validate_metadata(metadata, records):
    rows = metadata["sessions"]
    if (metadata["status"] != "complete_primary_intersection_metadata"
            or metadata["all_sixteen_qualified_before_values"] is not True
            or metadata["scientific_trial_count_increment"] != 0
            or any(metadata[key] is not False for key in ("spike_values_read", "LFP_XY_values_read", "scores_computed"))
            or len(rows) != 16 or [r["session_id"] for r in rows] != list(SESSIONS)
            or set(records) != set(SESSIONS)):
        raise ValueError("All sixteen qualified guide-only joins required before any neural values")
    descriptions = {row["session_id"]: validate_description(row, records[row["session_id"]]) for row in rows}
    if (metadata["pilot_session"] != choose_pilot(rows)
            or metadata["global_full_HW_duplicate"] != all(r["unique_new_fit_supports"] == 0 for r in rows)):
        raise ValueError("The frozen metadata-only pilot and full-support duplicate flags must replay")
    return descriptions


def fit_supports(x, y, labels, ids, description, check=lambda: None):
    if (np.asarray(ids).tolist() != description["trial_ids"]
            or np.asarray(labels).tolist() != description["trial_directions"]
            or x.shape != (len(y), 18, len(description["original_electrode_ids"]))):
        raise ValueError("Unchanged original trial/channel source arrays required")
    planned = support_sets(description["original_electrode_ids"], description["intersection_electrode_ids"],
                           description["session_id"])
    if description["support_sets"] != planned:
        raise ValueError("Exact fixed intersection and ordinary support sets required")
    fits, routes, seen = {}, [], {}
    for support in planned:
        check()
        signature = tuple(support["electrode_ids"])
        if support["full_support"]:
            key = "full_HW"
        elif signature in seen:
            key = seen[signature]
        else:
            key = f"fit{len(fits):03d}"
            # Only channels change. Preserve original y/labels and their layout.
            selected = x[:, :, support["electrode_positions"]]
            baseline, predictions, metadata = fit_view(selected, y, labels, check)
            fits[key] = {"baseline": baseline, "predictions": predictions, "metadata": metadata,
                         "electrode_ids": support["electrode_ids"]}
            seen[signature] = key
        routes.append(key)
    return fits, routes


def average_banks(banks):
    if len(banks) != 8:
        raise ValueError("All eight fixed count-matched support banks required")
    first = banks[0]
    for bank in banks:
        for key in ("recipe", "lambdas", "baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
            if bank[key] != first[key]:
                raise ValueError("Identical native target/B/denominator support required")
    rows = []
    for index, penalty in enumerate(LAMBDAS):
        correct = float(np.mean([bank["lambda_rows"][index]["candidate_SSE"] for bank in banks]))
        wrong = float(np.mean([bank["lambda_rows"][index]["wrong_pair_SSE"] for bank in banks]))
        rows.append({"lambda": penalty, "candidate_R2": 1-correct/first["denominator"],
            "candidate_SSE": correct, "wrong_pair_SSE": wrong,
            "delta_R2": (first["baseline_SSE"]-correct)/first["denominator"],
            "identity_increment": (wrong-correct)/first["denominator"]})
    return {**first, "lambda_rows": rows, "support_replicate_ids": list(REPLICATES),
            "support_aggregation": "mean correct/wrong SSE; never averaged predictions",
            "support_seed_rows": [{"replicate": r, "lambda_rows": bank["lambda_rows"]}
                                  for r, bank in zip(REPLICATES, banks)]}


def finalize_banks(y, labels, ids, description, fits, routes, reference):
    if (len(routes) != 9 or not np.array_equal(reference["actual_xy"], y)
            or not np.array_equal(reference["directions"], labels)
            or not np.array_equal(reference["trial_ids"], ids)
            or np.asarray(reference["candidate_xy_by_lambda"]).shape != (5, *y.shape)
            or not np.isfinite(reference["candidate_xy_by_lambda"]).all()):
        raise ValueError("Exact original-order full-HW targets/identities and five finite predictions required")
    # Recreate only storage layout when every support is reused; no model fit.
    baseline = next(iter(fits.values()))["baseline"] if fits else np.empty_like(y)
    if not fits:
        baseline[...] = reference["baseline_xy"]
    if not np.array_equal(reference["baseline_xy"], baseline):
        raise ValueError("Column restriction must preserve original behavioral baseline exactly")
    full = score_bank(y, labels, baseline, reference["candidate_xy_by_lambda"], REFERENCE)
    if not np.isfinite(full["denominator"]) or full["denominator"] <= 0:
        raise ValueError("Original positive pooled native target denominator required")
    if "score_bank" in reference:
        for key in ("lambdas", "lambda_rows", "baseline_SSE", "baseline_R2", "denominator", "direction_counts"):
            if full[key] != reference["score_bank"][key]:
                raise ValueError("Saved full-HW prediction scores did not replay exactly: "+key)
    views, payload = [], {}
    for support, key in zip(description["support_sets"], routes):
        if key == "full_HW":
            if not support["full_support"]:
                raise ValueError("Only full physical support can reuse original HW")
            prediction, metadata = reference["candidate_xy_by_lambda"], reference["fold_metadata"]
        else:
            fit = fits[key]
            if fit["electrode_ids"] != support["electrode_ids"] or not np.array_equal(fit["baseline"], baseline):
                raise ValueError("Shared support cache and unchanged B must replay exactly")
            prediction, metadata = fit["predictions"], fit["metadata"]
            payload[key+"_candidate_xy_by_lambda"] = prediction
        bank = score_bank(y, labels, baseline, prediction, support["recipe"])
        views.append({"support": support, "fit_key": key, "bank": bank,
            "fit_diagnostics": [{"held_direction": row["held_direction"],
                "fitting_bin_rows": row["fitting_bin_rows"],
                **{key: row["paired_weighted_covariance"][key] for key in
                   ("trace", "numerical_rank", "ridge_df_by_lambda")}} for row in metadata]})
    candidate = {**views[0]["bank"], "views": views[:1], "full_HW_bank": full}
    control = {**average_banks([view["bank"] for view in views[1:]]), "views": views[1:], "full_HW_bank": full}
    payload["baseline_xy"] = baseline
    return {RECIPES[0]: candidate, RECIPES[1]: control}, payload


def recipe_charges(sessions):
    full_i = all(sessions[s][RECIPES[0]]["views"][0]["support"]["full_support"] for s in SESSIONS)
    full_r = all(v["support"]["full_support"] for s in SESSIONS for v in sessions[s][RECIPES[1]]["views"])
    same = all(v["support"]["electrode_ids"] == sessions[s][RECIPES[0]]["views"][0]["support"]["electrode_ids"]
               for s in SESSIONS for v in sessions[s][RECIPES[1]]["views"])
    return ({RECIPES[0]: int(not full_i), RECIPES[1]: int(not (full_r or same))},
            {RECIPES[0]: "full_HW" if full_i else None,
             RECIPES[1]: "full_HW" if full_r else RECIPES[0] if same else None})


def summarize_banks(sessions, reference_panel, *, prior_valid, prior_patience, prior_alternatives, prior_nulls):
    if (set(sessions) != set(SESSIONS) or reference_panel["recipe"] != REFERENCE
            or len(reference_panel["sessions"]) != 16
            or {r["session_id"] for r in reference_panel["sessions"]} != set(SESSIONS)
            or any(type(value) is not int or value < 0 for value in (prior_valid, prior_patience, prior_alternatives, prior_nulls))
            or not 28 <= prior_valid <= 64):
        raise ValueError("Complete original sessions/reference and explicit live accounting required")
    originals = validate_original_selections(
        {s: sessions[s][RECIPES[0]]["full_HW_bank"] for s in SESSIONS}, reference_panel)
    chosen_banks = {s: sessions[s][RECIPES[0]] for s in SESSIONS}
    selected, paired, seed_rows = {r: [] for r in RECIPES}, [], [[] for _ in REPLICATES]
    anchors = {name: [] for name in ("independently_selected", "intersection_lambda_matched")}
    for session in SESSIONS:
        index, training, scores = select_paired_lambda(chosen_banks, session)
        original = originals[session]
        old_index = original["lambda_index"]
        if (original["lambda"] != LAMBDAS[old_index]
                or set(original["outer_training_sessions"]) != set(SESSIONS)-{session}):
            raise ValueError("Preserved full-HW independent outer selection required")
        for recipe in RECIPES:
            bank = sessions[session][recipe]
            row = {"session_id": session, "animal": ANIMALS[session], "lambda_index": index,
                "outer_training_sessions": training, "intersection_inner_animal_balanced_scores": scores,
                "baseline_R2": bank["baseline_R2"], **bank["lambda_rows"][index]}
            row["selected_supports"] = [{"support": v["support"], "fit_key": v["fit_key"],
                **v["bank"]["lambda_rows"][index], "denominator": v["bank"]["denominator"],
                "selected_fit_diagnostics": [{**{k: d[k] for k in ("held_direction", "fitting_bin_rows", "trace", "numerical_rank")},
                    "effective_df": d["ridge_df_by_lambda"][index]["effective_df"]} for d in v["fit_diagnostics"]]}
                for v in bank["views"]]
            selected[recipe].append(row)
        candidate, control = (selected[r][-1] for r in RECIPES)
        paired.append({"session_id": session, "animal": ANIMALS[session], "lambda": LAMBDAS[index],
                       **{k: candidate[k]-control[k] for k in ("delta_R2", "identity_increment")}})
        for replicate in REPLICATES:
            view = sessions[session][RECIPES[1]]["views"][replicate]
            seed_rows[replicate].append({"session_id": session, "animal": ANIMALS[session],
                "lambda": LAMBDAS[index], **view["bank"]["lambda_rows"][index]})
        bank = chosen_banks[session]
        for name, j in (("independently_selected", old_index), ("intersection_lambda_matched", index)):
            full = bank["full_HW_bank"]["lambda_rows"][j]
            anchors[name].append({"session_id": session, "animal": ANIMALS[session],
                "lambda": LAMBDAS[index], "full_HW_lambda": LAMBDAS[j],
                "delta_R2": (full["candidate_SSE"]-candidate["candidate_SSE"])/bank["denominator"],
                "identity_increment": candidate["identity_increment"]-full["identity_increment"]})
    charges, duplicates = recipe_charges(sessions)
    if prior_valid+sum(charges.values()) > 64:
        raise ValueError("Original maximum scientific depth cannot be exceeded")
    panels = []
    for recipe in RECIPES:
        panel = {"recipe": recipe, **aggregate_rows(selected[recipe])}
        screens = {"primary_at_least_0p01": panel["primary_increment"] >= .01,
                   "identity_at_least_0p005": panel["identity_increment"] >= .005, **directional_screens(panel)}
        panel.update(reference_margin_screens=screens, all_reference_margin_screens_pass=all(screens.values()),
            scientific_trial_count_increment=charges[recipe], sensitivity_only=True, promotion_authorized=False)
        panels.append(panel)
    count, patience, transitions = prior_valid, prior_patience, []
    for recipe in RECIPES:
        count += charges[recipe]
        patience += charges[recipe]
        transitions.append({"recipe": recipe, "charge": charges[recipe], "completed_count": count,
                            "patience": patience, "incumbent_reset": False})
    return {"panels": panels, "intersection_minus_random_support": aggregate_rows(paired),
        "random_support_panels": [{"replicate": r, **aggregate_rows(rows)} for r, rows in zip(REPLICATES, seed_rows)],
        "reference_contrasts": {name: {**aggregate_rows(rows),
            "primary_formula": "(SSE_full_HW-SSE_intersection)/D_original"} for name, rows in anchors.items()},
        "scientific_trial_count_by_recipe": charges, "structural_duplicate_of": duplicates,
        "scientific_trial_count_increment": sum(charges.values()),
        "stress_diagnostic_count_increment": sum(charges.values()), "literal_null_control_count_increment": 0,
        "alternative_explanation_count_increment": sum(charges.values()),
        "prior_valid_comparisons": prior_valid, "prior_patience": prior_patience,
        "prior_alternative_checks": prior_alternatives, "prior_literal_null_controls": prior_nulls,
        "total_valid_primary_trials_including_prior": count, "post_initial_comparisons": count-6,
        "post_initial_null_controls": prior_nulls,
        "post_initial_alternative_checks": prior_alternatives+sum(charges.values()),
        "recipe_order": list(RECIPES), "prospective_patience_transitions": transitions,
        "sensitivity_only": True, "incumbent_reset": False, "promotion_authorized": False,
        "search_complete": False, "spike_contamination_identified": False,
        "limitation": "Guide membership does not identify spike contamination; this remains an unresolved scientific limitation."}
