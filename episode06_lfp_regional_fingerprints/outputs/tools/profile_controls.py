"""EP06 paired-profile omission and band-label sensitivity controls.

Inputs are already computed PairedSessionProfiles, never neural arrays or
files. Seven omissions remove a session from the full comparison, refit both
source-animal classifiers, and evaluate the remaining opposite-animal rows.
This is a sensitivity analysis, not prediction of the omitted session.

The fixed band control permutes score/error payloads into canonical singleton
band slots independently by session, sharing each permutation across its two
physical regions. PCG64 SeedSequence(6006202, animal_index, original_position)
is drawn once within the supplied candidate band set. Identity draws remain;
excluded bands never enter. Both classifiers are then refitted. These are two
scientific controls, not fourteen new candidates or a permutation p-value.
Accuracy need not fall to chance for the sensitivity result to be valid.

Each control independently becomes wholly unavailable if any required fit or
support is undefined. Partial values and all seven omission slots are kept;
no available-fold average replaces a complete result. Raw signed-profile
agreement is reported separately and never substituted for classifier refits.
The caller owns candidate selection, admission, resource limits and reporting.
"""
from dataclasses import asdict, replace

import numpy as np

import profile_classifier as classifiers
import profile_references as references
from primary_profile import paired_profile


ANIMALS = ("Mihili", "Chewie-L")
ROSTERS = (
    ("Mihili_CO_20140217.mat", "Mihili_CO_20140218.mat", "Mihili_CO_20140304.mat"),
    ("Chewie_CO_20160929.mat", "Chewie_CO_20161005.mat", "Chewie_CO_20161007.mat",
     "Chewie_CO_20161014.mat"),
)
ORIGINAL_POSITIONS = ((1, 2, 4), (1, 2, 4, 5))
BAND_PERMUTATION_SEED = 6006202


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource/cancellation check required")
    check()


def _panel(mihili, chewie, mihili_sessions, chewie_sessions, check):
    _check(check)
    if (mihili_sessions, chewie_sessions) != ROSTERS:
        raise ValueError("exact ordered seven-session development rosters bind original-position seeds")
    panels, rosters, blocks, missing = references._panel(
        mihili, chewie, mihili_sessions, chewie_sessions, check)
    if blocks and any(len(block.bands) != 1 for block in blocks):
        raise ValueError("band-label control requires the candidate's declared singleton bands")
    return panels, rosters, blocks, missing


def _classify(panels, rosters, source, recipe, check):
    _check(check)
    fitted = classifiers.fit_profile_classifier(panels[source], source_animal=ANIMALS[source],
        expected_sessions=rosters[source],
        labels=references._labels(ANIMALS[source], rosters[source], (0,) * len(rosters[source])),
        recipe=recipe, check=check)
    predicted = classifiers.predict_profile_classifier(fitted, panels[1 - source],
        target_animal=ANIMALS[1 - source], expected_sessions=rosters[1 - source], check=check)
    score = classifiers.score_profile_predictions(predicted,
        labels=references._labels(ANIMALS[1 - source], rosters[1 - source], (0,) * len(rosters[1 - source])),
        check=check)
    return {"source_animal": ANIMALS[source], "target_animal": ANIMALS[1 - source],
        "source_sessions": list(rosters[source]), "target_sessions": list(rosters[1 - source]),
        "balanced_accuracy": score.balanced_accuracy, "tied_rows": score.tied_rows,
        "class_0_recall": score.class_0_recall, "class_1_recall": score.class_1_recall,
        "predictions": [asdict(row) for row in predicted.rows]}


def _omission_slots():
    return [{"omitted_animal": animal, "omitted_session": session,
             "directions": [None, None], "complete": False,
             "equal_direction_mean_balanced_accuracy": None, "unavailable_reason": None}
            for animal, roster in zip(ANIMALS, ROSTERS) for session in roster]


def _leaveout(panels, rosters, recipe, missing, check):
    result = {"control": "leave_one_development_session_out", "complete": False,
              "fit_attempts": 0, "unavailable_reason": missing, "omissions": _omission_slots()}
    if missing is not None:
        return result
    for omitted_animal, panel in enumerate(panels):
        for omitted_index in range(len(panel)):
            _check(check)
            slot = result["omissions"][sum(len(x) for x in panels[:omitted_animal]) + omitted_index]
            reduced_panels = list(panels)
            reduced_rosters = list(rosters)
            reduced_panels[omitted_animal] = panel[:omitted_index] + panel[omitted_index + 1:]
            roster = rosters[omitted_animal]
            reduced_rosters[omitted_animal] = roster[:omitted_index] + roster[omitted_index + 1:]
            for source in range(2):
                result["fit_attempts"] += 1
                try:
                    slot["directions"][source] = _classify(
                        reduced_panels, reduced_rosters, source, recipe, check)
                except classifiers.ClassificationUnavailable as error:
                    slot["unavailable_reason"] = str(error)
                    result["unavailable_reason"] = str(error)
                    return result
            slot["complete"] = True
            slot["equal_direction_mean_balanced_accuracy"] = .5 * sum(
                direction["balanced_accuracy"] for direction in slot["directions"])
    result["complete"] = True
    return result


def permute_paired_band_labels(mihili_profiles, chewie_profiles, *,
                              mihili_sessions, chewie_sessions, check):
    """Return ((Mihili, Chewie) paired-profile tuples, permutation ledger).

    Ledger indices are destination-slot -> original-payload indices. Original
    profiles, trial/axis identities, residual energies and canonical block
    labels remain untouched; the returned profiles intentionally relabel bands.
    """
    panels, _, blocks, missing = _panel(mihili_profiles, chewie_profiles,
                                      mihili_sessions, chewie_sessions, check)
    if missing is not None:
        raise classifiers.ClassificationUnavailable(missing)
    shuffled, ledger = [], []
    for animal_index, panel in enumerate(panels):
        transformed = []
        for supplied, position in zip(panel, ORIGINAL_POSITIONS[animal_index]):
            _check(check)
            seed = [BAND_PERMUTATION_SEED, animal_index, position]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
            indices = tuple(int(value) for value in rng.permutation(len(blocks)))
            profile = supplied.profile
            regional = []
            for region in (profile.m1, profile.pmd):
                scores = tuple(replace(region.blocks[source], block=blocks[destination])
                               for destination, source in enumerate(indices))
                regional.append(replace(region, blocks=scores))
            transformed.append(replace(supplied, profile=paired_profile(*regional, check=check)))
            ledger.append({"animal": supplied.animal, "session_id": profile.session_id,
                "original_session_position": position, "seed_entropy": seed,
                "destination_to_source_indices": list(indices),
                "identity_permutation": indices == tuple(range(len(blocks)))})
        shuffled.append(tuple(transformed))
    return tuple(shuffled), tuple(ledger)


def _band_control(panels, rosters, recipe, missing, check):
    result = {"control": "paired_session_band_label_permutation", "complete": False,
        "fit_attempts": 0, "unavailable_reason": missing, "permutations": None,
        "directions": [None, None], "equal_direction_mean_balanced_accuracy": None,
        "signed_agreement_permuted": None, "interpretation": "sensitivity_no_automatic_chance_veto"}
    if missing is not None:
        return result
    transformed, ledger = permute_paired_band_labels(*panels,
        mihili_sessions=rosters[0], chewie_sessions=rosters[1], check=check)
    result["permutations"] = list(ledger)
    result["signed_agreement_permuted"] = asdict(references.signed_profile_agreement(
        *transformed, mihili_sessions=rosters[0], chewie_sessions=rosters[1], check=check))
    for source in range(2):
        result["fit_attempts"] += 1
        try:
            result["directions"][source] = _classify(transformed, rosters, source, recipe, check)
        except classifiers.ClassificationUnavailable as error:
            result["unavailable_reason"] = str(error)
            return result
    result["complete"] = True
    result["equal_direction_mean_balanced_accuracy"] = .5 * sum(
        direction["balanced_accuracy"] for direction in result["directions"])
    return result


def evaluate_profile_controls(mihili_profiles, chewie_profiles, *,
                              mihili_sessions, chewie_sessions, recipe, check):
    """Return independent leaveout/band controls and the separate raw signed report.

    The baseline classifier result is supplied by the caller's previously
    selected candidate report; this helper does not refit that unchanged
    baseline, select a candidate, average incomplete folds or assign a p-value.
    """
    recipe = classifiers._classifier_recipe(recipe)
    panels, rosters, blocks, missing = _panel(mihili_profiles, chewie_profiles,
                                            mihili_sessions, chewie_sessions, check)
    signed = references.signed_profile_agreement(*panels, mihili_sessions=rosters[0],
                                                chewie_sessions=rosters[1], check=check)
    leaveout = _leaveout(panels, rosters, recipe, missing, check)
    band = _band_control(panels, rosters, recipe, missing, check)
    _check(check)
    return {"kind": "EP06_profile_sensitivity_controls", "classifier_recipe": asdict(recipe),
        "blocks": [asdict(block) for block in blocks], "label_definition": {"M1": 1, "PMd": 0},
        "signed_agreement_original": asdict(signed), "leave_one_out": leaveout,
        "band_permutation": band, "scientific_control_count": 2,
        "claim_scope": "development_sensitivity_controls_only", "scientific_acceptance": None}
