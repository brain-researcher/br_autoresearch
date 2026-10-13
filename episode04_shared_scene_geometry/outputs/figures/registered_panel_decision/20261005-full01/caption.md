# EP04: complete registered development-panel decision

Recorded status: `closed_no_candidate`. This figure preserves the saved decision;
it does not select, promote or lock a candidate. An indicated candidate remains
pending its registered exact full development rerun. OOD access and outcomes
remain closed.

**A. Native gain.** All five participant gains for each of the three VLMs are
the saved native RSA VLM-minus-DINOv2 Fisher-z contrasts, averaged over that
participant's five folds and three primary ROIs. White diamonds show the saved
equal-participant mean (`effects.gain`), without recomputing it. Reference lines
mark zero and the registered practical mean-gain threshold of 0.02. Individual
participants are not required to reach 0.02. No uncertainty intervals are shown.

**B. Alignment and null thresholds.** Points show saved *unsubtracted* VLM
alignment, averaged equally across participants, folds and primary ROIs. These
are alignment values, not VLM-minus-DINOv2 gains. The two lines separately show
the stored familywise image-permutation 95th percentile
(0.002948835579241263) and random-geometry 95th percentile
(0.000721504297824517). The registered quantile uses `higher`
and requires strict exceedance. Neither distribution nor threshold is recomputed.

**C. Complete gates.** All twelve saved Boolean gates are shown for every model.
PASS/FAIL is copied from the decision, including the five sensitivity gates;
the plot does not reevaluate or combine them into a new decision.

This is selection-exposed development evidence from five participants, not
population confirmation. All models and participants remain visible regardless
of the recorded outcome. The figure supplies no OOD result or new inference.

Sole scientific input: `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/runs/panel_decision/registered_panel_decision.json`.
Rendering: Matplotlib 3.8.3; no source arrays or other reports read.
