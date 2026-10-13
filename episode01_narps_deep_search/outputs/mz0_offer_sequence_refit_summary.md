# MZ0 offer-sequence omission refits

**Complete, 2026-10-05 16:28 UTC coordinator terminal confirmation: all 16 selected omissions completed.** ER-offers-03, the final component, finished at 16:27:58.425241 UTC; its task exited successfully at 16:27:59 UTC. This table uses the completed saved reports.

Each row deletes one complete version-specific offer group from both fitting and scoring in the same saved synthetic MZ0 cohort. All retained participants keep their original 256 trials and equal participant weights. EI and ER offer IDs are distinct; equal ordinal labels do not imply matched sequences. The adopted identities are the eight ordered-offer groups per version from the [qualified schedule report][schedule], not copied onset-sensitive identities.

Each omission freshly refits all retained subject and population parameters for M0 and MZ in both primary directions, `{1,4}` → `{2,3}` and the reverse. Likelihood, priors, starts, tolerances and the original 500-sweep ceiling remain unchanged. Each held-out half contributes its 128-trial log-score sum divided by 256; the two contributions give each person's full-session score. The runner uses four participant workers.

Means below are saved MZ-minus-M0 log-score advantages in nats/trial. **Refit mean** uses the omission-specific fitted parameters; **fixed-prediction mean** removes the same people from the original full-cohort score panel without refitting. EI/ER columns are the refitted version means. Values are displayed to 12 decimal places, and process times to six; source reports retain full precision. Sweep order is forward M0 / forward MZ / reverse M0 / reverse MZ.

| Omitted offer group / source report | Status | Retained N (EI / ER) | Refit mean | Fixed-prediction mean | Refit EI mean | Refit ER mean | Sweeps | Process wall seconds |
|---|---|---:|---:|---:|---:|---:|---|---:|
| [EI-offers-01][ei01] | Complete | 102 (48/54) | 0.012256023818 | 0.012217937868 | 0.009761179956 | 0.014473662806 | 129/141/86/108 | 229.574260 |
| [EI-offers-02][ei02] | Complete | 99 (45/54) | 0.012934388003 | 0.012922822863 | 0.011086212059 | 0.014474534624 | 130/142/89/108 | 467.325562 |
| [EI-offers-03][ei03] | Complete | 100 (46/54) | 0.013290016784 | 0.013328052797 | 0.011898439732 | 0.014475434272 | 133/145/95/112 | 462.229342 |
| [EI-offers-04][ei04] | Complete | 101 (47/54) | 0.013194785270 | 0.013243394345 | 0.011725474155 | 0.014473630129 | 129/141/85/108 | 457.506205 |
| [EI-offers-05][ei05] | Complete | 103 (49/54) | 0.012697909460 | 0.012709932144 | 0.010739734569 | 0.014474771861 | 127/140/85/106 | 482.175635 |
| [EI-offers-06][ei06] | Complete | 101 (47/54) | 0.012855343935 | 0.012849693782 | 0.010991381303 | 0.014477681782 | 133/145/102/111 | 480.766109 |
| [EI-offers-07][ei07] | Complete | 102 (48/54) | 0.012899101709 | 0.012925090903 | 0.011125892354 | 0.014475287802 | 131/143/92/109 | 537.614075 |
| [EI-offers-08][ei08] | Complete | 102 (48/54) | 0.013175859020 | 0.013195785912 | 0.011714845356 | 0.014474537833 | 135/148/93/114 | 559.918272 |
| [ER-offers-01][er01] | Complete | 99 (54/45) | 0.012565958707 | 0.012497637450 | 0.011149201544 | 0.014266067303 | 51/84/63/96 | 363.590818 |
| [ER-offers-02][er02] | Complete | 100 (54/46) | 0.012092856098 | 0.012042722181 | 0.011151130914 | 0.013198359575 | 91/108/57/95 | 417.624624 |
| [ER-offers-03][er03] | Complete | 102 (54/48) | 0.012976734727 | 0.013040021492 | 0.011156758566 | 0.015024207908 | 129/142/10/82 | 441.454808 |
| [ER-offers-04][er04] | Complete | 100 (54/46) | 0.012763462287 | 0.012814825133 | 0.011149155514 | 0.014658518063 | 156/168/95/116 | 553.197880 |
| [ER-offers-05][er05] | Complete | 102 (54/48) | 0.012644577294 | 0.012667063194 | 0.011151828132 | 0.014323920102 | 122/134/64/98 | 454.464566 |
| [ER-offers-06][er06] | Complete | 102 (54/48) | 0.013104932320 | 0.013092357097 | 0.011146389542 | 0.015308292945 | 86/99/94/114 | 437.611414 |
| [ER-offers-07][er07] | Complete | 102 (54/48) | 0.012494723946 | 0.012528688817 | 0.011148655110 | 0.014009051387 | 151/161/107/121 | 557.090020 |
| [ER-offers-08][er08] | Complete | 103 (54/49) | 0.012932692187 | 0.012905417619 | 0.011147555037 | 0.014899986189 | 153/163/113/128 | 587.384125 |

All 16 completed reports retain four converged fits under the original 500-sweep settings, four completed score components, no final subject optimizer failures and successful worker cleanup. Source and membership validation passed. The largest recorded primary-score replay discrepancy among these completed reports is 3.3306690738754696e-16, below the existing absolute tolerance of 1e-12. EI-offers-02 and ER-offers-03 specifically record zero. Recorded lower numeric-support contacts occur in the same 1 / 3 / 1 / 3 pattern: M0 has ER kappa; MZ has EI rho, ER kappa and ER rho. These are version log-standard-deviation contacts at -3.9999998503354774 on the declared [-4, 1] support; they remain reported rather than repaired or suppressed.

Both refitted version means are positive in every row. This table describes same-cohort deletion sensitivity only: the rows are dependent refits, not independent replications. No pooled omission mean, cluster confidence interval, unseen-template generalization, interval-recovery qualification or full-gate verdict is computed.

Times are each report's recorded process wall duration, not CPU charges. The pilot and three array tasks used separate four-CPU allocations with different scheduling conditions. Shared allocation time—including the separate bootstrap engineering step in task 2—is accounted once by the coordinator; it is not attributed again by summing or scaling these rows.

Sources: [original synthetic MZ0 report][original]; [EI-offers-01 pilot][ei01]; [remaining-omission array directory][array]. Each table label links its explicit completed report.

[original]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-SESOI-replicate000-attempt001/single_process_recovery.json
[schedule]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/public_asset_qualification/20261003-schedule01/schedule_support.json
[array]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480
[ei01]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refit-46627419-EI-offers-01-attempt001/mz0_offer_sequence_refit.json
[ei02]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-02/mz0_offer_sequence_refit.json
[ei03]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-03/mz0_offer_sequence_refit.json
[ei04]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-04/mz0_offer_sequence_refit.json
[ei05]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-05/mz0_offer_sequence_refit.json
[ei06]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-06/mz0_offer_sequence_refit.json
[ei07]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-07/mz0_offer_sequence_refit.json
[ei08]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/EI-offers-08/mz0_offer_sequence_refit.json
[er01]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-01/mz0_offer_sequence_refit.json
[er02]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-02/mz0_offer_sequence_refit.json
[er03]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-03/mz0_offer_sequence_refit.json
[er04]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-04/mz0_offer_sequence_refit.json
[er05]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-05/mz0_offer_sequence_refit.json
[er06]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-06/mz0_offer_sequence_refit.json
[er07]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-07/mz0_offer_sequence_refit.json
[er08]: /oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-offer-sequence-refits-46628480/ER-offers-08/mz0_offer_sequence_refit.json
