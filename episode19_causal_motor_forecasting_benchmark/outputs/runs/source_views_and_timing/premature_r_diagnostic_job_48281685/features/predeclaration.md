### October 17: premature-R diagnostic and control-precision check — pre-declared before execution

This follow-up is post hoc to v5. Added synthetic shapes, cutoffs, coverage requirements and operational choices are assumptions. It asks whether the v5 tail of long lags (L − R ≥ 288 ms) comes from a premature reference onset R produced by low-amplitude activity or drift (H_A), or from a detector that misses substantial movement before its label L (H_B). Accelerometer continuity cannot establish whether a departure belongs to the reach.

1. Use only the fifteen development people and the supplied accelerometer/TRIG readers: no EEG, WAY or held-out data. The D2 labels, the v5 k=5 onsets, all existing artifacts and both v5 UNDERIDENTIFIED outcomes stay unchanged, and nothing is frozen.
2. Verify the input closures and the fixed counts (3,964 events, 3,962 determinate, 291 tail). Write only into fresh diagnostic directories and stages, and re-check the four new scripts' hashes between stages.
3. Run in an actual Slurm compute shell with at most eight workers and one BLAS/OpenMP thread each. Participant subsets are allowed only in health mode. Health mode writes engineering counts and statuses only: no feature values, synthetic outcomes, decisions, provider output or figure.
4. Stage order: synthetic checks; real features (PRD decision); CPC strata/regression; CPC controls (CPC decision). Provider AccStartIndex is read only after both decisions are closed, with verified cue/start joins, explicit matched denominators and no provider flag filtering.
5. Compare tail events (determinate lag ≥ 288 ms) with reference events (0 ≤ lag < 288 ms) within each person. Require native validity over [ws − 720, L + 100], where ws is the first qualifying-window start, and a later-segment amplitude A_F ≥ θ5. Count every exclusion.
6. Fixed quantities:
   - b0 = mean a over [R − 520, R − 20) and the departure d = a − b0;
   - the later bout segment F = [L, L + 100) and A_F = ||mean d over F||;
   - H = L − t½, where t½ is the first native 40-ms window mean reaching half of A_F (searched through L + 60). If half is never reached, H is censored below −60 ms.
   Quarter-amplitude H, ρ_R, Λ and the moving-baseline versions are sensitivities.
7. Continuity: C_E and C_L are the cosines between the F vector and the 40-ms mean departure at R (C_E) or just before L (C_L). Successive non-overlapping 40-ms windows are descriptive. Also report:
   - the pre-R baseline slope and its 0.27-s residual scale;
   - whether removing that trend un-qualifies the first window ("drift explained"), and the trend's alignment with F;
   - anchor occupancy, prior activity, bout gaps, sample occupancy and rise_per_s.
   Neither monotone acceleration nor a link between sensor direction and reach direction is assumed.
8. Statistics: within-person median differences (proportion differences for indicators), requiring at least three tail and twenty reference observations per feature. H censoring bounds are propagated. People are summarized by their median, with 10,000 person-bootstrap resamples (seed 20261017); the intervals are diagnostic and unadjusted.
9. Synthetic known positives use the 32 cached v5 backgrounds per person and orthogonal direction pairs from SeedSequence([20261017, pid, 7]). S = 1920 ms. Labels are actual D2 labels after a valid, quiet 500-ms arming history. Noiseless flat axes are exempt from the noise-validity screen.
10. Shapes: each sustained ramp is shifted so that its noiseless R is 2160 ms (2163 ms at 1 kHz).
    - KP0 (reference) is an 80-ms raised-cosine step to 4·HIGH.
    - KPA (H_A type) adds a perpendicular drift whose residual is ≥ 1.3·θ5 while X stays ≤ 0.85·LOW.
    - KPB (H_B type) is a collinear ramp (residual ≥ 1.05·θ5, X ≤ 0.98·LOW), plus a small step that crosses LOW and a later HIGH step after F. Nine fixed candidates are tried in order (delays {350, 320, 380} ms × LOW factors {1.3, 1.2, 1.4}); the first with a supported 300–400-ms noiseless lag is used, without consulting any feature.
11. Synthetic set rules mirror the real ones: KP0 events need a noisy reference lag, and KPB/KPA events a noisy tail lag. A person is covered with ≥ 24 usable paired backgrounds for both KPB and KPA. Interpret only the decision set (feature-eligible, covered people), which must include at least twelve people. Sub-04 is expected to be KPA-uncovered: its LOW/θ5 is 1.24, while KPA needs about 1.47.
12. Synthetic contrasts (KPx − KP0 per person) use the same function as the real contrasts. KPB_power: a ΔH estimate ≥ 100 ms with CI lower bound > 0, a ΔC_E CI lower bound > −0.2 and a KPB median C_E ≥ 0.5, so the rule returns O1 on H_B-type data. Per-person synthetic values are descriptive, not gates.
13. KPA_power: a ΔH CI inside [−50, 50] ms, a ΔC_E CI upper bound < −0.2 and a KPA median C_E < 0.5, so the rule returns equivalence plus discontinuity on H_A-type data. KPA_drift_power also requires a drift-explained Δ CI lower bound > 0, a KPA median drift cosine < 0.5 and a KPB median drift cosine ≥ 0.5.
14. Weakens premature R from low-amplitude activity/drift (H_A): O1, a ΔH estimate ≥ 100 ms with CI lower bound > 0 and a ΔC_E CI lower bound > −0.2, given KPA_power. Tail departures then reach half the later amplitude at least 100 ms further before L than reference departures do. O1 motivates testing an anchored-baseline or slope-sensitive detector. It does not establish reach identity or a worst-case bound.
15. Weakens the detector missing reach movement (H_B): O2 or O3. In both, the ΔH CI lies inside [−50, 50] ms, so the extra L − R is spent below half the later amplitude.
    - O2 adds either a discontinuity (ΔC_E CI upper bound < −0.2 and tail median C_E < 0.5) or a drift explanation (drift-explained Δ CI lower bound > 0 and tail median drift cosine < 0.5).
    - O3 instead shows continuity (ΔC_E CI lower bound > −0.2, tail median C_E ≥ 0.5) with no positive drift signature.
    These weaken a substantial miss. They do not validate R or L, or prove that drift is absent.
16. Each statement is evaluated separately with its dependencies: H equivalence needs KPB_power; O1, the discontinuity branch and O3's continuity need KPA_power; the drift branch needs KPA_drift_power. A failed dependency makes only that statement UNDERIDENTIFIED. A supported O1 is reported first, then O2, then O3; otherwise the outcome is O4 (UNRESOLVED, or UNDERIDENTIFIED with reasons).
17. CPC keeps v5's 32 selected backgrounds and their IDs first, then appends every remaining eligible inventory row in session/run/start-tone order. One shared control loop must reproduce every first-32 v5 field exactly before any expanded output; a mismatch stops the stage.
18. Development-slope strata (descriptive):
    - Rise rate = (mean r over [R + 40, R + 80) − mean r over [R, R + 40)) / (0.04 s · θ5), from each person's determinate D2 k=5 events.
    - Strata sit at the person's q10 and q50, for linear and quadratic shapes.
    - Amplitudes come from a doubling bracket, 4,000 geometric grid points per phase and thirty local refinements. Choose the closest rate among phase/cap-valid candidates (ties go to the smaller amplitude) and require ≤ 5% relative error.
    - Target and achieved rates are recorded, and infeasible strata stay uncovered.
19. Keep all six v5 families, the exact targets and truths, the phase/cap/anchor/association rules, k ∈ {3, 5, 8}, and seed key 2 for original directions (key 4 for strata). The gates are unchanged: near ≥ 0.90; long and linear/quadratic fast ≥ 0.95; coverage ≥ 0.95; twenty backgrounds per required band; sham ≤ 0.05; all fifteen people.
20. Report, per trace, the target-320 lag error (and the opposite-signed onset error). Also report margin curves for targets 272–320 and gate intervals from 2,000 background resamples (seed keys 20/21 for all/original backgrounds). Population means and curves use 10,000 equal-weight person resamples.
21. CP1 places each interval below, across or above its unchanged gate, or marks it uncovered. An interval crossing a gate is inconclusive: it does not show that the v5 failure was sampling noise. A systematic synthetic precision failure cannot be repaired by detector changes alone.
22. CP2 calls a cell timing-error consistent only if all of the following hold; otherwise it names association/anchor or target-dependence/coverage limits:
    - the target-320 error covers ≥ 95% of traces;
    - it mispredicts ≤ 1% of near rows;
    - ≤ 1% of near rows are unresolved;
    - the predicted near fraction differs from the observed one by ≤ 0.01.
23. CP3 compares development-slope near sensitivity with 0.90 (descriptive). CP4 reports shams, alarm reasons and unanchorable backgrounds. Sub-02's reported v5 sham fraction (0.070) and two unanchorable backgrounds are retained. Larger prospective samples, reference-resolution changes and gate questions remain scientist decisions.
24. Before this declaration the orchestrator had seen the v5 per-person tail counts, noise references and the pooled v5 control-error distribution. Alternatives compared: three outcome patterns, two H_A signatures, nine KPB candidate shapes (chosen on lag/support only), one KPA shape, one KP0 shape and four descriptive slope strata.
25. Report one 300-dpi six-panel figure labelled "development calibration/diagnostic evidence; post hoc", and exactly three summary lines. Design implications:
    - O1: test an anchored or slope-sensitive detector.
    - O2: run a new reference-association/detrending test; no threshold reduction is indicated.
    - O3: the onset-definition question goes to the scientist before a lower or anchored LOW is tested.
    - O4: no automatic change.
    Nothing here grants EEG scoring clearance.
