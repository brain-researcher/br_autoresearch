# Bounded engineering retry for the guide-only intersection qualification

Prepared 2026-10-04 after root's complete technical-failure report.
**Adopted engineering revision by root, 08:54 UTC.** Isolated implementation
and focused tests are authorized; source retry follows only after passing
code review and compute checks. Numerical intersection remains unlaunched.

## Observed failure and limits of the diagnosis

Preserve `runs/source_support/primary_intersection_metadata_attempt001`.
The original 60-second internal / 90-second external guide-only attempt began
08:49:09.281 UTC and ended 08:50:09.303 UTC with exit 1, elapsed 60.022 seconds,
29.298867 CPU seconds and 93,704 KiB peak RSS. Root reports an ordinary h5py
dereference of a `M1_spikes` shape handle in the traceback. The artifact is a
technical failure, not complete metadata; no scientific comparison is charged.
The recorded spike/LFP/XY value-read flags remain false. There is no numeric
intersection fit or score to release.

The code performs 17 field resolutions per retained trial: 10 from the
inherited identity/event `metadata()` method, 4 from `guides()`, and 3 further
shape handles. LFP and spike shape handles are resolved twice. Each inherited
`field()` call obtains a named reference-table dataset, reads one MATLAB cell
reference, and dereferences the target dataset. Constant guides are checked
again on every retained trial, as required.

These repeated operations are established by static inspection. Their share
of runtime was not profiled, and the timeout does not prove they are its sole
cause. Substantial CPU use and an ordinary dereference stack do not establish
stuck I/O, corrupted data, slow hardware or a filesystem fault. The failed
report did not retain per-session completion timing, so it cannot support an
extrapolation from a known completed fraction.

## Proposed access optimization with unchanged scientific qualification

Use a new isolated subclass of the qualified read-only `M1Session`, scoped
only to this metadata tool. Preserve the original adapter and numerical
readers untouched.

1. Once per open session, cache the small MATLAB cell-reference tables for
   the ten identity/event fields and the five guide/shape fields. These are
   HDF5 object references, not the numerical arrays they point to. Preserve
   each table's exact size and Fortran-order flattening, matching the original
   `unravel_index(..., order="F")` lookup. A field resolution retrieves the
   cached reference and still dereferences its original dataset read-only.
   Only original retained-trial references are dereferenced. References for
   other source trials may occur in the table buffer but are not followed,
   evaluated, added to eligibility or used for guide selection.
2. Keep every original identity, direction, event, source-window and native
   time check. Do not infer a trial's metadata from its neighbor, filename or
   a constant-field shortcut. Null or malformed references still fail.
3. Keep the original `qualify_session()` function and inherited `guides()`
   path unchanged. Cache resolved dataset handles only for the current trial,
   at most fifteen handles; clear that cache whenever the trial index changes.
   Its seventeen original field calls then require fifteen target dataset
   dereferences. The repeated LFP/spike shape calls return the same handle.
   Never index or convert the numerical LFP, spike or XY handles. This removes
   duplicate dereferences without duplicating or removing any qualification
   logic, guide check or simultaneous-time-axis check.
4. Continue checking literal guide-array invariance on every original trial,
   including unit labels zero or other finite stored labels. Keep the same
   original sixteen sessions, complete trials, physical join, no-empty rule,
   ordinary random-support literals, support order, duplicates and pilot
   workload formula. No outcomes select the retry or its resource bound.
5. Emit one bounded progress record per completed session with identity,
   elapsed time and guide-read counts only, and retain completed session
   metadata in a failure report. These are diagnostics, not permission to
   consume a partial artifact or skip already visited sessions on a retry.
   A complete artifact still requires all sixteen sessions.

Source-free fixtures must establish cached-reference equivalence with the
original lookup for both row/column MATLAB tables and nonconsecutive original
trial indices; null/size/index failure; no array conversion or slicing of
neural handles; unchanged guide/support/queue output; retained-trial-only
dereferencing; and all-sixteen completion gating. Run them only in root's
verified allocation. Exact guide-derived output equality is required;
resource/progress fields may differ.

## Explicit proposed retry bounds

After root adoption, code review and passing focused fixtures, propose a fresh
`runs/source_support/primary_intersection_metadata_attempt002` attempt with
**one CPU, 2 GiB, numerical threads one, 300 seconds internally and 330
seconds externally**. The increased finite time is a prospective engineering
allocation under the existing overall budgets, not an automatic extension,
scientific-policy change or guarantee of completion. The earlier attempt
demonstrated failure of the selected 60-second limit; the new limit is a
bounded planning choice, not a fitted runtime prediction.

Do not resume or overwrite attempt001, remove checks, skip sessions, change
guide semantics, read spike values or launch numerical fitting. Root records
all consumed resources and applicable engineering retry accounting; scientific
depth stays unchanged. If attempt002 fails, preserve it and diagnose that
failure before any further decision. Only a complete metadata release can
authorize the already adopted intersection numerical implementation.
