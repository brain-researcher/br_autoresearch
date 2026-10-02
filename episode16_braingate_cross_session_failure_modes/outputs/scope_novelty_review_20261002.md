# EP16 scope, novelty and clarity review — 2026-10-02

Question: across longer gaps, is less direction signal locally recoverable, is more of that signal lost in transport, and can measured recording change reproduce or 32 labels repair the loss?

## Closest work and the remaining contribution

[NoMAD (2025)](https://www.nature.com/articles/s41467-025-59652-y) already establishes long-term neural-decoder stabilization as a substantial research direction. [Wilson et al., PRI-T (2025)](https://www.nature.com/articles/s41551-025-01536-z) addresses long-term instability and iterative adaptation. Decoder drift, long-gap failure and recalibration success are therefore not new findings by themselves.

EP16's possible contribution is their common operational comparison: target-local recoverability, transported loss normalized by the same above-null signal, a label-blind recording-state emulator, and a fixed low-label recovery curve. Source/target difficulty and chronological support matter more than another model leaderboard.

## Decision and repair

Keep the question and component design. The result branch now reports coexisting operational signatures rather than mechanisms. An emulator can show that measured changes are sufficient under its assumptions; it cannot apportion biological causes. Repair by 32 labels similarly identifies recoverability, not a unique source of drift.

The fixed trial budgets and both transfer folds are essential interpretation conditions, not online adaptation demonstrations. The primary direction-decoding proxy does not establish restored closed-loop BCI performance. The nine-person cohort supports participant-level profiles, not population subtypes.

The existing unresolved choices about the population used for the recovery estimate and literal past-only versus reversed-fold exposure remain explicit design work; this writing review does not silently choose them. Paper strength is conditional on separating these accounts with adequate precision, not on finding any long-gap deficit.

No endpoint, margin, model family, trial role or resource permission changes. No empirical scoring was performed.
