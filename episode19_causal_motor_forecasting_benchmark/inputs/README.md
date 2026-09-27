# Episode 19 inputs

This directory is read-only and intentionally contains no mixed source
archive. The WAY-EEG-GAL, self-paced reaching, historical code, and deferred
AJILE12 sources remain in shared research storage until episode-specific data
views are prepared.

Before signal scoring, EP19 needs these separated views:

1. WAY series 1–7 for development;
2. WAY series 8–9 for the final held-out evaluation;
3. all data from the 15 self-paced development participants; and
4. all data from the eight self-paced held-out participants for the same final
   evaluation.

No such view is currently present. Public availability does not make a held-out
series or participant available to development code.

This directory will also need one readable file listing the finite model
changes and numeric ranges allowed during development. That list must be set
before a candidate EEG score is returned. A plain, readable table is enough.

Do not write fitted models, onset results, scores, caches, or temporary outputs
under `inputs/`. Authorized temporary extraction belongs in episode-specific
scratch.
