# Campaign index

EP01--EP17 except EP18, plus EP19--EP20, are the 19 current formal episode
slots. Each slot has one direct `GOAL.md`, `DATASETS.md`, and search policy;
EP02 uses `SEARCH_POLICY.json` and the other 18 use `SEARCH_POLICY.yaml`. There
is no parallel restart tree or numbered local revision chain. EP18 is the sole
incomplete local draft, so it is not counted or launchable. The two old NARPS
directories remain immutable prior records rather than current episodes.

Local files describe the intended research contract. Explicit scientist
invocation starts an episode-managed run; Brain Researcher MCP remains
available for later review, reward, and accepted scientific transitions. All
current policies are unregistered. Their legacy `launch_blocked` fields are
inert authoring metadata and are not consulted by the local launcher.

## Episode-managed launch

`bin/codex-episode` is deliberately thin. It checks only the direct-child
workspace, required core files, scratch location, and active-process lock,
then starts or resumes Codex in that bounded workspace. Contract versioning,
data validation, runtime checks, adaptive search, falsification, and held-out
evaluation belong to the episode rather than the launcher. The older readiness
projections remain optional records for a future confirmatory/canonical
workflow; they do not authorize or block local episode execution.

## Current episodes

| EP | Current adaptive question | Valid-trial range | Episode-held-out stage |
| --- | --- | ---: | --- |
| EP01 | Which spatial, masking, scaling, estimation, and aggregation mechanism explains S0-to-S8 map change and transports beyond NARPS? | 30--60 | One locked `ds000005` gain/loss transport audit |
| EP02 | Does physiological mesolimbic dopamine gate the magnitude of policy updates rather than provide a signed teaching/reinforcement signal? | 28--48 | Intended one-shot whole-mouse 6-versus-5 intervention audit; opening is currently prohibited because generic calibration v2 returned `selected_rule: null` and the role handoff is unsealed |
| EP03 | Which condition-semantics representation transports across datasets when predicting effect-map geometry? | 30--72 | New independent dataset groups |
| EP04 | Which language/vision components predict high-level visual-cortex geometry under sealed stimulus shift? | 24--64 | Untouched OOD pool or new compatible data |
| EP05 | Which compact LFP representation retains condition-general kinematic information across whole sessions? | 28--64 | Prospective compatible whole sessions |
| EP06 | Which M1-versus-PMd recording-domain fingerprint is stable across sessions while exposing hardware confounding? | 20--48 | Held sessions internally; third animal for external generalization |
| EP07 | Which source-pooling and alignment policy adds value for an already calibrated target session? | 32--80 | Locked trials from calibrated target sessions |
| EP08 | Which post-pilot electrode-retention and calibration-trial policy preserves motor information most efficiently? | 40--112 | Four held sessions internally; third animal for external claims |
| EP09 | Which dendritic representation predicts distal axonal arbors beyond anatomy and metadata? | 40--96 | Group-sealed biological audit |
| EP10 | Which sparse, low-rank, or hierarchical model predicts residual co-projections beyond a searched Q0? | 40--88 | Group-sealed audit plus 99 complete search-null reruns |
| EP11 | Do reusable projection-morphology modes require discrete components, continuous gradients, or a hybrid? | 48--120 | At least eight sealed biological groups plus 99 search-null reruns |
| EP12 | Are curated MaleCNS types sufficient, or do reproducible residual wiring modes improve held-out-type prediction? | 36--96 | Whole-type audit within one male plus 99 search-null reruns |
| EP13 | Which exact BOLD-identifiability conjecture survives counterexample-guided repair and proof checking? | 16--64 | Exact proof/refutation; finite audit is implementation evidence only |
| EP14 | Which CPU-scale, site-free ROI representation improves unseen-acquisition brain-age prediction without increasing acquisition decodability? | 48--120 | Locked probe followed by one-shot 757-row audit |
| EP15 | Which constrained class adequately predicts reliable individual Task-B-only cerebellar variation from anatomy and Task A: parcel membership, smooth relocation, shared isometry, or stable non-isometry? | 40--96 | One locked opening for 12 participant-level Task-B outcomes |
| EP16 | Which operational signature explains excess long-lag failure of an offline BrainGate source-session target-direction mapping: observable measurement sufficiency, low-budget remappability, reduced session-local recoverable signal, or reproducible heterogeneity? | 40--96 | One locked 3-participant tri-state adequacy opening after structural 6/3 assignment and immutable evaluator-side scorability |
| EP17 | Which controlled visual-model relations are robust, measurement-sensitive, conditional, or underidentified under isolated beta/support/ceiling contracts? | 32--72 | One concept-disjoint CNeuroMod audit in the same four participants, without audit refitting |
| EP19 | Does EEG add strict prospective movement-onset information beyond past-only peripheral/context signals, and does model rank survive the horizon change? | 24--40 | One joint WAY series 8+9 and whole-participant self-paced audit opening |
| EP20 | Which legal NeuroCam-derived hardware/software design robustly improves a paper-derived virtual reference frontier? | 32--64 | One sealed structural/device shift plus post-lock empirical plausibility diagnostic |

EP02's exact role-pack materialization passed integrity checks, but its handoff
state is `materialized_unsealed`: same-UID access to the provider source and
network reacquisition prevent a confidentiality claim. Its outcome-blind
generic calibration v2 returned `selected_rule: null` with 0 of 216 eligible,
so no audit opening or positive mechanistic terminal is authorized. Endpoint and
time-axis semantics, the assignment mechanism or a presigned conditional-
exchangeability basis, endpoint-faithful calibration with an absolute raw-unit
margin, scientist signoff, the author-code port/configuration and resource
profile, a permission-separated evaluator, and canonical registration remain
unresolved for EP02's own confirmatory stage; they do not block starting the
episode.

## Shared governance

- EP01 treats both legacy NARPS runs and all `ds001734` outcomes as exposed
  development lineage. They count as zero new trials and cannot satisfy its
  one-shot external audit.
- EP05--EP08 share one LFP exposure/split ledger and are correlated evidence.
- EP09--EP11 share one SEU-A1876 exposure/split ledger and are correlated
  evidence.
- EP17 no longer uses NSD and has no shared-data group with EP04. Before its
  sealed CNeuroMod concepts open, EP17 must either reserve concept-disjoint
  roles with EP18 or freeze EP18's complete stimulus roles, feature controls,
  and decision rule; later evidence is otherwise correlated/design-exposed.
- Every episode requires branch coverage, at least two outcome-adaptive
  successor cycles, at least two incumbent/challenger decisions, and at least
  40% post-coverage falsification or ablation work.
- The incumbent is nonterminal. One configuration is locked before the
  episode's held-out evaluation, which opens at most once under its policy.
- Continued search after audit is a new episode with a new audit source, not a
  numbered local revision of the same run.

## Historical NARPS priors and current EP02

| Prior ID or slot | Former alias | Portfolio role |
| --- | --- | --- |
| `narps_prior_v1` | EP01 | Frozen, fully exposed development prior; old canonical loop is not reusable |
| `narps_prior_v2` | EP02 | Frozen, fully exposed nonterminal development prior; its returned reward action is expired and non-replayable here |
| EP02 | historical alias belongs only to `narps_prior_v2` | Current formal Dudman F20-17 adaptive contract; planned and not yet run, and not the audit half of EP01 |

EP18 remains an incomplete draft record, not a reserved empty ID and not a
formal episode. Its core scientific contracts are present, but it still needs
input/output guards and seven workspace projections before entering this
table.

The reusable design under `_examples/narps_deep_search_template/` is now
instantiated as the new formal EP01 at `episode01_narps_deep_search/`; the
example remains as design provenance.

## Evidence status

Zero episodes currently count as realized adaptive-search evidence. The direct
contracts, including the new EP01, are structurally deep enough, but none has a
realized append-only trial lineage, configuration lock, or held-out result.
`ADAPTIVE_EVIDENCE_POLICY.yaml` remains a post-run classification policy. These
checks are performed inside a launched episode and may legitimately end it as
`technical_failure` or `closed_no_candidate`; they are not prelaunch gates.
