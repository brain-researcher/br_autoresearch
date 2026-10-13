# EP12 generated-data method freeze

Status: frozen for executable generated-data qualification. Real connectivity
outcome access remains disabled.

## Episode and storage boundary

| Role | Absolute location |
| --- | --- |
| EP12 | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency` |
| Read-only inputs | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/inputs` |
| Durable outputs | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs` |
| Temporary work | `/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency` |
| Read-only MaleCNS source | `/oak/stanford/groups/russpold/data/br_autoresearch_data/flyem_male_cns/gcs-male-cns-v1.0-flat-connectome` |

“Before launch” means before opening real connectivity outcome values. The
method implementation and generated-data tests in this directory are active
prelaunch work, not a reason to defer implementation.

## Five frozen choices

1. **Eligibility and roles.** A focal neuron must have a nonempty provider
   `type`, `status == Traced`, and `somaSide` equal to `L` or `R`. A type needs
   at least five focal neurons on each side. Whole types are split exactly
   80/20 by a deterministic SHA-256 key while balancing minimum-side-count
   deciles. The annotation-only preflight yields 1,020 eligible types: 816
   development and 204 final.
2. **Prediction object.** Every model predicts the same neuron-by-partner count
   vector under a conditional multinomial likelihood. Counts and totals are
   retained. Known partner types and unknown, untyped, fragment,
   proofreading, missing-annotation, and out-of-vocabulary mass remain
   explicit. Unreconstructed connections are not imputed.
3. **Bounded T/U/M families.** T is a type template plus common nuisance
   effects. U is selected from exactly two prespecified implementations: a
   rank-at-most-two linear logistic-normal factor model and a one-dimensional
   cubic-Hermite smooth connected curve. M has two or three group offsets,
   shared within-group covariance, and within-group continuous variation. No
   additional continuous family is added during qualification.
4. **Fair transfer and decision calculation.** Source-side cross-validation
   selects within each family. T, U, and M use the same likelihood, ILR basis,
   nuisance design, partner bins, and numerical integration settings. Target
   latent coordinates and M labels are marginalized; target refitting,
   reclustering, and alignment are forbidden. The score is held-out log
   predictive density per observed outgoing count. Whole types are the
   uncertainty unit, reciprocal directions are covered simultaneously, and
   the unchanged policy margin is 0.02 nats per observed outgoing count.
5. **Qualification and cost.** The same fitter used for the planned method is
   run on generated template, linear-continuous, curved-continuous,
   sparse-intermediate, grouped-with-continuum, weak-group, endpoint-loss,
   anatomy/quality-confounded, and influential-cell scenarios. Error rates,
   coverage, resolution, integration stability, numerical failures, and CPU
   cost must come from executions. Low-resolution strata remain in the table
   and force `unresolved`; they are not excluded.

The executable values and checks are in
`outputs/executor/SCIENTIFIC_CONTRACT.yaml` and
`outputs/executor/SYNTHETIC_QUALIFICATION_SPEC.yaml`. A changed YAML value that
the code does not implement causes qualification to fail rather than being
silently ignored.

## Claim boundary

Even after qualification, a positive EP12 statement is limited to evidence
for reproducible groups beyond the two qualified continuous references. The
generated-data stage cannot support a MaleCNS biological conclusion.

The fixed-method implementation is not yet the full adaptive executor in
`SEARCH_POLICY.yaml`: the wider partner-vocabulary, representation, rank,
covariance, and regularization grammar, mode-support intervals, complete
search null, and one-shot final evaluator remain closed gates. Real
connectivity must stay disabled until those gates and adequacy sensitivity
pass and a subsequent user instruction authorizes outcome access.
