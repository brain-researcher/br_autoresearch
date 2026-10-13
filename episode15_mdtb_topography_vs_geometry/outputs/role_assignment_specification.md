# EP15 exact Task-B-blind participant assignment

Numerical implementation adopted by root at21:05UTC on2026-10-04 after root
and independent static review and nine passing compute fixtures. Execution
is selected after root's21:23UTC conservative EP03 lineage review, recorded
in the experiment log. Participant demographics/private reliability remain
unread by this assigner until the selected compute attempt executes.
The review treats exposure as potentially correlated and does not establish
source independence or complete EP03 inventory coverage. This implements
DATASETS.md:77–104 and SEARCH_POLICY.yaml:data.role_assignment. It does not
activate margins, candidate scoring, Task-B access or the three role-filtered
data handoffs.

## Documentary lineage gate

DATASETS.md:196–198 requires comparison with EP03's source inventory before
roles freeze. Root has inspected only EP03's completed local metadata reports:
`metadata01/source_metadata_report.json`,
`design-support03/design_support_report.json`, and
`group-map-metadata02/group_map_metadata_report.json`. A search found no
`ds002105` in those reports; the last describes 31 accessions, 46 routes and
297 filenames. This is a limited current-inventory comparison, not evidence
of independence or of a complete intended 577-file inventory. No source
outcomes or stopped contributor-route data were opened. Root's21:23UTC log
entry completes the review through a conservative potentially-correlated
exposure determination and explicitly supersedes the earlier blocking
interpretation. Inventory completeness remains unresolved.

The assigner requires both `--lineage-review-complete` and a
`--lineage-review-section` identifying an existing, nonempty level-two section
in this episode's `outputs/experiment_log.md`. It checks this gate before any
demographic or private reliability read. Root records the inventory comparison,
remaining exposure uncertainty and decision in that section, then supplies
the explicit flag. The flag attests completion of the documented review; it
does not assert absent overlap, an independent corpus, full inventory coverage
or physical isolation. The program verifies the reference exists, not the
scientific truth of its prose. Missing confirmation or reference blocks the
whole operation. No new receipt file is required.

## Inputs and exact objectives

Use exactly the same 24 unique IDs as the completed metadata join. The trusted
process reads only the release's `participants.tsv`, the completed private
`task_a_reliability.csv`, and named local operational metadata. No archive,
image, regressor, task outcome or map is opened. Require the private producer's
successful all24 report, matching artifact path, complete finite-cohort flags,
unchanged support and zero Task-B access. Read each text input with a fixed
bound; the source participant table and private CSV each have a 1 MiB cap.
Do not alter or copy the private artifact.

Require participant columns `participant_id`, `sex`, `age`, `native_english`,
`ses-rest`, and the private CSV's two established fields. All IDs must occur
exactly once in both inputs and match the verified 24-person cohort. Treat
the established missing tokens (empty, n/a, na, nan, none, case-insensitive
after whitespace stripping) as one explicit level per categorical field.
For nonmissing categories strip outer whitespace but preserve case and exact
source spelling; do not merge distinct category codes. Missing/nonfinite age
or reliability blocks assignment. No imputation, exclusion or replacement.

Enumerate all `choose(24,12) = 2,704,156` sorted audit-ID tuples, including both
members of every complementary pair. Development is the complementary 12.
Minimize the following six components in exactly this order, then the sorted
audit-ID tuple:

1. Maximum absolute audit-minus-development count difference across every
   observed level of sex, native-English status and optional-rest status.
2. Sum of those absolute level-count differences, counting each level once.
3. Absolute age mean difference, standardized by the all24 population SD.
4. Age empirical 1-Wasserstein distance, using equal participant weights,
   standardized by the same SD.
5. Absolute Task-A reliability mean difference, with its all24 population SD.
6. Task-A reliability empirical 1-Wasserstein distance, with the same SD.

For either numeric variable, empirical W1 is
`sum_j abs(sorted_audit[j] - sorted_development[j]) / 12`.
If the variable has zero all24 spread, both of its components are identically
zero. This causes neither a variance floor nor a participant exclusion.

Interpret the serialized age and reliability decimal strings as exact
rationals. Decimal construction itself is exact; do not perform arithmetic
under a finite Decimal precision or round values into bins. Scale the 24
values to common-denominator Python integers once. For selection, compare
`abs(2*sum(audit)-sum(all))` and the sorted-pair absolute-difference sum.
Their common denominator, group size and positive cohort SD are constant
across assignments, so omitting them gives exactly the same lexicographic
ordering. Complementary assignments therefore remain exact ties, with no
epsilon or tolerance. The sorted audit tuple breaks only an exact tie.
Only after selection convert the six aggregate objective values to JSON
numbers for the balance report; that conversion cannot affect selection.

The streaming implementation applies categorical comparisons first, then
age, then reliability, skipping later computations only when an earlier
component already proves an assignment worse. It still visits and counts
all 2,704,156 tuples. It retains one current optimum and fixed input vectors
in memory, without storing a ranked list or writing participant values.
It never treats a partial search as a split.

## Publication and access boundary

Publish one complete JSON artifact, `participant_roles.json`, containing the
two sorted 12-person ID lists, aggregate categorical counts and six aggregate
balance objectives, method conventions and exhaustive-count completion.
No participant ages, category rows, reliability values, role-specific numeric
means, SDs, minima, maxima, percentiles or ranked participant summaries are
exported. The explicit balance report is required by the original contract;
its aggregate objective values are not participant-level reliability output.

Publish atomically with no overwrite, only after all assignments finish and
the complete 12/12 partition is validated. A failure or deadline before
completion publishes no roles or current-best tuple. Remove an incomplete
temporary publication on handled failure. A separate operational JSON report
contains stage, visited count (last 4096-tuple checkpoint on interruption),
bytes, time, RSS and fixed failure categories;
it never includes exception text or private input values. A deadline just
after atomic publication is recorded as such and never triggers auto-retry.

The trusted process may read the existing private CSV; the coordinator and
candidate workers must not inspect it or raw demographics. This is a logical
steward boundary under the same UID, not an OS seal. The resulting role IDs
authorize no dataset mount. Creating development, audit-calibration and sealed
audit handoffs with actual reader separation remains separate required work.
No Task-B outcomes are opened by this operation. No margin is activated.

## Resources and coordinator execution

One CPU, 2 GiB, no GPU, 540-second internal deadline and 600-second external
limit, within the existing preparation scratch cap of 2 GiB. All work is
in memory apart from bounded text inputs and final small JSON outputs; no
image extracts, extra dataset copies or durable private intermediates.
Enumeration cost has not yet been measured. The first complete bounded
attempt provides its actual cost; a timeout stops without a partial split,
resource expansion or automatic retry.

After adoption, documented lineage review and an attempt entry, root runs
inside its verified allocation (replace the section title with the exact
level-two heading root recorded):

```sh
python3 outputs/tools/assign_participant_roles.py --output-dir outputs/runs/role_assignment/attempt001 --lineage-review-complete --lineage-review-section "EP15 pre-role lineage review" --max-seconds 540
```

Focused source-free tests cover independently calculated exhaustive objectives,
objective priority, W1, exact decimal/complement ties, constant-spread behavior,
explicit missing categories and blocking missing numeric data, prerequisite
guards, completion-only atomic publication and public-output disclosure.
Root owns tests, source/private reads, adoption, execution and shared records.
