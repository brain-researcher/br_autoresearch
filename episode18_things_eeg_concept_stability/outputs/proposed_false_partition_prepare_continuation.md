# Selected continuation of EP18 false-partition cache preparation

Selected October12,23:13Pacific under standing engineering/compute authority;
see `experiment_log.md` for the selection, job ID and accounting. The original
proposal below records its rationale. Compute inspection confirms36of36 pair
caches and3of6 context folds complete. Syntax and independent source review
pass. One prepare-only4CPU/28GiB/4h continuation reserves16CPUh. No scientific
amendment, search, EEG fitting or automatic successor is selected.

## Diagnosed stop and retained scope

Job 47584692 ended with exit 75 because its wrapper deliberately gave cache
preparation 10,800 seconds of the four-hour allocation. Its public
`initial_47584692/prepare/report.json` records `resource_incomplete`,
`budget_stop_reason: time_or_signal`, 10,794.505 seconds and peak RSS
4,294,221,824 bytes. The public wrapper exit says `prepare_exit:75` and
`search_exit:-1`. The empty stdout/stderr are expected: this producer writes its
operational status to JSON. This evidence is a preparation time limit, not an
out-of-memory failure, exhausted search or failed scientific comparison.

The original source commits pair-cache progress after each 128-row block and
reuses completed pair files. It restarts only an unfinished context fold.
The public `checkpoint_available:false` field concerns search checkpointing;
it does not say whether preparation progress exists. Actual cache progress
has not been inspected while preparing this proposal. A projection in compute
will determine completed-pair counts, the current pair's committed row count,
completed-context counts and whether the final manifest already exists.

Preserve these original paths:

- Config: `outputs/config/false_partition_sub07_20261012_001.json`.
- Cache: `outputs/cache/false_partition_sub07_20261012_001`.
- Completed input binding: `outputs/runs/false_partition_preparation/inputs_20261012_001`.
- Predecessor public operations: `outputs/runs/false_partition_preparation/false_partition_operations_20261012_001/initial_47584692`.

Do not modify `false_partition_search.py`: its source SHA is already included
in the existing configuration/cache binding. The proposed wrapper compares
the current producer and config with the predecessor and original cache
settings. It neither recreates inputs nor writes a new scientific recipe.
All six feature families, six folds, 22,248 main images, 855 context channels,
the fixed 928-ms horizon and original eight-bank requirements remain intact.
The subsequent search's seed, eight members, 64 starts and 100,000 swaps per
start remain unchanged, but this wrapper never starts that search.

## Proposed finite resource selection

Use **4 CPUs, 28 GiB and at most 4 hours**, with **one numerical worker and one
BLAS/OpenMP thread**, a **24 GiB observed-RSS guard**, and the unchanged **8 GiB
array admission** for preparation. This reserves at most **16 CPU-hours**.
The extra CPUs accompany the memory request; no fourfold numerical speedup is
claimed. The coordinator must verify current site admission/normalization and
that this fits the episode's remaining ledger and concurrent-core limit.

The measured first segment used about 4.00 GiB. Later context construction
maps six complete pair files at once, about 11.1 GiB in total, plus one roughly
145-MiB context array and ordinary overhead. Thus the first-segment measurement
does not justify 1 CPU/7 GiB. Preparation does not load the roughly 71-GiB search
arrays; the previous 96-GiB allocation included that later search. The proposed
28-GiB allocation is a conservative source-based preparation bound, not an
empirical guarantee that every remaining stage fits. The guard stops and
preserves progress if the bound proves inadequate.

The cooperative deadline is the minimum of actual job start plus four hours
minus 60 seconds, actual Slurm end minus 60 seconds, and the original absolute
deadline 1791927000 minus 60 seconds. Native termination and the final bounded
metadata projection use that reserved tail. No original deadline is extended.
The existing 72-GiB cache cap stays fixed; no second cache or bulk copy is made.
Completed files and the original attempt remain in place. The source's existing
temporary pair/context files retain their original continuation semantics.

## Concrete coordinator commands

First, in a verified coordinator-owned compute shell or explicitly allocated
job step, obtain the bounded metadata-only projection:

```bash
bash "$OAK/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/code/run_false_partition_prepare_continue001.sh" inspect
```

This invocation uses the existing qualified Python environment and reads only
bounded JSON metadata plus the small producer source. It prints counts and
operational resource facts; it reads no feature arrays, event tables, EEG,
candidate assignments or scores, and writes no output directory. It belongs
inside compute. Its child has a 25-second timeout plus five-second kill grace;
reserve enough step time for the existing module setup as well.

If the cache manifest is already present, do not launch preparation again.
If metadata/config binding or a partial commit is inconsistent, diagnose it
without deleting or overwriting the existing records. If resumable work remains,
record the explicit selection and reserve the 16 CPU-hours before submission.
Confirm job 47584692 is terminal and no other process owns this cache. Precreate
only this new operations parent:

`outputs/runs/false_partition_preparation/prepare_continuation_operations_20261013_001`.

Submit from the episode directory with a coordinator-discovered partition,
`--nodes=1 --ntasks=1 --cpus-per-task=4 --mem=28G --time=04:00:00 --no-requeue`,
unique `%j` stdout/stderr paths in that operations parent, the new wrapper
as script, and its single argument `prepare`. No array, GPU or automatic
dependency successor is selected. A script-level exclusive claim prevents
another copy of this continuation from writing concurrently; preserve that
claim after failure. Each attempt's actual Slurm state and CPU-time charge
must still be checked and recorded by the coordinator.

The operation publishes `progress_before.json`, a new `prepare/` operational
segment, `progress_after.json`, and `exit.json`. Completion proves only the
image/context cache is ready; it does not qualify the false-bank comparator,
compute an EEG result, open the audit or authorize search scaling. Exit 75
remains a resource stop requiring an explicit next selection. No prior passing
tests are repeated and no new test suite is introduced for this unchanged
producer invocation.
