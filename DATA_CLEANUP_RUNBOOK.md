# Cleanup A receipt and quarantine runbook

This runbook covers only the five exact paths in
[`DATA_CLEANUP_CANDIDATES.json`](DATA_CLEANUP_CANDIDATES.json). It does not
authorize a rename or deletion. The current helper can inspect a candidate and
create a protected **pending** receipt; it intentionally has no quarantine,
delete, purge, or arbitrary-path command.

## Scope

| Candidate | Exact cleanup unit | Snapshot reclaim |
| --- | --- | ---: |
| `scratch_things_eeg1_ingest` | Entire completed THINGS ingest staging root | 59,759,665,152 bytes |
| `scratch_ep05_payload` | EP05 `payload/` leaf only | 9,571,819,520 bytes |
| `personal_link_001201_full` | Independent personal LINK tree | 12,564,307,968 bytes |
| `personal_link_001201_sentinels` | Three independent sentinel files | 131,903,488 bytes |
| `personal_epic_ptir_coating` | Independent personal workbook copy | 139,264 bytes |

The snapshot total is 82,027,835,392 allocated bytes. It is an estimate, not a
quota promise. Do not substitute a parent path. In particular, the whole EP05
runtime, every EP04 path, historical NARPS, frozen priors, OpenBHB, EP20
examples, recovery, restricted data, evaluator vaults, and the legacy
worktree are outside Cleanup A.

## Prepare pending receipts

List the fixed allowlist or inspect one candidate without writing anything:

```bash
bin/data-cleanup list
bin/data-cleanup plan scratch_things_eeg1_ingest
```

Create a protected pending receipt under the external manifest root:

```bash
bin/data-cleanup prepare-receipt scratch_things_eeg1_ingest \
  --receipt-id cleanup-a-20260923-things-review
```

The receipt ID must be new; the command never overwrites a prior record.

`prepare-receipt` creates only an owner-private receipt directory and a
mode-`0600` `preflight.json` plus the exact mode-`0600`
`candidate_catalog.json` bytes whose SHA256 is bound by the receipt. On this
setgid OAK parent the directory may be reported as `2700`; its group and other
permission bits remain zero. The helper rejects a mismatch in the catalogued
metadata guard fields or metadata-inventory digest, an existing receipt ID, an
unknown logical ID, stale location-manifest hash, or a catalog that grants
quarantine or deletion authority. It does not create either quarantine root.

The current v2 planning receipts prepared on 2026-09-23 are:

```text
/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/manifests/cleanup-a-20260923-things-v2/preflight.json
/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/manifests/cleanup-a-20260923-ep05-payload-v2/preflight.json
/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/manifests/cleanup-a-20260923-link-full-v2/preflight.json
/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/manifests/cleanup-a-20260923-link-sentinels-v2/preflight.json
/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/manifests/cleanup-a-20260923-epic-v2/preflight.json
```

All five record a matching advisory metadata snapshot, the observed
authoritative-root identity, a self-contained receipt ID and proposed future
destination, and pending gates. The metadata digest covers names and stat/link
metadata, not file contents. These are immutable planning records, not
equivalence receipts, execution-preflight receipts, or quarantine approvals.

The same IDs without `-v2` are preserved superseded v1 planning receipts. They
bind an earlier uncommitted catalog hash and must never be promoted or edited.
A future execution window must create a new execution-preflight receipt rather
than filling in or modifying either planning-receipt version.

## Gates before a future quarantine

For one logical ID at a time:

1. Re-stat the exact source and authoritative counterpart. Bind the source
   device, inode, mode, ownership, link count, mtime, counts, logical bytes,
   allocated bytes, and a stable NUL-safe tree-inventory digest to the receipt.
2. Run the candidate-specific equivalence check from the catalog on Slurm.
   Hash the evidence files themselves and record their paths, hashes, command,
   job ID, exit status, and UTC completion time. A path/size comparison alone
   is not sufficient where the catalog requires SHA256. For THINGS, rerun
   current per-file verification over both 412-file trees and separately
   approve the disposition of all non-payload staging metadata; the historical
   MD5 logs alone are not sufficient.
3. Check `squeue`, relevant `scontrol show job` records, local processes,
   scheduled jobs, and open handles. A login-node `lsof` result is not
   cluster-wide proof. Record the limitation and obtain an explicit
   no-external-writer attestation. Repeat the tree inventory and require two
   stable digests.
4. Smoke-test the authoritative copy through the workflow that will consume
   it. Confirm that no locator still depends on the candidate path.
5. Obtain a quarantine-only authorization binding the receipt SHA256, logical
   ID, exact source path, device, inode, exact derived destination, and a UTC
   expiry. This authorization must explicitly say that permanent deletion is
   not authorized.

Any mismatch, active or uncertain writer, changed inode/device, missing
evidence, writable/untrusted counterpart, unexpected link topology, or failed
smoke test stops the operation.

## Future quarantine operation

The mutating command is deliberately not implemented yet. A later reviewed
implementation must accept only a catalog logical ID and signed/bound receipt;
it must not accept arbitrary source, destination, or quarantine-root flags.

The destination must be derived as:

```text
<fixed same-filesystem quarantine root>/<receipt-id>/<logical-id>
```

Required mechanics:

- create the appropriate fixed quarantine root at mode `0700` only in the
  authorized execution window;
- re-run every guard immediately before mutation;
- reject a source or destination-parent symlink and detect a dangling
  destination symlink with `lstat`;
- require source and destination parent to have the same `st_dev`;
- use no-clobber `renameat2(RENAME_NOREPLACE)` and fail closed on `EXDEV`,
  `EEXIST`, or unavailable syscall support; never fall back to copy-and-delete;
- write and fsync a pre-rename intent receipt, then verify that device and
  inode are unchanged at the destination and write the result receipt;
- create no compatibility symlink at the old path; and
- set `not_before_delete_at` to at least seven days after quarantine.

If a postflight fails and no consumer has written to the quarantined tree,
rollback is the inverse guarded rename to the absent original path, requiring
the recorded destination inode. A source/destination conflict or any new write
stops automatic rollback.

## Permanent deletion

Deletion is a later, separate decision. After the hold and a second smoke test,
request authorization naming the quarantined logical ID, exact path, device,
inode, receipt hash, and allocated bytes. No cleanup helper should expose a
permanent-delete command. Use an independently reviewed, literal-path deletion
procedure only after that authorization. Clean empty parents with `rmdir`, not
recursive deletion.

If space is urgent, the scientist may waive the waiting period explicitly;
equivalence, quiescence, receipt binding, exact-target guards, and separate
deletion authorization remain mandatory.
