# Role-separated audit firewall

## Current status

**AUDIT ACCESS BLOCKED PENDING ROLE SEPARATION.** The present workspace has not
yet provisioned distinct candidate, evaluator, and steward roles, candidate
denial from audit payloads, evaluator-only mounts, or network denial during the
audit. This status blocks audit access only. It does not block source
qualification, outcome-blind implementation work, or development work whose
own endpoint and access requirements are satisfied.

`FIREWALL_CONTRACT.json` is a concise statement of the required access
boundary. It is not a schema or receipt template.

## Required controls

Provision these controls before audit access:

1. Assign distinct candidate, evaluator, and steward roles. The candidate must
   not be able to assume an evaluator or steward role.
2. Keep the mixed source and audit packs in storage the candidate cannot read,
   traverse, remount, or change. Give the candidate only development data and
   outcome-blind aggregate structural information.
3. Disable network egress for the candidate and evaluator during the audit so
   the public mixed-role payload cannot be reacquired.
4. Mount the eleven-mouse primary audit pack only for the evaluator and only
   after the versioned write-once final lock is frozen.
5. Return one atomic aggregate packet. Do not return mouse-, arm-, fold-,
   metric-, or QC-level feedback that could update the search.
6. Commit the primary conclusion inside the evaluator transaction before any
   optional boundary-payload access; the boundary diagnostic cannot change or
   rescue the primary terminal.

## Lean verification

The responsible operator and data steward record one short pass/revise note
identifying the people or runtime roles, source release, record-role mapping,
candidate-visible locations, evaluator-only locations, network state, final
lock version, and evaluator output path. Directly test candidate denial and the
aggregate-only evaluator behavior. No checksum bundle, signature, key pin,
custom schema, or receipt chain is required.

Repeat the check only if the access layout, source release, record-role map,
final lock, or evaluator behavior changes, or if a concrete inconsistency is
observed. Same-user mode bits or a candidate-launched container are not enough
when that same candidate can undo them.
