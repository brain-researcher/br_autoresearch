# Permission-separated firewall

## Current status and tracked template

**BLOCKED.** The present workspace does not provide independently attested
separation between candidate, steward, and evaluator principals. Network
denial and an evaluator-only mount are also unprovisioned.

The tracked `FIREWALL_CONTRACT.json` is deliberately incomplete: audit-pack,
policy, and configuration-lock commitments and both trusted signer pins are
null or absent. It is a provisioning template, not a receipt. Those values
must be filled only after the corresponding runtime objects are frozen.

Mode bits, POSIX ACLs controlled by the same uid, a Slurm job under that uid,
or a candidate container are useful accidental-access controls, but they do
not satisfy this contract. Candidate scoring and audit access therefore remain
unavailable.

## Provisioning handoff

An infrastructure operator and data steward must perform the following
outside the candidate principal:

1. Provision distinct candidate, evaluator, and steward principals. The
   candidate must not be able to assume either other principal.
2. Put the mixed source and evaluator packs in steward/evaluator-owned storage
   for which the candidate lacks read, directory-traverse, ACL-change, and
   ownership rights. Do not candidate-mount those paths.
3. Give the candidate only the development pack, audit-structural packet,
   public commitments, frozen code, and frozen configuration.
4. Disable network egress for both candidate and evaluator runtimes so the
   public Figshare payload cannot be reacquired during the run.
5. Mount the primary and optional boundary packs only in the evaluator after
   the configuration lock. The evaluator must emit one atomic aggregate
   packet and no partial mouse, arm, fold, metric, or QC feedback.
6. Pin one Ed25519 public-key fingerprint for the independent infrastructure
   operator and a different key and signer identity for the independent data
   steward. Both must attest the same episode, source, audit-pack, policy,
   configuration-lock, runtime identities, and external evidence IDs.
7. Set an observation and expiry time. The portable verifier accepts at most
   the contract's 24-hour validity window and five-minute clock skew.

Changing signer pins changes the contract and requires renewed review.

## Verification

After provisioning, both signers sign the canonical `receipt_payload` in
`EXTERNAL_RECEIPT_TEMPLATE.json`. `verify_permission_firewall.py` checks the
contract and payload bindings, three distinct runtime principals, distinct
signer identities and key fingerprints, validity interval, all required
checks, and both Ed25519 signatures. It reads no outcome file.

A self-authored local file or a negative probe made from inside the candidate
runtime cannot attest that runtime's confinement. Completed contracts,
receipts, assessments, and checksum sidecars belong in the runtime evidence
store, not in the tracked episode specification.
