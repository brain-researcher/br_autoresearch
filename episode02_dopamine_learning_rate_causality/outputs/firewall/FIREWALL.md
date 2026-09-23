# Permission-separated firewall

## Current status

**BLOCKED.** The present workspace does not provide independently attested
separation between candidate, steward, and evaluator principals. Network
denial and an evaluator-only mount are also unprovisioned.

Mode bits, POSIX ACLs controlled by the same uid, a Slurm job under that uid,
or a candidate-launched container are useful accidental-access controls, but
they do not satisfy this contract. Launch therefore remains unauthorized.

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
   operator and one for the independent data steward in
   `FIREWALL_CONTRACT.json`. Both must attest the same runtime identities,
   source commitments, and external evidence for every required check.

Changing signer pins changes the contract and requires renewed review.

## Verification

Verification must be performed by an independent infrastructure operator and
data steward after provisioning. A self-authored local file or a negative
probe made from inside the candidate runtime cannot attest that runtime's
confinement. Verification records belong in the runtime evidence store, not
in the tracked episode specification.
