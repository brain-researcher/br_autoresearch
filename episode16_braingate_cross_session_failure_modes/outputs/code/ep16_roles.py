"""EP16 role seal: which participants' neural and cursor fields a code path may decode.

This is an instruction and code-path seal, not OS isolation (disclosed in reports).
Development code may decode neural/cursor fields for development participants only.
T9 neural/cursor fields can be decoded only with a SealedEvaluation token issued by
code/evaluate_t9_sealed.py after its freeze-manifest and write-once attempt checks.
T2, T6, T7 and T8 are never readable. Structural fields remain readable through the
existing structural JSONL for every participant.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

DEVELOPMENT = frozenset({"T11"})
SUPPLEMENTARY_DEVELOPMENT = frozenset({"T3", "T5", "T10"})  # component-level checks only
SEALED_EVALUATION = frozenset({"T9"})
UNOPENED = frozenset({"T2", "T6", "T7", "T8"})
KNOWN = DEVELOPMENT | SUPPLEMENTARY_DEVELOPMENT | SEALED_EVALUATION | UNOPENED
PROTECTED_FIELDS = ("neural", "cursor_position")
OPENED_RECORD = "t9_opened.json"
EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "runs" / "t9_evaluation"

_ISSUER = object()  # only evaluate_t9_sealed.open_seal() passes this


class RoleSealError(PermissionError):
    """A code path asked for protected fields it may not decode."""


class SealedEvaluation:
    """Proof that a sealed T9 attempt opened its write-once directory."""

    def __init__(self, issuer, participant, output_dir, nonce):
        if issuer is not _ISSUER:
            raise RoleSealError("SealedEvaluation tokens are issued only by evaluate_t9_sealed.open_seal")
        if participant not in SEALED_EVALUATION:
            raise RoleSealError(f"{participant} is not the sealed evaluation participant")
        self.participant, self.output_dir, self.nonce = participant, Path(output_dir), nonce

    def check(self):
        match = re.fullmatch(r"attempt([0-9]{2,})", self.output_dir.name)
        if (self.output_dir.parent.resolve() != EVALUATION_ROOT.resolve() or self.output_dir.is_symlink()
                or match is None or int(match.group(1)) < 1
                or self.output_dir.name != f"attempt{int(match.group(1)):02d}"):
            raise RoleSealError("Sealed output must be a direct attemptNN child of the anchored evaluation root")
        path = self.output_dir / OPENED_RECORD
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get("nonce") != self.nonce or record.get("participant") != self.participant:
            raise RoleSealError("Sealed evaluation record is missing or does not match this token")


def authorize_protected_read(participant, token=None):
    """Return the access mode, or raise before any protected value is decoded."""
    if participant not in KNOWN:
        raise RoleSealError(f"Unknown participant {participant!r}")
    if participant in UNOPENED:
        raise RoleSealError(f"{participant} stays unopened under the case-series amendment")
    if participant in SEALED_EVALUATION:
        if not isinstance(token, SealedEvaluation) or token.participant != participant:
            raise RoleSealError(f"{participant} neural/cursor fields are readable only by the sealed evaluation")
        token.check()
        return "sealed_evaluation"
    if token is not None:
        raise RoleSealError("A sealed token cannot be used to open development data")
    return "development"
