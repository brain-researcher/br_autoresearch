"""Fresh quantizer objects sharing only admitted immutable numeric payloads.

The first boundary fully validates and snapshots ordinary caller input. Later
boundaries may share its builtin float tuples, but NEVER share the quantizer
object itself. This retains isolation against caller field reassignment without
revalidating every threshold in each storage window. No global cache, retained
window history, digest cache, or new quantizer arithmetic is introduced.

The private exact payload type is internal provenance, not a security
boundary against arbitrary Python reflection into private implementation state.
Public-field reassignment is detected by identity, including equal-valued new
tuples and signed-zero header changes. Subclasses and non-builtin-float headers
retain ordinary dataclasses.replace behavior; their arithmetic is not coerced.
Provenance-free eligible transfers receive full base validation.
"""

from dataclasses import replace

from acquisition_readout import QuantizerTransfer


class _ValidatedPayload(tuple):
    __slots__ = ()


_FIELDS = ("input_min_volts", "input_max_volts", "thresholds_volts", "code_volts")


def snapshot_quantizer(value):
    """Return a distinct owned snapshot; reuse only proven immutable values.

    At most one constant-size provenance tuple is held per distinct snapshot.
    Its four strong references also prevent identity recycling. The original
    tuple storage remains counted by the existing live-plan numeric accounting.
    No transient threshold-sized allocation occurs on the admitted fast path.
    """
    if not isinstance(value, QuantizerTransfer):
        raise ValueError("Explicit quantizer required")
    payload = (getattr(value, "_validated_quantizer_payload", None)
               if type(value) is QuantizerTransfer else None)
    if type(payload) is _ValidatedPayload:
        if any(getattr(value, name) is not expected for name, expected in zip(_FIELDS, payload)):
            raise ValueError("Owned quantizer fields were reassigned after validation")
    else:
        # Preserve subclass methods and exact endpoint arithmetic (for example,
        # a Fraction can compare differently to a rounded conversion float).
        # The selected recipe has exact base type and builtin float endpoints.
        if (type(value) is not QuantizerTransfer
                or type(value.input_min_volts) is not float
                or type(value.input_max_volts) is not float):
            return replace(value)
        # Validate the actual caller payload before granting provenance. The
        # base constructor owns fresh tuples containing only builtin floats.
        validated = QuantizerTransfer(value.input_min_volts, value.input_max_volts,
                                      value.thresholds_volts, value.code_volts)
        payload = _ValidatedPayload((validated.input_min_volts, validated.input_max_volts,
                                     validated.thresholds_volts, validated.code_volts))
    # Preserve the public dataclass type, equality, field enumeration and
    # dataclasses.replace behavior. Provenance is not a dataclass field.
    owned = object.__new__(QuantizerTransfer)
    for name, item in zip(_FIELDS, payload):
        object.__setattr__(owned, name, item)
    object.__setattr__(owned, "_validated_quantizer_payload", payload)
    return owned
