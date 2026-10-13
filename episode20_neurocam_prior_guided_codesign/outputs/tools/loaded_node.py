"""Selected scalar loading and independent-source PSD equation for EP20.

Implements electrical_model_and_probes.md section 3 at one frequency. This
is a frequency-domain circuit component, not a switched-noise simulation or
the complete reference model. No device, temperature or noise value defaults
are supplied. Unknown contributions must stay unresolved, not be passed as
zero. Correlated sources require a separate cross-spectral network calculation.
"""

from dataclasses import dataclass
import math
from typing import Mapping

from passive_contact import equilibrium_noise_psd


def _impedance(name: str, value: complex) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must have finite real and imaginary parts")
    return result


def _psd(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class LoadedNodeSpectrum:
    """Voltage transfer, parallel impedance and separate one-sided PSD terms.

    Transfer is dimensionless; impedance is in ohms; all returned noise terms
    are node-voltage PSDs in V²/Hz. These scalar terms do not imply independence
    across physical sites, and in particular do not create site-independent
    draws of shared reference noise.
    """

    signal_transfer: complex
    parallel_impedance_ohm: complex
    contact_node_psd_v2_per_hz: float
    voltage_noise_psd_v2_per_hz: float
    current_node_psd_v2_per_hz: float
    other_node_psds_v2_per_hz: tuple[tuple[str, float], ...]
    total_node_psd_v2_per_hz: float


def independent_loaded_node_spectrum(
    *,
    source_impedance_ohm: complex,
    input_impedance_ohm: complex,
    owned_contact_impedance_ohm: complex,
    temperature_kelvin: float,
    voltage_noise_psd_v2_per_hz: float,
    current_noise_psd_a2_per_hz: float,
    other_node_psds_v2_per_hz: Mapping[str, float],
) -> LoadedNodeSpectrum:
    """Propagate explicitly independent noise contributions to a loaded node.

    All impedances/PSDs refer to the same caller-specified frequency and source
    condition. ``source_impedance_ohm`` is the complete source path, including
    its contact exactly once. ``owned_contact_impedance_ohm`` is the passive
    portion whose equilibrium noise is owned here: use interface-only when a
    coupled solver owns access-resistance noise, otherwise the full contact.
    It is not added to the supplied complete source path a second time.

    The voltage PSD is S_e at the node; the current PSD is S_i injected at the
    node and propagated through Z_s || Z_in. Named other contributions must
    already have been propagated from their physical stages to this node and
    must exclude the contact, S_e and S_i terms counted here. An empty mapping
    explicitly represents no such contributions in a caller's stated model;
    it cannot certify their physical absence. This independent-source branch
    cannot replace required cross-spectra when sources are correlated.

    No integration, primary-band prefilter, aliasing approximation, scalar-RMS
    fit, sampling schedule or conversion-work claim is made by this function.
    """
    source = _impedance("source impedance", source_impedance_ohm)
    load = _impedance("input impedance", input_impedance_ohm)
    contact = _impedance("owned contact impedance", owned_contact_impedance_ohm)
    denominator = _impedance("source plus input impedance", source + load)
    if denominator == 0:
        raise ValueError("The loaded path is singular; no epsilon replacement is permitted")

    voltage_psd = _psd("voltage noise PSD", voltage_noise_psd_v2_per_hz)
    current_psd = _psd("current noise PSD", current_noise_psd_a2_per_hz)
    other = []
    for name, value in other_node_psds_v2_per_hz.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every other node-noise contribution needs a nonempty name")
        other.append((name, _psd(f"{name} node PSD", value)))

    transfer = _impedance("signal transfer", load / denominator)
    parallel = _impedance("parallel impedance", source * transfer)
    contact_psd = _psd(
        "loaded contact PSD",
        abs(transfer) ** 2
        * equilibrium_noise_psd(impedance_ohm=contact, temperature_kelvin=temperature_kelvin),
    )
    current_node_psd = _psd("loaded current PSD", abs(parallel) ** 2 * current_psd)
    total = _psd(
        "total node PSD",
        math.fsum([contact_psd, voltage_psd, current_node_psd] + [value for _, value in other]),
    )
    return LoadedNodeSpectrum(
        signal_transfer=transfer,
        parallel_impedance_ohm=parallel,
        contact_node_psd_v2_per_hz=contact_psd,
        voltage_noise_psd_v2_per_hz=voltage_psd,
        current_node_psd_v2_per_hz=current_node_psd,
        other_node_psds_v2_per_hz=tuple(other),
        total_node_psd_v2_per_hz=total,
    )
