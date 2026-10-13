"""Selected EP20 passive-contact equations, with no fitted device constants.

See outputs/electrical_model_and_probes.md, sections 2 and 3. All absolute
parameters must be supplied explicitly. This frequency-domain component is
neither a causal CPE realization nor a complete NeuroCam measurement model.
It supplies no TFT, loading, switching, DAQ or additional noise terms.
"""

from dataclasses import dataclass
import math


REFERENCE_PAD_AREA_UM2 = 9450.0
BOLTZMANN_J_PER_K = 1.380649e-23  # Exact SI constant, not a device parameter.


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ContactParameters:
    """Parameters at the published L-pad area, in the stated SI units.

    gamma=1 is E-area; gamma=0.7 is the selected E-sublinear sensitivity.
    Neither choice identifies any of the required absolute parameters.
    """

    r_access_l_ohm: float
    g_ct_l_siemens: float
    q_l_siemens_seconds_alpha: float
    alpha: float
    gamma: float

    def __post_init__(self) -> None:
        for name in (
            "r_access_l_ohm",
            "g_ct_l_siemens",
            "q_l_siemens_seconds_alpha",
            "alpha",
            "gamma",
        ):
            _finite(name, getattr(self, name))
        if self.r_access_l_ohm < 0 or self.g_ct_l_siemens < 0:
            raise ValueError("Access resistance and charge-transfer conductance must be nonnegative")
        if self.q_l_siemens_seconds_alpha <= 0:
            raise ValueError("CPE coefficient must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")
        if self.gamma not in (1.0, 0.7):
            raise ValueError("gamma must be the selected E-area (1) or E-sublinear (0.7) law")


@dataclass(frozen=True)
class ImpedanceParts:
    """Separate access and interface impedances, in ohms.

    Use interface_ohm alone if a coupled solver already owns access resistance.
    Use total_ohm only when this component owns both contributions. The same
    ownership applies when passing an impedance to equilibrium_noise_psd.
    """

    access_ohm: float
    interface_ohm: complex

    @property
    def total_ohm(self) -> complex:
        return self.access_ohm + self.interface_ohm


def contact_impedance_parts(
    *, frequency_hz: float, pad_area_um2: float, parameters: ContactParameters
) -> ImpedanceParts:
    """Evaluate Z_e(j*2*pi*f, A/A_L) without conflating its two contributions.

    Frequencies are nonnegative and are not restricted to the scoring band.
    Finite-charge-transfer DC is supported; blocking-contact DC is singular
    and raises ValueError rather than receiving an epsilon or finite proxy.
    """

    _finite("frequency_hz", frequency_hz)
    _finite("pad_area_um2", pad_area_um2)
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be nonnegative")
    if pad_area_um2 <= 0:
        raise ValueError("pad_area_um2 must be positive")
    if frequency_hz == 0 and parameters.g_ct_l_siemens == 0:
        raise ValueError("Blocking-contact DC impedance is unbounded")

    area_ratio = pad_area_um2 / REFERENCE_PAD_AREA_UM2
    omega = 2 * math.pi * frequency_hz
    if parameters.alpha == 1:
        # Preserve the exactly lossless ideal-capacitor limit; cos(pi/2)
        # would introduce a small fictitious real admittance numerically.
        cpe_admittance = complex(0.0, parameters.q_l_siemens_seconds_alpha * omega)
    else:
        phase = math.pi * parameters.alpha / 2
        cpe_admittance = (
            parameters.q_l_siemens_seconds_alpha
            * omega**parameters.alpha
            * complex(math.cos(phase), math.sin(phase))
        )
    interface_admittance = area_ratio**parameters.gamma * (
        parameters.g_ct_l_siemens + cpe_admittance
    )
    return ImpedanceParts(
        access_ohm=parameters.r_access_l_ohm * area_ratio**-0.5,
        interface_ohm=1 / interface_admittance,
    )


def equilibrium_noise_psd(*, impedance_ohm: complex, temperature_kelvin: float) -> float:
    """One-sided equilibrium voltage PSD 4*k_B*T*Re[Z], in V^2/Hz.

    Supply the passive contribution owned by this model exactly once, using
    the interface or total impedance as appropriate. This does not model the
    separately characterized noise of biased TFTs or readout electronics.
    Zero temperature is the formal classical limit, not a device assumption.
    """

    _finite("temperature_kelvin", temperature_kelvin)
    impedance = complex(impedance_ohm)
    _finite("real impedance", impedance.real)
    _finite("imaginary impedance", impedance.imag)
    if temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be nonnegative")
    if impedance.real < 0:
        raise ValueError("Equilibrium noise requires a passive impedance with nonnegative real part")
    return 4 * BOLTZMANN_J_PER_K * temperature_kelvin * impedance.real
