"""Kennett recursive R/T matrix method for CPA-homogenized layers.

Propagates seismic waves through a stack of isotropic layers using
Kennett's recursive reflection/transmission algorithm. Each layer can
be independently CPA-homogenized from a heterogeneous cube lattice.

Three-level architecture:
  - Inner: single-cube T-matrix (effective_contrasts.py)
  - Middle: CPA self-consistency (cpa_iteration.py)
  - Outer: Kennett layer stacking (this module)

Reference: Kennett (1983), Seismic Wave Propagation in Stratified Media.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .cpa_iteration import (
    CubicEffectiveMedium,
    Phase,
    compute_cpa,
)

# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class FluidLayer:
    """Single acoustic (fluid) layer — no shear wave support.

    Attributes:
        alpha: P-wave velocity (m/s).
        rho: Density (kg/m^3).
        thickness: Layer thickness (m). Use np.inf for half-space.
        Q_alpha: P-wave quality factor. Use np.inf for lossless.
        beta: Always 0.0 (fluid).
        Q_beta: Unused (kept for interface compatibility).
    """

    alpha: float
    rho: float
    thickness: float
    Q_alpha: float = np.inf
    beta: float = 0.0
    Q_beta: float = np.inf

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            msg = f"alpha must be positive, got {self.alpha}"
            raise ValueError(msg)
        if self.rho <= 0:
            msg = f"rho must be positive, got {self.rho}"
            raise ValueError(msg)
        if self.thickness <= 0:
            msg = f"thickness must be positive, got {self.thickness}"
            raise ValueError(msg)
        if self.beta != 0.0:
            msg = f"FluidLayer beta must be 0, got {self.beta}"
            raise ValueError(msg)


@dataclass
class IsotropicLayer:
    """Single isotropic elastic layer.

    Attributes:
        alpha: P-wave velocity (m/s).
        beta: S-wave velocity (m/s). Must be > 0 (solid).
        rho: Density (kg/m^3).
        thickness: Layer thickness (m). Use np.inf for half-space.
        Q_alpha: P-wave quality factor. Use np.inf for lossless.
        Q_beta: S-wave quality factor. Use np.inf for lossless.
    """

    alpha: float
    beta: float
    rho: float
    thickness: float
    Q_alpha: float = np.inf
    Q_beta: float = np.inf

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            msg = f"alpha must be positive, got {self.alpha}"
            raise ValueError(msg)
        if self.beta <= 0:
            msg = f"beta must be positive, got {self.beta}"
            raise ValueError(msg)
        if self.rho <= 0:
            msg = f"rho must be positive, got {self.rho}"
            raise ValueError(msg)
        if self.thickness <= 0:
            msg = f"thickness must be positive, got {self.thickness}"
            raise ValueError(msg)
        if not np.isinf(self.Q_alpha) and self.Q_alpha <= 0:
            msg = f"Q_alpha must be positive or inf, got {self.Q_alpha}"
            raise ValueError(msg)
        if not np.isinf(self.Q_beta) and self.Q_beta <= 0:
            msg = f"Q_beta must be positive or inf, got {self.Q_beta}"
            raise ValueError(msg)


@dataclass
class LayerStack:
    """Ordered stack of layers (top-to-bottom).

    The last layer is the half-space (thickness = inf).
    Layers can be FluidLayer (acoustic) or IsotropicLayer (elastic).

    Attributes:
        layers: List of FluidLayer | IsotropicLayer objects (top to bottom).
    """

    layers: Sequence[FluidLayer | IsotropicLayer] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.layers) < 2:
            msg = f"Need >= 2 layers, got {len(self.layers)}"
            raise ValueError(msg)
        if not np.isinf(self.layers[-1].thickness):
            msg = (
                f"Last layer must be half-space (thickness=inf), "
                f"got {self.layers[-1].thickness}"
            )
            raise ValueError(msg)

    @property
    def n_layers(self) -> int:
        """Number of layers in the stack."""
        return len(self.layers)

    @classmethod
    def from_arrays(
        cls,
        alpha: np.ndarray,
        beta: np.ndarray,
        rho: np.ndarray,
        thickness: np.ndarray,
        Q_alpha: np.ndarray | None = None,
        Q_beta: np.ndarray | None = None,
    ) -> "LayerStack":
        """Create stack from arrays of layer properties."""
        n = len(alpha)
        if Q_alpha is None:
            Q_alpha = np.full(n, np.inf)
        if Q_beta is None:
            Q_beta = np.full(n, np.inf)
        layers = [
            IsotropicLayer(
                alpha=float(alpha[i]),
                beta=float(beta[i]),
                rho=float(rho[i]),
                thickness=float(thickness[i]),
                Q_alpha=float(Q_alpha[i]),
                Q_beta=float(Q_beta[i]),
            )
            for i in range(n)
        ]
        return cls(layers=layers)

    @classmethod
    def homogeneous(
        cls,
        alpha: float,
        beta: float,
        rho: float,
        n_layers: int = 3,
        thickness: float = 100.0,
    ) -> "LayerStack":
        """Create a homogeneous stack (all layers identical) for testing."""
        layers = [
            IsotropicLayer(alpha=alpha, beta=beta, rho=rho, thickness=thickness)
            for _ in range(n_layers - 1)
        ]
        layers.append(IsotropicLayer(alpha=alpha, beta=beta, rho=rho, thickness=np.inf))
        return cls(layers=layers)


@dataclass
class PSVCoefficients:
    """P-SV interfacial scattering coefficients (2x2 matrices).

    Index 0 = P-wave, index 1 = S-wave.
    Modified form with sqrt(eta*rho) normalization for unitary recursion.
    """

    Rd: np.ndarray  # (2,2) downgoing reflection
    Ru: np.ndarray  # (2,2) upgoing reflection
    Td: np.ndarray  # (2,2) downgoing transmission
    Tu: np.ndarray  # (2,2) upgoing transmission


@dataclass
class SHCoefficients:
    """SH interfacial scattering coefficients (scalars).

    Modified form with sqrt(impedance) normalization.
    """

    Rd: complex  # downgoing reflection
    Ru: complex  # upgoing reflection
    Td: complex  # downgoing transmission
    Tu: complex  # upgoing transmission


@dataclass
class KennettResult:
    """Result of Kennett reflectivity computation.

    Attributes:
        RD_psv: Cumulative P-SV reflection matrix, shape (nfreq, 2, 2).
        RD_sh: Cumulative SH reflection coefficient, shape (nfreq,).
        omega: Angular frequencies used, shape (nfreq,).
        p: Horizontal slowness (ray parameter).
        stack: The layer stack used.
    """

    RD_psv: np.ndarray
    RD_sh: np.ndarray
    omega: np.ndarray
    p: float
    stack: LayerStack

    @property
    def RPP(self) -> np.ndarray:
        """PP reflection coefficient, shape (nfreq,)."""
        return self.RD_psv[:, 0, 0]

    @property
    def RSS(self) -> np.ndarray:
        """SS reflection coefficient, shape (nfreq,)."""
        return self.RD_psv[:, 1, 1]

    @property
    def RPS(self) -> np.ndarray:
        """PS reflection coefficient, shape (nfreq,)."""
        return self.RD_psv[:, 0, 1]

    @property
    def RSP(self) -> np.ndarray:
        """SP reflection coefficient, shape (nfreq,)."""
        return self.RD_psv[:, 1, 0]

    @property
    def RSH(self) -> np.ndarray:
        """SH reflection coefficient, shape (nfreq,)."""
        return self.RD_sh


# ── Vertical slowness ───────────────────────────────────────────────────


def _complex_slowness(velocity: float, Q: float) -> complex:
    """Complex slowness accounting for attenuation.

    Args:
        velocity: Phase velocity (m/s).
        Q: Quality factor. Use np.inf for lossless.

    Returns:
        Complex slowness s = 1/v for Q=inf, otherwise attenuative.
    """
    if np.isinf(Q):
        return complex(1.0 / velocity)
    twoQ = 2.0 * Q
    twoQsq = twoQ * twoQ
    denom = (1.0 + twoQsq) * velocity
    return complex(twoQsq / denom, twoQ / denom)


def _vertical_slowness(slowness: complex, p: float) -> complex:
    """Vertical slowness with Im(eta) > 0 branch cut.

    eta = sqrt(s^2 - p^2), choosing the branch with Im(eta) > 0
    for evanescent waves, or Re(eta) > 0 for propagating waves.
    """
    T = (slowness + p) * (slowness - p)
    eta = np.sqrt(complex(T))
    # Branch cut: Im(eta) > 0 for evanescent; Re(eta) > 0 for propagating
    if eta.imag < 0.0 or (eta.imag == 0.0 and eta.real < 0.0):
        eta = -eta
    return eta


# ── Interfacial coefficients ────────────────────────────────────────────


def psv_solid_solid(
    p: float,
    eta1: complex,
    neta1: complex,
    rho1: float,
    beta1: complex,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
) -> PSVCoefficients:
    """Modified P-SV scattering coefficients at a solid-solid interface.

    Reimplements the Aki & Richards (1980) formulation with sqrt(eta*rho)
    normalization for unitary Kennett recursion. Tu = Td.T (reciprocity).

    Args:
        p: Horizontal slowness (ray parameter).
        eta1, eta2: Vertical P-wave slowness in layers 1, 2.
        neta1, neta2: Vertical S-wave slowness in layers 1, 2.
        rho1, rho2: Densities.
        beta1, beta2: Complex S-wave velocity (1/complex_s_slowness).

    Returns:
        PSVCoefficients with modified 2x2 R/T matrices.
    """
    rtrho1 = np.sqrt(rho1)
    rtrho2 = np.sqrt(rho2)
    rteta1 = np.sqrt(eta1)
    rteta2 = np.sqrt(eta2)
    rtneta1 = np.sqrt(neta1)
    rtneta2 = np.sqrt(neta2)
    rtza1 = rteta1 * rtrho1
    rtza2 = rteta2 * rtrho2
    rtzb1 = rtneta1 * rtrho1
    rtzb2 = rtneta2 * rtrho2

    psq = complex(p * p)
    crho1 = complex(rho1)
    crho2 = complex(rho2)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2 - crho1 * beta1 * beta1

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    E = b * eta1 + c * eta2
    F = b * neta1 + c * neta2
    G = a - d * eta1 * neta2
    H = a - d * eta2 * neta1

    Det = E * F + G * H * psq

    E = E / Det
    F = F / Det
    G = G / Det
    H = H / Det

    Q_val = (b * eta1 - c * eta2) * F
    R_val = (a + d * eta1 * neta2) * H * psq
    S_val = (a * b + c * d * eta2 * neta2) * p / Det
    T_val = (b * neta1 - c * neta2) * E
    U_val = (a + d * eta2 * neta1) * G * psq
    V_val = (a * c + b * d * eta1 * neta1) * p / Det

    m2ci = complex(0.0, -2.0)

    Rd = np.array(
        [
            [Q_val - R_val, m2ci * rteta1 * rtneta1 * S_val],
            [m2ci * rteta1 * rtneta1 * S_val, T_val - U_val],
        ],
        dtype=np.complex128,
    )

    Td = np.array(
        [
            [2 * rtza1 * rtza2 * F, m2ci * rtzb1 * rtza2 * G * p],
            [m2ci * rtza1 * rtzb2 * H * p, 2 * rtzb1 * rtzb2 * E],
        ],
        dtype=np.complex128,
    )

    Tu = Td.T.copy()

    Ru = np.array(
        [
            [-(Q_val + U_val), m2ci * rteta2 * rtneta2 * V_val],
            [m2ci * rteta2 * rtneta2 * V_val, -(T_val + R_val)],
        ],
        dtype=np.complex128,
    )

    return PSVCoefficients(Rd=Rd, Ru=Ru, Td=Td, Tu=Tu)


def psv_fluid_solid(
    p: float,
    eta1: complex,
    rho1: float,
    eta2: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
) -> PSVCoefficients:
    """Modified P-SV scattering coefficients at a fluid-solid interface.

    Translated from PhD OBSMat (kennetslo.f). Layer 1 is acoustic (beta=0,
    neta1=0), layer 2 is elastic. Uses sqrt(eta*rho) normalization.

    Args:
        p: Horizontal slowness (ray parameter).
        eta1: Vertical P-wave slowness in fluid.
        rho1: Fluid density.
        eta2: Vertical P-wave slowness in elastic layer.
        neta2: Vertical S-wave slowness in elastic layer.
        rho2: Elastic layer density.
        beta2: Complex S-wave velocity in elastic layer.

    Returns:
        PSVCoefficients with acoustic-elastic structure.
        Rd has only [0,0] nonzero; Td column 1 zero; Tu row 1 zero.
    """
    rtrho1 = np.sqrt(rho1)
    rtrho2 = np.sqrt(rho2)
    rteta1 = np.sqrt(eta1)
    rteta2 = np.sqrt(eta2)
    rtneta2 = np.sqrt(neta2)
    rtza1 = rtrho1 * rteta1
    rtza2 = rtrho2 * rteta2
    rtzb2 = rtrho2 * rtneta2

    psq = complex(p * p)
    crho1 = complex(rho1)
    crho2 = complex(rho2)
    drho = crho2 - crho1
    dmu = crho2 * beta2 * beta2  # no beta1 term (acoustic)

    d = 2.0 * dmu
    psqd = psq * d
    a = drho - psqd
    b = crho2 - psqd
    c = crho1 + psqd

    # Acoustic-elastic forms: neta1=0 simplifies F and H
    E = b * eta1 + c * eta2
    F = b
    G = a - d * eta1 * neta2
    H = -d * eta2

    Det = E * F + G * H * psq

    E = E / Det
    F = F / Det
    G = G / Det
    H = H / Det

    T1 = (b * eta1 - c * eta2) * F
    T2 = (a + d * eta1 * neta2) * H * psq
    T4 = b * E
    T5 = d * eta2 * G * psq
    T6 = b * d * eta1 * p / Det  # uses original Det (before normalization)

    mci = complex(0.0, -1.0)

    PdPu = T1 - T2
    PdPd = 2.0 * rtza1 * rtza2 * F
    PdSd = 2.0 * mci * rtza1 * rtzb2 * H * p
    PuPu = PdPd
    SuPu = PdSd
    PuPd = -(T1 + T5)
    PuSd = 2.0 * mci * rteta2 * rtneta2 * T6
    SuPd = PuSd
    SuSd = -(T2 + T4)

    z = 0.0 + 0j
    Rd = np.array([[PdPu, z], [z, z]], dtype=np.complex128)
    Td = np.array([[PdPd, z], [PdSd, z]], dtype=np.complex128)
    Tu = np.array([[PuPu, SuPu], [z, z]], dtype=np.complex128)
    Ru = np.array([[PuPd, SuPd], [PuSd, SuSd]], dtype=np.complex128)

    return PSVCoefficients(Rd=Rd, Ru=Ru, Td=Td, Tu=Tu)


def sh_solid_solid(
    neta1: complex,
    rho1: float,
    beta1: complex,
    neta2: complex,
    rho2: float,
    beta2: complex,
) -> SHCoefficients:
    """Modified SH scattering coefficients at a solid-solid interface.

    Uses SH impedance Z = rho * beta^2 * neta with sqrt(Z) normalization
    for unitary recursion. Tu = Td (reciprocity), Ru = -Rd.

    Args:
        neta1, neta2: Vertical S-wave slowness in layers 1, 2.
        rho1, rho2: Densities.
        beta1, beta2: Complex S-wave velocity.

    Returns:
        SHCoefficients with modified scalar R/T.
    """
    Z1 = complex(rho1) * beta1 * beta1 * neta1
    Z2 = complex(rho2) * beta2 * beta2 * neta2
    Zsum = Z1 + Z2
    Rd = (Z1 - Z2) / Zsum
    Td = 2.0 * np.sqrt(Z1 * Z2) / Zsum
    return SHCoefficients(Rd=Rd, Ru=-Rd, Td=Td, Tu=Td)


# ── Batch 2x2 utilities ────────────────────────────────────────────────


def _batch_inv2x2(M: np.ndarray) -> np.ndarray:
    """Analytical inverse of batch of 2x2 matrices.

    Args:
        M: Array of shape (..., 2, 2).

    Returns:
        Inverse, shape (..., 2, 2).
    """
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    det = a * d - b * c
    inv = np.empty_like(M)
    inv[..., 0, 0] = d / det
    inv[..., 0, 1] = -b / det
    inv[..., 1, 0] = -c / det
    inv[..., 1, 1] = a / det
    return inv


def _batch_matmul2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batch 2x2 matrix multiply with broadcasting.

    Supports (2,2)@(n,2,2), (n,2,2)@(2,2), (n,2,2)@(n,2,2).
    """
    if A.ndim == 2 and B.ndim == 3:
        return np.einsum("ij,wjk->wik", A, B)
    if A.ndim == 3 and B.ndim == 2:
        return np.einsum("wij,jk->wik", A, B)
    if A.ndim == 3 and B.ndim == 3:
        return np.einsum("wij,wjk->wik", A, B)
    msg = f"Unsupported shapes: {A.shape}, {B.shape}"
    raise ValueError(msg)


# ── Kennett recursion ──────────────────────────────────────────────────


def _kennett_psv_recursion(
    stack: LayerStack, p: float, omega: np.ndarray
) -> np.ndarray:
    """Kennett upward sweep for P-SV waves.

    Args:
        stack: Layer stack (top-to-bottom, last = half-space).
        p: Horizontal slowness.
        omega: Angular frequencies, shape (nfreq,).

    Returns:
        Cumulative downgoing reflection RD, shape (nfreq, 2, 2).
    """
    nfreq = len(omega)
    nlayer = stack.n_layers
    layers = stack.layers

    # Complex slownesses and vertical slownesses for all layers
    s_p = np.array([_complex_slowness(lay.alpha, lay.Q_alpha) for lay in layers])
    s_s = np.array(
        [
            _complex_slowness(lay.beta, lay.Q_beta) if lay.beta > 0 else complex(0.0)
            for lay in layers
        ]
    )
    beta_c = np.array([1.0 / s if s != 0.0 else complex(0.0) for s in s_s])
    eta = np.array([_vertical_slowness(s_p[i], p) for i in range(nlayer)])
    neta = np.array(
        [
            _vertical_slowness(s_s[i], p) if s_s[i] != 0.0 else complex(0.0)
            for i in range(nlayer)
        ]
    )

    # Phase factors: ea[i] = exp(i*omega*eta[i]*h[i]), shape (nfreq,)
    ea = np.ones((nlayer, nfreq), dtype=np.complex128)
    eb = np.ones((nlayer, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(layers[i].thickness):
            ea[i] = np.exp(1j * omega * eta[i] * layers[i].thickness)
            if layers[i].beta > 0:
                eb[i] = np.exp(1j * omega * neta[i] * layers[i].thickness)

    # Precompute interfacial coefficients (frequency-independent)
    scat_psv: list[PSVCoefficients] = []
    for il in range(nlayer - 1):
        lay_above = layers[il]
        lay_below = layers[il + 1]
        is_fluid_above = isinstance(lay_above, FluidLayer)
        is_fluid_below = isinstance(lay_below, FluidLayer)

        if is_fluid_above and not is_fluid_below:
            scat_psv.append(
                psv_fluid_solid(
                    p=p,
                    eta1=eta[il],
                    rho1=lay_above.rho,
                    eta2=eta[il + 1],
                    neta2=neta[il + 1],
                    rho2=lay_below.rho,
                    beta2=beta_c[il + 1],
                )
            )
        elif not is_fluid_above and not is_fluid_below:
            scat_psv.append(
                psv_solid_solid(
                    p=p,
                    eta1=eta[il],
                    neta1=neta[il],
                    rho1=lay_above.rho,
                    beta1=beta_c[il],
                    eta2=eta[il + 1],
                    neta2=neta[il + 1],
                    rho2=lay_below.rho,
                    beta2=beta_c[il + 1],
                )
            )
        else:
            msg = (
                f"Interface {il}: fluid-fluid or solid-fluid not supported. "
                f"Got {type(lay_above).__name__} over {type(lay_below).__name__}."
            )
            raise ValueError(msg)

    # Kennett upward sweep: start from half-space (RRd = 0)
    RRd = np.zeros((nfreq, 2, 2), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # Sweep from bottom interface upward
    for iface in range(nlayer - 2, -1, -1):
        i_below = iface + 1  # layer below this interface
        coeff = scat_psv[iface]

        # Phase: two-way travel through layer below
        eaea = ea[i_below] ** 2
        ebeb = eb[i_below] ** 2
        eaeb = ea[i_below] * eb[i_below]

        # MT = E * RRd * E (diagonal phase scaling)
        MT = np.empty((nfreq, 2, 2), dtype=np.complex128)
        MT[:, 0, 0] = eaea * RRd[:, 0, 0]
        MT[:, 0, 1] = eaeb * RRd[:, 0, 1]
        MT[:, 1, 0] = eaeb * RRd[:, 1, 0]
        MT[:, 1, 1] = ebeb * RRd[:, 1, 1]

        # U = (I - Ru @ MT)^{-1}
        RuMT = _batch_matmul2x2(coeff.Ru, MT)
        I_minus_RuMT = I2[np.newaxis, :, :] - RuMT
        U = _batch_inv2x2(I_minus_RuMT)

        # RRd = Rd + Tu @ MT @ U @ Td
        MTU = _batch_matmul2x2(MT, U)
        TuMTU = _batch_matmul2x2(coeff.Tu, MTU)
        TuMTUTd = _batch_matmul2x2(TuMTU, coeff.Td)
        RRd = coeff.Rd[np.newaxis, :, :] + TuMTUTd

    return RRd


def _kennett_sh_recursion(stack: LayerStack, p: float, omega: np.ndarray) -> np.ndarray:
    """Kennett upward sweep for SH waves.

    Args:
        stack: Layer stack (top-to-bottom, last = half-space).
        p: Horizontal slowness.
        omega: Angular frequencies, shape (nfreq,).

    Returns:
        Cumulative downgoing SH reflection, shape (nfreq,).
    """
    nfreq = len(omega)
    nlayer = stack.n_layers
    layers = stack.layers

    s_s = np.array(
        [
            _complex_slowness(lay.beta, lay.Q_beta) if lay.beta > 0 else complex(0.0)
            for lay in layers
        ]
    )
    beta_c = np.array([1.0 / s if s != 0.0 else complex(0.0) for s in s_s])
    neta = np.array(
        [
            _vertical_slowness(s_s[i], p) if s_s[i] != 0.0 else complex(0.0)
            for i in range(nlayer)
        ]
    )

    # Phase factors
    eb = np.ones((nlayer, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(layers[i].thickness) and layers[i].beta > 0:
            eb[i] = np.exp(1j * omega * neta[i] * layers[i].thickness)

    # Precompute SH interfacial coefficients (skip fluid interfaces)
    scat_sh: list[SHCoefficients | None] = []
    for il in range(nlayer - 1):
        if isinstance(layers[il], FluidLayer) or isinstance(layers[il + 1], FluidLayer):
            scat_sh.append(None)
        else:
            scat_sh.append(
                sh_solid_solid(
                    neta1=neta[il],
                    rho1=layers[il].rho,
                    beta1=beta_c[il],
                    neta2=neta[il + 1],
                    rho2=layers[il + 1].rho,
                    beta2=beta_c[il + 1],
                )
            )

    # Scalar Kennett upward sweep
    RRd = np.zeros(nfreq, dtype=np.complex128)

    for iface in range(nlayer - 2, -1, -1):
        i_below = iface + 1
        coeff = scat_sh[iface]

        if coeff is None:
            # Fluid layer: SH doesn't propagate, RRd passes unchanged
            continue

        ebeb = eb[i_below] ** 2
        MT = ebeb * RRd
        U = 1.0 / (1.0 - coeff.Ru * MT)
        RRd = coeff.Rd + coeff.Tu * MT * U * coeff.Td

    return RRd


def kennett_layers(stack: LayerStack, p: float, omega: np.ndarray) -> KennettResult:
    """Compute Kennett reflectivity for a layer stack.

    Main entry point: runs both P-SV (2x2) and SH (scalar) recursions.

    Args:
        stack: Layer stack (top-to-bottom, last = half-space).
        p: Horizontal slowness (ray parameter), s/m.
        omega: Angular frequencies, shape (nfreq,). Must not include omega=0.

    Returns:
        KennettResult with cumulative reflection matrices.
    """
    omega = np.asarray(omega)
    if np.isrealobj(omega):
        omega = omega.astype(np.float64)
    RD_psv = _kennett_psv_recursion(stack, p, omega)
    RD_sh = _kennett_sh_recursion(stack, p, omega)
    return KennettResult(RD_psv=RD_psv, RD_sh=RD_sh, omega=omega, p=p, stack=stack)


# ── Batched Kennett (slowness × frequency) ────────────────────────────


def kennett_reflectivity_batch(
    stack: LayerStack,
    p_samples: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Batched Kennett P-SV reflectivity over slowness and frequency.

    Vectorises the Kennett recursion over all (p, omega) pairs using 4D
    arrays of shape (np_slow, nfreq, 2, 2). Returns only the PP component.

    Args:
        stack: Layer stack (top-to-bottom, last = half-space).
        p_samples: Horizontal slowness samples, shape (np_slow,).
        omega: Complex angular frequencies, shape (nfreq,).

    Returns:
        RRd_PP: PP reflection coefficient, shape (np_slow, nfreq).
    """
    p_samples = np.asarray(p_samples, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.complex128)
    np_slow = len(p_samples)
    nfreq = len(omega)
    nlayer = stack.n_layers
    layers = stack.layers

    # Complex slownesses per layer
    s_p = np.array([_complex_slowness(lay.alpha, lay.Q_alpha) for lay in layers])
    s_s = np.array(
        [
            _complex_slowness(lay.beta, lay.Q_beta) if lay.beta > 0 else complex(0.0)
            for lay in layers
        ]
    )
    beta_c = np.array([1.0 / s if s != 0.0 else complex(0.0) for s in s_s])

    # Vertical slownesses: (nlayer, np_slow)
    eta = np.zeros((nlayer, np_slow), dtype=np.complex128)
    neta = np.zeros((nlayer, np_slow), dtype=np.complex128)
    for i in range(nlayer):
        for jp in range(np_slow):
            eta[i, jp] = _vertical_slowness(s_p[i], p_samples[jp])
            if s_s[i] != 0.0:
                neta[i, jp] = _vertical_slowness(s_s[i], p_samples[jp])

    # Classify interfaces
    is_fluid_solid = np.array(
        [
            isinstance(layers[il], FluidLayer)
            and not isinstance(layers[il + 1], FluidLayer)
            for il in range(nlayer - 1)
        ]
    )

    # Precompute scattering matrices: (n_interfaces, np_slow, 2, 2)
    n_interfaces = nlayer - 1
    Rd_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Ru_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Tu_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)
    Td_all = np.zeros((n_interfaces, np_slow, 2, 2), dtype=np.complex128)

    for il in range(n_interfaces):
        for jp in range(np_slow):
            p_val = float(p_samples[jp])
            if is_fluid_solid[il]:
                coeff = psv_fluid_solid(
                    p=p_val,
                    eta1=eta[il, jp],
                    rho1=layers[il].rho,
                    eta2=eta[il + 1, jp],
                    neta2=neta[il + 1, jp],
                    rho2=layers[il + 1].rho,
                    beta2=beta_c[il + 1],
                )
            else:
                coeff = psv_solid_solid(
                    p=p_val,
                    eta1=eta[il, jp],
                    neta1=neta[il, jp],
                    rho1=layers[il].rho,
                    beta1=beta_c[il],
                    eta2=eta[il + 1, jp],
                    neta2=neta[il + 1, jp],
                    rho2=layers[il + 1].rho,
                    beta2=beta_c[il + 1],
                )
            Rd_all[il, jp] = coeff.Rd
            Ru_all[il, jp] = coeff.Ru
            Tu_all[il, jp] = coeff.Tu
            Td_all[il, jp] = coeff.Td

    # Phase factors: (nlayer, np_slow, nfreq)
    ea = np.ones((nlayer, np_slow, nfreq), dtype=np.complex128)
    eb = np.ones((nlayer, np_slow, nfreq), dtype=np.complex128)
    for i in range(nlayer):
        if not np.isinf(layers[i].thickness):
            phase_p = (
                1j * eta[i, :, np.newaxis] * omega[np.newaxis, :] * layers[i].thickness
            )
            ea[i] = np.exp(phase_p)
            if layers[i].beta > 0:
                phase_s = (
                    1j
                    * neta[i, :, np.newaxis]
                    * omega[np.newaxis, :]
                    * layers[i].thickness
                )
                eb[i] = np.exp(phase_s)

    # Batched Kennett upward sweep: (np_slow, nfreq, 2, 2)
    RRd = np.zeros((np_slow, nfreq, 2, 2), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    for iface in range(n_interfaces - 1, -1, -1):
        i_below = iface + 1

        Rd_if = Rd_all[iface]  # (np_slow, 2, 2)
        Ru_if = Ru_all[iface]
        Tu_if = Tu_all[iface]
        Td_if = Td_all[iface]

        ea_below = ea[i_below]  # (np_slow, nfreq)
        eb_below = eb[i_below]

        eaea = ea_below**2
        ebeb = eb_below**2
        eaeb = ea_below * eb_below

        # Phase-shifted cumulative reflection: MT = E · RRd · E
        MT = np.empty_like(RRd)
        MT[:, :, 0, 0] = eaea * RRd[:, :, 0, 0]
        MT[:, :, 0, 1] = eaeb * RRd[:, :, 0, 1]
        MT[:, :, 1, 0] = eaeb * RRd[:, :, 1, 0]
        MT[:, :, 1, 1] = ebeb * RRd[:, :, 1, 1]

        # U = (I - Ru · MT)^{-1}
        RuMT = np.matmul(Ru_if[:, np.newaxis, :, :], MT)
        I_minus_RuMT = I2 - RuMT
        U = _batch_inv2x2(I_minus_RuMT)

        # RRd = Rd + Tu · MT · U · Td
        MTU = np.matmul(MT, U)
        TuMTU = np.matmul(Tu_if[:, np.newaxis, :, :], MTU)
        TuMTUTd = np.matmul(TuMTU, Td_if[:, np.newaxis, :, :])
        RRd = Rd_if[:, np.newaxis, :, :] + TuMTUTd

    return RRd[:, :, 0, 0]


# ── CPA bridge ─────────────────────────────────────────────────────────


def cubic_to_isotropic_layer(
    eff: CubicEffectiveMedium,
    thickness: float,
    Q_alpha: float = np.inf,
    Q_beta: float = np.inf,
) -> IsotropicLayer:
    """Convert a CPA effective medium to an isotropic layer.

    Uses mu_off for the isotropic shear modulus, consistent with
    CubicEffectiveMedium.as_reference_medium().

    Args:
        eff: CPA-converged effective medium.
        thickness: Layer thickness (m).
        Q_alpha: P-wave quality factor.
        Q_beta: S-wave quality factor.

    Returns:
        IsotropicLayer with velocities from the effective moduli.
    """
    mu = eff.mu_off
    rho = eff.rho
    lam = eff.lam
    alpha = float(np.sqrt((lam + 2.0 * mu) / rho))
    beta = float(np.sqrt(mu / rho))
    return IsotropicLayer(
        alpha=alpha,
        beta=beta,
        rho=float(rho),
        thickness=thickness,
        Q_alpha=Q_alpha,
        Q_beta=Q_beta,
    )


def cpa_stack_from_phases(
    layer_phases: list[list[Phase]],
    omega: float,
    a: float,
    thickness: float | None = None,
    Q_alpha: float = np.inf,
    Q_beta: float = np.inf,
    half_space_index: int = -1,
    **cpa_kwargs,
) -> LayerStack:
    """Build a layer stack by CPA-homogenizing each layer independently.

    Args:
        layer_phases: Per-layer list of Phase objects. Each sublist defines
            the phases in one layer (volume fractions must sum to 1).
        omega: Angular frequency for CPA (rad/s).
        a: Cube half-width (m).
        thickness: Layer thickness (m). Default: 2*a (one cube).
        Q_alpha: P-wave quality factor for all layers.
        Q_beta: S-wave quality factor for all layers.
        half_space_index: Which layer to use as half-space (-1 = last).
        **cpa_kwargs: Passed to compute_cpa (max_iter, tol, damping).

    Returns:
        LayerStack with CPA-homogenized isotropic layers.
    """
    if thickness is None:
        thickness = 2.0 * a
    n = len(layer_phases)
    layers: list[IsotropicLayer] = []
    for i, phases in enumerate(layer_phases):
        result = compute_cpa(phases, omega, a, **cpa_kwargs)
        is_halfspace = (
            (i == n - 1) if half_space_index == -1 else (i == half_space_index)
        )
        h = np.inf if is_halfspace else thickness
        layers.append(
            cubic_to_isotropic_layer(result.effective_medium, h, Q_alpha, Q_beta)
        )
    return LayerStack(layers=layers)


# ── Random stack generation ────────────────────────────────────────────


def random_heterogeneous_stack(
    ref_alpha: float,
    ref_beta: float,
    ref_rho: float,
    n_layers: int,
    a: float,
    omega: float,
    Dlambda: float,
    Dmu: float,
    Drho: float,
    phi_mean: float = 0.5,
    phi_std: float = 0.1,
    seed: int | None = None,
    **cpa_kwargs,
) -> LayerStack:
    """Generate a random heterogeneous stack with CPA homogenization.

    Each layer is a two-phase composite with random volume fraction
    drawn from N(phi_mean, phi_std^2), clipped to [0.01, 0.99].

    Args:
        ref_alpha, ref_beta, ref_rho: Background medium properties.
        n_layers: Number of layers (including half-space).
        a: Cube half-width (m).
        omega: Angular frequency for CPA (rad/s).
        Dlambda, Dmu, Drho: Inclusion contrasts.
        phi_mean: Mean volume fraction.
        phi_std: Standard deviation of volume fraction.
        seed: Random seed for reproducibility.
        **cpa_kwargs: Passed to compute_cpa.

    Returns:
        LayerStack with CPA-homogenized layers.
    """
    rng = np.random.default_rng(seed)
    ref_lam = ref_rho * (ref_alpha**2 - 2 * ref_beta**2)
    ref_mu = ref_rho * ref_beta**2
    thickness = 2.0 * a

    layer_phases: list[list[Phase]] = []
    for _ in range(n_layers):
        phi = float(np.clip(rng.normal(phi_mean, phi_std), 0.01, 0.99))
        matrix = Phase(lam=ref_lam, mu=ref_mu, rho=ref_rho, volume_fraction=1.0 - phi)
        inclusion = Phase(
            lam=ref_lam + Dlambda,
            mu=ref_mu + Dmu,
            rho=ref_rho + Drho,
            volume_fraction=phi,
        )
        layer_phases.append([matrix, inclusion])

    return cpa_stack_from_phases(
        layer_phases, omega, a, thickness=thickness, **cpa_kwargs
    )


def random_velocity_stack(
    alpha_mean: float,
    beta_mean: float,
    rho_mean: float,
    n_layers: int,
    thickness: float,
    dalpha_std: float = 0.0,
    dbeta_std: float = 0.0,
    drho_std: float = 0.0,
    seed: int | None = None,
    Q_alpha: float = np.inf,
    Q_beta: float = np.inf,
) -> LayerStack:
    """Generate a random velocity stack (no CPA, direct perturbation).

    Velocities drawn from N(mean, std^2), ensuring positive values.

    Args:
        alpha_mean, beta_mean, rho_mean: Mean properties.
        n_layers: Number of layers (including half-space).
        thickness: Layer thickness (m).
        dalpha_std, dbeta_std, drho_std: Standard deviations.
        seed: Random seed.
        Q_alpha, Q_beta: Quality factors for all layers.

    Returns:
        LayerStack with random isotropic layers.
    """
    rng = np.random.default_rng(seed)
    layers: list[IsotropicLayer] = []
    for i in range(n_layers):
        alpha = max(alpha_mean + rng.normal(0, dalpha_std), alpha_mean * 0.5)
        beta = max(beta_mean + rng.normal(0, dbeta_std), beta_mean * 0.5)
        rho = max(rho_mean + rng.normal(0, drho_std), rho_mean * 0.5)
        h = np.inf if i == n_layers - 1 else thickness
        layers.append(
            IsotropicLayer(
                alpha=float(alpha),
                beta=float(beta),
                rho=float(rho),
                thickness=h,
                Q_alpha=Q_alpha,
                Q_beta=Q_beta,
            )
        )
    return LayerStack(layers=layers)
