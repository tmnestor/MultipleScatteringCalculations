#!/usr/bin/env python3
"""Decision measurement: does the inter-voxel propagator's omitted radiation
(imaginary) part produce a measurable error in the slab Foldy-Lax solve?

MOTIVATED FIX 5 (radiation/imaginary part of the volume-averaged inter-voxel
propagator).  This script does NOT itself compare the module against the
arbiter — it prints only the MAGNITUDE of the omitted imaginary part
(Measurement A) and its slab-solve impact (Measurement B), which is what
motivated the fix.  The actual module-vs-arbiter validation lives in
``TestRadiationImaginaryPart`` in
``cubic_scattering/tests/test_inter_voxel_propagator.py``, which checks the
analytic imaginary-part moment series in
``cubic_scattering.inter_voxel_propagator`` against the same kind of complex
volume-averaged Kupradze quadrature used here.

Two measurements are printed:

Measurement A — magnitude of the OMITTED imaginary part.
    The analytic ``inter_voxel_propagator_9x9`` builds a REAL even-power-ω²ⁿ
    dynamic series; it structurally cannot represent the odd-power IMAGINARY
    (radiation) terms of the elastodynamic Green's tensor e^{ikr}/r.  To find
    how big the omitted part is, we compute the TRUE volume-averaged complex
    propagator by direct double Gauss-Legendre quadrature of the full complex
    Kupradze G (and its R-derivatives, by finite differences of the averaged
    G), and report |Im block| / |Re block| per 9×9 block (G, C, S) at
    face/edge/corner separations across ka ∈ {0.1, 0.3, 0.5, 0.8}.

    Quadrature caveat (from scripts/face_s_rederivation.py): the FACE-contact
    S block of the FD-of-<G> arbiter is biased by the shared lateral node set
    sampling the singular 1/w³ ray; that bias is REAL (static-kernel) and does
    not corrupt the IMAGINARY part we are measuring, but the face-S |Re| it is
    divided by is unreliable, so the face-S ratio is flagged.  G and C kernels
    (1/w .. 1/w²) and all edge/corner blocks are unaffected.

Measurement B — does it move the slab→Kennett error? (THE DECISION)
    A uniform N_z≥2 slab (so vertical inter-voxel coupling is exercised) at
    moderate contrast.  For ka ∈ {0.1, 0.2, 0.3, 0.5, 0.8} compute R_PP three
    ways and the relative error vs the exact Kennett reference:
        (1) volume_averaged=False           (point-propagator baseline)
        (2) volume_averaged=True, n_orders=2 (current real-only ω⁴ series)
        (3) volume_averaged=True, n_orders=3 (real-only ω⁶ series)
    The vol_avg(True) − vol_avg(False) difference isolates the propagator's
    own effect from the shared T-matrix/Weyl-truncation error.

Run:
    conda run -n seismic python scripts/test_radiation_part_need.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.resonance_tmatrix import (  # noqa: E402
    _voigt_contract,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
    uniform_slab_material,
)

# ── Shared study parameters ──────────────────────────────────────────────────
REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
N_GAUSS = 10  # Gauss-Legendre points per axis for the volume averages


# ══════════════════════════════════════════════════════════════════════════
#  Quadrature machinery (reused from scripts/t27_coupling_study.py)
# ══════════════════════════════════════════════════════════════════════════


def gauss_grid(n: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Legendre grid on the cube [-a, a]^3."""
    x, w = leggauss(n)
    x = x * a
    w = w * a
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    wts = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    return pts, wts


def greens_tensor(rvecs: np.ndarray, omega: float, ref: ReferenceMedium) -> np.ndarray:
    """Vectorised full complex Kupradze Green's tensor (same physics as
    resonance_tmatrix.elastodynamic_greens)."""
    r = np.linalg.norm(rvecs, axis=-1)
    g = rvecs / r[..., None]
    kP = omega / ref.alpha
    kS = omega / ref.beta
    eP = np.exp(1j * kP * r)
    eS = np.exp(1j * kS * r)
    nfP = (1.0 - 1j * kP * r) * eP / r**3
    nfS = (1.0 - 1j * kS * r) * eS / r**3
    phi = kS**2 * eS / r - nfS + nfP
    psi = 3.0 * nfS - 3.0 * nfP + kP**2 * eP / r - kS**2 * eS / r
    pref = 1.0 / (4.0 * np.pi * ref.rho * omega**2)
    return pref * (
        phi[..., None, None] * np.eye(3)
        + psi[..., None, None] * g[..., :, None] * g[..., None, :]
    )


def avg_greens(Rax: np.ndarray, omega: float, a: float, n: int = N_GAUSS) -> np.ndarray:
    """Double volume average <G>(R) = (1/V²) ∫∫ G(R + r − r') dV dV'."""
    pts, w = gauss_grid(n, a)
    V = (2.0 * a) ** 3
    sep = Rax[None, None, :] + pts[:, None, :] - pts[None, :, :]
    G = greens_tensor(sep, omega, REF)
    return np.einsum("m,q,mqij->ij", w, w, G) / V**2


def avg_point_propagator_fd(
    Rax: np.ndarray, omega: float, a: float, h: float, n: int = N_GAUSS
) -> np.ndarray:
    """Volume-averaged COMPLEX 9×9 propagator via FD R-derivatives of <G>.

    This is the TRUE object the analytic inter_voxel_propagator_9x9
    approximates — but evaluated with the full complex Kupradze G, so it
    carries the radiation (imaginary) part that the analytic real series omits.
    """
    e = np.eye(3)
    A0 = avg_greens(Rax, omega, a, n)
    Ap = [avg_greens(Rax + h * e[k], omega, a, n) for k in range(3)]
    Am = [avg_greens(Rax - h * e[k], omega, a, n) for k in range(3)]
    Gd = np.zeros((3, 3, 3), dtype=complex)
    Gdd = np.zeros((3, 3, 3, 3), dtype=complex)
    for k in range(3):
        Gd[:, :, k] = (Ap[k] - Am[k]) / (2.0 * h)
        Gdd[:, :, k, k] = (Ap[k] - 2.0 * A0 + Am[k]) / h**2
    for k in range(3):
        for ll in range(k + 1, 3):
            App = avg_greens(Rax + h * e[k] + h * e[ll], omega, a, n)
            Apm = avg_greens(Rax + h * e[k] - h * e[ll], omega, a, n)
            Amp = avg_greens(Rax - h * e[k] + h * e[ll], omega, a, n)
            Amm = avg_greens(Rax - h * e[k] - h * e[ll], omega, a, n)
            Gdd[:, :, k, ll] = Gdd[:, :, ll, k] = (App - Apm - Amp + Amm) / (4.0 * h**2)
    C, H, S = _voigt_contract(Gd, Gdd)
    P = np.zeros((9, 9), dtype=complex)
    P[:3, :3] = A0
    P[:3, 3:] = C
    P[3:, :3] = H
    P[3:, 3:] = S
    return P


# ══════════════════════════════════════════════════════════════════════════
#  Measurement A
# ══════════════════════════════════════════════════════════════════════════

_SEPARATIONS = {
    "face   (1,0,0)": np.array([1.0, 0.0, 0.0]),
    "edge   (1,1,0)": np.array([1.0, 1.0, 0.0]),
    "corner (1,1,1)": np.array([1.0, 1.0, 1.0]),
}
_KA_A = (0.1, 0.3, 0.5, 0.8)


def _block_ratio(P: np.ndarray, rows: slice, cols: slice) -> float:
    """|Im block| / |Re block| using Frobenius norms; nan if Re block ≈ 0."""
    blk = P[rows, cols]
    re = np.linalg.norm(blk.real)
    im = np.linalg.norm(blk.imag)
    if re < 1e-300:
        return float("nan")
    return im / re


def run_measurement_a() -> None:
    """Measure the omitted imaginary part of the true volume-averaged
    propagator, per block per separation per ka."""
    print("=" * 78)
    print("MEASUREMENT A — magnitude of the OMITTED imaginary (radiation) part")
    print("=" * 78)
    print(
        "True volume-averaged COMPLEX propagator by direct quadrature of the\n"
        "full Kupradze G; reported as |Im block| / |Re block| (Frobenius).\n"
        "The analytic module omits ALL of this Im part (real even-ω series).\n"
        f"Unit pitch a=0.5 (cube side 1), N_gauss={N_GAUSS}.\n"
        "  * face-S |Re| is quadrature-biased (shared singular ray) -> flagged.\n"
    )
    a = 0.5  # unit pitch: cube side = 2a = 1
    # FD step: << closest node-pair gap across a touching face, >> roundoff.
    h = 0.01
    header = f"{'separation':>16} {'ka':>5}  {'|Im/Re| G':>11}  {'|Im/Re| C':>11}  {'|Im/Re| S':>11}"
    print(header)
    print("-" * len(header))
    for name, R in _SEPARATIONS.items():
        for ka in _KA_A:
            omega = ka * REF.beta / a  # ka = omega * a / beta
            P = avg_point_propagator_fd(R, omega, a, h)
            rg = _block_ratio(P, slice(0, 3), slice(0, 3))
            rc = _block_ratio(P, slice(0, 3), slice(3, 9))
            rs = _block_ratio(P, slice(3, 9), slice(3, 9))
            flag = " *" if name.startswith("face") else ""
            print(f"{name:>16} {ka:5.2f}  {rg:11.4f}  {rc:11.4f}  {rs:11.4f}{flag}")
        print()


# ══════════════════════════════════════════════════════════════════════════
#  Measurement B
# ══════════════════════════════════════════════════════════════════════════

_KA_B = (0.1, 0.2, 0.3, 0.5, 0.8)
K_HAT = np.array([1.0, 0.0, 0.0])  # normal P-wave incidence


def _rpp_for_mode(
    geom: SlabGeometry,
    mat: SlabMaterial,
    omega: float,
    *,
    volume_averaged: bool,
    n_orders: int,
) -> complex:
    """Solve the periodic slab and extract specular R_PP."""
    res = compute_slab_scattering(
        geom,
        mat,
        omega,
        K_HAT,
        wave_type="P",
        gmres_tol=1e-9,
        volume_averaged=volume_averaged,
        n_orders=n_orders,
        periodic=True,
    )
    T_local = compute_slab_tmatrices(geom, mat, omega)
    return slab_rpp_periodic(res, T_local, p=0.0)


def run_measurement_b() -> None:
    """Compare slab R_PP to Kennett for the three propagator modes vs ka."""
    print("=" * 78)
    print("MEASUREMENT B — does the omitted part move the slab→Kennett error?")
    print("=" * 78)
    M = 4  # horizontal periodic tiling (specular response is M-independent)
    N_z = 3  # >= 2 so vertical inter-voxel coupling is exercised
    a = 1.0  # cube half-width; H = N_z * 2a
    H = N_z * 2.0 * a
    print(
        f"Uniform slab: M={M}, N_z={N_z}, a={a} m, H={H:.1f} m, "
        f"normal P incidence (periodic).\n"
        f"Background a/b/rho={REF.alpha}/{REF.beta}/{REF.rho}; "
        f"contrast dl/dm/drho={CONTRAST.Dlambda:.0e}/{CONTRAST.Dmu:.0e}/"
        f"{CONTRAST.Drho:.0f}.\n"
        "Rel-error = |R_FL - R_Kennett| / |R_Kennett|.\n"
    )
    geom = SlabGeometry(M=M, N_z=N_z, a=a)
    mat = uniform_slab_material(geom, REF, CONTRAST)

    header = (
        f"{'ka':>5}  {'|R_K|':>10}  "
        f"{'err pt(VA=F)':>12}  {'err VA n=2':>11}  {'err VA n=3':>11}  "
        f"{'|VAn2-pt|/|R_K|':>15}"
    )
    print(header)
    print("-" * len(header))
    for ka in _KA_B:
        omega = ka * REF.beta / a
        R_K = kennett_reference_rpp(REF, CONTRAST, H=H, omega=omega)
        R_pt = _rpp_for_mode(geom, mat, omega, volume_averaged=False, n_orders=2)
        R_v2 = _rpp_for_mode(geom, mat, omega, volume_averaged=True, n_orders=2)
        R_v3 = _rpp_for_mode(geom, mat, omega, volume_averaged=True, n_orders=3)
        absK = abs(R_K)
        e_pt = abs(R_pt - R_K) / absK
        e_v2 = abs(R_v2 - R_K) / absK
        e_v3 = abs(R_v3 - R_K) / absK
        diff_va = abs(R_v2 - R_pt) / absK
        print(
            f"{ka:5.2f}  {absK:10.4e}  "
            f"{e_pt:12.4e}  {e_v2:11.4e}  {e_v3:11.4e}  {diff_va:15.4e}"
        )
    print(
        "\nINTERPRETATION KEY:\n"
        "  - VA tracks/beats pt and stays <~1-2% across ka  -> Fix 5 low priority.\n"
        "  - VA error grows systematically with ka, worse than/equal to pt, or\n"
        "    n=2 vs n=3 doesn't converge                    -> radiation-gap signature.\n"
        "  - |VAn2-pt|/|R_K| ~ e_pt (propagator effect drowned by shared error)\n"
        "    -> propagator is not the bottleneck; Fix 5 won't help."
    )


def main() -> None:
    run_measurement_a()
    print()
    run_measurement_b()


if __name__ == "__main__":
    main()
