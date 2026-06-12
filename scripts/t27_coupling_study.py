#!/usr/bin/env python3
"""T27 inter-voxel coupling study: quadrature propagator, calibration, decision table.

Decision-measurement for whether a 27-component lattice solver is justified:
how much do the 18 quadratic inter-voxel coupling channels (T27 basis indices
9-26) change a coupled two-voxel response, beyond the validated 9-component
(displacement + Voigt strain) chain?

Conventions (explicit)
----------------------
State per voxel: ``c`` = Galerkin expansion coefficients of the exciting
displacement field on the 27 polynomial basis ``phi_a`` (physical coordinates,
cube [-a, a]^3, ordering documented in ``tmatrix_assembly.py``).  Projection:
``c = M^-1 <phi, u>`` with mass matrix ``M_ab = int phi_a . phi_b dV``.
For the first 9 modes these coefficients coincide with the validated 9-state
[u(0), Voigt strain with engineering shear] up to O((ka)^2, curvature).

Single site: total internal field ``c_tot = (I + T27) c_exc`` with T27 from
``assemble_tmatrix_27``.  NOTE: T27's internal normalisation is only
unambiguous at a = 1 (its displacement-channel entries scale as 1/a^2 at fixed
ka, so scaled vs physical quadratic coordinates cannot be distinguished away
from a = 1).  This study uses a = 1 m throughout.

Source: equivalent polynomial force density ``q = sum_b s_b phi_b`` with
``s = S_hat c_tot`` and ``S_hat = M^-1 (omega^2 Drho M - Bel)``, the Galerkin
projection of the contrast operator onto the polynomial density space
(``Bel_ab = int eps[phi_a] : DC : eps[phi_b] dV``; boundary tractions are
absorbed by integration by parts).  Full source operator:
``T_hat = S_hat (I + T27)``.

Propagation: ``P_ab(R) = intint phi_a(r) . G(R + r - r') . phi_b(r') dV dV'``
with ``R = x_field - x_source`` and G the Kupradze full-space Green's tensor.
Scattered-field coefficients at the receiving voxel: ``c_sc = M^-1 P(R) s``.
Identity used and verified: ``P(-R) = P(R)^T`` (G is even and symmetric).

Foldy-Lax for two voxels at x1 and x2 = x1 + R::

    c1 = c1_inc + M^-1 P(-R) T_hat c2
    c2 = c2_inc + M^-1 P(R)  T_hat c1

Calibration: ``M9^-1 P[0:9, 0:9] W^-1 -> exact_propagator_9x9`` as a/R -> 0,
with ``W = diag(V, V, V, -V a^2/3 x6)``, ``V = (2a)^3``.  The minus sign on
the strain channels is the codebase T-matrix force/moment sign convention
(cf. the slab_rpp_periodic force-sign notes).  At touching separations the
volume-averaged propagator genuinely deviates from the point propagator by
O((a/R)^2) — a real finite-size effect, not a normalisation failure.

Step 2b: the analytic nearest-neighbour propagator in
``inter_voxel_propagator.py`` (``inter_voxel_propagator_9x9``, used by
``slab_scattering`` when ``volume_averaged=True``) is compared against two
quadrature references: (i) the FD volume-averaged point propagator (the
analytic module's own definition: uniform multipole source, volume-averaged
field, ``_voigt_contract`` conventions) and (ii) this study's Galerkin object.
The analytic module is hardcoded to unit pitch (cube side 1, half-width 0.5).

Quadratic-channel isolation: the raw monomial squares r_p^2 carry net-force
(monopole) content (M couples them to the constant modes), so zeroing raw
P rows/cols 9-26 breaks a monopole cancellation and produces artifacts.  The
primary "quadratic channels OFF in P" variant therefore M-orthogonalises the
squares against the constants (r_p^2 - a^2/3, zero monopole content) before
zeroing the quadratic channels.  The raw-basis variant is reported alongside
for transparency.

Run:
    conda run -n seismic python scripts/t27_coupling_study.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
    compute_cube_tmatrix_galerkin,
)
from cubic_scattering.horizontal_greens import exact_propagator_9x9  # noqa: E402
from cubic_scattering.incident_field import cube_overlap_integrals  # noqa: E402
from cubic_scattering.inter_voxel_propagator import (  # noqa: E402
    inter_voxel_propagator_9x9,
)
from cubic_scattering.resonance_tmatrix import (  # noqa: E402
    _sub_cell_tmatrix_9x9,
    _voigt_contract,
    elastodynamic_greens,
)
from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27  # noqa: E402

# ── Study parameters ─────────────────────────────────────────────────────────

REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
A = 1.0  # cube half-width (m) — T27 normalisation requires a = 1
KA_VALUES = (0.1, 0.3, 0.5)
N_GAUSS = 8  # production Gauss-Legendre points per axis

# Quadratic monomial exponents, axis order r1 = axis 0 (z), r2 = axis 1 (x),
# r3 = axis 2 (y) — matches incident_field._QUAD_EXPONENTS.
_QUAD_EXP = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
# Shear pairs for basis modes 6-8 — matches VOIGT_PAIRS[3:6].
_SHEAR_PAIRS = [(1, 2), (0, 2), (0, 1)]


# ── Quadrature grid and basis evaluation ─────────────────────────────────────


def gauss_grid(n: int, a: float) -> tuple[NDArray, NDArray]:
    """Tensor-product Gauss-Legendre grid on the cube [-a, a]^3.

    Args:
        n: Points per axis (n^3 total).
        a: Cube half-width (m).

    Returns:
        (points, weights): shapes (n^3, 3) and (n^3,).
    """
    x, w = leggauss(n)
    x = x * a
    w = w * a
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    wts = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    return pts, wts


def basis_values(pts: NDArray) -> NDArray:
    """Evaluate the 27 vector basis functions at points.

    Args:
        pts: Points, shape (npts, 3), axis order (z, x, y) = (r1, r2, r3).

    Returns:
        Phi: shape (27, npts, 3) — phi_a,i(r).
    """
    npts = pts.shape[0]
    Phi = np.zeros((27, npts, 3))
    for i in range(3):
        Phi[i, :, i] = 1.0
    for k in range(3):
        Phi[3 + k, :, k] = pts[:, k]
    for s, (p, q) in enumerate(_SHEAR_PAIRS):
        Phi[6 + s, :, p] = 0.5 * pts[:, q]
        Phi[6 + s, :, q] = 0.5 * pts[:, p]
    for d in range(3):
        for m, (e1, e2, e3) in enumerate(_QUAD_EXP):
            mono = pts[:, 0] ** e1 * pts[:, 1] ** e2 * pts[:, 2] ** e3
            Phi[9 + 6 * d + m, :, d] = mono
    return Phi


def basis_strains(pts: NDArray) -> NDArray:
    """Strain tensors eps[phi_a] at points.

    Args:
        pts: Points, shape (npts, 3).

    Returns:
        E: shape (27, npts, 3, 3) — symmetric strain of each basis function.
    """
    npts = pts.shape[0]
    E = np.zeros((27, npts, 3, 3))
    for k in range(3):
        E[3 + k, :, k, k] = 1.0
    for s, (p, q) in enumerate(_SHEAR_PAIRS):
        E[6 + s, :, p, q] = 0.5
        E[6 + s, :, q, p] = 0.5
    for d in range(3):
        for m, exps in enumerate(_QUAD_EXP):
            grad = np.zeros((npts, 3))
            for ax in range(3):
                if exps[ax] == 0:
                    continue
                e2 = list(exps)
                e2[ax] -= 1
                grad[:, ax] = exps[ax] * (
                    pts[:, 0] ** e2[0] * pts[:, 1] ** e2[1] * pts[:, 2] ** e2[2]
                )
            idx = 9 + 6 * d + m
            E[idx, :, d, :] += 0.5 * grad
            E[idx, :, :, d] += 0.5 * grad
    return E


def mass_matrix(a: float, n: int = 6) -> NDArray:
    """Mass matrix M_ab = int phi_a . phi_b dV (exact for n >= 3).

    Args:
        a: Cube half-width (m).
        n: Gauss points per axis (degree-4 integrand: exact for n >= 3).

    Returns:
        M: shape (27, 27).
    """
    pts, w = gauss_grid(n, a)
    Phi = basis_values(pts)
    return np.einsum("ani,bni,n->ab", Phi, Phi, w)


def bel_matrix(a: float, contrast: MaterialContrast, n: int = 6) -> NDArray:
    """Elastic stiffness bilinear Bel_ab = int eps[phi_a] : DC : eps[phi_b] dV.

    DC is the isotropic contrast tensor (Dlambda, Dmu).

    Args:
        a: Cube half-width (m).
        contrast: Material contrast.
        n: Gauss points per axis (degree-2 integrand: exact for n >= 2).

    Returns:
        Bel: shape (27, 27), units Pa * m^3-ish (density-source convention).
    """
    pts, w = gauss_grid(n, a)
    E = basis_strains(pts)
    tr = np.einsum("anii->an", E)
    Bel = contrast.Dlambda * np.einsum("an,bn,n->ab", tr, tr, w)
    Bel += 2.0 * contrast.Dmu * np.einsum("anij,bnij,n->ab", E, E, w)
    return Bel


# ── Vectorised Green's tensor and quadrature propagator ──────────────────────


def greens_tensor(rvecs: NDArray, omega: float, ref: ReferenceMedium) -> NDArray:
    """Vectorised Kupradze full-space Green's tensor.

    Same physics as resonance_tmatrix.elastodynamic_greens, evaluated on
    an array of displacement vectors.

    Args:
        rvecs: Displacement vectors x_field - x_source, shape (..., 3).
        omega: Angular frequency (rad/s).
        ref: Background medium.

    Returns:
        G: shape (..., 3, 3), complex.
    """
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


def quad_propagator_27(
    R: NDArray,
    omega: float,
    ref: ReferenceMedium,
    a: float,
    n: int = N_GAUSS,
    chunk: int = 256,
) -> NDArray:
    """27x27 inter-voxel propagator by direct double quadrature.

    P_ab(R) = intint phi_a(r) . G(R + r - r') . phi_b(r') dV dV'
    with R = x_field - x_source.  Both cubes have half-width a; they must
    not overlap (|R| components must keep the cubes disjoint).

    Args:
        R: Separation vector x_field_voxel - x_source_voxel, shape (3,),
            axis order (z, x, y).
        omega: Angular frequency (rad/s).
        ref: Background medium.
        a: Cube half-width (m).
        n: Gauss-Legendre points per axis.
        chunk: Field-point chunk size (memory cap for the G array).

    Returns:
        P: shape (27, 27), complex.
    """
    pts, w = gauss_grid(n, a)
    Phi = basis_values(pts)  # (27, N, 3)
    Bw = (Phi * w[None, :, None]).reshape(27, -1)  # (27, 3N) source side
    npts = pts.shape[0]
    P = np.zeros((27, 27), dtype=complex)
    for start in range(0, npts, chunk):
        sl = slice(start, min(start + chunk, npts))
        sep = R[None, None, :] + pts[sl, None, :] - pts[None, :, :]
        G = greens_tensor(sep, omega, ref)  # (nc, N, 3, 3)
        nc = G.shape[0]
        Gf = G.transpose(0, 2, 1, 3).reshape(3 * nc, 3 * npts)
        Bw_c = (Phi[:, sl, :] * w[None, sl, None]).reshape(27, -1)
        P += Bw_c @ Gf @ Bw.T
    return P


# ── Single-site source operator ──────────────────────────────────────────────


def source_operator(
    omega: float,
    a: float,
    ref: ReferenceMedium,
    contrast: MaterialContrast,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build the 27x27 source operator T_hat = S_hat (I + T27).

    S_hat = M^-1 (omega^2 Drho M - Bel) maps total-field coefficients to
    equivalent polynomial source-density coefficients (Galerkin projection
    of the contrast operator, codebase T-matrix sign convention).

    Args:
        omega: Angular frequency (rad/s).
        a: Cube half-width (m) — must be 1.0 for T27 normalisation.
        ref: Background medium.
        contrast: Material contrast.

    Returns:
        (T_hat, M, T27): source operator, mass matrix, single-site T27.
    """
    M = mass_matrix(a)
    Bel = bel_matrix(a, contrast)
    S_hat = omega**2 * contrast.Drho * np.eye(27) - np.linalg.solve(M, Bel)
    galerkin = compute_cube_tmatrix_galerkin(omega, a, ref, contrast)
    T27 = assemble_tmatrix_27(galerkin)
    return S_hat @ (np.eye(27) + T27), M, T27


def source_operator_9(
    omega: float,
    a: float,
    contrast: MaterialContrast,
    M: NDArray,
    T27: NDArray,
) -> NDArray:
    """9x9 truncated source operator: S_hat9 (I + T27[0:9, 0:9])."""
    M9 = M[:9, :9]
    Bel9 = bel_matrix(a, contrast)[:9, :9]
    S9 = omega**2 * contrast.Drho * np.eye(9) - np.linalg.solve(M9, Bel9)
    return S9 @ (np.eye(9) + T27[:9, :9])


def ortho_transform(a: float) -> NDArray:
    """Basis change X (psi_a = sum_b X[a,b] phi_b) M-orthogonalising squares.

    The quadratic square monomials r_p^2 e_d are replaced by
    (r_p^2 - a^2/3) e_d, which have zero overlap with the constant modes
    (zero net-force content).  All other modes are unchanged.
    """
    X = np.eye(27)
    for d in range(3):
        for m in range(3):
            X[9 + 6 * d + m, d] = -(a**2) / 3.0
    return X


# ── Two-voxel Foldy-Lax solve ────────────────────────────────────────────────


def solve_pair(
    Pq: NDArray,
    M: NDArray,
    T_hat: NDArray,
    c0: NDArray,
    kvec: NDArray,
    R: NDArray,
    variant: str,
    a: float,
) -> NDArray:
    """Solve the coupled two-voxel system and return the 9-component observable.

    The observable is the pair's total effective source (net force + first
    moment content of each voxel's polynomial source density):
    s = sum_m (M T_hat c_exc,m)[0:9].

    Args:
        Pq: Full 27x27 quadrature propagator P(R), R = x2 - x1.
        M: 27x27 mass matrix.
        T_hat: Source operator — 27x27, or 9x9 for variant "nine_only".
        c0: Incident-field coefficients at voxel 1 (27,).
        kvec: Incident wave vector (axis order z, x, y).
        R: Separation vector x2 - x1.
        variant: One of:
            "full" — full 27x27 coupling (i);
            "quad_off_P" — quadratic channels of P zeroed in the
                M-orthogonalised basis (primary variant ii);
            "quad_off_P_raw" — raw-basis P rows/cols 9-26 zeroed
                (artifact-prone, reported for transparency);
            "quad_off_K" — rows/cols 9-26 of the coupling operator
                K = M^-1 P T_hat zeroed (alternative clean isolation);
            "nine_only" — 9-component truncation everywhere (iii).
        a: Cube half-width (m).

    Returns:
        s: shape (9,), complex — summed effective source.
    """
    if variant == "nine_only":
        nb = 9
        M_use = M[:9, :9]
        Th = T_hat
        c0v = c0[:9]
        K21 = np.linalg.solve(M_use, Pq[:9, :9]) @ Th
        K12 = np.linalg.solve(M_use, Pq[:9, :9].T) @ Th
    else:
        nb = 27
        M_use = M
        Th = T_hat
        c0v = c0
        if variant == "quad_off_P":
            X = ortho_transform(a)
            Xinv = np.linalg.inv(X)
            Mp = X @ M @ X.T
            Thp = Xinv.T @ Th @ X.T
            Ks = []
            for Pdir in (Pq, Pq.T):
                Pp = X @ Pdir @ X.T
                Pp[9:, :] = 0.0
                Pp[:, 9:] = 0.0
                Kp = np.linalg.solve(Mp, Pp) @ Thp
                Ks.append(X.T @ Kp @ Xinv.T)
            K21, K12 = Ks
        elif variant == "quad_off_P_raw":
            P21 = Pq.copy()
            P12 = Pq.T.copy()
            for P_ in (P21, P12):
                P_[9:, :] = 0.0
                P_[:, 9:] = 0.0
            K21 = np.linalg.solve(M_use, P21) @ Th
            K12 = np.linalg.solve(M_use, P12) @ Th
        else:  # "full" or "quad_off_K"
            K21 = np.linalg.solve(M_use, Pq) @ Th
            K12 = np.linalg.solve(M_use, Pq.T) @ Th
            if variant == "quad_off_K":
                for K in (K21, K12):
                    K[9:, :] = 0.0
                    K[:, 9:] = 0.0
    A_sys = np.eye(2 * nb, dtype=complex)
    A_sys[:nb, nb:] -= K12
    A_sys[nb:, :nb] -= K21
    rhs = np.concatenate([c0v, c0v * np.exp(1j * float(np.dot(kvec, R)))])
    c = np.linalg.solve(A_sys, rhs)
    return (M_use @ (Th @ c[:nb]))[:9] + (M_use @ (Th @ c[nb:]))[:9]


# ── Calibration helpers ──────────────────────────────────────────────────────

_BLOCKS = {
    "G": (slice(0, 3), slice(0, 3)),
    "C": (slice(0, 3), slice(3, 9)),
    "H": (slice(3, 9), slice(0, 3)),
    "S": (slice(3, 9), slice(3, 9)),
}


def weight_matrix(a: float) -> NDArray:
    """Diagonal W mapping density coefficients to point-source strengths."""
    V = (2.0 * a) ** 3
    return np.diag([V] * 3 + [-V * a**2 / 3.0] * 6)


def galerkin_9_block(Rax: NDArray, omega: float, a: float, n: int = N_GAUSS) -> NDArray:
    """Quadrature truth in the point-propagator convention: M9^-1 Pq[0:9,0:9] W^-1.

    Args:
        Rax: Separation in axis order (z, x, y).
        omega: Angular frequency.
        a: Cube half-width.
        n: Gauss points per axis.

    Returns:
        D: shape (9, 9) complex — directly comparable to exact_propagator_9x9
        (and to inter_voxel_propagator_9x9 at unit pitch, a = 0.5).
    """
    Pq = quad_propagator_27(Rax, omega, REF, a, n=n)
    M9 = mass_matrix(a)[:9, :9]
    return np.linalg.solve(M9, Pq[:9, :9]) @ np.linalg.inv(weight_matrix(a))


def block_devs(X: NDArray, ref_mat: NDArray) -> dict[str, float]:
    """Per-block max deviation of X from ref_mat, normalised by block scale."""
    out = {}
    for bn, (r_, c_) in _BLOCKS.items():
        scale = np.max(np.abs(ref_mat[r_, c_]))
        out[bn] = float(np.max(np.abs(X[r_, c_] - ref_mat[r_, c_])) / scale)
    return out


def nine_block_deviation(
    Rax: NDArray, omega: float, a: float, n: int = N_GAUSS
) -> dict[str, float]:
    """Block-scale relative deviation of M9^-1 Pq W^-1 from exact_propagator_9x9.

    Args:
        Rax: Separation in axis order (z, x, y).
        omega: Angular frequency.
        a: Cube half-width.
        n: Gauss points per axis.

    Returns:
        Per-block max deviation normalised by the block's max magnitude.
    """
    D = galerkin_9_block(Rax, omega, a, n=n)
    # exact_propagator_9x9 signature is (x, y, z); axis order here is (z, x, y)
    Pp = exact_propagator_9x9(Rax[1], Rax[2], Rax[0], omega, REF)
    return block_devs(D, Pp)


def run_calibration() -> None:
    """Print the Step-2 calibration: machinery, normalisation, finite-size."""
    print("=" * 78)
    print("STEP 2 — CALIBRATION")
    print("=" * 78)
    a = A
    ka = 0.1
    omega = ka * REF.beta / a

    # 1. Vectorised Green's tensor vs validated elastodynamic_greens
    rng = np.random.default_rng(7)
    pts_test = rng.uniform(1.5, 5.0, size=(5, 3))
    errs = []
    for r_vec in pts_test:
        G_ref = elastodynamic_greens(r_vec, omega, REF)
        G_vec = greens_tensor(r_vec[None, :], omega, REF)[0]
        errs.append(np.max(np.abs(G_vec - G_ref)) / np.max(np.abs(G_ref)))
    print(
        f"\nGreen's tensor (vectorised vs elastodynamic_greens): "
        f"max rel err = {max(errs):.2e}"
    )

    # 2. Reciprocity P(-R) = P(R)^T
    Rt = np.array([2.0 * a, 1.0 * a, 0.5 * a])
    Pp_ = quad_propagator_27(Rt, omega, REF, a, n=6)
    Pm_ = quad_propagator_27(-Rt, omega, REF, a, n=6)
    rec = np.max(np.abs(Pm_ - Pp_.T)) / np.max(np.abs(Pp_))
    print(f"Reciprocity P(-R) = P(R)^T: max rel err = {rec:.2e}")

    # 3. Point-propagator limit a -> 0 at fixed R (validates normalisation W)
    print("\nNormalisation W = diag(V, V, V, -V a^2/3 x6), field side M9^-1.")
    print("Point limit (fixed R = 2 m, omega for ka(a=1) = 0.1; shrink a):")
    print(f"  {'a/R':>6}  {'G':>9}  {'C':>9}  {'H':>9}  {'S':>9}")
    Rfix = np.array([2.0, 0.0, 0.0])
    for a_small in (0.25, 0.1, 0.05):
        dev = nine_block_deviation(Rfix, omega, a_small, n=8)
        print(
            f"  {a_small / 2.0:>6.3f}  " + "  ".join(f"{dev[b]:>9.2e}" for b in "GCHS")
        )

    # 4. Touching separations at a = 1 (the genuine volume-averaging effect)
    print("\nTouching separations at a = 1 (block-scale deviation from point")
    print("propagator — real O((a/R)^2) volume-averaging, not a bug):")
    for name, Rax in (
        ("face", np.array([2.0, 0.0, 0.0])),
        ("corner", np.array([2.0, 2.0, 2.0])),
    ):
        dev = nine_block_deviation(Rax, omega, A, n=N_GAUSS)
        print(f"  {name:>6}: " + "  ".join(f"{b}={dev[b]:.2e}" for b in "GCHS"))

    # 5. Quadrature convergence n = 8 vs 10 vs 12 at face separation
    print("\nQuadrature convergence of P entries at face separation (worst case,")
    print("near-singular 1/r^3 kernel across the shared face):")
    Rface = np.array([2.0 * a, 0.0, 0.0])
    P_by_n = {n: quad_propagator_27(Rface, omega, REF, a, n=n) for n in (8, 10, 12)}
    for n_hi in (10, 12):
        diff = np.abs(P_by_n[n_hi] - P_by_n[8])
        scale = np.abs(P_by_n[n_hi])
        mask = scale > 1e-6 * np.max(scale)
        print(
            f"  n=8 vs n={n_hi}: max rel change (significant entries) = "
            f"{np.max(diff[mask] / scale[mask]):.2e}"
        )

    # 6. Source operator 9-block vs validated T_loc
    T_hat, M, _ = source_operator(omega, a, REF, CONTRAST)
    rayleigh = compute_cube_tmatrix(omega, a, REF, CONTRAST)
    T_loc = _sub_cell_tmatrix_9x9(rayleigh, omega, a)
    S_mine = weight_matrix(a) @ T_hat[:9, :9]
    mask = np.abs(T_loc) > 1e-8 * np.max(np.abs(T_loc))
    ratios = S_mine[mask] / T_loc[mask]
    print(
        "\nSource operator W T_hat[0:9,0:9] vs validated T_loc "
        "(Galerkin vs Path-A amplification):"
    )
    print(
        f"  ratio range over nonzero entries: "
        f"[{np.min(np.abs(ratios)):.4f}, {np.max(np.abs(ratios)):.4f}]"
    )


# ── Step 2b: existing analytic volume-averaged propagator vs quadrature ──────


def avg_greens(Rax: NDArray, omega: float, a: float, n: int = N_GAUSS) -> NDArray:
    """Double volume average <G>(R) = (1/V^2) intint G(R + r - r') dV dV'."""
    pts, w = gauss_grid(n, a)
    V = (2.0 * a) ** 3
    sep = Rax[None, None, :] + pts[:, None, :] - pts[None, :, :]
    G = greens_tensor(sep, omega, REF)
    return np.einsum("m,q,mqij->ij", w, w, G) / V**2


def avg_point_propagator_fd(
    Rax: NDArray, omega: float, a: float, h: float = 0.005, n: int = N_GAUSS
) -> NDArray:
    """Volume average of the point 9x9 propagator via FD derivatives of <G>.

    This is the object the analytic inter_voxel_propagator_9x9 documents:
    uniform unit-strength multipole source over the source voxel, field
    averaged over the field voxel, with the _voigt_contract conventions.
    All derivative blocks are R-derivatives of <G>(R), evaluated by
    second-order central differences.

    The FD step must be much smaller than the closest quadrature node-pair
    gap across a touching face (~0.04 at n = 8, a = 0.5); h = 0.005 matches
    direct quadrature of the derivative kernels to ~1% there.  The S block
    at face contact retains a slow (log-type) n-convergence shared with the
    direct integral; the study cross-checks by direct n-refinement.

    Args:
        Rax: Separation in axis order (z, x, y).
        omega: Angular frequency.
        a: Cube half-width.
        h: FD step (same length units as Rax).
        n: Gauss points per axis for each <G> evaluation.

    Returns:
        P: shape (9, 9) complex — [[<G>, C], [H, S]].
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


def patch_h_engineering(P9: NDArray) -> NDArray:
    """Double the shear (engineering Voigt) rows of the H block.

    HISTORICAL: at the time of the study, inter_voxel_propagator_9x9 set
    H = C^T, which dropped the factor 2 that the engineering shear-strain
    rows carry in the validated convention (its own S block applies it via
    mult_pq = 2), and this patch restored it.  The module has since been
    FIXED (H = W C^T with W = diag(1,1,1,2,2,2)), so applying this patch
    to current output double-counts the factor; it is kept only to
    reproduce the pre-fix study tables.
    """
    out = P9.copy()
    out[6:9, 0:3] *= 2.0
    return out


def run_propagator_comparison() -> None:
    """Step 2b: arbitrate inter_voxel_propagator_9x9 against quadrature truth."""
    print()
    print("=" * 78)
    print("STEP 2b — EXISTING ANALYTIC VOLUME-AVERAGED 9x9 PROPAGATOR vs QUADRATURE")
    print("=" * 78)
    a = 0.5  # the analytic module is hardcoded to UNIT PITCH (cube side 1)
    omega_static = 1e-3 * REF.beta / a

    # 1. Scale convention: analytic G vs quadrature <G> at two half-widths
    print("\nScale convention (analytic module has no length argument):")
    for a_try in (0.5, 1.0):
        Rphys = np.array([2.0 * a_try, 0.0, 0.0])
        om = 1e-3 * REF.beta / a_try
        G_q = avg_greens(Rphys, om, a_try, n=8)
        P9 = inter_voxel_propagator_9x9((1, 0, 0), REF.alpha, REF.beta, REF.rho, 0.0, 0)
        ratio = np.real(P9[0, 0] / G_q[0, 0])
        print(
            f"  half-width a={a_try}: analytic G[0,0] / quadrature <G>[0,0] = {ratio:.4f}"
        )
    print("  -> matches at a = 0.5 only: hardcoded to pitch d = 1 (slab usage with")
    print("     d != 1, e.g. tests at a = 1, scales G/C/S wrongly by d, d^2, d^3).")

    # 2. Static per-block comparison vs both references
    print("\nStatic comparison (ka -> 0), per-block max deviation / block scale.")
    print("  ref FD-avg = volume-averaged point propagator (the analytic module's")
    print("  own definition); ref Galerkin = this study's M9^-1 P_quad W^-1.")
    print(
        f"  {'sep':>6} {'ref':>9}  {'G':>9}  {'C':>9}  {'H':>9}  {'S':>9}   (H-patched: H')"
    )
    seps = {
        "face": ((1, 0, 0), np.array([1.0, 0.0, 0.0])),
        "edge": ((1, 1, 0), np.array([1.0, 1.0, 0.0])),
        "corner": ((1, 1, 1), np.array([1.0, 1.0, 1.0])),
    }
    for name, (Rlat, Rphys) in seps.items():
        P9 = inter_voxel_propagator_9x9(Rlat, REF.alpha, REF.beta, REF.rho, 0.0, 0)
        P9p = patch_h_engineering(P9)
        for ref_name, ref_mat in (
            ("FD-avg", avg_point_propagator_fd(Rphys, omega_static, a, n=8)),
            ("Galerkin", galerkin_9_block(Rphys, omega_static, a, n=8)),
        ):
            dev = block_devs(P9, ref_mat)
            dev_p = block_devs(P9p, ref_mat)
            print(
                f"  {name:>6} {ref_name:>9}  "
                + "  ".join(f"{dev[b]:>9.2e}" for b in "GCHS")
                + f"   (H'={dev_p['H']:.2e})"
            )

    # 3. Specific structural findings (static, evidence entries)
    print("\nStructural findings (static, unit pitch):")
    # H = C^T misses engineering factor 2 on shear rows
    P9e = inter_voxel_propagator_9x9((1, 1, 0), REF.alpha, REF.beta, REF.rho, 0.0, 0)
    Dfd = avg_point_propagator_fd(np.array([1.0, 1.0, 0.0]), omega_static, a, n=8)
    mask = np.abs(Dfd[6:9, 0:3]) > 0.05 * np.max(np.abs(Dfd[6:9, 0:3]))
    h_ratio = np.real(P9e[6:9, 0:3][mask] / Dfd[6:9, 0:3][mask])
    print(
        f"  [H bug] H = C^T drops engineering 2x on shear rows: measured "
        f"H_analytic/H_true on edge shear rows = "
        f"[{h_ratio.min():.4f}, {h_ratio.max():.4f}] (should be 1)"
    )
    # Corner S breaks its own S3 symmetry
    P9c = inter_voxel_propagator_9x9((1, 1, 1), REF.alpha, REF.beta, REF.rho, 0.0, 0)
    Sc = np.real(P9c[3:, 3:])
    print(
        f"  [corner S bug] S3-symmetry partners unequal: S[1,4]={Sc[1, 4]:.3e} vs "
        f"S[0,3]={Sc[0, 3]:.3e}; S[3,5]={Sc[3, 5]:.3e} vs S[3,4]={Sc[3, 4]:.3e}"
    )
    # Face S magnitude/sign vs n-refined volume average
    P9f = inter_voxel_propagator_9x9((1, 0, 0), REF.alpha, REF.beta, REF.rho, 0.0, 0)
    Dfd_f = avg_point_propagator_fd(np.array([1.0, 0.0, 0.0]), omega_static, a, n=10)
    print(
        f"  [face S] analytic S[0,0]={np.real(P9f[3, 3]):.3e} vs volume-avg truth "
        f"~{np.real(Dfd_f[3, 3]):.3e} (n-refinement trends to ~1.0e-11: 3x+ low); "
        f"S[4,4]={np.real(P9f[7, 7]):.3e} vs ~{np.real(Dfd_f[7, 7]):.3e} (sign flip)"
    )

    # 4. Dynamic corrections at edge/corner (clean statics) vs FD-avg reference.
    # The analytic omega^(2n) series is identically REAL: it can represent the
    # near-field dispersion but not the imaginary (radiation) part of the
    # propagator.  Report the real-part residual and the missing Im fraction.
    print("\nDynamic residual (edge/corner, where statics are clean; H patched x2):")
    print("  dev(Re) = real-part deviation; Im/scale = imaginary part of the truth,")
    print("  which the (real) omega^2n series omits entirely.")
    print(
        f"  {'sep':>6} {'ka':>5} {'n_ord':>5}  "
        f"{'G dev(Re)':>10} {'C dev(Re)':>10} {'S dev(Re)':>10}  "
        f"{'G Im/sc':>8} {'S Im/sc':>8}"
    )
    for name in ("edge", "corner"):
        Rlat, Rphys = seps[name]
        for ka in (0.1, 0.5):
            om = ka * REF.beta / a
            ref_mat = avg_point_propagator_fd(Rphys, om, a, n=8)
            for n_ord in (2, 3):
                P9 = patch_h_engineering(
                    inter_voxel_propagator_9x9(
                        Rlat, REF.alpha, REF.beta, REF.rho, om, n_ord
                    )
                )
                dev_re = block_devs(np.real(P9), np.real(ref_mat))
                row = f"  {name:>6} {ka:>5.1f} {n_ord:>5}  "
                row += "".join(f"{dev_re[b]:>10.2e} " for b in "GCS")
                for b in "GS":
                    r_, c_ = _BLOCKS[b]
                    scale = np.max(np.abs(ref_mat[r_, c_]))
                    row += f" {np.max(np.abs(np.imag(ref_mat[r_, c_]))) / scale:>8.2e}"
                print(row)


# ── Decision table ───────────────────────────────────────────────────────────


def run_decision_table() -> None:
    """Print the Step-4 decision table for all separations, ka, and waves."""
    print()
    print("=" * 78)
    print("STEPS 3-4 — TWO-VOXEL COUPLED SOLVE AND DECISION TABLE")
    print("=" * 78)
    a = A
    M = mass_matrix(a)
    seps = {
        "face": np.array([2 * a, 0.0, 0.0]),
        "edge": np.array([2 * a, 2 * a, 0.0]),
        "corner": np.array([2 * a, 2 * a, 2 * a]),
    }
    print(f"""
Observable: s = sum_voxels (M T_hat c_exc)[0:9] (net force + stress-dipole
content of the pair's polynomial source).  Columns:
  d(i,ii)      quadratic channels OFF in P only, M-orthogonalised (PRIMARY)
  d(i,ii_raw)  same but raw-basis zeroing of P rows/cols 9-26 (artifact-prone:
               raw squares carry monopole content; see docstring)
  d(i,ii_K)    rows/cols 9-26 of coupling operator K = M^-1 P T_hat zeroed
  d(i,iii)     9-component truncation everywhere (total quadratic effect)
Incidence along z; P pol = z, SV pol = x.  a = {a} m, n = {N_GAUSS} per axis.
""")
    hdr = (
        f"{'wave':>4} {'sep':>6} {'ka':>4} {'d(i,ii)':>11} "
        f"{'d(i,ii_raw)':>12} {'d(i,ii_K)':>11} {'d(i,iii)':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    # Pre-compute propagators (independent of wave type)
    Pq_cache: dict[tuple[str, float], NDArray] = {}
    for sname, R in seps.items():
        for ka in KA_VALUES:
            omega = ka * REF.beta / a
            Pq_cache[(sname, ka)] = quad_propagator_27(R, omega, REF, a, n=N_GAUSS)

    for wave in ("P", "SV"):
        for sname, R in seps.items():
            for ka in KA_VALUES:
                omega = ka * REF.beta / a
                kmag = omega / (REF.alpha if wave == "P" else REF.beta)
                kvec = np.array([kmag, 0.0, 0.0])
                pol = (
                    np.array([1.0, 0.0, 0.0])
                    if wave == "P"
                    else np.array([0.0, 1.0, 0.0])
                )
                c0 = np.linalg.solve(M, cube_overlap_integrals(kvec, pol, a))
                T_hat, _, T27 = source_operator(omega, a, REF, CONTRAST)
                T_hat9 = source_operator_9(omega, a, CONTRAST, M, T27)
                Pq = Pq_cache[(sname, ka)]
                obs_i = solve_pair(Pq, M, T_hat, c0, kvec, R, "full", a)
                norm_i = np.linalg.norm(obs_i)
                ds = {}
                for key, var in (
                    ("ii", "quad_off_P"),
                    ("ii_raw", "quad_off_P_raw"),
                    ("ii_K", "quad_off_K"),
                ):
                    o = solve_pair(Pq, M, T_hat, c0, kvec, R, var, a)
                    ds[key] = np.linalg.norm(obs_i - o) / norm_i
                o3 = solve_pair(Pq, M, T_hat9, c0, kvec, R, "nine_only", a)
                ds["iii"] = np.linalg.norm(obs_i - o3) / norm_i
                print(
                    f"{wave:>4} {sname:>6} {ka:>4.1f} {ds['ii']:>11.3e} "
                    f"{ds['ii_raw']:>12.3e} {ds['ii_K']:>11.3e} {ds['iii']:>11.3e}"
                )


def main() -> None:
    """Run calibration and decision table."""
    print("T27 inter-voxel coupling study")
    print(f"Background: alpha={REF.alpha}, beta={REF.beta}, rho={REF.rho}")
    print(
        f"Contrast: Dlambda={CONTRAST.Dlambda:.1e}, Dmu={CONTRAST.Dmu:.1e}, "
        f"Drho={CONTRAST.Drho}"
    )
    run_calibration()
    run_propagator_comparison()
    run_decision_table()


if __name__ == "__main__":
    main()
