#!/usr/bin/env python3
"""Rederivation of the face-separation S-block constants — measured verdict.

Provenance
----------
Written 2026-06-13 to arbitrate the face-contact S-block defect reported by
scripts/t27_coupling_study.py (step 2b: "S[0,0] ~3x low, shear-shear sign
flip" vs its FD volume-averaged point-propagator arbiter).

VERDICT (measured): the committed constants in
cubic_scattering/inter_voxel_propagator.py (FACE_A11/A22, FACE_B1111/1122/
2222/2233) are CORRECT for the module's own definition

    P_ijkl(R) = intint_{A x B} d2 G_ik(s - s' + R) / dx_j dx_l  ds ds'
    (docs/inter_voxel_propagator.tex eq. Pijkl-def)

to 13+ digits.  The study's face "truth" was a quadrature artifact.

Three independent bias-free routes agree:

1. Delta-collapse (route B, `collapse_constants`): the defining tent-
   correlation integrals, reduced exactly to smooth 2D/1D integrals by
   collapsing tent derivatives (T''(d) = delta(d+1) - 2 delta(d) +
   delta(d-1); fourth same-axis derivative inserts delta'' acting as the
   kernel second derivative).  Evaluated with scipy adaptive quadrature:
   reproduces every committed constant to ~1e-16 — i.e. the committed
   table IS the exact evaluation of the defining integrals (same values
   as Mathematica/InterVoxelPropagator.wl).

2. Subdivision fixed point (route C, `subdivision_tables`): split each
   unit cube into 8 half-cubes; static homogeneity d2G(s x) = s^-3 d2G(x)
   gives the EXACT identity  table(n) = (1/8) sum_m mult(m) table(m)
   over the 64 half-cube pair separations m.  Touching child pairs refer
   back to the face/edge/corner tables themselves (contraction factors
   1/2, 1/4, 1/8); all other children are strictly separated smooth Gauss
   integrals (machine precision).  The unique fixed point reproduces the
   committed face (AND edge AND corner) tables to ~1e-13.  No singular
   quadrature anywhere.

3. Dyadic-shell 3D quadrature (route D, `dyadic_contact_table`): the 6D
   double average reduces exactly to the 3D correlation integral
   int T(w1-1) T(w2) T(w3) d2G(w) d^3w with an absolutely convergent
   point singularity at the support corner w=0 (the tent weight vanishes
   linearly there).  L-inf dyadic shells about w=0, kink-split tensor
   Gauss per box: agrees with route C to ~1e-8 (NG=12).

The artifact (route E, `artifact_demonstration`): tensor-product
double-cube Gauss quadrature of the contact integral shares one lateral
node set between the two cubes, so it samples the singular ray
w_perp = 0 exactly, with cumulative spurious weight that does NOT vanish
under n-refinement for the 1/w^3 S kernel (scaling: kernel ~ n^6 at the
closest diagonal pairs x weights n^-4 (axis) x n^-2 (lateral diagonal)
= O(1)).  Measured: S00 "converges" 8.45e-12 (n=4) -> 1.023e-11 (n=16)
with ~1/n^2 steps toward a BIASED limit ~1.05e-11 vs true 3.0089e-12,
and the shear-shear entry to -2.35e-12 vs true +6.515e-13 (the reported
"sign flip").  The FD-of-<G> arbiter at h=0.005, n=8-10 sits in the
invalid regime h <~ 1/n^2 (closest node-pair gap) and reproduces the same
biased values (FD2 drifts 9.2e-12 -> 11.1e-12 as n grows at fixed h) —
hence the study's spurious "FD/direct cross-agreement".  The weight
vanishes quadratically (edge) and cubically (corner) at w=0, and the
G/C/H kernels are only 1/w .. 1/w^2, so every other block and separation
is bias-free — exactly matching the study's pattern (G exact, C ~5%,
edge/corner S at 5.6e-5, ONLY face S "broken").

Dynamic audit (face DYN tables, orders 1-3, `dynamic_audit`): the dynamic
orders are audited against the bias-free Delta-P arbiter
< d2[G(omega) - G(0)] > (the subtracted kernel is O(omega^2/w): no contact
bias; 6D Gauss n=8 vs n=10 drift ~6e-4).  Measured max |Re dev| / scale of
the module's Voigt Delta-S (n=10 arbiter; n=8 in parentheses):

    ka=0.3:  orders<=1: 1.97e-01   <=2: 1.23e-02   <=3: 9.7e-04 (1.6e-03)
    ka=0.5:  orders<=1: 6.74e-01   <=2: 1.17e-01   <=3: 8.5e-03 (8.3e-03)

Pure geometric series truncation (each order gains ~(ka)^2 with the
expected coefficient); the face dynamic tables are SOUND — no defect.

Run:
    conda run -n seismic python scripts/face_s_rederivation.py        # fast
    conda run -n seismic python scripts/face_s_rederivation.py --full # + B/E
    conda run -n seismic python scripts/face_s_rederivation.py --dynamic
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cubic_scattering.inter_voxel_propagator import (  # noqa: E402
    FACE_A11,
    FACE_A22,
    FACE_B1111,
    FACE_B1122,
    FACE_B2222,
    FACE_B2233,
    _get_oh_perm,
    _rotate_tensor4,
    corner_propagator,
    edge_propagator,
    face_propagator,
)

# Reference medium (study parameters)
ALPHA, BETA, RHO = 5000.0, 3000.0, 2500.0
MU = RHO * BETA**2
NU = (ALPHA**2 - 2 * BETA**2) / (2 * (ALPHA**2 - BETA**2))
ETA = 1.0 / (2.0 * (1.0 - NU))

COMMITTED = {
    "A11": FACE_A11,
    "A22": FACE_A22,
    "B1111": FACE_B1111,
    "B1122": FACE_B1122,
    "B2222": FACE_B2222,
    "B2233": FACE_B2233,
}


# ── Static second-derivative kernel (Kelvin) ─────────────────────────────────


def d2G_static(w: np.ndarray) -> np.ndarray:
    """Analytic d2 G_ij / dw_k dw_l of the static Kelvin tensor.

    G_ij = (1/(4 pi mu)) delta_ij / r - (1/(16 pi mu (1-nu))) d2r/dwi dwj
    d2(1/r)/dwk dwl = (3 nk nl - delta_kl)/r^3
    d4r/(di dj dk dl) = [-(dd, 3 perms) + 3 (d nn, 6 perms) - 15 nnnn]/r^3

    Args:
        w: Separation vectors, shape (..., 3).

    Returns:
        Shape (..., 3, 3, 3, 3), index order [i, j, k, l] = d2 G_ij/dwk dwl.
    """
    r = np.linalg.norm(w, axis=-1)
    n = w / r[..., None]
    d = np.eye(3)
    r3 = r**3
    t1 = (3.0 * n[..., :, None] * n[..., None, :] - d) / r3[..., None, None]
    nn = n[..., :, None] * n[..., None, :]
    dd_term = (
        np.einsum("ij,kl->ijkl", d, d)
        + np.einsum("ik,jl->ijkl", d, d)
        + np.einsum("il,jk->ijkl", d, d)
    )
    d_nn = (
        np.einsum("ij,...kl->...ijkl", d, nn)
        + np.einsum("ik,...jl->...ijkl", d, nn)
        + np.einsum("il,...jk->...ijkl", d, nn)
        + np.einsum("jk,...il->...ijkl", d, nn)
        + np.einsum("jl,...ik->...ijkl", d, nn)
        + np.einsum("kl,...ij->...ijkl", d, nn)
    )
    nnnn = np.einsum("...ij,...kl->...ijkl", nn, nn)
    d4r = (-dd_term + 3.0 * d_nn - 15.0 * nnnn) / r3[..., None, None, None, None]
    out = np.einsum("ij,...kl->...ijkl", d, t1) / (4.0 * np.pi * MU)
    out -= d4r / (16.0 * np.pi * MU * (1.0 - NU))
    return out


# ── Route B: delta-collapse defining integrals (scipy) ───────────────────────


def collapse_constants() -> dict[str, float]:
    """Face A/B constants from the defining tent-collapse integrals.

    Unit pitch, R = (1,0,0); tent T(d) = max(0, 1-|d|):
      A_jl   = -1/(4 pi) d2/dRj dRl  int T3(w - R) / |w| d^3w
      B_ijkl = -1/(8 pi) d4/dR^4     int T3(w - R) |w|   d^3w
    Same-axis fourth derivatives insert delta'' -> kernel second derivative
    (d2|w|/dw1^2 = (w2^2 + w3^2)/rho^3) at the collapse points.
    """
    from scipy import integrate

    def dbl(f, x0, x1, y0, y1):
        return integrate.dblquad(f, x0, x1, y0, y1, epsabs=1e-13, epsrel=1e-13)[0]

    def one(f, x0, x1):
        return integrate.quad(f, x0, x1, epsabs=1e-14, epsrel=1e-14, limit=400)[0]

    def Ia(c):  # A11 2D master, [0,1]^2 fold x4
        return 4 * dbl(
            lambda v, u: (1 - u) * (1 - v) / np.sqrt(c * c + u * u + v * v),
            0,
            1,
            0,
            1,
        )

    def Ja(c):  # A22 2D master, w1 in [0,2] shifted tent, w3 fold x2
        return 2 * dbl(
            lambda v, u: (1 - abs(u - 1)) * (1 - v) / np.sqrt(u * u + c * c + v * v),
            0,
            2,
            0,
            1,
        )

    def Kb(c):  # B1111 kernel (w2^2+w3^2)/rho^3
        return 4 * dbl(
            lambda v, u: (
                (1 - u) * (1 - v) * (u * u + v * v) / (c * c + u * u + v * v) ** 1.5
            ),
            0,
            1,
            0,
            1,
        )

    def Mb(c):  # B2222 kernel (w1^2+w3^2)/rho^3, shifted tent in w1
        return 2 * dbl(
            lambda v, u: (
                (1 - abs(u - 1))
                * (1 - v)
                * (u * u + v * v)
                / (c * c + u * u + v * v) ** 1.5
            ),
            0,
            2,
            0,
            1,
        )

    def Lb(C):  # B1122 1D master
        return 2 * one(lambda t: (1 - t) * np.sqrt(C + t * t), 0, 1)

    def Nb(C):  # B2233 1D master, shifted tent
        return one(lambda u: (1 - abs(u - 1)) * np.sqrt(C + u * u), 0, 2)

    w3 = {0: 1.0, 1: -2.0, 2: 1.0}
    wpm = {-1: 1.0, 0: -2.0, 1: 1.0}
    pi4, pi8 = 4 * np.pi, 8 * np.pi
    return {
        "A11": -sum(w3[c] * Ia(c) for c in (0, 1, 2)) / pi4,
        "A22": -sum(wpm[c] * Ja(abs(c)) for c in (-1, 0, 1)) / pi4,
        "B1111": -sum(w3[c] * Kb(c) for c in (0, 1, 2)) / pi8,
        "B2222": -sum(wpm[c] * Mb(abs(c)) for c in (-1, 0, 1)) / pi8,
        "B1122": -sum(
            w3[a] * wpm[b] * Lb(a * a + b * b) for a in (0, 1, 2) for b in (-1, 0, 1)
        )
        / pi8,
        "B2233": -sum(
            wpm[b] * wpm[c] * Nb(b * b + c * c) for b in (-1, 0, 1) for c in (-1, 0, 1)
        )
        / pi8,
    }


# ── Route C: subdivision fixed point ─────────────────────────────────────────

AXIS_PAR = [(1, 1.0), (2, 2.0), (3, 1.0)]  # parent component 1
AXIS_TRA = [(-1, 1.0), (0, 2.0), (1, 1.0)]  # parent component 0


def smooth_table(n_vec: tuple[int, int, int], ng: int) -> np.ndarray:
    """Raw P table for strictly separated unit cubes at lattice sep n_vec."""
    x, w = leggauss(ng)
    x = x * 0.5
    w = w * 0.5
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    wts = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    R = np.array(n_vec, dtype=float)
    acc = np.zeros((3, 3, 3, 3))
    chunk = 128
    for s in range(0, pts.shape[0], chunk):
        sl = slice(s, min(s + chunk, pts.shape[0]))
        sep = R[None, None, :] + pts[sl, None, :] - pts[None, :, :]
        acc += np.einsum("m,q,mqijkl->ijkl", wts[sl], wts, d2G_static(sep))
    return acc


def _child_offsets(parent: tuple[int, int, int]):
    axes = [AXIS_PAR if c == 1 else AXIS_TRA for c in parent]
    return [
        ((a0, a1, a2), m0 * m1 * m2)
        for a0, m0 in axes[0]
        for a1, m1 in axes[1]
        for a2, m2 in axes[2]
    ]


def _is_touching(n) -> bool:
    return max(abs(c) for c in n) == 1


def _classify(n) -> str:
    key = tuple(sorted(abs(c) for c in n))
    return {(0, 0, 1): "face", (0, 1, 1): "edge", (1, 1, 1): "corner"}[key]


def subdivision_tables(ng: int = 10, tol: float = 1e-13) -> dict[str, np.ndarray]:
    """Touching-pair tables (Gdd index order d2<G>_ij/dRk dRl) by fixed point.

    Args:
        ng: Gauss points per axis for the smooth separated-pair integrals.
        tol: Fixed-point convergence tolerance (relative).

    Returns:
        {"face": F, "edge": E, "corner": C} canonical-direction tensors.
    """
    canon = {"face": (1, 0, 0), "edge": (1, 1, 0), "corner": (1, 1, 1)}
    children = {k: _child_offsets(v) for k, v in canon.items()}
    needed = sorted(
        {n for offs in children.values() for n, _ in offs if not _is_touching(n)}
    )
    smooth = {n: smooth_table(n, ng) for n in needed}

    cur = {
        "face": face_propagator(MU, NU),
        "edge": edge_propagator(MU, NU),
        "corner": corner_propagator(MU, NU),
    }

    def table_of(n, tabs):
        if not _is_touching(n):
            return smooth[n]
        return _rotate_tensor4(tabs[_classify(n)], _get_oh_perm(n))

    for _ in range(120):
        new: dict[str, np.ndarray] = {}
        for kind in ("corner", "edge", "face"):
            rhs = np.zeros((3, 3, 3, 3))
            for n, mult in children[kind]:
                rhs += mult * table_of(n, {**cur, **new})
            new[kind] = rhs / 8.0
        delta = max(
            np.max(np.abs(new[k] - cur[k])) / np.max(np.abs(new[k])) for k in new
        )
        cur = new
        if delta < tol:
            break
    return cur


def constants_from_gdd(F: np.ndarray) -> dict[str, float]:
    """Invert a face Gdd tensor (d2<G>_ij/dRk dRl) to the A/B constants."""
    B1122 = MU * F[0, 1, 1, 0] / ETA
    A11 = -(MU * F[1, 1, 0, 0]) + ETA * B1122
    A22 = -(MU * F[0, 0, 1, 1]) + ETA * B1122
    return {
        "A11": A11,
        "A22": A22,
        "B1111": (A11 + MU * F[0, 0, 0, 0]) / ETA,
        "B1122": B1122,
        "B2222": (A22 + MU * F[1, 1, 1, 1]) / ETA,
        "B2233": MU * F[1, 2, 1, 2] / ETA,
    }


# ── Route D: dyadic-shell 3D correlation quadrature ──────────────────────────


def dyadic_contact_table(ng: int = 12) -> np.ndarray:
    """Face contact table via the 3D correlation integral, dyadic shells.

    int_{[0,2]x[-1,1]^2} T(w1-1) T(w2) T(w3) d2G(w) d^3w with L-inf dyadic
    shells about the (absolutely convergent) corner singularity at w=0.
    Boxes are split at the tent-kink planes w2=0, w3=0.
    """
    xg, wg = leggauss(ng)

    def gauss_box(lo, hi):
        xs = [0.5 * (h + l) + 0.5 * (h - l) * xg for l, h in zip(lo, hi, strict=True)]
        ws = [0.5 * (h - l) * wg for l, h in zip(lo, hi, strict=True)]
        W1, W2, W3 = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        WT = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
        pts = np.stack([W1, W2, W3], axis=-1).reshape(-1, 3)
        tent = (
            np.maximum(0.0, 1.0 - np.abs(pts[:, 0] - 1.0))
            * np.maximum(0.0, 1.0 - np.abs(pts[:, 1]))
            * np.maximum(0.0, 1.0 - np.abs(pts[:, 2]))
        )
        vals = tent[:, None, None, None, None] * d2G_static(pts)
        return np.einsum("n,n...->...", WT.reshape(-1), vals)

    def split0(lo, hi):
        return [(lo, 0.0), (0.0, hi)] if lo < 0.0 < hi else [(lo, hi)]

    def shell_boxes(s):
        h = s / 2.0
        raw = [((h, -s, -s), (s, s, s))]
        for lo2, hi2 in ((-s, -h), (-h, h), (h, s)):
            for lo3, hi3 in ((-s, -h), (-h, h), (h, s)):
                if (lo2, hi2) == (-h, h) and (lo3, hi3) == (-h, h):
                    continue
                raw.append(((0.0, lo2, lo3), (h, hi2, hi3)))
        out = []
        for lo, hi in raw:
            for l2, h2 in split0(lo[1], hi[1]):
                for l3, h3 in split0(lo[2], hi[2]):
                    out.append(((lo[0], l2, l3), (hi[0], h2, h3)))
        return out

    total = np.zeros((3, 3, 3, 3))
    for l2, h2 in ((-1.0, 0.0), (0.0, 1.0)):
        for l3, h3 in ((-1.0, 0.0), (0.0, 1.0)):
            total += gauss_box((1.0, l2, l3), (2.0, h2, h3))
    s = 1.0
    for _ in range(60):
        shell = np.zeros((3, 3, 3, 3))
        for lo, hi in shell_boxes(s):
            shell += gauss_box(lo, hi)
        total += shell
        if np.max(np.abs(shell)) < 1e-19:
            break
        s /= 2.0
    return total


# ── Route E: artifact demonstration (the study's quadrature at contact) ─────


def biased_tensor_gauss(n: int) -> float:
    """The study-style 6D tensor-Gauss S00 value at exact face contact.

    Demonstrates the diagonal bias: both cubes share one lateral node set,
    so w_perp = 0 (the singular ray) is sampled exactly with O(1)
    cumulative weight for the 1/w^3 kernel.  Does NOT converge to the
    true integral under n-refinement.
    """
    x, w = leggauss(n)
    x = x * 0.5
    w = w * 0.5
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    wts = (w[:, None, None] * w[None, :, None] * w[None, None, :]).reshape(-1)
    R = np.array([1.0, 0.0, 0.0])
    acc = 0.0
    chunk = 128
    for s in range(0, pts.shape[0], chunk):
        sl = slice(s, min(s + chunk, pts.shape[0]))
        sep = R[None, None, :] + pts[sl, None, :] - pts[None, :, :]
        block = d2G_static(sep)[..., 0, 0, 0, 0]
        acc += np.einsum("m,q,mq->", wts[sl], wts, block)
    return acc / 1.0  # V^2 = 1 at unit pitch


# ── Dynamic audit: face DYN tables (orders 1-3) vs bias-free ΔP arbiter ─────


def _greens_dynamic(rvecs: np.ndarray, omega: float) -> np.ndarray:
    """Vectorised Kupradze full-space Green's tensor (study conventions)."""
    r = np.linalg.norm(rvecs, axis=-1)
    g = rvecs / r[..., None]
    kP = omega / ALPHA
    kS = omega / BETA
    eP = np.exp(1j * kP * r)
    eS = np.exp(1j * kS * r)
    nfP = (1.0 - 1j * kP * r) * eP / r**3
    nfS = (1.0 - 1j * kS * r) * eS / r**3
    phi = kS**2 * eS / r - nfS + nfP
    psi = 3.0 * nfS - 3.0 * nfP + kP**2 * eP / r - kS**2 * eS / r
    pref = 1.0 / (4.0 * np.pi * RHO * omega**2)
    return pref * (
        phi[..., None, None] * np.eye(3)
        + psi[..., None, None] * g[..., :, None] * g[..., None, :]
    )


def _G_kelvin(w: np.ndarray) -> np.ndarray:
    """Static Kelvin Green's tensor, vectorised."""
    r = np.linalg.norm(w, axis=-1)
    n = w / r[..., None]
    d = np.eye(3)
    pref = 1.0 / (16.0 * np.pi * MU * (1.0 - NU))
    return pref * (
        (3.0 - 4.0 * NU) * d / r[..., None, None]
        + n[..., :, None] * n[..., None, :] / r[..., None, None]
    )


def dynamic_audit(ka_values: tuple[float, ...] = (0.3, 0.5), n: int = 8) -> None:
    """Audit face DYN S tables (orders 1-3) against the ΔP arbiter.

    ΔP(ω) = <d2[G(ω) - G(0)]> at face contact: the subtracted kernel is
    O(ω²/w), so the contact diagonal bias of the static 1/w³ kernel is
    absent and 6D Gauss converges (n=8 vs 10 drift ~6e-4 measured).
    Second derivatives by central FD (h=1e-4) on the SUBTRACTED kernel.
    """
    from cubic_scattering.inter_voxel_propagator import (
        _P_to_voigt_S,
        dynamic_inter_voxel_propagator,
        inter_voxel_propagator,
    )
    from cubic_scattering.resonance_tmatrix import _voigt_contract

    a = 0.5
    x, wgt = leggauss(n)
    x = x * a
    wgt = wgt * a
    pts = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    wts = (wgt[:, None, None] * wgt[None, :, None] * wgt[None, None, :]).reshape(-1)
    V = (2.0 * a) ** 3
    R = np.array([1.0, 0.0, 0.0])
    h = 1e-4
    e = np.eye(3)

    for ka in ka_values:
        omega = ka * BETA / a

        def dG(w, om=omega):
            return _greens_dynamic(w, om) - _G_kelvin(w)

        acc = np.zeros((3, 3, 3, 3), dtype=complex)
        chunk = 32
        for s in range(0, pts.shape[0], chunk):
            sl = slice(s, min(s + chunk, pts.shape[0]))
            sep = R[None, None, :] + pts[sl, None, :] - pts[None, :, :]
            blk = np.zeros(sep.shape[:-1] + (3, 3, 3, 3), dtype=complex)
            G0 = dG(sep)
            for k in range(3):
                blk[..., k, k] = (
                    dG(sep + h * e[k]) - 2 * G0 + dG(sep - h * e[k])
                ) / h**2
            for k in range(3):
                for ll in range(k + 1, 3):
                    blk[..., k, ll] = blk[..., ll, k] = (
                        dG(sep + h * e[k] + h * e[ll])
                        - dG(sep + h * e[k] - h * e[ll])
                        - dG(sep - h * e[k] + h * e[ll])
                        + dG(sep - h * e[k] - h * e[ll])
                    ) / (4 * h**2)
            acc += np.einsum("m,q,mq...->...", wts[sl], wts, blk)
        Gdd = acc / V**2
        _, _, dS_ref = _voigt_contract(np.zeros((3, 3, 3), dtype=complex), Gdd)

        S0 = _P_to_voigt_S(inter_voxel_propagator((1, 0, 0), MU, NU))
        print(f"\nDynamic audit, face, ka={ka} (omega={omega:.0f} rad/s):")
        for n_ord in (1, 2, 3):
            Pw = dynamic_inter_voxel_propagator(
                (1, 0, 0), ALPHA, BETA, RHO, omega, n_orders=n_ord
            )
            dS_mod = _P_to_voigt_S(Pw) - S0
            dev = np.max(np.abs(np.real(dS_mod) - np.real(dS_ref))) / np.max(
                np.abs(np.real(dS_ref))
            )
            print(f"  module orders<={n_ord}: max |Re dev| / scale = {dev:.3e}")


# ── Main report ──────────────────────────────────────────────────────────────


def main() -> None:
    full = "--full" in sys.argv

    print("=" * 76)
    print("FACE-SEPARATION S-BLOCK REDERIVATION — measured truth table")
    print("=" * 76)

    print("\nRoute C: subdivision fixed point (ng=10, no singular quadrature)")
    tabs = subdivision_tables(ng=10)
    sub_const = constants_from_gdd(tabs["face"])

    print("Route D: dyadic-shell 3D correlation quadrature (NG=12)")
    dy = dyadic_contact_table(ng=12)
    dy_const = constants_from_gdd(dy)

    rows = {"routeC (subdivision)": sub_const, "routeD (dyadic)": dy_const}
    if full:
        print("Route B: delta-collapse defining integrals (scipy)")
        rows["routeB (collapse)"] = collapse_constants()

    print(f"\n{'const':>7} {'committed':>22}", end="")
    for name in rows:
        print(f" {name:>22}", end="")
    print(" " + "rel devs")
    for k, v in COMMITTED.items():
        line = f"{k:>7} {v:>22.15f}"
        devs = []
        for vals in rows.values():
            line += f" {vals[k]:>22.15f}"
            devs.append(f"{abs(vals[k] - v) / abs(v):.1e}")
        print(line + "  " + "/".join(devs))

    print("\nIdentities (committed constants):")
    print(f"  Tr A = A11 + 2 A22           = {FACE_A11 + 2 * FACE_A22:+.3e}")
    print(
        f"  B1111 + 2 B1122 - A11        = "
        f"{FACE_B1111 + 2 * FACE_B1122 - FACE_A11:+.3e}"
    )
    print(
        f"  B1122 + B2222 + B2233 - A22  = "
        f"{FACE_B1122 + FACE_B2222 + FACE_B2233 - FACE_A22:+.3e}"
    )

    print("\nEdge/corner cross-validation (subdivision vs module tables):")
    for kind, mod in (
        ("edge", edge_propagator(MU, NU)),
        ("corner", corner_propagator(MU, NU)),
    ):
        # Map module P (interleaved) to Gdd ordering via entry identities is
        # unnecessary here: compare the module-implied Gdd entries instead.
        # The subdivision tensor IS Gdd-ordered; the module P tensor stores
        # P_ijkl = sym d2<G>_ik/dRj dRl, so Gdd[i,j,k,l] = P-entries with
        # i<->G-row, j<->G-col: Gdd[i,j,k,l] = module P[i,k,j,l] for the
        # kl-symmetrised part.  Spot-check the all-equal-index entries.
        dev = max(
            abs(tabs[kind][i, i, i, i] - mod[i, i, i, i]) / np.max(np.abs(mod))
            for i in range(3)
        )
        print(f"  {kind}: max diag-entry dev = {dev:.2e}")

    if full:
        print("\nRoute E: the artifact — study-style 6D tensor-Gauss at contact")
        print(
            f"  (true S00-entry value = Gdd[0,0,0,0] = {tabs['face'][0, 0, 0, 0]:+.6e})"
        )
        prev = None
        for n in (4, 6, 8, 10, 12):
            v = biased_tensor_gauss(n)
            step = "" if prev is None else f"  step={v - prev:+.2e}"
            prev = v
            print(f"  n={n:>2}: tensor-Gauss S00 = {v:+.6e}{step}")
        print("  -> drifts to a biased ~1.05e-11, never to the true value;")
        print("     this (and the FD arbiter in its h <~ 1/n^2 regime) is the")
        print("     origin of the study's spurious face-S defect report.")

    if "--dynamic" in sys.argv:
        dynamic_audit()


if __name__ == "__main__":
    main()
