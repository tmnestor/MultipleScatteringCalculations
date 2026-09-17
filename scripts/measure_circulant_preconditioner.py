"""A BLOCK-CIRCULANT PRECONDITIONER for the strong-scattering regime.

THE PROBLEM IT ADDRESSES. `measure_strong_scattering.py` established that the
sweep solver returns the RIGHT answer past rho = 1 -- agreeing with a dense
direct solve to 9e-10 where the Neumann series diverges outright -- but that the
COST explodes: 16 propagator applications at rho = 0.38 against 11648 at
rho = 1.69, and no convergence at all by rho ~ 79.

⚠ THAT IS A DIFFERENT DIFFICULTY FROM THE ONE THE SWEEP ARCHITECTURE SOLVES.
Widening the evanescent range costs nothing in iterations; raising the
SCATTERING STRENGTH costs almost everything. The sweep removes the first and
does nothing for the second, so the second needs a preconditioner -- an ordinary
Krylov problem, not an architectural one.

════════════════════════════════════════════════════════════════════════════
THE PRECONDITIONER
════════════════════════════════════════════════════════════════════════════

The system is (I - G0 T0) psi = psi_inc. Two facts make a good preconditioner
cheap here:

  1. The lattice coupling G0 is TRANSLATION INVARIANT. It depends on the
     separation between cells, not on where they sit, so laterally it is a
     CONVOLUTION -- diagonalised by the FFT.
  2. The state is only 9 components per site, and the lattice is thin in z.
     So in Fourier space the operator decouples into one small dense block per
     lateral wavenumber, of size (n_z x 9), which can be INVERTED EXACTLY.

Concretely, writing the lateral Fourier transform as a hat,

    [(I - G0 T0) psi]^(z, k) = psi^(z,k) - sum_{z'} Ghat(z - z', k) T0 psi^(z',k),

so for each lateral wavenumber k the operator is a block-Toeplitz matrix in the
depth index, with 9x9 blocks

    A_k[z, z'] = delta_{zz'} I_9  -  Ghat(z - z', k) T0 .

The preconditioner applies A_k^{-1} at every k: FFT the residual laterally,
solve the (n_z*9) x (n_z*9) system at each wavenumber, inverse FFT. With n_z = 2
that is an 18x18 solve per wavenumber -- negligible beside one propagator
application.

WHY IT SHOULD WORK. It inverts EXACTLY the part of the operator that is
translation invariant and uniform. What it leaves for the Krylov solver is only
what departs from that: the deviation of each cell's T0 from the reference one,
and the lateral wrap-around, because the circulant approximation is periodic
while the sweep operator is not.

WHERE IT DEGRADES, stated rather than discovered later:

  * A HETEROGENEOUS medium. The preconditioner is built from ONE representative
    T0, so it inverts the homogenised problem. The stronger the variation
    between cells, the less it removes. In the uniform test below it is
    therefore seen at its BEST, and the improvement quoted is an upper bound on
    what a varying medium would get.
  * A THICK stack. The per-wavenumber block grows as (n_z * 9)^2 to build and
    (n_z * 9)^3 to factor, so the cost stops being negligible for deep models.
  * NEAR A RESONANCE of the preconditioned block, where A_k becomes singular
    for some k and the inverse is ill-conditioned rather than helpful.

⚠ THIS IS NOT NOVEL. Circulant preconditioning of the discrete dipole
approximation is established practice; the contribution here is only that the
lattice kernel this project already builds (`build_slab_kernels`, periodic) IS
the circulant operator, so the preconditioner costs nothing to obtain.

Run:  conda run -n seismic python scripts/measure_circulant_preconditioner.py
Seismic units (km, km/s, g/cm3), time convention e^{-i omega t}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.directional_sweeps import (  # noqa: E402
    SweepGrid3D,
    apply_g0_3d,
    build_g0_cache_3d,
)
from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from cubic_scattering.slab_scattering import SlabGeometry, build_slab_kernels  # noqa: E402
from cubic_scattering.voigt_tmatrix import effective_stiffness_voigt  # noqa: E402

REF = ReferenceMedium(5.0, 3.0, 2.5)
PITCH, N_SIDE, N_Z, FREQ = 0.0375, 6, 2, 6.0
CON = MaterialContrast(2.0, 1.0, 0.1)


def _cell_blocks(scale: float, hetero: float, shape, omega, rng):
    """Per-cell 9x9 T-matrices for a HETEROGENEOUS medium.

    Each cell's contrast is the mean contrast times a positive random factor of
    relative spread `hetero`. hetero = 0 is the uniform medium -- the case in
    which a circulant preconditioner is exact by construction and therefore
    proves nothing.

    Returns (per-cell blocks, the MEAN block the preconditioner is built from).
    """
    a = 0.5 * PITCH
    v = PITCH**3
    n_z, n_x, n_y = shape
    facs = np.maximum(1.0 + hetero * rng.standard_normal((n_z, n_x, n_y)), 0.05)

    blocks = np.zeros((n_z, n_x, n_y, 9, 9), dtype=complex)
    uniq: dict[float, np.ndarray] = {}
    for iz in range(n_z):
        for ix in range(n_x):
            for iy in range(n_y):
                f = round(float(facs[iz, ix, iy]), 4)
                if f not in uniq:
                    con = MaterialContrast(
                        scale * f * CON.Dlambda, scale * f * CON.Dmu, scale * f * CON.Drho
                    )
                    r = compute_cube_tmatrix(float(omega.real), a, REF, con)
                    blk = np.zeros((9, 9), dtype=complex)
                    blk[:3, :3] = float(omega.real) ** 2 * complex(r.Drho_star) * v * np.eye(3)
                    blk[3:, 3:] = v * effective_stiffness_voigt(
                        r.Dlambda_star, r.Dmu_star_diag, r.Dmu_star_off
                    )
                    uniq[f] = blk
                blocks[iz, ix, iy] = uniq[f]
    # The preconditioner gets ONE block: the mean. That is the homogenised
    # problem, and the gap between it and the true per-cell blocks is exactly
    # what the Krylov solver is left to handle.
    return blocks, blocks.reshape(-1, 9, 9).mean(axis=0)


def _setup(scale: float, hetero: float, seed: int = 20260917):
    """(cache, per-cell t0, mean t0, psi_inc, shape, omega)."""
    omega = 2.0 * np.pi * FREQ * (1 + 0.03j)
    grid = SweepGrid3D(n_z=N_Z, n_x=N_SIDE, n_y=N_SIDE, pitch=PITCH)
    cache = build_g0_cache_3d(grid, REF, omega)
    shape = (grid.n_z, grid.n_x, grid.n_y)
    rng = np.random.default_rng(seed)
    t0, t0_mean = _cell_blocks(scale, hetero, shape, omega, rng)

    k_s = float(omega.real) / abs(complex(REF.beta))
    psi = np.zeros((*shape, 9), dtype=complex)
    psi[..., 1] = np.exp(1j * k_s * np.arange(grid.n_z) * PITCH)[:, None, None]
    return cache, t0, t0_mean, psi, shape, omega


def build_preconditioner(t0_ref: np.ndarray, omega: complex, shape):
    """Factor A_k = I - Ghat(z-z', k) T0 at every lateral wavenumber.

    Returns a callable applying A_k^{-1} to a state array. The blocks are
    inverted ONCE, outside the Krylov loop -- that is the whole economy of the
    scheme.
    """
    n_z, n_x, n_y = shape
    geom = SlabGeometry(M=n_x, N_z=n_z, a=0.5 * PITCH)
    # The periodic lattice kernel IS the circulant operator this preconditions
    # with. Built at the M^2 Bloch points, indexed [dz_index, k1, k2, 9, 9] with
    # dz_index = dz_vox + (N_z - 1).
    khat = build_slab_kernels(
        geom,
        float(omega.real),
        REF,
        periodic=True,
        lattice_ewald=True,
        volume_averaged=True,
        n_orders=2,
    )
    size = n_z * 9
    inv = np.zeros((n_x, n_y, size, size), dtype=complex)
    for p in range(n_x):
        for q in range(n_y):
            mat = np.eye(size, dtype=complex)
            for z in range(n_z):
                for zp in range(n_z):
                    g = khat[(z - zp) + (n_z - 1), p, q]
                    mat[9 * z : 9 * z + 9, 9 * zp : 9 * zp + 9] -= g @ t0_ref
            inv[p, q] = np.linalg.inv(mat)

    def apply(vec: np.ndarray) -> np.ndarray:
        arr = vec.reshape((*shape, 9))
        hat = np.fft.fft2(arr, axes=(1, 2))
        flat = hat.transpose(1, 2, 0, 3).reshape(n_x, n_y, size)
        out = np.einsum("pqab,pqb->pqa", inv, flat)
        back = out.reshape(n_x, n_y, n_z, 9).transpose(2, 0, 1, 3)
        return np.fft.ifft2(back, axes=(1, 2)).ravel()

    return apply


def build_near_field_preconditioner(t0: np.ndarray, omega: complex, shape, radius: int = 1):
    """M = I - G0^near T0, with each cell's OWN T0. Factored once, sparse.

    ════════════════════════════════════════════════════════════════════════
    WHY THIS ONE IS RIGHT WHERE THE CIRCULANT ONE IS NOT
    ════════════════════════════════════════════════════════════════════════

    The circulant preconditioner assumes the operator is TRANSLATION INVARIANT
    -- that is what lets the FFT diagonalise it -- so it has to be built from a
    single representative T0 and inverts the HOMOGENISED problem. A genuinely
    varying medium is exactly what that assumption denies, so its useful
    content falls away as the spread grows.

    This one makes no such assumption. It keeps every coupling within `radius`
    cells, each with the ACTUAL T0 of the cells involved, and inverts that
    sparse operator exactly. Heterogeneity is carried, not averaged away.

    The physical argument is that at strong contrast the difficulty is
    dominated by strong coupling between NEARBY cells: the near field is what
    makes the spectrum bad, and the far field is comparatively weak and smooth.
    Inverting the near field exactly therefore removes the part of the spectrum
    that is causing the trouble, and leaves the Krylov solver a well-behaved
    remainder.

    COST. The matrix has (2*radius+1)^3 neighbour blocks of 9x9 per site, so it
    is sparse; it is assembled and factored ONCE, outside the Krylov loop, and
    each application is a sparse triangular solve. Cost grows quickly with
    `radius`, which is the knob trading setup against iteration count.

    ⚠ For a WHOLE-SPACE background the coupling between distinct cells is the
    closed-form propagator, which is what is assembled here. With a LAYERED
    background the plane-to-plane blocks differ and this construction would
    need the layered tables instead -- it is not simply reusable there.
    """
    from cubic_scattering.resonance_tmatrix import _propagator_block_9x9
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    n_z, n_x, n_y = shape
    n_sites = n_z * n_x * n_y
    size = n_sites * 9

    def idx(iz, ix, iy):
        return (iz * n_x + ix) * n_y + iy

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []
    for iz in range(n_z):
        for ix in range(n_x):
            for iy in range(n_y):
                m = idx(iz, ix, iy)
                for b in range(9):  # identity part
                    rows.append(9 * m + b)
                    cols.append(9 * m + b)
                    vals.append(1.0 + 0.0j)
                for dz in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            jz, jx, jy = iz + dz, ix + dx, iy + dy
                            if not (0 <= jz < n_z and 0 <= jx < n_x and 0 <= jy < n_y):
                                continue
                            if dz == 0 and dx == 0 and dy == 0:
                                continue  # self-coupling is inside T0, not G0
                            n = idx(jz, jx, jy)
                            sep = PITCH * np.array([dz, dx, dy], dtype=float)
                            g = _propagator_block_9x9(sep, float(omega.real), REF)
                            blk = -(g @ t0[jz, jx, jy])
                            for b in range(9):
                                for c in range(9):
                                    if blk[b, c] != 0.0:
                                        rows.append(9 * m + b)
                                        cols.append(9 * n + c)
                                        vals.append(complex(blk[b, c]))

    mat = csc_matrix((vals, (rows, cols)), shape=(size, size), dtype=complex)
    lu = splu(mat)
    return lambda vec: lu.solve(vec)


def solve(cache, t0, psi, shape, precond=None, tol=1e-10, max_iter=4000):
    """(n_matvec, residual) for the Foldy-Lax solve, optionally preconditioned."""
    size = int(np.prod(shape)) * 9
    count = {"n": 0}

    def matvec(v):
        count["n"] += 1
        arr = v.reshape((*shape, 9))
        return (arr - apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, arr), cache)).ravel()

    op = LinearOperator((size, size), matvec=matvec, dtype=complex)
    m_op = None if precond is None else LinearOperator((size, size), matvec=precond, dtype=complex)
    b = psi.ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        sol, _ = gmres(op, b, rtol=tol, maxiter=max_iter, M=m_op)
        resid = float(np.linalg.norm(op.matvec(sol) - b) / np.linalg.norm(b))
    return count["n"], resid, sol


def main() -> int:
    print("=" * 78)
    print("BLOCK-CIRCULANT PRECONDITIONER FOR STRONG SCATTERING")
    print("=" * 78)
    print("\n  A_k[z,z'] = delta I_9 - Ghat(z-z', k) T0_MEAN, inverted exactly at")
    print("  every lateral wavenumber.  The FFT diagonalises the translation-")
    print("  invariant part; what is left for GMRES is the departure from it.")
    print()
    print("  ⚠ THE UNIFORM CASE IS NOT A TEST.  With one T0 everywhere the")
    print("  preconditioner inverts the operator exactly by construction (bar")
    print("  wrap-around), so it can only succeed and proves nothing.  It is")
    print("  included ONLY as the hetero = 0 control against which the")
    print("  heterogeneous rows are read.")
    print()
    print("  hetero = relative spread of the per-cell contrast about its mean.")
    print("\n  TWO preconditioners, on the SAME heterogeneous problems:")
    print("    circulant  -- built from the MEAN T0, assumes translation")
    print("                  invariance, inverts the HOMOGENISED problem")
    print("    near-field -- keeps couplings within 1 cell with each cell's OWN")
    print("                  T0, assumes nothing about uniformity")
    hdr = (
        f"\n  {'hetero':>7} {'rho':>8} {'spread':>8} {'plain':>8}"
        f" {'circulant':>9} {'near-fld':>9} {'agree':>10}"
    )
    print(hdr)

    scale = 100.0  # the regime where the unpreconditioned solve is genuinely hard
    for hetero in (0.0, 0.25, 0.5, 1.0, 2.0):
        cache, t0, t0_mean, psi, shape, omega = _setup(scale, hetero)

        rng = np.random.default_rng(20260917)
        vec = rng.normal(size=(*shape, 9)) + 1j * rng.normal(size=(*shape, 9))
        vec /= np.linalg.norm(vec)
        rho = 0.0
        for _ in range(20):
            nxt = apply_g0_3d(np.einsum("zxyab,zxyb->zxya", t0, vec), cache)
            rho = float(np.linalg.norm(nxt))
            if rho == 0.0:
                break
            vec = nxt / rho

        # How far the medium actually departs from the mean the preconditioner
        # was built from -- reported so "hetero" is a measured spread, not a
        # nominal knob.
        flat = t0.reshape(-1, 9, 9)
        spread = float(
            np.mean(np.linalg.norm(flat - t0_mean, axis=(1, 2))) / max(np.linalg.norm(t0_mean), 1e-300)
        )

        n_plain, r_plain, s_plain = solve(cache, t0, psi, shape)
        circ = build_preconditioner(t0_mean, omega, shape)
        n_circ, r_circ, s_circ = solve(cache, t0, psi, shape, precond=circ)
        near = build_near_field_preconditioner(t0, omega, shape, radius=1)
        n_near, r_near, s_near = solve(cache, t0, psi, shape, precond=near)

        ok_plain, ok_circ, ok_near = (r_plain < 1e-8, r_circ < 1e-8, r_near < 1e-8)
        # Agreement is only meaningful where BOTH converged; otherwise it would
        # compare a converged answer against a partial one and read as a defect.
        if ok_plain and ok_near:
            agree = f"{np.linalg.norm(s_near - s_plain) / np.linalg.norm(s_plain):10.2e}"
        elif ok_circ and ok_near:
            agree = f"{np.linalg.norm(s_near - s_circ) / np.linalg.norm(s_circ):10.2e}"
        else:
            agree = f"{'--':>10}"
        p_txt = f"{n_plain:>8}" if ok_plain else f"{'FAIL':>8}"
        c_txt = f"{n_circ:>9}" if ok_circ else f"{'FAIL':>9}"
        nf_txt = f"{n_near:>9}" if ok_near else f"{'FAIL':>9}"
        print(f"  {hetero:7.2f} {rho:8.3f} {spread:8.3f} {p_txt} {c_txt} {nf_txt} {agree}")

    print("\n" + "=" * 78)
    print("WHAT THIS DOES AND DOES NOT SHOW.")
    print()
    print("  READ THE hetero > 0 ROWS.  The hetero = 0 row is a control, not a")
    print("  result: there the preconditioner inverts the operator exactly and")
    print("  could not fail.  The question is how much survives once the medium")
    print("  actually varies, which is the only case that matters in practice.")
    print()
    print("  The preconditioner is built from the MEAN T0, so it inverts the")
    print("  HOMOGENISED problem.  Its useful content is therefore expected to")
    print("  fall as the spread grows, and the rows show by how much.")
    print()
    print("  Cost per application is one lateral FFT pair plus an (n_z*9)-square")
    print("  solve per wavenumber -- negligible against one propagator")
    print("  application here, but growing as (n_z*9)^3 for deep stacks.")
    print()
    print("  ⚠ NOT NOVEL. Circulant preconditioning of the discrete dipole")
    print("  approximation is established practice.  What is cheap here is only")
    print("  that this project already builds the circulant operator.")
    print()
    print("  ⚠ SCOPE: one realisation per spread, n_z = 2, a single frequency.")
    print("  A spread is a random draw, so a single realisation is an anecdote;")
    print("  several seeds would be needed before quoting a speedup as typical.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
