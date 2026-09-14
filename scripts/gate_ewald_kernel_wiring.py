"""GATE: the wired Ewald kernel, including the dz != 0 branch.

WHY THIS EXISTS SEPARATELY FROM THE INVARIANCE GATE. `gate_lateral_sum_invariance`
runs at N_z = 1, so it only ever exercises dz = 0. That is the hard case
mathematically, but it is not the whole kernel: a multi-plane slab needs dz != 0
too, and that branch differs in a way no same-plane test can see -- R = 0 is
INCLUDED there (two cubes in different planes at the same lateral position are
distinct scatterers) and EXCLUDED at dz = 0. Getting that backwards drops or
double-counts the single largest term in the inter-plane kernel, and the
invariance gate would still pass.

WHAT IS CHECKED:
  [E1] dz != 0 against an independent exact route. `gate_interplane_bloch_sum`
       already established that for dz != 0 the Bloch sum equals the reciprocal
       sum of `sweep_kernels.vertical_kernel_9x9`, exponentially convergent and
       exact. That construction shares NO code with the Ewald path -- different
       module, different representation, spectral rather than real-space -- so
       it is a genuine arbiter for the branch the invariance gate cannot reach.
  [E2] eta-independence of the assembled kernel, BY BLOCK, at dz = 0 and
       dz != 0. The splitting parameter is bookkeeping; the kernel cannot depend
       on it.

       THE S BLOCK IS LOOSER THAN THE REST, FOR A MEASURED REASON. At this
       frequency kappa_P d = 0.006, and the reciprocal half's out-of-plane
       derivatives carry powers of eta while the answer carries powers of k_z.
       For the low-|G| orders k_z ~ kappa << eta, so the fourth-order strain
       term cancels by about (eta/kappa)^4 ~ 1e9 and loses the corresponding
       digits. This is a property of Ewald at small kappa d, not a defect in the
       sum: measured separately, the SCALAR lattice tensors are eta-independent
       to ~1e-14 at every order and every frequency, and the assembled kernel's
       eta-dependence falls as (kappa d)^2 exactly as the cancellation ratio
       does (3e-4 at omega = 60, 7e-11 at omega = 6000).
  [E3] cutoff convergence. Raising the Ewald cutoff must not move the answer.
       eta-independence alone can be satisfied by two equally truncated sums.
  [E4] the R = 0 convention, made visible. It is the one asymmetry between the
       two branches, so it is stated as a number rather than left implicit.
  [E5] THE TEST THAT DECIDES WHETHER [E2]'s S BLOCK MATTERS: end-to-end
       stability of the computed reflection coefficient under eta. A kernel
       block being noisy is only a problem if the noise reaches the observable.
       Varying eta over a 5x range must not move the answer. This is the check
       that earns the loose S-block tolerance above -- without it, that
       tolerance would just be a number chosen to make the gate pass.

Run:  conda run -n seismic python scripts/gate_ewald_kernel_wiring.py
SI units (m, m/s, kg/m3, Pa).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import (  # noqa: E402
    MaterialContrast,
    ReferenceMedium,
)
from cubic_scattering.lattice_kupradze import (  # noqa: E402
    bloch_block_ewald_9x9,
    bloch_kernel_hat_9x9,
)
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    build_slab_kernels,
    compute_slab_scattering,
    compute_slab_tmatrices,
    kennett_reference_rpp,
    slab_rpp_periodic,
)
from cubic_scattering.sweep_kernels import vertical_kernel_9x9  # noqa: E402

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
OMEGA = 60.0
D_PITCH = 0.5
M_CELLS = 4
CONTRAST = MaterialContrast(2.0e9, 1.0e9, 100.0)


def _floor(eta: float, cutoff: int) -> float:
    """The one-plane reflection error vs exact Kennett, through the wired path."""
    geom = SlabGeometry(M=M_CELLS, N_z=1, a=0.5 * D_PITCH)
    ones = np.ones((1, M_CELLS, M_CELLS))
    mat = SlabMaterial(
        Dlambda=CONTRAST.Dlambda * ones,
        Dmu=CONTRAST.Dmu * ones,
        Drho=CONTRAST.Drho * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, mat, OMEGA)
    kh = build_slab_kernels(
        geom,
        OMEGA,
        REF,
        periodic=True,
        lattice_ewald=True,
        ewald_eta=eta,
        ewald_cutoff=cutoff,
    )
    res = compute_slab_scattering(
        geom,
        mat,
        OMEGA,
        np.array([1.0, 0.0, 0.0]),
        "P",
        periodic=True,
        gmres_tol=1e-12,
        kernel_hat=kh,
    )
    exact = kennett_reference_rpp(REF, CONTRAST, geom.d, OMEGA)
    return float(abs(slab_rpp_periodic(res, t0) - exact) / abs(exact))


def _reciprocal_reference(k_par: np.ndarray, dz: float, n_g: int) -> np.ndarray:
    """(1/d^2) sum_G Ghat(k_par + G, dz) -- the independently gated route.

    Exact and exponentially convergent for dz != 0, per gate_interplane_bloch_sum.
    """
    b = 2.0 * np.pi / D_PITCH
    acc = np.zeros((9, 9), dtype=complex)
    for m in range(-n_g, n_g + 1):
        for n in range(-n_g, n_g + 1):
            kx = k_par[0] + b * m
            ky = k_par[1] + b * n
            acc += vertical_kernel_9x9(np.array([kx]), ky, dz, OMEGA, REF)[:, :, 0]
    return acc / D_PITCH**2


def main() -> int:
    print("=" * 88)
    print("GATE -- the wired Ewald kernel, dz = 0 and dz != 0")
    print(f"  M = {M_CELLS}, pitch = {D_PITCH} m, omega = {OMEGA} rad/s")
    print("=" * 88)

    blocks = (
        ("G", slice(0, 3), slice(0, 3)),
        ("C", slice(0, 3), slice(3, 9)),
        ("H", slice(3, 9), slice(0, 3)),
        ("S", slice(3, 9), slice(3, 9)),
    )

    def _by_block(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        """Relative difference within each block, not against the kernel max.

        Normalising by the global max would hide the S block entirely: it is
        ~60x smaller than G here, so a 3e-4 error in S reads as 5e-6 overall.
        """
        out = {}
        for name, rows, cols in blocks:
            sub_a = a[..., rows, cols]
            out[name] = float(np.abs(sub_a - b[..., rows, cols]).max() / np.abs(sub_a).max())
        return out

    # ---- [E1] the two independent routes against each other -----------------
    print("\n  [E1] dz != 0: the EWALD route vs the SPECTRAL route (what is used)")
    print("       Two constructions sharing no code -- real-space Ewald with")
    print("       Kupradze derivatives, against a reciprocal sum of the spectral")
    print("       9x9. Compared away from k_par = 0, where both are well")
    print("       conditioned; AT k_par = 0 the Ewald route loses (eta/kappa)^n")
    print("       and is shown separately as the reason it is not used here.")
    print(f"       {'dz (m)':>7} {'Bloch pt':>10} {'G':>10} {'C':>10} {'H':>10} {'S':>10}")
    e1_main, e1_s = 0.0, 0.0
    gamma_s = 0.0
    for dz in (D_PITCH, 2.0 * D_PITCH):
        ker = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF)
        for n1, n2 in ((0, 0), (1, 0), (2, 3)):
            k_par = 2.0 * np.pi * np.array([n1, n2], dtype=float) / (M_CELLS * D_PITCH)
            ewald = bloch_block_ewald_9x9(k_par, dz, D_PITCH, OMEGA, REF)
            per = _by_block(ker[n1, n2], ewald)
            if (n1, n2) == (0, 0):
                gamma_s = max(gamma_s, per["S"])
            else:
                e1_main = max(e1_main, per["G"], per["C"], per["H"])
                e1_s = max(e1_s, per["S"])
            vals = " ".join(f"{per[n]:10.2e}" for n, _, _ in blocks)
            print(f"       {dz:7.2f} {f'({n1},{n2})':>10} {vals}")
    print(f"       At k_par = 0 the Ewald S block departs by {gamma_s:.1e} and grows with")
    print("       dz -- the direction a refinement ladder runs. Hence the split.")

    # The route actually used must still match the gated reference exactly.
    print("\n  [E1b] the spectral route vs gate_interplane_bloch_sum's own reference")
    e1b = 0.0
    for dz in (D_PITCH, 2.0 * D_PITCH):
        ker = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF)
        for n1, n2 in ((0, 0), (2, 3)):
            k_par = 2.0 * np.pi * np.array([n1, n2], dtype=float) / (M_CELLS * D_PITCH)
            ref_val = _reciprocal_reference(k_par, dz, n_g=6)
            rel = float(np.abs(ker[n1, n2] - ref_val).max() / np.abs(ref_val).max())
            e1b = max(e1b, rel)
            print(f"       dz = {dz:4.2f}  ({n1},{n2}):  {rel:.2e}")

    # ---- [E2] eta-independence of the assembled kernel ----------------------
    print("\n  [E2] eta-independence by block")
    eta0 = float(np.sqrt(np.pi) / D_PITCH)
    print(f"       {'dz (m)':>7} {'':>10} {'G':>10} {'C':>10} {'H':>10} {'S':>10}")
    e2_main, e2_s = 0.0, 0.0
    for dz in (0.0, D_PITCH):
        a = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF, eta=eta0)
        b = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF, eta=1.6 * eta0, cutoff=6)
        per = _by_block(a, b)
        e2_main = max(e2_main, per["G"], per["C"], per["H"])
        e2_s = max(e2_s, per["S"])
        vals = " ".join(f"{per[n]:10.2e}" for n, _, _ in blocks)
        print(f"       {dz:7.2f} {'':>10} {vals}")
    print(f"       kappa_P d = {OMEGA / REF.alpha * D_PITCH:.4f}; the S block loses about")
    print("       (eta/kappa)^4 in the fourth-derivative term. [E5] decides if it matters.")

    # ---- [E3] cutoff convergence --------------------------------------------
    print("\n  [E3] cutoff convergence")
    e3 = 0.0
    for dz in (0.0, D_PITCH):
        a = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF, cutoff=4)
        b = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, dz, OMEGA, REF, cutoff=7)
        rel = float(np.abs(a - b).max() / np.abs(a).max())
        e3 = max(e3, rel)
        print(f"       dz = {dz:5.2f} m   cutoff 4 -> 7:  {rel:.2e}")

    # ---- [E4] the R = 0 asymmetry, stated as a number ------------------------
    print("\n  [E4] the R = 0 convention (included at dz != 0, excluded at dz = 0)")
    ker_dz = bloch_kernel_hat_9x9(M_CELLS, D_PITCH, D_PITCH, OMEGA, REF)
    from cubic_scattering.resonance_tmatrix import _propagator_block_9x9  # noqa: PLC0415

    direct = _propagator_block_9x9(np.array([D_PITCH, 0.0, 0.0]), OMEGA, REF)
    share = float(np.abs(direct).max() / np.abs(ker_dz[0, 0]).max())
    print(f"       the R = 0 term is {share:.1%} of the dz = {D_PITCH} m kernel at k = 0.")
    print("       If it were wrongly excluded the inter-plane coupling would lose")
    print("       its largest single contribution, and no same-plane test would see it.")

    # ---- [E5] does any of it reach the observable? --------------------------
    print("\n  [E5] end-to-end: the computed reflection coefficient vs eta")
    print(f"       {'eta/eta0':>9} {'cutoff':>7} {'floor':>16} {'rel':>11}")
    e5 = 0.0
    base = None
    for fac, cut in ((1.0, 4), (0.4, 16), (1.4, 8), (2.0, 10)):
        val = _floor(eta0 * fac, cut)
        if base is None:
            base = val
        rel = abs(val - base) / base
        e5 = max(e5, rel)
        print(f"       {fac:9.2f} {cut:7d} {val:16.10e} {rel:11.2e}")
    print("       A noisy kernel block only matters if the noise arrives here.")

    print("\n" + "=" * 88)
    print(f"  [E1]  Ewald vs spectral   G/C/H: {e1_main:.2e}   S: {e1_s:.2e}")
    print(f"  [E1b] spectral vs gated reference: {e1b:.2e}")
    print(f"  [E2]  eta-independence    G/C/H: {e2_main:.2e}   S: {e2_s:.2e}")
    print(f"  [E3]  cutoff convergence       : {e3:.2e}")
    print(f"  [E5]  end-to-end eta stability : {e5:.2e}")
    ok = e1_main < 1e-7 and e2_main < 1e-8 and e1_s < 1e-6 and e2_s < 1e-3
    ok = ok and e1b < 1e-12 and e3 < 1e-9 and e5 < 1e-5
    if ok:
        print("\n  PASS: both branches of the wired kernel are correct. The dz != 0")
        print("  branch agrees with a spectral construction that shares no code with")
        print("  it, so the R = 0 convention and the Bloch phase sign are confirmed")
        print("  against something independent rather than against themselves.")
        print()
        print("  dz != 0 uses the SPECTRAL sum and dz = 0 the Ewald sum, because")
        print("  Ewald is needed only where the plain reciprocal sum diverges, and")
        print("  at dz != 0 it is the worse-conditioned of the two. The residual")
        print(f"  S-block eta-dependence at dz = 0 is {e2_s:.1e}, and [E5] shows it does")
        print(f"  not reach the observable: a 5x change in eta moves the answer by {e5:.1e}.")
        print("  A case relying on the strain-strain coupling directly should")
        print("  re-check [E5] rather than assume this carries over.")
    else:
        print("\n  FAIL:", end=" ")
        if e1_main >= 1e-7:
            print("the two routes disagree -- check the R = 0 inclusion and the sign.", end=" ")
        if e1b >= 1e-12:
            print("the spectral route no longer matches its own gated reference.", end=" ")
        if e2_main >= 1e-8:
            print("G/C/H eta-dependent -- not the known S-block cancellation.", end=" ")
        if e1_s >= 1e-6 or e2_s >= 1e-3:
            print("S block worse than the cancellation alone explains.", end=" ")
        if e3 >= 1e-9:
            print("not converged in the cutoff.", end=" ")
        if e5 >= 1e-5:
            print("the kernel error REACHES the observable -- lower eta, raise cutoff.", end=" ")
        print()
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
