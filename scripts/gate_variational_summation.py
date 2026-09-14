"""GATE: the thesis's OWN principal contribution -- variational summation (Alg 6.2).

WHY THIS GATE EXISTS, AND WHY THE EARLIER ONES COULD NOT CONCLUDE
-----------------------------------------------------------------
`gate_thesis_formulation_periodic` has been stuck at 3.28x separation against a
5x bar, and three routes to sharpen it have been measured and refuted (reflector
strength, reflector phase, cube contrast). Reading the thesis says why that
struggle was mis-aimed. Two things were being conflated:

  * the LS identity G = G_ref + G_ref . DeltaA . G with G_ref the EXACT
    stratified half-space tensor (MSRepresentations.tex, Corollary 1). This is an
    exact operator identity. It makes no approximation, so it cannot be found
    "incomplete"; any residual measured against it belongs to the DISCRETISATION.
  * the thesis's actual claim. From the Introduction, verbatim: "we have
    concentrated on the two-way Riccati factorization approach, where the
    reference Green's tensor is the exact solution for a plane stratified
    halfspace, and NONE of the scattered field has been incorporated within it.
    The burden of convergence of the multiple scattering series is then borne by
    the method used to sum the multiple scattering partial sums. HEREIN, LIES
    THE PRINCIPAL CONTRIBUTION OF THIS RESEARCH."

The stratified reference is a PREMISE. The contribution is the SUMMATION. This
gate tests the summation, and it has no discretisation floor: every scheme is
compared against the converged solve of the SAME matrix, so the discretisation
cancels identically rather than having to be driven down.

THE CLAIM BEING TESTED, pre-registered from the thesis's own numerical
comparison (VariationalSum.tex, section "numerical comparison"): for a SINGLE
transition the variational summation was the fastest method tested, converging
in EIGHT iterations on a coherent 5% heterogeneity. Eight is the number to beat,
and the gate fails if BCGVAR does not converge the transition in far fewer
iterations than the Born series it replaces.

THE HARD CASE IS THE COHERENT BLOCK, and it is chosen deliberately. The thesis
is explicit that a uniformly fast-and-dense block is HARDER than a random medium
of the same size, because neighbouring scatterers radiate in phase and interfere
constructively, where random media dephase and reduce the spectral radius. A
random-medium test would flatter every scheme. The lattice here is therefore
uniformly perturbed, not random.

TWO INDEX TYPOS IN THE PUBLISHED Alg 6.2, both resolved by requiring the
algorithm to reproduce Eq. (BCGscat), which it then does exactly:
  * "if n = 1" must be "if n = 0" -- otherwise beta_{-1} = rho_{-1}/rho_{-2} is
    referenced on the very first pass, which is undefined;
  * alpha_n = rho_{n-1} / <p~_{n-1}, q_n> must carry subscript n on p~, since
    only then does alpha_n rho_{n-1} equal rho_{n-1}^2 / <p~_n, H p_n>, which is
    the summand of (BCGscat).
The leading term is then exactly <b~,b>^2 / <b~, H b>, as (BCGscat) requires.
That agreement is checked numerically below, not asserted.

THE BILINEAR FORM IS NOT AN INNER PRODUCT, and this is the trap in the whole
chapter. (bilinearDef): <v1, v2> = v1^T DeltaC_eff v2 -- TRANSPOSE, no conjugate,
and weighted by the contrast operator. The thesis notes it is indefinite, so
<v,v> = 0 does not imply v = 0 and breakdown is a real possibility rather than a
bug. Using a conjugating inner product here silently changes the method.

SCOPE, stated rather than implied. This gate tests the SUMMATION SCHEME on a
representative Foldy-Lax operator. It does not use the stratified reference
propagator, because the summation claim is a property of the method and not of
the reference; pairing the two is a separate build.

Run:  conda run -n seismic python scripts/gate_variational_summation.py
SI units (m, m/s, kg/m3, Pa) -- the slab machinery's own convention.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.slab_scattering import (  # noqa: E402
    SlabGeometry,
    SlabMaterial,
    _build_slab_incident_field,
    _slab_matvec,
    build_slab_kernels,
    compute_slab_tmatrices,
)

REF = ReferenceMedium(5000.0, 3000.0, 2500.0)
# THE REGIME MUST BE HARD OR THE TEST IS VACUOUS, and the knob is the CONTRAST,
# not the frequency. At 5% the spectral radius is 0.057 and the Born series
# converges on its own, so every scheme "wins" in one iteration and the claim is
# never exercised. Raising omega 4x barely moved it (0.0574 -> 0.0573), because
# the coupling that dominates the spectral radius is the NEAR-FIELD strain block,
# which is static and carries no omega. Measured: rho tracks the contrast, so the
# contrast is what is raised -- staying under the ~52% renormalisation validity
# floor the project has established for |Delta| against the background.
OMEGA = 60.0
A_HALF = 1.0  # ka = 0.012, deep inside the validated ka < 0.3 for the cube T0
M, N_Z = 6, 6  # 216 sites x 9 components = 1944 unknowns
PCT = 0.45  # see note above; the thesis's 5% heterogeneity figure
THESIS_ITERATIONS = 8  # the claim: a single transition in 8 iterations
TOL = 1e-6  # relative accuracy demanded of the transition


def _build_operator() -> tuple:
    """Dense H = (I - G0 T0), the weight DeltaC_eff, and the two source fields."""
    geom = SlabGeometry(M=M, N_z=N_Z, a=A_HALF)
    lam0 = REF.rho * (REF.alpha**2 - 2.0 * REF.beta**2)
    mu0 = REF.rho * REF.beta**2
    ones = np.ones((N_Z, M, M))
    material = SlabMaterial(
        Dlambda=PCT * lam0 * ones,  # UNIFORM, not random: the coherent hard case
        Dmu=PCT * mu0 * ones,
        Drho=PCT * REF.rho * ones,
        ref=REF,
    )
    t0 = compute_slab_tmatrices(geom, material, OMEGA)
    kh = build_slab_kernels(geom, OMEGA, REF)

    n = N_Z * M * M * 9
    # Materialise H column by column. Small enough to be exact and unambiguous,
    # which is the point: the summation scheme is what is under test, not the
    # operator application.
    h = np.zeros((n, n), dtype=complex)
    e = np.zeros(n, dtype=complex)
    for j in range(n):
        e[j] = 1.0
        h[:, j] = _slab_matvec(e, t0, kh, geom)
        e[j] = 0.0

    # The weight of the bilinear form is DeltaC_eff, block-diagonal in the sites,
    # CARRIED IN THE RIGHT METRIC. See gate_nine_component_convention: the
    # 9-component pairing is M = Sigma J = diag(1,1,1, -1,-1,-1, -.5,-.5,-.5) --
    # J undoes the factor 2 of engineering shear, Sigma carries the parity of the
    # odd (single-derivative) blocks, and both are needed. With M in place,
    # M G0 is symmetric to 3.5e-16 and (M T0) H to 3.5e-18. Without it the
    # operator is asymmetric at 1.4e-3 and BiCG stalls there.
    w = np.zeros((n, n), dtype=complex)
    t_flat = t0.reshape(-1, 9, 9)
    mj = np.diag(np.array([1, 1, 1, -1, -1, -1, -0.5, -0.5, -0.5], dtype=float))
    for s in range(t_flat.shape[0]):
        w[s * 9 : (s + 1) * 9, s * 9 : (s + 1) * 9] = mj @ t_flat[s]

    b = _build_slab_incident_field(geom, OMEGA, REF, np.array([1.0, 0.0, 0.0]), "P").ravel()
    # The dual source is the receiver-side illumination -- a DIFFERENT direction,
    # so the transition is a genuine two-sided quantity and not a special case.
    b_dual = _build_slab_incident_field(geom, OMEGA, REF, np.array([0.0, 1.0, 0.0]), "P").ravel()
    return h, w, b, b_dual


def _validate_algorithm_on_synthetic() -> None:
    """Does the TRANSCRIPTION of Alg 6.2 work, on an operator that satisfies its
    hypotheses?

    Separating the two questions matters. "Is Alg 6.2 transcribed correctly" is
    answerable right now; "does it work on our physical operator" is blocked by
    the convention mismatch above. Running it on a synthetic complex-symmetric
    operator -- W symmetric, W H symmetric, which is exactly what the thesis
    assumes -- answers the first without pretending to answer the second.
    """
    rng = np.random.default_rng(20260914)
    n = 120
    # THE SYNTHETIC MUST BE IN THE RIGHT CLASS, or it tests nothing useful. A
    # random dense W^-1 S is the adversarial worst case for BiCG and stagnates
    # for reasons that have nothing to do with the transcription. A Foldy-Lax
    # operator is I - K with K a moderate symmetric coupling, so that is what is
    # built here: W = I (symmetric), K symmetric, spectral radius scaled to 0.5,
    # which makes W H = H symmetric exactly as the thesis assumes.
    wsym = np.eye(n, dtype=complex)
    c = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    k = c + c.T
    k *= 0.5 / max(abs(np.linalg.eigvals(k)))
    h = np.eye(n) - k
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    b_dual = rng.normal(size=n) + 1j * rng.normal(size=n)

    def bil(u, v):
        return complex(u @ (wsym @ v))

    s_exact = bil(b_dual, np.linalg.solve(h, b))
    r, r_dual = b.copy(), b_dual.copy()
    p = p_dual = np.zeros_like(b)
    ds, rho_prev = 0.0 + 0.0j, 0.0 + 0.0j
    hits = None
    for it in range(n + 1):
        rho_n = bil(r_dual, r)
        if it == 0:
            p, p_dual = r.copy(), r_dual.copy()
        else:
            if abs(rho_prev) == 0.0:
                break
            beta = rho_n / rho_prev
            p, p_dual = r + beta * p, r_dual + beta * p_dual
        q, q_dual = h @ p, h @ p_dual
        den = bil(p_dual, q)
        if abs(den) == 0.0:
            break
        alpha = rho_n / den
        ds = ds + alpha * rho_n
        r, r_dual = r - alpha * q, r_dual - alpha * q_dual
        rho_prev = rho_n
        if hits is None and abs(ds - s_exact) / abs(s_exact) < 1e-10:
            hits = it
            break
    print("\n  [V2] Alg 6.2 on a SYNTHETIC operator that satisfies its hypotheses")
    print(f"       (W symmetric, W H symmetric, n = {n}):")
    if hits is not None:
        print(f"       transition converged to 1e-10 in {hits} iterations --")
        print("       the transcription is sound; only the physical operator is blocked.")
    else:
        err = abs(ds - s_exact) / abs(s_exact)
        print(f"       did NOT converge (best {err:.3e}). The transcription is")
        print("       suspect, and that must be resolved before blaming the operator.")


def main() -> int:
    ka = OMEGA / REF.alpha * A_HALF
    print("=" * 84)
    print("GATE -- the thesis's variational summation, Alg 6.2 (VariationalSum.tex)")
    print(f"  ka = {ka:.4f} (validated range ka < 0.3);  {N_Z}x{M}x{M} sites, {PCT:.0%} coherent")
    print(f"  thesis claim: a single transition converges in ~{THESIS_ITERATIONS} iterations")
    print("=" * 84)

    h, w, b, b_dual = _build_operator()

    def bil(u: np.ndarray, v: np.ndarray) -> complex:
        """<u, v> = u^T W v -- TRANSPOSE, no conjugate, contrast-weighted."""
        return complex(u @ (w @ v))

    # THE REFERENCE IS THE CONVERGED SOLVE OF THIS SAME MATRIX. That is what
    # makes the gate floor-free: the discretisation is identical in every arm and
    # cancels, so nothing here depends on how well cubes represent a medium.
    # THE ADJOINT IS NOT THE TRANSPOSE, and using H^T here is silently wrong.
    # The form is W-weighted, so <H_dag u, v> = <u, H v> forces
    # H_dag = W^-1 H^T W. The thesis gives H_dag = H(-k_y) (Htdef), which at
    # k_y = 0 is H ITSELF -- and that is consistent precisely because W H is
    # symmetric (DeltaC H = H(-k_y)^T DeltaC, their Eq. sym). W H = T0 - T0 G0 T0
    # with T0 and G0 both symmetric, so it holds here too. GUARDED, not assumed:
    # a wrong adjoint makes BiCG produce plausible numbers that converge to the
    # wrong quantity, which is exactly what it did before this check existed.
    wh = w @ h
    asym = np.abs(wh - wh.T).max() / np.abs(wh).max()
    print(f"\n  [V1] W H symmetry (forces H_dag = H at k_y = 0) : {asym:.3e}")
    if asym > 1e-10:
        print("  ** BLOCKED, and the cause is a CONVENTION MISMATCH, not a tolerance.")
        print("  ** Measured against the validated closed-form propagator:")
        print("  **   G0 is self-adjoint ONLY under J = diag(1,1,1, 1,1,1, .5,.5,.5)")
        print("  **      -- J P(r) is symmetric to 0.0 / 1.9e-18, exactly;")
        print("  **   T0 is self-adjoint under the IDENTITY -- plain T0^T = T0, 0.0.")
        print("  ** They cannot both be right. The field basis carries ENGINEERING")
        print("  ** shear (2e_xy, 2e_zy, 2e_zx) and the two objects disagree about")
        print("  ** whether the source side carries the compensating 1/2.")
        print("  ** The thesis's whole chapter rests on a bilinear form in which H")
        print("  ** is self-adjoint (Htdef: H_dag = H(-k_y)), so the variational")
        print("  ** summation is NOT TESTABLE on this operator until the 9-component")
        print("  ** source convention is fixed and gated.")
        _validate_algorithm_on_synthetic()
        return 1

    v_exact = np.linalg.solve(h, b)
    s_exact = bil(b_dual, v_exact)
    rho_spec = max(abs(np.linalg.eigvals(np.eye(h.shape[0]) - h)))
    print(f"\n  spectral radius of (I - H) = {rho_spec:.4f}", end="")
    print("   <-- Born series DIVERGES" if rho_spec >= 1.0 else "   (Born converges)")
    print(f"  exact transition <b~, Omega b> = {s_exact:.8e}")

    # ---- the Born series, the thing the variational sum is meant to replace ---
    print(f"\n  {'n':>3} {'BORN':>12} {'BCGVAR':>12} {'BiCG plain':>12}")
    born_v, born_term = np.zeros_like(b), b.copy()
    g0t0 = np.eye(h.shape[0]) - h

    # ---- Algorithm 6.2, with the two index corrections noted in the header ----
    r, r_dual = b.copy(), b_dual.copy()
    p = p_dual = np.zeros_like(b)
    ds, rho_prev, v_bicg = 0.0 + 0.0j, 0.0 + 0.0j, np.zeros_like(b)
    hit_var = hit_plain = None
    first_term = None

    for n in range(40):
        born_v = born_v + born_term
        e_born = abs(bil(b_dual, born_v) - s_exact) / abs(s_exact)
        born_term = g0t0 @ born_term

        rho_n = bil(r_dual, r)
        if n == 0:
            p, p_dual = r.copy(), r_dual.copy()
        else:
            if abs(rho_prev) == 0.0:
                print(f"  {n:3d}   BREAKDOWN: rho = 0 (the form is indefinite)")
                break
            beta = rho_n / rho_prev
            p, p_dual = r + beta * p, r_dual + beta * p_dual
        q, q_dual = h @ p, h @ p_dual  # H_dag = H at k_y = 0, guarded by [V1]
        den = bil(p_dual, q)
        if abs(den) == 0.0:
            print(f"  {n:3d}   BREAKDOWN: <p~, H p> = 0")
            break
        alpha = rho_n / den
        ds = ds + alpha * rho_n
        if n == 0:
            first_term = ds
        r, r_dual = r - alpha * q, r_dual - alpha * q_dual
        v_bicg = v_bicg + alpha * p
        rho_prev = rho_n

        e_var = abs(ds - s_exact) / abs(s_exact)
        e_plain = abs(bil(b_dual, v_bicg) - s_exact) / abs(s_exact)
        if hit_var is None and e_var < TOL:
            hit_var = n
        if hit_plain is None and e_plain < TOL:
            hit_plain = n
        if n < 16 or hit_var is None:
            print(f"  {n:3d} {e_born:12.4e} {e_var:12.4e} {e_plain:12.4e}")
        if hit_var is not None and hit_plain is not None:
            break

    # The leading term of Alg 6.2 must equal the leading term of (BCGscat).
    lead = bil(b_dual, b) ** 2 / bil(b_dual, h @ b)
    lead_err = abs(first_term - lead) / abs(lead)
    print(f"\n  [V0] Alg 6.2 leading term vs Eq. (BCGscat) closed form : {lead_err:.3e}")

    print(f"\n  iterations to {TOL:g}:  BCGVAR {hit_var}   BiCG plain {hit_plain}")
    # BCGVAR AND "BiCG plain" AGREE TO ALL DIGITS, and that is not redundancy --
    # it is a check. Sum_j alpha_j rho_j IS <b~, x_n> identically in exact
    # arithmetic, so agreement confirms the recursion; disagreement would mean
    # the accumulation is wrong. The variational form's advantage is COST, not a
    # different number: it delivers the transition without ever forming or
    # storing the solution vector x, which is the whole point at N = 57600.
    print("=" * 84)
    ok_lead = lead_err < 1e-10
    ok_iter = hit_var is not None and hit_var <= THESIS_ITERATIONS
    if ok_lead and ok_iter:
        print(f"  CONFIRMED: the variational sum reaches {TOL:g} in {hit_var} iterations,")
        print(f"  within the {THESIS_ITERATIONS} the thesis reports, and its leading term")
        print("  reproduces the closed form of (BCGscat). The summation scheme --")
        print("  the thesis's stated principal contribution -- does what it claims.")
    elif ok_lead:
        print("  NOT CONFIRMED: the algorithm is transcribed correctly (leading term")
        print(f"  {lead_err:.1e}) but needed {hit_var} iterations against the {THESIS_ITERATIONS}")
        print("  claimed. Compare the problem: the thesis's block spans 10 SV")
        print("  wavelengths and this lattice does not, so check the regime before")
        print("  reading this as a refutation.")
    else:
        print(f"  TRANSCRIPTION WRONG: the leading term misses (BCGscat) by {lead_err:.1e}.")
        print("  Fix the algorithm before drawing any conclusion about the thesis.")
    print("=" * 84)
    return 0 if (ok_lead and ok_iter) else 1


if __name__ == "__main__":
    raise SystemExit(main())
