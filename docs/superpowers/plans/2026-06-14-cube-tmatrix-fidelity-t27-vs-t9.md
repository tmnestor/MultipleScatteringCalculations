# Cube Multipole Far-Field — T₂₇ vs T₉ Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — invoke `superpowers:test-driven-development` (red-green-refactor) for every Task below, and `superpowers:verification-before-completion` before any "done"/PASS claim. Every implementation step is preceded by a failing test and followed by a verification command whose output you must read before proceeding.

**Goal:** Build one finite-size, higher-multipole far-field radiation operator `cube_multipole_far_field` for the single-site cube T-matrix, and radiate **both** representations through it **apples-to-apples** — same bare physical ΔC, same finite-size phase, same full-field equivalent-source formula — differing **only** in the scattered interior field's basis: T9 populates the 9 displacement+strain modes (from the existing 9×9 sub-cell T-matrix), T27 populates all 27 Galerkin modes (3 displacement + 6 strain + 18 quadratic), radiated via closed-form polynomial×plane-wave cube integrals. The T9↔T27 far-field difference is therefore purely the richer basis (clean RQ2 attribution). The existing `cube_far_field` is retained **only** as a point/k→0-limit cross-check, NOT as the study's T9. Then drive a fidelity study comparing T₉ and T₂₇ against the exact elastic Mie sphere across contrast × ka × polarization × angle, emitting CSV/JSON cost-accuracy datasets. No plotting.

**Architecture:** The scattered far-field source is the polarization body force `f_eq(r') = ω²Δρ·u(r') + ∇'·(ΔC:ε(r'))`. The interior field `u(r') = Σ_α c_sc[α] φ_α(r')` is reconstructed from the T-matrix mode amplitudes `c_sc = assemble_tmatrix_27(galerkin) @ c_inc`. Each trial function `φ_α` is a polynomial (const/linear/quadratic monomial × Cartesian direction) over the cube `[-a,a]³`. Its far-field radiation integral `∫_cube φ_α(r') e^{-ik_sc·r'} d³r'` **separates into a product of three 1D integrals** `I_p(k_j) = ∫_{-a}^{a} t^p e^{-i k_j t} dt`, p∈{0,1,2}, closed-form with removable k→0 limits. The far-field amplitude projects the Fourier-transformed source onto P/SV/SH using the **same `r_hat`/`sv_hat`/`sh_hat` basis and the same `-Q/(4πρc²)` convention** as the existing `cube_far_field`. Each layer is pinned against a Gauss-quadrature arbiter before the next is built. A Mathematica script cross-checks the closed forms symbolically.

**Tech Stack:** Python 3 (numpy, scipy for quadrature arbiter only), conda env `seismic`, pytest, ruff, mypy. Wolfram Language (`wolframscript`) for symbolic cross-check. Coordinate system per CLAUDE.md: **z = axis 0 (down), x = axis 1, y = axis 2**; Voigt order `(ε11, ε22, ε33, 2ε23, 2ε13, 2ε12)`.

---

## File Structure

| Path | Created/Modified | Single responsibility |
|------|------------------|------------------------|
| `cubic_scattering/cube_radiation.py` | Create | 1D primitives `radiation_I1d`, 3D monomial radiation `radiation_monomial`, per-mode source→radiation assembly, and `cube_multipole_far_field`. The whole analytic radiation operator. |
| `cubic_scattering/tests/test_cube_multipole_far_field.py` | Create | All unit/integration tests: primitives vs scipy, monomial vs 3D Gauss, operator == Gauss-quadrature arbiter for BOTH 9- and 27-mode `c_sc`, `cube_far_field` k→0 cross-check (not bit-for-bit), T27→Mie Rayleigh convergence, `resonance_far_field` cross-check, reciprocity, low-ka regression pins. |
| `cubic_scattering/__init__.py` | Modify (imports ~115–120, `__all__` ~216–217) | Export `cube_multipole_far_field`. |
| `Mathematica/CubeMultipoleRadiation.wl` | Create | Symbolic derivation of the closed-form 1D/3D radiation integrals + LaTeX fragment, cross-checking the Python forms. |
| `scripts/cube_tmatrix_fidelity_study.py` | Create | CLI study driver: sweeps the §5 matrix, computes per-channel L²/L∞ errors for T9 & T27 vs Mie, measures cost vectors, writes CSV/JSON + summary. No plotting. |

---

## Task 1 — Branch + scaffold empty module and test file

**Files:**
- Create `cubic_scattering/cube_radiation.py`
- Create `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Create the feature branch off `main` (do NOT implement on `main`):
  ```bash
  git checkout main && git pull --ff-only 2>/dev/null; git checkout -b feature/cube-multipole-far-field
  ```
- [ ] Verify branch:
  ```bash
  git branch --show-current
  ```
  Expected output: `feature/cube-multipole-far-field`
- [ ] Create `cubic_scattering/cube_radiation.py` with only a module docstring and imports (no functions yet):
  ```python
  """cube_radiation.py

  Finite-size higher-multipole far-field radiation operator for the single-site
  cube T-matrix.

  The scattered far-field source is the polarization body force
      f_eq(r') = omega**2 * Drho * u(r') + div'(DC : eps(r'))
  whose Fourier transform at k_sc = k * r_hat, projected onto P/SV/SH, is the
  far-field amplitude.  The interior field u(r') = sum_alpha c_sc[alpha] phi_alpha(r')
  is reconstructed from the 27 Galerkin trial-function amplitudes; each phi_alpha is a
  Cartesian-monomial polynomial over the cube [-a, a]**3, so its radiation integral
  separates into a product of 1D integrals

      I_p(k_j) = int_{-a}^{a} t**p exp(-i k_j t) dt,   p in {0, 1, 2}

  with removable k -> 0 limits.

  Coordinate system (CLAUDE.md): z = axis 0 (down), x = axis 1, y = axis 2.
  Voigt order: (e11, e22, e33, 2 e23, 2 e13, 2 e12).
  """

  from __future__ import annotations

  import numpy as np
  ```
- [ ] Create `cubic_scattering/tests/test_cube_multipole_far_field.py` with header and shared fixtures mirroring `test_scattered_field.py`:
  ```python
  """test_cube_multipole_far_field.py

  Tests for the finite-size higher-multipole far-field radiation operator.
  """

  import numpy as np

  from cubic_scattering import (
      MaterialContrast,
      ReferenceMedium,
      compute_cube_tmatrix_galerkin,
      compute_elastic_mie,
      mie_far_field,
  )
  from cubic_scattering.incident_field import cube_overlap_integrals
  from cubic_scattering.scattered_field import cube_far_field
  from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

  REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
  CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
  WEAK_CONTRAST = MaterialContrast(
      Dlambda=REF.mu * 1e-4, Dmu=REF.mu * 1e-4, Drho=REF.rho * 1e-4
  )


  def _setup(ka: float, a: float = 10.0, contrast: MaterialContrast = CONTRAST):
      """Build (omega, galerkin, k_vec, pol, c_inc, c_sc) for P-wave along axis 0 (z)."""
      omega = ka * REF.beta / a
      g = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)
      T27 = assemble_tmatrix_27(g)
      kP = omega / REF.alpha
      k_vec = np.array([kP, 0.0, 0.0])
      pol = np.array([1.0, 0.0, 0.0])
      c_inc = cube_overlap_integrals(k_vec, pol, a)
      c_sc = T27 @ c_inc
      return omega, g, k_vec, pol, c_inc, c_sc
  ```
  > NOTE: this `_setup` uses **z = axis 0** as the propagation axis (CLAUDE.md), unlike the legacy `test_scattered_field.py` which used axis 2. Both are valid because the operator is rotationally consistent; we standardize on axis 0 here. The arbiter and k→0 cross-check tests (Task 6) compare operators *under the same `k_vec`/`pol`*, so the choice of axis does not affect those pins.
- [ ] Lint/format/type the new files:
  ```bash
  conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
  conda run -n seismic ruff format cubic_scattering/
  conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
  ```
  Expected: ruff "All checks passed!" / files reformatted, mypy "Success" (or only pre-existing unrelated notes).
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "🏗️ scaffold cube_radiation module and test file"
  ```

---

## Task 2 — 1D radiation primitive `radiation_I1d(p, k, a)` for p ∈ {0,1,2}

The primitive is `I_p(k) = ∫_{-a}^{a} t^p e^{-i k t} dt`. Note the **negative** sign in the exponent (this is `e^{-ik_sc·r'}`), which is the complex conjugate of `incident_field._monomial_fourier_1d`'s `e^{+ikx}`. By symmetry `I_p(k) = conj(_monomial_fourier_1d(p, k, a))` for real `k` and real-`a`, but we derive and test it directly with closed forms and removable limits so the module is self-contained and the Mathematica cross-check is on these exact forms.

Closed forms (all real-`a`):
- `I_0(k) = 2 sin(ka)/k`, limit k→0 → `2a`.
- `I_1(k) = i d/dk I_0 = (2i/k²)(sin(ka) − ka cos(ka))` **with opposite sign vs the `+ik` version**: for `e^{-ikt}`, `I_1(k) = ∫ t e^{-ikt} dt = -(2i/k²)(sin(ka) − ka cos(ka))`, limit k→0 → `0`.
- `I_2(k) = -d²/dk² I_0 = (2/k³)(2 ka cos(ka) + (k²a² − 2) sin(ka))`, limit k→0 → `2a³/3`.

The derivation of signs (odd power t → odd integrand against cos, even against sin) must be confirmed against the Gauss/scipy arbiter in the test below, not assumed.

**Files:**
- Modify `cubic_scattering/cube_radiation.py` (append after imports)
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py` (append)

Steps:
- [ ] Write the failing test. Append to the test file:
  ```python
  from cubic_scattering.cube_radiation import radiation_I1d


  def _quad_I1d(p: int, k: float, a: float) -> complex:
      """Reference 1D integral int_{-a}^{a} t**p exp(-i k t) dt via dense quadrature."""
      t = np.linspace(-a, a, 200001)
      return np.trapezoid(t**p * np.exp(-1j * k * t), t)


  def test_radiation_I1d_matches_quadrature():
      for p in (0, 1, 2):
          for k in (0.0, 1e-7, 0.013, 0.37, 2.4, -1.1):
              a = 1.7
              got = radiation_I1d(p, k, a)
              if k == 0.0:
                  exact = {0: 2.0 * a, 1: 0.0, 2: 2.0 * a**3 / 3.0}[p]
                  assert abs(got - exact) < 1e-12, (p, k, got, exact)
              else:
                  ref = _quad_I1d(p, k, a)
                  assert abs(got - ref) < 1e-8, (p, k, got, ref)


  def test_radiation_I1d_smallk_limits_exact():
      a = 0.9
      assert abs(radiation_I1d(0, 0.0, a) - 2.0 * a) < 1e-14
      assert abs(radiation_I1d(1, 0.0, a) - 0.0) < 1e-14
      assert abs(radiation_I1d(2, 0.0, a) - 2.0 * a**3 / 3.0) < 1e-14
  ```
  > The quadrature arbiter tolerance is `1e-8` (trapezoid on 2·10⁵ points, oscillatory integrand); the k→0 limit is pinned to `1e-12`–`1e-14` against the analytic limit. This is the §4.3 "≤1e-12" gate at k→0 and the smooth-regime quadrature gate.
- [ ] Run it (expected FAIL — `radiation_I1d` does not exist):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k radiation_I1d -v
  ```
  Expected: `ImportError: cannot import name 'radiation_I1d'` → collection error / FAIL.
- [ ] Minimal implementation. Append to `cube_radiation.py`:
  ```python
  def radiation_I1d(p: int, k: float, a: float) -> complex:
      """Return the 1D radiation integral int_{-a}^{a} t**p exp(-i k t) dt.

      Closed form for p in {0, 1, 2} with removable k -> 0 limits.

      Args:
          p: Monomial power (0, 1, or 2).
          k: Wavenumber component along this axis (rad/m); may be zero or negative.
          a: Cube half-width (m); cube extends over [-a, a].

      Returns:
          Complex value of the integral.

      Raises:
          ValueError: If p is not in {0, 1, 2}.
      """
      ka = k * a
      abs_ka = abs(ka)

      if p == 0:
          if abs_ka < 1e-8:
              ka2 = ka * ka
              return complex(2.0 * a * (1.0 - ka2 / 6.0 + ka2 * ka2 / 120.0))
          return complex(2.0 * np.sin(ka) / k)

      if p == 1:
          # int t exp(-ikt) dt = -(2i/k**2)(sin(ka) - ka cos(ka))
          if abs_ka < 1e-6:
              ka2 = ka * ka
              return complex(-2j * a**3 * k / 3.0 * (1.0 - ka2 / 10.0 + ka2 * ka2 / 280.0))
          return complex(-2j / k**2 * (np.sin(ka) - ka * np.cos(ka)))

      if p == 2:
          # int t**2 exp(-ikt) dt = (2/k**3)(2 ka cos(ka) + (k**2 a**2 - 2) sin(ka))
          if abs_ka < 1e-5:
              ka2 = ka * ka
              return complex(2.0 * a**3 / 3.0 * (1.0 - 3.0 * ka2 / 10.0 + ka2 * ka2 / 56.0))
          return complex(2.0 / k**3 * (2.0 * ka * np.cos(ka) + (ka**2 - 2.0) * np.sin(ka)))

      msg = f"radiation_I1d: power p={p} not supported (only 0, 1, 2)"
      raise ValueError(msg)
  ```
  > Note: the `p==1` branch differs from `incident_field._monomial_fourier_1d(1, ...)` by an overall sign because of the `-ikt` exponent. `p==0` and `p==2` (even integrands) are sign-identical to the `+ik` version since `cos` is even. This sign asymmetry is exactly what the quadrature arbiter pins.
- [ ] Run it (expected PASS):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k radiation_I1d -v
  ```
  Expected: `2 passed`.
- [ ] Lint/format/type:
  ```bash
  conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
  conda run -n seismic ruff format cubic_scattering/
  conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
  ```
  Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✨ add 1D cube radiation primitive I_p(k) with k->0 limits"
  ```

---

## Task 3 — 3D monomial radiation `radiation_monomial(exp, k_sc, a)`

For a monomial `r0^e1 · r1^e2 · r2^e3` (with `r0=z, r1=x, r2=y` per CLAUDE.md), the cube radiation integral factorizes:
`∫_cube r0^e1 r1^e2 r2^e3 e^{-ik_sc·r'} d³r' = I_{e1}(k_sc[0]) · I_{e2}(k_sc[1]) · I_{e3}(k_sc[2])`.

**Files:**
- Modify `cubic_scattering/cube_radiation.py`
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Write the failing test. Append:
  ```python
  from cubic_scattering.cube_radiation import radiation_monomial


  def _quad_monomial(exp, k_sc, a, n=121):
      """3D Gauss-Legendre reference for int_cube r^exp exp(-i k_sc . r') d3r'."""
      nodes, weights = np.polynomial.legendre.leggauss(n)
      t = nodes * a
      w = weights * a
      e1, e2, e3 = exp
      f0 = (t**e1) * np.exp(-1j * k_sc[0] * t) * w
      f1 = (t**e2) * np.exp(-1j * k_sc[1] * t) * w
      f2 = (t**e3) * np.exp(-1j * k_sc[2] * t) * w
      return f0.sum() * f1.sum() * f2.sum()


  def test_radiation_monomial_matches_gauss():
      a = 1.3
      k_sc = np.array([0.41, -0.22, 0.67])
      for exp in [(0, 0, 0), (1, 0, 0), (0, 2, 0), (2, 0, 0), (1, 1, 0),
                  (0, 1, 1), (2, 0, 0), (0, 0, 2), (1, 0, 1)]:
          got = radiation_monomial(exp, k_sc, a)
          ref = _quad_monomial(exp, k_sc, a)
          assert abs(got - ref) < 1e-10, (exp, got, ref)


  def test_radiation_monomial_zero_k_is_volume_moment():
      a = 2.0
      k0 = np.array([0.0, 0.0, 0.0])
      # const -> volume (2a)**3; r0**2 -> (2a)**2 * 2a**3/3
      assert abs(radiation_monomial((0, 0, 0), k0, a) - (2 * a) ** 3) < 1e-12
      assert abs(radiation_monomial((2, 0, 0), k0, a) - (2 * a) ** 2 * (2 * a**3 / 3)) < 1e-10
  ```
- [ ] Run it (expected FAIL — `radiation_monomial` missing):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k radiation_monomial -v
  ```
  Expected: ImportError / FAIL.
- [ ] Minimal implementation. Append:
  ```python
  def radiation_monomial(
      exp: tuple[int, int, int], k_sc: np.ndarray, a: float
  ) -> complex:
      """Radiate a single Cartesian monomial r0**e1 r1**e2 r2**e3 over the cube.

      Computes int_cube r0**e1 r1**e2 r2**e3 exp(-i k_sc . r') d3r' as the product
      of three 1D primitives.  Axis order is (r0, r1, r2) = (z, x, y).

      Args:
          exp: Monomial powers (e1, e2, e3), each in {0, 1, 2}.
          k_sc: Scattered wavevector (3,) = k * r_hat (rad/m).
          a: Cube half-width (m).

      Returns:
          Complex radiation integral.
      """
      e1, e2, e3 = exp
      return (
          radiation_I1d(e1, float(k_sc[0]), a)
          * radiation_I1d(e2, float(k_sc[1]), a)
          * radiation_I1d(e3, float(k_sc[2]), a)
      )
  ```
- [ ] Run it (expected PASS):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k radiation_monomial -v
  ```
  Expected: `2 passed`.
- [ ] Lint/format/type (same three commands as Task 2). Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✨ add 3D cube monomial radiation via separated 1D integrals"
  ```

---

## Task 4 — Per-mode equivalent-source radiation `mode_source_radiation(c_sc, k_sc, contrast, omega, a)`

This is the load-bearing physics step. The far-field source is
`f_eq(r') = ω²Δρ·u(r') + ∇'·(ΔC:ε(r'))`, with `u(r') = Σ_α c_sc[α] φ_α(r')`.

The 27 trial functions are read directly from `compute_gerade_blocks._build_basis_components()` (first 27 of the 57-element basis). Each `φ_α` is a list of `(comp, exp, coeff)` triples: `comp ∈ {0,1,2}` is the displacement Cartesian component (z,x,y); `exp=(e1,e2,e3)` the monomial powers; `coeff` a scalar. Concretely:
- indices 0–2: `(k, (0,0,0), 1.0)` for k=0,1,2 — **constant displacement** `e_k`.
- indices 3–5: `(k, exp_k, 1.0)` with `exp_k` the unit power in axis k — **axial strain** `r_k e_k`.
- indices 6–8: symmetrized shear, pairs `(1,2),(0,2),(0,1)`, each `[(b,exp_a,0.5),(a,exp_b,0.5)]` — **shear strain**.
- indices 9–26: `(d, exp, 1.0)` for d=0,1,2 and `exp ∈ _QUAD_EXP = [(2,0,0),(0,2,0),(0,0,2),(0,1,1),(1,0,1),(1,1,0)]` — **18 quadratic modes**.

The two source contributions, written in terms of the **vector amplitude** `Q̃(r̂)` whose P/SV/SH projection gives the far field (matching the existing convention `f_P = -Q_P/(4πρα²)`, `f_S = -Q_S/(4πρβ²)`):

1. **Force monopole part** (from `ω²Δρ·u`). The radiated vector force is
   `F̃(k_sc) = ω²Δρ · Σ_α c_sc[α] · ∫_cube φ_α(r') e^{-ik_sc·r'} d³r'`.
   Since each `φ_α = Σ_terms coeff · e_{comp} · r^exp`, its vector radiation is
   `Σ_terms coeff · e_{comp} · radiation_monomial(exp, k_sc, a)`. Sum component-wise into a `(3,)` complex vector `F̃`.

2. **Stress-divergence part** (from `∇'·(ΔC:ε)`). For each mode, the strain field is
   `ε_ij(r') = ½(∂_i u_j + ∂_j u_i)` with `u_j(r') = Σ c_sc[α] φ_α[...]`. The stress is `σ = ΔC : ε` (cubic-symmetry Voigt stiffness contrast `effective_stiffness_voigt`-shaped, **but built from the bare contrast** for the full-field divergence — see derivation note below). In the far field, `∫_cube ∂'_k(Δσ_jk) e^{-ik_sc·r'} d³r'`. Integrating by parts (the trial functions do NOT vanish on the cube boundary, so the boundary term is the physical surface traction and is **kept**):
   `∫ ∂'_k(Δσ_jk) e^{-ik_sc·r'} d³r' = ∮_{∂cube} Δσ_jk n_k e^{-ik_sc·r'} dS + i k_sc,k ∫ Δσ_jk e^{-ik_sc·r'} d³r'`.
   In the far-field amplitude convention used by `cube_far_field`, the stress enters as the **stress moment tensor** radiated with the phase, projected via `r̂`. The cleanest, derivation-safe encoding (and the one the Gauss arbiter pins) is to radiate the full polynomial stress field directly: build the Voigt strain field per mode as a **polynomial in r'** (each strain component is a sum of monomials with powers ≤1), apply the bare cubic stiffness contrast `Dc = effective_stiffness_voigt-shaped from (contrast.Dlambda, contrast.Dmu, contrast.Dmu)` (i.e. the *physical* `ΔC`, NOT the effective `Δc*`), and radiate `Δσ_jk(r')` monomials, assembling the stress-moment vector exactly as the divergence/by-parts identity above prescribes.

   **Derivation procedure (do this in code, validate against Gauss):**
   - For mode α with terms `(comp_j, exp, coeff)`, the displacement is `u_j(r') = c_sc[α]·coeff·r'^exp` (only component `comp_j` nonzero per term). Its gradient `∂_i u_j = c_sc[α]·coeff·exp_i·r'^{exp−e_i}` (lower the i-th power by 1; zero if `exp_i==0`).
   - Form the symmetric Voigt strain polynomial: a length-6 array, each entry a `dict[exp_tuple → complex coeff]`, by symmetrizing `½(∂_i u_j + ∂_j u_i)` over all modes weighted by `c_sc`.
   - Apply `Dc_phys = effective_stiffness_voigt(contrast.Dlambda, contrast.Dmu, contrast.Dmu)` → Voigt stress polynomial (6 monomial-dicts). (Using `Dmu` for both diag and off gives the **isotropic physical** `ΔC`; the cube's *effective* anisotropy lives in `Δc*` used only by the T9 dipole, not in the bare full-field divergence.)
   - Convert the 6 Voigt-stress monomial-dicts to a 3×3 tensor-of-dicts `Δσ_jk(r')` via `_voigt_to_tensor`-style mapping.
   - The radiated stress-moment vector is `S̃_j(k_sc) = i Σ_k k_sc,k · Σ_{exp} σ_jk[exp]·radiation_monomial(exp, k_sc, a) + (surface term)`. **Implement the surface term explicitly** as `∮ Δσ_jk n_k e^{-ik_sc·r'} dS` summed over the six faces `r0=±a, r1=±a, r2=±a` (each face integral is a 2D product of `radiation_I1d` over the in-face axes times `(±a)^{exp}·e^{∓i k_j a}` on the normal axis). Equivalently, radiate the *divergence field* `(∇'·Δσ)_j(r')` directly as a polynomial (its monomials have powers ≤0 for quadratic modes, i.e. constants) PLUS the by-parts identity; the function returns the same `S̃`.

   > **Validation gate (this is what makes the derivation safe, not fabricated):** the assembled `mode_source_radiation` is compared, mode-by-mode and for the full `c_sc`, against a **direct Gauss-quadrature of `f_eq(r')` on a Legendre grid** (Task 5 arbiter) to ≤1e-10. Do NOT hand-fabricate the closed form; assemble it from `radiation_monomial`/`radiation_I1d` and let the arbiter confirm the by-parts bookkeeping (sign of `i k_sc`, the surface term, the strain symmetrization factor ½, the engineering-shear factor of 2 in Voigt rows 3–5).

The function returns `(F_tilde, S_tilde)` as two `(3,)` complex vectors so the far-field projection (Task 6) is identical in form to `cube_far_field`.

**Files:**
- Modify `cubic_scattering/cube_radiation.py`
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Write the failing test (analytic vs direct-field Gauss quadrature). Append:
  ```python
  from cubic_scattering.compute_gerade_blocks import _build_basis_components
  from cubic_scattering.cube_radiation import mode_source_radiation


  def _feq_gauss(c_sc, k_sc, contrast, omega, a, n=80):
      """Direct Gauss-quadrature of int_cube f_eq(r') exp(-i k_sc . r') d3r'.

      f_eq_j = omega**2 Drho u_j + d_k( DC_jklm eps_lm ).
      Returns (F_quad, S_quad): the omega**2 Drho u part and the stress-moment part
      (i k_sc_k tensor-moment + surface traction) separately, to mirror the analytic
      split.  Built from analytic polynomial derivatives + Gauss radiation of the
      polynomial stress, so it is an INDEPENDENT arbiter of the by-parts algebra.
      """
      # Implemented in the test with explicit polynomial differentiation of the
      # basis monomials and a dense Legendre grid; see test body.
      ...


  def test_mode_source_radiation_matches_field_quadrature():
      omega, g, k_vec, pol, c_inc, c_sc = _setup(0.3)
      a = 10.0
      r_hat = np.array([np.cos(0.7), np.sin(0.7), 0.0])
      kS = omega / REF.beta
      k_sc = kS * r_hat
      F_a, S_a = mode_source_radiation(c_sc, k_sc, CONTRAST, omega, a)
      F_q, S_q = _feq_gauss(c_sc, k_sc, CONTRAST, omega, a)
      assert np.allclose(F_a, F_q, atol=1e-10 * (abs(F_q).max() + 1e-30))
      assert np.allclose(S_a, S_q, atol=1e-10 * (abs(S_q).max() + 1e-30))
  ```
  > Implement `_feq_gauss` fully in the test using the same basis (`_build_basis_components()[:27]`), differentiating each monomial analytically to build `u`, `eps`, then `Dc_phys @ eps_voigt` to get `sigma`, then numerically integrating both `omega**2*Drho*u*exp(-ik.r)` (→ F_q) and the stress moment `i*k_sc·sigma·exp + surface traction` (→ S_q) on a Legendre grid. This is a genuinely independent path: it differentiates+contracts on the grid rather than via the closed-form `radiation_monomial`.
- [ ] Run it (expected FAIL — `mode_source_radiation` missing):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k mode_source_radiation -v
  ```
  Expected: ImportError / FAIL.
- [ ] Implement `mode_source_radiation` in `cube_radiation.py` following the derivation procedure above:
  - Load `basis = _build_basis_components()[:27]`.
  - Accumulate the displacement-monomial radiation into `F_tilde` (3,) and the strain polynomial into a Voigt-of-dicts; apply `effective_stiffness_voigt(contrast.Dlambda, contrast.Dmu, contrast.Dmu)`; convert to 3×3 stress-of-dicts; build `S_tilde` (3,) via the `i k_sc · moment + surface` identity, each piece evaluated with `radiation_monomial` / face-integrals built from `radiation_I1d`.
  - Multiply `F_tilde` by `omega**2 * contrast.Drho`.
  - Return `(F_tilde, S_tilde)`.
  Keep helper(s) `_strain_polynomial(c_sc, basis)` and `_radiate_stress_moment(stress_dicts, k_sc, a)` private (leading underscore), Google docstrings, ≤108 cols.
- [ ] Run it (expected PASS):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k mode_source_radiation -v
  ```
  Expected: `1 passed`.
- [ ] Lint/format/type (three commands). Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✨ radiate equivalent body-force source from T27 mode amplitudes"
  ```

---

## Task 5 — Standalone Gauss-quadrature arbiter `full_field_far_field_quadrature(...)`

A reusable arbiter (used by Task 4's test and by Task 7's `resonance` cross-check) that reconstructs the **total** interior field `u(r')`, `ε(r')` on a Legendre grid from `c_total = c_inc + c_sc` and the basis, builds `f_eq`, Fourier-integrates at `k_sc=k·r̂`, and projects onto P/SV/SH using the **same `r_hat`/`sv_hat`/`sh_hat`/`-Q/(4πρc²)` convention** as `cube_far_field`. This concretizes §4.3 "Direct Gauss-quadrature radiation". It must use the **identical** total-field convention as the Task 6 analytic operator so the two match to ≤1e-9 (the apples-to-apples correctness pin).

**Files:**
- Modify `cubic_scattering/cube_radiation.py`
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Write the failing test. Append:
  ```python
  from cubic_scattering.cube_radiation import (
      cube_multipole_far_field,  # used in Task 6; import now to keep one import block
      full_field_far_field_quadrature,
  )


  def test_quadrature_arbiter_self_consistent_volume():
      """At k_sc -> 0 the arbiter's force monopole reduces to omega**2 Drho * int u."""
      omega, g, k_vec, pol, c_inc, c_sc = _setup(0.05)
      a = 10.0
      theta = np.array([0.0, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi])
      f_P, f_SV, f_SH = full_field_far_field_quadrature(
          c_sc, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol, n_gauss=64
      )
      assert f_P.shape == theta.shape
      assert np.all(np.isfinite(f_P))
  ```
  (This test only pins shape/finiteness; the *cross-validation* of the arbiter happens in Task 6 where the analytic operator must match it.)
- [ ] Run it (expected FAIL — `full_field_far_field_quadrature`/`cube_multipole_far_field` missing):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k quadrature_arbiter -v
  ```
  Expected: ImportError / FAIL.
- [ ] Implement `full_field_far_field_quadrature` in `cube_radiation.py`:
  - Signature: `full_field_far_field_quadrature(c_sc, c_inc, theta, ref, contrast, omega, a, k_vec=None, pol=None, n_gauss=64) -> (f_P, f_SV, f_SH)`.
  - `c_total = np.asarray(c_inc, complex) + np.asarray(c_sc, complex)` (require both shape `(27,)`).
  - Build Legendre nodes/weights on `[-a,a]³`; evaluate `u(r')` and `ε(r')` from `basis=_build_basis_components()[:27]` weighted by `c_total` (analytic monomial eval + analytic derivatives for ε).
  - `sigma = Dc_phys : eps` with `Dc_phys = effective_stiffness_voigt(Dlambda, Dmu, Dmu)`.
  - `f_eq_j = omega**2*Drho*u_j + (div sigma)_j` (compute `div sigma` analytically from the polynomial derivatives on the grid).
  - For each θ: `r_hat, sv_hat, sh_hat` exactly as in `cube_far_field` (same `perp1/perp2` construction), `k_sc = k_out·r_hat` with `k_out = kP` for the P channel projection and `kS` for the S channels (radiate `f_eq` with `exp(-i k_sc·r')` using the channel's own wavenumber, matching the `-ikP`/`-ikS` split in `cube_far_field`).
  - Project: `Q = ∫ f_eq e^{-ik_sc·r'} d³r'` (Gauss sum); `f_P = -(r̂·Q)/(4πρα²)`; `Q_perp = Q - (r̂·Q)r̂`; `u_S=-Q_perp/(4πρβ²)`; `f_SV=sv̂·u_S`, `f_SH=sĥ·u_S`.

  > CONVENTION (must match the Task 6 analytic operator EXACTLY): the arbiter radiates the **total** basis-projected interior field `c_total = c_inc + c_sc` — there is NO separately-added exact-plane-wave incident term. Both representations share this identical 27-mode `c_inc`; they differ only in the scattered part (`c_sc` 9-mode-padded vs 27-mode). This is what makes the analytic operator and arbiter agree to ≤1e-9 and keeps the T9↔T27 comparison apples-to-apples. (The basis-projected incident field is the same finite-basis object the T-matrix itself is built on; its O((ka)³) truncation at high ka is an inherent fidelity ceiling affecting both representations equally — a finding, not a bug.)
- [ ] Run it (expected PASS):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k quadrature_arbiter -v
  ```
  Expected: `1 passed`.
- [ ] Lint/format/type. Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✨ add Gauss-quadrature far-field arbiter for radiation validation"
  ```

---

## Task 6 — `cube_multipole_far_field(...)`: one full-field operator (apples-to-apples)

Public operator. **Single radiation path** for both representations — the only difference is which scattered modes `c_sc` populates (9 vs 27). Signature:
```python
cube_multipole_far_field(
    c_sc, c_inc, theta, ref, contrast, omega, a, k_vec=None, pol=None,
) -> (f_P, f_SV, f_SH)
```
- `c_sc` is a length-27 complex vector of scattered mode amplitudes (T9: only indices 0–8 populated, 9–26 zero; T27: all 27). `c_inc` is the length-27 incident overlap (same for both representations).
- The radiated **total** interior field is `Σ_α (c_inc + c_sc)[α] φ_α(r')` — matching `cube_far_field`'s use of `c_total = c_inc + c_sc`. Both representations share the identical incident-field source; they differ ONLY in the scattered part's basis. This is the apples-to-apples comparison.
- Radiate via the analytic full-field operator: force monopole `F̃` (WITH finite-size phase from `mode_source_radiation`) and stress moment `S̃`, projected per θ with the same `r_hat/sv_hat/sh_hat` and `-Q/(4πρc²)` convention as `cube_far_field`.

**Design decision (apples-to-apples, per approved spec amendment 2026-06-14):** there is NO `representation` branch and NO delegation to `cube_far_field`. Both T9 and T27 go through the bare-ΔC full-field path; the caller (Task 10) builds the 9- vs 27-mode `c_sc`. The T9↔T27 far-field gap is therefore purely the richer basis — neither the finite-size phase nor the dipole formulation differs between them. `cube_far_field` is retained ONLY as an independent point/k→0-limit cross-check (it uses the *effective* Δc*×incident-strain point dipole; it agrees with this operator's 9-mode output to leading order as ka→0, NOT bit-for-bit at finite ka — see the cross-check test below). The **primary** correctness pin is analytic == Gauss-quadrature arbiter, for BOTH the 9-mode and 27-mode `c_sc`.

**Files:**
- Modify `cubic_scattering/cube_radiation.py`
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Write failing tests. Append (note: BOTH the 9-mode and 27-mode `c_sc` are validated against the same quadrature arbiter — this is the apples-to-apples correctness pin):
  ```python
  def test_operator_matches_quadrature_T27():
      omega, g, k_vec, pol, c_inc, c_sc = _setup(0.3)  # c_sc = 27-mode (all populated)
      a = 10.0
      theta = np.linspace(0, np.pi, 25)
      f = cube_multipole_far_field(c_sc, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol)
      q = full_field_far_field_quadrature(
          c_sc, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol, n_gauss=80
      )
      scale = sum(abs(qx).max() for qx in q) + 1e-30
      for fx, qx in zip(f, q):
          assert np.max(np.abs(fx - qx)) < 1e-9 * scale


  def test_operator_matches_quadrature_T9():
      omega, g, k_vec, pol, c_inc, c_sc = _setup(0.3)
      a = 10.0
      c_sc9 = c_sc.copy()
      c_sc9[9:] = 0.0  # 9-mode (displacement+strain) interior field only
      theta = np.linspace(0, np.pi, 25)
      f = cube_multipole_far_field(c_sc9, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol)
      q = full_field_far_field_quadrature(
          c_sc9, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol, n_gauss=80
      )
      scale = sum(abs(qx).max() for qx in q) + 1e-30
      for fx, qx in zip(f, q):
          assert np.max(np.abs(fx - qx)) < 1e-9 * scale
  ```
  > `_setup` returns the 27-mode `c_sc = assemble_tmatrix_27(g) @ c_inc`. The T9 amplitudes used by the STUDY (Task 10) come from the 9×9 sub-cell T-matrix (`_sub_cell_tmatrix_9x9`), padded to length 27 with zeros; the unit test above zero-truncates the 27-mode vector purely to exercise the operator's 9-mode path against the arbiter. The cross-check vs `cube_far_field` is a SEPARATE k→0 test (next step).
- [ ] Write the `cube_far_field` k→0 cross-check (NOT bit-for-bit — documents the effective-vs-bare dipole consistency at leading order):
  ```python
  def test_9mode_operator_approaches_cube_far_field_as_ka_small():
      # At very small ka the finite-size phase ->1 and the bare-DC reconstructed-strain
      # 9-mode radiation must agree with cube_far_field's effective-Dc* incident-strain
      # point dipole to leading order (effective-medium identity).
      omega, g, k_vec, pol, c_inc, c_sc = _setup(0.01)
      a = 10.0
      c_sc9 = c_sc.copy(); c_sc9[9:] = 0.0
      theta = np.linspace(0, np.pi, 30)
      got = cube_multipole_far_field(c_sc9, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol)
      ref = cube_far_field(c_inc, c_sc, theta, REF, g, CONTRAST, omega, a, k_vec, pol)
      scale = sum(abs(rx).max() for rx in ref) + 1e-30
      for gx, rx in zip(got, ref):
          assert np.max(np.abs(gx - rx)) < 5e-3 * scale  # leading-order agreement, ka=0.01
  ```
  > Tolerance `5e-3` at ka=0.01 reflects the residual O((ka)²) + effective-vs-bare-dipole difference, NOT a bug. Tighten/loosen empirically during implementation and pin the measured value with a comment. If agreement is markedly worse than O((ka)²), STOP — it signals a sign/normalization error in the stress-moment bookkeeping.
- [ ] Run them (expected FAIL — `cube_multipole_far_field` not implemented):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k "operator_matches or approaches_cube" -v
  ```
  Expected: FAIL (function body missing / NotImplemented).
- [ ] Implement `cube_multipole_far_field`:
  - `c_sc = np.asarray(c_sc, complex)`; require `c_sc.shape == (27,)` and `c_inc.shape == (27,)`, else fail-fast `ValueError` naming the bad shape (no silent pad — the caller pads T9 explicitly, per guardrails).
  - `theta = np.atleast_1d(...)`; `kP=omega/ref.alpha`, `kS=omega/ref.beta`; default `k_vec=[kP,0,0]`, `pol=k_vec/|k_vec|` (axis 0 = z, CLAUDE.md).
  - `c_total = c_inc + c_sc` (the radiated interior field is the TOTAL field).
  - Build `perp1/perp2` exactly as `cube_far_field`. For each θ: `r_hat,sv_hat,sh_hat`; compute `F̃_P,S̃_P = mode_source_radiation(c_total, kP*r_hat, contrast, omega, a)` for the P projection and `F̃_S,S̃_S = mode_source_radiation(c_total, kS*r_hat, contrast, omega, a)` for the S projection (source radiated with the receiving channel's wavenumber, mirroring `cube_far_field`'s `-ikP`/`-ikS` split). Assemble `Q_P = r̂·(F̃_P + S̃_P)`, `f_P=-Q_P/(4πρα²)`; `Q_S_perp = (F̃_S+S̃_S) - (r̂·(F̃_S+S̃_S))r̂`, `u_S=-Q_S_perp/(4πρβ²)`, `f_SV=sv̂·u_S`, `f_SH=sĥ·u_S`.

  > `mode_source_radiation` (Task 4) already folds the `i k_sc` and surface terms into `S̃` and the `ω²Δρ` factor into `F̃` (radiating `c_total`, so the incident-displacement density force is included automatically — no separate incident term needed). The projection is a literal `r̂·(F̃+S̃)` — structurally identical to `cube_far_field`'s `Q_P = rF - ikP rSr`, but with the finite-size phase retained. The arbiter tests confirm the whole chain for both 9- and 27-mode `c_sc`.
- [ ] Run them (expected PASS):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k "operator_matches or approaches_cube" -v
  ```
  Expected: `3 passed`.
- [ ] Export from package. Modify `cubic_scattering/__init__.py`: add `cube_multipole_far_field` to the `from .cube_radiation import (...)` block (or add such a block near the `scattered_field` imports at ~115) and to `__all__` (~216). Run:
  ```bash
  conda run -n seismic python -c "from cubic_scattering import cube_multipole_far_field; print('ok')"
  ```
  Expected: `ok`.
- [ ] Lint/format/type. Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/cube_radiation.py cubic_scattering/__init__.py cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✨ add cube_multipole_far_field: single full-field operator (apples-to-apples T9/T27)"
  ```

---

## Task 7 — Cross-checks: resonance agreement, Mie Rayleigh convergence, reciprocity

**Files:**
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Write the failing tests. Append:
  ```python
  from cubic_scattering import compute_resonance_tmatrix, resonance_far_field


  def test_T27_resonance_agreement_rayleigh():
      """T27 multipole far-field and resonance far-field agree at low ka."""
      a = 10.0
      ka = 0.05
      omega = ka * REF.beta / a
      g = compute_cube_tmatrix_galerkin(omega, a, REF, CONTRAST)
      T27 = assemble_tmatrix_27(g)
      kP = omega / REF.alpha
      k_vec = np.array([kP, 0.0, 0.0])
      pol = np.array([1.0, 0.0, 0.0])
      c_inc = cube_overlap_integrals(k_vec, pol, a)
      c_sc = T27 @ c_inc
      theta = np.linspace(0, np.pi, 19)
      f_P, _, _ = cube_multipole_far_field(
          c_sc, c_inc, theta, REF, CONTRAST, omega, a, k_vec, pol
      )
      res = compute_resonance_tmatrix(omega, a, REF, CONTRAST, n_sub=3, wave_type="P")
      r_P, _, _ = resonance_far_field(res, theta, REF, CONTRAST, omega, a, k_vec, pol)
      denom = np.max(np.abs(r_P)) + 1e-30
      assert np.max(np.abs(f_P - r_P)) < 0.15 * denom


  def test_T27_to_mie_rayleigh_convergence():
      """T27 far-field approaches equal-volume Mie sphere as ka -> 0."""
      a = 10.0
      errs = []
      for ka in (0.2, 0.05):
          omega = ka * REF.beta / a
          g = compute_cube_tmatrix_galerkin(omega, a, REF, WEAK_CONTRAST)
          T27 = assemble_tmatrix_27(g)
          kP = omega / REF.alpha
          k_vec = np.array([kP, 0.0, 0.0])
          pol = np.array([1.0, 0.0, 0.0])
          c_inc = cube_overlap_integrals(k_vec, pol, a)
          c_sc = T27 @ c_inc
          theta = np.linspace(0, np.pi, 61)
          f_P, _, _ = cube_multipole_far_field(
              c_sc, c_inc, theta, REF, WEAK_CONTRAST, omega, a, k_vec, pol
          )
          R = a * (6.0 / np.pi) ** (1.0 / 3.0)
          mie = compute_elastic_mie(omega, R, REF, WEAK_CONTRAST, n_max=8)
          m_P, _, _ = mie_far_field(mie, theta, "P")
          num = np.trapezoid(np.abs(f_P - m_P) ** 2, theta)
          den = np.trapezoid(np.abs(m_P) ** 2, theta) + 1e-300
          errs.append(np.sqrt(num / den))
      assert errs[1] < errs[0], f"L2 error should shrink as ka->0: {errs}"
      assert errs[1] < 0.10, f"weak-contrast Rayleigh L2 error {errs[1]:.3f} too large"


  def test_reciprocity_PtoSV_vs_SVtoP():
      """Five-channel reciprocity: f_{P->SV}/f_{SV->P} ratio is -1 (per project work)."""
      a = 10.0
      ka = 0.1
      omega = ka * REF.beta / a
      g = compute_cube_tmatrix_galerkin(omega, a, REF, CONTRAST)
      T27 = assemble_tmatrix_27(g)
      kP = omega / REF.alpha
      kS = omega / REF.beta
      k_hat = np.array([1.0, 0.0, 0.0])
      th = np.array([0.6])
      # P incidence -> SV channel
      kP_vec, pol_P = kP * k_hat, k_hat
      c_inc_P = cube_overlap_integrals(kP_vec, pol_P, a)
      c_sc_P = T27 @ c_inc_P
      _, f_PtoSV, _ = cube_multipole_far_field(
          c_sc_P, c_inc_P, th, REF, CONTRAST, omega, a, kP_vec, pol_P
      )
      # SV incidence -> P channel
      sv = np.array([0.0, 1.0, 0.0])
      kS_vec = kS * k_hat
      c_inc_S = cube_overlap_integrals(kS_vec, sv, a)
      c_sc_S = T27 @ c_inc_S
      f_SVtoP, _, _ = cube_multipole_far_field(
          c_sc_S, c_inc_S, th, REF, CONTRAST, omega, a, kS_vec, sv
      )
      # Reciprocity holds up to the wavenumber/normalisation factor; pin the sign
      # and order of magnitude (ratio ~ -1 after the (kS/kP) renormalisation).
      ratio = (f_PtoSV[0] / f_SVtoP[0]) * (kS / kP)
      assert ratio.real < 0, f"reciprocity sign wrong: ratio={ratio}"
      assert 0.3 < abs(ratio) < 3.0, f"reciprocity magnitude off: |ratio|={abs(ratio)}"
  ```
  > The reciprocity tolerances are deliberately loose (sign + order of magnitude) because the exact five-channel ratio carries a `(kS/kP)` renormalisation and channel-geometry factor; tighten only if the controller's self-review confirms the exact normalisation from `five-channel-verification`. If the controller has the exact relation, replace the magnitude band with `abs(ratio + 1) < 1e-2`.
- [ ] Run them (expected FAIL or partial — pins not yet satisfied if any convention slipped):
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k "resonance_agreement or mie_rayleigh or reciprocity" -v
  ```
  Expected initially: may FAIL if a sign/normalisation in Task 6 is off — if so, invoke `superpowers:systematic-debugging`, fix in `cube_radiation.py`, re-run. Do NOT loosen tolerances to pass.
- [ ] Once green (expected `3 passed`), lint/format/type. Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✅ cross-check T27 radiation vs resonance, Mie Rayleigh, reciprocity"
  ```

---

## Task 8 — Low-ka regression pins

Freeze concrete numeric far-field values at two `(contrast, ka)` points so future refactors cannot silently drift, per spec §4.5.

**Files:**
- Modify `cubic_scattering/tests/test_cube_multipole_far_field.py`

Steps:
- [ ] Add a placeholder test that PRINTS the values to capture (run once, read output, then hardcode):
  ```python
  def test_print_regression_values():
      a = 10.0
      for ka, contrast, tag in ((0.05, CONTRAST, "mod"), (0.1, WEAK_CONTRAST, "weak")):
          omega = ka * REF.beta / a
          g = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)
          T27 = assemble_tmatrix_27(g)
          kP = omega / REF.alpha
          k_vec = np.array([kP, 0.0, 0.0])
          pol = np.array([1.0, 0.0, 0.0])
          c_inc = cube_overlap_integrals(k_vec, pol, a)
          c_sc = T27 @ c_inc
          th = np.array([0.0, np.pi / 2, np.pi])
          f_P, f_SV, _ = cube_multipole_far_field(
              c_sc, c_inc, th, REF, contrast, omega, a, k_vec, pol
          )
          print(tag, "f_P=", repr(f_P.tolist()))
          print(tag, "f_SV=", repr(f_SV.tolist()))
  ```
- [ ] Run it to capture values:
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -k print_regression_values -s -v
  ```
  Expected: prints two `f_P=` and two `f_SV=` lists. **Copy the printed numbers.**
- [ ] Replace the print test with hardcoded pins (substitute the captured complex values; tolerance `rtol=1e-9, atol=1e-18`):
  ```python
  def test_regression_pins_low_ka():
      a = 10.0
      cases = {
          "mod": (0.05, CONTRAST, <PASTE f_P list>, <PASTE f_SV list>),
          "weak": (0.1, WEAK_CONTRAST, <PASTE f_P list>, <PASTE f_SV list>),
      }
      for ka, contrast, fP_ref, fSV_ref in cases.values():
          omega = ka * REF.beta / a
          g = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)
          T27 = assemble_tmatrix_27(g)
          kP = omega / REF.alpha
          k_vec = np.array([kP, 0.0, 0.0])
          pol = np.array([1.0, 0.0, 0.0])
          c_inc = cube_overlap_integrals(k_vec, pol, a)
          c_sc = T27 @ c_inc
          th = np.array([0.0, np.pi / 2, np.pi])
          f_P, f_SV, _ = cube_multipole_far_field(
              c_sc, c_inc, th, REF, contrast, omega, a, k_vec, pol
          )
          assert np.allclose(f_P, np.array(fP_ref), rtol=1e-9, atol=1e-18)
          assert np.allclose(f_SV, np.array(fSV_ref), rtol=1e-9, atol=1e-18)
  ```
- [ ] Run the full new test module:
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_cube_multipole_far_field.py -v
  ```
  Expected: all tests pass (the `print_regression_values` test is now removed).
- [ ] Run the legacy far-field suite to confirm no regression in `cube_far_field`:
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/test_scattered_field.py cubic_scattering/tests/test_resonance_far_field.py -v
  ```
  Expected: all pass.
- [ ] Lint/format/type. Expected: clean.
- [ ] Commit:
  ```bash
  git add cubic_scattering/tests/test_cube_multipole_far_field.py
  git commit -m "✅ freeze low-ka T27 far-field regression pins"
  ```

---

## Task 9 — Mathematica symbolic cross-check `Mathematica/CubeMultipoleRadiation.wl`

Cross-check the Python closed forms `I_0, I_1, I_2` symbolically and emit a LaTeX fragment, following the project's sequential-integration + `Chop[Re[...]]` + `ToString` wolframclient discipline (the plane-wave kernel is entire, so the integrals are elementary; keep the discipline regardless).

**Files:**
- Create `Mathematica/CubeMultipoleRadiation.wl`

Steps:
- [ ] Write a test-by-construction Wolfram script. Create `Mathematica/CubeMultipoleRadiation.wl`:
  ```wolfram
  (* CubeMultipoleRadiation.wl — closed-form cube radiation integrals I_p(k) *)
  (* Cross-checks cubic_scattering/cube_radiation.py:radiation_I1d and emits LaTeX. *)
  Print["=============================================================="];
  Print["  CubeMultipoleRadiation.wl"];
  Print["=============================================================="];

  (* I_p(k) = Integrate[t^p Exp[-I k t], {t,-a,a}], sequential / assumption-guarded *)
  Ip[p_] := Ip[p] = Assuming[a > 0 && Element[k, Reals],
    Simplify[Integrate[t^p Exp[-I k t], {t, -a, a}, GenerateConditions -> False]]];

  I0 = Ip[0];   (* expect 2 Sin[a k]/k *)
  I1 = Ip[1];   (* expect -(2 I/k^2)(Sin[a k] - a k Cos[a k]) *)
  I2 = Ip[2];   (* expect (2/k^3)(2 a k Cos[a k] + (a^2 k^2 - 2) Sin[a k]) *)

  (* k -> 0 limits *)
  lim0 = Limit[I0, k -> 0];  (* 2 a *)
  lim1 = Limit[I1, k -> 0];  (* 0 *)
  lim2 = Limit[I2, k -> 0];  (* 2 a^3/3 *)
  Print["lim I0 = ", lim0, " | lim I1 = ", lim1, " | lim I2 = ", lim2];

  (* Cross-check vs the Python closed forms (verify zero difference) *)
  pyI0 = 2 Sin[a k]/k;
  pyI1 = -(2 I/k^2) (Sin[a k] - a k Cos[a k]);
  pyI2 = (2/k^3) (2 a k Cos[a k] + (a^2 k^2 - 2) Sin[a k]);
  Print["I0 diff: ", Simplify[I0 - pyI0]];
  Print["I1 diff: ", Simplify[I1 - pyI1]];
  Print["I2 diff: ", Simplify[I2 - pyI2]];

  (* Emit a LaTeX fragment for the downstream PDF *)
  tex = StringJoin[
    "\\begin{align}\n",
    "I_0(k) &= ", ToString[TeXForm[pyI0]], "\\\\\n",
    "I_1(k) &= ", ToString[TeXForm[pyI1]], "\\\\\n",
    "I_2(k) &= ", ToString[TeXForm[pyI2]], "\n\\end{align}\n"];
  Export[FileNameJoin[{NotebookDirectory[] /. {} -> Directory[],
     "CubeMultipoleRadiation_results.tex"}], tex, "Text"];
  Print["Wrote CubeMultipoleRadiation_results.tex"];
  Print["DONE"];
  ```
  > Per CLAUDE.md, run with `/Applications/Wolfram.app/Contents/MacOS/wolframscript`. The three `diff` lines must each print `0`. (No `Mathematica/CubeMultipoleRadiation_results.tex` is committed unless the controller wants the LaTeX fragment tracked; the `.wl` is the deliverable here.)
- [ ] Run it and confirm zero diffs:
  ```bash
  /Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/CubeMultipoleRadiation.wl
  ```
  Expected: `I0 diff: 0`, `I1 diff: 0`, `I2 diff: 0`, `lim I0 = 2 a`, `lim I1 = 0`, `lim I2 = (2 a^3)/3`, `DONE`.
- [ ] Commit (do NOT commit the generated `_results.tex` unless asked):
  ```bash
  git add Mathematica/CubeMultipoleRadiation.wl
  git commit -m "🔬 Mathematica cross-check of cube radiation closed forms + LaTeX"
  ```

---

## Task 10 — Study driver `scripts/cube_tmatrix_fidelity_study.py`

CLI sweeping the §5 matrix, computing per-channel L²/L∞ errors for T9 and T27 vs Mie (equal-volume primary, equal-radius secondary), measuring cost vectors, writing CSV/JSON + a compact summary. **No plotting** (figures are the downstream PDF). Mirrors the `scripts/t27_coupling_study.py` CLI/docstring style.

**Files:**
- Create `scripts/cube_tmatrix_fidelity_study.py`

Steps:
- [ ] Create the driver skeleton with the full sweep, error metrics, cost vectors, and IO. Create `scripts/cube_tmatrix_fidelity_study.py`:
  ```python
  #!/usr/bin/env python3
  """cube_tmatrix_fidelity_study.py — T27 vs T9 single-site fidelity against Mie.

  Sweeps (contrast x ka_beta x incident polarization) and, for each angle grid,
  computes per-channel L2/Linf relative errors of the T9 and T27 cube far-field
  vs the exact elastic Mie sphere (equal-volume primary match R = a (6/pi)**(1/3),
  equal-radius secondary match R = a).  Also records cost vectors (T-matrix
  dimension, assembly wall-time, number of radiated multipole terms).  Writes
  CSV + JSON + a compact text summary.  NO plotting (figures live downstream).

  Coordinate system (CLAUDE.md): z = axis 0 (down), x = axis 1, y = axis 2.

  Usage:
      conda run -n seismic python scripts/cube_tmatrix_fidelity_study.py \
          --out results/cube_fidelity --a 10.0 --n-theta 181
  """
  from __future__ import annotations

  import argparse
  import csv
  import json
  import time
  from dataclasses import asdict, dataclass
  from pathlib import Path

  import numpy as np

  from cubic_scattering import (
      MaterialContrast,
      ReferenceMedium,
      compute_cube_tmatrix,
      compute_cube_tmatrix_galerkin,
      compute_elastic_mie,
      cube_multipole_far_field,
      mie_far_field,
  )
  from cubic_scattering.incident_field import cube_overlap_integrals
  from cubic_scattering.slab_scattering import _sub_cell_tmatrix_9x9
  from cubic_scattering.tmatrix_assembly import assemble_tmatrix_27

  REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)

  CONTRASTS = {
      "weak": MaterialContrast(
          Dlambda=REF.lam * 1e-2, Dmu=REF.mu * 1e-2, Drho=REF.rho * 1e-2
      ),
      "moderate": MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0),
      "strong_pos": MaterialContrast(
          Dlambda=0.45 * REF.lam, Dmu=0.45 * REF.mu, Drho=0.45 * REF.rho
      ),
      "strong_neg": MaterialContrast(
          Dlambda=-0.45 * REF.lam, Dmu=-0.45 * REF.mu, Drho=-0.45 * REF.rho
      ),
  }
  KA_BETA = [0.05, 0.1, 0.3, 0.5, 1.0, 1.5]
  POLARIZATIONS = ["P", "SV", "SH"]
  CHANNELS = ["PtoP", "PtoSV", "SVtoP", "SVtoSV", "SHtoSH"]


  @dataclass
  class ErrorRow:
      contrast: str
      ka_beta: float
      pol: str
      channel: str
      match: str          # "equal_volume" | "equal_radius"
      L2_T9: float
      Linf_T9: float
      L2_T27: float
      Linf_T27: float
      gain_L2: float      # L2_T9 - L2_T27


  @dataclass
  class CostRow:
      contrast: str
      ka_beta: float
      dim_T9: int
      dim_T27: int
      assemble_T9_s: float
      assemble_T27_s: float
      n_terms_T9: int
      n_terms_T27: int


  def _rel_errors(f_rep, f_mie, theta):
      num2 = np.trapezoid(np.abs(f_rep - f_mie) ** 2, theta)
      den2 = np.trapezoid(np.abs(f_mie) ** 2, theta) + 1e-300
      l2 = float(np.sqrt(num2 / den2))
      mx = np.max(np.abs(f_mie)) + 1e-300
      linf = float(np.max(np.abs(f_rep - f_mie)) / mx)
      return l2, linf


  def _channel_pick(pol, f_P, f_SV, f_SH):
      """Map a (pol, channel) to the right far-field array."""
      return {
          ("P", "PtoP"): f_P, ("P", "PtoSV"): f_SV,
          ("SV", "SVtoP"): f_P, ("SV", "SVtoSV"): f_SV,
          ("SH", "SHtoSH"): f_SH,
      }
  ```
  Then the main sweep loop (one function `run(a, n_theta, out)`), filling `ErrorRow`/`CostRow` lists:
  - For each contrast, ka_beta: set `omega = ka_beta * REF.beta / a`, `kP/kS`, `theta = np.linspace(0, np.pi, n_theta)`.
  - Time and build `t9 = compute_cube_tmatrix(omega, a, REF, contrast)` (dim 9) and `g = compute_cube_tmatrix_galerkin(omega, a, REF, contrast)` (dim 27); `T27 = assemble_tmatrix_27(g)`; `T9_9x9 = _sub_cell_tmatrix_9x9(t9, omega, a)` (the same 9×9 the slab uses — import from `cubic_scattering.slab_scattering`; VERIFY its input convention matches `c_inc[:9]` by reading the slab call site).
  - For each `pol in POLARIZATIONS`: build `k_vec/pol` via `plane_wave_PSV_SH(np.array([1.0,0,0]), omega, REF)` (z=axis0), pick the matching wave; `c_inc = cube_overlap_integrals(k_vec, pol, a)` (length 27).
    - **T27 scattered amplitudes:** `c_sc27 = T27 @ c_inc` (all 27 modes populated).
    - **T9 scattered amplitudes (apples-to-apples):** `c_sc9 = np.zeros(27, complex); c_sc9[:9] = T9_9x9 @ c_inc[:9]` (only the 9 displacement+strain modes scatter; quadratic modes zero). Both share the identical 27-mode `c_inc`.
    - **T9 far-field:** `cube_multipole_far_field(c_sc9, c_inc, theta, REF, contrast, omega, a, k_vec, pol)`.
    - **T27 far-field:** `cube_multipole_far_field(c_sc27, c_inc, theta, REF, contrast, omega, a, k_vec, pol)`.
    - The ONLY difference between the two calls is `c_sc9` vs `c_sc27` — same operator, same `c_inc`, same bare ΔC, same finite-size phase ⟹ the gap is purely the richer scattered basis (clean RQ2 attribution).
    - Mie far-field for the equal-volume sphere `R=a*(6/π)**(1/3)` and equal-radius `R=a`: `compute_elastic_mie(...)`, `mie_far_field(mie, theta, pol)`.
    - For each channel belonging to this `pol`, compute `_rel_errors` for T9 and T27 vs Mie (both matches) → append `ErrorRow`.
  - `n_terms_T9 = 9`, `n_terms_T27 = 27` (radiated-mode counts; the spec's "number of multipole terms evaluated").
  - Append `CostRow` with wall-times.
- [ ] Add `main()` with argparse (`--out`, `--a` default 10.0, `--n-theta` default 181) and writers:
  - `out.with_suffix(".errors.csv")`, `out.with_suffix(".cost.csv")` via `csv.DictWriter` over `asdict(row)`.
  - `out.with_suffix(".json")` with `{"errors": [...], "cost": [...], "meta": {...}}`.
  - A printed compact summary: per channel, the `(ka, contrast)` cells where `L2_T27 < L2_T9` (the "win map"), and the median gain.
  - Create the output directory with `Path(out).parent.mkdir(parents=True, exist_ok=True)` (this is an explicit study output dir under `results/`, NOT a temp file — allowed).
- [ ] Smoke-run a reduced sweep to confirm it executes end-to-end (small n_theta):
  ```bash
  conda run -n seismic python scripts/cube_tmatrix_fidelity_study.py --out results/cube_fidelity_smoke --a 10.0 --n-theta 31
  ```
  Expected: prints a summary; writes `results/cube_fidelity_smoke.errors.csv`, `.cost.csv`, `.json`. Confirm files exist:
  ```bash
  ls -la results/cube_fidelity_smoke.errors.csv results/cube_fidelity_smoke.cost.csv results/cube_fidelity_smoke.json
  ```
  Expected: three files listed, non-zero size.
- [ ] Lint/format/type the driver (note: `scripts/` may be outside the default ruff target; run explicitly):
  ```bash
  conda run -n seismic ruff check scripts/cube_tmatrix_fidelity_study.py --fix --ignore ARG001,ARG002,F841,E741
  conda run -n seismic ruff format scripts/cube_tmatrix_fidelity_study.py
  conda run -n seismic mypy scripts/cube_tmatrix_fidelity_study.py --ignore-missing-imports
  ```
  Expected: clean.
- [ ] Commit (do NOT commit the smoke-run `results/` artifacts; add them to `.gitignore` only if the repo doesn't already ignore `results/`):
  ```bash
  git add scripts/cube_tmatrix_fidelity_study.py
  git commit -m "📊 add T27-vs-T9 cube fidelity study driver (CSV/JSON, no plotting)"
  ```

---

## Task 11 — Full suite green + finish

**Files:** none (verification + handoff).

Steps:
- [ ] Run the entire cube_scattering test suite to confirm no regressions anywhere:
  ```bash
  conda run -n seismic python -m pytest cubic_scattering/tests/ -q
  ```
  Expected: all pass (no new failures vs `main`).
- [ ] Final lint/format/type sweep:
  ```bash
  conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
  conda run -n seismic ruff format cubic_scattering/
  conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
  ```
  Expected: clean.
- [ ] Confirm the branch is ahead of `main` with the expected commits:
  ```bash
  git log --oneline main..HEAD
  ```
  Expected: the Task 1–10 commits in order.
- [ ] Invoke `superpowers:finishing-a-development-branch` to choose merge/PR/cleanup. Do not push or merge unless the user asks.

---

## Self-Review (writing-plans checklist)

**Spec coverage:**
- §4.1 `cube_multipole_far_field` — one full-field operator, apples-to-apples T9/T27 → Task 6. ✔
- §4.2 Mathematica `CubeMultipoleRadiation.wl` + LaTeX fragment → Task 9. ✔
- §4.3 arbiters: Gauss-quadrature (Task 5 + Task 4/6 gates ≤1e-10/1e-9, for BOTH 9- and 27-mode `c_sc`), `resonance_far_field` cross-check (Task 7), `cube_far_field` k→0 cross-check (Task 6, NOT bit-for-bit), reciprocity (Task 7). ✔
- §4.4 study driver CSV/JSON, no plotting → Task 10. ✔
- §4.5 tests: analytic==quadrature for 9- and 27-mode `c_sc` (Tasks 2–6), `cube_far_field` k→0 cross-check (Task 6), T27→Mie Rayleigh (Task 7), resonance cross-check (Task 7), reciprocity (Task 7), low-ka pins (Task 8). ✔
- §5 study matrix (contrasts incl. one strong-negative, ka_β grid, P/SV/SH, 5 channels, equal-volume + equal-radius) → Task 10. ✔
- The load-bearing build sequence (1D primitives → 3D monomial → per-mode source → operator → Mathematica → driver → tests) is honored in Tasks 2→10. ✔

**Placeholder scan:** No "TBD"/"similar to Task N"/"add error handling". The only deferred-literal is the Task-8 regression-pin numbers, which are explicitly captured by running the print-test first (a standard freeze-the-golden-value step), not a fabricated value. The Task-4 per-trial-function stress closed form is intentionally specified as a *derivation procedure pinned by the Gauss arbiter* (per the prompt's instruction not to fabricate the formula), with the trial-function polynomials given concretely from `_build_basis_components`. ✔

**Type / name consistency across tasks:**
- `radiation_I1d(p:int, k:float, a:float)->complex`, `radiation_monomial(exp:tuple, k_sc:np.ndarray, a:float)->complex`, `mode_source_radiation(c_total, k_sc, contrast, omega, a)->(F_tilde(3,), S_tilde(3,))` (radiates whatever 27-vector it is given — the operator passes `c_inc+c_sc`), `full_field_far_field_quadrature(c_sc, c_inc, theta, ref, contrast, omega, a, k_vec, pol, n_gauss)->(f_P,f_SV,f_SH)`, `cube_multipole_far_field(c_sc, c_inc, theta, ref, contrast, omega, a, k_vec, pol)->(f_P,f_SV,f_SH)` — used identically in every test and the driver. ✔
- Field/API names verified against code: `ReferenceMedium(alpha,beta,rho)` with `.lam`/`.mu` properties; `MaterialContrast(Dlambda,Dmu,Drho)`; `GalerkinTMatrixResult.Dlambda_star/Dmu_star_diag/Dmu_star_off/T1c/T2c/T3c`; `assemble_tmatrix_27(galerkin)`; `cube_overlap_integrals(k_vec,pol,a)`; `cube_far_field(c_inc,c_sc,theta,ref,galerkin,contrast,omega,a,k_vec,pol)`; `mie_far_field(mie,theta,"P"|"SV"|"SH")`; `compute_elastic_mie(omega,radius,ref,contrast,n_max)`; basis from `compute_gerade_blocks._build_basis_components()[:27]` with `_QUAD_EXP`; `effective_stiffness_voigt(Dlambda,Dmu_diag,Dmu_off)`; `_voigt_to_tensor`; Voigt order and z=axis0 confirmed. ✔
- Critical correctness note (apples-to-apples, approved spec amendment 2026-06-14): BOTH representations radiate through the one full-field operator with the **bare physical** `ΔC` and the **reconstructed total** strain field; T9 and T27 differ ONLY in which scattered modes `c_sc` populates (9 vs 27), so the far-field gap is purely the richer basis. `cube_far_field` (which uses the **effective** `Δc*`×**incident** strain point dipole) is NOT the study's T9 — it is kept solely as an independent k→0 cross-check, agreeing with the operator's 9-mode output to leading order, NOT bit-for-bit at finite ka. The primary correctness pin for both 9- and 27-mode `c_sc` is analytic == Gauss-quadrature arbiter. ✔

**Guardrails honored:** conda `seismic`; ruff (with the exact ignore list) + ruff format + mypy after each Python change; ≤108 cols; `pathlib.Path`; Google docstrings; fail-fast `ValueError` on bad `c_sc`/`c_inc` shape and bad `p`; tests under `cubic_scattering/tests/`; **no commit attribution anywhere**; gitmoji commit subjects. ✔

---

## Execution Handoff (two options)

**Option A — Subagent-driven (recommended).** Execute with `superpowers:subagent-driven-development`: dispatch each Task (2–10) to a fresh subagent with the Task's exact steps, requiring it to paste the FAIL output, the PASS output, and the lint/type output before reporting back. Tasks are sequential (each builds on the prior module function), so run them in order, not in parallel — but Task 9 (Mathematica) and Task 10 (driver) can run in parallel after Task 8 is green, since they only consume the finished operator. Use `superpowers:requesting-code-review` after Task 8 (operator complete) and again after Task 10.

**Option B — Inline.** Execute Tasks 1–11 directly in this session using `superpowers:executing-plans`, pausing at the review checkpoints after Task 6 (operator + arbiter pin) and Task 8 (all cross-checks green) for the user to inspect before building the Mathematica script and study driver. Prefer this if the user wants to watch the red-green transitions live.
