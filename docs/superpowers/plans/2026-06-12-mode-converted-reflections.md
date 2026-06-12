# Mode-Converted Reflections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full specular reflection matrix (R_PP, R_PS, R_SP, R_SS + scalar R_SH) for the
periodic slab Foldy-Lax solver, validated against Kennett, and a 2×2 P-SV sub-ocean
recursion in the ocean-bottom study so internal P→S→P conversion feeds the observable.

**Architecture:** One shared Weyl extractor (`slab_weyl_amplitudes`) projects the
layer-averaged Foldy-Lax sources onto outgoing P/SV/SH plane waves; a wrapper runs the
three incident solves and assembles the matrix. The Kennett reference comes from the
existing `KennettResult` channel properties. The ocean-bottom water step becomes the 2×2
recursion with the existing `psv_fluid_solid` coefficients.

**Tech Stack:** Python 3.12, NumPy/SciPy (GMRES), pytest. Conda env `seismic`
(`conda run -n seismic <cmd>`). Repo: `/Users/tod/Desktop/MultipleScatteringCalculations`.

**Spec:** `docs/superpowers/specs/2026-06-12-mode-converted-reflections-design.md`

---

## Conventions you must understand before coding

Read these once; every task below depends on them.

1. **Coordinates:** z = axis 0 (down), x = axis 1, y = axis 2. Slowness vector of an
   upgoing (reflected) wave of mode m at horizontal slowness p: `s⃗_m = (−η_m, p, 0)`
   with `η_m = sqrt(1/c_m² − p²)`, branch Im η ≥ 0 (handled by `_vertical_slowness` in
   `kennett_layers.py`). The complex-unit propagation direction is `d̂_m = c_m s⃗_m`
   (no conjugation — analytic continuation past critical).

2. **T-matrix force sign:** the slab Weyl extraction negates ONLY the force term:
   `Q_P = −d̂·f − iω s⃗·σ·d̂` (see `slab_rpp_periodic` docstring and project memory).
   Do not copy the global minus from `slab_reflected_field`.

3. **Oblique stress coupling (pre-existing bug to fix):** the current
   `slab_rpp_periodic` uses `−i k_z (r̂·σ·r̂)` where `k_z = ω η_P`. The Weyl/plane-wave
   derivation gives `−iω (s⃗·σ·d̂)` — the wave-vector contraction, `ω s⃗ = k_P d̂`, so the
   stress term is `−i k_P (d̂·σ·d̂)`, NOT `−i k_z (d̂·σ·d̂)`. The two agree at p = 0 only
   (`k_z = k_P` there). No existing test exercises oblique slab-vs-Kennett, so this was
   never caught. Task 2 fixes it via the generalized extractor; Task 4's oblique R_PP
   test is the proof.

4. **Modified (energy-normalized) convention:** `kennett_layers` coefficients use the
   `sqrt(η ρ)` normalization (see `PSVCoefficients` docstring) which makes reflection
   matrices SYMMETRIC. The slab extractor produces displacement-amplitude ratios.
   Conversion (incident and reflected in the same sediment medium, ρ cancels):
   `R̃_ij = sqrt(η_i / η_j) · R_ij` with i = outgoing mode row, j = incident mode column.
   Diagonal entries are identical in both conventions. ALL comparisons against Kennett
   and ALL matrix mixing in the ocean-bottom recursion happen in the modified
   convention.

5. **Test parameters** (validated set): background α=5000 m/s, β=3000 m/s, ρ=2500 kg/m³;
   moderate contrast Δλ=+2 GPa, Δμ=+1 GPa, Δρ=+100 kg/m³; weak contrast = 1e-4 ×
   background moduli/density. Critical slownesses: 1/α = 2.0e-4 s/m, 1/β ≈ 3.33e-4 s/m.

Commands: run tests with
`conda run -n seismic python -m pytest cubic_scattering/tests/<file>::<test> -v`.
After every Python change:
`conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`.
Commits: gitmoji style, NO attribution lines.

---

### Task 1: Pin the Kennett matrix convention (symmetry test)

**Files:**
- Test: `cubic_scattering/tests/test_kennett_layers.py` (append)

- [ ] **Step 1: Write the test**

Append to `cubic_scattering/tests/test_kennett_layers.py`:

```python
class TestModifiedConventionSymmetry:
    """The modified (sqrt(eta*rho)-normalized) reflection matrix is symmetric.

    This is the property that makes the off-diagonal channel comparison
    index-order-proof: RD_psv[0, 1] == RD_psv[1, 0] in the modified
    convention, regardless of which slot means P-in vs P-out.
    """

    def test_rd_psv_symmetric_oblique(self):
        """Two-halfspace contrast at oblique p: RD_psv must be symmetric."""
        stack = LayerStack(
            layers=[
                IsotropicLayer(alpha=5000.0, beta=3000.0, rho=2500.0, thickness=50.0),
                IsotropicLayer(alpha=5400.0, beta=3200.0, rho=2600.0, thickness=np.inf),
            ]
        )
        p = 1.0e-4  # oblique, sub-critical (1/alpha = 2e-4)
        result = kennett_layers(stack, p=p, omega=np.array([150.0]))
        RD = result.RD_psv[0]
        assert abs(RD[0, 1]) > 1e-6, "expected nonzero conversion at oblique p"
        np.testing.assert_allclose(RD[0, 1], RD[1, 0], rtol=1e-8)
```

- [ ] **Step 2: Run the test**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_kennett_layers.py::TestModifiedConventionSymmetry -v`
Expected: PASS (the recursion is built on the modified coefficients). If it FAILS,
STOP and report — the conversion formula in Convention 4 would need rederiving and
every later task depends on it.

- [ ] **Step 3: Commit**

```bash
git add cubic_scattering/tests/test_kennett_layers.py
git commit -m "✅ test: pin modified-convention symmetry of Kennett P-SV reflection matrix"
```

---

### Task 2: Shared Weyl extractor `slab_weyl_amplitudes`

**Files:**
- Modify: `cubic_scattering/slab_scattering.py` (add dataclass + function before
  `slab_rpp_periodic`, then rewrite `slab_rpp_periodic` as a wrapper)
- Test: `cubic_scattering/tests/test_slab_scattering.py` (append)

- [ ] **Step 1: Write the failing regression test**

Append to `cubic_scattering/tests/test_slab_scattering.py`:

```python
class TestWeylAmplitudes:
    """slab_weyl_amplitudes: shared specular extractor for P/SV/SH."""

    def _solve_uniform(self, p, wave_type="P"):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        contrast = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
        geom = SlabGeometry(M=8, N_z=1, a=2.0)
        mat = uniform_slab_material(geom, ref, contrast)
        omega = 150.0
        c = ref.alpha if wave_type == "P" else ref.beta
        eta = np.sqrt(1.0 / c**2 - p**2 + 0j)
        k_hat = np.array([float(np.real(eta * c)), p * c, 0.0])
        result = compute_slab_scattering(
            geom, mat, omega, k_hat, wave_type=wave_type, periodic=True
        )
        T_local = compute_slab_tmatrices(geom, mat, omega)
        return result, T_local

    def test_rp_matches_rpp_periodic_normal_incidence(self):
        """At p=0 the new extractor reproduces slab_rpp_periodic exactly."""
        result, T_local = self._solve_uniform(p=0.0)
        amps = slab_weyl_amplitudes(result, T_local, p=0.0)
        legacy = slab_rpp_periodic(result, T_local, p=0.0)
        np.testing.assert_allclose(amps.R_P, legacy, rtol=1e-12)

    def test_conversions_vanish_at_normal_incidence(self):
        """P incidence at p=0: no SV or SH specular conversion."""
        result, T_local = self._solve_uniform(p=0.0)
        amps = slab_weyl_amplitudes(result, T_local, p=0.0)
        assert abs(amps.R_SV) < 1e-10 * max(abs(amps.R_P), 1e-30)
        assert abs(amps.R_SH) < 1e-10 * max(abs(amps.R_P), 1e-30)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py::TestWeylAmplitudes -v`
Expected: FAIL with `ImportError`/`NameError: slab_weyl_amplitudes` (add the import to
the test file's existing `from cubic_scattering.slab_scattering import (...)` block:
`slab_weyl_amplitudes`).

- [ ] **Step 3: Implement the extractor**

In `cubic_scattering/slab_scattering.py`, insert immediately above `slab_rpp_periodic`:

```python
@dataclass
class WeylAmplitudes:
    """Specular Weyl amplitudes from one periodic-slab solve.

    Displacement-amplitude convention (unit-displacement incident wave).
    Convert to the Kennett modified convention with sqrt(eta_out/eta_in)
    before comparing off-diagonal channels (see SlabReflectionMatrix).

    Attributes:
        R_P: Outgoing specular P amplitude.
        R_SV: Outgoing specular SV amplitude (sagittal polarisation).
        R_SH: Outgoing specular SH amplitude (y polarisation).
        p: Horizontal slowness used (s/m).
        eta_P: Vertical P slowness (complex past critical).
        eta_S: Vertical S slowness (complex past critical).
    """

    R_P: complex
    R_SV: complex
    R_SH: complex
    p: float
    eta_P: complex
    eta_S: complex


def slab_weyl_amplitudes(
    result: SlabResult, T_local: NDArray, *, p: float = 0.0
) -> WeylAmplitudes:
    """Extract all specular outgoing amplitudes (P, SV, SH) via Weyl sums.

    The 2D lattice sum replaces exp(ikr)/(4πr) with i/(2k_z d²)·exp(ik_z|z|)
    per mode. The source coupling uses the full reflected wave vector
    ω·s⃗_m = k_m·d̂_m (NOT the vertical wavenumber — they differ at p>0):

        Q_P  = −d̂_P·f − iω (s⃗_P·σ·d̂_P)            (scalar)
        Q⃗_S  = −f − iω (σ·s⃗_S)                      (vector)
        R_m  = −i/(2 ω η_m d² ρ c_m²) Σ_l Q_m,l exp(iω η_m z_l)

    with the SV/SH amplitudes the ŝv/ŝh projections of Q⃗_S. The force term
    is negated (T-matrix +ω²Δρ V u convention, opposite to the
    Lippmann-Schwinger body force). Sources are averaged over the M²
    horizontal cubes per layer (specular/coherent response).

    Args:
        result: Solved slab scattering result (use periodic=True).
        T_local: Per-cube T-matrices, shape (N_z, M, M, 9, 9).
        p: Horizontal slowness (s/m).

    Returns:
        WeylAmplitudes in the displacement convention.
    """
    geom = result.geometry
    ref = result.material.ref
    omega = result.omega
    d = geom.d

    eta_P = _vertical_slowness(_complex_slowness(ref.alpha, np.inf), p)
    eta_S = _vertical_slowness(_complex_slowness(ref.beta, np.inf), p)
    kz_P = omega * eta_P
    kz_S = omega * eta_S

    # Upgoing slowness vectors, complex-unit directions, S polarisations
    s_vec_P = np.array([-eta_P, p, 0.0], dtype=complex)
    s_vec_S = np.array([-eta_S, p, 0.0], dtype=complex)
    d_P = ref.alpha * s_vec_P
    sv_hat = ref.beta * np.array([p, eta_S, 0.0], dtype=complex)
    sh_hat = np.array([0.0, 0.0, 1.0])

    source = np.einsum("lmnab,lmnb->lmna", T_local, result.psi)
    centres = geom.all_centres()

    tot_P = 0.0 + 0.0j
    tot_SV = 0.0 + 0.0j
    tot_SH = 0.0 + 0.0j
    for lz in range(geom.N_z):
        f_avg = np.mean(source[lz, :, :, :3], axis=(0, 1))
        sig_avg = _voigt_to_tensor(np.mean(source[lz, :, :, 3:], axis=(0, 1)))
        z_l = centres[lz, 0, 0, 0]

        Q_P = -np.dot(d_P, f_avg) - 1j * omega * np.dot(s_vec_P, sig_avg @ d_P)
        tot_P += Q_P * np.exp(1j * kz_P * z_l)

        Q_S = -f_avg - 1j * omega * (sig_avg @ s_vec_S)
        phase_S = np.exp(1j * kz_S * z_l)
        tot_SV += np.dot(sv_hat, Q_S) * phase_S
        tot_SH += np.dot(sh_hat, Q_S) * phase_S

    pref_P = -1j / (2.0 * kz_P * d**2 * ref.rho * ref.alpha**2)
    pref_S = -1j / (2.0 * kz_S * d**2 * ref.rho * ref.beta**2)

    return WeylAmplitudes(
        R_P=complex(pref_P * tot_P),
        R_SV=complex(pref_S * tot_SV),
        R_SH=complex(pref_S * tot_SH),
        p=p,
        eta_P=complex(eta_P),
        eta_S=complex(eta_S),
    )
```

`dataclass` is already imported at the top of the module.

- [ ] **Step 4: Rewrite `slab_rpp_periodic` as a wrapper**

Replace the entire body of `slab_rpp_periodic` (keep signature and docstring, append
the note below to the docstring):

```python
def slab_rpp_periodic(
    result: SlabResult, T_local: NDArray, *, p: float = 0.0
) -> complex:
    """... (keep existing docstring) ...

    Note:
        Delegates to slab_weyl_amplitudes. The oblique stress coupling now
        uses the full wave vector (−iω s⃗·σ·d̂); the previous −i k_z σ_rr form
        was correct only at p=0.
    """
    return slab_weyl_amplitudes(result, T_local, p=p).R_P
```

- [ ] **Step 5: Run the new tests and the existing slab suite**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py cubic_scattering/tests/test_slab_convergence.py -q`
Expected: ALL PASS. The p=0 regression test passes because the old and new stress
couplings coincide at normal incidence; all existing Kennett comparisons are at p=0.

- [ ] **Step 6: Lint + commit**

```bash
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
git add cubic_scattering/slab_scattering.py cubic_scattering/tests/test_slab_scattering.py
git commit -m "✨ feat: shared Weyl extractor for P/SV/SH specular slab amplitudes"
```

---

### Task 3: SH incidence in the slab incident-field builder

**Files:**
- Modify: `cubic_scattering/slab_scattering.py` (`_build_slab_incident_field`, ~line 426;
  `compute_slab_scattering` docstring ~line 509)
- Test: `cubic_scattering/tests/test_slab_scattering.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestSHIncidence:
    """SH plane-wave incidence (polarisation ŷ)."""

    def test_sh_polarisation_is_y(self):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        geom = SlabGeometry(M=4, N_z=1, a=2.0)
        k_hat = np.array([0.8, 0.6, 0.0])
        psi0 = _build_slab_incident_field(geom, 150.0, ref, k_hat, "SH")
        # Displacement components: only u_y nonzero
        assert np.max(np.abs(psi0[..., 0])) < 1e-14
        assert np.max(np.abs(psi0[..., 1])) < 1e-14
        assert np.max(np.abs(psi0[..., 2])) > 0.99

    def test_unknown_wave_type_fails_fast(self):
        ref = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
        geom = SlabGeometry(M=4, N_z=1, a=2.0)
        with pytest.raises(ValueError, match="wave_type"):
            _build_slab_incident_field(
                geom, 150.0, ref, np.array([1.0, 0.0, 0.0]), "X"
            )
```

(Import `_build_slab_incident_field` in the test file's import block.)

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py::TestSHIncidence -v`
Expected: FAIL — `ValueError: wave_type must be 'P' or 'S', got 'SH'`.

- [ ] **Step 3: Implement**

In `_build_slab_incident_field`, change the `elif`/`else` chain to:

```python
    if wave_type == "P":
        k_mag = omega / ref.alpha
        pol = k_hat.copy()
    elif wave_type == "S":
        k_mag = omega / ref.beta
        # SV polarisation: in vertical plane, perpendicular to k_hat
        z_hat = np.array([1.0, 0.0, 0.0])
        cross = np.cross(k_hat, z_hat)
        if np.linalg.norm(cross) < 1e-10:
            # Vertical incidence — pick x-direction
            pol = np.array([0.0, 1.0, 0.0])
        else:
            pol = np.cross(cross, k_hat)
            pol = pol / np.linalg.norm(pol)
    elif wave_type == "SH":
        k_mag = omega / ref.beta
        # SH polarisation: horizontal, perpendicular to the sagittal plane
        pol = np.array([0.0, 0.0, 1.0])
    else:
        msg = f"wave_type must be 'P', 'S', or 'SH', got '{wave_type}'"
        raise ValueError(msg)
```

Update the `wave_type` doc lines in `_build_slab_incident_field` and
`compute_slab_scattering` to `'P', 'S' (SV), or 'SH'`.

- [ ] **Step 4: Run tests**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py::TestSHIncidence -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
git add cubic_scattering/slab_scattering.py cubic_scattering/tests/test_slab_scattering.py
git commit -m "✨ feat: SH plane-wave incidence for the slab solver"
```

---

### Task 4: `SlabReflectionMatrix`, `kennett_reference_matrix`, and channel validation

**Files:**
- Modify: `cubic_scattering/slab_scattering.py` (add after `slab_rpp_periodic`;
  make `kennett_reference_rpp` delegate)
- Modify: `cubic_scattering/__init__.py` (export new names)
- Test: `cubic_scattering/tests/test_slab_scattering.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestSlabReflectionMatrix:
    """Full specular matrix vs Kennett for a uniform slab.

    Modified (energy-normalized) convention throughout — the convention
    in which the Kennett matrix is symmetric (TestModifiedConventionSymmetry).
    """

    REF = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
    CONTRAST = MaterialContrast(Dlambda=2.0e9, Dmu=1.0e9, Drho=100.0)
    OMEGA = 150.0

    def _slab_matrix(self, p, M=8, N_z=1, a=2.0, contrast=None):
        contrast = contrast or self.CONTRAST
        geom = SlabGeometry(M=M, N_z=N_z, a=a)
        mat = uniform_slab_material(geom, self.REF, contrast)
        return slab_reflection_matrix(geom, mat, self.OMEGA, p=p)

    def test_normal_incidence_diagonal(self):
        """p=0: R_PP and R_SS match Kennett; conversions vanish; R_SS == R_SH."""
        p = 0.0
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 0], kref.R_PP, rtol=0.05)
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.05)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.05)
        # SH and SS degenerate at normal incidence
        np.testing.assert_allclose(slab.R_sh, R_mod[1, 1], rtol=1e-6)
        assert abs(R_mod[0, 1]) < 0.01 * abs(R_mod[0, 0])
        assert abs(R_mod[1, 0]) < 0.01 * abs(R_mod[1, 1])

    def test_oblique_all_channels(self):
        """Sub-critical oblique p: all five channels vs Kennett."""
        p = 1.0e-4  # sin(theta_P) = 0.5
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 0], kref.R_PP, rtol=0.07)
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.07)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.07)
        # Off-diagonals: nonzero and matching (Kennett symmetric, so either index)
        assert abs(kref.R_PS) > 1e-4
        np.testing.assert_allclose(R_mod[1, 0], kref.R_PS, rtol=0.10)
        np.testing.assert_allclose(R_mod[0, 1], kref.R_SP, rtol=0.10)

    def test_modified_matrix_symmetric(self):
        """Reciprocity: the slab's modified matrix is symmetric like Kennett's."""
        slab = self._slab_matrix(p=1.0e-4)
        R_mod = slab.to_modified()
        np.testing.assert_allclose(R_mod[0, 1], R_mod[1, 0], rtol=0.05)

    def test_born_scaling_conversion_channel(self):
        """Weak contrast: doubling the contrast doubles R_PS (Born linearity)."""
        weak = MaterialContrast(
            Dlambda=self.REF.mu * 1e-4,
            Dmu=self.REF.mu * 1e-4,
            Drho=self.REF.rho * 1e-4,
        )
        weak2 = MaterialContrast(
            Dlambda=2 * weak.Dlambda, Dmu=2 * weak.Dmu, Drho=2 * weak.Drho
        )
        p = 1.0e-4
        r1 = self._slab_matrix(p, contrast=weak).to_modified()[1, 0]
        r2 = self._slab_matrix(p, contrast=weak2).to_modified()[1, 0]
        np.testing.assert_allclose(r2 / r1, 2.0, rtol=0.05)

    def test_post_critical_smoke(self):
        """p past the P-critical slowness: finite, branch-consistent R_SS."""
        p = 2.5e-4  # > 1/alpha = 2e-4 (P evanescent), < 1/beta (SV propagating)
        slab = self._slab_matrix(p)
        kref = kennett_reference_matrix(
            self.REF, self.CONTRAST, H=4.0, omega=self.OMEGA, p=p
        )
        R_mod = slab.to_modified()
        assert np.isfinite(R_mod).all()
        np.testing.assert_allclose(R_mod[1, 1], kref.R_SS, rtol=0.15)
        np.testing.assert_allclose(slab.R_sh, kref.R_SH, rtol=0.15)
```

(Add `slab_reflection_matrix`, `kennett_reference_matrix` to the test imports.)

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py::TestSlabReflectionMatrix -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

In `cubic_scattering/slab_scattering.py`, after `slab_rpp_periodic`:

```python
@dataclass
class SlabReflectionMatrix:
    """Specular reflection matrix of the periodic heterogeneous slab.

    Attributes:
        R_psv: 2×2 P-SV matrix, displacement convention.
            Rows = outgoing mode (0=P, 1=SV); columns = incident mode.
        R_sh: SH→SH coefficient.
        p: Horizontal slowness (s/m).
        omega: Angular frequency (rad/s).
        eta_P: Vertical P slowness in the background.
        eta_S: Vertical S slowness in the background.
    """

    R_psv: NDArray[np.complexfloating]
    R_sh: complex
    p: float
    omega: float
    eta_P: complex
    eta_S: complex
    n_gmres_iters: tuple[int, int, int] = (0, 0, 0)

    def to_modified(self) -> NDArray[np.complexfloating]:
        """Convert to the Kennett modified (energy-normalized) convention.

        R̃_ij = sqrt(η_i/η_j) R_ij (same medium above and below the datum,
        so the densities cancel). Diagonal entries are unchanged; the
        modified matrix is symmetric by reciprocity.
        """
        n = np.array([np.sqrt(self.eta_P), np.sqrt(self.eta_S)])
        return self.R_psv * np.outer(n, 1.0 / n)


def slab_reflection_matrix(
    geometry: SlabGeometry,
    material: SlabMaterial,
    omega: float,
    *,
    p: float = 0.0,
    gmres_tol: float = 1e-6,
    max_iter: int = 500,
    volume_averaged: bool = False,
    n_orders: int = 2,
    include_sh: bool = True,
) -> SlabReflectionMatrix:
    """Full specular reflection matrix via three periodic Foldy-Lax solves.

    Runs P-, SV-, and SH-incident solves at the same horizontal slowness p
    (incident direction per mode: k̂_m = (η_m c_m, p c_m, 0)) and assembles
    the 2×2 P-SV matrix plus the SH coefficient from the shared Weyl
    extractor. SH decouples from P-SV in the horizontally averaged
    (specular) response; the SH-incident solve only populates R_sh.

    Args:
        geometry: Slab lattice geometry.
        material: Per-cube material contrasts.
        omega: Angular frequency (rad/s).
        p: Horizontal slowness (s/m).
        gmres_tol: GMRES relative tolerance.
        max_iter: Maximum GMRES iterations.
        volume_averaged: Use volume-averaged inter-voxel propagator.
        n_orders: Dynamic correction orders for the volume-averaged propagator.

    Returns:
        SlabReflectionMatrix (displacement convention; use .to_modified()
        for Kennett comparison and recursion mixing).
    """
    ref = material.ref
    eta_P = _vertical_slowness(_complex_slowness(ref.alpha, np.inf), p)
    eta_S = _vertical_slowness(_complex_slowness(ref.beta, np.inf), p)
    k_hat_P = np.array(
        [float(np.real(eta_P * ref.alpha)), p * ref.alpha, 0.0]
    )
    k_hat_S = np.array(
        [float(np.real(eta_S * ref.beta)), p * ref.beta, 0.0]
    )

    T_local = compute_slab_tmatrices(geometry, material, omega)

    incidences = [("P", k_hat_P), ("S", k_hat_S)]
    if include_sh:
        incidences.append(("SH", k_hat_S))

    amps: dict[str, WeylAmplitudes] = {}
    iters: dict[str, int] = {"P": 0, "S": 0, "SH": 0}
    for wave_type, k_hat in incidences:
        result = compute_slab_scattering(
            geometry,
            material,
            omega,
            k_hat,
            wave_type=wave_type,
            gmres_tol=gmres_tol,
            max_iter=max_iter,
            periodic=True,
            volume_averaged=volume_averaged,
            n_orders=n_orders,
        )
        amps[wave_type] = slab_weyl_amplitudes(result, T_local, p=p)
        iters[wave_type] = result.n_gmres_iter

    R_psv = np.array(
        [
            [amps["P"].R_P, amps["S"].R_P],
            [amps["P"].R_SV, amps["S"].R_SV],
        ],
        dtype=complex,
    )
    return SlabReflectionMatrix(
        R_psv=R_psv,
        R_sh=amps["SH"].R_SH if include_sh else 0.0j,
        p=p,
        omega=omega,
        eta_P=complex(eta_P),
        eta_S=complex(eta_S),
        n_gmres_iters=(iters["P"], iters["S"], iters["SH"]),
    )


@dataclass
class KennettChannelReference:
    """All five Kennett reflection channels for a uniform 3-layer stack.

    Modified (energy-normalized) convention, as stored by kennett_layers.
    """

    R_PP: complex
    R_PS: complex
    R_SP: complex
    R_SS: complex
    R_SH: complex


def kennett_reference_matrix(
    ref: ReferenceMedium,
    contrast: MaterialContrast,
    H: float,
    omega: float,
    *,
    p: float = 0.0,
) -> KennettChannelReference:
    """Kennett reference for all five channels of a uniform layer.

    Same 3-layer stack as kennett_reference_rpp:
    background(dummy) | perturbed(H) | background(halfspace), at slowness p.

    Args:
        ref: Background elastic medium.
        contrast: Material contrast defining the perturbed layer.
        H: Layer thickness (m).
        omega: Angular frequency (rad/s).
        p: Horizontal slowness (s/m).

    Returns:
        KennettChannelReference with complex coefficients.
    """
    lam_bg = ref.rho * (ref.alpha**2 - 2.0 * ref.beta**2)
    mu_bg = ref.rho * ref.beta**2
    lam_p = lam_bg + contrast.Dlambda
    mu_p = mu_bg + contrast.Dmu
    rho_p = ref.rho + contrast.Drho
    alpha_p = float(np.sqrt((lam_p + 2.0 * mu_p) / rho_p))
    beta_p = float(np.sqrt(mu_p / rho_p))

    stack = LayerStack(
        layers=[
            IsotropicLayer(
                alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=100.0
            ),
            IsotropicLayer(alpha=alpha_p, beta=beta_p, rho=rho_p, thickness=H),
            IsotropicLayer(
                alpha=ref.alpha, beta=ref.beta, rho=ref.rho, thickness=np.inf
            ),
        ]
    )
    result = kennett_layers(stack, p=p, omega=np.array([omega]))
    return KennettChannelReference(
        R_PP=complex(result.RPP[0]),
        R_PS=complex(result.RPS[0]),
        R_SP=complex(result.RSP[0]),
        R_SS=complex(result.RSS[0]),
        R_SH=complex(result.RSH[0]),
    )
```

Make `kennett_reference_rpp` delegate (replace its stack-building body):

```python
    return kennett_reference_matrix(ref, contrast, H, omega, p=0.0).R_PP
```

Export in `cubic_scattering/__init__.py` (alongside the existing slab exports):
`WeylAmplitudes`, `slab_weyl_amplitudes`, `SlabReflectionMatrix`,
`slab_reflection_matrix`, `KennettChannelReference`, `kennett_reference_matrix`.

- [ ] **Step 4: Run the validation tests — SIGN-PINNING DECISION POINT**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py::TestSlabReflectionMatrix -v`

Decision procedure (the SV polarisation sign convention is pinned here, not a priori):
- All PASS → conventions correct, proceed.
- `test_oblique_all_channels` fails with off-diagonal ratio ≈ −1 (check by printing
  `R_mod[1,0] / kref.R_PS`) → flip the SV polarisation sign in
  `slab_weyl_amplitudes`: `sv_hat = -ref.beta * np.array([p, eta_S, 0.0], dtype=complex)`,
  re-run, and record the final convention in the `WeylAmplitudes` docstring.
- Off-diagonal magnitude mismatch by a p-dependent factor → the modified-convention
  conversion direction is inverted: change `to_modified` to `np.outer(1.0 / n, n)`,
  re-run. (Only one of these two can be wrong; the Born test plus the symmetry test
  discriminate: symmetry failure = sign/normalization on the slab side, symmetric but
  wrong magnitude = conversion direction.)
- Diagonal failures (R_PP oblique) → re-derive the stress coupling; do NOT loosen
  tolerances. This is the pre-existing oblique bug surface (Convention 3) and the
  whole point of the test.

- [ ] **Step 5: Run full slab + kennett test files**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_slab_scattering.py cubic_scattering/tests/test_kennett_layers.py cubic_scattering/tests/test_slab_convergence.py -q`
Expected: ALL PASS.

- [ ] **Step 6: Lint + commit**

```bash
git add cubic_scattering/slab_scattering.py cubic_scattering/__init__.py cubic_scattering/tests/test_slab_scattering.py
git commit -m "✨ feat: full specular reflection matrix (R_PS/R_SP/R_SS/R_SH) with Kennett validation"
```

---

### Task 5: Ocean-bottom 2×2 sub-ocean recursion

**Files:**
- Modify: `cubic_scattering/ocean_bottom.py` (`_kennett_water_step` ~line 116;
  `compute_ocean_bottom_reflection` ~lines 251–315; `OceanBottomResult` ~line 85;
  module docstring)
- Test: `cubic_scattering/tests/test_ocean_bottom.py` (modify + append)

- [ ] **Step 1: Read the existing degenerate-limit test**

Open `cubic_scattering/tests/test_ocean_bottom.py` and find the test asserting the
decomposed path equals the full Kennett stack at rtol 1e-10 (around line 234). It must
keep passing unchanged after this task — it is the strongest correctness anchor.

- [ ] **Step 2: Write the new failing tests**

Append to `cubic_scattering/tests/test_ocean_bottom.py`:

```python
class TestModeConvertedOceanBottom:
    """2×2 sub-ocean recursion: internal P→S→P feeds the observable R_PP."""

    def _config(self, contrast, p, M=4, N_z=1):
        sed_ref = ReferenceMedium(alpha=2000.0, beta=800.0, rho=1800.0)
        geom = SlabGeometry(M=M, N_z=N_z, a=1.0)
        mat = uniform_slab_material(geom, sed_ref, contrast)
        return OceanBottomConfig(
            water_alpha=1500.0,
            water_rho=1000.0,
            water_depth=100.0,
            sed_ref=sed_ref,
            hs_alpha=3000.0,
            hs_beta=1700.0,
            hs_rho=2200.0,
            geometry=geom,
            material=mat,
            f_peak=25.0,
            T=0.5,
            nw=32,
            f_min=10.0,
            f_max=60.0,
            p=p,
        )

    def test_weak_contrast_matches_scalar_limit(self):
        """Weak contrast at oblique p: conversion is second-order, so the
        2×2 R_total must match the scalar-recursion value to first order."""
        weak = MaterialContrast(Dlambda=1e5, Dmu=1e5, Drho=0.1)
        cfg = self._config(weak, p=2.0e-4)  # oblique, sub-critical in water
        result = compute_ocean_bottom_reflection(cfg, progress=False)
        # Scalar reference: rebuild MT with PP entries only
        R_pp_slab = result.R_slab_psv[:, 0, 0]
        assert result.R_slab_psv.shape[1:] == (2, 2)
        # The PP slab entry must dominate the off-diagonals at weak contrast
        active = np.abs(R_pp_slab) > 0
        off = np.abs(result.R_slab_psv[active][:, 1, 0])
        diag = np.abs(R_pp_slab[active])
        assert np.all(off <= diag + 1e-12)

    def test_moderate_contrast_conversion_changes_rpp(self):
        """Moderate contrast at oblique p: the 2×2 recursion shifts R_total
        relative to a PP-only recursion (the restored physics)."""
        contrast = MaterialContrast(Dlambda=0.5e9, Dmu=0.3e9, Drho=200.0)
        cfg = self._config(contrast, p=2.0e-4)
        result = compute_ocean_bottom_reflection(cfg, progress=False)
        # PP-only comparison: zero out the off-diagonal slab entries and
        # re-run the water step
        from cubic_scattering.ocean_bottom import _kennett_water_step

        active = np.where(np.abs(result.R_total) > 0)[0]
        MT_full = result.MT_psv[active]
        MT_pp_only = MT_full.copy()
        MT_pp_only[:, 0, 1] = 0.0
        MT_pp_only[:, 1, 0] = 0.0
        R_full = _kennett_water_step(MT_full, cfg)
        R_pp_only = _kennett_water_step(MT_pp_only, cfg)
        diff = np.max(np.abs(R_full - R_pp_only))
        scale = np.max(np.abs(R_full))
        assert diff > 1e-6 * scale, "conversion had no effect — physics missing"
        assert diff < 0.5 * scale, "conversion implausibly large"
```

(Imports: extend the existing test-file import block with whatever it lacks:
`MaterialContrast`, `OceanBottomConfig`, `ReferenceMedium`, `SlabGeometry`,
`compute_ocean_bottom_reflection`, `uniform_slab_material`.)

- [ ] **Step 3: Run to verify failure**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_ocean_bottom.py::TestModeConvertedOceanBottom -v`
Expected: FAIL — `R_slab_psv`/`MT_psv` do not exist and `_kennett_water_step` takes a
scalar array.

- [ ] **Step 4: Implement the 2×2 water step**

Replace `_kennett_water_step` in `cubic_scattering/ocean_bottom.py`:

```python
def _kennett_water_step(
    MT_psv: NDArray,
    cfg: OceanBottomConfig,
) -> NDArray:
    """Kennett recursion at the water-sediment interface (2×2 P-SV).

    Computes R = Rd + Tu · MT · (I − Ru · MT)⁻¹ · Td with the fluid-solid
    coefficients (modified convention) and returns the water-side PP
    observable R[0, 0]. The full 2×2 MT keeps P↔SV conversion inside the
    sediment package in the reverberation operator (I − Ru·MT)⁻¹.

    Args:
        MT_psv: Phase-shifted sub-ocean P-SV reflectivity (modified
            convention), shape (nfreq, 2, 2).
        cfg: Ocean-bottom configuration.

    Returns:
        Water-side PP reflection coefficient, shape (nfreq,).
    """
    p = cfg.p

    s_water = _complex_slowness(cfg.water_alpha, np.inf)
    eta_water = _vertical_slowness(s_water, p)
    s_sed_p = _complex_slowness(cfg.sed_ref.alpha, np.inf)
    eta_sed = _vertical_slowness(s_sed_p, p)
    s_sed_s = _complex_slowness(cfg.sed_ref.beta, np.inf)
    neta_sed = _vertical_slowness(s_sed_s, p)
    beta_sed = 1.0 / s_sed_s

    coeff = psv_fluid_solid(
        p, eta_water, cfg.water_rho, eta_sed, neta_sed, cfg.sed_ref.rho, beta_sed
    )

    eye = np.eye(2, dtype=complex)
    out = np.zeros(MT_psv.shape[0], dtype=complex)
    for i in range(MT_psv.shape[0]):
        U = np.linalg.inv(eye - coeff.Ru @ MT_psv[i])
        R = coeff.Rd + coeff.Tu @ MT_psv[i] @ U @ coeff.Td
        out[i] = R[0, 0]
    return out
```

- [ ] **Step 5: Implement the matrix assembly in the compute loop**

In `compute_ocean_bottom_reflection`:

a. Add the S vertical slowness next to the existing P one (after line ~222):

```python
    s_sed_s = _complex_slowness(cfg.sed_ref.beta, np.inf)
    neta_sed_s = _vertical_slowness(s_sed_s, p)
```

b. Replace `R_sed_hs = sub_result.RPP[0]` with the full matrix:

```python
    R_sed_hs_psv = sub_result.RD_psv[0]  # (2, 2), modified convention
```

c. Replace the scalar `E2_sed` with per-mode one-way phases and a phase operator:

```python
    # Per-mode one-way phases through the sediment package;
    # two-way conversion path phase is E_i · E_j (down as j, up as i)
    E_P = np.exp(1j * omega_damped * eta_sed * H)
    E_S = np.exp(1j * omega_damped * neta_sed_s * H)
    E_diag = np.stack([E_P, E_S], axis=-1)  # (nwm, 2)
```

d. Replace the slab solve block (the `R_slab` array, `k_hat`, and the loop) with:

```python
    R_slab_psv = np.zeros((nwm, 2, 2), dtype=complex)
    n_gmres_iters: list[int] = []
    freq_elapsed: list[float] = []

    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(active_indices, desc="Slab scattering", leave=False)
    else:
        iterator = active_indices

    for iw in iterator:
        t_freq = time.perf_counter()
        w = float(omega_real[iw])
        slab = slab_reflection_matrix(
            cfg.geometry,
            cfg.material,
            w,
            p=p,
            gmres_tol=gmres_tol,
            volume_averaged=volume_averaged,
            n_orders=n_orders,
        )
        R_slab_psv[iw] = slab.to_modified()
        n_gmres_iters.append(max(slab.n_gmres_iters))
        freq_elapsed.append(time.perf_counter() - t_freq)
```

The `slab_reflection_matrix` call above must pass `include_sh=False` (add it to the
keyword arguments): SH cannot couple through the fluid, so the third solve would be
wasted. Per-frequency cost is therefore 2 Foldy-Lax solves (P- and SV-incident);
`slab.n_gmres_iters` (defined in Task 4) carries the per-incidence GMRES counts.

e. Replace the `MT_total` line with the matrix version:

```python
    # MT_ij = E_i · R_sed_hs_ij · E_j + R_slab_ij  (modified convention)
    MT_psv = (
        E_diag[:, :, None] * R_sed_hs_psv[None, :, :] * E_diag[:, None, :]
        + R_slab_psv
    )
    R_total = np.zeros(nwm, dtype=complex)
    R_total[active_indices] = _kennett_water_step(MT_psv[active_indices], cfg)
```

f. Extend `OceanBottomResult`: add fields (after `R_slab`):

```python
    R_slab_psv: NDArray[np.complexfloating] = field(
        default_factory=lambda: np.zeros((0, 2, 2), dtype=complex)
    )
    MT_psv: NDArray[np.complexfloating] = field(
        default_factory=lambda: np.zeros((0, 2, 2), dtype=complex)
    )
```

Populate them in the return statement (`R_slab_psv=R_slab_psv, MT_psv=MT_psv`) and
keep the legacy scalar field as the PP entry: `R_slab=R_slab_psv[:, 0, 0]` (the
plotting code in `ocean_bottom/run_study.py` reads `R_slab`). Update the
`OceanBottomResult` docstring fields list.

g. Update the import in `ocean_bottom.py`: replace `slab_rpp_periodic` and
`compute_slab_scattering`/`compute_slab_tmatrices` imports with
`slab_reflection_matrix` (keep the others used elsewhere). Update the module
docstring paragraph about "the fluid-solid interface PP path" to describe the 2×2
recursion.

- [ ] **Step 6: Run the new tests and the full ocean-bottom file**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_ocean_bottom.py -v`
Expected: ALL PASS, including the pre-existing degenerate-limit test at rtol 1e-10
(zero contrast → R_slab_psv = 0 and MT reduces to the phased background matrix; the
2×2 water step must reproduce the full-stack Kennett RPP exactly because the same
matrix recursion runs inside `kennett_layers`). If the degenerate test fails, the
phase operator (step c/e) disagrees with `_kennett_psv_recursion`'s internal phase
accounting — compare against `kennett_layers.py` lines ~549–665 and fix the phase
convention (one-way vs two-way, layer datum) rather than adjusting tolerances.

- [ ] **Step 7: Lint + commit**

```bash
git add cubic_scattering/ocean_bottom.py cubic_scattering/tests/test_ocean_bottom.py cubic_scattering/slab_scattering.py cubic_scattering/tests/test_slab_scattering.py
git commit -m "✨ feat: 2×2 P-SV sub-ocean recursion — internal mode conversion feeds R_PP"
```

---

### Task 6: Log diagnostics + docs

**Files:**
- Modify: `cubic_scattering/ocean_bottom.py` (`write_log`, ~lines 449–462)
- Modify: `ocean_bottom/README.md` (caveats section)
- Test: `cubic_scattering/tests/test_ocean_bottom.py` (extend existing write_log test)

- [ ] **Step 1: Extend the per-frequency log table**

In `write_log`, replace the per-frequency header and row writes with:

```python
        f.write("## Per-frequency diagnostics\n\n")
        f.write(f"{'freq_Hz':>10s}  {'|R_PP|':>10s}  {'|R_PS|':>10s}  ")
        f.write(f"{'|R_SS|':>10s}  {'|R_total|':>10s}  ")
        f.write(f"{'GMRES_iter':>10s}  {'elapsed_ms':>10s}\n")
        f.write("-" * 80 + "\n")

        for i, iw in enumerate(active_indices):
            fhz = freq_hz[iw]
            r_pp = abs(result.R_slab_psv[iw, 0, 0])
            r_ps = abs(result.R_slab_psv[iw, 1, 0])
            r_ss = abs(result.R_slab_psv[iw, 1, 1])
            r_total = abs(result.R_total[iw])
            gm = result.n_gmres_iters[i] if i < len(result.n_gmres_iters) else -1
            dt = result.freq_elapsed[i] * 1e3 if i < len(result.freq_elapsed) else -1
            f.write(f"{fhz:10.2f}  {r_pp:10.2e}  {r_ps:10.2e}  ")
            f.write(f"{r_ss:10.2e}  {r_total:10.2e}  ")
            f.write(f"{gm:10d}  {dt:10.2f}\n")
        f.write("\n")
```

And in the Summary section add after the existing peak lines:

```python
        f.write(f"Peak |R_PS|:   {np.max(np.abs(result.R_slab_psv[:, 1, 0])):.6f}\n")
        f.write(f"Peak |R_SS|:   {np.max(np.abs(result.R_slab_psv[:, 1, 1])):.6f}\n")
```

- [ ] **Step 2: Update the existing write_log test**

Find the existing `write_log` test in `test_ocean_bottom.py`; extend its assertions:

```python
        text = log_path.read_text()
        assert "|R_PS|" in text
        assert "|R_SS|" in text
```

- [ ] **Step 3: Update `ocean_bottom/README.md`**

In the caveats / not-yet-supported section, replace the "Only PP reflection
extracted" caveat with:

```markdown
- The sub-ocean recursion is the full 2×2 P-SV matrix: P→S→P conversion inside
  the heterogeneous slab and sediment package contributes to the observed R_PP.
  The slab-level channels (R_PS, R_SP, R_SS in the modified convention) are
  reported in the log file. SH cannot couple through the water column and is
  not computed in this study (use `slab_reflection_matrix` directly for SH).
- Per-frequency cost is 2 Foldy-Lax solves (P- and SV-incident).
```

- [ ] **Step 4: Run tests, lint, commit**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_ocean_bottom.py -q
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
git add cubic_scattering/ocean_bottom.py cubic_scattering/tests/test_ocean_bottom.py ocean_bottom/README.md
git commit -m "📝 docs: mode-conversion diagnostics in ocean-bottom log and README"
```

---

### Task 7: Full-suite gate

**Files:** none new.

- [ ] **Step 1: Run the complete test suite**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/ -q`
Expected: ALL PASS (626+ tests including the 5 slow sphere tests; budget ~10 min).

- [ ] **Step 2: Final lint/type sweep**

```bash
conda run -n seismic ruff check cubic_scattering/ --ignore ARG001,ARG002,F841,E741
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
```

Expected: clean. If anything was missed in earlier tasks, fix and amend the relevant
commit message convention (new commit, no force-push).

- [ ] **Step 3: Run one end-to-end ocean-bottom study as a smoke test**

```bash
conda run -n seismic python ocean_bottom/run_study.py ocean_bottom/example_config_weak.yml --p 0.05
```

Expected: completes without error; the log file shows the new |R_PS|/|R_SS| columns
with small values (weak contrast) and |R_total| consistent with previous runs at the
1e-3 level (weak conversion is second-order).

---

## Self-review notes

- Spec coverage: extractor (Task 2), SH incidence (Task 3), matrix + Kennett
  reference + all five validation criteria (Task 4: diagonal, off-diagonal, Born,
  symmetry/reciprocity, post-critical), ocean-bottom 2×2 + degenerate limit +
  physics-gain regression (Task 5), diagnostics/docs (Task 6), full gate (Task 7).
  The spec's "oblique stress coupling" realism requirement maps to Convention 3 +
  Task 4 Step 4.
- The sign-pinning decision point (Task 4 Step 4) is a specified two-outcome
  procedure with exact one-line remedies, not a placeholder.
- Type consistency: `WeylAmplitudes`, `SlabReflectionMatrix.to_modified()`,
  `KennettChannelReference`, `R_slab_psv`/`MT_psv` names match across Tasks 2-6.
  `include_sh` is declared in Task 4 (builder) and consumed in Task 5.
