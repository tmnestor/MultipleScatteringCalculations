# Marine3D Phase 2 — 2½-D Disorder-Resolved Crust — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a per-voxel contrast field on the Marine3D crust planes and resolve the resulting multiple scattering with a matrix-free GMRES matvec, validated against an independently re-derived Wolfram reference and recorded in a LaTeX companion note.

**Architecture:** Heterogeneity lives entirely in per-voxel 9×9 $(u,\varepsilon)$ $T_0$ screens on a periodic $x$-lattice; the reference stack stays exactly as Phase 1 left it. Each GMRES iterate applies $\Delta\mathcal{C}_{eff}$ locally in real space, then $\mathbf{P}^z$ (inter-plane + self-plane reverberation, diagonal in $k_x$, applied as Riccati sweeps) and $\mathbf{P}^x$ (direct intra-plane, circulant in $x$). The deliverable is a dressed `RRd_PP(p, ω)` at the seabed datum that drops straight into the Phase-1 gather pipeline.

**Tech Stack:** Python 3.12, NumPy, SciPy, conda env `seismic`, pytest; Wolfram Mathematica (`wolframscript`); LuaLaTeX.

**Spec:** [`2026-07-25-marine25d-disorder-resolved-design.md`](../specs/2026-07-25-marine25d-disorder-resolved-design.md)

## Global Constraints

- **Work repo:** `~/Desktop/Marine3D`. Self-contained — zero imports from `cubic_scattering`, `GlobalMatrix`, `Kennett_Reflectivity`. Verify: `grep -rE "cubic_scattering|GlobalMatrix|Kennett_Reflectivity" marine3d/` → no hits.
- **LaTeX repo:** `~/Desktop/MultipleScatteringCalculations/LatexPDFs/`. Wolfram: `~/Desktop/MultipleScatteringCalculations/Mathematica/`.
- **Units:** seismic — km/s, g/cm³, km, GPa. **Time convention $e^{-i\omega t}$.**
- **Coordinates:** z = axis 0 (down), x = axis 1, y = axis 2.
- **9-vector basis (fixed, non-negotiable):** `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)` — the ordering `layered_greens_9x9` documents. Every 9×9 object in this plan is in this basis.
- **2½-D:** heterogeneity varies in $(z,x)$, invariant in $y$. All Green's functions evaluated at `ky = 0`.
- **YAML is the single source of truth.** Every key required; no Python-side defaults; no silent fallbacks.
- **Fail-fast with the 4-element diagnostic:** what is wrong / where (absolute path + dotted key) / a valid example / the recovery step.
- **B904:** `raise ... from None` or `from err` inside `except` blocks.
- **Lint/type:** `ruff check --fix --ignore ARG001,ARG002,F841,E741`, `ruff format`, `mypy --ignore-missing-imports`. Line length ≤ 108.
- **Tests** are local-only (`Marine3D/tests/` is gitignored), mirror source layout, run via `conda run -n seismic`.
- **Commits:** gitmoji conventional style, **no Claude attribution**, never `--no-verify`.
- **Full suite** (two modules need un-ported deps): `conda run -n seismic python -m pytest tests/ -q --ignore=tests/test_resonance_far_field.py --ignore=tests/test_inter_voxel_propagator.py` → expect `468 passed, 7 skipped` before any Phase-2 tests are added.

---

## File Structure

| File | Responsibility |
|---|---|
| `marine3d/crust_field.py` | **Create.** Per-voxel $(\Delta\lambda,\Delta\mu,\Delta\rho)$ field → per-voxel 9×9 $T_0$; validity-floor enforcement. No solver, no Green's functions. |
| `marine3d/intraplane_px.py` | **Create.** $\mathbf{P}^x$: direct intra-plane coupling, circulant in $x$, self-term excluded. |
| `marine3d/vertical_pz.py` | **Create.** $\mathbf{P}^z$: inter-plane coupling **and** self-plane reverberation, diagonal in $k_x$. Owns the $G_{ii}$ subtraction. |
| `marine3d/matvec_25d.py` | **Create.** $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})$ apply; GMRES driver; Born as one iterate. |
| `marine3d/reflectivity_25d.py` | **Create.** Outer loop over incident $p$ and $\omega$; specular ($n{=}0$) extraction; diffraction gate; dressed `RRd_PP`. |
| `marine3d/tmatrix/resonance_tmatrix.py` | **Modify.** Promote `_sub_cell_tmatrix_9x9` to public `sub_cell_tmatrix_9x9`, keeping the private name as an alias. |
| `marine3d/config.py` | **Modify.** Add the `crust_heterogeneity:` and `solver:` config blocks with fail-fast validation. |
| `configs/marine_reference.yaml` | **Modify.** Add the new required keys with explicit values. |
| `Mathematica/Marine25DReference.wl` | **Create.** Independent direct-inversion 2½-D reference; emits a JSON fixture. |
| `LatexPDFs/Marine25DDisorderResolved/Marine25DDisorderResolved.tex` | **Create.** The companion note — the product. |

**Stage ordering.** Stage A (Tasks 1–3) needs no solver and closes validation rungs 3 and the basis gate. Stage B (Tasks 4–5) builds the solver and closes rungs 1, 6, 7. Stage C (Tasks 6–7) builds the arbiter and closes rungs 2, 4, 5, 8, 9. Stage D (Task 8) writes the note with the measured numbers.

---

# Stage A — Field and operators (no solver)

## Task 1: Per-voxel contrast field and 9×9 $T_0$

**Files:**
- Create: `marine3d/crust_field.py`
- Modify: `marine3d/tmatrix/resonance_tmatrix.py` (promote `_sub_cell_tmatrix_9x9`)
- Test: `tests/test_crust_field.py`

**Interfaces:**
- Consumes: `marine3d.tmatrix.effective_contrasts.{ReferenceMedium, MaterialContrast, compute_cube_tmatrix}`, `marine3d.tmatrix.voigt_tmatrix.effective_stiffness_voigt`.
- Produces:
  - `sub_cell_tmatrix_9x9(rayleigh: CubeTMatrixResult, omega: float, a_sub: float) -> NDArray` — public alias of the existing private helper, shape `(9, 9)`.
  - `@dataclass CrustField` with fields `n_z: int`, `n_x: int`, `pitch: float`, `dlambda: NDArray`, `dmu: NDArray`, `drho: NDArray` (each shape `(n_z, n_x)`), `reference: ReferenceMedium`.
  - `CrustField.tmatrices(omega: float) -> NDArray` — shape `(n_z, n_x, 9, 9)`.
  - `build_crust_field(...) -> CrustField` and `check_validity_floor(field: CrustField, phi: float) -> None`.

- [ ] **Step 1: Promote the 9×9 sub-cell helper to public**

In `marine3d/tmatrix/resonance_tmatrix.py`, rename `_sub_cell_tmatrix_9x9` to `sub_cell_tmatrix_9x9`, then immediately after the function body add the backward-compatible alias so existing internal callers keep working:

```python
# Backward-compatible private alias (internal callers predate the promotion).
_sub_cell_tmatrix_9x9 = sub_cell_tmatrix_9x9
```

Add `"sub_cell_tmatrix_9x9"` to the module's `__all__` if one is present.

- [ ] **Step 2: Write the failing basis-convention test**

This is the gate that the $T_0$ Voigt ordering matches the Green's-function 9-vector basis. Do **not** skip it — a silent permutation here would corrupt every downstream result while still producing plausible numbers.

```python
# tests/test_crust_field.py
import numpy as np
import pytest

from marine3d.crust_field import CrustField, build_crust_field, check_validity_floor
from marine3d.tmatrix.effective_contrasts import MaterialContrast, ReferenceMedium, compute_cube_tmatrix
from marine3d.tmatrix.resonance_tmatrix import sub_cell_tmatrix_9x9

# Seismic units: km/s, g/cm³, GPa, km.
REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.6)
OMEGA = 2.0 * np.pi * 5.0
A_SUB = 0.01  # km half-width


class TestBasisConvention:
    """T0's 9-vector basis must be the one layered_greens_9x9 documents."""

    def test_t0_is_block_diagonal_3_plus_6(self):
        """Displacement and strain blocks do not mix: T0[0:3, 3:9] == 0."""
        rayleigh = compute_cube_tmatrix(
            omega=OMEGA, a=A_SUB, ref=REF,
            contrast=MaterialContrast(Dlambda=0.2, Dmu=0.1, Drho=0.05),
        )
        T = sub_cell_tmatrix_9x9(rayleigh, OMEGA, A_SUB)
        assert T.shape == (9, 9), f"T0 shape {T.shape}, expected (9, 9)"
        assert np.allclose(T[0:3, 3:9], 0.0), "displacement-strain coupling must vanish"
        assert np.allclose(T[3:9, 0:3], 0.0), "strain-displacement coupling must vanish"

    def test_displacement_block_is_isotropic_density_term(self):
        """T0[0:3, 0:3] == ω²·Δρ*·V·I₃ — isotropic, so basis order cannot matter."""
        rayleigh = compute_cube_tmatrix(
            omega=OMEGA, a=A_SUB, ref=REF,
            contrast=MaterialContrast(Dlambda=0.2, Dmu=0.1, Drho=0.05),
        )
        T = sub_cell_tmatrix_9x9(rayleigh, OMEGA, A_SUB)
        V = (2.0 * A_SUB) ** 3
        expected = OMEGA**2 * complex(rayleigh.Drho_star) * V
        assert np.allclose(np.diag(T[0:3, 0:3]), expected)
        off = T[0:3, 0:3] - np.diag(np.diag(T[0:3, 0:3]))
        assert np.allclose(off, 0.0), "density block must be diagonal"

    def test_strain_block_has_cubic_voigt_structure(self):
        """Voigt block: first 3×3 dense (λ*, μ*_diag), last 3×3 diagonal (μ*_off)."""
        rayleigh = compute_cube_tmatrix(
            omega=OMEGA, a=A_SUB, ref=REF,
            contrast=MaterialContrast(Dlambda=0.2, Dmu=0.1, Drho=0.05),
        )
        T = sub_cell_tmatrix_9x9(rayleigh, OMEGA, A_SUB)
        S = T[3:9, 3:9]
        # Normal-shear coupling vanishes for cubic symmetry.
        assert np.allclose(S[0:3, 3:6], 0.0), "normal-shear coupling must vanish"
        assert np.allclose(S[3:6, 0:3], 0.0), "shear-normal coupling must vanish"
        # Shear block is diagonal in the three shear components.
        shear = S[3:6, 3:6]
        assert np.allclose(shear - np.diag(np.diag(shear)), 0.0), "shear block must be diagonal"

    def test_isotropic_contrast_gives_isotropic_normal_block(self):
        """With Δμ_diag == Δμ_off the normal block is λ*δ + 2μ*·I — no cubic split."""
        rayleigh = compute_cube_tmatrix(
            omega=OMEGA, a=A_SUB, ref=REF,
            contrast=MaterialContrast(Dlambda=0.2, Dmu=0.0, Drho=0.0),
        )
        T = sub_cell_tmatrix_9x9(rayleigh, OMEGA, A_SUB)
        D = T[3:6, 3:6]
        # Δμ = 0 ⇒ all nine entries of the normal block equal Δλ*·V.
        assert np.allclose(D, D[0, 0]), f"Δμ=0 must give a uniform normal block, got\n{D}"
```

- [ ] **Step 3: Run the basis tests to verify they fail**

Run: `cd ~/Desktop/Marine3D && conda run -n seismic python -m pytest tests/test_crust_field.py::TestBasisConvention -v`

Expected: FAIL — `ImportError: cannot import name 'sub_cell_tmatrix_9x9'` before Step 1 is applied, or `ModuleNotFoundError: No module named 'marine3d.crust_field'` from the module-level import.

If any of these four tests fails **after** the module exists, stop and investigate: the $T_0$ basis does not match the Green's-function basis and a permutation matrix is required before anything downstream can be trusted. Do not "fix" it by transposing until you know which convention is wrong.

- [ ] **Step 4: Write `crust_field.py`**

```python
"""Per-voxel crust contrast field and its 9×9 (u, ε) T-matrices.

Holds the disorder for the 2½-D crust: an ``(n_z, n_x)`` grid of material
contrasts relative to the crust mean, and the per-voxel 9×9 T₀ built from
them.  Pure data plus T₀ construction — no Green's functions, no solver.

Basis (fixed): ``(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)``.

Public API
----------
CrustField : the contrast field and its T-matrices
build_crust_field : factory
check_validity_floor : renormalisation validity gate
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from marine3d.tmatrix.effective_contrasts import (
    MaterialContrast,
    ReferenceMedium,
    compute_cube_tmatrix,
)
from marine3d.tmatrix.resonance_tmatrix import sub_cell_tmatrix_9x9


@dataclass
class CrustField:
    """Per-voxel contrast field on the 2½-D crust grid.

    Args:
        n_z: Number of crust depth planes.
        n_x: Number of voxels along x in each plane.
        pitch: Voxel edge length (km).  Voxels are cubes of side ``pitch``.
        dlambda: Δλ per voxel relative to the crust mean (GPa), shape (n_z, n_x).
        dmu: Δμ per voxel (GPa), shape (n_z, n_x).
        drho: Δρ per voxel (g/cm³), shape (n_z, n_x).
        reference: The crust mean, i.e. the background these contrasts are
            referenced to.
    """

    n_z: int
    n_x: int
    pitch: float
    dlambda: NDArray
    dmu: NDArray
    drho: NDArray
    reference: ReferenceMedium

    def __post_init__(self) -> None:
        """Validate grid shapes."""
        for name in ("dlambda", "dmu", "drho"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (self.n_z, self.n_x):
                msg = (
                    f"CrustField.{name} has shape {arr.shape}, expected "
                    f"({self.n_z}, {self.n_x}). Every contrast array must cover "
                    f"the full (n_z, n_x) voxel grid."
                )
                raise ValueError(msg) from None
            setattr(self, name, arr)
        if self.pitch <= 0.0:
            msg = f"CrustField.pitch must be positive, got {self.pitch}"
            raise ValueError(msg) from None

    @property
    def length(self) -> float:
        """Lattice period L = n_x · pitch (km)."""
        return self.n_x * self.pitch

    def tmatrices(self, omega: float) -> NDArray:
        """Per-voxel 9×9 T₀ at angular frequency ``omega``.

        Args:
            omega: Angular frequency (rad/s).

        Returns:
            Array of shape ``(n_z, n_x, 9, 9)``, complex.
        """
        a_sub = 0.5 * self.pitch
        out = np.zeros((self.n_z, self.n_x, 9, 9), dtype=np.complex128)
        for iz in range(self.n_z):
            for ix in range(self.n_x):
                contrast = MaterialContrast(
                    Dlambda=float(self.dlambda[iz, ix]),
                    Dmu=float(self.dmu[iz, ix]),
                    Drho=float(self.drho[iz, ix]),
                )
                rayleigh = compute_cube_tmatrix(
                    omega=omega, a=a_sub, ref=self.reference, contrast=contrast
                )
                out[iz, ix] = sub_cell_tmatrix_9x9(rayleigh, omega, a_sub)
        return out
```

- [ ] **Step 5: Run the basis tests to verify they pass**

Run: `conda run -n seismic python -m pytest tests/test_crust_field.py::TestBasisConvention -v`
Expected: 4 passed.

- [ ] **Step 6: Write the failing validity-floor and zero-contrast tests**

```python
class TestZeroContrast:
    """Δ = 0 must give an identically zero T₀ — the Phase-1 reference case."""

    def test_zero_contrast_gives_zero_tmatrix(self):
        field = build_crust_field(
            n_z=2, n_x=4, pitch=0.02, reference=REF,
            dlambda=np.zeros((2, 4)), dmu=np.zeros((2, 4)), drho=np.zeros((2, 4)),
        )
        T = field.tmatrices(OMEGA)
        assert T.shape == (2, 4, 9, 9)
        assert np.allclose(T, 0.0), (
            f"zero contrast must give zero T₀, got max|T| = {np.max(np.abs(T)):.3e}"
        )


class TestValidityFloor:
    """|Δ| < φ · background, enforced at construction with a 4-element diagnostic."""

    def test_floor_violation_raises_with_diagnostic(self):
        lam0 = REF.rho * (REF.alpha**2 - 2.0 * REF.beta**2)
        field = build_crust_field(
            n_z=1, n_x=2, pitch=0.02, reference=REF,
            dlambda=np.full((1, 2), 5.0 * lam0), dmu=np.zeros((1, 2)),
            drho=np.zeros((1, 2)),
        )
        with pytest.raises(ValueError) as exc:
            check_validity_floor(field, phi=0.52)
        msg = str(exc.value)
        assert "dlambda" in msg, "must name WHAT is wrong"
        assert "crust_heterogeneity" in msg, "must name WHERE to fix it (dotted key)"
        assert "example" in msg.lower(), "must show a valid example"
        assert "reduce" in msg.lower() or "increase" in msg.lower(), "must give a recovery step"

    def test_floor_satisfied_passes(self):
        field = build_crust_field(
            n_z=1, n_x=2, pitch=0.02, reference=REF,
            dlambda=np.full((1, 2), 0.2), dmu=np.zeros((1, 2)), drho=np.zeros((1, 2)),
        )
        check_validity_floor(field, phi=0.52)  # must not raise
```

- [ ] **Step 7: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_crust_field.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_crust_field'`.

- [ ] **Step 8: Add `build_crust_field` and `check_validity_floor`**

```python
def build_crust_field(
    n_z: int,
    n_x: int,
    pitch: float,
    reference: ReferenceMedium,
    dlambda: NDArray,
    dmu: NDArray,
    drho: NDArray,
) -> CrustField:
    """Construct a CrustField from explicit per-voxel contrast arrays.

    Args:
        n_z: Number of crust depth planes.
        n_x: Number of voxels along x.
        pitch: Voxel edge length (km).
        reference: The crust mean.
        dlambda: Δλ per voxel (GPa), shape (n_z, n_x).
        dmu: Δμ per voxel (GPa), shape (n_z, n_x).
        drho: Δρ per voxel (g/cm³), shape (n_z, n_x).

    Returns:
        The validated CrustField.
    """
    return CrustField(
        n_z=n_z, n_x=n_x, pitch=pitch,
        dlambda=np.asarray(dlambda, dtype=np.float64),
        dmu=np.asarray(dmu, dtype=np.float64),
        drho=np.asarray(drho, dtype=np.float64),
        reference=reference,
    )


def check_validity_floor(field: CrustField, phi: float) -> None:
    """Enforce |Δ| < φ · background on every voxel.

    The renormalisation used by the effective-contrast T₀ is only valid below
    this floor (sphere-packing study).  Checked once at construction, never
    mid-solve.

    Args:
        field: The contrast field to check.
        phi: Packing-fraction floor coefficient (e.g. 0.52).

    Raises:
        ValueError: If any voxel violates the floor, with a diagnostic naming
            the offending array, the worst voxel, the limit, and the fix.
    """
    ref = field.reference
    mu0 = ref.rho * ref.beta**2
    lam0 = ref.rho * (ref.alpha**2 - 2.0 * ref.beta**2)
    checks = (
        ("dlambda", field.dlambda, lam0, "GPa"),
        ("dmu", field.dmu, mu0, "GPa"),
        ("drho", field.drho, ref.rho, "g/cm³"),
    )
    for name, arr, background, unit in checks:
        limit = phi * abs(background)
        worst = float(np.max(np.abs(arr)))
        if worst >= limit:
            iz, ix = np.unravel_index(int(np.argmax(np.abs(arr))), arr.shape)
            msg = (
                f"Crust heterogeneity exceeds the renormalisation validity floor.\n"
                f"  WHAT:  |{name}| = {worst:.4g} {unit} at voxel (z={iz}, x={ix}); "
                f"the floor is phi * background = {phi} * {abs(background):.4g} "
                f"= {limit:.4g} {unit}.\n"
                f"  WHERE: {_CONFIG_PATH}, key crust_heterogeneity.{name}\n"
                f"  EXAMPLE of a valid value:\n"
                f"    crust_heterogeneity:\n"
                f"      {name}: {0.5 * limit:.4g}   # {unit}, well inside the floor\n"
                f"  RECOVERY: reduce crust_heterogeneity.{name} below {limit:.4g} {unit}, "
                f"or increase the contrast by re-referencing to a different crust mean."
            )
            raise ValueError(msg) from None
```

Define `_CONFIG_PATH` at module scope as the absolute path of `configs/marine_reference.yaml`, resolved with `pathlib.Path`:

```python
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "marine_reference.yaml"
```

- [ ] **Step 9: Run all Task 1 tests**

Run: `conda run -n seismic python -m pytest tests/test_crust_field.py -v`
Expected: 7 passed.

- [ ] **Step 10: Confirm nothing regressed and lint**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic python -m pytest tests/ -q --ignore=tests/test_resonance_far_field.py --ignore=tests/test_inter_voxel_propagator.py
conda run -n seismic ruff check marine3d/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
```
Expected: `475 passed, 7 skipped`; `All checks passed!`; `Success: no issues found`.

- [ ] **Step 11: Commit**

```bash
cd ~/Desktop/Marine3D
git add marine3d/crust_field.py marine3d/tmatrix/resonance_tmatrix.py
git commit -m "✨ crust_field: per-voxel contrast field and 9x9 T0 screens"
```

---

## Task 2: $\mathbf{P}^x$ — direct intra-plane coupling

**Files:**
- Create: `marine3d/intraplane_px.py`
- Test: `tests/test_intraplane_px.py`

**Interfaces:**
- Consumes: `marine3d.tmatrix.horizontal_greens.horizontal_greens_ky_residue_9x9` (or `horizontal_greens_fft_9x9`), `marine3d.crust_field.CrustField`.
- Produces:
  - `build_px_kernel(field: CrustField, omega: float) -> NDArray` — the intra-plane 9×9 Green's tensor on the lattice offsets, shape `(n_x, 9, 9)`, with the **self-term `[0]` identically zero**. No quadrature parameters.
  - `apply_px(kernel: NDArray, psi: NDArray) -> NDArray` — circulant apply along $x$; `psi` and the return value both shape `(n_z, n_x, 9)`.

**The kernel is built from the closed form, not by quadrature (decided 2026-07-25).** `marine3d.tmatrix.horizontal_greens.exact_propagator_9x9(x, y, z, omega, ref)` is the Kupradze analytic propagator — no quadrature, pinned to 1e-14 by the repo's own tests, and it takes a **signed** `x`. Use it directly:

```python
exact_propagator_9x9(dx_signed, 0.0, 0.0, complex(omega), field.reference)
```

where `dx_signed` is the **minimum-image** displacement for lattice offset `m` on the periodic $x$-lattice:

```python
m_signed = ((m + n_x // 2) % n_x) - n_x // 2      # maps m to [-n_x/2, +n_x/2)
dx_signed = m_signed * field.pitch
```

Note the half-open range is `[-n_x/2, +n_x/2)`: for `n_x = 8`, `m = 4` folds to `-4`, i.e. the half-period offset resolves to `-L/2`. That offset is genuinely two-valued on a periodic lattice and some convention must be chosen; this is the one. `n_x` is even in production, so **downstream code must not assume `+L/2`**.

*Why not the FFT route.* `horizontal_greens_fft_9x9` **approximates** this same object by quadrature. Measured at `pitch = 0.02 km`, crust mean `(3.0, 1.5, 2.6)`, `omega = 2π·5`: the accuracy is governed by the dimensionless product `k_max·Δx`, and the originally-specified `kz_max = 40` gives `40 × 0.02 = 0.8` and a **101% wrong** kernel. Reaching 1e-3 needs `k_max ≈ 800`, i.e. ~10⁴× the cost, to approximate something available exactly. Both transverse cutoffs must also rise together — raising one alone leaves ~88% error, which is an artifact of the other truncation rather than a convergence wall. Damping does not help (6.0946e-3 undamped vs 6.0919e-3 at 3% damping, matched cutoffs) and must not be added: undamped is the project convention and the rung-8 energy balance depends on it.

*Consequences for the tests.* Two tests originally specified here were incapable of failing and are replaced below:

- `test_reflection_symmetry_in_x` was **tautological** on the FFT route, which only accepts `|Δx|` — negative offsets could only come from the parity map, so the test restated its own construction and passed on any kernel. With signed `Δx` the two directions are computed independently and the parity check has teeth.
- `test_decays_with_offset` passed on the 101%-wrong kernel.
- `test_kernel_converges_in_kz_quadrature` is **deleted**. It refined sampling at fixed cutoff, and sampling is flat: `n_kz` 256→2048 moves the error 9.9882e-1 → 9.9886e-1. It would have certified a kernel that was 99.9% wrong. There is nothing to converge on the closed-form route.

**Where the propagator's correctness actually rests, now that the spectral route is gone.** The Task 2 tests validate the kernel's *assembly* — self-term, minimum-image folding, parity, symmetry — not the Green's tensor itself. The propagator is pinned by two things outside this task:

1. the repo's own `test_exact_propagator_9x9_vs_resonance`, which checks it to **1e-14** against an independent implementation; and
2. a cross-method measurement made during Task 2 and preserved here because the code that produced it was then deleted: the converged **spectral** route agrees with the closed form to **2.5196e-04**, falling monotonically 2.13e-1 → 8.18e-4 → 2.52e-4 as `cutoff_ratio` goes 5 → 12.5 → 25.

Item 2 is the evidence that the closed form is not merely self-consistent. Cite it in the Task 8 LaTeX note; it cannot be regenerated from the shipped code.

**Note on the kernel source.** `horizontal_greens.py` was cherry-picked as a script-style module with module-level `OMEGA/RHO/ALPHA/BETA` defaults. Read its signatures before wiring; pass the crust reference explicitly rather than relying on those defaults. The kernel is the **homogeneous whole-space** intra-plane Green's tensor for the crust mean at $\Delta z = 0$, $\Delta y = 0$ — it must contain no stratification, because the stratified part belongs to Task 3.

- [ ] **Step 1: Write the failing circulant-apply tests**

These test the *apply*, which is pure linear algebra and fully determined — independent of whatever the physics kernel turns out to be.

```python
# tests/test_intraplane_px.py
import numpy as np
import pytest

from marine3d.intraplane_px import apply_px, build_px_kernel


class TestCirculantApply:
    """apply_px must be the circulant convolution along x, self-term excluded."""

    def test_matches_explicit_double_sum(self):
        """apply_px == Σ_j K[(i-j) mod n_x] @ psi[j], to machine precision."""
        rng = np.random.default_rng(0)
        n_z, n_x = 2, 8
        kernel = rng.normal(size=(n_x, 9, 9)) + 1j * rng.normal(size=(n_x, 9, 9))
        kernel[0] = 0.0  # self-term excluded by construction
        psi = rng.normal(size=(n_z, n_x, 9)) + 1j * rng.normal(size=(n_z, n_x, 9))

        got = apply_px(kernel, psi)

        expected = np.zeros_like(psi)
        for iz in range(n_z):
            for i in range(n_x):
                for j in range(n_x):
                    expected[iz, i] += kernel[(i - j) % n_x] @ psi[iz, j]

        err = np.max(np.abs(got - expected))
        assert err < 1e-12, f"circulant apply mismatch: max|err| = {err:.3e}"

    def test_zero_kernel_gives_zero(self):
        psi = np.ones((3, 4, 9), dtype=np.complex128)
        out = apply_px(np.zeros((4, 9, 9), dtype=np.complex128), psi)
        assert np.allclose(out, 0.0)

    def test_planes_do_not_mix(self):
        """P^x is intra-plane: exciting one plane must not excite another."""
        rng = np.random.default_rng(1)
        n_z, n_x = 3, 8
        kernel = rng.normal(size=(n_x, 9, 9)) + 1j * rng.normal(size=(n_x, 9, 9))
        kernel[0] = 0.0
        psi = np.zeros((n_z, n_x, 9), dtype=np.complex128)
        psi[1] = 1.0  # excite the middle plane only

        out = apply_px(kernel, psi)

        assert np.allclose(out[0], 0.0), "plane 0 must stay unexcited"
        assert np.allclose(out[2], 0.0), "plane 2 must stay unexcited"
        assert not np.allclose(out[1], 0.0), "the excited plane must respond"

    def test_shape_mismatch_raises(self):
        psi = np.ones((2, 8, 9), dtype=np.complex128)
        with pytest.raises(ValueError, match="n_x"):
            apply_px(np.zeros((4, 9, 9), dtype=np.complex128), psi)
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_intraplane_px.py::TestCirculantApply -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marine3d.intraplane_px'`.

- [ ] **Step 3: Implement `apply_px`**

```python
"""P^x — direct intra-plane coupling on the periodic x-lattice.

Couples voxels within the SAME crust plane through the homogeneous whole-space
crust medium.  Stratification (energy leaving the plane, reflecting off the
free surface / seabed / crust base, and returning) is NOT here: that is the
self-plane reverberation and it belongs to :mod:`marine3d.vertical_pz`.  The
split is the G_ii contract; see the Phase-2 design §5.3.

Basis: ``(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)``.

Public API
----------
build_px_kernel : intra-plane 9×9 Green's tensor on the lattice offsets
apply_px : circulant apply along x
"""

import numpy as np
from numpy.typing import NDArray


def apply_px(kernel: NDArray, psi: NDArray) -> NDArray:
    """Apply the intra-plane coupling by circulant convolution along x.

    Computes ``out[z, i] = Σ_j kernel[(i - j) mod n_x] @ psi[z, j]`` via FFT.
    The self-term ``kernel[0]`` is expected to be zero (a voxel does not
    couple to itself); this function does not enforce it, so that callers can
    test the pure convolution independently.

    Args:
        kernel: Intra-plane Green's tensor on lattice offsets, shape
            ``(n_x, 9, 9)``.
        psi: Field per plane and voxel, shape ``(n_z, n_x, 9)``.

    Returns:
        Coupled field, shape ``(n_z, n_x, 9)``.

    Raises:
        ValueError: If the kernel and field disagree on ``n_x``.
    """
    if kernel.shape[0] != psi.shape[1]:
        msg = (
            f"kernel has n_x={kernel.shape[0]} but psi has n_x={psi.shape[1]}. "
            f"Both must be sampled on the same periodic x-lattice."
        )
        raise ValueError(msg) from None

    # Circulant matrix-vector product: diagonalised by the DFT along x.
    k_hat = np.fft.fft(kernel, axis=0)              # (n_x, 9, 9)
    psi_hat = np.fft.fft(psi, axis=1)               # (n_z, n_x, 9)
    out_hat = np.einsum("xab,zxb->zxa", k_hat, psi_hat)
    return np.fft.ifft(out_hat, axis=1)
```

- [ ] **Step 4: Run to verify the apply tests pass**

Run: `conda run -n seismic python -m pytest tests/test_intraplane_px.py::TestCirculantApply -v`
Expected: 4 passed.

- [ ] **Step 5: Write the failing kernel-physics tests**

```python
from marine3d.crust_field import build_crust_field
from marine3d.tmatrix.effective_contrasts import ReferenceMedium

REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.6)
OMEGA = 2.0 * np.pi * 5.0


def _field(n_x=8, pitch=0.02):
    z = np.zeros((1, n_x))
    return build_crust_field(n_z=1, n_x=n_x, pitch=pitch, reference=REF,
                             dlambda=z, dmu=z, drho=z)


class TestKernelPhysics:
    """The kernel is the homogeneous whole-space intra-plane Green's tensor."""

    def test_self_term_is_zero(self):
        """A voxel must not couple to itself: kernel[0] == 0 exactly."""
        K = build_px_kernel(_field(), OMEGA, n_kz=256, kz_max=40.0)
        assert np.allclose(K[0], 0.0), (
            f"self-term must be excluded, got max|K[0]| = {np.max(np.abs(K[0])):.3e}"
        )

    def test_reflection_symmetry_in_x(self):
        """Homogeneous background ⇒ K[+m] and K[−m] related by the x-parity map.

        Under x → −x the basis components u_x, 2ε_xy, 2ε_zx are odd and the
        rest even, so K[−m] = P K[+m] P with P = diag of those signs.
        """
        n_x = 8
        K = build_px_kernel(_field(n_x=n_x), OMEGA, n_kz=256, kz_max=40.0)
        signs = np.array([1, -1, 1, 1, 1, 1, -1, 1, -1], dtype=float)
        P = np.diag(signs)
        for m in range(1, n_x // 2):
            got, mirror = K[n_x - m], P @ K[m] @ P
            err = np.max(np.abs(got - mirror))
            scale = max(np.max(np.abs(K[m])), 1e-30)
            assert err / scale < 1e-8, (
                f"x-parity broken at offset m={m}: rel err {err / scale:.3e}"
            )

    def test_decays_with_offset(self):
        """Amplitude must fall off with separation — a sanity check on the kernel."""
        K = build_px_kernel(_field(n_x=16, pitch=0.02), OMEGA, n_kz=256, kz_max=40.0)
        near = np.max(np.abs(K[1]))
        far = np.max(np.abs(K[8]))
        assert far < near, f"kernel grew with distance: |K[1]|={near:.3e}, |K[8]|={far:.3e}"

    def test_minimum_image_offsets(self):
        """Offset m > n_x/2 must use the SHORTER periodic image, with sign.

        On a periodic lattice, offset m and offset m − n_x are the same pair of
        voxels. The kernel must use the minimum-image displacement, so K[n_x−1]
        is the near neighbour at −pitch, not a distant voxel at +(n_x−1)·pitch.
        Getting this wrong makes distant voxels appear strongly coupled.
        """
        from marine3d.tmatrix.horizontal_greens import exact_propagator_9x9

        n_x = 8
        field = _field(n_x=n_x, pitch=0.02)
        K = build_px_kernel(field, OMEGA)

        near = exact_propagator_9x9(-field.pitch, 0.0, 0.0, complex(OMEGA), REF)
        err = np.max(np.abs(K[n_x - 1] - near)) / np.max(np.abs(near))
        assert err < 1e-12, (
            f"K[n_x-1] is not the minimum image at -pitch: rel err {err:.3e}. "
            f"Offsets must fold to [-n_x/2, +n_x/2)."
        )

    def test_greens_tensor_is_symmetric(self):
        """Elastodynamic reciprocity: the 3x3 displacement block is symmetric.

        NOTE on this test's real strength: `exact_greens` builds the block as
        f·δ_ij + g·γ_i γ_j, which is symmetric BY CONSTRUCTION. So this catches
        a mis-transcription of the closed form, not an independent violation of
        reciprocity. It is weaker than it looks — do not cite it as evidence
        that reciprocity holds.
        """
        field = _field(n_x=8, pitch=0.02)
        K = build_px_kernel(field, OMEGA)
        for m in (1, 2, 3):
            G = K[m][:3, :3]
            asym = np.max(np.abs(G - G.T)) / max(np.max(np.abs(G)), 1e-30)
            assert asym < 1e-10, (
                f"displacement block not symmetric at offset m={m}: {asym:.3e}"
            )
```

- [ ] **Step 6: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_intraplane_px.py::TestKernelPhysics -v`
Expected: FAIL — `ImportError: cannot import name 'build_px_kernel'`.

- [ ] **Step 7: Implement `build_px_kernel`**

Read `marine3d/tmatrix/horizontal_greens.py` first — specifically `horizontal_greens_ky_residue_9x9` and `horizontal_greens_fft_9x9` — and pick whichever takes the reference medium explicitly. Then:

1. For each lattice offset `m` in `1 .. n_x-1`, compute `Δx = m · pitch` folded to the shorter periodic image (`Δx = min(m, n_x - m) · pitch` with the sign carried), and evaluate the 9×9 intra-plane Green's tensor at `(Δx, Δy=0, Δz=0)` for the crust reference.
2. Set `kernel[0] = 0`.
3. Return shape `(n_x, 9, 9)`.

Pass `omega`, and the reference `alpha`, `beta`, `rho` explicitly to the `horizontal_greens` routine — **do not rely on its module-level defaults**, which are values from the original standalone script and are not the crust mean.

- [ ] **Step 8: Run all Task 2 tests**

Run: `conda run -n seismic python -m pytest tests/test_intraplane_px.py -v`
Expected: 8 passed.

If `test_reflection_symmetry_in_x` fails, the sign vector encodes the basis parity — check it against the fixed basis ordering before assuming the kernel is wrong.

- [ ] **Step 9: Lint and commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic ruff check marine3d/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/intraplane_px.py
git commit -m "✨ intraplane_px: direct intra-plane coupling, circulant in x"
```

---

## Task 3: $\mathbf{P}^z$ and the $G_{ii}$ contract

**This is the highest-risk task in the plan.** Both failure modes — dropping the self-plane reverberation, and double-counting the direct term — produce plausible-looking output. Rung 3 runs before any solver exists precisely so that it fails here rather than as a factor of two in a gather.

**Files:**
- Create: `marine3d/vertical_pz.py`
- Test: `tests/test_vertical_pz.py`

**Interfaces:**
- Consumes: `marine3d.gmm.layered_greens.layered_greens_9x9`, `marine3d.gmm.layer_model_adapter.LayerModel`, `marine3d.gmm.interlayer_ms.build_interlayer_greens_matrix_9x9`.
- Produces:
  - `whole_space_gii(field: CrustField, omega: complex, kx: NDArray) -> NDArray` — the homogeneous whole-space same-plane Green's tensor in the $k_x$ domain, shape `(n_kx, 9, 9)`. The spectral counterpart of Task 2's kernel.
  - `self_plane_reverb(model: LayerModel, omega: complex, kx: NDArray, iface: int, field: CrustField) -> NDArray` — $G^{\text{reverb}}_{ii} = G^{\text{strat}}_{ii} - G^{\text{whole}}$, shape `(n_kx, 9, 9)`.
  - `build_pz_blocks(model, omega, kx, ifaces, field) -> NDArray` — full $\mathbf{P}^z$ including the restored diagonal, shape `(n_kx, 9·N_z, 9·N_z)`.
  - `apply_pz(blocks: NDArray, psi_hat: NDArray) -> NDArray` — apply in the $k_x$ domain; `psi_hat` shape `(n_z, n_kx, 9)`.

- [ ] **Step 1: Write the failing $G_{ii}$ contract test (rung 3)**

```python
# tests/test_vertical_pz.py
import numpy as np
import pytest

from marine3d.crust_field import build_crust_field
from marine3d.gmm.layered_greens import layered_greens_9x9
from marine3d.tmatrix.effective_contrasts import ReferenceMedium
from marine3d.vertical_pz import (
    apply_pz,
    build_pz_blocks,
    self_plane_reverb,
    whole_space_gii,
)

REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.6)
OMEGA = 2.0 * np.pi * 5.0 + 0.05j   # damped, as the pipeline uses
KX = np.array([0.01, 0.5, 1.5, 4.0])  # rad/km
IFACE = 1


def _model():
    """Marine LayerModel: ocean + 2 crust planes + half-space (see Task 3 Step 3)."""
    from marine3d.gmm.layer_model_adapter import LayerModel
    return LayerModel(
        alpha=np.array([1.5, 3.0, 3.0, 5.0]),
        beta=np.array([0.0, 1.5, 1.5, 3.0]),
        rho=np.array([1.0, 2.6, 2.6, 3.2]),
        thickness=np.array([2.0, 0.5, 0.5, np.inf]),
        Q_alpha=np.full(4, np.inf),
        Q_beta=np.full(4, np.inf),
    )


def _field(n_x=8):
    z = np.zeros((2, n_x))
    return build_crust_field(n_z=2, n_x=n_x, pitch=0.02, reference=REF,
                             dlambda=z, dmu=z, drho=z)


class TestGiiContract:
    """RUNG 3 — the module boundary between P^x and P^z must partition G_ii."""

    def test_direct_plus_reverb_equals_stratified(self):
        """G_whole + G_reverb == G_strat, to machine precision, at every k_x.

        This is the contract. If it fails, either P^x and P^z are computing
        the direct term in different conventions, or the subtraction in
        self_plane_reverb is wrong. Do NOT proceed to Task 4 until it passes.
        """
        model, field = _model(), _field()
        ky = np.zeros_like(KX)

        g_strat = layered_greens_9x9(
            model, OMEGA, KX, ky, source_iface=IFACE, receiver_iface=IFACE
        )
        g_whole = whole_space_gii(field, OMEGA, KX)
        g_reverb = self_plane_reverb(model, OMEGA, KX, IFACE, field)

        err = np.max(np.abs((g_whole + g_reverb) - g_strat))
        scale = max(np.max(np.abs(g_strat)), 1e-30)
        assert err / scale < 1e-12, (
            f"G_ii contract violated: |G_whole + G_reverb - G_strat| / scale = "
            f"{err / scale:.3e}. P^x and P^z are not partitioning the same-plane "
            f"coupling — either a term is dropped or the direct part is counted twice."
        )

    def test_reverb_is_not_negligible(self):
        """Guards against 'passing' the contract by making G_reverb ≈ 0.

        The marine seabed reflection coefficient is ≈0.79, so self-plane
        reverberation is a first-order effect, not a correction.
        """
        model, field = _model(), _field()
        g_reverb = self_plane_reverb(model, OMEGA, KX, IFACE, field)
        g_whole = whole_space_gii(field, OMEGA, KX)
        ratio = np.max(np.abs(g_reverb)) / max(np.max(np.abs(g_whole)), 1e-30)
        assert ratio > 1e-3, (
            f"self-plane reverberation is {ratio:.3e} of the direct term — "
            f"implausibly small for a marine model with a strong seabed. "
            f"Check that the stratified Green's function is being evaluated "
            f"in the full stack, not a crust-only sub-model."
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_vertical_pz.py::TestGiiContract -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marine3d.vertical_pz'`.

- [ ] **Step 3: Implement `whole_space_gii` and `self_plane_reverb`**

The contract is a definition, so implement `self_plane_reverb` **as** the subtraction — never as an independently derived formula:

```python
def self_plane_reverb(model, omega, kx, iface, field):
    """G_reverb = G_strat(i,i) − G_whole, both in the k_x domain."""
    ky = np.zeros_like(np.asarray(kx, dtype=np.float64))
    g_strat = layered_greens_9x9(model, omega, kx, ky, iface, iface)
    return g_strat - whole_space_gii(field, omega, kx)
```

That makes `test_direct_plus_reverb_equals_stratified` true by construction — which is the point. **The real content of rung 3 is `whole_space_gii` being the same object Task 2's kernel is**, in the other domain. Add that equivalence test explicitly in Step 5; without it, the contract test is circular.

For `whole_space_gii`: evaluate the homogeneous whole-space same-plane Green's tensor for the crust mean at $\Delta z = 0$, $k_y = 0$, on the supplied $k_x$. Use the same underlying routine Task 2 uses, in its spectral form.

- [ ] **Step 4: Run the contract tests**

Run: `conda run -n seismic python -m pytest tests/test_vertical_pz.py::TestGiiContract -v`
Expected: 2 passed.

- [ ] **Step 5: Write and pass the cross-domain equivalence test**

This is what stops the contract from being circular: the spectral `whole_space_gii` and the real-space `build_px_kernel` must be the same operator.

**⚠ The independent source changed when Task 2 dropped the spectral route.** This test was originally designed with `horizontal_greens_fft_9x9` supplying an independent spectral whole-space Green's function. That code path is gone. **Do not implement `whole_space_gii` as the DFT of `build_px_kernel`** — that would make this test compare an object to itself and the whole $G_{ii}$ contract would certify nothing.

Use the **GMM machinery** as the independent source instead. `layered_greens_9x9` computes the stratified Green's function through an entirely separate implementation (modal eigenvectors + Riccati sweeps). Hand it a **degenerate stack whose layers all carry the crust-mean properties**: with no impedance contrast anywhere there are no interfaces to reflect from, so its same-plane Green's function *is* the whole-space one. That gives a genuine second implementation of the same physics:

```python
def _homogeneous_model(field, n_layers=4, thickness=0.5):
    """LayerModel with crust-mean properties in EVERY layer — no interfaces.

    A stack with no impedance contrast has nothing to reflect from, so
    layered_greens_9x9(i, i) on it returns the whole-space same-plane
    Green's function, computed through the GMM/Riccati path rather than
    through the Kupradze closed form. That independence is the point.
    """
    from marine3d.gmm.layer_model_adapter import LayerModel

    ref = field.reference
    thick = np.full(n_layers, thickness)
    thick[-1] = np.inf
    return LayerModel(
        alpha=np.full(n_layers, ref.alpha),
        beta=np.full(n_layers, ref.beta),
        rho=np.full(n_layers, ref.rho),
        thickness=thick,
        Q_alpha=np.full(n_layers, np.inf),
        Q_beta=np.full(n_layers, np.inf),
    )
```

If this test fails, that is a real finding about the two implementations disagreeing — escalate it rather than reconciling by redefining one in terms of the other.

```python
from marine3d.intraplane_px import build_px_kernel


class TestPxSpectralRealSpaceEquivalence:
    """whole_space_gii(k_x) must be the DFT of build_px_kernel(Δx), self-term aside."""

    def test_dft_of_real_space_kernel_matches_spectral(self):
        n_x = 16
        field = _field(n_x=n_x)
        omega_real = 2.0 * np.pi * 5.0

        K = build_px_kernel(field, omega_real)                          # (n_x, 9, 9)
        k_lattice = 2.0 * np.pi * np.fft.fftfreq(n_x, d=field.pitch)   # rad/km

        g_spec = whole_space_gii(field, omega_real, k_lattice)          # (n_x, 9, 9)
        k_hat = np.fft.fft(K, axis=0)

        # The real-space kernel excludes the self-term; the spectral form does
        # not. They differ by exactly the k_x-independent self-term, i.e. the
        # mean over k_x. Remove it from both before comparing.
        k_hat -= k_hat.mean(axis=0, keepdims=True)
        g_spec -= g_spec.mean(axis=0, keepdims=True)

        err = np.max(np.abs(k_hat - g_spec))
        scale = max(np.max(np.abs(g_spec)), 1e-30)
        assert err / scale < 1e-6, (
            f"P^x real-space and spectral forms disagree: rel err {err / scale:.3e}. "
            f"The G_ii contract is only meaningful if these are the same operator."
        )
```

Run: `conda run -n seismic python -m pytest tests/test_vertical_pz.py::TestPxSpectralRealSpaceEquivalence -v`
Expected: PASS. If it fails, reconcile the two before continuing — one of them has the wrong normalisation, sign, or $k_z$ integration range.

- [ ] **Step 6: Write the failing `build_pz_blocks` / `apply_pz` tests**

```python
class TestPzBlocks:
    """P^z carries inter-plane coupling AND the restored diagonal."""

    def test_diagonal_block_is_the_reverb_term(self):
        model, field = _model(), _field()
        ifaces = [1, 2]
        blocks = build_pz_blocks(model, OMEGA, KX, ifaces, field)
        assert blocks.shape == (len(KX), 18, 18)
        expected = self_plane_reverb(model, OMEGA, KX, 1, field)
        got = blocks[:, 0:9, 0:9]
        err = np.max(np.abs(got - expected))
        assert err < 1e-12, (
            f"P^z diagonal block is not the self-plane reverberation "
            f"(max|err| = {err:.3e}). The cherry-picked builder zeroes it; "
            f"Phase 2 must restore it."
        )

    def test_offdiagonal_matches_cherry_picked_builder(self):
        """Off-diagonal blocks must equal the existing, already-tested builder."""
        from marine3d.gmm.interlayer_ms import build_interlayer_greens_matrix_9x9

        model, field = _model(), _field()
        ifaces = [1, 2]
        ky = np.zeros_like(KX)
        reference = build_interlayer_greens_matrix_9x9(model, OMEGA, KX, ky, ifaces)
        blocks = build_pz_blocks(model, OMEGA, KX, ifaces, field)
        err = np.max(np.abs(blocks[:, 0:9, 9:18] - reference[:, 0:9, 9:18]))
        assert err < 1e-12, f"off-diagonal P^z drifted from the ported builder: {err:.3e}"

    def test_apply_pz_matches_dense_matmul(self):
        rng = np.random.default_rng(3)
        n_kx, n_z = len(KX), 2
        blocks = (rng.normal(size=(n_kx, 18, 18))
                  + 1j * rng.normal(size=(n_kx, 18, 18)))
        psi = rng.normal(size=(n_z, n_kx, 9)) + 1j * rng.normal(size=(n_z, n_kx, 9))

        got = apply_pz(blocks, psi)

        expected = np.zeros_like(psi)
        for ik in range(n_kx):
            flat = psi[:, ik, :].reshape(18)
            out = blocks[ik] @ flat
            expected[:, ik, :] = out.reshape(n_z, 9)

        err = np.max(np.abs(got - expected))
        assert err < 1e-12, f"apply_pz mismatch: {err:.3e}"
```

- [ ] **Step 7: Run to verify failure, then implement, then re-run**

Run: `conda run -n seismic python -m pytest tests/test_vertical_pz.py -v`
Expected first: FAIL (`cannot import name 'build_pz_blocks'`). After implementing: 7 passed.

`build_pz_blocks` calls `build_interlayer_greens_matrix_9x9` for the off-diagonal blocks (reusing the tested code rather than re-deriving it) and fills each diagonal block with `self_plane_reverb`. `apply_pz` reshapes `(n_z, n_kx, 9) → (n_kx, 9·n_z)`, batches the matmul, and reshapes back.

- [ ] **Step 8: Lint, full suite, commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic python -m pytest tests/ -q --ignore=tests/test_resonance_far_field.py --ignore=tests/test_inter_voxel_propagator.py
conda run -n seismic ruff check marine3d/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/vertical_pz.py
git commit -m "✨ vertical_pz: inter-plane P^z with restored self-plane reverberation"
```

---

## Task 2b: Bloch lattice sum for $\mathbf{P}^x$

**Added 2026-07-25 after review of Task 2.** Task 2 ships a **nearest-image** kernel: `apply_px` is circulant, i.e. it assumes the $x$-lattice tiles periodically, but the kernel retains only the closest image of each voxel pair. For a genuinely periodic lattice the coupling is the sum over all images. Neglected images fall off as `pitch/L = 1/n_x`, so at `n_x = 8` this is a **10–15%** error — not rounding, and it would propagate uncontrolled into the rung-5 Wolfram comparison and the rung-8 energy balance.

**Files:**
- Modify: `marine3d/intraplane_px.py`
- Test: `tests/test_intraplane_px.py`
- Reference: `MultipleScatteringCalculations/Mathematica/IntraPlaneLatticeSum.wl`, `IntraPlaneVectorLattice.wl`, `IntraPlaneKambeVector.wl` and their `_reference.json` fixtures (validated in the Phase 3b intra-plane work: 8 gates pass, S-matrix unitary, reciprocity 4.1e-8).

**Interfaces:**
- `build_px_kernel(field: CrustField, omega: float, k_bloch: float) -> NDArray` — **breaking signature change**, see below.

### The Bloch phase makes the kernel incidence-dependent

The correct coupling carries the Bloch phase of the incident wave:

$$\mathbf{P}^x_{ij} \;=\; \sum_{n} \mathbf{G}\!\left(\Delta x_{ij} + nL\right) e^{\,i k_{\text{bloch}} nL}, \qquad k_{\text{bloch}} = \omega p .$$

So $\mathbf{P}^x$ **depends on the incident slowness $p$** and can no longer be built once per frequency. Consequences that ripple outward:

- `build_px_kernel` gains a `k_bloch` argument.
- Task 4's `build_matvec` must build the kernel **inside** the per-$p$ loop, not hoist it.
- Task 5's outer loop pays one kernel build per $(p, \omega)$ pair. The closed-form route is microseconds per offset, so this is affordable — it would **not** have been on the deleted quadrature route, which is a second reason that decision was right.

### Two routes, and why the naive one is not enough

Direct summation of $\sum_n \mathbf{G}(\Delta x + nL)$ is at best **conditionally convergent**: the whole-space Green's tensor decays as $1/r$, so the terms fall off like $1/n$. Truncating it is not a convergence question but an ordering question. Use **Ewald acceleration**, which is exactly what the existing `IntraPlaneLatticeSum.wl` / `IntraPlaneKambeVector.wl` machinery implements and validates.

By Poisson summation the lattice-summed real-space kernel and the spectral Green's function sampled at the Bloch-shifted harmonics $k_x^{(n)} = k_{\text{bloch}} + 2\pi n/L$ are the same object. Either representation is acceptable; the Ewald route is preferred because the spectral route needs the $(k_y, k_z)$ quadrature that Task 2 removed for accuracy and cost reasons.

**Port target: 1-D lattice, Cartesian 9×9.** The existing machinery is a 2-D planar lattice in a spherical multipole basis. This task needs the 1-D $x$-lattice in the Cartesian 9×9 $(u,\varepsilon)$ basis. Read `IntraPlaneLatticeSum.wl` for the Ewald splitting and convergence treatment; do not transcribe its multipole algebra.

### Gates

- [ ] **Convergence:** the Ewald sum must be independent of the splitting parameter $\eta$ over at least a decade. This is the standard Ewald self-check and it is load-bearing — an $\eta$-dependent result means the real-space and reciprocal-space halves are not paired correctly. Precedent: in Phase 3b the reciprocal `erfc` **pairing** bug was invisible at $z = 0$ and *only* an $\eta$-independence check could detect it.
- [ ] **Reduction:** as $L \to \infty$ at fixed pitch (i.e. $n_x \to \infty$), the lattice sum must approach the nearest-image kernel Task 2 already ships. That is the sense in which the current kernel is the leading term.
- [ ] **Magnitude:** at `n_x = 8` the lattice sum must differ from the nearest-image kernel by the predicted 10–15%. If the difference is negligible, the image sum is not actually being performed.
- [ ] **Cross-check:** against the stored `IntraPlaneLatticeSum_reference.json` values where the geometries correspond. Consume these as **numbers copied into the Marine3D fixtures directory** — do not import across repos.

---

## Task 2c: Volume-averaged nearest-neighbour coupling in $\mathbf{P}^x$

**Added 2026-07-25 after measurement.** $\mathbf{P}^x$ evaluates the Green's tensor between cube **centres** (point coupling). Its dominant term is the nearest neighbour at $\Delta x = \text{pitch}$ — *touching* cubes — which is exactly where point coupling is least valid.

### Measured at Phase 2 parameters

Crust mean $(\alpha,\beta,\rho) = (3.0, 1.5, 2.6)$, pitch $0.02$ km, $\omega = 2\pi\cdot 5$, so $k_S a = 0.209$:

| Block | point | volume-averaged | rel diff |
|---|---|---|---|
| G (3×3 displacement) | 6.78e-1 | 5.83e-1 | **17.1%** |
| C (3×6) | 3.55e+1 | 2.25e+1 | **58.2%** |
| H (6×3) | 3.55e+1 | 2.25e+1 | **58.2%** |
| S (6×6 strain) | 3.41e+3 | 1.18e+3 | **189.8%** |

**The error is not diluted.** The nearest neighbour carries **84.9%** of the kernel sum ($m = \pm 1$ at 42.4% each), and the S block is also the largest in magnitude. So $\mathbf{P}^x$ is dominated by a term whose biggest block is off by a factor of ~2.9.

For scale: this dwarfs the 10–15% periodic-image effect that Task 2b addresses. The two are **orthogonal** — 2b fixes the far images, 2c fixes the near term — and neither substitutes for the other.

### This is not a convention artifact — settled, do not re-litigate

The difference is a **genuine converged projection difference**: the Galerkin (volume-averaged) object and the point-propagator object are different objects at touching faces. Face S differs by ~0.5 of block scale and S44's sign genuinely differs; corner ~8%. The committed `FACE_*` constants are pinned by three independent arbiters — delta-collapse re-evaluation to 1e-16, subdivision fixed point to 1e-13, dyadic-shell 3D quadrature to 1e-8 — and regression-guarded by `TestFaceSBlockArbiter`.

**⚠ Do not use `avg_point_propagator_fd` as the arbiter here.** FD-of-⟨G⟩ at h=0.005, n=8–10 sits in the invalid $h \lesssim 1/n^2$ regime and *reproduces the face-S quadrature bias*, yielding a spurious "FD/direct cross-agreement". The bias is specific to face S: tensor-product double-cube Gauss shares one lateral node set and so samples the singular ray $w_\perp = 0$ of the $1/w^3$ kernel with O(1) cumulative weight. The gold-standard touching-pair arbiter is the **subdivision identity** (split each cube into 8 half-cubes; homogeneity gives `table(n) = (1/8)·Σ mult(m)·table(m)`).

The volume-averaged propagator is also the **consistent** partner for this solver: `sub_cell_tmatrix_9x9` produces a volume-integrated $T_0$ ($V\cdot\Delta c^*$), so it must be coupled by a volume-integrated propagator.

### Design

**Files:** modify `marine3d/intraplane_px.py`; test `tests/test_intraplane_px.py`.

**The rule:** any voxel pair at separation **exactly one pitch** uses `inter_voxel_propagator_9x9`; every other separation uses `exact_propagator_9x9`. On a 1-D $x$-lattice the only nearest-neighbour type that arises is the **face** neighbour `R_lattice = (0, 1, 0)` in the $(z,x,y)$ convention — no edge or corner cases, which is a real simplification over the 3-D problem.

This composes with Task 2b: inside the Bloch lattice sum, a pair at offset $m$ and image $n$ sits at separation $(m + n\,n_x)\cdot\text{pitch}$, so the volume-averaged treatment applies precisely to the terms where that product is $\pm 1$.

```python
inter_voxel_propagator_9x9(
    (0, 1, 0), field.reference.alpha, field.reference.beta, field.reference.rho,
    omega, n_orders=3, d=field.pitch,
)
```

`d` is **required** and must be the physical pitch — the tables are derived on a unit-pitch lattice and scale as $d^{-1}$ (G), $d^{-2}$ (C/H), $d^{-3}$ (S). `n_orders=3` is right at $k_S a = 0.209$: the face dynamic tables' truncation residual is 0.097% at ka=0.3 and 0.85% at ka=0.5.

### Gates

- [ ] **Magnitude.** Reproduce the table above: the nearest-neighbour block must change by 17.1% / 58.2% / 58.2% / 189.8% (G/C/H/S) at these parameters. If the kernel does not change, the volume-averaged branch is not being taken.
- [ ] **Pitch scaling.** Doubling `d` must scale the blocks by $1/2$, $1/4$, $1/4$, $1/8$ (G/C/H/S) exactly — this is the homogeneity identity and it catches a mis-passed pitch, which is otherwise silent.
- [ ] **Only the nearest neighbour changes.** Every $|m| \ge 2$ entry must be bit-identical to the Task 2 kernel.
- [ ] **Static limit.** As $\omega \to 0$ the volume-averaged block must approach the static `face_propagator(mu, nu)` tables.

### Known gap — state it, do not paper over it

`inter_voxel_propagator_9x9` supports **nearest neighbours only**; `R_lattice = (0, 2, 0)` raises "not a nearest neighbour". So the $|m| = 2$ error is **unmeasured**: separation there is only 2× the cube side, where "separation greatly exceeds cube size" does not yet hold. Given the $|m|=1$ errors above, a residual at $|m|=2$ is plausible. It carries 10.8% of the kernel sum (5.4% each side), so a 10% error there would be ~1% of $\mathbf{P}^x$ — an order below the effect this task fixes, but not zero. Record it as a known limit in the LaTeX note rather than implying the coupling is exact beyond the first neighbour.

---

# Stage B — Matvec and solver

## Task 4: The matvec, GMRES, and Born

**Files:**
- Create: `marine3d/matvec_25d.py`
- Test: `tests/test_matvec_25d.py`

**Interfaces:**
- Consumes: `apply_px`, `build_px_kernel`, `apply_pz`, `build_pz_blocks`, `CrustField.tmatrices`.
- Produces:
  - `@dataclass MatvecOperator` with `.apply(psi: NDArray) -> NDArray` implementing $(\mathbf{I}-[\mathbf{P}^z+\mathbf{P}^x]\Delta\mathcal{C}_{eff})\psi$, `psi` shape `(n_z, n_x, 9)`.
  - `build_matvec(model, field, omega, k_bloch, ifaces, *, n_kz, kz_max) -> MatvecOperator`.
  - `solve_gmres(op, rhs, *, tol, maxiter) -> tuple[NDArray, GmresInfo]` where `GmresInfo` carries `n_iter: int`, `residuals: list[float]`, `converged: bool`.
  - `born_solution(op, rhs) -> NDArray` — one Jacobi iterate, `rhs + [P^z+P^x]ΔC·rhs`.

- [ ] **Step 1: Write the failing operator tests**

```python
# tests/test_matvec_25d.py
import numpy as np
import pytest

from marine3d.crust_field import build_crust_field
from marine3d.matvec_25d import born_solution, build_matvec, solve_gmres
from marine3d.tmatrix.effective_contrasts import ReferenceMedium

REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.6)
OMEGA = 2.0 * np.pi * 5.0 + 0.05j
K_BLOCH = 0.5


def _model():
    from marine3d.gmm.layer_model_adapter import LayerModel
    return LayerModel(
        alpha=np.array([1.5, 3.0, 3.0, 5.0]),
        beta=np.array([0.0, 1.5, 1.5, 3.0]),
        rho=np.array([1.0, 2.6, 2.6, 3.2]),
        thickness=np.array([2.0, 0.5, 0.5, np.inf]),
        Q_alpha=np.full(4, np.inf),
        Q_beta=np.full(4, np.inf),
    )


def _field(n_x=8, amp=0.0):
    rng = np.random.default_rng(7)
    shape = (2, n_x)
    d = amp * rng.normal(size=shape)
    return build_crust_field(n_z=2, n_x=n_x, pitch=0.02, reference=REF,
                             dlambda=d, dmu=0.5 * d, drho=0.02 * d)


class TestOperatorLimits:
    def test_zero_contrast_gives_identity(self):
        """ΔC = 0 ⇒ the operator is the identity, to machine precision."""
        op = build_matvec(_model(), _field(amp=0.0), OMEGA, K_BLOCH, [1, 2],
                          n_kz=256, kz_max=40.0)
        rng = np.random.default_rng(11)
        psi = rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9))
        err = np.max(np.abs(op.apply(psi) - psi))
        assert err < 1e-12, f"zero-contrast operator is not the identity: {err:.3e}"

    def test_linearity(self):
        op = build_matvec(_model(), _field(amp=0.05), OMEGA, K_BLOCH, [1, 2],
                          n_kz=256, kz_max=40.0)
        rng = np.random.default_rng(13)
        a = rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9))
        b = rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9))
        c1, c2 = 2.0 + 1j, -0.5 + 3j
        lhs = op.apply(c1 * a + c2 * b)
        rhs = c1 * op.apply(a) + c2 * op.apply(b)
        assert np.max(np.abs(lhs - rhs)) < 1e-10


class TestBornIsOneIterate:
    """RUNG 7 — Born must be the first iterate of the same operator."""

    def test_born_equals_first_jacobi_iterate(self):
        op = build_matvec(_model(), _field(amp=0.02), OMEGA, K_BLOCH, [1, 2],
                          n_kz=256, kz_max=40.0)
        rng = np.random.default_rng(17)
        rhs = rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9))
        # (I − A)ψ = rhs with ψ⁰ = rhs gives ψ¹ = rhs + A·rhs = 2·rhs − op.apply(rhs).
        expected = 2.0 * rhs - op.apply(rhs)
        err = np.max(np.abs(born_solution(op, rhs) - expected))
        assert err < 1e-12, f"Born is not one iterate of the operator: {err:.3e}"

    def test_born_approaches_gmres_as_contrast_vanishes(self):
        """Born error must shrink with contrast — the linearisation is consistent."""
        rng = np.random.default_rng(19)
        rhs = rng.normal(size=(2, 8, 9)) + 1j * rng.normal(size=(2, 8, 9))
        errors = []
        for amp in (0.04, 0.02, 0.01):
            op = build_matvec(_model(), _field(amp=amp), OMEGA, K_BLOCH, [1, 2],
                              n_kz=256, kz_max=40.0)
            exact, _ = solve_gmres(op, rhs, tol=1e-12, maxiter=200)
            errors.append(np.max(np.abs(born_solution(op, rhs) - exact))
                          / np.max(np.abs(exact)))
        assert errors[0] > errors[1] > errors[2], (
            f"Born error did not shrink monotonically with contrast: {errors}"
        )


class TestGmresAgainstDense:
    """RUNG 6 — the matrix-free path must equal the dense one exactly."""

    def test_gmres_matches_dense_solve(self):
        op = build_matvec(_model(), _field(amp=0.05), OMEGA, K_BLOCH, [1, 2],
                          n_kz=256, kz_max=40.0)
        n_z, n_x = 2, 8
        n = n_z * n_x * 9

        # Materialise the operator column by column — only tractable because
        # this configuration is deliberately tiny.
        A = np.zeros((n, n), dtype=np.complex128)
        for j in range(n):
            e = np.zeros(n, dtype=np.complex128)
            e[j] = 1.0
            A[:, j] = op.apply(e.reshape(n_z, n_x, 9)).reshape(n)

        rng = np.random.default_rng(23)
        rhs = rng.normal(size=n) + 1j * rng.normal(size=n)
        dense = np.linalg.solve(A, rhs)
        krylov, info = solve_gmres(op, rhs.reshape(n_z, n_x, 9), tol=1e-12, maxiter=500)

        assert info.converged, f"GMRES did not converge in {info.n_iter} iterations"
        err = np.max(np.abs(krylov.reshape(n) - dense)) / np.max(np.abs(dense))
        assert err < 1e-9, f"matrix-free vs dense mismatch: rel err {err:.3e}"
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_matvec_25d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marine3d.matvec_25d'`.

- [ ] **Step 3: Implement the operator**

The apply, in order — note $\mathbf{P}^x$ is applied in real space (Task 2, self-term unambiguous) and $\mathbf{P}^z$ in $k_x$:

```python
def apply(self, psi):
    """(I − [P^z + P^x] ΔC_eff) ψ."""
    # 1. Local screen apply, real space, block-diagonal per voxel.
    src = np.einsum("zxab,zxb->zxa", self.tmatrices, psi)

    # 2. P^x — circulant along x, real space.
    out_x = apply_px(self.px_kernel, src)

    # 3. P^z — diagonal in k_x, so transform, apply, transform back.
    src_hat = np.fft.fft(src, axis=1)
    out_z = np.fft.ifft(apply_pz(self.pz_blocks, src_hat), axis=1)

    return psi - (out_x + out_z)
```

`solve_gmres` wraps `scipy.sparse.linalg.gmres` with a `LinearOperator` over the flattened `(n_z·n_x·9,)` vector, recording the residual history through the callback. `born_solution(op, rhs)` returns `2·rhs − op.apply(rhs)`.

The Bloch wavenumber enters through the $k_x$ grid handed to `build_pz_blocks`: `k_bloch + 2π·fftfreq(n_x, d=pitch)`.

- [ ] **Step 4: Run all Task 4 tests**

Run: `conda run -n seismic python -m pytest tests/test_matvec_25d.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint, full suite, commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic python -m pytest tests/ -q --ignore=tests/test_resonance_far_field.py --ignore=tests/test_inter_voxel_propagator.py
conda run -n seismic ruff check marine3d/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/matvec_25d.py
git commit -m "✨ matvec_25d: matrix-free (I − [P^z+P^x]ΔC) apply, GMRES, Born"
```

---

## Task 5: Dressed reflectivity, specular extraction, diffraction gate

**Files:**
- Create: `marine3d/reflectivity_25d.py`
- Modify: `marine3d/config.py`, `configs/marine_reference.yaml`
- Test: `tests/test_reflectivity_25d.py`

**Interfaces:**
- Consumes: `build_matvec`, `solve_gmres`, Phase-1 `kennett_reflectivity_batch`, `LayerStack`.
- Produces:
  - `check_diffraction_orders(field: CrustField, p: float, omega: float, alpha_min: float) -> None` — raises on an open order.
  - `dressed_reflectivity(stack, field, p_samples, omega_damped, *, n_kz, kz_max, tol, maxiter) -> tuple[NDArray, list[GmresInfo]]` — shape `(np_slow, nfreq)`, the same object `kennett_reflectivity_batch` returns.

- [ ] **Step 1: Write the failing diffraction-gate tests**

```python
# tests/test_reflectivity_25d.py
import numpy as np
import pytest

from marine3d.crust_field import build_crust_field
from marine3d.reflectivity_25d import check_diffraction_orders, dressed_reflectivity
from marine3d.tmatrix.effective_contrasts import ReferenceMedium

REF = ReferenceMedium(alpha=3.0, beta=1.5, rho=2.6)


def _field(n_x=8, pitch=0.02, amp=0.0):
    rng = np.random.default_rng(29)
    d = amp * rng.normal(size=(1, n_x))
    return build_crust_field(n_z=1, n_x=n_x, pitch=pitch, reference=REF,
                             dlambda=d, dmu=0.5 * d, drho=0.02 * d)


class TestDiffractionGate:
    """A periodic lattice is not a laterally invariant medium."""

    def test_open_order_aborts_with_diagnostic(self):
        """A long period at high frequency opens an order — must abort, not warn."""
        field = _field(n_x=64, pitch=0.5)  # L = 32 km
        omega = 2.0 * np.pi * 40.0
        with pytest.raises(ValueError) as exc:
            check_diffraction_orders(field, p=0.3, omega=omega, alpha_min=1.5)
        msg = str(exc.value)
        assert "diffraction" in msg.lower()
        assert "order" in msg.lower(), "must name the order that opened"
        for token in ("p=", "omega", "L"):
            assert token in msg, f"diagnostic must name {token}"

    def test_specular_only_passes(self):
        """Short period, low frequency: only n=0 propagates."""
        field = _field(n_x=8, pitch=0.02)  # L = 0.16 km
        check_diffraction_orders(field, p=0.1, omega=2.0 * np.pi * 5.0, alpha_min=1.5)


class TestReducesToReference:
    """RUNG 1 — Δ = 0 must reproduce the Phase-1 reflectivity exactly."""

    def test_zero_contrast_equals_kennett(self):
        from marine3d.earth_model import build_marine_reference_model
        from marine3d.kennett.kennett_layers import FluidLayer, IsotropicLayer
        from marine3d.kennett.kennett_layers import kennett_reflectivity_batch

        ocean = FluidLayer(alpha=1.5, rho=1.0, thickness=2.0)
        crust = IsotropicLayer(alpha=3.0, beta=1.5, rho=2.6, thickness=1.0)
        hs = IsotropicLayer(alpha=5.0, beta=3.0, rho=3.2, thickness=np.inf)
        stack = build_marine_reference_model(
            ocean=ocean, crust=crust, crust_thickness=1.0, halfspace=hs,
            n_crust_planes=1,
        )
        p_samples = np.array([0.05, 0.1, 0.2])
        omega = np.array([2 * np.pi * 5.0 + 0.4j])

        reference = kennett_reflectivity_batch(stack, p_samples, omega)
        dressed, _ = dressed_reflectivity(
            stack, _field(amp=0.0), p_samples, omega,
            n_kz=256, kz_max=40.0, tol=1e-12, maxiter=100,
        )

        err = np.max(np.abs(dressed - reference))
        scale = np.max(np.abs(reference))
        assert err / scale < 1e-12, (
            f"zero-contrast dressed reflectivity differs from the Phase-1 "
            f"reference by {err / scale:.3e} — the disorder path is leaking "
            f"into the Δ=0 case."
        )


class TestWrapAround:
    """A periodic x-lattice must not imprint its period on the answer."""

    def test_invariant_under_doubling_n_x(self):
        """Same physical contrast, twice the period ⇒ same specular reflectivity."""
        base = _field(n_x=8, pitch=0.02, amp=0.01)
        doubled = build_crust_field(
            n_z=1, n_x=16, pitch=0.02, reference=REF,
            dlambda=np.tile(base.dlambda, (1, 2)),
            dmu=np.tile(base.dmu, (1, 2)),
            drho=np.tile(base.drho, (1, 2)),
        )
        from marine3d.earth_model import build_marine_reference_model
        from marine3d.kennett.kennett_layers import FluidLayer, IsotropicLayer

        stack = build_marine_reference_model(
            ocean=FluidLayer(alpha=1.5, rho=1.0, thickness=2.0),
            crust=IsotropicLayer(alpha=3.0, beta=1.5, rho=2.6, thickness=1.0),
            crust_thickness=1.0,
            halfspace=IsotropicLayer(alpha=5.0, beta=3.0, rho=3.2, thickness=np.inf),
            n_crust_planes=1,
        )
        p_samples = np.array([0.05, 0.1])
        omega = np.array([2 * np.pi * 5.0 + 0.4j])
        kw = dict(n_kz=256, kz_max=40.0, tol=1e-11, maxiter=200)

        r1, _ = dressed_reflectivity(stack, base, p_samples, omega, **kw)
        r2, _ = dressed_reflectivity(stack, doubled, p_samples, omega, **kw)

        err = np.max(np.abs(r2 - r1)) / np.max(np.abs(r1))
        assert err < 1e-8, (
            f"tiling the contrast changed the specular reflectivity by {err:.3e}; "
            f"the lattice period is leaking into the physics"
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n seismic python -m pytest tests/test_reflectivity_25d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marine3d.reflectivity_25d'`.

- [ ] **Step 3: Implement the diffraction gate**

An order $n \neq 0$ is open when the lateral wavenumber $k_x^{(n)} = \omega p + 2\pi n / L$ satisfies $|k_x^{(n)}| < \omega / \alpha_{\min}$, i.e. it can propagate in the slowest medium. Raise on the first such $n$, naming $n$, $p$, $\omega$, $L$, the offending $k_x^{(n)}$, and the threshold — plus the recovery (shorten $L$, i.e. reduce `n_x · pitch`, or cap the frequency).

- [ ] **Step 4: Implement `dressed_reflectivity`**

For each $\omega$ and each incident $p$: gate; build the matvec at $k_{\text{bloch}} = \omega p$; build the incident field in the reference stack; GMRES; extract the specular ($n=0$) upgoing-P amplitude at the seabed datum; assemble `(np_slow, nfreq)`. Return the array and the per-solve `GmresInfo` list so stagnation is visible to the caller, never swallowed.

- [ ] **Step 5: Run all Task 5 tests**

Run: `conda run -n seismic python -m pytest tests/test_reflectivity_25d.py -v`
Expected: 5 passed.

- [ ] **Step 6: Add the config blocks**

Add to `configs/marine_reference.yaml` — every key required, no defaults in Python:

```yaml
crust_heterogeneity:
  n_x: 8                  # voxels along x per plane
  pitch: 0.02             # km, voxel edge
  dlambda: 0.2            # GPa, contrast amplitude
  dmu: 0.1                # GPa
  drho: 0.02              # g/cm³
  validity_phi: 0.52      # renormalisation floor coefficient
  seed: 20260725          # RNG seed for the disorder realisation

solver:
  gmres_tol: 1.0e-10
  gmres_maxiter: 500
  # No P^x quadrature keys: the intra-plane kernel is built from the Kupradze
  # closed form (exact_propagator_9x9), which has no cutoffs to configure.
```

Extend `marine3d/config.py` with a loader that fails fast on any missing key, using the same 4-element diagnostic as `check_validity_floor`. Add a test asserting a missing key raises with all four elements present.

- [ ] **Step 7: Lint, full suite, commit**

```bash
cd ~/Desktop/Marine3D
conda run -n seismic python -m pytest tests/ -q --ignore=tests/test_resonance_far_field.py --ignore=tests/test_inter_voxel_propagator.py
conda run -n seismic ruff check marine3d/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format marine3d/
conda run -n seismic mypy marine3d/ --ignore-missing-imports
git add marine3d/reflectivity_25d.py marine3d/config.py configs/marine_reference.yaml
git commit -m "✨ reflectivity_25d: dressed RRd_PP with specular extraction and diffraction gate"
```

---

# Stage C — Arbiter and physics validation

## Task 6: The Wolfram 2½-D reference

**Files:**
- Create: `~/Desktop/MultipleScatteringCalculations/Mathematica/Marine25DReference.wl`
- Create: `~/Desktop/MultipleScatteringCalculations/Mathematica/marine25d_reference.json` (generated)

**Interfaces:**
- Produces: a JSON fixture with, for each of a handful of configurations, the input parameters and the exact `R_specular` complex value.

**The whole point is independence.** No splitting, no sweeps, no FFT, no reuse of the Python decomposition. Assemble the full system and invert it directly.

- [ ] **Step 1: Write the reference script**

Structure:

1. Build the stratified Green's function between crust plane positions directly from the layer stack (`ocean | crust | half-space`), symbolically where possible and at 20+ digits otherwise.
2. Build the per-voxel $T_0$ from $(\Delta\lambda, \Delta\mu, \Delta\rho)$ using the same *physics definition* as `sub_cell_tmatrix_9x9` — but derived independently, not transcribed from the Python.
3. Assemble the **full** dense $(\mathbf{I} - G T)$ over all $N_z \cdot N_x$ voxels for a fixed incident $p$, and solve with `LinearSolve`.
4. Extract the specular reflected amplitude.
5. Emit JSON.

Apply the established pitfalls:
- `NIntegrate` on singular integrands: `PrecisionGoal -> 12, AccuracyGoal -> 12, MaxRecursion -> 25`.
- Sequential integration for rectangular-domain master integrals, with `Assumptions` on the inner integrals and `GenerateConditions -> False` at every step.
- Return values as `ToString[Chop[Re[N[expr, 20]]]]` (and separately the imaginary part) so `wolframclient` can deserialise them.

Configurations to emit: $N_z \in \{1, 2\}$, $N_x \in \{2, 4\}$, contrast amplitudes $\{0.01, 0.05\}$, two incident slownesses. Eight rows is plenty — the fixture's job is independence, not coverage.

- [ ] **Step 2: Run it headless**

Run:
```bash
cd ~/Desktop/MultipleScatteringCalculations
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/Marine25DReference.wl
```
Expected: writes `Mathematica/marine25d_reference.json`, and prints a summary table of the eight rows.

- [ ] **Step 3: Commit**

```bash
cd ~/Desktop/MultipleScatteringCalculations
git add Mathematica/Marine25DReference.wl Mathematica/marine25d_reference.json
git commit -m "✨ Mathematica: independent 2½-D marine reference by direct inversion"
```

---

## Task 7: Close the validation ladder

**Files:**
- Create: `tests/test_marine25d_validation.py`
- Copy: `Mathematica/marine25d_reference.json` → `~/Desktop/Marine3D/tests/fixtures/marine25d_reference.json`

Each class below is one rung. Run them in order; a failure localises to the rung.

- [ ] **Step 1: Rung 5 — against the Wolfram reference**

Load the fixture, rebuild each configuration in Python, and require agreement to `1e-8` relative. This is the strongest gate in the plan: the two paths share no algorithmic structure.

On failure, do **not** tune tolerances. Bisect by rung: if rung 2 (uniform plane) passes and rung 5 fails, the error is in the lateral coupling; if rung 2 also fails, it is in the $p$-diagonal path or the $T_0$ basis.

- [ ] **Step 2: Rung 2 — laterally uniform plane vs Kennett**

Set every voxel in a plane to the same contrast. The result must equal a Kennett stack in which that plane is replaced by a homogeneous layer with the effective moduli. Tolerance is bounded below by the single-site $T_0$ form-factor accuracy — state the achieved figure in the test's failure message rather than asserting a tighter bound than the physics supports.

- [ ] **Step 3: Rung 4 — periodic plane vs the Kambe layer R/T**

Consume the stored, already-validated Kambe layer $R/T$ values from the intra-plane energy-balance work as **numbers**, copied into the fixture directory. Do not import across repos. Tolerance: the quadrature floor of that reference.

- [ ] **Step 4: Rung 8 — reciprocity and energy balance**

Reciprocity: source–receiver exchange, target ≲ 1e-7. Energy balance: with damping off (`Q = inf`, real $\omega$) and no open diffraction order, $|R|^2 + |T|^2 = 1$, target ≲ 1e-4. Phase 3b achieved 4.1e-8 and 1.2e-5 respectively on the intra-plane problem; treat those as the standard.

- [ ] **Step 5: Rung 9 — the thesis Ch6 convergence signature**

Record GMRES iteration counts at 5% and 10% heterogeneity. Thesis Ch6 reports ≈8 and ≈28. Assert order-of-magnitude agreement (say, 4–16 and 14–56) and **print the measured counts** — they go into the LaTeX validation table in Task 8.

- [ ] **Step 6: Run the whole ladder and record every number**

Run: `conda run -n seismic python -m pytest tests/test_marine25d_validation.py -v -s`

Capture the measured residual for every rung. These are the numbers Task 8 writes into the note. Do not proceed to Task 8 with projected values.

- [ ] **Step 7: Commit**

```bash
cd ~/Desktop/Marine3D
git add marine3d/
git commit -m "✅ validation: 2½-D disorder-resolved ladder rungs 1-9"
```

(Tests are gitignored; this commit captures any source fixes the ladder forced.)

---

# Stage D — The product

## Task 8: The LaTeX companion note

**Files:**
- Create: `~/Desktop/MultipleScatteringCalculations/LatexPDFs/Marine25DDisorderResolved/Marine25DDisorderResolved.tex`

Follow the established companion-note convention exactly — read `LatexPDFs/IntraPlaneSpectralSweep/IntraPlaneSpectralSweep.tex` first and match its preamble: `% !TEX program = lualatex`, `fontspec` with Latin Modern Roman, `amsmath`/`amssymb`/`mathtools`, `geometry` at 1in margins, `microtype`, `booktabs`, `tikz`, `hyperref` loaded late with the same colour scheme, and `\hypersetup` with `pdftitle`/`pdfauthor`.

- [ ] **Step 1: Write the note**

Sections:

1. **Anchor.** Thesis Ch5: `LSmat`, `Ssplit`, `PstratDef`, `ULsweeps`/`Upsweep`/`Downsweep`. State precisely what is inherited and what is extended — per-voxel screens replacing one scatterer per interface, and the introduction of $\mathbf{P}^x$.
2. **The 2½-D disorder-resolved specialisation.** The periodic $x$-lattice as an explicit modelling commitment, not an implementation detail. Bloch incidence versus lattice harmonics, and why the specular amplitude is a complete description only while no other order is open.
3. **The operator split.** The $G_{ii}$ contract as an equation: $G^{\text{strat}}_{ii} = G^{\text{direct}}_{ii} + G^{\text{reverb}}_{ii}$, with each term's home named.
4. **The algorithm.** The domain-alternating matvec as a numbered algorithm environment, with per-iterate costs. Born as one Jacobi iterate; GMRES as the resummation.
5. **The diffraction gate.** $|\omega p + 2\pi n/L| < \omega/\alpha_{\min}$ as the open-order condition, and its consequence.
6. **Validation.** The table.

- [ ] **Step 2: Fill the validation table with the measured numbers from Task 7**

| Rung | Gate | Measured |
|---|---|---|
| 1 | $\Delta = 0$ → Phase-1 reflectivity | *(from Task 7)* |
| 2 | Uniform plane → Kennett | *(from Task 7)* |
| 3 | $G_{ii}$ contract | *(from Task 3)* |
| 4 | Periodic plane → Kambe $R/T$ | *(from Task 7)* |
| 5 | Disordered → Wolfram direct solve | *(from Task 7)* |
| 6 | GMRES ≡ dense | *(from Task 4)* |
| 7 | Born = one iterate | *(from Task 4)* |
| 8 | Reciprocity; energy | *(from Task 7)* |
| 9 | Convergence signature | *(from Task 7)* |

**Actual measured residuals only.** Any rung that did not pass is recorded as failed, with the measured value — per `CLAUDE.md`, a formula in a `.tex` without a passing check behind it is unvalidated and must be labelled so.

- [ ] **Step 3: Compile twice**

```bash
cd ~/Desktop/MultipleScatteringCalculations/LatexPDFs/Marine25DDisorderResolved
/usr/local/bin/lualatex -interaction=nonstopmode Marine25DDisorderResolved.tex
/usr/local/bin/lualatex -interaction=nonstopmode Marine25DDisorderResolved.tex
```
Expected: `Marine25DDisorderResolved.pdf` produced, no unresolved references. Missing packages: `tlmgr install <pkg>`.

- [ ] **Step 4: Commit**

```bash
cd ~/Desktop/MultipleScatteringCalculations
git add LatexPDFs/Marine25DDisorderResolved/
git commit -m "📝 LatexPDFs: 2½-D disorder-resolved marine scattering companion note"
```

- [ ] **Step 5: Update the Phase-2 status in the plan header**

Add a Status section to this file recording: what closed, the measured ladder numbers, any deferred items, and the Phase 3 resume point. Commit with `📝 plan: Phase 2 complete status + Phase 3 resume point`.

---

## Self-Review

**1. Spec coverage.** §3 seam → Task 5. §4 model → Task 1. §4.1 periodic lattice → Tasks 2, 5 (wrap-around test). §4.2 Bloch vs harmonics → Tasks 4 (k_bloch), 5 (specular extraction). §5.1 matvec → Task 4. §5.2 modules → Tasks 1–5, one each. §5.3 $G_{ii}$ contract → Task 3. §6 Track L → Task 8; Track W → Task 6; Track P → Tasks 1–5. §7 ladder: rung 1 → Task 5; rung 2 → Task 7; rung 3 → Task 3; rung 4 → Task 7; rung 5 → Tasks 6–7; rungs 6, 7 → Task 4; rungs 8, 9 → Task 7. §8 gates: diffraction → Task 5; wrap-around → Task 5; validity floor → Task 1; GMRES stagnation → Task 4 (`GmresInfo`). §9 non-goals → nothing implements them, as intended.

**2. Placeholders.** Tasks 1–5 carry complete test code. Tasks 6–8 specify structure and exact commands rather than full source: the Wolfram reference must be *independently derived* to have any evidential value, so pre-writing it here would defeat its purpose, and the LaTeX table is by construction unfillable until Task 7 produces the numbers. Both are deliberate, and both are stated as such in-place.

**3. Type consistency.** `CrustField` fields (`n_z`, `n_x`, `pitch`, `dlambda`, `dmu`, `drho`, `reference`) are used identically in Tasks 1–5. `apply_px(kernel, psi)` and `apply_pz(blocks, psi_hat)` keep argument order throughout. `psi` is `(n_z, n_x, 9)` everywhere in real space and `(n_z, n_kx, 9)` in the spectral domain. `sub_cell_tmatrix_9x9` is public from Task 1 Step 1 onward and referenced by that name only. `GmresInfo` (`n_iter`, `residuals`, `converged`) is defined in Task 4 and consumed in Task 5.

**Known gap, stated rather than hidden:** Task 2 Step 7 and Task 5 Steps 3–4 give the algorithm and the exact routines to call but not literal source, because the correct call into `horizontal_greens.py` depends on which of its two 9×9 entry points takes the reference medium explicitly — a fact to establish by reading it, not by guessing here. The tests that pin those steps are fully specified, so the contract is unambiguous even where the implementation is not pre-written.
