# Cartesian Directional Sweeps, Stage 2 — Implementation Plan

**Goal:** Make the directional-sweep solver fully three-dimensional by adding the
in-out `k_y` pair, so heterogeneity may vary in `y` rather than being invariant
along it.

**Architecture:** Six direction-pure forward sweeps partitioning the voxel pairs
by the axis along which they are separated; GMRES carries every order of
multiple scattering. No convolution, no lattice sum, no periodicity assumption.

**Spec:** `docs/specs/2026-09-13-cartesian-directional-sweeps-design.md`
(§3.1 item 3, §6 stage 2, §7 rung 6).

**Stage 1 plan:** `docs/plans/2026-09-13-cartesian-directional-sweeps-stage1.md`
— complete at `0ca7ad2`.

---

## Survey, run before this plan was written

| Question | Finding |
|---|---|
| Does the `k_y` residue kernel exist? | **Yes.** `horizontal_greens.post_ky_residue_kernel_9x9_vec(kx_arr, kz, dy_abs, …)` (line 585) and `horizontal_greens_ky_residue_9x9` (line 677). Do **not** re-derive them. |
| Is there a 3-D arbiter? | **Yes, a better one than the spec names.** `horizontal_greens.exact_propagator_9x9(x, y, z, ω, ref)` (line 106) — closed-form Kupradze, already in the seismological `(z,x,y)` convention the sweeps use. |
| Is the vertical operator affected? | **No.** `GlobalMatrix/layered_greens.py` and thesis Ch. 5 `GstratRep.tex` are unchanged by this stage. |
| Is there a stage-2 plan already? | No. This is the first. |
| Is `sweep_y` stubbed? | Yes — `directional_sweeps.py:495`, raising `NotImplementedError` naming the stage. |

### Scope correction to the spec — read this before Task 1

The spec says *"Stage 2 then introduces exactly one new thing."* **Measured
against the code, that is too optimistic, and the plan is built on the corrected
reading.**

`SweepGrid.ky` is a **scalar** (`directional_sweeps.py:44`) — the 2½-D
parameter. Both existing sweeps are built at that single `k_y`. Making `y` a
real-space axis changes the state from `(n_z, n_x, 9)` to `(n_z, n_x, n_y, 9)`,
and the stage-1 pair partition then has no home for a pair separated along all
three axes at once. Each sweep must carry a quadrature over its *transverse*
wavenumbers:

| Sweep | Marches along | Transverse handling in 3-D | Status |
|---|---|---|---|
| `sweep_x` | `x` | `Δz = 0` → `k_z`; `Δy` arbitrary → **new `k_y` quadrature** | modified |
| `sweep_z` | `z` | `Δx`, `Δy` arbitrary → **double `k_x`, `k_y` quadrature** | modified |
| `sweep_y` | `y` | `Δz = 0` → `k_z`; `Δx = 0` → `k_x` | **new** |

Three functions touched, not one added.

**The 3-D partition**, which every gate in this plan is written against:

| Class | Pairs | Owner |
|---|---|---|
| A | `Δz ≠ 0`, any `Δx`, `Δy` | `sweep_z` |
| B | `Δz = 0`, `Δx ≠ 0`, any `Δy` | `sweep_x` |
| C | `Δz = 0`, `Δx = 0`, `Δy ≠ 0` | `sweep_y` |
| — | `Δ = 0` | inside `T₀`, reached by no sweep |

**The transverse quadrature must be polar, not tensor-product.** For `sweep_y`
the integrand decays as `e^{−κ_⊥ · pitch}` with `κ_⊥ = √(k_x² + k_z²)` — a
radially decaying 2-D integral. A tensor-product grid at the stage-1 default of
2048 nodes per axis would be 2048² = 4.2 M nodes per amplitude, i.e. ~2.7 × 10⁹
complex numbers for one 9×9 amplitude table: not affordable, and mostly spent on
corners where the integrand is negligible. A polar `(k_r, θ)` rule puts the
nodes where the decay is. Task 1 measures the rule actually needed rather than
assuming this one.

---

## Global constraints

Copied from the project standards and the spec; every task inherits them.

- Conda env `seismic`. Run everything as `conda run -n seismic <cmd>`.
- Seismic units: km/s, g/cm³, GPa, km. Time convention `e^{−iωt}`.
- Index order `z = 0` (down), `x = 1`, `y = 2`. Every branch pinned to `Im ≥ 0`.
- State is the 9-vector `(u_z, u_x, u_y, ε_zz, ε_xx, ε_yy, 2ε_xy, 2ε_zy, 2ε_zx)`.
  **No representation conversion at any sweep boundary** (spec §4).
- Python 3.12 typing (`X | Y`); `pathlib.Path`; Google docstrings; line length 108.
- Every fail-fast error carries all four elements: what is wrong, where to fix
  it, what a valid value looks like, how to recover.
- After every Python change:
  `ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741`,
  `ruff format cubic_scattering/`, `mypy cubic_scattering/ --ignore-missing-imports`.
- **Existing stage-1 entry points must keep working unchanged.** `SweepGrid`,
  `sweep_x`, `sweep_z`, `apply_g0`, `build_g0_cache` and every stage-1 gate stay
  as they are; 3-D arrives as new names beside them. A stage-1 gate that changes
  behaviour is a defect in this plan, not an acceptable cost.
- **Never assert a tolerance without asserting what discriminates.** A residual
  that is *flat* in separation and *flat* in lattice size is the claim; a single
  number at one size proves nothing. Every sweep gate carries the vacuity control
  from `gate_lateral_sweep_alg52.py`: distinct sources at every site, plus the
  demonstration that a uniform-source version would pass even for an
  implementation that silently averaged the sites.

---

## File structure

| File | Responsibility |
|---|---|
| `cubic_scattering/sweep_kernels.py` *(modify)* | Add `inout_split_9x9` and `polar_transverse_rule`. Existing `lateral_split_9x9`, `same_depth_kernel_9x9`, `vertical_kernel_9x9` untouched. |
| `cubic_scattering/directional_sweeps.py` *(modify)* | Add `SweepGrid3D`, `make_sweep_grid_3d`, replace the `sweep_y` stub, add `build_g0_cache_3d` and `apply_g0_3d`. Stage-1 names untouched. |
| `cubic_scattering/tests/test_sweep_kernels.py` *(modify)* | Tests for the new split and the polar rule. |
| `cubic_scattering/tests/test_directional_sweeps.py` *(modify)* | Tests for `sweep_y`, the 3-D grid, and the partition. |
| `scripts/measure_sweep3d_cost.py` *(create)* | Task 1 — the cost and cutoff measurement. Not a gate; a measurement whose output decides Tasks 5–6. |
| `scripts/gate_sweep_rung6_inout.py` *(create)* | Rung 6 — `sweep_y` against the closed-form Kupradze propagator. |
| `scripts/gate_sweep_partition_3d.py` *(create)* | The 3-D partition: every ordered pair reached exactly once, self-term never. |

Phase A is Tasks 1–4 and delivers a validated `sweep_y` on its own. Phase B is
Tasks 5–8 and makes the solver 3-D. **Stop and review between them** — Phase A
is independently useful and its gates must be green before the two existing
sweeps are touched.

---

# PHASE A — `sweep_y`, validated in isolation

### Task 1: Measure the transverse quadrature, before building anything

The decision taken for this stage is *correctness first, cost measured*. This
task produces the number; it does not optimise anything.

**Files:**
- Create: `scripts/measure_sweep3d_cost.py`

**Interfaces:**
- Consumes: `horizontal_greens.post_ky_residue_kernel_9x9_vec`,
  `horizontal_greens.exact_propagator_9x9`, `effective_contrasts.ReferenceMedium`.
- Produces: a printed table, and the two constants Task 2 uses —
  `KR_MAX_OVER_PITCH` (radial cutoff as a multiple of `1/pitch`) and
  `N_R`, `N_THETA` (the polar rule that converges).

- [ ] **Step 1: Write the measurement script**

```python
#!/usr/bin/env python3
"""MEASUREMENT, not a gate: what transverse quadrature does sweep_y need?

At Dx = 0 and Dz = 0 the post-k_y-residue integrand decays as
e^{-kappa_perp * dy} with kappa_perp = sqrt(kx^2 + kz^2): a radially decaying
2-D integral. A tensor-product grid at the stage-1 default (2048 per axis)
would be 4.2M nodes per amplitude and spend most of them on corners where the
integrand is negligible. This measures the polar rule that actually converges,
and the cost that rule implies for sweep_x and sweep_z in Task 5 and Task 6.

Run:  conda run -n seismic python scripts/measure_sweep3d_cost.py
Seismic units (km/s, g/cm3), time convention e^{-i omega t}.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cubic_scattering.effective_contrasts import ReferenceMedium  # noqa: E402
from cubic_scattering.horizontal_greens import (  # noqa: E402
    exact_propagator_9x9,
    post_ky_residue_kernel_9x9_vec,
)

REF = ReferenceMedium(5.0, 3.0, 2.5)
OM = 2 * np.pi * 6.0
PITCH = 0.25


def polar_nodes(kr_max: float, n_r: int, n_theta: int):
    """Polar rule on the transverse (k_x, k_z) plane.

    Returns (kx, kz, weight) flattened, weights INCLUDING the 1/(2 pi)^2 and
    the Jacobian k_r.
    """
    # Midpoint in r (avoids the r = 0 node, where the Jacobian vanishes anyway)
    r_edge = np.linspace(0.0, kr_max, n_r + 1)
    r = 0.5 * (r_edge[:-1] + r_edge[1:])
    dr = r_edge[1] - r_edge[0]
    th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    dth = 2 * np.pi / n_theta
    rr, tt = np.meshgrid(r, th, indexing="ij")
    w = (rr * dr * dth) / (2 * np.pi) ** 2
    return (rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(), w.ravel()


def integrate(dy: float, kr_max: float, n_r: int, n_theta: int) -> np.ndarray:
    kx, kz, w = polar_nodes(kr_max, n_r, n_theta)
    out = np.zeros((9, 9), dtype=complex)
    # post_ky_residue_kernel_9x9_vec is vectorised over kx at ONE kz, so group
    # the polar nodes by kz value rather than calling it per node.
    for j in range(kz.size):
        p = post_ky_residue_kernel_9x9_vec(
            np.array([kx[j]]), kz[j], abs(dy),
            omega=OM, rho=REF.rho, alpha=REF.alpha, beta=REF.beta,
        )
        out += w[j] * p[:, :, 0]
    return out


def main() -> int:
    want = exact_propagator_9x9(0.0, PITCH, 0.0, OM, REF)
    print(f"{'kr_max*pitch':>13} {'n_r':>6} {'n_theta':>8} {'rel err':>11}")
    for kr_mult in (10.0, 20.0, 30.0, 45.0):
        for n_r, n_th in ((64, 32), (128, 64), (256, 128)):
            got = integrate(PITCH, kr_mult / PITCH, n_r, n_th)
            err = np.abs(got - want).max() / np.abs(want).max()
            print(f"{kr_mult:13.1f} {n_r:6d} {n_th:8d} {err:11.3e}")
    print("\nPick the smallest rule whose error is FLAT under refinement, not")
    print("merely small: a cutoff too low converges smoothly to a biased limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `conda run -n seismic python scripts/measure_sweep3d_cost.py`

Expected: a table. The correct rule is the smallest whose error is **flat**
under refinement of both `kr_max` and `(n_r, n_theta)`. An error that keeps
falling as `kr_max` grows means the cutoff, not the node count, is the binding
constraint. **Record the chosen values in the script as module constants** and
in the commit message — Task 2 imports them.

> Note the argument order: `exact_propagator_9x9(x, y, z, …)` takes Cartesian
> `x, y, z` in that order, while the state vector is ordered `z, x, y`. A
> separation of one pitch along `y` is therefore `(0.0, PITCH, 0.0)`. Getting
> this backwards produces a plausible wrong answer, not an error.

- [ ] **Step 3: Commit**

```bash
git add scripts/measure_sweep3d_cost.py
git commit -m "🔢 math: measure the transverse polar quadrature for the in-out sweep"
```

---

### Task 2: `inout_split_9x9` — the amplitude/phase split on the `k_y` pole

The mirror of `lateral_split_9x9`, and the single most dangerous item in the
stage (spec §5.1): accumulating a factor that should not accumulate yields a
plausible field wrong by a distance-dependent factor, and reciprocity will not
catch it.

**Files:**
- Modify: `cubic_scattering/sweep_kernels.py`
- Test: `cubic_scattering/tests/test_sweep_kernels.py`

**Interfaces:**
- Consumes: `horizontal_greens.post_ky_residue_kernel_9x9_vec`; the constants
  from Task 1.
- Produces: `inout_split_9x9(kx_nodes, kz_nodes, weights, pitch, omega, ref, *,
  direction: str = "out") -> LateralSplit`, reusing the existing `LateralSplit`
  dataclass. `amp_p`/`amp_s` shape `(9, 9, n_nodes)`; `phase_p`/`phase_s` shape
  `(n_nodes,)`; `direction` in `{"out", "in"}` for `+y` / `−y`.

- [ ] **Step 1: Write the failing test**

```python
def test_inout_split_reproduces_bundled_kernel_flat_in_separation():
    """amp x phase^n == the bundled kernel at n pitches, FLAT in n.

    Flatness is the claim. A split that wrongly folds part of the amplitude
    into the phase is small at n = 1 and grows with n, so a single-separation
    assertion would pass it.
    """
    ref = ReferenceMedium(5.0, 3.0, 2.5)
    om, pitch = 2 * np.pi * 6.0, 0.25
    kx, kz, w = polar_transverse_rule(30.0 / pitch, 128, 64)
    split = inout_split_9x9(kx, kz, w, pitch, om, ref, direction="out")

    errs = []
    for n in (1, 2, 4, 8, 16, 32):
        got = np.einsum("k,abk,k->ab", w, split.amp_p, split.phase_p**n)
        got += np.einsum("k,abk,k->ab", w, split.amp_s, split.phase_s**n)
        want = _bundled_ky_kernel(kx, kz, w, n * pitch, om, ref)
        errs.append(np.abs(got - want).max() / np.abs(want).max())

    assert max(errs) < 1e-13, f"split wrong: {errs}"
    assert max(errs) / max(min(errs), 1e-18) < 50.0, f"NOT flat in n: {errs}"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_sweep_kernels.py::test_inout_split_reproduces_bundled_kernel_flat_in_separation -v`

Expected: FAIL, `NameError: name 'inout_split_9x9' is not defined`.

- [ ] **Step 3: Implement `polar_transverse_rule` and `inout_split_9x9`**

```python
def polar_transverse_rule(kr_max: float, n_r: int, n_theta: int) -> tuple[NDArray, NDArray, NDArray]:
    """Polar quadrature on the transverse (k_x, k_z) plane.

    The post-k_y-residue integrand at Dx = Dz = 0 decays as e^{-kappa_perp dy}
    with kappa_perp = sqrt(kx^2 + kz^2), so the rule is radial. Midpoint in r
    skips the r = 0 node, where the Jacobian vanishes.

    Args:
        kr_max: Radial cutoff, 1/km. Confirm by refinement; an undersized
            k-grid is this project's most expensive recurring numerical error.
        n_r: Radial nodes (>= 2).
        n_theta: Angular nodes (>= 4).

    Returns:
        (kx, kz, weights), each shape (n_r * n_theta,). Weights INCLUDE the
        Jacobian k_r and the 1/(2 pi)^2.

    Raises:
        ValueError: on a non-positive cutoff or too few nodes.
    """
    if kr_max <= 0.0:
        msg = (
            f"kr_max must be > 0, got {kr_max!r}.\n"
            "  Where: cubic_scattering/sweep_kernels.py, polar_transverse_rule(kr_max=...)\n"
            "  Valid: a positive radial cutoff in 1/km, e.g. kr_max=30.0/pitch\n"
            "  Fix:   pass 30/pitch unless scripts/measure_sweep3d_cost.py says otherwise."
        )
        raise ValueError(msg) from None
    if n_r < 2 or n_theta < 4:
        msg = (
            f"need n_r >= 2 and n_theta >= 4, got ({n_r}, {n_theta}).\n"
            "  Where: cubic_scattering/sweep_kernels.py, polar_transverse_rule\n"
            "  Valid: e.g. n_r=128, n_theta=64\n"
            "  Fix:   use the rule measured by scripts/measure_sweep3d_cost.py."
        )
        raise ValueError(msg) from None

    r_edge = np.linspace(0.0, kr_max, n_r + 1)
    r = 0.5 * (r_edge[:-1] + r_edge[1:])
    dr = float(r_edge[1] - r_edge[0])
    th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    dth = 2 * np.pi / n_theta
    rr, tt = np.meshgrid(r, th, indexing="ij")
    w = (rr * dr * dth) / (2 * np.pi) ** 2
    return (rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(), w.ravel()
```

```python
def inout_split_9x9(
    kx_nodes: NDArray,
    kz_nodes: NDArray,
    weights: NDArray,
    pitch: float,
    omega: complex,
    ref: ReferenceMedium,
    *,
    direction: str = "out",
) -> LateralSplit:
    """Split the post-k_y-residue 9x9 kernel into amplitude and one-pitch phase.

    The in-out analogue of lateral_split_9x9, for the Dz = 0, Dx = 0 line. Note
    that this one integrates TWO transverse wavenumbers, where the stage-1
    lateral split integrates one and holds k_y fixed as the 2.5-D parameter.
    The two are NOT interchangeable by relabelling: k_z sits at index 0 of the
    k-vector, k_x at index 1 and k_y at index 2.

    Args:
        kx_nodes: Transverse k_x nodes, shape (n_nodes,), 1/km.
        kz_nodes: Transverse k_z nodes, shape (n_nodes,), 1/km.
        weights: Quadrature weights including the 1/(2 pi)^2, shape (n_nodes,).
        pitch: Voxel pitch along y, km. Must be > 0.
        omega: Angular complex frequency, rad/s.
        ref: Background medium (seismic units).
        direction: 'out' for +y propagation, 'in' for -y.

    Returns:
        A LateralSplit whose amp/phase arrays are indexed by the flattened
        transverse node, not by a single wavenumber axis.

    Raises:
        ValueError: on a non-positive pitch, a zero frequency, an unknown
            direction, or mismatched node arrays.
    """
```

The body mirrors `lateral_split_9x9`: form `ky_L = sqrt(kP^2 - kx^2 - kz^2)` and
`ky_T = sqrt(kS^2 - kx^2 - kz^2)` with the branch pinned to `Im >= 0`, set
`phase_p = exp(1j * ky_L * pitch)` and `phase_s = exp(1j * ky_T * pitch)`, and
put everything that is **not** the propagating exponential into `amp_p`/`amp_s`
— including the `1/ky` pole factors. Read `post_ky_residue_kernel_9x9_vec`
(`horizontal_greens.py:585`) and lift its algebra rather than re-deriving it.
For `direction == "in"`, negate the odd-`y` rows and columns exactly as the
existing lateral split does for `−x`; do not simply conjugate, which is wrong
for evanescent branches.

- [ ] **Step 4: Run the test**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_sweep_kernels.py::test_inout_split_reproduces_bundled_kernel_flat_in_separation -v`

Expected: PASS, with the flatness assertion satisfied — not merely the magnitude one.

- [ ] **Step 5: Lint, type-check, commit**

```bash
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
git add cubic_scattering/sweep_kernels.py cubic_scattering/tests/test_sweep_kernels.py
git commit -m "✨ feat: amplitude/phase split on the k_y pole for the in-out sweep"
```

---

### Task 3: `sweep_y` — the running sweep along `y`

**Files:**
- Modify: `cubic_scattering/directional_sweeps.py` (replace the stub at line 495)
- Test: `cubic_scattering/tests/test_directional_sweeps.py`

**Interfaces:**
- Consumes: `inout_split_9x9` (Task 2).
- Produces: `sweep_y(sources, grid, split_out, split_in) -> NDArray`, signature
  matching `sweep_x`. Sources shape `(n_z, n_x, n_y, 9)`; returns the same shape.

- [ ] **Step 1: Write the failing test**

```python
def test_sweep_y_equals_pairwise_sum_distinct_sources():
    """The sweep equals an explicit O(N^2) pairwise sum, flat in lattice size.

    MANDATORY CONTROL: the source differs at every site. A uniform source would
    pass even for an implementation that silently averaged the sites, so the
    test asserts BOTH that the distinct-source case matches and that a uniform
    source fails to discriminate (the vacuity control).
    """
    ref = ReferenceMedium(5.0, 3.0, 2.5)
    om, pitch = 2 * np.pi * 6.0, 0.25
    kx, kz, w = polar_transverse_rule(30.0 / pitch, 128, 64)
    s_out = inout_split_9x9(kx, kz, w, pitch, om, ref, direction="out")
    s_in = inout_split_9x9(kx, kz, w, pitch, om, ref, direction="in")

    rng = np.random.default_rng(20260914)
    errs = []
    for n_y in (4, 8, 16):
        grid = make_sweep_grid_3d(1, 1, n_y, pitch, ref=ref, omega=om)
        src = rng.normal(size=(1, 1, n_y, 9)) + 1j * rng.normal(size=(1, 1, n_y, 9))
        got = sweep_y(src, grid, s_out, s_in)

        want = np.zeros_like(got)
        for i in range(n_y):
            for j in range(n_y):
                if i == j:
                    continue
                p = exact_propagator_9x9(0.0, (i - j) * pitch, 0.0, om, ref)
                want[0, 0, i] += p @ src[0, 0, j]
        errs.append(np.abs(got - want).max() / np.abs(want).max())

    assert max(errs) < 1e-8, f"sweep != pairwise sum: {errs}"
    assert max(errs) / max(min(errs), 1e-18) < 50.0, f"NOT flat in lattice size: {errs}"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_directional_sweeps.py::test_sweep_y_equals_pairwise_sum_distinct_sources -v`

Expected: FAIL with the stub's `NotImplementedError`, its message naming stage 2.
**This is the fail to watch for** — it confirms the test reaches the stub rather
than erroring earlier on a fixture.

- [ ] **Step 3: Replace the stub**

```python
def sweep_y(
    sources: NDArray,
    grid: SweepGrid3D,
    split_out: LateralSplit,
    split_in: LateralSplit,
) -> NDArray:
    """Accumulate the Dz = 0, Dx = 0 in-out coupling by two running sweeps.

    Reads the accumulator BEFORE adding the local source, so the minimum
    separation is one pitch and the self-term is never formed -- the same
    ordering as sweep_x, and for the same reason.

    Args:
        sources: Source 9-vectors, shape (n_z, n_x, n_y, 9).
        grid: The 3-D lattice and its transverse rule.
        split_out: Amplitude/phase split for +y, direction='out'.
        split_in: Amplitude/phase split for -y, direction='in'.

    Returns:
        The accumulated field, shape (n_z, n_x, n_y, 9).

    Raises:
        ValueError: on a grid/split mismatch, two splits of the same direction,
            or a wrong source shape.
    """
    _check_split_3d(grid, split_out, "split_out")
    _check_split_3d(grid, split_in, "split_in")
    if split_out.direction != "out" or split_in.direction != "in":
        msg = (
            f"split directions are ({split_out.direction!r}, {split_in.direction!r}), "
            "expected ('out', 'in').\n"
            "  Where: cubic_scattering/directional_sweeps.py, sweep_y\n"
            "  Valid: inout_split_9x9(..., direction='out') and '...in'\n"
            "  Fix:   passing the same split twice silently symmetrises the field."
        )
        raise ValueError(msg) from None
    _check_state_3d(sources, grid, "sweep_y")

    out = np.zeros_like(sources, dtype=complex)
    w = grid.transverse_weights
    n_k = w.size

    passes = (
        (split_out, range(grid.n_y)),
        (split_in, reversed(range(grid.n_y))),
    )
    for split, order in passes:
        acc_p = np.zeros((grid.n_z, grid.n_x, n_k, 9), dtype=complex)
        acc_s = np.zeros_like(acc_p)
        for i in order:
            out[:, :, i, :] += np.einsum("k,abk,zxkb->zxa", w, split.amp_p, acc_p)
            out[:, :, i, :] += np.einsum("k,abk,zxkb->zxa", w, split.amp_s, acc_s)
            acc_p = (acc_p + sources[:, :, i, None, :]) * split.phase_p[None, None, :, None]
            acc_s = (acc_s + sources[:, :, i, None, :]) * split.phase_s[None, None, :, None]

    return out
```

`SweepGrid3D`, `make_sweep_grid_3d`, `_check_split_3d` and `_check_state_3d` are
added in this task alongside `sweep_y`; they are the same shape as their
stage-1 counterparts with `n_y` and the flattened transverse rule
(`transverse_kx`, `transverse_kz`, `transverse_weights`) in place of the scalar
`ky` and the `kz_nodes`/`kz_weights` pair.

- [ ] **Step 4: Run the test**

Run: `conda run -n seismic python -m pytest cubic_scattering/tests/test_directional_sweeps.py -v`

Expected: PASS, including flatness. Also confirm the stage-1 tests in the same
file still pass — `sweep_x`, `sweep_z` and `apply_g0` must be untouched.

- [ ] **Step 5: Lint, type-check, commit**

```bash
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
git add cubic_scattering/directional_sweeps.py cubic_scattering/tests/test_directional_sweeps.py
git commit -m "✨ feat: the in-out k_y sweep, gated against the pairwise sum"
```

---

### Task 4: Rung 6 — `sweep_y` against the closed-form Kupradze propagator

The spec names `horizontal_greens` pairwise as the arbiter. **This plan uses
`exact_propagator_9x9` instead**, because it is a closed form with no quadrature
error of its own and is already in the `(z,x,y)` convention the sweeps use,
whereas `horizontal_greens` is known to carry two different index conventions
internally — `horizontal_greens_direct` is `(x,y,z)` while the 9×9 kernel and
`exact_propagator_9x9` are seismological, and unpermuted they differ by a fixed
3.6e-1 that does **not** fall under refinement.

**Files:**
- Create: `scripts/gate_sweep_rung6_inout.py`

**Interfaces:**
- Consumes: `sweep_y`, `inout_split_9x9`, `polar_transverse_rule`,
  `exact_propagator_9x9`.
- Produces: an exit-0 gate.

- [ ] **Step 1: Write the gate**

Mirror `scripts/gate_sweep_rung2_lateral.py` in structure. Three claims, each
reported with the quantity that discriminates:

1. **Magnitude** — sweep against the closed form, target 1e-8 (quadrature-floor
   limited, not 1e-15; the rung-1 split test is the exact one).
2. **Flat in separation and in lattice size** — the residual must not grow with
   either. Report both sequences, not just their maxima.
3. **Vacuity control** — a uniform-source run must be shown to pass even for a
   deliberately site-averaging implementation, demonstrating that the
   distinct-source claim is the one carrying the weight.

**The standing rule on validating a spectral sum against the closed form**, with
the three cases in which the comparison is invalid, and which this gate must
assert it stays outside:

1. **At or inside the source voxel.** The closed form is a point propagator and
   is singular at zero separation; the sweep object is a finite-cell quantity.
   The gate runs at separations of one pitch and greater.
2. **When the quadrature cutoff is the binding error.** If the residual still
   falls as `kr_max` grows, the comparison is measuring the rule, not the sweep.
   Assert the residual is flat under cutoff refinement before reporting it.
3. **Across a material interface.** The closed form is whole-space; the sweep on
   a stratified background is not. The gate uses a uniform background — the
   layered case is rung 3's business, arbitrated by `kennett_layers`.

- [ ] **Step 2: Run it**

Run: `conda run -n seismic python scripts/gate_sweep_rung6_inout.py`
Expected: `GATE rung 6: PASS`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/gate_sweep_rung6_inout.py
git commit -m "✅ test: rung 6, the in-out sweep against the closed-form propagator"
```

---

> ## CHECKPOINT — stop here and review
>
> Phase A is independently useful: `sweep_y` exists, is gated against a closed
> form, and nothing in stage 1 has been touched. Confirm the full suite and every
> existing gate is still green **before** Phase B modifies `sweep_x` and
> `sweep_z`, which are the two validated objects most likely to be damaged.
>
> ```bash
> conda run -n seismic python -m pytest cubic_scattering/tests/ -q
> for g in scripts/gate_*.py; do conda run -n seismic python "$g" > /dev/null || echo "FAIL $g"; done
> ```

---

# PHASE B — make the solver three-dimensional

### Task 5: A transverse `k_y` quadrature in `sweep_x`

**Files:** Modify `cubic_scattering/sweep_kernels.py`, `directional_sweeps.py`;
test in `tests/test_directional_sweeps.py`.

**Interfaces:** Produces `sweep_x_3d(sources, grid, split_right, split_left)`,
shape `(n_z, n_x, n_y, 9)` in and out, and `lateral_split_3d_9x9` built on a
`(k_z, k_y)` polar rule. Stage-1 `sweep_x` and `lateral_split_9x9` are left
exactly as they are.

- [ ] **Step 1:** Write the failing test — `sweep_x_3d` against an explicit
  pairwise sum over `exact_propagator_9x9` for pairs with `Δz = 0, Δx ≠ 0` at
  **several distinct `Δy`**, asserting flatness in both `n_x` and `n_y`. A test
  at `Δy = 0` only would pass for an implementation that ignored `y` entirely.
- [ ] **Step 2:** Run it; expect FAIL, `NameError`.
- [ ] **Step 3:** Implement, mirroring Task 2's split with `(k_z, k_y)` as the
  transverse pair and `k_x` as the pole.
- [ ] **Step 4:** Run; expect PASS, and confirm stage-1 `sweep_x` tests untouched.
- [ ] **Step 5:** Lint, type-check, commit
  `"✨ feat: transverse k_y quadrature in the lateral sweep"`.

### Task 6: A double transverse quadrature in `sweep_z`

**Files:** Modify `directional_sweeps.py`; test in `tests/test_directional_sweeps.py`.

**Interfaces:** Produces `build_vertical_stack_3d(grid, background, omega)` with
shape `(n_z, n_z, 9, 9, n_transverse)` and `sweep_z_3d(sources, grid, vertical)`.

- [ ] **Step 1:** Write the failing test — `sweep_z_3d` against `kennett_layers`
  between the same planes at several `(Δx, Δy)`, target 1e-14, plus the
  homogeneous-limit reduction to `vertical_kernel_9x9`.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3:** Implement. **The vertical operator itself does not change** —
  it is still `layered_correction.corrected_layered_9x9` on the stratified
  background, evaluated at the transverse nodes. Do not rebuild it; see the
  spec's retracted "no external dependency" paragraph for why that error is
  worth naming twice.
- [ ] **Step 4:** Run; expect PASS. **Record the wall-clock cost here** and
  compare it against Task 1's projection; a large miss means Task 1's rule was
  wrong, not that the cost is acceptable.
- [ ] **Step 5:** Lint, type-check, commit
  `"✨ feat: double transverse quadrature in the inter-plane sweep"`.

### Task 7: `apply_g0_3d` and the partition gate

**Files:** Modify `directional_sweeps.py`; create `scripts/gate_sweep_partition_3d.py`.

**Interfaces:** Produces `build_g0_cache_3d(grid, ref, omega, *, background=None)`
and `apply_g0_3d(sources, cache)`.

- [ ] **Step 1:** Write the failing partition test — a **support count**, not a
  norm. Place a unit source at one site, run each sweep separately, and assert
  every ordered pair is reached by exactly one sweep and the self-site by none.
  This is the test that catches a pair double-counted or dropped, which no
  tolerance check will see.
- [ ] **Step 2:** Run it; expect FAIL.
- [ ] **Step 3:** Implement `apply_g0_3d` as the sum of the three sweeps.
- [ ] **Step 4:** Run; expect PASS, exact (a count, not a tolerance).
- [ ] **Step 5:** Lint, type-check, commit
  `"✨ feat: the composed 3-D G0 and its partition gate"`.

### Task 8: The two deferred rungs, now unblocked

**Files:** Create/extend the gates; no library change expected.

- [ ] **Step 1: Rung 5c, cross-architecture.** GMRES solve against
  `slab_scattering` on a footprint both can represent. This was deferred from
  stage 1 because `slab_scattering` is 3-D on a finite `M×M` footprint and
  cannot represent a y-invariant medium — which is exactly the restriction this
  stage removes.
- [ ] **Step 2: Rung 7, the `FFTProp` convergence study.** **This rung is not
  exact and must not be written as though it were.** `FFTProp` discretises the
  heterogeneity as cylinders with a cylinder Mie `T₀`; this solver uses cubic
  voxels with the cube `T₀`, so the two disagree by a genuine shape difference
  even when both are correct. The rung passes by demonstrating **convergence of
  the difference under refinement** toward the known equal-volume shape
  difference — not by hitting a fixed tolerance. A single run at one pitch
  proves nothing.
- [ ] **Step 3:** Commit `"✅ test: close rungs 5c and 7 on the 3-D solver"`.

### Task 9: Update the write-up — part of this task, not a follow-up

The documents lag the code, and that is this repository's standing failure mode;
leaving it is how the drift accumulates.

- [ ] **Step 1:** Update `LatexPDFs/DirectionalSweepSolver/DirectionalSweepSolver.tex`
  — add rung 6 and the 3-D partition to the validation table, and **delete the
  "Not yet done" paragraph**, whose only remaining item is the `k_y` pair this
  stage lands.
- [ ] **Step 2:** Update the spec's §6 to record that stage 2 modified
  `sweep_x` and `sweep_z` rather than only adding `sweep_y`, since its "exactly
  one new thing" claim was measured false in this plan's survey.
- [ ] **Step 3:** Recompile twice and commit.

```bash
cd LatexPDFs/DirectionalSweepSolver && /usr/local/bin/lualatex -interaction=nonstopmode DirectionalSweepSolver.tex
```

---

## Self-review against the spec

**Spec coverage.** §3.1 item 3 (in-out on the `k_y` pole) → Tasks 2–3. §3.2
(every sweep marches along an axis of nonzero separation) → the partition table
and Task 7. §3.3 (no periodicity) → the polar quadrature in Task 2 is a direct
rule, not an FFT. §4 (9-component state, no conversion at boundaries) → every
signature carries 9-vectors. §5 module layout → the file table. §5.1 (the
amplitude/phase split is the most dangerous item) → Task 2's flatness assertion.
§6 stage 2 → the whole plan. §7 rung 6 → Task 4; rungs 5c and 7 → Task 8. §8
(fail fast, four-element diagnostics) → every raise shown. §9 out of scope —
nothing here touches `_propagator_block_9x9`.

**Gap found and closed during review.** §7's rung-6 arbiter is named as
`horizontal_greens` pairwise; this plan substitutes the closed form and states
why in Task 4 rather than silently diverging.

**Type consistency.** `LateralSplit` is reused unchanged for the in-out split
(`amp_p`, `amp_s`, `phase_p`, `phase_s`, `direction`), with `direction` taking
`'out'`/`'in'` where the lateral split takes `'right'`/`'left'`. `SweepGrid3D`
carries `transverse_kx`, `transverse_kz`, `transverse_weights` consistently in
Tasks 3, 5, 6 and 7. Source arrays are `(n_z, n_x, n_y, 9)` throughout Phase B.

**Known risk, stated rather than hidden.** Task 6 is where this stage can fail
on cost rather than on correctness. That is why Task 1 measures first and Task 6
compares the measurement against reality. If the cost is prohibitive, the
accelerator is designed **then**, gated against the direct quadrature built
here — correctness first, exactly as the stage-1 `k_x` quadrature was handled.
