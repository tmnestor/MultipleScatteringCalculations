# Stratified Wrapper Correction — Implementation Plan

> **COMPLETE 13 Sep 2026.** Commits `f5af6e5`, `582e825`, `de8b079`, `d4b7b00`,
> `68152af`. Read this header and the closing sections; the task bodies below are
> the plan *as written beforehand*, kept for the record.
>
> **The checkboxes are deliberately left unticked.** Execution diverged from the
> written steps far enough that ticking them would misrepresent what happened:
> Tasks 1–3 collapsed entirely once the real cause was found, Task 5's experiment
> reversed the *reasoning* behind its own recommendation, and Task 6 discovered
> that one of the gates it was told to re-run is false. The narrative of what was
> actually done is in `docs/wrapper_problem_state_2026-09-13.md` §7; this file
> records what was *intended*, and the gap between the two is the useful part.
>
> **STATUS after the first hour: Tasks 1–3 are DONE and collapsed.**
> `K` needs no stratified generalisation. The recorded failure was a placement
> artefact — the GATE D pairs sat *on* material discontinuities, where `η_S` is
> two-valued and `K` is simply not defined. With the planes in a layer interior
> and a fast slab crossed twice between them, the clean law holds to **1.7e-15**
> over nine geometries (`scripts/gate_stratified_correction.py`).
>
> Tasks 1–3 below are superseded and kept only for the record — the two-half-space
> reference and the symbolic derivation are no longer needed.
>
> **Tasks 3a and 4 are also DONE.** `cubic_scattering/layered_correction.py` ships
> the correction with the placement guard; `test_layered_correction.py` has 14
> tests, all passing, including GATE F through the module and the thesis Box 5.3
> validation. The four gate scripts all exit 0 and now import the module rather
> than duplicating it. **Start at Task 5.**
>
> One naming correction against the text below: the source operator is
> `source_jump_operator` returning **(6, 9)**, not `source_operator_9x6`. The
> module also exposes `strain_from_state` (the A operator, reimplemented locally
> so the package does not depend on the sibling repository to be importable) and
> `correct_6x6`, the pure-numpy primitive that applies D1 and D2 to an
> already-computed raw 6×6.
>
> **Task 5 is DONE.** Measured, not argued: baseline 119 passed, D1 applied
> 118 passed / 1 failed. The single failure is an algebraic self-consistency
> identity, not a physical check, and cannot see a common-mode error in `G`.
> Verdict: carry locally, raise a note upstream. See
> `scripts/probe_upstream_fix.py`.
>
> **Tasks 6 and 7 are DONE. THE PLAN IS COMPLETE.** Commits `f5af6e5`,
> `582e825`, `de8b079`, `d4b7b00`, `68152af`.
>
> Task 6 produced the sharpest finding: **GATE E is a false gate**, incompatible
> with GATE F, and the exact reference fails it identically. It was passing
> because of a defect. That is the whole of N1. N5 was re-taken: the spread
> roughly halves but does not close, and the aliasing account stands.
>
> **The next piece of real work is not in this plan**: lattice-sum `P^z` into the
> discrete representation before adding it to `P^x` (build the real-space
> stratified propagator between voxel centres, then lattice-sum and FFT it
> through the pipeline `_build_slab_kernels` already uses). That is a
> construction, not a rescaling, and it wants its own plan. **Read "Carried
> forward: where the time actually goes" at the end of this document before
> starting it** — the profile says 98% of a solve is the kernel build and 1% is
> the GMRES loop, which decides what to vectorise and what to leave alone.

**Goal:** Extend the wrapper correction from the homogeneous limit to a stratified
reference, so that `P^z` and `P^x` can finally be composed in one resolvent.

**Architecture:** Build an exact two-half-space jump-response Green's function as an
absolute reference (the stratified analogue of the whole-space reference that broke
the homogeneous case), measure the correction operator **K** across a real interface
against it, then derive the same result symbolically from the two reciprocity laws
and check the two agree. Promote the result to a module, decide upstream-vs-carry by
experiment rather than opinion, and re-run the gates that were measured through the
uncorrected wrapper.

**Tech stack:** Python 3.12 / NumPy in conda env `seismic`; Wolfram Mathematica for
the symbolic leg; LaTeX (lualatex) for the documents.

**Spec:** `docs/wrapper_problem_state_2026-09-13.md` §6 (and the same material as §9
of `LatexPDFs/WrapperProblem/WrapperProblem.tex`). Read both before starting.

## Global Constraints

- Conda env is `seismic`. Run everything as `conda run -n seismic <cmd>`.
- Seismic units throughout: km/s, g/cm³, GPa, km. Time convention `e^{−iωt}`.
- Index order is z = 0 (down), x = 1, y = 2. Never introduce a radial/transverse
  frame into a result; `(r,t)` is an implementation detail of `layered_greens_6x6`,
  not physics.
- Voigt pairs `(zz, xx, yy, xy, zy, zx)`; strain rows carry engineering doubling
  (`2ε` on shears), stress source slots likewise (`2σ`). This is the origin of
  `W = diag(1,1,1,1,1,1,½,½,½)`.
- Every formula must pass in **both** Python and Mathematica before it goes into a
  `.tex`. One implementation agreeing with itself is not evidence.
- No heredocs in shell commands — they hang. Write files with the editor; pass
  commit messages via `git commit -F /tmp/msg.txt`.
- `cat` returns empty under automation. Use `head`/`sed -n`/`grep`.
- Lint after every Python change:
  `conda run -n seismic ruff check <path> --fix --ignore ARG001,ARG002,F841,E741`,
  then `ruff format <path>`, then `mypy <path> --ignore-missing-imports`.
  Max line length 108. Google-style docstrings. `raise ... from None` in except blocks.
- Do **not** modify `/Users/tod/Desktop/SeismicInversion/GlobalMatrix` until Task 5
  has decided the question by experiment.

## What is already established (do not re-derive)

In the homogeneous limit, with `k̂ = (k_x,k_y)/|k_∥|`, `P∥ = k̂k̂ᵀ`, `P⊥ = I₂ − P∥`,
and `η_S = √(s_S² − p²)` the vertical S slowness:

```
K = 1 ⊕ (−P∥ + η_S P⊥)              acting on (σ_zz, σ_xz, σ_yz)

G_true = diag(I₃, K) · Φ(G₆) · [ −diag(I₃, (−iω)² K) ]⁻¹
Φ(G₆) = G₆ with the [3:6,3:6] block multiplied by (−iω)
B      = J₆ A_src(−k)ᵀ W · diag(−I₃, I₆)
```

Verified: corrected `G₆` equals the exact whole-space jump response to 5e-15;
GATE F 1.1e-15; GATE D 5.1e-16 against the bare law `G₁ = J₆G₂ᵀJ₆`.

Measured to FAIL when `K` is built from each interface's own local `η_S` in a
stratified model: 0.009–0.398, no diagonal weight fits. That failure is the subject
of this plan.

## File structure

| File | Responsibility |
|---|---|
| `scripts/stratified_reference.py` (new) | Exact two-half-space jump-response 6×6. Reduces to the whole-space reference when the media match. |
| `scripts/gate_stratified_k.py` (new) | Measures `K_source` and `K_receiver` across a real interface against that reference. Decides whether `K` is local to one medium. |
| `Mathematica/StratifiedK.wl` (new) | Symbolic solve of the consistency equation relating `K` to the two reciprocity laws. Emits a JSON of the closed form. |
| `cubic_scattering/layered_correction.py` (new) | The public, tested correction: `k_operator`, `corrected_layered_6x6`, `source_operator_9x6`. |
| `cubic_scattering/tests/test_layered_correction.py` (new) | Regression tests pinning all of the above. |
| `scripts/gate_wrapper_resolution.py` (modify) | Import from the module instead of redefining. |
| `scripts/composed_matvec.py`, `scripts/discrete_pz_kernel.py` (modify) | N5 re-run through the corrected wrapper. |
| `LatexPDFs/WrapperProblem/WrapperProblem.tex`, `docs/wrapper_problem_state_2026-09-13.md` (modify) | Final documentation pass. |

---

## Task 1: Exact two-half-space jump-response reference

The homogeneous case was broken by one absolute comparison against a quantity with
no convention freedom. This builds the stratified equivalent. A single interface
between two half-spaces is the simplest genuinely stratified medium that still has a
closed-form answer, and it isolates the one thing at issue: what happens when the
source medium and the receiver medium differ.

**Files:**
- Create: `scripts/stratified_reference.py`
- Reuse: `scripts/audit_thesis_ch3_box53.py` (its `dz()` builds `D_z` from the
  thesis's `Peigen`/`SVeigen`/`SHeigen` + `epsdef`, already validated at 1e-16)
- Reuse: `scripts/gate_wrapper_resolution.py` (`whole_space_jump_response`)

**Interfaces:**
- Produces: `two_halfspace_jump_response(med_r, med_s, kx, ky, z_r, z_s, w) -> (6,6)`
  where `med_* = (rho, alpha, beta)` with complex velocities, `z_s > 0 > z_r`
  (interface at z = 0, source below, receiver above), basis
  `(u_z, u_x, u_y, T_zz, T_xz, T_yz)`.
- Produces: `mode_split(med, kx, ky, w) -> (E_d, E_u)`, each `(6,3)`.

- [ ] **Step 1: Write the failing reduction test**

The gate that matters: when both media are identical the two-half-space answer must
collapse onto the whole-space answer already in hand.

```python
# scripts/stratified_reference.py  (test block at the bottom, run via main)
def test_reduces_to_whole_space() -> float:
    """Identical media on both sides must reproduce the whole-space response."""
    med = (2.6, 4.0 - 0.1j, 2.22 - 0.0555j)
    w = 2 * np.pi * 48.0
    kx, ky = w * 0.12 * 0.6, w * 0.12 * 0.8
    got = two_halfspace_jump_response(med, med, kx, ky, z_r=-0.4, z_s=0.6, w=w)
    want = whole_space_jump_response(kx, ky, -1.0, w)   # dz = z_r - z_s = -1.0
    return float(la.norm(got - want) / la.norm(want))
```

- [ ] **Step 2: Run it and watch it fail**

```bash
conda run -n seismic python scripts/stratified_reference.py
```

Expected: `NameError: name 'two_halfspace_jump_response' is not defined`.

- [ ] **Step 3: Implement the mode split**

The jump `S` at `z_s` radiates upgoing above and downgoing below. With `z`
increasing downward, for `z < z_s` only upgoing exists, for `z > z_s` only downgoing:

```python
def mode_split(med, kx, ky, w):
    """Downgoing and upgoing eigenvector blocks, columns (P, SV, SH).

    Args:
        med: (rho, alpha, beta), velocities may be complex.
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        w: Angular frequency (rad/s).

    Returns:
        (E_d, E_u), each shape (6, 3), in basis (u_z,u_x,u_y,T_zz,T_xz,T_yz).
    """
    d = thesis_dz(med, kx, ky, w)     # (6,6), column order +P,+S,+H,-P,-S,-H
    return d[:, 0:3], d[:, 3:6]


def amplitudes_from_jump(med, kx, ky, w, jump):
    """Solve E_d a_d - E_u a_u = jump for the radiated amplitudes.

    Args:
        med: (rho, alpha, beta).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        w: Angular frequency (rad/s).
        jump: 6-vector state discontinuity at the source.

    Returns:
        (a_d, a_u), each shape (3,).
    """
    e_d, e_u = mode_split(med, kx, ky, w)
    sol = la.solve(np.column_stack([e_d, -e_u]), jump)
    return sol[:3], sol[3:]
```

- [ ] **Step 4: Implement the interface and the assembly**

Continuity of the full 6-vector at `z = 0` gives reflection `r` back into the source
medium and transmission `t` into the receiver medium:

```python
def two_halfspace_jump_response(med_r, med_s, kx, ky, z_r, z_s, w):
    """Exact jump-response 6x6 across one interface at z = 0.

    Source at z_s > 0 (medium med_s, below); receiver at z_r < 0 (medium med_r,
    above).  No other reflector exists, so the receiver sees only transmitted
    upgoing energy.

    Args:
        med_r: (rho, alpha, beta) of the receiver half-space.
        med_s: (rho, alpha, beta) of the source half-space.
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        z_r: Receiver depth (km), negative.
        z_s: Source depth (km), positive.
        w: Angular frequency (rad/s).

    Returns:
        6x6 whose column m is the state at z_r for a unit jump in slot m.
    """
    e_d_s, e_u_s = mode_split(med_s, kx, ky, w)
    e_u_r, _ = mode_split(med_r, kx, ky, w)[1], None
    out = np.zeros((6, 6), dtype=complex)
    for m in range(6):
        jump = np.zeros(6, dtype=complex)
        jump[m] = 1.0
        _, a_u = amplitudes_from_jump(med_s, kx, ky, w, jump)
        # propagate upward from z_s to the interface at z = 0
        a_if = phase(med_s, kx, ky, w, z_s) * a_u
        # continuity:  E_u_s a_if + E_d_s r = E_u_r t
        sol = la.solve(np.column_stack([e_d_s, -e_u_r]), -e_u_s @ a_if)
        t = sol[3:]
        # propagate upward from the interface to z_r
        out[:, m] = e_u_r @ (phase(med_r, kx, ky, w, -z_r) * t)
    return out
```

with a mode-wise phase helper:

```python
def phase(med, kx, ky, w, dist):
    """exp(i k_z c * dist) per mode, for propagation over `dist` km.

    Args:
        med: (rho, alpha, beta).
        kx: Horizontal wavenumber, x component (rad/km).
        ky: Horizontal wavenumber, y component (rad/km).
        w: Angular frequency (rad/s).
        dist: Propagation distance (km), positive.

    Returns:
        Shape (3,) complex, order (P, SV, SH).
    """
    _, alpha, beta = med
    kzp = vertical(w / alpha, kx, ky)
    kzs = vertical(w / beta, kx, ky)
    return np.exp(1j * np.array([kzp, kzs, kzs]) * dist)
```

- [ ] **Step 5: Run the reduction test until it passes**

```bash
conda run -n seismic python scripts/stratified_reference.py
```

Expected: `reduces_to_whole_space: < 1e-12`. If it does not, the usual suspects in
order: the sign convention in `E_d a_d − E_u a_u = jump`; the branch of the vertical
wavenumber (`Im ≥ 0` so `exp(i k_z d)` decays); the direction of `phase` (both
propagations here are **upward**, distances positive).

- [ ] **Step 6: Add a second, independent gate — reciprocity**

The reference must obey the bare law in its own right, with source and receiver
swapped across the interface:

```python
def test_reciprocity(med_a, med_b, kx, ky, w) -> float:
    """G(r<-s)(+k) = J6 [ G(s<-r)(-k) ]^T J6 for the exact reference."""
    g1 = two_halfspace_jump_response(med_a, med_b, kx, ky, -0.4, 0.6, w)
    g2 = two_halfspace_jump_response(med_b, med_a, -kx, -ky, -0.6, 0.4, w)
    return float(la.norm(g1 - J6 @ g2.T @ J6) / la.norm(g1))
```

Expected: `< 1e-12` with genuinely different media, e.g.
`med_a = (2.3, 3.2-0.08j, 1.8-0.045j)`, `med_b = (2.8, 5.0-0.125j, 2.8-0.07j)`.
This gate is load-bearing: it is the law the corrected `G₆` must satisfy, proven
here on an object with no implementation in it.

- [ ] **Step 7: Lint and commit**

```bash
conda run -n seismic ruff check scripts/stratified_reference.py --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format scripts/stratified_reference.py
conda run -n seismic mypy scripts/stratified_reference.py --ignore-missing-imports
git add scripts/stratified_reference.py
git commit -m "🔬 evidence: exact two-half-space jump-response reference"
```

---

## Task 2: Measure K across a real interface

**Files:**
- Create: `scripts/gate_stratified_k.py`
- Reuse: `scripts/stratified_reference.py`, `GlobalMatrix.layered_greens.layered_greens_6x6`

**Interfaces:**
- Consumes: `two_halfspace_jump_response`, `mode_split` from Task 1.
- Produces: a printed table of `K_source`, `K_receiver` per geometry, and a verdict
  line naming which medium each is built from.

- [ ] **Step 1: Build a two-layer `LayerModel` matching the analytic reference**

`layered_greens_6x6` needs an ocean layer at the top; make it thin, heavily damped
and far away so it does not contaminate, and put source and receiver in two thick,
contrasting layers either side of one interface.

```python
def two_layer_model(med_a, med_b, q=2.0):
    """Ocean + thick upper layer (med_a) + half-space (med_b)."""
    return LayerModel.from_arrays(
        alpha=[1.5, med_a[1].real, med_b[1].real],
        beta=[0.0, med_a[2].real, med_b[2].real],
        rho=[1.03, med_a[0], med_b[0]],
        thickness=[3.0, 8.0, np.inf],
        Q_alpha=[q, q, q],
        Q_beta=[1e10, q, q],
    )
```

- [ ] **Step 2: Write the failing measurement**

Solve for the row and column corrections separately, exactly as in the homogeneous
case, but now allowing them to differ:

```python
def measure_k(model, med_r, med_s, kx, ky, w, j, i):
    """Solve G6_corrected = L . Phi(G6) . S^-1 = reference for L and S.

    Returns:
        (k_row, k_col, residual) with k_row, k_col shape (3,3).
    """
    g6 = layered_greens_6x6(model, w, np.array([kx]), np.array([ky]),
                            source_iface=j, receiver_iface=i)[0].copy()
    g6[3:, 3:] *= (-1j * w)                       # D1
    ref = two_halfspace_jump_response(med_r, med_s, kx, ky, z_r, z_s, w)
    # displacement rows fix the source columns; then the traction rows fix k_row
    k_col = la.solve(ref[0:3, 3:6], g6[0:3, 3:6])
    k_row = ref[3:6, :] @ la.pinv(g6[3:6, :])
    resid = la.norm(g6 - la.inv(block(k_row)) @ ref @ block_col(k_col)) / la.norm(g6)
    return k_row, k_col, resid
```

- [ ] **Step 3: Run and record**

```bash
conda run -n seismic python scripts/gate_stratified_k.py
```

There is no pass/fail here yet — this step produces the numbers. Record, for at
least three contrasts and three `(p, azimuth)` pairs, whether:

- `k_col` matches `1 ⊕ (−P∥ + η_S^(source) P⊥)` — **K is local to the source medium**;
- `k_row` matches `1 ⊕ (−P∥ + η_S^(receiver) P⊥)` — **K is local to the receiver medium**;
- neither matches, in which case print the solved matrices and move to Step 4.

- [ ] **Step 4: Decide the ansatz, and say so in the script's output**

Three outcomes, each with a defined next move. Write the verdict into the script so
it is reproducible rather than remembered:

1. **Both local.** The homogeneous formula carries over with per-medium `η_S`. But
   this contradicts the recorded failure, so if it appears, first re-check that the
   failing run used the *interface's own* layer (`_interface_elastic_properties`
   returns layer `max(iface,1)`, which for an interface between two different layers
   is the one **below**; the correct choice may be the one above).
   **That off-by-one is the single most likely explanation of the recorded failure
   and should be checked before anything else.**
2. **`k_row` and `k_col` are each local but to *different* media than assumed.**
   Fix the lookup and re-run Task 2.
3. **Neither is local.** The correction is not a per-interface operator. Fall back
   to Task 3's derivation, which does not assume locality.

- [ ] **Step 5: Commit the measurement**

```bash
git add scripts/gate_stratified_k.py
git commit -m "🔬 evidence: K measured across a single interface"
```

---

## Task 3: Derive K from the two reciprocity laws

Independent of Task 2, and the cross-check the project's standard requires.

**Files:**
- Create: `Mathematica/StratifiedK.wl`
- Test: extend `scripts/gate_stratified_k.py` with a comparison against the emitted
  closed form.

**The derivation.** Write `G₁ = G(i←j)(+k)`, `G₂ = G(j←i)(−k)`. The raw 6×6 obeys the
medium-independent law verified at 6.3e-16:

```
G₁ = c · SD · J₆ G₂ᵀ J₆ · SD ,    SD = diag(1,−1,−1,−iω,+iω,+iω),  c = i/ω
```

The corrected object is `G^c = L Φ(G) S⁻¹` and must obey `G₁^c = J₆ (G₂^c)ᵀ J₆`.
Substituting and requiring the identity to hold as an operator statement separates
into one condition per index:

```
(A)   c · L_i · SD · J₆  =  J₆ · S_i⁻ᵀ           (receiver index)
(B)   J₆ · SD · S_j⁻¹    =  L_jᵀ · J₆            (source index)
```

Both involve a single interface, and `SD`, `J₆`, `c` are medium-independent.
Eliminating `S_i` between them gives a closed matrix equation for `L_i`.

> **Watch for this.** If (A) and (B) force `L` to be medium-independent, that
> contradicts the measured homogeneous `K`, which contains `η_S`. Should that
> happen, the conclusion is that the per-interface `L Φ(G) S⁻¹` ansatz is wrong —
> not that the measurement is. Record it as a negative result and rely on Task 2.

- [ ] **Step 1: Write the failing symbolic check**

```mathematica
(* Mathematica/StratifiedK.wl *)
J6 = ArrayFlatten[{{0 IdentityMatrix[3], IdentityMatrix[3]},
                   {-IdentityMatrix[3], 0 IdentityMatrix[3]}}];
SD = DiagonalMatrix[{1, -1, -1, -I w, I w, I w}];
c  = I/w;
(* unknown row correction: 1 (+) a general symmetric 2x2 on the shear block *)
L  = ArrayFlatten[{{IdentityMatrix[3], 0 IdentityMatrix[3]},
                   {0 IdentityMatrix[3], Kmat}}];
condA = c L . SD . J6 - J6 . Inverse[Transpose[Smat]];
condB = J6 . SD . Inverse[Smat] - Transpose[L] . J6;
sol = Solve[Thread[Flatten[condA] == 0] && Thread[Flatten[condB] == 0], ...];
```

- [ ] **Step 2: Run it**

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/StratifiedK.wl
```

Use plain `Simplify`, never `FullSimplify` (it hangs on this class of expression).
Cross-check every symbolic result numerically at 20 digits before believing it.

- [ ] **Step 3: Emit the closed form as JSON and compare in Python**

Export with `ToString[...]` and parse with `float()` — `wolframclient` cannot
deserialise raw Mathematica numbers and will raise `TypeError`. Chop spurious
imaginary parts with `Chop[Re[N[expr, 20]]]` where the result should be real.

- [ ] **Step 4: Gate — derivation must reproduce the homogeneous K**

```bash
conda run -n seismic python scripts/gate_stratified_k.py --check-derivation
```

Expected: the derived `K` evaluated with a single medium equals
`1 ⊕ (−P∥ + η_S P⊥)` to < 1e-12. If it does not, the derivation is wrong, not the
measurement — the homogeneous `K` is pinned by an absolute comparison at 1.8e-14.

- [ ] **Step 5: Commit**

```bash
git add Mathematica/StratifiedK.wl scripts/gate_stratified_k.py
git commit -m "🔬 derive: stratified K from the two reciprocity laws"
```

---

## Task 3a: Guard the placement restriction (replaces Tasks 1–3)

`K` is exact wherever the medium is continuous and undefined on a material
interface. That restriction must be enforced in code, not remembered.

**Files:**
- Modify: `scripts/gate_stratified_correction.py` (add the guard's test)
- Later folded into `cubic_scattering/layered_correction.py` in Task 4.

- [ ] **Step 1: Write the failing test**

```python
def test_rejects_a_plane_on_a_discontinuity():
    """A source or receiver plane at a material interface must raise."""
    mod = sandwich_model()          # interface 3 is A|B, interface 1 is A|A
    with pytest.raises(ValueError, match="material discontinuity"):
        corrected_layered_6x6(mod, w=75.4, kx=5.4, ky=7.2,
                              source_iface=3, receiver_iface=1)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_layered_correction.py -k discontinuity -v
```

Expected: `Failed: DID NOT RAISE`.

- [ ] **Step 3: Implement the check**

An interface `k` is a discontinuity when layer `k` and layer `k+1` differ in any
of `(rho, alpha, beta)`. Compare the model's own arrays; do not infer from
`_interface_elastic_properties`, which silently picks one side.

```python
def _assert_continuous(model, iface, role):
    a, b = max(iface, 1), max(iface, 1) + 1
    if b >= model.n_layers:
        return                       # half-space below: nothing to disagree with
    same = (model.rho[a] == model.rho[b] and model.alpha[a] == model.alpha[b]
            and model.beta[a] == model.beta[b])
    if not same:
        msg = (
            f"{role} interface {iface} lies on a material discontinuity "
            f"(layer {a}: a={model.alpha[a]}, b={model.beta[a]}, r={model.rho[a]}; "
            f"layer {b}: a={model.alpha[b]}, b={model.beta[b]}, r={model.rho[b]}).\n"
            "  Why: the correction operator K is built from the local vertical S\n"
            "  slowness eta_S, which is two-valued at a material interface, so K\n"
            "  is not defined there.  Measured: no choice of side reaches better\n"
            "  than 3.9e-3, against 1.7e-15 for an interior plane.\n"
            "  Fix: place the plane inside a layer.  Subdivide the layer if a\n"
            "  scatterer really must sit at that depth — subdivision is\n"
            "  transparent (verified to 1.2e-15 in the marine reference model)."
        )
        raise ValueError(msg) from None
```

- [ ] **Step 4: Run until green, and confirm the interior gate still passes**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_layered_correction.py -v
conda run -n seismic python scripts/gate_stratified_correction.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/gate_stratified_correction.py
git commit -m "✅ test: K is exact in stratified media; guard the placement restriction"
```

---

## Task 4: Promote the correction to a tested module

**Files:**
- Create: `cubic_scattering/layered_correction.py`
- Create: `cubic_scattering/tests/test_layered_correction.py`
- Modify: `scripts/gate_wrapper_resolution.py` (import rather than redefine)

**Interfaces:**
- Produces:
  - `k_operator(kx, ky, omega, s_s) -> (3,3)` — the `1 ⊕ (−P∥ + η_S P⊥)` operator.
  - `corrected_layered_6x6(model, omega, kx, ky, source_iface, receiver_iface) -> (6,6)`
    — D1 and D2 applied, in basis `(u_z,u_x,u_y,T_zz,T_xz,T_yz)`.
  - `source_operator_9x6(kx, ky, rho, alpha, beta) -> (6,9)` — the derived **B**
    including the D3 sign, mapping `(F_i, Δσ*_Vα)` to a state jump.

- [ ] **Step 1: Write the failing regression tests**

```python
# cubic_scattering/tests/test_layered_correction.py
def test_k_operator_is_projector_form():
    """K = 1 (+) (-P_par + eta_S P_perp), off-diagonal when kx ky != 0."""
    w, p, c, s = 2 * np.pi * 48.0, 0.12, 0.6, 0.8
    k = k_operator(w * p * c, w * p * s, w, S_S)
    assert abs(k[0, 0] - 1.0) < 1e-14
    assert abs(k[1, 2] - k[2, 1]) < 1e-14
    off = np.linalg.norm(k - np.diag(np.diag(k))) / np.linalg.norm(k)
    assert off > 0.6, "K must not be diagonal off-axis"


def test_source_operator_matches_box53_explosion():
    """B on an isotropic moment must reproduce thesis Box 5.3."""
    kx, ky = 0.9, 0.7
    src = np.zeros(9)
    src[3] = src[4] = src[5] = 1.0          # M_zz = M_xx = M_yy = 1
    got = source_operator_9x6(kx, ky, RHO, ALPHA, BETA) @ src
    want = np.array([1 / (RHO * ALPHA**2), 0, 0, 0,
                     2j * kx * BETA**2 / ALPHA**2, 2j * ky * BETA**2 / ALPHA**2])
    assert np.allclose(got, want, atol=1e-14)


def test_gate_f_passes():
    """The whole point: W M symmetric to machine precision."""
    assert gate_f_residual() < 1e-10
```

- [ ] **Step 2: Run to verify they fail**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_layered_correction.py -v
```

Expected: `ModuleNotFoundError: No module named 'cubic_scattering.layered_correction'`.

- [ ] **Step 3: Write the module**

Lift `k3`, `corrected_6x6` and the `B` construction out of
`scripts/gate_wrapper_resolution.py`, generalise `k3` to take the source layer's
complex S slowness as an argument (rather than closing over a module-level
constant), and fold in whatever Task 2/3 established about which medium supplies it.

Fail fast on the geometry that has no answer:

```python
if np.hypot(kx, ky) == 0.0:
    msg = (
        "k_operator is undefined at kx = ky = 0: the projectors P_par and "
        "P_perp require a horizontal propagation direction.\n"
        "  Where: the caller's wavenumber grid must exclude the origin.\n"
        "  Expected: hypot(kx, ky) > 0, e.g. kx=1e-6 for a near-normal ray.\n"
        "  Fix: offset the origin sample, as slab_scattering does at p=1e-6."
    )
    raise ValueError(msg) from None
```

- [ ] **Step 4: Run until green, then the full suite**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_layered_correction.py -v
conda run -n seismic python -m pytest cubic_scattering/tests/ -q
```

The full suite is slow (it exceeded 900 s once). Run it in the background and check
the summary; do not skip it — this module is imported by the slab path.

- [ ] **Step 5: Rewire the gate script and confirm it still passes**

```bash
conda run -n seismic python scripts/gate_wrapper_resolution.py
```

Expected: GATE F still ~1e-15, GATE D still ~5e-16, exit 0.

- [ ] **Step 6: Lint and commit**

```bash
conda run -n seismic ruff check cubic_scattering/layered_correction.py --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/layered_correction.py --ignore-missing-imports
git add cubic_scattering/layered_correction.py cubic_scattering/tests/test_layered_correction.py scripts/gate_wrapper_resolution.py
git commit -m "✨ layered_correction: the wrapper correction as a tested module"
```

---

## Task 5: Upstream or carry — decide by experiment

The recommendation on file is to carry the correction locally, on the reasoning that
the external solver's reflectivity outputs are right *because* these errors cancel in
the ratios forming R and T. That is a hypothesis, and it is cheap to test.

**Files:**
- Create: `scripts/probe_upstream_fix.py` (read-only probe; makes no lasting edit)

- [ ] **Step 1: Write the probe**

Monkeypatch `assemble_greens_6x6` in-process so the stress-row × stress-column block
receives its second `(−iω)`, then run the external suite against the patched module.
Do not edit the external repository.

```python
import GlobalMatrix.layered_greens as lg

_orig = lg.assemble_greens_6x6

def patched(G_psv, G_sh, cos_phi, sin_phi, omega):
    out = _orig(G_psv, G_sh, cos_phi, sin_phi, omega)
    out[..., 3:, 3:] *= (-1j * omega)
    return out

lg.assemble_greens_6x6 = patched
```

- [ ] **Step 2: Run the external suite, patched and unpatched**

```bash
cd /Users/tod/Desktop/SeismicInversion/GlobalMatrix
conda run -n seismic python -m pytest test_layered_greens.py test_gmm.py test_riccati.py test_riccati_source.py -q
```

Record both results. The question this answers: **do the existing upstream tests
constrain the block at all?**

- [ ] **Step 3: Record the verdict**

- If the patched suite **passes**, the block is unconstrained upstream and D1 is a
  latent defect that happens not to be exercised. Recommend a patch upstream, and
  say so in the documents — but still do not apply it in this repository's work.
- If the patched suite **fails**, the cancellation hypothesis is confirmed, carrying
  the correction locally is right, and the failing test names are the evidence.

Either way the outcome is a paragraph, not a code change here.

- [ ] **Step 4: Commit the probe and the verdict**

```bash
git add scripts/probe_upstream_fix.py
git commit -m "🔬 evidence: does the upstream suite constrain the stress-stress block?"
```

---

## Task 6: Re-run the gates that were measured through the broken wrapper

**Files:**
- Modify: `scripts/composed_matvec.py`, `scripts/discrete_pz_kernel.py`
- Modify: `scripts/gate_9x9_source_convention.py` (GATE E leg only)

- [ ] **Step 1: GATE E on the corrected 6×6**

Point `corrected_9x9` at `cubic_scattering.layered_correction.corrected_layered_6x6`
and the derived **B**, then run:

```bash
conda run -n seismic python scripts/gate_9x9_source_convention.py
```

Expected: GATE E still passes (it did at 9.7e-16 with the shipped operators, and is
blind to the global sign that changed). If it now fails, that is informative and
blocks Task 7 — the swap relation and the symmetry must hold together.

- [ ] **Step 2: N5 — re-take the stratified-vs-lateral kernel ratio**

Swap the uncorrected 6×6 for the corrected one in both N5 scripts:

```bash
conda run -n seismic python scripts/composed_matvec.py
conda run -n seismic python scripts/discrete_pz_kernel.py
```

The recorded values to beat: ratio varying 10–14× across wavenumber, a roughly
constant factor near 12 with ~200% scatter, growing under damping. Three outcomes:

- ratio → 1 within quadrature error: N5 is closed, and the composition is unblocked;
- ratio → a constant ≠ 1: a single remaining normalisation; find it, it is now a
  one-parameter problem;
- scatter persists: the lateral kernel is also implicated, and `P^x` needs its own
  absolute calibration — a new plan, not a step here.

- [ ] **Step 3: Commit**

```bash
git add scripts/composed_matvec.py scripts/discrete_pz_kernel.py scripts/gate_9x9_source_convention.py
git commit -m "🔬 evidence: GATE E and N5 re-taken through the corrected wrapper"
```

---

## Task 7: Documentation pass

Updating the documents is part of this work, not a follow-up. The `.tex` files lag
the code by default in this repository, and that is the standing failure mode.

**Files:**
- Modify: `docs/wrapper_problem_state_2026-09-13.md` (§6)
- Modify: `LatexPDFs/WrapperProblem/WrapperProblem.tex` (§9)

- [ ] **Step 1: Update the markdown**

Replace §6.1 with the measured stratified `K`, §6.2 item 1 with Task 5's verdict,
and item 3 with the N5 outcome. Keep the status labels honest: measured, derived, or
hypothesis.

- [ ] **Step 2: Update the LaTeX and recompile in place**

```bash
cd LatexPDFs/WrapperProblem
/usr/local/bin/lualatex -interaction=nonstopmode WrapperProblem.tex
/usr/local/bin/lualatex -interaction=nonstopmode WrapperProblem.tex
```

Twice, for cross-references. `pdflatex` will fail — the document uses `fontspec`.
Check the log for `^!` and undefined references before believing it built.

- [ ] **Step 3: Verify no assistant-tooling names leak into tracked files**

Run the two-command check from the engineering standards (the one that greps tracked
content and the one that greps commit messages). Both must return nothing but
legitimate third-party package names, and only after reading each one.

- [ ] **Step 4: Commit**

```bash
git add docs/ LatexPDFs/WrapperProblem/
git commit -m "📝 docs: the stratified wrapper correction"
```

---

## Carried forward: where the time actually goes (measured 13 Sep 2026)

Not part of this plan. Recorded here because the lattice-sum work that follows
rebuilds exactly the code path this concerns, and doing the vectorisation *then*
costs almost nothing extra while doing it as a separate pass means touching and
re-validating the same code twice.

**Profile of an end-to-end Foldy–Lax solve** (`compute_slab_scattering`, M = 32,
N_z = 4, periodic, uniform contrast; `cProfile`, cumulative):

| | cumtime | share |
|---|---|---|
| whole solve | 0.623 s | 100% |
| `_build_slab_kernels` | 0.611 s | **98%** |
| ├ `_propagator_block_9x9` × 3695 | 0.422 s | 68% |
| ├ `elastodynamic_greens_deriv` × 3695 | 0.305 s | 49% |
| └ `_voigt_contract` × 3695 | 0.108 s | 17% |
| `_slab_matvec` (the GMRES loop) | 0.006 s | **1%** |
| FFTs | ~0.006 s | 1% |

GMRES converged in **2 matvecs**. The iterative solve is one percent of runtime.

**Consequence: do not optimise the matvec.** Reshaping the batched 9×9 complex
apply into one `zgemm`, or threading it, attacks 1% of the solve — Amdahl caps
the return at ~1.01×. A microbenchmark of the matvec shape *looks* like the hot
loop (8.7 GFLOP/s against 293 for `zgemm`, and it does not thread at all), but
the profile shows those `einsum` calls live inside the kernel build, not the
solve. Right operation, wrong place.

**The real target is `_propagator_block_9x9` / `elastodynamic_greens_deriv`.**
3695 calls at ~115 µs each, almost all Python and NumPy per-call overhead on tiny
arrays — `_radial_functions`, `np.outer`, 33 293 calls to `np.zeros`, `einsum`
over 3×3×3 tensors. The fix is to **batch the whole lattice into one pass** over
arrays of shape `(n_pairs, 3, 3, 3)` instead of looping, which removes the
overhead rather than adding cores. Plausibly 10–50×.

Two things make this a good fit for the lattice-sum task specifically:

- It is pure whole-space Green's-tensor evaluation — no iteration, no solver,
  embarrassingly data-parallel.
- Batching elementwise evaluation does **not** reassociate sums the way a
  `zgemm` reshape would, so the 1e-15 gates are not at risk. Re-run
  `gate_wrapper_resolution.py` and `gate_stratified_correction.py` afterwards
  regardless; both are cheap and both assert at 1e-10.

**Hardware is not the lever.** This machine (M1 Pro, 8 P-cores + 2 E) reaches
352 GFLOP/s float64 `dgemm` at 8 threads and 49 at one — a 7.1× scaling, so
OpenBLAS and the silicon are both healthy. But `_build_slab_kernels` takes 524 ms
at M = 32 whether OMP_NUM_THREADS is 1 or 8: the hot path is single-core Python
overhead on small arrays. A newer Mac would give perhaps 1.5–2× and leave the
extra cores idle exactly as they are now.

Reproduce: `scripts/` has no profiling harness; the run above used `cProfile`
around `compute_slab_scattering` with `uniform_slab_material(geom, ref,
MaterialContrast(2.0, 1.0, 0.1))`.

---

## Self-review notes

- **Spec coverage.** §6.1 (stratified extension) → Tasks 1–4. §6.2 item 1
  (upstream vs carry) → Task 5. §6.2 item 2 (GATE E) → Task 6 Step 1. §6.2 item 3
  (N5) → Task 6 Step 2. §6 item 4 of the earlier draft (homogeneous-limit caveat) is
  subsumed by Task 1, which removes the caveat by building a stratified reference.
- **Known risk, deliberately left in.** Task 3 may prove the per-interface ansatz
  impossible. That is why Task 2 (measurement) precedes it and does not depend on it.
- **Most likely quick win.** Task 2 Step 4 outcome 1: `_interface_elastic_properties`
  returns layer `max(iface,1)` — the layer *below* the interface. At an interface
  between two different layers the correction may need the layer above. If so the
  recorded stratified failure is an off-by-one, not a physics gap. Check this first;
  it could collapse Tasks 2 and 3 into an afternoon.
