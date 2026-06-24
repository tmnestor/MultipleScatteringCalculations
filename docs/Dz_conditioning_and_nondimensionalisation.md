# Conditioning of the Energy-Normalised Eigen-Matrix `D_z`, and Non-Dimensionalisation

**Context.** The thesis §3.1 (`GRepresentations.tex`) builds the energy-normalised
displacement–traction eigen-matrix `D_z(k_x,k_y)` and the canonical symplectic `J₆`
(`Mathematica/ThesisInterfaceRT.wl`). Instantiated in **SI units** the matrix looks alarmingly
ill-conditioned (`cond(D_z) ~ 10¹⁰`). This note shows that is a **pure units artifact**, not a
defect of the representation, explains the non-dimensionalisation that removes it, and addresses
the natural worry that conditioning grows with angular frequency `ω`.

---

## 1. The key fact: `cond(D_z) = ρ ω v`, exactly

The state vector `b = (u, t)` stacks a **displacement** `u` (dimension `[L]`) on a **traction**
`t` (dimension `[M L⁻¹ T⁻²]` = stress). These are different *kinds* of quantity, so their
numerical magnitudes depend on the unit of length and the unit of stress *separately*. From the
eigenvectors (`Peigen`/`SVeigen`/`SHeigen`) with `ε = 1/√(2ρω²k_z)`:

```
|u| ~ ε·k   = √( k / (ρω²) )
|t| ~ ε·ρω² = √( ρω² / k )
|t| / |u|   = ρω² / k = ρω² / (ω/v) = ρ ω v
```

The matrix therefore has three rows of magnitude `|u|` and three of magnitude `|t|`, so its
singular values span a factor `|t|/|u|`, giving

> **`cond(D_z) ≈ ρ ω v`** — a *dimensional* number whose value is set entirely by the unit system.

Verified numerically (`SingularValueList`, `α=5 km/s` scenario):

| unit system | length | mass | `ρωv` (predicted) | `cond(D_z)` (measured) | `‖D·D⁻¹−I‖` (machine) |
|---|---|---|---|---|---|
| SI (m/s, kg/m³, Pa) | m | kg | 1.88e10 | 1.86e10 | 9.1e-7 |
| seismic (km/s, g/cm³, GPa) | km | g | 1.88e4 | 1.86e4 | 5.7e-13 |
| nondimensional | β/ω | ρ | 1.67 | 1.66 | 5.6e-16 |

The prediction matches the measured condition number to ~1%. It is not a property of the operator
— it is the displacement/traction magnitude ratio wearing a unit label.

---

## 2. Non-dimensionalisation

### 2.1 The principle

Every physical quantity is built from three fundamentals — **mass `[M]`, length `[L]`, time
`[T]`**. You are free to choose **three characteristic scales**, one per fundamental, and then
measure everything as a pure-number multiple of those scales. The equations become unit-free; with
well-chosen scales every number is `O(1)`.

### 2.2 The natural scales for elastodynamics

| fundamental | scale | natural choice |
|---|---|---|
| velocity (couples `L`↔`T`) | `c₀` | a reference wave speed, `β` |
| frequency (sets `T`) | `ω₀` | the angular frequency `ω` |
| density (sets `M`) | `ρ₀` | a reference density `ρ` |

Everything derived follows:

```
time scale      t₀ = 1/ω
length scale    L₀ = c₀/ω₀ = β/ω        (a reference "wavelength ÷ 2π")
stress scale    σ₀ = ρ₀ c₀² = ρ β²       (a reference modulus, GPa-scale)
```

Dimensionless variables (tildes):

```
x̃ = x/L₀ = x ω/β        ũ = u/L₀ = u ω/β        t̃ = t/(ρβ²)
α̃ = α/β,  β̃ = 1         ω̃ = ω t₀ = 1            k̃ = k L₀ = β/v_phase
```

### 2.3 Why this fixes `D_z`

Measuring the displacement block in `L₀ = β/ω` and the traction block in `σ₀ = ρβ²` makes the two
halves commensurate. The balancing diagonal is

```
S = diag( ω/β, ω/β, ω/β,  1/(ρβ²), 1/(ρβ²), 1/(ρβ²) ),     D̃_z = S · D_z
```

and then

```
cond(D̃_z) = ρ̃ ω̃ ṽ = 1 · 1 · (α/β) ≈ 5/3 ≈ 1.67.
```

The numbers that survive are precisely the **dimensionless groups that are the physics**: the
velocity ratio `α/β`, the density contrast across the interface, and the dimensionless slownesses
`k̃ = β/v_phase`. Everything else was units.

### 2.4 Seismic units are a *partial* non-dimensionalisation

```
              length    mass         time (ω)     cond(D_z) = ρωv
SI            m         kg           s             1.9e10
seismic       km        g (·cm⁻³)    s             1.9e4    ← rescales L and M, not T
nondimensional β/ω     ρ            1/ω            1.66     ← also rescales T → all O(1)
```

Switching `m→km` and `kg/m³→g/cm³` divides `ρ` and `v` by `10³` each, knocking `ρωv` down by `10⁶`
(→ `1.9e4`, already machine-precision-clean). Full non-dimensionalisation additionally sets `ω̃=1`,
removing the last factor and leaving `cond ≈ α/β`.

---

## 3. The frequency concern: does conditioning deteriorate as `ω` grows?

**In fixed dimensional units, yes — linearly.** `cond(D_z) = ρωv ∝ ω`. Holding seismic units
fixed and sweeping `ω` (at fixed horizontal slowness `p = k/ω`, i.e. fixed incidence angle):

| `ω` (rad/s) | `cond` — fixed seismic units | `cond` — frequency-adaptive nondim (`L₀=β/ω`) |
|---|---|---|
| 1.5e3 | 1.86e4 | 1.659 |
| 1.5e4 | 1.86e5 | 1.659 |
| 1.5e5 | 1.86e6 | 1.659 |
| 1.5e6 | 1.86e7 | 1.659 |
| 1.5e7 | 1.86e8 | 1.659 |

**But the growth is the same units artifact, and it vanishes under non-dimensionalisation.** With
the frequency-adaptive length scale `L₀ = β/ω`, `cond` is **flat at ≈ α/β for every frequency**.
Physically: at high `ω` the wavelength `β/ω` is tiny, so a physical displacement measured in *fixed*
km is a vanishingly small number next to a GPa stress — the two blocks drift apart in magnitude.
Measure displacement in **wavelengths** (`L₀ = β/ω`) instead of km and the imbalance never appears.

So conditioning does **not** intrinsically deteriorate with frequency. It only appears to if you
insist on a frequency-independent length unit.

### 3.1 Three ways to stay well-conditioned at any `ω`

1. **Non-dimensionalise** (best): compute in `{α/β, 1, 1, 1}` variables, then re-dimensionalise the
   outputs — displacements by `L₀ = β/ω`, stresses by `ρβ²`. `cond ≈ α/β` at all frequencies.
2. **Block-rescale**: left-multiply by `S = diag(ω/β·I₃, 1/(ρβ²)·I₃)` before any inverse/solve;
   solve; undo the scaling. Equivalent to (1).
3. **Avoid inversion entirely** — which is what the **thesis already does**. It never forms `D_z⁻¹`
   by elimination; it uses the symplectic inverse (`D1def`)

   ```
   D_z⁻¹(k) = −i J₆ D_zᵀ(−k) J₆
   ```

   — transpose, sign flips, and a re-pairing, with **no division and no conditioning** at all. The
   R/T solve and the propagator algebra inherit this, so the `ρωv` condition number never enters the
   thesis's actual computational path.

---

## 4. Why the representation is sound regardless

Two facts make the conditioning a non-issue for the thesis:

- **The symplectic identity `(J₆ D_z(−k))ᵀ D_z(k) = i J₆` (`dinv2`) holds at `~2×10⁻¹⁶` in *every*
  unit system** — SI, seismic, nondimensional. That bilinear form pairs displacement *with*
  traction (`u·t` products are `O(1)` whatever the units), so it is well-conditioned by
  construction. This is the relation the energy normalisation `ε` exists to satisfy, and it is the
  one the propagation algebra uses.
- **The thesis never inverts `D_z` naively** (§3.1, `D1def`), so the only operation that ever
  degraded — a brute-force `D_z·D_z⁻¹` in SI units — is one the thesis does not perform, in units it
  does not use.

**Bottom line.** `cond(D_z) = ρωv` is a dimensional label, not a defect; it is removed by
non-dimensionalisation (or by working in seismic units), it does not deteriorate with frequency once
the length scale tracks the wavelength, and the thesis's symplectic machinery sidesteps it entirely.

---

*Numerics reproduced by `Mathematica/ThesisInterfaceRT.wl` (seismic units; all self-checks pass at
machine precision). See also `docs/superpowers/specs/2026-06-25-thesis-interface-rt-design.md`.*
