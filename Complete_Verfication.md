# Complete Verification: P/SV/SH In x P/SV/SH Out

Full angular-pattern comparison of Mie theory vs Foldy-Lax voxelized sphere
for all non-zero scattering channels.

## Current State

**Working (P-incidence only)**:
- `compute_elastic_mie` solves 4x4 P-SV BVP with P-incident RHS -> `a_n` (P->P), `b_n` (P->SV)
- Also solves 2x2 SH BVP -> `c_n` (SH->SH)
- `mie_far_field` extracts f_PP and f_PS from large-r evaluation
- `foldy_lax_far_field` returns u_P and u_S for arbitrary (k_hat, pol, wave_type)
- Mathematica: `mieFarFieldP`, `cubeFarFieldP`, `incidentState` (P-wave only)
- Python tests: 17 tests, all P-incidence, magnitude-only comparison

**Missing**:
- SV-incident Mie solve (code exists in `_mie_incident_psv(..., "S")` but never called)
- SV/SH far-field angular functions (m=1 partial waves use P_n^1, not P_n)
- SV/SH decomposition of Foldy-Lax u_S into theta-hat and phi-hat components
- Mathematica S-wave far-field functions and incident states
- Phase-sensitive test comparisons (Re/Im, not just |f|)

## Scattering Matrix Structure

For an isotropic sphere, P-SV and SH decouple at each partial wave order n.
The 3x3 scattering matrix has 5 non-zero channels:

```
             P-inc     SV-inc     SH-inc
  P-out:   f_PP(th)   f_SP(th)     0
  SV-out:  f_PS(th)   f_SS(th)     0
  SH-out:    0           0       f_SH(th)
```

Zeros arise because SH decouples from P-SV for a sphere.

**Key subtlety**: P-incidence is axially symmetric (m=0), but SV and SH
plane-wave incidence have azimuthal m=1 dependence. The radial scattering
T-matrix is independent of m, but the far-field angular functions differ:

| Channel  | m | Angular function         | Azimuthal factor |
|----------|---|--------------------------|------------------|
| P->P     | 0 | P_n(cos th)              | 1                |
| P->SV    | 0 | dP_n/dth                 | 1                |
| SV->P    | 1 | P_n^1(cos th) = dP_n/dth | cos(phi)         |
| SV->SV   | 1 | dP_n^1/dth               | cos(phi)         |
| SH->SH   | 1 | P_n^1/sin(th)            | sin(phi)         |

Identity: P_n^1(cos th) = dP_n/dth (Condon-Shortley convention).

For comparison, evaluate SV channels in the xz-plane (phi=0, cos(phi)=1)
and SH channel at phi=pi/2 (sin(phi)=1).

---

## Phase 1: Python -- Extend Mie for SV Incidence

### 1a. Add SV-incident solve to `compute_elastic_mie`

In the n>=1 loop (line 645), add a SECOND solve of the same 4x4 matrix
with `_mie_incident_psv(n, ..., incident_type="S")`:

```python
# Existing P-incident solve
rhs_psv_P = _mie_incident_psv(n, omega, radius, ref, incident_type="P")
sol_P = np.linalg.solve(M_psv, rhs_psv_P)
a_n[n] = sign * sol_P[0]   # P->P
b_n[n] = sign * sol_P[1]   # P->SV

# NEW: SV-incident solve (same matrix, different RHS)
rhs_psv_S = _mie_incident_psv(n, omega, radius, ref, incident_type="S")
sol_S = np.linalg.solve(M_psv, rhs_psv_S)
a_n_sv[n] = sign * sol_S[0]   # SV->P
b_n_sv[n] = sign * sol_S[1]   # SV->SV
```

Files: `cubic_scattering/sphere_scattering.py`

### 1b. Update `MieResult` dataclass

Add two new fields:

```python
a_n_sv: NDArray[np.complexfloating]  # SV->P coefficients
b_n_sv: NDArray[np.complexfloating]  # SV->SV coefficients
```

File: `cubic_scattering/sphere_scattering.py` (line 79)

---

## Phase 2: Python -- Mie Far-Field for All Channels

### 2a. Angular helper functions

Add to `sphere_scattering.py`:

```python
def _Pn1(n, theta):
    """Associated Legendre P_n^1(cos theta) = dP_n/dtheta."""
    # = _dPn_dtheta(n, theta) -- already exists!

def _dPn1_dtheta(n, theta):
    """d/dtheta of P_n^1(cos theta) = d^2 P_n / dtheta^2.
    From Legendre ODE: d^2P_n/dth^2 = -cos(th)/sin(th) dP_n/dth - n(n+1) P_n
    """

def _Pn1_over_sintheta(n, theta):
    """P_n^1(cos theta) / sin(theta).
    Limit as theta->0: n(n+1)/2.
    """
```

### 2b. Generalize `mie_far_field`

Rename existing to `mie_far_field` with `incident_type` parameter:

```python
def mie_far_field(
    mie_result: MieResult,
    theta_arr: NDArray,
    incident_type: str = "P",       # "P", "SV", or "SH"
) -> tuple[NDArray, NDArray, NDArray]:
    """Returns (f_P, f_SV, f_SH) far-field amplitudes."""
```

**P-incidence** (m=0, existing logic):
- f_PP(th) = Sum a_n (-i)^n P_n(cos th)
- f_PS(th) = Sum b_n (-i)^n dP_n/dth
- f_PSH = 0

**SV-incidence** (m=1, evaluate at phi=0):
- f_SP(th) = Sum a_n_sv (-i)^n P_n^1(cos th)
- f_SS(th) = Sum b_n_sv (-i)^n dP_n^1/dth
- f_SSH = 0 (at phi=0; non-zero at phi != 0 from SH-channel coupling)

**SH-incidence** (m=1, evaluate at phi=pi/2):
- f_SHP = 0
- f_SHSV = 0
- f_SH(th) = Sum c_n (-i)^n P_n^1(cos th)/sin(th)

### 2c. Normalisation

**Critical**: The m=1 plane-wave expansion coefficient differs from m=0.
The bare scattering T-matrix at each n is:

```
T_n^PP = a_n^PP / C_P(n),    where C_P(n) = (2n+1) i^n / (i k_P)
T_n^SP = a_n^SP / C_S(n),    where C_S(n) = (2n+1) i^n / (i k_S)
```

For m=0 far-field: f = Sum_n [T_n * C_inc(n)] * (-i)^n * P_n
For m=1 far-field: f = Sum_n [T_n * C_inc_m1(n)] * (-i)^n * P_n^1

The m=1 plane-wave coefficient for x_hat exp(ikS z) is:
  C_m1(n) = i^{n+1} (2n+1) / [n(n+1) i k_S]

So the ratio C_m1/C_m0 = i / n(n+1).

**Implementation**: Either:
(a) Divide out C_m0 from the stored coefficients, multiply by C_m1
(b) Store bare T-matrix elements instead

Recommend option (a): Compute `T_n = a_n / C_P(n)` then `a_n_m1 = T_n * C_m1(n)`.
This keeps the existing code unchanged and adds the m=1 conversion in `mie_far_field`.

File: `cubic_scattering/sphere_scattering.py`

---

## Phase 3: Python -- Foldy-Lax SV/SH Decomposition

### 3a. Add theta-hat/phi-hat projection to `foldy_lax_far_field`

The function already returns `(u_P, u_S)` as 3-vectors. Add a helper to
decompose `u_S` into SV (in scattering plane) and SH (perpendicular):

```python
def decompose_SV_SH(
    u_S: NDArray,          # (M, 3) S-wave displacement
    r_hat_arr: NDArray,    # (M, 3) observation directions
    k_hat: NDArray,        # (3,) incident direction
) -> tuple[NDArray, NDArray]:
    """Decompose S-wave into SV and SH components.

    SV = theta_hat . u_S (in scattering plane)
    SH = phi_hat . u_S   (perpendicular to scattering plane)

    theta_hat and phi_hat are defined by the scattering plane
    containing k_hat and r_hat.
    """
```

For observation in the xz-plane (r_hat = [cos th, sin th, 0]):
- theta_hat = [-sin th, cos th, 0]
- phi_hat = [0, 0, 1]

For observation in the yz-plane (r_hat = [cos th, 0, sin th]):
- theta_hat = [-sin th, 0, cos th]
- phi_hat = [0, -1, 0]

File: `cubic_scattering/sphere_scattering.py`

---

## Phase 4: Mathematica -- CubeAnalytic.wl

### 4a. Add `cubeFarFieldS` (S-wave Foldy-Lax far field)

Analogous to `cubeFarFieldP` but for the S-wave Green's tensor:

```mathematica
cubeFarFieldS[theta_, kS_, beta_, rho_,
              centres_, sources_, voigtPairs_, voigtWeight_] :=
  Module[{rhat, QS, ...},
    (* S-wave: Q_S = F - ikS sigma.rhat, projected perpendicular to rhat *)
    (* Returns {f_SV, f_SH} *)
  ]
```

The S-wave far-field formula:
  Q_S = F - i k_S (sigma . r_hat)
  Q_S_perp = Q_S - (r_hat . Q_S) r_hat
  u_S = -G_S^far * Q_S_perp

Decompose Q_S_perp into theta-hat (SV) and phi-hat (SH) components.

File: `Mathematica/CubeAnalytic.wl`

---

## Phase 5: Mathematica -- Notebook Updates

### 5a. Generalize `incidentState`

Replace the hardcoded P-wave incident state with a parameterized function:

```mathematica
incidentState[pos_, "P"] :=   (* u = z_hat exp(ikP z), eps_zz = ikP exp(ikP z) *)
incidentState[pos_, "SV"] :=  (* u = x_hat exp(ikS z), eps_zx = ikS/2 exp(ikS z) *)
incidentState[pos_, "SH"] :=  (* u = y_hat exp(ikS z), eps_zy = ikS/2 exp(ikS z) *)
```

9-vector: {u_z, u_x, u_y, eps_zz, eps_xx, eps_yy, 2*eps_xy, 2*eps_zy, 2*eps_zx}

P-wave:  {exp(ikPz), 0, 0, ikP*exp(ikPz), 0, 0, 0, 0, 0}
SV-wave: {0, exp(ikSz), 0, 0, 0, 0, 0, 0, ikS*exp(ikSz)}
SH-wave: {0, 0, exp(ikSz), 0, 0, 0, 0, ikS*exp(ikSz), 0}

### 5b. Three Lax-Foldy solves

Solve (I - P.T) W = W_inc for each incident type. Reuse the SAME
system matrix (it doesn't depend on the incident wave), just change
the RHS:

```mathematica
rhsP = buildRHS["P"];   psiExcP = LinearSolve[sysMatrix, rhsP];
rhsSV = buildRHS["SV"]; psiExcSV = LinearSolve[sysMatrix, rhsSV];
rhsSH = buildRHS["SH"]; psiExcSH = LinearSolve[sysMatrix, rhsSH];
```

### 5c. Add Mie SV-incident coefficients

In `computeMieCoefficients`, add the SV-incident 4x4 solve:

```mathematica
(* Existing P-incident *)
solPSV = LinearSolve[Mpsv, rhsP];
anPP[[n+1]] = sign * solPSV[[1]];
bnPS[[n+1]] = sign * solPSV[[2]];

(* NEW: SV-incident *)
solSV = LinearSolve[Mpsv, rhsSV];
anSP[[n+1]] = sign * solSV[[1]];
bnSS[[n+1]] = sign * solSV[[2]];
```

### 5d. Add Mie far-field functions

```mathematica
mieFarFieldPP[theta_, mie_] := Sum[an[[n+1]] (-I)^n LegendreP[n, Cos[theta]], {n,0,nMax}]
mieFarFieldPS[theta_, mie_] := Sum[bn[[n+1]] (-I)^n D[LegendreP[n,x],x]/.x->Cos[theta] * (-Sin[theta]), {n,1,nMax}]
mieFarFieldSP[theta_, mie_] := Sum[anSV[[n+1]] * renorm[n] * (-I)^n * Pn1[theta], {n,1,nMax}]
mieFarFieldSS[theta_, mie_] := Sum[bnSV[[n+1]] * renorm[n] * (-I)^n * dPn1dth[theta], {n,1,nMax}]
mieFarFieldSH[theta_, mie_] := Sum[cn[[n+1]] * renorm[n] * (-I)^n * Pn1[theta]/Sin[theta], {n,1,nMax}]
```

where `renorm[n]` accounts for the m=0 -> m=1 expansion coefficient ratio.

### 5e. Comparison plots

For each of the 5 non-zero channels, plot Re[f] and Im[f]:
- Row 1: P->P, P->SV (P incidence, existing + new)
- Row 2: SV->P, SV->SV (SV incidence, new)
- Row 3: SH->SH (SH incidence, new)

Each plot: Mie (blue solid) vs Lax-Foldy (red dashed), with Re and Im panels.

File: `Mathematica/LaxFoldy_VoxelSphere_vs_Mie.nb`

---

## Phase 6: Python Tests

### 6a. New test file or extend existing

Add `test_complete_scattering_matrix` to `test_sphere_scattering.py`.

### 6b. Test cases

**Parameters** (use standard validated set):
- Background: alpha=5000, beta=3000, rho=2500
- Moderate contrast: Dlambda=+2e9, Dmu=+1e9, Drho=+100
- ka values: 0.1 (Rayleigh), 0.5 (transition), 1.5 (resonance)

**For each ka, test all 5 channels**:

```python
def test_PP_channel(ka):
    """P-incidence -> P-scattered: Re/Im comparison."""

def test_PS_channel(ka):
    """P-incidence -> SV-scattered: Re/Im comparison."""

def test_SP_channel(ka):
    """SV-incidence -> P-scattered: Re/Im comparison."""

def test_SS_channel(ka):
    """SV-incidence -> SV-scattered: Re/Im comparison."""

def test_SH_channel(ka):
    """SH-incidence -> SH-scattered: Re/Im comparison."""
```

Each test:
1. Compute Mie far-field amplitude at ~20 angles
2. Compute Foldy-Lax far-field amplitude at same angles
3. Compare Re and Im separately (not just magnitude!)
4. Tolerance: ~10% for ka=0.1, ~20% for ka=0.5 (finite voxelization error)

### 6c. Reciprocity test

For a sphere (isotropic), reciprocity gives:
  k_P^2 f_PS(theta) = k_S^2 f_SP(theta)

(up to normalisation conventions). Verify this holds for Mie coefficients.

### 6d. Optical theorem tests

For each incident type, the forward scattering amplitude is related to the
total cross-section:
  sigma_ext = (4 pi / k_inc) Im[f(theta=0)]

Verify for P, SV, and SH incidence.

---

## Implementation Order

1. **Phase 1** (Python Mie SV-incident): Small change, no new math
2. **Phase 2** (Python far-field angular functions): Core math, careful with normalisation
3. **Phase 3** (Python Foldy-Lax decomposition): Straightforward geometry
4. **Phase 6** (Python tests): Validate Phases 1-3
5. **Phase 4** (Mathematica CubeAnalytic.wl): Port Python far-field to Mathematica
6. **Phase 5** (Mathematica notebook): Visual comparison

Start with Phase 1+3 (easy) then Phase 2 (hard math), test in Phase 6,
then port to Mathematica in Phases 4+5.

---

## Risk: m=1 Normalisation

The trickiest part is the conversion between m=0 stored coefficients
(from `compute_elastic_mie`) and the m=1 far-field evaluation. The
expansion of x_hat exp(ikS z) into vector spherical harmonics gives
coefficients that differ from the m=0 expansion by a factor of
i / [n(n+1)]. Getting this normalisation wrong will produce:
- Correct SHAPE of angular pattern (since the angular functions are right)
- Wrong MAGNITUDE (by an n-dependent factor)

**Mitigation**: Test at ka << 1 (Rayleigh limit) where only n=0,1,2
contribute, making the normalisation easy to verify analytically.
Also verify reciprocity f_PS vs f_SP as an independent check.
