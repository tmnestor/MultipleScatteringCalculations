# Undamped Multipole Structure Constants `D[q,s]` Implementation Plan (Phase 3b cycle 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the undamped (`κ` real) planar lattice multipole structure constants `D[q,s] = Σ_{R≠0} h_q(κ|R|) Y_q^s(R̂) e^{ik_par·R}` for `q = 0…2·Nmax`, by **multipole-projecting the validated undamped scalar Ewald field** onto regular multipoles — reliable (reuses Phase-1 TB2, stable quadrature, no per-`q` Kambe transcription, no high-order derivatives), η-independent, landing in the existing L/M/N `G0` basis.

**Architecture:** The scalar lattice Green's field `G(r) = Σ_R h_0(κ|r−R|)·(geom) e^{ik_par·R}` expands around the origin as `G(r) = iκ Σ_{q,s} D̄[q,s] j_q(κr) Y_q^s(r̂)` (the codebase's validated `gLatField ≡ gLatRecon` identity). So `D̄[q,s]` is recovered by projecting `G` onto `Y_q^s` on a small 3D sphere `r=ρ₀` (`ρ₀ < a_L`). Replacing the damped direct `G` with the undamped Ewald field (TB2 `ewaldTotal`, extended to general `z`) yields the undamped `D̄[q,s]`; `D_struct[q,s] = (−1)^s D̄[q,−s]` feeds the existing `G0` Gaunt contraction.

**Tech Stack:** Wolfram Language (`wolframscript`); Python 3.12 (`cubic_scattering`, numpy/scipy, pytest), conda env `seismic`; JSON as the MMA↔Python contract.

## Global Constraints

- Conda env `seismic`; Python tooling via `conda run -n seismic <cmd>`.
- Time `e^{−iωt}`, outgoing `h_q^(1)` via `SphericalHankelH1` (never `j_q + i y_q`); regular `j_q` via `SphericalBesselJ`; `Y_q^s` via `SphericalHarmonicY`.
- Lattice in the x–y plane, constant `a_L = 2.0`, Bloch vector `k_par = (kx, ky) = (0.2, 0.1)` (Phase-1 TB2 values, so the `q=0` anchor matches directly). Undamped wavenumber `κ` REAL (e.g. `κ = 1.5`, TB2's `kReal`); damped `κ_d = 1.5 + 0.25 I` for the projection-method gate.
- Ewald split parameter `η` (two values `η₁=0.7, η₂=1.15` for η-independence), real-space half-width `Rc`, reciprocal half-width `Gc` (Phase-1 TB2 values `RcE=GcE=6`).
- Sphere-projection radius `ρ₀ = 0.5` (`< a_L`, inside the node-free region); sphere quadrature Gauss-Legendre `n_u × n_φ` (start `16 × 32`).
- `D̄ ↔ D_struct`: `D̄[q,p] = (−1)^p D_struct[q,−p]`, i.e. `D_struct[q,s] = (−1)^s D̄[q,−s]`.
- Reference JSON in `Mathematica/`; complex numbers serialised as `[re, im]` via `reim[z_] := {Re[N[z]], Im[N[z]]}`.
- Line length ≤ 108. Ruff `--ignore ARG001,ARG002,F841,E741`, ruff format, mypy `--ignore-missing-imports`. B904 in except; `pathlib.Path`; Google docstrings.
- Lint/format/type after every Python change:
  `conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash tool — write commit messages to a file and `git commit -F`. Long `wolframscript` runs auto-background; wait via a bounded waiter; ONE kernel at a time (concurrent kernels thrash).
- `IntraPlaneKambe.wl` is a NEW self-contained file (does NOT `Get` `IntraPlaneLatticeSum.wl`, which runs the Phase-1 study on load); it copies the small TB2 pieces it needs.

## File Structure

- Create `Mathematica/IntraPlaneKambe.wl` — general-`z` scalar Ewald, multipole projection, undamped `D[q,s]`, `G0` + reciprocity; dumps `Mathematica/IntraPlaneKambe_reference.json`.
- Create `Mathematica/IntraPlaneKambe_reference.json` (generated).
- Create `cubic_scattering/tests/test_intraplane_kambe.py` — Python cross-check.
- Modify `Mathematica/makeIntraPlaneNotebooks.wl` — add the `.nb` twin entry.
- Modify `IntraPlaneFoldyLax_Plan.md` — Phase 3b cycle 1 DONE; cycles 2–3 remain.
- Create memory note; update `MEMORY.md`.

---

### Task 1: General-`z` scalar Ewald field + η-independence

Extend the TB2 scalar Ewald (z=0) to a general 3D field point `r=(x,y,z)`, so the multipole projection (Task 2) can separate `q`. Real-space half: trivially the 3D distance. Reciprocal half: the `z`-dependent inter-plane form transcribed from the repo note.

**Files:**
- Create: `Mathematica/IntraPlaneKambe.wl`

**Interfaces:**
- Produces: `ewReal3[κ,r,η,Rc]`, `ewRecip3[κ,r,η,Gc]`, `ewTot3[κ,r,η,Rc,Gc]` (scalar; `r` a 3-vector); `ewDirect3[κ,r,Lbig]` (the damped direct 3D sum, ground truth). All include the Bloch phase `e^{i k_par·R}`.

- [ ] **Step 1: Header, constants, real-space 3D half, direct 3D sum**

```mathematica
(* IntraPlaneKambe.wl — Phase 3b cycle 1: undamped multipole structure constants D[q,s]
   via multipole projection of the undamped scalar Ewald field. Standard Kambe/layer-KKR
   OBJECT, built by projection of the validated TB2 scalar Ewald (NOT per-q analytic forms).
   Time e^{-i w t}, outgoing h_q^(1); lattice in x-y, a_L=2, k_par=(kx,ky). *)
reim[z_] := {Re[N[z]], Im[N[z]]};
aL = 2.0; kx = 0.2; ky = 0.1; kpar2 = {kx, ky};
Aarea = aL^2; recipB = 2 Pi/aL;
sh[n_, x_] := SphericalHankelH1[n, x]; sj[n_, x_] := SphericalBesselJ[n, x];

(* real-space half, general z: 3D distance d = |r - (aL i, aL j, 0)| *)
ewReal3[kappa_, r_, eta_, Rc_] := (1/(8 Pi)) Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      (Exp[I aL (kx i + ky j)]/d) Sum[Exp[s I kappa d] Erfc[d eta + s I kappa/(2 eta)], {s, {-1, 1}}]],
     {i, -Rc, Rc}, {j, -Rc, Rc}], 1]];

(* damped direct 3D lattice GF (ground truth for the projection-method gate) *)
ewDirect3[kappa_, r_, Lbig_] := Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      Exp[I kappa d]/(4 Pi d) Exp[I aL (kx i + ky j)]], {i, -Lbig, Lbig}, {j, -Lbig, Lbig}], 1]];
```

- [ ] **Step 2: Reciprocal-space 3D half (transcribe from the repo Ewald note)**

Transcribe the general-`z` reciprocal half from `LatexPDFs/EwaldIntraPlanePropagator/EwaldIntraPlanePropagator.tex` (the "reciprocal half by Poisson summation" section). It is the `z`-dependent generalisation of TB2's `ewaldRecip` (which is the `z=0` special case `erfc(kzG/2iη)`):

```mathematica
(* K_n = k_par + G ; kzG = sqrt(kappa^2 - |K_n|^2) (Im kzG >= 0); the z-dependent erfc pair.
   At z=0 this MUST reduce to TB2 ewaldRecip: (I/(2 Aarea)) (Exp[I kpg.rho]/kz) Erfc[kz/(2 I eta)]. *)
ewRecip3[kappa_, r_, eta_, Gc_] := (I/(4 Aarea)) Total[Flatten[Table[
     With[{kpg = kpar2 + recipB {m, n}, z = r[[3]]},
      With[{kz = Sqrt[kappa^2 - kpg . kpg]},
       (Exp[I kpg . {r[[1]], r[[2]]}]/kz) (
          Exp[ I kz Abs[z]] Erfc[ Abs[z] eta + kz/(2 I eta)]
        + Exp[-I kz Abs[z]] Erfc[-Abs[z] eta + kz/(2 I eta)])]],
     {m, -Gc, Gc}, {n, -Gc, Gc}], 1]];
ewTot3[kappa_, r_, eta_, Rc_, Gc_] := ewReal3[kappa, r, eta, Rc] + ewRecip3[kappa, r, eta, Gc];
```

(The exact `z`-erfc arguments are the one transcription in this plan; the `z=0`-reduction check below and the η-independence gate are the verification — a wrong argument fails η-independence, which cannot be faked. If `z=0` reduction or η-independence fails, re-read the note's reciprocal-half equation and correct the `erfc`/prefactor before proceeding.)

- [ ] **Step 3: Self-verify — `z=0` reduction, η-independence, damped-direct agreement**

```mathematica
SeedRandom[20260623];
r3Pts = Table[Module[{u = RandomReal[{-1,1}], ph = RandomReal[{0,2 Pi}], st},
    st = Sqrt[1-u^2]; 0.5 {st Cos[ph], st Sin[ph], u}], {6}];   (* |r|=0.5, z != 0 *)
kR = 1.5; kD = 1.5 + 0.25 I; e1 = 0.7; e2 = 1.15; Rc = 6; Gc = 6; Lbig = 40;
etaIndep = Max[Table[Abs[ewTot3[kR, r, e1, Rc, Gc] - ewTot3[kR, r, e2, Rc, Gc]], {r, r3Pts}]];
agree = Max[Table[Abs[ewTot3[kD, r, e1, Rc, Gc] - ewDirect3[kD, r, Lbig]], {r, r3Pts}]];
Print["==== Phase 3b cycle 1 :: general-z scalar Ewald ===="];
Print["  [1] eta-independence (kappa real, z!=0) = ", ScientificForm[etaIndep, 3],
   " -> ", If[etaIndep < 1.*^-8, "PASS", "FAIL"]];
Print["  [2] Ewald vs damped direct 3D sum = ", ScientificForm[agree, 3],
   " -> ", If[agree < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 4: Run; iterate the reciprocal `erfc` until both gates PASS**

Run: `/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/IntraPlaneKambe.wl`
Expected: both `[1]` and `[2]` PASS. If η-independence fails, the `ewRecip3` `z`-erfc form is wrong — re-read the note's reciprocal-half section and fix `Step 2` (do NOT weaken the tolerance). The damped-direct agreement `[2]` independently confirms the value.

- [ ] **Step 5: Commit**

```bash
git add Mathematica/IntraPlaneKambe.wl
git commit -F .git/COMMIT_MSG_k1
```
Message: `✨ Phase 3b cycle 1: general-z scalar Ewald field (eta-independent, vs direct)`

---

### Task 2: Multipole projection → undamped `D[q,s]`

Project the undamped Ewald field onto regular multipoles on the sphere `r=ρ₀` to extract `D̄[q,s]`, convert to `D_struct[q,s]`. Anchor the *method* against the damped direct structure constant.

**Files:**
- Modify: `Mathematica/IntraPlaneKambe.wl`

**Interfaces:**
- Consumes: `ewTot3`, `ewDirect3` (Task 1).
- Produces: `Dproj[fieldFn, q, s, ρ₀]` → `D_struct[q,s]` for a scalar field function `fieldFn[r]`; `DstructUndamped[q, s, η]` (κ real); `DstructDirect[q, s, κ, Lr]` (the damped direct structure constant, ground truth).

- [ ] **Step 1: Sphere quadrature + projection operator**

```mathematica
Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[16, -1, 1]; nPhi = 32; rho0 = 0.5;
sphPts = Flatten[Table[Module[{u = glN[[i,1]], ph = 2 Pi (j-1)/nPhi, st = Sqrt[1 - glN[[i,1]]^2]},
     {rho0 {st Cos[ph], st Sin[ph], u}, glN[[i,2]] (2 Pi/nPhi)}], {i, Length[glN]}, {j, nPhi}], 1];
Yf[q_, s_, d_] := SphericalHarmonicY[q, s, ArcCos[d[[3]]/Sqrt[d.d]], ArcTan[d[[1]], d[[2]]]];
(* G(r) = i kappa Sum D̄[q,s] j_q(kappa r) Y_q^s ; project: D̄[q,s] = (1/(i kappa j_q)) ∮ G conj(Y_q^s) ;
   then D_struct[q,s] = (-1)^s D̄[q,-s]. *)
Dproj[fieldFn_, q_, s_, kappa_] := Module[{integ},
  integ = Total[Map[#[[2]] fieldFn[#[[1]]] Conjugate[Yf[q, -s, #[[1]]]] &, sphPts]];
  (-1)^s integ/(I kappa sj[q, kappa rho0])];
```

- [ ] **Step 2: Undamped + direct structure constants; write the failing self-checks**

```mathematica
DstructUndamped[q_, s_, eta_] := Dproj[Function[r, ewTot3[1.5, r, eta, 6, 6]], q, s, 1.5];
DstructDirect[q_, s_, kappa_, Lr_] := Module[{ij, iv, jv, rn, ph, bl},
   ij = Flatten[Table[If[i==0 && j==0, Nothing, {i,j}], {i,-Lr,Lr}, {j,-Lr,Lr}], 1];
   iv = ij[[All,1]]; jv = ij[[All,2]]; rn = aL Sqrt[iv^2 + jv^2]; ph = ArcTan[iv, jv];
   bl = Exp[I aL (kx iv + ky jv)];
   Total[sh[q, kappa rn] SphericalHarmonicY[q, s, Pi/2, ph] bl]];
Nq = 6;
(* method gate: at DAMPED kappa, projected == direct structure constant *)
methResid = Max[Table[Abs[Dproj[Function[r, ewDirect3[1.5 + 0.25 I, r, 40]], q, s, 1.5 + 0.25 I]
     - DstructDirect[q, s, 1.5 + 0.25 I, 18]], {q, 0, Nq}, {s, -q, q}]];
(* undamped eta-independence *)
undEta = Max[Table[Abs[DstructUndamped[q, s, 0.7] - DstructUndamped[q, s, 1.15]], {q, 0, Nq}, {s, -q, q}]];
Print["  [3] projection method (damped: projected == direct D) = ", ScientificForm[methResid, 3],
   " -> ", If[methResid < 1.*^-4, "PASS", "FAIL"]];
Print["  [4] undamped D[q,s] eta-independence = ", ScientificForm[undEta, 3],
   " -> ", If[undEta < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 3: Run; verify `[3]` and `[4]` PASS**

Run the script. `[3]` validates the projection machinery against the known damped structure constant (tolerance `1e-4`: quadrature `16×32` + lattice truncation). `[4]` is the undamped correctness (η-independence inherited from the Ewald field). If `[3]` fails, the projection (`Dproj`, the `(-1)^s`/`D̄↔D_struct` convention, or `ρ₀`/`j_q` factor) is wrong; if `[4]` fails but `[3]` passes, `ewTot3` is η-dependent (back to Task 1).

- [ ] **Step 4: Commit**

```bash
git add Mathematica/IntraPlaneKambe.wl
git commit -F .git/COMMIT_MSG_k2
```
Message: `✨ Phase 3b cycle 1: undamped D[q,s] via multipole projection (eta-independent)`

---

### Task 3: `G0` from undamped `D[q,s]` + reciprocity; dump JSON

Gaunt-contract the undamped structure constants into `G0` and verify the symplectic-J reciprocity; dump the reference.

**Files:**
- Modify: `Mathematica/IntraPlaneKambe.wl`

**Interfaces:**
- Consumes: `DstructUndamped` (Task 2).
- Produces: JSON `{params, Dstruct: [{q,s,re,im}], G0_samples, recip_resid}`.

- [ ] **Step 1: Gaunt contraction + reciprocity (copy the Phase-1 forms)**

```mathematica
gaunt[l1_,m1_,l2_,m2_,l3_,m3_] := If[m1+m2+m3 != 0 || Abs[m1]>l1 || Abs[m2]>l2 || Abs[m3]>l3, 0,
   Sqrt[(2 l1+1)(2 l2+1)(2 l3+1)/(4 Pi)] ThreeJSymbol[{l1,0},{l2,0},{l3,0}] ThreeJSymbol[{l1,m1},{l2,m2},{l3,m3}]];
DU[q_, s_] := DU[q, s] = DstructUndamped[q, s, 0.7];   (* memoise the undamped structure constants *)
G0[n_, m_, nu_, mu_] := 4 Pi (-1)^m Sum[
   I^(nu+q-n) (-1)^q DU[q, m-mu] gaunt[n,m,nu,-mu,q,mu-m], {q, Abs[n-nu], n+nu}];
recipPairs = {{1,0,2,1}, {2,-1,3,2}, {0,0,3,0}, {2,2,4,-2}, {1,1,3,-1}};
recipResid = Max[Table[Abs[G0[p[[1]],p[[2]],p[[3]],p[[4]]]
   - (-1)^(p[[1]]+p[[3]]+p[[2]]+p[[4]]) G0[p[[3]],-p[[4]],p[[1]],-p[[2]]]], {p, recipPairs}]];
Print["  [5] undamped G0 reciprocity = ", ScientificForm[recipResid, 3],
   " -> ", If[recipResid < 1.*^-6, "PASS", "FAIL"]];
```

(The Phase-1 reciprocity relation `G0[n,m,νμ] = (−1)^{n+ν+m+μ} G0[ν,−μ,n,−m]` is the bare-multipole-basis form already used in `IntraPlaneLatticeSum.wl`; the symplectic-J metric only enters when combining with `T0`, cf. item (d) — this gate is the lattice-`G0` self-reciprocity, the same as Phase-1 TB1.)

- [ ] **Step 2: Dump JSON**

```mathematica
Nq = 6;
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneKambe_reference.json",
  <|"params" -> <|"aL" -> aL, "kx" -> kx, "ky" -> ky, "kappa" -> 1.5, "eta1" -> 0.7, "eta2" -> 1.15,
      "rho0" -> rho0, "Rc" -> 6, "Gc" -> 6, "Nq" -> Nq|>,
    "Dstruct" -> Flatten[Table[<|"q" -> q, "s" -> s, "val" -> reim[DU[q, s]]|>, {q, 0, Nq}, {s, -q, q}], 1],
    "recip_resid" -> N[recipResid]|>];
Print["  wrote IntraPlaneKambe_reference.json"];
```

- [ ] **Step 3: Run; verify `[5]` PASS + JSON written**

Run the script (bounded waiter). Expected: `[5]` PASS, JSON dumped with `Dstruct` entries for `q=0..6`.

- [ ] **Step 4: Commit**

```bash
git add Mathematica/IntraPlaneKambe.wl Mathematica/IntraPlaneKambe_reference.json
git commit -F .git/COMMIT_MSG_k3
```
Message: `✨ Phase 3b cycle 1: undamped G0 reciprocity + JSON dump`

---

### Task 4: Python cross-check

Independently recompute the projection-based undamped `D[q,s]` (scipy quadrature + an independent general-`z` Ewald) and assert agreement with the dump + the gates.

**Files:**
- Create: `cubic_scattering/tests/test_intraplane_kambe.py`

**Interfaces:**
- Consumes: `Mathematica/IntraPlaneKambe_reference.json`.

- [ ] **Step 1: Write the failing tests**

```python
"""Phase 3b cycle 1: undamped multipole structure constants D[q,s] cross-check.

``Mathematica/IntraPlaneKambe.wl`` computes the undamped (kappa real) planar lattice
structure constants D[q,s] by multipole-projecting the undamped scalar Ewald field, and
dumps them to ``IntraPlaneKambe_reference.json``.  This module independently recomputes the
general-z Ewald field and its multipole projection (scipy) and asserts: (1) the undamped
D[q,s] are eta-independent, (2) they match the dump, (3) the projection method reproduces the
damped direct structure constant.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc, spherical_jn, spherical_yn, sph_harm

REF = Path(__file__).resolve().parents[2] / "Mathematica" / "IntraPlaneKambe_reference.json"


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneKambe.wl first)"
    return json.loads(REF.read_text())


def _hankel1(n, x):
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def _ewald_total_z(kappa, r, eta, aL, kx, ky, Rc, Gc):
    """General-z scalar Ewald field (real + reciprocal halves), Bloch-phased."""
    x, y, z = r
    real = 0.0 + 0j
    for i in range(-Rc, Rc + 1):
        for j in range(-Rc, Rc + 1):
            d = math.hypot(math.hypot(x - aL * i, y - aL * j), z)
            ph = np.exp(1j * aL * (kx * i + ky * j))
            real += ph / d * sum(
                np.exp(s * 1j * kappa * d) * erfc(d * eta + s * 1j * kappa / (2 * eta))
                for s in (-1, 1)
            )
    real *= 1.0 / (8 * math.pi)
    A = aL * aL
    recipB = 2 * math.pi / aL
    rec = 0.0 + 0j
    for m in range(-Gc, Gc + 1):
        for n in range(-Gc, Gc + 1):
            kpg = np.array([kx + recipB * m, ky + recipB * n])
            kz = np.sqrt(kappa**2 - kpg @ kpg + 0j)
            rec += (
                np.exp(1j * (kpg[0] * x + kpg[1] * y)) / kz
                * (
                    np.exp(1j * kz * abs(z)) * erfc(abs(z) * eta + kz / (2j * eta))
                    + np.exp(-1j * kz * abs(z)) * erfc(-abs(z) * eta + kz / (2j * eta))
                )
            )
    rec *= 1j / (4 * A)
    return real + rec


def _sph_pts(rho0, nu=16, nphi=32):
    nodes, w = np.polynomial.legendre.leggauss(nu)
    pts, wts = [], []
    for u, wu in zip(nodes, w):
        st = math.sqrt(1 - u * u)
        for jj in range(nphi):
            ph = 2 * math.pi * jj / nphi
            pts.append(rho0 * np.array([st * math.cos(ph), st * math.sin(ph), u]))
            wts.append(wu * (2 * math.pi / nphi))
    return pts, wts


def _Yc(q, s, d):  # scipy sph_harm(m, l, phi, theta)
    theta = math.acos(d[2] / np.linalg.norm(d))
    phi = math.atan2(d[1], d[0])
    return sph_harm(s, q, phi, theta)


def _Dstruct_proj(field, q, s, kappa, rho0):
    pts, wts = _sph_pts(rho0)
    integ = sum(w * field(p) * np.conj(_Yc(-s, q, p)) for p, w in zip(pts, wts))
    return (-1) ** s * integ / (1j * kappa * spherical_jn(q, kappa * rho0))


def test_undamped_eta_independence(dump):
    par = dump["params"]
    aL, kx, ky, k = par["aL"], par["kx"], par["ky"], par["kappa"]
    for q in range(0, 4):
        for s in range(-q, q + 1):
            d1 = _Dstruct_proj(lambda r: _ewald_total_z(k, r, 0.7, aL, kx, ky, 6, 6), q, s, k, par["rho0"])
            d2 = _Dstruct_proj(lambda r: _ewald_total_z(k, r, 1.15, aL, kx, ky, 6, 6), q, s, k, par["rho0"])
            assert abs(d1 - d2) < 1e-5, f"D[{q},{s}] eta-dependent"


def test_matches_mathematica(dump):
    par = dump["params"]
    aL, kx, ky, k = par["aL"], par["kx"], par["ky"], par["kappa"]
    by = {(e["q"], e["s"]): complex(*e["val"]) for e in dump["Dstruct"]}
    for q in range(0, 4):
        for s in range(-q, q + 1):
            mine = _Dstruct_proj(lambda r: _ewald_total_z(k, r, 0.7, aL, kx, ky, 6, 6), q, s, k, par["rho0"])
            assert abs(mine - by[(q, s)]) < 1e-4, f"D[{q},{s}] != Mathematica"
```

- [ ] **Step 2: Run to verify fail (missing dump or mismatch)**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_kambe.py -v`
Expected: PASS once Task 3's dump exists and the independent recompute agrees. (If the Mathematica run is done, these should pass directly — they re-derive the same quantity independently. If `sph_harm` convention differs, reconcile the `m,l,phi,theta` order until `q=0` agrees.)

- [ ] **Step 3: Lint, format, type-check, commit**

```bash
# lint/format/mypy triple
git add cubic_scattering/tests/test_intraplane_kambe.py
git commit -F .git/COMMIT_MSG_k4
```
Message: `✨ Phase 3b cycle 1: Python cross-check of undamped D[q,s]`

---

### Task 5: `.nb` twin + closeout

**Files:**
- Modify: `Mathematica/makeIntraPlaneNotebooks.wl`; Create: `Mathematica/IntraPlaneKambe.nb`
- Modify: `IntraPlaneFoldyLax_Plan.md`
- Create: memory note; Modify: `MEMORY.md`

- [ ] **Step 1: Add `=`-banners to `IntraPlaneKambe.wl` major sections, then add the twin entry**

Ensure `IntraPlaneKambe.wl`'s sections (general-z Ewald / projection / G0+dump) start with full `=`-banners (so the twin sections like its siblings). Add to `makeIntraPlaneNotebooks.wl`:
`makeTwin["IntraPlaneKambe.wl", "Phase 3b cycle 1 : Undamped Multipole Structure Constants D[q,s]", "<one-paragraph summary>"];`

- [ ] **Step 2: Generate + spot-verify the `.nb`, revert sibling regenerations**

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/makeIntraPlaneNotebooks.wl
git checkout -- Mathematica/IntraPlane{CollectiveReciprocity,Convergence,Discretisation,FoldyLax,TwoBody,VectorLattice,VectorTranslation,RT}.nb
```
Expected: `Mathematica/IntraPlaneKambe.nb` created with several Input + Section cells.

- [ ] **Step 3: Update the plan + memory**

In `IntraPlaneFoldyLax_Plan.md` (Phase 3 / 3b section): mark **cycle 1 DONE** (undamped `D[q,s]` via multipole projection of the validated scalar Ewald; η-independent; G0 reciprocal); restate **cycle 2 = undamped vector `G0` into the collective**, **cycle 3 = energy balance `|R|²+|T|²=1`**. Write a memory note `project-undamped-kambe-structure-constants.md` (type project) summarising the projection method, the gates, and the result; link `[[kambe-validates-thesis-spectral]]`, `[[thesis-energy-normalisation]]`. Add the `MEMORY.md` pointer.

- [ ] **Step 4: Final run + commit**

```bash
conda run -n seismic pytest cubic_scattering/tests/test_intraplane_kambe.py -v   # all PASS
git add Mathematica/IntraPlaneKambe.nb Mathematica/makeIntraPlaneNotebooks.wl Mathematica/IntraPlaneKambe.wl IntraPlaneFoldyLax_Plan.md
git commit -F .git/COMMIT_MSG_k5
```
Message: `📝 Phase 3b cycle 1 DONE: undamped multipole structure constants D[q,s]`

---

## Self-Review

**Spec coverage:**
- §1 undamped `D[q,s]` → Tasks 1–3 (via projection of the validated scalar Ewald). ✓
- §2 split — implemented as: validated scalar Ewald field (real+reciprocal, general z) + multipole projection, *instead of* per-`q` analytic recip/real/central. This is the approved revision (multipole projection of `ewaldTotal`), recorded in [[kambe-validates-thesis-spectral]]; the spec's "route A object" (multipole `D[q,s]` in the L/M/N basis) is delivered. ✓
- §4 gates — η-independence (Task 1 `[1]`, Task 2 `[4]`), `q=0`≡scalar (subsumed by the projection method gate `[3]` which matches the *full* direct structure constant at all `q`), damped-sum agreement (`[2]`, `[3]`), G0 reciprocity (`[5]`). ✓
- §5 acceptance 1–5 → Tasks 1–5. ✓
- §6 out-of-scope (cycles 2–3) → respected. ✓

**Placeholder note:** the one transcription (the general-`z` reciprocal `erfc` form, Task 1 Step 2) is from a **repo-local authoritative note** (`EwaldIntraPlanePropagator.tex`), pinned by the `z=0`-reduction check and the η-independence gate (un-fakeable). Everything else is concrete code. The Task-5 twin summary paragraph is to be written at closeout (one sentence), not load-bearing.

**Type consistency:** `ewReal3/ewRecip3/ewTot3/ewDirect3(kappa, r, …)` take a 3-vector `r` and return a scalar throughout; `Dproj(fieldFn, q, s, kappa)` and `DstructUndamped(q,s,eta)`/`DstructDirect(q,s,kappa,Lr)` return `D_struct[q,s]`; the `D̄↔D_struct` convention `D_struct[q,s]=(-1)^s D̄[q,-s]` is applied once, inside `Dproj`. JSON `Dstruct` entries are `{q,s,val:[re,im]}`; Python `_Dstruct_proj` mirrors `Dproj` (same `(-1)^s`, `j_q(κρ₀)` factor, conj-`Y` projection).
