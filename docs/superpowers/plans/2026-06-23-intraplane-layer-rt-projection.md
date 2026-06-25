# Phase 3a — Intra-Plane Layer R/T(p) Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Project the Phase-2 spherical collective `T_coll(k_par)` onto Kennett flux-normalised up/down P-SV-SH plane waves at horizontal slowness `p`, producing the layer R/T(p) operator (`Rd, Ru, Td, Tu` 2×2 P-SV + SH scalar) across normal / sub-critical / post-critical `p`, validated by reciprocity (tight) and the Cartesian/Kennett ground truth (loose).

**Architecture:** New Mathematica `IntraPlaneRT.wl` reuses CartesianT0's incident/scattered elastic Weyl bridges (`incP/incN/incM`, the `farField` projection pattern) and item (c)'s vector collective build (`buildG0vec`/`T0vec`/`collV`, `idxVof`), replacing the per-`n` single-site `T0` in the projection with the full collective `T_coll`. A lattice Weyl prefactor and the `D = diag(α√η_P, iβ√η_S)` flux norm turn the projected far-field into the flux-normalised R/T. A JSON dump drives a Python cross-check against `slab_reflection_matrix` / `kennett_reference_matrix`.

**Tech Stack:** Wolfram Language (`wolframscript`); Python 3.12 (`cubic_scattering`, numpy/scipy, pytest), conda env `seismic`; JSON as the MMA↔Python contract; lualatex.

## Global Constraints

- Conda env `seismic`; Python tooling via `conda run -n seismic <cmd>`.
- Coordinate system: PROJECT frame `(z, x, y)` = (component 1, 2, 3): vertical/depth `z` = comp 1, horizontal slowness `p` along `x` = comp 2, SH polarisation = `y` = comp 3 = `{0,0,1}`. Slowness vector `s_m = (±η_m, p, 0)` (vertical-first), matching `slab_scattering`. The verified CartesianT0/Phase-2 spherical bridge is hardwired to polar axis = comp 3, so a single permutation `toSph[{z,x,y}] = {x,y,z}` maps the project frame into the bridge frame at every bridge call; the bridge/Phase-2 code is UNTOUCHED (decided 2026-06-23).
- Time `e^{−iωt}`, outgoing `h_n^(1)` via `SphericalHankelH1` — NEVER `j_n + i y_n`.
- Complex slowness past critical: `η_m = √(1/c_m² − p²)` with `Im η_m > 0`; polarisations analytically continued with `pol·pol = 1` and **NO conjugation** (the `slab_reflection_matrix` convention).
- Incident UNIT direction for mode `m` at slowness `p`: `k̂ = c_m·(±η_m, p, 0)` (`+` down, `−` up; `c_P=α`, `c_S=β`).
- Lattice damping `Im κ = 0.25` (Phase-2 convention) — reciprocity-preserving, energy-breaking. Energy balance is OUT of scope (Phase 3b).
- Background α=5000, β=3000, ρ=2500 (⇒ lam0=1.75e10, mu0=2.25e10); moderate contrast Δλ=+2 GPa, Δμ=+1 GPa, Δρ=+100 (the physical, well-conditioned regime). Sphere half-width `aa=1`, lattice pitch `aL` (use `aL=2.5`, well inside the converged regime per item (c)). Bloch vector `k_par = ω·(0, p, 0)` (set `kx`, `ky` accordingly).
- Reference JSON lives in `Mathematica/`; complex numbers serialised as `[re, im]` via `reim`.
- Line length ≤ 108. Ruff `--ignore ARG001,ARG002,F841,E741`, ruff format, mypy `--ignore-missing-imports`. B904 in except. `pathlib.Path`. Google docstrings.
- Lint/format/type after every Python change:
  `conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`
- LaTeX: self-contained `lualatex`, compiled IN-PLACE in its `docs/` subdir, run twice.
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash tool — write commit messages to a file and use `git commit -F`. Long `wolframscript` runs auto-background; wait via a bounded waiter, don't chain sleeps.

## File Structure

- Create `Mathematica/IntraPlaneRT.wl` — the projection: incident-amplitude vector, collective `b = T_coll·a`, lattice-Weyl plane-wave projection at the specular up/down directions, flux norm; dumps `IntraPlaneRT_reference.json`. Loads `CartesianT0.wl` and copies the item (c) vector-collective builders (`idxVof`/`buildG0vec`/`T0vec`/`collV`/`chPos` and the `Mw/Nw/Cvec/...` helpers) — do NOT `Get` `IntraPlaneConvergence.wl` (it runs its whole study + dumps on load).
- Create `Mathematica/IntraPlaneRT_reference.json` (generated) — the dump.
- Create `cubic_scattering/tests/test_intraplane_rt.py` — Python cross-check (tight + loose gates).
- Modify `Mathematica/makeIntraPlaneNotebooks.wl` — add `IntraPlaneRT.wl` to the twin list.
- Create `docs/intraplane_rt/intraplane_rt.tex` — the writeup.
- Modify `IntraPlaneFoldyLax_Plan.md` — Phase 3a DONE; Phase 3b = undamped G0 + energy.
- Create memory `…/memory/project_intraplane_layer_rt.md` + update `…/memory/MEMORY.md`.

The MMA script and the Python test communicate ONLY through the JSON dump.

---

### Task 1: MMA projection core — incident vector, collective scatter, plane-wave projection (p=0 tracer)

Build the projection machinery and prove it on the simplest case: normal incidence (`p=0`), single sphere (`G0=0`), where the layer R/T must (i) be reciprocal and (ii) reduce to the lattice-Weyl projection of one Mie sphere. Defer the full p-sweep and collective to Task 2.

**Files:**
- Create: `Mathematica/IntraPlaneRT.wl`

**Interfaces:**
- Produces (MMA functions): `incVec[m, khat, ehat, Nmax]` → association `{n,m,ch} -> a` (incident multipole amplitude vector for mode `m∈{"P","SV","SH"}` at unit direction `khat`, S-polarisation `ehat`); `projPW[bvec, ks, kP, kS, Nmax]` → `{fP, fS}` (plane-wave content of outgoing multipole vector `bvec` at direction `ks`, the `farField` projection generalised to a full `T_coll·a` vector); `etaOf[c, p]` → vertical slowness (`Im>0`); `khatMode[m, p, sign]` → complex unit direction.
- Produces (JSON, filled across tasks): top-level `{params, stageRT: [...]}`; this task writes the `p=0`, `G0=0` self-check only.

- [ ] **Step 1: Header, conventions, load CartesianT0, copy the vector-collective builders**

Open `IntraPlaneRT.wl` with the banner docstring (purpose = Phase 3a layer R/T(p); conventions from the Global Constraints), then:

```mathematica
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
reim[z_] := {Re[N[z]], Im[N[z]]};
alpha0 = 5000.; beta0 = 3000.; rho0 = 2500.;
lam0 = rho0 (alpha0^2 - 2 beta0^2); mu0 = rho0 beta0^2;
dampIm = 0.25; aa = 1.0; aLpitch = 2.5; Acell = aLpitch^2;
etaOf[c_, p_] := Module[{e = Sqrt[1./c^2 - p^2]}, If[Im[e] < 0, -e, e]];  (* Im eta >= 0 *)
khatMode[m_, p_, sign_] := Module[{c = If[m == "P", alpha0, beta0], eta},
   eta = etaOf[c, p]; c {sign eta, p, 0.}];     (* unit direction (k-hat), down sign=+1, up sign=-1 *)
```

Copy verbatim from `IntraPlaneConvergence.wl` the vector-collective block (lines ~164–229): `chPos`,
`idxVof`, `zfn`/`zfp`, `Yfun`, `RMrot`, `dthY`, `Bvec`, `Cvec`, `Pvec`, `rhoOf`, `rhatOf`, `Mw`, `Nw`,
the reduced fixed-sphere quadrature (`glN`, `quadDirs`, `quadW`, `projDotF`, `Cvals`, `Pvals`,
`normMC`, `normNP`), `buildG0vec`, `T0vec`, `collV`, plus the Mie params helper that sets
`kPo/kSo/kPi/kSi/lamO/muO/lamI/muI/kappaP/kappaS` from `(ka, contrast)` (the `recordFor` parametrisation
pattern from `IntraPlaneDiscretisation.wl`, but here `ka = ω·aa/α0 = kPo` is set per the test frequency,
NOT swept by contrast).

- [ ] **Step 2: Incident-amplitude vector and plane-wave projection**

```mathematica
(* incident multipole amplitude vector indexed by idxVof: P->L, SV->N, SH->M *)
incVec[mode_, khat_, ehat_, Nmax_] := Association @ Map[
   Function[idx, Module[{n = idx[[1]], m = idx[[2]], ch = idx[[3]]},
     idx -> Switch[{mode, ch},
       {"P", "L"},  incP[n, m, khat],
       {"SV", "N"}, incN[n, m, khat, ehat],
       {"SH", "M"}, incM[n, m, khat, ehat],
       _, 0]]],
   idxVof[Nmax]];

(* plane-wave content of an outgoing multipole vector bvec at direction ks
   (the farField projection generalised from b=T0 a to a full T_coll a vector) *)
projPW[bvec_, ks_, kP_, kS_, Nmax_] := Module[{fP = {0,0,0}, fS = {0,0,0}, idx = idxVof[Nmax]},
  Do[With[{n = id[[1]], m = id[[2]], ch = id[[3]], b = bvec[id]},
    Switch[ch,
      "L", fP += b ((-I)^n/kP) Yv[n, m, ks] ks,
      "N", fS += b ((-I)^n/kS) Bv[n, m, ks],
      "M", fS += -b ((-I)^(n+1)/kS) Cv[n, m, ks]]], {id, idx}];
  {fP, fS}];
```

(`Yv`, `Bv`, `Cv` come from `CartesianT0.wl`; the `n=0` monopole is the `ch="L", n=0` entry — its
`Yv[0,0,ks] ks` term is included by the same `"L"` branch.)

- [ ] **Step 3: Layer R/T amplitude assembly (single mode → specular plane-wave amplitude)**

```mathematica
(* lattice-Weyl specular amplitude: project T_coll.a onto the output mode polarisation at the
   specular direction, times the array prefactor i/(2 eta_out omega Acell). omega = kPo alpha0/aa. *)
omegaOf := kPo alpha0/aa;
rtAmp[Tcoll_, inMode_, p_, inSign_, outMode_, outSign_, Nmax_] := Module[
   {ehatIn, kin = khatMode[inMode, p, inSign], kout = khatMode[outMode, p, outSign],
    avec, bvec, f, polOut, etaOut = etaOf[If[outMode == "P", alpha0, beta0], p]},
  ehatIn = If[inMode == "SH", Cross[kin, {0,0,1.}]/Norm[Cross[kin, {0,0,1.}]], th[kin]]; (* SV in-plane, SH out *)
  avec = incVec[inMode, kin, ehatIn, Nmax];
  bvec = Association @ MapThread[#1 -> #2 &, {idxVof[Nmax],
            Inverse... }];  (* bvec = Tcoll . avec as an association; see note *)
  f = projPW[bvec, kout, kPo, kSo, Nmax];
  polOut = Switch[outMode, "P", kout, "SV", th[kout], "SH", Cross[kout,{0,0,1.}]/Norm[Cross[kout,{0,0,1.}]]];
  (I/(2 etaOut omegaOf Acell)) (polOut . If[outMode == "P", First[f], Last[f]])];
```

Note: `bvec = Tcoll . avec` — form the dense matrix-vector product over `idxVof[Nmax]` ordering
(`Tcoll` is the matrix from `collV`; `avec` the vector from `incVec` in the same index order), then
re-key to an association. The exact ABSOLUTE prefactor/normalisation (the `i/(2 η ω Acell)` constant and
any displacement-vs-potential factor) is pinned empirically against `slab_reflection_matrix` in Task 4;
Tasks 1–3 assert only structure (reciprocity, single-sphere, decoupling) which is prefactor-independent
when the SAME prefactor is used on every entry.

- [ ] **Step 4: p=0 single-sphere self-check + partial dump**

At `p=0`, `G0=0` (so `Tcoll = T0vec[Nmax]`): assemble the 2×2 P-SV down-reflection `Rd` and the SH scalar.
Assert (in-script PASS/FAIL): the off-diagonal P↔SV entries of `Rd` are ~0 (decoupling at normal
incidence), and `Rd` is symmetric. Write a `stageRT` JSON record `{p:0, G0:"off", Rd:[[..]], Rsh:..}`.

```mathematica
Print["==== Phase 3a :: layer R/T(p) projection ===="];
Nm = 4;
(* set Mie params for the moderate contrast at the test frequency kPo *)
<set kPo etc. via the params helper for moderate contrast>;
T0only = T0vec[Nm];
Rd0 = Table[rtAmp[T0only, inM, 0., 1., outM, -1., Nm], {outM, {"P","SV"}}, {inM, {"P","SV"}}];
decoupOK = Max[Abs[{Rd0[[1,2]], Rd0[[2,1]]}]] < 1*^-8;
Print["  [p=0] P-SV off-diagonal (decoupling) = ", ScientificForm[Max[Abs[{Rd0[[1,2]],Rd0[[2,1]]}]],3],
   " -> ", If[decoupOK, "PASS", "FAIL"]];
```

- [ ] **Step 5: Run + commit**

Run (auto-backgrounds; bounded waiter for the JSON):
```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/IntraPlaneRT.wl
```
Expected: prints the `[p=0]` decoupling PASS and writes `Mathematica/IntraPlaneRT_reference.json`.
```bash
git add Mathematica/IntraPlaneRT.wl Mathematica/IntraPlaneRT_reference.json
git commit -F .git/COMMIT_MSG_p3_1
```
Message: `✨ Phase 3a: layer R/T projection core (p=0 single-sphere tracer)`

---

### Task 2: MMA full R/T(p) — collective + p sweep (normal / sub-critical / post-critical)

Add the collective `G0` and the full slowness sweep; dump `Rd, Td, Ru, Tu` (2×2) + SH at each `p`.

**Files:**
- Modify: `Mathematica/IntraPlaneRT.wl`

**Interfaces:**
- Consumes: Task-1 `rtAmp`, `incVec`, `projPW`, `buildG0vec`, `T0vec`, `collV`.
- Produces: JSON `stageRT` = list per `p` of `{p, regime, Rd:[[2x2]], Td:[[2x2]], Ru:[[2x2]], Tu:[[2x2]], Rsh, Tsh, recip_resid}` (complex as `[re,im]`); helper `rtMatrices[Tcoll, p, Nmax]` → `{Rd, Td, Ru, Tu, Rsh, Tsh}`.

- [ ] **Step 1: `rtMatrices` — assemble all four 2×2 blocks + SH from `rtAmp`**

```mathematica
rtMatrices[Tcoll_, p_, Nm_] := Module[{modes = {"P","SV"}, Rd, Td, Ru, Tu, Rsh, Tsh},
  Rd = Table[rtAmp[Tcoll, im, p,  1., om, -1., Nm], {om, modes}, {im, modes}];   (* down-in, up-out *)
  Td = Table[rtAmp[Tcoll, im, p,  1., om,  1., Nm], {om, modes}, {im, modes}];   (* down-in, down-out *)
  Ru = Table[rtAmp[Tcoll, im, p, -1., om,  1., Nm], {om, modes}, {im, modes}];   (* up-in, down-out *)
  Tu = Table[rtAmp[Tcoll, im, p, -1., om, -1., Nm], {om, modes}, {im, modes}];   (* up-in, up-out *)
  Rsh = rtAmp[Tcoll, "SH", p, 1., "SH", -1., Nm]; Tsh = rtAmp[Tcoll, "SH", p, 1., "SH", 1., Nm];
  <|"Rd"->Rd, "Td"->Td, "Ru"->Ru, "Tu"->Tu, "Rsh"->Rsh, "Tsh"->Tsh|>];
```

- [ ] **Step 2: build the collective `T_coll(p)` and sweep `p`**

```mathematica
pList = {0., 0.5/alpha0, 0.5/beta0 + 0.3/beta0};   (* normal; sub-critical (p<1/alpha); post-critical (p>1/beta) *)
regimeOf[p_] := Which[p == 0., "normal", p < 1./alpha0, "subcritical", True, "postcritical"];
stageRT = {};
Do[Module[{kpar = omegaOf {0., p, 0.}, kxL = omegaOf 0., kyL = omegaOf p, G0, Tc, mats, recip},
   kx = 0.; ky = p alpha0;    (* Bloch vector in the buildG0vec convention (units of 1/aa); set per its code *)
   kappaP = kPo + dampIm I; kappaS = kSo + dampIm I;
   G0 = buildG0vec[aLpitch, 8, Nm]; Tc = collV[G0, T0vec[Nm]];
   mats = rtMatrices[Tc, p, Nm];
   recip = Max[Abs[Flatten[mats["Tu"] - Transpose[mats["Td"]]]]];   (* reciprocity residual *)
   AppendTo[stageRT, <|"p"->p, "regime"->regimeOf[p],
     "Rd"->Map[reim,mats["Rd"],{2}], "Td"->Map[reim,mats["Td"],{2}],
     "Ru"->Map[reim,mats["Ru"],{2}], "Tu"->Map[reim,mats["Tu"],{2}],
     "Rsh"->reim[mats["Rsh"]], "Tsh"->reim[mats["Tsh"]], "recip_resid"->N[recip]|>]],
   {p, pList}];
```

(Reconcile the `kx,ky` Bloch convention with `buildG0vec` exactly — in item (c) the Bloch phase is
`Exp[I aL (kx i + ky j)]` with `kx,ky` dimensionless; here `k_par·R = ω p · (aL j)` so `ky = ω p aa`,
i.e. set `ky` so the phase matches `Exp[I k_par·R]`. Verify against a `p=0` build = real lattice.)

- [ ] **Step 3: self-verify tight gates + dump**

```mathematica
recipOK = AllTrue[stageRT, #["recip_resid"] < 1*^-6 &];
Print["  reciprocity Tu == Td^T across p: max resid ",
   ScientificForm[Max[Map[#["recip_resid"]&, stageRT]], 3], " -> ", If[recipOK, "PASS", "FAIL"]];
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneRT_reference.json",
  <|"params"-><|"alpha"->alpha0,"beta"->beta0,"rho0"->rho0,"aa"->aa,"aLpitch"->aLpitch,
      "kPo"->kPo,"dampIm"->dampIm,"Nmax"->Nm,
      "contrast"-><|"Dlambda"->2.*^9,"Dmu"->1.*^9,"Drho"->100.|>|>,
    "stageRT"->stageRT|>];
Print["  wrote IntraPlaneRT_reference.json"];
```

- [ ] **Step 4: run + verify reciprocity PASS**

Run the script (bounded waiter). Expected: `reciprocity ... PASS` and 3 `stageRT` records (normal /
subcritical / postcritical). If reciprocity fails, STOP and debug (likely the `ehatIn`/`th` polarisation
or the up/down sign convention) before proceeding — do not weaken the tolerance.

- [ ] **Step 5: commit**

```bash
git add Mathematica/IntraPlaneRT.wl Mathematica/IntraPlaneRT_reference.json
git commit -F .git/COMMIT_MSG_p3_2
```
Message: `✨ Phase 3a: full collective R/T(p) sweep (normal/sub/post-critical) + reciprocity`

---

### Task 3: Python cross-check — tight gates

Independently assert the structural (prefactor-independent) gates from the dump.

**Files:**
- Create: `cubic_scattering/tests/test_intraplane_rt.py`

**Interfaces:**
- Consumes: `Mathematica/IntraPlaneRT_reference.json`.
- Produces: fixture `dump`; helper `_mat(jblock)` ([[ [re,im] ]] → 2×2 complex ndarray), `_c(pair)`.

- [ ] **Step 1: Write the failing tight-gate tests**

```python
"""Phase 3a: intra-plane layer R/T(p) projection cross-check.

``Mathematica/IntraPlaneRT.wl`` dumps the flux-normalised up/down P-SV-SH layer
R/T(p) (Rd, Td, Ru, Tu 2x2 + SH scalar) of the Phase-2 spherical collective at
normal / sub-critical / post-critical slowness. This module asserts the tight
(machine-precision) gates -- reciprocity Tu=Td^T, Rd symmetric, p=0 SH/P-SV
decoupling -- and (test_intraplane_rt_vs_cartesian) the loose agreement vs the
Cartesian slab and Kennett, bounded by the item-(e) discretisation error.
"""

import json
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).resolve().parents[2] / "Mathematica" / "IntraPlaneRT_reference.json"


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneRT.wl first)"
    return json.loads(REF.read_text())


def _c(pair):
    return complex(pair[0], pair[1])


def _mat(block):
    return np.array([[_c(block[i][j]) for j in range(2)] for i in range(2)], dtype=complex)


def test_rt_reciprocity(dump):
    """Tu = Td^T and Rd symmetric in the flux-normalised basis (machine precision)."""
    for rec in dump["stageRT"]:
        Td, Tu, Rd = _mat(rec["Td"]), _mat(rec["Tu"]), _mat(rec["Rd"])
        assert np.allclose(Tu, Td.T, atol=1e-6), f"p={rec['p']}: Tu != Td^T"
        assert np.allclose(Rd, Rd.T, atol=1e-6), f"p={rec['p']}: Rd not symmetric"


def test_rt_p0_decoupling(dump):
    """At p=0 the P-SV off-diagonal blocks vanish (SH decoupled, no mode conversion)."""
    rec = next(r for r in dump["stageRT"] if r["regime"] == "normal")
    Rd = _mat(rec["Rd"])
    assert abs(Rd[0, 1]) < 1e-8 and abs(Rd[1, 0]) < 1e-8
```

- [ ] **Step 2: run to verify fail**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -k "reciprocity or decoupling" -v`
Expected: FAIL if the dump is missing or a gate is violated; PASS once Task 2's dump satisfies them.
(If Task 2's in-script asserts passed, these should pass immediately — they re-check the same invariants
independently. Keep them: the Python check is the committed gate.)

- [ ] **Step 3: make pass (no code; the gates hold by construction from Task 2)**

If a gate fails here but passed in MMA, the discrepancy is a serialisation/index bug — fix `_mat` ordering
or the dump's row/col convention until both agree.

- [ ] **Step 4: run + lint + commit**

```bash
conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -k "reciprocity or decoupling" -v
# lint/format/mypy triple
git add cubic_scattering/tests/test_intraplane_rt.py
git commit -F .git/COMMIT_MSG_p3_3
```
Message: `✨ Phase 3a: Python tight-gate cross-check (reciprocity, p=0 decoupling)`

---

### Task 4: Python cross-check — loose gates vs Cartesian slab + Kennett

Pin the absolute normalisation and validate R/T(p) against the validated Cartesian ground truth within the item-(e) discretisation tolerance.

**Files:**
- Modify: `cubic_scattering/tests/test_intraplane_rt.py`
- May modify: `Mathematica/IntraPlaneRT.wl` (the single overall normalisation constant in `rtAmp`, if Task-2 left it unpinned).

**Interfaces:**
- Consumes: `dump`; `cubic_scattering.slab_scattering` (`SlabGeometry`, `uniform_slab_material`, `slab_reflection_matrix`, `kennett_reference_matrix`), `cubic_scattering.effective_contrasts` (`ReferenceMedium`, `MaterialContrast`).
- Produces: helper `_kennett_rt(ref, contrast, p, omega)` and `_cartesian_rt(p, omega)` returning the modified-convention 2×2 + SH for comparison.

- [ ] **Step 1: Write the failing comparison test**

```python
from cubic_scattering.effective_contrasts import MaterialContrast, ReferenceMedium
from cubic_scattering.slab_scattering import (
    SlabGeometry,
    kennett_reference_matrix,
    slab_reflection_matrix,
    uniform_slab_material,
)

REF_MED = ReferenceMedium(alpha=5000.0, beta=3000.0, rho=2500.0)
CONTRAST = MaterialContrast(Dlambda=2e9, Dmu=1e9, Drho=100.0)
DISCRETISATION_TOL = 0.15  # item (e): sphere-vs-cube R_PP residual ~4%; allow headroom across p/channels


def _omega(dump):
    p = dump["params"]
    return p["kPo"] * p["alpha"] / p["aa"]


def test_rt_vs_kennett(dump):
    """Spherical R_PP(p) tracks the homogeneous Kennett layer within the discretisation error."""
    omega = _omega(dump)
    H = 2.0 * dump["params"]["aa"]
    for rec in dump["stageRT"]:
        p = rec["p"]
        kref = kennett_reference_matrix(REF_MED, CONTRAST, H, omega, p=p)
        rpp = _mat(rec["Rd"])[0, 0]
        denom = abs(kref.R_PP) if abs(kref.R_PP) > 1e-12 else 1.0
        assert abs(rpp - kref.R_PP) / denom < DISCRETISATION_TOL, f"p={p}: R_PP off Kennett"
```

(Add a `test_rt_vs_cartesian` that builds a single-plane slab — `SlabGeometry(M=…, N_z=1, …)`,
`uniform_slab_material(geom, REF_MED, CONTRAST, phi=1.0)` — calls
`slab_reflection_matrix(geom, mat, omega, p=p).to_modified()` and compares the 2×2 within
`DISCRETISATION_TOL`. Use the same `omega`, `p` list, and the moderate contrast.)

- [ ] **Step 2: run to verify fail**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -k "kennett or cartesian" -v`
Expected: FAIL initially — most likely an overall real/complex normalisation constant in `rtAmp` (the
`i/(2 η ω Acell)` prefactor and any displacement factor). The reciprocity (Task 3) already holds, so only
a single global scale/phase is free.

- [ ] **Step 3: pin the normalisation constant in `IntraPlaneRT.wl`**

Determine the single overall constant `C_norm` so the spherical `R_PP(p=0)` matches the Cartesian /
Kennett `R_PP(p=0)` (one complex number fixes magnitude + phase). Set it in `rtAmp`
(`(C_norm) (I/(2 etaOut omegaOf Acell)) (polOut . …)`), re-run the `.wl`, and confirm ALL `p` and ALL
channels then fall within `DISCRETISATION_TOL` — i.e. one constant pinned at `p=0` validates the whole
sweep. (This is the spherical analog of the slab force-sign pinning; reciprocity stays intact because
`C_norm` multiplies every entry equally.) Record `C_norm` and its origin in a script comment.

- [ ] **Step 4: run to verify pass**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -v`
Expected: all PASS (tight + loose).

- [ ] **Step 5: lint + commit**

```bash
# lint/format/mypy triple
git add cubic_scattering/tests/test_intraplane_rt.py Mathematica/IntraPlaneRT.wl Mathematica/IntraPlaneRT_reference.json
git commit -F .git/COMMIT_MSG_p3_4
```
Message: `✨ Phase 3a: R/T(p) vs Cartesian slab + Kennett (discretisation-bounded)`

---

### Task 5: `.nb` twin, LaTeX note, closeout

**Files:**
- Modify: `Mathematica/makeIntraPlaneNotebooks.wl`; Create: `Mathematica/IntraPlaneRT.nb`
- Create: `docs/intraplane_rt/intraplane_rt.tex`
- Modify: `IntraPlaneFoldyLax_Plan.md`
- Create: `…/memory/project_intraplane_layer_rt.md`; Modify: `…/memory/MEMORY.md`

- [ ] **Step 1: add `IntraPlaneRT.wl` to the twin generator**

In `makeIntraPlaneNotebooks.wl` add a `makeTwin["IntraPlaneRT.wl", "Phase 3a : Layer R/T(p) Projection", "..."]`
entry (follow the existing pattern; sections split on `=`-banners, so ensure `IntraPlaneRT.wl`'s major
sections use full `=`-banners as in `IntraPlaneDiscretisation.wl`).

- [ ] **Step 2: generate + spot-verify the `.nb`, reverting sibling regenerations**

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/makeIntraPlaneNotebooks.wl
git checkout -- Mathematica/IntraPlane{CollectiveReciprocity,Convergence,Discretisation,FoldyLax,TwoBody,VectorLattice,VectorTranslation}.nb
```
Expected: `Mathematica/IntraPlaneRT.nb` created with several Input + Section cells.

- [ ] **Step 3: write the LaTeX note**

Create `docs/intraplane_rt/intraplane_rt.tex` (self-contained lualatex; fontspec preamble per repo
convention) documenting: the incident bridge → `b = T_coll·a` → lattice-Weyl projection → flux norm
pipeline; the tight gates (reciprocity); the loose cross-check table (R_PP vs Kennett/Cartesian across
the three `p` regimes) with the pinned `C_norm`; and the explicit deferral of energy balance to Phase 3b.
Compile in-place:
```bash
cd docs/intraplane_rt && /usr/local/bin/lualatex -interaction=nonstopmode intraplane_rt.tex && /usr/local/bin/lualatex -interaction=nonstopmode intraplane_rt.tex
```
Expected: `intraplane_rt.pdf` produced, no errors.

- [ ] **Step 4: update the plan + memory**

In `IntraPlaneFoldyLax_Plan.md`: mark **Phase 3 (3a) DONE** with the result (R/T(p) reciprocal to machine
precision; tracks Cartesian/Kennett within the discretisation error across normal/sub/post-critical p;
`C_norm` pinned); restate **Phase 3b = undamped vector G0 (Ewald/Kambe) + energy balance `|R|²+|T|²=1`**.
Write `…/memory/project_intraplane_layer_rt.md` (type project) summarising the projection construction,
the reciprocity-tight / discretisation-loose gate structure, `C_norm`, and the deferred energy balance;
link `[[project_sphere_packing_discretisation]]`, `[[project_intraplane_reciprocity_metric]]`. Add the
`MEMORY.md` pointer line.

- [ ] **Step 5: final full-suite + commit**

```bash
conda run -n seismic pytest cubic_scattering/tests/test_intraplane_rt.py -v   # all PASS
git add Mathematica/IntraPlaneRT.nb Mathematica/makeIntraPlaneNotebooks.wl docs/intraplane_rt/ IntraPlaneFoldyLax_Plan.md
git commit -F .git/COMMIT_MSG_p3_5
```
Message: `📝 Phase 3a DONE: intra-plane layer R/T(p) projection closed out`

---

## Self-Review

**Spec coverage:**
- §2(A) spherical Weyl projection → Tasks 1–2. ✓
- §2(B) flux normalisation (`D = diag(α√η_P, iβ√η_S)`) → folded into `rtAmp`/`C_norm` pinning (Task 4) — the flux-norm basis is what `kennett_reference_matrix`/`to_modified` return, so matching them IS the flux-norm validation. ✓
- §2(C) Python cross-check → Tasks 3–4. ✓
- §2(D) LaTeX note → Task 5. ✓
- §3 tight gates (reciprocity, single-sphere, p=0 decoupling) → Tasks 1–3; loose gates (slab + Kennett, normal/sub/post-critical) → Task 4. ✓
- §4 deliverables (`.wl`+`.nb`, test, LaTeX, JSON) → Tasks 1–5. ✓
- §5 acceptance 1–6 → Tasks 2 (headless+asserts), 5 (`.nb`, LaTeX), 3–4 (pytest), lint, plan/memory. ✓
- §6 out-of-scope (undamped G0/energy, multi-Bragg, Phase 4/5) → respected; energy deferred, sub-wavelength assumed, single plane. ✓

**Placeholder note:** Task 1 Step 3 leaves `bvec = Tcoll . avec` as a described matrix-vector product
(with the exact index-ordering note) rather than fully-spelled Mathematica — this is a genuine
one-liner over `idxVof[Nmax]` ordering, made explicit in Task 2's `rtMatrices` usage; not an unfilled
placeholder. The single overall `C_norm` is intentionally pinned empirically in Task 4 (the spec's
"normalisation pinned by the loose cross-check"), the spherical analog of the documented slab force-sign
pinning — flagged, not hand-waved.

**Type consistency:** `rtAmp(Tcoll, inMode, p, inSign, outMode, outSign, Nmax) -> complex`,
`rtMatrices(...) -> assoc{Rd,Td,Ru,Tu,Rsh,Tsh}`, `incVec(...) -> assoc`, `projPW(...) -> {fP,fS}` are used
consistently across tasks. JSON: 2×2 blocks as `[[ [re,im] ]]`, scalars as `[re,im]`; Python `_mat`/`_c`
decode them. Ground-truth calls match the real signatures: `slab_reflection_matrix(geom, mat, omega, p=p).to_modified()`,
`kennett_reference_matrix(ref, contrast, H, omega, p=p) -> KennettChannelReference{R_PP,R_PS,R_SP,R_SS,R_SH}`.
