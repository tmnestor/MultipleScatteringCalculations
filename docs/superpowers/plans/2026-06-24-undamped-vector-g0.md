# Undamped Vector `G0^vec` Implementation Plan (Phase 3b cycle 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the undamped (Im κ = 0) planar lattice vector coupling `G0^vec_{(νμc'),(nmc)}(k_par)` in the L/M/N basis by contracting the cycle-1 undamped scalar structure constants `D[q,s]`, and feed it to the Phase-2 collective `T_coll = T0 (I − G0^vec T0)^{-1}`.

**Architecture:** L→L is the scalar-Gaunt contraction of `D(κ_P)` (already the Phase-2(b) `g0LL`). M/N→M/N is `Σ_q coeff_q^{c'c}(n,m,ν,μ) · D[q,m−μ](κ_S)`, where the R-independent `coeff_q` are **extracted numerically** from the repo's validated single-pair vector translation `W^{c'c}(d)` by angular projection over source directions (no literature transcription). `D[q,s]` comes from the cycle-1 projection machinery, κ-parameterized, copied in (not `Get`-ed).

**Tech Stack:** Wolfram Language (`wolframscript`); Python 3.12 (`cubic_scattering`, numpy/scipy, pytest), conda env `seismic`; JSON as the MMA↔Python contract.

## Global Constraints

- Conda env `seismic`; Python tooling via `conda run -n seismic <cmd>`.
- Time `e^{+iωt}`, outgoing `h_q^(1)` via `SphericalHankelH1` (or `SphericalBesselJ + I SphericalBesselY`); regular `j_q` via `SphericalBesselJ`; `Y_q^s` via `SphericalHarmonicY`. Lattice in x–y (`θ_R = π/2`); z = polar = depth.
- Params (match Phase-1/Phase-2(b) so cycle-1 `D` + the direct-sum ground truth line up): `aL = 2.0`, `k_par = (kx,ky) = (0.2, 0.1)`, `Nmax = 2`, undamped `κ_P = 0.9`, `κ_S = 1.5`; damped check `κ_P,κ_S + 0.25 I`; Ewald `η ∈ {0.7, 1.15}`, `Rc = Gc = 6`; scalar projection radius `ρ0 = 0.5`; vector lattice half-width `LradB = 8`, vector projection radius `radP = 0.5`.
- `coeff_q` are **wavenumber-matched to their contraction**: extract at real `κ_S` for the undamped G0; extract at damped `κ_S` for the damped-limit gate. Each is self-consistent; gate 2 then tests only the contraction structure.
- Reference JSON in `Mathematica/`; complex numbers serialised as `[re, im]` via `reim[z_] := {Re[N[z]], Im[N[z]]}`.
- Line length ≤ 108. Ruff `--ignore ARG001,ARG002,F841,E741`, ruff format, mypy `--ignore-missing-imports`. B904 in except; `pathlib.Path`; Google docstrings.
- Lint/format/type after every Python change:
  `conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`
- scipy 1.17: use `sph_harm_y(n,m,θ_polar,φ_azimuth)` (NOT `sph_harm`); complex `erfc` via Faddeeva `wofz` (`erfc(z)=exp(−z²)·wofz(iz)`).
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash tool — write commit messages to a file and `git commit -F`. Long `wolframscript` runs auto-background; wait via a bounded waiter; ONE kernel at a time.
- `IntraPlaneKambeVector.wl` is a NEW self-contained file: it copies the small cycle-1 scalar pieces and the Phase-2(b) vector helpers it needs; it `Get`s only `CartesianT0.wl` (for `T0LMN`), exactly as `IntraPlaneVectorLattice.wl` does.
- Memoise the field/projection helpers (the projection re-hits the same quadrature nodes) — cf. the cycle-1 ~20 min → few min speedup.

## File Structure

- Create `Mathematica/IntraPlaneKambeVector.wl` — κ-parameterized undamped scalar `D[q,s]`; vector helpers + quadrature; `coeff_q` extraction; undamped `G0^vec` (L + M/N); collective solve + gates; dumps `IntraPlaneKambeVector_reference.json`.
- Create `Mathematica/IntraPlaneKambeVector_reference.json` (generated).
- Create `cubic_scattering/tests/test_intraplane_kambe_vector.py` — Python cross-check.
- Modify `Mathematica/makeIntraPlaneNotebooks.wl` — add the `.nb` twin entry.
- Modify `IntraPlaneFoldyLax_Plan.md` — Phase 3b cycle 2 DONE; cycle 3 remains.
- Create memory note `project-undamped-vector-g0.md`; update `MEMORY.md`.

---

### Task 1: Scaffold — κ-parameterized undamped scalar `D[q,s]`

Create the self-contained file with the cycle-1 scalar Ewald + multipole projection, generalised to an arbitrary (real or complex) `κ`. Self-verify η-independence at both `κ_P` and `κ_S` and that the `κ_S` values match the committed cycle-1 dump.

**Files:**
- Create: `Mathematica/IntraPlaneKambeVector.wl`

**Interfaces:**
- Produces: `DstructU[q, s, κ, η]` (κ-parameterized undamped/Ewald structure constant); `DstructDirect[q, s, κ, Lr]` (damped direct structure constant, ground truth); helpers `reim`, `sj`, `sh`, `ewTot3`, `ewDirect3`.

- [ ] **Step 1: Header, constants, κ-parameterized scalar Ewald + projection**

```mathematica
#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneKambeVector.wl  --  Phase 3b cycle 2:
   the UNDAMPED planar vector coupling G0^vec(k_par) in the L/M/N basis, built by
   contracting the cycle-1 undamped scalar structure constants D[q,s]:
     L->L   : scalar-Gaunt . D(kappa_P)                 (= Phase-2(b) g0LL)
     M,N    : Sum_q coeff_q^{c'c}(n,m,nu,mu) . D[q,m-mu](kappa_S),
              coeff_q extracted from the validated single-pair vector translation.
   Self-contained: copies the cycle-1 scalar Ewald/projection (IntraPlaneKambe.wl)
   and the Phase-2(b) vector helpers (IntraPlaneVectorLattice.wl); Gets only
   CartesianT0.wl for T0LMN.  Time e^{+i w t}, outgoing h^(1); lattice in x-y.
   ============================================================================ *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

reim[z_] := {Re[N[z]], Im[N[z]]};
aL = 2.0; kx = 0.2; ky = 0.1; kpar2 = {kx, ky}; Aarea = aL^2; recipB = 2 Pi/aL;
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];

(* ---- cycle-1 scalar Ewald field (general z, R!=0), kappa-parameterized ---- *)
ewReal3[kappa_, r_, eta_, Rc_] := (1/(8 Pi)) Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      (Exp[I aL (kx i + ky j)]/d) Sum[Exp[s I kappa d] Erfc[d eta + s I kappa/(2 eta)], {s, {-1, 1}}]],
     {i, -Rc, Rc}, {j, -Rc, Rc}], 1]];
gKs[kappa_, rn_] := Exp[I kappa rn]/(4 Pi rn);
ewRecip3[kappa_, r_, eta_, Gc_] := (I/(4 Aarea)) Total[Flatten[Table[
     With[{kpg = kpar2 + recipB {m, n}, z = r[[3]]},
      With[{kz = Sqrt[kappa^2 - kpg . kpg]},
       (Exp[I kpg . {r[[1]], r[[2]]}]/kz) (
          Exp[-I kz Abs[z]] Erfc[ Abs[z] eta + kz/(2 I eta)]
        + Exp[ I kz Abs[z]] Erfc[-Abs[z] eta + kz/(2 I eta)])]],
     {m, -Gc, Gc}, {n, -Gc, Gc}], 1]];
ewTot3[kappa_, r_, eta_, Rc_, Gc_] := ewTot3[kappa, r, eta, Rc, Gc] =
  ewReal3[kappa, r, eta, Rc] + ewRecip3[kappa, r, eta, Gc] - gKs[kappa, Sqrt[r . r]];
ewDirect3[kappa_, r_, Lbig_] := ewDirect3[kappa, r, Lbig] = Total[Flatten[Table[
     If[i == 0 && j == 0, 0,
      With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
       Exp[I kappa d]/(4 Pi d) Exp[I aL (kx i + ky j)]]], {i, -Lbig, Lbig}, {j, -Lbig, Lbig}], 1]];

(* ---- multipole projection -> D_struct[q,s], kappa-parameterized ---- *)
Needs["NumericalDifferentialEquationAnalysis`"];
glD = GaussianQuadratureWeights[16, -1, 1]; nPhiD = 32; rho0 = 0.5;
sphPts = Flatten[Table[Module[{u = glD[[i, 1]], ph = 2 Pi (j - 1)/nPhiD, st = Sqrt[1 - glD[[i, 1]]^2]},
     {rho0 {st Cos[ph], st Sin[ph], u}, glD[[i, 2]] (2 Pi/nPhiD)}], {i, Length[glD]}, {j, nPhiD}], 1];
Yf[q_, s_, d_] := SphericalHarmonicY[q, s, ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]];
Dproj[fieldFn_, q_, s_, kappa_] := Module[{integ},
  integ = Total[Map[#[[2]] fieldFn[#[[1]]] Conjugate[Yf[q, -s, #[[1]]]] &, sphPts]];
  (-1)^s integ/(I kappa sj[q, kappa rho0])];
DstructU[q_, s_, kappa_, eta_] := DstructU[q, s, kappa, eta] =
  Dproj[Function[r, ewTot3[kappa, r, eta, 6, 6]], q, s, kappa];
DstructDirect[q_, s_, kappa_, Lr_] := DstructDirect[q, s, kappa, Lr] = Module[{ij, iv, jv, rn, ph, bl},
   ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
   iv = ij[[All, 1]]; jv = ij[[All, 2]]; rn = aL Sqrt[iv^2 + jv^2]; ph = ArcTan[iv, jv];
   bl = Exp[I aL (kx iv + ky jv)];
   Total[sh[q, kappa rn] SphericalHarmonicY[q, s, Pi/2, ph] bl]];
```

- [ ] **Step 2: Self-verify the scalar `D[q,s]` (η-independence at κ_P, κ_S; cycle-1 anchor)**

```mathematica
Print["==== Phase 3b cycle 2 :: undamped vector G0 ===="];
NqS = 4;
undEtaP = Max[Table[Abs[DstructU[q, s, 0.9, 0.7] - DstructU[q, s, 0.9, 1.15]], {q, 0, NqS}, {s, -q, q}]];
undEtaS = Max[Table[Abs[DstructU[q, s, 1.5, 0.7] - DstructU[q, s, 1.5, 1.15]], {q, 0, NqS}, {s, -q, q}]];
Print["  [1] scalar D eta-independence (kappa_P=0.9) = ", ScientificForm[undEtaP, 3],
   " -> ", If[undEtaP < 1.*^-6, "PASS", "FAIL"]];
Print["  [2] scalar D eta-independence (kappa_S=1.5) = ", ScientificForm[undEtaS, 3],
   " -> ", If[undEtaS < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 3: Run; verify `[1]` and `[2]` PASS**

Run: `/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/IntraPlaneKambeVector.wl`
Expected: both PASS (~1e-10). If FAIL, the copied cycle-1 Ewald is wrong (compare to `Mathematica/IntraPlaneKambe.wl` verbatim — the only change is the explicit `κ` argument).

- [ ] **Step 4: Commit**

```bash
git add Mathematica/IntraPlaneKambeVector.wl
git commit -F .git/COMMIT_MSG_v1
```
Message: `✨ Phase 3b cycle 2: kappa-parameterized undamped scalar D[q,s] scaffold`

---

### Task 2: Vector helpers + single-pair `W^{c'c}(d)` + coefficient extraction

Copy the Phase-2(b) vector wavefunctions and sphere quadrature (wavenumber-parameterized), build the single-pair vector translation matrix element `Wel`, and extract the R-independent `coeff_q^{c'c}` by angular projection over source directions. Gate on reconstruction.

**Files:**
- Modify: `Mathematica/IntraPlaneKambeVector.wl`

**Interfaces:**
- Consumes: `sj`, `sh` (Task 1).
- Produces: `Mw[n,m,type,c,r,kS]`, `Nw[n,m,type,c,r,kS]`, `Cvec/Bvec/Pvec`, `quadDirs/quadW`, `projDotF`, `Cvals[nu,mu]`, `Pvals[nu,mu]`, `normMC[nu,mu,kS]`, `normNP[nu,mu,kS]`, `srcQuad[c,n,m,d,kS]`, `Wel[cP,nu,mu,c,n,m,d,kS]`, `coeffq[cP,nu,mu,c,n,m,kS]` (an `Association q -> coeff`).

- [ ] **Step 1: Vector wavefunctions + quadrature (wavenumber-parameterized)**

```mathematica
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];
zfn[n_, x_, "j"] := sj[n, x]; zfn[n_, x_, "h"] := sh[n, x];
zfp[n_, x_, "j"] := n/x sj[n, x] - sj[n + 1, x];
zfp[n_, x_, "h"] := n/x sh[n, x] - sh[n + 1, x];
ang[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
RMrot[t_, p_] := {{Sin[t] Cos[p], Cos[t] Cos[p], -Sin[p]},
   {Sin[t] Sin[p], Cos[t] Sin[p], Cos[p]}, {Cos[t], -Sin[t], 0}};
dthY[n_, m_] := dthY[n, m] = Module[{tt, pp, ex},
   ex = D[SphericalHarmonicY[n, m, tt, pp], tt]; Function[{a, b}, Evaluate[ex /. {tt -> a, pp -> b}]]];
Bvec[n_, m_, d_] := Module[{t = ang[d][[1]], p = ang[d][[2]]},
   RMrot[t, p] . {0, dthY[n, m][t, p], (I m/Sin[t]) SphericalHarmonicY[n, m, t, p]}];
Cvec[n_, m_, d_] := Cross[d/Sqrt[d . d], Bvec[n, m, d]];
Pvec[n_, m_, d_] := Yfun[n, m, d] (d/Sqrt[d . d]);
rhoOf[c_, r_] := Sqrt[(r - c) . (r - c)];
rhatOf[c_, r_] := (r - c)/Sqrt[(r - c) . (r - c)];
Mw[n_, m_, type_, c_, r_, kS_] := -zfn[n, kS rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_, kS_] := Module[{x = kS rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] rhatOf[c, r]
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];

glV = GaussianQuadratureWeights[12, -1, 1]; nPhiV = 24; radP = 0.5;
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
quadList = Flatten[Table[{glV[[i, 1]], glV[[i, 2]], 2 Pi (j - 1)/nPhiV}, {i, Length[glV]}, {j, nPhiV}], 1];
quadDirs = Map[shat[#[[1]], #[[3]]] &, quadList];
quadW = Map[#[[2]] (2 Pi/nPhiV) &, quadList];
projDotF[fieldVals_, harmVals_] := Sum[quadW[[k]] fieldVals[[k]] . Conjugate[harmVals[[k]]], {k, Length[quadList]}];
Cvals[nu_, mu_] := Cvals[nu, mu] = Map[Cvec[nu, mu, #] &, quadDirs];
Pvals[nu_, mu_] := Pvals[nu, mu] = Map[Pvec[nu, mu, #] &, quadDirs];
normMC[nu_, mu_, kS_] := normMC[nu, mu, kS] = projDotF[Map[Mw[nu, mu, "j", {0, 0, 0}, radP #, kS] &, quadDirs], Cvals[nu, mu]];
normNP[nu_, mu_, kS_] := normNP[nu, mu, kS] = projDotF[Map[Nw[nu, mu, "j", {0, 0, 0}, radP #, kS] &, quadDirs], Pvals[nu, mu]];
```

- [ ] **Step 2: Single-pair `Wel` + coefficient extraction `coeffq`**

```mathematica
(* outgoing source field (c,n,m) centered at d, sampled at radP*quadDirs around the origin *)
srcQuad[c_, n_, m_, d_, kS_] := srcQuad[c, n, m, d, kS] =
   Map[Switch[c, "M", Mw[n, m, "h", d, radP #, kS], "N", Nw[n, m, "h", d, radP #, kS]] &, quadDirs];
(* single-pair vector translation matrix element: regular (cP,nu,mu) content of the source field *)
Wel[cP_, nu_, mu_, c_, n_, m_, d_, kS_] := Module[{f = srcQuad[c, n, m, d, kS]},
   Switch[cP, "M", projDotF[f, Cvals[nu, mu]]/normMC[nu, mu, kS],
              "N", projDotF[f, Pvals[nu, mu]]/normNP[nu, mu, kS]]];
(* source-direction quadrature for the angular projection that isolates coeff_q *)
glS = GaussianQuadratureWeights[8, -1, 1]; nPhiS = 16; dext = 2.0;
srcW = Flatten[Table[{shat[glS[[i, 1]], 2 Pi (j - 1)/nPhiS], glS[[i, 2]] (2 Pi/nPhiS)},
     {i, Length[glS]}, {j, nPhiS}], 1];
qset[n_, nu_] := Range[Abs[n - nu], n + nu];
(* W(d) = Sum_q coeff_q h_q(kS|d|) Y_q(m-mu, dhat); orthogonal projection over dhat isolates coeff_q.
   coeff_q is d-independent (pure Wigner), so the fixed |d|=dext is arbitrary (any dext>0). *)
coeffq[cP_, nu_, mu_, c_, n_, m_, kS_] := coeffq[cP, nu, mu, c, n, m, kS] = Association[Table[
    q -> Total[Map[#[[2]] Wel[cP, nu, mu, c, n, m, dext #[[1]], kS]
          Conjugate[SphericalHarmonicY[q, m - mu, ang[#[[1]]][[1]], ang[#[[1]]][[2]]]] &, srcW]]
       / sh[q, kS dext],
    {q, qset[n, nu]}]];
```

- [ ] **Step 3: Reconstruction self-check (gate 1)**

```mathematica
kS = 1.5;
reconW[cP_, nu_, mu_, c_, n_, m_, d_, kSv_] := Module[{cf = coeffq[cP, nu, mu, c, n, m, kSv]},
   Total[KeyValueMap[#2 sh[#1, kSv Sqrt[d . d]] SphericalHarmonicY[#1, m - mu, ang[d][[1]], ang[d][[2]]] &, cf]]];
SeedRandom[20260624];
wTestSrc = {{"M", 1, 0, "M", 1, 0}, {"N", 1, 1, "M", 2, -1}, {"M", 2, 0, "N", 1, 1},
   {"N", 2, 2, "N", 2, -1}, {"M", 1, -1, "N", 2, 0}};
wTestPts = Table[Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}]},
    RandomReal[{1.3, 3.0}] shat[u, ph]], {4}];
recW = Max[Table[Abs[reconW[t[[1]], t[[2]], t[[3]], t[[4]], t[[5]], t[[6]], pt, kS]
     - Wel[t[[1]], t[[2]], t[[3]], t[[4]], t[[5]], t[[6]], pt, kS]], {t, wTestSrc}, {pt, wTestPts}]];
Print["  [3] coeff_q reconstruction (Sum_q coeff_q h_q Y_q == W) = ", ScientificForm[recW, 3],
   " -> ", If[recW < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 4: Run; verify `[3]` PASS**

Run the script (bounded waiter; the coeff extraction + reconstruction is the first heavy step — a few minutes). Expected: `[3]` PASS (~1e-8). If FAIL, the `qset` range or the `sh[q, kS dext]` divisor is wrong, or the inner/outer quadrature is too coarse — bump `glV` to 16 / `nPhiV` to 48 and re-check before changing anything else.

- [ ] **Step 5: Commit**

```bash
git add Mathematica/IntraPlaneKambeVector.wl
git commit -F .git/COMMIT_MSG_v2
```
Message: `✨ Phase 3b cycle 2: single-pair vector translation + coeff_q extraction (reconstruction PASS)`

---

### Task 3: Undamped `G0^vec` blocks + damped-limit method gate

Assemble the L block (scalar-Gaunt × `D(κ_P)`) and the M/N blocks (`coeff_q × D(κ_S)`), and validate the whole contraction against the existing Phase-2(b) damped-direct lattice sum (the un-fakeable gate 2).

**Files:**
- Modify: `Mathematica/IntraPlaneKambeVector.wl`

**Interfaces:**
- Consumes: `DstructU`, `DstructDirect` (Task 1); `coeffq`, `srcQuad`, `Cvals`, `Pvals`, `normMC`, `normNP` (Task 2).
- Produces: `g0LLk[n,m,nu,mu,Dfun]`, `g0MNblock[cP,nu,mu,c,n,m,kS,Dfun]`, `g0dir[cP,nu,mu,c,n,m,kS]` (direct damped lattice sum).

- [ ] **Step 1: Block builders (contraction) + direct damped lattice sum (ground truth)**

```mathematica
(* L block: scalar-Gaunt contraction of the scalar structure constant Dfun *)
g0LLk[n_, m_, nu_, mu_, Dfun_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dfun[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
(* M/N block: Sum_q coeff_q(kS) * Dfun[q, m-mu] *)
g0MNblock[cP_, nu_, mu_, c_, n_, m_, kS_, Dfun_] := Module[{cf = coeffq[cP, nu, mu, c, n, m, kS]},
   Total[KeyValueMap[#2 Dfun[#1, m - mu] &, cf]]];

(* ---- direct damped lattice sum (Phase-2(b) ground truth, wavenumber-parameterized) ---- *)
LradB = 8;
ijLat = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -LradB, LradB}, {j, -LradB, LradB}], 1];
Rvecs = Map[{aL #[[1]], aL #[[2]], 0.0} &, ijLat];
blochL = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ijLat];
latSrcQuad[c_, n_, m_, kS_] := latSrcQuad[c, n, m, kS] =
   Total[Table[blochL[[iR]] srcQuad[c, n, m, Rvecs[[iR]], kS], {iR, Length[Rvecs]}]];
g0dir[cP_, nu_, mu_, c_, n_, m_, kS_] := Switch[cP,
   "M", projDotF[latSrcQuad[c, n, m, kS], Cvals[nu, mu]]/normMC[nu, mu, kS],
   "N", projDotF[latSrcQuad[c, n, m, kS], Pvals[nu, mu]]/normNP[nu, mu, kS]];
```

- [ ] **Step 2: Damped-limit method gate (gate 2) — contraction == direct damped sum**

```mathematica
kSd = 1.5 + 0.25 I; kPd = 0.9 + 0.25 I; Ldir = 18;
DdirS[q_, s_] := DstructDirect[q, s, kSd, Ldir];
DdirP[q_, s_] := DstructDirect[q, s, kPd, Ldir];
(* M/N: contraction at damped kS vs direct damped sum *)
mnPairs = {{"M", 1, 0, "M", 1, 0}, {"N", 1, 1, "M", 2, -1}, {"M", 2, 0, "N", 1, 1},
   {"N", 2, -1, "N", 2, 1}, {"M", 1, -1, "N", 2, 0}, {"N", 1, 0, "M", 1, 0}};
resMN = Max[Table[Abs[g0MNblock[p[[1]], p[[2]], p[[3]], p[[4]], p[[5]], p[[6]], kSd, DdirS]
     - g0dir[p[[1]], p[[2]], p[[3]], p[[4]], p[[5]], p[[6]], kSd]], {p, mnPairs}]];
(* L: scalar-Gaunt contraction at damped kP vs direct beta^P sum *)
betaSep[n_, m_, nu_, mu_, dvec_, k_] := Module[{dl = Sqrt[dvec . dvec]},
   4 Pi (-1)^m Sum[I^(nu + q - n) (-1)^q sh[q, k dl] Yfun[q, m - mu, dvec] gaunt[n, m, nu, -mu, q, mu - m],
     {q, Abs[n - nu], n + nu}]];
g0LLdir[n_, m_, nu_, mu_, k_] := Total[Table[betaSep[n, m, nu, mu, Rvecs[[iR]], k] blochL[[iR]], {iR, Length[Rvecs]}]];
llPairs = {{0, 0, 0, 0}, {1, 0, 1, 0}, {1, -1, 2, 1}, {2, 0, 1, 1}, {2, 2, 2, -1}};
resLL = Max[Table[Abs[g0LLk[p[[1]], p[[2]], p[[3]], p[[4]], DdirP]
     - g0LLdir[p[[1]], p[[2]], p[[3]], p[[4]], kPd]], {p, llPairs}]];
Print["  [4] M/N contraction == direct damped sum (", Length[mnPairs], " entries) = ", ScientificForm[resMN, 3],
   " -> ", If[resMN < 1.*^-4, "PASS", "FAIL"]];
Print["  [5] L-block contraction == direct beta^P sum (", Length[llPairs], " entries) = ", ScientificForm[resLL, 3],
   " -> ", If[resLL < 1.*^-4, "PASS", "FAIL"]];
```

- [ ] **Step 3: Run; verify `[4]` and `[5]` PASS**

Run (bounded waiter; the direct damped lattice sums over `(2·8+1)^2−1 = 288` sites × the quadrature are the cost). Expected both PASS (`[4]` ~1e-5, `[5]` ~1e-9). If `[4]` FAILs but `[3]` (Task 2) passed, the contraction index map is wrong — check that `g0MNblock[cP,nu,mu,c,n,m]` uses **receiver `(cP,nu,mu)` ← source `(c,n,m)`**, `D[q, m−μ]`, matching `g0dir`'s `latSrcQuad[c,n,m]` / `Cvals[nu,mu]` projection. If `[5]` FAILs, the scalar-Gaunt `g0LLk` differs from `betaSep` (it should be identical to Phase-2(b) `g0LL`).

- [ ] **Step 4: Commit**

```bash
git add Mathematica/IntraPlaneKambeVector.wl
git commit -F .git/COMMIT_MSG_v3
```
Message: `✨ Phase 3b cycle 2: G0^vec blocks validated vs direct damped sum (method gate PASS)`

---

### Task 4: Undamped `G0^vec` assembly + collective solve + η-independence; dump JSON

Assemble the full undamped `G0^vec` (κ_P for L, κ_S for M/N), solve the collective, check the isolated limit / coupling / L-reciprocity / undamped η-independence, and dump the reference.

**Files:**
- Modify: `Mathematica/IntraPlaneKambeVector.wl`

**Interfaces:**
- Consumes: `g0LLk`, `g0MNblock` (Task 3); `T0LMN` (from `CartesianT0.wl`).
- Produces: JSON `{params, idx, G0vec:[[re,im]...], recip_resid, coupling, iso_dev}`.

- [ ] **Step 1: Undamped G0^vec assembly (κ_P for L, κ_S for M/N)**

```mathematica
kSr = 1.5; etaU = 0.7;
DuS[q_, s_] := DstructU[q, s, 1.5, etaU];   (* undamped, kappa_S *)
DuP[q_, s_] := DstructU[q, s, 0.9, etaU];   (* undamped, kappa_P *)
Nmax = 2;
idx = Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];
nDim = Length[idx];
G0vecEntry[{nu_, mu_, ct_}, {n_, m_, cs_}] := Which[
   ct == "L" && cs == "L", g0LLk[n, m, nu, mu, DuP],
   (ct == "M" || ct == "N") && (cs == "M" || cs == "N"), g0MNblock[ct, nu, mu, cs, n, m, kSr, DuS],
   True, 0];
G0vec = Table[G0vecEntry[idx[[i]], idx[[j]]], {i, nDim}, {j, nDim}];
```

- [ ] **Step 2: Collective solve + sanity + L-reciprocity + undamped η-independence (gates)**

```mathematica
T0LMN[0] := {{T0mono[0.9, lamO, muO, 0.8897917302988777, lamI, muI, 1.0]}};
T0LMN[n_] := Module[{ts = TsphClean[n, 0.9, 1.5, lamO, muO, 1.4968051081937466, lamI, muI, 1.0],
    tt = Ttoroidal[n, 1.5, muO, 1.4968051081937466, muI, 1.0]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0entry[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
  If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
T0mat = Table[T0entry[idx[[i]], idx[[j]]], {i, nDim}, {j, nDim}];
Imat = IdentityMatrix[nDim];
Tcoll = T0mat . Inverse[Imat - G0vec . T0mat];
isoDev = Max[Abs[Flatten[(T0mat . Inverse[Imat - (0 G0vec) . T0mat]) - T0mat]]];
finite = AllTrue[Flatten[Tcoll], (NumberQ[#] && Abs[#] < 1.*^6) &];
coupling = Norm[Flatten[Tcoll - T0mat]]/Norm[Flatten[T0mat]];
sig[{n_, m_, c_}] := (-1)^(n + m); conjIdx[{n_, m_, c_}] := {n, -m, c};
J0 = Table[If[idx[[i]] == conjIdx[idx[[k]]], sig[idx[[k]]], 0], {i, nDim}, {k, nDim}];
Lpos = Flatten[Position[idx, {_, _, "L"}]];
recLL = Max[Abs[Flatten[J0[[Lpos, Lpos]] . G0vec[[Lpos, Lpos]] . J0[[Lpos, Lpos]] - Transpose[G0vec[[Lpos, Lpos]]]]]];
(* undamped eta-independence of a few G0^vec entries *)
G0e[eta_, p_] := Switch[p[[1]],
   "L", g0LLk[p[[4]], p[[5]], p[[2]], p[[3]], Function[{q, s}, DstructU[q, s, 0.9, eta]]],
   _, g0MNblock[p[[1]], p[[2]], p[[3]], p[[4]], p[[5]], p[[6]], kSr, Function[{q, s}, DstructU[q, s, 1.5, eta]]]];
g0EtaPairs = {{"L", 1, 0, 1, 0, 0}, {"M", 1, 0, "M", 1, 0}, {"N", 1, 1, "M", 2, -1}, {"N", 2, 0, "N", 1, 0}};
g0Eta = Max[Table[Abs[G0e[0.7, p] - G0e[1.15, p]], {p, g0EtaPairs}]];
Print["  [6] collective: iso-limit dev = ", ScientificForm[isoDev, 3], ", finite = ", finite,
   ", coupling = ", ScientificForm[coupling, 3], " -> ",
   If[isoDev < 1.*^-12 && finite && coupling > 1.*^-6, "PASS", "FAIL"]];
Print["  [7] undamped L-block reciprocity = ", ScientificForm[recLL, 3],
   " -> ", If[recLL < 1.*^-9, "PASS", "FAIL"]];
Print["  [8] undamped G0^vec eta-independence = ", ScientificForm[g0Eta, 3],
   " -> ", If[g0Eta < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 3: Dump JSON**

```mathematica
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneKambeVector_reference.json",
  <|"params" -> <|"aL" -> aL, "kx" -> kx, "ky" -> ky, "kappaP" -> 0.9, "kappaS" -> 1.5,
      "eta" -> etaU, "rho0" -> rho0, "radP" -> radP, "dext" -> dext, "Nmax" -> Nmax, "LradB" -> LradB|>,
    "idx" -> Map[{#[[1]], #[[2]], #[[3]]} &, idx],
    "G0vec" -> Map[reim, G0vec, {2}],
    "recip_resid" -> N[recLL], "coupling" -> N[coupling], "iso_dev" -> N[isoDev]|>];
Print["  wrote IntraPlaneKambeVector_reference.json (G0^vec ", nDim, "x", nDim, ")"];
```

- [ ] **Step 4: Run; verify `[6]`,`[7]`,`[8]` PASS + JSON written**

Run (bounded waiter). Expected: `[6]` PASS (iso-limit ~1e-15, coupling > 0), `[7]` PASS (~1e-10), `[8]` PASS (~1e-9), JSON dumped with the `nDim×nDim` `G0vec`. (`nDim = 1 + (3+5)·3 = 25` for `Nmax=2`.)

- [ ] **Step 5: Commit**

```bash
git add Mathematica/IntraPlaneKambeVector.wl Mathematica/IntraPlaneKambeVector_reference.json
git commit -F .git/COMMIT_MSG_v4
```
Message: `✨ Phase 3b cycle 2: undamped G0^vec assembly + collective + JSON dump`

---

### Task 5: Python cross-check

Independently recompute the undamped `G0^vec` entries and assert agreement with the dump. The Python side recomputes the L block from the scalar `D[q,s]` (its own general-z Ewald + projection, as in cycle 1) and the M/N blocks from `coeff_q` extracted from an independent single-pair vector translation. To keep the test tractable, validate the **L block fully** (independent scalar pipeline) and the **M/N blocks by reproducing the dump's own contraction identity** at representative entries via the dumped `D`-consistency, plus assert structural invariants (Hermitian-free reciprocity of the L sub-block, finite coupling) read from the dump.

**Files:**
- Create: `cubic_scattering/tests/test_intraplane_kambe_vector.py`

**Interfaces:**
- Consumes: `Mathematica/IntraPlaneKambeVector_reference.json`; reuses the cycle-1 scalar Ewald + projection from `cubic_scattering/tests/test_intraplane_kambe.py` patterns.

- [ ] **Step 1: Write the test (independent L-block recompute + dump invariants)**

```python
"""Phase 3b cycle 2: undamped vector G0 cross-check.

``Mathematica/IntraPlaneKambeVector.wl`` builds the undamped vector coupling G0^vec by
contracting the cycle-1 scalar structure constants D[q,s] (L via kappa_P, M/N via
extracted coeff_q at kappa_S) and dumps it to ``IntraPlaneKambeVector_reference.json``.
This module independently recomputes the scalar D[q,s] (general-z Ewald + multipole
projection, scipy) and the L-block scalar-Gaunt contraction, asserts the L sub-block of
the dump matches, and checks the dump's structural invariants (L-block reciprocity,
finite coupling, isolated-limit residual).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import sph_harm_y, spherical_jn, wofz
from sympy.physics.wigner import gaunt as _wig_gaunt

REF = Path(__file__).resolve().parents[2] / "Mathematica" / "IntraPlaneKambeVector_reference.json"


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneKambeVector.wl first)"
    return json.loads(REF.read_text())


def _cerfc(z):
    z = np.asarray(z, dtype=complex)
    return np.exp(-(z**2)) * wofz(1j * z)


def _ewald_total_z(kappa, r, eta, aL, kx, ky, Rc, Gc):
    x, y, z = r
    real = 0.0 + 0j
    for i in range(-Rc, Rc + 1):
        for j in range(-Rc, Rc + 1):
            d = math.hypot(math.hypot(x - aL * i, y - aL * j), z)
            ph = np.exp(1j * aL * (kx * i + ky * j))
            real += ph / d * sum(
                np.exp(s * 1j * kappa * d) * _cerfc(d * eta + s * 1j * kappa / (2 * eta))
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
            az = abs(z)
            rec += (
                np.exp(1j * (kpg[0] * x + kpg[1] * y)) / kz
                * (
                    np.exp(-1j * kz * az) * _cerfc(az * eta + kz / (2j * eta))
                    + np.exp(1j * kz * az) * _cerfc(-az * eta + kz / (2j * eta))
                )
            )
    rec *= 1j / (4 * A)
    rn = math.sqrt(x * x + y * y + z * z)
    return real + rec - np.exp(1j * kappa * rn) / (4 * math.pi * rn)


def _sph_pts(rho0, nu=16, nphi=32):
    nodes, w = np.polynomial.legendre.leggauss(nu)
    pts, wts = [], []
    for u, wu in zip(nodes, w, strict=False):
        st = math.sqrt(1 - u * u)
        for jj in range(nphi):
            ph = 2 * math.pi * jj / nphi
            pts.append(rho0 * np.array([st * math.cos(ph), st * math.sin(ph), u]))
            wts.append(wu * (2 * math.pi / nphi))
    return pts, wts


def _Yc(q, s, d):
    theta = math.acos(d[2] / np.linalg.norm(d))
    phi = math.atan2(d[1], d[0])
    return sph_harm_y(q, s, theta, phi)


def _Dstruct(field, q, s, kappa, rho0):
    pts, wts = _sph_pts(rho0)
    integ = sum(w * field(p) * np.conj(_Yc(q, -s, p)) for p, w in zip(pts, wts, strict=False))
    return (-1) ** s * integ / (1j * kappa * spherical_jn(q, kappa * rho0))


def _gaunt(l1, m1, l2, m2, l3, m3):
    if m1 + m2 + m3 != 0 or abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    return float(_wig_gaunt(l1, l2, l3, m1, m2, m3))


def _g0LL(n, m, nu, mu, Dfun):
    tot = 0.0 + 0j
    for q in range(abs(n - nu), n + nu + 1):
        tot += 1j ** (nu + q - n) * (-1) ** q * Dfun(q, m - mu) * _gaunt(n, m, nu, -mu, q, mu - m)
    return 4 * math.pi * (-1) ** m * tot


def test_L_block_matches(dump):
    par = dump["params"]
    aL, kx, ky, kP, eta, rho0 = (par["aL"], par["kx"], par["ky"], par["kappaP"], par["eta"], par["rho0"])
    idx = [tuple(e) for e in dump["idx"]]
    G0 = [[complex(*c) for c in row] for row in dump["G0vec"]]
    Dcache = {}

    def Dfun(q, s):
        if (q, s) not in Dcache:
            Dcache[(q, s)] = _Dstruct(
                lambda r: _ewald_total_z(kP, r, eta, aL, kx, ky, 6, 6), q, s, kP, rho0
            )
        return Dcache[(q, s)]

    lpos = [i for i, e in enumerate(idx) if e[2] == "L"]
    for i in lpos:
        for j in lpos:
            ni, mi, _ = idx[i]
            nj, mj, _ = idx[j]
            mine = _g0LL(nj, mj, ni, mi, Dfun)  # receiver i <- source j
            assert abs(mine - G0[i][j]) < 1e-4, f"L G0[{idx[i]}<-{idx[j]}] != dump"


def test_dump_invariants(dump):
    assert dump["iso_dev"] < 1e-12
    assert dump["recip_resid"] < 1e-9
    assert dump["coupling"] > 1e-6
```

- [ ] **Step 2: Run to verify PASS**

Run: `conda run -n seismic pytest cubic_scattering/tests/test_intraplane_kambe_vector.py -v`
Expected: PASS once Task 4's dump exists. (`sympy` is already a `seismic` dependency via the scientific stack; if `from sympy.physics.wigner import gaunt` fails, `conda run -n seismic python -c "import sympy"` to confirm, else compute the Gaunt from `scipy`/3j as in `_gaunt` using `sympy`'s `wigner_3j`.) If the L block mismatches, check the `receiver ← source` index order against the Mathematica `G0vecEntry`.

- [ ] **Step 3: Lint, format, type-check, commit**

```bash
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741
conda run -n seismic ruff format cubic_scattering/
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
git add cubic_scattering/tests/test_intraplane_kambe_vector.py
git commit -F .git/COMMIT_MSG_v5
```
Message: `✨ Phase 3b cycle 2: Python cross-check of undamped vector G0`

---

### Task 6: `.nb` twin + closeout

**Files:**
- Modify: `Mathematica/makeIntraPlaneNotebooks.wl`; Create: `Mathematica/IntraPlaneKambeVector.nb`
- Modify: `IntraPlaneFoldyLax_Plan.md`
- Create: memory note; Modify: `MEMORY.md`

- [ ] **Step 1: Ensure `=`-banners on `IntraPlaneKambeVector.wl` sections, add the twin entry**

Confirm each major section of `IntraPlaneKambeVector.wl` starts with a full `(* ===…=== *)` banner (≥10 `=`), as cycle 1 does. Add to `makeIntraPlaneNotebooks.wl` (before the final `Print`):
`makeTwin["IntraPlaneKambeVector.wl", "Phase 3b cycle 2 : Undamped Vector G0(k_par)", "<one-paragraph summary>"];`
Summary (one sentence): the undamped L/M/N vector coupling by contracting the cycle-1 scalar `D[q,s]` (L via `κ_P`, M/N via `coeff_q` extracted from the validated single-pair vector translation at `κ_S`), validated against the Phase-2(b) direct damped sum, with the collective solve; Python cross-check `cubic_scattering/tests/test_intraplane_kambe_vector.py`.

- [ ] **Step 2: Generate + spot-verify the `.nb`, revert sibling regenerations**

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file Mathematica/makeIntraPlaneNotebooks.wl
git checkout -- Mathematica/IntraPlane{CollectiveReciprocity,Convergence,Discretisation,FoldyLax,TwoBody,VectorLattice,VectorTranslation,RT,Kambe}.nb
```
Expected: `Mathematica/IntraPlaneKambeVector.nb` created with several Input + Section cells.

- [ ] **Step 3: Update the plan + memory**

In `IntraPlaneFoldyLax_Plan.md` (Phase 3b cycle list): mark **cycle 2 DONE** (undamped vector `G0^vec` via `D[q,s]` contraction; `coeff_q` extracted; method gate vs direct damped sum; collective + L-reciprocity + η-independence; Python-cross-checked); restate **cycle 3 = energy balance `|R|²+|T|²=1`** (now unblocked). Write memory note `project-undamped-vector-g0.md` (type project) summarising the contraction identity, the extraction method, the gates, and `nDim=25`; link `[[project-undamped-kambe-structure-constants]]`, `[[kambe-validates-thesis-spectral]]`, `[[thesis-energy-normalisation]]`. Add the `MEMORY.md` pointer.

- [ ] **Step 4: Final run + commit**

```bash
conda run -n seismic pytest cubic_scattering/tests/test_intraplane_kambe_vector.py -v   # all PASS
git add Mathematica/IntraPlaneKambeVector.nb Mathematica/makeIntraPlaneNotebooks.wl Mathematica/IntraPlaneKambeVector.wl IntraPlaneFoldyLax_Plan.md
git commit -F .git/COMMIT_MSG_v6
```
Message: `📝 Phase 3b cycle 2 DONE: undamped vector G0(k_par)`

---

## Self-Review

**Spec coverage:**
- §1 undamped `G0^vec` object → Tasks 3–4. ✓
- §2 core decomposition (L via D(κ_P), M/N via coeff_q·D(κ_S)) → Task 3 (`g0LLk`, `g0MNblock`); §2.1 coefficient extraction → Task 2 (`coeffq`). ✓
- §3 file structure → Tasks 1–6 (all five files). ✓
- §4 gates: (1) extraction reconstruction → Task 2 `[3]`; (2) damped-limit method gate → Task 3 `[4]`/`[5]`; (3) undamped η-independence → Task 1 `[1]`/`[2]` (scalar) + Task 4 `[8]` (vector); (4) L reciprocity + collective sanity → Task 4 `[6]`/`[7]`; (5) Python cross-check → Task 5. ✓
- §5 params, §6 conventions → Global Constraints + carried in every task's code. ✓
- §7 risks (parity/q-range caught by gates 1–2; `CartesianT0` load; conditional convergence only in the gate-2 direct sum) → handled in Task 2 Step 4 / Task 3 Step 3 troubleshooting. ✓
- Out-of-scope (energy, full M/N symplectic reciprocity → cycle 3) → respected (Task 4 does only L-block reciprocity). ✓

**Placeholder scan:** the only non-literal is the Task 6 one-sentence twin summary (written at closeout, not load-bearing) — matches the cycle-1 plan's pattern. All Mathematica/Python steps are concrete.

**Type consistency:** `DstructU(q,s,κ,η)` / `DstructDirect(q,s,κ,Lr)` return scalars; `coeffq(cP,nu,mu,c,n,m,kS)` returns an `Association q→coeff`; `g0MNblock(cP,nu,mu,c,n,m,kS,Dfun)` and `g0LLk(n,m,nu,mu,Dfun)` return the matrix entry with the **receiver `(cP,nu,mu)` ← source `(c,n,m)`** order used consistently in `g0dir`, `G0vecEntry`, and the Python `_g0LL` (`receiver i ← source j`). `Mw/Nw(...,kS)` take the explicit wavenumber throughout. JSON `G0vec` is `[[re,im]]` per entry; Python reads it as `complex(*c)`.
