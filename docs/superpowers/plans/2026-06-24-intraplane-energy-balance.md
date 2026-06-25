# Intra-Plane Layer Energy Balance Implementation Plan (Phase 3b cycle 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the lossless energy balance of the Phase-3a layer R/T(p) — `S_p† S_p = I` on the propagating channels — built on the cycle-2 **undamped** vector coupling `G0^vec` (Im κ = 0), across normal / sub-critical / post-critical `p`.

**Architecture:** A new self-contained `IntraPlaneEnergyBalance.wl` lifts the cycle-2 undamped `G0^vec` builders to **physical parameters** (`aL = 2.5`, `κ_P = kPo = 0.3`, `κ_S = kSo = 0.5`, Bloch `k_x = ω·p`), feeds them to the collective solve `T_coll = T0 (I − G0^vec T0)^{-1}`, projects with the Phase-3a thesis-ε R/T machinery (reused verbatim), assembles the propagating-channel scattering matrix `S = [[Rd, Tu],[Td, Ru]]` (+ SH 2×2), and gates unitarity. A sub-wavelength precondition (no open diffraction orders) makes the `p → p` specular unitarity statement well-posed.

**Tech Stack:** Wolfram Language (`wolframscript`); Python 3.12 (`cubic_scattering`, numpy/scipy, pytest), conda env `seismic`; JSON as the MMA↔Python contract.

## Global Constraints

- Conda env `seismic`; Python tooling via `conda run -n seismic <cmd>`.
- Time `e^{−iωt}`, outgoing `h_q^(1)` via `SphericalBesselJ + I SphericalBesselY`; regular `j_q` via `SphericalBesselJ`; `Y_q^s` via `SphericalHarmonicY`. Lattice in x–y (`θ_R = π/2`); z = polar = depth (axis 0).
- **Physical params** (moderate contrast, match Phase-3a `IntraPlaneRT.wl`): background `α0 = 5000`, `β0 = 3000`, `ρ0 = 2500`; `aa = 1.0`; pitch `aLpitch = 2.5`, `A_cell = aLpitch²`; `kaTest = 0.3` ⇒ after `setMie`: `kPo = 0.3`, `kSo = 0.5`, `ω = ωOf = kPo·α0/aa = 1500`; contrast `Δλ = 2e9`, `Δμ = 1e9`, `Δρ = 100`.
- **Undamped κ** (the cycle-3 change vs Phase-3a's `+0.25 I`): `κ_P = kPo`, `κ_S = kSo`, both real. Ewald split `η ∈ {0.7, 1.15}`, lattice/recip half-widths `Rc = Gc = 6` (bump to 8 only if gate [2] η-independence fails); scalar projection radius `ρ0 = 0.5`; vector projection radius `radP = 0.5`; coeff-extraction `dext = 2.0`.
- **Sweep:** `pcritP = 1/α0 = 2e-4`, `pcritS = 1/β0 = 3.333e-4`; `pList = {pNormal = 1e-6, 0.5·pcritP, 0.8·pcritS}` (normal / sub-critical / post-critical-P-evanescent).
- **Energy metric (R1, resolved empirically in Task 2):** print BOTH `‖S†S − I‖` (plain unitary) and `‖S†ΣS − Σ‖` (SV-parity-twisted, `Σ` = +1 on P channels, −1 on SV channels) at `pNormal`; keep whichever is machine-zero in `energyMetric ∈ {"plain","sigma"}` and use it for gate [3] and the dump.
- Reference JSON in `Mathematica/`; complex numbers serialised as `[re, im]` via `reim[z_] := {Re[N[z]], Im[N[z]]}`.
- Line length ≤ 108. Ruff `--ignore ARG001,ARG002,F841,E741`, ruff format, mypy `--ignore-missing-imports`. B904 in except; `pathlib.Path`; Google docstrings.
- Lint/format/type after every Python change:
  `conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && conda run -n seismic ruff format cubic_scattering/ && conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports`
- scipy 1.17: use `sph_harm_y(n,m,θ_polar,φ_azimuth)` (NOT `sph_harm`); complex `erfc` via Faddeeva `wofz` (`erfc(z)=exp(−z²)·wofz(iz)`).
- NO Claude attribution in commits. NEVER write "ATO" (use "PROD"). No heredocs in the Bash tool — write commit messages to a file and `git commit -F /tmp/msg.txt`. Long `wolframscript` runs auto-background; wait via a bounded waiter; ONE kernel at a time.
- `IntraPlaneEnergyBalance.wl` is a NEW self-contained file: it copies the cycle-2 builders and the Phase-3a projection block, and `Get`s only `CartesianT0.wl` (for `T0mono`/`TsphClean`/`Ttoroidal`/`Yv`/`Bv`/`Cv`). `IntraPlaneRT.wl` and `IntraPlaneKambeVector.wl` are UNTOUCHED.
- **Cache hygiene (critical):** the Ewald field depends on the per-`p` globals `kx, ky`. Do NOT self-memoise the structure constant on `(q,s,κ,η)` alone — build a fresh `Dtab` Association per `p`. `coeff_q` is `(kx,ky)`-independent (pure Wigner; `κ_S` fixed across the sweep) → memoise it globally once.

## File Structure

- Create `Mathematica/IntraPlaneEnergyBalance.wl` — undamped `G0^vec` at physical params (gates [1],[2]); reused thesis-ε R/T projection; propagating-channel S-matrix + unitarity (gate [3]); reciprocity (gate [4]); `Nmax` sub-study (gate [5]); dumps `IntraPlaneEnergyBalance_reference.json`.
- Create `Mathematica/IntraPlaneEnergyBalance_reference.json` (generated).
- Create `cubic_scattering/tests/test_intraplane_energy.py` — Python cross-check.
- Modify `Mathematica/makeIntraPlaneNotebooks.wl` — add the cycle-3 `.nb` twin entry.
- Modify `IntraPlaneFoldyLax_Plan.md` — Phase 3b cycle 3 DONE; Phase 3b closed.
- Create memory note `project-intraplane-energy-balance.md`; update `MEMORY.md`.

---

### Task 1: Undamped `G0^vec` at physical parameters + gates [1], [2]

Scaffold the new file: copy the cycle-2 builders, lift them to physical `(aL, κ_P, κ_S)` and a per-`p` Bloch vector, and self-verify (a) the sub-wavelength **no-open-diffraction-orders** precondition and (b) Ewald η-independence at the physical κ.

**Files:**
- Create: `Mathematica/IntraPlaneEnergyBalance.wl`

**Interfaces:**
- Produces: `DfieldU[κ, η, q, s]` (non-memoised undamped structure constant at the current `kx,ky,aL`); `buildDtab[κ, η, qmax]` (per-`p` Association `{q,s} -> value`); `coeffq[cP,νμ,c,nm,kS]` (globally memoised); `g0LLk[n,m,ν,μ,Dfun]`, `g0MNblock[cP,ν,μ,c,n,m,kS,Dfun]`; `buildG0undamped[pval, Nmax]` (the per-`p` undamped `G0^vec` matrix); `idxVof[Nmax]`; constants `aLpitch, kPo, kSo, omegaOf, recipB`.
- Consumes: `CartesianT0.wl` globals (`T0mono, TsphClean, Ttoroidal, Yv, Bv, Cv`).

- [ ] **Step 1: Header, constants, κ-parameterized undamped scalar Ewald field**

Create `Mathematica/IntraPlaneEnergyBalance.wl` starting with:

```mathematica
#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneEnergyBalance.wl  --  Phase 3b cycle 3:
   lossless ENERGY BALANCE of the Phase-3a layer R/T(p).  Builds the cycle-2
   UNDAMPED vector G0^vec (Im kappa = 0) at PHYSICAL parameters, runs the
   thesis-eps R/T projection (reused from IntraPlaneRT.wl), assembles the
   propagating-channel scattering matrix S = [[Rd,Tu],[Td,Ru]] (+ SH 2x2), and
   gates unitarity S^dag S = I.  Sub-wavelength precondition (no open diffraction
   orders) makes the p->p specular statement well-posed.
   Self-contained: copies the cycle-2 undamped builders (IntraPlaneKambeVector.wl)
   and the Phase-3a projection (IntraPlaneRT.wl); Gets only CartesianT0.wl.
   Time e^{-i w t}, outgoing h^(1); lattice in x-y; project frame (z,x,y).
   ============================================================================ *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

reim[z_] := {Re[N[z]], Im[N[z]]};
alpha0 = 5000.; beta0 = 3000.; rho0Bg = 2500.;
lam0 = rho0Bg (alpha0^2 - 2 beta0^2); mu0 = rho0Bg beta0^2;
aa = 1.0; aLpitch = 2.5; aL = aLpitch; Acell = aLpitch^2;
Aarea = aL^2; recipB = 2 Pi/aL;
kaTest = 0.3; etaU = 0.7;
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];
kx = 0.; ky = 0.;                               (* set per-p in the sweep *)

(* ---- cycle-1 scalar Ewald field (general z, R!=0), kappa-parameterized; uses globals kx,ky,aL ---- *)
ewReal3[kappa_, r_, eta_, Rc_] := (1/(8 Pi)) Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      (Exp[I aL (kx i + ky j)]/d) Sum[Exp[s I kappa d] Erfc[d eta + s I kappa/(2 eta)], {s, {-1, 1}}]],
     {i, -Rc, Rc}, {j, -Rc, Rc}], 1]];
gKs[kappa_, rn_] := Exp[I kappa rn]/(4 Pi rn);
ewRecip3[kappa_, r_, eta_, Gc_] := (I/(4 Aarea)) Total[Flatten[Table[
     With[{kpg = {kx, ky} + recipB {m, n}, z = r[[3]]},
      With[{kz = Sqrt[kappa^2 - kpg . kpg]},
       (Exp[I kpg . {r[[1]], r[[2]]}]/kz) (
          Exp[-I kz Abs[z]] Erfc[ Abs[z] eta + kz/(2 I eta)]
        + Exp[ I kz Abs[z]] Erfc[-Abs[z] eta + kz/(2 I eta)])]],
     {m, -Gc, Gc}, {n, -Gc, Gc}], 1]];
ewTot3[kappa_, r_, eta_, Rc_, Gc_] :=
  ewReal3[kappa, r, eta, Rc] + ewRecip3[kappa, r, eta, Gc] - gKs[kappa, Sqrt[r . r]];
```

- [ ] **Step 2: Non-memoised structure-constant projection + per-`p` `Dtab`**

Append (the key cache-hygiene change vs cycle 2: NO self-memo on `DstructU`; build a fresh `Dtab` per `p`):

```mathematica
Needs["NumericalDifferentialEquationAnalysis`"];
glD = GaussianQuadratureWeights[16, -1, 1]; nPhiD = 32; rho0Proj = 0.5;
sphPts = Flatten[Table[Module[{u = glD[[i, 1]], ph = 2 Pi (j - 1)/nPhiD, st = Sqrt[1 - glD[[i, 1]]^2]},
     {rho0Proj {st Cos[ph], st Sin[ph], u}, glD[[i, 2]] (2 Pi/nPhiD)}], {i, Length[glD]}, {j, nPhiD}], 1];
Yf[q_, s_, d_] := SphericalHarmonicY[q, s, ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]];
(* NON-memoised: depends on the per-p globals kx,ky via ewTot3 *)
DfieldU[kappa_, eta_, q_, s_] := Module[{integ},
  integ = Total[Map[#[[2]] ewTot3[kappa, #[[1]], eta, 6, 6] Conjugate[Yf[q, -s, #[[1]]]] &, sphPts]];
  (-1)^s integ/(I kappa sj[q, kappa rho0Proj])];
buildDtab[kappa_, eta_, qmax_] := Association[
   Flatten[Table[{q, s} -> DfieldU[kappa, eta, q, s], {q, 0, qmax}, {s, -q, q}], 1]];
```

- [ ] **Step 3: Copy the cycle-2 vector helpers + `coeff_q` extraction + block builders**

Append, **verbatim from `Mathematica/IntraPlaneKambeVector.wl` lines 69–151**, this block (vector wavefunctions, quadrature, single-pair `Wel`, `coeffq`, `g0LLk`, `g0MNblock`). Reproduced here for completeness:

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
srcQuad[c_, n_, m_, d_, kS_] := srcQuad[c, n, m, d, kS] =
   Map[Switch[c, "M", Mw[n, m, "h", d, radP #, kS], "N", Nw[n, m, "h", d, radP #, kS]] &, quadDirs];
Wel[cP_, nu_, mu_, c_, n_, m_, d_, kS_] := Module[{f = srcQuad[c, n, m, d, kS]},
   Switch[cP, "M", projDotF[f, Cvals[nu, mu]]/normMC[nu, mu, kS],
              "N", projDotF[f, Pvals[nu, mu]]/normNP[nu, mu, kS]]];
glS = GaussianQuadratureWeights[8, -1, 1]; nPhiS = 16; dext = 2.0;
srcW = Flatten[Table[{shat[glS[[i, 1]], 2 Pi (j - 1)/nPhiS], glS[[i, 2]] (2 Pi/nPhiS)},
     {i, Length[glS]}, {j, nPhiS}], 1];
qset[n_, nu_] := Range[Abs[n - nu], n + nu];
coeffq[cP_, nu_, mu_, c_, n_, m_, kS_] := coeffq[cP, nu, mu, c, n, m, kS] = Association[Table[
    q -> Total[Map[#[[2]] Wel[cP, nu, mu, c, n, m, dext #[[1]], kS]
          Conjugate[SphericalHarmonicY[q, m - mu, ang[#[[1]]][[1]], ang[#[[1]]][[2]]]] &, srcW]]
       / sh[q, kS dext],
    {q, qset[n, nu]}]];
g0LLk[n_, m_, nu_, mu_, Dfun_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dfun[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
g0MNblock[cP_, nu_, mu_, c_, n_, m_, kS_, Dfun_] := Module[{cf = coeffq[cP, nu, mu, c, n, m, kS]},
   Total[KeyValueMap[#2 Dfun[#1, m - mu] &, cf]]];
```

- [ ] **Step 4: `setMie`, `idxVof`, and the per-`p` undamped `G0^vec` assembly**

Append. `setMie` and `T0LMN`/`idxVof`/`chPos` are copied from `IntraPlaneRT.wl` lines 35–49; `omegaOf` from line 39:

```mathematica
setMie[ka_, Dl_, Dm_, Dr_] := Module[{rhoI = rho0Bg + Dr, lamI0 = lam0 + Dl, muI0 = mu0 + Dm, alphaI, betaI},
  alphaI = Sqrt[(lamI0 + 2 muI0)/rhoI]; betaI = Sqrt[muI0/rhoI];
  kPo = ka; kSo = ka alpha0/beta0; kPi = ka alpha0/alphaI; kSi = ka alpha0/betaI;
  lamO = lam0; muO = mu0; lamI = lamI0; muI = muI0;];
omegaOf := kPo alpha0/aa;
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
idxVof[Nmax_] := Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
T0vec[Nmax_] := Module[{idx = idxVof[Nmax], T0e},
  T0e[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
    If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
  Table[T0e[idx[[i]], idx[[j]]], {i, Length[idx]}, {j, Length[idx]}]];
collV[G0_, T0_] := T0 . Inverse[IdentityMatrix[Length[T0]] - G0 . T0];

(* per-p undamped G0^vec: set Bloch globals, build fresh Dtab (cache hygiene), contract *)
buildG0undamped[pval_, Nmax_] := Module[{idx = idxVof[Nmax], nD, DtabP, DtabS, DuP, DuS, Gentry},
  kx = omegaOf pval; ky = 0.;
  DtabP = buildDtab[kPo, etaU, 2 Nmax]; DtabS = buildDtab[kSo, etaU, 2 Nmax];
  DuP = Function[{q, s}, DtabP[{q, s}]]; DuS = Function[{q, s}, DtabS[{q, s}]];
  nD = Length[idx];
  Gentry[{nu_, mu_, ct_}, {n_, m_, cs_}] := Which[
    ct == "L" && cs == "L", g0LLk[n, m, nu, mu, DuP],
    (ct == "M" || ct == "N") && (cs == "M" || cs == "N"), g0MNblock[ct, nu, mu, cs, n, m, kSo, DuS],
    True, 0];
  Table[Gentry[idx[[i]], idx[[j]]], {i, nD}, {j, nD}]];
```

- [ ] **Step 5: Gate [1] (no open diffraction orders) + gate [2] (η-independence at physical κ)**

Append:

```mathematica
Print["==== Phase 3b cycle 3 :: layer energy balance (undamped G0^vec) ===="];
setMie[kaTest, 2.*^9, 1.*^9, 100.];
pcritP = 1./alpha0; pcritS = 1./beta0;
pList = {1.*^-6, 0.5 pcritP, 0.8 pcritS};
regimeOf[p_] := Which[p <= 1.*^-5, "normal", p < pcritP, "subcritical", True, "postcritical"];

(* [1] sub-wavelength precondition: every reciprocal G!=0 evanescent at the larger kappa (kSo).
   margin = min_{G!=0}|k_par+G| - kSo > 0  ==>  only the specular G=0 order propagates. *)
gShells = Flatten[Table[If[m == 0 && nn == 0, Nothing, {m, nn}], {m, -3, 3}, {nn, -3, 3}], 1];
openMargin[p_] := Min[Map[Norm[{omegaOf p, 0.} + recipB #] &, gShells]] - kSo;
diffMargin = Min[Map[openMargin, pList]];
Print["  [1] no open diffraction orders: min margin = ", ScientificForm[diffMargin, 3],
   " -> ", If[diffMargin > 0., "PASS", "FAIL"]];
If[diffMargin <= 0.,
   Print["  ABORT: open diffraction order -> specular 4x4 unitarity is the WRONG statement."]; Abort[]];

(* [2] undamped eta-independence at physical kappa (kPo=0.3, kSo=0.5), fresh kx,ky from pNormal *)
kx = omegaOf (1.*^-6); ky = 0.;
etaIndepP = Max[Table[Abs[DfieldU[kPo, 0.7, q, s] - DfieldU[kPo, 1.15, q, s]], {q, 0, 4}, {s, -q, q}]];
etaIndepS = Max[Table[Abs[DfieldU[kSo, 0.7, q, s] - DfieldU[kSo, 1.15, q, s]], {q, 0, 4}, {s, -q, q}]];
etaIndep = Max[etaIndepP, etaIndepS];
Print["  [2] undamped eta-independence (kP=", kPo, ", kS=", kSo, ") = ", ScientificForm[etaIndep, 3],
   " -> ", If[etaIndep < 1.*^-6, "PASS", "FAIL"]];
```

- [ ] **Step 6: Run the script; confirm gates [1] and [2] PASS**

Run (auto-backgrounds; wait with a bounded loop, ONE kernel):

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
  /Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneEnergyBalance.wl
```

Expected: `[1] ... PASS` (margin ≈ 1.6, positive) and `[2] ... PASS` (η-independence < 1e-6, like cycle-1/2 ~1e-11). If [2] FAILs, bump `Rc = Gc` from 6 to 8 in `DfieldU` and re-run. Do NOT proceed until both PASS.

- [ ] **Step 7: Commit**

```bash
printf '%s\n' "✨ Phase 3b cycle 3 Task 1: undamped G0^vec at physical params + gates [1] no-diffraction, [2] eta-independence" > /tmp/msg_c3t1.txt
git add Mathematica/IntraPlaneEnergyBalance.wl
git commit -F /tmp/msg_c3t1.txt
```

---

### Task 2: Thesis-ε R/T projection + S-matrix assembly + resolve the energy metric (R1)

Copy the Phase-3a projection verbatim, assemble the propagating-channel S-matrix at `pNormal`, and settle whether energy is plain `S†S=I` or `Σ`-twisted (gate-[3] preliminary at one `p`).

**Files:**
- Modify: `Mathematica/IntraPlaneEnergyBalance.wl`

**Interfaces:**
- Consumes: `buildG0undamped`, `T0vec`, `collV`, `idxVof`, `omegaOf`, `kPo`, `kSo` (Task 1).
- Produces: `rtBlocksE[Tcoll, p, Nmax]` (assoc with `Rd,Td,Ru,Tu` 2×2 P-SV blocks + `Rsh,Tsh,Rsh_u,Tsh_u` scalars); `propModes[p]`; `assembleS[e, modes]`; `sigMetric[modes]`; `energyResid[e, modes, metric]`; the resolved global `energyMetric`.

- [ ] **Step 1: Copy the Phase-3a slowness geometry + thesis-ε projection**

Append, **verbatim from `Mathematica/IntraPlaneRT.wl` lines 124–193** (the `toSph`/`etaOf`/`khatPhys` geometry, `KhatSf`/`epsP,S,H`/`ehatTh`/`muTh`, `incPac/incNac/incMac`/`incVec`, `projPW`, `rtAmpE`, `rtBlocksE`). Reproduced here for completeness:

```mathematica
toSph[v_] := {v[[2]], v[[3]], v[[1]]};
etaOf[c_, p_] := Module[{e = Sqrt[1./c^2 - p^2]}, If[Im[e] < 0, -e, e]];
khatPhys[m_, p_, sign_] := Module[{c = If[m == "P", alpha0, beta0]}, c {sign etaOf[c, p], p, 0.}];
KhatSf := omegaOf/beta0;
epsP[p_] := 1/Sqrt[2 rho0Bg omegaOf^2 (omegaOf etaOf[alpha0, p])];
epsS[p_] := omegaOf/(beta0 KhatSf Sqrt[2 rho0Bg omegaOf^2 (omegaOf etaOf[beta0, p])]);
epsH[p_] := 1/(KhatSf Sqrt[2 rho0Bg omegaOf^2 (omegaOf etaOf[beta0, p])]);
ehatTh[m_, p_, sgn_] := Switch[m,
   "P", alpha0 {sgn etaOf[alpha0, p], p, 0.},
   "SV", beta0 {p, -sgn etaOf[beta0, p], 0.},
   "SH", {0., 0., 1.}];
muTh[m_, p_, sgn_] := Switch[m,
   "P", epsP[p] I omegaOf/alpha0,
   "SV", epsS[p] I omegaOf/beta0,
   "SH", epsH[p] KhatSf^2];
incPac[n_, m_, k_] := 4 Pi I^(n - 1) (-1)^m Yv[n, -m, k];
incNac[n_, m_, k_, e_] := -4 Pi I^(n + 1)/(n (n + 1)) (-1)^m (e . Bv[n, -m, k]);
incMac[n_, m_, k_, e_] := -4 Pi I^n/(n (n + 1)) (-1)^m (e . Cv[n, -m, k]);
incVec[mode_, khatSph_, ehatSph_, Nmax_] := Map[
   Function[idx, Module[{n = idx[[1]], m = idx[[2]], ch = idx[[3]]},
     Switch[{mode, ch}, {"P", "L"}, incPac[n, m, khatSph], {"SV", "N"}, incNac[n, m, khatSph, ehatSph],
       {"SH", "M"}, incMac[n, m, khatSph, ehatSph], _, 0]]], idxVof[Nmax]];
projPW[bvec_, ksSph_, Nmax_] := Module[{fP = {0, 0, 0}, fS = {0, 0, 0}, idx = idxVof[Nmax]},
  Do[With[{n = idx[[i, 1]], m = idx[[i, 2]], ch = idx[[i, 3]], b = bvec[[i]]},
    Switch[ch,
      "L", fP += b ((-I)^n/kPo) Yv[n, m, ksSph] ksSph,
      "N", fS += b ((-I)^n/kSo) Bv[n, m, ksSph],
      "M", fS += -b ((-I)^(n + 1)/kSo) Cv[n, m, ksSph]]], {i, Length[idx]}];
  {fP, fS}];
rtAmpE[Tcoll_, inMode_, p_, inSign_, outMode_, outSign_, Nmax_] := Module[
   {kin = toSph[khatPhys[inMode, p, inSign]], kout = toSph[khatPhys[outMode, p, outSign]],
    ein = toSph[ehatTh[inMode, p, inSign]], eout = toSph[ehatTh[outMode, p, outSign]],
    avec, bvec, f, etaOut = etaOf[If[outMode == "P", alpha0, beta0], p]},
  avec = incVec[inMode, kin, ein, Nmax];
  bvec = Tcoll . avec;
  f = projPW[bvec, kout, Nmax];
  (I/(2 etaOut omegaOf Acell)) (muTh[inMode, p, inSign]/muTh[outMode, p, outSign]) *
    (eout . If[outMode == "P", First[f], Last[f]])];
```

- [ ] **Step 2: `rtBlocksE` extended with up-incident SH (full SH 2×2)**

Append (the Phase-3a `rtBlocksE` from `IntraPlaneRT.wl` lines 189–193, **with two extra SH keys** for the up-incident SH so the SH S-matrix can be built):

```mathematica
rtBlocksE[Tcoll_, p_, Nmax_] := Module[{modes = {"P", "SV"}, blk},
  blk[inSign_, outSign_] := Table[rtAmpE[Tcoll, im, p, inSign, om, outSign, Nmax], {om, modes}, {im, modes}];
  <|"Rd" -> blk[1., -1.], "Td" -> blk[1., 1.], "Ru" -> blk[-1., 1.], "Tu" -> blk[-1., -1.],
    "Rsh" -> rtAmpE[Tcoll, "SH", p, 1., "SH", -1., Nmax],   (* down-in, up-out  *)
    "Tsh" -> rtAmpE[Tcoll, "SH", p, 1., "SH", 1., Nmax],    (* down-in, down-out *)
    "Rsh_u" -> rtAmpE[Tcoll, "SH", p, -1., "SH", 1., Nmax], (* up-in, down-out *)
    "Tsh_u" -> rtAmpE[Tcoll, "SH", p, -1., "SH", -1., Nmax]|>]; (* up-in, up-out *)
```

- [ ] **Step 3: Propagating-mode selection, S assembly, both energy metrics**

Append:

**BALLISTIC IDENTITY (correction, 2026-06-24).** `rtAmpE` returns only the *scattered* plane-wave amplitude. The physical transmission also carries the **unscattered (ballistic) incident wave**, which in the ε-normalised flux basis is the identity: `T_full = I_mode + T_scattered` on the forward-transmission blocks (`Td`, `Tu`, and the SH transmission entries); reflections have no ballistic term. Without this, `S ≈ 0` for a weak scatterer and `‖S†S − I‖ ≈ 1`. The identity preserves Phase-3a reciprocity (`Σ·I·Σ = I`, so `Tu = Σ·Td·Σ` still holds, and it never enters `Rd,Ru`).

```mathematica
modePos = <|"P" -> 1, "SV" -> 2|>;
propModes[p_] := Select[{"P", "SV"}, Abs[Im[etaOf[If[# == "P", alpha0, beta0], p]]] < 1.*^-9 &];
(* P-SV S-matrix over a mode subset: rows=(up-modes, down-modes), cols=(down-modes, up-modes).
   Forward-transmission blocks Td,Tu carry the ballistic identity (unscattered incident wave). *)
assembleS[e_, modes_] := Module[{mi = modePos /@ modes, idm, Rd, Td, Ru, Tu},
  idm = IdentityMatrix[Length[modes]];
  Rd = e["Rd"][[mi, mi]]; Td = e["Td"][[mi, mi]] + idm;
  Ru = e["Ru"][[mi, mi]]; Tu = e["Tu"][[mi, mi]] + idm;
  ArrayFlatten[{{Rd, Tu}, {Td, Ru}}]];
(* Sig metric: +1 on P channels, -1 on SV channels, repeated for up/down *)
sigMetric[modes_] := DiagonalMatrix[Flatten[Table[Map[If[# == "SV", -1., 1.] &, modes], {2}]]];
unitResid[S_] := Max[Abs[Flatten[ConjugateTranspose[S] . S - IdentityMatrix[Length[S]]]]];
sigResid[S_, M_] := Max[Abs[Flatten[ConjugateTranspose[S] . M . S - M]]];
energyResid[e_, modes_, metric_] := Module[{S = assembleS[e, modes]},
  If[metric === "sigma", sigResid[S, sigMetric[modes]], unitResid[S]]];
(* SH 2x2 with ballistic identity on the transmission entries [1,2]=Tsh_u, [2,1]=Tsh_d *)
shS[e_] := {{e["Rsh"], e["Tsh_u"] + 1.}, {e["Tsh"] + 1., e["Rsh_u"]}};
shEnergyResid[e_] := Module[{s = shS[e]}, Abs[Abs[s[[1, 1]]]^2 + Abs[s[[2, 1]]]^2 - 1.]];
```

- [ ] **Step 4: Resolve the energy metric (R1) at `pNormal`**

Append — build `T_coll` at `pNormal`, print BOTH metric residuals, set `energyMetric`:

```mathematica
Nm = 2;
G0n = buildG0undamped[1.*^-6, Nm]; Tcn = collV[G0n, T0vec[Nm]];
en = rtBlocksE[Tcn, 1.*^-6, Nm]; mds = propModes[1.*^-6];
Sn = assembleS[en, mds];
residPlain = unitResid[Sn]; residSigma = sigResid[Sn, sigMetric[mds]];
residSH = shEnergyResid[en];
energyMetric = If[residPlain <= residSigma, "plain", "sigma"];
Print["  R1 metric @pNormal: |S^dag S - I| = ", ScientificForm[residPlain, 3],
   ", |S^dag Sig S - Sig| = ", ScientificForm[residSigma, 3],
   " -> energyMetric = ", energyMetric];
Print["  SH @pNormal: |Rsh|^2+|Tsh|^2-1 = ", ScientificForm[residSH, 3]];
```

- [ ] **Step 5: Run; confirm one metric is machine-small and record it**

Run the script (as Task 1 Step 6). Expected: one of `residPlain`/`residSigma` is small (the energy floor — likely larger than 1e-6 because the undamped Ewald + quadrature compound; e.g. 1e-3…1e-5), the other O(1); `energyMetric` set accordingly; `residSH` at the same floor. Record the achieved floor — it sets the gate-[3] tolerance in Task 3 (R3). If BOTH are O(1), STOP: the projection/normalization is wrong, debug before proceeding (do not loosen the tolerance to mask it).

- [ ] **Step 6: Commit**

```bash
printf '%s\n' "✨ Phase 3b cycle 3 Task 2: thesis-eps R/T projection + S-matrix; resolve energy metric (R1) at pNormal" > /tmp/msg_c3t2.txt
git add Mathematica/IntraPlaneEnergyBalance.wl
git commit -F /tmp/msg_c3t2.txt
```

---

### Task 3: p-sweep with propagating restriction + gates [3], [4], [5]; JSON dump

Run the full sweep, gate unitarity (restricted) and reciprocity at every `p`, run the `Nmax` convergence sub-study, and dump the reference JSON.

**Files:**
- Modify: `Mathematica/IntraPlaneEnergyBalance.wl`
- Create: `Mathematica/IntraPlaneEnergyBalance_reference.json`

**Interfaces:**
- Consumes: everything from Tasks 1–2, plus `energyMetric` (resolved in Task 2 Step 4; recomputed in this run before the sweep).
- Produces: `IntraPlaneEnergyBalance_reference.json` with per-`p` S-matrices, channel sets, energy + reciprocity residuals, the `Nmax` study, and `energyMetric`.

- [ ] **Step 1: The p-sweep (energy gate [3] + reciprocity gate [4])**

Append (note: `energyMetric` is set by the Task-2 Step-4 block, which runs earlier in the same kernel):

```mathematica
Sig2 = DiagonalMatrix[{1., -1.}];
stageEB = {}; energyResids = {}; recipResids = {};
Do[Module[{G0, Tc, e, modes, Spsv, eR, shResid, antiRd, antiRu, tparity},
   G0 = buildG0undamped[p, Nm]; Tc = collV[G0, T0vec[Nm]];
   e = rtBlocksE[Tc, p, Nm]; modes = propModes[p];
   Spsv = assembleS[e, modes];
   eR = energyResid[e, modes, energyMetric];
   shResid = shEnergyResid[e];
   (* reciprocity (Phase-3a symplectic): only on the propagating P-SV sub-block *)
   antiRd = If[Length[modes] == 2, Max[Abs[e["Rd"][[1, 2]] + e["Rd"][[2, 1]]]], 0.];
   antiRu = If[Length[modes] == 2, Max[Abs[e["Ru"][[1, 2]] + e["Ru"][[2, 1]]]], 0.];
   tparity = If[Length[modes] == 2, Max[Abs[Flatten[e["Tu"] - Sig2 . e["Td"] . Sig2]]], 0.];
   AppendTo[energyResids, eR]; AppendTo[recipResids, Max[antiRd, antiRu, tparity]];
   AppendTo[stageEB, <|"p" -> p, "regime" -> regimeOf[p], "propModes" -> modes,
     "etaP" -> reim[etaOf[alpha0, p]], "etaS" -> reim[etaOf[beta0, p]],
     "S_psv" -> Map[reim, Spsv, {2}], "S_sh" -> Map[reim, shS[e], {2}],
     "energy_resid" -> N[eR], "sh_energy_resid" -> N[shResid],
     "recip_Rd_anti" -> N[antiRd], "recip_Ru_anti" -> N[antiRu], "recip_T_parity" -> N[tparity]|>];
   Print["  p=", ScientificForm[p, 3], " (", regimeOf[p], ", modes=", modes, "): energy(",
     energyMetric, ")=", ScientificForm[eR, 3], ", SH=", ScientificForm[shResid, 3],
     ", recip=", ScientificForm[Max[antiRd, antiRu, tparity], 3]]],
   {p, pList}];
enTol = 1.*^-3;   (* gate-[3] tolerance: set from the Task-2 observed floor (R3) *)
energyOK = Max[energyResids] < enTol;
recipOK = Max[recipResids] < 1.*^-6;
Print["  [3] energy unitarity (", energyMetric, ", restricted) max = ",
   ScientificForm[Max[energyResids], 3], " -> ", If[energyOK, "PASS", "FAIL"]];
Print["  [4] reciprocity preserved (undamped) max = ", ScientificForm[Max[recipResids], 3],
   " -> ", If[recipOK, "PASS", "FAIL"]];
```

- [ ] **Step 2: Nmax convergence sub-study (gate [5])**

Append — energy residual at the sub-critical `p` (fully propagating, most stringent) for `Nmax = 1,2,3`:

```mathematica
pStudy = 0.5 pcritP;
nmaxStudy = Table[Module[{G0, Tc, e, modes},
    G0 = buildG0undamped[pStudy, nmx]; Tc = collV[G0, T0vec[nmx]];
    e = rtBlocksE[Tc, pStudy, nmx]; modes = propModes[pStudy];
    {nmx, N[energyResid[e, modes, energyMetric]]}], {nmx, 1, 3}];
Print["  [5] Nmax convergence (energy resid @subcritical): ", nmaxStudy];
```

- [ ] **Step 3: JSON dump**

Append:

```mathematica
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneEnergyBalance_reference.json",
  <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho0" -> rho0Bg, "aa" -> aa,
      "aLpitch" -> aLpitch, "kPo" -> kPo, "kSo" -> kSo, "Nmax" -> Nm, "etaU" -> etaU,
      "energyMetric" -> energyMetric, "enTol" -> enTol,
      "contrast" -> <|"Dlambda" -> 2.*^9, "Dmu" -> 1.*^9, "Drho" -> 100.|>|>,
    "diffMargin" -> N[diffMargin], "etaIndep" -> N[etaIndep],
    "stageEB" -> stageEB, "nmaxStudy" -> nmaxStudy|>];
Print["  wrote IntraPlaneEnergyBalance_reference.json"];
Print["Phase 3b cycle 3 (IntraPlaneEnergyBalance.wl) loaded."];
```

- [ ] **Step 4: Run; confirm gates [3], [4], [5]**

Run the script (as Task 1 Step 6). Expected: `[3] ... PASS` (energy residual below `enTol` at all `p`, including SV-only at post-critical), `[4] ... PASS` (reciprocity preserved, ~1e-6 or below), `[5]` prints a residual sequence — confirm it **does not blow up** with `Nmax` (plateau at `Nmax=2` is the expectation; if `Nmax=2` ≫ `Nmax=3`, the production `Nmax` must rise and the spec's open question is answered "higher needed"). If `enTol = 1e-3` is below the observed floor from Task 2, set `enTol` to the achieved floor (R3) and document it; do NOT relax it to hide a real defect. JSON file written.

- [ ] **Step 5: Commit**

```bash
printf '%s\n' "✅ Phase 3b cycle 3 Task 3: p-sweep energy unitarity (gate 3), reciprocity (gate 4), Nmax study (gate 5) + JSON dump" > /tmp/msg_c3t3.txt
git add Mathematica/IntraPlaneEnergyBalance.wl Mathematica/IntraPlaneEnergyBalance_reference.json
git commit -F /tmp/msg_c3t3.txt
```

---

### Task 4: Python cross-check

Independent reload of the dumped S-matrices: recompute the unitarity/parity invariants, the SH 2×2 energy, the reciprocity residuals, and the diffraction-order margin (independent of Mathematica).

**Files:**
- Create: `cubic_scattering/tests/test_intraplane_energy.py`

**Interfaces:**
- Consumes: `Mathematica/IntraPlaneEnergyBalance_reference.json`.

- [ ] **Step 1: Write the failing test**

Create `cubic_scattering/tests/test_intraplane_energy.py`:

```python
"""Phase 3b cycle 3: layer energy-balance cross-check.

``Mathematica/IntraPlaneEnergyBalance.wl`` builds the undamped vector coupling G0^vec
at physical parameters, projects it to the layer R/T(p), assembles the propagating
scattering matrix S, and dumps ``IntraPlaneEnergyBalance_reference.json``. This module
independently reloads the dumped S-matrices and re-verifies unitarity (S^dag S = I, or
the Sigma-twisted invariant), the SH 2x2 energy, the symplectic reciprocity residuals,
and the sub-wavelength no-open-diffraction-order margin.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

REF = (
    Path(__file__).resolve().parents[2]
    / "Mathematica"
    / "IntraPlaneEnergyBalance_reference.json"
)


@pytest.fixture(scope="module")
def dump():
    assert REF.exists(), f"missing {REF} (run IntraPlaneEnergyBalance.wl first)"
    return json.loads(REF.read_text())


def _cplx(reim_mat):
    """Map a nested [re, im] dump array to a complex numpy array."""
    a = np.asarray(reim_mat, dtype=float)
    return a[..., 0] + 1j * a[..., 1]


def _sig_metric(modes):
    diag = [(-1.0 if m == "SV" else 1.0) for m in modes] * 2
    return np.diag(diag).astype(complex)


def test_energy_unitarity_each_p(dump):
    """S^dag M S == M on the propagating sub-block at every p (M per energyMetric)."""
    metric = dump["params"]["energyMetric"]
    tol = dump["params"]["enTol"]
    worst = 0.0
    for st in dump["stageEB"]:
        s = _cplx(st["S_psv"])
        m = _sig_metric(st["propModes"]) if metric == "sigma" else np.eye(s.shape[0], dtype=complex)
        resid = np.max(np.abs(s.conj().T @ m @ s - m))
        worst = max(worst, resid)
    assert worst < tol, f"energy ({metric}) residual {worst:.3e} >= tol {tol:.3e}"


def test_sh_energy_each_p(dump):
    """|Rsh|^2 + |Tsh|^2 == 1 (SH down-incident) at every p."""
    tol = dump["params"]["enTol"]
    for st in dump["stageEB"]:
        ssh = _cplx(st["S_sh"])  # [[Rsh_d, Tsh_u], [Tsh_d, Rsh_u]]
        rsh_d, tsh_d = ssh[0, 0], ssh[1, 0]
        assert abs(abs(rsh_d) ** 2 + abs(tsh_d) ** 2 - 1.0) < tol


def test_reciprocity_preserved(dump):
    """Undamped build keeps Phase-3a symplectic reciprocity (Rd,Ru antisym; reported residuals)."""
    for st in dump["stageEB"]:
        assert st["recip_Rd_anti"] < 1e-6
        assert st["recip_Ru_anti"] < 1e-6
        assert st["recip_T_parity"] < 1e-6


def test_no_open_diffraction_orders(dump):
    """Independently recompute the sub-wavelength margin: min_{G!=0}|k_par+G| - kSo > 0."""
    pr = dump["params"]
    omega = pr["kPo"] * pr["alpha"] / pr["aa"]
    recip_b = 2 * math.pi / pr["aLpitch"]
    shells = [
        (m, n) for m in range(-3, 4) for n in range(-3, 4) if not (m == 0 and n == 0)
    ]
    worst_margin = math.inf
    for st in dump["stageEB"]:
        kpar = np.array([omega * st["p"], 0.0])
        margin = min(
            np.linalg.norm(kpar + recip_b * np.array([m, n])) for m, n in shells
        ) - pr["kSo"]
        worst_margin = min(worst_margin, margin)
    assert worst_margin > 0.0, f"open diffraction order: margin {worst_margin:.3e}"
    assert abs(worst_margin - dump["diffMargin"]) < 1e-6


def test_nmax_does_not_diverge(dump):
    """Energy residual must not blow up with Nmax (convergence, not divergence)."""
    study = {int(nmx): r for nmx, r in dump["nmaxStudy"]}
    assert study[3] <= 10.0 * study[2], "energy residual diverges with Nmax"
```

- [ ] **Step 2: Run; verify it fails for the right reason (if dump absent) then passes**

```bash
conda run -n seismic python -m pytest cubic_scattering/tests/test_intraplane_energy.py -v
```

Expected: all 5 tests PASS against the Task-3 dump. (If the dump were missing, the fixture assert fails with a clear "run IntraPlaneEnergyBalance.wl first" message — the intended failure mode.)

- [ ] **Step 3: Lint/format/type**

```bash
conda run -n seismic ruff check cubic_scattering/ --fix --ignore ARG001,ARG002,F841,E741 && \
conda run -n seismic ruff format cubic_scattering/ && \
conda run -n seismic mypy cubic_scattering/ --ignore-missing-imports
```

Expected: clean (no errors).

- [ ] **Step 4: Commit**

```bash
printf '%s\n' "✅ Phase 3b cycle 3 Task 4: Python cross-check of dumped S-matrix unitarity, SH energy, reciprocity, diffraction margin" > /tmp/msg_c3t4.txt
git add cubic_scattering/tests/test_intraplane_energy.py
git commit -F /tmp/msg_c3t4.txt
```

---

### Task 5: `.nb` twin + plan/memory closeout

Add the deliverable notebook entry, mark the plan done, and record the memory.

**Files:**
- Modify: `Mathematica/makeIntraPlaneNotebooks.wl`
- Modify: `IntraPlaneFoldyLax_Plan.md`
- Create: `/Users/tod/.claude/projects/-Users-tod-Desktop-MultipleScatteringCalculations/memory/project-intraplane-energy-balance.md`
- Modify: `/Users/tod/.claude/projects/-Users-tod-Desktop-MultipleScatteringCalculations/memory/MEMORY.md`

**Interfaces:**
- Consumes: the committed `IntraPlaneEnergyBalance.wl`.

- [ ] **Step 1: Add the `.nb` twin entry**

In `Mathematica/makeIntraPlaneNotebooks.wl`, after the `IntraPlaneKambeVector.wl` `makeTwin[...]` entry (around line 107), add a `makeTwin` call following the exact pattern of the neighbouring entries:

```mathematica
makeTwin["IntraPlaneEnergyBalance.wl", "Phase 3b cycle 3 : Layer Energy Balance (undamped G0^vec, S-matrix unitarity)",
   "Builds the cycle-2 undamped vector G0^vec at physical parameters, runs the thesis-eps "
   <> "R/T projection, assembles the propagating-channel scattering matrix S = [[Rd,Tu],[Td,Ru]] "
   <> "(+ SH 2x2), and gates unitarity S^dag S = I across normal / sub-critical / post-critical p.  "
   <> "Sub-wavelength precondition (no open diffraction orders) makes the specular statement well-posed.  "
   <> "Executable twin: IntraPlaneEnergyBalance.wl.  Python cross-check: "
   <> "cubic_scattering/tests/test_intraplane_energy.py."];
```

- [ ] **Step 2: Generate the `.nb` and verify it round-trips**

```bash
/Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
  /Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/makeIntraPlaneNotebooks.wl
ls -la /Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneEnergyBalance.nb
```

Expected: `IntraPlaneEnergyBalance.nb` written (non-zero size). No need to re-run the physics.

- [ ] **Step 3: Update `IntraPlaneFoldyLax_Plan.md`**

Mark Phase 3b cycle 3 DONE and Phase 3b closed. Edit the cycle-3 bullet (around line 240) to a DONE entry mirroring the cycle-1/2 style: record the gate residuals (diffraction margin, η-independence, energy residual + the chosen `energyMetric`, reciprocity, the `Nmax` study outcome), the file names, and the answer to the open question (Nmax=2 sufficient or not). Update the Status line (line 3) and the Section-2 table row (line 34) to "Phase 3b DONE". Use the actual fresh-run numbers.

- [ ] **Step 4: Write the memory note**

Create `/Users/tod/.claude/projects/-Users-tod-Desktop-MultipleScatteringCalculations/memory/project-intraplane-energy-balance.md`:

```markdown
---
name: project-intraplane-energy-balance
description: Phase 3b cycle 3 DONE — undamped G0^vec layer R/T satisfies S-matrix unitarity (lossless energy balance), closing Phase 3b
metadata:
  type: project
---

Phase 3b cycle 3 DONE (`Mathematica/IntraPlaneEnergyBalance.wl` + `.nb`,
`IntraPlaneEnergyBalance_reference.json`, `cubic_scattering/tests/test_intraplane_energy.py`):
the lossless **energy balance** of the Phase-3a layer R/T(p). Swaps the Phase-2 damped G0
(`Im κ=0.25`, artificial loss) for the cycle-2 **undamped** G0^vec (Ewald `D[q,s]`) at PHYSICAL
params (`aL=2.5`, `κ_P=0.3`, `κ_S=0.5`, Bloch `k_x=ω p`), then gates **S-matrix unitarity**
`S†MS=M` on the propagating channels.

- **Energy metric (R1):** <plain S†S=I | Σ-twisted>, residual <fill>; SH `|Rsh|²+|Tsh|²` <fill>.
- **Gate [1] no open diffraction orders** is LOAD-BEARING: at `κ·aL≈1.25 < 2π`, every reciprocal
  `G≠0` is evanescent (margin <fill> > 0), so only the specular `G=0` order propagates → `p→p`
  specular unitarity is well-posed. The medium is laterally PERIODIC, not invariant; Bloch conserves
  `k_par mod G`. If a diffraction order ever opens, specular `S†S` is sub-unitary by the diffracted
  flux and the script Aborts (refuses), not soft-fails.
- **Post-critical p:** P evanescent → restrict S to SV(+SH) sub-block (evanescent modes carry no flux).
- **Nmax:** <Nmax=2 sufficient | higher needed>; energy residual sequence <fill>.
- **Reciprocity preserved** under undamping (<fill> ≤ 1e-6) — undamping fixes energy without breaking
  the Phase-3a symplectic reciprocity.

Two spaces never conflated: internal multipole 25×25 (Foldy-Lax) vs external plane-wave channels
(S-matrix). `projPW` + Weyl prefactor sums all multipoles into one amplitude per external channel.
See [[project-undamped-vector-g0]], [[project-intraplane-layer-rt]], [[thesis-energy-normalisation]],
[[kambe-validates-thesis-spectral]].
```

(Fill the `<...>` placeholders with the actual fresh-run values before committing.)

- [ ] **Step 5: Update `MEMORY.md` index**

Add one line under the Memory Files section of `MEMORY.md`:

```markdown
- [Intra-plane energy balance](project-intraplane-energy-balance.md) — Phase 3b cycle 3 DONE: undamped G0^vec layer R/T is S-matrix-unitary (lossless); gate [1] no-open-diffraction is load-bearing; Phase 3b closed
```

- [ ] **Step 6: Commit**

```bash
printf '%s\n' "📝 Phase 3b cycle 3 DONE: layer energy balance closed (.nb twin, plan, memory)" > /tmp/msg_c3t5.txt
git add Mathematica/makeIntraPlaneNotebooks.wl Mathematica/IntraPlaneEnergyBalance.nb IntraPlaneFoldyLax_Plan.md
git commit -F /tmp/msg_c3t5.txt
```

(The memory files live outside the repo; they are written but not part of this commit.)

---

## Self-Review

**Spec coverage:**
- §1 goal (energy balance on undamped G0^vec) → Tasks 1–3.
- §2 physical foundation (periodic-not-invariant; gate [1] load-bearing; Abort on open order) → Task 1 Step 5; memory note.
- §3 two-space channel structure (multipole 25×25 vs plane-wave S) → Task 1 (G0^vec/T0vec) + Task 2 (projPW/assembleS); documented in memory.
- §4 the build (lift cycle-2 to physical params; reuse RT projection; assemble + restrict S) → Tasks 1–3.
- §5 gates [1]–[5] → Task 1 (1,2), Task 3 (3,4,5).
- §6 R1 metric (plain vs Σ) → Task 2 Step 4; R2 Nmax → Task 3 Step 2; R3 tolerance → Task 2 Step 5 + Task 3 Step 4; R4 row+col restriction → Task 2 Step 3 (`assembleS` uses `mi` for both) + Task 3 (post-critical modes).
- §7 deliverables (`.wl`, `.json`, Python test, `.nb`, plan, memory) → Tasks 1–5.

**Placeholder scan:** the only `<...>` placeholders are in the memory-note template (Task 5 Step 4), explicitly to be filled with fresh-run numbers — not plan placeholders. All code steps contain runnable code.

**Type consistency:** `buildG0undamped[p, Nmax]`, `rtBlocksE[Tcoll, p, Nmax]`, `assembleS[e, modes]`, `propModes[p]`, `energyResid[e, modes, metric]`, `energyMetric`, `enTol` used consistently across Tasks 1–4 (MMA) and the Python `_sig_metric`/`_cplx`/dump keys (`S_psv`, `S_sh`, `propModes`, `energyMetric`, `enTol`, `diffMargin`, `nmaxStudy`) match the Task-3 `Export`.
