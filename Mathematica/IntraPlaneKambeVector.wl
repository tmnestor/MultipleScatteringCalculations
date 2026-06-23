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

Print["==== Phase 3b cycle 2 :: undamped vector G0 ===="];
NqS = 4;
undEtaP = Max[Table[Abs[DstructU[q, s, 0.9, 0.7] - DstructU[q, s, 0.9, 1.15]], {q, 0, NqS}, {s, -q, q}]];
undEtaS = Max[Table[Abs[DstructU[q, s, 1.5, 0.7] - DstructU[q, s, 1.5, 1.15]], {q, 0, NqS}, {s, -q, q}]];
Print["  [1] scalar D eta-independence (kappa_P=0.9) = ", ScientificForm[undEtaP, 3],
   " -> ", If[undEtaP < 1.*^-6, "PASS", "FAIL"]];
Print["  [2] scalar D eta-independence (kappa_S=1.5) = ", ScientificForm[undEtaS, 3],
   " -> ", If[undEtaS < 1.*^-6, "PASS", "FAIL"]];

(* ============================================================================
   Task 2: vector wavefunctions + quadrature (wavenumber-parameterized)
   ============================================================================ *)
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

(* ============================================================================
   Task 2: single-pair Wel + coefficient extraction coeffq
   ============================================================================ *)
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

(* ============================================================================
   Task 2 gate [3]: reconstruction self-check (Sum_q coeff_q h_q Y_q == Wel)
   ============================================================================ *)
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

(* ============================================================================
   Task 3: G0^vec block builders + damped-limit method gate
   ============================================================================ *)
(* L block: scalar-Gaunt contraction of the scalar structure constant Dfun *)
g0LLk[n_, m_, nu_, mu_, Dfun_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dfun[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
(* M/N block: Sum_q coeff_q(kS) * Dfun[q, m-mu] *)
g0MNblock[cP_, nu_, mu_, c_, n_, m_, kS_, Dfun_] := Module[{cf = coeffq[cP, nu, mu, c, n, m, kS]},
   Total[KeyValueMap[#2 Dfun[#1, m - mu] &, cf]]];

(* ---- direct damped lattice sum (Phase-2(b) ground truth, wavenumber-parameterized) ----
   NOTE: the structure-constant radius Ldir (below) MUST equal this direct-sum lattice
   radius LradB.  g0MNblock/g0LLk contract DstructDirect[..,Ldir]; the ground-truth
   g0dir/g0LLdir sum srcQuad/betaSep over Rvecs/blochL (radius LradB).  These are two
   truncations of the SAME damped (Im kappa = 0.25) conditionally-convergent lattice sum,
   so the damped-limit gate is a term-by-term algebraic identity ONLY when both truncate
   at the identical radius (verified: matched radius -> ~4e-11; radius 8 vs 18 -> ~1e-2
   residual truncation gap, which previously failed gates [4]/[5]). *)
LradB = 8;
ijLat = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -LradB, LradB}, {j, -LradB, LradB}], 1];
Rvecs = Map[{aL #[[1]], aL #[[2]], 0.0} &, ijLat];
blochL = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ijLat];
latSrcQuad[c_, n_, m_, kS_] := latSrcQuad[c, n, m, kS] =
   Total[Table[blochL[[iR]] srcQuad[c, n, m, Rvecs[[iR]], kS], {iR, Length[Rvecs]}]];
g0dir[cP_, nu_, mu_, c_, n_, m_, kS_] := Switch[cP,
   "M", projDotF[latSrcQuad[c, n, m, kS], Cvals[nu, mu]]/normMC[nu, mu, kS],
   "N", projDotF[latSrcQuad[c, n, m, kS], Pvals[nu, mu]]/normNP[nu, mu, kS]];

(* ============================================================================
   Task 3 gate [4]: M/N contraction == direct damped sum
   Task 3 gate [5]: L-block contraction == direct beta^P sum
   ============================================================================ *)
kSd = 1.5 + 0.25 I; kPd = 0.9 + 0.25 I; Ldir = LradB;  (* MUST match direct-sum radius *)
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

(* ============================================================================
   Task 4 Step 1: Undamped G0^vec assembly (kappa_P for L, kappa_S for M/N)
   ============================================================================ *)
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

(* ============================================================================
   Task 4 Step 2: Collective solve + gates [6],[7],[8]
   ============================================================================ *)
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
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

(* ============================================================================
   Task 4 Step 3: Dump JSON reference
   ============================================================================ *)
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneKambeVector_reference.json",
  <|"params" -> <|"aL" -> aL, "kx" -> kx, "ky" -> ky, "kappaP" -> 0.9, "kappaS" -> 1.5,
      "eta" -> etaU, "rho0" -> rho0, "radP" -> radP, "dext" -> dext, "Nmax" -> Nmax, "LradB" -> LradB|>,
    "idx" -> Map[{#[[1]], #[[2]], #[[3]]} &, idx],
    "G0vec" -> Map[reim, G0vec, {2}],
    "recip_resid" -> N[recLL], "coupling" -> N[coupling], "iso_dev" -> N[isoDev]|>];
Print["  wrote IntraPlaneKambeVector_reference.json (G0^vec ", nDim, "x", nDim, ")"];
