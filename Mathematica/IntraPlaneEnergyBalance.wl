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
   Time e^{+i w t}, outgoing h^(1); lattice in x-y; project frame (z,x,y).
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

(* ============================================================================
   Step 2: Non-memoised structure-constant projection + per-p Dtab
   ============================================================================ *)
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

(* ============================================================================
   Step 3: Vector helpers + coeff_q extraction + block builders
   (verbatim from IntraPlaneKambeVector.wl lines 69-151)
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

(* ============================================================================
   Step 4: setMie, idxVof, T0LMN, collV, buildG0undamped
   ============================================================================ *)
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

(* ============================================================================
   Step 5: Gate [1] (no open diffraction orders) + Gate [2] (eta-independence)
   ============================================================================ *)
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
