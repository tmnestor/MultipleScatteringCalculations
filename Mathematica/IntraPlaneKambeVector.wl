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
