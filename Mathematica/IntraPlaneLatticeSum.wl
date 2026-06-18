#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneLatticeSum.wl  --  Phase 1 of the Intra-Plane Foldy-Lax program:
   the planar lattice sum of the Phase-0 translation operator at fixed Bloch
   vector k_par, giving the intra-plane coupling G0(k_par).

   G0_{nm,nu mu}(k_par) = Sum_{R != 0} beta_{nm,nu mu}(R) e^{i k_par . R}
     = 4 Pi (-1)^m Sum_q i^(nu+q-n) (-1)^q D[q, m-mu] gaunt(n,m,nu,-mu,q,mu-m),
   where the scalar STRUCTURE CONSTANTS carry the lattice sum:
     D[q,s]  = Sum_{R != 0} h_q(kappa|R|) Y_q^s(^R) e^{i k_par . R}.
   The same constants reconstruct the lattice-summed scalar field (conjugate index):
     G_kappa^{R!=0}(r) = Sum_{R!=0} g_kappa(r-R) e^{i k_par.R}
                       = i kappa Sum_{q,p} Dbar[q,p] j_q(kappa r) Y_q^p(^r),
     Dbar[q,p] = Sum_{R!=0} h_q(kappa|R|) conj(Y_q^p(^R)) e^{i k_par.R},
     g_kappa(s) = e^{i kappa |s|}/(4 Pi |s|).

   Conventions inherited from Phase 0 / CartesianT0.wl: time e^{+i w t}, outgoing
   h_n^(1), orthonormal SphericalHarmonicY, (x,y,z) with z = polar axis = depth.
   The lattice lies in the x-y plane (R has zero 3rd component, theta_R = Pi/2).
   A small Im(kappa) damping makes the conditionally-convergent sum converge (the
   Ewald acceleration for the undamped sum is Phase 1 tracer bullet 2).

   TDD RED: the structure constants are stubbed to 0, so the multipole
   reconstruction of the lattice-summed field is 0 and MUST FAIL.
   ============================================================================ *)

(* ---- helpers reused from Phase 0 (IntraPlaneTranslation.wl) ---- *)
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];
ang[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];

(* ---- planar square lattice + Bloch phase (precomputed flat lists) ---- *)
aL = 2.0;                                  (* lattice constant *)
kappaP = 0.9 + 0.25 I; kappaS = 1.5 + 0.25 I;   (* damped P,S wavenumbers, Im>0 *)
kx = 0.2; ky = 0.1;                        (* horizontal Bloch vector k_par=(kx,ky,0) *)
Lrad = 18;                                 (* lattice half-width (shells) *)

ijList = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lrad, Lrad}, {j, -Lrad, Lrad}], 1];
iVals = ijList[[All, 1]]; jVals = ijList[[All, 2]];
rNorms = aL Sqrt[iVals^2 + jVals^2];       (* |R| *)
phiList = ArcTan[iVals, jVals];            (* azimuth of R; theta_R = Pi/2 *)
blochList = Exp[I aL (kx iVals + ky jVals)];   (* e^{i k_par . R} *)

(* ============================================================================
   SCALAR STRUCTURE CONSTANTS  [ STUB -- RED stage ]
   ============================================================================ *)
(* memoised: each is a Total over the precomputed lattice lists (theta_R = Pi/2) *)
Dstruct[q_, s_, kappa_] := Dstruct[q, s, kappa] =
   Total[sh[q, kappa rNorms] SphericalHarmonicY[q, s, Pi/2, phiList] blochList];
Dbar[q_, p_, kappa_] := Dbar[q, p, kappa] =
   Total[sh[q, kappa rNorms] Conjugate[SphericalHarmonicY[q, p, Pi/2, phiList]] blochList];

(* ---- lattice-summed scalar field (direct, the ground truth) ---- *)
gLatField[kappa_, r_] :=
  Module[{nn = Sqrt[(r[[1]] - aL iVals)^2 + (r[[2]] - aL jVals)^2 + r[[3]]^2]},
   Total[Exp[I kappa nn]/(4 Pi nn) blochList]];

(* ---- multipole reconstruction of the same field from the structure constants ---- *)
gLatRecon[kappa_, r_, Qmax_] :=
  Module[{rr = Sqrt[r . r]},
   I kappa Sum[Dbar[q, p, kappa] sj[q, kappa rr] Yfun[q, p, r], {q, 0, Qmax}, {p, -q, q}]];

(* ---- test fixture: field points near the origin, |r| < aL ---- *)
kTest = kappaS; QmaxTest = 12;
SeedRandom[20260618];
fieldPts = Table[
   Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}], st},
    st = Sqrt[1 - u^2]; 0.5 {st Cos[ph], st Sin[ph], u}], {6}];

tolField = 1*^-6;
resField = Max[Table[Abs[gLatField[kTest, pt] - gLatRecon[kTest, pt, QmaxTest]], {pt, fieldPts}]];
Print["==== Phase 1 :: planar lattice sum (damped, direct) ===="];
Print["  aL=", aL, ", kappa=", kTest, ", k_par=(", kx, ",", ky, "), Lrad=", Lrad, ", Qmax=", QmaxTest];
Print["  [a] |G_lat(direct field) - multipole reconstruction| (", Length[fieldPts], " pts) = ",
   resField, "  -> ", If[resField < tolField, "PASS", "FAIL"]];

(* [b] convergence of the damped lattice sum in lattice radius *)
DstructAt[q_, s_, kappa_, Lr_] := Module[{ij, iv, jv, rn, ph, bl},
   ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
   iv = ij[[All, 1]]; jv = ij[[All, 2]];
   rn = aL Sqrt[iv^2 + jv^2]; ph = ArcTan[iv, jv]; bl = Exp[I aL (kx iv + ky jv)];
   Total[sh[q, kappa rn] SphericalHarmonicY[q, s, Pi/2, ph] bl]];
convResid = Max[Table[Abs[DstructAt[q, s, kTest, Lrad] - DstructAt[q, s, kTest, Lrad + 8]], {q, 0, 4}, {s, -q, q}]];
Print["  [b] structure-constant convergence |D(Lrad) - D(Lrad+8)| (damped) = ", convResid,
   "  -> ", If[convResid < 1*^-3, "PASS", "FAIL"]];

(* [c] G0 = Gaunt-contracted structure constants (= Sum_R beta(R) e^{i k_par.R}); reciprocity *)
G0[n_, m_, nu_, mu_, kappa_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dstruct[q, m - mu, kappa] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
recipPairs = {{1, 0, 2, 1}, {2, -1, 3, 2}, {0, 0, 3, 0}, {2, 2, 4, -2}, {1, 1, 3, -1}};
recipResid[kappa_] := Max[Table[
    Abs[G0[p[[1]], p[[2]], p[[3]], p[[4]], kappa]
       - (-1)^(p[[1]] + p[[3]] + p[[2]] + p[[4]]) G0[p[[3]], -p[[4]], p[[1]], -p[[2]], kappa]], {p, recipPairs}]];
Print["  [c] G0 reciprocity (kappaP) = ", recipResid[kappaP], "  -> ", If[recipResid[kappaP] < 1*^-10, "PASS", "FAIL"]];
Print["      G0 reciprocity (kappaS) = ", recipResid[kappaS], "  -> ", If[recipResid[kappaS] < 1*^-10, "PASS", "FAIL"]];
Print["  sample G0[1,0,2,1; kappaS] = ", G0[1, 0, 2, 1, kappaS]];
Print["Phase 1 TB1 (IntraPlaneLatticeSum.wl) loaded + verified."];
