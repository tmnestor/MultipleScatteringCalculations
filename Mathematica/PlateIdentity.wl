#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   PlateIdentity.wl

   Task 0 of plans/voxel_generator_riccati.md, Mathematica steps 1-2.

   WHY THIS EXISTS

   A single layer of space-filling cubic voxels is to supply the O(h)
   generator of the depth Riccati equation.  Its in-plane coupling does not
   vanish as h -> 0: the 1/r^3 neighbour field summed over the plane is O(1)
   per cube, and together with the self term it must reproduce the local field
   of a THIN PLATE.  Tromp & Snieder (1989) dropped exactly this and are wrong
   at O(eps^2) for it (TrompSniederBornGenerator.wl).

   THE CLAIMS

   Static, isotropic background, strain response to a uniform stress
   polarisation tau in every cube of the plane z = 0 (cube side d = 1):

     eps = - Gamma . tau,     Gamma(x) = -sym d_j d_l G_ik(x),
                              GammaHat(k) = sym k_j k_l K(k)^-1_ik / |k|^2.

   1. Gamma_plate = sym(n_j n_l K(n)^-1_ik), n = e3, derived two ways:
      (a) from traction continuity across a uniformly polarised slab;
      (b) as the aspect -> 0 limit of the oblate-spheroid Hill tensor
          P(e) = (e / 4 pi) Int_{S^2} GammaHat(xi) / |A xi|^3,
          A = diag(1, 1, e), whose e = 1 case is checked against the sphere.
   2. In Fourier space only the G_par = 0 order of a plane of CUBE sources
      survives (the cube form factor vanishes at every other reciprocal
      vector), and GammaHat(0,0,kz) is kz-independent, so both the collocation
      (receiver centre) and Galerkin (receiver average) sums equal Gamma_plate
      exactly.
   3. Real space, collocation: the cube's centre-point self term plus the
      source-averaged field of every other cube, evaluated at the receiver
      centre, reproduces Gamma_plate.  This is the operator a consistent
      collocation solver applies.
   4. The solver's DEFAULT pairing -- source-averaged on the contact shell,
      bare midpoint beyond -- misses Gamma_plate by the midpoint bias computed
      here.  The bias is static and d-independent, so refinement cannot
      remove it.

   Coordinates here are x, y, z = 1, 2, 3 with the plate normal e3 (the
   package orders z first; the Python step must permute).

   Run:
       wolframscript -file Mathematica/PlateIdentity.wl
   ===================================================================== *)

$HistoryLength = 0;
Needs["NumericalDifferentialEquationAnalysis`"];

fmt[v_] := Module[{x = N[v], e},
   If[x == 0, "0", e = Floor[Log10[Abs[x]]];
    ToString[NumberForm[x/10^e, {3, 2}]] <> If[e == 0, "", "e" <> ToString[e]]]];
nPass = 0; nTotal = 0;
check[name_, ok_, detail_] := (nTotal++; If[TrueQ[ok], nPass++];
   Print["  [", If[TrueQ[ok], "PASS", "FAIL"], "] ", name, ": ", detail]);

(* ---- background: the project's test medium, GPa ---- *)
mu = 45/2; lam = 35/2;              (* alpha = 5, beta = 3 km/s, rho = 2.5 *)
nu = lam/(2 (lam + mu));            (* = 7/32 *)
delta[i_, j_] := KroneckerDelta[i, j];
c0 = Table[lam delta[i, j] delta[k, l] + mu (delta[i, k] delta[j, l] + delta[i, l] delta[j, k]),
   {i, 3}, {j, 3}, {k, 3}, {l, 3}];

symm[f_] := Table[(f[i, j, k, l] + f[j, i, k, l] + f[i, j, l, k] + f[j, i, l, k])/4,
   {i, 3}, {j, 3}, {k, 3}, {l, 3}];
maxAbs[t_] := Max[Abs[Flatten[N[t, 30]]]];
relDiff[a_, b_] := maxAbs[a - b]/maxAbs[b];

(* GammaHat for a unit direction n *)
kInvIso[n_] := (IdentityMatrix[3] - Outer[Times, n, n])/mu + Outer[Times, n, n]/(lam + 2 mu);
gammaHat[n_] := Module[{ki = kInvIso[n]}, symm[n[[#2]] n[[#4]] ki[[#1, #3]] &]];

voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
toVoigt[t_] := Table[t[[Sequence @@ voigt[[a]], Sequence @@ voigt[[b]]]], {a, 6}, {b, 6}];
eshelby[g_] := Table[Sum[g[[i, j, m, n]] c0[[m, n, k, l]], {m, 3}, {n, 3}], {i, 3}, {j, 3}, {k, 3}, {l, 3}];

printVoigt[m_] := Do[Print["     ", StringRiffle[
     If[NumericQ[#] && Abs[#] < 10^-12, "0", ToString[If[Precision[#] === Infinity, InputForm[#], NumberForm[#, 4]]]] & /@ row,
     "   "]], {row, m}];

gPlate = gammaHat[{0, 0, 1}];
Print["Background mu = ", mu, " GPa, lambda = ", lam, " GPa, nu = ", nu];
Print["\nmu * Gamma_plate, tensor components, Voigt order xx yy zz yz xz xy:"];
printVoigt[mu toVoigt[gPlate]];

(* =====================================================================
   1. Gamma_plate two ways
   ===================================================================== *)
Print["\n1. The plate operator"];

(* (a) slab ODE: u = u(z); traction continuity C_i3k3 u_k' + tau_i3 = 0 inside,
   u' = 0 outside, so eps_ij = sym(u'_i n_j) with u' = -K^-1 (tau . n).
   K is built from c0 directly, not from kInvIso. *)
kSlab = Table[c0[[i, 3, k, 3]], {i, 3}, {k, 3}];
gSlab = Module[{kinv = Inverse[kSlab]},
   symm[kinv[[#1, #3]] delta[#2, 3] delta[#4, 3] &]];
check["(a) slab traction continuity == sym(n n K^-1)", relDiff[gSlab, gPlate] == 0,
  "rel diff " <> fmt[relDiff[gSlab, gPlate]]];

(* Eshelby form, against the textbook thin-plate values *)
sPlate = eshelby[gPlate];
check["(a) S_3333 = 1, S_3311 = nu/(1-nu), S_1313 = 1/2, S_1111 = S_1122 = 0",
  sPlate[[3, 3, 3, 3]] == 1 && sPlate[[3, 3, 1, 1]] == nu/(1 - nu) &&
   sPlate[[1, 3, 1, 3]] == 1/2 && sPlate[[1, 1, 1, 1]] == 0 && sPlate[[1, 1, 2, 2]] == 0,
  "S_3311 = " <> ToString[InputForm[sPlate[[3, 3, 1, 1]]]] <> " = nu/(1-nu)"];

(* (b) oblate-spheroid Hill tensor.  GammaHat depends on xi through
   (theta, phi); integrate phi analytically, theta numerically. *)
hill[e_?NumericQ] := Module[{th, ph, xi, integrand, phiAvg},
   xi = {Sin[th] Cos[ph], Sin[th] Sin[ph], Cos[th]};
   phiAvg = Integrate[gammaHat[xi], {ph, 0, 2 Pi}];
   integrand = phiAvg e Sin[th]/(Sin[th]^2 + e^2 Cos[th]^2)^(3/2)/(4 Pi);
   (* components identically zero by symmetry are skipped, not integrated *)
   Map[If[TrueQ[Simplify[#] == 0], 0,
       NIntegrate[#, {th, 0, Pi}, WorkingPrecision -> 30, PrecisionGoal -> 18,
        MaxRecursion -> 40]] &, integrand, {4}]];

hSphere = hill[1];
sSphere = eshelby[hSphere];
check["(b) e = 1 reproduces the sphere, S_1111 = (7-5nu)/(15(1-nu))",
  Abs[sSphere[[1, 1, 1, 1]] - (7 - 5 nu)/(15 (1 - nu))] < 10^-15,
  "S_1111 " <> fmt[sSphere[[1, 1, 1, 1]]]];
errs = Table[{e, relDiff[hill[e], gPlate]}, {e, {1/100, 1/1000, 1/10000}}];
Print["   aspect  |P(e) - Gamma_plate| / |Gamma_plate|"];
Do[Print["   ", fmt[r[[1]]], "   ", fmt[r[[2]]]], {r, errs}];
rate = Log[errs[[2, 2]]/errs[[3, 2]]]/Log[10];
check["(b) oblate limit -> Gamma_plate, first order in aspect", errs[[3, 2]] < 10^-3 && Abs[rate - 1] < 0.1,
  "error at 1e-4: " <> fmt[errs[[3, 2]]] <> ", observed order " <> fmt[rate]];

(* =====================================================================
   2. The Fourier-space in-plane sum
   ===================================================================== *)
Print["\n2. Spectral same-plane sum"];
sinc[x_] := Sin[x]/x;
check["cube form factor vanishes at every nonzero in-plane reciprocal vector",
  Simplify[Sin[m Pi], Element[m, Integers]] === 0,
  "sinc(G_x d/2) sinc(G_y d/2) = 0 for G = 2 pi (m1, m2)/d != 0"];
kz = Symbol["kz"];
check["GammaHat(0, 0, kz) is independent of kz",
  Simplify[gammaHat[{0, 0, kz}/Sqrt[kz^2]] - gPlate, kz != 0 && Element[kz, Reals]] ===
   ConstantArray[0, {3, 3, 3, 3}], "degree-0 in k"];
single = Integrate[sinc[q/2], {q, -Infinity, Infinity}]/(2 Pi);
double = Integrate[sinc[q/2]^2, {q, -Infinity, Infinity}]/(2 Pi);
check["collocation weight (1/2pi) Int sinc(kz d/2) dkz = 1 (d = 1)", single == 1, ToString[single]];
check["Galerkin weight (1/2pi d) Int d sinc^2(kz d/2) dkz = 1", double == 1, ToString[double]];

(* =====================================================================
   3-4. Real space, collocation, and the default solver's midpoint bias
   ===================================================================== *)
Print["\n3. Real-space collocation sum (d = 1)"];
xs = {x1, x2, x3}; r = Sqrt[xs . xs];
kelvin = Table[((3 - 4 nu) delta[i, k] + xs[[i]] xs[[k]]/r^2)/(16 Pi mu (1 - nu) r), {i, 3}, {k, 3}];
gamReal = -symm[D[kelvin[[#1, #3]], xs[[#2]], xs[[#4]]] &];
lap[f_] := Sum[D[f, v, v], {v, xs}];
ser2 = lap /@ Flatten[gamReal]/24;                                    (* <f> - f, d^2 *)
ser4 = (Sum[D[#, {v, 4}], {v, xs}]/1920 + Sum[D[#, {xs[[a]], 2}, {xs[[b]], 2}], {a, 3}, {b, a + 1, 3}]/576) & /@
   Flatten[gamReal];                                                  (* d^4 *)
gamC = Compile[{{y1, _Real}, {y2, _Real}, {y3, _Real}},
   Evaluate[N[Flatten[gamReal] /. Thread[xs -> {y1, y2, y3}]]]];
serC = Compile[{{y1, _Real}, {y2, _Real}, {y3, _Real}},
   Evaluate[N[(ser2 + ser4) /. Thread[xs -> {y1, y2, y3}]]]];

(* source-cube average of Gamma, evaluated at the receiver centre (origin):
   <Gamma>(R) = Int_cube(R) Gamma(0 - x') dx' = Int Gamma(x') (Gamma is even). *)
gauss[n_] := Transpose[GaussianQuadratureWeights[n, -1/2, 1/2, 20]];
cellAvg[{n1_, n2_}, ng_] := Module[{nodes, wts, g = gauss[ng]},
   {nodes, wts} = N[g];
   Sum[wts[[a]] wts[[b]] wts[[c]] gamC[n1 + nodes[[a]], n2 + nodes[[b]], nodes[[c]]],
    {a, ng}, {b, ng}, {c, ng}]];

(* consistency of the two averaging routes where both apply *)
Do[
  gA = cellAvg[cell, 24]; gB = gamC[cell[[1]], cell[[2]], 0.] + serC[cell[[1]], cell[[2]], 0.];
  check["Gauss vs d^2+d^4 series at R = " <> ToString[cell], Max[Abs[gA - gB]]/Max[Abs[gA]] < 10^-6,
   "rel diff " <> fmt[Max[Abs[gA - gB]]/Max[Abs[gA]]]],
  {cell, {{8, 0}, {6, 5}, {12, 3}}}];
gA1 = cellAvg[{1, 0}, 24]; gA2 = cellAvg[{1, 0}, 32];
check["contact cell Gauss converged (24 vs 32 nodes)", Max[Abs[gA1 - gA2]]/Max[Abs[gA2]] < 10^-12,
  "rel diff " <> fmt[Max[Abs[gA1 - gA2]]/Max[Abs[gA2]]]];

(* collocation self term: centre-point field of a uniform cube source.
   G_ik = [(4-4nu) delta_ik / r - d_i d_k r] / (16 pi mu (1-nu)).  Cube potentials
   Phi = Int 1/r, Psi = Int r.  At the centre, by cubic symmetry,
   Phi_jl = -(4 pi/3) delta_jl, and Psi_ikjl = c (dd + dd + dd) + b delta_ikjl
   with 5c + b = -8 pi/3 (from Laplacian Psi = 2 Phi) and c = Psi_1122 computed
   by the divergence theorem on the two faces x1 = +-1/2. *)
fFace = D[r, x1, x2, x2];
faceInt = FullSimplify[Integrate[fFace /. x1 -> 1/2, {x2, -1/2, 1/2}, {x3, -1/2, 1/2}]];
cc = 2 faceInt; bb = -8 Pi/3 - 5 cc;
Print["   collocation self term: Psi_1122(0) = ", ToString[InputForm[cc]], ", Psi_1111(0) = ",
  ToString[InputForm[FullSimplify[3 cc + bb]]], "  (exact)"];
psi4[i_, k_, j_, l_] := cc (delta[i, k] delta[j, l] + delta[i, j] delta[k, l] + delta[i, l] delta[k, j]) +
   bb If[i == k == j == l, 1, 0];
phi2[j_, l_] := -(4 Pi/3) delta[j, l];
gSelf = -symm[((4 - 4 nu) delta[#1, #3] phi2[#2, #4] - psi4[#1, #3, #2, #4])/(16 Pi mu (1 - nu)) &];

rNear = 12; rMax = 400;
nearCells = Select[Tuples[Range[-rNear, rNear], 2], # != {0, 0} &];
avgNear = Association[Table[cell -> cellAvg[cell, If[Max[Abs[cell]] <= 2, 24, 12]], {cell, nearCells}]];
pointNear = Association[Table[cell -> gamC[cell[[1]], cell[[2]], 0.], {cell, nearCells}]];

farCells = Select[Flatten[Table[{i, j}, {i, -rMax, rMax}, {j, -rMax, rMax}], 1],
   Max[Abs[#]] > rNear && Norm[#] <= rMax &];
pointFar = Total[gamC[#[[1]], #[[2]], 0.] & /@ farCells];
serFar = Total[serC[#[[1]], #[[2]], 0.] & /@ farCells];
(* continuum tail beyond |R| = rMax: Gamma is homogeneous of degree -3 *)
tailPt = N[Flatten[(1/rMax) Integrate[gamReal /. Thread[xs -> {Cos[t], Sin[t], 0}], {t, 0, 2 Pi}]]];

sumCollocation = Flatten[N[gSelf]] + Total[Values[avgNear]] + pointFar + serFar + tailPt;
eColl = Max[Abs[sumCollocation - Flatten[N[gPlate]]]]/maxAbs[gPlate];
check["self (centre) + source-averaged plane sum == Gamma_plate", eColl < 10^-5,
  "rel residual " <> fmt[eColl] <> " (floor: lattice-vs-continuum tail beyond R = " <> ToString[rMax] <> ")"];

(* 4. default pairing: average only on the contact shell (Chebyshev radius r0),
   bare midpoint beyond.  bias = sum_{beyond} (point - average). *)
Print["\n4. Midpoint bias of the default pairing (static, d-independent)"];
biasFor[r0_] := Total[Table[pointNear[c] - avgNear[c], {c, Select[nearCells, Max[Abs[#]] > r0 &]}]] - serFar;
Do[
  b = biasFor[r0];
  rel = Max[Abs[b]]/maxAbs[gPlate];
  bV = mu toVoigt[ArrayReshape[b, {3, 3, 3, 3}]];
  Print["   average out to Chebyshev radius ", r0, ": max|bias| / max|Gamma_plate| = ", fmt[rel]];
  If[r0 == 1, Print["   mu * bias, Voigt order xx yy zz yz xz xy:"]; printVoigt[bV]],
  {r0, {1, 2, 3, 4}}];
check["default (contact-only) pairing does NOT reproduce Gamma_plate", Max[Abs[biasFor[1]]]/maxAbs[gPlate] > 10^-4,
  "rel " <> fmt[Max[Abs[biasFor[1]]]/maxAbs[gPlate]]];

Print["\n", nPass, "/", nTotal, " checks passed"];
Exit[If[nPass == nTotal, 0, 1]];
