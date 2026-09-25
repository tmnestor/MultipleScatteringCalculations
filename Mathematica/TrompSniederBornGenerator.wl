#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   TrompSniederBornGenerator.wl

   Independent cross-check of scripts/gate_tromp_snieder_uniform_band.py.

   WHY THIS EXISTS

   Tromp & Snieder, Geophys. J. 96, 447-456 (1989), obtain matrix Riccati
   equations (their 3.6a-d) for the R/T of a stratified band by applying
   invariant imbedding to the FIRST-BORN response of a thin homogeneous layer
   (their 2.12a-f, 2.15a-f), which they state is exact at O(dz) because the
   n-th Born term is O(dz^n).  The Python gate measured that the equations are
   exact for density contrast but wrong at O(eps^2) for stiffness contrast.

   THE CLAIM PROVED HERE, STRONGER THAN THE PYTHON GATE

   Write the elastodynamic system as d/dz b = A b, b = (u, t), t_i = sigma_i3.
   A is linear in rho but NOT in the stiffness: it contains Inverse[C33].
   Let A_lin = A0 + (d/de A(c0 + e cs, rho0 + e rhos))|_{e=0}.  Then

     (A) the thin-layer coefficients of their eqs 2.12/2.15 are EXACTLY the
         interaction-picture blocks of A_lin - A0 in their plane-wave basis,
         for any slowness and any (anisotropic) perturbation; and
     (B) consequently their Riccati solution for a uniform band equals, to
         integration precision, the EXACT response of a fictitious band with
         system matrix A_lin -- at every slowness, not only at p = 0.

   A density-only perturbation leaves A_lin = A exactly, so there they are
   right.  Any stiffness perturbation does not.

   INDEPENDENCE FROM THE PYTHON GATE

   * Reference: exact band propagator MatrixExp[A H] from the first-order
     system, NOT the Kennett recursion the Python gate uses.
   * Their coefficients are transcribed term by term with the paper's four
     sign patterns, NOT through the wavevector contraction the Python uses.
   * The Riccati system is their eqs 3.6a-d as printed, integrated by NDSolve
     at 32 digits.
   * The finite-contrast errors are then compared against the numbers the
     Python gate printed (which were scored against Kennett).

   CONVENTIONS: x, y, z = indices 1, 2, 3 as in the paper (z down);
   exp(I(k x - w t)); displacement-normalised polarisation vectors (2.3)-(2.4).

   Run:
       wolframscript -file Mathematica/TrompSniederBornGenerator.wl
   ===================================================================== *)

$HistoryLength = 0;
prec = 40;
(* one-line, 3 significant figures, for exact, arbitrary- and machine-precision input *)
fmt[v_] := Module[{x = N[v], e},
   If[x == 0, "0", e = Floor[Log10[Abs[x]]];
    ToString[NumberForm[x/10^e, {3, 2}]] <> If[e == 0, "", "e" <> ToString[e]]]];
nPass = 0; nTotal = 0;
check[name_, ok_, detail_] := (nTotal++; If[TrueQ[ok], nPass++];
   Print["  [", If[TrueQ[ok], "PASS", "FAIL"], "] ", name, ": ", detail]);

(* ---- background: the project's test parameters, SI, exact ---- *)
alpha0 = 5000; beta0 = 3000; rho0 = 2500;
mu0 = rho0 beta0^2; lam0 = rho0 alpha0^2 - 2 mu0; mM0 = lam0 + 2 mu0;
omega = 20 Pi; hBand = 100;
pList = {0, 8/100000, 15/100000};

delta[i_, j_] := KroneckerDelta[i, j];
cIso[l_, m_] := Table[l delta[i, j] delta[k, n] + m (delta[i, k] delta[j, n] + delta[i, n] delta[j, k]),
   {i, 3}, {j, 3}, {k, 3}, {n, 3}];
c0 = cIso[lam0, mu0];

(* ---- the first-order system matrix, from the stiffness tensor ---- *)
cab[c_, a_, b_] := Table[c[[i, a, k, b]], {i, 3}, {k, 3}];   (* (C_ab)_ik = c_iakb *)
aMat[c_, rho_, k_] := Module[{c33i = Inverse[cab[c, 3, 3]], c31 = cab[c, 3, 1],
    c13 = cab[c, 1, 3], c11 = cab[c, 1, 1]},
   ArrayFlatten[{{-I k c33i . c31, c33i},
      {-rho omega^2 IdentityMatrix[3] + k^2 c11 - k^2 c13 . c33i . c31, -I k c13 . c33i}}]];

aLin[cs_, rhos_, k_] := Module[{e},
   aMat[c0, rho0, k] + (D[aMat[c0 + e cs, rho0 + e rhos, k], e] /. e -> 0)];

(* ---- background plane-wave basis, eqs (2.3)-(2.4) ---- *)
nuA[k_] := Sqrt[(omega/alpha0)^2 - k^2];
nuB[k_] := Sqrt[(omega/beta0)^2 - k^2];
nu[k_] := {nuA[k], nuB[k], nuB[k]};
modOut = {mM0, mu0, mu0};
pD[k_] := {(alpha0/omega) {k, 0, nuA[k]}, (beta0/omega) {-nuB[k], 0, k}, {0, 1, 0}};
pU[k_] := {(alpha0/omega) {k, 0, -nuA[k]}, (beta0/omega) {nuB[k], 0, k}, {0, 1, 0}};
kzD[k_] := nu[k]; kzU[k_] := -nu[k];

(* column = (p, t) with t_i = c0_i3kl (I kvec_l) p_k *)
column[p_, kz_, k_] := Join[p, Table[Sum[c0[[i, 3, kk, l]] I {k, 0, kz}[[l]] p[[kk]], {kk, 3}, {l, 3}], {i, 3}]];
dMat[k_] := Transpose[Join[
    MapThread[column[#1, #2, k] &, {pD[k], kzD[k]}],
    MapThread[column[#1, #2, k] &, {pU[k], kzU[k]}]]];
eMat[k_, z_] := DiagonalMatrix[Exp[I Join[kzD[k], kzU[k]] z]];

(* ---- their thin-layer coefficients, transcribed term by term ----
   bracket_ij = rhos w^2 d_ij - k^2 c_i11j  s33 nuo nut c_i33j  s13 k nut c_i13j  s31 k nuo c_i31j
   kind  out  in   s33 s13 s31   phase exp(I sph (nut - or + nuo) z)
   tD    D    D    -   -   -     exp(I (nut - nuo) z)      (2.12a-c)
   rU    U    D    +   -   +     exp(I (nut + nuo) z)      (2.12d-f)
   tU    U    U    -   +   +     exp(-I (nut - nuo) z)     (2.15a-c)
   rD    D    U    +   +   -     exp(-I (nut + nuo) z)     (2.15d-f)                  *)
signs = <|"tD" -> {-1, -1, -1}, "rU" -> {1, -1, 1}, "tU" -> {-1, 1, 1}, "rD" -> {1, 1, -1}|>;
pols[k_] := <|"tD" -> {pD[k], pD[k]}, "rU" -> {pU[k], pD[k]}, "tU" -> {pU[k], pU[k]},
    "rD" -> {pD[k], pU[k]}|>;
phase[kind_, nut_, nuo_, z_] := Switch[kind,
   "tD", Exp[I (nut - nuo) z], "rU", Exp[I (nut + nuo) z],
   "tU", Exp[-I (nut - nuo) z], "rD", Exp[-I (nut + nuo) z]];

(* The paper's c_iabj, read with its indices in the printed order.  (By minor
   symmetry it equals (C_ab)_ij of aMat, but it is deliberately not reused.) *)
cPaper[cs_, a_, b_] := Table[cs[[i, a, b, j]], {i, 3}, {j, 3}];
tsBlock[kind_, cs_, rhos_, k_, z_, flip_ : 1] := Module[{s = signs[kind], po, pi, nus = nu[k]},
   {po, pi} = pols[k][kind];
   Table[
    po[[sg]] . (rhos omega^2 IdentityMatrix[3] - k^2 cPaper[cs, 1, 1]
        + s[[1]] nus[[sg]] nus[[tau]] cPaper[cs, 3, 3]
        + s[[2]] k nus[[tau]] cPaper[cs, 1, 3]
        + flip s[[3]] k nus[[sg]] cPaper[cs, 3, 1]) . pi[[tau]]
     * I phase[kind, nus[[tau]], nus[[sg]], z]/(2 modOut[[sg]] nus[[sg]]),
    {sg, 3}, {tau, 3}]];

(* interaction-picture generator of a perturbation dA, blocks in the paper's roles:
   down amplitudes grow with z, up amplitudes are integrated from below, so
   tD = G_dd, rU = -G_ud, tU = -G_uu, rD = G_du. *)
genBlocks[dA_, k_, z_] := Module[{g = Inverse[eMat[k, z]] . Inverse[dMat[k]] . dA . dMat[k] . eMat[k, z]},
   <|"tD" -> g[[1 ;; 3, 1 ;; 3]], "rU" -> -g[[4 ;; 6, 1 ;; 3]],
     "tU" -> -g[[4 ;; 6, 4 ;; 6]], "rD" -> g[[1 ;; 3, 4 ;; 6]]|>];

maxRel[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];

Print["Background alpha=", alpha0, " beta=", beta0, " rho=", rho0, "; band H=", hBand, " m; f=10 Hz"];

(* =====================================================================
   0. The basis diagonalises the background system
   ===================================================================== *)
Print["\n0. Basis: D0^-1 A0 D0 = diag(I kz)"];
Do[
  r = maxRel[N[Inverse[dMat[k]] . aMat[c0, rho0, k] . dMat[k], prec], N[I DiagonalMatrix[Join[kzD[k], kzU[k]]], prec]];
  check["k = w p, p = " <> fmt[k/omega], r < 10^-30, "rel residual " <> fmt[r]],
  {k, omega pList}];

(* =====================================================================
   A. Their thin-layer coefficients ARE the linearised generator
   ===================================================================== *)
Print["\nA. Paper eqs 2.12/2.15 == interaction-picture blocks of A_lin - A0"];
SeedRandom[1989];
randAniso[] := Module[{v = RandomInteger[{-9, 9}, {6, 6}], vv, voigt},
   vv = (v + Transpose[v]) 10^8;
   voigt[i_, j_] := If[i == j, i, 9 - i - j];
   Table[vv[[voigt[i, j], voigt[k, n]]], {i, 3}, {j, 3}, {k, 3}, {n, 3}]];
perturbs = {
   {"isotropic (dlam 2 GPa, dmu 1 GPa, drho 100)", cIso[2 10^9, 10^9], 100},
   {"density only (drho 250)", cIso[0, 0], 250},
   {"random anisotropic c_s, drho -40", randAniso[], -40}};
Do[
  {label, cs, rhos} = pert;
  dA = aLin[cs, rhos, k] - aMat[c0, rho0, k];
  worstA = Max[Table[
     gb = genBlocks[dA, k, z];
     Max[Table[maxRel[N[tsBlock[kind, cs, rhos, k, z], prec], N[gb[kind], prec]],
       {kind, {"tD", "rU", "tU", "rD"}}]],
     {z, {0, 37, 100}}]];
  check[label <> ", p = " <> fmt[k/omega], worstA < 10^-30, "worst rel residual " <> fmt[worstA]],
  {pert, perturbs}, {k, omega pList}];

(* control: one flipped sign in rU must be caught *)
kc = omega pList[[2]]; cs = cIso[2 10^9, 10^9];
gb = genBlocks[aLin[cs, 100, kc] - aMat[c0, rho0, kc], kc, 37];
rCtl = maxRel[N[tsBlock["rU", cs, 100, kc, 37, -1], prec], N[gb["rU"], prec]];
check["control: sign of the k nu_o c_i31j term flipped in rU", rCtl > 10^-2,
  "rel residual " <> fmt[rCtl] <> " (must be O(1e-2) or larger)"];

(* the mechanism: A is linear in rho, not in the stiffness *)
Print["\n   Curvature of A in the contrast (second derivative at e = 0)"];
curv[cs_, rhos_, k_] := Module[{e}, Max[Abs[Flatten[N[
       D[aMat[c0 + e cs, rho0 + e rhos, k], {e, 2}] /. e -> 0, prec]]]]];
check["density direction: d^2A/de^2 = 0", curv[cIso[0, 0], 250, omega pList[[2]]] == 0,
  "max |entry| " <> fmt[curv[cIso[0, 0], 250, omega pList[[2]]]]];
check["stiffness direction: d^2A/de^2 != 0", curv[cIso[2 10^9, 10^9], 0, omega pList[[2]]] > 0,
  "max |entry| " <> fmt[curv[cIso[2 10^9, 10^9], 0, omega pList[[2]]]]];

(* =====================================================================
   B. Their Riccati equations (3.6a-d) for a uniform band
   ===================================================================== *)
rFromPropagator[a_, k_] := Module[{q, pr},
   pr = MatrixExp[N[a, prec] hBand];
   q = N[Inverse[eMat[k, hBand]], prec] . Inverse[N[dMat[k], prec]] . pr . N[dMat[k], prec];
   -Inverse[q[[4 ;; 6, 4 ;; 6]]] . q[[4 ;; 6, 1 ;; 3]]];

rTrompSnieder[cs_, rhos_, k_] := Module[{tD, rU, tU, rD, RU, RD, TD, TU, sol, z},
   {tD, rU, tU, rD} = Table[N[tsBlock[kind, cs, rhos, k, z], prec], {kind, {"tD", "rU", "tU", "rD"}}];
   sol = NDSolve[{
      RU'[z] == TU[z] . rU . TD[z],                                   (* 3.6a *)
      RD'[z] == rD + tD . RD[z] + RD[z] . tU + RD[z] . rU . RD[z],    (* 3.6b *)
      TD'[z] == tD . TD[z] + RD[z] . rU . TD[z],                      (* 3.6c *)
      TU'[z] == TU[z] . tU + TU[z] . rU . RD[z],                      (* 3.6d *)
      TD[0] == IdentityMatrix[3], TU[0] == IdentityMatrix[3],
      RD[0] == ConstantArray[0, {3, 3}], RU[0] == ConstantArray[0, {3, 3}]},
     {RU, RD, TD, TU}, {z, 0, hBand},
     WorkingPrecision -> 32, PrecisionGoal -> 22, AccuracyGoal -> 26, MaxSteps -> Infinity];
   RU[hBand] /. First[sol]];

invariants[r_, k_] := If[k == 0, {r[[1, 1]], r[[2, 2]], r[[3, 3]]},
   {r[[1, 1]], r[[2, 2]], r[[3, 3]], r[[1, 2]] r[[2, 1]]}];
invRel[a_, b_, k_] := Abs[invariants[a, k] - invariants[b, k]]/Abs[invariants[b, k]];

(* Python gate output (errors vs Kennett), 3 significant figures, order RPP RSS RSH [RPS*RSP] *)
pyErr = <|
   "moderate" -> {{3.97*^-2, 2.52*^-2, 2.52*^-2}, {4.18*^-2, 2.45*^-2, 2.51*^-2, 6.24*^-2},
     {5.16*^-2, 2.32*^-2, 2.48*^-2, 6.68*^-2}},
   "plus20" -> {{2.11*^-1, 2.42*^-1, 2.42*^-1}, {1.97*^-1, 2.99*^-1, 2.54*^-1, 5.53*^-1},
     {1.57*^-2, 5.99*^-2, 3.18*^-1, 1.63}},
   "minus20" -> {{1.88*^-1, 1.52*^-1, 1.52*^-1}, {1.86*^-1, 1.95*^-1, 1.67*^-1, 3.47*^-1},
     {2.71*^-2, 1.12*^-1, 2.17*^-1, 5.51*^-1}}|>;

bands = {
   {"density +10%", "none", cIso[0, 0], rho0/10},
   {"density +40%", "none", cIso[0, 0], 2 rho0/5},
   {"project moderate", "moderate", cIso[2 10^9, 10^9], 100},
   {"moduli +20%", "plus20", cIso[lam0/5, mu0/5], 0},
   {"moduli -20%", "minus20", cIso[-lam0/5, -mu0/5], 0}};

Print["\nB. Uniform band: their Riccati (3.6) vs exact propagators"];
Do[
  {label, key, cs, rhos} = band;
  Do[
   k = omega pList[[ip]];
   rTS = rTrompSnieder[cs, rhos, k];
   rLin = rFromPropagator[aLin[cs, rhos, k], k];
   rTrue = rFromPropagator[aMat[c0 + cs, rho0 + rhos, k], k];
   pTag = label <> ", p = " <> fmt[pList[[ip]]];
   eLin = maxRel[rTS, rLin];
   check[pTag <> ": equals the A_lin band", eLin < 10^-18, "rel " <> fmt[eLin]];
   eTrue = invRel[rTS, rTrue, k];
   If[key === "none",
    check[pTag <> ": equals the true band", Max[eTrue] < 10^-18, "worst " <> fmt[Max[eTrue]]],
    py = pyErr[key][[ip]];
    agree = Max[Abs[eTrue - py]/py];
    check[pTag <> ": true-band error matches Python/Kennett", agree < 6/1000,
     "errors " <> StringRiffle[fmt /@ eTrue, " "] <> "; max rel diff vs Python " <> fmt[agree]]],
   {ip, Length[pList]}],
  {band, bands}];

Print["\n", nPass, "/", nTotal, " checks passed"];
Exit[If[nPass == nTotal, 0, 1]];
