#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_IncidentS.wl  --  notebook 12 of the continuum-limit study.

   THE QUESTION.  Every convergence result so far (notebooks 5-10) is for an
   incident P wave; S appeared only as the converted wave R_PS.  Do the same
   orders and the same closed-form constants hold for an incident S wave?

   PART 1 -- normal incidence, 1-D.  For a normally incident S wave the layer is
   notebook 7's model with M -> mu, alpha -> beta, dM -> dmu (shear stress
   sigma_xz = mu du_x/dz, local term -delta/mu).  Notebook 7's definitions are
   loaded with exactly that substitution; nothing else changes.
     [1] orders: collocation 2, mean-only Galerkin 2, first moment 4;
     [2] Born order: the closed forms of notebook 10 with k = k_S,
         collocation 1 + (kd)^2/8 (R), 1 - (kd)^2/24 (T); mean only +-(kd)^2/12;
         first moment 1 - (kd)^4/720.
   PART 2 -- the full 9x9 Bloch-Galerkin of notebook 9, incident S.
     [3] at k_par = 0, incident SH: the first-moment voxel reproduces Part 1;
     [4] oblique SV (20, 30 deg; critical angle asin(beta/alpha) = 36.9 deg):
         orders of R_SS and R_SP, mean only and first moments;
     [5] oblique SH (20, 40 deg): orders of R_SH.
   References: the exact layer -- the P-SV propagator of notebook 9 with the
   SV-down state as the incident wave; for SH the 2x2 propagator of (u_y, s_yz).
   ============================================================================ *)

(* ---------------------------------------------------------------------------
   PART 1: notebook 7 with the S-wave substitution
   --------------------------------------------------------------------------- *)
nb7 = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_FourthOrder.wl", "Text"];
nb7 = StringTake[nb7, StringPosition[nb7, "Print[\"==== ContinuumLimit_FourthOrder ::"][[1, 1]] - 1];
If[! StringContainsQ[nb7, "dM = dLam + 2 dMu;\n"], Print["ABORT: notebook 7's parameter line has changed"]; Exit[1]];
nb7 = StringReplace[nb7, "dM = dLam + 2 dMu;\n" -> "dM = dMu; mP = mu; al = be;   (* the S-wave substitution *)\n"];
ToExpression[nb7, InputForm];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_IncidentS :: the voxel schemes for an incident S wave ===="];
Print["  PART 1: normal incidence (1-D), modulus mu, speed beta, contrast dmu"];

om = 300;
ex = N[exactScat[om], 40];
ladder = {1, 2, 4, 8, 16};
runs = Association[Table[sc -> Table[Abs[(solveLayer[om, n, If[sc == "G1", 2, 1], If[sc == "C", "C", "G"]] - ex)/ex],
      {n, ladder}], {sc, {"C", "G0", "G1"}}]];
Do[Print["   n = ", StringPadLeft[ToString[ladder[[i]]], 2], ":  (C) ", sci /@ runs["C"][[i]], "   (G0) ",
   sci /@ runs["G0"][[i]], "   (G1) ", sci /@ runs["G1"][[i]]], {i, Length[ladder]}];
order[sc_, col_] := N[Log[2, runs[sc][[-3, col]]/runs[sc][[-1, col]]]/2];
Do[Print["      order (n = 4 -> 16) ", sc, ":  R ", ToString[NumberForm[order[sc, 1], 3], OutputForm], ", T ",
   ToString[NumberForm[order[sc, 2], 3], OutputForm]], {sc, {"C", "G0", "G1"}}];
Print["  [1] S: collocation and mean-only second order, first moment fourth: ",
  chk[AllTrue[{order["C", 1], order["C", 2], order["G0", 1], order["G0", 2]}, 1.8 < # < 2.2 &] &&
    AllTrue[{order["G1", 1], order["G1", 2]}, 3.7 < # < 4.3 &]]];

(* [2] Born order: notebook 10's per-cell ratios, k = k_S *)
Clear[k, q, s, x];
i0[c_] := 2 hB Sinc[c hB];
i1[c_] := Integrate[(s/hB) Exp[I c s], {s, -hB, hB}];
ratio["C"] = i0[q]/i0[k + q];
ratio["G0"] = (i0[k]/(2 hB)) i0[q]/i0[k + q];
ratio["G1"] = ((i0[k]/(2 hB)) i0[q] + (i1[k]/(2 hB/3)) i1[q])/i0[k + q];
Module[{worst = 0, kk = om/be},
  Block[{dRho = 10^-4 dRho, dM = 10^-4 dM},
   Module[{exB = N[exactScat[om], 40]},
    Do[Module[{nb = If[sc == "G1", 2, 1], mode = If[sc == "C", "C", "G"], got, pr},
       Do[got = solveLayer[om, n, nb, mode]/exB - 1;
        pr = N[{ratio[sc] /. {q -> kk, k -> kk, hB -> dLayer/(2 n)}, ratio[sc] /. {q -> -kk, k -> kk, hB -> dLayer/(2 n)}}, 30] - 1;
        worst = Max[worst, Abs[(got - pr)/pr]], {n, {1, 2, 4, 8}}]], {sc, {"C", "G0", "G1"}}]]];
  Print["  [2] Born order (contrast x 1e-4): the P-wave closed forms with k = k_S, all three schemes, n = 1..8:",
   " worst |measured - closed form|/|closed form| = ", sci[worst], " -> ", chk[worst < 10^-2]]];
Print["      (collocation 1 + (k_S d)^2/8 in R and 1 - (k_S d)^2/24 in T; mean only +-(k_S d)^2/12; first",
  " moment 1 - (k_S d)^4/720)"];
g1Normal = solveLayer[om, 4, 2, "G"][[1]];   (* for [3] *)
exNormal = ex[[1]];

(* ---------------------------------------------------------------------------
   PART 2: notebook 9's Bloch-Galerkin, incident S
   --------------------------------------------------------------------------- *)
oksP1 = oks;   (* notebook 9 resets oks and chk when loaded *)
Clear[al, mP, dM, om];
nb9 = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_Oblique.wl", "Text"];
nb9 = StringTake[nb9, StringPosition[nb9, "pMax = 8;"][[1, 1]] - 1];
ToExpression[nb9, InputForm];
Print["  PART 2: the 9x9 Bloch-Galerkin voxel (notebook 9), incident S"];

(* the incident wave generalised: P, SV (in the x-z plane) or SH (along y), unit displacement *)
solveInc[omv_, kxv_, n_, useB_, pMax_, inc_] := Module[
  {d = dLayer/n, hh, zs, gl, ker, cpl, dd = N[delta9[omv]], nU, big, rhs, sol, kinc, uinc, psiInc, refl, bas, kw},
  hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
  bas = basis[[useB]]; nU = 9 Length[bas];
  gl = Flatten[Table[{kxv + 2 Pi p/d, 2 Pi q/d}, {p, -pMax, pMax}, {q, -pMax, pMax}], 1];
  ker = kernFun[omv, #[[1]], #[[2]]] & /@ N[gl, 20];
  cpl[m_, a_, b_] := cpl[m, a, b] = (1/d^2) Sum[Module[{kap = gl[[i]], kr = ker[[i]], lat, zz},
       lat = latF[a[[2]], kap[[1]], hh] latF[a[[3]], kap[[2]], hh] latF[b[[2]], -kap[[1]], hh] latF[b[[3]], -kap[[2]], hh];
       zz = Which[
         m == 0, kr[[1]] hh gramZ[[a[[1]] + 1, b[[1]] + 1]] +
          Sum[w[[1]] jPlus[w[[3]], hh][[a[[1]] + 1, b[[1]] + 1]] + w[[2]] jMinus[w[[3]], hh][[a[[1]] + 1, b[[1]] + 1]], {w, kr[[2 ;; 3]]}],
         m > 0, Sum[w[[1]] Exp[I w[[3]] m d] zMom[a[[1]], w[[3]], hh] zMom[b[[1]], -w[[3]], hh], {w, kr[[2 ;; 3]]}],
         True, Sum[w[[2]] Exp[-I w[[3]] m d] zMom[a[[1]], -w[[3]], hh] zMom[b[[1]], w[[3]], hh], {w, kr[[2 ;; 3]]}]];
       lat zz], {i, Length[gl]}];
  big = ArrayFlatten[Table[
     ArrayFlatten[Table[
       If[i == j && ai == bi, d^3 Times @@ ({1, 1/3}[[# + 1]] & /@ bas[[ai]]) IdentityMatrix[9], 0] -
        cpl[i - j, bas[[ai]], bas[[bi]]] . dd, {ai, Length[bas]}, {bi, Length[bas]}]], {i, n}, {j, n}]];
  kw = If[inc == "P", omv/al, omv/be];
  kinc = {Sqrt[kw^2 - kxv^2], kxv, 0};
  uinc = Switch[inc, "P", kinc/kw, "SV", {-kxv, kinc[[1]], 0}/kw, "SH", {0, 0, 1}];
  psiInc = rowsOf[uinc, kinc];
  rhs = Flatten[Table[psiInc Exp[I kinc[[1]] zs[[i]]] latF[bas[[ai, 2]], kxv, hh] latF[bas[[ai, 3]], 0, hh] zMom[bas[[ai, 1]], kinc[[1]], hh],
     {i, n}, {ai, Length[bas]}]];
  sol = LinearSolve[N[big, 20], N[rhs, 20]];
  refl = Table[Module[{kr = kernFun[omv, kxv, 0], w},
     w = kr[[1 + W]];
     Sum[(1/d^2) latF[bas[[bi, 2]], -kxv, hh] latF[bas[[bi, 3]], 0, hh] Exp[I w[[3]] zs[[j]]] zMom[bas[[bi, 1]], w[[3]], hh] *
       (w[[2]] . dd . sol[[nU (j - 1) + 9 (bi - 1) + 1 ;; nU (j - 1) + 9 bi]])[[1 ;; 3]], {j, n}, {bi, Length[bas]}]], {W, 2}];
  refl];   (* {S-mode displacement, P-mode displacement}, (z, x, y), at z = 0 *)

(* exact layer, incident SV: notebook 9's P-SV propagator with the SV-down state; unit displacement *)
exactReflSV[omv_, kxv_] := Module[{p = kxv/omv, ea, eb, pd, pu, su, sd, prop, sol, lam1 = lam + dLam, mu1 = mu + dMu,
    rh1 = rho + dRho},
   ea = eta[al, p]; eb = eta[be, p];
   pu = stateOf[lam, mu, {p, -ea}, p, -ea, omv];
   sd = stateOf[lam, mu, {eb, -p}, p, eb, omv]; su = stateOf[lam, mu, {-eb, -p}, p, -eb, omv];
   pd = stateOf[lam, mu, {p, ea}, p, ea, omv];
   prop = MatrixExp[N[aPSV[lam1, mu1, rh1, p, omv] dLayer, 30]];
   With[{sc = DiagonalMatrix[{1, 1, 1/(mu omv), 1/(mu omv)}]},
    sol = LinearSolve[sc . Transpose[{prop . pu, prop . su, -pd, -sd}], -sc . prop . sd]];
   (* slowness-normalised states: the unit-displacement SV carries the factor beta *)
   be {sol[[1]] {-ea, p, 0}, sol[[2]] {-p, -eb, 0}}];
(* exact layer, incident SH: d/dz (u, s) = {{0, 1/mu1}, {mu1 w^2 p^2 - rho1 w^2, 0}} (u, s) *)
exactReflSH[omv_, kxv_] := Module[{p = kxv/omv, eb, prop, rr, tt, mu1 = mu + dMu, rh1 = rho + dRho},
   eb = eta[be, p];
   prop = MatrixExp[N[{{0, 1/mu1}, {mu1 omv^2 p^2 - rh1 omv^2, 0}} dLayer, 30]];
   {rr, tt} = {rr, tt} /. First@Solve[prop . {1 + rr, mu I omv eb (1 - rr)} == tt {1, mu I omv eb}, {rr, tt}];
   rr {0, 0, 1}];

pMax = 8;
(* [3] normal incidence, SH: the Bloch voxel against Part 1 (the reflected displacement at z = 0) *)
Module[{r},
  r = solveInc[300, 0, 4, {1, 2}, 0, "SH"][[1, 3]];
  Print["  [3] k_par = 0, incident SH, z-moment voxel, n = 4: relative error ", sci[Abs[r/exactReflSH[300, 0][[3]] - 1]],
   "   Part 1's (G1): ", sci[Abs[g1Normal/exNormal - 1]], " -> ",
   chk[Abs[Abs[r/exactReflSH[300, 0][[3]] - 1]/Abs[g1Normal/exNormal - 1] - 1] < 10^-3]]];

relErr[a_, b_] := Norm[a - b]/Norm[b];
ladder = {1, 2, 4, 8};
ord[e_] := N[Log[2, e[[2]]/e[[4]]]/2];   (* n = 2 -> 8 *)
okSV = {}; okSH = {};
Do[Module[{om0 = 300, kx0, ex2, e0, e1},
   kx0 = N[om0/be Sin[th Degree], 20]; ex2 = exactReflSV[om0, kx0];
   e0 = Table[With[{r = solveInc[om0, kx0, n, {1}, pMax, "SV"]}, {relErr[r[[1]], ex2[[2]]], relErr[r[[2]], ex2[[1]]]}], {n, ladder}];
   e1 = Table[With[{r = solveInc[om0, kx0, n, {1, 2, 3}, pMax, "SV"]}, {relErr[r[[1]], ex2[[2]]], relErr[r[[2]], ex2[[1]]]}],
     {n, ladder}];
   Print["  SV incident at ", th, " deg, omega = ", om0, "; |error| / |R|, {R_SS, R_SP}:"];
   Do[Print["      n = ", ladder[[i]], ":  mean only ", sci /@ e0[[i]], "    mean + first moments ", sci /@ e1[[i]]], {i, 4}];
   Print["      orders (n = 2 -> 8): mean only ", ToString[NumberForm[ord[e0[[All, 1]]], 3], OutputForm], " / ",
    ToString[NumberForm[ord[e0[[All, 2]]], 3], OutputForm], "    first moments ",
    ToString[NumberForm[ord[e1[[All, 1]]], 3], OutputForm], " / ", ToString[NumberForm[ord[e1[[All, 2]]], 3], OutputForm]];
   AppendTo[okSV, AllTrue[{ord[e0[[All, 1]]], ord[e0[[All, 2]]]}, 1.8 < # < 2.2 &] &&
     AllTrue[{ord[e1[[All, 1]]], ord[e1[[All, 2]]]}, 3.7 < # < 4.3 &]]],
  {th, {20, 30}}];
Print["  [4] SV: mean-only second order, first moments FOURTH, in R_SS and R_SP: ", chk[And @@ okSV]];
Do[Module[{om0 = 300, kx0, ex2, e0, e1},
   kx0 = N[om0/be Sin[th Degree], 20]; ex2 = exactReflSH[om0, kx0];
   e0 = Table[relErr[solveInc[om0, kx0, n, {1}, pMax, "SH"][[1]], ex2], {n, ladder}];
   e1 = Table[relErr[solveInc[om0, kx0, n, {1, 2, 3}, pMax, "SH"][[1]], ex2], {n, ladder}];
   Print["  SH incident at ", th, " deg: mean only ", sci /@ e0, "  order ", ToString[NumberForm[ord[e0], 3], OutputForm],
    ";  first moments ", sci /@ e1, "  order ", ToString[NumberForm[ord[e1], 3], OutputForm]];
   AppendTo[okSH, 1.8 < ord[e0] < 2.2 && 3.7 < ord[e1] < 4.3]], {th, {20, 40}}];
Print["  [5] SH: mean-only second order, first moments FOURTH: ", chk[And @@ okSH]];

Print["==== ContinuumLimit_IncidentS (stage 12): ",
  If[And @@ Join[oksP1, oks], "ALL " <> ToString[Length[oksP1] + Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
