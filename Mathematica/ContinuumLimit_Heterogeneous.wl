#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Heterogeneous.wl  --  notebook 13 of the continuum-limit study.

   THE QUESTION.  Notebooks 5-12 establish the voxel schemes on a layer of
   IDENTICAL voxels.  A gridded Earth model is not uniform: it is a stack of model
   cells, each of constant properties, with real interfaces between them.  Do the
   orders -- second for the uniform-field voxel, fourth for the first-moment voxel
   -- survive those interfaces, and what does fourth order buy at a given accuracy?

   THE MODEL.  A layer of 8 model cells of 0.5 m (D = 4 m); in each, dlambda, dmu,
   drho drawn uniformly within +-15% of the background (SeedRandom[1996]): physical
   contrasts, a random stratification.  The voxels refine the MODEL grid: each model
   cell holds m planes of voxels, m = 1, 2, 4, ..., so voxels never straddle an
   interface -- the natural discretisation of a gridded model.  omega = 1500 rad/s:
   k_P D = 1.2, k_S D = 2.0.

   SCHEMES: notebook 7's 1-D chain (normal incidence, P and S; collocation C,
   mean-only G0, first moment G1) and notebook 9's 9x9 Bloch-Galerkin (P at 20 deg),
   both with the contrast now varying plane by plane -- nothing else changes.
   REFERENCE: the exact stratified layer, the product of the model cells'
   propagators (1-D: (u, s); oblique: notebook 9's P-SV system).

   CHECKS: [1] the stack reference reproduces the single-layer references when all
   cells are equal; [2] with equal cells the heterogeneous solvers reproduce
   notebooks 7 and 9; [3] 1-D P: C and G0 second order, G1 fourth; [4] 1-D S: the
   same; [5] oblique P, 20 deg: G0 second, G1 fourth in R_PP and R_PS.
   ============================================================================ *)

base = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
SeedRandom[1996];
nCells = 8; dCell = 1/2; DHet = nCells dCell; omH = 1500;
{al0, be0, rho0} = {5000, 3000, 2500};
mu0 = rho0 be0^2; lam0 = rho0 al0^2 - 2 mu0;
cells = Table[Rationalize[#, 10^-6] & /@ {RandomReal[{-0.15, 0.15}] lam0, RandomReal[{-0.15, 0.15}] mu0,
     RandomReal[{-0.15, 0.15}] rho0}, {nCells}];   (* {dlambda, dmu, drho} per model cell *)
allOks = {};

(* ===========================================================================
   PART 1: the 1-D chain, P or S, per-plane contrast
   =========================================================================== *)
run1D[mode_] := Module[{txt, res},
  txt = Import[base <> "ContinuumLimit_FourthOrder.wl", "Text"];
  txt = StringTake[txt, StringPosition[txt, "Print[\"==== ContinuumLimit_FourthOrder ::"][[1, 1]] - 1];
  If[mode == "S", txt = StringReplace[txt, "dM = dLam + 2 dMu;\n" -> "dM = dMu; mP = mu; al = be;\n"]];
  ToExpression[txt, InputForm];
  (* the modulus contrast of the channel: P  dlambda + 2 dmu, S  dmu *)
  modC[c_] := If[mode == "S", c[[2]], c[[1]] + 2 c[[2]]];
  (* notebook 7's solve, the contrast diag(w^2 drho_j, dM_j) now per plane; thickness dTot *)
  solveHet[om_, dqs_, nb_, scheme_, dTot_] := Module[{kk = om/al, n = Length[dqs], d, hh, zs, dQ, nU, blk, rhs, mat,
      sol, obs, cell},
    d = dTot/n; hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
    dQ[j_] := DiagonalMatrix[{om^2 dqs[[j, 1]], dqs[[j, 2]]}];
    nU = 2 nb;
    blk[i_, j_, a_, b_] := Which[
      scheme == "C",
      If[i == j,
       {{(I/(2 mP kk)) 2 (Exp[I kk hh] - 1)/(I kk), 0}, {0, -kk^2 (I/(2 mP kk)) 2 (Exp[I kk hh] - 1)/(I kk) - 1/mP}},
       With[{sg = Sign[zs[[i]] - zs[[j]]]}, kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[1]][-kk sg, hh]]],
      i == j, selfF[[a, b]][kk, hh],
      True, With[{sg = Sign[zs[[i]] - zs[[j]]]},
       kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[a]][kk sg, hh] momF[[b]][-kk sg, hh]]];
    cell[i_, a_] := If[scheme == "C", 1, 2 hh {1, 1/3}[[a]]];
    mat = ArrayFlatten[Table[ArrayFlatten[Table[
         If[i == j && a == b, cell[i, a] IdentityMatrix[2], 0] - blk[i, j, a, b] . dQ[j], {a, nb}, {b, nb}]],
       {i, n}, {j, n}]];
    rhs = Flatten[Table[If[scheme == "C", {gInc[om, zs[[i]]], I kk gInc[om, zs[[i]]]},
        gInc[om, zs[[i]]] momF[[a]][kk, hh] {1, I kk}], {i, n}, {a, nb}]];
    sol = LinearSolve[N[mat, 30], N[rhs, 30]];
    obs[zo_] := Sum[With[{sg = Sign[zo - zs[[j]]]},
       Sum[((kMat[kk, sg] Exp[I kk sg (zo - zs[[j]])] momF[[b]][-kk sg, hh]) . dQ[j] .
           sol[[nU (j - 1) + 2 (b - 1) + 1 ;; nU (j - 1) + 2 b]])[[1]], {b, nb}]], {j, n}];
    {obs[-1], obs[dTot + 2]}];
  (* the exact stratified layer: (u, s) propagators of the model cells, top to bottom *)
  exactRT1D[om_, cs_, dc_] := Module[{k0 = om/al, prop, rr, tt},
    prop = Dot @@ Reverse[Table[MatrixExp[N[{{0, 1/(mP + modC[c])}, {-(rho + c[[3]]) om^2, 0}} dc, 40]], {c, cs}]];
    {rr, tt} /. First@Solve[prop . {1 + rr, mP I k0 (1 - rr)} == tt {1, mP I k0}, {rr, tt}]];
  exactHet[om_, cs_, dc_] := Module[{k0 = om/al, rt = exactRT1D[om, cs, dc], dTot = Length[cs] dc},
    {rt[[1]] gInc[om, 0] Exp[-I k0 (-1)], rt[[2]] gInc[om, 0] Exp[I k0 (dTot + 2 - dTot)] - gInc[om, dTot + 2]}];
  planes[m_] := Flatten[Table[ConstantArray[{c[[3]], modC[c]}, m], {c, cells}], 1];

  (* [1], [2] equal cells: the stack and the solver reproduce notebook 7 (D = 2, observers -1 and 4) *)
  Module[{cEq = Table[{dLam, dMu, dRho}, {4}], exNb7 = N[exactScat[300], 30], exSt, s1, s2},
   exSt = exactHet[300, cEq, 1/2];
   s1 = solveLayer[300, 4, 2, "G"]; s2 = solveHet[300, ConstantArray[{dRho, dM}, 4], 2, "G", 2];
   Print["  [", mode, "] equal cells: stack vs notebook 7's layer ", sci[Max[Abs[(exSt - exNb7)/exNb7]]],
    ";  per-plane solver vs notebook 7's (G1, n = 4) ", sci[Max[Abs[(s1 - s2)/s1]]]];
   AppendTo[allOks, Max[Abs[(exSt - exNb7)/exNb7]] < 10^-20 && Max[Abs[(s1 - s2)/s1]] < 10^-20]];

  Module[{ex = exactHet[omH, cells, dCell], ms = {1, 2, 4, 8, 16}, errs, ord},
   errs = Association[Table[sc -> Table[Abs[(solveHet[omH, planes[m], If[sc == "G1", 2, 1], If[sc == "C", "C", "G"], DHet] -
           ex)/ex], {m, ms}], {sc, {"C", "G0", "G1"}}]];
   Print["  ", mode, ", normal incidence, omega = ", omH, "; |error|/|scattered| {R, T}, m voxel planes per model cell:"];
   Do[Print["      m = ", StringPadLeft[ToString[ms[[i]]], 2], " (n = ", StringPadLeft[ToString[nCells ms[[i]]], 3], "):  C ",
     sci /@ errs["C"][[i]], "   G0 ", sci /@ errs["G0"][[i]], "   G1 ", sci /@ errs["G1"][[i]]], {i, Length[ms]}];
   ord[sc_] := N[Log[2, errs[sc][[-3]]/errs[sc][[-1]]]/2];   (* m = 4 -> 16 *)
   Do[Print["      order (m = 4 -> 16) ", sc, ": R ", ToString[NumberForm[ord[sc][[1]], 3], OutputForm], ", T ",
     ToString[NumberForm[ord[sc][[2]], 3], OutputForm]], {sc, {"C", "G0", "G1"}}];
   res = <|"errs" -> errs, "ms" -> ms, "ord" -> Association[Table[sc -> ord[sc], {sc, {"C", "G0", "G1"}}]],
     "RT" -> exactRT1D[omH, cells, dCell]|>];
  res];

resP = run1D["P"];
Print["  [3] P: C and G0 second order, G1 FOURTH, across the model interfaces: ",
  pass[AppendTo[allOks, AllTrue[Flatten[{resP["ord"]["C"], resP["ord"]["G0"]}], 1.8 < # < 2.2 &] &&
      AllTrue[resP["ord"]["G1"], 3.7 < # < 4.3 &]]; Last[allOks]]];
resS = run1D["S"];
Print["  [4] S: the same: ",
  pass[AppendTo[allOks, AllTrue[Flatten[{resS["ord"]["C"], resS["ord"]["G0"]}], 1.8 < # < 2.2 &] &&
      AllTrue[resS["ord"]["G1"], 3.7 < # < 4.3 &]]; Last[allOks]]];

(* the cost of an accuracy: voxel planes needed for a reflection error of 1e-6, from each scheme's own
   asymptotic law fitted at m = 16 *)
Module[{need},
  need[r_, sc_, p_] := Module[{e = r["errs"][sc][[-1, 1]], n16 = nCells 16}, N[n16 (e/10^-6)^(1/p)]];
  Print["  planes of voxels for a reflection error of 1e-6 (P): collocation ", Round[need[resP, "C", 2]], ", mean only ",
   Round[need[resP, "G0", 2]], ", first moment ", Round[need[resP, "G1", 4]],
   "   (unknowns per plane 9 vs 36 in 3-D)"];
  costP = {need[resP, "G0", 2], need[resP, "G1", 4]}];

(* ===========================================================================
   PART 2: the 9x9 Bloch-Galerkin, P at 20 degrees, per-plane contrast
   =========================================================================== *)
Clear[al, be, rho, mu, mP, lam, dLam, dMu, dRho, dM, om];
nb9 = Import[base <> "ContinuumLimit_Oblique.wl", "Text"];
ToExpression[StringTake[nb9, StringPosition[nb9, "pMax = 8;"][[1, 1]] - 1], InputForm];
delta9c[omv_, c_] := Module[{c6h = Table[Which[v <= 3 && w <= 3, c[[1]] + If[v == w, 2 c[[2]], 0], v == w, 2 c[[2]], True, 0],
      {v, 6}, {w, 6}]}, ArrayFlatten[{{omv^2 c[[3]] IdentityMatrix[3], 0}, {0, c6h}}]];
solveObliqueHet[omv_, kxv_, cs_, useB_, pMax_, dTot_] := Module[
  {n = Length[cs], d, hh, zs, gl, ker, cpl, dd, nU, big, rhs, sol, kinc, uinc, psiInc, refl, bas},
  d = dTot/n; hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
  dd = Table[N[delta9c[omv, c]], {c, cs}];
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
        cpl[i - j, bas[[ai]], bas[[bi]]] . dd[[j]], {ai, Length[bas]}, {bi, Length[bas]}]], {i, n}, {j, n}]];
  kinc = {Sqrt[(omv/al)^2 - kxv^2], kxv, 0}; uinc = kinc/(omv/al);
  psiInc = rowsOf[uinc, kinc];
  rhs = Flatten[Table[psiInc Exp[I kinc[[1]] zs[[i]]] latF[bas[[ai, 2]], kxv, hh] latF[bas[[ai, 3]], 0, hh] zMom[bas[[ai, 1]], kinc[[1]], hh],
     {i, n}, {ai, Length[bas]}]];
  sol = LinearSolve[N[big, 20], N[rhs, 20]];
  refl = Table[Module[{kr = kernFun[omv, kxv, 0], w},
     w = kr[[1 + W]];
     Sum[(1/d^2) latF[bas[[bi, 2]], -kxv, hh] latF[bas[[bi, 3]], 0, hh] Exp[I w[[3]] zs[[j]]] zMom[bas[[bi, 1]], w[[3]], hh] *
       (w[[2]] . dd[[j]] . sol[[nU (j - 1) + 9 (bi - 1) + 1 ;; nU (j - 1) + 9 bi]])[[1 ;; 3]], {j, n}, {bi, Length[bas]}]], {W, 2}];
  refl];
exactRHet[omv_, kxv_, cs_, dc_] := Module[{p = kxv/omv, ea, eb, pd, pu, su, sd, prop, sol},
   ea = eta[al, p]; eb = eta[be, p];
   pd = stateOf[lam, mu, {p, ea}, p, ea, omv]; pu = stateOf[lam, mu, {p, -ea}, p, -ea, omv];
   sd = stateOf[lam, mu, {eb, -p}, p, eb, omv]; su = stateOf[lam, mu, {-eb, -p}, p, -eb, omv];
   prop = Dot @@ Reverse[Table[MatrixExp[N[aPSV[lam + c[[1]], mu + c[[2]], rho + c[[3]], p, omv] dc, 30]], {c, cs}]];
   With[{sc = DiagonalMatrix[{1, 1, 1/(mu omv), 1/(mu omv)}]},
    LinearSolve[sc . Transpose[{prop . pu, prop . su, -pd, -sd}], -sc . prop . pd]][[1 ;; 2]]];
exactReflHet[omv_, kxv_, cs_, dc_] := Module[{p = kxv/omv, ea, eb, pd, pu, su, sd, prop, sol},
   ea = eta[al, p]; eb = eta[be, p];
   pd = stateOf[lam, mu, {p, ea}, p, ea, omv]; pu = stateOf[lam, mu, {p, -ea}, p, -ea, omv];
   sd = stateOf[lam, mu, {eb, -p}, p, eb, omv]; su = stateOf[lam, mu, {-eb, -p}, p, -eb, omv];
   prop = Dot @@ Reverse[Table[MatrixExp[N[aPSV[lam + c[[1]], mu + c[[2]], rho + c[[3]], p, omv] dc, 30]], {c, cs}]];
   With[{sc = DiagonalMatrix[{1, 1, 1/(mu omv), 1/(mu omv)}]},
    sol = LinearSolve[sc . Transpose[{prop . pu, prop . su, -pd, -sd}], -sc . prop . pd]];
   al {sol[[1]] {-ea, p, 0}, sol[[2]] {-p, -eb, 0}}];

(* [2] (oblique) equal cells: the per-plane solver and the stack reproduce notebook 9 (D = 2) *)
Module[{kx0 = N[300/al Sin[20 Degree], 20], cEq = Table[{dLam, dMu, dRho}, {4}], a, b, ea, eb},
  a = solveOblique[300, kx0, 4, {1, 2, 3}, 2]; b = solveObliqueHet[300, kx0, cEq, {1, 2, 3}, 2, 2];
  ea = exactRefl[300, kx0]; eb = exactReflHet[300, kx0, cEq, 1/2];
  Print["  [oblique] equal cells: per-plane solver vs notebook 9 ", sci[relErr[Flatten[a], Flatten[b]]],
   ";  stack vs notebook 9's layer ", sci[relErr[Flatten[eb], Flatten[ea]]]];
  AppendTo[allOks, relErr[Flatten[a], Flatten[b]] < 10^-12 && relErr[Flatten[eb], Flatten[ea]] < 10^-12]];
Print["  [1]-[2] references and solvers reproduce notebooks 7 and 9 with equal cells: ", pass[And @@ allOks[[{1, 3, 5}]]]];

Module[{kx0 = N[omH/al Sin[20 Degree], 20], ex, ms = {1, 2, 4}, e0, e1, ord, pl},
  ex = exactReflHet[omH, kx0, cells, dCell];
  pl[m_] := Flatten[Table[ConstantArray[c, m], {c, cells}], 1];
  e0 = Table[With[{r = solveObliqueHet[omH, kx0, pl[m], {1}, 8, DHet]}, {relErr[r[[2]], ex[[1]]], relErr[r[[1]], ex[[2]]]}], {m, ms}];
  e1 = Table[With[{r = solveObliqueHet[omH, kx0, pl[m], {1, 2, 3}, 8, DHet]}, {relErr[r[[2]], ex[[1]]], relErr[r[[1]], ex[[2]]]}],
    {m, ms}];
  ord[e_] := N[Log[2, e[[2]]/e[[3]]]];   (* m = 2 -> 4 *)
  Print["  P at 20 deg, omega = ", omH, "; |error|/|R| {R_PP, R_PS}:"];
  Do[Print["      m = ", ms[[i]], " (n = ", nCells ms[[i]], "):  mean only ", sci /@ e0[[i]], "   first moments ", sci /@ e1[[i]]],
   {i, 3}];
  Print["      orders (m = 2 -> 4): mean only ", ToString[NumberForm[ord[e0[[All, 1]]], 3], OutputForm], " / ",
   ToString[NumberForm[ord[e0[[All, 2]]], 3], OutputForm], "   first moments ", ToString[NumberForm[ord[e1[[All, 1]]], 3], OutputForm],
   " / ", ToString[NumberForm[ord[e1[[All, 2]]], 3], OutputForm]];
  AppendTo[allOks, AllTrue[{ord[e0[[All, 1]]], ord[e0[[All, 2]]]}, 1.7 < # < 2.3 &] &&
    AllTrue[{ord[e1[[All, 1]]], ord[e1[[All, 2]]]}, 3.6 < # < 4.4 &]];
  Print["  [5] oblique P: mean only second order, first moments FOURTH, R_PP and R_PS: ", pass[Last[allOks]]]];

(* exports for the independent checks: the exact stack coefficients (against Kennett) and raw discrete
   reflections at m = 1, |p|, |q| <= 2 (against the Python implementation, per-plane contrast) *)
reim[z_] := {Re[z], Im[z]};
rawCases = Flatten[Table[Module[{kx = If[th == 0, 0, N[omH/al Sin[th Degree], 20]], pm = If[th == 0, 0, 2], r},
     r = solveObliqueHet[omH, kx, cells, bset, pm, DHet];
     <|"incident" -> "P", "omega" -> omH, "theta_deg" -> th, "kx" -> N[kx], "n" -> nCells, "basis" -> bset, "p_max" -> pm,
      "refl_S" -> Map[reim, N[r[[1]]]], "refl_P" -> Map[reim, N[r[[2]]]]|>], {th, {0, 20}}, {bset, {{1}, {1, 2, 3}}}], 1];
Export[base <> "ContinuumLimit_heterogeneous_ref.json",
  <|"alpha" -> al0, "beta" -> be0, "rho" -> rho0, "omega" -> omH, "D" -> N[DHet], "cell_thickness" -> N[dCell],
   "cells_dlambda_dmu_drho" -> N[cells],
   "exact_normal_P_RT" -> Map[reim, N[resP["RT"]]], "exact_normal_S_RT" -> Map[reim, N[resS["RT"]]],
   "exact_oblique_20deg_RPP_RPS" -> Map[reim, N[exactRHet[omH, N[omH/al Sin[20 Degree], 20], cells, dCell]]],
   "cases" -> rawCases|>, "RawJSON"];
Print["==== ContinuumLimit_Heterogeneous (stage 13): ",
  If[And @@ allOks, "ALL " <> ToString[Length[allOks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
