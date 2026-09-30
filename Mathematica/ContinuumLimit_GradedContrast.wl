#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_GradedContrast.wl  --  notebook 15 of the continuum-limit study.

   THE QUESTION.  A volume-integral calculation samples the medium as cells of
   constant properties.  If the medium varies smoothly, what does that sampling
   cost, and what does a cell whose CONTRAST also carries a local Legendre
   expansion recover?

   THE SCHEME.  Notebook 14's voxel carries its field to Legendre degree p.  Here
   the contrast of cell j is its Legendre projection to degree r,
       Delta(z) ~ sum_c f_c^(j) P_c(s/h) Delta_bar,  f_c^(j) = (2c+1)/(2h) int f(z_j + s) P_c(s/h) ds,
   and the product with the field is re-expanded by Legendre linearisation,
       P_c P_b = sum_e A_{cbe} P_e,  A_{cbe} = (2e+1)/2 int_{-1}^{1} P_c P_b P_e dx,
   so the Galerkin coupling is the existing kernel block of degree e <= p + r:
       int int phi_a K Delta phi_b = sum_{c,e} f_c A_{cbe} B_{ae} Delta_bar.
   Nothing new is needed of the kernel; only its moments to higher degree.

   THE CLAIM (stated before any run).  Order of convergence = min(2p + 2, 2r + 2):
   constant-per-cell contrast (r = 0) caps every field basis at second order; a
   contrast projected to the field's degree (r = p) restores 2p + 2.

   CHECKS: [1] the exact graded layer (high-precision ODE) reproduces the closed
   form for a constant contrast and conserves energy for the graded ones, and is
   converged in its own precision; [2] with a constant contrast the graded solver
   IS notebook 14's; [3] linear profile: (p, r) = (0,0) 2, (1,0) 2, (1,1) 4,
   (2,1) 6 (r = 1 is exact for a linear profile); [4] smooth profile
   1 + sin(2 pi z / D) / 2: (1,0) 2, (1,1) 4, (2,1) 4, (2,2) 6.
   z down; e^{-i w t}; SI units; 50-digit arithmetic.
   ============================================================================ *)

(* notebook 14's definitions (notebook 7's model and the degree-p solver), its checks silenced *)
nbText = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_SecondMoment.wl", "Text"];
Block[{Print = Null &},
  ToExpression[StringTake[nbText, StringPosition[nbText,
       "(* ---------------------------------------------------------------------------\n   [2] regression"][[1, 1]] - 1],
    InputForm]];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_GradedContrast :: cells whose contrast carries a Legendre expansion ===="];

(* kernel moments and same-cell blocks to degree 4 (field degree 2 times contrast degree 2) *)
nbMax = 5;
Clear[s, t, k, h];
leg = Table[With[{j = j}, Function[Evaluate[Expand[LegendreP[j, #/h]]]]], {j, 0, nbMax - 1}];
momF = Table[Function[{cc, hh}, Evaluate[mom[a, cc] /. h -> hh]], {a, nbMax}];
selfF = Table[Function[{kk, hh}, Evaluate[Simplify[selfBlock[a, b]] /. {k -> kk, h -> hh}]], {a, nbMax}, {b, nbMax}];
(* Legendre linearisation, 1-based: A[[c, b, e]] for P_{c-1} P_{b-1} = sum_e A P_{e-1} *)
lin = Table[(2 e - 1)/2 Integrate[LegendreP[c - 1, x] LegendreP[b - 1, x] LegendreP[e - 1, x], {x, -1, 1}],
   {c, 3}, {b, 3}, {e, nbMax}];

(* ---- the graded layer: contrast profile f(z) on [0, D], Delta(z) = f(z) Delta_bar ---- *)
prec = 50;
(* the exact layer: state (u, y), y = M u' / mP, so that both components are O(1) and the ODE's
   error control is meaningful; y' = -w^2 rho u / mP, integrated through the layer *)
rtGraded[om_, f_, wp_] := Module[{k0 = om/al, mz, rz, sol1, sol2, pm, rr, tt, uu, tz},
   mz[z_] := mP + dM f[z]; rz[z_] := rho + dRho f[z];
   {sol1, sol2} = Table[NDSolveValue[{uu'[z] == mP tz[z]/mz[z], tz'[z] == -om^2 rz[z] uu[z]/mP, uu[0] == ic[[1]], tz[0] == ic[[2]]},
       {uu[dLayer], tz[dLayer]}, {z, 0, dLayer}, WorkingPrecision -> wp, PrecisionGoal -> wp/2,
       AccuracyGoal -> wp/2, MaxSteps -> Infinity, Method -> "Extrapolation"], {ic, {{1, 0}, {0, 1}}}];
   pm = Transpose[{sol1, sol2}];
   (* above: u = 1 + R, y = i k0 (1 - R); below: u = T, y = i k0 T *)
   {rr, tt} /. First@Solve[pm . {1 + rr, I k0 (1 - rr)} == {tt, I k0 tt}, {rr, tt}]];
scatFromRT[om_, {rr_, tt_}] := Module[{k0 = om/al},
   {rr gInc[om, 0] Exp[-I k0 zObsR], tt gInc[om, 0] Exp[I k0 (zObsT - dLayer)] - gInc[om, zObsT]}];

(* the discrete graded layer: field degree nb - 1, contrast degree nr - 1 *)
solveGraded[om_, n_, nb_, nr_, f_] := Module[{kk = N[om/al, prec], d = dLayer/n, hh, hE, zs, dQ, fc, kb, blk, rhs, mat, sol,
    obs},
   hE = d/2; hh = N[hE, prec]; zs = Table[(j - 1/2) d, {j, n}];
   dQ = N[DiagonalMatrix[{om^2 dRho, dM}], prec];
   (* the profile's Legendre coefficients on each cell, exactly *)
   fc = Table[N[(2 c - 1)/(2 hE) Integrate[f[zs[[j]] + ss] LegendreP[c - 1, ss/hE], {ss, -hE, hE}], prec],
     {j, n}, {c, nr}];
   kb[i_, j_, a_, e_] := If[i == j, selfF[[a, e]][kk, hh],
     With[{sg = Sign[zs[[i]] - zs[[j]]]},
      kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[a]][kk sg, hh] momF[[e]][-kk sg, hh]]];
   blk[i_, j_, a_, b_] := Sum[fc[[j, c]] lin[[c, b, e]] kb[i, j, a, e], {c, nr}, {e, nb + nr - 1}] . dQ;
   mat = ArrayFlatten[Table[ArrayFlatten[Table[
        If[i == j && a == b, 2 hh/(2 a - 1) IdentityMatrix[2], 0] - blk[i, j, a, b], {a, nb}, {b, nb}]],
      {i, n}, {j, n}]];
   rhs = Flatten[Table[gInc[om, zs[[i]]] momF[[a]][kk, hh] {1, I kk}, {i, n}, {a, nb}]];
   sol = LinearSolve[N[mat, prec], N[rhs, prec]];
   obs[zo_] := Sum[With[{sg = Sign[zo - zs[[j]]]},
      Sum[fc[[j, c]] lin[[c, b, e]] ((kMat[kk, sg] Exp[I kk sg (zo - zs[[j]])] momF[[e]][-kk sg, hh]) . dQ .
           sol[[2 nb (j - 1) + 2 (b - 1) + 1 ;; 2 nb (j - 1) + 2 b]])[[1]],
       {b, nb}, {c, nr}, {e, nb + nr - 1}]], {j, n}];
   {obs[zObsR], obs[zObsT]}];

om = 300;
fConst = (1 &);
fLin = Function[z, 1 + (2 z/dLayer - 1)/2];
fSin = Function[z, 1 + Sin[2 Pi z/dLayer]/2];

(* ---------------------------------------------------------------------------
   [1] the exact graded layer
   --------------------------------------------------------------------------- *)
rtC = rtGraded[om, fConst, 90];
devC = Max[Abs[rtC - rtExact[om]]/Abs[rtExact[om]]];
refs = Association["linear" -> rtGraded[om, fLin, 90], "smooth" -> rtGraded[om, fSin, 90]];
energy = Max[Table[Abs[Abs[refs[p][[1]]]^2 + Abs[refs[p][[2]]]^2 - 1], {p, Keys[refs]}]];
conv = Max[Table[Max[Abs[rtGraded[om, If[p == "linear", fLin, fSin], 120] - refs[p]]/Abs[refs[p]]], {p, Keys[refs]}]];
Print["  [1] exact graded layer (ODE, Bulirsch-Stoer, 90 digits): constant contrast vs closed form ", sci[devC],
  "; |R|^2 + |T|^2 - 1 ", sci[energy], "; 90 vs 120 digits ", sci[conv], " -> ",
  chk[devC < 10^-30 && energy < 10^-30 && conv < 10^-30]];
exRef = Association[Table[p -> scatFromRT[om, refs[p]], {p, Keys[refs]}]];

(* ---------------------------------------------------------------------------
   [2] a constant contrast is notebook 14's voxel
   --------------------------------------------------------------------------- *)
reg = Max[Table[Max[Abs[(solveGraded[om, n, nb, 3, fConst] - solveLayerP[om, n, nb])/solveLayerP[om, n, nb]]],
    {n, {1, 2, 4}}, {nb, 3}]];
Print["  [2] constant contrast, contrast degree 2 carried: graded solver vs notebook 14, n = 1, 2, 4, p = 0..2: ",
  sci[reg], " -> ", chk[reg < 10^-35]];

(* ---------------------------------------------------------------------------
   [3], [4] orders against the prediction min(2p + 2, 2r + 2)
   --------------------------------------------------------------------------- *)
ladder = {2, 4, 8, 16};
exported = {};
(* predicted order: 2p + 2 if the degree-r projection represents the profile exactly (r >= its
   polynomial degree), otherwise min(2p + 2, 2r + 2) *)
orderRun[name_, prof_, profDeg_, cases_] := Table[Module[{p = pr[[1]], r = pr[[2]], e, o, want},
     e = Table[Abs[(solveGraded[om, n, p + 1, r + 1, prof] - exRef[name])/exRef[name]], {n, ladder}];
     o = N[(Log[2, e[[-3]]/e[[-2]]] + Log[2, e[[-2]]/e[[-1]]])/2];
     want = If[r >= profDeg, 2 p + 2, Min[2 p + 2, 2 r + 2]];
     AppendTo[exported, <|"profile" -> name, "p" -> p, "r" -> r, "errors_R_T" -> N[e, 16]|>];
     Print["      p = ", p, ", r = ", r, ":  n = 2..16 (R) ", sci /@ e[[All, 1]], "   order ", fmt[o[[1]], 4], " / ",
      fmt[o[[2]], 4], "   predicted ", want];
     AllTrue[o, Abs[# - want] < 0.2 &]], {pr, cases}];
Print["  [3] linear profile 1 + (2z/D - 1)/2, omega = ", om, ", order R / T:"];
ok3 = orderRun["linear", fLin, 1, {{0, 0}, {1, 0}, {1, 1}, {2, 1}}];
Print["      as predicted: ", chk[And @@ ok3]];
Print["  [4] smooth profile 1 + sin(2 pi z/D)/2, omega = ", om, ", order R / T:"];
ok4 = orderRun["smooth", fSin, Infinity, {{1, 0}, {1, 1}, {2, 1}, {2, 2}}];
Print["      as predicted: ", chk[And @@ ok4]];

(* the data of the independent Python cross-check: the errors, and numerical samples of every closed form
   the scheme uses, so the closed forms themselves are tested and not only the assembled result *)
cplx[z_] := {Re[#], Im[#]} &[N[z, 30]];
samplePts = {{3/50, 1/16}, {3/50, 1/2}, {6/25, 1}};   (* (k, h) *)
closed = Table[<|"k" -> N[pt[[1]], 30], "h" -> N[pt[[2]], 30],
    (* int_{-h}^{h} P_{a-1}(s/h) e^{i c s} ds at c = +k and c = -k *)
    "moment_plus" -> Table[cplx[momF[[a]][pt[[1]], pt[[2]]]], {a, nbMax}],
    "moment_minus" -> Table[cplx[momF[[a]][-pt[[1]], pt[[2]]]], {a, nbMax}],
    (* same-cell block int int P_{a-1} K P_{b-1}, including the local term, as a 2 x 2 complex matrix *)
    "self" -> Table[Map[cplx, selfF[[a, b]][pt[[1]], pt[[2]]], {2}], {a, nbMax}, {b, nbMax}]|>, {pt, samplePts}];
Export[DirectoryName[$InputFileName] <> "ContinuumLimit_graded_contrast_data.json",
  <|"omega" -> om, "n" -> ladder, "runs" -> exported, "mP" -> N[mP, 30],
    "linearisation" -> N[lin, 30], "closed_forms" -> closed|>, "RawJSON"];

Print["==== ContinuumLimit_GradedContrast (stage 15): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
