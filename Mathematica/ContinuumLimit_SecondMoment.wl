#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_SecondMoment.wl  --  notebook 14 of the continuum-limit study.

   THE CLAIM.  Notebook 7 gave the voxel its mean and first Legendre moment and
   measured fourth order; notebook 10 derived its leading Born error, -(k d)^4/720.
   Carrying Legendre moments up to degree p should converge at order 2p + 2, with
   the leading Born error, per cell and so for every n,
       discrete / exact - 1 = s_p c_p (k d)^{2p+2},
       c_p = ((p+1)!)^2 / ((2p+2)! (2p+3)!),
       s_p = (-1)^p in reflection and -1 in transmission,
   so c_0 = 1/12, c_1 = 1/720, c_2 = 1/100800 (the second-moment voxel, sixth order),
   c_3 = 1/25401600.  (Notebook 10's projection ratio, extended to degree p.)

   CHECKS: [1] the closed form, symbolically, for p = 0 .. 5: every lower power of
   k d vanishes and the leading coefficient is s_p c_p; [2] the general-degree
   solver reproduces notebook 7's mean-only and first-moment voxels; [3] the chain
   at Born order (contrast x 1e-6) against the closed form, degrees 2 and 3;
   [4] at the full contrast the second-moment voxel converges at sixth order in
   reflection and transmission, at three frequencies; [5] the third-moment voxel
   at eighth order; [6] ONE plane of degree-p voxels against the exact layer as a series in its
   thickness D: identical through D^{2p+2}, first difference at D^{2p+3} (notebook 2c's thin-layer
   series, reproduced term by term), for a stiffer and a softer layer.  Everything in 40-digit
   arithmetic: the errors reach 1e-22.
   ============================================================================ *)

(* notebook 7's model and chain (its definitions, not its checks), loaded first: it defines its own oks/chk *)
nbText = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_FourthOrder.wl", "Text"];
ToExpression[StringTake[nbText, StringPosition[nbText, "Print[\"==== ContinuumLimit_FourthOrder ::"][[1, 1]] - 1], InputForm];
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
fmt[x_, d_] := ToString[NumberForm[N[x], d], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_SecondMoment :: voxels that carry Legendre moments up to degree p ===="];

(* ---------------------------------------------------------------------------
   [1] the Born ratio for degree p, and its series in x = k d
   --------------------------------------------------------------------------- *)
Clear[k, q, h, s, x];
iP[j_, c_] := Integrate[LegendreP[j, s/h] Exp[I c s], {s, -h, h}];
(* the exact cell integral int e^{i (k+q) s} ds in the form regular at k + q = 0 (transmission) *)
ratioP[p_] := Sum[iP[j, k] iP[j, q]/(2 h/(2 j + 1)), {j, 0, p}]/(2 h Sinc[(k + q) h]);
cP[p_] := ((p + 1)!)^2/((2 p + 2)! (2 p + 3)!);
Print["  [1] discrete / exact - 1 at Born order, per cell (x = k d):"];
closedOK = Table[Module[{serR, serT, lowR, lowT},
    serR = Normal[Series[ratioP[p] /. q -> k /. h -> x/(2 k), {x, 0, 2 p + 2}]] - 1;
    serT = Normal[Series[ratioP[p] /. q -> -k /. h -> x/(2 k), {x, 0, 2 p + 2}]] - 1;
    serR = Simplify[serR, x > 0]; serT = Simplify[serT, x > 0];
    Print["      p = ", p, ":  reflection ", InputForm[serR], "    transmission ", InputForm[serT]];
    serR === (-1)^p cP[p] x^(2 p + 2) && serT === -cP[p] x^(2 p + 2)], {p, 0, 5}];
Print["      c_p = ((p+1)!)^2/((2p+2)!(2p+3)!) = ", cP /@ Range[0, 5]];
Print["      lower powers vanish, leading term (-1)^p c_p x^(2p+2) in R and -c_p x^(2p+2) in T, p = 0..5: ",
  chk[And @@ closedOK]];

(* ---------------------------------------------------------------------------
   notebook 7's chain for any number of Legendre functions
   --------------------------------------------------------------------------- *)
om = 300; ex = N[exactScat[om], 40];
ref7 = Table[solveLayer[om, n, nb, "G"], {nb, 2}, {n, {1, 2, 4}}];   (* notebook 7's own tables, before they change *)

nbMax = 4;   (* P0 .. P3 *)
Clear[s, t, k, h];
leg = Table[With[{j = j}, Function[Evaluate[Expand[LegendreP[j, #/h]]]]], {j, 0, nbMax - 1}];
momF = Table[Function[{cc, hh}, Evaluate[mom[a, cc] /. h -> hh]], {a, nbMax}];
selfF = Table[Function[{kk, hh}, Evaluate[Simplify[selfBlock[a, b]] /. {k -> kk, h -> hh}]], {a, nbMax}, {b, nbMax}];

solveLayerP[om_, n_, nb_] := Module[{kk = om/al, d = dLayer/n, hh, zs, dQ, nU, blk, rhs, mat, sol, obs},
   hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
   dQ = DiagonalMatrix[{om^2 dRho, dM}];
   nU = 2 nb;
   blk[i_, j_, a_, b_] := If[i == j, selfF[[a, b]][kk, hh],
     With[{sg = Sign[zs[[i]] - zs[[j]]]},
      kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[a]][kk sg, hh] momF[[b]][-kk sg, hh]]];
   (* int_{-h}^{h} P_{a-1}^2 = 2h/(2a - 1) *)
   mat = ArrayFlatten[Table[ArrayFlatten[Table[
        If[i == j && a == b, 2 hh/(2 a - 1) IdentityMatrix[2], 0] - blk[i, j, a, b] . dQ, {a, nb}, {b, nb}]],
      {i, n}, {j, n}]];
   rhs = Flatten[Table[gInc[om, zs[[i]]] momF[[a]][kk, hh] {1, I kk}, {i, n}, {a, nb}]];
   sol = LinearSolve[N[mat, 40], N[rhs, 40]];
   obs[zo_] := Sum[With[{sg = Sign[zo - zs[[j]]]},
      Sum[((kMat[kk, sg] Exp[I kk sg (zo - zs[[j]])] momF[[b]][-kk sg, hh]) . dQ .
          sol[[nU (j - 1) + 2 (b - 1) + 1 ;; nU (j - 1) + 2 b]])[[1]], {b, nb}]], {j, n}];
   {obs[zObsR], obs[zObsT]}];

(* ---------------------------------------------------------------------------
   [2] regression: degrees 0 and 1 are notebook 7's voxels
   --------------------------------------------------------------------------- *)
dev = Max[Table[Abs[(solveLayerP[om, {1, 2, 4}[[m]], nb] - ref7[[nb, m]])/ref7[[nb, m]]], {nb, 2}, {m, 3}]];
Print["  [2] general-degree solver vs notebook 7 (mean-only and first-moment, n = 1, 2, 4): max relative difference ",
  sci[dev], " -> ", chk[dev < 10^-30]];

(* ---------------------------------------------------------------------------
   [3] the chain at Born order against the closed form, degrees 2 and 3
   --------------------------------------------------------------------------- *)
{dLam0, dMu0, dRho0} = {dLam, dMu, dRho};
eps = 10^-6;
{dLam, dMu, dRho} = eps {dLam0, dMu0, dRho0}; dM = dLam + 2 dMu;
kk = om/al; exB = N[exactScat[om], 40];
worst = 0;
Print["  [3] the chain at Born order (contrast x 1e-6), omega = ", om, ": (discrete/exact - 1) measured (closed form)"];
Do[Do[Module[{d = dLayer/n, got, predR, predT},
     got = solveLayerP[om, n, p + 1]/exB - 1;
     predR = N[ratioP[p] /. {q -> kk, k -> kk, h -> d/2}, 30] - 1;
     predT = N[ratioP[p] /. {q -> -kk, k -> kk, h -> d/2}, 30] - 1;
     worst = Max[worst, Abs[got[[1]] - predR]/Abs[predR], Abs[got[[2]] - predT]/Abs[predT]];
     Print["      p = ", p, ", n = ", n, ":  R ", sci[Re[got[[1]]]], " (", sci[Re[predR]], ")    T ",
      sci[Re[got[[2]]]], " (", sci[Re[predT]], ")"]], {n, {1, 2, 4}}], {p, {2, 3}}];
Print["      worst |measured - closed form| / |closed form| = ", sci[worst], " (the O(contrast) remainder) -> ",
  chk[worst < 10^-2]];
{dLam, dMu, dRho} = {dLam0, dMu0, dRho0}; dM = dLam + 2 dMu;

(* ---------------------------------------------------------------------------
   [4], [5] the full contrast: orders of the second- and third-moment voxels
   --------------------------------------------------------------------------- *)
ladder = {1, 2, 4, 8};
errs[om2_, nb_] := Module[{exO = N[exactScat[om2], 40]},
   Table[Abs[(solveLayerP[om2, n, nb] - exO)/exO], {n, ladder}]];
orderOf[e_] := N[(Log[2, e[[-3]]/e[[-2]]] + Log[2, e[[-2]]/e[[-1]]])/2];   (* mean over the two finest halvings of d *)
Print["  [4] the second-moment voxel (p = 2) at the full contrast, |error| / |scattered|, {R, T}:"];
ord2 = Table[Module[{e = errs[om2, 3], o},
     o = orderOf[e];
     Print["      omega ", StringPadLeft[ToString[om2], 3], ":  n = 1, 2, 4, 8: ", Map[sci, e, {2}], "   order ",
      fmt[o[[1]], 4], " / ", fmt[o[[2]], 4]];
     o], {om2, {60, 300, 600}}];
Print["      sixth order in reflection and transmission at every frequency: ",
  chk[AllTrue[Flatten[ord2], 5.8 < # < 6.2 &]]];
Print["  [5] the third-moment voxel (p = 3), omega = 300:"];
Module[{e = errs[300, 4], o},
  o = orderOf[e];
  Print["      n = 1, 2, 4, 8: ", Map[sci, e, {2}], "   order ", fmt[o[[1]], 4], " / ", fmt[o[[2]], 4]];
  Print["      eighth order: ", chk[AllTrue[o, 7.7 < # < 8.3 &]]]];

(* ---------------------------------------------------------------------------
   [6] ONE plane of voxels against the thin-layer series (notebook 2c): exact series in the thickness D,
   at the full contrast, for a stiffer and a softer layer.  Claim: R_plane - R_exact = O(D^{2p+3}),
   so a single plane of degree-p voxels reproduces the exact series term by term through D^{2p+2}.
   --------------------------------------------------------------------------- *)
Clear[DD];
kk0 = om/al;
(* the layer lies below the source: |z - zSrc| = z - zSrc, so the series in D is not blocked by Abs *)
gBelow[z_] := I/(2 mP kk0) Exp[I kk0 (z - zSrc)];
planeR[nb_] := Module[{hh = DD/2, zc = DD/2, dQ = DiagonalMatrix[{om^2 dRho, dM}], mat, rhs, sol, uR},
   mat = ArrayFlatten[Table[If[a == b, 2 hh/(2 a - 1) IdentityMatrix[2], 0] - selfF[[a, b]][kk0, hh] . dQ,
      {a, nb}, {b, nb}]];
   rhs = Flatten[Table[gBelow[zc] momF[[a]][kk0, hh] {1, I kk0}, {a, nb}]];
   sol = LinearSolve[mat, rhs];
   uR = Sum[((kMat[kk0, -1] Exp[-I kk0 (zObsR - zc)] momF[[b]][kk0, hh]) . dQ . sol[[2 b - 1 ;; 2 b]])[[1]],
     {b, nb}];
   uR/(gBelow[0] Exp[-I kk0 zObsR])];
Print["  [6] one plane of voxels vs the exact layer, series in D (omega = ", om, "): |coefficient| of D^1 .. D^(2p+3)"];
planeOK = Flatten[Table[
    Block[{dLam = sgn dLam0, dMu = sgn dMu0, dRho = sgn dRho0, dM = sgn (dLam0 + 2 dMu0)},
     Module[{rEx = Block[{dLayer = DD}, rtExact[om][[1]]], ser, cf},
      Table[
       ser = Normal[Series[planeR[p + 1] - rEx, {DD, 0, 2 p + 3}]];
       cf = Table[Coefficient[ser, DD, j], {j, 1, 2 p + 3}];
       Print["      ", If[sgn > 0, "stiffer", "softer "], " layer, p = ", p, ":  ", sci /@ Abs[cf]];
       (* exact zeros through D^{2p+2}, a nonzero D^{2p+3} *)
       AllTrue[Most[cf], PossibleZeroQ] && ! PossibleZeroQ[Last[cf]], {p, 0, 3}]]], {sgn, {1, -1}}]];
Print["      exact through D^(2p+2), first difference at D^(2p+3), p = 0..3, both layers: ", chk[And @@ planeOK]];

(* the data of the paper's convergence figure *)
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_second_moment_data.json",
  <|"omega" -> 300, "n" -> ladder,
    "errors_R_T" -> <|"G0" -> N[errs[300, 1], 16], "G1" -> N[errs[300, 2], 16], "G2" -> N[errs[300, 3], 16],
      "G3" -> N[errs[300, 4], 16]|>|>, "RawJSON"];

Print["==== ContinuumLimit_SecondMoment (stage 14): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
