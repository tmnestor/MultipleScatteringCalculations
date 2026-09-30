#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_ErrorConstant.wl  --  notebook 6 of the continuum-limit study.

   THE LEADING ERROR OF THE DISCRETE LAYER, IN CLOSED FORM.  Notebook 5 showed
   that the S-free collocation (the exact cell-averaged kernel with the closure
   T-matrix) converges to the continuous layer at exactly second order.  Its
   kernel is integrated over each cell EXACTLY; the only approximation is that
   the internal field is taken as uniform across a cell, equal to its value at
   the centre.  At Born order that makes the error a pure midpoint-rule error,
   and it is closed-form for every n:

     REFLECTION  (kernel and incident field run in opposite directions: the
     integrand carries e^{2 i k z}):
         R_disc / R_exact = sinc(k h) / sinc(2 k h) = 1 + (k d)^2/8 + O(d^4);
     TRANSMISSION  (kernel and field run the same way: the integrand is flat):
         T_scat,disc / T_scat,exact = sinc(k h) = 1 - (k d)^2/24 + O(d^4),
   h = d/2, k the background wavenumber.  Opposite signs: NO scalar correction
   of the single site can cancel both.  A fourth-order scheme needs the cell's
   first moment -- the variation of the field ACROSS the cell -- which a
   uniform-strain (collocation) single site cannot carry: the gradient-carrying
   (T27-type) closure with a matching generalised propagator.

   CHECKS:
     [1] the closed forms against the chain of notebook 5 at weak contrast
         (Born order), n = 1 .. 16, reflection and transmission;
     [2] at the gate's contrast the ratio departs from the Born form by the
         multiple-scattering / interior-wavenumber correction -- measured;
     [3] the package T-matrix (whose form-factor corrections cut the
         reflection constant 4.45x, notebook 5) on TRANSMISSION.
   Coordinates (z, x, y), z down; e^{-i w t}; SI units.
   ============================================================================ *)

nbText = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_Chain.wl", "Text"];
ToExpression[StringTake[nbText, StringPosition[nbText, "omegas = Rationalize"][[1, 1]] - 1], InputForm];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
{dLam0, dMu0, dRho0} = {dLam, dMu, dRho};
setContrast[s_] := ({dLam, dMu, dRho} = s {dLam0, dMu0, dRho0};
   c6 = Table[Which[v <= 3 && w <= 3, dLam + If[v == w, 2 dMu, 0], v == w, 2 dMu, True, 0], {v, 6}, {w, 6}]);
zBelow = 4;
tExact[om_, dl_] := Module[{k0 = om/al, k1, m1 = lam + dLam + 2 (mu + dMu), r1 = rho + dRho, rr, tt, bb, cc},
   k1 = om Sqrt[r1/m1];
   tt /. First@Solve[{1 + rr == bb + cc, mP I k0 (1 - rr) == m1 I k1 (bb - cc),
       bb Exp[I k1 dl] + cc Exp[-I k1 dl] == tt, m1 I k1 (bb Exp[I k1 dl] - cc Exp[-I k1 dl]) == mP I k0 tt}, {rr, tt, bb, cc}]];
exactScat[om_, "R"] := uScatExact[om, dLayer];
exactScat[om_, "T"] := Module[{k0 = om/al}, tExact[om, dLayer] psiInc[om, 0][[1]] Exp[I k0 (zBelow - dLayer)] - psiInc[om, zBelow][[1]]];
(* the chain, collocation (B) or Foldy-Lax with a given T (A), observed at zo *)
chainAt[om_, n_, zo_, tt_ : None, s9_ : None] := Module[{d = dLayer/n, v, zs, dd, big, e0, e, obs},
   v = d^3; zs = Table[(j - 1/2) d, {j, n}]; dd = delta9[om];
   e0 = Flatten[psiInc[om, #] & /@ zs];
   If[tt === None,
    big = ArrayFlatten[Table[If[i == j, kPlate[om, d], kBetween[om, zs[[i]] - zs[[j]], d]] . dd, {i, n}, {j, n}]];
    e = LinearSolve[N[IdentityMatrix[9 n] - v big, 40], N[e0, 40]];
    obs = psiInc[om, zo] + v Sum[kBetween[om, zo - zs[[j]], d] . dd . e[[9 (j - 1) + 1 ;; 9 j]], {j, n}],
    big = ArrayFlatten[Table[If[i == j, kPlate[om, d] - s9/v, kBetween[om, zs[[i]] - zs[[j]], d]] . tt, {i, n}, {j, n}]];
    e = LinearSolve[N[IdentityMatrix[9 n] - big, 40], N[e0, 40]];
    obs = psiInc[om, zo] + Sum[kBetween[om, zo - zs[[j]], d] . tt . e[[9 (j - 1) + 1 ;; 9 j]], {j, n}]];
   obs[[1]] - psiInc[om, zo][[1]]];
om = 300; k0 = om/al;
predicted[n_, "R"] := With[{h = dLayer/(2 n)}, Sinc[k0 h]/Sinc[2 k0 h]];
predicted[n_, "T"] := With[{h = dLayer/(2 n)}, Sinc[k0 h]];

Print["==== ContinuumLimit_ErrorConstant :: the leading error of the discrete layer, closed form ===="];
(* ---------------------------------------------------------------------------
   [1] Born order (contrast x 1e-4): discrete / exact scattered field vs the closed forms
   --------------------------------------------------------------------------- *)
setContrast[1/10000];
res1 = Table[Module[{zo = If[kind == "R", zObs, zBelow], ex, rat},
    ex = N[exactScat[om, kind], 40];
    Table[rat = chainAt[om, n, zo]/ex; {kind, n, rat, N[predicted[n, kind], 20]}, {n, {1, 2, 4, 8, 16}}]],
   {kind, {"R", "T"}}];
Print["  [1] Born order (contrast x 1e-4), omega = ", om, ": discrete / exact scattered field vs the closed form"];
Do[Print["      ", r[[1]], "  n = ", StringPadLeft[ToString[r[[2]]], 2], ":  ratio - 1 = ", sci[Re[r[[3]] - 1]],
   "   predicted ", sci[r[[4]] - 1], "   |difference| ", sci[Abs[r[[3]] - r[[4]]]]], {r, Flatten[res1, 1]}];
worst1 = Max[Abs[#[[3]] - #[[4]]]/Abs[#[[4]] - 1] & /@ Flatten[res1, 1]];
Print["      worst |ratio - predicted| / |predicted - 1| = ", sci[worst1], " (the O(contrast) remainder) -> ", chk[worst1 < 10^-2]];
Print["      reflection and transmission errors have OPPOSITE signs: ",
  chk[Re[res1[[1, 3, 3]] - 1] > 0 && Re[res1[[2, 3, 3]] - 1] < 0]];

(* ---------------------------------------------------------------------------
   [2] at the gate's contrast: the constant (ratio - 1)/(k0 d)^2 departs from 1/8 and -1/24
   --------------------------------------------------------------------------- *)
setContrast[1];
Print["  [2] the O(d^2) constant (ratio - 1)/(k0 d)^2 at n = 16, against the contrast:"];
Do[setContrast[s];
  Module[{d = dLayer/16, cR, cT},
   cR = (chainAt[om, 16, zObs]/N[exactScat[om, "R"], 40] - 1)/(k0 d)^2;
   cT = (chainAt[om, 16, zBelow]/N[exactScat[om, "T"], 40] - 1)/(k0 d)^2;
   Print["      contrast x ", s, ":  reflection ", sci[Re[cR]], "   transmission ", sci[Re[cT]]]],
  {s, {1/10000, 1/10, 1}}];
Print["      (Born limit: 1/8 = 0.125 and -1/24 = -0.0417)"];
setContrast[1];

(* ---------------------------------------------------------------------------
   [3] the package T-matrix on transmission (its reflection constant is 4.45x smaller than collocation's)
   --------------------------------------------------------------------------- *)
Print["  [3] package T vs collocation, error / scattered, gate's contrast, omega = ", om, ":"];
Do[Module[{c = cellOf[om, n], tp, sp, eAR, eBR, eAT, eBT},
   tp = mat[c["T"]]; sp = mat[c["self"]];
   eAR = Abs[chainAt[om, n, zObs, tp, sp]/N[exactScat[om, "R"], 40] - 1];
   eBR = Abs[chainAt[om, n, zObs]/N[exactScat[om, "R"], 40] - 1];
   eAT = Abs[chainAt[om, n, zBelow, tp, sp]/N[exactScat[om, "T"], 40] - 1];
   eBT = Abs[chainAt[om, n, zBelow]/N[exactScat[om, "T"], 40] - 1];
   Print["      n = ", StringPadLeft[ToString[n], 2], ":  reflection package ", sci[eAR], " / collocation ", sci[eBR],
    "  (", ToString[NumberForm[eAR/eBR, 3], OutputForm], "x)    transmission package ", sci[eAT], " / collocation ", sci[eBT],
    "  (", ToString[NumberForm[eAT/eBT, 3], OutputForm], "x)"]],
  {n, {4, 8, 16}}];

Print["==== ContinuumLimit_ErrorConstant (stage 6): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
