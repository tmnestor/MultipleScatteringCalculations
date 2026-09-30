#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_FourthOrderError.wl  --  notebook 10 of the continuum-limit study.

   THE LEADING ERROR OF THE FIRST-MOMENT VOXEL, IN CLOSED FORM.  Notebook 6 gave
   the collocation's leading error exactly; notebooks 7 and 9 measured the
   first-moment voxel's order as 4.  At first order in the contrast (Born) the
   error of every scheme here is closed-form for every n:
     * the Galerkin solution of the unscattered problem is the L2 PROJECTION of the
       incident field onto each cell's basis ({1} for the mean-only voxel,
       {1, s/h} for the first-moment voxel); collocation takes the centre value;
     * the kernel is integrated over each cell EXACTLY, so the Born error is the
       kernel's cell integral applied to (projection - field);
     * every Born integrand is exponential: the field e^{i k z} and the kernel
       e^{i q z'} toward an observer outside the layer, q = +k for reflection
       (the two-way phase) and q = -k for transmission.  Displacement and strain
       rows carry the same z-dependence, and the local delta of the kernel never
       reaches an outside observer.
   Per cell, discrete / exact =
       [ int e^{i q s} P[e^{i k s}] ds ] / [ int e^{i q s} e^{i k s} ds ],   s in [-h, h],
   identical for every cell, hence for the whole layer and every n.

   CHECKS: [1] the three ratios symbolically, and their series in k h; [2] against
   the discrete chain of notebook 7 at Born order (contrast x 1e-4), n = 1 .. 8,
   reflection and transmission, all three schemes.
   ============================================================================ *)

(* notebook 7's chain (its definitions, not its checks), loaded first: it defines its own oks/chk *)
nbText = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_FourthOrder.wl", "Text"];
ToExpression[StringTake[nbText, StringPosition[nbText, "Print[\"==== ContinuumLimit_FourthOrder ::"][[1, 1]] - 1], InputForm];
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_FourthOrderError :: the first-moment voxel's leading error ===="];

Clear[k, q, h, s];
i0[c_] := 2 h Sinc[c h];                                 (* int e^{i c s} ds, regular at c = 0 *)
i1[c_] := Integrate[(s/h) Exp[I c s], {s, -h, h}];      (* first Legendre moment *)
(* projection coefficients of e^{i k s} onto {1, s/h}: Gram 2h, 2h/3 *)
a0 = i0[k]/(2 h); a1 = i1[k]/(2 h/3);
exactCell = i0[k + q];
ratio["collocation"] = i0[q]/exactCell;
ratio["mean"] = a0 i0[q]/exactCell;
ratio["first moment"] = (a0 i0[q] + a1 i1[q])/exactCell;
schemes = {"collocation", "mean", "first moment"};

(* ---------------------------------------------------------------------------
   [1] the ratios and their series, reflection (q = k) and transmission (q = -k), in x = k d = 2 k h
   --------------------------------------------------------------------------- *)
Clear[x];
ser[sc_, sign_] := Simplify[Normal[Series[ratio[sc] /. {q -> sign k} /. h -> x/(2 k) , {x, 0, 6}]], x > 0];
Print["  [1] discrete / exact, per cell and so for every n (x = k d):"];
Do[Print["      ", StringPadRight[sc, 13], " reflection   ", InputForm[ser[sc, 1]]];
  Print["      ", StringPadRight[" ", 13], " transmission ", InputForm[ser[sc, -1]]], {sc, schemes}];
coef4R = SeriesCoefficient[ser["first moment", 1], {x, 0, 4}];
coef4T = SeriesCoefficient[ser["first moment", -1], {x, 0, 4}];
Print["      first-moment voxel: reflection 1 + (", coef4R, ") (k d)^4,  transmission 1 + (", coef4T, ") (k d)^4"];
Print["      its (k d)^2 terms vanish in both: ",
  chk[SeriesCoefficient[ser["first moment", 1], {x, 0, 2}] === 0 && SeriesCoefficient[ser["first moment", -1], {x, 0, 2}] === 0]];
Print["      the collocation constants reproduce notebook 6 (1/8, -1/24): ",
  chk[SeriesCoefficient[ser["collocation", 1], {x, 0, 2}] === 1/8 && SeriesCoefficient[ser["collocation", -1], {x, 0, 2}] === -1/24]];

(* ---------------------------------------------------------------------------
   [2] against the discrete chain (notebook 7) at Born order
   --------------------------------------------------------------------------- *)
{dLam, dMu, dRho} = 10^-4 {dLam, dMu, dRho}; dM = dLam + 2 dMu;
om = 300; kk = om/al; ex = N[exactScat[om], 40];
worst = 0;
Print["  [2] the chain at Born order (contrast x 1e-4), omega = ", om, ": (discrete/exact - 1) measured vs closed form"];
Do[Module[{nb = If[sc == "first moment", 2, 1], mode = If[sc == "collocation", "C", "G"], row},
   row = Table[Module[{d = dLayer/n, got, predR, predT},
      got = solveLayer[om, n, nb, mode]/ex - 1;
      predR = N[ratio[sc] /. {q -> kk, k -> kk, h -> d/2}, 20] - 1;
      predT = N[ratio[sc] /. {q -> -kk, k -> kk, h -> d/2}, 20] - 1;
      worst = Max[worst, Abs[got[[1]] - predR]/Abs[predR], Abs[got[[2]] - predT]/Abs[predT]];
      {n, got, {predR, predT}}], {n, {1, 2, 4, 8}}];
   Print["      ", sc, ":"];
   Do[Print["         n = ", r[[1]], ":  R ", sci[Re[r[[2, 1]]]], " (", sci[Re[r[[3, 1]]]], ")    T ",
      sci[Re[r[[2, 2]]]], " (", sci[Re[r[[3, 2]]]], ")"], {r, row}]],
  {sc, schemes}];
Print["      worst |measured - closed form| / |closed form| = ", sci[worst], " (the O(contrast) remainder) -> ", chk[worst < 10^-2]];

Print["==== ContinuumLimit_FourthOrderError (stage 10): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
