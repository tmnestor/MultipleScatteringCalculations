#!/usr/bin/env wolframscript
(* ============================================================================
   GradedSphere_LowFrequency.wl  --  the exact low-frequency series of the graded sphere.

   WHY.  The scattering of a cell model at low frequency is a power series in ka whose coefficients a
   frequency-independent closed form of the couplings can deliver order by order. To validate them it
   needs the EXACT series of the body it models. For a homogeneous sphere that series is known in closed
   form (MieAsymptotic.wl); for the graded sphere of the continuum-limit paper (a homogeneous core and a
   C2 smoothstep shell, so the cell model has no staircase) it is not, and this notebook builds it.

   THE METHOD.  The per-order T-matrix of the sphere, T_n(w), w = k_P a, is analytic in w near 0 (the
   regular radial solution is entire in the frequency and the scattering coefficients are its ratios).
   Its Taylor coefficients follow from Cauchy's formula on a circle |w| = rho in the complex plane,
       t_k = (1 / M) sum_j T(rho e^(2 pi i j / M)) (rho e^(2 pi i j / M))^-k,
   whose error is the aliasing of the coefficients k + M, k + 2M, ..., geometric in M, so the
   coefficients come out to the working precision without any difference quotient in the frequency.
   T comes from TakeuchiSaito.wl (TSSphereTMatrix, at complex frequency).

   CHECKS: [1] the extraction against the CLOSED-FORM series of the homogeneous sphere (MieAsymptotic,
   exact contrasts) for n = 0, 1, 2; [2] two radii and two point counts agree; [3] the graded sphere's
   series summed at small real w reproduces the direct T-matrix; [4] as the shell narrows to a step the
   graded coefficients approach the homogeneous closed form. Then the graded coefficients are exported.
   Time e^{-i w t}; SI units, background alpha 5000, beta 3000, rho 2500 (lam0/mu0 = 7/9).
   ============================================================================ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "TakeuchiSaito.wl"}]];

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
Print["==== GradedSphere_LowFrequency :: the exact low-frequency series of the graded sphere ===="];

alpha = 5000; beta = 3000; rho = 2500;
bg = {rho (alpha^2 - 2 beta^2), rho beta^2, rho};
dC = {2 10^9, 1 10^9, 100};   (* the gate contrast: d lambda, d mu, d rho *)
aR = 10;                        (* radius *)
wp = 90;                        (* working precision: a graded T-matrix is good to wp/2 digits *)
omOf[w_] := w alpha/aR;         (* w = k_P a *)
(* the profile across the shell, x = (a - r)/(a - b): "smoothstep" (the paper's, vanishing like x^3 at the
   surface), "sin2" = sin^2(pi x / 2) (like x^2) or "smoother" = 35x^4 - 84x^5 + 70x^6 - 20x^7 (like x^4),
   chosen by the first command-line argument *)
profileName = If[Length[$ScriptCommandLine] > 1, $ScriptCommandLine[[2]], "smoothstep"];
smooth[x_] := Which[profileName === "sin2", Sin[Pi x/2]^2,
   profileName === "smoother", 35 x^4 - 84 x^5 + 70 x^6 - 20 x^7, True, 10 x^3 - 15 x^4 + 6 x^5];
material[f_] := {Function[x, bg[[1]] + dC[[1]] f[x]], Function[x, bg[[2]] + dC[[2]] f[x]], Function[x, bg[[3]] + dC[[3]] f[x]]};
graded[b_] := material[Function[x, Piecewise[{{1, x < b}}, smooth[(aR - x)/(aR - b)]]]];
uniform = material[1 &];

(* the Taylor coefficients t_0 .. t_kmax of w -> T(w) (a matrix-valued function) by Cauchy's formula *)
cauchy[tfun_, rhoC_, m_, kmax_] := Module[{nodes, vals},
  nodes = Table[rhoC Exp[2 Pi I j/m], {j, 0, m - 1}];
  vals = tfun /@ nodes;
  Table[Sum[vals[[j + 1]] nodes[[j + 1]]^-k, {j, 0, m - 1}]/m, {k, 0, kmax}]];

tFun[n_, mat_, b_] := Function[w, Module[{r = TSSphereTMatrix[n, SetPrecision[omOf[w], wp], bg, mat, {b, aR},
      WorkingPrecision -> wp]}, {r["Tpsv"], r["Tsh"]}]];

(* ---------------------------------------------------------------------------
   [1] the homogeneous sphere: extraction against its EXACT series
   The homogeneous sphere's T-matrix is a closed form in spherical Bessel functions (TSShellsTMatrix with one
   shell), so Mathematica expands it in w exactly; the Cauchy coefficients must reproduce that series to the
   working precision. (The library itself agrees with the package's independent Mie arbiter to 1e-17,
   ContinuumLimit_GradedSphere.wl [2]. MieAsymptotic.wl's a_n are in another normalisation: lambda0 + 2 =
   (alpha / beta)^2 from mu0 = rho0 = 1, the plane-wave factor i^n (2n+1), and a partial-wave factor that is
   itself frequency dependent, so they are not compared coefficient by coefficient here.)
   --------------------------------------------------------------------------- *)
inner = {bg[[1]] + dC[[1]], bg[[2]] + dC[[2]], bg[[3]] + dC[[3]]};
Module[{worst = 0},
  Do[Module[{ser, ex},
     ser = CoefficientList[Normal[Series[TSShellsTMatrix[n, w alpha/aR, bg, {aR}, {inner}]["Tpsv"][[1, 1]], {w, 0, 12}]], w];
     ex = cauchy[tFun[n, uniform, aR], 1/5, 64, 12];
     Do[If[k + 1 <= Length[ser] && ser[[k + 1]] =!= 0,
        worst = Max[worst, Abs[ex[[k + 1, 1, 1, 1]] - ser[[k + 1]]]/Abs[ser[[k + 1]]]]], {k, 0, 12}]], {n, 0, 3}];
  Print["  [1] homogeneous sphere, Cauchy coefficients w^0..w^12 vs the exact series (n = 0..3): ", sci[worst],
   " -> ", chk[worst < 10^-25]]];

(* ---------------------------------------------------------------------------
   [2] two radii and two point counts
   --------------------------------------------------------------------------- *)
Module[{a1, a2, a3, worst},
  a1 = cauchy[tFun[1, graded[aR/10], aR/10], 1/5, 48, 12];
  a2 = cauchy[tFun[1, graded[aR/10], aR/10], 3/10, 48, 12];
  a3 = cauchy[tFun[1, graded[aR/10], aR/10], 1/5, 64, 12];
  (* each coefficient against the largest of the order: some coefficients are exactly zero (w^4 here) *)
  worst = Max[Table[Max[Abs[Flatten[{a1[[k]] - a2[[k]]}]], Abs[Flatten[{a1[[k]] - a3[[k]]}]]], {k, 3, 13}]]/
    Max[Abs[Flatten[a1]]];
  Print["  [2] graded sphere (n = 1): coefficients 2..12 from radii 0.2 and 0.3, and 48 and 64 points: ",
   sci[worst], " -> ", chk[worst < 10^-15]]];

(* ---------------------------------------------------------------------------
   the graded sphere: the coefficients
   --------------------------------------------------------------------------- *)
kMax = 24;  (* the higher orders start at higher powers, so twelve terms are too few for them *)
gradedCoeffs = Table[cauchy[tFun[n, graded[aR/10], aR/10], 1/5, 64, kMax], {n, 0, 4}];

(* [3] the series summed at small real w against the direct T-matrix *)
Module[{worst = 0},
  Do[Module[{w0 = SetPrecision[wv, wp], direct, summed},
     direct = tFun[n, graded[aR/10], aR/10][w0];
     summed = Sum[gradedCoeffs[[n + 1, k + 1]] w0^k, {k, 0, kMax}];
     worst = Max[worst, Max[Abs[Flatten[direct - summed]]]/Max[Abs[Flatten[direct]]]]], {n, 0, 4}, {wv, {1/100, 1/20}}];
  Print["  [3] graded sphere, series to w^", kMax, " summed at w = 0.01 and 0.05 vs the direct T-matrix: ",
   sci[worst], " -> ", chk[worst < 10^-12]]];

(* [4] the shell narrowed towards a step: the coefficients approach the homogeneous closed form *)
Module[{lead, rows = {}},
  lead[b_] := cauchy[tFun[0, graded[b], b], 1/5, 48, 4][[4, 1, 1, 1]];  (* the leading, w^3, coefficient *)
  Module[{u = cauchy[tFun[0, uniform, aR], 1/5, 48, 4][[4, 1, 1, 1]]},
   Do[AppendTo[rows, {N[1 - b/aR], Abs[lead[b] - u]/Abs[u]}], {b, {9 aR/10, 99 aR/100, 999 aR/1000}}]];
  Print["  [4] monopole w^3 coefficient, shell width / a vs relative departure from the homogeneous sphere: ",
   Map[sci, rows, {2}]];
  (* the departure is proportional to the shell width: tenfold per tenfold narrowing, within 10% *)
  Print["  [4] departure ratios per tenfold narrowing ", N[rows[[1 ;; 2, 2]]/rows[[2 ;; 3, 2]], 4], " -> ",
   chk[And @@ Thread[Abs[rows[[1 ;; 2, 2]]/rows[[2 ;; 3, 2]] - 10] < 1]]]];

(* export *)
Module[{out = FileNameJoin[{DirectoryName[$InputFileName],
      If[profileName === "smoothstep", "GradedSphere_LowFrequency.json", "GradedSphere_LowFrequency_" <> profileName <> ".json"]}],
   cpx, data},
  (* zeros as plain 0: Mathematica writes a zero of finite precision as "0.e-94", which is not valid JSON *)
  (* a zero of low accuracy ("0.e-79") compares undecidedly with a threshold, so test it with PossibleZeroQ *)
  cpx[z_] := Map[If[PossibleZeroQ[#], 0, N[#, 30]] &, {Re[z], Im[z]}];
  data = <|"body" -> <|"a" -> aR, "core" -> aR/10,
       "profile" -> Which[profileName === "sin2", "sin2 sin(pi x/2)^2", profileName === "smoother", "smoother 35x^4-84x^5+70x^6-20x^7", True, "smoothstep 10x^3-15x^4+6x^5"] <> ", x=(a-r)/(a-b)",
       "background" -> {alpha, beta, rho}, "contrast" -> dC, "w" -> "k_P a"|>,
     "orders" -> Table[<|"n" -> n, "Tpsv" -> Table[Map[cpx, gradedCoeffs[[n + 1, k + 1, 1]], {2}], {k, 0, kMax}],
         "Tsh" -> Table[cpx[gradedCoeffs[[n + 1, k + 1, 2]]], {k, 0, kMax}]|>, {n, 0, 4}]|>;
  Export[out, data, "RawJSON"];
  Print["  wrote ", out]];

Print["  ", Count[oks, True], "/", Length[oks], " checks passed"];
