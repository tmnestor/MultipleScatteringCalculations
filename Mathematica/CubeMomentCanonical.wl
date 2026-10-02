#!/usr/bin/env wolframscript
(* ==========================================================================
   THE STORED MOMENTS IN ONE CANONICAL, STABLE FORM

   The engine Pell-reduces its logarithms: Log[p - q Sqrt[3]] with p^2 - 3 q^2 = 1 becomes
   n Log[2 - Sqrt[3]].  At the higher weights the integrator also returns logarithms the reduction does
   not recognise (Log[Sqrt[3] - 1], Log[1 + Sqrt[3]], Log[16 (97 - 56 Sqrt[3])], ArcTanh[...]) and
   powers of two written as Log[64], Log[512], Log[4096].  The VALUES are right (every one agrees with
   the independent numerical route), but the forms are not canonical:

     * two equal moments can look different, so a zero test between closed forms can fail to decide
       (this happened: a chain-rule check on grade (3,5) reported a failure that is exactly zero);
     * sums of large logarithms that cancel lose digits at machine precision, which is what the Pell
       reduction exists to prevent.

   Every scalar moment of the cube is  Del^s  times a constant, s = m - D + W + 3, and the constant lies
   in the rational span of a short basis.  This file finds that representation for each stored value:

       c = q0 + q1 Sqrt[3] + q2 Pi + q3 Log[2] + q4 Log[3] + q5 Log[2 - Sqrt[3]]      (q_i rational)

   by an integer-relation search at 400 digits, and then VERIFIES it against the stored closed form at
   1500 digits, so a relation found by chance cannot pass.  A value with no relation in this basis is
   reported and left as it is.

   Output:  CubeScalarMomentsCanonical.m   {m, {d..}, {w..}} -> canonical closed form.
   Run:     wolframscript -file Mathematica/CubeMomentCanonical.wl
   ========================================================================== *)

dir = DirectoryName[$InputFileName];
st = Association[Get[FileNameJoin[{dir, "CubeScalarMoments.m"}]]];
basis = {1, Sqrt[3], Pi, Log[2], Log[3], Log[2 - Sqrt[3]]};

canon[key_, expr_] := Module[{s, c, cn, vec, cand, res},
  s = key[[1]] - Length[key[[2]]] + Length[key[[3]]] + 3;
  c = expr /. Del -> 1;
  cn = Quiet[Block[{$MaxExtraPrecision = 4000}, Re[N[c, 400]]]];
  If[Abs[cn] < 10^-300, Return[{0, True}]];
  vec = Quiet[FindIntegerNullVector[Join[{cn}, N[basis, 400]], 10^12]];
  If[! ListQ[vec] || vec[[1]] == 0, Return[{expr, False}]];
  cand = -(Rest[vec] . basis)/vec[[1]];
  res = Quiet[Block[{$MaxExtraPrecision = 6000}, N[cand - c, 1500]]];
  If[Abs[Re[res]] < 10^-1200 && Abs[Im[res]] < 10^-1200, {Del^s cand, True}, {expr, False}]];

t0 = AbsoluteTime[];
out = <||>; failed = {}; changed = 0;
KeyValueMap[Function[{k, v}, Module[{r = canon[k, v]},
     out[k] = r[[1]];
     If[! r[[2]], AppendTo[failed, k]];
     If[r[[2]] && LeafCount[r[[1]]] < LeafCount[v], changed++]]], st];
Put[Normal[KeySort[out]], FileNameJoin[{dir, "CubeScalarMomentsCanonical.m"}]];
Print["canonical forms: ", Length[out] - Length[failed], " of ", Length[out], " in the basis {1, Sqrt[3], Pi, Log[2], Log[3], Log[2 - Sqrt[3]]}, each verified to 1200 digits"];
Print["  shortened: ", changed, ";  no relation found: ", Length[failed]];
If[failed =!= {}, Print["  unreduced: ", Take[failed, UpTo[12]]]];
Print["  largest remaining leaf count: ", Max[LeafCount /@ Values[out]],
  "   functions present: ", Union[Cases[Values[out], (h : Log | ArcTanh | ArcTan | ArcCoth)[_] :> h, Infinity]]];
Print["  arguments of the logarithms: ", Union[Cases[Values[out], Log[a_] :> a, Infinity]]];
Print["  ", Round[AbsoluteTime[] - t0], " s"];
