#!/usr/bin/env wolframscript
(* ==========================================================================
   ADAPTIVE CUBATURE AT THE CORNER OF THE OCTANT, AGAINST THE CLOSED FORMS

   The scalar moments of the cube with fewer than two derivatives on the kernel,
       E[-1; d; w] = int_cube x_w d_d (1/r) d^3x ,   D = |d| <= 1,
   are ordinary integrals: the integrand is bounded at r = 0 for these weights but not smooth there.
   CubeMomentHigherGrades.wl checks every non-vanishing index class of the grades (1,3), (0,4) and (1,5)
   (9, 6 and 18 classes) against numerical integration, 8 times the octant [0, 1/2]^3, where the
   singular point sits at a corner of the region.  This file measures what the integration settings do
   there, class by class, against the closed forms of CubeMomentCore.wl:

     plain      Method -> "GlobalAdaptive" (the check's setting until 30 October 2026)
     duffy      Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"} (its setting now)
     maxerr     Method -> {"GlobalAdaptive", "MaxErrorIncreases" -> 10^4}, on the classes where plain warns

   all with PrecisionGoal 10, AccuracyGoal 12, MaxRecursion 25, WorkingPrecision 20.  Each integration
   runs quietly: the messages it would issue are recorded, by name, as part of the result.

   Run:  wolframscript -file Mathematica/CubeMomentCornerQuadrature.wl
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];
pass[b_] := If[TrueQ[b], "PASS", "****FAIL****"];
eNote[v_] := If[v == 0, "0", With[{e = Floor[Log10[v]]}, ToString[NumberForm[N[v/10^e], {2, 1}]] <> "e" <> ToString[e]]];

(* the closed form, from the store (as CubeMomentHigherGrades.wl reads it) *)
eC[m_, ds_List, w_List] := Block[{Print},
   cached["E_" <> ToString[m] <> "_d" <> StringJoin[ToString /@ Sort[ds]] <> "_w" <> StringJoin[ToString /@ Sort[w]],
    E$[m, Sort[ds], Sort[w]]]];

(* the non-vanishing index classes of a grade: sorted derivative and weight lists, every axis an even count *)
reps[dd_, ww_] := Select[
   Flatten[Table[{ds, ws}, {ds, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, dd]]},
     {ws, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, ww]]}], 1],
   AllTrue[Range[3], Function[ax, EvenQ[Count[Join[#[[1]], #[[2]]], ax]]]] &];

xs = {x, y, z};
integrand[ds_, ws_] := (Times @@ (xs[[#]] & /@ ws)) D[1/Sqrt[x^2 + y^2 + z^2], Sequence @@ (xs[[#]] & /@ ds)];
common = {PrecisionGoal -> 10, AccuracyGoal -> 12, MaxRecursion -> 25, WorkingPrecision -> 20};
methods = <|
   "plain" -> "GlobalAdaptive",
   "duffy" -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"},
   "maxerr" -> {"GlobalAdaptive", "MaxErrorIncreases" -> 10^4}|>;

(* one integration: {value, the names of the messages it issued, seconds}; NIntegrate holds its arguments, so the
   integrand and the options go in evaluated.  The handler sees every message, also those switched off by default
   (General::newsym), which no run would ever print: a message whose name evaluates to $Off[...] is not recorded *)
SetAttributes[recordMessage, HoldFirst];
recordMessage[names_Symbol][h_] := With[{mn = Extract[h, {1, 1}, HoldForm]},
   If[Head[ReleaseHold[mn]] =!= $Off, AppendTo[names, ToString[mn]]]];
integrate[g_, method_] := Module[{names = {}, v, t},
   {t, v} = AbsoluteTiming[Internal`HandlerBlock[
      {"Message", recordMessage[names]},
      Quiet[8 NIntegrate[Evaluate[g], {x, 0, 1/2}, {y, 0, 1/2}, {z, 0, 1/2}, Method -> method,
         Evaluate[Sequence @@ common]]]]];
   {v, DeleteCases[DeleteDuplicates[names], "$Off[]" | "General::stop"], t}];

Print["=============================================================="];
Print["ADAPTIVE CUBATURE AT THE CORNER OF THE OCTANT"];
Print["=============================================================="];
rows = {};
Do[
  Module[{dd = grade[[1]], ww = grade[[2]]},
   Print["  grade (", dd, ",", ww, "): ", Length[reps[dd, ww]], " classes"];
   Do[
    Module[{ds = cl[[1]], ws = cl[[2]], ref, g, res, line},
     ref = N[eC[-1, ds, ws] /. Del -> 1, 30];
     g = integrand[ds, ws];
     res = AssociationMap[integrate[g, methods[#]] &, {"plain", "duffy"}];
     If[res["plain"][[2]] =!= {}, res["maxerr"] = integrate[g, methods["maxerr"]]];
     line = "    d " <> ToString[ds] <> " w " <> ToString[ws] <> ":";
     Do[
      With[{r = res[k], rel = Abs[res[k][[1]] - ref]/Max[1, Abs[ref]]},
       line = line <> "   " <> k <> " " <> eNote[N[rel]] <> " " <> ToString[r[[2]]] <> " " <> ToString[Round[r[[3]], 0.1]] <> " s";
       AppendTo[rows, <|"grade" -> grade, "d" -> ds, "w" -> ws, "method" -> k, "relative_error" -> N[rel],
         "messages" -> r[[2]], "seconds" -> N[r[[3]]]|>]],
      {k, Keys[res]}];
     Print[line]],
    {cl, reps[dd, ww]}]],
  {grade, {{1, 3}, {0, 4}, {1, 5}}}];

(* the summary the papers cite *)
Print["  summary, over every class:"];
Do[
  Module[{r = Select[rows, #method == k &]},
   Print["    ", k, ": ", Length[r], " integrations, ", Count[r, _?(#messages =!= {} &)], " with a message (",
    StringRiffle[DeleteDuplicates[Flatten[#messages & /@ r]], ", "], "), largest relative error ",
    eNote[Max[#["relative_error"] & /@ r]], ", total ", Round[Total[#seconds & /@ r]], " s"]],
  {k, {"plain", "duffy", "maxerr"}}];

okDuffy = AllTrue[Select[rows, #method == "duffy" &], #messages === {} && #["relative_error"] < 10^-12 &];
okPlain = AllTrue[Select[rows, #method == "plain" &], #["relative_error"] < 10^-8 &];
Print["  Duffy coordinates: no message and agreement below 1e-12 on every class: ", pass[okDuffy]];
Print["  plain GlobalAdaptive: within the check's 1e-8 on every class, messages or not: ", pass[okPlain]];
Export[FileNameJoin[{DirectoryName[$InputFileName], "CubeMoment_corner_quadrature.json"}], rows, "JSON"];
Print["  exported CubeMoment_corner_quadrature.json"];
Print[If[okDuffy && okPlain, "ALL PASS", "SOME FAILED"]];
