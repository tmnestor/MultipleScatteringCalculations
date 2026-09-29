#!/usr/bin/env wolframscript
(* ============================================================================
   GradedVoxel_WeakKernel.wl -- the graded voxel's Galerkin double integrals by
   integration by parts onto the CELLS' polynomials.

   THE CONSTRUCTION.  A static term coef * d^idx r^m between the x-cell (centre
   R, half-width 1, test polynomial f) and the x'-cell (centre 0, source
   polynomial g) is a distribution for m = -1, |idx| = 2 and m = 1, |idx| >= 3.
   Move k = max(0, |idx| - (m + 1)) derivatives off the kernel, one per cell
   (the first moved index p onto the x'-cell, the second q onto the x-cell):

     k = 1:  I = <f, d_p g> - sum_{faces F' of V2} n'_p <f, g>_{V1 x F'}
     k = 2:  I = sum_F n_q <f, d_p g>_{F x V2} - sum_{F, F'} n_q n'_p <f, g>_{F x F'}
                 - <d_q f, d_p g>_{V1 x V2} + sum_F' n'_p <d_q f, g>_{V1 x F'}

   with <f, g>_{A x B} = int_A int_B f(x) g(x') Phi(x - x'), Phi = d^rest r^m
   pointwise, at most 1/r: absolutely convergent on every pair, coincident faces
   included.  Each pair integral is reduced per axis to the difference variable
   t = x - x' (a cross-correlation of two intervals, either possibly a point) and
   integrated numerically at 20 digits with a Duffy singularity handler: the
   singular point t = 0 always falls on a corner of a piece.

   This is independent of the Python s-form (graded_voxel.blocks), which moves
   the derivatives onto the AUTOCORRELATION W instead (plane deltas at its kinks).

   GATES (G1): (a) the k = 2 identity on a smooth kernel against brute-force 6-D
   integration; (b) the cube's Coulomb self-energy; (c) the Eshelby sum rule
   sum_p <1, d_p d_p (1/r) 1> = -4 Pi V; (d) the x'-cell shrunk to a point: the
   moment engine's single-average moment.

   EXPORT (G3).  Term-level integrals int int L_a d^idx r^m m_c for the singular
   static terms, a few (a, c), at the four touching orbit representatives (h = 1),
   for the Python cross-check (test_graded_voxel_site.py).  The assembly of terms
   into 9x9 blocks is checked separately in Python (static_term_table against the
   Kelvin kernel), so only the singular integrals need a second implementation.
   ============================================================================ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];  (* defines x, y, z, rr, X, hw, Del *)

oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
sci[v_] := ToString[NumberForm[N[v], 3], OutputForm];
Print["==== GradedVoxel_WeakKernel ===="];

t0 = Symbol["t0"]; t1 = Symbol["t1"]; t2 = Symbol["t2"]; tt = {t0, t1, t2};
tv = Symbol["tv"];
rT = Sqrt[t0^2 + t1^2 + t2^2];
xs = {Symbol["x0"], Symbol["x1"], Symbol["x2"]};
ys = {Symbol["y0"], Symbol["y1"], Symbol["y2"]};
cube = {{-1, 1}, {-1, 1}, {-1, 1}};
wpNum = 20;

(* per-axis cross-correlation of x^al on [a1, b1] with x'^be on [a2, b2], in tv = x - x' *)
xcorrPiece[al_, be_, {a1_, b1_}, {a2_, b2_}, lo_, hi_] := Module[{u, mid = (lo + hi)/2, low, up},
  low = If[a1 >= a2 + mid, a1, a2 + tv];
  up = If[b1 <= b2 + mid, b1, b2 + tv];
  Expand[Integrate[u^al (u - tv)^be, {u, low, up}]]];

xcorr[al_, be_, {a1_, b1_}, {a2_, b2_}] := Module[{bps},
  Which[
    a1 == b1 && a2 == b2, {{"delta", a1 - a2, a1 - a2, a1^al a2^be}},
    a1 == b1, {{"piece", a1 - b2, a1 - a2, Expand[a1^al (a1 - tv)^be]}},
    a2 == b2, {{"piece", a1 - a2, b1 - a2, Expand[(tv + a2)^al a2^be]}},
    True, (bps = Union[{a1 - b2, a1 - a2, b1 - b2, b1 - a2}];
      Table[{"piece", bps[[i]], bps[[i + 1]], xcorrPiece[al, be, {a1, b1}, {a2, b2}, bps[[i]], bps[[i + 1]]]},
        {i, Length[bps] - 1}])]];

(* integrate prod_i poly_i(t_i) * phi(t) over the product of per-axis parts *)
boxInt[parts_List, phi_] := Module[{free, fixed, integrand, lims},
  free = Select[Range[3], parts[[#, 1]] === "piece" &];
  fixed = Complement[Range[3], free];
  integrand = (Times @@ Table[parts[[i, 4]] /. tv -> tt[[i]], {i, 3}]) phi;
  integrand = integrand /. Thread[tt[[fixed]] -> (parts[[#, 2]] & /@ fixed)];
  If[free === {}, Return[integrand]];
  lims = {tt[[#]], parts[[#, 2]], parts[[#, 3]]} & /@ free;
  NIntegrate[integrand, Evaluate[Sequence @@ lims], WorkingPrecision -> wpNum, PrecisionGoal -> If[wpNum === MachinePrecision, 12, 13],
    AccuracyGoal -> If[wpNum === MachinePrecision, 14, 18], MaxRecursion -> 40,
    Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"}]];

(* <f, g>_{A x B}: boxes per axis {lo, hi}, lo == hi on a face's normal axis *)
pairInt[f_, g_, boxA_, boxB_, phi_] := Module[{fr, gr, total = 0},
  fr = CoefficientRules[Expand[f], xs]; gr = CoefficientRules[Expand[g], ys];
  If[fr === {} || gr === {}, Return[0]];
  Do[total += fc[[2]] gc[[2]] Total[boxInt[#, phi] & /@ Tuples[
        Table[xcorr[fc[[1, i]], gc[[1, i]], boxA[[i]], boxB[[i]]], {i, 3}]]],
    {fc, fr}, {gc, gr}];
  total];

boxFaces[box_, p_] := {{ReplacePart[box, p -> {box[[p, 2]], box[[p, 2]]}], +1},
                        {ReplacePart[box, p -> {box[[p, 1]], box[[p, 1]]}], -1}};

(* the k = 2 formula for any pointwise-integrable phi: p onto the x'-cell, q onto the x-cell (1-based) *)
ibp2[f_, g_, p_, q_, V1_, V2_, phi_] := (
  Total[#[[2]] pairInt[f, D[g, ys[[p]]], #[[1]], V2, phi] & /@ boxFaces[V1, q]]
  - Total[Flatten[Table[F1[[2]] F2[[2]] pairInt[f, g, F1[[1]], F2[[1]], phi],
      {F1, boxFaces[V1, q]}, {F2, boxFaces[V2, p]}]]]
  - pairInt[D[f, xs[[q]]], D[g, ys[[p]]], V1, V2, phi]
  + Total[#[[2]] pairInt[D[f, xs[[q]]], g, V1, #[[1]], phi] & /@ boxFaces[V2, p]]);

(* one term d^idx r^m (idx 0-based axes): x-cell centre R, half-width 1, test f; x'-cell centre 0,
   half-width eps, source g *)
termIntegral[m_, idx_List, f_, g_, R_List, eps_: 1] := Module[{k, moved, rest, phi, V1, V2, p},
  k = If[m >= 0 && EvenQ[m], 0, Max[0, Length[idx] - (m + 1)]];
  moved = Take[idx, k]; rest = Drop[idx, k];
  phi = If[rest === {}, rT^m, D[rT^m, Sequence @@ (tt[[# + 1]] & /@ rest)]];
  V1 = Transpose[{R - 1, R + 1}]; V2 = {{-eps, eps}, {-eps, eps}, {-eps, eps}};
  Which[
    k == 0, pairInt[f, g, V1, V2, phi],
    k == 1, (p = moved[[1]] + 1;
      pairInt[f, D[g, ys[[p]]], V1, V2, phi] - Total[#[[2]] pairInt[f, g, V1, #[[1]], phi] & /@ boxFaces[V2, p]]),
    k == 2, ibp2[f, g, moved[[1]] + 1, moved[[2]] + 1, V1, V2, phi]]];

testExps = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
srcExps = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
fA[a_, R_] := Times @@ ((xs - R)^testExps[[a + 1]]);   (* a, c 0-based as in Python *)
gC[c_] := Times @@ (ys^srcExps[[c + 1]]);

If[MemberQ[$ScriptCommandLine, "--time-one"],
  Print["  one term (m = -1, idx = {0,0}, a = 0, c = 0, self): ",
    AbsoluteTiming[termIntegral[-1, {0, 0}, 1, 1, {0, 0, 0}]]];
  Exit[0]];

(* ---------------- G1 gates ---------------- *)
If[! MemberQ[$ScriptCommandLine, "--export-only"],

(* (a) the k = 2 identity on a smooth kernel: <x0, d_0 d_1 exp(-r^2) y1> *)
Module[{ibp, brute, sm = Exp[-rT^2]},
  ibp = ibp2[xs[[1]], ys[[2]], 1, 2, cube, cube, sm];
  (* NIntegrate needs plain symbols as variables, not xs[[i]] *)
  brute = With[{a0 = xs[[1]], a1 = xs[[2]], a2 = xs[[3]], b0 = ys[[1]], b1 = ys[[2]], b2 = ys[[3]],
      ig = (xs[[1]] ys[[2]] D[sm, t0, t1]) /. Thread[tt -> xs - ys]},
    NIntegrate[ig, {a0, -1, 1}, {a1, -1, 1}, {a2, -1, 1}, {b0, -1, 1}, {b1, -1, 1}, {b2, -1, 1},
      WorkingPrecision -> 20, PrecisionGoal -> 12]];
  Print["  [a] IBP identity on exp(-r^2): rel ", sci[ibp/brute - 1], "  ", chk[Abs[ibp/brute - 1] < 10^-10]]];
If[MemberQ[$ScriptCommandLine, "--gate-a"], Exit[If[And @@ oks, 0, 1]]];

(* (b) Coulomb self-energy: <1, 1/r 1> over the cube of half-width 1 = C (2h)^5 *)
Module[{v = pairInt[1, 1, cube, cube, 1/rT],
        c = 2 ((1 + Sqrt[2] - 2 Sqrt[3])/5 - Pi/3 + Log[(1 + Sqrt[2]) (2 + Sqrt[3])])},
  Print["  [b] Coulomb self-energy / (C 2^5) - 1 = ", sci[v/(c 32) - 1], "  ", chk[Abs[v/(c 32) - 1] < 10^-12]]];

(* (c) Eshelby sum rule: sum_p <1, d_p d_p (1/r) 1> = -4 Pi V, V = 8 *)
Module[{v = Sum[termIntegral[-1, {p, p}, 1, 1, {0, 0, 0}], {p, 0, 2}]},
  Print["  [c] Eshelby sum rule / (-32 Pi) - 1 = ", sci[v/(-32 Pi) - 1], "  ", chk[Abs[v/(-32 Pi) - 1] < 10^-12]]];

(* (d) point limit: shrink the x'-cell (still centred on the singular point); <1, d_0 d_0 (1/r) 1>/(2 eps)^3
   -> Int_V d_0 d_0 (1/r), the engine's distributional single average E$[-1, {1, 1}, {}] at Del = 2.
   Error O(eps^2): Richardson. *)
Module[{v1, v2, v3, r1, r2, eng},
  v1 = termIntegral[-1, {0, 0}, 1, 1, {0, 0, 0}, 1/10]/(2/10)^3;
  v2 = termIntegral[-1, {0, 0}, 1, 1, {0, 0, 0}, 1/20]/(2/20)^3;
  v3 = termIntegral[-1, {0, 0}, 1, 1, {0, 0, 0}, 1/40]/(2/40)^3;
  r1 = (4 v2 - v1)/3; r2 = (4 v3 - v2)/3;
  eng = N[E$[-1, {1, 1}, {}] /. Del -> 2, 20];
  Print["  [d] point limit vs moment engine: raw ", sci[v3/eng - 1], ", Richardson ", sci[((16 r2 - r1)/15)/eng - 1],
    "  ", chk[Abs[((16 r2 - r1)/15)/eng - 1] < 10^-8]]];

];  (* end of the G1 gates *)

(* ---------------- G3 export: term-level integrals ---------------- *)
(* the terms whose derivatives are MOVED (k = 1, 2): where the distributional handling lives. The k = 0
   terms are plain weakly singular integrals (gate [b] covers that path). 20-digit runs took ~6 min per
   job; machine precision with PrecisionGoal 12 is used here, checked against the 20-digit values kept for
   the first self-block jobs. *)
terms = {{-1, {0}}, {-1, {0, 0}}, {-1, {0, 1}}, {1, {0, 1, 2}}, {1, {0, 0, 1, 1}}, {1, {0, 1, 1, 2}}};
pairsAC = {{0, 0}, {2, 4}};
If[MemberQ[$ScriptCommandLine, "--export-only"], wpNum = MachinePrecision];
offsets = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}};
(* --retry-timeouts: recompute only the records that timed out, with a one-hour cap, and rewrite the file *)
If[MemberQ[$ScriptCommandLine, "--retry-timeouts"],
  Module[{file = FileNameJoin[{DirectoryName[$InputFileName], "GradedVoxel_term_integrals.jsonl"}], recs, out},
    recs = ImportString[#, "RawJSON"] & /@ Select[StringSplit[Import[file, "Text"], "\n"], StringLength[#] > 0 &];
    out = Map[Function[r, If[r["value"] =!= Null, r,
        Module[{res = AbsoluteTiming[TimeConstrained[
             N[termIntegral[r["m"], r["idx"], fA[r["a"], 2 r["offset"]], gC[r["c"]], 2 r["offset"]], wpNum],
             3600, $TimedOut]]},
          Print["  retry ", {r["m"], r["idx"], r["a"], r["c"], r["offset"]}, "  ", Round[res[[1]]], " s",
            If[res[[2]] === $TimedOut, "  TIMED OUT", ""]];
          Append[r, {"seconds" -> N[Round[res[[1]], 1/10]],
            "value" -> If[res[[2]] === $TimedOut, Null, {Re[res[[2]]], Im[res[[2]]]}]}]]]], recs];
    Export[file, StringRiffle[ExportString[#, "JSON", "Compact" -> True] & /@ out, "\n"] <> "\n", "Text"];
    Print["  retried; nulls left: ", Count[out, r_ /; r["value"] === Null]];
    Exit[0]]];

(* offsets outermost: the self block first, so a partial run is still useful *)
jobs = Flatten[Table[{tm, ac, o}, {o, offsets}, {tm, terms}, {ac, pairsAC}], 2];
(* sequential, each job timed and capped, each result appended at once: a slow pair integral cannot
   stall the rest, and whatever finished is on disk *)
outFile = FileNameJoin[{DirectoryName[$InputFileName], "GradedVoxel_term_integrals.jsonl"}];
If[FileExistsQ[outFile], DeleteFile[outFile]];
t0clock = AbsoluteTime[];
Do[Module[{j = jobs[[n]], res, v},
    res = AbsoluteTiming[TimeConstrained[
       N[termIntegral[j[[1, 1]], j[[1, 2]], fA[j[[2, 1]], 2 j[[3]]], gC[j[[2, 2]]], 2 j[[3]]], wpNum],
       600, $TimedOut]];
    v = res[[2]];
    Module[{st = OpenAppend[outFile]},
      WriteString[st, ExportString[<|"m" -> j[[1, 1]], "idx" -> j[[1, 2]], "a" -> j[[2, 1]], "c" -> j[[2, 2]],
          "offset" -> j[[3]], "seconds" -> N[Round[res[[1]], 1/10]],
          "value" -> If[v === $TimedOut, Null, {Re[v], Im[v]}]|>, "JSON", "Compact" -> True], "\n"];
      Close[st]];
    Print["  job ", n, "/", Length[jobs], " ", j, "  ", Round[res[[1]], 0.1], " s", If[v === $TimedOut, "  TIMED OUT", ""]]],
  {n, Length[jobs]}];
Print["  export: ", Length[jobs], " term integrals in ", Round[AbsoluteTime[] - t0clock], " s"];
Print["  gates passed: ", Count[oks, True], "/", Length[oks]];
