#!/usr/bin/env wolframscript
(* ==========================================================================
   THE MOMENTS BEYOND THE SIX: WHICH ARE NEEDED, AND DOES THE ENGINE REACH THEM?

   The hierarchy of degree q (displacement expanded to q-th gradients about the
   centre, the equation and its first q derivatives imposed at the centre) needs
   the moments of grade (D, W) = (derivatives on G, degree of the weight)

       density term   :  D = m,      W <= q,      m = 0..q,   D + W even
       stiffness term :  D = m + 1,  W <= q - 1,  m = 0..q,   D + W even.

   q = 2 gives the six of the Basic System, G (0,0), K (0,2), N (1,1), M (2,0),
   P (2,2), Q (3,1).  On the layer the order of convergence with both contrasts
   is 2 for q = 1, 2;  4 for q = 3, 4;  6 for q = 5, 6
   (ContinuumLimit_GradientHierarchy.wl, 36/36).  So:

       fourth order needs q = 3:  four new grades  (1,3) (3,3) (4,0) (4,2)
       sixth  order needs q = 5:  fifteen new grades in all (listed by [1]).

   Each moment reduces to the scalar moments  E[m; d1..dD'; w1..wW]  of r^m, with
   D' = D or D + 2 (the tensor's own d_i d_n), which CubeMomentCore.wl evaluates
   for any D' and W by peeling derivatives onto the faces.

   THE GATES.  A new grade is trusted only if it passes checks with answers
   known independently of the engine:

     (a) the Laplacian chain   Sum_p E[m; {p,p} + rest; w] = m (m+1) E[m-2; rest; w]   for m >= 1,
         because Lap r^m = m (m+1) r^(m-2);
     (b) the delta rule        Sum_p E[-1; {p,p} + rest; w] = -4 Pi (-1)^|rest| (d_rest w)(0),
         because Lap (1/r) = -4 Pi delta: the sharp test that the distributional
         content (the Eshelby delta and its derivatives) is present and right;
     (c) parity: E vanishes unless every axis carries an even number of indices.

   This file: [1] lists the grades;  [2] runs (a), (b), (c) on the new grades of
   q = 3 and times them;  [3] the same for q = 5, as far as the time allows.

   Run:  wolframscript -file Mathematica/CubeMomentHigherGrades.wl [maxGradeSum]
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];
pass[b_] := If[TrueQ[b], "PASS", "****FAIL****"];
maxSum = If[Length[$ScriptCommandLine] > 1, ToExpression[$ScriptCommandLine[[2]]], 6];

(* every scalar moment used here is stored on disk (Mathematica/cache/, never committed): the symbolic face
   integrals at four and more derivatives take minutes each, and the tensor assembly will ask for them again *)
eC[m_, ds_List, w_List] := Block[{Print},
   cached["E_" <> ToString[m] <> "_d" <> StringJoin[ToString /@ Sort[ds]] <> "_w" <> StringJoin[ToString /@ Sort[w]],
    E$[m, Sort[ds], Sort[w]]]];

(* an ordinary integral, for the grades with fewer than two derivatives, where no trace rule applies:
   the integrand d^D (1/r) w is integrable for D <= 1, and is integrated numerically over the cube *)
nintCheck[dd_, ww_] := Module[{ds, ws, rs, val, num, xs = {x, y, z}, f},
  rs = reps[dd, ww];
  If[rs === {}, Return[True]];
  {ds, ws} = Last[rs];
  val = N[eC[-1, ds, ws] /. Del -> 1, 20];
  f = (Times @@ (xs[[#]] & /@ ws)) D[1/Sqrt[x^2 + y^2 + z^2], Sequence @@ (xs[[#]] & /@ ds)];
  num = 8 NIntegrate[f, {x, 0, 1/2}, {y, 0, 1/2}, {z, 0, 1/2}, Method -> "GlobalAdaptive",
     PrecisionGoal -> 10, AccuracyGoal -> 12, MaxRecursion -> 25, WorkingPrecision -> 20];
  Abs[val - num] < 10^-8 Max[1, Abs[val]]];

(* the zero test between closed forms.  The engine's zeroQ evaluates at 50 extra digits and falls back on
   Simplify; at the higher weights the integrator returns logarithms the Pell reduction does not recognise
   (Log[Sqrt[3] - 1], Log[1 + Sqrt[3]], ArcTanh), an exact zero is then not certified, and Simplify can take
   the kernel down.  Here the difference is evaluated with 3000 extra digits: an exact zero returns a zero
   of that accuracy.  (CubeMomentCanonical.wl puts every stored form into one canonical basis afterwards.) *)
exactZero[e_] := Block[{$MaxExtraPrecision = 3000},
   With[{v = Quiet[N[e /. Del -> 37/29, 40]]}, Abs[v] < 10^-30]];

grades[q_] := Union[
   Flatten[Table[If[EvenQ[m + w], {m, w}, Nothing], {m, 0, q}, {w, 0, q}], 1],
   Flatten[Table[If[EvenQ[m + 1 + w], {m + 1, w}, Nothing], {m, 0, q}, {w, 0, q - 1}], 1]];

Print["=============================================================="];
Print["MOMENTS BEYOND THE SIX"];
Print["=============================================================="];
Print["[1] grades (D, W) needed by the hierarchy of degree q"];
Do[Print["    q = ", q, ": ", Length[grades[q]], " grades; new beyond q = ", q - 1, ": ",
   Complement[grades[q], grades[q - 1]]], {q, 2, 5}];
Print["    q = 2 is the six: ", grades[2]];

(* (d_rest w)(0) for a monomial weight w = x_{w1}..x_{wW} and a derivative multi-index rest *)
derivAtZero[rest_List, w_List] := Module[{xs = {xa, xb, xc}, f},
  f = Times @@ (xs[[#]] & /@ w);
  (D[f, Sequence @@ (xs[[#]] & /@ rest)]) /. Thread[xs -> 0]];

(* representative index sets for a grade: all sorted derivative lists and weights with even axis counts *)
reps[dd_, ww_] := Select[
   Flatten[Table[{ds, ws}, {ds, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, dd]]},
     {ws, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, ww]]}], 1],
   AllTrue[Range[3], Function[ax, EvenQ[Count[Join[#[[1]], #[[2]]], ax]]]] &];

checkGrade[{dd_, ww_}] := Module[{t0 = AbsoluteTime[], okA = True, okB = True, okC = True, rs, nA = 0, nB = 0},
  (* (a) and (b): add a traced pair to a rest list of length dd - 2 *)
  If[dd >= 2,
   Do[
    Module[{rest = pr[[1]], w = pr[[2]]},
     (* (b) kernel 1/r *)
     With[{lhs = Sum[eC[-1, Join[{p, p}, rest], w], {p, 3}],
       rhs = -4 Pi (-1)^Length[rest] derivAtZero[rest, w]},
      nB++; If[! exactZero[lhs - rhs], okB = False; Print["      (b) fails at rest ", rest, " w ", w]]];
     (* (a) kernel r *)
     With[{lhs = Sum[eC[1, Join[{p, p}, rest], w], {p, 3}], rhs = 2 eC[-1, rest, w]},
      nA++; If[! exactZero[lhs - rhs], okA = False; Print["      (a) fails at rest ", rest, " w ", w]]]],
    {pr, Flatten[Table[{ds, ws}, {ds, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, dd - 2]]},
        {ws, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, ww]]}], 1]}]];
  (* (c) parity on one odd component *)
  If[dd + ww >= 1,
   With[{ds = ConstantArray[1, dd], ws = If[ww > 0, Join[ConstantArray[1, ww - 1], {2}], {}]},
    If[ww > 0, okC = exactZero[eC[-1, ds, ws]]]]];
  rs = reps[dd, ww];
  If[dd < 2,
   okA = okB = nintCheck[dd, ww];
   Print["    grade (", dd, ",", ww, "): ", Length[rs], " non-vanishing index classes;  no trace rule (D < 2); ",
    "against numerical integration over the cube ", pass[okA], "   (c) parity ", pass[okC], "   ",
    Round[AbsoluteTime[] - t0], " s"],
   Print["    grade (", dd, ",", ww, "): ", Length[rs], " non-vanishing index classes;  (a) chain x", nA, " ",
    pass[okA], "   (b) delta rule x", nB, " ", pass[okB], "   (c) parity ", pass[okC], "   ",
    Round[AbsoluteTime[] - t0], " s"]];
  okA && okB && okC];

(* the tensor's own d_i d_n acts on the kernel r (and r^3, ... dynamically), so a moment of grade (D, W)
   also needs E[1; D + 2 derivatives; W].  Gate: the chain rule on those, Sum_p E[1;{p,p}+rest;w] = 2 E[-1;rest;w],
   with rest of length D: every traced component of the D + 2 derivative moment against the gated 1/r moment. *)
checkTensorKernel[{dd_, ww_}] := Module[{t0 = AbsoluteTime[], ok = True, n = 0},
  Do[
   With[{lhs = Sum[eC[1, Join[{p, p}, pr[[1]]], pr[[2]]], {p, 3}], rhs = 2 eC[-1, pr[[1]], pr[[2]]]},
    n++; If[! exactZero[lhs - rhs], ok = False; Print["      tensor-kernel chain fails at rest ", pr[[1]], " w ", pr[[2]]]]],
   {pr, Flatten[Table[{ds, ws}, {ds, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, dd]]},
       {ws, DeleteDuplicates[Sort /@ Tuples[{1, 2, 3}, ww]]}], 1]}];
  Print["    grade (", dd, ",", ww, "): kernel r with ", dd + 2, " derivatives, chain x", n, " ", pass[ok], "   ",
   Round[AbsoluteTime[] - t0], " s"];
  ok];

(* a second argument restricts the run to grades with D + W >= minSum and skips [2]: for a second kernel
   working on the heavy grades while the first does the light ones (the cache on disk is shared) *)
minSum = If[Length[$ScriptCommandLine] > 2, ToExpression[$ScriptCommandLine[[3]]], 0];
If[minSum == 0,
  Print[];
  Print["[2] the new grades of q = 3 (fourth order with both contrasts)"];
  ok3 = checkGrade /@ Complement[grades[3], grades[2]];
  Print["    all gates on the four new grades: ", pass[And @@ ok3]];
  Print["    the same grades, the kernel of the tensor's second term:"];
  ok3t = checkTensorKernel /@ Complement[grades[3], grades[2]];
  Print["    tensor-kernel gates on the four new grades: ", pass[And @@ ok3t]]];

Print[];
Print["[3] the further grades of q = 4 and q = 5 (sixth order), ", minSum, " <= D + W <= ", maxSum];
more = Select[Complement[grades[5], grades[3]], minSum <= Total[#] <= maxSum &];
ok5 = checkGrade /@ SortBy[more, Total];
Print["    all gates on these ", Length[more], " grades: ", pass[And @@ ok5]];
Print["    grades not attempted (D + W > ", maxSum, "): ", Select[Complement[grades[5], grades[3]], Total[#] > maxSum &]];
Print["=============================================================="];
