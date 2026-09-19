#!/usr/bin/env wolframscript
(* Stage 2: TEST every harvested variant against direct differentiation.

   `MyRecursiveHarmonicDerivatives.nb` defines d_1_3 six times, d_1 and d_2
   four times each, and ap/am/bp/bm twice, all with differing right-hand sides.
   Nothing in the document says which is current.  But each operator has a
   knowable answer: applied to the scalar spherical wave function it must equal
   the corresponding Cartesian partial derivative of that same function written
   in x, y, z.  So the variants are measured, not chosen.

       d_i  [K,n,m,r,th,ph][J]  ==  D[ J(x,y,z), x_i ]
       d_ij [K,n,m,r,th,ph][J]  ==  D[ J(x,y,z), x_i, x_j ]

   The ladder coefficients are tested first and separately, because every d
   operator depends on them: if ap/am/bp/bm were wrong, EVERY variant would
   fail and the failure would say nothing about the variants.

   Tested at several (n, m) and at a point off every axis and off every
   coordinate plane, so a term that happens to vanish on a symmetry plane
   cannot pass by accident.

   Run:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/verifyDerivativeOperators.wl
*)

defs = Get["/tmp/archive_all_defs.m"];
Print["=== ", Length[defs], " harvested definitions ==="];

lhsStr[HoldComplete[SetDelayed[l_, _]]] :=
  StringReplace[ToString[InputForm[HoldComplete[l]]],
    {StartOfString ~~ "HoldComplete[" -> "", "]" ~~ EndOfString -> ""}];

key[d_] := Module[{s = lhsStr[d]},
  Which[
    StringStartsQ[s, "Subscript[d, "],
      StringTake[s, {14, StringPosition[s, "]"][[1, 1]] - 1}],
    True, First[StringSplit[s, "["]]]];

install[d_] := ReleaseHold[d /. HoldComplete -> Hold];

(* ---------------------------------------------------------------- *)
(* The ladder coefficients, and the scalar wave function.            *)
(* ---------------------------------------------------------------- *)

ladderIdx = Flatten[Position[key /@ defs, #] & /@ {"ap", "am", "bp", "bm"}];
Print["  ladder coefficient definitions at: ", ladderIdx];

(* Are the two sets the same?  Install the first set, record; install the
   second, compare.  If they differ the difference is reported rather than
   silently resolved by load order. *)
Clear[ap, am, bp, bm];
Do[install[defs[[i]]], {i, Select[ladderIdx, # <= 10 &]}];
firstSet = Table[{ap[n, m], am[n, m], bp[n, m], bm[n, m]}, {n, 0, 4}, {m, -n, n}];
Clear[ap, am, bp, bm];
Do[install[defs[[i]]], {i, Select[ladderIdx, # > 10 &]}];
secondSet = Table[{ap[n, m], am[n, m], bp[n, m], bm[n, m]}, {n, 0, 4}, {m, -n, n}];
Print["  the two ladder sets agree: ",
  Simplify[firstSet - secondSet] === 0 * firstSet];

(* Keep the second (later) set installed; the agreement check above says
   whether that choice is material. *)

Clear[J];
J[K_, n_, m_, r_, th_, ph_] :=
  SphericalBesselJ[n, K r] SphericalHarmonicY[n, m, th, ph];

(* Cartesian form of the same function, for direct differentiation. *)
rX[x_, y_, z_] := Sqrt[x^2 + y^2 + z^2];
thX[x_, y_, z_] := ArcCos[z/rX[x, y, z]];
phX[x_, y_, z_] := ArcTan[x, y];
GX[K_, n_, m_][x_, y_, z_] :=
  SphericalBesselJ[n, K rX[x, y, z]] *
    SphericalHarmonicY[n, m, thX[x, y, z], phX[x, y, z]];

(* Test point, off every axis and plane. *)
$p = {37/100, -61/100, 83/100};
$K = 11/10;
$cases = {{1, 0}, {1, 1}, {2, 1}, {2, -1}, {3, 2}};

sph[p_] := {Sqrt[p . p], ArcCos[p[[3]]/Sqrt[p . p]], ArcTan[p[[1]], p[[2]]]};

directD[K_, n_, m_, vars_] := Module[{x, y, z, e},
  e = GX[K, n, m][x, y, z];
  Do[e = D[e, {x, y, z}[[v]]], {v, vars}];
  N[e /. Thread[{x, y, z} -> $p], 30]];

opD[d_, K_, n_, m_] := Module[{sp = sph[$p], res},
  Clear[dop];
  res = Quiet@Check[N[install[d]; Evaluate[ToExpression[
      "Subscript[d, " <> key[d] <> "][" <> ToString[InputForm[K]] <> ", " <>
      ToString[n] <> ", " <> ToString[m] <> ", " <>
      ToString[InputForm[sp[[1]]]] <> ", " <> ToString[InputForm[sp[[2]]]] <> ", " <>
      ToString[InputForm[sp[[3]]]] <> "][J]"]], 30], $Failed];
  res];

(* ---------------------------------------------------------------- *)
(* The sweep.                                                        *)
(* ---------------------------------------------------------------- *)

Print["\n--- testing each variant against direct differentiation ---"];
Print["    (relative difference, worst over ", Length[$cases], " (n,m) cases)"];

dIdx = Select[Range[Length[defs]], StringStartsQ[lhsStr[defs[[#]]], "Subscript[d, "] &];
groups = GroupBy[dIdx, key[defs[[#]]] &];

verified = <||>;
Do[
  Module[{k = kk, idxs = groups[kk], vars, best = None, bestErr = Infinity},
    vars = ToExpression /@ StringSplit[k, ", "];
    Print["\n  d_", k, "   variants at ", idxs];
    Do[
      Module[{err = 0, got, want, ok = True},
        Clear[d];
        Do[install[defs[[j]]], {j, Select[ladderIdx, # > 10 &]}];
        install[defs[[ii]]];
        Do[
          got = opD[defs[[ii]], $K, cse[[1]], cse[[2]]];
          want = directD[$K, cse[[1]], cse[[2]], vars];
          If[got === $Failed || ! NumericQ[Abs[want]],
            ok = False,
            err = Max[err, Abs[got - want]/Max[Abs[want], 10^-30]]],
          {cse, $cases}];
        Print["    index ", ii, "   ", If[ok, ToString[N[err, 3]], "ERROR"]];
        If[ok && err < bestErr, bestErr = err; best = ii]],
      {ii, idxs}];
    If[best =!= None && bestErr < 10^-20,
      verified[k] = best;
      Print["    -> VERIFIED: index ", best, "  (", N[bestErr, 3], ")"],
      Print["    -> NO VARIANT PASSES (best ", N[bestErr, 3], ")"]]],
  {kk, Keys[groups]}];

Print["\n=== summary ==="];
Do[Print["  d_", k, "  <- definition index ", verified[k]], {k, Keys[verified]}];
Print["  verified ", Length[verified], " of ", Length[groups], " operators"];
Export["/tmp/verified_ops.m", verified];
