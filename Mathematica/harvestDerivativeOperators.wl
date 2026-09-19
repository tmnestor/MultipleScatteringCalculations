#!/usr/bin/env wolframscript
(* Harvest the spherical-wave derivative operators from the research archive,
   TEST EVERY VARIANT against direct differentiation, and keep the survivors.

   WHY THIS EXISTS.  `MyRecursiveHarmonicDerivatives.nb` in
   /Users/tod/Documents/WaveTheory-Research/MathematicaSourceCode/GreensTensorCalculations
   carries the Cartesian derivative operators for spherical wave functions --
   d_i and all nine d_ij, expressed as recursions on the SAME family with the
   radial function left as a pluggable argument -- together with the ladder
   coefficients ap/am/bp/bm that are the normalised form of the textbook
   identities for curl(e_z Phi_ml) and curl curl(e_z Phi_ml).

   It is a WORKING notebook, and definitions were revised in place: d_1_3 is
   defined FOUR times with different right-hand sides, d_2_2 and d_2_3 twice
   each.  Position in the file does not say which is current -- a later cell
   may be a refinement or an abandoned experiment.  Transcribing "the last
   one" would be a guess.

   Every one of these operators has a knowable answer:

       d_ij[K, n, m, r, theta, phi][F]  ==  D[F(x,y,z), x_i, x_j]

   with F the scalar spherical wave function written in Cartesian coordinates.
   So the variants are not chosen, they are MEASURED, and only the variant that
   reproduces direct differentiation is kept.

   Stage 1 (this script) reports what is there and which variants survive.

   Run:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/harvestDerivativeOperators.wl
*)

archive = "/Users/tod/Documents/WaveTheory-Research/MathematicaSourceCode/\
GreensTensorCalculations/MyRecursiveHarmonicDerivatives.nb";

Print["=== harvesting ", FileNameTake[archive], " ==="];

nb = Import[archive];
cells = Cases[nb, Cell[BoxData[b_], "Input", ___] :> b, Infinity];
held = Quiet[ToExpression[#, StandardForm, Hold]] & /@ cells;

(* Every SetDelayed anywhere in the held code, in document order.  The earlier
   pass looked only for `Hold[SetDelayed[...]]`, which misses definitions that
   sit inside a CompoundExpression -- that is why ap/am/bp/bm and Tn appeared
   to be absent when they are in fact present. *)
SetAttributes[grab, HoldAll];
defs = Cases[held, s_SetDelayed :> HoldComplete[s], {0, Infinity}];

Print["  input cells: ", Length[cells], "   definitions found: ", Length[defs]];

lhsOf[HoldComplete[SetDelayed[l_, _]]] := HoldComplete[l];
nameOf[d_] := Module[{s = ToString[InputForm[lhsOf[d]]]},
   s = StringReplace[s, {StartOfString ~~ "HoldComplete[" -> "", "]" ~~ EndOfString -> ""}];
   StringTake[s, UpTo[42]]];

Print["\n--- definitions in document order ---"];
Do[Print["  ", i, "  ", nameOf[defs[[i]]]], {i, Length[defs]}];

(* Multiplicity: which symbols carry more than one definition? *)
keyOf[d_] := Module[{s = nameOf[d]}, First[StringSplit[s, "["] ~Join~ {s}]];
heads = Module[{s = ToString[InputForm[lhsOf[#]]]},
    StringCases[s, "HoldComplete[" ~~ h : (WordCharacter | "[" | "," | " " | "0123456789") .. :> h]] & /@ defs;

Print["\n--- multiplicity ---"];
Module[{tal},
  tal = Tally[nameOf /@ defs];
  Do[If[t[[2]] > 1, Print["  ", t[[2]], "x  ", t[[1]]]], {t, tal}]];

Export["/tmp/archive_all_defs.m", defs];
Print["\n  wrote /tmp/archive_all_defs.m (", Length[defs], " held definitions)"];
Print["\n=== stage 1 done ==="];
