#!/usr/bin/env wolframscript
(* Emit the VERIFIED archive definitions as a self-contained Wolfram source
   block, for inclusion in MieSphericalWaves.wl.

   The definitions are written out programmatically rather than transcribed.
   Several of the d_ij bodies exceed a thousand characters and some are defined
   in terms of one another, so hand-copying is precisely how a silent error
   would enter -- and this project's record says representation and transcription
   bridges are where its defects have actually lived.

   Which variant is emitted is not a judgement: verifyDerivativeOperators.wl
   tests every variant against direct Cartesian differentiation and records the
   index of one that reproduces it to 1e-29.  (In the event ALL variants agree,
   so the choice is immaterial -- but it is measured rather than assumed.)

   Run after verifyDerivativeOperators.wl:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/emitVerifiedDefs.wl
*)

defs = Get["/tmp/archive_all_defs.m"];
verified = Get["/tmp/verified_ops.m"];

lhsStr[HoldComplete[SetDelayed[l_, _]]] :=
  StringReplace[ToString[InputForm[HoldComplete[l]]],
    {StartOfString ~~ "HoldComplete[" -> "", "]" ~~ EndOfString -> ""}];
rhsStr[HoldComplete[SetDelayed[_, r_]]] :=
  StringReplace[ToString[InputForm[HoldComplete[r]]],
    {StartOfString ~~ "HoldComplete[" -> "", "]" ~~ EndOfString -> ""}];
src[i_] := lhsStr[defs[[i]]] <> " :=\n  " <> rhsStr[defs[[i]]] <> ";";

nameAt[i_] := First[StringSplit[lhsStr[defs[[i]]], "["]];

(* Fixed pieces: the coordinate maps, the ladder coefficients (the two sets in
   the notebook were checked identical), the scalar wave function, the solid
   harmonic and the traction operator. *)
fixed = {1, 2, 3, 4, 5, 6, 7, 8, 9, 53};

out = {};
AppendTo[out, "(* ---------------------------------------------------------------"];
AppendTo[out, "   HARVESTED FROM THE RESEARCH ARCHIVE"];
AppendTo[out, "   MyRecursiveHarmonicDerivatives.nb, in"];
AppendTo[out, "   ~/Documents/WaveTheory-Research/MathematicaSourceCode/GreensTensorCalculations"];
AppendTo[out, ""];
AppendTo[out, "   Emitted programmatically by Mathematica/emitVerifiedDefs.wl."];
AppendTo[out, "   DO NOT EDIT BY HAND -- regenerate instead, so that the source of"];
AppendTo[out, "   truth stays the archive plus the verification, not this copy."];
AppendTo[out, "   --------------------------------------------------------------- *)"];
AppendTo[out, ""];

Do[
  AppendTo[out, "(* archive definition " <> ToString[i] <> ": " <> nameAt[i] <> " *)"];
  AppendTo[out, src[i]];
  AppendTo[out, ""],
  {i, fixed}];

AppendTo[out, "(* The Cartesian derivative operators.  Each index below is the"];
AppendTo[out, "   variant that verifyDerivativeOperators.wl measured against direct"];
AppendTo[out, "   differentiation; all variants in the archive agreed, to 1e-29. *)"];
AppendTo[out, ""];
Do[
  With[{i = verified[k]},
    AppendTo[out, "(* d_" <> k <> "  <- archive definition " <> ToString[i] <>
      ", verified = D[.., x_" <> StringReplace[k, ", " -> "] D[.., x_"] <> "] *)"];
    AppendTo[out, src[i]];
    AppendTo[out, ""]],
  {k, Keys[verified]}];

Export["/tmp/verified_defs_block.wl", StringRiffle[out, "\n"], "Text"];
Print["wrote /tmp/verified_defs_block.wl  (",
  StringLength[StringRiffle[out, "\n"]], " chars, ",
  Length[fixed] + Length[verified], " definitions)"];
