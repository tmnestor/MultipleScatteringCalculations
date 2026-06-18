#!/usr/bin/env wolframscript
(* Generate the Phase-1 deliverable notebook IntraPlaneLatticeSum.nb from the
   verified executable twin IntraPlaneLatticeSum.wl, then verify faithfulness by
   re-importing, extracting the Input cells and evaluating them.  Robust line-based
   split: a line is a banner edge only if it contains a long (>=10) run of '='. *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
wlpath = dir <> "IntraPlaneLatticeSum.wl";
nbpath = dir <> "IntraPlaneLatticeSum.nb";

body = StringReplace[Import[wlpath, "Text"], StartOfString ~~ "#!/usr/bin/env wolframscript" ~~ "\n" -> ""];
lines = StringSplit[body, "\n"];
bannerEdge[l_] := StringContainsQ[l, Repeated["=", {10, Infinity}]];

segs = {}; codeBuf = {}; banBuf = {}; inB = False;
emitCode[] := (
   If[codeBuf =!= {} && StringTrim[StringRiffle[codeBuf, "\n"]] =!= "",
      AppendTo[segs, {"code", StringTrim[StringRiffle[codeBuf, "\n"]]}]];
   codeBuf = {});
Do[
  With[{l = lines[[i]]},
   If[bannerEdge[l],
    If[! inB, emitCode[]; inB = True; banBuf = {l},
     AppendTo[banBuf, l]; AppendTo[segs, {"banner", StringRiffle[banBuf, "\n"]}]; inB = False; banBuf = {}],
    If[inB, AppendTo[banBuf, l], AppendTo[codeBuf, l]]]],
  {i, Length[lines]}];
emitCode[];

cleanText[c_] := StringTrim[StringReplace[c,
    {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "", Repeated["-", {4, Infinity}] -> ""}]];
firstBanner = True;
toCell[{"banner", c_}] := (If[firstBanner, firstBanner = False; Cell[cleanText[c], "Text"],
    Cell[cleanText[c], "Section"]]);
toCell[{"code", c_}] := Cell[c, "Input"];

cells = Join[
   {Cell["Phase 1 : Ewald-Accelerated Intra-Plane Lattice Sum", "Title"],
    Cell["Planar lattice sum of the Phase-0 translation operator at fixed Bloch vector k_par, "
       <> "giving the intra-plane coupling G0(k_par) = Sum_{R!=0} beta(R) e^{i k_par.R}. TB1: damped "
       <> "direct sum, structure constants, reciprocity. TB2: Ewald acceleration (real + reciprocal "
       <> "split, EwaldIntraPlanePropagator.tex), eta-independent to machine precision for the "
       <> "conditionally-convergent (undamped) sum. TB3: Ewald <-> multipole-G0 connection. "
       <> "Executable twin: IntraPlaneLatticeSum.wl. Independently validated against Python in "
       <> "cubic_scattering/tests/test_intraplane_lattice.py.", "Text"]},
   toCell /@ segs];

Export[nbpath, Notebook[cells]];
Print["wrote ", nbpath, " : ", Length[cells], " cells (",
   Count[cells, Cell[_, "Input", ___]], " Input, ", Count[cells, Cell[_, "Section", ___]], " Section)"];

nbimp = Import[nbpath];
icells = Cases[nbimp, Cell[c_String, "Input", ___] :> c, Infinity];
Print["re-imported .nb; evaluating ", Length[icells], " extracted Input cells ..."];
Do[ToExpression[icells[[i]]], {i, Length[icells]}];
Print["IntraPlaneLatticeSum.nb verified: extracted cells reproduce the run above."];
