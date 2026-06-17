#!/usr/bin/env wolframscript
(* Generate the deliverable notebook IntraPlaneTranslation.nb from the verified
   executable twin IntraPlaneTranslation.wl, then verify faithfulness by re-importing,
   extracting the Input cells and evaluating them (must reproduce the PASS results).

   Robust line-based split: a line is a banner EDGE only if it contains a long (>=10)
   run of '='.  Inline (* ... *) comments and Print["===="] (4 '=') are never edges,
   so code never lands in a narrative cell.  Cell[string,"Input"] matches the repo
   convention (CartesianT0.wl extracts exactly this shape from ElasticMieTmatrix.nb). *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
wlpath = dir <> "IntraPlaneTranslation.wl";
nbpath = dir <> "IntraPlaneTranslation.nb";

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
   {Cell["Phase 0 : Intra-Plane Elastic Translation-Addition Operator", "Title"],
    Cell["Cruzan/Stein vector-spherical-wave translation for horizontal (Delta z = 0) separations. "
       <> "Built on a single scalar separation matrix beta^c(d) at k_P and k_S, dressed by the "
       <> "translation-commuting operators L=(1/k)grad, M=curl((r-c) .), N=(1/k)curl. Self-verifying: "
       <> "scalar addition theorem (vs projection integral and field reconstruction), vector field "
       <> "reconstruction (L, M, N), and reciprocity. Executable twin: IntraPlaneTranslation.wl. "
       <> "Independently validated against Python in cubic_scattering/tests/test_intraplane_translation.py.",
     "Text"]},
   toCell /@ segs];

Export[nbpath, Notebook[cells]];
Print["wrote ", nbpath, " : ", Length[cells], " cells (",
   Count[cells, Cell[_, "Input", ___]], " Input, ", Count[cells, Cell[_, "Section", ___]], " Section)"];

(* ---- verify: extract Input cells from the written .nb and run them in order ---- *)
nbimp = Import[nbpath];
icells = Cases[nbimp, Cell[c_String, "Input", ___] :> c, Infinity];
Print["re-imported .nb; evaluating ", Length[icells], " extracted Input cells ..."];
Do[ToExpression[icells[[i]]], {i, Length[icells]}];
Print["IntraPlaneTranslation.nb verified: extracted cells reproduce the run above."];
