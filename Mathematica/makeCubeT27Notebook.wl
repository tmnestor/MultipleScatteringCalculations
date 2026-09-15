#!/usr/bin/env wolframscript
(* Generate CubeT27CommutantBlocks.nb from its verified executable twin
   CubeT27CommutantBlocks.wl, then verify faithfulness by re-importing.
   Same banner convention as makeCubeT9Notebook.wl / makeLatticeNotebook.wl:
   a line is a banner edge if it contains a run of >= 10 '=' characters;
   banner blocks become Text cells, everything between them becomes Input. *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
wlpath = dir <> "CubeT27CommutantBlocks.wl";
nbpath = dir <> "CubeT27CommutantBlocks.nb";

body = StringReplace[Import[wlpath, "Text"],
   StartOfString ~~ "#!/usr/bin/env wolframscript" ~~ "\n" -> ""];
lines = StringSplit[body, "\n"];
bannerEdge[l_] := StringContainsQ[l, Repeated["=", {10, Infinity}]];

segs = {}; codeBuf = {}; banBuf = {}; inB = False;
emitCode[] := (
   If[codeBuf =!= {} && StringTrim[StringRiffle[codeBuf, "\n"]] =!= "",
      AppendTo[segs, {"code", StringTrim[StringRiffle[codeBuf, "\n"]]}]];
   codeBuf = {});
Do[With[{l = lines[[i]]},
   If[bannerEdge[l],
      If[! inB, emitCode[]; inB = True; banBuf = {l},
         AppendTo[banBuf, l]; AppendTo[segs, {"banner", StringRiffle[banBuf, "\n"]}];
         inB = False; banBuf = {}],
      If[inB, AppendTo[banBuf, l], AppendTo[codeBuf, l]]]],
 {i, Length[lines]}];
emitCode[];

cleanText[c_] := StringTrim[StringReplace[c,
   {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "",
    Repeated["-", {4, Infinity}] -> ""}]];

cells = Map[
   If[#[[1]] === "banner",
      Cell[cleanText[#[[2]]], "Text"],
      Cell[BoxData[#[[2]]], "Input"]] &, segs];

nb = Notebook[
   Join[
     {Cell["The 27\[Times]27 cubic system: five blocks, explicit and inverted",
        "Title"],
      Cell["Companion to docs/commutant_block_structure.tex. The system M is \
transcribed verbatim from A33.nb; it commutes with O_h acting as \
R\[CircleTimes]R\[CircleTimes]R on \
\!\(\*SuperscriptBox[\(\[DoubleStruckCapitalR]\), \(3\)]\)\[CircleTimes]\
\!\(\*SuperscriptBox[\(\[DoubleStruckCapitalR]\), \(3\)]\)\[CircleTimes]\
\!\(\*SuperscriptBox[\(\[DoubleStruckCapitalR]\), \(3\)]\), so Schur's lemma \
forces it block-diagonal in a basis fixed by the symmetry alone. Evaluate the \
notebook top to bottom.", "Text"]},
     cells],
   WindowSize -> {1200, 850}];
Export[nbpath, nb];

chk = Import[nbpath];
Print["wrote ", nbpath];
Print["  segments: ", Length[segs],
      "   text cells: ", Count[segs, {"banner", _}],
      "   input cells: ", Count[segs, {"code", _}]];
Print["  re-imported cell count: ", Length[Cases[chk, Cell[__], Infinity]]];
