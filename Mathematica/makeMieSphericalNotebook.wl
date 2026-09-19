#!/usr/bin/env wolframscript
(* Generate MieSphericalWaves.nb from its verified executable twin
   MieSphericalWaves.wl, then verify faithfulness by re-importing and counting
   cells.  Same banner convention as makeMatrixVectorNotebook.wl: a line is a
   banner edge if it contains a run of >= 10 '=' characters; banner blocks
   become Text cells, everything between them becomes Input cells.

   The .wl is itself a build product (template + machine-emitted archive
   definitions), so the chain is

     archive notebook -> harvest -> verify -> emit -> assemble -> .wl -> .nb

   and no step transcribes anything by hand. *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
wlpath = dir <> "MieSphericalWaves.wl";
nbpath = dir <> "MieSphericalWaves.nb";

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

If[inB,
  Print["**** unbalanced banner: an opening '=' rule has no closing rule ****"];
  Exit[1]];

cleanText[c_] := StringTrim[StringReplace[c,
   {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "",
    Repeated["-", {4, Infinity}] -> ""}]];

cells = Map[
   If[#[[1]] === "banner",
      Cell[cleanText[#[[2]]], "Text"],
      Cell[BoxData[#[[2]]], "Input"]] &, segs];

nb = Notebook[Prepend[cells,
   Cell["Spherical vector waves: the elastic Mie field, its traction, and its \
plane-wave spectrum", "Title"]], WindowSize -> {1100, 800}];
Export[nbpath, nb];

chk = Import[nbpath];
Print["wrote ", nbpath];
Print["  segments: ", Length[segs],
      "   text cells: ", Count[segs, {"banner", _}],
      "   input cells: ", Count[segs, {"code", _}]];
Print["  re-imported cell count: ", Length[Cases[chk, Cell[__], Infinity]]];
