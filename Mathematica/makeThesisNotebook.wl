#!/usr/bin/env wolframscript
(* makeThesisNotebook.wl -- generate faithful .nb twins for the thesis-validation
   .wl scripts (deterministic cell-wrapping; the .wl is the executable source). *)

dir = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
bannerEdge[l_] := StringContainsQ[l, Repeated["=", {10, Infinity}]];
cleanText[c_] := StringTrim[StringReplace[c,
    {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "", Repeated["-", {4, Infinity}] -> ""}]];

makeTwin[wlfile_, title_, intro_] := Module[
   {wlpath, nbpath, body, lines, segs = {}, codeBuf = {}, banBuf = {}, inB = False,
    firstBanner = True, emit, toCell, cells},
  wlpath = dir <> wlfile; nbpath = dir <> StringReplace[wlfile, ".wl" -> ".nb"];
  body = StringReplace[Import[wlpath, "Text"], StartOfString ~~ "#!/usr/bin/env wolframscript" ~~ "\n" -> ""];
  lines = StringSplit[body, "\n"];
  emit[] := (If[codeBuf =!= {} && StringTrim[StringRiffle[codeBuf, "\n"]] =!= "",
      AppendTo[segs, {"code", StringTrim[StringRiffle[codeBuf, "\n"]]}]]; codeBuf = {});
  Do[With[{l = lines[[i]]},
    If[bannerEdge[l],
     If[! inB, emit[]; inB = True; banBuf = {l},
      AppendTo[banBuf, l]; AppendTo[segs, {"banner", StringRiffle[banBuf, "\n"]}]; inB = False; banBuf = {}],
     If[inB, AppendTo[banBuf, l], AppendTo[codeBuf, l]]]], {i, Length[lines]}];
  emit[];
  toCell[{"banner", c_}] := If[firstBanner, firstBanner = False; Cell[cleanText[c], "Text"], Cell[cleanText[c], "Section"]];
  toCell[{"code", c_}] := Cell[c, "Input"];
  cells = Join[{Cell[title, "Title"], Cell[intro, "Text"]}, toCell /@ segs];
  Export[nbpath, Notebook[cells]];
  Print["wrote ", nbpath, " : ", Length[cells], " cells (", Count[cells, Cell[_, "Input", ___]],
     " Input, ", Count[cells, Cell[_, "Section", ___]], " Section)"];
];

makeTwin["ThesisInterfaceRT.wl", "Thesis Section 3.1 : Energy-Normalised Interface R/T",
   "Thesis Section 3.1 energy-normalised displacement-traction eigenbasis D_z and canonical "
   <> "symplectic J6 for two half-spaces, plus the plane-interface R/T scaffold.  Faithful to "
   <> "GRepresentations.tex (Peigen/SVeigen/SHeigen + epsdef + the symplectic identity dinv2).  "
   <> "Conventions: b=(u_z,u_x,u_y,t_z,t_x,t_y); +=downgoing; column order (+P,+S,+H,-P,-S,-H); "
   <> "TIME e^{-i w t} (thesis convention, NOT the project's e^{+i w t}).  Self-checks: symplectic "
   <> "identity, D_z.D_z^{-1}=I, traction=Hooke(displacement), and interface |R|^2+|T|^2=1.  "
   <> "Executable twin: ThesisInterfaceRT.wl."];
