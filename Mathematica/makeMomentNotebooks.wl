#!/usr/bin/env wolframscript
(* Generate a .nb twin for each moment script from its executable .wl.

   Same banner convention as makeCubeT9Notebook.wl / makeCubeT27Notebook.wl:
   a line is a banner edge if it contains a run of >= 10 '=' characters;
   banner blocks become Text cells, everything between them becomes Input.

   The .wl stays the source of truth -- it is what runs and what the gates
   execute.  The .nb is generated from it, never hand-edited, and regenerating
   is how it is kept honest. *)

dir = DirectoryName[$InputFileName];

titles = {
  "CubeMomentCore"     -> {"The cube moment engine",
     "Shared core for the six Green's-tensor moments of the closed set. Every \
moment reduces to scalar moments of r^m over the cube; the derivatives are \
distributional, so the r=0 delta content arrives inside the surface integral \
rather than being added by hand."},
  "CubeMomentCoreTest" -> {"Gate: the cube moment engine",
     "Each check has an answer known without computing the moment. The \
Laplacian sum rule is the sharp one: a spherical excision returns 0 where the \
right answer is -4\[Pi], and no symmetry test can see the difference."},
  "CubeGMoment" -> {"The G moment of a cube",
     "\!\(\*SubscriptBox[\(G\), \(ij\)]\) = \[Integral] \
\!\(\*SubsuperscriptBox[\(G\), \(ij\), \(0\)]\) dV. Grading (D,W) = (0,0). The \
density response of the uniform-displacement block."},
  "CubeKMoment" -> {"The K moment of a cube",
     "\!\(\*SuperscriptBox[\(K\), \(rs\)]\)\!\(\*SubscriptBox[\(\\\ \), \
\(ij\)]\) = \[Integral] \!\(\*SubsuperscriptBox[\(G\), \(ij\), \(0\)]\) \
\!\(\*SubscriptBox[\(x\), \(r\)]\)\!\(\*SubscriptBox[\(x\), \(s\)]\) dV. \
Grading (D,W) = (0,2). The most benign of the six \[LongDash] no delta \
content at all."},
  "CubeNMoment" -> {"The N moment of a cube",
     "\!\(\*SuperscriptBox[\(N\), \(r\)]\) = \[Integral] \
\!\(\*SubscriptBox[\(\[PartialD]\), \(k\)]\)\!\(\*SubsuperscriptBox[\(G\), \
\(in\), \(0\)]\) \!\(\*SubscriptBox[\(x\), \(r\)]\) dV. Grading (D,W) = (1,1) \
\[LongDash] the only moment with both gradings odd, which is why the \
first-gradient block decouples."},
  "CubeMMoment" -> {"The M moment of a cube",
     "\!\(\*SubscriptBox[\(M\), \(in, pk\)]\) = \[Integral] \
\!\(\*SubscriptBox[\(\[PartialD]\), \(p\)]\)\!\(\*SubscriptBox[\(\[PartialD]\), \
\(k\)]\)\!\(\*SubsuperscriptBox[\(G\), \(in\), \(0\)]\) dV. Grading (D,W) = \
(2,0). The Eshelby-type object; cross-checked against the T9 first-principles \
derivation."},
  "CubePMoment" -> {"The P moment of a cube",
     "\!\(\*SuperscriptBox[\(P\), \(rs\)]\) = \[Integral] \
\!\(\*SubscriptBox[\(\[PartialD]\), \(q\)]\)\!\(\*SubscriptBox[\(\[PartialD]\), \
\(p\)]\)\!\(\*SubsuperscriptBox[\(G\), \(ij\), \(0\)]\) \
\!\(\*SubscriptBox[\(x\), \(r\)]\)\!\(\*SubscriptBox[\(x\), \(s\)]\) dV. \
Grading (D,W) = (2,2). The inertial half of the 27\[Times]27 block; \
cross-checked against Pmat.nb."},
  "CubeQMoment" -> {"The Q moment of a cube",
     "\!\(\*SuperscriptBox[\(Q\), \(r\)]\) = \[Integral] \
\!\(\*SubscriptBox[\(\[PartialD]\), \(q\)]\)\!\(\*SubscriptBox[\(\[PartialD]\), \
\(p\)]\)\!\(\*SubscriptBox[\(\[PartialD]\), \(k\)]\)\!\(\*SubsuperscriptBox[\(G\
\), \(in\), \(0\)]\) \!\(\*SubscriptBox[\(x\), \(r\)]\) dV. Grading (D,W) = \
(3,1), the deepest of the six; the elastic half of the 27\[Times]27 block, \
cross-checked against QmatS0 in QMat.nb."}};

bannerEdge[l_] := StringContainsQ[l, Repeated["=", {10, Infinity}]];
cleanText[c_] := StringTrim[StringReplace[c,
   {"(*" -> "", "*)" -> "", Repeated["=", {3, Infinity}] -> "",
    Repeated["-", {4, Infinity}] -> ""}]];

convert[stem_, ttl_, blurb_] := Module[
   {wlpath, nbpath, body, lines, segs, codeBuf, banBuf, inB, emitCode, cells, nb},
   wlpath = FileNameJoin[{dir, stem <> ".wl"}];
   nbpath = FileNameJoin[{dir, stem <> ".nb"}];
   If[! FileExistsQ[wlpath], Print["  SKIP (missing) ", stem]; Return[]];
   body = StringReplace[Import[wlpath, "Text"],
      StartOfString ~~ "#!/usr/bin/env wolframscript" ~~ "\n" -> ""];
   lines = StringSplit[body, "\n"];
   segs = {}; codeBuf = {}; banBuf = {}; inB = False;
   emitCode[] := (
      If[codeBuf =!= {} && StringTrim[StringRiffle[codeBuf, "\n"]] =!= "",
         AppendTo[segs, {"code", StringTrim[StringRiffle[codeBuf, "\n"]]}]];
      codeBuf = {});
   Do[With[{l = lines[[t]]},
      If[bannerEdge[l],
         If[! inB, emitCode[]; inB = True; banBuf = {l},
            AppendTo[banBuf, l];
            AppendTo[segs, {"banner", StringRiffle[banBuf, "\n"]}];
            inB = False; banBuf = {}],
         If[inB, AppendTo[banBuf, l], AppendTo[codeBuf, l]]]],
    {t, Length[lines]}];
   emitCode[];
   cells = Map[
      If[#[[1]] === "banner", Cell[cleanText[#[[2]]], "Text"],
         Cell[BoxData[#[[2]]], "Input"]] &, segs];
   nb = Notebook[
      Join[{Cell[ttl, "Title"], Cell[blurb, "Text"]}, cells],
      WindowSize -> {1200, 850}];
   Export[nbpath, nb];
   Print["  wrote ", FileNameTake[nbpath],
         "   text cells: ", Count[segs, {"banner", _}],
         "   input cells: ", Count[segs, {"code", _}],
         "   re-imported: ",
         Length[Cases[Import[nbpath], Cell[__], Infinity]]]];

Print["generating moment notebooks in ", dir];
Do[convert[t[[1]], t[[2, 1]], t[[2, 2]]], {t, titles}];
Print["done."];
