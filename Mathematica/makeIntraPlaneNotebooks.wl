#!/usr/bin/env wolframscript
(* Generalised deliverable-notebook generator for the Phase-2 .wl files.  Splits each
   executable .wl into Title/Text/Section/Input cells on the long-'=' banner edges (the
   same robust rule as makeIntraPlaneNotebook.wl) and writes the .nb twin.  Generation is
   a faithful, deterministic cell-wrapping of the .wl source; the .wl scripts are already
   self-verified (and Python-cross-checked), so round-trip is spot-checked separately. *)

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

makeTwin["IntraPlaneFoldyLax.wl", "Phase 2 TB1 : Single-Site T0 + Foldy-Lax Scaffold",
   "Assembles the single-site full-wave Mie T0 in the L/M/N multipole basis from the verified "
   <> "CartesianT0.wl (spheroidal L-N 2x2, toroidal M, n=0 monopole), the collective "
   <> "T_coll = T0 (I - G0 T0)^{-1}, the single-voxel limit, and the monopole-channel collective "
   <> "solve with the Phase-1 scalar G0.  Executable twin: IntraPlaneFoldyLax.wl."];

makeTwin["IntraPlaneVectorTranslation.wl", "Phase 2 TB2 : Vector Translation as an Explicit L/M/N Matrix",
   "Extracts the Phase-0 elastic vector translation as an explicit matrix W^{c'c}_{nu mu,n m}(d): "
   <> "L->L via beta^P, and M,N->M,N by sign-safe projection onto the orthogonal P/B/C vector "
   <> "spherical harmonics.  Executable twin: IntraPlaneVectorTranslation.wl."];

makeTwin["IntraPlaneTwoBody.wl", "Phase 2 (a) : Two-Voxel Direct Foldy-Lax",
   "Two-voxel direct Foldy-Lax with the pairwise vector translation W(d).  Verifies the field "
   <> "reconstruction of W, the isolated limit, the direct Neumann/Born multiple-scattering series "
   <> "against the matrix inverse, the closed-form monopole two-body solve, and the fixed point.  "
   <> "Executable twin: IntraPlaneTwoBody.wl."];

makeTwin["IntraPlaneVectorLattice.wl", "Phase 2 (b) : Lattice-Summed Multi-Channel Vector G0(k_par)",
   "The planar Bloch sum of the pairwise vector translation: L->L closed form (Phase-1 scalar G0 at "
   <> "kappa_P), M,N by damped direct lattice sum.  Verified by vector field reconstruction against "
   <> "a direct damped Bloch sum, geometric convergence, and the collective solve.  Executable twin: "
   <> "IntraPlaneVectorLattice.wl.  Independently cross-checked in Python "
   <> "(cubic_scattering/tests/test_intraplane_collective.py)."];

makeTwin["IntraPlaneCollectiveReciprocity.wl", "Phase 2 (d) : Collective Reciprocity (Symplectic Metric)",
   "Reciprocity of the collective Foldy-Lax operator via the symplectic-J reconciliation.  A single "
   <> "symplectic channel metric D = diag((kS/kP)^{3/2}, I sqrt(n(n+1)), sqrt(n(n+1))) makes BOTH T0 "
   <> "and G0 sigma-symmetric, so T_coll is reciprocal.  Executable twin: "
   <> "IntraPlaneCollectiveReciprocity.wl.  Cross-checked in Python."];

Print["Phase-2 notebook twins generated."];
