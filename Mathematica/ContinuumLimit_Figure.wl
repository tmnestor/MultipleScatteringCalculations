#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Figure.wl  --  the convergence figure of the continuum-limit paper.

   Recomputes, from notebook 7 (normal incidence, 1-D scheme), notebook 9 (oblique, spectral 3-D scheme) and
   notebook 12 (its incident-wave solver) definitions, the relative reflection error against the number of voxel planes n, and writes
   LatexPDFs/ContinuumLimit/convergence.pdf.  Nothing is transcribed by hand.
   ============================================================================ *)

base = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
loadDefs[file_, stop_] := Module[{txt = Import[base <> file, "Text"]},
   ToExpression[StringTake[txt, StringPosition[txt, stop][[1, 1]] - 1], InputForm]];

(* normal incidence: notebook 7 (collocation C, mean-only G0, first moment G1), omega = 300 *)
loadDefs["ContinuumLimit_FourthOrder.wl", "Print[\"==== ContinuumLimit_FourthOrder ::"];
om = 300; ex = N[exactScat[om], 40];
nN = {1, 2, 4, 8, 16, 32};
normal = Association[Table[sc -> Table[{n, Abs[((solveLayer[om, n, If[sc == "G1", 2, 1], If[sc == "C", "C", "G"]] - ex)/ex)[[1]]]},
      {n, nN}], {sc, {"C", "G0", "G1"}}]];
Print["normal incidence done"];

(* oblique, 20 degrees, the converted S wave: notebook 9, lattice sum |p|, |q| <= 8 *)
loadDefs["ContinuumLimit_Oblique.wl", "pMax = 8;"];
om0 = 300; kx0 = N[om0/al Sin[20 Degree], 20]; exO = exactRefl[om0, kx0];
nO = {1, 2, 4, 8};
oblique = Association[Table[sc -> Table[{n, relErr[solveOblique[om0, kx0, n, If[sc == "G1", {1, 2, 3}, {1}], 8][[1]], exO[[2]]]},
      {n, nO}], {sc, {"G0", "G1"}}]];
Print["oblique done"];

(* incident SV at 20 degrees, the reflected S wave: notebook 12's solver on notebook 9's definitions *)
nb12 = Import[base <> "ContinuumLimit_IncidentS.wl", "Text"];
ToExpression[StringTake[nb12, {StringPosition[nb12, "(* the incident wave generalised"][[1, 1]],
    StringPosition[nb12, "pMax = 8;
(* [3]"][[1, 1]] - 1}], InputForm];
kxS = N[om0/be Sin[20 Degree], 20]; exS = exactReflSV[om0, kxS];
incS = Association[Table[sc -> Table[{n, relErr[solveInc[om0, kxS, n, If[sc == "G1", {1, 2, 3}, {1}], 8, "SV"][[1]], exS[[2]]]},
      {n, nO}], {sc, {"G0", "G1"}}]];
Print["incident SV done"];

(* cache the data, so the layout can be revised without recomputing the lattice sums *)
Export[base <> "ContinuumLimit_figure_data.json", <|"normal" -> N[normal], "oblique" -> N[oblique], "incidentSV" -> N[incS]|>, "RawJSON"];
guide[p_, n0_, e0_] := Table[{n, e0 (n0/n)^p}, {n, {1, 32}}];
styleOf = <|"C" -> Directive[Gray, Dashed], "G0" -> Directive[RGBColor[0.2, 0.4, 0.8]], "G1" -> Directive[RGBColor[0.8, 0.2, 0.2]]|>;
plot = ListLogLogPlot[
   {normal["C"], normal["G0"], normal["G1"], oblique["G0"], oblique["G1"], incS["G0"], incS["G1"],
    guide[2, 1, 4*^-3], guide[4, 1, 1.5*^-6]},
   Joined -> {True, True, True, True, True, True, True, True, True},
   PlotMarkers -> {"\[FilledCircle]", "\[FilledSquare]", "\[FilledDiamond]", "\[EmptySquare]", "\[EmptyDiamond]",
     "\[EmptyUpTriangle]", "\[EmptyDownTriangle]", None, None},
   PlotStyle -> {styleOf["C"], styleOf["G0"], styleOf["G1"],
     Directive[RGBColor[0.1, 0.6, 0.5], Dashed], Directive[RGBColor[0.9, 0.55, 0.1], Dashed],
     Directive[RGBColor[0.45, 0.25, 0.7], Dotted], Directive[RGBColor[0.55, 0.35, 0.15], Dotted],
     Directive[GrayLevel[0.6], Thin], Directive[GrayLevel[0.6], Thin]},
   PlotLegends -> Placed[LineLegend[{"collocation, normal", "mean only, normal", "first moment, normal",
       "mean only, P\[Rule]S 20\[Degree]", "first moment, P\[Rule]S 20\[Degree]",
       "mean only, SV\[Rule]S 20\[Degree]", "first moment, SV\[Rule]S 20\[Degree]", "slope 2 / slope 4"},
      LegendMarkers -> {"\[FilledCircle]", "\[FilledSquare]", "\[FilledDiamond]", "\[EmptySquare]", "\[EmptyDiamond]",
        "\[EmptyUpTriangle]", "\[EmptyDownTriangle]", None},
      LabelStyle -> {FontFamily -> "Times", 10}, LegendLayout -> {"Column", 2}], Below],
   Frame -> True, FrameLabel -> {"planes of voxels, n", "relative reflection error"},
   FrameStyle -> Directive[Black, 11], LabelStyle -> {FontFamily -> "Times", 11},
   PlotRange -> {{0.9, 36}, {1*^-14, 1*^-2}}, ImageSize -> 480, GridLines -> None];
Export["/Users/tod/Desktop/MultipleScatteringCalculations/LatexPDFs/ContinuumLimit/convergence.pdf", plot];
Print["wrote convergence.pdf; normal G1 at n = 32: ", normal["G1"][[-1, 2]], ";  oblique G1 at n = 8: ", oblique["G1"][[-1, 2]],
  ";  SV G1 at n = 8: ", incS["G1"][[-1, 2]]];
