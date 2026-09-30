#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Figure.wl  --  the uniform-layer data of the continuum-limit paper's convergence figure.

   Recomputes, from notebook 7 (normal incidence, 1-D scheme), notebook 9 (oblique, spectral 3-D scheme) and
   notebook 12 (its incident-wave solver) definitions, the relative reflection error against the number of voxel planes n, and writes
   Mathematica/ContinuumLimit_figure_data.json.  The figure itself, with its regression fits, is drawn
   from that file by scripts/plot_convergence_orders.py.  Nothing is transcribed by hand.
   ============================================================================ *)

base = DirectoryName[$InputFileName] <> "";
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
Print["wrote the figure data; normal G1 at n = 32: ", normal["G1"][[-1, 2]], ";  oblique G1 at n = 8: ", oblique["G1"][[-1, 2]],
  ";  SV G1 at n = 8: ", incS["G1"][[-1, 2]]];
