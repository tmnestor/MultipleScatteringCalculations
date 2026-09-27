#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_IncidentSExport.wl  --  reference values of notebook 12's scheme
   for the independent Python implementation (scripts/crosscheck_first_moment_voxel.py).

   Loads notebook 9's definitions and notebook 12's incident-wave solver (not their
   checks) and exports the RAW specular reflection displacements {S part, P part}
   for incident SV and SH, at |p|, |q| <= 2, as ContinuumLimit_ObliqueExport.wl does
   for incident P.
   ============================================================================ *)

base = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/";
nb9 = Import[base <> "ContinuumLimit_Oblique.wl", "Text"];
ToExpression[StringTake[nb9, StringPosition[nb9, "pMax = 8;"][[1, 1]] - 1], InputForm];
nb12 = Import[base <> "ContinuumLimit_IncidentS.wl", "Text"];
ToExpression[StringTake[nb12, {StringPosition[nb12, "(* the incident wave generalised"][[1, 1]],
    StringPosition[nb12, "pMax = 8;\n(* [3]"][[1, 1]] - 1}], InputForm];

reim[z_] := {Re[z], Im[z]};
cases = Flatten[Table[
    Module[{kx = If[th == 0, 0, N[om/be Sin[th Degree], 20]], r, pm = If[th == 0, 0, 2]},
     r = solveInc[om, kx, n, bset, pm, inc];
     <|"incident" -> inc, "omega" -> om, "theta_deg" -> th, "kx" -> N[kx], "n" -> n, "basis" -> bset, "p_max" -> pm,
      "refl_S" -> Map[reim, N[r[[1]]]], "refl_P" -> Map[reim, N[r[[2]]]]|>],
    {inc, {"SV", "SH"}}, {om, {300}}, {th, {0, 20}}, {n, {1, 2, 4}}, {bset, {{1}, {1, 2, 3}}}], 4];
Export[base <> "ContinuumLimit_incidentS_ref.json",
  <|"alpha" -> al, "beta" -> be, "rho" -> rho, "contrast" -> <|"dlambda" -> dLam, "dmu" -> dMu, "drho" -> dRho|>,
   "D" -> dLayer, "basis_order" -> "Legendre degrees in (z, x, y): 1 mean, 2 z-moment, 3 x-moment, 4 y-moment",
   "cases" -> cases|>, "RawJSON"];
Print["exported ", Length[cases], " cases"];
