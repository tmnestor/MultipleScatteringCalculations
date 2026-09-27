#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_ObliqueExport.wl  --  reference values of notebook 9's scheme for
   the independent Python implementation (scripts/crosscheck_first_moment_voxel.py).

   Loads notebook 9's definitions (not its checks) and exports the RAW specular
   reflection displacements {S part, P part} of the Galerkin chain at a modest
   lattice truncation |p|, |q| <= 2, so the cross-check compares the discrete scheme
   itself, entry by entry, not only its convergence against the exact layer.
   ============================================================================ *)

nbText = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_Oblique.wl", "Text"];
ToExpression[StringTake[nbText, StringPosition[nbText, "pMax = 8;"][[1, 1]] - 1], InputForm];
reim[z_] := {Re[z], Im[z]};
cases = Flatten[Table[
    Module[{kx = If[th == 0, 0, N[om/al Sin[th Degree], 20]], r},
     r = solveOblique[om, kx, n, bset, If[th == 0, 0, 2]];
     <|"omega" -> om, "theta_deg" -> th, "kx" -> N[kx], "n" -> n, "basis" -> bset, "p_max" -> If[th == 0, 0, 2],
      "refl_S" -> Map[reim, N[r[[1]]]], "refl_P" -> Map[reim, N[r[[2]]]]|>],
    {om, {300}}, {th, {0, 20}}, {n, {1, 2, 4}}, {bset, {{1}, {1, 2, 3}}}], 3];
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_oblique_ref.json",
  <|"alpha" -> al, "beta" -> be, "rho" -> rho, "contrast" -> <|"dlambda" -> dLam, "dmu" -> dMu, "drho" -> dRho|>,
   "D" -> dLayer, "basis_order" -> "Legendre degrees in (z, x, y): 1 mean, 2 z-moment, 3 x-moment, 4 y-moment",
   "cases" -> cases|>, "RawJSON"];
Print["exported ", Length[cases], " cases"];
