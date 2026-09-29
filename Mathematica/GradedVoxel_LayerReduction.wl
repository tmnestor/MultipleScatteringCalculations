#!/usr/bin/env wolframscript
(* ============================================================================
   GradedVoxel_LayerReduction.wl -- gate G4a: a layer of identical graded cubes,
   graded along z only, reduces to notebook 15's one-dimensional graded scheme.

   Notebook 8 (ContinuumLimit_Reduction3D.wl) showed that the 3-D first-moment
   voxel in a laterally uniform layer at normal incidence reduces EXACTLY to the
   1-D scheme of notebook 7: the lateral first moments are never excited, and the
   mean and z-moment carry laterally constant weights whose lateral form factor
   vanishes on the reciprocal lattice.  A contrast linear in z within each cell
   keeps both facts, and its Legendre linearisation along z is notebook 15's:

   [1] the lateral first moment of (1 + g z) e^{i k z} over a voxel vanishes
       (odd in x, the product is even), so the lateral moments stay unexcited;
   [2] with a z-only gradient the linearised source functions of the mean and the
       z-moment are {1, xi_z, xi_z^2}: laterally constant, so notebook 8's form
       factor argument applies unchanged (and its sinc vanishes on the lattice);
   [3] the 3-D monomial linearisation restricted to z, converted to Legendre
       P0..P2, equals notebook 15's A_cbe = (2e+1)/2 int P_c P_b P_e for
       c, b in {0, 1}.
   ============================================================================ *)

oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
Print["==== GradedVoxel_LayerReduction (G4a) ===="];
Clear[xx, zz, hh, gg, kk, pp, xi];

(* [1] *)
Module[{mom = Integrate[(xx/hh) (1 + gg zz) Exp[I kk zz], {xx, -hh, hh}]},
  Print["  [1] lateral first moment of (1 + g z) e^{ikz}: ", mom, "  ", chk[Simplify[mom] === 0]]];

(* [2] the 3-D linearisation (monomials of degree <= 2) of a z-only linear contrast times the mean and the
   z-moment: only monomials in xi_z appear. Axis 0 is z. *)
Module[{test = {1, Symbol["x0"]}, contrast = 1 + gg Symbol["x0"], srcs, lateral},
  srcs = Expand[contrast #] & /@ test;
  lateral = Select[Flatten[CoefficientRules[#, {Symbol["x0"], Symbol["x1"], Symbol["x2"]}][[All, 1]] & /@ srcs, 1],
    #[[2]] != 0 || #[[3]] != 0 &];
  Print["  [2] source monomials of the mean and z-moment: ", srcs, "; lateral exponents: ", lateral, "  ",
    chk[lateral === {}]];
  Print["      lateral form factor of a laterally constant weight on the lattice: sinc(p pi) = ",
    Simplify[Sin[pp Pi]/(pp Pi), pp \[Element] Integers && pp != 0], "  ",
    chk[Simplify[Sin[pp Pi]/(pp Pi), pp \[Element] Integers && pp != 0] === 0]]];

(* [3] monomial products {1, xi} x {1, xi} re-expanded in Legendre P0..P2, against notebook 15's A_cbe *)
Module[{A15, Amono, toLeg},
  A15 = Table[(2 e - 1)/2 Integrate[LegendreP[c - 1, xi] LegendreP[b - 1, xi] LegendreP[e - 1, xi], {xi, -1, 1}],
    {c, 2}, {b, 2}, {e, 3}];
  toLeg[poly_] := Table[(2 e - 1)/2 Integrate[poly LegendreP[e - 1, xi], {xi, -1, 1}], {e, 3}];
  Amono = Table[toLeg[xi^(c - 1) xi^(b - 1)], {c, 2}, {b, 2}];
  Print["  [3] linearisation, 3-D monomials restricted to z vs notebook 15: ", Amono === A15, "  ",
    chk[Amono === A15]]];

Print["  gates passed: ", Count[oks, True], "/", Length[oks]];
Exit[If[And @@ oks, 0, 1]];
