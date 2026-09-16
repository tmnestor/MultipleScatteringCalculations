#!/usr/bin/env wolframscript
(* ==========================================================================
   THE G MOMENT OF A CUBE

       G_ij = Int_V G0_ij(0,r') dr'                       (D,W) = (0,0)

   The zeroth moment of the Green's tensor over the cube, about its centre.
   It is the one moment with no derivative and no weight, so it carries no
   delta-function subtlety at all: the integrand is O(1/r), absolutely
   integrable in three dimensions.  Everything harder is in its siblings.

   Where it appears (A33.nb, "The Basic System"):

       A11 = (delta_ij - w^2 drho G_ij)

   i.e. G is the entire density response of the leading (uniform-displacement)
   block.  Being a rank-2 cubic invariant it must be a multiple of delta_ij --
   the cube cannot distinguish its own axes at this order, so THERE IS NO
   ANISOTROPY IN A11.  That is checked here rather than assumed.

   SCALE.  G ~ Del^2.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE G MOMENT OF A CUBE     G_ij = Int_V G0_ij dV"];
Print["=============================================================="];

Gs[i_, j_] := gStatic[i, j, {}, {}];

Print[];
Print["[1] the general cubic form (rank 2: the only invariant is delta_ij)"];
form = cubicTensorForm[Gs, 2];
Do[Print["    ", partLabel[form[[1, u]], {"i", "j"}], "   coefficient  ",
         form[[2, u]]], {u, Length[form[[1]]]}];

Print[];
Print["[2] verification on every component"];
bad = verifyForm[Gs, form, 2, Tuples[{1, 2, 3}, 2]];
Print["    mismatches over all 9 components: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[bad]]];

Print[];
Print["[3] the closed form"];
gg = Simplify[Gs[1, 1], Assumptions -> Del > 0];
Print["    G_ij = g delta_ij ,   g = ", gg];
Print["    numerically at Del=1, lam=mu=1:  ",
      N[gg /. {Del -> 1, lam -> 1, mu -> 1}, 10]];
Print["    off-diagonal G_12 = ", Simplify[Gs[1, 2]], "   (must be 0)"];

Print[];
Print["[4] cross-check: G_ij is a0/b0 weighted against the cube's 1/r moment"];
Print["    G_ii summed = (2 a0 + b0) Int_V dV/r, since x_i x_i/r^3 = 1/r."];
a0 = (lam + 3 mu)/(8 Pi mu (lam + 2 mu));
b0 = (lam + mu)/(8 Pi mu (lam + 2 mu));
lhs = Simplify[Sum[Gs[i, i], {i, 3}], Assumptions -> Del > 0];
rhs = Simplify[(3 a0 + b0) E$[-1, {}, {}], Assumptions -> Del > 0];
Print["    trace  = ", lhs];
Print["    (3a0+b0) Int dV/r = ", rhs];
Print["    agree: ", zeroQ[lhs - rhs]];

Print[];
Print["[5] the dynamic moment, series through r^5"];
Print["    G_11 = ", Simplify[gDyn[1, 1, {}, {}], Assumptions -> Del > 0]];

Print["=============================================================="];
