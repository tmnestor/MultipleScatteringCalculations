#!/usr/bin/env wolframscript
(* ==========================================================================
   THE K MOMENT OF A CUBE

       K^rs_ij = Int_V G0_ij(0,r') x'_r x'_s dr'            (D,W) = (0,2)

   The second x-moment of the Green's tensor.  Where it appears (A33.nb,
   "The Basic System"):

       A13 = (N^r_in,k dc_nksj - (1/2) w^2 drho K^rs_ij)

   so K enters only through the density contrast, paired with the 1/2 from the
   Taylor expansion of the internal field.  Like G, it carries no derivative,
   so its integrand is O(1/r) and absolutely integrable -- no delta-function
   content whatsoever.  It is the most benign of the six.

   SCALE.  K ~ Del^4.  It is the only moment of the six that does, which makes
   the exponent a useful check on the whole grading: the general rule is
   E ~ Del^(m - D + W + 3), here (-1) - 0 + 2 + 3 = 4.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE K MOMENT OF A CUBE     K^rs_ij = Int_V G0_ij x_r x_s dV"];
Print["=============================================================="];

Ks[i_, j_, r_, s_] := gStatic[i, j, {}, {r, s}];

Print[];
Print["[1] the general cubic form, rank 4 (indices i,j,r,s)"];
form = cubicTensorForm[Ks, 4];
Do[Print["    ", partLabel[form[[1, u]], {"i", "j", "r", "s"}],
         "\n        = ", form[[2, u]]], {u, Length[form[[1]]]}];

Print[];
Print["[2] verification on all 81 components"];
bad = verifyForm[Ks, form, 4, Tuples[{1, 2, 3}, 4]];
Print["    mismatches: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[Take[bad, UpTo[5]]]]];

Print[];
Print["[3] the conventional A,B,C decomposition"];
Print["    K^rs_ij = A d_ij d_rs + B (d_ir d_js + d_is d_jr) + C E_ijrs"];
Acub = Simplify[Ks[1, 1, 2, 2]];
Bcub = Simplify[Ks[1, 2, 1, 2]];
Ccub = Simplify[Ks[1, 1, 1, 1] - Acub - 2 Bcub];
Print["    A = ", Acub];
Print["    B = ", Bcub];
Print["    C = ", Ccub];

Print[];
Print["[4] SCALE CHECK.  Every component must be exactly quartic in Del."];
Do[Print["    ", nm[[1]], "/Del^4 = ", Simplify[nm[[2]]/Del^4],
         "   (free of Del: ", FreeQ[Simplify[nm[[2]]/Del^4], Del], ")"],
 {nm, {{"A", Acub}, {"B", Bcub}, {"C", Ccub}}}];

Print[];
Print["[5] CROSS-CHECK against the scalar cube moments.  With"];
Print["    G0_ij = a0 d_ij/r + b0 x_i x_j/r^3, K is a pure combination of"];
Print["    E[-1;;{r,s}] and E[-3;{i,j} weighted] -- checked here by"];
Print["    contracting i=j and r=s, where the x_i x_j/r^3 term collapses:"];
Print["        Sum_ij d_ij K^rs_ij = (3 a0 + b0) E[-1;;{r,s}]  is NOT the"];
Print["    identity; the correct contraction is over i=j with x_i x_i/r^3 ="];
Print["    1/r, giving (3 a0 + b0) Int_V x_r x_s dV/r."];
a0 = (lam + 3 mu)/(8 Pi mu (lam + 2 mu));
b0 = (lam + mu)/(8 Pi mu (lam + 2 mu));
Do[With[{rr2 = pr[[1]], ss = pr[[2]]},
   lhs = Simplify[Sum[Ks[i, i, rr2, ss], {i, 3}], Assumptions -> Del > 0];
   rhs = Simplify[(3 a0 + b0) E$[-1, {}, {rr2, ss}], Assumptions -> Del > 0];
   Print["    (r,s)=", pr, "   trace = ", lhs];
   Print["                  (3a0+b0) E[-1;;{r,s}] = ", rhs];
   Print["                  agree: ",
         zeroQ[lhs - rhs]]],
 {pr, {{1, 1}, {1, 2}}}];

Print[];
Print["[6] the plain second moment of the cube, for reference"];
Print["    Int_V x_1^2 dV/r = ", Simplify[E$[-1, {}, {1, 1}]]];
Print["    Int_V x_1 x_2 dV/r = ", Simplify[E$[-1, {}, {1, 2}]], "  (parity)"];

Print[];
Print["[7] the dynamic moment, series through r^5"];
Print["    K^11_11 = ",
      Simplify[gDyn[1, 1, {}, {1, 1}], Assumptions -> Del > 0]];

Print["=============================================================="];
