#!/usr/bin/env wolframscript
(* ==========================================================================
   THE N MOMENT OF A CUBE

       N^r_in,k = Int_V d'_k (G0_in(0,r')) x'_r dr'         (D,W) = (1,1)

   One derivative, one weight.  Where it appears (A33.nb, "The Basic System"):

       A13 = (N^r_in,k dc_nksj - (1/2) w^2 drho K^rs_ij)
       A22 = (delta_pr delta_ij + w^2 drho N^r_ij,p - M_in,pk dc_nkrj)

   WHY THIS ONE IS DIFFERENT.  N is the only moment of the six whose two
   gradings are BOTH odd: D = 1 and W = 1.  Every other moment has D and W
   both even.  That is the parity rule which makes the closed set close --
   the A22 block (first gradients) couples only to itself, so second gradients
   cannot feed back into it.  Everything odd lives here.

   The consequence, established elsewhere in this project, is that second
   gradients cannot correct the shear channel, which is why the T9 -> T27 ->
   T57 sequence oscillates (1.000 -> 0.749 -> 0.958) instead of converging.
   That behaviour is a property of the grading, not of the discretisation.

   SCALE.  N ~ Del^2.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE N MOMENT OF A CUBE     N^r_in,k = Int_V d_k G0_in x_r dV"];
Print["=============================================================="];

Ns[i_, n_, k_, r_] := gStatic[i, n, {k}, {r}];

Print[];
Print["[1] the general cubic form, rank 4 (indices i,n,k,r)"];
form = cubicTensorForm[Ns, 4];
Do[Print["    ", partLabel[form[[1, u]], {"i", "n", "k", "r"}],
         "\n        = ", form[[2, u]]], {u, Length[form[[1]]]}];

Print[];
Print["[2] verification on all 81 components"];
bad = verifyForm[Ns, form, 4, Tuples[{1, 2, 3}, 4]];
Print["    mismatches: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[Take[bad, UpTo[5]]]]];

Print[];
Print["[3] the conventional A,B,C decomposition"];
Print["    N^r_in,k = A d_in d_kr + B (d_ik d_nr + d_ir d_nk) + C E_inkr"];
Acub = Simplify[Ns[1, 1, 2, 2]];
Bcub = Simplify[Ns[1, 2, 1, 2]];
Ccub = Simplify[Ns[1, 1, 1, 1] - Acub - 2 Bcub];
Print["    A = ", Acub];
Print["    B = ", Bcub];
Print["    C = ", Ccub];
Print["    scale check, all ~ Del^2: ",
      {Simplify[Acub/Del^2], Simplify[Bcub/Del^2], Simplify[Ccub/Del^2]}];

Print[];
Print["[4] CROSS-CHECK by integration by parts.  Since d_k(G x_r) ="];
Print["    (d_k G) x_r + G delta_kr, and Int_V d_k(G x_r) dV is a pure"];
Print["    surface term, N must satisfy"];
Print["        N^r_in,k + delta_kr G_in = Sur_S n_k G0_in x_r dA ."];
Print["    Both sides are computed independently below."];
surfN[i_, n_, k_, r_] := Module[{F, o1, o2, uu, vv},
   F = ((1/(4 Pi mu)) (d[i, n] rr^-1 + hc D[rr, X[[i]], X[[n]]])) X[[r]];
   {o1, o2} = Complement[{1, 2, 3}, {k}];
   Integrate[
     (F /. {X[[k]] ->  hw, X[[o1]] -> uu, X[[o2]] -> vv}) -
     (F /. {X[[k]] -> -hw, X[[o1]] -> uu, X[[o2]] -> vv}),
     {uu, -hw, hw}, {vv, -hw, hw}, Assumptions -> Del > 0]];
Do[With[{ii = q[[1]], nn = q[[2]], kk = q[[3]], rr2 = q[[4]]},
   lhs = Simplify[Ns[ii, nn, kk, rr2] + d[kk, rr2] gStatic[ii, nn, {}, {}],
                  Assumptions -> Del > 0];
   rhs = Simplify[surfN[ii, nn, kk, rr2], Assumptions -> Del > 0];
   Print["    (i,n,k,r)=", q, "   agree: ",
         zeroQ[lhs - rhs]]],
 {q, {{1, 1, 1, 1}, {1, 1, 2, 2}, {1, 2, 1, 2}, {1, 2, 2, 3}}}];

Print[];
Print["[5] parity.  N is the odd moment: components with an odd index"];
Print["    multiplicity must vanish identically."];
Do[Print["    N", q, " = ", Simplify[Ns @@ q]],
 {q, {{1, 1, 1, 2}, {1, 2, 3, 3}, {1, 2, 1, 3}}}];

Print[];
Print["[6] the dynamic moment, series through r^5"];
Print["    N^1_11,1 = ", Simplify[gDyn[1, 1, {1}, {1}], Assumptions -> Del > 0]];

Print["=============================================================="];
