#!/usr/bin/env wolframscript
(* ==========================================================================
   THE M MOMENT OF A CUBE

       M_in,pk = Int_V d'_p d'_k (G0_in(0,r')) dr'          (D,W) = (2,0)

   This is the Eshelby-type object: the second-derivative moment that converts
   an eigenstrain into an internal field.  Where it appears in the closed set
   (A33.nb, "The Basic System"):

       A22 = (delta_pr delta_ij + w^2 drho N^r_ij,p - M_in,pk dc_nkrj)
       A31 = -w^2 drho M_ij,pq

   THE POINT OF DIFFICULTY.  The integrand goes as 1/r^3, which is NOT
   absolutely integrable in three dimensions.  The usual remedy is to bolt a
   delta-function term on "by Eshelby"; forget it and the depolarisation comes
   out with the wrong sign.  The engine needs no such patch -- the moment is
   computed as a distribution paired with 1_V, so the r=0 content arrives
   inside the surface integral.  See CubeMomentCore.wl.

   SCALE.  M ~ Del^0, independent of cube size.

   CROSS-CHECK.  CubeT9FromFirstPrinciples.wl computes the same object under
   the name I_{ijkl} = Int d_i d_j G_kl dV, so M_in,pk = I_{pk,in}, by an
   independent route (a hand-written face integral of the Kupradze form rather
   than the general engine).  Its three components were themselves validated
   against the Python master-integral route to 10 digits.  Both are checked.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE M MOMENT OF A CUBE     M_in,pk = Int_V d_p d_k G0_in dV"];
Print["=============================================================="];

Ms[i_, n_, p_, k_] := gStatic[i, n, {p, k}, {}];

Print[];
Print["[1] the general cubic form, rank 4 (indices i,n,p,k)"];
form = cubicTensorForm[Ms, 4];
Do[Print["    ", partLabel[form[[1, u]], {"i", "n", "p", "k"}],
         "\n        = ", form[[2, u]]], {u, Length[form[[1]]]}];

Print[];
Print["[2] verification on all 81 components"];
bad = verifyForm[Ms, form, 4, Tuples[{1, 2, 3}, 4]];
Print["    mismatches: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[Take[bad, UpTo[5]]]]];

Print[];
Print["[3] the conventional A,B,C decomposition"];
Print["    M_in,pk = A d_in d_pk + B (d_ip d_nk + d_ik d_np) + C E_inpk"];
Acub = Simplify[Ms[1, 1, 2, 2]];
Bcub = Simplify[Ms[1, 2, 1, 2]];
Ccub = Simplify[Ms[1, 1, 1, 1] - Acub - 2 Bcub];
Print["    A = ", Acub];
Print["    B = ", Bcub];
Print["    C = ", Ccub];

Print[];
Print["[4] symmetry of the moment (must hold identically)"];
Print["    M_in,pk == M_ni,pk : ",
      zeroQ[Ms[1, 2, 1, 3] - Ms[2, 1, 1, 3]]];
Print["    M_in,pk == M_in,kp : ",
      zeroQ[Ms[1, 2, 1, 3] - Ms[1, 2, 3, 1]]];

Print[];
Print["[5] CROSS-CHECK against CubeT9FromFirstPrinciples.wl"];
num = {lam -> 175/10 10^9, mu -> 225/10 10^9};
Print["    A (this engine) = ", N[Acub /. num, 10],
      "     reference = -1.2201107459*10^-11"];
Print["    B (this engine) = ", N[Bcub /. num, 10],
      "     reference =  2.6137073561*10^-12"];
Print["    C (this engine) = ", N[Ccub /. num, 10],
      "     reference = -3.5870552989*10^-12"];

Print[];
Print["[6] CROSS-CHECK: the closed-form shear self-term of"];
Print["    CubeT9FromFirstPrinciples.wl,"];
Print["        S_shear = (Pi(lam+2mu) - Sqrt[3](lam+mu)) / (3 Pi mu (lam+2mu))"];
Print["    The B component carries the whole deviatoric part: expanding,"];
Print["    Sqrt[3](lam+mu)/(3 Pi mu (lam+2mu)) is exactly 2B, so the identity"];
Print["    connecting the two is  S_shear == 1/(3mu) - 2B."];
Sshear = (Pi (lam + 2 mu) - Sqrt[3] (lam + mu))/(3 Pi mu (lam + 2 mu));
Print["    S_shear reference = ", Simplify[Sshear]];
Print["    1/(3mu) - 2B      = ", Simplify[1/(3 mu) - 2 Bcub]];
Print["    agree: ", zeroQ[Sshear - (1/(3 mu) - 2 Bcub)]];

Print[];
Print["[7] trace identity, derived independently of the engine."];
Print["    G0_in = (1/4Pi mu)[d_in/r + hc d_i d_n r], and Lap(1/r) = -4Pi d^3,"];
Print["    Lap(d_i d_n r) = 2 d_i d_n (1/r), whose cube integral is -4Pi/3 d_in."];
Print["    So  Sum_p M_in,pp = -d_in (1 + 2 hc/3)/mu = -d_in (2lam+5mu)/"];
Print["    (3 mu (lam+2mu)), with no reference to the moment engine at all."];
want7 = -(2 lam + 5 mu)/(3 mu (lam + 2 mu));
got7 = Simplify[Sum[Ms[1, 1, p, p], {p, 3}]];
Print["    Sum_p M_11,pp = ", got7];
Print["    predicted     = ", Simplify[want7]];
Print["    agree: ", zeroQ[got7 - want7]];
Print["    Sum_p M_12,pp = ", Simplify[Sum[Ms[1, 2, p, p], {p, 3}]],
      "   (must be 0)"];

Print[];
Print["[8] the dynamic moment, series through r^5"];
Print["    M_11,22 = ", Simplify[gDyn[1, 1, {2, 2}, {}], Assumptions -> Del > 0]];

Print["=============================================================="];
