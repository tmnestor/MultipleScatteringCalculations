#!/usr/bin/env wolframscript
(* ==========================================================================
   GATE for CubeMomentCore.wl.

   Every check below has an answer that is known WITHOUT computing the moment,
   so a passing engine is being compared against something independent rather
   than against itself.  The first four pin the delta-function bookkeeping,
   which is the one thing this project has repeatedly got wrong: a spherical
   excision returns 0 for the Laplacian sum rule where the right answer is
   -4 Pi, and no symmetry or reciprocity test can see the difference.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

nfail = 0;

(* zeroQ escalates Simplify -> FullSimplify -> 120-digit numerics and records
   which tier settled it; see the note at the top of CubeMomentCore.wl for why
   Simplify alone is not a usable predicate on these expressions. *)
chk[lbl_, lhs_, rhs_] := Module[{ok},
   ok = zeroQ[lhs - rhs];
   If[! ok, nfail++];
   Print["  ", If[ok, "PASS", "FAIL"], "  [", $zeroQreason, "]  ", lbl];
   If[! ok, Print["        got      ", lhs];
            Print["        expected ", rhs];
            Print["        residual ", Simplify[lhs - rhs,
                                                Assumptions -> Del > 0]];
            Print["        numeric  ", N[(lhs - rhs) /. {Del -> 1,
                                          lam -> 3/2, mu -> 1}, 25]]]];

Print["=============================================================="];
Print["GATE: THE CUBE MOMENT ENGINE"];
Print["=============================================================="];

Print[];
Print["[1] the D=0 anchor:  Int_V dV/r  over the unit cube, against an"];
Print["    INDEPENDENT numerical quadrature of the same integral.  This is"];
Print["    the one moment with no derivative, so quadrature is legitimate"];
Print["    here (1/r is absolutely integrable in 3-D) -- which is exactly why"];
Print["    it makes a clean anchor and why the same trick fails for the rest."];
Print["    Cartesian quadrature cannot do it -- the singularity is INSIDE the"];
Print["    box and adaptive bisection chases it forever.  Doing the radial"];
Print["    direction analytically removes it: Int_0^R (1/r) r^2 dr = R^2/2,"];
Print["    so  Int_V dV/r = Int_4Pi R(Omega)^2/2 dOmega  with R the distance"];
Print["    to the cube face, R = (Del/2)/max(|n1|,|n2|,|n3|).  That integrand"];
Print["    is bounded and piecewise smooth, and shares nothing with the"];
Print["    engine's face-peeling route."];
i0 = E$[-1, {}, {}];
Print["    closed form  ", i0];
rad[th_, ph_] := (1/2)/Max[Abs[Sin[th] Cos[ph]], Abs[Sin[th] Sin[ph]],
                           Abs[Cos[th]]];
quad = NIntegrate[rad[th, ph]^2/2 Sin[th], {th, 0, Pi}, {ph, 0, 2 Pi},
   MaxRecursion -> 20, PrecisionGoal -> 8, WorkingPrecision -> 20];
Print["    closed form at Del=1  ", N[i0 /. Del -> 1, 14]];
Print["    solid-angle quadrature ", N[quad, 14]];
reldiff = Abs[N[(i0 /. Del -> 1) - quad, 20]]/Abs[quad];
Print["    relative difference   ", N[reldiff, 3]];
If[reldiff < 10^-7, Print["  PASS  closed form matches independent quadrature"],
   nfail++; Print["  FAIL  closed form disagrees with quadrature"]];

Print[];
Print["[2] Laplacian sum rule.  Lap(1/r) = -4 Pi delta, and the origin is"];
Print["    inside V, so Sum_p E[-1;{p,p};] must be exactly -4 Pi."];
Print["    A spherical excision gives 0 here -- that is the whole trap."];
chk["Sum_p E[-1;{p,p};] == -4 Pi", Sum[E$[-1, {p, p}, {}], {p, 3}], -4 Pi];

Print[];
Print["[3] and by cubic symmetry each diagonal term is -4 Pi/3"];
Do[chk["E[-1;{" <> ToString[p] <> "," <> ToString[p] <> "};] == -4 Pi/3",
       E$[-1, {p, p}, {}], -4 Pi/3], {p, 3}];
chk["E[-1;{1,2};] == 0 (parity)", E$[-1, {1, 2}, {}], 0];

Print[];
Print["[4] the delta content, reported by difference against the excised twin"];
Print["    E - excisedE for {1,1} = ",
      Simplify[E$[-1, {1, 1}, {}] - excisedE[-1, {1, 1}, {}]],
      "   (expect -4 Pi/3)"];

Print[];
Print["[5] weighted Laplacian.  <-4 Pi delta, x_r x_s> = 0, since the weight"];
Print["    vanishes at the origin -- a sharp test of the W>0 recursion."];
Do[chk["Sum_p E[-1;{p,p};" <> ToString[pr] <> "] == 0",
       Sum[E$[-1, {p, p}, pr], {p, 3}], 0], {pr, {{1, 1}, {1, 2}, {2, 3}}}];

Print[];
Print["[6] third and fourth derivatives.  <-4 Pi d_p d_q delta, 1_V> = 0"];
Print["    because 1_V is flat at the origin.  This is the D=4 case that the"];
Print["    Q and P moments actually need."];
Do[chk["Sum_k E[-1;{" <> ToString[pr] <> ",k,k};] == 0",
       Sum[E$[-1, Join[pr, {k, k}], {}], {k, 3}], 0], {pr, {{1, 1}, {1, 2}}}];

Print[];
Print["[7] Lap r = 2/r  =>  Sum_p E[1;{i,n,p,p};] == 2 E[-1;{i,n};]"];
Do[chk["biharmonic sum rule at (i,n)=" <> ToString[pr],
       Sum[E$[1, Join[pr, {p, p}], {}], {p, 3}],
       2 E$[-1, pr, {}]], {pr, {{1, 1}, {1, 2}}}];

Print[];
Print["[8] Lap r^3 = 12 r  =>  Sum_p E[3;{i,n,p,p};] == 12 E[1;{i,n};]"];
Do[chk["r^3 sum rule at (i,n)=" <> ToString[pr],
       Sum[E$[3, Join[pr, {p, p}], {}], {p, 3}],
       12 E$[1, pr, {}]], {pr, {{1, 1}, {1, 2}}}];

Print[];
Print["[9] scaling.  E[m;D;W] is homogeneous of degree m - D + W + 3 in Del,"];
Print["    so the six moments scale as  G,N,P ~ Del^2   K ~ Del^4   M,Q ~ Del^0."];
deg[e_] := Simplify[D[e, Del] Del/e];
Print["    M-type  E[-1;{1,1};]  free of Del: ", FreeQ[E$[-1, {1, 1}, {}], Del]];
Print["    G-type  E[-1;;]             degree in Del = ", deg[E$[-1, {}, {}]]];
Print["    K-type  E[-1;;{1,1}]        degree in Del = ", deg[E$[-1, {}, {1, 1}]]];

Print[];
Print["[10] the tensor assembly reproduces Kupradze in the static limit."];
Print["     G0_in = (1/4Pi mu)[d_in g_b + (1/kb^2) d_i d_n (g_b - g_a)] must"];
Print["     give a0 = (lam+3mu)/(8 Pi mu (lam+2mu)), b0 = (lam+mu)/(same)."];
hst = Simplify[((ka^2 - kb^2)/(2 kb^2)) /. lameRule];
chk["hc == (ka^2-kb^2)/(2 kb^2) under the Lame substitution", hst, hc];
gst = (1/(4 Pi mu)) (d[1, 1]/rr + hst D[rr, x, x]);
a0t = (lam + 3 mu)/(8 Pi mu (lam + 2 mu));
b0t = (lam + mu)/(8 Pi mu (lam + 2 mu));
chk["G0_11 static == a0/r + b0 x^2/r^3",
    Simplify[gst], Simplify[a0t/rr + b0t x^2/rr^3]];
gst12 = (1/(4 Pi mu)) (d[1, 2]/rr + hst D[rr, x, y]);
chk["G0_12 static == b0 x y/r^3", Simplify[gst12], Simplify[b0t x y/rr^3]];

Print[];
Print["=============================================================="];
Print[If[nfail == 0, "PASS -- the engine is sound.",
         "FAIL -- " <> ToString[nfail] <> " check(s) failed."]];
Print["=============================================================="];
