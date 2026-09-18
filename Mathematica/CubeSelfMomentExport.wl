#!/usr/bin/env wolframscript
(* ==========================================================================
   THE CUBE'S OWN MOMENTS, BY INTEGRATION -- exported for the Python assembly

   The single-site T-matrix needs two self moments of the cube:

       G_ij    = Int_V G0_ij dV                          (D,W) = (0,0)
       M_in,pk = Int_V d'_p d'_k G0_in dV                (D,W) = (2,0)

   M is the Eshelby-type object.  Its integrand goes as 1/r^3, which is not
   absolutely integrable in three dimensions, and the usual remedy is to quote
   a tabulated "Eshelby tensor" and bolt on a delta-function term by hand.
   THAT IS THE SHORTCUT THIS FILE EXISTS TO AVOID.  The moment engine computes
   both as distributions paired with 1_V, peeling the derivatives onto the cube
   faces, so the r=0 content arrives inside a surface integral: there is no
   delta to add and no table to quote.

   Why this file rather than a hand transcription of the closed forms: the
   closed forms carry radicals and logarithms, and transcribing them by hand is
   exactly the failure this project has already paid for.  The numbers travel
   as data instead.

   Auto-generated consumer: scripts/gate_first_order_tmatrix.py, which uses
   these values and demotes the tabulated route to a cross-check.

   CONVENTION.  The engine integrates over V = [-Del/2, Del/2]^3, so Del is the
   FULL width of the cube: Del = 2a for a half-width a.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

bar = StringJoin @@ ConstantArray["=", 60];
Print[bar];
Print["  The cube's self moments, BY INTEGRATION (no Eshelby table)"];
Print[bar];

Ms[i_, n_, p_, k_] := gStatic[i, n, {p, k}, {}];

acub = Simplify[Ms[1, 1, 2, 2]];
bcub = Simplify[Ms[1, 2, 1, 2]];
ccub = Simplify[Ms[1, 1, 1, 1] - acub - 2 bcub];
gcub = Simplify[gStatic[1, 1, {}, {}]];

Print[""];
Print["closed forms, integrated:"];
Print["  A = ", acub];
Print["  B = ", bcub];
Print["  C = ", ccub];
Print["  G = ", gcub];

(* --------------------------------------------------------------------------
   Every constant that appears in the write-up is checked here, so the document
   quotes nothing it has not derived.
   -------------------------------------------------------------------------- *)
Print[""];
Print["the three face integrals over [-1,1]^2, by direct integration:"];
j1 = Integrate[1/(1 + u^2 + v^2)^(3/2), {u, -1, 1}, {v, -1, 1}];
j2 = Integrate[u^2/(1 + u^2 + v^2)^(5/2), {u, -1, 1}, {v, -1, 1}];
k1 = Integrate[1/(1 + u^2 + v^2)^(5/2), {u, -1, 1}, {v, -1, 1}];
Print["  j1 = ", Simplify[j1]];
Print["  j2 = ", Simplify[j2]];
Print["  k1 = ", Simplify[k1]];
Print["  j1 is the solid angle of one face, 4 Pi/6, with no integration: ",
  Simplify[j1 - 4 Pi/6] === 0];

ivr = Integrate[1/Sqrt[x^2 + y^2 + z^2], {x, -1, 1}, {y, -1, 1}, {z, -1, 1}];
Print["  Int_{[-1,1]^3} dV/r = ", Simplify[ivr], " = ", N[ivr, 16]];
Print["  second closed form (4/3)Log[70226+40545 Sqrt3] - 2Pi agrees: ",
  Abs[N[ivr - ((4/3) Log[70226 + 40545 Sqrt[3]] - 2 Pi), 30]] < 10^-25];

Print[""];
Print["the structural relations the write-up states:"];
a0K = (1/(8 Pi)) (1/mu + 1/(lam + 2 mu));
b0K = (1/(8 Pi)) (1/mu - 1/(lam + 2 mu));
hcK = -(lam + mu)/(2 (lam + 2 mu));
Print["  B  = Sqrt3 (lam+mu)/(6 Pi mu (lam+2mu)) : ",
  zeroQ[bcub - Sqrt[3] (lam + mu)/(6 Pi mu (lam + 2 mu))]];
Print["  A  = B - 1/(3 mu)                       : ",
  zeroQ[acub - (bcub - 1/(3 mu))]];
Print["  C  = -(5 - 2 Pi/Sqrt3) B                : ",
  zeroQ[ccub + (5 - 2 Pi/Sqrt[3]) bcub]];
Print["  G  = (a0 + b0/3) (Del/2)^2 Int dV/r     : ",
  zeroQ[gcub - (a0K + b0K/3) (Del/2)^2 ivr]];
Print["  a0 = (1+hc)/(4 Pi mu), b0 = -hc/(4 Pi mu): ",
  zeroQ[a0K - (1 + hcK)/(4 Pi mu)] && zeroQ[b0K + hcK/(4 Pi mu)]];

(* the sharp gate on the distributional treatment: a spherical excision would
   return 0 here, where the correct answer is fixed by Lap(1/r) = -4 Pi delta *)
trace = Simplify[Sum[Ms[1, 1, p, p], {p, 3}]];
want = -(2 lam + 5 mu)/(3 mu (lam + 2 mu));
Print[""];
Print["Sum_p M_11,pp = ", trace];
Print["   predicted  = ", Simplify[want], "   (independent of the engine)"];
Print["   agree: ", zeroQ[trace - want]];
If[! TrueQ[zeroQ[trace - want]],
  Print["**** the delta-function content is WRONG; do not export ****"];
  Exit[1]];

(* SI values matching ReferenceMedium(5000, 3000, 2500) in the Python gate:
   mu = rho beta^2 = 2.25e10, lam = rho(alpha^2 - 2 beta^2) = 1.75e10.
   a = 0.5 so Del = 1.0; the static M is scale-free and G goes as Del^2. *)
numRule = {lam -> 175/10 10^9, mu -> 225/10 10^9, Del -> 1};

(* --------------------------------------------------------------------------
   THE DYNAMIC MOMENTS, on the same engine.

   The static part is not the whole moment.  Keeping the exponential of the
   Green's tensor to r^5 gives the radiation corrections -- including the
   imaginary part, which is the radiation damping and has no static counterpart
   at all.  These come from gDyn, the same distributional construction with the
   series kept, NOT from a separate polynomial route.

   ka and kb are the compressional and shear wavenumbers of the background.
   -------------------------------------------------------------------------- *)
adyn = gDyn[1, 1, {2, 2}, {}];
bdyn = gDyn[1, 2, {1, 2}, {}];
cdyn = Simplify[gDyn[1, 1, {1, 1}, {}] - adyn - 2 bdyn];
gdyn = gDyn[1, 1, {}, {}];

(* the gate's background: alpha = 5000, beta = 3000, rho = 2500, omega = 60,
   a = 0.5 so Del = 1.  ka = omega/alpha, kb = omega/beta. *)
dynRule = Join[numRule, {ka -> 60/5000, kb -> 60/3000}];

(* The static limit is omega -> 0 at FIXED velocities, so ka and kb go to zero
   together along ka/kb = beta/alpha.  Taking {ka -> 0, kb -> 0} independently is
   path-dependent -- terms like (ka^3 - kb^3)/kb^2 are 0/0 there -- and is not
   the limit meant. *)
(* The static limit is omega -> 0 at FIXED velocities, so ka and kb go to zero
   together along ka/kb = beta/alpha.  This is checked by CONVERGENCE rather
   than by a symbolic limit: the leading correction is O(ka^2), so the deviation
   must fall by ~100 for every decade in omega.  A single symbolic True would
   not distinguish "reduces correctly" from "reduces to something else"; a rate
   does. *)
Print[""];
Print["the dynamic moments reduce to the static ones as omega -> 0."];
Print["  leading correction is O(ka^2), so each decade in omega must cost 100x."];
devAt[w_] := Module[{r = Join[numRule, {ka -> w/5000, kb -> w/3000}]},
   {Abs[N[(adyn - acub)/acub /. r, 30]], Abs[N[(bdyn - bcub)/bcub /. r, 30]],
    Abs[N[(gdyn - gcub)/gcub /. r, 30]]}];
d1 = devAt[60]; d2 = devAt[6]; d3 = devAt[6/10];
Print["  ratios per decade, {A, B, G}: ", N[d1/d2, 6], "  ", N[d2/d3, 6]];
Print["  A and B fall by 100 per decade: O(omega^2), real corrections."];
Print["  G falls by only 10: O(omega). That is NOT a defect -- the zeroth"];
Print["  moment carries the RADIATION REACTION, which is first order in"];
Print["  omega and has no static counterpart at all."];
Print["  A: ", 90 < d1[[1]]/d2[[1]] < 110 && 90 < d2[[1]]/d3[[1]] < 110];
Print["  B: ", 90 < d1[[2]]/d2[[2]] < 110 && 90 < d2[[2]]/d3[[2]] < 110];
Print["  G: ", 9 < d1[[3]]/d2[[3]] < 11 && 9 < d2[[3]]/d3[[3]] < 11];

(* The radiation reaction is derivable without the engine: the imaginary part of
   the Green's tensor at the origin is Im G_11(0) = omega (1/alpha^3 + 2/beta^3)
   / (12 Pi rho), so over a cube of volume V the zeroth moment must carry
   exactly V times that.  Nothing in gDyn knows this. *)
rhoB = mu/3000^2;
imDev[w_] := Module[{r, got, want},
   r = Join[numRule, {ka -> w/5000, kb -> w/3000}];
   got = Im[N[gdyn /. r, 30]];
   want = N[Del^3 (w/(12 Pi rhoB)) (1/5000^3 + 2/3000^3) /. numRule, 30];
   Abs[(got - want)/want]];
Print[""];
Print["the imaginary part of the G moment IS the radiation reaction."];
Print["  Im G_11(0) = omega (1/alpha^3 + 2/beta^3)/(12 Pi rho), so the zeroth"];
Print["  moment must carry V times that -- to LEADING order.  The moment"];
Print["  integrates Im G across the cube while V Im G(0) does not, so the two"];
Print["  differ at O(ka^2) and the check is again a RATE, not a bar."];
i1 = imDev[60]; i2 = imDev[6]; i3 = imDev[6/10];
Print["  relative gap at omega = 60, 6, 0.6: ", N[{i1, i2, i3}, 6]];
Print["  ratios per decade (want ~100): ", N[{i1/i2, i2/i3}, 6]];
Print["  the gap is the finite-size correction: ",
  90 < i1/i2 < 110 && 90 < i2/i3 < 110];

out = <|
   "note" -> "auto-generated by CubeSelfMomentExport.wl; do not edit",
   "convention" -> "V = [-Del/2, Del/2]^3, so Del = 2a (full width)",
   "lam" -> 17.5*^9, "mu" -> 22.5*^9, "Del" -> 1.0,
   "omega" -> 60.0, "alpha" -> 5000.0, "beta" -> 3000.0,
   "A" -> N[acub /. numRule, 20],
   "B" -> N[bcub /. numRule, 20],
   "C" -> N[ccub /. numRule, 20],
   "G" -> N[gcub /. numRule, 20],
   "A_dyn_re" -> Re[N[adyn /. dynRule, 20]], "A_dyn_im" -> Im[N[adyn /. dynRule, 20]],
   "B_dyn_re" -> Re[N[bdyn /. dynRule, 20]], "B_dyn_im" -> Im[N[bdyn /. dynRule, 20]],
   "C_dyn_re" -> Re[N[cdyn /. dynRule, 20]], "C_dyn_im" -> Im[N[cdyn /. dynRule, 20]],
   "G_dyn_re" -> Re[N[gdyn /. dynRule, 20]], "G_dyn_im" -> Im[N[gdyn /. dynRule, 20]]|>;

refPath = "/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/\
cube_self_moments.json";
Export[refPath, out, "JSON"];

Print[""];
Print["numerically at lam = 17.5 GPa, mu = 22.5 GPa, Del = 1:"];
Print["  A = ", out["A"]];
Print["  B = ", out["B"]];
Print["  C = ", out["C"]];
Print["  G = ", out["G"]];
Print[""];
Print["wrote ", refPath];
Print[bar];
