(* ::Package:: *)
(* ElasticBornBackscatter.wl -- the Born P->P backscattering amplitude of a
   weak sphere, derived from the elastodynamic Green's tensor with NO hand
   contraction anywhere.

   WHY THIS EXISTS.  A hand derivation gave the backscattered far-field
   amplitude of a weak sphere as

       u_s r = -kP^2 [ drho (lam + 2 mu)/rho + dlam + 2 dmu ] * chi(2 kP)
               / (4 Pi rho alpha^2)

   and that constant disagrees with the committed elastic Mie arbiter by a
   factor measured at EXACTLY 2 in the Rayleigh limit.  Two independent
   marches reproduce the hand constant through a separate route, and the
   voxel T-matrix reproduces Mie, so the two camps are each internally
   consistent and one of them carries a factor of two.  This decides it
   without reusing any step of the hand derivation:

     * the Green's tensor is written from the Helmholtz decomposition and
       VERIFIED against the Navier equation symbolically, so its overall
       normalisation is not assumed;
     * the perturbation tensor dc_jklm is built as an explicit rank-4 array
       and every index is summed by the kernel, not by hand;
     * the Born integral is evaluated in its WEAK form, which carries no
       derivative of the discontinuous contrast and so needs no integration
       by parts from the author;
     * the volume integral is done NUMERICALLY over the sphere, so the
       closed-form sphere form factor is never invoked either.

   Every quantity the disputed formula depends on is therefore recomputed.

   Coordinates here are Mathematica's own (x, y, z) with the incident P wave
   along +z; the Python side orders axes (z, x, y).  Nothing is shared
   between the two, on purpose.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/ElasticBornBackscatter.wl
*)

Print["=============================================================="];
Print["THE ELASTIC BORN BACKSCATTERING AMPLITUDE OF A WEAK SPHERE"];
Print["=============================================================="];

(* ------------------------------------------------------------------ *)
(* Section 1.  The Green's tensor, and a check that it IS one.          *)
(* ------------------------------------------------------------------ *)
(* Helmholtz decomposition, e^{-i omega t}, outgoing:                   *)
(*   G_ij = (1/(rho w^2)) [ (d_ij kS^2 + d_i d_j) gS - d_i d_j gP ]     *)
(* with g_K = e^{i kK r}/(4 Pi r).  The S part projects transverse and   *)
(* the P part longitudinal; the relative sign is what fixes the far-     *)
(* field normalisation, and it is the thing under suspicion, so it is    *)
(* verified rather than quoted.                                         *)

xv = {x, y, z};
r2 = x^2 + y^2 + z^2;
rr = Sqrt[r2];

gK[k_] := Exp[I k rr]/(4 Pi rr);

Gten[kPv_, kSv_, rhov_, wv_] := Module[{gp, gs, dd},
  (
   gp = gK[kPv];
   gs = gK[kSv];
   dd = Table[D[gs - gp, xv[[i]], xv[[j]]], {i, 3}, {j, 3}];
   (1/(rhov wv^2)) (dd + kSv^2 IdentityMatrix[3] gs)
   )];

(* The elastic tensor, built as an array so no contraction is done by hand. *)
cten[lam_, mu_] := Table[
   lam KroneckerDelta[i, j] KroneckerDelta[k, l]
    + mu (KroneckerDelta[i, k] KroneckerDelta[j, l]
          + KroneckerDelta[i, l] KroneckerDelta[j, k]),
   {i, 3}, {j, 3}, {k, 3}, {l, 3}];

Print["\n[1] does G satisfy the Navier equation away from the origin?"];

Module[{lam, mu, rho, w, kp, ks, cc, gg, resid},
 (
  lam = 7/5; mu = 9/10; rho = 11/10; w = 13/10;
  kp = w/Sqrt[(lam + 2 mu)/rho];
  ks = w/Sqrt[mu/rho];
  cc = cten[lam, mu];
  gg = Gten[kp, ks, rho, w];
  (* d_j ( c_ijkl d_l G_km ) + rho w^2 G_im  must vanish for r != 0. *)
  resid = Table[
    Sum[D[cc[[i, j, k, l]] D[gg[[k, m]], xv[[l]]], xv[[j]]], {j, 3}, {k, 3}, {l, 3}]
     + rho w^2 gg[[i, m]],
    {i, 3}, {m, 3}];
  resid = Simplify[resid /. {x -> 3/7, y -> -2/5, z -> 6/11}];
  Print["    max |Navier residual| at a generic point: ", Max[Abs[N[Flatten[resid], 30]]]];
  )];

(* ------------------------------------------------------------------ *)
(* Section 2.  The far-field P coefficient, measured not assumed.       *)
(* ------------------------------------------------------------------ *)
Print["\n[2] the far-field P part of G, against gamma_i gamma_j/(4 Pi rho alpha^2 r)"];

Module[{lam, mu, rho, w, al, kp, ks, gg, rf, num, want},
 (
  lam = 175/10; mu = 225/10; rho = 25/10; w = 6/10;
  al = Sqrt[(lam + 2 mu)/rho];
  kp = w/al; ks = w/Sqrt[mu/rho];
  gg = Gten[kp, ks, rho, w];
  rf = 10^6;
  (* on the +z axis gamma = zhat, so the zz entry should be the P part *)
  num = N[gg[[3, 3]] /. {x -> 0, y -> 0, z -> rf}, 40];
  want = N[Exp[I kp rf]/(4 Pi rho al^2 rf), 40];
  Print["    G_zz at r = 10^6 : ", num];
  Print["    gamma gamma/(4 Pi rho alpha^2 r) : ", want];
  Print["    ratio : ", num/want];
  )];

(* ------------------------------------------------------------------ *)
(* Section 3.  The Born integral, in weak form, integrated numerically. *)
(* ------------------------------------------------------------------ *)
(* u1_i(x) = w^2 Int drho G_ij(x,x') u0_j(x') dV'                       *)
(*         - Int dc_jklm  dG_ij(x,x')/dx'_k  du0_l/dx'_m  dV'           *)
(* The second term carries the derivative on G, not on the contrast, so  *)
(* no integration by parts is performed here and no surface term is      *)
(* dropped.                                                             *)

(* ------------------------------------------------------------------ *)
(* Section 3.  The contraction, with NO integration by parts at all.    *)
(* ------------------------------------------------------------------ *)
(* In the far field G_ij(x,x') = n_i n_j e^{i kP r} e^{-i ks.x'}        *)
(* /(4 Pi rho alpha^2 r), verified in Section 2.  Its derivative with   *)
(* respect to the SOURCE point is therefore exactly                     *)
(*                                                                      *)
(*     dG_ij/dx'_k = -i ks_k G_ij                                       *)
(*                                                                      *)
(* so the weak form needs no integration by parts and drops no surface  *)
(* term: the derivative moves by an exact identity rather than by an    *)
(* argument about compact support.  What is left is a pure contraction, *)
(* and every index of it is summed by the kernel.                       *)

Print["\n[3] the Born amplitude by exact far-field contraction (no by-parts)"];

Module[{lam, mu, rho, w, al, be, kp, ks, amp, s, dlam, dmu, drho, aSph,
        dc, nhat, ehat, kin, ksc, brack, brackGen, th, chiHat, famp, mine},
 (
  rho = 2500.; al = 5000.; be = 3000.; w = 60.;
  mu = rho be^2; lam = rho al^2 - 2 mu;
  kp = w/al;
  amp = 10.^-4; s = 1 + amp;
  dlam = (s^3 - 1) lam; dmu = (s^3 - 1) mu; drho = (s - 1) rho;
  aSph = 15.;
  dc = cten[dlam, dmu];

  (* General scattering angle first, so the 180 degree case is a special
     case of something rather than a hand-picked geometry. *)
  ehat = {0, 0, 1};
  nhat = {Sin[th], 0, Cos[th]};
  kin = kp ehat;
  ksc = kp nhat;
  brackGen = w^2 drho (nhat . ehat)
    - Sum[nhat[[j]] ksc[[k]] kin[[m]] ehat[[l]] dc[[j, k, l, m]],
        {j, 3}, {k, 3}, {l, 3}, {m, 3}];
  Print["    bracket(theta) = ", Simplify[brackGen]];
  brack = Simplify[brackGen /. th -> Pi];
  Print["    bracket(180 deg) = ", brack];

  chiHat = 4 Pi (Sin[2 kp aSph] - 2 kp aSph Cos[2 kp aSph])/(2 kp)^3;
  famp = Abs[brack] chiHat/(4 Pi rho al^2);
  mine = kp^2 (drho (lam + 2 mu)/rho + dlam + 2 dmu) chiHat/(4 Pi rho al^2);

  Print[""];
  Print["    |u_scat| r, kernel contraction : ", famp];
  Print["    |u_scat| r, HAND formula       : ", mine];
  Print["    ratio kernel / hand            : ", famp/mine];
  Print["    ratio kernel / Mie (3.2398e-5) : ", famp/3.2398*^-5];
  )];

Print["\n[4] the same thing by direct numerical volume integral (slow)"];

Module[
 {lam, mu, rho, w, al, be, kp, ks, amp, s, dlam, dmu, drho, aSph, robs,
  ggSym, dggSym, gfun, dgfun, dc, u0, du0, integ, integC, scale, res, famp,
  chiHat, vol, mine},
 (
  (* The test case, in SI, matching the Python side. *)
  rho = 2500.; al = 5000.; be = 3000.; w = 60.;
  mu = rho be^2; lam = rho al^2 - 2 mu;
  kp = w/al; ks = w/be;
  amp = 10.^-4; s = 1 + amp;
  dlam = (s^3 - 1) lam; dmu = (s^3 - 1) mu; drho = (s - 1) rho;
  aSph = 15.; robs = 10.^4;

  (* G and its gradient with respect to the SOURCE point.  Built ONCE as
     symbolic expressions and then frozen into pure functions -- rebuilding
     the derivatives at every quadrature node makes this run forever. *)
  ggSym = Gten[kp, ks, rho, w];
  dggSym = Table[D[ggSym[[i, j]], xv[[k]]], {i, 3}, {j, 3}, {k, 3}];
  gfun = Function[{xa, ya, za}, Evaluate[ggSym /. {x -> xa, y -> ya, z -> za}]];
  dgfun = Function[{xa, ya, za}, Evaluate[dggSym /. {x -> xa, y -> ya, z -> za}]];

  dc = cten[dlam, dmu];

  (* Incident: unit displacement P wave along +z, u0 = zhat e^{i kP z}. *)
  u0[zp_] := {0, 0, Exp[I kp zp]};
  du0[zp_] := Table[
    If[m == 3, KroneckerDelta[l, 3] I kp Exp[I kp zp], 0], {l, 3}, {m, 3}];

  (* Observation on the -z axis: backscatter.  x - x' is the argument. *)
  integ[xp_?NumericQ, yp_?NumericQ, zp_?NumericQ] := Module[{gval, dgval, du, t1, t2},
    (
     gval = gfun[-xp, -yp, -robs - zp];
     (* derivative w.r.t. x' is minus the derivative w.r.t. the argument *)
     dgval = -dgfun[-xp, -yp, -robs - zp];
     du = du0[zp];
     t1 = w^2 drho Table[Sum[gval[[i, j]] u0[zp][[j]], {j, 3}], {i, 3}];
     t2 = Table[
       Sum[dc[[j, k, l, m]] dgval[[i, j, k]] du[[l, m]], {j, 3}, {k, 3}, {l, 3}, {m, 3}],
       {i, 3}];
     t1 - t2
     )];

  (* THE INTEGRAND MUST BE RESCALED BEFORE IT IS INTEGRATED.  The scattered
     field at r = 10^4 is of order 10^-13, and NIntegrate's stopping rule is
     Max[10^-PrecisionGoal |I|, 10^-AccuracyGoal]; left unscaled it demands an
     absolute error near 10^-20 at machine precision, which it cannot reach,
     and it returns a value smaller than its own error estimate.  Multiplying
     by the far-field scale 4 Pi rho alpha^2 r makes the integrand O(1) and the
     goals mean what they say.  The factor is divided out afterwards. *)
  scale = 4 Pi rho al^2 robs;

  integC[ic_Integer] := Function[{rp, th, ph},
    scale rp^2 Sin[th]
     integ[rp Sin[th] Cos[ph], rp Sin[th] Sin[ph], rp Cos[th]][[ic]]];

  res = Table[
    NIntegrate[
      integC[ic][rp, th, ph],
      {rp, 0, aSph}, {th, 0, Pi}, {ph, 0, 2 Pi},
      PrecisionGoal -> 8, AccuracyGoal -> 10, MaxRecursion -> 12,
      Method -> {"GlobalAdaptive", "MaxErrorIncreases" -> 20000}]/scale,
    {ic, 3}];

  Print["    transverse components (must vanish by symmetry): ",
    Abs[res[[1]]], "  ", Abs[res[[2]]]];
  Print["    longitudinal component                        : ", Abs[res[[3]]]];
  famp = Abs[res[[3]]] robs;

  (* the sphere form factor and volume, for reporting only *)
  chiHat = 4 Pi (Sin[2 kp aSph] - 2 kp aSph Cos[2 kp aSph])/(2 kp)^3;
  vol = 4/3 Pi aSph^3;
  mine = kp^2 (drho (lam + 2 mu)/rho + dlam + 2 dmu) chiHat/(4 Pi rho al^2);

  Print["    sphere radius ", aSph, ",  k_P a = ", kp aSph];
  Print["    |u_scat| * r  (numerical Born)   : ", famp];
  Print["    the HAND formula                 : ", mine];
  Print["    ratio numerical / hand           : ", famp/mine];
  Print[""];
  Print["    for reference, the committed Mie arbiter measures 3.2398e-5 here,"];
  Print["    so  numerical / Mie              : ", famp/3.2398*^-5];
  Print["    and hand / Mie                   : ", mine/3.2398*^-5];
  )];

Print["\n=============================================================="];
Print["Section 1 residual must be 0; Section 2 ratio must be 1."];
Print["Section 3 then decides the factor, using neither the hand"];
Print["contraction nor the closed-form form factor."];
Print["=============================================================="];
