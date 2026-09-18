#!/usr/bin/env wolframscript
(* ============================================================================
   TASK 1, STEP 1 --- the whole-space propagator Gamma, from A alone

   Plan: docs/plans/2026-09-18-first-order-propagator-moment.md

   Everything downstream integrates Gamma laterally.  This file builds it and
   proves it is the Green's matrix of the first-order system, before any
   integral is attempted.

   WHY SPECTRAL PROJECTORS AND NOT EIGENVECTORS.  The thesis writes the
   eigen-matrix column by column (GRepresentations.tex, Peigen/SVeigen/SHeigen)
   with normalisation factors eps_P, eps_S, eps_H left symbolic.  Those factors
   are exactly where the k_par -> 0 trap lives: they are fixed by a flux
   normalisation that degenerates at normal incidence, where the SV and SH
   polarisations cease to be distinguished by the horizontal direction.  The
   SPECTRAL PROJECTOR onto an eigenspace is invariant under any rescaling of the
   eigenvectors that span it, so building Gamma from projectors removes the
   ambiguity instead of carrying it.

       P_d = V_d (W_d^t V_d)^-1 W_d^t

   with V_d the right and W_d the left eigenvectors of the downgoing branch.
   This is unchanged if any column of V_d is rescaled.

   THE RADIATING GREEN'S MATRIX.  For d_z q = A q + d delta(z - z'),

       Gamma(z,z') = +P_d Exp[A (z-z')]   for z > z'
                     -P_u Exp[A (z-z')]   for z < z'

   so that it decays away from the source in both directions and jumps by
   P_d + P_u = I across it.  Both properties are checked below rather than
   asserted.

   OMEGA IS COMPLEX THROUGHOUT.  k_{z,c} = Sqrt[k_c^2 - k^2] vanishes on the
   circles |k| = k_P and |k| = k_S, which the lateral integral of Task 1 must
   cross.  The directional-sweep work already measured that a real-omega
   quadrature does not converge there (1.9e-1 -> 1.1e-1 over an 8x refinement,
   non-monotone) and types omega complex for exactly this reason.  The same
   applies here and is built in from the start, not discovered later.
   ============================================================================ *)

bar = StringJoin @@ ConstantArray["=", 60];
Print[bar];
Print["  TASK 1 step 1: Gamma, the whole-space first-order Green's matrix"];
Print[bar];

nPass = 0; nFail = 0;
report[label_, bool_] := (
   If[TrueQ[bool], nPass++, nFail++];
   Print[If[TrueQ[bool], "  PASS  ", "  ****FAIL****  "], label]);

(* --------------------------------------------------------------------------
   A(k1,k2) in the basis (u_z, u_x, u_y, T_zz, T_xz, T_yz) -- the thesis form,
   proved equal to the paper's operator matrix in MatrixVectorWaveEquation.wl.
   -------------------------------------------------------------------------- *)
amat[lam_, mu_, rho_, w_, k1_, k2_] := Module[{kc, gam, aa, bb, zet, chi, rw2},
   kc = lam + 2 mu;
   gam = lam/kc; aa = 1/kc; bb = 1/mu;
   zet = 4 mu (lam + mu)/kc; chi = 2 mu lam/kc;
   rw2 = rho w^2;
   {{0, -I gam k1, -I gam k2, aa, 0, 0},
    {-I k1, 0, 0, 0, bb, 0},
    {-I k2, 0, 0, 0, 0, bb},
    {-rw2, 0, 0, 0, -I k1, -I k2},
    {0, -rw2 + zet k1^2 + mu k2^2, k1 k2 (chi + mu), -I k1 gam, 0, 0},
    {0, k1 k2 (chi + mu), -rw2 + zet k2^2 + mu k1^2, -I k2 gam, 0, 0}}];

(* numeric background: alpha = 5, beta = 3, rho = 2.5 (km/s, g/cm^3), and a
   COMPLEX omega, damping 0.02 -- inside the 0.01-0.03 band the sweep work
   measured as restoring convergence across the branch circles. *)
alphaN = 5.0; betaN = 3.0; rhoN = 2.5;
muN = rhoN betaN^2; lamN = rhoN alphaN^2 - 2 muN;
wRe = 6.0; damp = 0.02; wN = wRe (1 + I damp);

aNum[k1_, k2_] := amat[lamN, muN, rhoN, wN, k1, k2];

(* --------------------------------------------------------------------------
   Spectral projectors, built without normalising anything.
   -------------------------------------------------------------------------- *)
projectors[a_] := Module[{ev, rv, lv, down, up, vd, wd, vu, wu, pd, pu},
   {ev, rv} = Eigensystem[a];
   (* Eigensystem returns right eigenvectors as ROWS, so the matrix with them as
      COLUMNS is Transpose[rv] = V, and V^-1 A V = Lambda means the ROWS of
      V^-1 are the left eigenvectors.  One transpose too many here silently
      produces idempotent matrices that do not sum to I. *)
   lv = Inverse[Transpose[rv]];
   (* downgoing = decaying as z increases, i.e. Re[eigenvalue] < 0 once omega
      carries damping.  With a lossless medium this test is degenerate, which is
      a second reason omega is complex here. *)
   down = Select[Range[6], Re[ev[[#]]] < 0 &];
   up = Complement[Range[6], down];
   vd = Transpose[rv[[down]]]; wd = Transpose[lv[[down]]];
   vu = Transpose[rv[[up]]]; wu = Transpose[lv[[up]]];
   pd = vd.Inverse[Transpose[wd].vd].Transpose[wd];
   pu = vu.Inverse[Transpose[wu].vu].Transpose[wu];
   {pd, pu, ev, down}];

(* ⚠ NEVER FORM MatrixExp[A dz].  It contains the GROWING branch, with entries
   of order Exp[+|Re s| dz]; projecting afterwards cancels them only to relative
   machine precision, so at dz = 640 the growing entries reach 1e12 and leave
   1e-4 of absolute noise on a true answer of 3e-8.  That is the Thomson-Haskell
   instability -- the disease invariant imbedding exists to avoid -- and it
   appears here as a decay rate that goes NEGATIVE at large separation.

   Applying the exponential to the PROJECTED eigenvalues instead never forms the
   growing branch at all:

       Gamma(dz > 0) = Sum_{down} Exp[s_i dz] v_i w_i^t
       Gamma(dz < 0) = -Sum_{up}   Exp[s_i dz] v_i w_i^t

   with w_i . v_j = delta_ij by construction, since lv = Inverse[Transpose[rv]].

   For the moment integral |z - z'| <= 2a, where this never bites.  It will bite
   for a layered background with thick layers, which is Task 6. *)
gammaSpec[a_, dz_] := Module[{ev, rv, lv, sel},
   {ev, rv} = Eigensystem[a];
   lv = Inverse[Transpose[rv]];
   sel = Select[Range[6], If[dz > 0, Re[ev[[#]]] < 0, Re[ev[[#]]] > 0] &];
   (If[dz > 0, 1, -1]) Sum[Exp[ev[[i]] dz] Outer[Times, rv[[i]], lv[[i]]], {i, sel}]];

gammaOf[a_, pd_, pu_, dz_] := gammaSpec[a, dz];

(* --------------------------------------------------------------------------
   Checks, at a generic lateral wavenumber.
   -------------------------------------------------------------------------- *)
k1t = 0.7; k2t = 0.4;
at = aNum[k1t, k2t];
{pdt, put, evt, downt} = projectors[at];

Print[""];
Print["--- the spectrum -----------------------------------------------------"];
Print["  eigenvalues: ", Chop[evt, 10^-12]];
kzP = Sqrt[(wN/alphaN)^2 - k1t^2 - k2t^2];
kzS = Sqrt[(wN/betaN)^2 - k1t^2 - k2t^2];
Print["  expected +-I kzP, +-I kzS (twice): kzP = ", kzP, "  kzS = ", kzS];
(* compare with a TOLERANCE: === on floating point is exact structural equality
   and will reject a correct spectrum over the last bit *)
nPmode = Count[evt^2, x_ /; Abs[x + kzP^2] < 10^-8 Abs[kzP^2]];
nSmode = Count[evt^2, x_ /; Abs[x + kzS^2] < 10^-8 Abs[kzS^2]];
Print["  modes found: ", nPmode, " P (want 2), ", nSmode, " S (want 4)"];
report["spectrum is {+-I kzP} and {+-I kzS} twice", nPmode === 2 && nSmode === 4];
report["the downgoing branch has exactly three members", Length[downt] === 3];

Print[""];
Print["--- the projectors ---------------------------------------------------"];
report["P_d + P_u = I", Max[Abs[pdt + put - IdentityMatrix[6]]] < 10^-10];
report["P_d is idempotent", Max[Abs[pdt.pdt - pdt]] < 10^-10];
report["P_u is idempotent", Max[Abs[put.put - put]] < 10^-10];
report["P_d commutes with A", Max[Abs[pdt.at - at.pdt]] < 10^-10];

(* the projector must be invariant under rescaling the eigenvectors -- the whole
   reason for using it.  Rebuild with each right eigenvector randomly rescaled
   and confirm P_d is unchanged. *)
SeedRandom[7];
rescaleTest := Module[{ev, rv, lv, sc, rv2, lv2, down, vd, wd},
   {ev, rv} = Eigensystem[at];
   sc = RandomComplex[{0.3 - 0.3 I, 3 + 3 I}, 6];
   rv2 = MapThread[#2 #1 &, {rv, sc}];
   lv2 = Inverse[Transpose[rv2]];
   down = Select[Range[6], Re[ev[[#]]] < 0 &];
   vd = Transpose[rv2[[down]]]; wd = Transpose[lv2[[down]]];
   vd.Inverse[Transpose[wd].vd].Transpose[wd]];
report["P_d is invariant under rescaling the eigenvectors (the point of it)",
  Max[Abs[rescaleTest - pdt]] < 10^-10];

Print[""];
Print["--- Gamma solves the system ------------------------------------------"];
dz0 = 0.37;
num = (gammaOf[at, pdt, put, dz0 + 10^-6] - gammaOf[at, pdt, put, dz0 - 10^-6])/(2 10^-6);
report["d_z Gamma = A Gamma for z > z'",
  Max[Abs[num - at.gammaOf[at, pdt, put, dz0]]]/Max[Abs[at.gammaOf[at, pdt, put, dz0]]] < 10^-6];
dzm = -0.37;
numm = (gammaOf[at, pdt, put, dzm + 10^-6] - gammaOf[at, pdt, put, dzm - 10^-6])/(2 10^-6);
report["d_z Gamma = A Gamma for z < z'",
  Max[Abs[numm - at.gammaOf[at, pdt, put, dzm]]]/Max[Abs[at.gammaOf[at, pdt, put, dzm]]] < 10^-6];

jump = gammaOf[at, pdt, put, 10^-9] - gammaOf[at, pdt, put, -10^-9];
report["the jump across the source is exactly I",
  Max[Abs[jump - IdentityMatrix[6]]] < 10^-6];

Print[""];
Print["--- decay, which is what picks the radiating branch ------------------"];
(* The point of this check is the BRANCH, not the magnitude.  At damping 0.02
   the slowest decay rate is Re[s] = 0.0324 per km, a decay length of ~31 km, so
   Gamma is still O(1) at dz = 40 -- that is correct, not a failure.  What must
   hold is that it DECREASES: the wrong branch would grow exponentially.  For
   the moment integral only |z - z'| <= 2a matters anyway, where there is
   essentially no decay at all; the far field is tested solely to confirm the
   branch selection. *)
gAt[d_] := Max[Abs[gammaOf[at, pdt, put, d]]];
slow = Min[Abs[Re[evt]]];
Print["  slowest decay rate Re[s] = ", slow, "  -> decay length ", 1/slow, " km"];
(* The implied rate must CONVERGE to the slowest eigenvalue, not match it at an
   arbitrary separation: at dz = 40 the faster P mode has not yet died and the
   ratio still carries its admixture.  Checking a rate at one finite distance
   would reject a correct propagator. *)
rateAt[d1_, d2_] := Log[gAt[d1]/gAt[d2]]/(d2 - d1);
pairs = {{40.0, 80.0}, {80.0, 160.0}, {160.0, 320.0}, {320.0, 640.0}};
rates = rateAt @@ # & /@ pairs;
Print["  |Gamma| at dz = 10, 40, 80: ", {gAt[10.0], gAt[40.0], gAt[80.0]}];
Print["  implied rate over successive octaves: ", rates];
Print["  must converge to Re[s] = ", slow];
report["the decay rate converges to the slowest eigenvalue (radiating branch)",
  Abs[Last[rates]/slow - 1] < 0.01 &&
   Abs[rates[[-1]] - slow] < Abs[rates[[1]] - slow]];
report["Gamma decays in the OTHER direction too (branch selection is symmetric)",
  Max[Abs[gammaOf[at, pdt, put, -640.0]]] < Max[Abs[gammaOf[at, pdt, put, -320.0]]] <
   Max[Abs[gammaOf[at, pdt, put, -40.0]]]];

(* the branch circles the lateral integral must cross *)
Print[""];
Print["--- the branch circles Task 1 must integrate across -------------------"];
Print["  |k| = kP = ", Re[wN]/alphaN, "   |k| = kS = ", Re[wN]/betaN];
Print["  with omega complex (damping ", damp, ") kz never vanishes on the real"];
Print["  k axis, so the integrand stays bounded there."];
kAtP = Re[wN]/alphaN;
atP = aNum[kAtP, 0.0];
{pdP, puP, evP, downP} = projectors[atP];
Print["  smallest |eigenvalue| exactly on the P circle: ",
  Min[Abs[evP]], "  (would be 0 at real omega)"];
report["the propagator is still finite on the P branch circle",
  Min[Abs[evP]] > 10^-4 && Max[Abs[gammaOf[atP, pdP, puP, 0.5]]] < 10^6];

(* ============================================================================
   TASK 1, STEP 2 --- the z double integral, in closed form

   Gamma is exponential in z - z', so the vertical integral is analytic.  But it
   is NOT a single symmetric integral: the two halves of Gamma carry DIFFERENT
   matrices,

       Gamma(dz > 0) = +Sum_down Exp[-q_i  dz] v_i w_i^t
       Gamma(dz < 0) = -Sum_up   Exp[-q_j |dz|] v_j w_j^t

   so the double integral splits over the two TRIANGLES z > z' and z < z', and
   the two cannot be combined into an integral over the square.  Treating the
   kernel as a function of |z - z'| alone would silently symmetrise the
   propagator and destroy the up/down distinction that the first-order system
   exists to carry.

   The trial fields are affine, so the weights are 1 and z.  Four triangular
   integrals are needed,

       T[m,n](q) = Int_{-a}^{a} dz Int_{-a}^{z} dz'  z^m z'^n Exp[-q (z - z')],

   and the lower triangle is T[n,m] by relabelling.
   ============================================================================ *)

Print[""];
Print["--- STEP 2: the z double integral over each triangle -----------------"];

tri[m_, n_] := tri[m, n] = Simplify[
    Integrate[z^m zp^n Exp[-q (z - zp)], {z, -aa, aa}, {zp, -aa, z}],
    Assumptions -> Re[q] > 0 && aa > 0];

Do[Print["  T[", m, ",", n, "] = ", tri[m, n]], {m, 0, 1}, {n, 0, 1}];

(* numerical control: the closed forms must reproduce brute-force quadrature *)
qT = 1.3 + 0.21 I; aT = 0.5;
(* NIntegrate at default MachinePrecision returns only 6-8 digits, so a bare
   call disagrees with the closed form at ~1e-9 and looks like an error in the
   closed form.  The project's standing setting is used instead. *)
numTri[m_, n_] := NIntegrate[
   z^m zp^n Exp[-qT (z - zp)], {z, -aT, aT}, {zp, -aT, z},
   PrecisionGoal -> 12, AccuracyGoal -> 12, MaxRecursion -> 25];
Do[Print["  T[", m, ",", n, "] closed ", (tri[m, n] /. {q -> qT, aa -> aT}),
   "   quadrature ", numTri[m, n]], {m, 0, 1}, {n, 0, 1}];
report["the four closed forms reproduce direct quadrature",
  Max[Table[Abs[(tri[m, n] /. {q -> qT, aa -> aT}) - numTri[m, n]],
     {m, 0, 1}, {n, 0, 1}]] < 10^-10];

(* the symmetric combination, as a check on the triangle split: summing the two
   triangles with the SAME matrix must give the familiar square integral *)
jsquare = Simplify[tri[0, 0] + tri[0, 0], Assumptions -> Re[q] > 0 && aa > 0];
jclosed = 2 (2 aa q - 1 + Exp[-2 aa q])/q^2;
report["T[0,0] + T[0,0] = 2(2aq - 1 + Exp[-2aq])/q^2  (the square integral)",
  Simplify[jsquare - jclosed, Assumptions -> Re[q] > 0 && aa > 0] === 0];

(* parity: with a symmetric kernel the odd weight must vanish over the square *)
report["the odd weight cancels between the triangles (parity)",
  Simplify[tri[1, 0] + tri[0, 1] - Integrate[z Exp[-q Abs[z - zp]], {z, -aa, aa},
      {zp, -aa, aa}, Assumptions -> Re[q] > 0 && aa > 0],
    Assumptions -> Re[q] > 0 && aa > 0] === 0];

Print[""];
Print["--- the large-q behaviour that drives the lateral asymptote ----------"];
Do[Print["  T[", m, ",", n, "] ~ ",
   Normal@Series[tri[m, n], {q, Infinity, 2}]], {m, 0, 1}, {n, 0, 1}];
report["T[0,0] ~ 2a/q at large q (each triangle carries half the square)",
  Simplify[Limit[q tri[0, 0], q -> Infinity] - 2 aa,
    Assumptions -> aa > 0] === 0];

(* the whole point of step 2: the closed-form z integration must reproduce a
   brute-force double integral of the ACTUAL Gamma, matrices and all *)
Print[""];
Print["--- the assembled z-integrated kernel vs brute force -----------------"];
zKernel[a6_, aHalf_] := Module[{ev, rv, lv, dn, upp, acc},
   {ev, rv} = Eigensystem[a6];
   lv = Inverse[Transpose[rv]];
   dn = Select[Range[6], Re[ev[[#]]] < 0 &];
   upp = Complement[Range[6], dn];
   acc = Sum[(tri[0, 0] /. {q -> -ev[[i]], aa -> aHalf}) Outer[Times, rv[[i]], lv[[i]]],
      {i, dn}]
     - Sum[(tri[0, 0] /. {q -> ev[[j]], aa -> aHalf}) Outer[Times, rv[[j]], lv[[j]]],
        {j, upp}];
   acc];

(* gammaSpec branches on If[dz > 0, ...], which does not evaluate on a symbolic
   dz -- the integrand then comes back as literal 0.  Guard the argument as
   NumericQ, precompute the eigen-decomposition once, and integrate the two
   triangles separately so the kink at z = z' never sits inside a panel. *)
Module[{ev, rv, lv, dn, upp, outers},
  {ev, rv} = Eigensystem[at];
  lv = Inverse[Transpose[rv]];
  dn = Select[Range[6], Re[ev[[#]]] < 0 &];
  upp = Complement[Range[6], dn];
  outers = Table[Outer[Times, rv[[i]], lv[[i]]], {i, 6}];
  gDn[dz_?NumericQ] := Sum[Exp[ev[[i]] dz] outers[[i]], {i, dn}];
  gUp[dz_?NumericQ] := -Sum[Exp[ev[[j]] dz] outers[[j]], {j, upp}];
];
(* NIntegrate preprocesses the integrand symbolically before any NumericQ guard
   can bite, which turns gDn[z - zp][[r,c]] into Part of an unevaluated head.
   SymbolicProcessing -> 0 stops that; the guards then do their job. *)
entDn[r_, c_][zv_?NumericQ, zpv_?NumericQ] := gDn[zv - zpv][[r, c]];
entUp[r_, c_][zv_?NumericQ, zpv_?NumericQ] := gUp[zv - zpv][[r, c]];
bruteZ = Table[
   NIntegrate[entDn[r, c][z, zp], {z, -aT, aT}, {zp, -aT, z},
     Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}]
   + NIntegrate[entUp[r, c][z, zp], {z, -aT, aT}, {zp, z, aT},
     Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}],
   {r, 6}, {c, 6}];
closedZ = zKernel[at, aT];
Print["  max |closed - brute| = ", Max[Abs[closedZ - bruteZ]]];
Print["  relative             = ", Max[Abs[closedZ - bruteZ]]/Max[Abs[bruteZ]]];
report["the closed-form z integration reproduces the brute-force double integral",
  Max[Abs[closedZ - bruteZ]]/Max[Abs[bruteZ]] < 10^-8];

(* ============================================================================
   TASK 1, STEP 3a --- the large-k structure, and the scaling that exposes it

   A is NOT uniformly O(k).  Rows 1-3 carry k, but rows 5-6 carry k^2, because
   the basis mixes displacement with traction and those have different
   dimensions.  Expanding A in 1/k as it stands mixes orders between blocks and
   the eigen-perturbation is a mess.

   The traction of an evanescent field scales as mu k u, so the natural balance
   is q -> (u, T/k).  Under S = diag(1,1,1,k,k,k),

       Atil = S^-1 A S

   is uniformly O(k): the (1-3, 4-6) block gains a k, the (4-6, 1-3) block loses
   one, and the diagonal blocks are unchanged.  Then Atil/k has a finite limit,
   the eigenvalues tend to +-k, and the projectors tend to k-independent limits
   -- which is what makes I_inf extractable at all.

   The scaling is a similarity transform, so it changes no eigenvalue.  It is a
   change of variables for the ASYMPTOTICS only.
   ============================================================================ *)

Print[""];
Print["--- STEP 3a: the scaling that makes the 1/k expansion uniform --------"];

sScale[kk_] := DiagonalMatrix[{1, 1, 1, kk, kk, kk}];
aTil[kk_, th_] := Module[{s = sScale[kk]},
   Inverse[s].aNum[kk Cos[th], kk Sin[th]].s];

thT = 0.37;
Print["  max |A/k| entries at k = 10^2, 10^3, 10^4 (UNSCALED, should blow up):"];
Print["   ", Table[Max[Abs[aNum[kk Cos[thT], kk Sin[thT]]]]/kk,
   {kk, {10.^2, 10.^3, 10.^4}}]];
Print["  max |Atil/k| at the same k (SCALED, should settle):"];
scl = Table[Max[Abs[aTil[kk, thT]]]/kk, {kk, {10.^2, 10.^3, 10.^4}}];
Print["   ", scl];
report["the scaling makes A uniformly O(k)",
  Abs[scl[[3]]/scl[[2]] - 1] < 10^-3 && Abs[scl[[2]]/scl[[1]] - 1] < 10^-2];
report["the unscaled matrix does NOT settle (so the scaling is doing work)",
  Max[Abs[aNum[10.^4 Cos[thT], 10.^4 Sin[thT]]]]/10.^4 >
   10 Max[Abs[aNum[10.^2 Cos[thT], 10.^2 Sin[thT]]]]/10.^2];

report["the scaling is a similarity transform, so the spectrum is unchanged",
  Max[Abs[Sort[Eigenvalues[aTil[137.0, thT]]] -
      Sort[Eigenvalues[aNum[137.0 Cos[thT], 137.0 Sin[thT]]]]]] < 10^-6];

Print[""];
Print["--- the eigenvalues approach +-k, as evanescence requires ------------"];
Do[Print["   k = ", kk, "   |s|/k = ",
   Sort[Abs[Eigenvalues[aNum[kk Cos[thT], kk Sin[thT]]]]/kk]],
 {kk, {10.^2, 10.^3, 10.^4}}];
report["every |s|/k -> 1",
  Max[Abs[Abs[Eigenvalues[aNum[10.^4 Cos[thT], 10.^4 Sin[thT]]]]/10.^4 - 1]] < 10^-6];

(* ----------------------------------------------------------------------------
   STEP 3b --- the large-k behaviour of the z-integrated kernel

   With the eigenvalues at +-k the z integral T00 ~ 2a/q ~ 2a/k, so the kernel
   decays like 1/k while its matrices tend to constants.  The order in 1/k that
   each entry carries is READ OFF numerically rather than assumed: the symbolic
   6x6 eigen-perturbation is avoidable, and a measured power is harder to fool
   oneself with than a derived one.
   ---------------------------------------------------------------------------- *)
Print[""];
Print["--- STEP 3b: the order in 1/k that the z-integrated kernel carries ----"];

(* ⚠ The eigen-decomposition must be done on the SCALED matrix.  On the
   unscaled one the down- and up-going eigenvectors become nearly parallel as
   k grows -- the evanescent limit degenerates -- and Inverse reports a badly
   conditioned matrix while the kernel entries collapse to 1e-13 of pure
   roundoff, from which any fitted power is meaningless.  That is precisely the
   conditioning the scaling of step 3a was introduced to remove, so it must
   actually be used: decompose Atil, assemble there, then map back with
   K = S K_scaled S^-1.  Extended precision on top, since the exponentials
   underflow at machine precision once 2 a q reaches a few hundred. *)
kernelAt[kk_, th_, aHalf_] := Module[{prec = 40, s, a6, ev, rv, lv, dn, upp, ks2},
   s = SetPrecision[sScale[kk], prec];
   a6 = SetPrecision[aTil[kk, th], prec];
   {ev, rv} = Eigensystem[a6];
   lv = Inverse[Transpose[rv]];
   dn = Select[Range[6], Re[ev[[#]]] < 0 &];
   upp = Complement[Range[6], dn];
   ks2 = Sum[(tri[0, 0] /. {q -> -ev[[i]], aa -> aHalf}) Outer[Times, rv[[i]], lv[[i]]],
       {i, dn}]
     - Sum[(tri[0, 0] /. {q -> ev[[j]], aa -> aHalf}) Outer[Times, rv[[j]], lv[[j]]],
        {j, upp}];
   s.ks2.Inverse[s]];

(* the scaled route must agree with step 2's verified kernel at moderate k,
   where both are well conditioned -- otherwise the transform back is wrong *)
report["the scaled assembly reproduces step 2's kernel at moderate k",
  Max[Abs[kernelAt[Sqrt[k1t^2 + k2t^2], ArcTan[k1t, k2t], aT] - closedZ]]/
    Max[Abs[closedZ]] < 10^-10];

ks = {10.^2, 10.^3, 10.^4};
kerns = kernelAt[#, thT, 0.5] & /@ ks;
Print["  entry (1,1) at k = ", ks, ":"];
Print["   ", (#[[1, 1]] & /@ kerns)];
Print["  implied power of k, log ratio over a decade, entrywise:"];
powers = Table[
   If[Abs[kerns[[2, r, c]]] > 10^-30 && Abs[kerns[[3, r, c]]] > 10^-30,
    Round[Log[10, Abs[kerns[[3, r, c]]/kerns[[2, r, c]]]], 0.01], Indeterminate],
   {r, 6}, {c, 6}];
Do[Print["   ", powers[[r]]], {r, 6}];
Print["  (-1 means the entry decays as 1/k; 0 means it does NOT decay at all,"];
Print["   and 0 is what the k-integral cannot tolerate in the strain channel.)"];

(* ============================================================================
   TASK 1, STEP 3c --- what I_inf actually is, and therefore how to integrate it

   The non-decaying block has a limit C = lim_{k -> inf} K(k).  Whether C depends
   on the DIRECTION khat decides the whole treatment:

   * if C is direction-independent, then Khat -> C, a constant, and by Parseval
     that piece contributes  C Int_V f g dx  -- a LOCAL term, finite, in closed
     form, with no regulator needed at all.  The "divergence" is then an artefact
     of integrating term by term instead of recognising the structure.

   * if C depends on khat, the constant is a delta carrying an angular kernel,
     the contribution is a principal-value-type distribution, and the regulator
     and the epsilon -> 0 limit are unavoidable.

   This is measured before anything is integrated.
   ============================================================================ *)

Print[""];
Print["--- STEP 3c: is the asymptote direction-dependent? -------------------"];

(* the five entries that step 3b showed do not decay *)
nonDecay = {{4, 1}, {5, 2}, {5, 3}, {6, 2}, {6, 3}};

cAt[th_] := Module[{k = 10.^4, kk},
   kk = kernelAt[k, th, 0.5];
   (kk[[Sequence @@ #]] & /@ nonDecay)];

thetas = {0.0, 0.3, 0.7854, 1.1, 1.5708, 2.4};
cvals = cAt /@ thetas;
Print["  theta        C[T_zz<-u_z]        C[T_xz<-u_x]"];
Do[Print["  ", thetas[[i]], "   ", Re[Chop[cvals[[i, 1]], 10^-12]],
   "   ", Re[Chop[cvals[[i, 2]], 10^-12]]], {i, Length[thetas]}];

(* convergence: the limit must be stable in k, or it is not a limit *)
cK[th_, k_] := Module[{kk = kernelAt[k, th, 0.5]},
   (kk[[Sequence @@ #]] & /@ nonDecay)];
Print[""];
Print["  stability in k at theta = 0.7854:"];
Do[Print["   k = ", k, "  ", Re[Chop[cK[0.7854, k], 10^-12]]],
 {k, {10.^3, 10.^4, 10.^5}}];
report["the asymptote is a genuine limit (stable over two decades in k)",
  Max[Abs[cK[0.7854, 10.^5] - cK[0.7854, 10.^4]]] <
   10^-3 Max[Abs[cK[0.7854, 10.^4]]]];

(* ----------------------------------------------------------------------------
   The asymptote in CLOSED FORM.  The numbers are recognisable: mu = rho beta^2
   = 22.5 and nu1 = 4 mu (lam+mu)/(lam+2mu) = 57.6, the same nu1 that appears in
   A12.  The claim is

       C[T_zz <- u_z]      = -nu1                        (isotropic)
       C[T_ab <- u_b]      = mu (khat_a khat_b - d_ab)   (in-plane)

   Checked as a RATE: the residual must fall as 1/k, since the corrections to
   the evanescent limit are O(k_c/k).  Matching at one k would not distinguish a
   correct closed form from a coincidence.
   ---------------------------------------------------------------------------- *)
nu1N = 4 muN (lamN + muN)/(lamN + 2 muN);
resAt[k_] := Module[{kk = kernelAt[k, 0.7854, 0.5], ch, cx, cy},
   ch = kk[[4, 1]]; cx = kk[[5, 2]]; cy = kk[[5, 3]];
   {Abs[ch + nu1N]/nu1N,
    Abs[cx - muN (Cos[0.7854]^2 - 1)]/muN,
    Abs[cy - muN Cos[0.7854] Sin[0.7854]]/muN}];
r3 = resAt[10.^3]; r4 = resAt[10.^4]; r5 = resAt[10.^5];
Print[""];
Print["  closed-form asymptote: C[T_zz<-u_z] = -nu1 = ", -nu1N];
Print["                         C_inplane    = mu (khat khat - delta), mu = ", muN];
Print["  relative residual at k = 1e3, 1e4, 1e5:"];
Print["   ", r3]; Print["   ", r4]; Print["   ", r5];
Print["  ratios per decade (want ~10, i.e. O(1/k)): ", r3/r4, "  ", r4/r5];
report["C[T_zz<-u_z] -> -nu1, residual falling as 1/k",
  9 < r3[[1]]/r4[[1]] < 11 && 9 < r4[[1]]/r5[[1]] < 11];
report["C_inplane -> mu (khat_a khat_b - delta_ab), residual falling as 1/k",
  9 < r3[[2]]/r4[[2]] < 11 && 9 < r4[[2]]/r5[[2]] < 11 &&
   9 < r3[[3]]/r4[[3]] < 11 && 9 < r4[[3]]/r5[[3]] < 11];

spread = Max[Table[Max[Abs[cvals[[i]] - cvals[[1]]]], {i, Length[cvals]}]];
scaleC = Max[Abs[Flatten[cvals]]];
Print[""];
Print["  max variation over theta = ", spread, "   relative ", spread/scaleC];
report["C DOES depend on direction (so the angular structure is real)",
  spread/scaleC > 10^-6];

(* If it is direction-dependent, the in-plane part should be built from the only
   two rank-2 structures available, delta_ab and khat_a khat_b.  Test that:
   the (T_xz <- u_x) entry should be P + Q Cos[th]^2 and (T_xz <- u_y) should be
   Q Cos[th] Sin[th], with a SINGLE pair (P, Q) fitting every angle. *)
inPlane[th_] := Module[{kk = kernelAt[10.^4, th, 0.5]},
   {kk[[5, 2]], kk[[5, 3]], kk[[6, 2]], kk[[6, 3]]}];
ipv = inPlane /@ thetas;
qFit = (ipv[[1, 1]] - ipv[[5, 1]]);           (* th = 0 minus th = pi/2 *)
pFit = ipv[[5, 1]];
Print[""];
Print["  fitting K[T_xz<-u_x] = P + Q Cos[th]^2 on two angles gives"];
Print["   P = ", Chop[pFit, 10^-12], "   Q = ", Chop[qFit, 10^-12]];
predXX = Table[pFit + qFit Cos[th]^2, {th, thetas}];
predXY = Table[qFit Cos[th] Sin[th], {th, thetas}];
Print["  predicted vs measured, K[T_xz<-u_x]:"];
Do[Print["   th=", thetas[[i]],
   "  pred ", Re[Chop[predXX[[i]], 10^-12]], "   meas ", Re[Chop[ipv[[i, 1]], 10^-12]]],
 {i, Length[thetas]}];
report["the in-plane asymptote IS  P delta_ab + Q khat_a khat_b  (2 constants, 6 angles)",
  Max[Abs[predXX - ipv[[All, 1]]]] < 10^-6 Max[Abs[ipv[[All, 1]]]] &&
   Max[Abs[predXY - ipv[[All, 2]]]] < 10^-6 Max[Abs[ipv[[All, 1]]]]];

Print[""];
Print[bar];
Print["  ", nPass, " passed, ", nFail, " failed"];
Print[bar];
If[nFail > 0, Exit[1]];
