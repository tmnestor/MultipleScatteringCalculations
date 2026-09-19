#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   MieSphericalWaves.wl

   Spherical vector-wave machinery for the elastic Mie field: the Cartesian
   derivative operators, the stress and the traction on a plane, and the
   plane-wave (Weyl) spectrum.

   WHAT THIS CONSOLIDATES, AND WHY IN ONE PLACE

   The material was spread across three sources that had never been brought
   together, and each carried something the others lacked:

     * MyRecursiveHarmonicDerivatives.nb (research archive, 2007-2019) -- the
       solid harmonic in CARTESIAN arguments, the ladder coefficients
       ap/am/bp/bm, the Cartesian derivative operators d_i and all nine d_ij
       as recursions on the same family with the radial function left as a
       pluggable argument, and the traction operator Tn for an arbitrary
       normal.  General in (n, m).
     * PW2H0.nb (same archive) -- the Weyl plane-wave identity for h_0, with
       the evanescent branch handled correctly.
     * the elastic Mie potentials, the full stress, and tau_3 = sigma . zhat,
       derived in Cartesian for this note (2026-09-19) and checked against
       Navier.  That derivation began life as a standalone MieStressTensor.wl,
       which was folded into Section 2 here and removed, so that there is one
       place to read and one place to change.

   The ladder coefficients are the normalised form of the textbook identities
   for curl(e_z Phi_ml) and curl curl(e_z Phi_ml): the l+1 and l-1 couplings
   with weights (l-m+1)/((l+1)(2l+1)) and (l+m)/(l(2l+1)).

   WHY THE ARCHIVE DEFINITIONS ARE EMITTED, NOT TRANSCRIBED

   MyRecursiveHarmonicDerivatives.nb is a working notebook: d_1_3 is defined
   six times, d_1 and d_2 four times each, ap/am/bp/bm twice.  Nothing in the
   document says which line is current.  Rather than pick, every variant was
   tested against direct Cartesian differentiation -- a knowable answer -- and
   they all agree, to 1e-29.  So the redundancy is harmless, but that is a
   measurement and not an assumption.  The definitions below are written out by
   Mathematica/emitVerifiedDefs.wl from the archive, so no character of them
   passes through a human hand.

   Regenerate with:
     wolframscript -file Mathematica/harvestDerivativeOperators.wl
     wolframscript -file Mathematica/verifyDerivativeOperators.wl
     wolframscript -file Mathematica/emitVerifiedDefs.wl
     wolframscript -file Mathematica/assembleMieSphericalWaves.wl

   Run:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/MieSphericalWaves.wl
   ===================================================================== *)

Print["=== MieSphericalWaves: consolidated spherical vector-wave machinery ==="];

(*@INSERT_ARCHIVE_DEFS@*)

(* =====================================================================
   SECTION 1.  The harvested operators reproduce direct differentiation.

   Re-checked here rather than taken on trust from the build step, so that the
   notebook a reader opens is self-validating.  The ladder coefficients are
   exercised implicitly: every d operator is built from them, so an error there
   would fail every line below.
   ===================================================================== *)

Print["\n[1] the Cartesian derivative operators, against direct differentiation"];

rX[x_, y_, z_] := Sqrt[x^2 + y^2 + z^2];
thX[x_, y_, z_] := ArcCos[z/rX[x, y, z]];
phX[x_, y_, z_] := ArcTan[x, y];
GX[K_, n_, m_][x_, y_, z_] :=
  SphericalBesselJ[n, K rX[x, y, z]] *
    SphericalHarmonicY[n, m, thX[x, y, z], phX[x, y, z]];

$p = {37/100, -61/100, 83/100};
$K = 11/10;
$nm = {{1, 0}, {1, 1}, {2, 1}, {2, -1}, {3, 2}};
sph[p_] := {Sqrt[p . p], ArcCos[p[[3]]/Sqrt[p . p]], ArcTan[p[[1]], p[[2]]]};

Module[{sp = sph[$p], worst1 = 0, worst2 = 0, got, want, x, y, z},
  Do[
    Do[
      got = N[Subscript[d, i][$K, nm[[1]], nm[[2]], sp[[1]], sp[[2]], sp[[3]]][J], 30];
      want = N[D[GX[$K, nm[[1]], nm[[2]]][x, y, z], {x, y, z}[[i]]] /.
         Thread[{x, y, z} -> $p], 30];
      worst1 = Max[worst1, Abs[got - want]/Max[Abs[want], 10^-30]],
      {i, 3}];
    Do[
      got = N[Subscript[d, i, j][$K, nm[[1]], nm[[2]], sp[[1]], sp[[2]], sp[[3]]][J], 30];
      want = N[D[D[GX[$K, nm[[1]], nm[[2]]][x, y, z], {x, y, z}[[i]]], {x, y, z}[[j]]] /.
         Thread[{x, y, z} -> $p], 30];
      worst2 = Max[worst2, Abs[got - want]/Max[Abs[want], 10^-30]],
      {i, 3}, {j, 3}],
    {nm, $nm}];
  Print["      first derivatives,  worst relative: ", N[worst1, 3]];
  Print["      second derivatives, worst relative: ", N[worst2, 3]]];

(* The solid harmonic must agree with the built-in at a real direction, and
   unlike the built-in it must stay finite for a COMPLEX direction -- which is
   what the evanescent part of the plane-wave spectrum needs, and what a
   real-angle evaluation cannot do. *)
Print["\n      the solid harmonic Y[n,m,x,y,z] against SphericalHarmonicY:"];
Module[{worst = 0, q = $p, s},
  s = sph[q];
  Do[worst = Max[worst, Abs[N[Y[nm[[1]], nm[[2]], q[[1]], q[[2]], q[[3]]] -
       SphericalHarmonicY[nm[[1]], nm[[2]], s[[2]], s[[3]]], 30]]], {nm, $nm}];
  Print["        real direction, worst absolute: ", N[worst, 3]];
  Print["        complex direction stays finite: ",
    AllTrue[$nm, FreeQ[N[Y[#[[1]], #[[2]], 3/10, -1/5, 2 I]], DirectedInfinity | Indeterminate] &]]];

(* =====================================================================
   SECTION 2.  The elastic Mie field: potentials, stress, traction.

   Everything Cartesian.  Deriving in spherical components and rotating
   afterwards invites curvilinear basis factors and a representation
   conversion, which is where this project's defects have historically lived.

   Angular factors are polynomial in z/r and carry no removable singularity on
   the axis:
     m = 0:  P_n(cos theta)
     m = 1:  P_n^1(cos theta) cos(phi) = -P_n'(z/r) (x/r), using the
             Condon-Shortley relation and Sqrt[1-u^2] cos(phi) = x/r.
   ===================================================================== *)

Print["\n[2] the elastic Mie field"];

angF[n_, 0][x_, y_, z_] := LegendreP[n, z/rX[x, y, z]];
angF[n_, 1][x_, y_, z_] :=
  -(D[LegendreP[n, u], u] /. u -> z/rX[x, y, z]) * (x/rX[x, y, z]);

radF[n_, k_][x_, y_, z_] := SphericalHankelH1[n, k rX[x, y, z]];
potF[n_, m_, k_][x_, y_, z_] := radF[n, k][x, y, z] angF[n, m][x, y, z];

(* P (L-type), SV (N-type), SH (M-type); r is the POSITION vector. *)
uP[n_, m_, kP_][x_, y_, z_] := Grad[potF[n, m, kP][x, y, z], {x, y, z}];
uSV[n_, m_, kS_][x_, y_, z_] :=
  Curl[Curl[{x, y, z} potF[n, m, kS][x, y, z], {x, y, z}], {x, y, z}];
uSH[n_, m_, kS_][x_, y_, z_] :=
  Curl[{x, y, z} potF[n, m, kS][x, y, z], {x, y, z}];

stressOf[u_, lam_, mu_, x_, y_, z_] := Module[{G = Grad[u, {x, y, z}]},
  lam Tr[G] IdentityMatrix[3] + mu (G + Transpose[G])];

(* tau_3 = sigma . zhat.  Equivalently Tn from the archive with norm = zhat,
   which Section 3 checks. *)
tau3Of[u_, lam_, mu_, x_, y_, z_] := stressOf[u, lam, mu, x, y, z][[All, 3]];

(* Exact rationals: N[expr, 30] cannot manufacture precision the inputs lack,
   and with machine-precision parameters the reference dump was good only to
   about 1e-9 -- enough to make a correct Python implementation look wrong. *)
$lam = 22/10; $mu = 13/10; $rho = 17/10; $om = 9/10;
$kP = $om/Sqrt[($lam + 2 $mu)/$rho];
$kS = $om/Sqrt[$mu/$rho];
Print["      kP = ", N[$kP, 12], "   kS = ", N[$kS, 12]];

(* =====================================================================
   SECTION 3.  Validation of the field: divergence, Navier, and Tn.

   [A] div u = -kP^2 phi for the P family and 0 for both shear families.
   [B] Navier, div sigma + rho omega^2 u = 0, in the exterior.  A physics
       identity a wrong component cannot accidentally satisfy, testing every
       component at once.  Normalised by the TERM scale, not by |u|: the two
       terms cancel, so an absolute residual would look small for a wrong
       stress simply because both sides are small.
   [C] the archive's Tn with norm = zhat reproduces tau_3, which ties the two
       independent routes together.
   ===================================================================== *)

Print["\n[3] validation"];
evalAt[e_, x_, y_, z_] := N[e /. Thread[{x, y, z} -> $p], 30];

Module[{x, y, z, want},
  Print["      [A] divergence identities"];
  Do[
    want = -$kP^2 potF[n, m, $kP][x, y, z];
    Print["        n=", n, " m=", m,
      "  |div uP + kP^2 phi| ", Abs[evalAt[Div[uP[n, m, $kP][x, y, z], {x, y, z}] - want, x, y, z]],
      "  |div uSV| ", Abs[evalAt[Div[uSV[n, m, $kS][x, y, z], {x, y, z}], x, y, z]],
      "  |div uSH| ", Abs[evalAt[Div[uSH[n, m, $kS][x, y, z], {x, y, z}], x, y, z]]],
    {n, 1, 2}, {m, 0, 1}]];

Module[{x, y, z, u, sig, res, scale},
  Print["      [B] Navier residual (relative)"];
  Do[
    u = Switch[fam, "P", uP[n, m, $kP][x, y, z],
      "SV", uSV[n, m, $kS][x, y, z], "SH", uSH[n, m, $kS][x, y, z]];
    sig = stressOf[u, $lam, $mu, x, y, z];
    res = Table[Sum[D[sig[[i, j]], {x, y, z}[[j]]], {j, 3}], {i, 3}] + $rho $om^2 u;
    scale = Max[Abs[evalAt[$rho $om^2 u, x, y, z]]];
    Print["        ", fam, " n=", n, " m=", m, "  ", Max[Abs[evalAt[res, x, y, z]]]/scale],
    {fam, {"P", "SV", "SH"}}, {n, 1, 2}, {m, 0, 1}]];

Module[{x, y, z, u, G, uu, viaTn, viaSig, worst = 0},
  Print["      [C] the archive traction operator Tn against tau_3"];
  Do[
    u = Switch[fam, "P", uP[n, 0, $kP][x, y, z],
      "SV", uSV[n, 0, $kS][x, y, z], "SH", uSH[n, 0, $kS][x, y, z]];
    G = Grad[u, {x, y, z}];
    uu[i_, j_] := G[[i, j]];
    viaTn = Table[Tn[i, uu, {0, 0, 1}] /. {λ -> $lam, μ -> $mu}, {i, 3}];
    viaSig = tau3Of[u, $lam, $mu, x, y, z];
    worst = Max[worst, Max[Abs[evalAt[viaTn - viaSig, x, y, z]]] /
       Max[Abs[evalAt[viaSig, x, y, z]]]],
    {fam, {"P", "SV", "SH"}}, {n, 1, 2}];
  Print["        worst relative difference: ", N[worst, 3]]];

(* =====================================================================
   SECTION 4.  The plane-wave (Weyl) spectrum.

   PW2H0.nb verifies numerically that h_0(K R) is a superposition of plane
   waves with kz on the branch Im(kz) >= 0 -- propagating inside the light
   circle, evanescent outside.  The multipole generalisation replaces the
   constant angular factor by the harmonic evaluated at the PLANE-WAVE
   DIRECTION.

   THE SUBTLETY, MEASURED.  Outside the light circle kz is imaginary, so that
   direction is COMPLEX and its polar angle is complex.  Evaluating the
   harmonic through real angles silently discards the evanescent content: a
   probe done that way reproduced n = 0 to quadrature accuracy and then failed
   progressively for n = 1, 2, 3.  The solid harmonic Y[n,m,x,y,z] is
   polynomial in its arguments and takes the complex direction without
   complaint, which is exactly why the archive form is the one to use.
   ===================================================================== *)

Print["\n[4] the Weyl spectrum: branch and the complex direction"];

kzOf[K_, kx_, ky_] := Module[{s = Sqrt[K^2 - kx^2 - ky^2]}, If[Im[s] < 0, -s, s]];

Module[{K = 1 + 4/1000 I, prop, evan},
  prop = kzOf[K, 1/5, 1/10];
  evan = kzOf[K, 3, 1];
  Print["      inside the light circle, kz  = ", N[prop, 10], "  (propagating)"];
  Print["      outside, kz                  = ", N[evan, 10], "  (Im >= 0: decaying)"];
  Print["      solid harmonic at the complex direction is finite: ",
    FreeQ[N[Y[2, 1, 3, 1, evan/K]], DirectedInfinity | Indeterminate]]];

(* =====================================================================
   SECTION 5.  The plane-wave spectrum of the L- and N-type multipoles.

   For the L-type, u = grad phi is a Fourier multiplier, so the scalar Weyl
   identity gives the answer at once.  With phi = h_n(kP r) P_n(cos theta) the
   normalisations cancel and

       phihat(q) = P_n(kz/kP) / (2 Pi kP i^n kz) ,

   which at n = 0 is the classical Weyl identity.

   The N-type is not immediate.  u = curl curl (r psi) carries an explicit
   POSITION VECTOR, and a single plane-wave component of psi does not map to a
   single plane-wave component of u: writing
   N = grad[d_r(r psi)] + kS^2 r psi, both terms carry z-dependence beyond
   e^{i kz z} and those pieces must cancel between them.  Acting on one
   plane-wave component,

       curl curl (r e^{i k.r}) = e^{i k.r} [ kS^2 r - k (k.r) ] - 2 i k e^{i k.r},

   which displays the obstruction: the r-dependent terms are removed only by
   integrating by parts in k, not term by term.

   THE CLOSED FORM, which this section tests.  At each lateral wavenumber the
   N-type field is a pure SV plane wave with amplitude

       A(q) = P_n'(kz_dir/kS) (q/kS) / (2 Pi i^(n+1) kz) ,            (SVhat)

   with kz on the outgoing branch, kz_dir = -kz above the sphere, and the SV
   polarisation ehat = (-q, kz_dir cos psi, kz_dir sin psi)/kS in (z, x, y).

   Note (q/kS) = sin(theta_k) and P_n' sin(theta) = -dP_n/dtheta, so (SVhat) is
   dP_n/dtheta EVALUATED AT THE PLANE-WAVE DIRECTION -- exactly mirroring the
   real-space u_theta ~ dP_n/dtheta, just as the L-type spectrum mirrors P_n.
   That is the vector spherical harmonic result in the form this note needs.

   HOW IT IS ESTABLISHED.  Not by symbolic integration -- the kx, ky integral
   has no closed form Mathematica will produce.  Both sides are outgoing
   solutions of the elastic wave equation in the half-space beyond the sphere,
   so agreeing on a set of points there identifies them.  The test below
   evaluates the exact curl curl field and the plane-wave superposition at
   several points, above and below, for several n.

   The branch point at q = kS is removed by substitution rather than
   integrated through: q = kS sin(theta) below the light circle and
   q = kS cosh(t) above make q dq / kz equal to kS sin(theta) dtheta and
   -i kS cosh(t) dt respectively.  Driving a quadrature across the integrable
   1/kz singularity instead converges algebraically and plateaus -- measured, at
   5e-7 in the Python twin, however many nodes it was given.
   ===================================================================== *)

Print["\n[5] the plane-wave spectrum of the N-type (SV) multipole"];

(* Exact N-type field, m = 0, from the Cartesian construction of Section 2. *)
svExact[n_, kS_, p_] := Module[{x, y, z},
  N[uSV[n, 0, kS][x, y, z] /. Thread[{x, y, z} -> p], 30]];

(* The plane-wave superposition built on (SVhat), by the Sommerfeld path.
   The azimuthal integral is closed form for m = 0:
     Int dpsi e^{i q rho Cos[psi - phi]}          = 2 Pi BesselJ[0, q rho]
     Int dpsi Cos[psi] e^{...}                    = 2 Pi I Cos[phi] BesselJ[1, q rho]  *)
svSpectrum[n_, kS_, p_, qmax_] := Module[
  {z = p[[3]], x = p[[1]], y = p[[2]], rho, cph, sph, up, dP, piece, uz, urad},
  rho = Sqrt[x^2 + y^2];
  {cph, sph} = If[rho > 0, {x/rho, y/rho}, {1, 0}];
  up = z < 0;
  dP[u_] := D[LegendreP[n, t], t] /. t -> u;

  (* Each piece returns {u_z contribution, radial contribution}.  `base` is
     q dq / kz, already regular after the substitution. *)
  piece[qf_, kzf_, basef_, lo_, hi_, var_] := Module[{q, kz, kzd, amp, common},
    NIntegrate[
      q = qf; kz = kzf; kzd = If[up, -kz, kz];
      amp = dP[kzd/kS] (q/kS)/(2 Pi I^(n + 1));
      common = basef amp Exp[I kz Abs[z]];
      {2 Pi common (-q/kS) BesselJ[0, q rho],
       2 Pi I common (kzd/kS) BesselJ[1, q rho]},
      {var, lo, hi}, MaxRecursion -> 16, PrecisionGoal -> 14,
      AccuracyGoal -> 20, WorkingPrecision -> 30]];

  {uz, urad} =
    piece[kS Sin[th], kS Cos[th], kS Sin[th], 0, Pi/2, th] +
    piece[kS Cosh[tt], I kS Sinh[tt], -I kS Cosh[tt], 0, ArcCosh[qmax/kS], tt];
  (* (x, y, z) ordering, matching uSV.  The Python twin orders its axes
     (z, x, y); returning that ordering here instead is exactly the
     representation slip this note keeps warning about, and it showed up as a
     clean 5/3 -- a permutation, not a numerical error. *)
  {urad cph, urad sph, uz}];

Module[{kS = $kS, pts, ex, gu, worst = 0},
  pts = {{3, -2, -12}, {40, 25, -12}, {-8, 5, 14}};
  Print["      ", "n", "   ", "point", "                 relative difference"];
  Do[
    ex = svExact[n, kS, p];
    gu = svSpectrum[n, kS, {p[[1]], p[[2]], p[[3]]}, 6];
    (* The Cartesian ordering here is (x, y, z); svExact returns the same. *)
    worst = Max[worst, Max[Abs[gu - ex]]/Max[Abs[ex]]];
    Print["      ", n, "   ", p, "   ", N[Max[Abs[gu - ex]]/Max[Abs[ex]], 4]],
    {n, 1, 3}, {p, pts}];
  Print["      worst relative difference: ", N[worst, 4]]];

(* =====================================================================
   SECTION 6.  The plane-wave spectrum of the M-type (SH) multipole.

   The M-type is u = curl(r chi), one curl rather than two, and for an
   axisymmetric chi it is purely azimuthal:

       u_r = u_theta = 0 ,     u_phi = -z_n(kS r) dP_n/dtheta .

   Its spectrum carries the SAME angular function as the N-type -- dP_n/dtheta
   evaluated at the plane-wave direction -- on the SH polarisation
   ehat = (-Sin[psi], Cos[psi], 0):

       A(q) = P_n'(kz_dir/kS) (q/kS) / (2 Pi kS i^n kz) .              (SHhat)

   COMPARE (SVhat).  The two differ by a factor -i kS, which is NOT a phase:
   curl curl (r .) carries one more derivative than curl (r .), so potentials
   normalised alike give amplitudes differing by a power of kS.  Both were
   measured against their own exact field, and the constant came out
   independent of n in each case -- which is the evidence that the i^n and
   P_n' structure is right, since a wrong n-dependence would show up as a
   "constant" that drifted with order.

   The azimuthal reduction differs from the N-type because the polarisation is
   azimuthal rather than in the vertical plane:

       Int dpsi (-Sin[psi]) e^{i q rho Cos[psi - phi]} = -2 Pi I Sin[phi] BesselJ[1, q rho]
       Int dpsi ( Cos[psi]) e^{...}                    =  2 Pi I Cos[phi] BesselJ[1, q rho]

   so there is no J_0 term at all and u_z vanishes identically, as it must.
   ===================================================================== *)

Print["\n[6] the plane-wave spectrum of the M-type (SH) multipole"];

shExact[n_, kS_, p_] := Module[{x, y, z},
  N[uSH[n, 0, kS][x, y, z] /. Thread[{x, y, z} -> p], 30]];

shSpectrum[n_, kS_, p_, qmax_] := Module[
  {z = p[[3]], x = p[[1]], y = p[[2]], rho, cph, sph, up, dP, piece, ii},
  rho = Sqrt[x^2 + y^2];
  {cph, sph} = If[rho > 0, {x/rho, y/rho}, {1, 0}];
  up = z < 0;
  dP[u_] := D[LegendreP[n, t], t] /. t -> u;

  piece[qf_, kzf_, basef_, lo_, hi_, var_] :=
    NIntegrate[
      Module[{q = qf, kz = kzf, kzd, amp},
        kzd = If[up, -kz, kz];
        amp = dP[kzd/kS] (q/kS)/(2 Pi kS I^n);
        basef amp Exp[I kz Abs[z]] BesselJ[1, q rho]],
      {var, lo, hi}, MaxRecursion -> 16, PrecisionGoal -> 14,
      AccuracyGoal -> 20, WorkingPrecision -> 30];

  ii = piece[kS Sin[th], kS Cos[th], kS Sin[th], 0, Pi/2, th] +
       piece[kS Cosh[tt], I kS Sinh[tt], -I kS Cosh[tt], 0, ArcCosh[qmax/kS], tt];
  (* (x, y, z) ordering, matching uSH. *)
  {-2 Pi I sph ii, 2 Pi I cph ii, 0}];

Module[{kS = $kS, pts, ex, gu, worst = 0, worstz = 0},
  pts = {{3, -2, -12}, {40, 25, -12}, {-8, 5, 14}};
  Print["      ", "n", "   ", "point", "                 relative difference"];
  Do[
    ex = shExact[n, kS, p];
    gu = shSpectrum[n, kS, p, 6];
    worst = Max[worst, Max[Abs[gu - ex]]/Max[Abs[ex]]];
    worstz = Max[worstz, Abs[ex[[3]]]/Max[Abs[ex]]];
    Print["      ", n, "   ", p, "   ", N[Max[Abs[gu - ex]]/Max[Abs[ex]], 4]],
    {n, 1, 3}, {p, pts}];
  Print["      worst relative difference: ", N[worst, 4]];
  Print["      exact u_z / |u| (must vanish for a toroidal field): ", N[worstz, 4]]];

(* =====================================================================
   SECTION 7.  The channel spectra, general in m.

   Sections 5 and 6 gave the N- and M-type spectra at m = 0, where each shear
   family feeds a single transverse polarisation.  That is a degeneracy of
   m = 0, not the general structure.  Writing A for the potential's angular
   function evaluated at the PLANE-WAVE DIRECTION, the two transverse vector
   harmonics are

       grad_s A        =  dA/dtheta theta_hat + (1/sin) dA/dphi phi_hat
       rhat x grad_s A = -(1/sin) dA/dphi theta_hat + dA/dtheta phi_hat

   with theta_hat and phi_hat at that direction being e_SV and e_SH.  So each
   shear potential feeds BOTH polarisations, and the amplitudes are

       L-type :  i kP / D  *  A                        along e_P
       N-type :  i kS / D  * [ dA/dth e_SV + (1/sin) dA/dph e_SH ]
       M-type :    -1  / D  * [ -(1/sin) dA/dph e_SV + dA/dth e_SH ]

   with D = 2 Pi K i^n kz.  The three constants i kP, i kS and -1 are
   independent of n AND of m, which is the content worth testing: forcing a
   single angular function onto both polarisations instead produces a spurious
   n-dependence, and that is exactly how the structure was found.

   With u = kz_dir/K and s = q/K,

       m = 0:  A = P_n(u),             dA/dth = -s P_n'(u),
               (1/sin) dA/dph = 0
       m = 1:  A = -s P_n'(u) Cos[psi], dA/dth = (-u P_n'(u) + s^2 P_n''(u)) Cos[psi],
               (1/sin) dA/dph = P_n'(u) Sin[psi]

   MACHINE PRECISION AND VECTORISED ARRAYS.  The quadrature is tens of
   thousands of nodes per case.  Carrying exact rationals through it, or
   accumulating in a nested Do loop, makes it unusable -- a first attempt
   written that way produced nothing at all.  Everything below is numeric and
   built with Table/Total.
   ===================================================================== *)

Print["\n[7] the channel spectra, general in m"];

Needs["NumericalDifferentialEquationAnalysis`"];

Module[{kPn, kSn, nq = 400, npsi = 64, pts, polOf, angTriple, superpose,
        exactOf, worst = 0, fmt},

  (* Print does not format ScientificForm under wolframscript, so numbers are
     rendered to strings explicitly. *)
  fmt[x_] := ToString[ScientificForm[N[x], 3]];

  kPn = N[$kP]; kSn = N[$kS];
  pts = {{3., -2., -12.}, {-8., 5., 14.}};

  (* Unit polarisations at direction (kx, ky, kzd), Cartesian (x, y, z). *)
  polOf["P", kx_, ky_, kzd_, K_] := {kx, ky, kzd}/K;
  polOf["SV", kx_, ky_, kzd_, K_] := Module[{q = Sqrt[kx^2 + ky^2]},
     {kzd kx/q, kzd ky/q, -q}/K];
  polOf["SH", kx_, ky_, kzd_, K_] := Module[{q = Sqrt[kx^2 + ky^2]},
     {-ky/q, kx/q, 0.}];

  (* {A, dA/dtheta, (1/sin) dA/dphi} at the plane-wave direction. *)
  angTriple[n_, m_, kx_, ky_, kzd_, K_] := Module[
    {u = kzd/K, q = Sqrt[kx^2 + ky^2], s, cps, sps, p1, p2},
    s = q/K; cps = kx/q; sps = ky/q;
    p1 = D[LegendreP[n, t], t] /. t -> u;
    p2 = D[LegendreP[n, t], {t, 2}] /. t -> u;
    If[m == 0,
      {LegendreP[n, u], -s p1, 0.},
      {-s p1 cps, (-u p1 + s^2 p2) cps, p1 sps}]];

  (* The plane-wave superposition for one family, by the Sommerfeld path in q
     and a uniform rule in psi (smooth and periodic, so spectrally accurate). *)
  superpose[fam_, n_, m_, K_, p_] := Module[
    {x = p[[1]], y = p[[2]], z = p[[3]], up, gl, gw, th, wth, tt, wtt,
     tmax, pieces, tot},
    up = z < 0;
    (* GAUSS-LEGENDRE, not Chebyshev.  A first version used Chebyshev nodes
       reweighted by Sqrt[1-x^2], which is the natural rule for a 1/Sqrt
       weight and a poor one for the smooth integrand left after the
       Sommerfeld substitution: it held the check at 3e-4 while the same
       construction in double precision elsewhere reaches 1e-14. *)
    tmax = ArcCosh[6./K];
    {th, wth} = Transpose[GaussianQuadratureWeights[nq, 0, Pi/2]];
    {tt, wtt} = Transpose[GaussianQuadratureWeights[nq, 0, tmax]];

    pieces = {
      {K Sin[th], K Cos[th] + 0. I, wth K Sin[th]},
      {K Cosh[tt], I K Sinh[tt], wtt (-I) K Cosh[tt]}};

    tot = {0., 0., 0.};
    Do[
      Module[{q = pc[[1]], kz = pc[[2]], base = pc[[3]], kzd, psi, wps},
        kzd = If[up, -kz, kz];
        psi = N[Table[2 Pi (j - 1)/npsi, {j, npsi}]];
        wps = 2 Pi/npsi;
        tot = tot + Total[Flatten[
          Table[
            Module[{kx = q[[i]] Cos[psi[[j]]], ky = q[[i]] Sin[psi[[j]]],
                    a, dth, dph, common},
              {a, dth, dph} = angTriple[n, m, kx, ky, kzd[[i]], K];
              common = base[[i]] wps/(2 Pi K I^n) *
                 Exp[I (kx x + ky y + kz[[i]] Abs[z])];
              Which[
                fam === "P",
                  I K common a polOf["P", kx, ky, kzd[[i]], K],
                fam === "SV",
                  I K common (dth polOf["SV", kx, ky, kzd[[i]], K]
                              + dph polOf["SH", kx, ky, kzd[[i]], K]),
                True,
                  -common (-dph polOf["SV", kx, ky, kzd[[i]], K]
                           + dth polOf["SH", kx, ky, kzd[[i]], K])]],
            {i, nq}, {j, npsi}], 1]]],
      {pc, pieces}];
    tot];

  exactOf[fam_, n_, m_, K_, p_] := Module[{x, y, z, u},
    u = Switch[fam, "P", uP[n, m, K][x, y, z], "SV", uSV[n, m, K][x, y, z],
         "SH", uSH[n, m, K][x, y, z]];
    N[u /. Thread[{x, y, z} -> p]]];

  Print["      family  m  n     relative difference (both points)"];
  Do[
    Module[{K, ex, gu, rel},
      K = If[fam === "P", kPn, kSn];
      rel = Table[
        ex = exactOf[fam, n, m, K, p];
        gu = superpose[fam, n, m, K, p];
        Max[Abs[gu - ex]]/Max[Abs[ex]],
        {p, pts}];
      worst = Max[worst, Max[rel]];
      Print["      ", fam, "\t ", m, "  ", n, "    ",
        fmt[rel[[1]]], "   ", fmt[rel[[2]]]]],
    {fam, {"P", "SV", "SH"}}, {m, {0, 1}}, {n, 1, 3}];

  Print["      worst over all 18 cases: ", fmt[worst]]];

(* =====================================================================
   SECTION 8.  Reference export for the Python implementation.
   ===================================================================== *)

Print["\n[8] reference export"];
Module[{x, y, z, out = {}, u, t3, pts, p},
  pts = {{37/100, -61/100, 83/100} 3, {110/100, 40/100, -230/100},
         {-90/100, 170/100, 110/100}, {5/100, 2/100, 4}};
  Do[
    u = Switch[fam, "P", uP[n, m, $kP][x, y, z],
      "SV", uSV[n, m, $kS][x, y, z], "SH", uSH[n, m, $kS][x, y, z]];
    t3 = tau3Of[u, $lam, $mu, x, y, z];
    Do[p = pts[[ip]];
      AppendTo[out, <|"family" -> fam, "n" -> n, "m" -> m, "point" -> N[p, 20],
        "u" -> (ReIm /@ N[u /. Thread[{x, y, z} -> p], 30]),
        "tau3" -> (ReIm /@ N[t3 /. Thread[{x, y, z} -> p], 30])|>],
      {ip, Length[pts]}],
    {fam, {"P", "SV", "SH"}}, {n, 1, 3}, {m, 0, 1}];
  Export["Mathematica/MieSphericalWaves_reference.json",
    <|"note" -> "Cartesian (x,y,z), wave along +z. Python orders axes (z,x,y).",
      "lam" -> N[$lam, 20], "mu" -> N[$mu, 20], "rho" -> N[$rho, 20],
      "omega" -> N[$om, 20], "kP" -> N[$kP, 20], "kS" -> N[$kS, 20],
      "cases" -> out|>];
  Print["      wrote ", Length[out], " cases"]];

Print["\n=== done ==="];
