#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   BornSecondOrderStatic.wl

   The STATIC self-term of second-order elastic Born scattering by a ball,
   exactly -- the piece that singularity subtraction adds back in real space.

   WHAT IT IS.  For a P wave at normal incidence scattered back into P, with the
   Kelvin (static) Green's tensor between the two Born vertices:

     u1_i = w^2 drho Phi_iz + i k (dlam d_j Phi_ij + 2 dmu d_z Phi_iz),
     Phi_ij(r) = Int_ball Gst_ij(r - r') e^{i k z'} dV',
     S = Int_ball e^{i k z} [ -w^2 drho u1_z + i k (dlam div u1 + 2 dmu d_z u1_z) ] dV ,

   the weak-form vertices of ElasticBornBackscatter.wl (replacing u1 by the
   incident wave returns its first-order bracket).  Gst = delta_ij/(4 pi mu r)
   - cB d_i d_j r, cB = (1/mu - 1/(lam + 2 mu))/(8 pi): Kelvin's tensor.

   HOW, EXACTLY.  e^{i k z'} is a Taylor polynomial (k a = 1.44; 36 terms leave
   1e-30).  For a polynomial density on a ball both potentials are polynomials
   inside it -- Eshelby's polynomial property:
     N_j = Int_ball z'^j / (4 pi |r - r'|),   M_j = Int_ball z'^j |r - r'| ,
   from the Legendre expansions
     1/|r-r'| = Sum r<^n / r>^(n+1) P_n ,
     |r-r'|   = Sum [ r<^(n+2)/((2n+3) r>^(n+1)) - r<^n/((2n-1) r>^(n-1)) ] P_n ,
   with elementary radial integrals.  Then Phi_ij = (delta_ij/mu) N - cB d_i d_j M,
   everything is a polynomial, and the last integral over the ball is exact.

   CHECKS: Poisson (Lap N = -z^j); Lap M = 8 pi N; N against a direct numerical
   integral; the static Navier equation on Phi.  Output: S to 30 digits, in
   BornSecondOrderStatic.json.

   Run:  wolframscript -file Mathematica/BornSecondOrderStatic.wl
   ===================================================================== *)

$wp = 40;
rho0 = 2500; al0 = 5000; be0 = 3000; w0 = 60; aR = 120;
mu0 = rho0 be0^2; lam0 = rho0 al0^2 - 2 mu0;
kk = w0/al0;
drho = rho0; dlam = 3 lam0; dmu = 3 mu0;       (* first-order contrasts per unit eps *)
cB = (1/mu0 - 1/(lam0 + 2 mu0))/(8 Pi);
J = 36;

rr = Sqrt[x^2 + y^2 + z^2];
(* r^m P_n(cos theta) as a polynomial, m >= n, m - n even *)
solidPoly[m_, n_] := Expand[(x^2 + y^2 + z^2)^((m - n)/2) Expand[rr^n LegendreP[n, z/rr]]];
(* cos^j theta = Sum_n a[j, n] P_n(cos theta) *)
aCoef[j_, n_] := aCoef[j, n] = (2 n + 1)/2 Integrate[u^j LegendreP[n, u], {u, -1, 1}];

nPot[j_] := nPot[j] = Expand[Sum[
     aCoef[j, n]/(2 n + 1) (solidPoly[j + 2, n]/(j + n + 3)
        + (aR^(j - n + 2) solidPoly[n, n] - solidPoly[j + 2, n])/(j - n + 2)),
     {n, Mod[j, 2], j, 2}]];

mPot[j_] := mPot[j] = Expand[Sum[
     aCoef[j, n] 4 Pi/(2 n + 1) (
        solidPoly[j + 4, n]/((2 n + 3) (j + n + 5))
      - solidPoly[j + 4, n]/((2 n - 1) (j + n + 3))
      + (aR^(j - n + 2) solidPoly[n + 2, n] - solidPoly[j + 4, n])/((2 n + 3) (j - n + 2))
      - (aR^(j - n + 4) solidPoly[n, n] - solidPoly[j + 4, n])/((2 n - 1) (j - n + 4))),
     {n, Mod[j, 2], j, 2}]];

lap[f_] := D[f, {x, 2}] + D[f, {y, 2}] + D[f, {z, 2}];

Print["[1] the ball potentials of a monomial density, against their equations"];
Module[{worstP = 0, worstM = 0},
  Do[
    worstP = Max[worstP, Abs[N[(lap[nPot[j]] + z^j) /. {x -> 13/10, y -> -7, z -> 41}, 30]]];
    worstM = Max[worstM, Abs[N[(lap[mPot[j]] - 8 Pi nPot[j]) /. {x -> 13/10, y -> -7, z -> 41}, 30]]],
    {j, 0, 7}];
  Print["    max |Lap N + z^j| = ", ToString[N[worstP, 3], FortranForm],
    "    max |Lap M - 8 pi N| = ", ToString[N[worstM, 3], FortranForm]];
  okLap = worstP < 10^-20 && worstM < 10^-15];

Print["[2] N against a direct numerical integral (j = 3, one interior point)"];
Module[{p = {20, -30, 45}, direct, formula},
  direct = NIntegrate[
     (rp Cos[tp])^3 rp^2 Sin[tp] /(4 Pi Sqrt[(p[[1]] - rp Sin[tp] Cos[pp])^2 + (p[[2]] - rp Sin[tp] Sin[pp])^2
        + (p[[3]] - rp Cos[tp])^2]),
     {rp, 0, aR}, {tp, 0, Pi}, {pp, 0, 2 Pi}, PrecisionGoal -> 10, AccuracyGoal -> 30,
     MaxRecursion -> 20, WorkingPrecision -> 30];
  formula = N[nPot[3] /. {x -> p[[1]], y -> p[[2]], z -> p[[3]]}, 30];
  Print["    relative difference ", ToString[N[Abs[direct - formula]/Abs[formula], 3], FortranForm]];
  okDirect = Abs[direct - formula]/Abs[formula] < 10^-8];

(* Taylor-expanded potentials of e^{i k z'} *)
nE = Sum[(I kk)^j/j! nPot[j], {j, 0, J}];
mE = Sum[(I kk)^j/j! mPot[j], {j, 0, J}];
xs = {x, y, z};
phi = Table[KroneckerDelta[i, j] nE/mu0 - cB D[mE, xs[[i]], xs[[j]]], {i, 3}, {j, 3}];

Print["[3] static Navier on Phi_{., z}: mu Lap + (lam+mu) grad div = -e^{ikz} zhat (Taylor)"];
Module[{col, nav, pt = {17, -23, 31}, want},
  col = phi[[All, 3]];
  nav = mu0 Map[lap, col] + (lam0 + mu0) Table[D[Sum[D[col[[i]], xs[[i]]], {i, 3}], xs[[j]]], {j, 3}];
  want = {0, 0, -Exp[I kk pt[[3]]]};
  resid = N[Norm[(nav /. Thread[xs -> pt]) - want], 30];
  Print["    residual ", ToString[N[resid, 3], FortranForm]];
  okNav = resid < 10^-20];

u1 = Table[w0^2 drho phi[[i, 3]] + I kk (dlam Sum[D[phi[[i, j]], xs[[j]]], {j, 3}] + 2 dmu D[phi[[i, 3]], z]), {i, 3}];
integrand = Expand[-w0^2 drho u1[[3]] + I kk (dlam Sum[D[u1[[i]], xs[[i]]], {i, 3}] + 2 dmu D[u1[[3]], z])];

(* Int_ball e^{ikz} poly dV: cylindrical, poly is a polynomial in (x^2 + y^2, z) *)
ballInt[poly_] := Module[{pc},
  pc = Expand[poly /. {x -> s Cos[f], y -> s Sin[f]}];
  pc = Integrate[pc, {f, 0, 2 Pi}];
  Integrate[Exp[I kk z] Integrate[Expand[pc s], {s, 0, Sqrt[aR^2 - z^2]}], {z, -aR, aR}]];

sStatic = N[ballInt[integrand], 30];
Print["[4] S_static = ", sStatic];

ok = okLap && okDirect && okNav;
Export[FileNameJoin[{DirectoryName[$InputFileName], "BornSecondOrderStatic.json"}],
  <|"S_static" -> {N[Re[sStatic], 25], N[Im[sStatic], 25]}, "taylor_terms" -> J,
    "checks" -> <|"laplace" -> okLap, "direct" -> okDirect, "navier" -> okNav|>|>, "JSON"];
Print[If[ok, "PASS", "FAIL"], "  wrote BornSecondOrderStatic.json"];
Exit[If[ok, 0, 1]];
