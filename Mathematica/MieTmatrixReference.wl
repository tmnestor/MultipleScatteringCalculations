#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   MieTmatrixReference.wl

   High-precision per-order Mie T-matrices for the elastic sphere, the arbiter
   for mie_tmatrix_psv / mie_tmatrix_sh in cubic_scattering/sphere_scattering.py.

   WHY HIGH PRECISION.  The direct solve imposes continuity of
   (u_r, u_theta, sigma_rr, sigma_rtheta) with the scattered AND the interior
   amplitudes as unknowns.  Its columns carry j_n inside and h_n outside, which
   separate by thirty orders of magnitude by n = 12 at k_S a = 2.4, and in SI
   units its traction rows sit 1e9 above its displacement rows.  In machine
   precision that system has a condition number of 1e33 at n = 12.  At fifty
   digits it is solved exactly for every purpose here, so it arbitrates the
   machine-precision solve without sharing its rounding.  (Measured against it,
   the machine-precision 4x4 is right to 1e-13: the condition number is
   ill-scaling, which partial pivoting does not feel.)

   WHY THE STRESSES ARE DERIVED HERE.  The Python radial formulas for
   sigma_rr and sigma_rtheta were simplified by hand with the Bessel equation.
   Here u = grad(phi) + curl curl(r psi) is formed from the potentials in
   spherical coordinates and differentiated symbolically, so the stresses do not
   share that simplification.

   CONVENTIONS, matched to the Python.  Potentials phi = z_n(k_P r) P_n(cos t),
   psi = z_n(k_S r) P_n(cos t); z = j inside and for the incident wave, h^(1)
   for the scattered one; time e^{-i w t}.  T[n] maps INCIDENT potential
   coefficients (P, S) to SCATTERED ones (P, S); the SH scalar maps the incident
   toroidal coefficient to the scattered one.

   OUTPUT.  MieTmatrixReference.json: for each parameter set and each n, the
   2x2 P-SV T (n >= 1), the 1x1 P T (n = 0) and the SH scalar, as re/im pairs.

   Run:
       wolframscript -file Mathematica/MieTmatrixReference.wl
   ===================================================================== *)

$wp = 50;

(* Displacement (u_r, u_t) and traction (s_rr, s_rt) at r of the potential
   z_n(k r) P_n(cos t), for the P family (grad phi) and the SV family
   (curl curl r psi), as the coefficient of P_n (for u_r, s_rr) and of
   dP_n/dt (for u_t, s_rt). *)
radialZ[n_, x_, "j"] := SphericalBesselJ[n, x];
radialZ[n_, x_, "h"] := SphericalHankelH1[n, x];

fieldsOf[fam_, n_, k_, lam_, mu_, type_, a_] := Module[
  {r, t, f, ur, ut, divu, srr, srt, pn, dpn, vals},
  f = radialZ[n, k r, type];
  pn = LegendreP[n, Cos[t]];
  If[fam === "P",
    ur = D[f pn, r];
    ut = D[f pn, t]/r,
    (* curl curl (r_vec psi) for psi = f(r) P_n: u_r = n(n+1) f/r P_n,
       u_t = (1/r) d/dr (r f) dP_n/dt *)
    ur = n (n + 1) f/r pn;
    ut = (1/r) D[r f, r] D[pn, t]];
  divu = (1/r^2) D[r^2 ur, r] + (1/(r Sin[t])) D[Sin[t] ut, t];
  srr = lam divu + 2 mu D[ur, r];
  srt = mu (D[ut, r] - ut/r + (1/r) D[ur, t]);
  dpn = D[pn, t];
  (* Read the radial coefficients at an angle where P_n and dP_n are nonzero.
     The monopole has no tangential part (dP_0 = 0) and needs only u_r, s_rr. *)
  vals = If[n == 0, {ur/pn, 0, srr/pn, 0}, {ur/pn, ut/dpn, srr/pn, srt/dpn}] /. t -> 7/10;
  N[vals /. r -> a, $wp]];

tmatrices[par_] := Module[
  {om, a, al, be, rho, s, lamO, muO, rhoI, lamI, muI, alI, beI, kPo, kSo, kPi, kSi,
   res = {}, M, rhs, sol, fPs, fSs, fPi, fSi, fPj, fSj, tsh, hS, jSi, jSo},
  {om, a, al, be, rho, s} = par;
  lamO = rho (al^2 - 2 be^2); muO = rho be^2;
  (* all of alpha, beta, rho scaled by s inside: moduli by s^3 *)
  rhoI = s rho; lamI = s^3 lamO; muI = s^3 muO;
  alI = Sqrt[(lamI + 2 muI)/rhoI]; beI = Sqrt[muI/rhoI];
  kPo = om/al; kSo = om/be; kPi = om/alI; kSi = om/beI;
  Table[
    If[n == 0,
      (* monopole: P only, rows u_r and s_rr *)
      fPs = fieldsOf["P", 0, kPo, lamO, muO, "h", a];
      fPi = fieldsOf["P", 0, kPi, lamI, muI, "j", a];
      fPj = fieldsOf["P", 0, kPo, lamO, muO, "j", a];
      M = {{fPs[[1]], -fPi[[1]]}, {fPs[[3]], -fPi[[3]]}};
      sol = LinearSolve[M, -{fPj[[1]], fPj[[3]]}];
      <|"n" -> 0, "Tpsv" -> {{sol[[1]], 0}, {0, 0}}, "Tsh" -> 0|>,
      fPs = fieldsOf["P", n, kPo, lamO, muO, "h", a];
      fSs = fieldsOf["S", n, kSo, lamO, muO, "h", a];
      fPi = fieldsOf["P", n, kPi, lamI, muI, "j", a];
      fSi = fieldsOf["S", n, kSi, lamI, muI, "j", a];
      fPj = fieldsOf["P", n, kPo, lamO, muO, "j", a];
      fSj = fieldsOf["S", n, kSo, lamO, muO, "j", a];
      M = Transpose[{fPs, fSs, -fPi, -fSi}];
      sol = LinearSolve[M, -Transpose[{fPj, fSj}]];
      (* SH: u_phi = z_n(k r) (a common angular factor), s_rphi = mu (d/dr - 1/r) u_phi *)
      hS[k_, type_, m_] := Module[{r},
        N[{radialZ[n, k r, type], m (D[radialZ[n, k r, type], r] - radialZ[n, k r, type]/r)} /. r -> a, $wp]];
      tsh = LinearSolve[Transpose[{hS[kSo, "h", muO], -hS[kSi, "j", muI]}], -hS[kSo, "j", muO]][[1]];
      <|"n" -> n, "Tpsv" -> sol[[1 ;; 2]], "Tsh" -> tsh|>],
    {n, 0, 25}]];

(* Parameter sets, exact rationals: {omega, a, alpha, beta, rho, s}. *)
sets = <|
  "gate" -> {60, 120, 5000, 3000, 2500, 11/10},
  "soft" -> {60, 120, 5000, 3000, 2500, 7/10},
  "lowka" -> {6, 120, 5000, 3000, 2500, 11/10}|>;

reim[z_] := {N[Re[z], 20], N[Im[z], 20]};
out = Association[KeyValueMap[#1 -> Map[
      <|"n" -> #["n"], "Tpsv" -> Map[reim, #["Tpsv"], {2}], "Tsh" -> reim[#["Tsh"]]|> &,
      tmatrices[#2]] &, sets]];

(* Self-check before writing: flux unitarity of the P-SV S-matrix per order,
   S = W (I + 2 T) W^-1 with W = diag(sqrt(alpha), sqrt(beta n(n+1))), and the
   SH element 1 + 2 Tsh, for the lossless spheres here. *)
Do[Module[{par = sets[key], rows = tmatrices[sets[key]], worst = 0},
   Do[Module[{n = row["n"], W, S},
      If[n >= 1,
        W = DiagonalMatrix[{Sqrt[par[[3]]], Sqrt[par[[4]] n (n + 1)]}];
        S = W . (IdentityMatrix[2] + 2 row["Tpsv"]) . Inverse[W];
        worst = Max[worst, Norm[ConjugateTranspose[S] . S - IdentityMatrix[2]],
          Abs[Abs[1 + 2 row["Tsh"]] - 1]]]], {row, rows}];
   Print[key, ": worst unitarity defect over n = 1..25: ", ToString[N[worst, 3], FortranForm]]],
  {key, Keys[sets]}];

Export[FileNameJoin[{DirectoryName[$InputFileName], "MieTmatrixReference.json"}],
  <|"conventions" -> "T maps incident potential coefficients (P,S) to scattered (P,S); SH scalar likewise",
    "sets" -> Map[<|"omega" -> #[[1]], "radius" -> #[[2]], "alpha" -> #[[3]], "beta" -> #[[4]],
        "rho" -> #[[5]], "scale" -> N[#[[6]], 20]|> &, sets],
    "tmatrices" -> out|>, "JSON"];
Print["wrote MieTmatrixReference.json"];
