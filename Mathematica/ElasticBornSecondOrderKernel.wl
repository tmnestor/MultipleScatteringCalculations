#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   ElasticBornSecondOrderKernel.wl

   The spectral kernel of SECOND-ORDER elastic Born scattering, per lateral
   wavenumber q, for a P wave at normal incidence scattered back into P.

   WHAT IT IS FOR.  A laterally periodic array of spheres and the isolated
   sphere share their first-order Born response exactly (Poisson summation),
   so their difference begins at second order: double scattering between a
   sphere and its images.  Per lateral wavenumber q that is

       f2(q) = (1/(4 Pi rho alpha^2)) Int dz Int dz' e^{i kP (z + z')}
                 F(q, z) F(q, z') K(q; z - z') ,

   with F(q, z) the lateral transform of the sphere's cross-section at depth
   z; the array takes (1/A) Sum_g over diffraction orders, the isolated sphere
   Int d^2q / (2 Pi)^2.  This file derives K.

   CONVENTIONS, those of ElasticBornBackscatter.wl (validated against Mie in
   magnitude there and in complex value since): time e^{-i w t}; (x, y, z)
   with z the depth and the incident P wave u0 = zhat e^{i kP z}; far-field
   observation along n = -zhat; first-order weak form
       u1_i = Int [ w^2 drho G_ij u0_j - dc_jklm d'_k G_ij d'_m u0_l ] dV' ,
   and far-field amplitude f = (1/(4 Pi rho alpha^2)) Int e^{-i ksc.x}
       [ w^2 drho n.u + i dc_jklm n_j ksc_k d_m u_l ] dV .

   SPECTRAL FORM.  With G(x - x') = Int d^2q/(2 Pi)^2 e^{i q.(rho - rho')}
   Ghat(q; zeta), zeta = z - z', both vertex derivatives become
   D = (i qx, i qy, p) with p = d/dzeta acting on Ghat (d'/dz' = -p enters with
   the weak form's minus sign).  Ghat = [kS^2 gS I + D D (gS - gP)]/(rho w^2),
   gK = (i/(2 kapK)) e^{i kapK |zeta|}, kapK = Sqrt[kK^2 - q^2].

   THE DISTRIBUTIONAL DERIVATIVES.  p^n gK has a regular part
   (i/(2 kap)) (i kap s)^n e^{i kap |zeta|}, s = Sign[zeta], and contact parts:
       n = 2 :  - delta ,   n = 3 : - delta' ,   n = 4 : kap^2 delta - delta'' .
   The kernel is a sum c_{K,n}(q) p^n gK; its delta' and delta'' coefficients
   must cancel between S and P for the Born series to exist at all, and that
   is CHECKED here, not assumed.  The delta coefficient is the spectral form of
   the Eshelby self-term.

   OUTPUT.  ElasticBornSecondOrderKernel.json: at several (qx, qy) the regular
   coefficients A_{K,s} (kernel = Sum_K A_{K,s} e^{i kapK |zeta|} for sign s)
   and the contact coefficient, for the Python twin to reproduce.

   Run:
       wolframscript -file Mathematica/ElasticBornSecondOrderKernel.wl
   ===================================================================== *)

cten[lam_, mu_] := Table[
   lam KroneckerDelta[i, j] KroneckerDelta[k, l]
    + mu (KroneckerDelta[i, k] KroneckerDelta[j, l] + KroneckerDelta[i, l] KroneckerDelta[j, k]),
   {i, 3}, {j, 3}, {k, 3}, {l, 3}];

(* The kernel as a polynomial in p acting on the symbols gS, gP. *)
kernelPoly[qx_, qy_, kP_, kS_, rho_, w_, drho_, dlam_, dmu_] := Module[
  {dd, gh, dc, e, n, kin, ksc, u1, wvec},
  dd = {I qx, I qy, p};
  gh = (kS^2 gS IdentityMatrix[3] + Outer[Times, dd, dd] (gS - gP))/(rho w^2);
  dc = cten[dlam, dmu];
  e = {0, 0, 1}; n = {0, 0, -1};
  kin = kP e; ksc = kP n;
  (* first vertex: u1hat_i = w^2 drho Ghat_ij e_j + dc_jklm D_k Ghat_ij (i kin_m) e_l *)
  u1 = Table[
    w^2 drho Sum[gh[[i, j]] e[[j]], {j, 3}]
     + Sum[dc[[j, k, l, m]] dd[[k]] gh[[i, j]] I kin[[m]] e[[l]], {j, 3}, {k, 3}, {l, 3}, {m, 3}],
    {i, 3}];
  (* second vertex, row: w^2 drho n_l + i dc_jklm n_j ksc_k D_m acting on u1_l *)
  Expand[Sum[w^2 drho n[[l]] u1[[l]], {l, 3}]
    + Sum[I dc[[j, k, l, m]] n[[j]] ksc[[k]] dd[[m]] u1[[l]], {j, 3}, {k, 3}, {l, 3}, {m, 3}]]];

(* Split into regular coefficients and contact coefficients. *)
decompose[poly_, kP_, kS_, qx_, qy_] := Module[
  {kap, coef, reg, c0, c1, c2, sgn},
  kap = <|gP -> Sqrt[kP^2 - qx^2 - qy^2], gS -> Sqrt[kS^2 - qx^2 - qy^2]|>;
  coef[g_, nn_] := Coefficient[Coefficient[poly, g], p, nn];
  (* regular part, for zeta of sign sgn, as coefficient of e^{i kap |zeta|} *)
  reg[g_, sg_] := Sum[coef[g, nn] (I/(2 kap[g])) (I kap[g] sg)^nn, {nn, 0, 4}];
  c0 = Sum[-coef[g, 2] + coef[g, 4] kap[g]^2, {g, {gS, gP}}];
  c1 = Sum[-coef[g, 3], {g, {gS, gP}}];
  c2 = Sum[-coef[g, 4], {g, {gS, gP}}];
  <|"regP+" -> reg[gP, 1], "regP-" -> reg[gP, -1], "regS+" -> reg[gS, 1], "regS-" -> reg[gS, -1],
    "c0" -> c0, "c1" -> c1, "c2" -> c2|>];

Print["[1] the derivative-contact cancellation, symbolically, for general q and contrasts"];
Module[{poly, dec},
  poly = kernelPoly[qx, qy, kP, kS, rho, w, drho, dlam, dmu];
  Print["    highest power of p in the kernel: ", Exponent[poly, p]];
  dec = decompose[poly, kP, kS, qx, qy];
  Print["    delta'  coefficient: ", Simplify[dec["c1"]]];
  Print["    delta'' coefficient: ", Simplify[dec["c2"]]];
  Print["    delta   coefficient: ", Simplify[dec["c0"]]];
  okContact = Simplify[dec["c1"]] === 0 && Simplify[dec["c2"]] === 0;
  Print["    ", If[okContact, "PASS", "FAIL"], ": only a delta (no delta', delta'') survives"]];

(* Numeric dump at the gate's background and first-order contrasts. *)
rho0 = 2500; al0 = 5000; be0 = 3000; w0 = 60;
mu0 = rho0 be0^2; lam0 = rho0 al0^2 - 2 mu0;
kP0 = w0/al0; kS0 = w0/be0;
(* first-order contrasts per unit eps for alpha, beta, rho all scaled by 1+eps *)
drho0 = rho0; dlam0 = 3 lam0; dmu0 = 3 mu0;

qs = {{0, 0}, {0.004, 0}, {0.0105, 0}, {0.0105, 0.0105}, {0.015, 0.006}, {0.05, -0.02}, {0.3, 0.1}};
reim[z_] := {N[Re[z], 18], N[Im[z], 18]};
rows = Map[Function[qq, Module[{poly, dec},
     poly = kernelPoly[qq[[1]], qq[[2]], kP0, kS0, rho0, w0, drho0, dlam0, dmu0];
     dec = decompose[poly, kP0, kS0, qq[[1]], qq[[2]]];
     <|"q" -> N[qq], "regP+" -> reim[dec["regP+"]], "regP-" -> reim[dec["regP-"]],
       "regS+" -> reim[dec["regS+"]], "regS-" -> reim[dec["regS-"]], "c0" -> reim[dec["c0"]]|>]], qs];

(* Rotation invariance about z: the kernel should depend on |q| only. *)
Module[{a1, a2},
  a1 = decompose[kernelPoly[0.0105, 0, kP0, kS0, rho0, w0, drho0, dlam0, dmu0], kP0, kS0, 0.0105, 0];
  a2 = decompose[kernelPoly[0.0105 Cos[0.7], 0.0105 Sin[0.7], kP0, kS0, rho0, w0, drho0, dlam0, dmu0],
     kP0, kS0, 0.0105 Cos[0.7], 0.0105 Sin[0.7]];
  dev = Max[Abs[N[Values[a1] - Values[a2]]]]/Max[Abs[N[Values[a1]]]];
  Print["[2] rotation invariance about z: relative change on rotating q by 0.7 rad: ", ToString[dev, FortranForm]];
  Print["    ", If[dev < 1*^-12, "PASS", "FAIL"]]];

Export[FileNameJoin[{DirectoryName[$InputFileName], "ElasticBornSecondOrderKernel.json"}],
  <|"background" -> <|"rho" -> rho0, "alpha" -> al0, "beta" -> be0, "omega" -> w0|>,
    "contrast_per_eps" -> <|"drho" -> drho0, "dlam" -> N[dlam0], "dmu" -> N[dmu0]|>,
    "rows" -> rows|>, "JSON"];
Print["wrote ElasticBornSecondOrderKernel.json"];
Exit[If[okContact && dev < 1*^-12, 0, 1]];
