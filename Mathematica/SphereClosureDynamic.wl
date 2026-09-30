#!/usr/bin/env wolframscript
(* ============================================================================
   SphereClosureDynamic.wl  --  the uniform-field closure of a sphere against
   exact Mie, at O((ka)^2), in closed form.

   WHY.  scripts/gate_sphere_closure_vs_mie.py measured that the single-site
   closure (continuum-limit paper, section 6), applied to a ball with its
   DYNAMIC moments, departs from the exact Mie concentration E_n = a_n/a_n^Born
   at exactly second order in k a and first order in the contrast.  This
   notebook derives that departure's coefficient.

   UNITS.  a = mu0 = rho0 = 1, lam0 = lambda0/mu0, w = k_S a (so omega = w,
   k_P = w/Sqrt[lam0 + 2]); contrasts dlam, dmu, drho in the same units.  These
   are the conventions of MieAsymptotic.wl, whose cached Cramer solution
   (MieAsymptoticRaw.wl) supplies the Mie side.

   THE CLOSURE.  Uniform internal field; the self terms are the ball's moments
     g        : Int_ball G_ij dV = g delta_ij,
     M_in,pk  = Int_ball d_p d_k G_in dV = Oint_{|x|=1} n_p d_k G_in dA
   (the surface route: the delta term is inside the surface integral).  The
   channels are the eigenvalues of the first-gradient block I - M.dc (A1g for
   the monopole, T2g for the quadrupole) and 1 - w^2 drho g for the dipole;
   the closure's concentration is their inverse.

   FIRST ORDER IN THE CONTRAST.  With contrast eps c,
     closure  1/(1 - eps sigma_n(w)) = 1 + eps sigma_n(w) + ...
     Mie      a_n/a_n^Born          = 1 + eps e_n(w)     + ...
   and the departure closure/Mie - 1 = eps (sigma_n - e_n) + O(eps^2).  Each
   channel is driven by one contrast (n=0: dlam, n=1: drho, n=2: dmu), as in the
   Python gate.

   CHECKS: [1] the static channels are Eshelby's (sigma_n(0) = e_n(0)), both
   sides; [2] the departure starts at w^2 (no w^0, w^1 term); then the closed
   forms, and their values at the validated background for the Python gate.
   ============================================================================ *)

oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
Print["==== SphereClosureDynamic :: the sphere's closure against Mie at O((ka)^2) ===="];
Get[DirectoryName[$InputFileName] <> "MieAsymptoticRaw.wl"];

(* ---------------------------------------------------------------------------
   the closure side: the dynamic ball moments in closed form
   --------------------------------------------------------------------------- *)
kS = w; kP = w/Sqrt[lam0 + 2];
f[k_, r_] := Exp[I k r]/r;
hS[r_] := f[kS, r] - f[kP, r];
(* d_k G_in = (1/4Pi)[delta_in x_k f_S' + w^-2 (da xxx + (a/r)(d_ik x_n + d_nk x_i - 2 xxx) + db x_k d_in)],
   h = f_S - f_P, a = h'' - h'/r, b = h'/r; on |x| = 1, x = n *)
Module[{r, h1, h2, h3, a, b, da, db, fs1},
  h1 = D[hS[r], r]; h2 = D[hS[r], {r, 2}]; h3 = D[hS[r], {r, 3}];
  fs1 = D[f[kS, r], r];
  a = h2 - h1/r; b = h1/r; da = D[a, r]; db = D[b, r];
  {c1, c2, c3} = {fs1 + db/w^2, (a/r)/w^2, (da - 2 a/r)/w^2}/(4 Pi) /. r -> 1];
(* angular averages <n_p n_k> = d/3, <n_p n_i n_n n_k> = (dd + dd + dd)/15, sphere area 4 Pi *)
dd[i_, j_] := KroneckerDelta[i, j];
av2[p_, k_] := dd[p, k]/3;
av4[p_, i_, n_, k_] := (dd[p, i] dd[n, k] + dd[p, n] dd[i, k] + dd[p, k] dd[i, n])/15;
M[i_, n_, p_, k_] := 4 Pi (c1 dd[i, n] av2[p, k] + c2 (dd[i, k] av2[p, n] + dd[n, k] av2[p, i])
     + c3 av4[p, i, n, k]);
gBall = Module[{R}, R[k_] := 4 Pi (Exp[I k] (1/(I k) + 1/k^2) - 1/k^2);
   (2 R[kS]/(4 Pi) + R[kP]/(4 Pi (lam0 + 2)))/3];

dc[i_, j_, k_, l_, dl_, dm_] := dl dd[i, j] dd[k, l] + dm (dd[i, k] dd[j, l] + dd[i, l] dd[j, k]);
(* channel eigenvalues of I - M.dc, projected as in CubeA22Block.wl: row (p,i), column (r,j) *)
a22[dl_, dm_] := Table[dd[p, r] dd[i, j] - Sum[M[i, n, p, k] dc[n, k, r, j, dl, dm], {n, 3}, {k, 3}],
   {i, 3}, {p, 3}, {j, 3}, {r, 3}];
proj[t_, v_] := Sum[v[[i, p]] t[[i, p, j, r]] v[[j, r]], {i, 3}, {p, 3}, {j, 3}, {r, 3}]/
   Sum[v[[i, p]]^2, {i, 3}, {p, 3}];
vA1g = IdentityMatrix[3];
vT2g = Table[dd[i, 1] dd[p, 2] + dd[i, 2] dd[p, 1], {i, 3}, {p, 3}];

ser[x_, o_] := Normal[Series[x, {w, 0, o}]];
sigma[0] = ser[-Coefficient[ser[proj[a22[eps, 0], vA1g], 3], eps], 3];
sigma[1] = ser[w^2 gBall, 3];
sigma[2] = ser[-Coefficient[ser[proj[a22[0, eps], vT2g], 3], eps], 3];
Do[sigma[n] = Expand[Simplify[sigma[n], lam0 > 0]], {n, 0, 2}];

(* ---------------------------------------------------------------------------
   the Mie side: e_n(w) = [eps^2 coefficient]/[eps coefficient] of a_n
   --------------------------------------------------------------------------- *)
drive = {0 -> {dlam -> eps, dmu -> 0, drho -> 0}, 1 -> {dlam -> 0, dmu -> 0, drho -> eps},
   2 -> {dlam -> 0, dmu -> eps, drho -> 0}};
Do[Module[{an, a1, a2},
   an = MieAsymptoticRaw[n]["a"] /. (n /. drive);
   an = Normal[Series[an, {eps, 0, 2}]];
   a1 = Coefficient[an, eps, 1]; a2 = Coefficient[an, eps, 2];
   mieE[n] = Expand[Simplify[ser[a2/a1, 3], lam0 > 0]]], {n, 0, 2}];

(* ---------------------------------------------------------------------------
   checks and the closed forms
   --------------------------------------------------------------------------- *)
Print["  channels driven by: n=0 dlam, n=1 drho, n=2 dmu  (a = mu0 = rho0 = 1, w = k_S a)"];
Do[
  Print["  n = ", n, ":  closure sigma_", n, "(w) = ", InputForm[sigma[n]]];
  Print["          Mie     e_", n, "(w)     = ", InputForm[mieE[n]]],
  {n, 0, 2}];
Print["  [1] static channels agree (sigma_n(0) == e_n(0), n = 0,1,2): ",
  chk[And @@ Table[Simplify[(sigma[n] - mieE[n]) /. w -> 0] === 0, {n, 0, 2}]]];
Print["      and are Eshelby's: n=0 -1/(lam0+2), n=2 -2(3 lam0+8)/(15(lam0+2)), n=1 0: ",
  chk[Simplify[{sigma[0], sigma[1], sigma[2]} /. w -> 0] ===
    Simplify[{-1/(lam0 + 2), 0, -2 (3 lam0 + 8)/(15 (lam0 + 2))}]]];
dep = Table[Expand[Simplify[sigma[n] - mieE[n], lam0 > 0]], {n, 0, 2}];
Print["  [2] the departure has no w^0 or w^1 term: ",
  chk[And @@ Table[Simplify[Coefficient[dep[[n + 1]], w, 0]] === 0 &&
       Simplify[Coefficient[dep[[n + 1]], w, 1]] === 0, {n, 0, 2}]]];
Print["  THE CLOSED FORMS -- closure/Mie - 1 = d_n w^2 x (driving contrast) + O(w^3, contrast^2):"];
dcoef = Table[FullSimplify[Coefficient[dep[[n + 1]], w, 2], lam0 > 0], {n, 0, 2}];
Do[Print["      d_", n, " = ", InputForm[dcoef[[n + 1]]]], {n, 0, 2}];
Print["      (the closure's own w^2 terms, for reference: ",
  InputForm[Table[FullSimplify[Coefficient[sigma[n], w, 2], lam0 > 0], {n, 0, 2}]], ")"];

Print["  [3] the w^3 (radiation-damping) terms agree exactly, n = 0,1,2: ",
  chk[And @@ Table[Simplify[Coefficient[dep[[n + 1]], w, 3], lam0 > 0] === 0, {n, 0, 2}]]];

(* dimensional forms: w = k_S a, (k_P a)^2 = w^2/(lam0+2), beta^2/alpha^2 = 1/(lam0+2);
   driving contrasts dlam = Dlambda/mu, drho = Drho/rho, dmu = Dmu/mu *)
Clear[kPa, kSa];
dim = {
   -(kPa^2) (Dlambda/(lam + 2 mu))/10,
   (2 kSa^2 + kPa^2)/30 (Drho/rho),
   -(3 kSa^2 + 2 kPa^2/(lam0 + 2)) (Dmu/mu)/75};
toNondim = {kSa -> w, kPa -> w/Sqrt[lam0 + 2], lam -> lam0 mu, Dlambda -> dlam mu, Dmu -> dmu mu,
   Drho -> drho rho};
Print["  [4] dimensional forms  -(k_P a)^2 Dlam/(10(lam+2mu)),  (2(k_S a)^2 + (k_P a)^2) Drho/(30 rho),"];
Print["      -(3(k_S a)^2 + 2(k_P a)^2 beta^2/alpha^2) Dmu/(75 mu)  reproduce d_n w^2 x contrast: ",
  chk[And @@ Table[Simplify[(dim[[n + 1]] /. toNondim) -
        dcoef[[n + 1]] w^2 ({dlam, drho, dmu}[[n + 1]]), mu > 0 && lam0 > 0] === 0, {n, 0, 2}]]];

(* the validated background of the Python gate: alpha 5, beta 3, rho 2.5 -> lam0 = 7/9;
   dlam = 2/22.5, drho = 0.1/2.5, dmu = 1/22.5 *)
bg = {lam0 -> 7/9};
drv = {2/(45/2), 1/25, 1/(45/2)};
Print["  at the validated background (lam0 = 7/9), departure / w^2 = ",
  N[Table[dcoef[[n + 1]] drv[[n + 1]] /. bg, {n, 0, 2}], 6]];
Print["      the Python gate measured at k_S a = 0.1 (divided by w^2): 1.118e-3, 3.159e-3, 1.897e-3"];

Print["==== SphereClosureDynamic: ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
