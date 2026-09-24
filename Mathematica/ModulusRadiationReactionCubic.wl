(* ::Package:: *)
(* ModulusRadiationReactionCubic.wl

   Symbolic check of the cubic modulus radiation reaction used by
   cubic_scattering/effective_contrasts.py::_modulus_radiation_reaction_cubic.

   A point stress dipole M = V C:e (C the real cubic effective contrast) radiates
       u_P = (i kP/(4 Pi rho alpha^2)) (r.M.r) r,
       u_S = (i kS/(4 Pi rho beta^2)) (I - r r).M.r      (per e^{ikr}/r),
   each carrying radial flux (1/2) rho omega^2 c |u|^2.  Energy conservation:
       Im[e:DeltaC*:e] = -(2/(omega V)) P_rad(M)   for every symmetric e.
   The angular integral is done EXPLICITLY here (no 1/15 identities assumed),
   the quadratic form in e is polarised into a 4-tensor, and its three cubic
   invariants are compared to the closed form.

   Run: wolframscript -file Mathematica/ModulusRadiationReactionCubic.wl
*)

ClearAll["Global`*"];
assume = {om > 0, rho > 0, al > 0, be > 0, V > 0};

d = IdentityMatrix[3];
cubicTensor[lam_, mo_, md_] := Table[
   lam d[[i, j]] d[[k, l]] + mo (d[[i, k]] d[[j, l]] + d[[i, l]] d[[j, k]]) +
    2 (md - mo) Boole[i == j == k == l], {i, 3}, {j, 3}, {k, 3}, {l, 3}];

C4 = cubicTensor[lam, mo, md];

(* generic symmetric strain *)
e = Table[Symbol["e" <> ToString[Min[i, j]] <> ToString[Max[i, j]]], {i, 3}, {j, 3}];
M = V TensorContract[TensorProduct[C4, e], {{3, 5}, {4, 6}}];

r = {Sin[th] Cos[ph], Sin[th] Sin[ph], Cos[th]};
Mr = M . r;
rMr = r . Mr;
kP = om/al; kS = om/be;
uP2 = (kP/(4 Pi rho al^2))^2 rMr^2;
uSv = (kS/(4 Pi rho be^2)) (Mr - rMr r);
flux = (1/2) rho om^2 (al uP2 + be uSv . uSv);

Prad = Integrate[Expand[flux] Sin[th], {th, 0, Pi}, {ph, 0, 2 Pi}, Assumptions -> assume];
quad = Simplify[-(2/(om V)) Prad];      (* = e:ImC:e *)

(* Polarise: ImC_ijkl = (1/2) d^2 quad / de_ij de_kl on the symmetric parametrisation *)
imL = Simplify[D[quad, e11, e22]/2];                  (* C_1122 *)
imMo = Simplify[D[quad, e12, e12]/8];                 (* e12 appears 2x in e, C_1212 *)
imC1111 = Simplify[D[quad, e11, e11]/2];
imMd = Simplify[(imC1111 - imL)/2];

(* closed form, as implemented in Python *)
cP = om^4/(8 Pi rho al^5); cS = om^4/(8 Pi rho be^5);
kLam = (cP - cS)/15; kMu = (2 cP + 3 cS)/30;
tr = 3 lam + 2 md;
refL = -(2 V/om) ((kLam tr + 2 kMu lam) tr + 4 kMu lam md);
refMo = -(8 V/om) kMu mo^2;
refMd = -(8 V/om) kMu md^2;

checks = {
   "Im lambda*" -> Simplify[imL - refL, assume],
   "Im mu*_off" -> Simplify[imMo - refMo, assume],
   "Im mu*_diag" -> Simplify[imMd - refMd, assume],
   (* the quadratic form must be exactly cubic: no other invariants *)
   "cubic closure" -> Simplify[quad - Total[Flatten[
         e*TensorContract[TensorProduct[cubicTensor[imL, imMo, imMd], e], {{3, 5}, {4, 6}}]]], assume]
   };

Print[Column[checks]];
If[AllTrue[Values[checks], # === 0 &],
  Print["PASS: cubic closed form equals the explicit far-field radiated power."],
  Print["FAIL"]; Exit[1]];
