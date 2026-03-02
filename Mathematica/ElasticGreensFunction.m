(* ============================================================== *)
(* ElasticGreensFunction.m                                        *)
(* Symbolic implementation of Appendix A:                         *)
(*   Elastic Green's Function and Its Spatial Derivative          *)
(* From: Shekhar et al. (2023) - Integral equation method for     *)
(*   microseismic wavefield modelling in anisotropic elastic media *)
(* ============================================================== *)

(* Clear any pre-existing definitions *)
ClearAll["ElasticGreens`*"];
BeginPackage["ElasticGreens`"];

(* Public symbols *)
GreensNearField::usage =
  "GreensNearField[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the (k,l) component of the near-field Green's tensor G^NF (Eq. A.4).";
GreensNearFieldDeriv::usage =
  "GreensNearFieldDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the spatial derivative G^NF_{kl,i} (Eq. A.7).";
GreensFarFieldP::usage =
  "GreensFarFieldP[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the (k,l) component of the far-field P-wave Green's tensor G^P (Eq. A.8).";
GreensFarFieldPDeriv::usage =
  "GreensFarFieldPDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the spatial derivative G^P_{kl,i} (Eq. A.9).";
GreensFarFieldS::usage =
  "GreensFarFieldS[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the (k,l) component of the far-field S-wave Green's tensor G^S (Eq. A.10).";
GreensFarFieldSDeriv::usage =
  "GreensFarFieldSDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the spatial derivative G^S_{kl,i} (Eq. A.11).";
GreensTotal::usage =
  "GreensTotal[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the full Green's tensor G = G^NF + G^P + G^S.";
GreensTotalDeriv::usage =
  "GreensTotalDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns the full spatial derivative G_{kl,i}.";
HTensor::usage =
  "HTensor[i, j, k, x, \[Omega], \[Alpha], \[Beta], \[Rho]] returns H^(0)_{ijk} = (1/2)(G_{ij,k} + G_{ik,j}) (Eq. 10).";

Begin["`Private`"];

(* ============================================================== *)
(* Helper quantities                                               *)
(* ============================================================== *)

(* Distance r = ||x|| where x = x_r - x_s is the relative position vector *)
r[x_] := Sqrt[x[[1]]^2 + x[[2]]^2 + x[[3]]^2];

(* Kronecker delta *)
\[Delta][i_, j_] := KroneckerDelta[i, j];

(* ============================================================== *)
(* Near-field Green's tensor: G^NF (Eq. A.4)                      *)
(* G^NF_{kl} = -C * E_{kl}(r) / U(r)                            *)
(* where:                                                          *)
(*   C = (1/(8\[Pi]\[Rho]\[Beta]^2)) * (1 - \[Beta]^2/\[Alpha]^2)*)
(*   E_{kl}(r) = \[Delta]_{kl} - 3 x_k x_l / r^2                *)
(*   U(r) = r                                                     *)
(* ============================================================== *)

cNF[\[Alpha]_, \[Beta]_, \[Rho]_] :=
  1/(8 Pi \[Rho] \[Beta]^2) (1 - \[Beta]^2/\[Alpha]^2);

eKL[k_, l_, x_] := \[Delta][k, l] - 3 x[[k]] x[[l]]/r[x]^2;

uR[x_] := r[x];

GreensNearField[k_, l_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  -cNF[\[Alpha], \[Beta], \[Rho]] eKL[k, l, x] / uR[x];

(* ============================================================== *)
(* Near-field derivative: G^NF_{kl,i} (Eq. A.7)                   *)
(* Using the quotient/product rule result from the paper:          *)
(* G^NF_{kl,i} = (C/U^3)[D_i E_{kl} - (3/U^2)(2F_{kl}D_i       *)
(*               - (\[Delta]_{ik}D_l + \[Delta]_{il}D_k)U^2)]    *)
(* where:                                                          *)
(*   D_i = x_i                                                    *)
(*   F_{kl} = x_k x_l                                             *)
(*   E_{kl,i} = 3(2 x_k x_l x_i - (\[Delta]_{ik}x_l + \[Delta]_{il}x_k)r^2) / r^4 *)
(*   U_{,i} = x_i / r                                             *)
(* ============================================================== *)

dI[i_, x_] := x[[i]];

fKL[k_, l_, x_] := x[[k]] x[[l]];

eKLi[k_, l_, i_, x_] :=
  3 (2 x[[k]] x[[l]] x[[i]] - (\[Delta][i, k] x[[l]] + \[Delta][i, l] x[[k]]) r[x]^2) / r[x]^4;

uRi[i_, x_] := x[[i]] / r[x];

GreensNearFieldDeriv[k_, l_, i_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  cNF[\[Alpha], \[Beta], \[Rho]] / uR[x]^3 * (
    dI[i, x] eKL[k, l, x] -
    3/uR[x]^2 (2 fKL[k, l, x] dI[i, x] -
      (\[Delta][i, k] dI[l, x] + \[Delta][i, l] dI[k, x]) uR[x]^2)
  );

(* ============================================================== *)
(* Far-field P-wave Green's tensor: G^P (Eq. A.8)                 *)
(* G^P_{kl} = X(r) Y_{kl}(r) / U(r)                             *)
(* where:                                                          *)
(*   X(r) = (1/(4\[Pi]\[Rho]\[Alpha]^2)) Exp[I \[Omega] r/\[Alpha]] *)
(*   Y_{kl}(r) = x_k x_l / r^2                                   *)
(*   U(r) = r                                                     *)
(* ============================================================== *)

xFuncP[x_, \[Omega]_, \[Alpha]_, \[Rho]_] :=
  1/(4 Pi \[Rho] \[Alpha]^2) Exp[I \[Omega] r[x] / \[Alpha]];

yKL[k_, l_, x_] := x[[k]] x[[l]] / r[x]^2;

GreensFarFieldP[k_, l_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  xFuncP[x, \[Omega], \[Alpha], \[Rho]] yKL[k, l, x] / uR[x];

(* ============================================================== *)
(* Far-field P-wave derivative: G^P_{kl,i} (Eq. A.9)              *)
(* G^P_{kl,i} = (X_{,i} Y_{kl} + X Y_{kl,i}) U - X Y_{kl} U_{,i} *)
(*             / U^2                                               *)
(* = (X_{,i} Y_{kl} + X Y_{kl,i})/U - X Y_{kl} U_{,i}/U^2      *)
(* where:                                                          *)
(*   X_{,i} = (I \[Omega] X(r) x_i) / (\[Alpha] r)              *)
(*   Y_{kl,i} = (r^2(\[Delta]_{ik}x_l + \[Delta]_{il}x_k) - 2 x_k x_l x_i) / r^4 *)
(* ============================================================== *)

xFuncPi[i_, x_, \[Omega]_, \[Alpha]_, \[Rho]_] :=
  I \[Omega] xFuncP[x, \[Omega], \[Alpha], \[Rho]] x[[i]] / (\[Alpha] r[x]);

yKLi[k_, l_, i_, x_] :=
  (r[x]^2 (\[Delta][i, k] x[[l]] + \[Delta][i, l] x[[k]]) - 2 x[[k]] x[[l]] x[[i]]) / r[x]^4;

GreensFarFieldPDeriv[k_, l_, i_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  (xFuncPi[i, x, \[Omega], \[Alpha], \[Rho]] yKL[k, l, x] +
   xFuncP[x, \[Omega], \[Alpha], \[Rho]] yKLi[k, l, i, x]) / uR[x] -
  xFuncP[x, \[Omega], \[Alpha], \[Rho]] yKL[k, l, x] uRi[i, x] / uR[x]^2;

(* Simplified form using the paper's notation (Eq. A.9):           *)
(* G^P_{kl,i} = (X_{,i} Y_{kl} + X Y_{kl,i}) U - X Y_{kl} U_{,i} / U^2 *)

(* ============================================================== *)
(* Far-field S-wave Green's tensor: G^S (Eq. A.10)                *)
(* G^S_{kl} = V(r) W_{kl}(r) / U(r)                             *)
(* where:                                                          *)
(*   V(r) = (1/(4\[Pi]\[Rho]\[Beta]^2)) Exp[I \[Omega] r/\[Beta]] *)
(*   W_{kl}(r) = \[Delta]_{kl} - x_k x_l / r^2                   *)
(* ============================================================== *)

vFuncS[x_, \[Omega]_, \[Beta]_, \[Rho]_] :=
  1/(4 Pi \[Rho] \[Beta]^2) Exp[I \[Omega] r[x] / \[Beta]];

wKL[k_, l_, x_] := \[Delta][k, l] - x[[k]] x[[l]] / r[x]^2;

GreensFarFieldS[k_, l_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  vFuncS[x, \[Omega], \[Beta], \[Rho]] wKL[k, l, x] / uR[x];

(* ============================================================== *)
(* Far-field S-wave derivative: G^S_{kl,i} (Eq. A.11)             *)
(* G^S_{kl,i} = (V_{,i} W_{kl} + V W_{kl,i}) U - V W_{kl} U_{,i} / U^2 *)
(* where:                                                          *)
(*   V_{,i} = (I \[Omega] V(r) x_i) / (\[Beta] r)               *)
(*   W_{kl,i} = (-r^2(\[Delta]_{ik}x_l + \[Delta]_{il}x_k) + 2 x_k x_l x_i) / r^4 *)
(* ============================================================== *)

vFuncSi[i_, x_, \[Omega]_, \[Beta]_, \[Rho]_] :=
  I \[Omega] vFuncS[x, \[Omega], \[Beta], \[Rho]] x[[i]] / (\[Beta] r[x]);

wKLi[k_, l_, i_, x_] :=
  (-r[x]^2 (\[Delta][i, k] x[[l]] + \[Delta][i, l] x[[k]]) + 2 x[[k]] x[[l]] x[[i]]) / r[x]^4;

GreensFarFieldSDeriv[k_, l_, i_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  (vFuncSi[i, x, \[Omega], \[Beta], \[Rho]] wKL[k, l, x] +
   vFuncS[x, \[Omega], \[Beta], \[Rho]] wKLi[k, l, i, x]) / uR[x] -
  vFuncS[x, \[Omega], \[Beta], \[Rho]] wKL[k, l, x] uRi[i, x] / uR[x]^2;

(* ============================================================== *)
(* Full Green's tensor: G = G^NF + G^P + G^S                      *)
(* ============================================================== *)

GreensTotal[k_, l_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  GreensNearField[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] +
  GreensFarFieldP[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]] +
  GreensFarFieldS[k, l, x, \[Omega], \[Alpha], \[Beta], \[Rho]];

(* Full spatial derivative: G_{kl,i} *)
GreensTotalDeriv[k_, l_, i_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  GreensNearFieldDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] +
  GreensFarFieldPDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]] +
  GreensFarFieldSDeriv[k, l, i, x, \[Omega], \[Alpha], \[Beta], \[Rho]];

(* ============================================================== *)
(* H tensor: H^(0)_{ijk} = (1/2)(G_{ij,k} + G_{ik,j})  (Eq. 10) *)
(* This is the first-order spatial derivative of the background    *)
(* Green's function, symmetric in the last two indices.            *)
(* ============================================================== *)

HTensor[i_, j_, k_, x_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  1/2 (GreensTotalDeriv[i, j, k, x, \[Omega], \[Alpha], \[Beta], \[Rho]] +
       GreensTotalDeriv[i, k, j, x, \[Omega], \[Alpha], \[Beta], \[Rho]]);

End[];
EndPackage[];


(* ============================================================== *)
(* VERIFICATION AND USAGE EXAMPLES                                 *)
(* ============================================================== *)

Print["=== ElasticGreens Package Loaded ==="];
Print[""];

(* --- Example: Build the full 3x3 Green's tensor symbolically --- *)
Print["--- Full 3x3 Green's tensor (symbolic) ---"];
xSym = {x1, x2, x3};
gMatrix = Table[
  GreensTotal[k, l, xSym, \[Omega], \[Alpha], \[Beta], \[Rho]],
  {k, 1, 3}, {l, 1, 3}
];
Print[MatrixForm[gMatrix]];

(* --- Verification 1: Symmetry of G_{kl} = G_{lk} --- *)
Print[""];
Print["--- Verification: G_{kl} symmetry ---"];
symCheck = Table[
  Simplify[GreensTotal[k, l, xSym, \[Omega], \[Alpha], \[Beta], \[Rho]] -
           GreensTotal[l, k, xSym, \[Omega], \[Alpha], \[Beta], \[Rho]]],
  {k, 1, 3}, {l, 1, 3}
];
Print["G[k,l] - G[l,k] = ", MatrixForm[symCheck]];
Print["All zero (symmetric)? ", symCheck === Table[0, {3}, {3}]];

(* --- Verification 2: Symmetry of H_{ijk} in last two indices --- *)
Print[""];
Print["--- Verification: H_{ijk} symmetry in j,k ---"];
hSymCheck = Table[
  Simplify[HTensor[i, j, k, xSym, \[Omega], \[Alpha], \[Beta], \[Rho]] -
           HTensor[i, k, j, xSym, \[Omega], \[Alpha], \[Beta], \[Rho]]],
  {i, 1, 3}, {j, 1, 3}, {k, 1, 3}
];
Print["H[i,j,k] - H[i,k,j] all zero? ",
  And @@ Flatten[Map[# === 0 &, hSymCheck, {3}]]];

(* --- Verification 3: Numerical evaluation at a test point --- *)
Print[""];
Print["--- Numerical evaluation at test point ---"];
(* Typical isotropic parameters: Vp=5000 m/s, Vs=3000 m/s, rho=2500 kg/m^3 *)
(* Receiver at (100, 0, 0) m from source, frequency 10 Hz *)
xTest = {100.0, 0.0, 0.0};
\[Omega]Test = 2 Pi 10.0;  (* 10 Hz *)
\[Alpha]Test = 5000.0;     (* P-wave velocity m/s *)
\[Beta]Test = 3000.0;      (* S-wave velocity m/s *)
\[Rho]Test = 2500.0;       (* density kg/m^3 *)

gNum = Table[
  GreensTotal[k, l, xTest, \[Omega]Test, \[Alpha]Test, \[Beta]Test, \[Rho]Test],
  {k, 1, 3}, {l, 1, 3}
];
Print["G (numerical, 3x3):"];
Print[MatrixForm[N[gNum]]];

(* --- Verification 4: Far-field limit check --- *)
(* At large r, G should be dominated by far-field terms *)
(* and G^NF should be negligible (decays as 1/r vs 1/r for far-field, *)
(* but without the oscillatory exp factor, so relative contribution *)
(* depends on omega*r/alpha and omega*r/beta) *)
Print[""];
Print["--- Far-field dominance check (large r) ---"];
xFar = {10000.0, 0.0, 0.0};
gNFfar = Abs[GreensNearField[1, 1, xFar, \[Omega]Test, \[Alpha]Test, \[Beta]Test, \[Rho]Test]];
gPfar = Abs[GreensFarFieldP[1, 1, xFar, \[Omega]Test, \[Alpha]Test, \[Beta]Test, \[Rho]Test]];
gSfar = Abs[GreensFarFieldS[1, 1, xFar, \[Omega]Test, \[Alpha]Test, \[Beta]Test, \[Rho]Test]];
Print["|G^NF_{11}| at r=10km: ", N[gNFfar]];
Print["|G^P_{11}|  at r=10km: ", N[gPfar]];
Print["|G^S_{11}|  at r=10km: ", N[gSfar]];
Print["Ratio |G^NF|/|G^P|: ", N[gNFfar/gPfar], "  (should be << 1 for far field)"];

(* --- Example: Build H tensor for Voigt notation source term --- *)
Print[""];
Print["--- H tensor example: H_{1,J} for J=1..6 (Voigt) ---"];
(* Voigt mapping: J=1->(1,1), J=2->(2,2), J=3->(3,3), *)
(*                J=4->(2,3), J=5->(1,3), J=6->(1,2)  *)
voigtMap = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
hVoigt = Table[
  With[{jj = voigtMap[[J, 1]], kk = voigtMap[[J, 2]]},
    HTensor[1, jj, kk, xTest, \[Omega]Test, \[Alpha]Test, \[Beta]Test, \[Rho]Test]
  ],
  {J, 1, 6}
];
Print["H_{1,J} = ", N[hVoigt]];

Print[""];
Print["=== Verification Complete ==="];
