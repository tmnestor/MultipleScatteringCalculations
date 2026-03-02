(* ================================================================ *)
(* TMatrixAnalytic.m                                                *)
(* Analytical T-Matrix for a Small Spherical Scatterer              *)
(* in an Isotropic Elastic Background Medium                        *)
(*                                                                  *)
(* Evaluates Equation 8 (Lippmann-Schwinger) of Shekhar et al.     *)
(* (2023) for a sphere of radius a with constant isotropic          *)
(* contrast (\[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu], \[CapitalDelta]\[Rho]).                                *)
(*                                                                  *)
(* All integrals are performed ANALYTICALLY in spherical            *)
(* coordinates using Mathematica's symbolic calculus.                *)
(* ================================================================ *)

ClearAll["Global`*"];

Print["================================================================"];
Print["  Analytical T-Matrix for Spherical Scatterer                   "];
Print["================================================================"];

(* ================================================================ *)
(* SECTION 1: GREEN'S TENSOR DECOMPOSITION                          *)
(* ================================================================ *)
(* In an isotropic homogeneous background, the elastodynamic        *)
(* Green's tensor has the form (Appendix A):                        *)
(*   G_{ij}(x) = f(r) \[Delta]_{ij} + g(r) \[HAT]x_i \[HAT]x_j                  *)
(* with \[HAT]x_i = x_i/r the unit direction vector.                      *)
(*                                                                  *)
(* From the near-field (NF), P-wave (P), and S-wave (S) parts:     *)
(*   f(r) = V(r)/r - C/r                                           *)
(*   g(r) = 3C/r + X(r)/r - V(r)/r                                 *)
(* where:                                                           *)
(*   C   = (1 - \[Beta]^2/\[Alpha]^2)/(8\[Pi]\[Rho]\[Beta]^2)                          *)
(*   X(r) = exp(i\[Omega]r/\[Alpha])/(4\[Pi]\[Rho]\[Alpha]^2)                          *)
(*   V(r) = exp(i\[Omega]r/\[Beta])/(4\[Pi]\[Rho]\[Beta]^2)                            *)
(* ================================================================ *)

Print["\n--- Section 1: Radial Functions f(r), g(r) ---"];

(* Background parameters: \[Alpha]=Vp, \[Beta]=Vs, \[Rho]=density *)
(* Define the constituent functions *)
CC[\[Alpha]_, \[Beta]_, \[Rho]_] := (1 - \[Beta]^2/\[Alpha]^2)/(8 Pi \[Rho] \[Beta]^2);
XX[r_, \[Omega]_, \[Alpha]_, \[Rho]_] := Exp[I \[Omega] r/\[Alpha]]/(4 Pi \[Rho] \[Alpha]^2);
VV[r_, \[Omega]_, \[Beta]_, \[Rho]_] := Exp[I \[Omega] r/\[Beta]]/(4 Pi \[Rho] \[Beta]^2);

(* Radial functions *)
ff[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  VV[r, \[Omega], \[Beta], \[Rho]]/r - CC[\[Alpha], \[Beta], \[Rho]]/r;

gg[r_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] :=
  3 CC[\[Alpha], \[Beta], \[Rho]]/r + XX[r, \[Omega], \[Alpha], \[Rho]]/r - VV[r, \[Omega], \[Beta], \[Rho]]/r;

(* Simplify f(r) *)
fSimp = FullSimplify[ff[r, \[Omega], \[Alpha], \[Beta], \[Rho]],
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}];
gSimp = FullSimplify[gg[r, \[Omega], \[Alpha], \[Beta], \[Rho]],
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}];
Print["f(r) = ", fSimp];
Print["g(r) = ", gSimp];

(* ================================================================ *)
(* SECTION 2: ANGULAR INTEGRAL IDENTITIES                           *)
(* ================================================================ *)
(* \[Integral] d\[CapitalOmega] 1 = 4\[Pi]                                                *)
(* \[Integral] d\[CapitalOmega] \[HAT]x_i \[HAT]x_j = (4\[Pi]/3) \[Delta]_{ij}                             *)
(* \[Integral] d\[CapitalOmega] \[HAT]x_i \[HAT]x_j \[HAT]x_k \[HAT]x_l                                   *)
(*   = (4\[Pi]/15)(\[Delta]_{ij}\[Delta]_{kl} + \[Delta]_{ik}\[Delta]_{jl} + \[Delta]_{il}\[Delta]_{jk})    *)
(* All odd-rank integrals vanish.                                   *)
(* ================================================================ *)

(* ================================================================ *)
(* SECTION 3: VOLUME INTEGRAL \[CapitalGamma]_0 = \[Integral] G_{ij} d^3x' / \[Delta]_{ij}        *)
(* ================================================================ *)
(* \[Integral]_sphere G_{ij}(x') d^3x'                                     *)
(*   = \[Delta]_{ij} \[Integral]_0^a r^2 [4\[Pi] f(r) + (4\[Pi]/3) g(r)] dr             *)
(*   = \[Delta]_{ij} \[CapitalGamma]_0                                                    *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 3: Analytical Integration of \[CapitalGamma]_0                    "];
Print["================================================================"];

(* The integrand for \[CapitalGamma]_0 *)
\[CapitalGamma]0integrand = r^2 (4 Pi ff[r, \[Omega], \[Alpha], \[Beta], \[Rho]] +
                    (4 Pi/3) gg[r, \[Omega], \[Alpha], \[Beta], \[Rho]]);
\[CapitalGamma]0integrand = FullSimplify[\[CapitalGamma]0integrand,
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}];

Print["Integrand for \[CapitalGamma]_0 = ", \[CapitalGamma]0integrand];

(* Perform the radial integral analytically *)
\[CapitalGamma]0exact = Integrate[\[CapitalGamma]0integrand, {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}];
\[CapitalGamma]0exact = FullSimplify[\[CapitalGamma]0exact];

Print["\n\[CapitalGamma]_0 (exact, closed-form):"];
Print[\[CapitalGamma]0exact];

(* Introduce dimensionless wavenumber-radius products *)
(* k_P = \[Omega]/\[Alpha],  k_S = \[Omega]/\[Beta] *)
(* \[Xi]_P = k_P a = \[Omega] a/\[Alpha],  \[Xi]_S = k_S a = \[Omega] a/\[Beta] *)
\[CapitalGamma]0dimless = \[CapitalGamma]0exact /. {\[Omega] -> \[Xi]P \[Alpha]/a} // FullSimplify;
Print["\n\[CapitalGamma]_0 in terms of \[Xi]_P = \[Omega]a/\[Alpha]:"];
Print[\[CapitalGamma]0dimless];

(* Small sphere expansion: \[Xi]_P << 1, \[Xi]_S << 1 *)
\[CapitalGamma]0series = Series[\[CapitalGamma]0exact, {\[Omega], 0, 2}] // Normal // FullSimplify;
Print["\n\[CapitalGamma]_0 (small sphere, leading order in \[Omega]):"];
Print[\[CapitalGamma]0series];

(* Also expand in terms of sphere radius a *)
\[CapitalGamma]0seriesA = Series[\[CapitalGamma]0exact, {a, 0, 4}] // Normal // FullSimplify;
Print["\n\[CapitalGamma]_0 (expanded in a to O(a^4)):"];
Print[\[CapitalGamma]0seriesA];

(* ================================================================ *)
(* SECTION 4: SECOND DERIVATIVE INTEGRAL                            *)
(* \[Integral] G_{ij,kl} d^3x' = A \[Delta]_{ij}\[Delta]_{kl} + B(\[Delta]_{ik}\[Delta]_{jl}+\[Delta]_{il}\[Delta]_{jk})  *)
(*                                                                  *)
(* Strategy: Compute the full second derivative of G_{ij} w.r.t.   *)
(* Cartesian coordinates, integrate each angular structure over     *)
(* the unit sphere, then integrate radially.                        *)
(*                                                                  *)
(* G_{ij,kl} involves angular structures: \[Delta]_{ij}\[Delta]_{kl}, \[Delta]_{ij}\[HAT]x_k\[HAT]x_l, *)
(* \[HAT]x_i\[HAT]x_j\[Delta]_{kl}, \[HAT]x_i\[HAT]x_j\[HAT]x_k\[HAT]x_l, and mixed \[Delta]\[HAT]x\[HAT]x terms.     *)
(* After angular integration, only isotropic 4th-rank structures   *)
(* survive.                                                         *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 4: Second Derivative Integral (A, B coefficients)     "];
Print["================================================================"];

(* ---- METHOD: Direct symbolic differentiation in Cartesian ---- *)
(* Define G_{ij} symbolically for specific (i,j) and differentiate *)

(* Helper: symbolic r in terms of Cartesian *)
rCart[x1_, x2_, x3_] := Sqrt[x1^2 + x2^2 + x3^2];

(* G_{ij} in Cartesian coordinates *)
GCart[i_, j_, x1_, x2_, x3_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] := Module[
  {rr, xi, xj, xVec = {x1, x2, x3}},
  rr = rCart[x1, x2, x3];
  xi = xVec[[i]]; xj = xVec[[j]];
  ff[rr, \[Omega], \[Alpha], \[Beta], \[Rho]] KroneckerDelta[i, j] +
  gg[rr, \[Omega], \[Alpha], \[Beta], \[Rho]] xi xj/rr^2
];

(* Second derivative G_{ij,kl} *)
GCartDeriv2[i_, j_, k_, l_, x1_, x2_, x3_, \[Omega]_, \[Alpha]_, \[Beta]_, \[Rho]_] := Module[
  {xVec = {x1, x2, x3}},
  D[GCart[i, j, x1, x2, x3, \[Omega], \[Alpha], \[Beta], \[Rho]], xVec[[k]], xVec[[l]]]
];

(* ---- Determine A and B via two independent contractions ---- *)
(* Contraction 1: I_{iikl} = (3A + 2B) \[Delta]_{kl}                  *)
(*   => 3A + 2B = I_{ii11} (taking k=l=1)                         *)

Print["Computing contraction I_{ii11} (trace over first two indices)..."];

(* Sum G_{ii,11} over i=1,2,3, then integrate over sphere *)
traceIntegrand1 = Sum[
  GCartDeriv2[i, i, 1, 1, x1, x2, x3, \[Omega], \[Alpha], \[Beta], \[Rho]],
  {i, 1, 3}];

(* Convert to spherical coordinates: x1=r Sin\[Theta] Cos\[Phi], x2=r Sin\[Theta] Sin\[Phi], x3=r Cos\[Theta] *)
traceIntegrand1Sph = traceIntegrand1 /. {
  x1 -> r Sin[\[Theta]] Cos[\[Phi]],
  x2 -> r Sin[\[Theta]] Sin[\[Phi]],
  x3 -> r Cos[\[Theta]]
};

(* Full volume integral in spherical coordinates *)
(* d^3x = r^2 Sin\[Theta] dr d\[Theta] d\[Phi] *)
Print["Performing angular integration for I_{ii11}..."];

contract1angular = Integrate[
  traceIntegrand1Sph r^2 Sin[\[Theta]], {\[Phi], 0, 2 Pi}, {\[Theta], 0, Pi},
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}
];
contract1angular = FullSimplify[contract1angular,
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}];
Print["Angular integral (I_{ii11}): ", contract1angular];

(* Radial integral *)
Print["Performing radial integration for I_{ii11}..."];
contract1full = Integrate[contract1angular, {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}];
contract1full = FullSimplify[contract1full];
Print["I_{ii11} = 3A + 2B = ", contract1full];

(* Contraction 2: I_{ij1j} summed over j, for i=1                 *)
(* I_{1jj1} = (A + 4B) for i=l=1                                 *)
(* Actually: I_{ijjl} = (A + 4B)\[Delta]_{il}, so I_{1jj1} = A + 4B  *)

Print["\nComputing contraction I_{1jj1} (mixed trace)..."];

traceIntegrand2 = Sum[
  GCartDeriv2[1, j, j, 1, x1, x2, x3, \[Omega], \[Alpha], \[Beta], \[Rho]],
  {j, 1, 3}];

traceIntegrand2Sph = traceIntegrand2 /. {
  x1 -> r Sin[\[Theta]] Cos[\[Phi]],
  x2 -> r Sin[\[Theta]] Sin[\[Phi]],
  x3 -> r Cos[\[Theta]]
};

Print["Performing angular integration for I_{1jj1}..."];
contract2angular = Integrate[
  traceIntegrand2Sph r^2 Sin[\[Theta]], {\[Phi], 0, 2 Pi}, {\[Theta], 0, Pi},
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}
];
contract2angular = FullSimplify[contract2angular,
  Assumptions -> {r > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0}];
Print["Angular integral (I_{1jj1}): ", contract2angular];

Print["Performing radial integration for I_{1jj1}..."];
contract2full = Integrate[contract2angular, {r, 0, a},
  Assumptions -> {a > 0, \[Omega] > 0, \[Alpha] > 0, \[Beta] > 0, \[Rho] > 0, \[Alpha] > \[Beta]}];
contract2full = FullSimplify[contract2full];
Print["I_{1jj1} = A + 4B = ", contract2full];

(* Solve for A and B *)
Print["\nSolving for A and B..."];
ABsolution = Solve[{
  3 Acoeff + 2 Bcoeff == contract1full,
  Acoeff + 4 Bcoeff == contract2full
}, {Acoeff, Bcoeff}];

Aexact = Acoeff /. ABsolution[[1]] // FullSimplify;
Bexact = Bcoeff /. ABsolution[[1]] // FullSimplify;

Print["\n*** EXACT CLOSED-FORM RESULTS ***"];
Print["A = ", Aexact];
Print["\nB = ", Bexact];

(* Small sphere expansion *)
AexactSeries = Series[Aexact, {a, 0, 3}] // Normal // FullSimplify;
BexactSeries = Series[Bexact, {a, 0, 3}] // Normal // FullSimplify;
Print["\nA (expanded in a): ", AexactSeries];
Print["B (expanded in a): ", BexactSeries];

(* ================================================================ *)
(* SECTION 5: COMPLETE T-MATRIX ASSEMBLY                            *)
(* ================================================================ *)
(* The self-consistent interior fields are:                         *)
(*                                                                  *)
(* DISPLACEMENT:                                                    *)
(*   u_i = u^(0)_i / (1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0)                          *)
(*                                                                  *)
(* STRAIN (isotropic decomposition):                                *)
(*   \[Theta] = \[Theta]^(0) / (1 - 3T_1 - 2T_2)                              *)
(*   e_{mn} = e^(0)_{mn} / (1 - 2T_2)                             *)
(*                                                                  *)
(* where T_1, T_2 come from contracting                            *)
(*   S_{mnjk} = (1/2)(I_{mjkn} + I_{njkm})                       *)
(* with the isotropic stiffness contrast                           *)
(*   \[CapitalDelta]c_{jklp} = \[CapitalDelta]\[Lambda] \[Delta]_{jk}\[Delta]_{lp} + \[CapitalDelta]\[Mu](\[Delta]_{jl}\[Delta]_{kp}+\[Delta]_{jp}\[Delta]_{kl})  *)
(*                                                                  *)
(* The scattered field at exterior point x is:                      *)
(*   u^{scat}_i(x) = V [H_{ijk}(x,x_s) \[CapitalDelta]c^*_{jklm} \[Epsilon]^(0)_{lm}  *)
(*                      + \[Omega]^2 G_{ij}(x,x_s) \[CapitalDelta]\[Rho]^* u^(0)_j]     *)
(* with V = (4/3)\[Pi]a^3 and effective (renormalized) contrasts.       *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 5: T-Matrix Assembly                                  "];
Print["================================================================"];

(* ---- Compute T_1 and T_2 ---- *)
(* S_{mnjk} = (A/2)(\[Delta]_{mj}\[Delta]_{kn}+\[Delta]_{nj}\[Delta]_{km}) + B \[Delta]_{mn}\[Delta]_{jk}    *)
(*          + (B/2)(\[Delta]_{mk}\[Delta]_{jn}+\[Delta]_{nk}\[Delta]_{jm})                  *)
(*                                                                  *)
(* Contracted with \[CapitalDelta]c_{jklp}:                                     *)
(*   T_{mnlp} = \[Sum]_{jk} S_{mnjk} \[CapitalDelta]c_{jklp}                       *)
(*            = T_1 \[Delta]_{mn}\[Delta]_{lp} + T_2(\[Delta]_{ml}\[Delta]_{np}+\[Delta]_{mp}\[Delta]_{nl}) *)

(* T_1 = T_{1122}: coefficient of \[Delta]_{mn}\[Delta]_{lp} *)
(* From the contraction (computed symbolically): *)
T1exact = FullSimplify[
  Aexact \[CapitalDelta]\[Lambda] + (Aexact + 2 Bexact) \[CapitalDelta]\[Mu] + 3 Bexact \[CapitalDelta]\[Lambda]
];

(* Let me compute T1 and T2 properly via explicit index contraction *)
(* S_{mnjk} for specific indices *)
Sfunc[m_, n_, j_, k_, Av_, Bv_] :=
  (Av/2)(KroneckerDelta[m,j] KroneckerDelta[k,n] +
         KroneckerDelta[n,j] KroneckerDelta[k,m]) +
  Bv KroneckerDelta[m,n] KroneckerDelta[j,k] +
  (Bv/2)(KroneckerDelta[m,k] KroneckerDelta[j,n] +
         KroneckerDelta[n,k] KroneckerDelta[j,m]);

(* \[CapitalDelta]c_{jklp} isotropic *)
DcFunc[j_, k_, l_, p_, dl_, dm_] :=
  dl KroneckerDelta[j,k] KroneckerDelta[l,p] +
  dm (KroneckerDelta[j,l] KroneckerDelta[k,p] +
      KroneckerDelta[j,p] KroneckerDelta[k,l]);

(* T_{mnlp} = Sum_{j,k} S_{mnjk} \[CapitalDelta]c_{jklp} *)
Tfunc[m_, n_, l_, p_, Av_, Bv_, dl_, dm_] :=
  Sum[Sfunc[m, n, j, k, Av, Bv] DcFunc[j, k, l, p, dl, dm],
    {j, 3}, {k, 3}];

(* Extract T_1 = T_{1122} and T_2 = T_{1212} *)
T1fromContraction = Simplify[Tfunc[1, 1, 2, 2, Aexact, Bexact, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];
T2fromContraction = Simplify[Tfunc[1, 2, 1, 2, Aexact, Bexact, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];

(* Verify: T_{1111} = T_1 + 2 T_2 *)
T1111check = Simplify[Tfunc[1, 1, 1, 1, Aexact, Bexact, \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu]]];
Print["Verification: T_{1111} = T_1 + 2T_2? ",
  Simplify[T1111check - T1fromContraction - 2 T2fromContraction] === 0];

T1exact = FullSimplify[T1fromContraction];
T2exact = FullSimplify[T2fromContraction];

Print["\n*** T-MATRIX COUPLING COEFFICIENTS (EXACT) ***"];
Print["T_1 (volumetric) = ", T1exact];
Print["\nT_2 (shear)      = ", T2exact];

(* ---- Amplification factors ---- *)
Print["\n*** SELF-CONSISTENT AMPLIFICATION FACTORS ***"];

(* Displacement *)
ampDisp = 1/(1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]0exact) // FullSimplify;
Print["\nDisplacement amplification:"];
Print["  A_u = 1/(1 - \[Omega]^2 \[CapitalDelta]\[Rho] \[CapitalGamma]_0) = ", ampDisp];

(* Dilatation *)
ampDilat = 1/(1 - 3 T1exact - 2 T2exact) // FullSimplify;
Print["\nDilatation amplification:"];
Print["  A_\[Theta] = 1/(1 - 3T_1 - 2T_2) = ", ampDilat];

(* Deviatoric strain *)
ampShear = 1/(1 - 2 T2exact) // FullSimplify;
Print["\nDeviatoric strain amplification:"];
Print["  A_e = 1/(1 - 2T_2) = ", ampShear];

(* ---- Effective (renormalized) contrasts ---- *)
Print["\n*** EFFECTIVE CONTRASTS (T-MATRIX) ***"];

\[CapitalDelta]\[Rho]star = \[CapitalDelta]\[Rho] ampDisp // FullSimplify;
\[CapitalDelta]\[Lambda]star = FullSimplify[
  \[CapitalDelta]\[Lambda] ampDilat + (2/3) \[CapitalDelta]\[Mu] (ampDilat - ampShear)
];
\[CapitalDelta]\[Mu]star = \[CapitalDelta]\[Mu] ampShear // FullSimplify;

Print["\n\[CapitalDelta]\[Rho]* (effective density contrast):"];
Print["  ", \[CapitalDelta]\[Rho]star];
Print["\n\[CapitalDelta]\[Lambda]* (effective volumetric contrast):"];
Print["  ", \[CapitalDelta]\[Lambda]star];
Print["\n\[CapitalDelta]\[Mu]* (effective shear contrast):"];
Print["  ", \[CapitalDelta]\[Mu]star];

(* ================================================================ *)
(* SECTION 6: SCATTERED FIELD FORMULA                               *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 6: Complete Scattered Field                           "];
Print["================================================================"];

Print["\nThe scattered displacement at observation point x due to a"];
Print["small sphere of radius a centered at x_s is:\n"];
Print["  u^{scat}_i(x) = V * {"];
Print["    H_{ijk}(x, x_s) [\[CapitalDelta]\[Lambda]* \[Theta]^(0) \[Delta]_{jk} + 2\[CapitalDelta]\[Mu]* \[Epsilon]^(0)_{jk}]"];
Print["    + \[Omega]^2 G_{ij}(x, x_s) \[CapitalDelta]\[Rho]* u^(0)_j"];
Print["  }"];
Print["\nwhere V = (4/3)\[Pi]a^3 and the effective contrasts above"];
Print["encode the full self-consistent multiple scattering within the sphere."];

(* ================================================================ *)
(* SECTION 7: SMALL SPHERE (RAYLEIGH) LIMIT                        *)
(* ================================================================ *)
(* Expand everything to leading order in a (or equivalently \[Omega])    *)
(* This gives the Rayleigh scattering regime.                       *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 7: Rayleigh (Small Sphere) Limit                     "];
Print["================================================================"];

(* Leading-order expansions *)
\[CapitalGamma]0leading = Series[\[CapitalGamma]0exact, {a, 0, 4}] // Normal // FullSimplify;
Aleading = Series[Aexact, {a, 0, 3}] // Normal // FullSimplify;
Bleading = Series[Bexact, {a, 0, 3}] // Normal // FullSimplify;

Print["\nLeading order in sphere radius a:"];
Print["\[CapitalGamma]_0 \[TildeTilde] ", \[CapitalGamma]0leading];
Print["A \[TildeTilde] ", Aleading];
Print["B \[TildeTilde] ", Bleading];

T1leading = Series[T1exact, {a, 0, 3}] // Normal // FullSimplify;
T2leading = Series[T2exact, {a, 0, 3}] // Normal // FullSimplify;
Print["\nT_1 \[TildeTilde] ", T1leading];
Print["T_2 \[TildeTilde] ", T2leading];

(* In the Rayleigh limit, the amplification factors approach 1 *)
(* (Born approximation) to leading order. The T-matrix corrections *)
(* enter at higher order in (ka). *)

ampDispLeading = Series[ampDisp, {a, 0, 4}] // Normal // FullSimplify;
ampDilatLeading = Series[ampDilat, {a, 0, 3}] // Normal // FullSimplify;
ampShearLeading = Series[ampShear, {a, 0, 3}] // Normal // FullSimplify;

Print["\nAmplification factors (Rayleigh limit):"];
Print["A_u \[TildeTilde] ", ampDispLeading];
Print["A_\[Theta] \[TildeTilde] ", ampDilatLeading];
Print["A_e \[TildeTilde] ", ampShearLeading];

(* ================================================================ *)
(* SECTION 8: BORN APPROXIMATION CHECK                              *)
(* ================================================================ *)
(* In the Born approximation, all amplification factors = 1,        *)
(* and the effective contrasts equal the bare contrasts.             *)
(* This should be recovered in the limit a -> 0.                    *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 8: Born Approximation Verification                    "];
Print["================================================================"];

Print["\nIn the Born limit (a -> 0 or weak contrast):"];
Print["  \[CapitalDelta]\[Rho]* -> \[CapitalDelta]\[Rho]"];
Print["  \[CapitalDelta]\[Lambda]* -> \[CapitalDelta]\[Lambda]"];
Print["  \[CapitalDelta]\[Mu]* -> \[CapitalDelta]\[Mu]"];

bornCheck\[Rho] = Limit[\[CapitalDelta]\[Rho]star, a -> 0];
bornCheck\[Lambda] = Limit[\[CapitalDelta]\[Lambda]star, a -> 0];
bornCheck\[Mu] = Limit[\[CapitalDelta]\[Mu]star, a -> 0];

Print["Limit \[CapitalDelta]\[Rho]* as a->0: ", bornCheck\[Rho], "  (should be \[CapitalDelta]\[Rho])"];
Print["Limit \[CapitalDelta]\[Lambda]* as a->0: ", bornCheck\[Lambda], "  (should be \[CapitalDelta]\[Lambda])"];
Print["Limit \[CapitalDelta]\[Mu]* as a->0: ", bornCheck\[Mu], "  (should be \[CapitalDelta]\[Mu])"];

(* ================================================================ *)
(* SECTION 9: NUMERICAL EVALUATION                                  *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 9: Numerical Example                                  "];
Print["================================================================"];

(* Background medium *)
\[Alpha]n = 5000;    (* P-wave velocity, m/s *)
\[Beta]n = 3000;     (* S-wave velocity, m/s *)
\[Rho]n = 2500;      (* density, kg/m^3 *)
fn = 10;           (* frequency, Hz *)
\[Omega]n = 2 Pi fn;
an = 10;           (* sphere radius, m *)

(* Isotropic contrast *)
\[CapitalDelta]\[Lambda]n = 2.0 10^9;   (* Pa *)
\[CapitalDelta]\[Mu]n = 1.0 10^9;    (* Pa *)
\[CapitalDelta]\[Rho]n = 100.0;       (* kg/m^3 *)

(* Background Lam\[EAcute] parameters *)
\[Lambda]n = \[Rho]n (\[Alpha]n^2 - 2 \[Beta]n^2);
\[Mu]n = \[Rho]n \[Beta]n^2;
Print["Background: \[Lambda] = ", \[Lambda]n/10^9, " GPa, \[Mu] = ", \[Mu]n/10^9, " GPa"];
Print["k_P a = ", N[\[Omega]n an/\[Alpha]n], ", k_S a = ", N[\[Omega]n an/\[Beta]n]];

(* Evaluate all quantities *)
numRules = {\[Alpha] -> \[Alpha]n, \[Beta] -> \[Beta]n, \[Rho] -> \[Rho]n, \[Omega] -> \[Omega]n, a -> an,
  \[CapitalDelta]\[Lambda] -> \[CapitalDelta]\[Lambda]n, \[CapitalDelta]\[Mu] -> \[CapitalDelta]\[Mu]n, \[CapitalDelta]\[Rho] -> \[CapitalDelta]\[Rho]n};

Print["\n\[CapitalGamma]_0 = ", N[\[CapitalGamma]0exact /. numRules]];
Print["A = ", N[Aexact /. numRules]];
Print["B = ", N[Bexact /. numRules]];
Print["T_1 = ", N[T1exact /. numRules]];
Print["T_2 = ", N[T2exact /. numRules]];

Print["\nAmplification factors:"];
Print["  A_u (displacement)     = ", N[ampDisp /. numRules]];
Print["  A_\[Theta] (dilatation)       = ", N[ampDilat /. numRules]];
Print["  A_e (deviatoric strain) = ", N[ampShear /. numRules]];

Print["\nEffective contrasts:"];
Print["  \[CapitalDelta]\[Rho]* = ", N[\[CapitalDelta]\[Rho]star /. numRules], " kg/m^3"];
Print["  \[CapitalDelta]\[Lambda]* = ", N[\[CapitalDelta]\[Lambda]star /. numRules], " Pa"];
Print["  \[CapitalDelta]\[Mu]* = ", N[\[CapitalDelta]\[Mu]star /. numRules], " Pa"];

(* ================================================================ *)
(* SECTION 10: EXPORT RESULTS AS REPLACEMENT RULES                  *)
(* ================================================================ *)

Print["\n================================================================"];
Print["  Section 10: Exportable Replacement Rules                      "];
Print["================================================================"];

Print["\nThe T-matrix is fully defined by the three effective contrasts:"];
Print["  \[CapitalDelta]\[Rho]*, \[CapitalDelta]\[Lambda]*, \[CapitalDelta]\[Mu]*"];
Print["which are explicit functions of (\[Omega], a, \[Alpha], \[Beta], \[Rho], \[CapitalDelta]\[Lambda], \[CapitalDelta]\[Mu], \[CapitalDelta]\[Rho])."];
Print["\nTo use in the integral equation solver:"];
Print["  Replace \[CapitalDelta]c_{jklm}\[Epsilon]_{lm} with \[CapitalDelta]c*_{jklm}\[Epsilon]^(0)_{lm}"];
Print["  Replace \[CapitalDelta]\[Rho] u_j with \[CapitalDelta]\[Rho]* u^(0)_j"];
Print["  Multiply by sphere volume V = (4/3)\[Pi]a^3"];

(* Save key results *)
TMatrixResults = <|
  "\[CapitalGamma]0" -> \[CapitalGamma]0exact,
  "A" -> Aexact,
  "B" -> Bexact,
  "T1" -> T1exact,
  "T2" -> T2exact,
  "AmpDisplacement" -> ampDisp,
  "AmpDilatation" -> ampDilat,
  "AmpDeviatoric" -> ampShear,
  "EffectiveDensity" -> \[CapitalDelta]\[Rho]star,
  "EffectiveLambda" -> \[CapitalDelta]\[Lambda]star,
  "EffectiveMu" -> \[CapitalDelta]\[Mu]star
|>;

Print["\nAll results stored in TMatrixResults association."];
Print["\n================================================================"];
Print["  Computation Complete                                          "];
Print["================================================================"];
