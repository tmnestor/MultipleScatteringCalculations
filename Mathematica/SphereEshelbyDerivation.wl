(* SphereEshelbyDerivation.wl

   Derives multipole Eshelby concentration factors E_n for elastic sphere
   scattering via series expansion of the Mie boundary problem.

   Three key tricks that avoid the SphericalBesselJ simplification hang:
   1. sin/cos recurrence for Bessel functions (pure polynomials after Series)
   2. Physical wavespeeds directly (no independent slowness variables)
   3. Series[detCr/detM, ...] instead of Cancel (avoids algebraic GCD)

   Non-dimensionalized: mu0 = rho0 = a = 1.
   Free symbols: lam0 (=lambda0/mu0), dlam, dmu, drho.

   T. M. Nestor -- ANU
*)

ClearAll["Global`*"]

(* ===== Spherical Bessel functions via sin/cos recurrence ===== *)

jn[0, z_] := Sin[z]/z;
jn[1, z_] := Sin[z]/z^2 - Cos[z]/z;
jn[n_Integer /; n >= 2, z_] :=
  (jn[n, z] = (2 n - 1)/z jn[n - 1, z] - jn[n - 2, z]);

yn[0, z_] := -Cos[z]/z;
yn[1, z_] := -Cos[z]/z^2 - Sin[z]/z;
yn[n_Integer /; n >= 2, z_] :=
  (yn[n, z] = (2 n - 1)/z yn[n - 1, z] - yn[n - 2, z]);

hn[n_, z_] := jn[n, z] + I yn[n, z];

(* ===== Elastic field functions (match sphere_scattering.py) ===== *)

(* P-wave: phi = zn(kr) Pn. Returns {ur, utheta, srr, srtheta}. *)
pField[n_, k_, r_, lam_, mu_, zfn_] :=
  Module[{z = k r, t, zv, zp},
    zv = zfn[n, t] /. t -> z;
    zp = D[zfn[n, t], t] /. t -> z;
    {k zp, zv/r,
     -(lam + 2 mu) k^2 zv - 4 mu k zp/r +
       2 mu n (n + 1) zv/r^2,
     2 mu (k zp - zv/r)/r}];

(* S-wave: psi = zn(kr) Pn. Returns {ur, utheta, srr, srtheta}. *)
sField[n_, k_, r_, mu_, zfn_] :=
  Module[{z = k r, t, zv, zp, nn1 = n (n + 1)},
    zv = zfn[n, t] /. t -> z;
    zp = D[zfn[n, t], t] /. t -> z;
    {nn1 zv/r, (zv + z zp)/r,
     2 mu nn1 (k zp/r - zv/r^2),
     mu ((2 nn1 - 2 - z^2) zv - 2 z zp)/r^2}];

(* ===== Main derivation ===== *)

deriveEn[n_Integer] := Module[
  {w, eps, lami, mui,
   kPo, kSo, kPi, kSi,
   scP, scS, inP, inS, icP,
   mat, rhs, coeff, matS, rhsS, ord,
   cramer, dM, dCr, aSer, aBorn, en, raw},

  ord = 2 n + 8;
  lami = lam0 + dlam;
  mui = 1 + dmu;

  (* Wavenumbers with PHYSICAL wavespeeds -- no independent slowness.
     This ensures the Born limit correctly linearizes wavespeed
     dependence on contrasts via Series[Sqrt[...], {eps,0,1}]. *)
  kPo = w/Sqrt[lam0 + 2];
  kSo = w;   (* Sqrt[rho0/mu0] = 1 exactly *)
  kPi = w Sqrt[(1 + drho)/(lami + 2 mui)];
  kSi = w Sqrt[(1 + drho)/mui];

  If[n == 0,
    (* --- 2x2: P-wave only --- *)
    scP = pField[0, kPo, 1, lam0, 1, hn];
    inP = pField[0, kPi, 1, lami, mui, jn];
    icP = pField[0, kPo, 1, lam0, 1, jn];
    coeff = 1/(I kPo);
    mat = {{scP[[1]], -inP[[1]]}, {scP[[3]], -inP[[3]]}};
    rhs = -coeff {icP[[1]], icP[[3]]};
    ,
    (* --- 4x4: P-SV coupled --- *)
    scP = pField[n, kPo, 1, lam0, 1, hn];
    scS = sField[n, kSo, 1, 1, hn];
    inP = pField[n, kPi, 1, lami, mui, jn];
    inS = sField[n, kSi, 1, mui, jn];
    icP = pField[n, kPo, 1, lam0, 1, jn];
    coeff = (2 n + 1) I^n/(I kPo);
    mat = Transpose[{scP, scS, -inP, -inS}];
    rhs = -coeff icP;
  ];

  (* Series-expand every matrix entry in w *)
  Print["  [", n, "] Series expanding (ord=", ord, ")..."];
  matS = Map[Normal[Series[#, {w, 0, ord}]] &, mat, {2}];
  rhsS = Normal[Series[#, {w, 0, ord}]] & /@ rhs;

  (* Cramer's rule: replace column 1 with RHS *)
  Print["  [", n, "] Det (", Length[matS], "x", Length[matS], ")..."];
  cramer = matS;
  cramer[[All, 1]] = rhsS;
  dM = Det[matS];
  dCr = Det[cramer];

  (* Series of ratio -- fast Laurent division, no Cancel needed *)
  Print["  [", n, "] Laurent division..."];
  aSer = Normal[Series[dCr/dM, {w, 0, ord}]];

  (* Born limit: eps-scale contrasts only.
     Sqrt wavespeed dependence on (dlam,dmu,drho) is linearized
     automatically by Series[..., {eps,0,1}]. *)
  Print["  [", n, "] Born limit..."];
  aBorn = Normal[Series[
      aSer /. {dlam -> eps dlam, dmu -> eps dmu, drho -> eps drho},
      {eps, 0, 1}]] /. eps -> 1;

  (* E_n = lim_{w->0} a_n(full) / a_n(Born) *)
  Print["  [", n, "] Limit + Cancel..."];
  en = Cancel[Limit[aSer/aBorn, w -> 0]];

  (* Simplify with 30s timeout *)
  Print["  [", n, "] FullSimplify (30s timeout)..."];
  raw = en;
  en = TimeConstrained[
    FullSimplify[en,
      Assumptions -> {lam0 > 0, dmu > -1, drho > -1, dlam > -lam0}],
    30, raw];
  en
];

(* ===== Run ===== *)

Print["Deriving Eshelby concentration factors E_n ...\n"];
Print["Known: E_0 = K0/(K0 + \[Alpha]_E \[CapitalDelta]K),  ",
      "E_1 = 1,  ",
      "E_2 = 1/(1 + \[Beta]_E \[CapitalDelta]\[Mu]/\[Mu]_0)\n"];

results = Association[];
Do[
  {t, en} = AbsoluteTiming[deriveEn[n]];
  results[n] = en;
  Print["E_", n, " = ", en];
  Print["  (", NumberForm[t, {4, 1}], " s)\n"];
, {n, 0, 5}];

(* ===== Numerical verification ===== *)
(* Python test parameters: alpha=5000, beta=3000, rho=2500,
   Dlambda=2e9, Dmu=1e9, Drho=100.
   mu0 = rho*beta^2 = 22.5e9,  lam0 = rho*alpha^2 - 2*mu0 = 17.5e9.
   Non-dim: lam0/mu0 = 7/9, dlam/mu0 = 4/45, dmu/mu0 = 2/45, drho/rho0 = 1/25. *)

Print["\n===== Numerical check ====="];
testSubs = {lam0 -> 7/9, dlam -> 4/45, dmu -> 2/45, drho -> 1/25};
Do[
  val = results[n] /. testSubs;
  Print["E_", n, " = ", N[val, 10]];
, {n, 0, 5}];

Print["\n===== Python reference values ====="];
Print["E_0 = 0.9590828991"];
Print["E_1 = 1.0000029890"];
Print["E_2 = 0.9784326389"];
Print["E_3 = 0.9748482740"];
Print["E_4 = 0.9734666883"];
Print["E_5 = 0.9727525952"];
