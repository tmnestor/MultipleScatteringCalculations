(* ::Package:: *)
(* MieAsymptoticVerify.wl — Validation of MieAsymptotic.wl outputs against
   textbook special cases. Load the saved leading/interior/raw files and
   verify the symbolic identities below.

   Run:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/MieAsymptoticVerify.wl

   Checks:
     V1.  E_0 = (lam0+2) / (lam0+2 + dlam + 2 dmu/3)   [bulk-modulus Eshelby]
     V2.  E_1 = 1                                       [no Eshelby for translation]
     V3.  E_2 = 1 / (1 + betaE * dmu)                   [shear Eshelby]
              with betaE = 2 (8 + 3 lam0) / (15 (lam0 + 2))
                        = 2 (4 - 5 nu) / (15 (1 - nu))
     V4.  u_r^{int,(2)} / u_r^{int,(2)}|_{contrasts=0}   ->  1 / (1 + betaE * dmu)
              i.e. interior strain reduces to the static Eshelby concentration.
     V5.  Same identity for u_theta^{int,(2)}.
     V6.  Numerical sanity at the Python test parameters.
*)

ClearAll["Global`*"];

$dir = DirectoryName[$InputFileName];
Get[FileNameJoin[{$dir, "MieAsymptoticRaw.wl"}]];
Get[FileNameJoin[{$dir, "MieAsymptoticLeading.wl"}]];
Get[FileNameJoin[{$dir, "MieAsymptoticInterior.wl"}]];

Print["==== Loaded raw / leading / interior expressions ===="];

(* ============================================================ *)
(* Helper: Born-linearize a full (rational in dlam, dmu, drho)
   expression by series-expanding in eps and keeping leading order. *)
bornize[expr_] :=
  Normal[Series[
    expr /. {dlam -> eps dlam, dmu -> eps dmu, drho -> eps drho},
    {eps, 0, 1}]] /. eps -> 1;

(* Eshelby concentration factor, computed in the symbolic w -> 0 limit. *)
eshelbyEn[full_] := Module[{born, ratio},
  born = bornize[full];
  ratio = Cancel[Limit[full/born, w -> 0]];
  Together[ratio]
];

(* ============================================================ *)
(* V1.  E_0 == (lam0+2) / (lam0+2 + dlam + 2 dmu/3)             *)
(* ============================================================ *)
Print["\n---- V1: E_0 = (lam0+2) / (lam0+2 + Delta K) ----"];
a0 = MieAsymptoticRaw[0]["a"];
e0 = eshelbyEn[a0];
e0Expected = (lam0 + 2) / (lam0 + 2 + dlam + 2 dmu/3);
Print["  E_0 (computed) = ", e0];
Print["  E_0 (expected) = ", Together[e0Expected]];
diff1 = Together[e0 - e0Expected];
Print["  difference     = ", diff1, "   ", If[diff1 === 0, "PASS", "FAIL"]];

(* ============================================================ *)
(* V2.  E_1 == 1                                                *)
(* ============================================================ *)
Print["\n---- V2: E_1 = 1 ----"];
a1 = MieAsymptoticRaw[1]["a"];
e1 = eshelbyEn[a1];
Print["  E_1 (computed) = ", Simplify[e1]];
diff2 = Together[e1 - 1];
Print["  difference from 1 = ", diff2, "   ",
      If[diff2 === 0, "PASS", "FAIL"]];

(* ============================================================ *)
(* V3.  E_2 == 1 / (1 + betaE * dmu)                            *)
(*       with betaE = 2 (8 + 3 lam0) / (15 (lam0 + 2))          *)
(* ============================================================ *)
Print["\n---- V3: E_2 = 1 / (1 + betaE * dmu) ----"];
betaE = 2 (8 + 3 lam0) / (15 (lam0 + 2));
a2 = MieAsymptoticRaw[2]["a"];
e2 = eshelbyEn[a2];
e2Expected = 1 / (1 + betaE * dmu);
Print["  E_2 (computed) = ", Together[e2]];
Print["  E_2 (expected) = ", Together[e2Expected]];
diff3 = Together[e2 - e2Expected];
Print["  difference     = ", diff3, "   ", If[diff3 === 0, "PASS", "FAIL"]];

(* Cross-check betaE via Poisson ratio formula *)
nu = lam0 / (2 (lam0 + 1));
betaE2 = Together[2 (4 - 5 nu) / (15 (1 - nu))];
Print["  betaE via (lam0,8,3) formula : ", Together[betaE]];
Print["  betaE via (4-5 nu)/(15(1-nu)): ", betaE2];
Print["  match: ", If[Together[betaE - betaE2] === 0, "PASS", "FAIL"]];

(* ============================================================ *)
(* V4.  u_r^{int,(2)}_leading reduces to static Eshelby         *)
(* ------------------------------------------------------------ *)
(* The Eshelby concentration factor relates the INTERIOR strain *)
(* to the background (incident) strain at the inclusion site:   *)
(*                                                              *)
(*    eps^int = E_2 * eps^infinity                              *)
(*                                                              *)
(* eps^infinity is the strain that would be at r=0 if there     *)
(* were no inclusion — i.e. evaluate u_r^{(2)} at zero contrast. *)
(* So the right comparison is full / no-contrast (NOT the Born  *)
(* linearization, which adds an O(Delta) correction term).      *)
(* ============================================================ *)
Print["\n---- V4: u_r^{(2)} interior -> Eshelby concentration ----"];
ur2 = MieAsymptoticInterior[2]["u_r"];
ur2lead = SeriesCoefficient[Series[ur2, {w, 0, 2}], 1];
Print["  u_r^{(2)}_lead = ", Short[ur2lead, 4]];

ur2NoContrast = ur2lead /. {dlam -> 0, dmu -> 0, drho -> 0};
Print["  no-contrast    = ", Short[ur2NoContrast, 4]];

ratio4 = Together[ur2lead / ur2NoContrast];
Print["  ratio          = ", Short[ratio4, 4]];
Print["  expected E_2   = ", Together[1/(1 + betaE dmu)]];
diff4 = Together[ratio4 - 1/(1 + betaE dmu)];
Print["  difference     = ", diff4, "   ",
      If[diff4 === 0, "PASS", "FAIL"]];

(* ============================================================ *)
(* V5.  Same identity for u_theta^{int,(2)}                      *)
(* ============================================================ *)
Print["\n---- V5: u_theta^{(2)} interior -> same Eshelby ----"];
uth2 = MieAsymptoticInterior[2]["u_theta"];
uth2lead = SeriesCoefficient[Series[uth2, {w, 0, 2}], 1];
uth2NoContrast = uth2lead /. {dlam -> 0, dmu -> 0, drho -> 0};
ratio5 = Together[uth2lead / uth2NoContrast];
diff5 = Together[ratio5 - 1/(1 + betaE dmu)];
Print["  ratio          = ", Short[ratio5, 4]];
Print["  difference     = ", diff5, "   ",
      If[diff5 === 0, "PASS", "FAIL"]];

(* ============================================================ *)
(* V6.  Numerical sanity check at Python test parameters         *)
(* ============================================================ *)
Print["\n---- V6: Numerical sanity at test parameters ----"];
testSubs = {lam0 -> 7/9, dlam -> 4/45, dmu -> 2/45, drho -> 1/25};
Print["  betaE numeric  = ", N[betaE /. testSubs, 12]];
Print["  E_2 numeric    = ", N[e2 /. testSubs, 12]];
Print["  expected       = ", N[1/(1 + betaE dmu) /. testSubs, 12]];
Print["  textbook ref   : Eshelby beta_E for ν=lam0/(2(lam0+1))=", N[nu /. testSubs, 6]];

Print["\n==== Verification done ===="];
