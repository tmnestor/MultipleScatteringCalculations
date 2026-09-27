#!/usr/bin/env wolframscript
(* ============================================================================
   CubeShearSplit.wl  --  the cube's shear channels against the sphere's.

   The static strain self-term of a cube has three O_h channels, the eigenvalues
   of the first-gradient block A22 = I - M.dc assembled from the moment engine
   (CubeA22Block.wl, loaded here: the channels are DERIVED, not transcribed).
   A sphere has one deviatoric channel, Eshelby's
       1 + 2 dmu S_sph,   2 mu S_sph = 2 (4 - 5 nu) / (15 (1 - nu)).
   CLAIMS, symbolically for every isotropic background:
     [1] the isotropic average (3 T2g + 2 Eg)/5 of the cube's shear channels IS
         the sphere's -- the Sqrt[3] terms cancel;
     [2] the split S_diag - S_shear = (lam + mu)(5 Sqrt[3] - 2 Pi)/(6 Pi mu (lam + 2 mu)),
         positive for every admissible background (lam + mu > 0);
     [3] the bulk channel is the sphere's, 1 + dK/(lam + 2 mu), dK = dlam + 2 dmu/3;
     [C] control: T2g alone is NOT the sphere's (the check can fail).
   The Python half, through the production surface-constant route, is
   scripts/gate_cube_shear_split.py.
   ============================================================================ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeA22Block.wl"}]];

oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
Print[];
Print["==== CubeShearSplit :: the cube's shear channels against the sphere's ===="];

cA1g = proj[$A1gv]; cEg = proj[$Egv]; cT2g = proj[$T2gv];
sShear = Simplify[(cT2g - 1)/(2 dmu)];
sDiag = Simplify[(cEg - 1)/(2 dmu)];
nu = lam/(2 (lam + mu));
sSph = (4 - 5 nu)/(15 mu (1 - nu));             (* 2 mu S_sph = 2(4 - 5 nu)/(15(1 - nu)) *)
Print["  S_shear (T2g) = ", InputForm[sShear]];
Print["  S_diag  (Eg)  = ", InputForm[sDiag]];
Print["  S_sphere      = ", InputForm[Simplify[sSph]]];

Print["  [1] (3 S_shear + 2 S_diag)/5 == S_sphere: ",
  chk[Simplify[(3 sShear + 2 sDiag)/5 - sSph] === 0]];
Print["  [2] S_diag - S_shear == (lam+mu)(5 Sqrt[3] - 2 Pi)/(6 Pi mu (lam+2mu)): ",
  chk[Simplify[sDiag - sShear - (lam + mu) (5 Sqrt[3] - 2 Pi)/(6 Pi mu (lam + 2 mu))] === 0]];
Print["      and 5 Sqrt[3] - 2 Pi = ", N[5 Sqrt[3] - 2 Pi, 10], " > 0: ", chk[5 Sqrt[3] - 2 Pi > 0]];
Print["  [3] bulk channel == sphere's 1 + dK/(lam + 2 mu): ",
  chk[Simplify[cA1g - (1 + (dlam + 2 dmu/3)/(lam + 2 mu))] === 0]];
Print["  [C] control, T2g alone differs from the sphere: ",
  chk[Simplify[sShear - sSph] =!= 0]];
Print["  at the validated background (lam = 17.5, mu = 22.5): 2 mu S_shear = ",
  N[2 mu sShear /. {lam -> 35/2, mu -> 45/2}, 10], ",  2 mu S_diag = ",
  N[2 mu sDiag /. {lam -> 35/2, mu -> 45/2}, 10], ",  sphere = ",
  N[2 mu sSph /. {lam -> 35/2, mu -> 45/2}, 10]];

Print["==== CubeShearSplit: ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
