(* ::Package:: *)
(* StaticInterfaceTransmission.wl -- the elastostatic field transmitted through a welded interface,
   in the lateral-wavenumber domain and in space.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md, route A, the transmitted image.
   Python twins: scripts/derive_static_interface_transmission.py (spectral) and
   scripts/derive_interface_image_terms.py transmitted (spatial, as canonical terms).

   WHAT.  Medium A (lam_A, mu_A) fills z < 0, medium B (lam_B, mu_B) fills z > 0, welded at z = 0; a unit
   point force in A at zp < 0; the receiver in B at z > 0.  As in StaticInterfaceImage.wl: the field in A is
   Kelvin's plus a reflected part, the field in B is transmitted, u and the z-plane traction continuous.  The
   TRANSMITTED part GT[q; z, zp] (rows u_(z,x,y), columns force (z,x,y), frame x along the wavevector) is
   the whole static coupling across the interface.  Every entry is Exp[-q (z - zp)]/q times a polynomial in
   q z, q zp; for identical media it is Kelvin's field.

   IN SPACE.  With zeta = z - zp > 0 the transform rules of StaticImageSpace.wl hold unchanged:
   q^m Exp[-q zeta] -> (-D_zeta)^(m+1) Phi0, Phi1 (m = -2), Phi2 (m = -3).

   OUTPUT.  StaticInterfaceTransmission.json: spectral entries and their values at the Python twin's test
   points; spatial entries, an NIntegrate check, and spatial values at test points.

   Run (in a notebook):  Get[SystemDialogInput["FileOpen"]]   (choose this file)
   or:                   wolframscript -file Mathematica/StaticInterfaceTransmission.wl
*)

ClearAll["Global`*"];
dir = DirectoryName[$InputFileName];
$Assumptions = q > 0 && lamA > 0 && muA > 0 && lamB > 0 && muB > 0 && zp < 0 && z > 0;

stress[u_, lam_, mu_] := Module[{ux = u[[1]], uy = u[[2]], uz = u[[3]]},
  <|"zz" -> (lam + 2 mu) D[uz, z] + lam I q ux,
    "xz" -> mu (D[ux, z] + I q uz),
    "yz" -> mu D[uy, z],
    "xx" -> lam D[uz, z] + (lam + 2 mu) I q ux,
    "xy" -> mu I q uy|>];
navier[u_, lam_, mu_] := Module[{s = stress[u, lam, mu]},
  {I q s["xx"] + D[s["xz"], z], I q s["xy"] + D[s["yz"], z], I q s["xz"] + D[s["zz"], z]}];
traction[u_, lam_, mu_] := Module[{s = stress[u, lam, mu]}, {s["xz"], s["yz"], s["zz"]}];
basis[lam_, mu_, s_] := Module[{a, b, c, d, u, eqs, sol},
  u = {(a + b z) Exp[s q z], 0, (c + d z) Exp[s q z]};
  eqs = Simplify[navier[u, lam, mu] Exp[-s q z]];
  sol = First@Solve[Flatten[CoefficientList[#, z] & /@ eqs] == 0, {c, d}];
  Join[{u /. sol /. {a -> 1, b -> 0}, u /. sol /. {a -> 0, b -> 1}}, {{0, Exp[s q z], 0}}]];
combo[bs_, cs_] := cs . bs;

solve[f_] := Module[{F, cb, ca, cr, ct, below, above, kel, uK, uR, uT, eqs, sol},
  F = UnitVector[3, f];
  cb = Array[cbb, 3]; ca = Array[caa, 3]; cr = Array[crr, 3]; ct = Array[ctt, 3];
  below = combo[basis[lamA, muA, -1], cb]; above = combo[basis[lamA, muA, 1], ca];
  eqs = Join[(below - above /. z -> zp),
    (traction[below, lamA, muA] - traction[above, lamA, muA] /. z -> zp) + F];
  kel = First@Solve[eqs == 0, Join[cb, ca]];
  uK = below /. kel;
  uR = combo[basis[lamA, muA, 1], cr]; uT = combo[basis[lamB, muB, -1], ct];
  eqs = Join[(uK + uR - uT /. z -> 0),
    (traction[uK, lamA, muA] + traction[uR, lamA, muA] - traction[uT, lamB, muB] /. z -> 0)];
  sol = First@Solve[eqs == 0, Join[cr, ct]];
  {Simplify[uT /. sol], Simplify[uK]}];

res = solve /@ {1, 2, 3};
perm = {3, 1, 2};
GT = Transpose[res[[All, 1]]][[perm, perm]];
GK = Transpose[res[[All, 2]]][[perm, perm]];
Print["identical media give Kelvin's field: ", Simplify[(GT /. {lamB -> lamA, muB -> muA}) - GK] === ConstantArray[0, {3, 3}]];

testsSpec = {{2.3, 0.4, -0.9, 1.7, 1.1, 2.6, 1.9}, {11.0, 0.05, -0.12, 0.6, 0.9, 1.4, 2.2}};
valsSpec = Table[N[GT /. Thread[{q, z, zp, lamA, muA, lamB, muB} -> t], 20], {t, testsSpec}];

(* in space *)
X = Exp[-q (z - zp)];
qpoly[f_] := Module[{cl = CoefficientList[Expand[Simplify[q f/X]], q]},
  Association[Table[(n - 2) -> Simplify[cl[[n]]], {n, Length[cl]}]]];
R = Sqrt[rx^2 + ry^2 + zeta^2];
phi[0] = 1/(2 Pi R); phi[1] = -Log[R + zeta]/(2 Pi); phi[2] = (zeta Log[R + zeta] - R)/(2 Pi);
tr[m_] := If[m >= -1, (-1)^(m + 1) D[phi[0], {zeta, m + 1}], phi[-1 - m]];
tsum[assoc_, shift_] := Total[KeyValueMap[#2 tr[#1 + shift] &, assoc]];
{a, b, c, d, e} = {GT[[1, 1]], GT[[1, 2]], GT[[2, 1]], GT[[2, 2]], GT[[3, 3]]};
{pa, pb, pc, pd, pe} = qpoly /@ {a, b, c, d, e};
pdm = Merge[{pd, Map[Minus, pe]}, Total];
dh = {rx, ry};
GS = ConstantArray[0, {3, 3}];
GS[[1, 1]] = tsum[pa, 0];
Do[
  GS[[1, h + 1]] = D[tsum[Map[#/I &, pb], -1], dh[[h]]];
  GS[[h + 1, 1]] = D[tsum[Map[#/I &, pc], -1], dh[[h]]];
  Do[GS[[h + 1, h2 + 1]] = If[h == h2, tsum[pe, 0], 0] - D[tsum[pdm, -2], dh[[h]], dh[[h2]]], {h2, 2}],
  {h, 2}];

pars = {lamA -> 1.7, muA -> 1.1, lamB -> 2.6, muB -> 1.9};
{z0, zp0, rx0, ry0} = {0.3, -0.5, 0.4, -0.2};
specCart[qq_, ph_] := Module[{Q = {{1, 0, 0}, {0, Cos[ph], -Sin[ph]}, {0, Sin[ph], Cos[ph]}}},
  Q . (GT /. pars /. {q -> qq, z -> z0, zp -> zp0}) . Transpose[Q]];
fourier = Table[
   NIntegrate[qq Exp[I qq (Cos[ph] rx0 + Sin[ph] ry0)] specCart[qq, ph][[i, j]]/(4 Pi^2),
     {qq, 0, Infinity}, {ph, 0, 2 Pi}, WorkingPrecision -> 20, PrecisionGoal -> 12, MaxRecursion -> 20],
   {i, 3}, {j, 3}];
space = N[GS /. pars /. {rx -> rx0, ry -> ry0, zeta -> z0 - zp0, z -> z0, zp -> zp0}, 20];
fourierCheck = Max[Abs[fourier - space]]/Max[Abs[space]];
Print["spatial form against the Fourier integral: ", fourierCheck];

testsSpace = {{0.4, -0.2, 0.8, 0.3, -0.5, 1.7, 1.1, 2.6, 1.9}, {0.02, 0.03, 0.15, 0.1, -0.05, 0.6, 0.9, 1.4, 2.2}};
varsSpace = {rx, ry, zeta, z, zp, lamA, muA, lamB, muB};
valsSpace = Table[N[GS /. Thread[varsSpace -> t], 20], {t, testsSpace}];

Export[FileNameJoin[{dir, "StaticInterfaceTransmission.json"}],
  <|"order" -> "rows u_(z,x,y), columns force (z,x,y); source in A (zp < 0), receiver in B (z > 0)",
    "entries" -> Map[ToString[#, InputForm] &, GT, {2}],
    "tests" -> testsSpec, "values_re" -> Re[valsSpec], "values_im" -> Im[valsSpec],
    "space_entries" -> Map[ToString[#, InputForm] &, GS, {2}],
    "space_fourier_check" -> ToString[fourierCheck],
    "space_tests" -> testsSpace, "space_values_re" -> Re[valsSpace], "space_values_im" -> Im[valsSpace]|>, "JSON"];
Print["written StaticInterfaceTransmission.json"];
