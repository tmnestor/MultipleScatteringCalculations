(* ::Package:: *)
(* StaticImageSpace.wl -- the elastostatic image of a welded interface, in space.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md, option 1, step 1 of the Mathematica phase.

   INPUT.  StaticInterfaceImage.json (from StaticInterfaceImage.wl): the reflected static kernel in the
   lateral-wavenumber domain, frame x along the wavevector, rows u_(z,x,y), columns force (z,x,y).  Every
   entry is  Exp[q (z + zp)]/q  times a polynomial in q z, q zp.

   THE TRANSFORM.  With zeta = -(z + zp) > 0 (both points in medium A, z < 0), rho the lateral separation and
   R = Sqrt[rho^2 + zeta^2]:
       (1 / 4 Pi^2) Integrate[Exp[I k.rho] q^m Exp[-q zeta], k in R^2] =
           (-D_zeta)^(m+1) Phi0     (m >= -1),     Phi1 (m = -2),     Phi2 (m = -3),
       Phi0 = 1/(2 Pi R),  Phi1 = -Log[R + zeta]/(2 Pi),  Phi2 = (zeta Log[R + zeta] - R)/(2 Pi),
   each the zeta-antiderivative of the one before (up to sign).  A factor I qhat_h becomes D_rho_h acting on
   the transform of (entry)/q, and qhat_h qhat_h' becomes -D_rho_h D_rho_h' on the transform of (entry)/q^2.
   In Cartesian form, with a, b, c, d, e the frame entries zz, zx', x'z, x'x', y'y':
       G_zz = a,  G_zh = b qhat_h,  G_hz = c qhat_h,  G_hh' = e delta_hh' + (d - e) qhat_h qhat_h'.
   Phi1 and Phi2 are the Mindlin-type terms (Rongved 1955): the constant q^-1 coefficient of b and c, and the
   q^-1, q^0 coefficients of d - e, are non-zero in general.

   CHECK.  At a test point the result is compared with NIntegrate of the 2-D Fourier integral of the
   spectral kernel (polar coordinates; the integrand decays as Exp[-q zeta]).

   OUTPUT.  StaticImageSpace.json: the nine spatial entries as InputForm strings, in the variables
   rho_x -> rx, rho_y -> ry, zeta, z, zp, and their values at the test points, for the Python twin
   scripts/derive_static_image_space.py.

   Run (in a notebook):  Get[FileNameJoin[{NotebookDirectory[], "StaticImageSpace.wl"}]]
   or:                   wolframscript -file Mathematica/StaticImageSpace.wl
*)

ClearAll["Global`*"];
dir = DirectoryName[$InputFileName];
spec = Import[FileNameJoin[{dir, "StaticInterfaceImage.json"}], "RawJSON"];
ent = Map[ToExpression, spec["entries"], {2}];
{a, b, c, d, e} = {ent[[1, 1]], ent[[1, 2]], ent[[2, 1]], ent[[2, 2]], ent[[3, 3]]};

X = Exp[q (z + zp)];
(* {power of q -> coefficient} of entry = X * sum coef q^power *)
qpoly[f_] := Module[{cl = CoefficientList[Expand[Simplify[q f/X]], q]},
  Association[Table[(n - 2) -> Simplify[cl[[n]]], {n, Length[cl]}]]];

R = Sqrt[rx^2 + ry^2 + zeta^2];
phi[0] = 1/(2 Pi R); phi[1] = -Log[R + zeta]/(2 Pi); phi[2] = (zeta Log[R + zeta] - R)/(2 Pi);
tr[m_] := If[m >= -1, (-1)^(m + 1) D[phi[0], {zeta, m + 1}], phi[-1 - m]];
tsum[assoc_, shift_] := Total[KeyValueMap[#2 tr[#1 + shift] &, assoc]];

{pa, pb, pc, pd, pe} = qpoly /@ {a, b, c, d, e};
pdm = Merge[{pd, Map[Minus, pe]}, Total];
dh = {rx, ry};

G = ConstantArray[0, {3, 3}];
G[[1, 1]] = tsum[pa, 0];
Do[
  G[[1, h + 1]] = D[tsum[Map[#/I &, pb], -1], dh[[h]]];
  G[[h + 1, 1]] = D[tsum[Map[#/I &, pc], -1], dh[[h]]];
  Do[G[[h + 1, h2 + 1]] = If[h == h2, tsum[pe, 0], 0] - D[tsum[pdm, -2], dh[[h]], dh[[h2]]], {h2, 2}],
  {h, 2}];

(* check against the 2-D Fourier integral at one point *)
pars = {lamA -> 1.7, muA -> 1.1, lamB -> 2.6, muB -> 1.9};
{z0, zp0, rx0, ry0} = {-0.3, -0.5, 0.4, -0.2};
specCart[qq_, ph_] := Module[{Q = {{1, 0, 0}, {0, Cos[ph], -Sin[ph]}, {0, Sin[ph], Cos[ph]}}},
  Q . (ent /. pars /. {q -> qq, z -> z0, zp -> zp0}) . Transpose[Q]];
fourier = Table[
   NIntegrate[qq Exp[I qq (Cos[ph] rx0 + Sin[ph] ry0)] specCart[qq, ph][[i, j]]/(4 Pi^2),
     {qq, 0, Infinity}, {ph, 0, 2 Pi}, WorkingPrecision -> 20, PrecisionGoal -> 12, MaxRecursion -> 20],
   {i, 3}, {j, 3}];
space = N[G /. pars /. {rx -> rx0, ry -> ry0, zeta -> -(z0 + zp0), z -> z0, zp -> zp0}, 20];
Print["spatial form against the Fourier integral: ", Max[Abs[fourier - space]]/Max[Abs[space]]];

tests = {{0.4, -0.2, 0.8, -0.3, -0.5, 1.7, 1.1, 2.6, 1.9}, {0.02, 0.03, 0.15, -0.1, -0.05, 0.6, 0.9, 1.4, 2.2}};
vars = {rx, ry, zeta, z, zp, lamA, muA, lamB, muB};
vals = Table[N[G /. Thread[vars -> t], 20], {t, tests}];
Export[FileNameJoin[{dir, "StaticImageSpace.json"}],
  <|"order" -> "rows u_(z,x,y), columns force (z,x,y), package axes; zeta = -(z + zp)",
    "entries" -> Map[ToString[#, InputForm] &, G, {2}],
    "test_args" -> ToString /@ vars, "tests" -> tests,
    "values_re" -> Re[vals], "values_im" -> Im[vals]|>, "JSON"];
Print["written StaticImageSpace.json"];
