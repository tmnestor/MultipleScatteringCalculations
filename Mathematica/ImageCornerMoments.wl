(* ::Package:: *)
(* ImageCornerMoments.wl -- closed forms of the corner moments of the static interface image.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md, route A, the face closed forms.
   Python twin: cubic_scattering/image_moments.py, checked by scripts/derive_image_corner_moments.py.

   THE MOMENTS.  On the box [0,a] x [0,b] x [0,c] in (rho_x, rho_y, zeta) with the singular point at the
   corner,  M = Integrate[rho_x^p rho_y^q zeta^r D^alpha Phi_j],  Phi0 = 1/(2 Pi R),
   Phi1 = -Log[R + zeta]/(2 Pi), Phi2 = (zeta Log[R + zeta] - R)/(2 Pi).

   1. THE LINE OF IMAGES.  For lateral alpha (|alpha| >= 1 for Phi1, >= 2 for Phi2):
        D^alpha Phi1[rho, zeta] = Integrate[D^alpha Phi0[rho, t], {t, zeta, Infinity}],
        D^alpha Phi2[rho, zeta] = Integrate[(t - zeta) D^alpha Phi0[rho, t], {t, zeta, Infinity}].
      Checked symbolically below for a few alpha.

   2. THE TWO BASE INTEGRALS.  After every reduction (Euler's identity, finite parts, integration by parts)
      the Mindlin moments leave two integrals over the column [0,a] x [0,b] x [c, Infinity):
        B1 = Integrate[R^-3],   B2 = Integrate[x y R^-5].
      In polar coordinates the radial integrals are elementary, leaving one angular integral each:
        B1 = Sum over the two triangles of Integrate[Log[c + Sqrt[r^2 + c^2]] - Log[2 c], theta],
        B2 = Sum over the two triangles of Integrate[Cos Sin (G2[r] - G2[0]), theta],
             G2[r] = c/(3 Sqrt[r^2 + c^2]) + 2/3 Log[c + Sqrt[r^2 + c^2]],
      with r = a Sec[theta] on [0, ArcTan[b/a]] and r = b Csc[theta] on [ArcTan[b/a], Pi/2].
      For the cube, a = b = c (both are scale-free), they are
        B1 = (5/4) Cl2(Pi/3) - Catalan,          Cl2(x) = Im PolyLog[2, Exp[I x]],
        B2 = (1/3) Log[(1 + Sqrt[2])^2 / (2 (1 + Sqrt[3]))].
      B1: the two triangles are equal by symmetry, and Kummer's formula for Im Li2(r e^(i theta)), at
      r = 1/Sqrt[2] and theta = Pi/12, 7Pi/12 (both with omega = Pi/6), plus the duplication formula of Cl2,
      collapses Mathematica's dilogarithms.  B2: with v = Cos[theta]^2 the angular integral is elementary.
      Both are checked below against Mathematica's own reduction and against NIntegrate.

   3. REFERENCES.  A sample of corner moments by NIntegrate at 30 digits, among them the degree-zero
      Mindlin cases, for the twin to compare.

   OUTPUT.  ImageCornerMoments.json.

   Run (in a notebook):  Get[FileNameJoin[{NotebookDirectory[], "ImageCornerMoments.wl"}]]
   or:                   wolframscript -file Mathematica/ImageCornerMoments.wl
*)

ClearAll["Global`*"];
dir = DirectoryName[$InputFileName];
$Assumptions = x \[Element] Reals && y \[Element] Reals && zeta > 0 && t > 0;

R[x_, y_, z_] := Sqrt[x^2 + y^2 + z^2];
phi[0][x_, y_, z_] := 1/(2 Pi R[x, y, z]);
phi[1][x_, y_, z_] := -Log[R[x, y, z] + z]/(2 Pi);
phi[2][x_, y_, z_] := (z Log[R[x, y, z] + z] - R[x, y, z])/(2 Pi);
dphi[j_, {ax_, ay_, az_}] := D[phi[j][x, y, zeta], {x, ax}, {y, ay}, {zeta, az}];

(* 1. the line of images *)
lineCheck = Table[
   Module[{lhs, rhs},
    lhs = dphi[1, al];
    rhs = Integrate[(dphi[0, al] /. zeta -> t), {t, zeta, Infinity}];
    {al, Simplify[lhs - rhs] === 0}],
   {al, {{1, 0, 0}, {2, 0, 0}, {1, 1, 0}}}];
lineCheck2 = Table[
   Module[{lhs, rhs},
    lhs = dphi[2, al];
    rhs = Integrate[(t - zeta) (dphi[0, al] /. zeta -> t), {t, zeta, Infinity}];
    {al, Simplify[lhs - rhs] === 0}],
   {al, {{2, 0, 0}, {1, 1, 0}}}];
Print["line of images, Phi1: ", lineCheck];
Print["line of images, Phi2: ", lineCheck2];

(* 2. the base integrals *)
g2[r_, c_] := c/(3 Sqrt[r^2 + c^2]) + 2/3 Log[c + Sqrt[r^2 + c^2]];
polar[g_, a_, b_] := Integrate[g[th, a Sec[th]], {th, 0, ArcTan[b/a]}] +
   Integrate[g[th, b Csc[th]], {th, ArcTan[b/a], Pi/2}];
b1closed = TimeConstrained[
   Simplify[polar[Function[{th, r}, Log[c + Sqrt[r^2 + c^2]] - Log[2 c]], a, b] /. {a -> 2, b -> 2, c -> 2}],
   1800, $Failed];
b2closed = TimeConstrained[
   Simplify[polar[Function[{th, r}, Cos[th] Sin[th] (g2[r, c] - g2[0, c])], a, b] /. {a -> 2, b -> 2, c -> 2}],
   1800, $Failed];
(* The t integral exactly first: a direct three-dimensional NIntegrate over the semi-infinite column misses
   its precision goal silently (B2 was wrong at 1.7e-8 in the first run, B1 at 1.5e-15). *)
tB1 = Simplify[Integrate[(x^2 + y^2 + t^2)^(-3/2), {t, 2, Infinity}], x > 0 && y > 0];
tB2 = Simplify[Integrate[x y (x^2 + y^2 + t^2)^(-5/2), {t, 2, Infinity}], x > 0 && y > 0];
b1num = NIntegrate[tB1, {x, 0, 2}, {y, 0, 2}, WorkingPrecision -> 40, PrecisionGoal -> 30, MaxRecursion -> 40];
b2num = NIntegrate[tB2, {x, 0, 2}, {y, 0, 2}, WorkingPrecision -> 40, PrecisionGoal -> 30, MaxRecursion -> 40];
Print["B1 closed: ", b1closed, "  = ", If[b1closed === $Failed, "-", N[b1closed, 30]], "   NIntegrate: ", b1num];
Print["B2 closed: ", b2closed, "  = ", If[b2closed === $Failed, "-", N[b2closed, 30]], "   NIntegrate: ", b2num];
cl2[x_] := Im[PolyLog[2, Exp[I x]]];
b1simple = 5/4 cl2[Pi/3] - Catalan;
b2simple = Log[(1 + Sqrt[2])^2/(2 (1 + Sqrt[3]))]/3;
(* the angular integrals of B1 and B2 for the cube, exactly as the twin derives them *)
b1polar = 2 NIntegrate[Log[(1 + Sqrt[1 + Sec[th]^2])/2], {th, 0, Pi/4}, WorkingPrecision -> 60, PrecisionGoal -> 50];
b2v = Integrate[Sqrt[v/(v + 1)]/3 + 2/3 Log[1 + Sqrt[(v + 1)/v]] - 1/3 - 2/3 Log[2], {v, 1/2, 1}];
b1simpleCheck = {N[b1simple - b1polar, 50], N[b1simple - b1num, 40]};
b2simpleCheck = {N[b2simple - b2v, 50], FullSimplify[b2simple - b2v], N[b2simple - b2num, 40]};
Print["B1 = (5/4) Cl2(Pi/3) - Catalan = ", N[b1simple, 40], "   minus angular integral, minus NIntegrate: ", b1simpleCheck];
Print["B2 = (1/3) Log[(1+Sqrt2)^2/(2(1+Sqrt3))] = ", N[b2simple, 40], "   minus v integral (N, exact), minus NIntegrate: ", b2simpleCheck];

(* 3. reference corner moments: {j, alpha, {p, q, r}, signs {sx, sy}} on sides {2, 2, 2} *)
cases = {
   {0, {0, 0, 0}, {0, 0, 0}, {1, 1}},
   {0, {1, 1, 0}, {1, 0, 1}, {-1, 1}},
   {0, {0, 0, 2}, {1, 1, 2}, {1, 1}},
   {0, {2, 1, 1}, {1, 2, 3}, {1, -1}},
   {1, {1, 0, 0}, {0, 0, 1}, {1, 1}},
   {1, {2, 0, 0}, {0, 0, 0}, {1, 1}},       (* a degree-zero column *)
   {1, {2, 0, 0}, {0, 1, 1}, {1, -1}},
   {1, {2, 2, 0}, {2, 0, 2}, {1, -1}},
   {1, {3, 1, 0}, {1, 1, 0}, {-1, 1}},      (* a degree-zero column *)
   {2, {1, 1, 0}, {0, 0, 1}, {-1, -1}},
   {2, {2, 0, 0}, {0, 0, 0}, {1, 1}},       (* a degree-zero column *)
   {2, {4, 0, 0}, {0, 0, 2}, {1, 1}},
   {2, {2, 2, 0}, {1, 1, 1}, {1, 1}}};
(* The zeta integral exactly first, then two dimensions with a Duffy singularity handler at the corner: a
   direct three-dimensional NIntegrate with the singular point at a corner missed its precision goal silently
   in the first run (errors from 1e-12 to 5e-7 against three independent routes that agree to 30 digits). *)
refval[{j_, al_, {p_, q_, r_}, {sx_, sy_}}] := Module[{f, inner},
   f = dphi[j, al] /. {x -> sx u, y -> sy v};
   inner = Simplify[Integrate[zeta^r f, {zeta, 0, 2}, Assumptions -> u > 0 && v > 0], u > 0 && v > 0];
   NIntegrate[(sx u)^p (sy v)^q inner, {u, 0, 2}, {v, 0, 2},
    Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"},
    WorkingPrecision -> 40, PrecisionGoal -> 28, MaxRecursion -> 40]];
refs = refval /@ cases;
(* the simplest reference has an independent closed form: Paper 2's master integral, 1/(2 Pi) times
   int over [0,2]^3 of 1/R; graded_voxel.moments.box_integral gives 0.757602154836948200068335430129... *)
Print["reference 1 (Phi0, no derivative, s^0): ", N[refs[[1]], 30], "  (expected 0.757602154836948200068335430129)"];
Print["references done"];

Export[FileNameJoin[{dir, "ImageCornerMoments.json"}],
  <|"line_of_images_phi1" -> ToString[lineCheck], "line_of_images_phi2" -> ToString[lineCheck2],
    "B1_closed" -> ToString[b1closed, InputForm], "B2_closed" -> ToString[b2closed, InputForm],
    "B1" -> ToString[N[If[b1closed === $Failed, b1num, b1closed], 30], InputForm],
    "B2" -> ToString[N[If[b2closed === $Failed, b2num, b2closed], 30], InputForm],
    "B1_nintegrate" -> ToString[b1num, InputForm], "B2_nintegrate" -> ToString[b2num, InputForm],
    "B1_simple" -> ToString[N[b1simple, 50], InputForm], "B2_simple" -> ToString[N[b2simple, 50], InputForm],
    "B1_simple_check" -> ToString[b1simpleCheck, InputForm], "B2_simple_check" -> ToString[b2simpleCheck, InputForm],
    "cases" -> cases, "references" -> (ToString[#, InputForm] & /@ refs)|>, "JSON"];
Print["written ImageCornerMoments.json"];
