#!/usr/bin/env wolframscript
(* Export the exact m = 1 multipole fields, for the spectrum probe to score
   against.

   DIVISION OF LABOUR.  Mathematica differentiates the potentials exactly at a
   few points, which is cheap.  The plane-wave quadrature that the probe needs
   is tens of thousands of nodes and belongs in vectorised numerics, not in a
   nested Do loop over exact rationals -- a first attempt written that way was
   still producing nothing after several minutes, because every node was being
   evaluated symbolically.

   Run:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript -file \
       Mathematica/exportMOneReference.wl
*)

rX[x_, y_, z_] := Sqrt[x^2 + y^2 + z^2];

(* m = 0 and m = 1 angular factors, as in MieSphericalWaves.wl Section 2.
   The m = 1 form is P_n^1(cos theta) cos(phi) = -P_n'(z/r) (x/r), which is
   polynomial in z/r and regular on the axis. *)
angF[n_, 0][x_, y_, z_] := LegendreP[n, z/rX[x, y, z]];
angF[n_, 1][x_, y_, z_] :=
  -(D[LegendreP[n, u], u] /. u -> z/rX[x, y, z]) (x/rX[x, y, z]);

radF[n_, k_][x_, y_, z_] := SphericalHankelH1[n, k rX[x, y, z]];
potF[n_, m_, k_][x_, y_, z_] := radF[n, k][x, y, z] angF[n, m][x, y, z];

uP[n_, m_, k_][x_, y_, z_] := Grad[potF[n, m, k][x, y, z], {x, y, z}];
uSV[n_, m_, k_][x_, y_, z_] :=
  Curl[Curl[{x, y, z} potF[n, m, k][x, y, z], {x, y, z}], {x, y, z}];
uSH[n_, m_, k_][x_, y_, z_] :=
  Curl[{x, y, z} potF[n, m, k][x, y, z], {x, y, z}];

(* Seismic units, matching the Python probe: alpha 5, beta 3, rho 2.5, and a
   frequency giving ka = 0.3 on a radius 10 sphere. *)
$beta = 3; $alpha = 5; $rad = 10; $ka = 3/10;
$om = $ka $beta/$rad;
$kP = $om/$alpha; $kS = $om/$beta;

pts = {{3, -2, -12}, {-8, 5, 14}, {40, 25, -12}, {6, 11, -20}};

out = {};
Do[
  Module[{k, u, x, y, z},
    k = If[fam === "P", $kP, $kS];
    u = Switch[fam, "P", uP[n, m, k][x, y, z], "SV", uSV[n, m, k][x, y, z],
         "SH", uSH[n, m, k][x, y, z]];
    Do[
      AppendTo[out, <|
        "family" -> fam, "n" -> n, "m" -> m, "point" -> N[p, 20],
        "u" -> (ReIm /@ N[u /. Thread[{x, y, z} -> p], 25])|>],
      {p, pts}]],
  {fam, {"P", "SV", "SH"}}, {m, {0, 1}}, {n, 1, 3}];

Export["Mathematica/MOneReference.json",
  <|"note" -> "Cartesian (x,y,z), wave along +z.  Python orders axes (z,x,y).",
    "alpha" -> N[$alpha, 20], "beta" -> N[$beta, 20],
    "omega" -> N[$om, 20], "kP" -> N[$kP, 20], "kS" -> N[$kS, 20],
    "cases" -> out|>];

Print["wrote ", Length[out], " cases to Mathematica/MOneReference.json"];
Print["  kP = ", N[$kP, 10], "   kS = ", N[$kS, 10]];
