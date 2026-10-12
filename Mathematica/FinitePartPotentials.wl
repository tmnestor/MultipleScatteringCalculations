(* ::Package:: *)
(* FinitePartPotentials.wl -- the potentials Phi_j of the interface kernels as Hadamard finite parts.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (12 October: K1 in space).
   Twin: scripts/derive_finite_part_potentials.py (sympy).

   WHAT.  The lateral transform of q^m Exp[-q zeta], zeta > 0, is
       (1/(2 Pi)) Integrate[q^(m+1) Exp[-q zeta] BesselJ[0, q rho], {q, 0, Infinity}]
         = Gamma[m + 2] R^(-m-2) LegendreP[m + 1, zeta/R] / (2 Pi),      R = Sqrt[rho^2 + zeta^2],
   for m > -2.  Phi_j belongs to m = -1 - j.  For j >= 1 Gamma has a pole there, and Phi_j is Hadamard's finite
   part: the coefficient of eps^0 in the Laurent series at m = -1 - j + eps (the analytic continuation in the
   power of q, the series' convention).  Here Mathematica takes that coefficient itself, with no formula for
   the degree derivative of LegendreP supplied, so the twin shares no step with the sympy route.

   CHECKS.  The ladder D[Phi_(j+1), zeta] == -Phi_j; harmonicity; the eps^0 coefficient against the sympy closed
   form (scripts/data/finite_part_potentials.json); the residue (eps^-1 coefficient) is the solid harmonic
   R^(j-1) LegendreP[j-1, zeta/R] times (-1)^(j-1)/((j-1)! 2 Pi).

   OUTPUT.  Mathematica/FinitePartPotentials.json: Phi_j, j = 0 .. 6, as InputForm strings in R, zeta.

   Run:  wolframscript -file Mathematica/FinitePartPotentials.wl
*)

ClearAll["Global`*"];
$Assumptions = rho > 0 && zeta > 0 && R > 0;
jMax = 6;

transform[m_] := Gamma[m + 2] R^(-m - 2) LegendreP[m + 1, zeta/R] / (2 Pi);

(* Hadamard's finite part at m = -1 - j: the eps^0 Laurent coefficient *)
fp[0] = 1/(2 Pi R);
fp[j_] := fp[j] = Module[{s},
   s = Series[transform[-1 - j + eps], {eps, 0, 0}];
   FullSimplify[SeriesCoefficient[s, 0] /. R -> Sqrt[rho^2 + zeta^2], Assumptions -> $Assumptions]];
residue[j_] := FullSimplify[SeriesCoefficient[Series[transform[-1 - j + eps], {eps, 0, 0}], -1]];

phis = Table[fp[j], {j, 0, jMax}];

Print["ladder d_zeta Phi_(j+1) + Phi_j:   ",
  Table[FullSimplify[D[phis[[j + 2]], zeta] + phis[[j + 1]]], {j, 0, jMax - 1}]];
Print["Laplacian (axisymmetric):          ",
  Table[FullSimplify[D[p, {rho, 2}] + D[p, rho]/rho + D[p, {zeta, 2}]], {p, Rest[phis]}]];
Print["residue - (-1)^n R^n P_n / (n! 2 Pi): ",
  Table[FullSimplify[residue[j] - (-1)^(j - 1) R^(j - 1) LegendreP[j - 1, zeta/R]/((j - 1)! 2 Pi)],
    {j, 1, jMax}]];

(* against the sympy twin *)
twin = Import[FileNameJoin[{DirectoryName[$InputFileName], "..", "scripts", "data",
     "finite_part_potentials.json"}], "RawJSON"];
Print["sympy twin - Mathematica, j = 0..6:   ",
  Table[FullSimplify[(ToExpression[twin["phi_wl"][[j + 1]]] /. R -> Sqrt[rho^2 + zeta^2]) - phis[[j + 1]]],
    {j, 0, jMax}]];
(* FullSimplify may leave Derivative[1, 0][LegendreP] unreduced; the numbers decide *)
Print["sympy twin - Mathematica, 30 digits, (rho, zeta) = (7/10, 9/20), (7/10, 13/10): ",
  Table[Max[Abs[N[((ToExpression[twin["phi_wl"][[j + 1]]] /. R -> Sqrt[rho^2 + zeta^2]) - phis[[j + 1]]) /.
       {rho -> 7/10, zeta -> #}, 30] & /@ {9/20, 13/10}]], {j, 0, jMax}]];

Export[FileNameJoin[{DirectoryName[$InputFileName], "FinitePartPotentials.json"}],
  <|"variables" -> {"rho", "zeta"},
    "definition" -> "Hadamard finite part (eps^0 Laurent coefficient at m = -1 - j) of the lateral transform of q^m Exp[-q zeta]",
    "phi" -> (ToString[#, InputForm] & /@ phis)|>, "JSON"];
Print["written Mathematica/FinitePartPotentials.json"];
