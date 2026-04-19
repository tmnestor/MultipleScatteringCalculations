(* MpTest.wl — Compute canonical master integral Mp[0,0,0] *)
(* Mp[p,q,r] = ∫₀¹ ∫₀¹ ∫₀¹ x^p y^q z^r / √(x²+y²+z²) dx dy dz *)

(* Symbolic evaluation *)
mpSymbolic = Integrate[1/Sqrt[x^2 + y^2 + z^2], {x, 0, 1}, {y, 0, 1}, {z, 0, 1}];
mpSimplified = FullSimplify[mpSymbolic];

(* Numerical cross-check *)
mpNumerical = NIntegrate[1/Sqrt[x^2 + y^2 + z^2],
  {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
  PrecisionGoal -> 16, WorkingPrecision -> 30];

(* TeXForm of the integral and result *)
texIntegral = ToString[HoldForm[Integrate[1/Sqrt[x^2 + y^2 + z^2],
  {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]], TeXForm];
texResult = ToString[mpSimplified, TeXForm];

(* Output *)
{texIntegral, texResult, N[mpSimplified, 20], mpNumerical}
