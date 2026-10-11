(* ::Package:: *)
(* InterfaceOmega2.wl -- the O(omega^2) term of the reflected and transmitted spectral kernels of a welded
   interface: the next term, after the static image, of their expansion at large lateral wavenumber.

   Paper: LatexPDFs/InterfaceCouplingIntegrals (asymptotic subtraction beyond the static image).
   Python twin: scripts/derive_interface_omega2.py (which also checks the result against the exact dynamic
   kernels at 50 digits).

   WHAT.  Medium A (lamA, muA, rhoA) fills z < 0, medium B (lamB, muB, rhoB) fills z > 0, welded at z = 0; a unit
   point force in A at depth zp < 0; frame x along the lateral wavevector (q, 0).  G = G0 + omega^2 G1 + ...,
   G0 the static kernels.  R1: both points in A (the reflected part).  T1: the receiver in B.

   HOW.  With u = u0 + omega^2 u1, the Navier equations div sigma + rho omega^2 u + F delta = 0 give
   div sigma(u1) = -rho u0 at order omega^2, with no source: u1 and its traction continuous at zp and at 0.
   In each region u0 = poly(z) e^{s q z}; u1 = a particular solution (poly of degree + 2, the double root) plus
   the homogeneous solutions regular there.  Kelvin's omega^2 term is solved first; it forces R1 and T1.

   CHECKS.  Identical media: R1 = 0 and T1 = Kelvin's omega^2 term.

   OUTPUT.  InterfaceOmega2.json: R1, T1 entries (InputForm, order rows u_(z,x,y), columns force (z,x,y)) and
   their values at the Python twin's test points.

   Run (in a notebook):  Get[SystemDialogInput["FileOpen"]]   (choose this file)
   or:                   wolframscript -file Mathematica/InterfaceOmega2.wl
*)

ClearAll["Global`*"];
dir = DirectoryName[$InputFileName];
$Assumptions = q > 0 && lamA > 0 && muA > 0 && rhoA > 0 && lamB > 0 && muB > 0 && rhoB > 0 && zp < 0;

(* displacement u = {ux, uy, uz} * e^{i q x} *)
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
homog[lam_, mu_, s_, cs_] := cs . basis[lam, mu, s];

polyPart[e_, s_] := Expand[Simplify[e Exp[-s q z]]];
particular[u0_, lam_, mu_, rho_, s_] := Module[{deg, n, p, ans, eqs, vars, sol},
  deg = Max[Exponent[polyPart[#, s], z] & /@ u0];
  n = deg + 3;
  ans = Table[Sum[p[k, i] z^i, {i, 0, n - 1}] Exp[s q z], {k, 3}];
  eqs = Flatten[CoefficientList[polyPart[#, s], z] & /@ (navier[ans, lam, mu] + rho u0)];
  vars = Flatten[Table[p[k, i], {k, 3}, {i, 0, n - 1}]];
  sol = First@Solve[eqs == 0, vars];
  Expand[ans /. sol /. (# -> 0 & /@ vars)]];

solveForce[f_] := Module[{F, kb, ka, below, above, k0b, k0a, r, t, r0, t0, eqs, sol, lb, la, k1b, k1a, s, w, r1, t1},
  F = UnitVector[3, f];
  kb = Array[kbb, 3]; ka = Array[kaa, 3];
  below = homog[lamA, muA, -1, kb]; above = homog[lamA, muA, 1, ka];
  eqs = Join[(below - above /. z -> zp), (traction[below, lamA, muA] - traction[above, lamA, muA] /. z -> zp) + F];
  sol = First@Solve[eqs == 0, Join[kb, ka]];
  k0b = Simplify[below /. sol]; k0a = Simplify[above /. sol];
  r = Array[rr, 3]; t = Array[tt, 3];
  r0 = homog[lamA, muA, 1, r]; t0 = homog[lamB, muB, -1, t];
  eqs = Join[(k0b + r0 - t0 /. z -> 0),
    (traction[k0b, lamA, muA] + traction[r0, lamA, muA] - traction[t0, lamB, muB] /. z -> 0)];
  sol = First@Solve[eqs == 0, Join[r, t]];
  r0 = Simplify[r0 /. sol]; t0 = Simplify[t0 /. sol];
  (* Kelvin's omega^2 term *)
  lb = Array[lbb, 3]; la = Array[laa, 3];
  k1b = particular[k0b, lamA, muA, rhoA, -1] + homog[lamA, muA, -1, lb];
  k1a = particular[k0a, lamA, muA, rhoA, 1] + homog[lamA, muA, 1, la];
  eqs = Join[(k1b - k1a /. z -> zp), (traction[k1b, lamA, muA] - traction[k1a, lamA, muA] /. z -> zp)];
  sol = First@Solve[eqs == 0, Join[lb, la]];
  k1b = Simplify[k1b /. sol];
  (* the reflected and transmitted omega^2 terms *)
  s = Array[ss, 3]; w = Array[ww, 3];
  r1 = particular[r0, lamA, muA, rhoA, 1] + homog[lamA, muA, 1, s];
  t1 = particular[t0, lamB, muB, rhoB, -1] + homog[lamB, muB, -1, w];
  eqs = Join[(k1b + r1 - t1 /. z -> 0),
    (traction[k1b, lamA, muA] + traction[r1, lamA, muA] - traction[t1, lamB, muB] /. z -> 0)];
  sol = First@Solve[eqs == 0, Join[s, w]];
  {Simplify[r1 /. sol], Simplify[t1 /. sol], k1b}];

res = solveForce /@ {1, 2, 3};
perm = {3, 1, 2};  (* (x, y, z) -> (z, x, y) *)
R1 = Transpose[res[[All, 1]]][[perm, perm]];
T1 = Transpose[res[[All, 2]]][[perm, perm]];
K1 = Transpose[res[[All, 3]]][[perm, perm]];
same = {lamB -> lamA, muB -> muA, rhoB -> rhoA};
Print["identical media, R1 = 0: ", Simplify[R1 /. same] === ConstantArray[0, {3, 3}]];
Print["identical media, T1 = Kelvin's omega^2 term: ", Simplify[(T1 /. same) - K1] === ConstantArray[0, {3, 3}]];

vars = {q, z, zp, lamA, muA, rhoA, lamB, muB, rhoB};
testsR = {{2.3, -0.4, -0.9, 1.7, 1.1, 2.5, 2.6, 1.9, 2.8}, {11.0, -0.05, -0.12, 0.6, 0.9, 2.2, 1.4, 2.2, 2.6}};
testsT = {{2.3, 0.4, -0.9, 1.7, 1.1, 2.5, 2.6, 1.9, 2.8}, {2.3, 0.4, -0.9, 0.6, 0.9, 2.2, 1.4, 2.2, 2.6}};
valR = Table[N[R1 /. Thread[vars -> SetPrecision[t, 30]], 20], {t, testsR}];
valT = Table[N[T1 /. Thread[vars -> SetPrecision[t, 30]], 20], {t, testsT}];

Export[FileNameJoin[{dir, "InterfaceOmega2.json"}],
  <|"order" -> "rows u_(z,x,y), columns force (z,x,y); frame x along the lateral wavevector; G = G0 + omega^2 G1",
    "R1" -> Map[ToString[#, InputForm] &, R1, {2}], "T1" -> Map[ToString[#, InputForm] &, T1, {2}],
    "tests_R" -> testsR, "tests_T" -> testsT,
    "R1_re" -> Re[valR], "R1_im" -> Im[valR], "T1_re" -> Re[valT], "T1_im" -> Im[valT]|>, "JSON"];
Print["written InterfaceOmega2.json"];
