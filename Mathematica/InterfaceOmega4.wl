(* ::Package:: *)
(* InterfaceOmega4.wl -- the O(omega^4) term of the reflected and transmitted spectral kernels of a welded
   interface, with symbolic Lame constants and densities.

   Plan: docs/2026-10-10-stratified-reference-legendre-cells-3d.md (13 October: the remainder, its tail by omega^4).
   Python twin: scripts/derive_interface_omega4.py (exact rational media; checked against the exact dynamic kernels
   at 50 digits, the residual falling 64-fold per halving of omega).  Order omega^2: InterfaceOmega2.wl.

   WHAT.  G = G0 + omega^2 G1 + omega^4 G2 + ..., at large lateral wavenumber q.  The perturbation of the static
   problem continued: div sigma(u_n) = -rho u_(n-1), no source, u_n and its traction continuous at zp (Kelvin's
   term) and at z = 0 (the reflected R_n in A and the transmitted T_n in B).  Each u_n is a polynomial two degrees
   higher than u_(n-1) times the same exponential, plus the regular homogeneous solutions.

   CHECKS.  Identical media: R_n = 0 and T_n = Kelvin's term, n = 1, 2.  The Python twin compares R2, T2 with
   the values exported here at its media.

   OUTPUT.  InterfaceOmega4.json: R2, T2 (InputForm; rows u_(z,x,y), columns force (z,x,y)); their values at the
   gates' media (A: alpha 5, beta 3, rho 2.5; B: 6.5, 3.6, 2.8) at test points.

   Run:  wolframscript -file Mathematica/InterfaceOmega4.wl       (expect tens of minutes: the Simplify steps)
*)

ClearAll["Global`*"];
dir = DirectoryName[$InputFileName];
$Assumptions = q > 0 && lamA > 0 && muA > 0 && rhoA > 0 && lamB > 0 && muB > 0 && rhoB > 0 && zp < 0;
nOrder = 2;

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

(* {R_0..R_n, T_0..T_n, K_0..K_n} columns for a unit force along f *)
solveForce[f_] := Module[{F, kb, ka, below, above, eqs, sol, kB, kA, R, T, r, t, cr, ct, lb, la, nb, na},
  F = UnitVector[3, f];
  kb = Array[kbb, 3]; ka = Array[kaa, 3];
  below = homog[lamA, muA, -1, kb]; above = homog[lamA, muA, 1, ka];
  eqs = Join[(below - above /. z -> zp), (traction[below, lamA, muA] - traction[above, lamA, muA] /. z -> zp) + F];
  sol = First@Solve[eqs == 0, Join[kb, ka]];
  kB = {Simplify[below /. sol]}; kA = {Simplify[above /. sol]};
  R = {}; T = {};
  Do[
    If[n > 0,
      lb = Array[lbb[n], 3]; la = Array[laa[n], 3];
      nb = particular[kB[[-1]], lamA, muA, rhoA, -1] + homog[lamA, muA, -1, lb];
      na = particular[kA[[-1]], lamA, muA, rhoA, 1] + homog[lamA, muA, 1, la];
      eqs = Join[(nb - na /. z -> zp), (traction[nb, lamA, muA] - traction[na, lamA, muA] /. z -> zp)];
      sol = First@Solve[eqs == 0, Join[lb, la]];
      AppendTo[kB, Simplify[nb /. sol]]; AppendTo[kA, Simplify[na /. sol]]];
    cr = Array[rr[n], 3]; ct = Array[tt[n], 3];
    r = homog[lamA, muA, 1, cr]; t = homog[lamB, muB, -1, ct];
    If[n > 0, r += particular[R[[-1]], lamA, muA, rhoA, 1]; t += particular[T[[-1]], lamB, muB, rhoB, -1]];
    eqs = Join[(kB[[n + 1]] + r - t /. z -> 0),
      (traction[kB[[n + 1]], lamA, muA] + traction[r, lamA, muA] - traction[t, lamB, muB] /. z -> 0)];
    sol = First@Solve[eqs == 0, Join[cr, ct]];
    AppendTo[R, Simplify[r /. sol]]; AppendTo[T, Simplify[t /. sol]],
    {n, 0, nOrder}];
  {R, T, kB}];

res = solveForce /@ {1, 2, 3};
perm = {3, 1, 2};  (* (x, y, z) -> (z, x, y) *)
mat[which_, n_] := Transpose[res[[All, which, n + 1]]][[perm, perm]];
R2 = mat[1, 2]; T2 = mat[2, 2]; K2 = mat[3, 2];
same = {lamB -> lamA, muB -> muA, rhoB -> rhoA};
Do[Print["identical media, order ", n, ": R = 0 ", Simplify[mat[1, n] /. same] === ConstantArray[0, {3, 3}],
    ", T = Kelvin ", Simplify[(mat[2, n] /. same) - mat[3, n]] === ConstantArray[0, {3, 3}]], {n, 1, nOrder}];

(* values at the gates' media: lam = rho (alpha^2 - 2 beta^2), mu = rho beta^2 *)
gate = {lamA -> 5/2 (25 - 18), muA -> 5/2 9, rhoA -> 5/2,
        lamB -> 14/5 (169/4 - 2 (18/5)^2), muB -> 14/5 (18/5)^2, rhoB -> 14/5};
pts = {{3, -2/5, -7/10}, {7/10, -1/20, -13/10}};
ptsT = {{3, 2/5, -7/10}, {7/10, 11/10, -1/5}};
valR = Table[N[R2 /. gate /. Thread[{q, z, zp} -> p], 30], {p, pts}];
valT = Table[N[T2 /. gate /. Thread[{q, z, zp} -> p], 30], {p, ptsT}];

Export[FileNameJoin[{dir, "InterfaceOmega4.json"}],
  <|"order" -> "rows u_(z,x,y), columns force (z,x,y); frame x along the lateral wavevector; G = G0 + w^2 G1 + w^4 G2",
    "R2" -> Map[ToString[#, InputForm] &, R2, {2}], "T2" -> Map[ToString[#, InputForm] &, T2, {2}],
    "points_R" -> pts, "points_T" -> ptsT,
    "R2_gate_re" -> Map[ToString[#, InputForm] &, Re[valR], {3}], "R2_gate_im" -> Map[ToString[#, InputForm] &, Im[valR], {3}],
    "T2_gate_re" -> Map[ToString[#, InputForm] &, Re[valT], {3}], "T2_gate_im" -> Map[ToString[#, InputForm] &, Im[valT], {3}]|>,
  "JSON"];
Print["written InterfaceOmega4.json"];
