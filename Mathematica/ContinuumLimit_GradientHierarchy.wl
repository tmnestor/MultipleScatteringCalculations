(* ============================================================================
   ContinuumLimit_GradientHierarchy.wl

   The single-site hierarchy in gradients about the centre, as a scheme on the layer.

   QUESTION.  The closed hierarchy of the single site expands the displacement of a voxel in a Taylor
   series about its centre and imposes the Lippmann-Schwinger equation and its derivatives AT the
   centre.  Truncated at the first gradient it is the uniform-strain closure.  What does each further
   gradient buy?

   MODEL.  The layer 0 < z < D at normal incidence, P: contrast a = om^2 drho in the density and b = dM
   in the modulus, a unit plane wave Exp[I k z] incident, n cells of half-width h.  In cell j
       u(z_j + xi) = Sum_{m <= q} U_j^(m) xi^m/m! ,     strain = its derivative,
   and  u = u0 + a A + b B',  A = Int g u,  B = Int g u',  g = I Exp[I k |z - z'|]/(2 M k).
   Derivatives of A and B above the first are reduced by  M g'' + rho om^2 g = -delta:
       A'' = -u/M - k^2 A ,   B'' = -u'/M - k^2 B   (the delta term kept).
   The cell integrals are done in closed form (exponential times monomial), exact arithmetic to 40
   digits.  This is independent of scripts/measure_layer_taylor_hierarchy.py, which uses Gauss
   quadrature in double precision.

   PREDICTION (parity of the first neglected power of the source):
       density only:  q = 1 -> 2,  2 -> 4,  3 -> 4,  4 -> 6,  5 -> 6,  6 -> 8
       modulus only:  q = 1 -> 2,  2 -> 2,  3 -> 4,  4 -> 4,  5 -> 6,  6 -> 6
       both:          as modulus only.
   So fourth order with both contrasts needs the third gradients (q = 3) and sixth order the fifth
   (q = 5).

   CHECKS.  [1] orders as predicted, R and T, all q, three contrasts;  [2] agreement with the Python
   implementation where it is above double-precision round-off;  [3] the leading error is a pure power:
   error/(k d)^order tends to a constant (printed, for the closed forms of the next step).
   ============================================================================ *)

base = DirectoryName[$InputFileName];
prec = 60;
al = 5000; rho = 2500; mP = rho al^2; dLayer = 2; zR = -1; zT = 4;
dRho0 = 100; dM0 = 4 10^9;
pass[b_] := If[TrueQ[b], "PASS", "****FAIL****"];
sci[x_] := Module[{e = Floor[Log10[Abs[N[x]]]]}, ToString[NumberForm[N[x]/10^e, {4, 2}]] <> "e" <> ToString[e]];
allOks = {};

(* the exact layer: continuity of u and M u' at both faces *)
exactScat[om_, drho_, dm_] := Module[{k0 = om/al, m1 = mP + dm, rho1 = rho + drho, k1, z0, z1, r, aa, bb, t, sol},
  k1 = om Sqrt[rho1/m1]; z0 = mP k0; z1 = m1 k1;
  sol = First@Solve[{1 + r == aa + bb, z0 (1 - r) == z1 (aa - bb),
      aa Exp[I k1 dLayer] + bb Exp[-I k1 dLayer] == t,
      z1 (aa Exp[I k1 dLayer] - bb Exp[-I k1 dLayer]) == z0 t}, {r, aa, bb, t}];
  N[{(r /. sol) Exp[-I k0 zR], (t /. sol) Exp[I k0 (zT - dLayer)] - Exp[I k0 zT]}, prec]];

(* closed-form integral of Exp[-I s k xi] xi^m/m! over (lo, hi), s = +1 or -1 *)
Do[fInt[m, s] = Function[{kk, lo, hi}, Evaluate[
     Integrate[Exp[-I s kq xq] xq^m/m!, {xq, loq, hiq}] /. {kq -> kk, loq -> lo, hiq -> hi}]],
  {m, 0, 6}, {s, {1, -1}}];

(* Int g(delta - xi) xi^m/m! and Int dg/dz(delta - xi) xi^m/m! over the cell (-h, h), m = 0..q *)
cellInts[kk_, h_, delta_, q_] := Module[{c = I/(2 mP kk), gi, di, pieces},
  pieces = Which[
    delta >= h, {{1, -h, h}},
    delta <= -h, {{-1, -h, h}},
    True, {{1, -h, delta}, {-1, delta, h}}];
  gi = Table[Sum[c Exp[I pc[[1]] kk delta] fInt[m, pc[[1]]][kk, pc[[2]], pc[[3]]], {pc, pieces}], {m, 0, q}];
  di = Table[Sum[I kk pc[[1]] c Exp[I pc[[1]] kk delta] fInt[m, pc[[1]]][kk, pc[[2]], pc[[3]]], {pc, pieces}], {m, 0, q}];
  {gi, di}];

solveHier[om_, n_, q_, drho_, dm_] := Module[
  {kk = om/al, a = om^2 drho, b = dm, d = dLayer/n, h, zc, nu = q + 1, big, rhs, sol, gi, di, aDer, bDer, own, locA, locB},
  h = d/2; zc = Table[(j - 1/2) d, {j, n}];
  big = N[IdentityMatrix[n nu], prec];
  Do[
   {gi, di} = N[cellInts[kk, h, zc[[i]] - zc[[j]], q], prec];
   own = If[i == j, 1, 0];
   aDer = {gi, di};
   bDer = {Join[{0}, gi[[1 ;; q]]], Join[{0}, di[[1 ;; q]]]};
   Do[
    locA = Table[If[mm == m - 2, own, 0], {mm, 0, q}];
    locB = Table[If[mm == m - 1, own, 0], {mm, 0, q}];
    AppendTo[aDer, -locA/mP - kk^2 aDer[[m - 1]]];
    AppendTo[bDer, -locB/mP - kk^2 bDer[[m - 1]]],
    {m, 2, q + 1}];
   Do[big[[(i - 1) nu + m + 1, (j - 1) nu + 1 ;; j nu]] -= a aDer[[m + 1]] + b bDer[[m + 2]], {m, 0, q}],
   {i, n}, {j, n}];
  rhs = N[Flatten[Table[(I kk)^m Exp[I kk zc[[i]]], {i, n}, {m, 0, q}]], prec];
  sol = Partition[LinearSolve[big, rhs], nu];
  Table[
   Sum[({gi, di} = N[cellInts[kk, h, zo - zc[[j]], q], prec];
     a gi . sol[[j]] + b di[[1 ;; q]] . sol[[j, 2 ;;]]), {j, n}],
   {zo, {zR, zT}}]];

Print["==== ContinuumLimit_GradientHierarchy :: the gradient hierarchy as a scheme on the layer ===="];
om = 300; ns = {2, 4, 8, 16, 32};
cases = {{"density only", dRho0, 0, <|1 -> 2, 2 -> 4, 3 -> 4, 4 -> 6, 5 -> 6, 6 -> 8|>},
   {"modulus only", 0, dM0, <|1 -> 2, 2 -> 2, 3 -> 4, 4 -> 4, 5 -> 6, 6 -> 6|>},
   {"both", dRho0, dM0, <|1 -> 2, 2 -> 2, 3 -> 4, 4 -> 4, 5 -> 6, 6 -> 6|>}};
results = <||>;
Do[
  Module[{label = cs[[1]], drho = cs[[2]], dm = cs[[3]], want = cs[[4]], ex, errs, ord, ok, kd},
   ex = exactScat[om, drho, dm];
   Print["  ", label, ": relative error of the scattered field {R, T}"];
   Do[
    errs = Table[Abs[(solveHier[om, n, q, drho, dm] - ex)/ex], {n, ns}];
    ord = Log[2, errs[[-2]]/errs[[-1]]];
    ok = AllTrue[ord, Abs[# - want[q]] < 0.3 &];
    AppendTo[allOks, ok];
    kd = (om/al) dLayer/Last[ns];
    Print["    q = ", q, ":  n = 16 ", sci /@ errs[[-2]], "   n = 32 ", sci /@ errs[[-1]],
     "   order ", ToString[NumberForm[N[#], 4] & /@ ord, OutputForm], "   predicted ", want[q], "   ", pass[ok],
     "   error/(k d)^order at n = 32: ", sci /@ (errs[[-1]]/kd^want[q])];
    results[label <> " q=" <> ToString[q]] = N[errs, 17],
    {q, 1, 6}]],
  {cs, cases}];
Print["  [1] every order as predicted (36 orders): ", pass[And @@ allOks]];

(* [2] the first-order factor in closed form.  A source held to its Taylor polynomial of degree dg about the
   cell centre radiates the exact first-order term times 1 + E, with t = k h:
       dg odd :  R and T alike   (-1)^((dg-1)/2) t^(dg+1)/(dg+2)!
       dg even:  R  (-1)^(dg/2) t^(dg+2)/(dg+2)! ,   T  (-1)^(dg/2+1) (dg+1) t^(dg+2)/(dg+3)!
   The source degree is q for the density term and q - 1 for the modulus term.  Derived here symbolically
   from  Int P_dg(I t x) Exp[-/+ I t x] dx  and compared with the scheme on one cell at a contrast of 10^-20,
   where the second-order term is far below every factor tested. *)
bornLead[dg_, t_] := If[OddQ[dg],
   {1, 1} (-1)^((dg - 1)/2) t^(dg + 1)/(dg + 2)!,
   {(-1)^(dg/2) t^(dg + 2)/(dg + 2)!, (-1)^(dg/2 + 1) (dg + 1) t^(dg + 2)/(dg + 3)!}];
symOk = AllTrue[Range[0, 6], Function[dg, Module[{pp, ft, fr, tt, xx, lr, lt, ordr},
     pp = Sum[(I tt xx)^m/m!, {m, 0, dg}];
     ft = Integrate[pp Exp[-I tt xx], {xx, -1, 1}]/2;
     fr = Integrate[pp Exp[I tt xx], {xx, -1, 1}]/Integrate[Exp[2 I tt xx], {xx, -1, 1}];
     ordr = If[OddQ[dg], dg + 1, dg + 2];
     lr = Normal[Series[fr - 1, {tt, 0, ordr}]]; lt = Normal[Series[ft - 1, {tt, 0, ordr}]];
     Simplify[{lr, lt} - bornLead[dg, tt]] === {0, 0}]]];
Print["  [2a] the closed-form leading factor is the series of the cell integral, source degrees 0..6: ", pass[symOk]];
AppendTo[allOks, symOk];
Module[{omB = 150, sc = 10^-20, t, okAll = True},
  t = (omB/al) dLayer/2;
  Do[
   Module[{drho = cs[[2]] sc, dm = cs[[3]] sc, shift = cs[[4]], ex, got, meas, pred, rel, ok},
    ex = exactScat[omB, drho, dm];
    Do[
     got = solveHier[omB, 1, q, drho, dm];
     meas = Re[got/ex - 1]; pred = N[bornLead[q - shift, t], 30];
     rel = Abs[(meas - pred)/pred]; ok = Max[rel] < 5/1000; okAll = okAll && ok;
     Print["    ", cs[[1]], " q = ", q, " (source degree ", q - shift, "):  R ", sci[meas[[1]]], " vs ", sci[pred[[1]]],
      "   T ", sci[meas[[2]]], " vs ", sci[pred[[2]]], "   ", pass[ok]],
     {q, 1, 6}]],
   {cs, {{"density", dRho0, 0, 0}, {"modulus", 0, dM0, 1}}}];
  Print["  [2b] scheme on one cell against the closed form, k h = ", N[t], ", to 0.5%: ", pass[okAll]];
  AppendTo[allOks, okAll]];

Export[base <> "ContinuumLimit_gradient_hierarchy.json",
  <|"omega" -> om, "n" -> ns, "errors_R_T" -> results|>, "JSON"];
Print["  exported ContinuumLimit_gradient_hierarchy.json"];
Print[If[And @@ allOks, "ALL PASS", "SOME FAILED"]];
