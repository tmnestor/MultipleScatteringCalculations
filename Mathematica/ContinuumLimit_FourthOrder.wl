#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_FourthOrder.wl  --  notebook 7 of the continuum-limit study.

   THE TARGET.  Notebook 6: a uniform-field single site stops at second order,
   with the closed-form error (k d)^2/8 in reflection and -(k d)^2/24 in
   transmission, and no scalar correction removes both.  What is missing is the
   variation of the field ACROSS each voxel.  Here each voxel carries its mean
   AND its first moment, and the discrete equations are the continuum equation
   TESTED with the same two functions (Galerkin, Legendre P0 and P1 per cell),
   the kernel entering through exact cell-to-cell double integrals.  For
   functionals such as the reflected field, Galerkin with degree-p functions
   converges at order 2p + 2: fourth order for p = 1.

   THE 1-D MODEL, from scratch (normal-incidence P).  Field w = (u, e), e = du/dz;
   polarisation q = (w^2 drho u, dM e), dM = dlambda + 2 dmu; the continuum
   Lippmann-Schwinger equation
       w(z) = w0(z) + int K(z - z') q(z') dz',   K = {{g, g'}, {g', g''}},
       g(z) = i e^{i k |z|}/(2 M k),  g'' = -k^2 g - delta(z)/M,
   derivatives with respect to the receiver z (the 9x9 kernel's P channel,
   notebooks 2a and 3).  M = lambda + 2 mu, k = w/alpha.

   SCHEMES (unknowns per cell):
     (C)  collocation at the cell centre, piecewise-constant field -- must
          reproduce notebook 5's collocation (check of this model);
     (G0) Galerkin, cell mean only;
     (G1) Galerkin, cell mean and first Legendre moment -- the target.
   Observables: the scattered u above (reflection) and below (transmission)
   the layer, against the exact layer (notebook 2c's closed form).
   Coordinates z down; e^{-i w t}; SI units.
   ============================================================================ *)

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);

{al, be, rho} = {5000, 3000, 2500};
mu = rho be^2; mP = rho al^2; lam = mP - 2 mu;
{dLam, dMu, dRho} = {2 10^9, 1 10^9, 100};
dM = dLam + 2 dMu;
dLayer = 2; zSrc = -12; zObsR = -1; zObsT = 4;

(* ---- exact layer (notebook 2c) ---- *)
rtExact[om_] := Module[{k0 = om/al, m1 = mP + dM, r1 = rho + dRho, k1, rr, tt, bb, cc},
   k1 = om Sqrt[r1/m1];
   {rr, tt} /. First@Solve[{1 + rr == bb + cc, mP I k0 (1 - rr) == m1 I k1 (bb - cc),
       bb Exp[I k1 dLayer] + cc Exp[-I k1 dLayer] == tt, m1 I k1 (bb Exp[I k1 dLayer] - cc Exp[-I k1 dLayer]) == mP I k0 tt},
      {rr, tt, bb, cc}]];
gInc[om_, z_] := I/(2 mP (om/al)) Exp[I (om/al) Abs[z - zSrc]];
exactScat[om_] := Module[{k0 = om/al, rt = rtExact[om]},
   {rt[[1]] gInc[om, 0] Exp[-I k0 zObsR], rt[[2]] gInc[om, 0] Exp[I k0 (zObsT - dLayer)] - gInc[om, zObsT]}];

(* ---- kernel pieces, symbolic in k (h = half cell) ---- *)
Clear[s, t, k, h, zc];
(* for receiver above (sg = -1) or below (sg = +1) the source, g = I/(2 mP k) e^{i k sg (z - z')}:
   K = gfac {{1, i k sg}, {i k sg, -k^2}} e^{i k sg (z - z')} *)
kMat[kk_, sg_] := (I/(2 mP kk)) {{1, I kk sg}, {I kk sg, -kk^2}};
leg = {1 &, (#/h) &};  (* Legendre P0, P1 on the cell, local coordinate *)
(* int_{-h}^{h} phi_a(s) e^{i c s} ds, closed form *)
mom[a_, c_] := Integrate[leg[[a]][s] Exp[I c s], {s, -h, h}];
momF = Table[Function[{cc, hh}, Evaluate[mom[a, cc] /. h -> hh]], {a, 2}];
(* same-cell double integral int int phi_a(s) K(s - t) phi_b(t) ds dt, by splitting at t = s; plus the delta *)
selfBlock[a_, b_] := Module[{gA, gB, full},
   gA = Integrate[leg[[a]][s] leg[[b]][t] Exp[I k (s - t)], {t, -h, s}];   (* receiver below the source: s > t *)
   gB = Integrate[leg[[a]][s] leg[[b]][t] Exp[-I k (s - t)], {t, s, h}];   (* receiver above: s < t *)
   full = Integrate[#, {s, -h, h}] & /@ {gA, gB};
   (I/(2 mP k)) ({{1, I k}, {I k, -k^2}} full[[1]] + {{1, -I k}, {-I k, -k^2}} full[[2]]) +
    {{0, 0}, {0, -Integrate[leg[[a]][s] leg[[b]][s], {s, -h, h}]/mP}}];
selfF = Table[Function[{kk, hh}, Evaluate[Simplify[selfBlock[a, b]] /. {k -> kk, h -> hh}]], {a, 2}, {b, 2}];

(* ---- the discrete solve: nb basis functions per cell (1 or 2), Galerkin or centre collocation ---- *)
solveLayer[om_, n_, nb_, scheme_] := Module[{kk = om/al, d = dLayer/n, hh, zs, dQ, nU, blk, rhs, mat, sol, obs, cell},
   hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
   dQ = DiagonalMatrix[{om^2 dRho, dM}];
   nU = 2 nb;
   (* block (i, a | j, b): the equation of cell i tested with phi_a, from cell j's basis phi_b *)
   blk[i_, j_, a_, b_] := Which[
     scheme == "C",   (* collocation at the centre: the kernel integrated over cell j at z_i *)
     If[i == j,
      (* plate term: int_{-h}^{h} K(-t) dt, value at centre: odd entries cancel, delta gives -1/M *)
      {{(I/(2 mP kk)) 2 (Exp[I kk hh] - 1)/(I kk), 0}, {0, -kk^2 (I/(2 mP kk)) 2 (Exp[I kk hh] - 1)/(I kk) - 1/mP}},
      With[{sg = Sign[zs[[i]] - zs[[j]]]}, kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[1]][-kk sg, hh]]],
     i == j, selfF[[a, b]][kk, hh],
     True, With[{sg = Sign[zs[[i]] - zs[[j]]]},
      kMat[kk, sg] Exp[I kk sg (zs[[i]] - zs[[j]])] momF[[a]][kk sg, hh] momF[[b]][-kk sg, hh]]];
   (* normalisation of the test functions: int phi_a^2 = 2h {1, 1/3} *)
   cell[i_, a_] := If[scheme == "C", 1, 2 hh {1, 1/3}[[a]]];
   mat = ArrayFlatten[Table[
      ArrayFlatten[Table[
        If[i == j && a == b, cell[i, a] IdentityMatrix[2], 0] - blk[i, j, a, b] . dQ, {a, nb}, {b, nb}]],
      {i, n}, {j, n}]];
   rhs = Flatten[Table[
      If[scheme == "C",
       {gInc[om, zs[[i]]], I kk gInc[om, zs[[i]]]},
       gInc[om, zs[[i]]] momF[[a]][kk, hh] {1, I kk}], {i, n}, {a, nb}]];
   sol = LinearSolve[N[mat, 40], N[rhs, 40]];
   (* scattered u at the observers: sum_j int_cell K(z_o - z') q(z') dz' *)
   obs[zo_] := Sum[With[{sg = Sign[zo - zs[[j]]]},
      Sum[((kMat[kk, sg] Exp[I kk sg (zo - zs[[j]])] momF[[b]][-kk sg, hh]) . dQ .
          sol[[nU (j - 1) + 2 (b - 1) + 1 ;; nU (j - 1) + 2 b]])[[1]], {b, nb}]], {j, n}];
   {obs[zObsR], obs[zObsT]}];

Print["==== ContinuumLimit_FourthOrder :: a voxel that carries its first moment ===="];
om = 300;
ex = N[exactScat[om], 40];
ladder = {1, 2, 4, 8, 16};
runs = Association[Table[sc -> Table[Abs[(solveLayer[om, n, If[sc == "G1", 2, 1], If[sc == "C", "C", "G"]] - ex)/ex], {n, ladder}],
    {sc, {"C", "G0", "G1"}}]];
Print["  omega = ", om, " rad/s, D = ", dLayer, " m, the gate's contrast; |error| / |scattered|, {reflection, transmission}:"];
Do[Print["   n = ", StringPadLeft[ToString[ladder[[i]]], 2], ":  (C) ", sci /@ runs["C"][[i]], "   (G0) ", sci /@ runs["G0"][[i]],
   "   (G1) ", sci /@ runs["G1"][[i]]], {i, Length[ladder]}];
order[sc_, col_] := N[Log[2, runs[sc][[-3, col]]/runs[sc][[-1, col]]]/2];
Print["  observed order (n = 4 -> 16), reflection / transmission:"];
Do[Print["      ", sc, ":  ", ToString[NumberForm[order[sc, 1], 3], OutputForm], " / ", ToString[NumberForm[order[sc, 2], 3], OutputForm]],
  {sc, {"C", "G0", "G1"}}];
(* (C) must be notebook 5's collocation: its reflection error at n = 16, omega = 300 was 6.99e-6 *)
Print["  [1] (C) reproduces notebook 5's collocation (reflection, n = 16: 6.99e-6): ", sci[runs["C"][[-1, 1]]], " -> ",
  chk[Abs[runs["C"][[-1, 1]]/6.99*^-6 - 1] < 0.01]];
Print["  [2] (G1) converges at fourth order in reflection AND transmission: ",
  chk[3.7 < order["G1", 1] < 4.3 && 3.7 < order["G1", 2] < 4.3]];

(* the order at the other frequencies (k_S a at n = 1: 0.02 and 0.2) *)
Print["  (G1) at other frequencies, n = 4 -> 16:"];
ordOK = Table[Module[{exO = N[exactScat[om2], 40], e},
     e = Table[Abs[(solveLayer[om2, n, 2, "G"] - exO)/exO], {n, {4, 8, 16}}];
     Print["      omega ", om2, ":  errors n = 4, 8, 16 ", Map[sci, e, {2}], "  orders ",
      ToString[NumberForm[N[Log[2, e[[1]]/e[[3]]]/2], 3], OutputForm]];
     AllTrue[N[Log[2, e[[1]]/e[[3]]]/2], 3.7 < # < 4.3 &]], {om2, {60, 600}}];
Print["  [3] fourth order at every frequency: ", chk[And @@ ordOK]];

Print["==== ContinuumLimit_FourthOrder (stage 7): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
