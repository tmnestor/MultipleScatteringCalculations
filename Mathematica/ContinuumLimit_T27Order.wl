#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_T27Order.wl  --  notebook 11 of the continuum-limit study.

   THE QUESTION.  Notebook 7 reached fourth order with a voxel carrying the mean
   and first moment of (u, e) as INDEPENDENT fields.  Does the 27-mode Galerkin
   single site with a matched propagator reach it too?  Its basis is a SINGLE
   displacement field, polynomial up to quadratic per cell, with the strain
   DERIVED from it.  At normal incidence (P) that is, per cell,
       u(s) = sum_b c_b P_b(s/h),   b = 0, 1, 2   (P2: the 1-D T27 content),
       e(s) = u'(s),
   and the scheme is the Galerkin projection of the u-row of the continuum
   Lippmann-Schwinger equation (notebook 7's model):
       u(z) = u0(z) + int [ g(z - z') w^2 drho u(z') + g'(z - z') dM e(z') ] dz'
   tested with the same P_b, coupled through exact cell-to-cell double integrals
   (the matched propagator).

   TWO STRAINS.  A displacement that is discontinuous across cells has a
   distributional derivative with a delta at every interior interface:
     "cell"  e = u' inside each cell only -- the 27-mode single site's own
             source, int_cell dc : grad u;
     "dist"  e = u' + sum_I [u]_I delta(z - z_I) -- the interface jumps kept.

   PREDICTION (Born order, stated before running).  With the cellwise strain the
   stress term, integrated by parts, leaves boundary terms e^{ikz}(Pi u0 - u0) at
   every interface.  The L2-projection error at the endpoints is O(h^{p+1}) and
   does not cancel between neighbours for p = 2, so a STIFFNESS contrast stays
   SECOND order whatever p; a DENSITY contrast converges at 2p + 2.

   INTERPRETATION (checks [5]-[6]).  The voxels of a uniform layer are
   homogeneous: dM is the same on both sides of every interior interface, so the
   continuum has no source there and the surface terms of neighbouring cells
   cancel.  The cellwise scheme breaks that, because each voxel's displacement is
   projected independently and jumps by O(h^3) at every interior interface; those
   spurious interface sources ARE its second-order error.  Keeping the jumps
   (distributional strain) restores the cancellation exactly, leaving only the
   two outer faces, where dM does jump: O(h^{p+1}), third order for P2.

   CHECKS: [1] with dM = 0, the P1 scheme is notebook 7's G1 exactly; then the
   observed orders, by channel (density only, stiffness only, both); [5] at Born
   order the solves equal an independent bulk + interface + face decomposition;
   [6] the order of each part.
   ============================================================================ *)

nbText = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_FourthOrder.wl", "Text"];
ToExpression[StringTake[nbText, StringPosition[nbText, "Print[\"==== ContinuumLimit_FourthOrder ::"][[1, 1]] - 1],
  InputForm];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_T27Order :: the 27-mode (displacement-only) Galerkin, and its order ===="];

Clear[s, t, k, h];
legT = {1 &, (#/h) &, ((3 (#/h)^2 - 1)/2) &};
dlegT = {0 &, (1/h) &, (3 #/h^2) &};
mT[a_, c_] := Integrate[legT[[a]][s] Exp[I c s], {s, -h, h}];
dT[b_, c_] := Integrate[dlegT[[b]][s] Exp[I c s], {s, -h, h}];
mTF = Table[Function[{cc, hh}, Evaluate[mT[a, cc] /. h -> hh]], {a, 3}];
dTF = Table[Function[{cc, hh}, Evaluate[dT[b, cc] /. h -> hh]], {b, 3}];
(* same cell, split at t = s: {g-part, g'-part} of int int psi_a(s) K_row1(s - t) {phi_b, phi_b'}(t) *)
selfT[a_, b_] := Module[{below, above},
   below = Integrate[legT[[a]][s] Integrate[Exp[I k (s - t)] {legT[[b]][t], I k dlegT[[b]][t]}, {t, -h, s}], {s, -h, h}];
   above = Integrate[legT[[a]][s] Integrate[Exp[-I k (s - t)] {legT[[b]][t], -I k dlegT[[b]][t]}, {t, s, h}], {s, -h, h}];
   (I/(2 mP k)) (below + above)];
selfTF = Table[Function[{kk, hh}, Evaluate[Simplify[selfT[a, b]] /. {k -> kk, h -> hh}]], {a, 3}, {b, 3}];

(* nb basis functions per cell (2: P1, 3: P2); strain "cell" or "dist"; contrasts passed explicitly *)
solveT27[om_, n_, nb_, strain_, dr_, dm_] := Module[
   {kk = SetPrecision[om/al, 60], d = SetPrecision[dLayer/n, 60], hh, zs, zI, idx, mat, rhs, sol, src, obs, jump},
   (* 60-digit numerics from the start: exact entries that cancel to zero cannot be certified by N *)
   hh = d/2; zs = Table[(j - 1/2) d, {j, n}]; zI = Table[j d, {j, n - 1}];
   idx[i_, a_] := nb (i - 1) + a;
   (* the source of cell j's basis b seen through the kernel toward a point on side sg *)
   src[sg_, b_] := om^2 dr mTF[[b]][-kk sg, hh] + I kk sg dm dTF[[b]][-kk sg, hh];
   mat = Table[0, {n nb}, {n nb}];
   Do[mat[[idx[i, a], idx[i, a]]] += 2 hh/(2 a - 1), {i, n}, {a, nb}];
   Do[If[i == j,
      mat[[idx[i, a], idx[j, b]]] -= Total[{om^2 dr, dm} selfTF[[a, b]][kk, hh]],
      With[{sg = Sign[zs[[i]] - zs[[j]]]},
       mat[[idx[i, a], idx[j, b]]] -= (I/(2 mP kk)) Exp[I kk sg (zs[[i]] - zs[[j]])] mTF[[a]][kk sg, hh] src[sg, b]]],
     {i, n}, {j, n}, {a, nb}, {b, nb}];
   (* interface jumps: [u]_I = u_{I+1}(-h) - u_I(+h), P_b(1) = 1, P_b(-1) = (-1)^b *)
   If[strain == "dist",
    Do[With[{sg = Sign[zs[[i]] - zI[[m]]]},
       Do[mat[[idx[i, a], idx[m + 1, b]]] -= (I/(2 mP kk)) I kk sg dm Exp[I kk sg (zs[[i]] - zI[[m]])] mTF[[a]][kk sg, hh] (-1)^(b - 1);
        mat[[idx[i, a], idx[m, b]]] += (I/(2 mP kk)) I kk sg dm Exp[I kk sg (zs[[i]] - zI[[m]])] mTF[[a]][kk sg, hh],
        {b, nb}]], {i, n}, {a, nb}, {m, n - 1}]];
   rhs = Flatten[Table[gInc[om, zs[[i]]] mTF[[a]][kk, hh], {i, n}, {a, nb}]];
   sol = LinearSolve[N[mat, 40], N[rhs, 40]];
   obs[zo_] := Sum[With[{sg = Sign[zo - zs[[j]]]},
       (I/(2 mP kk)) Exp[I kk sg (zo - zs[[j]])] Sum[src[sg, b] sol[[idx[j, b]]], {b, nb}]], {j, n}] +
     If[strain == "dist", Sum[With[{sg = Sign[zo - zI[[m]]]},
        (I/(2 mP kk)) I kk sg dm Exp[I kk sg (zo - zI[[m]])]
         (Sum[(-1)^(b - 1) sol[[idx[m + 1, b]]], {b, nb}] - Sum[sol[[idx[m, b]]], {b, nb}])], {m, n - 1}], 0];
   {obs[zObsR], obs[zObsT]}];

(* exact layer at a chosen contrast: notebook 7's rtExact reads the globals dM, dRho *)
exactAt[om_, dr_, dm_] := Block[{dRho = dr, dM = dm}, N[exactScat[om], 40]];

om = 300;
(* [1] density only, P1: the u-row decouples from e in notebook 7's G1, so the two schemes coincide *)
Module[{a, b},
  a = solveT27[om, 4, 2, "cell", dRho, 0];
  b = Block[{dM = 0}, solveLayer[om, 4, 2, "G"]];
  Print["  [1] dM = 0: the P1 displacement-only scheme == notebook 7's G1, n = 4: |diff|/|value| = ",
   sci[Max[Abs[(a - b)/b]]], " -> ", chk[Max[Abs[(a - b)/b]] < 10^-25]]];

ladder = {2, 4, 8, 16};
report[label_, nb_, strain_, dr_, dm_] := Module[{ex = exactAt[om, dr, dm], errs, ords},
   errs = Table[Abs[(solveT27[om, n, nb, strain, dr, dm] - ex)/ex], {n, ladder}];
   ords = N[Log[2, errs[[-2]]/errs[[-1]]]];
   Print["      ", StringPadRight[label, 34], "n = 16: R ", sci[errs[[-1, 1]]], ", T ", sci[errs[[-1, 2]]],
    "   order (8 -> 16): R ", ToString[NumberForm[ords[[1]], 3], OutputForm], ", T ",
    ToString[NumberForm[ords[[2]], 3], OutputForm]];
   ords];
Print["  relative error of the scattered u, omega = 300 rad/s, D = 2 m; order from n = 8 -> 16:"];
res = Association[];
Do[Print["    ", ch[[1]], ":"];
  Do[res[{ch[[1]], sc[[1]], sc[[2]]}] = report[
     If[sc[[1]] == 2, "P1", "P2 (the 27-mode content)"] <> ", strain " <> sc[[2]], sc[[1]], sc[[2]], ch[[2]], ch[[3]]],
   {sc, {{2, "cell"}, {3, "cell"}, {2, "dist"}, {3, "dist"}}}],
  {ch, {{"density only", dRho, 0}, {"stiffness only", 0, dM}, {"both (the gate's contrast)", dRho, dM}}}];

Print["  [2] PREDICTION: with the cellwise strain, P2 is second order for a stiffness contrast: ",
  chk[AllTrue[res[{"stiffness only", 3, "cell"}], 1.7 < # < 2.3 &]]];
Print["  [3] density only: P1 and P2 converge at 2p + 2 (4 and 6), with either strain: ",
  chk[AllTrue[Flatten[{res[{"density only", 2, "cell"}] - 4, res[{"density only", 3, "cell"}] - 6}], Abs[#] < 0.3 &]]];
Print["  [4] the 27-mode content at the gate's contrast: cellwise strain second order, distributional third: ",
  chk[AllTrue[res[{"both (the gate's contrast)", 3, "cell"}], 1.7 < # < 2.3 &] &&
    AllTrue[res[{"both (the gate's contrast)", 3, "dist"}], 2.7 < # < 3.3 &]]];
Print["  => neither reaches fourth order; notebook 7's independent-strain voxel does."];

(* ---------------------------------------------------------------------------
   [5]-[6] WHERE THE ERROR LIVES, at Born order, against an independent formula.
   At zeroth order in the contrast the Galerkin solution is the L2 projection
   Pi u0 of the incident field, cell by cell.  Toward the reflection observer the
   source is F[u] = int e^{ikt} (w^2 drho u - i k dM u') dt, so per cell, with
   err = Pi u0 - u0 (exact integration by parts),
     int_j e^{ikt}(w^2 drho err - i k dM err') = (w^2 drho - k^2 dM) int_j e^{ikt} err
                                                 - i k dM [e^{ikt} err]_{cell ends}.
   Summed over cells, the end terms pair up at each interior interface as
   e^{ikz_I} (err_{j}(z_I-) - err_{j+1}(z_I+)) = -e^{ikz_I} [Pi u0]_I   (u0 is continuous),
   and they CANCEL only if the projection is continuous.  The distributional
   strain adds exactly +i k dM e^{ikz_I} [Pi u]_I, leaving the two outer faces.
   [5] the discrete solve at contrast x 1e-8 equals bulk + interfaces + faces;
   [6] the interior-interface sum is what carries the cellwise scheme's
       second-order error, and it is absent from the distributional scheme.
   --------------------------------------------------------------------------- *)
bornParts[om_, n_, nb_, dr_, dm_] := Module[{kk = SetPrecision[om/al, 60], d = SetPrecision[dLayer/n, 60],
    hh, zs, zI, coef, piU, err, bulk, ends, inter, faces, pre},
   hh = d/2; zs = Table[(j - 1/2) d, {j, n}]; zI = Table[j d, {j, n - 1}];
   coef[j_] := Table[(2 a - 1)/(2 hh) gInc[om, zs[[j]]] mTF[[a]][kk, hh], {a, nb}];
   piU[j_, t_] := coef[j] . Table[legT[[a]][t] /. h -> hh, {a, nb}];
   err[j_, t_] := piU[j, t] - gInc[om, zs[[j]]] Exp[I kk t];
   pre[j_] := (I/(2 mP kk)) Exp[I kk (zs[[j]] - zObsR)];   (* observer above: sg = -1 *)
   bulk = Sum[pre[j] (om^2 dr - kk^2 dm) NIntegrate[Exp[I kk t] err[j, t], {t, -hh, hh}, WorkingPrecision -> 30],
     {j, n}];
   inter = Sum[pre[j] (-I kk dm) Exp[I kk hh] err[j, hh], {j, n - 1}] +
     Sum[pre[j] (I kk dm) Exp[-I kk hh] err[j, -hh], {j, 2, n}];
   faces = pre[n] (-I kk dm) Exp[I kk hh] err[n, hh] + pre[1] (I kk dm) Exp[-I kk hh] err[1, -hh];
   {bulk, inter, faces}];
Module[{eps = 10^-8, nb = 3, n = 8, parts, cellErr, distErr, ex, ok5},
  parts = bornParts[om, n, nb, dRho, dM];
  ex = Block[{dRho = eps dRho, dM = eps dM}, N[exactScat[om], 60]][[1]];
  cellErr = (solveT27[om, n, nb, "cell", eps dRho, eps dM][[1]] - ex)/eps;
  distErr = (solveT27[om, n, nb, "dist", eps dRho, eps dM][[1]] - ex)/eps;
  Print["  [5] Born order (contrast x 1e-8), P2, n = ", n, ", reflection: error / contrast scale"];
  Print["      bulk (superconvergent)   ", sci[Abs[parts[[1]]]]];
  Print["      interior interfaces      ", sci[Abs[parts[[2]]]]];
  Print["      outer faces              ", sci[Abs[parts[[3]]]]];
  Print["      cellwise scheme: solve ", sci[Abs[cellErr]], "   bulk + interfaces + faces ", sci[Abs[Total[parts]]],
   "   rel. diff ", sci[Abs[cellErr - Total[parts]]/Abs[cellErr]]];
  Print["      distributional:  solve ", sci[Abs[distErr]], "   bulk + faces              ",
   sci[Abs[parts[[1]] + parts[[3]]]], "   rel. diff ", sci[Abs[distErr - parts[[1]] - parts[[3]]]/Abs[distErr]]];
  ok5 = Abs[cellErr - Total[parts]]/Abs[cellErr] < 10^-4 && Abs[distErr - parts[[1]] - parts[[3]]]/Abs[distErr] < 10^-4;
  Print["      the solves equal the independent decomposition: ", chk[ok5]]];
Module[{pts},
  pts = Table[Abs /@ bornParts[om, n, 3, dRho, dM], {n, {8, 16}}];
  Print["  [6] orders 8 -> 16 of each part (P2): bulk ", ToString[NumberForm[N[Log[2, pts[[1, 1]]/pts[[2, 1]]]], 3], OutputForm],
   ", interfaces ", ToString[NumberForm[N[Log[2, pts[[1, 2]]/pts[[2, 2]]]], 3], OutputForm],
   ", faces ", ToString[NumberForm[N[Log[2, pts[[1, 3]]/pts[[2, 3]]]], 3], OutputForm],
   "  (bulk 2p+2 = 6, interfaces n x h^3 = 2, faces h^3 = 3): ",
   chk[Abs[Log[2, pts[[1, 2]]/pts[[2, 2]]] - 2] < 0.2 && Abs[Log[2, pts[[1, 3]]/pts[[2, 3]]] - 3] < 0.2 &&
     Log[2, pts[[1, 1]]/pts[[2, 1]]] > 5.5]]];
Print["==== ContinuumLimit_T27Order (stage 11): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
