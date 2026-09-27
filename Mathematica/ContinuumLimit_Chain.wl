#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Chain.wl  --  notebook 5 of the continuum-limit study.

   THE QUESTION, now sharp.  A layer of thickness D is filled with n planes of
   cubes (d = D/n); a P plane force at z_s above it; normal incidence.  How does
   the discrete Foldy-Lax field converge to the continuous layer's as n grows,
   and with which single-site T-matrix?

   WHAT NOTEBOOKS 3 AND 4 GIVE.  With the exact cell-averaged kernel and the
   collocation closure T = V Delta (I - S Delta)^{-1}, the self term S cancels
   and the chain IS
       e_i = e0_i + V sum_j K_all(i, j) Delta e_j,
   with K_all the ALL-CELLS kernel: between planes (and to any receiver outside
   a cell) the continuum plane-wave term x sinc(k_W h), within the plane the
   plate term P0.  Both are EXACT cell integrals of the 1-D continuum kernel, so
   the chain is the collocation of the continuum equation with a piecewise-
   constant strain -- S-free, so the package's O((k a)^2) self-term error
   (notebook 4) cannot enter it.
   The package's own T carries far-field form-factor corrections that depart
   from the closure by ~0.25 (k a)^2 (notebook 4); does that help or hurt?

   CHAINS (all against the exact continuous layer, notebook 2c's closed form):
     (A)  Foldy-Lax, exact kernel K_all - S/V, the package T-matrix;
     (A') Foldy-Lax, the same kernel, the closure T built from the package S
          -- must equal (B): the cancellation on the real chain;
     (B)  the S-free collocation.
   Observable: u_z at a fixed depth above the layer, error relative to the
   scattered field.  Coordinates (z, x, y), z down; e^{-i w t}; SI units.
   ============================================================================ *)

ref = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_chain.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
{al, be, rho, dLayer} = Rationalize[{ref["alpha"], ref["beta"], ref["rho"], ref["D"]}, 0];
{dLam, dMu, dRho} = Rationalize[{ref["contrast"]["dlambda"], ref["contrast"]["dmu"], ref["contrast"]["drho"]}, 0];
mu = rho be^2; mP = rho al^2; lam = mP - 2 mu;
zSrc = -12; zObs = -1;
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
eng = {1, 1, 1, 2, 2, 2};

Print["==== ContinuumLimit_Chain :: the discrete layer converging to the continuous one ===="];

(* ---- the 9x9 of a displacement response uu and wavevector kv (notebooks 2a, 3) ---- *)
nine[uu_, kv_] := Module[{rows, cols},
   rows[col_] := Join[col, Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] col[[voigt[[v, 2]]]] +
          I kv[[voigt[[v, 2]]]] col[[voigt[[v, 1]]]]), {v, 6}]];
   cols = Join[Table[uu[[All, j]], {j, 3}],
     Table[(1/2) (I kv[[voigt[[w, 2]]]] uu[[All, voigt[[w, 1]]]] + I kv[[voigt[[w, 1]]]] uu[[All, voigt[[w, 2]]]]), {w, 6}]];
   Transpose[rows /@ cols]];
planeWave[om_, z_, which_] := Module[{s = Sign[z], k = If[which == "P", om/al, om/be], kv, amp, uu},
   kv = {k s, 0, 0};
   amp = I/(2 rho om^2) Exp[I k Abs[z]]/k;
   uu = If[which == "P", amp Outer[Times, kv, kv], amp (k^2 IdentityMatrix[3] - Outer[Times, kv, kv])];
   nine[uu, kv]];
(* the all-cells kernel between a receiver and a cell centred dz away (|dz| >= d/2): exact cell integral *)
kBetween[om_, dz_, d_] := (1/d^2) (planeWave[om, dz, "P"] Sinc[om/al d/2] + planeWave[om, dz, "S"] Sinc[om/be d/2]);
(* the all-cells kernel within the plane: the plate term P0 (notebook 3 [3a]; package plate_average_9x9) *)
kPlate[om_, d_] := Module[{h = d/2, kP = om/al, kS = om/be, avg, out = ConstantArray[0, {9, 9}]},
   avg[m_, k_] := (Exp[I k h] - 1)/(d m k^2);
   out[[1, 1]] = avg[mP, kP]; out[[2, 2]] = out[[3, 3]] = avg[mu, kS];
   out[[4, 4]] = -1/(mP d) - (rho om^2/mP) avg[mP, kP];
   out[[8, 8]] = out[[9, 9]] = -1/(2 mu d) - (rho om^2/(2 mu)) avg[mu, kS];
   out/d^2];
(* contrast operator in the kernel's moment convention (shear 2 dMu, notebook 4) *)
c6 = Table[Which[v <= 3 && w <= 3, dLam + If[v == w, 2 dMu, 0], v == w, 2 dMu, True, 0], {v, 6}, {w, 6}];
delta9[om_] := ArrayFlatten[{{om^2 dRho IdentityMatrix[3], 0}, {0, c6}}];
(* incident: a unit vertical plane force at zSrc, P only: u_z = g, e_zz = dg/dz *)
psiInc[om_, z_] := Module[{k = om/al, g}, g = I/(2 mP k) Exp[I k Abs[z - zSrc]];
   {g, 0, 0, I k Sign[z - zSrc] g, 0, 0, 0, 0, 0}];

(* ---- the exact continuous layer: R referred to z = 0 for e^{i k0 z} incident (notebook 2c) ---- *)
rExact[om_, dl_] := Module[{k0 = om/al, k1, m1 = lam + dLam + 2 (mu + dMu), r1 = rho + dRho, rr, tt, bb, cc},
   k1 = om Sqrt[r1/m1];
   rr /. First@Solve[{1 + rr == bb + cc, mP I k0 (1 - rr) == m1 I k1 (bb - cc),
       bb Exp[I k1 dl] + cc Exp[-I k1 dl] == tt, m1 I k1 (bb Exp[I k1 dl] - cc Exp[-I k1 dl]) == mP I k0 tt}, {rr, tt, bb, cc}]];
uScatExact[om_, dl_] := Module[{k0 = om/al}, rExact[om, dl] psiInc[om, 0][[1]] Exp[-I k0 zObs]];

(* ---- the chains ---- *)
cellOf[om_, n_] := SelectFirst[ref["cells"], #["omega"] == N[om] && #["n"] == n &];
chainB[om_, n_, dl_] := Module[{d = dl/n, v, zs, dd, big, e0, e, obs},
   v = d^3; zs = Table[(j - 1/2) d, {j, n}]; dd = delta9[om];
   big = ArrayFlatten[Table[If[i == j, kPlate[om, d], kBetween[om, zs[[i]] - zs[[j]], d]] . dd, {i, n}, {j, n}]];
   e0 = Flatten[psiInc[om, #] & /@ zs];
   e = LinearSolve[N[IdentityMatrix[9 n] - v big, 30], N[e0, 30]];
   obs = psiInc[om, zObs] + v Sum[kBetween[om, zObs - zs[[j]], d] . dd . e[[9 (j - 1) + 1 ;; 9 j]], {j, n}];
   obs[[1]] - psiInc[om, zObs][[1]]];
(* Foldy-Lax with the exact kernel K_all - S/V and a given 9x9 T *)
chainFL[om_, n_, dl_, tt_, s9_] := Module[{d = dl/n, v, zs, big, e0, psi, obs},
   v = d^3; zs = Table[(j - 1/2) d, {j, n}];
   big = ArrayFlatten[Table[If[i == j, kPlate[om, d] - s9/v, kBetween[om, zs[[i]] - zs[[j]], d]] . tt, {i, n}, {j, n}]];
   e0 = Flatten[psiInc[om, #] & /@ zs];
   psi = LinearSolve[N[IdentityMatrix[9 n] - big, 30], N[e0, 30]];
   obs = psiInc[om, zObs] + Sum[kBetween[om, zObs - zs[[j]], d] . tt . psi[[9 (j - 1) + 1 ;; 9 j]], {j, n}];
   obs[[1]] - psiInc[om, zObs][[1]]];

omegas = Rationalize[Union[#["omega"] & /@ ref["cells"]], 0];
ladder = Union[#["n"] & /@ ref["cells"]];
Print["  layer D = ", dLayer, " m, P plane force at ", zSrc, " m, u_z observed at ", zObs, " m; error / |scattered u_z|"];
results = Table[Module[{ex = N[uScatExact[om, dLayer], 30], rows},
    rows = Table[Module[{c = cellOf[om, n], tp, sp, tCl, eA, eA2, eB},
       tp = mat[c["T"]]; sp = mat[c["self"]];
       tCl = (dLayer/n)^3 delta9[om] . Inverse[IdentityMatrix[9] - sp . delta9[om]];
       eA = Abs[chainFL[om, n, dLayer, tp, sp] - ex]/Abs[ex];
       eA2 = Abs[chainFL[om, n, dLayer, tCl, sp] - chainB[om, n, dLayer]]/Abs[ex];
       eB = Abs[chainB[om, n, dLayer] - ex]/Abs[ex];
       {n, N[om/be dLayer/(2 n)], eA, eB, eA2}], {n, ladder}];
    om -> rows], {om, omegas}];
Do[Print["  omega = ", om, " rad/s:"];
  Print["      n    k_S a      (A) package T     (B) collocation    (A') closure-FL vs (B)"];
  Do[Print["     ", StringPadLeft[ToString[r[[1]]], 2], "   ", sci[r[[2]]], "      ", sci[r[[3]]], "          ", sci[r[[4]]],
     "           ", sci[r[[5]]]], {r, Association[results][om]}],
  {om, omegas}];

orderOf[col_, om_] := Module[{rw = Association[results][om], e},
   e = rw[[All, col]]; N[Log[2, e[[-3]]/e[[-1]]]/2]];   (* over n = 4 -> 16 *)
Print["  observed order (n = 4 -> 16):"];
Do[Print["      omega ", om, ":  (A) ", ToString[NumberForm[orderOf[3, om], 4], OutputForm], "   (B) ", ToString[NumberForm[orderOf[4, om], 4], OutputForm],
   "   error ratio (A)/(B) ", ToString[NumberForm[Mean[Association[results][om][[All, 3]]/Association[results][om][[All, 4]]], 4], OutputForm]], {om, omegas}];
worstA2 = Max[Flatten[Table[Association[results][om][[All, 5]], {om, omegas}]]];
Print["  [1] the self term cancels on the real chain: closure Foldy-Lax = S-free collocation, worst ",
  sci[worstA2], " -> ", chk[worstA2 < 10^-10]];
Print["  [2] the collocation converges at second order: ",
  chk[AllTrue[omegas, 1.8 < orderOf[4, #] < 2.2 &]]];

Print["==== ContinuumLimit_Chain (stage 5): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
