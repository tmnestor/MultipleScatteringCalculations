#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Oblique.wl  --  notebook 9 of the continuum-limit study.

   THE QUESTION.  At oblique incidence a laterally uniform layer is still a
   chain of planes, but the tiling identity holds only to O(k_par h) and the
   lateral first moments couple through the evanescent lattice orders
   (notebook 8).  Does the Galerkin voxel still converge at second order with
   the mean alone, and at fourth order with the first moments?

   THE SCHEME, spectral.  Voxel basis phi in {1, x/h, y/h, z/h} (Legendre,
   separable), the 9-component state on each; Galerkin (tested with the same
   functions).  The Bloch coupling between planes i, j (lateral wavenumber k_par)
   is, by Poisson summation,
     C_ab(i, j) = (1/d^2) sum_g  L_a(kappa) L_b(-kappa)  Z_ab(kappa; z_i - z_j),
     kappa = k_par + g,  L = int_cell phi^lat e^{i kappa.x},
     Z_ab = int int phi_a^z(z) phi_b^z(z') Ghat(kappa; z - z') dz dz'.
   Ghat(kappa; z) is the inverse transform in k_z of the whole-space tensor
       Ghat3(k) = d_ij/(mu (k_z^2 - gS^2)) - k_i k_j/(rho w^2) [1/(k_z^2 - gS^2) - 1/(k_z^2 - gP^2)],
   gW^2 = kW^2 - kappa^2 (Im gW >= 0), in the 9x9 of notebook 2a (strain rows
   i k_a, moment columns +i k_b).  Ghat3 decays as 1/k_z^2 and the 9x9 adds at
   most two powers of k_z, so each entry is  c0 + sum_W (alpha_W + beta_W k_z)/(k_z^2 - gW^2):
   a delta of weight c0 and, per mode, [i alpha/(2g) + i beta sign(z)/2] e^{i g |z|}.
   Double averaging makes the g-sum converge WITHOUT Ewald.

   REFERENCE: the exact oblique layer (notebook 2c, route 1, validated two ways
   and by energy flux): reflected P and SV for an incident P wave.

   CHECKS: [A] at k_par = 0 the scheme reproduces notebook 7; [B] mean-only
   Galerkin at oblique incidence: order; [C] mean + first moments: order.
   Coordinates (z, x, y), z down; e^{-i w t}; SI units.
   ============================================================================ *)

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);

{al, be, rho} = {5000, 3000, 2500};
mu = rho be^2; mP = rho al^2; lam = mP - 2 mu;
{dLam, dMu, dRho} = {2 10^9, 1 10^9, 100};
dLayer = 2;
voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
eng = {1, 1, 1, 2, 2, 2};
c6 = Table[Which[v <= 3 && w <= 3, dLam + If[v == w, 2 dMu, 0], v == w, 2 dMu, True, 0], {v, 6}, {w, 6}];
delta9[om_] := ArrayFlatten[{{om^2 dRho IdentityMatrix[3], 0}, {0, c6}}];

nine[uu_, kv_] := Module[{rows, cols},
   rows[col_] := Join[col, Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] col[[voigt[[v, 2]]]] +
          I kv[[voigt[[v, 2]]]] col[[voigt[[v, 1]]]]), {v, 6}]];
   cols = Join[Table[uu[[All, j]], {j, 3}],
     Table[(1/2) (I kv[[voigt[[w, 2]]]] uu[[All, voigt[[w, 1]]]] + I kv[[voigt[[w, 1]]]] uu[[All, voigt[[w, 2]]]]), {w, 6}]];
   Transpose[rows /@ cols]];
rowsOf[u_, kv_] := Join[u, Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] u[[voigt[[v, 2]]]] + I kv[[voigt[[v, 2]]]] u[[voigt[[v, 1]]]]), {v, 6}]];

Print["==== ContinuumLimit_Oblique :: the first-moment voxel at oblique incidence ===="];

(* ---- the kernel in k_z, split per mode; symbols gS, gP stand for the vertical wavenumbers ---- *)
Clear[kz, qx, qy, om, gS, gP];
kvec = {kz, qx, qy};
partS = Table[KroneckerDelta[i, j]/mu - kvec[[i]] kvec[[j]]/(rho om^2), {i, 3}, {j, 3}];  (* over kz^2 - gS^2 *)
partP = Table[kvec[[i]] kvec[[j]]/(rho om^2), {i, 3}, {j, 3}];                         (* over kz^2 - gP^2 *)
numS = nine[partS, kvec]; numP = nine[partP, kvec];
split[num_, g_] := Module[{qr = PolynomialQuotientRemainder[Expand[num], kz^2 - g^2, kz]},
   {qr[[1]], Coefficient[qr[[2]], kz, 0], Coefficient[qr[[2]], kz, 1]}];
sS = Map[split[#, gS] &, numS, {2}]; sP = Map[split[#, gP] &, numP, {2}];
(* the S and P quotients each carry k_z^2 (from the k k term); they cancel in the SUM, which must be
   k_z-free: a pure delta *)
c0Sym = Expand[sS[[All, All, 1]] + sP[[All, All, 1]]];
Print["  kernel split: the summed quotient is free of k_z (a delta only): ", chk[FreeQ[c0Sym, kz]]];
aPlus[s_, g_] := I s[[All, All, 2]]/(2 g) + I s[[All, All, 3]]/2;
aMinus[s_, g_] := I s[[All, All, 2]]/(2 g) - I s[[All, All, 3]]/2;
kernFun = Function[{omv, qxv, qyv},
   Module[{kS = omv/be, kP = omv/al, gSv, gPv, rules},
    gSv = Sqrt[kS^2 - qxv^2 - qyv^2]; gPv = Sqrt[kP^2 - qxv^2 - qyv^2];   (* principal root: Im >= 0 *)
    rules = {om -> omv, qx -> qxv, qy -> qyv, gS -> gSv, gP -> gPv};
    {c0Sym /. rules, {aPlus[sS, gS] /. rules, aMinus[sS, gS] /. rules, gSv},
     {aPlus[sP, gP] /. rules, aMinus[sP, gP] /. rules, gPv}}]];

(* ---- Legendre moments: lateral factor, z moments, same-cell z double integrals ---- *)
Clear[s, t, k, h, g];
latF[0, kk_, hh_] := 2 hh Sinc[kk hh];
latF[1, kk_, hh_] := If[Abs[kk hh] < 10^-6, 2 I kk hh^2/3, 2 I (Sin[kk hh] - kk hh Cos[kk hh])/(kk^2 hh)];
zMom[0, gg_, hh_] := latF[0, gg, hh]; zMom[1, gg_, hh_] := latF[1, gg, hh];   (* int phi(s) e^{i g s} ds *)
legZ = {1 &, (#/h) &};
jPlusSym = Table[Integrate[legZ[[a]][s] legZ[[b]][t] Exp[I g (s - t)], {s, -h, h}, {t, -h, s}], {a, 2}, {b, 2}];
jMinusSym = Table[Integrate[legZ[[a]][s] legZ[[b]][t] Exp[-I g (s - t)], {s, -h, h}, {t, s, h}], {a, 2}, {b, 2}];
jPlus = Function[{gg, hh}, Evaluate[jPlusSym /. {g -> gg, h -> hh}]];
jMinus = Function[{gg, hh}, Evaluate[jMinusSym /. {g -> gg, h -> hh}]];
gramZ = {{2, 0}, {0, 2/3}};   (* int phi_a phi_b ds / h *)

(* the voxel basis: Legendre degrees in (z, x, y) -- the mean, then the z-, x- and y-moments *)
basis = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
nb = Length[basis];

(* ---- the Galerkin chain at Bloch wavenumber (kx, 0), n planes, g-sum |p|, |q| <= pMax ---- *)
solveOblique[omv_, kxv_, n_, useB_, pMax_] := Module[
  {d = dLayer/n, hh, zs, gl, ker, cpl, dd = N[delta9[omv]], nU, big, rhs, sol, kinc, uinc, psiInc, refl, bas},
  hh = d/2; zs = Table[(j - 1/2) d, {j, n}];
  bas = basis[[useB]]; nU = 9 Length[bas];
  gl = Flatten[Table[{kxv + 2 Pi p/d, 2 Pi q/d}, {p, -pMax, pMax}, {q, -pMax, pMax}], 1];
  ker = kernFun[omv, #[[1]], #[[2]]] & /@ N[gl, 20];
  (* coupling block (plane separation m, basis a <- b), summed over g *)
  cpl[m_, a_, b_] := cpl[m, a, b] = (1/d^2) Sum[Module[{kap = gl[[i]], kr = ker[[i]], lat, zz},
       lat = latF[a[[2]], kap[[1]], hh] latF[a[[3]], kap[[2]], hh] latF[b[[2]], -kap[[1]], hh] latF[b[[3]], -kap[[2]], hh];
       zz = Which[
         m == 0, kr[[1]] hh gramZ[[a[[1]] + 1, b[[1]] + 1]] +
          Sum[w[[1]] jPlus[w[[3]], hh][[a[[1]] + 1, b[[1]] + 1]] + w[[2]] jMinus[w[[3]], hh][[a[[1]] + 1, b[[1]] + 1]], {w, kr[[2 ;; 3]]}],
         m > 0, Sum[w[[1]] Exp[I w[[3]] m d] zMom[a[[1]], w[[3]], hh] zMom[b[[1]], -w[[3]], hh], {w, kr[[2 ;; 3]]}],
         True, Sum[w[[2]] Exp[-I w[[3]] m d] zMom[a[[1]], -w[[3]], hh] zMom[b[[1]], w[[3]], hh], {w, kr[[2 ;; 3]]}]];
       lat zz], {i, Length[gl]}];
  big = ArrayFlatten[Table[
     ArrayFlatten[Table[
       If[i == j && ai == bi, d^3 Times @@ ({1, 1/3}[[# + 1]] & /@ bas[[ai]]) IdentityMatrix[9], 0] -
        cpl[i - j, bas[[ai]], bas[[bi]]] . dd, {ai, Length[bas]}, {bi, Length[bas]}]], {i, n}, {j, n}]];
  (* incident P plane wave, displacement (eta, p, 0) in (z, x, y) with k = w (eta, p, 0) *)
  kinc = {Sqrt[(omv/al)^2 - kxv^2], kxv, 0};
  uinc = kinc/(omv/al);
  psiInc = rowsOf[uinc, kinc];
  rhs = Flatten[Table[psiInc Exp[I kinc[[1]] zs[[i]]] latF[bas[[ai, 2]], kxv, hh] latF[bas[[ai, 3]], 0, hh] zMom[bas[[ai, 1]], kinc[[1]], hh],
     {i, n}, {ai, Length[bas]}]];
  sol = LinearSolve[N[big, 20], N[rhs, 20]];
  (* the specular reflected plane waves, P and S, displacement at z = 0 *)
  refl = Table[Module[{kr = kernFun[omv, kxv, 0], w},
     w = kr[[1 + W]];
     Sum[(1/d^2) latF[bas[[bi, 2]], -kxv, hh] latF[bas[[bi, 3]], 0, hh] Exp[I w[[3]] zs[[j]]] zMom[bas[[bi, 1]], w[[3]], hh] *
       (w[[2]] . dd . sol[[nU (j - 1) + 9 (bi - 1) + 1 ;; nU (j - 1) + 9 bi]])[[1 ;; 3]], {j, n}, {bi, Length[bas]}]], {W, 2}];
  refl];

(* ---- the exact layer (notebook 2c, route 1): R_PP, R_PS for displacement-normalised P ---- *)
aPSV[lm_, m_, r_, p_, w_] := {{0, -I w p, 1/m, 0}, {-I w p lm/(lm + 2 m), 0, 0, 1/(lm + 2 m)},
   {-r w^2 + w^2 p^2 4 m (lm + m)/(lm + 2 m), 0, 0, -I w p lm/(lm + 2 m)}, {0, -r w^2, -I w p, 0}};
eta[c_, p_] := Sqrt[1/c^2 - p^2];
stateOf[lm_, m_, u_, p_, q_, w_] := {u[[1]], u[[2]], m I w (q u[[1]] + p u[[2]]), lm I w p u[[1]] + (lm + 2 m) I w q u[[2]]};
exactRT[omv_, kxv_] := Module[{p = kxv/omv, ea, eb, pd, pu, su, sd, prop, rpp, rps, tpp, tps, sol, lam1 = lam + dLam, mu1 = mu + dMu, rh1 = rho + dRho},
   ea = eta[al, p]; eb = eta[be, p];
   pd = stateOf[lam, mu, {p, ea}, p, ea, omv]; pu = stateOf[lam, mu, {p, -ea}, p, -ea, omv];
   sd = stateOf[lam, mu, {eb, -p}, p, eb, omv]; su = stateOf[lam, mu, {-eb, -p}, p, -eb, omv];
   prop = MatrixExp[N[aPSV[lam1, mu1, rh1, p, omv] dLayer, 30]];
   (* the state mixes displacements (~1) with tractions (~mu w |u|): scale the traction rows before solving *)
   With[{sc = DiagonalMatrix[{1, 1, 1/(mu omv), 1/(mu omv)}]},
    sol = LinearSolve[sc . Transpose[{prop . pu, prop . su, -pd, -sd}], -sc . prop . pd]];
   sol[[1 ;; 2]]];
(* reflected displacement at z = 0 from exact R: P up (p, -eta_a) and S up (-eta_b, -p) in (x, z) *)
exactRefl[omv_, kxv_] := Module[{p = kxv/omv, rr = exactRT[omv, kxv]},
   (* notebook 2c's polarisations are SLOWNESS vectors, |(p, eta_a)| = 1/alpha; the incident here has unit
      displacement, so the exact reflected displacements carry a factor alpha *)
   al {rr[[1]] {-eta[al, p], p, 0}, rr[[2]] {-p, -eta[be, p], 0}}];   (* (z, x, y) *)

relErr[a_, b_] := Norm[a - b]/Norm[b];

pMax = 8;   (* the lattice sum is converged to 3 digits here: 4.47e-9 / 4.46e-9 / 4.46e-9 at 4, 8, 32 *)
ladder = {1, 2, 4, 8};
(* ---------------------------------------------------------------------------
   [A] normal incidence: notebook 7's numbers (omega = 300: first moment 2.76e-7, 1.72e-8, 1.07e-9)
   --------------------------------------------------------------------------- *)
Module[{om0 = 300, ex, e1},
  ex = exactRefl[om0, 0];
  e1 = Table[relErr[solveOblique[om0, 0, n, {1, 2}, 0][[2]], ex[[1]]], {n, {1, 2, 4}}];
  Print["  [A] normal incidence, z-moment voxel, n = 1, 2, 4: ", sci /@ e1, "  (notebook 7: 2.76e-7, 1.72e-8, 1.07e-9) -> ",
   chk[Max[Abs[e1/{2.76*^-7, 1.72*^-8, 1.07*^-9} - 1]] < 0.01]]];

(* ---------------------------------------------------------------------------
   [B], [C] oblique incidence: orders of the mean-only and the first-moment voxel
   --------------------------------------------------------------------------- *)
okB = {}; okC = {};
Do[Module[{om0 = 300, kx0, ex, e0, e1, ord},
   kx0 = N[om0/al Sin[th Degree], 20]; ex = exactRefl[om0, kx0];
   e0 = Table[With[{r = solveOblique[om0, kx0, n, {1}, pMax]}, {relErr[r[[2]], ex[[1]]], relErr[r[[1]], ex[[2]]]}], {n, ladder}];
   e1 = Table[With[{r = solveOblique[om0, kx0, n, {1, 2, 3}, pMax]}, {relErr[r[[2]], ex[[1]]], relErr[r[[1]], ex[[2]]]}], {n, ladder}];
   ord[e_] := N[Log[2, e[[2]]/e[[4]]]/2];   (* n = 2 -> 8 *)
   Print["  P incident at ", th, " deg, omega = ", om0, "; |error| / |R|, {R_PP, R_PS}:"];
   Do[Print["      n = ", ladder[[i]], ":  mean only ", sci /@ e0[[i]], "    mean + first moments ", sci /@ e1[[i]]], {i, 4}];
   Print["      orders (n = 2 -> 8): mean only ", ToString[NumberForm[ord[e0[[All, 1]]], 3], OutputForm], " / ",
    ToString[NumberForm[ord[e0[[All, 2]]], 3], OutputForm], "    first moments ",
    ToString[NumberForm[ord[e1[[All, 1]]], 3], OutputForm], " / ", ToString[NumberForm[ord[e1[[All, 2]]], 3], OutputForm]];
   AppendTo[okB, AllTrue[{ord[e0[[All, 1]]], ord[e0[[All, 2]]]}, 1.8 < # < 2.2 &]];
   AppendTo[okC, AllTrue[{ord[e1[[All, 1]]], ord[e1[[All, 2]]]}, 3.7 < # < 4.3 &]]],
  {th, {20, 40}}];
Print["  [B] mean-only voxel: second order at oblique incidence: ", chk[And @@ okB]];
Print["  [C] first-moment voxel: FOURTH order at oblique incidence, P and S: ", chk[And @@ okC]];

(* ---------------------------------------------------------------------------
   [D] the y-moment decouples at k_y = 0 (mirror symmetry y -> -y)
   --------------------------------------------------------------------------- *)
Module[{om0 = 300, kx0, a, b},
  kx0 = N[om0/al Sin[20 Degree], 20];
  a = solveOblique[om0, kx0, 2, {1, 2, 3}, pMax]; b = solveOblique[om0, kx0, 2, {1, 2, 3, 4}, pMax];
  Print["  [D] adding the y-moment changes the reflection by ", sci[Norm[Flatten[a - b]]/Norm[Flatten[a]]], " -> ",
   chk[Norm[Flatten[a - b]]/Norm[Flatten[a]] < 10^-12]]];

Print["==== ContinuumLimit_Oblique (stage 9): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
