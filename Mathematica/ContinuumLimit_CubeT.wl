#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_CubeT.wl  --  notebook 4 of the continuum-limit study.

   WHY.  Notebook 3 proved the tiling identity: over all cells of a plane, the
   cube's own included, the source-cell-averaged Green's tensor sums to the
   plate's field, so the exact Foldy-Lax kernel at normal incidence is
   (plate term) - self/V.  That is exact only if the T-matrix carries the SAME
   self term.  For the collocation closure (uniform internal field, derived from
   first principles in CubeT9FromFirstPrinciples.wl) the cube's own field enters
   as
       e = e_exc + S Delta e      =>      T = V Delta (I - S Delta)^{-1},
   S = int_cube Gamma(-y) dy (9x9), Delta = blockdiag(w^2 drho I3, the Voigt
   contrast stiffness).  Then the self cell CANCELS algebraically:
       psi = psi0 + K T psi,  e = (I - S Delta)^{-1} psi   =>   e = psi0 + V (K + S/V) Delta e,
   and K + S/V is the ALL-CELLS kernel, which the tiling identity makes the
   plate's.  So the whole discrete scheme is a collocation of the continuum
   equation -- if, and only if, T is this closure.

   THIS NOTEBOOK:
     [1] S, dynamic, analytically -- the static Kelvin part and the series of the
         radiating remainder, every cube integral by one divergence-theorem step
         and continuous face antiderivatives (no quadrature) -- against the
         package's cube_self_9x9 (Gamma0, A^c B^c C^c by Taylor moments: an
         independent route), at k_S a = 0.01 .. 0.3;
     [2] the package's 9x9 T-matrix against the closure V Delta (I - S Delta)^{-1}:
         where they differ, at what order in k a (the package T also carries
         far-field form-factor corrections);
     [3] the cancellation itself, on the matrices.
   FINDINGS (2026-09-28):
     * STATICALLY the package T IS the collocation closure with the very S the
       exact kernel removes (both blocks to 2e-5 at k_S a = 0.01, the difference
       falling as (k a)^2), so the self cell cancels and the discrete scheme is an
       exact collocation of the continuum at static order.
     * DYNAMICALLY the package T departs from the closure by ~0.25 (k a)^2 (2.3%
       at k_S a = 0.3), nearly independent of which S is used: that is its
       far-field form-factor corrections (c2, c4), which break the exact
       self-cancellation at O((k a)^2).
     * The package's S (Gamma0; A^c, B^c, C^c) is off the exact S in its REAL part
       by ~0.36 (k a)^2 (u<-f) and ~0.6 (k a)^2 (e<-M): its Taylor-moment route
       keeps only the odd-power (polynomial) terms of the Green's tensor, on the
       stated premise that only those are integrable over the cube.  The dropped
       even-power terms (k^2 r, r^3, x x / r, ...) are integrable -- just not
       polynomial -- and the face calculus here integrates them exactly.  The
       imaginary parts agree to leading order.  Not changed in the package: the
       T-matrix's c2 correction was validated against Mie with these integrals,
       so the choice is left to the chain notebook's measured order.
   Green's tensor (Kupradze), e^{-i w t}:
       G_ij = (1/(4 pi rho w^2)) [ kS^2 d_ij phiS + d_i d_j (phiS - phiP) ],  phi = e^{i k r}/r.
   Its static part is (cA + cB) d_ij/r - cB d_i d_j r,  cB = (1/mu - 1/M_P)/(8 pi),
   cA = 1/(4 pi mu) - cB (notebook 3, [3b]).  Coordinates (z, x, y); SI units.
   ============================================================================ *)

ref = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_cubet.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
{al, be, rho, d} = Rationalize[{ref["alpha"], ref["beta"], ref["rho"], ref["d"]}, 0];
mu = rho be^2; mP = rho al^2; lam = mP - 2 mu;
{dLam, dMu, dRho} = Rationalize[{ref["contrast"]["dlambda"], ref["contrast"]["dmu"], ref["contrast"]["drho"]}, 0];
h = d/2; vol = d^3;
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
relMax[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];
voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
eng = {1, 1, 1, 2, 2, 2};

Print["==== ContinuumLimit_CubeT :: the single site the tiling identity needs ===="];

(* ---------------------------------------------------------------------------
   Face calculus (notebook 3): int_cube d_p g = oint g n_p, faces x_p = +-h never through the origin,
   so a 2-D antiderivative in the other two coordinates is continuous there; corner sums.
   --------------------------------------------------------------------------- *)
Clear[x1, x2, x3];
X = {x1, x2, x3}; r = Sqrt[X . X];
faceAnti[g_, p_] := faceAnti[g, p] = Module[{o = Complement[{1, 2, 3}, {p}]},
   Integrate[Integrate[g, X[[o[[1]]]], Assumptions -> X[[p]] != 0], X[[o[[2]]]], Assumptions -> X[[p]] != 0]];
fluxP[g_, p_] := Module[{hf = faceAnti[g, p], o = Complement[{1, 2, 3}, {p}]},   (* oint g n_p *)
   Sum[sgn su sv (hf /. Thread[X[[{p, o[[1]], o[[2]]}]] -> {sgn h, su h, sv h}]), {sgn, {-1, 1}}, {su, {-1, 1}}, {sv, {-1, 1}}]];
fluxXN[g_] := h Sum[Module[{hf = faceAnti[g, p], o = Complement[{1, 2, 3}, {p}]},  (* oint g (x.n) *)
     Sum[su sv (hf /. Thread[X[[{p, o[[1]], o[[2]]}]] -> {sgn h, su h, sv h}]), {sgn, {-1, 1}}, {su, {-1, 1}}, {sv, {-1, 1}}]],
    {p, 3}];
(* cube integrals of the basis, all omega-independent:  int d_a d_b f  and  int f  for f = 1/r, r^m *)
intDD[f_, a_, b_] := intDD[f, a, b] = fluxP[D[f, X[[b]]], a];
intR[m_] := intR[m] = If[m == -1, fluxXN[1/r]/2, fluxXN[r^m]/(3 + m)];      (* int r^m = oint r^m (x.n)/(3+m) *)
intDDDD[f_, a_, e_, b_, c_] := intDDDD[f, a, e, b, c] = fluxP[D[f, X[[e]], X[[b]], X[[c]]], a];

nMax = 14;
selfAnalytic[om_] := Module[{kP = om/al, kS = om/be, pre = 1/(4 Pi rho om^2), cn, sn, gInt, iTen, s9},
   cn[n_] := I^n (kS^n - kP^n)/n!;   (* phiS - phiP = sum_n cn r^(n-1); n = 2 is the static -cB d d r *)
   sn[n_] := (I kS)^n/n!;           (* phiS = 1/r + sum_{n >= 1} sn r^(n-1) *)
   (* u <- f:  int G_ij *)
   gInt = Table[If[i == j || False, 0, 0] +
      KroneckerDelta[i, j] ((cA + cB) intR[-1] + pre kS^2 Sum[sn[n] intR[n - 1], {n, 1, nMax}]) -
      cB intDD[r, i, j] + pre Sum[cn[n] intDD[r^(n - 1), i, j], {n, 3, nMax}], {i, 3}, {j, 3}];
   (* e <- M:  I[a, e, b, c] = int d_a d_e G_bc *)
   iTen[{a_, e_, b_, c_}] := KroneckerDelta[b, c] ((cA + cB) intDD[1/r, a, e] +
        pre kS^2 Sum[sn[n] intDD[r^(n - 1), a, e], {n, 2, nMax}]) -
      cB intDDDD[r, a, e, b, c] + pre Sum[cn[n] intDDDD[r^(n - 1), a, e, b, c], {n, 3, nMax}];
   s9 = ConstantArray[0, {9, 9}];
   s9[[1 ;; 3, 1 ;; 3]] = gInt;
   Do[Module[{a = voigt[[v, 1]], b = voigt[[v, 2]], c = voigt[[w, 1]], e = voigt[[w, 2]]},
     s9[[3 + v, 3 + w]] = eng[[v]] (1/4) (iTen[{a, e, b, c}] + iTen[{a, c, b, e}] + iTen[{b, e, a, c}] + iTen[{b, c, a, e}])],
    {v, 6}, {w, 6}];
   N[s9, 20]];
cB = (1/mu - 1/mP)/(8 Pi); cA = 1/(4 Pi mu) - cB;

(* ---------------------------------------------------------------------------
   [1a] the series is the exact Green's tensor: Kelvin + remainder vs the closed-form Kupradze tensor,
   40 digits, at the highest frequency (k_S a = 0.3), at points out to the cube's corner
   --------------------------------------------------------------------------- *)
Module[{om = 1800, kP, kS, pre, gSer, gEx, errs},
  kP = om/al; kS = om/be; pre = 1/(4 Pi rho om^2);
  gSer = Table[KroneckerDelta[i, j] ((cA + cB)/r + pre kS^2 Sum[(I kS)^n/n! r^(n - 1), {n, 1, nMax}]) - cB D[r, X[[i]], X[[j]]] +
      pre Sum[I^n (kS^n - kP^n)/n! D[r^(n - 1), X[[i]], X[[j]]], {n, 3, nMax}], {i, 3}, {j, 3}];
  gEx = Table[pre (kS^2 KroneckerDelta[i, j] Exp[I kS r]/r + D[(Exp[I kS r] - Exp[I kP r])/r, X[[i]], X[[j]]]), {i, 3}, {j, 3}];
  errs = Table[Max[Abs[N[(gSer - gEx) /. Thread[X -> pt], 40]]]/Max[Abs[N[gEx /. Thread[X -> pt], 40]]],
    {pt, {{3/10, 1/5, 1/10}, {1/20, 0, 1/40}, {1/2, 1/2, 1/2}}}];
  Print["  [1a] series Green's tensor vs the exact Kupradze tensor, k_S a = 0.3, three points to the corner: worst ",
   sci[Max[errs]], " -> ", chk[Max[errs] < 10^-12]]];

(* ---------------------------------------------------------------------------
   [1b] S (exact: [1a] + the exact face calculus of notebook 3) against the package's cube_self_9x9
   (Gamma0 and A^c, B^c, C^c from Taylor moments -- the integrals the cube T-matrix itself uses)
   --------------------------------------------------------------------------- *)
t0 = AbsoluteTime[];
selves = Association[Table[c["omega"] -> selfAnalytic[Rationalize[c["omega"], 0]], {c, ref["cases"]}]];
Print["  [1b] self term S = int_cube Gamma, exact vs the package (", Round[AbsoluteTime[] - t0], " s); |diff| / max|S| per block, Re and Im:"];
res1 = Table[Module[{om = c["omega"], mine = selves[c["omega"]], pk = mat[c["self"]], blk},
    blk[sl_, f_] := Max[Abs[Flatten[f[mine[[sl, sl]] - pk[[sl, sl]]]]]]/Max[Abs[Flatten[mine[[sl, sl]]]]];
    {N[om/be h], blk[1 ;; 3, Re], blk[1 ;; 3, Im], blk[4 ;; 9, Re], blk[4 ;; 9, Im]}], {c, ref["cases"]}];
Do[Print["      k_S a = ", sci[r1[[1]]], ":  u<-f Re ", sci[r1[[2]]], " Im ", sci[r1[[3]]], "    e<-M Re ", sci[r1[[4]]], " Im ", sci[r1[[5]]]],
  {r1, res1}];
slope[col_] := (Log[res1[[-1, col]]] - Log[res1[[2, col]]])/(Log[res1[[-1, 1]]] - Log[res1[[2, 1]]]);
Print["      growth with k_S a (0.05 -> 0.3): u<-f Re ", NumberForm[slope[2], 3], ", e<-M Re ", NumberForm[slope[4], 3],
  ";  static agreement (k_S a = 0.01, all parts < 2e-4): ", chk[Max[res1[[1, 2 ;;]]] < 2 10^-4]];

(* ---------------------------------------------------------------------------
   [2] the package T against the collocation closure T_col = V Delta (I - S Delta)^{-1}
   --------------------------------------------------------------------------- *)
(* the contrast operator IN THE KERNEL'S MOMENT CONVENTION: its shear moment columns carry the symmetrised
   1/2 (notebook 2a), so the shear diagonal is 2 dMu, not dMu -- measured: the package T's shear entry is
   2 dMu* exactly *)
c6 = Table[Which[v <= 3 && w <= 3, dLam + If[v == w, 2 dMu, 0], v == w, 2 dMu, True, 0], {v, 6}, {w, 6}];
delta9[om_] := ArrayFlatten[{{om^2 dRho IdentityMatrix[3], 0}, {0, c6}}];
tCol[om_, s9_] := vol delta9[om] . Inverse[IdentityMatrix[9] - s9 . delta9[om]];
Print["  [2] the package's 9x9 T against the collocation closure V Delta (I - S Delta)^{-1}, with the package's S and with the exact S:"];
res2 = Table[Module[{om = Rationalize[c["omega"], 0], tp = mat[c["T"]], tcPkg, tcEx, blk},
    tcPkg = N[tCol[om, mat[c["self"]]], 20]; tcEx = N[tCol[om, selves[c["omega"]]], 20];
    blk[a_, b_, sl_] := relMax[a[[sl, sl]], b[[sl, sl]]];
    {N[om/be h], blk[tp, tcPkg, 1 ;; 3], blk[tp, tcPkg, 4 ;; 9], blk[tp, tcEx, 1 ;; 3], blk[tp, tcEx, 4 ;; 9]}], {c, ref["cases"]}];
Do[Print["      k_S a = ", sci[r2[[1]]], ":  vs closure(package S): density ", sci[r2[[2]]], " stiffness ", sci[r2[[3]]],
   "    vs closure(exact S): density ", sci[r2[[4]]], " stiffness ", sci[r2[[5]]]], {r2, res2}];
Print["      static limit (k_S a = 0.01): the package T IS the collocation closure: ",
  chk[Max[res2[[1, 2 ;;]]] < 2 10^-4]];

(* ---------------------------------------------------------------------------
   [3] the cancellation, on the matrices: for any kernel K, with T = V Delta (I - S Delta)^{-1},
       (I - K T)(V... ) -- psi = psi0 + K T psi  and  e = psi0 + (V K + S) Delta e  with psi = (I - S Delta) e.
   Checked on a random 2-plane kernel (the identity is algebraic: (I - S Delta) - V K Delta = I - (V K + S) Delta).
   --------------------------------------------------------------------------- *)
Module[{om = 600, s9, dl, tt, kk, psi0, psi, e, big1, big2, sBig, dBig, tBig},
  SeedRandom[7];
  s9 = selves[600.]; dl = N[delta9[om]]; tt = N[tCol[om, s9]];
  kk = RandomComplex[{-1 - I, 1 + I}, {18, 18}] 10^-12;   (* any inter-cell kernel, self excluded *)
  psi0 = RandomComplex[{-1 - I, 1 + I}, 18];
  tBig = ArrayFlatten[{{tt, 0}, {0, tt}}]; sBig = ArrayFlatten[{{s9, 0}, {0, s9}}]; dBig = ArrayFlatten[{{dl, 0}, {0, dl}}];
  psi = LinearSolve[IdentityMatrix[18] - kk . tBig, psi0];
  e = LinearSolve[IdentityMatrix[18] - (N[vol] kk + sBig) . dBig, psi0];
  Print["  [3] Foldy-Lax (K excludes self, T the closure) vs collocation (all-cells kernel VK + S, bare Delta):"];
  Print["      |psi - (I - S Delta) e| / |psi| = ", sci[Max[Abs[psi - (IdentityMatrix[18] - sBig . dBig) . e]]/Max[Abs[psi]]],
   " -> ", chk[Max[Abs[psi - (IdentityMatrix[18] - sBig . dBig) . e]]/Max[Abs[psi]] < 10^-10]]];

Print["==== ContinuumLimit_CubeT (stage 4): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
