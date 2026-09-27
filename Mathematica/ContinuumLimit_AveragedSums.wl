#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_AveragedSums.wl  --  notebook 3 of the continuum-limit study.

   THE CLAIM.  Cubes TILE the plane: sum_R chi(x - R) = 1 for the cube indicator
   chi.  In Fourier terms, the lateral form factor of the cube,
       F(G) = sinc(G_x h) sinc(G_y h),   h = d/2,
   vanishes at every reciprocal-lattice vector G = 2 pi (p, q)/d other than 0,
   since sinc(p pi) = 0 for integer p != 0.  So when the inter-cell coupling is the
   SOURCE-CELL AVERAGE of the point Green's tensor, every evanescent diffraction
   order of the specular lattice sum is multiplied by zero, and:

     (a) between planes (m != 0) the averaged sum is EXACTLY the continuum's
         plane-wave term times the vertical form factor sinc(k_W h), W = P, S --
         no evanescent local term at all (contrast notebook 2a's point sums);
     (b) within a plane (m = 0) the averaged sum over ALL cells, the cube's own
         included, is the field at the mid-plane of a uniform plate of thickness
         d.  With the self cell excluded (it lives in the T-matrix):
             V S_avg(0) + self = V P0,   P0 = (1/d^2) <g_plate>_{|z'| < h},
         the TILING IDENTITY.  self = int_cube Gamma(-y) dy is the centre-
         collocated cube integral that the cube T-matrix carries.

   (a) is checked against the package's exact-cell-average kernel, built by a
   DIFFERENT route (sinc form factors per mode).  (b) is proved here STATICALLY
   and ANALYTICALLY -- no quadrature anywhere: every cube integral of the Kelvin
   tensor's derivatives is a corner sum of closed-form face antiderivatives --
   and then used as the exact value against which the package's same-plane sum
   (Ewald + a near shell averaged by Gauss quadrature + an O(d^2) Taylor tail)
   is measured.  The dynamic (O((k d)^2)) part of (b) is the next stage.

   Conventions as notebook 2a: (z, x, y); 9 rows (u; e_zz, e_xx, e_yy, 2e_xy,
   2e_zy, 2e_zx); moment columns are +d (receiver derivative), pinned in 2a;
   time e^{-i w t}; SI units.
   ============================================================================ *)

ref = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_averaged.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
{om, al, be, rho} = Rationalize[{ref["omega"], ref["alpha"], ref["beta"], ref["rho"]}, 0];
kP = om/al; kS = om/be; mu = rho be^2; lam = rho al^2 - 2 mu; mP = lam + 2 mu;
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
relMax[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];

voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
eng = {1, 1, 1, 2, 2, 2};
blocks = {{"u<-f", 1 ;; 3, 1 ;; 3}, {"u<-M", 1 ;; 3, 4 ;; 9}, {"e<-f", 4 ;; 9, 1 ;; 3}, {"e<-M", 4 ;; 9, 4 ;; 9}};

(* the 9x9 of a displacement response matrix uu (column j = force j) and a wavevector kv:
   strain rows by i k_a, moment columns +i k (receiver-derivative convention, pinned in 2a) *)
nine[uu_, kv_] := Module[{rows, cols},
   rows[col_] := Join[col, Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] col[[voigt[[v, 2]]]] +
          I kv[[voigt[[v, 2]]]] col[[voigt[[v, 1]]]]), {v, 6}]];
   cols = Join[Table[uu[[All, j]], {j, 3}],
     Table[(1/2) (I kv[[voigt[[w, 2]]]] uu[[All, voigt[[w, 1]]]] + I kv[[voigt[[w, 1]]]] uu[[All, voigt[[w, 2]]]]), {w, 6}]];
   Transpose[rows /@ cols]];

Print["==== ContinuumLimit_AveragedSums :: the source-cell-averaged specular sums ===="];

(* ---------------------------------------------------------------------------
   [1] the form factor on the reciprocal lattice
   --------------------------------------------------------------------------- *)
ffTable = Table[Sinc[p Pi] Sinc[q Pi], {p, -3, 3}, {q, -3, 3}];
Print["  [1] sinc(p pi) sinc(q pi), p, q = -3..3: nonzero only at (0,0): ",
  chk[Total[Abs[Flatten[ffTable]]] == 1 && ffTable[[4, 4]] == 1],
  "   (symbolically, Sin[p Pi] = 0 for integer p: ", Simplify[Sin[p Pi], p \[Element] Integers], ")"];

(* ---------------------------------------------------------------------------
   [2] between planes: S_avg(m) = (1/d^2) sum_W G^_W(0; m d) sinc(k_W h)
   G^_W at k_par = 0 is the plane-wave term of notebook 2a: kv = (k_W sign z, 0, 0).
   --------------------------------------------------------------------------- *)
planeWave[z_, which_] := Module[{s = Sign[z], k = If[which == "P", kP, kS], kv, amp, uu},
   kv = {k s, 0, 0};
   amp = I/(2 rho om^2) Exp[I k Abs[z]]/k;
   uu = If[which == "P", amp Outer[Times, kv, kv], amp (k^2 IdentityMatrix[3] - Outer[Times, kv, kv])];
   nine[uu, kv]];
pred2[d_, m_] := (1/d^2) (planeWave[m d, "P"] Sinc[kP d/2] + planeWave[m d, "S"] Sinc[kS d/2]);
res2 = Table[
   If[e["m"] == 0, Nothing, {e["d"], e["m"], relMax[N[pred2[Rationalize[e["d"], 0], e["m"]]], mat[e["S"]]]}],
   {e, ref["lattice"]}];
worst2 = Max[res2[[All, 3]]];
Print["  [2] m != 0: the package's averaged sum IS the continuum plane-wave term x sinc(k_W h), all 81"];
Print["      entries, m = -4..4 (not 0), d = 0.5, 1, 2: worst ", sci[worst2], " -> ", chk[worst2 < 10^-9]];
Print["      (the point kernel's evanescent local term -- notebook 2a, E(1) d^3 = 11x the continuum term --"];
Print["       is exactly what the cell average removes)"];

(* ---------------------------------------------------------------------------
   [3a] the plate term P0 = (1/d^2) (1/d) int_{-h}^{h} g_plate(-z') dz', from the 1-D spectral kernel.
   The plane Green's tensor at k_par = 0 is FT_kz of [C k k - rho w^2]^{-1}, k = (kz, 0, 0).
   Each entry = c0 + A_P/(M_P k^2 - rho w^2) + A_S/(mu k^2 - rho w^2) (+ odd terms, zero on
   averaging); FT: c0 -> c0 delta(z); 1/(M k^2 - rho w^2) -> i e^{i k0 |z|}/(2 M k0).
   --------------------------------------------------------------------------- *)
Clear[kz];
kv1 = {kz, 0, 0};
uu1 = Inverse[DiagonalMatrix[{mP kz^2, mu kz^2, mu kz^2}] - rho om^2 IdentityMatrix[3]];
g9k = Simplify[nine[uu1, kv1]];
avgG[mm_, k0_, d_] := (Exp[I k0 d/2] - 1)/(d mm k0^2);  (* (1/d) int_{-h}^{h} i e^{i k0|z|}/(2 M k0) dz *)
plate[d_] := (1/d^2) Map[Function[ex,
     Module[{even = Simplify[(ex + (ex /. kz -> -kz))/2], c0, rem, aP, aS},
      c0 = Limit[even, kz -> Infinity];
      rem = Together[even - c0];
      aP = Limit[rem (mP kz^2 - rho om^2), kz -> kP];
      aS = Limit[rem (mu kz^2 - rho om^2), kz -> kS];
      c0/d + aP avgG[mP, kP, d] + aS avgG[mu, kS, d]]], g9k, {2}];
(* sanity: the local (delta) weights are the plate depolarisation of notebook 2b *)
Module[{c0s = Map[Limit[#, kz -> Infinity] &, g9k, {2}]},
  Print["  [3a] the plate kernel's delta weights (x mu): e_zz<-M_zz ", sci[c0s[[4, 4]] mu],
   " (= -mu/(lambda+2mu) = ", sci[-mu/mP], "), 2e_zy<-M_zy ", sci[c0s[[8, 8]] mu], " (= -1/2): ",
   chk[Abs[c0s[[4, 4]] + 1/mP] < 10^-20 && Abs[c0s[[8, 8]] + 1/(2 mu)] < 10^-20]]];

(* ---------------------------------------------------------------------------
   [3b] the self term, ANALYTICALLY.  The static Kelvin tensor is
       G_bc = (cA + cB) d_bc / r - cB d_b d_c r,   cB = (1/mu - 1/M_P)/(8 pi),  cA = 1/(4 pi mu) - cB,
   so every cube integral of d_a d_e G_bc is a cube integral of derivatives of 1/r and of r.  One
   divergence-theorem step moves each onto faces x_p = c_p +- h, and those are NEVER zero (they sit at
   odd multiples of h) -- so the 2-D antiderivatives in the two face coordinates are continuous there
   and the face integral is a corner sum.  The Eshelby delta at r = 0 is carried by the divergence
   theorem itself, not by quadrature.  (A triple antiderivative evaluated at the cube's corners is NOT
   usable: its atan terms jump across the coordinate planes through the origin, and differentiating
   twice there loses exactly the delta.)
   --------------------------------------------------------------------------- *)
Clear[x1, x2, x3];
X3 = {x1, x2, x3}; r3 = Sqrt[X3 . X3];
cB = (1/mu - 1/mP)/(8 Pi); cA = 1/(4 Pi mu) - cB;
faceAnti[g_, p_] := faceAnti[g, p] = Module[{o = Complement[{1, 2, 3}, {p}]},
   Integrate[Integrate[g, X3[[o[[1]]]], Assumptions -> X3[[p]] != 0], X3[[o[[2]]]], Assumptions -> X3[[p]] != 0]];
(* int over cells centred at (c1, c2, c3) (arrays allowed) of D_ds kern, half side hh *)
cubeInt[kern_, ds_List, ctr_, hh_] := Module[{p = First[ds], g, hf, o, f},
   g = D[kern, Sequence @@ (X3[[#]] & /@ Rest[ds])];
   hf = faceAnti[g, p]; o = Complement[{1, 2, 3}, {p}];
   f = Function[{a, b, c}, Evaluate[hf /. Thread[X3 -> {a, b, c}]]];
   Sum[Module[{v = ctr},
      v[[p]] = v[[p]] + sgn hh; v[[o[[1]]]] = v[[o[[1]]]] + su hh; v[[o[[2]]]] = v[[o[[2]]]] + sv hh;
      sgn su sv f @@ v], {sgn, {-1, 1}}, {su, {-1, 1}}, {sv, {-1, 1}}]];
(* I[a, e, b, c](cell) = int_cell d_a d_e G_bc   (static) *)
iCube[{a_, e_, b_, c_}, ctr_, hh_] := (cA + cB) KroneckerDelta[b, c] cubeInt[1/r3, {a, e}, ctr, hh] -
   cB cubeInt[r3, {a, e, b, c}, ctr, hh];
parityOK[idx_] := AllTrue[Table[EvenQ[Count[idx, ax]], {ax, 3}], TrueQ];  (* even in every coordinate *)
combos = Select[Tuples[{1, 2, 3}, 4], parityOK];
(* the e<-M 6x6 from an I tensor: rows (a, b) engineering, columns (c, e) symmetrised *)
eM6[iT_] := Table[Module[{a = voigt[[v, 1]], b = voigt[[v, 2]], c = voigt[[w, 1]], e = voigt[[w, 2]]},
    eng[[v]] (1/4) (iT[{a, e, b, c}] + iT[{a, c, b, e}] + iT[{b, e, a, c}] + iT[{b, c, a, e}])], {v, 6}, {w, 6}];

(* the package's OWN closed form for the static cube (effective_contrasts._static_eshelby_ABC), re-derived
   here from its geometric constants: A = I_1122, B = I_1212, C = I_1111 - I_1122 - 2 I_1212 *)
Module[{a0 = (al^2 + be^2)/(8 Pi rho al^2 be^2), b0 = (al^2 - be^2)/(8 Pi rho al^2 be^2),
   j1 = 2 Pi/3, j2 = -2 (Sqrt[3] - Pi)/9, k1 = 2 (2 Sqrt[3] + Pi)/9, abcPkg, abcMine, sc = {0, 0, 0}},
  abcPkg = {2 (-a0 j1 - 3 b0 j2), 2 b0 (j1 - 3 j2), 6 b0 (3 j2 - k1)};
  abcMine = {iCube[{2, 2, 1, 1}, sc, 1/2], iCube[{1, 2, 1, 2}, sc, 1/2],
    iCube[{1, 1, 1, 1}, sc, 1/2] - iCube[{2, 2, 1, 1}, sc, 1/2] - 2 iCube[{1, 2, 1, 2}, sc, 1/2]};
  Print["  [3b] self cube, static, face antiderivatives vs the package's closed form (A, B, C): ",
   sci[Max[Abs[N[abcMine, 30] - N[abcPkg, 30]]/Abs[N[abcPkg, 30]]]], " -> ",
   chk[Max[Abs[N[abcMine, 30] - N[abcPkg, 30]]/Abs[N[abcPkg, 30]]] < 10^-14]]];

(* ---------------------------------------------------------------------------
   [3c] the STATIC tiling identity: over ALL cells of the plane, self included,
       sum_R int_cell(R) d_a d_e G_bc = int_slab d_a d_e G_bc = the plate's static weight,
   -1/M_P for (1,1,1,1), -1/mu for (1,1,2,2) and (1,1,3,3), zero otherwise.  Static sums are
   scale-free, so d = 1.  Square partial sums |p|, |q| <= N, Richardson in N (degree -3 in 2-D).
   --------------------------------------------------------------------------- *)
nList = {20, 40, 80, 160}; nBig = Last[nList];
pts = Flatten[Table[{p, q}, {p, -nBig, nBig}, {q, -nBig, nBig}], 1];
cheb = Max[Abs[#]] & /@ pts;
selfIdx = Position[pts, {0, 0}][[1, 1]];
ctrs = {ConstantArray[0., Length[pts]], N[pts[[All, 1]]], N[pts[[All, 2]]]};
latt = Association[Table[
    Module[{vals = Re[iCube[cmb, ctrs, 1/2]], partial},
     partial = Table[Total[Pick[vals, Thread[cheb <= n]]], {n, nList}];
     cmb -> {LinearSolve[N[Table[{1, 1/n, 1/n^2, 1/n^3}, {n, nList}]], partial][[1]], vals[[selfIdx]]}],
    {cmb, combos}]];
plateW[{a_, e_, b_, c_}] := If[a == 1 && e == 1, Which[b == 1 && c == 1, -1/mP, b == c, -1/mu, True, 0], 0];
res3c = Table[Abs[latt[cmb][[1]] - N[plateW[cmb]]] mu, {cmb, combos}];
Print["  [3c] static tiling identity, all 21 parity-even components: sum over ALL cells = plate weight;"];
Print["       worst |difference| x mu ", sci[Max[res3c]], " (the weights are O(1) x 1/mu) -> ", chk[Max[res3c] < 10^-7]];

(* ---------------------------------------------------------------------------
   [3d] the package's same-plane averaged sum against the EXACT value, all cells minus self.
   Static (near-static dump, k_S d = 2e-4), e<-M block, three settings of the package's two
   convergence parameters: the near-shell radius r0 (beyond it an O(d^2) Taylor tail) and the
   Gauss order of the near-shell average.
   --------------------------------------------------------------------------- *)
exact6 = eM6[Function[t, If[parityOK[t], latt[t][[1]] - latt[t][[2]], 0]]];
Print["  [3d] exact static V S_avg(0), e<-M diagonal x mu: ", ToString[NumberForm[#, 8], OutputForm] & /@ (Diagonal[exact6] mu)];
(* the static part is the REAL part; the imaginary part is radiation (O(k d) of the static) plus the
   package's round-off as omega -> 0, reported separately *)
res3d = Table[{e["r0_cells"], e["n_gauss"],
    Max[Abs[Flatten[Re[mat[e["S"]][[4 ;; 9, 4 ;; 9]]] - exact6]]]/Max[Abs[Flatten[exact6]]],
    Max[Abs[Flatten[Im[mat[e["S"]][[4 ;; 9, 4 ;; 9]]]]]]/Max[Abs[Flatten[exact6]]]},
   {e, ref["static_s0_d1"]}];
Do[Print["       package, r0_cells ", r[[1]], ", n_gauss ", r[[2]], ":  |Re package - exact| / max|exact| = ", sci[r[[3]]],
   "    (|Im| / max|exact| = ", sci[r[[4]]], ")"], {r, res3d}];
Print["       the package's default carries a STATIC error of ", sci[res3d[[1, 3]]],
  "; it converges onto the exact value as BOTH parameters grow: ",
  chk[res3d[[3, 3]] < res3d[[2, 3]] < res3d[[1, 3]] && res3d[[3, 3]] < 5 10^-5]];
Print["       By [3c], the exact route needs no lattice sum at k_par = 0:  V S_avg(0) = V P0 - self."];
(* the exact static e<-M block of V S_avg(0), x mu, for the package's test of its closed-form kernel *)
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_tiling_exact.json",
  <|"medium" -> <|"alpha" -> N[al], "beta" -> N[be], "rho" -> N[rho]|>,
   "note" -> "exact static V S_avg(0) at k_par = 0, e<-M block (rows/cols e_zz e_xx e_yy 2e_xy 2e_zy 2e_zx), times mu",
   "eM_times_mu" -> N[exact6 mu]|>, "RawJSON"];

Print["==== ContinuumLimit_AveragedSums (stage 3): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
