#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_ThinLayer.wl  --  notebook 2c of the continuum-limit study.

   WHY.  A layer of cubes must, in the limit, reproduce a continuous layer.  The
   thinnest case -- ONE plane of cubes -- stands for a thin homogeneous layer, and
   what it can and cannot represent is decided order by order in k D.  This
   notebook builds that TARGET: the exact response of a homogeneous layer of
   thickness D embedded in a homogeneous space, its Taylor expansion in k D and in
   the contrast, and its thin-layer asymptotics (the equivalent interface).

   CONVENTIONS.  z down; time e^{-i w t}; lateral dependence e^{i w p x}, p the
   horizontal slowness.  P-SV state b = (u_x, u_z, t_xz, t_zz), t = traction on a
   horizontal plane; SH state (u_y, t_yz).  First-order system d b/dz = A b, from
       t_xz = mu (d_z u_x + d_x u_z),  t_zz = lambda d_x u_x + (lambda + 2 mu) d_z u_z,
       -rho w^2 u_i = d_j sigma_ij.
   Plane waves: P  u ~ (p, +-eta_a),  S  u ~ (+-eta_b, -p),  eta_c = sqrt(1/c^2 - p^2),
   + for downgoing (e^{+i w eta z}).  SI units.

   CHECKS:
     [1] normal incidence, closed form, against the package's layered field
         (notebook 1's reference JSON);
     [2] Taylor series in k0 D, coefficients exact in the contrast;
     [3] oblique P-SV and SH by TWO routes (the layer's matrix exponential; explicit
         plane waves inside the layer), energy-flux conservation, and p -> 0 = [1];
     [4] the series in D at oblique incidence: the O(D) term is the thin-layer jump
         (A1 - A0) D; truncation order verified against the exact result;
     [5] asymptotics: a soft thin layer, mu1 = D / eta at fixed compliance eta,
         tends to the linear-slip interface as D -> 0.
   ============================================================================ *)

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);  (* a gated check *)

(* background and the gate's contrast (SI) *)
{al0, be0, rh0} = {5000, 3000, 2500};
{dLam, dMu, dRho} = {2 10^9, 1 10^9, 100};
om = 60;
lam0 = rh0 (al0^2 - 2 be0^2); mu0 = rh0 be0^2;
lam1 = lam0 + dLam; mu1 = mu0 + dMu; rh1 = rh0 + dRho;
al1 = Sqrt[(lam1 + 2 mu1)/rh1]; be1 = Sqrt[mu1/rh1];

Print["==== ContinuumLimit_ThinLayer :: a thin homogeneous layer in a homogeneous space ===="];

(* ---------------------------------------------------------------------------
   [1] normal incidence: a 1-D problem with modulus M and density rho per wave type.
   Incident e^{i k0 z} from above on [0, D]; R referred to z = 0, T to z = D.
   --------------------------------------------------------------------------- *)
Clear[dd, m0s, m1s, r0s, r1s, w];
k[m_, r_] := w Sqrt[r/m]; zz[m_, r_] := Sqrt[r m];
rt1D[m0_, r0_, m1_, r1_, d_] := Module[{k0 = k[m0, r0], k1 = k[m1, r1], rr, tt, bb, cc, sol},
   sol = First@Solve[{
       1 + rr == bb + cc,
       m0 I k0 (1 - rr) == m1 I k1 (bb - cc),
       bb Exp[I k1 d] + cc Exp[-I k1 d] == tt,
       m1 I k1 (bb Exp[I k1 d] - cc Exp[-I k1 d]) == m0 I k0 tt}, {rr, tt, bb, cc}];
   {rr, tt} /. sol];
{rP, tP} = rt1D[lam0 + 2 mu0, rh0, lam1 + 2 mu1, rh1, dd] /. w -> om;
{rS, tS} = rt1D[mu0, rh0, mu1, rh1, dd] /. w -> om;

(* against the package: notebook 1's reference, a plane force at z_src above a layer of the same
   contrast. Reflected displacement at a receiver z above the layer = R * g(0 - z_src) * e^{-i k0 z}. *)
ref1 = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_reference.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
res1 = Module[{a0 = cplx[ref1["background"]["alpha"]], r0 = ref1["background"]["rho"],
    a1 = cplx[ref1["layer"]["alpha"]], r1 = ref1["layer"]["rho"], zs = ref1["z_src"], dl = ref1["D"], om1 = ref1["omega"],
    m0, m1, k0, g, rr},
   m0 = r0 a0^2; m1 = r1 a1^2; k0 = om1/a0;
   g[z_] := I/(2 m0 k0) Exp[I k0 Abs[z]];
   rr = First[rt1D[m0, r0, m1, r1, dl] /. w -> om1];
   Table[
    With[{z = rc["z"]},
     If[z >= 0, Nothing,
      Abs[(mat[rc["G_layer"]][[1, 1]] - mat[rc["G_background"]][[1, 1]]) - rr g[0 - zs] Exp[-I k0 z]]/
       Abs[rr g[0 - zs]]]],
    {rc, ref1["layered"]}]];
Print["  [1] normal-incidence P reflection, closed form vs the package's layered field: worst ",
  sci[Max[res1]], " -> ", chk[Max[res1] < 10^-6]];
enP = Abs[rP]^2 + Abs[tP]^2 /. dd -> 2;
Print["      energy |R|^2 + |T|^2 - 1 (lossless, D = 2 m): P ", sci[N[enP] - 1], ",  S ",
  sci[N[Abs[rS]^2 + Abs[tS]^2 /. dd -> 2] - 1]];

(* ---------------------------------------------------------------------------
   [2] Taylor series in D, coefficients exact in the contrast (symbolic media).
   --------------------------------------------------------------------------- *)
Clear[mm0, mm1, rr0, rr1];
{rSym, tSym} = rt1D[mm0, rr0, mm1, rr1, dd];
serR = Simplify[Normal[Series[rSym, {dd, 0, 3}]], Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}];
serT = Simplify[Normal[Series[tSym Exp[-I w Sqrt[rr0/mm0] dd], {dd, 0, 2}]],
   Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}];
Print["  [2] R to O(D^3), exact in the contrast (M, rho: modulus and density; 0 background, 1 layer):"];
Do[Print["      D^", j, ":  ", InputForm[FullSimplify[Coefficient[serR, dd, j],
     Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}]]], {j, 1, 3}];
Print["  T e^{-i k0 D} to O(D^2):"];
Do[Print["      D^", j, ":  ", InputForm[FullSimplify[Coefficient[serT, dd, j],
     Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}]]], {j, 0, 2}];
(* the first-order term is linear in (rho1 - rho0) and (1/M1 - 1/M0): an equivalent interface *)
first = FullSimplify[Coefficient[serR, dd, 1], Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}];
jumpForm = I w/(2 Sqrt[rr0 mm0]) (rr1 - rr0) (-1) mm0/1 + 0;  (* placeholder, replaced below *)
jumpForm = (I w/2) Sqrt[mm0/rr0] ((rr1 - rr0)/mm0 - rr0 (1/mm1 - 1/mm0));
Print["      O(D) coefficient = (i w/2) sqrt(M0/rho0) [ (rho1-rho0)/M0 - rho0 (1/M1 - 1/M0) ] : ",
  chk[PossibleZeroQ[FullSimplify[first - jumpForm, Assumptions -> {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0}]]]];
(* the same coefficients in the layer's wavenumber k1 = w sqrt(rho1/M1) and the impedance ratio
   zeta = Z1/Z0, Z = sqrt(M rho): the form printed in the paper, to O(D^4) *)
Clear[zeta, kOne];
asm = {mm0 > 0, mm1 > 0, rr0 > 0, rr1 > 0, w > 0};
serR4 = Normal[Series[rSym, {dd, 0, 4}]];
toZ = {zeta -> Sqrt[mm1 rr1/(mm0 rr0)], kOne -> w Sqrt[rr1/mm1]};
zForm = {(I kOne/2) (zeta - 1/zeta), (kOne^2/4) (zeta^-2 - zeta^2),
    -(I kOne^3/24) (zeta - 1/zeta) (3 zeta^-2 + 2 + 3 zeta^2), (kOne^4/48) (zeta^2 - zeta^-2) (3 zeta^2 - 2 + 3 zeta^-2)};
zOK = Table[PossibleZeroQ[FullSimplify[Coefficient[serR4, dd, j] - (zForm[[j]] /. toZ), Assumptions -> asm]], {j, 4}];
(* the D^4 coefficient, found in the same variables: solve for it as a polynomial in zeta *)
(* Z1 = zeta Z0, rho1 = k1 Z1/w, M1 = Z1^2/rho1 = w zeta Z0/k1 *)
c4 = FullSimplify[Coefficient[serR4, dd, 4] /. {mm1 -> w zeta Sqrt[mm0 rr0]/kOne, rr1 -> kOne zeta Sqrt[mm0 rr0]/w},
   Assumptions -> Join[asm, {zeta > 0, kOne > 0}]];
Print["      the same, in k1 and zeta = Z1/Z0:  D^1..D^4 ", InputForm /@ zForm, " -> ", chk[And @@ zOK]];
Print["      D^4 coefficient: ", InputForm[Factor[c4]]];
errSer = Table[Abs[(rP - (serR /. {mm0 -> lam0 + 2 mu0, mm1 -> lam1 + 2 mu1, rr0 -> rh0, rr1 -> rh1, w -> om}))/rP] /. dd -> dv,
   {dv, {2, 1, 0.5, 0.25}}];
Print["      series (to D^3) vs exact, D = 2, 1, 0.5, 0.25 m: ", sci /@ N[errSer], "  -> ratio per halving ",
  sci /@ N[Most[errSer]/Rest[errSer]], " (D^3 truncation: 8)"];

(* ---------------------------------------------------------------------------
   [3] oblique incidence.  P-SV A matrix, SH A matrix; plane-wave states.
   --------------------------------------------------------------------------- *)
aPSV[lm_, m_, r_, p_] := {
   {0, -I w p, 1/m, 0},
   {-I w p lm/(lm + 2 m), 0, 0, 1/(lm + 2 m)},
   {-r w^2 + w^2 p^2 4 m (lm + m)/(lm + 2 m), 0, 0, -I w p lm/(lm + 2 m)},
   {0, -r w^2, -I w p, 0}};
aSH[m_, r_, p_] := {{0, 1/m}, {-r w^2 + m w^2 p^2, 0}};
eta[c_, p_] := Sqrt[1/c^2 - p^2];
(* P-SV plane-wave state (u_x, u_z, t_xz, t_zz) for a displacement u, vertical slowness q (signed) *)
stateOf[lm_, m_, u_, p_, q_] := {u[[1]], u[[2]], m I w (q u[[1]] + p u[[2]]), lm I w p u[[1]] + (lm + 2 m) I w q u[[2]]};
waves[lm_, m_, r_, p_] := Module[{a = Sqrt[(lm + 2 m)/r], b = Sqrt[m/r], ea, eb},
   ea = eta[a, p]; eb = eta[b, p];
   <|"Pd" -> {stateOf[lm, m, {p, ea}, p, ea], ea}, "Pu" -> {stateOf[lm, m, {p, -ea}, p, -ea], -ea},
     "Sd" -> {stateOf[lm, m, {eb, -p}, p, eb], eb}, "Su" -> {stateOf[lm, m, {-eb, -p}, p, -eb], -eb}|>];
(* route 1: layer matrix exponential.  incident P down, unit; R referred to z = 0, T to z = D *)
rtRoute1[p_, d_, ww_] := Module[{w0 = waves[lam0, mu0, rh0, p] /. w -> ww, prop, rpp, rps, tpp, tps, sol},
   prop = MatrixExp[(aPSV[lam1, mu1, rh1, p] /. w -> ww) d];
   sol = First@Solve[Thread[
       prop . (w0["Pd"][[1]] + rpp w0["Pu"][[1]] + rps w0["Su"][[1]]) == tpp w0["Pd"][[1]] + tps w0["Sd"][[1]]],
      {rpp, rps, tpp, tps}];
   {rpp, rps, tpp, tps} /. sol];
(* route 2: explicit plane waves inside the layer *)
rtRoute2[p_, d_, ww_] := Module[{w0 = waves[lam0, mu0, rh0, p] /. w -> ww, w1 = waves[lam1, mu1, rh1, p] /. w -> ww,
    c, rpp, rps, tpp, tps, sol, top, bot},
   top = Sum[c[j] w1[[j]][[1]], {j, 4}];
   bot = Sum[c[j] w1[[j]][[1]] Exp[I ww w1[[j]][[2]] d], {j, 4}];
   sol = First@Solve[Join[
       Thread[w0["Pd"][[1]] + rpp w0["Pu"][[1]] + rps w0["Su"][[1]] == top],
       Thread[bot == tpp w0["Pd"][[1]] + tps w0["Sd"][[1]]]], {rpp, rps, tpp, tps, c[1], c[2], c[3], c[4]}];
   {rpp, rps, tpp, tps} /. sol];
(* vertical energy flux of a state: < w/2 Im(u* . t) > with the sign fixed so downgoing is positive *)
flux[s_] := (om/2) Im[Conjugate[s[[1 ;; 2]]] . s[[3 ;; 4]]];
pa = N[Sin[Pi/6]/al0];  (* 30 degrees P incidence *)
r1v = rtRoute1[pa, 2., om]; r2v = rtRoute2[pa, 2., om];
Print["  [3] oblique P-SV, 30 deg, D = 2 m: route 1 (matrix exponential) vs route 2 (plane waves): ",
  sci[Max[Abs[r1v - r2v]]/Max[Abs[r1v]]], " -> ", chk[Max[Abs[r1v - r2v]]/Max[Abs[r1v]] < 10^-10]];
Module[{w0 = waves[lam0, mu0, rh0, pa] /. w -> om, fi, fo},
  fi = flux[w0["Pd"][[1]]];
  fo = Abs[r1v[[1]]]^2 flux[w0["Pu"][[1]]] + Abs[r1v[[2]]]^2 flux[w0["Su"][[1]]] +
    Abs[r1v[[3]]]^2 flux[w0["Pd"][[1]]] + Abs[r1v[[4]]]^2 flux[w0["Sd"][[1]]];
  (* upgoing fluxes are negative: incident = transmitted - reflected, i.e. fi = |fo_down| + |fo_up| *)
  Print["      energy: incident flux ", sci[fi], ", reflected + transmitted ",
   sci[Abs[r1v[[1]]]^2 Abs[flux[w0["Pu"][[1]]]] + Abs[r1v[[2]]]^2 Abs[flux[w0["Su"][[1]]]] +
     Abs[r1v[[3]]]^2 flux[w0["Pd"][[1]]] + Abs[r1v[[4]]]^2 flux[w0["Sd"][[1]]]], "  -> ",
   chk[Abs[(Abs[r1v[[1]]]^2 Abs[flux[w0["Pu"][[1]]]] + Abs[r1v[[2]]]^2 Abs[flux[w0["Su"][[1]]]] +
         Abs[r1v[[3]]]^2 flux[w0["Pd"][[1]]] + Abs[r1v[[4]]]^2 flux[w0["Sd"][[1]]]) - fi]/fi < 10^-10]]];
rNear = rtRoute1[10.^-8/al0, 2., om];
Print["      p -> 0: R_PP vs [1]'s R_P ", sci[Abs[-rNear[[1]] - (rP /. dd -> 2)]/Abs[rP /. dd -> 2]],
  " (the P-SV R_PP is referred to the potential sign; |R| compared: ",
  sci[Abs[Abs[rNear[[1]]] - Abs[rP /. dd -> 2]]/Abs[rP /. dd -> 2]], ")"];

(* ---------------------------------------------------------------------------
   [4] the series in D at oblique incidence.  The propagator of a thin layer is
   I + A1 D + A1^2 D^2/2 + ...; to first order the layer is the interface jump
   (A1 - A0) D, since the background propagator e^{A0 D} is divided out.
   --------------------------------------------------------------------------- *)
(* the D-series of R by ORDER-BY-ORDER perturbation of the linear system M(D) x = r(D), with M and r
   polynomial in D through the truncated propagator: M0 x_k = r_k - sum_{j=1..k} M_j x_{k-j} *)
seriesCoeffs[p_, order_, aL_] := Module[{w0 = waves[lam0, mu0, rh0, p] /. w -> om, mj, rj, x},
   mj[0] = Transpose[{w0["Pu"][[1]], w0["Su"][[1]], -w0["Pd"][[1]], -w0["Sd"][[1]]}];
   rj[0] = -w0["Pd"][[1]];
   Do[With[{pj = MatrixPower[aL, j]/j!},
      mj[j] = Transpose[{pj . w0["Pu"][[1]], pj . w0["Su"][[1]], {0, 0, 0, 0}, {0, 0, 0, 0}}];
      rj[j] = -pj . w0["Pd"][[1]]], {j, 1, order}];
   Do[x[kk] = LinearSolve[mj[0], rj[kk] - Sum[mj[j] . x[kk - j], {j, 1, kk}]], {kk, 0, order}];
   Table[x[kk][[1 ;; 2]], {kk, 0, order}]];  (* {R_PP, R_PS} coefficients of D^0 .. D^order *)
a1pa = aPSV[lam1, mu1, rh1, pa] /. w -> om;
Print["  [4] R_PP and R_PS at 30 deg: exact vs the D-series truncated at O(D^k):"];
Do[Module[{cf = seriesCoeffs[pa, ord, a1pa], errs},
   errs = Table[Module[{ex = rtRoute1[pa, dv, om][[1 ;; 2]]},
      Max[Abs[Sum[cf[[j + 1]] dv^j, {j, 0, ord}] - ex]]/Max[Abs[ex]]], {dv, {1., 0.5, 0.25}}];
   Print["      k = ", ord, ":  D = 1, 0.5, 0.25 m: ", sci /@ errs, "   ratio per halving ",
    sci /@ (Most[errs]/Rest[errs]), "  (relative to R ~ D: expect 2^", ord, ") -> ",
    chk[Max[Abs[(Most[errs]/Rest[errs])/2^ord - 1]] < 0.05]]],
  {ord, {1, 2, 3}}];
(* equivalent interface: state below = (I + (A1 - A0) D) state above; its O(D) R coefficient is the
   first-order perturbation with A1 - A0 in place of A1 (the A0 D part only rephases the transmission) *)
Module[{c1 = seriesCoeffs[pa, 1, a1pa][[2]], cJ = seriesCoeffs[pa, 1, a1pa - (aPSV[lam0, mu0, rh0, pa] /. w -> om)][[2]]},
  Print["      equivalent interface (A1 - A0) D: O(D) coefficients R_PP, R_PS ", sci /@ cJ, " vs the layer's ",
   sci /@ c1, " -> ", chk[Max[Abs[cJ - c1]] < 10^-12 Max[Abs[c1]]]]];


(* ---------------------------------------------------------------------------
   [5] asymptotics: a SOFT thin layer.  SH, layer shear modulus mu1 = D / etaC at fixed compliance
   etaC; as D -> 0 the layer becomes a linear-slip interface: t continuous, [u_y] = etaC t.
   --------------------------------------------------------------------------- *)
Module[{etaC = 1.*^-11, pS = N[Sin[Pi/6]/be0], eb0, zs0, rSlip, rLayer, errs},
  eb0 = eta[be0, pS];
  (* SH states (u_y, t_yz): down (1, i w mu0 eb0), up (1, -i w mu0 eb0) *)
  rLayer[d_] := Module[{m1 = d/etaC, prop, rr, tt, sol},
    prop = MatrixExp[(aSH[m1, rh0, pS] /. w -> om) d];
    sol = First@Solve[Thread[prop . ({1, I om mu0 eb0} + rr {1, -I om mu0 eb0}) == tt {1, I om mu0 eb0}], {rr, tt}];
    rr /. sol];
  (* linear slip: t continuous, u jumps by etaC t *)
  rSlip = Module[{rr, tt, sol}, sol = First@Solve[{
       tt == (1 + rr) + etaC (I om mu0 eb0 (1 - rr)),
       tt I om mu0 eb0 == I om mu0 eb0 (1 - rr)}, {rr, tt}]; rr /. sol];
  errs = Table[Abs[rLayer[d] - rSlip]/Abs[rSlip], {d, {0.1, 0.05, 0.025}}];
  Print["  [5] soft thin SH layer (mu1 = D/eta, eta = ", etaC, " m/Pa) -> linear slip: |R_layer - R_slip|/|R_slip| at D = 0.1, 0.05, 0.025 m: ",
   sci /@ errs, "  -> falls as D: ", chk[errs[[1]]/errs[[2]] > 1.8 && errs[[2]]/errs[[3]] > 1.8]];
  Print["      R_slip = ", sci[rSlip], "  (Schoenberg's linear-slip interface)"]];

Print["==== ContinuumLimit_ThinLayer (stage 2c): ", If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
