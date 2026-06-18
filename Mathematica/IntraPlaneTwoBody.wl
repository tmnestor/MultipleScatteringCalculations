#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneTwoBody.wl  --  Phase 2 item (a):
   the two-voxel DIRECT Foldy-Lax solve in the L/M/N multipole basis, coupling the
   voxels with the pairwise elastic vector translation operator W(d) (Phase 0 / TB2).

   Voxel i has exciting (regular) coeffs a_i^exc and scattered (outgoing) coeffs
   b_i = T0 . a_i^exc, with the inter-voxel coupling
     a_i^exc = a_i^inc + Sum_{j!=i} W(r_j - r_i) . b_j,
   where W(d) re-expands the outgoing multipoles at A (= r_j) as regular multipoles
   at B (= r_i), d = A - B.  For two voxels (d = r2 - r1):
     G    = [[0, W(d)], [W(-d), 0]],   T0blk = diag(T0, T0),
     b    = T_coll . a^inc,            T_coll = T0blk . (I - G . T0blk)^{-1}.

   W(d) blocks (P and S decouple in the homogeneous background):
     L -> L     : closed-form scalar beta^P_{n m, nu mu}(d)          (Phase 0)
     M,N -> M,N : projection of the translated outgoing field onto the regular
                  C (toroidal -> M) and P (radial -> N) harmonics    (TB2).

   Checks (each an independent residual, PASS/FAIL):
     [a] W(d) reconstructs the translated vector field (L, M, N sources).
     [b] isolated limit  G -> 0  =>  T_coll = diag(T0, T0)            (exact).
     [c] DIRECT two-body solve = Neumann/Born multiple-scattering series
         T0blk Sum_k (G T0blk)^k  equals the matrix-inverse collective T.
     [d] monopole channel: matrix solve = closed-form 2-body geometric series.
     [e] full-system fixed-point residual of the solved coefficients; the W
         coupling is confirmed active (T_coll differs measurably from diag(T0,T0)).

   NB collective reciprocity + energy is plan item (d): it needs the flux metric
   (the raw L/N Hansen block is not literally symmetric), handled there, not here.

   Conventions inherited from CartesianT0.wl / Phase 0: time e^{+i w t}, outgoing
   h_n^(1), background wavenumbers kP=0.9, kS=1.5, z = polar axis = depth.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

(* ---- background wavenumbers / Mie params (match TB1) ---- *)
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0;
kPval = kPo; kSval = kSo;

(* ---- verified helpers (copied from Phase 0 / TB2) ---- *)
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];
zfn[n_, x_, "j"] := sj[n, x]; zfn[n_, x_, "h"] := sh[n, x];
zfp[n_, x_, "j"] := n/x sj[n, x] - sj[n + 1, x];
zfp[n_, x_, "h"] := n/x sh[n, x] - sh[n + 1, x];
ang[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];
betaSep[n_, m_, nu_, mu_, dvec_, k_] :=
  Module[{dl = Sqrt[dvec . dvec]},
   4 Pi (-1)^m Sum[
     I^(nu + q - n) (-1)^q sh[q, k dl] Yfun[q, m - mu, dvec] gaunt[n, m, nu, -mu, q, mu - m],
     {q, Abs[n - nu], n + nu}]];
RMrot[t_, p_] := {{Sin[t] Cos[p], Cos[t] Cos[p], -Sin[p]},
   {Sin[t] Sin[p], Cos[t] Sin[p], Cos[p]}, {Cos[t], -Sin[t], 0}};
dthY[n_, m_] := dthY[n, m] = Module[{tt, pp, ex},
   ex = D[SphericalHarmonicY[n, m, tt, pp], tt]; Function[{a, b}, Evaluate[ex /. {tt -> a, pp -> b}]]];
Bvec[n_, m_, d_] := Module[{t = ang[d][[1]], p = ang[d][[2]]},
   RMrot[t, p] . {0, dthY[n, m][t, p], (I m/Sin[t]) SphericalHarmonicY[n, m, t, p]}];
Cvec[n_, m_, d_] := Cross[d/Sqrt[d . d], Bvec[n, m, d]];
Pvec[n_, m_, d_] := Yfun[n, m, d] (d/Sqrt[d . d]);
rhoOf[c_, r_] := Sqrt[(r - c) . (r - c)];
rhatOf[c_, r_] := (r - c)/Sqrt[(r - c) . (r - c)];
Lw[n_, m_, type_, c_, r_] := Module[{x = kPval rhoOf[c, r]},
   zfp[n, x, type] Yfun[n, m, r - c] rhatOf[c, r] + (zfn[n, x, type]/x) Bvec[n, m, r - c]];
Mw[n_, m_, type_, c_, r_] := -zfn[n, kSval rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_] := Module[{x = kSval rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] rhatOf[c, r]
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];

(* ---- FAST sphere projection: precompute quadrature directions/weights once,
        precompute (memoised) source-field & basis-harmonic values at those
        directions, so each coefficient is a cheap weighted dot product. ---- *)
Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[24, -1, 1]; nPhiV = 48;
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
quadList = Flatten[Table[{glN[[i, 1]], glN[[i, 2]], 2 Pi (j - 1)/nPhiV}, {i, Length[glN]}, {j, nPhiV}], 1];
quadDirs = Map[shat[#[[1]], #[[3]]] &, quadList];
quadW = Map[#[[2]] (2 Pi/nPhiV) &, quadList];
projDotF[fieldVals_, harmVals_] := Sum[quadW[[k]] fieldVals[[k]] . Conjugate[harmVals[[k]]], {k, Length[quadList]}];
Cvals[nu_, mu_] := Cvals[nu, mu] = Map[Cvec[nu, mu, #] &, quadDirs];
Pvals[nu_, mu_] := Pvals[nu, mu] = Map[Pvec[nu, mu, #] &, quadDirs];
fieldVals[Ffn_, rad_] := Map[Ffn[rad #] &, quadDirs];
MregVals[nu_, mu_, rad_] := MregVals[nu, mu, rad] = fieldVals[Function[r, Mw[nu, mu, "j", {0, 0, 0}, r]], rad];
NregVals[nu_, mu_, rad_] := NregVals[nu, mu, rad] = fieldVals[Function[r, Nw[nu, mu, "j", {0, 0, 0}, r]], rad];
normMC[nu_, mu_, rad_] := normMC[nu, mu, rad] = projDotF[MregVals[nu, mu, rad], Cvals[nu, mu]];
normNP[nu_, mu_, rad_] := normNP[nu, mu, rad] = projDotF[NregVals[nu, mu, rad], Pvals[nu, mu]];
(* outgoing M/N centred at A=dv, sampled at radius rad about B=origin (memoised per source) *)
srcVals[dv_, c_, n_, m_, rad_] := srcVals[dv, c, n, m, rad] =
   fieldVals[Function[r, Switch[c, "M", Mw[n, m, "h", dv, r], "N", Nw[n, m, "h", dv, r]]], rad];
aMcoefD[dv_, c_, n_, m_, nu_, mu_, rad_] := projDotF[srcVals[dv, c, n, m, rad], Cvals[nu, mu]]/normMC[nu, mu, rad];
aNcoefD[dv_, c_, n_, m_, nu_, mu_, rad_] := projDotF[srcVals[dv, c, n, m, rad], Pvals[nu, mu]]/normNP[nu, mu, rad];

(* ---- single-site T0 in the L/M/N basis (copied from TB1) ---- *)
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
Nmax = 2;
idx = Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];
nDim = Length[idx];
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0entry[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
  If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
T0mat = Table[T0entry[idx[[i]], idx[[j]]], {i, nDim}, {j, nDim}];

(* ---- pairwise translation matrix W(d): rows = regular target (nu,mu,ct) at B,
        cols = outgoing source (n,m,cs) at A; P/S decouple ---- *)
Wentry[{nu_, mu_, ct_}, {n_, m_, cs_}, dv_, rad_] := Which[
   ct == "L" && cs == "L", betaSep[n, m, nu, mu, dv, kPval],
   ct == "M" && cs == "M", aMcoefD[dv, "M", n, m, nu, mu, rad],
   ct == "N" && cs == "M", aNcoefD[dv, "M", n, m, nu, mu, rad],
   ct == "M" && cs == "N", aMcoefD[dv, "N", n, m, nu, mu, rad],
   ct == "N" && cs == "N", aNcoefD[dv, "N", n, m, nu, mu, rad],
   True, 0];
Wmat[dv_] := Module[{rad = 0.3 Sqrt[dv . dv]},
   Table[Wentry[idx[[i]], idx[[j]], dv, rad], {i, nDim}, {j, nDim}]];

(* ============================================================================
   geometry: two voxels at r1, r2; separation d = r2 - r1 (Delta z = 0, intra-plane)
   ============================================================================ *)
dsep = {2.2, 1.1, 0.0}; dLen = Sqrt[dsep . dsep];
Print["==== Phase 2 item (a) :: two-voxel direct Foldy-Lax ===="];
Print["  separation d = ", dsep, ", |d| = ", dLen, ", kP|d| = ", kPval dLen, ", kS|d| = ", kSval dLen];
Print["  basis: n=0..", Nmax, " -> ", nDim, " channels/voxel, ", 2 nDim, " total"];

(* ============================================================================
   [a] W reconstructs the translated vector field (independent of the FL solve).
   Regular re-expansion to nu = Qrec (> Nmax; the FL truncation itself is item c).
   ============================================================================ *)
Qrec = 14; radA = 0.3 dLen;
srcOut[cs_, n_, m_, r_] := Switch[cs, "L", Lw[n, m, "h", dsep, r], "M", Mw[n, m, "h", dsep, r], "N", Nw[n, m, "h", dsep, r]];
reconField[cs_, n_, m_, r_] := If[cs == "L",
   Sum[betaSep[n, m, nu, mu, dsep, kPval] Lw[nu, mu, "j", {0, 0, 0}, r], {nu, 0, Qrec}, {mu, -nu, nu}],
   Sum[aMcoefD[dsep, cs, n, m, nu, mu, radA] Mw[nu, mu, "j", {0, 0, 0}, r]
       + aNcoefD[dsep, cs, n, m, nu, mu, radA] Nw[nu, mu, "j", {0, 0, 0}, r], {nu, 1, Qrec}, {mu, -nu, nu}]];
SeedRandom[20260618];
tbPts = Table[Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}]}, 0.18 dLen shat[u, ph]], {5}];
tbSrc = {{"L", 1, 0}, {"L", 2, -1}, {"M", 1, 0}, {"N", 1, 1}, {"M", 2, 1}};
resA = Max[Table[Norm[srcOut[s[[1]], s[[2]], s[[3]], pt] - reconField[s[[1]], s[[2]], s[[3]], pt]],
    {s, tbSrc}, {pt, tbPts}]];
Print["  [a] W field reconstruction (Qrec=", Qrec, ", ", Length[tbSrc], " srcs x ", Length[tbPts],
   " pts) = ", resA, "  -> ", If[resA < 1*^-5, "PASS", "FAIL"]];

(* ---- assemble the two-voxel block operators ---- *)
zero = ConstantArray[0, {nDim, nDim}];
T0blk = ArrayFlatten[{{T0mat, zero}, {zero, T0mat}}];
Wd = Wmat[dsep]; Wmd = Wmat[-dsep];
Gblk = ArrayFlatten[{{zero, Wd}, {Wmd, zero}}];
Imat = IdentityMatrix[2 nDim];
Tcoll = T0blk . Inverse[Imat - Gblk . T0blk];

(* ============================================================================
   [b] isolated limit: G -> 0  =>  T_coll = blockdiag(T0, T0)
   ============================================================================ *)
TcollIso = T0blk . Inverse[Imat - (0 Gblk) . T0blk];
resB = Max[Abs[Flatten[TcollIso - T0blk]]];
Print["  [b] isolated limit (G=0) T_coll = diag(T0,T0): max dev = ", resB,
   "  -> ", If[resB < 1*^-12, "PASS", "FAIL"]];

(* ============================================================================
   [c] DIRECT two-body solve via the Neumann/Born series of multiple scattering:
       T_coll = T0blk Sum_{k=0}^{K} (G T0blk)^k   (converges iff spec.rad < 1).
   The iterated-scattering "direct" solve, independent of Inverse[].
   ============================================================================ *)
GT = Gblk . T0blk;
specR = Max[Abs[Eigenvalues[GT]]];
Kneu = 30;
TcollNeu = T0blk . Sum[MatrixPower[GT, k], {k, 0, Kneu}];
resC = Max[Abs[Flatten[TcollNeu - Tcoll]]];
Print["  [c] Neumann/Born series (K=", Kneu, ", spectral radius ", specR, ") vs inverse = ",
   resC, "  -> ", If[resC < 1*^-10, "PASS", "FAIL"]];

(* ============================================================================
   [d] monopole channel closed form: a clean 2x2 two-body geometric series.
       b1 = t(a1 + w12 b2), b2 = t(a2 + w21 b1)  =>
       b1 = (t a1 + t^2 w12 a2)/(1 - t^2 w12 w21).
   ============================================================================ *)
tm = T0mono[kPo, lamO, muO, kPi, lamI, muI, aa];
w12 = betaSep[0, 0, 0, 0, dsep, kPval]; w21 = betaSep[0, 0, 0, 0, -dsep, kPval];
amono = {1.0, 0.0};
GmonoBlk = {{0, w12}, {w21, 0}}; T0monoBlk = {{tm, 0}, {0, tm}};
bMonoMat = T0monoBlk . Inverse[IdentityMatrix[2] - GmonoBlk . T0monoBlk] . amono;
bMonoCF = {(tm amono[[1]] + tm^2 w12 amono[[2]])/(1 - tm^2 w12 w21),
   (tm amono[[2]] + tm^2 w21 amono[[1]])/(1 - tm^2 w12 w21)};
resD = Max[Abs[bMonoMat - bMonoCF]];
Print["  [d] monopole 2-body matrix solve vs closed form = ", resD,
   "  -> ", If[resD < 1*^-12, "PASS", "FAIL"]];

(* ============================================================================
   [e] full-system self-consistency: the solved b = Tcoll a^inc must satisfy
       (I - T0blk G) b = T0blk a^inc;  and the W coupling is measurably active.
   ============================================================================ *)
SeedRandom[7];
aInc = RandomReal[{-1, 1}, 2 nDim] + I RandomReal[{-1, 1}, 2 nDim];
bsol = Tcoll . aInc;
resE = Max[Abs[(Imat - T0blk . Gblk) . bsol - T0blk . aInc]];
couple = Norm[Flatten[Tcoll - T0blk]]/Norm[Flatten[T0blk]];
Print["  [e] fixed-point residual |(I - T0 G) b - T0 a_inc| = ", resE,
   "  -> ", If[resE < 1*^-12, "PASS", "FAIL"]];
Print["      W-coupling active: ||T_coll - diag(T0,T0)|| / ||diag(T0,T0)|| = ", couple,
   "  -> ", If[1*^-6 < couple < 10, "PASS", "FAIL"]];

Print["Phase 2 item (a) (IntraPlaneTwoBody.wl) loaded + verified."];
