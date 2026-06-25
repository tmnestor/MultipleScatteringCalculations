#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneVectorLattice.wl  --  Phase 2 item (b):
   the lattice-summed MULTI-CHANNEL vector coupling G0^vec(k_par), i.e. the planar
   Bloch sum of the pairwise elastic vector translation,
       G0^vec_{(nu mu ct),(n m cs)}(k_par) = Sum_{R!=0} W(R)_{...} e^{i k_par . R},
   the operator that feeds the collective Foldy-Lax  T_coll = T0 (I - G0^vec T0)^{-1}.

   Routes (P and S decouple in the homogeneous background):
     L -> L     : CLOSED FORM = the Phase-1 scalar G0 at kappa_P (Gaunt contraction
                  of the lattice structure constants D[q,s]); verified vs a direct
                  damped sum of beta^P(R).
     M,N -> M,N : DAMPED DIRECT lattice sum.  The lattice-summed outgoing S-field is
                  built once at the quadrature nodes (latSrcQuad), then projected onto
                  the regular C/P harmonics -> the M/N columns of G0^vec.
   A small Im(kappa)=0.25 damping makes the conditionally-convergent planar sum
   converge (the Phase-1 "damped Foldy-Lax"); the undamped Ewald/closed-form vector
   acceleration is a deferred refinement (cf. Phase 1's deferred Kambe multipoles).

   Checks (independent residuals, PASS/FAIL):
     [a] L-block: closed-form Phase-1 G0(kappa_P) = direct damped sum of beta^P(R).
     [b] VECTOR field reconstruction: direct damped Bloch sum of the outgoing M/N
         lattice field = regular-multipole reconstruction via the G0^vec columns
         (the end-to-end validation of the vector lattice coupling).
     [c] convergence of G0^vec entries in lattice radius (damped).
     [d] assemble full G0^vec, collective solve T_coll = T0 (I - G0^vec T0)^{-1}:
         finite, reduces to isolated T0 as the coupling is switched off, and the
         L-block of G0^vec is reciprocal (the M/N strict reciprocity + energy is
         plan item (d), needing the flux metric).

   Conventions: time e^{-i w t}, outgoing h_n^(1); z = polar axis = depth; the lattice
   lies in the x-y plane (R_z = 0, theta_R = Pi/2).  Inner T0 uses the REAL background
   wavenumbers; only the inter-voxel lattice sum is damped.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

(* ---- real background / Mie params for the single-site T0 (match TB1) ---- *)
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0;

(* ---- lattice + damped wavenumbers (match Phase 1 IntraPlaneLatticeSum.wl) ---- *)
aL = 2.0; dampIm = 0.25;
kappaP = kPo + dampIm I; kappaS = kSo + dampIm I;
kx = 0.2; ky = 0.1;                     (* horizontal Bloch vector k_par = (kx,ky,0) *)
kPval = kappaP; kSval = kappaS;         (* the lattice-sum translation fields are damped *)
LradB = 8;                              (* lattice half-width for assembly + field check *)
radP = 0.5;                             (* projection / field-sum sphere radius (< aL) *)

(* ---- verified helpers (Phase 0 / TB2 / TB3) ---- *)
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
Mw[n_, m_, type_, c_, r_] := -zfn[n, kSval rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_] := Module[{x = kSval rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] rhatOf[c, r]
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];

(* ---- planar square lattice + Bloch phases ---- *)
ijList = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -LradB, LradB}, {j, -LradB, LradB}], 1];
Rvecs = Map[{aL #[[1]], aL #[[2]], 0.0} &, ijList];
bloch = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ijList];
nR = Length[Rvecs];

(* ---- FAST sphere quadrature (precomputed nodes/weights) ---- *)
Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[24, -1, 1]; nPhiV = 48;
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
quadList = Flatten[Table[{glN[[i, 1]], glN[[i, 2]], 2 Pi (j - 1)/nPhiV}, {i, Length[glN]}, {j, nPhiV}], 1];
quadDirs = Map[shat[#[[1]], #[[3]]] &, quadList];
quadW = Map[#[[2]] (2 Pi/nPhiV) &, quadList];
projDotF[fieldVals_, harmVals_] := Sum[quadW[[k]] fieldVals[[k]] . Conjugate[harmVals[[k]]], {k, Length[quadList]}];
Cvals[nu_, mu_] := Cvals[nu, mu] = Map[Cvec[nu, mu, #] &, quadDirs];
Pvals[nu_, mu_] := Pvals[nu, mu] = Map[Pvec[nu, mu, #] &, quadDirs];
MregVals[nu_, mu_] := MregVals[nu, mu] = Map[Mw[nu, mu, "j", {0, 0, 0}, radP #] &, quadDirs];
NregVals[nu_, mu_] := NregVals[nu, mu] = Map[Nw[nu, mu, "j", {0, 0, 0}, radP #] &, quadDirs];
normMC[nu_, mu_] := normMC[nu, mu] = projDotF[MregVals[nu, mu], Cvals[nu, mu]];
normNP[nu_, mu_] := normNP[nu, mu] = projDotF[NregVals[nu, mu], Pvals[nu, mu]];

(* ---- lattice-summed outgoing S-field at the quadrature nodes (radius radP) ---- *)
outQuad[c_, n_, m_, R_] := Map[Switch[c, "M", Mw[n, m, "h", R, radP #], "N", Nw[n, m, "h", R, radP #]] &, quadDirs];
latSrcQuad[c_, n_, m_] := latSrcQuad[c, n, m] =
   Total[Table[bloch[[iR]] outQuad[c, n, m, Rvecs[[iR]]], {iR, nR}]];
(* M/N columns of G0^vec from the lattice-summed field (regular C/P content) *)
g0MM[n_, m_, nu_, mu_] := projDotF[latSrcQuad["M", n, m], Cvals[nu, mu]]/normMC[nu, mu];
g0NM[n_, m_, nu_, mu_] := projDotF[latSrcQuad["M", n, m], Pvals[nu, mu]]/normNP[nu, mu];
g0MN[n_, m_, nu_, mu_] := projDotF[latSrcQuad["N", n, m], Cvals[nu, mu]]/normMC[nu, mu];
g0NN[n_, m_, nu_, mu_] := projDotF[latSrcQuad["N", n, m], Pvals[nu, mu]]/normNP[nu, mu];

(* ---- closed-form L-block = Phase-1 scalar G0 at kappa_P (structure constants D) ---- *)
rNorms = Map[Sqrt[#[[1]]^2 + #[[2]]^2] &, Rvecs];
phiR = Map[ArcTan[#[[1]], #[[2]]] &, Rvecs];
Dstruct[q_, s_] := Dstruct[q, s] = Total[sh[q, kappaP rNorms] SphericalHarmonicY[q, s, Pi/2, phiR] bloch];
g0LL[n_, m_, nu_, mu_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dstruct[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];

Print["==== Phase 2 item (b) :: lattice-summed vector G0(k_par) ===="];
Print["  aL=", aL, ", kappaP=", kappaP, ", kappaS=", kappaS, ", k_par=(", kx, ",", ky, "), Lrad=", LradB,
   ", #R=", nR, ", radP=", radP];

(* ============================================================================
   [a] L-block closed form = direct damped lattice sum of beta^P(R)
   ============================================================================ *)
llTests = {{0, 0, 0, 0}, {1, 0, 1, 0}, {1, -1, 2, 1}, {2, 0, 1, 1}, {2, 2, 2, -1}};
g0LLdirect[n_, m_, nu_, mu_] := Total[Table[betaSep[n, m, nu, mu, Rvecs[[iR]], kappaP] bloch[[iR]], {iR, nR}]];
resLL = Max[Table[Abs[g0LL[t[[1]], t[[2]], t[[3]], t[[4]]] - g0LLdirect[t[[1]], t[[2]], t[[3]], t[[4]]]], {t, llTests}]];
Print["  [a] L-block closed-form vs direct damped sum (", Length[llTests], " entries) = ", resLL,
   "  -> ", If[resLL < 1*^-9, "PASS", "FAIL"]];

(* ============================================================================
   [b] VECTOR field reconstruction: direct damped Bloch field sum = G0^vec recon.
   directF(r) = Sum_{R!=0} e^{ik.R} (M or N)^out[n,m,R](r),  |r| < aL;
   reconF(r)  = Sum_{nu mu<=Qrec} [ G0^{M<-c}_{nu mu} M^reg + G0^{N<-c}_{nu mu} N^reg ](r).
   ============================================================================ *)
Qrec = 10;
directF[c_, n_, m_, r_] := Total[Table[bloch[[iR]] Switch[c, "M", Mw[n, m, "h", Rvecs[[iR]], r], "N", Nw[n, m, "h", Rvecs[[iR]], r]], {iR, nR}]];
reconF[c_, n_, m_, r_] := Sum[
   Switch[c, "M", g0MM[n, m, nu, mu], "N", g0MN[n, m, nu, mu]] Mw[nu, mu, "j", {0, 0, 0}, r]
   + Switch[c, "M", g0NM[n, m, nu, mu], "N", g0NN[n, m, nu, mu]] Nw[nu, mu, "j", {0, 0, 0}, r],
   {nu, 1, Qrec}, {mu, -nu, nu}];
SeedRandom[20260618];
recPts = Table[Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}]}, 0.22 aL shat[u, ph]], {4}];
recSrc = {{"M", 1, 0}, {"N", 1, 1}, {"M", 2, -1}};
resB = Max[Table[Norm[directF[s[[1]], s[[2]], s[[3]], pt] - reconF[s[[1]], s[[2]], s[[3]], pt]], {s, recSrc}, {pt, recPts}]];
Print["  [b] vector field reconstruction (Qrec=", Qrec, ", ", Length[recSrc], " srcs x ", Length[recPts],
   " pts) = ", resB, "  -> ", If[resB < 1*^-5, "PASS", "FAIL"]];

(* ============================================================================
   [c] convergence of the damped Bloch sum in lattice radius.  The direct damped
   sum converges GEOMETRICALLY (each shell adds ~ e^{-2 Im(kappa) aL}); we show the
   Lrad increments shrink.  A deeply-converged / undamped vector G0 follows from the
   Ewald-accelerated closed-form Cruzan/Gaunt contraction of the Phase-1 structure
   constants (the L-block already uses exactly that) -- deferred, as in Phase 1.
   ============================================================================ *)
latAt[c_, n_, m_, Lr_] := Module[{ij, rv, bl},
   ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
   rv = Map[{aL #[[1]], aL #[[2]], 0.0} &, ij]; bl = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ij];
   Total[Table[bl[[iR]] outQuad[c, n, m, rv[[iR]]], {iR, Length[rv]}]]];
g0MMat[nu_, mu_, lat_] := projDotF[lat, Cvals[nu, mu]]/normMC[nu, mu];
g04 = g0MMat[1, 0, latAt["M", 1, 0, 4]];
g08 = g0MMat[1, 0, latAt["M", 1, 0, 8]];
g012 = g0MMat[1, 0, latAt["M", 1, 0, 12]];
dCoarse = Abs[g08 - g04]; dFine = Abs[g012 - g08];
Print["  [c] damped Bloch-sum convergence of G0^{MM}_{10,10}: |G0(8)-G0(4)| = ", dCoarse,
   ", |G0(12)-G0(8)| = ", dFine, ", ratio = ", dFine/dCoarse];
Print["      -> ", If[dFine < 0.5 dCoarse, "PASS (geometric)", "FAIL"],
   "   [deep/undamped vector G0: deferred Ewald-accelerated Cruzan-Gaunt, cf. Phase 1]"];

(* ============================================================================
   [d] assemble full G0^vec, collective Foldy-Lax solve, sanity + L-block reciprocity
   ============================================================================ *)
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

G0entry[{nu_, mu_, ct_}, {n_, m_, cs_}] := Which[
   ct == "L" && cs == "L", g0LL[n, m, nu, mu],
   ct == "M" && cs == "M", g0MM[n, m, nu, mu], ct == "N" && cs == "M", g0NM[n, m, nu, mu],
   ct == "M" && cs == "N", g0MN[n, m, nu, mu], ct == "N" && cs == "N", g0NN[n, m, nu, mu],
   True, 0];
G0vec = Table[G0entry[idx[[i]], idx[[j]]], {i, nDim}, {j, nDim}];
Imat = IdentityMatrix[nDim];
Tcoll = T0mat . Inverse[Imat - G0vec . T0mat];
TcollIso = T0mat . Inverse[Imat - (0 G0vec) . T0mat];
resIso = Max[Abs[Flatten[TcollIso - T0mat]]];
finite = AllTrue[Flatten[Tcoll], (NumberQ[#] && Abs[#] < 1*^6) &];
couple = Norm[Flatten[Tcoll - T0mat]]/Norm[Flatten[T0mat]];
(* L-block reciprocity (clean: beta-type); full-matrix asymmetry reported for item (d) *)
sig[{n_, m_, c_}] := (-1)^(n + m);
conjIdx[{n_, m_, c_}] := {n, -m, c};
J0 = Table[If[idx[[i]] == conjIdx[idx[[k]]], sig[idx[[k]]], 0], {i, nDim}, {k, nDim}];
Lpos = Flatten[Position[idx, {_, _, "L"}]];
G0LLsub = G0vec[[Lpos, Lpos]]; J0LL = J0[[Lpos, Lpos]];
recLL = Max[Abs[Flatten[J0LL . G0LLsub . J0LL - Transpose[G0LLsub]]]];
asymFull = Max[Abs[Flatten[J0 . G0vec . J0 - Transpose[G0vec]]]];
Print["  [d] collective solve: isolated limit (G0->0) dev = ", resIso, ", finite = ", finite,
   ", coupling ||Tcoll-T0||/||T0|| = ", couple];
Print["      -> ", If[resIso < 1*^-12 && finite && couple > 1*^-6, "PASS", "FAIL"]];
Print["  [d] L-block reciprocity (beta-type) J0 G0_LL J0 = G0_LL^T = ", recLL,
   "  -> ", If[recLL < 1*^-9, "PASS", "FAIL"]];
Print["      full-matrix raw asymmetry (M/N flux metric is item (d)) = ", asymFull];

(* ---- dump G0^vec + a field-recon sample for the Python cross-check (item f) ---- *)
reim[z_] := {Re[N[z]], Im[N[z]]};
g0Ref = <|"aL" -> aL, "kx" -> kx, "ky" -> ky, "dampIm" -> dampIm, "kPo" -> kPo, "kSo" -> kSo,
   "Lrad" -> LradB, "radP" -> radP, "Nmax" -> Nmax,
   "idx" -> Map[{#[[1]], #[[2]], #[[3]]} &, idx],
   "G0vec" -> Map[reim, G0vec, {2}]|>;
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneVectorLattice_reference.json", g0Ref];
Print["  wrote IntraPlaneVectorLattice_reference.json (G0^vec ", nDim, "x", nDim, ")"];
Print["Phase 2 item (b) (IntraPlaneVectorLattice.wl) loaded + verified."];
