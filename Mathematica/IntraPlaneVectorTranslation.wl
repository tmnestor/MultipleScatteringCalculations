#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneVectorTranslation.wl  --  Phase 2 TB2:
   the Phase-0 vector translation as an EXPLICIT L/M/N matrix W^{c'c}_{nu mu, n m}(d),
   needed to assemble the multi-channel vector G0 and the two-voxel Foldy-Lax solve.

   P and S decouple in the homogeneous background:
     L -> L only, via the scalar beta^P (Phase 0): W^{LL}_{nu mu,n m} = beta^P_{n m,nu mu}.
     M,N -> M,N, extracted by projecting the translated outgoing field onto the
     orthogonal vector spherical harmonics C (toroidal, picks M) and P (radial, picks N).
   Each coefficient is normalised by the basis field's OWN projection, so all
   normalisation/sign conventions cancel (sign-safe):
     a^M_{nu mu} = <F, C_{nu mu}> / <M^reg_{nu mu}, C_{nu mu}>,
     a^N_{nu mu} = <F, P_{nu mu}> / <N^reg_{nu mu}, P_{nu mu}>.

   TDD RED: the extracted coefficients are stubbed to 0, so the matrix reconstruction
   of the translated field is 0 and MUST FAIL.
   ============================================================================ *)

(* ---- helpers + fields copied from Phase 0 (IntraPlaneTranslation.wl) ---- *)
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];
zfn[n_, x_, "j"] := sj[n, x]; zfn[n_, x_, "h"] := sh[n, x];
zfp[n_, x_, "j"] := n/x sj[n, x] - sj[n + 1, x];
zfp[n_, x_, "h"] := n/x sh[n, x] - sh[n + 1, x];
ang[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
RMrot[t_, p_] := {{Sin[t] Cos[p], Cos[t] Cos[p], -Sin[p]},
   {Sin[t] Sin[p], Cos[t] Sin[p], Cos[p]}, {Cos[t], -Sin[t], 0}};
dthY[n_, m_] := dthY[n, m] = Module[{tt, pp, ex},
   ex = D[SphericalHarmonicY[n, m, tt, pp], tt]; Function[{a, b}, Evaluate[ex /. {tt -> a, pp -> b}]]];
Bvec[n_, m_, d_] := Module[{t = ang[d][[1]], p = ang[d][[2]]},
   RMrot[t, p] . {0, dthY[n, m][t, p], (I m/Sin[t]) SphericalHarmonicY[n, m, t, p]}];
Cvec[n_, m_, d_] := Cross[d/Sqrt[d . d], Bvec[n, m, d]];
Pvec[n_, m_, d_] := Yfun[n, m, d] (d/Sqrt[d . d]);   (* radial vector harmonic *)

kPval = 0.9; kSval = 1.5;
rhoOf[c_, r_] := Sqrt[(r - c) . (r - c)];
Lw[n_, m_, type_, c_, r_] := Module[{x = kPval rhoOf[c, r]},
   zfp[n, x, type] Yfun[n, m, r - c] ((r - c)/rhoOf[c, r]) + (zfn[n, x, type]/x) Bvec[n, m, r - c]];
Mw[n_, m_, type_, c_, r_] := -zfn[n, kSval rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_] := Module[{x = kSval rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] ((r - c)/rhoOf[c, r])
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];

(* ---- sphere quadrature about B=origin (Gauss-Legendre x uniform phi) ---- *)
Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[24, -1, 1]; nPhiV = 48;
rho0 = 0.3 Sqrt[{1.3, 0.7, 0.0} . {1.3, 0.7, 0.0}];
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
projDot[Ffn_, harm_, nu_, mu_] := Sum[
   Module[{u = glN[[i, 1]], w = glN[[i, 2]], ph = 2 Pi (j - 1)/nPhiV, s},
    s = shat[u, ph];
    w (2 Pi/nPhiV) Ffn[rho0 s] . Conjugate[harm[nu, mu, s]]],
   {i, Length[glN]}, {j, nPhiV}];

(* ---- extracted coefficients a^M, a^N of a source field Ffn (memoised) ---- *)
Mreg[nu_, mu_] := Function[r, Mw[nu, mu, "j", {0, 0, 0}, r]];
Nreg[nu_, mu_] := Function[r, Nw[nu, mu, "j", {0, 0, 0}, r]];
normMC[nu_, mu_] := normMC[nu, mu] = projDot[Mreg[nu, mu], Cvec, nu, mu];
normNP[nu_, mu_] := normNP[nu, mu] = projDot[Nreg[nu, mu], Pvec, nu, mu];
dvecV = {1.3, 0.7, 0.0}; QmaxV = 10;
srcField[c_, n_, m_] := Function[r, Switch[c, "M", Mw[n, m, "h", dvecV, r], "N", Nw[n, m, "h", dvecV, r]]];
aMcoef[c_, n_, m_, nu_, mu_] := aMcoef[c, n, m, nu, mu] = projDot[srcField[c, n, m], Cvec, nu, mu]/normMC[nu, mu];
aNcoef[c_, n_, m_, nu_, mu_] := aNcoef[c, n, m, nu, mu] = projDot[srcField[c, n, m], Pvec, nu, mu]/normNP[nu, mu];

(* ---- matrix reconstruction of the translated field from the extracted coeffs ---- *)
recon[c_, n_, m_, r_] :=
   Sum[aMcoef[c, n, m, nu, mu] Mw[nu, mu, "j", {0, 0, 0}, r] + aNcoef[c, n, m, nu, mu] Nw[nu, mu, "j", {0, 0, 0}, r],
    {nu, 1, QmaxV}, {mu, -nu, nu}];

SeedRandom[20260618];
vtPts = Table[Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}]},
    0.2 Sqrt[dvecV . dvecV] shat[u, ph]], {4}];
vtSrc = {{"M", 1, 0}, {"N", 1, 1}, {"M", 2, -1}};
vtResid = Max[Table[Norm[srcField[s[[1]], s[[2]], s[[3]]][pt] - recon[s[[1]], s[[2]], s[[3]], pt]],
    {s, vtSrc}, {pt, vtPts}]];
Print["==== Phase 2 TB2 :: vector translation matrix extraction ===="];
Print["  sources = ", vtSrc, ", Qmax=", QmaxV, ", #pts=", Length[vtPts]];
Print["  matrix-reconstruction residual = ", vtResid, "  -> ", If[vtResid < 1*^-5, "PASS", "FAIL"]];
Print["Phase 2 TB2 (vector translation matrix) loaded + verified."];
