#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneRT.wl  --  Phase 3a: planar layer R/T(p) PROJECTION.

   Projects the Phase-2 spherical collective T_coll(k_par) onto Kennett
   flux-normalised up/down P-SV-SH plane waves at horizontal slowness p,
   producing Rd, Ru, Td, Tu (2x2 P-SV) + SH scalar, across normal /
   sub-critical / post-critical p.  Reciprocity (Tu = Td^T, Rd symmetric) in
   the FLUX-NORMALISED basis is the tight machine-precision gate (the raw
   displacement convention is reciprocal only under the symplectic D-metric,
   cf. item (d)).  The absolute normalisation is pinned against the Cartesian
   slab / Kennett in Python.  Energy balance |R|^2+|T|^2=1 is OUT of scope
   (needs the undamped vector G0) -> Phase 3b.

   COORDINATES: PROJECT frame (z,x,y) = (component 1, 2, 3): vertical/depth z =
   component 1, horizontal slowness p along x = component 2, SH polarisation =
   y-axis = component 3 = {0,0,1}.  Slowness/unit direction for mode m:
   k_hat = c_m (+-eta_m, p, 0) (down +eta, up -eta), eta_m = sqrt(1/c_m^2 - p^2),
   Im eta >= 0.  The verified CartesianT0 / Phase-2 spherical bridge is hardwired
   to polar axis = component 3 (its ang[d] = ArcCos[d[[3]]/|d|]); a single
   permutation toSph[{z,x,y}] = {x,y,z} maps the project frame into the bridge's
   internal frame at every bridge call (so vertical z -> polar comp 3).  The
   bridge / Phase-2 lattice code is UNTOUCHED.  Time e^{+i w t}, outgoing h_n^(1)
   (SphericalHankelH1).  Inner T0 at REAL background wavenumbers; lattice damped.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

reim[z_] := {Re[N[z]], Im[N[z]]};
alpha0 = 5000.; beta0 = 3000.; rho0 = 2500.;
lam0 = rho0 (alpha0^2 - 2 beta0^2); mu0 = rho0 beta0^2;
dampIm = 0.25; aa = 1.0; aLpitch = 2.5; Acell = aLpitch^2;
kaTest = 0.3; Nm = 3; LradV = 4; radP = 0.5;   (* Lr=4: damped lattice (Im k=0.25) converged; ±R symmetric keeps reciprocity *)

setMie[ka_, Dl_, Dm_, Dr_] := Module[{rhoI = rho0 + Dr, lamI0 = lam0 + Dl, muI0 = mu0 + Dm, alphaI, betaI},
  alphaI = Sqrt[(lamI0 + 2 muI0)/rhoI]; betaI = Sqrt[muI0/rhoI];
  kPo = ka; kSo = ka alpha0/beta0; kPi = ka alpha0/alphaI; kSi = ka alpha0/betaI;
  lamO = lam0; muO = mu0; lamI = lamI0; muI = muI0;];
omegaOf := kPo alpha0/aa;

(* ---- single-site T0 in the clean L/M/N basis ---- *)
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
idxVof[Nmax_] := Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];

(* ---- scalar lattice machinery (item c) ---- *)
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalHankelH1[n, x];
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];
g0LLfun[aL_, Lr_] := Module[{ij, Rv, bl, rN, ph, Dstruct, g0LL},
  ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
  Rv = Map[{aL #[[1]], aL #[[2]], 0.0} &, ij];
  bl = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ij];
  rN = Map[Sqrt[#[[1]]^2 + #[[2]]^2] &, Rv];
  ph = Map[ArcTan[#[[1]], #[[2]]] &, Rv];
  Dstruct[q_, s_] := Dstruct[q, s] = Total[sh[q, kappaP rN] SphericalHarmonicY[q, s, Pi/2, ph] bl];
  g0LL[n_, m_, nu_, mu_] := 4 Pi (-1)^m Sum[
     I^(nu + q - n) (-1)^q Dstruct[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
  g0LL];

(* ---- vector lattice machinery (item c, full L/M/N) ---- *)
zfn[n_, x_, "j"] := sj[n, x]; zfn[n_, x_, "h"] := sh[n, x];
zfp[n_, x_, "j"] := n/x sj[n, x] - sj[n + 1, x];
zfp[n_, x_, "h"] := n/x sh[n, x] - sh[n + 1, x];
angV[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, angV[d][[1]], angV[d][[2]]];
RMrot[t_, p_] := {{Sin[t] Cos[p], Cos[t] Cos[p], -Sin[p]},
   {Sin[t] Sin[p], Cos[t] Sin[p], Cos[p]}, {Cos[t], -Sin[t], 0}};
dthY[n_, m_] := dthY[n, m] = Module[{tt, pp, ex},
   ex = D[SphericalHarmonicY[n, m, tt, pp], tt]; Function[{a, b}, Evaluate[ex /. {tt -> a, pp -> b}]]];
Bvec[n_, m_, d_] := Module[{t = angV[d][[1]], p = angV[d][[2]]},
   RMrot[t, p] . {0, dthY[n, m][t, p], (I m/Sin[t]) SphericalHarmonicY[n, m, t, p]}];
Cvec[n_, m_, d_] := Cross[d/Sqrt[d . d], Bvec[n, m, d]];
Pvec[n_, m_, d_] := Yfun[n, m, d] (d/Sqrt[d . d]);
rhoOf[c_, r_] := Sqrt[(r - c) . (r - c)];
rhatOf[c_, r_] := (r - c)/Sqrt[(r - c) . (r - c)];
Mw[n_, m_, type_, c_, r_] := -zfn[n, kappaS rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_] := Module[{x = kappaS rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] rhatOf[c, r]
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];
Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[16, -1, 1]; nPhiV = 32;
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
quadList = Flatten[Table[{glN[[i, 1]], glN[[i, 2]], 2 Pi (j - 1)/nPhiV}, {i, Length[glN]}, {j, nPhiV}], 1];
quadDirs = Map[shat[#[[1]], #[[3]]] &, quadList];
quadW = Map[#[[2]] (2 Pi/nPhiV) &, quadList];
projDotF[fieldVals_, harmVals_] := Sum[quadW[[k]] fieldVals[[k]] . Conjugate[harmVals[[k]]], {k, Length[quadList]}];
Cvals[nu_, mu_] := Cvals[nu, mu] = Map[Cvec[nu, mu, #] &, quadDirs];
Pvals[nu_, mu_] := Pvals[nu, mu] = Map[Pvec[nu, mu, #] &, quadDirs];
normMC[nu_, mu_] := normMC[nu, mu] = projDotF[Map[Mw[nu, mu, "j", {0, 0, 0}, radP #] &, quadDirs], Cvals[nu, mu]];
normNP[nu_, mu_] := normNP[nu, mu] = projDotF[Map[Nw[nu, mu, "j", {0, 0, 0}, radP #] &, quadDirs], Pvals[nu, mu]];
buildG0vec[aL_, Lr_, Nmax_] := Module[
   {ijl, Rvecs, bloch, nR, outQuad, latSrc, g0LL, idx, nD, g0MM, g0NM, g0MN, g0NN, Gentry},
  ijl = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
  Rvecs = Map[{aL #[[1]], aL #[[2]], 0.0} &, ijl];
  bloch = Map[Exp[I aL (kx #[[1]] + ky #[[2]])] &, ijl]; nR = Length[Rvecs];
  outQuad[c_, n_, m_, R_] := Map[Switch[c, "M", Mw[n, m, "h", R, radP #], "N", Nw[n, m, "h", R, radP #]] &, quadDirs];
  latSrc[c_, n_, m_] := latSrc[c, n, m] = Total[Table[bloch[[iR]] outQuad[c, n, m, Rvecs[[iR]]], {iR, nR}]];
  g0LL = g0LLfun[aL, Lr];
  g0MM[n_, m_, nu_, mu_] := projDotF[latSrc["M", n, m], Cvals[nu, mu]]/normMC[nu, mu];
  g0NM[n_, m_, nu_, mu_] := projDotF[latSrc["M", n, m], Pvals[nu, mu]]/normNP[nu, mu];
  g0MN[n_, m_, nu_, mu_] := projDotF[latSrc["N", n, m], Cvals[nu, mu]]/normMC[nu, mu];
  g0NN[n_, m_, nu_, mu_] := projDotF[latSrc["N", n, m], Pvals[nu, mu]]/normNP[nu, mu];
  idx = idxVof[Nmax]; nD = Length[idx];
  Gentry[{nu_, mu_, ct_}, {n_, m_, cs_}] := Which[
    ct == "L" && cs == "L", g0LL[n, m, nu, mu],
    ct == "M" && cs == "M", g0MM[n, m, nu, mu], ct == "N" && cs == "M", g0NM[n, m, nu, mu],
    ct == "M" && cs == "N", g0MN[n, m, nu, mu], ct == "N" && cs == "N", g0NN[n, m, nu, mu], True, 0];
  Table[Gentry[idx[[i]], idx[[j]]], {i, nD}, {j, nD}]];
T0vec[Nmax_] := Module[{idx = idxVof[Nmax], T0e},
  T0e[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
    If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
  Table[T0e[idx[[i]], idx[[j]]], {i, Length[idx]}, {j, Length[idx]}]];
collV[G0_, T0_] := T0 . Inverse[IdentityMatrix[Length[T0]] - G0 . T0];

(* ---- PROJECT-frame slowness geometry (z,x,y); permutation into the bridge frame ---- *)
toSph[v_] := {v[[2]], v[[3]], v[[1]]};          (* project (z,x,y) -> bridge (x,y,z): vertical z -> polar comp3 *)
etaOf[c_, p_] := Module[{e = Sqrt[1./c^2 - p^2]}, If[Im[e] < 0, -e, e]];
khatPhys[m_, p_, sign_] := Module[{c = If[m == "P", alpha0, beta0]}, c {sign etaOf[c, p], p, 0.}]; (* (z,x,y) *)
(* ============================================================================
   thesis (PhD Section 3.1) plane-wave ENERGY normalisation + R/T projection
   ============================================================================ *)
(* ---- thesis (PhD Section 3.1, Eqs. Peigen/SVeigen/SHeigen + epsdef) plane-wave ENERGY
   normalisation.  k_{z,c} = omega eta_c; KhatS = KhatH = omega/beta0 (k_y = 0).  The eps-normalised
   eigenvector displacement is u^E = eps_c * (thesis displacement vector); its unit direction ehatTh
   (project frame) and complex scale muTh = ehatTh . u^E set the incident and projection.  Down
   sign=+1, up sign=-1.  Projecting onto / building from these IS the symplectic energy normalisation
   (the eps make (J6 Dz(-k))^T Dz(k) = i J6); no separate flux D. ---- *)
KhatSf := omegaOf/beta0;
epsP[p_] := 1/Sqrt[2 rho0 omegaOf^2 (omegaOf etaOf[alpha0, p])];
epsS[p_] := omegaOf/(beta0 KhatSf Sqrt[2 rho0 omegaOf^2 (omegaOf etaOf[beta0, p])]);
epsH[p_] := 1/(KhatSf Sqrt[2 rho0 omegaOf^2 (omegaOf etaOf[beta0, p])]);
ehatTh[m_, p_, sgn_] := Switch[m,           (* thesis eigenvector displacement direction (project z,x,y) *)
   "P", alpha0 {sgn etaOf[alpha0, p], p, 0.},       (* longitudinal *)
   "SV", beta0 {p, -sgn etaOf[beta0, p], 0.},       (* SV transverse (thesis sign) *)
   "SH", {0., 0., 1.}];                             (* SH = project y *)
muTh[m_, p_, sgn_] := Switch[m,             (* muTh = ehatTh . (eps_c * eigvec disp) *)
   "P", epsP[p] I omegaOf/alpha0,
   "SV", epsS[p] I omegaOf/beta0,
   "SH", epsH[p] KhatSf^2];

(* ---- incident multipole amplitudes, ANALYTIC-CONTINUATION form.  CartesianT0's incP/incN/incM
   use Conjugate[Yv[n,m,k]] / Conjugate[e.Bv]; for a COMPLEX (evanescent) direction k that conjugates
   the direction, which is the WRONG analytic continuation.  Using the real-direction identity
   Conjugate[Yv[n,m,k]] = (-1)^m Yv[n,-m,k] (and the same for Bv/Cv) gives coefficients that EQUAL
   CartesianT0's for real k and correctly continue for complex k.  ehat assumed real (true while the
   incident mode propagates; the only evanescent incidence in the sweep is P, which carries no ehat). *)
incPac[n_, m_, k_] := 4 Pi I^(n - 1) (-1)^m Yv[n, -m, k];
incNac[n_, m_, k_, e_] := -4 Pi I^(n + 1)/(n (n + 1)) (-1)^m (e . Bv[n, -m, k]);
incMac[n_, m_, k_, e_] := -4 Pi I^n/(n (n + 1)) (-1)^m (e . Cv[n, -m, k]);
incVec[mode_, khatSph_, ehatSph_, Nmax_] := Map[
   Function[idx, Module[{n = idx[[1]], m = idx[[2]], ch = idx[[3]]},
     Switch[{mode, ch}, {"P", "L"}, incPac[n, m, khatSph], {"SV", "N"}, incNac[n, m, khatSph, ehatSph],
       {"SH", "M"}, incMac[n, m, khatSph, ehatSph], _, 0]]], idxVof[Nmax]];

(* ---- plane-wave content of an outgoing multipole vector at bridge-frame direction ks ---- *)
projPW[bvec_, ksSph_, Nmax_] := Module[{fP = {0, 0, 0}, fS = {0, 0, 0}, idx = idxVof[Nmax]},
  Do[With[{n = idx[[i, 1]], m = idx[[i, 2]], ch = idx[[i, 3]], b = bvec[[i]]},
    Switch[ch,
      "L", fP += b ((-I)^n/kPo) Yv[n, m, ksSph] ksSph,
      "N", fS += b ((-I)^n/kSo) Bv[n, m, ksSph],
      "M", fS += -b ((-I)^(n + 1)/kSo) Cv[n, m, ksSph]]], {i, Length[idx]}];
  {fP, fS}];

(* ---- one specular R/T amplitude in the THESIS ENERGY normalisation.  Incident = the eps-normalised
   down/up eigenvector |sign,inMode> (direction ehatTh, scale muTh) expanded into multipoles; project
   the specular scattered displacement onto the eps-normalised |outSign,outMode> eigenvector.  The
   lattice Weyl prefactor i/(2 eta_out omega Acell) is the physical array-to-plane-wave conversion.
   R^E = Weyl (muTh_in/muTh_out) (ehatTh_out . scattered plane-wave displacement). ---- *)
rtAmpE[Tcoll_, inMode_, p_, inSign_, outMode_, outSign_, Nmax_] := Module[
   {kin = toSph[khatPhys[inMode, p, inSign]], kout = toSph[khatPhys[outMode, p, outSign]],
    ein = toSph[ehatTh[inMode, p, inSign]], eout = toSph[ehatTh[outMode, p, outSign]],
    avec, bvec, f, etaOut = etaOf[If[outMode == "P", alpha0, beta0], p]},
  avec = incVec[inMode, kin, ein, Nmax];
  bvec = Tcoll . avec;
  f = projPW[bvec, kout, Nmax];
  (I/(2 etaOut omegaOf Acell)) (muTh[inMode, p, inSign]/muTh[outMode, p, outSign]) *
    (eout . If[outMode == "P", First[f], Last[f]])];

(* ---- assemble all R/T blocks in the THESIS ENERGY normalisation (rtAmpE) ---- *)
rtBlocksE[Tcoll_, p_, Nmax_] := Module[{modes = {"P", "SV"}, blk},
  blk[inSign_, outSign_] := Table[rtAmpE[Tcoll, im, p, inSign, om, outSign, Nmax], {om, modes}, {im, modes}];
  <|"Rd" -> blk[1., -1.], "Td" -> blk[1., 1.], "Ru" -> blk[-1., 1.], "Tu" -> blk[-1., -1.],
    "Rsh" -> rtAmpE[Tcoll, "SH", p, 1., "SH", -1., Nmax],
    "Tsh" -> rtAmpE[Tcoll, "SH", p, 1., "SH", 1., Nmax]|>];

(* ============================================================================
   sweep over (normal / sub-critical / post-critical) p, build collective T_coll(p)
   ============================================================================ *)
Print["==== Phase 3a :: layer R/T(p) projection (thesis energy normalisation) ===="];
setMie[kaTest, 2.*^9, 1.*^9, 100.];             (* moderate contrast *)
pcritP = 1./alpha0; pcritS = 1./beta0;
(* "normal" uses p=1e-6 (not exactly 0): at p=0 the propagation is along the bridge polar
   axis where the azimuth is undefined (ArcTan[0,0]).  p=1e-6 << pcritP=2e-4 is normal to ~12 digits. *)
pNormal = 1.*^-6;
pList = {pNormal, 0.5 pcritP, 0.8 pcritS};       (* normal; sub-critical; post-(P)critical (P evanescent) *)
regimeOf[p_] := Which[p <= 1.*^-5, "normal", p < pcritP, "subcritical", True, "postcritical"];

(* thesis symplectic reciprocity (PhD Section 3.1): in the eps-energy normalisation the reflection is
   ANTISYMMETRIC (Rd = -Rd^T, Ru = -Ru^T) and the transmission obeys the parity relation Tu = Sig.Td.Sig
   with Sig = diag(1,-1) (SV carries the symplectic parity sign). *)
Sig = DiagonalMatrix[{1., -1.}];
stageRT = {};
Do[Module[{G0, Tc, e, antiRd, antiRu, tparity},
   (* Bloch vector: project k_par = omega(0,p,0) horizontal (p along x); the bridge lattice
      Rvecs={aL i,aL j,0} is the toSph image of the project x-y plane, so kx=k_par,x=omega p, ky=0 *)
   kx = omegaOf p; ky = 0.;
   kappaP = kPo + dampIm I; kappaS = kSo + dampIm I;
   G0 = buildG0vec[aLpitch, LradV, Nm]; Tc = collV[G0, T0vec[Nm]];
   e = rtBlocksE[Tc, p, Nm];                       (* thesis energy normalisation *)
   antiRd = Max[Abs[e["Rd"][[1, 2]] + e["Rd"][[2, 1]]]];
   antiRu = Max[Abs[e["Ru"][[1, 2]] + e["Ru"][[2, 1]]]];
   tparity = Max[Abs[Flatten[e["Tu"] - Sig . e["Td"] . Sig]]];
   AppendTo[stageRT, <|"p" -> p, "regime" -> regimeOf[p],
     "etaP" -> reim[etaOf[alpha0, p]], "etaS" -> reim[etaOf[beta0, p]],
     "Rd" -> Map[reim, e["Rd"], {2}], "Td" -> Map[reim, e["Td"], {2}],
     "Ru" -> Map[reim, e["Ru"], {2}], "Tu" -> Map[reim, e["Tu"], {2}],
     "Rsh" -> reim[e["Rsh"]], "Tsh" -> reim[e["Tsh"]],
     "recip_Rd_anti" -> N[antiRd], "recip_Ru_anti" -> N[antiRu], "recip_T_parity" -> N[tparity]|>];
   Print["  p=", ScientificForm[p, 3], " (", regimeOf[p], "): |Rd+Rd^T|=",
     ScientificForm[antiRd, 3], ", |Ru+Ru^T|=", ScientificForm[antiRu, 3],
     ", |Tu-Sig.Td.Sig|=", ScientificForm[tparity, 3]]],
   {p, pList}];

recipOK = AllTrue[stageRT, #["recip_Rd_anti"] < 1.*^-6 && #["recip_Ru_anti"] < 1.*^-6 &&
     #["recip_T_parity"] < 1.*^-6 &];
Print["  thesis symplectic reciprocity (Rd,Ru antisymmetric; Tu=Sig.Td.Sig) -> ",
   If[recipOK, "PASS", "FAIL"]];

Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneRT_reference.json",
  <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho0" -> rho0, "aa" -> aa,
      "aLpitch" -> aLpitch, "kPo" -> kPo, "dampIm" -> dampIm, "Nmax" -> Nm,
      "contrast" -> <|"Dlambda" -> 2.*^9, "Dmu" -> 1.*^9, "Drho" -> 100.|>|>,
    "stageRT" -> stageRT|>];
Print["  wrote IntraPlaneRT_reference.json"];
Print["Phase 3a (IntraPlaneRT.wl) loaded."];
