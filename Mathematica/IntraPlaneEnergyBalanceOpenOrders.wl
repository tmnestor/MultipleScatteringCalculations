#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneEnergyBalanceOpenOrders.wl  --  energy balance of the periodic
   sphere array over EVERY open diffraction order: the independent second
   implementation behind scripts/gate_layer_kkr_energy_balance.py.

   IntraPlaneEnergyBalance.wl checks the specular S-matrix only, and refuses to
   run when a diffraction order opens.  Here the lattice is super-wavelength
   (k_S a_L = 12): 5 P and 9 SV + 9 SH orders are open on each side, and the
   S-matrix spans all 46 channels.

   SHARES NO CODE WITH THE PYTHON (cubic_scattering/layer_kkr.py):
     T0      CartesianT0.wl (symplectic-bracket Mie, clean L/M/N), not sphere_scattering;
     D[q,s]  Ewald real + reciprocal sums evaluated here, projected on their own sphere;
     G0      M,N blocks by numerically extracted pair-translation coefficients
             (IntraPlaneKambeVector.wl), not the angular-momentum route;
     in/out  CartesianT0's vector Jacobi-Anger and far-field bridges.

   THE S-MATRIX.  Reference plane z = 0 (sphere centres), unit displacement along
   e_P = k^, e_SV = (u cos phi, u sin phi, -s), e_SH = (-sin phi, cos phi, 0) at
   k^ = (s cos phi, s sin phi, u) -- the Python gate's channels, so the two
   S-matrices are comparable ENTRY BY ENTRY, not only in modulus.  A far field
   u ~ f e^{ikr}/r, summed over the lattice, is the plane wave
   (2 pi i/(A k_z)) f(k^_G) in every order G (Weyl).  Flux basis
   w = sqrt(c^2 k_z/omega), two-sided.

   CHECKS:
     [1] eta-independence of D[q,s] with open orders;
     [2] S^H S = I;
     [3] CONTROL: coupling OFF breaks it;
     [4] CONTROL: IntraPlaneEnergyBalance.wl's Weyl factor i/(2 k_z A) breaks it;
     [5] CONTROL: IntraPlaneEnergyBalance.wl's incident split (SV -> N only,
         SH -> M only) breaks it.
   Dumps IntraPlaneEnergyBalanceOpenOrders_reference.json for the Python gate.

   Units: sphere radius a = 1.  a_L = 5, k_P a = 1.44, alpha/beta/rho x 1.1
   inside: the Python gate's 600 m / 120 m / omega = 60 / eps = 0.1.
   Run:  wolframscript -file IntraPlaneEnergyBalanceOpenOrders.wl [Nmax]
   Time e^{-i w t}, outgoing h^(1); lattice in x-y, z down.
   ============================================================================ *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
Needs["NumericalDifferentialEquationAnalysis`"];

Off[General::munfl];   (* e^{-800} real-space Ewald tails underflow to zero, harmlessly *)
fmt[x_] := Module[{e = Floor[Log10[x]]}, ToString[NumberForm[x/10.^e, {3, 2}]] <> "e" <> ToString[e]];
Nm = If[Length[$ScriptCommandLine] > 1, ToExpression[$ScriptCommandLine[[2]]], 3];
reim[z_] := {Re[N[z]], Im[N[z]]};
alpha0 = 5000.; beta0 = 3000.; rho0Bg = 2500.;
lam0 = rho0Bg (alpha0^2 - 2 beta0^2); mu0 = rho0Bg beta0^2;
epsC = 0.1;
aa = 1.0; aL = 5.0; Acell = aL^2; Aarea = aL^2; recipB = 2 Pi/aL;
kaP = 1.44; etaU = 0.7;
sj[n_, x_] := SphericalBesselJ[n, x];
sh[n_, x_] := SphericalBesselJ[n, x] + I SphericalBesselY[n, x];
kx = 0.; ky = 0.;   (* the k_par = 0 Bloch sector holds every open order *)

(* ============================================================================
   scalar Ewald lattice field minus the R = 0 term (IntraPlaneEnergyBalance.wl)
   ============================================================================ *)
ewReal3[kappa_, r_, eta_, Rc_] := (1/(8 Pi)) Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      (Exp[I aL (kx i + ky j)]/d) Sum[Exp[s I kappa d] Erfc[d eta + s I kappa/(2 eta)], {s, {-1, 1}}]],
     {i, -Rc, Rc}, {j, -Rc, Rc}], 1]];
gKs[kappa_, rn_] := Exp[I kappa rn]/(4 Pi rn);
ewRecip3[kappa_, r_, eta_, Gc_] := (I/(4 Aarea)) Total[Flatten[Table[
     With[{kpg = {kx, ky} + recipB {m, n}, z = r[[3]]},
      With[{kz = Sqrt[kappa^2 - kpg . kpg]},
       (Exp[I kpg . {r[[1]], r[[2]]}]/kz) (
          Exp[-I kz Abs[z]] Erfc[ Abs[z] eta + kz/(2 I eta)]
        + Exp[ I kz Abs[z]] Erfc[-Abs[z] eta + kz/(2 I eta)])]],
     {m, -Gc, Gc}, {n, -Gc, Gc}], 1]];
ewTot3[kappa_, r_, eta_] :=
  ewReal3[kappa, r, eta, 4] + ewRecip3[kappa, r, eta, 8] - gKs[kappa, Sqrt[r . r]];

(* ============================================================================
   structure constants: the field evaluated ONCE per kappa, then projected
   ============================================================================ *)
glD = GaussianQuadratureWeights[16, -1, 1]; nPhiD = 32; rho0Proj = 2.0;
sphPts = Flatten[Table[Module[{u = glD[[i, 1]], ph = 2 Pi (j - 1)/nPhiD, st = Sqrt[1 - glD[[i, 1]]^2]},
     {rho0Proj {st Cos[ph], st Sin[ph], u}, glD[[i, 2]] (2 Pi/nPhiD)}], {i, Length[glD]}, {j, nPhiD}], 1];
Yf[q_, s_, d_] := SphericalHarmonicY[q, s, ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]];
buildDtab[kappa_, eta_, qmax_] := Module[{fv = Map[ewTot3[kappa, #[[1]], eta] &, sphPts]},
  Association[Flatten[Table[{q, s} -> ((-1)^s Total[MapThread[#2[[2]] #1 Conjugate[Yf[q, -s, #2[[1]]]] &,
        {fv, sphPts}]]/(I kappa sj[q, kappa rho0Proj])), {q, 0, qmax}, {s, -q, q}], 1]]];

(* ============================================================================
   vector pair translation + lattice contraction (IntraPlaneKambeVector.wl)
   ============================================================================ *)
gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];
zfn[n_, x_, "j"] := sj[n, x]; zfn[n_, x_, "h"] := sh[n, x];
zfp[n_, x_, "j"] := n/x sj[n, x] - sj[n + 1, x];
zfp[n_, x_, "h"] := n/x sh[n, x] - sh[n + 1, x];
Yfun[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
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
Mw[n_, m_, type_, c_, r_, kS_] := -zfn[n, kS rhoOf[c, r], type] Cvec[n, m, r - c];
Nw[n_, m_, type_, c_, r_, kS_] := Module[{x = kS rhoOf[c, r]},
   (n (n + 1)/x) zfn[n, x, type] Yfun[n, m, r - c] rhatOf[c, r]
     + ((zfn[n, x, type] + x zfp[n, x, type])/x) Bvec[n, m, r - c]];
glV = GaussianQuadratureWeights[12, -1, 1]; nPhiV = 24; radP = 0.5;
shat[u_, ph_] := {Sqrt[1 - u^2] Cos[ph], Sqrt[1 - u^2] Sin[ph], u};
quadList = Flatten[Table[{glV[[i, 1]], glV[[i, 2]], 2 Pi (j - 1)/nPhiV}, {i, Length[glV]}, {j, nPhiV}], 1];
quadDirs = Map[shat[#[[1]], #[[3]]] &, quadList];
quadW = Map[#[[2]] (2 Pi/nPhiV) &, quadList];
projDotF[fieldVals_, harmVals_] := Total[quadW Total[fieldVals Conjugate[harmVals], {2}]];
Cvals[nu_, mu_] := Cvals[nu, mu] = Map[Cvec[nu, mu, #] &, quadDirs];
Pvals[nu_, mu_] := Pvals[nu, mu] = Map[Pvec[nu, mu, #] &, quadDirs];
normMC[nu_, mu_, kS_] := normMC[nu, mu, kS] = projDotF[Map[Mw[nu, mu, "j", {0, 0, 0}, radP #, kS] &, quadDirs], Cvals[nu, mu]];
normNP[nu_, mu_, kS_] := normNP[nu, mu, kS] = projDotF[Map[Nw[nu, mu, "j", {0, 0, 0}, radP #, kS] &, quadDirs], Pvals[nu, mu]];
srcQuad[c_, n_, m_, d_, kS_] := srcQuad[c, n, m, d, kS] =
   Map[Switch[c, "M", Mw[n, m, "h", d, radP #, kS], "N", Nw[n, m, "h", d, radP #, kS]] &, quadDirs];
Wel[cP_, nu_, mu_, c_, n_, m_, d_, kS_] := Module[{f = srcQuad[c, n, m, d, kS]},
   Switch[cP, "M", projDotF[f, Cvals[nu, mu]]/normMC[nu, mu, kS],
              "N", projDotF[f, Pvals[nu, mu]]/normNP[nu, mu, kS]]];
glS = GaussianQuadratureWeights[8, -1, 1]; nPhiS = 16; dext = 2.0;
srcW = Flatten[Table[{shat[glS[[i, 1]], 2 Pi (j - 1)/nPhiS], glS[[i, 2]] (2 Pi/nPhiS)},
     {i, Length[glS]}, {j, nPhiS}], 1];
qset[n_, nu_] := Range[Abs[n - nu], n + nu];
coeffq[cP_, nu_, mu_, c_, n_, m_, kS_] := coeffq[cP, nu, mu, c, n, m, kS] = Association[Table[
    q -> Total[Map[#[[2]] Wel[cP, nu, mu, c, n, m, dext #[[1]], kS]
          Conjugate[SphericalHarmonicY[q, m - mu, ang[#[[1]]][[1]], ang[#[[1]]][[2]]]] &, srcW]]
       / sh[q, kS dext],
    {q, qset[n, nu]}]];
g0LLk[n_, m_, nu_, mu_, Dfun_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q Dfun[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
g0MNblock[cP_, nu_, mu_, c_, n_, m_, kS_, Dfun_] := Module[{cf = coeffq[cP, nu, mu, c, n, m, kS]},
   Total[KeyValueMap[#2 Dfun[#1, m - mu] &, cf]]];

(* ============================================================================
   single-site T0 (CartesianT0), the index, the coupling
   ============================================================================ *)
setMie[ka_, Dl_, Dm_, Dr_] := Module[{rhoI = rho0Bg + Dr, lamI0 = lam0 + Dl, muI0 = mu0 + Dm, alphaI, betaI},
  alphaI = Sqrt[(lamI0 + 2 muI0)/rhoI]; betaI = Sqrt[muI0/rhoI];
  kPo = ka; kSo = ka alpha0/beta0; kPi = ka alpha0/alphaI; kSi = ka alpha0/betaI;
  lamO = lam0; muO = mu0; lamI = lamI0; muI = muI0;];
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
idxVof[nmax_] := Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, nmax}], 1];
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
T0vec[nmax_] := Module[{idx = idxVof[nmax], T0e},
  T0e[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
    If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
  Table[T0e[idx[[i]], idx[[j]]], {i, Length[idx]}, {j, Length[idx]}]];
collV[G0_, T0_] := T0 . Inverse[IdentityMatrix[Length[T0]] - G0 . T0];
buildG0[nmax_] := Module[{idx = idxVof[nmax], nD, DtabP, DtabS, DuP, DuS, Gentry},
  DtabP = buildDtab[kPo, etaU, 2 nmax]; DtabS = buildDtab[kSo, etaU, 2 nmax];
  DuP = Function[{q, s}, DtabP[{q, s}]]; DuS = Function[{q, s}, DtabS[{q, s}]];
  nD = Length[idx];
  Gentry[{nu_, mu_, ct_}, {n_, m_, cs_}] := Which[
    ct == "L" && cs == "L", g0LLk[n, m, nu, mu, DuP],
    (ct == "M" || ct == "N") && (cs == "M" || cs == "N"), g0MNblock[ct, nu, mu, cs, n, m, kSo, DuS],
    True, 0];
  Table[Gentry[idx[[i]], idx[[j]]], {i, nD}, {j, nD}]];

(* ============================================================================
   The open channels, and CartesianT0's bridges made safe at the poles.
   Y and grad_Omega Y are smooth on the sphere but CartesianT0 evaluates them in
   (theta, phi), singular at k^ = +-z (normal incidence, G = 0).  The mean over
   k^ +- delta x^ is the pole value to O(delta^2).
   ============================================================================ *)
dPole = 1.*^-5;
nearPole[d_] := Sqrt[d[[1]]^2 + d[[2]]^2] < 1.*^-12;
poleAvg[f_, d_] := If[nearPole[d],
   (f[Normalize[d + {dPole, 0, 0}]] + f[Normalize[d - {dPole, 0, 0}]])/2, f[d]];
Ys[n_, m_, d_] := poleAvg[Yv[n, m, #] &, d];
Bs[n_, m_, d_] := poleAvg[Bv[n, m, #] &, d];
Cs[n_, m_, d_] := Cross[d, Bs[n, m, d]];

kOf[mode_] := If[mode == "P", kPo, kSo];
cOf[mode_] := If[mode == "P", alpha0, beta0];
openChannels[] := Module[{jmax = Floor[kSo/recipB] + 1, out = {}},
  Do[Do[Do[With[{q = recipB Sqrt[i^2 + j^2], k = kOf[mode]},
       If[q < k && Sqrt[k^2 - q^2]/k > 1.*^-3, AppendTo[out, {mode, i, j, up}]]],
      {i, -jmax, jmax}, {j, -jmax, jmax}], {mode, {"P", "SV", "SH"}}], {up, {False, True}}];
  out];
geom[{mode_, i_, j_, up_}] := Module[{k = kOf[mode], gx = recipB i, gy = recipB j, q, cph, sph, s, u},
  q = Sqrt[gx^2 + gy^2]; {cph, sph} = If[q > 0, {gx/q, gy/q}, {1., 0.}];
  s = q/k; u = Sqrt[1 - s^2] If[up, -1, 1];
  <|"khat" -> {s cph, s sph, u}, "kz" -> k Abs[u],
    "e" -> Switch[mode, "P", {s cph, s sph, u}, "SV", {u cph, u sph, -s}, "SH", {-sph, cph, 0.}]|>];

(* incident regular coefficients; split = True is IntraPlaneEnergyBalance.wl's SV -> N, SH -> M *)
incVecS[mode_, g_, nmax_, split_] := Map[Function[idx, Module[{n = idx[[1]], m = idx[[2]], ch = idx[[3]],
      k = g["khat"], e = g["e"]},
     Which[
      mode == "P", If[ch == "L", 4 Pi I^(n - 1) Conjugate[Ys[n, m, k]], 0],
      ch == "N" && ! (split && mode == "SH"), -4 Pi I^(n + 1)/(n (n + 1)) Conjugate[e . Bs[n, m, k]],
      ch == "M" && ! (split && mode == "SV"), -4 Pi I^n/(n (n + 1)) Conjugate[e . Cs[n, m, k]],
      True, 0]]], idxVof[nmax]];
(* far field f(k^) of outgoing coefficients b: u ~ f e^{ikr}/r (CartesianT0 farField) *)
farF[bvec_, k_, nmax_] := Module[{fP = {0, 0, 0}, fS = {0, 0, 0}, idx = idxVof[nmax]},
  Do[With[{n = idx[[i, 1]], m = idx[[i, 2]], ch = idx[[i, 3]], b = bvec[[i]]},
    Switch[ch,
      "L", fP += b ((-I)^n/kPo) Ys[n, m, k] k,
      "N", fS += b ((-I)^n/kSo) Bs[n, m, k],
      "M", fS += -b ((-I)^(n + 1)/kSo) Cs[n, m, k]]], {i, Length[idx]}];
  {fP, fS}];

sMatrix[Tc_, nmax_, weyl_, split_] := Module[{ch = openChannels[], gs, w, s, nc},
  gs = geom /@ ch; nc = Length[ch];
  w = Table[Sqrt[cOf[ch[[i, 1]]]^2 gs[[i]]["kz"]/(kPo alpha0)], {i, nc}];
  s = Table[0. I, {nc}, {nc}];
  Do[Module[{b = Tc . incVecS[ch[[jj, 1]], gs[[jj]], nmax, split]},
     Do[With[{f = farF[b, gs[[ii]]["khat"], nmax]},
        s[[ii, jj]] = weyl[gs[[ii]]["kz"]] (gs[[ii]]["e"] . If[ch[[ii, 1]] == "P", f[[1]], f[[2]]])],
      {ii, nc}]], {jj, nc}];
  s += IdentityMatrix[nc];
  (w s)/ConstantArray[w, nc]];   (* row i times w_i, column j over w_j *)
resid[s_] := Norm[ConjugateTranspose[s] . s - IdentityMatrix[Length[s]], "Frobenius"]/Sqrt[Length[s]];
weylTrue[kz_] := 2 Pi I/(Acell kz);
weylOld[kz_] := I/(2 kz Acell);

(* ============================================================================
   The checks
   ============================================================================ *)
Print["==== open-order energy balance of the sphere array (N_max = ", Nm, ") ===="];
setMie[kaP, (1.1^3 - 1) lam0, (1.1^3 - 1) mu0, 0.1 rho0Bg];
chans = openChannels[];
Print["  k_P a = ", kPo, ", k_S a = ", kSo, ", a_L = ", aL, "; open channels: ", Length[chans],
  " (P ", Count[chans, {"P", __}], ", SV ", Count[chans, {"SV", __}], ", SH ", Count[chans, {"SH", __}], ")"];

etaInd = Max[Table[Module[{a = buildDtab[kap, 0.7, 4], b = buildDtab[kap, 1.0, 4]},
     Max[Abs[Values[a] - Values[b]]]/Max[Abs[Values[a]]]], {kap, {kPo, kSo}}]];
Print["  [1] D[q,s] eta-independence (0.7 vs 1.0, q <= 4), relative: ", fmt[etaInd],
  " -> ", If[etaInd < 1.*^-8, "PASS", "FAIL"]];

t0 = AbsoluteTime[];
G0 = buildG0[Nm]; T0m = T0vec[Nm];
Print["  G0 built in ", Round[AbsoluteTime[] - t0], " s"];
Tcoll = collV[G0, T0m];
Tiso = T0m;

sOn = sMatrix[Tcoll, Nm, weylTrue, False];
rOn = resid[sOn];
Print["  [2] S^H S = I over every open channel: ", fmt[rOn],
  " -> ", If[rOn < 1.*^-5, "PASS", "FAIL"]];
rOff = resid[sMatrix[Tiso, Nm, weylTrue, False]];
Print["  [3] CONTROL coupling OFF: ", fmt[rOff], " -> ", If[rOff > 100 rOn, "PASS (fails)", "FAIL"]];
rWeyl = resid[sMatrix[Tcoll, Nm, weylOld, False]];
Print["  [4] CONTROL Weyl factor i/(2 k_z A): ", fmt[rWeyl],
  " -> ", If[rWeyl > 100 rOn, "PASS (fails)", "FAIL"]];
rSplit = resid[sMatrix[Tcoll, Nm, weylTrue, True]];
Print["  [5] CONTROL incident SV -> N only, SH -> M only: ", fmt[rSplit],
  " -> ", If[rSplit > 100 rOn, "PASS (fails)", "FAIL"]];

Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneEnergyBalanceOpenOrders_reference.json",
  <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho" -> rho0Bg, "radius" -> aa, "aL" -> aL,
      "kPa" -> kPo, "kSa" -> kSo, "eps" -> epsC, "Nmax" -> Nm, "etaU" -> etaU, "rho0Proj" -> rho0Proj|>,
    "channels" -> Map[<|"mode" -> #[[1]], "i" -> #[[2]], "j" -> #[[3]], "upward" -> #[[4]]|> &, chans],
    "S" -> Map[reim, sOn, {2}],
    "resid_on" -> rOn, "resid_off" -> rOff, "resid_weyl_old" -> rWeyl, "resid_split_old" -> rSplit,
    "eta_indep" -> etaInd|>];
Print["  wrote IntraPlaneEnergyBalanceOpenOrders_reference.json"];
