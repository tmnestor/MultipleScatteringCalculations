(* ::Package:: *)
(* FiveChannelDriver.wl -- headless driver for FiveChannelExtension.wl *)
(*  *)
(* Replicates the LaxFoldy_VoxelSphere_vs_Mie.nb context (Mie machinery, *)
(* medium parameters, voxel grid, Green's tensors, 9x9 T-matrix, FFT *)
(* P-incident solve) and then runs the five-channel extension, so the *)
(* full 5-channel Mie vs Lax-Foldy comparison is reproducible without *)
(* the notebook front end: *)
(*  *)
(*   wolframscript -file FiveChannelDriver.wl *)
(*  *)
(* Units: km/s, g/cm^3, GPa, km (consistent: GPa = g/cm^3 (km/s)^2). *)

(* DirectoryName is "" when invoked as `wolframscript -file <name>` from *)
(* inside Mathematica/ — fall back to the current directory. *)
Module[{dir = DirectoryName[First[$ScriptCommandLine]]},
  If[dir =!= "", SetDirectory[dir]]];
Get["CubeAnalytic.wl"];

(* ============================================================ *)
(* 1. Mie machinery (notebook section 1)                         *)
(* ============================================================ *)

jn[n_, z_] := SphericalBesselJ[n, z];
yn[n_, z_] := SphericalBesselY[n, z];
hn[n_, z_] := SphericalHankelH1[n, z];
jnp[n_, z_] := (n/z) jn[n, z] - jn[n + 1, z];
hnp[n_, z_] := (n/z) hn[n, z] - hn[n + 1, z];

pWaveFields[n_, k_, r_, lam_, mu_, zFunc_, zpFunc_] :=
  Module[{z, zv, zp, nn1, ur, ut, srr, srt},
    z = k r; zv = zFunc[n, z]; zp = zpFunc[n, z]; nn1 = n (n + 1);
    ur = k zp;
    ut = zv/r;
    srr = -(lam + 2 mu) k^2 zv - 4 mu k (zp/r) + 2 mu nn1 (zv/r^2);
    srt = 2 mu ((k zp - zv/r)/r);
    {ur, ut, srr, srt}];

sWaveFields[n_, k_, r_, mu_, zFunc_, zpFunc_] :=
  Module[{z, zv, zp, nn1, ur, ut, srr, srt},
    z = k r; zv = zFunc[n, z]; zp = zpFunc[n, z]; nn1 = n (n + 1);
    ur = nn1 (zv/r);
    ut = (zv + z zp)/r;
    srr = 2 mu nn1 (k (zp/r) - zv/r^2);
    srt = mu (((2 nn1 - 2 - z^2) zv - 2 z zp)/r^2);
    {ur, ut, srr, srt}];

miePSVMatrix[n_, omega_, a_, alphaOut_, betaOut_, rhoOut_,
             lamOut_, muOut_, alphaIn_, betaIn_, rhoIn_,
             lamIn_, muIn_] :=
  Module[{fPs, fSs, fPi, fSi},
    fPs = pWaveFields[n, omega/alphaOut, a, lamOut, muOut, hn, hnp];
    fSs = sWaveFields[n, omega/betaOut, a, muOut, hn, hnp];
    fPi = pWaveFields[n, omega/alphaIn, a, lamIn, muIn, jn, jnp];
    fSi = sWaveFields[n, omega/betaIn, a, muIn, jn, jnp];
    Transpose[{fPs, fSs, -fPi, -fSi}]];

mieIncidentPSV[n_, omega_, a_, alpha_, beta_, rho_, lam_, mu_] :=
  Module[{kP, coeff, fPinc},
    kP = omega/alpha;
    coeff = (2 n + 1) I^n/(I kP);
    fPinc = pWaveFields[n, kP, a, lam, mu, jn, jnp];
    -coeff fPinc];

(* ============================================================ *)
(* 2. Medium parameters (notebook section 3)                     *)
(* ============================================================ *)

alphaBg = 5.; betaBg = 3.; rhoBg = 2.5;
muBg = rhoBg betaBg^2;
lamBg = rhoBg (alphaBg^2 - 2 betaBg^2);
Dlam = 3.; Dmu = 1.; Drho = 0.1;
lamIn = lamBg + Dlam; muIn = muBg + Dmu; rhoIn = rhoBg + Drho;
alphaIn = Sqrt[(lamIn + 2 muIn)/rhoIn];
betaIn = Sqrt[muIn/rhoIn];
aRadius = 0.5; freq = 0.5; omega = 2 Pi freq;
kP = omega/alphaBg; kS = omega/betaBg;
nMax = Max[5, Ceiling[kS aRadius + 4 (kS aRadius)^(1./3.) + 2]];

Print["Background: Vp=", alphaBg, " Vs=", betaBg, " rho=", rhoBg];
Print["Contrasts: Dlam=", Dlam, " Dmu=", Dmu, " Drho=", Drho];
Print["k_P a = ", N[kP aRadius], ",  k_S a = ", N[kS aRadius],
  ",  n_max = ", nMax];

(* ============================================================ *)
(* 3. Voxel grid (notebook section 5)                            *)
(* ============================================================ *)

nPerDiam = 9;
d = 2 aRadius/nPerDiam;
halfGrid = Table[(-nPerDiam/2 + 0.5 + i) d, {i, 0, nPerDiam - 1}];
cubeCentres = {};
Do[Module[{pos = {zz, xx, yy}},
   If[Norm[pos] < aRadius, AppendTo[cubeCentres, pos]]],
  {zz, halfGrid}, {xx, halfGrid}, {yy, halfGrid}];
cubeCentres = N[cubeCentres];
nCubes = Length[cubeCentres];
Print["Voxels inside sphere: ", nCubes];

(* ============================================================ *)
(* 4. Green's tensor + 9x9 propagator (notebook section 6)       *)
(* ============================================================ *)

cNF[alpha_, beta_, rho_] := (1 - beta^2/alpha^2)/(8 Pi rho beta^2);
vP[r_, om_, alpha_, rho_] := Exp[I om r/alpha]/(4 Pi rho alpha^2);
vS[r_, om_, beta_, rho_] := Exp[I om r/beta]/(4 Pi rho beta^2);
fRadial[r_, om_, alpha_, beta_, rho_] :=
  (vS[r, om, beta, rho] - cNF[alpha, beta, rho])/r;
gRadial[r_, om_, alpha_, beta_, rho_] :=
  (3 cNF[alpha, beta, rho] + vP[r, om, alpha, rho] - vS[r, om, beta, rho])/r;
fRadialDeriv[r_, om_, alpha_, beta_, rho_] :=
  ((I om r/beta - 1) vS[r, om, beta, rho] + cNF[alpha, beta, rho])/r^2;
gRadialDeriv[r_, om_, alpha_, beta_, rho_] :=
  ((I om r/alpha - 1) vP[r, om, alpha, rho] -
   (I om r/beta - 1) vS[r, om, beta, rho] -
   3 cNF[alpha, beta, rho])/r^2;
fRadialDeriv2[r_, om_, alpha_, beta_, rho_] :=
  (((I om r/beta)^2 - 2 I om r/beta + 2) vS[r, om, beta, rho] -
   2 cNF[alpha, beta, rho])/r^3;
gRadialDeriv2[r_, om_, alpha_, beta_, rho_] :=
  (((I om r/alpha)^2 - 2 I om r/alpha + 2) vP[r, om, alpha, rho] -
   ((I om r/beta)^2 - 2 I om r/beta + 2) vS[r, om, beta, rho] +
   6 cNF[alpha, beta, rho])/r^3;

greensTensor[xVec_, om_, alpha_, beta_, rho_] :=
  Module[{r, nhat, fv, gv},
    r = Norm[xVec]; nhat = xVec/r;
    fv = fRadial[r, om, alpha, beta, rho];
    gv = gRadial[r, om, alpha, beta, rho];
    fv IdentityMatrix[3] + gv Outer[Times, nhat, nhat]];

greensDerivTensor[xVec_, om_, alpha_, beta_, rho_] :=
  Module[{r, nhat, gv, fp, gp},
    r = Norm[xVec]; nhat = xVec/r;
    gv = gRadial[r, om, alpha, beta, rho];
    fp = fRadialDeriv[r, om, alpha, beta, rho];
    gp = gRadialDeriv[r, om, alpha, beta, rho];
    Table[(fp KroneckerDelta[i, j] + gp nhat[[i]] nhat[[j]]) nhat[[k]] +
      (gv/r) (KroneckerDelta[i, k] nhat[[j]] +
              KroneckerDelta[j, k] nhat[[i]] -
              2 nhat[[i]] nhat[[j]] nhat[[k]]),
      {i, 3}, {j, 3}, {k, 3}]];

greensSecondDerivTensor[xVec_, om_, alpha_, beta_, rho_] :=
  Module[{r, nhat, dd, gv, fp, gp, fpp, gpp, t1, t2, t3, t4, t7},
    r = Norm[xVec]; nhat = xVec/r; dd = IdentityMatrix[3];
    gv = gRadial[r, om, alpha, beta, rho];
    fp = fRadialDeriv[r, om, alpha, beta, rho];
    gp = gRadialDeriv[r, om, alpha, beta, rho];
    fpp = fRadialDeriv2[r, om, alpha, beta, rho];
    gpp = gRadialDeriv2[r, om, alpha, beta, rho];
    t1 = fp/r; t2 = fpp - fp/r;
    t3 = gp/r - 2 gv/r^2; t4 = gv/r^2;
    t7 = gpp - 5 gp/r + 8 gv/r^2;
    Table[
      t1 dd[[i, j]] dd[[k, l]] + t2 dd[[i, j]] nhat[[k]] nhat[[l]] +
      t3 nhat[[i]] nhat[[j]] dd[[k, l]] +
      t4 (dd[[i, k]] dd[[j, l]] + dd[[j, k]] dd[[i, l]]) +
      t3 (dd[[i, l]] nhat[[j]] nhat[[k]] + dd[[j, l]] nhat[[i]] nhat[[k]] +
          dd[[i, k]] nhat[[j]] nhat[[l]] + dd[[j, k]] nhat[[i]] nhat[[l]]) +
      t7 nhat[[i]] nhat[[j]] nhat[[k]] nhat[[l]],
      {i, 3}, {j, 3}, {k, 3}, {l, 3}]];

voigtPairs = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
voigtWeight = {1, 1, 1, 2, 2, 2};

propagator9x9[xVec_, om_, alpha_, beta_, rho_] :=
  Module[{P, G3, Gd, G2d, H3, m, n, pp, qq, I1, J1, fac},
    G3 = greensTensor[xVec, om, alpha, beta, rho];
    Gd = greensDerivTensor[xVec, om, alpha, beta, rho];
    G2d = greensSecondDerivTensor[xVec, om, alpha, beta, rho];
    H3 = Table[(Gd[[i, j, k]] + Gd[[i, k, j]])/2, {i, 3}, {j, 3}, {k, 3}];
    P = ConstantArray[0. + 0. I, {9, 9}];
    P[[1 ;; 3, 1 ;; 3]] = G3;
    Do[{pp, qq} = voigtPairs[[J1]];
      Do[P[[i, 3 + J1]] = H3[[i, pp, qq]], {i, 3}], {J1, 6}];
    Do[{m, n} = voigtPairs[[I1]];
      Do[P[[3 + I1, l]] =
        voigtWeight[[I1]] ((Gd[[m, l, n]] + Gd[[n, l, m]])/2), {l, 3}],
      {I1, 6}];
    Do[{m, n} = voigtPairs[[I1]]; {pp, qq} = voigtPairs[[J1]];
      fac = voigtWeight[[I1]];
      P[[3 + I1, 3 + J1]] =
        fac ((G2d[[m, pp, qq, n]] + G2d[[m, qq, pp, n]] +
              G2d[[n, pp, qq, m]] + G2d[[n, qq, pp, m]])/4),
      {I1, 6}, {J1, 6}];
    P];

(* ============================================================ *)
(* 5. 9x9 T-matrix from effective contrasts (notebook section 7) *)
(* ============================================================ *)

{Atot, Btot, Ctot} = cubeABC[omega, d/2, alphaBg, betaBg, rhoBg];
T1coeff = Dlam (Atot + 4 Btot + Ctot) + 2 Dmu Btot;
T2coeff = Dmu (Atot + Btot);
T3coeff = 2 Dmu Ctot;
gamma0Val = cubeGamma0[omega, d/2, alphaBg, betaBg, rhoBg];
ampU = 1/(1 - omega^2 Drho gamma0Val);
ampTheta = 1/(1 - 3 T1coeff - 2 T2coeff - T3coeff);
ampEoff = 1/(1 - 2 T2coeff);
ampEdiag = 1/(1 - 2 T2coeff - T3coeff);
DlamStar = Dlam ampTheta + (2./3.) Dmu (ampTheta - ampEdiag);
DmuStarDiag = Dmu ampEdiag;
DmuStarOff = Dmu ampEoff;
DrhoStar = Drho ampU;
Vcube = d^3;
DCstarVoigt = ConstantArray[0., {6, 6}];
Do[DCstarVoigt[[i, i]] = 2 DmuStarDiag, {i, 3}];
Do[DCstarVoigt[[i, i]] = 2 DmuStarOff, {i, 4, 6}];
Do[Do[DCstarVoigt[[i, j]] += DlamStar, {j, 3}], {i, 3}];
tMatrix9 = ConstantArray[0. + 0. I, {9, 9}];
tMatrix9[[1 ;; 3, 1 ;; 3]] = Vcube omega^2 DrhoStar IdentityMatrix[3];
tMatrix9[[4 ;; 9, 4 ;; 9]] = Vcube N[DCstarVoigt];
Print["9x9 T-matrix assembled."];

(* ============================================================ *)
(* 6. P-incident FFT solve + sources (notebook sections 8-9)     *)
(* ============================================================ *)

Get["FFTLaxFoldy.wl"];
excField = fftSolveSystem[cubeCentres, nPerDiam, d, tMatrix9,
  omega, alphaBg, betaBg, rhoBg, incidentState];
sourcesList = Table[
  tMatrix9 . excField[[9 (kk - 1) + 1 ;; 9 kk]], {kk, nCubes}];

nAngles = 37;
thetaGrid = Table[(i - 1) Pi/(nAngles - 1), {i, nAngles}];

(* ============================================================ *)
(* 7. Five-channel extension (Mie SV/SH + LF SV/SH + comparison) *)
(* ============================================================ *)

Get["FiveChannelExtension.wl"];
