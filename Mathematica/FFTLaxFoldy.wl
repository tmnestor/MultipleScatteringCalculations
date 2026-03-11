(* ::Package:: *)
(* FFTLaxFoldy.wl -- FFT-accelerated Lax-Foldy solver *)
(* Replaces O(N^3) dense assembly + LinearSolve with *)
(* O(N_iter * N log N) GMRES + FFT convolution. *)
(*  *)
(* Key insight: P(x_j - x_k) depends only on the separation *)
(* vector, and the cubes sit on a regular grid, so the *)
(* matrix-vector product is a 3D discrete convolution *)
(* computable via FFT. *)
(*  *)
(* Loaded via: Get[NotebookDirectory[] <> "FFTLaxFoldy.wl"] *)

(* ============================================================ *)
(* Incident P-wave state vector (default for P along +z) *)
(* Defines globally so downstream cells can use it *)
(* ============================================================ *)

incidentState[pos_] := Module[{kPv, phase},
  kPv = omega/alphaBg;
  phase = Exp[I kPv pos[[1]]];
  {phase, 0., 0.,           (* displacement *)
   I kPv phase, 0., 0.,    (* normal strains: only eps_zz *)
   0., 0., 0.}             (* shear strains *)
];

(* ============================================================ *)
(* GMRES solver (full, unrestarted) *)
(* *)
(* matvec : function computing A.x *)
(* b : right-hand side *)
(* x0 : initial guess *)
(* relTol : relative residual tolerance *)
(* maxIter: maximum GMRES iterations *)
(* Returns: solution vector x *)
(* ============================================================ *)

gmresSolve[matvec_, b_, x0_, relTol_, maxIter_] :=
  Module[{x = x0, r0, beta, bNorm, absTol, V, H = {},
    g, cR, sR, m = 0, w, hj, rr, temp, y, res},

  bNorm = Norm[b];
  absTol = relTol bNorm;
  r0 = b - matvec[x];
  beta = Norm[r0];
  If[beta < absTol, Print["  Already converged."]; Return[x]];

  V = {r0/beta};
  g = ConstantArray[0. + 0. I, maxIter + 2]; g[[1]] = beta;
  cR = ConstantArray[0. + 0. I, maxIter];
  sR = ConstantArray[0. + 0. I, maxIter];

  Print["  GMRES |r\:2080|/|b| = ", ScientificForm[beta/bNorm, 4]];

  Do[
    (* Arnoldi step *)
    w = matvec[V[[j]]];
    hj = ConstantArray[0. + 0. I, j + 1];
    Do[hj[[i]] = Conjugate[V[[i]]] . w;
       w -= hj[[i]] V[[i]], {i, j}];
    hj[[j + 1]] = Norm[w];

    (* Apply previous Givens rotations *)
    Do[
      temp = Conjugate[cR[[i]]] hj[[i]] +
             Conjugate[sR[[i]]] hj[[i + 1]];
      hj[[i + 1]] = -sR[[i]] hj[[i]] + cR[[i]] hj[[i + 1]];
      hj[[i]] = temp,
    {i, j - 1}];

    (* New Givens rotation to zero out h_{j+1,j} *)
    rr = Sqrt[Abs[hj[[j]]]^2 + Abs[hj[[j + 1]]]^2];
    cR[[j]] = hj[[j]]/rr;
    sR[[j]] = hj[[j + 1]]/rr;
    hj[[j]] = rr;

    (* Update residual norm via g vector *)
    g[[j + 1]] = -sR[[j]] g[[j]];
    g[[j]] = Conjugate[cR[[j]]] g[[j]];

    AppendTo[H, hj[[1 ;; j]]];

    (* Lucky breakdown: Krylov subspace is invariant *)
    If[Abs[hj[[j + 1]]] < $MachineEpsilon Norm[hj],
      Print["  GMRES: lucky breakdown at iter ", j];
      m = j; Break[]];
    AppendTo[V, w/hj[[j + 1]]];

    res = Abs[g[[j + 1]]]/bNorm;
    m = j;
    If[Mod[j, 5] == 0 || j == 1 || res < relTol,
      Print["  iter ", j, ": |r|/|b| = ",
            ScientificForm[res, 4]]];
    If[res < relTol, Break[]],
  {j, maxIter}];

  (* Back-substitution: solve upper-triangular system *)
  y = ConstantArray[0. + 0. I, m];
  Do[y[[i]] = (g[[i]] - Sum[H[[k, i]] y[[k]],
       {k, i + 1, m}])/H[[i, i]],
    {i, m, 1, -1}];

  Print["  Converged: ", m, " iterations, |r|/|b| = ",
    ScientificForm[Abs[g[[m + 1]]]/bNorm, 4]];
  x + Sum[y[[i]] V[[i]], {i, m}]
];

(* ============================================================ *)
(* FFT Lax-Foldy Solve *)
(* *)
(* centres : list of {z,x,y} cube centre positions *)
(* nPerD : cubes per sphere diameter *)
(* dd : cube side length *)
(* tMat : 9x9 T-matrix *)
(* om, al, be, rh : omega, alpha, beta, rho (background) *)
(* incState : function mapping position -> 9-component state *)
(* relTol : GMRES relative tolerance (default 1e-8) *)
(* *)
(* Returns: excField (flat 9*N complex vector) *)
(* ============================================================ *)

fftSolveSystem[centres_, nPerD_, dd_, tMat_,
               om_, al_, be_, rh_,
               incState_, relTol_: 1.*^-8] :=
  Module[{nC, nP, nPV, dm, halfG, aR, gridIdx,
    kernel, kernelFFT, pack, unpack, matvec, rhs, t0},

  t0 = AbsoluteTime[];
  nC = Length[centres];
  nP = 2 nPerD - 1;
  nPV = N[nP^3];
  dm = 9 nC;

  (* Reconstruct grid parameters *)
  aR = nPerD dd/2;
  halfG = Table[(-nPerD/2 + 0.5 + i) dd, {i, 0, nPerD - 1}];

  (* Grid index mapping: cube -> {iz, ix, iy} on nPerD^3 grid *)
  gridIdx = Reap[
    Do[If[Norm[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}] < aR,
      Sow[{iz, ix, iy}]],
    {iz, nPerD}, {ix, nPerD}, {iy, nPerD}]][[2, 1]];

  If[Length[gridIdx] != nC,
    Print["ERROR: grid index count ", Length[gridIdx],
      " != nCubes ", nC];
    Return[$Failed]];

  Print[Style["FFT-Accelerated Lax\[Dash]Foldy Solve", Bold, 14]];
  Print["  Active cubes: ", nC, " / ", nPerD^3,
    "   (", dm, " DOF)"];
  Print["  FFT grid: ", nP, " x ", nP, " x ", nP,
    "   (", nP^3, " points)"];
  Print["  Propagator evaluations: ", nP^3 - 1,
    "  (vs ", nC (nC - 1), " for dense)"];

  (* ---- Propagator x T kernel ---- *)
  (* Compute -P(dx).T at every distinct grid separation *)
  (* and embed in circular array for FFT convolution. *)
  kernel = ConstantArray[0. + 0. I, {9, 9, nP, nP, nP}];

  Do[If[!(dz == 0 && dx == 0 && dy == 0),
    Module[{xSep, block, iz, ix, iy},
      xSep = N[{dz, dx, dy} dd];
      block = -(propagator9x9[xSep, om, al, be, rh] . tMat);
      (* Circular embedding: negative offsets wrap to end *)
      iz = Mod[dz, nP] + 1;
      ix = Mod[dx, nP] + 1;
      iy = Mod[dy, nP] + 1;
      Do[kernel[[ii, jj, iz, ix, iy]] = block[[ii, jj]],
        {ii, 9}, {jj, 9}]]],
  {dz, -(nPerD - 1), nPerD - 1},
  {dx, -(nPerD - 1), nPerD - 1},
  {dy, -(nPerD - 1), nPerD - 1}];

  Print["  Propagator: ",
    Round[AbsoluteTime[] - t0, 0.1], " s"];

  (* FFT each of the 81 components, pre-scaled for the *)
  (* convolution theorem: y = IFFT[sqrt(M) FFT[h] FFT[x]] *)
  kernelFFT = Table[
    Sqrt[nPV] Fourier[kernel[[ii, jj]]],
    {ii, 9}, {jj, 9}];

  Print["  Kernel FFT: ",
    Round[AbsoluteTime[] - t0, 0.1], " s total setup"];

  (* ---- Local pack/unpack/matvec closures ---- *)
  (* These capture gridIdx, nP, dm, nC, kernelFFT from *)
  (* the enclosing Module scope. *)

  pack[wFlat_] := Module[{grids, idx, gi},
    grids = ConstantArray[0. + 0. I, {9, nP, nP, nP}];
    Do[idx = 9 (j - 1); gi = gridIdx[[j]];
      Do[grids[[c, gi[[1]], gi[[2]], gi[[3]]]] =
           wFlat[[idx + c]], {c, 9}],
    {j, nC}];
    grids];

  unpack[grids_] := Module[{wFlat, idx, gi},
    wFlat = ConstantArray[0. + 0. I, dm];
    Do[idx = 9 (j - 1); gi = gridIdx[[j]];
      Do[wFlat[[idx + c]] =
           grids[[c, gi[[1]], gi[[2]], gi[[3]]]],
        {c, 9}],
    {j, nC}];
    wFlat];

  (* Compute (I - P.T) w via FFT convolution *)
  (* Kernel stores -P.T, so w + conv = (I - P.T) w *)
  matvec[wFlat_] := Module[{wFFT, yFFT},
    wFFT = Map[Fourier, pack[wFlat]];
    yFFT = Table[
      Sum[kernelFFT[[i, j]] wFFT[[j]], {j, 9}],
      {i, 9}];
    wFlat + unpack[Map[InverseFourier, yFFT]]];

  (* ---- Build RHS: incident field at all cubes ---- *)
  rhs = ConstantArray[0. + 0. I, dm];
  Do[Module[{winc, idx}, idx = 9 (j - 1);
    winc = incState[centres[[j]]];
    Do[rhs[[idx + c]] = winc[[c]], {c, 9}]],
  {j, nC}];

  (* ---- Solve with GMRES ---- *)
  Print["\n  Solving ", dm, "-DOF system..."];
  Module[{exc, resNorm},
    exc = gmresSolve[matvec, rhs, rhs, relTol, 200];
    Print["  Wall time: ",
      Round[AbsoluteTime[] - t0, 0.1], " s"];

    (* Residual verification *)
    resNorm = Norm[matvec[exc] - rhs]/Norm[rhs];
    Print["  Residual: ||Ax-b||/||b|| = ",
      ScientificForm[resNorm, 4]];

    Print["\n  Exciting field at cube 1:"];
    Print["    u = ", exc[[1 ;; 3]]];
    Print["    eps = ", exc[[4 ;; 9]]];
    exc]
];

(* ============================================================ *)
(* Convergence wrapper: drop-in replacement for *)
(* laxFoldyForwardAmplitude[nPerD] *)
(* Returns {nCubes, forwardAmplitude} *)
(* *)
(* Requires globals: aRadius, omega, alphaBg, betaBg, rhoBg, *)
(* Dlam, Dmu, Drho, incidentState, voigtPairs, voigtWeight, *)
(* cubeABC, cubeGamma0, cubeFarFieldP, propagator9x9 *)
(* ============================================================ *)

laxFoldyForwardAmplitudeFFT[nPerD_, relTol_: 1.*^-8] :=
  Module[{dd, halfG, centres, nC, VC,
    Av, Bv, Cv, T1v, T2v, T3v, g0v,
    aU, aT, aEo, aEd, dlS, dmSD, dmSO, drS, DCv, tMat,
    exc, kPv, srcList, fwd},

  dd = 2 aRadius/nPerD;
  halfG = Table[(-nPerD/2 + 0.5 + i) dd,
    {i, 0, nPerD - 1}];

  centres = N[Reap[
    Do[If[Norm[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}] <
          aRadius,
      Sow[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}]],
    {iz, nPerD}, {ix, nPerD}, {iy, nPerD}]][[2, 1]]];
  nC = Length[centres];
  VC = dd^3;

  Print["  filling = ", nC VC / (4./3. Pi aRadius^3)];

  (* T-matrix with cubic Eshelby correction *)
  {Av, Bv, Cv} = cubeABC[omega, dd/2,
    alphaBg, betaBg, rhoBg];
  T1v = Dlam (Av + 4 Bv + Cv) + 2 Dmu Bv;
  T2v = Dmu (Av + Bv);
  T3v = 2 Dmu Cv;

  g0v = cubeGamma0[omega, dd/2,
    alphaBg, betaBg, rhoBg];
  aU = 1/(1 - omega^2 Drho g0v);
  aT = 1/(1 - 3 T1v - 2 T2v - T3v);
  aEo = 1/(1 - 2 T2v);
  aEd = 1/(1 - 2 T2v - T3v);

  dlS = Dlam aT + (2/3) Dmu (aT - aEd);
  dmSD = Dmu aEd;
  dmSO = Dmu aEo;
  drS = Drho aU;

  DCv = ConstantArray[0., {6, 6}];
  Do[DCv[[ii, ii]] = 2 dmSD, {ii, 3}];
  Do[DCv[[ii, ii]] = 2 dmSO, {ii, 4, 6}];
  Do[Do[DCv[[ii, jj]] += dlS, {jj, 3}], {ii, 3}];

  tMat = ConstantArray[0. + 0. I, {9, 9}];
  tMat[[1 ;; 3, 1 ;; 3]] =
    VC omega^2 drS IdentityMatrix[3];
  tMat[[4 ;; 9, 4 ;; 9]] = VC N[DCv];

  (* Solve via FFT + GMRES *)
  exc = fftSolveSystem[centres, nPerD, dd, tMat,
    omega, alphaBg, betaBg, rhoBg,
    incidentState, relTol];

  (* Forward scattering amplitude *)
  kPv = omega/alphaBg;
  srcList = Table[
    tMat . exc[[9 (k - 1) + 1 ;; 9 k]],
    {k, nC}];
  fwd = cubeFarFieldP[0., kPv, alphaBg, rhoBg,
    centres, srcList, voigtPairs, voigtWeight];

  {nC, fwd}
];

(* ============================================================ *)
(* Convergence study runner *)
(* Requires: mieFarFieldP, mie (from Mie sections) *)
(* Returns: convergenceData table *)
(* ============================================================ *)

convergenceStudyFFT[nPerDiamList_: {3, 5, 7, 9, 11}] :=
  Module[{mieForward, data},

  mieForward = mieFarFieldP[0., mie];

  Print["\n", Style["Convergence Study (FFT + GMRES)",
    Bold, 14]];
  Print["Mie reference |f(0)| = ", Abs[mieForward], "\n"];

  data = Table[
    Module[{result, nC, fwd, relErr, dd, fill},
      result = laxFoldyForwardAmplitudeFFT[np];
      nC = result[[1]];
      fwd = result[[2]];
      dd = 2 aRadius/np;
      fill = nC dd^3 / (4./3. Pi aRadius^3);
      relErr = Abs[fwd - mieForward]/Abs[mieForward];
      Print["  -> N=", nC, ", f(0)=", fwd,
        ", rel.err=", ScientificForm[relErr, 3], "\n"];
      {np, nC, fill, Abs[fwd], relErr}],
    {np, nPerDiamList}];

  Print["\n--- Summary ---"];
  Print[PaddedForm[
    TableForm[data,
      TableHeadings -> {None,
        {"n/diam", "N_cubes", "V_fill",
         "|f_LF(0)|", "Rel Error"}}],
    {10, 6}]];
  data
];

(* ============================================================ *)
(* Diagnostic: compare FFT matvec vs dense matvec              *)
(* Run this for nPerD=3 (19 cubes, instant) to verify the FFT  *)
(* convolution matches the direct matrix assembly exactly.      *)
(* ============================================================ *)

verifyFFTMatvec[nPerD_: 3] :=
  Module[{dd, halfG, centres, nC, VC, dm,
    Av, Bv, Cv, T1v, T2v, T3v, g0v,
    aU, aT, aEo, aEd, dlS, dmSD, dmSO, drS, DCv, tMat,
    (* FFT infrastructure *)
    nP, nPV, aR, gridIdx, kernel, kernelFFT,
    pack, unpack, matvecFFT,
    (* Dense matrix *)
    sysMat,
    (* Test *)
    xTest, yFFT, yDense, relDiff},

  dd = 2 aRadius/nPerD;
  halfG = Table[(-nPerD/2 + 0.5 + i) dd, {i, 0, nPerD - 1}];
  centres = N[Reap[
    Do[If[Norm[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}] < aRadius,
      Sow[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}]],
    {iz, nPerD}, {ix, nPerD}, {iy, nPerD}]][[2, 1]]];
  nC = Length[centres];
  VC = dd^3;
  dm = 9 nC;

  (* T-matrix (same as laxFoldyForwardAmplitudeFFT) *)
  {Av, Bv, Cv} = cubeABC[omega, dd/2, alphaBg, betaBg, rhoBg];
  T1v = Dlam (Av + 4 Bv + Cv) + 2 Dmu Bv;
  T2v = Dmu (Av + Bv);
  T3v = 2 Dmu Cv;
  g0v = cubeGamma0[omega, dd/2, alphaBg, betaBg, rhoBg];
  aU = 1/(1 - omega^2 Drho g0v);
  aT = 1/(1 - 3 T1v - 2 T2v - T3v);
  aEo = 1/(1 - 2 T2v);
  aEd = 1/(1 - 2 T2v - T3v);
  dlS = Dlam aT + (2/3) Dmu (aT - aEd);
  dmSD = Dmu aEd;
  dmSO = Dmu aEo;
  drS = Drho aU;
  DCv = ConstantArray[0., {6, 6}];
  Do[DCv[[ii, ii]] = 2 dmSD, {ii, 3}];
  Do[DCv[[ii, ii]] = 2 dmSO, {ii, 4, 6}];
  Do[Do[DCv[[ii, jj]] += dlS, {jj, 3}], {ii, 3}];
  tMat = ConstantArray[0. + 0. I, {9, 9}];
  tMat[[1 ;; 3, 1 ;; 3]] = VC omega^2 drS IdentityMatrix[3];
  tMat[[4 ;; 9, 4 ;; 9]] = VC N[DCv];

  Print[Style["FFT vs Dense Matvec Verification", Bold, 14]];
  Print["  nPerD = ", nPerD, ", nCubes = ", nC, ", DOF = ", dm];

  (* ---- Build FFT matvec ---- *)
  nP = 2 nPerD - 1;
  nPV = N[nP^3];
  aR = nPerD dd/2;
  gridIdx = Reap[
    Do[If[Norm[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}] < aR,
      Sow[{iz, ix, iy}]],
    {iz, nPerD}, {ix, nPerD}, {iy, nPerD}]][[2, 1]];
  kernel = ConstantArray[0. + 0. I, {9, 9, nP, nP, nP}];
  Do[If[!(dz == 0 && dx == 0 && dy == 0),
    Module[{xSep, block, izz, ixx, iyy},
      xSep = N[{dz, dx, dy} dd];
      block = -(propagator9x9[xSep, omega, alphaBg, betaBg, rhoBg] . tMat);
      izz = Mod[dz, nP] + 1;
      ixx = Mod[dx, nP] + 1;
      iyy = Mod[dy, nP] + 1;
      Do[kernel[[ii, jj, izz, ixx, iyy]] = block[[ii, jj]],
        {ii, 9}, {jj, 9}]]],
  {dz, -(nPerD - 1), nPerD - 1},
  {dx, -(nPerD - 1), nPerD - 1},
  {dy, -(nPerD - 1), nPerD - 1}];
  kernelFFT = Table[Sqrt[nPV] Fourier[kernel[[ii, jj]]],
    {ii, 9}, {jj, 9}];

  pack[wFlat_] := Module[{grids, idx, gi},
    grids = ConstantArray[0. + 0. I, {9, nP, nP, nP}];
    Do[idx = 9 (j - 1); gi = gridIdx[[j]];
      Do[grids[[c, gi[[1]], gi[[2]], gi[[3]]]] =
           wFlat[[idx + c]], {c, 9}],
    {j, nC}]; grids];

  unpack[grids_] := Module[{wFlat, idx, gi},
    wFlat = ConstantArray[0. + 0. I, dm];
    Do[idx = 9 (j - 1); gi = gridIdx[[j]];
      Do[wFlat[[idx + c]] =
           grids[[c, gi[[1]], gi[[2]], gi[[3]]]],
        {c, 9}],
    {j, nC}]; wFlat];

  matvecFFT[wFlat_] := Module[{wF, yF},
    wF = Map[Fourier, pack[wFlat]];
    yF = Table[Sum[kernelFFT[[i, j]] wF[[j]], {j, 9}], {i, 9}];
    wFlat + unpack[Map[InverseFourier, yF]]];

  Print["  FFT kernel built."];

  (* ---- Build dense system matrix ---- *)
  sysMat = IdentityMatrix[dm] + 0. I;
  Do[If[j != k,
    Module[{xjk, Pjk, block, idxJ, idxK},
      xjk = centres[[j]] - centres[[k]];
      Pjk = propagator9x9[xjk, omega, alphaBg, betaBg, rhoBg];
      block = -Pjk . tMat;
      idxJ = 9 (j - 1); idxK = 9 (k - 1);
      Do[Do[sysMat[[idxJ + ii, idxK + jj]] = block[[ii, jj]],
        {jj, 9}], {ii, 9}]]],
  {j, nC}, {k, nC}];

  Print["  Dense matrix built."];

  (* ---- Compare on random test vector ---- *)
  SeedRandom[42];
  xTest = RandomComplex[{-1 - I, 1 + I}, dm];

  yFFT = matvecFFT[xTest];
  yDense = sysMat . xTest;

  relDiff = Norm[yFFT - yDense]/Norm[yDense];
  Print["\n  ||A_FFT x - A_dense x|| / ||A_dense x|| = ",
    ScientificForm[relDiff, 4]];
  Print["  Max element-wise |diff|: ",
    ScientificForm[Max[Abs[yFFT - yDense]], 4]];

  If[relDiff < 1.*^-10,
    Print[Style["  PASS: FFT matvec matches dense.", Bold, Darker[Green]]],
    Print[Style["  FAIL: FFT matvec differs from dense!", Bold, Red]];
    Print["  First 9 components of difference:"];
    Print["    FFT:   ", yFFT[[1 ;; 9]]];
    Print["    Dense: ", yDense[[1 ;; 9]]];
    Print["    Diff:  ", yFFT[[1 ;; 9]] - yDense[[1 ;; 9]]]];

  (* ---- Part 2: Compare GMRES vs LinearSolve solutions ---- *)
  Print["\n", Style["GMRES vs LinearSolve Comparison", Bold, 14]];

  Module[{rhsVec, excGMRES, excDense, solDiff, resGMRES, resDense},
    (* Build RHS *)
    rhsVec = ConstantArray[0. + 0. I, dm];
    Do[Module[{winc, idx}, idx = 9 (j - 1);
      winc = incidentState[centres[[j]]];
      Do[rhsVec[[idx + c]] = winc[[c]], {c, 9}]],
    {j, nC}];

    (* Solve with GMRES *)
    Print["  Solving with GMRES..."];
    excGMRES = gmresSolve[matvecFFT, rhsVec, rhsVec, 1.*^-10, 200];

    (* Solve with LinearSolve *)
    Print["  Solving with LinearSolve..."];
    excDense = LinearSolve[sysMat, rhsVec];

    (* Compare *)
    solDiff = Norm[excGMRES - excDense]/Norm[excDense];
    resGMRES = Norm[matvecFFT[excGMRES] - rhsVec]/Norm[rhsVec];
    resDense = Norm[sysMat . excDense - rhsVec]/Norm[rhsVec];

    Print["\n  ||x_GMRES - x_Dense|| / ||x_Dense|| = ",
      ScientificForm[solDiff, 4]];
    Print["  GMRES true residual:  ||Ax-b||/||b|| = ",
      ScientificForm[resGMRES, 4]];
    Print["  Dense true residual:  ||Ax-b||/||b|| = ",
      ScientificForm[resDense, 4]];

    If[solDiff < 1.*^-6,
      Print[Style["  PASS: GMRES and LinearSolve agree.",
        Bold, Darker[Green]]],
      Print[Style["  FAIL: GMRES solution differs from LinearSolve!",
        Bold, Red]];
      Print["  Cube 1 excitation:"];
      Print["    GMRES: ", excGMRES[[1 ;; 9]]];
      Print["    Dense: ", excDense[[1 ;; 9]]]];

    (* ---- Part 3: Compare forward scattering amplitudes ---- *)
    Print["\n", Style["Forward Amplitude Comparison", Bold, 14]];
    Module[{kPv, srcGMRES, srcDense, fwdGMRES, fwdDense, fwdDiff},
      kPv = omega/alphaBg;
      srcGMRES = Table[
        tMat . excGMRES[[9 (k - 1) + 1 ;; 9 k]], {k, nC}];
      srcDense = Table[
        tMat . excDense[[9 (k - 1) + 1 ;; 9 k]], {k, nC}];
      fwdGMRES = cubeFarFieldP[0., kPv, alphaBg, rhoBg,
        centres, srcGMRES, voigtPairs, voigtWeight];
      fwdDense = cubeFarFieldP[0., kPv, alphaBg, rhoBg,
        centres, srcDense, voigtPairs, voigtWeight];
      fwdDiff = Abs[fwdGMRES - fwdDense]/Abs[fwdDense];
      Print["  f_P(0) GMRES:      ", fwdGMRES];
      Print["  f_P(0) LinearSolve: ", fwdDense];
      Print["  Relative diff:     ",
        ScientificForm[fwdDiff, 4]];
      If[fwdDiff < 1.*^-6,
        Print[Style["  PASS: Forward amplitudes agree.",
          Bold, Darker[Green]]],
        Print[Style["  FAIL: Forward amplitudes differ!",
          Bold, Red]]]];
  ];

  relDiff
];

(* ============================================================ *)
(* Side-by-side convergence: Dense vs FFT at each nPerD        *)
(* Requires same globals as convergenceStudyFFT + verifyFFTMatvec *)
(* This is the definitive diagnostic for convergence issues.    *)
(* ============================================================ *)

convergenceStudyDenseVsFFT[nPerDiamList_: {3, 5, 7}] :=
  Module[{mieForward},

  mieForward = mieFarFieldP[0., mie];
  Print["\n", Style["Dense vs FFT Convergence Comparison", Bold, 14]];
  Print["Mie reference f(0) = ", mieForward,
    ",  |f(0)| = ", Abs[mieForward], "\n"];

  Table[
    Module[{dd, halfG, centres, nC, VC, dm,
      Av, Bv, Cv, T1v, T2v, T3v, g0v,
      aU, aT, aEo, aEd, dlS, dmSD, dmSO, drS, DCv, tMat,
      (* Dense infrastructure *)
      sysMat, rhsVec, excDense,
      (* FFT infrastructure *)
      excFFT,
      (* Results *)
      kPv, srcDense, srcFFT, fwdDense, fwdFFT,
      errDense, errFFT, solDiff},

    dd = 2 aRadius/np;
    halfG = Table[(-np/2 + 0.5 + i) dd, {i, 0, np - 1}];
    centres = N[Reap[
      Do[If[Norm[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}] < aRadius,
        Sow[{halfG[[iz]], halfG[[ix]], halfG[[iy]]}]],
      {iz, np}, {ix, np}, {iy, np}]][[2, 1]]];
    nC = Length[centres];
    VC = dd^3;
    dm = 9 nC;

    (* T-matrix *)
    {Av, Bv, Cv} = cubeABC[omega, dd/2,
      alphaBg, betaBg, rhoBg];
    T1v = Dlam (Av + 4 Bv + Cv) + 2 Dmu Bv;
    T2v = Dmu (Av + Bv);
    T3v = 2 Dmu Cv;
    g0v = cubeGamma0[omega, dd/2, alphaBg, betaBg, rhoBg];
    aU = 1/(1 - omega^2 Drho g0v);
    aT = 1/(1 - 3 T1v - 2 T2v - T3v);
    aEo = 1/(1 - 2 T2v);
    aEd = 1/(1 - 2 T2v - T3v);
    dlS = Dlam aT + (2/3) Dmu (aT - aEd);
    dmSD = Dmu aEd;
    dmSO = Dmu aEo;
    drS = Drho aU;
    DCv = ConstantArray[0., {6, 6}];
    Do[DCv[[ii, ii]] = 2 dmSD, {ii, 3}];
    Do[DCv[[ii, ii]] = 2 dmSO, {ii, 4, 6}];
    Do[Do[DCv[[ii, jj]] += dlS, {jj, 3}], {ii, 3}];
    tMat = ConstantArray[0. + 0. I, {9, 9}];
    tMat[[1 ;; 3, 1 ;; 3]] =
      VC omega^2 drS IdentityMatrix[3];
    tMat[[4 ;; 9, 4 ;; 9]] = VC N[DCv];

    Print[Style["--- nPerD = " <> ToString[np] <>
      " (" <> ToString[nC] <> " cubes, " <>
      ToString[dm] <> " DOF, fill=" <>
      ToString[NumberForm[nC VC / (4./3. Pi aRadius^3), 4]] <>
      ") ---", Bold]];

    (* Dense solve *)
    sysMat = IdentityMatrix[dm] + 0. I;
    rhsVec = ConstantArray[0. + 0. I, dm];
    Do[Module[{winc, idx}, idx = 9 (j - 1);
      winc = incidentState[centres[[j]]];
      Do[rhsVec[[idx + c]] = winc[[c]], {c, 9}]],
    {j, nC}];
    Do[If[j != k,
      Module[{xjk, Pjk, block, idxJ, idxK},
        xjk = centres[[j]] - centres[[k]];
        Pjk = propagator9x9[xjk, omega, alphaBg, betaBg, rhoBg];
        block = -Pjk . tMat;
        idxJ = 9 (j - 1); idxK = 9 (k - 1);
        Do[Do[sysMat[[idxJ + ii, idxK + jj]] = block[[ii, jj]],
          {jj, 9}], {ii, 9}]]],
    {j, nC}, {k, nC}];
    excDense = LinearSolve[sysMat, rhsVec];

    (* FFT solve *)
    excFFT = fftSolveSystem[centres, np, dd, tMat,
      omega, alphaBg, betaBg, rhoBg, incidentState];

    (* Compare solutions *)
    solDiff = Norm[excFFT - excDense]/Norm[excDense];
    Print["  ||x_FFT - x_Dense|| / ||x_Dense|| = ",
      ScientificForm[solDiff, 4]];

    (* Forward amplitudes *)
    kPv = omega/alphaBg;
    srcDense = Table[tMat . excDense[[9(k-1)+1 ;; 9 k]], {k, nC}];
    srcFFT = Table[tMat . excFFT[[9(k-1)+1 ;; 9 k]], {k, nC}];
    fwdDense = cubeFarFieldP[0., kPv, alphaBg, rhoBg,
      centres, srcDense, voigtPairs, voigtWeight];
    fwdFFT = cubeFarFieldP[0., kPv, alphaBg, rhoBg,
      centres, srcFFT, voigtPairs, voigtWeight];
    errDense = Abs[fwdDense - mieForward]/Abs[mieForward];
    errFFT = Abs[fwdFFT - mieForward]/Abs[mieForward];

    Print["  Dense:  f(0) = ", fwdDense,
      ",  |err| = ", ScientificForm[errDense, 4]];
    Print["  FFT:    f(0) = ", fwdFFT,
      ",  |err| = ", ScientificForm[errFFT, 4]];

    If[solDiff < 1.*^-6,
      Print[Style["  PASS", Bold, Darker[Green]]],
      Print[Style["  FAIL: solutions differ!", Bold, Red]]];
    Print[];

    {np, nC, fwdDense, fwdFFT, errDense, errFFT, solDiff}],
  {np, nPerDiamList}]
];

Print["FFTLaxFoldy.wl loaded: gmresSolve, fftSolveSystem, ",
  "laxFoldyForwardAmplitudeFFT, convergenceStudyFFT, ",
  "verifyFFTMatvec, convergenceStudyDenseVsFFT"];
