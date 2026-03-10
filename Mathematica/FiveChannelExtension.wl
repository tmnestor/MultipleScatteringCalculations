(* ::Package:: *)
(* FiveChannelExtension.wl -- SV/SH Mie + Lax-Foldy five-channel comparison *)
(* Requires sections 1-8 of LaxFoldy_VoxelSphere_vs_Mie.nb to be evaluated *)
(* first (defines: pWaveFields, sWaveFields, miePSVMatrix, mieIncidentPSV, *)
(* systemMatrix, cubeCentres, tMatrix9, excField, etc.)                     *)

Print["=== Five-Channel Extension: SV and SH Incidence ==="];

(* ============================================================ *)
(* 1. New Mie helper functions                                   *)
(* ============================================================ *)

(* SV-incident RHS for the 4x4 P-SV boundary system *)
mieIncidentSV[n_, omega_, a_, alpha_, beta_, rho_, lam_, mu_] :=
  Module[{kS, coeff, fSinc},
  kS = omega/beta;
  coeff = (2 n + 1) I^n/(I kS);
  fSinc = sWaveFields[n, kS, a, mu, jn, jnp];
  -coeff {fSinc[[1]], fSinc[[2]], fSinc[[3]], fSinc[[4]]}
];

(* SH 2x2 boundary matrix (n >= 1) *)
(* M-type (toroidal): u_phi ~ z_n(kS r), sigma_{r phi} ~ mu kS z_n' *)
mieSHMatrix[n_, omega_, a_, betaOut_, muOut_, betaIn_, muIn_] :=
  Module[{kSout, kSin, zOut, zIn, hOut, hpOut, jIn, jpIn},
  kSout = omega/betaOut;
  kSin = omega/betaIn;
  zOut = kSout a;
  zIn = kSin a;
  hOut = hn[n, zOut];
  hpOut = hnp[n, zOut];
  jIn = jn[n, zIn];
  jpIn = jnp[n, zIn];
  (* Row 1: u_phi continuity: h_n(kS_out a) = j_n(kS_in a) *)
  (* Row 2: stress continuity: mu_out kS_out h_n' = mu_in kS_in j_n' *)
  {{hOut, -jIn}, {muOut kSout hpOut, -muIn kSin jpIn}}
];

(* SH incident RHS — M-type: displacement j_n, stress mu kS j_n' *)
mieIncidentSH[n_, omega_, a_, beta_, mu_] :=
  Module[{kS, coeff, z, jInc, jpInc},
  kS = omega/beta;
  coeff = (2 n + 1) I^n/(I kS);
  z = kS a;
  jInc = jn[n, z];
  jpInc = jnp[n, z];
  -coeff {jInc, mu kS jpInc}
];

Print["  Mie helper functions defined (mieIncidentSV, mieSHMatrix, mieIncidentSH)."];

(* ============================================================ *)
(* 2. Extended Mie solver: P-inc + SV-inc + SH coefficients     *)
(* ============================================================ *)

computeMieCoefficientsAll[omega_, a_,
  alphaOut_, betaOut_, rhoOut_, lamOut_, muOut_,
  alphaIn_, betaIn_, rhoIn_, lamIn_, muIn_,
  nMaxIn_] :=
  Module[{kPout, kSout, kPin, kSin,
    anArr, bnArr, anSVArr, bnSVArr, cnArr,
    urS0, srrS0, urI0, srrI0, urInc0, srrInc0,
    coeff0, M0, rhs0, sol0,
    Mpsv, rhsPsv, solPsv, rhsSV, solSV,
    Msh, rhsSH, solSH, sign, n, dummy},

  kPout = omega/alphaOut;
  kSout = omega/betaOut;
  kPin = omega/alphaIn;
  kSin = omega/betaIn;

  anArr = ConstantArray[0. + 0. I, nMaxIn + 1];
  bnArr = ConstantArray[0. + 0. I, nMaxIn + 1];
  anSVArr = ConstantArray[0. + 0. I, nMaxIn + 1];
  bnSVArr = ConstantArray[0. + 0. I, nMaxIn + 1];
  cnArr = ConstantArray[0. + 0. I, nMaxIn + 1];

  (* n=0 monopole: 2x2 P-wave only *)
  {urS0, dummy, srrS0, dummy} =
    pWaveFields[0, kPout, a, lamOut, muOut, hn, hnp];
  {urI0, dummy, srrI0, dummy} =
    pWaveFields[0, kPin, a, lamIn, muIn, jn, jnp];
  {urInc0, dummy, srrInc0, dummy} =
    pWaveFields[0, kPout, a, lamOut, muOut, jn, jnp];
  coeff0 = 1/(I kPout);
  M0 = {{urS0, -urI0}, {srrS0, -srrI0}};
  rhs0 = {-coeff0 urInc0, -coeff0 srrInc0};
  sol0 = LinearSolve[N[M0], N[rhs0]];
  anArr[[1]] = sol0[[1]];

  (* n >= 1: 4x4 P-SV coupled + 2x2 SH *)
  Do[
    sign = (-1.)^n;
    Mpsv = N[miePSVMatrix[n, omega, a,
      alphaOut, betaOut, rhoOut, lamOut, muOut,
      alphaIn, betaIn, rhoIn, lamIn, muIn]];

    (* P-incident *)
    rhsPsv = N[mieIncidentPSV[n, omega, a,
      alphaOut, betaOut, rhoOut, lamOut, muOut]];
    solPsv = LinearSolve[Mpsv, rhsPsv];
    anArr[[n + 1]] = sign solPsv[[1]];
    bnArr[[n + 1]] = sign solPsv[[2]];

    (* SV-incident *)
    rhsSV = N[mieIncidentSV[n, omega, a,
      alphaOut, betaOut, rhoOut, lamOut, muOut]];
    solSV = LinearSolve[Mpsv, rhsSV];
    anSVArr[[n + 1]] = sign solSV[[1]];
    bnSVArr[[n + 1]] = sign solSV[[2]];

    (* SH *)
    Msh = N[mieSHMatrix[n, omega, a, betaOut, muOut, betaIn, muIn]];
    rhsSH = N[mieIncidentSH[n, omega, a, betaOut, muOut]];
    solSH = LinearSolve[Msh, rhsSH];
    cnArr[[n + 1]] = sign solSH[[1]],
  {n, 1, nMaxIn}];

  <|"a" -> anArr, "b" -> bnArr,
    "aSV" -> anSVArr, "bSV" -> bnSVArr, "c" -> cnArr,
    "nMax" -> nMaxIn|>
];

Print["  Extended Mie solver defined (computeMieCoefficientsAll)."];

(* ============================================================ *)
(* 3. Mie far-field functions for all 5 channels                *)
(* ============================================================ *)

(* Angular function helpers *)
dPndTheta[n_, theta_] := -Sin[theta] D[LegendreP[n, x], x] /. x -> Cos[theta];

dPn1dTheta[n_, theta_] :=
  Module[{cosT = Cos[theta], sinT = Sin[theta], Pn, dPn},
  If[Abs[sinT] < 1.*^-12,
    (* Limit: use finite difference *)
    Module[{dt = 1.*^-6},
      (dPndTheta[n, theta + dt] - dPndTheta[n, theta - dt])/(2 dt)],
    Pn = LegendreP[n, cosT];
    dPn = dPndTheta[n, theta];
    -cosT/sinT dPn - n (n + 1) Pn
  ]];

Pn1OverSinTheta[n_, theta_] :=
  Module[{cosT = Cos[theta], sinT = Sin[theta]},
  If[Abs[sinT] < 1.*^-12,
    (* Limit of -P'_n(cos theta) at poles *)
    If[cosT > 0,
      N[-n (n + 1)/2],              (* theta = 0: -P'_n(1) *)
      N[(-1)^n n (n + 1)/2]],       (* theta = pi: -P'_n(-1) *)
    dPndTheta[n, theta]/sinT
  ]];

(* PP: P-inc -> P-scattered (m=0) *)
(* Uses full radial functions at large r, then extracts via r*exp(-ikPr) *)
mieFarFieldPP[theta_, mieCoeffs_, omega_, alphaBg_, lamBg_, muBg_] :=
  Module[{nMaxC, an, kPv, rEval, uR = 0. + 0. I, n},
  nMaxC = mieCoeffs["nMax"]; an = mieCoeffs["a"];
  kPv = omega/alphaBg;
  rEval = 1.*^6 * aRadius;
  Do[
    Module[{Pn, urP, dummy},
      Pn = N[LegendreP[n, Cos[theta]]];
      {urP, dummy, dummy, dummy} = pWaveFields[n, kPv, rEval, lamBg, muBg, hn, hnp];
      uR += an[[n + 1]] urP Pn],
  {n, 0, nMaxC}];
  uR rEval Exp[-I kPv rEval]
];

(* PS: P-inc -> SV-scattered (m=0) *)
mieFarFieldPS[theta_, mieCoeffs_, omega_, alphaBg_, betaBg_, muBg_] :=
  Module[{nMaxC, bn, kSv, rEval, uTheta = 0. + 0. I, n},
  nMaxC = mieCoeffs["nMax"]; bn = mieCoeffs["b"];
  kSv = omega/betaBg;
  rEval = 1.*^6 * aRadius;
  Do[
    Module[{dPn, dummy, utS},
      dPn = N[dPndTheta[n, theta]];
      {dummy, utS, dummy, dummy} = sWaveFields[n, kSv, rEval, muBg, hn, hnp];
      uTheta += bn[[n + 1]] utS dPn],
  {n, 1, nMaxC}];
  uTheta rEval Exp[-I kSv rEval]
];

(* SP: SV-inc -> P-scattered (m=1, renorm = -1/[n(n+1)]) *)
mieFarFieldSP[theta_, mieCoeffs_, omega_, alphaBg_, lamBg_, muBg_] :=
  Module[{nMaxC, anSV, kPv, rEval, uR = 0. + 0. I, n},
  nMaxC = mieCoeffs["nMax"]; anSV = mieCoeffs["aSV"];
  kPv = omega/alphaBg;
  rEval = 1.*^6 * aRadius;
  Do[
    Module[{renorm, Pn1, urP, dummy},
      renorm = -1./(n (n + 1));
      Pn1 = N[dPndTheta[n, theta]];  (* P_n^1 = dP_n/dtheta *)
      {urP, dummy, dummy, dummy} = pWaveFields[n, kPv, rEval, lamBg, muBg, hn, hnp];
      uR += anSV[[n + 1]] renorm urP Pn1],
  {n, 1, nMaxC}];
  uR rEval Exp[-I kPv rEval]
];

(* SS: SV-inc -> SV-scattered (m=1, renorm = -1/[n(n+1)]) *)
mieFarFieldSS[theta_, mieCoeffs_, omega_, betaBg_, muBg_] :=
  Module[{nMaxC, bnSV, kSv, rEval, uTheta = 0. + 0. I, n},
  nMaxC = mieCoeffs["nMax"]; bnSV = mieCoeffs["bSV"];
  kSv = omega/betaBg;
  rEval = 1.*^6 * aRadius;
  Do[
    Module[{renorm, dPn1, dummy, utS},
      renorm = -1./(n (n + 1));
      dPn1 = N[dPn1dTheta[n, theta]];
      {dummy, utS, dummy, dummy} = sWaveFields[n, kSv, rEval, muBg, hn, hnp];
      uTheta += bnSV[[n + 1]] renorm utS dPn1],
  {n, 1, nMaxC}];
  uTheta rEval Exp[-I kSv rEval]
];

(* SH: SH-inc -> SH-scattered (m=1, N-type: b_n_sv with pi_n) *)
mieFarFieldSH[theta_, mieCoeffs_, omega_, betaBg_, muBg_] :=
  Module[{nMaxC, bnSV, kSv, rEval, uPhi = 0. + 0. I, n},
  nMaxC = mieCoeffs["nMax"]; bnSV = mieCoeffs["bSV"];
  kSv = omega/betaBg;
  rEval = 1.*^6 * aRadius;
  Do[
    Module[{renormN, piN, dummy, utS},
      renormN = -1./(n (n + 1));
      piN = N[Pn1OverSinTheta[n, theta]];
      {dummy, utS, dummy, dummy} = sWaveFields[n, kSv, rEval, muBg, hn, hnp];
      uPhi += bnSV[[n + 1]] renormN utS piN],
  {n, 1, nMaxC}];
  uPhi rEval Exp[-I kSv rEval]
];

Print["  Far-field functions defined (PP, PS, SP, SS, SH)."];

(* ============================================================ *)
(* 4. Incident state vectors for P / SV / SH                    *)
(* 9-vector: {u_z, u_x, u_y, eps_zz, eps_xx, eps_yy,           *)
(*            2*eps_xy, 2*eps_zy, 2*eps_zx}                     *)
(* ============================================================ *)

incidentStateP[pos_] :=
  Module[{kPv, phase},
  kPv = omega/alphaBg;
  phase = Exp[I kPv pos[[1]]];
  {phase, 0., 0., I kPv phase, 0., 0., 0., 0., 0.}
];

incidentStateSV[pos_] :=
  Module[{kSv, phase},
  kSv = omega/betaBg;
  phase = Exp[I kSv pos[[1]]];
  {0., phase, 0., 0., 0., 0., 0., 0., I kSv phase}
];

incidentStateSH[pos_] :=
  Module[{kSv, phase},
  kSv = omega/betaBg;
  phase = Exp[I kSv pos[[1]]];
  {0., 0., phase, 0., 0., 0., 0., I kSv phase, 0.}
];

Print["  Incident state functions defined (P, SV, SH)."];

(* ============================================================ *)
(* 5. Compute extended Mie coefficients                          *)
(* ============================================================ *)

mieAll = computeMieCoefficientsAll[omega, aRadius,
  alphaBg, betaBg, rhoBg, lamBg, muBg,
  alphaIn, betaIn, rhoIn, lamIn, muIn, nMax];

Print["\nExtended Mie coefficients computed:"];
Print["  a_n (P->P):  ", mieAll["a"][[1 ;; Min[4, nMax + 1]]]];
Print["  b_n (P->SV): ", mieAll["b"][[1 ;; Min[4, nMax + 1]]]];
Print["  aSV_n (SV->P):  ", mieAll["aSV"][[1 ;; Min[4, nMax + 1]]]];
Print["  bSV_n (SV->SV): ", mieAll["bSV"][[1 ;; Min[4, nMax + 1]]]];
Print["  c_n (SH->SH):   ", mieAll["c"][[1 ;; Min[4, nMax + 1]]]];

(* ============================================================ *)
(* 6. Lax-Foldy solves for SV and SH incidence                  *)
(*    Reuse the system matrix from section 7 of the notebook     *)
(* ============================================================ *)

Print["\nBuilding SV and SH RHS vectors..."];

(* SV-incident RHS *)
rhsVectorSV = ConstantArray[0. + 0. I, 9 nCubes];
Do[
  Module[{winc, idx},
    idx = 9 (j - 1);
    winc = incidentStateSV[cubeCentres[[j]]];
    Do[rhsVectorSV[[idx + ii]] = winc[[ii]], {ii, 9}];
  ],
{j, nCubes}];

(* SH-incident RHS *)
rhsVectorSH = ConstantArray[0. + 0. I, 9 nCubes];
Do[
  Module[{winc, idx},
    idx = 9 (j - 1);
    winc = incidentStateSH[cubeCentres[[j]]];
    Do[rhsVectorSH[[idx + ii]] = winc[[ii]], {ii, 9}];
  ],
{j, nCubes}];

Print["Solving Lax-Foldy for SV incidence..."];
excFieldSV = LinearSolve[systemMatrix, rhsVectorSV];
Print["Solving Lax-Foldy for SH incidence..."];
excFieldSH = LinearSolve[systemMatrix, rhsVectorSH];
Print["Both solves complete."];

(* Compute effective sources for each incidence *)
sourcesListSV = Table[
  tMatrix9 . excFieldSV[[9 (kk - 1) + 1 ;; 9 kk]],
{kk, nCubes}];

sourcesListSH = Table[
  tMatrix9 . excFieldSH[[9 (kk - 1) + 1 ;; 9 kk]],
{kk, nCubes}];

Print["Sources computed for SV and SH incidence."];

(* ============================================================ *)
(* 7. Compute far-field amplitudes for all 5 channels            *)
(* ============================================================ *)

Print["\nComputing far-field amplitudes for all 5 channels..."];

(* --- P-incidence: PP and PS --- *)
(* PP: use existing cubeFarFieldP with sourcesList from section 8 *)
lfPP = Table[
  cubeFarFieldP[thetaGrid[[i]], kP, alphaBg, rhoBg,
    cubeCentres, sourcesList, voigtPairs, voigtWeight],
{i, nAngles}];

(* PS: S-wave far-field from P-incident sources, take SV component *)
lfPS = Table[
  Module[{sv, sh},
    {sv, sh} = cubeFarFieldS[thetaGrid[[i]], kS, betaBg, rhoBg,
      cubeCentres, sourcesList, voigtPairs, voigtWeight];
    sv],
{i, nAngles}];

(* --- SV-incidence: SP and SS --- *)
lfSP = Table[
  cubeFarFieldP[thetaGrid[[i]], kP, alphaBg, rhoBg,
    cubeCentres, sourcesListSV, voigtPairs, voigtWeight],
{i, nAngles}];

lfSS = Table[
  Module[{sv, sh},
    {sv, sh} = cubeFarFieldS[thetaGrid[[i]], kS, betaBg, rhoBg,
      cubeCentres, sourcesListSV, voigtPairs, voigtWeight];
    sv],
{i, nAngles}];

(* --- SH-incidence: SH --- *)
lfSH = Table[
  Module[{sv, sh},
    {sv, sh} = cubeFarFieldS[thetaGrid[[i]], kS, betaBg, rhoBg,
      cubeCentres, sourcesListSH, voigtPairs, voigtWeight];
    sh],
{i, nAngles}];

(* --- Mie far-field amplitudes --- *)
miePP = Table[mieFarFieldPP[thetaGrid[[i]], mieAll, omega, alphaBg, lamBg, muBg],
  {i, nAngles}];
miePS = Table[mieFarFieldPS[thetaGrid[[i]], mieAll, omega, alphaBg, betaBg, muBg],
  {i, nAngles}];
mieSP = Table[mieFarFieldSP[thetaGrid[[i]], mieAll, omega, alphaBg, lamBg, muBg],
  {i, nAngles}];
mieSS = Table[mieFarFieldSS[thetaGrid[[i]], mieAll, omega, betaBg, muBg],
  {i, nAngles}];
mieSH = Table[mieFarFieldSH[thetaGrid[[i]], mieAll, omega, betaBg, muBg],
  {i, nAngles}];

Print["All far-field amplitudes computed."];

(* ============================================================ *)
(* 8. Comparison table and error metrics                         *)
(* ============================================================ *)

fiveChannelCompare[label_, lfData_, mieData_] :=
  Module[{refMag, errRe, errIm, errMag},
  refMag = Max[Max[Abs[mieData]], Max[Abs[lfData]], 1.*^-30];
  errRe = Max[Abs[Re[lfData] - Re[mieData]]]/refMag;
  errIm = Max[Abs[Im[lfData] - Im[mieData]]]/refMag;
  errMag = Max[Abs[Abs[lfData] - Abs[mieData]]]/refMag;
  Print["  ", label, ":  err_Re=", NumberForm[errRe, {4, 3}],
    "  err_Im=", NumberForm[errIm, {4, 3}],
    "  err_|f|=", NumberForm[errMag, {4, 3}]];
];

Print["\n=== Five-Channel Error Summary ==="];
Print["  ka_P = ", N[kP aRadius], ",  ka_S = ", N[kS aRadius]];
fiveChannelCompare["P->P  ", lfPP, miePP];
fiveChannelCompare["P->SV ", lfPS, miePS];
fiveChannelCompare["SV->P ", lfSP, mieSP];
fiveChannelCompare["SV->SV", lfSS, mieSS];
fiveChannelCompare["SH->SH", lfSH, mieSH];

(* ============================================================ *)
(* 9. Comparison plots: Re and Im for all 5 channels             *)
(* ============================================================ *)

Print["\nGenerating comparison plots..."];

fiveChannelPlot[label_, lfData_, mieData_] :=
  Module[{thetaDeg, pltRe, pltIm},
  thetaDeg = thetaGrid 180/Pi;
  pltRe = Show[
    ListLinePlot[Transpose[{thetaDeg, Re[mieData]}],
      PlotStyle -> {Thick, Blue},
      PlotLegends -> {"Mie (exact)"}],
    ListLinePlot[Transpose[{thetaDeg, Re[lfData]}],
      PlotStyle -> {Thick, Red, Dashed},
      PlotLegends -> {"Lax-Foldy (" <> ToString[nCubes] <> " cubes)"}],
    FrameLabel -> {"Scattering angle \[Theta] (deg)", "Re[f]"},
    PlotLabel -> label <> " Re[f]: Lax-Foldy vs Mie",
    Frame -> True, ImageSize -> 500, PlotRange -> All
  ];
  pltIm = Show[
    ListLinePlot[Transpose[{thetaDeg, Im[mieData]}],
      PlotStyle -> {Thick, Blue},
      PlotLegends -> {"Mie (exact)"}],
    ListLinePlot[Transpose[{thetaDeg, Im[lfData]}],
      PlotStyle -> {Thick, Red, Dashed},
      PlotLegends -> {"Lax-Foldy (" <> ToString[nCubes] <> " cubes)"}],
    FrameLabel -> {"Scattering angle \[Theta] (deg)", "Im[f]"},
    PlotLabel -> label <> " Im[f]: Lax-Foldy vs Mie",
    Frame -> True, ImageSize -> 500, PlotRange -> All
  ];
  Print[GraphicsRow[{pltRe, pltIm}, ImageSize -> 1000,
    PlotLabel -> Style[label, Bold, 14]]];
];

fiveChannelPlot["P \[Rule] P", lfPP, miePP];
fiveChannelPlot["P \[Rule] SV", lfPS, miePS];
fiveChannelPlot["SV \[Rule] P", lfSP, mieSP];
fiveChannelPlot["SV \[Rule] SV", lfSS, mieSS];
fiveChannelPlot["SH \[Rule] SH", lfSH, mieSH];

(* ============================================================ *)
(* 10. Publication dashboard: 5x2 grid exported as PDF          *)
(* ============================================================ *)

Print["\nBuilding publication dashboard..."];

makeChannelPanel[lfData_, mieData_, comp_, channelLabel_, isBottom_] :=
  Module[{thetaDeg, yData, pltLabel, xLabel, yLabel},
  thetaDeg = thetaGrid 180/Pi;
  yData = If[comp === "Re", Re, Im];
  yLabel = If[comp === "Re", "Re[f]", "Im[f]"];
  pltLabel = If[comp === "Re", Style[channelLabel, FontFamily -> "Helvetica", FontSize -> 9], None];
  xLabel = If[isBottom, "\[Theta] (deg)", None];
  ListLinePlot[
    {Transpose[{thetaDeg, yData[mieData]}],
     Transpose[{thetaDeg, yData[lfData]}]},
    PlotStyle -> {
      {AbsoluteThickness[1.4], RGBColor[0.27, 0.51, 0.71]},           (* steel blue, solid *)
      {AbsoluteThickness[1.4], RGBColor[0.80, 0.14, 0.18], Dashed}},  (* crimson, dashed *)
    Frame -> True,
    FrameLabel -> {{yLabel, None}, {xLabel, None}},
    PlotLabel -> pltLabel,
    PlotRange -> All,
    FrameTicksStyle -> Directive[FontFamily -> "Helvetica", FontSize -> 7],
    LabelStyle -> Directive[FontFamily -> "Helvetica", FontSize -> 8],
    ImagePadding -> {{45, 8}, {If[isBottom, 30, 8], If[pltLabel =!= None, 18, 5]}},
    ImageSize -> {240, 110},
    PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.05], Scaled[0.05]}}
  ]
];

Module[{channelSpecs, grid, legend, mieLine, lfLine, dashboardFig, exportPath},

  channelSpecs = {
    {"P \[Rule] P",   lfPP, miePP},
    {"P \[Rule] SV",  lfPS, miePS},
    {"SV \[Rule] P",  lfSP, mieSP},
    {"SV \[Rule] SV", lfSS, mieSS},
    {"SH \[Rule] SH", lfSH, mieSH}
  };

  grid = GraphicsGrid[
    Table[
      {makeChannelPanel[channelSpecs[[row, 2]], channelSpecs[[row, 3]],
          "Re", channelSpecs[[row, 1]], row == 5],
       makeChannelPanel[channelSpecs[[row, 2]], channelSpecs[[row, 3]],
          "Im", channelSpecs[[row, 1]], row == 5]},
    {row, 5}],
    Spacings -> {5, 2},
    ImageSize -> 482     (* ~170 mm *)
  ];

  (* Shared legend *)
  mieLine = Graphics[{RGBColor[0.27, 0.51, 0.71], AbsoluteThickness[1.4],
    Line[{{0, 0}, {20, 0}}]}, ImageSize -> {22, 8}];
  lfLine = Graphics[{RGBColor[0.80, 0.14, 0.18], AbsoluteThickness[1.4], Dashing[{4, 3}],
    Line[{{0, 0}, {20, 0}}]}, ImageSize -> {22, 8}];

  legend = Framed[
    Row[{mieLine, Style[" Mie (exact)   ", FontFamily -> "Helvetica", FontSize -> 8],
         lfLine,  Style[" Lax-Foldy (" <> ToString[nCubes] <> " cubes)",
                        FontFamily -> "Helvetica", FontSize -> 8]},
      Spacer[4]],
    FrameStyle -> GrayLevel[0.6], RoundingRadius -> 3,
    FrameMargins -> {{8, 8}, {3, 3}}
  ];

  dashboardFig = Column[{grid, legend}, Alignment -> Center, Spacings -> 1];

  (* Export to global variable for interactive tweaking *)
  dashboardFigure = dashboardFig;

  exportPath = FileNameJoin[{NotebookDirectory[], "FiveChannel_Dashboard.pdf"}];
  Export[exportPath, dashboardFig, "PDF"];
  Print["  Dashboard exported to: ", exportPath];
];

Print["\n=== Five-Channel Extension Complete ==="];
