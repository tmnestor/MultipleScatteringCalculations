(* ::Package:: *)
(* InterVoxelPropagatorCHDynamic.wl
   Dynamic corrections for C/H displacement-strain coupling blocks.

   Computes third derivatives of triharmonic (rho^3) and pentaharmonic (rho^5)
   potentials for the omega^2 and omega^4 corrections to dG/dR:

     dG^(1)/dR_k = c_1 [delta_{ij} dPsi/dR_k  - eta_1 d^3X/(dR_i dR_j dR_k)]
     dG^(2)/dR_k = c_2 [delta_{ij} dX/dR_k    - eta_2 d^3Omega/(dR_i dR_j dR_k)]

   Values computed:
     n=1 (omega^2): d^3 X/dR^3    (triharmonic, kernel rho^3)   -- 8 values
     n=2 (omega^4): d^3 Omega/dR^3 (pentaharmonic, kernel rho^5) -- 8 values

   Isotropic pieces via Laplacian identity:
     dPsi/dR_k   = (1/12) Sum_j d^3X/(dR_j^2 dR_k)       [nabla^2 rho^3 = 12 rho]
     dX/dR_k     = (1/30) Sum_j d^3Omega/(dR_j^2 dR_k)    [nabla^2 rho^5 = 30 rho^3]

   Method: delta-prime / delta / step collapse (same as FirstDeriv.wl)
   with smoother kernels — no singularity issues.

   Validation: direct 3D NIntegrate of dPsi/dR, dX/dR vs Laplacian-derived.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorCHDynamic.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  DYNAMIC C/H CORRECTIONS: d^3X/dR^3 and d^3Omega/dR^3"];
Print["  For omega^2 and omega^4 displacement-strain coupling"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];

(* ---- Utility functions (same as FirstDeriv.wl) ---- *)
tent[x_] := Max[0, 1 - Abs[x]];
stepFn[x_] := Piecewise[{{+1, 0 < x < 1}, {-1, 1 < x < 2}}, 0];

(* NIntegrate settings — smoother kernels than static, converge easily *)
wpNI = 25;
pgNI = 18;


(* ================================================================ *)
(* Section 1: FACE  R = (1,0,0) — C_{4v} symmetry                   *)
(*                                                                    *)
(* Tents: tent(w0 - 1) shifted, tent(w1) centered, tent(w2) centered *)
(* D_000 = d^3W/dR_0^3 : delta-prime on w0, tent on w1 and w2       *)
(* D_011 = d^3W/(dR_0 dR_1^2) : step on w0, delta on w1, tent on w2 *)
(* ================================================================ *)

Print["==== Section 1: FACE R = (1,0,0) — C_{4v} ===="];
Print[];

(* --- D_000: delta-prime collapse ---
   delta'(w0-c) on rho^np gives -d/dw0[rho^np]|_{w0=c} = -np*c*rho^{np-2}
   F_{np}(c) = np * c * int tent(w1) tent(w2) rho^{np-2}(c,w1,w2) dw1 dw2
   D_000 = Sum alpha_a * F(c_a),  alpha = {+1,-2,+1}, c = {0,1,2}
*)

faceDpF[c_?NumericQ, np_Integer] := If[c == 0, 0`25,
  np * c * NIntegrate[
    tent[w1] tent[w2] (c^2 + w1^2 + w2^2)^((np - 2)/2),
    {w1, -1, 0, 1}, {w2, -1, 0, 1},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

(* --- D_011: step x delta collapse ---
   d/dR_0 -> -step(w0),  d^2/dR_1^2 -> delta at b in {-1,0,1}
   G_{np}(b) = int [-step(w0)] tent(w2) rho^np(w0,b,w2) dw0 dw2
   D_011 = 2*G(1) - 2*G(0)  by |b| symmetry
*)

faceSdG[b_?NumericQ, np_Integer] := NIntegrate[
  (-stepFn[w0]) tent[w2] (w0^2 + b^2 + w2^2)^(np/2),
  {w0, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

(* ---- Triharmonic (rho^3) ---- *)
Print["--- d^3 X/dR^3 (rho^3 kernel) ---"];
t0 = AbsoluteTime[];

fxF1 = faceDpF[1, 3]; fxF2 = faceDpF[2, 3];
D000faceX = faceDpF[0, 3] - 2*fxF1 + fxF2;

fxG0 = faceSdG[0, 3]; fxG1 = faceSdG[1, 3];
D011faceX = 2*fxG1 - 2*fxG0;

Print["  D_000 = ", NumberForm[D000faceX, 18]];
Print["  D_011 = ", NumberForm[D011faceX, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* ---- Pentaharmonic (rho^5) ---- *)
Print["--- d^3 Omega/dR^3 (rho^5 kernel) ---"];
t0 = AbsoluteTime[];

foF1 = faceDpF[1, 5]; foF2 = faceDpF[2, 5];
D000faceOm = faceDpF[0, 5] - 2*foF1 + foF2;

foG0 = faceSdG[0, 5]; foG1 = faceSdG[1, 5];
D011faceOm = 2*foG1 - 2*foG0;

Print["  D_000 = ", NumberForm[D000faceOm, 18]];
Print["  D_011 = ", NumberForm[D011faceOm, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* Laplacian-derived isotropic pieces *)
dPsi0face = (D000faceX + 2*D011faceX) / 12;
dX0face = (D000faceOm + 2*D011faceOm) / 30;
Print["  dPsi_0(face) = (D3X_000 + 2*D3X_011)/12 = ", NumberForm[dPsi0face, 18]];
Print["  dX_0(face)   = (D3Om_000 + 2*D3Om_011)/30 = ", NumberForm[dX0face, 18]];
Print[];


(* ================================================================ *)
(* Section 2: EDGE  R = (1,1,0) — C_{2v} symmetry                   *)
(*                                                                    *)
(* Tents: tent(w0-1) shifted, tent(w1-1) shifted, tent(w2) centered *)
(* D_000: delta-prime on w0, tent(w1-1) on w1, tent on w2           *)
(* D_001: delta on w0, step on w1, tent on w2                       *)
(* D_022: step on w0, tent(w1-1) on w1, delta on w2                 *)
(* ================================================================ *)

Print["==== Section 2: EDGE R = (1,1,0) — C_{2v} ===="];
Print[];

(* --- D_000: delta-prime on w0 --- *)
edgeDpF[c_?NumericQ, np_Integer] := If[c == 0, 0`25,
  np * c * NIntegrate[
    tent[w1 - 1] tent[w2] (c^2 + w1^2 + w2^2)^((np - 2)/2),
    {w1, 0, 1, 2}, {w2, -1, 0, 1},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

(* --- D_001: delta on w0 at {0,1,2}, step on w1 ---
   H(c) = int [-step(w1)] tent(w2) rho^np(c,w1,w2) dw1 dw2
   D_001 = 1*H(0) + (-2)*H(1) + 1*H(2)
*)
edgeDsH[c_?NumericQ, np_Integer] := NIntegrate[
  (-stepFn[w1]) tent[w2] (c^2 + w1^2 + w2^2)^(np/2),
  {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

(* --- D_022: step on w0, delta on w2 at b={-1,0,1} ---
   K(b) = int [-step(w0)] tent(w1-1) rho^np(w0,w1,b) dw0 dw1
   D_022 = 2*K(1) - 2*K(0)  by |b| symmetry
*)
edgeSdK[b_?NumericQ, np_Integer] := NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] (w0^2 + w1^2 + b^2)^(np/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

(* ---- Triharmonic (rho^3) ---- *)
Print["--- d^3 X/dR^3 (rho^3 kernel) ---"];
t0 = AbsoluteTime[];

exF1 = edgeDpF[1, 3]; exF2 = edgeDpF[2, 3];
D000edgeX = edgeDpF[0, 3] - 2*exF1 + exF2;

exH0 = edgeDsH[0, 3]; exH1 = edgeDsH[1, 3]; exH2 = edgeDsH[2, 3];
D001edgeX = exH0 - 2*exH1 + exH2;

exK0 = edgeSdK[0, 3]; exK1 = edgeSdK[1, 3];
D022edgeX = 2*exK1 - 2*exK0;

Print["  D_000 = ", NumberForm[D000edgeX, 18]];
Print["  D_001 = ", NumberForm[D001edgeX, 18]];
Print["  D_022 = ", NumberForm[D022edgeX, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* ---- Pentaharmonic (rho^5) ---- *)
Print["--- d^3 Omega/dR^3 (rho^5 kernel) ---"];
t0 = AbsoluteTime[];

eoF1 = edgeDpF[1, 5]; eoF2 = edgeDpF[2, 5];
D000edgeOm = edgeDpF[0, 5] - 2*eoF1 + eoF2;

eoH0 = edgeDsH[0, 5]; eoH1 = edgeDsH[1, 5]; eoH2 = edgeDsH[2, 5];
D001edgeOm = eoH0 - 2*eoH1 + eoH2;

eoK0 = edgeSdK[0, 5]; eoK1 = edgeSdK[1, 5];
D022edgeOm = 2*eoK1 - 2*eoK0;

Print["  D_000 = ", NumberForm[D000edgeOm, 18]];
Print["  D_001 = ", NumberForm[D001edgeOm, 18]];
Print["  D_022 = ", NumberForm[D022edgeOm, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* Laplacian-derived isotropic pieces *)
(* Face: D_000 + D_001 + D_022 = LF * dW, but for edge axis 0:
   Sum_k d^3W/(dR_k^2 dR_0) = d^3W_000 + d^3W_001 + d^3W_022  (= D_000 + D_001 + D_022)
   [D_000 = d^3W/dR_0^3, D_001 = d^3W/(dR_0^2 dR_1), but Laplacian sums dR_k^2 dR_0:
    k=0: d^3W/dR_0^3 = D_000,  k=1: d^3W/(dR_1^2 dR_0) = D_001,  k=2: d^3W/(dR_2^2 dR_0) = D_022]
*)
dPsi0edge = (D000edgeX + D001edgeX + D022edgeX) / 12;
dX0edge = (D000edgeOm + D001edgeOm + D022edgeOm) / 30;
Print["  dPsi_0(edge) = (D3X_000 + D3X_001 + D3X_022)/12 = ", NumberForm[dPsi0edge, 18]];
Print["  dX_0(edge)   = (D3Om_000 + D3Om_001 + D3Om_022)/30 = ", NumberForm[dX0edge, 18]];
Print[];


(* ================================================================ *)
(* Section 3: CORNER  R = (1,1,1) — S_3 symmetry                    *)
(*                                                                    *)
(* Tents: tent(w0-1), tent(w1-1), tent(w2-1) all shifted            *)
(* D_000: delta-prime on w0, tent(w1-1), tent(w2-1)                 *)
(* D_001: delta on w0, step on w1, tent(w2-1)                       *)
(* D_012: step x step x step (3D NIntegrate, no collapse)           *)
(* ================================================================ *)

Print["==== Section 3: CORNER R = (1,1,1) — S_3 ===="];
Print[];

(* --- D_000: delta-prime on w0 --- *)
cornerDpF[c_?NumericQ, np_Integer] := If[c == 0, 0`25,
  np * c * NIntegrate[
    tent[w1 - 1] tent[w2 - 1] (c^2 + w1^2 + w2^2)^((np - 2)/2),
    {w1, 0, 1, 2}, {w2, 0, 1, 2},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

(* --- D_001: delta on w0 at {0,1,2}, step on w1 --- *)
cornerDsH[c_?NumericQ, np_Integer] := NIntegrate[
  (-stepFn[w1]) tent[w2 - 1] (c^2 + w1^2 + w2^2)^(np/2),
  {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

(* --- D_012: step x step x step (3D NIntegrate, no collapse) ---
   D_012 = -int step(w0) step(w1) step(w2) rho^np d^3w
   Overall sign: (-1)^3 = -1 from three d/dR derivatives
*)
cornerS3[np_Integer] := -NIntegrate[
  stepFn[w0] stepFn[w1] stepFn[w2] (w0^2 + w1^2 + w2^2)^(np/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16, MaxRecursion -> 20
];

(* ---- Triharmonic (rho^3) ---- *)
Print["--- d^3 X/dR^3 (rho^3 kernel) ---"];
t0 = AbsoluteTime[];

cxF1 = cornerDpF[1, 3]; cxF2 = cornerDpF[2, 3];
D000cornerX = cornerDpF[0, 3] - 2*cxF1 + cxF2;

cxH0 = cornerDsH[0, 3]; cxH1 = cornerDsH[1, 3]; cxH2 = cornerDsH[2, 3];
D001cornerX = cxH0 - 2*cxH1 + cxH2;

D012cornerX = cornerS3[3];

Print["  D_000 = ", NumberForm[D000cornerX, 18]];
Print["  D_001 = ", NumberForm[D001cornerX, 18]];
Print["  D_012 = ", NumberForm[D012cornerX, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* ---- Pentaharmonic (rho^5) ---- *)
Print["--- d^3 Omega/dR^3 (rho^5 kernel) ---"];
t0 = AbsoluteTime[];

coF1 = cornerDpF[1, 5]; coF2 = cornerDpF[2, 5];
D000cornerOm = cornerDpF[0, 5] - 2*coF1 + coF2;

coH0 = cornerDsH[0, 5]; coH1 = cornerDsH[1, 5]; coH2 = cornerDsH[2, 5];
D001cornerOm = coH0 - 2*coH1 + coH2;

D012cornerOm = cornerS3[5];

Print["  D_000 = ", NumberForm[D000cornerOm, 18]];
Print["  D_001 = ", NumberForm[D001cornerOm, 18]];
Print["  D_012 = ", NumberForm[D012cornerOm, 18]];
Print["  Time: ", Round[AbsoluteTime[] - t0, 0.1], " s"];
Print[];

(* Laplacian-derived isotropic pieces *)
(* Corner S_3: D_000 + 2*D_001 = LF * dW *)
dPsi0corner = (D000cornerX + 2*D001cornerX) / 12;
dX0corner = (D000cornerOm + 2*D001cornerOm) / 30;
Print["  dPsi_0(corner) = (D3X_000 + 2*D3X_001)/12 = ", NumberForm[dPsi0corner, 18]];
Print["  dX_0(corner)   = (D3Om_000 + 2*D3Om_001)/30 = ", NumberForm[dX0corner, 18]];
Print[];


(* ================================================================ *)
(* Section 4: DIRECT VALIDATION                                      *)
(*                                                                    *)
(* Compute dPsi/dR_0 and dX/dR_0 via direct 3D NIntegrate:          *)
(*   dPsi/dR_0 = int [-step(w0)] tent_1 tent_2 rho d^3w             *)
(*   dX/dR_0   = int [-step(w0)] tent_1 tent_2 rho^3 d^3w           *)
(* Compare with Laplacian-derived values from Section 1-3.           *)
(* ================================================================ *)

Print["================================================================"];
Print["  DIRECT VALIDATION: dPsi/dR and dX/dR via 3D NIntegrate"];
Print["================================================================"];
Print[];

(* --- Face R=(1,0,0) --- *)
Print["--- Face direct validation ---"];
t0 = AbsoluteTime[];

dPsi0faceDir = NIntegrate[
  (-stepFn[w0]) tent[w1] tent[w2] Sqrt[w0^2 + w1^2 + w2^2],
  {w0, 0, 1, 2}, {w1, -1, 0, 1}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
dX0faceDir = NIntegrate[
  (-stepFn[w0]) tent[w1] tent[w2] (w0^2 + w1^2 + w2^2)^(3/2),
  {w0, 0, 1, 2}, {w1, -1, 0, 1}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];

Print["  dPsi_0 Laplacian = ", NumberForm[dPsi0face, 15]];
Print["  dPsi_0 direct    = ", NumberForm[dPsi0faceDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dPsi0face - dPsi0faceDir], 3]];
Print[];
Print["  dX_0 Laplacian   = ", NumberForm[dX0face, 15]];
Print["  dX_0 direct      = ", NumberForm[dX0faceDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dX0face - dX0faceDir], 3]];
Print["  (Face time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];

(* --- Edge R=(1,1,0) --- *)
Print["--- Edge direct validation ---"];
t0 = AbsoluteTime[];

dPsi0edgeDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2] Sqrt[w0^2 + w1^2 + w2^2],
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
dX0edgeDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2] (w0^2 + w1^2 + w2^2)^(3/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];

Print["  dPsi_0 Laplacian = ", NumberForm[dPsi0edge, 15]];
Print["  dPsi_0 direct    = ", NumberForm[dPsi0edgeDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dPsi0edge - dPsi0edgeDir], 3]];
Print[];
Print["  dX_0 Laplacian   = ", NumberForm[dX0edge, 15]];
Print["  dX_0 direct      = ", NumberForm[dX0edgeDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dX0edge - dX0edgeDir], 3]];
Print["  (Edge time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];

(* --- Corner R=(1,1,1) --- *)
Print["--- Corner direct validation ---"];
t0 = AbsoluteTime[];

dPsi0cornerDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2 - 1] Sqrt[w0^2 + w1^2 + w2^2],
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
dX0cornerDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2 - 1] (w0^2 + w1^2 + w2^2)^(3/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];

Print["  dPsi_0 Laplacian = ", NumberForm[dPsi0corner, 15]];
Print["  dPsi_0 direct    = ", NumberForm[dPsi0cornerDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dPsi0corner - dPsi0cornerDir], 3]];
Print[];
Print["  dX_0 Laplacian   = ", NumberForm[dX0corner, 15]];
Print["  dX_0 direct      = ", NumberForm[dX0cornerDir, 15]];
Print["  |diff|           = ", ScientificForm[Abs[dX0corner - dX0cornerDir], 3]];
Print["  (Corner time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ================================================================ *)
(* Section 5: EXPORT VALUES                                          *)
(* ================================================================ *)

Print["================================================================"];
Print["  EXPORTING VALUES"];
Print["================================================================"];
Print[];

exportFile = FileNameJoin[{DirectoryName[$InputFileName],
  "InterVoxelPropagatorCHDynamicValues.wl"}];

exportLines = {
  "(* InterVoxelPropagatorCHDynamicValues.wl",
  "   Auto-generated by InterVoxelPropagatorCHDynamic.wl",
  "   d^3 X/dR^3  (triharmonic third derivatives, kernel rho^3)",
  "   d^3 Omega/dR^3  (pentaharmonic third derivatives, kernel rho^5)",
  "   Plus Laplacian-derived first derivatives dPsi/dR, dX/dR.",
  "   Raw values WITHOUT normalization factors.",
  "   Date: " <> DateString[],
  "*)",
  "",
  "(* ===== d^3 X / dR^3  (triharmonic, for omega^2 correction) ===== *)",
  "",
  "(* FACE *)",
  "D3XfaceD000 = " <> ToString[N[D000faceX, 20], InputForm] <> ";",
  "D3XfaceD011 = " <> ToString[N[D011faceX, 20], InputForm] <> ";",
  "",
  "(* EDGE *)",
  "D3XedgeD000 = " <> ToString[N[D000edgeX, 20], InputForm] <> ";",
  "D3XedgeD001 = " <> ToString[N[D001edgeX, 20], InputForm] <> ";",
  "D3XedgeD022 = " <> ToString[N[D022edgeX, 20], InputForm] <> ";",
  "",
  "(* CORNER *)",
  "D3XcornerD000 = " <> ToString[N[D000cornerX, 20], InputForm] <> ";",
  "D3XcornerD001 = " <> ToString[N[D001cornerX, 20], InputForm] <> ";",
  "D3XcornerD012 = " <> ToString[N[D012cornerX, 20], InputForm] <> ";",
  "",
  "(* ===== d^3 Omega / dR^3  (pentaharmonic, for omega^4 correction) ===== *)",
  "",
  "(* FACE *)",
  "D3OmfaceD000 = " <> ToString[N[D000faceOm, 20], InputForm] <> ";",
  "D3OmfaceD011 = " <> ToString[N[D011faceOm, 20], InputForm] <> ";",
  "",
  "(* EDGE *)",
  "D3OmedgeD000 = " <> ToString[N[D000edgeOm, 20], InputForm] <> ";",
  "D3OmedgeD001 = " <> ToString[N[D001edgeOm, 20], InputForm] <> ";",
  "D3OmedgeD022 = " <> ToString[N[D022edgeOm, 20], InputForm] <> ";",
  "",
  "(* CORNER *)",
  "D3OmcornerD000 = " <> ToString[N[D000cornerOm, 20], InputForm] <> ";",
  "D3OmcornerD001 = " <> ToString[N[D001cornerOm, 20], InputForm] <> ";",
  "D3OmcornerD012 = " <> ToString[N[D012cornerOm, 20], InputForm] <> ";",
  "",
  "(* ===== Laplacian-derived first derivatives ===== *)",
  "",
  "(* dPsi/dR_0 = (1/12) * Laplacian trace of d^3X/dR^3 *)",
  "dPsi0face = " <> ToString[N[dPsi0face, 20], InputForm] <> ";",
  "dPsi0edge = " <> ToString[N[dPsi0edge, 20], InputForm] <> ";",
  "dPsi0corner = " <> ToString[N[dPsi0corner, 20], InputForm] <> ";",
  "",
  "(* dX/dR_0 = (1/30) * Laplacian trace of d^3Omega/dR^3 *)",
  "dXd0face = " <> ToString[N[dX0face, 20], InputForm] <> ";",
  "dXd0edge = " <> ToString[N[dX0edge, 20], InputForm] <> ";",
  "dXd0corner = " <> ToString[N[dX0corner, 20], InputForm] <> ";"
};

Export[exportFile, StringJoin[Riffle[exportLines, "\n"]], "Text"];
Print["  Wrote: ", exportFile];
Print[];


(* ================================================================ *)
(* Section 6: PYTHON-READY CONSTANTS                                 *)
(* ================================================================ *)

Print["================================================================"];
Print["  PYTHON CONSTANTS (copy-paste ready)"];
Print["================================================================"];
Print[];

Print["# d^3 X / dR_i dR_j dR_k  (triharmonic third derivatives, omega^2)"];
Print["FACE_D3X_000 = ", CForm[N[D000faceX, 18]]];
Print["FACE_D3X_011 = ", CForm[N[D011faceX, 18]]];
Print[];
Print["EDGE_D3X_000 = ", CForm[N[D000edgeX, 18]]];
Print["EDGE_D3X_001 = ", CForm[N[D001edgeX, 18]]];
Print["EDGE_D3X_022 = ", CForm[N[D022edgeX, 18]]];
Print[];
Print["CORNER_D3X_000 = ", CForm[N[D000cornerX, 18]]];
Print["CORNER_D3X_001 = ", CForm[N[D001cornerX, 18]]];
Print["CORNER_D3X_012 = ", CForm[N[D012cornerX, 18]]];
Print[];

Print["# d^3 Omega / dR_i dR_j dR_k  (pentaharmonic third derivatives, omega^4)"];
Print["FACE_D3OM_000 = ", CForm[N[D000faceOm, 18]]];
Print["FACE_D3OM_011 = ", CForm[N[D011faceOm, 18]]];
Print[];
Print["EDGE_D3OM_000 = ", CForm[N[D000edgeOm, 18]]];
Print["EDGE_D3OM_001 = ", CForm[N[D001edgeOm, 18]]];
Print["EDGE_D3OM_022 = ", CForm[N[D022edgeOm, 18]]];
Print[];
Print["CORNER_D3OM_000 = ", CForm[N[D000cornerOm, 18]]];
Print["CORNER_D3OM_001 = ", CForm[N[D001cornerOm, 18]]];
Print["CORNER_D3OM_012 = ", CForm[N[D012cornerOm, 18]]];
Print[];

Print["# dPsi/dR_k  (biharmonic first derivatives, from Laplacian of d^3X)"];
Print["FACE_DPSI_0 = ", CForm[N[dPsi0face, 18]]];
Print["EDGE_DPSI_0 = ", CForm[N[dPsi0edge, 18]]];
Print["CORNER_DPSI_0 = ", CForm[N[dPsi0corner, 18]]];
Print[];

Print["# dX/dR_k  (triharmonic first derivatives, from Laplacian of d^3Omega)"];
Print["FACE_DX_0 = ", CForm[N[dX0face, 18]]];
Print["EDGE_DX_0 = ", CForm[N[dX0edge, 18]]];
Print["CORNER_DX_0 = ", CForm[N[dX0corner, 18]]];
Print[];


Print["================================================================"];
Print["  TOTAL TIME: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
