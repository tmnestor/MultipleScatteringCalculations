(* ::Package:: *)
(* InterVoxelPropagatorOmega6.wl
   Computes all omega^6 extension values for the inter-voxel propagator:

   Part A: d^2 H / dR^2  (heptaharmonic Hessian, rho^7 kernel)
           -> 7 values for G block order 3
   Part B: d^3 H / dR^3  (heptaharmonic third derivatives, rho^7 kernel)
           -> 8 values for C/H block order 3
   Part C: dOmega/dR_k   (pentaharmonic first derivative, derived from Laplacian)
           -> 3 values for C/H block order 3

   All integrals use NIntegrate with WP=25, PG=18.
   The rho^7 kernel is smooth (no singularity), so NIntegrate gives 15+ digits.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
       -file Mathematica/InterVoxelPropagatorOmega6.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  OMEGA-6 VALUES: d^2H/dR^2 + d^3H/dR^3 + dOmega/dR"];
Print["  For G and C/H block omega^6 corrections"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];

(* ---- Utility functions ---- *)
tent[x_] := Max[0, 1 - Abs[x]];
stepFn[x_] := Piecewise[{{+1, 0 < x < 1}, {-1, 1 < x < 2}}, 0];

tentWeights = {+1, -2, +1};
wpNI = 25;
pgNI = 18;


(* ================================================================ *)
(* PART A: d^2 H / dR^2  (Heptaharmonic Hessian)                   *)
(*                                                                   *)
(* G block omega^6 correction:                                       *)
(*   G^(3) = coeff_3 * [delta * Omega_val - (eta_3/56) * d^2H/dR^2]*)
(*                                                                   *)
(* Diagonal entries via delta-function collapse -> 2D NIntegrate    *)
(* Off-diagonal entries via step x step -> 3D NIntegrate            *)
(* ================================================================ *)

Print["==== PART A: d^2H/dR^2 (Heptaharmonic Hessian) ===="];
Print[];


(* ---- A1. FACE R=(1,0,0), C_{4v} symmetry ---- *)
(* A_{11} != A_{22}=A_{33}, A_{12}=0 *)

Print["--- Face R=(1,0,0) ---"];
t0 = AbsoluteTime[];

(* A_{11}: delta collapse on w0 -> 2D tent on (w1,w2) *)
(* tent centered at 0: knots at {0,1,2}, factor 4 from quadrants *)
I2HeptaTentNI[c_?NumericQ] := NIntegrate[
  (1 - u)(1 - v)(c^2 + u^2 + v^2)^(7/2),
  {u, 0, 1}, {v, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

dW5A11face = 4 * Sum[tentWeights[[ic]] * I2HeptaTentNI[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW5_A11 = ", NumberForm[dW5A11face, 18]];

(* A_{22}: delta collapse on w1 -> shifted tent(w0-1) x tent(w2) *)
faceShiftedTentHepta[c_?NumericQ] := NIntegrate[
  tent[w0 - 1] tent[w2] (w0^2 + c^2 + w2^2)^(7/2),
  {w0, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

dW5A22face = 2 faceShiftedTentHepta[1] - 2 faceShiftedTentHepta[0];
Print["  dW5_A22 = dW5_A33 = ", NumberForm[dW5A22face, 18]];
Print["  dW5_A12 = 0 (by C_{4v} symmetry)"];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ---- A2. EDGE R=(1,1,0), C_{2v} symmetry ---- *)
(* A_{11}=A_{22}, A_{33}, A_{12} *)

Print["--- Edge R=(1,1,0) ---"];
t0 = AbsoluteTime[];

(* A_{11}: delta collapse on w0 -> shifted tent(w1-1) x tent(w2) *)
edgeShiftedTentHepta[c_?NumericQ] := NIntegrate[
  tent[w1 - 1] tent[w2] (c^2 + w1^2 + w2^2)^(7/2),
  {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

dW5A11edge = Sum[tentWeights[[ic]] * edgeShiftedTentHepta[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW5_A11 = dW5_A22 = ", NumberForm[dW5A11edge, 18]];

(* A_{33}: delta collapse on w2 -> double-shifted tent(w0-1) x tent(w1-1) *)
edgeDblTentHepta[c_?NumericQ] := NIntegrate[
  tent[w0 - 1] tent[w1 - 1] (w0^2 + w1^2 + c^2)^(7/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

dW5A33edge = 2 edgeDblTentHepta[1] - 2 edgeDblTentHepta[0];
Print["  dW5_A33 = ", NumberForm[dW5A33edge, 18]];

(* A_{12}: step(w0) x step(w1) x tent(w2) -- smooth 3D integral *)
dW5A12edge = NIntegrate[
  stepFn[w0] stepFn[w1] tent[w2] (w0^2 + w1^2 + w2^2)^(7/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16, MaxRecursion -> 20
];
Print["  dW5_A12 = ", NumberForm[dW5A12edge, 18]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ---- A3. CORNER R=(1,1,1), S_3 symmetry ---- *)
(* A_{11}=A_{22}=A_{33}, A_{12}=A_{13}=A_{23} *)

Print["--- Corner R=(1,1,1) ---"];
t0 = AbsoluteTime[];

(* A_{11}: delta collapse on w0 -> double-shifted tent(w1-1) x tent(w2-1) *)
cornerDblTentHepta[c_?NumericQ] := NIntegrate[
  tent[w1 - 1] tent[w2 - 1] (c^2 + w1^2 + w2^2)^(7/2),
  {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

dW5A11corner = Re[Sum[tentWeights[[ic]] * cornerDblTentHepta[{0, 1, 2}[[ic]]], {ic, 1, 3}]];
Print["  dW5_A11 = dW5_A22 = dW5_A33 = ", NumberForm[dW5A11corner, 18]];

(* A_{12}: step(w0) x step(w1) x tent(w2-1) -- smooth 3D integral *)
dW5A12corner = Re[NIntegrate[
  stepFn[w0] stepFn[w1] tent[w2 - 1] (w0^2 + w1^2 + w2^2)^(7/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16, MaxRecursion -> 20
]];
Print["  dW5_A12 = dW5_A13 = dW5_A23 = ", NumberForm[dW5A12corner, 18]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ---- A4. Laplacian Trace Validation ---- *)
(* Tr(d^2H/dR^2) = 56 * Omega00(R) *)
(* where Omega00 = tent-weighted rho^5 potential *)

Print["--- Laplacian Trace Validation: Tr(d^2H) = 56 * Omega00 ---"];

tentPW[t_] := Piecewise[{{1 - t, 0 <= t <= 1}, {1 + t, -1 <= t < 0}}, 0];

Omega00[{Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
  NIntegrate[
    tentPW[w1 - Rx] tentPW[w2 - Ry] tentPW[w3 - Rz] *
      (w1^2 + w2^2 + w3^2)^(5/2),
    {w1, Rx - 1, Rx, Rx + 1},
    {w2, Ry - 1, Ry, Ry + 1},
    {w3, Rz - 1, Rz, Rz + 1},
    WorkingPrecision -> 25, PrecisionGoal -> 14, MaxRecursion -> 15
  ];

om00face = Omega00[{1, 0, 0}];
trFace = dW5A11face + 2 dW5A22face;
Print["  Face: Tr = ", NumberForm[trFace, 16], "   56*Om00 = ", NumberForm[56 om00face, 16],
      "   |D| = ", ScientificForm[Abs[trFace - 56 om00face], 3]];

om00edge = Omega00[{1, 1, 0}];
trEdge = 2 dW5A11edge + dW5A33edge;
Print["  Edge: Tr = ", NumberForm[trEdge, 16], "   56*Om00 = ", NumberForm[56 om00edge, 16],
      "   |D| = ", ScientificForm[Abs[trEdge - 56 om00edge], 3]];

om00corner = Omega00[{1, 1, 1}];
trCorner = 3 dW5A11corner;
Print["  Corner: Tr = ", NumberForm[trCorner, 16], "   56*Om00 = ", NumberForm[56 om00corner, 16],
      "   |D| = ", ScientificForm[Abs[trCorner - 56 om00corner], 3]];
Print[];


(* ================================================================ *)
(* PART B: d^3 H / dR^3  (Heptaharmonic Third Derivatives)         *)
(*                                                                   *)
(* C/H block omega^6 correction:                                     *)
(*   dG^(3)/dR_k = coeff_3 * [delta * dOm_k - (eta_3/56) * D3H]   *)
(*                                                                   *)
(* Same delta-prime/delta/step collapse as CHDynamic.wl, now np=7.  *)
(* ================================================================ *)

Print["==== PART B: d^3H/dR^3 (Heptaharmonic Third Derivatives) ===="];
Print[];


(* ---- B1. FACE R=(1,0,0) ---- *)
(* D_000, D_011 *)

Print["--- Face d^3H/dR^3 (rho^7) ---"];
t0 = AbsoluteTime[];

(* D_000: delta-prime on w0, tent(w1) x tent(w2) *)
(* d(rho^7)/dc = 7*c*rho^5 *)
faceDpF7[c_?NumericQ] := If[c == 0, 0`25,
  7 * c * NIntegrate[
    tent[w1] tent[w2] (c^2 + w1^2 + w2^2)^(5/2),
    {w1, -1, 0, 1}, {w2, -1, 0, 1},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

D3HfaceD000 = faceDpF7[0] - 2 faceDpF7[1] + faceDpF7[2];
Print["  D_000 = ", NumberForm[D3HfaceD000, 18]];

(* D_011: step(w0) x delta(w1) x tent(w2) *)
faceSdG7[b_?NumericQ] := NIntegrate[
  (-stepFn[w0]) tent[w2] (w0^2 + b^2 + w2^2)^(7/2),
  {w0, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

D3HfaceD011 = 2 faceSdG7[1] - 2 faceSdG7[0];
Print["  D_011 = ", NumberForm[D3HfaceD011, 18]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ---- B2. EDGE R=(1,1,0) ---- *)
(* D_000, D_001, D_022 *)

Print["--- Edge d^3H/dR^3 (rho^7) ---"];
t0 = AbsoluteTime[];

(* D_000: delta-prime on w0, tent(w1-1) x tent(w2) *)
edgeDpF7[c_?NumericQ] := If[c == 0, 0`25,
  7 * c * NIntegrate[
    tent[w1 - 1] tent[w2] (c^2 + w1^2 + w2^2)^(5/2),
    {w1, 0, 1, 2}, {w2, -1, 0, 1},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

D3HedgeD000 = edgeDpF7[0] - 2 edgeDpF7[1] + edgeDpF7[2];
Print["  D_000 = ", NumberForm[D3HedgeD000, 18]];

(* D_001: delta(w0) x step(w1) x tent(w2) *)
edgeDsH7[c_?NumericQ] := NIntegrate[
  (-stepFn[w1]) tent[w2] (c^2 + w1^2 + w2^2)^(7/2),
  {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

D3HedgeD001 = edgeDsH7[0] - 2 edgeDsH7[1] + edgeDsH7[2];
Print["  D_001 = ", NumberForm[D3HedgeD001, 18]];

(* D_022: step(w0) x tent(w1-1) x delta(w2) *)
edgeSdK7[b_?NumericQ] := NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] (w0^2 + w1^2 + b^2)^(7/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

D3HedgeD022 = 2 edgeSdK7[1] - 2 edgeSdK7[0];
Print["  D_022 = ", NumberForm[D3HedgeD022, 18]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ---- B3. CORNER R=(1,1,1) ---- *)
(* D_000, D_001, D_012 *)

Print["--- Corner d^3H/dR^3 (rho^7) ---"];
t0 = AbsoluteTime[];

(* D_000: delta-prime on w0, tent(w1-1) x tent(w2-1) *)
cornerDpF7[c_?NumericQ] := If[c == 0, 0`25,
  7 * c * NIntegrate[
    tent[w1 - 1] tent[w2 - 1] (c^2 + w1^2 + w2^2)^(5/2),
    {w1, 0, 1, 2}, {w2, 0, 1, 2},
    WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
  ]
];

D3HcornerD000 = cornerDpF7[0] - 2 cornerDpF7[1] + cornerDpF7[2];
Print["  D_000 = ", NumberForm[D3HcornerD000, 18]];

(* D_001: delta(w0) x step(w1) x tent(w2-1) *)
cornerDsH7[c_?NumericQ] := NIntegrate[
  (-stepFn[w1]) tent[w2 - 1] (c^2 + w1^2 + w2^2)^(7/2),
  {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> pgNI
];

D3HcornerD001 = cornerDsH7[0] - 2 cornerDsH7[1] + cornerDsH7[2];
Print["  D_001 = ", NumberForm[D3HcornerD001, 18]];

(* D_012: step x step x step (3D NIntegrate) *)
D3HcornerD012 = Re[-NIntegrate[
  stepFn[w0] stepFn[w1] stepFn[w2] (w0^2 + w1^2 + w2^2)^(7/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16, MaxRecursion -> 20
]];
Print["  D_012 = ", NumberForm[D3HcornerD012, 18]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ================================================================ *)
(* PART C: dOmega/dR_k  (Pentaharmonic First Derivative)           *)
(*                                                                   *)
(* Derived via Laplacian identity:                                   *)
(*   dOm_k = (1/56) * Sum_j d^3H/(dR_j^2 dR_k)                    *)
(*   [since nabla^2(rho^7) = 56*rho^5]                              *)
(* ================================================================ *)

Print["==== PART C: dOmega/dR_k (from Laplacian of d^3H) ===="];
Print[];

dOm0face = (D3HfaceD000 + 2 D3HfaceD011) / 56;
Print["  dOm_0(face)   = (D3H_000 + 2*D3H_011)/56 = ", NumberForm[dOm0face, 18]];

dOm0edge = (D3HedgeD000 + D3HedgeD001 + D3HedgeD022) / 56;
Print["  dOm_0(edge)   = (D3H_000 + D3H_001 + D3H_022)/56 = ", NumberForm[dOm0edge, 18]];

dOm0corner = (D3HcornerD000 + 2 D3HcornerD001) / 56;
Print["  dOm_0(corner) = (D3H_000 + 2*D3H_001)/56 = ", NumberForm[dOm0corner, 18]];
Print[];


(* ---- C1. Direct validation of dOmega via 3D NIntegrate ---- *)
Print["--- Direct validation: dOmega/dR_0 via 3D NIntegrate ---"];
Print["    (compare with Laplacian-derived values above)"];
t0 = AbsoluteTime[];

dOm0faceDir = NIntegrate[
  (-stepFn[w0]) tent[w1] tent[w2] (w0^2 + w1^2 + w2^2)^(5/2),
  {w0, 0, 1, 2}, {w1, -1, 0, 1}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
Print["  Face: Laplacian = ", NumberForm[dOm0face, 16],
      "   Direct = ", NumberForm[dOm0faceDir, 16],
      "   |D| = ", ScientificForm[Abs[dOm0face - dOm0faceDir], 3]];

dOm0edgeDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2] (w0^2 + w1^2 + w2^2)^(5/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, -1, 0, 1},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
Print["  Edge: Laplacian = ", NumberForm[dOm0edge, 16],
      "   Direct = ", NumberForm[dOm0edgeDir, 16],
      "   |D| = ", ScientificForm[Abs[dOm0edge - dOm0edgeDir], 3]];

dOm0cornerDir = NIntegrate[
  (-stepFn[w0]) tent[w1 - 1] tent[w2 - 1] (w0^2 + w1^2 + w2^2)^(5/2),
  {w0, 0, 1, 2}, {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> wpNI, PrecisionGoal -> 16
];
Print["  Corner: Laplacian = ", NumberForm[dOm0corner, 16],
      "   Direct = ", NumberForm[dOm0cornerDir, 16],
      "   |D| = ", ScientificForm[Abs[dOm0corner - dOm0cornerDir], 3]];
Print["  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* ================================================================ *)
(* EXPORT: Print Python-ready constants                              *)
(* ================================================================ *)

Print["================================================================"];
Print["  PYTHON CONSTANTS (copy-paste ready)"];
Print["================================================================"];
Print[];

Print["# === d^2 H / dR^2 (heptaharmonic Hessian, for G block omega^6) ==="];
Print["# FACE"];
Print["DYN4_FACE_A11_RAW = ", CForm[N[dW5A11face, 18]]];
Print["DYN4_FACE_A22_RAW = ", CForm[N[dW5A22face, 18]]];
Print[];
Print["# EDGE"];
Print["DYN4_EDGE_A11_RAW = ", CForm[N[dW5A11edge, 18]]];
Print["DYN4_EDGE_A33_RAW = ", CForm[N[dW5A33edge, 18]]];
Print["DYN4_EDGE_A12_RAW = ", CForm[N[dW5A12edge, 18]]];
Print[];
Print["# CORNER"];
Print["DYN4_CORNER_A11_RAW = ", CForm[N[dW5A11corner, 18]]];
Print["DYN4_CORNER_A12_RAW = ", CForm[N[dW5A12corner, 18]]];
Print[];

Print["# === d^3 H / dR^3 (heptaharmonic third derivatives, for C/H block omega^6) ==="];
Print["# FACE"];
Print["FACE_D3H_000 = ", CForm[N[D3HfaceD000, 18]]];
Print["FACE_D3H_011 = ", CForm[N[D3HfaceD011, 18]]];
Print[];
Print["# EDGE"];
Print["EDGE_D3H_000 = ", CForm[N[D3HedgeD000, 18]]];
Print["EDGE_D3H_001 = ", CForm[N[D3HedgeD001, 18]]];
Print["EDGE_D3H_022 = ", CForm[N[D3HedgeD022, 18]]];
Print[];
Print["# CORNER"];
Print["CORNER_D3H_000 = ", CForm[N[D3HcornerD000, 18]]];
Print["CORNER_D3H_001 = ", CForm[N[D3HcornerD001, 18]]];
Print["CORNER_D3H_012 = ", CForm[N[D3HcornerD012, 18]]];
Print[];

Print["# === dOmega/dR_k (pentaharmonic first derivatives) ==="];
Print["FACE_DOM_0 = ", CForm[N[dOm0face, 18]]];
Print["EDGE_DOM_0 = ", CForm[N[dOm0edge, 18]]];
Print["CORNER_DOM_0 = ", CForm[N[dOm0corner, 18]]];
Print[];


(* ================================================================ *)
(* SUMMARY                                                           *)
(* ================================================================ *)

Print["================================================================"];
Print["  SUMMARY"];
Print["================================================================"];
Print[];

Print["  d^2 H / dR^2 (7 values):"];
Print["    Face: A11=", NumberForm[dW5A11face, 16], "  A22=A33=", NumberForm[dW5A22face, 16]];
Print["    Edge: A11=A22=", NumberForm[dW5A11edge, 16], "  A33=", NumberForm[dW5A33edge, 16],
      "  A12=", NumberForm[dW5A12edge, 16]];
Print["    Corner: A11=", NumberForm[dW5A11corner, 16], "  A12=", NumberForm[dW5A12corner, 16]];
Print[];

Print["  d^3 H / dR^3 (8 values):"];
Print["    Face: D000=", NumberForm[D3HfaceD000, 16], "  D011=", NumberForm[D3HfaceD011, 16]];
Print["    Edge: D000=", NumberForm[D3HedgeD000, 16], "  D001=", NumberForm[D3HedgeD001, 16],
      "  D022=", NumberForm[D3HedgeD022, 16]];
Print["    Corner: D000=", NumberForm[D3HcornerD000, 16], "  D001=", NumberForm[D3HcornerD001, 16],
      "  D012=", NumberForm[D3HcornerD012, 16]];
Print[];

Print["  dOmega/dR (3 values):"];
Print["    Face=", NumberForm[dOm0face, 16], "  Edge=", NumberForm[dOm0edge, 16],
      "  Corner=", NumberForm[dOm0corner, 16]];
Print[];

Print["  Total: 18 values"];
Print["  Total time: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
