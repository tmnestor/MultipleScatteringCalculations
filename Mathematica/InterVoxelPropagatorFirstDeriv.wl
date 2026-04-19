(* ::Package:: *)
(* InterVoxelPropagatorFirstDeriv.wl
   Biharmonic third derivatives d^3 Psi/dR_i dR_j dR_k for the C/H blocks
   of the 9x9 inter-voxel propagator.

   8 independent values:
     Face (C_4v):  D_000, D_011
     Edge (C_2v):  D_000, D_001, D_022
     Corner (S_3): D_000, D_001, D_012

   Method: delta-function / delta-prime collapse reduces 3D -> 2D NIntegrate.
   Corner D_012 uses direct 3D NIntegrate (step x step x step).

   Validation:
     1. Laplacian identity: D_000 + 2*D_011 = 2*dPhi_0 (face), etc.
     2. Finite-difference cross-check of Psi00 and Phi00.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorFirstDeriv.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  BIHARMONIC THIRD DERIVATIVES d^3 Psi / dR^3"];
Print["  For displacement-strain coupling (C/H blocks)"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];

(* Utility functions *)
tent[x_] := Max[0, 1 - Abs[x]];
stepFn[x_] := Piecewise[{{+1, 0 < x < 1}, {-1, 1 < x < 2}}, 0];


(* ================================================================ *)
(* Section 1: FACE R = (1,0,0) — 2 independent values               *)
(*                                                                    *)
(* Tents: tent(w1-1) shifted, tent(w2) centered, tent(w3) centered  *)
(* ================================================================ *)

Print["==== Section 1: FACE R = (1,0,0) — C_{4v} ===="];
Print[];

(* --- D_000 = d^3 Psi / dR_0^3 ---
   M_00'''(w1-1) = delta'(w1) - 2 delta'(w1-1) + delta'(w1-2)
   delta'(w1-c) acts on |w| via: int delta'(w1-c) f(w1) dw1 = -f'(c)
   f'(c) = c / sqrt(c^2 + w2^2 + w3^2)

   D_000 = sum_a alpha_a * F(c_a)
   where F(c) = int tent(w2) tent(w3) * c/sqrt(c^2+w2^2+w3^2) dw2 dw3
   alpha = {+1, -2, +1} at c = {0, 1, 2}
*)

Print["--- D_000(face): delta-prime collapse ---"];

faceF[c_?NumericQ] := If[c == 0, 0,
  NIntegrate[
    tent[w2] tent[w3] c / Sqrt[c^2 + w2^2 + w3^2],
    {w2, -1, 0, 1}, {w3, -1, 0, 1},
    WorkingPrecision -> 25, PrecisionGoal -> 18
  ]
];

t0 = AbsoluteTime[];
faceF1 = faceF[1];
faceF2 = faceF[2];
Print["  F(1) = ", NumberForm[faceF1, 18]];
Print["  F(2) = ", NumberForm[faceF2, 18]];

D000face = 1*faceF[0] + (-2)*faceF1 + 1*faceF2;
Print["  D_000(face) = ", NumberForm[D000face, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* --- D_011 = d^3 Psi / (dR_0 dR_1^2) ---
   d/dR_0: -M_00'(w1-1) = -step(w1), step = +1 on (0,1), -1 on (1,2)
   d^2/dR_1^2: M_00''(w2) = delta(w2+1) - 2 delta(w2) + delta(w2-1)

   After delta collapse on w2 at b in {-1,0,1}, weights {+1,-2,+1}:
   By |b| symmetry: D_011 = 2*G(1) - 2*G(0)

   G(b) = int [-step(w1)] tent(w3) sqrt(w1^2 + b^2 + w3^2) dw1 dw3
        = int [-1 on (0,1), +1 on (1,2)] * tent(w3) * sqrt(...) dw1 dw3
*)

Print["--- D_011(face): step x delta collapse ---"];

t0 = AbsoluteTime[];

faceG[b_?NumericQ] := NIntegrate[
  (-stepFn[w1]) tent[w3] Sqrt[w1^2 + b^2 + w3^2],
  {w1, 0, 1, 2}, {w3, -1, 0, 1},
  WorkingPrecision -> 25, PrecisionGoal -> 18
];

faceG0 = faceG[0];
faceG1 = faceG[1];
Print["  G(0) = ", NumberForm[faceG0, 18]];
Print["  G(1) = ", NumberForm[faceG1, 18]];

D011face = 2*faceG1 - 2*faceG0;
Print["  D_011(face) = ", NumberForm[D011face, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* Laplacian check: D_000 + 2*D_011 = 2*dPhi_0 *)
dPhi0face = (D000face + 2*D011face) / 2;
Print["  dPhi_0(face) = (D000 + 2*D011)/2 = ", NumberForm[dPhi0face, 18]];
Print[];


(* ================================================================ *)
(* Section 2: EDGE R = (1,1,0) — 3 independent values               *)
(*                                                                    *)
(* Tents: tent(w1-1) shifted, tent(w2-1) shifted, tent(w3) centered *)
(* ================================================================ *)

Print["==== Section 2: EDGE R = (1,1,0) — C_{2v} ===="];
Print[];

(* --- D_000 = d^3 Psi / dR_0^3 ---
   Delta-prime on w1 at {0,1,2}
   Remaining: tent(w2-1) * tent(w3) * c / sqrt(c^2 + w2^2 + w3^2)
*)

Print["--- D_000(edge): delta-prime collapse ---"];

edgeF[c_?NumericQ] := If[c == 0, 0,
  NIntegrate[
    tent[w2 - 1] tent[w3] c / Sqrt[c^2 + w2^2 + w3^2],
    {w2, 0, 1, 2}, {w3, -1, 0, 1},
    WorkingPrecision -> 25, PrecisionGoal -> 18
  ]
];

t0 = AbsoluteTime[];
edgeF1 = edgeF[1];
edgeF2 = edgeF[2];
Print["  F(1) = ", NumberForm[edgeF1, 18]];
Print["  F(2) = ", NumberForm[edgeF2, 18]];

D000edge = 1*edgeF[0] + (-2)*edgeF1 + 1*edgeF2;
Print["  D_000(edge) = ", NumberForm[D000edge, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* --- D_001 = d^3 Psi / (dR_0^2 dR_1) ---
   d^2/dR_0^2: M_00''(w1-1) -> delta at c = {0,1,2}
   d/dR_1: -M_00'(w2-1) -> -step(w2)

   After delta collapse on w1:
   D_001 = sum_a alpha_a * H(c_a)
   H(c) = int [-step(w2)] tent(w3) sqrt(c^2 + w2^2 + w3^2) dw2 dw3
*)

Print["--- D_001(edge): delta x step collapse ---"];

t0 = AbsoluteTime[];

edgeH[c_?NumericQ] := NIntegrate[
  (-stepFn[w2]) tent[w3] Sqrt[c^2 + w2^2 + w3^2],
  {w2, 0, 1, 2}, {w3, -1, 0, 1},
  WorkingPrecision -> 25, PrecisionGoal -> 18
];

edgeH0 = edgeH[0];
edgeH1 = edgeH[1];
edgeH2 = edgeH[2];
Print["  H(0) = ", NumberForm[edgeH0, 18]];
Print["  H(1) = ", NumberForm[edgeH1, 18]];
Print["  H(2) = ", NumberForm[edgeH2, 18]];

D001edge = 1*edgeH0 + (-2)*edgeH1 + 1*edgeH2;
Print["  D_001(edge) = ", NumberForm[D001edge, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* --- D_022 = d^3 Psi / (dR_0 dR_2^2) ---
   d/dR_0: -M_00'(w1-1) -> -step(w1)
   d^2/dR_2^2: M_00''(w3) -> delta at b = {-1,0,1}

   After delta collapse on w3:
   D_022 = sum_b beta_b * K(b)
   K(b) = int [-step(w1)] tent(w2-1) sqrt(w1^2 + w2^2 + b^2) dw1 dw2
   beta = {+1, -2, +1} at b = {-1, 0, 1}
   By |b| symmetry: D_022 = 2*K(1) - 2*K(0)
*)

Print["--- D_022(edge): step x delta collapse ---"];

t0 = AbsoluteTime[];

edgeK[b_?NumericQ] := NIntegrate[
  (-stepFn[w1]) tent[w2 - 1] Sqrt[w1^2 + w2^2 + b^2],
  {w1, 0, 1, 2}, {w2, 0, 1, 2},
  WorkingPrecision -> 25, PrecisionGoal -> 18
];

edgeK0 = edgeK[0];
edgeK1 = edgeK[1];
Print["  K(0) = ", NumberForm[edgeK0, 18]];
Print["  K(1) = ", NumberForm[edgeK1, 18]];

D022edge = 2*edgeK1 - 2*edgeK0;
Print["  D_022(edge) = ", NumberForm[D022edge, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* Laplacian check: D_000 + D_001 + D_022 = 2*dPhi_0 *)
dPhi0edge = (D000edge + D001edge + D022edge) / 2;
Print["  dPhi_0(edge) = (D000 + D001 + D022)/2 = ", NumberForm[dPhi0edge, 18]];
Print[];


(* ================================================================ *)
(* Section 3: CORNER R = (1,1,1) — 3 independent values             *)
(*                                                                    *)
(* Tents: tent(w1-1), tent(w2-1), tent(w3-1) all shifted            *)
(* ================================================================ *)

Print["==== Section 3: CORNER R = (1,1,1) — S_3 ===="];
Print[];

(* --- D_000 = d^3 Psi / dR_0^3 ---
   Delta-prime on w1 at {0,1,2}
   Remaining: tent(w2-1) * tent(w3-1) * c / sqrt(c^2 + w2^2 + w3^2)
*)

Print["--- D_000(corner): delta-prime collapse ---"];

cornerF[c_?NumericQ] := If[c == 0, 0,
  NIntegrate[
    tent[w2 - 1] tent[w3 - 1] c / Sqrt[c^2 + w2^2 + w3^2],
    {w2, 0, 1, 2}, {w3, 0, 1, 2},
    WorkingPrecision -> 25, PrecisionGoal -> 18
  ]
];

t0 = AbsoluteTime[];
cornerF1 = cornerF[1];
cornerF2 = cornerF[2];
Print["  F(1) = ", NumberForm[cornerF1, 18]];
Print["  F(2) = ", NumberForm[cornerF2, 18]];

D000corner = 1*cornerF[0] + (-2)*cornerF1 + 1*cornerF2;
Print["  D_000(corner) = ", NumberForm[D000corner, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* --- D_001 = d^3 Psi / (dR_0^2 dR_1) ---
   d^2/dR_0^2: delta on w1 at {0,1,2}
   d/dR_1: -step on w2

   H(c) = int [-step(w2)] tent(w3-1) sqrt(c^2 + w2^2 + w3^2) dw2 dw3
*)

Print["--- D_001(corner): delta x step collapse ---"];

t0 = AbsoluteTime[];

cornerH[c_?NumericQ] := NIntegrate[
  (-stepFn[w2]) tent[w3 - 1] Sqrt[c^2 + w2^2 + w3^2],
  {w2, 0, 1, 2}, {w3, 0, 1, 2},
  WorkingPrecision -> 25, PrecisionGoal -> 18
];

cornerH0 = cornerH[0];
cornerH1 = cornerH[1];
cornerH2 = cornerH[2];
Print["  H(0) = ", NumberForm[cornerH0, 18]];
Print["  H(1) = ", NumberForm[cornerH1, 18]];
Print["  H(2) = ", NumberForm[cornerH2, 18]];

D001corner = 1*cornerH0 + (-2)*cornerH1 + 1*cornerH2;
Print["  D_001(corner) = ", NumberForm[D001corner, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* --- D_012 = d^3 Psi / (dR_0 dR_1 dR_2) ---
   All three derivatives are step functions (no delta collapse).
   d/dR_0: -step(w1), d/dR_1: -step(w2), d/dR_2: -step(w3)
   Overall sign: (-1)^3 = -1

   D_012 = -int step(w1) step(w2) step(w3) |w| dw
*)

Print["--- D_012(corner): step x step x step (3D NIntegrate) ---"];

t0 = AbsoluteTime[];

D012corner = -NIntegrate[
  stepFn[w1] stepFn[w2] stepFn[w3] Sqrt[w1^2 + w2^2 + w3^2],
  {w1, 0, 1, 2}, {w2, 0, 1, 2}, {w3, 0, 1, 2},
  WorkingPrecision -> 25, PrecisionGoal -> 16, MaxRecursion -> 20
];

Print["  D_012(corner) = ", NumberForm[D012corner, 18],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];


(* Laplacian check: D_000 + 2*D_001 = 2*dPhi_0 *)
dPhi0corner = (D000corner + 2*D001corner) / 2;
Print["  dPhi_0(corner) = (D000 + 2*D001)/2 = ", NumberForm[dPhi0corner, 18]];
Print[];


(* ================================================================ *)
(* Section 4: FINITE-DIFFERENCE VALIDATION                          *)
(*                                                                    *)
(* Cross-check: compute Psi00(R) and Phi00(R) by 3D NIntegrate,    *)
(* then FD to get d^3 Psi/dR^3 and dPhi/dR.                        *)
(* ================================================================ *)

Print["==== Section 4: Finite-Difference Validation ===="];
Print[];

(* Biharmonic potential *)
Psi00[R_List] := NIntegrate[
  tent[w1 - R[[1]]] tent[w2 - R[[2]]] tent[w3 - R[[3]]]
    Sqrt[w1^2 + w2^2 + w3^2],
  {w1, R[[1]] - 1, R[[1]] + 1},
  {w2, R[[2]] - 1, R[[2]] + 1},
  {w3, R[[3]] - 1, R[[3]] + 1},
  WorkingPrecision -> 30, PrecisionGoal -> 22, MaxRecursion -> 20
];

(* Newton potential *)
Phi00[R_List] := NIntegrate[
  tent[w1 - R[[1]]] tent[w2 - R[[2]]] tent[w3 - R[[3]]]
    / Sqrt[w1^2 + w2^2 + w3^2],
  {w1, R[[1]] - 1, R[[1]] + 1},
  {w2, R[[2]] - 1, R[[2]] + 1},
  {w3, R[[3]] - 1, R[[3]] + 1},
  Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"},
  WorkingPrecision -> 30, PrecisionGoal -> 20, MaxRecursion -> 20
];

hFD = 1/10000;  (* FD step *)
e1 = {1, 0, 0}; e2 = {0, 1, 0}; e3 = {0, 0, 1};

(* --- FD formulas ---
   d^3 f/dx^3 ~ [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h^3)
   d^3 f/(dx^2 dy) ~ [f(x+h,y+h) - 2f(x,y+h) + f(x-h,y+h)
                      - f(x+h,y-h) + 2f(x,y-h) - f(x-h,y-h)] / (2h^3)
   d^3 f/(dx dy dz) ~ [f(h,h,h) - f(h,h,-h) - f(h,-h,h) + f(h,-h,-h)
                        -f(-h,h,h) + f(-h,h,-h) + f(-h,-h,h) - f(-h,-h,-h)] / (8h^3)
*)

(* === Face FD check === *)
Print["--- Face FD check at R = (1,0,0), h = ", hFD, " ---"];
R0face = {1, 0, 0};

Print["  Computing Psi00 evaluations ..."];
t0 = AbsoluteTime[];

psiF = Table[{shift, Psi00[R0face + shift]}, {shift,
  {2 hFD e1, hFD e1, -hFD e1, -2 hFD e1,
   hFD e1 + hFD e2, -hFD e1 + hFD e2,
   hFD e1 - hFD e2, -hFD e1 - hFD e2,
   hFD e2, -hFD e2}}];

(* Build lookup *)
ClearAll[psiFL];
Do[psiFL[psiF[[i, 1]]] = psiF[[i, 2]], {i, Length[psiF]}];

D000faceFD = (psiFL[2 hFD e1] - 2 psiFL[hFD e1] + 2 psiFL[-hFD e1] - psiFL[-2 hFD e1]) / (2 hFD^3);
D011faceFD = (psiFL[hFD e1 + hFD e2] - 2 psiFL[hFD e2] + psiFL[-hFD e1 + hFD e2]
             - psiFL[hFD e1 - hFD e2] + 2 psiFL[-hFD e2] - psiFL[-hFD e1 - hFD e2]) / (2 hFD^3);

Print["  D_000 analytical = ", NumberForm[D000face, 14]];
Print["  D_000 FD         = ", NumberForm[D000faceFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D000face - D000faceFD], 3]];
Print[];

Print["  D_011 analytical = ", NumberForm[D011face, 14]];
Print["  D_011 FD         = ", NumberForm[D011faceFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D011face - D011faceFD], 3]];
Print["  (Face FD time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];

(* dPhi FD check *)
Print["  Computing Phi00 for dPhi check ..."];
dPhi0faceFD = (Phi00[R0face + hFD e1] - Phi00[R0face - hFD e1]) / (2 hFD);

Print["  dPhi_0(face) Laplacian = ", NumberForm[dPhi0face, 14]];
Print["  dPhi_0(face) FD        = ", NumberForm[dPhi0faceFD, 14]];
Print["  |diff|                 = ", ScientificForm[Abs[dPhi0face - dPhi0faceFD], 3]];
Print[];


(* === Edge FD check === *)
Print["--- Edge FD check at R = (1,1,0), h = ", hFD, " ---"];
R0edge = {1, 1, 0};

Print["  Computing Psi00 evaluations ..."];
t0 = AbsoluteTime[];

psiE = Table[{shift, Psi00[R0edge + shift]}, {shift,
  {2 hFD e1, hFD e1, -hFD e1, -2 hFD e1,
   hFD e1 + hFD e2, -hFD e1 + hFD e2,
   hFD e1 - hFD e2, -hFD e1 - hFD e2,
   hFD e2, -hFD e2,
   hFD e1 + hFD e3, -hFD e1 + hFD e3,
   hFD e1 - hFD e3, -hFD e1 - hFD e3,
   hFD e3, -hFD e3}}];

ClearAll[psiEL];
Do[psiEL[psiE[[i, 1]]] = psiE[[i, 2]], {i, Length[psiE]}];

D000edgeFD = (psiEL[2 hFD e1] - 2 psiEL[hFD e1] + 2 psiEL[-hFD e1] - psiEL[-2 hFD e1]) / (2 hFD^3);
D001edgeFD = (psiEL[hFD e1 + hFD e2] - 2 psiEL[hFD e2] + psiEL[-hFD e1 + hFD e2]
             - psiEL[hFD e1 - hFD e2] + 2 psiEL[-hFD e2] - psiEL[-hFD e1 - hFD e2]) / (2 hFD^3);
D022edgeFD = (psiEL[hFD e1 + hFD e3] - 2 psiEL[hFD e3] + psiEL[-hFD e1 + hFD e3]
             - psiEL[hFD e1 - hFD e3] + 2 psiEL[-hFD e3] - psiEL[-hFD e1 - hFD e3]) / (2 hFD^3);

Print["  D_000 analytical = ", NumberForm[D000edge, 14]];
Print["  D_000 FD         = ", NumberForm[D000edgeFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D000edge - D000edgeFD], 3]];
Print[];

Print["  D_001 analytical = ", NumberForm[D001edge, 14]];
Print["  D_001 FD         = ", NumberForm[D001edgeFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D001edge - D001edgeFD], 3]];
Print[];

Print["  D_022 analytical = ", NumberForm[D022edge, 14]];
Print["  D_022 FD         = ", NumberForm[D022edgeFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D022edge - D022edgeFD], 3]];
Print["  (Edge FD time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];

(* dPhi edge FD check *)
Print["  Computing Phi00 for edge dPhi check ..."];
dPhi0edgeFD = (Phi00[R0edge + hFD e1] - Phi00[R0edge - hFD e1]) / (2 hFD);

Print["  dPhi_0(edge) Laplacian = ", NumberForm[dPhi0edge, 14]];
Print["  dPhi_0(edge) FD        = ", NumberForm[dPhi0edgeFD, 14]];
Print["  |diff|                 = ", ScientificForm[Abs[dPhi0edge - dPhi0edgeFD], 3]];
Print[];


(* === Corner FD check === *)
Print["--- Corner FD check at R = (1,1,1), h = ", hFD, " ---"];
R0corner = {1, 1, 1};

Print["  Computing Psi00 evaluations ..."];
t0 = AbsoluteTime[];

psiC = Table[{shift, Psi00[R0corner + shift]}, {shift,
  {2 hFD e1, hFD e1, -hFD e1, -2 hFD e1,
   hFD e1 + hFD e2, -hFD e1 + hFD e2,
   hFD e1 - hFD e2, -hFD e1 - hFD e2,
   hFD e2, -hFD e2,
   hFD e1 + hFD e2 + hFD e3, hFD e1 + hFD e2 - hFD e3,
   hFD e1 - hFD e2 + hFD e3, hFD e1 - hFD e2 - hFD e3,
   -hFD e1 + hFD e2 + hFD e3, -hFD e1 + hFD e2 - hFD e3,
   -hFD e1 - hFD e2 + hFD e3, -hFD e1 - hFD e2 - hFD e3}}];

ClearAll[psiCL];
Do[psiCL[psiC[[i, 1]]] = psiC[[i, 2]], {i, Length[psiC]}];

D000cornerFD = (psiCL[2 hFD e1] - 2 psiCL[hFD e1] + 2 psiCL[-hFD e1] - psiCL[-2 hFD e1]) / (2 hFD^3);
D001cornerFD = (psiCL[hFD e1 + hFD e2] - 2 psiCL[hFD e2] + psiCL[-hFD e1 + hFD e2]
               - psiCL[hFD e1 - hFD e2] + 2 psiCL[-hFD e2] - psiCL[-hFD e1 - hFD e2]) / (2 hFD^3);
D012cornerFD = (psiCL[hFD e1 + hFD e2 + hFD e3] - psiCL[hFD e1 + hFD e2 - hFD e3]
               - psiCL[hFD e1 - hFD e2 + hFD e3] + psiCL[hFD e1 - hFD e2 - hFD e3]
               - psiCL[-hFD e1 + hFD e2 + hFD e3] + psiCL[-hFD e1 + hFD e2 - hFD e3]
               + psiCL[-hFD e1 - hFD e2 + hFD e3] - psiCL[-hFD e1 - hFD e2 - hFD e3]) / (8 hFD^3);

Print["  D_000 analytical = ", NumberForm[D000corner, 14]];
Print["  D_000 FD         = ", NumberForm[D000cornerFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D000corner - D000cornerFD], 3]];
Print[];

Print["  D_001 analytical = ", NumberForm[D001corner, 14]];
Print["  D_001 FD         = ", NumberForm[D001cornerFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D001corner - D001cornerFD], 3]];
Print[];

Print["  D_012 NIntegrate = ", NumberForm[D012corner, 14]];
Print["  D_012 FD         = ", NumberForm[D012cornerFD, 14]];
Print["  |diff|           = ", ScientificForm[Abs[D012corner - D012cornerFD], 3]];
Print["  (Corner FD time: ", Round[AbsoluteTime[] - t0, 0.1], " s)"];
Print[];

(* dPhi corner FD check *)
Print["  Computing Phi00 for corner dPhi check ..."];
dPhi0cornerFD = (Phi00[R0corner + hFD e1] - Phi00[R0corner - hFD e1]) / (2 hFD);

Print["  dPhi_0(corner) Laplacian = ", NumberForm[dPhi0corner, 14]];
Print["  dPhi_0(corner) FD        = ", NumberForm[dPhi0cornerFD, 14]];
Print["  |diff|                   = ", ScientificForm[Abs[dPhi0corner - dPhi0cornerFD], 3]];
Print[];


(* ================================================================ *)
(* Section 5: SUMMARY — Python-ready constants                      *)
(* ================================================================ *)

Print["================================================================"];
Print["  PYTHON CONSTANTS (copy-paste ready)"];
Print["================================================================"];
Print[];

Print["# d^3 Psi / dR_i dR_j dR_k  (biharmonic third derivatives)"];
Print["FACE_D3PSI_000 = ", CForm[N[D000face, 18]]];
Print["FACE_D3PSI_011 = ", CForm[N[D011face, 18]]];
Print[];
Print["EDGE_D3PSI_000 = ", CForm[N[D000edge, 18]]];
Print["EDGE_D3PSI_001 = ", CForm[N[D001edge, 18]]];
Print["EDGE_D3PSI_022 = ", CForm[N[D022edge, 18]]];
Print[];
Print["CORNER_D3PSI_000 = ", CForm[N[D000corner, 18]]];
Print["CORNER_D3PSI_001 = ", CForm[N[D001corner, 18]]];
Print["CORNER_D3PSI_012 = ", CForm[N[D012corner, 18]]];
Print[];
Print["# dPhi/dR_k  (Newton first derivatives, from Laplacian identity)"];
Print["FACE_DPHI_0 = ", CForm[N[dPhi0face, 18]]];
Print["EDGE_DPHI_0 = ", CForm[N[dPhi0edge, 18]]];
Print["CORNER_DPHI_0 = ", CForm[N[dPhi0corner, 18]]];
Print[];

Print["================================================================"];
Print["  TOTAL TIME: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
