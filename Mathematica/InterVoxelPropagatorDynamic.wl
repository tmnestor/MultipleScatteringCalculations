(* ::Package:: *)
(* InterVoxelPropagatorDynamic.wl
   Higher-order potential derivatives for the dynamic inter-voxel propagator.

   Phase 1 (COMPLETE) computed static P^(0) using Newton potential Phi (1/rho)
   and biharmonic Psi (rho).  This script extends to:
     - Triharmonic X (rho^3) for P^(1) at O(omega^2)
     - Pentaharmonic Omega (rho^5) for P^(2) at O(omega^4)

   Key insight: delta'' evaluates d^2(kernel)/dc^2, NOT d^4.  So:
     d^2(rho)/dc^2   = 1/rho - c^2/rho^3          [static B]
     d^2(rho^3)/dc^2 = 3*rho + 3*c^2/rho           [dynamic B^(1)]
     d^2(rho^5)/dc^2 = 5*rho^3 + 15*c^2*rho         [dynamic B^(2)]
   All reduce to existing I2A masters at higher degree shifts.

   Outputs raw derivatives (without 1/(4pi), 1/(8pi) normalization).
   Phase 2B will apply elastic tensor factors c1, c2, eta'.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorDynamic.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  INTER-VOXEL PROPAGATOR — DYNAMIC CORRECTIONS"];
Print["  Higher potentials: Psi (rho), X (rho^3), Omega (rho^5)"];
Print["  Face + Edge + Corner, Orders 1 and 2"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];


(* ================================================================ *)
(* Section 0. Parameters                                             *)
(* ================================================================ *)

mu = 1;
nu = 1/4;
eta = 1/(2(1 - nu));
a = 1;
h = a/2;
tentWeights = {+1, -2, +1};

Print["Parameters: mu=", mu, ", nu=", nu, ", eta=", eta // N, ", a=", a];
Print[];


(* ================================================================ *)
(* Section 1. 2D Master Integral Definitions                         *)
(*                                                                   *)
(* All definitions memoized.  Canonicalization I2A[p,q,c] with p>=q. *)
(* ================================================================ *)

Print["==== Section 1: 2D Master Integral Definitions ===="];
Print[];

(* --- A-channel: 1/rho kernel, unshifted --- *)
ClearAll[I2A];
I2A[p_Integer, q_Integer, c_] /; p < q := I2A[q, p, c];
I2A[p_Integer, q_Integer, c_] :=
  I2A[p, q, c] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q / Sqrt[c^2 + u^2 + v^2],
           {v, 0, 1}, Assumptions -> u > 0 && c >= 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2A[", p, ",", q, ",", c, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- B-channel: c^2/rho^3 kernel, unshifted --- *)
ClearAll[I2B];
I2B[p_Integer, q_Integer, c_] /; p < q := I2B[q, p, c];
I2B[p_Integer, q_Integer, 0] := 0;
I2B[p_Integer, q_Integer, c_] :=
  I2B[p, q, c] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q * c^2 / (c^2 + u^2 + v^2)^(3/2),
           {v, 0, 1}, Assumptions -> u > 0 && c > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2B[", p, ",", q, ",", c, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- Single-shifted A-channel: shift on u --- *)
ClearAll[I2Ashifted];
I2Ashifted[p_Integer, q_Integer, c_, shift_] :=
  I2Ashifted[p, q, c, shift] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q / Sqrt[c^2 + (u + shift)^2 + v^2],
           {v, 0, 1}, Assumptions -> u > 0 && c >= 0 && shift > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Ashifted[", p, ",", q, ",", c, ",", shift, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- Single-shifted B-channel --- *)
ClearAll[I2Bshifted];
I2Bshifted[p_Integer, q_Integer, c_, shift_] :=
  I2Bshifted[p, q, c, shift] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q * c^2 / (c^2 + (u + shift)^2 + v^2)^(3/2),
           {v, 0, 1}, Assumptions -> u > 0 && c > 0 && shift > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Bshifted[", p, ",", q, ",", c, ",", shift, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- Double-shifted A-channel --- *)
ClearAll[I2Adshift];
I2Adshift[p_Integer, q_Integer, c_, s1_, s2_] :=
  I2Adshift[p, q, c, s1, s2] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q / Sqrt[c^2 + (u + s1)^2 + (v + s2)^2],
           {v, 0, 1}, Assumptions -> u > 0 && c >= 0 && s1 > 0 && s2 > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Adshift[", p, ",", q, ",", c, ",", s1, ",", s2, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- Double-shifted B-channel --- *)
ClearAll[I2Bdshift];
I2Bdshift[p_Integer, q_Integer, 0, s1_, s2_] := 0;
I2Bdshift[p_Integer, q_Integer, c_, s1_, s2_] :=
  I2Bdshift[p, q, c, s1, s2] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q * c^2 / (c^2 + (u + s1)^2 + (v + s2)^2)^(3/2),
           {v, 0, 1}, Assumptions -> u > 0 && c > 0 && s1 > 0 && s2 > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, 1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Bdshift[", p, ",", q, ",", c, ",", s1, ",", s2, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];


(* ================================================================ *)
(* Section 1b. Higher-Order 2D Integral Wrappers                     *)
(*                                                                   *)
(* Purely algebraic — expand to I2A at higher degree shifts.         *)
(* Identity: rho = (c^2+u^2+v^2)/rho, so                           *)
(*   I2Psi[p,q,c] = c^2 I2A[p,q,c] + I2A[p+2,q,c] + I2A[p,q+2,c] *)
(*   I2X[p,q,c] = c^2 I2Psi[p,q,c] + I2Psi[p+2,q,c] + I2Psi[p,q+2,c] *)
(* ================================================================ *)

(* --- Unshifted: Psi (rho), X (rho^3) --- *)
ClearAll[I2Psi, I2X];
I2Psi[p_Integer, q_Integer, c_] :=
  c^2 * I2A[p, q, c] + I2A[p + 2, q, c] + I2A[p, q + 2, c];
I2X[p_Integer, q_Integer, c_] :=
  c^2 * I2Psi[p, q, c] + I2Psi[p + 2, q, c] + I2Psi[p, q + 2, c];

(* --- Single-shifted: rho^2 = (c^2+s^2) + u^2 + 2su + v^2 --- *)
ClearAll[I2PsiS, I2XS];
I2PsiS[p_Integer, q_Integer, c_, s_] :=
  (c^2 + s^2) * I2Ashifted[p, q, c, s] +
  I2Ashifted[p + 2, q, c, s] + 2 s * I2Ashifted[p + 1, q, c, s] +
  I2Ashifted[p, q + 2, c, s];
I2XS[p_Integer, q_Integer, c_, s_] :=
  (c^2 + s^2) * I2PsiS[p, q, c, s] +
  I2PsiS[p + 2, q, c, s] + 2 s * I2PsiS[p + 1, q, c, s] +
  I2PsiS[p, q + 2, c, s];

(* --- Double-shifted: rho^2 = (c^2+s1^2+s2^2) + u^2 + 2s1*u + v^2 + 2s2*v --- *)
ClearAll[I2PsiD, I2XD];
I2PsiD[p_Integer, q_Integer, c_, s1_, s2_] :=
  (c^2 + s1^2 + s2^2) * I2Adshift[p, q, c, s1, s2] +
  I2Adshift[p + 2, q, c, s1, s2] + 2 s1 * I2Adshift[p + 1, q, c, s1, s2] +
  I2Adshift[p, q + 2, c, s1, s2] + 2 s2 * I2Adshift[p, q + 1, c, s1, s2];
I2XD[p_Integer, q_Integer, c_, s1_, s2_] :=
  (c^2 + s1^2 + s2^2) * I2PsiD[p, q, c, s1, s2] +
  I2PsiD[p + 2, q, c, s1, s2] + 2 s1 * I2PsiD[p + 1, q, c, s1, s2] +
  I2PsiD[p, q + 2, c, s1, s2] + 2 s2 * I2PsiD[p, q + 1, c, s1, s2];


(* ================================================================ *)
(* Section 2. 1D Elementary Integrals                                *)
(*                                                                   *)
(* I1exact[C]   = int_0^1 (1-t) sqrt(C+t^2) dt    [biharmonic]     *)
(* I1cubic[C]   = int_0^1 (1-t) (C+t^2)^{3/2} dt  [triharmonic]   *)
(* I1quintic[C] = int_0^1 (1-t) (C+t^2)^{5/2} dt  [pentaharmonic] *)
(* ================================================================ *)

Print["==== Section 2: 1D Elementary Integrals ===="];
Print[];

(* --- Biharmonic (existing) --- *)
ClearAll[I1exact];
I1exact[0] = 1/6;
I1exact[c2_] := Module[{sc1 = Sqrt[c2 + 1], sc = Sqrt[c2]},
  1/2 (sc1 + c2 * ArcSinh[1/sc]) - 1/3 (sc1^3 - sc^3)
];

(* --- I1A (existing) --- *)
ClearAll[I1A];
I1A[c2_] := Module[{sc = Sqrt[c2], sc1 = Sqrt[c2 + 1]},
  ArcSinh[1/sc] - (sc1 - sc)
];

(* --- Triharmonic: int_0^1 (1-t)(C+t^2)^{3/2} dt --- *)
ClearAll[I1cubic];
I1cubic[c2_] := I1cubic[c2] = Module[{result, t0r},
  t0r = AbsoluteTime[];
  result = Integrate[(1 - t) (c2 + t^2)^(3/2), {t, 0, 1},
    Assumptions -> c2 >= 0, GenerateConditions -> False];
  result = Simplify[result];
  Print["  I1cubic[", c2, "] = ", NumberForm[N[result, 20], 16],
        "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
  result
];

(* --- Pentaharmonic: int_0^1 (1-t)(C+t^2)^{5/2} dt --- *)
ClearAll[I1quintic];
I1quintic[c2_] := I1quintic[c2] = Module[{result, t0r},
  t0r = AbsoluteTime[];
  result = Integrate[(1 - t) (c2 + t^2)^(5/2), {t, 0, 1},
    Assumptions -> c2 >= 0, GenerateConditions -> False];
  result = Simplify[result];
  Print["  I1quintic[", c2, "] = ", NumberForm[N[result, 20], 16],
        "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
  result
];

(* --- Shifted 1D tent integrals for biharmonic (existing pattern) --- *)
ClearAll[Jpiece2memo];
Jpiece2memo[bigC_] := Jpiece2memo[bigC] = Module[{result},
  result = Integrate[(1 - s) Sqrt[bigC + (s + 1)^2], {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  Simplify[result]
];
Jtent1Dshifted[bigC_] :=
  (1/3) ((bigC + 1)^(3/2) - bigC^(3/2)) + Jpiece2memo[bigC];

(* --- Shifted 1D tent for triharmonic --- *)
ClearAll[Jpiece2cubic];
Jpiece2cubic[bigC_] := Jpiece2cubic[bigC] = Module[{result, t0r},
  t0r = AbsoluteTime[];
  result = Integrate[(1 - s) (bigC + (s + 1)^2)^(3/2), {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  result = Simplify[result];
  Print["  Jpiece2cubic[", bigC, "] = ", NumberForm[N[result, 20], 16],
        "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
  result
];
Jtent1DshiftedCubic[bigC_] :=
  (1/5) ((bigC + 1)^(5/2) - bigC^(5/2)) + Jpiece2cubic[bigC];

(* --- Shifted 1D tent for pentaharmonic --- *)
ClearAll[Jpiece2quintic];
Jpiece2quintic[bigC_] := Jpiece2quintic[bigC] = Module[{result, t0r},
  t0r = AbsoluteTime[];
  result = Integrate[(1 - s) (bigC + (s + 1)^2)^(5/2), {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  result = Simplify[result];
  Print["  Jpiece2quintic[", bigC, "] = ", NumberForm[N[result, 20], 16],
        "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
  result
];
Jtent1DshiftedQuintic[bigC_] :=
  (1/7) ((bigC + 1)^(7/2) - bigC^(7/2)) + Jpiece2quintic[bigC];

(* --- Validate 1D integrals --- *)
Print["--- Validating 1D integrals ---"];
Do[
  Module[{exact, nval, err},
    exact = I1cubic[c2];
    nval = NIntegrate[(1 - t) (c2 + t^2)^(3/2), {t, 0, 1},
             WorkingPrecision -> 25, PrecisionGoal -> 16];
    err = Abs[N[exact, 25] - nval];
    Print["  I1cubic[", c2, "]   err=", ScientificForm[err, 3],
          If[err < 10^-12, "  OK", "  FAIL"]];
  ],
  {c2, {0, 1, 2, 4, 5, 8}}
];
Do[
  Module[{exact, nval, err},
    exact = I1quintic[c2];
    nval = NIntegrate[(1 - t) (c2 + t^2)^(5/2), {t, 0, 1},
             WorkingPrecision -> 25, PrecisionGoal -> 16];
    err = Abs[N[exact, 25] - nval];
    Print["  I1quintic[", c2, "] err=", ScientificForm[err, 3],
          If[err < 10^-12, "  OK", "  FAIL"]];
  ],
  {c2, {0, 1, 2, 4, 5, 8}}
];
Print[];


(* ================================================================ *)
(* Section 3. Pre-compute I2A at degrees 0-6                         *)
(*                                                                   *)
(* Degree 0-4 already needed for static; 5-6 are NEW for B^(2).    *)
(* ================================================================ *)

Print["==== Section 3: Pre-compute I2A at Higher Degrees ===="];
Print[];

Print["--- Standard I2A[p,q,c] for degrees 0-6, c in {0,1,2} ---"];
Do[I2A[p, q, c],
  {c, {0, 1, 2}},
  {p, 0, 6},
  {q, 0, Min[p, 6 - p]}
];
Print[];

Print["--- I2B[p,q,c] for degrees 0-2, c in {1,2} ---"];
Do[I2B[p, q, c], {c, {1, 2}}, {p, 0, 2}, {q, 0, p}];
Print[];

Print["--- Shifted I2Ashifted[p,q,c,1] for degrees 0-6, c in {0,1,2} ---"];
Do[I2Ashifted[p, q, c, 1],
  {c, {0, 1, 2}},
  {p, 0, 5},
  {q, 0, Min[3, 6 - p]}
];
Print[];

Print["--- Shifted I2Bshifted[p,q,c,1] for degrees 0-1, c in {1,2} ---"];
Do[I2Bshifted[p, q, c, 1], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];
Print[];

Print["--- Double-shifted I2Adshift for edge/corner ---"];
Do[I2Adshift[p, q, c, 1, 1],
  {c, {0, 1, 2}},
  {p, 0, 5},
  {q, 0, Min[3, 6 - p]}
];
Print[];

Print["--- I2Bdshift for corner ---"];
Do[I2Bdshift[p, q, c, 1, 1], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];
Print[];

Print["--- Pre-compute shifted 1D tent integrals ---"];
Do[Jpiece2memo[bigC], {bigC, {0, 1, 2, 4, 5, 8}}];
Do[Jpiece2cubic[bigC], {bigC, {0, 1, 2, 4, 5, 8}}];
Do[Jpiece2quintic[bigC], {bigC, {0, 1, 2, 4, 5, 8}}];
Print[];


(* ================================================================ *)
(* Section 4. FACE-ADJACENT DYNAMIC DERIVATIVES  R = (1,0,0)        *)
(*                                                                   *)
(* C_{4v} symmetry.  Independent:                                    *)
(*   A: A_{11}, A_{22}=A_{33}                                       *)
(*   B: B_{1111}, B_{1122}=B_{1133}, B_{2222}, B_{2233}             *)
(*                                                                   *)
(* Compute for two dynamic orders:                                   *)
(*   Order 1 (P^(1)): dW2_A from Psi (rho), dW3_B from X (rho^3)   *)
(*   Order 2 (P^(2)): dW3_A from X (rho^3), dW4_B from Omega (rho^5)*)
(* ================================================================ *)

Print["================================================================"];
Print["  FACE-ADJACENT R = (1,0,0)  — Dynamic Derivatives"];
Print["================================================================"];
Print[];

(* --- Tent-weighted 2D integrals for centered tent (1-u)(1-v) --- *)
(* kernel: I2A -> 1/rho, I2Psi -> rho, I2X -> rho^3 *)
I2Atent[c_] := I2A[0, 0, c] - 2 I2A[1, 0, c] + I2A[1, 1, c];
I2Btent[c_] := I2B[0, 0, c] - 2 I2B[1, 0, c] + I2B[1, 1, c];
I2Psitent[c_] := I2Psi[0, 0, c] - 2 I2Psi[1, 0, c] + I2Psi[1, 1, c];
I2Xtent[c_] := I2X[0, 0, c] - 2 I2X[1, 0, c] + I2X[1, 1, c];

(* --- Shifted tent integrals for A_{22} topology --- *)
(* tent(w1-1)*tent(w3): split w1 in [0,1] (tent=w1) and [1,2] (tent=1-s, shifted) *)
(* Factor 2 from w3 even symmetry fold *)
J22A[c_] := 2 * (
  (I2A[1, 0, c] - I2A[1, 1, c]) +
  (I2Ashifted[0, 0, c, 1] - I2Ashifted[1, 0, c, 1]
   - I2Ashifted[0, 1, c, 1] + I2Ashifted[1, 1, c, 1])
);

J22Psi[c_] := 2 * (
  (I2Psi[1, 0, c] - I2Psi[1, 1, c]) +
  (I2PsiS[0, 0, c, 1] - I2PsiS[1, 0, c, 1]
   - I2PsiS[0, 1, c, 1] + I2PsiS[1, 1, c, 1])
);

J22X[c_] := 2 * (
  (I2X[1, 0, c] - I2X[1, 1, c]) +
  (I2XS[0, 0, c, 1] - I2XS[1, 0, c, 1]
   - I2XS[0, 1, c, 1] + I2XS[1, 1, c, 1])
);

J22B[0] := 0;
J22B[c_] := 2 * (
  (I2B[1, 0, c] - I2B[1, 1, c]) +
  (I2Bshifted[0, 0, c, 1] - I2Bshifted[1, 0, c, 1]
   - I2Bshifted[0, 1, c, 1] + I2Bshifted[1, 1, c, 1])
);


(* ---- 4a. dW2_A: d^2 Psi / dR_j dR_l  (Psi kernel = rho) ---- *)

Print["--- 4a: dW2_A (d^2 Psi/dR^2, kernel rho) ---"];

(* A_{11}: delta on w1 at {0,1,2}, centered tent on (w2,w3) *)
dW2A11face = 4 * Sum[tentWeights[[ic]] * I2Psitent[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW2_A11 = ", NumberForm[N[dW2A11face, 16], 16]];

(* A_{22}: delta on w2 at {-1,0,1}, shifted tent(w1-1)*tent(w3) *)
(* By |c| fold: 2*J22Psi[1] - 2*J22Psi[0] *)
dW2A22face = 2 J22Psi[1] - 2 J22Psi[0];
Print["  dW2_A22 = dW2_A33 = ", NumberForm[N[dW2A22face, 16], 16]];
Print[];


(* ---- 4b. dW3_B: d^4 X / dR_i dR_j dR_k dR_l  (X kernel = rho^3) ---- *)
(* d^2(rho^3)/dc^2 = 3*rho + 3*c^2/rho *)

Print["--- 4b: dW3_B (d^4 X/dR^4, kernel rho^3) ---"];

(* B_{1111}: delta'' on w1, kernel 3*rho + 3*c^2/rho *)
dW3B1111face = 4 * Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    3 * I2Psitent[c] + 3 c^2 * I2Atent[c]
  ],
  {ic, 1, 3}
];
Print["  dW3_B1111 = ", NumberForm[N[dW3B1111face, 16], 16]];

(* B_{1122}: double delta (w1 and w2), 1D integral with rho^3 kernel *)
aVals = {0, 1, 2};   bVals = {-1, 0, 1};
dW3B1122face = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    2 * I1cubic[aVals[[ia]]^2 + bVals[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW3_B1122 = dW3_B1133 = ", NumberForm[N[dW3B1122face, 16], 16]];

(* B_{2222}: delta'' on w2, shifted tent, kernel 3*rho + 3*c^2/rho *)
(* c values {-1,0,1}, by |c|: 2*f(1) - 2*f(0) *)
(* At c=0: 3*J22Psi[0] (c^2 term vanishes) *)
(* At c=1: 3*J22Psi[1] + 3*J22A[1] *)
dW3B2222face = 2 * (3 J22Psi[1] + 3 J22A[1]) - 2 * 3 J22Psi[0];
Print["  dW3_B2222 = ", NumberForm[N[dW3B2222face, 16], 16]];

(* B_{2233}: double delta (w2 and w3), 1D shifted tent, rho^3 kernel *)
bVals23 = {-1, 0, 1};
dW3B2233face = Sum[
  tentWeights[[ib2]] * tentWeights[[ib3]] *
    Jtent1DshiftedCubic[bVals23[[ib2]]^2 + bVals23[[ib3]]^2],
  {ib2, 1, 3}, {ib3, 1, 3}
];
Print["  dW3_B2233 = ", NumberForm[N[dW3B2233face, 16], 16]];

(* Laplacian check: Sum_k dW3_B_{11kk} = 12 * dW2_A_{11} *)
lapl3face = N[dW3B1111face + 2 dW3B1122face, 16];
ref3face = N[12 dW2A11face, 16];
Print["  Laplacian: dW3_B1111 + 2*dW3_B1122 = ", NumberForm[lapl3face, 16]];
Print["             12 * dW2_A11             = ", NumberForm[ref3face, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl3face - ref3face], 3]];

lapl3face2 = N[dW3B1122face + dW3B2222face + dW3B2233face, 16];
ref3face2 = N[12 dW2A22face, 16];
Print["  Laplacian: dW3_B1122+B2222+B2233 = ", NumberForm[lapl3face2, 16]];
Print["             12 * dW2_A22           = ", NumberForm[ref3face2, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl3face2 - ref3face2], 3]];
Print[];


(* ---- 4c. dW3_A: d^2 X / dR_j dR_l  (X kernel = rho^3) ---- *)

Print["--- 4c: dW3_A (d^2 X/dR^2, kernel rho^3) ---"];

(* A_{11}: delta on w1, centered tent, rho^3 kernel *)
dW3A11face = 4 * Sum[tentWeights[[ic]] * I2Xtent[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW3_A11 = ", NumberForm[N[dW3A11face, 16], 16]];

(* A_{22}: delta on w2, shifted tent, rho^3 kernel *)
dW3A22face = 2 J22X[1] - 2 J22X[0];
Print["  dW3_A22 = dW3_A33 = ", NumberForm[N[dW3A22face, 16], 16]];
Print[];


(* ---- 4d. dW4_B: d^4 Omega / dR^4  (Omega kernel = rho^5) ---- *)
(* d^2(rho^5)/dc^2 = 5*rho^3 + 15*c^2*rho *)

Print["--- 4d: dW4_B (d^4 Omega/dR^4, kernel rho^5) ---"];

(* B_{1111}: delta'' on w1, kernel 5*rho^3 + 15*c^2*rho *)
dW4B1111face = 4 * Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    5 * I2Xtent[c] + 15 c^2 * I2Psitent[c]
  ],
  {ic, 1, 3}
];
Print["  dW4_B1111 = ", NumberForm[N[dW4B1111face, 16], 16]];

(* B_{1122}: double delta, 1D with rho^5 kernel *)
dW4B1122face = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    2 * I1quintic[aVals[[ia]]^2 + bVals[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW4_B1122 = dW4_B1133 = ", NumberForm[N[dW4B1122face, 16], 16]];

(* B_{2222}: delta'' on w2, shifted tent, kernel 5*rho^3 + 15*c^2*rho *)
dW4B2222face = 2 * (5 J22X[1] + 15 J22Psi[1]) - 2 * 5 J22X[0];
Print["  dW4_B2222 = ", NumberForm[N[dW4B2222face, 16], 16]];

(* B_{2233}: double delta (w2, w3), shifted 1D tent, rho^5 kernel *)
dW4B2233face = Sum[
  tentWeights[[ib2]] * tentWeights[[ib3]] *
    Jtent1DshiftedQuintic[bVals23[[ib2]]^2 + bVals23[[ib3]]^2],
  {ib2, 1, 3}, {ib3, 1, 3}
];
Print["  dW4_B2233 = ", NumberForm[N[dW4B2233face, 16], 16]];

(* Laplacian check: Sum_k dW4_B_{11kk} = 30 * dW3_A_{11} *)
lapl4face = N[dW4B1111face + 2 dW4B1122face, 16];
ref4face = N[30 dW3A11face, 16];
Print["  Laplacian: dW4_B1111 + 2*dW4_B1122 = ", NumberForm[lapl4face, 16]];
Print["             30 * dW3_A11             = ", NumberForm[ref4face, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl4face - ref4face], 3]];

lapl4face2 = N[dW4B1122face + dW4B2222face + dW4B2233face, 16];
ref4face2 = N[30 dW3A22face, 16];
Print["  Laplacian: dW4_B1122+B2222+B2233 = ", NumberForm[lapl4face2, 16]];
Print["             30 * dW3_A22           = ", NumberForm[ref4face2, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl4face2 - ref4face2], 3]];
Print[];


(* ================================================================ *)
(* Section 5. EDGE-ADJACENT DYNAMIC DERIVATIVES  R = (1,1,0)        *)
(*                                                                   *)
(* C_{2v} symmetry.  Independent:                                    *)
(*   A: A_{11}=A_{22}, A_{33}, A_{12} (via Laplacian)               *)
(*   B: B_{1111}=B_{2222}, B_{1122}, B_{1133}=B_{2233},            *)
(*      B_{3333}, B_{1112}=B_{1222}, B_{1233}                       *)
(* ================================================================ *)

Print["================================================================"];
Print["  EDGE-ADJACENT R = (1,1,0)  — Dynamic Derivatives"];
Print["================================================================"];
Print[];

(* --- Tent topologies for edge --- *)
(* Jshifted: tent(w-1)*tent(v) with shift=1 on u-axis *)
(* Same as face J22 but also used for A_{11} here *)
JshiftedA[c_] := J22A[c];
JshiftedPsi[c_] := J22Psi[c];
JshiftedX[c_] := J22X[c];

(* B-channel shifted tent *)
JshiftedB[0] := 0;
JshiftedB[c_] := 2 * (
  (I2B[1, 0, c] - I2B[1, 1, c]) +
  (I2Bshifted[0, 0, c, 1] - I2Bshifted[1, 0, c, 1]
   - I2Bshifted[0, 1, c, 1] + I2Bshifted[1, 1, c, 1])
);

(* Jdbl: double-shifted tent tent(w-1)*tent(v-1) *)
JdblA[c_] := (
  I2A[1, 1, c]
  + 2 * (I2Ashifted[0, 1, c, 1] - I2Ashifted[1, 1, c, 1])
  + (I2Adshift[0, 0, c, 1, 1] - I2Adshift[1, 0, c, 1, 1]
     - I2Adshift[0, 1, c, 1, 1] + I2Adshift[1, 1, c, 1, 1])
);

JdblPsi[c_] := (
  I2Psi[1, 1, c]
  + 2 * (I2PsiS[0, 1, c, 1] - I2PsiS[1, 1, c, 1])
  + (I2PsiD[0, 0, c, 1, 1] - I2PsiD[1, 0, c, 1, 1]
     - I2PsiD[0, 1, c, 1, 1] + I2PsiD[1, 1, c, 1, 1])
);

JdblX[c_] := (
  I2X[1, 1, c]
  + 2 * (I2XS[0, 1, c, 1] - I2XS[1, 1, c, 1])
  + (I2XD[0, 0, c, 1, 1] - I2XD[1, 0, c, 1, 1]
     - I2XD[0, 1, c, 1, 1] + I2XD[1, 1, c, 1, 1])
);

JdblB[0] := 0;
JdblB[c_] := (
  I2B[1, 1, c]
  + 2 * (I2Bshifted[0, 1, c, 1] - I2Bshifted[1, 1, c, 1])
  + (I2Bdshift[0, 0, c, 1, 1] - I2Bdshift[1, 0, c, 1, 1]
     - I2Bdshift[0, 1, c, 1, 1] + I2Bdshift[1, 1, c, 1, 1])
);


(* ---- 5a. dW2_A edge (Psi, kernel rho) ---- *)

Print["--- 5a: dW2_A edge (d^2 Psi/dR^2) ---"];

(* A_{11}: delta on w1 at {0,1,2}, remaining tent(w2-1)*tent(w3) *)
dW2A11edge = Sum[tentWeights[[ic]] * JshiftedPsi[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW2_A11 = dW2_A22 = ", NumberForm[N[dW2A11edge, 16], 16]];

(* A_{33}: delta on w3 at {-1,0,1}, remaining tent(w1-1)*tent(w2-1) *)
(* By |c| fold: 2*JdblPsi[1] - 2*JdblPsi[0] *)
dW2A33edge = 2 JdblPsi[1] - 2 JdblPsi[0];
Print["  dW2_A33 = ", NumberForm[N[dW2A33edge, 16], 16]];
Print[];


(* ---- 5b. dW3_B edge (X, kernel rho^3) ---- *)

Print["--- 5b: dW3_B edge (d^4 X/dR^4) ---"];

(* B_{1111}: delta'' on w1, shifted tent, kernel 3*rho + 3*c^2/rho *)
dW3B1111edge = Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    3 * JshiftedPsi[c] + 3 c^2 * JshiftedA[c]
  ],
  {ic, 1, 3}
];
Print["  dW3_B1111 = dW3_B2222 = ", NumberForm[N[dW3B1111edge, 16], 16]];

(* B_{1122}: double delta (w1, w2) at {0,1,2}x{0,1,2}, 1D tent(w3), rho^3 *)
aValsE = {0, 1, 2};  bValsE = {0, 1, 2};
dW3B1122edge = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    2 * I1cubic[aValsE[[ia]]^2 + bValsE[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW3_B1122 = ", NumberForm[N[dW3B1122edge, 16], 16]];

(* B_{1133}: double delta (w1, w3) at {0,1,2}x{-1,0,1}, shifted 1D tent(w2-1), rho^3 *)
cValsE = {-1, 0, 1};
dW3B1133edge = Sum[
  tentWeights[[ia]] * tentWeights[[ic]] *
    Jtent1DshiftedCubic[aValsE[[ia]]^2 + cValsE[[ic]]^2],
  {ia, 1, 3}, {ic, 1, 3}
];
Print["  dW3_B1133 = dW3_B2233 = ", NumberForm[N[dW3B1133edge, 16], 16]];

(* B_{3333}: delta'' on w3, double-shifted tent, kernel 3*rho + 3*c^2/rho *)
(* c in {-1,0,1}, by |c| fold: 2*f(1) - 2*f(0) *)
dW3B3333edge = 2 * (3 JdblPsi[1] + 3 JdblA[1]) - 2 * 3 JdblPsi[0];
Print["  dW3_B3333 = ", NumberForm[N[dW3B3333edge, 16], 16]];

(* B_{1112}: delta' on w1, step on w2, tent(w3)
   kernel d(rho^3)/dc = 3*c*rho -> factor 3*c in integrand
   F(c) = int step(w2-1)*tent(w3) * rho(c,w2,w3) dw2 dw3
   = 2*(I2Psi1mv[c] - I2PsiS1mv[c]) where 1mv = (1-v) weight *)
I2Psi1mv[c_] := I2Psi[0, 0, c] - I2Psi[0, 1, c];
I2PsiS1mv[c_] := I2PsiS[0, 0, c, 1] - I2PsiS[0, 1, c, 1];

J1112Psi[c_] := 2 * 3 c * (I2Psi1mv[c] - I2PsiS1mv[c]);

(* delta' sum: sum_a (-alpha_a)*J(a) = 0 + 2*J(1) - 1*J(2) *)
dW3B1112edge = 2 J1112Psi[1] - J1112Psi[2];
Print["  dW3_B1112 = dW3_B1222 = ", NumberForm[N[dW3B1112edge, 16], 16]];

(* B_{1233}: step(w1)*step(w2) x delta(w3) with rho^3 kernel
   delta on w3 at {-1,0,1}: 2*J(1) - 2*J(0)
   J(c) = step x step integral of (w1^2+w2^2+c^2)^{3/2}
   = I2X[0,0,c] - 2*I2XS[0,0,c,1] + I2XD[0,0,c,1,1]
   (using I2X for rho^3 kernel) *)

(* Wait — the step x step integral uses the BIHARMONIC identity for rho^3.
   Actually I2X[0,0,c] IS the integral of (c^2+u^2+v^2)^{3/2} on [0,1]^2. *)
J1233X[c_] := I2X[0, 0, c] - 2 I2XS[0, 0, c, 1] + I2XD[0, 0, c, 1, 1];

dW3B1233edge = 2 J1233X[1] - 2 J1233X[0];
Print["  dW3_B1233 = ", NumberForm[N[dW3B1233edge, 16], 16]];

(* A_{12} via Laplacian: A_{12} = B_{1112} + B_{1222} + B_{1233} = 2*B_{1112} + B_{1233} *)
dW2A12edge = (dW3B1112edge + dW3B1112edge + dW3B1233edge) / 12;
(* Wait — the Laplacian identity is Sum_k B_{12kk} = A_{12}
   but for dynamic: Sum_k dW3_B_{12kk} = 12 * dW2_A_{12}
   So dW2_A12 = (dW3_B1211 + dW3_B1222 + dW3_B1233) / 12
   = (dW3_B1112 + dW3_B1222 + dW3_B1233) / 12
   = (2*dW3_B1112 + dW3_B1233) / 12 *)
dW2A12edge = (2 dW3B1112edge + dW3B1233edge) / 12;
Print["  dW2_A12 (via Laplacian) = ", NumberForm[N[dW2A12edge, 16], 16]];

(* Laplacian checks *)
lapl3edge1 = N[dW3B1111edge + dW3B1122edge + dW3B1133edge, 16];
ref3edge1 = N[12 dW2A11edge, 16];
Print["  Laplacian: B1111+B1122+B1133 = ", NumberForm[lapl3edge1, 16]];
Print["             12*dW2_A11        = ", NumberForm[ref3edge1, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl3edge1 - ref3edge1], 3]];

lapl3edge3 = N[dW3B1133edge + dW3B1133edge + dW3B3333edge, 16];
ref3edge3 = N[12 dW2A33edge, 16];
Print["  Laplacian: 2*B1133+B3333 = ", NumberForm[lapl3edge3, 16]];
Print["             12*dW2_A33    = ", NumberForm[ref3edge3, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl3edge3 - ref3edge3], 3]];
Print[];


(* ---- 5c. dW3_A edge (X, kernel rho^3) ---- *)

Print["--- 5c: dW3_A edge (d^2 X/dR^2) ---"];

dW3A11edge = Sum[tentWeights[[ic]] * JshiftedX[{0, 1, 2}[[ic]]], {ic, 1, 3}];
Print["  dW3_A11 = dW3_A22 = ", NumberForm[N[dW3A11edge, 16], 16]];

dW3A33edge = 2 JdblX[1] - 2 JdblX[0];
Print["  dW3_A33 = ", NumberForm[N[dW3A33edge, 16], 16]];

(* dW3_A12 via Laplacian: Sum_k dW4_B_{12kk} = 30 * dW3_A_{12}
   Will be computed after dW4_B in section 5d *)
Print[];


(* ---- 5d. dW4_B edge (Omega, kernel rho^5) ---- *)

Print["--- 5d: dW4_B edge (d^4 Omega/dR^4) ---"];

(* B_{1111}: delta'' on w1, kernel 5*rho^3 + 15*c^2*rho *)
dW4B1111edge = Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    5 * JshiftedX[c] + 15 c^2 * JshiftedPsi[c]
  ],
  {ic, 1, 3}
];
Print["  dW4_B1111 = dW4_B2222 = ", NumberForm[N[dW4B1111edge, 16], 16]];

(* B_{1122}: double delta, 1D tent(w3), rho^5 *)
dW4B1122edge = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    2 * I1quintic[aValsE[[ia]]^2 + bValsE[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW4_B1122 = ", NumberForm[N[dW4B1122edge, 16], 16]];

(* B_{1133}: double delta (w1, w3), shifted 1D tent(w2-1), rho^5 *)
dW4B1133edge = Sum[
  tentWeights[[ia]] * tentWeights[[ic]] *
    Jtent1DshiftedQuintic[aValsE[[ia]]^2 + cValsE[[ic]]^2],
  {ia, 1, 3}, {ic, 1, 3}
];
Print["  dW4_B1133 = dW4_B2233 = ", NumberForm[N[dW4B1133edge, 16], 16]];

(* B_{3333}: delta'' on w3, double-shifted tent, kernel 5*rho^3 + 15*c^2*rho *)
dW4B3333edge = 2 * (5 JdblX[1] + 15 JdblPsi[1]) - 2 * 5 JdblX[0];
Print["  dW4_B3333 = ", NumberForm[N[dW4B3333edge, 16], 16]];

(* B_{1112}: delta' on w1, step on w2, tent(w3)
   kernel d(rho^5)/dc = 5*c*rho^3 -> factor 5*c *)
I2X1mv[c_] := I2X[0, 0, c] - I2X[0, 1, c];
I2XS1mv[c_] := I2XS[0, 0, c, 1] - I2XS[0, 1, c, 1];

J1112X[c_] := 2 * 5 c * (I2X1mv[c] - I2XS1mv[c]);

dW4B1112edge = 2 J1112X[1] - J1112X[2];
Print["  dW4_B1112 = dW4_B1222 = ", NumberForm[N[dW4B1112edge, 16], 16]];

(* B_{1233}: step x step x delta, rho^5 kernel
   Need integral of (c^2+u^2+v^2)^{5/2} — define I2Omega wrapper *)
ClearAll[I2Omega];
I2Omega[p_Integer, q_Integer, c_] :=
  c^2 * I2X[p, q, c] + I2X[p + 2, q, c] + I2X[p, q + 2, c];

ClearAll[I2OmegaS];
I2OmegaS[p_Integer, q_Integer, c_, s_] :=
  (c^2 + s^2) * I2XS[p, q, c, s] +
  I2XS[p + 2, q, c, s] + 2 s * I2XS[p + 1, q, c, s] +
  I2XS[p, q + 2, c, s];

ClearAll[I2OmegaD];
I2OmegaD[p_Integer, q_Integer, c_, s1_, s2_] :=
  (c^2 + s1^2 + s2^2) * I2XD[p, q, c, s1, s2] +
  I2XD[p + 2, q, c, s1, s2] + 2 s1 * I2XD[p + 1, q, c, s1, s2] +
  I2XD[p, q + 2, c, s1, s2] + 2 s2 * I2XD[p, q + 1, c, s1, s2];

J1233Omega[c_] := I2Omega[0, 0, c] - 2 I2OmegaS[0, 0, c, 1] + I2OmegaD[0, 0, c, 1, 1];

dW4B1233edge = 2 J1233Omega[1] - 2 J1233Omega[0];
Print["  dW4_B1233 = ", NumberForm[N[dW4B1233edge, 16], 16]];

(* dW3_A12 via Laplacian: Sum_k dW4_B_{12kk} = 30 * dW3_A_{12} *)
dW3A12edge = (2 dW4B1112edge + dW4B1233edge) / 30;
Print["  dW3_A12 (via Laplacian) = ", NumberForm[N[dW3A12edge, 16], 16]];

(* Laplacian checks for order 2 *)
lapl4edge1 = N[dW4B1111edge + dW4B1122edge + dW4B1133edge, 16];
ref4edge1 = N[30 dW3A11edge, 16];
Print["  Laplacian: B1111+B1122+B1133 = ", NumberForm[lapl4edge1, 16]];
Print["             30*dW3_A11        = ", NumberForm[ref4edge1, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl4edge1 - ref4edge1], 3]];

lapl4edge3 = N[2 dW4B1133edge + dW4B3333edge, 16];
ref4edge3 = N[30 dW3A33edge, 16];
Print["  Laplacian: 2*B1133+B3333 = ", NumberForm[lapl4edge3, 16]];
Print["             30*dW3_A33    = ", NumberForm[ref4edge3, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl4edge3 - ref4edge3], 3]];
Print[];


(* ================================================================ *)
(* Section 6. CORNER-ADJACENT DYNAMIC DERIVATIVES  R = (1,1,1)      *)
(*                                                                   *)
(* S_3 symmetry.  Independent:                                       *)
(*   A: A_{11}=A_{22}=A_{33}, A_{12}=A_{13}=A_{23} (via Laplacian) *)
(*   B: B_{1111}, B_{1122}, B_{1112}, B_{1123}                      *)
(* ================================================================ *)

Print["================================================================"];
Print["  CORNER-ADJACENT R = (1,1,1)  — Dynamic Derivatives"];
Print["================================================================"];
Print[];


(* ---- 6a. dW2_A corner (Psi, kernel rho) ---- *)

Print["--- 6a: dW2_A corner ---"];

(* A_{11}: delta on w1 at {0,1,2}, remaining tent(w2-1)*tent(w3-1) = JdblPsi *)
dW2A11corner = Re[Sum[tentWeights[[ic]] * JdblPsi[{0, 1, 2}[[ic]]], {ic, 1, 3}]];
Print["  dW2_A11 = dW2_A22 = dW2_A33 = ", NumberForm[N[dW2A11corner, 16], 16]];
Print[];


(* ---- 6b. dW3_B corner (X, kernel rho^3) ---- *)

Print["--- 6b: dW3_B corner ---"];

(* B_{1111}: delta'' on w1, double-shifted tent, kernel 3*rho + 3*c^2/rho *)
dW3B1111corner = Re[Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    3 * JdblPsi[c] + 3 c^2 * JdblA[c]
  ],
  {ic, 1, 3}
]];
Print["  dW3_B1111 = ", NumberForm[N[dW3B1111corner, 16], 16]];

(* B_{1122}: double delta (w1, w2) at {0,1,2}x{0,1,2}
   Remaining 1D: tent(w3-1), rho^3 kernel *)
dW3B1122corner = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    Jtent1DshiftedCubic[aValsE[[ia]]^2 + bValsE[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW3_B1122 = ", NumberForm[N[dW3B1122corner, 16], 16]];

(* B_{1112}: delta' on w1, step on w2, tent(w3-1)
   kernel d(rho^3)/dc = 3*c*rho
   F(c) involves step(w2-1)*tent(w3-1) integration *)

(* For corner, the step integral over w2 combined with tent(w3-1) involves
   4 pieces: w2 in [0,1]/[1,2] x w3 in [0,1]/[1,2] *)
FstepPsi[a_] := (
  I2Psi[0, 1, a]
  - I2PsiS[0, 1, a, 1]
  + (I2PsiS[0, 0, a, 1] - I2PsiS[1, 0, a, 1])
  - (I2PsiD[0, 0, a, 1, 1] - I2PsiD[0, 1, a, 1, 1])
);
J1112cornerPsi[a_] := 3 a * FstepPsi[a];

dW3B1112corner = Re[2 J1112cornerPsi[1] - J1112cornerPsi[2]];
Print["  dW3_B1112 = ", NumberForm[N[dW3B1112corner, 16], 16]];

(* B_{1123}: delta on w1 at {0,1,2}, step(w2)*step(w3) with rho^3 kernel
   J(c) = I2X[0,0,c] - 2*I2XS[0,0,c,1] + I2XD[0,0,c,1,1] *)
dW3B1123corner = Re[Sum[
  tentWeights[[ia]] * J1233X[aValsE[[ia]]],
  {ia, 1, 3}
]];
Print["  dW3_B1123 = ", NumberForm[N[dW3B1123corner, 16], 16]];

(* dW2_A12 via Laplacian: Sum_k dW3_B_{12kk} = 12*dW2_A_{12}
   By S_3: B_{1211}=B_{1112}, B_{1222}=B_{1112}, B_{1233}=B_{1123}
   So: 12*dW2_A12 = 2*dW3_B1112 + dW3_B1123 *)
dW2A12corner = Re[(2 dW3B1112corner + dW3B1123corner) / 12];
Print["  dW2_A12 (via Laplacian) = ", NumberForm[N[dW2A12corner, 16], 16]];

(* Laplacian check: B_{1111}+2*B_{1122} = 12*A_{11} *)
lapl3corner = N[dW3B1111corner + 2 dW3B1122corner, 16];
ref3corner = N[12 dW2A11corner, 16];
Print["  Laplacian: B1111+2*B1122 = ", NumberForm[lapl3corner, 16]];
Print["             12*dW2_A11    = ", NumberForm[ref3corner, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl3corner - ref3corner], 3]];
Print[];


(* ---- 6c. dW3_A corner (X, kernel rho^3) ---- *)

Print["--- 6c: dW3_A corner ---"];

dW3A11corner = Re[Sum[tentWeights[[ic]] * JdblX[{0, 1, 2}[[ic]]], {ic, 1, 3}]];
Print["  dW3_A11 = dW3_A22 = dW3_A33 = ", NumberForm[N[dW3A11corner, 16], 16]];
Print[];


(* ---- 6d. dW4_B corner (Omega, kernel rho^5) ---- *)

Print["--- 6d: dW4_B corner ---"];

(* B_{1111}: delta'' on w1, double-shifted tent, kernel 5*rho^3 + 15*c^2*rho *)
dW4B1111corner = Re[Sum[
  tentWeights[[ic]] * Module[{c = {0, 1, 2}[[ic]]},
    5 * JdblX[c] + 15 c^2 * JdblPsi[c]
  ],
  {ic, 1, 3}
]];
Print["  dW4_B1111 = ", NumberForm[N[dW4B1111corner, 16], 16]];

(* B_{1122}: double delta (w1, w2), shifted 1D tent(w3-1), rho^5 *)
dW4B1122corner = Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    Jtent1DshiftedQuintic[aValsE[[ia]]^2 + bValsE[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  dW4_B1122 = ", NumberForm[N[dW4B1122corner, 16], 16]];

(* B_{1112}: delta' on w1, step on w2, tent(w3-1)
   kernel d(rho^5)/dc = 5*c*rho^3 *)
FstepX[a_] := (
  I2X[0, 1, a]
  - I2XS[0, 1, a, 1]
  + (I2XS[0, 0, a, 1] - I2XS[1, 0, a, 1])
  - (I2XD[0, 0, a, 1, 1] - I2XD[0, 1, a, 1, 1])
);
J1112cornerX[a_] := 5 a * FstepX[a];

dW4B1112corner = Re[2 J1112cornerX[1] - J1112cornerX[2]];
Print["  dW4_B1112 = ", NumberForm[N[dW4B1112corner, 16], 16]];

(* B_{1123}: delta on w1, step x step, rho^5 kernel *)
dW4B1123corner = Re[Sum[
  tentWeights[[ia]] * J1233Omega[aValsE[[ia]]],
  {ia, 1, 3}
]];
Print["  dW4_B1123 = ", NumberForm[N[dW4B1123corner, 16], 16]];

(* dW3_A12 via Laplacian: 30*dW3_A12 = 2*dW4_B1112 + dW4_B1123 *)
dW3A12corner = Re[(2 dW4B1112corner + dW4B1123corner) / 30];
Print["  dW3_A12 (via Laplacian) = ", NumberForm[N[dW3A12corner, 16], 16]];

(* Laplacian check *)
lapl4corner = N[dW4B1111corner + 2 dW4B1122corner, 16];
ref4corner = N[30 dW3A11corner, 16];
Print["  Laplacian: B1111+2*B1122 = ", NumberForm[lapl4corner, 16]];
Print["             30*dW3_A11    = ", NumberForm[ref4corner, 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl4corner - ref4corner], 3]];
Print[];


(* ================================================================ *)
(* Section 7. Finite-Difference Cross-Validation                     *)
(*                                                                   *)
(* Define Psi, X, Omega potentials via NIntegrate, then use central  *)
(* differences to validate analytical derivatives.                   *)
(* ================================================================ *)

Print["==== Section 7: Finite-Difference Cross-Validation ===="];
Print[];

deltaFD = 5/10000;  (* exact rational to avoid precision issues *)

(* Use piecewise tent product (no Max[]) with kink subdivision for accuracy *)
tentPW[t_] := Piecewise[{{1 - t, 0 <= t <= 1}, {1 + t, -1 <= t < 0}}, 0];

(* Generic potential W_n(R) = int tent(w-R)*kernel(w) dw
   Single tent function, with kink-aware subdivision. *)
potential[kernel_, {Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
  NIntegrate[
    tentPW[w1 - Rx] tentPW[w2 - Ry] tentPW[w3 - Rz] * kernel[w1, w2, w3],
    {w1, Rx - 1, Rx, Rx + 1}, {w2, Ry - 1, Ry, Ry + 1}, {w3, Rz - 1, Rz, Rz + 1},
    WorkingPrecision -> 25, PrecisionGoal -> 12, MaxRecursion -> 15
  ];

Psi00[R_] := potential[Function[{w1, w2, w3}, Sqrt[w1^2 + w2^2 + w3^2]], R];
X00[R_] := potential[Function[{w1, w2, w3}, (w1^2 + w2^2 + w3^2)^(3/2)], R];
Omega00[R_] := potential[Function[{w1, w2, w3}, (w1^2 + w2^2 + w3^2)^(5/2)], R];

(* Central-difference second derivative *)
d2FD[f_, R0_, j_] := Module[{ej = deltaFD UnitVector[3, j]},
  (f[R0 + ej] - 2 f[R0] + f[R0 - ej]) / deltaFD^2
];

(* --- Face R=(1,0,0): FD validation of second derivatives only --- *)
(* NOTE: Fourth-derivative FD requires delta^-4 ~ 10^14 amplification, *)
(* which exceeds NIntegrate accuracy. Use Laplacian identity checks instead. *)
(* NOTE: dW3_A11_face FD may disagree due to NIntegrate precision limits *)
(* on rho^3 kernel in the longitudinal direction. The Laplacian identity *)
(* dW4_B1111+2*dW4_B1122 = 30*dW3_A11 provides authoritative validation. *)
Print["--- Face R=(1,0,0) FD validation (second derivatives only) ---"];
R0face = {1, 0, 0};

d2Psi11FD = d2FD[Psi00, R0face, 1];
d2Psi22FD = d2FD[Psi00, R0face, 2];
fdDiff[analytical_, fd_] := Abs[Re[N[analytical]] - N[fd]];
Print["  dW2_A11: analytical=", NumberForm[Re[N[dW2A11face, 10]], 10],
      "  FD=", NumberForm[N[d2Psi11FD], 10],
      "  |D|=", ScientificForm[fdDiff[dW2A11face, d2Psi11FD], 3]];
Print["  dW2_A22: analytical=", NumberForm[Re[N[dW2A22face, 10]], 10],
      "  FD=", NumberForm[N[d2Psi22FD], 10],
      "  |D|=", ScientificForm[fdDiff[dW2A22face, d2Psi22FD], 3]];

d2X11FD = d2FD[X00, R0face, 1];
d2X22FD = d2FD[X00, R0face, 2];
Print["  dW3_A11: analytical=", NumberForm[Re[N[dW3A11face, 10]], 10],
      "  FD=", NumberForm[N[d2X11FD], 10],
      "  |D|=", ScientificForm[fdDiff[dW3A11face, d2X11FD], 3]];
Print["  dW3_A22: analytical=", NumberForm[Re[N[dW3A22face, 10]], 10],
      "  FD=", NumberForm[N[d2X22FD], 10],
      "  |D|=", ScientificForm[fdDiff[dW3A22face, d2X22FD], 3]];
Print[];


(* --- Edge R=(1,1,0) --- *)
Print["--- Edge R=(1,1,0) FD validation ---"];
R0edge = {1, 1, 0};

d2Psi11edgeFD = d2FD[Psi00, R0edge, 1];
d2Psi33edgeFD = d2FD[Psi00, R0edge, 3];
Print["  dW2_A11: analytical=", NumberForm[Re[N[dW2A11edge, 10]], 10],
      "  FD=", NumberForm[N[d2Psi11edgeFD], 10],
      "  |D|=", ScientificForm[fdDiff[dW2A11edge, d2Psi11edgeFD], 3]];
Print["  dW2_A33: analytical=", NumberForm[Re[N[dW2A33edge, 10]], 10],
      "  FD=", NumberForm[N[d2Psi33edgeFD], 10],
      "  |D|=", ScientificForm[fdDiff[dW2A33edge, d2Psi33edgeFD], 3]];

d2X11edgeFD = d2FD[X00, R0edge, 1];
Print["  dW3_A11: analytical=", NumberForm[Re[N[dW3A11edge, 10]], 10],
      "  FD=", NumberForm[N[d2X11edgeFD], 10],
      "  |D|=", ScientificForm[fdDiff[dW3A11edge, d2X11edgeFD], 3]];
Print[];


(* --- Corner R=(1,1,1) --- *)
Print["--- Corner R=(1,1,1) FD validation ---"];
R0corner = {1, 1, 1};

d2Psi11cornerFD = d2FD[Psi00, R0corner, 1];
Print["  dW2_A11: analytical=", NumberForm[Re[N[dW2A11corner, 10]], 10],
      "  FD=", NumberForm[N[d2Psi11cornerFD], 10],
      "  |D|=", ScientificForm[fdDiff[dW2A11corner, d2Psi11cornerFD], 3]];

d2X11cornerFD = d2FD[X00, R0corner, 1];
Print["  dW3_A11: analytical=", NumberForm[Re[N[dW3A11corner, 10]], 10],
      "  FD=", NumberForm[N[d2X11cornerFD], 10],
      "  |D|=", ScientificForm[fdDiff[dW3A11corner, d2X11cornerFD], 3]];
Print[];


(* ================================================================ *)
(* Section 8. Export                                                 *)
(* ================================================================ *)

Print["==== Section 8: Export ===="];
Print[];

outFile = FileNameJoin[{DirectoryName[$InputFileName], "InterVoxelPropagatorDynamicValues.wl"}];
Print["  Exporting to ", outFile, " ..."];

exportStr = StringJoin[
  "(* InterVoxelPropagatorDynamicValues.wl\n",
  "   Auto-generated by InterVoxelPropagatorDynamic.wl\n",
  "   Raw derivatives d^2 W_m / dR_j dR_l and d^4 W_m / dR^4\n",
  "   WITHOUT 1/(4pi) or 1/(8pi) normalisation factors.\n",
  "   Date: ", DateString[], "\n*)\n\n",

  "(* ===== FACE R=(1,0,0) ===== *)\n\n",
  "(* Order 1: Psi (rho) -> A, X (rho^3) -> B *)\n",
  "dW2A11face = ", ToString[N[dW2A11face, 20], InputForm], ";\n",
  "dW2A22face = ", ToString[N[dW2A22face, 20], InputForm], ";\n",
  "dW3B1111face = ", ToString[N[dW3B1111face, 20], InputForm], ";\n",
  "dW3B1122face = ", ToString[N[dW3B1122face, 20], InputForm], ";\n",
  "dW3B2222face = ", ToString[N[dW3B2222face, 20], InputForm], ";\n",
  "dW3B2233face = ", ToString[N[dW3B2233face, 20], InputForm], ";\n\n",
  "(* Order 2: X (rho^3) -> A, Omega (rho^5) -> B *)\n",
  "dW3A11face = ", ToString[N[dW3A11face, 20], InputForm], ";\n",
  "dW3A22face = ", ToString[N[dW3A22face, 20], InputForm], ";\n",
  "dW4B1111face = ", ToString[N[dW4B1111face, 20], InputForm], ";\n",
  "dW4B1122face = ", ToString[N[dW4B1122face, 20], InputForm], ";\n",
  "dW4B2222face = ", ToString[N[dW4B2222face, 20], InputForm], ";\n",
  "dW4B2233face = ", ToString[N[dW4B2233face, 20], InputForm], ";\n\n",

  "(* ===== EDGE R=(1,1,0) ===== *)\n\n",
  "(* Order 1 *)\n",
  "dW2A11edge = ", ToString[N[dW2A11edge, 20], InputForm], ";\n",
  "dW2A33edge = ", ToString[N[dW2A33edge, 20], InputForm], ";\n",
  "dW2A12edge = ", ToString[N[dW2A12edge, 20], InputForm], ";\n",
  "dW3B1111edge = ", ToString[N[dW3B1111edge, 20], InputForm], ";\n",
  "dW3B1122edge = ", ToString[N[dW3B1122edge, 20], InputForm], ";\n",
  "dW3B1133edge = ", ToString[N[dW3B1133edge, 20], InputForm], ";\n",
  "dW3B3333edge = ", ToString[N[dW3B3333edge, 20], InputForm], ";\n",
  "dW3B1112edge = ", ToString[N[dW3B1112edge, 20], InputForm], ";\n",
  "dW3B1233edge = ", ToString[N[dW3B1233edge, 20], InputForm], ";\n\n",
  "(* Order 2 *)\n",
  "dW3A11edge = ", ToString[N[dW3A11edge, 20], InputForm], ";\n",
  "dW3A33edge = ", ToString[N[dW3A33edge, 20], InputForm], ";\n",
  "dW3A12edge = ", ToString[N[dW3A12edge, 20], InputForm], ";\n",
  "dW4B1111edge = ", ToString[N[dW4B1111edge, 20], InputForm], ";\n",
  "dW4B1122edge = ", ToString[N[dW4B1122edge, 20], InputForm], ";\n",
  "dW4B1133edge = ", ToString[N[dW4B1133edge, 20], InputForm], ";\n",
  "dW4B3333edge = ", ToString[N[dW4B3333edge, 20], InputForm], ";\n",
  "dW4B1112edge = ", ToString[N[dW4B1112edge, 20], InputForm], ";\n",
  "dW4B1233edge = ", ToString[N[dW4B1233edge, 20], InputForm], ";\n\n",

  "(* ===== CORNER R=(1,1,1) ===== *)\n\n",
  "(* Order 1 *)\n",
  "dW2A11corner = ", ToString[N[dW2A11corner, 20], InputForm], ";\n",
  "dW2A12corner = ", ToString[N[dW2A12corner, 20], InputForm], ";\n",
  "dW3B1111corner = ", ToString[N[dW3B1111corner, 20], InputForm], ";\n",
  "dW3B1122corner = ", ToString[N[dW3B1122corner, 20], InputForm], ";\n",
  "dW3B1112corner = ", ToString[N[dW3B1112corner, 20], InputForm], ";\n",
  "dW3B1123corner = ", ToString[N[dW3B1123corner, 20], InputForm], ";\n\n",
  "(* Order 2 *)\n",
  "dW3A11corner = ", ToString[N[dW3A11corner, 20], InputForm], ";\n",
  "dW3A12corner = ", ToString[N[dW3A12corner, 20], InputForm], ";\n",
  "dW4B1111corner = ", ToString[N[dW4B1111corner, 20], InputForm], ";\n",
  "dW4B1122corner = ", ToString[N[dW4B1122corner, 20], InputForm], ";\n",
  "dW4B1112corner = ", ToString[N[dW4B1112corner, 20], InputForm], ";\n",
  "dW4B1123corner = ", ToString[N[dW4B1123corner, 20], InputForm], ";\n"
];

Export[outFile, exportStr, "Text"];
Print["  Done."];
Print[];


(* ================================================================ *)
(* Section 9. Summary                                                *)
(* ================================================================ *)

Print["================================================================"];
Print["  SUMMARY OF ALL DYNAMIC DERIVATIVES"];
Print["================================================================"];
Print[];

Print["  FACE R=(1,0,0), C_{4v}:"];
Print["    Order 1: dW2_A11=", NumberForm[N[dW2A11face, 12], 12],
      "  dW2_A22=", NumberForm[N[dW2A22face, 12], 12]];
Print["    Order 1: dW3_B1111=", NumberForm[N[dW3B1111face, 12], 12],
      "  B1122=", NumberForm[N[dW3B1122face, 12], 12],
      "  B2222=", NumberForm[N[dW3B2222face, 12], 12],
      "  B2233=", NumberForm[N[dW3B2233face, 12], 12]];
Print["    Order 2: dW3_A11=", NumberForm[N[dW3A11face, 12], 12],
      "  dW3_A22=", NumberForm[N[dW3A22face, 12], 12]];
Print["    Order 2: dW4_B1111=", NumberForm[N[dW4B1111face, 12], 12],
      "  B1122=", NumberForm[N[dW4B1122face, 12], 12],
      "  B2222=", NumberForm[N[dW4B2222face, 12], 12],
      "  B2233=", NumberForm[N[dW4B2233face, 12], 12]];
Print[];

Print["  EDGE R=(1,1,0), C_{2v}:"];
Print["    Order 1: dW2_A11=", NumberForm[N[dW2A11edge, 12], 12],
      "  A33=", NumberForm[N[dW2A33edge, 12], 12],
      "  A12=", NumberForm[N[dW2A12edge, 12], 12]];
Print["    Order 1: dW3_B1111=", NumberForm[N[dW3B1111edge, 12], 12],
      "  B1122=", NumberForm[N[dW3B1122edge, 12], 12],
      "  B1133=", NumberForm[N[dW3B1133edge, 12], 12]];
Print["             B3333=", NumberForm[N[dW3B3333edge, 12], 12],
      "  B1112=", NumberForm[N[dW3B1112edge, 12], 12],
      "  B1233=", NumberForm[N[dW3B1233edge, 12], 12]];
Print["    Order 2: dW3_A11=", NumberForm[N[dW3A11edge, 12], 12],
      "  A33=", NumberForm[N[dW3A33edge, 12], 12],
      "  A12=", NumberForm[N[dW3A12edge, 12], 12]];
Print["    Order 2: dW4_B1111=", NumberForm[N[dW4B1111edge, 12], 12],
      "  B1122=", NumberForm[N[dW4B1122edge, 12], 12],
      "  B1133=", NumberForm[N[dW4B1133edge, 12], 12]];
Print["             B3333=", NumberForm[N[dW4B3333edge, 12], 12],
      "  B1112=", NumberForm[N[dW4B1112edge, 12], 12],
      "  B1233=", NumberForm[N[dW4B1233edge, 12], 12]];
Print[];

Print["  CORNER R=(1,1,1), S_3:"];
Print["    Order 1: dW2_A11=", NumberForm[N[dW2A11corner, 12], 12],
      "  A12=", NumberForm[N[dW2A12corner, 12], 12]];
Print["    Order 1: dW3_B1111=", NumberForm[N[dW3B1111corner, 12], 12],
      "  B1122=", NumberForm[N[dW3B1122corner, 12], 12],
      "  B1112=", NumberForm[N[dW3B1112corner, 12], 12],
      "  B1123=", NumberForm[N[dW3B1123corner, 12], 12]];
Print["    Order 2: dW3_A11=", NumberForm[N[dW3A11corner, 12], 12],
      "  A12=", NumberForm[N[dW3A12corner, 12], 12]];
Print["    Order 2: dW4_B1111=", NumberForm[N[dW4B1111corner, 12], 12],
      "  B1122=", NumberForm[N[dW4B1122corner, 12], 12],
      "  B1112=", NumberForm[N[dW4B1112corner, 12], 12],
      "  B1123=", NumberForm[N[dW4B1123corner, 12], 12]];
Print[];

Print["================================================================"];
Print["  DYNAMIC PROPAGATOR COMPLETE"];
Print["  Total time: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
