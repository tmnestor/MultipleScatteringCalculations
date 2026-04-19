(* ::Package:: *)
(* InterVoxelPropagatorCorner.wl
   Analytical inter-voxel strain propagator for CORNER-adjacent cubes.
   R = (a, a, a) with C_{3v} = S_3 symmetry (full permutation of axes).

   ALL three tents shifted: M00(w1-1), M00(w2-1), M00(w3-1).

   Independent values under S_3:
     A: A_{11}=A_{22}=A_{33}, A_{12}=A_{13}=A_{23}  (2 values)
     B: B_{1111}, B_{1122}, B_{1112}, B_{1123}        (4 values)

   The Laplacian identity A_{12} = 2*B_{1112} + B_{1123} is used
   (3D step x step x 1/rho integration is unreliable in Mathematica).

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorCorner.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  INTER-VOXEL ELASTIC STRAIN PROPAGATOR -- CORNER ADJACENT"];
Print["  R = (a, a, a) with S_3 symmetry"];
Print["  Analytical reduction via delta-function collapse"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];

(* ================================================================ *)
(* Section 0. Parameters                                             *)
(* ================================================================ *)

mu = 1;
nu = 1/4;
eta = 1/(2(1 - nu));   (* = 2/3 for nu=1/4 *)
rho0 = 1;
cs = Sqrt[mu/rho0];
cp = Sqrt[(1 - nu)/((1 + nu)(1 - 2 nu))] cs;
a = 1;
h = a/2;

Print["Parameters:"];
Print["  mu = ", mu, ", nu = ", nu, ", eta = ", eta // N];
Print["  a = ", a, ", h = ", h];
Print[];


(* ================================================================ *)
(* Section 1. 2D Master integrals (same as edge script)              *)
(* ================================================================ *)

Print["==== Section 1: 2D Master Integrals ===="];
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

(* --- Single-shifted A-channel --- *)
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
(* Section 2. 1D Elementary integrals                                *)
(* ================================================================ *)

Print["==== Section 2: 1D Elementary Integrals ===="];
Print[];

(* I1exact[C] = int_0^1 (1-t) sqrt(C + t^2) dt *)
ClearAll[I1exact];
I1exact[0] = 1/6;
I1exact[c2_] := Module[{sc1 = Sqrt[c2 + 1], sc = Sqrt[c2]},
  1/2 (sc1 + c2 * ArcSinh[1/sc]) - 1/3 (sc1^3 - sc^3)
];

(* Shifted 1D biharmonic: int_0^1 (1-t) sqrt(C + (t+1)^2) dt *)
ClearAll[Jpiece2memo];
Jpiece2memo[bigC_] := Jpiece2memo[bigC] = Module[{result},
  result = Integrate[(1 - s) Sqrt[bigC + (s + 1)^2], {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  Simplify[result]
];

(* Full shifted 1D tent integral: int_0^2 tent(t-1) sqrt(C+t^2) dt *)
Jtent1Dshifted[bigC_] :=
  (1/3) ((bigC + 1)^(3/2) - bigC^(3/2)) + Jpiece2memo[bigC];

(* Validation *)
Do[
  exact = I1exact[c2];
  nval = NIntegrate[(1 - t) Sqrt[c2 + t^2], {t, 0, 1},
           WorkingPrecision -> 25, PrecisionGoal -> 16];
  err = Abs[N[exact, 25] - nval];
  Print["  I1[", c2, "] = ", NumberForm[N[exact, 16], 16],
        "  err = ", ScientificForm[err, 3],
        If[err < 10^-12, "  OK", "  FAIL"]];
  ,
  {c2, {0, 1, 2, 4, 5}}
];
Print[];


(* ================================================================ *)
(* Section 3. Newton Potential Derivatives A_{jl} for R = (1,1,1)   *)
(*                                                                   *)
(* S_3 symmetry: A_{11}=A_{22}=A_{33}, A_{12}=A_{13}=A_{23}        *)
(*                                                                   *)
(* All three tents shifted: support [0,2]^3.                        *)
(* ================================================================ *)

Print["==== Section 3: Newton Potential Derivatives (A_jl) -- Corner ===="];
Print[];

tentWeights = {+1, -2, +1};

(* --- A_{11}: d^2 Phi/dR1^2 --- delta collapse on w1 ---
   w1 delta points: {R1-1, R1, R1+1} = {0, 1, 2}, weights {+1,-2,+1}
   Remaining: tent(w2-1)*tent(w3-1) — BOTH shifted!
   This is EXACTLY the Jdbl(c) function from the edge geometry.       *)

Print["--- A_{11} = A_{22} = A_{33} for R = (1,1,1) ---"];
Print["  (delta collapse on w1, remaining tent(w2-1)*tent(w3-1))"];

(* Precompute needed integrals for Jdbl *)
Print["  Computing shifted and double-shifted integrals ..."];
Do[I2Ashifted[p, q, c, 1], {c, {0, 1, 2}}, {p, 0, 1}, {q, 0, 1}];
Do[I2Adshift[p, q, c, 1, 1], {c, {0, 1, 2}}, {p, 0, 1}, {q, 0, 1}];

(* Jdbl(c): double-shifted tent product — same definition as edge *)
Jdbl[c_] := (
  I2A[1, 1, c]
  + 2 * (I2Ashifted[0, 1, c, 1] - I2Ashifted[1, 1, c, 1])
  + (I2Adshift[0, 0, c, 1, 1] - I2Adshift[1, 0, c, 1, 1]
     - I2Adshift[0, 1, c, 1, 1] + I2Adshift[1, 1, c, 1, 1])
);

Do[
  Print["  Jdbl[", c, "] = ", NumberForm[N[Jdbl[c], 16], 16]];
  , {c, {0, 1, 2}}
];

A11corner = Re[-1/(4 Pi) * Sum[
  tentWeights[[ic]] * Jdbl[{0, 1, 2}[[ic]]],
  {ic, 1, 3}
]];
Print["  A_{11} = A_{22} = A_{33} = ", NumberForm[N[A11corner, 16], 16]];

A22corner = A11corner;
A33corner = A11corner;
Print[];


(* --- A_{12}: DEFERRED to Section 4b (Laplacian identity) ---
   Step x step x 1/rho is unreliable via sequential integration.   *)

Print["--- A_{12} = A_{13} = A_{23} for R = (1,1,1) ---"];
Print["  DEFERRED: will use Laplacian identity A_{12} = 2*B_{1112} + B_{1123}"];
A13corner = Null;
A23corner = Null;
Print[];


(* ================================================================ *)
(* Section 4. Biharmonic Fourth Derivatives B_{ijkl}                *)
(*                                                                   *)
(* Independent under S_3:                                            *)
(*   B_{1111} = B_{2222} = B_{3333}                                *)
(*   B_{1122} = B_{1133} = B_{2233}                                *)
(*   B_{1112} = B_{1113} = B_{2221} = ...  (6 equiv by S_3)       *)
(*   B_{1123}  (= B_{1213} = B_{1233} by derivative commutativity) *)
(* ================================================================ *)

Print["==== Section 4: Biharmonic Fourth Derivatives (B_ijkl) -- Corner ===="];
Print[];


(* --- B_{1122}: d^4 Psi/(dR1^2 dR2^2) --- DOUBLE DELTA COLLAPSE ---
   w1 deltas at {0,1,2}, w2 deltas at {0,1,2}.
   Remaining: 1D in w3 with tent(w3-1), kernel |w|.
   Use Jtent1Dshifted(C) for the shifted tent.                       *)

Print["--- B_{1122} (double delta collapse -> 1D, shifted tent) ---"];

aVals = {0, 1, 2};
bVals = {0, 1, 2};

B1122corner = -1/(8 Pi) * Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    Jtent1Dshifted[aVals[[ia]]^2 + bVals[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  B_{1122} = ", NumberForm[N[B1122corner, 16], 16]];
Print["  B_{1133} = B_{2233} = B_{1122}  (S_3)"];
B1133corner = B1122corner;
B2233corner = B1122corner;
Print[];


(* --- B_{1111}: d^4 Psi/dR1^4 --- SAME-AXIS via delta'' ---
   w1 delta points: {0,1,2}, weights {+1,-2,+1}.
   d^2/dw1^2[|w|] = 1/rho - c^2/rho^3
   Remaining: tent(w2-1)*tent(w3-1) — double-shifted.

   JdblB(c) for B-channel double-shifted.                            *)

Print["--- B_{1111} (same-axis delta'', double-shifted tent) ---"];

Print["  Computing B-channel integrals ..."];
Do[I2Bshifted[p, q, c, 1], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];
Do[I2Bdshift[p, q, c, 1, 1], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];
(* Also need unshifted B-channel for Jdbl *)
Do[I2B[p, q, c], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];

JdblB[0] := 0;
JdblB[c_] := (
  I2B[1, 1, c]
  + 2 * (I2Bshifted[0, 1, c, 1] - I2Bshifted[1, 1, c, 1])
  + (I2Bdshift[0, 0, c, 1, 1] - I2Bdshift[1, 0, c, 1, 1]
     - I2Bdshift[0, 1, c, 1, 1] + I2Bdshift[1, 1, c, 1, 1])
);

B1111corner = Re[-1/(8 Pi) * Sum[
  tentWeights[[ic]] * (Jdbl[{0, 1, 2}[[ic]]] - JdblB[{0, 1, 2}[[ic]]]),
  {ic, 1, 3}
]];
Print["  B_{1111} = ", NumberForm[N[B1111corner, 16], 16]];
B2222corner = B1111corner;
B3333corner = B1111corner;
Print["  B_{2222} = B_{3333} = B_{1111}  (S_3)"];
Print[];


(* --- B_{1112}: d^4 Psi/(dR1^3 dR2) --- delta' on w1, step on w2 ---
   SAME structure as edge B_{1112}, but remaining tent is
   tent(w3-1) SHIFTED, not tent(w3) centered.

   delta'(w1-a) with weights {+1,-2,+1} at a={0,1,2}:
   delta' acts as: -d/dw1[|w|] = -(a/rho)
   Kernel after collapse: a/rho(a,w2,w3)

   M00'(w2-1): step, +1 on [0,1], -1 on [1,2]
   tent(w3-1): fold to [0,2], split [0,1] and [1,2]

   F(a) = int M00'(w2-1)*tent(w3-1) / rho(a,w2,w3) dw2 dw3

   The tent(w3-1) replaces tent(w3) from the edge case. The w3 integral
   over [0,2] with tent(w3-1) splits as:
     [0,1]: tent = w3, combine with step on w2 (4 pieces: 2 w3-pieces x 2 w2-pieces)
     [1,2]: tent = 2-w3, sub s3=w3-1 -> (1-s3)

   Expand F(a) into step(w2) x tent(w3-1) pieces.                    *)

Print["--- B_{1112} (delta' x step x shifted tent) ---"];

(* Precompute shifted integrals for the tent(w3-1) structure.
   The integral decomposes into 4 pieces:
   F(a) = [unshifted w2 x unshifted w3] - [shifted w2 x unshifted w3]
        + [unshifted w2 x shifted w3] - [shifted w2 x shifted w3]
   with appropriate tent weights.

   Piece 1: w2 in [0,1] (+1), w3 in [0,1] (tent=w3)
     int_0^1 int_0^1 w3 / rho(a,t,v) dt dv = I2A[0,1,a]
   Piece 2: w2 in [1,2] (-1), w3 in [0,1] (tent=w3), sub w2=t+1
     int_0^1 int_0^1 v / rho(a,t+1,v) dt dv = I2Ashifted[0,1,a,1]
   Piece 3: w2 in [0,1] (+1), w3 in [1,2] (tent=2-w3), sub w3=s+1
     int_0^1 int_0^1 (1-s) / rho(a,t,s+1) dt ds = I2Ashifted[0,0,a,1] - I2Ashifted[0,1,a,1]
     (where shift is on v variable -- but I2Ashifted shifts u, not v)
     Actually I2Ashifted[p,q,c,shift] shifts the u variable.
     Here t is u and s is v, with shift on v:
     = int_0^1 int_0^1 (1-s)/sqrt(a^2 + t^2 + (s+1)^2) dt ds
     By u<->v symmetry of the kernel: this equals
     int_0^1 int_0^1 (1-u)/sqrt(a^2 + (u+1)^2 + v^2) du dv
     = I2Ashifted[0,0,a,1] - I2Ashifted[1,0,a,1]
   Piece 4: w2 in [1,2] (-1), w3 in [1,2], sub w2=t+1, w3=s+1
     int_0^1 int_0^1 (1-s)/sqrt(a^2 + (t+1)^2 + (s+1)^2) dt ds
     = I2Adshift[0,0,a,1,1] - I2Adshift[0,1,a,1,1]

   F(a) = [Piece1 - Piece2 + Piece3 - Piece4]
   = I2A[0,1,a]
     - I2Ashifted[0,1,a,1]
     + (I2Ashifted[0,0,a,1] - I2Ashifted[1,0,a,1])
     - (I2Adshift[0,0,a,1,1] - I2Adshift[0,1,a,1,1])                *)

Fstep[a_] := (
  I2A[0, 1, a]
  - I2Ashifted[0, 1, a, 1]
  + (I2Ashifted[0, 0, a, 1] - I2Ashifted[1, 0, a, 1])
  - (I2Adshift[0, 0, a, 1, 1] - I2Adshift[0, 1, a, 1, 1])
);

(* J1112(a) = a * F(a) — includes the c/rho factor from delta' action *)
J1112corner[a_] := a * Fstep[a];

(* d4Psi/(dR1^3 dR2) = sum_a (-alpha_a) * J1112(a)
   = -1*0 + 2*J1112(1) - 1*J1112(2)                                  *)
d4Psi1112corner = 2 * J1112corner[1] - J1112corner[2];
B1112corner = Re[-1/(8 Pi) * d4Psi1112corner];
Print["  B_{1112} = ", NumberForm[N[B1112corner, 16], 16]];
Print["  (6 equivalent components by S_3)"];
Print[];


(* --- B_{1123}: d^4 Psi/(dR1^2 dR2 dR3) --- delta on w1, step x step ---
   M00''(w1-1) -> deltas at {0,1,2} with {+1,-2,+1}
   M00'(w2-1) -> step
   M00'(w3-1) -> step

   After w1 collapse at w1=a:
   int step(w2-1)*step(w3-1)*sqrt(a^2+w2^2+w3^2) dw2 dw3

   Split [0,2]^2 into 4 quadrants for step x step:
   (++) = int_0^1 int_0^1 rho(a,u,v)
   (+-) = int_0^1 int_0^1 rho(a,u,v+1)
   (-+) = int_0^1 int_0^1 rho(a,u+1,v)
   (--) = int_0^1 int_0^1 rho(a,u+1,v+1)

   Use corrected biharmonic identity for shifted integrals:
   rho(a,u+s1,v+s2) = [(a^2+s1^2+s2^2) + u^2 + 2s1*u + v^2 + 2s2*v] / rho *)

Print["--- B_{1123} (delta x step x step, biharmonic) ---"];

(* Precompute degree-2 shifted integrals needed *)
Print["  Computing degree-2 shifted/double-shifted integrals ..."];
Do[I2Ashifted[p, 0, c, 1], {c, {0, 1, 2}}, {p, 2, 2}];
Do[I2Ashifted[0, p, c, 1], {c, {0, 1, 2}}, {p, 2, 2}];
Do[I2Adshift[p, 0, c, 1, 1], {c, {0, 1, 2}}, {p, 2, 2}];
Do[I2Adshift[0, p, c, 1, 1], {c, {0, 1, 2}}, {p, 2, 2}];

(* Biharmonic (Psi) integrals with correct shifted identities *)
I2PsiUnshifted[c_] :=
  c^2 * I2A[0, 0, c] + I2A[2, 0, c] + I2A[0, 2, c];

I2PsiShifted1[c_, s_] :=
  (c^2 + s^2) * I2Ashifted[0, 0, c, s] +
  I2Ashifted[2, 0, c, s] + 2 s * I2Ashifted[1, 0, c, s] +
  I2Ashifted[0, 2, c, s];

I2PsiDshift[c_, s1_, s2_] :=
  (c^2 + s1^2 + s2^2) * I2Adshift[0, 0, c, s1, s2] +
  I2Adshift[2, 0, c, s1, s2] + 2 s1 * I2Adshift[1, 0, c, s1, s2] +
  I2Adshift[0, 2, c, s1, s2] + 2 s2 * I2Adshift[0, 1, c, s1, s2];

(* Step x step biharmonic: (++) - (+-) - (-+) + (--)
   By u<->v symmetry: (+-) = (-+), so = (++) - 2*(+-) + (--)      *)
Jbiharm1123[c_] := (
  I2PsiUnshifted[c]
  - 2 * I2PsiShifted1[c, 1]
  + I2PsiDshift[c, 1, 1]
);

(* Need I2A[0,0,c] and I2A[2,0,c] for unshifted Psi *)
Do[I2A[0, 0, c], {c, {0, 1, 2}}];
Do[I2A[2, 0, c], {c, {0, 1, 2}}];

d4Psi1123corner = Sum[
  tentWeights[[ia]] * Jbiharm1123[aVals[[ia]]],
  {ia, 1, 3}
];
B1123corner = Re[-1/(8 Pi) * d4Psi1123corner];
Print["  B_{1123} = ", NumberForm[N[B1123corner, 16], 16]];
Print[];


(* ================================================================ *)
(* Section 4b. A_{12} from Laplacian identity                       *)
(*                                                                   *)
(* A_{12} = Sum_k B_{12kk} = B_{1211} + B_{1222} + B_{1233}        *)
(* By S_3: B_{1211} = B_{1112}, B_{1222} = B_{1112} (swap 1<->2),  *)
(*   B_{1233} = B_{1123} (swap indices).                             *)
(* So A_{12} = 2*B_{1112} + B_{1123}.                               *)
(* ================================================================ *)

Print["==== Section 4b: A_{12} from Laplacian Identity ===="];
Print[];

A12corner = Re[2 * B1112corner + B1123corner];
Print["  A_{12} = 2*B_{1112} + B_{1123}"];
Print["  A_{12} = A_{13} = A_{23} = ", NumberForm[N[A12corner, 16], 16]];
A13corner = A12corner;
A23corner = A12corner;
Print[];


(* ================================================================ *)
(* Section 5. Validation                                             *)
(* ================================================================ *)

Print["==== Section 5: Validation ===="];
Print[];

(* Laplacian check: A_{11} = B_{1111} + B_{1122} + B_{1133}
   = B_{1111} + 2*B_{1122} *)
lapl1 = N[B1111corner + 2 B1122corner, 16];
Print["  Laplacian: B_{1111} + 2*B_{1122} = ", NumberForm[lapl1, 16]];
Print["             A_{11}                 = ", NumberForm[N[A11corner, 16], 16]];
Print["             |diff| = ", ScientificForm[Abs[lapl1 - N[A11corner, 16]], 3]];
Print[];

(* A_{12} = 2*B_{1112} + B_{1123} — true by construction *)
Print["  A_{12} = 2*B_{1112} + B_{1123} — true by construction"];
Print[];

(* Finite-difference cross-validation *)
Print["--- FD Cross-Validation ---"];
Print[];

Phi00corner[{Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
  NIntegrate[
    Max[0, 1 - Abs[w1 - Rx]] Max[0, 1 - Abs[w2 - Ry]] Max[0, 1 - Abs[w3 - Rz]] /
      Sqrt[w1^2 + w2^2 + w3^2],
    {w1, Rx - 1, Rx + 1},
    {w2, Ry - 1, Ry + 1},
    {w3, Rz - 1, Rz + 1},
    Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"},
    WorkingPrecision -> 16, MaxRecursion -> 15, PrecisionGoal -> 10
  ];

deltaFD = 0.0005;
R0 = {1.0, 1.0, 1.0};

Print["  Computing FD derivatives at R=(1,1,1) ..."];

d2Phi11FD = (Phi00corner[R0 + {deltaFD,0,0}] - 2 Phi00corner[R0] +
             Phi00corner[R0 - {deltaFD,0,0}]) / deltaFD^2;
d2Phi12FD = (Phi00corner[R0 + {deltaFD,deltaFD,0}]
           - Phi00corner[R0 + {deltaFD,-deltaFD,0}]
           - Phi00corner[R0 + {-deltaFD,deltaFD,0}]
           + Phi00corner[R0 + {-deltaFD,-deltaFD,0}]) / (4 deltaFD^2);

A11FD = -d2Phi11FD / (4 Pi);
A12FD = -d2Phi12FD / (4 Pi);

Print["  A_{11}: analytical = ", NumberForm[N[A11corner, 10], 10],
      "   FD = ", NumberForm[A11FD, 10],
      "   |D| = ", ScientificForm[Abs[N[A11corner, 16] - A11FD], 3]];
Print["  A_{12}: analytical = ", NumberForm[N[A12corner, 10], 10],
      "   FD = ", NumberForm[A12FD, 10],
      "   |D| = ", ScientificForm[Abs[N[A12corner, 16] - A12FD], 3]];
Print[];


(* ================================================================ *)
(* Section 6. Static Propagator Assembly — Corner Adjacent           *)
(*                                                                   *)
(* P_{ijkl}(R) = -1/(2mu) [delta_ik A_jl + delta_jk A_il           *)
(*                          - 2 eta B_ijkl]                         *)
(*                                                                   *)
(* Independent under S_3:                                            *)
(*   P_{1111}, P_{1122}, P_{1212}, P_{1112}, P_{1123}, P_{1213}    *)
(* ================================================================ *)

Print["==== Section 6: Static Propagator Assembly (Corner-Adjacent) ===="];
Print[];

(* P_{1111}: 2*A_{11} - 2eta*B_{1111} *)
P1111corner = -1/(2 mu) (2 A11corner - 2 eta B1111corner);

(* P_{1122}: -2eta*B_{1122} *)
P1122corner = -1/(2 mu) (-2 eta B1122corner);

(* P_{1212}: delta_11*A_22 + delta_21*A_12 - 2eta*B_{1212}
   = A_{22} - 2eta*B_{1122}  (since B_{1212}=B_{1122}) *)
P1212corner = -1/(2 mu) (A22corner - 2 eta B1122corner);

(* P_{1112}: delta_11*A_12 + delta_11*A_12 - 2eta*B_{1112}
   = 2*A_{12} - 2eta*B_{1112} *)
P1112corner = -1/(2 mu) (2 A12corner - 2 eta B1112corner);

(* P_{1123}: i=1,j=1,k=2,l=3. delta_12*A_13 + delta_12*A_13 - 2eta*B_{1123}
   delta_12=0, so P_{1123} = -1/(2mu)*(-2eta*B_{1123}) *)
P1123corner = -1/(2 mu) (-2 eta B1123corner);

(* P_{1213}: i=1,j=2,k=1,l=3. delta_11*A_23 + delta_21*A_13 - 2eta*B_{1213}
   = A_{23} - 2eta*B_{1123}  (B_{1213}=B_{1123} by derivative commutation) *)
P1213corner = -1/(2 mu) (A23corner - 2 eta B1123corner);

Print["========================================================"];
Print["  STATIC STRAIN PROPAGATOR -- Corner Adjacent R = (a,a,a)"];
Print["========================================================"];
Print["  P_{1111} = P_{2222} = P_{3333} = ", NumberForm[N[P1111corner, 16], 16]];
Print["  P_{1122} = P_{1133} = P_{2233} = ", NumberForm[N[P1122corner, 16], 16]];
Print["  P_{1212} = P_{1313} = P_{2323} = ", NumberForm[N[P1212corner, 16], 16]];
Print["  P_{1112} (S_3 orbit, 6 equiv)   = ", NumberForm[N[P1112corner, 16], 16]];
Print["  P_{1123} (S_3 orbit, 3 equiv)   = ", NumberForm[N[P1123corner, 16], 16]];
Print["  P_{1213} (S_3 orbit, 3 equiv)   = ", NumberForm[N[P1213corner, 16], 16]];
Print["========================================================"];
Print[];


(* ================================================================ *)
(* Section 7. Summary and timing                                    *)
(* ================================================================ *)

Print["================================================================"];
Print["  CORNER-ADJACENT PROPAGATOR COMPLETE"];
Print["  Total time: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
