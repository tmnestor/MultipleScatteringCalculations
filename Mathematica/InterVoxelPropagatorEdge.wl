(* ::Package:: *)
(* InterVoxelPropagatorEdge.wl
   Analytical inter-voxel strain propagator for EDGE-adjacent cubes.
   R = (a, a, 0) with C_{2v} symmetry (x<->y mirror, z reflection).

   Extends InterVoxelPropagator.wl (face-adjacent) to the edge geometry.
   Two shifted tents M00(w1-1), M00(w2-1) plus one centered M00(w3).

   New features vs face:
   - Off-diagonal A_{12} != 0 (computed via step-function product, not delta collapse)
   - A_{33} requires double-shifted tent product tent(w1-1)*tent(w2-1)
   - 6 independent B_{ijkl} components (vs 3 for face)

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorEdge.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  INTER-VOXEL ELASTIC STRAIN PROPAGATOR — EDGE ADJACENT"];
Print["  R = (a, a, 0) with C_{2v} symmetry"];
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
(* Section 1. 2D Master integrals                                    *)
(*                                                                   *)
(* Reuse from face script: I2A, I2B, I2Ashifted, I2Bshifted.        *)
(* New: I2Adshift, I2Bdshift for double-shifted integrals.           *)
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

(* --- Single-shifted B-channel: shift on u --- *)
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

(* --- Double-shifted A-channel: shift on BOTH u and v --- *)
(* I2Adshift[p,q,c,s1,s2] = int_0^1 int_0^1 u^p v^q / sqrt(c^2+(u+s1)^2+(v+s2)^2) *)
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

(* --- Biharmonic 2D masters from identity: sqrt(C+u^2+v^2) = C/rho + u^2/rho + v^2/rho --- *)
(* I2Psi[p,q,c] = c^2 I2A[p,q,c] + I2A[p+2,q,c] + I2A[p,q+2,c] *)
ClearAll[I2Psi];
I2Psi[p_Integer, q_Integer, c_] :=
  c^2 * I2A[p, q, c] + I2A[p + 2, q, c] + I2A[p, q + 2, c];


(* ================================================================ *)
(* Section 2. 1D Elementary integrals (double-delta collapse)        *)
(* ================================================================ *)

Print["==== Section 2: 1D Elementary Integrals ===="];
Print[];

(* I1[C] = int_0^1 (1-t) sqrt(C + t^2) dt — for biharmonic kernel *)
ClearAll[I1exact];
I1exact[0] = 1/6;
I1exact[c2_] := Module[{sc1 = Sqrt[c2 + 1], sc = Sqrt[c2]},
  1/2 (sc1 + c2 * ArcSinh[1/sc]) - 1/3 (sc1^3 - sc^3)
];

(* I1A[C] = int_0^1 (1-t) / sqrt(C + t^2) dt — for Newton kernel *)
ClearAll[I1A];
I1A[c2_] := Module[{sc = Sqrt[c2], sc1 = Sqrt[c2 + 1]},
  ArcSinh[1/sc] - (sc1 - sc)
];

(* Validate *)
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
(* Section 3. Newton Potential Derivatives A_{jl} for R = (1,1,0)   *)
(*                                                                   *)
(* C_{2v} symmetry: A_{11}=A_{22}, A_{33}, A_{12}!=0, A_{13}=A_{23}=0 *)
(*                                                                   *)
(* Tent structure:                                                   *)
(*   w1: M00(w1-1) shifted, support [0,2]                          *)
(*   w2: M00(w2-1) shifted, support [0,2]                          *)
(*   w3: M00(w3)   centered, support [-1,1]                        *)
(* ================================================================ *)

Print["==== Section 3: Newton Potential Derivatives (A_jl) — Edge ===="];
Print[];

tentWeights = {+1, -2, +1};  (* delta coefficients for M00'' *)

(* --- A_{11}: d^2 Phi/dR1^2 --- delta collapse on w1 ---
   w1 delta points: {R1-1, R1, R1+1} = {0, 1, 2}, weights {+1,-2,+1}
   Remaining tents: M00(w2-1)*M00(w3)

   The remaining 2D integral has the SAME topology as face A_{22}:
   one shifted tent (w2-1) and one centered tent (w3).

   J_shifted(c) = int tent(w2-1)*tent(w3) / sqrt(c^2+w2^2+w3^2) dw2 dw3
   = 2 * [int_0^1 int_0^1 w2*(1-v)/rho(c,w2,v) dw2 dv
        + int_0^1 int_0^1 (1-s)*(1-v)/rho(c,s+1,v) ds dv]           *)

Print["--- A_{11} = A_{22} for R = (1,1,0) ---"];
Print["  (delta collapse on w1, remaining tent(w2-1)*tent(w3))"];

(* Precompute shifted integrals needed *)
Print["  Computing shifted integrals ..."];
Do[I2Ashifted[p, q, c, 1], {c, {0, 1, 2}}, {p, 0, 1}, {q, 0, 1}];

(* J_shifted(c): tent(w-1)*tent(v) integrated over [0,2]x[-1,1] *)
(* = 2 * [unshifted piece on [0,1]^2 + shifted piece on [0,1]^2] *)
Jshifted[c_] := 2 * (
  (* [0,1]^2: tent(w2-1)=w2, tent(w3)=1-v *)
  (I2A[1, 0, c] - I2A[1, 1, c])
  +
  (* [1,2] piece, sub s=w2-1: tent(w2-1)=1-s, tent(w3)=1-v *)
  (I2Ashifted[0, 0, c, 1] - I2Ashifted[1, 0, c, 1]
   - I2Ashifted[0, 1, c, 1] + I2Ashifted[1, 1, c, 1])
);

Do[
  Print["  Jshifted[", c, "] = ", NumberForm[N[Jshifted[c], 16], 16]];
  , {c, {0, 1, 2}}
];

A11edge = -1/(4 Pi) * Sum[
  tentWeights[[ic]] * Jshifted[{0, 1, 2}[[ic]]],
  {ic, 1, 3}
];
Print["  A_{11} = ", NumberForm[N[A11edge, 16], 16]];

(* By C_{2v}: A_{22} = A_{11} *)
A22edge = A11edge;
Print["  A_{22} = A_{11}  (C_{2v} symmetry)"];
Print[];


(* --- A_{33}: d^2 Phi/dR3^2 --- delta collapse on w3 ---
   w3 delta points: {R3-1, R3, R3+1} = {-1, 0, 1}, weights {+1,-2,+1}
   By |c| symmetry: sum = 2*J_dbl(1) - 2*J_dbl(0)
   Remaining tents: M00(w1-1)*M00(w2-1) — BOTH shifted!

   J_dbl(c) = int_{[0,2]^2} tent(w1-1)*tent(w2-1) / sqrt(w1^2+w2^2+c^2) dw1 dw2
   Split each [0,2] = [0,1]+[1,2]:
   = int_0^1 int_0^1 u*v/rho_00 + int_0^1 int_0^1 u*(1-t)/rho_01
     + int_0^1 int_0^1 (1-s)*v/rho_10 + int_0^1 int_0^1 (1-s)(1-t)/rho_11
   where rho_00=sqrt(u^2+v^2+c^2), rho_01=sqrt(u^2+(t+1)^2+c^2),
         rho_10=sqrt((s+1)^2+v^2+c^2), rho_11=sqrt((s+1)^2+(t+1)^2+c^2)
   Pieces (2) and (3) are equal by u<->v symmetry.                  *)

Print["--- A_{33} for R = (1,1,0) ---"];
Print["  (delta collapse on w3, remaining tent(w1-1)*tent(w2-1))"];

(* Precompute double-shifted integrals *)
Print["  Computing double-shifted integrals ..."];
Do[I2Adshift[p, q, c, 1, 1], {c, {0, 1}}, {p, 0, 1}, {q, 0, 1}];

(* J_dbl(c): double-shifted tent product *)
Jdbl[c_] := (
  (* Piece 1: [0,1]x[0,1], tent=u*v *)
  I2A[1, 1, c]
  +
  (* Pieces 2+3: shifted on one axis, tent=u*(1-t) or (1-s)*v *)
  (* Equal by symmetry; I2Ashifted[q,p,c,1] swaps u<->v shift *)
  2 * (I2Ashifted[0, 1, c, 1] - I2Ashifted[1, 1, c, 1])
  +
  (* Piece 4: double-shifted, tent=(1-s)*(1-t) *)
  (I2Adshift[0, 0, c, 1, 1] - I2Adshift[1, 0, c, 1, 1]
   - I2Adshift[0, 1, c, 1, 1] + I2Adshift[1, 1, c, 1, 1])
);

Do[
  Print["  Jdbl[", c, "] = ", NumberForm[N[Jdbl[c], 16], 16]];
  , {c, {0, 1}}
];

(* w3 deltas at {-1,0,1}, by |c| fold: 2*J(1) - 2*J(0) *)
A33edge = -1/(4 Pi) * (2 Jdbl[1] - 2 Jdbl[0]);
Print["  A_{33} = ", NumberForm[N[A33edge, 16], 16]];
Print[];


(* --- A_{12}: d^2 Phi/(dR1 dR2) --- DEFERRED ---
   A_{12} involves step-function products M00'(w1-1)*M00'(w2-1) with no
   delta-function collapse, requiring full 3D integration.  Mathematica's
   sequential Integrate produces incorrect results for this case (branch-cut
   artifacts in the outermost integral).

   SOLUTION: Use the exact Laplacian identity  A_{12} = Sum_k B_{12kk}
   = B_{1112} + B_{1222} + B_{1233}.  This is derived from nabla^2|w| = 2/|w|
   and avoids 3D integration entirely.  Each B value uses delta-function
   collapse (2D or 1D integrals) which Mathematica handles correctly.

   A_{12} will be computed after the B values in Section 4b below.       *)

Print["--- A_{12} for R = (1,1,0) ---"];
Print["  DEFERRED: will be computed from Laplacian identity A_{12} = Sum_k B_{12kk}"];
Print["  (3D sequential integration unreliable for step x step x 1/rho kernel)"];

(* Off-diagonals with z vanish *)
A13edge = 0;
A23edge = 0;
Print["  A_{13} = A_{23} = 0  (z-reflection symmetry)"];
Print[];


(* ================================================================ *)
(* Section 4. Biharmonic Fourth Derivatives B_{ijkl}                *)
(*                                                                   *)
(* Independent components under C_{2v} (1<->2, z-reflection):       *)
(*   B_{1111}=B_{2222}, B_{1112}=B_{1222}, B_{1122},               *)
(*   B_{1133}=B_{2233}, B_{1233}, B_{3333}                         *)
(* Components with odd number of 3-indices vanish.                  *)
(* ================================================================ *)

Print["==== Section 4: Biharmonic Fourth Derivatives (B_ijkl) — Edge ===="];
Print[];

(* --- B_{1122}: d^4 Psi/(dR1^2 dR2^2) --- DOUBLE DELTA COLLAPSE ---
   d^2/dR1^2 collapses w1 to {0,1,2} with alpha = {+1,-2,+1}.
   d^2/dR2^2 collapses w2 to {0,1,2} with beta = {+1,-2,+1}.
   Remaining: 1D integral in w3 with tent(w3), kernel |w|.

   d^4 Psi/(dR1^2 dR2^2) = Sum_{a,b} alpha_a * beta_b *
     2 * int_0^1 (1-t) sqrt(a^2+b^2+t^2) dt
   = Sum_{a,b} alpha_a * beta_b * 2 * I1exact[a^2+b^2]            *)

Print["--- B_{1122} (double delta collapse -> 1D) ---"];

aValsEdge = {0, 1, 2};    (* w1 eval points for R1=1 *)
bValsEdge = {0, 1, 2};    (* w2 eval points for R2=1 *)

B1122edge = -1/(8 Pi) * Sum[
  tentWeights[[ia]] * tentWeights[[ib]] *
    2 * I1exact[aValsEdge[[ia]]^2 + bValsEdge[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  B_{1122} = ", NumberForm[N[B1122edge, 16], 16]];
Print[];


(* --- B_{1133}: d^4 Psi/(dR1^2 dR3^2) --- DOUBLE DELTA COLLAPSE ---
   d^2/dR1^2 collapses w1 to {0,1,2}, alpha = {+1,-2,+1}.
   d^2/dR3^2 collapses w3 to {-1,0,1}, gamma = {+1,-2,+1}.
   Remaining: 1D in w2 with tent(w2-1), kernel |w|.

   The w2 tent is SHIFTED: support [0,2], tent(w2-1) = 1-|w2-1|.
   Split [0,1] (tent=w2) and [1,2] (tent=2-w2, sub s=w2-1 -> 1-s).

   1D integral at (a,c):
   J_tent1D(C) = int_0^2 tent(w2-1) sqrt(C+w2^2) dw2
   = int_0^1 w2 sqrt(C+w2^2) dw2 + int_0^1 (1-s) sqrt(C+(s+1)^2) ds *)

Print["--- B_{1133} (double delta collapse -> 1D, shifted tent) ---"];

Jpiece1[bigC_] := (1/3) ((bigC + 1)^(3/2) - bigC^(3/2));

ClearAll[Jpiece2memo];
Jpiece2memo[bigC_] := Jpiece2memo[bigC] = Module[{result},
  result = Integrate[(1 - s) Sqrt[bigC + (s + 1)^2], {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  Simplify[result]
];

Jtent1Dshifted[bigC_] := Jpiece1[bigC] + Jpiece2memo[bigC];

cValsEdge = {-1, 0, 1};   (* w3 eval points for R3=0 *)

B1133edge = -1/(8 Pi) * Sum[
  tentWeights[[ia]] * tentWeights[[ic]] *
    Jtent1Dshifted[aValsEdge[[ia]]^2 + cValsEdge[[ic]]^2],
  {ia, 1, 3}, {ic, 1, 3}
];
Print["  B_{1133} = ", NumberForm[N[B1133edge, 16], 16]];

(* By C_{2v}: B_{2233} = B_{1133} *)
B2233edge = B1133edge;
Print["  B_{2233} = B_{1133}  (C_{2v})"];
Print[];


(* --- B_{1111}: d^4 Psi/dR1^4 --- SAME-AXIS via delta'' ---
   M00''''(w1-1) gives delta'' at w1 = {0,1,2} with weights {+1,-2,+1}.
   delta''(w1-c) acts as f''(c) on the kernel.
   d^2/dw1^2[|w|] = (w2^2+w3^2)/(w1^2+w2^2+w3^2)^{3/2}
                   = 1/rho - w1^2/rho^3 = 1/rho - c^2/rho^3

   Remaining 2D integral: tent(w2-1)*tent(w3) * [1/rho - c^2/rho^3]
   = Jshifted(c) [A-channel] - JshiftedB(c) [B-channel]             *)

Print["--- B_{1111} (same-axis delta'', shifted tent) ---"];

(* B-channel shifted: tent(w-1)*tent(v) with c^2/rho^3 kernel *)
(* Same topology as Jshifted but with B-channel *)
Print["  Computing shifted B-channel integrals ..."];
Do[I2Bshifted[p, q, c, 1], {c, {1, 2}}, {p, 0, 1}, {q, 0, 1}];

JshiftedB[0] := 0;
JshiftedB[c_] := 2 * (
  (I2B[1, 0, c] - I2B[1, 1, c])
  +
  (I2Bshifted[0, 0, c, 1] - I2Bshifted[1, 0, c, 1]
   - I2Bshifted[0, 1, c, 1] + I2Bshifted[1, 1, c, 1])
);

B1111edge = -1/(8 Pi) * Sum[
  tentWeights[[ic]] * (Jshifted[{0, 1, 2}[[ic]]] - JshiftedB[{0, 1, 2}[[ic]]]),
  {ic, 1, 3}
];
Print["  B_{1111} = ", NumberForm[N[B1111edge, 16], 16]];

(* By C_{2v}: B_{2222} = B_{1111} *)
B2222edge = B1111edge;
Print["  B_{2222} = B_{1111}  (C_{2v})"];
Print[];


(* --- B_{3333}: d^4 Psi/dR3^4 --- SAME-AXIS via delta'' ---
   w3 delta points: {-1,0,1}, delta'' weights {+1,-2,+1}.
   d^2/dw3^2[|w|] = (w1^2+w2^2)/rho^3 = 1/rho - c^2/rho^3
   Remaining: tent(w1-1)*tent(w2-1) — double-shifted.

   Jdbl(c) for A-channel already defined.
   Need JdblB(c) for B-channel.                                     *)

Print["--- B_{3333} (same-axis delta'', double-shifted tent) ---"];

Print["  Computing double-shifted B-channel integrals ..."];
Do[I2Bdshift[p, q, c, 1, 1], {c, {1}}, {p, 0, 1}, {q, 0, 1}];

JdblB[0] := 0;
JdblB[c_] := (
  I2B[1, 1, c]
  + 2 * (I2Bshifted[0, 1, c, 1] - I2Bshifted[1, 1, c, 1])
  + (I2Bdshift[0, 0, c, 1, 1] - I2Bdshift[1, 0, c, 1, 1]
     - I2Bdshift[0, 1, c, 1, 1] + I2Bdshift[1, 1, c, 1, 1])
);

(* w3 deltas at {-1,0,1}, fold |c|: 2*J(1) - 2*J(0) *)
B3333edge = -1/(8 Pi) * (2 * (Jdbl[1] - JdblB[1]) - 2 * (Jdbl[0] - 0));
Print["  B_{3333} = ", NumberForm[N[B3333edge, 16], 16]];
Print[];


(* --- B_{1112}: d^4 Psi/(dR1^3 dR2) --- delta' on w1, step on w2 ---
   M00'''(w1-1) = -2 delta'(w1-1) + delta'(w1) + delta'(w1-2)
   M00'(w2-1) = +1 for w2 in (0,1), -1 for w2 in (1,2)
   M00(w3) = tent(w3)

   delta'(w1-c) acts as: int f(w1) delta'(w1-c) dw1 = -f'(c)
   d/dw1[|w|] = w1/|w|

   Contribution from delta'(w1-c) with weight alpha:
   alpha * (-1) * [c/rho(c,w2,w3)] * M00'(w2-1) * tent(w3)

   c=0 term: c/rho = 0 -> vanishes!
   c=1 term (weight -2): (-2)*(-1)*(1/rho(1,w2,w3))
   c=2 term (weight +1): (+1)*(-1)*(2/rho(2,w2,w3))

   So d^4Psi/(dR1^3 dR2) = int M00'(w2-1)*tent(w3)*[2/rho_1 - 2/rho_2] dw2 dw3

   The M00'(w2-1) step function splits w2 into [0,1] and [1,2].
   Fold w3 by even symmetry (factor 2).                              *)

Print["--- B_{1112} (delta' x step, 2D integrals) ---"];

(* Need: int_0^1 int_0^1 (1-v)/rho(c,t,v) dt dv — standard I2A *)
(* and:  int_0^1 int_0^1 (1-v)/rho(c,t+1,v) dt dv — shifted     *)

(* Tent-weighted A-channel with (1-v) only *)
I2A1mv[c_] := I2A[0, 0, c] - I2A[0, 1, c];
I2Ashifted1mv[c_] := I2Ashifted[0, 0, c, 1] - I2Ashifted[0, 1, c, 1];

(* The delta' at w1=a with weight alpha_a acts as:
     alpha_a * (-a/rho_a) on the biharmonic kernel |w|.
   This gives kernel coefficients:  a=0: 0,  a=1 (alpha=-2): 2/rho_1,  a=2 (alpha=+1): -2/rho_2.

   J1112(c) = 2D integral of M00'(w2-1)*tent(w3)*(c/rho_c) = 2*c*Delta_c
   where Delta_c = I2A1mv(c) - I2Ashifted1mv(c).

   d4Psi = sum_a (-alpha_a) * J1112(a) = -1*0 + 2*J1112(1) - 1*J1112(2)
   Note: coefficient on J1112(2) is 1, NOT 2, because J1112 already contains
   the factor c=2 from the delta' action.                                      *)

J1112[c_] := 2 * c * (I2A1mv[c] - I2Ashifted1mv[c]);

d4Psi1112 = 2 * J1112[1] - J1112[2];
B1112edge = -1/(8 Pi) * d4Psi1112;
Print["  B_{1112} = ", NumberForm[N[B1112edge, 16], 16]];

(* By C_{2v}: B_{1222} = B_{1112} *)
B1222edge = B1112edge;
Print["  B_{1222} = B_{1112}  (C_{2v})"];
Print[];


(* --- B_{1233}: d^4 Psi/(dR1 dR2 dR3^2) --- step x step x delta ---
   M00'(w1-1): step, +1 on [0,1], -1 on [1,2]
   M00'(w2-1): step, +1 on [0,1], -1 on [1,2]
   M00''(w3):  delta, -2 at w3=0, +1 at w3=-1, +1 at w3=+1

   The delta collapses w3 to {-1,0,1}. By |c| symmetry of kernel
   under w3 -> -w3: 2*I(1) - 2*I(0).

   At fixed w3=c, remaining 2D integral:
   int M00'(w1-1)*M00'(w2-1)*|w|_{w3=c} dw1 dw2
   = int (sign pattern) * sqrt(w1^2+w2^2+c^2) dw1 dw2

   Split into 4 quadrants:
   = int_0^1 int_0^1 sqrt(u^2+t^2+c^2) dudt
   - int_0^1 int_0^1 sqrt(u^2+(t+1)^2+c^2) dudt
   - int_0^1 int_0^1 sqrt((u+1)^2+t^2+c^2) dudt
   + int_0^1 int_0^1 sqrt((u+1)^2+(t+1)^2+c^2) dudt               *)

Print["--- B_{1233} (step x step x delta, biharmonic kernel) ---"];

(* Use identity: sqrt(C+u^2+v^2) = C/rho + u^2/rho + v^2/rho
   to express biharmonic integrals in terms of A-channel masters *)

(* Quadrant integrals for biharmonic kernel *)
(* (++) = I2Psi[0,0,c] = c^2*I2A[0,0,c] + 2*I2A[2,0,c] *)
(* (+-) = I2Psi_shifted_v[0,0,c,1] *)
(* (-+) = I2Psi_shifted_u[0,0,c,1] = (+-) by symmetry *)
(* (--) = I2Psi_dshift[0,0,c,1,1] *)

(* Biharmonic identity for SHIFTED integrals:
   sqrt(c^2 + (u+s1)^2 + v^2) = [c^2 + (u+s1)^2 + v^2] / rho
   = [c^2 + u^2 + 2*s1*u + s1^2 + v^2] / rho
   = (c^2+s1^2)/rho + u^2/rho + 2*s1*u/rho + v^2/rho            *)

I2Psishifted[c_, shift_] :=
  (c^2 + shift^2) * I2Ashifted[0, 0, c, shift] +
  I2Ashifted[2, 0, c, shift] + 2 shift * I2Ashifted[1, 0, c, shift] +
  I2Ashifted[0, 2, c, shift];

(* Double-shifted:
   sqrt(c^2 + (u+s1)^2 + (v+s2)^2)
   = (c^2+s1^2+s2^2)/rho + u^2/rho + 2*s1*u/rho + v^2/rho + 2*s2*v/rho *)

I2Psidshift[c_, s1_, s2_] :=
  (c^2 + s1^2 + s2^2) * I2Adshift[0, 0, c, s1, s2] +
  I2Adshift[2, 0, c, s1, s2] + 2 s1 * I2Adshift[1, 0, c, s1, s2] +
  I2Adshift[0, 2, c, s1, s2] + 2 s2 * I2Adshift[0, 1, c, s1, s2];

(* Need higher-degree shifted integrals *)
Print["  Computing degree-2 shifted and double-shifted integrals ..."];
Do[I2Ashifted[p, 0, c, 1], {c, {0, 1}}, {p, 0, 2}];
Do[I2Ashifted[0, p, c, 1], {c, {0, 1}}, {p, 2, 2}];
Do[I2Adshift[p, 0, c, 1, 1], {c, {0, 1}}, {p, 0, 2}];
Do[I2Adshift[0, p, c, 1, 1], {c, {0, 1}}, {p, 2, 2}];

J1233biharmonic[c_] := (
  I2Psi[0, 0, c]
  - 2 * I2Psishifted[c, 1]
  + I2Psidshift[c, 1, 1]
);

d4Psi1233 = 2 * J1233biharmonic[1] - 2 * J1233biharmonic[0];
B1233edge = -1/(8 Pi) * d4Psi1233;
Print["  B_{1233} = ", NumberForm[N[B1233edge, 16], 16]];
Print[];


(* ================================================================ *)
(* Section 4b. A_{12} from Laplacian identity                       *)
(*                                                                   *)
(* nabla^2|w| = 2/|w| implies Sum_k B_{ijkk} = A_{ij}.            *)
(* For ij=12: A_{12} = B_{1211} + B_{1222} + B_{1233}             *)
(*          = B_{1112} + B_{1222} + B_{1233}  (index symmetry)     *)
(* ================================================================ *)

Print["==== Section 4b: A_{12} from Laplacian Identity ===="];
Print[];

A12edge = B1112edge + B1222edge + B1233edge;
Print["  A_{12} = B_{1112} + B_{1222} + B_{1233}"];
Print["  A_{12} = ", NumberForm[N[A12edge, 16], 16]];
Print[];


(* ================================================================ *)
(* Section 5. Validation: Laplacian identities + FD cross-check     *)
(*                                                                   *)
(* Sum_k B_{ijkk} = A_{ij}  (from nabla^2|w| = 2/|w|)             *)
(* Checks 1-3: A_{11}, A_{22}, A_{33} (independent validation)     *)
(* Check 4: A_{12} (true by construction, validated by FD)          *)
(* ================================================================ *)

Print["==== Section 5: Laplacian Identity Checks ===="];
Print[];

(* Check 1: B_{1111} + B_{1122} + B_{1133} = A_{11} *)
lapl1 = N[B1111edge + B1122edge + B1133edge, 16];
Print["  B_{1111}+B_{1122}+B_{1133} = ", NumberForm[lapl1, 16]];
Print["  A_{11}                      = ", NumberForm[N[A11edge, 16], 16]];
Print["  |diff| = ", ScientificForm[Abs[lapl1 - N[A11edge, 16]], 3]];
Print[];

(* Check 2: B_{1122} + B_{2222} + B_{2233} = A_{22} = A_{11} *)
lapl2 = N[B1122edge + B2222edge + B2233edge, 16];
Print["  B_{1122}+B_{2222}+B_{2233} = ", NumberForm[lapl2, 16]];
Print["  A_{22}                      = ", NumberForm[N[A22edge, 16], 16]];
Print["  |diff| = ", ScientificForm[Abs[lapl2 - N[A22edge, 16]], 3]];
Print[];

(* Check 3: B_{1133} + B_{2233} + B_{3333} = A_{33} *)
lapl3 = N[B1133edge + B2233edge + B3333edge, 16];
Print["  B_{1133}+B_{2233}+B_{3333} = ", NumberForm[lapl3, 16]];
Print["  A_{33}                      = ", NumberForm[N[A33edge, 16], 16]];
Print["  |diff| = ", ScientificForm[Abs[lapl3 - N[A33edge, 16]], 3]];
Print[];

(* Check 4: A_{12} = B_{1112}+B_{1222}+B_{1233} — true by construction *)
Print["  Check 4: A_{12} = Sum_k B_{12kk} — true by construction"];
Print["  A_{12} = ", NumberForm[N[A12edge, 16], 16]];
Print[];


(* ================================================================ *)
(* Section 6. Static Propagator Assembly — Edge Adjacent             *)
(*                                                                   *)
(* P_{ijkl}(R) = -1/(2mu) [delta_ik A_jl + delta_jk A_il           *)
(*                          - 2 eta B_ijkl]                         *)
(* ================================================================ *)

Print["==== Section 6: Static Propagator Assembly (Edge-Adjacent) ===="];
Print[];

(* P_{1111}: delta_11*A_11 + delta_11*A_11 - 2eta*B_1111 *)
P1111edge = -1/(2 mu) (2 A11edge - 2 eta B1111edge);

(* P_{2222} = P_{1111} by C_{2v} *)
P2222edge = P1111edge;

(* P_{3333}: delta_33*A_33 + delta_33*A_33 - 2eta*B_3333 *)
P3333edge = -1/(2 mu) (2 A33edge - 2 eta B3333edge);

(* P_{1122}: delta_12=0, delta_12=0, B_1122 *)
P1122edge = -1/(2 mu) (-2 eta B1122edge);

(* P_{1133}: delta_13=0, delta_13=0, B_1133 *)
P1133edge = -1/(2 mu) (-2 eta B1133edge);

(* P_{2233} = P_{1133} by C_{2v} *)
P2233edge = P1133edge;

(* P_{1212}: i=k=1,j=l=2. delta_11*A_22 + delta_21*A_12 - 2eta*B_1212
   delta_11=1, delta_21=0. B_1212 = B_1122 (index symmetry).
   P_{1212} = -1/(2mu)[A_{22} - 2eta*B_{1122}] *)
P1212edge = -1/(2 mu) (A22edge - 2 eta B1122edge);

(* P_{1313}: i=k=1,j=l=3. delta_11*A_33 + delta_31*A_13 - 2eta*B_1313
   delta_11=1, delta_31=0, A_13=0. B_1313 = B_1133.
   P_{1313} = -1/(2mu)[A_{33} - 2eta*B_{1133}] *)
P1313edge = -1/(2 mu) (A33edge - 2 eta B1133edge);

(* P_{2323} = P_{1313} by C_{2v} *)
P2323edge = P1313edge;

(* P_{1112}: i=j=k=1,l=2. delta_11*A_12 + delta_11*A_12 - 2eta*B_1112
   = -1/(2mu)[2*A_12 - 2eta*B_1112] *)
P1112edge = -1/(2 mu) (2 A12edge - 2 eta B1112edge);

(* P_{1233}: i=k=1,j=2,l=3 -> wait, need to be careful.
   P_{1233}: using formula P_{ijkl} = -1/(2mu)[delta_ik A_jl + delta_jk A_il - 2eta B_ijkl]
   i=1,j=2,k=3,l=3: delta_13*A_23 + delta_23*A_13 - 2eta*B_1233
   = 0 + 0 - 2eta*B_1233/(2mu) *)
P1233edge = -1/(2 mu) (-2 eta B1233edge);

Print["========================================================"];
Print["  STATIC STRAIN PROPAGATOR — Edge Adjacent R = (a,a,0)"];
Print["========================================================"];
Print["  P_{1111} = ", NumberForm[N[P1111edge, 16], 16]];
Print["  P_{2222} = ", NumberForm[N[P2222edge, 16], 16], "  (= P_{1111})"];
Print["  P_{3333} = ", NumberForm[N[P3333edge, 16], 16]];
Print["  P_{1122} = ", NumberForm[N[P1122edge, 16], 16]];
Print["  P_{1133} = ", NumberForm[N[P1133edge, 16], 16]];
Print["  P_{2233} = ", NumberForm[N[P2233edge, 16], 16], "  (= P_{1133})"];
Print["  P_{1212} = ", NumberForm[N[P1212edge, 16], 16]];
Print["  P_{1313} = ", NumberForm[N[P1313edge, 16], 16]];
Print["  P_{2323} = ", NumberForm[N[P2323edge, 16], 16], "  (= P_{1313})"];
Print["  P_{1112} = ", NumberForm[N[P1112edge, 16], 16]];
Print["  P_{1233} = ", NumberForm[N[P1233edge, 16], 16]];
Print["========================================================"];
Print[];


(* ================================================================ *)
(* Section 7. Cross-validation via 3D NIntegrate + finite diffs     *)
(* ================================================================ *)

Print["==== Section 7: Finite-Difference Cross-Validation ===="];
Print[];

Phi00edge[{Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
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
R0edge = {1.0, 1.0, 0.0};

Print["  Computing FD derivatives of Phi at R=(1,1,0) ..."];

d2Phi11FD = (Phi00edge[R0edge + {deltaFD,0,0}] - 2 Phi00edge[R0edge] +
             Phi00edge[R0edge - {deltaFD,0,0}]) / deltaFD^2;
d2Phi33FD = (Phi00edge[R0edge + {0,0,deltaFD}] - 2 Phi00edge[R0edge] +
             Phi00edge[R0edge - {0,0,deltaFD}]) / deltaFD^2;
d2Phi12FD = (Phi00edge[R0edge + {deltaFD,deltaFD,0}]
           - Phi00edge[R0edge + {deltaFD,-deltaFD,0}]
           - Phi00edge[R0edge + {-deltaFD,deltaFD,0}]
           + Phi00edge[R0edge + {-deltaFD,-deltaFD,0}]) / (4 deltaFD^2);

A11FD = -d2Phi11FD / (4 Pi);
A33FD = -d2Phi33FD / (4 Pi);
A12FD = -d2Phi12FD / (4 Pi);

Print["  A_{11}: analytical = ", NumberForm[N[A11edge, 10], 10],
      "   FD = ", NumberForm[A11FD, 10],
      "   |D| = ", ScientificForm[Abs[N[A11edge, 16] - A11FD], 3]];
Print["  A_{33}: analytical = ", NumberForm[N[A33edge, 10], 10],
      "   FD = ", NumberForm[A33FD, 10],
      "   |D| = ", ScientificForm[Abs[N[A33edge, 16] - A33FD], 3]];
Print["  A_{12}: analytical = ", NumberForm[N[A12edge, 10], 10],
      "   FD = ", NumberForm[A12FD, 10],
      "   |D| = ", ScientificForm[Abs[N[A12edge, 16] - A12FD], 3]];
Print[];


(* ================================================================ *)
(* Section 8. Summary and timing                                    *)
(* ================================================================ *)

Print["================================================================"];
Print["  EDGE-ADJACENT PROPAGATOR COMPLETE"];
Print["  Total time: ", Round[AbsoluteTime[] - t0global, 0.1], " s"];
Print["================================================================"];
