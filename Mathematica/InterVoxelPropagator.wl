(* ::Package:: *)
(* InterVoxelPropagator.wl
   Analytical inter-voxel strain propagator via delta-function collapse.

   R-derivatives of tent functions produce deltas that collapse 3D Fourier
   integrals to 2D master integrals (sequential Integrate) or 1D elementary
   integrals (double-delta collapse).  Replaces WS1-3 finite differences
   with 14+ digit analytical results.

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagator.wl

   Outputs:
     - Console log of every integral, validation check, and propagator component.
     - InterVoxelPropagatorValues.wl with all numerical results.
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  INTER-VOXEL ELASTIC STRAIN PROPAGATOR"];
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
Print["  rho = ", rho0, ", cs = ", cs // N, ", cp = ", cp // N];
Print["  a = ", a, ", h = ", h];
Print[];


(* ================================================================ *)
(* Section 1. 2D Master integrals                                    *)
(*                                                                   *)
(* I2A[p,q,c] = int_0^1 int_0^1 u^p v^q / sqrt(c^2+u^2+v^2) du dv *)
(* I2B[p,q,c] = int_0^1 int_0^1 u^p v^q c^2/(c^2+u^2+v^2)^{3/2}   *)
(* I2Psi[p,q,c] = int_0^1 int_0^1 u^p v^q sqrt(c^2+u^2+v^2) du dv *)
(*                                                                   *)
(* Extended: I2A[p,q,c,L1,L2] for non-unit bounds.                  *)
(* Canonicalize: I2A[p,q,c] with p >= q (swap symmetry in u,v).     *)
(* ================================================================ *)

Print["==== Section 1: 2D Master Integrals ===="];
Print[];

(* --- A-channel: 1/rho kernel --- *)
ClearAll[I2A];

(* Canonicalization: swap p,q if p < q (symmetric in u,v) *)
I2A[p_Integer, q_Integer, c_] /; p < q := I2A[q, p, c];

(* Unit-square memoized *)
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

(* Extended bounds *)
ClearAll[I2Aext];
I2Aext[p_Integer, q_Integer, c_, L1_, L2_] :=
  I2Aext[p, q, c, L1, L2] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q / Sqrt[c^2 + u^2 + v^2],
           {v, 0, L2}, Assumptions -> u > 0 && c >= 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, L1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Aext[", p, ",", q, ",", c, ",", L1, ",", L2, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- B-channel: c^2/rho^3 kernel --- *)
ClearAll[I2B];

I2B[p_Integer, q_Integer, c_] /; p < q := I2B[q, p, c];

(* c=0 case: kernel vanishes identically *)
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

(* Extended bounds for B-channel *)
ClearAll[I2Bext];
I2Bext[p_Integer, q_Integer, c_, L1_, L2_] :=
  I2Bext[p, q, c, L1, L2] = Module[{iv, iuv, t0r, result},
    t0r = AbsoluteTime[];
    iv = Integrate[u^p * v^q * c^2 / (c^2 + u^2 + v^2)^(3/2),
           {v, 0, L2}, Assumptions -> u > 0 && c > 0,
           GenerateConditions -> False];
    iuv = Integrate[iv, {u, 0, L1}, GenerateConditions -> False];
    result = Simplify[iuv];
    Print["  I2Bext[", p, ",", q, ",", c, ",", L1, ",", L2, "] = ",
          NumberForm[N[result, 20], 16],
          "  (", Round[AbsoluteTime[] - t0r, 0.1], " s)"];
    result
  ];

(* --- Biharmonic: sqrt(c^2+u^2+v^2) kernel --- *)
(* Identity: sqrt(C+u^2+v^2) = C/sqrt(C+u^2+v^2) + (u^2+v^2)/sqrt(C+u^2+v^2)
   So I2Psi[p,q,c] = c^2 * I2A[p,q,c] (with 1/rho)
                    + I2A[p+2,q,c] + I2A[p,q+2,c]
   when c != 0.  This reduces biharmonic to A-channel masters. *)
ClearAll[I2Psi];
I2Psi[p_Integer, q_Integer, c_] :=
  c^2 * I2A[p, q, c] + I2A[p + 2, q, c] + I2A[p, q + 2, c];


(* --- Compute all needed 2D masters --- *)
(* For face-adjacent R=(1,0,0): delta on w1, tent on (w2,w3) symmetric.
   Tent weights at c in {0, 1, 2} with coefficients {+1, -2, +1}.
   Polynomial degrees (p,q) in {0,1} for the tent product. *)

Print["--- Computing A-channel masters for c in {0,1,2} ---"];
Do[I2A[p, q, c], {c, {0, 1, 2}}, {p, 0, 2}, {q, 0, p}];

Print[];
Print["--- Computing B-channel masters for c in {1,2} ---"];
Do[I2B[p, q, c], {c, {1, 2}}, {p, 0, 2}, {q, 0, p}];

Print[];
Print["--- Computing higher-degree A-channel for biharmonic ---"];
Do[I2A[p, q, c], {c, {0, 1, 2}}, {p, 0, 4}, {q, 0, Min[p, 2]}];

(* --- Validate each symbolic result against NIntegrate --- *)
Print[];
Print["--- Validation: symbolic vs NIntegrate ---"];

validationPairs = {};
Do[
  sym = I2A[p, q, c];
  If[sym === 0, Continue[]];
  nval = NIntegrate[u^p * v^q / Sqrt[c^2 + u^2 + v^2],
           {u, 0, 1}, {v, 0, 1},
           WorkingPrecision -> 25, PrecisionGoal -> 16];
  err = Abs[N[sym, 25] - nval];
  status = If[err < 10^-12, "OK", "FAIL"];
  AppendTo[validationPairs, {{"I2A", p, q, c}, err, status}];
  If[status == "FAIL",
    Print["  FAIL: I2A[", p, ",", q, ",", c, "] err = ", ScientificForm[err, 3]]];
  ,
  {c, {0, 1, 2}}, {p, 0, 4}, {q, 0, Min[p, 2]}
];

Do[
  sym = I2B[p, q, c];
  If[sym === 0, Continue[]];
  nval = NIntegrate[u^p * v^q * c^2 / (c^2 + u^2 + v^2)^(3/2),
           {u, 0, 1}, {v, 0, 1},
           WorkingPrecision -> 25, PrecisionGoal -> 16];
  err = Abs[N[sym, 25] - nval];
  status = If[err < 10^-12, "OK", "FAIL"];
  AppendTo[validationPairs, {{"I2B", p, q, c}, err, status}];
  If[status == "FAIL",
    Print["  FAIL: I2B[", p, ",", q, ",", c, "] err = ", ScientificForm[err, 3]]];
  ,
  {c, {1, 2}}, {p, 0, 2}, {q, 0, p}
];

nFail = Count[validationPairs, {_, _, "FAIL"}];
Print["  Validated ", Length[validationPairs], " integrals, ",
      nFail, " failures."];
Print[];


(* ================================================================ *)
(* Section 2. 1D Elementary integrals (double-delta collapse)        *)
(*                                                                   *)
(* When two delta functions collapse two of the three w-dimensions,  *)
(* the remaining integral is 1D:                                     *)
(*   I1[C] = int_0^1 (1-t) sqrt(C + t^2) dt                        *)
(* where (1-t) is the remaining tent function value.                 *)
(*                                                                   *)
(* Closed form:                                                      *)
(*   I1[C] = (1/2)[sqrt(C+1) + C*ArcSinh[1/sqrt(C)]]               *)
(*         - (1/3)[(C+1)^{3/2} - C^{3/2}]                           *)
(* with I1[0] = 1/2 - 1/3 = 1/6.                                    *)
(* ================================================================ *)

Print["==== Section 2: 1D Elementary Integrals ===="];
Print[];

ClearAll[I1exact];
I1exact[0] = 1/6;
I1exact[c2_] := Module[{sc1 = Sqrt[c2 + 1], sc = Sqrt[c2]},
  1/2 (sc1 + c2 * ArcSinh[1/sc]) - 1/3 (sc1^3 - sc^3)
];

(* Validate *)
Print["--- I1[C] = int_0^1 (1-t) sqrt(C+t^2) dt ---"];
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

(* Also need the A-channel variant for double-delta collapse *)
(* I1A[C] = int_0^1 (1-t) / sqrt(C + t^2) dt *)
ClearAll[I1A];
I1A[0] = 1;  (* = int_0^1 (1-t)/t dt diverges... need to reconsider *)
(* Actually for C=0: 1/sqrt(t^2) = 1/t, and int_0^1 (1-t)/t dt diverges.
   But this case only arises when both other dimensions have c=0,
   which means R=0 (self-interaction) — not an inter-voxel case. *)
I1A[c2_] := Module[{sc = Sqrt[c2], sc1 = Sqrt[c2 + 1]},
  ArcSinh[1/sc] - (sc1 - sc)
];

Print["--- I1A[C] = int_0^1 (1-t) / sqrt(C+t^2) dt ---"];
Do[
  exact = I1A[c2];
  nval = NIntegrate[(1 - t)/Sqrt[c2 + t^2], {t, 0, 1},
           WorkingPrecision -> 25, PrecisionGoal -> 16];
  err = Abs[N[exact, 25] - nval];
  Print["  I1A[", c2, "] = ", NumberForm[N[exact, 16], 16],
        "  err = ", ScientificForm[err, 3],
        If[err < 10^-12, "  OK", "  FAIL"]];
  ,
  {c2, {1, 2, 4, 5}}
];
Print[];


(* ================================================================ *)
(* Section 3. Newton Potential Derivatives A_{jl}                    *)
(*                                                                   *)
(* A_{jl}(R) = -d^2[Phi/(4pi)] / dR_j dR_l                         *)
(* where Phi(R) = int_A int_B 1/|s-s'+R| d^3s d^3s'                *)
(*                                                                   *)
(* The tent function M''(d) = -2 delta(d) + delta(d+1) + delta(d-1) *)
(* collapses the 3D w-integral to 2D.                               *)
(*                                                                   *)
(* For face-adjacent R = (a,0,0) = (1,0,0):                         *)
(*   The tent in w1 has support centered at R1=1.                    *)
(*   M(w1-1) is nonzero for w1 in [0,2], tent peaks at w1=1.        *)
(*   d^2/dR1^2 acts on M(w1-R1) -> M''(w1-R1) = delta weights.     *)
(*   The tents in w2,w3 are centered at 0, support [-1,1].          *)
(*                                                                   *)
(* The 3D integral factorizes after delta collapse.                  *)
(* ================================================================ *)

Print["==== Section 3: Newton Potential Derivatives (A_jl) ===="];
Print[];

(* --- Tent function cross-correlation ---
   M00(d) = int_{overlap} phi(s) phi(s-d) ds  where phi(s) = indicator[-h,h]
   For h=1/2: M00(d) = 1 - |d|  for |d| <= 1, 0 otherwise.
   The second derivative is M00''(d) = -2 delta(d) + delta(d-1) + delta(d+1).

   When we differentiate d^2 Phi/dR_j^2, with Phi containing M00(w_j - R_j),
   we get Sum_{c} alpha[c] * (2D integral at w_j = c) where:
     c = R_j - 1, R_j, R_j + 1   with alpha = {+1, -2, +1}              *)

tentWeights = {+1, -2, +1};  (* delta coefficients for M00'' *)

(* For face-adjacent R = {1,0,0}:
   delta on w1 (differentiated direction):
     w1 values = {R1-1, R1, R1+1} = {0, 1, 2}
   tent on w2: M00(w2) = 1-|w2| for w2 in [-1,1]
     -> by even symmetry, fold to 2 * int_0^1 (1-w2) [integrand] dw2
   tent on w3: same as w2 *)

(* --- A_{11}: d^2/dR1^2 of Phi/(4pi) ---
   Delta collapse in w1 -> 2D integral in (w2, w3) with tent product.
   w1 evaluates at c in {0,1,2}, weighted by {+1,-2,+1}.
   The (w2,w3) tent product T(w2)T(w3) = (1-|w2|)(1-|w3|)
   is even in both w2,w3, so fold to [0,1]^2 with factor 4.

   A_{11} = -1/(4pi) * d^2 Phi/dR1^2
          = -1/(4pi) * 4 * Sum_c alpha[c] * I2A_tent[c]

   where I2A_tent[c] = int_0^1 int_0^1 (1-u)(1-v) / sqrt(c^2+u^2+v^2) du dv
                      = I2A[0,0,c] - 2*I2A[1,0,c] + I2A[1,1,c]           *)

Print["--- A_{11} for R = (1,0,0) ---"];

I2Atent[c_] := I2A[0, 0, c] - 2 I2A[1, 0, c] + I2A[1, 1, c];

(* Compute tent-weighted 2D integrals *)
Do[
  Print["  I2Atent[", c, "] = ", NumberForm[N[I2Atent[c], 16], 16]];
  , {c, {0, 1, 2}}
];

(* Full A_{11} with delta weights and octant factor *)
(* Factor breakdown: 4 from (w2,w3) symmetry fold to [0,1]^2,
   and -1/(4pi) from the Phi -> A relation *)
A11face = -1/(4 Pi) * 4 * Sum[
  tentWeights[[ic]] * I2Atent[{0, 1, 2}[[ic]]],
  {ic, 1, 3}
];
Print["  A_{11} = ", NumberForm[N[A11face, 16], 16]];
Print[];

(* --- A_{22}: d^2/dR2^2 of Phi/(4pi) ---
   Delta collapse in w2 -> 2D integral in (w1, w3).
   w2 evaluates at c2 in {R2-1, R2, R2+1} = {-1, 0, 1} for R2=0.
   But M00(w2-R2) = M00(w2) has support [-1,1].
   M00''(w2) puts deltas at w2 = -1, 0, +1.
   At w2=+/-1: M00 tent = 0 (tent vanishes at its boundary).
   So effectively only w2=0 contributes with weight -2.

   Wait — more carefully: the second derivative of the CROSS-CORRELATION
   Phi(R) involves the second derivative acting on the w2-R2 tent.
   d^2/dR2^2 M00(w2-R2) evaluated at specific w2 values gives:
     w2 = R2+1 = 1:  weight +1, tent_w1(w1-1) * tent_w3(w3-0)
     w2 = R2   = 0:  weight -2, tent_w1(w1-1) * tent_w3(w3-0)
     w2 = R2-1 = -1: weight +1, tent_w1(w1-1) * tent_w3(w3-0)

   The (w1,w3) integrand has tent(w1-1)*tent(w3)/sqrt(c^2+w1^2+w3^2)
   where c = w2 value = {-1, 0, +1}.

   tent(w1-1) is nonzero for w1 in [0,2], peaks at w1=1.
   tent(w3) is nonzero for w3 in [-1,1], peaks at w3=0.

   By |c| symmetry: c=-1 and c=+1 give the same integral.
   So the sum is: +1*I(1) + (-2)*I(0) + (+1)*I(1) = 2*I(1) - 2*I(0)

   Each I(c) = int_0^2 int_{-1}^1 tent(w1-1)*tent(w3)/sqrt(c^2+w1^2+w3^2) dw1 dw3

   Substituting u=w1 (u in [0,2], tent(u-1) = 1-|u-1|)
   and v=|w3| (fold by even symmetry, factor 2):

   I(c) = 2 * int_0^2 int_0^1 (1-|u-1|)(1-v)/sqrt(c^2+u^2+v^2) du dv

   Split u in [0,1] and [1,2]:
     [0,1]: tent(u-1) = u,     so integrand has u*(1-v)/rho
     [1,2]: tent(u-1) = 2-u,   let u'=u-1: (1-u')*(1-v)/rho(c,u'+1,v)
                                 = (1-u')*(1-v)/sqrt(c^2+(u'+1)^2+v^2) for u' in [0,1]
   This is NOT a simple I2A with standard bounds. Let's compute directly. *)

Print["--- A_{22} for R = (1,0,0) ---"];

(* Direct 2D integration for A_{22}. The tent product in (w1,w3) is:
   T(w1,w3) = tent(w1-1) * tent(w3)
   with tent(x) = max(0, 1-|x|) and h=1/2 -> tent period = 1.

   Actually let me reconsider. With h=a/2=1/2, the tent function is:
   M00(d) = max(0, 1 - |d|/h) * h  ... no.
   M00(d) = max(0, 2h - |d|) for the cross-correlation of two boxes of width 2h.
   With h=1/2: M00(d) = max(0, 1 - |d|).

   So d^2 M00/dd^2 = -2 delta(d) + delta(d-1) + delta(d+1), OK.

   For the (w1,w3) integral at fixed w2=c:
   The remaining tent factors are M00(w1-R1)*M00(w3-R3)
   = M00(w1-1)*M00(w3-0) = (1-|w1-1|)(1-|w3|)
   for |w1-1| < 1 and |w3| < 1.

   So the 2D integral at w2=c is:
   J(c) = int_{w1=0}^{2} int_{w3=-1}^{1} (1-|w1-1|)(1-|w3|)/sqrt(c^2+w1^2+w3^2) dw1 dw3

   Fold w3 by symmetry (factor 2), substitute t=|w3|:
   J(c) = 2 * int_0^2 int_0^1 (1-|w1-1|)(1-t)/sqrt(c^2+w1^2+t^2) dw1 dt

   Split w1 into [0,1] (where 1-|w1-1|=w1) and [1,2] (where 1-|w1-1|=2-w1):
   J(c) = 2 * [int_0^1 int_0^1 w1*(1-t)/sqrt(c^2+w1^2+t^2) dw1 dt
             + int_1^2 int_0^1 (2-w1)*(1-t)/sqrt(c^2+w1^2+t^2) dw1 dt]

   For the first piece: this is a 2D integral with polynomial w1*(1-t) on [0,1]^2
   = I2A[1,0,c] - I2A[1,1,c]  (with u=w1, v=t)

   For the second piece, substitute s = w1 - 1 (s in [0,1]):
   int_0^1 int_0^1 (1-s)*(1-t)/sqrt(c^2+(s+1)^2+t^2) ds dt
   This is a SHIFTED integral — not expressible as simple I2A.
   We need a dedicated function for this. *)

(* Shifted 2D integral: compute directly *)
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

(* J(c) for A_{22} *)
J22[c_] := 2 * (
  (* [0,1]^2 piece: w1*(1-t) = u*(1-v) *)
  (I2A[1, 0, c] - I2A[1, 1, c])
  +
  (* [1,2] piece, shifted: (1-s)*(1-t) *)
  (I2Ashifted[0, 0, c, 1] - I2Ashifted[1, 0, c, 1]
   - I2Ashifted[0, 1, c, 1] + I2Ashifted[1, 1, c, 1])
);

Print["  Computing shifted integrals for A_{22} ..."];
Do[I2Ashifted[p, q, c, 1], {c, {0, 1}}, {p, 0, 1}, {q, 0, 1}];

(* A_{22} = -1/(4pi) * Sum_c alpha[c] * J22[|c|]
   c values: {-1, 0, +1}, by |c| symmetry: 2*J22[1] - 2*J22[0] ... no.
   alpha = {+1, -2, +1} for c = {-1, 0, 1}.
   J22[c] depends on |c| only (rho is even in c).
   So sum = (+1)*J22[1] + (-2)*J22[0] + (+1)*J22[1] = 2*J22[1] - 2*J22[0]. *)

A22face = -1/(4 Pi) * (2 J22[1] - 2 J22[0]);
Print["  A_{22} = ", NumberForm[N[A22face, 16], 16]];

(* By C_4v symmetry of R along [100]: A_{33} = A_{22} *)
A33face = A22face;
Print["  A_{33} = A_{22} = ", NumberForm[N[A33face, 16], 16]];

(* Off-diagonals vanish by symmetry *)
A12face = 0;
A13face = 0;
A23face = 0;
Print["  A_{12} = A_{13} = A_{23} = 0  (by C_4v symmetry)"];
Print[];

(* --- Validate A_jl against WS2 finite-difference computation ---
   WS2 computes Phi(R) by 3D NIntegrate then uses finite differences.
   Our analytical values should agree to ~6 digits (WS2 precision). *)

Print["--- Validation: A_jl via direct 3D NIntegrate + finite differences ---"];

(* Newton potential *)
Phi00[{Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
  NIntegrate[
    Max[0, 1 - Abs[w1 - Rx]] Max[0, 1 - Abs[w2 - Ry]] Max[0, 1 - Abs[w3 - Rz]] /
      Sqrt[w1^2 + w2^2 + w3^2],
    {w1, Rx - 1, Rx + 1},
    {w2, Ry - 1, Ry + 1},
    {w3, Rz - 1, Rz + 1},
    Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"},
    WorkingPrecision -> 16, MaxRecursion -> 15, PrecisionGoal -> 10
  ];

delta = 0.0005;
R0 = {1.0, 0.0, 0.0};

d2Phi11FD = (Phi00[R0 + {delta,0,0}] - 2 Phi00[R0] + Phi00[R0 - {delta,0,0}]) / delta^2;
d2Phi22FD = (Phi00[R0 + {0,delta,0}] - 2 Phi00[R0] + Phi00[R0 - {0,delta,0}]) / delta^2;

A11FD = -d2Phi11FD / (4 Pi);
A22FD = -d2Phi22FD / (4 Pi);

Print["  A_{11} analytical = ", NumberForm[N[A11face, 12], 12]];
Print["  A_{11} FD (WS2)   = ", NumberForm[A11FD, 12]];
Print["  |difference|       = ", ScientificForm[Abs[N[A11face, 16] - A11FD], 3]];
Print[];
Print["  A_{22} analytical = ", NumberForm[N[A22face, 12], 12]];
Print["  A_{22} FD (WS2)   = ", NumberForm[A22FD, 12]];
Print["  |difference|       = ", ScientificForm[Abs[N[A22face, 16] - A22FD], 3]];
Print[];


(* ================================================================ *)
(* Section 4. Biharmonic Potential Fourth Derivatives B_{ijkl}       *)
(*                                                                   *)
(* B_{ijkl}(R) = -1/(8pi) d^4 Psi / dR_i dR_j dR_k dR_l           *)
(* where Psi(R) = int_A int_B |s-s'+R| d^3s d^3s'                  *)
(*                                                                   *)
(* Key identity for the kernel under delta collapse:                 *)
(*   d^2|w|/dw1^2 = (w2^2+w3^2)/|w|^3 = 1/|w| - w1^2/|w|^3       *)
(*                                                                   *)
(* So B_{ijkl} decomposes into A-channel (1/rho) and B-channel      *)
(* (c^2/rho^3) contributions.                                       *)
(* ================================================================ *)

Print["==== Section 4: Biharmonic Fourth Derivatives (B_ijkl) ===="];
Print[];

(* --- B_{1111}: d^4 Psi / dR1^4 ---
   Two d^2/dR1^2 applications on the w1-tent.
   First d^2 gives delta collapse: w1 in {0,1,2} with weights {+1,-2,+1}.
   Second d^2 acts on the 2D kernel d^2|w|/dw1^2 = (u^2+v^2)/(c^2+u^2+v^2)^{3/2}
                                                   = 1/rho - c^2/rho^3

   So B_{1111} = -1/(8pi) * 4 * Sum_c alpha[c] * [I2A_tent[c] - I2B_tent[c]]

   where I2B_tent[c] has the c^2/rho^3 kernel with tent product (1-u)(1-v). *)

I2Btent[c_] := I2B[0, 0, c] - 2 I2B[1, 0, c] + I2B[1, 1, c];

Print["--- B_{1111} for R = (1,0,0) ---"];
Do[
  Print["  I2Btent[", c, "] = ", NumberForm[N[I2Btent[c], 16], 16]];
  , {c, {0, 1, 2}}
];

B1111face = -1/(8 Pi) * 4 * Sum[
  tentWeights[[ic]] * (I2Atent[{0, 1, 2}[[ic]]] - I2Btent[{0, 1, 2}[[ic]]]),
  {ic, 1, 3}
];
Print["  B_{1111} = ", NumberForm[N[B1111face, 16], 16]];
Print[];

(* --- B_{1122}: d^4 Psi / dR1^2 dR2^2 --- (double delta collapse)
   d^2/dR1^2 collapses w1 to {0,1,2} with weights alpha = {+1,-2,+1}.
   d^2/dR2^2 collapses w2 to {-1,0,+1} with weights beta = {+1,-2,+1}.
   This leaves a 1D integral in w3 with tent(w3) = (1-|w3|).

   The kernel after both collapses:
   d^2|w|/dw1^2 was already differentiated in w1.
   Now d^2/dw2^2 of the remaining 2D piece gives a 1D integral.

   Actually, let me think about this more carefully.
   Psi = int M(w1-R1) M(w2-R2) M(w3-R3) |w| d^3w

   d^4 Psi/dR1^2 dR2^2 = int M''(w1-R1) M''(w2-R2) M(w3-R3) |w| d^3w

   Both M'' insert delta sums:
   = Sum_{a in {0,1,2}} Sum_{b in {-1,0,1}} alpha_a * beta_b *
     int_{-1}^{1} (1-|w3|) * sqrt(a^2 + b^2 + w3^2) dw3

   = Sum_{a,b} alpha_a * beta_b * 2 * int_0^1 (1-t) sqrt(a^2+b^2+t^2) dt

   = Sum_{a,b} alpha_a * beta_b * 2 * I1[a^2 + b^2]                      *)

Print["--- B_{1122} for R = (1,0,0) (double-delta collapse -> 1D) ---"];

alphaWeights = {+1, -2, +1};    (* for w1 at {0,1,2} *)
betaWeights = {+1, -2, +1};     (* for w2 at {-1,0,1} *)
aVals = {0, 1, 2};              (* w1 evaluation points *)
bVals = {-1, 0, 1};             (* w2 evaluation points *)

B1122face = -1/(8 Pi) * Sum[
  alphaWeights[[ia]] * betaWeights[[ib]] *
    2 * I1exact[aVals[[ia]]^2 + bVals[[ib]]^2],
  {ia, 1, 3}, {ib, 1, 3}
];
Print["  B_{1122} = ", NumberForm[N[B1122face, 16], 16]];

(* By C_4v: B_{1133} = B_{1122} *)
B1133face = B1122face;
Print["  B_{1133} = B_{1122}  (C_4v symmetry)"];
Print[];

(* --- B_{2222}: d^4 Psi / dR2^4 ---
   All four derivatives on the w2-tent at R2=0.
   d^4/dR2^4 M00(w2) gives a fourth-derivative delta distribution:
   M00''''(w2) = +4 delta(w2) - 2 delta(w2-1) - 2 delta(w2+1)
                 + delta(w2-2) + delta(w2+2)

   But M00(w2) has support [-1,1], so w2=+/-2 are outside the tent support
   and those delta evaluations give zero (the tent factor on that axis is zero
   at the boundary, but the kernel sqrt(c^2+...) is evaluated at a point
   where the other tents still contribute).

   Wait — M00''''(d) for M00 = max(0,1-|d|) piecewise linear:
   M00'(d) = -sign(d) for |d|<1, undefined at d=0,+/-1
   M00''(d) = -2 delta(d) + delta(d-1) + delta(d+1)
   We need d^4/dR2^4 M00(w2-R2).

   But the fourth derivative of the piecewise linear tent doesn't exist
   as a classical function — we'd need d^2/dR2^2 of M00''(w2-R2),
   which involves derivatives of delta functions.

   Actually, d^4 Psi/dR2^4 should be computed by applying d^2/dR2^2 TWICE.
   The first application gives:
   d^2 Psi/dR2^2 = int M(w1-R1) M''(w2-R2) M(w3-R3) |w| d^3w
   = Sum_b beta_b * int M(w1-R1) M(w3-R3) |w|_{w2=b} dw1 dw3

   The second d^2/dR2^2 can't act on M again (it's already been collapsed).
   Instead we need the second derivative of this 2D function w.r.t. R2.

   But R2 only appeared in the w2-tent and b=R2+offset, so after collapse
   the b values ARE functions of R2. Let me reconsider.

   More carefully: Psi(R) involves M00(w2-R2). After first d^2/dR2^2,
   we collapsed w2. The result is a function of R2 through the b values.
   Taking another d^2/dR2^2 of the collapsed expression:

   d^2 Psi/dR2^2 = Sum_b beta_b F(R2+offset_b, R1, R3)
   where F involves 2D integrals.

   d^4 Psi/dR2^4 = Sum_b beta_b d^2 F/dR2^2

   But F(w2_fixed, R1, R3) doesn't depend on R2 anymore — the w2 coordinate
   is fixed by the delta. So d^2 F/dR2^2 = 0.

   This means we need to think about it differently.
   The fourth derivative requires the SECOND delta collapse on the same axis,
   which means we need the second derivative of the 2D kernel
   sqrt(c^2+w1^2+w3^2) w.r.t. c (i.e. d^2/dc^2).

   d^2/dR2^2 [d^2 Psi/dR2^2] =
     Sum_b beta_b * d^2/dR2^2 [int tent(w1) tent(w3) sqrt((R2+b_offset)^2+w1^2+w3^2)]

   Here b_offset is fixed, and d/dR2 acts on sqrt((R2+b)^2+...).
   After collapsing the first time at w2=R2+b, the kernel becomes
   sqrt((R2+b)^2+w1^2+w3^2) with R2 still a free variable.

   Actually, let me step back. The tent derivatives act on the M00(w_i - R_i)
   factors, NOT on the kernel |w|. So:

   d^4 Psi/dR2^4 = int M(w1-R1) d^4[M(w2-R2)]/dR2^4 M(w3-R3) |w| d^3w
   = int M(w1-R1) M''''(w2-R2) M(w3-R3) |w| d^3w

   M00(d) = max(0, 1-|d|) is piecewise linear. Its derivatives:
   M00' = -sign(d) for |d|<1
   M00'' = -2 delta(d) + delta(d-1) + delta(d+1)
   M00''' = -2 delta'(d) + delta'(d-1) + delta'(d+1)
   M00'''' = -2 delta''(d) + delta''(d-1) + delta''(d+1)

   This involves delta DERIVATIVES, which act as:
   int f(w) delta'(w-a) dw = -f'(a)
   int f(w) delta''(w-a) dw = f''(a)

   So d^4 Psi/dR2^4 involves d^2/dw2^2 of the kernel |w| at
   w2 = R2-1, R2, R2+1 = -1, 0, 1 (for R2=0).

   d^2|w|/dw2^2 = (w1^2+w3^2)/|w|^3

   And then the delta'' terms give:
   int M(w1-1) M(w3) delta''(w2) |w| dw2 = d^2/dw2^2[|w|] at w2=0
   = (w1^2+w3^2)/|w|^3 at w2=0 = (w1^2+w3^2)/(w1^2+w3^2)^{3/2} = 1/sqrt(w1^2+w3^2)

   But wait — the delta'' from M'''' involves M00'''' which has delta'',
   not delta. Let me be more precise.

   M00''''(d) acts on f(w2) as:
   int M00''''(w2-R2) f(w2) dw2
   = [-2 f''(R2) + f''(R2+1) + f''(R2-1)]

   So d^4 Psi/dR2^4 at R2=0:
   = int M(w1-1) M(w3) [-2 (d^2|w|/dw2^2)|_{w2=0}
                         + (d^2|w|/dw2^2)|_{w2=1}
                         + (d^2|w|/dw2^2)|_{w2=-1}] dw1 dw3

   With d^2|w|/dw2^2 = (w1^2+w3^2)/(w1^2+w2^2+w3^2)^{3/2}
                      = 1/rho - w2^2/rho^3

   So B_{2222} = -1/(8pi) * 4 * Sum_{c2 in {-1,0,1}} gamma[c2] *
     int_0^1 int_0^2 tent(w1-1) tent(w3) [1/rho - c2^2/rho^3]_{w2=c2} dw1 dw3

   where gamma = {+1, -2, +1} (same weights but now from delta'').

   By |c2| symmetry of the kernel: the 2D integral depends only on |c2|.
   gamma at c2={-1,0,1} = {+1,-2,+1}, and |c2| pattern = {1,0,1}.
   Sum = (+1)*I(1) + (-2)*I(0) + (+1)*I(1) = 2*I(1) - 2*I(0).

   This has the same structure as A_{22} but with kernel 1/rho - c^2/rho^3
   instead of just 1/rho.

   The tent product in (w1,w3) is tent(w1-1)*tent(w3) — same as for A_{22}. *)

Print["--- B_{2222} for R = (1,0,0) ---"];

(* We need shifted B-channel integrals too *)
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

(* J22 with kernel 1/rho - c^2/rho^3 for B_{2222} *)
(* Recall J22[c] already has the A-channel (1/rho) tent integral.
   We need the B-channel (c^2/rho^3) tent integral with the same
   (w1,w3) tent product = tent(w1-1)*tent(w3). *)

J22B[c_] := If[c == 0, 0,
  2 * (
    (* [0,1]^2 piece *)
    (I2B[1, 0, c] - I2B[1, 1, c])
    +
    (* [1,2] shifted piece *)
    (I2Bshifted[0, 0, c, 1] - I2Bshifted[1, 0, c, 1]
     - I2Bshifted[0, 1, c, 1] + I2Bshifted[1, 1, c, 1])
  )
];

(* Wait — the (w1,w3) roles here: for B_{2222}, the remaining axes
   after w2-collapse are w1 and w3. The w1-tent is tent(w1-1) centered
   at R1=1, and w3-tent is tent(w3) centered at R3=0.
   The u-integration is w1 in [0,2] and v-integration is w3 in [0,1]
   (after folding by w3-symmetry with factor 2).

   But in J22[c] I defined it with u=w1 and v=w3, and the kernel
   sqrt(c^2 + w1^2 + w3^2) — however for B_{2222} the kernel is
   1/rho - c2^2/rho^3 = (w1^2+w3^2)/(c2^2+w1^2+w3^2)^{3/2}

   This is NOT simply I2A - I2B. Let me redo:
   (w1^2+w3^2)/(c^2+w1^2+w3^2)^{3/2}
   = (c^2+w1^2+w3^2-c^2)/(c^2+w1^2+w3^2)^{3/2}
   = 1/sqrt(c^2+w1^2+w3^2) - c^2/(c^2+w1^2+w3^2)^{3/2}
   = 1/rho - c^2/rho^3

   So the 2D integral with this kernel and tent product is:
   J22_kernel(c) = J22(c) [A-channel, 1/rho] - J22B(c) [B-channel, c^2/rho^3]
   and for c=0: kernel = 1/sqrt(w1^2+w3^2) = J22(0). OK. *)

Print["  Computing shifted B-channel integrals ..."];
Do[I2Bshifted[p, q, 1, 1], {p, 0, 1}, {q, 0, 1}];

B2222face = -1/(8 Pi) * (2 * (J22[1] - J22B[1]) - 2 * (J22[0] - 0));
Print["  B_{2222} = ", NumberForm[N[B2222face, 16], 16]];

(* B_{2233}: d^4 Psi / dR2^2 dR3^2
   d^2/dR2^2 collapses w2 to {-1,0,+1}, weights {+1,-2,+1}.
   d^2/dR3^2 collapses w3 to {-1,0,+1}, weights {+1,-2,+1}.
   Remaining: 1D integral in w1 with tent(w1-1), and kernel |w|.

   int tent(w1-1) |w|_{w2=b,w3=c} dw1 = int_0^2 (1-|w1-1|) sqrt(b^2+c^2+w1^2) dw1
   Split [0,1] and [1,2]:
   = int_0^1 w1 sqrt(C+w1^2) dw1 + int_0^1 (1-s) sqrt(C+(s+1)^2) ds
   where C = b^2 + c^2.

   First piece: int_0^1 t sqrt(C+t^2) dt = (1/3)[(C+1)^{3/2} - C^{3/2}]
   Second piece: let F(shift) = int_0^1 (1-s) sqrt(C+(s+shift)^2) ds
                 — a shifted 1D integral. *)

Print[];
Print["--- B_{2233} for R = (1,0,0) (double delta collapse, w2 and w3) ---"];

(* 1D building blocks *)
(* int_0^1 t sqrt(C + t^2) dt *)
Jpiece1[bigC_] := (1/3) ((bigC + 1)^(3/2) - bigC^(3/2));

(* int_0^1 (1-s) sqrt(C + (s+1)^2) ds *)
Jpiece2[bigC_] := Module[{result},
  result = Integrate[(1 - s) Sqrt[bigC + (s + 1)^2], {s, 0, 1},
             Assumptions -> bigC >= 0, GenerateConditions -> False];
  Simplify[result]
];

(* Total 1D integral for w1-tent at shift 1 *)
Jtent1D[bigC_] := Jpiece1[bigC] + Jpiece2[bigC];

B2233face = -1/(8 Pi) * Sum[
  alphaWeights[[ib2]] * betaWeights[[ib3]] *
    Jtent1D[bVals[[ib2]]^2 + bVals[[ib3]]^2],
  {ib2, 1, 3}, {ib3, 1, 3}
];
Print["  B_{2233} = ", NumberForm[N[B2233face, 16], 16]];
Print[];

(* --- B_{1212}: d^4 Psi / dR1^2 dR2^2
   Same as B_{1122} by symmetry of mixed partial derivatives.
   B_{1212} = d^4 Psi/(dR1 dR2 dR1 dR2) = d^4 Psi/(dR1^2 dR2^2) = B_{1122} *)
B1212face = B1122face;
Print["  B_{1212} = B_{1122} = ", NumberForm[N[B1212face, 16], 16],
      "  (mixed partial symmetry)"];

(* B_{2323} *)
B2323face = B2233face;
Print["  B_{2323} = B_{2233} = ", NumberForm[N[B2323face, 16], 16],
      "  (mixed partial symmetry)"];
Print[];

(* --- Algebraic validation: Laplacian identity ---
   B_{iijj} summed over one pair: B_{1111} + B_{1122} + B_{1133} = A_{11}
   because d^2/dR1^2 (d^2 Psi/(dR1^2+dR2^2+dR3^2)) = d^2/dR1^2 * Laplacian(Psi)
   and Laplacian(|r|) = 2/|r|, so Laplacian(Psi) ~ Phi.
   More precisely: nabla^2 Psi = 2 Phi, so
   Sum_k B_{ijkk} = -1/(8pi) * d^4 Psi / (dRi dRj Sum_k dRk^2)
                   = -1/(8pi) * d^2/(dRi dRj) [nabla^2 Psi]
                   = -1/(8pi) * d^2/(dRi dRj) [2 Phi]
                   = -2/(8pi) * d^2 Phi/(dRi dRj)
                   = 2/(4pi) * [-1/(4pi) d^2 Phi/(dRi dRj)] * (4pi)
   Hmm, let me just check: Sum_k B_{11kk} should relate to A_{11}.

   Actually: Sum_k d^4 Psi/(dR1^2 dRk^2) = d^2/dR1^2 [nabla^2 Psi]
   nabla^2 |w| = 2/|w|, so nabla^2 Psi = int M1 M2 M3 * 2/|w| d^3w = 2 Phi

   Sum_k B_{11kk} = -1/(8pi) * d^2[2 Phi]/dR1^2 = -1/(4pi) d^2 Phi/dR1^2 = A_{11}

   So: B_{1111} + B_{1122} + B_{1133} = A_{11}                           *)

Print["--- Laplacian identity check: B_{1111} + B_{1122} + B_{1133} = A_{11} ---"];
laplCheck = N[B1111face + B1122face + B1133face, 16];
Print["  LHS = B_{1111} + B_{1122} + B_{1133} = ", NumberForm[laplCheck, 16]];
Print["  RHS = A_{11} = ", NumberForm[N[A11face, 16], 16]];
Print["  |difference| = ", ScientificForm[Abs[laplCheck - N[A11face, 16]], 3]];
Print[];

Print["--- Second Laplacian check: B_{2211} + B_{2222} + B_{2233} = A_{22} ---"];
laplCheck2 = N[B1122face + B2222face + B2233face, 16];
Print["  LHS = B_{1122} + B_{2222} + B_{2233} = ", NumberForm[laplCheck2, 16]];
Print["  RHS = A_{22} = ", NumberForm[N[A22face, 16], 16]];
Print["  |difference| = ", ScientificForm[Abs[laplCheck2 - N[A22face, 16]], 3]];
Print[];


(* ================================================================ *)
(* Section 5. Static Propagator Assembly — Face Adjacent             *)
(*                                                                   *)
(* P^stat_{ijkl}(R) = -1/(2mu) [delta_ik A_jl + delta_jk A_il      *)
(*                               - 2 eta B_ijkl]                    *)
(*                                                                   *)
(* For face-adjacent R = (a,0,0) with C_4v symmetry, the            *)
(* independent components are:                                       *)
(*   P_{1111}, P_{1122}, P_{2222}, P_{2233}, P_{1212}, P_{2323}     *)
(* ================================================================ *)

Print["==== Section 5: Static Propagator Assembly (Face-Adjacent) ===="];
Print[];

(* P_{1111}: i=j=k=l=1.  delta_{ik}=1, delta_{jk}=1.
   P = -1/(2mu) [A_{11} + A_{11} - 2 eta B_{1111}] *)
P1111face = -1/(2 mu) (2 A11face - 2 eta B1111face);

(* P_{1122}: i=j=1, k=l=2.  delta_{12}=0, delta_{12}=0.
   P = -1/(2mu) [0 + 0 - 2 eta B_{1122}] = eta/(mu) B_{1122} *)
P1122face = -1/(2 mu) (-2 eta B1122face);

(* P_{1133}: same structure, = P_{1122} by C_4v *)
P1133face = P1122face;

(* P_{2222}: i=j=k=l=2.  delta_{22}=1, delta_{22}=1.
   P = -1/(2mu) [2 A_{22} - 2 eta B_{2222}] *)
P2222face = -1/(2 mu) (2 A22face - 2 eta B2222face);

(* P_{2233}: i=j=2, k=l=3.  delta_{23}=0, delta_{23}=0.
   P = -1/(2mu) [-2 eta B_{2233}] = eta/mu B_{2233} *)
P2233face = -1/(2 mu) (-2 eta B2233face);

(* P_{1212}: i=k=1, j=l=2.  delta_{11}=1, delta_{21}=0.
   P = -1/(2mu) [1*A_{22} + 0 - 2 eta B_{1212}]
   = -1/(2mu) [A_{22} - 2 eta B_{1122}] *)
P1212face = -1/(2 mu) (A22face - 2 eta B1122face);

(* P_{2323}: i=k=2, j=l=3.  delta_{22}=1, delta_{32}=0.
   P = -1/(2mu) [A_{33} + 0 - 2 eta B_{2323}]
   = -1/(2mu) [A_{22} - 2 eta B_{2233}] *)
P2323face = -1/(2 mu) (A22face - 2 eta B2233face);

(* P_{1313} = P_{1212} by C_4v *)
P1313face = P1212face;

(* P_{3333} = P_{2222} by C_4v *)
P3333face = P2222face;

Print["========================================================"];
Print["  STATIC STRAIN PROPAGATOR — Face Adjacent R = (a,0,0)"];
Print["========================================================"];
Print["  P_{1111} = ", NumberForm[N[P1111face, 16], 16]];
Print["  P_{1122} = ", NumberForm[N[P1122face, 16], 16]];
Print["  P_{1133} = ", NumberForm[N[P1133face, 16], 16], "  (= P_{1122})"];
Print["  P_{2222} = ", NumberForm[N[P2222face, 16], 16]];
Print["  P_{2233} = ", NumberForm[N[P2233face, 16], 16]];
Print["  P_{3333} = ", NumberForm[N[P3333face, 16], 16], "  (= P_{2222})"];
Print["  P_{1212} = ", NumberForm[N[P1212face, 16], 16]];
Print["  P_{1313} = ", NumberForm[N[P1313face, 16], 16], "  (= P_{1212})"];
Print["  P_{2323} = ", NumberForm[N[P2323face, 16], 16]];
Print["========================================================"];
Print[];

(* Cross-check: validate against WS2 finite-difference values *)
Print["--- Validating against WS2 finite differences ---"];

Psi00[{Rx_?NumericQ, Ry_?NumericQ, Rz_?NumericQ}] :=
  NIntegrate[
    Max[0, 1 - Abs[w1 - Rx]] Max[0, 1 - Abs[w2 - Ry]] Max[0, 1 - Abs[w3 - Rz]] *
      Sqrt[w1^2 + w2^2 + w3^2],
    {w1, Rx - 1, Rx + 1},
    {w2, Ry - 1, Ry + 1},
    {w3, Rz - 1, Rz + 1},
    WorkingPrecision -> 16, MaxRecursion -> 12, PrecisionGoal -> 10
  ];

(* Fourth derivatives via nested finite differences *)
D4FD[f_, R0fd_, i_, j_, k_, l_] := Module[{ei, ej, ek, el, g},
  ek = delta UnitVector[3, k];
  el = delta UnitVector[3, l];
  g[RR_] := Module[{eii = delta UnitVector[3, i], ejj = delta UnitVector[3, j]},
    If[i == j,
      (f[RR + eii] - 2 f[RR] + f[RR - eii]) / delta^2,
      (f[RR+eii+ejj] - f[RR+eii-ejj] - f[RR-eii+ejj] + f[RR-eii-ejj])/(4 delta^2)
    ]
  ];
  If[k == l,
    (g[R0fd + ek] - 2 g[R0fd] + g[R0fd - ek]) / delta^2,
    (g[R0fd+ek+el] - g[R0fd+ek-el] - g[R0fd-ek+el] + g[R0fd-ek-el]) / (4 delta^2)
  ]
];

Print["  Computing WS2-style finite differences (this is slow) ..."];
d4Psi1111FD = D4FD[Psi00, R0, 1, 1, 1, 1];
d4Psi1122FD = D4FD[Psi00, R0, 1, 1, 2, 2];
B1111FD = -1/(8 Pi) d4Psi1111FD;
B1122FD = -1/(8 Pi) d4Psi1122FD;

P1111FD = -1/(2 mu) (2 A11FD - 2 eta B1111FD);
P1122FD = -1/(2 mu) (-2 eta B1122FD);

Print["  P_{1111}: analytical = ", NumberForm[N[P1111face, 10], 10],
      "  FD = ", NumberForm[P1111FD, 10],
      "  |D| = ", ScientificForm[Abs[N[P1111face, 16] - P1111FD], 3]];
Print["  P_{1122}: analytical = ", NumberForm[N[P1122face, 10], 10],
      "  FD = ", NumberForm[P1122FD, 10],
      "  |D| = ", ScientificForm[Abs[N[P1122face, 16] - P1122FD], 3]];
Print[];
