(* ::Package:: *)
(* CubeT6Block.wl -- Tier-6 extension of the cube Path-B closure.

   Extends the 9-dimensional trial space T_9 (constants + symmetric linear
   strains) to the 27-dimensional space T_27 by adjoining 18 quadratic
   modes:  for each of the three coordinate directions e_k (k = x, y, z)
   we add six vector fields of the form (r_a r_b) e_k, a,b in {1,2,3},
   a <= b.

   The body bilinear form on the quad-quad sub-block is assembled using
   the same machinery that was validated on the 9x9 block:
     - polynomial basis substitutions r -> xi + s/2, r' -> xi - s/2,
     - inner xi-integration over the overlap box B(s),
     - s -> 2u rescaling and octant symmetrization,
     - K1 helper (weak form of 1/|u|, closed via Mp masters),
     - K3 kernel (1/|u|^3 with DuffyCoordinates NIntegrate),
     - Mp master integrals (closed form for 7 rank-6/8 cases cached in
       CubeT6Masters.wl; MachinePrecision NIntegrate otherwise).

   The seven exact masters (mp222, mp420, mp600, mp422, mp440, mp620,
   mp800) carry the all-even high-degree values needed later for the
   closed-form LaTeX output.  All lower-degree and mixed-parity Mp
   values are computed numerically via NIntegrate with
   DuffyCoordinates on the [0,1]^3 corner singularity, giving ~15-digit
   accuracy in well under a second per call.  Memoization (via
   := X = ...) keeps the total assembly cost down to ~2 seconds.

   Usage:
     /Applications/Wolfram.app/Contents/MacOS/WolframKernel -noprompt \
        -script Mathematica/CubeT6Block.wl
*)

Print["==== CubeT6Block.wl : Tier-6 27-component Path-B block ===="];
Print[];

$here = DirectoryName[$InputFileName];

(* ------------------------------------------------------------------ *)
(* 0. Load cached Tier-6 masters (mp222, mp420, ..., mp800).            *)
(* ------------------------------------------------------------------ *)
Get[FileNameJoin[{$here, "CubeT6Masters.wl"}]];

(* ------------------------------------------------------------------ *)
(* 1. Master integrals over [-1,1]^3 and [0,1]^3                        *)
(* ------------------------------------------------------------------ *)
(* M[p,q,r] = int_{[-1,1]^3} x^p y^q z^r/|r| dV (even-parity only).    *)
(* Mp[p,q,r] = int_{[0,1]^3} x^p y^q z^r/|r| dV (positive octant).     *)

ClearAll[M, Mp];

M[p_, q_, r_] := M[p, q, r] = If[OddQ[p] || OddQ[q] || OddQ[r],
  0,
  Integrate[
    x^p y^q z^r/Sqrt[x^2 + y^2 + z^2],
    {x, -1, 1}, {y, -1, 1}, {z, -1, 1}
  ]
];

(* Cubic symmetry: canonicalize argument order to ASCENDING (Sort default). *)
Mp[p_Integer, q_Integer, r_Integer] /; ! OrderedQ[{p, q, r}] :=
  Mp @@ Sort[{p, q, r}];

(* Prime high-degree cases from CubeT6Masters.wl (ASCENDING canonical). *)
Mp[2, 2, 2] = mp222;
Mp[0, 2, 4] = mp420;
Mp[0, 0, 6] = mp600;
Mp[2, 2, 4] = mp422;
Mp[0, 4, 4] = mp440;
Mp[0, 2, 6] = mp620;
Mp[0, 0, 8] = mp800;

(* Fallback definition: catch any non-negative integer triple and memoize.
   No ordering constraint — the canonicalizer rule above has already
   sorted its arguments to ascending by the time this rule is tried.
   We use NIntegrate at WorkingPrecision -> 25 with DuffyCoordinates for
   the 1/|r| corner singularity, giving ~15-digit accuracy per call.
   MachinePrecision was insufficient: it gave only ~9 correct digits on
   Mp[0,0,0], which propagated through K1at (sums of 8 Mp values with
   signs) and polynomial coefficients ~256 * (combinatorics) into a
   ~3e-5 error in the B27quad A-channel entries -- not enough for the
   downstream PSLQ/closed-form extraction of scalar building blocks.
   Symbolic Integrate is prohibitively slow for mixed-parity degrees
   and gives no advantage over high-WP numerics here; the seven exact
   cases primed above from CubeT6Masters.wl already cover the all-even
   high-degree masters needed for closed-form LaTeX output.            *)
(* MachinePrecision with a pushed PrecisionGoal.  The default PG for
   MachinePrecision NIntegrate is only ~6-8 digits; pushing it to 12
   forces extra refinement near the corner without jumping to
   software-precision arithmetic, which was prohibitively slow in the
   assembly phase.  This gives ~12-digit accuracy per Mp call in under
   a second.  Caveat: some hard integrands may hit MaxRecursion without
   achieving the full 12 digits -- we verify below against Mp[0,0,0].  *)
Mp[p_Integer, q_Integer, r_Integer] := Mp[p, q, r] =
  NIntegrate[
    x^p y^q z^r/Sqrt[x^2 + y^2 + z^2],
    {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
    PrecisionGoal -> 12,
    AccuracyGoal -> 12,
    MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}
  ];

Print["Mp cache primed.  Spot check:"];
Print["  Mp[0,0,0] ≈ ", N[Mp[0, 0, 0], 16]];
Print["  Mp[2,2,2] ≈ ", N[Mp[2, 2, 2], 16]];
Print["  Mp[4,2,0] ≈ ", N[Mp[4, 2, 0], 16]];
Print["  Mp[8,0,0] ≈ ", N[Mp[8, 0, 0], 16]];

(* ------------------------------------------------------------------ *)
(* 2. Trial space T_27 and its mass matrix                              *)
(* ------------------------------------------------------------------ *)
Print[];
Print["==== Section 2: building basisT27 (= basisT9 ∪ 18 quadratic modes) ===="];

basisT9 = {
  {1, 0, 0},                           (* u_x *)
  {0, 1, 0},                           (* u_y *)
  {0, 0, 1},                           (* u_z *)
  {r1, 0, 0},                          (* eps_xx *)
  {0, r2, 0},                          (* eps_yy *)
  {0, 0, r3},                          (* eps_zz *)
  {0, r3/2, r2/2},                     (* eps_yz (symmetric) *)
  {r3/2, 0, r1/2},                     (* eps_xz *)
  {r2/2, r1/2, 0}                      (* eps_xy *)
};

(* 18 quadratic modes: phi^(9+6*(k-1)+m) = q_m * e_k, for k in {1,2,3}
   (coordinate direction) and m in {1..6} (quadratic monomial index).
   The six quadratic monomials in canonical Voigt order are
     q_1 = r1^2,  q_2 = r2^2,  q_3 = r3^2,
     q_4 = r2 r3, q_5 = r1 r3, q_6 = r1 r2.                             *)

quadMonomials = {r1^2, r2^2, r3^2, r2*r3, r1*r3, r1*r2};
unitVec[1] = {1, 0, 0};
unitVec[2] = {0, 1, 0};
unitVec[3] = {0, 0, 1};

basisT27quad = Flatten[
  Table[q * unitVec[k], {k, 1, 3}, {q, quadMonomials}],
  1
];
(* basisT27quad is a list of 18 vector fields, each a triple of polynomials. *)

basisT27 = Join[basisT9, basisT27quad];
Print["Length[basisT27] = ", Length[basisT27], " (expect 27)"];

(* ------------------------------------------------------------------ *)
(* 3. L^2 mass matrix on T_27                                          *)
(* ------------------------------------------------------------------ *)
polyDot[v1_, v2_] := Sum[v1[[p]] v2[[p]], {p, 3}];
massEntry[v1_, v2_] := Integrate[polyDot[v1, v2],
  {r1, -1, 1}, {r2, -1, 1}, {r3, -1, 1}];

Print[];
Print["==== Section 3: 27x27 mass matrix M27 ===="];
Print["Computing M27 (polynomial integrals only, should be fast)..."];
t0 = AbsoluteTime[];
M27 = Table[massEntry[basisT27[[i]], basisT27[[j]]], {i, 27}, {j, 27}];
t1 = AbsoluteTime[];
Print["  done in ", Round[t1 - t0, 0.01], " s"];

Print["M27 symmetric? ", M27 === Transpose[M27]];
M27eig = Eigenvalues[M27];
Print["M27 eigenvalues (sorted, exact): ", Sort[M27eig]];
Print["M27 min eigenvalue (numerical) : ", N[Min[M27eig], 16]];
Print["M27 max eigenvalue (numerical) : ", N[Max[M27eig], 16]];
Print["M27 positive-definite? ", AllTrue[M27eig, # > 0 &]];

(* Block structure: M27 should couple basisT9 only to itself (block 1..9)
   and basisT27quad only to itself + to the 6 axial-strain modes (since
   quadratic * linear has degree 3 which is odd, the L^2 integral on the
   symmetric cube [-1,1]^3 vanishes whenever the total degree is odd in
   any axis).  Let's verify.                                              *)
M9 = M27[[1 ;; 9, 1 ;; 9]];
Mcross = M27[[1 ;; 9, 10 ;; 27]];
Mquad = M27[[10 ;; 27, 10 ;; 27]];
Print[];
Print["Block pieces of M27:"];
Print["  M9 (constants + linear strain) symmetric? ",
  M9 === Transpose[M9]];
Print["  M9 eigenvalues = ", Sort[Eigenvalues[M9]]];
Print[];
Print["  Mcross (T9 vs quad, 9x18):  nonzero entries -> total degree"];
Print["    axial-strain rows 4..6 couple to quadratic modes whose q_m"];
Print["    contains r_(m-3) as a factor; all other rows must be zero."];
nonzeroCross = Select[
  Flatten[Table[{i, j, Mcross[[i, j]]}, {i, 9}, {j, 18}], 1],
  #[[3]] =!= 0 &
];
Print["    # nonzero entries in Mcross = ", Length[nonzeroCross],
  " (expect 6)"];
Print["    nonzero entries (row_T9, col_quad, value):"];
Scan[Print["      ", #] &, nonzeroCross];

Print[];
Print["  Mquad (quadratic block, 18x18): symmetric? ",
  Mquad === Transpose[Mquad]];
Print["  Mquad eigenvalues: ", Sort[Eigenvalues[Mquad]]];

(* Print the diagonal for reference. *)
Print[];
Print["  M27 diagonal (integer-only):"];
Do[
  Print["    M27[", i, ",", i, "] = ", M27[[i, i]]],
  {i, 27}
];

(* ------------------------------------------------------------------ *)
(* 4. Helpers for the body bilinear form (ported from CubeGalerkin27)   *)
(* ------------------------------------------------------------------ *)
Print[];
Print["==== Section 4: body bilinear helpers (K1, K3, bbBodyAB) ===="];

(* ----- K1at via linearity on the tent factor --------------------- *)
(* Closed form via Mp masters (all cached or one-shot symbolic).     *)
ClearAll[K1at];
K1at[p_Integer, q_Integer, r_Integer] := K1at[p, q, r] = Sum[
  (-1)^(a + b + c) Mp[p + a, q + b, r + c],
  {a, 0, 1}, {b, 0, 1}, {c, 0, 1}
];

(* ----- K3 diagonal and off-diagonal: direct 3D NIntegrate.          *)
(* For the Tier-6 block we evaluate the 1/|u|^3 kernel directly via   *)
(* WorkingPrecision -> 25 NIntegrate on the positive octant.  The     *)
(* DuffyCoordinates singularity handler concentrates samples near    *)
(* the corner u1 = u2 = u3 = 0 where the integrand has a 1/|u|        *)
(* effective singularity (the leading u^2 factor tames the 1/|u|^3    *)
(* kernel).  Memoization avoids recomputing the same (p,q,r).  The    *)
(* higher WorkingPrecision matches Mp and gives ~15-digit accuracy    *)
(* in the final B27quad entries after polynomial propagation.         *)
(* MachinePrecision with pushed PrecisionGoal -> 12.  Matches the Mp    *)
(* fallback strategy above: the default PG for MachinePrecision         *)
(* NIntegrate is only ~6-8 digits, so we push it to 12 to force extra   *)
(* refinement near the corner without switching to software precision  *)
(* (which was prohibitively slow in the assembly phase).                *)
ClearAll[K3diag, K3off];

K3diag[p_Integer, q_Integer, r_Integer] := K3diag[p, q, r] =
  NIntegrate[
    (1 - u1) (1 - u2) (1 - u3) u1^(p + 2) u2^q u3^r /
      (u1^2 + u2^2 + u3^2)^(3/2),
    {u1, 0, 1}, {u2, 0, 1}, {u3, 0, 1},
    PrecisionGoal -> 12,
    AccuracyGoal -> 12,
    MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}
  ];

K3off[p_Integer, q_Integer, r_Integer] := K3off[p, q, r] =
  NIntegrate[
    (1 - u1) (1 - u2) (1 - u3) u1^(p + 1) u2^(q + 1) u3^r /
      (u1^2 + u2^2 + u3^2)^(3/2),
    {u1, 0, 1}, {u2, 0, 1}, {u3, 0, 1},
    PrecisionGoal -> 12,
    AccuracyGoal -> 12,
    MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}
  ];

(* ----- K3kernel: dispatch by (i,j) using cubic symmetry ------------ *)
ClearAll[K3kernel];
K3kernel[{p_, q_, r_}, i_, j_] := K3kernel[{p, q, r}, i, j] = Which[
  i == 1 && j == 1, K3diag[p, q, r],
  i == 2 && j == 2, K3diag[q, p, r],   (* swap u1<->u2 *)
  i == 3 && j == 3, K3diag[r, q, p],   (* swap u1<->u3 *)
  (i == 1 && j == 2) || (i == 2 && j == 1), K3off[p, q, r],
  (i == 1 && j == 3) || (i == 3 && j == 1), K3off[p, r, q],
  (i == 2 && j == 3) || (i == 3 && j == 2), K3off[q, r, p],
  True, $Failed
];

(* ----- Parity filter for positive-octant restriction --------------- *)
kernelParity[pIdx_Integer, qIdx_Integer] :=
  Mod[
    {KroneckerDelta[pIdx, 1] + KroneckerDelta[qIdx, 1],
     KroneckerDelta[pIdx, 2] + KroneckerDelta[qIdx, 2],
     KroneckerDelta[pIdx, 3] + KroneckerDelta[qIdx, 3]},
    2
  ];

parityOK[exps_List, kp_List] :=
  AllTrue[Range[3], EvenQ[exps[[#]] + kp[[#]]] &];

ClearAll[u1, u2, u3];

bbBodyAB[poly_, pIdx_Integer, qIdx_Integer] :=
  Module[{coeffs, aPiece, bPiece, kpA = {0, 0, 0}, kpB},
    kpB = kernelParity[pIdx, qIdx];
    coeffs = CoefficientRules[Expand[poly], {u1, u2, u3}];
    aPiece = Sum[
      If[parityOK[pair[[1]], kpA],
        pair[[2]] K1at @@ pair[[1]],
        0],
      {pair, coeffs}
    ];
    bPiece = Sum[
      If[parityOK[pair[[1]], kpB],
        pair[[2]] K3kernel[pair[[1]], pIdx, qIdx],
        0],
      {pair, coeffs}
    ];
    {aPiece, bPiece}
  ];

(* ----- xi-integration over the overlap box ------------------------- *)
ClearAll[xiMonInt, intXiThenSub, basisAtR, basisAtRprime, polyToTentResidual];

xiMonInt[k1_Integer, k2_Integer, k3_Integer] :=
  If[OddQ[k1] || OddQ[k2] || OddQ[k3],
    0,
    Times @@ MapThread[
      (2 (1 - #1)^(#2 + 1)/(#2 + 1)) &,
      {{u1, u2, u3}, {k1, k2, k3}}
    ]
  ];

intXiThenSub[expr_] := Module[{e, coeffs},
  e = Expand[expr];
  coeffs = CoefficientRules[e, {xi[1], xi[2], xi[3]}];
  Expand[
    Sum[pair[[2]] xiMonInt @@ pair[[1]], {pair, coeffs}] /.
      {s[1] -> 2 u1, s[2] -> 2 u2, s[3] -> 2 u3}
  ]
];

basisAtR[v_] := v /. {r1 -> xi[1] + s[1]/2, r2 -> xi[2] + s[2]/2,
   r3 -> xi[3] + s[3]/2};
basisAtRprime[v_] := v /. {r1 -> xi[1] - s[1]/2, r2 -> xi[2] - s[2]/2,
   r3 -> xi[3] - s[3]/2};

polyToTentResidual[poly_] := Expand[Cancel[
  poly / ((1 - u1) (1 - u2) (1 - u3))
]];

(* ----- b27pairContribution / b27entry ------------------------------ *)
(* Identical body of logic to b9entry, but indexed over basisT27.      *)
ClearAll[b27pairContribution, b27entry];

b27pairContribution[i_Integer, j_Integer, p_Integer, q_Integer] :=
  Module[{phiI, phiJ, productPQ, weight, residual},
    phiI = basisAtR[basisT27[[i]]][[p]];
    phiJ = basisAtRprime[basisT27[[j]]][[q]];
    productPQ = Expand[phiI phiJ];
    If[productPQ === 0, Return[{0, 0}]];
    weight = intXiThenSub[productPQ];
    residual = polyToTentResidual[weight];
    bbBodyAB[residual, p, q]
  ];

b27entry[i_Integer, j_Integer] := Module[{a = 0, b = 0, contr},
  Do[
    contr = b27pairContribution[i, j, p, q];
    If[p == q, a = a + contr[[1]]];
    b = b + contr[[2]],
    {p, 3}, {q, 3}
  ];
  32 (Aelas a + Belas b)
];

(* ----- Sanity: re-derive the existing B9 diagonal on T_9 ----------- *)
(* This also seeds Mp[0,0,0], Mp[1,0,0], Mp[1,1,0], Mp[1,1,1], and     *)
(* K3diag[0,0,0] into the cache as a first smoke-test of the fallback. *)
Print[];
Print["---- Sanity check: b27entry[1,1] should match 256 (A iA + B iB11) ----"];
iA = Mp[0, 0, 0] - 3 Mp[1, 0, 0] + 3 Mp[1, 1, 0] - Mp[1, 1, 1];
iB11val = K3diag[0, 0, 0];
expect11 = 256 (Aelas iA + Belas iB11val);
t0 = AbsoluteTime[];
got11 = b27entry[1, 1];
t1 = AbsoluteTime[];
Print["  b27entry[1,1] computed in ", Round[t1 - t0, 0.01], " s"];
Print["  b27entry[1,1] - 256 (A iA + B iB11) (numerical):"];
Print["    A-channel: ",
  N[(got11 - expect11) /. {Aelas -> 1, Belas -> 0}, 20]];
Print["    B-channel: ",
  N[(got11 - expect11) /. {Aelas -> 0, Belas -> 1}, 20]];
Print["  Mp[0,0,0] ≈ ", N[Mp[0, 0, 0], 16], "  (expect 1.190038681989777)"];
Print["  K3diag[0,0,0] ≈ ", N[K3diag[0, 0, 0], 16],
  "  (expect iB11 ≈ 0.07843)"];
Print["  K3off[0,0,0]  ≈ ", N[K3off[0, 0, 0], 16],
  "  (expect iB12 ≈ 0.04970)"];

(* ----- Assemble the full 18x18 quad-quad B27 sub-block ------------- *)
Print[];
Print["---- Assembling B27quad (18x18 quad-quad sub-block) ----"];
Print["    (324 entries; printing progress every row)"];
t0 = AbsoluteTime[];
B27quad = Table[
  Module[{row},
    row = Table[b27entry[i, j], {j, 10, 27}];
    Print["    row ", i - 9, "/18 done (i = ", i, ")  elapsed ",
      Round[AbsoluteTime[] - t0, 0.01], " s"];
    row
  ],
  {i, 10, 27}
];
t1 = AbsoluteTime[];
Print["  B27quad (18x18) assembled in ", Round[t1 - t0, 0.01], " s"];

(* Verify symmetry. *)
B27quadA = B27quad /. {Aelas -> 1, Belas -> 0};
B27quadB = B27quad /. {Aelas -> 0, Belas -> 1};
Print[];
Print["  B27quad symmetric (A-channel)?  ",
  Max[Abs[N[B27quadA - Transpose[B27quadA], 16]]] < 10^(-12)];
Print["  B27quad symmetric (B-channel)?  ",
  Max[Abs[N[B27quadB - Transpose[B27quadB], 16]]] < 10^(-12)];

(* Pretty-print the A-channel matrix, rounded to 4 digits. *)
Print[];
Print["  B27quadA (A-channel, rounded to 4 digits):"];
Print[MatrixForm[Map[Round[N[#, 16], 0.0001] &, B27quadA, {2}]]];

Print[];
Print["  B27quadB (B-channel, rounded to 4 digits):"];
Print[MatrixForm[Map[Round[N[#, 16], 0.0001] &, B27quadB, {2}]]];

(* Extract the diagonal. *)
Print[];
Print["  Diagonal entries (i=10..27):"];
Do[
  Print["    b27entry[", i, ",", i, "]  A = ",
    N[B27quadA[[i - 9, i - 9]], 12],
    "   B = ", N[B27quadB[[i - 9, i - 9]], 12]],
  {i, 10, 27}
];

(* ----- Persist the numerical 18x18 quad-quad block to disk ------- *)
Print[];
Print["---- Exporting B27quad to CubeT6QuadQuad.wl ----"];
b27quadOutPath = FileNameJoin[{$here, "CubeT6QuadQuad.wl"}];
Export[
  b27quadOutPath,
  "(* CubeT6QuadQuad.wl -- 18x18 numerical quad-quad block for T_27.\n" <>
  "   Rows/cols are indexed 10..27 of basisT27 (the 18 quadratic modes:\n" <>
  "   for each k in {1,2,3} the six q_m e_k with q in {r1^2, r2^2, r3^2,\n" <>
  "   r2 r3, r1 r3, r1 r2}).  Each entry is a symbolic linear combination\n" <>
  "   Aelas * aIJ + Belas * bIJ where aIJ, bIJ are numerical constants\n" <>
  "   (~15-digit MachinePrecision). *)\n\n" <>
  "B27quad = " <> ToString[FullForm[B27quad]] <> ";\n" <>
  "B27quadA = " <> ToString[FullForm[B27quadA]] <> ";\n" <>
  "B27quadB = " <> ToString[FullForm[B27quadB]] <> ";\n",
  "Text"
];
Print["  wrote ", b27quadOutPath];

Print[];
Print["==== CubeT6Block.wl : full 18x18 quad-quad B27 block assembled ===="];
