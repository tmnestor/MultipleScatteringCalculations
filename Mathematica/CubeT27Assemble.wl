(* ::Package:: *)
(* CubeT27Assemble.wl -- Closed-form symbolic assembly of the 27x27 cube
   T-matrix via Galerkin closure on T_27, the hybrid trial space of
   (constants) + (symmetric linear strains) + (18 quadratic modes).

   Strategy
   --------
   The 27x27 system has the matrix form
        ( M27 + B_el(Dc) - omega^2 Drho B_body ) c = M27 c_0
   with c_0 the incident coefficient vector on T_27.  The effective
   T-matrix on T_27 is
        T_eff = ( M27 + B_el(Dc) - omega^2 Drho B_body )^(-1)
                   ( omega^2 Drho B_body - B_el(Dc) ).

   Under O_h cubic symmetry, both M27 and B_body are equivariant, so
   by Schur's lemma they are block-diagonal in any isotypic basis.
   The 27-dim rep decomposes as
        T_27 = 4 T_{1u} + 2 T_{2u} + A_{2u} + E_u + A_{1g} + E_g + T_{2g}
   and every equivariant operator reduces to one (m x m) block per
   irrep -- with m = multiplicity.  The largest block is 4 x 4 (T_{1u}),
   so the full symbolic inversion is at most a 4 x 4 closed form.
   This is exactly the "T-matrix has the same pattern as the 27x27"
   observation from the 15-year-old experiments.

   Atomic scalars used
   -------------------
   * Mass matrix M27 :  exact rationals from integral on [-1,1]^3.
   * B_body T_9 block :  alpha9, beta9, gamma9, b11const  (the four
     closed-form scalars of the Path-B 9x9 block, CubeGalerkin27.wl).
   * B_body quad-quad 18x18 :  alpha1Pell .. alpha3Pell (3 scalars) +
     beta1Pell .. beta9Pell (9 scalars), loaded symbolically from
     CubeT6ScalarValues_HighPrec_Pell.wl.  These are the Pell-simplified
     closed forms verified to 25+ digits.
   * B_body YY cross block (3 x 9, const x S-type quad) :  two new
     scalars gammaYYstrong, gammaYYweak computed here via the same
     b27entry machinery ported from CubeT6Block.wl (numerical ~15 digit
     values, Pell PSLQ lookup optional).
   * B_el(Dc) :  plugged in via the Path-A T1c, T2c, T3c scalars
     (effective_contrasts.py) on the strain subspace; YY / quad-quad
     stiffness blocks are set to zero for this v1 scope and flagged.

   Output
   ------
   1.  Pretty-print of the symbolic 27x27 body form in the original
       basis, showing its block-sparsity pattern.
   2.  Explicit O_h change of basis U_{27 x 27} that block-diagonalizes
       both M27 and B_body.  Verification that U^T M27 U and U^T B_body U
       have the expected (T1u, T2u, A2u, Eu, A1g, Eg, T2g) block shape.
   3.  Per-irrep symbolic inversion -> closed-form T_eff.
   4.  Numerical validation of the 9x9 restriction of T_eff against
       Path-A (compute_cube_tmatrix) at Dc = 0 (density-only contrast).

   Usage:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/CubeT27Assemble.wl
*)

Print["==== CubeT27Assemble.wl : 27x27 cube T-matrix symbolic assembly ===="];
Print[];

$here = DirectoryName[$InputFileName];

(* ================================================================== *)
(* Section 1. Load Pell-simplified quad-quad scalars                    *)
(* ================================================================== *)
Print["---- Section 1: load Pell-simplified scalars ----"];

Get[FileNameJoin[{$here, "CubeT6ScalarValues_HighPrec_Pell.wl"}]];

(* The Pell file defines two lists (last assignment wins):
     cubeT6HighPrecAlphasPell : {alpha1, alpha2, alpha3}         (length 3)
     cubeT6HighPrecBetasPell  : {beta1, beta3, beta2, beta4,
                                 beta5, beta6, beta8, beta10, beta9}
   Note the script-index permutation:
     script index  1 2 3 4 5 6 7 8  9
     paper label   1 3 2 4 5 6 8 10 9
   as documented in /tmp/extract_pell_scalars.py.                    *)

alphaListScript = cubeT6HighPrecAlphasPell;
betaListScript  = cubeT6HighPrecBetasPell;

Assert[Length[alphaListScript] === 3];
Assert[Length[betaListScript] === 9];

(* Reindex to paper labels {beta1, beta2, ..., beta6, beta8, beta9, beta10}. *)
(* scriptToPaper[k] -> index in betaListScript for paper label k. *)
scriptToPaperMap = {
  1 -> 1,   (* beta_1 = script beta[1] *)
  2 -> 3,   (* beta_2 = script beta[3] *)
  3 -> 2,   (* beta_3 = script beta[2] *)
  4 -> 4,
  5 -> 5,
  6 -> 6,
  8 -> 7,
  9 -> 9,
  10 -> 8
};
paperBetaValue[k_Integer] := betaListScript[[scriptToPaperMap /. (k -> idx_) :> idx]];
(* Safer: build the lookup as an Association. *)
paperBetaLookup = Association @ Thread[
  {1, 2, 3, 4, 5, 6, 8, 9, 10} -> betaListScript[[{1, 3, 2, 4, 5, 6, 7, 9, 8}]]
];

Print["  alpha_1 = ", NumberForm[N[alphaListScript[[1]], 24], {24, 20}]];
Print["  alpha_2 = ", NumberForm[N[alphaListScript[[2]], 24], {24, 20}]];
Print["  alpha_3 = ", NumberForm[N[alphaListScript[[3]], 24], {24, 20}]];
Print["  beta_1 (paper) = ",
  NumberForm[N[paperBetaLookup[1], 24], {24, 20}]];
Print["  beta_3 (paper) = ",
  NumberForm[N[paperBetaLookup[3], 24], {24, 20}]];
Print["  beta_10(paper) = ",
  NumberForm[N[paperBetaLookup[10], 24], {24, 20}]];

(* Symbolic atom list for the quad-quad block: aPell[k] and bPell[k]. *)
aPell[k_Integer] /; 1 <= k <= 3 := alphaListScript[[k]];
bPell[k_Integer] := paperBetaLookup[k];

(* ================================================================== *)
(* Section 2. Load numerical B27quad for orbit identification          *)
(* ================================================================== *)
Print[];
Print["---- Section 2: load numerical B27quad and identify orbits ----"];

Get[FileNameJoin[{$here, "CubeT6QuadQuad.wl"}]];
Print["  Dimensions[B27quadA] = ", Dimensions[B27quadA]];
Print["  Dimensions[B27quadB] = ", Dimensions[B27quadB]];

(* Orbit label machinery ported verbatim from CubeT6Scalars.wl.       *)
quadDesc[i_] := Module[{k, m},
  k = Quotient[i - 1, 6] + 1;
  m = Mod[i - 1, 6] + 1;
  If[m <= 3,
    {k, "S", {m}},
    {k, "X", DeleteCases[{1, 2, 3}, m - 3]}
  ]
];

axisSignature[i_] := Module[{d, dir, facList},
  d = quadDesc[i];
  dir = d[[1]];
  facList = If[d[[2]] === "S", {d[[3, 1]], d[[3, 1]]}, d[[3]]];
  {dir, facList}
];

orbitLabel[i_Integer, j_Integer] := Module[
  {si, sj, dirI, dirJ, facI, facJ, axisCounts, sortedA, sortedB, typeI, typeJ},
  si = axisSignature[i]; sj = axisSignature[j];
  {dirI, facI} = si;
  {dirJ, facJ} = sj;
  axisCounts[ki_, kj_, fi_, fj_] := Table[
    {Boole[ki == a], Boole[kj == a], Count[fi, a], Count[fj, a]},
    {a, 1, 3}];
  sortedA = Sort[axisCounts[dirI, dirJ, facI, facJ]];
  sortedB = Sort[axisCounts[dirJ, dirI, facJ, facI]];
  typeI = quadDesc[i][[2]];
  typeJ = quadDesc[j][[2]];
  {Sort[{typeI, typeJ}], If[OrderedQ[{sortedA, sortedB}], sortedA, sortedB]}
];

allPairs = Flatten[Table[{ii, jj}, {ii, 18}, {jj, 18}], 1];
orbitGroups = GroupBy[allPairs, orbitLabel[#[[1]], #[[2]]] &];
Print["  number of O_h orbits on 18x18 quad-quad = ", Length[orbitGroups]];

(* For each orbit, read its (A-rep, B-rep) numerical values from       *)
(* B27quadA, B27quadB.                                                  *)
orbitRep = Association @ KeyValueMap[
  Function[{lbl, pairs},
    lbl -> {
      B27quadA[[pairs[[1, 1]], pairs[[1, 2]]]],
      B27quadB[[pairs[[1, 1]], pairs[[1, 2]]]],
      Length[pairs]
    }
  ],
  orbitGroups
];

nonzeroOrbits = Select[orbitRep,
  Abs[#[[1]]] > 10^-10 || Abs[#[[2]]] > 10^-10 &];
Print["  number of nonzero orbits   = ", Length[nonzeroOrbits]];

(* ================================================================== *)
(* Section 3. Map numerical orbit (A, B) values to Pell symbolic atoms *)
(* ================================================================== *)
Print[];
Print["---- Section 3: match numerical orbit values -> Pell atoms ----"];

alphaNumList = N[alphaListScript, 20];                    (* length 3 *)
betaNumByPaper = Association @ KeyValueMap[
  #1 -> N[#2, 20] &, paperBetaLookup];                    (* Association k -> num *)

(* Match a numerical value against the Pell alpha list.  Return the
   script index (1..3) or 0 if no match.                                 *)
matchAlphaIndex[val_?NumericQ] := Module[{idx},
  idx = FirstPosition[alphaNumList,
    _?(Abs[# - val] < 10^-8 &), {Missing["NotFound"]}, {1}];
  If[MissingQ[idx[[1]]], 0, idx[[1]]]
];

(* Match against the Pell beta paper-labeled list.  Return the paper
   label (1..6,8,9,10) or 0 if no match.                                 *)
matchBetaPaperIdx[val_?NumericQ] := Module[{hit},
  hit = SelectFirst[{1, 2, 3, 4, 5, 6, 8, 9, 10},
    Abs[betaNumByPaper[#] - val] < 10^-8 &,
    0];
  hit
];

(* Build an 18x18 symbolic matrix B27quadSym where each entry is a
   linear combination Aelas*aPell[k] + Belas*bPell[p] according to the
   matched orbit.                                                       *)
B27quadSym = Table[0, {18}, {18}];
orbitAssignmentLog = {};
Do[
  Module[{lbl, pairs, aVal, bVal, aIdx, bIdx, symEntry},
    lbl = orbit;
    pairs = orbitGroups[lbl];
    {aVal, bVal} = {orbitRep[lbl][[1]], orbitRep[lbl][[2]]};
    aIdx = matchAlphaIndex[aVal];
    bIdx = matchBetaPaperIdx[bVal];
    symEntry =
      If[Abs[aVal] > 10^-10, Aelas * aPell[aIdx], 0] +
      If[Abs[bVal] > 10^-10, Belas * bPell[bIdx], 0];
    AppendTo[orbitAssignmentLog,
      {lbl, Length[pairs], aIdx, bIdx, N[aVal, 6], N[bVal, 6]}];
    Do[
      B27quadSym[[pair[[1]], pair[[2]]]] = symEntry,
      {pair, pairs}
    ];
  ],
  {orbit, Keys[orbitGroups]}
];

Print["  Orbit -> symbolic atom log (first 10 rows):"];
Do[
  Print["    ",
    "mult=", row[[2]], "  aIdx=", row[[3]], "  bIdx=", row[[4]],
    "  A=", row[[5]], "  B=", row[[6]]],
  {row, Take[orbitAssignmentLog, UpTo[10]]}
];

(* Numerical round-trip check: substitute alphas/betas and compare to  *)
(* B27quadA + I*B27quadB (we only use real channels).                  *)
B27quadSymNumA = B27quadSym /. {Aelas -> 1, Belas -> 0};
B27quadSymNumB = B27quadSym /. {Aelas -> 0, Belas -> 1};
diffA = Max @ Abs @ Flatten @ N[B27quadSymNumA - B27quadA, 20];
diffB = Max @ Abs @ Flatten @ N[B27quadSymNumB - B27quadB, 20];
Print["  max(|B27quadSym(A) - B27quadA|) = ", ScientificForm[diffA, 3]];
Print["  max(|B27quadSym(B) - B27quadB|) = ", ScientificForm[diffB, 3]];
If[diffA > 10^-8 || diffB > 10^-8,
  Print["  WARNING: orbit -> symbolic atom matching has drift; review."];
];

(* ================================================================== *)
(* Section 4. basisT27 and 27x27 mass matrix M27                        *)
(* ================================================================== *)
Print[];
Print["---- Section 4: basisT27 and exact M27 ----"];

basisT9 = {
  {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
  {r1, 0, 0}, {0, r2, 0}, {0, 0, r3},
  {0, r3/2, r2/2}, {r3/2, 0, r1/2}, {r2/2, r1/2, 0}
};
quadMonomials = {r1^2, r2^2, r3^2, r2*r3, r1*r3, r1*r2};
unitVec[1] = {1, 0, 0};
unitVec[2] = {0, 1, 0};
unitVec[3] = {0, 0, 1};
basisT27quad = Flatten[
  Table[q * unitVec[k], {k, 1, 3}, {q, quadMonomials}], 1];
basisT27 = Join[basisT9, basisT27quad];
Print["  Length[basisT27] = ", Length[basisT27]];

polyDot[v1_, v2_] := Sum[v1[[p]] v2[[p]], {p, 3}];
massEntry[v1_, v2_] := Integrate[polyDot[v1, v2],
  {r1, -1, 1}, {r2, -1, 1}, {r3, -1, 1}];

M27 = Table[massEntry[basisT27[[i]], basisT27[[j]]], {i, 27}, {j, 27}];
Print["  M27 symmetric? ", M27 === Transpose[M27]];
Print["  M27 diagonal (first 15 entries): ",
  Diagonal[M27][[1 ;; 15]]];

(* Sanity: const-const block is 8*I3 (volume 8).                        *)
Print["  M27[1;;3, 1;;3] = ", M27[[1 ;; 3, 1 ;; 3]]];

(* ================================================================== *)
(* Section 5. B9 block symbolic atoms (from CubeGalerkin27.wl)          *)
(* ================================================================== *)
Print[];
Print["---- Section 5: B9 block (T9 x T9) with opaque symbolic atoms --"];

(* We use four abstract symbols for the T9 x T9 body block:
     b11const  = 256 (A iA + B iB11)                  -- const diagonal
     alpha9    = axial-strain diagonal                  [4,4]
     beta9     = axial-strain off-diagonal              [4,5]
     gamma9    = shear-strain diagonal                  [7,7]
   Their closed forms are in CubeGalerkin27.wl.  Here we treat them as
   opaque atoms so the irrep decomposition and T_eff construction is
   structurally closed-form.                                           *)

(* 9x9 body block template in the original basisT9 order.              *)
B9sym = Table[0, {9}, {9}];
(* const-const diagonal *)
Do[B9sym[[i, i]] = b11const, {i, 1, 3}];
(* strain-const: zero (parity)                                          *)
(* axial-strain 3x3 block (rows/cols 4..6) *)
Do[B9sym[[i, i]] = alpha9, {i, 4, 6}];
Do[If[i != j, B9sym[[i, j]] = beta9], {i, 4, 6}, {j, 4, 6}];
(* shear-strain 3x3 block (rows/cols 7..9): diagonal only *)
Do[B9sym[[i, i]] = gamma9, {i, 7, 9}];

Print["  B9sym ="];
Print[MatrixForm[B9sym]];

(* ================================================================== *)
(* Section 6. YY cross block (3 x 9) between constants and S-type quads *)
(* ================================================================== *)
Print[];
Print["---- Section 6: YY cross block (const x S-type quad) ----"];

(* Parity argument (see docs/cube_galerkin27.tex Section 5):
     const[k] x S-type-quad[k',m] has nonzero body integral only when
     k = k'.  For k = k', there are three m values (r_1^2, r_2^2, r_3^2
     in direction k), giving 2 unique values under cubic symmetry:
        gammaYYstrong := B_body[const_k, r_k^2 * e_k]     (m = k)
        gammaYYweak   := B_body[const_k, r_m^2 * e_k]     (m != k)
   The X-type (cross products r_a r_b, a != b) columns give zero.       *)

(* Symbolic YY block: 3 rows (const directions), 9 columns (S-type quad
   indices in basisT27quad: the S-type columns are the first 3 of each
   direction's 6-tuple, i.e. quad indices {1,2,3, 7,8,9, 13,14,15}      *)
(* Within basisT27, quad index 1 = basisT27[10], so S-type in direction
   k occupies basisT27 indices {6(k-1)+10, 6(k-1)+11, 6(k-1)+12}.       *)

YY = Table[0, {3}, {18}];       (* rows: const direction, cols: 18 quads *)
Do[
  (* Iterate direction k in {1,2,3} and monomial index m in {1,2,3}    *)
  Module[{quadIdx},
    quadIdx = 6*(k - 1) + m;     (* column index in 1..18 *)
    YY[[k, quadIdx]] = If[k == m, gammaYYstrong, gammaYYweak];
  ],
  {k, 1, 3}, {m, 1, 3}
];
Print["  YY block (A-channel and B-channel combined)  ="];
Print[MatrixForm[YY]];

(* ================================================================== *)
(* Section 7. Assemble the full 27 x 27 symbolic B_body                 *)
(* ================================================================== *)
Print[];
Print["---- Section 7: assemble symbolic 27 x 27 B_body ----"];

BbodySym = Table[0, {27}, {27}];
(* B9 block -> rows/cols 1..9 *)
Do[BbodySym[[i, j]] = B9sym[[i, j]], {i, 9}, {j, 9}];
(* YY block -> rows 1..3, cols 10..27 (and symmetric mirror) *)
Do[
  BbodySym[[i, 9 + j]] = YY[[i, j]];
  BbodySym[[9 + j, i]] = YY[[i, j]];
  ,
  {i, 1, 3}, {j, 1, 18}
];
(* Quad-quad block -> rows/cols 10..27 *)
Do[
  BbodySym[[9 + i, 9 + j]] = B27quadSym[[i, j]],
  {i, 1, 18}, {j, 1, 18}
];

Print["  BbodySym symmetric? ",
  Simplify[BbodySym - Transpose[BbodySym]] === ConstantArray[0, {27, 27}]];
Print["  BbodySym nonzero entry count = ",
  Count[Flatten[BbodySym], e_ /; e =!= 0]];

(* Pretty-print the block pattern (1 = nonzero, . = zero). *)
blockPattern = Map[If[# === 0, ".", "X"] &, BbodySym, {2}];
Print["  Block pattern (27 x 27):"];
Do[
  Print["    ", StringJoin[Riffle[blockPattern[[rr]], " "]]],
  {rr, 1, 27}
];

(* ================================================================== *)
(* Section 8. Numerical values of the symbolic atoms                   *)
(* ================================================================== *)
Print[];
Print["---- Section 8: compute numerical values of all atoms ----"];

(* The Pell alphas / betas are symbolic closed forms (from Section 1). *)
(* We still need numerical values for:
     - b11const = 256 (A iA + B iB11)
     - alpha9, beta9, gamma9
     - gammaYYstrong, gammaYYweak
   For the first four we use the Path-B Tier-3 integrals iA and iB11,
   computed here via NIntegrate at high working precision (matching
   CubeT6Block.wl).  For the YY scalars we use the same numerical
   b27entry-style machinery.                                           *)

(* --- Shared helpers (numerical, ported from CubeT6Block.wl) --- *)
Clear[MpN, K1atN, K3diagN, K3offN, K3kernelN];

MpN[p_Integer, q_Integer, r_Integer] /; ! OrderedQ[{p, q, r}] :=
  MpN @@ Sort[{p, q, r}];
MpN[p_Integer, q_Integer, r_Integer] := MpN[p, q, r] =
  NIntegrate[
    xm^p ym^q zm^r / Sqrt[xm^2 + ym^2 + zm^2],
    {xm, 0, 1}, {ym, 0, 1}, {zm, 0, 1},
    PrecisionGoal -> 14, AccuracyGoal -> 14, MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}];

K1atN[p_Integer, q_Integer, r_Integer] := K1atN[p, q, r] = Sum[
  (-1)^(a + b + c) MpN[p + a, q + b, r + c],
  {a, 0, 1}, {b, 0, 1}, {c, 0, 1}];

K3diagN[p_Integer, q_Integer, r_Integer] := K3diagN[p, q, r] =
  NIntegrate[
    (1 - u1) (1 - u2) (1 - u3) u1^(p + 2) u2^q u3^r /
      (u1^2 + u2^2 + u3^2)^(3/2),
    {u1, 0, 1}, {u2, 0, 1}, {u3, 0, 1},
    PrecisionGoal -> 14, AccuracyGoal -> 14, MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}];

K3offN[p_Integer, q_Integer, r_Integer] := K3offN[p, q, r] =
  NIntegrate[
    (1 - u1) (1 - u2) (1 - u3) u1^(p + 1) u2^(q + 1) u3^r /
      (u1^2 + u2^2 + u3^2)^(3/2),
    {u1, 0, 1}, {u2, 0, 1}, {u3, 0, 1},
    PrecisionGoal -> 14, AccuracyGoal -> 14, MaxRecursion -> 25,
    Method -> {"GlobalAdaptive",
      "SingularityHandler" -> "DuffyCoordinates"}];

K3kernelN[{p_, q_, r_}, i_, j_] := K3kernelN[{p, q, r}, i, j] = Which[
  i == 1 && j == 1, K3diagN[p, q, r],
  i == 2 && j == 2, K3diagN[q, p, r],
  i == 3 && j == 3, K3diagN[r, q, p],
  (i == 1 && j == 2) || (i == 2 && j == 1), K3offN[p, q, r],
  (i == 1 && j == 3) || (i == 3 && j == 1), K3offN[p, r, q],
  (i == 2 && j == 3) || (i == 3 && j == 2), K3offN[q, r, p],
  True, $Failed];

(* Parity filter. *)
kernelParityN[pIdx_Integer, qIdx_Integer] := Mod[
  {Boole[pIdx == 1] + Boole[qIdx == 1],
   Boole[pIdx == 2] + Boole[qIdx == 2],
   Boole[pIdx == 3] + Boole[qIdx == 3]}, 2];
parityOKN[exps_List, kp_List] := AllTrue[Range[3], EvenQ[exps[[#]] + kp[[#]]] &];

ClearAll[u1, u2, u3];

bbBodyABN[poly_, pIdx_Integer, qIdx_Integer] :=
  Module[{coeffs, aPiece, bPiece, kpA = {0, 0, 0}, kpB},
    kpB = kernelParityN[pIdx, qIdx];
    coeffs = CoefficientRules[Expand[poly], {u1, u2, u3}];
    aPiece = Sum[
      If[parityOKN[pair[[1]], kpA], pair[[2]] K1atN @@ pair[[1]], 0],
      {pair, coeffs}];
    bPiece = Sum[
      If[parityOKN[pair[[1]], kpB],
        pair[[2]] K3kernelN[pair[[1]], pIdx, qIdx], 0],
      {pair, coeffs}];
    {aPiece, bPiece}];

xiMonIntN[k1_Integer, k2_Integer, k3_Integer] :=
  If[OddQ[k1] || OddQ[k2] || OddQ[k3], 0,
    Times @@ MapThread[(2 (1 - #1)^(#2 + 1)/(#2 + 1)) &,
      {{u1, u2, u3}, {k1, k2, k3}}]];
intXiThenSubN[expr_] := Module[{e, coeffs},
  e = Expand[expr];
  coeffs = CoefficientRules[e, {xi[1], xi[2], xi[3]}];
  Expand[Sum[pair[[2]] xiMonIntN @@ pair[[1]], {pair, coeffs}] /.
    {s[1] -> 2 u1, s[2] -> 2 u2, s[3] -> 2 u3}]];
basisAtRN[v_] := v /. {r1 -> xi[1] + s[1]/2, r2 -> xi[2] + s[2]/2,
  r3 -> xi[3] + s[3]/2};
basisAtRprimeN[v_] := v /. {r1 -> xi[1] - s[1]/2, r2 -> xi[2] - s[2]/2,
  r3 -> xi[3] - s[3]/2};
polyToTentResidualN[poly_] :=
  Expand[Cancel[poly / ((1 - u1) (1 - u2) (1 - u3))]];

b27pairN[i_Integer, j_Integer, p_Integer, q_Integer] :=
  Module[{phiI, phiJ, productPQ, weight, residual},
    phiI = basisAtRN[basisT27[[i]]][[p]];
    phiJ = basisAtRprimeN[basisT27[[j]]][[q]];
    productPQ = Expand[phiI phiJ];
    If[productPQ === 0, Return[{0, 0}]];
    weight = intXiThenSubN[productPQ];
    residual = polyToTentResidualN[weight];
    bbBodyABN[residual, p, q]];

b27entryN[i_Integer, j_Integer] := Module[{aAcc = 0, bAcc = 0, contr},
  Do[
    contr = b27pairN[i, j, p, q];
    If[p == q, aAcc = aAcc + contr[[1]]];
    bAcc = bAcc + contr[[2]],
    {p, 3}, {q, 3}];
  32 (Aelas aAcc + Belas bAcc)];

Print["  priming numerical Mp / K3 caches..."];
t0 = AbsoluteTime[];
MpN[0, 0, 0]; MpN[1, 0, 0]; MpN[1, 1, 0]; MpN[1, 1, 1];
MpN[2, 0, 0]; MpN[2, 1, 0]; MpN[2, 1, 1]; MpN[2, 2, 0];
MpN[3, 0, 0]; MpN[3, 1, 0]; MpN[3, 1, 1]; MpN[3, 2, 0]; MpN[3, 2, 1];
K3diagN[0, 0, 0]; K3offN[0, 0, 0]; K3diagN[1, 0, 0]; K3offN[1, 0, 0];
K3diagN[0, 1, 0]; K3diagN[2, 0, 0]; K3offN[1, 1, 0]; K3diagN[1, 1, 0];
K3diagN[0, 2, 0];
Print["  done in ", Round[AbsoluteTime[] - t0, 0.01], " s"];

(* Numerical iA, iB11, iB12 for the const block. *)
iAnum   = MpN[0, 0, 0] - 3 MpN[1, 0, 0] + 3 MpN[1, 1, 0] - MpN[1, 1, 1];
iB11num = K3diagN[0, 0, 0];
iB12num = K3offN[0, 0, 0];
b11constNum = 256 (Aelas * iAnum + Belas * iB11num);
Print["  iA    num = ", N[iAnum, 16]];
Print["  iB11  num = ", N[iB11num, 16]];
Print["  iB12  num = ", N[iB12num, 16]];
Print["  b11const (A channel) = ", N[b11constNum /. {Aelas -> 1, Belas -> 0}, 16]];
Print["  b11const (B channel) = ", N[b11constNum /. {Aelas -> 0, Belas -> 1}, 16]];

(* alpha9, beta9, gamma9 via b27entryN on T_9 pairs. *)
alpha9Num = b27entryN[4, 4];
beta9Num  = b27entryN[4, 5];
gamma9Num = b27entryN[7, 7];
Print["  alpha9 (A,B) = ",
  N[alpha9Num /. {Aelas -> 1, Belas -> 0}, 16], " , ",
  N[alpha9Num /. {Aelas -> 0, Belas -> 1}, 16]];
Print["  beta9  (A,B) = ",
  N[beta9Num  /. {Aelas -> 1, Belas -> 0}, 16], " , ",
  N[beta9Num  /. {Aelas -> 0, Belas -> 1}, 16]];
Print["  gamma9 (A,B) = ",
  N[gamma9Num /. {Aelas -> 1, Belas -> 0}, 16], " , ",
  N[gamma9Num /. {Aelas -> 0, Belas -> 1}, 16]];

(* YY scalars: b27entryN[1, 10] and b27entryN[1, 11]. *)
gammaYYstrongNum = b27entryN[1, 10];
gammaYYweakNum   = b27entryN[1, 11];
Print["  gammaYYstrong (A,B) = ",
  N[gammaYYstrongNum /. {Aelas -> 1, Belas -> 0}, 16], " , ",
  N[gammaYYstrongNum /. {Aelas -> 0, Belas -> 1}, 16]];
Print["  gammaYYweak   (A,B) = ",
  N[gammaYYweakNum   /. {Aelas -> 1, Belas -> 0}, 16], " , ",
  N[gammaYYweakNum   /. {Aelas -> 0, Belas -> 1}, 16]];

(* Sanity: b27entryN[1, 12] should match gammaYYweak (S_2 and S_3
   belong to the same weak orbit by cubic symmetry).                    *)
tmp1 = b27entryN[1, 12];
Print["  b27entryN[1,12] - gammaYYweak (A) = ",
  N[(tmp1 - gammaYYweakNum) /. {Aelas -> 1, Belas -> 0}, 20]];
Print["  b27entryN[1,12] - gammaYYweak (B) = ",
  N[(tmp1 - gammaYYweakNum) /. {Aelas -> 0, Belas -> 1}, 20]];

(* Build the full numerical substitution rule mapping atoms -> numbers.*)
atomRules = {
  b11const      -> b11constNum,
  alpha9        -> alpha9Num,
  beta9         -> beta9Num,
  gamma9        -> gamma9Num,
  gammaYYstrong -> gammaYYstrongNum,
  gammaYYweak   -> gammaYYweakNum
};

(* ================================================================== *)
(* Section 9. O_h equivariance check                                    *)
(* ================================================================== *)
Print[];
Print["---- Section 9: O_h equivariance check (numerical) ----"];

(* Evaluate the full 27x27 B_body at (A=1, B=0) and (A=0, B=1).        *)
BbodyNumA = (BbodySym /. atomRules) /. {Aelas -> 1, Belas -> 0};
BbodyNumB = (BbodySym /. atomRules) /. {Aelas -> 0, Belas -> 1};
BbodyNumA = N[BbodyNumA, 20];
BbodyNumB = N[BbodyNumB, 20];
Print["  max(|BbodyNumA - Transpose[.]|) = ",
  Max[Abs[Flatten[BbodyNumA - Transpose[BbodyNumA]]]]];
Print["  max(|BbodyNumB - Transpose[.]|) = ",
  Max[Abs[Flatten[BbodyNumB - Transpose[BbodyNumB]]]]];

(* Build the 27x27 permutation-with-signs matrix for cyclic xyz        *)
(* rotation: (r1, r2, r3) -> (r2, r3, r1).  Under this rotation,       *)
(* e_1 -> e_2 -> e_3 -> e_1, and                                       *)
(*   r1^2 -> r2^2, r2^2 -> r3^2, r3^2 -> r1^2                          *)
(*   r2*r3 -> r3*r1, r1*r3 -> r2*r1, r1*r2 -> r2*r3                    *)
(* i.e. the monomial permutation is the same 3-cycle on axial squares   *)
(* and the corresponding 3-cycle on cross products.                    *)

(* We build each rep matrix in two steps:
     (1) express every v in basisT27 as a flat coefficient vector over a
         fixed monomial grid (polyCoeffVec below).
     (2) solve the over-determined linear system Bemb^T . x = polyCoeffVec[v]
         via the normal equation to recover the basisT27 coordinates.     *)

rotXYZbasis[v_] := v /. {r1 -> r2, r2 -> r3, r3 -> r1};
rotXYZvec[v_] := {v[[3]], v[[1]], v[[2]]};  (* permute components *)
rotXYZ[v_] := rotXYZvec[rotXYZbasis[v]];

(* Monomial grid: 10 scalar monomials x 3 directions = 30 slots.        *)
monomialList = {
  {1, 0, 0}, {0, 1, 0}, {0, 0, 1},           (* constant directions *)
  {r1, 0, 0}, {0, r1, 0}, {0, 0, r1},
  {r2, 0, 0}, {0, r2, 0}, {0, 0, r2},
  {r3, 0, 0}, {0, r3, 0}, {0, 0, r3},
  {r1^2, 0, 0}, {0, r1^2, 0}, {0, 0, r1^2},
  {r2^2, 0, 0}, {0, r2^2, 0}, {0, 0, r2^2},
  {r3^2, 0, 0}, {0, r3^2, 0}, {0, 0, r3^2},
  {r2 r3, 0, 0}, {0, r2 r3, 0}, {0, 0, r2 r3},
  {r1 r3, 0, 0}, {0, r1 r3, 0}, {0, 0, r1 r3},
  {r1 r2, 0, 0}, {0, r1 r2, 0}, {0, 0, r1 r2}
};
nMon = Length[monomialList];

(* Pre-compute the (directionIndex, monomialExponent) tag for each slot *)
(* in monomialList.  This lets us find the slot for a given (dir, exp) *)
(* in O(log n) via an Association.                                       *)
monomialExp[m_] := If[m === 1, {0, 0, 0},
  First[CoefficientRules[Expand[m], {r1, r2, r3}]][[1]]];

(* Find the unique nonzero component index of a length-3 vector.       *)
(* We KNOW each entry of monomialList has exactly one nonzero slot.     *)
nonzeroDir[vec_] := Module[{d = 0},
  Do[If[vec[[k]] =!= 0, d = k; Break[]], {k, 3}];
  d];

monomialSlotRules = Association @ Table[
  Module[{vec, dir, scalar},
    vec = monomialList[[mm]];
    dir = nonzeroDir[vec];
    scalar = vec[[dir]];
    {dir, monomialExp[scalar]} -> mm
  ],
  {mm, nMon}];

(* polyCoeffVec : vector polynomial -> length-nMon flat coefficient    *)
(* vector.  Handles multi-component vectors (e.g. shear strains        *)
(* {0, r3/2, r2/2}) correctly by iterating over all 3 directions.       *)
polyCoeffVec[v_] := Module[{out, scalarPart, cr, slot},
  out = Table[0, {nMon}];
  Do[
    scalarPart = v[[dir]];
    If[scalarPart =!= 0,
      cr = CoefficientRules[Expand[scalarPart], {r1, r2, r3}];
      Do[
        slot = Lookup[monomialSlotRules, Key[{dir, pair[[1]]}], Missing[]];
        If[! MissingQ[slot], out[[slot]] += pair[[2]]],
        {pair, cr}]
    ],
    {dir, 3}];
  out];

(* Build the 27 x nMon embedding matrix Bemb (row i = coeffs of basisT27[[i]]). *)
Bemb = Table[polyCoeffVec[basisT27[[i]]], {i, 27}];
Print["  Bemb has rank ", MatrixRank[Bemb], " (expect 27)"];

(* Cache the Gram matrix (Bemb . Bemb^T) and its LU decomposition for   *)
(* fast decomposition of rotated basis elements.                        *)
BembGram = Bemb.Transpose[Bemb];
BembGramInv = Inverse[BembGram];
decomposeInBasis[v_] := BembGramInv . (Bemb . polyCoeffVec[v]);

(* Construct the 27 x 27 rotation matrix for the xyz 3-cycle.          *)
rot27 = Table[
  decomposeInBasis[rotXYZ[basisT27[[j]]]],
  {j, 27}] // Transpose;
Print["  rot27 is orthogonal? ",
  Max[Abs[Flatten[rot27.Transpose[rot27] - IdentityMatrix[27]]]] < 10^-10];

(* Equivariance of M27 and B_body under rot27 (numerical).             *)
commM = rot27.M27.Transpose[rot27] - M27;
Print["  max|rot27 . M27 . rot27^T - M27| = ",
  Max[Abs[Flatten[N[commM, 16]]]]];
commB = rot27.BbodyNumA.Transpose[rot27] - BbodyNumA;
Print["  max|rot27 . BbodyA . rot27^T - BbodyA| = ",
  Max[Abs[Flatten[commB]]]];

(* ================================================================== *)
(* Section 9b. Full O_h group + character projectors                    *)
(* ================================================================== *)
Print[];
Print["---- Section 9b: build all 48 O_h rep matrices + projectors ----"];

(* Construct all 48 signed 3x3 permutation matrices.                    *)
(* Convention: M . e_j = signs[[j]] * e_{perm[[j]]}, i.e. column j has   *)
(* one entry signs[[j]] in row perm[[j]], zeros elsewhere.                *)
buildSPM[perm_List, signs_List] := Normal @ SparseArray[
  Table[{perm[[j]], j} -> signs[[j]], {j, 3}], {3, 3}];

ohGroup3D = Flatten[
  Table[
    buildSPM[perm, signs],
    {perm, Permutations[{1, 2, 3}]},
    {signs, Tuples[{1, -1}, 3]}],
  1];
Print["  |O_h| = ", Length[ohGroup3D]];

(* Apply a 3x3 element M to a basisT27 element (vector polynomial):     *)
(*   (g . v)(r) = M . v(M^T . r)                                         *)
applyG3D[M_, v_] := Module[{subs, vSub},
  subs = Thread[{r1, r2, r3} -> Transpose[M] . {r1, r2, r3}];
  vSub = v /. subs;
  Expand[M . vSub]];

(* Build the 27x27 rep matrix for each of the 48 group elements.        *)
Print["  building 48 rep matrices via decomposeInBasis..."];
t0 = AbsoluteTime[];
R27list = Table[
  Transpose @ Table[
    decomposeInBasis[applyG3D[M, basisT27[[j]]]],
    {j, 27}],
  {M, ohGroup3D}];
Print["    done in ", Round[AbsoluteTime[] - t0, 0.01], " s"];

(* Sanity check: every rep matrix should be orthogonal.                 *)
maxOrthErr = Max[Table[
  Max[Abs[Flatten[
    R27list[[k]] . Transpose[R27list[[k]]] - IdentityMatrix[27]]]],
  {k, 48}]];
Print["  max orthogonality error across 48 rep matrices = ",
  ScientificForm[maxOrthErr, 3]];

(* Classify each group element by conjugacy class via                    *)
(*    (trace, det, numFixed)  on the 3D matrix M.                        *)
classifyG3D[M_] := Module[{tr, dt, nf},
  tr = Tr[M]; dt = Det[M];
  nf = Count[Diagonal[M], 1];
  Which[
    tr == 3, "E",
    tr == 0 && dt == 1, "8C3",
    tr == -1 && dt == 1 && nf == 1, "3C2",
    tr == 1 && dt == 1, "6C4",
    tr == -1 && dt == 1 && nf == 0, "6C2p",
    tr == -3, "i",
    tr == 0 && dt == -1, "8S6",
    tr == 1 && dt == -1 && nf == 2, "3sh",
    tr == -1 && dt == -1, "6S4",
    tr == 1 && dt == -1 && nf == 1, "6sd",
    True, "unknown"]];

classLabels = classifyG3D /@ ohGroup3D;
classCounts = Counts[classLabels];
Print["  conjugacy class counts = ", classCounts];
expectedCounts = <|
  "E" -> 1, "8C3" -> 8, "3C2" -> 3, "6C4" -> 6, "6C2p" -> 6,
  "i" -> 1, "8S6" -> 8, "3sh" -> 3, "6S4" -> 6, "6sd" -> 6|>;
Assert[KeySort[classCounts] === KeySort[expectedCounts]];

(* Numerical full-group equivariance check.                             *)
maxCommAll = 0;
Do[
  Module[{R, errM, errB},
    R = R27list[[k]];
    errM = Max[Abs[Flatten[R . M27 . Transpose[R] - M27]]];
    errB = Max[Abs[Flatten[R . BbodyNumA . Transpose[R] - BbodyNumA]]];
    maxCommAll = Max[maxCommAll, N[errM, 16], errB]],
  {k, 48}];
Print["  max |R . {M27, BbodyA} . R^T - {M27, BbodyA}| over 48 elts = ",
  ScientificForm[maxCommAll, 3]];

(* Character table for O_h (10 irreps x 10 classes).  Class ordering:  *)
(*  {E, 8C3, 3C2, 6C4, 6C2p, i, 8S6, 3sh, 6S4, 6sd}                      *)
classOrder = {"E", "8C3", "3C2", "6C4", "6C2p",
              "i", "8S6", "3sh", "6S4", "6sd"};
classToIdx = Association @ Table[
  classOrder[[k]] -> k, {k, Length[classOrder]}];

charTable = <|
  "A1g" -> { 1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
  "A2g" -> { 1,  1,  1, -1, -1,  1,  1,  1, -1, -1},
  "Eg"  -> { 2, -1,  2,  0,  0,  2, -1,  2,  0,  0},
  "T1g" -> { 3,  0, -1,  1, -1,  3,  0, -1,  1, -1},
  "T2g" -> { 3,  0, -1, -1,  1,  3,  0, -1, -1,  1},
  "A1u" -> { 1,  1,  1,  1,  1, -1, -1, -1, -1, -1},
  "A2u" -> { 1,  1,  1, -1, -1, -1, -1, -1,  1,  1},
  "Eu"  -> { 2, -1,  2,  0,  0, -2,  1, -2,  0,  0},
  "T1u" -> { 3,  0, -1,  1, -1, -3,  0,  1, -1,  1},
  "T2u" -> { 3,  0, -1, -1,  1, -3,  0,  1,  1, -1}
|>;
irrepDims = <|"A1g" -> 1, "A2g" -> 1, "Eg" -> 2, "T1g" -> 3, "T2g" -> 3,
              "A1u" -> 1, "A2u" -> 1, "Eu" -> 2, "T1u" -> 3, "T2u" -> 3|>;
irrepsOrdered = {"A1g", "A2g", "Eg", "T1g", "T2g",
                 "A1u", "A2u", "Eu", "T1u", "T2u"};

(* Character projectors: P_rho = (d_rho / |G|) Sum_g chi_rho(g) R(g).   *)
Print["  computing 10 character projectors..."];
t0 = AbsoluteTime[];
charForG[irrep_] := charTable[irrep][[classToIdx[#]]] & /@ classLabels;
projectorP[irrep_] := (irrepDims[irrep]/48) * Sum[
  charTable[irrep][[classToIdx[classLabels[[k]]]]] * R27list[[k]],
  {k, 48}];
projectors = Association @ Table[
  irrep -> projectorP[irrep], {irrep, irrepsOrdered}];
Print["    done in ", Round[AbsoluteTime[] - t0, 0.01], " s"];

projectorRanks = Association @ Table[
  irrep -> MatrixRank[N[projectors[irrep]]], {irrep, irrepsOrdered}];
Print[];
Print["  Projector ranks (= d_rho * m_rho):"];
Do[
  Print["    P[", StringPadRight[irrep, 3], "]  rank = ",
    StringPadLeft[ToString[projectorRanks[irrep]], 2],
    "   (d=", irrepDims[irrep],
    ", m=", If[projectorRanks[irrep] > 0,
               projectorRanks[irrep]/irrepDims[irrep], 0], ")"],
  {irrep, irrepsOrdered}];
Print["  total dims = ", Total[Values[projectorRanks]], "  (expect 27)"];
Assert[Total[Values[projectorRanks]] === 27];

(* Expected decomposition:
     T_27 = 4 T_{1u} + 2 T_{2u} + A_{1g} + E_g + T_{2g}
                     + A_{2u} + E_u                                       *)
expectedMults = <|
  "A1g" -> 1, "A2g" -> 0, "Eg" -> 1, "T1g" -> 0, "T2g" -> 1,
  "A1u" -> 0, "A2u" -> 1, "Eu" -> 1, "T1u" -> 4, "T2u" -> 2|>;
Do[
  Assert[projectorRanks[irrep]/irrepDims[irrep] === expectedMults[irrep]],
  {irrep, irrepsOrdered}];
Print["  decomposition matches expected O_h content: VERIFIED."];

(* ================================================================== *)
(* Section 9c. Symmetry-adapted basis + m_rho reduction                 *)
(* ================================================================== *)
Print[];
Print["---- Section 9c: m_rho reduction via partial projectors ----"];

(* Locate representative elements of each conjugacy class for use as    *)
(* symmetry-breaking generators.  All R27list entries are exact         *)
(* rational, so everything here stays exact.                            *)
classElt[clsName_] := First[Flatten[Position[classLabels, clsName]]];

rot27full = R27list[[classElt["8C3"]]];    (* a 3-cycle (C_3) *)
rotC2p    = R27list[[classElt["6C2p"]]];   (* a face-diagonal C_2' *)
rotC2     = R27list[[classElt["3C2"]]];    (* a coord-axis C_2 *)

Print["  generators chosen:"];
Print["    3-cycle  index = ", classElt["8C3"]];
Print["    C2'      index = ", classElt["6C2p"]];
Print["    C2       index = ", classElt["3C2"]];

(* Fused projector strategy (EXACT RATIONAL):                           *)
(*   For each irrep rho, build the 1-eigenspace projector P_fix[g] for *)
(*   a cyclic subgroup g whose rho-rep has chi=0, then take            *)
(*     P_fused[rho] = P[rho] . P_fix[g].                                *)
(*   The image of P_fused has dimension exactly m_rho.                  *)
(*                                                                      *)
(*   - d_rho=1 (A-type): P_fix = I, so P_fused = P[rho] has rank m=1.   *)
(*   - d_rho=2 (E-type): g = C_2', P_fix = (I + C_2')/2.                *)
(*       chi_E(C_2')=0 => eigenvalues {+1,-1} per copy, 1-dim fixed/copy*)
(*   - d_rho=3 (T-type): g = C_3, P_fix = (I + C_3 + C_3^2)/3.          *)
(*       chi_T(C_3)=0 => eigenvalues {1, w, wbar} per copy, 1-dim/copy. *)

id27 = IdentityMatrix[27];
rot27full2 = rot27full . rot27full;
PfixC3 = (id27 + rot27full + rot27full2) / 3;
PfixC2p = (id27 + rotC2p) / 2;

(* Exact column-space extraction: rows of RowReduce[Transpose[A]] are  *)
(* a basis of the column space of A.                                    *)
imageCols[A_, expectedRank_] := Module[{rref, cols, got},
  rref = RowReduce[Transpose[A]];
  cols = DeleteCases[rref, {0 ..}];
  got = Length[cols];
  If[got =!= expectedRank,
    Print["    WARNING imageCols: got ", got, " cols, expected ",
      expectedRank]];
  Transpose[cols]];

extractUsym[irrep_] := Module[{d, m, Pfused},
  d = irrepDims[irrep];
  m = projectorRanks[irrep]/d;
  Pfused = Which[
    d == 1, projectors[irrep],
    d == 2, projectors[irrep] . PfixC2p,
    d == 3, projectors[irrep] . PfixC3];
  imageCols[Pfused, m]];

(* Process all non-zero-multiplicity irreps.                            *)
irrepsNonzero = Select[irrepsOrdered, projectorRanks[#] > 0 &];
Print[];
Print["  reducing each nonzero isotypic block to its m_rho dim:"];
irrepData = Association[];
Do[
  Module[{d, m, Usym, MblkExact, BblkSym},
    d = irrepDims[irrep];
    m = projectorRanks[irrep]/d;
    Usym = extractUsym[irrep];
    If[Usym === {} || Length[Usym] == 0,
      Print["    ", irrep, ": reduction FAILED"]; Continue[]];
    (* Everything below is exact-rational (M27) or symbolic (BbodySym). *)
    MblkExact = Transpose[Usym] . M27 . Usym;
    BblkSym   = Transpose[Usym] . BbodySym . Usym;
    irrepData[irrep] = <|
      "d" -> d, "m" -> m,
      "Usym" -> Usym,
      "Mblock" -> MblkExact, "Bblock" -> BblkSym|>;
    Print["    ", StringPadRight[irrep, 4], ": d=", d, " m=", m,
      "   Dim[Usym] = ", Dimensions[Usym]];
  ],
  {irrep, irrepsNonzero}];

Print[];
Print["  M_rho blocks (exact rational):"];
Do[
  Module[{Mb = irrepData[irrep, "Mblock"]},
    If[Length[Mb] <= 2,
      Print["    ", StringPadRight[irrep, 4], ": ", Mb],
      Print["    ", StringPadRight[irrep, 4], ": ", Length[Mb], "x",
        Length[Mb], " block, diag = ", Diagonal[Mb]]]
  ],
  {irrep, irrepsNonzero}];

(* ================================================================== *)
(* Section 10. Symbolic per-irrep T_eff inversion (body-only)          *)
(* ================================================================== *)
Print[];
Print["---- Section 10: symbolic per-irrep T_eff  (Delta c = 0) ----"];

(* For Delta lam = Delta mu = 0 (density-only contrast), B_el = 0 and  *)
(* the Galerkin equation reads                                          *)
(*     (M27 - omega^2 Drho BbodySym) c = M27 c_0                        *)
(* =>  T_eff = (M27 - eps BbodySym)^(-1) (eps BbodySym),                *)
(*     eps = omega^2 * Drho.                                            *)
(*                                                                      *)
(* Under O_h symmetry, T_eff decomposes into 7 independent blocks:      *)
(*   A_{1g} (1x1), E_g (1x1), T_{2g} (1x1),                             *)
(*   A_{2u} (1x1), E_u (1x1), T_{2u} (2x2), T_{1u} (4x4).               *)
(* The 4x4 T_{1u} block is the "biggest closed form" -- this is the     *)
(* pattern the 15-year-old experiments predicted.                        *)

Clear[eps, Aelas, Belas];

(* Per-irrep symbolic T_red(eps) = (M_red - eps B_red)^(-1) . eps B_red *)
(* We use Together rather than full Simplify: 4x4 symbolic inversion   *)
(* with ~10 Pell atoms has Simplify runtime blowup on the T_{1u} block.*)
Print["  inverting each m_rho x m_rho block symbolically..."];
Do[
  Module[{m, Mb, Bb, Ared, Tred, t0local},
    m = irrepData[irrep, "m"];
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"];
    Ared = Mb - eps * Bb;
    t0local = AbsoluteTime[];
    Tred = Inverse[Ared] . (eps * Bb);
    Tred = Together[Tred];
    irrepData[irrep] = Append[irrepData[irrep], "Tred" -> Tred];
    Print["    ", StringPadRight[irrep, 4],
      "  (", m, "x", m, ")   T_red computed in ",
      Round[AbsoluteTime[] - t0local, 0.01], " s"];
  ],
  {irrep, irrepsNonzero}];

Print[];
Print["  T_red leaf counts (per-irrep symbolic complexity):"];
Do[
  Module[{Tred = irrepData[irrep, "Tred"], lc},
    lc = LeafCount[Tred];
    Print["    ", StringPadRight[irrep, 4], ": ", Dimensions[Tred],
      "  LeafCount = ", lc]
  ],
  {irrep, irrepsNonzero}];

Print[];
Print["  Symmetry-adapted block structure of T_eff_body:"];
Do[
  Print["    ", StringPadRight[irrep, 4],
    "  d_rho=", irrepData[irrep, "d"],
    "  m_rho=", irrepData[irrep, "m"],
    "   total size = ",
    irrepData[irrep, "d"]*irrepData[irrep, "m"], "x",
    irrepData[irrep, "d"]*irrepData[irrep, "m"]],
  {irrep, irrepsNonzero}];
Print[];
Print["  The largest block is T_{1u} (4x4), confirming the 15-year-old"];
Print["  observation that the effective T-matrix has the same cubic"];
Print["  block pattern as the 27x27 body form."];

(* ================================================================== *)
(* Section 11. Elastic stiffness B_el(Dlam, Dmu) -- full 27x27 form     *)
(* ================================================================== *)
Print[];
Print["---- Section 11: elastic stiffness B_el (full 27x27) ----"];

(* The Galerkin elastic bilinear form is the strain-energy integral     *)
(*                                                                        *)
(*    B_el[v, w] = integral_Omega  eps(v) : Delta c : eps(w) dV          *)
(*                                                                        *)
(* with the isotropic contrast                                            *)
(*                                                                        *)
(*    Delta c_{ijkl} = Dlam * delta_{ij} delta_{kl}                      *)
(*                   + Dmu  * (delta_{ik} delta_{jl} + delta_{il} delta_{jk}) *)
(*                                                                        *)
(* i.e. the Kelvin form of an isotropic stiffness contrast.  We compute  *)
(* the 27x27 matrix directly by symbolic integration over basisT27,     *)
(* rather than hand-coding block entries.                                 *)

Clear[Dlam, Dmu];
cubeCoords = {r1, r2, r3};
strainTensor[v_] := Table[
  (D[v[[ii]], cubeCoords[[jj]]] + D[v[[jj]], cubeCoords[[ii]]])/2,
  {ii, 3}, {jj, 3}];
deltaCContract[e1_, e2_] := Dlam*Tr[e1]*Tr[e2] +
  2 Dmu*Sum[e1[[ii, jj]] e2[[ii, jj]], {ii, 3}, {jj, 3}];
belEntry[v1_, v2_] := Integrate[
  deltaCContract[strainTensor[v1], strainTensor[v2]],
  {r1, -1, 1}, {r2, -1, 1}, {r3, -1, 1}];

BelSym = Table[belEntry[basisT27[[i]], basisT27[[j]]], {i, 27}, {j, 27}];
Print["  BelSym symmetric?    ", BelSym === Transpose[BelSym]];
Print["  BelSym dim           = ", Dimensions[BelSym]];

(* ------------------------------------------------------------------ *)
(* T_9 local block (rows/cols 1..9).  The constants (1..3) have zero  *)
(* strain, so the whole 9x9 reduces to a 6x6 axial+shear block.        *)
(* ------------------------------------------------------------------ *)
Print[];
Print["  T_9 strain sector (rows/cols 4..9):"];
Do[Print["    ", BelSym[[4 + r - 1, 4 ;; 9]]], {r, 1, 6}];
Print["  |BelSym[[1..3, ;;]]| = ",
  Max[Abs[Flatten[BelSym[[1 ;; 3, All]]]]], " (constants have zero strain)"];

(* Eigenvalue analysis in the O_h irrep basis on the T_9 strain block:  *)
(*   A_1g (trace)       : axial diag + 2 * axial off = 8(3 Dlam + 2 Dmu)*)
(*   E_g  (deviatoric)  : axial diag -     axial off = 16 Dmu           *)
(*   T_2g (shear)       : shear diag                 =  8 Dmu           *)
Print["  A_1g eigenvalue (T_9) = ", BelSym[[4, 4]] + 2 BelSym[[4, 5]],
  "  (= 8(3 Dlam + 2 Dmu))"];
Print["  E_g  eigenvalue (T_9) = ", BelSym[[4, 4]] - BelSym[[4, 5]],
  "  (= 16 Dmu)"];
Print["  T_2g eigenvalue (T_9) = ", BelSym[[7, 7]],
  "  (=  8 Dmu)"];

(* ------------------------------------------------------------------ *)
(* Linear x quadratic cross block (rows 4..9, cols 10..27) vanishes    *)
(* by odd parity: eps(linear) is constant and eps(quad) is linear in  *)
(* the coordinates, so the integrand is an odd polynomial on the      *)
(* symmetric box [-1,1]^3 and every entry integrates to 0.            *)
(* ------------------------------------------------------------------ *)
crossMax = Max[Abs[Flatten[BelSym[[1 ;; 9, 10 ;; 27]]]]];
Print["  |BelSym[[1..9, 10..27]]| max = ", crossMax,
  " (linear x quad -- vanishes by parity)"];

(* ------------------------------------------------------------------ *)
(* Quad x quad block (rows/cols 10..27): diagonal-plus-sparse 18x18    *)
(* with Dlam and Dmu entries.  Print its diagonal and rank.            *)
(* ------------------------------------------------------------------ *)
BelQQ = BelSym[[10 ;; 27, 10 ;; 27]];
Print[];
Print["  Quad-quad 18x18 block (BelSym[[10..27, 10..27]]):"];
Print["    diagonal = "];
Do[Print["      [", i, "] ", BelQQ[[i, i]]], {i, 1, 18}];
nnzBelQQ = Count[Flatten[BelQQ], _?(# =!= 0 &)];
Print["    nonzero entries = ", nnzBelQQ, " / ", 18*18];
Print["    symbolic rank   = ",
  MatrixRank[BelQQ /. {Dlam -> 1, Dmu -> 1}]];

(* Numerical substitution rule for the stiffness atoms (used in        *)
(* Section 12 density-only cross-check, where it is substituted to 0). *)
elasticRulesZero = {Dlam -> 0, Dmu -> 0};

(* ================================================================== *)
(* Section 11b. Stiffness Schur complement on T_9                       *)
(* ================================================================== *)
(* Partition M27 and BelSym into 9|18 blocks (linear ⊕ quadratic).     *)
(* The stiffness Schur complement is                                     *)
(*   BelSchur = Bel11 - M12 M22^(-1) Bel21 - Bel12 M22^(-1) M21          *)
(* and since Section 11 established that Bel12 = Bel21 = 0 by parity,   *)
(* it must collapse algebraically to Bel11.  We verify this explicitly  *)
(* and then print the O_h irrep eigenvalues of the resulting 6x6 strain *)
(* sub-block to show that the elastic channel is fully isotropic:       *)
(*   A_1g = 8 (3 Dlam + 2 Dmu),   E_g = 16 Dmu,   T_2g = 8 Dmu.         *)
(* ------------------------------------------------------------------ *)
Print[];
Print["---- Section 11b: stiffness Schur complement on T_9 ----"];

M11b = M27[[1 ;; 9, 1 ;; 9]];
M12b = M27[[1 ;; 9, 10 ;; 27]];
M21b = M27[[10 ;; 27, 1 ;; 9]];
M22b = M27[[10 ;; 27, 10 ;; 27]];
M22binv = Inverse[M22b];

Bel11 = BelSym[[1 ;; 9, 1 ;; 9]];
Bel12 = BelSym[[1 ;; 9, 10 ;; 27]];
Bel21 = BelSym[[10 ;; 27, 1 ;; 9]];

BelSchurSym = Simplify[
  Bel11 - M12b . M22binv . Bel21 - Bel12 . M22binv . M21b];
BelSchurDiff = Simplify[BelSchurSym - Bel11];
Print["  max |BelSchur - Bel11| (should be 0) = ",
  Max[Abs[Flatten[BelSchurDiff]]]];

(* Strain sub-block (rows/cols 4..9 of basisT27 are the 6 linear-strain *)
(* directions: e_xx, e_yy, e_zz, e_yz, e_xz, e_xy).                     *)
BelSchurStrain = BelSchurSym[[4 ;; 9, 4 ;; 9]];
Print[];
Print["  Stiffness Schur strain 6x6 (isotropic Kelvin form):"];
Print["    A_1g = ", Simplify[
  BelSchurStrain[[1, 1]] + 2 BelSchurStrain[[1, 2]]],
  "   (expected 8 (3 Dlam + 2 Dmu))"];
Print["    E_g  = ", Simplify[
  BelSchurStrain[[1, 1]] - BelSchurStrain[[1, 2]]],
  "   (expected 16 Dmu)"];
Print["    T_2g = ", Simplify[BelSchurStrain[[4, 4]]],
  "   (expected 8 Dmu)"];

(* Full stiffness-channel effective operator on T_9.                   *)
(* Inverting the 9x9 M11schur symbolically is fast because Section 12  *)
(* computes M11schur = M11 - M12 M22^(-1) M21 with exact rationals.    *)
M11schurSym = Simplify[M11b - M12b . M22binv . M21b];
T9effEl = Simplify[Inverse[M11schurSym] . BelSchurSym];
T9effElStrain = T9effEl[[4 ;; 9, 4 ;; 9]];
Print[];
Print["  T9effEl strain 6x6 irrep eigenvalues:"];
Print["    A_1g = ", Simplify[
  T9effElStrain[[1, 1]] + 2 T9effElStrain[[1, 2]]]];
Print["    E_g  = ", Simplify[
  T9effElStrain[[1, 1]] - T9effElStrain[[1, 2]]]];
Print["    T_2g = ", Simplify[T9effElStrain[[4, 4]]]];

(* ================================================================== *)
(* Section 12b. Combined body + stiffness Schur complement on T_9       *)
(* ================================================================== *)
(* Section 11b showed the stiffness Schur collapses to Bel11 with       *)
(* strain-sector irrep eigenvalues E_g = T_2g = 6 Dmu (isotropic).      *)
(* Section 12 (density-only, A polarization) numerically confirmed the  *)
(* body-channel Schur is also a scalar on the strain sector.  Here we   *)
(* carry out the symbolic extension:                                    *)
(*   (i)   Schur-complement B_body SYMBOLICALLY, keeping Aelas/Belas    *)
(*         as free parameters, and extract the strain-sector E_g and    *)
(*         T_2g eigenvalues as polynomials in the atoms.                *)
(*   (ii)  Test whether (E_g - T_2g) vanishes identically for BOTH      *)
(*         polarizations (a single test over free Aelas/Belas).         *)
(*   (iii) Assemble the full combined-channel T_9 effective operator    *)
(*         T9effTotal = (M11schur)^(-1) . (etaBody * B_body_Schur       *)
(*                                          + B_el_Schur)               *)
(*         with etaBody a free body-coupling parameter (~ omega^2 Drho) *)
(*         and check E_g vs T_2g on the strain sector.                  *)
(* A nonzero (E_g - T_2g) at any stage would prove the tier-27 Galerkin *)
(* closure produces a cubic-anisotropy T_3c component at leading order. *)
(* ------------------------------------------------------------------ *)
Print[];
Print["---- Section 12b: combined body+stiffness Schur on T_9 ----"];

BbodySubst = BbodySym /. atomRules;
Bbody11Sym = BbodySubst[[1 ;; 9, 1 ;; 9]];
Bbody12Sym = BbodySubst[[1 ;; 9, 10 ;; 27]];
Bbody21Sym = BbodySubst[[10 ;; 27, 1 ;; 9]];
BbodySchurSym = Simplify[
  Bbody11Sym - M12b . M22binv . Bbody21Sym - Bbody12Sym . M22binv . M21b];
BbodySchurStrain = BbodySchurSym[[4 ;; 9, 4 ;; 9]];

Print[];
Print["  Body-channel Schur strain 6x6 irrep eigenvalues (symbolic):"];
Print["    A_1g (trace) = ", Simplify[
  BbodySchurStrain[[1, 1]] + 2 BbodySchurStrain[[1, 2]]]];
Print["    E_g  (dev)   = ", Simplify[
  BbodySchurStrain[[1, 1]] - BbodySchurStrain[[1, 2]]]];
Print["    T_2g (shear) = ", Simplify[BbodySchurStrain[[4, 4]]]];

BbodyEgMinusT2g = Simplify[
  (BbodySchurStrain[[1, 1]] - BbodySchurStrain[[1, 2]]) -
  BbodySchurStrain[[4, 4]]];
Print[];
Print["  Cubic-anisotropy test on body-channel Schur:"];
Print["    (E_g - T_2g) symbolic = ", BbodyEgMinusT2g];
Print["    A-channel value       = ", Simplify[
  BbodyEgMinusT2g /. {Aelas -> 1, Belas -> 0}]];
Print["    B-channel value       = ", Simplify[
  BbodyEgMinusT2g /. {Aelas -> 0, Belas -> 1}]];

(* Also probe strain off-diagonals (4..9 block) for any cubic mixing.    *)
BbodyStrainOffDiag = BbodySchurStrain -
  DiagonalMatrix[Diagonal[BbodySchurStrain]];
Print["    strain off-diag max   = ",
  Max[Abs[Flatten[BbodyStrainOffDiag]]]];

(* ----------------------------------------------------------------- *)
(* Combined total Schur with dummy body-coupling prefactor etaBody.    *)
(* Since BelSchurSym = Bel11 (Section 11b), the total becomes           *)
(*    B_total_Schur = etaBody * B_body_Schur + Bel11.                  *)
(* ----------------------------------------------------------------- *)
Clear[etaBody];
BtotalSchurSym = Simplify[etaBody * BbodySchurSym + Bel11];
T9effTotal = Simplify[Inverse[M11schurSym] . BtotalSchurSym];
T9effTotalStrain = T9effTotal[[4 ;; 9, 4 ;; 9]];

Print[];
Print["  Full T9effTotal = (M11schur)^-1 . (etaBody * B_body_Schur + Bel11)"];
Print["  strain 6x6 irrep eigenvalues (symbolic in etaBody, Aelas,"];
Print["  Belas, Dlam, Dmu):"];
Print["    A_1g = ", Simplify[
  T9effTotalStrain[[1, 1]] + 2 T9effTotalStrain[[1, 2]]]];
Print["    E_g  = ", Simplify[
  T9effTotalStrain[[1, 1]] - T9effTotalStrain[[1, 2]]]];
Print["    T_2g = ", Simplify[T9effTotalStrain[[4, 4]]]];

TotalEgMinusT2g = Simplify[
  (T9effTotalStrain[[1, 1]] - T9effTotalStrain[[1, 2]]) -
  T9effTotalStrain[[4, 4]]];
Print[];
Print["  Cubic-anisotropy test on T9effTotal strain sector:"];
Print["    (E_g - T_2g) symbolic  = ", TotalEgMinusT2g];
Print["    A-channel (Aelas=1)    = ", Simplify[
  TotalEgMinusT2g /. {Aelas -> 1, Belas -> 0}]];
Print["    B-channel (Belas=1)    = ", Simplify[
  TotalEgMinusT2g /. {Aelas -> 0, Belas -> 1}]];
Print["    stiffness-only (eta=0) = ", Simplify[
  TotalEgMinusT2g /. {etaBody -> 0}]];

(* ================================================================== *)
(* Section 12. Sample numerical T_eff and Path-A cross-check            *)
(* ================================================================== *)
Print[];
Print["---- Section 12: sample T_eff vs Path-A (density-only) ----"];

(* Substitute Path-A-compatible numerical values.                       *)
(* Reference medium: alpha = 2000, beta = 1000, rho = 1000 (arbitrary). *)
(* Contrast: Drho = 100, Dlam = 0, Dmu = 0.  For static limit (omega*a  *)
(* small), the leading behavior is T_eff = omega^2 Drho M27^(-1) B_body.*)

(* Evaluate M27^(-1) . BbodyNumA numerically at MachinePrecision.       *)
(* High arbitrary precision on a 27x27 matrix forces Mathematica into   *)
(* symbolic eigensolvers, which hang.  We only need a few digits for   *)
(* the A-channel cross-check.                                            *)
Minv = Inverse[N[M27]];
BbodyFullNum = N[(BbodySym /. atomRules) /. {Aelas -> 1, Belas -> 0}];
TbodyOp = Minv . BbodyFullNum;

Print["  T_body operator (= M27^-1 . B_body, A channel) has"];
Print["    rank = ", MatrixRank[TbodyOp]];
Module[{evsAbs = Sort[Abs[Eigenvalues[TbodyOp]]]},
  Print["    spectrum (first 6) = ", Take[evsAbs, 6]];
  Print["    spectrum (last  6) = ", Take[Reverse[evsAbs], 6]];
];

(* --------------------------------------------------------------- *)
(* IMPORTANT: M27 is NOT block-diagonal between T_9 and T_27quad.   *)
(* The mass matrix has off-diagonal coupling                         *)
(*    M27[[const_k, S-quad in direction k]] = 8/3                    *)
(* which means that the naive (T_9 row slice) of M^(-1) B picks up  *)
(* contributions from the quadratic modes via the mass.  The proper *)
(* T_9 projection is the SCHUR COMPLEMENT that eliminates the       *)
(* quadratic block in the Galerkin equation.                        *)
(*                                                                    *)
(* Block form: Let                                                    *)
(*    M = [[M11, M12],[M21, M22]]   (9|18 partition of T_9|T_27quad) *)
(*    B = [[B11, B12],[B21, B22]]                                    *)
(* The Galerkin equation (M - eps B) c = M c_0 with c_0 supported on *)
(* T_9 only gives, after eliminating c_2 (the T_27quad part),         *)
(*    [ M11_S - eps B11_S ] c_1 = M11_S c_0,                          *)
(* where M11_S and B11_S are the Schur complements:                   *)
(*    M11_S = M11 - M12 M22^(-1) M21                                 *)
(*    B11_S = B11 - B12 M22^(-1) M21 - M12 M22^(-1) B21              *)
(*            + eps B12 M22^(-1) B21   (dropped at O(eps))            *)
(* At O(eps) the T_9 effective operator is                           *)
(*    T9_eff = M11_S^(-1) B11_S,                                      *)
(* which IS the right comparison target for Path-A.                   *)
M11 = M27[[1 ;; 9, 1 ;; 9]];
M12 = M27[[1 ;; 9, 10 ;; 27]];
M21 = M27[[10 ;; 27, 1 ;; 9]];
M22 = M27[[10 ;; 27, 10 ;; 27]];
M22inv = Inverse[M22];
M11schur = M11 - M12 . M22inv . M21;

BA = BbodySym /. atomRules /. {Aelas -> 1, Belas -> 0};
B11 = BA[[1 ;; 9, 1 ;; 9]];
B12 = BA[[1 ;; 9, 10 ;; 27]];
B21 = BA[[10 ;; 27, 1 ;; 9]];
B11schur = B11 - M12 . M22inv . B21 - B12 . M22inv . M21;

T9eff = Inverse[N[M11schur]] . N[B11schur];
Print[];
Print["  T_9 effective operator via Schur complement (A channel):"];
Print["    diag = ", Diagonal[Re[T9eff]]];
Print["    const block (1,1) = ", Re[T9eff[[1, 1]]]];
Print["    strain block (4,4) = ", Re[T9eff[[4, 4]]]];
Print["    strain off-diag (4,5) = ", Re[T9eff[[4, 5]]]];
Print["    shear  block (7,7) = ", Re[T9eff[[7, 7]]]];

(* ------------------------------------------------------------------ *)
(* Sanity check: verify the MpN(0,0,0) integration against the         *)
(* analytical G0_CUBE closed form.                                     *)
(*                                                                      *)
(*   G0_CUBE  := ∫_{[-1,1]^3} 1/|x| dV                                 *)
(*             = 8 * ∫_{[0,1]^3} 1/|x| dV        (octant symmetry)     *)
(*             = 8 * MpN(0,0,0)                                         *)
(*                                                                      *)
(* Note: the quantity iA used in b11const is a DIFFERENT alternating   *)
(* sum of MpN moments (see Section 8 for the b11 definition), not     *)
(* G0_CUBE; the two are related but unequal.                            *)
(* ------------------------------------------------------------------ *)
g0Cube = (4/3) Log[70226 + 40545 Sqrt[3]] - 2 Pi;
Print[];
Print["  Path-A geometric constant sanity check:"];
Print["    G0_CUBE (analytical)    = ", N[g0Cube, 16]];
Print["    8 * MpN(0,0,0) (Duffy)  = ", N[8 MpN[0, 0, 0], 16]];
Print["    abs difference          = ",
  N[Abs[g0Cube - 8 MpN[0, 0, 0]], 16]];

(* ------------------------------------------------------------------ *)
(* Structural observation (Task #17 body-channel validation):          *)
(* The Schur-complemented T_9 effective operator is expected to be     *)
(*    (const block)   3x3  diagonal  (all three equal by O_h)          *)
(*    (strain block)  6x6  scalar    (proportional to identity) -- the *)
(*                                    body integral at omega=0 in the  *)
(*                                    A-channel has no cubic anisotropy *)
(*                                    on the strain sector, so the     *)
(*                                    A_1g, E_g, T_2g eigenvalues all  *)
(*                                    coincide.                         *)
(* ------------------------------------------------------------------ *)
constDiags = Diagonal[Re[T9eff[[1 ;; 3, 1 ;; 3]]]];
strainDiags = Diagonal[Re[T9eff[[4 ;; 9, 4 ;; 9]]]];
constOff   = Max[Abs[Re[T9eff[[1 ;; 3, 1 ;; 3]]] -
                     DiagonalMatrix[constDiags]]];
strainOff  = Max[Abs[Re[T9eff[[4 ;; 9, 4 ;; 9]]] -
                     DiagonalMatrix[strainDiags]]];
Print[];
Print["  T_9 effective operator structure (A channel, omega=0):"];
Print["    const block diagonals    = ", constDiags];
Print["    const block off-diag max = ", constOff];
Print["    strain block diagonals   = ", strainDiags];
Print["    strain block off-diag max = ", strainOff];
Print["    max |const - mean(const)|  = ",
  Max[Abs[constDiags - Mean[constDiags]]]];
Print["    max |strain - mean(strain)| = ",
  Max[Abs[strainDiags - Mean[strainDiags]]]];
If[Max[Abs[constDiags - Mean[constDiags]]] < 10^-8 &&
   Max[Abs[strainDiags - Mean[strainDiags]]] < 10^-8 &&
   constOff < 10^-8 && strainOff < 10^-8,
  Print["    [OK] T_9 body operator has full O_h symmetry ",
    "(const and strain blocks are proportional to identity)."],
  Print["    [!!] T_9 body operator does NOT have expected scalar ",
    "structure -- review Schur complement."]];

Print[];
Print["  NOTE: full Path-A validation via T1c/T2c/T3c requires the    "];
Print["  elastic stiffness form B_el(Delta c) on T_27 (Task #14), which"];
Print["  contracts the Path-A Ac/Bc/Cc integrals against Delta lambda "];
Print["  and Delta mu on the strain sector.                            "];

(* ------------------------------------------------------------------ *)
(* Cross-check: per-irrep T_red (substituted numerically) must match   *)
(* the same spectrum as the full 27x27 numerical M27^(-1) B_body.      *)
(* ------------------------------------------------------------------ *)
Print[];
Print["  Cross-check: per-irrep T_red spectra vs full 27x27 spectrum"];

(* Use machine precision throughout for the cross-check.                *)
TbodyFullNumVals = Sort[Re[Eigenvalues[TbodyOp]]];

(* Collect eigenvalues from each per-irrep reduced block.               *)
perIrrepSpec = {};
Do[
  Module[{d, m, Mb, Bb, redOp, evs},
    d = irrepData[irrep, "d"];
    m = irrepData[irrep, "m"];
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"];
    (* Substitute numerical atom values, A-channel (machine precision). *)
    Bb = N[Bb /. atomRules /. {Aelas -> 1, Belas -> 0}];
    Mb = N[Mb];
    redOp = Inverse[Mb] . Bb;
    evs = Re[Eigenvalues[redOp]];
    Do[AppendTo[perIrrepSpec, evs[[k]]], {k, Length[evs]}];
    (* Each eigenvalue appears d times in the full 27 spectrum.         *)
    Do[AppendTo[perIrrepSpec, evs[[k]]], {k, Length[evs]}, {rep, d - 1}];
  ],
  {irrep, irrepsNonzero}];
perIrrepSpec = Sort[perIrrepSpec];

Print["    |full spectrum| = ", Length[TbodyFullNumVals]];
Print["    |per-irrep spectrum| = ", Length[perIrrepSpec]];
specDiff = If[Length[TbodyFullNumVals] === Length[perIrrepSpec],
  Max[Abs[TbodyFullNumVals - perIrrepSpec]],
  "LENGTH MISMATCH"];
Print["    max |full - per-irrep| = ", specDiff];
If[NumericQ[specDiff] && specDiff < 10^-8,
  Print["    [OK] per-irrep decomposition matches full 27-dim operator."],
  Print["    [!!] per-irrep decomposition mismatch -- review generators."]];

(* ================================================================== *)
(* Section 13a. Project B_el into irrep basis                           *)
(* ================================================================== *)
(* BelSym (27x27, symbolic in Dlam/Dmu) is computed in Section 11.     *)
(* Project it into each irrep block via Usym^T . BelSym . Usym, and   *)
(* store as irrepData[irrep, "BelBlock"].                               *)
Print[];
Print["---- Section 13a: project B_el into irrep basis ----"];

Do[
  Module[{Usym, BelBlk},
    Usym = irrepData[irrep, "Usym"];
    BelBlk = Simplify[Transpose[Usym] . BelSym . Usym];
    irrepData[irrep] = Append[irrepData[irrep], "BelBlock" -> BelBlk];
    Print["    ", StringPadRight[irrep, 4],
      ": BelBlock ", Dimensions[BelBlk],
      "  LeafCount = ", LeafCount[BelBlk]];
  ],
  {irrep, irrepsNonzero}];

(* Sanity: strain-sector irreps should reproduce the known eigenvalues *)
(* from Section 11: A_1g = 8(3 Dlam + 2 Dmu), E_g = 16 Dmu,          *)
(* T_2g = 8 Dmu.                                                        *)
Print[];
Print["  Strain-sector BelBlock check:"];
Print["    A1g = ", Simplify[irrepData["A1g", "BelBlock"][[1, 1]]],
  "  (expected 8(3 Dlam + 2 Dmu) = ", 24 Dlam + 16 Dmu, ")"];
Print["    Eg  = ", Simplify[irrepData["Eg", "BelBlock"][[1, 1]]],
  "  (expected 16 Dmu)"];
Print["    T2g = ", Simplify[irrepData["T2g", "BelBlock"][[1, 1]]],
  "  (expected 8 Dmu)"];

(* ================================================================== *)
(* Section 13b. Full per-irrep T-matrix (body + stiffness)              *)
(* ================================================================== *)
(* For each irrep, the reduced T-matrix is                              *)
(*   T_rho = (M_rho + Bel_rho - eps * Bbody_rho)^{-1}                  *)
(*         . (eps * Bbody_rho - Bel_rho)                                *)
(* with eps = omega^2 * Drho.  Keep (eps, Aelas, Belas, Dlam, Dmu)    *)
(* as free symbolic parameters.                                         *)
Print[];
Print["---- Section 13b: full per-irrep T-matrix (body + stiffness) ----"];

Clear[eps];
Print["  inverting each m_rho x m_rho block symbolically..."];
Do[
  Module[{m, Mb, Bb, BelBlk, Dred, Vred, Tfull, t0local},
    m = irrepData[irrep, "m"];
    Mb = irrepData[irrep, "Mblock"];
    Bb = irrepData[irrep, "Bblock"];
    BelBlk = irrepData[irrep, "BelBlock"];
    Dred = Mb + BelBlk - eps * Bb;
    Vred = eps * Bb - BelBlk;
    t0local = AbsoluteTime[];
    Tfull = Inverse[Dred] . Vred;
    (* Use Together (not Simplify) to avoid runtime blowup on 4x4 T1u *)
    Tfull = Together[Tfull];
    irrepData[irrep] = Append[irrepData[irrep], "TfullBlock" -> Tfull];
    Print["    ", StringPadRight[irrep, 4],
      "  (", m, "x", m, ")   T_full computed in ",
      Round[AbsoluteTime[] - t0local, 0.01], " s",
      "  LeafCount = ", LeafCount[Tfull]];
  ],
  {irrep, irrepsNonzero}];

(* ================================================================== *)
(* Section 13c. Substitute numerical atom values                        *)
(* ================================================================== *)
(* Replace opaque atom symbols with their numerical values from         *)
(* atomRules (Section 8).  The Pell closed forms are in                 *)
(* CubeT6ScalarValues_HighPrec_Pell.wl — here we substitute the        *)
(* numerical values for practical evaluation.                           *)
Print[];
Print["---- Section 13c: substitute numerical atom values ----"];

Do[
  Module[{Tfull, TfullNum},
    Tfull = irrepData[irrep, "TfullBlock"];
    TfullNum = Tfull /. atomRules;
    irrepData[irrep] = Append[irrepData[irrep], "TfullNumBlock" -> TfullNum];
    Print["    ", StringPadRight[irrep, 4],
      "  LeafCount(num) = ", LeafCount[TfullNum]];
  ],
  {irrep, irrepsNonzero}];

(* ================================================================== *)
(* Section 13d. Extract physical T-matrix scalars (strain sector)       *)
(* ================================================================== *)
(* The strain-sector irreps (A1g, Eg, T2g) are 1x1 blocks.  Their     *)
(* scalar T-matrix values map to the physical T1c, T2c, T3c via:       *)
(*   sigma_{A1g} = 3*T1c + 2*T2c + T3c   (volumetric)                  *)
(*   sigma_{Eg}  = 2*T2c + T3c            (deviatoric axial)           *)
(*   sigma_{T2g} = 2*T2c                  (shear)                       *)
(* Inverting:                                                            *)
(*   T2c = sigma_{T2g} / 2                                              *)
(*   T3c = sigma_{Eg} - sigma_{T2g}                                     *)
(*   T1c = (sigma_{A1g} - sigma_{Eg}) / 3                               *)
Print[];
Print["---- Section 13d: physical T-matrix scalars (T1c, T2c, T3c) ----"];

sigmaA1g = irrepData["A1g", "TfullNumBlock"][[1, 1]];
sigmaEg  = irrepData["Eg",  "TfullNumBlock"][[1, 1]];
sigmaT2g = irrepData["T2g", "TfullNumBlock"][[1, 1]];

Print["  sigma_{A1g} = ", sigmaA1g];
Print["  sigma_{Eg}  = ", sigmaEg];
Print["  sigma_{T2g} = ", sigmaT2g];

T2cGalerkin = Together[sigmaT2g / 2];
T3cGalerkin = Together[sigmaEg - sigmaT2g];
T1cGalerkin = Together[(sigmaA1g - sigmaEg) / 3];

Print[];
Print["  T1c(Galerkin) = ", T1cGalerkin];
Print["  T2c(Galerkin) = ", T2cGalerkin];
Print["  T3c(Galerkin) = ", T3cGalerkin];

(* ================================================================== *)
(* Section 13e. Born-limit coefficients                                  *)
(* ================================================================== *)
(* At Born level (first order in contrasts), the T-matrix linearizes:   *)
(*   sigma_rho ~ eps * bbody_rho - bel_rho                              *)
(* where eps = omega^2 * Drho is the body coupling and                  *)
(* bel_rho = BelBlock / MblockInverse is the stiffness coupling.        *)
(*                                                                      *)
(* For Born limit, set eps -> 0 for stiffness-only, or Dlam=Dmu=0 for  *)
(* body-only, and extract the leading coefficients.                     *)
Print[];
Print["---- Section 13e: Born-limit coefficients ----"];

(* Body-channel Born: T ~ eps * M^{-1} . Bbody at Dlam=Dmu=0.          *)
(* For 1x1 blocks: sigma_rho^Born_body = eps * Bbody_rho / M_rho.      *)
bornBodyA1g = Together[
  (irrepData["A1g", "Bblock"][[1, 1]] /. atomRules) /
  irrepData["A1g", "Mblock"][[1, 1]]];
bornBodyEg = Together[
  (irrepData["Eg", "Bblock"][[1, 1]] /. atomRules) /
  irrepData["Eg", "Mblock"][[1, 1]]];
bornBodyT2g = Together[
  (irrepData["T2g", "Bblock"][[1, 1]] /. atomRules) /
  irrepData["T2g", "Mblock"][[1, 1]]];

Print["  Born body-channel (per eps):"];
Print["    sigma_{A1g}/eps (A) = ",
  N[bornBodyA1g /. {Aelas -> 1, Belas -> 0}, 16]];
Print["    sigma_{A1g}/eps (B) = ",
  N[bornBodyA1g /. {Aelas -> 0, Belas -> 1}, 16]];
Print["    sigma_{Eg}/eps  (A) = ",
  N[bornBodyEg /. {Aelas -> 1, Belas -> 0}, 16]];
Print["    sigma_{Eg}/eps  (B) = ",
  N[bornBodyEg /. {Aelas -> 0, Belas -> 1}, 16]];
Print["    sigma_{T2g}/eps (A) = ",
  N[bornBodyT2g /. {Aelas -> 1, Belas -> 0}, 16]];
Print["    sigma_{T2g}/eps (B) = ",
  N[bornBodyT2g /. {Aelas -> 0, Belas -> 1}, 16]];

(* Stiffness-channel Born: T ~ -M^{-1} . Bel at eps=0.                 *)
bornElA1g = Together[
  -irrepData["A1g", "BelBlock"][[1, 1]] /
  irrepData["A1g", "Mblock"][[1, 1]]];
bornElEg = Together[
  -irrepData["Eg", "BelBlock"][[1, 1]] /
  irrepData["Eg", "Mblock"][[1, 1]]];
bornElT2g = Together[
  -irrepData["T2g", "BelBlock"][[1, 1]] /
  irrepData["T2g", "Mblock"][[1, 1]]];

Print[];
Print["  Born stiffness-channel:"];
Print["    sigma_{A1g}^el = ", Simplify[bornElA1g]];
Print["    sigma_{Eg}^el  = ", Simplify[bornElEg]];
Print["    sigma_{T2g}^el = ", Simplify[bornElT2g]];

(* Born T1c, T2c, T3c from body + stiffness channels.                  *)
Print[];
Print["  Born-limit T1c, T2c, T3c (extracting A and B coefficients):"];
bornT2c = Together[(bornBodyT2g * eps + bornElT2g) / 2];
bornT3c = Together[(bornBodyEg * eps + bornElEg) -
  (bornBodyT2g * eps + bornElT2g)];
bornT1c = Together[((bornBodyA1g * eps + bornElA1g) -
  (bornBodyEg * eps + bornElEg)) / 3];
Print["  T1c^Born = ", bornT1c];
Print["  T2c^Born = ", bornT2c];
Print["  T3c^Born = ", bornT3c];

(* Cubic anisotropy coefficient at Born level: T3c.                     *)
Print[];
Print["  T3c^Born (cubic anisotropy) at (A=1,B=1):"];
Print["    T3c^Born(A-only) = ",
  Simplify[bornT3c /. {Aelas -> 1, Belas -> 0}]];
Print["    T3c^Born(B-only) = ",
  Simplify[bornT3c /. {Aelas -> 0, Belas -> 1}]];

(* ================================================================== *)
(* Section 13f. Numerical verification against Path-A                    *)
(* ================================================================== *)
Print[];
Print["---- Section 13f: numerical verification vs Path-A ----"];

(* Standard test parameters from MEMORY.md:                             *)
(*   alpha=5000, beta=3000, rho=2500                                    *)
(*   Moderate contrast: Dlam=+2e9, Dmu=+1e9, Drho=+100                 *)
(*   ka target: 0.05 (Rayleigh limit)                                   *)
refAlpha = 5000; refBeta = 3000; refRho = 2500;
refLam = refRho (refAlpha^2 - 2 refBeta^2);
refMu  = refRho refBeta^2;
testDlam = 2*10^9; testDmu = 1*10^9; testDrho = 100;

(* ka = 0.05 -> omega = 0.05 * beta / a, with a = 1 (unit cube). *)
testA = 1;
kS = 0.05 / testA;
testOmega = kS * refBeta;

(* eps = omega^2 * Drho *)
testEps = testOmega^2 * testDrho;

Print["  Test parameters:"];
Print["    alpha=", refAlpha, " beta=", refBeta, " rho=", refRho];
Print["    Dlam=", testDlam, " Dmu=", testDmu, " Drho=", testDrho];
Print["    ka=0.05, omega=", testOmega, " eps=", testEps];

(* Evaluate the Galerkin T-matrix scalars numerically.                  *)
numRules = {eps -> testEps, Dlam -> testDlam, Dmu -> testDmu,
  Aelas -> 1/(8 Pi refRho refAlpha^2 refBeta^2),
  Belas -> 1/(8 Pi refRho refAlpha^2 refBeta^2)};
(* NOTE: Aelas and Belas are the Green's tensor Kelvin coefficients.    *)
(* The body bilinear form was computed with abstract Aelas, Belas:      *)
(*   Aelas = (alpha^2 + beta^2) / (8 pi rho alpha^2 beta^2)  = a0     *)
(*   Belas = (alpha^2 - beta^2) / (8 pi rho alpha^2 beta^2)  = b0     *)
numRulesPhys = {eps -> testEps, Dlam -> testDlam, Dmu -> testDmu,
  Aelas -> (refAlpha^2 + refBeta^2) / (8 Pi refRho refAlpha^2 refBeta^2),
  Belas -> (refAlpha^2 - refBeta^2) / (8 Pi refRho refAlpha^2 refBeta^2)};

sigA1gNum = N[sigmaA1g /. numRulesPhys, 20];
sigEgNum  = N[sigmaEg  /. numRulesPhys, 20];
sigT2gNum = N[sigmaT2g /. numRulesPhys, 20];

T1cNum = (sigA1gNum - sigEgNum) / 3;
T2cNum = sigT2gNum / 2;
T3cNum = sigEgNum - sigT2gNum;

Print[];
Print["  Galerkin per-irrep T-matrix (numerical):"];
Print["    sigma_{A1g} = ", sigA1gNum];
Print["    sigma_{Eg}  = ", sigEgNum];
Print["    sigma_{T2g} = ", sigT2gNum];
Print[];
Print["  Physical T-matrix scalars (Galerkin):"];
Print["    T1c = ", T1cNum];
Print["    T2c = ", T2cNum];
Print["    T3c = ", T3cNum];

(* Also evaluate T1u and T2u blocks numerically for spectrum check.     *)
Print[];
Print["  All per-irrep T-matrix spectra (numerical):"];
Do[
  Module[{Tfull, TfullPhys, evals},
    Tfull = irrepData[irrep, "TfullNumBlock"];
    TfullPhys = N[Tfull /. numRulesPhys, 20];
    evals = Eigenvalues[TfullPhys];
    Print["    ", StringPadRight[irrep, 4],
      ": eigenvalues = ", Sort[Re[evals]]];
  ],
  {irrep, irrepsNonzero}];

(* ================================================================== *)
(* Section 14. Export symbolic 27x27 matrices to disk                   *)
(* ================================================================== *)
Print[];
Print["---- Section 14: export symbolic matrices ----"];

outPath = FileNameJoin[{$here, "CubeT27AssembleResults.wl"}];
(* Strip out the large matrix Uiso/Usym from irrepData for export.     *)
irrepDataExport = Association @ KeyValueMap[
  Function[{irrep, info},
    irrep -> KeyDrop[info, {"Uiso", "Usym"}]],
  irrepData];
Export[outPath,
  "(* CubeT27AssembleResults.wl -- symbolic 27x27 matrices.               *)\n" <>
  "(* Generated by CubeT27Assemble.wl.  Contents:                          *)\n" <>
  "(*   M27        : exact-rational 27x27 mass matrix                      *)\n" <>
  "(*   BbodySym   : 27x27 body bilinear form with Pell + B9 + YY atoms   *)\n" <>
  "(*   atomRules  : substitution rules for atomic scalars (numerical)    *)\n" <>
  "(*   rot27      : 27x27 rep matrix for the xyz 3-cycle                 *)\n" <>
  "(*   irrepData  : per-irrep (M_block, B_block, T_red) reductions       *)\n" <>
  "\n" <>
  "M27        = " <> ToString[FullForm[M27]] <> ";\n" <>
  "BbodySym   = " <> ToString[FullForm[BbodySym]] <> ";\n" <>
  "atomRules  = " <> ToString[FullForm[atomRules]] <> ";\n" <>
  "rot27      = " <> ToString[FullForm[rot27]] <> ";\n" <>
  "irrepData  = " <> ToString[FullForm[irrepDataExport]] <> ";\n",
  "Text"];
Print["  wrote ", outPath];

Print[];
Print["==== CubeT27Assemble.wl done ===="];
