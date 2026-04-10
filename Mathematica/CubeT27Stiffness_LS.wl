(* CubeT27Stiffness_LS.wl -- LS-convolved stiffness for ungerade irreps.

   The stiffness contribution to the Galerkin T-matrix is NOT the direct
   strain energy B_el = integral eps:Dc:eps dV, but the LS-convolved form:

     Bstiff_LS[alpha, beta] = <phi_alpha | G * div(Dc:eps(phi_beta)) >

   For quadratic modes on [-1,1]^3, div(Dc:eps(phi_beta)) splits into:
   1. Constant VOLUME force f_vol (inside Omega)
   2. LINEAR surface traction t (on partial Omega)

   The volume part uses existing Bbody entries (no new NIntegrate).
   The surface part requires 5D NIntegrate with 1/|r-r'| singularity.

   Strategy:
   ---------
   1. Reconstruct Usym from O_h group theory (cheap, exact rational)
   2. Compute BstiffVol in 27x27 basis from BbodySym (algebraic)
   3. Project BstiffVol to irrep blocks via Usym
   4. Compute BstiffSurf directly in the irrep basis (fewer NIntegrate calls)
   5. Export 4-channel per-irrep stiffness matrices

   Usage:
     wolframscript -file Mathematica/CubeT27Stiffness_LS.wl
*)

Print["==== CubeT27Stiffness_LS.wl ===="];
Print[];
$here = DirectoryName[$InputFileName];

(* ================================================================== *)
(* Section 1. Load data + patch beta_7                                  *)
(* ================================================================== *)
Print["---- Section 1: load data ----"];
Get[FileNameJoin[{$here, "CubeT6QuadQuad.wl"}]];
beta7val = B27quadB[[2, 17]];
Print["  beta_7 = ", NumberForm[beta7val, 16]];

Get[FileNameJoin[{$here, "CubeT27AssembleResults.wl"}]];

(* Patch Missing[KeyAbsent, 0] -> beta_7 *)
BbodySym = BbodySym /. Missing["KeyAbsent", 0] -> beta7val;
Do[
  Module[{Bb},
    Bb = irrepData[irrep, "Bblock"] /. Missing["KeyAbsent", 0] -> beta7val;
    irrepData[irrep] = Append[irrepData[irrep], "Bblock" -> Bb]],
  {irrep, Keys[irrepData]}];
Print["  data loaded, beta_7 patched."];
Print[];

(* ================================================================== *)
(* Section 2. Reconstruct O_h group and 27x27 representation            *)
(* ================================================================== *)
Print["---- Section 2: reconstruct O_h group + 27D representation ----"];
t0 = AbsoluteTime[];

(* basisT27 definition (must match CubeT27Assemble.wl exactly) *)
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
Assert[Length[basisT27] === 27];

(* Monomial grid for decomposition *)
monomialList = {
  {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
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

monomialExp[m_] := If[m === 1, {0, 0, 0},
  First[CoefficientRules[Expand[m], {r1, r2, r3}]][[1]]];

nonzeroDir[vec_] := Module[{d = 0},
  Do[If[vec[[k]] =!= 0, d = k; Break[]], {k, 3}]; d];

monomialSlotRules = Association @ Table[
  Module[{vec, dir, scalar},
    vec = monomialList[[mm]];
    dir = nonzeroDir[vec];
    scalar = vec[[dir]];
    {dir, monomialExp[scalar]} -> mm],
  {mm, nMon}];

polyCoeffVec[v_] := Module[{out, scalarPart, cr, slot},
  out = Table[0, {nMon}];
  Do[
    scalarPart = v[[dir]];
    If[scalarPart =!= 0,
      cr = CoefficientRules[Expand[scalarPart], {r1, r2, r3}];
      Do[
        slot = Lookup[monomialSlotRules, Key[{dir, pair[[1]]}], Missing[]];
        If[! MissingQ[slot], out[[slot]] += pair[[2]]],
        {pair, cr}]],
    {dir, 3}];
  out];

Bemb = Table[polyCoeffVec[basisT27[[i]]], {i, 27}];
BembGram = Bemb . Transpose[Bemb];
BembGramInv = Inverse[BembGram];
decomposeInBasis[v_] := BembGramInv . (Bemb . polyCoeffVec[v]);

(* Build all 48 O_h elements as 3x3 signed permutation matrices *)
buildSPM[perm_List, signs_List] := Normal @ SparseArray[
  Table[{perm[[j]], j} -> signs[[j]], {j, 3}], {3, 3}];

ohGroup3D = Flatten[
  Table[buildSPM[perm, signs],
    {perm, Permutations[{1, 2, 3}]},
    {signs, Tuples[{1, -1}, 3]}], 1];
Assert[Length[ohGroup3D] === 48];

(* Group action on basisT27: (g.v)(r) = M . v(M^T . r) *)
applyG3D[M_, v_] := Module[{subs, vSub},
  subs = Thread[{r1, r2, r3} -> Transpose[M] . {r1, r2, r3}];
  vSub = v /. subs;
  Expand[M . vSub]];

(* Build 48 representation matrices (27x27) *)
R27list = Table[
  Transpose @ Table[
    decomposeInBasis[applyG3D[M, basisT27[[j]]]],
    {j, 27}],
  {M, ohGroup3D}];

Print["  48 rep matrices built in ", Round[AbsoluteTime[] - t0, 0.1], " s"];

(* Verify orthogonality *)
maxOrthErr = Max[Table[
  Max[Abs[Flatten[R27list[[k]] . Transpose[R27list[[k]]] - IdentityMatrix[27]]]],
  {k, 48}]];
Print["  max orthogonality error = ", ScientificForm[maxOrthErr, 3]];

(* Classify conjugacy classes *)
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
Print["  class counts = ", classCounts];

(* Character table *)
classOrder = {"E", "8C3", "3C2", "6C4", "6C2p",
              "i", "8S6", "3sh", "6S4", "6sd"};
classToIdx = Association @ Table[classOrder[[k]] -> k, {k, Length[classOrder]}];

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
irrepDims = <|"A1g"->1,"A2g"->1,"Eg"->2,"T1g"->3,"T2g"->3,
  "A1u"->1,"A2u"->1,"Eu"->2,"T1u"->3,"T2u"->3|>;
irrepsOrdered = {"A1g","A2g","Eg","T1g","T2g","A1u","A2u","Eu","T1u","T2u"};

(* Character projectors *)
projectors = Association[];
projectorRanks = Association[];
Do[
  Module[{d, proj},
    d = irrepDims[irrep];
    proj = (d/48) * Sum[
      charTable[irrep][[classToIdx[classLabels[[k]]]]] * R27list[[k]],
      {k, 48}];
    projectors[irrep] = proj;
    projectorRanks[irrep] = MatrixRank[proj]],
  {irrep, irrepsOrdered}];

Print["  projector ranks: ", Normal[projectorRanks]];
Assert[Total[Values[projectorRanks]] === 27];

(* ================================================================== *)
(* Section 3. Extract Usym via symmetry-breaking                        *)
(* ================================================================== *)
Print[];
Print["---- Section 3: extract Usym per irrep ----"];

(* Symmetry-breaking projectors *)
classElt[clsName_] := First[Flatten[Position[classLabels, clsName]]];
rot27full = R27list[[classElt["8C3"]]];
rotC2p    = R27list[[classElt["6C2p"]]];

id27 = IdentityMatrix[27];
rot27full2 = rot27full . rot27full;
PfixC3 = (id27 + rot27full + rot27full2) / 3;
PfixC2p = (id27 + rotC2p) / 2;

imageCols[A_, expectedRank_] := Module[{rref, cols, got},
  rref = RowReduce[Transpose[A]];
  cols = DeleteCases[rref, {0 ..}];
  got = Length[cols];
  If[got =!= expectedRank,
    Print["    WARNING imageCols: got ", got, " cols, expected ", expectedRank]];
  Transpose[cols]];

extractUsym[irrep_] := Module[{d, m, Pfused},
  d = irrepDims[irrep];
  m = projectorRanks[irrep]/d;
  If[m == 0, Return[{}]];
  Pfused = Which[
    d == 1, projectors[irrep],
    d == 2, projectors[irrep] . PfixC2p,
    d == 3, projectors[irrep] . PfixC3];
  imageCols[Pfused, m]];

(* Compute Usym for all nonzero irreps *)
irrepsNonzero = Select[irrepsOrdered, projectorRanks[#] > 0 &];
UsymData = Association[];
Do[
  Module[{Usym, m},
    m = projectorRanks[irrep]/irrepDims[irrep];
    Usym = extractUsym[irrep];
    UsymData[irrep] = Usym;
    Print["  ", StringPadRight[irrep, 4],
      ": m=", m, "  Usym dims=", Dimensions[Usym]]],
  {irrep, irrepsNonzero}];

(* Verify: Usym^T . M27 . Usym matches stored Mblock *)
Print[];
Print["  Verifying Usym against stored Mblock:"];
Do[
  Module[{Usym, Mrec, Mstored, errM},
    Usym = UsymData[irrep];
    Mrec = Transpose[Usym] . M27 . Usym;
    Mstored = irrepData[irrep, "Mblock"];
    errM = Max[Abs[Flatten[N[Mrec - Mstored, 20]]]];
    Print["    ", StringPadRight[irrep, 4],
      ": max|Mrec - Mstored| = ", ScientificForm[errM, 3]];
    Assert[errM < 10^-12]],
  {irrep, irrepsNonzero}];

(* Verify: Usym^T . BbodySym . Usym matches stored Bblock *)
Print[];
Print["  Verifying Usym against stored Bblock (at atomRules):"];
Do[
  Module[{Usym, Brec, Bstored, BrecNum, BstoredNum, errB},
    Usym = UsymData[irrep];
    Brec = Transpose[Usym] . BbodySym . Usym;
    Bstored = irrepData[irrep, "Bblock"];
    BrecNum = N[Brec /. atomRules /. {Aelas -> 1, Belas -> 1}, 20];
    BstoredNum = N[Bstored /. atomRules /. {Aelas -> 1, Belas -> 1}, 20];
    errB = Max[Abs[Flatten[BrecNum - BstoredNum]]];
    Print["    ", StringPadRight[irrep, 4],
      ": max|Brec - Bstored| = ", ScientificForm[errB, 3]];
    Assert[errB < 10^-8]],
  {irrep, irrepsNonzero}];

Print[];
Print["  Usym reconstruction VERIFIED."];

(* ================================================================== *)
(* Section 4. Volume force for each quadratic mode                      *)
(* ================================================================== *)
Print[];
Print["---- Section 4: volume forces f_vol for quadratic modes ----"];

(* For S-type r_a^2 e_p:
     a = p: f = 2(Dlam + 2*Dmu) e_p
     a != p: f = 2*Dmu e_p
   For X-type: f = 0 *)

fVol = Table[{0, 0, 0}, {27}];
Do[
  Module[{p, localIdx, a},
    p = Quotient[qi - 1, 6] + 1;
    localIdx = Mod[qi - 1, 6] + 1;
    If[localIdx <= 3,
      a = localIdx;
      If[a == p,
        fVol[[9 + qi]] = 2*(Dlam + 2*Dmu) * UnitVector[3, p],
        fVol[[9 + qi]] = 2*Dmu * UnitVector[3, p]]]],
  {qi, 1, 18}];

nNonzero = Count[fVol, v_ /; v =!= {0, 0, 0}];
Print["  Nonzero volume forces: ", nNonzero, " out of 27 modes"];

(* ================================================================== *)
(* Section 5. BstiffVol in 27x27 basis and project to irreps            *)
(* ================================================================== *)
Print[];
Print["---- Section 5: BstiffVol = f_j . Bbody[:, j] + project to irreps ----"];

(* BstiffVol[alpha, beta] = sum_j f_j^(beta) * BbodySym[alpha, j] *)
BstiffVol27 = Table[0, {27}, {27}];
Do[
  Module[{fj = fVol[[beta]]},
    If[fj =!= {0, 0, 0},
      Do[
        BstiffVol27[[alpha, beta]] =
          Sum[fj[[j]] * BbodySym[[alpha, j]], {j, 1, 3}],
        {alpha, 1, 27}]]],
  {beta, 1, 27}];

Print["  nonzero entries in BstiffVol27: ",
  Count[Flatten[BstiffVol27], x_ /; x =!= 0]];

(* Project BstiffVol to irrep blocks via Usym *)
ungeradeIrreps = {"T1u", "T2u", "A2u", "Eu"};

BstiffVolIrrep = Association[];
Do[
  Module[{Usym, Bvol},
    Usym = UsymData[irrep];
    Bvol = Transpose[Usym] . BstiffVol27 . Usym;
    (* Bvol is symbolic in (Aelas, Belas) x (Dlam, Dmu) *)
    BstiffVolIrrep[irrep] = Bvol;

    (* Print structure at test point *)
    Print["  ", irrep, " BstiffVol (A=1,B=0,Dl=1,Dm=0):"];
    Module[{BvolNum = N[Bvol /. atomRules /.
        {Aelas -> 1, Belas -> 0, Dlam -> 1, Dmu -> 0}, 16]},
      If[Length[BvolNum] <= 2,
        Print["    ", BvolNum],
        Do[Print["    row ", i, ": ", BvolNum[[i]]], {i, Length[BvolNum]}]]]],
  {irrep, ungeradeIrreps}];

(* ================================================================== *)
(* Section 6. Surface traction and integral setup                       *)
(* ================================================================== *)
Print[];
Print["---- Section 6: surface integral setup ----"];

(* Strain tensor for quadratic mode qi (1..18) at point r *)
strainTensor[qi_Integer, rv_] := Module[
  {p, localIdx, a, b, c, eps = ConstantArray[0, {3, 3}]},
  p = Quotient[qi - 1, 6] + 1;
  localIdx = Mod[qi - 1, 6] + 1;
  If[localIdx <= 3,
    a = localIdx;
    Do[eps[[k, l]] += rv[[a]] (KroneckerDelta[l, p] KroneckerDelta[k, a]
                            + KroneckerDelta[k, p] KroneckerDelta[l, a]),
      {k, 3}, {l, 3}],
    b = Switch[localIdx, 4, 2, 5, 1, 6, 1];
    c = Switch[localIdx, 4, 3, 5, 3, 6, 2];
    Do[eps[[k, l]] += (1/2)(
      KroneckerDelta[k, p] (KroneckerDelta[l, b] rv[[c]] + rv[[b]] KroneckerDelta[l, c])
      + KroneckerDelta[l, p] (KroneckerDelta[k, b] rv[[c]] + rv[[b]] KroneckerDelta[k, c])),
      {k, 3}, {l, 3}]];
  eps];

stressTensor[eps_, dlam_, dmu_] := Module[{tr = Tr[eps]},
  dlam * tr * IdentityMatrix[3] + 2 dmu * eps];

(* Traction on face x_{faceAxis} = faceSign *)
tractionOnFace[qi_Integer, faceAxis_Integer, faceSign_Integer, rv_] :=
  Module[{rFace, eps, sigma, n},
    rFace = rv;
    rFace[[faceAxis]] = faceSign;
    eps = strainTensor[qi, rFace];
    sigma = stressTensor[eps, Dlam, Dmu];
    n = faceSign * UnitVector[3, faceAxis];
    sigma . n];

(* Green's tensor: G_ij(x) = Aelas/|x| delta_ij + Belas x_i x_j/|x|^3 *)
greensTensor[x_] := Module[{rr = Sqrt[x . x]},
  Aelas/rr * IdentityMatrix[3] + Belas * Outer[Times, x, x] / rr^3];

(* Displacement function for mode index in basisT27 (1..27) *)
displacementMode[modeIdx_Integer, rv_] := Module[{qi, p, localIdx, a, b, c},
  Which[
    modeIdx <= 3, UnitVector[3, modeIdx],
    modeIdx <= 9, basisT27[[modeIdx]] /. Thread[{r1, r2, r3} -> rv],
    True,
      qi = modeIdx - 9;
      p = Quotient[qi - 1, 6] + 1;
      localIdx = Mod[qi - 1, 6] + 1;
      If[localIdx <= 3,
        a = localIdx;
        rv[[a]]^2 * UnitVector[3, p],
        b = Switch[localIdx, 4, 2, 5, 1, 6, 1];
        c = Switch[localIdx, 4, 3, 5, 3, 6, 2];
        rv[[b]] rv[[c]] * UnitVector[3, p]]]];

(* ================================================================== *)
(* Section 7. Surface stiffness in the IRREP basis                      *)
(* ================================================================== *)
Print[];
Print["---- Section 7: surface stiffness in irrep basis ----"];

(* For each irrep, the Usym columns define the irrep-basis modes.
   The a-th irrep-basis mode is:
     Psi_a(r) = sum_alpha Usym[alpha, a] * phi_alpha(r)

   The irrep-basis surface stiffness is:
     BstiffSurf_rho[a,b] = sum_faces integral Psi_a(r) . G(r-r') . T_b(r') dr dr'
   where T_b(r') = sum_beta Usym[beta, b] * t_beta(r') is the traction
   of the b-th irrep mode, and the sum is over all 6 cube faces.

   Only modes with modeIdx > 9 (quadratic) have nonzero traction.
*)

(* Evaluate the irrep-basis mode function at point rv *)
irrepMode[irrep_, col_Integer, rv_] := Module[{Usym, result},
  Usym = UsymData[irrep];
  result = {0, 0, 0};
  Do[
    If[Usym[[alpha, col]] =!= 0,
      result += Usym[[alpha, col]] * displacementMode[alpha, rv]],
    {alpha, 27}];
  Expand[result]];

(* Evaluate the irrep-basis traction at point rv on face (fAxis, fSign) *)
irrepTraction[irrep_, col_Integer, fAxis_Integer, fSign_Integer, rv_] :=
  Module[{Usym, result},
    Usym = UsymData[irrep];
    result = {0, 0, 0};
    Do[
      If[beta > 9 && Usym[[beta, col]] =!= 0,
        result += Usym[[beta, col]] *
          tractionOnFace[beta - 9, fAxis, fSign, rv]],
      {beta, 27}];
    Expand[result]];

(* Surface integral for one (a,b) pair in irrep basis, one face *)
bstiffSurfIrrep1Face[irrep_, a_Integer, b_Integer,
    fAxis_Integer, fSign_Integer, AelasV_, BelasV_, DlamV_, DmuV_] :=
  Module[{freeAxes, rv, rprime, xdiff, PsiA, Tb, Gmat, integrand, result},
    freeAxes = DeleteCases[{1, 2, 3}, fAxis];

    rv = {rv1, rv2, rv3};
    rprime = Table[0, 3];
    rprime[[fAxis]] = fSign;
    rprime[[freeAxes[[1]]]] = sv1;
    rprime[[freeAxes[[2]]]] = sv2;

    PsiA = irrepMode[irrep, a, rv];
    Tb = irrepTraction[irrep, b, fAxis, fSign, rprime];

    (* Substitute numerical channel values *)
    PsiA = PsiA /. {Aelas -> AelasV, Belas -> BelasV, Dlam -> DlamV, Dmu -> DmuV};
    Tb = Tb /. {Aelas -> AelasV, Belas -> BelasV, Dlam -> DlamV, Dmu -> DmuV};

    xdiff = rv - rprime;
    Gmat = greensTensor[xdiff] /. {Aelas -> AelasV, Belas -> BelasV};

    integrand = PsiA . Gmat . Tb;

    If[Simplify[integrand] === 0, Return[0]];

    result = NIntegrate[
      integrand,
      {rv1, -1, 1}, {rv2, -1, 1}, {rv3, -1, 1},
      {sv1, -1, 1}, {sv2, -1, 1},
      PrecisionGoal -> 10, AccuracyGoal -> 10,
      MaxRecursion -> 20,
      Method -> {"GlobalAdaptive",
                 "SingularityHandler" -> "DuffyCoordinates"}];
    result];

(* Full surface stiffness for one (a,b) pair summed over 6 faces *)
bstiffSurfIrrepFull[irrep_, a_Integer, b_Integer,
    AelasV_, BelasV_, DlamV_, DmuV_] :=
  Sum[
    bstiffSurfIrrep1Face[irrep, a, b, fAxis, fSign,
      AelasV, BelasV, DlamV, DmuV],
    {fAxis, 1, 3}, {fSign, {-1, 1}}];

(* ================================================================== *)
(* Section 8. Compute surface stiffness ANALYTICALLY via Mp/MpB        *)
(* ================================================================== *)
Print[];
Print["---- Section 8: analytical surface stiffness via master integrals ----"];

(* 4 channels: {Aelas, Belas, Dlam, Dmu} *)
channels = {
  {1, 0, 1, 0},  (* A*Dlam *)
  {1, 0, 0, 1},  (* A*Dmu *)
  {0, 1, 1, 0},  (* B*Dlam *)
  {0, 1, 0, 1}   (* B*Dmu *)
};
channelLabels = {"A_Dlam", "A_Dmu", "B_Dlam", "B_Dmu"};

(* ---- Section 8a: Master integrals (symbolic + NIntegrate fallback) ---- *)
Print["  Section 8a: computing master integrals ..."];
t0mi = AbsoluteTime[];

ClearAll[MpSurf, MpBSurf];

(* A-channel: Mp[p,q,r] = Integral_[0,1]^3 x^p y^q z^r / sqrt(x^2+y^2+z^2)
   Use symbolic Integrate (fast, seconds per value for most indices).
   4-5 values return Undefined — fall back to NIntegrate for those. *)
MpSurf[p_Integer, q_Integer, r_Integer] /; !OrderedQ[{p, q, r}] :=
  MpSurf @@ Sort[{p, q, r}];
MpSurf[p_, q_, r_] := MpSurf[p, q, r] = Module[{raw, real},
  raw = Integrate[
    x^p y^q z^r / Sqrt[x^2 + y^2 + z^2],
    {z, 0, 1}, {y, 0, 1}, {x, 0, 1}];
  real = Simplify[Re[ComplexExpand[raw]]];
  (* If symbolic result is not a valid number, fall back to NIntegrate *)
  If[NumericQ[N[real]] && FreeQ[real, Undefined | Indeterminate],
    real,
    Print["    Mp[", p, ",", q, ",", r,
      "]: Integrate returned non-numeric, using NIntegrate fallback"];
    NIntegrate[x^p y^q z^r / Sqrt[x^2 + y^2 + z^2],
      {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
      PrecisionGoal -> 12, AccuracyGoal -> 12, MaxRecursion -> 25]]];

(* B-channel: MpB[p,q,r] = Integral_[0,1]^3 x^p y^q z^r / (x^2+y^2+z^2)^(3/2)
   Converges for p+q+r >= 2.  Same strategy: symbolic first, NIntegrate fallback.
   Identity: MpB[p+2,q,r] + MpB[p,q+2,r] + MpB[p,q,r+2] = Mp[p,q,r]
   Used for p+q+r < 2 where the integral diverges. *)
MpBSurf[p_Integer, q_Integer, r_Integer] /; !OrderedQ[{p, q, r}] :=
  MpBSurf @@ Sort[{p, q, r}];
MpBSurf[p_, q_, r_] := MpBSurf[p, q, r] =
  If[p + q + r < 2,
    (* Use recurrence from higher indices *)
    MpSurf[p, q, r] - MpBSurf[p + 2, q, r] - MpBSurf[p, q + 2, r],
    Module[{raw, real},
      raw = Integrate[
        x^p y^q z^r / (x^2 + y^2 + z^2)^(3/2),
        {z, 0, 1}, {y, 0, 1}, {x, 0, 1}];
      real = Simplify[Re[ComplexExpand[raw]]];
      If[NumericQ[N[real]] && FreeQ[real, Undefined | Indeterminate],
        real,
        Print["    MpB[", p, ",", q, ",", r,
          "]: Integrate returned non-numeric, using NIntegrate fallback"];
        NIntegrate[x^p y^q z^r / (x^2 + y^2 + z^2)^(3/2),
          {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
          PrecisionGoal -> 12, AccuracyGoal -> 12, MaxRecursion -> 25]]]];

(* Pre-compute all needed Mp values (31 unique sorted triples,
   max index 4, r <= 2 per the notebook Section 5 analysis).
   Most complete in seconds; 4-5 fall back to NIntegrate. *)
neededMpTriples = Union[Sort /@ Flatten[
  Table[{p, q, rr}, {p, 0, 4}, {q, 0, 4}, {rr, 0, 2}], 2]];
Print["    Computing ", Length[neededMpTriples], " unique Mp values ..."];
Do[
  Module[{t0 = AbsoluteTime[], val},
    val = MpSurf @@ triple;
    Print["    Mp", triple, " = ", NumberForm[N[val, 16], 16],
      "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"]],
  {triple, neededMpTriples}];

(* MpB values are computed on demand during assembly via memoization.
   No need to pre-compute: the assembly loop triggers exactly what's needed. *)

Print["  Master integrals done in ",
  Round[AbsoluteTime[] - t0mi, 1], " s"];

(* ---- Section 8b: Analytical surface integral per face ---- *)
Print[];
Print["  Section 8b: analytical surface integral assembly ..."];

(* analyticalSurf1Face computes the surface bilinear integral for one face
   ANALYTICALLY by:
   1. Build phi_alpha(r) and traction t_beta(r') symbolically
   2. Substitute (sigma, xi, v) coordinates
   3. Expand into monomials, integrate sigma analytically
   4. Collect xi1^p xi2^q v^r coefficients -> sum of Mp or MpB values

   Returns the integral value (exact symbolic, evaluated to high precision).
*)
analyticalSurf1Face[irrep_, aa_Integer, bb_Integer,
    fAxis_Integer, fSign_Integer, AelasV_, BelasV_, DlamV_, DmuV_] :=
  Module[{shared, rv, sv, PsiA, Tb, xdiff,
          sig1, xi1, sig2, xi2, vv,
          intA, intB, polyA, polyB, rhoSq,
          monoRules, sigIntA, sigIntB,
          expandedA, expandedB, result = 0,
          crA, crB, expo, coeff, mpval},

    shared = DeleteCases[{1, 2, 3}, fAxis];

    (* Build mode and traction symbolically using rv variables *)
    rv = {rv1, rv2, rv3};
    sv = Table[0, 3];
    sv[[fAxis]] = fSign;
    sv[[shared[[1]]]] = ss1;
    sv[[shared[[2]]]] = ss2;

    PsiA = irrepMode[irrep, aa, rv];
    Tb = irrepTraction[irrep, bb, fAxis, fSign, sv];

    (* Substitute channel values *)
    PsiA = PsiA /. {Aelas -> AelasV, Belas -> BelasV,
                     Dlam -> DlamV, Dmu -> DmuV};
    Tb = Tb /. {Aelas -> AelasV, Belas -> BelasV,
                 Dlam -> DlamV, Dmu -> DmuV};

    (* If traction is zero, return 0 *)
    If[Expand[Tb] === {0, 0, 0}, Return[0]];

    (* Displacement vector: x = r - r' *)
    xdiff = rv - sv;

    (* A-channel integral: sum_i PsiA_i * Tb_i / |x| *)
    intA = AelasV * Expand[PsiA . Tb];
    (* B-channel integral: sum_{ij} PsiA_i * Tb_j * x_i * x_j / |x|^3 *)
    intB = BelasV * Expand[Sum[PsiA[[ii]] * Tb[[jj]] * xdiff[[ii]] * xdiff[[jj]],
      {ii, 3}, {jj, 3}]];

    If[Expand[intA] === 0 && Expand[intB] === 0, Return[0]];

    (* Apply (sigma, xi, v) substitution:
       For shared axes a1, a2:
         r_{a_k} = sig_k + xi_k,  s_{a_k} = sig_k - xi_k
       For normal axis:
         r_normal = fSign*(1 - 2*vv)  (so r_normal = fSign at vv=0)
       x_{a_k} = r_{a_k} - s_{a_k} = 2*xi_k
       x_normal = r_normal - fSign = -2*fSign*vv
    *)
    monoRules = {
      rv[[shared[[1]]]] -> sig1 + xi1,
      rv[[shared[[2]]]] -> sig2 + xi2,
      rv[[fAxis]] -> fSign (1 - 2 vv),
      ss1 -> sig1 - xi1,
      ss2 -> sig2 - xi2
    };

    polyA = Expand[intA /. monoRules];
    polyB = Expand[intB /. monoRules];

    (* For A-channel: integrand is polyA / (2*rho) where
       rho = sqrt(xi1^2 + xi2^2 + vv^2)
       For B-channel: integrand is polyB / (8*rho^3) where
       the factor 1/8 comes from |x|^3 = (2rho)^3 = 8*rho^3
       and the x_i x_j already have factors of 2 built in.

       Actually: x_{a1} = 2*xi1, x_{a2} = 2*xi2, x_normal = -2*fSign*vv
       So x_i x_j / |x|^3 = (2 xi_i)(2 xi_j) / (2 rho)^3
                            = 4 xi_i xi_j / (8 rho^3)
                            = xi_i xi_j / (2 rho^3)
       where xi_i is the (sigma,xi,v) version of x_i/2.

       But polyB already has the x_i x_j factors as 2*xi1 etc,
       so polyB/(2rho)^3 = polyB / (8 rho^3).

       Let's work in terms of the xdiff substitution directly.
       After monoRules, xdiff becomes:
         x[shared1] = 2*xi1, x[shared2] = 2*xi2, x[normal] = -2*fSign*vv
       So polyB = BelasV * sum PsiA_i Tb_j (2*xi1 or 2*xi2 or -2*fSign*vv)_i * (...)_j
       and the kernel 1/|x|^3 = 1/(2*rho)^3 = 1/(8*rho^3).

       For the FULL integral (including Jacobian etc):
       Original integral = int_{[-1,1]^3} int_{[-1,1]^2} ... dV(r) dS(s)
       After (sig,xi,v) substitution:
         Jacobian: |d(r_a, s_a)/d(sig_a, xi_a)| = 2 per pair, 2 pairs -> 4
         dr_normal = -2*fSign*dvv -> factor |−2| = 2
         Total Jacobian = 4 * 2 = 8

       Domain: |sig_a| + |xi_a| <= 1 (diamond), vv in [0, 1]
       By octant symmetry in (sig1, xi1, sig2, xi2): factor 4 per pair = 16
       (only if integrand has definite parity — we enforce this below)

       Restricted domain: sig_a in [0, 1-xi_a], xi_a in [0, 1], vv in [0, 1]
       Total prefactor = 8 (Jacobian) * 16 (octant) = 128

       For A-channel: integral = 128 * int_[0,1]^3 [int sigma] polyA/(2*rho)
                                = 64 * int_[0,1]^3 [after sigma int] / rho

       For B-channel: integral = 128 * int_[0,1]^3 [int sigma] polyB/(8*rho^3)
                                = 16 * int_[0,1]^3 [after sigma int] / rho^3
    *)

    (* Integrate sigma1 and sigma2 analytically.
       The integrand is a polynomial in (sig1, sig2, xi1, xi2, vv).
       Monomials with odd power in sig_a or xi_a vanish by octant symmetry.

       int_0^{1-xi} sig^k dsig = (1-xi)^{k+1}/(k+1)

       We expand each monomial sig1^a1 sig2^a2 xi1^b1 xi2^b2 vv^c, and:
       - Drop terms with odd a1, a2, b1, or b2 (octant parity)
       - Integrate sig: (1-xi1)^{a1+1}/(a1+1) * (1-xi2)^{a2+1}/(a2+1)
       - Expand (1-xi)^n in powers of xi, collect monomials xi1^p xi2^q vv^r
    *)

    (* Helper: expand polynomial, integrate sigma, return polynomial in xi1,xi2,vv *)
    sigmaIntegrate[poly_] := Module[{cr, totalResult = 0, ea1, ea2, eb1, eb2, ec,
        cval, sigFactor, expanded1, expanded2, term},
      cr = CoefficientRules[Expand[poly], {sig1, sig2, xi1, xi2, vv}];
      Do[
        {ea1, ea2, eb1, eb2, ec} = rule[[1]];
        cval = rule[[2]];
        (* Octant parity: only even powers of sig and xi survive *)
        If[OddQ[ea1] || OddQ[ea2] || OddQ[eb1] || OddQ[eb2], Continue[]];
        (* Sigma integration: int_0^{1-xi} sig^k dsig = (1-xi)^{k+1}/(k+1) *)
        sigFactor = (1 - xi1)^(ea1 + 1)/(ea1 + 1) * (1 - xi2)^(ea2 + 1)/(ea2 + 1);
        (* Expand (1-xi)^n *)
        expanded1 = Expand[sigFactor * xi1^eb1 * xi2^eb2 * vv^ec];
        totalResult += cval * expanded1,
        {rule, cr}];
      Expand[totalResult]];

    (* Process A-channel *)
    If[Expand[polyA] =!= 0,
      sigIntA = sigmaIntegrate[polyA];
      (* Now sigIntA is a polynomial in xi1, xi2, vv.
         Each monomial xi1^p xi2^q vv^r contributes coeff * Mp[p,q,r].
         Prefactor: 64 (from 128/2) *)
      crA = CoefficientRules[Expand[sigIntA], {xi1, xi2, vv}];
      Do[
        expo = rule[[1]];
        coeff = rule[[2]];
        If[coeff =!= 0,
          mpval = MpSurf[expo[[1]], expo[[2]], expo[[3]]];
          result += 64 * coeff * mpval],
        {rule, crA}]];

    (* Process B-channel *)
    If[Expand[polyB] =!= 0,
      sigIntB = sigmaIntegrate[polyB];
      (* Each monomial xi1^p xi2^q vv^r contributes coeff * MpB[p,q,r].
         Prefactor: 16 (from 128/8) *)
      crB = CoefficientRules[Expand[sigIntB], {xi1, xi2, vv}];
      Do[
        expo = rule[[1]];
        coeff = rule[[2]];
        If[coeff =!= 0,
          mpval = MpBSurf[expo[[1]], expo[[2]], expo[[3]]];
          result += 16 * coeff * mpval],
        {rule, crB}]];

    N[result, 16]];

(* Full surface stiffness for one (a,b) pair summed over 6 faces — analytical *)
analyticalSurfFull[irrep_, aa_Integer, bb_Integer,
    AelasV_, BelasV_, DlamV_, DmuV_] :=
  Sum[
    analyticalSurf1Face[irrep, aa, bb, fAxis, fSign,
      AelasV, BelasV, DlamV, DmuV],
    {fAxis, 1, 3}, {fSign, {-1, 1}}];

(* ---- Section 8c: Loop over irreps ---- *)
Print[];
Print["  Section 8c: computing surface stiffness per irrep ..."];

BstiffSurfIrrep = Association[];
t0all = AbsoluteTime[];

Do[
  Module[{m, BsurfCh, t0irrep},
    m = irrepData[irrep, "m"];
    BsurfCh = ConstantArray[0, {m, m, 4}];
    t0irrep = AbsoluteTime[];
    Print["  Computing ", irrep, " (", m, "x", m, " x 4 channels)..."];

    Do[
      Do[
        Module[{Av, Bv, Dlv, Dmv, val, t0entry},
          {Av, Bv, Dlv, Dmv} = channels[[ch]];
          t0entry = AbsoluteTime[];
          val = analyticalSurfFull[irrep, aa, bb, Av, Bv, Dlv, Dmv];
          BsurfCh[[aa, bb, ch]] = val;
          Print["    [", aa, ",", bb, "] ch=", channelLabels[[ch]],
            " = ", NumberForm[val, 12],
            "  (", Round[AbsoluteTime[] - t0entry, 0.1], " s)"]],
        {ch, 1, 4}],
      {aa, 1, m}, {bb, 1, m}];

    BstiffSurfIrrep[irrep] = BsurfCh;
    Print["    ", irrep, " done in ",
      Round[AbsoluteTime[] - t0irrep, 1], " s"];
    Print[]],
  {irrep, ungeradeIrreps}];

Print["  Total analytical surface computation: ",
  Round[AbsoluteTime[] - t0all, 1], " s"];

(* ---- Section 8d: Cross-check against stored NIntegrate results ---- *)
Print[];
Print["  Section 8d: cross-check against NIntegrate reference ..."];
Module[{refPath, refData, maxErr = 0, refSurf},
  refPath = FileNameJoin[{$here, "CubeT27StiffnessLS_Results.wl"}];
  If[FileExistsQ[refPath],
    refData = Get[refPath];
    refSurf = Lookup[refData, "BstiffSurfIrrep", None];
    If[refSurf =!= None,
      Do[
        Module[{refBlock, newBlock, m, err},
          refBlock = Lookup[refSurf, irrep, None];
          If[refBlock =!= None,
            newBlock = BstiffSurfIrrep[irrep];
            m = irrepData[irrep, "m"];
            Do[
              Module[{refVal, newVal, abserr},
                refVal = refBlock[[aa, bb, ch]];
                newVal = newBlock[[aa, bb, ch]];
                If[refVal =!= 0,
                  abserr = Abs[newVal - refVal];
                  If[abserr > maxErr, maxErr = abserr];
                  If[abserr > 0.01 Abs[refVal],
                    Print["    WARNING: ", irrep, "[", aa, ",", bb,
                      "] ch=", ch, " ref=", refVal, " new=", newVal,
                      " err=", abserr]]]],
              {aa, m}, {bb, m}, {ch, 4}];
            Print["    ", irrep, " max |new - ref| = ",
              ScientificForm[maxErr, 4]];
            maxErr = 0]],
        {irrep, ungeradeIrreps}],
      Print["    (no BstiffSurfIrrep in reference file)")],
    Print["    (reference file not found, skipping cross-check)"]]
];

(* ================================================================== *)
(* Section 9. Combine volume + surface, extract 4-channel matrices      *)
(* ================================================================== *)
Print[];
Print["---- Section 9: total BstiffLS = vol + surf (4 channels) ----"];

BstiffLSIrrep = Association[];
Do[
  Module[{m, BvolSym, BsurfCh, BtotCh},
    m = irrepData[irrep, "m"];
    BvolSym = BstiffVolIrrep[irrep] /. atomRules;

    (* Extract volume part per channel *)
    BtotCh = ConstantArray[0, {m, m, 4}];
    Do[
      Module[{Av, Bv, Dlv, Dmv, BvolNum},
        {Av, Bv, Dlv, Dmv} = channels[[ch]];
        BvolNum = N[BvolSym /. {Aelas -> Av, Belas -> Bv,
          Dlam -> Dlv, Dmu -> Dmv}, 16];
        (* Total = vol + surf *)
        Do[
          BtotCh[[a, b, ch]] = BvolNum[[a, b]] + BstiffSurfIrrep[irrep][[a, b, ch]],
          {a, m}, {b, m}]],
      {ch, 4}];

    BstiffLSIrrep[irrep] = BtotCh;

    Print["  ", irrep, " total BstiffLS (per channel):"];
    Do[
      Print["    ", channelLabels[[ch]], ":"];
      Module[{mat = BtotCh[[All, All, ch]]},
        If[m == 1,
          Print["      ", NumberForm[mat[[1, 1]], 12]],
          Do[Print["      row ", i, ": ",
            Map[Function[x, NumberForm[x, 12]], mat[[i]]]], {i, m}]]],
      {ch, 4}];
    Print[]],
  {irrep, ungeradeIrreps}];

(* ================================================================== *)
(* Section 10. Gerade cross-check: LS stiffness vs Eshelby              *)
(* ================================================================== *)
Print[];
Print["---- Section 10: gerade cross-check ----"];

(* For gerade irreps, BstiffVol should be zero (no volume force for
   strain modes) and BstiffSurf should reproduce the Eshelby contraction.
   The total BstiffLS for gerade irreps at (a0,b0)x(Dlam,Dmu) should match:
     sigma_A1g = 3*T1c + 2*T2c + T3c
     sigma_Eg = 2*T2c + T3c
     sigma_T2g = 2*T2c
   where T1c = Ac*Dlam + (2Ac+4Bc+Cc)*Dmu
         T2c = Bc*Dmu
         T3c = Cc*Dmu
   are expressed via the Eshelby integrals A^c, B^c, C^c.

   We verify this by computing BstiffVol for gerade irreps (should be 0)
   and noting that the gerade stiffness is already handled by the
   self-consistent amplification code path (not this script). *)

Do[
  Module[{Usym, BvolTest},
    Usym = UsymData[irrep];
    BvolTest = Transpose[Usym] . BstiffVol27 . Usym;
    BvolTest = N[BvolTest /. atomRules /.
      {Aelas -> 1, Belas -> 0, Dlam -> 1, Dmu -> 1}, 16];
    Print["  ", irrep, " BstiffVol = ", BvolTest, "  (expect 0)"]],
  {irrep, {"A1g", "Eg", "T2g"}}];

(* ================================================================== *)
(* Section 11. Python-ready output                                      *)
(* ================================================================== *)
Print[];
Print["==== PYTHON COPY-PASTE FORMAT ===="];
Print[];

Do[
  Module[{m, BtotCh},
    m = irrepData[irrep, "m"];
    BtotCh = BstiffLSIrrep[irrep];

    Print["    # ", irrep, " BstiffLS (", m, "x", m, " x 4 channels)"];
    Do[
      Print["    # Channel: ", channelLabels[[ch]]];
      Module[{mat = BtotCh[[All, All, ch]]},
        If[m == 1,
          Print["    Bstiff_", channelLabels[[ch]], "_", irrep, " = ",
            CForm[mat[[1, 1]]]],
          Print["    Bstiff_", channelLabels[[ch]], "_", irrep, " = np.array(["];
          Do[
            Print["        [",
              StringRiffle[Map[ToString[CForm[#]] &, mat[[i]]], ", "],
              "],"],
            {i, m}];
          Print["    ])"];
        ]],
      {ch, 4}];
    Print[]],
  {irrep, ungeradeIrreps}];

(* ================================================================== *)
(* Section 12. Export results to .wl file                               *)
(* ================================================================== *)
Print[];
Print["---- Section 12: export results ----"];

outPath = FileNameJoin[{$here, "CubeT27StiffnessLS_Results.wl"}];
Put[
  <|"BstiffLSIrrep" -> Normal[BstiffLSIrrep],
    "BstiffVolIrrep" -> Normal[Table[
      irrep -> N[BstiffVolIrrep[irrep] /. atomRules /.
        {Aelas -> 1, Belas -> 0, Dlam -> 1, Dmu -> 0}, 16],
      {irrep, ungeradeIrreps}]],
    "BstiffSurfIrrep" -> Normal[BstiffSurfIrrep],
    "channels" -> channels,
    "channelLabels" -> channelLabels,
    "UsymData" -> Normal[UsymData]|>,
  outPath];
Print["  Wrote ", outPath];

Print[];
Print["==== CubeT27Stiffness_LS.wl done ===="];
