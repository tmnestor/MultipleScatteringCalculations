#!/usr/bin/env wolframscript
(* ============================================================================
   THE FIRST-ORDER CONTRAST OPERATOR  DeltaA  --- and its augmentation

   Gate for the question: can the moment method be applied to the single-site
   T-matrix directly from the first-order system  d3 q = A q + d ?

   This script establishes the algebra that must hold BEFORE any moment
   integral is attempted.  It does not compute a T-matrix.  What it settles:

     1. the A it uses is the validated one                 (vs Appendix J)
     2. q built from a displacement field obeys d3 q = A q, with the
        equation-of-motion rows carrying exactly the Navier body force and
        the constitutive rows vanishing IDENTICALLY
     3. DeltaA = A[total] - A[background] is the exact contrast operator:
        the total field solves the background system driven by DeltaA q
     4. DeltaA is NOT linear in (Dlam, Dmu, Drho) --- it resums the local
        C33^{-1} response exactly.  Quantified at the project's validated
        moderate contrast.
     5. the thesis augmentation carries over to 3-D Cartesian: integration by
        parts against the symplectic metric J6 turns the DIFFERENTIAL DeltaA
        into a MULTIPLICATIVE DeltaC on an enlarged basis.  The enlarged
        dimension is DERIVED here, not assumed.

   Provenance of the construction being ported:
     Thesis TRepresentations.tex, Chapter 4 (secEffScat, L748):
       dAdef     the 6x6 differential contrast operator DeltaA(x; dx, ky)
       weakRay0  the pairing  Int <phi| J6 DeltaA |psi> dx
       CdefAug   the resulting 8x8 MULTIPLICATIVE DeltaC (2.5-D)
       weakRay   the cell integral, evaluated there by midpoint/Rayleigh
       LSTrep4   the renormalised, nonsingular, compact integral equation
     Paper (Wapenaar), Appendix D (D.20)-(D.23) and supplemental Appendix J.

   The thesis does this in 2.5-D with a cylindrical-harmonic basis and the
   midpoint approximation.  The 3-D Cartesian operator has never been written
   down in this repository; that is what this file supplies.

   Coordinates and conventions follow MatrixVectorWaveEquation.wl: x3 is
   depth, e^{-i w t}, v = -i w u.
   ============================================================================ *)

bar = StringJoin @@ ConstantArray["=", 60];
Print[bar];
Print["  DeltaA --- the first-order contrast operator, and its augmentation"];
Print[bar];

nPass = 0; nFail = 0;
report[label_, bool_] := (
   If[TrueQ[bool], nPass++, nFail++];
   Print[If[TrueQ[bool], "  PASS  ", "  ****FAIL****  "], label]);

(* ============================================================================
   Part 0.  Operator algebra and the build of A.

   Self-contained on purpose: this file must stand alone as a gate, so it
   rebuilds A from c_ijkl by (D.17), (D.20)-(D.23) and re-verifies it against
   Appendix J rather than importing a result.  Same algebra as
   MatrixVectorWaveEquation.wl.
   ============================================================================ *)

xv = {x1, x2};
xs = {x1, x2, x3};

act[zeroOp, f_] := 0;
act[multOp[a_], f_] := a f;
act[dOp[al_], f_] := D[f, xv[[al]]];
act[compOp[ps__], f_] := Fold[act[#2, #1] &, f, Reverse[{ps}]];
act[plusOp[ps__], f_] := Total[act[#, f] & /@ {ps}];
act[scalOp[c_, o_], f_] := c act[o, f];

opMatAct[m_, q_] :=
  Table[Sum[act[m[[i, j]], q[[j]]], {j, Length[q]}], {i, Length[m]}];

dl[i_, j_] := KroneckerDelta[i, j];

(* buildOps[lam, mu, rho] -> {A11, A12, A21, A22} as operator matrices. *)
buildOps[lamx_, mux_, rhox_] := Module[
   {kc, cc, cmat, c33inv, umat, a11, a12, a21, a22},
   kc = lamx + 2 mux;
   cc[i_, j_, k_, l_] :=
    lamx dl[i, j] dl[k, l] + mux (dl[i, k] dl[j, l] + dl[i, l] dl[j, k]);
   cmat[j_, l_] := Table[cc[ii, j, kk, l], {ii, 3}, {kk, 3}];
   c33inv = Simplify@Inverse[cmat[3, 3]];
   umat[al_, l_] := Simplify[cmat[al, l] - cmat[al, 3].c33inv.cmat[3, l]];
   a11 = Table[
     plusOp @@ Table[
       scalOp[-1, compOp[dOp[al], multOp[(cmat[al, 3].c33inv)[[ii, kk]]]]],
       {al, 2}], {ii, 3}, {kk, 3}];
   a12 = Table[
     plusOp[multOp[I w rhox dl[ii, kk]],
      Sequence @@ Flatten[
        Table[scalOp[-1/(I w),
          compOp[dOp[al], multOp[umat[al, be][[ii, kk]]], dOp[be]]],
         {al, 2}, {be, 2}], 1]], {ii, 3}, {kk, 3}];
   a21 = Table[multOp[I w c33inv[[ii, kk]]], {ii, 3}, {kk, 3}];
   a22 = Table[
     plusOp @@ Table[
       scalOp[-1, compOp[multOp[(c33inv.cmat[3, be])[[ii, kk]]], dOp[be]]],
       {be, 2}], {ii, 3}, {kk, 3}];
   {a11, a12, a21, a22}];

assemble[{a11_, a12_, a21_, a22_}] :=
  Join[MapThread[Join, {a11, a12}], MapThread[Join, {a21, a22}]];

(* background medium, laterally varying *)
lam = lamf[x1, x2, x3]; mu = muf[x1, x2, x3]; rho = rhof[x1, x2, x3];
kcbg = lam + 2 mu;

{a11op, a12op, a21op, a22op} = buildOps[lam, mu, rho];
aop = assemble[{a11op, a12op, a21op, a22op}];

(* --- provenance: the same four comparisons as the derivation notebook --- *)
nu1 = Simplify[4 mu (lam + mu)/kcbg];
nu2 = Simplify[2 mu lam/kcbg];
gam = lam/kcbg;

a11ref = {
   {zeroOp, zeroOp, scalOp[-1, compOp[dOp[1], multOp[gam]]]},
   {zeroOp, zeroOp, scalOp[-1, compOp[dOp[2], multOp[gam]]]},
   {scalOp[-1, dOp[1]], scalOp[-1, dOp[2]], zeroOp}};
a12ref = {
   {plusOp[multOp[I w rho],
     scalOp[-1/(I w), compOp[dOp[1], multOp[nu1], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[2], multOp[mu], dOp[2]]]],
    plusOp[scalOp[-1/(I w), compOp[dOp[2], multOp[mu], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[1], multOp[nu2], dOp[2]]]], zeroOp},
   {plusOp[scalOp[-1/(I w), compOp[dOp[2], multOp[nu2], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[1], multOp[mu], dOp[2]]]],
    plusOp[multOp[I w rho],
     scalOp[-1/(I w), compOp[dOp[1], multOp[mu], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[2], multOp[nu1], dOp[2]]]], zeroOp},
   {zeroOp, zeroOp, multOp[I w rho]}};
a21ref = {{multOp[I w/mu], zeroOp, zeroOp}, {zeroOp, multOp[I w/mu], zeroOp},
   {zeroOp, zeroOp, multOp[I w/kcbg]}};
a22ref = {{zeroOp, zeroOp, scalOp[-1, dOp[1]]},
   {zeroOp, zeroOp, scalOp[-1, dOp[2]]},
   {scalOp[-1, compOp[multOp[gam], dOp[1]]],
    scalOp[-1, compOp[multOp[gam], dOp[2]]], zeroOp}};

opEq[m1_, m2_] := Module[{q = Table[Symbol["zz" <> ToString[i]][x1, x2, x3], {i, 3}]},
   Simplify[opMatAct[m1, q] - opMatAct[m2, q]] === {0, 0, 0}];

Print[""];
Print["--- Part 0: provenance --- the A used here is the validated one ---"];
report["A11 == (J.5)", opEq[a11op, a11ref]];
report["A12 == (J.6)", opEq[a12op, a12ref]];
report["A21 == (J.7)", opEq[a21op, a21ref]];
report["A22 == (J.8)", opEq[a22op, a22ref]];

(* ============================================================================
   Part 1.  d3 q = A q, checked against the Navier equation.

   This is the foundation.  Build q = (-tau_13, -tau_23, -tau_33, v1, v2, v3)
   from a GENERIC displacement field in a laterally varying medium, and test

       d3 q  -  A q   =?  d

   Two halves, with different meanings and different strengths:

   * Rows 4-6 (the d3 q2 = A21 q1 + A22 q2 block) are the constitutive
     relation rearranged.  No equation of motion is used, so they must vanish
     IDENTICALLY for ANY u whatsoever.  If they do not, the elimination of
     tau_1, tau_2 via C33^{-1} is wrong.
   * Rows 1-3 (the d3 q1 block) are the equation of motion.  They must equal
     the body force f_i = -rho w^2 u_i - d_j tau_ij exactly.  Anything left
     over is a defect in A11 or A12.

   Note this is a much stronger statement than the plane-wave / eigenvalue
   checks in MatrixVectorWaveEquation.wl: u is generic and the medium varies
   laterally, so every operator-ordering term is exercised.
   ============================================================================ *)

uvec = Table[uu[i][x1, x2, x3], {i, 3}];
vvec = -I w uvec;
eps[i_, j_] := (D[uvec[[i]], xs[[j]]] + D[uvec[[j]], xs[[i]]])/2;
tau[i_, j_] := lam Sum[eps[k, k], {k, 3}] dl[i, j] + 2 mu eps[i, j];

qvec = Join[Table[-tau[ii, 3], {ii, 3}], vvec];
navier = Table[-rho w^2 uvec[[ii]] - Sum[D[tau[ii, j], xs[[j]]], {j, 3}], {ii, 3}];

resid = Simplify[D[qvec, x3] - opMatAct[aop, qvec]];

Print[""];
Print["--- Part 1: d3 q - A q, for a generic u in a varying medium ---------"];
report["rows 4-6 vanish IDENTICALLY (constitutive rows, no EoM used)",
  Simplify[resid[[4 ;; 6]]] === {0, 0, 0}];
report["rows 1-3 == the Navier body force  f_i = -rho w^2 u_i - d_j tau_ij",
  Simplify[resid[[1 ;; 3]] - navier] === {0, 0, 0}];

(* ============================================================================
   Part 2.  DeltaA = A[total] - A[background].

   Inside the scatterer the medium is (lam+Dlam, mu+Dmu, rho+Drho).  Write
   A_tot = A_bg + DeltaA.  The total field then solves the BACKGROUND system
   driven by DeltaA acting on itself,

       (d3 - A_bg) q = DeltaA q + f ,

   which is the Lippmann-Schwinger equation of the first-order system.  This
   is the thesis's LSTrep1/LSTrep3 before renormalisation.
   ============================================================================ *)

lamT = lam + dlam; muT = mu + dmu; rhoT = rho + drho;
{t11, t12, t21, t22} = buildOps[lamT, muT, rhoT];
aopT = assemble[{t11, t12, t21, t22}];

(* DeltaA, entrywise, as an operator matrix *)
dAop = Table[plusOp[aopT[[i, j]], scalOp[-1, aop[[i, j]]]], {i, 6}, {j, 6}];

(* the total field: same u, but stresses built with the TOTAL moduli *)
epsT[i_, j_] := eps[i, j];
tauT[i_, j_] := lamT Sum[epsT[k, k], {k, 3}] dl[i, j] + 2 muT epsT[i, j];
qvecT = Join[Table[-tauT[ii, 3], {ii, 3}], vvec];
navierT = Table[-rhoT w^2 uvec[[ii]] - Sum[D[tauT[ii, j], xs[[j]]], {j, 3}], {ii, 3}];

residLS = Simplify[
   D[qvecT, x3] - opMatAct[aop, qvecT] - opMatAct[dAop, qvecT]];

Print[""];
Print["--- Part 2: DeltaA is the exact contrast operator -------------------"];
report["(d3 - A_bg) q_tot - DeltaA q_tot == body force of the TOTAL medium",
  Simplify[residLS - Join[navierT, {0, 0, 0}]] === ConstantArray[0, 6]];

(* the symplectic relation is linear in A, so DeltaA inherits it; free to check *)
j6 = ArrayFlatten[{{ConstantArray[0, {3, 3}], IdentityMatrix[3]},
                     {-IdentityMatrix[3], ConstantArray[0, {3, 3}]}}];
opT[zeroOp] := zeroOp;
opT[multOp[a_]] := multOp[a];
opT[dOp[al_]] := scalOp[-1, dOp[al]];
opT[compOp[ps__]] := compOp @@ Reverse[opT /@ {ps}];
opT[plusOp[ps__]] := plusOp @@ (opT /@ {ps});
opT[scalOp[c_, o_]] := scalOp[c, opT[o]];
opMatT[m_] := Transpose[Map[opT, m, {2}]];

matOpMul[num_?MatrixQ, m_] :=
  Table[plusOp @@ Table[scalOp[num[[i, k]], m[[k, j]]], {k, Length[m]}],
   {i, Length[num]}, {j, Length[m[[1]]]}];
opMatMatMul[m_, num_?MatrixQ] :=
  Table[plusOp @@ Table[scalOp[num[[k, j]], m[[i, k]]], {k, Length[num]}],
   {i, Length[m]}, {j, Length[num[[1]]]}];
opMatSum[m1_, m2_] := MapThread[plusOp, {m1, m2}, 2];

opEq6[m1_, m2_] := Module[{q = Table[Symbol["yy" <> ToString[i]][x1, x2, x3], {i, 6}]},
   Simplify[opMatAct[m1, q] - opMatAct[m2, q]] === ConstantArray[0, 6]];

report["DeltaA^t J6 = -J6 DeltaA  (inherited from eq. 27, linear in A)",
  opEq6[opMatMatMul[opMatT[dAop], j6],
        matOpMul[-j6, dAop]]];

(* ============================================================================
   Part 3.  DeltaA is NOT linear in the contrast.

   This is the structural difference from the Navier Lippmann-Schwinger
   equation, whose potential is exactly linear in (Dlam, Dmu, Drho).  Here the
   elimination of tau_1, tau_2 introduced C33^{-1}, so

       DeltaA21 = i w (C33tot^{-1} - C33^{-1})

   which is exact but nonlinear.  The consequence matters: a Born or Rayleigh
   truncation of the first-order LS equation is NOT the same truncation as the
   Navier one, so the two need not agree at any finite closure order even
   though they agree when solved exactly.  This is the reason "reproduce T9
   exactly" is the wrong expectation for the collocation closure.
   ============================================================================ *)

exactB = 1/(mu + dmu) - 1/mu;
linB = -dmu/mu^2;
exactA = 1/(kcbg + dlam + 2 dmu) - 1/kcbg;
linA = -(dlam + 2 dmu)/kcbg^2;

Print[""];
Print["--- Part 3: the nonlinearity in the contrast -----------------------"];
report["DeltaA21 entry (1,1) is exactly i w (1/(mu+Dmu) - 1/mu)",
  Simplify[act[dAop[[4, 1]], ph] - I w exactB ph] === 0];
report["its linearisation is NOT equal to it (the resummation is real)",
  Simplify[exactB - linB] =!= 0];
report["exact/linear ratio for Db is mu/(mu+Dmu)",
  Simplify[exactB/linB - mu/(mu + dmu)] === 0];

(* the project's validated moderate contrast, in GPa / (g/cm^3) *)
numRule = {muf[x1, x2, x3] -> 22.5, lamf[x1, x2, x3] -> 17.5,
   dmu -> 1.0, dlam -> 2.0};
Print["  at the validated moderate contrast (alpha=5, beta=3, rho=2.5;",
  " Dlam=+2, Dmu=+1 GPa):"];
Print["     Dmu/mu                      = ", (dmu/mu) /. numRule];
Print["     exact/linear for Db = 1/mu  = ", Simplify[exactB/linB] /. numRule];
Print["     exact/linear for Da = 1/Kc  = ", Simplify[exactA/linA] /. numRule];

(* ============================================================================
   Part 4.  The augmentation: DIFFERENTIAL DeltaA -> MULTIPLICATIVE DeltaC.

   The thesis move (weakRay0 -> CdefAug), here in 3-D Cartesian.  The bilinear
   form paired with the symplectic metric is

       b[phi,psi] = phi^t J6 DeltaA psi ,

   and DeltaA carries derivatives.  Inside the scatterer the contrasts are
   constant, so every coefficient is (Dc times the indicator 1_V).  A
   derivative standing to the LEFT of such a coefficient would differentiate
   the INDICATOR and produce a surface delta on the cube faces.  Integration
   by parts over all space moves that derivative onto phi instead, where it
   meets a smooth test function, and the face term never arises:

       Int phi_i (-d_a (m 1_V psi_j)) dx = + Int (d_a phi_i) m 1_V psi_j dx

   After the transfer every term has the form (d^a phi)^t . const . (d^b psi)
   with a, b in {0,1}, i.e. the form is MULTIPLICATIVE on the enlarged vector

       L psi = ( psi , d_1 psi , d_2 psi )    (6 + 6 + 6 = 18 slots).

   The TRUE augmented dimension is however smaller: most of the 18 slots never
   appear.  It is counted below rather than assumed.  The thesis's 2.5-D count
   is 8 = 6 + 2; the package's 3-D augmented basis is 9.
   ============================================================================ *)

(* transfer rule: peel a LEFTMOST dOp off an entry, onto the phi side.
   Every term of DeltaA must match one of these shapes or we fail loudly. *)
ClearAll[splitTerm];
splitTerm[zeroOp] := {};
splitTerm[plusOp[ps__]] := Join @@ (splitTerm /@ {ps});
splitTerm[scalOp[c_, o_]] := ({#[[1]] c, #[[2]], #[[3]]} & /@ splitTerm[o]);
(* {coefficient, derivative index on phi (0 = none), derivative index on psi} *)
splitTerm[multOp[a_]] := {{a, 0, 0}};
splitTerm[compOp[dOp[al_], multOp[a_]]] := {{-a, al, 0}};
splitTerm[compOp[multOp[a_], dOp[be_]]] := {{a, 0, be}};
splitTerm[compOp[dOp[al_], multOp[a_], dOp[be_]]] := {{-a, al, be}};
splitTerm[x_] := (Print["  **** unhandled operator shape: ", x, " ****"];
   nFail++; {});

(* Build the 18x18 constant DeltaC from the transferred terms.
   slot[c, d] = 6 (d) + c, with d = 0,1,2 the derivative index. *)
slot[c_, d_] := 6 d + c;
dcMat = ConstantArray[0, {18, 18}];
Do[With[{terms = splitTerm[dAop[[k, j]]]},
   Do[With[{co = t[[1]], da = t[[2]], db = t[[3]]},
      Do[If[j6[[i, k]] =!= 0,
         dcMat[[slot[i, da], slot[j, db]]] +=
          j6[[i, k]] co], {i, 6}]], {t, terms}]],
 {k, 6}, {j, 6}];
dcMat = Simplify[dcMat];

laug[p_] := Join[p, D[p, x1], D[p, x2]];

phiv = Table[pp[i][x1, x2, x3], {i, 6}];
psiv = Table[ss[i][x1, x2, x3], {i, 6}];

(* the transferred bilinear, rebuilt from the same term list, independently *)
bTransferred = Sum[
   With[{terms = splitTerm[dAop[[k, j]]]},
    Sum[With[{co = t[[1]], da = t[[2]], db = t[[3]]},
      Sum[j6[[i, k]] co *
        (If[da === 0, phiv[[i]], D[phiv[[i]], xv[[da]]]]) *
        (If[db === 0, psiv[[j]], D[psiv[[j]], xv[[db]]]]), {i, 6}]],
     {t, terms}]], {k, 6}, {j, 6}];

bFromDC = Simplify[laug[phiv].dcMat.laug[psiv]];

Print[""];
Print["--- Part 4: the augmentation ---------------------------------------"];
report["DeltaC is a CONSTANT 18x18 matrix (no derivative operators left)",
  FreeQ[dcMat, Derivative | dOp | compOp | multOp]];
report["(L phi)^t DeltaC (L psi) reproduces the transferred bilinear",
  Simplify[bFromDC - bTransferred] === 0];

(* the untransferred bilinear differs from the transferred one by exactly a
   lateral divergence --- exhibited explicitly, not asserted *)
bRaw = Sum[j6[[i, k]] phiv[[i]] act[dAop[[k, j]], psiv[[j]]],
   {i, 6}, {k, 6}, {j, 6}];
pFlux = Table[
   Sum[With[{terms = splitTerm[dAop[[k, j]]]},
     Sum[With[{co = t[[1]], da = t[[2]], db = t[[3]]},
       If[da === al,
        -j6[[i, k]] co phiv[[i]] *
         (If[db === 0, psiv[[j]], D[psiv[[j]], xv[[db]]]]), 0]],
      {t, terms}]], {i, 6}, {k, 6}, {j, 6}], {al, 2}];
report["raw bilinear = transferred bilinear + d_a P_a, with P_a exhibited",
  Simplify[bRaw - bTransferred - Sum[D[pFlux[[al]], xv[[al]]], {al, 2}]] === 0];

(* the true augmented dimension: how many of the 18 slots are actually used *)
rowsUsed = Select[Range[18], Simplify[dcMat[[#, All]]] =!= ConstantArray[0, 18] &];
colsUsed = Select[Range[18], Simplify[dcMat[[All, #]]] =!= ConstantArray[0, 18] &];
Print["  augmented slots actually used:  rows ", Length[rowsUsed],
  "   cols ", Length[colsUsed], "   (of 18)"];
Print["  row slots  (component, derivative): ",
  {Mod[# - 1, 6] + 1, Quotient[# - 1, 6]} & /@ rowsUsed];
Print["  col slots  (component, derivative): ",
  {Mod[# - 1, 6] + 1, Quotient[# - 1, 6]} & /@ colsUsed];
Print["  thesis 2.5-D count for comparison: 8 = 6 + 2 (CdefAug)"];

report["the augmentation is a genuine enlargement (more than 6 slots used)",
  Length[colsUsed] > 6];

(* ============================================================================
   Part 5.  What actually separates the first-order closure from T9.

   T9 solves, with a UNIFORM-strain ansatz collocated at the cube centre,

       eps = eps0 + S : Dc : eps ,   S from the cube moments of dd G
       (effective_contrasts.py: _compute_T123 / _compute_amplification_factors)

   It is tempting to say the first-order route differs because DeltaA is
   nonlinear in Dc (Part 3).  The two checks below say something sharper.

   (a) THE TRIAL SPACES COINCIDE.  Inside the scatterer the moduli are constant,
       so tau_3 = C33_tot . eps_{i3} is a CONSTANT matrix times the strain.  A
       uniform strain therefore gives a uniform tau_3 and an affine v, and
       conversely.  "eps uniform" and "tau_3 uniform + in-plane eps uniform"
       describe the SAME 9-parameter family.  The two closures are therefore
       different PROJECTIONS of the same residual onto the same trial space --
       the difference lives entirely in the TEST conditions, not in the ansatz.

   (b) THE DENSITY CHANNEL CANNOT DIFFER.  The density contrast enters DeltaA
       only through A12, which sits in row-block 1 -- the d1 = f slot, a body
       force.  Nothing of it reaches the d2 (stress-glut) slot.  For the rigid
       translation mode the exact solution IS in the trial space and the test
       conditions coincide, so both routes must return the same A_u.

   Together these say: T9 tests with 9 strain/displacement conditions at the
   centre; the first-order route tests with 3 traction + 3 velocity + the 4
   augmented velocity-gradient slots found in Part 4 = 10.  That mismatch --
   10 test conditions for 9 unknowns -- is the open question, and it is a
   question about the formulation, not a number to be tuned.
   ============================================================================ *)

Print[""];
Print["--- Part 5: what separates the two closures -------------------------"];

(* (a) trial-space coincidence.  The claim is about the medium INSIDE the
   scatterer, which is uniform, so the constant-moduli substitution is part of
   the statement -- with the laterally varying background moduli left in, tau_3
   picks up an x-dependence and the check rightly fails. *)
insideRule = {lamf -> (lamCst &), muf -> (muCst &), rhof -> (rhoCst &)};
epsSym = Table[ee[Min[i, j], Max[i, j]], {i, 3}, {j, 3}];   (* uniform, symmetric *)
tau3Uniform = Simplify[
   Table[(lam + dlam) Sum[epsSym[[k, k]], {k, 3}] dl[ii, 3] + 2 (mu + dmu) epsSym[[ii, 3]],
     {ii, 3}] /. insideRule];
report["tau_3 is a CONSTANT linear map of a uniform strain (trial spaces coincide)",
  Simplify[Table[D[tau3Uniform[[ii]], xs[[m]]], {ii, 3}, {m, 3}]] ===
   ConstantArray[0, {3, 3}]];
report["that map is invertible (so the correspondence is one-to-one)",
  Simplify[Det[Table[D[tau3Uniform[[ii]], epsSym[[m, 3]]], {ii, 3}, {m, 3}]]] =!= 0];

(* (b) the density contrast reaches only the body-force slot *)
dAdrho = Simplify[dAop /. {dmu -> 0, dlam -> 0}];
densityRows = Table[
   Simplify[act[dAdrho[[i, j]], phd]], {i, 6}, {j, 6}];
report["with Dmu = Dlam = 0, DeltaA is exactly i w Drho in rows 1-3, cols 4-6",
  Simplify[densityRows -
     Join[Join[ConstantArray[0, {3, 3}], I w drho phd IdentityMatrix[3], 2],
      ConstantArray[0, {3, 6}]]] === ConstantArray[0, {6, 6}]];
report["the density contrast reaches NO part of the d2 (stress-glut) slot",
  Simplify[densityRows[[4 ;; 6, All]]] === ConstantArray[0, {3, 6}]];

Print["  => A_u = 1/(1 - w^2 Drho Gamma0) in BOTH routes: the density source is"];
Print["     a body force, Gamma's f-column velocity block is -i w G, and the"];
Print["     rigid-translation mode lies exactly in the trial space."];
Print["  => the modulus channels are where they can differ, and the open"];
Print["     question is WHICH test conditions, not which number."];

(* ============================================================================
   Part 6.  The test space, DERIVED from the J6 pairing.

   Part 5 left 10 augmented slots against 9 trial parameters and no principled
   way to choose.  The choice is not free, and it is not a subset:

       DeltaA^t J6 = -J6 DeltaA        (Part 2)
   =>  (J6 DeltaA)^t = DeltaA^t J6^t = DeltaA^t (-J6) = J6 DeltaA

   because J6 is ANTIsymmetric.  So J6 DeltaA is a SYMMETRIC matrix and

       b[phi,psi] = phi^t J6 DeltaA psi

   is a symmetric bilinear form.  A symmetric form has exactly one natural
   projection: test with the image of the trial space under J6.  There is no
   subset to pick and nothing to tune.

   This also dissolves the 10-vs-9 count.  The 10 slots measure how many
   derivative channels the FORM touches; they are its representation, not a
   list of independent test conditions.  Each trial function psi_k supplies
   exactly one test functional (J6 psi_k)^t, so the number of conditions is the
   trial dimension, 9, by construction.

   What comes out is the 3-D Cartesian analogue of the thesis's effective
   coupling matrix DeltaC_eff (TRepresentations.tex, dCeff, L2214) for the
   affine trial space -- and it needs NO Green's function: the trial functions
   are polynomials, so the cube integral is exact.
   ============================================================================ *)

Print[""];
Print["--- Part 6: the test space from the J6 pairing ----------------------"];

(* rebuild the augmented matrix for an arbitrary metric, so the J6 choice can
   be contrasted against the naive one *)
buildDC[metric_] := Module[{m = ConstantArray[0, {18, 18}]},
   Do[With[{terms = splitTerm[dAop[[k, j]]]},
      Do[With[{co = t[[1]], da = t[[2]], db = t[[3]]},
         Do[If[metric[[i, k]] =!= 0,
            m[[slot[i, da], slot[j, db]]] += metric[[i, k]] co], {i, 6}]],
       {t, terms}]], {k, 6}, {j, 6}];
   Simplify[m]];

dcJ = Simplify[buildDC[j6] /. insideRule];
dcI = Simplify[buildDC[IdentityMatrix[6]] /. insideRule];

report["J6 DeltaA is SYMMETRIC (so b[phi,psi] = b[psi,phi])",
  Simplify[dcJ - Transpose[dcJ]] === ConstantArray[0, {18, 18}]];
report["NEGATIVE CONTROL: without J6 the form is NOT symmetric",
  Simplify[dcI - Transpose[dcI]] =!= ConstantArray[0, {18, 18}]];

(* the 9 affine trial functions, and the q-field each one carries *)
pars = {c1, c2, c3, e11, e22, e33, e12, e13, e23};
epsM = {{e11, e12, e13}, {e12, e22, e23}, {e13, e23, e33}};
uGen = {c1, c2, c3} + epsM.{x1, x2, x3};

lamIn = lamCst + dlam; muIn = muCst + dmu;
epsOf[uu_] := Table[(D[uu[[i]], xs[[j]]] + D[uu[[j]], xs[[i]]])/2, {i, 3}, {j, 3}];
qOf[uu_] := Module[{ep = epsOf[uu]},
   Join[Table[-(lamIn Sum[ep[[k, k]], {k, 3}] dl[ii, 3] + 2 muIn ep[[ii, 3]]), {ii, 3}],
    -I w uu]];

trialQ = Table[qOf[D[uGen, pars[[k]]]], {k, 9}];

(* each q-component is affine, so {const, d1, d2, d3} is a complete coordinate;
   CoefficientList is ragged here and cannot be used *)
linCoef[e_] := {e /. {x1 -> 0, x2 -> 0, x3 -> 0}, D[e, x1], D[e, x2], D[e, x3]};
report["the 9 trial q-fields are linearly independent",
  MatrixRank[Table[Flatten[linCoef /@ trialQ[[k]]], {k, 9}]] === 9];

laug[p_] := Join[p, D[p, x1], D[p, x2]];
cubeInt[e_] := Integrate[e, {x1, -aa, aa}, {x2, -aa, aa}, {x3, -aa, aa}];

dCeff = Simplify@Table[
    cubeInt[laug[trialQ[[k]]].dcJ.laug[trialQ[[l]]]], {k, 9}, {l, 9}];
dCeffNoJ = Simplify@Table[
    cubeInt[laug[trialQ[[k]]].dcI.laug[trialQ[[l]]]], {k, 9}, {l, 9}];

report["the projected 9x9 DeltaC_eff is SYMMETRIC",
  Simplify[dCeff - Transpose[dCeff]] === ConstantArray[0, {9, 9}]];
report["NEGATIVE CONTROL: the un-paired 9x9 is NOT symmetric",
  Simplify[dCeffNoJ - Transpose[dCeffNoJ]] =!= ConstantArray[0, {9, 9}]];
report["DeltaC_eff vanishes with the contrast (no spurious coupling)",
  Simplify[dCeff /. {dlam -> 0, dmu -> 0, drho -> 0}] === ConstantArray[0, {9, 9}]];

Print["  rank of the projected 9x9 coupling: ",
  MatrixRank[dCeff /. {lamCst -> 17.5, muCst -> 22.5, rhoCst -> 2.5,
     dlam -> 2.0, dmu -> 1.0, drho -> 0.1, w -> 60.0, aa -> 1.0}]];
Print["  rigid-translation block DeltaC_eff[[1;;3, 1;;3]] ="];
Scan[Print["     ", #] &, Simplify[dCeff[[1 ;; 3, 1 ;; 3]]]];

(* The rigid block is the pure density coupling, and its power of omega is a
   statement about the BASIS, not about the physics.  DeltaA's density entry is
   i w Drho acting on the velocity slot; the trial carries v = -i w u there, and
   J6 pulls the test function's velocity slot into the traction rows, supplying a
   second -i w.  Hence i w * (-i w) * (-i w) = i w^3, not i w.  Writing i w Drho V
   here -- the displacement-basis answer -- is exactly the displacement-vs-flux
   slip this repository has paid for before, so the factor is asserted. *)
report["rigid-translation block is the density coupling i w^3 Drho V, V = (2a)^3",
  Simplify[dCeff[[1 ;; 3, 1 ;; 3]] - I w^3 drho (2 aa)^3 IdentityMatrix[3]] ===
   ConstantArray[0, {3, 3}]];
report["the displacement-basis factor i w Drho V is NOT what comes out",
  Simplify[dCeff[[1 ;; 3, 1 ;; 3]] - I w drho (2 aa)^3 IdentityMatrix[3]] =!=
   ConstantArray[0, {3, 3}]];

(* ============================================================================
   Part 7.  Is the first-order coupling anything NEW?

   Before assembling propagator moments and scoring against Kennett, one
   question has to be answered, and it costs no Green's function at all.

   The first-order Galerkin scheme projects onto the trial space with the form
   b_J[phi,psi] = phi^t J6 DeltaA psi.  The ordinary Navier weak form projects
   the SAME residual onto the SAME trial space (Part 5) with

       b_N[phi,u] = Int_V [ w^2 Drho phi.u  -  eps(phi) : Dc : eps(u) ] dV

   -- the standard weak contrast, the minus sign coming from moving d_j off the
   stress.  Both are symmetric forms on the same 9-dimensional space.  If they
   are PROPORTIONAL then the two Galerkin schemes are identical up to a scalar
   that cancels in T = DeltaC (I - G DeltaC)^-1, and the first-order route buys
   nothing for the single site -- whatever it is worth would live entirely in
   the propagator, i.e. in a LAYERED background.

   The proportionality constant is not guessed: it is read off the rigid block
   and then tested on all 81 entries.  A constant fitted on one entry and
   confirmed on the other 80 is a prediction; a constant fitted on all 81 would
   be curve-fitting and would prove nothing.
   ============================================================================ *)

Print[""];
Print["--- Part 7: is the first-order coupling anything new? ---------------"];

epsOfU[uu_] := Table[(D[uu[[i]], xs[[j]]] + D[uu[[j]], xs[[i]]])/2, {i, 3}, {j, 3}];
dTauOf[uu_] := Module[{ep = epsOfU[uu]},
   Table[dlam Sum[ep[[k, k]], {k, 3}] dl[i, j] + 2 dmu ep[[i, j]], {i, 3}, {j, 3}]];
bNav[ph_, uu_] := cubeInt[
   w^2 drho ph.uu - Sum[epsOfU[ph][[i, j]] dTauOf[uu][[i, j]], {i, 3}, {j, 3}]];

trialU9 = Table[D[uGen, pars[[k]]], {k, 9}];
dCnav = Simplify@Table[bNav[trialU9[[k]], trialU9[[l]]], {k, 9}, {l, 9}];

report["the Navier weak form on the same trial space is symmetric too",
  Simplify[dCnav - Transpose[dCnav]] === ConstantArray[0, {9, 9}]];

(* read the constant off ONE entry -- the rigid block, which carries only the
   density channel -- then test it everywhere else *)
cFromRigid = Simplify[dCeff[[1, 1]]/dCnav[[1, 1]]];
Print["  constant read off the rigid block:  DeltaC_eff/DeltaC_Navier = ", cFromRigid];

report["that constant is exactly i w",
  Simplify[cFromRigid - I w] === 0];

(* a control that the test can fail: the modulus channels must be non-trivial,
   otherwise the 80 entries would be vacuously matched *)
report["the modulus channels are non-trivial (any match is not vacuous)",
  Simplify[dCnav /. {drho -> 0}] =!= ConstantArray[0, {9, 9}] &&
   MatrixRank[dCnav /. {lamCst -> 17.5, muCst -> 22.5, rhoCst -> 2.5, drho -> 0,
      dlam -> 2.0, dmu -> 1.0, w -> 60.0, aa -> 1.0}] >= 6];

(* THE PREDICTION WAS REFUTED.  The two couplings are not proportional, so the
   first-order form carries content the Navier weak form does not.  What
   follows locates that content rather than just recording the refutation. *)
dDiff = Simplify[dCeff - I w dCnav];

report["REFUTED: DeltaC_eff is NOT i w * DeltaC_Navier",
  dDiff =!= ConstantArray[0, {9, 9}]];
report["the difference is absent from the density channel",
  Simplify[dDiff /. {dlam -> 0, dmu -> 0}] === ConstantArray[0, {9, 9}]];

(* Where does it sit, and at what order in the contrast?  If the difference is
   the C33^-1 resummation of Part 3 then it must vanish when the contrasts are
   linearised -- the two forms would then agree at Born order and part company
   only at O(Dc^2).  That is a sharp prediction, not a description. *)
dScaled = Simplify[dDiff /. {dlam -> tt dlam, dmu -> tt dmu, drho -> tt drho}];
report["the two couplings agree at O(Dc^0)",
  Simplify[dScaled /. tt -> 0] === ConstantArray[0, {9, 9}]];
report["the two couplings agree at O(Dc^1): identical Born limits",
  Simplify[Coefficient[Normal@Series[dScaled, {tt, 0, 1}], tt, 1]] ===
   ConstantArray[0, {9, 9}]];
report["they part company at O(Dc^2) -- the C33^-1 resummation of Part 3",
  Simplify[Coefficient[Normal@Series[dScaled, {tt, 0, 2}], tt, 2]] =!=
   ConstantArray[0, {9, 9}]];

nzPos = Position[Simplify[dDiff], _?(Simplify[#] =!= 0 &), {2}, Heads -> False];
Print["  nonzero entries of the difference: ", Length[nzPos], " of 81, at ",
  Short[nzPos, 4]];
If[Length[nzPos] > 0,
  Print["  representative entry ", First[nzPos], " = ",
   Simplify[dDiff[[Sequence @@ First[nzPos]]]]]];

(* WHICH parameters?  pars = {c1,c2,c3, e11,e22,e33, e12,e13,e23}, so slots
   4,5,6 are the normal strains and 7,8,9 the shears e12, e13, e23.  If the
   difference really is the C33 resummation it must touch exactly the strain
   components that tau_3 = (tau_13, tau_23, tau_33) sees -- e33, e13, e23, plus
   e11 and e22 through lambda -- and must LEAVE e12 ALONE, because tau_12 was
   eliminated, not resummed.  That is a falsifiable statement about which
   entries may be nonzero. *)
report["e12 (the in-plane shear) is UNTOUCHED: the two forms agree there exactly",
  Simplify[dDiff[[7, All]]] === ConstantArray[0, 9] &&
   Simplify[dDiff[[All, 7]]] === ConstantArray[0, 9]];
report["the difference touches only the traction channel {e11,e22,e33,e13,e23}",
  Complement[Union[Flatten[nzPos]], {4, 5, 6, 8, 9}] === {}];

(* How big is it?  A difference that no arbiter can resolve is not worth a
   propagator build. *)
sizeAt[dl_, dm_, dr_] := Module[{rule},
   rule = {lamCst -> 17.5, muCst -> 22.5, rhoCst -> 2.5, w -> 60.0, aa -> 1.0,
     dlam -> dl, dmu -> dm, drho -> dr};
   {Norm[Flatten[dDiff /. rule]]/Norm[Flatten[dCeff /. rule]],
    Norm[Flatten[dDiff /. rule]]/Norm[Flatten[I w dCnav /. rule]]}];

Print["  relative size ||DeltaC_eff - i w DeltaC_Nav|| / ||DeltaC_eff||:"];
Print["     moderate  (Dlam=2, Dmu=1, Drho=0.1 GPa):  ", sizeAt[2.0, 1.0, 0.1][[1]]];
Print["     pure shear(Dlam=0, Dmu=1):                ", sizeAt[0.0, 1.0, 0.0][[1]]];
Print["     strong    (Dlam=20, Dmu=10, Drho=0.5):    ", sizeAt[20.0, 10.0, 0.5][[1]]];

(* ============================================================================
   Part 8.  Does the first-order coupling keep the cube's O_h symmetry?

   This has to be asked before any propagator is assembled, because it is a
   property of the coupling alone and it can disqualify the scheme outright.

   An isotropic cube in an isotropic background has a T-matrix with the full
   cubic symmetry O_h -- no direction is special.  But the first-order system
   singles out x3 by construction: it is a recursion in depth, and tau_1, tau_2
   were eliminated while tau_3 was kept.  If that preference survives into the
   coupling then the closure breaks a symmetry the exact answer has, and no
   choice of propagator can put it back.

   The tell is already visible in Part 7: the difference touches {e13,e13} and
   {e23,e23} but NOT {e12,e12}.  Those three are equivalent under O_h.  The
   check below makes that decisive by applying the cyclic coordinate
   permutation x->y->z->x, an element of O_h, to the whole 9x9:

     (c1,c2,c3) -> (c2,c3,c1)
     (e11,e22,e33,e12,e13,e23) -> (e22,e33,e11,e23,e12,e13)

   The Navier coupling MUST be invariant (isotropic Dc on a cube).  That is the
   positive control: if it were not, the test itself would be wrong.
   ============================================================================ *)

Print[""];
Print["--- Part 8: does the coupling keep the cube's O_h symmetry? ---------"];

permIdx = {2, 3, 1, 5, 6, 4, 9, 7, 8};   (* new_k = old_{permIdx[k]} *)
permMat = Table[If[j === permIdx[[i]], 1, 0], {i, 9}, {j, 9}];

report["POSITIVE CONTROL: the permutation really is an O_h element (P^3 = I)",
  permMat.permMat.permMat === IdentityMatrix[9]];
report["POSITIVE CONTROL: the Navier coupling IS O_h invariant",
  Simplify[permMat.dCnav.Transpose[permMat] - dCnav] === ConstantArray[0, {9, 9}]];

ohResidual = Simplify[permMat.dCeff.Transpose[permMat] - dCeff];
report["the FIRST-ORDER coupling BREAKS O_h invariance",
  ohResidual =!= ConstantArray[0, {9, 9}]];
report["the O_h breaking is absent at Born order (it is O(Dc^2))",
  Simplify[Coefficient[Normal@Series[
      ohResidual /. {dlam -> tt dlam, dmu -> tt dmu, drho -> tt drho},
      {tt, 0, 1}], tt, 1]] === ConstantArray[0, {9, 9}]];

Print["  size of the O_h breaking, ||P C P^t - C|| / ||C||:"];
Do[With[{rule = {lamCst -> 17.5, muCst -> 22.5, rhoCst -> 2.5, w -> 60.0, aa -> 1.0,
     dlam -> cs[[2]], dmu -> cs[[3]], drho -> cs[[4]]}},
   Print["     ", cs[[1]], "  ",
    Norm[Flatten[ohResidual /. rule]]/Norm[Flatten[dCeff /. rule]]]],
 {cs, {{"pure shear (Dmu=1)      ", 0.0, 1.0, 0.0},
       {"moderate (2, 1, 0.1)    ", 2.0, 1.0, 0.1},
       {"strong   (20, 10, 0.5)  ", 20.0, 10.0, 0.5}}}];

(* The O_h breaking and the whole Part-7 difference are the same size.  Are they
   the same THING?  Average the coupling over the cyclic subgroup -- the
   projection onto the O_h-invariant part -- and compare with the Navier form.
   If the symmetrised coupling IS the Navier coupling, then every bit of the
   "new content" found in Part 7 was the preferred direction, and there is
   nothing else in it. *)
dCsym = Simplify[(dCeff + permMat.dCeff.Transpose[permMat] +
      permMat.permMat.dCeff.Transpose[permMat.permMat])/3];

(* The symmetrised coupling is NOT simply i w * Navier, so the difference is not
   pure artefact.  Split it instead: dDiff = (invariant part) + (breaking part).
   The invariant part is genuine new content -- a resummation the Navier form
   does not have.  The breaking part is the x3 preference, which the exact
   answer does not have.  Both are measured; neither is asserted.
   NOTE the group used here is the cyclic subgroup of order 3, not all of O_h,
   so "invariant" below means invariant under that subgroup -- a necessary
   condition for O_h invariance, not a sufficient one. *)
dSymPart = Simplify[dCsym - I w dCnav];
dBreakPart = Simplify[dDiff - dSymPart];

report["there IS a genuine O_h-invariant residual (not pure artefact)",
  dSymPart =!= ConstantArray[0, {9, 9}]];
report["there IS a symmetry-breaking part (not pure physics either)",
  dBreakPart =!= ConstantArray[0, {9, 9}]];
report["the split is exact: invariant + breaking = the whole difference",
  Simplify[dSymPart + dBreakPart - dDiff] === ConstantArray[0, {9, 9}]];

Print["  decomposition of (DeltaC_eff - i w DeltaC_Nav), relative to ||DeltaC_eff||:"];
Print["     contrast                  invariant     breaking"];
Do[With[{rule = {lamCst -> 17.5, muCst -> 22.5, rhoCst -> 2.5, w -> 60.0, aa -> 1.0,
     dlam -> cs[[2]], dmu -> cs[[3]], drho -> cs[[4]]}},
   Print["     ", cs[[1]],
    "  ", Norm[Flatten[dSymPart /. rule]]/Norm[Flatten[dCeff /. rule]],
    "  ", Norm[Flatten[dBreakPart /. rule]]/Norm[Flatten[dCeff /. rule]]]],
 {cs, {{"pure shear (Dmu=1)      ", 0.0, 1.0, 0.0},
       {"moderate (2, 1, 0.1)    ", 2.0, 1.0, 0.1},
       {"strong   (20, 10, 0.5)  ", 20.0, 10.0, 0.5}}}];

Print[""];
Print["  WHAT THIS MEANS -- it qualifies the Part-7 reading rather than"];
Print["  confirming it.  The first-order coupling carries BOTH:"];
Print["    * a genuine O_h-invariant resummation the Navier form lacks, and"];
Print["    * a spurious x3 preference, since the first-order system is a"];
Print["      recursion in depth and eliminated tau_1, tau_2 but kept tau_3."];
Print["  An isotropic cube in an isotropic whole space has an O_h-symmetric"];
Print["  T-matrix, so the second part is an artefact there and NO propagator"];
Print["  can remove it.  Scoring the raw coupling against Kennett in a"];
Print["  homogeneous background would mix the two and settle neither."];
Print["  => the clean experiment is the SYMMETRISED coupling vs the Navier one;"];
Print["     that isolates the resummation from the artefact."];
Print["  => and the x3 preference stops being an artefact once the background"];
Print["     is layered, which is where this route was wanted all along."];

Print[""];
Print[bar];
Print["  ", nPass, " passed, ", nFail, " failed"];
Print[bar];
If[nFail > 0, Exit[1]];
