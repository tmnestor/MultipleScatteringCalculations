#!/usr/bin/env wolframscript
(* ============================================================================
   The elastodynamic matrix-vector wave equation  d3 q = A q + d

   Derives the isotropic first-order system in x3 from the stiffness tensor
   alone, and validates it against two independent objects.

   Source of the general derivation:
     Wapenaar, "Unified matrix-vector wave equation, reciprocity and
     representations", Appendix D --- equations (D.17), (D.20)-(D.23).
   Source of the isotropic target:
     same paper, supplemental Appendix J --- equations (J.1)-(J.10).

   NOTHING in Part 3 is transcribed from Appendix J.  A11, A12, A21, A22 are
   built from c_ijkl = lam d_ij d_kl + mu (d_ik d_jl + d_il d_jk) by the
   general Appendix-D formulae, and then COMPARED against J.5-J.8, which are
   transcribed separately in Part 4 purely as a target.

   Two independent arbiters, per the project's evidence standard:
     (A) Appendix J of the paper           -- Part 4
     (B) the thesis system matrix A(kx,ky) -- Part 7
         Thesis_Recompiled_2026/GRepresentations.tex, eq. (Akdef), an
         independently derived displacement-stress form in a different field
         ordering and a different normalisation.  The two must be related by
         one constant similarity transform.  This is the far-field-limit
         style check: an object the derivation could not have been fitted to.

   Coordinates.  The paper's x3 is depth (positive down); x1, x2 are lateral.
   The thesis orders its 6-vector (z,x,y) = (x3,x1,x2).  Part 7 carries the
   permutation explicitly; do not assume the orderings agree.

   Time convention e^{-iwt} in both sources, so d/dt -> -i w and v = -i w u.
   ============================================================================ *)

(* Built as a literal so the string carries no long run of '=' in the source:
   the notebook generator uses such a run to mark a text-cell boundary. *)
bar = StringJoin @@ ConstantArray["=", 60];

Print[bar];
Print["  d3 q = A q + d   ---   isotropic elastodynamic first-order system"];
Print[bar];

(* ============================================================================
   Part 0.  A minimal symbolic algebra of differential operators.

   Every entry of A is a composition of multiplications by x-dependent medium
   parameters and lateral derivatives.  The ORDER matters and is the whole
   content of the derivation: in A11 the derivative stands to the LEFT of the
   stiffness and so differentiates the product, while in A22 it stands to the
   RIGHT and differentiates only the field.  Representing entries as bare
   expressions would silently destroy that distinction, so operators are kept
   as expression trees with an explicit action and an explicit transpose.

     multOp[a]      f -> a f
     dOp[alpha]     f -> d_alpha f          (alpha in {1,2}, lateral only)
     compOp[A,B]    f -> A[B[f]]
     plusOp[...]    f -> sum of actions
     scalOp[c,A]    f -> c A[f]

   Operator transposition (the ^t of the paper's eqs (16)-(18)) is the formal
   adjoint under the bilinear form int g^t A f dx, i.e. integration by parts
   with vanishing boundary terms:  d_alpha^t = -d_alpha, mult^t = mult, and
   composition reverses.
   ============================================================================ *)

xv = {x1, x2};

act[zeroOp, f_] := 0;
act[multOp[a_], f_] := a f;
act[dOp[al_], f_] := D[f, xv[[al]]];
act[compOp[ps__], f_] := Fold[act[#2, #1] &, f, Reverse[{ps}]];
act[plusOp[ps__], f_] := Total[act[#, f] & /@ {ps}];
act[scalOp[c_, o_], f_] := c act[o, f];

opT[zeroOp] := zeroOp;
opT[multOp[a_]] := multOp[a];
opT[dOp[al_]] := scalOp[-1, dOp[al]];
opT[compOp[ps__]] := compOp @@ Reverse[opT /@ {ps}];
opT[plusOp[ps__]] := plusOp @@ (opT /@ {ps});
opT[scalOp[c_, o_]] := scalOp[c, opT[o]];

(* matrix of operators acting on a vector of fields *)
opMatAct[m_, q_] :=
  Table[Sum[act[m[[i, j]], q[[j]]], {j, Length[q]}], {i, Length[m]}];

(* operator-matrix transpose: transpose the array AND each entry *)
opMatT[m_] := Transpose[Map[opT, m, {2}]];

opMatPlus[m1_, m2_] := MapThread[plusOp, {m1, m2}, 2];
opMatScal[c_, m_] := Map[scalOp[c, #] &, m, {2}];

(* Equality of two operator matrices is decided by their action on a vector of
   GENERIC functions.  Generic means the comparison is an operator identity,
   not a coincidence at some particular field. *)
genericVec[n_, tag_] :=
  Table[Symbol[tag <> ToString[i]][x1, x2, x3], {i, n}];

opMatZeroQ[m_, n_] := Module[{q = genericVec[n, "phi"]},
   Simplify[opMatAct[m, q]] === ConstantArray[0, n]];

opMatEqualQ[m1_, m2_, n_] :=
  opMatZeroQ[opMatPlus[m1, opMatScal[-1, m2]], n];

(* Display.  MatrixForm does not render under `wolframscript -file`, so rows
   are printed one per line, with the x-arguments and Derivative heads folded
   into short names: muf[x1,x2,x3] -> mu, Derivative[1,0,0][muf][...] -> D10mu.
   Inside the notebook the same expressions render as ordinary output. *)
(* Display names must be UNBOUND symbols.  "lam", "mu", "rho" are already
   bound below to the field expressions, so Symbol["mu"] would evaluate
   straight back to muf[x1,x2,x3] and the shortening would be a no-op. *)
nameOf[lamf] = "LAM"; nameOf[muf] = "MU"; nameOf[rhof] = "RHO";
nameOf[psi] = "PSI";
nameOf[ff[i_]] := "f" <> ToString[i];
nameOf[hh[i_, j_]] := "h" <> ToString[i] <> ToString[j];

shortRule = {
   Derivative[m_, n_, 0][f_][x1, x2, x3] /; StringQ[nameOf[f]] :>
    Symbol["D" <> ToString[m] <> ToString[n] <> nameOf[f]],
   f_[x1, x2, x3] /; StringQ[nameOf[f]] :> Symbol[nameOf[f]]};

tidy[e_] := e /. shortRule;
disp[label_, m_] := (Print[label]; Scan[Print["     ", tidy[#]] &, m]);

(* show each operator entry by its action on a placeholder field psi *)
opShow[m_] := Map[Simplify[act[#, psi[x1, x2, x3]]] &, m, {2}];

nPass = 0; nFail = 0;
report[label_, bool_] := (
   If[TrueQ[bool], nPass++, nFail++];
   Print[If[TrueQ[bool], "  PASS  ", "  ****FAIL****  "], label]);

(* ============================================================================
   Part 1.  The isotropic stiffness matrices C_jl.

   c_ijkl = lam d_ij d_kl + mu (d_ik d_jl + d_il d_jk)                   (J.1)
   rho_ij = rho d_ij                                                     (J.2)
   (C_jl)_ik = c_ijkl                          [paper, Appendix D, below D.12]

   The medium parameters are functions of x, so lateral derivatives act on
   them.  This is the whole difference between the operator form and the
   laterally invariant form the thesis carries.
   ============================================================================ *)

lam = lamf[x1, x2, x3];
mu = muf[x1, x2, x3];
rho = rhof[x1, x2, x3];
kc = lam + 2 mu;                                  (* Kc = lam + 2 mu *)

dl[i_, j_] := KroneckerDelta[i, j];
cc[i_, j_, k_, l_] := lam dl[i, j] dl[k, l] + mu (dl[i, k] dl[j, l] + dl[i, l] dl[j, k]);

cmat[j_, l_] := Table[cc[ii, j, kk, l], {ii, 3}, {kk, 3}];

Print[""];
Print["--- Part 1: stiffness matrices C_jl -------------------------------"];
disp["C11 =", cmat[1, 1]];
disp["C13 =", cmat[1, 3]];
disp["C33 =", cmat[3, 3]];

(* J.3 / J.4 transcribed as a target for C_jl only *)
cRefTab = {
   {1, 1, {{kc, 0, 0}, {0, mu, 0}, {0, 0, mu}}},
   {1, 2, {{0, lam, 0}, {mu, 0, 0}, {0, 0, 0}}},
   {1, 3, {{0, 0, lam}, {0, 0, 0}, {mu, 0, 0}}},
   {2, 1, {{0, mu, 0}, {lam, 0, 0}, {0, 0, 0}}},
   {2, 2, {{mu, 0, 0}, {0, kc, 0}, {0, 0, mu}}},
   {2, 3, {{0, 0, 0}, {0, 0, lam}, {0, mu, 0}}},
   {3, 1, {{0, 0, mu}, {0, 0, 0}, {lam, 0, 0}}},
   {3, 2, {{0, 0, 0}, {0, 0, mu}, {0, lam, 0}}},
   {3, 3, {{mu, 0, 0}, {0, mu, 0}, {0, 0, kc}}}};

report["C_jl from c_ijkl reproduces (J.3)-(J.4) for all 9 blocks",
  And @@ (Simplify[cmat[#[[1]], #[[2]]] - #[[3]]] === ConstantArray[0, {3, 3}] & /@ cRefTab)];

report["C_jl^t = C_lj  (required by c_ijkl = c_klij)",
  And @@ Flatten@Table[
     Simplify[Transpose[cmat[j, l]] - cmat[l, j]] === ConstantArray[0, {3, 3}],
     {j, 3}, {l, 3}]];

c33inv = Simplify@Inverse[cmat[3, 3]];
disp["C33^{-1} =", c33inv];

(* ============================================================================
   Part 2.  Eliminating the in-plane tractions: the reduced stiffness U.

   U_al = C_al - C_a3 C33^{-1} C_3l                                     (D.17)

   U is what survives after tau_1 and tau_2 have been removed using the
   x3-derivative relation (D.14).  Its two structural properties, U_a3 = O and
   U_ab^t = U_ba, are what make the symmetry relations (16)-(18) hold, so they
   are checked, not assumed.  The isotropic U_ab is where nu1 and nu2 are
   BORN --- they are not independent definitions.
   ============================================================================ *)

umat[al_, l_] := Simplify[cmat[al, l] - cmat[al, 3].c33inv.cmat[3, l]];

Print[""];
Print["--- Part 2: reduced stiffness U_al = C_al - C_a3 C33^-1 C_3l ------"];
disp["U11 =", umat[1, 1]];
disp["U12 =", umat[1, 2]];
disp["U21 =", umat[2, 1]];
disp["U22 =", umat[2, 2]];

report["U_a3 = O for a = 1,2  (D.17)",
  And @@ Table[Simplify[umat[al, 3]] === ConstantArray[0, {3, 3}], {al, 2}]];

report["U_ab^t = U_ba  (D.17)",
  And @@ Flatten@Table[
     Simplify[Transpose[umat[al, be]] - umat[be, al]] === ConstantArray[0, {3, 3}],
     {al, 2}, {be, 2}]];

report["U_ab has a null third row and column (tau_33 fully eliminated)",
  And @@ Flatten@Table[
     {Simplify[umat[al, be][[3, All]]] === {0, 0, 0},
      Simplify[umat[al, be][[All, 3]]] === {0, 0, 0}},
     {al, 2}, {be, 2}]];

(* nu1, nu2 read off U, then compared with (J.9)-(J.10) *)
nu1 = Simplify[umat[1, 1][[1, 1]]];
nu2 = Simplify[umat[1, 2][[1, 2]]];
Print["nu1 read off U11[[1,1]] = ", tidy[nu1]];
Print["nu2 read off U12[[1,2]] = ", tidy[nu2]];
report["nu1 = 4 mu (lam+mu)/(lam+2mu)   (J.9)",
  Simplify[nu1 - 4 mu (lam + mu)/kc] === 0];
report["nu2 = 2 mu lam/(lam+2mu)        (J.10)",
  Simplify[nu2 - 2 mu lam/kc] === 0];
report["nu1 - nu2 = 2 mu  (thesis footnote: zeta - chi = 2 mu)",
  Simplify[nu1 - nu2 - 2 mu] === 0];

(* ============================================================================
   Part 3.  The operator matrices, built from the general Appendix-D formulae.

   A11 = -d_a C_a3 C33^{-1}                                             (D.20)
   A12 =  i w rho - (1/(i w)) d_a U_ab d_b                              (D.21)
   A21 =  i w C33^{-1}                                                  (D.22)
   A22 = -C33^{-1} C_3b d_b                                             (D.23)

   Sum over repeated Greek indices runs over 1,2 only.

   Read the operator ordering carefully.  In (D.20) d_a stands to the LEFT of
   the stiffness, so it differentiates the product (C_a3 C33^{-1}) f.  In
   (D.23) d_b stands to the RIGHT, so it differentiates f alone.  In (D.21)
   the middle factor U_ab sits BETWEEN the two derivatives.  For a laterally
   invariant medium these distinctions collapse; for a laterally varying one
   they are the physics.
   ============================================================================ *)

a11op = Table[
   plusOp @@ Table[
     scalOp[-1, compOp[dOp[al], multOp[(cmat[al, 3].c33inv)[[ii, kk]]]]],
     {al, 2}],
   {ii, 3}, {kk, 3}];

a12op = Table[
   plusOp[
    multOp[I w rho dl[ii, kk]],
    Sequence @@ Flatten[
      Table[scalOp[-1/(I w),
        compOp[dOp[al], multOp[umat[al, be][[ii, kk]]], dOp[be]]],
       {al, 2}, {be, 2}], 1]],
   {ii, 3}, {kk, 3}];

a21op = Table[multOp[I w c33inv[[ii, kk]]], {ii, 3}, {kk, 3}];

a22op = Table[
   plusOp @@ Table[
     scalOp[-1, compOp[multOp[(c33inv.cmat[3, be])[[ii, kk]]], dOp[be]]],
     {be, 2}],
   {ii, 3}, {kk, 3}];

Print[""];
Print["--- Part 3: operator matrices from (D.20)-(D.23), acting on psi ---"];
disp["A11 psi =", opShow[a11op]];
disp["A21 psi =", opShow[a21op]];
disp["A22 psi =", opShow[a22op]];
disp["A12 psi =", opShow[a12op]];

(* ============================================================================
   Part 4.  ARBITER (A): Appendix J of the paper.

   J.5-J.8 are transcribed here as a target and nowhere else.  The comparison
   is of operators, decided by action on generic fields, so it tests the
   derivative ORDERING as well as the coefficients.
   ============================================================================ *)

gam = lam/kc;

a11ref = {
   {zeroOp, zeroOp, scalOp[-1, compOp[dOp[1], multOp[gam]]]},
   {zeroOp, zeroOp, scalOp[-1, compOp[dOp[2], multOp[gam]]]},
   {scalOp[-1, dOp[1]], scalOp[-1, dOp[2]], zeroOp}};

a12ref = {
   {plusOp[multOp[I w rho],
     scalOp[-1/(I w), compOp[dOp[1], multOp[nu1], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[2], multOp[mu], dOp[2]]]],
    plusOp[
     scalOp[-1/(I w), compOp[dOp[2], multOp[mu], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[1], multOp[nu2], dOp[2]]]],
    zeroOp},
   {plusOp[
     scalOp[-1/(I w), compOp[dOp[2], multOp[nu2], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[1], multOp[mu], dOp[2]]]],
    plusOp[multOp[I w rho],
     scalOp[-1/(I w), compOp[dOp[1], multOp[mu], dOp[1]]],
     scalOp[-1/(I w), compOp[dOp[2], multOp[nu1], dOp[2]]]],
    zeroOp},
   {zeroOp, zeroOp, multOp[I w rho]}};

a21ref = {
   {multOp[I w/mu], zeroOp, zeroOp},
   {zeroOp, multOp[I w/mu], zeroOp},
   {zeroOp, zeroOp, multOp[I w/kc]}};

a22ref = {
   {zeroOp, zeroOp, scalOp[-1, dOp[1]]},
   {zeroOp, zeroOp, scalOp[-1, dOp[2]]},
   {scalOp[-1, compOp[multOp[gam], dOp[1]]],
    scalOp[-1, compOp[multOp[gam], dOp[2]]], zeroOp}};

Print[""];
Print["--- Part 4: arbiter (A) --- Appendix J, equations (J.5)-(J.8) -----"];
report["A11 from (D.20) == (J.5)", opMatEqualQ[a11op, a11ref, 3]];
report["A12 from (D.21) == (J.6)", opMatEqualQ[a12op, a12ref, 3]];
report["A21 from (D.22) == (J.7)", opMatEqualQ[a21op, a21ref, 3]];
report["A22 from (D.23) == (J.8)", opMatEqualQ[a22op, a22ref, 3]];

(* The ordering is load-bearing: a corrupted A22 that multiplies AFTER
   differentiating must fail.  A check that cannot fail is no check. *)
a22bad = {
   {zeroOp, zeroOp, scalOp[-1, dOp[1]]},
   {zeroOp, zeroOp, scalOp[-1, dOp[2]]},
   {scalOp[-1, compOp[dOp[1], multOp[gam]]],
    scalOp[-1, compOp[dOp[2], multOp[gam]]], zeroOp}};
report["NEGATIVE CONTROL: A22 with the derivative wrongly outside is rejected",
  ! opMatEqualQ[a22op, a22bad, 3]];

(* ============================================================================
   Part 5.  The symmetry relations (16)-(18), at operator level.

   A11^t = -A22,    A12^t = A12,    A21^t = A21

   These hold for the OPERATORS, i.e. under integration by parts, and they are
   precisely what makes the reciprocity theorems of the paper work.  They are
   also the operator-level ancestor of the thesis's quasi-Hamiltonian relation
   checked in Part 7, so passing here and failing there would localise a fault
   to the Fourier reduction rather than the derivation.
   ============================================================================ *)

Print[""];
Print["--- Part 5: operator symmetry relations (16)-(18) -----------------"];
report["A11^t = -A22   (16)", opMatEqualQ[opMatT[a11op], opMatScal[-1, a22op], 3]];
report["A12^t =  A12   (17)", opMatEqualQ[opMatT[a12op], a12op, 3]];
report["A21^t =  A21   (18)", opMatEqualQ[opMatT[a21op], a21op, 3]];

(* ============================================================================
   Part 6.  The source vector, completing d3 q = A q + d.

   d1 = f + (1/(i w)) d_a U_ab h_b                                      (D.18)
   d2 = C33^{-1} C_3l h_l                                               (D.19)

   with q1 = -tau_3, q2 = v                                             (D.10)

   h_kl is the deformation-rate density; h_l is its l-th column, (h_l)_k = h_kl.
   ============================================================================ *)

hvec[l_] := Table[hh[kk, l][x1, x2, x3], {kk, 3}];
fvec = Table[ff[ii][x1, x2, x3], {ii, 3}];

d1src = Simplify[
   fvec + (1/(I w)) Sum[D[umat[al, be].hvec[be], xv[[al]]], {al, 2}, {be, 2}]];
d2src = Simplify[c33inv.Sum[cmat[3, l].hvec[l], {l, 3}]];

Print[""];
Print["--- Part 6: source vector (D.18)-(D.19) --------------------------"];
disp["d1 =", d1src];
disp["d2 =", d2src];
report["d vanishes with the sources (no spurious inhomogeneous term)",
  Simplify[Join[d1src, d2src] /. {ff[_] -> (0 &), hh[_, _] -> (0 &)}] ===
   ConstantArray[0, 6]];

(* ============================================================================
   Part 7.  ARBITER (B): the thesis system matrix A(kx,ky).

   Thesis_Recompiled_2026/GRepresentations.tex, equation (Akdef).  This is an
   independently derived object: displacement-stress rather than velocity-
   traction, ordered (z,x,y) rather than (x1,x2,x3), and normalised so that
   rho w^2 appears where the paper has i w rho.  It therefore cannot be a
   restatement of Appendix J, and agreement is real evidence.

   Reduce the operator form to a laterally invariant medium and act on a
   lateral plane wave exp(i(k1 x1 + k2 x2)), so d_a -> i k_a.  Both sources use
   exp(-i w t) and a forward transform exp(-i k.x), hence d_a -> +i k_a.
   ============================================================================ *)

constRule = {lamf -> (lamC &), muf -> (muC &), rhof -> (rhoC &)};

fourier[m_] := Module[{ph = Exp[I (k1 x1 + k2 x2)]},
   Table[Simplify[(act[m[[i, j]], ph]/ph) /. constRule], {i, 3}, {j, 3}]];

a6 = ArrayFlatten[{{fourier[a11op], fourier[a12op]},
                   {fourier[a21op], fourier[a22op]}}];

Print[""];
Print["--- Part 7: arbiter (B) --- thesis eq. (Akdef) --------------------"];
disp["A(k1,k2), paper ordering q = (-tau_13,-tau_23,-tau_33,v1,v2,v3):", a6];

(* thesis shorthand, local to (Akdef) *)
kcC = lamC + 2 muC;
gamT = lamC/kcC; aT = 1/kcC; bT = 1/muC;
zetT = 4 muC (lamC + muC)/kcC; chiT = 2 muC lamC/kcC;

aThesis = {
   {0, -I gamT k1, -I gamT k2, aT, 0, 0},
   {-I k1, 0, 0, 0, bT, 0},
   {-I k2, 0, 0, 0, 0, bT},
   {-rhoC w^2, 0, 0, 0, -I k1, -I k2},
   {0, -rhoC w^2 + zetT k1^2 + muC k2^2, k1 k2 (chiT + muC), -I k1 gamT, 0, 0},
   {0, k1 k2 (chiT + muC), -rhoC w^2 + zetT k2^2 + muC k1^2, -I k2 gamT, 0, 0}};

report["thesis zeta == nu1 (J.9)", Simplify[zetT - (nu1 /. constRule)] === 0];
report["thesis chi  == nu2 (J.10)", Simplify[chiT - (nu2 /. constRule)] === 0];

(* Field-vector map b = S q.
     paper   q = (-tau_13, -tau_23, -tau_33, v1, v2, v3)
     thesis  b = ( u3, u1, u2, tau_33, tau_31, tau_32 )
   with v = -i w u and tau symmetric.  S is constant in x3, so
   d3 b = S d3 q = S A q = S A S^{-1} b, hence A_thesis = S A S^{-1}. *)
smap = Table[0, {6}, {6}];
smap[[1, 6]] = 1/(-I w);       (* u3 = v3/(-i w) *)
smap[[2, 4]] = 1/(-I w);       (* u1 = v1/(-i w) *)
smap[[3, 5]] = 1/(-I w);       (* u2 = v2/(-i w) *)
smap[[4, 3]] = -1;             (* tau_33 = -q3 *)
smap[[5, 1]] = -1;             (* tau_31 = -q1 *)
smap[[6, 2]] = -1;             (* tau_32 = -q2 *)

aMapped = Simplify[smap.a6.Inverse[smap]];
report["S A_paper S^{-1} == A_thesis (Akdef), entry by entry",
  Simplify[aMapped - aThesis] === ConstantArray[0, {6, 6}]];

(* ============================================================================
   Part 8.  Structural checks on the reduced 6x6 system.

   (a) the quasi-Hamiltonian / symplectic relation, thesis eq. (ATdef) and
       paper eq. (27) A^t J6 = -J6 A  (the paper writes this J6 as N).  In the wavenumber domain operator
       transposition d_a -> -d_a becomes k -> -k.
   (b) the eigenvalues must be +-i k_z,c with one P pair and two S pairs, i.e.
       the characteristic polynomial factors as (s^2 + kzP^2)(s^2 + kzS^2)^2.

   (b) is the check that binds the algebra to physics: nothing in Parts 1-7
   knows the wave speeds, so recovering alpha^2 = (lam+2mu)/rho and
   beta^2 = mu/rho from the spectrum is independent of every transcription.
   ============================================================================ *)

Print[""];
Print["--- Part 8: structure of the reduced system -----------------------"];

j6 = ArrayFlatten[{{ConstantArray[0, {3, 3}], IdentityMatrix[3]},
                     {-IdentityMatrix[3], ConstantArray[0, {3, 3}]}}];

report["A^t(-k) J6 = -J6 A(k)  (paper eq. 27 / thesis eq. ATdef)",
  Simplify[Transpose[a6 /. {k1 -> -k1, k2 -> -k2}].j6 + j6.a6] ===
   ConstantArray[0, {6, 6}]];

report["the same relation holds for the thesis matrix",
  Simplify[Transpose[aThesis /. {k1 -> -k1, k2 -> -k2}].j6 + j6.aThesis] ===
   ConstantArray[0, {6, 6}]];

charpoly = Factor[Simplify[CharacteristicPolynomial[a6, s]]];
Print["characteristic polynomial: ", charpoly];

kzp2 = rhoC w^2/kcC - k1^2 - k2^2;      (* w^2/alpha^2 - kt^2 *)
kzs2 = rhoC w^2/muC - k1^2 - k2^2;      (* w^2/beta^2  - kt^2 *)
report["char. poly == (s^2 + kzP^2)(s^2 + kzS^2)^2  [alpha^2=(lam+2mu)/rho, beta^2=mu/rho]",
  Simplify[charpoly - (s^2 + kzp2) (s^2 + kzs2)^2] === 0];

report["spectrum is the 3 vertical slownesses in +- pairs (trace A = 0)",
  Simplify[Tr[a6]] === 0];

(* Vertical incidence: k1 = k2 = 0 must give exactly two distinct speeds,
   alpha once and beta twice. *)
(* At normal incidence the six eigenvalues are +-i w/alpha once each and
   +-i w/beta twice each, so their squares carry multiplicities 2 and 4. *)
evNormal = Simplify[Eigenvalues[a6 /. {k1 -> 0, k2 -> 0}]];
Print["eigenvalues at normal incidence: ", evNormal];
report["normal incidence: +-i w/alpha (once), +-i w/beta (twice)",
  Sort[Simplify[evNormal^2]] ===
   Sort[{-rhoC w^2/kcC, -rhoC w^2/kcC,
         -rhoC w^2/muC, -rhoC w^2/muC, -rhoC w^2/muC, -rhoC w^2/muC}]];

Print[""];
Print[bar];
Print["  ", nPass, " passed, ", nFail, " failed"];
Print[bar];
If[nFail > 0, Exit[1]];
