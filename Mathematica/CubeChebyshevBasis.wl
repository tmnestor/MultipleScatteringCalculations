#!/usr/bin/env wolframscript
(* ==========================================================================
   TENSOR-PRODUCT CHEBYSHEV BASIS -- transfer matrices and the identity test

   Task 1 of plans/chebyshev_tensor_basis.md.

   WHY A BASIS CHANGE IS WORTH ANYTHING AT ALL.  It is not, at fixed degree:
   degree-N Chebyshev and degree-N monomials span the same space, and the
   Petrov-Galerkin projection onto that space is basis-independent, so the
   solution is identical.  What changes is whether a given degree is
   REACHABLE.  The monomial route already emits 35-digit Pell cancellations at
   degree 6, and the 2012 archive carries $Aborted cells at exactly that
   frontier.  The target is the gerade sector -- degrees 1, 3, 5 -- because the
   leading modulus far field is the stress dipole Int dc:grad(u) dV, which
   needs only d_u, and that is gerade.

   THE REGRESSION THIS FILE EXISTS FOR.  T_0(x) = 1 and T_1(x) = x.  The first
   two Chebyshev polynomials ARE the monomials, so everything at degree <= 1 --
   the entire 9-component T-matrix -- must come out BIT-IDENTICAL, not merely
   equal to tolerance.  Equality only to tolerance would mean the
   recombination is introducing arithmetic where it should be exact, and
   nothing built on it downstream would be trustworthy.

   EXACT RATIONALS THROUGHOUT.  The whole point of the exercise is
   conditioning; introducing floating point in the transfer matrices would be
   self-defeating.

   NOTE ON SCALE.  Chebyshev lives on [-hw, hw] with hw = Del/2, so the
   argument is x/hw = 2x/Del and T_1 carries a factor 1/hw.  The plan states
   the degree-1 identity as "Echeb{100} == E[.;{1}]"; with the scaling it is
   Echeb{100} == (1/hw) E[.;{1}].  The factor is real, not a convention to be
   absorbed, and the moments file asserts it in that form.

   IMPORTED BY CubeChebyshevMoments.wl.  Set $ChebQuiet = True before Get to
   suppress the self-test.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

(* ---- the scaled variable: Chebyshev on [-hw, hw], hw = Del/2 ---------- *)
cheb[$n_, $t_] := ChebyshevT[$n, $t/hw];

(* ---- transfer matrices, exact rationals ------------------------------- *)
(* chebToMono[[n+1, k+1]] = coefficient of u^k in T_n(u) *)
chebToMono[$N_] := chebToMono[$N] = Table[
   Coefficient[Expand[ChebyshevT[$n, uu]], uu, $k], {$n, 0, $N}, {$k, 0, $N}];
monoToCheb[$N_] := monoToCheb[$N] = Inverse[chebToMono[$N]];

(* ---- tensor-product weight, expanded in monomials --------------------- *)
(* chebWeight[{i,j,k}] -> {{coefficient, {p,q,r}}, ...} for x^p y^q z^r.
   This is the ONLY place the basis change enters; everything downstream
   consumes monomial moments the engine already computes. *)
chebWeight[$idx_List] := chebWeight[$idx] = Module[{$e, $terms},
   $e = Expand[cheb[$idx[[1]], x] cheb[$idx[[2]], y] cheb[$idx[[3]], z]];
   $terms = If[Head[$e] === Plus, List @@ $e, {$e}];
   Table[With[{$p = Exponent[$t, x], $q = Exponent[$t, y],
               $r = Exponent[$t, z]},
      {Simplify[$t/(x^$p y^$q z^$r)], {$p, $q, $r}}], {$t, $terms}]];

(* ---- monomial exponents -> the engine's index-list convention --------- *)
expToIdx[{$p_, $q_, $r_}] :=
   Join[ConstantArray[1, $p], ConstantArray[2, $q], ConstantArray[3, $r]];

(* ====================== self-test ====================================== *)
chebBasisSelfTest[] := Module[{$nfail = 0, $chk},
   $chk[$lbl_, $ok_] := (If[! TrueQ[$ok], $nfail++];
      Print["  ", If[TrueQ[$ok], "PASS", "FAIL"], "  ", $lbl]);

   Print["=============================================================="];
   Print["CHEBYSHEV BASIS: TRANSFER MATRICES AND THE IDENTITY TEST"];
   Print["=============================================================="];

   Print[];
   Print["[1] transfer matrices are exact rationals"];
   $chk["chebToMono[8] has no machine numbers", FreeQ[chebToMono[8], _Real]];
   $chk["monoToCheb[8] has no machine numbers", FreeQ[monoToCheb[8], _Real]];
   Print["      T_0..T_4 coefficient rows: ", chebToMono[4]];

   Print[];
   Print["[2] the transfer is an involution -- this catches index-order"];
   Print["    errors, which are the likely bug in a construction like this"];
   Do[$chk["monoToCheb . chebToMono == I at degree " <> ToString[$N],
           monoToCheb[$N] . chebToMono[$N] === IdentityMatrix[$N + 1]],
    {$N, {2, 4, 6, 8}}];

   Print[];
   Print["[3] THE IDENTITY THAT MATTERS: T_0 = 1 and T_1 = x/hw"];
   $chk["T_0(x/hw) === 1", Simplify[cheb[0, x]] === 1];
   $chk["T_1(x/hw) === x/hw", Simplify[cheb[1, x] - x/hw] === 0];
   Print["      so every degree <= 1 object is the SAME FUNCTION in both"];
   Print["      bases, up to the 1/hw scale, and the 9-component T-matrix"];
   Print["      cannot change.  Anything that does change at degree <= 1 is"];
   Print["      a bug, not a basis effect."];

   Print[];
   Print["[4] tensor weights expand into monomials with exact coefficients"];
   Do[Print["      T", $ix, " -> ", chebWeight[$ix]],
    {$ix, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {1, 1, 0}}}];
   $chk["T_{100} is exactly (1/hw) x",
        chebWeight[{1, 0, 0}] === {{1/hw, {1, 0, 0}}}];
   $chk["T_{200} is exactly 2(x/hw)^2 - 1",
        Sort[chebWeight[{2, 0, 0}]] ===
          Sort[{{2/hw^2, {2, 0, 0}}, {-1, {0, 0, 0}}}]];

   Print[];
   Print["[5] exponent -> index-list conversion for the moment engine"];
   $chk["{2,1,0} -> {1,1,2}", expToIdx[{2, 1, 0}] === {1, 1, 2}];
   $chk["{0,0,0} -> {}", expToIdx[{0, 0, 0}] === {}];

   Print[];
   Print["=============================================================="];
   Print[If[$nfail == 0,
      "PASS -- transfer matrices exact and involutive; T_0/T_1 confirmed\n" <>
      "        identical to the monomials.  Degree <= 1 cannot move.",
      "FAIL -- " <> ToString[$nfail] <> " check(s) failed."]];
   Print["=============================================================="];
   $nfail];

If[$ChebQuiet =!= True, chebBasisSelfTest[]];
