#!/usr/bin/env wolframscript
(* ==========================================================================
   CHEBYSHEV MOMENTS BY RECOMBINATION -- no new integrals

   Task 2 of plans/chebyshev_tensor_basis.md.

   THE ECONOMY THAT MAKES THIS PLAN WORTH DOING.  Chebyshev polynomials are
   INTEGER combinations of monomials, so every Chebyshev-weighted moment is a
   finite recombination of moments the engine already computes:

       Echeb[m; ds; {i,j,k}] = Sum_terms  coeff * E[m; ds; monomial]

   No new master integral is derived here, and none may be.  The file asserts
   that mechanically: if `Integrate` appears anywhere in a computed Echeb, the
   recombination has been done wrong and the run fails.  That check is cheap
   and it is the difference between a basis change and a new derivation.

   SCALE.  Chebyshev lives on [-hw, hw], hw = Del/2, so T_1(x/hw) = x/hw and
   the degree-1 identity carries a factor 1/hw.  The plan writes it without
   that factor; the factor is real and is asserted here in its correct form
   rather than absorbed silently.

   PARITY IS PRESERVED.  T_n has parity (-1)^n, the same as x^n, so the
   gerade/ungerade split of the closed set survives the basis change verbatim.
   That matters because the whole motivation is to raise the GERADE degree --
   if the basis change mixed the sectors, the motivation would evaporate.
   ========================================================================== *)

$ChebQuiet = True;
Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeChebyshevBasis.wl"}]];

Print["=============================================================="];
Print["CHEBYSHEV MOMENTS BY RECOMBINATION"];
Print["=============================================================="];

$nfail = 0;
chk[$lbl_, $ok_] := (If[! TrueQ[$ok], $nfail++];
   Print["  ", If[TrueQ[$ok], "PASS", "FAIL"], "  ", $lbl]);

(* ---- the recombination ------------------------------------------------ *)
Echeb[$m_, $ds_List, $idx_List] := Echeb[$m, $ds, $idx] =
  Module[{$val},
   $val = Total[
      (#[[1]] E$[$m, $ds, expToIdx[#[[2]]]]) & /@ chebWeight[$idx]];
   (* NO NEW INTEGRALS.  If this fires, the recombination is wrong. *)
   If[! FreeQ[$val, Integrate],
      Print["  !! Echeb produced an unevaluated Integrate at m=", $m,
            " ds=", $ds, " idx=", $idx];
      Abort[]];
   pellSimplify[$val]];

Print[];
Print["[1] the degree-0 and degree-1 identities, exact"];
Print["    T_0 = 1 and T_1 = x/hw, so these are the same moments"];
Do[With[{$m = $c[[1]], $ds = $c[[2]]},
   chk["Echeb[" <> ToString[$m] <> ";" <> ToString[$ds] <> ";{0,0,0}] == E[..;{}]",
       zeroQ[Echeb[$m, $ds, {0, 0, 0}] - E$[$m, $ds, {}]]];
   chk["Echeb[" <> ToString[$m] <> ";" <> ToString[$ds] <> ";{1,0,0}] == (1/hw) E[..;{1}]",
       zeroQ[Echeb[$m, $ds, {1, 0, 0}] - E$[$m, $ds, {1}]/hw]]],
 {$c, {{-1, {1, 1}}, {1, {1, 2}}, {-1, {1, 2, 3}}}}];

Print[];
Print["[2] the degree-2 identity: T_2 = 2(x/hw)^2 - 1"];
Do[With[{$m = $c[[1]], $ds = $c[[2]]},
   chk["Echeb[" <> ToString[$m] <> ";" <> ToString[$ds] <> ";{2,0,0}] == 2/hw^2 E[..;{1,1}] - E[..;{}]",
       zeroQ[Echeb[$m, $ds, {2, 0, 0}]
             - (2 E$[$m, $ds, {1, 1}]/hw^2 - E$[$m, $ds, {}])]]],
 {$c, {{-1, {1, 1}}, {1, {1, 2}}}}];

Print[];
Print["[3] PARITY SURVIVES.  T_n has parity (-1)^n, so a Chebyshev weight of"];
Print["    odd total degree must vanish exactly where an odd monomial does."];
Do[With[{$idx = $p[[1]], $ds = $p[[2]]},
   chk["Echeb[-1;" <> ToString[$ds] <> ";" <> ToString[$idx] <> "] == 0 by parity",
       zeroQ[Echeb[-1, $ds, $idx]]]],
 {$p, {{{1, 0, 0}, {1, 1}}, {{0, 1, 0}, {2, 2}}, {{1, 1, 1}, {1, 2}},
       {{3, 0, 0}, {1, 1}}}}];

Print[];
Print["[4] no new integrals were derived"];
Print["    The runtime guard inside Echeb aborts the run if any recombination"];
Print["    leaves an unevaluated Integrate, so reaching this line at all is"];
Print["    the evidence.  Asserted explicitly on a fresh value rather than by"];
Print["    grepping the source -- a source grep for the string matches its own"];
Print["    literal and is worthless."];
chk["a freshly computed Echeb is free of Integrate",
    FreeQ[Echeb[-1, {1}, {3, 0, 0}], Integrate]];
chk["and free of Undefined / ConditionalExpression",
    FreeQ[Echeb[-1, {1}, {3, 0, 0}], Undefined] &&
    FreeQ[Echeb[-1, {1}, {3, 0, 0}], ConditionalExpression]];

Print[];
Print["[5] worked GERADE moments at degree 3 -- the tier that moves the far"];
Print["    field.  Chosen so the total multiplicity of each axis is EVEN;"];
Print["    an odd one vanishes identically and would demonstrate nothing."];
Do[With[{$m = $c[[1]], $ds = $c[[2]], $idx = $c[[3]]},
   Print["    Echeb[", $m, ";", $ds, ";", $idx, "] = ",
         Echeb[$m, $ds, $idx]]],
 {$c, {{-1, {1}, {3, 0, 0}}, {-1, {2}, {2, 1, 0}}, {1, {1, 1, 1}, {3, 0, 0}}}}];
chk["the degree-3 gerade moment is genuinely non-zero",
    ! zeroQ[Echeb[-1, {1}, {3, 0, 0}]]];

Print[];
Print["=============================================================="];
Print[If[$nfail == 0,
   "PASS -- Chebyshev moments are recombinations of existing moments.\n" <>
   "        No new master integrals; parity preserved.",
   "FAIL -- " <> ToString[$nfail] <> " check(s) failed."]];
Print["=============================================================="];
