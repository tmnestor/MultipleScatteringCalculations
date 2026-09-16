#!/usr/bin/env wolframscript
(* ==========================================================================
   ASSEMBLE A33 FROM THE MOMENT ENGINE, AND CHECK IT AGAINST THE ARCHIVE

   The 27x27 second-gradient block of the closed set (A33.nb, "The Basic
   System"; assembly transcribed from GreensTensorMoments6.nb cell 28):

       A33[ups[p,q,i], ups[r,s,j]]
           = d_qr d_ps d_ij
           + Sum_{n,k} Q^r_in,pqk dc[n,k,s,j]
           - (1/2) w^2 drho P^rs_ij,pq ,

       ups[p,q,i] = 9(i-1) + 3(q-1) + p ,
       dc[i,j,k,l] = dlam d_ij d_kl + dmu (d_ik d_jl + d_il d_jk) .

   THIS FILE DOES THE STATIC ELASTIC PART, which is what the archive's own
   `rules` substitution covers:

       A33static = d_qr d_ps d_ij + Sum_{n,k} Q^r_in,pqk dc[n,k,s,j] .

   WHY ONLY THE STATIC PART HERE.  Four notebooks in the archive
   (A33Static.nb, A33InvVersion2.nb, A33InvVersion3.nb, InvserseOfA33.nb)
   carry a `rules` list, and all four agree: SIXTEEN keys, a..q with o
   skipped, and no drho anywhere in them.  The letter matrix in A33.nb needs
   NINETEEN parameters (a..t).  So the static elastic block has three
   degeneracies that the full dynamic block -- which does carry drho, through
   the -(1/2) w^2 drho P term -- breaks.  Checking the 16 is therefore the
   sharp, well-posed comparison; the remaining three are a dynamic statement
   and are not tested here.

   ⚠ THE SOURCE-DERIVATIVE CONVENTION IS LOAD-BEARING.  Q is defined with
   d' = -d and has D = 3, so Qmom = -gStatic.  Getting it backwards flips the
   entire elastic response of this block while leaving every symmetry, parity
   and scaling check intact.  See CubeMomentCore.wl.

   ⚠ VARIABLE HYGIENE.  Every local here is $-prefixed.  The parameters of
   this system are the single letters a..t and the indices i,j,k,n,p,q,r,s --
   between them they occupy most of the alphabet, and a collision does not
   error, it silently substitutes.  (Extracting the archive's own rules list
   with a script variable named `f` returned the `f` entry keyed by a file
   path.)
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["A33 (static elastic part) FROM THE MOMENT ENGINE"];
Print["=============================================================="];

(* ---- Q via its general tensor form: 31 coefficients, then reconstruct ---- *)
Print[];
Print["[1] building the general cubic form of Q (31 invariants) ..."];
$Qs[$i_, $n_, $p_, $q_, $k_, $r_] := gStatic[$i, $n, {$p, $q, $k}, {$r}];
$form = cached["Qform", cubicTensorForm[$Qs, 6]];
Print["    invariants: ", Length[$form[[1]]]];

(* source-derivative convention: D = 3 *)
$Q[$i_, $n_, $p_, $q_, $k_, $r_] :=
   $Q[$i, $n, $p, $q, $k, $r] =
   srcFactor[3] reconstruct[$form, {$i, $n, $p, $q, $k, $r}];

$dc[$i_, $j_, $k_, $l_] :=
   dlam d[$i, $j] d[$k, $l] + dmu (d[$i, $k] d[$j, $l] + d[$i, $l] d[$j, $k]);

$ups[$p_, $q_, $i_] := 9 ($i - 1) + 3 ($q - 1) + $p;

Print[];
Print["[2] assembling the 27x27 ..."];
$A33 = cached["A33static", Module[{$B},
   $B = ConstantArray[0, {27, 27}];
   Do[$B[[$ups[$p, $q, $i], $ups[$r, $s, $j]]] =
        Simplify[d[$q, $r] d[$p, $s] d[$i, $j]
          + Sum[$Q[$i, $n, $p, $q, $k, $r] $dc[$n, $k, $s, $j], {$n, 3}, {$k, 3}]],
    {$i, 3}, {$q, 3}, {$p, 3}, {$j, 3}, {$s, 3}, {$r, 3}];
   $B]];
Print["    done."];

(* ---- how many distinct entries? ----------------------------------------
   Grouped by a NUMERIC FINGERPRINT, not by pairwise symbolic comparison.
   DeleteDuplicates with a symbolic test is O(n^2): 729 entries would mean a
   quarter of a million Simplify calls, which is how an earlier verification
   run in this project came to take over an hour.  Two independent rational
   evaluation points at 40 digits separate these expressions perfectly well,
   and the survivors are then compared symbolically -- O(n) instead. *)
Print[];
Print["[3] distinct entries of the assembled block"];
$pt1 = {lam -> 7/3, mu -> 11/5, dlam -> 13/7, dmu -> 17/11};
$pt2 = {lam -> 5/2, mu -> 3/7, dlam -> 2/9, dmu -> 19/4};
$fp[$e_] := Round[N[{$e /. $pt1, $e /. $pt2}, 40], 10^-30];
$groups = GatherBy[Flatten[$A33], $fp];
$vals = #[[1]] & /@ $groups;
$nz = Select[$vals, $fp[#] =!= $fp[0] &];
Print["    distinct values (incl. 0): ", Length[$vals]];
Print["    distinct NON-ZERO values : ", Length[$nz],
      "    (archive's rules list has 16)"];
Print["    multiplicities: ", Sort[Length /@ $groups]];

(* ---- the archive's sixteen, transcribed from A33InvVersion3.nb cell 1 ---- *)
(* Extracted via ToExpression on the cell boxes, so radicals are intact.  A
   plain-text dump of these notebooks renders SqrtBox[3] as a bare 3 and would
   silently turn 4/Sqrt[3] into 4/3. *)
$den = 9 Pi mu (lam + 2 mu);
$archive = {
 "a" -> -(2 Sqrt[3] (4 dmu lam + 3 dlam mu + 10 dmu mu)
          + 3 Pi mu (dlam + 2 dmu - 3 (lam + 2 mu)))/$den,
 "b" -> (dmu (3 Pi lam - Sqrt[3] (8 lam + 5 mu)))/$den,
 "c" -> (-3 (2 Sqrt[3] + Pi) dlam mu + 4 Sqrt[3] dmu (lam + mu))/$den,
 "d" -> (-3 Pi (dlam + 2 dmu) mu + Sqrt[3] (3 dlam mu + dmu (-5 lam + mu)))/$den,
 "e" -> (-3 Pi (dmu - 3 mu) (lam + 2 mu) + Sqrt[3] dmu (7 lam + 10 mu))/$den,
 "f" -> (dmu (-3 Pi (lam + 2 mu) + Sqrt[3] (7 lam + 10 mu)))/$den,
 "g" -> (-3 Pi dlam mu + Sqrt[3] (3 dlam mu + 4 dmu (lam + mu)))/$den,
 "i" -> (dmu (lam + mu))/(3 Sqrt[3] Pi mu (lam + 2 mu)),
 "h" -> (-3 Pi dlam mu + Sqrt[3] (3 dlam mu + dmu (lam + mu)))/$den,
 "j" -> (Sqrt[3] (3 dlam mu + dmu (-5 lam + mu))
         - 3 Pi mu (dlam + 2 dmu - 3 (lam + 2 mu)))/$den,
 "k" -> -(3 Pi (dmu - 3 mu) (lam + 2 mu) + 2 Sqrt[3] dmu (lam + 4 mu))/$den,
 "l" -> (dmu (-3 Pi (lam + 2 mu) + Sqrt[3] (4 lam + 7 mu)))/$den,
 "m" -> -(dmu (3 Pi (lam + 2 mu) + 2 Sqrt[3] (lam + 4 mu)))/$den,
 "n" -> (-3 Pi (dlam mu + 2 dmu (lam + 2 mu))
         + Sqrt[3] (3 dlam mu + 2 dmu (5 lam + 8 mu)))/$den,
 "p" -> (-3 Pi (dmu - 3 mu) (lam + 2 mu) + Sqrt[3] dmu (4 lam + 7 mu))/$den,
 "q" -> (3 (Sqrt[3] - Pi) dlam mu - (11 Sqrt[3] - 6 Pi) dmu (lam + mu))/$den};

Print[];
Print["[4] does every archive value occur as an entry of the assembled block?"];
Print["    matched on the numeric fingerprint, then confirmed symbolically."];
$missing = {}; $matched = {};
Do[With[{$hit = Select[$nz, $fp[#] === $fp[$e[[2]]] &]},
   If[$hit === {}, AppendTo[$missing, $e[[1]]],
      AppendTo[$matched, {$e[[1]], zeroQ[$hit[[1]] - $e[[2]]]}]]],
 {$e, $archive}];
Print["    archive values NOT found in the block: ",
      If[$missing === {}, "none", $missing]];
Print["    symbolic confirmation of the matches:"];
Do[Print["      ", $m[[1]], " : ", If[$m[[2]], "PASS", "FAIL (fingerprint "
         <> "matched but symbolic check failed)"]], {$m, $matched}];

Print[];
Print["[5] and conversely -- is every block entry one of the archive's 16?"];
$extra = Select[$nz,
   Function[$v, ! AnyTrue[$archive, $fp[$v] === $fp[#[[2]]] &]]];
Print["    block entries NOT in the archive list: ", Length[$extra],
      If[$extra === {}, "   PASS", "   FAIL"]];
Do[Print["      ", Simplify[$x]], {$x, Take[$extra, UpTo[4]]}];

Print[];
Print["[6] structural checks on the assembled block"];
Print["    symmetric under (p,q,i)<->(r,s,j)? ",
      $fp[Total[Flatten[$A33 - Transpose[$A33]]]] === $fp[0]];
(* The zero-contrast limit is NOT the identity, and should not be.  The
   leading term is d_qr d_ps d_ij: r pairs with q and s pairs with p, so the
   surviving entry of row ups[p,q,i] sits in column ups[q,p,i] -- the
   PERMUTATION that swaps p and q, which is the identity only on p = q.

   That is the redundancy of the 27-component basis: d_p d_q u is symmetric,
   so the 27 slots carry only 18 independent second derivatives, and the
   leading operator is the symmetriser rather than the identity.  The archive
   knows this -- GreensTensorMoments6.nb cell 28 carries a commented-out
   18x18 assembly that folds the pairs with a factor 2 on the off-diagonals. *)
$P0 = $A33 /. {dlam -> 0, dmu -> 0};
$swap = Normal[SparseArray[
   Table[{$ups[$p, $q, $i], $ups[$q, $p, $i]} -> 1,
         {$i, 3}, {$p, 3}, {$q, 3}] // Flatten, {27, 27}]];
Print["    zero-contrast limit is the p<->q swap (not the identity)? ",
      $P0 === $swap];
Print["    that swap squares to the identity? ", $swap . $swap === IdentityMatrix[27]];
Print["    (so the 27 slots carry 18 independent second derivatives)"];
Print["    block is invertible at zero contrast?  det = ", Det[$P0]];

(* ==========================================================================
   THE DENSITY HALF, AND THE FULL NINETEEN

       A33 = d_qr d_ps d_ij + Sum_nk Q dc  -  (1/2) om^2 drho P^rs_ij,pq

   P has D = 2, so the source-derivative factor is +1 and Pmom is gStatic
   unchanged.  Adding this term is what breaks the degeneracies of the static
   elastic block: that block has SIXTEEN distinct non-zero entries, while the
   letter matrix M of A33.nb needs NINETEEN.

   THE TEST IS THE PATTERN, NOT THE COUNT.  Counting distinct values would be
   satisfied by any assembly that happened to produce 19 numbers.  What is
   checked here is the PARTITION OF THE 729 POSITIONS induced by equality:
   two positions carry the same letter in M if and only if they carry the same
   value in the assembled block.  That pins the placement of every entry, and
   only then are the letters read off.
   ========================================================================== *)

Print[];
Print["[7] adding the density half,  -(1/2) om^2 drho P^rs_ij,pq"];
(* P is a rank-6 cubic tensor exactly like Q, so it is built the same way:
   31 invariant coefficients, then reconstructed.  Evaluating it component by
   component instead costs up to 729 gStatic calls rather than 31 -- same
   answer, ~23x the symbolic work, and the penalty is far worse for the
   dynamic assembly where each call is much more expensive. *)
$Ps[$i_, $j_, $p_, $q_, $r_, $s_] := gStatic[$i, $j, {$p, $q}, {$r, $s}];
$formP = cached["Pform", cubicTensorForm[$Ps, 6]];
$P[$i_, $j_, $p_, $q_, $r_, $s_] := $P[$i, $j, $p, $q, $r, $s] =
   srcFactor[2] reconstruct[$formP, {$i, $j, $p, $q, $r, $s}];
$A33full = cached["A33full", Module[{$B},
   $B = ConstantArray[0, {27, 27}];
   Do[$B[[$ups[$p, $q, $i], $ups[$r, $s, $j]]] =
        $A33[[$ups[$p, $q, $i], $ups[$r, $s, $j]]]
        - (1/2) om^2 drho $P[$i, $j, $p, $q, $r, $s],
    {$i, 3}, {$q, 3}, {$p, 3}, {$j, 3}, {$s, 3}, {$r, 3}];
   $B]];
Print["    done."];

$pt1f = Join[$pt1, {om -> 5/4, drho -> 3/8, Del -> 9/7}];
$pt2f = Join[$pt2, {om -> 2/5, drho -> 7/9, Del -> 4/3}];
$fpf[$e_] := Round[N[{$e /. $pt1f, $e /. $pt2f}, 40], 10^-30];
$gf = GatherBy[Range[729], $fpf[Flatten[$A33full][[#]]] &];
Print["    distinct values (incl. 0): ", Length[$gf]];
Print["    distinct NON-ZERO values : ",
      Count[$gf, $g_ /; $fpf[Flatten[$A33full][[$g[[1]]]]] =!= $fpf[0]]];

(* ---- the archive's letter matrix, transcribed from A33.nb cell 2 ---- *)
Print[];
Print["[8] the archive's 19-letter matrix M (A33.nb cell 2)"];
$M11 = {{a,0,0,0,b,0,0,0,b},{0,e,0,f,0,0,0,0,0},{0,0,e,0,0,0,f,0,0},{0,k,0,l,0,0,0,0,0},{m,0,0,0,n,0,0,0,p},{0,0,0,0,0,s,0,t,0},{0,0,k,0,0,0,l,0,0},{0,0,0,0,0,t,0,s,0},{m,0,0,0,p,0,0,0,n}};
$M12 = {{0,c,0,d,0,0,0,0,0},{g,0,0,0,h,0,0,0,i},{0,0,0,0,0,k,0,j,0},{g,0,0,0,h,0,0,0,i},{0,q,0,r,0,0,0,0,0},{0,0,s,0,0,0,j,0,0},{0,0,0,0,0,k,0,j,0},{0,0,s,0,0,0,j,0,0},{0,s,0,k,0,0,0,0,0}};
$M13 = {{0,0,c,0,0,0,d,0,0},{0,0,0,0,0,j,0,k,0},{g,0,0,0,i,0,0,0,h},{0,0,0,0,0,j,0,k,0},{0,0,s,0,0,0,k,0,0},{0,s,0,j,0,0,0,0,0},{g,0,0,0,i,0,0,0,h},{0,s,0,j,0,0,0,0,0},{0,0,q,0,0,0,r,0,0}};
$M21 = {{0,r,0,q,0,0,0,0,0},{h,0,0,0,g,0,0,0,i},{0,0,0,0,0,s,0,j,0},{h,0,0,0,g,0,0,0,i},{0,d,0,c,0,0,0,0,0},{0,0,k,0,0,0,j,0,0},{0,0,0,0,0,s,0,j,0},{0,0,k,0,0,0,j,0,0},{0,k,0,s,0,0,0,0,0}};
$M22 = {{n,0,0,0,m,0,0,0,p},{0,l,0,k,0,0,0,0,0},{0,0,s,0,0,0,t,0,0},{0,f,0,e,0,0,0,0,0},{b,0,0,0,a,0,0,0,b},{0,0,0,0,0,e,0,f,0},{0,0,t,0,0,0,s,0,0},{0,0,0,0,0,k,0,l,0},{p,0,0,0,m,0,0,0,n}};
$M23 = {{0,0,0,0,0,s,0,k,0},{0,0,j,0,0,0,k,0,0},{0,j,0,s,0,0,0,0,0},{0,0,j,0,0,0,k,0,0},{0,0,0,0,0,c,0,d,0},{i,0,0,0,g,0,0,0,h},{0,j,0,s,0,0,0,0,0},{i,0,0,0,g,0,0,0,h},{0,0,0,0,0,q,0,r,0}};
$M31 = {{0,0,r,0,0,0,q,0,0},{0,0,0,0,0,j,0,s,0},{h,0,0,0,i,0,0,0,g},{0,0,0,0,0,j,0,s,0},{0,0,k,0,0,0,s,0,0},{0,k,0,j,0,0,0,0,0},{h,0,0,0,i,0,0,0,g},{0,k,0,j,0,0,0,0,0},{0,0,d,0,0,0,c,0,0}};
$M32 = {{0,0,0,0,0,k,0,s,0},{0,0,j,0,0,0,s,0,0},{0,j,0,k,0,0,0,0,0},{0,0,j,0,0,0,s,0,0},{0,0,0,0,0,r,0,q,0},{i,0,0,0,h,0,0,0,g},{0,j,0,k,0,0,0,0,0},{i,0,0,0,h,0,0,0,g},{0,0,0,0,0,d,0,c,0}};
$M33 = {{n,0,0,0,p,0,0,0,m},{0,s,0,t,0,0,0,0,0},{0,0,l,0,0,0,k,0,0},{0,t,0,s,0,0,0,0,0},{p,0,0,0,n,0,0,0,m},{0,0,0,0,0,l,0,k,0},{0,0,f,0,0,0,e,0,0},{0,0,0,0,0,f,0,e,0},{b,0,0,0,b,0,0,0,a}};
$M = ArrayFlatten[{{$M11,$M12,$M13},{$M21,$M22,$M23},{$M31,$M32,$M33}}];
$letters = {a,b,c,d,e,f,g,h,i,j,k,l,m,n,p,q,r,s,t};
Print["    parameters in M: ", Length[$letters],
      "    non-zero positions: ", Count[Flatten[$M], $x_ /; $x =!= 0]];

Print[];
Print["[9] do the two equality-partitions of the 729 positions coincide?"];
$gM = GatherBy[Range[729], Flatten[$M][[#]] &];
$sortPart[$g_] := Sort[Sort /@ $g];
$same = $sortPart[$gf] === $sortPart[$gM];
Print["    partition from the assembled block: ", Length[$gf], " classes"];
Print["    partition from the archive's M    : ", Length[$gM], " classes"];
Print["    partitions identical: ", $same, If[$same, "   PASS", "   FAIL"]];
If[! $same,
   Print["    class-size multiset (block)  : ", Sort[Length /@ $gf]];
   Print["    class-size multiset (archive): ", Sort[Length /@ $gM]]];

(* ==========================================================================
   [9b][9c] WHY THE ARCHIVE'S M IS COARSER

   The support agrees exactly -- 183 non-zero positions in both -- so this is
   not a structural disagreement about which entries vanish.  The archive's
   partition is COARSER: it merges entries the assembled block separates.

   That direction is diagnostic.  Adding dynamic terms breaks degeneracies and
   can only make the true partition FINER, so no amount of missing dynamics
   explains a coarser M.  What does explain it is reading the letters off a
   NUMERIC matrix: A33InvVersion3.nb runs at lam = mu = 48 and dlam = dmu = 5,
   and at those values genuinely distinct expressions coincide.
   ========================================================================== *)

Print[];
Print["[9b] is the archive's partition a COARSENING of the block's?"];
$blockOf = ConstantArray[0, 729];
Do[Do[$blockOf[[$x]] = $c, {$x, $gf[[$c]]}], {$c, Length[$gf]}];
$coarsening = AllTrue[$gM, Length[DeleteDuplicates[$blockOf[[#]]]] == 1 &];
Print["    every archive class is a union of block classes: ", $coarsening];
Print["    (if True, M merges entries that are genuinely different)"];

Print[];
Print["[9c] TEST: does the block's partition collapse onto the archive's"];
Print["     at the archive's own parameter values, lam = mu, dlam = dmu?"];
$degen = {lam -> 48, mu -> 48, dlam -> 5, dmu -> 5, om -> 3, drho -> 1/5,
          Del -> 1};
$fpd[$e_] := Round[N[$e /. $degen, 40], 10^-25];
$gd = GatherBy[Range[729], $fpd[Flatten[$A33full][[#]]] &];
Print["    classes at lam=mu, dlam=dmu: ", Length[$gd],
      "     archive M: ", Length[$gM]];
Print["    partitions identical there: ",
      $sortPart[$gd] === $sortPart[$gM]];

Print[];
Print["[10] reading off the nineteen"];
If[$same,
   Do[With[{$pos = FirstPosition[Flatten[$M], $L][[1]]},
      Print["    ", $L, " = ",
            Simplify[Flatten[$A33full][[$pos]], Assumptions -> Del > 0]]],
    {$L, $letters}],
   Print["    skipped -- the partitions do not match, so the letter"];
   Print["    assignment is not established."]];

Print[];
Print["=============================================================="];
Print[If[$missing === {} && $extra === {},
   "static 16: PASS -- the engine reproduces the archive's A33 entries.",
   "static 16: FAIL -- the assembled block disagrees with the archive."]];
Print[If[$same,
   "full 19:   PASS -- placement and values both reproduced.",
   "full 19:   NOT REPRODUCED at this level of dynamics -- see [9]."]];
Print["=============================================================="];
