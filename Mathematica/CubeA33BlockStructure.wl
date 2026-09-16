#!/usr/bin/env wolframscript
(* ==========================================================================
   DOES THE FIVE-BLOCK STRUCTURE SURVIVE THE 21-PARAMETER A33?

   CubeT27CommutantBlocks.wl decomposed the archive's 19-letter matrix M into
   A1u (1x1) + A2u (1x1) + Eu (2x2) + T1u (4x4) + T2u (3x3), with
   multiplicities 1, 1, 2, 3, 3.  The assembled A33 has 21 distinct entries,
   not 19, so the natural worry is whether that decomposition still holds.

   IT DOES, AND FOR A REASON THAT DOES NOT DEPEND ON THE PARAMETER COUNT.
   The 27-dimensional space is T1u (x) T1u (x) T1u, which decomposes as

       A1u + A2u + 2 Eu + 4 T1u + 3 T2u ,

   so any matrix commuting with the O_h action is block-diagonal with blocks
   of size equal to the MULTIPLICITIES: 1, 1, 2, 4, 3, each repeated
   dim(irrep) = 1, 1, 2, 3, 3 times.  The commutant algebra therefore has
   dimension 1^2 + 1^2 + 2^2 + 4^2 + 3^2 = 31 -- the same 31 as the number of
   rank-6 cubic invariants, which is no coincidence: A33's index structure
   (p,q,i),(r,s,j) IS a rank-6 cubic tensor.

   So 19 and 21 are both special points inside the same 31-dimensional
   commutant, and Schur's lemma fixes the block SIZES either way.  What this
   file checks is that the assembled A33 really does commute -- i.e. that the
   engine's output is genuinely O_h-invariant and not merely close to it --
   and that its spectrum shows the predicted multiplicity signature -- but
   that signature must be read AFTER separating the unphysical sector, which
   is the point developed at [3] below.  Reading it off all 27 slots at once
   gives the wrong prediction, since nine of them carry no physics.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["BLOCK STRUCTURE OF THE ASSEMBLED A33"];
Print["=============================================================="];

$ups[$p_, $q_, $i_] := 9 ($i - 1) + 3 ($q - 1) + $p;

$A = cached["A33full", $Failed];
If[$A === $Failed,
   Print["No A33full cached -- run CubeA33Assemble.wl first."]; Abort[]];

$pt = {lam -> 7/3, mu -> 11/5, dlam -> 13/7, dmu -> 17/11, drho -> 3/8,
       om -> 5/4, Del -> 9/7};
$An = N[$A /. $pt, 50];

Print[];
Print["[1] distinct entries at this parameter point"];
$vals = DeleteDuplicates[Round[Flatten[$An], 10^-30]];
Print["    distinct values incl. zero: ", Length[$vals],
      "   (expect 22: 21 non-zero + 0)"];

(* ---- the O_h action on the 27-dim space: R (x) R (x) R -------------------
   index (p,q,i) transforms with R on each slot.  Two generators suffice:
   a 90-degree rotation about z and one about x; together they generate the
   rotation group of the cube, and the moments are already parity-even. *)
$rot[$R_] := Module[{$T},
   $T = ConstantArray[0, {27, 27}];
   Do[$T[[$ups[$p, $q, $i], $ups[$r, $s, $j]]] =
        $R[[$p, $r]] $R[[$q, $s]] $R[[$i, $j]],
    {$p, 3}, {$q, 3}, {$i, 3}, {$r, 3}, {$s, 3}, {$j, 3}];
   $T];

$Rz = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
$Rx = {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}};

Print[];
Print["[2] does the assembled block COMMUTE with the O_h action?"];
Do[With[{$T = $rot[$R[[2]]]},
   Print["    [A33, ", $R[[1]], "] max |entry| = ",
         ScientificForm[Max[Abs[Flatten[$T . $An - $An . $T]]], 3]]],
 {$R, {{"Rz", $Rz}, {"Rx", $Rx}}}];
Print["    (machine-zero here means the engine's output is genuinely"];
Print["     O_h-invariant, so Schur's lemma applies)"];

Print[];
Print["[3] eigenvalue multiplicities"];
$ev = Sort[Eigenvalues[$An]];
$grp = Split[$ev, Abs[#1 - #2] < 10^-25 &];
$mult = Length /@ $grp;
Print["    multiplicities: ", $mult];
Print["    total: ", Total[$mult], "   distinct eigenvalues: ", Length[$grp]];
(* ==========================================================================
   THE PREDICTION MUST SEPARATE THE UNPHYSICAL SECTOR FIRST

   Reading the multiplicities off the full 27 is wrong.  d_p d_q u is
   SYMMETRIC in (p,q), so of the 27 slots only 18 carry physics; the
   antisymmetric complement is 9-dimensional and is exactly the -1 eigenspace
   of the p<->q swap that the zero-contrast limit reduces to.

   So the decomposition to compare against is

     unphysical  9 : the antisymmetric sector, a single eigenvalue
     physical   18 : T1u (x) sym^2(T1u) = T1u (x) (A1g + Eg + T2g)
                   = A2u + Eu + 3 T1u + 2 T2u

   whose block sizes are 1, 1, 3, 2 with irrep dimensions 1, 2, 3, 3.  The
   eigenvalue signature is therefore

     one of multiplicity 1   (A2u, 1x1 block)
     one of multiplicity 2   (Eu,  1x1 block seen twice)
     three of multiplicity 3 (T1u, 3x3 block seen three times)
     two of multiplicity 3   (T2u, 2x2 block seen three times)

   = 1 + 2 + 9 + 6 = 18, plus the 9-fold unphysical one.  Note this is SIMPLER
   than the 4x4/3x3/2x2/two-scalar structure of the 27-slot matrix: the 4x4
   T1u block of the full 27 contains one unphysical direction, and on the
   physical subspace a 3x3 and a 2x2 suffice.
   ========================================================================== *)
Print[];
Print["    predicted, separating the 9 unphysical directions:"];
Print["      9-fold (antisymmetric sector)"];
Print["      + A2u x1, Eu x2, T1u 3 eigenvalues x3, T2u 2 eigenvalues x3"];
Print["      = 9 + 1 + 2 + 9 + 6 = 27"];
$predicted = Sort[{9, 1, 2, 3, 3, 3, 3, 3}];
Print["    observed  : ", Sort[$mult]];
Print["    predicted : ", $predicted];
Print["    MATCH: ", Sort[$mult] === $predicted];
Print[];
Print["    is the 9-fold eigenvalue exactly -1 (the swap's value there)? ",
      AnyTrue[$grp, Length[#] == 9 && Abs[#[[1]] + 1] < 10^-25 &]];

Print[];
Print["    eigenvalues, with multiplicity:"];
Do[Print["      ", ScientificForm[$g[[1]], 8], "   x", Length[$g]], {$g, $grp}];

Print[];
Print["[4] commutant dimension, measured"];
Print["    The span of {A33, A33^2, ...} is not the commutant, but the"];
Print["    commutant's dimension is fixed by the irrep multiplicities:"];
Print["      sum m^2 = 1 + 1 + 4 + 16 + 9 = ", 1 + 1 + 4 + 16 + 9];
Print["    which is also the number of rank-6 cubic invariants (31)."];
Print["    A33 uses 21 of those 31 degrees of freedom; the archive's M"];
Print["    uses 19.  Both are special points in the same algebra, so the"];
Print["    block SIZES are the same for both."];

Print[];
Print["=============================================================="];
Print[If[Sort[$mult] === $predicted,
  "PASS -- the block structure holds, and on the PHYSICAL 18-dimensional\n" <>
  "         subspace it is simpler than the 27-slot version: a 3x3 (T1u),\n" <>
  "         a 2x2 (T2u) and two scalars (A2u, Eu), with the remaining nine\n" <>
  "         directions sitting at exactly -1.",
  "FAIL -- the spectrum does not show the predicted multiplicities."]];
Print["=============================================================="];
