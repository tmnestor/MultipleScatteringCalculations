#!/usr/bin/env wolframscript
(* ==========================================================================
   WHICH PLACEMENT OF THE LEADING TERM REPRODUCES THE ARCHIVE'S 19 LETTERS?

   The disagreement between the assembled A33 and the archive's matrix M is
   localised (CubeA33PartitionDiff.wl): the block's partition REFINES M's, and
   exactly two letters split, each 24 -> 18 + 6.  The six that split off are

       k : column = (q,p,i)   -- where d_qr d_ps d_ij is non-zero
       s : column = row, p != q -- the true diagonal, where it vanishes

   Both families are defined by the LEADING TERM, not by the moments.  So the
   question is what sits in front of Q.dc - (1/2) w^2 drho P.  Four candidates
   are tried here:

       none     :  no leading term at all
       swap     :  d_qr d_ps d_ij   (the assembly's current choice, from
                                     GreensTensorMoments6.nb cell 28)
       diag     :  d_pr d_qs d_ij   (the identity on the true diagonal)
       sym      :  (1/2)(d_qr d_ps + d_pr d_qs) d_ij  -- the symmetriser,
                   which is what a basis carrying only 18 independent second
                   derivatives in 27 slots actually calls for

   Only the leading term changes; Q and P come from the cache untouched, so
   this isolates one variable.  Whichever placement yields M's 20 classes AND
   the same partition identifies the convention, and the 19 letters can then
   be read off.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE LEADING TERM: WHICH PLACEMENT GIVES THE ARCHIVE'S 19?"];
Print["=============================================================="];

$ups[$p_, $q_, $i_] := 9 ($i - 1) + 3 ($q - 1) + $p;
$dc[$i_, $j_, $k_, $l_] :=
   dlam d[$i, $j] d[$k, $l] + dmu (d[$i, $k] d[$j, $l] + d[$i, $l] d[$j, $k]);

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
$Mf = Flatten[$M];
$gM = GatherBy[Range[729], $Mf[[#]] &];
$sortPart[$g_] := Sort[Sort /@ $g];
Print[];
Print["archive M: ", Length[$gM], " classes"];

(* Q and P from the cache, untouched -- only the leading term varies *)
$fQ = cached["Qform", $Failed];
$fP = cached["Pform", $Failed];
If[$fQ === $Failed || $fP === $Failed,
   Print["Run CubeA33Assemble.wl first to populate the cache."]; Abort[]];

$pt = {lam -> 7/3, mu -> 11/5, dlam -> 13/7, dmu -> 17/11, drho -> 3/8,
       om -> 5/4, Del -> 9/7};
$pt2 = {lam -> 5/2, mu -> 3/7, dlam -> 2/9, dmu -> 19/4, drho -> 7/9,
        om -> 2/5, Del -> 4/3};

lead["none"][$p_, $q_, $i_, $r_, $s_, $j_] := 0;
lead["swap"][$p_, $q_, $i_, $r_, $s_, $j_] := d[$q, $r] d[$p, $s] d[$i, $j];
lead["diag"][$p_, $q_, $i_, $r_, $s_, $j_] := d[$p, $r] d[$q, $s] d[$i, $j];
lead["sym"][$p_, $q_, $i_, $r_, $s_, $j_] :=
   (1/2) (d[$q, $r] d[$p, $s] + d[$p, $r] d[$q, $s]) d[$i, $j];

(* The archive's commented-out 18x18 assembly in GreensTensorMoments6.nb
   cell 28 scales the off-diagonal pairs:  If[r == s, temp, 2 temp].  Folding
   the symmetric pair (r,s) into one slot doubles its weight.  Applied to the
   27-slot block that is a COLUMN scaling by 2 wherever r != s -- tried here
   on top of each leading-term placement. *)
colScale[$scaled_][$r_, $s_] := If[$scaled && $r =!= $s, 2, 1];

buildN[$which_, $point_] := buildN[$which, $point, False];
buildN[$which_, $point_, $scaled_] :=
  Module[{$cQ, $cP, $QQ, $PP, $B, $dcN, $half},
   $cQ = {$fQ[[1]], N[$fQ[[2]] /. $point, 40]};
   $cP = {$fP[[1]], N[$fP[[2]] /. $point, 40]};
   $QQ[$i_, $n_, $p_, $q_, $k_, $r_] := $QQ[$i, $n, $p, $q, $k, $r] =
      srcFactor[3] reconstruct[$cQ, {$i, $n, $p, $q, $k, $r}];
   $PP[$i_, $j_, $p_, $q_, $r_, $s_] := $PP[$i, $j, $p, $q, $r, $s] =
      srcFactor[2] reconstruct[$cP, {$i, $j, $p, $q, $r, $s}];
   $dcN[$i_, $j_, $k_, $l_] := N[$dc[$i, $j, $k, $l] /. $point, 40];
   $half = N[(1/2) om^2 drho /. $point, 40];
   $B = ConstantArray[0, {27, 27}];
   Do[$B[[$ups[$p, $q, $i], $ups[$r, $s, $j]]] =
        colScale[$scaled][$r, $s] (
          lead[$which][$p, $q, $i, $r, $s, $j]
          + Sum[$QQ[$i, $n, $p, $q, $k, $r] $dcN[$n, $k, $s, $j], {$n, 3}, {$k, 3}]
          - $half $PP[$i, $j, $p, $q, $r, $s]),
    {$i, 3}, {$q, 3}, {$p, 3}, {$j, 3}, {$s, 3}, {$r, 3}];
   Flatten[$B]];

Print[];
Print["leading term        classes   support matches   partition matches M"];
$found = $None;
Do[Module[{$w = $v[[1]], $sc = $v[[2]], $A1, $A2, $g, $supp, $ok, $lbl},
   $A1 = buildN[$w, $pt, $sc]; $A2 = buildN[$w, $pt2, $sc];
   $g = GatherBy[Range[729], Round[{$A1[[#]], $A2[[#]]}, 10^-25] &];
   (* support: |value| below tolerance counts as zero (was comparing a rounded
      high-precision zero to machine 0. structurally, which is never True) *)
   $supp = Sort[Select[Range[729], Abs[$A1[[#]]] > 10^-20 &]] ===
           Sort[Select[Range[729], $Mf[[#]] =!= 0 &]];
   $ok = $sortPart[$g] === $sortPart[$gM];
   If[$ok && $found === $None, $found = $w <> If[$sc, " x2", ""]];
   $lbl = $w <> If[$sc, " (2x off-diag)", ""];
   Print["  ", StringPadRight[$lbl, 22], StringPadLeft[ToString[Length[$g]], 5],
         "        ", StringPadRight[ToString[$supp], 8], "       ", $ok,
         If[$ok, "   <<< MATCH", ""]]],
 {$v, {{"none", False}, {"swap", False}, {"diag", False}, {"sym", False},
       {"none", True}, {"swap", True}, {"diag", True}, {"sym", True}}}];

Print[];
If[$found =!= $None,
   Print["[MATCH] leading term '", $found, "' reproduces M exactly."],
   Print["[NO MATCH] none of the four placements reproduces M's partition."];
   Print["    The disagreement is then not the leading term alone."]];
Print["=============================================================="];
