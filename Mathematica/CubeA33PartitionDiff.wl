#!/usr/bin/env wolframscript
(* ==========================================================================
   WHERE EXACTLY DO THE TWO PARTITIONS DISAGREE?

   The assembled A33 and the archive's 19-letter matrix M agree on the SUPPORT
   -- both have the same 183 non-zero positions -- but not on which entries are
   equal.  The block gives 22 equality-classes, M gives 20, and the two
   partitions CROSS: neither refines the other.

   Truncation does not explain it.  The dynamic sweep gives 22 classes at
   ng,nh = (0,2), (2,2), (2,4) and (4,4) alike: adding the k^2 and k^4 terms
   changes no equality whatsoever.  Nor does transcription (M re-extracted
   from the cell boxes matches to the position) or the degeneracy at
   lam = mu, dlam = dmu (which accounts for one class, 22 -> 21).

   So rather than buy two more expensive truncations, this file asks the
   diagnostic question directly: NAME the positions where the two disagree,
   and print their (p,q,i),(r,s,j) indices.  A disagreement that is really an
   index convention will show up as a systematic relation between the indices
   of the merged positions -- for instance M treating (r,s) as unordered where
   the assembly does not -- and that is visible by inspection of a handful of
   cases, at no computational cost beyond what is already cached.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["WHERE THE PARTITIONS DISAGREE"];
Print["=============================================================="];

$ups[$p_, $q_, $i_] := 9 ($i - 1) + 3 ($q - 1) + $p;
(* inverse of ups: position -> {p,q,i} *)
$unups[$x_] := {Mod[$x - 1, 3] + 1, Quotient[Mod[$x - 1, 9], 3] + 1,
                Quotient[$x - 1, 9] + 1};

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

(* the assembled block, static elastic + density half, from the cache *)
$A = cached["A33full", $Failed];
If[$A === $Failed,
   Print["No A33full in the cache -- run CubeA33Assemble.wl first."]; Abort[]];
$Af = Flatten[$A];

$p1 = {lam -> 7/3, mu -> 11/5, dlam -> 13/7, dmu -> 17/11, drho -> 3/8,
       om -> 5/4, Del -> 9/7};
$p2 = {lam -> 5/2, mu -> 3/7, dlam -> 2/9, dmu -> 19/4, drho -> 7/9,
       om -> 2/5, Del -> 4/3};
$fp[$e_] := Round[N[{$e /. $p1, $e /. $p2}, 40], 10^-25];
$key = $fp /@ $Af;

Print[];
Print["[1] positions M calls EQUAL that the block calls DIFFERENT"];
Print["    (M merges them; the assembly does not)"];
$byLetter = GroupBy[Range[729], $Mf[[#]] &];
$split = {};
Do[With[{$L = $k, $pos = $byLetter[$k]},
   If[$L =!= 0 && Length[DeleteDuplicates[$key[[$pos]]]] > 1,
      AppendTo[$split, {$L, $pos}]]],
 {$k, Keys[$byLetter]}];
Print["    letters that the block splits: ", $split[[All, 1]]];
Do[With[{$L = $s[[1]], $pos = $s[[2]]},
   Print[];
   Print["    letter ", $L, "  (", Length[$pos], " positions) splits into ",
         Length[DeleteDuplicates[$key[[$pos]]]], " values:"];
   Do[With[{$grp = Select[$pos, $key[[#]] === $v &]},
      Print["      value class of size ", Length[$grp], " :"];
      Do[With[{$rc = {Quotient[$x - 1, 27] + 1, Mod[$x - 1, 27] + 1}},
         Print["        row ", $unups[$rc[[1]]], " (p,q,i)   col ",
               $unups[$rc[[2]]], " (r,s,j)"]],
       {$x, Take[$grp, UpTo[4]]}]],
    {$v, DeleteDuplicates[$key[[$pos]]]}]],
 {$s, Take[$split, UpTo[3]]}];

Print[];
Print["[2] positions the block calls EQUAL that M calls DIFFERENT"];
$byVal = GroupBy[Select[Range[729], $Mf[[#]] =!= 0 &], $key[[#]] &];
$merge = {};
Do[With[{$pos = $byVal[$k]},
   If[Length[DeleteDuplicates[$Mf[[$pos]]]] > 1,
      AppendTo[$merge, {DeleteDuplicates[$Mf[[$pos]]], $pos}]]],
 {$k, Keys[$byVal]}];
Print["    block values spanning several letters: ", Length[$merge]];
Do[Print["      letters ", $m[[1]], "  over ", Length[$m[[2]]], " positions"],
 {$m, Take[$merge, UpTo[6]]}];

Print[];
Print["=============================================================="];
Print["Read the index patterns above: a disagreement that is an index"];
Print["convention shows as a systematic relation between the merged"];
Print["positions' (p,q,i),(r,s,j), not as a scatter."];
Print["=============================================================="];
