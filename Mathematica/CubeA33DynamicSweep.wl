#!/usr/bin/env wolframscript
(* ==========================================================================
   WHICH TRUNCATION IS THE ARCHIVE'S 19-LETTER MATRIX?

   The static A33 assembled from the moment engine reproduces the archive's
   sixteen `rules` entries exactly, but its equality-partition of the 729
   positions has 22 classes against the 19 letters (20 classes) of the matrix
   M in A33.nb cell 2 -- and the two partitions CROSS: neither refines the
   other.  Three explanations were tested and two are dead:

     * M mis-transcribed          -- REFUTED, re-extracted from the cell boxes
                                     and compared: 0 differing positions.
     * accidental degeneracy at
       lam = mu, dlam = dmu       -- accounts for ONE class (22 -> 21), not
                                     the gap to 20.
     * M is the DYNAMIC block     -- the survivor.  Adding dynamic terms is
                                     not a pure refinement: entries that
                                     differ statically can coincide once the
                                     k^2, k^4, k^6 terms are added, which is
                                     the only mechanism that produces crossing
                                     partitions.

   WHY A SWEEP RATHER THAN ONE ASSEMBLY.  The archive truncates the two
   channels ASYMMETRICALLY (GreensTensorMoments6.nb cell 28):

       g_b :  1/r - Kb^2 r/2 + Kb^4 r^3/24                        -> m <= 3
       h   :  (Ka^2-Kb^2)/2 r + (Kb^4-Ka^4)/24 r^3
                                 + (Ka^6-Kb^6)/720 r^5            -> m <= 5

   Assembling at one guessed order and failing to match would not say whether
   the physics or merely the truncation differs.  Sweeping the order and
   reporting the class count at each is diagnostic either way: whichever
   reproduces M's 20 classes identifies the truncation, and if none does, the
   disagreement is structural and worth knowing about.

   The moments are shared across the sweep through E$'s memo and the on-disk
   cache, so this costs far less than one assembly per order.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["DYNAMIC A33: WHICH TRUNCATION REPRODUCES THE ARCHIVE'S 19?"];
Print["=============================================================="];

$ups[$p_, $q_, $i_] := 9 ($i - 1) + 3 ($q - 1) + $p;
$dc[$i_, $j_, $k_, $l_] :=
   dlam d[$i, $j] d[$k, $l] + dmu (d[$i, $k] d[$j, $l] + d[$i, $l] d[$j, $k]);

(* the two channels, truncated independently:
   ng = highest t kept in g_b,  nh = highest t kept in h  (m = t - 1)      *)
$gTrunc[$ng_] := Table[{$t - 1, (I kb)^$t/$t!}, {$t, 0, $ng}];
$hTrunc[$nh_] := Table[{$t - 1, ((I kb)^$t - (I ka)^$t)/($t! kb^2)},
                       {$t, 0, $nh}];

$Qdyn[$ng_, $nh_][$i_, $n_, $p_, $q_, $k_, $r_] :=
   srcFactor[3] (1/(4 Pi mu)) (
      d[$i, $n] seriesMoment[$gTrunc[$ng], {$p, $q, $k}, {$r}]
      + seriesMoment[$hTrunc[$nh], {$i, $n, $p, $q, $k}, {$r}]);

$Pdyn[$ng_, $nh_][$i_, $j_, $p_, $q_, $r_, $s_] :=
   srcFactor[2] (1/(4 Pi mu)) (
      d[$i, $j] seriesMoment[$gTrunc[$ng], {$p, $q}, {$r, $s}]
      + seriesMoment[$hTrunc[$nh], {$i, $j, $p, $q}, {$r, $s}]);

(* ---- the archive's letter matrix (A33.nb cell 2), verified transcription -- *)
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
$gM = GatherBy[Range[729], Flatten[$M][[#]] &];
$sortPart[$g_] := Sort[Sort /@ $g];
Print[];
Print["archive M: ", Length[$gM], " classes (19 letters + zero)"];

(* numeric fingerprint at two independent rational points, now including the
   dynamic parameters *)
$p1 = {lam -> 7/3, mu -> 11/5, dlam -> 13/7, dmu -> 17/11, drho -> 3/8,
       om -> 5/4, Del -> 9/7, ka -> 4/9, kb -> 7/6};
$p2 = {lam -> 5/2, mu -> 3/7, dlam -> 2/9, dmu -> 19/4, drho -> 7/9,
       om -> 2/5, Del -> 4/3, ka -> 3/11, kb -> 8/5};
$fp[$e_] := Round[N[{$e /. $p1, $e /. $p2}, 40], 10^-28];
$fpN[$pair_] := Round[$pair, 10^-25];

(* ⚠ TWO RULES, BOTH LEARNED THE HARD WAY IN THIS SESSION.

   1. BOTH TENSORS GO THROUGH THEIR 31-INVARIANT FORM, NEVER COMPONENTWISE.
      Q and P are rank-6 cubic invariants, so 31 coefficients determine all
      729 components.  Calling the moment once per index tuple means 729
      series evaluations per truncation and does not finish in 20 minutes.

   2. THE COEFFICIENTS ARE NUMERICISED BEFORE THE BLOCK IS BUILT.  Keeping
      them symbolic through `reconstruct` produces 729 large expressions,
      each then multiplied by nine dc terms -- that run was killed by the
      operating system for exhausting memory.

      Nothing is lost, because the question this file asks is which ENTRIES
      ARE EQUAL, and equality is decided by a numeric fingerprint at two
      independent rational parameter points.  The 31 symbolic coefficients
      are still computed once per truncation and cached; only the assembly is
      arithmetic.  Symbolic forms are then produced for the ONE truncation
      that matches, where they are actually wanted. *)

formQ[$ng_, $nh_] := cached["QformDyn" <> ToString[$ng] <> "x" <> ToString[$nh],
   cubicTensorForm[$Qdyn[$ng, $nh], 6]];
formP[$ng_, $nh_] := cached["PformDyn" <> ToString[$ng] <> "x" <> ToString[$nh],
   cubicTensorForm[$Pdyn[$ng, $nh], 6]];

(* numeric assembly at one parameter point *)
assembleN[$ng_, $nh_, $pt_] :=
  Module[{$fQ, $fP, $cQ, $cP, $QQ, $PP, $B, $dcN},
   $fQ = formQ[$ng, $nh]; $fP = formP[$ng, $nh];
   $cQ = {$fQ[[1]], N[$fQ[[2]] /. $pt, 40]};
   $cP = {$fP[[1]], N[$fP[[2]] /. $pt, 40]};
   $QQ[$i_, $n_, $p_, $q_, $k_, $r_] := $QQ[$i, $n, $p, $q, $k, $r] =
      reconstruct[$cQ, {$i, $n, $p, $q, $k, $r}];
   $PP[$i_, $j_, $p_, $q_, $r_, $s_] := $PP[$i, $j, $p, $q, $r, $s] =
      reconstruct[$cP, {$i, $j, $p, $q, $r, $s}];
   $dcN[$i_, $j_, $k_, $l_] := N[$dc[$i, $j, $k, $l] /. $pt, 40];
   $B = ConstantArray[0, {27, 27}];
   Do[$B[[$ups[$p, $q, $i], $ups[$r, $s, $j]]] =
        d[$q, $r] d[$p, $s] d[$i, $j]
        + Sum[$QQ[$i, $n, $p, $q, $k, $r] $dcN[$n, $k, $s, $j], {$n, 3}, {$k, 3}]
        - N[(1/2) om^2 drho /. $pt, 40] $PP[$i, $j, $p, $q, $r, $s],
    {$i, 3}, {$q, 3}, {$p, 3}, {$j, 3}, {$s, 3}, {$r, 3}];
   $B];

Print[];
Print["sweeping truncations  (ng = order kept in g_b, nh = in h;  m = t-1)"];
Print["  ng=0,nh=2 is the static limit;  ng=4,nh=6 is the archive's own"];
Print["  asymmetric truncation;  ng=6,nh=6 keeps both to r^5."];
Print[];
(* ⚠ ONE CASE PER PROCESS.  Running the whole sweep in a single kernel was
   killed twice by the operating system for exhausting memory: each
   cubicTensorForm holds its symbolic coefficients, and the deepest
   truncations carry the r^5 terms.  Held forms accumulate across cases and
   never get released.

   Passing ng and nh on the command line runs exactly one case in a fresh
   kernel, so memory is reclaimed between them.  The on-disk cache makes this
   nearly free -- a case whose forms are already stored is re-read rather than
   recomputed -- so the split costs process start-up and nothing else. *)
$argCase = If[Length[$ScriptCommandLine] >= 3,
   {ToExpression[$ScriptCommandLine[[2]]], ToExpression[$ScriptCommandLine[[3]]]},
   $None];
$cases = If[$argCase === $None,
   {{0, 2}, {2, 2}, {2, 4}, {4, 4}, {4, 6}, {6, 6}}, {$argCase}];

$results = {};
Do[With[{$ng = $case[[1]], $nh = $case[[2]]},
   Module[{$A1, $A2, $g, $ok},
     $A1 = Flatten[assembleN[$ng, $nh, $p1]];
     $A2 = Flatten[assembleN[$ng, $nh, $p2]];
     $g = GatherBy[Range[729], $fpN[{$A1[[#]], $A2[[#]]}] &];
     $ok = $sortPart[$g] === $sortPart[$gM];
     AppendTo[$results, {$ng, $nh, Length[$g], $ok}];
     Print["  ng=", $ng, " nh=", $nh, "   classes: ", Length[$g],
           "   matches M: ", $ok,
           If[$ok, "   <<< MATCH", ""]]]],
 {$case, $cases}];

Print[];
$win = Select[$results, #[[4]] &];
If[$win =!= {},
   Print["[MATCH] truncation ng=", $win[[1, 1]], " nh=", $win[[1, 2]],
         " reproduces the archive's 19 letters."];
   (* only now is symbolic work worth doing, and only on the 19 positions
      actually needed rather than on all 729 *)
   Module[{$fQ, $fP, $letters, $pos, $i, $q0, $p0, $j, $s0, $r0, $val},
     $fQ = formQ[$win[[1, 1]], $win[[1, 2]]];
     $fP = formP[$win[[1, 1]], $win[[1, 2]]];
     $letters = {a,b,c,d,e,f,g,h,i,j,k,l,m,n,p,q,r,s,t};
     Print["    reading them off (symbolic, 19 entries only):"];
     Do[$pos = FirstPosition[$M, $L];
        {$i, $q0, $p0} = {Quotient[$pos[[1]] - 1, 9] + 1,
            Quotient[Mod[$pos[[1]] - 1, 9], 3] + 1, Mod[$pos[[1]] - 1, 3] + 1};
        {$j, $s0, $r0} = {Quotient[$pos[[2]] - 1, 9] + 1,
            Quotient[Mod[$pos[[2]] - 1, 9], 3] + 1, Mod[$pos[[2]] - 1, 3] + 1};
        $val = d[$q0, $r0] d[$p0, $s0] d[$i, $j]
           + Sum[reconstruct[$fQ, {$i, $n, $p0, $q0, $k, $r0}]
                 $dc[$n, $k, $s0, $j], {$n, 3}, {$k, 3}]
           - (1/2) om^2 drho reconstruct[$fP, {$i, $j, $p0, $q0, $r0, $s0}];
        Print["      ", $L, " = ", pellSimplify[$val]],
      {$L, $letters}]],
   Print["[NO MATCH] no swept truncation reproduces M's partition."];
   Print["    class counts: ", $results[[All, {1, 2, 3}]]];
   Print["    M has ", Length[$gM], "."];
   Print["    That would make the disagreement structural rather than a"];
   Print["    matter of truncation order, and the next thing to check is the"];
   Print["    index convention of P in the assembly."]];

Print["=============================================================="];
