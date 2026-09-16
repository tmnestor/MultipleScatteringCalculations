#!/usr/bin/env wolframscript
(* ==========================================================================
   STEP 2: DOES THE CLOSED SET CHANGE THE LEADING FAR FIELD?

   The prediction was "no change from T9", and the reason turns out to be
   structural rather than numerical.

   The middle equation of The Basic System (A33.nb cell 17) is

       (d_pr d_ij + w^2 drho N^r_ij,p - M_in,pk dc_nkrj) d_r u_j = d_p u0_i

   and it involves NOTHING ELSE.  A22 determines the first gradient d_u on its
   own: it does not couple to u, and it does not couple to d_d_u.  That is the
   parity decoupling -- u has degree 0, d_u degree 1, d_d_u degree 2, and even
   couples only to even.

   The leading modulus far field is the stress dipole Int dc : grad(u) dV,
   which needs only d_u.  Therefore A33 CANNOT change it.  Inverting A33
   exactly -- which is what A33InvVersion3.nb does -- affects the second
   gradients, and those reach the far field only through the O(k^2) coupling
   A13 . A33^-1 . d_d_u0 into the monopole channel.  The leading scattering
   amplitude is untouched.

   THE TEST.  T9 is the uniform-internal-strain (Eshelby) closure, so T9's
   modulus response IS A22 with drho = 0.  If the A22 assembled here from the
   moment engine reproduces the shear self-term of
   CubeT9FromFirstPrinciples.wl, then "the closed set's leading far field ==
   T9's" is an identity, not a coincidence -- and the Eshelby T-matrix cannot
   be improved at leading order by any amount of work on A33.

   WHAT WOULD FALSIFY IT.  A22 failing to reproduce S_shear, or A22 turning
   out to depend on the second gradients after all.  Both are checked.

   ⚠ N has D = 1, so the source-derivative convention contributes (-1)^1.
   M has D = 2 and is unaffected.  See CubeMomentCore.wl.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["A22: THE FIRST-GRADIENT BLOCK, AND THE LEADING FAR FIELD"];
Print["=============================================================="];

$ups2[$p_, $i_] := 3 ($i - 1) + $p;
$dc[$i_, $j_, $k_, $l_] :=
   dlam d[$i, $j] d[$k, $l] + dmu (d[$i, $k] d[$j, $l] + d[$i, $l] d[$j, $k]);

(* rank-4 cubic tensors: 4 invariants each, built from the form not
   componentwise *)
$Nf = cached["Nform", cubicTensorForm[
   Function[{$i, $n, $k, $r}, Nmom[$i, $n, $k, $r]], 4]];
$Mf = cached["Mform", cubicTensorForm[
   Function[{$i, $n, $p, $k}, Mmom[$i, $n, $p, $k]], 4]];
Print[];
Print["[1] invariants: N ", Length[$Nf[[1]]], "   M ", Length[$Mf[[1]]],
      "   (rank 4 -> 4 each)"];

$N[$i_, $j_, $p_, $r_] := $N[$i, $j, $p, $r] = reconstruct[$Nf, {$i, $j, $p, $r}];
$M[$i_, $n_, $p_, $k_] := $M[$i, $n, $p, $k] = reconstruct[$Mf, {$i, $n, $p, $k}];

Print[];
Print["[2] assembling A22 (9x9)"];
$A22 = Table[
   d[$p, $r] d[$i, $j] + om^2 drho $N[$i, $j, $p, $r]
   - Sum[$M[$i, $n, $p, $k] $dc[$n, $k, $r, $j], {$n, 3}, {$k, 3}],
   {$i, 3}, {$p, 3}, {$j, 3}, {$r, 3}];
$A22 = Table[$A22[[$i, $p, $j, $r]],
   {$i, 3}, {$p, 3}, {$j, 3}, {$r, 3}];
$A22m = Table[
   $A22[[Quotient[$a - 1, 3] + 1, Mod[$a - 1, 3] + 1,
         Quotient[$b - 1, 3] + 1, Mod[$b - 1, 3] + 1]],
   {$a, 9}, {$b, 9}];
Print["    done.  (row = (p,i) as 3(i-1)+p, column = (r,j))"];

Print[];
Print["[3] the STATIC modulus case, drho = 0 -- this is T9's regime"];
$A22s = Simplify[$A22m /. drho -> 0, Assumptions -> Del > 0];
Print["    distinct entries: ",
      Length[DeleteDuplicates[
        Round[N[Flatten[$A22s] /. {lam -> 7/3, mu -> 11/5, dlam -> 13/7,
              dmu -> 17/11}, 40], 10^-25]]]];

(* 9 = T1u (x) T1u = A1g + Eg + T1g + T2g, dimensions 1 + 2 + 3 + 3.
   The shear channels are Eg (deviatoric diagonal) and T2g (off-diagonal). *)
$e0 = Table[d[$p, $i]/Sqrt[3], {$i, 3}, {$p, 3}];                  (* A1g  *)
$eEg = {Table[(d[$p, 1] d[$i, 1] - d[$p, 2] d[$i, 2])/Sqrt[2], {$i, 3}, {$p, 3}],
        Table[(d[$p, 1] d[$i, 1] + d[$p, 2] d[$i, 2]
               - 2 d[$p, 3] d[$i, 3])/Sqrt[6], {$i, 3}, {$p, 3}]};  (* Eg  *)
$eT2g = Table[(d[$p, $u[[1]]] d[$i, $u[[2]]]
               + d[$p, $u[[2]]] d[$i, $u[[1]]])/Sqrt[2],
   {$u, {{1, 2}, {1, 3}, {2, 3}}}] /. $x_ :> $x;
$T2gv = Table[Flatten[Table[(d[$p, $u[[1]]] d[$i, $u[[2]]]
               + d[$p, $u[[2]]] d[$i, $u[[1]]])/Sqrt[2], {$i, 3}, {$p, 3}]],
   {$u, {{1, 2}, {1, 3}, {2, 3}}}];
$A1gv = {Flatten[$e0]};
$Egv = Flatten[#] & /@ $eEg;

proj[$vs_] := Module[{$v = $vs[[1]]},
   Simplify[($v . $A22s . $v)/($v . $v), Assumptions -> Del > 0]];

Print[];
Print["[4] the channel eigenvalues of A22 (static)"];
Print["    A1g (bulk)        : ", proj[$A1gv]];
Print["    Eg  (dev. shear)  : ", proj[$Egv]];
Print["    T2g (off-diag)    : ", proj[$T2gv]];

Print[];
Print["[5] CROSS-CHECK against CubeT9FromFirstPrinciples.wl"];
Print["    Its shear self-term is"];
Print["      S_shear = (Pi(lam+2mu) - Sqrt[3](lam+mu)) / (3 Pi mu (lam+2mu))"];
Print[];
Print["    Eg and T2g are DIFFERENT irreps and must not be expected to agree:"];
Print["    their difference is the cubic anisotropy the package already"];
Print["    carries as Dmu*_diag - Dmu*_off.  T9's single shear self-term is"];
Print["    the OFF-DIAGONAL one, so the identity to check is"];
Print[];
Print["        T2g channel  ==  1 + 2 dmu S_shear"];
$Sshear = (Pi (lam + 2 mu) - Sqrt[3] (lam + mu))/(3 Pi mu (lam + 2 mu));
$predict = 1 + 2 dmu $Sshear;
Print["    predicted : ", Simplify[$predict]];
Print["    T2g gives : ", proj[$T2gv]];
Print["    T2g == 1 + 2 dmu S_shear : ", zeroQ[proj[$T2gv] - $predict]];
$t2gOK = zeroQ[proj[$T2gv] - $predict];
Print[];
Print["    the DIAGONAL shear channel, for comparison -- write it as"];
Print["    Eg = 1 + 2 dmu S_diag and read off S_diag:"];
$Sdiag = Simplify[(proj[$Egv] - 1)/(2 dmu), Assumptions -> Del > 0];
Print["      S_diag  = ", $Sdiag];
Print["      S_shear = ", Simplify[$Sshear]];
Print["      they differ (cubic anisotropy): ", ! zeroQ[$Sdiag - $Sshear]];
Print[];
Print["    bulk channel, for completeness:"];
Print["      A1g = ", Simplify[proj[$A1gv]],
      "   = 1 + (3 dlam + 2 dmu)/(3(lam+2mu))"];
Print["      matches that closed form: ",
      zeroQ[proj[$A1gv] - (1 + (3 dlam + 2 dmu)/(3 (lam + 2 mu)))]];

Print[];
Print["[6] THE STRUCTURAL POINT, checked rather than asserted:"];
Print["    A22 depends on no second-gradient quantity.  Its entries are built"];
Print["    from N (D=1) and M (D=2) only -- Q and P appear nowhere -- so no"];
Print["    inverse of A33, however exact, can alter d_u."];
Print["    A22 free of Q and P: ",
      FreeQ[$A22m, Q] && FreeQ[$A22m, P]];
Print["    A22 involves only lam, mu, dlam, dmu, om, drho, Del: ",
      Complement[Cases[$A22m, $s_Symbol /; ! NumericQ[$s], Infinity] //
         DeleteDuplicates, {lam, mu, dlam, dmu, om, drho, Del, Pi, Sqrt}] === {}];

Print[];
Print["=============================================================="];
Print[If[$t2gOK,
  "PASS -- A22 reproduces T9's off-diagonal shear self-term exactly.\n" <>
  "        The closed set's leading modulus far field IS T9's, as an\n" <>
  "        identity: d_u is fixed by A22 alone, and A33 cannot reach it.",
  "FAIL -- A22 does not reproduce T9; the decoupling argument needs review."]];
Print["=============================================================="];
