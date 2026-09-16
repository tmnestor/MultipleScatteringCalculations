#!/usr/bin/env wolframscript
(* ==========================================================================
   THE P MOMENT OF A CUBE

       P^rs_ij,pq = Int_V d'_q d'_p (G0_ij(0,r')) x'_r x'_s dr'   (D,W) = (2,2)

   Two derivatives and two weights.  Where it appears (A33.nb, "The Basic
   System") -- it is the density half of the 27x27 block:

       A33 = (delta_qr delta_ps delta_ij + Q^r_in,pqk dc_nksj
                                         - (1/2) w^2 drho P^rs_ij,pq)

   so P carries the whole inertial response of the second-gradient block, and
   Q carries the whole elastic response.  Between them they are A33.

   SCALE.  P ~ Del^2.  (General rule for a scalar piece: Del^(m-D+W+3).  The
   delta_ij term has m=-1, D=2, W=2 -> 2; the d_i d_n term has m=1, D=4, W=2
   -> 2.  Both agree, as they must.)

   STATUS OF THE ARCHIVE.  Pmat.nb evaluates six scalar components of the
   m = -1 kernel in closed form; five of them can be traced unambiguously to
   their generating input cells and are reproduced here, component by
   component, against an engine that shares no code with it.  The sixth is
   excluded for a stated reason -- see the note above the reference list.

   The COMPLETE general tensor form of P is not in Pmat.nb at all.  It is in
   GreensTensorMoments6.nb, cell 28, alongside Qmat and the assembly of the
   27x27 block; CubeMomentArchiveCheck.wl does that comparison.  Beware also
   that Pmat.nb reuses the letters a..f later in the same notebook for a
   different tensor.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE P MOMENT OF A CUBE   P^rs_ij,pq = Int_V d_q d_p G0_ij x_r x_s dV"];
Print["=============================================================="];

Ps[i_, j_, p_, q_, r_, s_] := gStatic[i, j, {p, q}, {r, s}];

Print[];
Print["[1] CROSS-CHECK FIRST, against Pmat.nb (2012), scalar kernel m = -1."];
Print["    Pmat computes  Int_V x_r x_s d_i d_j d_p d_q (1/r) dV  directly,"];
Print["    with a per-component integration order.  This engine peels the"];
Print["    derivatives onto the faces instead.  No shared code."];
(* The third entry of Pmat.nb is DELIBERATELY OMITTED.  Its variable is named
   P_112323, which reads as (i,j,p,q,r,s) = (1,1,2,3,2,3), but the assignment
   line immediately above it sets r = 3; s = 3.  Label and assignment disagree,
   so there is no way to tell which index set the printed value belongs to, and
   a check against a value whose indices are unknown proves nothing either way.
   GreensTensorMoments6.nb gives that channel a non-zero coefficient, so the 0
   printed in Pmat.nb is more likely a mis-ordered output cell than a result.
   The complete comparison is in CubeMomentArchiveCheck.wl. *)
ref = {
  {"a = P_221111", {2, 2, 1, 1}, {1, 1}, (8/9) (5 Sqrt[3] - 3 Pi)},
  {"b = P_333131", {3, 3, 3, 1}, {3, 1}, -(4/9) (2 Sqrt[3] + 3 Pi)},
  {"d = P_112233", {1, 1, 2, 2}, {3, 3}, 4/(3 Sqrt[3])},
  {"e = P_111111", {1, 1, 1, 1}, {1, 1}, -(8/9) (10 Sqrt[3] + 3 Pi)},
  {"f = P_111122", {1, 1, 1, 1}, {2, 2}, -(4/9) (11 Sqrt[3] - 6 Pi)}};
nbad = 0;
Do[With[{lbl = t[[1]], ds = t[[2]], ws = t[[3]], want = t[[4]]},
   got = Simplify[E$[-1, ds, ws], Assumptions -> Del > 0];
   ok = zeroQ[got - want];
   If[! ok, nbad++];
   Print["    ", If[ok, "PASS", "FAIL"], "  ", lbl, "  = ", got,
         If[ok, "", "   archive: " <> ToString[want]]]], {t, ref}];
Print["    ", If[nbad == 0, "all five traceable values reproduce Pmat.nb",
                 ToString[nbad] <> " MISMATCH(ES) against Pmat.nb"]];

Print[];
Print["[2] the general cubic form, rank 6 (indices i,j,p,q,r,s)"];
form = cubicTensorForm[Ps, 6];
Print["    invariants in the basis: ", Length[form[[1]]], "  (expect 31)"];
Do[If[! zeroQ[form[[2, u]]],
      Print["    ", partLabel[form[[1, u]], {"i", "j", "p", "q", "r", "s"}],
            "\n        = ", Simplify[form[[2, u]], Assumptions -> Del > 0]]],
 {u, Length[form[[1]]]}];
Print["    (invariants with a vanishing coefficient are omitted)"];

Print[];
Print["[3] verification of the reconstruction on a random sample"];
SeedRandom[20260916];
sample = DeleteDuplicates[Table[RandomInteger[{1, 3}, 6], 45]];
bad = verifyForm[Ps, form, 6, sample];
Print["    sampled ", Length[sample], " components, mismatches: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[Take[bad, UpTo[5]]]]];

Print[];
Print["[4] index symmetries that must hold identically"];
Print["    P^rs_ij,pq == P^rs_ji,pq : ",
      zeroQ[Ps[1, 2, 1, 3, 2, 3] - Ps[2, 1, 1, 3, 2, 3]]];
Print["    P^rs_ij,pq == P^rs_ij,qp : ",
      zeroQ[Ps[1, 2, 1, 3, 2, 3] - Ps[1, 2, 3, 1, 2, 3]]];
Print["    P^rs_ij,pq == P^sr_ij,pq : ",
      zeroQ[Ps[1, 2, 1, 3, 2, 3] - Ps[1, 2, 1, 3, 3, 2]]];

Print[];
Print["[5] SCALE CHECK.  Every component must be exactly quadratic in Del."];
Do[Print["    P", q, "/Del^2 = ",
         Simplify[(Ps @@ q)/Del^2, Assumptions -> Del > 0],
         "   (free of Del: ", FreeQ[Simplify[(Ps @@ q)/Del^2], Del], ")"],
 {q, {{1, 1, 1, 1, 1, 1}, {1, 1, 2, 2, 3, 3}, {1, 2, 1, 2, 3, 3}}}];

Print[];
Print["[6] the dynamic moment, series through r^5"];
Print["    P^11_11,11 = ",
      Simplify[gDyn[1, 1, {1, 1}, {1, 1}], Assumptions -> Del > 0]];

Print["=============================================================="];
