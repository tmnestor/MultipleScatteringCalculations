#!/usr/bin/env wolframscript
(* ==========================================================================
   THE Q MOMENT OF A CUBE

       Q^r_in,pqk = Int_V d'_q d'_p d'_k (G0_in(0,r')) x'_r dr'   (D,W) = (3,1)

   Three derivatives, one weight -- the deepest of the six.  Where it appears
   (A33.nb, "The Basic System"):

       A33 = (delta_qr delta_ps delta_ij + Q^r_in,pqk dc_nksj
                                         - (1/2) w^2 drho P^rs_ij,pq)

   so Q carries the ENTIRE elastic response of the 27x27 second-gradient
   block.  Connecting the block's nineteen entries a..t to the moment
   integrals is exactly the job of this file plus CubePMoment.wl.

   WHY IT IS THE DEEPEST.  The d_i d_n inside the Green's tensor adds two, so
   the h-channel needs FIVE derivatives of r.  That is the case where a
   spherical excision and the distributional value differ, and where direct
   box integration is least trustworthy.  The engine peels derivatives onto
   the faces, where every one of them is bounded.  See CubeMomentCore.wl.

   SCALE.  Q ~ Del^0, independent of cube size.  (delta_in term: m=-1, D=3,
   W=1 -> Del^0; d_i d_n term: m=1, D=5, W=1 -> Del^0.)

   STATUS OF THE ARCHIVE.  QMat.nb is the INCOMPLETE one of the two: two of
   its cells read $Aborted and it records no general tensor form.  What it does
   contain, and what is used here as an independent cross-check, is QmatS0 --
   the delta_in channel, with a full tensor form and dynamic terms through
   K_beta^4.  Its static part is checked below on all 81 components of
   (p,q,k,r), not just on the handful the notebook printed.
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["THE Q MOMENT OF A CUBE   Q^r_in,pqk = Int_V d_q d_p d_k G0_in x_r dV"];
Print["=============================================================="];

(* the engine value in FIELD derivatives; Qmom in the core applies the
   (-1)^3 of the source convention.  Both are reported below. *)
Qs[i_, n_, p_, q_, k_, r_] := gStatic[i, n, {p, q, k}, {r}];

Print[];
Print["[1] CROSS-CHECK against GreensTensorMoments6.nb (2012)."];
Print["    NOTE: the complete Qmat is NOT in QMat.nb -- that notebook has only"];
Print["    the delta_in channel plus two $Aborted cells.  The finished form,"];
Print["    Qmat = QmatS0 + (a epsm10s + b epsm11s + c epsm00s + d epsm3), is in"];
Print["    GreensTensorMoments6.nb cell 28.  The full comparison lives in"];
Print["    CubeMomentArchiveCheck.wl; what is checked here is the convention."];
Print[];
Print["    THE SOURCE-DERIVATIVE SIGN.  The Basic System defines Q with"];
Print["    derivatives w.r.t. the SOURCE point, d' = -d, so with D = 3"];
Print["    derivatives the archive value is MINUS this engine's field-"];
Print["    derivative value.  M (D = 2) is unaffected, which is what makes the"];
Print["    diagnosis specific rather than a fudge: a genuine engine error"];
Print["    would not respect the parity of D."];
(* ⚠ outer parentheses required -- see the note in CubeMomentArchiveCheck.wl.
   Without them the := ends at the close of the first term and the "- ..."
   line becomes a separate statement, silently dropping it.  That failure is
   invisible except on the three all-equal components, where the dropped term
   is the only one that contributes. *)
qref[p_, q_, k_, r_] := (
   (4/3) (Sqrt[3] - Pi) ((1 - d[k, r]) (d[p, k] d[q, r] + d[q, k] d[p, r])
                          + d[p, q] d[k, r] (1 - d[p, k]))
   - (4/3) (2 Sqrt[3] + Pi) d[p, q] d[p, k] d[k, r]);
nbad = 0;
Do[With[{p = ix[[1]], q = ix[[2]], k = ix[[3]], r = ix[[4]]},
   If[! zeroQ[srcFactor[3] E$[-1, {p, q, k}, {r}] - qref[p, q, k, r]],
      nbad++]],
 {ix, Tuples[{1, 2, 3}, 4]}];
Print["    delta_in channel, all 81 components of (p,q,k,r): mismatches ",
      nbad, If[nbad == 0, "   PASS", "   FAIL"]];

Print[];
Print["[2] the general cubic form, rank 6 (indices i,n,p,q,k,r)"];
form = cubicTensorForm[Qs, 6];
Print["    invariants in the basis: ", Length[form[[1]]], "  (expect 31)"];
Do[If[! zeroQ[form[[2, u]]],
      Print["    ", partLabel[form[[1, u]], {"i", "n", "p", "q", "k", "r"}],
            "\n        = ", Simplify[form[[2, u]], Assumptions -> Del > 0]]],
 {u, Length[form[[1]]]}];
Print["    (invariants with a vanishing coefficient are omitted)"];

Print[];
Print["[3] verification of the reconstruction on a random sample"];
SeedRandom[20260916];
sample = DeleteDuplicates[Table[RandomInteger[{1, 3}, 6], 45]];
bad = verifyForm[Qs, form, 6, sample];
Print["    sampled ", Length[sample], " components, mismatches: ", Length[bad],
      If[bad === {}, "   PASS", "   FAIL " <> ToString[Take[bad, UpTo[5]]]]];

Print[];
Print["[4] index symmetries that must hold identically"];
Print["    d_p d_q d_k is totally symmetric, G0_in is symmetric in (i,n)."];
Print["    Q swap p,q : ",
      zeroQ[Qs[1, 2, 1, 2, 3, 1] - Qs[1, 2, 2, 1, 3, 1]]];
Print["    Q swap p,k : ",
      zeroQ[Qs[1, 2, 1, 2, 3, 1] - Qs[1, 2, 3, 2, 1, 1]]];
Print["    Q swap i,n : ",
      zeroQ[Qs[1, 2, 1, 2, 3, 1] - Qs[2, 1, 1, 2, 3, 1]]];

Print[];
Print["[5] parity.  D + W = 4 is even, so odd-multiplicity components vanish."];
Do[Print["    Q", ix, " = ", Simplify[Qs @@ ix]],
 {ix, {{1, 1, 1, 1, 1, 2}, {1, 1, 1, 2, 3, 3}}}];

Print[];
Print["[6] SCALE CHECK.  Q must be free of Del entirely."];
Do[Print["    Q", ix, " = ", Simplify[Qs @@ ix, Assumptions -> Del > 0],
         "   (free of Del: ", FreeQ[Simplify[Qs @@ ix], Del], ")"],
 {ix, {{1, 1, 2, 2, 1, 1}, {1, 1, 1, 2, 1, 2}, {1, 2, 1, 3, 2, 3}}}];

Print[];
Print["[7] the independent components, in closed form"];
comps = {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 2, 2}, {1, 1, 2, 2, 1, 1},
         {1, 2, 1, 2, 1, 1}, {1, 1, 1, 2, 1, 2}, {1, 2, 1, 1, 2, 1},
         {1, 1, 2, 2, 3, 3}, {1, 2, 1, 3, 2, 3}, {1, 1, 1, 1, 3, 3}};
Do[Print["    Q", c, " = ", Simplify[Qs @@ c, Assumptions -> Del > 0]],
 {c, comps}];

Print[];
Print["[8] the dynamic moment, series through r^5"];
Print["    Q^1_11,111 = ",
      Simplify[gDyn[1, 1, {1, 1, 1}, {1}], Assumptions -> Del > 0]];

Print["=============================================================="];
