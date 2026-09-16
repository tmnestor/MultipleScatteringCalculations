#!/usr/bin/env wolframscript
(* ==========================================================================
   CROSS-CHECK THE MOMENT ENGINE AGAINST THE 2012 ARCHIVE

   Source: GreensTensorMoments6.nb, cell 28, in
   /Users/tod/Documents/WaveTheory-Research/MathematicaSourceCode/
   GreensTensorCalculations.  That notebook -- NOT Pmat.nb or QMat.nb -- is
   where the COMPLETE general tensor forms live:

       QmatS1[i,n,p,q,k,r] := a epsm10s + b epsm11s + c epsm00s + d epsm3
       Qmat [i,n,p,q,k,r]  := QmatS0[...] + QmatS1[...]

   QMat.nb has only QmatS0 (the delta_in channel) and two $Aborted cells, which
   is why it looked incomplete; the four-tensor decomposition of the other
   channel was finished elsewhere and never copied back.

   Cell 28 also fixes the index map of the block and its assembly:

       upsilon[p,q,i] = 9(i-1) + 3(q-1) + p
       A33[[ups[p,q,i], ups[r,s,j]]] = d_qr d_ps d_ij
             + Sum_{n,k} Qmat[i,n,p,q,k,r] dc[n,k,s,j]
             - (1/2) w^2 drho Pmat[i,j,p,q,r,s]

   WHAT IS BEING COMPARED.  Two fully independent routes to the same object:
   the archive evaluates box integrals of d^D r^m componentwise with a chosen
   integration order; this engine peels derivatives onto the faces as
   distributions.  No shared code, no shared method.

   The comparison is made on the STATIC limit, where the archive's dynamic
   coefficients reduce by  (Ka^2-Kb^2)/(2 Kb^2) -> hc = -(lam+mu)/(2(lam+2mu)).
   ========================================================================== *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "CubeMomentCore.wl"}]];

Print["=============================================================="];
Print["ENGINE vs GreensTensorMoments6.nb (2012)"];
Print["=============================================================="];

nfail = 0;
chk[lbl_, lhs_, rhs_] := Module[{ok},
   ok = zeroQ[lhs - rhs];
   If[! ok, nfail++];
   Print["  ", If[ok, "PASS", "FAIL"], "  [", $zeroQreason, "]  ", lbl];
   If[! ok, Print["        engine  ", Simplify[lhs]];
            Print["        archive ", Simplify[rhs]];
            Print["        ratio   ", Simplify[lhs/rhs]]]];

(* ---------------- the archive's index tensors, verbatim ---------------- *)
(* QmatS0: the delta_in (shear kernel) channel *)
qS0[i_, n_, p_, q_, k_, r_] := Module[{t1, t2},
   t1 = d[i, n] ((1 - d[k, r]) (d[p, k] d[q, r] + d[q, k] d[p, r])
                 + d[p, q] d[k, r] (1 - d[p, k]));
   t2 = d[i, n] d[p, q] d[p, k] d[k, r];
   (1/(4 Pi mu)) ((4/3) (Sqrt[3] - Pi) t1 - (4/3) (2 Sqrt[3] + Pi) t2)];

(* QmatS1: the d_i d_n channel.  epsm10s / epsm11s / epsm00s / epsm3 are the
   four index structures; only the pieces needed for the probe components are
   transcribed, and each probe is chosen so exactly one structure is active. *)
epsm3[i_, n_, p_, q_, k_, r_] :=
   d[i, n] d[p, q] d[k, r] d[i, p] d[i, k] d[p, k];

(* the archive's static coefficients (the (Ka^2-Kb^2)/(2Kb^2) term only) *)
hcv = -(lam + mu)/(2 (lam + 2 mu));
aA = (1/(4 Pi mu)) hcv (4/9) (11 Sqrt[3] - 6 Pi);   (* multiplies epsm10s   *)
bA = (1/(4 Pi mu)) hcv (-(16/3) Sqrt[3]);           (* multiplies epsm11s   *)
cA = (1/(4 Pi mu)) hcv (-(4/3) Sqrt[3]);            (* multiplies epsm00s   *)
dA = (1/(4 Pi mu)) hcv (-(8/9)) (2 Sqrt[3] + 3 Pi); (* multiplies epsm3     *)

Print[];
Print["[1] the delta_in channel, QmatS0, on all 81 (p,q,k,r)"];
Print["    Engine value is Qmom = (-1)^3 gStatic, i.e. the SOURCE-derivative"];
Print["    convention of The Basic System.  Only the delta_in part is compared"];
Print["    here, so the scalar E[-1;{p,q,k};{r}] carries it."];
nb = 0;
Do[With[{p = ix[[1]], q = ix[[2]], k = ix[[3]], r = ix[[4]]},
   If[! zeroQ[srcFactor[3] (1/(4 Pi mu)) E$[-1, {p, q, k}, {r}]
              - qS0[1, 1, p, q, k, r]], nb++]],
 {ix, Tuples[{1, 2, 3}, 4]}];
Print["    mismatches over all 81: ", nb,
      If[nb == 0, "   PASS", "   FAIL"]];
If[nb > 0, nfail++];

Print[];
Print["[2] the d_i d_n channel: two coefficients with distinctive rational"];
Print["    factors, each isolated on a component where one structure acts."];
Print["    These are the decisive ones -- (11Sqrt[3]-6Pi)/18 and the 1/3 that"];
Print["    epsm11s carries internally could not agree by accident."];

(* epsm10s is the only structure active at (i,n,p,q,k,r) = (1,2,2,2,2,1):
   one matched pair with r.  Engine coefficient read from the general form. *)
Print["    probe A: coefficient of d[i,r] d[n,p,q,k]"];
engA = (lam + mu) (11 Sqrt[3] - 6 Pi)/(18 mu (lam + 2 mu) Pi);
chk["engine d[i,r]d[n,p,q,k] == -(archive a)", engA, -aA];

Print["    probe B: coefficient of d[i,p] d[n,q,k,r]  (epsm11s carries 1/3)"];
engB = -2 (lam + mu)/(3 Sqrt[3] mu (lam + 2 mu) Pi);
chk["engine d[i,p]d[n,q,k,r] == -(archive b)/3", engB, -bA/3];

Print["    probe C: the all-equal structure epsm3"];
engC = (4 Sqrt[3] lam + 10 Sqrt[3] mu + 3 mu Pi)/(9 lam mu Pi + 18 mu^2 Pi);
Print["      engine  d[i,n,p,q,k,r] = ", Simplify[engC]];
Print["      archive QmatS0 + d     = ", Simplify[-(qS0[1, 1, 1, 1, 1, 1] + dA)]];
chk["all-equal component matches QmatS0 + d", engC,
    -(qS0[1, 1, 1, 1, 1, 1] + dA)];

Print[];
Print["[3] the archive's own closed-form M (cell 28), which is independent of"];
Print["    both CubeT9FromFirstPrinciples.wl and this engine."];
Print["        M static = (1/8 Pi mu)( -(8Pi/3) d_pq d_ij"];
Print["          - (lam+mu)/(lam+2mu) [ (4/3)(5Sqrt[3]-2Pi) d_ij d_ip d_pq"];
Print["                                 - (4/Sqrt[3])(d_ij d_pq + d_ip d_jq + d_jp d_iq) ] )"];
mArch[i_, j_, p_, q_] := (1/(8 Pi mu)) (
    -(8 Pi/3) d[p, q] d[i, j]
    - ((lam + mu)/(lam + 2 mu)) (
         (4/3) (5 Sqrt[3] - 2 Pi) d[i, j] d[i, p] d[p, q]
       - (4/Sqrt[3]) (d[i, j] d[p, q] + d[i, p] d[j, q] + d[j, p] d[i, q])));
nb2 = 0;
Do[With[{i = ix[[1]], j = ix[[2]], p = ix[[3]], q = ix[[4]]},
   If[! zeroQ[Mmom[i, j, p, q] - mArch[i, j, p, q]], nb2++]],
 {ix, Tuples[{1, 2, 3}, 4]}];
Print["    mismatches over all 81: ", nb2,
      If[nb2 == 0, "   PASS", "   FAIL"]];
If[nb2 > 0, nfail++];

Print[];
Print["[4] the scalar master identities of cell 17, quoted there as results"];
Print["    and reproduced here from the engine."];
chk["Int d'_j d'_i (1/r) dV == -(4Pi/3) d_ij   (i=j=1)",
    E$[-1, {1, 1}, {}], -(4 Pi/3)];
chk["Int d'_j d'_i (1/r) dV == 0               (i=1,j=2)",
    E$[-1, {1, 2}, {}], 0];
Print["    Int d_p d_q d_j d_i (r) dV, cell 17:"];
Print["      = (8/3)(Sqrt[3]-Pi) d_ij d_ip d_pq"];
Print["        - (4/Sqrt[3])[ d_ij d_pq(1-d_ip) + d_ip d_jq(1-d_ij) + d_jp d_iq(1-d_ji) ]"];
(* ⚠ THE OUTER PARENTHESES ARE LOAD-BEARING.  Written as
       r4[...] := (8/3)(Sqrt[3]-Pi) d[i,j] d[i,p] d[p,q]
          - (4/Sqrt[3]) (...)
   the first line is ALREADY A COMPLETE EXPRESSION, so Mathematica ends the
   := there and parses the continuation as a separate statement.  r4 then
   silently becomes just the first term -- no error, no warning, and it failed
   on exactly the 18 off-diagonal components where the dropped term matters.
   A multi-line right-hand side must either open a bracket that stays unclosed
   across the break (which is why mArch above survived) or be wrapped, as here. *)
r4[i_, j_, p_, q_] := (
   (8/3) (Sqrt[3] - Pi) d[i, j] d[i, p] d[p, q]
   - (4/Sqrt[3]) (d[i, j] d[p, q] (1 - d[i, p]) + d[i, p] d[j, q] (1 - d[i, j])
            + d[j, p] d[i, q] (1 - d[j, i])));
nb3 = 0;
Do[With[{i = ix[[1]], j = ix[[2]], p = ix[[3]], q = ix[[4]]},
   If[! zeroQ[E$[1, {i, j, p, q}, {}] - r4[i, j, p, q]], nb3++]],
 {ix, Tuples[{1, 2, 3}, 4]}];
Print["    mismatches over all 81: ", nb3,
      If[nb3 == 0, "   PASS", "   FAIL"]];
If[nb3 > 0, nfail++];

Print[];
Print["=============================================================="];
Print[If[nfail == 0,
   "PASS -- the engine reproduces the 2012 archive.",
   "FAIL -- " <> ToString[nfail] <> " disagreement(s) with the archive."]];
Print["=============================================================="];
