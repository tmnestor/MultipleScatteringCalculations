#!/usr/bin/env wolframscript
(* ==========================================================================
   THE MOMENT ENGINE -- shared core for CubeGMoment, CubeKMoment, CubeNMoment,
   CubeMMoment, CubePMoment, CubeQMoment.

   ---------------------------------------------------------------------------
   THE BASIC SYSTEM (A33.nb, cell 17, verbatim).  Six moments of the
   elastodynamic Green's tensor over the cube, all about its centre:

       G_ij        = Int G0_ij(0,r') dr'
       K^rs_ij     = Int G0_ij(0,r') x'_r x'_s dr'
       N^r_in,k    = Int d'_k (G0_in) x'_r dr'
       M_in,pk     = Int d'_p d'_k (G0_in) dr'
       P^rs_ij,pq  = Int d'_q d'_p (G0_ij) x'_r x'_s dr'
       Q^r_in,pqk  = Int d'_q d'_p d'_k (G0_in) x'_r dr'

   They are graded by (D, W) = (derivatives on G, degree of the x-weight):

       G (0,0)   K (0,2)   N (1,1)   M (2,0)   P (2,2)   Q (3,1)

   D + W is even in every one of the six.  That is the parity rule that makes
   the closed set close: the A22 block (odd) decouples from A11/A13/A31/A33.

   ---------------------------------------------------------------------------
   THE REDUCTION TO SCALAR KERNELS.  The elastodynamic tensor is

       G0_in = (1/(4 Pi mu)) [ delta_in g_b(r) + (1/kb^2) d_i d_n (g_b - g_a) ]

   with g_k(r) = Exp[I k r]/r.  VERIFIED against Kupradze in the static limit:
   it returns a0 = (lam+3mu)/(8 Pi mu (lam+2mu)), b0 = (lam+mu)/(8 Pi mu
   (lam+2mu)), which are (1/8Pi)(1/mu +- 1/(lam+2mu)).

   Expanding g_k = Sum_t (I k)^t r^(t-1)/t! turns every one of the six moments
   into a finite sum of SCALAR moments

       E[m; d1..dD; w1..wW] = Int_V x_w1 .. x_wW  d_d1 .. d_dD  r^m  dV ,
       V = [-Del/2, Del/2]^3 .

   The tensor's own d_i d_n adds two to D on the second term -- which is why
   Pmat.nb computes FOUR derivatives for a moment defined with two.

   ---------------------------------------------------------------------------
   HOW E IS COMPUTED, AND WHY NOT BY DIRECT Integrate.

   Direct integration of d^D r^m over the box is unsafe twice over.  Mathematica
   returns Undefined on the all-at-once box integral unless the variable order
   happens to suit it (Pmat.nb varies the order component by component, which
   is the symptom, not a fix).  Worse, for m = -1 the classical integrand is
   only CONDITIONALLY convergent: d^4 (1/r) is a harmonic (traceless) tensor of
   degree -5, so its angular average vanishes and the radial divergence is
   multiplied by zero.  The value then depends on the shape of the exclusion --
   this is precisely the Eshelby delta-function term that naive quadrature
   misses and gets the sign wrong without.

   The derivatives here are therefore DISTRIBUTIONAL, and E is evaluated by
   pairing against the test function w.1_V rather than by integrating a
   function.  Writing T' = d_rest r^m, and using d_q 1_V = -n_q delta_dV,

       <d_q T', w 1_V> = -<T', (d_q w) 1_V> - <T', w d_q 1_V>
                       =  Sur_dV n_q w T' dA  -  <T', (d_q w) 1_V> .

   That is the whole engine:

       E[m; {q}+rest; w] = outerFace(rest, w, q)
                           - Sum_u [w_u == q] E[m; rest; w minus u] .

   Two things make it exact rather than merely plausible:

   * T' is only ever evaluated ON THE BOUNDARY, at distance Del/2 from the
     origin, where every derivative of r^m is smooth and bounded.  The singular
     pointwise structure of d^D r^m at the origin is never formed, so there is
     no principal value to take and no delta to insert by hand;
   * the recursion strictly lowers both D and W, terminating at D = 0 on
     Int_V x_w r^m dV, an ordinary convergent integral for m >= -1.

   The delta term is not added and cannot be dropped -- it is already inside
   the outer surface integral.  The Laplacian sum rule
   Sum_p E[-1;{p,p};] == -4 Pi is the sharp test of that, and it is checked in
   CubeMomentCoreTest.wl along with its weighted and biharmonic analogues.

   An EXCISED variant is also provided (excisedE) for diagnosis only: it
   subtracts the inner-sphere term and so returns the delta-free principal
   value.  The difference between the two IS the delta content, which is how
   the engine reports it.  excisedE is never used to build a moment.

   ---------------------------------------------------------------------------
   CONVENTIONS.  Del is the FULL width (cube [-Del/2, Del/2]^3), matching
   A33.nb / Pmat.nb / QMat.nb.  Note that CubeT9FromFirstPrinciples.wl uses the
   HALF-width a; translate with Del = 2a.

   NAME COLLISION.  Pmat.nb uses a..f for P COMPONENTS and QMat.nb a.. for Q
   components, while A33.nb uses a..t for the ENTRIES of the 27x27 system.
   Same letters, unrelated meanings.  Nothing here is named a..t.

   VARIABLE HYGIENE.  Every integration variable is localised.  The parameters
   of this system are single letters (i, j, k, m, n, p, q, r, s, ...) and a
   clobbered integration variable does NOT error -- Integrate silently receives
   a number where it expects a symbol and returns something that looks like a
   result.  Never introduce a global loop variable in a file that Gets this.
   ========================================================================== *)

$MomentCoreLoaded = True;
$MaxExtraPrecision = 400;

(* ==========================================================================
   DECIDING WHETHER TWO CLOSED FORMS ARE EQUAL

   Simplify[a - b] === 0 is NOT a usable test here.  These moments come out
   spelled in Log, ArcCoth and ArcSinh forms of the same algebraic constants
   -- Log[26 - 15 Sqrt[3]] and -6 ArcCoth[Sqrt[3]] are the same number -- and
   Simplify reconciles them only sometimes.  Using it as the predicate reports
   identities that hold to 78 digits as failures, which is how the K and N
   reconstructions first came back with "mismatches" that were not there.

   A Simplify that does not reach zero means UNPROVEN, not unequal.  So the
   test escalates: Simplify, then FullSimplify, then numerically at 120-digit
   working precision over three independent random parameter points, judged
   RELATIVE to the size of the terms being cancelled (an absolute threshold
   would pass anything small).  Only a value that survives all three is called
   non-zero.
   ========================================================================== *)

momentSyms = {Del, lam, mu, ka, kb};

numericZero[e_] := AllTrue[{11, 23, 41}, Function[sd,
   SeedRandom[sd];
   Module[{pt, val, scale},
     pt = Thread[momentSyms -> RandomReal[{1/2, 3}, Length[momentSyms],
            WorkingPrecision -> 120]];
     val = N[e /. pt, 120];
     scale = N[Total[Abs /@ Level[Expand[e] /. pt, {1}]] + 1, 120];
     TrueQ[Abs[val]/scale < 10^-60]]]];

(* True when e is zero; the reason is recorded in $zeroQreason.

   ORDER MATTERS FOR COST, NOT FOR CORRECTNESS.  The numeric test is tried
   FIRST because it is milliseconds, while FullSimplify on these log-heavy
   expressions can take minutes each and there are hundreds of them -- putting
   the symbolic tiers first made a full verification run over an hour.
   Symbolic proof is then attempted only for the ones numerics says are NOT
   zero, which is where a definite answer actually matters. *)
zeroQ[e_] := Module[{s},
   If[e === 0 || e === 0., $zeroQreason = "exact"; Return[True]];
   If[Quiet[Check[numericZero[e], False]],
      $zeroQreason = "numeric"; Return[True]];
   s = Quiet[Simplify[e, Assumptions -> Del > 0]];
   If[s === 0, $zeroQreason = "Simplify"; Return[True]];
   If[Quiet[Check[numericZero[s], False]],
      $zeroQreason = "numeric after Simplify"; Return[True]];
   If[Quiet[TimeConstrained[
        FullSimplify[s, Assumptions -> Del > 0], 60]] === 0,
      $zeroQreason = "FullSimplify"; Return[True]];
   $zeroQreason = "NONZERO"; False];

(* ---------------- geometry and symbols ---------------- *)
X  = {x, y, z};
rr = Sqrt[x^2 + y^2 + z^2];
hw = Del/2;                     (* half-width; Del is the FULL width *)
asm = {Del > 0, Element[{x, y, z}, Reals]};
d[i_, j_] := KroneckerDelta[i, j];

(* weight monomial and its kernel derivative *)
wt[w_List] := Times @@ (X[[#]] & /@ w);
dk[m_, ds_List] := dk[m, Sort[ds]] =
   Fold[D[#1, X[[#2]]] &, rr^m, Sort[ds]];

(* ---------------------------------------------------------------------------
   ONE VARIABLE AT A TIME.  An all-at-once multiple Integrate over the box
   returns Undefined for the higher kernels -- it is not an error, it is a
   symbol, and it propagates silently into whatever is built on top of it.
   (That is how G_11 first came back as "Undefined" from the r^5 term of the
   dynamic series.)  Integrating one variable at a time, carrying the range of
   the OUTER variables as assumptions and switching off GenerateConditions,
   is what makes it evaluate.  Pmat.nb works around the same problem by
   choosing a different variable order per component; this is the general fix.
   --------------------------------------------------------------------------- *)
seqInt[e_, vars_List, lo_, hi_] := Module[{acc = e, u},
   Do[u = vars[[t]];
      acc = Integrate[acc, {u, lo, hi},
         Assumptions -> Join[{Del > 0},
           (lo <= # <= hi) & /@ Drop[vars, t]],
         GenerateConditions -> False], {t, Length[vars]}];
   acc];

(* ---------------- D = 0 : the plain volume integral ---------------- *)
(* Int_V (prod x_w) r^m dV.  Zero by parity unless every axis appears an even
   number of times; otherwise eight times the positive-octant integral. *)
volInt[m_, w_List] := volInt[m, Sort[w]] = Module[{fx, uu, vv, ww},
   If[AnyTrue[Count[w, #] & /@ {1, 2, 3}, OddQ], Return[0]];
   fx = (wt[w] rr^m) /. {x -> uu, y -> vv, z -> ww};
   8 seqInt[fx, {ww, vv, uu}, 0, hw]];

(* ---------------- outer faces: the two planes normal to axis q ---------- *)
outerFace[m_, rest_List, w_List, q_] := outerFace[m, Sort[rest], Sort[w], q] =
  Module[{F, o1, o2, uu, vv},
   F = dk[m, rest] wt[w];
   {o1, o2} = Complement[{1, 2, 3}, {q}];
   seqInt[
     (F /. {X[[q]] ->  hw, X[[o1]] -> uu, X[[o2]] -> vv}) -
     (F /. {X[[q]] -> -hw, X[[o1]] -> uu, X[[o2]] -> vv}),
     {vv, uu}, -hw, hw]];

(* ---------------- inner sphere: DIAGNOSTIC ONLY ------------------------- *)
(* Sur_{|r|=eps} rhat_q (prod x_w) (d^rest r^m) dA, as eps -> 0.  This is the
   flux of the singular part through a vanishing sphere, i.e. exactly the delta
   content that a spherical excision would throw away.  It is NOT part of E --
   subtracting it is what produced 0 instead of -4 Pi for the Laplacian sum
   rule.  It is retained so that excisedE can report the delta content by
   difference.  Classified by exponent AND angular average: a negative exponent
   with a vanishing angular average is still zero. *)
innerSphere[m_, rest_List, w_List, q_] :=
  innerSphere[m, Sort[rest], Sort[w], q] =
  Module[{nv, th, ph, ep, expr, ang, pw},
   nv = {Sin[th] Cos[ph], Sin[th] Sin[ph], Cos[th]};
   expr = nv[[q]] wt[w] dk[m, rest];
   expr = expr /. Thread[X -> ep nv];
   (* eps^2 from dA, Sin[th] from dOmega *)
   expr = PowerExpand[Simplify[expr ep^2 Sin[th],
       Assumptions -> {ep > 0, 0 < th < Pi}]];
   If[expr === 0, Return[0]];
   pw = Exponent[expr, ep];
   ang = Integrate[Simplify[expr/ep^pw], {th, 0, Pi}, {ph, 0, 2 Pi}];
   ang = Simplify[ang];
   Which[
     ang === 0 || ang == 0, 0,                 (* angular average kills it *)
     pw > 0, 0,                                (* vanishes with the sphere *)
     pw == 0, ang,                             (* THE DELTA TERM -- keep it *)
     True, (Print["  !! DIVERGENT inner-sphere term: m=", m, " rest=", rest,
                  " w=", w, " q=", q, "  eps^", pw, " * ", ang];
            Abort[])]];

(* ---------------- the scalar moment E, by recursion on D ---------------- *)
(* E[m; ds; ws] = < d_ds r^m , (prod x_ws) 1_V >, distributionally.
   The delta content is inside outerFace; nothing is added or removed here. *)
(* Nothing unevaluated is allowed through.  Integrate returns the SYMBOL
   Undefined rather than an error when it gives up on the box, and an
   unevaluated Integrate or a ConditionalExpression is just as poisonous --
   each looks like a result and propagates silently into everything built on
   it.  Fail here, naming the moment, rather than three files downstream. *)
assertEvaluated[res_, m_, ds_, w_] := (
   If[! FreeQ[res, Undefined] || ! FreeQ[res, Integrate] ||
      ! FreeQ[res, ConditionalExpression] || ! FreeQ[res, Indeterminate],
      Print["  !! UNEVALUATED MOMENT  m=", m, "  ds=", ds, "  w=", w];
      Print["     got: ", res];
      Abort[]];
   res);

E$[m_, ds_List, w_List] := E$[m, Sort[ds], Sort[w]] =
  Module[{q, rest},
   If[ds === {}, Return[assertEvaluated[volInt[m, w], m, ds, w]]];
   q = First[ds]; rest = Rest[ds];
   assertEvaluated[
     Simplify[
       outerFace[m, rest, w, q]
       - Sum[If[w[[u]] === q, E$[m, rest, Drop[w, {u}]], 0], {u, Length[w]}],
       Assumptions -> Del > 0], m, ds, w]];

(* Diagnostic twin: the spherically-excised (delta-free) principal value.
   E$ - excisedE is the delta content of the moment. *)
excisedE[m_, ds_List, w_List] := excisedE[m, Sort[ds], Sort[w]] =
  Module[{q, rest},
   If[ds === {}, Return[volInt[m, w]]];
   q = First[ds]; rest = Rest[ds];
   Simplify[
     outerFace[m, rest, w, q] - innerSphere[m, rest, w, q]
     - Sum[If[w[[u]] === q, excisedE[m, rest, Drop[w, {u}]], 0],
           {u, Length[w]}],
     Assumptions -> Del > 0]];

(* ---------------- the scalar series g_k(r) = Exp[I k r]/r -------------- *)
(* Returned as a list of {m, coefficient} pairs in powers r^m.  nord counts
   terms of the exponential, so nord = 6 keeps r^-1 .. r^5 -- exactly the
   range QMat.nb carries. *)
gSeries[kk_, nord_] := Table[{t - 1, (I kk)^t/t!}, {t, 0, nord}];

(* h = (g_b - g_a)/kb^2, the term the tensor differentiates twice *)
hSeries[ka_, kb_, nord_] :=
   Table[{t - 1, ((I kb)^t - (I ka)^t)/(t! kb^2)}, {t, 0, nord}];

seriesMoment[ser_List, ds_List, w_List] :=
   Total[#[[2]] E$[#[[1]], ds, w] & /@ ser];

(* ---------------- the Green's tensor moment ---------------------------- *)
(* gDyn[i,n,ds,w] = Int_V (prod x_w) d_ds G0_in dV, keeping the exponential
   series to $nord terms -- $nord = 6 reaches r^5, exactly the range QMat.nb
   carries.  Symbols: mu, ka, kb (kb = shear wavenumber). *)
$nord = 6;
gDyn[i_, n_, ds_List, w_List] :=
  Simplify[
    (1/(4 Pi mu)) (
       d[i, n] seriesMoment[gSeries[kb, $nord], ds, w]
       + seriesMoment[hSeries[ka, kb, $nord], Join[{i, n}, ds], w]),
    Assumptions -> Del > 0];

(* ==========================================================================
   THE GENERAL TENSOR FORM, WITHOUT FITTING

   Each moment is an isotropic-medium integral over a CUBE, so it is invariant
   under O_h -- the 48 signed axis permutations -- acting on every index.

   Reflection invariance (the diagonal +-1 elements) forces a component to
   vanish unless every axis value occurs an EVEN number of times among the
   indices.  Axis-permutation invariance (S_3) then says a surviving component
   depends only on WHICH POSITIONS SHARE A VALUE, not on the values.  So the
   invariants are indexed by set partitions of the n index positions into at
   most 3 blocks, each of even size:

       Lam_Pi[i1..in] = 1 if the indices are constant on each block of Pi AND
                          take distinct values on distinct blocks, else 0.

   These are linearly independent (their supports are disjoint), so they form a
   basis, and the expansion coefficient is read off directly:

       c_Pi = T evaluated at the indices that realise Pi
              (block 1 -> axis 1, block 2 -> axis 2, block 3 -> axis 3).

   No linear solve, no fitting, no least squares -- 4 evaluations pin a rank-4
   moment and 31 pin a rank-6 one.  The reconstruction is then CHECKED against
   independently computed components, including ones whose pattern has an odd
   block and must therefore vanish.
   ========================================================================== *)

setPartitions[{}] := {{}};
setPartitions[s_List] := Module[{f = First[s], rest = Rest[s]},
   Join @@ Table[
     Join[Table[ReplacePart[p, u -> Append[p[[u]], f]], {u, Length[p]}],
          {Append[p, {f}]}],
     {p, setPartitions[rest]}]];

(* Canonical order: each block sorted, blocks ordered by their first element.
   Without this the representative index list for a partition depends on the
   order the recursion happened to build the blocks in -- so the coefficient
   would be read at, say, {2,2,1,1} instead of {1,1,2,2}.  The two are equal
   by cubic symmetry but are DIFFERENT EXPRESSIONS, which makes the printed
   form non-reproducible and every downstream comparison harder than it needs
   to be. *)
canonPart[p_] := SortBy[Sort /@ p, First];

(* the admissible partitions: every block even, at most 3 blocks *)
cubicPartitions[n_] :=
   cubicPartitions[n] = SortBy[
      canonPart /@ Select[setPartitions[Range[n]],
         Length[#] <= 3 && AllTrue[#, EvenQ[Length[#]] &] &],
      {Length[#] &, # &}];

(* the index assignment that realises a partition *)
patternIndices[pi_, n_] := Module[{idx = ConstantArray[0, n]},
   Do[Do[idx[[pos]] = b, {pos, pi[[b]]}], {b, Length[pi]}]; idx];

(* Lam_Pi evaluated at an index list *)
lamAt[pi_, idx_] := Module[{vals},
   vals = (idx[[First[#]]] & /@ pi);
   If[Length[Union[vals]] =!= Length[pi], Return[0]];
   If[AllTrue[Range[Length[pi]],
        Function[b, Length[Union[idx[[pi[[b]]]]]] === 1]], 1, 0]];

(* build the general form: returns {partitions, coefficients} *)
cubicTensorForm[fn_, n_] := Module[{pis, cs},
   pis = cubicPartitions[n];
   cs = Table[fn @@ patternIndices[pis[[u]], n], {u, Length[pis]}];
   {pis, cs}];

reconstruct[{pis_, cs_}, idx_] :=
   Total[Table[cs[[u]] lamAt[pis[[u]], idx], {u, Length[pis]}]];

(* pretty-print a partition as a product of generalised Kroneckers *)
partLabel[pi_, names_List] := StringRiffle[
   Table["d[" <> StringRiffle[names[[pi[[b]]]], ","] <> "]", {b, Length[pi]}],
   " "];

(* Verify the reconstruction against the moment on a set of index lists.
   Uses zeroQ, not Simplify === 0 -- see the note at the top of this file. *)
verifyForm[fn_, form_, n_, idxs_List] := Module[{bad = {}},
   Do[If[! zeroQ[fn @@ ix - reconstruct[form, ix]], AppendTo[bad, ix]],
      {ix, idxs}];
   bad];

(* Static limit, straight in Lame parameters.  g -> 1/r and
   h -> hc r  with  hc = (ka^2-kb^2)/(2 kb^2) = -(lam+mu)/(2(lam+2mu)),
   which reproduces Kupradze exactly (checked in CubeMomentCoreTest.wl):
       a0 = (lam+3mu)/(8 Pi mu (lam+2mu)),  b0 = (lam+mu)/(8 Pi mu (lam+2mu)). *)
hc = -(lam + mu)/(2 (lam + 2 mu));
gStatic[i_, n_, ds_List, w_List] :=
  Simplify[
    (1/(4 Pi mu)) (d[i, n] E$[-1, ds, w] + hc E$[1, Join[{i, n}, ds], w]),
    Assumptions -> Del > 0];

(* the same substitution applied to a dynamic result *)
lameRule = {ka -> Sqrt[mu/(lam + 2 mu)] kb};

(* ==========================================================================
   SOURCE versus FIELD DERIVATIVES -- the (-1)^D convention

   "The Basic System" (A33.nb cell 17) defines all six moments with derivatives
   with respect to the SOURCE point r', e.g.

       Q^r_in,pqk = Int d'_q d'_p d'_k (G0_in(0,r')) x'_r dr' .

   G0(0,r') depends on the separation, so d' = -d and a moment with D
   derivatives differs from this engine's field-derivative value by (-1)^D:

       D even (G, K, M, P) -- identical, no factor
       D odd  (N, Q)       -- OPPOSITE SIGN

   This is not cosmetic.  Q enters A33 multiplied by the stiffness contrast,
   so getting it backwards flips the entire elastic response of the 27x27
   block while leaving every symmetry, parity and scaling check intact --
   nothing internal to the moment can detect it.

   It was found by comparing against GreensTensorMoments6.nb, where all 21
   non-zero components of Q's delta_in channel came back negated while M
   (D = 2) agreed to ten digits.  M agreeing is what makes the diagnosis
   specific: a mistake in the engine would not respect the parity of D.
   ========================================================================== *)

srcFactor[nd_Integer] := (-1)^nd;

(* the moments exactly as "The Basic System" defines them *)
Gmom[i_, j_]                     := gStatic[i, j, {}, {}];
Kmom[i_, j_, r_, s_]             := gStatic[i, j, {}, {r, s}];
Nmom[i_, n_, k_, r_]             := srcFactor[1] gStatic[i, n, {k}, {r}];
Mmom[i_, n_, p_, k_]             := gStatic[i, n, {p, k}, {}];
Pmom[i_, j_, p_, q_, r_, s_]     := gStatic[i, j, {p, q}, {r, s}];
Qmom[i_, n_, p_, q_, k_, r_]     := srcFactor[3] gStatic[i, n, {p, q, k}, {r}];

(* index map of the 27x27 block, transcribed from GreensTensorMoments6.nb:
       upsilon[p,q,i] = 9(i-1) + 3(q-1) + p                                  *)
upsilon[p_, q_, i_] := 9 (i - 1) + 3 (q - 1) + p;
