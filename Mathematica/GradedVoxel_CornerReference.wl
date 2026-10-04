#!/usr/bin/env wolframscript
(* ============================================================================
   GradedVoxel_CornerReference.wl -- the static Galerkin coupling blocks of two
   cubes touching at a CORNER or an EDGE, at 40 digits, with no reduction.

     K_ac = Int_[-1,1]^6 xi^a xi'^c P(R + xi - xi') = Int_[-2,2]^3 W_ac(s) P(R + s) ds,

   half-width 1, R = 2 o, W_ac(s) = prod_i w(a_i, c_i; s_i) and
   w(e, f; sig) = Int xi^e (xi - sig)^f dxi over [-1,1] meet [sig-1, sig+1],
   integrated EXACTLY here from that definition (a polynomial on [-2,0] and on
   [0,2]).  P is the static propagator, sum over the 41 terms coef * d^idx r^m
   (m = -1 and m = 1) whose 9 x 9 coefficients the spec file carries.

   WHY NO DISTRIBUTION.  The singular point s* = -R is a vertex of the pieces of
   W, where W vanishes as rho^3 (corner) or rho^2 (edge); the worst term,
   d^4 r ~ rho^-3, is then absolutely integrable.  On the piece holding s*,
   Duffy pyramids about s* turn the integrand into u1^k times a smooth function
   (k >= 0 an integer), so a tensor Gauss rule converges exponentially; the
   other pieces are smooth.  Nothing is shared with the package's routes (no
   moved derivatives, no master integrals, no double precision).

   GATES: (a) a polynomial kernel |R + s|^2 against its exact 6-D integral (tests W,
              the pieces, the Duffy map and the weights to 30 digits);
          (b) integration by parts on the hardest term, d^4 r ~ rho^-3:
              int W d_2 F = -int (d_2 W) F, F = d0 d0 d1 r: two different
              integrands, equal only if the singular integration is accurate;
          (c) the rule at n and n + 8 points agrees to 1e-20.
   (Pointwise identities such as Laplacian(1/r) = 0 hold at every node and test
   nothing about the integration, so they are not used.)
   EXPORT: Mathematica/GradedVoxel_corner_reference_<cell>.json, read by
           scripts/gate_galerkin_corner_reference.py.

   Run:  wolframscript -file Mathematica/GradedVoxel_CornerReference.wl linear 20
         (cell = linear | quadratic, n = Gauss points per axis)
   ============================================================================ *)

Needs["NumericalDifferentialEquationAnalysis`"];
args = Rest[$ScriptCommandLine];
cell = If[Length[args] >= 1, args[[1]], "linear"];
nG = If[Length[args] >= 2, ToExpression[args[[2]]], 20];
prec = 40;
dir = DirectoryName[$InputFileName];
spec = Import[FileNameJoin[{dir, "GradedVoxel_CornerReference_spec.json"}], "RawJSON"];
{nSource, nTestMono} = If[cell === "quadratic", {35, 10}, {10, 4}];
testExp = spec["test_exponents"][[;; nTestMono]];
srcExp = spec["source_exponents"][[;; nSource]];
terms = spec["terms"];
oks = {};
check[name_, ok_] := (AppendTo[oks, ok]; Print[If[ok, "PASS  ", "FAIL  "], name]);

(* the autocorrelation of one axis, exactly; branch -1: sig in [-2,0], +1: sig in [0,2] *)
ClearAll[wexpr];
wexpr[e_, f_, br_] := wexpr[e, f, br] = Module[{xi},
   Expand[If[br == -1, Integrate[xi^e (xi - sig)^f, {xi, -1, sig + 1}],
     Integrate[xi^e (xi - sig)^f, {xi, sig - 1, 1}]]]];

(* the radial terms d^idx r^m as expressions in x1, x2, x3 *)
vars = {x1, x2, x3};
fexpr[m_, idx_] := D[(x1^2 + x2^2 + x3^2)^(m/2), Sequence @@ (vars[[# + 1]] & /@ idx)];
fList = fexpr[#["m"], #["idx"]] & /@ terms;

(* tensor Gauss rule on [0,1] *)
rule[n_] := rule[n] = GaussianQuadratureWeights[n, 0, 1, prec];

(* nodes (n^3 x 3) and weights in s for one piece; Duffy pyramids when s* is a vertex *)
pieceNodes[br_, sstar_, n_] := Module[{lo, hi, isVertex, inside, g, u, w, pts, wts, d, y},
  lo = If[# == -1, -2, 0] & /@ br; hi = lo + 2;
  isVertex = And @@ MapThread[(#1 == #2 || #1 == #3) &, {sstar, lo, hi}];
  inside = And @@ MapThread[(#2 <= #1 <= #3) &, {sstar, lo, hi}];
  If[inside && ! isVertex, Print["s* inside a piece but not at a vertex: not corner/edge contact"]; Abort[]];
  g = rule[n];
  If[! isVertex,
   (* plain tensor Gauss on the box *)
   u = Tuples[g[[All, 1]], 3]; w = Times @@@ Tuples[g[[All, 2]], 3];
   pts = (lo + 2 #) & /@ u; wts = 8 w,
   (* Duffy: y = s - s* points into the piece, y in [0,2]^3, split by the largest coordinate *)
   d = MapThread[If[#1 == #2, 1, -1] &, {sstar, lo}];
   u = Tuples[g[[All, 1]], 3]; w = Times @@@ Tuples[g[[All, 2]], 3];
   pts = {}; wts = {};
   Do[
    y = Map[Function[uu, Module[{v = {uu[[1]] uu[[2]], uu[[1]] uu[[3]]}, out},
         out = Insert[v, uu[[1]], k]; 2 out]], u];
    pts = Join[pts, (sstar + d #) & /@ y];
    wts = Join[wts, 8 w (u[[All, 1]]^2)],
    {k, 1, 3}]];
  {pts, wts}];

(* int W_ac(s) f(R + s) ds for each kernel expression f in fl, summed over the 8 pieces.  With dax > 0 the
   weight is d W_ac / d s_dax instead (the derivative of that axis's polynomial; W stays continuous) *)
moments[off_, n_, fl_, dax_ : 0] := Module[{R, sstar, acc, pts, wts, sv, wmat, fvals, X, wf},
  R = 2 off; sstar = -R;
  wf[e_, f_, br_, i_] := If[i == dax, D[wexpr[e, f, br], sig], wexpr[e, f, br]];
  acc = ConstantArray[0, {Length[fl], nTestMono, nSource}];
  Do[
   {pts, wts} = pieceNodes[br, sstar, n];
   sv = Transpose[pts];
   wmat = Table[wts * Times @@ Table[With[{pv = wf[testExp[[a, i]], srcExp[[c, i]], br[[i]], i]},
        If[NumericQ[pv], ConstantArray[pv, Length[pts]], pv /. sig -> sv[[i]]]], {i, 3}],
     {a, nTestMono}, {c, nSource}];
   X = R + # & /@ pts;
   fvals = Table[With[{v = fl[[t]] /. Thread[vars -> Transpose[X]]},
      If[ListQ[v], v, ConstantArray[v, Length[pts]]]], {t, Length[fl]}];
   acc += Table[wmat . fvals[[t]], {t, Length[fl]}],
   {br, Tuples[{-1, 1}, 3]}];
  acc];

(* a plain decimal string Python can read: mantissa to 22 places and a power of ten *)
fmt[x_] := If[x == 0, "0", Module[{e = Floor[Log10[Abs[x]]]},
    ToString[NumberForm[N[x/10^e, 30], {25, 22}]] <> "e" <> ToString[e]]];

pos[m_, idx_] := First@FirstPosition[terms, _?(#["m"] == m && #["idx"] == idx &), {0}];

medium = spec["medium"];
{al, be, rh} = Rationalize[#, 0] & /@ {medium["alpha"], medium["beta"], medium["rho"]};
mu = rh be^2; pref = 1/(4 Pi mu);
c1 = pref; c2 = -pref (1 - be^2/al^2)/2;  (* weights of delta_ij/r and of d_i d_j r in the static G *)
fim = Rationalize[#, 0] & /@ spec["field_in_monomials"];

result = <||>;
Do[
 off = spec["offsets"][name];
 t0 = AbsoluteTime[];
 u = moments[off, nG, fList];
 Print[name, " ", off, ": ", nTestMono, " x ", nSource, " cells, n = ", nG, ", ",
  Round[AbsoluteTime[] - t0, 0.1], " s"];
 (* gate (a): a polynomial kernel |R + s|^2, against the exact 6-D integral *)
 pexact = Table[Module[{x, y},
     Integrate[Times @@ (Array[x, 3]^testExp[[a]]) Times @@ (Array[y, 3]^srcExp[[c]]) *
       Total[(2 off + Array[x, 3] - Array[y, 3])^2],
      Sequence @@ Table[{x[i], -1, 1}, {i, 3}], Sequence @@ Table[{y[i], -1, 1}, {i, 3}]]],
    {a, nTestMono}, {c, nSource}];
 pquad = First@moments[off, nG, {x1^2 + x2^2 + x3^2}];
 ea = Max[Abs[pquad - pexact]]/Max[Abs[pexact]];
 check[name <> " (a) polynomial kernel against the exact integral: " <> ToString[N[ea, 3]], ea < 10^-30];
 (* gate (b): integration by parts on the hardest term, int W d_2 F = -int (d_2 W) F, F = d0 d0 d1 r *)
 tHard = pos[1, {0, 0, 1, 2}];
 fLow = fexpr[1, {0, 0, 1}];
 ibp = u[[tHard]] + First@moments[off, nG, {fLow}, 3];
 eb = Max[Abs[ibp]]/Max[Abs[u[[tHard]]]];
 check[name <> " (b) integration by parts on d^4 r: " <> ToString[N[eb, 3]], eb < 10^-20];
 (* gate (c): convergence of the rule *)
 u2 = moments[off, nG + 8, fList];
 conv = Max[Abs[u2 - u]]/Max[Abs[u2]];
 check[name <> " (c) n against n + 8 points: " <> ToString[N[conv, 3]], conv < 10^-20];
 (* the static block, monomial rows, then field rows: blk[a, c, 81] *)
 blk = Sum[(If[terms[[t]]["m"] == -1, c1, c2]) *
    Outer[Times, u2[[t]], Flatten[Rationalize[#, 0] & /@ terms[[t]]["coef"]]], {t, Length[terms]}];
 If[nTestMono == 10, blk = fim . blk];
 result[name] = <|
   "block" -> Map[fmt, blk, {3}],
   "terms" -> Table[<|"m" -> terms[[t]]["m"], "idx" -> terms[[t]]["idx"],
       "U" -> Map[fmt, u2[[t]], {2}]|>,
     {t, Length[terms]}]|>,
 {name, {"corner", "edge"}}];

Export[FileNameJoin[{dir, "GradedVoxel_corner_reference_" <> cell <> ".json"}], result, "RawJSON"];
Print[Count[oks, True], "/", Length[oks], " gates pass"];
