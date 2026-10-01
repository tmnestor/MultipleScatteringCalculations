#!/usr/bin/env wolframscript
(* ============================================================================
   GradedVoxel_MasterIntegrals.wl -- the master integrals of the graded voxel's
   coupling blocks, integrated directly.

     box   I[p,q,r; a,b,c; m] = Int_0^a Int_0^b Int_0^c x^p y^q z^r R^m dz dy dx
     face  J[q,r; a; b,c; m]  = Int_0^b Int_0^c y^q z^r (a^2+y^2+z^2)^(m/2) dz dy

   with m odd.  The Python module cubic_scattering/graded_voxel/moments.py
   evaluates them by three reductions (volume to faces by Euler's identity,
   faces to edges and edges to elementary functions by one derivative identity
   each).  Here they are integrated with no reduction at all: symbolically, one
   variable at a time, and where that does not finish within the time limit,
   numerically at 40 digits.  The two agree, or the closed forms are wrong.

   GATES: (a) the cube's Coulomb self-energy from the box integrals;
          (b) the solid angle of the octant, Pi/2, from the face integrals.
   EXPORT: Mathematica/GradedVoxel_master_integrals.json, read by
           cubic_scattering/tests/test_graded_voxel_moments.py.
   ============================================================================ *)

wp = 40; tlim = 60;
oks = {};
check[name_, ok_] := (AppendTo[oks, ok]; Print[If[ok, "PASS  ", "FAIL  "], name]);

num[f_, lims__] := NIntegrate[f, lims, WorkingPrecision -> wp, PrecisionGoal -> 28, AccuracyGoal -> 40,
   MaxRecursion -> 40, Method -> {"GlobalAdaptive", "SingularityHandler" -> "DuffyCoordinates"}];

(* symbolic, innermost first; Null when any stage fails to finish *)
sym[f_, lims_List] := Module[{g = f, res},
   res = TimeConstrained[
     Do[g = Integrate[g, l, Assumptions -> {x > 0, y > 0, z > 0}], {l, lims}]; g, tlim, $Failed];
   If[res === $Failed || ! FreeQ[res, Integrate | Undefined | Indeterminate | ComplexInfinity], Null, res]];

box[p_, q_, r_, a_, b_, c_, m_] := Module[{f = x^p y^q z^r (x^2 + y^2 + z^2)^(m/2), s},
   s = sym[f, {{z, 0, c}, {y, 0, b}, {x, 0, a}}];
   If[s === Null, {"numeric", num[f, {x, 0, a}, {y, 0, b}, {z, 0, c}]}, {"symbolic", N[s, wp]}]];

face[q_, r_, a_, b_, c_, m_] := Module[{f = y^q z^r (a^2 + y^2 + z^2)^(m/2), s},
   s = sym[f, {{z, 0, c}, {y, 0, b}}];
   If[s === Null, {"numeric", num[f, {y, 0, b}, {z, 0, c}]}, {"symbolic", N[s, wp]}]];

boxCases = {{0, 0, 0, 2, 2, 2, -1}, {1, 0, 0, 2, 2, 2, -1}, {1, 1, 0, 2, 2, 2, -1}, {1, 1, 1, 2, 2, 2, -1},
   {1, 1, 0, 2, 4, 2, -1}, {2, 0, 1, 2, 4, 2, -3}, {1, 2, 3, 4, 2, 2, -3}, {2, 2, 0, 2, 2, 4, -3},
   {0, 2, 2, 2, 2, 4, 1}, {4, 3, 2, 4, 4, 2, -3}, {6, 0, 2, 2, 4, 4, -3}};
faceCases = {{0, 0, 2, 2, 2, -3}, {0, 0, 2, 2, 4, -3}, {0, 0, 4, 2, 4, -3}, {0, 0, 2, 2, 4, -1},
   {1, 0, 2, 4, 2, -3}, {3, 4, 4, 2, 4, -3}, {5, 2, 2, 4, 4, 1}, {2, 3, 0, 4, 2, -3}, {1, 1, 0, 2, 2, -3},
   {0, 0, 0, 2, 4, -1}};

rows = {};
Do[Module[{v = box @@ cs},
   Print["box  ", cs, "  ", v[[1]], "  ", N[v[[2]], 20]];
   AppendTo[rows, <|"kind" -> "box", "args" -> cs, "how" -> v[[1]], "value" -> ToString[v[[2]], InputForm]|>]],
  {cs, boxCases}];
Do[Module[{v = face @@ cs},
   Print["face ", cs, "  ", v[[1]], "  ", N[v[[2]], 20]];
   AppendTo[rows, <|"kind" -> "face", "args" -> cs, "how" -> v[[1]], "value" -> ToString[v[[2]], InputForm]|>]],
  {cs, faceCases}];

val[kind_, cs_] := ToExpression[First[Select[rows, #kind == kind && #args == cs &]]["value"]];

(* (a) Coulomb self-energy of the cube of half-width 1: 8 Int_[0,2]^3 Prod (2 - s_i) / |s| *)
(* the cube is symmetric, so each (p, q, r) is read from its sorted representative *)
coulExact = 2 ((1 + Sqrt[2] - 2 Sqrt[3])/5 - Pi/3 + Log[(1 + Sqrt[2]) (2 + Sqrt[3])]) 2^5;
perm[{p_, q_, r_}] := Reverse[Sort[{p, q, r}]];
coul = 8 Sum[2^(3 - p - q - r) (-1)^(p + q + r) val["box", Join[perm[{p, q, r}], {2, 2, 2, -1}]],
    {p, 0, 1}, {q, 0, 1}, {r, 0, 1}];
check["(a) Coulomb self-energy of the cube from the box integrals", Abs[coul/coulExact - 1] < 10^-20];

(* (b) the flux of x/r^3 through the far faces of [0,2]^3 is the octant's solid angle *)
check["(b) solid angle of the octant", Abs[3*2 val["face", {0, 0, 2, 2, 2, -3}]/(Pi/2) - 1] < 10^-20];

Export[FileNameJoin[{DirectoryName[$InputFileName], "GradedVoxel_master_integrals.json"}], rows, "JSON"];
Print[Count[oks, True], "/", Length[oks], " checks passed; ",
  Count[rows, r_ /; r["how"] == "symbolic"], " symbolic, ", Count[rows, r_ /; r["how"] == "numeric"], " numeric"];
If[! And @@ oks, Exit[1]];
