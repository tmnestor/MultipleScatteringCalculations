(* ::Package:: *)
(* CubeT57MpB.wl -- B-channel master integrals for the T_57 cubic extension.

   MpB[p,q,r] = \int_{[0,1]^3} x^p y^q z^r / (x^2+y^2+z^2)^{3/2} dV

   These are the surface-stiffness master integrals.  The body bilinear form
   uses Mp[p,q,r] (1/rho kernel); the surface traction kernel
   x_i x_j / |x|^3 produces the B-channel MpB integrals (1/rho^3 kernel).

   Convergence condition: p+q+r >= 2  (integrand ~ rho^{p+q+r-3} near origin,
   integrable in 3D when p+q+r >= 2).

   Fundamental identity:
     MpB[p+2,q,r] + MpB[p,q+2,r] + MpB[p,q,r+2] = Mp[p,q,r]
   (from x^2+y^2+z^2 = rho^2, dividing 1/rho by 1/rho^3)

   T_27 surface stiffness used MpB at indices p,q <= 4, r <= 2 (29 triples).
   T_57 cubic modes push to p,q <= 6, r <= 3, plus B-channel kernel adds
   up to 2 more to individual indices.  We compute ALL canonical triples
   at total degree 2 through 10 (65 integrals).

   USAGE:
     wolframscript -file Mathematica/CubeT57MpB.wl
*)

$HistoryLength = 0;
Print["==== CubeT57MpB.wl: B-channel masters for T_57 ===="];
Print["Started: ", DateString[]];
Print[];

(* ============================================================ *)
(* Section 1: Integral definitions                               *)
(* ============================================================ *)

ClearAll[MpB, numMpB, numMp];

(* --- MpB: B-channel (1/rho^3 kernel) --- *)

(* Canonicalize by sorting indices *)
MpB[p_Integer, q_Integer, r_Integer] /; !OrderedQ[{p, q, r}] :=
  MpB @@ Sort[{p, q, r}];

(* Memoized symbolic integration with timeout *)
MpB[p_, q_, r_] := MpB[p, q, r] = Module[{raw, real},
  raw = TimeConstrained[
    Integrate[
      x^p y^q z^r / (x^2 + y^2 + z^2)^(3/2),
      {z, 0, 1}, {y, 0, 1}, {x, 0, 1}
    ],
    300  (* 5 minute timeout per integral *)
  ];
  If[raw === $Aborted,
    Print["      Integrate timed out, using NIntegrate fallback"];
    Return[numMpB[p, q, r]]
  ];
  real = Simplify[Re[ComplexExpand[raw]]];
  If[NumericQ[N[real]] && FreeQ[real, Undefined | Indeterminate],
    real,
    (* Try harder with TimeConstrained FullSimplify *)
    Module[{real2},
      real2 = TimeConstrained[FullSimplify[Re[ComplexExpand[raw]]], 60];
      If[real2 =!= $Aborted && NumericQ[N[real2]] &&
         FreeQ[real2, Undefined | Indeterminate],
        real2,
        Print["      Integrate non-numeric, using NIntegrate fallback"];
        numMpB[p, q, r]
      ]
    ]
  ]
];

(* High-precision numerical reference *)
numMpB[p_, q_, r_] := NIntegrate[
  x^p y^q z^r / (x^2 + y^2 + z^2)^(3/2),
  {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
  PrecisionGoal -> 16, WorkingPrecision -> 30
];

(* --- Mp: A-channel (1/rho kernel) — for recurrence verification only --- *)
numMp[p_, q_, r_] := NIntegrate[
  x^p y^q z^r / Sqrt[x^2 + y^2 + z^2],
  {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
  PrecisionGoal -> 16, WorkingPrecision -> 30
];

(* ============================================================ *)
(* Section 2: Sanity check — cubic symmetry + recurrence         *)
(* ============================================================ *)

Print["-- Sanity check: MpB[2,0,0] via cubic symmetry --"];
t0 = AbsoluteTime[];
val200 = MpB[2, 0, 0];
t1 = AbsoluteTime[];
nVal200 = N[val200, 20];
nNum200 = numMpB[2, 0, 0];
Print["  MpB[2,0,0] symbolic = ", val200];
Print["  MpB[2,0,0] numeric  = ", nVal200];
Print["  NIntegrate           = ", nNum200];
Print["  |diff|               = ", Abs[nVal200 - nNum200]];
Print["  time: ", Round[t1 - t0, 0.1], " s"];

(* Recurrence: MpB[2,0,0] + MpB[0,2,0] + MpB[0,0,2] = Mp[0,0,0]
   By O_h symmetry all three are equal, so 3*MpB[2,0,0] = Mp[0,0,0] *)
mp000Num = numMp[0, 0, 0];
Print["  Recurrence: 3*MpB[2,0,0] = ", N[3 nVal200, 16]];
Print["  Mp[0,0,0] (NInteg)       = ", mp000Num];
Print["  |diff|                    = ", Abs[3 nVal200 - mp000Num]];
Print[];

(* ============================================================ *)
(* Section 3: Enumerate all canonical triples (degree 2-10)      *)
(* ============================================================ *)

allTriples = {};
Do[
  Module[{pp = dd - qq - rr},
    If[pp >= qq, AppendTo[allTriples, {pp, qq, rr}]]],
  {dd, 2, 10}, {rr, 0, Floor[dd/3]}, {qq, rr, Floor[(dd - rr)/2]}];

Print["Canonical MpB triples to compute: ", Length[allTriples]];
degTally = Tally[Total /@ allTriples];
Do[Print["  degree ", dt[[1]], ": ", dt[[2]], " triples"],
  {dt, SortBy[degTally, First]}];
Print[];

(* ============================================================ *)
(* Section 4: Compute all MpB master integrals                   *)
(* ============================================================ *)

results = {};
tStart = AbsoluteTime[];

Do[
  Module[{p, q, r, deg, closed, nAna, nNum, diff, t0i, t1i, isExact, idx},
    idx = Position[allTriples, triple][[1, 1]];
    {p, q, r} = triple;
    deg = p + q + r;
    Print["[", idx, "/", Length[allTriples],
      "] MpB[", p, ",", q, ",", r, "]  (degree ", deg, ")"];

    t0i = AbsoluteTime[];
    closed = MpB[p, q, r];
    t1i = AbsoluteTime[];

    isExact = Not[MatchQ[Head[closed], Real | Integer]] &&
              FreeQ[closed, _Real] && FreeQ[closed, _?MachineNumberQ];
    nAna = N[closed, 20];
    nNum = numMpB[p, q, r];
    diff = Abs[nAna - nNum];

    If[isExact,
      Print["  closed form: ", InputForm[closed]],
      Print["  [NIntegrate fallback]"]
    ];
    Print["  value      = ", nAna];
    Print["  NIntegrate = ", nNum];
    Print["  |diff|     = ", diff];
    Print["  time: ", Round[t1i - t0i, 0.1], " s"];
    Print["  elapsed: ", Round[AbsoluteTime[] - tStart, 1], " s total"];
    Print[];

    AppendTo[results, {triple, closed, nAna, t1i - t0i, isExact}];
  ],
  {triple, allTriples}
];

(* ============================================================ *)
(* Section 5: Recurrence identity verification                   *)
(* ============================================================ *)

Print[];
Print["==== RECURRENCE VERIFICATION ===="];
Print["Identity: MpB[p+2,q,r] + MpB[p,q+2,r] + MpB[p,q,r+2] = Mp[p,q,r]"];
Print["Checking for all (p,q,r) with p+q+r in [0,8] ..."];
Print[];

(* Generate recurrence check triples: canonical (p >= q >= r >= 0),
   total degree 0 to 8, so MpB values are at degree 2-10 (all computed) *)
recTriples = {};
Do[
  Module[{pp = dd - qq - rr},
    If[pp >= qq, AppendTo[recTriples, {pp, qq, rr}]]],
  {dd, 0, 8}, {rr, 0, Floor[dd/3]}, {qq, rr, Floor[(dd - rr)/2]}];

maxRecErr = 0;
nChecked = 0;
Do[
  Module[{p, q, r, lhs, rhs, err},
    {p, q, r} = triple;
    (* LHS: sum of three MpB values at degree p+q+r+2 *)
    lhs = N[MpB[p + 2, q, r], 20] + N[MpB[p, q + 2, r], 20] +
          N[MpB[p, q, r + 2], 20];
    (* RHS: Mp[p,q,r] via NIntegrate *)
    rhs = numMp[p, q, r];
    err = Abs[lhs - rhs];
    nChecked++;
    If[err > 1*^-8,
      Print["  WARNING: (", p, ",", q, ",", r,
        "): |recurrence err| = ", err]];
    maxRecErr = Max[maxRecErr, err]],
  {triple, recTriples}];

Print["Checked ", nChecked, " recurrence identities."];
Print["Max recurrence error: ", maxRecErr];
If[maxRecErr < 1*^-8,
  Print["  ALL verified (< 1e-8)"],
  Print["  WARNING: some identities have large error!"]];
Print[];

(* ============================================================ *)
(* Section 6: Summary                                            *)
(* ============================================================ *)

Print["==== SUMMARY ===="];
nExact = Count[results, {_, _, _, _, True}];
nFallback = Length[results] - nExact;
tTotal = AbsoluteTime[] - tStart;
Print[Length[results], " MpB integrals computed in ",
  Round[tTotal, 1], " s"];
Print["  ", nExact, " exact closed forms"];
Print["  ", nFallback, " NIntegrate fallbacks"];
Print[];

Do[
  Module[{triple2, closed, nval, dt, exact},
    {triple2, closed, nval, dt, exact} = r;
    Print[StringForm["  MpB[``] = ``  (`` s, ``)",
      StringJoin[Riffle[ToString /@ triple2, ","]],
      NumberForm[nval, 16], Round[dt, 0.1],
      If[exact, "exact", "numeric"]]]],
  {r, results}];

(* ============================================================ *)
(* Section 7: Export to values file                              *)
(* ============================================================ *)

outFile = FileNameJoin[{DirectoryName[$InputFileName],
  "CubeT57MpBValues.wl"}];
strm = OpenWrite[outFile];

WriteString[strm,
  "(* CubeT57MpBValues.wl -- auto-generated by CubeT57MpB.wl *)\n"];
WriteString[strm,
  "(* B-channel master integrals MpB[p,q,r] for T_57 extension *)\n"];
WriteString[strm,
  "(* MpB[p,q,r] = Integral_{[0,1]^3} x^p y^q z^r / \
(x^2+y^2+z^2)^{3/2} dV *)\n"];
WriteString[strm,
  "(* Identity: MpB[p+2,q,r] + MpB[p,q+2,r] + MpB[p,q,r+2] = Mp[p,q,r] *)\n\n"];

Do[
  Module[{triple2, closed, nval, dt, exact, name, pqrStr},
    {triple2, closed, nval, dt, exact} = r;
    name = "mpB" <> StringJoin[ToString /@ triple2];
    pqrStr = StringJoin[Riffle[ToString /@ triple2, ","]];
    WriteString[strm,
      "(* MpB[" <> pqrStr <> "] *)\n"];
    If[exact,
      WriteString[strm,
        name <> " = " <> ToString[InputForm[closed]] <> ";\n"],
      WriteString[strm,
        name <> " = " <> ToString[InputForm[nval]] <>
        ";  (* NIntegrate fallback *)\n"]
    ];
    WriteString[strm,
      name <> "Num = " <> ToString[InputForm[nval]] <> ";\n\n"]],
  {r, results}];

WriteString[strm,
  "Print[\"CubeT57MpBValues.wl loaded: " <>
  ToString[Length[results]] <> " B-channel masters\"];\n"];
Close[strm];

Print[];
Print["Results written to: ", outFile];
Print[];
Print["Finished: ", DateString[]];
