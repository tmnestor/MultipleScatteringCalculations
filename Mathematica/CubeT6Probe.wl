(* ::Package:: *)
(* CubeT6Probe.wl -- Standalone probe of the new Mp master integrals
   required by the 27-component Path-B closure (Tier-6).

   The integrals over the positive octant [0,1]^3 are:

     Mp[p,q,r] = int_{[0,1]^3} x^p y^q z^r / sqrt(x^2+y^2+z^2) dV

   For all-even (p,q,r) these reduce to (1/8) M[p,q,r] where
     M[p,q,r] = int_{[-1,1]^3} x^p y^q z^r / sqrt(x^2+y^2+z^2) dV
   and Mathematica closes these in logarithms / Pell units.

   Tier-6 needs degree up to 8:
     quad-linear block: (2,2,2), (4,2,0), (6,0,0)           [degree 6]
     quad-quad   block: (4,2,2), (4,4,0), (6,2,0), (8,0,0)  [degree 8]

   This script computes each one, prints the closed form, and checks
   the numerical value against direct NIntegrate to >= 12 digits.
*)

Print["==== CubeT6Probe.wl: new masters for 27-component closure ===="];

(* ------- Closed-form definitions (all-even case via M/8) ------- *)
ClearAll[M, Mp];
M[p_, q_, r_] := M[p, q, r] = If[OddQ[p] || OddQ[q] || OddQ[r],
  0,
  Integrate[
    x^p y^q z^r/Sqrt[x^2 + y^2 + z^2],
    {x, -1, 1}, {y, -1, 1}, {z, -1, 1}
  ]
];

Mp[p_Integer, q_Integer, r_Integer] /; ! OrderedQ[{p, q, r}] :=
  Mp @@ Sort[{p, q, r}];
Mp[p_, q_, r_] := Mp[p, q, r] = Module[{raw, real},
  raw = If[EvenQ[p] && EvenQ[q] && EvenQ[r],
    M[p, q, r]/8,
    Integrate[
      x^p y^q z^r/Sqrt[x^2 + y^2 + z^2],
      {x, 0, 1}, {y, 0, 1}, {z, 0, 1}
    ]
  ];
  real = FullSimplify[Re[ComplexExpand[raw]]];
  real
];

(* Direct numerical check for cross-validation. *)
numMp[p_, q_, r_] := NIntegrate[
  x^p y^q z^r/Sqrt[x^2 + y^2 + z^2],
  {x, 0, 1}, {y, 0, 1}, {z, 0, 1},
  PrecisionGoal -> 14, WorkingPrecision -> 30
];

newMasters = {
  (* quad-linear block, degree 6 *)
  {2, 2, 2}, {4, 2, 0}, {6, 0, 0},
  (* quad-quad block, degree 8 *)
  {4, 2, 2}, {4, 4, 0}, {6, 2, 0}, {8, 0, 0}
};

Do[
  Module[{p, q, r, closed, numAnalytic, numDirect, diff, t0, t1},
    {p, q, r} = mm;
    Print[""];
    Print["---- Mp[", p, ",", q, ",", r, "] (degree ", p + q + r, ") ----"];
    t0 = AbsoluteTime[];
    closed = Mp[p, q, r];
    t1 = AbsoluteTime[];
    Print["  closed form (", Round[t1 - t0, 0.01], " s):"];
    Print["    ", closed];
    numAnalytic = N[closed, 20];
    numDirect   = numMp[p, q, r];
    diff        = numAnalytic - numDirect;
    Print["  analytic  ≈ ", numAnalytic];
    Print["  NIntegrate ≈ ", numDirect];
    Print["  difference = ", diff];
  ],
  {mm, newMasters}
];

Print[""];
Print["==== CubeT6Probe DONE ===="];
