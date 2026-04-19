(* CubeInterVoxelMasters.wl — Shifted master integrals for inter-voxel propagator *)
(*                                                                                 *)
(* For face-adjacent voxels separated by R = 2a x̂, the propagator's A-channel    *)
(* integral (after shift v₁ = u₁ + 2a) reduces to rectangular-domain integrals:   *)
(*                                                                                 *)
(*   MpRect[p,q,r; L₁,L₂,L₃] = ∫₀^L₁ ∫₀^L₂ ∫₀^L₃ x^p y^q z^r / ρ dx dy dz   *)
(*                                                                                 *)
(* where ρ = √(x²+y²+z²).  Same algebraic class as the standard Mp[p,q,r]        *)
(* (integrated over [0,1]³), but with different rectangular bounds.                *)
(*                                                                                 *)
(* CRITICAL: Mathematica's 3D Integrate[..., {x,0,L}, {y,0,M}, {z,0,N}] returns  *)
(* Undefined for L,M,N > 1.  The fix: integrate SEQUENTIALLY (z, then y, then x)  *)
(* with GenerateConditions->False and positivity Assumptions.  Each 1D step        *)
(* succeeds and yields exact closed forms.                                         *)
(*                                                                                 *)
(* Key domains (half-width a = 1):                                                 *)
(*   Face-adjacent  R = 2x̂        : [0,4] × [0,2]²                               *)
(*   Edge-adjacent  R = 2(x̂+ŷ)   : [0,4]² × [0,2]                                *)
(*   Corner-adjacent R = 2(x̂+ŷ+ẑ): [0,4]³                                         *)
(*   Standard Mp    R = 0          : [0,1]³  (after octant fold & rescale to a=1)  *)
(* ════════════════════════════════════════════════════════════════════════════════ *)

Print["=============================================================="];
Print["  CubeInterVoxelMasters.wl"];
Print["  Shifted master integrals for inter-voxel propagator"];
Print["=============================================================="];
Print[];


(* ================================================================ *)
(* Section 1: Sequential integration helpers                        *)
(* ================================================================ *)

Print["-- Section 1: Defining sequential integration helpers --"];
Print[];

(* A-channel: 1/rho kernel, sequential z -> y -> x *)
MpRect[p_, q_, r_, L1_, L2_, L3_] :=
  MpRect[p, q, r, L1, L2, L3] = Module[{iz, iyz, ixyz, t0},
    t0 = AbsoluteTime[];
    Print["  MpRect[", p, ",", q, ",", r, "; ",
          L1, ",", L2, ",", L3, "] ... "];
    iz = Integrate[x^p*y^q*z^r / Sqrt[x^2 + y^2 + z^2],
           {z, 0, L3}, Assumptions -> x > 0 && y > 0,
           GenerateConditions -> False];
    iyz = Integrate[iz, {y, 0, L2},
           Assumptions -> x > 0, GenerateConditions -> False];
    ixyz = Integrate[iyz, {x, 0, L1},
           GenerateConditions -> False];
    Print["    done in ", Round[AbsoluteTime[] - t0, 0.1], " s"];
    Simplify[ixyz]
  ];

(* B-channel: 1/rho^3 kernel, sequential z -> y -> x *)
MpBRect[p_, q_, r_, L1_, L2_, L3_] :=
  MpBRect[p, q, r, L1, L2, L3] = Module[{iz, iyz, ixyz, t0},
    t0 = AbsoluteTime[];
    Print["  MpBRect[", p, ",", q, ",", r, "; ",
          L1, ",", L2, ",", L3, "] ... "];
    iz = Integrate[x^p*y^q*z^r / (x^2 + y^2 + z^2)^(3/2),
           {z, 0, L3}, Assumptions -> x > 0 && y > 0,
           GenerateConditions -> False];
    iyz = Integrate[iz, {y, 0, L2},
           Assumptions -> x > 0, GenerateConditions -> False];
    ixyz = Integrate[iyz, {x, 0, L1},
           GenerateConditions -> False];
    Print["    done in ", Round[AbsoluteTime[] - t0, 0.1], " s"];
    Simplify[ixyz]
  ];

(* Shorthands for each neighbour type (half-width a = 1) *)
MpFace[p_, q_, r_]   := MpRect[p, q, r, 4, 2, 2];
MpEdge[p_, q_, r_]   := MpRect[p, q, r, 4, 4, 2];
MpCorner[p_, q_, r_] := MpRect[p, q, r, 4, 4, 4];

MpBFace[p_, q_, r_]   := MpBRect[p, q, r, 4, 2, 2];
MpBEdge[p_, q_, r_]   := MpBRect[p, q, r, 4, 4, 2];
MpBCorner[p_, q_, r_] := MpBRect[p, q, r, 4, 4, 4];


(* ================================================================ *)
(* Section 2: Scaling identity — [0,2]^3 from standard Mp           *)
(* ================================================================ *)

Print["-- Section 2: Scaling identity --"];
Print["  integral_[0,2]^3 x^p y^q z^r / rho  =  2^(p+q+r+2) * Mp[p,q,r]"];
Print[];

(* Standard Mp over [0,1]^3 — compute fresh *)
MpStd[p_, q_, r_] := MpStd[p, q, r] = Module[{iz, iyz, ixyz},
  iz = Integrate[x^p*y^q*z^r / Sqrt[x^2+y^2+z^2],
         {z,0,1}, Assumptions -> x>0 && y>0, GenerateConditions->False];
  iyz = Integrate[iz, {y,0,1}, Assumptions -> x>0, GenerateConditions->False];
  Integrate[iyz, {x,0,1}, GenerateConditions->False]
];

scalingTests = {{0,0,0}, {1,0,0}, {1,1,0}, {2,0,0}};
Do[
  {p, q, r} = triple;
  mp = MpStd[p, q, r];
  scaled = N[2^(p + q + r + 2) * mp, 20];
  direct = N[MpRect[p, q, r, 2, 2, 2], 20];
  err = Abs[scaled - direct];
  Print["  [", p, ",", q, ",", r, "]:  2^n*Mp = ",
        NumberForm[scaled, 16], "  direct = ",
        NumberForm[direct, 16], "  |D| = ", ScientificForm[err, 3],
        If[err < 10^-12, "  OK", "  FAIL"]];
  ,
  {triple, scalingTests}
];
Print[];


(* ================================================================ *)
(* Section 3: Face-adjacent A-channel [0,4] x [0,2]^2              *)
(* ================================================================ *)

Print["-- Section 3: Face-adjacent A-channel [0,4]x[0,2]^2 --"];
Print[];

faceTriples = {
  {0,0,0},                              (* degree 0 *)
  {1,0,0}, {0,1,0},                     (* degree 1 *)
  {2,0,0}, {1,1,0}, {0,2,0},            (* degree 2 *)
  {3,0,0}, {2,1,0}, {1,2,0}, {1,1,1}    (* degree 3 *)
};

faceAResults = {};
Do[
  {p, q, r} = triple;
  val = MpFace[p, q, r];
  symNum = N[val, 20];
  isExact = Not[MachineNumberQ[val]];
  AppendTo[faceAResults, {triple, val, symNum, isExact}];
  Print["  MpFace[", p, ",", q, ",", r, "] = ",
        NumberForm[symNum, 16],
        If[isExact, "  EXACT", "  numeric"]];
  ,
  {triple, faceTriples}
];
Print[];


(* ================================================================ *)
(* Section 4: Face-adjacent B-channel [0,4] x [0,2]^2              *)
(* ================================================================ *)

Print["-- Section 4: Face-adjacent B-channel [0,4]x[0,2]^2 --"];
Print["  (convergence requires p+q+r >= 2)"];
Print[];

bFaceTriples = {
  {2,0,0}, {1,1,0}, {0,2,0}, {0,0,2},  (* degree 2 *)
  {3,0,0}, {2,1,0}, {1,2,0}, {1,0,2},
  {0,3,0}, {0,2,1}, {1,1,1}             (* degree 3 *)
};

faceBResults = {};
Do[
  {p, q, r} = triple;
  val = MpBFace[p, q, r];
  symNum = N[val, 20];
  isExact = Not[MachineNumberQ[val]];
  AppendTo[faceBResults, {triple, val, symNum, isExact}];
  Print["  MpBFace[", p, ",", q, ",", r, "] = ",
        NumberForm[symNum, 16],
        If[isExact, "  EXACT", "  numeric"]];
  ,
  {triple, bFaceTriples}
];
Print[];


(* ================================================================ *)
(* Section 5: Recurrence identity verification                      *)
(* ================================================================ *)

Print["-- Section 5: Recurrence check --"];
Print["  MpBRect[p+2,q,r] + MpBRect[p,q+2,r] + MpBRect[p,q,r+2]"];
Print["    = MpRect[p,q,r]    (Laplacian identity)"];
Print[];

recTriples = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}, {0,0,1}};
Do[
  {p, q, r} = triple;
  lhs = N[MpBFace[p+2,q,r] + MpBFace[p,q+2,r] + MpBFace[p,q,r+2], 20];
  rhs = N[MpFace[p,q,r], 20];
  err = Abs[lhs - rhs];
  Print["  [", p, ",", q, ",", r, "]: |LHS - RHS| = ",
        ScientificForm[err, 3],
        If[err < 10^-10, "  OK", "  FAIL"]];
  ,
  {triple, recTriples}
];
Print[];


(* ================================================================ *)
(* Section 6: Edge and corner spot checks                           *)
(* ================================================================ *)

Print["-- Section 6: Edge & corner spot checks --"];

val = MpEdge[0, 0, 0];
Print["  MpEdge[0,0,0] = ", NumberForm[N[val, 16], 16],
      If[Not[MachineNumberQ[val]], "  EXACT", "  numeric"]];

val = MpCorner[0, 0, 0];
Print["  MpCorner[0,0,0] = ", NumberForm[N[val, 16], 16],
      If[Not[MachineNumberQ[val]], "  EXACT", "  numeric"]];


(* ================================================================ *)
(* Summary                                                          *)
(* ================================================================ *)

Print[];
Print["=============================================================="];
nExactA = Count[faceAResults, {_, _, _, True}];
nExactB = Count[faceBResults, {_, _, _, True}];
Print["  Face A-channel: ", nExactA, "/", Length[faceAResults],
      " exact closed forms"];
Print["  Face B-channel: ", nExactB, "/", Length[faceBResults],
      " exact closed forms"];
Print["=============================================================="];
