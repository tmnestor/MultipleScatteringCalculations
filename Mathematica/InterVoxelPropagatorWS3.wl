(* ::Package:: *)
(* InterVoxelPropagatorWS3.wl
   Phase 2C: 3D Fourier NIntegrate reference for the dynamic inter-voxel
   propagator.  Computes ΔP = P(ω) - P(0) via direct numerical integration
   of the subtracted strain kernel in Fourier space.

   The subtracted kernel ΔΓ̃ = Γ̃(ω) - Γ̃(0) is O(ω²/k²) at large k,
   making the 3D integral convergent.  Poles at |k| = ω/c are regularized
   via the retarded prescription ω → ω(1+iε).

   Run with:
     /Applications/Wolfram.app/Contents/MacOS/wolframscript \
        -file Mathematica/InterVoxelPropagatorWS3.wl
*)

$HistoryLength = 0;

Print["================================================================"];
Print["  INTER-VOXEL PROPAGATOR — 3D FOURIER NIntegrate REFERENCE"];
Print["  Phase 2C: Validate power series against direct computation"];
Print["================================================================"];
Print[];

t0global = AbsoluteTime[];

(* ================================================================ *)
(* Section 0. Parameters                                             *)
(* ================================================================ *)

cP = 5000.0;
cS = 3000.0;
rho = 2500.0;
eps = 1.0*^-3;   (* retarded prescription: ω → ω(1+iε) *)
kmax = 4 Pi;     (* integration cutoff — sinc²(2π)=0 at boundary *)

Print["Parameters: cP=", cP, " cS=", cS, " rho=", rho];
Print["eps=", eps, " kmax=", NumberForm[N[kmax], 6]];
Print[];


(* ================================================================ *)
(* Section 1. Integrand definition                                   *)
(*                                                                   *)
(* ΔΓ_{ijkl} = ωc²/(2ρ k⁴) × N_{ijkl}                             *)
(*                                                                   *)
(* N = 2 k_i k_j k_k k_l / D_P                                     *)
(*   + [(k_j k_l δ_{ik} + k_i k_l δ_{jk}) k² - 2 k_i k_j k_k k_l] *)
(*     / D_S                                                         *)
(*                                                                   *)
(* D_P = cP²(ωc² - cP² k²),  D_S = cS²(ωc² - cS² k²)             *)
(*                                                                   *)
(* Full integrand includes sinc² form factor and e^{ik·R} phase.    *)
(* ================================================================ *)

(* Integrand: only evaluates for numeric arguments (_?NumericQ) *)
dpInt[ii_Integer, jj_Integer, kk_Integer, ll_Integer,
      Rv_List, omega_?NumericQ,
      k1_?NumericQ, k2_?NumericQ, k3_?NumericQ] :=
  Module[{kv, kSq, omC2, DP, DS, Qterm, Tterm, numer},
    kv = {k1, k2, k3};
    kSq = k1^2 + k2^2 + k3^2 + 1.0*^-30;  (* regularize k=0 *)
    omC2 = (omega (1.0 + I eps))^2;
    DP = cP^2 (omC2 - cP^2 kSq);
    DS = cS^2 (omC2 - cS^2 kSq);
    Qterm = 2.0 kv[[ii]] kv[[jj]] kv[[kk]] kv[[ll]];
    Tterm = (kv[[jj]] kv[[ll]] KroneckerDelta[ii, kk] +
             kv[[ii]] kv[[ll]] KroneckerDelta[jj, kk]) * kSq;
    numer = Qterm / DP + (Tterm - Qterm) / DS;
    Sinc[k1/2.0]^2 Sinc[k2/2.0]^2 Sinc[k3/2.0]^2 *
      omC2 / (2.0 rho kSq^2) * numer *
      Exp[I (k1 Rv[[1]] + k2 Rv[[2]] + k3 Rv[[3]])] / (2 Pi)^3
  ];


(* ================================================================ *)
(* Section 2. NIntegrate wrapper                                      *)
(* ================================================================ *)

computeDP[ii_, jj_, kk_, ll_, Rv_List, omega_] :=
  Module[{result, t0, reVal, imVal, imRatio},
    t0 = AbsoluteTime[];
    result = NIntegrate[
      dpInt[ii, jj, kk, ll, Rv, omega, k1, k2, k3],
      {k1, -kmax, -2 Pi, 0, 2 Pi, kmax},
      {k2, -kmax, -2 Pi, 0, 2 Pi, kmax},
      {k3, -kmax, -2 Pi, 0, 2 Pi, kmax},
      PrecisionGoal -> 6, MaxRecursion -> 20
    ];
    reVal = Re[result];
    imVal = Im[result];
    imRatio = If[Abs[reVal] > 1.0*^-20, Abs[imVal/reVal], 0.0];
    Print["  DP[", ii, jj, kk, ll, "] R=", Rv,
          " ka=", NumberForm[omega/cP, 4],
          "  Re=", ScientificForm[reVal, 8],
          "  |Im/Re|=", ScientificForm[imRatio, 2],
          "  (", Round[AbsoluteTime[] - t0, 0.1], " s)"];
    reVal
  ];


(* ================================================================ *)
(* Section 3. Face-adjacent R=(1,0,0)                                *)
(*                                                                   *)
(* C₄ᵥ symmetry: 6 independent components                          *)
(*   P₁₁₁₁, P₁₁₂₂, P₂₂₂₂, P₂₂₃₃, P₁₂₁₂, P₂₃₂₃                *)
(* ================================================================ *)

Print["==== Section 3: Face R=(1,0,0) ===="];
Print[];

Rface = {1, 0, 0};

(* Component list: {i, j, k, l} in 1-indexed *)
faceComps = {{1,1,1,1}, {1,1,2,2}, {2,2,2,2}, {2,2,3,3}, {1,2,1,2}, {2,3,2,3}};

(* ka = 0.05 *)
Print["--- ka = 0.05 ---"];
omFace005 = 0.05 * cP;
dpFace005 = Table[
  computeDP[c[[1]], c[[2]], c[[3]], c[[4]], Rface, omFace005],
  {c, faceComps}
];
Print[];

(* ka = 0.1 *)
Print["--- ka = 0.1 ---"];
omFace01 = 0.1 * cP;
dpFace01 = Table[
  computeDP[c[[1]], c[[2]], c[[3]], c[[4]], Rface, omFace01],
  {c, faceComps}
];
Print[];

(* ka = 0.3 *)
Print["--- ka = 0.3 ---"];
omFace03 = 0.3 * cP;
dpFace03 = Table[
  computeDP[c[[1]], c[[2]], c[[3]], c[[4]], Rface, omFace03],
  {c, faceComps}
];
Print[];


(* ================================================================ *)
(* Section 4. Edge-adjacent R=(1,1,0)                                *)
(*                                                                   *)
(* C₂ᵥ symmetry: 6 independent components                          *)
(*   P₁₁₁₁, P₁₁₂₂, P₁₁₃₃, P₃₃₃₃, P₁₁₁₂, P₁₂₃₃                *)
(* ================================================================ *)

Print["==== Section 4: Edge R=(1,1,0) ===="];
Print[];

Redge = {1, 1, 0};
edgeComps = {{1,1,1,1}, {1,1,2,2}, {1,1,3,3}, {3,3,3,3}, {1,1,1,2}, {1,2,3,3}};

Print["--- ka = 0.1 ---"];
omEdge01 = 0.1 * cP;
dpEdge01 = Table[
  computeDP[c[[1]], c[[2]], c[[3]], c[[4]], Redge, omEdge01],
  {c, edgeComps}
];
Print[];


(* ================================================================ *)
(* Section 5. Corner-adjacent R=(1,1,1)                              *)
(*                                                                   *)
(* S₃ symmetry: 4 independent components                            *)
(*   P₁₁₁₁, P₁₁₂₂, P₁₁₁₂, P₁₁₂₃                                *)
(* ================================================================ *)

Print["==== Section 5: Corner R=(1,1,1) ===="];
Print[];

Rcorner = {1, 1, 1};
cornerComps = {{1,1,1,1}, {1,1,2,2}, {1,1,1,2}, {1,1,2,3}};

Print["--- ka = 0.1 ---"];
omCorner01 = 0.1 * cP;
dpCorner01 = Table[
  computeDP[c[[1]], c[[2]], c[[3]], c[[4]], Rcorner, omCorner01],
  {c, cornerComps}
];
Print[];


(* ================================================================ *)
(* Section 6. Export results                                         *)
(* ================================================================ *)

Print["==== Section 6: Export ===="];
Print[];

(* --- Mathematica .wl file --- *)
outFile = "Mathematica/InterVoxelPropagatorWS3Values.wl";

stream = OpenWrite[outFile];
WriteString[stream, "(* Auto-generated by InterVoxelPropagatorWS3.wl *)\n"];
WriteString[stream, "(* ΔP = P(ω) - P(0) from 3D Fourier NIntegrate *)\n"];
WriteString[stream, "(* cP=5000, cS=3000, rho=2500, eps=1e-3, kmax=4π *)\n\n"];

(* Helper to write one value *)
writeVal[s_, name_, val_] :=
  WriteString[s, name, " = ", CForm[val], ";\n"];

(* Face ka=0.05 *)
WriteString[stream, "(* Face R=(1,0,0), ka=0.05 *)\n"];
Do[
  With[{c = faceComps[[n]], v = dpFace005[[n]]},
    writeVal[stream,
      "dpFace005" <> ToString[c[[1]]] <> ToString[c[[2]]] <>
        ToString[c[[3]]] <> ToString[c[[4]]], v]
  ],
  {n, Length[faceComps]}
];

(* Face ka=0.1 *)
WriteString[stream, "\n(* Face R=(1,0,0), ka=0.1 *)\n"];
Do[
  With[{c = faceComps[[n]], v = dpFace01[[n]]},
    writeVal[stream,
      "dpFace01" <> ToString[c[[1]]] <> ToString[c[[2]]] <>
        ToString[c[[3]]] <> ToString[c[[4]]], v]
  ],
  {n, Length[faceComps]}
];

(* Face ka=0.3 *)
WriteString[stream, "\n(* Face R=(1,0,0), ka=0.3 *)\n"];
Do[
  With[{c = faceComps[[n]], v = dpFace03[[n]]},
    writeVal[stream,
      "dpFace03" <> ToString[c[[1]]] <> ToString[c[[2]]] <>
        ToString[c[[3]]] <> ToString[c[[4]]], v]
  ],
  {n, Length[faceComps]}
];

(* Edge ka=0.1 *)
WriteString[stream, "\n(* Edge R=(1,1,0), ka=0.1 *)\n"];
Do[
  With[{c = edgeComps[[n]], v = dpEdge01[[n]]},
    writeVal[stream,
      "dpEdge01" <> ToString[c[[1]]] <> ToString[c[[2]]] <>
        ToString[c[[3]]] <> ToString[c[[4]]], v]
  ],
  {n, Length[edgeComps]}
];

(* Corner ka=0.1 *)
WriteString[stream, "\n(* Corner R=(1,1,1), ka=0.1 *)\n"];
Do[
  With[{c = cornerComps[[n]], v = dpCorner01[[n]]},
    writeVal[stream,
      "dpCorner01" <> ToString[c[[1]]] <> ToString[c[[2]]] <>
        ToString[c[[3]]] <> ToString[c[[4]]], v]
  ],
  {n, Length[cornerComps]}
];

Close[stream];
Print["Wrote ", outFile];

(* --- Python-friendly output --- *)
Print[];
Print["# Python reference values (paste into test file):"];
Print["# Face ka=0.05"];
Do[
  With[{c = faceComps[[n]], v = dpFace005[[n]]},
    Print["WS3_FACE_005_", c[[1]], c[[2]], c[[3]], c[[4]], " = ",
          CForm[v]]
  ],
  {n, Length[faceComps]}
];
Print["# Face ka=0.1"];
Do[
  With[{c = faceComps[[n]], v = dpFace01[[n]]},
    Print["WS3_FACE_01_", c[[1]], c[[2]], c[[3]], c[[4]], " = ",
          CForm[v]]
  ],
  {n, Length[faceComps]}
];
Print["# Face ka=0.3"];
Do[
  With[{c = faceComps[[n]], v = dpFace03[[n]]},
    Print["WS3_FACE_03_", c[[1]], c[[2]], c[[3]], c[[4]], " = ",
          CForm[v]]
  ],
  {n, Length[faceComps]}
];
Print["# Edge ka=0.1"];
Do[
  With[{c = edgeComps[[n]], v = dpEdge01[[n]]},
    Print["WS3_EDGE_01_", c[[1]], c[[2]], c[[3]], c[[4]], " = ",
          CForm[v]]
  ],
  {n, Length[edgeComps]}
];
Print["# Corner ka=0.1"];
Do[
  With[{c = cornerComps[[n]], v = dpCorner01[[n]]},
    Print["WS3_CORNER_01_", c[[1]], c[[2]], c[[3]], c[[4]], " = ",
          CForm[v]]
  ],
  {n, Length[cornerComps]}
];

Print[];
Print["Total time: ", Round[AbsoluteTime[] - t0global, 1], " s"];
Print["Done."];
