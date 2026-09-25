#!/usr/bin/env wolframscript
(* ============================================================================
   InterfaceUnitarity.wl  --  P-SV / SH unitarity of the plane-interface
   scattering matrix, from the thesis energy-normalised eigenbasis.

   ANCHOR: Nestor (1996) Section 3.1, GRepresentations.tex -- (Peigen), (SVeigen),
   (SHeigen), (epsdef).  The eigenbasis is loaded from ThesisInterfaceRT.wl, which
   gates it (symplectic identity, Hooke consistency, down-incidence power).

   WHAT THIS ADDS.  ThesisInterfaceRT.wl checks only the column norms of the
   down-incidence block at one lossless point.  Here the FULL two-sided S-matrix
   is built (incidence from above AND below), so S^dagger S = I tests both the
   column norms (energy) and the column orthogonality (phase), and it is
   exercised across slowness, past a critical angle, and with attenuation.

   INDEPENDENCE.  R/T here come from continuity of the displacement-traction
   6-vector across z = 0.  kennett_layers.py uses the Aki & Richards closed-form
   coefficients.  The two share no code and no algebra.  Both are flux-normalised,
   so they may differ only by unit-modulus per-channel phases; |S_ij| and
   eig(S^dagger S) are invariant under those and are exported for the Python
   gate scripts/gate_interface_unitarity.py.

   CHANNELS.  ky = 0, kx = om p, so P-SV and SH decouple.  Medium 1 above,
   medium 2 below.  With b = D1.(d1;u1) = D2.(d2;u2) and Q = D1^{-1} D2:
        Td = Q11^{-1},  Rd = Q21 Td,  Ru = -Td Q12,  Tu = Q22 - Q21 Td Q12,
        S  = [[Rd, Tu], [Td, Ru]]   rows (u1; d2), columns (d1; u2),
   each block ordered (P, S, H).  P-SV = {1,2,4,5}, SH = {3,6}.

   ATTENUATION uses the kennett_layers._complex_slowness convention,
        s = (1/v) (2Q)^2/(1+(2Q)^2) + i (1/v) 2Q/(1+(2Q)^2),   Im s > 0,
   and the principal Sqrt gives Re kz > 0, Im kz > 0 as the code's branch does.

   Run:  wolframscript -file Mathematica/InterfaceUnitarity.wl
   ============================================================================ *)

(* ============================================================================
   1. Setup: thesis eigenbasis D_z, the two media, the attenuation convention.
      baseDir works both under wolframscript ($InputFileName) and in the .nb twin.
   ============================================================================ *)
baseDir = If[$InputFileName =!= "", DirectoryName[$InputFileName], NotebookDirectory[]];
Get[FileNameJoin[{baseDir, "ThesisInterfaceRT.wl"}]];

wp = 50;                                        (* working precision, digits *)
med1 = {3, 17/10, 23/10};                        (* alpha, beta, rho: km/s, g/cm^3 *)
med2 = {42/10, 24/10, 27/10};

cSlow[v_, Q_] := If[Q === Infinity, 1/v,
   With[{t = 2 Q}, (t^2 + I t)/((1 + t^2) v)]];
cMedium[{al_, be_, rh_}, {QP_, QS_}] := {1/cSlow[al, QP], 1/cSlow[be, QS], rh};

(* ============================================================================
   2. Two-sided interface S-matrix from continuity of b across z = 0.
      Q = D1^{-1} D2 in 3x3 blocks; S = [[Rd, Tu], [Td, Ru]], rows (u1; d2),
      columns (d1; u2).  Block helpers and the residual / eigenvalue measures.
   ============================================================================ *)
sMatrix[m1_, m2_, om_, p_] := Module[{D1, D2, Q, Q11, Q12, Q21, Q22, Td, Rd, Ru, Tu},
   D1 = Dz[Sequence @@ m1, om, om p, 0];
   D2 = Dz[Sequence @@ m2, om, om p, 0];
   Q = Inverse[D1] . D2;
   Q11 = Q[[1 ;; 3, 1 ;; 3]]; Q12 = Q[[1 ;; 3, 4 ;; 6]];
   Q21 = Q[[4 ;; 6, 1 ;; 3]]; Q22 = Q[[4 ;; 6, 4 ;; 6]];
   Td = Inverse[Q11]; Rd = Q21 . Td; Ru = -Td . Q12; Tu = Q22 - Q21 . Td . Q12;
   ArrayFlatten[{{Rd, Tu}, {Td, Ru}}]];

psv = {1, 2, 4, 5}; sh = {3, 6};
unitResid[S_] := Max[Abs[Flatten[ConjugateTranspose[S] . S - IdentityMatrix[Length[S]]]]];
gramEig[S_] := Sort[Re[Eigenvalues[ConjugateTranspose[S] . S]]];
sci[x_] := With[{v = N[x]}, If[v == 0, "0 (to working precision)", TextString[ScientificForm[v, 3]]]];
verdict[ok_] := If[TrueQ[ok], "PASS", "FAIL"];

(* ============================================================================
   3. Cases, evaluated at 50 digits.  p = 3/10 exceeds 1/alpha2 = 0.238, so P in
      medium 2 is evanescent there.  Each S is also built at om = 7 for gate [1b].
   ============================================================================ *)
cases = {
   <|"name" -> "lossless_p0", "p" -> 0, "Q" -> {Infinity, Infinity}|>,
   <|"name" -> "lossless_p0.1", "p" -> 1/10, "Q" -> {Infinity, Infinity}|>,
   <|"name" -> "lossless_p0.2", "p" -> 2/10, "Q" -> {Infinity, Infinity}|>,
   <|"name" -> "lossless_p0.3", "p" -> 3/10, "Q" -> {Infinity, Infinity}|>,
   <|"name" -> "lossy_Q50_30_p0.1", "p" -> 1/10, "Q" -> {50, 30}|>,
   <|"name" -> "lossy_Q20_10_p0.1", "p" -> 1/10, "Q" -> {20, 10}|>};

Print["==== InterfaceUnitarity :: two-sided interface S-matrix, ", wp, " digits ===="];
Print["  HS1 {a,b,r} = ", N[med1], "   HS2 = ", N[med2]];

results = Table[
   Module[{p = c["p"], m1, m2, S, S7, Spsv, Ssh, coup, omDev},
    m1 = N[cMedium[med1, c["Q"]], wp]; m2 = N[cMedium[med2, c["Q"]], wp];
    S = sMatrix[m1, m2, N[1, wp], N[p, wp]];
    S7 = sMatrix[m1, m2, N[7, wp], N[p, wp]];
    Spsv = S[[psv, psv]]; Ssh = S[[sh, sh]];
    coup = Max[Abs[Flatten[{S[[psv, sh]], S[[sh, psv]]}]]];
    omDev = Max[Abs[Flatten[S - S7]]];
    <|"name" -> c["name"], "p" -> N[p], "QP" -> ToString[c["Q"][[1]]], "QS" -> ToString[c["Q"][[2]]],
      "S" -> S, "Spsv" -> Spsv, "Ssh" -> Ssh, "coupling" -> coup, "omegaDev" -> omDev,
      "residPSV" -> unitResid[Spsv], "residSH" -> unitResid[Ssh],
      "eigPSV" -> gramEig[Spsv], "eigSH" -> gramEig[Ssh]|>],
   {c, cases}];
byName = AssociationThread[#["name"] & /@ results, results];

(* ============================================================================
   4. Gate [1] structure: P-SV and SH decouple at ky = 0; the interface S is
      independent of omega at fixed p.
   ============================================================================ *)
g1 = Max[#["coupling"] & /@ results]; g1b = Max[#["omegaDev"] & /@ results];
Print["  [1] P-SV <-> SH coupling (ky=0) max = ", sci[g1], " -> ", verdict[g1 < 10^-40]];
Print["  [1b] S(om=1) vs S(om=7)          max = ", sci[g1b], " -> ", verdict[g1b < 10^-40]];

(* ============================================================================
   5. Gate [2] lossless, all channels open: S^dagger S = I for each block.
   ============================================================================ *)
Do[With[{r = byName[n]},
   Print["  [2] ", n, ": |S^+S-I| P-SV = ", sci[r["residPSV"]], ", SH = ", sci[r["residSH"]],
    " -> ", verdict[Max[r["residPSV"], r["residSH"]] < 10^-40]]],
  {n, {"lossless_p0", "lossless_p0.1", "lossless_p0.2"}}];

(* ============================================================================
   6. Gate [3] past the P critical slowness of medium 2: the full 4x4 fails
      (negative control); the restriction to the open channels is unitary.
   ============================================================================ *)
With[{r = byName["lossless_p0.3"]},
  Module[{keep = {1, 2, 4}, Sr},
   Sr = r["Spsv"][[keep, keep]];
   Print["  [3a] p=0.3, all 4 P-SV channels  |S^+S-I| = ", sci[r["residPSV"]],
    "  (negative control: must be O(1)) -> ", verdict[r["residPSV"] > 10^-1]];
   Print["  [3b] p=0.3, 3 open channels      |S^+S-I| = ", sci[unitResid[Sr]],
    " -> ", verdict[unitResid[Sr] < 10^-40]];
   Print["  [3c] p=0.3, SH (both open)       |S^+S-I| = ", sci[r["residSH"]],
    " -> ", verdict[r["residSH"] < 10^-40]];
   byName["lossless_p0.3"] = Append[r, "residOpen" -> unitResid[Sr]]]];

(* ============================================================================
   7. Gate [4] attenuation: not unitary, and not a contraction -- the
      eigenvalues of S^dagger S straddle 1.
   ============================================================================ *)
Do[With[{r = byName[n]},
   Print["  [4] ", n, ": eig(S^+S) P-SV = ", TextString[NumberForm[N[r["eigPSV"]], 6]],
    "  SH = ", TextString[NumberForm[N[r["eigSH"]], 6]]];
   Print["      straddles 1 (P-SV and SH) -> ", verdict[Min[r["eigPSV"]] < 1 < Max[r["eigPSV"]] &&
      Min[r["eigSH"]] < 1 < Max[r["eigSH"]]]]],
  {n, {"lossy_Q50_30_p0.1", "lossy_Q20_10_p0.1"}}];

(* ============================================================================
   8. Gate [5] reciprocity, in THIS basis's phase convention.  Not S = S^T (that
      is the Kennett form): here P-SV obeys S^T = SigHat.S.SigHat, SigHat =
      diag(1,-1,1,-1), i.e. Rd^T = Sig.Rd.Sig, Ru^T = Sig.Ru.Sig, Tu = Sig.Td^T.Sig.
      Bilinear, so it must hold WITH attenuation too.  SH: plain symmetry.
   ============================================================================ *)
sigHat = DiagonalMatrix[{1, -1, 1, -1}];
recipPSV = Max[Abs[Flatten[Transpose[#["Spsv"]] - sigHat . #["Spsv"] . sigHat]]] & /@ results;
recipSH = Max[Abs[Flatten[Transpose[#["Ssh"]] - #["Ssh"]]]] & /@ results;
plainPSV = Max[Abs[Flatten[Transpose[#["Spsv"]] - #["Spsv"]]]] & /@ Select[results, #["p"] > 0 &];  (* p = 0: no conversion *)
Print["  [5] P-SV  S^T = SigHat.S.SigHat, all cases incl. lossy: max = ", sci[Max[recipPSV]],
  " -> ", verdict[Max[recipPSV] < 10^-40]];
Print["  [5] SH    S^T = S,               all cases incl. lossy: max = ", sci[Max[recipSH]],
  " -> ", verdict[Max[recipSH] < 10^-40]];
Print["  [5] P-SV  plain S^T = S (wrong form), p > 0:              min = ", sci[Min[plainPSV]],
  "  (negative control: must be O(1)) -> ", verdict[Min[plainPSV] > 10^-2]];

(* ============================================================================
   9. Export for scripts/gate_interface_unitarity.py: rephasing-invariant
      quantities |S_ij| and eig(S^dagger S), at machine precision.
   ============================================================================ *)
absM[m_] := N[Abs[m]];
Export[FileNameJoin[{baseDir, "InterfaceUnitarity_reference.json"}],
  <|"source" -> "Mathematica/InterfaceUnitarity.wl",
    "working_precision" -> wp,
    "medium1" -> N[med1], "medium2" -> N[med2],
    "q_convention" -> "kennett_layers._complex_slowness (Im s > 0)",
    "cases" -> Table[With[{r = byName[c["name"]]},
       <|"name" -> r["name"], "p" -> r["p"], "QP" -> r["QP"], "QS" -> r["QS"],
         "absSpsv" -> absM[r["Spsv"]], "absSsh" -> absM[r["Ssh"]],
         "eigPSV" -> N[r["eigPSV"]], "eigSH" -> N[r["eigSH"]],
         "residPSV" -> N[r["residPSV"]], "residSH" -> N[r["residSH"]]|>], {c, cases}]|>];
Print["  wrote InterfaceUnitarity_reference.json"];
Print["InterfaceUnitarity.wl loaded."];
