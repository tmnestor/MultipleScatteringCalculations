#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneSymplecticMetric.wl  --  Phase 2 item (d) reconciliation DIAGNOSTIC.
   Characterise exactly which reciprocity symmetry T0 (clean L/M/N) and G0
   (Phase-0/1) each satisfy, and extract the symplectic channel metric, so the
   transform that puts BOTH in one metric is determined (not guessed).

   Symmetry tests on a matrix block B (idx-indexed):
     plain : |B - B^T|
     sigma : |J0 B J0 - B^T|,   J0 = diag((-1)^{n+m}) (x) (m -> -m)
     wD    : same with B -> D B D^{-1}, D = diag(d_c(n)) channel/n weights.
   Metric ratio from T0's L-N coupling: tLN/tNL per n vs sqrt(n(n+1)), n(n+1).
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0;

T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
Nmax = 2;
idx = Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];
nDim = Length[idx];
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0entry[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
  If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
T0mat = Table[T0entry[idx[[i]], idx[[j]]], {i, nDim}, {j, nDim}];

g0data = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneVectorLattice_reference.json", "RawJSON"];
G0vec = Map[#[[1]] + I #[[2]] &, g0data["G0vec"], {2}];

(* sigma m-flip conjugation *)
sig[{n_, m_, c_}] := (-1)^(n + m);
conjIdx[{n_, m_, c_}] := {n, -m, c};
J0 = Table[If[idx[[i]] == conjIdx[idx[[k]]], sig[idx[[k]]], 0], {i, nDim}, {k, nDim}];

(* channel-weight diagonal D = diag(d_c(n)) *)
mkD[dL_, dM_, dN_] := DiagonalMatrix[Table[Switch[idx[[i, 3]], "L", dL[idx[[i, 1]]], "M", dM[idx[[i, 1]]], "N", dN[idx[[i, 1]]]], {i, nDim}]];

(* positions per channel *)
Lpos = Flatten[Position[idx, {_, _, "L"}]];
Mpos = Flatten[Position[idx, {_, _, "M"}]];
Npos = Flatten[Position[idx, {_, _, "N"}]];
MNpos = Sort[Join[Mpos, Npos]];

(* symmetry of the sub-block at positions pos, with J0 restricted to pos *)
plainSymB[full_, pos_] := Max[Abs[Flatten[full[[pos, pos]] - Transpose[full[[pos, pos]]]]]];
sigSymB[full_, pos_] := Module[{B = full[[pos, pos]], Jb = J0[[pos, pos]]},
   Max[Abs[Flatten[Jb . B . Jb - Transpose[B]]]]];

Print["==== Phase 2 (d) reconciliation diagnostic :: which symmetry holds ===="];
Print["  -- T0 (clean) --"];
Print["   T0 full        : plain ", plainSymB[T0mat, Range[nDim]], ", sigma ", sigSymB[T0mat, Range[nDim]]];
Print["   T0 L-block     : plain ", plainSymB[T0mat, Lpos], ", sigma ", sigSymB[T0mat, Lpos]];
Print["   T0 M-block     : plain ", plainSymB[T0mat, Mpos], ", sigma ", sigSymB[T0mat, Mpos]];
Print["   T0 N-block     : plain ", plainSymB[T0mat, Npos], ", sigma ", sigSymB[T0mat, Npos]];
Print["  -- G0 (Phase-0/1) --"];
Print["   G0 full        : plain ", plainSymB[G0vec, Range[nDim]], ", sigma ", sigSymB[G0vec, Range[nDim]]];
Print["   G0 L-block     : plain ", plainSymB[G0vec, Lpos], ", sigma ", sigSymB[G0vec, Lpos]];
Print["   G0 M-block     : plain ", plainSymB[G0vec, Mpos], ", sigma ", sigSymB[G0vec, Mpos]];
Print["   G0 N-block     : plain ", plainSymB[G0vec, Npos], ", sigma ", sigSymB[G0vec, Npos]];
Print["   G0 M<->N block : |G0[M,N] - G0[N,M]^T| (plain) ", Max[Abs[Flatten[G0vec[[Mpos, Npos]] - Transpose[G0vec[[Npos, Mpos]]]]]]];

(* metric weight that symmetrises T0's L-N block: d_N(n) = sqrt(tLN/tNL) *)
dNT0[n_] := dNT0[n] = Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa]},
   Sqrt[ts[[1, 2]]/ts[[2, 1]]]];
one[n_] := 1; sq[n_] := Sqrt[n (n + 1)];
Print["  -- d_N(n) = sqrt(tLN/tNL) from T0 (the symplectic N-weight) --"];
Do[Print["   n=", n, ": d_N = ", dNT0[n], ", sqrt(n(n+1)) = ", N[Sqrt[n (n + 1)]],
    ", (kPo/kSo)^(3/2) sqrt(n(n+1)) = ", N[(kPo/kSo)^(3/2) Sqrt[n (n + 1)]]], {n, 1, Nmax}];

(* Reparametrise: put the kP/kS factor on the L (P) channel, leave N,M the pure
   symplectic S-wave weights.  d_L = (kS/kP)^(3/2), d_N = sqrt(n(n+1)).  This
   symmetrises T0's L-N identically AND gives the S-S (M-N) coupling the right weight. *)
sigAsymB[D_, pos_] := Module[{Bc = D . G0vec . Inverse[D]},
   Max[Abs[Flatten[(J0 . Bc . J0 - Transpose[Bc])[[pos, pos]]]]]];
sigAsymF[A_, D_] := Module[{Bc = D . A . Inverse[D]}, Max[Abs[Flatten[J0 . Bc . J0 - Transpose[Bc]]]]];
dL[n_] := (kSo/kPo)^(3/2); dNsy[n_] := Sqrt[n (n + 1)];
Print["  -- D = diag(d_L=(kS/kP)^3/2, d_M, d_N=sqrt(n(n+1))); T0 sigma-asym = ",
   sigAsymF[T0mat, mkD[dL, one, dNsy]], " (should ~0)"];
(* Determine d_M directly: sigma-symmetry of the M-N coupling requires
   d_M(nu)^2/d_N(n)^2 = G0[(n m N),(nu mu M)] / (sig(nu,mu) sig(n,m) G0[(nu,-mu,M),(n,-m,N)]).
   Compute the implied d_M(nu)^2 (with d_N=sqrt(n(n+1))) over several elements. *)
posOf[t_] := First[First[Position[idx, t]]];
g0el[a_, b_] := G0vec[[posOf[a], posOf[b]]];
impliedDM2[nn_, mm_, nuu_, muu_] := Module[{ratio},
   ratio = g0el[{nn, mm, "N"}, {nuu, muu, "M"}]/
     (sig[{nuu, muu, "x"}] sig[{nn, mm, "x"}] g0el[{nuu, -muu, "M"}, {nn, -mm, "N"}]);
   ratio (nn (nn + 1))];   (* = d_M(nu)^2 *)
Print["  -- implied d_M(nu)^2 from M-N coupling elements (d_N=sqrt(n(n+1))) --"];
Do[Print["   (n=", e[[1]], ",m=", e[[2]], ")N <-> (nu=", e[[3]], ",mu=", e[[4]], ")M : d_M(", e[[3]],
    ")^2 = ", impliedDM2[e[[1]], e[[2]], e[[3]], e[[4]]],
    "   vs -n(n+1)=", -e[[3]] (e[[3]] + 1), ", -(nu(nu+1))=", N[-e[[3]] (e[[3]] + 1)]],
  {e, {{1, 0, 1, 0}, {1, 1, 1, 1}, {2, 0, 1, 0}, {1, 0, 2, 1}, {2, -1, 2, 1}, {2, 1, 2, -1}}}];
(* test the complex metric d_M(n) = I sqrt(n(n+1)) (and -I) *)
cand2 = {{"I sqrt(n(n+1))", Function[n, I Sqrt[n (n + 1)]]}, {"-I sqrt(n(n+1))", Function[n, -I Sqrt[n (n + 1)]]}};
Do[Print["   d_M=", c[[1]], " : G0 full sigma-asym = ", sigAsymF[G0vec, mkD[dL, c[[2]], dNsy]],
    ", M-N = ", sigAsymB[mkD[dL, c[[2]], dNsy], MNpos]], {c, cand2}];
Print["Diagnostic done."];
