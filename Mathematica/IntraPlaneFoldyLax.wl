#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneFoldyLax.wl  --  Phase 2 of the Intra-Plane Foldy-Lax program:
   the planar Foldy-Lax solve  (I - G0 . T0) a_exc = a_inc,  b = T0 . a_exc,
   in the spherical L/M/N multipole basis.

   T0 (single-site full-wave Mie T-matrix) comes from the verified CartesianT0.wl;
   G0 (intra-plane coupling at Bloch vector k_par) comes from Phase 1.

   TRACER BULLET 1 (this file): single-site T0 assembly + single-voxel limit
   (G0 = 0  =>  collective T = T0) + the closed monopole-channel collective solve
   with the Phase-1 scalar G0.  The full multi-channel vector G0 coupling needs the
   Phase-0 vector translation as an explicit L/M/N matrix (next tracer bullet).

   Conventions inherited from CartesianT0.wl: time e^{+i w t}, outgoing h_n^(1),
   clean L/M/N normalization, (x,y,z) with z = polar axis = depth.
   ============================================================================ *)

(* ---- single-site T_n from the verified CartesianT0.wl ---- *)
Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0;

(* per-(n,m) single-site T block in the (L,M,N) channel basis (sphere: diagonal in n,m) *)
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};               (* L only *)
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];  (* (L,M,N) x (L,M,N) *)

(* ---- multipole index basis and block-diagonal single-site T0 matrix ---- *)
NmaxFL = 2;
idx = Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, NmaxFL}], 1];
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0entry[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
  If[n1 == n2 && m1 == m2,
   If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
T0mat = Table[T0entry[idx[[i]], idx[[j]]], {i, Length[idx]}, {j, Length[idx]}];

(* ---- Foldy-Lax collective T-matrix:  T_coll = T0 (I - G0 T0)^{-1} ---- *)
collectiveT[G0mat_] := T0mat . Inverse[IdentityMatrix[Length[idx]] - G0mat . T0mat];

(* ============================================================================
   [a] single-site T0 sanity: assembled blocks reproduce CartesianT0 T_n
   ============================================================================ *)
ts1 = TsphClean[1, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa];
(* (L,M,N) at positions (1,2,3): spheroidal L-N coupling is at [[1,3]]/[[3,1]], SH at [[2,2]] *)
sane = Max[Abs[{T0LMN[1][[1, 1]] - ts1[[1, 1]], T0LMN[1][[1, 3]] - ts1[[1, 2]],
     T0LMN[1][[3, 1]] - ts1[[2, 1]], T0LMN[1][[3, 3]] - ts1[[2, 2]],
     T0LMN[1][[2, 2]] - Ttoroidal[1, kSo, muO, kSi, muI, aa]}]];
Print["==== Phase 2 :: single-site T0 + Foldy-Lax (TB1) ===="];
Print["  basis size = ", Length[idx], " multipole channels (n=0..", NmaxFL, ")"];
Print["  [a] T0 blocks reproduce CartesianT0 T_n: max dev = ", sane, "  -> ", If[sane < 1*^-14, "PASS", "FAIL"]];

(* ============================================================================
   [b] single-voxel limit: with G0 = 0 the collective T equals the single-site T0
   ============================================================================ *)
voxResid = Max[Abs[Flatten[collectiveT[0 T0mat] - T0mat]]];
Print["  [b] single-voxel limit (G0=0) collective T = T0: max dev = ", voxResid,
   "  -> ", If[voxResid < 1*^-12, "PASS", "FAIL"]];

(* ============================================================================
   [c] closed monopole-channel collective solve with the Phase-1 scalar G0.
   Monopole-monopole coupling G0^mono(k_par) = Sum_{R!=0} h_0(kappaP|R|) e^{i k_par.R}
   (damped so the sum converges); collective t_coll = t/(1 - G0^mono t), t = T0mono.
   ============================================================================ *)
aL = 2.0; kxB = 0.2; kyB = 0.1; LradFL = 24; kappaPd = kPo + 0.25 I;
shank0[x_] := SphericalBesselJ[0, x] + I SphericalBesselY[0, x];
latIJ = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -LradFL, LradFL}, {j, -LradFL, LradFL}], 1];
G0mono = Total[Map[Function[ij,
     shank0[kappaPd aL Sqrt[ij . ij]] Exp[I aL (kxB ij[[1]] + kyB ij[[2]])]], latIJ]];
tMono = T0mono[kPo, lamO, muO, kPi, lamI, muI, aa];
tCollMono = tMono/(1 - G0mono tMono);
tCollIso = tMono/(1 - 0 tMono);
Print["  [c] monopole collective solve (Phase-1 G0, damped):"];
Print["      G0^mono(k_par) = ", G0mono];
Print["      isolated  t = ", tMono];
Print["      collective t = ", tCollMono];
Print["      reduces to isolated as G0->0: ", If[Abs[tCollIso - tMono] < 1*^-14, "PASS", "FAIL"],
   ",  collective differs from isolated: ", If[Abs[tCollMono - tMono] > 1*^-6, "PASS", "FAIL"]];
Print["Phase 2 TB1 (IntraPlaneFoldyLax.wl) loaded + verified."];
