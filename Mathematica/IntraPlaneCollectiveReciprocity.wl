#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneCollectiveReciprocity.wl  --  Phase 2 item (d): RECIPROCITY of the
   collective (lattice) Foldy-Lax operator, in the SYMPLECTIC metric.

   The collective T of the reference cell is  T_coll = T0 (I - G0^vec T0)^{-1}, with
   T0 the single-site Mie operator (CartesianT0, symplectic-bracket) and G0^vec the
   lattice-summed vector coupling from item (b) (loaded from the dump).

   The physical (Betti / symplectic) reciprocity is the sigma-metric operator symmetry
       J0 (D A D^{-1}) J0 = (D A D^{-1})^T,   J0 = diag((-1)^{n+m}) (x) (m -> -m),
   where D = diag(d_c(n)) is the symplectic CHANNEL metric that reconciles the two
   building blocks (CartesianT0 "clean" L/M/N and the Phase-0/1 G0).  D is fixed (not
   fitted): a SINGLE D makes T0 sigma-symmetric (from its L-N P-SV coupling) AND G0
   sigma-symmetric (from its M-N SH-SV coupling):
       d_L(n) = (kS/kP)^(3/2)        (P channel; the P-vs-S wavenumber factor)
       d_N(n) = sqrt(n(n+1))         (SV poloidal; the Jspher n(n+1) weight)
       d_M(n) = I sqrt(n(n+1))       (SH toroidal; n(n+1) + the SH/SV I-phase of the
                                      incN ~ I^{n+1} / incM ~ I^n elastic bridge)
   The Foldy-Lax Born series T0 (G0 T0)^k is palindromic, so the collective inherits
   the symmetry: T0', G0' sigma-symmetric => T_coll' sigma-symmetric => reciprocal.

   NB this resolves the metric mismatch found earlier (a naive arbitrary-direction
   far-field test is the WRONG reciprocity statement for a lattice at fixed k_par).
   ENERGY / flux balance |R|^2+|T|^2=1 still needs the UNDAMPED G0 (the damping that
   regularises the lattice sum is an artificial loss) + Kennett flux norm: that is
   Phase 3.  Reciprocity is a symmetry and holds at damped (complex) k.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0;

(* ---- single-site T0 in the L/M/N idx basis ---- *)
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

(* ---- load the lattice-summed vector G0 (item b dump) ---- *)
g0data = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneVectorLattice_reference.json", "RawJSON"];
G0vec = Map[#[[1]] + I #[[2]] &, g0data["G0vec"], {2}];

(* ---- sigma m-flip conjugation and the symplectic channel metric D ---- *)
sig[{n_, m_, c_}] := (-1)^(n + m);
conjIdx[{n_, m_, c_}] := {n, -m, c};
J0 = Table[If[idx[[i]] == conjIdx[idx[[k]]], sig[idx[[k]]], 0], {i, nDim}, {k, nDim}];
dwt[{n_, m_, c_}] := Switch[c, "L", (kSo/kPo)^(3/2), "N", Sqrt[n (n + 1)], "M", I Sqrt[n (n + 1)]];
Dmet = DiagonalMatrix[dwt /@ idx];
Dinv = Inverse[Dmet];

(* sigma-reciprocity residual in the symplectic metric: |J0 (D A D^-1) J0 - (D A D^-1)^T| *)
recip[A_] := Module[{Bc = Dmet . A . Dinv}, Max[Abs[Flatten[J0 . Bc . J0 - Transpose[Bc]]]]];

Print["==== Phase 2 item (d) :: collective reciprocity (symplectic metric) ===="];
Print["  loaded G0^vec ", Dimensions[G0vec], " at k_par=(", g0data["kx"], ",", g0data["ky"],
   "), dampIm=", g0data["dampIm"], ", Lrad=", g0data["Lrad"]];
Print["  symplectic metric D = diag( d_L=(kS/kP)^3/2=", N[(kSo/kPo)^(3/2)],
   ", d_M=I sqrt(n(n+1)), d_N=sqrt(n(n+1)) )"];

(* [recon-T0] T0 is sigma-reciprocal in the symplectic metric *)
rT0 = recip[T0mat];
Print["  [recon-T0] J0 (D T0 D^-1) J0 = (D T0 D^-1)^T : ", rT0,
   "  -> ", If[rT0 < 1*^-12, "PASS", "FAIL"]];

(* [recon-G0] G0 is sigma-reciprocal in the SAME metric (the reconciliation) *)
rG0 = recip[G0vec];
Print["  [recon-G0] J0 (D G0 D^-1) J0 = (D G0 D^-1)^T : ", rG0,
   "  -> ", If[rG0 < 1*^-10, "PASS", "FAIL"]];

(* [collective] T_coll is sigma-reciprocal => the lattice scattering is reciprocal *)
Imat = IdentityMatrix[nDim];
Tcoll = T0mat . Inverse[Imat - G0vec . T0mat];
rTc = recip[Tcoll];
couple = Norm[Flatten[Tcoll - T0mat]]/Norm[Flatten[T0mat]];
Print["  [collective] J0 (D T_coll D^-1) J0 = (D T_coll D^-1)^T : ", rTc,
   "  -> ", If[rTc < 1*^-10, "PASS", "FAIL"]];
Print["  [coupling] ||T_coll - T0|| / ||T0|| = ", couple, "  -> ", If[couple > 1*^-6, "PASS", "FAIL"]];
Print["  NOTE energy/flux balance |R|^2+|T|^2=1 needs the undamped G0 + Kennett flux norm (Phase 3)."];
Print["Phase 2 item (d) (IntraPlaneCollectiveReciprocity.wl) loaded + verified."];
