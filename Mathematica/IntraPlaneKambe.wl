(* ============================================================================ *)
(* IntraPlaneKambe.wl — Phase 3b cycle 1: undamped multipole structure constants *)
(* D[q,s] via multipole projection of the undamped scalar Ewald field. Standard  *)
(* Kambe/layer-KKR OBJECT, built by projection of the validated TB2 scalar Ewald *)
(* (NOT per-q analytic forms). Time e^{-i w t}, outgoing h_q^(1); lattice in x-y, *)
(* a_L=2, k_par=(kx,ky). Self-contained: copies the small TB2 pieces it needs    *)
(* (does NOT Get IntraPlaneLatticeSum.wl, which runs the Phase-1 study on load).  *)
(* ============================================================================ *)

reim[z_] := {Re[N[z]], Im[N[z]]};
aL = 2.0; kx = 0.2; ky = 0.1; kpar2 = {kx, ky};
Aarea = aL^2; recipB = 2 Pi/aL;
sh[n_, x_] := SphericalHankelH1[n, x]; sj[n_, x_] := SphericalBesselJ[n, x];

(* ============================================================================ *)
(* Task 1: general-z scalar Ewald field (real + reciprocal halves) + direct sum  *)
(* ============================================================================ *)

(* real-space half, general z: 3D distance d = |r - (aL i, aL j, 0)| *)
ewReal3[kappa_, r_, eta_, Rc_] := (1/(8 Pi)) Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      (Exp[I aL (kx i + ky j)]/d) Sum[Exp[s I kappa d] Erfc[d eta + s I kappa/(2 eta)], {s, {-1, 1}}]],
     {i, -Rc, Rc}, {j, -Rc, Rc}], 1]];

(* bare scalar GF self-term g(r) = e^{i kappa |r|}/(4 Pi |r|) — the R=0 site, EXCLUDED from
   the structure-constant sum (Sum_{R!=0}); it is the singular h_0 self-field, not regular-
   multipole expandable, so it must be removed before the j_q projection (cf. TB2 gLatField/gK). *)
gK[kappa_, rn_] := Exp[I kappa rn]/(4 Pi rn);

(* damped direct 3D lattice GF over R!=0 (ground truth for the projection-method gate).
   Memoised on (kappa,r,Lbig): the projection re-evaluates the field at the SAME sphere points
   across all (q,s) pairs, so caching gives a ~Nq^2-fold speedup with identical values. *)
ewDirect3[kappa_, r_, Lbig_] := ewDirect3[kappa, r, Lbig] = Total[Flatten[Table[
     If[i == 0 && j == 0, 0,
      With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
       Exp[I kappa d]/(4 Pi d) Exp[I aL (kx i + ky j)]]], {i, -Lbig, Lbig}, {j, -Lbig, Lbig}], 1]];

(* K_n = k_par + G ; kz = sqrt(kappa^2 - |K_n|^2) (Im kz >= 0); z-dependent erfc pair.
   At z=0 this reduces to TB2 ewaldRecip: (I/(2 Aarea)) (Exp[I kpg.rho]/kz) Erfc[kz/(2 I eta)]. *)
ewRecip3[kappa_, r_, eta_, Gc_] := (I/(4 Aarea)) Total[Flatten[Table[
     With[{kpg = kpar2 + recipB {m, n}, z = r[[3]]},
      With[{kz = Sqrt[kappa^2 - kpg . kpg]},
       (* general-z reciprocal half: from int_{1/eta}^inf e^{-z^2/w^2} e^{kz^2 w^2/4} dw via
          int_t^inf e^{-p^2 w^2 - q^2/w^2} dw = (Sqrt[Pi]/4p)(e^{2pq}erfc(pt+q/t)+e^{-2pq}erfc(pt-q/t)),
          p=kz/(2I), q=|z|, t=1/eta. The growing e^{+|z|gamma} evanescent term pairs with the
          faster-decaying erfc(+|z|eta+.) . At z=0 both erfc->erfc(kz/2I eta), recovering TB2. *)
       (Exp[I kpg . {r[[1]], r[[2]]}]/kz) (
          Exp[-I kz Abs[z]] Erfc[ Abs[z] eta + kz/(2 I eta)]
        + Exp[ I kz Abs[z]] Erfc[-Abs[z] eta + kz/(2 I eta)])]],
     {m, -Gc, Gc}, {n, -Gc, Gc}], 1]];
(* ewReal3 + ewRecip3 = Sum_{all R} g(r-R); subtract the R=0 self-term gK to get Sum_{R!=0},
   the regular field projected for the structure constants (matches TB2 ewaldTotal - gK). *)
ewTot3[kappa_, r_, eta_, Rc_, Gc_] := ewTot3[kappa, r, eta, Rc, Gc] =
  ewReal3[kappa, r, eta, Rc] + ewRecip3[kappa, r, eta, Gc] - gK[kappa, Sqrt[r . r]];

(* ---- self-verify: eta-independence (kappa real), damped-direct agreement ---- *)
SeedRandom[20260623];
r3Pts = Table[Module[{u = RandomReal[{-1, 1}], ph = RandomReal[{0, 2 Pi}], st},
    st = Sqrt[1 - u^2]; 0.5 {st Cos[ph], st Sin[ph], u}], {6}];   (* |r|=0.5, z != 0 *)
kR = 1.5; kD = 1.5 + 0.25 I; e1 = 0.7; e2 = 1.15; Rc = 6; Gc = 6; Lbig = 40;
etaIndep = Max[Table[Abs[ewTot3[kR, r, e1, Rc, Gc] - ewTot3[kR, r, e2, Rc, Gc]], {r, r3Pts}]];
agree = Max[Table[Abs[ewTot3[kD, r, e1, Rc, Gc] - ewDirect3[kD, r, Lbig]], {r, r3Pts}]];
Print["==== Phase 3b cycle 1 :: general-z scalar Ewald ===="];
Print["  [1] eta-independence (kappa real, z!=0) = ", ScientificForm[etaIndep, 3],
   " -> ", If[etaIndep < 1.*^-8, "PASS", "FAIL"]];
Print["  [2] Ewald vs damped direct 3D sum = ", ScientificForm[agree, 3],
   " -> ", If[agree < 1.*^-6, "PASS", "FAIL"]];

(* ============================================================================ *)
(* Task 2: multipole projection -> undamped D[q,s]                               *)
(* ============================================================================ *)

Needs["NumericalDifferentialEquationAnalysis`"];
glN = GaussianQuadratureWeights[16, -1, 1]; nPhi = 32; rho0 = 0.5;
sphPts = Flatten[Table[Module[{u = glN[[i, 1]], ph = 2 Pi (j - 1)/nPhi, st = Sqrt[1 - glN[[i, 1]]^2]},
     {rho0 {st Cos[ph], st Sin[ph], u}, glN[[i, 2]] (2 Pi/nPhi)}], {i, Length[glN]}, {j, nPhi}], 1];
Yf[q_, s_, d_] := SphericalHarmonicY[q, s, ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]];
(* G(r) = i kappa Sum D̄[q,s] j_q(kappa r) Y_q^s ; project: D̄[q,s] = (1/(i kappa j_q)) ∮ G conj(Y_q^s);
   then D_struct[q,s] = (-1)^s D̄[q,-s]. *)
Dproj[fieldFn_, q_, s_, kappa_] := Module[{integ},
  integ = Total[Map[#[[2]] fieldFn[#[[1]]] Conjugate[Yf[q, -s, #[[1]]]] &, sphPts]];
  (-1)^s integ/(I kappa sj[q, kappa rho0])];

DstructUndamped[q_, s_, eta_] := Dproj[Function[r, ewTot3[1.5, r, eta, 6, 6]], q, s, 1.5];
DstructDirect[q_, s_, kappa_, Lr_] := Module[{ij, iv, jv, rn, ph, bl},
   ij = Flatten[Table[If[i == 0 && j == 0, Nothing, {i, j}], {i, -Lr, Lr}, {j, -Lr, Lr}], 1];
   iv = ij[[All, 1]]; jv = ij[[All, 2]]; rn = aL Sqrt[iv^2 + jv^2]; ph = ArcTan[iv, jv];
   bl = Exp[I aL (kx iv + ky jv)];
   Total[sh[q, kappa rn] SphericalHarmonicY[q, s, Pi/2, ph] bl]];
Nq = 6;
(* method gate: at DAMPED kappa, projected == direct structure constant *)
methResid = Max[Table[Abs[Dproj[Function[r, ewDirect3[1.5 + 0.25 I, r, 40]], q, s, 1.5 + 0.25 I]
     - DstructDirect[q, s, 1.5 + 0.25 I, 18]], {q, 0, Nq}, {s, -q, q}]];
(* undamped eta-independence *)
undEta = Max[Table[Abs[DstructUndamped[q, s, 0.7] - DstructUndamped[q, s, 1.15]], {q, 0, Nq}, {s, -q, q}]];
Print["  [3] projection method (damped: projected == direct D) = ", ScientificForm[methResid, 3],
   " -> ", If[methResid < 1.*^-4, "PASS", "FAIL"]];
Print["  [4] undamped D[q,s] eta-independence = ", ScientificForm[undEta, 3],
   " -> ", If[undEta < 1.*^-6, "PASS", "FAIL"]];

(* ============================================================================ *)
(* Task 3: G0 from undamped D[q,s] + reciprocity; dump JSON                      *)
(* ============================================================================ *)

gaunt[l1_, m1_, l2_, m2_, l3_, m3_] :=
  If[m1 + m2 + m3 != 0 || Abs[m1] > l1 || Abs[m2] > l2 || Abs[m3] > l3, 0,
   Sqrt[(2 l1 + 1) (2 l2 + 1) (2 l3 + 1)/(4 Pi)]
     ThreeJSymbol[{l1, 0}, {l2, 0}, {l3, 0}] ThreeJSymbol[{l1, m1}, {l2, m2}, {l3, m3}]];
DU[q_, s_] := DU[q, s] = DstructUndamped[q, s, 0.7];   (* memoise the undamped structure constants *)
G0[n_, m_, nu_, mu_] := 4 Pi (-1)^m Sum[
   I^(nu + q - n) (-1)^q DU[q, m - mu] gaunt[n, m, nu, -mu, q, mu - m], {q, Abs[n - nu], n + nu}];
recipPairs = {{1, 0, 2, 1}, {2, -1, 3, 2}, {0, 0, 3, 0}, {2, 2, 4, -2}, {1, 1, 3, -1}};
recipResid = Max[Table[Abs[G0[p[[1]], p[[2]], p[[3]], p[[4]]]
   - (-1)^(p[[1]] + p[[3]] + p[[2]] + p[[4]]) G0[p[[3]], -p[[4]], p[[1]], -p[[2]]]], {p, recipPairs}]];
Print["  [5] undamped G0 reciprocity = ", ScientificForm[recipResid, 3],
   " -> ", If[recipResid < 1.*^-6, "PASS", "FAIL"]];

(* ---- dump reference JSON ---- *)
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneKambe_reference.json",
  <|"params" -> <|"aL" -> aL, "kx" -> kx, "ky" -> ky, "kappa" -> 1.5, "eta1" -> 0.7, "eta2" -> 1.15,
      "rho0" -> rho0, "Rc" -> 6, "Gc" -> 6, "Nq" -> Nq|>,
    "Dstruct" -> Flatten[Table[<|"q" -> q, "s" -> s, "val" -> reim[DU[q, s]]|>, {q, 0, Nq}, {s, -q, q}], 1],
    "recip_resid" -> N[recipResid]|>];
Print["  wrote IntraPlaneKambe_reference.json"];
