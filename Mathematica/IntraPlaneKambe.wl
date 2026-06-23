(* ============================================================================ *)
(* IntraPlaneKambe.wl — Phase 3b cycle 1: undamped multipole structure constants *)
(* D[q,s] via multipole projection of the undamped scalar Ewald field. Standard  *)
(* Kambe/layer-KKR OBJECT, built by projection of the validated TB2 scalar Ewald *)
(* (NOT per-q analytic forms). Time e^{+i w t}, outgoing h_q^(1); lattice in x-y, *)
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

(* damped direct 3D lattice GF (ground truth for the projection-method gate) *)
ewDirect3[kappa_, r_, Lbig_] := Total[Flatten[Table[
     With[{d = Sqrt[(r[[1]] - aL i)^2 + (r[[2]] - aL j)^2 + r[[3]]^2]},
      Exp[I kappa d]/(4 Pi d) Exp[I aL (kx i + ky j)]], {i, -Lbig, Lbig}, {j, -Lbig, Lbig}], 1]];

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
ewTot3[kappa_, r_, eta_, Rc_, Gc_] := ewReal3[kappa, r, eta, Rc] + ewRecip3[kappa, r, eta, Gc];

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
