#!/usr/bin/env wolframscript
(* ============================================================================
   AdaptiveOctree_F3Linear.wl -- the third-order factor of linear cells, the algebra
   of its derivation.

   LINEAR CELLS hold the field and the contrast in 1, x_1, x_2, x_3.  The scheme's third
   term is Int Pi f U_a : C : Pi U_b (one projection, on the incident side), with
   U = Gamma*(Pi f .).  To order h^4 the difference from the exact Int f u_a : C : u_b is

     D = Int [ f u_a s + f du_a u_b - f du_a s + g u_a u_b - g u_a s - g du_a u_b ],
     s = u_b'' + du_b^low,  g = f'',  X'' = X - Pi X,

   and every term reduces to two cell rules:
     B(S, T) = <S'' T''> = h^4 [ Sum_i S_ii T_ii / 45 + Sum_{i<j} S_ij T_ij / 9 ],
     the residual's own field du^high = Sum_i f_ii/2 G_i P_i + Sum_{i<j} f_ij M_ij Q_ij in its
     pairings with the quadratic functions P_i = x_i^2 - h^2/3, Q_ij = x_i x_j.
   The bilinear contractions are written as products of scalars (one component).

   CHECKS
   (a) the residual of a quadratic is 1/2 Sum f_ii P_i + Sum_{i<j} f_ij Q_ij, and P_i, Q_ij are
       orthogonal to 1 and x_k and to each other, with <P_i^2> = 4 h^4/45, <Q_ij^2> = h^4/9;
   (b) B: for quadratic S and T, <S'' T''> = h^4 [Sum S_ii T_ii/45 + Sum_{i<j} S_ij T_ij/9];
   (c) the assembly of D into the closed form of scripts/measure_f3_closed_form.py:
         B(f C u_a, u_b) + B(f, w_ba) - H + B(f, w_ab) + B(f, u_a C u_b) - f K - u_a C B(f, u_b) - K2 C u_b;
   (d) in a layer: (h^4/45) kappa (6 f'^2 f'' + 3 f f''^2), which is 3 Int f^2 g - 3 Int f g^2, the
       layer's linear-cell e_3;
   (e) the second term, 2 B(f, a u_b) - a K2(f; b), averaged for a radial profile, is the F1 of
       AdaptiveOctree_T2Factor.wl: <W> = 8/225, and the K2 term Sum m(e_i)/225 + Sum M_ij/135.
   ============================================================================ *)

oks = {};
check[name_, ok_] := (AppendTo[oks, TrueQ[ok]]; Print[If[TrueQ[ok], "PASS  ", "FAIL  "], name]);
X = {x1, x2, x3};
cube[e_] := Integrate[e, {x1, -h, h}, {x2, -h, h}, {x3, -h, h}];
vol = (2 h)^3;
mean[e_] := cube[e]/vol;
basis = {1, x1, x2, x3};
proj[e_] := Sum[cube[e b] b/cube[b^2], {b, basis}];
pp = Table[X[[i]]^2 - h^2/3, {i, 3}];
qq[i_, j_] := X[[i]] X[[j]];
planes = {{1, 2}, {1, 3}, {2, 3}};

(* --- (a) the residual and its basis ---------------------------------------------- *)
quad[name_] := Sum[name[i, j] X[[i]] X[[j]]/If[i == j, 2, 1], {i, 3}, {j, i, 3}];  (* Hessian entries name[i, j] *)
lin[name_] := name[0] + Sum[name[i] X[[i]], {i, 3}];
fq = lin[f0] + quad[fH];
resid = Expand[fq - proj[fq]];
check["(a) the residual of a quadratic is 1/2 Sum f_ii P_i + Sum_{i<j} f_ij Q_ij",
  Simplify[resid - (Sum[fH[i, i] pp[[i]]/2, {i, 3}] + Sum[fH @@ pl qq @@ pl, {pl, planes}])] === 0];
fns = Join[pp, qq @@@ planes];
check["(a) P_i and Q_ij are orthogonal to 1, x_k and to each other, <P^2> = 4h^4/45, <Q^2> = h^4/9",
  Union[Flatten[Table[Simplify[cube[a b]], {a, fns}, {b, basis}]]] === {0} &&
   Simplify[Table[mean[a b], {a, fns}, {b, fns}] -
      DiagonalMatrix[{4 h^4/45, 4 h^4/45, 4 h^4/45, h^4/9, h^4/9, h^4/9}]] === ConstantArray[0, {6, 6}]];

(* --- (b) the cell rule B -------------------------------------------------------------- *)
sq = lin[s0] + quad[sH]; tq = lin[t0] + quad[tH];
bRule[sh_, th_] := h^4 (Sum[sh[i, i] th[i, i], {i, 3}]/45 + Sum[sh @@ pl th @@ pl, {pl, planes}]/9);
check["(b) <S'' T''> = h^4 [Sum S_ii T_ii/45 + Sum_{i<j} S_ij T_ij/9]",
  Simplify[mean[(sq - proj[sq]) (tq - proj[tq])] - bRule[sH, tH]] === 0];

(* --- (c) the assembly -------------------------------------------------------------------- *)
(* cell-scale parts at a point, scalar stand-ins: u_b'' and the residual's field of a *)
uB2 = Sum[ubH[i, i] pp[[i]]/2, {i, 3}] + Sum[ubH @@ pl qq @@ pl, {pl, planes}];
gC = Sum[fH[i, i] pp[[i]]/2, {i, 3}] + Sum[fH @@ pl qq @@ pl, {pl, planes}];
duA = Sum[fH[i, i] Ga[i] pp[[i]]/2, {i, 3}] + Sum[fH @@ pl Ma @@ pl qq @@ pl, {pl, planes}];
(* single-small terms, by the cell rule against smooth fields (their h^4 values as symbols where the
   moment lemma gives them: Bw = B(f, w_ab) + B(f, w_ba) - H, Bphi = B(f, u_a C u_b), BS = B(f C u_a, u_b)) *)
single = BS + Bw + Bphi;
(* double-small terms: cell averages of products of the cell-scale parts *)
double = -f CC mean[duA uB2] - ua CC mean[gC uB2] - ub CC mean[gC duA];
assembled = Expand[single + double];
kTerm = h^4 (Sum[fH[i, i] Ga[i] ubH[i, i], {i, 3}]/45 + Sum[fH @@ pl Ma @@ pl ubH @@ pl, {pl, planes}]/9);
bFub = h^4 (Sum[fH[i, i] ubH[i, i], {i, 3}]/45 + Sum[fH @@ pl ubH @@ pl, {pl, planes}]/9);
k2Term = h^4 (Sum[fH[i, i]^2 Ga[i], {i, 3}]/45 + Sum[(fH @@ pl)^2 Ma @@ pl, {pl, planes}]/9);
claim = Expand[BS + Bw + Bphi - f CC kTerm - ua CC bFub - ub CC k2Term];
check["(c) the terms assemble into the closed form (the term in du_a du_b is absent: Pi removes du_b^high)",
  Simplify[assembled - claim] === 0];

(* --- (d) the layer ------------------------------------------------------------------------- *)
(* f = f(z), local fields u_b = f Gb, u_a = f Ga, w_ab = w_ba = f^2 kap, kap = Ga C Gb; second
   derivatives along axis 3 only: f_33 = f2, (f C u_a)_33 = (f^2)'' C Ga, u_b,33 = f2 Gb *)
layerRules = {fH[1, 1] -> 0, fH[2, 2] -> 0, fH[1, 2] -> 0, fH[1, 3] -> 0, fH[2, 3] -> 0, fH[3, 3] -> f2,
   ubH[3, 3] -> f2 Gb3, Ga[3] -> Ga3, ua -> f Ga3, ub -> f Gb3};
fsq2 = 2 f1^2 + 2 f f2;  (* (f^2)'' with f' = f1, f'' = f2 *)
layerSingle = h^4/45 (f2 (fsq2 kap) + f2 (fsq2 kap) - fsq2 kap f2 + f2 fsq2 kap + f2 fsq2 kap);
(* BS = (f^2)'' f'' kap; B(f, w_ab) + B(f, w_ba) - H = f''(f^2 kap)'' + f''(f^2 kap)'' - (f^2)'' kap f''; Bphi = f'' (f^2 kap)'' *)
layerDouble = Expand[((-f CC kTerm - ua CC bFub - ub CC k2Term) /. layerRules) /. CC -> 1];
layerTotal = Expand[(layerDouble /. Ga3 Gb3 -> kap) + layerSingle];
check["(d) in a layer the integrand is (h^4/45) kappa (6 f'^2 f'' + 3 f f''^2): the layer's e_3",
  Simplify[layerTotal - h^4/45 kap (6 f1^2 f2 + 3 f f2^2)] === 0];
(* the layer's e_3 for linear cells: Int f^3 - Int (Pi f)^3 = 3 Int f^2 g - 3 Int f g^2 + ..., with
   Int f^2 g = B(f^2, f) = h^4/45 (f^2)'' f'' and <g^2> = h^4/45 f''^2 *)
check["(d) and that is 3 B(f^2, f) - 3 f <g^2>, the direct expansion of Int f^3 - Int (Pi f)^3",
  Simplify[h^4/45 (6 f1^2 f2 + 3 f f2^2) - (3 h^4/45 fsq2 f2 - 3 f h^4/45 f2^2)] === 0];

(* --- (e) the second term, averaged for a radial profile ---------------------------------- *)
unit = {Sin[t] Cos[ph], Sin[t] Sin[ph], Cos[t]};
avg[e_] := Integrate[e Sin[t], {t, 0, Pi}, {ph, 0, 2 Pi}]/(4 Pi);
wOf[v_] := Sum[v[[k]]^4, {k, 3}]/45 + Sum[v[[k]]^2 v[[l]]^2, {k, 3}, {l, k + 1, 3}]/9;
check["(e) <W> over directions is 8/225", avg[wOf[unit]] === 8/225];
check["(e) the K2 term averages to Sum m(e_i)/225 + Sum M_ij/135 (weights <xi_i^4>/45 = 1/225, <xi_i^2 xi_j^2>/9 = 1/135)",
  avg[unit[[1]]^4]/45 === 1/225 && avg[unit[[1]]^2 unit[[2]]^2]/9 === 1/135];

Print[Count[oks, True], "/", Length[oks], " checks passed"];
