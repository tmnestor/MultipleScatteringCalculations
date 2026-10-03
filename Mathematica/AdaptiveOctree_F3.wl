#!/usr/bin/env wolframscript
(* ============================================================================
   AdaptiveOctree_F3.wl -- the third-order factor of constant cells, the algebra
   of its derivation.

   THE CLAIM.  With u_b = Gamma*(f b), u_a = Gamma*(f a), w_ab = a:Gamma*(f C:u_b),
   G_ib = Gamma(e_i) b and constant cells of half-width h,

     T3 - T3_scheme = (h^2/3) Int [ grad f . grad(w_ab + w_ba)
        + f Sum_i ( d_i u_a:C:d_i u_b - d_i f (d_i u_a:C:G_ib + G_ia:C:d_i u_b) )
        + Sum_i d_i f ( u_a:C:d_i u_b + d_i u_a:C:u_b )
        - Sum_i (d_i f)^2 ( u_a:C:G_ib + G_ia:C:u_b ) ] dV

   to second order in h.  The bilinear contractions are written here as products of
   scalars (one component), which carries the algebra unchanged.

   CHECKS
   (a) the cell decomposition, exact for any fields:  Int_cell f A B =
       V <f><A><B> + <f> Int A'B' + <A> Int f'B' + <B> Int f'A' + Int f'A'B',  primes the
       departures from the cell means;
   (b) the cell averages of the sawtooths: <eta_i eta_j> = delta_ij h^2/3, and of the products
       of two linear residuals: <(F.eta)(G.eta)> = (h^2/3) F.G;
   (c) the assembly: with f' = grad f.eta, A' = Sum_i (d_i u_a - d_i f G_ia) eta_i (the
       smooth field's variation less the residual's field, a sawtooth on each axis carrying
       Gamma on that axis) and B' likewise, the terms of (a), together with the residual's
       own terms X1 (by the moment lemma, (h^2/3) grad f.grad(w_ab + w_ba)) and
       X2 = -f <delta u_a delta u_b>, give the integrand of THE CLAIM: the (d_i f)^2 G G terms
       cancel;
   (d) in a layer every field is local, u_b = f G_b, w_ab = f^2 kappa, kappa = G_a C G_b, and the
       integrand is 3 f f'^2 kappa: the layer's e_3 = 3 Int f g^2 / Int f^3;
   (e) the same expansion of the second-order term, Int f a u_b, gives
       2 grad f.grad(a u_b) - Sum_i (d_i f)^2 a G_ib, whose ratio for a radial profile is the
       2 - m_ax/m_bar of AdaptiveOctree_T2Factor.wl.
   ============================================================================ *)

oks = {};
check[name_, ok_] := (AppendTo[oks, TrueQ[ok]]; Print[If[TrueQ[ok], "PASS  ", "FAIL  "], name]);
X = {x1, x2, x3};
cube[e_] := Integrate[e, {x1, -h, h}, {x2, -h, h}, {x3, -h, h}];
vol = (2 h)^3;
mean[e_] := cube[e]/vol;

(* --- (a) the cell decomposition ----------------------------------------------- *)
poly[c_] := c[0] + Sum[c[i] X[[i]], {i, 3}] + Sum[c[i, j] X[[i]] X[[j]], {i, 3}, {j, i, 3}];
fP = poly[fc]; aP = poly[ac]; bP = poly[bc];
prime[e_] := e - mean[e];
lhs = cube[fP aP bP];
rhs = vol mean[fP] mean[aP] mean[bP] + mean[fP] cube[prime[aP] prime[bP]] +
   mean[aP] cube[prime[fP] prime[bP]] + mean[bP] cube[prime[fP] prime[aP]] +
   cube[prime[fP] prime[aP] prime[bP]];
check["(a) Int f A B = V<f><A><B> + <f>Int A'B' + <A>Int f'B' + <B>Int f'A' + Int f'A'B'",
  Simplify[lhs - rhs] === 0];

(* --- (b) cell averages of sawtooths -------------------------------------------- *)
check["(b) <eta_i eta_j> = delta_ij h^2/3",
  Simplify[Table[mean[X[[i]] X[[j]]], {i, 3}, {j, 3}] - h^2/3 IdentityMatrix[3]] === ConstantArray[0, {3, 3}]];
fv = {F1, F2, F3}; gv = {G1, G2, G3};
check["(b) <(F.eta)(G.eta)> = (h^2/3) F.G", Simplify[mean[(fv . X) (gv . X)] - h^2/3 fv . gv] === 0];

(* --- (c) the assembly ------------------------------------------------------------ *)
(* point values: f, u_a, u_b; gradients df_i, dua_i, dub_i; axis operators Ga_i, Gb_i; C a scalar *)
df = {df1, df2, df3}; dua = {dua1, dua2, dua3}; dub = {dub1, dub2, dub3};
Ga = {Ga1, Ga2, Ga3}; Gb = {Gb1, Gb2, Gb3};
fPrime = df . X;
aPrime = Sum[(dua[[i]] - df[[i]] Ga[[i]]) X[[i]], {i, 3}];
bPrime = Sum[(dub[[i]] - df[[i]] Gb[[i]]) X[[i]], {i, 3}];
(* the residual's own fields: delta u = Sum_i d_i f G_i eta_i (high part), on each axis *)
dUa = Sum[df[[i]] Ga[[i]] X[[i]], {i, 3}]; dUb = Sum[df[[i]] Gb[[i]] X[[i]], {i, 3}];
termsA = f mean[aPrime CC bPrime] + ua mean[fPrime CC bPrime] + ub mean[fPrime CC aPrime];
x2term = -f mean[dUa CC dUb];
x1term = h^2/3 gradW;  (* (h^2/3) grad f . grad (w_ab + w_ba), the moment lemma; gradW a symbol *)
assembled = Expand[termsA + x2term + x1term];
claim = Expand[h^2/3 (gradW +
      f Sum[dua[[i]] CC dub[[i]] - df[[i]] (dua[[i]] CC Gb[[i]] + Ga[[i]] CC dub[[i]]), {i, 3}] +
      Sum[df[[i]] (ua CC dub[[i]] + dua[[i]] CC ub), {i, 3}] -
      Sum[df[[i]]^2 (ua CC Gb[[i]] + Ga[[i]] CC ub), {i, 3}])];
check["(c) the terms assemble into the integrand of the claim", Simplify[assembled - claim] === 0];
check["(c) the (d_i f)^2 G G terms of X2 and of <f> Int A'B' cancel",
  Coefficient[Expand[termsA + x2term], Ga1 Gb1 df1^2] === 0];

(* --- (d) the layer ----------------------------------------------------------------- *)
(* one axis (3), local fields: u_b = f Gb3, u_a = f Ga3, d u = f' G, w_ab = w_ba = f^2 kap *)
layer = claim /. {df1 -> 0, df2 -> 0, dua1 -> 0, dua2 -> 0, dub1 -> 0, dub2 -> 0} /.
    {ua -> f Ga3, ub -> f Gb3, dua3 -> fp Ga3, dub3 -> fp Gb3, df3 -> fp,
     gradW -> fp (2 kap) (2 f fp)} /. {kap -> Ga3 CC Gb3};  (* f' d/dz (w_ab + w_ba), w = f^2 kap *)
check["(d) in a layer the integrand is (h^2/3) 3 f f'^2 G_a C G_b: the layer's e_3",
  Simplify[layer - h^2/3 3 f fp^2 Ga3 CC Gb3] === 0];

(* --- (e) the second-order term ------------------------------------------------------ *)
(* Int f a u_b: f -> <f> + f', u_b -> <u_b> + u_b' with u_b' = Sum (d_i u_b - d_i f G_ib) eta_i; the
   residual's own term is (h^2/3) grad f . grad(a u_b) by the moment lemma *)
t2terms = Expand[mean[fPrime aa (dub . X - Sum[df[[i]] Gb[[i]] X[[i]], {i, 3}])] + h^2/3 gradAU];
t2claim = Expand[h^2/3 (gradAU + aa df . dub - aa Sum[df[[i]]^2 Gb[[i]], {i, 3}])];
check["(e) T2 - T2_scheme = (h^2/3)[grad f.grad(a u_b) + a grad f.grad u_b - Sum_i (d_i f)^2 a G_ib]",
  Simplify[t2terms - t2claim] === 0];
(* grad f . grad(a u_b) and a grad f . grad u_b are the same smooth term (a is constant), so the
   bracket is 2 grad f.grad(a u_b) - Sum_i (d_i f)^2 a G_ib; for a radial profile the first is
   2 m_bar Int|grad f|^2 (isotropic spectrum) and the second m_ax Int|grad f|^2, F = 2 - m_ax/m_bar *)

Print[Count[oks, True], "/", Length[oks], " checks passed"];
