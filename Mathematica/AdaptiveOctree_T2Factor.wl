#!/usr/bin/env wolframscript
(* ============================================================================
   AdaptiveOctree_T2Factor.wl -- the second-order error of constant cells in three
   dimensions, and the factor by which it departs from the projection error.

   THE SETTING.  A stiffness contrast f(x) DC in an isotropic background, struck by
   a uniform strain e_in, observed through a uniform strain e_out (the long-wave
   limit).  The second-order term is the quadratic form

     T2 = <f, K f>,   K with Fourier multiplier  m(xi) = a : Gamma(xi) : b,
     a = DC : e_out,  b = DC : e_in,

   Gamma the static strain Green operator.  Cells of constant field and contrast
   (p = r = 0) compute <Pi f, K Pi f>, Pi the cell mean, so with g = f - Pi f

     (T2 - T2_scheme) / (T2 E) = (2 <g, K f> - <g, K g>) / (<f, K f> E),
     E = |g|^2 / |f|^2 ,

   an identity (Python: scripts/measure_t2_law_3d.py and
   scripts/measure_t2_quadratic_forms.py agree with it to under 1%).

   THE CLAIM.  As the cells shrink, for a body whose spectrum is isotropic (a radial
   profile),
                 F  ->  2 - m_ax / m_bar ,
   m_bar the average of m over directions, m_ax its mean over the three cube axes.
   In one dimension (a layer) F = 1, the law of the layer.

   CHECKS (the derivation, step by step)
   (a) the strain Green operator from the Navier equation: N(xi)^-1 in closed form;
   (b) its contraction m equals the formula used in Python;
   (c) the angular average m_bar in closed form, by integration over the sphere;
   (d) the low-frequency moments of the cell residual: for f linear in a cell,
       Int g x_i / V = G_i h^2/3 and Int g^2 / V = |G|^2 h^2/3 (the same constant),
       and the second moments vanish, so 2 <g, K f> -> 2 m_bar |g|^2;
   (e) the sawtooth carries all its energy on the cube axis: Parseval for its
       Fourier series (no mean, harmonics at pi m / h along that axis only) and the
       orthogonality of the sawtooths of different axes, so <g, K g> -> m_ax |g|^2;
   (f) the assembly: for a radial profile Int (d_i f)^2 = |grad f|^2 / 3 for each i,
       giving F = 2 - m_ax / m_bar; for a layer (f of one coordinate) F = 1;
   (g) closed forms for an incident P wave along axis 1 and an outgoing P or SV wave
       at angle theta in the plane of axes 1 and 2, and their values against the
       numbers printed by scripts/measure_t2_law_3d.py.

   LINEAR CELLS (p = r = 1: the field and the contrast in 1, x_1, x_2, x_3)
   (h) the residual of a quadratic f is  g = 1/2 Sum_i H_ii (x_i^2 - h^2/3) + Sum_{i<j} H_ij x_i x_j,
       orthogonal to 1 and x_i;
   (i) its energy is Sum_i H_ii^2 h^4/45 + Sum_{i<j} H_ij^2 h^4/9 per unit volume; its moments of
       degree 0, 1 and 3 vanish and those of degree 2 are 2 h^4/45 H_kk and h^4/9 H_kl; so its
       low-frequency transform is  W(xi) |xi|^4-weighted:  g^ -> h^4 W(xi) f^ with
       W = Sum_k xi_k^4 / 45 + Sum_{k<l} xi_k^2 xi_l^2 / 9  (cubic, not isotropic), and <g, f> = |g|^2;
   (j) the parabola x^2 - h^2/3 has no mean and its spectrum lies on its axis; the product x_i x_j of two
       sawtooths lies on the lattice of the (i, j) plane with weights 1/(m^2 n^2), m, n != 0, over which
       the averages of u^2 v^2 and u^4 (u, v the direction cosines) are 6 G/pi^2 - 2/5 and 9/10 - 6 G/pi^2,
       G Catalan's constant, from  Sum_{m,n != 0} 1/(m^2 + n^2)^2 = 4 zeta(2) beta(2) - 4 zeta(4);
   (k) the assembly, for a radial profile:
         F1 = [2 <W m> - Sum_i m(e_i)/225 - Sum_{i<j} M_ij/135] / (m_bar <W>),   <W> = 8/225,
       M_ij the average of m over the (i, j) lattice; a constant m and a layer both give F1 = 1;
   (l) closed forms for P and SV, and their values against scripts/measure_t2_law_3d.py's prediction.
   EXPORT: Mathematica/AdaptiveOctree_t2_factor.json, 30-digit values of F (p = 0) and F1 (p = 1).
   ============================================================================ *)

oks = {};
check[name_, ok_] := (AppendTo[oks, TrueQ[ok]]; Print[If[TrueQ[ok], "PASS  ", "FAIL  "], name]);

(* --- (a) the static strain Green operator ------------------------------------ *)
xi = {x1, x2, x3};
acoustic = mu (xi . xi) IdentityMatrix[3] + (lam + mu) Outer[Times, xi, xi];
ninv = Simplify[Inverse[acoustic]];
closed = (IdentityMatrix[3] - (lam + mu)/(lam + 2 mu) Outer[Times, xi, xi]/(xi . xi))/(mu (xi . xi));
check["(a) N(xi)^-1 = (I - (lam+mu)/(lam+2mu) xi xi/|xi|^2) / (mu |xi|^2)",
  Simplify[ninv - closed] === ConstantArray[0, {3, 3}]];

(* the strain of the displacement N^-1 (-i xi . sigma), sigma a stress source:
   eps_ij = sym_ij (xi_i (N^-1)_jk xi_l) sigma_kl, symmetrised in (ij) and in (kl) *)
gamma = Table[
   (xi[[i]] ninv[[j, k]] xi[[l]] + xi[[j]] ninv[[i, k]] xi[[l]] +
       xi[[i]] ninv[[j, l]] xi[[k]] + xi[[j]] ninv[[i, l]] xi[[k]])/4,
   {i, 3}, {j, 3}, {k, 3}, {l, 3}];
check["(a) Gamma is homogeneous of degree zero",
  Simplify[(gamma /. Thread[xi -> 2 xi]) - gamma] === ConstantArray[0, {3, 3, 3, 3}]];

(* --- (b) the contraction m(xi) = a : Gamma : b ------------------------------- *)
symm[name_] := Table[name[Min[i, j], Max[i, j]], {i, 3}, {j, 3}];
aS = symm[aa]; bS = symm[bb];
mOf[a_, b_, v_] := Module[{g = gamma /. Thread[xi -> v]},
   Sum[a[[i, j]] g[[i, j, k, l]] b[[k, l]], {i, 3}, {j, 3}, {k, 3}, {l, 3}]];
mPython[a_, b_, v_] := Module[{u = v/Sqrt[v . v], cb = (lam + mu)/(mu (lam + 2 mu))},
   u . ((a . b + b . a)/2) . u/mu - cb (u . a . u) (u . b . u)];
check["(b) a : Gamma(xi) : b = xi.sym(ab).xi / mu - (lam+mu)/(mu(lam+2mu)) (xi.a.xi)(xi.b.xi), |xi| = 1",
  Simplify[mOf[aS, bS, {Sin[t] Cos[ph], Sin[t] Sin[ph], Cos[t]}] -
     mPython[aS, bS, {Sin[t] Cos[ph], Sin[t] Sin[ph], Cos[t]}]] === 0];

(* --- (c) the angular average ------------------------------------------------- *)
unit = {Sin[t] Cos[ph], Sin[t] Sin[ph], Cos[t]};
mBarIntegral = Simplify[
   Integrate[Expand[mPython[aS, bS, unit]] Sin[t], {t, 0, Pi}, {ph, 0, 2 Pi}]/(4 Pi)];
mBarClosed[a_, b_] := Tr[(a . b + b . a)/2]/(3 mu) -
   (lam + mu)/(mu (lam + 2 mu)) (Tr[a] Tr[b] + 2 Total[Flatten[a b]])/15;
check["(c) m_bar = tr(sym ab)/(3 mu) - (lam+mu)/(mu(lam+2mu)) (tr a tr b + 2 a:b)/15",
  Simplify[mBarIntegral - mBarClosed[aS, bS]] === 0];
mAx[a_, b_] := Mean[Table[mPython[a, b, UnitVector[3, i]], {i, 3}]];

(* --- (d) the low-frequency moments of the cell residual ---------------------- *)
gVec = {G1, G2, G3}; X = {y1, y2, y3};
cube[expr_] := Integrate[expr, {y1, -h, h}, {y2, -h, h}, {y3, -h, h}];
vol = (2 h)^3;
gRes = gVec . X;  (* f linear in the cell, less its mean *)
firstMoments = Table[Simplify[cube[gRes X[[i]]]/vol], {i, 3}];
energy = Simplify[cube[gRes^2]/vol];
secondMoments = Table[Simplify[cube[gRes X[[i]] X[[j]]]], {i, 3}, {j, 3}];
check["(d) Int g x_i / V = G_i h^2/3 and Int g^2 / V = |G|^2 h^2/3: one constant",
  Simplify[firstMoments - gVec h^2/3] === {0, 0, 0} && Simplify[energy - (gVec . gVec) h^2/3] === 0];
check["(d) the second moments of the residual vanish (its tail is -i xi . G h^2/3 V + O(xi^3))",
  secondMoments === ConstantArray[0, {3, 3}]];
(* so the residual's low-frequency transform is (h^2/3) |xi|^2 f^ and
   <g, K f> -> (h^2/3) Int |xi|^2 |f^|^2 m(xi) = m_bar |g|^2 for an isotropic spectrum *)

(* --- (e) the sawtooth's spectrum --------------------------------------------- *)
coef[m_] = Simplify[Integrate[s Exp[-I Pi m s/h], {s, -h, h}]/(2 h), Element[m, Integers] && m != 0];
mean0 = Integrate[s, {s, -h, h}]/(2 h);
parseval = Simplify[2 Sum[Abs[coef[m]]^2 /. Abs[z_]^2 :> z Conjugate[z], {m, 1, Infinity}] - Integrate[s^2, {s, -h, h}]/(2 h),
   h > 0];
check["(e) the sawtooth has no mean, and its harmonics at pi m/h carry all its energy (Parseval)",
  mean0 === 0 && parseval === 0];
check["(e) sawtooths of different axes are orthogonal over the cell",
  Simplify[cube[y1 y2]] === 0 && Simplify[cube[y1 y3]] === 0 && Simplify[cube[y2 y3]] === 0];
(* the residual G . x is a sum of sawtooths, axis i with amplitude G_i: its energy on the
   axis directions is G_i^2 h^2/3 each, so <g, K g> -> Sum_i m(e_i) Int (d_i f)^2 h^2/3 *)

(* --- (f) assembly ------------------------------------------------------------- *)
(* F = [2 m_bar Sum_i w_i - Sum_i m(e_i) w_i] / m_bar,  w_i = Int (d_i f)^2 / Int |grad f|^2 *)
fFactor[a_, b_, w_] := (2 mBarClosed[a, b] Total[w] - Sum[mPython[a, b, UnitVector[3, i]] w[[i]], {i, 3}])/
   mBarClosed[a, b];
radialWeights = Module[{wx},
   (* for f(r): (d_1 f)^2 = f'(r)^2 x1^2/r^2, and the average of x1^2/r^2 over the sphere is 1/3 *)
   wx = Integrate[(unit[[1]])^2 Sin[t], {t, 0, Pi}, {ph, 0, 2 Pi}]/(4 Pi);
   {wx, wx, wx}];
check["(f) for a radial profile each axis carries a third of |grad f|^2", radialWeights === {1/3, 1/3, 1/3}];
check["(f) so F = 2 - m_ax/m_bar",
  Simplify[fFactor[aS, bS, radialWeights] - (2 - mAx[aS, bS]/mBarClosed[aS, bS])] === 0];
(* a layer: f = f(x3); its spectrum lies on the x3 axis, so <f, K f> = m(e3) |f|^2
   and the cross term too; the factor is (2 m(e3) - m(e3)) / m(e3) = 1 *)
layerF = (2 mPython[aS, bS, UnitVector[3, 3]] - mPython[aS, bS, UnitVector[3, 3]])/mPython[aS, bS, UnitVector[3, 3]];
check["(f) in a layer the factor is one: the law of the layer", Simplify[layerF] === 1];

(* --- (g) closed forms and the numbers of the measurement ---------------------- *)
dC[e_] := dlam Tr[e] IdentityMatrix[3] + 2 dmu e;
eIn = Outer[Times, {1, 0, 0}, {1, 0, 0}];
nv = {Cos[th], Sin[th], 0}; tv = {-Sin[th], Cos[th], 0};
fPP = FullSimplify[2 - mAx[dC[Outer[Times, nv, nv]], dC[eIn]]/mBarClosed[dC[Outer[Times, nv, nv]], dC[eIn]]];
fPS = FullSimplify[2 - mAx[dC[(Outer[Times, nv, tv] + Outer[Times, tv, nv])/2], dC[eIn]]/
     mBarClosed[dC[(Outer[Times, nv, tv] + Outer[Times, tv, nv])/2], dC[eIn]]];
Print["F_PP(theta) = ", fPP];
Print["F_PS(theta) = ", fPS];
check["(g) F_PS does not depend on the angle (where it is defined)",
  FullSimplify[D[fPS, th], Sin[2 th] != 0] === 0];

params = {lam -> 2500 5000^2 - 2 2500 3000^2, mu -> 2500 3000^2, dlam -> 2 10^9, dmu -> 10^9};
thetas = Subdivide[1/5, Pi - 1/5, 8];
printedP = {1.1009, 1.0693, 1.0131, 0.9518, 0.9234, 0.9518, 1.0131, 1.0693, 1.1009};
valsP = Table[N[fPP /. params /. th -> x, 30], {x, thetas}];
valS = N[fPS /. params /. th -> 1/5, 30];
check["(g) F_PP at the nine angles equals the measurement script's numbers to 5e-5",
  Max[Abs[valsP - printedP]] < 5 10^-5];
check["(g) F_PS equals the script's 1.5161 to 5e-5", Abs[valS - 1.5161] < 5 10^-5];

(* ================== LINEAR CELLS (p = r = 1) ================== *)
(* --- (h) the residual of a linear cell --------------------------------------- *)
hS = symm[HH];
fQuad = X . hS . X/2;
basis1 = {1, y1, y2, y3};
projQ = Sum[cube[fQuad b] b/cube[b^2], {b, basis1}];
gRes1 = Expand[fQuad - projQ];
gExpected = Sum[hS[[i, i]] (X[[i]]^2 - h^2/3)/2, {i, 3}] + Sum[hS[[i, j]] X[[i]] X[[j]], {i, 3}, {j, i + 1, 3}];
check["(h) the residual is 1/2 Sum H_ii (x_i^2 - h^2/3) + Sum_{i<j} H_ij x_i x_j", Simplify[gRes1 - gExpected] === 0];
check["(h) it is orthogonal to 1 and x_i", Simplify[cube[gRes1 #] & /@ basis1] === {0, 0, 0, 0}];

(* --- (i) its energy and moments --------------------------------------------- *)
energy1 = Simplify[cube[gRes1^2]/vol];
check["(i) energy Sum H_ii^2 h^4/45 + Sum_{i<j} H_ij^2 h^4/9",
  Simplify[energy1 - (Sum[hS[[i, i]]^2, {i, 3}] h^4/45 + Sum[hS[[i, j]]^2, {i, 3}, {j, i + 1, 3}] h^4/9)] === 0];
m2 = Table[Simplify[cube[gRes1 X[[k]] X[[l]]]/vol], {k, 3}, {l, 3}];
check["(i) second moments 2 h^4/45 H_kk on the diagonal and h^4/9 H_kl off it",
  Simplify[m2 - Table[If[k == l, 2 h^4/45 hS[[k, k]], h^4/9 hS[[k, l]]], {k, 3}, {l, 3}]] === ConstantArray[0, {3, 3}]];
check["(i) moments of degree 0, 1 and 3 vanish",
  Union[Simplify[Join[{cube[gRes1]}, cube[gRes1 #] & /@ X,
      Flatten[Table[cube[gRes1 X[[i]] X[[j]] X[[k]]], {i, 3}, {j, 3}, {k, 3}]]]]] === {0}];
(* with H -> -xi xi f^: g^ = -1/2 Sum xi_k xi_l m2_kl = h^4 W(xi) f^ *)
wOf[v_] := Sum[v[[k]]^4, {k, 3}]/45 + Sum[v[[k]]^2 v[[l]]^2, {k, 3}, {l, k + 1, 3}]/9;
gTail = Expand[-(1/2) Sum[v[k] v[l] (m2[[k, l]] /. HH[a_, b_] :> -v[a] v[b]), {k, 3}, {l, 3}]/h^4];
check["(i) the low-frequency transform of g is h^4 W(xi) f^", Simplify[gTail - wOf[{v[1], v[2], v[3]}]] === 0];
check["(i) and <g, f> = |g|^2: the energy in Fourier is h^4 W |xi|^4-weighted too",
  Simplify[(energy1 /. HH[a_, b_] :> v[a] v[b]) - h^4 wOf[{v[1], v[2], v[3]}]] === 0];
wBar = Simplify[Integrate[wOf[unit] Sin[t], {t, 0, Pi}, {ph, 0, 2 Pi}]/(4 Pi)];
check["(i) <W> = 8/225 over directions", wBar === 8/225];

(* --- (j) the spectra of the residual's pieces -------------------------------- *)
parab[m_] = Simplify[Integrate[(s^2 - h^2/3) Exp[-I Pi m s/h], {s, -h, h}]/(2 h), Element[m, Integers] && m != 0];
parsevalParab = Simplify[2 Sum[parab[m]^2, {m, 1, Infinity}] - Integrate[(s^2 - h^2/3)^2, {s, -h, h}]/(2 h), h > 0];
check["(j) the parabola has no mean and its axis harmonics carry all its energy",
  Integrate[s^2 - h^2/3, {s, -h, h}] === 0 && parsevalParab === 0];
latticeQ = 4 Zeta[2] DirichletBeta[2] - 4 Zeta[4];
rowSum[m_] = Sum[1/(m^2 + n^2)^2, {n, -Infinity, Infinity}];  (* closed form in Coth, for m != 0 *)
numQ = Quiet[NSum[2 (rowSum[m] - 1/m^4), {m, 1, Infinity}, WorkingPrecision -> 40, NSumTerms -> 200], NIntegrate::ncvb];
check["(j) Sum_{m,n != 0} 1/(m^2+n^2)^2 = 4 zeta(2) beta(2) - 4 zeta(4) (to 25 digits)",
  Abs[numQ - N[latticeQ, 40]] < 10^-25];
u2v2 = Simplify[latticeQ/(2 Zeta[2])^2];  (* weights 1/(m^2 n^2) sum to (2 zeta(2))^2 *)
check["(j) <u^2 v^2> over the plane lattice = 6 G/pi^2 - 2/5", Simplify[u2v2 - (6 Catalan/Pi^2 - 2/5)] === 0];
u4 = 1/2 - u2v2;  (* u^4 + 2 u^2 v^2 + v^4 = 1, and u, v enter alike *)
check["(j) the product of two sawtooths has no energy on the axes or at the origin (its factors have no mean)",
  mean0 === 0];

(* --- (k) assembly ------------------------------------------------------------ *)
pairM[a_, b_, i_, j_] := ((a . b + b . a)[[i, i]] + (a . b + b . a)[[j, j]])/(4 mu) -
   (lam + mu)/(mu (lam + 2 mu)) (u4 (a[[i, i]] b[[i, i]] + a[[j, j]] b[[j, j]]) +
      u2v2 (a[[i, i]] b[[j, j]] + a[[j, j]] b[[i, i]] + 4 a[[i, j]] b[[i, j]]));
mUnit[a_, b_, u_] := u . ((a . b + b . a)/2) . u/mu - (lam + mu)/(mu (lam + 2 mu)) (u . a . u) (u . b . u);
latticeMoment[{0, 0}] = 1; latticeMoment[{2, 0}] = 1/2; latticeMoment[{0, 2}] = 1/2;
latticeMoment[{4, 0}] = u4; latticeMoment[{0, 4}] = u4; latticeMoment[{2, 2}] = u2v2;
latticeMoment[_] = 0;  (* odd powers cancel between m and -m *)
latAvg[poly_] := Total[(#2 latticeMoment[#1]) & @@@ CoefficientRules[Expand[poly], {uu, vv}]];
check["(k) M_12 is the lattice average of m over the (1, 2) plane, monomial by monomial",
  Simplify[pairM[aS, bS, 1, 2] - latAvg[mUnit[aS, bS, {uu, vv, 0}]]] === 0];
wmBar[a_, b_] := Integrate[Expand[wOf[unit] mPython[a, b, unit]] Sin[t], {t, 0, Pi}, {ph, 0, 2 Pi}]/(4 Pi);
f1Factor[a_, b_] := (2 wmBar[a, b] - Sum[mPython[a, b, UnitVector[3, i]], {i, 3}]/225 -
      Sum[pairM[a, b, i, j], {i, 3}, {j, i + 1, 3}]/135)/(mBarClosed[a, b] 8/225);
check["(k) a constant m gives F1 = 1",
  Simplify[(2 wBar - 3/225 - 3/135)/wBar] === 1];
layer1 = Module[{m3 = mPython[aS, bS, UnitVector[3, 3]]}, (2 m3 - m3)/m3];
check["(k) a layer gives F1 = 1 (only H_33: its tail and its parabola both lie on axis 3)", Simplify[layer1] === 1];

(* --- (l) closed forms and the prediction script ------------------------------ *)
f1PP = Simplify[f1Factor[dC[Outer[Times, nv, nv]], dC[eIn]]];
f1PS = Simplify[f1Factor[dC[(Outer[Times, nv, tv] + Outer[Times, tv, nv])/2], dC[eIn]]];
Print["F1_PP(theta) = ", f1PP];
Print["F1_PS(theta) = ", f1PS];
vals1P = Table[N[f1PP /. params /. th -> x, 30], {x, thetas}];
vals1S = Table[N[f1PS /. params /. th -> x, 30], {x, thetas}];
check["(l) F1_PP at 0.2, 0.885 and pi/2 equals the Python prediction 1.0659, 1.0086, 0.9500 to 5e-5",
  Max[Abs[vals1P[[{1, 3, 5}]] - {1.0659, 1.0086, 0.9500}]] < 5 10^-5];
check["(l) F1_PS at 0.2 equals the Python prediction 1.3369 to 5e-5", Abs[vals1S[[1]] - 1.3369] < 5 10^-5];
Print["F1_PP = ", ToString[N[#, 12]] & /@ vals1P];
Print["F1_PS = ", ToString[N[#, 12]] & /@ vals1S];

Print["F_PP = ", ToString[N[#, 12]] & /@ valsP];
Print["F_PS = ", ToString[N[valS, 12]]];

Export[FileNameJoin[{DirectoryName[$InputFileName], "AdaptiveOctree_t2_factor.json"}],
  <|"theta" -> (ToString[NumberForm[N[#, 30], 25]] & /@ thetas),
    "F_PP" -> (ToString[NumberForm[#, 25]] & /@ valsP), "F_PS" -> ToString[NumberForm[valS, 25]],
    "F1_PP" -> (ToString[NumberForm[#, 25]] & /@ vals1P), "F1_PS" -> (ToString[NumberForm[#, 25]] & /@ vals1S),
    "parameters" -> "alpha 5000, beta 3000, rho 2500, dlambda 2e9, dmu 1e9; incident P along axis 1"|>, "JSON"];
Print[Count[oks, True], "/", Length[oks], " checks passed"];
