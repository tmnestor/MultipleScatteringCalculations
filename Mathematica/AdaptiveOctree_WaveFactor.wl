#!/usr/bin/env wolframscript
(* ============================================================================
   AdaptiveOctree_WaveFactor.wl -- the wave term of a polynomial cell, symbolically.

   THE CELL.  A cube of half-width h, centred at the origin, whose field is held in
   the Legendre products  Q_a = P_a1(x/h) P_a2(y/h) P_a3(z/h)  of total degree
   |a| <= p, with a contrast that is uniform in the cell.

   THE CLAIM.  At first order in the contrast the cell holds the L2 projection of
   the incident wave Exp[I kin.x] and radiates towards kout through the moments of
   Exp[-I kout.x].  Its far field is therefore the exact Born far field times

     1 + E_p = Sum_{|a| <= p} Prod_i (2 a_i + 1) j_ai(kin_i h) j_ai(kout_i h)
               / Prod_i j_0((kin_i - kout_i) h) ,

   with j_n the spherical Bessel functions.

   CHECKS
   (a) Int_{-1}^{1} P_n(t) Exp[I q t] dt = 2 I^n j_n(q), n = 0..4, symbolically;
   (b) the factor built directly (project, multiply, integrate over the cube, with
       no Bessel function) equals the formula, p = 0, 1, 2, at 40 digits;
   (c) the leading term:  E_p = -h^(2p+2) Sum_{|a| = p+1} Prod_i (2 a_i + 1)
       (kin_i kout_i)^a_i / ((2 a_i + 1)!!)^2 + higher order;  for p = 0 this is
       -(kin.kout) d^2 / 12, d = 2 h;
   (d) in one dimension (kin = k z, kout = +-k z) the leading term is
       -(+-1)^(p+1) c_p (k d)^(2p+2) with c_p = (2p+3) / (4^(p+1) ((2p+3)!!)^2):
       the constant of the layer, c_0 = 1/12, c_1 = 1/720, c_2 = 1/100800;
   (e) the sum over all degrees is the exact Born term (the addition theorem):
       the factor tends to one as p grows.
   EXPORT: Mathematica/AdaptiveOctree_wave_factor.json, 30-digit values read by
           cubic_scattering/tests/test_graded_voxel_octree.py.
   ============================================================================ *)

oks = {};
check[name_, ok_] := (AppendTo[oks, TrueQ[ok]]; Print[If[TrueQ[ok], "PASS  ", "FAIL  "], name]);

indices[p_] := Select[Tuples[Range[0, p], 3], Total[#] <= p &];
factor[kin_, kout_, h_, p_] := (
   Sum[Product[(2 a[[i]] + 1) SphericalBesselJ[a[[i]], kin[[i]] h] SphericalBesselJ[a[[i]], kout[[i]] h], {i, 3}],
     {a, indices[p]}]/Product[SphericalBesselJ[0, (kin[[i]] - kout[[i]]) h], {i, 3}]);

(* (a) the Legendre moment of an exponential *)
check["(a) Int P_n(t) Exp[I q t] dt = 2 I^n j_n(q), n = 0..4",
  AllTrue[Range[0, 4],
   FullSimplify[FunctionExpand[Integrate[LegendreP[#, t] Exp[I q t], {t, -1, 1}] - 2 I^# SphericalBesselJ[#, q]],
      Assumptions -> q > 0] === 0 &]];

(* (b) the factor without Bessel functions: project the incident wave, integrate against the outgoing *)
(* every function is a product over the three axes, so every integral over the cube is a product of
   integrals over an interval; each is done symbolically, one variable at a time *)
mom[n_, q_, h_] := Integrate[LegendreP[n, t/h] Exp[I q t], {t, -h, h}];
direct[kin_, kout_, h_, p_] := Module[{num, den},
   (* coefficient of Q_a in the projection of the incident wave, times the moment of Q_a against the
      outgoing wave, summed over the basis *)
   num = Sum[
     Product[mom[a[[i]], kin[[i]], h]/(2 h/(2 a[[i]] + 1)) mom[a[[i]], -kout[[i]], h], {i, 3}], {a, indices[p]}];
   den = Product[mom[0, kin[[i]] - kout[[i]], h], {i, 3}];
   num/den];
kinT = {3/100, -1/80, 1/50}; koutT = {-1/20, 1/25, 11/1000}; hT = 5/2;
errsB = Table[Abs[N[direct[kinT, koutT, hT, p] - factor[kinT, koutT, hT, p], 40]], {p, 0, 2}];
Print["   direct against the formula, p = 0, 1, 2: ", ScientificForm[N[errsB], 2]];
check["(b) the factor built directly equals the formula, p = 0, 1, 2, to 1e-35", Max[errsB] < 10^-35];

(* (c) the leading term *)
pw[x_, n_] := If[n == 0, 1, x^n];  (* x^0 = 1 also when x is zero *)
lead[kin_, kout_, h_, p_] := (
   -h^(2 p + 2) Sum[
      Product[(2 a[[i]] + 1) pw[kin[[i]] kout[[i]], a[[i]]]/((2 a[[i]] + 1)!!)^2, {i, 3}],
      {a, Select[Tuples[Range[0, p + 1], 3], Total[#] == p + 1 &]}]);
kinS = {a1, a2, a3}; koutS = {b1, b2, b3};
(* the Bessel functions replaced by their Taylor polynomials, so that the expansion is of a ratio of
   polynomials in eps *)
jT[n_, x_] := Normal[Series[SphericalBesselJ[n, y], {y, 0, 10}]] /. y -> x;
factorT[kin_, kout_, h_, p_] := (
   Sum[Product[(2 a[[i]] + 1) jT[a[[i]], kin[[i]] h] jT[a[[i]], kout[[i]] h], {i, 3}], {a, indices[p]}]/
    Product[jT[0, (kin[[i]] - kout[[i]]) h], {i, 3}]);
seriesOK = Table[
   Module[{ser = Series[factorT[eps kinS, eps koutS, hh, p] - 1, {eps, 0, 2 p + 2}]},
    Expand[SeriesCoefficient[ser, 2 p + 2] - lead[kinS, koutS, hh, p]] === 0
     && AllTrue[Range[0, 2 p + 1], Expand[SeriesCoefficient[ser, #]] === 0 &]],
   {p, 0, 2}];
check["(c) E_p has no term below order 2p+2 and its leading term is the closed form, p = 0, 1, 2", And @@ seriesOK];
check["(c) for p = 0 the leading term is -(kin.kout) d^2 / 12",
  Simplify[lead[kinS, koutS, dd/2, 0] + (kinS . koutS) dd^2/12] === 0];

(* (d) one dimension: the constant of the layer *)
cp[p_] := (2 p + 3)/(4^(p + 1) ((2 p + 3)!!)^2);
oneD = Table[
   Simplify[lead[{0, 0, k}, {0, 0, s k}, dd/2, p] + s^(p + 1) cp[p] (k dd)^(2 p + 2)] === 0, {p, 0, 3}, {s, {1, -1}}];
check["(d) along one axis the leading term is -(+-1)^(p+1) c_p (k d)^(2p+2), p = 0..3", And @@ Flatten[oneD]];
check["(d) c_0, c_1, c_2 = 1/12, 1/720, 1/100800", {cp[0], cp[1], cp[2]} === {1/12, 1/720, 1/100800}];

(* (e) the factor tends to one as the degree grows *)
gap = Table[Abs[N[factor[kinT, koutT, hT, p] - 1, 30]], {p, {0, 1, 2, 4, 6, 8}}];
Print["   |factor - 1| at p = 0, 1, 2, 4, 6, 8: ", ScientificForm[N[gap], 2]];
check["(e) the factor tends to one as p grows (below 1e-20 at p = 8)", Last[gap] < 10^-20 && OrderedQ[Reverse[gap]]];

(* the export: 30-digit values for the Python cross-check *)
cases = Flatten[Table[
    <|"k_in" -> N[kin, 30], "k_out" -> N[kout, 30], "h" -> N[h, 30], "p" -> p,
      "factor" -> ToString[N[factor[kin, kout, h, p], 30], InputForm]|>,
    {kin, {kinT, {3/50, 0, 0}}}, {kout, {koutT, {-1/10, 0, 0}, {0, 1/10, 0}}}, {h, {5/2, 5}}, {p, 0, 2}], 3];
Export[FileNameJoin[{DirectoryName[$InputFileName], "AdaptiveOctree_wave_factor.json"}], cases, "JSON"];
Print[Count[oks, True], "/", Length[oks], " checks passed"];
If[! And @@ oks, Exit[1]];
