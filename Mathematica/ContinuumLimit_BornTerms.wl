#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_BornTerms.wl: the first two terms of the layer's scattering
   series, in closed form.

   THE SERIES.  Multiply the whole contrast of the layer by a number s.  The
   scattered displacement at an observer is analytic in s at s = 0,
       u(s) = s T1 + s^2 T2 + s^3 T3 + ... ,
   and Tn is the part of the field that has been scattered n times.

   T1 AND T2 AS INTEGRALS.  With w0 = (u0, e0) the incident displacement and
   strain, D1 = diag(w^2 drho, dM) the contrast, f(z) its profile, and the
   kernel  K = {{g, g'}, {g', -k^2 g - delta/M}},  g(z) = (i/2Mk) Exp[i k |z|],
       T1 = Int row(z) . D1 f(z) w0(z) dz ,
       w1(z) = Int K(z - z') . D1 f(z') w0(z') dz'      (the field scattered once),
       T2 = Int row(z) . D1 f(z) w1(z) dz ,
   with row(z) = (g, g')(z_obs - z) the kernel's displacement row at the observer.
   Every integrand is an exponential times the profile, so every integral is
   elementary.

   CHECKS
   (a) uniform layer: T1 and T2 from the integrals equal the Taylor coefficients
       in s of the exact reflection and transmission of the layer (continuity of
       displacement and traction at its faces), in reflection and transmission;
   (b) the closed form  T1(reflection) = (w^2 drho + k^2 dM) c^2 Exp[-i k (zo + zs)]
       (Exp[2 i k D] - 1)/(2 i k),  c = i/(2 M k): the impedance contrast;
   (c) the five-point difference formulas of the paper, applied to the exact
       layer at eps = 1/100, reproduce T1 and T2 to their stated remainders
       4 eps^4 T5 and 4 eps^4 T6;
   (d) the smooth profile 1 + Sin[2 Pi z/D]/2: T1 and T2 from the integrals.
   EXPORT: Mathematica/ContinuumLimit_born_terms.json (30 digits), read by
           scripts/measure_layer_bases.py.
   ============================================================================ *)

oks = {};
check[name_, ok_] := (AppendTo[oks, TrueQ[ok]]; Print[If[TrueQ[ok], "PASS  ", "FAIL  "], name]);

al = 5000; rho = 2500; mP = rho al^2; dM = 4 10^9; dRho = 100;
dL = 2; zs = -12; zR = -1; zT = 4; om = 300; k = om/al;
c = I/(2 mP k);
dq = {om^2 dRho, dM};

(* incident state, and the kernel's displacement row at the two observers (both outside the layer) *)
w0[z_] := {c Exp[I k (z - zs)], I k c Exp[I k (z - zs)]};
rowR[z_] := c Exp[I k (z - zR)] {1, -I k};   (* observer above: z_obs - z < 0 *)
rowT[z_] := c Exp[I k (zT - z)] {1, I k};    (* observer below: z_obs - z > 0 *)
kPlus[u_] := c Exp[I k u] {{1, I k}, {I k, -k^2}};     (* z - z' = u > 0 *)
kMinus[u_] := c Exp[I k u] {{1, -I k}, {-I k, -k^2}};  (* z' - z = u > 0 *)

t1[row_, f_] := Integrate[f[z] (row[z] . (dq w0[z])), {z, 0, dL}];
(* the field scattered once at depth z inside the layer: below, above, and the local term *)
w1[f_][z_] := (
   Integrate[kPlus[z - zp] . (dq f[zp] w0[zp]), {zp, 0, z}]
    + Integrate[kMinus[zp - z] . (dq f[zp] w0[zp]), {zp, z, dL}]
    + {0, -f[z] dM w0[z][[2]]/mP});
t2[row_, f_] := Module[{ws = Simplify[w1[f][zz]]},
   Integrate[(f[zz] (row[zz] . (dq ws))), {zz, 0, dL}]];

one[z_] := 1;
smooth[z_] := 1 + Sin[2 Pi z/dL]/2;

(* ---- the exact uniform layer as a function of the scale s ---- *)
exact[s_] := Module[{m1 = mP + s dM, r1 = rho + s dRho, k1, ep, em, a, b, sol, g0},
   k1 = om Sqrt[r1/m1]; ep = Exp[I k1 dL]; em = Exp[-I k1 dL];
   a = {{1, 0, -1, -1}, {-mP I k, 0, -m1 I k1, m1 I k1}, {0, -1, ep, em},
     {0, -mP I k, m1 I k1 ep, -m1 I k1 em}};
   b = {-1, -mP I k, 0, 0};
   sol = LinearSolve[a, b];
   g0 = c Exp[-I k zs];
   {sol[[1]] g0 Exp[-I k zR], sol[[2]] g0 Exp[I k (zT - dL)] - c Exp[I k (zT - zs)]}];

Print["integrals for the uniform layer ..."];
i1 = {t1[rowR, one], t1[rowT, one]};
i2 = {t2[rowR, one], t2[rowT, one]};
Print["Taylor coefficients of the exact layer ..."];
(* one exact series per observer, to sixth order: the list is expanded component by component *)
ex = exact[s];
ser = Table[Series[ex[[j]], {s, 0, 6}], {j, 2}];
coef[n_] := Table[SeriesCoefficient[ser[[j]], n], {j, 2}];
c1 = coef[1]; c2 = coef[2];
errA = Max[Abs[N[(i1 - c1)/c1, 40]], Abs[N[(i2 - c2)/c2, 40]]];
Print["   integrals against Taylor coefficients, worst relative difference: ", ScientificForm[N[errA], 2]];
check["(a) T1 and T2 from the integrals equal the Taylor coefficients of the exact layer", errA < 10^-35];

closed = (om^2 dRho + k^2 dM) c^2 Exp[-I k (zR + zs)] (Exp[2 I k dL] - 1)/(2 I k);
check["(b) T1 in reflection is the closed form with the impedance contrast w^2 drho + k^2 dM",
  Simplify[i1[[1]] - closed] === 0];
Print["   T1 (reflection, transmission) = ", N[i1, 12]];
Print["   T2 (reflection, transmission) = ", N[i2, 12]];
Print["   |T2|/|T1| = ", N[Abs[i2]/Abs[i1], 6]];

(* ---- (c) the difference formulas and their remainders ---- *)
eps = 1/100;
ev = Association[Table[x -> N[exact[x], 60], {x, {eps, -eps, 2 eps, -2 eps}}]];
d1 = (8 (ev[eps] - ev[-eps]) - (ev[2 eps] - ev[-2 eps]))/(12 eps);
d2 = (16 (ev[eps] + ev[-eps]) - (ev[2 eps] + ev[-2 eps]))/(24 eps^2);
t5 = coef[5]; t6 = coef[6];
(* 4 T_n eps^n-terms cancel: 8 (eps^5) - 32 eps^5 = -24 eps^5, over 12 eps, twice: T1 - 4 eps^4 T5 *)
rem1 = Abs[N[(d1 - (c1 - 4 eps^4 t5))/c1, 30]]; rem2 = Abs[N[(d2 - (c2 - 4 eps^4 t6))/c2, 30]];
Print["   difference formula minus (T - 4 eps^4 T_(n+4)), relative: ", ScientificForm[N[{rem1, rem2}], 2]];
Print["   size of the remainders 4 eps^4 |T5|/|T1|, 4 eps^4 |T6|/|T2|: ",
  ScientificForm[N[{4 eps^4 Abs[t5]/Abs[c1], 4 eps^4 Abs[t6]/Abs[c2]}], 2]];
check["(c) the five-point formulas give T1 and T2 with remainders -4 eps^4 T5 and -4 eps^4 T6",
  Max[rem1] < 10^-10 && Max[rem2] < 10^-10];

(* ---- (d) the smooth profile ---- *)
Print["integrals for the smooth profile ..."];
s1 = {t1[rowR, smooth], t1[rowT, smooth]};
s2 = {t2[rowR, smooth], t2[rowT, smooth]};
check["(d) the smooth profile's T1 and T2 are finite closed forms", AllTrue[N[Join[s1, s2], 20], NumericQ]];
Print["   T1 = ", N[s1, 12]]; Print["   T2 = ", N[s2, 12]];

str[x_] := {ToString[N[Re[x], 30], InputForm], ToString[N[Im[x], 30], InputForm]};
Export[FileNameJoin[{DirectoryName[$InputFileName], "ContinuumLimit_born_terms.json"}],
  <|"omega" -> om,
    "const" -> <|"T1" -> (str /@ i1), "T2" -> (str /@ i2)|>,
    "smooth" -> <|"T1" -> (str /@ s1), "T2" -> (str /@ s2)|>|>, "JSON"];
Print[Count[oks, True], "/", Length[oks], " checks passed"];
If[! And @@ oks, Exit[1]];
