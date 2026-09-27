#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_StaticLocal.wl  --  notebook 2b of the continuum-limit study.

   WHY.  Notebook 2a showed that, of the discrete-only (evanescent) parts of the
   specular lattice sums, only the STRAIN-FROM-MOMENT block survives as the cubes
   shrink: E(m) d^3 does not depend on the pitch d.  A term that survives d -> 0
   is STATIC.  This notebook computes it from the static (Kelvin) point Green's
   tensor, independently of the package, for the same-plane sum S(0) and the
   inter-plane sums S(m); and it assembles their total in slab order, the lattice
   part of the local field that notebook 4 balances against the continuum.

   THE STATIC TENSOR (shear modulus mu, Poisson ratio nu):
       G_ij(x) = [ (3 - 4 nu) d_ij + x_i x_j / r^2 ] / (16 pi mu (1 - nu) r).
   Block: strain rows by the receiver derivative (engineering factor 2 on shear
   rows), moment columns by +d (the package's convention, pinned in 2a),
   symmetrised.  It is fourth order, homogeneous of degree -3, so every static
   lattice sum is (1/d^3) x a pure number x (1/mu): computed here at d = 1.

   SUMS.  Degree -3 in two dimensions converges absolutely but slowly; partial
   sums over squares |p|,|q| <= N behave as S + a/N + b/N^2 + c/N^3, and are
   Richardson-extrapolated from N = 40, 80, 160, 320.  The same-plane scalar
       T0 = sum' 1/r^3 = 4 zeta(3/2) beta(3/2)
   validates the procedure.

   CHECKS (against scripts/continuum_limit_specular.py's JSON):
     [1] T0 closed form against the extrapolated direct sum;
     [2] static S(0) against the package's same-plane point sum, strain-from-
         moment block: the difference must be dynamic, i.e. fall as d -> 0;
     [3] static S(m), m = 1, 2, against the package's S(m) minus the continuum
         plane-wave term (2a's E(m)), likewise;
     [4] the slab-order total L = S(0) + sum_{m != 0} S(m) = the plate's depolarisation
         (the shape term of the slab order) plus a cubic-symmetric intrinsic part.
   Coordinates (z, x, y), z down; time e^{-i w t}; SI units.
   ============================================================================ *)

ref = Import["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_specular.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
om = ref["omega"]; al = ref["alpha"]; be = ref["beta"]; rho = ref["rho"];
kP = om/al; kS = om/be; mu = rho be^2; nu = (al^2 - 2 be^2)/(2 (al^2 - be^2));
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
nf[x_, n_] := ToString[NumberForm[x, n], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
relMax[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];

voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};  (* e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx *)
eng = {1, 1, 1, 2, 2, 2};
labels = {"zz", "xx", "yy", "xy", "zy", "zx"};

(* ---- the static strain-from-moment block, symbolic then vectorised ---- *)
Clear[z, x, y];
xv = {z, x, y}; rad = Sqrt[xv . xv];
gK = Table[((3 - 4 nu) KroneckerDelta[i, j] + xv[[i]] xv[[j]]/rad^2)/(16 Pi mu (1 - nu) rad), {i, 3}, {j, 3}];
emBlock = Table[
   Module[{a = voigt[[v, 1]], b = voigt[[v, 2]], c = voigt[[w, 1]], e = voigt[[w, 2]], col},
    col = Table[(1/2) (D[gK[[i, c]], xv[[e]]] + D[gK[[i, e]], xv[[c]]]), {i, 3}];
    eng[[v]] (1/2) (D[col[[b]], xv[[a]]] + D[col[[a]], xv[[b]]])],
   {v, 6}, {w, 6}];
emFun = Function[{zz, xx, yy}, Evaluate[emBlock /. {z -> zz, x -> xx, y -> yy}]];

(* lattice sum over the plane z = m (d = 1), origin excluded at m = 0; Richardson in N *)
nList = {40, 80, 160, 320};
planeSum[m_, f_] := Module[{nMax = Last[nList], pts, vals, rr, partial, fit, nn},
   pts = Flatten[Table[{p, q}, {p, -nMax, nMax}, {q, -nMax, nMax}], 1];
   If[m == 0, pts = DeleteCases[pts, {0, 0}]];
   rr = Max[Abs[#]] & /@ pts;
   vals = f[N[m], N[pts[[All, 1]]], N[pts[[All, 2]]]];
   vals = Map[If[ListQ[#], #, ConstantArray[N[#], Length[pts]]] &, vals, {2}];  (* entries that vanish identically *)
   vals = Transpose[vals, {2, 3, 1}];  (* -> points x 6 x 6 *)
   partial = Table[Total[Pick[vals, Thread[rr <= n]]], {n, nList}];
   (* fit S + a/N + b/N^2 + c/N^3 through the four partial sums, entrywise *)
   LinearSolve[Table[{1, 1/n, 1/n^2, 1/n^3}, {n, nList}] // N, partial][[1]]];
emFunScalar = Function[{zz, xx, yy}, ConstantArray[1/(xx^2 + yy^2)^(3/2), {1, 1}]];

Print["==== ContinuumLimit_StaticLocal :: the static local term of the specular lattice sums ===="];
Print["  mu = ", sci[mu], " Pa, nu = ", nu, ";  1/mu = ", sci[1/mu], " (the natural scale of every entry)"];

(* ---------------------------------------------------------------------------
   [1] the procedure, on the one sum with a closed form
   --------------------------------------------------------------------------- *)
t0 = N[4 Zeta[3/2] DirichletBeta[3/2], 16];
t0sum = planeSum[0, Function[{zz, xx, yy}, {{1/(xx^2 + yy^2)^(3/2)}}]][[1, 1]];
err1 = Abs[t0sum - t0]/t0;
Print["  [1] T0 = 4 zeta(3/2) beta(3/2) = ", nf[t0, 12], ";  extrapolated direct sum ",
  nf[t0sum, 12], ";  rel ", sci[err1], " -> ", pass[err1 < 10^-9]];

(* ---------------------------------------------------------------------------
   [2] the same-plane sum S(0): static, against the package's point sum
   --------------------------------------------------------------------------- *)
s0 = planeSum[0, emFun];  (* x d^-3 *)
Print["  [2] static same-plane sum S(0) x d^3 x mu, strain-from-moment (nonzero entries):"];
Do[If[Abs[s0[[v, w]]] mu > 10^-10, Print["        ", labels[[v]], " <- ", labels[[w]], "   ", nf[s0[[v, w]] mu, 10]]],
  {v, 6}, {w, 6}];
pkg0 = Table[{e["d"], mat[e["S"]][[4 ;; 9, 4 ;; 9]] e["d"]^3}, {e, Select[ref["lattice"], #["m"] == 0 &]}];
res2 = Table[{p[[1]], relMax[p[[2]], s0]}, {p, pkg0}];
Do[Print["      d = ", r2[[1]], ":  package vs static ", sci[r2[[2]]], "   (k_S d = ", sci[kS r2[[1]]], ")"], {r2, res2}];
ok2 = res2[[1, 2]] < 10^-2 && res2[[1, 2]] < res2[[2, 2]] < res2[[3, 2]];
Print["      the difference is dynamic (falls with d): ", pass[ok2]];

(* ---------------------------------------------------------------------------
   [3] the inter-plane sums S(m), m = 1, 2.  The static plate term (G = 0) of this
   block VANISHES for m != 0 (a plane of uniform moments strains only its own
   plane), so the static S(m) is wholly the evanescent local term.  The package's
   S(m) carries in addition the dynamic plane-wave term; subtract it (2a's cont).
   --------------------------------------------------------------------------- *)
gam[k_] := k;  (* normal incidence *)
contEM[d_, m_] := Module[{zz = m d, s = Sign[m], out},
   (* continuum term (1/d^2) G^(0; m d), strain-from-moment block, at zero lateral wavenumber *)
   out = Sum[
     Module[{k = w[[1]], kv, amp, uu, rows},
      kv = {k s, 0, 0};
      amp = I/(2 rho om^2) Exp[I k Abs[zz]]/k;
      uu = If[w[[2]] == "P", amp Outer[Times, kv, kv], amp (k^2 IdentityMatrix[3] - Outer[Times, kv, kv])];
      rows[col_] := Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] col[[voigt[[v, 2]]]] +
            I kv[[voigt[[v, 2]]]] col[[voigt[[v, 1]]]]), {v, 6}];
      Transpose[Table[-(1/2) rows[(-I kv[[voigt[[c, 2]]]]) uu[[All, voigt[[c, 1]]]] +
           (-I kv[[voigt[[c, 1]]]]) uu[[All, voigt[[c, 2]]]]], {c, 6}]]],
     {w, {{kP, "P"}, {kS, "S"}}}];
   out/d^2];
(* m != 0: the real-space sum floors near 1e-7 (it cancels O(1) terms down to e^{-2 pi m}), so use the
   STATIC Poisson sum, exponentially convergent: the w -> 0 limit of 2a's evanescent spectral kernel,
   taken symbolically with exact media (a machine-real P/S cancellation would leave 1e-16/w^2 debris). *)
{alX, beX, rhoX} = Rationalize[{al, be, rho}, 0];
Clear[kx, ky, zp, w0];
staticSpec = Module[{kap = Sqrt[kx^2 + ky^2], parts},
   parts = Table[
     Module[{k = w0/c, g, kv, amp, uu},
      g = I Sqrt[kap^2 - k^2];                                  (* evanescent branch, Im g > 0 *)
      kv = {g, kx, ky};                                         (* z > 0 *)
      amp = I/(2 rhoX w0^2) Exp[I g zp]/g;
      uu = If[c == alX, amp Outer[Times, kv, kv], amp (k^2 IdentityMatrix[3] - Outer[Times, kv, kv])];
      Transpose[Table[-(1/2) Table[eng[[v]] (1/2) (I kv[[voigt[[v, 1]]]] #[[voigt[[v, 2]]]] +
               I kv[[voigt[[v, 2]]]] #[[voigt[[v, 1]]]]), {v, 6}] &[
          (-I kv[[voigt[[cc, 2]]]]) uu[[All, voigt[[cc, 1]]]] + (-I kv[[voigt[[cc, 1]]]]) uu[[All, voigt[[cc, 2]]]]],
        {cc, 6}]]],
     {c, {alX, beX}}];
   Simplify[Normal[Series[Total[parts], {w0, 0, 0}]], Assumptions -> {zp > 0, kx \[Element] Reals, ky \[Element] Reals}]];
staticSpecF = Function[{kxv, kyv, zv}, Evaluate[staticSpec /. {kx -> kxv, ky -> kyv, zp -> zv}]];
(* S(m), m > 0, d = 1: sum over G = 2 pi (p, q) != 0 (the G = 0 plate term of this block vanishes);
   m < 0 by the reflection z -> -z, which flips the sign of every entry odd in z *)
poissonPlane[m_] := Module[{pm = 8},
   Re[Sum[If[p == 0 && q == 0, 0, staticSpecF[2. Pi p, 2. Pi q, N[m]]], {p, -pm, pm}, {q, -pm, pm}]]];
zOdd = Table[If[OddQ[Count[Join[voigt[[v]], voigt[[w]]], 1]], -1, 1], {v, 6}, {w, 6}];
sm = Association[Join[
    Table[m -> poissonPlane[m], {m, 1, 4}],
    Table[-m -> zOdd poissonPlane[m], {m, 1, 4}]]];
Print["      static Poisson (m = 1) vs the real-space extrapolated sum: ",
  sci[relMax[sm[1], planeSum[1, emFun]]], " (the real-space floor, ~1e-7 absolute)"];
Print["  [3] static S(m) x d^3 x mu, max |entry|, against the package's S(m) minus its plane-wave term:"];
res3 = Table[
   Module[{pk = Select[ref["lattice"], #["m"] == m && #["d"] == d &][[1]], ev},
    ev = (mat[pk["S"]][[4 ;; 9, 4 ;; 9]] - contEM[d, m]) d^3;
    {m, d, relMax[ev, sm[m]]}],
   {m, {1, 2}}, {d, {0.5, 1.0, 2.0}}];
Do[Print["      m = ", m, ": static ", sci[Max[Abs[Flatten[sm[m]]]] mu], "   vs package  ",
   Row[Table["d=" <> ToString[r3[[2]]] <> ": " <> sci[r3[[3]]], {r3, res3[[m]]}], "   "]], {m, {1, 2}}];
ok3 = And @@ Table[res3[[m, 1, 3]] < 10^-2 && res3[[m, 1, 3]] < res3[[m, 3, 3]], {m, {1, 2}}];
Print["      the static term is the whole local part; the residual is dynamic: ", pass[ok3]];
Print["      decay of the static plane sums, max |S(m)| x mu for m = 1..4: ",
  sci /@ Table[Max[Abs[Flatten[sm[m]]]] mu, {m, 4}]];

(* ---------------------------------------------------------------------------
   [4] the slab-order total L = S(0) + sum_{m != 0} S(m): the static lattice local
   field of an infinite point-coupled plate, per unit d^-3.  Exponential in m, so
   m = +-4 is converged (see the decay above).
   --------------------------------------------------------------------------- *)
lTot = s0 + Total[Values[sm]];
Print["  [4] slab-order static lattice sum L x d^3 x mu (point kernel), nonzero entries:"];
Do[If[Abs[lTot[[v, w]]] mu > 10^-10, Print["        ", labels[[v]], " <- ", labels[[w]], "   ", nf[lTot[[v, w]] mu, 10]]],
  {v, 6}, {w, 6}];
Print["      the inter-plane share of L (max entry): ", sci[Max[Abs[Flatten[Total[Values[sm]]]]]/Max[Abs[Flatten[lTot]]]]];
(* Summing planes first imposes the SLAB order, so L = (the plate's depolarisation) + (a cubic-symmetric
   intrinsic part).  The plate term sits on the normal components only: -1/(lambda + 2 mu) on
   e_zz <- M_zz and -1/(2 mu) on the engineering 2e_zy, 2e_zx <- M_zy, M_zx; in units 1/mu,
   -beta^2/alpha^2 and -1/2.  So the cubic part must be recovered by adding them back. *)
lMu = lTot mu;
shapeDiag = lMu[[1, 1]] - lMu[[2, 2]]; shapeShear = lMu[[5, 5]] - lMu[[4, 4]];
offSpread = Max[Abs[{lMu[[1, 2]], lMu[[1, 3]], lMu[[2, 3]]} - lMu[[2, 3]]]];
err4 = Max[Abs[shapeDiag + beX^2/alX^2], Abs[shapeShear + 1/2], offSpread];
Print["      shape: zz - xx = ", nf[shapeDiag, 10], " (plate: -beta^2/alpha^2 = ", N[-beX^2/alX^2], ");  ",
  "zy - xy = ", nf[shapeShear, 10], " (plate: -1/2);  off-diagonal spread ", sci[offSpread]];
Print["      L = plate depolarisation + cubic part, to ", sci[err4], " -> ", pass[err4 < 10^-8]];
cubA = lMu[[2, 2]]; cubB = lMu[[2, 3]]; cubC = lMu[[4, 4]];
Print["      the cubic intrinsic part (x 1/mu): diagonal ", nf[cubA, 10], ", off-diagonal ",
  nf[cubB, 10], ", shear ", nf[cubC, 10]];
Print["      So the point-coupled plate reproduces the continuum plate EXACTLY iff the cube's self term"];
Print["      cancels this cubic part: self + L_cubic = 0.  Notebook 4 tests that condition."];

Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ContinuumLimit_staticlocal.json",
  <|"units" -> "x d^3 x mu, strain-from-moment block, rows/cols e_zz e_xx e_yy 2e_xy 2e_zy 2e_zx",
   "mu" -> mu, "nu" -> nu, "S0" -> s0 mu, "Sm" -> KeyMap[ToString, Map[# mu &, sm]], "L" -> lTot mu,
   "cubic" -> <|"diag" -> cubA, "off" -> cubB, "shear" -> cubC|>|>, "RawJSON"];

Print["==== ContinuumLimit_StaticLocal (stage 2b): ",
  If[err1 < 10^-9 && ok2 && ok3 && err4 < 10^-8, "ALL CHECKS PASS", "CHECKS FAILED"], " ===="];
