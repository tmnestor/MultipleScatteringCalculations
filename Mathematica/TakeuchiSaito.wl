(* ::Package:: *)

(* =============================================================================================================
   TakeuchiSaito.wl  --  the Takeuchi-Saito radial systems of an isotropic, radially varying elastic sphere:
                         derivation, the homogeneous-medium basis, and evaluation.

   A self-contained Wolfram Language file.  It depends on nothing outside itself.

   USE
     As a library:     Get["TakeuchiSaito.wl"]   defines the functions below (see ?TS* for their usage).
     In a notebook:    Get[".../TakeuchiSaito.wl"] (no output: it only defines), then TSSelfTest[] for the
                       derivation and verification report.
     As a script:      wolframscript -file TakeuchiSaito.wl  runs TSSelfTest[]: it derives every system from the
                       equations of motion, verifies each stated result, and prints a report ending
                       "ALL n CHECKS PASS" (about fifteen seconds).

   WHAT IS HERE
     1. The first-order radial systems  dy/dr = A(r) y  of Takeuchi & Saito (1972):
          spheroidal (P-SV)  y = (U, V, R, S),   toroidal (SH)  z = (W, T),
        derived symbolically from the equations of motion in spherical coordinates (TSDerive) and stated in
        closed form (TSSpheroidalMatrix, TSToroidalMatrix).
     2. The exact solutions in a homogeneous medium, from the scalar potentials, for every spherical Bessel
        kind (TSHomogeneousBasis, TSHomogeneousToroidal): the "spherical basis" in which any radially
        varying medium is matched to its homogeneous surroundings.
     3. Structural facts, each verified: no derivative of the material enters A; trace A = -4/r and
        trace B = -2/r; the Frobenius exponents at r = 0; the reciprocity invariant r^2 y1.J.y2
        (TSInvariantMatrix), constant between any two solutions in any radially varying medium.
     4. A high-precision propagator through any radial profile, with jumps allowed at given radii
        (TSPropagate), and the per-order T-matrix of a sphere whose properties vary with radius (TSSphereTMatrix).
     5. The closed-form T-matrix of a sphere of homogeneous concentric shells (TSShellsTMatrix), the exact
        reference for any piecewise-constant profile.

   CONVENTIONS
     Coordinates (r, theta, phi); time dependence e^{-i omega t}; SI units throughout (any consistent set works).
     Lame parameters lambda(r), mu(r); density rho(r).  L = n(n+1) for angular order n.
     Spheroidal field of order n (for Y = Y_n^m, any m; the radial equations do not depend on m):
         u = U(r) Y r_hat + V(r) grad_1 Y,         grad_1 = theta_hat d/dtheta + phi_hat (1/sin theta) d/dphi
         traction on a sphere  t = sigma . r_hat = R(r) Y r_hat + S(r) grad_1 Y
       so for m = 0:  u_r = U P_n,  u_theta = V dP_n/dtheta,  sigma_rr = R P_n,  sigma_rtheta = S dP_n/dtheta.
     Toroidal field:
         u = W(r) (-r_hat x grad_1 Y),             t = T(r) (-r_hat x grad_1 Y),
       so for m = 0:  u_phi = W dP_n/dtheta,  sigma_rphi = T dP_n/dtheta,  with T = mu (W' - W/r).
     Potentials (homogeneous medium, wavenumbers kP = omega/alpha, kS = omega/beta):
         P:   u = grad( z_n(kP r) Y )
         SV:  u = curl curl( r_vec z_n(kS r) Y )
         SH:  u = z_n(kS r) (-r_hat x grad_1 Y)
       with z_n one of j_n, y_n, h_n^(1), h_n^(2).  h^(1) is OUTGOING for e^{-i omega t}.
     T-matrix of order n: maps incident potential coefficients (P, SV) to scattered ones (2 x 2), and the
     incident toroidal coefficient to the scattered one (scalar), for incident z = j and scattered z = h^(1).

   REFERENCES
     H. Takeuchi and M. Saito (1972), Seismic surface waves, Methods in Computational Physics 11, 217-295.
     F. A. Dahlen and J. Tromp (1998), Theoretical Global Seismology, Princeton, ch. 8.
     Y.-H. Pao and C.-C. Mow (1973), Diffraction of Elastic Waves and Dynamic Stress Concentrations.
   ============================================================================================================= *)

BeginPackage["TakeuchiSaito`"];

TSSpheroidalMatrix::usage = "TSSpheroidalMatrix[n, omega, lambda, mu, rho, r] is the 4x4 matrix A of dy/dr = A y \
for the spheroidal state y = (U, V, R, S) of angular order n, with the material values at radius r.";
TSToroidalMatrix::usage = "TSToroidalMatrix[n, omega, mu, rho, r] is the 2x2 matrix B of dz/dr = B z for the \
toroidal state z = (W, T).";
TSDerive::usage = "TSDerive[] derives the spheroidal and toroidal matrices symbolically from the equations of \
motion in spherical coordinates, for general lambda[r], mu[r], rho[r], and returns <|\"A\" -> A, \"B\" -> B|> \
in the symbols lambda[r], mu[r], rho[r], L, omega, r of this context.";
TSHomogeneousBasis::usage = "TSHomogeneousBasis[n, omega, lambda, mu, rho, r, kind] is the 4x2 matrix whose \
columns are the spheroidal states (U, V, R, S) of the P and SV potential solutions of order n in a homogeneous \
medium, with radial function kind = \"j\", \"y\", \"h1\" or \"h2\".  For n = 0 the SV column is zero.";
TSHomogeneousToroidal::usage = "TSHomogeneousToroidal[n, omega, mu, rho, r, kind] is the toroidal state (W, T) \
of the SH solution of order n in a homogeneous medium.";
TSInvariantMatrix::usage = "TSInvariantMatrix[n] is the constant antisymmetric J for which r^2 y1.J.y2 is \
independent of r for any two spheroidal solutions y1, y2, in any radially varying isotropic medium.";
TSPropagate::usage = "TSPropagate[kind, n, omega, {lamF, muF, rhoF}, {r0, r1}, y0, opts] integrates the \
spheroidal (kind = \"spheroidal\", y0 of length 4, or length 2 (U, R) for n = 0) or toroidal (kind = \
\"toroidal\") system from r0 to r1, with material functions lamF[r], muF[r], rhoF[r].  Options: \
WorkingPrecision (default 50), \"Breaks\" (radii where a material function jumps; default {}: every \
segment between breaks takes the material from its own side of each break), \
\"TractionScale\" (default Automatic = r1/muF[r1]).  The integrator's precision and accuracy goals are \
WorkingPrecision/2, so a propagated state (and a T-matrix through a varying shell) is accurate to about \
WorkingPrecision/2 digits; a homogeneous sphere (b = a) is exact to WorkingPrecision.";
TSShellsTMatrix::usage = "TSShellsTMatrix[n, omega, {lam0, mu0, rho0}, {r1, ..., rN}, {{lam1, mu1, rho1}, ..., \
{lamN, muN, rhoN}}] is the order-n T-matrix <|\"Tpsv\" -> 2x2, \"Tsh\" -> scalar|> of a sphere of \
homogeneous concentric shells, in closed form: region i (r_{i-1} < r < r_i, r_0 = 0) carries the i-th material, \
and rN is the outer radius.  Option: WorkingPrecision (default 50).";
TSSelfTest::usage = "TSSelfTest[] derives the systems from the equations of motion, verifies every result \
stated in this file (9 checks, about fifteen seconds) and prints a report; it returns True when all pass.";
TSSphereTMatrix::usage = "TSSphereTMatrix[n, omega, {lam0, mu0, rho0}, {lamF, muF, rhoF}, {b, a}, opts] is \
<|\"Tpsv\" -> 2x2, \"Tsh\" -> scalar|>, the order-n T-matrix of a sphere of radius a in the homogeneous \
background (lam0, mu0, rho0): homogeneous for r < b with the material just below b (so a profile may jump \
at b), and varying as lamF[r], muF[r], rhoF[r] for b < r < a (give any further jumps as \"Breaks\").  b = a gives the homogeneous sphere (Mie).  Options as TSPropagate.";

Begin["`Private`"];

(* ------------------------------------------------------------------------------------------------------------
   1. The systems in closed form.  These are exactly what TSDerive[] returns (verified by the self-test).
   ------------------------------------------------------------------------------------------------------------ *)
TSSpheroidalMatrix[n_, om_, lam_, mu_, rho_, r_] := Module[{L = n (n + 1), m = lam + 2 mu, xi},
   xi = mu (3 lam + 2 mu)/m;
   {{-2 lam/(r m), L lam/(r m), 1/m, 0},
    {-1/r, 1/r, 0, 1/mu},
    {-om^2 rho + 4 xi/r^2, -2 L xi/r^2, -4 mu/(r m), L/r},
    {-2 xi/r^2, -om^2 rho + (2 mu/r^2) (2 L (lam + mu)/m - 1), -lam/(r m), -3/r}}];

TSToroidalMatrix[n_, om_, mu_, rho_, r_] := {{1/r, 1/mu}, {(n (n + 1) - 2) mu/r^2 - om^2 rho, -3/r}};

(* ------------------------------------------------------------------------------------------------------------
   The derivation.  Axisymmetric fields (m = 0) suffice for the radial equations (their m-independence is
   verified separately on exact 3-D solutions).  Legendre's equation P'' = -cot(t) P' - L P eliminates every
   second and higher angular derivative, after which each equation of motion separates into a multiple of P_n
   (radial component) or of dP_n/dt (tangential component).
   ------------------------------------------------------------------------------------------------------------ *)
TSDerive[] := Module[{ur, ut, err, ett, epp, ert, trE, srr, stt, spp, srt, eqR, eqT, red, legRule, coefP, coefD,
    Rdef, Sdef, dUV, ddUV, toY, dR, dS, rhs, A, up, erp, etp, srp, stp, eqP, Tdef, dW, ddW, dT, B,
    U, V, W, p, t, y1, y2, y3, y4, z1, z2},
   legRule = Derivative[k_][p][t] /; k >= 2 :> D[-Cot[t] Derivative[1][p][t] - L p[t], {t, k - 2}];
   red[e_] := Simplify[Expand[e //. legRule]];
   coefP[e_] := Simplify[Coefficient[Expand[e], p[t]] /. Derivative[1][p][t] -> 0];
   coefD[e_] := Simplify[Coefficient[Expand[e], Derivative[1][p][t]] /. p[t] -> 0];
   (* spheroidal: strains, stresses, and the r- and theta-components of div(sigma) + omega^2 rho u *)
   ur = U[r] p[t]; ut = V[r] p'[t];
   err = D[ur, r]; ett = (D[ut, t] + ur)/r; epp = (ur + ut Cot[t])/r; ert = (D[ut, r] - ut/r + D[ur, t]/r)/2;
   trE = err + ett + epp;
   srr = lambda[r] trE + 2 mu[r] err; stt = lambda[r] trE + 2 mu[r] ett;
   spp = lambda[r] trE + 2 mu[r] epp; srt = 2 mu[r] ert;
   eqR = red[D[srr, r] + D[srt, t]/r + (2 srr - stt - spp + srt Cot[t])/r + omega^2 rho[r] ur];
   eqT = red[D[srt, r] + D[stt, t]/r + ((stt - spp) Cot[t] + 3 srt)/r + omega^2 rho[r] ut];
   If[Simplify[coefD[eqR]] =!= 0 || Simplify[coefP[eqT]] =!= 0, Return[$Failed, Module]];
   (* the traction coefficients R, S define the state; U', V' follow from them, U'', V'' from the equations *)
   Rdef = coefP[red[srr]]; Sdef = coefD[red[srt]];
   toY = {U[r] -> y1, V[r] -> y2};
   dUV = First@Solve[{y3 == Rdef, y4 == Sdef} /. toY, {U'[r], V'[r]}];
   ddUV = First@Solve[{coefP[eqR] == 0, coefD[eqT] == 0}, {U''[r], V''[r]}];
   dR = D[Rdef, r] /. ddUV /. toY /. dUV;
   dS = D[Sdef, r] /. ddUV /. toY /. dUV;
   rhs = Simplify[{U'[r] /. dUV, V'[r] /. dUV, dR, dS}];
   A = Simplify[Table[D[rhs[[i]], {y1, y2, y3, y4}[[j]]], {i, 4}, {j, 4}]];
   (* toroidal *)
   up = W[r] p'[t];
   erp = (D[up, r] - up/r)/2; etp = (D[up, t] - up Cot[t])/(2 r);
   srp = 2 mu[r] erp; stp = 2 mu[r] etp;
   eqP = red[D[srp, r] + D[stp, t]/r + (3 srp + 2 stp Cot[t])/r + omega^2 rho[r] up];
   Tdef = coefD[red[srp]];
   dW = First@Solve[z2 == Tdef /. W[r] -> z1, W'[r]];
   ddW = First@Solve[coefD[eqP] == 0, W''[r]];
   dT = D[Tdef, r] /. ddW /. W[r] -> z1 /. dW;
   B = Simplify[Table[D[{W'[r] /. dW, dT}[[i]], {z1, z2}[[j]]], {i, 2}, {j, 2}]];
   <|"A" -> A, "B" -> B|>];

(* ------------------------------------------------------------------------------------------------------------
   2. The homogeneous-medium basis.  Displacements and tractions of the potentials, in closed form; the
   stresses use Bessel's equation, z'' = -(2/x) z' - (1 - L/x^2) z, to remove second derivatives.
   ------------------------------------------------------------------------------------------------------------ *)
zfun["j"] = SphericalBesselJ; zfun["y"] = SphericalBesselY;
zfun["h1"] = SphericalHankelH1; zfun["h2"] = SphericalHankelH2;
zAndDeriv[kind_, n_, x_] := With[{f = zfun[kind]}, {f[n, x], If[n == 0, -f[1, x], f[n - 1, x] - (n + 1)/x f[n, x]]}];

pColumn[n_, k_, lam_, mu_, r_, kind_] := Module[{z, zp, x = k r},
   {z, zp} = zAndDeriv[kind, n, x];
   {k zp, z/r,
    -(lam + 2 mu) k^2 z - 4 mu k zp/r + 2 mu n (n + 1) z/r^2,
    2 mu (k zp - z/r)/r}];
sColumn[n_, k_, mu_, r_, kind_] := Module[{z, zp, x = k r, L = n (n + 1)},
   If[n == 0, Return[{0, 0, 0, 0}, Module]];
   {z, zp} = zAndDeriv[kind, n, x];
   {L z/r, (z + x zp)/r, 2 mu L (k zp/r - z/r^2), mu ((2 L - 2 - x^2) z - 2 x zp)/r^2}];

TSHomogeneousBasis[n_, om_, lam_, mu_, rho_, r_, kind_] := Module[{kP = om Sqrt[rho/(lam + 2 mu)], kS = om Sqrt[rho/mu]},
   Transpose[{pColumn[n, kP, lam, mu, r, kind], sColumn[n, kS, mu, r, kind]}]];

TSHomogeneousToroidal[n_, om_, mu_, rho_, r_, kind_] := Module[{k = om Sqrt[rho/mu], z, zp},
   {z, zp} = zAndDeriv[kind, n, k r];
   {z, mu (k zp - z/r)}];

(* ------------------------------------------------------------------------------------------------------------
   3. The reciprocity invariant: d/dr (r^2 y1.J.y2) = r^2 y1.(A^T J + J A + (2/r) J).y2 = 0.  The constant
   antisymmetric J that makes the bracket vanish for every medium (found by TSDerive's self-test):
   ------------------------------------------------------------------------------------------------------------ *)
TSInvariantMatrix[n_] := With[{L = n (n + 1)}, {{0, 0, 1, 0}, {0, 0, 0, L}, {-1, 0, 0, 0}, {0, -L, 0, 0}}];

(* ------------------------------------------------------------------------------------------------------------
   4. Evaluation.  Tractions are scaled to the size of displacements for the integrator (the scale is undone
   on output), and the integration restarts at every break, so a jump in the material costs no accuracy.
   Bulirsch-Stoer extrapolation is used: it keeps its order at arbitrary precision.
   ------------------------------------------------------------------------------------------------------------ *)
Options[TSPropagate] = {WorkingPrecision -> 50, "Breaks" -> {}, "TractionScale" -> Automatic};
TSPropagate[kind_, n_, om_, {lamF_, muF_, rhoF_}, {r0_, r1_}, y0_, OptionsPattern[]] := Module[
   {wp = OptionValue[WorkingPrecision], sc, dim = Length[y0], D1, mat, pts, Y, x},
   sc = OptionValue["TractionScale"] /. Automatic -> r1/muF[r1];
   D1 = Which[kind === "toroidal", DiagonalMatrix[{1, sc}], dim == 2, DiagonalMatrix[{1, sc}],
     True, DiagonalMatrix[{1, 1, sc, sc}]];
   (* geometry at s, material at sm *)
   mat[s_, sm_] := D1 . Which[
       kind === "toroidal", TSToroidalMatrix[n, om, muF[sm], rhoF[sm], s],
       dim == 2, TSSpheroidalMatrix[0, om, lamF[sm], muF[sm], rhoF[sm], s][[{1, 3}, {1, 3}]],
       True, TSSpheroidalMatrix[n, om, lamF[sm], muF[sm], rhoF[sm], s]] . Inverse[D1];
   pts = Union[{r0, r1}, Select[OptionValue["Breaks"], r0 < # < r1 &]];
   If[r0 > r1, pts = Reverse[pts]];
   (* within each segment the material is evaluated a relative 10^-wp inside it, so that at a break, where
      a material function jumps, each segment sees its own side *)
   Inverse[D1] . Fold[Function[{y, seg}, Module[{lo = Min[seg], hi = Max[seg], del},
        del = (hi - lo) 10^-wp;
        NDSolveValue[{Y'[x] == mat[x, Min[Max[x, lo + del], hi - del]] . Y[x], Y[seg[[1]]] == SetPrecision[y, wp]},
         Y[seg[[2]]], {x, seg[[1]], seg[[2]]}, WorkingPrecision -> wp, PrecisionGoal -> wp/2,
         AccuracyGoal -> wp/2, MaxSteps -> Infinity, Method -> "Extrapolation"]]],
     SetPrecision[D1 . y0, wp], Partition[pts, 2, 1]]];

Options[TSSphereTMatrix] = Options[TSPropagate];
TSSphereTMatrix[n_, om_, {lam0_, mu0_, rho0_}, {lamF_, muF_, rhoF_}, {b_, a_}, opts : OptionsPattern[]] := Module[
   {wp = OptionValue[WorkingPrecision], ev, lc, mc, rc, bm, core, inside, out, inc, sol, zc, zin, tsh, rs, rs2},
   ev[e_] := SetPrecision[N[e, wp + 20], wp];
   (* the core's material is taken just below b, so a profile may jump at the core radius *)
   bm = b (1 - 10^-(wp + 10)); {lc, mc, rc} = {lamF[bm], muF[bm], rhoF[bm]};
   (* the traction rows of every matching system are scaled by a/mu0, to the size of the displacement rows:
      T does not change, and no digits are lost to the SI scale gap in the solve *)
   rs = DiagonalMatrix[{1, 1, a/mu0, a/mu0}]; rs2 = DiagonalMatrix[{1, a/mu0}];
   core = ev[TSHomogeneousBasis[n, om, lc, mc, rc, b, "j"]];
   out = ev[TSHomogeneousBasis[n, om, lam0, mu0, rho0, a, "h1"]];
   inc = ev[TSHomogeneousBasis[n, om, lam0, mu0, rho0, a, "j"]];
   If[n == 0,
    (* the monopole has only (U, R) and only a P wave *)
    inside = If[b == a, core[[{1, 3}, 1]],
      TSPropagate["spheroidal", 0, om, {lamF, muF, rhoF}, {b, a}, core[[{1, 3}, 1]], opts]];
    sol = LinearSolve[rs2 . Transpose[{out[[{1, 3}, 1]], -inside}], -rs2 . inc[[{1, 3}, 1]]];
    Return[<|"Tpsv" -> {{sol[[1]], 0}, {0, 0}}, "Tsh" -> 0|>, Module]];
   inside = Table[If[b == a, core[[All, c]],
      TSPropagate["spheroidal", n, om, {lamF, muF, rhoF}, {b, a}, core[[All, c]], opts]], {c, 2}];
   sol = LinearSolve[rs . Transpose[{out[[All, 1]], out[[All, 2]], -inside[[1]], -inside[[2]]}], -rs . inc];
   zc = ev[TSHomogeneousToroidal[n, om, mc, rc, b, "j"]];
   zin = If[b == a, zc, TSPropagate["toroidal", n, om, {lamF, muF, rhoF}, {b, a}, zc, opts]];
   tsh = LinearSolve[rs2 . Transpose[{ev[TSHomogeneousToroidal[n, om, mu0, rho0, a, "h1"]], -zin}],
      -rs2 . ev[TSHomogeneousToroidal[n, om, mu0, rho0, a, "j"]]][[1]];
   <|"Tpsv" -> sol[[1 ;; 2]], "Tsh" -> tsh|>];

(* ------------------------------------------------------------------------------------------------------------
   5. A sphere of homogeneous concentric shells, in closed form: regular (j) solutions in the core, j and y in
   every shell, outgoing (h1) outside; continuity of the state at every interface.  The exact reference for any
   piecewise-constant radial profile, and an independent check of TSSphereTMatrix with breaks.
   ------------------------------------------------------------------------------------------------------------ *)
Options[TSShellsTMatrix] = {WorkingPrecision -> 50};
TSShellsTMatrix[n_, om_, {lam0_, mu0_, rho0_}, radii_List, mats_List, OptionsPattern[]] := Module[
   {wp = OptionValue[WorkingPrecision], ev, nr = Length[radii], a = Last[radii], rows, sel, fams, sph, tor, solve},
   ev[e_] := SetPrecision[N[e, wp + 20], wp];
   (* region i = 1 .. nr carries mats[[i]] inside radii[[i]]; region nr + 1 is the background *)
   sph[i_, kind_, r_] := Module[{m = If[i > nr, {lam0, mu0, rho0}, mats[[i]]], bas},
     bas = ev[TSHomogeneousBasis[n, om, Sequence @@ m, r, kind]];
     If[n == 0, {bas[[{1, 3}, 1]]}, Transpose[bas]]];                 (* list of state columns *)
   tor[i_, kind_, r_] := Module[{m = If[i > nr, {mu0, rho0}, mats[[i, {2, 3}]]]},
     {ev[TSHomogeneousToroidal[n, om, Sequence @@ m, r, kind]]}];
   (* unknown columns: core j, then j and y in every shell, then h outside; each continuity block is
      (region i) - (region i + 1) at radii[[i]], and the incident j of the background goes to the right side *)
   solve[fld_, dim_, scale_] := Module[{cols, mat, rhs, nf = Length[fld[1, "j", a]]},
     cols = Join[{{1, "j"}}, Flatten[Table[{{i, "j"}, {i, "y"}}, {i, 2, nr}], 1], {{nr + 1, "h1"}}];
     mat = Flatten[Table[
        Transpose[Flatten[Table[Which[c[[1]] == i, fld[i, c[[2]], radii[[i]]],
             c[[1]] == i + 1, -fld[i + 1, c[[2]], radii[[i]]], True, ConstantArray[0, {nf, dim}]], {c, cols}], 1]],
        {i, nr}], 1];
     rhs = Flatten[Table[If[i == nr, Transpose[fld[nr + 1, "j", a]], ConstantArray[0, {dim, nf}]], {i, nr}], 1];
     (* traction rows scaled by a/mu0, as in TSSphereTMatrix *)
     mat = DiagonalMatrix[Flatten[Table[scale, {nr}]]] . mat;
     rhs = DiagonalMatrix[Flatten[Table[scale, {nr}]]] . rhs;
     LinearSolve[mat, rhs][[-nf ;;]]];
   If[n == 0,
    <|"Tpsv" -> {{solve[sph, 2, {1, a/mu0}][[1, 1]], 0}, {0, 0}}, "Tsh" -> 0|>,
    <|"Tpsv" -> solve[sph, 4, {1, 1, a/mu0, a/mu0}], "Tsh" -> solve[tor, 2, {1, a/mu0}][[1, 1]]|>]];

End[];
EndPackage[];

(* =============================================================================================================
   SELF-TEST.  TSSelfTest[] derives every system, verifies each stated result and prints a report; it returns
   True when every check passes.  Loading the file with Get only defines the functions; the self-test runs by
   itself only when the file is executed as a script (wolframscript -file TakeuchiSaito.wl).
   ============================================================================================================= *)
TSSelfTest[] := Quiet[
  Module[{oks = {}, chk, sci, pp, der, A, B, lam, mu, rho, r, L, omega, nn, J, aa, bracket, eqs, M0,
     exps, lm0, m0, rh0, w0, pars, profile, worst, worstR, sols, inv, okA, okB},
   chk[b_] := (AppendTo[oks, TrueQ[b]]; If[TrueQ[b], "PASS", "FAIL"]);
   (* print an expression without its internal context and module-variable suffixes *)
   pp[e_] := StringReplace[ToString[e, InputForm], {"TakeuchiSaito`Private`" -> "", RegularExpression["\\$\\d+"] -> ""}];
   (* two expected messages are silenced: N::meprec where an exact residual is identically zero (N cannot find
      digits of an exact 0), and Solve::svars where J is determined only up to its overall scale *)
   (* (these two expected messages are silenced by the Quiet around this function) *)
   sci[v_] := ToString[NumberForm[N[v], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
   (* the symbols in which TSDerive returns its matrices *)
   lam = TakeuchiSaito`Private`lambda; mu = TakeuchiSaito`Private`mu; rho = TakeuchiSaito`Private`rho;
   r = TakeuchiSaito`Private`r; L = TakeuchiSaito`Private`L; omega = TakeuchiSaito`Private`omega;
   Print["==== TakeuchiSaito :: radial systems of a radially varying isotropic elastic sphere ===="];

   (* [1] the derivation reproduces the closed forms, with no derivative of the material *)
   der = TSDerive[];
   A = der["A"]; B = der["B"];
   okA = Simplify[(A /. L -> nn (nn + 1)) - TSSpheroidalMatrix[nn, omega, lam[r], mu[r], rho[r], r]] ===
     ConstantArray[0, {4, 4}];
   okB = Simplify[(B /. L -> nn (nn + 1)) - TSToroidalMatrix[nn, omega, mu[r], rho[r], r]] === ConstantArray[0, {2, 2}];
   Print["  [1] derived from the equations of motion = the closed forms: spheroidal ", okA, ", toroidal ", okB,
     "; free of material derivatives: ", FreeQ[{A, B}, Derivative[_][lam | mu | rho]], " -> ",
     chk[okA && okB && FreeQ[{A, B}, Derivative[_][lam | mu | rho]]]];

   (* [2] traces: trace A = -4/r, trace B = -2/r, so the Wronskians scale as r^-4 and r^-2 *)
   Print["  [2] trace A = ", pp[Simplify[Tr[A]]], ", trace B = ", pp[Simplify[Tr[B]]], " -> ",
     chk[Simplify[Tr[A] + 4/r] === 0 && Simplify[Tr[B] + 2/r] === 0]];

   (* [3] the reciprocity invariant: A^T J + J A + (2/r) J = 0 for every medium, and J is the stated one *)
   (* the six independent entries of an antisymmetric J; the bracket must vanish identically in the material,
      the frequency and r, which makes every coefficient of its numerator a linear equation in them *)
   J = {{0, aa[1], aa[2], aa[3]}, {-aa[1], 0, aa[4], aa[5]}, {-aa[2], -aa[4], 0, aa[6]}, {-aa[3], -aa[5], -aa[6], 0}};
   bracket = Simplify[Transpose[A] . J + J . A + (2/r) J];
   eqs = DeleteCases[Flatten[CoefficientList[Numerator[Together[#]], {lam[r], mu[r], rho[r], omega, r}] & /@
       Flatten[bracket]], 0];
   sols = Solve[Thread[eqs == 0], Array[aa, 6]];
   inv = Simplify[J /. First[sols] /. aa[2] -> 1];
   Print["  [3] reciprocity: the antisymmetric J with d/dr(r^2 y1.J.y2) = 0 in every medium (normalised J13 = 1): ",
     pp[inv], " -> ", chk[Simplify[(inv /. L -> nn (nn + 1)) - TSInvariantMatrix[nn]] === ConstantArray[0, {4, 4}]]];

   (* [4] Frobenius exponents at r = 0, for the scaled state (U, V, r R, r S), from the closed form ([1] shows it is
      the derived system); the material is taken constant near the centre, where only its value there enters *)
   Module[{la, m, rh, rs, Sd},
     Sd = DiagonalMatrix[{1, 1, rs, rs}];
     M0 = Limit[rs Sd . TSSpheroidalMatrix[nn, omega, la, m, rh, rs] . Inverse[Sd] + DiagonalMatrix[{0, 0, 1, 1}], rs -> 0];
     exps = Sort[Simplify[Eigenvalues[M0], nn > 1]];
     Print["  [4] Frobenius exponents at r = 0: ", pp[exps],
       "   (regular: n - 1 and n + 1, the P and SV solutions near the centre) -> ",
       chk[Simplify[exps - Sort[{nn - 1, nn + 1, -nn, -nn - 2}]] === {0, 0, 0, 0}]]];

   (* [5] the homogeneous basis satisfies the systems, for every kind of Bessel function, n = 0..4 *)
   {lm0, m0, rh0, w0} = {5 10^9, 9 10^9, 2600, 400};
   worst = 0;
   Do[Module[{bas, dbas, zt, dzt, rr},
      bas = TSHomogeneousBasis[k, w0, lm0, m0, rh0, rr, kind];
      dbas = D[bas, rr];
      worst = Max[worst, Max[Abs[N[(dbas - TSSpheroidalMatrix[k, w0, lm0, m0, rh0, rr] . bas) /. rr -> 13/10, 40]]]/
         Max[Abs[N[bas /. rr -> 13/10, 40]]]];
      If[k >= 1,
       zt = TSHomogeneousToroidal[k, w0, m0, rh0, rr, kind];
       worst = Max[worst, Max[Abs[N[(D[zt, rr] - TSToroidalMatrix[k, w0, m0, rh0, rr] . zt) /. rr -> 13/10, 40]]]/
          Max[Abs[N[zt /. rr -> 13/10, 40]]]]]], {k, 0, 4}, {kind, {"j", "y", "h1", "h2"}}];
   Print["  [5] homogeneous basis (j, y, h1, h2; n = 0..4) satisfies dy/dr = A y and dz/dr = B z: ", sci[worst], " -> ",
     chk[worst < 10^-30]];

   (* [6] m-independence: exact 3-D fields of the potentials with Y = Y_3^2, differentiated in full spherical
      coordinates, give states (U, V, R, S) and (W, T) that satisfy the SAME systems as m = 0 *)
   Module[{th = 7/10, ph = 3/10, kP, kS, rr, t, f, Y, gradP, fieldP, fieldS, fieldT, strain, state, err = 0, k = 3, m = 2},
     kP = w0 Sqrt[rh0/(lm0 + 2 m0)]; kS = w0 Sqrt[rh0/m0];
     Y = SphericalHarmonicY[k, m, t, f];
     (* strains and the traction on the sphere, for a displacement {u_r, u_t, u_f} *)
     strain[u_] := Module[{ur = u[[1]], ut = u[[2]], uf = u[[3]], err0, ett, eff, ert, erf, div},
       err0 = D[ur, rr]; ett = D[ut, t]/rr + ur/rr; eff = D[uf, f]/(rr Sin[t]) + ur/rr + ut Cot[t]/rr;
       ert = (D[ur, t]/rr + D[ut, rr] - ut/rr)/2; erf = (D[ur, f]/(rr Sin[t]) + D[uf, rr] - uf/rr)/2;
       div = err0 + ett + eff;
       {lm0 div + 2 m0 err0, 2 m0 ert, 2 m0 erf}];
     (* spheroidal state: u_r = U Y, u_t = V dY/dt, u_f = V dY/df / sin t; t_r = R Y, t_t = S dY/dt *)
     state[u_] := Module[{tr = strain[u]}, {u[[1]]/Y, u[[2]]/D[Y, t], tr[[1]]/Y, tr[[2]]/D[Y, t]}];
     fieldP = Module[{g = SphericalBesselJ[k, kP rr] Y}, {D[g, rr], D[g, t]/rr, D[g, f]/(rr Sin[t])}];
     fieldS = Module[{z = SphericalBesselJ[k, kS rr], d},
       d = D[rr z, rr]/rr; {k (k + 1) z/rr Y, d D[Y, t], d D[Y, f]/Sin[t]}];
     Do[Module[{s = state[fu], sv},
        (* the phi-components must carry the same V and S *)
        err = Max[err, Abs[N[(fu[[3]]/(D[Y, f]/Sin[t]) - s[[2]]) /. {rr -> 13/10, t -> th, f -> ph}, 40]]];
        sv = s /. {t -> th, f -> ph};
        err = Max[err, Max[Abs[N[(D[sv, rr] - TSSpheroidalMatrix[k, w0, lm0, m0, rh0, rr] . sv) /. rr -> 13/10, 40]]]/
           Max[Abs[N[sv /. rr -> 13/10, 40]]]]], {fu, {fieldP, fieldS}}];
     (* toroidal: u = z (-r_hat x grad_1 Y) = {0, z dY/df / sin t, -z dY/dt}; W = u_f / (-dY/dt), T likewise *)
     fieldT = Module[{z = SphericalBesselJ[k, kS rr]}, {0, z D[Y, f]/Sin[t], -z D[Y, t]}];
     Module[{tr = strain[fieldT], zv},
       zv = {fieldT[[3]]/(-D[Y, t]), tr[[3]]/(-D[Y, t])} /. {t -> th, f -> ph};
       err = Max[err, Max[Abs[N[(D[zv, rr] - TSToroidalMatrix[k, w0, m0, rh0, rr] . zv) /. rr -> 13/10, 40]]]/
          Max[Abs[N[zv /. rr -> 13/10, 40]]]]];
     Print["  [6] m-independence: exact 3-D fields with Y_3^2 satisfy the same systems: ", sci[err], " -> ",
       chk[err < 10^-30]]];

   (* a graded sphere for the evaluation checks: the contrast falls from its core value to zero across the shell *)
   profile[x_] := With[{s = (10 - x)/5}, s^3 (10 - 15 s + 6 s^2)];
   pars = {{lm0, m0, rh0}, {lm0 + 2 10^9 profile[#] &, m0 + 1 10^9 profile[#] &, rh0 + 100 profile[#] &}};

   (* [7] the invariant through the graded medium.  Between the two REGULAR solutions (P and SV) it vanishes
      identically: both vanish at the centre, where r^2 y1.J.y2 -> 0, and it is constant.  Between a regular
      and an irregular solution (Bessel y in the core) it is nonzero, and must be the same at both radii. *)
   Module[{coreJ, coreY, nn0 = 3, w = 600, cm = {lm0 + 2 10^9, m0 + 1 10^9, rh0 + 100}, prop, i0, i1, reg0, reg1},
     coreJ = SetPrecision[N[TSHomogeneousBasis[nn0, w, Sequence @@ cm, 5, "j"], 70], 50];
     coreY = SetPrecision[N[TSHomogeneousBasis[nn0, w, Sequence @@ cm, 5, "y"], 70], 50];
     prop[y0_] := TSPropagate["spheroidal", nn0, w, pars[[2]], {5, 10}, y0];
     reg0 = 5^2 coreJ[[All, 1]] . TSInvariantMatrix[nn0] . coreJ[[All, 2]];
     reg1 = 10^2 prop[coreJ[[All, 1]]] . TSInvariantMatrix[nn0] . prop[coreJ[[All, 2]]];
     i0 = 5^2 coreJ[[All, 1]] . TSInvariantMatrix[nn0] . coreY[[All, 1]];
     i1 = 10^2 prop[coreJ[[All, 1]]] . TSInvariantMatrix[nn0] . prop[coreY[[All, 1]]];
     Print["  [7] r^2 y1.J.y2 through the graded shell (n = 3), r = 5 -> 10: regular P with irregular P ", sci[i0], " -> ",
       sci[i1], " (relative change ", sci[Abs[i1 - i0]/Abs[i0]], "); regular P with regular SV ", sci[Abs[reg0]], " -> ",
       sci[Abs[reg1]], " (relative to the scale ", sci[Abs[reg1]/Abs[i0]], ") -> ",
       chk[Abs[i1 - i0]/Abs[i0] < 10^-20 && Abs[reg1]/Abs[i0] < 10^-20]]];

   (* [8] T-matrices: uniform shell propagated = closed-form Mie; graded sphere unitary and reciprocal *)
   worst = 0; worstR = 0;
   Do[Module[{w = 600, uni, mie, gr, Wm, Sm},
      uni = TSSphereTMatrix[k, w, pars[[1]], {lm0 + 2 10^9 &, m0 + 1 10^9 &, rh0 + 100 &}, {5, 10}];
      mie = TSSphereTMatrix[k, w, pars[[1]], {lm0 + 2 10^9 &, m0 + 1 10^9 &, rh0 + 100 &}, {10, 10}];
      worst = Max[worst, Max[Abs[uni["Tpsv"] - mie["Tpsv"]]]/Max[Abs[mie["Tpsv"]]]];
      gr = TSSphereTMatrix[k, w, pars[[1]], pars[[2]], {5, 10}];
      If[k >= 1,
       (* flux normalisation: S = W (I + 2T) W^-1, W = diag(sqrt(alpha), sqrt(beta n(n+1))) *)
       Wm = DiagonalMatrix[{Sqrt[Sqrt[(lm0 + 2 m0)/rh0]], Sqrt[Sqrt[m0/rh0] k (k + 1)]}];
       Sm = Wm . (IdentityMatrix[2] + 2 gr["Tpsv"]) . Inverse[Wm];
       worstR = Max[worstR, Norm[ConjugateTranspose[Sm] . Sm - IdentityMatrix[2]], Abs[Sm[[1, 2]] - Sm[[2, 1]]],
         Abs[Abs[1 + 2 gr["Tsh"]] - 1]],
       worstR = Max[worstR, Abs[Abs[1 + 2 gr["Tpsv"][[1, 1]]] - 1]]]], {k, 0, 6}];
   Print["  [8] uniform shell propagated = closed-form homogeneous sphere (n = 0..6): ", sci[worst],
     "; graded sphere unitary and reciprocal (S = S^T): ", sci[worstR], " -> ", chk[worst < 10^-20 && worstR < 10^-20]];

   (* [9] a stepped sphere, a core and two shells (the outer one softer than the background): the propagator,
      with the jumps given as breaks, against the closed form of homogeneous shells *)
   Module[{w = 600, m1, m2, m3, pw, worst9 = 0},
     m1 = {lm0 + 2 10^9, m0 + 1 10^9, rh0 + 100}; m2 = {lm0 + 10^9, m0 + 5 10^8, rh0 + 50};
     m3 = {lm0 - 10^9, m0 - 5 10^8, rh0 - 50};
     pw[i_] := Function[x, Piecewise[{{m1[[i]], x < 5}, {m2[[i]], x < 15/2}}, m3[[i]]]];
     Do[Module[{t1, t2},
        t1 = TSSphereTMatrix[k, w, pars[[1]], {pw[1], pw[2], pw[3]}, {5, 10}, "Breaks" -> {15/2}];
        t2 = TSShellsTMatrix[k, w, pars[[1]], {5, 15/2, 10}, {m1, m2, m3}];
        worst9 = Max[worst9, Max[Abs[t1["Tpsv"] - t2["Tpsv"]]]/Max[Abs[t2["Tpsv"]]],
          If[k >= 1, Abs[t1["Tsh"] - t2["Tsh"]]/Abs[t2["Tsh"]], 0]]], {k, 0, 5}];
     Print["  [9] stepped sphere (core + two shells, n = 0..5): propagator with breaks = closed-form shells: ",
       sci[worst9], " -> ", chk[worst9 < 10^-20]]];

   Print["==== TakeuchiSaito: ", If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"],
     " ===="];
   And @@ oks],
  {N::meprec, Solve::svars}];

If[Length[$ScriptCommandLine] > 0 && StringEndsQ[First[$ScriptCommandLine], "TakeuchiSaito.wl"], TSSelfTest[]];
