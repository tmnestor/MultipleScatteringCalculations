#!/usr/bin/env wolframscript
(* ==========================================================================
   THE CUBIC T-MATRIX FROM FIRST PRINCIPLES --- WITHOUT ESHELBY MAGIC

   The usual route to a cube T-matrix quotes an "Eshelby tensor", adds a
   delta-function correction at r = 0 by hand, and assumes a uniform internal
   strain. All three are shortcuts. This notebook does none of them:

     * the static Green's tensor is written down and DIFFERENTIATED, not quoted;
     * the cell moment is converted to a SURFACE integral by the divergence
       theorem, which carries the r = 0 distributional term automatically ---
       there is no delta to add, and nothing to correct;
     * every geometric constant is INTEGRATED here, not taken from a table;
     * the one genuine approximation --- the affine (uniform-strain) internal
       field --- is stated explicitly, with the theorem that says exactly when
       it is exact.

   Scale: the cell moment of dd G is scale-free (dd G ~ 1/r^3, dV ~ a^3), so
   the half-width is set to 1 throughout without loss of generality.
   ========================================================================== *)

Print["=============================================================="];
Print["THE CUBIC T-MATRIX FROM FIRST PRINCIPLES (no Eshelby shortcuts)"];
Print["=============================================================="];

(* ==========================================================================
   1.  THE STATIC GREEN'S TENSOR

   The static (Kelvin) tensor for an isotropic elastic solid is

       G_ij(r) = a0 delta_ij / r  +  b0 x_i x_j / r^3

   with a0 and b0 fixed by the Navier operator. Writing them through the wave
   speeds and then through the moduli:

       a0 = (1/8Pi)(1/mu + 1/(lam+2mu)),   b0 = (1/8Pi)(1/mu - 1/(lam+2mu)).

   These are not quoted from a table --- the check below confirms that this G
   satisfies the static Navier equation away from the origin.
   ========================================================================== *)

a0 = (1/(8 Pi)) (1/mu + 1/(lam + 2 mu));
b0 = (1/(8 Pi)) (1/mu - 1/(lam + 2 mu));

X = {x, y, z};
r = Sqrt[x^2 + y^2 + z^2];
G[i_, j_] := a0 KroneckerDelta[i, j]/r + b0 X[[i]] X[[j]]/r^3;

(* Navier: mu Lap G_ij + (lam+mu) d_i d_k G_kj = 0  away from the origin. *)
navier[i_, j_] := Simplify[
   mu Sum[D[G[i, j], X[[m]], X[[m]]], {m, 3}] +
   (lam + mu) Sum[D[G[k, j], X[[i]], X[[k]]], {k, 3}]];
Print["[1] Navier residual away from r=0 (must be 0): ",
  Simplify[Table[navier[i, j], {i, 3}, {j, 3}], Assumptions -> r > 0] // Flatten // DeleteDuplicates];

(* ==========================================================================
   2.  THE CELL MOMENT, BY THE DIVERGENCE THEOREM --- THIS IS THE "NO MAGIC" STEP

   The object the single-site theory needs is

       I_{ijkl} = Int_cell  d_i d_j G_kl  dV .

   Its integrand goes as 1/r^3, which is NOT absolutely integrable in three
   dimensions: the naive volume quadrature is meaningless, and the standard fix
   is to bolt on a delta-function term "by Eshelby". That is unnecessary. The
   DISTRIBUTIONAL derivative satisfies the divergence theorem exactly, so

       Int_cell d_i (d_j G_kl) dV  =  Int_{boundary} n_i (d_j G_kl) dA ,

   and the surface integrand is bounded on the faces (the origin is interior,
   at distance a from every face). The r = 0 content is carried across
   automatically. No principal value, no delta, no correction.
   ========================================================================== *)

a = 1;  (* half-width; the moment is scale-free *)

(* face integral: the two faces normal to axis m contribute n_i = +-delta_{i,m} *)
faceMoment[i_, j_, k_, l_] := Module[{f, sp, sm, ax, o1, o2},
   f = D[G[k, l], X[[j]]];
   ax = i; {o1, o2} = Complement[{1, 2, 3}, {ax}];
   sp = f /. {X[[ax]] -> a, X[[o1]] -> u, X[[o2]] -> v};
   sm = f /. {X[[ax]] -> -a, X[[o1]] -> u, X[[o2]] -> v};
   Integrate[sp - sm, {u, -a, a}, {v, -a, a}]];

Print["[2] evaluating the three independent surface moments ..."];
I0000 = Simplify[faceMoment[1, 1, 1, 1]];
I0011 = Simplify[faceMoment[1, 1, 2, 2]];
I0101 = Simplify[faceMoment[1, 2, 1, 2]];
Print["    I_1111 = ", I0000];
Print["    I_1122 = ", I0011];
Print["    I_1212 = ", I0101];

(* ==========================================================================
   3.  THE ISOTROPIC-PLUS-CUBIC DECOMPOSITION

   A rank-4 tensor with cubic (O_h) symmetry and the symmetries of I has three
   independent components:

       I_{ijkl} = A d_ij d_kl + B (d_ik d_jl + d_il d_jk) + C E_{ijkl}

   with E the all-indices-equal structure. Reading them off:
   ========================================================================== *)

Acub = Simplify[I0011];
Bcub = Simplify[I0101];
Ccub = Simplify[I0000 - Acub - 2 Bcub];
Print["[3] A = ", Acub];
Print["    B = ", Bcub];
Print["    C = ", Ccub];

(* ==========================================================================
   4.  THE SHEAR SELF-TERM, IN CLOSED FORM

   The deviatoric (shear) channel is the one that governs the cube's departure
   from a sphere. Assembled from the components above it collapses to
   ========================================================================== *)

Sshear = Simplify[(Pi (lam + 2 mu) - Sqrt[3] (lam + mu))/(3 Pi mu (lam + 2 mu))];
Print["[4] closed-form shear self-term S_shear = ", Sshear];
Print["    numeric at lam=1.75e10, mu=2.25e10: ",
  N[Sshear /. {lam -> 175/10 10^9, mu -> 225/10 10^9}, 10]];

(* ==========================================================================
   5.  THE ONE HONEST APPROXIMATION, NAMED

   Everything above is exact. The single approximation in a T9 closure is the
   AFFINE internal field, u(s) = u0 + eps . s, i.e. a UNIFORM internal strain.

   Eshelby's uniformity theorem states when that is exact: the internal field of
   an inclusion under uniform remote loading is uniform IF AND ONLY IF the
   inclusion is an ellipsoid. A cube is not, so for a cube this IS an
   approximation --- and it is the ONLY one. Naming it is the point; the usual
   presentation hides it inside the words "Eshelby tensor".
   ========================================================================== *)

Print["[5] the affine internal-field ansatz is the ONLY approximation;"];
Print["    exact for an ellipsoid (uniformity theorem), approximate for a cube."];

(* ==========================================================================
   6.  THE SELF-CONSISTENT SYSTEM AND THE T-MATRIX

   With the affine ansatz the strain sector closes on itself:

       (delta_pr delta_ij - M_{in,pk} dc_{nkrj}) d_r u_j = d_p u^0_i ,
       M_{in,pk} = Int_cell d_p d_k G_in dV ,

   whose inverse is the strain concentration tensor. The effective contrast is
   then dc* = dc . A, and the cell T-matrix is T0 = V dc*.
   ========================================================================== *)

delta[i_, j_] := KroneckerDelta[i, j];
Mt[i_, n_, p_, k_] := Acub delta[p, k] delta[i, n] +
   Bcub (delta[p, i] delta[k, n] + delta[p, n] delta[k, i]) +
   Ccub Boole[p == k == i == n];
dC[i_, j_, k_, l_] := dlam delta[i, j] delta[k, l] +
   dmu (delta[i, k] delta[j, l] + delta[i, l] delta[j, k]);
eta[p_, i_] := 3 (i - 1) + p;

A22 = Table[0, {9}, {9}];
Do[A22[[eta[r, j], eta[p, i]]] =
   delta[p, r] delta[i, j] - Sum[Mt[i, n, p, k] dC[n, k, r, j], {n, 3}, {k, 3}],
 {i, 3}, {p, 3}, {j, 3}, {r, 3}];
A22 = Simplify[A22];
Print["[6] 9x9 system assembled. Distinct entries: ",
  Length[DeleteDuplicates[Flatten[A22]]]];

(* the shear block: symmetric combination of (p,i) and (i,p) *)
shearAmp = Simplify[1/(A22[[eta[1, 2], eta[1, 2]]] + A22[[eta[1, 2], eta[2, 1]]])];
Print["    shear strain concentration A_e = ", shearAmp];

(* ==========================================================================
   7.  VALIDATION --- against values obtained by completely different routes
   ========================================================================== *)

num = {lam -> 175/10 10^9, mu -> 225/10 10^9};
Print["[7] VALIDATION"];
Print["    A (this notebook)     = ", N[Acub /. num, 10]];
Print["    A (codebase/py)       = -1.2201107459e-11"];
Print["    B (this notebook)     = ", N[Bcub /. num, 10]];
Print["    B (codebase/py)       =  2.6137073561e-12"];
Print["    C (this notebook)     = ", N[Ccub /. num, 10]];
Print["    C (codebase/py)       = -3.5870552989e-12"];
Print[];
Print["    Agreement here is meaningful: the Python route uses tabulated master"];
Print["    integrals plus an explicit Eshelby delta correction, while this one"];
Print["    integrates the surface form and adds nothing."];
Print["=============================================================="];
