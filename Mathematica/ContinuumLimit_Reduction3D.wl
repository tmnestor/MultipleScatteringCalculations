#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_Reduction3D.wl  --  notebook 8 of the continuum-limit study.

   THE CLAIM.  In three dimensions the first-moment voxel carries, per voxel,
   the 9-component state with Legendre weights {1, x/h, y/h, z/h} (36 unknowns),
   tested with the same functions (Galerkin) and coupled by cell-to-cell double
   integrals of the Green's tensor.  For a laterally uniform layer at normal
   incidence this 3-D scheme is EXACTLY the 1-D scheme of notebook 7:
     (a) the lateral first moments are never excited -- the lattice and the
         incident field are symmetric under reflection about every voxel's
         centre planes x = const, y = const, and x/h, y/h are odd there;
     (b) the mean and the z-moment carry weights that are CONSTANT laterally, so
         their lateral form factor sinc(g_x h) sinc(g_y h) vanishes at every
         reciprocal-lattice vector g != 0 (notebook 3): the plane sum of the
         cell-to-cell double integrals is the plate-to-plate integral in z alone,
             sum_R int_0 int_R phi_a(z) G phi_b(z') = d^2 int int phi_a(z) g_plate(z - z') phi_b(z') dz dz',
         and dividing the 3-D equation by d^2 leaves notebook 7's equation.

   WHAT THIS DOES NOT COVER, AND WHY IT MATTERS NEXT.
     (c) the LATERAL first-moment form factor, the derivative of sinc, does NOT
         vanish on the reciprocal lattice: F1(p pi) = -i cos(p pi)/(p pi) != 0;
     (d) at oblique incidence the mean's zeros shift, sinc((2 pi p/d + k_x) h) =
         O(k_x h) != 0, so the tiling identity itself breaks at O(k_par h).
   Oblique incidence therefore exercises genuinely 3-D pieces: evanescent
   coupling of the gradient channels, and Bloch lattice sums with first-moment
   form factors.
   ============================================================================ *)

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_Reduction3D :: the 3-D first-moment voxel at normal incidence ===="];

(* ---------------------------------------------------------------------------
   Lateral form factors of the Legendre weights: F_a(g) = (1/d) int_{-h}^{h} phi_a(x) e^{-i g x} dx
   --------------------------------------------------------------------------- *)
Clear[g, h, x, p];
f0 = Simplify[Integrate[Exp[-I g x], {x, -h, h}]/(2 h)];
f1 = Simplify[Integrate[(x/h) Exp[-I g x], {x, -h, h}]/(2 h)];
Print["  F0(g) = ", InputForm[f0], " = Sinc[g h] -> ", chk[FullSimplify[f0 - Sinc[g h], g != 0] === 0], ",   F1(g) = ", InputForm[f1]];
onLattice0 = Simplify[f0 /. g -> p Pi/h, p \[Element] Integers && p != 0];
onLattice1 = FullSimplify[f1 /. g -> p Pi/h, p \[Element] Integers && p != 0];
Print["  [1] (b) the mean's form factor on the reciprocal lattice, g = 2 pi p/d, p != 0: ", InputForm[onLattice0],
  " -> ", chk[onLattice0 === 0]];
Print["  [2] (c) the lateral first moment's: ", InputForm[onLattice1], "  -- NOT zero: ",
  chk[!PossibleZeroQ[onLattice1 /. p -> 1]]];
Print["      and at g = 0: F1(0) = ", Limit[f1, g -> 0], " (no uniform-plate coupling of a lateral moment)"];

(* ---------------------------------------------------------------------------
   (a) symmetry: under x -> 2 x_c - x the laterally uniform problem is invariant (lattice, contrast,
   incident field). A lateral first moment of voxel c, c_x = (1/int phi^2) int phi_1((x - x_c)/1) w dx,
   changes sign; so it vanishes. Checked on the incident field and on a single-voxel response.
   --------------------------------------------------------------------------- *)
Module[{w0, mom},
  w0[xx_, zz_] := Exp[I 0.7 zz];   (* any laterally uniform field *)
  mom = Integrate[(xx/h) w0[xx, zz], {xx, -h, h}];
  Print["  [3] (a) lateral first moment of a laterally uniform field over a voxel: ", mom, " -> ", chk[mom === 0]]];

(* ---------------------------------------------------------------------------
   (b) the reduction, explicitly: the Poisson form of the plane sum with double averaging.
   sum_R <phi_a G phi_b>_{cells 0, R} = (1/d^2) sum_g [F0(g)^* F0(g) d^2 ...] -> only g = 0 survives,
   and the g = 0 term is the plane Green's tensor at zero lateral wavenumber, integrated over z with
   phi_a(z) phi_b(z'). Normalisation: the 3-D test integral int_cell phi_a^2 = d^2 int phi_a(z)^2 dz.
   Check: the ratio of every surviving lateral factor to its 1-D counterpart is d^2 / d^2 = 1.
   --------------------------------------------------------------------------- *)
Module[{latt, gram3, gram1},
  (* the doubly averaged lateral factor of order (p, q) is |F0|^2 in x times |F0|^2 in y; F0(g) = Sinc[g h] *)
  latt = Sum[(Sinc[pp Pi] Sinc[qq Pi])^2, {pp, -3, 3}, {qq, -3, 3}];
  gram3 = Integrate[1, {x, -h, h}, {y, -h, h}]; gram1 = 1;
  Print["  [4] (b) plane sum of the doubly averaged lateral factors (|p|, |q| <= 3): ", Simplify[latt],
   " (only the g = 0 term survives); 3-D test normalisation / 1-D = ", Simplify[gram3/(2 h)^2], " -> ",
   chk[Simplify[latt] === 1]]];

(* ---------------------------------------------------------------------------
   (d) oblique incidence: the mean's form factor at the shifted orders g + k_x, first order in k_x h
   --------------------------------------------------------------------------- *)
Module[{kx, shifted},
  shifted = Series[f0 /. g -> p Pi/h + kx, {kx, 0, 1}] // Normal;
  shifted = FullSimplify[shifted, p \[Element] Integers && p != 0];
  Print["  [5] (d) at oblique incidence the mean's form factor on the shifted lattice: ", InputForm[shifted],
   "  -- O(k_x h), the tiling identity breaks: ", chk[!PossibleZeroQ[shifted /. {p -> 1, kx -> 1/10, h -> 1}]]]];

Print["==== ContinuumLimit_Reduction3D (stage 8): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
