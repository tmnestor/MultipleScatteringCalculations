#!/usr/bin/env wolframscript
(* ::Package:: *)

(* =====================================================================
   ToroidalIncidentExpansion.wl

   The relative weight of the M-type (toroidal) family in an x-polarised S
   plane wave, in the potential conventions of MieSphericalWaves.wl Section 2.

   WHY THIS EXISTS

   The m = +/-1 plane-wave spectrum of the scattered S field needs both shear
   families.  The N-type (SV) part is fixed by the gate's renormalisation
   renorm(n) = -1/[n(n+1)] applied to the potential coefficient (2n+1) i^n/(i k).
   The M-type part is driven by the Mie SH coefficient c_n, which is normalised
   against its OWN incident expansion.  Joining the two needs the ratio of the
   M-type to the N-type incident coefficient, and that ratio is what this file
   determines.  It had previously been recorded as bounded by a scan, never
   determined.

   THE CLAIM

       xhat Exp[I k z] = Sum_n  alpha_n Curl[Curl[r psi_c]] + beta_n Curl[r psi_s]

       psi_c = j_n(k r) P_n^1(cos theta) cos(phi)
       psi_s = j_n(k r) P_n^1(cos theta) sin(phi)      (Condon-Shortley)

       alpha_n = -(2n+1) I^n / (I k n(n+1))            (the gate's coeff renorm)
       beta_n  =  I k alpha_n

   so the scattered M-type coefficient is (c_n / coeff_n) beta_n
   = I k c_n renorm(n).  Two features are the content: the M-type carries the
   SIN parity (an x-polarised wave's toroidal part is odd in phi), and the
   factor is I k, not 1 and not its reciprocal -- one curl fewer costs one
   power of k.

   THE CHECK

   The truncated sum is evaluated from the exact Cartesian families and
   compared against the plane wave at three off-axis points.  The wrong sign,
   beta_n = -I k alpha_n, is evaluated as the control: it must fail at O(1),
   or the check would not be discriminating.

   The independent Python checks are a least-squares FIT of the coefficients
   by finite-difference curls, and the optical theorem for S incidence on the
   gate's lossless sphere (scripts/gate_sphere_vs_impedance_march.py, part 9).

   Run:
       wolframscript -file Mathematica/ToroidalIncidentExpansion.wl
   ===================================================================== *)

k = 13/10; nmax = 22;
rX[x_, y_, z_] := Sqrt[x^2 + y^2 + z^2];
angC[n_][x_, y_, z_] := -(D[LegendreP[n, u], u] /. u -> z/rX[x, y, z]) x/rX[x, y, z];
angS[n_][x_, y_, z_] := -(D[LegendreP[n, u], u] /. u -> z/rX[x, y, z]) y/rX[x, y, z];
jr[n_][x_, y_, z_] := SphericalBesselJ[n, k rX[x, y, z]];

nType[n_] := nType[n] =
  Curl[Curl[{x, y, z} jr[n][x, y, z] angC[n][x, y, z], {x, y, z}], {x, y, z}];
mType[n_] := mType[n] = Curl[{x, y, z} jr[n][x, y, z] angS[n][x, y, z], {x, y, z}];
alpha[n_] := -(2 n + 1) I^n/(I k n (n + 1));

resid[sgn_, p_] := Module[{sum},
  sum = Sum[alpha[n] nType[n] + sgn I k alpha[n] mType[n], {n, 1, nmax}];
  N[Norm[(sum - {Exp[I k z], 0, 0}) /. Thread[{x, y, z} -> p]], 20]];

pts = {{3/10, -1/5, 7/10}, {-1/2, 4/5, -3/10}, {1/10, 1/10, -9/10}};
good = resid[+1, #] & /@ pts;
bad = resid[-1, #] & /@ pts;
(* ScientificForm renders as two-line text under wolframscript; FortranForm
   gives one line, 2.12e-26. *)
fmt[v_] := ToString[N[v, 3], FortranForm];
Print["beta = +I k alpha : worst residual ", fmt[Max[good]]];
Print["beta = -I k alpha : best residual  ", fmt[Min[bad]], "   (control, must be O(1))"];

ok = Max[good] < 10^-20 && Min[bad] > 10^-1;
Print[If[ok, "PASS", "FAIL"], ": the toroidal incident coefficient is beta_n = I k alpha_n"];
Exit[If[ok, 0, 1]];
