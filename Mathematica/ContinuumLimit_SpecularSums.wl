#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_SpecularSums.wl  --  notebook 2 of the continuum-limit study.

   WHY.  At normal incidence on a laterally uniform layer the discrete Foldy-Lax
   system is EXACTLY a 1-D chain of planes coupled by the specular lattice sums
   of the point Green's tensor,
       S(m) = sum_R G(R + m d z^),     square lattice of pitch d,
   m the plane separation in pitches.  By Poisson summation
       S(m) = (1/d^2) sum_G G^(G; m d),   G = (2 pi / d)(p, q),
   so S(m) is the continuum's plane-wave term G^(0; m d)/d^2 (notebook 1) PLUS
   evanescent diffraction orders.  Their exponential factor at fixed m is
   e^{-2 pi m sqrt(p^2+q^2)}, independent of d, while the strain blocks carry
   powers of |G| ~ 1/d: the evanescent part is a d-INDEPENDENT local term once
   multiplied by a cube volume d^3.  That term is what the continuum limit turns
   on, and this notebook computes it.

   THE SPECTRAL KERNEL, derived here from the Kupradze tensor and the Weyl
   identity:
       G^_ij = (i / 2 rho w^2) [ (kS^2 d_ij - kS_i kS_j) e^{i gS |z|}/gS
                                + kP_i kP_j e^{i gP |z|}/gP ],
       kW = (k_x, k_y, gW sign z), gW = sqrt(kW^2 - k_x^2 - k_y^2), Im gW >= 0.
   9-component rows: u, then strain (e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx) by the
   receiver derivative i kW_a.  Columns: force, then the moment source by the
   SOURCE derivative -i kW_b (d' = -d), times a sign PINNED against the package,
   not assumed: it comes out -1, i.e. the package's moment columns are +d G.

   CHECKS (against scripts/continuum_limit_specular.py's JSON):
     [1] the spectral kernel, all 81 entries, propagating and evanescent;
     [2] the lattice sums S(m), m != 0, at three pitches, against the package's
         exact Ewald point-propagator kernel;
     [3] the split of S(m) into the continuum term and the evanescent remainder,
         per block, and how each scales with the pitch.
   Coordinates (z, x, y), z down; time e^{-i w t}; SI units.
   ============================================================================ *)

ref = Import[DirectoryName[$InputFileName] <> "ContinuumLimit_specular.json", "RawJSON"];
cplx[{re_, im_}] := re + I im;
mat[g_] := Map[cplx, g, {2}];
om = ref["omega"]; al = ref["alpha"]; be = ref["beta"]; rho = ref["rho"];
kP = om/al; kS = om/be;
sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
relMax[a_, b_] := Max[Abs[Flatten[a - b]]]/Max[Abs[Flatten[b]]];

(* vertical wavenumber, branch Im >= 0 (outgoing / decaying) *)
gam[k_, kx_, ky_] := Module[{g = Sqrt[k^2 - kx^2 - ky^2 + 0 I]}, If[Im[g] < 0, -g, g]];

(* Voigt order of the 9-component state, as (z, x, y) = (1, 2, 3) index pairs *)
voigt = {{1, 1}, {2, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};  (* e_zz, e_xx, e_yy, 2e_xy, 2e_zy, 2e_zx *)
engineering = {1, 1, 1, 2, 2, 2};                          (* rows: shear strains carry the factor 2 *)

(* the 9x9 spectral kernel; mom: the sign of the moment columns (pinned by [1]) *)
spectral9[kx_, ky_, z_, mom_] := Module[{s = Sign[z], parts, u, full},
  parts = Table[
    Module[{k = w[[1]], g, kv, amp, uu},
     g = gam[k, kx, ky];
     kv = {g s, kx, ky};                     (* (z, x, y) *)
     amp = I/(2 rho om^2) Exp[I g Abs[z]]/g;
     uu = If[w[[2]] == "P", amp Outer[Times, kv, kv], amp (k^2 IdentityMatrix[3] - Outer[Times, kv, kv])];
     {kv, uu}],
    {w, {{kP, "P"}, {kS, "S"}}}];
  full = Sum[
    Module[{kv = p[[1]], uu = p[[2]], rows, block},
     (* rows: u_i, then strain e_ab = (1/2)(d_a u_b + d_b u_a), times engineering *)
     rows[col_] := Join[col, Table[engineering[[v]] (1/2) (I kv[[voigt[[v, 1]]]] col[[voigt[[v, 2]]]] +
            I kv[[voigt[[v, 2]]]] col[[voigt[[v, 1]]]]), {v, 6}]];
     (* columns: force j, then moment (a,b): the source derivative -i k_b applied to force column a,
        symmetrised, times the pinned sign mom *)
     block = Join[
       Table[rows[uu[[All, j]]], {j, 3}],
       Table[mom (1/2) rows[(-I kv[[voigt[[v, 2]]]]) uu[[All, voigt[[v, 1]]]] +
           (-I kv[[voigt[[v, 1]]]]) uu[[All, voigt[[v, 2]]]]], {v, 6}]];
     Transpose[block]],
    {p, parts}];
  full];

Print["==== ContinuumLimit_SpecularSums :: the specular lattice sums of the point Green's tensor ===="];
Print["  omega = ", om, "; alpha, beta, rho = ", {al, be, rho}, "; k_P = ", kP, ", k_S = ", kS];

(* ---------------------------------------------------------------------------
   [1] the spectral kernel: pin the moment-column shear factor, then compare all entries
   --------------------------------------------------------------------------- *)
candidates = {1, -1};  (* d' = -d gives +1; the package's moment columns may carry the opposite sign *)
errs = Table[Max[Table[relMax[spectral9[e["kx"], e["ky"], e["dz"], c], mat[e["G"]]], {e, ref["spectral"]}]],
   {c, candidates}];
best = candidates[[First@Ordering[errs, 1]]];
Print["  [1] spectral kernel, all 81 entries over 12 (k_x, k_y, dz), moment-shear factor candidates ",
  candidates, ": worst ", sci /@ errs];
Print["      pinned moment-column sign ", best, " -> ", pass[Min[errs] < 10^-8]];

(* ---------------------------------------------------------------------------
   [2] the specular lattice sums, m != 0: Poisson sum over the reciprocal lattice
   --------------------------------------------------------------------------- *)
pMax = 6;
latticeSum[d_, m_] := (1/d^2) Sum[spectral9[2 Pi p/d, 2 Pi q/d, m d, best], {p, -pMax, pMax}, {q, -pMax, pMax}];
cont[d_, m_] := (1/d^2) spectral9[0, 0, m d, best];
res2 = Table[
   If[e["m"] == 0, Nothing, {e["d"], e["m"], relMax[latticeSum[e["d"], e["m"]], mat[e["S"]]]}],
   {e, ref["lattice"]}];
worst2 = Max[res2[[All, 3]]];
Print["  [2] S(m), m != 0, Poisson sum (|p|,|q| <= ", pMax, ") vs the package's Ewald point kernel:"];
Do[Print["      d = ", r[[1]], "  m = ", r[[2]], ":  ", sci[r[[3]]]], {r, Select[res2, #[[2]] > 0 &]}];
Print["      worst over m = -4..4, three pitches: ", sci[worst2], " -> ", pass[worst2 < 10^-6]];

(* ---------------------------------------------------------------------------
   [3] continuum term + evanescent remainder: scaling with the pitch
   The remainder E(m) = S(m) - cont(m).  If it is a d-independent local term
   after multiplication by the cube volume d^3, E(m) d^3 is the same at every d.
   --------------------------------------------------------------------------- *)
blocks = {{"u<-f", 1 ;; 3, 1 ;; 3}, {"u<-M", 1 ;; 3, 4 ;; 9}, {"e<-f", 4 ;; 9, 1 ;; 3}, {"e<-M", 4 ;; 9, 4 ;; 9}};
Print["  [3] evanescent remainder E(m) = S(m) - (continuum term), max |entry| x d^3, by block:"];
Print["      (a d-INDEPENDENT column marks a scale-invariant local term)"];
Do[
  Print["      m = ", m, ":"];
  Do[
   Print["        ", bl[[1]], "   ",
    Row[Table["d=" <> ToString[d] <> ": " <> sci[Max[Abs[Flatten[(latticeSum[d, m] - cont[d, m])[[bl[[2]], bl[[3]]]]]]] d^3],
      {d, {0.5, 1.0, 2.0}}], "   "],
    "    continuum term x d^3 at d=1: ", sci[Max[Abs[Flatten[cont[1.0, m][[bl[[2]], bl[[3]]]]]]]]],
   {bl, blocks}],
  {m, {1, 2}}];

Print["==== ContinuumLimit_SpecularSums (stage 2a): ",
  If[Min[errs] < 10^-8 && worst2 < 10^-6, "ALL CHECKS PASS", "CHECKS FAILED"], " ===="];
