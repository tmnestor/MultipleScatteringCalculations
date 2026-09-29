#!/usr/bin/env wolframscript
(* ============================================================================
   ContinuumLimit_GradedSphere.wl  --  notebook 16 of the continuum-limit study.

   WHY.  A voxelised sphere with a sharp surface is dominated by the staircase
   error of its shape (section 10).  A sphere whose contrast falls SMOOTHLY to
   zero at its surface has no staircase: the voxels that cross the surface carry
   a vanishing contrast.  This notebook builds its exact scattering, the
   reference against which the voxel scheme's own error can be measured.

   THE BODY.  A homogeneous core r < b carrying the contrast Delta_0, and a shell
   b < r < a across which the contrast falls to zero with the C2 smoothstep
       f(r) = s(x),  s(x) = 10 x^3 - 15 x^4 + 6 x^5,  x = (a - r)/(a - b),
   so the contrast vanishes like (a - r)^3 at the surface.  In the core the
   regular solutions are the homogeneous Mie interior functions, exactly.

   THE METHOD.  Everything radial comes from the library TakeuchiSaito.wl (in
   this folder; see its header and TSSelfTest[]): the Takeuchi-Saito systems,
   derived from the equations of motion, the homogeneous basis, the propagator
   and the per-order T-matrix TSSphereTMatrix, in the convention of
   MieTmatrixReference.wl (incident potential coefficients to scattered ones);
   and the closed form of a sphere of homogeneous shells, TSShellsTMatrix.
   This notebook defines the body and the checks specific to it.

   CHECKS: [1] the library's own self-test (derivation, basis, invariant,
   propagator, shells); [2] a uniform shell reproduces MieTmatrixReference (the
   package's arbiter); [3] a two-shell stepped sphere with the benchmark's
   contrast: the propagator with a break equals the closed form; [4] the graded
   sphere's S-matrix is unitary per order; [5] 90 vs 120 digits; [6] N
   homogeneous shells (midpoint contrast) converge to the graded result at
   second order in 1/N.
   Time e^{-i w t}; SI units.
   ============================================================================ *)

Get[FileNameJoin[{DirectoryName[$InputFileName], "TakeuchiSaito.wl"}]];

sci[x_] := ToString[NumberForm[N[x], 3, NumberFormat -> (If[#3 == "", #1, Row[{#1, "e", #3}]] &)], OutputForm];
pass[b_] := If[TrueQ[b], "PASS", "FAIL"];
oks = {};
chk[b_] := (AppendTo[oks, TrueQ[b]]; pass[b]);
Print["==== ContinuumLimit_GradedSphere :: a sphere whose contrast falls smoothly to zero at its surface ===="];

(* [1] the library *)
Print["  [1] the Takeuchi-Saito library's self-test:"];
Print["  [1] -> ", chk[TSSelfTest[]]];

(* the background and a contrast profile: moduli and density as functions of r *)
background[al_, be_, rh_] := {rh (al^2 - 2 be^2), rh be^2, rh};
material[{lam0_, mu0_, rho0_}, {dl_, dm_, dr_}, f_] :=
  {Function[x, lam0 + dl f[x]], Function[x, mu0 + dm f[x]], Function[x, rho0 + dr f[x]]};

(* ---------------------------------------------------------------------------
   [2] a uniform shell is the homogeneous sphere of MieTmatrixReference.wl
   --------------------------------------------------------------------------- *)
refJ = Import[FileNameJoin[{DirectoryName[$InputFileName], "MieTmatrixReference.json"}], "RawJSON"];
Module[{g = refJ["tmatrices"]["gate"], s = 11/10, om0 = 60, a = 120, bg, worst = 0, cpx, mat},
  bg = background[5000, 3000, 2500];
  (* the reference scales alpha, beta, rho by s inside: moduli by s^3 *)
  mat = material[bg, {(s^3 - 1) bg[[1]], (s^3 - 1) bg[[2]], (s - 1) bg[[3]]}, 1 &];
  cpx[v_] := v[[1]] + I v[[2]];
  Do[Module[{mine = TSSphereTMatrix[n, om0, bg, mat, {a/2, a}, WorkingPrecision -> 60], want = g[[n + 1]]},
     worst = Max[worst, Max[Abs[mine["Tpsv"] - Map[cpx, want["Tpsv"], {2}]]]/Max[Abs[Map[cpx, want["Tpsv"], {2}]]],
       If[n > 0, Abs[mine["Tsh"] - cpx[want["Tsh"]]]/Abs[cpx[want["Tsh"]]], 0]]], {n, 0, 12}];
  Print["  [2] uniform shell (integrated across b = a/2 .. a) vs MieTmatrixReference 'gate', n = 0..12: ", sci[worst],
   " -> ", chk[worst < 10^-17]]];

(* ---------------------------------------------------------------------------
   the benchmark: gate contrast, radius 10 m, core b = a/2, smoothstep shell
   --------------------------------------------------------------------------- *)
smooth[x_] := 10 x^3 - 15 x^4 + 6 x^5;
aB = 10; bB = 5;
bgB = background[5000, 3000, 2500];
dB = {2 10^9, 1 10^9, 100};
profG = Function[x, Piecewise[{{1, x < bB}}, smooth[(aB - x)/(aB - bB)]]];
matG = material[bgB, dB, profG];
omAt[kSa_] := kSa 3000/aB;

(* [3] two-shell stepped sphere with the benchmark's contrast: propagator with a break vs the closed form *)
Module[{om0 = omAt[1], fs = {1, 3/4, 1/4}, rs = {bB, 15/2, aB}, step, worst = 0},
  step = Function[x, Piecewise[{{1, x < bB}, {3/4, x < 15/2}}, 1/4]];
  Do[Module[{ode, cf},
     ode = TSSphereTMatrix[n, om0, bgB, material[bgB, dB, step], {bB, aB}, "Breaks" -> {15/2}, WorkingPrecision -> 60];
     cf = TSShellsTMatrix[n, om0, bgB, rs, Table[bgB + f dB, {f, fs}], WorkingPrecision -> 60];
     worst = Max[worst, Max[Abs[ode["Tpsv"] - cf["Tpsv"]]]/Max[Abs[cf["Tpsv"]]],
       If[n > 0, Abs[ode["Tsh"] - cf["Tsh"]]/Abs[cf["Tsh"]], 0]]], {n, 0, 6}];
  Print["  [3] two-shell stepped sphere, k_S a = 1, n = 0..6: propagator with a break vs closed form ", sci[worst],
   " -> ", chk[worst < 10^-20]]];

(* [4], [5] the graded sphere: unitarity per order, and precision *)
nMax = 14;
graded = Association[Table[kSa -> Table[TSSphereTMatrix[n, omAt[kSa], bgB, matG, {bB, aB}, WorkingPrecision -> 90],
      {n, 0, nMax}], {kSa, {1/2, 1}}]];
Module[{worstU = 0, worstP = 0},
  Do[Module[{rows = graded[kSa], hi},
     Do[Module[{row = rows[[n + 1]], W, S},
        If[n >= 1,
         W = DiagonalMatrix[{Sqrt[5000], Sqrt[3000 n (n + 1)]}];
         S = W . (IdentityMatrix[2] + 2 row["Tpsv"]) . Inverse[W];
         worstU = Max[worstU, Norm[ConjugateTranspose[S] . S - IdentityMatrix[2]], Abs[Abs[1 + 2 row["Tsh"]] - 1]],
         worstU = Max[worstU, Abs[Abs[1 + 2 row["Tpsv"][[1, 1]]] - 1]]]], {n, 0, nMax}];
     Do[hi = TSSphereTMatrix[n, omAt[kSa], bgB, matG, {bB, aB}, WorkingPrecision -> 120];
      worstP = Max[worstP, Max[Abs[hi["Tpsv"] - rows[[n + 1]]["Tpsv"]]]/Max[Abs[hi["Tpsv"]]]], {n, {0, 1, 2, 5}}]],
   {kSa, {1/2, 1}}];
  Print["  [4] graded sphere, k_S a = 0.5 and 1, n = 0..", nMax, ": worst unitarity defect ", sci[worstU], " -> ",
   chk[worstU < 10^-35]];
  Print["  [5] 90 vs 120 digits: ", sci[worstP], " -> ", chk[worstP < 10^-35]]];

(* [6] N homogeneous shells, each at the contrast of its mid-radius, converge to the graded sphere at 1/N^2 *)
Module[{nn = 2, errs, ns = {4, 8, 16}, want, ord},
  want = graded[1][[nn + 1]]["Tpsv"];
  errs = Table[Module[{rs, mids, mats},
     rs = Join[{bB}, Table[bB + (aB - bB) i/nsh, {i, nsh}]];
     mids = Table[bB + (aB - bB) (i - 1/2)/nsh, {i, nsh}];
     mats = Join[{bgB + dB}, Table[bgB + profG[m] dB, {m, mids}]];
     Max[Abs[TSShellsTMatrix[nn, omAt[1], bgB, rs, mats, WorkingPrecision -> 60]["Tpsv"] - want]]/Max[Abs[want]]],
    {nsh, ns}];
  ord = N[Log[2, Most[errs]/Rest[errs]]];
  Print["  [6] N-shell closed form vs the graded sphere (k_S a = 1, n = 2), N = ", ns, ": ", sci /@ errs,
   "  order ", ord, " -> ", chk[AllTrue[ord, 1.8 < # < 2.2 &]]]];

(* export: per-order T-matrices of the graded sphere, in the format of MieTmatrixReference.json *)
reim[z_] := {N[Re[z], 20], N[Im[z], 20]};
Export[FileNameJoin[{DirectoryName[$InputFileName], "ContinuumLimit_graded_sphere_tmatrix.json"}],
  <|"conventions" -> "T maps incident potential coefficients (P,S) to scattered (P,S); SH scalar likewise",
    "profile" -> "core r < b carries the contrast; for b < r < a it is multiplied by s((a-r)/(a-b)), s = 10x^3-15x^4+6x^5",
    "radius" -> aB, "core" -> bB, "alpha" -> 5000, "beta" -> 3000, "rho" -> 2500,
    "contrast" -> <|"Dlambda" -> 2 10^9, "Dmu" -> 1 10^9, "Drho" -> 100|>,
    "sets" -> Association[Table[ToString[N[kSa]] -> <|"kSa" -> N[kSa], "omega" -> N[omAt[kSa]],
        "tmatrices" -> MapIndexed[<|"n" -> First[#2] - 1, "Tpsv" -> Map[reim, #1["Tpsv"], {2}], "Tsh" -> reim[#1["Tsh"]]|> &,
          graded[kSa]]|>, {kSa, {1/2, 1}}]]|>, "JSON"];

Print["==== ContinuumLimit_GradedSphere (stage 16): ",
  If[And @@ oks, "ALL " <> ToString[Length[oks]] <> " CHECKS PASS", "CHECKS FAILED"], " ===="];
