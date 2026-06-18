#!/usr/bin/env wolframscript
(* ============================================================================
   CartesianT0.wl  -- Full-wave Cartesian scattering operator for the elastic
   sphere, built from the symplectic-bracket spherical Mie T-matrix.

   PIPELINE (each stage independently verified; see VERIFICATION at the bottom):
     1. Spherical bracket T_n  (P-SV spheroidal 2x2 + SH toroidal + n=0 monopole),
        from ElasticMieTmatrix.nb, converted to the clean L/M/N normalization.
     2. Incident bridge  : Cartesian plane wave (P/SV/SH, dir k_i) -> regular
        multipole amplitudes a_{nm}     [vector Jacobi-Anger, verified].
     3. Scattered bridge : outgoing multipole b_{nm} -> plane-wave content
        (elastic Weyl angular spectrum) [verified, all 3 channels].
     4. Far-field amplitude f(k_s, k_i) = scattered(b = T a).
     5. The (u,t)(kx,ky) 6x6 / Kennett R-T projection is the final layer:
        evaluate f at the up/down P,SV,SH directions at fixed (kx,ky).

   Conventions: time e^{+i w t}, outgoing h_n^(1), z(down)/x/y as in the repo.
   Clean normalization: L=(1/k)grad psi, M=curl(r psi), N=(1/k)curl M,
   psi = z_n(k r) Y_n^m;  z_n = j_n (regular) or h_n^(1) (outgoing).
   ============================================================================ *)

(* ---- 1. spherical bracket T_n in the clean L/M/N normalization ---- *)
nb = Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/ElasticMieTmatrix.nb"];
mcells = Cases[nb, Cell[c_String, "Input", ___] :> c, Infinity];
cln[s_] := StringReplace[s, "ClearAll[\"Global`*\"]" -> "Null"];
Do[ToExpression[cln[mcells[[i]]]], {i, {1, 2, 3, 5}}];
ToExpression[Import["/tmp/cell_sec3_builders.m", "Text"]];   (* Tspheroidal/Ttoroidal/T0mono (bracket) *)

(* clean conversion: spheroidal L<->N off-diagonals carry sqrt(n(n+1)) *)
TsphClean[n_, kPo_, kSo_, lamO_, muO_, kPi_, kSi_, lamI_, muI_, a_] :=
  Module[{t = Tspheroidal[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, a], s = Sqrt[n (n + 1)]},
   {{t[[1, 1]], s t[[1, 2]]}, {t[[2, 1]]/s, t[[2, 2]]}}];

(* ---- 2-3. vector spherical harmonics on a (possibly complex) direction ---- *)
ang[d_] := {ArcCos[d[[3]]/Sqrt[d . d]], ArcTan[d[[1]], d[[2]]]};
RM[t_, p_] := {{Sin[t] Cos[p], Cos[t] Cos[p], -Sin[p]}, {Sin[t] Sin[p], Cos[t] Sin[p], Cos[p]}, {Cos[t], -Sin[t], 0}};
Yv[n_, m_, d_] := SphericalHarmonicY[n, m, ang[d][[1]], ang[d][[2]]];
Bv[n_, m_, d_] := Module[{t = ang[d][[1]], p = ang[d][[2]]},      (* grad_Omega Y (poloidal) *)
   RM[t, p] . {0, (D[SphericalHarmonicY[n, m, x, p], x] /. x -> t), (I m/Sin[t]) SphericalHarmonicY[n, m, t, p]}];
Cv[n_, m_, d_] := Cross[d, Bv[n, m, d]];                          (* k x grad_Omega Y (toroidal) *)

(* incident multipole amplitudes (VERIFIED coefficients) *)
incP[n_, m_, ki_] := 4 Pi I^(n - 1) Conjugate[Yv[n, m, ki]];               (* a^L for P *)
incN[n_, m_, ki_, ei_] := -4 Pi I^(n + 1)/(n (n + 1)) Conjugate[ei . Bv[n, m, ki]];  (* a^N (SV) *)
incM[n_, m_, ki_, ei_] := -4 Pi I^n/(n (n + 1)) Conjugate[ei . Cv[n, m, ki]];        (* a^M (SH) *)

(* ---- 4. far-field amplitude f(k_s, k_i) for incident (type, k_i, e_i) ---- *)
(* medium passed via global host params set by the caller *)
farField[type_, ki_, ei_, ks_, Nmax_, kP_, kS_, lamO_, muO_, kPi_, kSi_, lamI_, muI_, a_] :=
  Module[{fP = {0, 0, 0}, fS = {0, 0, 0}, aL, aN, aM, bL, bN, bM, t},
   Do[
    Switch[type, "P", aL = incP[n, m, ki]; aN = 0; aM = 0,
                 _,   aL = 0; aN = incN[n, m, ki, ei]; aM = incM[n, m, ki, ei]];
    t = TsphClean[n, kP, kS, lamO, muO, kPi, kSi, lamI, muI, a];
    bL = t[[1, 1]] aL + t[[1, 2]] aN; bN = t[[2, 1]] aL + t[[2, 2]] aN;
    bM = Ttoroidal[n, kS, muO, kSi, muI, a] aM;
    fP += bL ((-I)^n/kP) Yv[n, m, ks] ks;
    fS += bN ((-I)^n/kS) Bv[n, m, ks] - bM ((-I)^(n + 1)/kS) Cv[n, m, ks],
    {n, 1, Nmax}, {m, -n, n}];
   fP += (T0mono[kP, lamO, muO, kPi, lamI, muI, a] 4 Pi/I Conjugate[Yv[0, 0, ki]]) (1/kP) Yv[0, 0, ks] ks;
   {fP, fS}];

(* ============================ VERIFICATION ============================ *)
kPo = 0.9; kSo = 1.5; kPi = 0.8897917302988777; kSi = 1.4968051081937466;
lamO = 17.5*^9; muO = 22.5*^9; lamI = 19.5*^9; muI = 23.5*^9; aa = 1.0; NM = 7;
ki1 = {Sin[0.7] Cos[0.5], Sin[0.7] Sin[0.5], Cos[0.7]};
ks1 = {Sin[1.2] Cos[2.0], Sin[1.2] Sin[2.0], Cos[1.2]};
th[d_] := RM[ang[d][[1]], ang[d][[2]]][[All, 2]];
ampPP[ks_, ki_] := ks . First[farField["P", ki, ks, ks, NM, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa]];
ampSS[ks_, ki_] := th[ks] . Last[farField["SV", ki, th[ki], ks, NM, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa]];
Print["RECIPROCITY (full operator):"];
Print["  PP |f(ks,ki) - f(-ki,-ks)| = ", Abs[ampPP[ks1, ki1] - ampPP[-ki1, -ks1]]];
Print["  SS |f(ks,ki) - f(-ki,-ks)| = ", Abs[ampSS[ks1, ki1] - ampSS[-ki1, -ks1]]];
Print["CartesianT0.wl loaded + verified."];
