#!/usr/bin/env wolframscript
(* ============================================================================
   IntraPlaneRT.wl  --  Phase 3a: planar layer R/T(p) PROJECTION.

   Projects the Phase-2 spherical collective T_coll(k_par) onto Kennett
   flux-normalised up/down P-SV-SH plane waves at horizontal slowness p,
   producing the layer scattering operator Rd, Ru, Td, Tu (2x2 P-SV) + SH
   scalar, across normal / sub-critical / post-critical p.  Reciprocity
   (Tu = Td^T, Rd symmetric) is the tight machine-precision gate; the absolute
   normalisation is pinned against the Cartesian slab / Kennett in Python.

   Energy balance |R|^2+|T|^2=1 is OUT of scope (needs the undamped vector G0;
   the Phase-2 lattice damping Im k=0.25 is reciprocity-preserving but lossy)
   -> Phase 3b.

   Pipeline (per slowness p, incident mode m):
     incident plane wave (complex slowness s_m=(eta_m,p,0), z-first) -> regular
     multipole amplitude vector a  [incVec, CartesianT0 incident bridge]
       -> b = T_coll . a            [collective scatter]
       -> specular plane-wave content at the up/down P-SV-SH directions
          [projPW, CartesianT0 scattered Weyl bridge] x lattice prefactor
          i/(2 eta_out omega Acell)  [rtAmp].

   Coordinates z(down,axis0)/x(axis1)/y(axis2); slowness vector (eta,p,0) so p
   is along x and the SH polarisation is the y-axis {0,0,1}.  Time e^{+i w t},
   outgoing h_n^(1) (SphericalHankelH1).  Inner T0 at REAL background
   wavenumbers; only the inter-voxel lattice sum is damped.
   ============================================================================ *)

Get["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/CartesianT0.wl"];

reim[z_] := {Re[N[z]], Im[N[z]]};
alpha0 = 5000.; beta0 = 3000.; rho0 = 2500.;
lam0 = rho0 (alpha0^2 - 2 beta0^2); mu0 = rho0 beta0^2;
dampIm = 0.25; aa = 1.0; aLpitch = 2.5; Acell = aLpitch^2;
kaTest = 0.3;                                   (* k_P aa; sub-wavelength lattice (single Bragg order) *)

(* ---- Mie param setter: set the globals T0LMN/T0vec read, for (ka, contrast) ---- *)
setMie[ka_, Dl_, Dm_, Dr_] := Module[{rhoI = rho0 + Dr, lamI0 = lam0 + Dl, muI0 = mu0 + Dm, alphaI, betaI},
  alphaI = Sqrt[(lamI0 + 2 muI0)/rhoI]; betaI = Sqrt[muI0/rhoI];
  kPo = ka; kSo = ka alpha0/beta0; kPi = ka alpha0/alphaI; kSi = ka alpha0/betaI;
  lamO = lam0; muO = mu0; lamI = lamI0; muI = muI0;];
omegaOf := kPo alpha0/aa;

(* ---- single-site T0 in the clean L/M/N basis (from CartesianT0; item c) ---- *)
chPos = <|"L" -> 1, "M" -> 2, "N" -> 3|>;
T0LMN[0] := {{T0mono[kPo, lamO, muO, kPi, lamI, muI, aa]}};
T0LMN[n_] := Module[{ts = TsphClean[n, kPo, kSo, lamO, muO, kPi, kSi, lamI, muI, aa],
    tt = Ttoroidal[n, kSo, muO, kSi, muI, aa]},
   {{ts[[1, 1]], 0, ts[[1, 2]]}, {0, tt, 0}, {ts[[2, 1]], 0, ts[[2, 2]]}}];
idxVof[Nmax_] := Flatten[Table[
    If[n == 0, {{0, 0, "L"}}, Flatten[Table[{n, m, ch}, {m, -n, n}, {ch, {"L", "M", "N"}}], 1]],
    {n, 0, Nmax}], 1];
T0vec[Nmax_] := Module[{idx = idxVof[Nmax], T0e},
  T0e[{n1_, m1_, c1_}, {n2_, m2_, c2_}] :=
    If[n1 == n2 && m1 == m2, If[n1 == 0, T0LMN[0][[1, 1]], T0LMN[n1][[chPos[c1], chPos[c2]]]], 0];
  Table[T0e[idx[[i]], idx[[j]]], {i, Length[idx]}, {j, Length[idx]}]];

(* ---- slowness geometry: vertical slowness (Im>=0), incident/output unit directions ---- *)
etaOf[c_, p_] := Module[{e = Sqrt[1./c^2 - p^2]}, If[Im[e] < 0, -e, e]];
khatMode[m_, p_, sign_] := Module[{c = If[m == "P", alpha0, beta0]}, c {sign etaOf[c, p], p, 0.}];
polMode[m_, khat_] := Switch[m, "P", khat, "SV", th[khat], "SH", {0., 0., 1.}];  (* y-axis SH pol *)

(* ---- incident multipole amplitude vector (P->L, SV->N, SH->M); CartesianT0 bridge ---- *)
incVec[mode_, khat_, ehat_, Nmax_] := Map[
   Function[idx, Module[{n = idx[[1]], m = idx[[2]], ch = idx[[3]]},
     Switch[{mode, ch}, {"P", "L"}, incP[n, m, khat], {"SV", "N"}, incN[n, m, khat, ehat],
       {"SH", "M"}, incM[n, m, khat, ehat], _, 0]]], idxVof[Nmax]];

(* ---- plane-wave content of an outgoing multipole vector at direction ks (generalised farField) ---- *)
projPW[bvec_, ks_, Nmax_] := Module[{fP = {0, 0, 0}, fS = {0, 0, 0}, idx = idxVof[Nmax]},
  Do[With[{n = idx[[i, 1]], m = idx[[i, 2]], ch = idx[[i, 3]], b = bvec[[i]]},
    Switch[ch,
      "L", fP += b ((-I)^n/kPo) Yv[n, m, ks] ks,
      "N", fS += b ((-I)^n/kSo) Bv[n, m, ks],
      "M", fS += -b ((-I)^(n + 1)/kSo) Cv[n, m, ks]]], {i, Length[idx]}];
  {fP, fS}];

(* ---- one specular R/T amplitude: in-mode/sign -> out-mode/sign ---- *)
Cnorm = 1.;                                     (* overall normalisation; pinned in Python (Task 4) *)
rtAmp[Tcoll_, inMode_, p_, inSign_, outMode_, outSign_, Nmax_] := Module[
   {kin = khatMode[inMode, p, inSign], kout = khatMode[outMode, p, outSign], ein, avec, bvec, f,
    etaOut = etaOf[If[outMode == "P", alpha0, beta0], p]},
  ein = polMode[inMode, kin];
  avec = incVec[inMode, kin, ein, Nmax];
  bvec = Tcoll . avec;
  f = projPW[bvec, kout, Nmax];
  Cnorm (I/(2 etaOut omegaOf Acell)) (polMode[outMode, kout] . If[outMode == "P", First[f], Last[f]])];

(* ============================================================================
   [Task 1] p=0 single-sphere tracer: assemble Rd, check P-SV decoupling
   ============================================================================ *)
Print["==== Phase 3a :: layer R/T(p) projection ===="];
Nm = 4;
setMie[kaTest, 2.*^9, 1.*^9, 100.];             (* moderate contrast *)
T0only = T0vec[Nm];

(* internal consistency: rtAmp(T0only) must reproduce the single-sphere farField projection.
   (no separate assert here; the p=0 decoupling below exercises the same path.) *)
Rd0 = Table[rtAmp[T0only, im, 0., 1., om, -1., Nm], {om, {"P", "SV"}}, {im, {"P", "SV"}}];
offDiag = Max[Abs[{Rd0[[1, 2]], Rd0[[2, 1]]}]];
sym = Abs[Rd0[[1, 2]] - Rd0[[2, 1]]];
decoupOK = offDiag < 1.*^-8;
Print["  [p=0, G0=0] Rd diag = ", Map[ScientificForm[#, 3] &, {Rd0[[1, 1]], Rd0[[2, 2]]}]];
Print["  [p=0, G0=0] P-SV off-diagonal (decoupling) = ", ScientificForm[offDiag, 3],
   " -> ", If[decoupOK, "PASS", "FAIL"]];

(* ============================================================================
   dump (Task 1 partial: p=0 single-sphere record)
   ============================================================================ *)
Export["/Users/tod/Desktop/MultipleScatteringCalculations/Mathematica/IntraPlaneRT_reference.json",
  <|"params" -> <|"alpha" -> alpha0, "beta" -> beta0, "rho0" -> rho0, "aa" -> aa,
      "aLpitch" -> aLpitch, "kPo" -> kPo, "dampIm" -> dampIm, "Nmax" -> Nm,
      "contrast" -> <|"Dlambda" -> 2.*^9, "Dmu" -> 1.*^9, "Drho" -> 100.|>|>,
    "stageRT" -> {<|"p" -> 0., "regime" -> "normal", "G0" -> "off",
       "Rd" -> Map[reim, Rd0, {2}]|>}|>];
Print["  wrote IntraPlaneRT_reference.json (Task 1 partial)"];
Print["Phase 3a (IntraPlaneRT.wl) Task 1 loaded."];
